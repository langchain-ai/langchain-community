"""Utility for interacting with Splunk."""

import json
import logging
import time
from typing import Any, Dict, List, Optional, Union

import requests
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)


class SplunkAPIWrapper(BaseModel):
    """Wrapper around Splunk REST API for SPL query execution."""

    splunk_host: str = Field(..., description="Splunk server host")
    splunk_port: int = Field(default=8089, description="Splunk management port")
    splunk_token: str = Field(..., description="Splunk authentication token")
    splunk_scheme: str = Field(
        default="https", description="Connection scheme (http/https)"
    )
    verify_ssl: bool = Field(default=False, description="SSL certificate verification")
    timeout: int = Field(default=30, description="Request timeout in seconds")
    max_results: int = Field(
        default=100, description="Maximum number of results to return"
    )

    # Optional authentication fallback
    splunk_username: Optional[str] = Field(
        default=None, description="Splunk username (fallback)"
    )
    splunk_password: Optional[str] = Field(
        default=None, description="Splunk password (fallback)"
    )

    model_config = {
        "extra": "forbid",
        "arbitrary_types_allowed": True,
        "validate_assignment": True,
    }

    def __init__(self, **data: Any):
        """Initialize the SplunkAPIWrapper."""
        super().__init__(**data)
        self._session = self._create_session()

    @model_validator(mode="after")
    def validate_auth(self):
        """Validate that either token or username/password is provided."""
        token = self.splunk_token
        username = self.splunk_username
        password = self.splunk_password

        if not token and not (username and password):
            raise ValueError(
                "Either splunk_token or both splunk_username and splunk_password must be provided"
            )
        return self

    def _create_session(self) -> requests.Session:
        """Create and configure requests session with token authentication."""
        session = requests.Session()

        if self.splunk_token:
            # Use token authentication (preferred)
            session.headers.update(
                {
                    "Authorization": f"Bearer {self.splunk_token}",
                    "Content-Type": "application/x-www-form-urlencoded",
                }
            )
        elif self.splunk_username and self.splunk_password:
            # Fallback to basic authentication
            session.auth = (self.splunk_username, self.splunk_password)

        session.verify = self.verify_ssl
        session.timeout = self.timeout
        return session

    @property
    def base_url(self) -> str:
        """Get base URL for Splunk REST API."""
        return f"{self.splunk_scheme}://{self.splunk_host}:{self.splunk_port}"

    def test_connection(self) -> bool:
        """Test connection to Splunk server."""
        try:
            url = f"{self.base_url}/services/server/info"
            response = self._session.get(url)
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    def run_spl_query(
        self,
        query: str,
        max_results: Optional[int] = None,
        earliest_time: str = "-1h",
        latest_time: str = "now",
    ) -> List[Dict[str, Any]]:
        """Execute SPL query and return results."""
        max_results = max_results or self.max_results

        try:
            # Create search job
            search_url = f"{self.base_url}/services/search/jobs"
            search_data = {
                "search": query,
                "earliest_time": earliest_time,
                "latest_time": latest_time,
                "output_mode": "json",
                "max_count": max_results,
                "exec_mode": "blocking",  # Wait for completion
                "timeout": self.timeout,
            }

            logger.info(f"Executing SPL query: {query[:100]}...")
            response = self._session.post(search_url, data=search_data)
            response.raise_for_status()

            # Extract job ID
            job_data = response.json()
            job_id = job_data.get("sid")

            if not job_id:
                raise Exception("Failed to create search job")

            # Get results
            results_url = f"{self.base_url}/services/search/jobs/{job_id}/results"
            results_data = {"output_mode": "json", "count": max_results}

            # Poll for job completion with timeout
            status_url = f"{self.base_url}/services/search/jobs/{job_id}"
            start_time = time.time()

            while True:
                status_response = self._session.get(
                    status_url, params={"output_mode": "json"}
                )
                status_response.raise_for_status()
                status = status_response.json()

                dispatch_state = status["entry"][0]["content"]["dispatchState"]

                if dispatch_state == "DONE":
                    break
                elif dispatch_state == "FAILED":
                    raise Exception("Search job failed")
                elif time.time() - start_time > self.timeout:
                    # Cancel the job if it times out
                    cancel_url = (
                        f"{self.base_url}/services/search/jobs/{job_id}/control"
                    )
                    self._session.post(cancel_url, data={"action": "cancel"})
                    raise Exception("Search job timed out")

                time.sleep(1)  # Wait before next status check

            # Get final results
            results_response = self._session.get(results_url, params=results_data)
            results_response.raise_for_status()
            results = results_response.json()

            return results.get("results", [])

        except Exception as e:
            logger.error(f"SPL query execution failed: {e}")
            raise

    def get_indexes(self) -> List[str]:
        """Get list of available indexes."""
        try:
            url = f"{self.base_url}/services/data/indexes"
            params = {"output_mode": "json", "count": 0}  # Get all indexes
            response = self._session.get(url, params=params)
            response.raise_for_status()

            data = response.json()
            indexes = [entry["name"] for entry in data.get("entry", [])]
            return sorted(indexes)
        except Exception as e:
            logger.error(f"Failed to get indexes: {e}")
            return []

    def get_sourcetypes(
        self, index: Optional[str] = None, limit: int = 1000
    ) -> List[str]:
        """Get list of available sourcetypes."""
        try:
            if index:
                query = f"| metadata type=sourcetypes index={index} | head {limit}"
            else:
                query = f"| metadata type=sourcetypes | head {limit}"

            results = self.run_spl_query(query, max_results=limit)
            sourcetypes = [result.get("sourcetype", "") for result in results]
            return sorted([st for st in sourcetypes if st])
        except Exception as e:
            logger.error(f"Failed to get sourcetypes: {e}")
            return []

    def get_hosts(self, index: Optional[str] = None, limit: int = 1000) -> List[str]:
        """Get list of available hosts."""
        try:
            if index:
                query = f"| metadata type=hosts index={index} | head {limit}"
            else:
                query = f"| metadata type=hosts | head {limit}"

            results = self.run_spl_query(query, max_results=limit)
            hosts = [result.get("host", "") for result in results]
            return sorted([h for h in hosts if h])
        except Exception as e:
            logger.error(f"Failed to get hosts: {e}")
            return []

    def get_sources(self, index: Optional[str] = None, limit: int = 1000) -> List[str]:
        """Get list of available sources."""
        try:
            if index:
                query = f"| metadata type=sources index={index} | head {limit}"
            else:
                query = f"| metadata type=sources | head {limit}"

            results = self.run_spl_query(query, max_results=limit)
            sources = [result.get("source", "") for result in results]
            return sorted([s for s in sources if s])
        except Exception as e:
            logger.error(f"Failed to get sources: {e}")
            return []

    def validate_spl_query(self, query: str) -> Dict[str, Any]:
        """Validate SPL query syntax without executing it."""
        try:
            url = f"{self.base_url}/services/search/parser"
            data = {"q": query, "output_mode": "json", "parse_only": True}

            response = self._session.post(url, data=data)
            response.raise_for_status()
            result = response.json()

            return {"valid": True, "query": query, "parsed": result}

        except Exception as e:
            return {"valid": False, "query": query, "error": str(e)}

    def get_summary_info(self) -> Dict[str, Any]:
        """Get summary information about the Splunk environment."""
        try:
            info = {
                "indexes": self.get_indexes(),
                "total_indexes": 0,
                "sample_sourcetypes": [],
                "sample_hosts": [],
                "connection_status": (
                    "connected" if self.test_connection() else "failed"
                ),
            }

            info["total_indexes"] = len(info["indexes"])

            # Get sample sourcetypes from main index if available
            if "main" in info["indexes"]:
                info["sample_sourcetypes"] = self.get_sourcetypes("main")[:10]
                info["sample_hosts"] = self.get_hosts("main")[:10]
            elif info["indexes"]:
                # Use first available index
                first_index = info["indexes"][0]
                info["sample_sourcetypes"] = self.get_sourcetypes(first_index)[:10]
                info["sample_hosts"] = self.get_hosts(first_index)[:10]

            return info

        except Exception as e:
            logger.error(f"Failed to get summary info: {e}")
            return {"error": str(e), "connection_status": "failed"}
