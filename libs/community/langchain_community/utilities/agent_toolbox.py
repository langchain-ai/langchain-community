"""Utility for Agent Toolbox API.

Get a free API key at https://api.sendtoclaw.com/v1/auth/register
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import requests
from pydantic import BaseModel, Field, SecretStr


class AgentToolboxAPIWrapper(BaseModel):
    """Wrapper around Agent Toolbox API.

    Provide api_key directly or set AGENT_TOOLBOX_API_KEY env var.

    Example:
        .. code-block:: python

            from langchain_community.utilities import AgentToolboxAPIWrapper

            wrapper = AgentToolboxAPIWrapper(api_key="atb_xxx")
            result = wrapper.run("search", {"query": "AI agents"})
    """

    api_key: SecretStr = Field(
        default_factory=lambda: SecretStr(
            os.environ.get("AGENT_TOOLBOX_API_KEY", "")
        )
    )
    base_url: str = "https://api.sendtoclaw.com"
    timeout: float = 30.0

    def _post(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make authenticated POST request to Agent Toolbox API."""
        key = self.api_key.get_secret_value()
        if not key:
            raise ValueError(
                "Agent Toolbox API key required. "
                "Set AGENT_TOOLBOX_API_KEY env var or pass api_key=."
            )
        resp = requests.post(
            f"{self.base_url.rstrip('/')}/v1/{endpoint}",
            json=payload,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def run(self, endpoint: str, payload: Dict[str, Any]) -> str:
        """Run an Agent Toolbox API call and return formatted results."""
        result = self._post(endpoint, payload)
        data = result.get("data", result)
        if isinstance(data, str):
            return data
        return json.dumps(data, indent=2, ensure_ascii=False)

    def results(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Run an API call and return raw dict."""
        result = self._post(endpoint, payload)
        return result.get("data", result)
