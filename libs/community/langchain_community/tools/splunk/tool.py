"""Standalone Splunk tools that can be used independently."""

import json
import logging
from typing import Any, Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.utilities.splunk import SplunkAPIWrapper

logger = logging.getLogger(__name__)


class _InfoSplunkToolInput(BaseModel):
    """Input for InfoSplunkTool."""

    query: str = Field(
        default="",
        description="Always use an empty string as input.",
    )


class _ListSplunkIndexesToolInput(BaseModel):
    """Input for ListSplunkIndexesTool."""

    query: str = Field(
        default="",
        description="Always use an empty string as input.",
    )


class _QuerySplunkToolInput(BaseModel):
    """Input for QuerySplunkTool."""

    query: str = Field(..., description="A detailed and correct SPL query.")


class _QueryCheckerSplunkToolInput(BaseModel):
    """Input for QueryCheckerSplunkTool."""

    query: str = Field(..., description="A SPL query to validate.")


class InfoSplunkTool(BaseTool):
    """Tool for getting information about Splunk environment.

    This tool can be used independently to retrieve information about
    available Splunk indexes, sourcetypes, hosts, and connection status.
    """

    name: str = "splunk_info"
    description: str = (
        "Input to this tool is always an empty string. "
        "Output is information about the Splunk environment including "
        "available indexes, sourcetypes, hosts, and connection status. "
        "Use this tool to understand what data is available before writing SPL queries."
    )
    args_schema: Type[BaseModel] = _InfoSplunkToolInput

    splunk_wrapper: SplunkAPIWrapper

    def _run(
        self,
        query: str = "",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Get Splunk environment information."""
        try:
            info = self.splunk_wrapper.get_summary_info()
            return json.dumps(info, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error getting Splunk info: {e}")
            return f"Error getting Splunk info: {str(e)}"


class ListSplunkIndexesTool(BaseTool):
    """Tool for listing available Splunk indexes.

    This standalone tool provides a focused way to discover
    available data indexes in a Splunk environment.
    """

    name: str = "splunk_list_indexes"
    description: str = (
        "Input to this tool is always an empty string. "
        "Output is a list of available Splunk indexes. "
        "Use this to see what data indexes are available for searching."
    )
    args_schema: Type[BaseModel] = _ListSplunkIndexesToolInput

    splunk_wrapper: SplunkAPIWrapper

    def _run(
        self,
        query: str = "",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """List available indexes."""
        try:
            indexes = self.splunk_wrapper.get_indexes()
            return json.dumps({"indexes": indexes, "count": len(indexes)}, indent=2)
        except Exception as e:
            logger.error(f"Error listing indexes: {e}")
            return f"Error listing indexes: {str(e)}"


class QuerySplunkTool(BaseTool):
    """Tool for executing SPL queries against Splunk.

    This standalone tool allows direct execution of SPL queries
    and can be integrated into custom agents or workflows.
    """

    name: str = "splunk_query"
    description: str = (
        "Execute SPL (Search Processing Language) queries against Splunk. "
        "Input should be a valid SPL query string. "
        "Output will be the query results in JSON format. "
        "If the query is not correct, an error message will be returned. "
        "If an error is returned, rewrite the query, check the query, and try again. "
        "Always use proper SPL syntax - start with 'search' command for basic searches, "
        "or use pipe commands like '| stats', '| eval', '| where', etc. "
        "Include time ranges when possible for better performance."
    )
    args_schema: Type[BaseModel] = _QuerySplunkToolInput

    splunk_wrapper: SplunkAPIWrapper

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute SPL query."""
        try:
            # Validate query input
            if not query or not query.strip():
                return "Error: Empty query provided"

            query = query.strip()

            # Auto-add search command if needed (for basic queries)
            if not (
                query.lower().startswith(("search ", "|"))
                or any(
                    query.lower().startswith(cmd)
                    for cmd in [
                        "from ",
                        "inputlookup ",
                        "rest ",
                        "dbxquery ",
                        "savedsearch ",
                    ]
                )
            ):
                query = f"search {query}"

            # Execute the query
            results = self.splunk_wrapper.run_spl_query(query)

            if not results:
                return "Query executed successfully but returned no results."

            # Return formatted results
            result_data = {
                "query": query,
                "results_count": len(results),
                "results": results,
            }

            return json.dumps(result_data, indent=2, default=str)

        except Exception as e:
            logger.error(f"Error executing SPL query: {e}")
            return f"Error executing SPL query: {str(e)}"


class QueryCheckerSplunkTool(BaseTool):
    """Tool for checking and validating SPL queries.

    This standalone tool validates SPL query syntax and provides
    optimization suggestions without executing the query.
    """

    name: str = "splunk_query_checker"
    description: str = (
        "Use this tool to double check if your SPL query is correct before executing it. "
        "Always use this tool before executing a query with splunk_query! "
        "Input should be a SPL query string. "
        "Output will be validation results and suggestions for improvement."
    )
    args_schema: Type[BaseModel] = _QueryCheckerSplunkToolInput

    splunk_wrapper: SplunkAPIWrapper
    llm: Optional[BaseLanguageModel] = None

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Check SPL query validity."""
        try:
            if not query or not query.strip():
                return "Error: Empty query provided"

            query = query.strip()

            # Basic syntax validation using Splunk's parser
            validation_result = self.splunk_wrapper.validate_spl_query(query)

            # Basic checks and suggestions
            suggestions = []
            issues = []

            # Check for common patterns
            if not (query.lower().startswith(("search ", "|"))):
                suggestions.append(
                    "Consider starting with 'search' command or pipe command"
                )

            # Check for potentially expensive operations
            expensive_commands = [
                "join",
                "append",
                "appendcols",
                "transaction",
                "subsearch",
            ]
            for cmd in expensive_commands:
                if f" {cmd} " in query.lower() or query.lower().startswith(f"{cmd} "):
                    suggestions.append(
                        f"'{cmd}' can be expensive. Consider alternatives if performance is important."
                    )

            # Check for time range
            if not any(
                time_cmd in query.lower()
                for time_cmd in ["earliest=", "latest=", "timeformat"]
            ):
                suggestions.append(
                    "Consider adding time range constraints for better performance"
                )

            # Check for wildcards without index
            if query.startswith("search *") and "index=" not in query.lower():
                issues.append(
                    "Searching '*' without specifying an index can be very slow"
                )

            result = {
                "query": query,
                "validation": validation_result,
                "issues": issues,
                "suggestions": suggestions,
            }

            # Add LLM analysis if available
            if self.llm and validation_result.get("valid"):
                try:
                    llm_prompt = f"""
                    Please review this Splunk SPL query for best practices and optimization:
                    
                    Query: {query}
                    
                    Provide brief feedback on:
                    1. Query efficiency 
                    2. Best practice recommendations
                    3. Potential improvements
                    
                    Keep response concise and actionable.
                    """

                    llm_response = self.llm.invoke(llm_prompt)
                    result["llm_analysis"] = (
                        llm_response.content
                        if hasattr(llm_response, "content")
                        else str(llm_response)
                    )
                except Exception as e:
                    result["llm_analysis"] = f"LLM analysis failed: {str(e)}"
                    logger.warning(f"LLM analysis failed: {e}")

            return json.dumps(result, indent=2, default=str)

        except Exception as e:
            logger.error(f"Error checking SPL query: {e}")
            return f"Error checking SPL query: {str(e)}"


class ListSplunkSourcetypesTool(BaseTool):
    """Tool for listing available Splunk sourcetypes.

    This standalone tool provides detailed sourcetype discovery
    for specific indexes or across the entire Splunk environment.
    """

    name: str = "splunk_list_sourcetypes"
    description: str = (
        "List available sourcetypes in Splunk. "
        "Input can be an index name to list sourcetypes for a specific index, "
        "or empty string to list sourcetypes across all indexes. "
        "Output is a list of available sourcetypes."
    )

    splunk_wrapper: SplunkAPIWrapper

    def _run(
        self,
        query: str = "",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """List available sourcetypes."""
        try:
            index = query.strip() if query.strip() else None
            sourcetypes = self.splunk_wrapper.get_sourcetypes(index=index)

            result = {
                "index": index or "all",
                "sourcetypes": sourcetypes,
                "count": len(sourcetypes),
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error(f"Error listing sourcetypes: {e}")
            return f"Error listing sourcetypes: {str(e)}"


class ListSplunkHostsTool(BaseTool):
    """Tool for listing available Splunk hosts.

    This standalone tool discovers host information
    for specific indexes or across the entire environment.
    """

    name: str = "splunk_list_hosts"
    description: str = (
        "List available hosts in Splunk data. "
        "Input can be an index name to list hosts for a specific index, "
        "or empty string to list hosts across all indexes. "
        "Output is a list of available hosts."
    )

    splunk_wrapper: SplunkAPIWrapper

    def _run(
        self,
        query: str = "",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """List available hosts."""
        try:
            index = query.strip() if query.strip() else None
            hosts = self.splunk_wrapper.get_hosts(index=index)

            result = {"index": index or "all", "hosts": hosts, "count": len(hosts)}

            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error(f"Error listing hosts: {e}")
            return f"Error listing hosts: {str(e)}"
