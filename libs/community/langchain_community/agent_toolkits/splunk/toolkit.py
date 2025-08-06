"""Splunk toolkit for LangChain agents."""

from typing import List, Optional

from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.agent_toolkits.base import BaseToolkit
from langchain_community.tools.splunk.tool import (
    InfoSplunkTool,
    ListSplunkIndexesTool,
    QueryCheckerSplunkTool,
    QuerySplunkTool,
)
from langchain_community.utilities.splunk import SplunkAPIWrapper


class SplunkToolkit(BaseToolkit):
    """Splunk toolkit for LangChain agents."""

    splunk_wrapper: SplunkAPIWrapper
    llm: Optional[BaseLanguageModel] = None

    model_config = {"arbitrary_types_allowed": True}

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        list_splunk_indexes_tool = ListSplunkIndexesTool(
            splunk_wrapper=self.splunk_wrapper
        )

        info_splunk_tool_description = (
            "Input to this tool is always an empty string. "
            "Output is information about the Splunk environment including "
            "available indexes, sourcetypes, hosts, and connection status. "
            f"Be sure to understand the available data by calling {list_splunk_indexes_tool.name} first!"
        )
        info_splunk_tool = InfoSplunkTool(
            splunk_wrapper=self.splunk_wrapper, description=info_splunk_tool_description
        )

        query_splunk_tool_description = (
            "Execute SPL (Search Processing Language) queries against Splunk. "
            "Input should be a valid SPL query string. "
            "Output will be the query results in JSON format. "
            "If the query is not correct, an error message will be returned. "
            "If an error is returned, rewrite the query, check the query, and try again. "
            f"Always check available indexes with {list_splunk_indexes_tool.name} first! "
            "Always use proper SPL syntax - start with 'search' command for basic searches."
        )
        query_splunk_tool = QuerySplunkTool(
            splunk_wrapper=self.splunk_wrapper,
            description=query_splunk_tool_description,
        )

        tools = [
            list_splunk_indexes_tool,
            info_splunk_tool,
            query_splunk_tool,
        ]

        # Add query checker if LLM is available
        if self.llm:
            query_checker_tool_description = (
                "Use this tool to double check if your SPL query is correct before executing it. "
                f"Always use this tool before executing a query with {query_splunk_tool.name}! "
                "Input should be a SPL query string. "
                "Output will be validation results and suggestions for improvement."
            )
            query_checker_tool = QueryCheckerSplunkTool(
                splunk_wrapper=self.splunk_wrapper,
                llm=self.llm,
                description=query_checker_tool_description,
            )
            tools.append(query_checker_tool)

        return tools
