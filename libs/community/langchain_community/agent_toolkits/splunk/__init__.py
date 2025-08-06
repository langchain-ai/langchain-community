"""Splunk agent toolkit."""

from langchain_community.agent_toolkits.splunk.base import (
    create_splunk_agent,
    create_splunk_agent_from_api_wrapper,
)
from langchain_community.agent_toolkits.splunk.toolkit import (
    SplunkToolkit,
)

# Re-export tools for convenience
from langchain_community.tools.splunk import (
    InfoSplunkTool,
    ListSplunkHostsTool,
    ListSplunkIndexesTool,
    ListSplunkSourcetypesTool,
    QueryCheckerSplunkTool,
    QuerySplunkTool,
)

__all__ = [
    "SplunkToolkit",
    "InfoSplunkTool",
    "ListSplunkIndexesTool",
    "ListSplunkSourcetypesTool",
    "ListSplunkHostsTool",
    "QuerySplunkTool",
    "QueryCheckerSplunkTool",
    "create_splunk_agent",
    "create_splunk_agent_from_api_wrapper",
]
