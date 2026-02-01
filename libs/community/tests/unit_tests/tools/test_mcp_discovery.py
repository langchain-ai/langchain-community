from langchain_community.tools.mcp_discovery import MCPDiscoveryTool


def test_mcp_discovery_tool_name():
    tool = MCPDiscoveryTool()
    assert tool.name == "mcp_discovery"
