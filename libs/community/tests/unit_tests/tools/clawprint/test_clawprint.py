"""Unit tests for langchain_community.tools.clawprint.

All HTTP calls are mocked — no real API requests are made.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from langchain_community.tools.clawprint._client import (
    CLAWPRINT_DEFAULT_BASE_URL,
    ClawPrintAPIError,
    ClawPrintClient,
)
from langchain_community.tools.clawprint.tool import (
    ClawPrintCheckExchangeTool,
    ClawPrintDomainsTool,
    ClawPrintGetAgentTool,
    ClawPrintHireAgentTool,
    ClawPrintRegisterTool,
    ClawPrintSearchTool,
    ClawPrintToolkit,
    ClawPrintTrustTool,
)

# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def mock_client() -> ClawPrintClient:
    """Return a client with a mocked requests.Session."""
    client = ClawPrintClient(api_key="cp_test_key_123")
    client._session = MagicMock()
    return client


def _mock_response(data: Any, status_code: int = 200) -> MagicMock:
    """Build a mock requests.Response."""
    resp = MagicMock()
    resp.ok = 200 <= status_code < 300
    resp.status_code = status_code
    resp.json.return_value = data
    resp.text = json.dumps(data) if isinstance(data, dict) else str(data)
    return resp


# ======================================================================
# Client tests
# ======================================================================


class TestClawPrintClient:
    def test_default_base_url(self) -> None:
        client = ClawPrintClient()
        assert client.base_url == CLAWPRINT_DEFAULT_BASE_URL

    def test_custom_base_url_strips_slash(self) -> None:
        client = ClawPrintClient(base_url="https://custom.api.io/")
        assert client.base_url == "https://custom.api.io"

    def test_api_key_from_env(self) -> None:
        with patch.dict("os.environ", {"CLAWPRINT_API_KEY": "cp_env_key"}):
            client = ClawPrintClient()
            assert client.api_key == "cp_env_key"

    def test_explicit_key_overrides_env(self) -> None:
        with patch.dict("os.environ", {"CLAWPRINT_API_KEY": "cp_env_key"}):
            client = ClawPrintClient(api_key="cp_explicit")
            assert client.api_key == "cp_explicit"

    def test_auth_required_without_key_raises(self) -> None:
        ClawPrintClient(api_key=None)
        # Ensure env is clean
        with patch.dict("os.environ", {}, clear=True):
            client_no_key = ClawPrintClient(api_key=None)
            with pytest.raises(ClawPrintAPIError, match="requires an API key"):
                client_no_key._headers(auth_required=True)

    def test_auth_header_set(self) -> None:
        client = ClawPrintClient(api_key="cp_test_123")
        headers = client._headers(auth_required=True)
        assert headers["Authorization"] == "Bearer cp_test_123"

    def test_handle_response_success(self, mock_client: ClawPrintClient) -> None:
        resp = _mock_response({"results": [], "total": 0})
        result = mock_client._handle_response(resp)
        assert result == {"results": [], "total": 0}

    def test_handle_response_204(self, mock_client: ClawPrintClient) -> None:
        resp = _mock_response(None, status_code=204)
        result = mock_client._handle_response(resp)
        assert result == {}

    def test_handle_response_error(self, mock_client: ClawPrintClient) -> None:
        resp = _mock_response({"detail": "Not found"}, status_code=404)
        with pytest.raises(ClawPrintAPIError, match="404"):
            mock_client._handle_response(resp)

    def test_search_agents(self, mock_client: ClawPrintClient) -> None:
        mock_client._session.get.return_value = _mock_response(
            {"results": [{"handle": "bot1", "name": "Bot One"}], "total": 1}
        )
        result = mock_client.search_agents("code review")
        mock_client._session.get.assert_called_once()
        call_args = mock_client._session.get.call_args
        assert "/v1/agents/search" in call_args[0][0]
        assert call_args[1]["params"]["q"] == "code review"
        expected = {"results": [{"handle": "bot1", "name": "Bot One"}], "total": 1}
        assert result == expected

    def test_search_agents_with_filters(self, mock_client: ClawPrintClient) -> None:
        mock_client._session.get.return_value = _mock_response(
            {"results": [], "total": 0}
        )
        mock_client.search_agents("test", domain="code-review", min_score=80.0)
        call_args = mock_client._session.get.call_args
        assert call_args[1]["params"]["domain"] == "code-review"
        assert call_args[1]["params"]["min_score"] == 80.0

    def test_get_agent(self, mock_client: ClawPrintClient) -> None:
        mock_client._session.get.return_value = _mock_response(
            {"handle": "bot1", "name": "Bot One"}
        )
        result = mock_client.get_agent("@bot1")
        # validate_handle strips the @ prefix
        assert "/v1/agents/bot1" in mock_client._session.get.call_args[0][0]
        assert result["handle"] == "bot1"

    def test_get_trust(self, mock_client: ClawPrintClient) -> None:
        mock_client._session.get.return_value = _mock_response(
            {"handle": "bot1", "trust_score": 85, "grade": "B"}
        )
        result = mock_client.get_trust("@bot1")
        # validate_handle strips the @ prefix
        assert "/v1/trust/bot1" in mock_client._session.get.call_args[0][0]
        assert result["trust_score"] == 85

    def test_list_domains(self, mock_client: ClawPrintClient) -> None:
        mock_client._session.get.return_value = _mock_response(
            {"domains": ["code-review", "translation"]}
        )
        result = mock_client.list_domains()
        assert "/v1/domains" in mock_client._session.get.call_args[0][0]
        assert "code-review" in result["domains"]

    def test_create_exchange_request(self, mock_client: ClawPrintClient) -> None:
        mock_client._session.post.return_value = _mock_response(
            {"request_id": "req_abc123", "status": "pending"}
        )
        result = mock_client.create_exchange_request(
            domains=["code-review"], task="Review my code"
        )
        call_args = mock_client._session.post.call_args
        assert "/v1/exchange/requests" in call_args[0][0]
        payload = call_args[1]["json"]
        assert payload["domains"] == ["code-review"]
        assert payload["task"] == "Review my code"
        assert result["request_id"] == "req_abc123"

    def test_register_agent(self, mock_client: ClawPrintClient) -> None:
        mock_client._session.post.return_value = _mock_response(
            {"handle": "my-bot", "api_key": "cp_live_abc123"}
        )
        result = mock_client.register_agent(
            handle="my-bot",
            name="My Bot",
            description="Does useful things",
            domains=["data-analysis"],
        )
        call_args = mock_client._session.post.call_args
        assert "/v1/agents" in call_args[0][0]
        payload = call_args[1]["json"]
        assert payload["identity"]["handle"] == "my-bot"
        assert payload["identity"]["name"] == "My Bot"
        assert payload["services"][0]["domains"] == ["data-analysis"]
        assert result["api_key"] == "cp_live_abc123"

    def test_register_agent_with_url(self, mock_client: ClawPrintClient) -> None:
        mock_client._session.post.return_value = _mock_response(
            {"handle": "my-bot", "api_key": "cp_live_abc123"}
        )
        mock_client.register_agent(
            handle="my-bot",
            name="My Bot",
            description="Does useful things",
            domains=["data-analysis"],
            url="https://mybot.example.com",
        )
        payload = mock_client._session.post.call_args[1]["json"]
        assert payload["identity"]["url"] == "https://mybot.example.com"

    def test_get_exchange_request(self, mock_client: ClawPrintClient) -> None:
        mock_client._session.get.return_value = _mock_response(
            {"request_id": "req_abc123", "status": "matched"}
        )
        result = mock_client.get_exchange_request("req_abc123")
        assert (
            "/v1/exchange/requests/req_abc123"
            in (mock_client._session.get.call_args[0][0])
        )
        assert result["status"] == "matched"


# ======================================================================
# Tool tests
# ======================================================================


class TestClawPrintSearchTool:
    def test_name_and_description(self, mock_client: ClawPrintClient) -> None:
        tool = ClawPrintSearchTool(client=mock_client)
        assert tool.name == "clawprint_search"
        assert "search" in tool.description.lower()

    def test_run_basic_search(self, mock_client: ClawPrintClient) -> None:
        mock_client._session.get.return_value = _mock_response(
            {"results": [{"handle": "analyst", "name": "Data Analyst"}], "total": 1}
        )
        tool = ClawPrintSearchTool(client=mock_client)
        result = tool._run(query="data analysis")
        parsed = json.loads(result)
        assert parsed["results"][0]["handle"] == "analyst"

    def test_run_with_min_trust(self, mock_client: ClawPrintClient) -> None:
        """min_trust is passed directly to the API as min_score (0-100)."""
        mock_client._session.get.return_value = _mock_response(
            {"results": [], "total": 0}
        )
        tool = ClawPrintSearchTool(client=mock_client)
        tool._run(query="test", min_trust=80)
        params = mock_client._session.get.call_args[1]["params"]
        assert params["min_score"] == 80

    def test_invoke(self, mock_client: ClawPrintClient) -> None:
        mock_client._session.get.return_value = _mock_response(
            {"results": [], "total": 0}
        )
        tool = ClawPrintSearchTool(client=mock_client)
        result = tool.invoke({"query": "test"})
        assert isinstance(result, str)


class TestClawPrintGetAgentTool:
    def test_run(self, mock_client: ClawPrintClient) -> None:
        mock_client._session.get.return_value = _mock_response(
            {"handle": "@codebot", "name": "Code Bot"}
        )
        tool = ClawPrintGetAgentTool(client=mock_client)
        result = tool._run(handle="@codebot")
        parsed = json.loads(result)
        assert parsed["handle"] == "@codebot"


class TestClawPrintTrustTool:
    def test_run(self, mock_client: ClawPrintClient) -> None:
        mock_client._session.get.return_value = _mock_response(
            {
                "handle": "codebot",
                "trust_score": 92,
                "grade": "A",
                "verification": {"level": "platform-verified"},
            }
        )
        tool = ClawPrintTrustTool(client=mock_client)
        result = tool._run(handle="@codebot")
        parsed = json.loads(result)
        assert parsed["trust_score"] == 92


class TestClawPrintDomainsTool:
    def test_has_args_schema(self, mock_client: ClawPrintClient) -> None:
        tool = ClawPrintDomainsTool(client=mock_client)
        assert tool.args_schema is not None

    def test_run(self, mock_client: ClawPrintClient) -> None:
        mock_client._session.get.return_value = _mock_response(
            {"domains": ["code-review", "data-analysis", "translation"]}
        )
        tool = ClawPrintDomainsTool(client=mock_client)
        result = tool._run()
        parsed = json.loads(result)
        assert len(parsed["domains"]) == 3


class TestClawPrintRegisterTool:
    def test_name_and_description(self, mock_client: ClawPrintClient) -> None:
        tool = ClawPrintRegisterTool(client=mock_client)
        assert tool.name == "clawprint_register"
        assert "register" in tool.description.lower()

    def test_run(self, mock_client: ClawPrintClient) -> None:
        mock_client._session.post.return_value = _mock_response(
            {"handle": "my-bot", "api_key": "cp_live_new123"}
        )
        tool = ClawPrintRegisterTool(client=mock_client)
        result = tool._run(
            handle="my-bot",
            name="My Bot",
            description="Reviews code",
            domains=["code-review"],
        )
        parsed = json.loads(result)
        assert parsed["handle"] == "my-bot"
        assert parsed["api_key"] == "cp_live_new123"

    def test_run_with_url(self, mock_client: ClawPrintClient) -> None:
        mock_client._session.post.return_value = _mock_response(
            {"handle": "my-bot", "api_key": "cp_live_new123"}
        )
        tool = ClawPrintRegisterTool(client=mock_client)
        tool._run(
            handle="my-bot",
            name="My Bot",
            description="Reviews code",
            domains=["code-review"],
            url="https://mybot.dev",
        )
        payload = mock_client._session.post.call_args[1]["json"]
        assert payload["identity"]["url"] == "https://mybot.dev"


class TestClawPrintHireAgentTool:
    def test_run(self, mock_client: ClawPrintClient) -> None:
        mock_client._session.post.return_value = _mock_response(
            {"request_id": "req_xyz", "status": "pending"}
        )
        tool = ClawPrintHireAgentTool(client=mock_client)
        result = tool._run(domains=["code-review"], task="Review my PR")
        parsed = json.loads(result)
        assert parsed["request_id"] == "req_xyz"

    def test_run_with_requirements(self, mock_client: ClawPrintClient) -> None:
        mock_client._session.post.return_value = _mock_response(
            {"request_id": "req_xyz", "status": "pending"}
        )
        tool = ClawPrintHireAgentTool(client=mock_client)
        tool._run(
            domains=["code-review"],
            task="Review my PR",
            requirements={"budget": 100, "deadline": "2025-12-31"},
        )
        payload = mock_client._session.post.call_args[1]["json"]
        assert payload["requirements"]["budget"] == 100


class TestClawPrintCheckExchangeTool:
    def test_run(self, mock_client: ClawPrintClient) -> None:
        mock_client._session.get.return_value = _mock_response(
            {"request_id": "req_xyz", "status": "completed"}
        )
        tool = ClawPrintCheckExchangeTool(client=mock_client)
        result = tool._run(request_id="req_xyz")
        parsed = json.loads(result)
        assert parsed["status"] == "completed"


# ======================================================================
# Toolkit tests
# ======================================================================


class TestClawPrintToolkit:
    def test_get_tools_returns_seven(self) -> None:
        toolkit = ClawPrintToolkit(api_key="cp_test_key")
        tools = toolkit.get_tools()
        assert len(tools) == 7

    def test_tool_names(self) -> None:
        toolkit = ClawPrintToolkit(api_key="cp_test_key")
        tools = toolkit.get_tools()
        names = {t.name for t in tools}
        expected = {
            "clawprint_register",
            "clawprint_search",
            "clawprint_get_agent",
            "clawprint_trust",
            "clawprint_domains",
            "clawprint_hire",
            "clawprint_check_exchange",
        }
        assert names == expected

    def test_shared_client(self) -> None:
        toolkit = ClawPrintToolkit(api_key="cp_test_key")
        tools = toolkit.get_tools()
        clients = {id(t.client) for t in tools}  # type: ignore[attr-defined]
        assert len(clients) == 1, "All tools should share the same client"

    def test_default_no_key(self) -> None:
        """Toolkit can be created without a key (read-only use)."""
        with patch.dict("os.environ", {}, clear=True):
            toolkit = ClawPrintToolkit()
            tools = toolkit.get_tools()
            assert len(tools) == 7

    def test_get_tool_by_name(self) -> None:
        """get_tool() returns the correct tool for a valid name."""
        toolkit = ClawPrintToolkit(api_key="cp_test_key")
        tool = toolkit.get_tool("clawprint_search")
        assert isinstance(tool, ClawPrintSearchTool)
        assert tool.name == "clawprint_search"

    def test_get_tool_all_names(self) -> None:
        """get_tool() works for every tool name returned by get_tools()."""
        toolkit = ClawPrintToolkit(api_key="cp_test_key")
        all_tools = toolkit.get_tools()
        for t in all_tools:
            found = toolkit.get_tool(t.name)
            assert found.name == t.name

    def test_get_tool_invalid_name(self) -> None:
        """get_tool() raises KeyError with available names for invalid input."""
        toolkit = ClawPrintToolkit(api_key="cp_test_key")
        with pytest.raises(KeyError, match="No tool named 'nonexistent'") as exc_info:
            toolkit.get_tool("nonexistent")
        # Verify available names are included in the error message
        error_msg = str(exc_info.value)
        assert "clawprint_search" in error_msg
        assert "clawprint_trust" in error_msg


# ======================================================================
# Async tool tests
# ======================================================================


class TestAsyncToolRun:
    """Test that each tool's _arun method works with async client methods."""

    @pytest.mark.asyncio
    async def test_search_arun(self, mock_client: ClawPrintClient) -> None:
        mock_client.asearch_agents = AsyncMock(
            return_value={"results": [{"handle": "async-bot"}], "total": 1}
        )
        tool = ClawPrintSearchTool(client=mock_client)
        result = await tool._arun(query="async test")
        parsed = json.loads(result)
        assert parsed["results"][0]["handle"] == "async-bot"
        mock_client.asearch_agents.assert_awaited_once_with(
            "async test", domain=None, min_score=None
        )

    @pytest.mark.asyncio
    async def test_search_arun_with_filters(self, mock_client: ClawPrintClient) -> None:
        mock_client.asearch_agents = AsyncMock(return_value={"results": [], "total": 0})
        tool = ClawPrintSearchTool(client=mock_client)
        result = await tool._arun(query="filtered", domain="code-review", min_trust=90)
        parsed = json.loads(result)
        assert parsed == {"results": [], "total": 0}
        # min_trust passed directly as min_score (0-100)
        mock_client.asearch_agents.assert_awaited_once_with(
            "filtered", domain="code-review", min_score=90
        )

    @pytest.mark.asyncio
    async def test_get_agent_arun(self, mock_client: ClawPrintClient) -> None:
        mock_client.aget_agent = AsyncMock(
            return_value={"handle": "@codebot", "name": "Code Bot"}
        )
        tool = ClawPrintGetAgentTool(client=mock_client)
        result = await tool._arun(handle="@codebot")
        parsed = json.loads(result)
        assert parsed["handle"] == "@codebot"
        assert parsed["name"] == "Code Bot"
        mock_client.aget_agent.assert_awaited_once_with("@codebot")

    @pytest.mark.asyncio
    async def test_trust_arun(self, mock_client: ClawPrintClient) -> None:
        mock_client.aget_trust = AsyncMock(
            return_value={
                "handle": "codebot",
                "trust_score": 92,
                "grade": "A",
                "verification": {"level": "platform-verified"},
            }
        )
        tool = ClawPrintTrustTool(client=mock_client)
        result = await tool._arun(handle="@codebot")
        parsed = json.loads(result)
        assert parsed["trust_score"] == 92
        assert parsed["verification"]["level"] == "platform-verified"
        mock_client.aget_trust.assert_awaited_once_with("@codebot")

    @pytest.mark.asyncio
    async def test_domains_arun(self, mock_client: ClawPrintClient) -> None:
        mock_client.alist_domains = AsyncMock(
            return_value={"domains": ["code-review", "translation"]}
        )
        tool = ClawPrintDomainsTool(client=mock_client)
        result = await tool._arun()
        parsed = json.loads(result)
        assert "code-review" in parsed["domains"]
        assert len(parsed["domains"]) == 2
        mock_client.alist_domains.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_register_arun(self, mock_client: ClawPrintClient) -> None:
        mock_client.aregister_agent = AsyncMock(
            return_value={"handle": "my-bot", "api_key": "cp_live_async123"}
        )
        tool = ClawPrintRegisterTool(client=mock_client)
        result = await tool._arun(
            handle="my-bot",
            name="My Bot",
            description="Does things",
            domains=["code-review"],
        )
        parsed = json.loads(result)
        assert parsed["handle"] == "my-bot"
        assert parsed["api_key"] == "cp_live_async123"
        mock_client.aregister_agent.assert_awaited_once_with(
            handle="my-bot",
            name="My Bot",
            description="Does things",
            domains=["code-review"],
            url=None,
        )

    @pytest.mark.asyncio
    async def test_register_arun_with_url(self, mock_client: ClawPrintClient) -> None:
        mock_client.aregister_agent = AsyncMock(
            return_value={"handle": "my-bot", "api_key": "cp_live_async123"}
        )
        tool = ClawPrintRegisterTool(client=mock_client)
        result = await tool._arun(
            handle="my-bot",
            name="My Bot",
            description="Does things",
            domains=["code-review"],
            url="https://mybot.dev",
        )
        parsed = json.loads(result)
        assert parsed["handle"] == "my-bot"
        mock_client.aregister_agent.assert_awaited_once_with(
            handle="my-bot",
            name="My Bot",
            description="Does things",
            domains=["code-review"],
            url="https://mybot.dev",
        )

    @pytest.mark.asyncio
    async def test_hire_arun(self, mock_client: ClawPrintClient) -> None:
        mock_client.acreate_exchange_request = AsyncMock(
            return_value={"request_id": "req_async1", "status": "pending"}
        )
        tool = ClawPrintHireAgentTool(client=mock_client)
        result = await tool._arun(domains=["code-review"], task="Review my PR")
        parsed = json.loads(result)
        assert parsed["request_id"] == "req_async1"
        assert parsed["status"] == "pending"
        mock_client.acreate_exchange_request.assert_awaited_once_with(
            domains=["code-review"], task="Review my PR", requirements=None
        )

    @pytest.mark.asyncio
    async def test_hire_arun_with_requirements(
        self, mock_client: ClawPrintClient
    ) -> None:
        mock_client.acreate_exchange_request = AsyncMock(
            return_value={"request_id": "req_async2", "status": "pending"}
        )
        tool = ClawPrintHireAgentTool(client=mock_client)
        reqs = {"budget": 200, "deadline": "2025-06-01"}
        result = await tool._arun(
            domains=["data-analysis"], task="Analyze dataset", requirements=reqs
        )
        parsed = json.loads(result)
        assert parsed["request_id"] == "req_async2"
        mock_client.acreate_exchange_request.assert_awaited_once_with(
            domains=["data-analysis"], task="Analyze dataset", requirements=reqs
        )

    @pytest.mark.asyncio
    async def test_check_exchange_arun(self, mock_client: ClawPrintClient) -> None:
        mock_client.aget_exchange_request = AsyncMock(
            return_value={"request_id": "req_xyz", "status": "completed"}
        )
        tool = ClawPrintCheckExchangeTool(client=mock_client)
        result = await tool._arun(request_id="req_xyz")
        parsed = json.loads(result)
        assert parsed["status"] == "completed"
        mock_client.aget_exchange_request.assert_awaited_once_with("req_xyz")


# ======================================================================
# Async error handling tests
# ======================================================================


class TestAsyncErrorHandling:
    """Test that async methods return error strings on ClawPrintAPIError."""

    @pytest.mark.asyncio
    async def test_search_arun_api_error(self, mock_client: ClawPrintClient) -> None:
        mock_client.asearch_agents = AsyncMock(
            side_effect=ClawPrintAPIError(500, "Internal server error")
        )
        tool = ClawPrintSearchTool(client=mock_client)
        result = await tool._arun(query="test")
        assert result.startswith("Error:")
        assert "500" in result

    @pytest.mark.asyncio
    async def test_search_arun_value_error(self, mock_client: ClawPrintClient) -> None:
        mock_client.asearch_agents = AsyncMock(
            side_effect=ValueError("Search query cannot be empty.")
        )
        tool = ClawPrintSearchTool(client=mock_client)
        result = await tool._arun(query="")
        assert result.startswith("Error:")
        assert "empty" in result.lower()

    @pytest.mark.asyncio
    async def test_get_agent_arun_error(self, mock_client: ClawPrintClient) -> None:
        mock_client.aget_agent = AsyncMock(
            side_effect=ClawPrintAPIError(404, "Agent not found")
        )
        tool = ClawPrintGetAgentTool(client=mock_client)
        result = await tool._arun(handle="@nonexistent")
        assert result.startswith("Error:")
        assert "404" in result

    @pytest.mark.asyncio
    async def test_trust_arun_error(self, mock_client: ClawPrintClient) -> None:
        mock_client.aget_trust = AsyncMock(
            side_effect=ClawPrintAPIError(404, "Agent not found")
        )
        tool = ClawPrintTrustTool(client=mock_client)
        result = await tool._arun(handle="@nonexistent")
        assert result.startswith("Error:")
        assert "404" in result

    @pytest.mark.asyncio
    async def test_domains_arun_error(self, mock_client: ClawPrintClient) -> None:
        mock_client.alist_domains = AsyncMock(
            side_effect=ClawPrintAPIError(503, "Service unavailable")
        )
        tool = ClawPrintDomainsTool(client=mock_client)
        result = await tool._arun()
        assert result.startswith("Error:")
        assert "503" in result

    @pytest.mark.asyncio
    async def test_register_arun_error(self, mock_client: ClawPrintClient) -> None:
        mock_client.aregister_agent = AsyncMock(
            side_effect=ClawPrintAPIError(409, "Handle already taken")
        )
        tool = ClawPrintRegisterTool(client=mock_client)
        result = await tool._arun(
            handle="taken-bot",
            name="Taken",
            description="Already exists",
            domains=["test"],
        )
        assert result.startswith("Error:")
        assert "409" in result

    @pytest.mark.asyncio
    async def test_register_arun_validation_error(
        self, mock_client: ClawPrintClient
    ) -> None:
        mock_client.aregister_agent = AsyncMock(
            side_effect=ValueError("Invalid handle 'BAD!'.")
        )
        tool = ClawPrintRegisterTool(client=mock_client)
        result = await tool._arun(
            handle="BAD!",
            name="Bad Bot",
            description="Invalid",
            domains=["test"],
        )
        assert result.startswith("Error:")
        assert "Invalid" in result

    @pytest.mark.asyncio
    async def test_hire_arun_error(self, mock_client: ClawPrintClient) -> None:
        mock_client.acreate_exchange_request = AsyncMock(
            side_effect=ClawPrintAPIError(401, "Authentication required")
        )
        tool = ClawPrintHireAgentTool(client=mock_client)
        result = await tool._arun(domains=["test"], task="test task")
        assert result.startswith("Error:")
        assert "401" in result

    @pytest.mark.asyncio
    async def test_check_exchange_arun_error(
        self, mock_client: ClawPrintClient
    ) -> None:
        mock_client.aget_exchange_request = AsyncMock(
            side_effect=ClawPrintAPIError(404, "Request not found")
        )
        tool = ClawPrintCheckExchangeTool(client=mock_client)
        result = await tool._arun(request_id="req_nonexistent")
        assert result.startswith("Error:")
        assert "404" in result


# ======================================================================
# Sync error handling tests
# ======================================================================


class TestSyncErrorHandling:
    """Test that sync methods return error strings on exceptions."""

    def test_search_api_error(self, mock_client: ClawPrintClient) -> None:
        mock_client._session.get.return_value = _mock_response(
            {"detail": "Server error"}, status_code=500
        )
        tool = ClawPrintSearchTool(client=mock_client)
        result = tool._run(query="test")
        assert result.startswith("Error:")
        assert "500" in result

    def test_get_agent_api_error(self, mock_client: ClawPrintClient) -> None:
        mock_client._session.get.return_value = _mock_response(
            {"detail": "Not found"}, status_code=404
        )
        tool = ClawPrintGetAgentTool(client=mock_client)
        result = tool._run(handle="@nonexistent")
        assert result.startswith("Error:")
        assert "404" in result

    def test_trust_api_error(self, mock_client: ClawPrintClient) -> None:
        mock_client._session.get.return_value = _mock_response(
            {"detail": "Not found"}, status_code=404
        )
        tool = ClawPrintTrustTool(client=mock_client)
        result = tool._run(handle="@nonexistent")
        assert result.startswith("Error:")

    def test_domains_api_error(self, mock_client: ClawPrintClient) -> None:
        mock_client._session.get.return_value = _mock_response(
            {"detail": "Service down"}, status_code=503
        )
        tool = ClawPrintDomainsTool(client=mock_client)
        result = tool._run()
        assert result.startswith("Error:")

    def test_register_api_error(self, mock_client: ClawPrintClient) -> None:
        mock_client._session.post.return_value = _mock_response(
            {"detail": "Conflict"}, status_code=409
        )
        tool = ClawPrintRegisterTool(client=mock_client)
        result = tool._run(handle="taken", name="X", description="X", domains=["x"])
        assert result.startswith("Error:")

    def test_hire_api_error(self, mock_client: ClawPrintClient) -> None:
        mock_client._session.post.return_value = _mock_response(
            {"detail": "Unauthorized"}, status_code=401
        )
        tool = ClawPrintHireAgentTool(client=mock_client)
        result = tool._run(domains=["test"], task="test")
        assert result.startswith("Error:")

    def test_check_exchange_api_error(self, mock_client: ClawPrintClient) -> None:
        mock_client._session.get.return_value = _mock_response(
            {"detail": "Not found"}, status_code=404
        )
        tool = ClawPrintCheckExchangeTool(client=mock_client)
        result = tool._run(request_id="req_missing")
        assert result.startswith("Error:")
