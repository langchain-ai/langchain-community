"""Unit tests for Anakin tools."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from langchain_community.tools.anakin.tool import (
    AnakinAgenticSearchTool,
    AnakinScrapeTool,
    AnakinSearchTool,
)

# ------------------------------------------------------------------
# AnakinScrapeTool
# ------------------------------------------------------------------


class TestAnakinScrapeTool:
    def test_name_and_description(self) -> None:
        with patch.dict("os.environ", {"ANAKIN_API_KEY": "ak-test"}):
            tool = AnakinScrapeTool()
        assert tool.name == "anakin_scrape"
        assert "scrape" in tool.description.lower()

    def test_args_schema(self) -> None:
        with patch.dict("os.environ", {"ANAKIN_API_KEY": "ak-test"}):
            tool = AnakinScrapeTool()
        schema = tool.args_schema.model_json_schema()
        assert "url" in schema["properties"]
        assert "use_browser" in schema["properties"]

    @patch("langchain_community.utilities.anakin.requests.get")
    @patch("langchain_community.utilities.anakin.requests.post")
    def test_run_returns_markdown(
        self, mock_post: MagicMock, mock_get: MagicMock
    ) -> None:
        mock_post.return_value = MagicMock(
            json=lambda: {"jobId": "job_scrape"},
        )
        mock_post.return_value.raise_for_status = MagicMock()
        mock_get.return_value = MagicMock(
            json=lambda: {
                "status": "completed",
                "markdown": "# Page Title\n\nContent here.",
            },
        )
        mock_get.return_value.raise_for_status = MagicMock()

        tool = AnakinScrapeTool(api_key="ak-test")
        result = tool._run(url="https://example.com")

        assert isinstance(result, str)
        assert "Page Title" in result
        assert "Content here." in result

    @patch("langchain_community.utilities.anakin.requests.get")
    @patch("langchain_community.utilities.anakin.requests.post")
    def test_invoke(self, mock_post: MagicMock, mock_get: MagicMock) -> None:
        mock_post.return_value = MagicMock(
            json=lambda: {"jobId": "job_invoke"},
        )
        mock_post.return_value.raise_for_status = MagicMock()
        mock_get.return_value = MagicMock(
            json=lambda: {
                "status": "completed",
                "markdown": "Scraped content.",
            },
        )
        mock_get.return_value.raise_for_status = MagicMock()

        tool = AnakinScrapeTool(api_key="ak-test")
        result = tool.invoke({"url": "https://example.com"})
        assert isinstance(result, str)
        assert "Scraped content." in result


# ------------------------------------------------------------------
# AnakinSearchTool
# ------------------------------------------------------------------


class TestAnakinSearchTool:
    def test_name_and_description(self) -> None:
        with patch.dict("os.environ", {"ANAKIN_API_KEY": "ak-test"}):
            tool = AnakinSearchTool()
        assert tool.name == "anakin_search"
        assert "search" in tool.description.lower()

    def test_args_schema(self) -> None:
        with patch.dict("os.environ", {"ANAKIN_API_KEY": "ak-test"}):
            tool = AnakinSearchTool()
        schema = tool.args_schema.model_json_schema()
        assert "query" in schema["properties"]
        assert "limit" in schema["properties"]

    @patch("langchain_community.utilities.anakin.requests.post")
    def test_run_returns_formatted_results(self, mock_post: MagicMock) -> None:
        mock_post.return_value = MagicMock(
            json=lambda: {
                "results": [
                    {
                        "title": "Result One",
                        "snippet": "First snippet",
                        "url": "https://one.com",
                    },
                    {
                        "title": "Result Two",
                        "snippet": "Second snippet",
                        "url": "https://two.com",
                    },
                ]
            },
        )
        mock_post.return_value.raise_for_status = MagicMock()

        tool = AnakinSearchTool(api_key="ak-test")
        result = tool._run(query="test query")

        assert isinstance(result, str)
        assert "[1] Result One" in result
        assert "First snippet" in result
        assert "https://one.com" in result
        assert "[2] Result Two" in result

    @patch("langchain_community.utilities.anakin.requests.post")
    def test_run_empty_results(self, mock_post: MagicMock) -> None:
        mock_post.return_value = MagicMock(
            json=lambda: {"results": []},
        )
        mock_post.return_value.raise_for_status = MagicMock()

        tool = AnakinSearchTool(api_key="ak-test")
        result = tool._run(query="nothing")
        assert result == "No results found."

    @patch("langchain_community.utilities.anakin.requests.post")
    def test_invoke(self, mock_post: MagicMock) -> None:
        mock_post.return_value = MagicMock(
            json=lambda: {
                "results": [
                    {"title": "Hit", "snippet": "Found it", "url": "https://hit.com"}
                ]
            },
        )
        mock_post.return_value.raise_for_status = MagicMock()

        tool = AnakinSearchTool(api_key="ak-test")
        result = tool.invoke({"query": "test"})
        assert "Hit" in result


# ------------------------------------------------------------------
# AnakinAgenticSearchTool
# ------------------------------------------------------------------


class TestAnakinAgenticSearchTool:
    def test_name_and_description(self) -> None:
        with patch.dict("os.environ", {"ANAKIN_API_KEY": "ak-test"}):
            tool = AnakinAgenticSearchTool()
        assert tool.name == "anakin_agentic_search"
        assert "research" in tool.description.lower()

    def test_args_schema(self) -> None:
        with patch.dict("os.environ", {"ANAKIN_API_KEY": "ak-test"}):
            tool = AnakinAgenticSearchTool()
        schema = tool.args_schema.model_json_schema()
        assert "query" in schema["properties"]

    @patch("langchain_community.utilities.anakin.requests.get")
    @patch("langchain_community.utilities.anakin.requests.post")
    def test_run_returns_summary(
        self, mock_post: MagicMock, mock_get: MagicMock
    ) -> None:
        mock_post.return_value = MagicMock(
            json=lambda: {"job_id": "agent_tool_1", "status": "pending"},
        )
        mock_post.return_value.raise_for_status = MagicMock()
        mock_get.return_value = MagicMock(
            json=lambda: {
                "status": "completed",
                "generatedJson": {
                    "summary": "Detailed analysis of the topic reveals...",
                },
            },
        )
        mock_get.return_value.raise_for_status = MagicMock()

        tool = AnakinAgenticSearchTool(api_key="ak-test")
        result = tool._run(query="compare frameworks")

        assert isinstance(result, str)
        assert "Detailed analysis" in result

    @patch("langchain_community.utilities.anakin.requests.get")
    @patch("langchain_community.utilities.anakin.requests.post")
    def test_run_no_summary_fallback(
        self, mock_post: MagicMock, mock_get: MagicMock
    ) -> None:
        mock_post.return_value = MagicMock(
            json=lambda: {"job_id": "agent_empty", "status": "pending"},
        )
        mock_post.return_value.raise_for_status = MagicMock()
        mock_get.return_value = MagicMock(
            json=lambda: {
                "status": "completed",
                "generatedJson": {},
            },
        )
        mock_get.return_value.raise_for_status = MagicMock()

        tool = AnakinAgenticSearchTool(api_key="ak-test")
        result = tool._run(query="test")
        assert result == "No summary available."

    @patch("langchain_community.utilities.anakin.requests.get")
    @patch("langchain_community.utilities.anakin.requests.post")
    def test_invoke(self, mock_post: MagicMock, mock_get: MagicMock) -> None:
        mock_post.return_value = MagicMock(
            json=lambda: {"job_id": "agent_invoke", "status": "pending"},
        )
        mock_post.return_value.raise_for_status = MagicMock()
        mock_get.return_value = MagicMock(
            json=lambda: {
                "status": "completed",
                "generatedJson": {"summary": "Report content."},
            },
        )
        mock_get.return_value.raise_for_status = MagicMock()

        tool = AnakinAgenticSearchTool(api_key="ak-test")
        result = tool.invoke({"query": "deep dive"})
        assert "Report content." in result
