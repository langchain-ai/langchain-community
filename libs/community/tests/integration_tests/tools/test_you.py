import os

from langchain_community.tools.you import YouResearchTool
from langchain_community.utilities.you import YouSearchAPIWrapper


class TestYouResearchTool:
    @classmethod
    def setup_class(cls) -> None:
        if not os.getenv("YDC_API_KEY"):
            raise ValueError("YDC_API_KEY environment variable is not set")

    def test_invoke(self) -> None:
        tool = YouResearchTool(api_wrapper=YouSearchAPIWrapper())
        result = tool.invoke("what is quantum computing")

        assert isinstance(result, str)
        assert len(result) > 0
        # response should be markdown with a sources section
        assert "## Sources" in result
