import pytest
from langchain_core.runnables import Runnable

from langchain_community.chat_models.moonshot import MoonshotChat

mock_tool_list = [lambda: f"tool-id-{i}" for i in range(3)]


@pytest.mark.requires("openai")
def test_moonshot_bind_tools() -> None:
    llm = MoonshotChat(name="moonshot")
    ret: Runnable = llm.bind_tools(mock_tool_list)
    assert len(ret.kwargs["tools"]) == 3
