"""Tests ChatMLX tool calling."""

from typing import Dict

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from langchain_community.chat_models.mlx import ChatMLX
from langchain_community.llms.mlx_pipeline import MLXPipeline

# Use a Phi-3 model for more reliable tool-calling behavior
MODEL_ID = "mlx-community/phi-3-mini-128k-instruct"


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


@pytest.fixture(scope="module")
def chat() -> ChatMLX:
    """Return ChatMLX bound with the multiply tool or skip if unavailable."""
    try:
        llm = MLXPipeline.from_model_id(
            model_id=MODEL_ID, pipeline_kwargs={"max_new_tokens": 150}
        )
    except Exception:
        pytest.skip("Required MLX model isn't available.", allow_module_level=True)
    chat_model = ChatMLX(llm=llm)
    return chat_model.bind_tools(tools=[multiply], tool_choice=True)  # type: ignore[return-value]


def _call_tool(tool_call: Dict) -> ToolMessage:
    result = multiply.invoke(tool_call["args"])
    return ToolMessage(content=str(result), tool_call_id=tool_call.get("id", ""))


def test_mlx_tool_calls_soft(chat: ChatMLX) -> None:
    messages = [HumanMessage(content="Use the multiply tool to compute 2 * 3.")]
    ai_msg = chat.invoke(messages)
    tool_msg = _call_tool(ai_msg.tool_calls[0])
    final = chat.invoke(messages + [ai_msg, tool_msg])
    assert "6" in final.content


def test_mlx_tool_calls_hard(chat: ChatMLX) -> None:
    messages = [HumanMessage(content="Use the multiply tool to compute 2 * 3.")]
    ai_msg = chat.invoke(messages)
    assert isinstance(ai_msg, AIMessage)
    assert ai_msg.tool_calls
    tool_call = ai_msg.tool_calls[0]
    assert tool_call["name"] == "multiply"
    assert tool_call["args"] == {"a": 2, "b": 3}
