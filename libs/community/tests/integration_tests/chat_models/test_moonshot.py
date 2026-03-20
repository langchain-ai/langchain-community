"""Test Moonshot Chat Model."""

import os
from typing import Type, cast

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_tests.integration_tests import ChatModelIntegrationTests
from pydantic import SecretStr

from langchain_community.chat_models.moonshot import MoonshotChat


class TestMoonshotChat(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return MoonshotChat

    @property
    def chat_model_params(self) -> dict:
        return {"model": "moonshot-v1-8k"}

    @pytest.mark.xfail(reason="Not yet implemented.")
    def test_usage_metadata(self, model: BaseChatModel) -> None:
        super().test_usage_metadata(model)


def test_chat_moonshot_instantiate_with_alias() -> None:
    """Test MoonshotChat instantiate when using alias."""
    api_key = "your-api-key"
    chat = MoonshotChat(api_key=api_key)  # type: ignore[call-arg]
    assert cast(SecretStr, chat.moonshot_api_key).get_secret_value() == api_key


@pytest.mark.skipif(
        os.getenv("MOONSHOT_API_KEY") is None,
        reason="MOONSHOT_API_KEY environment variable not set."
)
@pytest.mark.parametrize("model_name", ["kimi-k2-turbo-preview", "kimi-k2.5"])
def test_chat_moonshot_tool_call(model_name: str) -> None:

    @tool
    def get_weather(location: str) -> str:
        """Get the weather for a given location."""
        return "Sunny"

    chat = MoonshotChat(model=model_name)
    conversation = [
        SystemMessage(content="You are a helful assistant."),
        HumanMessage(content="What is the weather in Beijing today?"),
    ]
    response = cast(AIMessage, chat.bind_tools([get_weather]).invoke(conversation))
    assert len(response.tool_calls) == 1
    conversation.append(response)

    tool_message = ToolMessage(
        content="Sunny",
        tool_call_id=response.tool_calls[0]["id"],
    )

    conversation.append(tool_message)
    response = cast(AIMessage, chat.invoke(conversation))
    assert "sunny" in response.content.lower()


