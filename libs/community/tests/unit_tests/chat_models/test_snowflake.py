"""Test ChatSnowflakeCortex."""

import os

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langchain_community.chat_models.snowflake import (
    ChatSnowflakeCortex,
    ChatSnowflakeCortexError,
    _convert_message_to_dict,
)


def test_messages_to_prompt_dict_with_valid_messages() -> None:
    messages = [
        SystemMessage(content="System Prompt"),
        HumanMessage(content="User message #1"),
        AIMessage(content="AI message #1"),
        HumanMessage(content="User message #2"),
        AIMessage(content="AI message #2"),
    ]
    result = [_convert_message_to_dict(m) for m in messages]
    expected = [
        {"role": "system", "content": "System Prompt"},
        {"role": "user", "content": "User message #1"},
        {"role": "assistant", "content": "AI message #1"},
        {"role": "user", "content": "User message #2"},
        {"role": "assistant", "content": "AI message #2"},
    ]
    assert result == expected


def test_create_chat_with_invalid_config_in_env() -> None:
    os.environ["SNOWFLAKE_CONFIG"] = "{invalid json"
    try:
        with pytest.raises(ChatSnowflakeCortexError):
            ChatSnowflakeCortex()
    finally:
        os.environ.pop("SNOWFLAKE_CONFIG", None)


def test_create_chat_with_config_in_args() -> None:
    os.environ.pop("SNOWFLAKE_CONFIG", None)
    with pytest.raises(
        ChatSnowflakeCortexError,
        match="Failed to create session: 251005: User is empty",
    ):
        ChatSnowflakeCortex(snowflake_config={"account": "test"})
