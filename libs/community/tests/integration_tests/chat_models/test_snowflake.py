"""Test ChatSnowflakeCortex
Note: This test must be run with the following environment variables set:
    SNOWFLAKE_ACCOUNT="YOUR_SNOWFLAKE_ACCOUNT",
    SNOWFLAKE_USERNAME="YOUR_SNOWFLAKE_USERNAME",
    SNOWFLAKE_PASSWORD="YOUR_SNOWFLAKE_PASSWORD",
    SNOWFLAKE_DATABASE="YOUR_SNOWFLAKE_DATABASE",
    SNOWFLAKE_SCHEMA="YOUR_SNOWFLAKE_SCHEMA",
    SNOWFLAKE_WAREHOUSE="YOUR_SNOWFLAKE_WAREHOUSE",
    SNOWFLAKE_ROLE="YOUR_SNOWFLAKE_ROLE",

OR you can pass a connection_params dict and use any other authentication method
that is supported by snowflake. For example:

    snowflake_config = {
        "user": os.getenv("SNOWFLAKE_USER"),
        "authenticator": os.getenv("SNOWFLAKE_AUTHENTICATOR"),
        "account": os.getenv("SNOWFLAKE_ACCOUNT"),
        "role": os.getenv("SNOWFLAKE_ROLE"),
        "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
        "database": os.getenv("SNOWFLAKE_DATABASE"),
        "schema": os.getenv("SNOWFLAKE_SCHEMA"),
    }

OR create a environment variable SNOWFLAKE_CONFIG with the following format:

SNOWFLAKE_CONFIG='{
    "user": "YOUR_SNOWFLAKE_USER",
    "authenticator": "YOUR_SNOWFLAKE_AUTHENTICATOR",
    "account": "YOUR_SNOWFLAKE_ACCOUNT",
    "role": "YOUR_SNOWFLAKE_ROLE",
    "warehouse": "YOUR_SNOWFLAKE_WAREHOUSE",
    "database": "YOUR_SNOWFLAKE_DATABASE",
    "schema": "YOUR_SNOWFLAKE_SCHEMA"}'
'
"""

import pytest
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from langchain_community.chat_models import ChatSnowflakeCortex


@pytest.fixture
def chat() -> ChatSnowflakeCortex:
    return ChatSnowflakeCortex()


def test_chat_snowflake_cortex(chat: ChatSnowflakeCortex) -> None:
    """Test ChatSnowflakeCortex."""
    message = HumanMessage(content="Hello")
    response = chat([message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_chat_snowflake_cortex_system_message(chat: ChatSnowflakeCortex) -> None:
    """Test ChatSnowflakeCortex for system message"""
    system_message = SystemMessage(content="You are to chat with the user.")
    human_message = HumanMessage(content="Hello")
    response = chat([system_message, human_message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_chat_snowflake_cortex_model() -> None:
    """Test ChatSnowflakeCortex handles model_name."""
    chat = ChatSnowflakeCortex(
        model="foo",
    )
    assert chat.model == "foo"


def test_chat_snowflake_cortex_generate(chat: ChatSnowflakeCortex) -> None:
    """Test ChatSnowflakeCortex with generate."""
    message = HumanMessage(content="Hello")
    response = chat.generate([[message], [message]])
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    for generations in response.generations:
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content
