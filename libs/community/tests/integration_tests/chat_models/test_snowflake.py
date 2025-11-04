"""Test ChatSnowflakeCortex
Note: This test must be run with the following environment variables set:
    SNOWFLAKE_ACCOUNT="YOUR_SNOWFLAKE_ACCOUNT",
    SNOWFLAKE_USERNAME="YOUR_SNOWFLAKE_USERNAME",
    SNOWFLAKE_PASSWORD="YOUR_SNOWFLAKE_PASSWORD",
    SNOWFLAKE_DATABASE="YOUR_SNOWFLAKE_DATABASE",
    SNOWFLAKE_SCHEMA="YOUR_SNOWFLAKE_SCHEMA",
    SNOWFLAKE_WAREHOUSE="YOUR_SNOWFLAKE_WAREHOUSE"
    SNOWFLAKE_ROLE="YOUR_SNOWFLAKE_ROLE",
    SNOWFLAKE_AUTHENTICATOR="YOUR_SNOWFLAKE_AUTHENTICATOR" (optional)
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
    response = chat.invoke([message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_chat_snowflake_cortex_system_message(chat: ChatSnowflakeCortex) -> None:
    """Test ChatSnowflakeCortex for system message"""
    system_message = SystemMessage(content="You are to chat with the user.")
    human_message = HumanMessage(content="Hello")
    response = chat.invoke([system_message, human_message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_chat_snowflake_cortex_model() -> None:
    """Test ChatSnowflakeCortex handles model_name."""
    chat = ChatSnowflakeCortex(
        model="foo",
    )
    assert chat.model == "foo"


def test_chat_snowflake_cortex_authenticator() -> None:
    """Test ChatSnowflakeCortex handles authenticator parameter."""
    chat = ChatSnowflakeCortex(
        authenticator="username_password_mfa",
    )
    assert chat.snowflake_authenticator == "username_password_mfa"


def test_chat_snowflake_cortex_authenticator_alias() -> None:
    """Test ChatSnowflakeCortex handles authenticator using alias."""
    chat = ChatSnowflakeCortex(
        authenticator="oauth",
    )
    assert chat.snowflake_authenticator == "oauth"


def test_chat_snowflake_cortex_authenticator_with_mfa() -> None:
    """Test ChatSnowflakeCortex works with MFA authenticator for token caching."""
    # This test verifies that the authenticator parameter is properly set
    # for MFA scenarios where token caching is beneficial
    chat = ChatSnowflakeCortex(
        authenticator="username_password_mfa",
    )

    # Verify the authenticator is set correctly
    assert chat.snowflake_authenticator == "username_password_mfa"

    # Test basic functionality - if authenticator setup is working,
    # the session should be created without errors
    message = HumanMessage(content="Hello")
    response = chat([message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


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


def test_chat_snowflake_cortex_message_with_special_characters(
    chat: ChatSnowflakeCortex,
) -> None:
    """Test ChatSnowflakeCortex with messages containing special characters.

    Args:
        chat: The ChatSnowflakeCortex instance to test with.
    """
    system_message = SystemMessage(content="You are to chat with the user.")
    human_message = HumanMessage(content="Can you give me the weather in Tokyo?\n\n")
    response = chat.invoke([system_message, human_message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)
