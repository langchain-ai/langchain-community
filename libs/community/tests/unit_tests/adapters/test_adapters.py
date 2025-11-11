"""
Comprehensive pytest suite for langchain_community.adapters.openai

Tests both Langchain 1.0+ (tool_calls/tool_call_chunks attributes)
and pre-1.0 (additional_kwargs) compatibility.

Run with: pytest test_openai_adapter.py -v
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any

# Import the functions to test
from langchain_community.adapters.openai import (
    convert_message_to_dict,
    convert_dict_to_message,
    convert_openai_messages,
    _convert_message_chunk,
    _convert_message_chunk_to_delta,
    _has_assistant_message,
    convert_messages_for_finetuning,
    ChatCompletion,
    Completions,
    aenumerate,
)

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    FunctionMessage,
    ToolMessage,
    ChatMessage,
)


# =============================================================================
# Test convert_message_to_dict
# =============================================================================

class TestConvertMessageToDict:
    """Test convert_message_to_dict function"""

    def test_human_message(self):
        """Test converting HumanMessage"""
        msg = HumanMessage(content="Hello")
        result = convert_message_to_dict(msg)

        assert result == {"role": "user", "content": "Hello"}

    def test_system_message(self):
        """Test converting SystemMessage"""
        msg = SystemMessage(content="You are helpful")
        result = convert_message_to_dict(msg)

        assert result == {"role": "system", "content": "You are helpful"}

    def test_ai_message_simple(self):
        """Test converting simple AIMessage"""
        msg = AIMessage(content="Hi there!")
        result = convert_message_to_dict(msg)

        assert result == {"role": "assistant", "content": "Hi there!"}

    def test_ai_message_with_tool_calls_langchain_1_0(self):
        """Test AIMessage with tool_calls attribute (Langchain 1.0+)"""
        msg = AIMessage(content="")
        # Simulate Langchain 1.0 tool_calls attribute
        msg.tool_calls = [
            {
                "id": "call_abc123",
                "name": "get_weather",
                "args": {"location": "London", "unit": "celsius"},
                "type": "tool_call"
            }
        ]

        result = convert_message_to_dict(msg)

        assert result["role"] == "assistant"
        assert result["content"] is None  # Should be None, not empty string
        assert "tool_calls" in result
        assert len(result["tool_calls"]) == 1

        tool_call = result["tool_calls"][0]
        assert tool_call["id"] == "call_abc123"
        assert tool_call["type"] == "function"
        assert tool_call["function"]["name"] == "get_weather"
        assert "location" in tool_call["function"]["arguments"]

    def test_ai_message_with_tool_calls_pre_1_0(self):
        """Test AIMessage with tool_calls in additional_kwargs (pre-1.0)"""
        msg = AIMessage(
            content="",
            additional_kwargs={
                "tool_calls": [
                    {
                        "id": "call_xyz789",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": '{"query": "python"}'
                        }
                    }
                ]
            }
        )

        result = convert_message_to_dict(msg)

        assert result["role"] == "assistant"
        assert result["content"] is None
        assert "tool_calls" in result
        assert result["tool_calls"][0]["id"] == "call_xyz789"

    def test_ai_message_with_function_call(self):
        """Test AIMessage with legacy function_call"""
        msg = AIMessage(
            content="",
            additional_kwargs={
                "function_call": {
                    "name": "get_current_weather",
                    "arguments": '{"location": "Boston"}'
                }
            }
        )

        result = convert_message_to_dict(msg)

        assert result["role"] == "assistant"
        assert result["content"] is None
        assert "function_call" in result
        assert result["function_call"]["name"] == "get_current_weather"

    def test_function_message(self):
        """Test converting FunctionMessage"""
        msg = FunctionMessage(content='{"temp": 72}', name="get_weather")
        result = convert_message_to_dict(msg)

        assert result == {
            "role": "function",
            "content": '{"temp": 72}',
            "name": "get_weather"
        }

    def test_tool_message(self):
        """Test converting ToolMessage"""
        msg = ToolMessage(
            content='{"result": "success"}',
            tool_call_id="call_123"
        )
        result = convert_message_to_dict(msg)

        assert result["role"] == "tool"
        assert result["content"] == '{"result": "success"}'
        assert result["tool_call_id"] == "call_123"

    def test_chat_message(self):
        """Test converting ChatMessage"""
        msg = ChatMessage(content="Custom message", role="custom")
        result = convert_message_to_dict(msg)

        assert result == {"role": "custom", "content": "Custom message"}

    def test_message_with_name_in_additional_kwargs(self):
        """Test message with name in additional_kwargs"""
        msg = HumanMessage(
            content="Hello",
            additional_kwargs={"name": "John"}
        )
        result = convert_message_to_dict(msg)

        assert result["name"] == "John"


# =============================================================================
# Test _convert_message_chunk
# =============================================================================

class TestConvertMessageChunk:
    """Test _convert_message_chunk function"""

    def test_simple_content_chunk(self):
        """Test converting simple content chunk"""
        chunk = AIMessageChunk(content="Hello")
        result = _convert_message_chunk(chunk, 0)

        assert result["role"] == "assistant"
        assert result["content"] == "Hello"

    def test_content_chunk_not_first(self):
        """Test content chunk that's not first (no role)"""
        chunk = AIMessageChunk(content=" world")
        result = _convert_message_chunk(chunk, 1)

        assert "role" not in result
        assert result["content"] == " world"

    def test_empty_content_chunk(self):
        """Test empty content chunk returns empty dict"""
        chunk = AIMessageChunk(content="")
        result = _convert_message_chunk(chunk, 1)

        assert result == {}

    def test_tool_call_chunks_langchain_1_0_first_chunk(self):
        """Test tool_call_chunks (Langchain 1.0+) - first chunk with ID and name"""
        chunk = AIMessageChunk(content="")
        chunk.tool_call_chunks = [
            {
                "id": "call_abc123",
                "name": "get_weather",
                "args": "",
                "index": 0,
                "type": "tool_call_chunk"
            }
        ]

        result = _convert_message_chunk(chunk, 0)

        assert result["role"] == "assistant"
        assert result["content"] is None
        assert "tool_calls" in result
        assert len(result["tool_calls"]) == 1

        tool_call = result["tool_calls"][0]
        assert tool_call["id"] == "call_abc123"
        assert tool_call["type"] == "function"
        assert tool_call["function"]["name"] == "get_weather"
        assert tool_call["function"]["arguments"] == ""

    def test_tool_call_chunks_langchain_1_0_args_chunk(self):
        """Test tool_call_chunks (Langchain 1.0+) - subsequent chunk with args"""
        chunk = AIMessageChunk(content="")
        chunk.tool_call_chunks = [
            {
                "name": "",
                "args": '{"location": "',
                "index": 0,
                "type": "tool_call_chunk"
            }
        ]

        result = _convert_message_chunk(chunk, 1)

        assert "role" not in result
        assert "tool_calls" in result
        assert result["tool_calls"][0]["function"]["arguments"] == '{"location": "'

    def test_tool_call_chunks_multiple_tools(self):
        """Test multiple tool_call_chunks in one chunk"""
        chunk = AIMessageChunk(content="")
        chunk.tool_call_chunks = [
            {
                "id": "call_1",
                "name": "tool1",
                "args": "",
                "index": 0,
                "type": "tool_call_chunk"
            },
            {
                "id": "call_2",
                "name": "tool2",
                "args": "",
                "index": 1,
                "type": "tool_call_chunk"
            }
        ]

        result = _convert_message_chunk(chunk, 0)

        assert len(result["tool_calls"]) == 2
        assert result["tool_calls"][0]["id"] == "call_1"
        assert result["tool_calls"][1]["id"] == "call_2"

    def test_tool_calls_in_additional_kwargs_pre_1_0(self):
        """Test tool_calls in additional_kwargs (pre-1.0)"""
        chunk = AIMessageChunk(
            content="",
            additional_kwargs={
                "tool_calls": [
                    {
                        "id": "call_xyz",
                        "type": "function",
                        "function": {"name": "search", "arguments": "{}"}
                    }
                ]
            }
        )

        result = _convert_message_chunk(chunk, 0)

        assert result["content"] is None
        assert "tool_calls" in result

    def test_function_call_in_additional_kwargs(self):
        """Test function_call in additional_kwargs"""
        chunk = AIMessageChunk(
            content="",
            additional_kwargs={
                "function_call": {
                    "name": "get_weather",
                    "arguments": "{}"
                }
            }
        )

        result = _convert_message_chunk(chunk, 0)

        assert result["content"] is None
        assert "function_call" in result
        assert result["function_call"]["name"] == "get_weather"

    def test_invalid_chunk_type(self):
        """Test that non-AIMessageChunk raises ValueError"""
        chunk = HumanMessage(content="test")

        with pytest.raises(ValueError, match="unexpected streaming chunk type"):
            _convert_message_chunk(chunk, 0)


# =============================================================================
# Test convert_dict_to_message
# =============================================================================

class TestConvertDictToMessage:
    """Test convert_dict_to_message function"""

    def test_user_role(self):
        """Test converting user role dict"""
        msg_dict = {"role": "user", "content": "Hello"}
        result = convert_dict_to_message(msg_dict)

        assert isinstance(result, HumanMessage)
        assert result.content == "Hello"

    def test_assistant_role(self):
        """Test converting assistant role dict"""
        msg_dict = {"role": "assistant", "content": "Hi there"}
        result = convert_dict_to_message(msg_dict)

        assert isinstance(result, AIMessage)
        assert result.content == "Hi there"

    def test_assistant_with_tool_calls(self):
        """Test converting assistant with tool_calls"""
        msg_dict = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{}"}
                }
            ]
        }
        result = convert_dict_to_message(msg_dict)

        assert isinstance(result, AIMessage)
        assert "tool_calls" in result.additional_kwargs

    def test_system_role(self):
        """Test converting system role dict"""
        msg_dict = {"role": "system", "content": "Be helpful"}
        result = convert_dict_to_message(msg_dict)

        assert isinstance(result, SystemMessage)
        assert result.content == "Be helpful"

    def test_function_role(self):
        """Test converting function role dict"""
        msg_dict = {
            "role": "function",
            "content": '{"temp": 72}',
            "name": "get_weather"
        }
        result = convert_dict_to_message(msg_dict)

        assert isinstance(result, FunctionMessage)
        assert result.content == '{"temp": 72}'
        assert result.name == "get_weather"

    def test_tool_role(self):
        """Test converting tool role dict"""
        msg_dict = {
            "role": "tool",
            "content": "Success",
            "tool_call_id": "call_123"
        }
        result = convert_dict_to_message(msg_dict)

        assert isinstance(result, ToolMessage)
        assert result.content == "Success"
        assert result.tool_call_id == "call_123"

    def test_custom_role(self):
        """Test converting custom role dict"""
        msg_dict = {"role": "custom", "content": "Custom message"}
        result = convert_dict_to_message(msg_dict)

        assert isinstance(result, ChatMessage)
        assert result.role == "custom"
        assert result.content == "Custom message"


# =============================================================================
# Test helper functions
# =============================================================================

class TestHelperFunctions:
    """Test helper functions"""

    def test_convert_openai_messages(self):
        """Test converting list of OpenAI messages"""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"}
        ]
        result = convert_openai_messages(messages)

        assert len(result) == 2
        assert isinstance(result[0], HumanMessage)
        assert isinstance(result[1], AIMessage)

    def test_convert_message_chunk_to_delta(self):
        """Test _convert_message_chunk_to_delta"""
        chunk = AIMessageChunk(content="Hello")
        result = _convert_message_chunk_to_delta(chunk, 0)

        assert "choices" in result
        assert len(result["choices"]) == 1
        assert "delta" in result["choices"][0]
        assert result["choices"][0]["delta"]["content"] == "Hello"

    def test_has_assistant_message_true(self):
        """Test _has_assistant_message returns True"""
        session = {
            "messages": [
                HumanMessage(content="Hi"),
                AIMessage(content="Hello")
            ]
        }

        assert _has_assistant_message(session) is True

    def test_has_assistant_message_false(self):
        """Test _has_assistant_message returns False"""
        session = {
            "messages": [
                HumanMessage(content="Hi"),
                SystemMessage(content="Be helpful")
            ]
        }

        assert _has_assistant_message(session) is False

    def test_convert_messages_for_finetuning(self):
        """Test convert_messages_for_finetuning"""
        sessions = [
            {
                "messages": [
                    HumanMessage(content="Hi"),
                    AIMessage(content="Hello")
                ]
            },
            {
                "messages": [
                    HumanMessage(content="Question?")
                ]
            }
        ]

        result = convert_messages_for_finetuning(sessions)

        # Should only include session with assistant message
        assert len(result) == 1
        assert len(result[0]) == 2
        assert result[0][0]["role"] == "user"
        assert result[0][1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_aenumerate(self):
        """Test async enumerate function"""
        async def async_gen():
            for i in ['a', 'b', 'c']:
                yield i

        result = []
        async for idx, val in aenumerate(async_gen()):
            result.append((idx, val))

        assert result == [(0, 'a'), (1, 'b'), (2, 'c')]


# =============================================================================
# Test ChatCompletion class (integration tests with mocking)
# =============================================================================

class TestChatCompletion:
    """Test ChatCompletion class"""

    @patch('langchain_community.adapters.openai.importlib.import_module')
    def test_create_non_streaming(self, mock_import):
        """Test ChatCompletion.create without streaming"""
        # Mock the model
        mock_model_cls = Mock()
        mock_model_instance = Mock()
        mock_model_instance.invoke.return_value = AIMessage(content="Response")
        mock_model_cls.return_value = mock_model_instance

        mock_module = Mock()
        mock_module.ChatOpenAI = mock_model_cls
        mock_import.return_value = mock_module

        # Test
        messages = [{"role": "user", "content": "Hello"}]
        result = ChatCompletion.create(messages, provider="ChatOpenAI")

        assert "choices" in result
        assert result["choices"][0]["message"]["content"] == "Response"
        mock_model_instance.invoke.assert_called_once()

    @patch('langchain_community.adapters.openai.importlib.import_module')
    def test_create_streaming(self, mock_import):
        """Test ChatCompletion.create with streaming"""
        # Mock the model
        mock_model_cls = Mock()
        mock_model_instance = Mock()
        mock_model_instance.stream.return_value = [
            AIMessageChunk(content="Hello"),
            AIMessageChunk(content=" world")
        ]
        mock_model_cls.return_value = mock_model_instance

        mock_module = Mock()
        mock_module.ChatOpenAI = mock_model_cls
        mock_import.return_value = mock_module

        # Test
        messages = [{"role": "user", "content": "Hello"}]
        result = ChatCompletion.create(messages, provider="ChatOpenAI", stream=True)

        chunks = list(result)
        assert len(chunks) == 2
        assert chunks[0]["choices"][0]["delta"]["content"] == "Hello"
        assert chunks[1]["choices"][0]["delta"]["content"] == " world"

    @pytest.mark.asyncio
    @patch('langchain_community.adapters.openai.importlib.import_module')
    async def test_acreate_non_streaming(self, mock_import):
        """Test ChatCompletion.acreate without streaming"""
        # Mock the model
        mock_model_cls = Mock()
        mock_model_instance = Mock()
        mock_model_instance.ainvoke = AsyncMock(return_value=AIMessage(content="Async response"))
        mock_model_cls.return_value = mock_model_instance

        mock_module = Mock()
        mock_module.ChatOpenAI = mock_model_cls
        mock_import.return_value = mock_module

        # Test
        messages = [{"role": "user", "content": "Hello"}]
        result = await ChatCompletion.acreate(messages, provider="ChatOpenAI")

        assert "choices" in result
        assert result["choices"][0]["message"]["content"] == "Async response"


# =============================================================================
# Test Completions class
# =============================================================================

class TestCompletions:
    """Test Completions class"""

    @patch('langchain_community.adapters.openai.importlib.import_module')
    def test_completions_create_non_streaming(self, mock_import):
        """Test Completions.create without streaming"""
        # Mock the model
        mock_model_cls = Mock()
        mock_model_instance = Mock()
        mock_model_instance.invoke.return_value = AIMessage(content="Response")
        mock_model_cls.return_value = mock_model_instance

        mock_module = Mock()
        mock_module.ChatOpenAI = mock_model_cls
        mock_import.return_value = mock_module

        # Test
        messages = [{"role": "user", "content": "Hello"}]
        result = Completions.create(messages, provider="ChatOpenAI")

        assert hasattr(result, 'choices')
        assert result.choices[0].message["content"] == "Response"


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
