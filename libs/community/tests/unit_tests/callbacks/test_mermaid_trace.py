import sys
import uuid
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.outputs import LLMResult

# Mock mermaid_trace and its submodules before importing the handler
mock_mermaid_trace = MagicMock()
mock_core = MagicMock()
mock_events = MagicMock()
mock_context = MagicMock()
mock_decorators = MagicMock()

sys.modules["mermaid_trace"] = mock_mermaid_trace
sys.modules["mermaid_trace.core"] = mock_core
sys.modules["mermaid_trace.core.events"] = mock_events
sys.modules["mermaid_trace.core.context"] = mock_context
sys.modules["mermaid_trace.core.decorators"] = mock_decorators

# Set up mock relationships
mock_mermaid_trace.core = mock_core
mock_core.events = mock_events
mock_core.context = mock_context
mock_core.decorators = mock_decorators

from langchain_community.callbacks.mermaid_trace import (  # noqa: E402
    MermaidTraceCallbackHandler,
)


@pytest.fixture
def mock_mt_components() -> Any:
    mock_logger = MagicMock()
    mock_decorators.get_flow_logger.return_value = mock_logger

    # Mock LogContext.get
    mock_context.LogContext.get.side_effect = lambda k, d=None: d

    yield mock_mermaid_trace, mock_logger


def test_mermaid_trace_callback_init(mock_mt_components: Any) -> None:
    _, _ = mock_mt_components
    handler = MermaidTraceCallbackHandler(host_name="TestHost")
    assert handler.host_name == "TestHost"


def test_on_chain_start(mock_mt_components: Any) -> None:
    mock_mt, mock_logger = mock_mt_components
    handler = MermaidTraceCallbackHandler(host_name="TestHost")

    inputs = {"input": "hello"}
    serialized = {"name": "TestChain"}

    handler.on_chain_start(serialized, inputs, run_id=uuid.uuid4())

    # Verify FlowEvent was created
    mock_events.FlowEvent.assert_called_once()

    # Verify logger.info was called
    assert mock_logger.info.called


def test_on_llm_end(mock_mt_components: Any) -> None:
    mock_mt, mock_logger = mock_mt_components
    handler = MermaidTraceCallbackHandler(host_name="TestHost")

    run_id = uuid.uuid4()
    handler.on_chain_start({"name": "TestChain"}, {}, run_id=run_id)

    response = LLMResult(generations=[[]])
    handler.on_llm_end(response, run_id=run_id)

    # Verify logger.info was called at least twice (once for start, once for end)
    assert mock_logger.info.call_count >= 2


def test_on_tool_lifecycle(mock_mt_components: Any) -> None:
    mock_mt, mock_logger = mock_mt_components
    handler = MermaidTraceCallbackHandler(host_name="TestHost")

    run_id = uuid.uuid4()
    handler.on_tool_start({"name": "TestTool"}, "input", run_id=run_id)
    handler.on_tool_end("output", run_id=run_id)

    assert mock_logger.info.call_count == 2
    mock_events.FlowEvent.assert_called()


def test_on_chain_error(mock_mt_components: Any) -> None:
    mock_mt, mock_logger = mock_mt_components
    handler = MermaidTraceCallbackHandler(host_name="TestHost")

    run_id = uuid.uuid4()
    handler.on_chain_start({"name": "TestChain"}, {}, run_id=run_id)

    error = ValueError("test error")
    handler.on_chain_error(error, run_id=run_id)

    # Verify stack is empty
    assert len(handler._participant_stack) == 0
    assert mock_logger.info.call_count == 2
    mock_events.FlowEvent.assert_called()


def test_on_retriever_lifecycle(mock_mt_components: Any) -> None:
    mock_mt, mock_logger = mock_mt_components
    handler = MermaidTraceCallbackHandler(host_name="TestHost")

    run_id = uuid.uuid4()
    handler.on_retriever_start({"name": "TestRetriever"}, "query", run_id=run_id)
    handler.on_retriever_end([], run_id=run_id)

    assert mock_logger.info.call_count == 2
    mock_events.FlowEvent.assert_called()


def test_on_agent_lifecycle(mock_mt_components: Any) -> None:
    mock_mt, mock_logger = mock_mt_components
    handler = MermaidTraceCallbackHandler(host_name="TestHost")

    mock_action = MagicMock()
    mock_action.tool = "search"
    mock_action.tool_input = "query"

    mock_finish = MagicMock()
    mock_finish.return_values = {"output": "result"}

    run_id = uuid.uuid4()
    handler.on_agent_action(mock_action, run_id=run_id)
    handler.on_agent_finish(mock_finish, run_id=run_id)

    assert mock_logger.info.call_count == 2
    mock_events.FlowEvent.assert_called()
