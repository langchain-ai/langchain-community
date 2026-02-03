import uuid
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.outputs import LLMResult

from langchain_community.callbacks.mermaid_trace import MermaidTraceCallbackHandler


@pytest.fixture
def mock_mt_components() -> Any:
    """Fixture to mock mermaid_trace components."""
    with patch(
        "langchain_community.callbacks.mermaid_trace.import_mermaid_trace"
    ) as mock_import:
        mock_mt = MagicMock()
        mock_import.return_value = mock_mt

        mock_logger = MagicMock()
        mock_mt.core.decorators.get_flow_logger.return_value = mock_logger

        # Mock LogContext.get
        mock_mt.core.context.LogContext.get.side_effect = lambda k, d=None: d

        yield mock_mt, mock_logger


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
    mock_mt.core.events.FlowEvent.assert_called_once()

    # Verify logger.info was called
    assert mock_logger.info.called


def test_on_llm_end(mock_mt_components: Any) -> None:
    mock_mt, mock_logger = mock_mt_components
    handler = MermaidTraceCallbackHandler(host_name="TestHost")

    run_id = uuid.uuid4()
    handler.on_llm_start({"name": "TestLLM"}, ["prompt"], run_id=run_id)

    response = LLMResult(generations=[[]])
    handler.on_llm_end(response, run_id=run_id)

    # Verify logger.info was called at least twice
    assert mock_logger.info.call_count >= 2


def test_on_tool_lifecycle(mock_mt_components: Any) -> None:
    mock_mt, mock_logger = mock_mt_components
    handler = MermaidTraceCallbackHandler(host_name="TestHost")

    run_id = uuid.uuid4()
    handler.on_tool_start({"name": "TestTool"}, "input", run_id=run_id)
    handler.on_tool_end("output", run_id=run_id)

    assert mock_logger.info.call_count == 2
    mock_mt.core.events.FlowEvent.assert_called()


def test_source_participant_on_end(mock_mt_components: Any) -> None:
    """Test that the source participant is correct when a run ends."""
    mock_mt, _ = mock_mt_components
    handler = MermaidTraceCallbackHandler(host_name="TestHost")

    root_run_id = uuid.uuid4()
    child_run_id = uuid.uuid4()

    # Start root chain
    handler.on_chain_start({"name": "RootChain"}, {}, run_id=root_run_id)
    # Start child chain
    handler.on_chain_start(
        {"name": "ChildChain"}, {}, run_id=child_run_id, parent_run_id=root_run_id
    )

    # Reset mock to capture only the next call
    mock_mt.core.events.FlowEvent.reset_mock()

    # End child chain
    handler.on_chain_end({}, run_id=child_run_id)

    # The child chain (target "ChildChain") is ending.
    # Its source (parent) should be "RootChain".
    # FlowEvent(source="ChildChain", target="RootChain", ...)
    call_args = mock_mt.core.events.FlowEvent.call_args[1]
    assert call_args["source"] == "ChildChain"
    assert call_args["target"] == "RootChain"


def test_on_chain_error(mock_mt_components: Any) -> None:
    mock_mt, mock_logger = mock_mt_components
    handler = MermaidTraceCallbackHandler(host_name="TestHost")

    run_id = uuid.uuid4()
    handler.on_chain_start({"name": "TestChain"}, {}, run_id=run_id)

    error = ValueError("test error")
    handler.on_chain_error(error, run_id=run_id)

    # Verify stack is empty
    assert len(handler._get_participant_stack(run_id)) == 0
    assert mock_logger.info.call_count == 2
    mock_mt.core.events.FlowEvent.assert_called()


def test_on_retriever_lifecycle(mock_mt_components: Any) -> None:
    mock_mt, mock_logger = mock_mt_components
    handler = MermaidTraceCallbackHandler(host_name="TestHost")

    run_id = uuid.uuid4()
    handler.on_retriever_start({"name": "TestRetriever"}, "query", run_id=run_id)
    handler.on_retriever_end([], run_id=run_id)

    assert mock_logger.info.call_count == 2
    mock_mt.core.events.FlowEvent.assert_called()


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
    mock_mt.core.events.FlowEvent.assert_called()
