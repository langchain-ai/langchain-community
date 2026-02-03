import sys
import threading
import time
import uuid
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def mock_mermaid_trace_module() -> Generator[None, None, None]:
    """Fixture to mock mermaid_trace module for the duration of the test."""
    mock_mt = MagicMock()
    mock_core = MagicMock()
    mock_events = MagicMock()
    mock_context = MagicMock()
    mock_decorators = MagicMock()

    mock_mt.core = mock_core
    mock_core.events = mock_events
    mock_core.context = mock_context
    mock_core.decorators = mock_decorators

    # Mock LogContext.get to return default value
    mock_context.LogContext.get.side_effect = lambda k, d=None: d

    modules = {
        "mermaid_trace": mock_mt,
        "mermaid_trace.core": mock_core,
        "mermaid_trace.core.events": mock_events,
        "mermaid_trace.core.context": mock_context,
        "mermaid_trace.core.decorators": mock_decorators,
    }

    with patch.dict(sys.modules, modules):
        yield


from langchain_community.callbacks.mermaid_trace import (  # noqa: E402
    MermaidTraceCallbackHandler,
)


def test_concurrency_safety() -> None:
    """Test that multiple concurrent runs don't interfere with each other's stacks."""
    handler = MermaidTraceCallbackHandler()

    # Mock logger to avoid actual logging overhead during concurrency test
    handler.logger = MagicMock()

    def run_task(
        run_id: uuid.UUID, steps: int, results_dict: dict[uuid.UUID, str]
    ) -> None:
        """Simulate a task with multiple nested steps."""
        try:
            # Step 1: Start root chain
            handler.on_chain_start({"name": f"Root_{run_id}"}, {}, run_id=run_id)
            time.sleep(0.01)  # Yield for concurrency

            # Step 2: Nested calls
            for i in range(steps):
                child_id = uuid.uuid4()
                handler.on_chain_start(
                    {"name": f"Child_{run_id}_{i}"},
                    {},
                    run_id=child_id,
                    parent_run_id=run_id,
                )
                time.sleep(0.01)

                # Check source in middle of nested call
                current_source = handler._get_current_source(child_id)
                if current_source != f"Child_{run_id}_{i}":
                    results_dict[run_id] = (
                        f"Error: Expected Child_{run_id}_{i}, got {current_source}"
                    )
                    return

                handler.on_chain_end({}, run_id=child_id, parent_run_id=run_id)
                time.sleep(0.01)

            # Final check: root stack should have exactly 1 item (the root chain)
            stack = handler._get_participant_stack(run_id)
            if len(stack) != 1 or stack[0] != f"Root_{run_id}":
                results_dict[run_id] = (
                    f"Error: Stack corrupted. Expected [Root_{run_id}], got {stack}"
                )
                return

            handler.on_chain_end({}, run_id=run_id)

            # After end, stack should be empty
            if len(handler._get_participant_stack(run_id)) != 0:
                results_dict[run_id] = "Error: Stack not cleared after run_id end"
                return

            results_dict[run_id] = "Success"
        except Exception as e:
            results_dict[run_id] = f"Exception: {str(e)}"

    # Execute concurrent tasks
    num_threads = 10
    threads = []
    results: dict[uuid.UUID, str] = {}

    for _ in range(num_threads):
        u_id = uuid.uuid4()
        t = threading.Thread(target=run_task, args=(u_id, 3, results))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # Verify all tasks succeeded
    assert len(results) == num_threads
    for run_id, status in results.items():
        assert status == "Success", f"Task {run_id} failed: {status}"
