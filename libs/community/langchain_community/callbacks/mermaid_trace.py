"""MermaidTrace callback handler."""

import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.utils import guard_import

if TYPE_CHECKING:
    from langchain_core.agents import AgentAction, AgentFinish
    from langchain_core.documents import Document
    from langchain_core.outputs import LLMResult


def import_mermaid_trace() -> Any:
    """Import the mermaid-trace python package.

    Raises:
        ImportError: If the mermaid-trace package is not installed.
    """
    return guard_import("mermaid_trace")


class MermaidTraceCallbackHandler(BaseCallbackHandler):
    """Callback Handler that records execution flow as Mermaid sequence diagrams.

    This handler intercepts LangChain events (Chain, LLM, Tool, Agent) and logs them as
    FlowEvents, which are then processed by MermaidTrace to generate diagrams.
    """

    def __init__(self, host_name: str = "LangChain"):
        """Initialize the callback handler.

        Args:
            host_name: The name of the host participant in the diagram.
                Defaults to "LangChain".
        """
        mt = import_mermaid_trace()
        self.host_name = host_name
        self.logger = mt.core.decorators.get_flow_logger()
        self._participant_stack: List[str] = []

    def _get_current_source(self) -> str:
        """Get the current source participant from stack or context."""
        mt = import_mermaid_trace()
        if self._participant_stack:
            return self._participant_stack[-1]
        source = mt.core.context.LogContext.get("current_participant", self.host_name)
        return str(source)

    def on_chain_start(
        self,
        serialized: Optional[Dict[str, Any]],
        inputs: Dict[str, Any],
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when chain starts running."""
        mt = import_mermaid_trace()
        target = (
            (serialized.get("name") if serialized else None)
            or kwargs.get("name")
            or "Chain"
        )
        source = self._get_current_source()

        event = mt.core.events.FlowEvent(
            source=source,
            target=target,
            action="Run Chain",
            message=f"Start Chain: {target}",
            trace_id=mt.core.context.LogContext.get("trace_id", str(uuid.uuid4())),
            params=str(inputs),
        )
        self.logger.info(
            f"{source} -> {target}: {event.action}", extra={"flow_event": event}
        )
        self._participant_stack.append(target)

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run when chain ends running."""
        mt = import_mermaid_trace()
        if not self._participant_stack:
            return

        target = self._participant_stack.pop()
        source = self._get_current_source()

        event = mt.core.events.FlowEvent(
            source=target,
            target=source,
            action="Return",
            message=f"End Chain: {target}",
            trace_id=mt.core.context.LogContext.get("trace_id", str(uuid.uuid4())),
            params=str(outputs),
        )
        self.logger.info(
            f"{target} -> {source}: {event.action}", extra={"flow_event": event}
        )

    def on_llm_start(
        self,
        serialized: Optional[Dict[str, Any]],
        prompts: List[str],
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when LLM starts running."""
        mt = import_mermaid_trace()
        target = (
            (serialized.get("name") if serialized else None)
            or kwargs.get("name")
            or "LLM"
        )
        source = self._get_current_source()

        event = mt.core.events.FlowEvent(
            source=source,
            target=target,
            action="Query LLM",
            message=f"Start LLM: {target}",
            trace_id=mt.core.context.LogContext.get("trace_id", str(uuid.uuid4())),
            params=str(prompts),
        )
        self.logger.info(
            f"{source} -> {target}: {event.action}", extra={"flow_event": event}
        )
        self._participant_stack.append(target)

    def on_llm_end(
        self,
        response: "LLMResult",
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run when LLM ends running."""
        mt = import_mermaid_trace()
        if not self._participant_stack:
            return

        target = self._participant_stack.pop()
        source = self._get_current_source()

        event = mt.core.events.FlowEvent(
            source=target,
            target=source,
            action="Response",
            message=f"End LLM: {target}",
            trace_id=mt.core.context.LogContext.get("trace_id", str(uuid.uuid4())),
            params=str(response),
        )
        self.logger.info(
            f"{target} -> {source}: {event.action}", extra={"flow_event": event}
        )

    def on_tool_start(
        self,
        serialized: Optional[Dict[str, Any]],
        input_str: str,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when tool starts running."""
        mt = import_mermaid_trace()
        target = (
            (serialized.get("name") if serialized else None)
            or kwargs.get("name")
            or "Tool"
        )
        source = self._get_current_source()

        event = mt.core.events.FlowEvent(
            source=source,
            target=target,
            action="Call Tool",
            message=f"Start Tool: {target}",
            trace_id=mt.core.context.LogContext.get("trace_id", str(uuid.uuid4())),
            params=str(input_str),
        )
        self.logger.info(
            f"{source} -> {target}: {event.action}", extra={"flow_event": event}
        )
        self._participant_stack.append(target)

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run when tool ends running."""
        mt = import_mermaid_trace()
        if not self._participant_stack:
            return

        target = self._participant_stack.pop()
        source = self._get_current_source()

        event = mt.core.events.FlowEvent(
            source=target,
            target=source,
            action="Tool Result",
            message=f"End Tool: {target}",
            trace_id=mt.core.context.LogContext.get("trace_id", str(uuid.uuid4())),
            params=str(output),
        )
        self.logger.info(
            f"{target} -> {source}: {event.action}", extra={"flow_event": event}
        )

    def on_retriever_start(
        self,
        serialized: Optional[Dict[str, Any]],
        query: str,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when retriever starts running."""
        mt = import_mermaid_trace()
        target = (
            (serialized.get("name") if serialized else None)
            or kwargs.get("name")
            or "Retriever"
        )
        source = self._get_current_source()

        event = mt.core.events.FlowEvent(
            source=source,
            target=target,
            action="Retrieve",
            message=f"Start Retrieval: {target}",
            trace_id=mt.core.context.LogContext.get("trace_id", str(uuid.uuid4())),
            params=query,
        )
        self.logger.info(
            f"{source} -> {target}: {event.action}", extra={"flow_event": event}
        )
        self._participant_stack.append(target)

    def on_retriever_end(
        self,
        documents: Sequence["Document"],
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when retriever ends running."""
        mt = import_mermaid_trace()
        if not self._participant_stack:
            return

        target = self._participant_stack.pop()
        source = self._get_current_source()

        event = mt.core.events.FlowEvent(
            source=target,
            target=source,
            action="Docs",
            message=f"End Retrieval: {target}",
            trace_id=mt.core.context.LogContext.get("trace_id", str(uuid.uuid4())),
            params=str(documents),
        )
        self.logger.info(
            f"{target} -> {source}: {event.action}", extra={"flow_event": event}
        )

    def on_agent_action(
        self,
        action: "AgentAction",
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run on agent action."""
        mt = import_mermaid_trace()
        target = f"Agent:{action.tool}"
        source = self._get_current_source()

        event = mt.core.events.FlowEvent(
            source=source,
            target=target,
            action="Action",
            message=f"Agent Action: {action.tool}",
            trace_id=mt.core.context.LogContext.get("trace_id", str(uuid.uuid4())),
            params=str(action.tool_input),
        )
        self.logger.info(
            f"{source} -> {target}: {event.action}", extra={"flow_event": event}
        )

    def on_agent_finish(
        self,
        finish: "AgentFinish",
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run on agent finish."""
        mt = import_mermaid_trace()
        target = "Agent"
        source = self.host_name

        event = mt.core.events.FlowEvent(
            source=target,
            target=source,
            action="Finish",
            message="Agent Finished",
            trace_id=mt.core.context.LogContext.get("trace_id", str(uuid.uuid4())),
            params=str(finish.return_values),
        )
        self.logger.info(
            f"{target} -> {source}: {event.action}", extra={"flow_event": event}
        )
