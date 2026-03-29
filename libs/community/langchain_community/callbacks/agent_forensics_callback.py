"""Agent Forensics callback handler for LangChain.

Records all agent activity to an SQLite event store for post-incident
forensic analysis, failure classification, and EU AI Act compliance.

Setup:
    pip install agent-forensics

Usage:
    from langchain_community.callbacks import AgentForensicsCallbackHandler

    handler = AgentForensicsCallbackHandler(session_id="order-123")
    agent.invoke({"input": "..."}, config={"callbacks": [handler]})

    print(handler.report())         # Forensic report
    print(handler.classify())       # Auto-detect failure patterns
    handler.save_markdown(".")      # Save report to file
"""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage


class AgentForensicsCallbackHandler(BaseCallbackHandler):
    """Callback handler that records agent actions for forensic analysis.

    Captures decisions, tool calls, LLM interactions, errors, and prompt drift.
    Stores events in an SQLite database for later analysis and reporting.

    Args:
        session_id: Unique identifier for this agent session.
        agent_id: Name of the agent being recorded.
        db_path: Path to the SQLite database file.

    Example:
        .. code-block:: python

            from langchain_community.callbacks import AgentForensicsCallbackHandler

            handler = AgentForensicsCallbackHandler(
                session_id="order-123",
                agent_id="shopping-agent",
            )

            agent.invoke(
                {"input": "Buy a wireless mouse"},
                config={"callbacks": [handler]},
            )

            # Generate report
            print(handler.report())

            # Auto-classify failures
            for f in handler.classify():
                print(f"[{f['severity']}] {f['type']}: {f['description']}")
    """

    def __init__(
        self,
        session_id: str = "default",
        agent_id: str = "default-agent",
        db_path: str = "forensics.db",
    ) -> None:
        try:
            from agent_forensics import Forensics
        except ImportError:
            raise ImportError(
                "agent-forensics is required for AgentForensicsCallbackHandler. "
                "Install it with: pip install agent-forensics"
            )
        self._forensics = Forensics(session=session_id, agent=agent_id, db_path=db_path)
        self._collector = self._forensics.langchain()

    # -- Delegate to the agent-forensics collector --

    def on_chat_model_start(self, serialized: dict, messages: list, **kwargs: Any) -> None:
        self._collector.on_chat_model_start(serialized, messages, **kwargs)

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        self._collector.on_llm_end(response, **kwargs)

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        self._collector.on_llm_error(error, **kwargs)

    def on_tool_start(self, serialized: dict, input_str: str, **kwargs: Any) -> None:
        self._collector.on_tool_start(serialized, input_str, **kwargs)

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        self._collector.on_tool_end(output, **kwargs)

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        self._collector.on_tool_error(error, **kwargs)

    # -- Forensics API --

    def report(self) -> str:
        """Generate a Markdown forensic report."""
        return self._forensics.report()

    def classify(self) -> list[dict]:
        """Auto-classify failure patterns in the session trace."""
        return self._forensics.classify()

    def save_markdown(self, path: str = ".") -> str:
        """Save the report as a Markdown file."""
        return self._forensics.save_markdown(path)

    def save_pdf(self, path: str = ".") -> str:
        """Save the report as a PDF file."""
        return self._forensics.save_pdf(path)

    def events(self) -> list:
        """Return all recorded events."""
        return self._forensics.events()
