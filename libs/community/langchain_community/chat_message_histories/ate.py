"""Chat message history backed by FoodForThought's ate CLI (.mv2 format)."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from typing import List, Optional, Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict

logger = logging.getLogger(__name__)


class AteChatMessageHistory(BaseChatMessageHistory):
    """Chat message history that stores messages in a .mv2 memory file
    via the ``ate`` CLI from FoodForThought.

    Unlike :class:`FileChatMessageHistory`, ate-backed history supports:

    * **Semantic search** over past messages
    * **Tagged memories** with arbitrary metadata
    * **AES-256 encryption** at rest
    * **Multi-agent concurrent access** via file-level locking

    Each message is stored as a memory entry whose ``text`` field holds the
    LangChain-serialised message dict and whose ``tags`` encode the
    ``session_id`` and message type (``human``, ``ai``, ``system``, …).

    Parameters
    ----------
    session_id:
        Unique identifier for this conversation session.  Used to filter
        memories so that multiple sessions can share the same .mv2 file.
    memory_path:
        Filesystem path to the ``.mv2`` memory file.
    auto_init:
        When *True* (the default) the constructor will run
        ``ate memory init`` if the *memory_path* does not yet exist.

    Raises
    ------
    ImportError
        If the ``ate`` CLI is not found on ``$PATH``.
    RuntimeError
        If ``ate memory init`` fails during auto-initialisation.

    Example
    -------
    .. code-block:: python

        from langchain_community.chat_message_histories import AteChatMessageHistory

        history = AteChatMessageHistory(
            session_id="user-42",
            memory_path="./chat.mv2",
        )
        history.add_user_message("Hello!")
        history.add_ai_message("Hi there!")
        print(history.messages)
    """

    def __init__(
        self,
        session_id: str,
        memory_path: str,
        *,
        auto_init: bool = True,
    ) -> None:
        if not shutil.which("ate"):
            raise ImportError(
                "Could not find the `ate` CLI on $PATH.  "
                "Install it with `pip install ate-memory` or "
                "`brew install ate-cli`.  "
                "See https://kindly.fyi/foodforthought for details."
            )
        if not session_id:
            raise ValueError("session_id must be a non-empty string.")
        if not memory_path:
            raise ValueError("memory_path must be a non-empty string.")

        self.session_id = session_id
        self.memory_path = memory_path

        if auto_init:
            self._ensure_initialised()

    # ------------------------------------------------------------------
    # BaseChatMessageHistory interface
    # ------------------------------------------------------------------

    @property
    def messages(self) -> List[BaseMessage]:  # type: ignore[override]
        """Return all messages for this session from the .mv2 file."""
        raw = self._run_ate(
            ["ate", "memory", "export", "--path", self.memory_path, "--format", "json"]
        )
        if not raw.strip():
            return []

        try:
            entries: list = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Failed to parse ate export output as JSON.")
            return []

        if not isinstance(entries, list):
            logger.warning("ate export did not return a JSON array.")
            return []

        session_tag = f"session:{self.session_id}"
        filtered: list[dict] = []
        for entry in entries:
            tags: Sequence[str] = entry.get("tags", [])
            if session_tag in tags:
                try:
                    msg_dict = json.loads(entry.get("text", "{}"))
                    filtered.append(msg_dict)
                except json.JSONDecodeError:
                    logger.warning(
                        "Skipping memory entry with non-JSON text: %s",
                        entry.get("id", "<unknown>"),
                    )

        if not filtered:
            return []

        return messages_from_dict(filtered)

    def add_message(self, message: BaseMessage) -> None:
        """Persist a single message to the .mv2 file."""
        serialised = json.dumps(message_to_dict(message))
        msg_type = message.type  # "human", "ai", "system", etc.
        tags = f"session:{self.session_id},type:{msg_type}"

        self._run_ate(
            [
                "ate",
                "memory",
                "add",
                "--path",
                self.memory_path,
                "--text",
                serialised,
                "--tags",
                tags,
                "--format",
                "json",
            ]
        )

    def clear(self) -> None:
        """Re-initialise the .mv2 file, destroying **all** sessions."""
        self._run_ate(
            ["ate", "memory", "init", "--path", self.memory_path]
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ensure_initialised(self) -> None:
        """Run ``ate memory init`` if the memory file does not exist."""
        import os

        if not os.path.exists(self.memory_path):
            result = self._run_ate(
                ["ate", "memory", "init", "--path", self.memory_path],
                check=True,
            )
            if result is None:
                raise RuntimeError(
                    f"Failed to initialise ate memory at {self.memory_path}"
                )

    @staticmethod
    def _run_ate(
        cmd: list[str],
        *,
        check: bool = False,
    ) -> str:
        """Execute an ate CLI command and return its stdout.

        Parameters
        ----------
        cmd:
            Full command list, e.g. ``["ate", "memory", "export", ...]``.
        check:
            If *True*, raise :class:`RuntimeError` on non-zero exit.

        Returns
        -------
        str
            The stdout of the process (stripped).
        """
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"ate CLI timed out after 30 s: {' '.join(cmd)}"
            ) from exc
        except FileNotFoundError as exc:
            raise RuntimeError(
                "ate CLI not found when attempting to run command."
            ) from exc

        if check and proc.returncode != 0:
            raise RuntimeError(
                f"ate CLI returned non-zero exit code {proc.returncode}: "
                f"{proc.stderr.strip()}"
            )

        if proc.returncode != 0:
            logger.warning(
                "ate CLI returned exit code %d: %s",
                proc.returncode,
                proc.stderr.strip(),
            )

        return proc.stdout.strip()
