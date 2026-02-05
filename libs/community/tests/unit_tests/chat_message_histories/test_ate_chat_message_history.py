"""Unit tests for AteChatMessageHistory.

All subprocess and shutil.which calls are fully mocked — no real CLI
invocations happen during testing.
"""

from __future__ import annotations

import json
import subprocess
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    message_to_dict,
)

from langchain_community.chat_message_histories.ate import AteChatMessageHistory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_export_json(entries: list[dict[str, Any]]) -> str:
    """Build the JSON string that ``ate memory export`` would return."""
    return json.dumps(entries)


def _make_memory_entry(
    message: Any,
    session_id: str,
    entry_id: str = "mem-1",
) -> dict[str, Any]:
    """Build a single memory entry dict as returned by ate export."""
    msg_dict = message_to_dict(message)
    return {
        "id": entry_id,
        "text": json.dumps(msg_dict),
        "tags": [f"session:{session_id}", f"type:{message.type}"],
    }


def _completed_process(stdout: str = "", returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=""
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_which():
    """Patch shutil.which so it always finds 'ate'."""
    with patch("langchain_community.chat_message_histories.ate.shutil.which", return_value="/usr/local/bin/ate"):
        yield


@pytest.fixture()
def mock_subprocess(mock_which):
    """Patch subprocess.run to return an empty success by default."""
    with patch("langchain_community.chat_message_histories.ate.subprocess.run", return_value=_completed_process()) as m:
        yield m


@pytest.fixture()
def mock_no_file(mock_which):
    """Patch os.path.exists to return False (file doesn't exist yet)."""
    with patch("os.path.exists", return_value=False):
        with patch("langchain_community.chat_message_histories.ate.subprocess.run", return_value=_completed_process()) as m:
            yield m


@pytest.fixture()
def mock_file_exists(mock_which):
    """Patch os.path.exists to return True."""
    with patch("os.path.exists", return_value=True):
        with patch("langchain_community.chat_message_histories.ate.subprocess.run", return_value=_completed_process()) as m:
            yield m


@pytest.fixture()
def history(mock_file_exists):
    """Return a ready-to-use history instance (file already exists)."""
    return AteChatMessageHistory(
        session_id="test-session",
        memory_path="/tmp/test.mv2",
    )


# ===========================================================================
# 1. Constructor tests
# ===========================================================================

class TestConstructor:
    def test_raises_when_ate_not_found(self):
        with patch("langchain_community.chat_message_histories.ate.shutil.which", return_value=None):
            with pytest.raises(ImportError, match="Could not find the `ate` CLI"):
                AteChatMessageHistory(session_id="s", memory_path="/tmp/test.mv2")

    def test_raises_on_empty_session_id(self, mock_which):
        with patch("os.path.exists", return_value=True):
            with patch("langchain_community.chat_message_histories.ate.subprocess.run", return_value=_completed_process()):
                with pytest.raises(ValueError, match="session_id"):
                    AteChatMessageHistory(session_id="", memory_path="/tmp/x.mv2")

    def test_raises_on_empty_memory_path(self, mock_which):
        with patch("os.path.exists", return_value=True):
            with patch("langchain_community.chat_message_histories.ate.subprocess.run", return_value=_completed_process()):
                with pytest.raises(ValueError, match="memory_path"):
                    AteChatMessageHistory(session_id="s", memory_path="")

    def test_auto_init_creates_file(self, mock_no_file):
        h = AteChatMessageHistory(session_id="s", memory_path="/tmp/new.mv2")
        # Should have called ate memory init
        mock_no_file.assert_called_once()
        cmd = mock_no_file.call_args[0][0]
        assert cmd == ["ate", "memory", "init", "--path", "/tmp/new.mv2"]

    def test_auto_init_skips_when_file_exists(self, mock_file_exists):
        h = AteChatMessageHistory(session_id="s", memory_path="/tmp/existing.mv2")
        # subprocess.run should NOT have been called (no init needed)
        mock_file_exists.assert_not_called()

    def test_auto_init_false_skips_init(self, mock_which):
        with patch("langchain_community.chat_message_histories.ate.subprocess.run") as mock_run:
            h = AteChatMessageHistory(
                session_id="s", memory_path="/tmp/x.mv2", auto_init=False
            )
            mock_run.assert_not_called()

    def test_auto_init_failure_raises_runtime_error(self, mock_which):
        with patch("os.path.exists", return_value=False):
            with patch(
                "langchain_community.chat_message_histories.ate.subprocess.run",
                return_value=_completed_process(returncode=1),
            ):
                with pytest.raises(RuntimeError, match="non-zero exit code"):
                    AteChatMessageHistory(session_id="s", memory_path="/tmp/x.mv2")

    def test_stores_session_id_and_path(self, mock_file_exists):
        h = AteChatMessageHistory(session_id="abc", memory_path="/data/chat.mv2")
        assert h.session_id == "abc"
        assert h.memory_path == "/data/chat.mv2"


# ===========================================================================
# 2. messages property tests
# ===========================================================================

class TestMessagesProperty:
    def test_empty_export(self, history, mock_file_exists):
        mock_file_exists.return_value = _completed_process(stdout="")
        assert history.messages == []

    def test_empty_array_export(self, history, mock_file_exists):
        mock_file_exists.return_value = _completed_process(stdout="[]")
        assert history.messages == []

    def test_returns_human_message(self, history, mock_file_exists):
        entry = _make_memory_entry(HumanMessage(content="hi"), "test-session")
        mock_file_exists.return_value = _completed_process(
            stdout=_make_export_json([entry])
        )
        msgs = history.messages
        assert len(msgs) == 1
        assert isinstance(msgs[0], HumanMessage)
        assert msgs[0].content == "hi"

    def test_returns_ai_message(self, history, mock_file_exists):
        entry = _make_memory_entry(AIMessage(content="hello"), "test-session")
        mock_file_exists.return_value = _completed_process(
            stdout=_make_export_json([entry])
        )
        msgs = history.messages
        assert len(msgs) == 1
        assert isinstance(msgs[0], AIMessage)

    def test_returns_system_message(self, history, mock_file_exists):
        entry = _make_memory_entry(SystemMessage(content="sys"), "test-session")
        mock_file_exists.return_value = _completed_process(
            stdout=_make_export_json([entry])
        )
        msgs = history.messages
        assert len(msgs) == 1
        assert isinstance(msgs[0], SystemMessage)

    def test_filters_by_session_id(self, history, mock_file_exists):
        mine = _make_memory_entry(HumanMessage(content="mine"), "test-session", "m1")
        other = _make_memory_entry(HumanMessage(content="other"), "other-session", "m2")
        mock_file_exists.return_value = _completed_process(
            stdout=_make_export_json([mine, other])
        )
        msgs = history.messages
        assert len(msgs) == 1
        assert msgs[0].content == "mine"

    def test_multiple_messages_preserve_order(self, history, mock_file_exists):
        entries = [
            _make_memory_entry(HumanMessage(content="first"), "test-session", "m1"),
            _make_memory_entry(AIMessage(content="second"), "test-session", "m2"),
            _make_memory_entry(HumanMessage(content="third"), "test-session", "m3"),
        ]
        mock_file_exists.return_value = _completed_process(
            stdout=_make_export_json(entries)
        )
        msgs = history.messages
        assert len(msgs) == 3
        assert msgs[0].content == "first"
        assert msgs[1].content == "second"
        assert msgs[2].content == "third"

    def test_invalid_json_returns_empty(self, history, mock_file_exists):
        mock_file_exists.return_value = _completed_process(stdout="NOT JSON")
        assert history.messages == []

    def test_non_list_json_returns_empty(self, history, mock_file_exists):
        mock_file_exists.return_value = _completed_process(stdout='{"key": "value"}')
        assert history.messages == []

    def test_entry_with_bad_text_skipped(self, history, mock_file_exists):
        good = _make_memory_entry(HumanMessage(content="good"), "test-session", "m1")
        bad = {
            "id": "m2",
            "text": "NOT-VALID-JSON",
            "tags": ["session:test-session", "type:human"],
        }
        mock_file_exists.return_value = _completed_process(
            stdout=_make_export_json([good, bad])
        )
        msgs = history.messages
        assert len(msgs) == 1
        assert msgs[0].content == "good"

    def test_export_calls_correct_command(self, history, mock_file_exists):
        mock_file_exists.return_value = _completed_process(stdout="[]")
        _ = history.messages
        cmd = mock_file_exists.call_args[0][0]
        assert cmd == [
            "ate", "memory", "export",
            "--path", "/tmp/test.mv2",
            "--format", "json",
        ]


# ===========================================================================
# 3. add_message tests
# ===========================================================================

class TestAddMessage:
    def test_add_human_message(self, history, mock_file_exists):
        msg = HumanMessage(content="hello")
        history.add_message(msg)
        mock_file_exists.assert_called_once()
        cmd = mock_file_exists.call_args[0][0]
        assert cmd[0:3] == ["ate", "memory", "add"]
        assert "--path" in cmd
        assert "--text" in cmd
        assert "--tags" in cmd

    def test_add_ai_message(self, history, mock_file_exists):
        msg = AIMessage(content="reply")
        history.add_message(msg)
        cmd = mock_file_exists.call_args[0][0]
        tags_idx = cmd.index("--tags") + 1
        assert "type:ai" in cmd[tags_idx]

    def test_add_system_message(self, history, mock_file_exists):
        msg = SystemMessage(content="sys prompt")
        history.add_message(msg)
        cmd = mock_file_exists.call_args[0][0]
        tags_idx = cmd.index("--tags") + 1
        assert "type:system" in cmd[tags_idx]

    def test_tags_contain_session_id(self, history, mock_file_exists):
        history.add_message(HumanMessage(content="x"))
        cmd = mock_file_exists.call_args[0][0]
        tags_idx = cmd.index("--tags") + 1
        assert "session:test-session" in cmd[tags_idx]

    def test_text_is_valid_json(self, history, mock_file_exists):
        msg = HumanMessage(content="check json")
        history.add_message(msg)
        cmd = mock_file_exists.call_args[0][0]
        text_idx = cmd.index("--text") + 1
        parsed = json.loads(cmd[text_idx])
        assert parsed["data"]["content"] == "check json"

    def test_add_message_includes_format_json(self, history, mock_file_exists):
        history.add_message(HumanMessage(content="x"))
        cmd = mock_file_exists.call_args[0][0]
        assert "--format" in cmd
        fmt_idx = cmd.index("--format") + 1
        assert cmd[fmt_idx] == "json"

    def test_add_user_message_convenience(self, history, mock_file_exists):
        history.add_user_message("hi")
        cmd = mock_file_exists.call_args[0][0]
        tags_idx = cmd.index("--tags") + 1
        assert "type:human" in cmd[tags_idx]

    def test_add_ai_message_convenience(self, history, mock_file_exists):
        history.add_ai_message("hello")
        cmd = mock_file_exists.call_args[0][0]
        tags_idx = cmd.index("--tags") + 1
        assert "type:ai" in cmd[tags_idx]


# ===========================================================================
# 4. clear tests
# ===========================================================================

class TestClear:
    def test_clear_calls_init(self, history, mock_file_exists):
        history.clear()
        mock_file_exists.assert_called_once()
        cmd = mock_file_exists.call_args[0][0]
        assert cmd == ["ate", "memory", "init", "--path", "/tmp/test.mv2"]

    def test_clear_reinitialises_file(self, history, mock_file_exists):
        """After clear, messages should be empty (given fresh init)."""
        history.clear()
        # Next export returns empty
        mock_file_exists.return_value = _completed_process(stdout="[]")
        assert history.messages == []


# ===========================================================================
# 5. Error handling tests
# ===========================================================================

class TestErrorHandling:
    def test_timeout_raises_runtime_error(self, history, mock_file_exists):
        mock_file_exists.side_effect = subprocess.TimeoutExpired(cmd="ate", timeout=30)
        with pytest.raises(RuntimeError, match="timed out"):
            history.messages

    def test_file_not_found_raises_runtime_error(self, history, mock_file_exists):
        mock_file_exists.side_effect = FileNotFoundError("ate not found")
        with pytest.raises(RuntimeError, match="not found"):
            history.messages

    def test_non_zero_exit_add_logs_warning(self, history, mock_file_exists, caplog):
        mock_file_exists.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="some error"
        )
        import logging
        with caplog.at_level(logging.WARNING):
            history.add_message(HumanMessage(content="x"))
        assert "non-zero" in caplog.text.lower() or "exit code" in caplog.text.lower()


# ===========================================================================
# 6. Session isolation tests
# ===========================================================================

class TestSessionIsolation:
    def test_different_sessions_see_own_messages(self, mock_file_exists):
        h1 = AteChatMessageHistory(session_id="s1", memory_path="/tmp/shared.mv2")
        h2 = AteChatMessageHistory(session_id="s2", memory_path="/tmp/shared.mv2")

        entries = [
            _make_memory_entry(HumanMessage(content="from s1"), "s1", "m1"),
            _make_memory_entry(HumanMessage(content="from s2"), "s2", "m2"),
        ]
        mock_file_exists.return_value = _completed_process(
            stdout=_make_export_json(entries)
        )

        msgs1 = h1.messages
        assert len(msgs1) == 1
        assert msgs1[0].content == "from s1"

        msgs2 = h2.messages
        assert len(msgs2) == 1
        assert msgs2[0].content == "from s2"

    def test_session_with_no_matching_messages(self, mock_file_exists):
        h = AteChatMessageHistory(session_id="lonely", memory_path="/tmp/x.mv2")
        entries = [
            _make_memory_entry(HumanMessage(content="nope"), "other", "m1"),
        ]
        mock_file_exists.return_value = _completed_process(
            stdout=_make_export_json(entries)
        )
        assert h.messages == []


# ===========================================================================
# 7. Edge cases & misc
# ===========================================================================

class TestEdgeCases:
    def test_message_with_special_characters(self, history, mock_file_exists):
        msg = HumanMessage(content='Hello "world" \n\ttab & <html>')
        history.add_message(msg)
        cmd = mock_file_exists.call_args[0][0]
        text_idx = cmd.index("--text") + 1
        parsed = json.loads(cmd[text_idx])
        assert parsed["data"]["content"] == 'Hello "world" \n\ttab & <html>'

    def test_message_with_unicode(self, history, mock_file_exists):
        msg = HumanMessage(content="こんにちは 🌍")
        history.add_message(msg)
        cmd = mock_file_exists.call_args[0][0]
        text_idx = cmd.index("--text") + 1
        parsed = json.loads(cmd[text_idx])
        assert parsed["data"]["content"] == "こんにちは 🌍"

    def test_whitespace_only_export(self, history, mock_file_exists):
        mock_file_exists.return_value = _completed_process(stdout="   \n  ")
        assert history.messages == []

    def test_entry_missing_tags_key(self, history, mock_file_exists):
        entry = {"id": "m1", "text": json.dumps(message_to_dict(HumanMessage(content="x")))}
        # No "tags" key — should not match any session
        mock_file_exists.return_value = _completed_process(
            stdout=_make_export_json([entry])
        )
        assert history.messages == []

    def test_empty_tags_list(self, history, mock_file_exists):
        entry = {
            "id": "m1",
            "text": json.dumps(message_to_dict(HumanMessage(content="x"))),
            "tags": [],
        }
        mock_file_exists.return_value = _completed_process(
            stdout=_make_export_json([entry])
        )
        assert history.messages == []
