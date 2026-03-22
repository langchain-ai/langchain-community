"""Unit tests for Spix LangChain tools (mocked httpx — no live API calls)."""

from unittest.mock import MagicMock, patch

import pytest

from langchain_community.tools.spix.tool import (
    SpixCallTool,
    SpixEmailTool,
    SpixSMSTool,
    _get_api_key,
)


# ---------------------------------------------------------------------------
# _get_api_key
# ---------------------------------------------------------------------------


def test_get_api_key_from_env(monkeypatch):
    monkeypatch.setenv("SPIX_API_KEY", "sk_env_key")
    assert _get_api_key(None) == "sk_env_key"


def test_get_api_key_explicit_overrides_env(monkeypatch):
    monkeypatch.setenv("SPIX_API_KEY", "sk_env_key")
    assert _get_api_key("sk_explicit") == "sk_explicit"


def test_get_api_key_raises_when_missing(monkeypatch):
    monkeypatch.delenv("SPIX_API_KEY", raising=False)
    with pytest.raises(ValueError, match="SPIX_API_KEY"):
        _get_api_key(None)


# ---------------------------------------------------------------------------
# SpixCallTool
# ---------------------------------------------------------------------------


@patch("langchain_community.tools.spix.tool.httpx.post")
def test_spix_call_tool_success(mock_post, monkeypatch):
    monkeypatch.setenv("SPIX_API_KEY", "sk_test")
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "ok": True,
        "data": {"session_id": "sess_abc123", "status": "initiated"},
    }
    mock_post.return_value = mock_response

    tool = SpixCallTool()
    result = tool._run(
        to="+19175550123",
        playbook_id="cmp_call_abc123",
        sender="+14155550101",
    )

    assert "sess_abc123" in result
    assert "initiated" in result
    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args
    assert call_kwargs[1]["json"]["to"] == "+19175550123"


@patch("langchain_community.tools.spix.tool.httpx.post")
def test_spix_call_tool_api_error(mock_post, monkeypatch):
    monkeypatch.setenv("SPIX_API_KEY", "sk_test")
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "ok": False,
        "error": {"code": "insufficient_credits", "message": "Not enough credits"},
    }
    mock_post.return_value = mock_response

    tool = SpixCallTool()
    with pytest.raises(RuntimeError, match="insufficient_credits"):
        tool._run(
            to="+19175550123",
            playbook_id="cmp_call_abc123",
            sender="+14155550101",
        )


def test_spix_call_tool_missing_api_key(monkeypatch):
    monkeypatch.delenv("SPIX_API_KEY", raising=False)
    tool = SpixCallTool()
    with pytest.raises(ValueError, match="SPIX_API_KEY"):
        tool._run(to="+19175550123", playbook_id="cmp_call_abc123", sender="+14155550101")


# ---------------------------------------------------------------------------
# SpixSMSTool
# ---------------------------------------------------------------------------


@patch("langchain_community.tools.spix.tool.httpx.post")
def test_spix_sms_tool_success(mock_post, monkeypatch):
    monkeypatch.setenv("SPIX_API_KEY", "sk_test")
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "ok": True,
        "data": {"message_id": "msg_xyz789", "segments": 1, "credits_used": 1},
    }
    mock_post.return_value = mock_response

    tool = SpixSMSTool()
    result = tool._run(
        to="+19175550123",
        sender="+14155550101",
        body="Your appointment is confirmed.",
    )

    assert "msg_xyz789" in result
    assert "Segments: 1" in result


@patch("langchain_community.tools.spix.tool.httpx.post")
def test_spix_sms_tool_with_playbook(mock_post, monkeypatch):
    monkeypatch.setenv("SPIX_API_KEY", "sk_test")
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "ok": True,
        "data": {"message_id": "msg_123", "segments": 1, "credits_used": 1},
    }
    mock_post.return_value = mock_response

    tool = SpixSMSTool()
    tool._run(
        to="+19175550123",
        sender="+14155550101",
        body="Hi!",
        playbook_id="cmp_sms_abc",
    )

    call_kwargs = mock_post.call_args
    assert call_kwargs[1]["json"]["playbook_id"] == "cmp_sms_abc"


# ---------------------------------------------------------------------------
# SpixEmailTool
# ---------------------------------------------------------------------------


@patch("langchain_community.tools.spix.tool.httpx.post")
def test_spix_email_tool_success(mock_post, monkeypatch):
    monkeypatch.setenv("SPIX_API_KEY", "sk_test")
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "ok": True,
        "data": {"message_id": "em_abc456", "credits_used": 2},
    }
    mock_post.return_value = mock_response

    tool = SpixEmailTool()
    result = tool._run(
        sender="support@spix.sh",
        to="john@example.com",
        subject="Order confirmed",
        body="Hi John, your order is confirmed.",
    )

    assert "em_abc456" in result
    call_kwargs = mock_post.call_args
    assert "/email/send" in call_kwargs[0][0]


# ---------------------------------------------------------------------------
# Tool metadata
# ---------------------------------------------------------------------------


def test_tool_names():
    assert SpixCallTool().name == "spix_call"
    assert SpixSMSTool().name == "spix_sms"
    assert SpixEmailTool().name == "spix_email"


def test_tool_args_schemas():
    from pydantic import BaseModel

    for tool_cls in [SpixCallTool, SpixSMSTool, SpixEmailTool]:
        schema = tool_cls().args_schema
        assert issubclass(schema, BaseModel)


def test_tool_descriptions_mention_channel():
    assert "phone" in SpixCallTool().description.lower()
    assert "sms" in SpixSMSTool().description.lower()
    assert "email" in SpixEmailTool().description.lower()
