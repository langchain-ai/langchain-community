"""Spix tools for LangChain.

Provides SpixCallTool, SpixSMSTool, and SpixEmailTool — LangChain tools that
let any agent make phone calls, send SMS, and send email via Spix.

Spix (https://spix.sh) is communications infrastructure for AI agents: real
phone numbers, AI voice calls (~500ms latency), SMS, and email.

Install:
    pip install langchain-community httpx

Usage:
    .. code-block:: python

        import os
        from langchain_community.tools.spix import SpixCallTool, SpixSMSTool, SpixEmailTool

        os.environ["SPIX_API_KEY"] = "your-api-key"
        tools = [SpixCallTool(), SpixSMSTool(), SpixEmailTool()]

Get an API key at https://app.spix.sh/api-keys.
"""

from __future__ import annotations

import os
from typing import Optional, Type

import httpx
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

SPIX_API_BASE = "https://api.spix.sh/v1"


def _get_api_key(api_key: Optional[str]) -> str:
    key = api_key or os.environ.get("SPIX_API_KEY")
    if not key:
        raise ValueError(
            "Spix API key not found. Pass api_key= or set the SPIX_API_KEY "
            "environment variable. Get your key at https://app.spix.sh/api-keys"
        )
    return key


def _spix_post(path: str, payload: dict, api_key: str) -> dict:
    """POST to Spix API and return parsed JSON. Raises on HTTP or API errors."""
    url = f"{SPIX_API_BASE}{path}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        resp = httpx.post(url, json=payload, headers=headers, timeout=30)
        data = resp.json()
    except httpx.TimeoutException:
        raise RuntimeError(f"Spix API request timed out: POST {path}")
    except Exception as exc:
        raise RuntimeError(f"Spix API request failed: {exc}") from exc

    if not data.get("ok"):
        error = data.get("error", {})
        code = error.get("code", "unknown_error")
        message = error.get("message", "An unknown error occurred")
        raise RuntimeError(f"Spix API error [{code}]: {message}")

    return data["data"]


# ---------------------------------------------------------------------------
# SpixCallTool
# ---------------------------------------------------------------------------


class _CallInput(BaseModel):
    to: str = Field(
        description="The E.164 phone number to call, e.g. '+19175550123'."
    )
    playbook_id: str = Field(
        description=(
            "The Spix call playbook ID, e.g. 'cmp_call_abc123'. "
            "The playbook defines the AI persona, script, and success criteria."
        )
    )
    sender: str = Field(
        description=(
            "The E.164 Spix number to call from, e.g. '+14155550101'. "
            "Must be rented on your account and bound to this playbook."
        )
    )


class SpixCallTool(BaseTool):
    """LangChain tool that places an outbound AI phone call via Spix.

    The call runs a playbook — an AI persona with a script, voice, and success
    criteria defined in your Spix dashboard. Returns immediately with a session
    ID; the call itself happens asynchronously on Spix's voice engine
    (~500ms latency using Deepgram Nova-3 STT + Claude LLM + Cartesia
    Sonic-3 TTS).

    Requires the ``SPIX_API_KEY`` environment variable or the ``api_key``
    constructor argument. Get a key at https://app.spix.sh/api-keys.

    Args:
        api_key: Spix API key. Falls back to the ``SPIX_API_KEY`` env var.

    Example:
        .. code-block:: python

            from langchain_community.tools.spix import SpixCallTool

            tool = SpixCallTool()
            result = tool.run({
                "to": "+19175550123",
                "playbook_id": "cmp_call_abc123",
                "sender": "+14155550101",
            })
    """

    name: str = "spix_call"
    description: str = (
        "Place an outbound AI phone call using Spix. "
        "The call runs a playbook that defines the AI persona and script. "
        "Use this when you need to speak to a person by phone. "
        "Input: to (E.164 number), playbook_id, sender (E.164 Spix number)."
    )
    args_schema: Type[BaseModel] = _CallInput
    api_key: Optional[str] = None

    def _run(
        self,
        to: str,
        playbook_id: str,
        sender: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        key = _get_api_key(self.api_key)
        result = _spix_post(
            "/calls",
            {"to": to, "playbook_id": playbook_id, "sender": sender},
            key,
        )
        session_id = result.get("session_id", "unknown")
        status = result.get("status", "unknown")
        return (
            f"Call placed successfully. "
            f"Session ID: {session_id}. "
            f"Status: {status}. "
            f"Track live: spix watch transcript {session_id}"
        )

    async def _arun(
        self,
        to: str,
        playbook_id: str,
        sender: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        key = _get_api_key(self.api_key)
        url = f"{SPIX_API_BASE}/calls"
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                url,
                json={"to": to, "playbook_id": playbook_id, "sender": sender},
                headers=headers,
            )
        data = resp.json()
        if not data.get("ok"):
            error = data.get("error", {})
            raise RuntimeError(
                f"Spix API error [{error.get('code')}]: {error.get('message')}"
            )
        result = data["data"]
        session_id = result.get("session_id", "unknown")
        status = result.get("status", "unknown")
        return (
            f"Call placed successfully. "
            f"Session ID: {session_id}. "
            f"Status: {status}. "
            f"Track live: spix watch transcript {session_id}"
        )


# ---------------------------------------------------------------------------
# SpixSMSTool
# ---------------------------------------------------------------------------


class _SMSInput(BaseModel):
    to: str = Field(
        description="The E.164 phone number to send the SMS to, e.g. '+19175550123'."
    )
    sender: str = Field(
        description=(
            "The E.164 Spix number to send from, e.g. '+14155550101'. "
            "Must be rented on your account."
        )
    )
    body: str = Field(
        description=(
            "The SMS body. Keep under 160 characters for a single segment. "
            "Longer messages are split into multiple segments and cost more credits."
        )
    )
    playbook_id: Optional[str] = Field(
        default=None,
        description=(
            "Optional SMS playbook ID. Recommended for agents. "
            "If omitted, resolved automatically from the sender number."
        ),
    )


class SpixSMSTool(BaseTool):
    """LangChain tool that sends an SMS via Spix.

    Args:
        api_key: Spix API key. Falls back to the ``SPIX_API_KEY`` env var.

    Example:
        .. code-block:: python

            from langchain_community.tools.spix import SpixSMSTool

            tool = SpixSMSTool()
            result = tool.run({
                "to": "+19175550123",
                "sender": "+14155550101",
                "body": "Your order is confirmed.",
            })
    """

    name: str = "spix_sms"
    description: str = (
        "Send an SMS message via Spix. "
        "Use this when you need to text a person. "
        "Input: to (E.164 number), sender (E.164 Spix number), body (message text), "
        "and optionally playbook_id."
    )
    args_schema: Type[BaseModel] = _SMSInput
    api_key: Optional[str] = None

    def _run(
        self,
        to: str,
        sender: str,
        body: str,
        playbook_id: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        key = _get_api_key(self.api_key)
        payload: dict = {"to": to, "sender": sender, "body": body}
        if playbook_id:
            payload["playbook_id"] = playbook_id
        result = _spix_post("/sms", payload, key)
        message_id = result.get("message_id", "unknown")
        segments = result.get("segments", "?")
        credits_used = result.get("credits_used", "?")
        return (
            f"SMS sent successfully. "
            f"Message ID: {message_id}. "
            f"Segments: {segments}. "
            f"Credits used: {credits_used}."
        )

    async def _arun(
        self,
        to: str,
        sender: str,
        body: str,
        playbook_id: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        key = _get_api_key(self.api_key)
        payload: dict = {"to": to, "sender": sender, "body": body}
        if playbook_id:
            payload["playbook_id"] = playbook_id
        url = f"{SPIX_API_BASE}/sms"
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(url, json=payload, headers=headers)
        data = resp.json()
        if not data.get("ok"):
            error = data.get("error", {})
            raise RuntimeError(
                f"Spix API error [{error.get('code')}]: {error.get('message')}"
            )
        result = data["data"]
        message_id = result.get("message_id", "unknown")
        segments = result.get("segments", "?")
        credits_used = result.get("credits_used", "?")
        return (
            f"SMS sent successfully. "
            f"Message ID: {message_id}. "
            f"Segments: {segments}. "
            f"Credits used: {credits_used}."
        )


# ---------------------------------------------------------------------------
# SpixEmailTool
# ---------------------------------------------------------------------------


class _EmailInput(BaseModel):
    sender: str = Field(
        description=(
            "The Spix inbox address to send from, e.g. 'support@spix.sh'. "
            "Must be a registered Spix inbox."
        )
    )
    to: str = Field(description="The recipient email address, e.g. 'john@example.com'.")
    subject: str = Field(description="Email subject line.")
    body: str = Field(description="Plain-text email body.")


class SpixEmailTool(BaseTool):
    """LangChain tool that sends an email via Spix.

    Args:
        api_key: Spix API key. Falls back to the ``SPIX_API_KEY`` env var.

    Example:
        .. code-block:: python

            from langchain_community.tools.spix import SpixEmailTool

            tool = SpixEmailTool()
            result = tool.run({
                "sender": "support@spix.sh",
                "to": "john@example.com",
                "subject": "Order confirmed",
                "body": "Hi John, your order #4421 is confirmed.",
            })
    """

    name: str = "spix_email"
    description: str = (
        "Send an email via Spix. "
        "Use this when you need to email a person. "
        "Input: sender (Spix inbox address), to (recipient email), "
        "subject (email subject), body (plain-text body)."
    )
    args_schema: Type[BaseModel] = _EmailInput
    api_key: Optional[str] = None

    def _run(
        self,
        sender: str,
        to: str,
        subject: str,
        body: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        key = _get_api_key(self.api_key)
        result = _spix_post(
            "/email/send",
            {"sender": sender, "to": to, "subject": subject, "body": body},
            key,
        )
        message_id = result.get("message_id", "unknown")
        credits_used = result.get("credits_used", "?")
        return (
            f"Email sent successfully. "
            f"Message ID: {message_id}. "
            f"Credits used: {credits_used}."
        )

    async def _arun(
        self,
        sender: str,
        to: str,
        subject: str,
        body: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        key = _get_api_key(self.api_key)
        url = f"{SPIX_API_BASE}/email/send"
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                url,
                json={"sender": sender, "to": to, "subject": subject, "body": body},
                headers=headers,
            )
        data = resp.json()
        if not data.get("ok"):
            error = data.get("error", {})
            raise RuntimeError(
                f"Spix API error [{error.get('code')}]: {error.get('message')}"
            )
        result = data["data"]
        message_id = result.get("message_id", "unknown")
        credits_used = result.get("credits_used", "?")
        return (
            f"Email sent successfully. "
            f"Message ID: {message_id}. "
            f"Credits used: {credits_used}."
        )
