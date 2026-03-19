"""GitHub Copilot Chat model integration via GitHub Models inference API."""

from __future__ import annotations

import json
import os
import time
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Type,
    Union,
)

import httpx
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.base import LangSmithParams
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.messages.tool import tool_call_chunk as create_tool_call_chunk
from langchain_core.output_parsers.openai_tools import (
    make_invalid_tool_call,
    parse_tool_call,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field, SecretStr, model_validator

# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

CLIENT_ID = "Iv1.b507a08c87ecfe98"


def get_copilot_token(
    client_id: str = CLIENT_ID, callback: Optional[Callable[[str], None]] = None
) -> Optional[str]:
    """Authenticate via GitHub Device Flow to get a Copilot Token.

    This function will block and wait for the user to complete the
    authorization in their browser.

    Args:
        client_id: The GitHub OAuth App Client ID to use. Defaults
            to the VS Code Copilot Chat client ID.
        callback: Optional callable that receives status messages instead of
            printing them. If None, messages are printed to stdout.

    Returns:
        The fetched Copilot Token string, or None if authentication failed.
    """

    def _print(msg: str) -> None:
        if callback:
            callback(msg)
        else:
            print(msg)  # noqa: T201

    _print("1. Requesting device code from GitHub...")
    with httpx.Client() as client:
        res = client.post(
            "https://github.com/login/device/code",
            headers={"Accept": "application/json"},
            data={"client_id": client_id, "scope": "read:user"},
        )
        res.raise_for_status()
        data = res.json()

    device_code = data.get("device_code")
    user_code = data.get("user_code")
    verification_uri = data.get("verification_uri")
    interval = data.get("interval", 5)

    _print("\n==========================================")
    _print(f"Please open your browser to: {verification_uri}")
    _print(f"And enter the authorization code: {user_code}")
    _print("==========================================\n")
    _print(f"Waiting for authorization (checking every {interval} seconds)...")

    access_token = None
    with httpx.Client() as client:
        while True:
            token_res = client.post(
                "https://github.com/login/oauth/access_token",
                headers={"Accept": "application/json"},
                data={
                    "client_id": client_id,
                    "device_code": device_code,
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                },
            ).json()

            if "access_token" in token_res:
                access_token = token_res["access_token"]
                _print("\nAuthorization successful! Exchanging for Copilot Token...")
                break
            elif token_res.get("error") == "authorization_pending":
                time.sleep(interval)
            else:
                _print(f"\nAuthorization failed: {token_res}")
                return None

        # Exchange the standard access token for a Copilot internal token
        copilot_res = client.get(
            "https://api.github.com/copilot_internal/v2/token",
            headers={
                "Authorization": f"token {access_token}",
                "Accept": "application/json",
                "Editor-Version": "vscode/1.104.1",
                "Editor-Plugin-Version": "copilot-chat/0.26.7",
            },
        )

        if copilot_res.status_code == 200:
            copilot_token = copilot_res.json().get("token")
            _print("Successfully acquired Copilot Token!")
            return copilot_token
        else:
            _print(f"Failed to acquire Copilot Token: {copilot_res.text}")
            return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ROLE_MAP = {
    "human": "user",
    "ai": "assistant",
    "system": "system",
    "developer": "developer",
    "tool": "tool",
}

_GITHUB_COPILOT_BASE_URL = "https://api.githubcopilot.com"
_INFERENCE_PATH = "/chat/completions"

COPILOT_EDITOR_VERSION = "vscode/1.104.1"
COPILOT_PLUGIN_VERSION = "copilot-chat/0.26.7"
COPILOT_INTEGRATION_ID = "vscode-chat"
COPILOT_USER_AGENT = "GitHubCopilotChat/0.26.7"

COPILOT_DEFAULT_HEADERS = {
    "Copilot-Integration-Id": COPILOT_INTEGRATION_ID,
    "User-Agent": COPILOT_USER_AGENT,
    "Editor-Version": COPILOT_EDITOR_VERSION,
    "Editor-Plugin-Version": COPILOT_PLUGIN_VERSION,
    "editor-version": COPILOT_EDITOR_VERSION,
    "editor-plugin-version": COPILOT_PLUGIN_VERSION,
    "copilot-vision-request": "true",
}


def _message_to_dict(message: BaseMessage) -> Dict[str, Any]:
    """Convert a LangChain message to the GitHub Models API message format."""
    if isinstance(message, SystemMessage):
        return {"role": "system", "content": message.content}
    elif isinstance(message, HumanMessage):
        # Support multimodal content (list of content blocks)
        if isinstance(message.content, list):
            parts = []
            for block in message.content:
                if isinstance(block, dict):
                    btype = block.get("type", "")
                    if btype == "text":
                        parts.append({"type": "text", "text": block["text"]})
                    elif btype == "image_url":
                        parts.append(
                            {
                                "type": "image_url",
                                "image_url": block.get("image_url", {}),
                            }
                        )
                    else:
                        parts.append(block)
                else:
                    parts.append({"type": "text", "text": str(block)})
            return {"role": "user", "content": parts}
        return {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        msg: Dict[str, Any] = {"role": "assistant", "content": message.content or ""}
        # Attach tool calls if present
        if message.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["args"]),
                    },
                }
                for tc in message.tool_calls
            ]
        elif message.additional_kwargs.get("tool_calls"):
            msg["tool_calls"] = message.additional_kwargs["tool_calls"]
        return msg
    elif isinstance(message, ToolMessage):
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "content": message.content,
        }
    elif isinstance(message, ChatMessage):
        role = _ROLE_MAP.get(message.role, message.role)
        return {"role": role, "content": message.content}
    else:
        # Fallback: treat as user message
        return {"role": "user", "content": str(message.content)}


def _format_tools_for_api(
    tools: Sequence[Union[Dict[str, Any], BaseTool, Type]],
) -> List[Dict[str, Any]]:
    """Convert LangChain tools into the OpenAI-compatible format
    expected by GitHub Models.
    """
    formatted = []
    for tool in tools:
        if isinstance(tool, dict) and tool.get("type") == "function":
            formatted.append(tool)
        else:
            oai_tool = convert_to_openai_tool(tool)  # type: ignore[arg-type]
            formatted.append(oai_tool)
    return formatted


def _parse_tool_calls(
    raw_tool_calls: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Parse raw API tool_calls into LangChain tool_calls format."""
    tool_calls: List[Dict[str, Any]] = []
    for raw in raw_tool_calls:
        try:
            parsed = parse_tool_call(raw, return_id=True)
            if parsed is not None:
                tool_calls.append(parsed)
        except Exception as exc:
            invalid = make_invalid_tool_call(raw, str(exc))
            tool_calls.append(dict(invalid))
    return tool_calls


# GitHub Models API only accepts "auto", "required", or "none" for tool_choice.
# LangChain internally uses "any" (equivalent to "required") and dict-style
# {"type": "function", "function": {"name": "..."}} for specific tool forcing.
_TOOL_CHOICE_MAP: Dict[str, str] = {
    "any": "required",
}


def _normalize_tool_choice(
    tool_choice: Any,
) -> Union[str, Dict[str, Any]]:
    """Normalise a tool_choice value for the GitHub Models API.

    - ``"any"``  →  ``"required"``  (LangChain internal alias)
    - dict ``{"type": "function", "function": {"name": "X"}}``  →  kept as-is
      (the API accepts this form for forcing a specific function)
    - any other string is passed through unchanged
    """
    if isinstance(tool_choice, str):
        return _TOOL_CHOICE_MAP.get(tool_choice, tool_choice)
    # dict form — pass through unchanged
    return tool_choice


def _build_ai_message(
    choice: Dict[str, Any], usage: Optional[Dict[str, Any]]
) -> AIMessage:
    """Build an AIMessage from a single API response choice."""
    msg = choice.get("message", {})
    content: Union[str, List] = msg.get("content") or ""
    finish_reason = choice.get("finish_reason", "")

    additional_kwargs: Dict[str, Any] = {}
    tool_calls = []
    raw_tool_calls = msg.get("tool_calls", [])
    if raw_tool_calls:
        additional_kwargs["tool_calls"] = raw_tool_calls
        tool_calls = _parse_tool_calls(raw_tool_calls)

    usage_metadata: Optional[UsageMetadata] = None
    if usage:
        usage_metadata = UsageMetadata(
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        )

    response_metadata: Dict[str, Any] = {
        "finish_reason": finish_reason,
    }
    if usage:
        response_metadata["usage"] = usage

    return AIMessage(
        content=content,
        additional_kwargs=additional_kwargs,
        tool_calls=tool_calls,
        response_metadata=response_metadata,
        usage_metadata=usage_metadata,
    )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class ChatGithubCopilot(BaseChatModel):
    """GitHub Copilot Chat model integration via the GitHub Models inference API.

    GitHub Models provides access to top AI models (OpenAI GPT-4.1, DeepSeek,
    Llama, and more) through a unified OpenAI-compatible REST API.  This class
    wraps that API so that every model available in the GitHub Models catalog
    can be used as a drop-in LangChain ``BaseChatModel``.

    Setup:
        Install ``langchain-community`` and set the ``GITHUB_TOKEN`` environment
        variable (a classic or fine-grained PAT with the ``models: read`` scope,
        or a GitHub Copilot subscription token).

        .. code-block:: bash

            pip install -U langchain-community
            export GITHUB_TOKEN="github_pat_..."

    Key init args — completion params:
        model: str
            Model ID in the ``{publisher}/{model_name}`` format, e.g.
            ``"openai/gpt-4.1"`` or ``"meta/llama-3.3-70b-instruct"``.
        temperature: Optional[float]
            Sampling temperature in ``[0, 1]``.  Higher → more creative.
        max_tokens: Optional[int]
            Maximum number of tokens to generate.
        top_p: Optional[float]
            Nucleus sampling probability mass in ``[0, 1]``.
        stop: Optional[List[str]]
            Stop sequences.
        frequency_penalty: Optional[float]
            Frequency penalty in ``[-2, 2]``.
        presence_penalty: Optional[float]
            Presence penalty in ``[-2, 2]``.
        seed: Optional[int]
            Random seed for deterministic sampling (best-effort).

    Key init args — client params:
        github_token: Optional[SecretStr]
            GitHub token.  Falls back to ``GITHUB_TOKEN`` env var.
        base_url: str
            Base URL of the GitHub Models API.
            Defaults to ``"https://models.github.ai"``.
        org: Optional[str]
            Organisation login.  When set, every request is attributed to that
            org (uses the ``/orgs/{org}/inference/chat/completions`` endpoint).
        api_version: str
            GitHub Models REST API version header value.
            Defaults to ``"2026-03-10"``.
        timeout: Optional[float]
            HTTP request timeout in seconds.
        max_retries: int
            Number of automatic retries on transient errors (default ``2``).

    Instantiate:
        .. code-block:: python

            from langchain_community.chat_models import ChatGithubCopilot

            llm = ChatGithubCopilot(
                model="openai/gpt-4.1",
                temperature=0,
                max_tokens=1024,
                # github_token="github_pat_...",  # or set GITHUB_TOKEN env var
            )

    Invoke:
        .. code-block:: python

            messages = [
                ("system", "You are a helpful translator. Translate to French."),
                ("human", "I love programming."),
            ]
            ai_msg = llm.invoke(messages)
            print(ai_msg.content)
            # "J'adore la programmation."

    Stream:
        .. code-block:: python

            for chunk in llm.stream(messages):
                print(chunk.content, end="", flush=True)

    Async:
        .. code-block:: python

            ai_msg = await llm.ainvoke(messages)

            async for chunk in llm.astream(messages):
                print(chunk.content, end="", flush=True)

    Tool calling:
        .. code-block:: python

            from pydantic import BaseModel, Field

            class GetWeather(BaseModel):
                '''Get the current weather in a given location.'''
                location: str = Field(
                    ..., description="City and state, e.g. Paris, France"
                )

            llm_with_tools = llm.bind_tools([GetWeather])
            ai_msg = llm_with_tools.invoke("What is the weather like in Paris?")
            print(ai_msg.tool_calls)
            # [{'name': 'GetWeather', 'args': {'location': 'Paris, France'},
            #   'id': '...'}]

    Structured output:
        .. code-block:: python

            from typing import Optional
            from pydantic import BaseModel, Field

            class Joke(BaseModel):
                '''Joke to tell user.'''
                setup: str = Field(description="The setup of the joke")
                punchline: str = Field(description="The punchline to the joke")
                rating: Optional[int] = Field(description="Funniness rating 1-10")

            structured_llm = llm.with_structured_output(Joke)
            structured_llm.invoke("Tell me a joke about cats")

    JSON mode:
        .. code-block:: python

            json_llm = llm.bind(response_format={"type": "json_object"})
            ai_msg = json_llm.invoke(
                "Return a JSON object with key 'numbers' and a list of 5 random ints."
            )
            print(ai_msg.content)

    Image input:
        .. code-block:: python

            import base64, httpx
            from langchain_core.messages import HumanMessage

            image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
            image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")
            message = HumanMessage(
                content=[
                    {"type": "text", "text": "Describe the weather in this image."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                    },
                ]
            )
            ai_msg = llm.invoke([message])
            print(ai_msg.content)

    Token usage:
        .. code-block:: python

            ai_msg = llm.invoke(messages)
            print(ai_msg.usage_metadata)
            # {'input_tokens': 28, 'output_tokens': 18, 'total_tokens': 46}

    Response metadata:
        .. code-block:: python

            ai_msg = llm.invoke(messages)
            print(ai_msg.response_metadata)
            # {'finish_reason': 'stop', 'usage': {'prompt_tokens': 28, ...}}
    """

    # ------------------------------------------------------------------
    # Fields
    # ------------------------------------------------------------------

    model_name: str = Field(alias="model")
    """Model ID in the ``{publisher}/{model_name}`` format.

    Examples: ``"openai/gpt-4.1"``, ``"meta/llama-3.3-70b-instruct"``.
    """

    github_token: Optional[SecretStr] = Field(default=None)
    """GitHub token with ``models: read`` scope.

    If not provided, the value of the ``GITHUB_TOKEN`` environment variable
    is used.
    """

    base_url: str = _GITHUB_COPILOT_BASE_URL
    """Base URL for the GitHub Copilot API."""

    temperature: Optional[float] = None
    """Sampling temperature in ``[0, 1]``."""

    max_tokens: Optional[int] = None
    """Maximum number of tokens to generate."""

    top_p: Optional[float] = None
    """Nucleus sampling probability mass in ``[0, 1]``."""

    stop: Optional[List[str]] = None
    """Stop sequences that terminate generation."""

    frequency_penalty: Optional[float] = None
    """Frequency penalty in ``[-2, 2]``."""

    presence_penalty: Optional[float] = None
    """Presence penalty in ``[-2, 2]``."""

    seed: Optional[int] = None
    """Random seed for (best-effort) deterministic sampling."""

    timeout: Optional[float] = None
    """HTTP request timeout in seconds."""

    max_retries: int = 2
    """Number of automatic retries on transient errors."""

    # ------------------------------------------------------------------
    # Pydantic v2 config — allow the ``model`` alias on construction
    # ------------------------------------------------------------------
    model_config = {"populate_by_name": True}

    # ------------------------------------------------------------------
    # Validators / setup
    # ------------------------------------------------------------------

    @model_validator(mode="before")
    @classmethod
    def _validate_token(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve the GitHub token from the environment if not supplied.

        Priority order:
        1. Explicitly passed ``github_token``
        2. Explicitly passed ``api_key`` alias
        3. ``GITHUB_TOKEN`` environment variable
        """
        token = values.get("github_token") or values.get("api_key")
        if not token:
            token = os.environ.get("GITHUB_TOKEN")
            if token:
                values["github_token"] = token
        return values

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def _token(self) -> str:
        """Return the raw GitHub token string."""
        if self.github_token:
            return self.github_token.get_secret_value()
        env_token = os.environ.get("GITHUB_TOKEN", "")
        if not env_token:
            raise ValueError(
                "A GitHub token is required.  Set the GITHUB_TOKEN environment "
                "variable or pass ``github_token`` when instantiating "
                "ChatGithubCopilot."
            )
        return env_token

    @property
    def _inference_url(self) -> str:
        """Return the full chat-completions endpoint URL."""
        return self.base_url.rstrip("/") + _INFERENCE_PATH

    def _build_headers(self) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        headers.update(COPILOT_DEFAULT_HEADERS)
        return headers

    @classmethod
    def get_available_models(
        cls, github_token: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get the list of available models from the GitHub Copilot API."""
        token = github_token or os.environ.get("GITHUB_TOKEN")
        if not token:
            raise ValueError(
                "A GitHub token is required. Set the GITHUB_TOKEN environment "
                "variable or pass ``github_token``."
            )

        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        headers.update(COPILOT_DEFAULT_HEADERS)

        url = f"{_GITHUB_COPILOT_BASE_URL}/models"

        with httpx.Client() as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            return data.get("data", [])

    def _build_payload(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Assemble the JSON body for the inference API."""
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": [_message_to_dict(m) for m in messages],
            "stream": stream,
        }
        if stream:
            payload["stream_options"] = {"include_usage": True}

        # Optional sampling params (kwargs override instance-level defaults)
        for field_name, api_key in [
            ("temperature", "temperature"),
            ("max_tokens", "max_tokens"),
            ("top_p", "top_p"),
            ("frequency_penalty", "frequency_penalty"),
            ("presence_penalty", "presence_penalty"),
            ("seed", "seed"),
        ]:
            value = kwargs.pop(api_key, None) or getattr(self, field_name, None)
            if value is not None:
                payload[api_key] = value

        # Stop sequences
        effective_stop = stop or self.stop
        if effective_stop:
            payload["stop"] = effective_stop

        # Tools / tool_choice
        tools = kwargs.pop("tools", None)
        if tools:
            payload["tools"] = _format_tools_for_api(tools)
            tool_choice = kwargs.pop("tool_choice", None)
            if tool_choice:
                payload["tool_choice"] = _normalize_tool_choice(tool_choice)

        # Response format (JSON mode / structured output)
        response_format = kwargs.pop("response_format", None)
        if response_format:
            payload["response_format"] = response_format

        # Pass through any remaining caller-supplied kwargs
        payload.update(kwargs)
        return payload

    def _do_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Perform a synchronous (non-streaming) HTTP POST with retries."""
        headers = self._build_headers()
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                response = httpx.post(
                    self._inference_url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return response.json()
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_exc = exc
                if attempt == self.max_retries:
                    raise
            except httpx.HTTPStatusError as exc:
                # Don't retry on 4xx client errors
                if exc.response.status_code < 500:
                    raise
                last_exc = exc
                if attempt == self.max_retries:
                    raise
        raise RuntimeError("Unexpected retry loop exit") from last_exc

    def _do_stream(self, payload: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """Perform a synchronous streaming HTTP POST and yield parsed SSE chunks."""
        headers = self._build_headers()
        with httpx.stream(
            "POST",
            self._inference_url,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                line = line.strip()
                if not line or line == "data: [DONE]":
                    continue
                if line.startswith("data: "):
                    line = line[len("data: ") :]
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    async def _do_request_async(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Perform an asynchronous (non-streaming) HTTP POST with retries."""
        headers = self._build_headers()
        last_exc: Optional[Exception] = None
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for attempt in range(self.max_retries + 1):
                try:
                    response = await client.post(
                        self._inference_url,
                        headers=headers,
                        json=payload,
                    )
                    response.raise_for_status()
                    return response.json()
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    last_exc = exc
                    if attempt == self.max_retries:
                        raise
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code < 500:
                        raise
                    last_exc = exc
                    if attempt == self.max_retries:
                        raise
        raise RuntimeError("Unexpected retry loop exit") from last_exc

    async def _do_stream_async(
        self, payload: Dict[str, Any]
    ) -> AsyncIterator[Dict[str, Any]]:
        """Perform an asynchronous streaming HTTP POST and yield parsed SSE chunks."""
        headers = self._build_headers()
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                self._inference_url,
                headers=headers,
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line or line == "data: [DONE]":
                        continue
                    if line.startswith("data: "):
                        line = line[len("data: ") :]
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue

    # ------------------------------------------------------------------
    # Stream delta → AIMessageChunk helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk_from_delta(
        delta: Dict[str, Any],
        finish_reason: Optional[str],
        usage: Optional[Dict[str, Any]],
    ) -> AIMessageChunk:
        """Convert a single SSE delta object into an ``AIMessageChunk``."""
        content = delta.get("content") or ""
        additional_kwargs: Dict[str, Any] = {}
        tool_call_chunks = []

        raw_tool_calls = delta.get("tool_calls") or []
        for raw_tc in raw_tool_calls:
            index = raw_tc.get("index", 0)
            tc_id = raw_tc.get("id")

            func = raw_tc.get("function", {})
            tool_call_chunks.append(
                create_tool_call_chunk(
                    name=func.get("name"),
                    args=func.get("arguments"),
                    id=tc_id,
                    index=index,
                )
            )

        response_metadata: Dict[str, Any] = {}
        if finish_reason:
            response_metadata["finish_reason"] = finish_reason

        usage_metadata: Optional[UsageMetadata] = None
        if usage:
            usage_metadata = UsageMetadata(
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            )
            response_metadata["usage"] = usage

        return AIMessageChunk(
            content=content,
            additional_kwargs=additional_kwargs,
            tool_call_chunks=tool_call_chunks,
            response_metadata=response_metadata,
            usage_metadata=usage_metadata,
        )

    # ------------------------------------------------------------------
    # LangChain BaseChatModel interface
    # ------------------------------------------------------------------

    @property
    def _llm_type(self) -> str:
        return "github-copilot"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    def _get_ls_params(
        self,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> LangSmithParams:
        params = self._identifying_params
        return LangSmithParams(
            ls_provider="github-copilot",
            ls_model_name=self.model_name,
            ls_model_type="chat",
            ls_temperature=params.get("temperature"),
            ls_max_tokens=params.get("max_tokens"),
            ls_stop=stop or self.stop or [],
        )

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Call the GitHub Models chat completions API and return a ChatResult."""
        payload = self._build_payload(messages, stop=stop, stream=False, **kwargs)
        response_data = self._do_request(payload)

        choices = response_data.get("choices", [])
        if not choices:
            raise ValueError(
                f"GitHub Models API returned no choices. Response: {response_data}"
            )

        usage = response_data.get("usage")
        generations = []
        for choice in choices:
            ai_msg = _build_ai_message(choice, usage)
            generations.append(ChatGeneration(message=ai_msg))

        return ChatResult(generations=generations)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream token-level chunks from the GitHub Models API."""
        payload = self._build_payload(messages, stop=stop, stream=True, **kwargs)

        for raw_chunk in self._do_stream(payload):
            choices = raw_chunk.get("choices", [])
            usage = raw_chunk.get(
                "usage"
            )  # present in the final chunk when include_usage=True

            if not choices and usage:
                # Final usage-only chunk
                chunk = ChatGenerationChunk(
                    message=AIMessageChunk(
                        content="",
                        usage_metadata=UsageMetadata(
                            input_tokens=usage.get("prompt_tokens", 0),
                            output_tokens=usage.get("completion_tokens", 0),
                            total_tokens=usage.get("total_tokens", 0),
                        ),
                        response_metadata={"usage": usage},
                    )
                )
                if run_manager:
                    run_manager.on_llm_new_token("", chunk=chunk)
                yield chunk
                continue

            for choice in choices:
                delta = choice.get("delta", {})
                finish_reason = choice.get("finish_reason")
                ai_chunk = self._chunk_from_delta(delta, finish_reason, usage)
                gen_chunk = ChatGenerationChunk(message=ai_chunk)

                if run_manager and ai_chunk.content:
                    run_manager.on_llm_new_token(str(ai_chunk.content), chunk=gen_chunk)
                yield gen_chunk

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async version of ``_generate``."""
        payload = self._build_payload(messages, stop=stop, stream=False, **kwargs)
        response_data = await self._do_request_async(payload)

        choices = response_data.get("choices", [])
        if not choices:
            raise ValueError(
                f"GitHub Models API returned no choices. Response: {response_data}"
            )

        usage = response_data.get("usage")
        generations = []
        for choice in choices:
            ai_msg = _build_ai_message(choice, usage)
            generations.append(ChatGeneration(message=ai_msg))

        return ChatResult(generations=generations)

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Async streaming version of ``_stream``."""
        payload = self._build_payload(messages, stop=stop, stream=True, **kwargs)

        async for raw_chunk in self._do_stream_async(payload):
            choices = raw_chunk.get("choices", [])
            usage = raw_chunk.get("usage")

            if not choices and usage:
                chunk = ChatGenerationChunk(
                    message=AIMessageChunk(
                        content="",
                        usage_metadata=UsageMetadata(
                            input_tokens=usage.get("prompt_tokens", 0),
                            output_tokens=usage.get("completion_tokens", 0),
                            total_tokens=usage.get("total_tokens", 0),
                        ),
                        response_metadata={"usage": usage},
                    )
                )
                if run_manager:
                    await run_manager.on_llm_new_token("", chunk=chunk)
                yield chunk
                continue

            for choice in choices:
                delta = choice.get("delta", {})
                finish_reason = choice.get("finish_reason")
                ai_chunk = self._chunk_from_delta(delta, finish_reason, usage)
                gen_chunk = ChatGenerationChunk(message=ai_chunk)

                if run_manager and ai_chunk.content:
                    await run_manager.on_llm_new_token(
                        str(ai_chunk.content), chunk=gen_chunk
                    )
                yield gen_chunk

    # ------------------------------------------------------------------
    # Tool calling support
    # ------------------------------------------------------------------

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], BaseTool, Type, Any]],
        *,
        tool_choice: Optional[Union[str, Literal["auto", "required", "none"]]] = None,
        **kwargs: Any,
    ) -> "ChatGithubCopilot":
        """Bind tools to this model, enabling tool calling.

        Args:
            tools: A list of tools to bind.  Accepts LangChain ``BaseTool``
                instances, Pydantic models, or pre-formatted OpenAI tool dicts.
            tool_choice: Controls tool selection.  One of ``"auto"``,
                ``"required"``, ``"none"``, or the name of a specific tool.
                Defaults to ``"auto"`` when tools are provided.

        Returns:
            A new ``ChatGithubCopilot`` instance with ``tools`` bound.

        Example:
            .. code-block:: python

                from pydantic import BaseModel, Field

                class SearchWeb(BaseModel):
                    '''Search the web for up-to-date information.'''
                    query: str = Field(..., description="The search query")

                llm_with_tools = llm.bind_tools([SearchWeb])
                ai_msg = llm_with_tools.invoke("Who won the 2024 Olympics 100m sprint?")
                print(ai_msg.tool_calls)
        """
        formatted_tools = _format_tools_for_api(tools)
        tool_choice_param: Optional[str] = tool_choice or (
            "auto" if formatted_tools else None
        )
        return self.bind(
            tools=formatted_tools,
            tool_choice=tool_choice_param,
            **kwargs,
        )  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Backwards-compatible alias (matches the generated stub name)
# ---------------------------------------------------------------------------

ChatGithubcopilotChat = ChatGithubCopilot
