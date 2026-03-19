"""GitHub Copilot Chat embeddings integration via GitHub Models Embeddings API."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Union

import httpx
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, Field, SecretStr, model_validator

_GITHUB_COPILOT_BASE_URL = "https://api.githubcopilot.com"
_EMBEDDINGS_PATH = "/embeddings"

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


class GithubcopilotChatEmbeddings(BaseModel, Embeddings):
    """GitHub Copilot Chat embedding model integration via the GitHub Models API.

    GitHub Models provides access to embedding models (e.g. OpenAI
    ``text-embedding-3-small``, ``text-embedding-3-large``) through a unified
    OpenAI-compatible REST API.  This class wraps the ``/inference/embeddings``
    endpoint so that any embedding model available in the GitHub Models catalog
    can be used as a drop-in LangChain ``Embeddings`` implementation.

    Setup:
        Install ``langchain-community`` and set the ``GITHUB_TOKEN`` environment
        variable (a classic or fine-grained PAT with the ``models: read`` scope,
        or a GitHub Copilot subscription token).

        .. code-block:: bash

            pip install -U langchain-community
            export GITHUB_TOKEN="github_pat_..."

    Key init args:
        model: str
            Model ID in the ``{publisher}/{model_name}`` format, e.g.
            ``"openai/text-embedding-3-small"``.
        github_token: Optional[SecretStr]
            GitHub token.  Falls back to ``GITHUB_TOKEN`` env var.
        base_url: str
            Base URL of the GitHub Models API.
            Defaults to ``"https://models.github.ai"``.
        org: Optional[str]
            Organisation login.  When set, requests are attributed to that org.
        api_version: str
            GitHub Models REST API version header value.
            Defaults to ``"2026-03-10"``.
        dimensions: Optional[int]
            The number of dimensions for the output embeddings.  Only supported
            by ``text-embedding-3`` and later models.
        encoding_format: str
            The format to return embeddings in.  Either ``"float"`` (default)
            or ``"base64"``.
        timeout: Optional[float]
            HTTP request timeout in seconds.
        max_retries: int
            Number of automatic retries on transient errors (default ``2``).

    Instantiate:
        .. code-block:: python

            from langchain_community.embeddings import GithubcopilotChatEmbeddings

            embed = GithubcopilotChatEmbeddings(
                model="openai/text-embedding-3-small",
                # github_token="github_pat_...",  # or set GITHUB_TOKEN env var
            )

    Embed single text:
        .. code-block:: python

            vector = embed.embed_query("What is the meaning of life?")
            print(len(vector))   # e.g. 1536

    Embed multiple texts:
        .. code-block:: python

            vectors = embed.embed_documents(
                ["Document one.", "Document two."]
            )
            print(len(vectors), len(vectors[0]))

    Async:
        .. code-block:: python

            vector = await embed.aembed_query("What is the meaning of life?")

            vectors = await embed.aembed_documents(
                ["Document one.", "Document two."]
            )
    """

    model_config = {"populate_by_name": True}

    model_name: str = Field(alias="model")
    """Embedding model ID in the ``{publisher}/{model_name}`` format.

    Examples: ``"openai/text-embedding-3-small"``,
    ``"openai/text-embedding-3-large"``.
    """

    github_token: Optional[SecretStr] = Field(default=None)
    """GitHub token with ``models: read`` scope.

    If not provided, the value of the ``GITHUB_TOKEN`` environment variable
    is used.
    """

    base_url: str = _GITHUB_COPILOT_BASE_URL
    """Base URL for the GitHub Copilot API."""

    dimensions: Optional[int] = None
    """Number of output embedding dimensions.

    Only supported by ``text-embedding-3`` and later models.
    """

    encoding_format: str = "float"
    """Format of the returned embeddings.  Either ``"float"`` or ``"base64"``."""

    timeout: Optional[float] = None
    """HTTP request timeout in seconds."""

    max_retries: int = 2
    """Number of automatic retries on transient errors."""

    # ------------------------------------------------------------------
    # Validators / setup
    # ------------------------------------------------------------------

    @model_validator(mode="before")
    @classmethod
    def _resolve_token(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve the GitHub token from the environment if not supplied."""
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
                "GithubcopilotChatEmbeddings."
            )
        return env_token

    @property
    def _embeddings_url(self) -> str:
        """Return the full embeddings endpoint URL."""
        return self.base_url.rstrip("/") + _EMBEDDINGS_PATH

    def _build_headers(self) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        headers.update(COPILOT_DEFAULT_HEADERS)
        return headers

    def _build_payload(self, input: Union[str, List[str]]) -> Dict[str, Any]:
        """Assemble the JSON body for the embeddings API."""
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "input": input,
            "encoding_format": self.encoding_format,
        }
        if self.dimensions is not None:
            payload["dimensions"] = self.dimensions
        return payload

    def _do_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Perform a synchronous HTTP POST with retries."""
        headers = self._build_headers()
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                response = httpx.post(
                    self._embeddings_url,
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
                if exc.response.status_code < 500:
                    raise
                last_exc = exc
                if attempt == self.max_retries:
                    raise
        raise RuntimeError("Unexpected retry loop exit") from last_exc

    async def _do_request_async(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Perform an asynchronous HTTP POST with retries."""
        headers = self._build_headers()
        last_exc: Optional[Exception] = None
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for attempt in range(self.max_retries + 1):
                try:
                    response = await client.post(
                        self._embeddings_url,
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

    @staticmethod
    def _extract_embeddings(response_data: Dict[str, Any]) -> List[List[float]]:
        """Extract the list of embedding vectors from an API response."""
        data = response_data.get("data", [])
        if not data:
            raise ValueError(
                f"GitHub Models Embeddings API returned no data. "
                f"Response: {response_data}"
            )
        # Sort by index to preserve input order (the API may reorder items)
        sorted_data = sorted(data, key=lambda x: x.get("index", 0))
        return [item["embedding"] for item in sorted_data]

    # ------------------------------------------------------------------
    # LangChain Embeddings interface
    # ------------------------------------------------------------------

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using the GitHub Models Embeddings API.

        Args:
            texts: The list of texts to embed.

        Returns:
            A list of embedding vectors, one per input text.
        """
        if not texts:
            return []
        payload = self._build_payload(texts)
        response_data = self._do_request(payload)
        return self._extract_embeddings(response_data)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text using the GitHub Models Embeddings API.

        Args:
            text: The text to embed.

        Returns:
            An embedding vector.
        """
        return self.embed_documents([text])[0]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronously embed a list of documents.

        Args:
            texts: The list of texts to embed.

        Returns:
            A list of embedding vectors, one per input text.
        """
        if not texts:
            return []
        payload = self._build_payload(texts)
        response_data = await self._do_request_async(payload)
        return self._extract_embeddings(response_data)

    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronously embed a single query text.

        Args:
            text: The text to embed.

        Returns:
            An embedding vector.
        """
        results = await self.aembed_documents([text])
        return results[0]
