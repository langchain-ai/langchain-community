"""Util that calls AskNews api."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from langchain_core.utils import get_from_dict_or_env
from pydantic import BaseModel, ConfigDict, model_validator


class AskNewsAPIWrapper(BaseModel):
    """Wrapper for AskNews API."""

    asknews_sync: Any = None
    asknews_async: Any = None
    asknews_client_id: Optional[str] = None
    """Client ID for the AskNews API."""
    asknews_client_secret: Optional[str] = None
    """Client Secret for the AskNews API."""
    asknews_api_key: Optional[str] = None
    """API Key for the AskNews API."""

    model_config = ConfigDict(
        extra="forbid",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that api credentials and python package exists in environment."""

        try:
            import asknews_sdk

        except ImportError:
            raise ImportError(
                "AskNews python package not found. "
                "Please install it with `pip install asknews`."
            )

        # Try to get credentials, but don't require them yet
        asknews_client_id = get_from_dict_or_env(
            values, "asknews_client_id", "ASKNEWS_CLIENT_ID", default=None
        )
        asknews_client_secret = get_from_dict_or_env(
            values, "asknews_client_secret", "ASKNEWS_CLIENT_SECRET", default=None
        )
        asknews_api_key = get_from_dict_or_env(
            values, "asknews_api_key", "ASKNEWS_API_KEY", default=None
        )

        # Determine which authentication method to use
        has_client_credentials = bool(asknews_client_id and asknews_client_secret)
        has_api_key = bool(asknews_api_key)

        if has_client_credentials and has_api_key:
            raise ValueError(
                "Both client credentials and api_key provided. Please provide "
                "either client credentials or API key, not both."
            )

        if not has_client_credentials and not has_api_key:
            raise ValueError(
                "No authentication credentials provided. Please provide either "
                "asknews_client_id/asknews_client_secret or asknews_api_key."
            )

        # Initialize SDK with appropriate authentication method
        if has_api_key:
            an_sync = asknews_sdk.AskNewsSDK(
                api_key=asknews_api_key,
                scopes=["news"],
            )
            an_async = asknews_sdk.AsyncAskNewsSDK(
                api_key=asknews_api_key,
                scopes=["news"],
            )
        else:
            an_sync = asknews_sdk.AskNewsSDK(
                client_id=asknews_client_id,
                client_secret=asknews_client_secret,
                scopes=["news"],
            )
            an_async = asknews_sdk.AsyncAskNewsSDK(
                client_id=asknews_client_id,
                client_secret=asknews_client_secret,
                scopes=["news"],
            )

        values["asknews_sync"] = an_sync
        values["asknews_async"] = an_async
        values["asknews_client_id"] = asknews_client_id
        values["asknews_client_secret"] = asknews_client_secret
        values["asknews_api_key"] = asknews_api_key

        return values

    def search_news(
        self, query: str, max_results: int = 10, hours_back: int = 0
    ) -> str:
        """Search news in AskNews API synchronously."""

        response = self.asknews_sync.news.search_news(
            query=query,
            n_articles=max_results,
            method="kw",
            hours_back=hours_back,
            return_type="string",
        )
        return response.as_string

    async def asearch_news(
        self, query: str, max_results: int = 10, hours_back: int = 0
    ) -> str:
        """Search news in AskNews API asynchronously."""
        response = await self.asknews_async.news.search_news(
            query=query,
            n_articles=max_results,
            method="kw",
            hours_back=hours_back,
            return_type="string",
        )
        return response.as_string
