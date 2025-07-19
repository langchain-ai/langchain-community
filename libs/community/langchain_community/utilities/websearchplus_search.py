"""Util that calls WebsearchPlus Search.

No setup required. Free.
https://www.websearch.plus/
"""
import logging
from typing import Any, Dict, List, Optional, Literal

from langchain_core.utils import get_from_dict_or_env
from pydantic import BaseModel, ConfigDict, model_validator, Field, SecretStr
import httpx



class WebSearchOptions(BaseModel):
    language: Optional[str] = Field(
        default=None,
        description="Language code (e.g., 'en' for English, 'zh-cn' for Chinese)"
    )
    qdr: Optional[str] = Field(
        default="any",
        pattern="^(any|h|d|w|m|y)$",
        description="Time range filter: h=1 hour, d=1 day, w=1 week, m=1 month, y=1 year, any=no limit"
    )
    type: Optional[str] = Field(
        default="search",
        pattern="^(search|news)$",
        description="Search type: 'search' for general web search, 'news' for news-specific search"
    )
    result_type: Optional[str] = Field(
        default="text",
        pattern="^(list|text)$",
        description="Output format: 'list' for structured results, 'text' for plain text (required for tool call responses)"
    )
    mode: Optional[str] = Field(
        default="smart",
        pattern="^(smart|full)$",
        description="Search mode: 'smart' for summarized content, 'full' for complete content extraction"
    )


class WebSearchOptionsDefinition(BaseModel):
    search_context_size: Optional[Literal["low", "medium", "high"]] = Field(
        default="medium",
        description="Number of results to include in context: low=1~2, medium=5, high=10"
    )
    options: Optional[WebSearchOptions] = Field(
        default=None,
        description="Advanced search option settings"
    )


class WebSearchPlusInput(WebSearchOptionsDefinition):
    """Input model for the WebSearchPlus tool."""
    query: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="The query string to search for"
    )
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebSearchPlusAPIWrapper(BaseModel):
    
    websearchplus_api_key: SecretStr

    model_config = ConfigDict(
        extra="forbid",
    )
    base_url: str = "https://api.websearch.plus/v1/web_search_plus"
    
    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that api key and endpoint exists in environment."""
        websearchplus_api_key = get_from_dict_or_env(
            values, "websearchplus_api_key", "WEBSEARCHPLUS_API_KEY"
        )
        values["websearchplus_api_key"] = websearchplus_api_key

        return values
    
    def _run(self, input: dict[str, Any],  **kwargs: Any) -> Any:
        """Run the search query and return results.
        
        Args:
            input: A dictionary containing the search query and options.
            **kwargs: Additional keyword arguments.
            
        Returns:
            List of search results or error information.
        """
        # Here you would implement the actual API call to WebSearchPlus
        
        try:
            logger.info(f"🔍 Web search for: {input}")
            headers = {
                "Authorization": f"Bearer {self.websearchplus_api_key.get_secret_value()}",
                "Content-Type": "application/json"
            }
            with httpx.Client() as client:
                resp = client.post(
                    self.base_url,
                    json=input,
                    headers=headers,
                    timeout=1000
                )
                resp.raise_for_status()
                data = resp.json()
                if data.get("status") == "completed":
                    return data.get("results", [])
                return [{"error": f"Search failed: {data.get('status')}"}]
        except Exception as e:
            logger.error(f"Search error: {e}")
            return [{"error": f"Search failed: {str(e)}"}]