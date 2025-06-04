"""Util that calls Chatopera Bot. https://bot.chatopera.com/"""

from typing import Any, Dict, List, Optional

from langchain_core.utils import get_from_dict_or_env
from pydantic import BaseModel, ConfigDict, Field, model_validator
try:
    from chatopera import Chatbot
except ImportError as e:
    print(f"ImportError: {e}")
    print("Solution, install with `pip install chatopera`")

class ChatoperaBotAPIWrapper(BaseModel):
    """Wrapper for Chatopera Search API."""

    bot: Optional[Chatbot] = None
    chatopera_bot_client_id: str
    chatopera_bot_client_secret: str
    k: int = 10
    search_kwargs: dict = Field(default_factory=dict)
    """Additional keyword arguments to pass to the search request."""

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    def _chatopera_bot_results(self, search_term: str, count: int) -> List[dict]:
        self.bot = Chatbot(self.chatopera_bot_client_id, self.chatopera_bot_client_secret)
        resp = self.bot.command("POST", "/faq/query", dict({
            "query": search_term,
            "fromUserId": "agent"
        }))
        results = []
        if resp["rc"] == 0:
            for x in resp["data"]:
                # logging.info("%f match: %s, reply: %s", x["score"], x["post"], x["reply"])
                for y in x["replies"]:
                    results.append(y["content"])
                break
        return results

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that api key and endpoint exists in environment."""
        chatopera_bot_client_id = get_from_dict_or_env(
            values, "chatopera_bot_client_id", "CHATOPERA_BOT_CLIENT_ID"
        )
        values["chatopera_bot_client_id"] = chatopera_bot_client_id

        chatopera_bot_client_secret = get_from_dict_or_env(
            values, "chatopera_bot_client_secret", "CHATOPERA_BOT_CLIENT_SECRET"
        )
        values["chatopera_bot_client_secret"] = chatopera_bot_client_secret

        return values

    def run(self, query: str) -> str:
        """Run query through Chatopera Bot Services and parse result."""
        snippets = []
        results = self._chatopera_bot_results(query, count=self.k)
        if len(results) == 0:
            return "No good Result was found"
        for result in results:
            snippets.append(result["snippet"])

        return " ".join(snippets)

    def results(self, query: str, num_results: int) -> List[Dict]:
        """Run query through Chatopera Bot Services and return metadata.

        Args:
            query: The query to search for.
            num_results: The number of results to return.

        Returns:
            A list of string with the contents.
        """
        results = self._chatopera_bot_results(query, count=num_results)
        if len(results) == 0:
            return ["No good Result was found"]
        
        return results