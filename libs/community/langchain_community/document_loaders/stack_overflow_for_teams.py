from typing import List, Iterable
from langchain_core.documents import Document

import requests
from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader


class StackOverflowTeamsApiV3Loader(BaseLoader):
    """Load documents from StackOverflow API v3."""

    def __init__(
        self,
        access_token: str,
        content_type: str,
        team: str = "",
        date_from: str = "",
        sort: str = "",
        order: str = "",
        endpoint: str = "api.stackoverflowteams.com"
    ) -> None:
        """
        Initializes the StackOverflowTeamsApiV3Loader with necessary parameters.

        Args:
            access_token (str): OAuth2 access token for authentication.
            team (str): The team identifier to fetch documents for.
            content_type (str): The type of content to fetch (e.g., "questions", "answers").
            date_from (str, optional): Filter results from this date (ISO 8601 format).
            sort (str, optional): Sort order for results (e.g., "creation", "activity").
            order (str, optional): Order direction ("asc" or "desc").
            endpoint (str, optional): API endpoint to use, default is "api.stackoverflowteams.com".

        Raises:
            ValueError: If any required parameter is missing or invalid.
            ImportError: If the requests library is not installed.
        """
        try:
            import requests
        except ImportError as e:
            raise ImportError(
                "Cannot import requests, please install with `pip install requests`."
            ) from e

        if not access_token:
            raise ValueError("access_token must be provided.")

        self.access_token = access_token
        self.team = team
        self.content_type = content_type
        self.date_from = date_from
        self.endpoint = endpoint
        self.sort = sort
        self.order = order

    def load(self) -> list[Document]:
        """Load documents from StackOverflow API v3."""

        results: List[Document] = []
        docs = self._doc_loader()
        results.extend(docs)

        return results

    def _doc_loader(self) -> Iterable[Document]:
        headers = {"Authorization": f"Bearer {self.access_token}"}

        # build our initial params
        params = {"page": "1"}
        # add any optional params
        if self.date_from:
            params["from"] = self.date_from
        if self.sort:
            params["sort"] = self.sort
        if self.order:
            params["order"] = self.order

        # handle team
        if self.team:
            fullEndpoint = f"https://{self.endpoint}/v3/teams/{self.team}/{self.content_type}"
        else:
            fullEndpoint = f"https://{self.endpoint}/v3/{self.content_type}"

        results = True
        while results:
            print(f"Loading page {params['page']}...")
            response = requests.get(
                fullEndpoint,
                headers=headers,
                timeout=30,
                params=params
            )

            response.raise_for_status()

            json_data = response.json()

            results = json_data.get("items", [])

            for item in results:
                metadata = {
                    "id": item["id"],
                    "type": item["type"],
                    "title": item["title"],
                    "creationDate": item["creationDate"],
                    "lastActivityDate": item["lastActivityDate"],
                    "viewCount": item["viewCount"],
                    "webUrl": item["webUrl"],
                    "isDeleted": item["isDeleted"],
                    "isObsolete": item["isObsolete"],
                    "isClosed": item["isClosed"],
                }

                yield Document(
                    page_content=item["body"],
                    metadata=metadata
                )

            # increment the page number
            params["page"] = str(int(params["page"]) + 1)
