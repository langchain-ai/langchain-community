import os

from langchain_community.retrievers.rokadoc import RokadocSearchRetriever


class TestRokadocSearchRetriever:
    @classmethod
    def setup_class(cls) -> None:
        if not os.getenv("ROKADOC_API_KEY"):
            raise ValueError("ROKADOC_API_KEY environment variable is not set")

    def test_invoke(self) -> None:
        rokadoc_retriever = RokadocSearchRetriever(api_key=os.getenv("ROKADOC_API_KEY"))

        actual = rokadoc_retriever.invoke("病院の日程に関して教えてください。")

        assert len(actual) > 0
