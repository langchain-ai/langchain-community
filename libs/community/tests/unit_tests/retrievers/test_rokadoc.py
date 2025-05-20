from typing import Any, Dict

from pytest_mock import MockerFixture

from langchain_community.retrievers.rokadoc import RokadocSearchRetriever


class MockResponse:
    def __init__(self, json_data: Dict, status_code: int) -> None:
        self.json_data = json_data
        self.status_code = status_code

    def json(self) -> Dict:
        return self.json_data


def mocked_requests_post(*args: Any, **kwargs: Any) -> MockResponse:
    return MockResponse(
        json_data={
            "search_result": [
                {
                    "context": "検索対象のファイル名: sample.pdf\n====\n\n側のタッチされた選手及びキャッチングされた攻撃側の選手はアウトと...",  # noqa: E501
                    "conversion_id": "xxxx",
                    "page_number": 1,
                    "pdf_name": "sample.pdf",
                    "unit": {
                        "body": "1チーム10 - 12名で、各チーム7名がコートに入り2チームで争う。",  # noqa: E501
                        "description": "カバディのルールについて説明がされています。",
                        "images": [
                            {
                                "caption": "コート上に赤いユニフォームと青いユニフォームを着た人が立っています",  # noqa: E501
                                "coordinates": [[100, 800], [200, 900]],
                                "page": 1,
                            }
                        ],
                        "tables": [
                            {
                                "coordinates": [[100, 800], [200, 900]],
                                "page": 1,
                                "table": '<table border="1"><caption>テーブル1</caption><tr><th></th><th>ルール</th><th>攻撃側</th><th>守備側</th></tr></table>',  # noqa: E501
                            }
                        ],
                        "title": "カバディのルール",
                        "unit": 1,
                    },
                }
            ]
        },
        status_code=200,
    )


def test_mock_RokadocSearchRetriever_invoke(
    mocker: MockerFixture,
) -> None:
    "Test function for the retriever of the vector database used in the production environment of the Rokadoc public beta version, implemented with mocking for testing purposes."  # noqa: E501
    mocker.patch("requests.get", side_effect=mocked_requests_post)

    rokadoc_retriever = RokadocSearchRetriever(api_key="REAL API KEY")

    result = rokadoc_retriever.invoke("病院の日程に関して教えてください。")

    assert len(result) == 1
    assert (
        result[0].page_content
        == "検索対象のファイル名: sample.pdf\n====\n\n側のタッチされた選手及びキャッチングされた攻撃側の選手はアウトと..."  # noqa: E501
    )

    assert result[0].metadata["page_number"] == 1
    assert result[0].metadata["pdf_name"] == "sample.pdf"
