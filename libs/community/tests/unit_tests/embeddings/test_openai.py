import os
from unittest.mock import patch

import pytest

from langchain_community.embeddings.openai import OpenAIEmbeddings

os.environ["OPENAI_API_KEY"] = "foo"


@pytest.mark.requires("openai")
def test_openai_invalid_model_kwargs() -> None:
    with pytest.raises(ValueError):
        OpenAIEmbeddings(model_kwargs={"model": "foo"})


@pytest.mark.requires("openai")
def test_openai_incorrect_field() -> None:
    with pytest.warns(match="not default parameter"):
        llm = OpenAIEmbeddings(foo="bar", openai_api_key="foo")  # type: ignore[call-arg]
    assert llm.model_kwargs == {"foo": "bar"}


@pytest.mark.requires("openai")
def test_embed_documents_with_custom_chunk_size() -> None:
    with (
        patch("openai.OpenAI") as mock_openai_class,
        patch("tiktoken.encoding_for_model") as mock_tiktoken,
    ):
        mock_client = mock_openai_class.return_value
        mock_embeddings_client = mock_client.embeddings

        # Mock tiktoken encoding
        mock_encoding = mock_tiktoken.return_value
        mock_encoding.encode.side_effect = [
            [1342, 19],
            [1342, 19],
            [1342, 19],
            [1342, 19],
        ]

        embeddings = OpenAIEmbeddings(chunk_size=2)
        texts = ["text1", "text2", "text3", "text4"]
        custom_chunk_size = 3

        mock_embeddings_client.create.side_effect = [
            {"data": [{"embedding": [0.1, 0.2]}, {"embedding": [0.3, 0.4]}]},
            {"data": [{"embedding": [0.5, 0.6]}, {"embedding": [0.7, 0.8]}]},
        ]

        embeddings.embed_documents(texts, chunk_size=custom_chunk_size)

        # Verify the expected token inputs - this should be called twice
        # with chunk_size=3
        assert mock_embeddings_client.create.call_count == 2
        mock_embeddings_client.create.assert_any_call(
            input=[[1342, 19]], **embeddings._invocation_params
        )
        mock_embeddings_client.create.assert_any_call(
            input=[[1342, 19]], **embeddings._invocation_params
        )
