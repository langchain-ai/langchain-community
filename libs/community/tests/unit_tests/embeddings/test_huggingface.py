import sys
from typing import TYPE_CHECKING

import pytest

from langchain_community.embeddings.huggingface import (
    HuggingFaceInferenceAPIEmbeddings,
    HuggingFaceInstructEmbeddings,
)

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


def test_hugginggface_inferenceapi_embedding_documents_init() -> None:
    """Test huggingface embeddings."""
    embedding = HuggingFaceInferenceAPIEmbeddings(api_key="abcd123")  # type: ignore[arg-type]
    assert "abcd123" not in repr(embedding)


def test_instruct_embeddings_import_error(mocker: "MockerFixture") -> None:
    """Test that initializing HuggingFaceInstructEmbeddings raises an error
    if InstructorEmbedding is not installed.

    Args:
        mocker: Fixture for patching sys.modules to simulate missing package.
    """
    # Simulate the 'InstructorEmbedding' package not being installed
    mocker.patch.dict(sys.modules, {"InstructorEmbedding": None})

    # Assert that an ImportError is raised
    with pytest.raises(ImportError) as exc_info:
        HuggingFaceInstructEmbeddings()

    # Assert that the error message contains the helpful text
    assert "pip install InstructorEmbedding" in str(exc_info.value)
