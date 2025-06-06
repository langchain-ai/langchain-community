"""Test Clarifai API wrapper.
In order to run this test, you need to have an account on Clarifai.
You can sign up for free at https://clarifai.com/signup.
pip install clarifai

You'll need to set env variable CLARIFAI_PAT to your personal access token key.
"""

from typing import Generator

from langchain_community.llms.clarifai import Clarifai

MODEL_URL = "https://clarifai.com/qwen/qwenLM/models/Qwen3-30B-A3B-GGUF"


def test_clarifai_call() -> None:
    """Test valid call to clarifai."""
    llm = Clarifai(model_url=MODEL_URL)
    output = llm.invoke(
        "A chain is a serial assembly of connected pieces, called links, \
        typically made of metal, with an overall character similar to that\
        of a rope in that it is flexible and curved in compression but \
        linear, rigid, and load-bearing in tension. A chain may consist\
        of two or more links.",
        max_tokens=15,
        top_p=0.9,
    )

    assert isinstance(output, str)
    assert llm._llm_type == "clarifai"
    assert llm.model_id == "Qwen3-30B-A3B-GGUF"


def test_clarifai_streaming() -> None:
    """Test streaming tokens from Clairfai."""
    llm = Clarifai(model_url=MODEL_URL)
    generator = llm.stream(
        "How do you say 'hello' in German?", stop=["'"], max_tokens=15, top_p=0.9
    )
    stream_results_string = ""
    assert isinstance(generator, Generator)

    for chunk in generator:
        assert isinstance(chunk, str)
        stream_results_string += chunk
    assert len(stream_results_string.strip()) > 1
