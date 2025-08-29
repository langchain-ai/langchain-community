"""Test mixedbreadai cross encoders."""

from langchain_community.cross_encoders import MixedbreadAICrossEncoder


def _assert(encoder: MixedbreadAICrossEncoder) -> None:
    query = "I love you"
    texts = ["I love you", "I like you", "I don't like you", "I hate you"]
    output = encoder.score([(query, text) for text in texts])

    # Check that we got scores for all texts
    assert len(output) == len(texts)
    
    # Check that scores are in descending order (most relevant first)
    for i in range(len(texts) - 1):
        assert output[i] > output[i + 1], f"Score at index {i} ({output[i]}) should be greater than score at index {i+1} ({output[i+1]})"


def test_mixedbreadai_cross_encoder() -> None:
    """Test MixedbreadAICrossEncoder with default model."""
    encoder = MixedbreadAICrossEncoder()
    _assert(encoder)


def test_mixedbreadai_cross_encoder_with_designated_model_name() -> None:
    """Test MixedbreadAICrossEncoder with specific model."""
    encoder = MixedbreadAICrossEncoder(model_name="mixedbread-ai/mxbai-rerank-base-v1")
    _assert(encoder)


def test_mixedbreadai_cross_encoder_without_normalization() -> None:
    """Test MixedbreadAICrossEncoder with normalization disabled."""
    encoder = MixedbreadAICrossEncoder(normalize_scores=False)
    _assert(encoder)



def test_mixedbreadai_cross_encoder_multilingual() -> None:
    """Test MixedbreadAICrossEncoder with multilingual content."""
    encoder = MixedbreadAICrossEncoder()
    query = "¿Cómo afecta la agricultura al clima?"
    texts = [
        "El cambio climático provoca sequías e inundaciones que afectan los cultivos.",
        "Climate change leads to droughts and floods, affecting crop yields.",
        "Agriculture is impacted by rising temperatures and unpredictable weather."
    ]
    output = encoder.score([(query, text) for text in texts])
    
    assert len(output) == len(texts)
    # Spanish text should rank higher than English for Spanish query
    assert output[0] > output[1]  # Spanish agriculture text > English agriculture text
    # Agriculture-related content in the query language should rank higher
    assert output[2] > output[3]  # Agriculture text > Data scientist text
