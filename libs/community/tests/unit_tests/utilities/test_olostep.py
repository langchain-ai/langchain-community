"""Unit tests for Olostep utility wrapper."""

from langchain_community.utilities.olostep import OlostepAPIWrapper


def test_api_wrapper_api_key_not_visible() -> None:
    """Test that the API key is not visible in repr."""
    wrapper = OlostepAPIWrapper(olostep_api_key="abcd123secretkey")  # type: ignore[arg-type]
    assert "abcd123secretkey" not in repr(wrapper)


def test_api_wrapper_headers() -> None:
    """Test that headers are correctly generated."""
    wrapper = OlostepAPIWrapper(olostep_api_key="test_key")  # type: ignore[arg-type]
    headers = wrapper._get_headers()
    assert "Authorization" in headers
    assert headers["Authorization"] == "Bearer test_key"
    assert headers["Content-Type"] == "application/json"
