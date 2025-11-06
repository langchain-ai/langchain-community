import pytest
from pydantic import SecretStr, ValidationError

from langchain_community.chat_models.ovhcloud import ChatOVHcloud

DEFAULT_API_BASE = "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1"
DEFAULT_MODEL = "gpt-oss-120b"


@pytest.mark.requires("openai")
def test__default_ovhcloud_api_base() -> None:
    chat = ChatOVHcloud(ovhcloud_api_token=SecretStr("test_token"))  # type: ignore[call-arg]
    assert chat.ovhcloud_api_base == DEFAULT_API_BASE


@pytest.mark.requires("openai")
def test__default_ovhcloud_api_token() -> None:
    chat = ChatOVHcloud(ovhcloud_api_token=SecretStr("test_token"))  # type: ignore[call-arg]
    assert chat.ovhcloud_api_token.get_secret_value() == "test_token"


@pytest.mark.requires("openai")
def test__default_model_name() -> None:
    chat = ChatOVHcloud(ovhcloud_api_token=SecretStr("test_token"))  # type: ignore[call-arg]
    assert chat.model_name == DEFAULT_MODEL


@pytest.mark.requires("openai")
def test__field_aliases() -> None:
    chat = ChatOVHcloud(ovhcloud_api_token=SecretStr("test_token"), model="custom-model")  # type: ignore[call-arg]
    assert chat.model_name == "custom-model"
    assert chat.ovhcloud_api_token.get_secret_value() == "test_token"


@pytest.mark.requires("openai")
def test__missing_ovhcloud_api_token() -> None:
    with pytest.raises(ValidationError) as e:
        ChatOVHcloud()
    assert "Did not find ovhcloud_api_token" in str(e)


@pytest.mark.requires("openai")
def test__all_fields_provided() -> None:
    chat = ChatOVHcloud(  # type: ignore[call-arg]
        ovhcloud_api_token=SecretStr("test_token"),
        model="custom-model",
        ovhcloud_api_base="https://custom.api/base/",
    )
    assert chat.ovhcloud_api_base == "https://custom.api/base/"
    assert chat.ovhcloud_api_token.get_secret_value() == "test_token"
    assert chat.model_name == "custom-model"

