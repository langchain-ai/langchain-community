from langchain_community.chat_models.openai import ChatOpenAI


def test_openai_api_key_repr_masked() -> None:
    chat = ChatOpenAI.model_construct(
        openai_api_key="sk-test-key",
        model_name="gpt-4o-mini",
        model_kwargs={},
        streaming=False,
        n=1,
        temperature=0.7,
    )

    assert "sk-test-key" not in repr(chat)
    assert "openai_api_key='**********'" in repr(chat)


def test_client_params_uses_unmasked_secret_value_when_needed() -> None:
    chat = ChatOpenAI.model_construct(
        openai_api_key="sk-test-key",
        openai_api_base="https://example.test",
        openai_organization="org-test",
        model_name="gpt-4o-mini",
        model_kwargs={},
        streaming=False,
        n=1,
        temperature=0.7,
    )

    # This code path is used for openai<1 compatibility.
    from langchain_community.chat_models import openai as openai_module

    original = openai_module.is_openai_v1
    openai_module.is_openai_v1 = lambda: False
    try:
        params = chat._client_params
    finally:
        openai_module.is_openai_v1 = original

    assert params["api_key"] == "sk-test-key"
