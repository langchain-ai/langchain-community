"""Standard LangChain interface tests"""

from typing import Tuple, Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_tests.unit_tests import ChatModelUnitTests

from langchain_community.chat_models.clarifai import DEFAULT_MODEL_URL, ChatClarifai


@pytest.mark.requires("clarifai")
class TestClarifaiStandard(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatClarifai

    @property
    def chat_model_params(self) -> dict:
        return {"pat": "clarifai-pat", "model_url": DEFAULT_MODEL_URL}

    @property
    def init_from_env_params(self) -> Tuple[dict, dict, dict]:
        return ({}, {}, {})
