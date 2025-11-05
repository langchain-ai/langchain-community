"""Test MLX Chat wrapper."""

from importlib import import_module
from typing import Any

import pytest
from langchain_core.messages import HumanMessage

from langchain_community.chat_models.mlx import ChatMLX
from langchain_community.llms.mlx_pipeline import MLXPipeline


class _FakeTokenizer:
    def __init__(self) -> None:
        self.tools = None

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=True,
        return_tensors=None,
        tools=None,
    ) -> str:
        self.tools = tools
        return "prompt"


class _FakeLLM(MLXPipeline):
    model_id: str = "fake-model"
    model: Any = None
    tokenizer: Any = None
    pipeline_kwargs: dict = {}

    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer()

    def _generate(self, prompts, stop=None, run_manager=None, **kwargs):
        class _Res:
            generations = [[type("G", (), {"text": "", "generation_info": {}})]]
            llm_output = {}

        return _Res()

    async def _agenerate(self, prompts, stop=None, run_manager=None, **kwargs):
        return self._generate(prompts, stop=stop, run_manager=run_manager, **kwargs)


def test_import_class() -> None:
    """Test that the class can be imported."""
    module_name = "langchain_community.chat_models.mlx"
    class_name = "ChatMLX"

    module = import_module(module_name)
    assert hasattr(module, class_name)


def test_generate_passes_tools_to_tokenizer() -> None:
    llm = _FakeLLM()
    chat = ChatMLX(llm=llm)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "foo",
                "description": "",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    chat._generate([HumanMessage(content="hi")], tools=tools)
    assert llm.tokenizer.tools == tools


@pytest.mark.asyncio
async def test_agenerate_passes_tools_to_tokenizer() -> None:
    llm = _FakeLLM()
    chat = ChatMLX(llm=llm)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "foo",
                "description": "",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    await chat._agenerate([HumanMessage(content="hi")], tools=tools)
    assert llm.tokenizer.tools == tools
