"""Volcengine Maas chat model integration using OpenAI protocol."""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Mapping, Optional, Type, Union

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env, pre_init
from pydantic import BaseModel, Field, SecretStr

from langchain_community.chat_models.openai import (
    ChatOpenAI,
    _convert_delta_to_message_chunk,
)


class VolcEngineMaasChat(ChatOpenAI):
    """Volcengine Maas chat model integration using OpenAI-compatible API.

    Setup:
        Install ``openai`` and set environment variable ``VOLC_API_KEY``.

        .. code-block:: bash

            pip install openai
            export VOLC_API_KEY="your-api-key"

    Key init args — completion params:
        model: str
            Name of Volcengine model to use (or endpoint ID).
        temperature: float
            Sampling temperature.
        max_tokens: Optional[int]
            Max number of tokens to generate.

    Key init args — client params:
        api_key: Optional[str]
            Volcengine API KEY. If not passed in will be read from env var VOLC_API_KEY.
        base_url: Optional[str]
            Base URL for API requests. Default is Volcengine ARK endpoint.

    See full list of supported init args and their descriptions in the params section.

    Instantiate:
        .. code-block:: python

            from langchain_community.chat_models import VolcEngineMaasChat

            chat = VolcEngineMaasChat(
                model="your-endpoint-id",
                api_key="your-api-key",
                base_url="https://ark.cn-beijing.volces.com/api/v3",
                # other params...
            )

    Invoke:
        .. code-block:: python

            messages = [
                ("system", "你是一个有帮助的AI助手。"),
                ("human", "你好，请介绍一下自己。"),
            ]
            chat.invoke(messages)

        .. code-block:: python

            AIMessage(
                content='你好！我是一个AI助手...',
                additional_kwargs={},
                response_metadata={
                    'token_usage': {
                        'completion_tokens': 20,
                        'prompt_tokens': 15,
                        'total_tokens': 35
                    },
                    'model_name': 'your-model',
                    'finish_reason': 'stop'
                },
                id='run-...'
            )

    Stream:
        .. code-block:: python

            for chunk in chat.stream(messages):
                print(chunk.content, end="", flush=True)

    Async:
        .. code-block:: python

            await chat.ainvoke(messages)

            # stream:
            # async for chunk in chat.astream(messages):
            #    print(chunk.content, end="", flush=True)

            # batch:
            # await chat.abatch([messages])

    Reasoning content support:
        For models that support reasoning (like deepseek-reasoner), the reasoning
        content will be available in the response metadata:

        .. code-block:: python

            response = chat.invoke(messages)
            if "reasoning_content" in response.additional_kwargs:
                print("Reasoning:", response.additional_kwargs["reasoning_content"])
    """

    volc_api_key: SecretStr = Field(default=SecretStr(""))
    """Volcengine API key."""

    volc_api_base: str = Field(default="https://ark.cn-beijing.volces.com/api/v3")
    """Base URL for Volcengine API requests."""

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"volc_api_key": "VOLC_API_KEY"}

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "volcengine-maas-chat"

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by Langchain."""
        return False

    @pre_init
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key exists and initialize OpenAI client."""
        values["volc_api_key"] = convert_to_secret_str(
            get_from_dict_or_env(
                values,
                ["volc_api_key", "api_key", "openai_api_key"],
                "VOLC_API_KEY",
            )
        )

        values["volc_api_base"] = get_from_dict_or_env(
            values,
            ["volc_api_base", "base_url", "openai_api_base"],
            "VOLC_API_BASE",
            default="https://ark.cn-beijing.volces.com/api/v3",
        )

        try:
            import openai
        except ImportError:
            raise ImportError(
                "Could not import openai python package. "
                "Please install it with `pip install openai`."
            )

        # Initialize OpenAI client with Volcengine endpoint
        client_params = {
            "api_key": values["volc_api_key"].get_secret_value(),
            "base_url": values["volc_api_base"],
        }

        if not values.get("client"):
            values["client"] = openai.OpenAI(**client_params).chat.completions
        if not values.get("async_client"):
            values["async_client"] = openai.AsyncOpenAI(
                **client_params
            ).chat.completions

        return values

    def _create_chat_result(self, response: Union[dict, BaseModel]) -> ChatResult:
        """Create chat result with reasoning_content support.

        This method extends the parent class to extract reasoning_content
        from the response if available (similar to Tongyi's implementation).
        """
        generations = []
        if not isinstance(response, dict):
            response = response.dict()

        for res in response["choices"]:
            message_dict = res["message"]
            additional_kwargs: Dict[str, Any] = {}

            # Extract reasoning_content if present
            if "reasoning_content" in message_dict:
                additional_kwargs["reasoning_content"] = message_dict[
                    "reasoning_content"
                ]

            # Extract tool_calls if present
            if "tool_calls" in message_dict:
                additional_kwargs["tool_calls"] = message_dict["tool_calls"]

            # Create AIMessage with reasoning_content in additional_kwargs
            message = AIMessage(
                content=message_dict.get("content", ""),
                additional_kwargs=additional_kwargs,
            )

            generation_info = dict(finish_reason=res.get("finish_reason"))
            if "logprobs" in res:
                generation_info["logprobs"] = res["logprobs"]

            gen = ChatGeneration(
                message=message,
                generation_info=generation_info,
            )
            generations.append(gen)

        token_usage = response.get("usage", {})
        llm_output = {
            "token_usage": token_usage,
            "model_name": self.model_name,
            "system_fingerprint": response.get("system_fingerprint", ""),
        }
        return ChatResult(generations=generations, llm_output=llm_output)

    @staticmethod
    def _convert_delta_to_message_chunk_with_reasoning(
        _dict: Mapping[str, Any], default_class: Type[BaseMessageChunk]
    ) -> BaseMessageChunk:
        """Convert delta response to message chunk with reasoning_content support.

        This method extends the parent class to handle reasoning_content
        in streaming responses, similar to Tongyi's implementation.

        Args:
            _dict: Delta response dictionary from the API.
            default_class: Default message chunk class to use.

        Returns:
            Appropriate message chunk with reasoning_content if present.
        """
        role = _dict.get("role")
        content = _dict.get("content") or ""
        additional_kwargs: Dict[str, Any] = {}

        # Handle reasoning_content in streaming (similar to Tongyi)
        if "reasoning_content" in _dict:
            additional_kwargs["reasoning_content"] = _dict["reasoning_content"]

        # Handle function_call
        if _dict.get("function_call"):
            function_call = dict(_dict["function_call"])
            if "name" in function_call and function_call["name"] is None:
                function_call["name"] = ""
            additional_kwargs["function_call"] = function_call

        # Handle tool_calls
        if _dict.get("tool_calls"):
            additional_kwargs["tool_calls"] = _dict["tool_calls"]

        # Return appropriate message chunk type
        if role == "assistant" or default_class == AIMessageChunk:
            return AIMessageChunk(content=content, additional_kwargs=additional_kwargs)

        # Fall back to parent implementation for other message types
        return _convert_delta_to_message_chunk(_dict, default_class)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream chat completions with reasoning_content support.

        This method overrides the parent to use our custom delta converter
        that handles reasoning_content in streaming responses.

        Args:
            messages: List of messages to send to the model.
            stop: Optional list of stop sequences.
            run_manager: Optional callback manager.
            **kwargs: Additional parameters to pass to the API.

        Yields:
            Chat generation chunks with reasoning_content if available.
        """
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, "stream": True}

        default_chunk_class: Type[BaseMessageChunk] = AIMessageChunk
        for chunk in self.completion_with_retry(
            messages=message_dicts, run_manager=run_manager, **params
        ):
            if not isinstance(chunk, dict):
                chunk = chunk.dict()
            if len(chunk["choices"]) == 0:
                continue
            choice = chunk["choices"][0]
            if choice["delta"] is None:
                continue

            # Use our custom converter that handles reasoning_content
            chunk_msg = self._convert_delta_to_message_chunk_with_reasoning(
                choice["delta"], default_chunk_class
            )
            finish_reason = choice.get("finish_reason")
            generation_info = (
                dict(finish_reason=finish_reason) if finish_reason is not None else None
            )
            default_chunk_class = chunk_msg.__class__
            cg_chunk = ChatGenerationChunk(
                message=chunk_msg, generation_info=generation_info
            )
            if run_manager:
                run_manager.on_llm_new_token(cg_chunk.text, chunk=cg_chunk)
            yield cg_chunk
