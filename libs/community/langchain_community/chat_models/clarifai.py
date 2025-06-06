"""Clarifai chat wrapper."""

from __future__ import annotations

import json
import logging
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
    Union,
)

from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    generate_from_stream,
)
from langchain_core.language_models.llms import create_base_retry_decorator
from langchain_core.messages import (
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessageChunk,
    FunctionMessageChunk,
    HumanMessageChunk,
    SystemMessageChunk,
    ToolMessageChunk,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.utils import (
    get_from_dict_or_env,
    get_pydantic_field_names,
    pre_init,
)
from pydantic import BaseModel, Field, model_validator

from langchain_community.adapters.openai import (
    convert_dict_to_message,
    convert_message_to_dict,
)



logger = logging.getLogger(__name__)

DEFAULT_MODEL_URL = "https://clarifai.com/meta/Llama-3/models/Llama-3_2-3B-Instruct"

def _create_retry_decorator(
    llm: ChatClarifai,
    run_manager: Optional[
        Union[AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun]
    ] = None,
) -> Callable[[Any], Any]:

    errors = [
        Exception
    ]
    return create_base_retry_decorator(
        error_types=errors, max_retries=llm.max_retries, run_manager=run_manager
    )


async def acompletion_with_retry(
    llm: ChatClarifai,
    run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    **kwargs: Any,
) -> Any:
    """Use tenacity to retry the async completion call."""

    retry_decorator = _create_retry_decorator(llm, run_manager=run_manager)

    @retry_decorator
    async def _completion_with_retry(**kwargs: Any) -> Any:
        return await llm.client.acreate(**kwargs)

    return await _completion_with_retry(**kwargs)


def _convert_delta_to_message_chunk(
    _dict: Mapping[str, Any], default_class: Type[BaseMessageChunk]
) -> BaseMessageChunk:
    role = _dict.get("role")
    content = _dict.get("content") or ""
    additional_kwargs: Dict = {}
    if _dict.get("function_call"):
        function_call = dict(_dict["function_call"])
        if "name" in function_call and function_call["name"] is None:
            function_call["name"] = ""
        additional_kwargs["function_call"] = function_call
    if _dict.get("tool_calls"):
        additional_kwargs["tool_calls"] = _dict["tool_calls"]

    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content)
    elif role == "assistant" or default_class == AIMessageChunk:
        return AIMessageChunk(content=content, additional_kwargs=additional_kwargs)
    elif role == "system" or default_class == SystemMessageChunk:
        return SystemMessageChunk(content=content)
    elif role == "function" or default_class == FunctionMessageChunk:
        return FunctionMessageChunk(content=content, name=_dict["name"])
    elif role == "tool" or default_class == ToolMessageChunk:
        return ToolMessageChunk(content=content, tool_call_id=_dict["tool_call_id"])
    elif role or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role)  # type: ignore[arg-type]
    else:
        return default_class(content=content)  # type: ignore[call-arg]


def _update_token_usage(
    overall_token_usage: Union[int, dict], new_usage: Union[int, dict]
) -> Union[int, dict]:
    # Token usage is either ints or dictionaries
    # `reasoning_tokens` is nested inside `completion_tokens_details`
    if isinstance(new_usage, int):
        if not isinstance(overall_token_usage, int):
            raise ValueError(
                f"Got different types for token usage: "
                f"{type(new_usage)} and {type(overall_token_usage)}"
            )
        return new_usage + overall_token_usage
    elif isinstance(new_usage, dict):
        if not isinstance(overall_token_usage, dict):
            raise ValueError(
                f"Got different types for token usage: "
                f"{type(new_usage)} and {type(overall_token_usage)}"
            )
        return {
            k: _update_token_usage(overall_token_usage.get(k, 0), v)
            for k, v in new_usage.items()
        }
    else:
        warnings.warn(f"Unexpected type for token usage: {type(new_usage)}")
        return new_usage



class ChatClarifai(BaseChatModel):
    """Clarifai Chat models."""

    model_url: Optional[str] = None
    """Model url to use."""
    model_id: Optional[str] = None
    """Model id to use."""
    model_version_id: Optional[str] = None
    """Model version id to use."""
    app_id: Optional[str] = None
    """Clarifai application id to use."""
    user_id: Optional[str] = None
    """Clarifai user id to use."""
    pat: Optional[str] = Field(default=None, exclude=True)  #: :meta private:
    """Clarifai personal access token to use."""
    token: Optional[str] = Field(default=None, exclude=True)  #: :meta private:
    """Clarifai session token to use."""
    api_base: str = "https://api.clarifai.com"
    """Clarifai API base url"""

    deployment_id: Optional[str] = None
    nodepool_id: Optional[str] = None
    compute_cluster_id: Optional[str] = None
    """Clarifai deployment params"""
    
    max_retries: int = Field(default=1)
    
    streaming: bool = False
    """Whether to stream the results or not."""

    model_kwargs: dict = Field(
        default_factory=lambda: {
            "temperature": 0.7,
            "max_tokens": 512,
            "top_p": 0.95,
        }.copy()
    )
    
    model_name: str = None
    
    model: Any = Field(default=None, exclude=True)  #: :meta private:
    
    
    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"pat": "CLARIFAI_PAT"}


    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "chat_models", "clarifai"]


    @model_validator(mode="before")
    @classmethod
    def build_extra(cls, values: Dict[str, Any]) -> Any:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = get_pydantic_field_names(cls)
        extra = values.get("model_kwargs", {})
        for field_name in list(values):
            if field_name in extra:
                raise ValueError(f"Found {field_name} supplied twice.")
            if field_name not in all_required_field_names:
                logger.warning(
                    f"""WARNING! {field_name} is not default parameter.
                    {field_name} was transferred to model_kwargs.
                    Please confirm that {field_name} is what you intended."""
                )
                extra[field_name] = values.pop(field_name)

        invalid_model_kwargs = all_required_field_names.intersection(extra.keys())
        if invalid_model_kwargs:
            raise ValueError(
                f"Parameters {invalid_model_kwargs} should be specified explicitly. "
                f"Instead they were passed in as part of `model_kwargs` parameter."
            )

        values["model_kwargs"] = extra
        return values
    

    @pre_init
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that we have all required info to access Clarifai
        platform and python package exists in environment."""
        try:
            from clarifai.client import Model
        except ImportError:
            raise ImportError(
                "Could not import clarifai python package. "
                "Please install it with `pip install clarifai`."
            )
        
        values["pat"] = get_from_dict_or_env(
            values, "pat", "CLARIFAI_PAT"
        )
        
        user_id = values.get("user_id")
        app_id = values.get("app_id")
        model_id = values.get("model_id")
        model_version_id = values.get("model_version_id")
        model_url = values.get("model_url")
        api_base = values.get("api_base")
        pat = values.get("pat")
        token = values.get("token")
        
        deployment_id = values.get("deployment_id")
        nodepool_id = values.get("nodepool_id")
        compute_cluster_id = values.get("compute_cluster_id")
        
        model = values.get("model")
        if not model or not isinstance(model, Model):
            model = Model(
                url=model_url,
                app_id=app_id,
                user_id=user_id,
                model_version=dict(id=model_version_id),
                pat=pat,
                token=token,
                model_id=model_id,
                base_url=api_base,
                deployment_id=deployment_id,
                nodepool_id=nodepool_id,
                compute_cluster_id=compute_cluster_id
            )
            
            values["model"] = model
        
        if not model_id:
            values["model_id"] = model.id
            values["model_name"] = model.id
        if not user_id:
            values["user_id"] = model.user_app_id.user_id
        if not model_id:
            values["app_id"] = model.user_app_id.app_id
        if not model_version_id:
            values["model_version_id"] = model.model_version.id
            
        
        return values

    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters.

        This information is used by the LangChain callback system, which
        is used for tracing purposes make it possible to monitor LLMs.
        """
        return {
            # The model name allows users to specify custom token counting
            # rules in LLM monitoring applications (e.g., in LangSmith users
            # can provide per token pricing for their model and monitor
            # costs for the given LLM.)
            "model_name": self.model_name,
            **self.model_kwargs,
        }

    def _create_message_dicts(
        self, messages: List[BaseMessage]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """ Preprocess LC to OpenAI Chat """
        message_dicts = [convert_message_to_dict(m) for m in messages]
        return message_dicts


    def _create_chat_result(self, response: Union[dict, BaseModel]) -> ChatResult:
        """ Postprocess OpenAI Chat to LC chat """
        
        generations = []
        if isinstance(response, BaseModel):
            response = response.model_dump()
        for res in response["choices"]:
            message = convert_dict_to_message(res["message"])
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
            "model_name": self.model_id,
        }
        return ChatResult(generations=generations, llm_output=llm_output)


    def _combine_llm_outputs(self, llm_outputs: List[Optional[dict]]) -> dict:
        overall_token_usage: dict = {}
        for output in llm_outputs:
            if output is None:
                # Happens in streaming
                continue
            token_usage = output["token_usage"]
            if token_usage is not None:
                for k, v in token_usage.items():
                    if k in overall_token_usage:
                        overall_token_usage[k] = _update_token_usage(
                            overall_token_usage[k], v
                        )
                    else:
                        overall_token_usage[k] = v

        combined = {"token_usage": overall_token_usage, "model_id": self.model_id}

        return combined


    # -------------- Generate ----------------- #
    def completion_with_retry(
        self, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any
    ) -> Any:
        """Use tenacity to retry the completion call."""

        retry_decorator = _create_retry_decorator(self, run_manager=run_manager)

        @retry_decorator
        def _completion_with_retry(**kwargs: Any) -> Any:
            
            params = {**self.model_kwargs}
            params.update(kwargs) 
            params.pop("stream", None)
            
            json_params = json.dumps(params)
            chat_response = self.model.openai_transport(msg=json_params)
            chat_response = json.loads(chat_response)
            
            return chat_response

        return _completion_with_retry(**kwargs)
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        This can be a call to an API, a call to a local model, or any other
        implementation that generates a response to the input prompt.

        Args:
            messages: the prompt composed of a list of messages.
            stop: a list of strings on which the model should stop generating.
                  If generation stops due to a stop token, the stop token itself
                  SHOULD BE INCLUDED as part of the output. This is not enforced
                  across models right now, but it's a good practice to follow since
                  it makes it much easier to parse the output of the model
                  downstream and understand why generation stopped.
            run_manager: A run manager with callbacks for the LLM.
        """
        
        should_stream = stream if stream is not None else self.streaming
        if should_stream:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)
        
        message_dicts = self._create_message_dicts(messages)
        response = self.completion_with_retry(
            messages=message_dicts, run_manager=run_manager, **kwargs
        )
        return self._create_chat_result(response)
    
    
    # -------------- Stream ----------------- #
    
    def stream_completion_with_retry(
        self, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any
    ) -> Any:
        """Use tenacity to retry the completion call."""

        retry_decorator = _create_retry_decorator(self, run_manager=run_manager)

        @retry_decorator
        def _completion_with_retry(**kwargs: Any) -> Any:
            params = {**self.model_kwargs}
            params.update(kwargs)
            params["stream"] = True
            json_params = json.dumps(params)
            chat_response_iteration = self.model.openai_stream_transport(msg=json_params)

            return chat_response_iteration

        return _completion_with_retry(**kwargs)
    
    
    def _stream(
        self,
        messages: List[BaseMessage],
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream the output of the model.

        This method should be implemented if the model can generate output
        in a streaming fashion. If the model does not support streaming,
        do not implement it. In that case streaming requests will be automatically
        handled by the _generate method.

        Args:
            messages: the prompt composed of a list of messages.
            stop: a list of strings on which the model should stop generating.
                  If generation stops due to a stop token, the stop token itself
                  SHOULD BE INCLUDED as part of the output. This is not enforced
                  across models right now, but it's a good practice to follow since
                  it makes it much easier to parse the output of the model
                  downstream and understand why generation stopped.
            run_manager: A run manager with callbacks for the LLM.
        """
        
        message_dicts = self._create_message_dicts(messages)
        params = {**kwargs, "stream": True}
        
        default_chunk_class = AIMessageChunk
        for chunk in self.stream_completion_with_retry(
            messages=message_dicts, run_manager=run_manager, **params
        ):
            chunk = json.loads(chunk)
            if not isinstance(chunk, dict):
                chunk = chunk.dict()
            if len(chunk["choices"]) == 0:
                continue
            choice = chunk["choices"][0]
            if choice["delta"] is None:
                continue
            chunk = _convert_delta_to_message_chunk(
                choice["delta"], default_chunk_class
            )
            finish_reason = choice.get("finish_reason")
            generation_info = (
                dict(finish_reason=finish_reason) if finish_reason is not None else None
            )
            default_chunk_class = chunk.__class__
            cg_chunk = ChatGenerationChunk(
                message=chunk, generation_info=generation_info
            )
            if run_manager:
                run_manager.on_llm_new_token(cg_chunk.text, chunk=cg_chunk)
            yield cg_chunk


    # --------------- Model Info ------------ #

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "clarifai-chat"

    