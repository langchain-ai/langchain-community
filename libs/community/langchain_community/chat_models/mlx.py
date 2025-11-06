"""MLX Chat Wrapper."""

import json
import logging
import re
import uuid
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
)

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    InvalidToolCall,
    SystemMessage,
    ToolCall,
)
from langchain_core.output_parsers.openai_tools import (
    make_invalid_tool_call,
    parse_tool_call,
)
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
    LLMResult,
)
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool

from langchain_community.llms.mlx_pipeline import MLXPipeline

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful, and honest assistant."""


def _parse_react_tool_calls(
    text: str,
) -> Tuple[list[ToolCall] | None, list[InvalidToolCall]]:
    """Extract ReAct-style tool calls from plain text output.

    Args:
        text: Raw model generation text.

    Returns:
        A tuple containing a list of parsed ``ToolCall`` objects if any were
        detected, otherwise ``None``, and a list of ``InvalidToolCall`` objects
        for unparseable patterns.
    """

    tool_calls: list[ToolCall] = []
    invalid_tool_calls: list[InvalidToolCall] = []

    bracket_pattern = r"Action:\s*(?P<name>[\w.-]+)\[(?P<input>[^\]]+)\]"
    separate_pattern = r"Action:\s*(?P<name>[^\n]+)\nAction Input:\s*(?P<input>[^\n]+)"

    matches = list(re.finditer(bracket_pattern, text))
    if not matches:
        matches = list(re.finditer(separate_pattern, text))

    for match in matches:
        name = match.group("name").strip()
        arg_text = match.group("input").strip()
        try:
            args = json.loads(arg_text)
            if not isinstance(args, dict):
                args = {"input": args}
        except Exception:
            args = {"input": arg_text}
        tool_calls.append(ToolCall(id=str(uuid.uuid4()), name=name, args=args))

    if not tool_calls and "Action:" in text:
        invalid_tool_calls.append(
            make_invalid_tool_call(
                {"name": "unknown", "args": text},
                "Could not parse ReAct tool call",
            )
        )
        return None, invalid_tool_calls

    return tool_calls or None, invalid_tool_calls


class ChatMLX(BaseChatModel):
    """MLX chat models.

    Works with `MLXPipeline` LLM.

    To use, you should have the ``mlx-lm`` python package installed.

    Example:
        .. code-block:: python

            from langchain_community.chat_models import chatMLX
            from langchain_community.llms import MLXPipeline

            llm = MLXPipeline.from_model_id(
                model_id="mlx-community/quantized-gemma-2b-it",
            )
            chat = chatMLX(llm=llm)

    """

    llm: MLXPipeline
    system_message: SystemMessage = SystemMessage(content=DEFAULT_SYSTEM_PROMPT)
    tokenizer: Any = None

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.tokenizer = self.llm.tokenizer

    def _parse_tool_args(self, arg_text: str) -> Dict[str, Any]:
        """Parse the arguments for a tool call.

        Args:
            arg_text: JSON string representation of the tool arguments.

        Returns:
            Parsed arguments dictionary. If parsing fails, returns a dict with
            the original text under the ``input`` key.
        """
        try:
            args = json.loads(arg_text)
        except json.JSONDecodeError:
            args = {"input": arg_text}
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("Unexpected error during tool argument parsing: %s", e)
            args = {"input": arg_text}
        return args

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        tools = kwargs.pop("tools", None)
        llm_input = self._to_chat_prompt(messages, tools=tools)
        llm_result = self.llm._generate(
            prompts=[llm_input], stop=stop, run_manager=run_manager, **kwargs
        )
        return self._to_chat_result(llm_result)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        tools = kwargs.pop("tools", None)
        llm_input = self._to_chat_prompt(messages, tools=tools)
        llm_result = await self.llm._agenerate(
            prompts=[llm_input], stop=stop, run_manager=run_manager, **kwargs
        )
        return self._to_chat_result(llm_result)

    def _to_chat_prompt(
        self,
        messages: List[BaseMessage],
        tokenize: bool = False,
        return_tensors: Optional[str] = None,
        tools: Sequence[dict] | None = None,
    ) -> str:
        """Convert messages to the prompt format expected by the wrapped LLM.

        Args:
            messages: Chat messages to include in the prompt.
            tokenize: Whether to return token IDs instead of text.
            return_tensors: Framework for returned tensors when ``tokenize`` is
                True.
            tools: Optional tool definitions to include in the prompt.
        """
        if not messages:
            raise ValueError("At least one HumanMessage must be provided!")

        if not isinstance(messages[-1], HumanMessage):
            raise ValueError("Last message must be a HumanMessage!")

        messages_dicts = [self._to_chatml_format(m) for m in messages]
        return self.tokenizer.apply_chat_template(
            messages_dicts,
            tokenize=tokenize,
            add_generation_prompt=True,
            return_tensors=return_tensors,
            tools=tools,
        )

    def _to_chatml_format(self, message: BaseMessage) -> dict:
        """Convert LangChain message to ChatML format."""

        if isinstance(message, SystemMessage):
            role = "system"
        elif isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, HumanMessage):
            role = "user"
        else:
            raise ValueError(f"Unknown message type: {type(message)}")

        return {"role": role, "content": message.content}

    @staticmethod
    def _to_chat_result(llm_result: LLMResult) -> ChatResult:
        chat_generations = []

        for g in llm_result.generations[0]:
            tool_calls: list[ToolCall] = []
            invalid_tool_calls: list[InvalidToolCall] = []
            additional_kwargs: Dict[str, Any] = {}

            if isinstance(g.generation_info, dict):
                raw_tool_calls = g.generation_info.get("tool_calls")
            else:
                raw_tool_calls = None

            if raw_tool_calls:
                additional_kwargs["tool_calls"] = raw_tool_calls
                for raw_tool_call in raw_tool_calls:
                    try:
                        tc = parse_tool_call(raw_tool_call, return_id=True)
                    except Exception as e:
                        invalid_tool_calls.append(
                            make_invalid_tool_call(raw_tool_call, str(e))
                        )
                    else:
                        if tc:
                            tool_calls.append(tc)
            else:
                react_tool_calls, invalid_reacts = _parse_react_tool_calls(g.text)
                if react_tool_calls is not None:
                    tool_calls.extend(react_tool_calls)
                invalid_tool_calls.extend(invalid_reacts)

            chat_generation = ChatGeneration(
                message=AIMessage(
                    content=g.text,
                    additional_kwargs=additional_kwargs,
                    tool_calls=tool_calls,
                    invalid_tool_calls=invalid_tool_calls,
                ),
                generation_info=g.generation_info,
            )
            chat_generations.append(chat_generation)

        return ChatResult(
            generations=chat_generations, llm_output=llm_result.llm_output
        )

    @property
    def _llm_type(self) -> str:
        return "mlx-chat-wrapper"

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        import mlx.core as mx
        from mlx_lm.utils import generate_step

        try:
            import mlx.core as mx
            from mlx_lm.sample_utils import make_logits_processors, make_sampler
            from mlx_lm.utils import generate_step

        except ImportError:
            raise ImportError(
                "Could not import mlx_lm python package. "
                "Please install it with `pip install mlx_lm`."
            )
        model_kwargs = kwargs.get("model_kwargs", self.llm.pipeline_kwargs) or {}
        temp: float = model_kwargs.get("temp", 0.0)
        max_new_tokens: int = model_kwargs.get("max_tokens", 100)
        repetition_penalty: Optional[float] = model_kwargs.get(
            "repetition_penalty", None
        )
        repetition_context_size: Optional[int] = model_kwargs.get(
            "repetition_context_size", None
        )
        top_p: float = model_kwargs.get("top_p", 1.0)
        min_p: float = model_kwargs.get("min_p", 0.0)
        min_tokens_to_keep: int = model_kwargs.get("min_tokens_to_keep", 1)

        llm_input = self._to_chat_prompt(messages, tokenize=True, return_tensors="np")

        prompt_tokens = mx.array(llm_input[0])

        eos_token_id = self.tokenizer.eos_token_id

        sampler = make_sampler(temp or 0.0, top_p, min_p, min_tokens_to_keep)

        logits_processors = make_logits_processors(
            None, repetition_penalty, repetition_context_size
        )

        for (token, prob), n in zip(
            generate_step(
                prompt_tokens,
                self.llm.model,
                sampler=sampler,
                logits_processors=logits_processors,
            ),
            range(max_new_tokens),
        ):
            # identify text to yield
            text: Optional[str] = None
            if not isinstance(token, int):
                text = self.tokenizer.decode(token.item())
            else:
                text = self.tokenizer.decode(token)

            # yield text, if any
            if text:
                chunk = ChatGenerationChunk(message=AIMessageChunk(content=text))
                if run_manager:
                    run_manager.on_llm_new_token(text, chunk=chunk)
                yield chunk

            # break if stop sequence found
            if token == eos_token_id or (stop is not None and text in stop):
                break

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type, Callable, BaseTool]],
        *,
        tool_choice: Optional[Union[dict, str, Literal["auto", "none"], bool]] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Bind tool-like objects to this chat model.

        Assumes model is compatible with OpenAI tool-calling API.

        Args:
            tools: A list of tool definitions to bind to this chat model.
                Supports any tool definition handled by
                :meth:`langchain_core.utils.function_calling.convert_to_openai_tool`.
            tool_choice: Which tool to require the model to call.
                Must be the name of the single provided function or
                "auto" to automatically determine which function to call
                (if any), or a dict of the form:
                {"type": "function", "function": {"name": <<tool_name>>}}.
            **kwargs: Any additional parameters to pass to the
                :class:`~langchain.runnable.Runnable` constructor.
        """

        formatted_tools: List[Dict[str, Any]] = [
            convert_to_openai_tool(tool) for tool in tools
        ]
        if tool_choice is not None and tool_choice:
            if len(formatted_tools) != 1:
                raise ValueError(
                    "When specifying `tool_choice`, you must provide exactly one "
                    f"tool. Received {len(formatted_tools)} tools."
                )
            if isinstance(tool_choice, str):
                if tool_choice not in ("auto", "none"):
                    tool_choice = {
                        "type": "function",
                        "function": {"name": tool_choice},
                    }
            elif isinstance(tool_choice, bool):
                tool_choice = formatted_tools[0]
            elif isinstance(tool_choice, dict):
                if (
                    formatted_tools[0]["function"]["name"]
                    != tool_choice["function"]["name"]
                ):
                    raise ValueError(
                        f"Tool choice {tool_choice} was specified, but the only "
                        f"provided tool was {formatted_tools[0]['function']['name']}."
                    )
            else:
                raise ValueError(
                    f"Unrecognized tool_choice type. Expected str, bool or dict. "
                    f"Received: {tool_choice}"
                )
            kwargs["tool_choice"] = tool_choice
        return super().bind(
            tools=cast(Sequence[Dict[str, Any]], formatted_tools),
            **kwargs,
        )
