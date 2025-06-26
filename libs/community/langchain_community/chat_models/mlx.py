"""MLX Chat Wrapper."""

import json
import re
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Type,
    Union,
)

from pydantic import PrivateAttr

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult, LLMResult
from langchain_core.utils.function_calling import convert_to_openai_tool

from langchain_community.llms.mlx_pipeline import MLXPipeline  # adjust import as needed

DEFAULT_SYSTEM_PROMPT = "You are a helpful, respectful, and honest assistant."


class ChatMLX(BaseChatModel):
    """MLX chat model wrapper."""

    @property
    def _llm_type(self) -> str:
        """Identifier for this LLM type (satisfies BaseChatModel)."""
        return "mlx"

    llm: MLXPipeline
    system_message: SystemMessage = SystemMessage(content=DEFAULT_SYSTEM_PROMPT)

    _tokenizer: Any = PrivateAttr()
    _tools: Optional[List[dict]] = PrivateAttr(default=None)

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        # stash the MLX tokenizer for later
        self._tokenizer = self.llm.tokenizer

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = self._to_chat_prompt(messages)
        result = self.llm._generate(prompts=[prompt], stop=stop, run_manager=run_manager, **kwargs)
        return self._to_chat_result(result)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = self._to_chat_prompt(messages)
        result = await self.llm._agenerate(prompts=[prompt], stop=stop, run_manager=run_manager, **kwargs)
        return self._to_chat_result(result)

    def _to_chat_prompt(
        self,
        messages: List[BaseMessage],
        tokenize: bool = False,
        return_tensors: Optional[str] = None,
    ) -> str:
        if not messages or not isinstance(messages[-1], HumanMessage):
            raise ValueError("Last message must be a HumanMessage!")

        chunks: List[Dict[str, Any]] = []

        # If tools are bound, inject them into the system prompt
        if self._tools:
            names = ", ".join(t["function"]["name"] for t in self._tools)
            tools_json = json.dumps({"tools": self._tools}, indent=2)
            chunks.append({
                "role": "system",
                "content": (
                    f"You have access to the following tools:\n{tools_json}\n\n"
                    f"When needed, respond with a JSON object like this:\n"
                    f'{{"name": "<tool_name>", "arguments": {{"arg1": "...", "arg2": "..."}}}}\n'
                    f"Available tools: {names}."
                )
            })

        # Add the actual conversation history
        chunks.extend(self._to_chatml_format(m) for m in messages)

        # Let MLX tokenize/apply its chat template
        return self._tokenizer.apply_chat_template(
            chunks,
            tokenize=tokenize,
            add_generation_prompt=True,
            return_tensors=return_tensors,
        )

    def _to_chatml_format(self, msg: BaseMessage) -> Dict[str, Any]:
        if isinstance(msg, SystemMessage):
            role = "system"
        elif isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        else:
            raise ValueError(f"Unknown message type: {type(msg)}")
        return {"role": role, "content": msg.content}

    @staticmethod
    def _to_chat_result(llm_result: LLMResult) -> ChatResult:
        gens: List[ChatGeneration] = []
        for gen in llm_result.generations[0]:
            raw = gen.text.strip()
            tool_calls: List[Dict[str, Any]] = []

            # Try full JSON parse first
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict) and "name" in parsed:
                    tool_calls = [parsed]
            except json.JSONDecodeError:
                # Fallback: regex extraction
                name_m = re.search(r'"name"\s*:\s*"([^"]+)"', raw)
                if name_m:
                    name = name_m.group(1)
                    args_m = re.search(r'"arguments"\s*:\s*({.*?})', raw, re.DOTALL)
                    args: Dict[str, Any] = {}
                    if args_m:
                        try:
                            args = json.loads(args_m.group(1))
                        except json.JSONDecodeError:
                            pass
                    tool_calls = [{"name": name, "arguments": args}]

            gens.append(
                ChatGeneration(
                    message=AIMessage(content=raw, tool_calls=tool_calls),
                    generation_info=gen.generation_info,
                )
            )

        return ChatResult(generations=gens, llm_output=llm_result.llm_output)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        import mlx.core as mx
        from mlx_lm.sample_utils import make_logits_processors, make_sampler
        from mlx_lm.utils import generate_step

        mk = kwargs.get("model_kwargs", self.llm.pipeline_kwargs)
        temp = mk.get("temp", 0.0)
        max_tokens = mk.get("max_tokens", 100)
        rep_pen = mk.get("repetition_penalty")
        rep_ctx = mk.get("repetition_context_size")
        top_p = mk.get("top_p", 1.0)
        min_p = mk.get("min_p", 0.0)
        keep = mk.get("min_tokens_to_keep", 1)

        inp = self._to_chat_prompt(messages, tokenize=True, return_tensors="np")
        prompt_tokens = mx.array(inp[0])
        eos_id = self._tokenizer.eos_token_id

        sampler = make_sampler(temp, top_p, min_p, keep)
        proc = make_logits_processors(None, rep_pen, rep_ctx)

        for (token, _), _ in zip(
            generate_step(prompt_tokens, self.llm.model, sampler, proc),
            range(max_tokens),
        ):
            txt = self._tokenizer.decode(token.item() if hasattr(token, "item") else token)
            if txt:
                chunk = ChatGenerationChunk(message=AIMessageChunk(content=txt))
                if run_manager:
                    run_manager.on_llm_new_token(txt, chunk=chunk)
                yield chunk
            if token == eos_id or (stop and txt in stop):
                break

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type, Callable, Any]],
        *,
        tool_choice: Optional[Union[dict, str, bool]] = None,
        **kwargs: Any,
    ) -> "ChatMLX":
        formatted = [convert_to_openai_tool(t) for t in tools]
        self._tools = formatted
        if tool_choice and len(formatted) != 1:
            raise ValueError(
                f"Tool choice specified but {len(formatted)} tools were bound; only one allowed."
            )
        return super().bind(tools=formatted, **kwargs)
#     def unbind_tools(self) -> "ChatMLX":
#         """Unbind any tools."""