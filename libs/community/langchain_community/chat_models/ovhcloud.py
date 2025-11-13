"""OVHcloud AI Endpoints chat wrapper. Relies heavily on ChatOpenAI."""

from typing import (
    Any,
    Callable,
    Dict,
    Literal,
    Optional,
    Sequence,
    Type,
    Union,
)

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env, pre_init
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field, SecretStr

from langchain_openai import ChatOpenAI
from langchain_community.utils.openai import is_openai_v1

DEFAULT_API_BASE = "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1"
DEFAULT_MODEL = "gpt-oss-120b"


class ChatOVHcloud(ChatOpenAI):
    """OVHcloud AI Endpoints Chat large language models.

    See https://www.ovhcloud.com/en/public-cloud/ai-endpoints/catalog/ for information about OVHcloud AI Endpoints.

    To use, you should have the ``openai`` python package installed and the
    environment variable ``OVHCLOUD_API_TOKEN`` set with your API token.
    Alternatively, you can use the ovhcloud_api_token keyword argument.

    Any parameters that are valid to be passed to the `openai.create` call can be passed
    in, even if not explicitly saved on this class.

    Example:
        .. code-block:: python

            from langchain_community.chat_models import ChatOVHcloud
            chat = ChatOVHcloud(model_name="gpt-oss-120b")
    """

    ovhcloud_api_base: str = Field(default=DEFAULT_API_BASE)
    ovhcloud_api_token: SecretStr = Field(default=SecretStr(""), alias="api_key")
    model_name: str = Field(default=DEFAULT_MODEL, alias="model")

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "ovhcloud-chat"

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"ovhcloud_api_token": "OVHCLOUD_API_TOKEN"}

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return False

    @pre_init
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values["ovhcloud_api_base"] = get_from_dict_or_env(
            values,
            "ovhcloud_api_base",
            "OVHCLOUD_API_BASE",
            default=DEFAULT_API_BASE,
        )
        values["ovhcloud_api_token"] = convert_to_secret_str(
            get_from_dict_or_env(
                values, "ovhcloud_api_token", "OVHCLOUD_API_TOKEN"
            )
        )
        values["model_name"] = get_from_dict_or_env(
            values,
            "model_name",
            "OVHCLOUD_MODEL_NAME",
            default=DEFAULT_MODEL,
        )

        try:
            import openai

            if is_openai_v1():
                client_params = {
                    "api_key": values["ovhcloud_api_token"].get_secret_value(),
                    "base_url": values["ovhcloud_api_base"],
                }
                if not values.get("client"):
                    values["client"] = openai.OpenAI(**client_params).chat.completions
                if not values.get("async_client"):
                    values["async_client"] = openai.AsyncOpenAI(
                        **client_params
                    ).chat.completions
            else:
                values["openai_api_base"] = values["ovhcloud_api_base"]
                values["openai_api_key"] = values["ovhcloud_api_token"].get_secret_value()
                values["client"] = openai.ChatCompletion
        except ImportError:
            raise ImportError(
                "Could not import openai python package. "
                "Please install it with `pip install openai`."
            )

        return values

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type, Callable, BaseTool]],
        *,
        tool_choice: Optional[
            Union[dict, str, Literal["auto", "none", "required", "any"], bool]
        ] = None,
        strict: Optional[bool] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Imitating bind_tool method from langchain_openai.ChatOpenAI"""

        formatted_tools = [
            convert_to_openai_tool(tool, strict=strict) for tool in tools
        ]
        if tool_choice:
            if isinstance(tool_choice, str):
                # tool_choice is a tool/function name
                if tool_choice not in ("auto", "none", "any", "required"):
                    tool_choice = {
                        "type": "function",
                        "function": {"name": tool_choice},
                    }
                # 'any' is not natively supported by OpenAI API.
                # We support 'any' since other models use this instead of 'required'.
                if tool_choice == "any":
                    tool_choice = "required"
            elif isinstance(tool_choice, bool):
                tool_choice = "required"
            elif isinstance(tool_choice, dict):
                tool_names = [
                    formatted_tool["function"]["name"]
                    for formatted_tool in formatted_tools
                ]
                if not any(
                    tool_name == tool_choice["function"]["name"]
                    for tool_name in tool_names
                ):
                    raise ValueError(
                        f"Tool choice {tool_choice} was specified, but the only "
                        f"provided tools were {tool_names}."
                    )
            else:
                raise ValueError(
                    f"Unrecognized tool_choice type. Expected str, bool or dict. "
                    f"Received: {tool_choice}"
                )
            kwargs["tool_choice"] = tool_choice
        return super().bind(tools=formatted_tools, **kwargs)

