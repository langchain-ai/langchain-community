from typing import Optional, Type
from langchain_core.callbacks import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, SecretStr
import requests
import httpx


class AgentModuleInput(BaseModel):
    module: str = Field(
        description=(
            "Module identifier. Examples: 'ETH_021' (FRIA), 'ETH_016' (prohibited practices), "
            "'ETH_015' (high-risk classification), 'ETH_017' (risk management), "
            "'ETH_013' (conformity assessment), 'ETH_020' (GPAI obligations)."
        )
    )
    vertical: str = Field(
        default="ethics",
        description="Knowledge vertical. Default: 'ethics'."
    )


class AgentModuleTool(BaseTool):
    """Tool for querying Agent Module validated EU AI Act compliance knowledge."""

    name: str = "agent_module_eu_ai_act"
    description: str = (
        "Query Agent Module for validated, deterministic EU AI Act compliance knowledge. "
        "Returns binary logic gates and specific statutory citations. "
        "Use for: FRIA requirements (ETH_021/Art.27), prohibited AI practices (ETH_016/Art.5), "
        "high-risk classification (ETH_015/Annex III), risk management (ETH_017/Art.9), "
        "conformity assessment (ETH_013/Art.43), GPAI obligations (ETH_020/Art.53-55). "
        "Confidence_required: 1.0 — no probabilistic inference. August 2026 enforcement deadline."
    )
    args_schema: Type[BaseModel] = AgentModuleInput
    am_key: Optional[SecretStr] = Field(default=None, description="Agent Module API key (X-AM-Key header)")

    def _run(
        self,
        module: str,
        vertical: str = "ethics",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        headers = {}
        if self.am_key:
            headers["X-AM-Key"] = self.am_key.get_secret_value()
        try:
            response = requests.get(
                "https://api.agent-module.dev/api/demo",
                params={"vertical": vertical, "module": module},
                headers=headers,
                timeout=10,
            )
            response.raise_for_status()
            return response.text
        except Exception as e:
            return f"Error querying Agent Module: {str(e)}"

    async def _arun(
        self,
        module: str,
        vertical: str = "ethics",
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        headers = {}
        if self.am_key:
            headers["X-AM-Key"] = self.am_key.get_secret_value()
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    "https://api.agent-module.dev/api/demo",
                    params={"vertical": vertical, "module": module},
                    headers=headers,
                    timeout=10,
                )
                response.raise_for_status()
                return response.text
            except Exception as e:
                return f"Error querying Agent Module: {str(e)}"
