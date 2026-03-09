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
    am_key: Optional[SecretStr] = Field(
        default=None,
        description="Agent Module API key (X-AM-Key header). Get one at agent-module.dev."
    )

    def _build_headers(self) -> dict:
        """Build request headers, injecting AM key if present."""
        if self.am_key:
            return {"X-AM-Key": self.am_key.get_secret_value()}
        return {}

    def _run(
        self,
        module: str,
        vertical: str = "ethics",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Query Agent Module knowledge base (sync)."""
        try:
            response = requests.get(
                "https://api.agent-module.dev/api/demo",
                params={"vertical": vertical, "module": module},
                headers=self._build_headers(),
                timeout=10,
            )
            response.raise_for_status()
            return response.text
        except requests.HTTPError as e:
            return f"Agent Module HTTP error ({e.response.status_code}): {str(e)}"
        except requests.RequestException as e:
            return f"Agent Module connection error: {str(e)}"

    async def _arun(
        self,
        module: str,
        vertical: str = "ethics",
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Query Agent Module knowledge base (async)."""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    "https://api.agent-module.dev/api/demo",
                    params={"vertical": vertical, "module": module},
                    headers=self._build_headers(),
                    timeout=10,
                )
                response.raise_for_status()
                return response.text
            except httpx.HTTPStatusError as e:
                return f"Agent Module HTTP error ({e.response.status_code}): {str(e)}"
            except httpx.RequestError as e:
                return f"Agent Module connection error: {str(e)}"
