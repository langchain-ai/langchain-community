from __future__ import annotations

from typing import List

from langchain_core._api.deprecation import deprecated
from langchain_core.tools import BaseTool
from langchain_core.tools.base import BaseToolkit

from langchain_community.tools.azure_ai_services import (
    AzureAiServicesDocumentIntelligenceTool,
    AzureAiServicesImageAnalysisTool,
    AzureAiServicesSpeechToTextTool,
    AzureAiServicesTextAnalyticsForHealthTool,
    AzureAiServicesTextToSpeechTool,
)


@deprecated(
    since="0.4.1",
    removal="1.0",
    alternative_import="langchain_azure_ai.tools.AIServicesToolkit",
)
class AzureAiServicesToolkit(BaseToolkit):
    """Toolkit for Azure AI Services."""

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""

        tools: List[BaseTool] = [
            AzureAiServicesDocumentIntelligenceTool(),  # type: ignore[call-arg]
            AzureAiServicesImageAnalysisTool(),
            AzureAiServicesSpeechToTextTool(),  # type: ignore[call-arg]
            AzureAiServicesTextToSpeechTool(),  # type: ignore[call-arg]
            AzureAiServicesTextAnalyticsForHealthTool(),  # type: ignore[call-arg]
        ]

        return tools
