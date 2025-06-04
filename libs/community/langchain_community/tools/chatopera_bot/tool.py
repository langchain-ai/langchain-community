"""Tool for Chatopera Bot Services, https://bot.chatopera.com/"""

from typing import Dict, List, Literal, Optional, Tuple

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool

from .api_wrapper import ChatoperaBotAPIWrapper


class ChatoperaBotRun(BaseTool):
    """Tool that queries the Chatopera Bot Services."""

    name: str = "chatopera_bot"
    description: str = (
        "A wrapper around Chatopera Bot Services. "
        "Useful for when you need to answer questions about current events. "
        "Input should be a search query."
    )
    api_wrapper: ChatoperaBotAPIWrapper

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        return self.api_wrapper.run(query)


class ChatoperaBotResults(BaseTool):
    """Chatopera Bot tool.

    Setup:
        Install ``langchain-community`` and set environment variable ``CHATOPERA_BOT_CLIENT_ID`` and ``CHATOPERA_BOT_CLIENT_SECRET``.

        .. code-block:: bash

            pip install -U langchain-community
            export CHATOPERA_BOT_CLIENT_ID="your-client-id"
            export CHATOPERA_BOT_CLIENT_SECRET="your-client-secret"

    Instantiation:
        .. code-block:: python

            from langchain_community.tools.chatopera_bot import ChatoperaBotAPIWrapper
            from langchain_community.tools.chatopera_bot import ChatoperaBotResults

            chatopera_bot_api = ChatoperaBotAPIWrapper()
            chatopera_bot = ChatoperaBotResults(max_results=2, api_wrapper=chatopera_bot_api)
            tools = [chatopera_bot]
            agent_executor = create_react_agent(llm, tools, checkpointer=checkpoints_saver)

    Invocation with args:
        .. code-block:: python

            tool.invoke({"query": "what is the weather in SF?"})

        .. code-block:: python

            "['foo', 'bar']"

    Invocation with ToolCall:

        .. code-block:: python

            tool.invoke({"args": {"query":"what is the weather in SF?"}, "id": "1", "name": tool.name, "type": "tool_call"})

        .. code-block:: python

            ToolMessage(
                content="['foo', 'bar']",
                name='bing_search_results_json',
                tool_call_id='1'
            )

    """  # noqa: E501

    name: str = "chatopera_bot_results_json"
    description: str = (
        "A wrapper around Chatopera Bot Services. "
        "Useful for when you need to answer questions about current events. "
        "Input should be a search query. Output is an array of the query results."
    )
    num_results: int = 4
    """Max search results to return, default is 4."""
    api_wrapper: ChatoperaBotAPIWrapper
    # https://api.python.langchain.com/en/latest/tools/langchain_core.tools.StructuredTool.html
    response_format: Literal["content"] = "content"

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[str, List[Dict]]:
        """Use the tool."""
        try:
            # logger.info("%s query %s", self.name, query)
            results = self.api_wrapper.results(query, self.num_results)
            # logger.info("%s result %s", self.name, results)
            return str(results), results
        except Exception as e:
            return repr(e), []
