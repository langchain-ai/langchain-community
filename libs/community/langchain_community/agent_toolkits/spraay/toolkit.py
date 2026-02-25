"""Spraay batch payment toolkit for LangChain agents."""

from __future__ import annotations

from typing import List

from langchain_core.tools import BaseTool, BaseToolkit

from langchain_community.tools.spraay.tool import (
    SpraayBatchSendETH,
    SpraayBatchSendETHVariable,
    SpraayBatchSendToken,
    SpraayBatchSendTokenVariable,
)


class SpraayToolkit(BaseToolkit):
    """Toolkit for interacting with Spraay batch payment protocol on Base.

    Provides tools for batch-sending ETH and ERC-20 tokens to multiple
    recipients in a single transaction using the Spraay protocol.

    Setup:
        Install ``web3``:

        .. code-block:: bash

            pip install web3

        Set environment variables:

        .. code-block:: bash

            export SPRAAY_PRIVATE_KEY="your-wallet-private-key"
            export SPRAAY_RPC_URL="https://mainnet.base.org"  # optional

    Example:
        .. code-block:: python

            from langchain_community.agent_toolkits.spraay import SpraayToolkit
            from langchain.agents import create_agent
            from langchain_openai import ChatOpenAI

            toolkit = SpraayToolkit()
            tools = toolkit.get_tools()

            llm = ChatOpenAI(model="gpt-4")
            agent = create_agent(llm, tools)

            result = agent.invoke({
                "messages": [("user", "Send 0.01 ETH to these 3 addresses: 0xAbc..., 0xDef..., 0x123...")]
            })
    """

    def get_tools(self) -> List[BaseTool]:
        """Get all Spraay batch payment tools."""
        return [
            SpraayBatchSendETH(),
            SpraayBatchSendToken(),
            SpraayBatchSendETHVariable(),
            SpraayBatchSendTokenVariable(),
        ]
