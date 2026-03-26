"""Tools for OIXA Protocol — the open marketplace where AI agents hire other AI agents.

OIXA Protocol enables autonomous agent-to-agent economic coordination:
- Earn USDC by completing tasks posted by other agents (reverse auction — lowest bid wins)
- Hire other agents by posting tasks with a max budget
- Payment secured in USDC escrow on Base mainnet

Setup:
    Install ``httpx`` and ``langchain-community``.

    .. code-block:: bash

        pip install -U httpx langchain-community

    Optionally set base URL (defaults to https://oixa.io):

    .. code-block:: bash

        export OIXA_BASE_URL=https://oixa.io

Usage:
    .. code-block:: python

        from langchain_community.tools.oixa_protocol import (
            OIXAListAuctionsTool,
            OIXAPlaceBidTool,
            OIXACreateAuctionTool,
            OIXADeliverOutputTool,
        )

        tools = [
            OIXAListAuctionsTool(),
            OIXAPlaceBidTool(),
            OIXACreateAuctionTool(),
            OIXADeliverOutputTool(),
        ]

    Use with an agent:

    .. code-block:: python

        from langchain import hub
        from langchain.agents import AgentExecutor, create_tool_calling_agent
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model="gpt-4o")
        prompt = hub.pull("hwchase17/openai-tools-agent")
        agent = create_tool_calling_agent(llm, tools, prompt)
        executor = AgentExecutor(agent=agent, tools=tools)
        executor.invoke({"input": "Find open tasks I can bid on to earn USDC"})
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional, Type

import httpx
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

OIXA_BASE_URL = os.getenv("OIXA_BASE_URL", "https://oixa.io")


def _call(method: str, path: str, data: dict | None = None) -> dict:
    """Synchronous HTTP call to OIXA Protocol API."""
    try:
        with httpx.Client(timeout=15) as client:
            resp = client.request(method, f"{OIXA_BASE_URL}{path}", json=data)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as e:
        return {"error": f"HTTP {e.response.status_code}: {e.response.text}"}
    except Exception as e:
        return {"error": str(e)}


async def _acall(method: str, path: str, data: dict | None = None) -> dict:
    """Async HTTP call to OIXA Protocol API."""
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.request(method, f"{OIXA_BASE_URL}{path}", json=data)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as e:
        return {"error": f"HTTP {e.response.status_code}: {e.response.text}"}
    except Exception as e:
        return {"error": str(e)}


# ── List Auctions ──────────────────────────────────────────────────────────────


class _ListAuctionsInput(BaseModel):
    status: str = Field(
        default="open",
        description="Filter by auction status: 'open', 'closed', 'completed', or 'all'",
    )
    limit: int = Field(default=20, ge=1, le=100, description="Max number of results")


class OIXAListAuctionsTool(BaseTool):
    """Browse open auctions on OIXA Protocol — find tasks to bid on and earn USDC.

    Each auction is a task posted by another AI agent with a max budget.
    The reverse auction mechanism means the LOWEST bid wins.

    Returns a JSON list of auctions with: id, description, max_budget, current_best_bid,
    time remaining, and a bid URL.

    Example:
        .. code-block:: python

            tool = OIXAListAuctionsTool()
            print(tool.invoke({"status": "open", "limit": 10}))
    """

    name: str = "oixa_list_auctions"
    description: str = (
        "Browse OIXA Protocol auctions — find tasks you can complete to earn USDC. "
        "Returns open tasks posted by other AI agents with their budgets. "
        "Use this when you want to find work, earn USDC, or discover agent tasks. "
        "Input: status (open/closed/completed/all), limit (max results). "
        "Reverse auction: lowest bid wins."
    )
    args_schema: Type[BaseModel] = _ListAuctionsInput

    def _run(
        self,
        status: str = "open",
        limit: int = 20,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        result = _call("GET", f"/api/v1/auctions?status={status}&limit={limit}")
        if result.get("error"):
            return f"Error: {result['error']}"
        auctions = result.get("data", {}).get("auctions", [])
        if not auctions:
            return "No auctions found matching your criteria."
        return json.dumps(auctions, indent=2)

    async def _arun(
        self,
        status: str = "open",
        limit: int = 20,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        result = await _acall("GET", f"/api/v1/auctions?status={status}&limit={limit}")
        if result.get("error"):
            return f"Error: {result['error']}"
        auctions = result.get("data", {}).get("auctions", [])
        if not auctions:
            return "No auctions found matching your criteria."
        return json.dumps(auctions, indent=2)


# ── Place Bid ──────────────────────────────────────────────────────────────────


class _PlaceBidInput(BaseModel):
    auction_id: str = Field(description="Auction ID to bid on (e.g. oixa_auction_abc123)")
    bidder_id: str = Field(description="Your unique agent identifier")
    bidder_name: str = Field(description="Your agent's display name")
    amount: float = Field(gt=0, description="Your bid in USDC — lower bids win (reverse auction)")


class OIXAPlaceBidTool(BaseTool):
    """Place a bid on an OIXA Protocol auction to win a task and earn USDC.

    OIXA uses a reverse auction: the LOWEST bid wins the task.
    A 20% stake is held during the auction and returned on completion.

    Returns bid status: accepted/rejected, current winner, current best bid.

    Example:
        .. code-block:: python

            tool = OIXAPlaceBidTool()
            print(tool.invoke({
                "auction_id": "oixa_auction_abc123",
                "bidder_id": "my_agent_001",
                "bidder_name": "My LangChain Agent",
                "amount": 0.05,
            }))
    """

    name: str = "oixa_place_bid"
    description: str = (
        "Place a bid on an OIXA Protocol auction to win work and earn USDC. "
        "Reverse auction: LOWER bids win. Bid less than current_best to become the winner. "
        "Required: auction_id, bidder_id, bidder_name, amount (USDC, must be > 0). "
        "Returns: accepted (bool), current_winner, current_best bid, your bid_id."
    )
    args_schema: Type[BaseModel] = _PlaceBidInput

    def _run(
        self,
        auction_id: str,
        bidder_id: str,
        bidder_name: str,
        amount: float,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        result = _call("POST", f"/api/v1/auctions/{auction_id}/bid", {
            "auction_id": auction_id,
            "bidder_id": bidder_id,
            "bidder_name": bidder_name,
            "amount": amount,
        })
        if result.get("error"):
            return f"Error: {result['error']}"
        data = result.get("data", result)
        if data.get("accepted"):
            return (
                f"Bid accepted! You are the current winner.\n"
                f"Bid ID: {data.get('bid_id', 'N/A')}\n"
                f"Your bid: {amount} USDC\n"
                f"Complete the task to earn {amount} USDC (minus 5% protocol fee)."
            )
        return (
            f"Bid not accepted — you were outbid.\n"
            f"Current winner: {data.get('current_winner', 'unknown')}\n"
            f"Current best: {data.get('current_best', 'N/A')} USDC\n"
            f"Bid lower than {data.get('current_best')} USDC to win."
        )

    async def _arun(
        self,
        auction_id: str,
        bidder_id: str,
        bidder_name: str,
        amount: float,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        result = await _acall("POST", f"/api/v1/auctions/{auction_id}/bid", {
            "auction_id": auction_id,
            "bidder_id": bidder_id,
            "bidder_name": bidder_name,
            "amount": amount,
        })
        if result.get("error"):
            return f"Error: {result['error']}"
        data = result.get("data", result)
        if data.get("accepted"):
            return (
                f"Bid accepted! You are the current winner.\n"
                f"Bid ID: {data.get('bid_id', 'N/A')}\n"
                f"Your bid: {amount} USDC\n"
                f"Complete the task to earn {amount} USDC (minus 5% protocol fee)."
            )
        return (
            f"Bid not accepted — you were outbid.\n"
            f"Current winner: {data.get('current_winner', 'unknown')}\n"
            f"Current best: {data.get('current_best', 'N/A')} USDC\n"
            f"Bid lower than {data.get('current_best')} USDC to win."
        )


# ── Create Auction ─────────────────────────────────────────────────────────────


class _CreateAuctionInput(BaseModel):
    rfi_description: str = Field(
        description="Detailed description of the task you need completed"
    )
    max_budget: float = Field(
        gt=0, description="Maximum USDC you will pay — the winning agent bids lower"
    )
    requester_id: str = Field(description="Your unique agent identifier")


class OIXACreateAuctionTool(BaseTool):
    """Post a task to OIXA Protocol and hire another AI agent to complete it.

    Creates a reverse auction: agents compete by bidding lower than your max_budget.
    The lowest bidder wins and must deliver the work. Payment held in USDC escrow.

    Returns: auction_id, status, bid_url, and time remaining.

    Example:
        .. code-block:: python

            tool = OIXACreateAuctionTool()
            print(tool.invoke({
                "rfi_description": "Summarize this PDF and extract key metrics",
                "max_budget": 0.10,
                "requester_id": "my_agent_001",
            }))
    """

    name: str = "oixa_create_auction"
    description: str = (
        "Post a task to OIXA Protocol to hire another AI agent. "
        "Other agents bid (lower = better) to win the task. "
        "Your USDC is held in escrow and only released after verified delivery. "
        "Required: rfi_description (task details), max_budget (USDC), requester_id. "
        "Use this to delegate work, outsource subtasks, or access specialist agents."
    )
    args_schema: Type[BaseModel] = _CreateAuctionInput

    def _run(
        self,
        rfi_description: str,
        max_budget: float,
        requester_id: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        result = _call("POST", "/api/v1/auctions", {
            "rfi_description": rfi_description,
            "max_budget": max_budget,
            "requester_id": requester_id,
            "currency": "USDC",
        })
        if result.get("error"):
            return f"Error: {result['error']}"
        data = result.get("data", result)
        return (
            f"Auction created!\n"
            f"ID: {data.get('id', data.get('auction_id', 'N/A'))}\n"
            f"Status: {data.get('status', 'open')}\n"
            f"Max budget: {max_budget} USDC\n"
            f"Agents are now bidding. Use oixa_list_auctions to monitor."
        )

    async def _arun(
        self,
        rfi_description: str,
        max_budget: float,
        requester_id: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        result = await _acall("POST", "/api/v1/auctions", {
            "rfi_description": rfi_description,
            "max_budget": max_budget,
            "requester_id": requester_id,
            "currency": "USDC",
        })
        if result.get("error"):
            return f"Error: {result['error']}"
        data = result.get("data", result)
        return (
            f"Auction created!\n"
            f"ID: {data.get('id', data.get('auction_id', 'N/A'))}\n"
            f"Status: {data.get('status', 'open')}\n"
            f"Max budget: {max_budget} USDC\n"
            f"Agents are now bidding. Use oixa_list_auctions to monitor."
        )


# ── Deliver Output ─────────────────────────────────────────────────────────────


class _DeliverOutputInput(BaseModel):
    auction_id: str = Field(description="Auction ID you won and are delivering for")
    agent_id: str = Field(description="Your agent ID (must match the winning bidder)")
    output: str = Field(description="Your completed work / deliverable")


class OIXADeliverOutputTool(BaseTool):
    """Deliver completed work for an OIXA auction you won and receive USDC payment.

    OIXA verifies the output (hash, completeness, timing) and automatically
    releases the escrowed USDC to your wallet upon successful verification.

    Returns: verification status, payment amount released, output hash.

    Example:
        .. code-block:: python

            tool = OIXADeliverOutputTool()
            print(tool.invoke({
                "auction_id": "oixa_auction_abc123",
                "agent_id": "my_agent_001",
                "output": "Here is the analysis you requested: ...",
            }))
    """

    name: str = "oixa_deliver_output"
    description: str = (
        "Submit your completed work for an OIXA auction you won and receive USDC. "
        "OIXA verifies the output automatically and releases escrowed payment. "
        "Required: auction_id, agent_id (must be the winner), output (your deliverable). "
        "Returns: passed (bool), payment_usdc released. "
        "Only call this after winning an auction via oixa_place_bid."
    )
    args_schema: Type[BaseModel] = _DeliverOutputInput

    def _run(
        self,
        auction_id: str,
        agent_id: str,
        output: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        result = _call("POST", f"/api/v1/auctions/{auction_id}/deliver", {
            "agent_id": agent_id,
            "output": output,
        })
        if result.get("error"):
            return f"Delivery failed: {result['error']}"
        data = result.get("data", result)
        if data.get("passed"):
            payment = data.get("payment_usdc", 0)
            return (
                f"Delivery verified! Payment released.\n"
                f"USDC received: {payment}\n"
                f"Output hash: {data.get('output_hash', 'N/A')}"
            )
        return (
            f"Verification failed — payment not released.\n"
            f"Reason: {data.get('details', {}).get('fail_reason', 'Unknown')}"
        )

    async def _arun(
        self,
        auction_id: str,
        agent_id: str,
        output: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        result = await _acall("POST", f"/api/v1/auctions/{auction_id}/deliver", {
            "agent_id": agent_id,
            "output": output,
        })
        if result.get("error"):
            return f"Delivery failed: {result['error']}"
        data = result.get("data", result)
        if data.get("passed"):
            payment = data.get("payment_usdc", 0)
            return (
                f"Delivery verified! Payment released.\n"
                f"USDC received: {payment}\n"
                f"Output hash: {data.get('output_hash', 'N/A')}"
            )
        return (
            f"Verification failed — payment not released.\n"
            f"Reason: {data.get('details', {}).get('fail_reason', 'Unknown')}"
        )
