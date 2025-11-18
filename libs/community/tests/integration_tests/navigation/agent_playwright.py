#!/usr/bin/env python3

import argparse
import asyncio
import os
import uuid
import logging
from typing import Annotated, TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
import nest_asyncio
from dotenv import load_dotenv
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    BaseMessage,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from playwright.async_api import async_playwright

from langchain_community.tools.playwright.click import ClickTool
from langchain_community.tools.playwright.current_page import CurrentWebPageTool
from langchain_community.tools.playwright.download_file import DownloadFileTool
from langchain_community.tools.playwright.drag_slider import DragSliderTool
from langchain_community.tools.playwright.dragndrop import DragAndDropTool
from langchain_community.tools.playwright.extract_dom_tree import ExtractDOMTreeTool
from langchain_community.tools.playwright.extract_hyperlinks import (
    ExtractHyperlinksTool,
)
from langchain_community.tools.playwright.extract_inputs import ExtractInputsTool
from langchain_community.tools.playwright.extract_text import ExtractTextTool
from langchain_community.tools.playwright.get_elements import GetElementsTool
from langchain_community.tools.playwright.hover_element import HoverElementTool
from langchain_community.tools.playwright.http_request import HttpRequestTool
from langchain_community.tools.playwright.input_text import InputTextTool
from langchain_community.tools.playwright.navigate import NavigateTool
from langchain_community.tools.playwright.navigate_back import NavigateBackTool
from langchain_community.tools.playwright.output_to_file import OutputToFileTool
from langchain_community.tools.playwright.press_key import PressKeyTool
from langchain_community.tools.playwright.screenshot import ScreenshotTool
from langchain_community.tools.playwright.scroll import ScrollTool
from langchain_community.tools.playwright.select_dropdown import SelectDropdownTool
from langchain_community.tools.playwright.switch_frame import SwitchFrameTool
from langchain_community.tools.playwright.upload_file import UploadFileTool

load_dotenv()


api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set")

nest_asyncio.apply()


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("agent_debug.log", mode="w"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    logger.info("Setting up LLM...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
    )

    logger.info("Launching Playwright...")
    playwright = None
    browser = None
    try:
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(
            args=[
                "--disable-gpu",
                "--disable-dev-shm-usage",
                "--disable-web-security",
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-setuid-sandbox",
            ],
            headless=False,
        )
        page = await browser.new_page(no_viewport=True)

        logger.info("Creating tools...")
        tools = [
            NavigateTool(page=page, async_browser=browser),
            NavigateBackTool(page=page, async_browser=browser),
            ExtractDOMTreeTool(page=page, async_browser=browser),
            ClickTool(page=page, async_browser=browser),
            InputTextTool(page=page, async_browser=browser),
            PressKeyTool(page=page, async_browser=browser),
            ScreenshotTool(page=page, async_browser=browser),
            CurrentWebPageTool(page=page, async_browser=browser),
            ExtractTextTool(page=page, async_browser=browser),
            ExtractHyperlinksTool(page=page, async_browser=browser),
            GetElementsTool(page=page, async_browser=browser),
            ScrollTool(page=page, async_browser=browser),
            DragAndDropTool(page=page, async_browser=browser),
            HoverElementTool(page=page, async_browser=browser),
            UploadFileTool(page=page, async_browser=browser),
            SwitchFrameTool(page=page, async_browser=browser),
            DragSliderTool(page=page, async_browser=browser),
            ExtractInputsTool(page=page, async_browser=browser),
            SelectDropdownTool(page=page, async_browser=browser),
            DownloadFileTool(),
            OutputToFileTool(),
            HttpRequestTool(),
        ]
        logger.info(f"Initialized {len(tools)} tools.")

        # Bind tools to LLM
        logger.info("Binding tools to LLM...")
        llm_with_tools = llm.bind_tools(tools=tools, parallel_tool_calls=False)

        system_prompt_content = """You are a web automation agent that can browse websites and perform tasks as requested.

AVAILABLE TOOLS:
- navigate: Navigate to a URL
- navigate_back: Go back to the previous page
- extract_dom_tree: Get the DOM structure of the current page
- click: Click on an element
- input_text: Enter text into an input field
- press_key: Press a keyboard key
- take_screenshot: Capture a screenshot
- get_current_page: Get the current page URL
- extract_text: Extract text from elements
- extract_hyperlinks: Get all hyperlinks on the page
- get_elements: Find elements matching a selector
- scroll: Scroll the page
- drag_and_drop: Drag and drop elements
- hover_element: Hover over an element
- upload_file: Upload a file
- switch_frame: Switch to an iframe
- drag_slider: Drag a slider element
- extract_inputs: Get all input fields
- select_dropdown: Select an option from a dropdown
- download_file: Download a file
- output_to_file: Save output to a file
- http_request: Make an HTTP request

EFFICIENT WORKFLOW - follow this exactly:
1. Navigate to the requested URL. The URLS must start with 'https://'.
2. **After navigation, call `extract_dom_tree` (with no arguments)** to get the simplified page structure.
3. Examine this simplified DOM to find cookie banners, navigation links, and other elements.
4. **Always accept cookies** if a banner is present (e.g., click "Accept All").
5. Perform the next action based on the task (e.g., click "Characters").
6. Call `extract_dom_tree` (no arguments) again **only if** the page state has significantly changed after an action (like navigation).
7. Continue step-by-step until the task is complete.

GENERAL GUIDELINES:
- **IMPORTANT: Never use the 'full_tree=True' argument** for `extract_dom_tree`. The default simplified tree is much more effective and reliable for analysis and fits within the context window.
- Use selectors found in the DOM extraction.
- Complete all requested actions in the proper sequence.

COMMON WEB TASKS:
- For search tasks: Input text and press Enter to submit the query
- For form filling: Find form fields, enter data, and submit the form
- For navigation: Identify and click on relevant links or buttons
- For extraction: Extract text or data from relevant elements

TOOL USAGE FORMAT:
When using tools, always use this format:
1. First, use `Maps` to go to the URL.
2. Then, use `extract_dom_tree({})` (no arguments) to understand the page.
3. Use other tools as needed.

Example tool usage:
1. navigate({"url": "https://example.com"})
2. extract_dom_tree({})
3. click({"selector": "button#accept-cookies"})
4. click({"selector": "a[href='/characters']"})
"""


        async def agent_node(state: AgentState):
            logger.info("--- Calling Agent Node ---")
            try:
                if state["messages"]:
                    logger.debug(f"Current state: {state['messages'][-1].pretty_repr()}")
                else:
                    logger.debug("Current state is empty.")

                response = await llm_with_tools.ainvoke(state["messages"])

                logger.info(f"LLM Response: {response.pretty_repr()}")
                return {"messages": [response]}
            except Exception as e:
                logger.error(f"Error in agent_node: {e}", exc_info=True)
                error_message = f"Error in agent node: {e}. Check logs for details. Stopping."
                return {"messages": [SystemMessage(content=error_message)]}

        tool_node = ToolNode(tools)

        def should_continue(state: AgentState):
            logger.info("--- Checking 'should_continue' ---")
            if not state["messages"]:
                logger.warning("State has no messages. Ending.")
                return END

            last_message = state["messages"][-1]
            logger.info(f"Last message type: {type(last_message).__name__}")

            if isinstance(last_message, SystemMessage):
                logger.warning("Last message was a SystemMessage, likely an error. Ending.")
                return END

            if last_message.tool_calls:
                logger.info(f"Decision: Agent wants to use {len(last_message.tool_calls)} tool(s). -> 'tools'")
                return "tools"
            else:
                logger.info("Decision: Agent has no tool calls. -> END")
                return END

        workflow = StateGraph(AgentState)

        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", tool_node)

        workflow.set_entry_point("agent")

        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "tools": "tools",
                END: END,
            },
        )

        workflow.add_edge("tools", "agent")

        logger.info("Compiling graph...")
        checkpointer = MemorySaver()
        graph = workflow.compile(checkpointer=checkpointer)

        parser = argparse.ArgumentParser()
        parser.add_argument("prompt", help="Prompt for the agent")
        args = parser.parse_args()

        thread_id = str(uuid.uuid4())
        logger.info(f"Using Thread ID: {thread_id}")

        initial_messages = [
            SystemMessage(content=system_prompt_content),
            HumanMessage(content=args.prompt),
        ]

        logger.info(f"🚀 Starting agent with prompt: {args.prompt}")

        final_state = await graph.ainvoke(
            {"messages": initial_messages},
            config={
                "configurable": {"thread_id": thread_id},
                "recursion_limit": 100
            },
        )

        final_message = final_state["messages"][-1]
        logger.info(f"\n✅ Job finished. Final Answer:")
        logger.info(final_message)

    except Exception as e:
        logger.critical(f"Unhandled exception during agent execution: {e}", exc_info=True)

    finally:
        logger.info("Cleaning up resources (browser and playwright)...")
        try:
            if browser:
                await browser.close()
                logger.info("Browser closed.")
            if playwright:
                await playwright.stop()
                logger.info("Playwright stopped.")
            logger.info("Cleanup successful.")
        except Exception as e:
            logger.error(f"Error during resource cleanup: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
