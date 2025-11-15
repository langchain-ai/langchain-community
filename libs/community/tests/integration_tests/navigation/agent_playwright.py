#!/usr/bin/env python3

import argparse
import asyncio
import os
import random
import uuid
from typing import Literal, TypedDict

import nest_asyncio
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from playwright.async_api import async_playwright
from pydantic import SecretStr

# Playwright tools
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
api_key = os.getenv("OPENROUTER_API_KEY")
base_url = os.getenv("OPENROUTER_BASE_URL")
site_url = os.getenv("YOUR_SITE_URL")
site_name = os.getenv("YOUR_SITE_NAME")

if not all([api_key, base_url]):
    raise ValueError("Required environment variables are not set")

nest_asyncio.apply()


# Define shared state
class AgentState(TypedDict):
    input: str
    step: Literal["planning", "navigating"]
    result: str
    steps: list[str]
    history: list[str]


async def main():
    # Set up LLM
    llm = ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model_name="mistralai/devstral-small-2505:free",
        temperature=0,
        streaming=True,
    )

    # Launch Playwright
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(
        # executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        args=[
            "--disable-gpu",
            "--disable-dev-shm-usage",
            "--disable-web-security",
            "--disable-blink-features=AutomationControlled",
            "--no-sandbox",
            "--disable-setuid-sandbox",
        ],
        headless=False,  # Optional: set to True if you want headless mode
    )
    page = await browser.new_page(no_viewport=True)

    # Create tools
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

    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools=tools, parallel_tool_calls=False)

    # Navigation prompt
    navigator_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="""You are a web automation agent that can browse websites and perform tasks as requested.

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
1. Navigate to requested URL. The URLS must start with 'https://'
2. Aways accept cookies when asked
3. Call extract_dom_tree ONCE to analyze the page structure
4. Examine the DOM to identify relevant elements (forms, buttons, inputs, etc.)
5. Perform ONE action based on the task requirements
6. ONLY call extract_dom_tree again if the page state has changed
7. Continue with the next action until the task is complete
8. Make sure ALL requirements are satisfied

GENERAL GUIDELINES:
- Extract the DOM tree ONCE after navigation to understand page structure
- Extract the DOM again ONLY after actions that change the page
- Use selectors found in the DOM extraction (don't hardcode selectors)
- Complete all requested actions in the proper sequence
- Take screenshots when explicitly requested or to show final results

COMMON WEB TASKS:
- For search tasks: Input text and press Enter to submit the query
- For form filling: Find form fields, enter data, and submit the form
- For navigation: Identify and click on relevant links or buttons
- For extraction: Extract text or data from relevant elements

IMPORTANT NOTES:
- For screenshots, use take_screenshot with simple parameters:
  take_screenshot with {"filename": "results"}
- Avoid using selector parameters for screenshots to prevent errors
- Use press_key to submit forms or trigger actions (e.g., Enter key)
- After inputting text in forms, always take the appropriate action to submit
- For any page, analyze the DOM first to discover the correct selectors
- When exporting to JSON, first try to capture and organize any inherent data structure (e.g. lists, tables, key–value pairs). 
  If the source offers no clear schema, fall back to extracting text and then wrap it in a sensible JSON format.

TOOL USAGE FORMAT:
When using tools, always use this format:
1. First, use navigate to go to the URL
2. Then use extract_dom_tree to understand the page structure
3. Use other tools as needed based on the task
4. Always check the results of each tool call

Example tool usage:
1. navigate({"url": "https://example.com"})
2. extract_dom_tree({})
3. click({"selector": "button.submit"})
"""
            ),
            HumanMessagePromptTemplate.from_template("{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # Navigator agent (executes tool calls)
    navigator_agent = AgentExecutor(
        agent=create_tool_calling_agent(llm_with_tools, tools, navigator_prompt),
        tools=tools,
        max_iterations=120,
        max_execution_time=360,
        verbose=True,  # <-- enable logging
    )

    # Build the LangGraph
    workflow = StateGraph(AgentState)

    async def planner_node(state: AgentState) -> AgentState:
        return {
            "input": state["input"],
            "step": "navigating",
            "result": "",
            "steps": [],
        }

    async def navigator_node(state: AgentState) -> AgentState:
        dom_tree = await ExtractDOMTreeTool(page=page, async_browser=browser).arun({})
        input_with_dom = f"{state['input']}\n\n[DOM]:\n{dom_tree}"

        steps = state.get("steps", [])
        history = state.get("history", [])
        current_input = input_with_dom

        while True:
            await page.wait_for_timeout(random.randint(1, 3))
            output = await navigator_agent.ainvoke({"input": current_input})

            for tool_call, result in output.get("intermediate_steps", []):
                step_text = f"Tool: {tool_call.tool}, Args: {tool_call.tool_input}, Result: {result}"
                steps.append(step_text)
                history.append(step_text)

            if "output" in output and output["output"]:
                break  # task complete
            else:
                # Refresh DOM and continue
                dom_tree = await ExtractDOMTreeTool(
                    page=page, async_browser=browser
                ).arun({})
                current_input = f"{state['input']}\n\n[DOM]:\n{dom_tree}"

        return {
            "input": state["input"],
            "step": "done",
            "result": output.get("output", ""),
            "steps": steps,
            "history": history,
        }

    checkpointer = MemorySaver()

    workflow.add_node("planner", planner_node)
    workflow.add_node("navigator", navigator_node)
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "navigator")
    workflow.add_edge("navigator", END)

    graph = workflow.compile(checkpointer=checkpointer)

    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", help="Prompt for the agent")
    args = parser.parse_args()

    # Run graph
    thread_id = str(uuid.uuid4())

    await graph.ainvoke(
        {
            "input": args.prompt,
            "step": "planning",
            "result": "",
            "steps": [],
            "history": [],
        },
        config={"configurable": {"thread_id": thread_id}},
    )

    print(f"\n✅ Job finished")

    await browser.close()
    await playwright.stop()


if __name__ == "__main__":
    asyncio.run(main())
