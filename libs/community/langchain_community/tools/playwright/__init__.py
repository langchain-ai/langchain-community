"""Browser tools and toolkit."""

from langchain_community.tools.playwright.check import CheckTool
from langchain_community.tools.playwright.click import ClickTool
from langchain_community.tools.playwright.current_page import CurrentWebPageTool
from langchain_community.tools.playwright.extract_dom_tree import ExtractDOMTreeTool
from langchain_community.tools.playwright.extract_hyperlinks import (
    ExtractHyperlinksTool,
)
from langchain_community.tools.playwright.extract_text import ExtractTextTool
from langchain_community.tools.playwright.get_elements import GetElementsTool
from langchain_community.tools.playwright.input_text import InputTextTool
from langchain_community.tools.playwright.navigate import NavigateTool
from langchain_community.tools.playwright.navigate_back import NavigateBackTool
from langchain_community.tools.playwright.press_key import PressKeyTool
from langchain_community.tools.playwright.screenshot import ScreenshotTool
from langchain_community.tools.playwright.scroll import ScrollTool
from langchain_community.tools.playwright.select_option import SelectOptionTool

__all__ = [
    "NavigateTool",
    "NavigateBackTool",
    "ExtractTextTool",
    "ExtractHyperlinksTool",
    "ExtractDOMTreeTool",
    "GetElementsTool",
    "ClickTool",
    "CurrentWebPageTool",
    "InputTextTool",
    "PressKeyTool",
    "SelectOptionTool",
    "CheckTool",
    "ScreenshotTool",
    "ScrollTool",
]
