from __future__ import annotations

from typing import Any, Optional, Type

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from pydantic import BaseModel, Field, model_validator

from langchain_community.tools.playwright.base import BaseBrowserTool
from langchain_community.tools.playwright.utils import (
    aget_current_page,
    get_current_page,
)


class ExtractTextToolInput(BaseModel):
    """Input for ExtractTextTool."""

    selector: Optional[str] = Field(
        default=None,
        description="Optional CSS selector to extract text from a specific element.",
    )
    full_html: Optional[bool] = Field(
        default=False,
        description="If True, returns the outer HTML instead of plain text.",
    )

    @model_validator(mode="before")
    @classmethod
    def check_dependencies(cls, values: dict) -> Any:
        try:
            from bs4 import BeautifulSoup  # noqa
        except ImportError:
            raise ImportError(
                "The 'beautifulsoup4' package is required. "
                "Please install it with 'pip install beautifulsoup4'."
            )
        return values


class ExtractTextTool(BaseBrowserTool):
    """Tool for extracting text or HTML from the current webpage."""

    name: str = "extract_text"
    description: str = (
        "Extract text or HTML from the current webpage. "
        "Use optional 'selector' to narrow scope."
    )
    args_schema: Type[BaseModel] = ExtractTextToolInput

    def _run(
        self,
        selector: Optional[str] = None,
        full_html: bool = False,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        from bs4 import BeautifulSoup

        if self.sync_browser is None:
            raise ValueError("Synchronous browser not provided.")

        page = get_current_page(self.sync_browser)
        html_content = page.content()
        soup = BeautifulSoup(html_content, "lxml")

        if selector:
            element = soup.select_one(selector)
            if not element:
                return f"No element found for selector: {selector}"
            return element.decode() if full_html else element.get_text(strip=True)
        else:
            return soup.prettify() if full_html else " ".join(soup.stripped_strings)

    async def _arun(
        self,
        selector: Optional[str] = None,
        full_html: bool = False,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        from bs4 import BeautifulSoup

        if self.async_browser is None:
            raise ValueError("Asynchronous browser not provided.")

        page = await aget_current_page(self.async_browser)
        html_content = await page.content()
        soup = BeautifulSoup(html_content, "lxml")

        if selector:
            element = soup.select_one(selector)
            if not element:
                return f"No element found for selector: {selector}"
            return element.decode() if full_html else element.get_text(strip=True)
        else:
            return soup.prettify() if full_html else " ".join(soup.stripped_strings)
