from __future__ import annotations

import asyncio
import base64
import datetime
import json
import os
import time
from typing import TYPE_CHECKING, Optional, Type

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from pydantic import BaseModel, Field

from langchain_community.tools.playwright.base import BaseBrowserTool
from langchain_community.tools.playwright.utils import (
    aget_current_page,
    get_current_page,
)

if TYPE_CHECKING:
    from playwright.async_api import Page as AsyncPage
    from playwright.sync_api import Page


class ScreenshotToolInput(BaseModel):
    """Input for ScreenshotTool."""

    selector: Optional[str] = Field(
        None,
        description=(
            "Optional CSS selector for taking a screenshot of a specific element. "
            "If not provided, takes a screenshot of the full page."
        ),
    )
    output_path: Optional[str] = Field(
        None,
        description=(
            "The directory where the screenshot should be saved. "
            "If not provided, defaults to './screenshots/'."
        ),
    )
    filename: Optional[str] = Field(
        None,
        description=(
            "Optional filename for the screenshot. If not provided, a name will be "
            "generated based on URL and timestamp."
        ),
    )
    full_page: bool = Field(
        False,
        description=(
            "Whether to take a screenshot of the full scrollable page or just the "
            "viewport. Only relevant for full page screenshots (when no selector "
            "is provided)."
        ),
    )
    include_timestamp: bool = Field(
        True,
        description=(
            "Whether to include a timestamp in the filename to prevent overwriting."
        ),
    )
    return_base64: bool = Field(
        False,
        description=(
            "Whether to return the screenshot as a base64 encoded string "
            "instead of a file path."
        ),
    )
    wait_for_load: bool = Field(
        True,
        description=(
            "Whether to wait for the page to be fully loaded before taking the "
            "screenshot."
        ),
    )
    wait_time: float = Field(
        2.0,
        description=(
            "Additional time in seconds to wait after page load events "
            "before taking the screenshot."
        ),
    )


class ScreenshotTool(BaseBrowserTool):
    """Tool for taking screenshots of the current page or specific elements."""

    name: str = "take_screenshot"
    description: str = (
        "Take a screenshot of the current page or a specific element. "
        "Can save to a specified path and return the filename. "
        "Parameters: output_path (directory), filename (optional), "
        "full_page (default: false). Returns a JSON object with the file_path "
        "where the screenshot was saved."
    )
    args_schema: Type[BaseModel] = ScreenshotToolInput

    visible_only: bool = True
    """Whether to consider only visible elements when selector is provided."""
    playwright_strict: bool = False
    """Whether to employ Playwright's strict mode when selecting elements."""
    playwright_timeout: float = 5_000
    """Timeout (in ms) for Playwright to wait for element to be ready."""
    default_screenshots_dir: str = "./screenshots"
    """Default directory to save screenshots if not specified."""

    def _selector_effective(self, selector: str) -> str:
        if not self.visible_only:
            return selector
        return f"{selector} >> visible=1"

    def _generate_filename(
        self,
        page_title: str,
        page_url: str,
        selector: Optional[str] = None,
        custom_filename: Optional[str] = None,
        include_timestamp: bool = True,
    ) -> str:
        """Generate a filename for the screenshot based on URL and timestamp."""
        if custom_filename:
            # If custom filename already has .png extension
            if custom_filename.lower().endswith(".png"):
                base_filename = custom_filename
            else:
                base_filename = f"{custom_filename}.png"
        else:
            # Generate filename from URL/title
            # Clean the URL to create a filename
            if page_title:
                clean_title = "".join(
                    c if c.isalnum() or c in ["-", "_"] else "_" for c in page_title
                )
                clean_title = clean_title[:30]  # Limit length
                base = clean_title
            else:
                # Use URL as fallback
                url_parts = page_url.split("//")[-1].split("/")
                domain = url_parts[0].split(":")[0]  # Remove port if present
                base = domain

            if selector:
                # Add selector info to filename (simplified)
                selector_part = selector.replace("#", "id_").replace(".", "class_")
                selector_part = "".join(
                    c if c.isalnum() or c in ["-", "_"] else "_" for c in selector_part
                )
                selector_part = selector_part[:20]  # Limit length
                base = f"{base}_{selector_part}"

            base_filename = f"{base}.png"

        # Add timestamp if requested
        if include_timestamp:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            # Insert timestamp before the extension
            parts = base_filename.rsplit(".", 1)
            if len(parts) > 1:
                return f"{parts[0]}_{timestamp}.{parts[1]}"
            else:
                return f"{base_filename}_{timestamp}.png"

        return base_filename

    def _ensure_directory_exists(self, directory_path: str) -> None:
        """Ensure the directory exists, creating it if necessary."""
        os.makedirs(directory_path, exist_ok=True)

    def _wait_for_page_load(self, page: Page) -> None:
        """Wait for page to be fully loaded."""
        # First wait for network to be idle (no more than 0 connections for 500ms)
        try:
            page.wait_for_load_state("networkidle", timeout=10000)
        except Exception:
            # If timeout, the page might still be usable
            pass

        # Then wait for DOM content to be loaded
        try:
            page.wait_for_load_state("domcontentloaded", timeout=5000)
        except Exception:
            # If timeout, the page might still be usable
            pass

    async def _async_wait_for_page_load(self, page: AsyncPage) -> None:
        """Wait for page to be fully loaded in async context."""
        # First wait for network to be idle (no more than 0 connections for 500ms)
        try:
            await page.wait_for_load_state("networkidle", timeout=10000)
        except Exception:
            # If timeout, the page might still be usable
            pass

        # Then wait for DOM content to be loaded
        try:
            await page.wait_for_load_state("domcontentloaded", timeout=5000)
        except Exception:
            # If timeout, the page might still be usable
            pass

    def _run(
        self,
        selector: Optional[str] = None,
        output_path: Optional[str] = None,
        filename: Optional[str] = None,
        full_page: bool = False,
        include_timestamp: bool = True,
        return_base64: bool = False,
        wait_for_load: bool = True,
        wait_time: float = 2.0,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        if self.sync_browser is None:
            raise ValueError(f"Synchronous browser not provided to {self.name}")

        page = get_current_page(self.sync_browser)

        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

        try:
            # Ensure page is focused and ready
            page.bring_to_front()

            # Wait for page to be fully loaded if requested
            if wait_for_load:
                self._wait_for_page_load(page)
                # Additional wait time to ensure all animations/renders complete
                time.sleep(wait_time)

            # Determine directory to save screenshot
            screenshots_dir = (
                output_path if output_path else self.default_screenshots_dir
            )
            self._ensure_directory_exists(screenshots_dir)

            # Get page info for filename generation
            try:
                page_title = page.title()
            except Exception:
                page_title = ""

            page_url = page.url

            # If returning base64, don't need to save to file
            if return_base64:
                try:
                    if selector:
                        # Take screenshot of a specific element
                        selector_effective = self._selector_effective(selector=selector)
                        element = page.locator(selector_effective)

                        # Make sure element exists
                        element_count = element.count()
                        if element_count == 0:
                            return json.dumps(
                                {
                                    "success": False,
                                    "error": (
                                        f"Unable to find element '{selector}' "
                                        "for screenshot"
                                    ),
                                }
                            )

                        # Get the first element that matches the selector
                        first_element = element.first

                        # Scroll element into view before screenshot
                        first_element.scroll_into_view_if_needed()
                        time.sleep(0.5)  # Short delay after scrolling

                        # === START OF FIX ===
                        # Get image data as bytes, not save to file
                        img_data = first_element.screenshot()
                        # === END OF FIX ===
                    else:
                        # Take full page screenshot
                        img_data = page.screenshot(full_page=full_page)

                    img_base64 = base64.b64encode(img_data).decode("utf-8")
                    return f"data:image/png;base64,{img_base64}"
                except PlaywrightTimeoutError:
                    return json.dumps(
                        {
                            "success": False,
                            "error": (
                                f"Unable to find element '{selector}' for screenshot"
                            ),
                        }
                    )
                except Exception as e:
                    return json.dumps(
                        {
                            "success": False,
                            "error": f"Error taking screenshot: {str(e)}",
                        }
                    )

            # Generate filename with proper path
            generated_filename = self._generate_filename(
                page_title, page_url, selector, filename, include_timestamp
            )
            full_path = os.path.join(screenshots_dir, generated_filename)

            # Take screenshot based on parameters
            if selector:
                # Take screenshot of a specific element
                selector_effective = self._selector_effective(selector=selector)
                try:
                    element = page.locator(selector_effective)

                    # Make sure element exists
                    element_count = element.count()
                    if element_count == 0:
                        return json.dumps(
                            {
                                "success": False,
                                "error": (
                                    f"Unable to find element '{selector}' "
                                    "for screenshot"
                                ),
                            }
                        )

                    # Get the first element that matches the selector
                    first_element = element.first

                    # Scroll element into view before screenshot
                    first_element.scroll_into_view_if_needed()
                    time.sleep(0.5)  # Short delay after scrolling

                    first_element.screenshot(path=full_path)
                    return json.dumps(
                        {
                            "success": True,
                            "message": "Screenshot saved successfully",
                            "file_path": full_path,
                            "type": "element",
                        }
                    )
                except Exception as e:
                    return json.dumps(
                        {
                            "success": False,
                            "error": f"Error taking screenshot of element: {str(e)}",
                            "suggestion": (
                                "Try a different selector or check if the element "
                                "is visible"
                            ),
                        }
                    )
            else:
                # Take full page screenshot
                try:
                    page.screenshot(path=full_path, full_page=full_page)
                    return json.dumps(
                        {
                            "success": True,
                            "message": "Screenshot saved successfully",
                            "file_path": full_path,
                            "type": "full_page" if full_page else "viewport",
                        }
                    )
                except Exception as e:
                    return json.dumps(
                        {
                            "success": False,
                            "error": f"Error taking screenshot: {str(e)}",
                            "suggestion": (
                                "The page might not be fully loaded. "
                                "Try increasing wait_time."
                            ),
                        }
                    )

        except Exception as e:
            return json.dumps(
                {"success": False, "error": f"Error taking screenshot: {str(e)}"}
            )

    async def _arun(
        self,
        selector: Optional[str] = None,
        output_path: Optional[str] = None,
        filename: Optional[str] = None,
        full_page: bool = False,
        include_timestamp: bool = True,
        return_base64: bool = False,
        wait_for_load: bool = True,
        wait_time: float = 2.0,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        if self.async_browser is None:
            raise ValueError(f"Asynchronous browser not provided to {self.name}")

        page = await aget_current_page(self.async_browser)

        from playwright.async_api import TimeoutError as PlaywrightTimeoutError

        try:
            # Ensure page is focused and ready
            await page.bring_to_front()

            # Wait for page to be fully loaded if requested
            if wait_for_load:
                await self._async_wait_for_page_load(page)
                # Additional wait time to ensure all animations/renders complete
                await asyncio.sleep(wait_time)

            # Determine directory to save screenshot
            screenshots_dir = (
                output_path if output_path else self.default_screenshots_dir
            )
            self._ensure_directory_exists(screenshots_dir)

            # Get page info for filename generation
            try:
                page_title = await page.title()
            except Exception:
                page_title = ""

            page_url = page.url

            # If returning base64, don't need to save to file
            if return_base64:
                try:
                    if selector:
                        # Take screenshot of a specific element
                        selector_effective = self._selector_effective(selector=selector)
                        element = page.locator(selector_effective)

                        # Make sure element exists
                        element_count = await element.count()
                        if element_count == 0:
                            return json.dumps(
                                {
                                    "success": False,
                                    "error": (
                                        f"Unable to find element '{selector}' "
                                        "for screenshot"
                                    ),
                                }
                            )

                        # Get the first element that matches the selector
                        first_element = element.first

                        # Scroll element into view before screenshot
                        await first_element.scroll_into_view_if_needed()
                        await asyncio.sleep(0.5)  # Short delay after scrolling

                        img_data = await first_element.screenshot()
                        img_base64 = base64.b64encode(img_data).decode("utf-8")
                        return f"data:image/png;base64,{img_base64}"
                    else:
                        # Take full page screenshot
                        img_data = await page.screenshot(full_page=full_page)

                        img_base64 = base64.b64encode(img_data).decode("utf-8")
                        return f"data:image/png;base64,{img_base64}"
                except PlaywrightTimeoutError:
                    return json.dumps(
                        {
                            "success": False,
                            "error": (
                                f"Unable to find element '{selector}' for screenshot"
                            ),
                        }
                    )
                except Exception as e:
                    return json.dumps(
                        {
                            "success": False,
                            "error": f"Error taking screenshot: {str(e)}",
                        }
                    )

            # Generate filename with proper path
            generated_filename = self._generate_filename(
                page_title, page_url, selector, filename, include_timestamp
            )
            full_path = os.path.join(screenshots_dir, generated_filename)

            # Take screenshot based on parameters
            if selector:
                # Take screenshot of a specific element
                selector_effective = self._selector_effective(selector=selector)
                try:
                    element = page.locator(selector_effective)

                    # Make sure element exists
                    element_count = await element.count()
                    if element_count == 0:
                        return json.dumps(
                            {
                                "success": False,
                                "error": (
                                    f"Unable to find element '{selector}' "
                                    "for screenshot"
                                ),
                            }
                        )

                    # Get the first element that matches the selector
                    first_element = element.first

                    # Scroll element into view before screenshot
                    await first_element.scroll_into_view_if_needed()
                    await asyncio.sleep(0.5)  # Short delay after scrolling

                    await first_element.screenshot(path=full_path)
                    return json.dumps(
                        {
                            "success": True,
                            "message": "Screenshot saved successfully",
                            "file_path": full_path,
                            "type": "element",
                        }
                    )
                except Exception as e:
                    return json.dumps(
                        {
                            "success": False,
                            "error": f"Error taking screenshot of element: {str(e)}",
                            "suggestion": (
                                "Try a different selector or check if the "
                                "element is visible"
                            ),
                        }
                    )
            else:
                # Take full page screenshot
                try:
                    await page.screenshot(path=full_path, full_page=full_page)
                    return json.dumps(
                        {
                            "success": True,
                            "message": "Screenshot saved successfully",
                            "file_path": full_path,
                            "type": "full_page" if full_page else "viewport",
                        }
                    )
                except Exception as e:
                    return json.dumps(
                        {
                            "success": False,
                            "error": f"Error taking screenshot: {str(e)}",
                            "suggestion": (
                                "The page might not be fully loaded. "
                                "Try increasing wait_time."
                            ),
                        }
                    )

        except Exception as e:
            return json.dumps(
                {"success": False, "error": f"Error taking screenshot: {str(e)}"}
            )
