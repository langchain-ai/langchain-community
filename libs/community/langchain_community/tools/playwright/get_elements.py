from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Type

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
    from playwright.sync_api import Page as SyncPage


class GetElementsToolInput(BaseModel):
    """Input for GetElementsTool."""

    selector: Optional[str] = Field(
        default="body",
        description=(
            "CSS selector like 'div', 'p', 'a', '#id', '.classname'. If not "
            "provided, defaults to 'body' to get the entire page."
        ),
    )
    attributes: Optional[List[str]] = Field(
        None,
        description=(
            "Set of attributes to retrieve for each element. If None, will extract "
            "standard useful attributes."
        ),
    )


def _build_script() -> str:
    return """
    ({ selector, requestedAttributes }) => {
        const elements = Array.from(document.querySelectorAll(selector));
        const results = [];

        const defaultAttributes = [
            'id', 'name', 'class', 'type', 'value', 'placeholder', 'href',
            'role', 'aria-label', 'title', 'alt', 'src', 'data-testid'
        ];

        const attributesToGet = requestedAttributes && requestedAttributes.length > 0 
            ? requestedAttributes 
            : defaultAttributes;

        for (const element of elements) {
            const result = {
                tag: element.tagName.toLowerCase(),
                text: element.innerText?.trim() || element.textContent?.trim() || null,
                isVisible: isElementVisible(element),
                attributes: {},
                boundingBox: element.getBoundingClientRect().toJSON()
            };

            for (const attr of attributesToGet) {
                if (element.hasAttribute(attr)) {
                    result.attributes[attr] = element.getAttribute(attr);
                }
            }

            if (isInteractiveElement(element)) {
                result.interactive = true;
                result.interactionType = getInteractionType(element);
            }

            results.push(result);
        }

        return results;

        function isElementVisible(el) {
            if (!el) return false;
            const style = window.getComputedStyle(el);
            return style.display !== 'none' && 
                   style.visibility !== 'hidden' && 
                   el.offsetWidth > 0 && 
                   el.offsetHeight > 0;
        }

        function isInteractiveElement(el) {
            const tags = [
                "a",
                "button",
                "input",
                "select",
                "textarea",
                "option",
                "label",
            ];
            const roles = ['button', 'link', 'checkbox', 'menuitem', 'tab', 'radio'];
            const role = el.getAttribute('role');
            return tags.includes(el.tagName.toLowerCase()) ||
                   (role && roles.includes(role)) ||
                   el.onclick || el.getAttribute('onclick') ||
                   el.getAttribute('tabindex') || el.getAttribute('contenteditable');
        }

        function getInteractionType(el) {
            const tag = el.tagName.toLowerCase();
            const type = el.getAttribute('type')?.toLowerCase();

            if (tag === 'a' || el.getAttribute('role') === 'link') return 'link';
            if (tag === "button" || el.getAttribute("role") === "button") {
                return "button";
            }
            if (tag === 'input') {
                if (
                    ["text", "email", "number", "password", "search"].includes(type)
                ) {
                    return "input";
                }
                if (["checkbox", "radio"].includes(type)) return type;
                if (["submit", "button"].includes(type)) return 'button';
            }
            if (tag === 'select') return 'select';
            if (tag === 'textarea') return 'input';
            return 'clickable';
        }
    }
    """


async def _aget_elements(
    page: AsyncPage, selector: str, attributes: Optional[Sequence[str]] = None
) -> List[Dict[str, Any]]:
    return await page.evaluate(
        _build_script(),
        {
            "selector": selector,
            "requestedAttributes": list(attributes) if attributes else None,
        },
    )


def _get_elements(
    page: SyncPage, selector: str, attributes: Optional[Sequence[str]] = None
) -> List[Dict[str, Any]]:
    return page.evaluate(
        _build_script(),
        {
            "selector": selector,
            "requestedAttributes": list(attributes) if attributes else None,
        },
    )


class GetElementsTool(BaseBrowserTool):
    name: str = "get_elements"
    description: str = (
        "Find elements on the current webpage using CSS selectors. "
        "Returns detailed information about matching elements including attributes, "
        "tag name, text content, interactivity status, and position. "
        "Defaults to 'body' if no selector is specified."
    )
    args_schema: Type[BaseModel] = GetElementsToolInput

    def _run(
        self,
        selector: str = "body",
        attributes: Optional[Sequence[str]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        if self.sync_browser is None:
            raise ValueError("Synchronous browser not initialized.")
        page = get_current_page(self.sync_browser)
        page.wait_for_load_state("load")
        try:
            results = _get_elements(page, selector, attributes)
            return json.dumps(
                {"elements": results, "count": len(results)}, ensure_ascii=False
            )
        except Exception as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    async def _arun(
        self,
        selector: str = "body",
        attributes: Optional[Sequence[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        if self.async_browser is None:
            raise ValueError("Asynchronous browser not initialized.")
        page = await aget_current_page(self.async_browser)
        try:
            results = await _aget_elements(page, selector, attributes)
            return json.dumps(
                {"elements": results, "count": len(results)}, ensure_ascii=False
            )
        except Exception as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)
