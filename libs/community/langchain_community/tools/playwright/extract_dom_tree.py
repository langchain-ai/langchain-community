from __future__ import annotations

import json
from typing import Optional, Type

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


class ExtractDOMTreeToolInput(BaseModel):
    """Input for ExtractDOMTreeTool."""

    selector: Optional[str] = Field(
        default=None,
        description=(
            "Optional CSS selector to scope the DOM extraction, "
            "like 'main' or '#content'."
        ),
    )
    full_tree: Optional[bool] = Field(
        default=False,
        description=(
            "If true, returns the full DOM tree; otherwise returns only key elements "
            "like metadata, forms, and interactive items."
        ),
    )


class ExtractDOMTreeTool(BaseBrowserTool):
    name: str = "extract_dom_tree"
    description: str = (
        "Extract a concise DOM tree or key page elements from the current webpage, "
        "including interactive elements like buttons, forms, and links. "
        "Use to understand structure before interacting."
    )
    args_schema: Type[BaseModel] = ExtractDOMTreeToolInput
    playwright_timeout: float = 30_000

    def _dom_extract_script(self) -> str:
        return """
         (args) => {
            const { selector, fullTree } = args;
            function isInteractive(el) {
                const tag = el.tagName?.toLowerCase();
                const roles = [
                    "button",
                    "link",
                    "checkbox",
                    "radio",
                    "combobox",
                    "tab",
                    "menuitem",
                ];
                const interTags = ['a','button','input','select','textarea','summary'];
                if (interTags.includes(tag)) return true;
                if (roles.includes(el.getAttribute?.('role'))) return true;
                if (el.hasAttribute?.('onclick') || el.onclick) return true;
                return false;
            }

            function getSelectorSuggestions(el) {
                const tag = el.tagName.toLowerCase();
                const s = [];
                if (el.id) s.push(`#${el.id}`);
                if (el.getAttribute("name")) {
                    s.push(`${tag}[name="${el.getAttribute('name')}"]`);
                }
                if (el.getAttribute("data-testid")) {
                    s.push(`[data-testid="${el.getAttribute('data-testid')}"]`);
                }
                if (
                    (tag === "button" || tag === "a") &&
                    el.innerText?.trim().length < 20
                ) {
                    s.push(`${tag}:has-text("${el.innerText.trim()}")`);
                }
                return s.slice(0, 2);
            }

            const summary = {
                metadata: {
                    url: document.location.href,
                    title: document.title,
                    description:
                        document.querySelector('meta[name="description"]')?.content ||
                        null,
                    lang: document.documentElement.lang || null
                },
                forms: [],
                searchBoxes: [],
                topInteractiveElements: []
            };

            const root = selector ? document.querySelector(selector) : document.body;
            if (!root) return { error: "Selector root not found" };

            const inputs = root.querySelectorAll('input, select, textarea, button, a');
            inputs.forEach(el => {
                if (!isInteractive(el)) return;

                const data = {
                    tag: el.tagName.toLowerCase(),
                    text: el.innerText?.trim().slice(0, 60) || null,
                    attrs: {},
                    selectors: getSelectorSuggestions(el)
                };

                ["name", "type", "id", "placeholder", "value", "href", "role"].forEach(
                    (attr) => {
                        if (el.getAttribute(attr)) {
                            data.attrs[attr] = el.getAttribute(attr).slice(0, 40);
                        }
                    }
                );

                summary.topInteractiveElements.push(data);

                if ((el.getAttribute('type') === 'search') || 
                    ['q', 'query'].includes(el.getAttribute('name')) || 
                    el.getAttribute('placeholder')?.toLowerCase().includes('search')) {
                    summary.searchBoxes.push(data);
                }
            });

            const forms = root.querySelectorAll('form');
            forms.forEach(form => {
                const inputs = form.querySelectorAll('input, select, textarea');
                summary.forms.push({
                    method: form.method || 'get',
                    action: form.action || null,
                    fields: Array.from(inputs).map(i => ({
                        name: i.name || null,
                        type: i.type || i.tagName.toLowerCase()
                    }))
                });
            });

            if (fullTree) {
                function slim(el, depth = 0, maxDepth = 3) {
                    if (depth > maxDepth || !el.tagName) return null;
                    const node = {
                        tag: el.tagName.toLowerCase(),
                        children: []
                    };
                    if (el.innerText?.trim()) {
                        node.text = el.innerText.trim().slice(0, 40);
                    }
                    for (let i = 0; i < Math.min(5, el.children.length); i++) {
                        const child = slim(el.children[i], depth + 1);
                        if (child) node.children.push(child);
                    }
                    return node;
                }
                summary.tree = slim(root);
            }

            return summary;
        }
        """

    def _run(
        self,
        selector: Optional[str] = None,
        full_tree: bool = False,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        if self.sync_browser is None:
            raise ValueError(f"Synchronous browser not provided to {self.name}")
        page = get_current_page(self.sync_browser)
        try:
            result = page.evaluate(
                self._dom_extract_script(),
                {"selector": selector, "fullTree": full_tree},
            )
            return json.dumps(result, separators=(",", ":"))
        except Exception as e:
            return f"Error extracting DOM tree: {str(e)}"

    async def _arun(
        self,
        selector: Optional[str] = None,
        full_tree: bool = False,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        if self.async_browser is None:
            raise ValueError(f"Asynchronous browser not provided to {self.name}")
        page = await aget_current_page(self.async_browser)
        try:
            result = await page.evaluate(
                self._dom_extract_script(),
                {"selector": selector, "fullTree": full_tree},
            )
            return json.dumps(result, separators=(",", ":"))
        except Exception as e:
            return f"Error extracting DOM tree: {str(e)}"
