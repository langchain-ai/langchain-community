from __future__ import annotations

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


class ExtractInputsInput(BaseModel):
    selector: Optional[str] = Field(
        default="form, body",
        description="CSS selector to limit scope (defaults to all forms and body)",
    )


class ExtractInputsTool(BaseBrowserTool):
    name: str = "extract_inputs"
    description: str = (
        "Extract all input, select, and textarea fields with their names, ids, "
        "types, and current values."
    )
    args_schema: Type[BaseModel] = ExtractInputsInput

    def _run(
        self,
        selector: Optional[str] = "form, body",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        page = get_current_page(self.sync_browser)
        try:
            script = self._js_script()
            result = page.evaluate(script, selector)
            return str(result)
        except Exception as e:
            return f"ExtractInputs error: {str(e)}"

    async def _arun(
        self,
        selector: Optional[str] = "form, body",
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        page = await aget_current_page(self.async_browser)
        try:
            script = self._js_script()
            result = await page.evaluate(script, selector)
            return str(result)
        except Exception as e:
            return f"ExtractInputs async error: {str(e)}"

    def _js_script(self) -> str:
        return """
        (scopeSelector) => {
            const container = document.querySelector(scopeSelector) || document.body;
            const fields = container.querySelectorAll('input, select, textarea');
            const result = [];
            fields.forEach(el => {
                const fieldInfo = {
                    tag: el.tagName.toLowerCase(),
                    type: el.type || null,
                    name: el.getAttribute('name') || null,
                    id: el.id || null,
                    value: el.value || null,
                    checked:
                        el.type === "checkbox" || el.type === "radio"
                            ? el.checked
                            : undefined,
                };
                result.push(fieldInfo);
            });
            return result;
        }
        """
