"""Computer use tool for GUI automation in isolated VM desktop."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Literal, Optional, Type, Union

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

if TYPE_CHECKING:
    from boxlite import ComputerBox


class ComputerUseInput(BaseModel):
    """Input schema for ComputerUseTool."""

    action: Literal[
        "screenshot",
        "click",
        "double_click",
        "right_click",
        "type",
        "key",
        "move",
        "drag",
        "scroll",
        "cursor_position",
        "screen_size",
    ] = Field(description="The action to perform")
    x: int = Field(default=0, description="X coordinate for mouse actions")
    y: int = Field(default=0, description="Y coordinate for mouse actions")
    text: str = Field(
        default="", description="Text for type action or key name for key action"
    )
    end_x: int = Field(default=0, description="End X coordinate for drag action")
    end_y: int = Field(default=0, description="End Y coordinate for drag action")
    direction: Literal["up", "down", "left", "right"] = Field(
        default="down",
        description="Scroll direction",
    )
    amount: int = Field(default=3, description="Scroll amount (number of clicks)")


class ComputerUseTool(BaseTool):
    """Tool for GUI automation in a secure, isolated VM desktop environment.

    BoxLite ComputerBox provides a full desktop environment (xfce) running inside
    an isolated VM. This tool allows AI agents to interact with graphical
    applications through mouse and keyboard actions.

    Setup:
        Install ``boxlite``:

        .. code-block:: bash

            pip install boxlite

    Instantiation:
        .. code-block:: python

            from langchain_community.tools.boxlite import ComputerUseTool

            tool = ComputerUseTool()

            # With custom resources
            tool = ComputerUseTool(memory=4096, cpus=4)

    Invocation with args:
        .. code-block:: python

            # Take a screenshot
            result = await tool.ainvoke({"action": "screenshot"})
            # Returns: {"type": "image", "data": "base64_png_data..."}

            # Click at coordinates
            result = await tool.ainvoke({"action": "click", "x": 100, "y": 200})

            # Type text
            result = await tool.ainvoke({"action": "type", "text": "Hello World"})

            # Press a key or key combination
            result = await tool.ainvoke({"action": "key", "text": "ctrl+c"})

    Invocation with ToolCall:
        .. code-block:: python

            tool.invoke({
                "args": {"action": "screenshot"},
                "id": "1",
                "name": tool.name,
                "type": "tool_call",
            })
    """

    name: str = "computer_use"
    description: str = (
        "Control a virtual desktop computer with mouse and keyboard. "
        "Use this for GUI automation, visual tasks, or testing graphical applications. "
        "Actions: screenshot (returns base64 PNG), click/double_click/right_click (at x,y), "
        "type (text input), key (keyboard keys like 'Return', 'ctrl+c'), "
        "move (mouse to x,y), drag (from x,y to end_x,end_y), "
        "scroll (at x,y in direction with amount), cursor_position, screen_size. "
        "The desktop runs in an isolated VM for security."
    )
    args_schema: Type[BaseModel] = ComputerUseInput

    memory: int = Field(default=4096, description="Memory in MiB")
    cpus: int = Field(default=4, description="Number of CPU cores")
    monitor_https_port: int = Field(
        default=3001,
        description="HTTPS port for web-based desktop access",
    )

    _desktop: Optional[ComputerBox] = PrivateAttr(default=None)
    _initialized: bool = PrivateAttr(default=False)

    async def _ensure_desktop(self) -> "ComputerBox":
        """Ensure desktop is initialized and ready."""
        try:
            from boxlite import ComputerBox
        except ImportError as e:
            raise ImportError(
                "Unable to import boxlite, please install with `pip install boxlite`."
            ) from e

        if self._desktop is None:
            self._desktop = ComputerBox(
                memory=self.memory,
                cpu=self.cpus,
                monitor_https_port=self.monitor_https_port,
            )
            await self._desktop.__aenter__()
            await self._desktop.wait_until_ready()
            self._initialized = True
        return self._desktop

    async def _cleanup(self) -> None:
        """Cleanup desktop resources."""
        if self._desktop is not None:
            await self._desktop.__aexit__(None, None, None)
            self._desktop = None
            self._initialized = False

    async def _arun(
        self,
        action: str,
        x: int = 0,
        y: int = 0,
        text: str = "",
        end_x: int = 0,
        end_y: int = 0,
        direction: str = "down",
        amount: int = 3,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Union[dict[str, Any], str]:
        """Execute GUI action asynchronously.

        Args:
            action: The action to perform.
            x: X coordinate for mouse actions.
            y: Y coordinate for mouse actions.
            text: Text for type/key actions.
            end_x: End X coordinate for drag.
            end_y: End Y coordinate for drag.
            direction: Scroll direction.
            amount: Scroll amount.
            run_manager: Callback manager.

        Returns:
            Action result (screenshot returns dict with base64 image).
        """
        try:
            desktop = await self._ensure_desktop()

            if action == "screenshot":
                data = await desktop.screenshot()
                return {"type": "image", "data": data.get("data", "")}

            elif action == "click":
                await desktop.mouse_move(x, y)
                await desktop.left_click()
                return f"Clicked at ({x}, {y})"

            elif action == "double_click":
                await desktop.mouse_move(x, y)
                await desktop.double_click()
                return f"Double-clicked at ({x}, {y})"

            elif action == "right_click":
                await desktop.mouse_move(x, y)
                await desktop.right_click()
                return f"Right-clicked at ({x}, {y})"

            elif action == "type":
                await desktop.type(text)
                return f"Typed: {text}"

            elif action == "key":
                await desktop.key(text)
                return f"Pressed key: {text}"

            elif action == "move":
                await desktop.mouse_move(x, y)
                return f"Moved cursor to ({x}, {y})"

            elif action == "drag":
                await desktop.left_click_drag(x, y, end_x, end_y)
                return f"Dragged from ({x}, {y}) to ({end_x}, {end_y})"

            elif action == "scroll":
                await desktop.scroll(x, y, direction, amount=amount)  # type: ignore[arg-type]
                return f"Scrolled {direction} by {amount} at ({x}, {y})"

            elif action == "cursor_position":
                pos_x, pos_y = await desktop.cursor_position()
                return {"x": pos_x, "y": pos_y}

            elif action == "screen_size":
                width, height = await desktop.get_screen_size()
                return {"width": width, "height": height}

            else:
                return f"Unknown action: {action}"

        except Exception as e:
            return f"Computer use error: {e!r}"

    def _run(
        self,
        action: str,
        x: int = 0,
        y: int = 0,
        text: str = "",
        end_x: int = 0,
        end_y: int = 0,
        direction: str = "down",
        amount: int = 3,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Union[dict[str, Any], str]:
        """Execute GUI action synchronously."""
        return asyncio.get_event_loop().run_until_complete(
            self._arun(
                action, x, y, text, end_x, end_y, direction, amount, run_manager=None
            )
        )
