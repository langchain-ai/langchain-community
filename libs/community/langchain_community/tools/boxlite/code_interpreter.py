"""Code interpreter tool for secure Python execution in isolated VMs."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Optional, Type

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from boxlite import CodeBox


class CodeInterpreterInput(BaseModel):
    """Input schema for CodeInterpreterTool."""

    code: str = Field(description="Python code to execute")
    packages: list[str] = Field(
        default_factory=list,
        description="Python packages to install before execution (e.g., ['numpy', 'pandas'])",
    )


class CodeInterpreterTool(BaseTool):
    """Tool for executing Python code in a secure, isolated VM sandbox.

    BoxLite CodeBox provides a sandboxed Python environment with full isolation
    from the host system. Perfect for AI agents that need to run generated code
    safely.

    Setup:
        Install ``boxlite``:

        .. code-block:: bash

            pip install boxlite

    Instantiation:
        .. code-block:: python

            from langchain_community.tools.boxlite import CodeInterpreterTool

            # Default Python sandbox
            tool = CodeInterpreterTool()

            # Custom configuration
            tool = CodeInterpreterTool(
                image="python:3.11-slim",
                memory_mib=4096,
                cpus=2,
            )

    Invocation with args:
        .. code-block:: python

            # Simple code execution
            result = await tool.ainvoke({"code": "print(2 + 2)"})
            # Returns: "4\\n"

            # With package installation
            result = await tool.ainvoke({
                "code": "import pandas as pd; print(pd.__version__)",
                "packages": ["pandas"]
            })

    Invocation with ToolCall:
        .. code-block:: python

            tool.invoke({
                "args": {
                    "code": "import math; print(math.pi)",
                    "packages": []
                },
                "id": "1",
                "name": tool.name,
                "type": "tool_call",
            })
    """

    name: str = "code_interpreter"
    description: str = (
        "Execute Python code in a secure, isolated VM sandbox. "
        "Use this for running Python code, data analysis, calculations, "
        "or any Python-based task that requires isolation. "
        "Supports installing packages dynamically. "
        "The sandbox is ephemeral - state is not preserved between calls. "
        "Input: Python code string and optional list of packages to install. "
        "Output: combined stdout and stderr from execution."
    )
    args_schema: Type[BaseModel] = CodeInterpreterInput

    image: str = Field(
        default="python:slim",
        description="Python container image (e.g., 'python:slim', 'python:3.11')",
    )
    memory_mib: int = Field(
        default=2048,
        description="Memory limit in MiB",
    )
    cpus: int = Field(
        default=2,
        description="Number of CPU cores",
    )

    async def _arun(
        self,
        code: str,
        packages: Optional[list[str]] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Execute Python code asynchronously in sandbox.

        Args:
            code: Python code to execute.
            packages: Packages to install before execution.
            run_manager: Callback manager for async operations.

        Returns:
            Combined stdout and stderr output.
        """
        try:
            from boxlite import CodeBox
        except ImportError as e:
            raise ImportError(
                "Unable to import boxlite, please install with `pip install boxlite`."
            ) from e

        packages = packages or []

        try:
            async with CodeBox(
                image=self.image,
                memory_mib=self.memory_mib,
                cpus=self.cpus,
            ) as codebox:
                # Install requested packages
                if packages:
                    await codebox.install_packages(*packages)

                # Execute code
                result = await codebox.run(code)
                return result
        except Exception as e:
            return f"Code execution error: {e!r}"

    def _run(
        self,
        code: str,
        packages: Optional[list[str]] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute Python code synchronously in sandbox.

        Args:
            code: Python code to execute.
            packages: Packages to install before execution.
            run_manager: Callback manager for sync operations.

        Returns:
            Combined stdout and stderr output.
        """
        return asyncio.get_event_loop().run_until_complete(
            self._arun(code, packages, run_manager=None)
        )
