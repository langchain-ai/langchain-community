"""Sandbox tool for secure command execution in isolated VMs."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Optional, Type

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from boxlite import SimpleBox


class SandboxInput(BaseModel):
    """Input schema for SandboxTool."""

    command: str = Field(description="The command to execute (e.g., 'ls', 'python')")
    args: list[str] = Field(
        default_factory=list,
        description="Arguments to pass to the command",
    )


class SandboxTool(BaseTool):
    """Tool for executing commands in a secure, isolated VM sandbox.

    BoxLite provides VM-level isolation using lightweight virtualization,
    ensuring complete separation between the sandbox and host system.

    Setup:
        Install ``boxlite``:

        .. code-block:: bash

            pip install boxlite

    Instantiation:
        .. code-block:: python

            from langchain_community.tools.boxlite import SandboxTool

            # Default Alpine Linux sandbox
            tool = SandboxTool()

            # Custom image with more resources
            tool = SandboxTool(
                image="ubuntu:22.04",
                memory_mib=2048,
                cpus=2,
            )

    Invocation with args:
        .. code-block:: python

            # Simple command
            result = await tool.ainvoke({"command": "echo", "args": ["hello"]})
            # Returns: {"exit_code": 0, "stdout": "hello\\n", "stderr": ""}

            # Complex command
            result = await tool.ainvoke({
                "command": "sh",
                "args": ["-c", "ls -la /"]
            })

    Invocation with ToolCall:
        .. code-block:: python

            tool.invoke({
                "args": {"command": "whoami", "args": []},
                "id": "1",
                "name": tool.name,
                "type": "tool_call",
            })
    """

    name: str = "sandbox"
    description: str = (
        "Execute commands in a secure, isolated VM sandbox. "
        "Use this for running untrusted code, system commands, or any operation "
        "that requires isolation from the host. The sandbox provides full Linux "
        "environment with configurable container images. "
        "Input: command name and optional arguments. "
        "Output: exit_code, stdout, and stderr."
    )
    args_schema: Type[BaseModel] = SandboxInput

    image: str = Field(
        default="alpine:latest",
        description="Container image to use (e.g., 'alpine:latest', 'ubuntu:22.04')",
    )
    memory_mib: int = Field(
        default=2048,
        description="Memory limit in MiB (minimum 1024 recommended for stable operation)",
    )
    cpus: int = Field(
        default=1,
        description="Number of CPU cores",
    )
    working_dir: Optional[str] = Field(
        default=None,
        description="Working directory inside the sandbox",
    )
    env: list[tuple[str, str]] = Field(
        default_factory=list,
        description="Environment variables as (key, value) tuples",
    )

    _box: Optional[SimpleBox] = None

    async def _arun(
        self,
        command: str,
        args: Optional[list[str]] = None,
        *,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> dict[str, Any]:
        """Execute command asynchronously in sandbox.

        Args:
            command: The command to execute.
            args: Arguments to pass to the command.
            run_manager: Callback manager for async operations.

        Returns:
            Dict with exit_code, stdout, and stderr.
        """
        try:
            from boxlite import SimpleBox
        except ImportError as e:
            raise ImportError(
                "Unable to import boxlite, please install with `pip install boxlite`."
            ) from e

        args = args or []

        try:
            async with SimpleBox(
                image=self.image,
                memory_mib=self.memory_mib,
                cpus=self.cpus,
                working_dir=self.working_dir,
                env=self.env,
            ) as box:
                result = await box.exec(command, *args)
                return {
                    "exit_code": result.exit_code,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
        except Exception as e:
            return {
                "exit_code": -1,
                "stdout": "",
                "stderr": f"Sandbox error: {e!r}",
            }

    def _run(
        self,
        command: str,
        args: Optional[list[str]] = None,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> dict[str, Any]:
        """Execute command synchronously in sandbox.

        Args:
            command: The command to execute.
            args: Arguments to pass to the command.
            run_manager: Callback manager for sync operations.

        Returns:
            Dict with exit_code, stdout, and stderr.
        """
        return asyncio.get_event_loop().run_until_complete(
            self._arun(command, args, run_manager=None)
        )
