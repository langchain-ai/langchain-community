"""BoxLite tools for secure code execution in isolated VMs.

BoxLite provides VM-level isolation using lightweight virtualization,
ensuring complete separation between the sandbox and host system.
"""

from langchain_community.tools.boxlite.browser import (
    BrowserTool as BoxliteBrowserTool,
)
from langchain_community.tools.boxlite.code_interpreter import (
    CodeInterpreterTool as BoxliteCodeInterpreterTool,
)
from langchain_community.tools.boxlite.computer_use import (
    ComputerUseTool as BoxliteComputerUseTool,
)
from langchain_community.tools.boxlite.sandbox import (
    SandboxTool as BoxliteSandboxTool,
)

__all__ = [
    "BoxliteBrowserTool",
    "BoxliteCodeInterpreterTool",
    "BoxliteComputerUseTool",
    "BoxliteSandboxTool",
]
