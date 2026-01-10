"""Tool for API governance and OpenAPI specification validation."""

from pathlib import Path
from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class APIGovernorInput(BaseModel):
    """Input for the API Governor tool."""

    spec_content: str = Field(
        description="The OpenAPI specification content in YAML or JSON format"
    )
    policy: str = Field(
        default="standard",
        description="Governance policy level: 'lenient', 'standard', or 'strict'",
    )
    output_format: str = Field(
        default="markdown",
        description="Output format: 'markdown', 'json', or 'sarif'",
    )


class APIGovernorTool(BaseTool):
    """Tool that validates OpenAPI specifications against governance policies.

    This tool uses api-governor to perform automated API governance checks
    including security validation, naming conventions, breaking change detection,
    and documentation requirements.

    The tool checks for:
    - Security issues (missing auth, weak schemes)
    - Naming convention violations
    - Documentation gaps
    - Error format consistency
    - Breaking changes (when baseline provided)

    Install the underlying package with: pip install api-governor
    """

    name: str = "api_governor"
    description: str = (
        "Validates OpenAPI specifications against governance policies. "
        "Checks for security issues, naming conventions, documentation gaps, "
        "and API design best practices. Useful for API review automation, "
        "CI/CD governance gates, and ensuring API consistency. "
        "Input should be OpenAPI spec content in YAML or JSON format."
    )
    args_schema: Type[BaseModel] = APIGovernorInput

    def _run(
        self,
        spec_content: str,
        policy: str = "standard",
        output_format: str = "markdown",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Validate an OpenAPI specification against governance policies.

        Args:
            spec_content: The OpenAPI spec content in YAML or JSON format.
            policy: Governance policy level ('lenient', 'standard', 'strict').
            output_format: Output format ('markdown', 'json', or 'sarif').
            run_manager: Callback manager for the tool run.

        Returns:
            Governance validation results in the specified format.
        """
        try:
            from api_governor import APIGovernor
        except ImportError:
            return (
                "api-governor package is not installed. "
                "Install it with: pip install api-governor"
            )

        import tempfile

        # Determine file extension based on content
        ext = ".yaml"
        content_stripped = spec_content.strip()
        if content_stripped.startswith("{"):
            ext = ".json"

        # Write spec content to a temporary file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=ext, delete=False
        ) as tmp_file:
            tmp_file.write(spec_content)
            tmp_path = Path(tmp_file.name)

        try:
            # Run governance checks
            governor = APIGovernor(spec_path=tmp_path, policy=policy)
            result = governor.run()

            if output_format == "json":
                from api_governor import JSONFormatter

                formatter = JSONFormatter(result)
                return formatter.format()
            elif output_format == "sarif":
                from api_governor import SARIFFormatter

                formatter = SARIFFormatter(result)
                import json

                return json.dumps(formatter.to_sarif(), indent=2)
            else:
                # Default markdown output
                output_parts = [f"# API Governance Report\n"]
                output_parts.append(f"**Status:** {result.status}\n")
                output_parts.append(f"**Policy:** {policy}\n\n")

                if result.findings:
                    output_parts.append("## Findings\n")
                    for finding in result.findings:
                        output_parts.append(
                            f"- **[{finding.severity.value}]** {finding.message}\n"
                        )
                        if finding.path:
                            output_parts.append(f"  - Path: `{finding.path}`\n")
                        if finding.recommendation:
                            output_parts.append(
                                f"  - Recommendation: {finding.recommendation}\n"
                            )
                else:
                    output_parts.append("No governance issues found.\n")

                output_parts.append("\n## Summary\n")
                output_parts.append(f"- Blockers: {len(result.blockers)}\n")
                output_parts.append(f"- Majors: {len(result.majors)}\n")
                output_parts.append(f"- Minors: {len(result.minors)}\n")

                return "".join(output_parts)

        finally:
            # Cleanup temporary file
            tmp_path.unlink(missing_ok=True)
