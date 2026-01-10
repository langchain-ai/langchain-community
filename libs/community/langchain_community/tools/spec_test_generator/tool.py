"""Tool for generating requirements and test cases from PRDs."""

from pathlib import Path
from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class SpecTestGeneratorInput(BaseModel):
    """Input for the Spec Test Generator tool."""

    prd_content: str = Field(
        description="The PRD (Product Requirements Document) content in markdown format"
    )
    output_format: str = Field(
        default="markdown",
        description="Output format: 'markdown', 'json', or 'gherkin'",
    )


class SpecTestGeneratorTool(BaseTool):
    """Tool that generates requirements and test cases from PRDs.

    This tool uses spec-test-generator to convert Product Requirements Documents
    into formal requirements with stable IDs, test cases, and traceability matrices.

    The tool generates:
    - Requirements with fingerprint-based stable IDs (REQ-xxxx)
    - Test cases linked to requirements (TEST-xxxx)
    - Traceability information

    Install the underlying package with: pip install spec-test-generator
    """

    name: str = "spec_test_generator"
    description: str = (
        "Converts PRDs (Product Requirements Documents) into formal requirements "
        "and test cases with stable, traceable IDs. Useful for generating test "
        "specifications, requirements documentation, and traceability matrices "
        "from product requirement documents. Input should be PRD content in markdown."
    )
    args_schema: Type[BaseModel] = SpecTestGeneratorInput

    def _run(
        self,
        prd_content: str,
        output_format: str = "markdown",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Generate requirements and test cases from PRD content.

        Args:
            prd_content: The PRD content in markdown format.
            output_format: Output format ('markdown', 'json', or 'gherkin').
            run_manager: Callback manager for the tool run.

        Returns:
            Generated requirements and test cases in the specified format.
        """
        try:
            from spec_test_generator import SpecTestGenerator
        except ImportError:
            return (
                "spec-test-generator package is not installed. "
                "Install it with: pip install spec-test-generator"
            )

        import tempfile

        # Write PRD content to a temporary file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False
        ) as tmp_file:
            tmp_file.write(prd_content)
            tmp_path = Path(tmp_file.name)

        try:
            # Generate specs
            generator = SpecTestGenerator(tmp_path)
            result = generator.generate()

            if output_format == "json":
                import json

                return json.dumps(
                    {
                        "requirements": [
                            {
                                "id": req.id,
                                "title": req.title,
                                "description": req.description,
                                "priority": req.priority.value,
                                "acceptance_criteria": req.acceptance_criteria,
                            }
                            for req in result.requirements
                        ],
                        "test_cases": [
                            {
                                "id": tc.id,
                                "title": tc.title,
                                "requirement_id": tc.requirement_id,
                                "steps": tc.steps,
                                "expected_result": tc.expected_result,
                            }
                            for tc in result.test_cases
                        ],
                    },
                    indent=2,
                )
            elif output_format == "gherkin":
                from spec_test_generator import GherkinGenerator

                gherkin_gen = GherkinGenerator(
                    result.requirements, result.test_cases
                )
                features = gherkin_gen.generate()
                return "\n\n".join(
                    f"# {name}\n{path.read_text()}"
                    for name, path in features.items()
                )
            else:
                # Default markdown output
                output_parts = ["# Generated Requirements\n"]
                for req in result.requirements:
                    output_parts.append(f"## {req.id}: {req.title}\n")
                    output_parts.append(f"{req.description}\n")
                    if req.acceptance_criteria:
                        output_parts.append("**Acceptance Criteria:**\n")
                        for ac in req.acceptance_criteria:
                            output_parts.append(f"- {ac}\n")
                    output_parts.append("\n")

                output_parts.append("# Generated Test Cases\n")
                for tc in result.test_cases:
                    output_parts.append(f"## {tc.id}: {tc.title}\n")
                    output_parts.append(f"**Requirement:** {tc.requirement_id}\n")
                    output_parts.append("**Steps:**\n")
                    for i, step in enumerate(tc.steps, 1):
                        output_parts.append(f"{i}. {step}\n")
                    output_parts.append(f"**Expected:** {tc.expected_result}\n\n")

                return "".join(output_parts)

        finally:
            # Cleanup temporary file
            tmp_path.unlink(missing_ok=True)
