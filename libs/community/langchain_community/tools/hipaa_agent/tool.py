"""HIPAA Agent tools for healthcare compliance scanning and monitoring.

Tools that wrap the HIPAA Agent API (https://hipaaagent.ai) for healthcare
compliance scanning, workflow validation, breach probability forecasting,
and vendor risk assessment.

Setup:
    Install the ``hipaa-agent`` package and set your API key:

    .. code-block:: bash

        pip install hipaa-agent

    .. code-block:: python

        import os
        os.environ["HIPAA_AGENT_API_KEY"] = "ha_live_..."

    Or pass ``api_key`` directly to each tool.

Usage:
    .. code-block:: python

        from langchain_community.tools.hipaa_agent import (
            HIPAAAgentComplianceScan,
            HIPAAAgentToolkit,
        )

        # Single tool
        tool = HIPAAAgentComplianceScan(api_key="ha_live_...")
        result = tool.invoke({"npi": "1234567890"})

        # All tools via toolkit
        toolkit = HIPAAAgentToolkit(api_key="ha_live_...")
        tools = toolkit.get_tools()
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Type

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


def _get_client(api_key: str, base_url: str) -> Any:
    """Lazy-import and instantiate the HIPAA Agent client."""
    try:
        from hipaa_agent import HIPAAAgent
    except ImportError as e:
        msg = (
            "Could not import hipaa_agent python package. "
            "Please install it with `pip install hipaa-agent`."
        )
        raise ImportError(msg) from e
    return HIPAAAgent(api_key=api_key, base_url=base_url)


def _format_result(data: Any) -> str:
    """Convert API response to a string for LangChain agent consumption."""
    if isinstance(data, str):
        return data
    try:
        return json.dumps(data, indent=2, default=str)
    except (TypeError, ValueError):
        return str(data)


# ---------------------------------------------------------------------------
# Input Schemas
# ---------------------------------------------------------------------------


class ComplianceScanInput(BaseModel):
    """Input for HIPAA compliance scan."""

    npi: str = Field(
        description=(
            "10-digit National Provider Identifier (NPI) of the healthcare "
            "practice to scan."
        )
    )
    domain: Optional[str] = Field(
        default=None,
        description=(
            "Practice website domain to scan. Auto-discovered from NPI if "
            "omitted."
        ),
    )


class ComplianceScoreInput(BaseModel):
    """Input for compliance score lookup."""

    npi: str = Field(description="10-digit NPI of the healthcare practice.")


class ValidateWorkflowInput(BaseModel):
    """Input for HIPAA workflow validation."""

    npi: str = Field(description="10-digit NPI of the healthcare practice.")
    data_type: str = Field(
        description=(
            "Type of data being transferred. One of: 'phi', 'claims', "
            "'lab_results', 'prescriptions', 'imaging'."
        )
    )
    destination: str = Field(
        description=(
            "Where the data is going. One of: 'clearinghouse', 'ehr_vendor', "
            "'cloud_storage', 'email', 'fax', 'patient_portal', 'hie'."
        )
    )


class BreachProbabilityInput(BaseModel):
    """Input for breach probability calculation."""

    npi: str = Field(description="10-digit NPI of the healthcare practice.")


class BreachLookupInput(BaseModel):
    """Input for HHS breach record lookup."""

    entity_name: Optional[str] = Field(
        default=None,
        description="Name of the healthcare entity to search for.",
    )
    state: Optional[str] = Field(
        default=None,
        description="Two-letter US state code (e.g. 'CA', 'NY').",
    )
    limit: int = Field(
        default=10,
        description="Maximum number of breach records to return (1-50).",
        ge=1,
        le=50,
    )


class VendorRiskInput(BaseModel):
    """Input for vendor risk assessment."""

    vendor_name: Optional[str] = Field(
        default=None,
        description="Name of the vendor / business associate.",
    )
    domain: Optional[str] = Field(
        default=None,
        description="Vendor's website domain.",
    )


class ComplianceStateInput(BaseModel):
    """Input for compliance state machine."""

    npi: str = Field(description="10-digit NPI of the healthcare practice.")


class ComplianceDeltaInput(BaseModel):
    """Input for compliance change detection."""

    npi: str = Field(description="10-digit NPI of the healthcare practice.")
    since: str = Field(
        description="ISO date to compare from (e.g. '2026-01-01')."
    )


class ControlsInput(BaseModel):
    """Input for HIPAA/NIST control signals."""

    npi: str = Field(description="10-digit NPI of the healthcare practice.")


class WebhookSubscribeInput(BaseModel):
    """Input for webhook registration."""

    npi: str = Field(description="10-digit NPI to monitor.")
    url: str = Field(
        description="HTTPS URL to receive webhook POST payloads."
    )
    events: List[str] = Field(
        description=(
            "Event types to subscribe to. Options: 'breach_detected', "
            "'score_dropped', 'baa_expiring', 'scan_completed', "
            "'control_failed', 'sra_expired'."
        )
    )


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


class HIPAAAgentComplianceScan(BaseTool):
    """Launch a 73-tool HIPAA infrastructure scan for a healthcare practice.

    Dispatches a comprehensive external scan that checks SSL/TLS, email
    authentication (SPF/DKIM/DMARC), open ports, privacy policies, breach
    exposure, and 60+ additional security controls. Returns the HIPAA Agent
    Compliance Score™ (letter grade A-F), numeric score, and finding count.

    Setup:
        Install the ``hipaa-agent`` package:

        .. code-block:: bash

            pip install hipaa-agent

    Usage:
        .. code-block:: python

            from langchain_community.tools.hipaa_agent import (
                HIPAAAgentComplianceScan,
            )

            tool = HIPAAAgentComplianceScan(api_key="ha_live_...")
            result = tool.invoke({"npi": "1234567890"})
    """

    name: str = "hipaa_compliance_scan"
    description: str = (
        "Launch a comprehensive HIPAA compliance scan for a healthcare "
        "practice. Input requires a 10-digit NPI (National Provider "
        "Identifier). Optionally accepts a website domain. Returns "
        "compliance grade (A-F), numeric score, finding count, and scan ID."
    )
    args_schema: Type[BaseModel] = ComplianceScanInput
    api_key: str = Field(exclude=True)
    base_url: str = Field(default="https://hipaaagent.ai", exclude=True)

    def _run(
        self,
        npi: str,
        domain: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Scan a healthcare practice for HIPAA compliance."""
        try:
            client = _get_client(self.api_key, self.base_url)
            result = client.scan(npi=npi, domain=domain)
            return _format_result(result)
        except Exception as e:
            return f"Error scanning practice: {e}"

    async def _arun(
        self,
        npi: str,
        domain: Optional[str] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Async not yet supported."""
        return self._run(npi=npi, domain=domain)


class HIPAAAgentComplianceScore(BaseTool):
    """Get the HIPAA Agent Compliance Score™ for a healthcare practice.

    Returns the letter grade (A-F), numeric score (0-100), and per-category
    breakdown across 10 compliance categories.

    Setup:
        .. code-block:: bash

            pip install hipaa-agent

    Usage:
        .. code-block:: python

            from langchain_community.tools.hipaa_agent import (
                HIPAAAgentComplianceScore,
            )

            tool = HIPAAAgentComplianceScore(api_key="ha_live_...")
            result = tool.invoke({"npi": "1234567890"})
    """

    name: str = "hipaa_compliance_score"
    description: str = (
        "Get the HIPAA Agent Compliance Score for a practice by NPI. "
        "Returns letter grade (A-F), numeric score (0-100), and breakdown "
        "across 10 security categories."
    )
    args_schema: Type[BaseModel] = ComplianceScoreInput
    api_key: str = Field(exclude=True)
    base_url: str = Field(default="https://hipaaagent.ai", exclude=True)

    def _run(
        self,
        npi: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Get compliance score for a practice."""
        try:
            client = _get_client(self.api_key, self.base_url)
            result = client.get_compliance_score(npi=npi)
            return _format_result(result)
        except Exception as e:
            return f"Error getting compliance score: {e}"

    async def _arun(
        self,
        npi: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Async not yet supported."""
        return self._run(npi=npi)


class HIPAAAgentValidateWorkflow(BaseTool):
    """Validate whether a data transfer workflow is HIPAA-compliant.

    Synchronous guardrail that checks 5 data types (PHI, claims, lab results,
    prescriptions, imaging) against 7 destination types. Returns whether the
    workflow is allowed, the risk level, required actions, and HIPAA citations.

    Setup:
        .. code-block:: bash

            pip install hipaa-agent

    Usage:
        .. code-block:: python

            from langchain_community.tools.hipaa_agent import (
                HIPAAAgentValidateWorkflow,
            )

            tool = HIPAAAgentValidateWorkflow(api_key="ha_live_...")
            result = tool.invoke({
                "npi": "1234567890",
                "data_type": "phi",
                "destination": "email",
            })
    """

    name: str = "hipaa_validate_workflow"
    description: str = (
        "Validate whether a healthcare data transfer workflow is "
        "HIPAA-compliant. Input requires NPI, data_type, and destination. "
        "Returns allowed/denied, risk level, required controls, and HIPAA "
        "citations."
    )
    args_schema: Type[BaseModel] = ValidateWorkflowInput
    api_key: str = Field(exclude=True)
    base_url: str = Field(default="https://hipaaagent.ai", exclude=True)

    def _run(
        self,
        npi: str,
        data_type: str,
        destination: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Validate a data transfer workflow."""
        try:
            client = _get_client(self.api_key, self.base_url)
            result = client.validate_workflow(
                workflow_type=f"{data_type}_to_{destination}",
                npi=npi,
            )
            return _format_result(result)
        except Exception as e:
            return f"Error validating workflow: {e}"

    async def _arun(
        self,
        npi: str,
        data_type: str,
        destination: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Async not yet supported."""
        return self._run(npi=npi, data_type=data_type, destination=destination)


class HIPAAAgentBreachProbability(BaseTool):
    """Calculate the 12-month breach probability for a healthcare practice.

    Uses HHS base rates for 15 medical specialties, adjusted by compliance
    grade, control gap penalties, and prior breach history.

    Setup:
        .. code-block:: bash

            pip install hipaa-agent

    Usage:
        .. code-block:: python

            from langchain_community.tools.hipaa_agent import (
                HIPAAAgentBreachProbability,
            )

            tool = HIPAAAgentBreachProbability(api_key="ha_live_...")
            result = tool.invoke({"npi": "1234567890"})
    """

    name: str = "hipaa_breach_probability"
    description: str = (
        "Calculate the 12-month data breach probability for a healthcare "
        "practice. Returns probability percentage, risk tier, confidence "
        "score, and contributing factors."
    )
    args_schema: Type[BaseModel] = BreachProbabilityInput
    api_key: str = Field(exclude=True)
    base_url: str = Field(default="https://hipaaagent.ai", exclude=True)

    def _run(
        self,
        npi: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Calculate breach probability for a practice."""
        try:
            client = _get_client(self.api_key, self.base_url)
            result = client.get_breach_probability(npi=npi)
            return _format_result(result)
        except Exception as e:
            return f"Error calculating breach probability: {e}"

    async def _arun(
        self,
        npi: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Async not yet supported."""
        return self._run(npi=npi)


class HIPAAAgentBreachLookup(BaseTool):
    """Search the HHS breach database for healthcare data breach records.

    Queries 850+ breaches affecting 384M+ individuals reported to the HHS
    Office for Civil Rights.

    Setup:
        .. code-block:: bash

            pip install hipaa-agent

    Usage:
        .. code-block:: python

            from langchain_community.tools.hipaa_agent import (
                HIPAAAgentBreachLookup,
            )

            tool = HIPAAAgentBreachLookup(api_key="ha_live_...")
            result = tool.invoke({"entity_name": "Example Health"})
    """

    name: str = "hipaa_breach_lookup"
    description: str = (
        "Search the HHS HIPAA breach database for healthcare data breach "
        "records. Can filter by entity name and/or US state code."
    )
    args_schema: Type[BaseModel] = BreachLookupInput
    api_key: str = Field(exclude=True)
    base_url: str = Field(default="https://hipaaagent.ai", exclude=True)

    def _run(
        self,
        entity_name: Optional[str] = None,
        state: Optional[str] = None,
        limit: int = 10,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Search the HHS breach database."""
        try:
            client = _get_client(self.api_key, self.base_url)
            result = client.get_breach(
                entity_name=entity_name,
                state=state,
                limit=limit,
            )
            return _format_result(result)
        except Exception as e:
            return f"Error looking up breaches: {e}"

    async def _arun(
        self,
        entity_name: Optional[str] = None,
        state: Optional[str] = None,
        limit: int = 10,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Async not yet supported."""
        return self._run(
            entity_name=entity_name, state=state, limit=limit
        )


class HIPAAAgentVendorRisk(BaseTool):
    """Check a vendor's risk profile for HIPAA compliance.

    Queries breach history and BAA coverage across practices served by the
    vendor.

    Setup:
        .. code-block:: bash

            pip install hipaa-agent

    Usage:
        .. code-block:: python

            from langchain_community.tools.hipaa_agent import (
                HIPAAAgentVendorRisk,
            )

            tool = HIPAAAgentVendorRisk(api_key="ha_live_...")
            result = tool.invoke({
                "vendor_name": "Cloud EHR Inc",
                "domain": "cloudehr.com",
            })
    """

    name: str = "hipaa_vendor_risk"
    description: str = (
        "Check a vendor or business associate's HIPAA risk profile. "
        "Returns risk rating, security score, breach history, and BAA "
        "coverage."
    )
    args_schema: Type[BaseModel] = VendorRiskInput
    api_key: str = Field(exclude=True)
    base_url: str = Field(default="https://hipaaagent.ai", exclude=True)

    def _run(
        self,
        vendor_name: Optional[str] = None,
        domain: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Check a vendor's HIPAA risk profile."""
        try:
            client = _get_client(self.api_key, self.base_url)
            result = client.check_vendor(
                vendor_name=vendor_name or "",
                vendor_domain=domain,
            )
            return _format_result(result)
        except Exception as e:
            return f"Error checking vendor risk: {e}"

    async def _arun(
        self,
        vendor_name: Optional[str] = None,
        domain: Optional[str] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Async not yet supported."""
        return self._run(vendor_name=vendor_name, domain=domain)


class HIPAAAgentComplianceState(BaseTool):
    """Track practice readiness against the May 2026 HIPAA Security Rule deadline.

    Evaluates 13 requirements and returns overall state, completed count,
    days remaining, and next action.

    Setup:
        .. code-block:: bash

            pip install hipaa-agent

    Usage:
        .. code-block:: python

            from langchain_community.tools.hipaa_agent import (
                HIPAAAgentComplianceState,
            )

            tool = HIPAAAgentComplianceState(api_key="ha_live_...")
            result = tool.invoke({"npi": "1234567890"})
    """

    name: str = "hipaa_compliance_state"
    description: str = (
        "Get the HIPAA compliance readiness state for a practice. "
        "Tracks 13 requirements against the May 2026 HIPAA Security Rule "
        "deadline. Returns state, completed controls, and days remaining."
    )
    args_schema: Type[BaseModel] = ComplianceStateInput
    api_key: str = Field(exclude=True)
    base_url: str = Field(default="https://hipaaagent.ai", exclude=True)

    def _run(
        self,
        npi: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Get compliance readiness state."""
        try:
            client = _get_client(self.api_key, self.base_url)
            result = client.get_compliance_state(npi=npi)
            return _format_result(result)
        except Exception as e:
            return f"Error getting compliance state: {e}"

    async def _arun(
        self,
        npi: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Async not yet supported."""
        return self._run(npi=npi)


class HIPAAAgentComplianceDelta(BaseTool):
    """Detect compliance changes since a given date.

    Compares current controls against historical scan data. Shows which
    controls improved or regressed.

    Setup:
        .. code-block:: bash

            pip install hipaa-agent

    Usage:
        .. code-block:: python

            from langchain_community.tools.hipaa_agent import (
                HIPAAAgentComplianceDelta,
            )

            tool = HIPAAAgentComplianceDelta(api_key="ha_live_...")
            result = tool.invoke({
                "npi": "1234567890",
                "since": "2026-01-01",
            })
    """

    name: str = "hipaa_compliance_delta"
    description: str = (
        "Compare current HIPAA compliance controls against a historical "
        "point in time. Returns changed controls with before/after status."
    )
    args_schema: Type[BaseModel] = ComplianceDeltaInput
    api_key: str = Field(exclude=True)
    base_url: str = Field(default="https://hipaaagent.ai", exclude=True)

    def _run(
        self,
        npi: str,
        since: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Get compliance changes since a date."""
        try:
            client = _get_client(self.api_key, self.base_url)
            result = client.get_compliance_delta(npi=npi, since=since)
            return _format_result(result)
        except Exception as e:
            return f"Error getting compliance delta: {e}"

    async def _arun(
        self,
        npi: str,
        since: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Async not yet supported."""
        return self._run(npi=npi, since=since)


class HIPAAAgentControls(BaseTool):
    """Get 13 HIPAA/NIST control-level signals for a practice.

    Maps scan findings to standardized controls. Each control returns
    pass/fail/partial with risk scores and remediation actions.

    Setup:
        .. code-block:: bash

            pip install hipaa-agent

    Usage:
        .. code-block:: python

            from langchain_community.tools.hipaa_agent import (
                HIPAAAgentControls,
            )

            tool = HIPAAAgentControls(api_key="ha_live_...")
            result = tool.invoke({"npi": "1234567890"})
    """

    name: str = "hipaa_controls"
    description: str = (
        "Get HIPAA/NIST security control assessment for a practice. "
        "Evaluates 13 controls with pass/fail/partial status and risk "
        "scores."
    )
    args_schema: Type[BaseModel] = ControlsInput
    api_key: str = Field(exclude=True)
    base_url: str = Field(default="https://hipaaagent.ai", exclude=True)

    def _run(
        self,
        npi: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Get HIPAA/NIST control signals."""
        try:
            client = _get_client(self.api_key, self.base_url)
            result = client.get_controls(npi=npi)
            return _format_result(result)
        except Exception as e:
            return f"Error getting controls: {e}"

    async def _arun(
        self,
        npi: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Async not yet supported."""
        return self._run(npi=npi)


class HIPAAAgentWebhookSubscribe(BaseTool):
    """Register a webhook for HIPAA compliance event notifications.

    Subscribes an HTTPS URL to receive POST notifications for 6 event types.
    Payloads are signed with HMAC-SHA256.

    Setup:
        .. code-block:: bash

            pip install hipaa-agent

    Usage:
        .. code-block:: python

            from langchain_community.tools.hipaa_agent import (
                HIPAAAgentWebhookSubscribe,
            )

            tool = HIPAAAgentWebhookSubscribe(api_key="ha_live_...")
            result = tool.invoke({
                "npi": "1234567890",
                "url": "https://example.com/webhook",
                "events": ["scan_completed", "breach_detected"],
            })
    """

    name: str = "hipaa_webhook_subscribe"
    description: str = (
        "Register a webhook URL to receive real-time HIPAA compliance "
        "notifications. Returns webhook ID and HMAC secret."
    )
    args_schema: Type[BaseModel] = WebhookSubscribeInput
    api_key: str = Field(exclude=True)
    base_url: str = Field(default="https://hipaaagent.ai", exclude=True)

    def _run(
        self,
        npi: str,
        url: str,
        events: List[str],
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Register a compliance webhook."""
        try:
            client = _get_client(self.api_key, self.base_url)
            result = client.subscribe_webhook(url=url, events=events)
            return _format_result(result)
        except Exception as e:
            return f"Error subscribing webhook: {e}"

    async def _arun(
        self,
        npi: str,
        url: str,
        events: List[str],
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Async not yet supported."""
        return self._run(npi=npi, url=url, events=events)


# ---------------------------------------------------------------------------
# Toolkit
# ---------------------------------------------------------------------------

_TOOL_CLASSES = [
    HIPAAAgentComplianceScan,
    HIPAAAgentComplianceScore,
    HIPAAAgentValidateWorkflow,
    HIPAAAgentBreachProbability,
    HIPAAAgentBreachLookup,
    HIPAAAgentVendorRisk,
    HIPAAAgentComplianceState,
    HIPAAAgentComplianceDelta,
    HIPAAAgentControls,
    HIPAAAgentWebhookSubscribe,
]


class HIPAAAgentToolkit:
    """Toolkit that bundles all HIPAA Agent tools.

    Usage:
        .. code-block:: python

            from langchain_community.tools.hipaa_agent import (
                HIPAAAgentToolkit,
            )

            toolkit = HIPAAAgentToolkit(api_key="ha_live_...")
            tools = toolkit.get_tools()

    Args:
        api_key: HIPAA Agent API key (``ha_live_*`` or ``ha_test_*``).
        base_url: API base URL. Defaults to ``https://hipaaagent.ai``.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://hipaaagent.ai",
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url

    def get_tools(self) -> List[BaseTool]:
        """Return all 10 HIPAA Agent tools as a list of BaseTool instances."""
        kwargs: Dict[str, str] = {
            "api_key": self.api_key,
            "base_url": self.base_url,
        }
        return [cls(**kwargs) for cls in _TOOL_CLASSES]
