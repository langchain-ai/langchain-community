"""Agent Toolbox tools.

13 production-ready tools for AI agents:
search, extract, screenshot, weather, finance, validate_email,
translate, geoip, news, whois, dns, pdf_extract, qr.

Get a free API key at https://api.sendtoclaw.com/v1/auth/register
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.utilities.agent_toolbox import AgentToolboxAPIWrapper


class AgentToolboxSearch(BaseTool):
    """Search the web using Agent Toolbox (DuckDuckGo backend)."""

    name: str = "agent_toolbox_search"
    description: str = (
        "Search the web and get titles, URLs, and snippets. "
        "Useful for finding up-to-date information on any topic."
    )
    api_wrapper: AgentToolboxAPIWrapper = Field(
        default_factory=AgentToolboxAPIWrapper
    )

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        return self.api_wrapper.run("search", {"query": query})


class AgentToolboxExtract(BaseTool):
    """Extract readable content from a web page."""

    name: str = "agent_toolbox_extract"
    description: str = (
        "Extract the main readable content from a web page URL. "
        "Input should be a URL."
    )
    api_wrapper: AgentToolboxAPIWrapper = Field(
        default_factory=AgentToolboxAPIWrapper
    )

    def _run(self, url: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        return self.api_wrapper.run("extract", {"url": url, "format": "markdown"})


class AgentToolboxScreenshot(BaseTool):
    """Take a screenshot of a web page."""

    name: str = "agent_toolbox_screenshot"
    description: str = "Capture a screenshot of a web page. Input should be a URL."
    api_wrapper: AgentToolboxAPIWrapper = Field(
        default_factory=AgentToolboxAPIWrapper
    )

    def _run(self, url: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        return self.api_wrapper.run("screenshot", {"url": url})


class AgentToolboxWeather(BaseTool):
    """Get weather and forecast for a location."""

    name: str = "agent_toolbox_weather"
    description: str = (
        "Get current weather conditions and forecast for any location. "
        "Input should be a city name or location."
    )
    api_wrapper: AgentToolboxAPIWrapper = Field(
        default_factory=AgentToolboxAPIWrapper
    )

    def _run(self, location: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        return self.api_wrapper.run("weather", {"location": location})


class AgentToolboxFinance(BaseTool):
    """Get stock quotes or currency exchange rates."""

    name: str = "agent_toolbox_finance"
    description: str = (
        "Get real-time stock quotes or currency exchange rates. "
        "Input should be a stock symbol (e.g. AAPL) or 'USD to EUR'."
    )
    api_wrapper: AgentToolboxAPIWrapper = Field(
        default_factory=AgentToolboxAPIWrapper
    )

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        return self.api_wrapper.run("finance", {"symbol": query, "type": "quote"})


class AgentToolboxEmailValidator(BaseTool):
    """Validate an email address."""

    name: str = "agent_toolbox_email_validator"
    description: str = (
        "Validate an email address by checking syntax, MX records, "
        "SMTP reachability, and disposable domain detection."
    )
    api_wrapper: AgentToolboxAPIWrapper = Field(
        default_factory=AgentToolboxAPIWrapper
    )

    def _run(self, email: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        return self.api_wrapper.run("validate-email", {"email": email})


class AgentToolboxTranslate(BaseTool):
    """Translate text between languages."""

    name: str = "agent_toolbox_translate"
    description: str = (
        "Translate text between 100+ languages. "
        "Input should be 'text|target_language' (e.g. 'Hello|zh')."
    )
    api_wrapper: AgentToolboxAPIWrapper = Field(
        default_factory=AgentToolboxAPIWrapper
    )

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        parts = query.split("|", 1)
        text = parts[0].strip()
        target = parts[1].strip() if len(parts) > 1 else "en"
        return self.api_wrapper.run("translate", {"text": text, "target": target})


class AgentToolboxGeoIP(BaseTool):
    """Look up geolocation for an IP address."""

    name: str = "agent_toolbox_geoip"
    description: str = "Get geolocation data for an IP address. Input should be an IP address."
    api_wrapper: AgentToolboxAPIWrapper = Field(
        default_factory=AgentToolboxAPIWrapper
    )

    def _run(self, ip: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        return self.api_wrapper.run("geoip", {"ip": ip})


class AgentToolboxNews(BaseTool):
    """Search for recent news articles."""

    name: str = "agent_toolbox_news"
    description: str = (
        "Search for recent news articles on any topic. "
        "Input should be a search query."
    )
    api_wrapper: AgentToolboxAPIWrapper = Field(
        default_factory=AgentToolboxAPIWrapper
    )

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        return self.api_wrapper.run("news", {"query": query})


class AgentToolboxWhois(BaseTool):
    """Look up WHOIS information for a domain."""

    name: str = "agent_toolbox_whois"
    description: str = "Get WHOIS registration data for a domain. Input should be a domain name."
    api_wrapper: AgentToolboxAPIWrapper = Field(
        default_factory=AgentToolboxAPIWrapper
    )

    def _run(self, domain: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        return self.api_wrapper.run("whois", {"domain": domain})


class AgentToolboxDns(BaseTool):
    """Query DNS records for a domain."""

    name: str = "agent_toolbox_dns"
    description: str = (
        "Query DNS records for a domain. "
        "Input should be 'domain' or 'domain|record_type' (e.g. 'google.com|MX')."
    )
    api_wrapper: AgentToolboxAPIWrapper = Field(
        default_factory=AgentToolboxAPIWrapper
    )

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        parts = query.split("|", 1)
        domain = parts[0].strip()
        rtype = parts[1].strip() if len(parts) > 1 else "A"
        return self.api_wrapper.run("dns", {"domain": domain, "type": rtype})


class AgentToolboxPdfExtract(BaseTool):
    """Extract text from a PDF file."""

    name: str = "agent_toolbox_pdf_extract"
    description: str = "Extract text content from a PDF file URL. Input should be a URL."
    api_wrapper: AgentToolboxAPIWrapper = Field(
        default_factory=AgentToolboxAPIWrapper
    )

    def _run(self, url: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        return self.api_wrapper.run("pdf-extract", {"url": url})


class AgentToolboxQr(BaseTool):
    """Generate a QR code."""

    name: str = "agent_toolbox_qr"
    description: str = "Generate a QR code image from text or URL. Input should be the text to encode."
    api_wrapper: AgentToolboxAPIWrapper = Field(
        default_factory=AgentToolboxAPIWrapper
    )

    def _run(self, text: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        return self.api_wrapper.run("qr", {"text": text})


class AgentToolboxRun(BaseTool):
    """Generic Agent Toolbox tool — call any endpoint."""

    name: str = "agent_toolbox"
    description: str = (
        "Run any Agent Toolbox API endpoint. "
        "Input should be 'endpoint|json_payload' "
        "(e.g. 'search|{\"query\": \"AI agents\"}'). "
    )
    api_wrapper: AgentToolboxAPIWrapper = Field(
        default_factory=AgentToolboxAPIWrapper
    )

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        parts = query.split("|", 1)
        endpoint = parts[0].strip()
        payload = json.loads(parts[1]) if len(parts) > 1 else {}
        return self.api_wrapper.run(endpoint, payload)
