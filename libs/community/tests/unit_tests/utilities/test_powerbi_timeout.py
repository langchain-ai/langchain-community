import asyncio
from typing import Any, Dict

import pytest

from langchain_community.utilities.powerbi import PowerBIDataset


class _DummyResponse:
    """Minimal fake response object for requests.post."""

    def __init__(self, status_code: int = 200, json_data: Dict[str, Any] | None = None):
        self.status_code = status_code
        self._json_data = json_data or {"results": []}

    def json(self) -> Dict[str, Any]:
        return self._json_data


class _DummyAioHTTPResponse:
    """Minimal fake aiohttp response object for async tests."""

    def __init__(self, status: int = 200, json_data: Dict[str, Any] | None = None):
        self.status = status
        self._json_data = json_data or {"results": []}
        self.content_type = "application/json"

    async def json(self, content_type: str | None = None) -> Dict[str, Any]:
        return self._json_data

    async def __aenter__(self) -> "_DummyAioHTTPResponse":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


class _DummyAioHTTPSession:
    """Minimal fake aiohttp ClientSession-like object."""

    def __init__(self) -> None:
        self.last_timeout = None
        self.last_payload: Dict[str, Any] | None = None

    async def post(
        self,
        url: str,
        headers: Dict[str, str],
        json: Dict[str, Any],
        timeout: Any,
    ) -> _DummyAioHTTPResponse:
        # Just record values, then return a dummy response.
        self.last_timeout = timeout
        self.last_payload = json
        return _DummyAioHTTPResponse(status=200)


def _make_dataset(**kwargs: Any) -> PowerBIDataset:
    """Helper to create a valid PowerBIDataset with minimal required fields."""
    base = dict(
        dataset_id="dummy-dataset",
        table_names=["DummyTable"],
        token="dummy-token",  # avoids needing azure credentials
    )
    base.update(kwargs)
    return PowerBIDataset(**base)


def test_run_uses_default_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """run() should use the instance-level request_timeout when no timeout is passed."""
    captured = {}

    def fake_post(url: str, json: Dict[str, Any], headers: Dict[str, str], timeout: float):
        captured["timeout"] = timeout
        return _DummyResponse()

    dataset = _make_dataset(request_timeout=12.5)

    import requests

    monkeypatch.setattr(requests, "post", fake_post)
    dataset.run("EVALUATE ROW(\"X\", 1)")

    assert captured["timeout"] == pytest.approx(12.5)


def test_run_uses_explicit_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """run() should prefer the explicit timeout argument over the default."""
    captured = {}

    def fake_post(url: str, json: Dict[str, Any], headers: Dict[str, str], timeout: float):
        captured["timeout"] = timeout
        return _DummyResponse()

    dataset = _make_dataset(request_timeout=20.0)

    import requests

    monkeypatch.setattr(requests, "post", fake_post)
    dataset.run("EVALUATE ROW(\"X\", 1)", timeout=5.0)

    assert captured["timeout"] == pytest.approx(5.0)


@pytest.mark.parametrize("bad_timeout", [0, -1, -0.5, "abc"])
def test_run_invalid_timeout_raises_value_error(bad_timeout: Any) -> None:
    """Invalid timeout values should raise ValueError."""
    dataset = _make_dataset()
    with pytest.raises(ValueError):
        dataset.run("EVALUATE ROW(\"X\", 1)", timeout=bad_timeout)


@pytest.mark.asyncio
async def test_arun_uses_default_timeout_with_internal_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """arun() should use the instance-level request_timeout when creating a new session."""

    dummy_session = _DummyAioHTTPSession()

    class _DummyClientSessionFactory:
        def __init__(self, session: _DummyAioHTTPSession) -> None:
            self._session = session

        async def __aenter__(self) -> _DummyAioHTTPSession:
            return self._session

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

    # monkeypatch aiohttp.ClientSession to return our dummy wrapper
    import aiohttp

    def fake_client_session(*args: Any, **kwargs: Any) -> _DummyClientSessionFactory:
        return _DummyClientSessionFactory(dummy_session)

    monkeypatch.setattr(aiohttp, "ClientSession", fake_client_session)

    dataset = _make_dataset(request_timeout=7.0)

    await dataset.arun("EVALUATE ROW(\"X\", 1)")

    # aiohttp.ClientTimeout is stored in dummy_session.last_timeout
    assert dummy_session.last_timeout.total == pytest.approx(7.0)


@pytest.mark.asyncio
async def test_arun_uses_explicit_timeout_with_external_session() -> None:
    """arun() should respect an explicit timeout when using a provided aiosession."""
    dummy_session = _DummyAioHTTPSession()
    dataset = _make_dataset(aiosession=dummy_session, request_timeout=30.0)

    await dataset.arun("EVALUATE ROW(\"X\", 1)", timeout=3.5)

    assert dummy_session.last_timeout.total == pytest.approx(3.5)
