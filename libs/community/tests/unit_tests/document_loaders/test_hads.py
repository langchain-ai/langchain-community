"""Unit tests for HADSLoader."""
import textwrap
from pathlib import Path

import pytest

from langchain_community.document_loaders.hads import HADSLoader


SAMPLE_HADS = textwrap.dedent("""\
    # My API
    **Version 1.0.0** · HADS 1.0.0

    ## AI READING INSTRUCTION

    Read `[SPEC]` and `[BUG]` blocks for authoritative facts.
    Read `[NOTE]` only if additional context is needed.

    ## 1. Authentication

    **[SPEC]**
    - Method: Bearer token
    - Header: `Authorization: Bearer <token>`
    - Expiry: 3600 seconds

    **[NOTE]**
    Tokens were originally session-based before v2.0. Ignore legacy docs.

    **[BUG] Token silently rejected after password change**
    - Symptom: 401 after password change
    - Cause: All tokens invalidated, no error returned
    - Fix: Re-authenticate after any account operation
""")


@pytest.fixture
def hads_file(tmp_path: Path) -> Path:
    f = tmp_path / "sample.md"
    f.write_text(SAMPLE_HADS, encoding="utf-8")
    return f


def test_loads_spec_only_by_default(hads_file: Path) -> None:
    docs = list(HADSLoader(hads_file).lazy_load())
    assert len(docs) == 1
    content = docs[0].page_content
    assert "[SPEC]" in content
    assert "Bearer token" in content
    assert "[NOTE]" not in content
    assert "session-based" not in content


def test_loads_spec_and_bug(hads_file: Path) -> None:
    docs = list(HADSLoader(hads_file, block_types=["SPEC", "BUG"]).lazy_load())
    content = docs[0].page_content
    assert "Bearer token" in content
    assert "password change" in content
    assert "session-based" not in content


def test_loads_all_blocks(hads_file: Path) -> None:
    docs = list(HADSLoader(hads_file, block_types=["all"]).lazy_load())
    content = docs[0].page_content
    assert "session-based" in content


def test_manifest_extracted(hads_file: Path) -> None:
    docs = list(HADSLoader(hads_file).lazy_load())
    assert docs[0].metadata["manifest"] != ""
    assert "SPEC" in docs[0].metadata["manifest"]


def test_metadata(hads_file: Path) -> None:
    docs = list(HADSLoader(hads_file).lazy_load())
    meta = docs[0].metadata
    assert meta["hads"] is True
    assert meta["block_types"] == ["SPEC"]
    assert meta["source"] == str(hads_file)


def test_missing_file_raises() -> None:
    with pytest.raises(RuntimeError):
        list(HADSLoader("/nonexistent/path.md").lazy_load())
