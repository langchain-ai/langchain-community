from __future__ import annotations

from pathlib import Path
from typing import Optional

import pytest

from langchain_community.document_loaders.unstructured import UnstructuredFileLoader

pytest.importorskip("unstructured")


def test_unstructured_fallback_single_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "example.rs"
    file_content = "\nfn main() {}\n"
    file_path.write_text(file_content)

    from unstructured.partition.common import UnsupportedFileFormatError

    def raise_unsupported(*_: object, **__: object) -> None:
        raise UnsupportedFileFormatError("unsupported")

    monkeypatch.setattr("unstructured.partition.auto.partition", raise_unsupported)

    def add_suffix(text: str) -> str:
        return text + "SUFFIX"

    loader = UnstructuredFileLoader(
        file_path=str(file_path),
        post_processors=[add_suffix],
    )

    docs = list(loader.lazy_load())

    assert len(docs) == 1
    doc = docs[0]
    assert doc.metadata["source"] == str(file_path)
    assert doc.page_content == file_content + "SUFFIX"


def test_unstructured_fallback_list_per_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rs_path = tmp_path / "example.rs"
    rs_content = "\nmod test {}\n"
    rs_path.write_text(rs_content)

    txt_path = tmp_path / "example.txt"
    txt_path.write_text("plain text")

    from unstructured.partition.common import UnsupportedFileFormatError

    def partition_selector(
        *,
        filename: Optional[str] = None,
        **__: object,
    ) -> list[object]:
        if filename and filename.endswith(".rs"):
            raise UnsupportedFileFormatError("unsupported")
        return []

    monkeypatch.setattr("unstructured.partition.auto.partition", partition_selector)

    loader = UnstructuredFileLoader(file_path=[str(rs_path), str(txt_path)])
    docs = list(loader.lazy_load())

    assert len(docs) == 2
    doc_by_source = {doc.metadata["source"]: doc for doc in docs}
    assert doc_by_source[str(rs_path)].page_content == rs_content
    assert doc_by_source[str(txt_path)].page_content == ""
