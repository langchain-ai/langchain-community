from __future__ import annotations

from typing import Any

from langchain_community.document_loaders.parsers.language import (
    tree_sitter_segmenter as _ts,
)

TreeSitterSegmenter = _ts.TreeSitterSegmenter
_collect_ordered_captures = _ts._collect_ordered_captures


class _FakeNode:
    def __init__(
        self,
        text: str,
        start_line: int,
        end_line: int,
        start_byte: int,
        end_byte: int,
    ) -> None:
        self.text = text.encode("utf-8")
        self.start_point = (start_line, 0)
        self.end_point = (end_line, 0)
        self.start_byte = start_byte
        self.end_byte = end_byte


class _FakeQuery:
    def __init__(self, captures: dict[str, list[_FakeNode]]) -> None:
        self._captures = captures

    def captures(self, _root_node: object) -> dict[str, list[_FakeNode]]:
        return self._captures


class _FakeLanguage:
    def __init__(self, captures: dict[str, list[_FakeNode]]) -> None:
        self._captures = captures

    def query(self, _query: str) -> _FakeQuery:
        return _FakeQuery(self._captures)


class _FakeTree:
    root_node = object()


class _FakeParser:
    def parse(self, _code: bytes) -> _FakeTree:
        return _FakeTree()


class _TestTreeSitterSegmenter(TreeSitterSegmenter):
    def __init__(self, code: str, captures: dict[str, list[_FakeNode]]) -> None:
        self.code = code
        self.source_lines = code.splitlines()
        self._captures = captures

    def get_language(self) -> Any:
        return _FakeLanguage(self._captures)

    def get_parser(self) -> _FakeParser:
        return _FakeParser()

    def get_chunk_query(self) -> str:
        return "(ignored)"

    def make_line_comment(self, text: str) -> str:
        return f"// {text}"


def test_extract_functions_classes_preserves_return_type_and_source_order() -> None:
    code = """class T {
  void baz(U) {
  }
};

auto T::bar() const -> int {
  return 1;
}"""
    class_node = _FakeNode("class T {\n  void baz(U) {\n  }\n}", 0, 2, 0, 31)
    method_node = _FakeNode("void baz(U) {\n  }", 1, 2, 10, 25)
    function_node = _FakeNode(
        "auto T::bar() const -> int {\n  return 1;\n}", 5, 7, 36, 81
    )
    captures = {
        "function": [method_node, function_node],
        "class": [class_node],
    }

    segmenter = _TestTreeSitterSegmenter(code, captures)

    extracted_code = segmenter.extract_functions_classes()

    assert isinstance(extracted_code, list)
    assert extracted_code == [
        "class T {\n  void baz(U) {\n  }\n}",
        "auto T::bar() const -> int {\n  return 1;\n}",
    ]


def test_collect_ordered_captures_returns_list_sorted_by_source_position() -> None:
    """_collect_ordered_captures must return a list in source order.

    tree-sitter 0.23+ returns captures grouped by capture name (dict) rather
    than a flat sequence.  This test verifies that the helper normalizes both
    shapes and that the result is always a list sorted by start_byte so
    containers appear before nested children.
    """
    container = _FakeNode("container", 0, 3, 0, 40)
    nested = _FakeNode("nested", 1, 2, 10, 25)
    separate = _FakeNode("separate", 5, 6, 50, 70)

    # --- new dict-shaped API (tree-sitter 0.23+) ---
    dict_captures: Any = {
        "inner": [nested, separate],
        "outer": [container],
    }
    result = _collect_ordered_captures(dict_captures)

    assert isinstance(result, list)
    nodes = [node for node, _name in result]
    assert nodes == [container, nested, separate], (
        "Expected source order: container (0), nested (10), separate (50)"
    )

    # --- legacy list-of-tuples API (tree-sitter <0.23) ---
    legacy_captures: Any = [
        (separate, "inner"),
        (container, "outer"),
        (nested, "inner"),
    ]
    result_legacy = _collect_ordered_captures(legacy_captures)

    assert isinstance(result_legacy, list)
    assert result_legacy == legacy_captures, (
        "Legacy sequence must be returned as-is (original order preserved)"
    )
