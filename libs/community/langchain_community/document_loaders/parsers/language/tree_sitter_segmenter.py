from abc import abstractmethod
from typing import TYPE_CHECKING, Any, List, Tuple

from langchain_community.document_loaders.parsers.language.code_segmenter import (
    CodeSegmenter,
)

if TYPE_CHECKING:
    from tree_sitter import Language, Node, Parser


def _collect_ordered_captures(query_captures: Any) -> List[Tuple[Any, str]]:
    """Normalize tree-sitter captures to a flat list in source order.

    ``tree-sitter`` 0.23+ returns captures grouped by capture name as a
    dict; older releases returned a flat ``(node, capture_name)`` sequence
    in document order. Sorting by ``(start_byte, -end_byte)`` ensures
    containing nodes come before their nested children, which the
    line-based dedup in segmenters depends on.
    """
    if isinstance(query_captures, dict):
        flat_captures = [
            (node, capture_name)
            for capture_name, nodes in query_captures.items()
            for node in nodes
        ]
        return sorted(
            flat_captures,
            key=lambda pair: (pair[0].start_byte, -pair[0].end_byte),
        )

    return list(query_captures)


class TreeSitterSegmenter(CodeSegmenter):
    """Abstract class for `CodeSegmenter`s that use the tree-sitter library."""

    def __init__(self, code: str):
        super().__init__(code)
        self.source_lines = self.code.splitlines()

        try:
            import tree_sitter  # noqa: F401
            import tree_sitter_language_pack  # noqa: F401
        except ImportError:
            raise ImportError(
                "Could not import tree_sitter/tree_sitter_language_pack Python "
                "packages. Please install them with "
                "`pip install tree-sitter tree-sitter-language-pack`."
            )

    def is_valid(self) -> bool:
        from tree_sitter import Query, QueryCursor

        language = self.get_language()
        error_query = Query(language, "(ERROR) @error")
        query_cursor = QueryCursor(error_query)

        parser = self.get_parser()
        tree = parser.parse(bytes(self.code, encoding="UTF-8"))

        captures = query_cursor.captures(tree.root_node)
        return len(captures) == 0

    def extract_functions_classes(self) -> List[str]:
        from tree_sitter import Query, QueryCursor

        language = self.get_language()
        query = Query(language, self.get_chunk_query())
        query_cursor = QueryCursor(query)

        parser = self.get_parser()
        tree = parser.parse(bytes(self.code, encoding="UTF-8"))
        captures = _collect_ordered_captures(query.captures(tree.root_node))

        processed_lines: set[int] = set()
        chunks: List[str] = []

        for capture_name, nodes in query_captures.items():
            for node in nodes:
                start_line = node.start_point[0]
                end_line = node.end_point[0]
                lines = range(start_line, end_line + 1)

                if any(line in processed_lines for line in lines):
                    continue

                processed_lines.update(lines)
                chunk_text = node.text.decode("UTF-8")
                chunks.append(chunk_text)

        return chunks

    def simplify_code(self) -> str:
        from tree_sitter import Query, QueryCursor

        language = self.get_language()
        query = Query(language, self.get_chunk_query())
        query_cursor = QueryCursor(query)

        parser = self.get_parser()
        tree = parser.parse(bytes(self.code, encoding="UTF-8"))
        processed_lines = set()

        simplified_lines = self.source_lines[:]

        captures = _collect_ordered_captures(query.captures(tree.root_node))
        for node, name in captures:
            start_line = node.start_point[0]
            end_line = node.end_point[0]

            lines = list(range(start_line, end_line + 1))
            if any(line in processed_lines for line in lines):
                continue

            simplified_lines[start_line] = self.make_line_comment(
                f"Code for: {self.source_lines[start_line]}"
            )
            for line_num in range(start_line + 1, end_line + 1):
                simplified_lines[line_num] = None  # type: ignore[call-overload]

            processed_lines.update(lines)

        return "\n".join(line for line in simplified_lines if line is not None)

    def get_parser(self) -> "Parser":
        from tree_sitter import Parser

        parser = Parser(self.get_language())
        return parser

    @abstractmethod
    def get_language(self) -> "Language":
        raise NotImplementedError()  # pragma: no cover

    @abstractmethod
    def get_chunk_query(self) -> str:
        raise NotImplementedError()  # pragma: no cover

    @abstractmethod
    def make_line_comment(self, text: str) -> str:
        raise NotImplementedError()  # pragma: no cover
