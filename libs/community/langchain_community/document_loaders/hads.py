"""HADS (Human-AI Document Standard) document loader."""
import re
from pathlib import Path
from typing import Iterator, List, Literal, Optional, Union

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader

BLOCK_PATTERN = re.compile(
    r"\*\*\[(SPEC|NOTE|BUG[^\]]*|\?)\]\*\*\n((?:(?!\*\*\[).*\n?)*)",
    re.MULTILINE,
)
MANIFEST_PATTERN = re.compile(
    r"##\s+AI READING INSTRUCTION\s*\n(.*?)(?=\n##|\Z)",
    re.DOTALL,
)

BlockType = Literal["SPEC", "NOTE", "BUG", "?", "all"]


class HADSLoader(BaseLoader):
    """Load HADS (Human-AI Document Standard) Markdown files.

    HADS is a lightweight convention for writing technical documentation
    that works equally well for humans and AI language models. Documents
    contain semantic block tags ([SPEC], [NOTE], [BUG], [?]) and an AI
    manifest that describes what to read for a given task.

    This loader can extract only the blocks relevant to a query — reducing
    token consumption by ~70% compared to loading the full document.

    Spec: https://github.com/catcam/hads

    Setup:
        Install ``langchain-community``.

        .. code-block:: bash

            pip install -U langchain-community

    Instantiate:
        .. code-block:: python

            from langchain_community.document_loaders import HADSLoader

            # Load only [SPEC] blocks (default — smallest context)
            loader = HADSLoader("./architecture.md")

            # Load [SPEC] and [BUG] blocks (for debugging tasks)
            loader = HADSLoader("./architecture.md", block_types=["SPEC", "BUG"])

            # Load all blocks (equivalent to a plain text loader)
            loader = HADSLoader("./architecture.md", block_types=["all"])

    Lazy load:
        .. code-block:: python

            docs = list(loader.lazy_load())
            print(docs[0].page_content)
            print(docs[0].metadata)
            # {'source': './architecture.md', 'hads': True,
            #  'block_types': ['SPEC'], 'manifest': '...', 'blocks_found': 12}

    Args:
        file_path: Path to the HADS Markdown file.
        block_types: Block types to extract. One or more of "SPEC", "NOTE",
            "BUG", "?", or "all". Defaults to ["SPEC"].
        encoding: File encoding. Defaults to "utf-8".
        include_section_headers: If True, prepend the nearest H2/H3 heading
            to each extracted block for context. Defaults to True.
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        block_types: Optional[List[BlockType]] = None,
        encoding: str = "utf-8",
        include_section_headers: bool = True,
    ):
        self.file_path = Path(file_path)
        self.block_types = block_types or ["SPEC"]
        self.encoding = encoding
        self.include_section_headers = include_section_headers

    def lazy_load(self) -> Iterator[Document]:
        """Load and filter HADS document blocks."""
        try:
            text = self.file_path.read_text(encoding=self.encoding)
        except Exception as e:
            raise RuntimeError(f"Error loading {self.file_path}") from e

        # Extract manifest
        manifest = ""
        manifest_match = MANIFEST_PATTERN.search(text)
        if manifest_match:
            manifest = manifest_match.group(1).strip()

        load_all = "all" in self.block_types

        if load_all:
            content = text
            blocks_found = len(BLOCK_PATTERN.findall(text))
        else:
            content = self._extract_blocks(text)
            blocks_found = len(BLOCK_PATTERN.findall(text))

        metadata = {
            "source": str(self.file_path),
            "hads": True,
            "block_types": self.block_types,
            "blocks_found": blocks_found,
        }
        if manifest:
            metadata["manifest"] = manifest

        yield Document(page_content=content.strip(), metadata=metadata)

    def _extract_blocks(self, text: str) -> str:
        """Extract only the requested block types from the document."""
        lines = text.split("\n")
        result: List[str] = []
        current_header = ""
        in_block = False
        block_tag = ""
        block_lines: List[str] = []

        for line in lines:
            # Track section headers for context
            if self.include_section_headers and re.match(r"^#{1,3} ", line):
                current_header = line
                in_block = False
                block_lines = []
                continue

            # Detect block start
            tag_match = re.match(r"\*\*\[([^\]]+)\]\*\*", line)
            if tag_match:
                # Flush previous block
                if in_block and self._should_include(block_tag):
                    if current_header:
                        result.append(current_header)
                    result.extend(block_lines)
                    result.append("")

                block_tag = tag_match.group(1).split()[0]  # "BUG" from "BUG title"
                in_block = True
                block_lines = [line]
                continue

            if in_block:
                # Empty line or new H2 ends the block
                if line.startswith("## ") or line.startswith("# "):
                    if self._should_include(block_tag):
                        if current_header:
                            result.append(current_header)
                        result.extend(block_lines)
                        result.append("")
                    in_block = False
                    block_lines = []
                    current_header = line
                else:
                    block_lines.append(line)

        # Flush final block
        if in_block and self._should_include(block_tag):
            if current_header:
                result.append(current_header)
            result.extend(block_lines)

        return "\n".join(result)

    def _should_include(self, tag: str) -> bool:
        if "all" in self.block_types:
            return True
        tag_upper = tag.upper()
        for requested in self.block_types:
            if requested == "?" and tag == "?":
                return True
            if requested != "?" and tag_upper.startswith(requested.upper()):
                return True
        return False
