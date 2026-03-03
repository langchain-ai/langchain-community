from pathlib import Path
from typing import Any, Iterator, List

import pytest
from langchain_core.documents import Document

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders.text import TextLoader


def test_raise_error_if_path_not_exist() -> None:
    loader = DirectoryLoader("./not_exist_directory")
    with pytest.raises(FileNotFoundError) as e:
        loader.load()

    assert str(e.value) == "Directory not found: './not_exist_directory'"


def test_raise_error_if_path_is_not_directory() -> None:
    loader = DirectoryLoader(__file__)
    with pytest.raises(ValueError) as e:
        loader.load()

    assert str(e.value) == f"Expected directory, got file: '{__file__}'"


class CustomLoader(TextLoader):
    """Test loader. Mimics interface of existing file loader."""

    def __init__(self, path: Path, **kwargs: Any) -> None:
        """Initialize the loader."""
        self.path = path

    def load(self) -> List[Document]:
        """Load documents."""
        with open(self.path, "r") as f:
            return [Document(page_content=f.read())]

    def lazy_load(self) -> Iterator[Document]:
        raise NotImplementedError("CustomLoader does not implement lazy_load()")


def test_exclude_ignores_matching_files(tmp_path: Path) -> None:
    txt_file = tmp_path / "test.txt"
    py_file = tmp_path / "test.py"
    txt_file.touch()
    py_file.touch()
    loader = DirectoryLoader(
        str(tmp_path),
        exclude=["*.py"],
        loader_cls=CustomLoader,
    )
    data = loader.load()
    assert len(data) == 1


def test_exclude_as_string_converts_to_sequence() -> None:
    loader = DirectoryLoader("./some_directory", exclude="*.py")
    assert loader.exclude == ("*.py",)


class CustomLoaderMetadataOnly(CustomLoader):
    """Test loader that just returns the file path in metadata. For test_directory_loader_glob_multiple."""  # noqa: E501

    def load(self) -> List[Document]:
        metadata = {"source": self.path}
        return [Document(page_content="", metadata=metadata)]

    def lazy_load(self) -> Iterator[Document]:
        return iter(self.load())


def test_directory_loader_glob_multiple() -> None:
    """Verify that globbing multiple patterns in a list works correctly."""

    path_to_examples = "tests/examples/"
    list_extensions = [".rst", ".txt"]
    list_globs = [f"**/*{ext}" for ext in list_extensions]
    is_file_type_loaded = {ext: False for ext in list_extensions}

    loader = DirectoryLoader(
        path=path_to_examples, glob=list_globs, loader_cls=CustomLoaderMetadataOnly
    )

    list_documents = loader.load()

    for doc in list_documents:
        path_doc = Path(doc.metadata.get("source", ""))
        ext_doc = path_doc.suffix

        if is_file_type_loaded.get(ext_doc, False):
            continue
        elif ext_doc in list_extensions:
            is_file_type_loaded[ext_doc] = True
        else:
            # Loaded a filetype that was not specified in extensions list
            assert False

    for ext in list_extensions:
        assert is_file_type_loaded.get(ext, False)


class TrackingLoader(TextLoader):
    """Loader that records which files it was instantiated with."""

    instances: List[str] = []

    def __init__(self, path: str, **kwargs: Any) -> None:
        TrackingLoader.instances.append(path)
        super().__init__(path, **kwargs)


class AltTrackingLoader(TextLoader):
    """Alternative tracking loader to distinguish from TrackingLoader."""

    instances: List[str] = []

    def __init__(self, path: str, **kwargs: Any) -> None:
        AltTrackingLoader.instances.append(path)
        super().__init__(path, **kwargs)


def test_suffix_loader_map_routes_by_extension(tmp_path: Path) -> None:
    """Files whose suffix is in the map use the mapped loader."""
    rs_file = tmp_path / "main.rs"
    rs_file.write_text("\nfn main() {}\n")
    txt_file = tmp_path / "readme.txt"
    txt_file.write_text("hello")

    TrackingLoader.instances = []
    AltTrackingLoader.instances = []

    loader = DirectoryLoader(
        str(tmp_path),
        glob="*",
        loader_cls=AltTrackingLoader,
        suffix_loader_map={".rs": TrackingLoader},
    )
    docs = loader.load()

    assert len(docs) == 2
    rs_paths = [p for p in TrackingLoader.instances if p.endswith(".rs")]
    txt_paths = [p for p in AltTrackingLoader.instances if p.endswith(".txt")]
    assert len(rs_paths) == 1
    assert len(txt_paths) == 1


def test_suffix_loader_map_default_none_preserves_behavior(tmp_path: Path) -> None:
    """When suffix_loader_map is not provided, all files use loader_cls."""
    txt_file = tmp_path / "file.txt"
    txt_file.write_text("content")

    TrackingLoader.instances = []

    loader = DirectoryLoader(
        str(tmp_path),
        glob="*",
        loader_cls=TrackingLoader,
    )
    docs = loader.load()

    assert len(docs) == 1
    assert any(p.endswith("file.txt") for p in TrackingLoader.instances)


def test_suffix_loader_map_multiple_suffixes(tmp_path: Path) -> None:
    """Multiple suffixes can each map to different loaders."""
    (tmp_path / "a.rs").write_text("\nfn a() {}")
    (tmp_path / "b.go").write_text("package main")
    (tmp_path / "c.txt").write_text("plain text")

    TrackingLoader.instances = []
    AltTrackingLoader.instances = []

    loader = DirectoryLoader(
        str(tmp_path),
        glob="*",
        loader_cls=TextLoader,
        suffix_loader_map={
            ".rs": TrackingLoader,
            ".go": AltTrackingLoader,
        },
    )
    docs = loader.load()

    assert len(docs) == 3
    assert any(p.endswith(".rs") for p in TrackingLoader.instances)
    assert any(p.endswith(".go") for p in AltTrackingLoader.instances)
    assert not any(p.endswith(".txt") for p in TrackingLoader.instances)
    assert not any(p.endswith(".txt") for p in AltTrackingLoader.instances)


def test_suffix_loader_map_case_insensitive(tmp_path: Path) -> None:
    """Suffix matching is case-insensitive (lowered before lookup)."""
    upper_file = tmp_path / "test.RS"
    upper_file.write_text("fn test() {}")

    TrackingLoader.instances = []

    loader = DirectoryLoader(
        str(tmp_path),
        glob="*",
        loader_cls=TextLoader,
        suffix_loader_map={".rs": TrackingLoader},
    )
    docs = loader.load()

    assert len(docs) == 1
    assert len(TrackingLoader.instances) == 1


def test_suffix_loader_map_with_multithreading(tmp_path: Path) -> None:
    """suffix_loader_map works correctly with multithreading enabled."""
    (tmp_path / "main.rs").write_text("\nfn main() {}")
    (tmp_path / "readme.txt").write_text("hello")

    TrackingLoader.instances = []
    AltTrackingLoader.instances = []

    loader = DirectoryLoader(
        str(tmp_path),
        glob="*",
        loader_cls=AltTrackingLoader,
        use_multithreading=True,
        suffix_loader_map={".rs": TrackingLoader},
    )
    docs = loader.load()

    assert len(docs) == 2
    assert any(p.endswith(".rs") for p in TrackingLoader.instances)
    assert any(p.endswith(".txt") for p in AltTrackingLoader.instances)
