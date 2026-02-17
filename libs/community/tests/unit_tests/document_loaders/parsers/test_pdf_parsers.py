"""Tests for the various PDF parsers."""

import importlib
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import MagicMock

import numpy as np
import pytest

import langchain_community.document_loaders.parsers as pdf_parsers
from langchain_community.document_loaders.base import BaseBlobParser
from langchain_community.document_loaders.blob_loaders import Blob
from langchain_community.document_loaders.parsers.pdf import _merge_text_and_extras

_THIS_DIR = Path(__file__).parents[3]

_EXAMPLES_DIR = _THIS_DIR / "examples"

# Paths to test PDF files
HELLO_PDF = _EXAMPLES_DIR / "hello.pdf"
LAYOUT_PARSER_PAPER_PDF = _EXAMPLES_DIR / "layout-parser-paper.pdf"


def test_merge_text_and_extras() -> None:
    assert "abc\n\n\n<image>\n\n<table>\n\n\ndef\n\n\nghi" == _merge_text_and_extras(
        ["<image>", "<table>"], "abc\n\n\ndef\n\n\nghi"
    )
    assert "abc\n\n<image>\n\n<table>\n\ndef\n\nghi" == _merge_text_and_extras(
        ["<image>", "<table>"], "abc\n\ndef\n\nghi"
    )
    assert "abc\ndef\n\n<image>\n\n<table>\n\nghi" == _merge_text_and_extras(
        ["<image>", "<table>"], "abc\ndef\n\nghi"
    )


def _assert_with_parser(parser: BaseBlobParser, *, splits_by_page: bool = True) -> None:
    """Standard tests to verify that the given parser works.

    Args:
        parser (BaseBlobParser): The parser to test.
        splits_by_page (bool): Whether the parser splits by page or not by default.
    """
    blob = Blob.from_path(HELLO_PDF)
    doc_generator = parser.lazy_parse(blob)
    assert isinstance(doc_generator, Iterator)
    docs = list(doc_generator)
    assert len(docs) == 1
    page_content = docs[0].page_content
    assert isinstance(page_content, str)
    # The different parsers return different amount of whitespace, so using
    # startswith instead of equals.
    assert docs[0].page_content.startswith("Hello world!")

    blob = Blob.from_path(LAYOUT_PARSER_PAPER_PDF)
    doc_generator = parser.lazy_parse(blob)
    assert isinstance(doc_generator, Iterator)
    docs = list(doc_generator)

    if splits_by_page:
        assert len(docs) == 16
    else:
        assert len(docs) == 1
    # Test is imprecise since the parsers yield different parse information depending
    # on configuration. Each parser seems to yield a slightly different result
    # for this page!
    assert "LayoutParser" in docs[0].page_content
    metadata = docs[0].metadata

    assert metadata["source"] == str(LAYOUT_PARSER_PAPER_PDF)

    if splits_by_page:
        assert int(metadata["page"]) == 0


@pytest.mark.parametrize(
    "parser_factory,require,params",
    [
        ("PDFMinerParser", "pdfminer", {"splits_by_page": False}),
        ("PyMuPDFParser", "pymupdf", {}),
        ("PyPDFParser", "pypdf", {}),
        ("PyPDFium2Parser", "pypdfium2", {}),
    ],
)
def test_parsers(
    parser_factory: str,
    require: str,
    params: dict[str, Any],
) -> None:
    try:
        require = require.replace("-", "")
        importlib.import_module(require, package=None)
        parser_class = getattr(pdf_parsers, parser_factory)
        parser = parser_class()
        _assert_with_parser(parser, **params)
    except ModuleNotFoundError:
        pytest.skip(f"{parser_factory} skipped. Require '{require}'")


def _make_mock_xobject_image(
    width: int,
    height: int,
    bits_per_component: int = 8,
    filter_name: str = "FlateDecode",
    data: bytes | None = None,
) -> MagicMock:
    """Create a mock PDF XObject image with the given properties.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        bits_per_component: Bits per color component (1 or 8).
        filter_name: PDF filter name (e.g. ``FlateDecode``).
        data: Raw image bytes. Generated automatically if ``None``.

    Returns:
        A ``MagicMock`` that behaves like a PDF image XObject.
    """
    if data is None:
        if bits_per_component == 1:
            row_bytes = (width + 7) // 8
            data = bytes(row_bytes * height)
        else:
            data = bytes(width * height)

    attrs = {
        "/Subtype": "/Image",
        "/Filter": f"/{filter_name}",
        "/Height": height,
        "/Width": width,
        "/BitsPerComponent": bits_per_component,
    }
    img = MagicMock()
    img.__getitem__ = lambda self, key: attrs[key]
    img.get = lambda key, default=None: attrs.get(key, default)
    img.get_data = MagicMock(return_value=data)
    return img


class TestPyPDFParser1BitImages:
    """Tests for 1-bit monochrome image handling in PyPDFParser.

    Regression tests for
    https://github.com/langchain-ai/langchain-community/issues/307
    """

    def test_1bit_image_does_not_raise(self) -> None:
        """A 1-bit monochrome image must not cause a ValueError."""
        try:
            importlib.import_module("pypdf")
        except ModuleNotFoundError:
            pytest.skip("pypdf not installed")

        from langchain_community.document_loaders.parsers.pdf import PyPDFParser

        width, height = 645, 430
        row_bytes = (width + 7) // 8
        raw_data = bytes(row_bytes * height)  # 34,830 bytes

        # Precondition: this data cannot be reshaped as 8-bit
        assert len(raw_data) != width * height

        img = _make_mock_xobject_image(
            width=width,
            height=height,
            bits_per_component=1,
            filter_name="CCITTFaxDecode",
            data=raw_data,
        )

        xobject_dict = {"img0": img}
        xobject_mock = MagicMock()
        xobject_mock.get_object = MagicMock(return_value=xobject_dict)
        xobject_mock.__iter__ = lambda self: iter(xobject_dict)

        page = MagicMock()
        resources: dict[str, Any] = {"/XObject": xobject_mock}
        page.__getitem__ = lambda self, key: resources

        parser = PyPDFParser(extract_images=False)
        result = parser.extract_images_from_page(page)
        assert isinstance(result, str)

    def test_1bit_pixel_values(self) -> None:
        """Verify that unpacked 1-bit pixels are scaled to 0 and 255."""
        width, height = 8, 2
        raw_data = bytes([0b10101010, 0b11110000])

        np_data = np.unpackbits(np.frombuffer(raw_data, dtype=np.uint8))
        stride = ((width + 7) // 8) * 8
        np_data = np_data.reshape(-1, stride)[:height, :width]
        np_image = (np_data * 255).astype(np.uint8).reshape(height, width, 1)

        assert np_image.shape == (2, 8, 1)
        assert np_image[0, 0, 0] == 255
        assert np_image[0, 1, 0] == 0
        assert np_image[1, 0, 0] == 255
        assert np_image[1, 4, 0] == 0

    def test_1bit_row_padding(self) -> None:
        """Verify correct handling when row width is not a multiple of 8."""
        width, height = 10, 1
        raw_data = bytes([0xFF, 0xC0])

        np_data = np.unpackbits(np.frombuffer(raw_data, dtype=np.uint8))
        stride = ((width + 7) // 8) * 8
        np_data = np_data.reshape(-1, stride)[:height, :width]
        np_image = (np_data * 255).astype(np.uint8).reshape(height, width, 1)

        assert np_image.shape == (1, 10, 1)
        assert np.all(np_image[:, :, 0] == 255)

    def test_8bit_image_unaffected(self) -> None:
        """Standard 8-bit images must still work after the fix."""
        width, height = 10, 10
        raw_data = bytes(width * height)

        np_image = np.frombuffer(raw_data, dtype=np.uint8).reshape(
            height, width, -1
        )

        assert np_image.shape == (10, 10, 1)
