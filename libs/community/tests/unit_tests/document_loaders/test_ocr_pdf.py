"""Tests for OCR PDF Loader."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from langchain_community.document_loaders.ocr_pdf import OCRPDFLoader


class TestOCRPDFLoader:
    """Test suite for OCRPDFLoader."""

    def test_initialization_with_valid_path(self, tmp_path: Path) -> None:
        """Test loader initialization with valid file path."""
        # Create a temporary PDF file
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"fake pdf content")

        loader = OCRPDFLoader(file_path=str(pdf_file))

        assert loader.file_path == pdf_file
        assert loader.tesseract_config == ""
        assert loader.dpi == 200
        assert loader.fmt == "JPEG"

    def test_initialization_with_custom_params(self, tmp_path: Path) -> None:
        """Test loader initialization with custom parameters."""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"fake pdf content")

        loader = OCRPDFLoader(
            file_path=str(pdf_file),
            tesseract_config="--psm 6",
            dpi=300,
            fmt="PNG",
            first_page=2,
            last_page=5,
        )

        assert loader.tesseract_config == "--psm 6"
        assert loader.dpi == 300
        assert loader.fmt == "PNG"
        assert loader.first_page == 2
        assert loader.last_page == 5

    def test_initialization_file_not_found(self) -> None:
        """Test loader initialization with non-existent file."""
        with pytest.raises(FileNotFoundError, match="PDF file not found"):
            OCRPDFLoader("nonexistent.pdf")

    @patch("pdf2image.convert_from_path")
    @patch("pytesseract.image_to_string")
    def test_missing_dependencies(
        self, mock_pytesseract, mock_pdf2image, tmp_path: Path
    ) -> None:
        """Test error handling when dependencies are missing."""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"fake pdf content")

        # Mock the import to raise ImportError
        with patch.dict("sys.modules", {"pdf2image": None, "pytesseract": None}):
            with patch(
                "builtins.__import__", side_effect=ImportError("pdf2image not found")
            ):
                with pytest.raises(
                    ImportError, match="OCRPDFLoader requires pdf2image"
                ):
                    OCRPDFLoader(str(pdf_file))

    @patch("pdf2image.convert_from_path")
    @patch("pytesseract.image_to_string")
    def test_load_with_mocked_ocr(self, mock_ocr, mock_convert, tmp_path: Path) -> None:
        """Test load() method with mocked OCR results."""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"fake pdf content")

        # Mock dependencies
        mock_pages = [MagicMock(), MagicMock(), MagicMock()]
        mock_convert.return_value = mock_pages
        mock_ocr.side_effect = [
            "Text from page 1",
            "Text from page 2",
            "",  # Empty text (should be skipped)
        ]

        loader = OCRPDFLoader(str(pdf_file))
        documents = loader.load()

        # Should only return 2 documents (skipping empty text)
        assert len(documents) == 2

        # Check first document
        assert isinstance(documents[0], Document)
        assert documents[0].page_content == "Text from page 1"
        assert documents[0].metadata == {
            "source": str(pdf_file),
            "page": 1,
            "total_pages": 3,
            "loader": "OCRPDFLoader",
            "ocr_engine": "tesseract",
        }

        # Check second document
        assert documents[1].page_content == "Text from page 2"
        assert documents[1].metadata["page"] == 2

    @patch("pdf2image.convert_from_path")
    @patch("pytesseract.image_to_string")
    def test_lazy_load_with_mocked_ocr(
        self, mock_ocr, mock_convert, tmp_path: Path
    ) -> None:
        """Test lazy_load() method with mocked OCR results."""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"fake pdf content")

        mock_pages = [MagicMock(), MagicMock()]
        mock_convert.return_value = mock_pages
        mock_ocr.side_effect = ["Page 1 content", "Page 2 content"]

        loader = OCRPDFLoader(str(pdf_file))
        documents = list(loader.lazy_load())

        assert len(documents) == 2
        assert all(isinstance(doc, Document) for doc in documents)

    @patch("pdf2image.convert_from_path")
    @patch("pytesseract.image_to_string")
    def test_load_with_tesseract_config(
        self, mock_ocr, mock_convert, tmp_path: Path
    ) -> None:
        """Test OCR with custom Tesseract configuration."""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"fake pdf content")

        mock_page = MagicMock()
        mock_convert.return_value = [mock_page]
        mock_ocr.return_value = "OCR result"

        loader = OCRPDFLoader(str(pdf_file), tesseract_config="--psm 6")
        list(loader.lazy_load())

        # Verify OCR was called with config
        mock_ocr.assert_called_once_with(image=mock_page, config="--psm 6")

    @patch("pdf2image.convert_from_path")
    @patch("pytesseract.image_to_string")
    def test_load_with_page_range(self, mock_ocr, mock_convert, tmp_path: Path) -> None:
        """Test loading with specific page range."""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"fake pdf content")

        mock_pages = [MagicMock(), MagicMock()]
        mock_convert.return_value = mock_pages
        mock_ocr.side_effect = ["Page 3 content", "Page 4 content"]

        loader = OCRPDFLoader(str(pdf_file), first_page=3, last_page=4)
        documents = loader.load()

        # Check that convert_from_path was called with page range
        mock_convert.assert_called_once()
        call_kwargs = mock_convert.call_args[1]
        assert call_kwargs["first_page"] == 3
        assert call_kwargs["last_page"] == 4

        # Check document metadata reflects correct page numbers
        assert documents[0].metadata["page"] == 3
        assert documents[1].metadata["page"] == 4

    @patch("pdf2image.convert_from_path")
    def test_conversion_error_handling(self, mock_convert, tmp_path: Path) -> None:
        """Test error handling during PDF to image conversion."""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"fake pdf content")

        mock_convert.side_effect = RuntimeError("Conversion failed")

        loader = OCRPDFLoader(str(pdf_file))

        with pytest.raises(RuntimeError, match="Failed to convert PDF to images"):
            list(loader.lazy_load())

    @patch("pdf2image.convert_from_path")
    @patch("pytesseract.image_to_string")
    def test_ocr_error_handling(
        self, mock_ocr, mock_convert, tmp_path: Path, caplog
    ) -> None:
        """Test error handling during OCR processing."""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"fake pdf content")

        mock_pages = [MagicMock(), MagicMock()]
        mock_convert.return_value = mock_pages
        mock_ocr.side_effect = [
            "Successful OCR",
            RuntimeError("OCR failed"),
        ]

        loader = OCRPDFLoader(str(pdf_file))
        documents = loader.load()

        # Should return only the successful document
        assert len(documents) == 1
        assert documents[0].page_content == "Successful OCR"

        # Should log error for failed page
        assert "OCR failed for page 2" in caplog.text

    @patch("pdf2image.convert_from_path")
    @patch("pytesseract.image_to_string")
    def test_empty_text_filtering(
        self, mock_ocr, mock_convert, tmp_path: Path, caplog
    ) -> None:
        """Test that pages with empty OCR results are filtered out."""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"fake pdf content")

        mock_pages = [MagicMock(), MagicMock()]
        mock_convert.return_value = mock_pages
        mock_ocr.side_effect = [
            "Good content",
            "   \n\t  ",
        ]  # Second is whitespace only

        loader = OCRPDFLoader(str(pdf_file))
        documents = loader.load()

        # Should only return document with actual content
        assert len(documents) == 1
        assert documents[0].page_content == "Good content"

        # Should log warning for empty page
        assert "No text extracted from page 2" in caplog.text

    def test_pathlib_path_support(self, tmp_path: Path) -> None:
        """Test that loader accepts pathlib.Path objects."""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"fake pdf content")

        # Should accept Path object directly
        loader = OCRPDFLoader(pdf_file)
        assert loader.file_path == pdf_file

    @patch("pdf2image.convert_from_path")
    @patch("pytesseract.image_to_string")
    def test_poppler_path_parameter(
        self, mock_ocr, mock_convert, tmp_path: Path
    ) -> None:
        """Test that poppler_path is passed to convert_from_path."""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"fake pdf content")

        mock_convert.return_value = [MagicMock()]
        mock_ocr.return_value = "Test content"

        loader = OCRPDFLoader(str(pdf_file), poppler_path="/custom/poppler/path")
        list(loader.lazy_load())

        # Verify poppler_path was passed
        call_kwargs = mock_convert.call_args[1]
        assert call_kwargs["poppler_path"] == "/custom/poppler/path"

    @patch("pdf2image.convert_from_path")
    @patch("pytesseract.image_to_string")
    def test_dpi_and_format_parameters(
        self, mock_ocr, mock_convert, tmp_path: Path
    ) -> None:
        """Test that DPI and format parameters are passed correctly."""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"fake pdf content")

        mock_convert.return_value = [MagicMock()]
        mock_ocr.return_value = "Test content"

        loader = OCRPDFLoader(str(pdf_file), dpi=300, fmt="PNG")
        list(loader.lazy_load())

        # Verify parameters were passed
        call_kwargs = mock_convert.call_args[1]
        assert call_kwargs["dpi"] == 300
        assert call_kwargs["fmt"] == "PNG"
