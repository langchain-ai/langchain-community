"""Loader for extracting text from scanned PDFs using OCR."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, List, Optional

from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader

logger = logging.getLogger(__name__)


class OCRPDFLoader(BaseLoader):
    """Load scanned PDF files using OCR (Optical Character Recognition).

    This loader converts PDF pages to images and applies Tesseract OCR
    to extract text from scanned documents.

    Setup:
        Install required packages:
        ```bash
        pip install pdf2image pytesseract
        ```

        Install system dependencies:
        - **Linux**: `sudo apt-get install poppler-utils tesseract-ocr`
        - **macOS**: `brew install poppler tesseract`
        - **Windows**: Download and install Poppler and Tesseract, add to PATH

    Example:
        ```python
        from langchain_community.document_loaders import OCRPDFLoader

        loader = OCRPDFLoader("scanned_document.pdf")
        documents = loader.load()

        # Access extracted text and metadata
        for doc in documents:
            print(f"Page {doc.metadata['page']}: {doc.page_content[:100]}...")
        ```
    """

    def __init__(
        self,
        file_path: str | Path,
        *,
        tesseract_config: str = "",
        poppler_path: Optional[str] = None,
        first_page: Optional[int] = None,
        last_page: Optional[int] = None,
        dpi: int = 200,
        fmt: str = "JPEG",
    ) -> None:
        """Initialize the OCR PDF loader.

        Args:
            file_path: Path to the PDF file to load.
            tesseract_config: Additional configuration options for Tesseract OCR.
                Example: "--psm 6" for uniform text blocks.
            poppler_path: Path to poppler installation (Windows only).
            first_page: First page to process (1-indexed). If None, starts from page 1.
            last_page: Last page to process (1-indexed). If None, processes all pages.
            dpi: Resolution for PDF to image conversion. Higher values improve
                OCR accuracy but increase processing time.
            fmt: Image format for conversion ("JPEG", "PNG", etc.).

        Raises:
            FileNotFoundError: If the specified PDF file does not exist.
            ImportError: If required dependencies are not installed.
        """
        try:
            import pdf2image  # noqa: F401
            import pytesseract  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "OCRPDFLoader requires pdf2image and pytesseract. "
                "Install with: pip install pdf2image pytesseract"
            ) from e

        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {self.file_path}")

        self.tesseract_config = tesseract_config
        self.poppler_path = poppler_path
        self.first_page = first_page
        self.last_page = last_page
        self.dpi = dpi
        self.fmt = fmt

    def load(self) -> List[Document]:
        """Load all pages and return as a list of Documents.

        Returns:
            List of Document objects, one per page with extracted text.
        """
        return list(self.lazy_load())

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load pages one at a time.

        Yields:
            Document objects with extracted text and metadata.

        Raises:
            Exception: If PDF processing or OCR fails.
        """
        try:
            from pdf2image import convert_from_path
            import pytesseract
        except ImportError as e:
            raise ImportError(
                "Required dependencies not found. "
                "Install with: pip install pdf2image pytesseract"
            ) from e

        try:
            # Convert PDF pages to images
            conversion_kwargs = {
                "pdf_path": self.file_path,
                "dpi": self.dpi,
                "fmt": self.fmt,
            }

            if self.poppler_path:
                conversion_kwargs["poppler_path"] = self.poppler_path
            if self.first_page:
                conversion_kwargs["first_page"] = self.first_page
            if self.last_page:
                conversion_kwargs["last_page"] = self.last_page

            pages = convert_from_path(**conversion_kwargs)
            total_pages = len(pages)

            logger.info(f"Processing {total_pages} pages from {self.file_path}")

        except Exception as e:
            raise RuntimeError(f"Failed to convert PDF to images: {e}") from e

        # Process each page with OCR
        start_page = self.first_page or 1

        for i, page_image in enumerate(pages):
            page_number = start_page + i

            try:
                # Extract text using Tesseract OCR
                ocr_kwargs = {"image": page_image}
                if self.tesseract_config:
                    ocr_kwargs["config"] = self.tesseract_config

                text = pytesseract.image_to_string(**ocr_kwargs)

                # Only yield documents with non-empty text
                if text.strip():
                    yield Document(
                        page_content=text.strip(),
                        metadata={
                            "source": str(self.file_path),
                            "page": page_number,
                            "total_pages": total_pages,
                            "loader": "OCRPDFLoader",
                            "ocr_engine": "tesseract",
                        },
                    )
                else:
                    logger.warning(f"No text extracted from page {page_number}")

            except Exception as e:
                logger.error(f"OCR failed for page {page_number}: {e}")
                # Continue processing other pages even if one fails
                continue
