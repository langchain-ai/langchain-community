"""Load documents using Landing AI's ADE (Agentic Document Extraction)."""

from pathlib import Path
from typing import Iterator, Literal, Optional, Union
from urllib.parse import urlparse

from langchain_core.documents import Document
from langchain_core.utils import get_from_dict_or_env

from langchain_community.document_loaders.base import BaseLoader


class LandingAIADEDocumentLoader(BaseLoader):
    """Load documents using Landing AI's ADE (Agentic Document Extraction).

    Landing AI ADE is an intelligent document processing system that converts
    various document formats into structured markdown with metadata.

    Supported formats include:
    - PDFs
    - Images: JPEG, JPG, PNG, APNG, BMP, DCX, DDS, DIB, GD, GIF, ICNS, JP2, PCX,
      PPM, PSD, TGA, TIFF, WEBP
    - Text documents: DOC, DOCX, ODT
    - Presentations: ODP, PPT, PPTX
    - Spreadsheets: CSV, XLSX

    To use this loader, you need a Landing AI API key. You can obtain one from:
    https://va.landing.ai/my/settings/api-key

    The API key can be passed directly or set as the VISION_AGENT_API_KEY
    environment variable.

    Example:
        .. code-block:: python

            from langchain_community.document_loaders import (
                LandingAIADEDocumentLoader
            )

            loader = LandingAIADEDocumentLoader(
                file_path="example.pdf",
                api_key="your-api-key",
                model="dpt-2-latest"
            )
            documents = loader.load()

        For URL-based loading:

        .. code-block:: python

            loader = LandingAIADEDocumentLoader(
                file_path="https://example.com/document.pdf",
                api_key="your-api-key"
            )
            documents = loader.load()
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        api_key: Optional[str] = None,
        model: str = "dpt-2-latest",
        environment: Literal["production", "eu"] = "production",
        **kwargs: dict,
    ) -> None:
        """Initialize the loader.

        Args:
            file_path: Path to the document file (local or URL).
            api_key: Landing AI API key. If not provided, will look for
                VISION_AGENT_API_KEY environment variable.
            model: Model to use for document parsing.
            environment: API environment (`'production'` or `'eu'`).
            **kwargs: Additional keyword arguments.
        """
        self.file_path = str(file_path)
        self.api_key = get_from_dict_or_env(
            kwargs, "api_key", "VISION_AGENT_API_KEY", default=api_key
        )
        self.model = model
        self.environment = environment

        try:
            from landingai_ade import LandingAIADE

            self.client = LandingAIADE(
                apikey=self.api_key, environment=self.environment
            )
        except ImportError:
            msg = (
                "Could not import landingai-ade python package. "
                "Please install it with `pip install landingai-ade`."
            )
            raise ImportError(msg)

    def lazy_load(self) -> Iterator[Document]:
        """Lazily load documents from the file.

        Yields:
            Document objects with page content and metadata.
        """
        url_parse_result = urlparse(self.file_path)

        if url_parse_result.scheme in ["http", "https"]:
            response = self.client.parse(
                document_url=self.file_path,
                model=self.model,
            )
        else:
            response = self.client.parse(
                document=Path(self.file_path),
                model=self.model,
            )

        for idx, chunk in enumerate(response.chunks):
            metadata = {"source": self.file_path, "chunk": idx}

            if chunk.grounding:
                metadata["page"] = chunk.grounding.page

            yield Document(
                page_content=chunk.markdown,
                metadata=metadata,
            )

    def load(self) -> list[Document]:
        """Load documents from the file.

        Returns:
            List of Document objects.
        """
        return list(self.lazy_load())
