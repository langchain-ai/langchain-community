from typing import Iterator

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document


class GeniusLoader(BaseLoader):
    """Load lyrics using the Genius API.

    This loader utilizes the `lyricsgenius` Python package to fetch song lyrics
    and metadata. You need a Genius API token, which can be generated at
    https://genius.com/api-clients.
    """

    def __init__(self, search_query: str, api_token: str = None):
        """Initialize with search query and API token.

        Args:
            search_query: The search query (e.g., "Imagine Dragons - Radioactive").
            api_token: Genius API Token. If not provided, looks for GENIUS_ACCESS_TOKEN env var.
        """
        self.search_query = search_query
        self.api_token = api_token

    def lazy_load(self) -> Iterator[Document]:
        """Load lyrics and metadata."""
        try:
            import lyricsgenius
        except ImportError:
            raise ImportError(
                "lyricsgenius package not found, please install it with "
                "`pip install lyricsgenius`"
            )

        # Initialize Genius client
        genius = lyricsgenius.Genius(self.api_token)

        # Search for the song (we use the first best match)
        song = genius.search_song(self.search_query)

        # If no song is found, yield nothing (empty iterator)
        if not song:
            return

        # Create the LangChain Document
        metadata = {
            "source": "Genius",
            "title": song.title,
            "artist": song.artist,
            "album": song.album,
            "url": song.url,
            "id": song.id,
        }

        yield Document(page_content=song.lyrics, metadata=metadata)
