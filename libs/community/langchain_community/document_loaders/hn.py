from typing import Any, List

from langchain_core.documents import Document

from langchain_community.document_loaders.web_base import WebBaseLoader


class HNLoader(WebBaseLoader):
    """Load `Hacker News` data.

    It loads data from either main page results or the comments page."""

    def load(self) -> List[Document]:
        """Get important HN webpage information.

        HN webpage components are:
            - title
            - content
            - source url,
            - time of post
            - author of the post
            - number of comments
            - rank of the post
        """
        soup_info = self.scrape()
        if "item" in self.web_path:
            return self.load_comments(soup_info)
        else:
            return self.load_results(soup_info)

    def load_comments(self, soup_info: Any) -> List[Document]:
        """Load comments from a HN post."""
        comments = soup_info.select("tr[class='athing comtr']")
        pagespace = soup_info.select_one("tr[id='pagespace']")
        title = pagespace.get("title") if pagespace else None
        return [
            Document(
                page_content=comment.text.strip(),
                metadata={"source": self.web_path, "title": title},
            )
            for comment in comments
        ]

    def load_results(self, soup: Any) -> List[Document]:
        """Load items from an HN page."""
        items = soup.select("tr[class='athing']")
        documents = []
        for lineItem in items:
            rank_elem = lineItem.select_one("span[class='rank']")
            ranking = rank_elem.text if rank_elem else None
            titleline = lineItem.find("span", {"class": "titleline"})
            if titleline:
                link_elem = titleline.find("a")
                link = link_elem.get("href") if link_elem else None
                title = titleline.text.strip()
            else:
                link = None
                title = None
            metadata = {
                "source": self.web_path,
                "title": title,
                "link": link,
                "ranking": ranking,
            }
            documents.append(
                Document(
                    page_content=title or "",
                    link=link,
                    ranking=ranking,
                    metadata=metadata,
                )
            )
        return documents
