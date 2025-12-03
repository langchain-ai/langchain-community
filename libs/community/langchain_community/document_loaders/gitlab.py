import urllib.parse
from abc import ABC
from datetime import datetime
from typing import Any, Callable, Dict, Iterator, List, Literal, Optional, Union

import requests
from langchain_core.documents import Document
from langchain_core.utils import get_from_dict_or_env
from pydantic import BaseModel, field_validator, model_validator

from langchain_community.document_loaders.base import BaseLoader


class BaseGitLabLoader(BaseLoader, BaseModel, ABC):
    """Load `GitLab` repository Issues."""

    project_path: str
    """Path of the project (e.g., 'group/project' or 'owner/project')"""
    access_token: str
    """Personal access token - see https://gitlab.com/-/user_settings/personal_access_tokens"""
    gitlab_api_url: str = "https://gitlab.com/api/v4"
    """URL of GitLab API (defaults to gitlab.com)"""

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that access token exists in environment."""
        values["access_token"] = get_from_dict_or_env(
            values, "access_token", "GITLAB_PERSONAL_ACCESS_TOKEN"
        )
        return values

    @property
    def headers(self) -> Dict[str, str]:
        return {
            "PRIVATE-TOKEN": self.access_token,
        }


class GitLabIssuesLoader(BaseGitLabLoader):
    """Load issues of a GitLab repository."""

    state: Optional[Literal["opened", "closed", "locked", "all"]] = None
    """Filter on issue state. Can be one of: 'opened', 'closed', 'locked', 'all'."""
    labels: Optional[List[str]] = None
    """Label names to filter on. Example: bug,ui,high."""
    milestone: Optional[str] = None
    """Filter on milestone title."""
    assignee_username: Optional[str] = None
    """Filter on assigned user by username."""
    author_username: Optional[str] = None
    """Filter on author user by username."""
    search: Optional[str] = None
    """Search issues against their title and description."""
    sort: Optional[Literal["created_at", "updated_at", "priority", "due_date", "label_priority", "milestone_due", "popularity", "weight"]] = None
    """What to sort results by. Can be one of: 'created_at', 'updated_at', 'priority', 'due_date', 'label_priority', 'milestone_due', 'popularity', 'weight'."""
    order_by: Optional[Literal["created_at", "updated_at"]] = None
    """Return issues ordered by 'created_at' or 'updated_at' fields."""
    updated_after: Optional[str] = None
    """Return issues updated after the specified time.
        This is a timestamp in ISO 8601 format: YYYY-MM-DDTHH:MM:SSZ."""
    updated_before: Optional[str] = None
    """Return issues updated before the specified time.
        This is a timestamp in ISO 8601 format: YYYY-MM-DDTHH:MM:SSZ."""
    page: Optional[int] = None
    """The page number for paginated results.
        Defaults to 1 in the GitLab API."""
    per_page: Optional[int] = None
    """Number of items per page.
        Defaults to 20 in the GitLab API."""

    @field_validator("updated_after", "updated_before")
    @classmethod
    def validate_timestamp(cls, v: Optional[str]) -> Optional[str]:
        if v:
            try:
                datetime.strptime(v, "%Y-%m-%dT%H:%M:%SZ")
            except ValueError:
                raise ValueError(
                    "Invalid value for timestamp. Expected a date string in "
                    f"YYYY-MM-DDTHH:MM:SSZ format. Received: {v}"
                )
        return v

    def lazy_load(self) -> Iterator[Document]:
        """
        Get issues of a GitLab repository.

        Returns:
            A list of Documents with attributes:
                - page_content
                - metadata
                    - url
                    - title
                    - author
                    - created_at
                    - updated_at
                    - closed_at
                    - user_notes_count
                    - state
                    - labels
                    - assignee
                    - assignees
                    - milestone
                    - iid
        """
        url: Optional[str] = self.url
        while url:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            issues = response.json()
            for issue in issues:
                doc = self.parse_issue(issue)
                yield doc
            # GitLab uses Link header for pagination
            if (
                response.headers.get("Link")
                and 'rel="next"' in response.headers.get("Link", "")
                and (not self.page and not self.per_page)
            ):
                # Extract next URL from Link header
                link_header = response.headers.get("Link", "")
                for link in link_header.split(","):
                    if 'rel="next"' in link:
                        url = link.split(";")[0].strip("<>")
                        break
                else:
                    url = None
            else:
                url = None

    def parse_issue(self, issue: dict) -> Document:
        """Create Document objects from a list of GitLab issues."""
        metadata = {
            "url": issue.get("web_url", ""),
            "title": issue.get("title", ""),
            "author": issue.get("author", {}).get("username", "") if issue.get("author") else None,
            "created_at": issue.get("created_at", ""),
            "updated_at": issue.get("updated_at", ""),
            "closed_at": issue.get("closed_at"),
            "user_notes_count": issue.get("user_notes_count", 0),
            "state": issue.get("state", ""),
            "labels": issue.get("labels", []),
            "assignee": issue.get("assignee", {}).get("username", "") if issue.get("assignee") else None,
            "assignees": [
                assignee.get("username", "") for assignee in issue.get("assignees", [])
            ],
            "milestone": issue.get("milestone", {}).get("title", "") if issue.get("milestone") else None,
            "iid": issue.get("iid"),
        }
        content = issue.get("description", "") if issue.get("description") is not None else ""
        return Document(page_content=content, metadata=metadata)

    @property
    def query_params(self) -> str:
        """Create query parameters for GitLab API."""
        labels = ",".join(self.labels) if self.labels else self.labels
        query_params_dict = {
            "state": self.state,
            "labels": labels,
            "milestone": self.milestone,
            "assignee_username": self.assignee_username,
            "author_username": self.author_username,
            "search": self.search,
            "sort": self.sort,
            "order_by": self.order_by,
            "updated_after": self.updated_after,
            "updated_before": self.updated_before,
            "page": self.page,
            "per_page": self.per_page,
        }
        query_params_list = [
            f"{k}={v}" for k, v in query_params_dict.items() if v is not None
        ]
        query_params = "&".join(query_params_list)
        return query_params

    @property
    def url(self) -> str:
        """Create URL for GitLab API."""
        # URL encode the project path
        encoded_project_path = urllib.parse.quote(self.project_path, safe="")
        return f"{self.gitlab_api_url}/projects/{encoded_project_path}/issues?{self.query_params}"


class GitLabFileLoader(BaseGitLabLoader, ABC):
    """Load GitLab File"""

    branch: str = "main"
    """Branch name to load files from"""

    file_filter: Optional[Callable[[str], bool]]
    """Optional filter function to filter file paths"""

    def get_file_paths(self) -> List[Dict]:
        """Get all file paths from the repository tree."""
        encoded_project_path = urllib.parse.quote(self.project_path, safe="")
        base_url = (
            f"{self.gitlab_api_url}/projects/{encoded_project_path}/repository/tree"
            f"?ref={self.branch}&recursive=true&per_page=100"
        )

        all_files = []
        url = base_url
        while url:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            files = response.json()
            all_files.extend(files)

            # GitLab uses Link header for pagination
            if (
                response.headers.get("Link")
                and 'rel="next"' in response.headers.get("Link", "")
            ):
                link_header = response.headers.get("Link", "")
                for link in link_header.split(","):
                    if 'rel="next"' in link:
                        url = link.split(";")[0].strip("<>")
                        break
                else:
                    url = None
            else:
                url = None

        """ one element in all_files
        {
            'id': 'a1b2c3d4...',
            'name': 'file.py',
            'type': 'blob',
            'path': 'path/to/file.py',
            'mode': '100644'
        }
        """
        return [
            f
            for f in all_files
            if f.get("type") == "blob"
            and not (self.file_filter and not self.file_filter(f.get("path", "")))
        ]

    def get_file_content_by_path(self, path: str) -> str:
        """Get file content by path."""
        encoded_project_path = urllib.parse.quote(self.project_path, safe="")
        encoded_path = urllib.parse.quote(path, safe="")
        queryparams = f"?ref={self.branch}" if self.branch else ""
        base_url = (
            f"{self.gitlab_api_url}/projects/{encoded_project_path}/repository/files/{encoded_path}/raw{queryparams}"
        )
        response = requests.get(base_url, headers=self.headers)
        response.raise_for_status()

        return response.text

    def lazy_load(self) -> Iterator[Document]:
        """Load files from GitLab repository."""
        files = self.get_file_paths()
        for file in files:
            content = self.get_file_content_by_path(file["path"])
            if content == "":
                continue

            metadata = {
                "path": file["path"],
                "name": file.get("name", ""),
                "source": f"{self.gitlab_api_url.replace('/api/v4', '')}/{self.project_path}/-/blob/{self.branch}/{file['path']}",
            }
            yield Document(page_content=content, metadata=metadata)

