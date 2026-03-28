"""Toolkit for interacting with Sail SQL."""

from typing import List

from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool
from langchain_core.tools.base import BaseToolkit
from pydantic import ConfigDict, Field

from langchain_community.tools.sail_sql.tool import (
    InfoSailSQLTool,
    ListSailSQLTool,
    QuerySailSQLTool,
    SailQueryCheckerTool,
)
from langchain_community.utilities.sail_sql import SailSQL


class SailSQLToolkit(BaseToolkit):
    """Toolkit for interacting with Sail SQL.

    Parameters:
        db: SailSQL. The Sail SQL database.
        llm: BaseLanguageModel. The language model.
    """

    db: SailSQL = Field(exclude=True)
    llm: BaseLanguageModel = Field(exclude=True)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        return [
            QuerySailSQLTool(db=self.db),
            InfoSailSQLTool(db=self.db),
            ListSailSQLTool(db=self.db),
            SailQueryCheckerTool(db=self.db, llm=self.llm),
        ]
