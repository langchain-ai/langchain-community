# flake8: noqa
"""Tools for interacting with Sail SQL."""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, model_validator, ConfigDict

from langchain_core.language_models import BaseLanguageModel
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.prompts import PromptTemplate
from langchain_community.utilities.sail_sql import SailSQL
from langchain_core.tools import BaseTool
from langchain_community.tools.sail_sql.prompt import QUERY_CHECKER


class BaseSailSQLTool(BaseModel):
    """Base tool for interacting with Sail SQL."""

    db: SailSQL = Field(exclude=True)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )


class QuerySailSQLTool(BaseSailSQLTool, BaseTool):
    """Tool for querying Sail SQL."""

    name: str = "query_sql_db"
    description: str = """
    Input to this tool is a detailed and correct SQL query, output is a result from Sail SQL.
    If the query is not correct, an error message will be returned.
    If an error is returned, rewrite the query, check the query, and try again.
    """

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute the query, return the results or an error message."""
        return self.db.run_no_throw(query)


class InfoSailSQLTool(BaseSailSQLTool, BaseTool):
    """Tool for getting metadata about Sail SQL."""

    name: str = "schema_sql_db"
    description: str = """
    Input to this tool is a comma-separated list of tables, output is the schema and sample rows for those tables.
    Be sure that the tables actually exist by calling list_tables_sql_db first!

    Example Input: "table1, table2, table3"
    """

    def _run(
        self,
        table_names: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Get the schema for tables in a comma-separated list."""
        return self.db.get_table_info_no_throw(table_names.split(", "))


class ListSailSQLTool(BaseSailSQLTool, BaseTool):
    """Tool for getting tables names."""

    name: str = "list_tables_sql_db"
    description: str = "Input is an empty string, output is a comma separated list of tables in Sail SQL."

    def _run(
        self,
        tool_input: str = "",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Get the schema for a specific table."""
        return ", ".join(self.db.get_usable_table_names())


class SailQueryCheckerTool(BaseSailSQLTool, BaseTool):
    """Use an LLM to check if a query is correct.
    Adapted from https://www.patterns.app/blog/2023/01/18/crunchbot-sql-analyst-gpt/"""

    template: str = QUERY_CHECKER
    llm: BaseLanguageModel
    llm_chain: Any = Field(init=False)
    name: str = "query_checker_sql_db"
    description: str = """
    Use this tool to double check if your query is correct before executing it.
    Always use this tool before executing a query with query_sql_db!
    """

    @model_validator(mode="before")
    @classmethod
    def initialize_llm_chain(cls, values: Dict[str, Any]) -> Any:
        if "llm_chain" not in values:
            from langchain_classic.chains.llm import LLMChain

            values["llm_chain"] = LLMChain(
                llm=values.get("llm"),  # type: ignore[arg-type]
                prompt=PromptTemplate(
                    template=QUERY_CHECKER, input_variables=["query"]
                ),
            )

        if values["llm_chain"].prompt.input_variables != ["query"]:
            raise ValueError(
                "LLM chain for SailQueryCheckerTool need to use ['query'] as input_variables "
                "for the embedded prompt"
            )

        return values

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the LLM to check the query."""
        return self.llm_chain.predict(
            query=query, callbacks=run_manager.get_child() if run_manager else None
        )

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        return await self.llm_chain.apredict(
            query=query, callbacks=run_manager.get_child() if run_manager else None
        )
