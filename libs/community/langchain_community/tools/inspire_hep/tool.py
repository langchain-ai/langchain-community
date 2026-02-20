"""INSPIRE HEP tools for LangChain agents."""

from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_community.utilities.inspire_hep import INSPIREHEPAPIWrapper


# Input Schemas
class SearchLiteratureInput(BaseModel):
    """Input for the INSPIRE HEP literature search tool."""
    
    query: str = Field(
        description="Search query to fetch physics papers from INSPIRE HEP. " \
        "Convert the query into keywords as the following Examples: 'quantum gravity', "
        "'string theory', 'topcite 1000+' for highly cited papers."
    )
    sort: str = Field(
        default= "mostrecent",
        description = "Sort order: 'mostrecent' for newest papers first,"
        " 'mostcited' for most cited papers firts"
    )


class GetAuthorPapersInput(BaseModel):
    """Input for the INSPIRE HEP author papers tool."""
    
    author_name: str = Field(
        description= "INSPIRE author identifier in format 'Lastname.Firstname.N' (e.g., 'Witten.Edward.1'). "
        "If the user provides a plain name like 'Edward Witten', ask them to find their "
        "INSPIRE identifier at https://inspirehep.net/authors first, as plain names don't work reliably."
    )

    sort: str = Field(
        default= "mostrecent",
        description = "Sort order: 'mostrecent' for the newest papers first of the author,"
        " 'mostcited' for the most cited papers first of the author"
    )
    


class GetPaperDetailsInput(BaseModel):
    """Input for the INSPIRE HEP paper details tool."""
    
    record_id: str = Field(
        description="INSPIRE record ID (a number). Example: '451647' for "
        "Maldacena's AdS/CFT paper."
    )


# Tool Classes
class INSPIRESearchLiteratureTool(BaseTool):
    """Tool for searching high energy physics papers on INSPIRE HEP.
    
    This tool searches the INSPIRE HEP database for physics literature
    based on a query string. Results can be sorted by citations or recency
    
    Example:
        >>> from langchain_community.tools.inspire_hep import INSPIRESearchLiteratureTool
        >>> tool = INSPIRESearchLiteratureTool()
        >>> result = tool.invoke({"query": "quantum gravity", "sort": "mostcited"})
        >>> print(result)
        'Title: Paper 1...\\nCitations: 234\\n---\\n...'
    """
    
    name: str = "inspire_search_literature"
    description: str = Field(
        default=(
            "Search for high energy physics papers on the INSPIRE HEP database. "
            "Use this when the user asks about physics papers, research on specific topics, "
            "The results cab be sorted using 'mostrecent' (default, newest papers first) or 'mostcited'(most cited papers first)"
            "Use 'mostcited' when user, for example, asks for 'most cited, 'highly cited', 'influential', or 'important' papers."
            "Use 'mostrecent' when user asks for 'recent', 'latest', 'new' or does not specify."
        ),
        description=" Instructions for when the LLM should use this tool"
    )
    args_schema: Type[BaseModel] = Field(
        default=SearchLiteratureInput,
        description="Input schema defining required parameters"
    )
    
    api_wrapper: INSPIREHEPAPIWrapper = Field(
        default_factory=INSPIREHEPAPIWrapper,
        description="API wrapper instance for making INSPIRE HEP requests"
    )
    
    def _run(
        self,
        query: str,
        sort : str="mostrecent",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute the literature search and return formatted results."""

    
        result = self.api_wrapper.search_literature(query, sort=sort)
    
        return result


class INSPIREGetAuthorPapersTool(BaseTool):
    """Tool for getting an author's most cited papers from INSPIRE HEP.
    
    This tool retrieves the most highly cited papers by a specific physicist
    using their INSPIRE author identifier.
    
    Example:
        >>> from langchain_community.tools.inspire_hep import INSPIREGetAuthorPapersTool
        >>> tool = INSPIREGetAuthorPapersTool()
        >>> result = tool.invoke({"author_name": "E.Witten.1"})
        >>> print(result)
        'Top papers by E.Witten.1:\\n1. AdS/CFT (5000 citations)\\n...'
    """
    
    name: str = "inspire_get_author_papers"
    description: str = (
        "Use this when the user asks about a specific author's work or publications. "
        "Input must be the author's INSPIRE identifier (e.g., 'E.Witten.1'), "
        "not a plain name. Users can find identifiers at https://inspirehep.net/authors."
        "Use 'mostcited' when user, for example, asks for 'most cited, 'highly cited', 'influential', or 'important' papers of the author."
        "Use 'mostrecent' when user asks for 'recent', 'latest', 'new' or does not specify."
    )
    args_schema: Type[BaseModel] = Field(
        default=GetAuthorPapersInput,
        description= "Input schema defining required parameters"
    )
    api_wrapper: INSPIREHEPAPIWrapper = Field(default_factory=INSPIREHEPAPIWrapper,
                description="API wrapper instance for making INSPIRE HEP requests")
    
    def _run(
        self,
        author_name: str,
        sort : str="mostrecent",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        return self.api_wrapper.get_author_papers(author_name, sort= sort)


class INSPIREGetPaperDetailsTool(BaseTool):
    """Tool for getting details of a specific paper from INSPIRE HEP.
    
    This tool retrieves detailed information about a physics paper using
    its INSPIRE record ID, including title, authors, citations, and abstract.
    
    Example:
        >>> from langchain_community.tools.inspire_hep import INSPIREGetPaperDetailsTool
        >>> tool = INSPIREGetPaperDetailsTool()
        >>> result = tool.invoke({"record_id": "451647"})
        >>> print(result)
        'Title: The Large N limit...\\nAuthors: Maldacena...\\n...'
    """
    
    name: str = "inspire_get_paper_details"
    description: str = (
        "Get detailed information about a specific physics paper using its "
        "INSPIRE record ID. Use this when you have a record ID and need full "
        "details including title, authors, citations, and abstract. "
        "Input should be an INSPIRE record ID (a number like '451647')."
    )
    args_schema: Type[BaseModel] = Field(
        default= GetPaperDetailsInput,
        description="Input schema defining required parameters"
    )
    api_wrapper: INSPIREHEPAPIWrapper = Field(default_factory=INSPIREHEPAPIWrapper,
                description="API wrapper instance for making INSPIRE HEP requests")
    
    def _run(
        self,
        record_id: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        return self.api_wrapper.get_paper_details(record_id)