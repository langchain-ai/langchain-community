"""Tool for the WebSearchPlus search API."""
import json
import warnings
from typing import List, Optional, Type, Union, Annotated, Literal

from langchain_core.tools import BaseTool
from pydantic import Field
from langchain_community.utilities.websearchplus_search import WebSearchPlusAPIWrapper, WebSearchPlusInput


class WebSearchPlusResults(BaseTool):
    """Web Search Plus is a search tool specifically developed for LLM that can intelligently filter out invalid information.
    It first intelligently segment the search results, and then uses hybrid search technology to find content with high relevance to your search term in the search results.
    * Cost Reduction
    * Better Representation
    * Reduced Hallucination
    
    Get 4M tokens for free, no credit card required.
    Please refer to the official documentation for more details.
    `websearch.plus <https://websearch.plus/>`_
    
    ### Request Body
    
    * query
    string — Required
    
    Search query, Maybe is from the LLM's tool_call. (1–200 characters)

    * search_context_size
    string — Optional
    
    Controls search depth: low, medium (default), or high. Affects number of results and pricing.
        - low: For some lightweight search needs, such as weather, stock prices, etc.
        - medium: Can be applied to most search scenarios.
        - high: Being able to conduct in-depth searches on a topic.
    
    * options.result_type
    string — Optional
    Result type, list: returns structured, text: concatenates the contents of the list into text to meet the content parameter requirements of call_tool in LLM, making it convenient for developers to directly write the results into the method calling LLM
        - list: returns structured result as a list.
        - text: concatenates the list into a single string suitable for input of LLM tool calling.
    
    * options.language
    string — Optional
    
    Language of expected results (ISO-639-1), "en" is (default).

    * options.type
    string — Optional
    
    Search type: search (default) or news.
        - search: Search across the entire network.
        - news: Only search for reliable news sources, very suitable for searching real-time content.
    
    * options.qdr
    string — Optional
    
    Time filter: any (default), h, d, w, m, y.
        - any: Publication time unlimited.
        - h: Publication time within one hour.
        - d: Publication time within one day.
        - w: Publication time within one week.
        - m: Publication time within one month.
        - y: Publication time within one year.
    
    * options.mode
    string — Optional
    
    If smart, returns hybrid search result. If full, returns full content of web page. Default: smart.
        - smart: Hybrid search can intelligently filter out over 50% of low relevance content.
        - full: Returning the entire text content of the search results page may generate a large amount of redundant information.
    
    ### Example
    .. code-block:: json
        {
            "query": "latest news about LLM",
            "search_context_size": "medium",
            "options":{
                "language": "en",
                "mode": "smart",
                "type": "search",
                "result_type": "text",
                "qdr": "w"
            }
        }
    Setup:
        Install ``langchain-community`` and ``httpx``.
    
        .. code-block:: bash
            pip install -U langchain-community httpx
    
    Instantiation:
        .. code-block:: python

from langchain_community.tools import WebSearchPlusResults
from langchain_community.utilities.websearchplus_search import WebSearchPlusAPIWrapper

            
            api_wrapper = WebSearchPlusAPIWrapper(websearchplus_api_key="your_api_key_here")  # type: ignore[arg-type]
            websearchplus_tool = WebSearchPlusResults(api_wrapper=api_wrapper)
    
    Invocation with ToolCall:
        .. code-block:: python
            tool.invoke(input = {"query":"2025 WWDC"})
            
        .. code-block:: python
            ToolMessage(content="[
                {
                    "content": " By MacRumors Staff on June 17, 2025 At a Glance WWDC is Apple's annual Worldwide Developers Conference where developers can attend sessions and interface with Apple engineers. Apple's 2025 event kicked off June 9 with a hybrid remote/in-person format. Announcements iOS 26 and iPadOS 26 macOS Tahoe 26 watchOS 26 tvOS 26 visionOS 26 Video Recap play Roundup Archived 06/2025 Subscribe for regular MacRumors news and future WWDC 2025 info. WWDC 2025 Overview Apple's 36th annual Worldwide Developers Conference began on Monday, June 9, 2025, and ended on Friday, June 13.",
                    "url": "https://www.macrumors.com/roundup/wwdc/",
                    "title": "WWDC 2025",
                    "score": 0.8140527606010437
                },
                {
                    "content": " While WWDC 2025 is an online event, Apple included a special in-person component for select developers, students, and members of the media. The in-person WWDC event took place on June 9 at the Apple Park campus in Cupertino, California. Invited attendees were able to watch the keynote and Platforms State of the Union at Apple Park, as well as meet with Apple employees.",
                    "url": "https://www.macrumors.com/roundup/wwdc/",
                    "title": "WWDC 2025",
                    "score": 0.7549149990081787
                }
            ]")
    """ # noqa: E501

    name: str = "websearchplus_results"
    description: str = (
        "A wrapper around the WebSearchPlus API. "
        "Useful for searching the web and retrieving results, intelligently filter out invalid information. "
        "Input should be a search query."
    )
    api_wrapper: WebSearchPlusAPIWrapper = Field(default_factory=WebSearchPlusAPIWrapper) # type: ignore[arg-type]
    """Output format of the search results."""
    
    def _run(self, **kwargs) -> Union[str, List[dict]]:
        """Use the tool to perform a web search.
        
        Args:
            **kwargs: Keyword arguments that match WebSearchPlusInput fields:
                query (str): The search query string
                search_context_size (str, optional): Controls search depth ('low', 'medium', 'high')
                options (dict, optional): Additional search configuration options
        
        Returns:
            Union[str, List[dict]]: Search results as either formatted text or a list of result objects
        
        Raises:
            Exception: If the web search fails, returns the error message as a string
        """
        try:
            input_model = WebSearchPlusInput(**kwargs)    
            results = self.api_wrapper._run(
                input=input_model.model_dump(exclude_unset=True, exclude_none=True)
                # input=input
            )
            return results
        except (httpx.RequestError, ConnectionError) as e:
            warnings.warn(f"Network error during WebSearchPlus search: {e}")
            return f"Error: Could not complete search due to network issues: {e}"
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Unexpected error in WebSearchPlus: {e}")
            raise