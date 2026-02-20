"""INSPIRE HEP API wrapper for accessing high energy physics literature."""

from typing import Optional
from pydantic import BaseModel, Field
import requests
import re


class INSPIREHEPAPIWrapper(BaseModel):
    """Wrapper for the INSPIRE HEP REST API.
    
    INSPIRE (https://inspirehep.net) is a trusted community hub that helps 
    researchers share and find accurate scholarly information in high energy physics.
    
    This wrapper provides programmatic access to search literature, retrieve 
    author papers, and get detailed paper information.
    
    Attributes:
        base_url: Base URL for the INSPIRE HEP API. Defaults to the production API.
        top_k_results: Default number of results to return per query. Must be 
            between 1 and 1000.
    
    Example:
        >>> from langchain_community.utilities import INSPIREHEPAPIWrapper
        >>> inspire = INSPIREHEPAPIWrapper()
        >>> inspire.search_literature("quantum gravity")
        'Title: Paper 1...\\nCitations: 234\\n---\\n...'
        
        >>> inspire.get_author_papers("E.Witten.1")
        'Top papers by E.Witten.1:\\n1. Paper title (1000 citations)\\n...'
    """
    
    base_url: str = "https://inspirehep.net/api"
    top_k_results: int = Field(default=10, gt=0, le=1000)
    
    class Config:
        """Pydantic config. Added for later convenience if the model type is not supported 
        by default"""
        arbitrary_types_allowed = True
    
    def _make_request(self, endpoint: str, params: dict = None) -> dict:
        """Make a rate-limited GET request to the INSPIRE HEP API.
        
        Args:
            endpoint: API endpoint path (e.g., 'literature/451647')
            params: {
                        'q':  Search query,
                        'sort': Sort order ('mostrecent', 'mostcited),   
                        'size': Results per page (number),  
                         'page': Page number 
                    }                                   Optional query parameters 
            
        Returns:
            Parsed JSON response as a dictionary
            
        Raises:
            ValueError: If the request fails or returns an error status code
        """
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.HTTPError as e:
            if response.status_code == 404:
                raise ValueError(f"Record not found: {endpoint}")
            elif response.status_code == 429:
                raise ValueError("Rate limit exceeded. Please wait 5 seconds.")
            else:
                raise ValueError(f"HTTP error: {e}")
        
        except requests.exceptions.Timeout:
            raise ValueError("Request timed out. Please try again.")
        
        except requests.exceptions.ConnectionError:
            raise ValueError("Could not connect to INSPIRE HEP.")
    
    def search_literature(self, query: str, sort: str = "mostrecent") -> str:
        """Search for literature records on INSPIRE HEP.
        
        Args:
            query: Search query string (e.g., 'quantum field theory', 
                'author:Witten', 'topcite 1000+')
            sort: Sort order. Option: 'mostrecent' (default) for newest papers,
             'mostcited' for most cited papers    
                
        Returns:
            Formatted string with paper titles and citation counts, 
            separated by '---'. Returns "No results found." if no matches.
            
        Example:
            >>> wrapper.search_literature("string theory")
            'Title: String Theory Paper\\nCitations: 500\\n---\\n...'
        """
        
        try:
           
            params = {'q': query, 'size': self.top_k_results, 'sort': sort}
            data = self._make_request("literature", params)
            hits = data['hits']['hits']
        
            if not hits:
                return "No results found."
            
            results = []
            for hit in hits:
                metadata = hit['metadata']
                title = metadata['titles'][0]['title']
                citations = metadata.get('citation_count', 0)
                results.append(f"Title: {title}\nCitations: {citations}")
            
            return "\n---\n".join(results)
        
        except ValueError as e:
            return f"Error: {e}"
    
    def get_author_papers(self, author_name: str, sort: str = "mostcited") -> str:
        """Get the most cited papers by a specific author by default. 
        The sort order can be modified in the input.
        
        Args:
            author_name: Author's INSPIRE identifier (e.g., 'E.Witten.1').
                Find identifiers at https://inspirehep.net/authors
                
        Returns:
            Formatted string listing papers with citation counts.
            Returns error message if author not found.
            
        Note:
            This method requires INSPIRE author identifiers, not plain names.
            Plain names like "Edward Witten" may not work reliably due to 
            ambiguity. Use the INSPIRE website to find the correct identifier.
            
        Example:
            >>> wrapper.get_author_papers("E.Witten.1")
            'Top papers by E.Witten.1:\\n1. AdS/CFT (5000 citations)\\n...'
        """
        try:
            params = {
                'q': f'a {author_name}',
                'size': self.top_k_results,
                'sort': sort
            }
            data = self._make_request("literature", params)
            hits = data['hits']['hits']
            
            if not hits:
                return f"No papers found for author: {author_name}"
            
            results = [f"Top papers by {author_name}:"]
            for i, hit in enumerate(hits, 1):
                metadata = hit['metadata']
                title = metadata['titles'][0]['title']
                
                # Clean up math formatting
                title = re.sub(r'<math[^>]*>(.*?)</math>', r'\1', title)  # Extract content from <math> tags
                title = re.sub(r'\$([^\$]+)\$', r'\1', title)  # Extract content from $ $ tags
            
                
                citations = metadata.get('citation_count', 0)
                results.append(f"{i}. {title} ({citations} citations)")
            
            return "\n".join(results)
        
        except ValueError as e:
            return f"Error: {e}"
    
    def get_paper_details(self, record_id: str) -> str:
        """Get detailed information about a specific paper.
        
        Args:
            record_id: INSPIRE record ID (e.g., '451647' for Maldacena's 
                famous AdS/CFT paper)
                
        Returns:
            Formatted string with title, authors, citations, and abstract.
            Returns error message if record not found.
            
        Example:
            >>> wrapper.get_paper_details("451647")
            'Title: The Large N limit...\\nAuthors: Maldacena, Juan...\\n...'
        """
        try:
            data = self._make_request(f"literature/{record_id}")
            metadata = data['metadata']
            
            title = metadata['titles'][0]['title']
            citations = metadata.get('citation_count', 0)
            authors = ', '.join(
                [a['full_name'] for a in metadata.get('authors', [])[:3]]
            )
            abstract = metadata.get('abstracts', [{}])[0].get('value', 'No abstract')[:300]
            
            return (
                f"Title: {title}\n"
                f"Authors: {authors}\n"
                f"Citations: {citations}\n"
                f"Abstract: {abstract}..."
            )
        
        except ValueError as e:
            return f"Error: {e}"