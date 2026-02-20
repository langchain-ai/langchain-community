"""INSPIRE HEP tools for searching physics literature.

INSPIRE (https://inspirehep.net) is a trusted community hub for high energy 
physics research. These tools provide programmatic access to search papers,
retrieve author publications, and get detailed paper information.

## Quick Start
```python
from langchain_community.tools.inspire_hep import (
    INSPIRESearchLiteratureTool,
    INSPIREGetAuthorPapersTool,
)

# Search for papers
search_tool = INSPIRESearchLiteratureTool()
result = search_tool.invoke({"query": "quantum gravity"})

# Get author's papers (requires INSPIRE identifier)
author_tool = INSPIREGetAuthorPapersTool()
result = author_tool.invoke({"author_name": "E.Witten.1"})
```

## Using with Agents
```python
from langchain.agents import create_react_agent
from langchain_groq import ChatGroq

tools = [
    INSPIRESearchLiteratureTool(),
    INSPIREGetAuthorPapersTool(),
]

llm = ChatGroq(model="llama-3.3-70b-versatile")
agent = create_react_agent(llm, tools)
```

## Note on Author Identifiers

The author papers tool requires INSPIRE author identifiers (e.g., 'E.Witten.1'),
not plain names. Find identifiers at https://inspirehep.net/authors.
"""

from langchain_community.tools.inspire_hep.tool import (
    INSPIREGetAuthorPapersTool,
    INSPIREGetPaperDetailsTool,
    INSPIRESearchLiteratureTool,
)

__all__ = [
    "INSPIRESearchLiteratureTool",
    "INSPIREGetAuthorPapersTool",
    "INSPIREGetPaperDetailsTool",
]