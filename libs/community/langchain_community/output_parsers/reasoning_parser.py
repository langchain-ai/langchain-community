import re
from typing import Any
from langchain_core.utils.pydantic import (
    TBaseModel,
)
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser


def strip_think_tags(text: str) -> str:
    """Removes <think>...</think> tags from text.

    Args:
        text: The input text that may contain think tags.
    """
    def remove_think_tags(text: str) -> str:
        """Remove content between <think> and </think> tags more safely."""
        result = []
        i = 0
        while i < len(text):
            # Look for opening tag
            open_tag_pos = text.find("<think>", i)
            if open_tag_pos == -1:
                # No more opening tags, add the rest and break
                result.append(text[i:])
                break
            
            # Add text before the opening tag
            result.append(text[i:open_tag_pos])
            
            # Look for closing tag
            close_tag_pos = text.find("</think>", open_tag_pos + 7)
            if close_tag_pos == -1:
                # No closing tag found, treat opening tag as literal text
                result.append("<think>")
                i = open_tag_pos + 7
            else:
                # Skip the content between tags and move past closing tag
                i = close_tag_pos + 9
        
        return "".join(result).strip()
    
    return remove_think_tags(text)


class ReasoningJsonOutputParser(JsonOutputParser):
    """A JSON output parser that strips reasoning tags before parsing.

    This parser removes any content enclosed in <think> tags from the input text
    before delegating to the parent JsonOutputParser for JSON parsing.

    """

    def parse(self, text: str) -> Any:
        """Parse text by first removing <think> tags.

        Args:
            text: Text to parse that may contain <think> tags

        Returns:
            Parsed output after removing reasoning blocks
        """
        cleaned = strip_think_tags(text)
        return super().parse(cleaned)



class ReasoningStructuredOutputParser(PydanticOutputParser):
    """A structured output parser that strips reasoning tags before parsing.

    This parser removes any content enclosed in <think> tags from the input text
    before delegating to the parent PydanticOutputParser for structured parsing.
    """


    def parse(self, text: str) -> TBaseModel:
        """Parse text by first removing <think> tags.

        Args:
            text: Text to parse that may contain <think> tags

        Returns:
            Parsed output after removing reasoning blocks
        """
        cleaned = strip_think_tags(text)
        return super().parse(cleaned)
