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

    Returns:
        The text with think tags removed and whitespace stripped.
    """
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


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
