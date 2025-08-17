import re
from langchain_core.output_parsers import JsonOutputParser, StructuredOutputParser



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
    
    Args:
        text: The text to parse, which may contain <think> reasoning tags.
    """
    async def parse(self, text: str):
        cleaned = strip_think_tags(text)
        return await super().parse(cleaned)


class ReasoningStructuredOutputParser(StructuredOutputParser):
    """A structured output parser that strips reasoning tags before parsing.
    
    This parser removes any content enclosed in <think> tags from the input text
    before delegating to the parent StructuredOutputParser for structured parsing.
    
    Args:
        text: The text to parse, which may contain <think> reasoning tags.
    """
    async def parse(self, text: str):
        cleaned = strip_think_tags(text)
        return await super().parse(cleaned)

