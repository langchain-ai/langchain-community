import re
from langchain_core.output_parsers import JsonOutputParser, StructuredOutputParser


def strip_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


class ReasoningJsonOutputParser(JsonOutputParser):
    async def parse(self, text: str):
        cleaned = strip_think_tags(text)
        return await super().parse(cleaned)


class ReasoningStructuredOutputParser(StructuredOutputParser):
    async def parse(self, text: str):
        cleaned = strip_think_tags(text)
        return await super().parse(cleaned)
