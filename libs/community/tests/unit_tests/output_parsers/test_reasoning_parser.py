import pytest
from pydantic import BaseModel, Field

from langchain_community.output_parsers.reasoning_parser import (
    ReasoningJsonOutputParser,
    ReasoningStructuredOutputParser,
    strip_think_tags,
)


def test_strip_think_tags() -> None:
    """Test that strip_think_tags removes the content between <think> tags."""
    # Test basic case
    text = "Hello <think>This is my reasoning</think> World"
    result = strip_think_tags(text)
    assert result == "Hello World"

    # Test with multiple think blocks
    text = (
        "<think>Thinking step 1</think>Output 1<think>Thinking step 2</think>Output 2"
    )
    result = strip_think_tags(text)
    assert result == "Output 1Output 2"

    # Test with multiline think blocks
    text = """I'm going to analyze this.
    <think>
    Step 1: Consider the problem
    Step 2: Evaluate options
    Step 3: Decide on approach
    </think>
    The answer is 42."""
    result = strip_think_tags(text)
    assert result == "I'm going to analyze this.\n    The answer is 42."

    # Test with no think tags
    text = "Plain text with no think tags"
    result = strip_think_tags(text)
    assert result == "Plain text with no think tags"


def test_reasoning_json_output_parser() -> None:
    """Test that ReasoningJsonOutputParser correctly parses JSON after stripping
    think tags."""
    parser = ReasoningJsonOutputParser()

    # Test with thinking and valid JSON
    text = """<think>
    Let me think about the structure here. I need to create a valid JSON object.
    The user asked for name, age, and occupation.
    </think>
    {
        "name": "John Doe",
        "age": 30,
        "occupation": "Software Engineer"
    }"""

    result = parser.parse(text)
    assert result == {"name": "John Doe", "age": 30, "occupation": "Software Engineer"}

    # Test with thinking at the end
    text = """{
        "name": "Jane Smith",
        "age": 28,
        "occupation": "Data Scientist"
    }
    <think>This looks good. I've provided all the required fields.</think>"""

    result = parser.parse(text)
    assert result == {"name": "Jane Smith", "age": 28, "occupation": "Data Scientist"}

    # Test with multiple thinking blocks
    text = """<think>First, I'll prepare the basic structure</think>
    {
        <think>Now I need to add the fields</think>
        "name": "Alex Johnson",
        "age": 35,
        <think>What occupation should I use?</think>
        "occupation": "Doctor"
    }"""

    result = parser.parse(text)
    assert result == {"name": "Alex Johnson", "age": 35, "occupation": "Doctor"}


def test_reasoning_json_output_parser_invalid_json() -> None:
    """Test that ReasoningJsonOutputParser raises an error for invalid JSON."""
    parser = ReasoningJsonOutputParser()

    # Test with thinking and invalid JSON
    text = """<think>
    I need to create a JSON object but I'm going to make a mistake on purpose.
    </think>
    {
        "name": "John Doe",
        "age": 30,
        occupation: "Software Engineer"  # Missing quotes around key
    }"""

    with pytest.raises(Exception):
        parser.parse(text)


class Person(BaseModel):
    """Test model for structured output parser."""

    name: str = Field(description="Person's name")
    age: int = Field(description="Person's age")
    occupation: str = Field(description="Person's occupation")


def test_reasoning_structured_output_parser() -> None:
    """Test that ReasoningStructuredOutputParser correctly parses structured data
    after stripping think tags."""
    parser = ReasoningStructuredOutputParser(pydantic_object=Person)

    # Test with thinking and valid structured data
    text = """<think>
    Let me think about the structure here. I need to create a valid Pydantic object.
    The model requires name, age, and occupation.
    </think>
    {
        "name": "John Doe",
        "age": 30,
        "occupation": "Software Engineer"
    }"""

    result = parser.parse(text)
    assert isinstance(result, Person)
    assert result.name == "John Doe"
    assert result.age == 30
    assert result.occupation == "Software Engineer"

    # Test with thinking at the end
    text = """{
        "name": "Jane Smith",
        "age": 28,
        "occupation": "Data Scientist"
    }
    <think>This looks good. I've provided all the required fields.</think>"""

    result = parser.parse(text)
    assert isinstance(result, Person)
    assert result.name == "Jane Smith"
    assert result.age == 28
    assert result.occupation == "Data Scientist"


def test_reasoning_structured_output_parser_invalid_data() -> None:
    """Test that ReasoningStructuredOutputParser raises an error for invalid data."""
    parser = ReasoningStructuredOutputParser(pydantic_object=Person)

    # Test with thinking and invalid structured data (missing required field)
    text = """<think>
    I need to create a Pydantic object but I'm going to make a mistake on purpose.
    </think>
    {
        "name": "John Doe",
        "age": "thirty"
    }"""

    with pytest.raises(Exception):
        parser.parse(text)
