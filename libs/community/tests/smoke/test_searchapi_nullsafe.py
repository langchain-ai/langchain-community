"""Smoke tests for SearchApiAPIWrapper null-safe parsing.

These tests do NOT call the real SearchApi API.
They only exercise the internal `_result_as_string` helper with
synthetic responses to ensure we don't crash on missing fields.
"""

from typing import Any, Dict

from langchain_community.utilities.searchapi import SearchApiAPIWrapper


def _call(result: Dict[str, Any]) -> str:
    # Convenience wrapper around the static method
    return SearchApiAPIWrapper._result_as_string(result)


def test_knowledge_graph_missing_description_returns_default() -> None:
    result = {
        "knowledge_graph": {
            # no "description" field
            "title": "Vilnius",
        }
    }

    output = _call(result)
    assert output == "No good search result found"


def test_organic_results_mixed_valid_and_invalid_items() -> None:
    result = {
        "organic_results": [
            {"snippet": "First valid snippet."},
            {"title": "No snippet here"},  # missing snippet -> should be skipped
            "not-a-dict",  # completely invalid -> should be skipped
            {"snippet": ""},  # empty string -> should be skipped
            {"snippet": "Second valid snippet."},
        ]
    }

    output = _call(result)
    # Only the two valid snippets should appear, joined by newline
    assert output == "First valid snippet.\nSecond valid snippet."


def test_jobs_missing_description_does_not_crash() -> None:
    result = {
        "jobs": [
            {"title": "AI Engineer"},  # missing description
            {"description": "Looking for an AI engineer."},
        ]
    }

    output = _call(result)
    # Should return only the description from the second item
    assert output == "Looking for an AI engineer."


def test_videos_missing_link_does_not_crash() -> None:
    result = {
        "videos": [
            {"title": "Video without link"},
            {"title": "Video with link", "link": "https://example.com/video"},
        ]
    }

    output = _call(result)
    # Only the fully valid video (title + link) should be included
    assert 'Title: "Video with link" Link: https://example.com/video' in output
    assert "Video without link" not in output


def test_images_missing_original_link_does_not_crash() -> None:
    result = {
        "images": [
            {"title": "Image without original"},
            {"title": "Image with bad original", "original": {}},
            {
                "title": "Image with link",
                "original": {"link": "https://example.com/image"},
            },
        ]
    }

    output = _call(result)
    # Only the fully valid image should be included
    assert 'Title: "Image with link" Link: https://example.com/image' in output
    assert "Image without original" not in output
    assert "Image with bad original" not in output


def test_non_dict_or_empty_result_returns_default() -> None:
    assert _call({}) == "No good search result found"
    # type: ignore[arg-type] – deliberate invalid input
    assert _call(None) == "No good search result found"  # type: ignore[arg-type]
