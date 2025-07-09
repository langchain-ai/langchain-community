"""Google Vertex AI MARS7 Text-to-Speech Tool.

This module provides a LangChain tool for converting text to speech using
Google Cloud's Vertex AI platform with the MARS7 model from CambAI.

The tool supports voice cloning with reference audio and multilingual synthesis.

Example:
    >>> from langchain_community.tools.google_vertex_camb import GoogleVertexCambTool
    >>> tool = GoogleVertexCambTool()
    >>> audio_file = tool.invoke("Hello world!")
"""

from langchain_community.tools.google_vertex_camb.tool import GoogleVertexCambTool

__all__ = ["GoogleVertexCambTool"]
