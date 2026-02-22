"""Tests for the Grobid parser."""

from typing import List
from unittest.mock import patch

import pytest

from langchain_community.document_loaders.parsers.grobid import GrobidParser

# Minimal TEI XML with authors and publication date in the header.
SAMPLE_TEI_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <titleStmt>
        <title>Sample Paper Title</title>
      </titleStmt>
      <sourceDesc>
        <biblStruct>
          <analytic>
            <author>
              <persName>
                <forename type="first">Alice</forename>
                <surname>Smith</surname>
              </persName>
            </author>
            <author>
              <persName>
                <forename type="first">Bob</forename>
                <surname>Jones</surname>
              </persName>
            </author>
          </analytic>
          <monogr>
            <imprint>
              <date type="published" when="2023-06-15"/>
            </imprint>
          </monogr>
        </biblStruct>
      </sourceDesc>
    </fileDesc>
  </teiHeader>
  <text>
    <body>
      <div>
        <head n="1">Introduction</head>
        <p>
          <s coords="1,100,200,50,10">First sentence.</s>
          <s coords="1,100,220,50,10">Second sentence.</s>
        </p>
      </div>
    </body>
  </text>
</TEI>
"""

# TEI XML with no authors or date in the header.
SAMPLE_TEI_XML_NO_HEADER_META = """\
<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <titleStmt><title>Untitled</title></titleStmt>
    </fileDesc>
  </teiHeader>
  <text>
    <body>
      <div>
        <head n="1">Section</head>
        <p>
          <s coords="1,10,20,30,5">Only sentence.</s>
        </p>
      </div>
    </body>
  </text>
</TEI>
"""

# TEI XML with multiple sections for testing metadata propagation.
SAMPLE_TEI_XML_MULTI_SECTION = """\
<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <titleStmt><title>Multi-Section Paper</title></titleStmt>
      <sourceDesc>
        <biblStruct>
          <analytic>
            <author>
              <persName>
                <forename type="first">Carol</forename>
                <surname>White</surname>
              </persName>
            </author>
          </analytic>
          <monogr>
            <imprint>
              <date type="published" when="2020"/>
            </imprint>
          </monogr>
        </biblStruct>
      </sourceDesc>
    </fileDesc>
  </teiHeader>
  <text>
    <body>
      <div>
        <head n="1">Introduction</head>
        <p><s coords="1,10,20,30,5">Intro sentence.</s></p>
      </div>
      <div>
        <head n="2">Methods</head>
        <p><s coords="2,10,20,30,5">Methods sentence.</s></p>
      </div>
    </body>
  </text>
</TEI>
"""


def _make_parser() -> GrobidParser:
    """Create a GrobidParser without connecting to a server."""
    with patch("requests.get"):
        return GrobidParser(segment_sentences=False)


@pytest.mark.requires("bs4", "lxml")
class TestGrobidProcessXml:
    """Tests for GrobidParser.process_xml header metadata extraction."""

    def test_authors_and_year_extracted(self) -> None:
        """Authors and year from the TEI header appear in metadata."""
        parser = _make_parser()
        docs = list(parser.process_xml("paper.pdf", SAMPLE_TEI_XML, False))

        assert len(docs) == 1
        assert docs[0].metadata["authors"] == "Alice Smith, Bob Jones"
        assert docs[0].metadata["year"] == "2023"

    def test_segment_sentences_preserves_header_metadata(self) -> None:
        """Each sentence document gets authors and year when segmenting."""
        parser = _make_parser()
        docs = list(parser.process_xml("paper.pdf", SAMPLE_TEI_XML, True))

        assert len(docs) == 2
        for doc in docs:
            assert doc.metadata["authors"] == "Alice Smith, Bob Jones"
            assert doc.metadata["year"] == "2023"

    def test_missing_authors_and_year(self) -> None:
        """When no authors or date exist in the header, values are None."""
        parser = _make_parser()
        docs = list(
            parser.process_xml("paper.pdf", SAMPLE_TEI_XML_NO_HEADER_META, False)
        )

        assert len(docs) == 1
        assert docs[0].metadata["authors"] == ""
        assert docs[0].metadata["year"] == ""

    def test_header_metadata_on_all_sections(self) -> None:
        """Authors and year propagate to documents from every section."""
        parser = _make_parser()
        docs = list(
            parser.process_xml("paper.pdf", SAMPLE_TEI_XML_MULTI_SECTION, False)
        )

        assert len(docs) == 2
        for doc in docs:
            assert doc.metadata["authors"] == "Carol White"
            assert doc.metadata["year"] == "2020"

    def test_existing_metadata_preserved(self) -> None:
        """Pre-existing metadata fields are not affected by the change."""
        parser = _make_parser()
        docs = list(parser.process_xml("paper.pdf", SAMPLE_TEI_XML, False))

        meta = docs[0].metadata
        assert meta["paper_title"] == "Sample Paper Title"
        assert meta["file_path"] == "paper.pdf"
        assert meta["section_title"] == "Introduction"
        assert meta["section_number"] == "1"
        assert "text" in meta
        assert "para" in meta
        assert "bboxes" in meta
        assert "pages" in meta

    def test_year_only_date(self) -> None:
        """A date with only a year (no month/day) is handled correctly."""
        parser = _make_parser()
        docs = list(
            parser.process_xml("paper.pdf", SAMPLE_TEI_XML_MULTI_SECTION, False)
        )

        # The XML has when="2020" (year only, no dashes)
        assert docs[0].metadata["year"] == "2020"

    def test_author_with_surname_only(self) -> None:
        """An author with only a surname (no forename) is still captured."""
        xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <sourceDesc>
        <biblStruct>
          <analytic>
            <author>
              <persName><surname>Darwin</surname></persName>
            </author>
          </analytic>
        </biblStruct>
      </sourceDesc>
    </fileDesc>
  </teiHeader>
  <text><body>
    <div>
      <head n="1">Intro</head>
      <p><s coords="1,10,20,30,5">A sentence.</s></p>
    </div>
  </body></text>
</TEI>
"""
        parser = _make_parser()
        docs = list(parser.process_xml("paper.pdf", xml, False))

        assert docs[0].metadata["authors"] == "Darwin"
