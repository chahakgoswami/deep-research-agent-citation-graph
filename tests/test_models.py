"""Unit tests for core data models."""

from __future__ import annotations

import pytest
from datetime import datetime
from uuid import UUID

from research_agent.models import CitationNode, Hop, ResearchQuery, Source


# ---------------------------------------------------------------------------
# ResearchQuery tests
# ---------------------------------------------------------------------------


class TestResearchQuery:
    def test_basic_creation(self):
        q = ResearchQuery(text="What is quantum entanglement?")
        assert q.text == "What is quantum entanglement?"
        assert q.max_hops == 3
        assert isinstance(q.id, str)
        # id should be a valid UUID
        UUID(q.id)

    def test_custom_max_hops(self):
        q = ResearchQuery(text="AI safety", max_hops=5)
        assert q.max_hops == 5

    def test_text_is_stripped(self):
        q = ResearchQuery(text="  leading spaces  ")
        assert q.text == "leading spaces"

    def test_blank_text_raises(self):
        with pytest.raises(Exception):
            ResearchQuery(text="   ")

    def test_empty_text_raises(self):
        with pytest.raises(Exception):
            ResearchQuery(text="")

    def test_max_hops_lower_bound(self):
        with pytest.raises(Exception):
            ResearchQuery(text="test", max_hops=0)

    def test_max_hops_upper_bound(self):
        with pytest.raises(Exception):
            ResearchQuery(text="test", max_hops=11)

    def test_max_hops_boundary_valid(self):
        q1 = ResearchQuery(text="test", max_hops=1)
        q2 = ResearchQuery(text="test", max_hops=10)
        assert q1.max_hops == 1
        assert q2.max_hops == 10

    def test_unique_ids(self):
        q1 = ResearchQuery(text="alpha")
        q2 = ResearchQuery(text="beta")
        assert q1.id != q2.id

    def test_created_at_is_datetime(self):
        q = ResearchQuery(text="test")
        assert isinstance(q.created_at, datetime)

    def test_metadata_defaults_to_empty_dict(self):
        q = ResearchQuery(text="test")
        assert q.metadata == {}

    def test_metadata_custom(self):
        q = ResearchQuery(text="test", metadata={"user": "alice"})
        assert q.metadata["user"] == "alice"


# ---------------------------------------------------------------------------
# Source tests
# ---------------------------------------------------------------------------


class TestSource:
    def test_basic_creation(self):
        s = Source(url="https://example.com/page")
        assert s.url == "https://example.com/page"
        assert s.title == ""
        assert s.snippet == ""

    def test_full_creation(self):
        s = Source(
            url="https://nasa.gov/climate",
            title="NASA Climate",
            snippet="Rising temperatures...",
            domain="nasa.gov",
            domain_authority=92.0,
            citation_count=4000,
        )
        assert s.domain == "nasa.gov"
        assert s.domain_authority == 92.0
        assert s.citation_count == 4000

    def test_domain_authority_bounds(self):
        with pytest.raises(Exception):
            Source(url="https://example.com", domain_authority=101.0)
        with pytest.raises(Exception):
            Source(url="https://example.com", domain_authority=-1.0)

    def test_citation_count_non_negative(self):
        with pytest.raises(Exception):
            Source(url="https://example.com", citation_count=-1)

    def test_published_at_optional(self):
        s = Source(url="https://example.com")
        assert s.published_at is None

    def test_raw_metadata_defaults_empty(self):
        s = Source(url="https://example.com")
        assert s.raw_metadata == {}

    def test_domain_extraction_from_url(self):
        """When domain is not provided, it should be auto-extracted from url."""
        s = Source(url="https://openai.com/research/gpt4", domain="")
        # Domain should be extracted
        assert s.domain == "openai.com"


# ---------------------------------------------------------------------------
# Hop tests
# ---------------------------------------------------------------------------


class TestHop:
    def test_basic_creation(self):
        h = Hop(hop_index=0, sub_query="What causes climate change?", parent_query_id="abc-123")
        assert h.hop_index == 0
        assert h.sub_query == "What causes climate change?"
        assert h.sources == []

    def test_negative_hop_index_raises(self):
        with pytest.raises(Exception):
            Hop(hop_index=-1, sub_query="test", parent_query_id="id")

    def test_sources_appended(self):
        s = Source(url="https://example.com")
        h = Hop(hop_index=1, sub_query="sub", parent_query_id="id", sources=[s])
        assert len(h.sources) == 1
        assert h.sources[0].url == "https://example.com"

    def test_timestamp_is_datetime(self):
        h = Hop(hop_index=0, sub_query="test", parent_query_id="id")
        assert isinstance(h.timestamp, datetime)

    def test_multiple_sources(self):
        sources = [Source(url=f"https://example.com/{i}") for i in range(3)]
        h = Hop(hop_index=2, sub_query="test", parent_query_id="id", sources=sources)
        assert len(h.sources) == 3


# ---------------------------------------------------------------------------
# CitationNode tests
# ---------------------------------------------------------------------------


class TestCitationNode:
    def _make_source(self, url: str = "https://example.com") -> Source:
        return Source(url=url, title="Test", snippet="A snippet.")

    def test_basic_creation(self):
        s = self._make_source()
        node = CitationNode(source=s, hop_index=0)
        assert node.hop_index == 0
        assert node.source.url == "https://example.com"
        assert node.grade is None
        assert node.confidence is None

    def test_unique_ids(self):
        s = self._make_source()
        n1 = CitationNode(source=s, hop_index=0)
        n2 = CitationNode(source=s, hop_index=0)
        assert n1.id != n2.id

    def test_grade_assignment(self):
        s = self._make_source()
        node = CitationNode(source=s, hop_index=1, grade="A", confidence=0.95)
        assert node.grade == "A"
        assert node.confidence == 0.95

    def test_confidence_bounds(self):
        s = self._make_source()
        with pytest.raises(Exception):
            CitationNode(source=s, hop_index=0, confidence=1.5)
        with pytest.raises(Exception):
            CitationNode(source=s, hop_index=0, confidence=-0.1)

    def test_contradicts_and_supports_default_empty(self):
        s = self._make_source()
        node = CitationNode(source=s, hop_index=0)
        assert node.contradicts == []
        assert node.supports == []

    def test_contradicts_and_supports_populated(self):
        s = self._make_source()
        node = CitationNode(
            source=s,
            hop_index=0,
            contradicts=["node-id-1"],
            supports=["node-id-2", "node-id-3"],
        )
        assert "node-id-1" in node.contradicts
        assert len(node.supports) == 2

    def test_claim_snippet_default_empty(self):
        s = self._make_source()
        node = CitationNode(source=s, hop_index=0)
        assert node.claim_snippet == ""

    def test_negative_hop_index_raises(self):
        s = self._make_source()
        with pytest.raises(Exception):
            CitationNode(source=s, hop_index=-1)
