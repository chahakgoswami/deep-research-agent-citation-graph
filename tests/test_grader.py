"""Tests for the SourceGrader module."""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

from research_agent.grader import (
    SourceGrader,
    _CITATION_COUNT_CEILING,
    _RECENCY_MAX_AGE_DAYS,
    _WEIGHT_CITATION,
    _WEIGHT_DOMAIN_AUTHORITY,
    _WEIGHT_RECENCY,
    _CONFIDENCE_PENALTY_PER_MISSING_SIGNAL,
)
from research_agent.models import CitationNode, Source


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REFERENCE_DATE = datetime(2024, 6, 1, tzinfo=timezone.utc)


def make_grader(ref: datetime = REFERENCE_DATE) -> SourceGrader:
    return SourceGrader(reference_date=ref)


def make_source(
    url: str = "https://example.com/article",
    domain_authority: float | None = 80.0,
    citation_count: int | None = 1000,
    published_at: datetime | None = datetime(2023, 6, 1, tzinfo=timezone.utc),
    **kwargs,
) -> Source:
    return Source(
        url=url,
        title=kwargs.get("title", "Test Article"),
        snippet=kwargs.get("snippet", "A test snippet."),
        domain=kwargs.get("domain", "example.com"),
        domain_authority=domain_authority,
        citation_count=citation_count,
        published_at=published_at,
    )


def make_node(
    url: str = "https://example.com/article",
    hop_index: int = 0,
    **source_kwargs,
) -> CitationNode:
    return CitationNode(source=make_source(url=url, **source_kwargs), hop_index=hop_index)


# ---------------------------------------------------------------------------
# SourceGrader.grade — basic behaviour
# ---------------------------------------------------------------------------


class TestSourceGraderGradeBasic:
    def test_grade_returns_same_node(self):
        grader = make_grader()
        node = make_node()
        result = grader.grade(node)
        assert result is node

    def test_grade_populates_grade_field(self):
        grader = make_grader()
        node = make_node()
        grader.grade(node)
        assert node.grade is not None
        assert node.grade in ("A", "B", "C", "D", "F")

    def test_grade_populates_confidence_field(self):
        grader = make_grader()
        node = make_node()
        grader.grade(node)
        assert node.confidence is not None
        assert 0.0 <= node.confidence <= 1.0

    def test_full_signals_give_max_confidence(self):
        """All three signals present → confidence == 1.0."""
        grader = make_grader()
        node = make_node(
            domain_authority=90.0,
            citation_count=5000,
            published_at=datetime(2023, 12, 1, tzinfo=timezone.utc),
        )
        grader.grade(node)
        assert node.confidence == 1.0

    def test_grade_a_for_excellent_source(self):
        """High DA, recent, many citations → grade A."""
        grader = make_grader()
        node = make_node(
            domain_authority=95.0,
            citation_count=9000,
            # Published 30 days before reference date
            published_at=REFERENCE_DATE - timedelta(days=30),
        )
        grader.grade(node)
        assert node.grade == "A"

    def test_grade_f_for_poor_source(self):
        """Very low DA, ancient, no citations → grade F."""
        grader = make_grader()
        node = make_node(
            domain_authority=5.0,
            citation_count=0,
            # Published 6 years ago (beyond the max-age ceiling)
            published_at=REFERENCE_DATE - timedelta(days=365 * 6),
        )
        grader.grade(node)
        assert node.grade == "F"


# ---------------------------------------------------------------------------
# Sub-scorer: domain authority
# ---------------------------------------------------------------------------


class TestDomainAuthorityScore:
    def test_high_da_contributes_high_score(self):
        grader = make_grader()
        score, present = grader._domain_authority_score(
            make_source(domain_authority=100.0)
        )
        assert present is True
        assert score == 100.0

    def test_zero_da_contributes_zero(self):
        grader = make_grader()
        score, present = grader._domain_authority_score(
            make_source(domain_authority=0.0)
        )
        assert present is True
        assert score == 0.0

    def test_missing_da_returns_neutral_and_absent(self):
        grader = make_grader()
        score, present = grader._domain_authority_score(
            make_source(domain_authority=None)
        )
        assert present is False
        assert score == 50.0  # neutral default


# ---------------------------------------------------------------------------
# Sub-scorer: recency
# ---------------------------------------------------------------------------


class TestRecencyScore:
    def test_very_recent_source_scores_high(self):
        grader = make_grader()
        # Published yesterday
        pub = REFERENCE_DATE - timedelta(days=1)
        score, present = grader._recency_score(make_source(published_at=pub))
        assert present is True
        assert score > 99.0

    def test_old_source_beyond_ceiling_scores_zero(self):
        grader = make_grader()
        pub = REFERENCE_DATE - timedelta(days=_RECENCY_MAX_AGE_DAYS + 100)
        score, present = grader._recency_score(make_source(published_at=pub))
        assert present is True
        assert score == 0.0

    def test_future_dated_source_treated_as_maximally_recent(self):
        grader = make_grader()
        pub = REFERENCE_DATE + timedelta(days=30)
        score, present = grader._recency_score(make_source(published_at=pub))
        assert present is True
        assert score == 100.0

    def test_missing_published_at_returns_neutral_and_absent(self):
        grader = make_grader()
        score, present = grader._recency_score(make_source(published_at=None))
        assert present is False
        assert score == 50.0

    def test_naive_datetime_handled(self):
        """Naive published_at should not raise."""
        grader = make_grader()
        pub = datetime(2023, 1, 1)  # no tzinfo
        score, present = grader._recency_score(make_source(published_at=pub))
        assert present is True
        assert 0.0 <= score <= 100.0


# ---------------------------------------------------------------------------
# Sub-scorer: citation count
# ---------------------------------------------------------------------------


class TestCitationScore:
    def test_zero_citations_scores_zero(self):
        grader = make_grader()
        score, present = grader._citation_score(make_source(citation_count=0))
        assert present is True
        assert score == 0.0

    def test_ceiling_citations_scores_100(self):
        grader = make_grader()
        score, present = grader._citation_score(
            make_source(citation_count=_CITATION_COUNT_CEILING)
        )
        assert present is True
        assert score == 100.0

    def test_above_ceiling_capped_at_100(self):
        grader = make_grader()
        score, present = grader._citation_score(
            make_source(citation_count=_CITATION_COUNT_CEILING * 2)
        )
        assert present is True
        assert score == 100.0

    def test_missing_citation_count_returns_neutral_and_absent(self):
        grader = make_grader()
        score, present = grader._citation_score(make_source(citation_count=None))
        assert present is False
        assert score == 50.0


# ---------------------------------------------------------------------------
# Composite score → letter grade mapping
# ---------------------------------------------------------------------------


class TestCompositeToGrade:
    @pytest.mark.parametrize(
        "composite, expected_grade",
        [
            (100.0, "A"),
            (85.0, "A"),
            (84.9, "B"),
            (70.0, "B"),
            (69.9, "C"),
            (55.0, "C"),
            (54.9, "D"),
            (40.0, "D"),
            (39.9, "F"),
            (0.0, "F"),
        ],
    )
    def test_threshold_boundaries(self, composite: float, expected_grade: str):
        assert SourceGrader._composite_to_grade(composite) == expected_grade


# ---------------------------------------------------------------------------
# Missing metadata edge cases
# ---------------------------------------------------------------------------


class TestMissingMetadataEdgeCases:
    def test_all_signals_missing_gives_lowest_confidence(self):
        grader = make_grader()
        node = make_node(
            domain_authority=None,
            citation_count=None,
            published_at=None,
        )
        grader.grade(node)
        expected_confidence = max(
            0.0, 1.0 - 3 * _CONFIDENCE_PENALTY_PER_MISSING_SIGNAL
        )
        assert node.confidence == pytest.approx(expected_confidence, abs=1e-4)

    def test_one_missing_signal_reduces_confidence(self):
        grader = make_grader()
        node = make_node(
            domain_authority=None,  # missing
            citation_count=1000,
            published_at=datetime(2023, 6, 1, tzinfo=timezone.utc),
        )
        grader.grade(node)
        expected = 1.0 - 1 * _CONFIDENCE_PENALTY_PER_MISSING_SIGNAL
        assert node.confidence == pytest.approx(expected, abs=1e-4)

    def test_two_missing_signals_reduce_confidence(self):
        grader = make_grader()
        node = make_node(
            domain_authority=None,
            citation_count=None,
            published_at=datetime(2023, 6, 1, tzinfo=timezone.utc),
        )
        grader.grade(node)
        expected = 1.0 - 2 * _CONFIDENCE_PENALTY_PER_MISSING_SIGNAL
        assert node.confidence == pytest.approx(expected, abs=1e-4)

    def test_all_missing_still_assigns_grade(self):
        """Even with all signals missing, a grade must be assigned."""
        grader = make_grader()
        node = make_node(
            domain_authority=None,
            citation_count=None,
            published_at=None,
        )
        grader.grade(node)
        assert node.grade in ("A", "B", "C", "D", "F")

    def test_source_with_null_url_still_grades(self):
        """A source with minimal fields should not raise."""
        source = Source(
            url="https://minimal.example.com",
            domain_authority=None,
            citation_count=None,
            published_at=None,
        )
        node = CitationNode(source=source, hop_index=0)
        grader = make_grader()
        grader.grade(node)
        assert node.grade is not None
        assert node.confidence is not None


# ---------------------------------------------------------------------------
# Duplicate URL handling via grade_all
# ---------------------------------------------------------------------------


class TestGradeAllDeduplicate:
    def test_grade_all_returns_same_list(self):
        grader = make_grader()
        nodes = [make_node(url=f"https://example.com/{i}") for i in range(3)]
        result = grader.grade_all(nodes)
        assert result is nodes

    def test_grade_all_grades_every_node(self):
        grader = make_grader()
        nodes = [make_node(url=f"https://example.com/{i}") for i in range(5)]
        grader.grade_all(nodes)
        for node in nodes:
            assert node.grade is not None
            assert node.confidence is not None

    def test_duplicate_url_gets_same_grade_as_first(self):
        grader = make_grader()
        url = "https://dup.example.com/article"
        node1 = make_node(url=url, domain_authority=90.0, citation_count=5000)
        node2 = make_node(url=url, domain_authority=90.0, citation_count=5000)
        grader.grade_all([node1, node2], deduplicate=True)
        assert node1.grade == node2.grade
        assert node1.confidence == node2.confidence

    def test_duplicate_url_without_dedup_grades_independently(self):
        """With deduplicate=False every node is graded independently."""
        grader = make_grader()
        url = "https://dup.example.com/article"
        node1 = make_node(url=url, domain_authority=90.0)
        node2 = make_node(url=url, domain_authority=90.0)
        grader.grade_all([node1, node2], deduplicate=False)
        # Both should still be graded
        assert node1.grade is not None
        assert node2.grade is not None

    def test_three_duplicates_all_match_first(self):
        grader = make_grader()
        url = "https://triple-dup.example.com/"
        nodes = [make_node(url=url) for _ in range(3)]
        grader.grade_all(nodes, deduplicate=True)
        grades = [n.grade for n in nodes]
        confidences = [n.confidence for n in nodes]
        assert len(set(grades)) == 1
        assert len(set(confidences)) == 1

    def test_mixed_unique_and_duplicate(self):
        grader = make_grader()
        nodes = [
            make_node(url="https://a.com", domain_authority=90.0),
            make_node(url="https://b.com", domain_authority=50.0),
            make_node(url="https://a.com", domain_authority=90.0),  # dup of [0]
        ]
        grader.grade_all(nodes, deduplicate=True)
        # All three must have grades
        assert all(n.grade is not None for n in nodes)
        # node[0] and node[2] share the same URL → same grade
        assert nodes[0].grade == nodes[2].grade

    def test_empty_list_returns_empty(self):
        grader = make_grader()
        result = grader.grade_all([])
        assert result == []


# ---------------------------------------------------------------------------
# Integration: grader with real fixture data via mock search backend
# ---------------------------------------------------------------------------


class TestSourceGraderIntegration:
    FIXTURE_PATH = Path(__file__).parent / "fixtures" / "mock_results.json"

    def _load_nodes(self, topic: str) -> list[CitationNode]:
        from research_agent.backends.mock_search import MockSearchBackend

        backend = MockSearchBackend(fixture_path=self.FIXTURE_PATH)
        sources = backend.search(topic, top_k=10)
        return [
            CitationNode(source=s, hop_index=i, claim_snippet=s.snippet)
            for i, s in enumerate(sources)
        ]

    def test_climate_change_nodes_all_graded(self):
        grader = make_grader()
        nodes = self._load_nodes("climate change")
        grader.grade_all(nodes)
        for node in nodes:
            assert node.grade in ("A", "B", "C", "D", "F")
            assert 0.0 <= node.confidence <= 1.0

    def test_nasa_source_grades_higher_than_denial_site(self):
        """NASA (DA=92, many citations) should outrank the denial site (DA=12)."""
        grader = make_grader()
        nodes = self._load_nodes("climate change")
        grader.grade_all(nodes)

        grade_order = {"A": 5, "B": 4, "C": 3, "D": 2, "F": 1}

        nasa_node = next(n for n in nodes if "nasa.gov" in n.source.url)
        denial_node = next(
            n for n in nodes if "climatedenialsite" in n.source.url
        )

        assert grade_order[nasa_node.grade] > grade_order[denial_node.grade], (
            f"NASA ({nasa_node.grade}) should grade higher than "
            f"denial site ({denial_node.grade})"
        )

    def test_ai_nodes_all_graded(self):
        grader = make_grader()
        nodes = self._load_nodes("artificial intelligence")
        grader.grade_all(nodes)
        for node in nodes:
            assert node.grade is not None

    def test_quantum_nodes_all_graded(self):
        grader = make_grader()
        nodes = self._load_nodes("quantum computing")
        grader.grade_all(nodes)
        for node in nodes:
            assert node.grade is not None

    def test_default_fallback_nodes_with_null_fields_graded(self):
        """The default fixture has a source with null published_at and citation_count."""
        from research_agent.backends.mock_search import MockSearchBackend

        backend = MockSearchBackend(fixture_path=self.FIXTURE_PATH)
        sources = backend.search("completely unknown topic xyz")
        nodes = [CitationNode(source=s, hop_index=i) for i, s in enumerate(sources)]
        grader = make_grader()
        grader.grade_all(nodes)
        for node in nodes:
            assert node.grade is not None
            assert node.confidence is not None
            # Confidence must be < 1.0 for sources with missing signals
            scholar_node = next(
                (n for n in nodes if "scholar.google" in n.source.url), None
            )
            if scholar_node:
                assert scholar_node.confidence < 1.0
