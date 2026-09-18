"""Tests for the ContradictionDetector module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pytest

from research_agent.contradiction_detector import (
    _SIMILARITY_CONTRADICTION_THRESHOLD,
    ContradictionDetector,
    ContradictionRecord,
    MockEmbeddingStore,
)
from research_agent.models import CitationNode, Source


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "mock_results.json"
SIMILARITY_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "embedding_similarity.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_node(
    url: str = "https://example.com",
    snippet: str = "A factual claim.",
    hop_index: int = 0,
) -> CitationNode:
    source = Source(
        url=url,
        title="Test",
        snippet=snippet,
        domain="example.com",
    )
    node = CitationNode(source=source, hop_index=hop_index, claim_snippet=snippet)
    return node


def make_detector(
    fixture_path: Path = SIMILARITY_FIXTURE_PATH,
) -> ContradictionDetector:
    store = MockEmbeddingStore(fixture_path=fixture_path)
    return ContradictionDetector(embedding_store=store)


# ---------------------------------------------------------------------------
# MockEmbeddingStore tests
# ---------------------------------------------------------------------------


class TestMockEmbeddingStore:
    def test_loads_fixture_scores(self):
        store = MockEmbeddingStore(fixture_path=SIMILARITY_FIXTURE_PATH)
        assert len(store._scores) > 0

    def test_known_pair_returns_fixture_score(self):
        store = MockEmbeddingStore(fixture_path=SIMILARITY_FIXTURE_PATH)
        node_a = make_node(url="https://www.nasa.gov/climate", snippet="climate warming")
        node_b = make_node(
            url="https://climatedenialsite.example.com/article",
            snippet="climate hoax",
        )
        score = store.similarity(node_a, node_b)
        assert score == pytest.approx(0.08, abs=1e-6)

    def test_symmetric_key_lookup(self):
        """Swapping node order should return the same score."""
        store = MockEmbeddingStore(fixture_path=SIMILARITY_FIXTURE_PATH)
        node_a = make_node(url="https://www.nasa.gov/climate", snippet="climate warming")
        node_b = make_node(
            url="https://climatedenialsite.example.com/article",
            snippet="climate hoax",
        )
        score_ab = store.similarity(node_a, node_b)
        score_ba = store.similarity(node_b, node_a)
        assert score_ab == score_ba

    def test_unknown_pair_falls_back_to_jaccard(self):
        store = MockEmbeddingStore(fixture_path=SIMILARITY_FIXTURE_PATH)
        node_a = make_node(url="https://unknown-a.com", snippet="cats are great pets")
        node_b = make_node(url="https://unknown-b.com", snippet="dogs are great pets")
        score = store.similarity(node_a, node_b)
        # Jaccard: union=5, intersection=3 (are, great, pets)
        assert 0.0 < score < 1.0

    def test_identical_snippets_high_jaccard(self):
        store = MockEmbeddingStore(fixture_path=SIMILARITY_FIXTURE_PATH)
        node_a = make_node(url="https://a.com", snippet="the quick brown fox")
        node_b = make_node(url="https://b.com", snippet="the quick brown fox")
        score = store.similarity(node_a, node_b)
        assert score == pytest.approx(1.0)

    def test_no_overlap_snippets_zero_jaccard(self):
        store = MockEmbeddingStore(fixture_path=SIMILARITY_FIXTURE_PATH)
        node_a = make_node(url="https://a.com", snippet="cats meow softly")
        node_b = make_node(url="https://b.com", snippet="quantum superposition")
        score = store.similarity(node_a, node_b)
        assert score == 0.0

    def test_missing_fixture_path_returns_empty_store(self, tmp_path: Path):
        """A missing fixture file should produce an empty store (no error)."""
        store = MockEmbeddingStore(fixture_path=tmp_path / "nonexistent.json")
        node_a = make_node(url="https://a.com", snippet="hello world")
        node_b = make_node(url="https://b.com", snippet="hello world")
        # Falls back to Jaccard — should not raise
        score = store.similarity(node_a, node_b)
        assert score == pytest.approx(1.0)

    def test_invalid_fixture_raises_value_error(self, tmp_path: Path):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps([1, 2, 3]))
        with pytest.raises(ValueError):
            MockEmbeddingStore(fixture_path=bad)

    def test_empty_snippets_jaccard_returns_one(self):
        """Two empty snippets are considered identical (both zero sets)."""
        store = MockEmbeddingStore(fixture_path=SIMILARITY_FIXTURE_PATH)
        node_a = make_node(url="https://a.com", snippet="")
        node_b = make_node(url="https://b.com", snippet="")
        score = store._jaccard("", "")
        assert score == 1.0

    def test_one_empty_snippet_jaccard_returns_zero(self):
        store = MockEmbeddingStore(fixture_path=SIMILARITY_FIXTURE_PATH)
        score = store._jaccard("some text here", "")
        assert score == 0.0


# ---------------------------------------------------------------------------
# ContradictionDetector._has_negation tests
# ---------------------------------------------------------------------------


class TestHasNegation:
    def test_snippet_with_hoax_flagged(self):
        assert ContradictionDetector._has_negation("Climate change is a hoax.") is True

    def test_snippet_with_not_flagged(self):
        assert ContradictionDetector._has_negation("This is not accurate.") is True

    def test_snippet_with_manipulated_flagged(self):
        assert ContradictionDetector._has_negation("The data is manipulated.") is True

    def test_snippet_with_disagree_flagged(self):
        assert ContradictionDetector._has_negation("Scientists disagree about this.") is True

    def test_neutral_snippet_not_flagged(self):
        assert ContradictionDetector._has_negation("Global temperatures have risen 1.1C.") is False

    def test_empty_snippet_not_flagged(self):
        assert ContradictionDetector._has_negation("") is False

    def test_case_insensitive(self):
        assert ContradictionDetector._has_negation("This is a HOAX!") is True


# ---------------------------------------------------------------------------
# ContradictionDetector._shared_topic_clusters tests
# ---------------------------------------------------------------------------


class TestSharedTopicClusters:
    def test_both_mention_climate(self):
        a = "Global warming is driven by CO2 emissions."
        b = "Climate scientists disagree about temperature trends."
        shared = ContradictionDetector._shared_topic_clusters(a, b)
        assert "climate" in shared

    def test_no_shared_cluster(self):
        a = "Quantum entanglement is fascinating."
        b = "The dog ate my homework."
        shared = ContradictionDetector._shared_topic_clusters(a, b)
        assert shared == []

    def test_both_mention_ai(self):
        a = "Large language models are artificial intelligence systems."
        b = "Machine learning algorithms improve over time."
        shared = ContradictionDetector._shared_topic_clusters(a, b)
        assert "ai" in shared

    def test_different_clusters_not_shared(self):
        a = "Climate warming is real."
        b = "Quantum computing uses qubits."
        shared = ContradictionDetector._shared_topic_clusters(a, b)
        assert "climate" not in shared
        assert "quantum" not in shared

    def test_both_mention_quantum(self):
        a = "Qubit manipulation enables quantum speed-up."
        b = "Quantum superposition allows parallel computation."
        shared = ContradictionDetector._shared_topic_clusters(a, b)
        assert "quantum" in shared


# ---------------------------------------------------------------------------
# ContradictionDetector.detect — core behaviour
# ---------------------------------------------------------------------------


class TestContradictionDetectorDetect:
    def test_returns_list(self):
        detector = make_detector()
        nodes = [
            make_node(url="https://a.com", snippet="climate warming is real"),
            make_node(url="https://b.com", snippet="climate hoax not real"),
        ]
        result = detector.detect(nodes)
        assert isinstance(result, list)

    def test_known_contradiction_pair_detected(self):
        """NASA vs denial site — low similarity in fixture + negation heuristic."""
        detector = make_detector()
        node_nasa = make_node(
            url="https://www.nasa.gov/climate",
            snippet="Global temperatures have risen 1.1C since pre-industrial times, "
                    "driven primarily by human CO2 emissions.",
        )
        node_denial = make_node(
            url="https://climatedenialsite.example.com/article",
            snippet="Scientists disagree about whether human activity causes warming. "
                    "The data is manipulated.",
        )
        records = detector.detect([node_nasa, node_denial])
        assert len(records) >= 1
        pair_ids = {(r.node_a_id, r.node_b_id) for r in records}
        pair_ids |= {(r.node_b_id, r.node_a_id) for r in records}
        assert (node_nasa.id, node_denial.id) in pair_ids or (
            node_denial.id,
            node_nasa.id,
        ) in pair_ids

    def test_similar_supportive_sources_not_contradicted(self):
        """NASA and IPCC agree — high fixture similarity, no negation."""
        detector = make_detector()
        node_nasa = make_node(
            url="https://www.nasa.gov/climate",
            snippet="Global temperatures have risen 1.1C since pre-industrial times, "
                    "driven primarily by human CO2 emissions.",
        )
        node_ipcc = make_node(
            url="https://www.ipcc.ch/report/ar6",
            snippet="Human influence has warmed the atmosphere, ocean and land. "
                    "Widespread and rapid changes have occurred.",
        )
        records = detector.detect([node_nasa, node_ipcc])
        assert len(records) == 0

    def test_detect_empty_list_returns_empty(self):
        detector = make_detector()
        assert detector.detect([]) == []

    def test_detect_single_node_returns_empty(self):
        detector = make_detector()
        node = make_node()
        assert detector.detect([node]) == []

    def test_contradicts_lists_updated_on_nodes(self):
        detector = make_detector()
        node_nasa = make_node(
            url="https://www.nasa.gov/climate",
            snippet="Global temperatures have risen 1.1C, driven by CO2 emissions.",
        )
        node_denial = make_node(
            url="https://climatedenialsite.example.com/article",
            snippet="Climate change is a hoax. Scientists disagree about warming.",
        )
        detector.detect([node_nasa, node_denial])
        # Both nodes should reference each other in their contradicts list
        assert node_denial.id in node_nasa.contradicts
        assert node_nasa.id in node_denial.contradicts

    def test_no_double_entry_in_contradicts_on_repeated_detect(self):
        """Running detect twice must not push duplicate IDs into contradicts."""
        detector = make_detector()
        node_a = make_node(
            url="https://www.nasa.gov/climate",
            snippet="CO2 driven warming is real climate change.",
        )
        node_b = make_node(
            url="https://climatedenialsite.example.com/article",
            snippet="Climate warming hoax, scientists disagree.",
        )
        detector.detect([node_a, node_b])
        detector.detect([node_a, node_b])
        assert node_a.contradicts.count(node_b.id) == 1
        assert node_b.contradicts.count(node_a.id) == 1

    def test_empty_snippet_nodes_skipped(self):
        detector = make_detector()
        node_empty = make_node(url="https://empty.com", snippet="")
        node_empty.claim_snippet = ""
        node_normal = make_node(
            url="https://normal.com",
            snippet="Climate warming data shows temperature rise.",
        )
        records = detector.detect([node_empty, node_normal])
        assert records == []

    def test_no_shared_cluster_no_contradiction(self):
        """Two negation-heavy snippets on different topics should not contradict."""
        detector = make_detector()
        node_a = make_node(
            url="https://a.com",
            snippet="Dogs are not good pets for apartments.",
        )
        node_b = make_node(
            url="https://b.com",
            snippet="Quantum computers do not yet outperform classical ones.",
        )
        records = detector.detect([node_a, node_b])
        assert records == []

    def test_three_node_mix_correct_count(self):
        """NASA + IPCC (agree) + denial site (contradicts both)."""
        detector = make_detector()
        node_nasa = make_node(
            url="https://www.nasa.gov/climate",
            snippet="Global temperatures have risen 1.1C, driven by CO2 emissions.",
        )
        node_ipcc = make_node(
            url="https://www.ipcc.ch/report/ar6",
            snippet="Human influence has warmed the atmosphere and land.",
        )
        node_denial = make_node(
            url="https://climatedenialsite.example.com/article",
            snippet="Climate change hoax. Data manipulated by scientists who disagree.",
        )
        records = detector.detect([node_nasa, node_ipcc, node_denial])
        # denial contradicts both nasa and ipcc; nasa and ipcc agree
        assert len(records) >= 2

    def test_record_has_explanation_string(self):
        detector = make_detector()
        node_a = make_node(
            url="https://www.nasa.gov/climate",
            snippet="Climate warming is driven by human CO2 emissions.",
        )
        node_b = make_node(
            url="https://climatedenialsite.example.com/article",
            snippet="Climate change is a hoax and data is manipulated.",
        )
        records = detector.detect([node_a, node_b])
        assert len(records) >= 1
        assert isinstance(records[0].explanation, str)
        assert len(records[0].explanation) > 0

    def test_record_has_similarity_score(self):
        detector = make_detector()
        node_a = make_node(
            url="https://www.nasa.gov/climate",
            snippet="Climate warming CO2 emissions temperature rising.",
        )
        node_b = make_node(
            url="https://climatedenialsite.example.com/article",
            snippet="Climate hoax manipulated data scientists disagree.",
        )
        records = detector.detect([node_a, node_b])
        assert len(records) >= 1
        assert 0.0 <= records[0].similarity_score <= 1.0

    def test_record_heuristic_flags_non_empty(self):
        detector = make_detector()
        node_a = make_node(
            url="https://www.nasa.gov/climate",
            snippet="Climate warming CO2 emissions rising temperatures.",
        )
        node_b = make_node(
            url="https://climatedenialsite.example.com/article",
            snippet="Climate change hoax, scientists disagree, data manipulated.",
        )
        records = detector.detect([node_a, node_b])
        assert len(records) >= 1
        assert len(records[0].heuristic_flags) > 0


# ---------------------------------------------------------------------------
# ContradictionDetector with custom similarity threshold
# ---------------------------------------------------------------------------


class TestContradictionDetectorThreshold:
    def test_high_threshold_causes_more_contradictions(self):
        """With threshold=0.99 almost every different pair contradicts."""
        store = MockEmbeddingStore(fixture_path=SIMILARITY_FIXTURE_PATH)
        detector_strict = ContradictionDetector(
            embedding_store=store, similarity_threshold=0.99
        )
        node_a = make_node(
            url="https://www.ibm.com/quantum",
            snippet="IBM offers cloud-based quantum computing services with 1000+ qubits.",
        )
        node_b = make_node(
            url="https://nature.com/articles/quantum-supremacy",
            snippet="Google quantum supremacy calculation done in 200 seconds.",
        )
        records_strict = detector_strict.detect([node_a, node_b])
        # High threshold → more pairs flagged as contradictions
        assert len(records_strict) >= 1

    def test_zero_threshold_never_flags_similarity(self):
        """With threshold=0.0 the low-similarity heuristic never fires."""
        store = MockEmbeddingStore(fixture_path=SIMILARITY_FIXTURE_PATH)
        # Still might flag negation, so use non-negation snippets
        detector_permissive = ContradictionDetector(
            embedding_store=store, similarity_threshold=0.0
        )
        node_a = make_node(
            url="https://www.nasa.gov/climate",
            snippet="Global temperatures have risen driven by CO2 climate warming.",
        )
        node_b = make_node(
            url="https://www.ipcc.ch/report/ar6",
            snippet="Human influence warming atmosphere ocean land climate change.",
        )
        records = detector_permissive.detect([node_a, node_b])
        # No negation on either side, threshold=0 so similarity heuristic can't fire
        assert all("low_similarity" not in r.heuristic_flags for r in records)


# ---------------------------------------------------------------------------
# Integration: detector on fixture-loaded CitationNodes
# ---------------------------------------------------------------------------


class TestContradictionDetectorIntegration:
    def _load_climate_nodes(self) -> List[CitationNode]:
        from research_agent.backends.mock_search import MockSearchBackend

        backend = MockSearchBackend(fixture_path=FIXTURE_PATH)
        sources = backend.search("climate change", top_k=10)
        return [
            CitationNode(
                source=s,
                hop_index=i,
                claim_snippet=s.snippet,
            )
            for i, s in enumerate(sources)
        ]

    def _load_ai_nodes(self) -> List[CitationNode]:
        from research_agent.backends.mock_search import MockSearchBackend

        backend = MockSearchBackend(fixture_path=FIXTURE_PATH)
        sources = backend.search("artificial intelligence", top_k=10)
        return [
            CitationNode(
                source=s,
                hop_index=i,
                claim_snippet=s.snippet,
            )
            for i, s in enumerate(sources)
        ]

    def test_climate_nodes_include_contradiction(self):
        """The denial site should contradict the scientific sources."""
        detector = make_detector()
        nodes = self._load_climate_nodes()
        records = detector.detect(nodes)
        assert len(records) >= 1

    def test_denial_node_contradicts_list_populated(self):
        detector = make_detector()
        nodes = self._load_climate_nodes()
        detector.detect(nodes)
        denial_node = next(
            n for n in nodes if "climatedenialsite" in n.source.url
        )
        assert len(denial_node.contradicts) >= 1

    def test_ai_winter_contradicts_positive_ai_sources(self):
        """'AI Winter' article should contradict GPT-4 / Gemini positivity."""
        detector = make_detector()
        nodes = self._load_ai_nodes()
        records = detector.detect(nodes)
        # At least one contradiction expected (AI Winter vs optimistic papers)
        assert len(records) >= 1

    def test_all_records_have_non_empty_explanation(self):
        detector = make_detector()
        nodes = self._load_climate_nodes()
        records = detector.detect(nodes)
        for rec in records:
            assert isinstance(rec.explanation, str)
            assert rec.explanation.strip() != ""

    def test_all_records_have_valid_node_ids(self):
        detector = make_detector()
        nodes = self._load_climate_nodes()
        node_ids = {n.id for n in nodes}
        records = detector.detect(nodes)
        for rec in records:
            assert rec.node_a_id in node_ids
            assert rec.node_b_id in node_ids

    def test_no_self_contradiction(self):
        """A node must not appear as both sides of a contradiction record."""
        detector = make_detector()
        nodes = self._load_climate_nodes()
        records = detector.detect(nodes)
        for rec in records:
            assert rec.node_a_id != rec.node_b_id

    def test_quantum_nodes_no_false_contradiction(self):
        """IBM and Nature quantum sources broadly agree — should not contradict."""
        from research_agent.backends.mock_search import MockSearchBackend

        backend = MockSearchBackend(fixture_path=FIXTURE_PATH)
        sources = backend.search("quantum computing", top_k=5)
        nodes = [
            CitationNode(source=s, hop_index=i, claim_snippet=s.snippet)
            for i, s in enumerate(sources)
        ]
        detector = make_detector()
        records = detector.detect(nodes)
        # IBM (2023) and Nature (2019) quantum snippets share cluster but
        # fixture similarity is 0.71 (above threshold) and neither negates.
        assert len(records) == 0
