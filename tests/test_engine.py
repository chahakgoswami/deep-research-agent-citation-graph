"""Tests for the multi-hop reasoning engine (HopChain & MultiHopEngine)."""

from __future__ import annotations

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from research_agent.backends.mock_search import MockSearchBackend
from research_agent.engine import HopChain, MultiHopEngine, SubQuestionExpander
from research_agent.models import Hop, ResearchQuery, Source

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "mock_results.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_query(text: str = "climate change", max_hops: int = 3) -> ResearchQuery:
    return ResearchQuery(text=text, max_hops=max_hops)


def make_backend() -> MockSearchBackend:
    return MockSearchBackend(fixture_path=FIXTURE_PATH)


# ---------------------------------------------------------------------------
# SubQuestionExpander tests
# ---------------------------------------------------------------------------


class TestSubQuestionExpander:
    def test_returns_string(self):
        expander = SubQuestionExpander()
        result = expander.expand("climate change", hop_index=0)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_topic_embedded_in_question(self):
        expander = SubQuestionExpander()
        result = expander.expand("quantum computing", hop_index=0)
        assert "quantum computing" in result

    def test_different_hop_indices_give_different_questions(self):
        expander = SubQuestionExpander()
        q0 = expander.expand("AI", 0)
        q1 = expander.expand("AI", 1)
        q2 = expander.expand("AI", 2)
        # All three should differ
        assert q0 != q1
        assert q1 != q2
        assert q0 != q2

    def test_wraps_around_when_hop_exceeds_templates(self):
        expander = SubQuestionExpander()
        n = len(expander._TEMPLATES)
        # hop 0 and hop n should produce the same template pattern
        q_first = expander.expand("topic", 0)
        q_wrapped = expander.expand("topic", n)
        assert q_first == q_wrapped

    def test_all_hops_contain_topic(self):
        expander = SubQuestionExpander()
        topic = "nuclear fusion"
        for i in range(len(expander._TEMPLATES)):
            result = expander.expand(topic, i)
            assert topic in result, f"Topic missing in hop {i}: {result}"


# ---------------------------------------------------------------------------
# HopChain tests
# ---------------------------------------------------------------------------


class TestHopChain:
    def _make_chain(self, num_hops: int = 3) -> HopChain:
        query = make_query(max_hops=num_hops)
        chain = HopChain(query=query)
        for i in range(num_hops):
            sources = [Source(url=f"https://example.com/hop{i}/{j}") for j in range(2)]
            hop = Hop(hop_index=i, sub_query=f"sub query {i}", parent_query_id=query.id, sources=sources)
            chain.hops.append(hop)
        return chain

    def test_depth_reflects_number_of_hops(self):
        chain = self._make_chain(3)
        assert chain.depth == 3

    def test_depth_zero_when_empty(self):
        query = make_query()
        chain = HopChain(query=query)
        assert chain.depth == 0

    def test_all_sources_flattened(self):
        chain = self._make_chain(3)
        # 3 hops * 2 sources each = 6
        assert len(chain.all_sources) == 6

    def test_all_sources_empty_when_no_hops(self):
        query = make_query()
        chain = HopChain(query=query)
        assert chain.all_sources == []

    def test_get_hop_returns_correct_hop(self):
        chain = self._make_chain(3)
        hop = chain.get_hop(1)
        assert hop.hop_index == 1
        assert hop.sub_query == "sub query 1"

    def test_get_hop_out_of_range_raises(self):
        chain = self._make_chain(2)
        with pytest.raises(IndexError):
            chain.get_hop(99)

    def test_sub_queries_returns_list(self):
        chain = self._make_chain(3)
        sq = chain.sub_queries()
        assert sq == ["sub query 0", "sub query 1", "sub query 2"]

    def test_sub_queries_empty_when_no_hops(self):
        query = make_query()
        chain = HopChain(query=query)
        assert chain.sub_queries() == []

    def test_query_stored_correctly(self):
        query = make_query(text="test topic", max_hops=2)
        chain = HopChain(query=query)
        assert chain.query.text == "test topic"


# ---------------------------------------------------------------------------
# MultiHopEngine tests
# ---------------------------------------------------------------------------


class TestMultiHopEngine:
    def test_run_returns_hop_chain(self):
        engine = MultiHopEngine(backend=make_backend())
        query = make_query(max_hops=2)
        chain = engine.run(query)
        assert isinstance(chain, HopChain)

    def test_chain_depth_equals_max_hops(self):
        for max_hops in [1, 2, 3, 5]:
            engine = MultiHopEngine(backend=make_backend())
            query = make_query(max_hops=max_hops)
            chain = engine.run(query)
            assert chain.depth == max_hops, (
                f"Expected depth {max_hops}, got {chain.depth}"
            )

    def test_terminates_at_max_depth_1(self):
        engine = MultiHopEngine(backend=make_backend())
        query = make_query(max_hops=1)
        chain = engine.run(query)
        assert chain.depth == 1

    def test_terminates_at_max_depth_10(self):
        engine = MultiHopEngine(backend=make_backend())
        query = make_query(max_hops=10)
        chain = engine.run(query)
        assert chain.depth == 10

    def test_each_hop_has_correct_index(self):
        engine = MultiHopEngine(backend=make_backend())
        query = make_query(max_hops=4)
        chain = engine.run(query)
        for i, hop in enumerate(chain.hops):
            assert hop.hop_index == i

    def test_each_hop_has_correct_parent_query_id(self):
        engine = MultiHopEngine(backend=make_backend())
        query = make_query(max_hops=3)
        chain = engine.run(query)
        for hop in chain.hops:
            assert hop.parent_query_id == query.id

    def test_each_hop_sources_is_list_of_sources(self):
        engine = MultiHopEngine(backend=make_backend())
        query = make_query(max_hops=3)
        chain = engine.run(query)
        for hop in chain.hops:
            assert isinstance(hop.sources, list)
            assert all(isinstance(s, Source) for s in hop.sources)

    def test_sources_retrieved_each_hop(self):
        engine = MultiHopEngine(backend=make_backend(), top_k=5)
        query = make_query(max_hops=2)
        chain = engine.run(query)
        for hop in chain.hops:
            assert len(hop.sources) > 0

    def test_top_k_limits_sources_per_hop(self):
        engine = MultiHopEngine(backend=make_backend(), top_k=1)
        query = make_query(max_hops=3)
        chain = engine.run(query)
        for hop in chain.hops:
            assert len(hop.sources) <= 1

    def test_sub_queries_differ_across_hops(self):
        engine = MultiHopEngine(backend=make_backend())
        query = make_query(text="artificial intelligence", max_hops=3)
        chain = engine.run(query)
        sub_queries = chain.sub_queries()
        # All sub-queries should be unique
        assert len(set(sub_queries)) == len(sub_queries)

    def test_sub_queries_contain_original_topic(self):
        engine = MultiHopEngine(backend=make_backend())
        query = make_query(text="quantum computing", max_hops=3)
        chain = engine.run(query)
        for sq in chain.sub_queries():
            assert "quantum computing" in sq

    def test_backend_called_once_per_hop(self):
        """Verify the backend is invoked exactly max_hops times."""
        mock_backend = MagicMock(spec=MockSearchBackend)
        mock_backend.search.return_value = [Source(url="https://example.com")]
        engine = MultiHopEngine(backend=mock_backend)
        query = make_query(max_hops=4)
        engine.run(query)
        assert mock_backend.search.call_count == 4

    def test_chain_query_reference_is_same_object(self):
        engine = MultiHopEngine(backend=make_backend())
        query = make_query(max_hops=2)
        chain = engine.run(query)
        assert chain.query is query

    def test_all_sources_accumulates_across_hops(self):
        engine = MultiHopEngine(backend=make_backend(), top_k=2)
        query = make_query(max_hops=3)
        chain = engine.run(query)
        # all_sources should equal the sum of sources per hop
        total = sum(len(h.sources) for h in chain.hops)
        assert len(chain.all_sources) == total

    def test_custom_expander_used(self):
        """Verify a custom expander's output is used as the sub-query."""
        custom_expander = MagicMock(spec=SubQuestionExpander)
        custom_expander.expand.return_value = "custom sub-question"
        mock_backend = MagicMock(spec=MockSearchBackend)
        mock_backend.search.return_value = []
        engine = MultiHopEngine(backend=mock_backend, expander=custom_expander)
        query = make_query(max_hops=2)
        chain = engine.run(query)
        for hop in chain.hops:
            assert hop.sub_query == "custom sub-question"

    def test_run_with_different_queries_give_independent_chains(self):
        engine = MultiHopEngine(backend=make_backend())
        q1 = make_query(text="climate change", max_hops=2)
        q2 = make_query(text="quantum computing", max_hops=3)
        chain1 = engine.run(q1)
        chain2 = engine.run(q2)
        assert chain1.depth == 2
        assert chain2.depth == 3
        assert chain1.query.id != chain2.query.id


# ---------------------------------------------------------------------------
# Integration: engine with real fixture backend
# ---------------------------------------------------------------------------


class TestMultiHopEngineIntegration:
    def test_full_run_climate_change(self):
        backend = make_backend()
        engine = MultiHopEngine(backend=backend, top_k=3)
        query = ResearchQuery(text="climate change", max_hops=3)
        chain = engine.run(query)

        assert chain.depth == 3
        assert len(chain.all_sources) > 0
        for hop in chain.hops:
            assert hop.sub_query
            for source in hop.sources:
                assert source.url
                assert source.domain

    def test_full_run_artificial_intelligence(self):
        backend = make_backend()
        engine = MultiHopEngine(backend=backend, top_k=5)
        query = ResearchQuery(text="artificial intelligence", max_hops=2)
        chain = engine.run(query)

        assert chain.depth == 2
        # At least one source should come from arxiv or deepmind (fixture data)
        urls = [s.url for s in chain.all_sources]
        assert any("arxiv" in u or "deepmind" in u for u in urls)

    def test_hop_timestamps_set(self):
        backend = make_backend()
        engine = MultiHopEngine(backend=backend)
        query = make_query(max_hops=2)
        chain = engine.run(query)
        from datetime import datetime
        for hop in chain.hops:
            assert isinstance(hop.timestamp, datetime)
