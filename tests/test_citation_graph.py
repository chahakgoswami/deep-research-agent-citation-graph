"""Tests for the CitationGraph class."""

from __future__ import annotations

from pathlib import Path
from typing import List

import pytest

from research_agent.citation_graph import (
    EDGE_CITED_BY,
    EDGE_SUPPORTS,
    CitationGraph,
)
from research_agent.models import CitationNode, Source

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "mock_results.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_source(url: str = "https://example.com", **kwargs) -> Source:
    return Source(
        url=url,
        title=kwargs.get("title", "Test Article"),
        snippet=kwargs.get("snippet", "A snippet."),
        domain=kwargs.get("domain", "example.com"),
        domain_authority=kwargs.get("domain_authority", 70.0),
        citation_count=kwargs.get("citation_count", 100),
    )


def make_node(
    url: str = "https://example.com",
    hop_index: int = 0,
    **source_kwargs,
) -> CitationNode:
    return CitationNode(
        source=make_source(url=url, **source_kwargs),
        hop_index=hop_index,
    )


def make_graph_with_nodes(n: int = 3) -> tuple[CitationGraph, list[CitationNode]]:
    """Return a graph with *n* nodes (no edges)."""
    graph = CitationGraph()
    nodes = [make_node(url=f"https://example.com/{i}", hop_index=i % 3) for i in range(n)]
    graph.add_nodes(nodes)
    return graph, nodes


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestCitationGraphConstruction:
    def test_empty_graph_has_zero_nodes(self):
        graph = CitationGraph()
        assert graph.node_count == 0

    def test_empty_graph_has_zero_edges(self):
        graph = CitationGraph()
        assert graph.edge_count == 0

    def test_add_single_node(self):
        graph = CitationGraph()
        node = make_node()
        graph.add_node(node)
        assert graph.node_count == 1

    def test_add_multiple_nodes(self):
        graph, nodes = make_graph_with_nodes(5)
        assert graph.node_count == 5

    def test_add_nodes_batch(self):
        graph = CitationGraph()
        nodes = [make_node(url=f"https://a.com/{i}") for i in range(4)]
        graph.add_nodes(nodes)
        assert graph.node_count == 4

    def test_get_node_returns_same_object(self):
        graph = CitationGraph()
        node = make_node()
        graph.add_node(node)
        retrieved = graph.get_node(node.id)
        assert retrieved is node

    def test_get_node_missing_raises_key_error(self):
        graph = CitationGraph()
        with pytest.raises(KeyError):
            graph.get_node("nonexistent-id")

    def test_add_node_overwrites_existing_id(self):
        """Adding a node with an existing id should silently replace it."""
        graph = CitationGraph()
        node = make_node()
        graph.add_node(node)
        # Build a new CitationNode but force same id
        node2 = make_node(url="https://other.com")
        node2.id = node.id  # forcibly reuse same id
        graph.add_node(node2)
        assert graph.node_count == 1
        assert graph.get_node(node.id).source.url == "https://other.com"

    def test_all_nodes_returns_all(self):
        graph, nodes = make_graph_with_nodes(3)
        all_nodes = graph.all_nodes()
        assert len(all_nodes) == 3
        node_ids = {n.id for n in all_nodes}
        assert node_ids == {n.id for n in nodes}

    def test_all_nodes_empty_graph(self):
        graph = CitationGraph()
        assert graph.all_nodes() == []


# ---------------------------------------------------------------------------
# Edge insertion
# ---------------------------------------------------------------------------


class TestEdgeInsertion:
    def test_add_cited_by_edge(self):
        graph, (n0, n1) = make_graph_with_nodes(2)
        graph.add_edge(n0.id, n1.id, edge_type=EDGE_CITED_BY)
        assert graph.edge_count == 1

    def test_add_supports_edge(self):
        graph, (n0, n1) = make_graph_with_nodes(2)
        graph.add_edge(n0.id, n1.id, edge_type=EDGE_SUPPORTS)
        assert graph.edge_count == 1

    def test_has_edge_true(self):
        graph, (n0, n1) = make_graph_with_nodes(2)
        graph.add_edge(n0.id, n1.id, edge_type=EDGE_CITED_BY)
        assert graph.has_edge(n0.id, n1.id) is True

    def test_has_edge_false_when_no_edge(self):
        graph, (n0, n1) = make_graph_with_nodes(2)
        assert graph.has_edge(n0.id, n1.id) is False

    def test_has_edge_with_edge_type_filter_true(self):
        graph, (n0, n1) = make_graph_with_nodes(2)
        graph.add_edge(n0.id, n1.id, edge_type=EDGE_SUPPORTS)
        assert graph.has_edge(n0.id, n1.id, edge_type=EDGE_SUPPORTS) is True

    def test_has_edge_with_wrong_edge_type_false(self):
        graph, (n0, n1) = make_graph_with_nodes(2)
        graph.add_edge(n0.id, n1.id, edge_type=EDGE_SUPPORTS)
        assert graph.has_edge(n0.id, n1.id, edge_type=EDGE_CITED_BY) is False

    def test_unknown_edge_type_raises_value_error(self):
        graph, (n0, n1) = make_graph_with_nodes(2)
        with pytest.raises(ValueError):
            graph.add_edge(n0.id, n1.id, edge_type="unknown-type")

    def test_missing_source_node_raises_key_error(self):
        graph = CitationGraph()
        node = make_node()
        graph.add_node(node)
        with pytest.raises(KeyError):
            graph.add_edge("nonexistent", node.id, edge_type=EDGE_CITED_BY)

    def test_missing_target_node_raises_key_error(self):
        graph = CitationGraph()
        node = make_node()
        graph.add_node(node)
        with pytest.raises(KeyError):
            graph.add_edge(node.id, "nonexistent", edge_type=EDGE_CITED_BY)

    def test_multiple_edges_between_different_pairs(self):
        graph, (n0, n1, n2) = make_graph_with_nodes(3)
        graph.add_edge(n0.id, n1.id, edge_type=EDGE_CITED_BY)
        graph.add_edge(n1.id, n2.id, edge_type=EDGE_SUPPORTS)
        graph.add_edge(n0.id, n2.id, edge_type=EDGE_CITED_BY)
        assert graph.edge_count == 3

    def test_edges_of_type_cited_by(self):
        graph, (n0, n1, n2) = make_graph_with_nodes(3)
        graph.add_edge(n0.id, n1.id, edge_type=EDGE_CITED_BY)
        graph.add_edge(n1.id, n2.id, edge_type=EDGE_SUPPORTS)
        cited_edges = graph.edges_of_type(EDGE_CITED_BY)
        assert len(cited_edges) == 1
        assert (n0.id, n1.id) in cited_edges

    def test_edges_of_type_supports(self):
        graph, (n0, n1, n2) = make_graph_with_nodes(3)
        graph.add_edge(n0.id, n1.id, edge_type=EDGE_CITED_BY)
        graph.add_edge(n1.id, n2.id, edge_type=EDGE_SUPPORTS)
        support_edges = graph.edges_of_type(EDGE_SUPPORTS)
        assert len(support_edges) == 1
        assert (n1.id, n2.id) in support_edges

    def test_edges_of_type_empty_when_none_of_type(self):
        graph, (n0, n1) = make_graph_with_nodes(2)
        graph.add_edge(n0.id, n1.id, edge_type=EDGE_CITED_BY)
        assert graph.edges_of_type(EDGE_SUPPORTS) == []

    def test_directional_edge_not_reversed(self):
        """An edge A→B does not imply B→A."""
        graph, (n0, n1) = make_graph_with_nodes(2)
        graph.add_edge(n0.id, n1.id, edge_type=EDGE_CITED_BY)
        assert graph.has_edge(n1.id, n0.id) is False


# ---------------------------------------------------------------------------
# Traversal: predecessors & successors
# ---------------------------------------------------------------------------


class TestTraversalPredecessorsSuccessors:
    def test_predecessors_returns_correct_nodes(self):
        graph, (n0, n1, n2) = make_graph_with_nodes(3)
        graph.add_edge(n0.id, n2.id, edge_type=EDGE_CITED_BY)
        graph.add_edge(n1.id, n2.id, edge_type=EDGE_CITED_BY)
        preds = graph.predecessors(n2.id)
        pred_ids = {n.id for n in preds}
        assert pred_ids == {n0.id, n1.id}

    def test_predecessors_empty_for_root(self):
        graph, (n0, n1) = make_graph_with_nodes(2)
        graph.add_edge(n0.id, n1.id, edge_type=EDGE_CITED_BY)
        assert graph.predecessors(n0.id) == []

    def test_predecessors_with_edge_type_filter(self):
        graph, (n0, n1, n2) = make_graph_with_nodes(3)
        graph.add_edge(n0.id, n2.id, edge_type=EDGE_CITED_BY)
        graph.add_edge(n1.id, n2.id, edge_type=EDGE_SUPPORTS)
        cited_preds = graph.predecessors(n2.id, edge_type=EDGE_CITED_BY)
        assert len(cited_preds) == 1
        assert cited_preds[0].id == n0.id

    def test_predecessors_unknown_node_raises(self):
        graph = CitationGraph()
        with pytest.raises(KeyError):
            graph.predecessors("ghost")

    def test_successors_returns_correct_nodes(self):
        graph, (n0, n1, n2) = make_graph_with_nodes(3)
        graph.add_edge(n0.id, n1.id, edge_type=EDGE_SUPPORTS)
        graph.add_edge(n0.id, n2.id, edge_type=EDGE_SUPPORTS)
        succs = graph.successors(n0.id)
        succ_ids = {n.id for n in succs}
        assert succ_ids == {n1.id, n2.id}

    def test_successors_empty_for_leaf(self):
        graph, (n0, n1) = make_graph_with_nodes(2)
        graph.add_edge(n0.id, n1.id, edge_type=EDGE_SUPPORTS)
        assert graph.successors(n1.id) == []

    def test_successors_with_edge_type_filter(self):
        graph, (n0, n1, n2) = make_graph_with_nodes(3)
        graph.add_edge(n0.id, n1.id, edge_type=EDGE_CITED_BY)
        graph.add_edge(n0.id, n2.id, edge_type=EDGE_SUPPORTS)
        cited_succs = graph.successors(n0.id, edge_type=EDGE_CITED_BY)
        assert len(cited_succs) == 1
        assert cited_succs[0].id == n1.id

    def test_successors_unknown_node_raises(self):
        graph = CitationGraph()
        with pytest.raises(KeyError):
            graph.successors("ghost")


# ---------------------------------------------------------------------------
# Traversal: authority hubs
# ---------------------------------------------------------------------------


class TestAuthorityHubs:
    def _build_hub_graph(self) -> tuple[CitationGraph, list[CitationNode]]:
        """Build a graph where n0 is cited by n1, n2, n3 and n1 is cited by n2."""
        graph, nodes = make_graph_with_nodes(5)
        n0, n1, n2, n3, n4 = nodes
        graph.add_edge(n1.id, n0.id, edge_type=EDGE_CITED_BY)  # n0 in-degree += 1
        graph.add_edge(n2.id, n0.id, edge_type=EDGE_CITED_BY)  # n0 in-degree += 1
        graph.add_edge(n3.id, n0.id, edge_type=EDGE_CITED_BY)  # n0 in-degree += 1
        graph.add_edge(n2.id, n1.id, edge_type=EDGE_CITED_BY)  # n1 in-degree += 1
        # n4 is an orphan (no edges)
        return graph, nodes

    def test_top_hub_has_highest_in_degree(self):
        graph, nodes = self._build_hub_graph()
        hubs = graph.authority_hubs(top_n=1)
        assert len(hubs) == 1
        assert hubs[0].id == nodes[0].id  # n0 has in-degree 3

    def test_top_n_limits_results(self):
        graph, _ = self._build_hub_graph()
        hubs = graph.authority_hubs(top_n=2)
        assert len(hubs) <= 2

    def test_top_n_zero_returns_empty(self):
        graph, _ = self._build_hub_graph()
        assert graph.authority_hubs(top_n=0) == []

    def test_top_n_exceeds_node_count_returns_all(self):
        graph, nodes = self._build_hub_graph()
        hubs = graph.authority_hubs(top_n=100)
        assert len(hubs) == graph.node_count

    def test_hub_order_descending_by_in_degree(self):
        graph, nodes = self._build_hub_graph()
        hubs = graph.authority_hubs(top_n=3)
        # n0 (in-degree 3) should come before n1 (in-degree 1)
        hub_ids = [h.id for h in hubs]
        assert hub_ids.index(nodes[0].id) < hub_ids.index(nodes[1].id)

    def test_authority_hubs_with_edge_type_filter(self):
        graph, (n0, n1, n2) = make_graph_with_nodes(3)
        graph.add_edge(n1.id, n0.id, edge_type=EDGE_CITED_BY)
        graph.add_edge(n2.id, n0.id, edge_type=EDGE_CITED_BY)
        graph.add_edge(n0.id, n2.id, edge_type=EDGE_SUPPORTS)  # n2 gets supports
        # Filter by cited-by: n0 should still be top hub
        hubs = graph.authority_hubs(top_n=1, edge_type=EDGE_CITED_BY)
        assert hubs[0].id == n0.id

    def test_authority_hubs_empty_graph(self):
        graph = CitationGraph()
        assert graph.authority_hubs(top_n=5) == []

    def test_authority_hubs_no_edges(self):
        graph, nodes = make_graph_with_nodes(3)
        hubs = graph.authority_hubs(top_n=3)
        # All nodes have in-degree 0, so all should be returned
        assert len(hubs) == 3


# ---------------------------------------------------------------------------
# Traversal: orphan nodes
# ---------------------------------------------------------------------------


class TestOrphanNodes:
    def test_all_nodes_orphans_in_empty_edge_graph(self):
        graph, nodes = make_graph_with_nodes(3)
        orphans = graph.orphan_nodes()
        assert len(orphans) == 3

    def test_connected_nodes_not_orphans(self):
        graph, (n0, n1, n2) = make_graph_with_nodes(3)
        graph.add_edge(n0.id, n1.id, edge_type=EDGE_CITED_BY)
        orphans = graph.orphan_nodes()
        orphan_ids = {o.id for o in orphans}
        assert n0.id not in orphan_ids
        assert n1.id not in orphan_ids
        # n2 has no edges
        assert n2.id in orphan_ids

    def test_no_orphans_when_all_connected(self):
        graph, (n0, n1, n2) = make_graph_with_nodes(3)
        graph.add_edge(n0.id, n1.id, edge_type=EDGE_CITED_BY)
        graph.add_edge(n1.id, n2.id, edge_type=EDGE_SUPPORTS)
        orphans = graph.orphan_nodes()
        assert orphans == []

    def test_empty_graph_returns_empty_orphans(self):
        graph = CitationGraph()
        assert graph.orphan_nodes() == []

    def test_orphans_with_edge_type_filter(self):
        """A node connected only via EDGE_SUPPORTS is orphan w.r.t. EDGE_CITED_BY."""
        graph, (n0, n1, n2) = make_graph_with_nodes(3)
        # Connect n0 -> n1 via supports only
        graph.add_edge(n0.id, n1.id, edge_type=EDGE_SUPPORTS)
        # n2 has no edges at all
        orphans_cited = graph.orphan_nodes(edge_type=EDGE_CITED_BY)
        orphan_ids = {o.id for o in orphans_cited}
        # n0 and n1 are connected via supports, so they are orphans w.r.t. cited-by
        assert n0.id in orphan_ids
        assert n1.id in orphan_ids
        assert n2.id in orphan_ids

    def test_single_node_graph_is_orphan(self):
        graph = CitationGraph()
        node = make_node()
        graph.add_node(node)
        orphans = graph.orphan_nodes()
        assert len(orphans) == 1
        assert orphans[0].id == node.id


# ---------------------------------------------------------------------------
# Subgraph by hop
# ---------------------------------------------------------------------------


class TestSubgraphByHop:
    def test_subgraph_contains_only_hop_nodes(self):
        graph = CitationGraph()
        n0 = make_node(url="https://a.com", hop_index=0)
        n1 = make_node(url="https://b.com", hop_index=0)
        n2 = make_node(url="https://c.com", hop_index=1)
        graph.add_nodes([n0, n1, n2])
        sub = graph.subgraph_by_hop(0)
        assert sub.node_count == 2
        with pytest.raises(KeyError):
            sub.get_node(n2.id)

    def test_subgraph_preserves_intra_hop_edges(self):
        graph = CitationGraph()
        n0 = make_node(url="https://a.com", hop_index=0)
        n1 = make_node(url="https://b.com", hop_index=0)
        n2 = make_node(url="https://c.com", hop_index=1)
        graph.add_nodes([n0, n1, n2])
        graph.add_edge(n0.id, n1.id, edge_type=EDGE_CITED_BY)
        graph.add_edge(n0.id, n2.id, edge_type=EDGE_CITED_BY)  # cross-hop edge
        sub = graph.subgraph_by_hop(0)
        assert sub.has_edge(n0.id, n1.id) is True
        # Cross-hop edge not present in subgraph
        assert sub.has_edge(n0.id, n2.id) is False

    def test_subgraph_empty_for_missing_hop(self):
        graph, nodes = make_graph_with_nodes(3)
        sub = graph.subgraph_by_hop(99)
        assert sub.node_count == 0


# ---------------------------------------------------------------------------
# from_hop_chain factory
# ---------------------------------------------------------------------------


class TestFromHopChain:
    def test_from_hop_chain_creates_correct_node_count(self):
        from research_agent.backends.mock_search import MockSearchBackend
        from research_agent.engine import MultiHopEngine
        from research_agent.models import ResearchQuery

        backend = MockSearchBackend(fixture_path=FIXTURE_PATH)
        engine = MultiHopEngine(backend=backend, top_k=2)
        query = ResearchQuery(text="climate change", max_hops=2)
        chain = engine.run(query)

        graph = CitationGraph.from_hop_chain(chain)
        expected = sum(len(h.sources) for h in chain.hops)
        assert graph.node_count == expected

    def test_from_hop_chain_nodes_have_correct_hop_index(self):
        from research_agent.backends.mock_search import MockSearchBackend
        from research_agent.engine import MultiHopEngine
        from research_agent.models import ResearchQuery

        backend = MockSearchBackend(fixture_path=FIXTURE_PATH)
        engine = MultiHopEngine(backend=backend, top_k=2)
        query = ResearchQuery(text="climate change", max_hops=3)
        chain = engine.run(query)

        graph = CitationGraph.from_hop_chain(chain)
        for hop in chain.hops:
            sub = graph.subgraph_by_hop(hop.hop_index)
            assert sub.node_count == len(hop.sources)

    def test_from_hop_chain_no_edges_by_default(self):
        from research_agent.backends.mock_search import MockSearchBackend
        from research_agent.engine import MultiHopEngine
        from research_agent.models import ResearchQuery

        backend = MockSearchBackend(fixture_path=FIXTURE_PATH)
        engine = MultiHopEngine(backend=backend, top_k=2)
        query = ResearchQuery(text="quantum computing", max_hops=2)
        chain = engine.run(query)

        graph = CitationGraph.from_hop_chain(chain)
        assert graph.edge_count == 0

    def test_from_hop_chain_claim_snippet_populated(self):
        from research_agent.backends.mock_search import MockSearchBackend
        from research_agent.engine import MultiHopEngine
        from research_agent.models import ResearchQuery

        backend = MockSearchBackend(fixture_path=FIXTURE_PATH)
        engine = MultiHopEngine(backend=backend, top_k=2)
        query = ResearchQuery(text="climate change", max_hops=1)
        chain = engine.run(query)

        graph = CitationGraph.from_hop_chain(chain)
        for node in graph.all_nodes():
            assert node.claim_snippet == node.source.snippet


# ---------------------------------------------------------------------------
# Integration: full pipeline with grader
# ---------------------------------------------------------------------------


class TestCitationGraphIntegration:
    def test_build_graded_graph_from_fixture(self):
        from research_agent.backends.mock_search import MockSearchBackend
        from research_agent.engine import MultiHopEngine
        from research_agent.grader import SourceGrader
        from research_agent.models import ResearchQuery

        backend = MockSearchBackend(fixture_path=FIXTURE_PATH)
        engine = MultiHopEngine(backend=backend, top_k=3)
        query = ResearchQuery(text="climate change", max_hops=2)
        chain = engine.run(query)

        graph = CitationGraph.from_hop_chain(chain)
        grader = SourceGrader()
        grader.grade_all(graph.all_nodes())

        for node in graph.all_nodes():
            assert node.grade in ("A", "B", "C", "D", "F")
            assert 0.0 <= node.confidence <= 1.0

    def test_wire_edges_and_find_hub(self):
        """Manually wire edges and verify hub detection."""
        graph = CitationGraph()
        nodes = [
            make_node(url=f"https://example.com/{i}", hop_index=0)
            for i in range(5)
        ]
        graph.add_nodes(nodes)

        # Make nodes[0] a hub: cited by all others
        for n in nodes[1:]:
            graph.add_edge(n.id, nodes[0].id, edge_type=EDGE_CITED_BY)

        hubs = graph.authority_hubs(top_n=1)
        assert hubs[0].id == nodes[0].id

    def test_orphan_detection_after_partial_wiring(self):
        graph, nodes = make_graph_with_nodes(5)
        # Wire first two nodes
        graph.add_edge(nodes[0].id, nodes[1].id, edge_type=EDGE_CITED_BY)
        orphans = graph.orphan_nodes()
        orphan_ids = {o.id for o in orphans}
        # nodes[2], nodes[3], nodes[4] are orphans
        for i in range(2, 5):
            assert nodes[i].id in orphan_ids
        assert nodes[0].id not in orphan_ids
        assert nodes[1].id not in orphan_ids
