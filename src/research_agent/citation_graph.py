"""CitationGraph module for the Research Agent.

Builds a directed graph of CitationNodes using networkx, with edges
representing 'cited-by' and 'supports' relationships.  Provides
traversal helpers to identify authority hubs and orphan sources.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import networkx as nx

from research_agent.models import CitationNode

# ---------------------------------------------------------------------------
# Edge type constants
# ---------------------------------------------------------------------------

EDGE_CITED_BY = "cited-by"
EDGE_SUPPORTS = "supports"


class CitationGraph:
    """A directed graph of :class:`~research_agent.models.CitationNode` objects.

    Nodes are keyed by ``CitationNode.id``.  Two edge types are supported:

    * **cited-by** – ``source → target`` means *source* is cited by *target*.
    * **supports** – ``source → target`` means *source* supports the claim
      made by *target*.

    Usage::

        graph = CitationGraph()
        graph.add_node(node_a)
        graph.add_node(node_b)
        graph.add_edge(node_a.id, node_b.id, edge_type="cited-by")
        hubs = graph.authority_hubs(top_n=3)
        orphans = graph.orphan_nodes()
    """

    def __init__(self) -> None:
        # Underlying directed graph
        self._graph: nx.DiGraph = nx.DiGraph()
        # Fast lookup: node_id -> CitationNode
        self._nodes: Dict[str, CitationNode] = {}

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    def add_node(self, node: CitationNode) -> None:
        """Add *node* to the graph.

        If a node with the same ``id`` already exists it is silently
        overwritten with the new object.

        Args:
            node: A :class:`~research_agent.models.CitationNode` instance.
        """
        self._nodes[node.id] = node
        self._graph.add_node(node.id, citation_node=node)

    def add_nodes(self, nodes: Iterable[CitationNode]) -> None:
        """Convenience wrapper to add multiple nodes at once."""
        for node in nodes:
            self.add_node(node)

    def get_node(self, node_id: str) -> CitationNode:
        """Return the :class:`CitationNode` with *node_id*.

        Raises:
            KeyError: If *node_id* is not present in the graph.
        """
        if node_id not in self._nodes:
            raise KeyError(f"Node '{node_id}' not found in CitationGraph.")
        return self._nodes[node_id]

    @property
    def node_count(self) -> int:
        """Total number of nodes in the graph."""
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        """Total number of directed edges in the graph."""
        return self._graph.number_of_edges()

    def all_nodes(self) -> List[CitationNode]:
        """Return all :class:`CitationNode` objects in insertion order."""
        return list(self._nodes.values())

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
    ) -> None:
        """Add a directed edge from *source_id* to *target_id*.

        Args:
            source_id: ID of the source :class:`CitationNode`.
            target_id: ID of the target :class:`CitationNode`.
            edge_type:  One of :data:`EDGE_CITED_BY` or :data:`EDGE_SUPPORTS`.

        Raises:
            KeyError:   If either node ID is not present in the graph.
            ValueError: If *edge_type* is not a recognised constant.
        """
        if source_id not in self._nodes:
            raise KeyError(f"Source node '{source_id}' not found in CitationGraph.")
        if target_id not in self._nodes:
            raise KeyError(f"Target node '{target_id}' not found in CitationGraph.")
        if edge_type not in (EDGE_CITED_BY, EDGE_SUPPORTS):
            raise ValueError(
                f"Unknown edge_type '{edge_type}'. "
                f"Use '{EDGE_CITED_BY}' or '{EDGE_SUPPORTS}'."
            )
        self._graph.add_edge(source_id, target_id, edge_type=edge_type)

    def has_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: Optional[str] = None,
    ) -> bool:
        """Return ``True`` if an edge exists between the two nodes.

        If *edge_type* is provided, also checks that the edge carries
        that specific type attribute.

        Args:
            source_id: ID of the source node.
            target_id: ID of the target node.
            edge_type:  Optional edge type filter.
        """
        if not self._graph.has_edge(source_id, target_id):
            return False
        if edge_type is None:
            return True
        return self._graph[source_id][target_id].get("edge_type") == edge_type

    def edges_of_type(
        self, edge_type: str
    ) -> List[Tuple[str, str]]:
        """Return all (source_id, target_id) pairs for edges of *edge_type*."""
        return [
            (u, v)
            for u, v, data in self._graph.edges(data=True)
            if data.get("edge_type") == edge_type
        ]

    # ------------------------------------------------------------------
    # Traversal helpers
    # ------------------------------------------------------------------

    def authority_hubs(
        self,
        top_n: int = 5,
        edge_type: Optional[str] = None,
    ) -> List[CitationNode]:
        """Return the top-*n* nodes ranked by in-degree (most-cited first).

        Nodes with a high in-degree are considered *authority hubs* — they
        are referenced (cited or supported) by many other nodes.

        Args:
            top_n:     Maximum number of hubs to return.
            edge_type: If given, only edges of this type are counted.
                       Otherwise all incoming edges are counted.

        Returns:
            List of :class:`CitationNode` objects sorted by descending
            in-degree, truncated to *top_n*.
        """
        if top_n <= 0:
            return []

        if edge_type is None:
            # Use the full in-degree from networkx
            scored: List[Tuple[str, int]] = [
                (node_id, deg)
                for node_id, deg in self._graph.in_degree()
            ]
        else:
            # Count only edges of the requested type
            type_edges = self.edges_of_type(edge_type)
            in_count: Dict[str, int] = {nid: 0 for nid in self._nodes}
            for _src, tgt in type_edges:
                in_count[tgt] = in_count.get(tgt, 0) + 1
            scored = list(in_count.items())

        scored.sort(key=lambda x: x[1], reverse=True)
        return [self._nodes[nid] for nid, _ in scored[:top_n] if nid in self._nodes]

    def orphan_nodes(self, edge_type: Optional[str] = None) -> List[CitationNode]:
        """Return nodes that have **no** incoming or outgoing edges.

        An orphan is a node that is completely isolated: it neither cites
        any other node nor is cited by any other node.

        Args:
            edge_type: If given, a node is an orphan only with respect to
                       edges of this type (it may still have other edges).

        Returns:
            List of isolated :class:`CitationNode` objects.
        """
        if edge_type is None:
            # Standard networkx isolates
            isolated_ids = set(nx.isolates(self._graph))
        else:
            # Build a subgraph containing only edges of the requested type
            type_edges = self.edges_of_type(edge_type)
            connected_ids: set[str] = set()
            for src, tgt in type_edges:
                connected_ids.add(src)
                connected_ids.add(tgt)
            isolated_ids = set(self._nodes.keys()) - connected_ids

        return [self._nodes[nid] for nid in isolated_ids if nid in self._nodes]

    def predecessors(self, node_id: str, edge_type: Optional[str] = None) -> List[CitationNode]:
        """Return all nodes that have a directed edge *to* *node_id*.

        Args:
            node_id:   Target node whose predecessors are sought.
            edge_type: Optional filter; only predecessors connected by
                       this edge type are returned.

        Raises:
            KeyError: If *node_id* is not in the graph.
        """
        if node_id not in self._nodes:
            raise KeyError(f"Node '{node_id}' not found in CitationGraph.")
        result: List[CitationNode] = []
        for pred_id in self._graph.predecessors(node_id):
            if edge_type is not None:
                if self._graph[pred_id][node_id].get("edge_type") != edge_type:
                    continue
            result.append(self._nodes[pred_id])
        return result

    def successors(self, node_id: str, edge_type: Optional[str] = None) -> List[CitationNode]:
        """Return all nodes that *node_id* has a directed edge *to*.

        Args:
            node_id:   Source node whose successors are sought.
            edge_type: Optional filter.

        Raises:
            KeyError: If *node_id* is not in the graph.
        """
        if node_id not in self._nodes:
            raise KeyError(f"Node '{node_id}' not found in CitationGraph.")
        result: List[CitationNode] = []
        for succ_id in self._graph.successors(node_id):
            if edge_type is not None:
                if self._graph[node_id][succ_id].get("edge_type") != edge_type:
                    continue
            result.append(self._nodes[succ_id])
        return result

    def subgraph_by_hop(self, hop_index: int) -> "CitationGraph":
        """Return a new :class:`CitationGraph` containing only nodes from *hop_index*.

        Edges are preserved where *both* endpoints belong to the hop.

        Args:
            hop_index: The hop index to filter by.

        Returns:
            A new :class:`CitationGraph` instance.
        """
        sub = CitationGraph()
        hop_nodes = [n for n in self._nodes.values() if n.hop_index == hop_index]
        sub.add_nodes(hop_nodes)
        hop_ids = {n.id for n in hop_nodes}
        for src, tgt, data in self._graph.edges(data=True):
            if src in hop_ids and tgt in hop_ids:
                sub.add_edge(src, tgt, edge_type=data["edge_type"])
        return sub

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_hop_chain(cls, hop_chain: object) -> "CitationGraph":
        """Build a :class:`CitationGraph` from a :class:`~research_agent.engine.HopChain`.

        Each source in the chain is wrapped in a :class:`CitationNode` and
        added to the graph.  No edges are inserted automatically (use
        :meth:`add_edge` after creation to wire relationships).

        Args:
            hop_chain: A :class:`~research_agent.engine.HopChain` instance.

        Returns:
            A populated :class:`CitationGraph`.
        """
        from research_agent.models import CitationNode as CN

        graph = cls()
        for hop in hop_chain.hops:
            for source in hop.sources:
                node = CN(
                    source=source,
                    hop_index=hop.hop_index,
                    claim_snippet=source.snippet,
                )
                graph.add_node(node)
        return graph
