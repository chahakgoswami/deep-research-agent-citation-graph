"""ContradictionDetector module for the Research Agent.

Compares claim snippets across CitationNodes using:
1. Keyword/negation heuristics (checks for explicit contradiction signal words).
2. Mocked embedding similarity via a deterministic fixture map.

When a contradiction is detected between two nodes, both nodes' ``contradicts``
lists are updated and a ``ContradictionRecord`` is produced with an explanation
string.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from research_agent.models import CitationNode

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Words / phrases that, when present in a snippet, signal a potentially
# contrary stance.  The detector checks whether two snippets from the same
# topic use opposing sentiment around these terms.
_NEGATION_WORDS: List[str] = [
    "not",
    "no",
    "never",
    "false",
    "hoax",
    "myth",
    "wrong",
    "misleading",
    "manipulated",
    "disagree",
    "disprove",
    "debunked",
    "contrary",
    "incorrect",
    "doubt",
]

# Keyword clusters: if two snippets both mention a keyword from the same
# cluster, they are considered to be speaking about the same topic and are
# therefore candidates for contradiction checking.
_TOPIC_CLUSTERS: Dict[str, List[str]] = {
    "climate": ["climate", "warming", "temperature", "CO2", "emissions", "greenhouse"],
    "ai": ["AI", "artificial intelligence", "machine learning", "neural", "model"],
    "quantum": ["quantum", "qubit", "superposition", "entanglement"],
}

# Similarity threshold below which two snippets are considered contradictory
# when they also share a topic cluster.
_SIMILARITY_CONTRADICTION_THRESHOLD = 0.35

# Default fixture path for the mock embedding similarity map.
_DEFAULT_SIMILARITY_FIXTURE = (
    Path(__file__).parent.parent.parent / "tests" / "fixtures" / "embedding_similarity.json"
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ContradictionRecord:
    """Represents a detected contradiction between two :class:`CitationNode` objects."""

    node_a_id: str
    node_b_id: str
    explanation: str
    similarity_score: float = 0.0
    heuristic_flags: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# MockEmbeddingStore
# ---------------------------------------------------------------------------


class MockEmbeddingStore:
    """Deterministic mock that returns pre-defined cosine-similarity scores.

    The fixture is a JSON object mapping ``"url_a::url_b"`` (in sorted order)
    to a float similarity score in [0, 1].

    If a pair is not found in the fixture the store falls back to a
    keyword-overlap Jaccard similarity between the two snippets.
    """

    def __init__(self, fixture_path: Optional[Path] = None) -> None:
        import json

        self._scores: Dict[str, float] = {}
        path = fixture_path or _DEFAULT_SIMILARITY_FIXTURE
        if path.exists():
            with path.open("r", encoding="utf-8") as fh:
                raw = json.load(fh)
            if not isinstance(raw, dict):
                raise ValueError("Embedding similarity fixture must be a JSON object.")
            self._scores = {k: float(v) for k, v in raw.items()}

    def similarity(self, node_a: CitationNode, node_b: CitationNode) -> float:
        """Return the similarity score for *node_a* and *node_b*.

        Looks up the fixture by ``url_a::url_b`` key (URLs sorted so the
        lookup is order-independent).  Falls back to Jaccard word overlap.
        """
        key = self._make_key(node_a.source.url, node_b.source.url)
        if key in self._scores:
            return self._scores[key]
        return self._jaccard(node_a.claim_snippet, node_b.claim_snippet)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_key(url_a: str, url_b: str) -> str:
        urls = sorted([url_a, url_b])
        return f"{urls[0]}::{urls[1]}"

    @staticmethod
    def _jaccard(text_a: str, text_b: str) -> float:
        """Word-level Jaccard similarity as a fallback similarity measure."""
        words_a = set(re.findall(r"\w+", text_a.lower()))
        words_b = set(re.findall(r"\w+", text_b.lower()))
        if not words_a and not words_b:
            return 1.0
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union)


# ---------------------------------------------------------------------------
# ContradictionDetector
# ---------------------------------------------------------------------------


class ContradictionDetector:
    """Detects contradictions between :class:`~research_agent.models.CitationNode` objects.

    Two nodes are flagged as contradictory when **any** of the following
    conditions hold:

    1. **Negation heuristic** – one snippet contains a strong negation word
       (e.g. *hoax*, *manipulated*, *false*) and both snippets mention the
       same topic cluster keyword(s).

    2. **Low embedding similarity** – the mocked embedding store reports a
       similarity score below :data:`_SIMILARITY_CONTRADICTION_THRESHOLD` AND
       both snippets share a common topic cluster.

    Usage::

        detector = ContradictionDetector()
        records = detector.detect(nodes)
        # Each record describes one contradicting pair.
        # The CitationNode.contradicts lists are also updated in-place.
    """

    def __init__(
        self,
        embedding_store: Optional[MockEmbeddingStore] = None,
        similarity_threshold: float = _SIMILARITY_CONTRADICTION_THRESHOLD,
        fixture_path: Optional[Path] = None,
    ) -> None:
        self._store: MockEmbeddingStore = embedding_store or MockEmbeddingStore(
            fixture_path=fixture_path
        )
        self._threshold: float = similarity_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self, nodes: List[CitationNode]
    ) -> List[ContradictionRecord]:
        """Compare every pair of *nodes* and return contradiction records.

        Also mutates each node's ``contradicts`` list so the graph carries
        contradiction relationships directly.

        Args:
            nodes: A list of :class:`~research_agent.models.CitationNode`
                   objects, each expected to have a non-empty
                   ``claim_snippet``.

        Returns:
            A list of :class:`ContradictionRecord` objects (one per
            contradicting pair, unordered).
        """
        records: List[ContradictionRecord] = []

        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                node_a = nodes[i]
                node_b = nodes[j]
                record = self._compare(node_a, node_b)
                if record is not None:
                    records.append(record)
                    # Update the nodes' contradicts lists (avoid duplicates)
                    if node_b.id not in node_a.contradicts:
                        node_a.contradicts.append(node_b.id)
                    if node_a.id not in node_b.contradicts:
                        node_b.contradicts.append(node_a.id)

        return records

    # ------------------------------------------------------------------
    # Pair-level comparison
    # ------------------------------------------------------------------

    def _compare(
        self, node_a: CitationNode, node_b: CitationNode
    ) -> Optional[ContradictionRecord]:
        """Return a :class:`ContradictionRecord` if the pair contradicts, else None."""
        snippet_a = node_a.claim_snippet
        snippet_b = node_b.claim_snippet

        # If either snippet is empty we cannot meaningfully compare.
        if not snippet_a.strip() or not snippet_b.strip():
            return None

        shared_clusters = self._shared_topic_clusters(snippet_a, snippet_b)

        flags: List[str] = []
        explanation_parts: List[str] = []

        # --- Heuristic 1: negation ---
        neg_a = self._has_negation(snippet_a)
        neg_b = self._has_negation(snippet_b)
        # A contradiction is suggested when ONE side negates (not both, as
        # two mutually-negating claims could just be two denials).
        if (neg_a != neg_b) and shared_clusters:
            flags.append("negation")
            negating_side = "source A" if neg_a else "source B"
            explanation_parts.append(
                f"Negation keyword detected in {negating_side} on shared "
                f"topic(s): {', '.join(shared_clusters)}."
            )

        # --- Heuristic 2: low embedding similarity on shared topic ---
        similarity = self._store.similarity(node_a, node_b)
        if similarity < self._threshold and shared_clusters:
            flags.append("low_similarity")
            explanation_parts.append(
                f"Low semantic similarity ({similarity:.3f} < {self._threshold}) "
                f"on shared topic(s): {', '.join(shared_clusters)}."
            )

        if not flags:
            return None

        explanation = " | ".join(explanation_parts)
        return ContradictionRecord(
            node_a_id=node_a.id,
            node_b_id=node_b.id,
            explanation=explanation,
            similarity_score=similarity,
            heuristic_flags=flags,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _has_negation(text: str) -> bool:
        """Return True if *text* contains any negation/contradiction keyword."""
        text_lower = text.lower()
        return any(word in text_lower for word in _NEGATION_WORDS)

    @staticmethod
    def _shared_topic_clusters(text_a: str, text_b: str) -> List[str]:
        """Return the names of topic clusters that appear in *both* texts."""
        a_lower = text_a.lower()
        b_lower = text_b.lower()
        shared: List[str] = []
        for cluster_name, keywords in _TOPIC_CLUSTERS.items():
            in_a = any(kw.lower() in a_lower for kw in keywords)
            in_b = any(kw.lower() in b_lower for kw in keywords)
            if in_a and in_b:
                shared.append(cluster_name)
        return shared
