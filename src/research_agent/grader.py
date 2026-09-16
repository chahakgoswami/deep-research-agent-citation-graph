"""Source grading module for the Research Agent.

Scores each retrieved Source on three simulated signals:
  - domain_authority  (0-100, fixture-provided)
  - recency           (derived from published_at date)
  - citation_count    (fixture-provided)

The weighted composite score is mapped to a letter grade (A-F) and a
confidence value, then attached to a CitationNode.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from research_agent.models import CitationNode, Source

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Weights must sum to 1.0
_WEIGHT_DOMAIN_AUTHORITY = 0.40
_WEIGHT_RECENCY = 0.30
_WEIGHT_CITATION = 0.30

# Grade thresholds (composite score 0-100)
_GRADE_THRESHOLDS: List[Tuple[float, str]] = [
    (85.0, "A"),
    (70.0, "B"),
    (55.0, "C"),
    (40.0, "D"),
    (0.0,  "F"),
]

# Confidence is reduced when signals are missing
_CONFIDENCE_PENALTY_PER_MISSING_SIGNAL = 0.15

# Citation count normalisation ceiling: sources at or above this count
# receive the maximum citation score.
_CITATION_COUNT_CEILING = 10_000

# Recency: a source published today scores 100; one published >= this many
# days ago scores 0.  Linear interpolation in between.
_RECENCY_MAX_AGE_DAYS = 365 * 5  # 5 years


# ---------------------------------------------------------------------------
# SourceGrader
# ---------------------------------------------------------------------------


class SourceGrader:
    """Assigns a letter grade and confidence score to CitationNodes.

    Usage::

        grader = SourceGrader()
        node = grader.grade(citation_node)
        # node.grade  -> e.g. 'A'
        # node.confidence -> e.g. 0.85

    The grader mutates and returns the same CitationNode object so it can
    be used in-place or in a pipeline.
    """

    def __init__(self, reference_date: Optional[datetime] = None) -> None:
        """Create a SourceGrader.

        Args:
            reference_date: The date to treat as "today" when computing
                recency scores.  Defaults to ``datetime.utcnow()``.  Pass
                an explicit value in tests for deterministic behaviour.
        """
        self._reference_date: datetime = reference_date or datetime.now(timezone.utc)
        # Ensure the reference date is timezone-aware
        if self._reference_date.tzinfo is None:
            self._reference_date = self._reference_date.replace(tzinfo=timezone.utc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def grade(self, node: CitationNode) -> CitationNode:
        """Grade *node* in-place and return it.

        Computes sub-scores for domain authority, recency, and citation
        count, blends them into a composite score, maps it to a letter
        grade, and attaches grade + confidence to the node.

        Args:
            node: A :class:`~research_agent.models.CitationNode` instance.

        Returns:
            The same *node* with ``grade`` and ``confidence`` populated.
        """
        source = node.source
        da_score, da_present = self._domain_authority_score(source)
        re_score, re_present = self._recency_score(source)
        ci_score, ci_present = self._citation_score(source)

        # Count missing signals for confidence penalty
        missing = sum(1 for present in (da_present, re_present, ci_present) if not present)
        confidence = max(0.0, 1.0 - missing * _CONFIDENCE_PENALTY_PER_MISSING_SIGNAL)

        # Weighted composite (missing signals contribute their default 50/100
        # so they do not catastrophically punish the score, but the lower
        # confidence reflects the uncertainty).
        composite = (
            da_score * _WEIGHT_DOMAIN_AUTHORITY
            + re_score * _WEIGHT_RECENCY
            + ci_score * _WEIGHT_CITATION
        )

        grade = self._composite_to_grade(composite)

        node.grade = grade
        node.confidence = round(confidence, 4)
        return node

    def grade_all(
        self,
        nodes: List[CitationNode],
        deduplicate: bool = True,
    ) -> List[CitationNode]:
        """Grade a list of CitationNodes, optionally de-duplicating by URL.

        When *deduplicate* is ``True``, only the first occurrence of each
        URL is graded; subsequent duplicates receive the same grade/
        confidence as the first occurrence (since they share identical
        source data) and the same ``grade`` / ``confidence`` are copied
        to them.

        Args:
            nodes:       List of :class:`~research_agent.models.CitationNode`.
            deduplicate: Whether to de-duplicate by source URL.

        Returns:
            The same list, each node having ``grade`` and ``confidence``
            populated.
        """
        seen: Dict[str, CitationNode] = {}  # url -> first graded node

        for node in nodes:
            url = node.source.url
            if deduplicate and url in seen:
                # Copy grade/confidence from the canonical node
                canonical = seen[url]
                node.grade = canonical.grade
                node.confidence = canonical.confidence
            else:
                self.grade(node)
                if deduplicate:
                    seen[url] = node

        return nodes

    # ------------------------------------------------------------------
    # Sub-scorers (each returns (score_0_to_100, signal_was_present))
    # ------------------------------------------------------------------

    def _domain_authority_score(self, source: Source) -> Tuple[float, bool]:
        """Return (score, present) for domain authority."""
        if source.domain_authority is None:
            return 50.0, False  # neutral default
        # domain_authority is already 0-100
        return float(source.domain_authority), True

    def _recency_score(self, source: Source) -> Tuple[float, bool]:
        """Return (score, present) for publication recency."""
        if source.published_at is None:
            return 50.0, False  # neutral default

        pub = source.published_at
        # Make timezone-aware if naive
        if pub.tzinfo is None:
            pub = pub.replace(tzinfo=timezone.utc)

        age_days = (self._reference_date - pub).days
        if age_days < 0:
            # Future-dated source — treat as maximally recent
            age_days = 0

        score = max(0.0, 1.0 - age_days / _RECENCY_MAX_AGE_DAYS) * 100.0
        return score, True

    def _citation_score(self, source: Source) -> Tuple[float, bool]:
        """Return (score, present) for citation count."""
        if source.citation_count is None:
            return 50.0, False  # neutral default

        score = min(source.citation_count / _CITATION_COUNT_CEILING, 1.0) * 100.0
        return score, True

    # ------------------------------------------------------------------
    # Grade mapping
    # ------------------------------------------------------------------

    @staticmethod
    def _composite_to_grade(composite: float) -> str:
        """Map a composite score (0-100) to a letter grade string."""
        for threshold, grade in _GRADE_THRESHOLDS:
            if composite >= threshold:
                return grade
        return "F"
