"""Multi-hop reasoning engine for the Research Agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from research_agent.backends.mock_search import MockSearchBackend
from research_agent.models import Hop, ResearchQuery, Source


@dataclass
class HopChain:
    """Container for the sequence of hops produced during a research session."""

    query: ResearchQuery
    hops: List[Hop] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def depth(self) -> int:
        """Number of hops executed so far."""
        return len(self.hops)

    @property
    def all_sources(self) -> List[Source]:
        """Flat list of every source retrieved across all hops."""
        sources: List[Source] = []
        for hop in self.hops:
            sources.extend(hop.sources)
        return sources

    def get_hop(self, index: int) -> Hop:
        """Return the hop at *index* (raises IndexError if out of range)."""
        return self.hops[index]

    def sub_queries(self) -> List[str]:
        """Return the list of sub-queries executed, in hop order."""
        return [hop.sub_query for hop in self.hops]


class SubQuestionExpander:
    """Generates follow-up sub-questions from a parent query / previous hop.

    This is a deterministic, mock implementation that derives sub-questions
    by appending templated suffixes so the engine can be tested without an
    LLM.  A real implementation would call an LLM here.
    """

    # Templates applied in round-robin order across hops
    _TEMPLATES: List[str] = [
        "What are the primary causes of {topic}?",
        "What is the latest research on {topic}?",
        "What are the key effects of {topic}?",
        "How do experts define {topic}?",
        "What controversies exist around {topic}?",
        "What future trends are predicted for {topic}?",
        "How is {topic} measured or quantified?",
        "What policy responses address {topic}?",
        "What historical context exists for {topic}?",
        "What open questions remain about {topic}?",
    ]

    def expand(self, query_text: str, hop_index: int) -> str:
        """Return a sub-question for *hop_index* derived from *query_text*.

        Args:
            query_text: The original or most-recent query text used as the topic.
            hop_index:  Zero-based index of the hop being generated.

        Returns:
            A sub-question string.
        """
        template = self._TEMPLATES[hop_index % len(self._TEMPLATES)]
        return template.format(topic=query_text)


class MultiHopEngine:
    """Drives the iterative multi-hop research process.

    For each hop (up to *max_hops* as specified on the :class:`ResearchQuery`),
    the engine:

    1. Generates a sub-question via :class:`SubQuestionExpander`.
    2. Searches the backend with that sub-question.
    3. Appends a :class:`Hop` record to the :class:`HopChain`.

    The process terminates once ``max_hops`` hops have been executed.
    """

    def __init__(
        self,
        backend: Optional[MockSearchBackend] = None,
        expander: Optional[SubQuestionExpander] = None,
        top_k: int = 5,
    ) -> None:
        self.backend: MockSearchBackend = backend or MockSearchBackend()
        self.expander: SubQuestionExpander = expander or SubQuestionExpander()
        self.top_k: int = top_k

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, query: ResearchQuery) -> HopChain:
        """Execute the full multi-hop research loop for *query*.

        Args:
            query: A validated :class:`ResearchQuery` instance.

        Returns:
            A :class:`HopChain` containing one :class:`Hop` per iteration.
        """
        chain = HopChain(query=query)

        for hop_index in range(query.max_hops):
            sub_query = self.expander.expand(query.text, hop_index)
            sources = self.backend.search(sub_query, top_k=self.top_k)
            hop = Hop(
                hop_index=hop_index,
                sub_query=sub_query,
                parent_query_id=query.id,
                sources=sources,
            )
            chain.hops.append(hop)

        return chain
