"""ReportCompiler module for the Research Agent.

Walks the finalized CitationGraph and assembles a structured final report
with sections:
  - Summary
  - Findings (per-hop)
  - Contradictions
  - Graded Sources

Inlines numbered citations, renders to plain text and Markdown, and
requires human-in-the-loop confirmation before writing output files.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from research_agent.citation_graph import CitationGraph
from research_agent.contradiction_detector import ContradictionRecord
from research_agent.models import CitationNode, ResearchQuery

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class ReportSection:
    """A single labelled section of the research report."""

    title: str
    body: str


@dataclass
class ResearchReport:
    """Structured representation of the compiled research report."""

    query: ResearchQuery
    sections: List[ReportSection] = field(default_factory=list)
    # Ordered list of CitationNodes used as numbered references
    citations: List[CitationNode] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    def to_plain_text(self) -> str:
        """Render the report as plain text."""
        lines: List[str] = []
        lines.append(f"RESEARCH REPORT")
        lines.append(f"Query: {self.query.text}")
        lines.append("=" * 72)
        lines.append("")

        for section in self.sections:
            lines.append(section.title.upper())
            lines.append("-" * 40)
            lines.append(section.body)
            lines.append("")

        # References
        if self.citations:
            lines.append("REFERENCES")
            lines.append("-" * 40)
            for i, node in enumerate(self.citations, start=1):
                src = node.source
                grade_str = f" [Grade: {node.grade}]" if node.grade else ""
                lines.append(
                    f"[{i}] {src.title or src.url} — {src.url}{grade_str}"
                )
            lines.append("")

        return "\n".join(lines)

    def to_markdown(self) -> str:
        """Render the report as Markdown."""
        lines: List[str] = []
        lines.append(f"# Research Report")
        lines.append(f"")
        lines.append(f"**Query:** {self.query.text}")
        lines.append("")

        for section in self.sections:
            lines.append(f"## {section.title}")
            lines.append("")
            lines.append(section.body)
            lines.append("")

        # References
        if self.citations:
            lines.append("## References")
            lines.append("")
            for i, node in enumerate(self.citations, start=1):
                src = node.source
                grade_str = f" *(Grade: {node.grade})*" if node.grade else ""
                lines.append(
                    f"{i}. [{src.title or src.url}]({src.url}){grade_str}"
                )
            lines.append("")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ReportCompiler
# ---------------------------------------------------------------------------


class ReportCompiler:
    """Compiles a :class:`ResearchReport` from a finalized :class:`CitationGraph`.

    Usage::

        compiler = ReportCompiler()
        report = compiler.compile(
            graph=graph,
            query=query,
            contradiction_records=records,
        )
        # Render without writing to disk:
        text = report.to_plain_text()
        md   = report.to_markdown()

        # Write to disk (with human confirmation prompt):
        compiler.write(
            report=report,
            output_dir=Path("./output"),
            output_formats=["txt", "md"],
        )
    """

    def __init__(self, wrap_width: int = 80) -> None:
        self._wrap_width = wrap_width

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compile(
        self,
        graph: CitationGraph,
        query: ResearchQuery,
        contradiction_records: Optional[List[ContradictionRecord]] = None,
    ) -> ResearchReport:
        """Walk *graph* and produce a :class:`ResearchReport`.

        Args:
            graph:                  The finalized :class:`CitationGraph`.
            query:                  The original :class:`ResearchQuery`.
            contradiction_records:  Optional list of detected contradictions
                                    from :class:`ContradictionDetector`.

        Returns:
            A fully populated :class:`ResearchReport`.
        """
        contradiction_records = contradiction_records or []
        all_nodes = graph.all_nodes()

        # Build the ordered citation list (all nodes, deduplicated by URL,
        # sorted by hop_index then by grade descending).
        citations = self._build_citation_list(all_nodes)

        # Create a quick lookup from node id to citation number.
        citation_index = {node.id: i + 1 for i, node in enumerate(citations)}

        report = ResearchReport(query=query, citations=citations)

        # --- Section 1: Summary ---
        report.sections.append(
            ReportSection(
                title="Summary",
                body=self._build_summary(
                    query, all_nodes, contradiction_records
                ),
            )
        )

        # --- Section 2: Findings (grouped by hop) ---
        report.sections.append(
            ReportSection(
                title="Findings",
                body=self._build_findings(graph, citation_index),
            )
        )

        # --- Section 3: Contradictions ---
        report.sections.append(
            ReportSection(
                title="Contradictions",
                body=self._build_contradictions(
                    contradiction_records, graph, citation_index
                ),
            )
        )

        # --- Section 4: Graded Sources ---
        report.sections.append(
            ReportSection(
                title="Graded Sources",
                body=self._build_graded_sources(citations, citation_index),
            )
        )

        return report

    def write(
        self,
        report: ResearchReport,
        output_dir: Path,
        output_formats: Optional[List[str]] = None,
        *,
        _confirm_fn=None,
    ) -> List[Path]:
        """Render and write the report to *output_dir* after human confirmation.

        Args:
            report:          The compiled :class:`ResearchReport`.
            output_dir:      Directory where output files are written.
            output_formats:  List of format strings: ``'txt'`` and/or ``'md'``.
                             Defaults to ``['txt', 'md']``.
            _confirm_fn:     Internal override for the confirmation prompt
                             (used in tests to avoid blocking on stdin).
                             Must be a callable that returns a str (the user
                             input).

        Returns:
            List of :class:`Path` objects for the files written.

        Raises:
            PermissionError:  If the user declines to write the files.
        """
        if output_formats is None:
            output_formats = ["txt", "md"]

        output_dir.mkdir(parents=True, exist_ok=True)

        # Build filenames
        safe_name = self._safe_filename(report.query.text)
        planned: List[tuple[Path, str]] = []
        if "txt" in output_formats:
            planned.append((output_dir / f"{safe_name}.txt", report.to_plain_text()))
        if "md" in output_formats:
            planned.append((output_dir / f"{safe_name}.md", report.to_markdown()))

        if not planned:
            return []

        # Human-in-the-loop confirmation
        file_list = "\n".join(f"  {p}" for p, _ in planned)
        prompt = (
            f"\nThe following files will be written:\n{file_list}\n"
            f"Confirm? [y/N]: "
        )

        if _confirm_fn is not None:
            answer = _confirm_fn(prompt)
        else:  # pragma: no cover
            answer = input(prompt)

        if answer.strip().lower() not in ("y", "yes"):
            raise PermissionError(
                "Report write cancelled by user."
            )

        written: List[Path] = []
        for path, content in planned:
            path.write_text(content, encoding="utf-8")
            written.append(path)

        return written

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _build_summary(
        self,
        query: ResearchQuery,
        nodes: List[CitationNode],
        contradictions: List[ContradictionRecord],
    ) -> str:
        total = len(nodes)
        # Count unique URLs
        unique_urls = len({n.source.url for n in nodes})
        # Count graded nodes
        graded = sum(1 for n in nodes if n.grade is not None)
        contradiction_count = len(contradictions)

        grade_counts: dict[str, int] = {}
        for n in nodes:
            if n.grade:
                grade_counts[n.grade] = grade_counts.get(n.grade, 0) + 1

        grade_summary = ", ".join(
            f"{g}: {c}" for g, c in sorted(grade_counts.items())
        )

        lines = [
            f"This report summarises research on the query: '{query.text}'.",
            f"A total of {total} sources ({unique_urls} unique URLs) were retrieved "
            f"across {query.max_hops} reasoning hop(s).",
        ]
        if graded:
            lines.append(f"{graded} sources were graded ({grade_summary}).")
        if contradiction_count:
            lines.append(
                f"{contradiction_count} contradiction(s) were detected between sources."
            )
        else:
            lines.append("No contradictions were detected between sources.")

        return "\n".join(lines)

    def _build_findings(
        self,
        graph: CitationGraph,
        citation_index: dict[str, int],
    ) -> str:
        # Group nodes by hop
        hop_buckets: dict[int, List[CitationNode]] = {}
        for node in graph.all_nodes():
            hop_buckets.setdefault(node.hop_index, []).append(node)

        if not hop_buckets:
            return "No findings were retrieved."

        parts: List[str] = []
        for hop_idx in sorted(hop_buckets.keys()):
            nodes = hop_buckets[hop_idx]
            parts.append(f"Hop {hop_idx + 1}:")
            for node in nodes:
                ref_num = citation_index.get(node.id, "?")
                snippet = node.claim_snippet or node.source.snippet
                snippet = textwrap.shorten(snippet, width=120, placeholder="...")
                parts.append(f"  [{ref_num}] {snippet}")
        return "\n".join(parts)

    def _build_contradictions(
        self,
        records: List[ContradictionRecord],
        graph: CitationGraph,
        citation_index: dict[str, int],
    ) -> str:
        if not records:
            return "No contradictions detected."

        parts: List[str] = []
        for i, rec in enumerate(records, start=1):
            try:
                node_a = graph.get_node(rec.node_a_id)
                node_b = graph.get_node(rec.node_b_id)
            except KeyError:
                continue
            ref_a = citation_index.get(node_a.id, "?")
            ref_b = citation_index.get(node_b.id, "?")
            parts.append(
                f"{i}. Sources [{ref_a}] and [{ref_b}] contradict each other."
            )
            parts.append(f"   Explanation: {rec.explanation}")
            parts.append(
                f"   Similarity score: {rec.similarity_score:.3f}"
            )
        return "\n".join(parts)

    def _build_graded_sources(
        self,
        citations: List[CitationNode],
        citation_index: dict[str, int],
    ) -> str:
        if not citations:
            return "No sources available."

        parts: List[str] = []
        for node in citations:
            ref = citation_index.get(node.id, "?")
            src = node.source
            grade_str = node.grade or "N/A"
            confidence_str = (
                f"{node.confidence:.2f}" if node.confidence is not None else "N/A"
            )
            da_str = (
                f"{src.domain_authority:.1f}" if src.domain_authority is not None else "N/A"
            )
            cc_str = (
                str(src.citation_count) if src.citation_count is not None else "N/A"
            )
            title = src.title or src.url
            parts.append(
                f"[{ref}] {title}"
            )
            parts.append(
                f"      URL: {src.url}"
            )
            parts.append(
                f"      Grade: {grade_str}  Confidence: {confidence_str}  "
                f"Domain Authority: {da_str}  Citations: {cc_str}"
            )
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_citation_list(nodes: List[CitationNode]) -> List[CitationNode]:
        """Deduplicate by URL, sort by hop_index then grade."""
        grade_order = {"A": 0, "B": 1, "C": 2, "D": 3, "F": 4, None: 5}
        seen_urls: set[str] = set()
        unique: List[CitationNode] = []
        for node in nodes:
            url = node.source.url
            if url not in seen_urls:
                seen_urls.add(url)
                unique.append(node)
        unique.sort(
            key=lambda n: (n.hop_index, grade_order.get(n.grade, 5))
        )
        return unique

    @staticmethod
    def _safe_filename(text: str, max_len: int = 50) -> str:
        """Convert a query string to a safe filename stem."""
        import re
        safe = re.sub(r"[^\w\s-]", "", text.lower())
        safe = re.sub(r"[\s_-]+", "_", safe).strip("_")
        return safe[:max_len] or "report"
