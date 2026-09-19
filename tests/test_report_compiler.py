"""Integration tests for the ReportCompiler module."""

from __future__ import annotations

from pathlib import Path
from typing import List

import pytest

from research_agent.backends.mock_search import MockSearchBackend
from research_agent.citation_graph import CitationGraph, EDGE_CITED_BY, EDGE_SUPPORTS
from research_agent.contradiction_detector import ContradictionDetector, MockEmbeddingStore
from research_agent.engine import MultiHopEngine
from research_agent.grader import SourceGrader
from research_agent.models import CitationNode, ResearchQuery, Source
from research_agent.report_compiler import ReportCompiler, ResearchReport, ReportSection

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "mock_results.json"
SIMILARITY_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "embedding_similarity.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_source(url: str = "https://example.com", **kwargs) -> Source:
    return Source(
        url=url,
        title=kwargs.get("title", "Test Article"),
        snippet=kwargs.get("snippet", "A test snippet."),
        domain=kwargs.get("domain", "example.com"),
        domain_authority=kwargs.get("domain_authority", 70.0),
        citation_count=kwargs.get("citation_count", 100),
    )


def make_node(
    url: str = "https://example.com",
    hop_index: int = 0,
    grade: str | None = "B",
    confidence: float | None = 0.85,
    **source_kwargs,
) -> CitationNode:
    node = CitationNode(
        source=make_source(url=url, **source_kwargs),
        hop_index=hop_index,
        grade=grade,
        confidence=confidence,
        claim_snippet=source_kwargs.get("snippet", "A test snippet."),
    )
    return node


def build_simple_graph(num_nodes: int = 3) -> tuple[CitationGraph, list[CitationNode]]:
    graph = CitationGraph()
    nodes = [
        make_node(
            url=f"https://example.com/{i}",
            hop_index=i % 2,
            title=f"Article {i}",
            snippet=f"Finding number {i} about the topic.",
        )
        for i in range(num_nodes)
    ]
    graph.add_nodes(nodes)
    return graph, nodes


def build_full_pipeline_graph(
    query_text: str = "climate change",
    max_hops: int = 2,
    top_k: int = 3,
) -> tuple[CitationGraph, ResearchQuery, list]:
    """Run the full pipeline and return graph, query, contradiction records."""
    backend = MockSearchBackend(fixture_path=FIXTURE_PATH)
    engine = MultiHopEngine(backend=backend, top_k=top_k)
    query = ResearchQuery(text=query_text, max_hops=max_hops)
    chain = engine.run(query)

    graph = CitationGraph.from_hop_chain(chain)

    grader = SourceGrader()
    grader.grade_all(graph.all_nodes())

    store = MockEmbeddingStore(fixture_path=SIMILARITY_FIXTURE_PATH)
    detector = ContradictionDetector(embedding_store=store)
    records = detector.detect(graph.all_nodes())

    return graph, query, records


# ---------------------------------------------------------------------------
# ReportSection tests
# ---------------------------------------------------------------------------


class TestReportSection:
    def test_creation(self):
        sec = ReportSection(title="Summary", body="This is the summary.")
        assert sec.title == "Summary"
        assert sec.body == "This is the summary."


# ---------------------------------------------------------------------------
# ResearchReport rendering tests
# ---------------------------------------------------------------------------


class TestResearchReportRendering:
    def _make_report(self) -> ResearchReport:
        query = ResearchQuery(text="quantum computing", max_hops=2)
        graph, nodes = build_simple_graph(3)
        compiler = ReportCompiler()
        return compiler.compile(graph=graph, query=query)

    def test_to_plain_text_returns_string(self):
        report = self._make_report()
        text = report.to_plain_text()
        assert isinstance(text, str)
        assert len(text) > 0

    def test_to_plain_text_contains_query(self):
        report = self._make_report()
        text = report.to_plain_text()
        assert "quantum computing" in text

    def test_to_plain_text_contains_research_report_header(self):
        report = self._make_report()
        text = report.to_plain_text()
        assert "RESEARCH REPORT" in text

    def test_to_plain_text_contains_all_section_titles(self):
        report = self._make_report()
        text = report.to_plain_text()
        for section in report.sections:
            assert section.title.upper() in text

    def test_to_plain_text_contains_references(self):
        report = self._make_report()
        text = report.to_plain_text()
        assert "REFERENCES" in text
        assert "[1]" in text

    def test_to_markdown_returns_string(self):
        report = self._make_report()
        md = report.to_markdown()
        assert isinstance(md, str)
        assert len(md) > 0

    def test_to_markdown_contains_h1_header(self):
        report = self._make_report()
        md = report.to_markdown()
        assert "# Research Report" in md

    def test_to_markdown_contains_query(self):
        report = self._make_report()
        md = report.to_markdown()
        assert "quantum computing" in md

    def test_to_markdown_contains_h2_sections(self):
        report = self._make_report()
        md = report.to_markdown()
        for section in report.sections:
            assert f"## {section.title}" in md

    def test_to_markdown_contains_numbered_references(self):
        report = self._make_report()
        md = report.to_markdown()
        assert "## References" in md
        assert "1." in md

    def test_to_markdown_contains_urls_as_links(self):
        report = self._make_report()
        md = report.to_markdown()
        for node in report.citations:
            assert f"({node.source.url})" in md

    def test_plain_text_sections_in_order(self):
        report = self._make_report()
        text = report.to_plain_text()
        positions = [text.index(s.title.upper()) for s in report.sections]
        assert positions == sorted(positions)


# ---------------------------------------------------------------------------
# ReportCompiler.compile tests
# ---------------------------------------------------------------------------


class TestReportCompilerCompile:
    def test_compile_returns_research_report(self):
        query = ResearchQuery(text="AI", max_hops=1)
        graph, _ = build_simple_graph(2)
        compiler = ReportCompiler()
        report = compiler.compile(graph=graph, query=query)
        assert isinstance(report, ResearchReport)

    def test_compile_query_stored(self):
        query = ResearchQuery(text="AI safety", max_hops=2)
        graph, _ = build_simple_graph(2)
        compiler = ReportCompiler()
        report = compiler.compile(graph=graph, query=query)
        assert report.query is query

    def test_compile_has_four_sections(self):
        query = ResearchQuery(text="test", max_hops=1)
        graph, _ = build_simple_graph(2)
        compiler = ReportCompiler()
        report = compiler.compile(graph=graph, query=query)
        assert len(report.sections) == 4

    def test_compile_section_titles(self):
        query = ResearchQuery(text="test", max_hops=1)
        graph, _ = build_simple_graph(2)
        compiler = ReportCompiler()
        report = compiler.compile(graph=graph, query=query)
        titles = [s.title for s in report.sections]
        assert titles == ["Summary", "Findings", "Contradictions", "Graded Sources"]

    def test_compile_citations_list_populated(self):
        query = ResearchQuery(text="test", max_hops=1)
        graph, nodes = build_simple_graph(3)
        compiler = ReportCompiler()
        report = compiler.compile(graph=graph, query=query)
        assert len(report.citations) == 3

    def test_compile_citations_deduplicated_by_url(self):
        """Two nodes with the same URL should appear only once in citations."""
        graph = CitationGraph()
        n1 = make_node(url="https://dup.com", hop_index=0)
        n2 = make_node(url="https://dup.com", hop_index=1)  # same URL
        n3 = make_node(url="https://unique.com", hop_index=0)
        graph.add_nodes([n1, n2, n3])
        query = ResearchQuery(text="test", max_hops=2)
        compiler = ReportCompiler()
        report = compiler.compile(graph=graph, query=query)
        assert len(report.citations) == 2

    def test_compile_summary_mentions_source_count(self):
        query = ResearchQuery(text="climate", max_hops=2)
        graph, nodes = build_simple_graph(4)
        compiler = ReportCompiler()
        report = compiler.compile(graph=graph, query=query)
        summary_body = report.sections[0].body
        assert "4" in summary_body

    def test_compile_summary_mentions_query_text(self):
        query = ResearchQuery(text="nuclear fusion", max_hops=1)
        graph, _ = build_simple_graph(1)
        compiler = ReportCompiler()
        report = compiler.compile(graph=graph, query=query)
        assert "nuclear fusion" in report.sections[0].body

    def test_compile_no_contradictions_section_says_none_detected(self):
        query = ResearchQuery(text="test", max_hops=1)
        graph, _ = build_simple_graph(2)
        compiler = ReportCompiler()
        report = compiler.compile(graph=graph, query=query, contradiction_records=[])
        contradiction_section = next(
            s for s in report.sections if s.title == "Contradictions"
        )
        assert "No contradictions detected" in contradiction_section.body

    def test_compile_with_contradictions_section_mentions_refs(self):
        from research_agent.contradiction_detector import ContradictionRecord
        query = ResearchQuery(text="climate", max_hops=1)
        graph, nodes = build_simple_graph(2)
        rec = ContradictionRecord(
            node_a_id=nodes[0].id,
            node_b_id=nodes[1].id,
            explanation="One side negates.",
            similarity_score=0.05,
            heuristic_flags=["negation"],
        )
        compiler = ReportCompiler()
        report = compiler.compile(
            graph=graph, query=query, contradiction_records=[rec]
        )
        contradiction_section = next(
            s for s in report.sections if s.title == "Contradictions"
        )
        assert "[1]" in contradiction_section.body or "[2]" in contradiction_section.body
        assert "One side negates." in contradiction_section.body

    def test_compile_findings_contains_hop_labels(self):
        query = ResearchQuery(text="test", max_hops=2)
        graph = CitationGraph()
        nodes = [
            make_node(url=f"https://hop{h}.com/{i}", hop_index=h)
            for h in range(2)
            for i in range(2)
        ]
        graph.add_nodes(nodes)
        compiler = ReportCompiler()
        report = compiler.compile(graph=graph, query=query)
        findings_body = report.sections[1].body
        assert "Hop 1" in findings_body
        assert "Hop 2" in findings_body

    def test_compile_graded_sources_lists_grades(self):
        query = ResearchQuery(text="test", max_hops=1)
        graph, nodes = build_simple_graph(2)
        compiler = ReportCompiler()
        report = compiler.compile(graph=graph, query=query)
        graded_body = report.sections[3].body
        # Default nodes have grade="B"
        assert "B" in graded_body

    def test_compile_empty_graph_still_produces_report(self):
        query = ResearchQuery(text="empty", max_hops=1)
        graph = CitationGraph()
        compiler = ReportCompiler()
        report = compiler.compile(graph=graph, query=query)
        assert isinstance(report, ResearchReport)
        assert len(report.sections) == 4

    def test_compile_inline_citation_numbers_present_in_findings(self):
        query = ResearchQuery(text="test", max_hops=1)
        graph, nodes = build_simple_graph(3)
        compiler = ReportCompiler()
        report = compiler.compile(graph=graph, query=query)
        findings_body = report.sections[1].body
        assert "[1]" in findings_body


# ---------------------------------------------------------------------------
# ReportCompiler.write — human-in-the-loop tests
# ---------------------------------------------------------------------------


class TestReportCompilerWrite:
    def _make_report_and_compiler(
        self, query_text: str = "test topic"
    ) -> tuple[ResearchReport, ReportCompiler]:
        query = ResearchQuery(text=query_text, max_hops=1)
        graph, _ = build_simple_graph(2)
        compiler = ReportCompiler()
        report = compiler.compile(graph=graph, query=query)
        return report, compiler

    def test_write_creates_txt_file_on_confirm(self, tmp_path: Path):
        report, compiler = self._make_report_and_compiler()
        written = compiler.write(
            report=report,
            output_dir=tmp_path,
            output_formats=["txt"],
            _confirm_fn=lambda prompt: "y",
        )
        assert len(written) == 1
        assert written[0].suffix == ".txt"
        assert written[0].exists()

    def test_write_creates_md_file_on_confirm(self, tmp_path: Path):
        report, compiler = self._make_report_and_compiler()
        written = compiler.write(
            report=report,
            output_dir=tmp_path,
            output_formats=["md"],
            _confirm_fn=lambda prompt: "y",
        )
        assert len(written) == 1
        assert written[0].suffix == ".md"
        assert written[0].exists()

    def test_write_creates_both_formats_on_confirm(self, tmp_path: Path):
        report, compiler = self._make_report_and_compiler()
        written = compiler.write(
            report=report,
            output_dir=tmp_path,
            output_formats=["txt", "md"],
            _confirm_fn=lambda prompt: "y",
        )
        assert len(written) == 2
        suffixes = {p.suffix for p in written}
        assert ".txt" in suffixes
        assert ".md" in suffixes

    def test_write_denied_raises_permission_error(self, tmp_path: Path):
        report, compiler = self._make_report_and_compiler()
        with pytest.raises(PermissionError, match="cancelled by user"):
            compiler.write(
                report=report,
                output_dir=tmp_path,
                output_formats=["txt"],
                _confirm_fn=lambda prompt: "n",
            )

    def test_write_no_files_written_when_denied(self, tmp_path: Path):
        report, compiler = self._make_report_and_compiler()
        try:
            compiler.write(
                report=report,
                output_dir=tmp_path,
                output_formats=["txt", "md"],
                _confirm_fn=lambda prompt: "n",
            )
        except PermissionError:
            pass
        assert list(tmp_path.iterdir()) == []

    def test_write_empty_answer_denied(self, tmp_path: Path):
        report, compiler = self._make_report_and_compiler()
        with pytest.raises(PermissionError):
            compiler.write(
                report=report,
                output_dir=tmp_path,
                output_formats=["txt"],
                _confirm_fn=lambda prompt: "",
            )

    def test_write_yes_uppercase_accepted(self, tmp_path: Path):
        report, compiler = self._make_report_and_compiler()
        written = compiler.write(
            report=report,
            output_dir=tmp_path,
            output_formats=["txt"],
            _confirm_fn=lambda prompt: "YES",
        )
        assert len(written) == 1

    def test_write_txt_content_matches_plain_text(self, tmp_path: Path):
        report, compiler = self._make_report_and_compiler("content check")
        written = compiler.write(
            report=report,
            output_dir=tmp_path,
            output_formats=["txt"],
            _confirm_fn=lambda prompt: "y",
        )
        content = written[0].read_text(encoding="utf-8")
        assert content == report.to_plain_text()

    def test_write_md_content_matches_markdown(self, tmp_path: Path):
        report, compiler = self._make_report_and_compiler("md content check")
        written = compiler.write(
            report=report,
            output_dir=tmp_path,
            output_formats=["md"],
            _confirm_fn=lambda prompt: "y",
        )
        content = written[0].read_text(encoding="utf-8")
        assert content == report.to_markdown()

    def test_write_creates_output_dir_if_missing(self, tmp_path: Path):
        report, compiler = self._make_report_and_compiler()
        new_dir = tmp_path / "subdir" / "nested"
        compiler.write(
            report=report,
            output_dir=new_dir,
            output_formats=["txt"],
            _confirm_fn=lambda prompt: "y",
        )
        assert new_dir.exists()

    def test_write_filename_derived_from_query(self, tmp_path: Path):
        report, compiler = self._make_report_and_compiler("climate change research")
        written = compiler.write(
            report=report,
            output_dir=tmp_path,
            output_formats=["txt"],
            _confirm_fn=lambda prompt: "y",
        )
        assert "climate" in written[0].stem

    def test_write_no_formats_returns_empty_list(self, tmp_path: Path):
        report, compiler = self._make_report_and_compiler()
        written = compiler.write(
            report=report,
            output_dir=tmp_path,
            output_formats=[],
            _confirm_fn=lambda prompt: "y",
        )
        assert written == []

    def test_write_prompt_received_by_confirm_fn(self, tmp_path: Path):
        report, compiler = self._make_report_and_compiler()
        received_prompts = []

        def capture(prompt: str) -> str:
            received_prompts.append(prompt)
            return "y"

        compiler.write(
            report=report,
            output_dir=tmp_path,
            output_formats=["txt"],
            _confirm_fn=capture,
        )
        assert len(received_prompts) == 1
        assert "written" in received_prompts[0].lower() or "confirm" in received_prompts[0].lower()


# ---------------------------------------------------------------------------
# Integration tests: full pipeline → report
# ---------------------------------------------------------------------------


class TestReportCompilerIntegration:
    def test_full_pipeline_climate_change(self, tmp_path: Path):
        graph, query, records = build_full_pipeline_graph(
            query_text="climate change", max_hops=2, top_k=3
        )
        compiler = ReportCompiler()
        report = compiler.compile(
            graph=graph,
            query=query,
            contradiction_records=records,
        )
        assert isinstance(report, ResearchReport)
        assert len(report.citations) > 0
        assert len(report.sections) == 4

    def test_full_pipeline_report_has_graded_citations(self, tmp_path: Path):
        graph, query, records = build_full_pipeline_graph(
            query_text="climate change", max_hops=2, top_k=3
        )
        compiler = ReportCompiler()
        report = compiler.compile(graph=graph, query=query, contradiction_records=records)
        # All citations should have been graded by the pipeline
        for citation in report.citations:
            assert citation.grade is not None

    def test_full_pipeline_txt_report_written(self, tmp_path: Path):
        graph, query, records = build_full_pipeline_graph(
            query_text="artificial intelligence", max_hops=1, top_k=3
        )
        compiler = ReportCompiler()
        report = compiler.compile(graph=graph, query=query, contradiction_records=records)
        written = compiler.write(
            report=report,
            output_dir=tmp_path,
            output_formats=["txt"],
            _confirm_fn=lambda p: "y",
        )
        assert len(written) == 1
        text = written[0].read_text(encoding="utf-8")
        assert "artificial intelligence" in text.lower()

    def test_full_pipeline_md_report_written(self, tmp_path: Path):
        graph, query, records = build_full_pipeline_graph(
            query_text="quantum computing", max_hops=1, top_k=2
        )
        compiler = ReportCompiler()
        report = compiler.compile(graph=graph, query=query, contradiction_records=records)
        written = compiler.write(
            report=report,
            output_dir=tmp_path,
            output_formats=["md"],
            _confirm_fn=lambda p: "y",
        )
        assert len(written) == 1
        md = written[0].read_text(encoding="utf-8")
        assert "# Research Report" in md

    def test_full_pipeline_citation_count_matches_unique_sources(self):
        graph, query, records = build_full_pipeline_graph(
            query_text="climate change", max_hops=2, top_k=3
        )
        compiler = ReportCompiler()
        report = compiler.compile(graph=graph, query=query, contradiction_records=records)
        unique_urls = len({n.source.url for n in graph.all_nodes()})
        assert len(report.citations) == unique_urls

    def test_full_pipeline_contradictions_detected_and_in_report(self):
        graph, query, records = build_full_pipeline_graph(
            query_text="climate change", max_hops=2, top_k=3
        )
        compiler = ReportCompiler()
        report = compiler.compile(
            graph=graph, query=query, contradiction_records=records
        )
        contradiction_section = next(
            s for s in report.sections if s.title == "Contradictions"
        )
        if records:
            assert "No contradictions detected" not in contradiction_section.body
        else:
            assert "No contradictions detected" in contradiction_section.body

    def test_full_pipeline_summary_mentions_contradiction_count(self):
        graph, query, records = build_full_pipeline_graph(
            query_text="climate change", max_hops=2, top_k=3
        )
        compiler = ReportCompiler()
        report = compiler.compile(
            graph=graph, query=query, contradiction_records=records
        )
        summary_body = report.sections[0].body
        if records:
            assert "contradiction" in summary_body.lower()

    def test_both_formats_written_and_readable(self, tmp_path: Path):
        graph, query, records = build_full_pipeline_graph(
            query_text="climate change", max_hops=1, top_k=2
        )
        compiler = ReportCompiler()
        report = compiler.compile(graph=graph, query=query, contradiction_records=records)
        written = compiler.write(
            report=report,
            output_dir=tmp_path,
            output_formats=["txt", "md"],
            _confirm_fn=lambda p: "y",
        )
        assert len(written) == 2
        for path in written:
            assert path.exists()
            content = path.read_text(encoding="utf-8")
            assert len(content) > 0

    def test_safe_filename_special_chars(self, tmp_path: Path):
        """Query text with special characters should not break filenames."""
        query = ResearchQuery(text="AI: What's next?", max_hops=1)
        graph, _ = build_simple_graph(1)
        compiler = ReportCompiler()
        report = compiler.compile(graph=graph, query=query)
        written = compiler.write(
            report=report,
            output_dir=tmp_path,
            output_formats=["txt"],
            _confirm_fn=lambda p: "y",
        )
        assert len(written) == 1
        assert written[0].exists()
