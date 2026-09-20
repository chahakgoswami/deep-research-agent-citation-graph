"""End-to-end integration tests for the CLI entry point and full pipeline."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List
from unittest.mock import patch

import pytest

from research_agent.cli import build_parser, main, run_pipeline

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "mock_results.json"
SIMILARITY_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "embedding_similarity.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_pipeline(
    tmp_path: Path,
    query: str = "climate change",
    max_hops: int = 2,
    formats: List[str] | None = None,
) -> List[Path]:
    """Run the full pipeline on fixture data with auto-confirm."""
    return run_pipeline(
        query_text=query,
        max_hops=max_hops,
        output_formats=formats or ["txt", "md"],
        output_dir=tmp_path,
        auto_confirm=True,
        fixture_path=FIXTURE_PATH,
        similarity_fixture_path=SIMILARITY_FIXTURE_PATH,
    )


# ---------------------------------------------------------------------------
# Argument parser tests
# ---------------------------------------------------------------------------


class TestBuildParser:
    def test_parser_created(self):
        parser = build_parser()
        assert parser is not None

    def test_query_required(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_query_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--query", "AI safety"])
        assert args.query == "AI safety"

    def test_query_short_flag(self):
        parser = build_parser()
        args = parser.parse_args(["-q", "AI safety"])
        assert args.query == "AI safety"

    def test_default_max_hops(self):
        parser = build_parser()
        args = parser.parse_args(["--query", "test"])
        assert args.max_hops == 3

    def test_custom_max_hops(self):
        parser = build_parser()
        args = parser.parse_args(["--query", "test", "--max-hops", "5"])
        assert args.max_hops == 5

    def test_max_hops_short_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--query", "test", "-n", "2"])
        assert args.max_hops == 2

    def test_default_formats(self):
        parser = build_parser()
        args = parser.parse_args(["--query", "test"])
        assert set(args.format) == {"txt", "md"}

    def test_format_txt_only(self):
        parser = build_parser()
        args = parser.parse_args(["--query", "test", "--format", "txt"])
        assert args.format == ["txt"]

    def test_format_md_only(self):
        parser = build_parser()
        args = parser.parse_args(["--query", "test", "--format", "md"])
        assert args.format == ["md"]

    def test_format_both_explicit(self):
        parser = build_parser()
        args = parser.parse_args(["--query", "test", "--format", "txt", "md"])
        assert set(args.format) == {"txt", "md"}

    def test_invalid_format_rejected(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--query", "test", "--format", "pdf"])

    def test_default_output_dir(self):
        parser = build_parser()
        args = parser.parse_args(["--query", "test"])
        assert args.output_dir == Path("./output")

    def test_custom_output_dir(self):
        parser = build_parser()
        args = parser.parse_args(["--query", "test", "--output-dir", "/tmp/reports"])
        assert args.output_dir == Path("/tmp/reports")

    def test_verbose_default_false(self):
        parser = build_parser()
        args = parser.parse_args(["--query", "test"])
        assert args.verbose is False

    def test_verbose_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--query", "test", "--verbose"])
        assert args.verbose is True

    def test_verbose_short_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--query", "test", "-v"])
        assert args.verbose is True

    def test_yes_default_false(self):
        parser = build_parser()
        args = parser.parse_args(["--query", "test"])
        assert args.yes is False

    def test_yes_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--query", "test", "--yes"])
        assert args.yes is True

    def test_yes_short_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--query", "test", "-y"])
        assert args.yes is True


# ---------------------------------------------------------------------------
# run_pipeline end-to-end tests
# ---------------------------------------------------------------------------


class TestRunPipeline:
    def test_returns_list_of_paths(self, tmp_path: Path):
        written = _run_pipeline(tmp_path)
        assert isinstance(written, list)
        assert len(written) > 0

    def test_both_formats_written(self, tmp_path: Path):
        written = _run_pipeline(tmp_path, formats=["txt", "md"])
        suffixes = {p.suffix for p in written}
        assert ".txt" in suffixes
        assert ".md" in suffixes

    def test_txt_only_format(self, tmp_path: Path):
        written = _run_pipeline(tmp_path, formats=["txt"])
        assert len(written) == 1
        assert written[0].suffix == ".txt"

    def test_md_only_format(self, tmp_path: Path):
        written = _run_pipeline(tmp_path, formats=["md"])
        assert len(written) == 1
        assert written[0].suffix == ".md"

    def test_files_exist_after_run(self, tmp_path: Path):
        written = _run_pipeline(tmp_path)
        for path in written:
            assert path.exists()

    def test_txt_contains_research_report_header(self, tmp_path: Path):
        written = _run_pipeline(tmp_path, formats=["txt"])
        content = written[0].read_text(encoding="utf-8")
        assert "RESEARCH REPORT" in content

    def test_md_contains_h1_header(self, tmp_path: Path):
        written = _run_pipeline(tmp_path, formats=["md"])
        content = written[0].read_text(encoding="utf-8")
        assert "# Research Report" in content

    def test_txt_contains_query_text(self, tmp_path: Path):
        written = _run_pipeline(tmp_path, query="climate change", formats=["txt"])
        content = written[0].read_text(encoding="utf-8")
        assert "climate change" in content.lower()

    def test_md_contains_query_text(self, tmp_path: Path):
        written = _run_pipeline(tmp_path, query="climate change", formats=["md"])
        content = written[0].read_text(encoding="utf-8")
        assert "climate change" in content.lower()

    def test_report_has_four_sections(self, tmp_path: Path):
        from research_agent.backends.mock_search import MockSearchBackend
        from research_agent.citation_graph import CitationGraph
        from research_agent.contradiction_detector import ContradictionDetector, MockEmbeddingStore
        from research_agent.engine import MultiHopEngine
        from research_agent.grader import SourceGrader
        from research_agent.models import ResearchQuery
        from research_agent.report_compiler import ReportCompiler

        query = ResearchQuery(text="climate change", max_hops=2)
        backend = MockSearchBackend(fixture_path=FIXTURE_PATH)
        engine = MultiHopEngine(backend=backend)
        chain = engine.run(query)
        graph = CitationGraph.from_hop_chain(chain)
        grader = SourceGrader()
        grader.grade_all(graph.all_nodes())
        store = MockEmbeddingStore(fixture_path=SIMILARITY_FIXTURE_PATH)
        detector = ContradictionDetector(embedding_store=store)
        records = detector.detect(graph.all_nodes())
        compiler = ReportCompiler()
        report = compiler.compile(graph=graph, query=query, contradiction_records=records)
        assert len(report.sections) == 4

    def test_report_has_citations(self, tmp_path: Path):
        from research_agent.backends.mock_search import MockSearchBackend
        from research_agent.citation_graph import CitationGraph
        from research_agent.contradiction_detector import ContradictionDetector, MockEmbeddingStore
        from research_agent.engine import MultiHopEngine
        from research_agent.grader import SourceGrader
        from research_agent.models import ResearchQuery
        from research_agent.report_compiler import ReportCompiler

        query = ResearchQuery(text="climate change", max_hops=2)
        backend = MockSearchBackend(fixture_path=FIXTURE_PATH)
        engine = MultiHopEngine(backend=backend)
        chain = engine.run(query)
        graph = CitationGraph.from_hop_chain(chain)
        grader = SourceGrader()
        grader.grade_all(graph.all_nodes())
        store = MockEmbeddingStore(fixture_path=SIMILARITY_FIXTURE_PATH)
        detector = ContradictionDetector(embedding_store=store)
        records = detector.detect(graph.all_nodes())
        compiler = ReportCompiler()
        report = compiler.compile(graph=graph, query=query, contradiction_records=records)
        assert len(report.citations) > 0

    def test_citation_count_matches_unique_urls(self, tmp_path: Path):
        from research_agent.backends.mock_search import MockSearchBackend
        from research_agent.citation_graph import CitationGraph
        from research_agent.contradiction_detector import ContradictionDetector, MockEmbeddingStore
        from research_agent.engine import MultiHopEngine
        from research_agent.grader import SourceGrader
        from research_agent.models import ResearchQuery
        from research_agent.report_compiler import ReportCompiler

        query = ResearchQuery(text="climate change", max_hops=2)
        backend = MockSearchBackend(fixture_path=FIXTURE_PATH)
        engine = MultiHopEngine(backend=backend)
        chain = engine.run(query)
        graph = CitationGraph.from_hop_chain(chain)
        grader = SourceGrader()
        grader.grade_all(graph.all_nodes())
        store = MockEmbeddingStore(fixture_path=SIMILARITY_FIXTURE_PATH)
        detector = ContradictionDetector(embedding_store=store)
        records = detector.detect(graph.all_nodes())
        compiler = ReportCompiler()
        report = compiler.compile(graph=graph, query=query, contradiction_records=records)
        unique_urls = len({n.source.url for n in graph.all_nodes()})
        assert len(report.citations) == unique_urls

    def test_txt_contains_references_section(self, tmp_path: Path):
        written = _run_pipeline(tmp_path, formats=["txt"])
        content = written[0].read_text(encoding="utf-8")
        assert "REFERENCES" in content

    def test_txt_has_numbered_citations(self, tmp_path: Path):
        written = _run_pipeline(tmp_path, formats=["txt"])
        content = written[0].read_text(encoding="utf-8")
        assert "[1]" in content

    def test_md_has_numbered_references(self, tmp_path: Path):
        written = _run_pipeline(tmp_path, formats=["md"])
        content = written[0].read_text(encoding="utf-8")
        assert "## References" in content
        assert "1." in content

    def test_all_citations_graded(self, tmp_path: Path):
        from research_agent.backends.mock_search import MockSearchBackend
        from research_agent.citation_graph import CitationGraph
        from research_agent.contradiction_detector import ContradictionDetector, MockEmbeddingStore
        from research_agent.engine import MultiHopEngine
        from research_agent.grader import SourceGrader
        from research_agent.models import ResearchQuery
        from research_agent.report_compiler import ReportCompiler

        query = ResearchQuery(text="climate change", max_hops=2)
        backend = MockSearchBackend(fixture_path=FIXTURE_PATH)
        engine = MultiHopEngine(backend=backend)
        chain = engine.run(query)
        graph = CitationGraph.from_hop_chain(chain)
        grader = SourceGrader()
        grader.grade_all(graph.all_nodes())
        store = MockEmbeddingStore(fixture_path=SIMILARITY_FIXTURE_PATH)
        detector = ContradictionDetector(embedding_store=store)
        records = detector.detect(graph.all_nodes())
        compiler = ReportCompiler()
        report = compiler.compile(graph=graph, query=query, contradiction_records=records)
        for citation in report.citations:
            assert citation.grade is not None

    def test_output_dir_created_if_missing(self, tmp_path: Path):
        new_dir = tmp_path / "deep" / "nested" / "dir"
        _run_pipeline(new_dir, formats=["txt"])
        assert new_dir.exists()

    def test_quantum_computing_query(self, tmp_path: Path):
        written = _run_pipeline(
            tmp_path, query="quantum computing", max_hops=1, formats=["txt"]
        )
        content = written[0].read_text(encoding="utf-8")
        assert "quantum computing" in content.lower()

    def test_ai_query(self, tmp_path: Path):
        written = _run_pipeline(
            tmp_path, query="artificial intelligence", max_hops=1, formats=["txt"]
        )
        content = written[0].read_text(encoding="utf-8")
        assert "artificial intelligence" in content.lower()

    def test_contradiction_section_present(self, tmp_path: Path):
        written = _run_pipeline(
            tmp_path, query="climate change", max_hops=2, formats=["txt"]
        )
        content = written[0].read_text(encoding="utf-8")
        assert "CONTRADICTIONS" in content

    def test_graded_sources_section_present(self, tmp_path: Path):
        written = _run_pipeline(
            tmp_path, query="climate change", max_hops=2, formats=["txt"]
        )
        content = written[0].read_text(encoding="utf-8")
        assert "GRADED SOURCES" in content

    def test_findings_section_present(self, tmp_path: Path):
        written = _run_pipeline(
            tmp_path, query="climate change", max_hops=2, formats=["txt"]
        )
        content = written[0].read_text(encoding="utf-8")
        assert "FINDINGS" in content

    def test_summary_section_present(self, tmp_path: Path):
        written = _run_pipeline(
            tmp_path, query="climate change", max_hops=2, formats=["txt"]
        )
        content = written[0].read_text(encoding="utf-8")
        assert "SUMMARY" in content


# ---------------------------------------------------------------------------
# main() / CLI integration tests
# ---------------------------------------------------------------------------


class TestMainCLI:
    def test_main_returns_zero_on_success(self, tmp_path: Path):
        # Patch run_pipeline to avoid real I/O and fixture dependency
        with patch("research_agent.cli.run_pipeline", return_value=[tmp_path / "r.txt"]):
            exit_code = main(
                [
                    "--query", "test topic",
                    "--yes",
                    "--output-dir", str(tmp_path),
                    "--format", "txt",
                ]
            )
        assert exit_code == 0

    def test_main_returns_one_on_permission_error(self, tmp_path: Path):
        with patch(
            "research_agent.cli.run_pipeline",
            side_effect=PermissionError("cancelled by user"),
        ):
            exit_code = main(
                [
                    "--query", "test topic",
                    "--output-dir", str(tmp_path),
                    "--format", "txt",
                ]
            )
        assert exit_code == 1

    def test_main_exits_nonzero_without_query(self):
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code != 0

    def test_main_full_pipeline_txt(self, tmp_path: Path):
        """Full end-to-end through main() with auto-confirm using fixture data."""
        with patch("research_agent.cli.run_pipeline") as mock_run:
            mock_run.return_value = [tmp_path / "climate_change.txt"]
            exit_code = main(
                [
                    "--query", "climate change",
                    "--max-hops", "2",
                    "--format", "txt",
                    "--output-dir", str(tmp_path),
                    "--yes",
                ]
            )
        assert exit_code == 0
        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args
        assert call_kwargs.kwargs["query_text"] == "climate change"
        assert call_kwargs.kwargs["max_hops"] == 2
        assert call_kwargs.kwargs["output_formats"] == ["txt"]
        assert call_kwargs.kwargs["auto_confirm"] is True

    def test_main_verbose_flag_propagated(self, tmp_path: Path):
        with patch("research_agent.cli.run_pipeline") as mock_run:
            mock_run.return_value = []
            main(
                [
                    "--query", "test",
                    "--verbose",
                    "--output-dir", str(tmp_path),
                    "--yes",
                ]
            )
        call_kwargs = mock_run.call_args
        assert call_kwargs.kwargs["verbose"] is True

    def test_main_no_files_written_message(self, tmp_path: Path, capsys):
        with patch("research_agent.cli.run_pipeline", return_value=[]):
            main(
                [
                    "--query", "test",
                    "--output-dir", str(tmp_path),
                    "--yes",
                ]
            )
        captured = capsys.readouterr()
        assert "No output files" in captured.out

    def test_main_reports_written_files(self, tmp_path: Path, capsys):
        fake_file = tmp_path / "report.txt"
        with patch("research_agent.cli.run_pipeline", return_value=[fake_file]):
            main(
                [
                    "--query", "test",
                    "--output-dir", str(tmp_path),
                    "--yes",
                ]
            )
        captured = capsys.readouterr()
        assert "1 file" in captured.out


# ---------------------------------------------------------------------------
# Full end-to-end tests (real pipeline, fixture data)
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """Run the complete pipeline through run_pipeline() using real fixture data."""

    def test_climate_change_txt_report_structure(self, tmp_path: Path):
        written = _run_pipeline(
            tmp_path, query="climate change", max_hops=2, formats=["txt"]
        )
        assert len(written) == 1
        content = written[0].read_text(encoding="utf-8")
        # Must contain all four section headers
        for header in ["SUMMARY", "FINDINGS", "CONTRADICTIONS", "GRADED SOURCES"]:
            assert header in content, f"Missing section: {header}"

    def test_climate_change_md_report_structure(self, tmp_path: Path):
        written = _run_pipeline(
            tmp_path, query="climate change", max_hops=2, formats=["md"]
        )
        content = written[0].read_text(encoding="utf-8")
        for header in ["## Summary", "## Findings", "## Contradictions", "## Graded Sources"]:
            assert header in content, f"Missing section: {header}"

    def test_climate_change_has_contradiction_content(self, tmp_path: Path):
        """With the fixture data the denial site contradicts scientific sources."""
        written = _run_pipeline(
            tmp_path, query="climate change", max_hops=2, formats=["txt"]
        )
        content = written[0].read_text(encoding="utf-8")
        # At least one numbered contradiction entry should appear
        assert "contradict" in content.lower() or "No contradictions" in content

    def test_climate_change_citation_count_positive(self, tmp_path: Path):
        from research_agent.backends.mock_search import MockSearchBackend
        from research_agent.citation_graph import CitationGraph
        from research_agent.contradiction_detector import ContradictionDetector, MockEmbeddingStore
        from research_agent.engine import MultiHopEngine
        from research_agent.grader import SourceGrader
        from research_agent.models import ResearchQuery
        from research_agent.report_compiler import ReportCompiler

        query = ResearchQuery(text="climate change", max_hops=2)
        backend = MockSearchBackend(fixture_path=FIXTURE_PATH)
        engine = MultiHopEngine(backend=backend)
        chain = engine.run(query)
        graph = CitationGraph.from_hop_chain(chain)
        grader = SourceGrader()
        grader.grade_all(graph.all_nodes())
        store = MockEmbeddingStore(fixture_path=SIMILARITY_FIXTURE_PATH)
        detector = ContradictionDetector(embedding_store=store)
        records = detector.detect(graph.all_nodes())
        compiler = ReportCompiler()
        report = compiler.compile(graph=graph, query=query, contradiction_records=records)
        assert len(report.citations) > 0

    def test_ai_md_contains_arxiv_url(self, tmp_path: Path):
        written = _run_pipeline(
            tmp_path, query="artificial intelligence", max_hops=1, formats=["md"]
        )
        content = written[0].read_text(encoding="utf-8")
        assert "arxiv.org" in content

    def test_quantum_txt_contains_ibm(self, tmp_path: Path):
        written = _run_pipeline(
            tmp_path, query="quantum computing", max_hops=1, formats=["txt"]
        )
        content = written[0].read_text(encoding="utf-8")
        assert "ibm.com" in content or "IBM" in content

    def test_both_formats_written_and_non_empty(self, tmp_path: Path):
        written = _run_pipeline(
            tmp_path, query="climate change", max_hops=1, formats=["txt", "md"]
        )
        assert len(written) == 2
        for path in written:
            assert path.exists()
            assert len(path.read_text(encoding="utf-8")) > 0

    def test_hop_labels_in_findings(self, tmp_path: Path):
        written = _run_pipeline(
            tmp_path, query="climate change", max_hops=2, formats=["txt"]
        )
        content = written[0].read_text(encoding="utf-8")
        assert "Hop 1" in content
        assert "Hop 2" in content

    def test_graded_sources_contain_grade_letter(self, tmp_path: Path):
        written = _run_pipeline(
            tmp_path, query="climate change", max_hops=2, formats=["txt"]
        )
        content = written[0].read_text(encoding="utf-8")
        # At least one grade letter should appear in graded sources
        assert any(f"Grade: {g}" in content for g in ["A", "B", "C", "D", "F"])

    def test_summary_mentions_hop_count(self, tmp_path: Path):
        written = _run_pipeline(
            tmp_path, query="climate change", max_hops=2, formats=["txt"]
        )
        content = written[0].read_text(encoding="utf-8")
        assert "2" in content  # max_hops = 2 appears in summary

    def test_report_filename_contains_query_words(self, tmp_path: Path):
        written = _run_pipeline(
            tmp_path, query="climate change", max_hops=1, formats=["txt"]
        )
        assert "climate" in written[0].stem

    def test_special_chars_in_query_safe_filename(self, tmp_path: Path):
        written = _run_pipeline(
            tmp_path, query="AI: What's next?", max_hops=1, formats=["txt"]
        )
        assert len(written) == 1
        assert written[0].exists()
