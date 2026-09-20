"""CLI entry point for the Deep Research Agent.

Usage examples::

    python -m research_agent.cli --query "climate change" --max-hops 2
    python -m research_agent.cli --query "AI" --max-hops 3 --format md --verbose
    python -m research_agent.cli --query "quantum computing" --output-dir ./out

Flags
-----
--query        (required) The research question / topic.
--max-hops     Maximum reasoning hops (1-10, default 3).
--format       Output format(s): txt, md, or both (default: txt md).
--output-dir   Directory to write report files (default: ./output).
--verbose      Enable verbose logging.
--yes          Skip the human-in-the-loop confirmation prompt.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

logger = logging.getLogger("research_agent")


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    root = logging.getLogger("research_agent")
    root.setLevel(level)
    root.addHandler(handler)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build and return the argparse parser."""
    parser = argparse.ArgumentParser(
        prog="research-agent",
        description="Deep Research Agent with Citation Graph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  research-agent --query \"climate change\" --max-hops 2 --format md\n"
        ),
    )
    parser.add_argument(
        "--query",
        "-q",
        required=True,
        help="The research question or topic.",
    )
    parser.add_argument(
        "--max-hops",
        "-n",
        type=int,
        default=3,
        metavar="N",
        help="Maximum number of reasoning hops (1-10, default: 3).",
    )
    parser.add_argument(
        "--format",
        "-f",
        nargs="+",
        choices=["txt", "md"],
        default=["txt", "md"],
        metavar="FORMAT",
        help="Output format(s): txt and/or md (default: txt md).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("./output"),
        metavar="DIR",
        help="Directory where report files are written (default: ./output).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose/debug logging.",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip the human-in-the-loop confirmation prompt.",
    )
    return parser


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


def run_pipeline(
    query_text: str,
    max_hops: int,
    output_formats: List[str],
    output_dir: Path,
    auto_confirm: bool = False,
    verbose: bool = False,
    fixture_path: Optional[Path] = None,
    similarity_fixture_path: Optional[Path] = None,
) -> List[Path]:
    """Execute the full research pipeline and return written file paths.

    This function is intentionally importable so end-to-end tests can call it
    directly without spawning a subprocess.

    Args:
        query_text:               The research question.
        max_hops:                 Maximum hops for the multi-hop engine.
        output_formats:           List of 'txt' / 'md'.
        output_dir:               Directory to write outputs.
        auto_confirm:             If True, skip the write-confirmation prompt.
        verbose:                  Enable debug logging.
        fixture_path:             Override for the mock search fixture.
        similarity_fixture_path:  Override for the embedding similarity fixture.

    Returns:
        List of :class:`pathlib.Path` objects for files written.
    """
    # Lazy imports keep start-up fast and let tests swap fixtures easily.
    from research_agent.backends.mock_search import MockSearchBackend
    from research_agent.citation_graph import CitationGraph
    from research_agent.contradiction_detector import (
        ContradictionDetector,
        MockEmbeddingStore,
    )
    from research_agent.engine import MultiHopEngine
    from research_agent.grader import SourceGrader
    from research_agent.models import ResearchQuery
    from research_agent.report_compiler import ReportCompiler

    # --- Step 1: build query ---
    logger.info("Building research query: '%s' (max_hops=%d)", query_text, max_hops)
    query = ResearchQuery(text=query_text, max_hops=max_hops)

    # --- Step 2: multi-hop search ---
    backend_kwargs = {}
    if fixture_path is not None:
        backend_kwargs["fixture_path"] = fixture_path
    backend = MockSearchBackend(**backend_kwargs)
    engine = MultiHopEngine(backend=backend)
    logger.info("Running multi-hop engine …")
    chain = engine.run(query)
    logger.debug(
        "HopChain depth=%d, total sources=%d", chain.depth, len(chain.all_sources)
    )

    # --- Step 3: build citation graph ---
    logger.info("Building citation graph …")
    graph = CitationGraph.from_hop_chain(chain)
    logger.debug("CitationGraph nodes=%d", graph.node_count)

    # --- Step 4: grade sources ---
    logger.info("Grading sources …")
    grader = SourceGrader()
    grader.grade_all(graph.all_nodes())

    # --- Step 5: detect contradictions ---
    logger.info("Detecting contradictions …")
    sim_kwargs = {}
    if similarity_fixture_path is not None:
        sim_kwargs["fixture_path"] = similarity_fixture_path
    store = MockEmbeddingStore(**sim_kwargs)
    detector = ContradictionDetector(embedding_store=store)
    records = detector.detect(graph.all_nodes())
    logger.debug("Contradiction records found: %d", len(records))

    # --- Step 6: compile report ---
    logger.info("Compiling report …")
    compiler = ReportCompiler()
    report = compiler.compile(graph=graph, query=query, contradiction_records=records)

    # --- Step 7: write files (with optional auto-confirm) ---
    if auto_confirm:
        confirm_fn = lambda prompt: "y"
    else:
        confirm_fn = None  # will call input() inside compiler.write

    logger.info("Writing report to '%s' …", output_dir)
    written = compiler.write(
        report=report,
        output_dir=output_dir,
        output_formats=output_formats,
        _confirm_fn=confirm_fn,
    )
    for path in written:
        logger.info("  Written: %s", path)
    return written


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    """Parse CLI arguments and run the pipeline.

    Returns:
        Exit code (0 = success, non-zero = failure).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    _configure_logging(args.verbose)

    try:
        written = run_pipeline(
            query_text=args.query,
            max_hops=args.max_hops,
            output_formats=args.format,
            output_dir=args.output_dir,
            auto_confirm=args.yes,
            verbose=args.verbose,
        )
        if written:
            print(f"Report written to {len(written)} file(s):")
            for p in written:
                print(f"  {p}")
        else:
            print("No output files were written.")
        return 0
    except PermissionError as exc:
        print(f"Aborted: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover
        logger.exception("Unexpected error: %s", exc)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
