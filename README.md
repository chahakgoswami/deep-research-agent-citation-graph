# Deep Research Agent with Citation Graph

Multi-hop research agent with citation graph, source grading, and contradiction detection.

**Domain:** Agentic AI
**Language:** python
**Demonstrates:** You can orchestrate long-horizon reasoning safely.

## 7-day build plan

- [ ] Day 1: Scaffold the project with a src/research_agent package, define core data models (ResearchQuery, Source, Hop, CitationNode) using dataclasses/Pydantic, implement a mock search backend that returns simulated web results from a local JSON fixture file, and write unit tests for all models and the mock search interface.
- [ ] Day 2: Build the multi-hop reasoning engine that accepts a query, iteratively expands sub-questions up to a configurable depth, calls the mock search backend each hop, stores retrieved sources in a HopChain object, and write tests verifying the hop chain grows correctly and terminates at max depth.
- [ ] Day 3: Implement source grading by adding a SourceGrader module that scores each retrieved source on simulated signals (domain authority, recency, citation count from fixtures), attaches a grade and confidence to each CitationNode, and write tests covering edge cases like missing metadata and duplicate URLs.
- [ ] Day 4: Build a CitationGraph class using networkx to connect CitationNodes with directed edges representing 'cited-by' and 'supports' relationships, implement graph traversal helpers to find authority hubs and orphan sources, and write tests for graph construction, edge insertion, and traversal correctness.
- [ ] Day 5: Add a ContradictionDetector that compares claim snippets across CitationNodes using simple keyword/semantic heuristics (mocked embedding similarity via a deterministic fixture map), flags contradicting source pairs with an explanation string, and write tests with fixture data that includes known contradictions.
- [ ] Day 6: Build the ReportCompiler that walks the finalized CitationGraph, assembles a structured final report with sections (summary, findings, contradictions, graded sources), inlines numbered citations, renders to both plain text and Markdown files, adds a human-in-the-loop confirmation prompt before writing output files, and write integration tests using temp directories.
- [ ] Day 7: Wire everything into a CLI entry point (research_agent/cli.py) using argparse with flags for query, max-hops, output format, and verbosity, add end-to-end integration tests that run the full pipeline on fixture data and assert report structure and citation counts, and package the project with pyproject.toml including optional dev/test extras.

_A comprehensive README with an architecture diagram is generated on Day 7._
