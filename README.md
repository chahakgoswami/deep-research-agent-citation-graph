# Deep Research Agent with Citation Graph

Multi-hop research agent with citation graph, source grading, and contradiction detection.

**Domain:** Agentic AI  
**Language:** Python 3.10+  
**Demonstrates:** Long-horizon multi-hop reasoning with safe, human-confirmed output.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLI (cli.py)                            │
│  argparse: --query --max-hops --format --output-dir --verbose   │
└────────────────────────────┬────────────────────────────────────┘
                             │  run_pipeline()
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   MultiHopEngine (engine.py)                    │
│  ResearchQuery ──► SubQuestionExpander ──► MockSearchBackend    │
│          └─────────────── HopChain ◄───────────────────┘        │
└────────────────────────────┬────────────────────────────────────┘
                             │  HopChain
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                 CitationGraph (citation_graph.py)               │
│  CitationNode (Source + hop_index + grade + claim_snippet)      │
│  Directed edges: cited-by | supports                            │
│  Helpers: authority_hubs(), orphan_nodes(), subgraph_by_hop()   │
└──────────┬────────────────────────────┬───────────────────────-─┘
           │ SourceGrader (grader.py)   │ ContradictionDetector
           │ domain_authority           │   (contradiction_detector.py)
           │ recency                    │ MockEmbeddingStore
           │ citation_count             │ negation heuristic
           │ → letter grade A-F         │ → ContradictionRecord[]
           └────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│               ReportCompiler (report_compiler.py)               │
│  Sections: Summary | Findings | Contradictions | Graded Sources │
│  Renders: plain text (.txt) and Markdown (.md)                  │
│  Human-in-the-loop confirmation before writing files            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Install

```bash
pip install -e ".[dev]"
```

### Run via CLI

```bash
# Basic usage (prompts for confirmation before writing)
research-agent --query "climate change" --max-hops 2

# Auto-confirm and write Markdown only
research-agent --query "quantum computing" --max-hops 3 --format md --yes

# Verbose mode with custom output directory
research-agent --query "artificial intelligence" --output-dir ./reports --verbose
```

### CLI Flags

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--query` | `-q` | *(required)* | Research question or topic |
| `--max-hops` | `-n` | `3` | Max reasoning hops (1-10) |
| `--format` | `-f` | `txt md` | Output format(s): `txt` and/or `md` |
| `--output-dir` | `-o` | `./output` | Directory for report files |
| `--verbose` | `-v` | off | Enable debug logging |
| `--yes` | `-y` | off | Skip confirmation prompt |

### Run Tests

```bash
pytest
pytest --cov=research_agent --cov-report=term-missing
```

---

## Project Structure

```
.
├── src/
│   └── research_agent/
│       ├── __init__.py
│       ├── cli.py                    # CLI entry point
│       ├── models.py                 # Pydantic data models
│       ├── engine.py                 # Multi-hop engine + HopChain
│       ├── grader.py                 # SourceGrader (A-F grading)
│       ├── citation_graph.py         # CitationGraph (networkx)
│       ├── contradiction_detector.py # ContradictionDetector
│       ├── report_compiler.py        # ReportCompiler
│       └── backends/
│           ├── __init__.py
│           └── mock_search.py        # MockSearchBackend
├── tests/
│   ├── fixtures/
│   │   ├── mock_results.json         # Simulated search results
│   │   └── embedding_similarity.json # Mocked embedding scores
│   ├── test_models.py
│   ├── test_mock_search.py
│   ├── test_engine.py
│   ├── test_grader.py
│   ├── test_citation_graph.py
│   ├── test_contradiction_detector.py
│   ├── test_report_compiler.py
│   └── test_cli.py                   # CLI + end-to-end tests
├── pyproject.toml
└── README.md
```

---

## 7-Day Build Plan

- [x] Day 1: Core data models + mock search backend
- [x] Day 2: Multi-hop reasoning engine + HopChain
- [x] Day 3: SourceGrader (domain authority, recency, citation count)
- [x] Day 4: CitationGraph with networkx + traversal helpers
- [x] Day 5: ContradictionDetector (heuristics + mocked embeddings)
- [x] Day 6: ReportCompiler (structured report, txt/md, human confirmation)
- [x] Day 7: CLI entry point, end-to-end tests, pyproject.toml packaging
