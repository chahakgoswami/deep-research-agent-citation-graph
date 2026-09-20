# Deep Research Agent with Citation Graph

A fully agentic, multi-hop research pipeline that iteratively expands sub-questions, retrieves sources, builds a citation graph, grades every source with letter grades (A–F), detects semantic contradictions between sources, and compiles a structured report — all wired to a safe CLI that requires human confirmation before writing any files to disk.

---

## Features

- **Multi-hop reasoning engine** — iteratively generates sub-questions and retrieves sources for up to 10 configurable hops, accumulating evidence in a `HopChain`
- **Citation graph** — directed `networkx` graph connecting `CitationNode` objects via `cited-by` and `supports` edges, with traversal helpers for authority hubs, orphan detection, and hop-based subgraphs
- **Source grading (A–F)** — `SourceGrader` blends three signals (domain authority, recency, citation count) into a weighted composite score and attaches a letter grade + confidence to every node
- **Contradiction detection** — `ContradictionDetector` compares every source pair using negation-keyword heuristics and mocked embedding similarity scores; contradicting nodes are cross-linked in the graph
- **Structured report compilation** — `ReportCompiler` walks the finalized graph and produces a four-section report (Summary, Findings, Contradictions, Graded Sources) with inline numbered citations, rendered as both plain text and Markdown
- **Human-in-the-loop confirmation** — the CLI and `ReportCompiler.write()` require explicit user approval (`y/yes`) before any file is written to disk
- **Fully tested** — 8 test modules covering unit, integration, and end-to-end scenarios with deterministic fixture data

---

## Architecture

```mermaid
flowchart TD
    A([👤 User / CLI]) -->|"--query --max-hops --format"| B[build_parser\ncli.py]
    B --> C[run_pipeline\ncli.py]
    C --> D[ResearchQuery\nmodels.py]
    D --> E[MultiHopEngine\nengine.py]
    E -->|"expand sub-question"| F[SubQuestionExpander\nengine.py]
    F -->|"sub-query string"| G[MockSearchBackend\nbackends/mock_search.py]
    G -->|"reads"| H[(mock_results.json\nfixture)]
    G -->|"List[Source]"| E
    E -->|"HopChain"| I[CitationGraph.from_hop_chain\ncitation_graph.py]
    I --> J[(CitationGraph\nnetworkx DiGraph)]
    J --> K[SourceGrader\ngrader.py]
    K -->|"grade + confidence\nA–F"| J
    J --> L[ContradictionDetector\ncontradiction_detector.py]
    L -->|"reads"| M[(embedding_similarity.json\nfixture)]
    L -->|"ContradictionRecord[]"| N[ReportCompiler\nreport_compiler.py]
    J --> N
    N --> O[ResearchReport\nSummary · Findings\nContradictions · Graded Sources]
    O -->|"✅ human confirms"| P[📄 report.txt]
    O -->|"✅ human confirms"| Q[📄 report.md]

    classDef user fill:#FFD966,stroke:#B8860B,color:#000;
    classDef cli fill:#A4C2F4,stroke:#3366CC,color:#000;
    classDef engine fill:#6FA8DC,stroke:#3366CC,color:#000;
    classDef backend fill:#B6D7A8,stroke:#38761D,color:#000;
    classDef fixture fill:#93C47D,stroke:#38761D,color:#000;
    classDef graph fill:#C9B1D9,stroke:#674EA7,color:#000;
    classDef grader fill:#FFB6C1,stroke:#CC0000,color:#000;
    classDef detector fill:#F9CB9C,stroke:#B45F06,color:#000;
    classDef compiler fill:#EA9999,stroke:#CC0000,color:#000;
    classDef output fill:#E06666,stroke:#990000,color:#fff;

    class A user;
    class B,C cli;
    class D,E,F engine;
    class G backend;
    class H,M fixture;
    class I,J graph;
    class K grader;
    class L detector;
    class N,O compiler;
    class P,Q output;
```

---

## Installation

### Prerequisites

- Python 3.10 or newer

### Install for development

```bash
git clone https://github.com/your-org/research-agent.git
cd research-agent
pip install -e ".[dev]"
```

This installs the package in editable mode together with `pytest` and `pytest-cov`.

### Core dependencies (installed automatically)

| Package | Purpose |
|---------|---------|
| `pydantic >= 2.0` | Validated data models (`ResearchQuery`, `Source`, `CitationNode`, …) |
| `networkx >= 3.0` | Directed citation graph and traversal algorithms |

---

## Usage

### CLI

```bash
# Research a topic (prompts for confirmation before writing)
research-agent --query "climate change" --max-hops 2

# Auto-confirm and write Markdown only
research-agent --query "quantum computing" --max-hops 3 --format md --yes

# Both formats, custom output directory, verbose logging
research-agent --query "artificial intelligence" \
               --max-hops 2 \
               --format txt md \
               --output-dir ./reports \
               --verbose --yes

# Single hop, text only
research-agent -q "nuclear fusion" -n 1 -f txt -y
```

### CLI Flags

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--query` | `-q` | *(required)* | Research question or topic |
| `--max-hops` | `-n` | `3` | Maximum reasoning hops (1–10) |
| `--format` | `-f` | `txt md` | Output format(s): `txt` and/or `md` |
| `--output-dir` | `-o` | `./output` | Directory for report files |
| `--verbose` | `-v` | off | Enable debug logging |
| `--yes` | `-y` | off | Skip human-in-the-loop confirmation prompt |

### Python API

```python
from pathlib import Path
from research_agent.cli import run_pipeline

# Run the full pipeline programmatically
written_files = run_pipeline(
    query_text="climate change",
    max_hops=2,
    output_formats=["txt", "md"],
    output_dir=Path("./output"),
    auto_confirm=True,
)
for path in written_files:
    print(path.read_text())
```

```python
# Or use individual components
from research_agent.backends.mock_search import MockSearchBackend
from research_agent.engine import MultiHopEngine
from research_agent.citation_graph import CitationGraph
from research_agent.grader import SourceGrader
from research_agent.contradiction_detector import ContradictionDetector, MockEmbeddingStore
from research_agent.report_compiler import ReportCompiler
from research_agent.models import ResearchQuery

query   = ResearchQuery(text="quantum computing", max_hops=3)
chain   = MultiHopEngine(backend=MockSearchBackend()).run(query)
graph   = CitationGraph.from_hop_chain(chain)
SourceGrader().grade_all(graph.all_nodes())

store   = MockEmbeddingStore()
records = ContradictionDetector(embedding_store=store).detect(graph.all_nodes())

compiler = ReportCompiler()
report   = compiler.compile(graph=graph, query=query, contradiction_records=records)
print(report.to_markdown())
```

### Running Tests

```bash
# All tests
pytest

# With coverage report
pytest --cov=research_agent --cov-report=term-missing

# A single test module
pytest tests/test_citation_graph.py -v

# A specific test class
pytest tests/test_grader.py::TestSourceGraderGradeBasic -v
```

---

## Project Structure

```
.
├── src/
│   └── research_agent/
│       ├── __init__.py                    # Package version (0.1.0)
│       ├── cli.py                         # CLI entry point (argparse + run_pipeline)
│       ├── models.py                      # Pydantic models: ResearchQuery, Source, Hop, CitationNode
│       ├── engine.py                      # MultiHopEngine, HopChain, SubQuestionExpander
│       ├── grader.py                      # SourceGrader — letter grades A-F
│       ├── citation_graph.py              # CitationGraph (networkx DiGraph)
│       ├── contradiction_detector.py      # ContradictionDetector + MockEmbeddingStore
│       ├── report_compiler.py             # ReportCompiler → ResearchReport (txt + md)
│       └── backends/
│           ├── __init__.py
│           └── mock_search.py             # MockSearchBackend (fixture-driven)
├── tests/
│   ├── fixtures/
│   │   ├── mock_results.json              # Simulated search results (3 topics + default)
│   │   └── embedding_similarity.json     # Deterministic similarity scores for known URL pairs
│   ├── __init__.py
│   ├── test_models.py                     # Unit tests for all Pydantic models
│   ├── test_mock_search.py                # Unit tests for MockSearchBackend
│   ├── test_engine.py                     # Unit + integration tests for MultiHopEngine
│   ├── test_grader.py                     # Unit + integration tests for SourceGrader
│   ├── test_citation_graph.py             # Unit + integration tests for CitationGraph
│   ├── test_contradiction_detector.py     # Unit + integration tests for ContradictionDetector
│   ├── test_report_compiler.py            # Integration tests for ReportCompiler
│   └── test_cli.py                        # CLI + full end-to-end tests
├── pyproject.toml                         # Build config, dependencies, entry point
└── README.md
```

---

## How It Works

### 1. Query validation (`models.py`)

A `ResearchQuery` is created from the user's string. Pydantic validates that the text is non-blank and that `max_hops` is between 1 and 10. Every model (`Hop`, `Source`, `CitationNode`) gets a UUID automatically.

### 2. Multi-hop expansion (`engine.py`)

`MultiHopEngine.run()` loops `max_hops` times. On each iteration `SubQuestionExpander` picks the next template from a round-robin list (e.g. *"What are the primary causes of {topic}?"*) and calls `MockSearchBackend.search()`. Results are wrapped into a `Hop` and accumulated in a `HopChain`.

### 3. Mock search backend (`backends/mock_search.py`)

`MockSearchBackend` loads `tests/fixtures/mock_results.json` at construction time. It matches the incoming query by case-insensitive substring against the fixture keys (`"climate change"`, `"artificial intelligence"`, `"quantum computing"`) and falls back to the `"default"` bucket for unknown queries. Results are deserialized into `Source` Pydantic objects.

### 4. Citation graph construction (`citation_graph.py`)

`CitationGraph.from_hop_chain()` wraps every `Source` in a `CitationNode` (preserving `hop_index` and copying `snippet` → `claim_snippet`) and inserts it into a `networkx.DiGraph`. Edges (`cited-by`, `supports`) can be added manually. Traversal helpers:

- `authority_hubs(top_n)` — ranks by in-degree
- `orphan_nodes()` — nodes with no edges
- `subgraph_by_hop(hop_index)` — hop-scoped subgraph
- `predecessors()` / `successors()` — with optional edge-type filter

### 5. Source grading (`grader.py`)

`SourceGrader` scores each `CitationNode` on three signals:

| Signal | Weight | Details |
|--------|--------|---------|
| Domain authority | 40 % | Fixture value 0–100; missing → neutral 50 |
| Recency | 30 % | Linear decay over 5 years; future dates capped at 100 |
| Citation count | 30 % | Capped at 10 000; missing → neutral 50 |

The weighted composite maps to a letter grade: ≥ 85 → **A**, ≥ 70 → **B**, ≥ 55 → **C**, ≥ 40 → **D**, below → **F**. Each missing signal subtracts 0.15 from the confidence score.

`grade_all()` de-duplicates by URL so nodes sharing the same source receive identical grades.

### 6. Contradiction detection (`contradiction_detector.py`)

`ContradictionDetector.detect()` compares every source pair using two heuristics applied together with a shared-topic-cluster guard:

1. **Negation heuristic** — one snippet but not the other contains a word from `_NEGATION_WORDS` (e.g. *hoax*, *manipulated*, *disagree*)
2. **Low similarity** — `MockEmbeddingStore` returns a score below the threshold (default 0.35)

`MockEmbeddingStore` first checks `tests/fixtures/embedding_similarity.json` for a pre-defined score keyed by `"url_a::url_b"` (sorted). Unknown pairs fall back to word-level Jaccard similarity. Detected contradictions update both nodes' `contradicts` lists and produce a `ContradictionRecord` with an explanation string.

### 7. Report compilation (`report_compiler.py`)

`ReportCompiler.compile()` assembles a `ResearchReport` with four `ReportSection` objects:

| # | Section | Content |
|---|---------|---------|
| 1 | **Summary** | Source counts, hop count, grade distribution, contradiction tally |
| 2 | **Findings** | Per-hop snippets with inline `[N]` citation numbers |
| 3 | **Contradictions** | Each `ContradictionRecord` with source refs and explanation |
| 4 | **Graded Sources** | Full source list with grade, confidence, domain authority, citation count |

`ResearchReport.to_plain_text()` and `.to_markdown()` render the same structured data to their respective formats. URLs are de-duplicated before numbering so a source retrieved in multiple hops appears only once in the reference list.

### 8. Human-in-the-loop write (`report_compiler.py` + `cli.py`)

`ReportCompiler.write()` always shows the planned output file paths and asks for confirmation. Only responses of `y` or `yes` (case-insensitive) proceed; anything else raises `PermissionError`. The CLI's `--yes` / `-y` flag injects an automatic `"y"` response for non-interactive pipelines.

---

## Notes & Roadmap

### Current limitations

- **Mock backend only** — `MockSearchBackend` reads from a static JSON fixture. All three topic buckets (`climate change`, `artificial intelligence`, `quantum computing`) and a `default` fallback are included, but the agent cannot fetch real web pages.
- **Deterministic sub-question expansion** — `SubQuestionExpander` uses a fixed round-robin template list rather than an LLM; every run with the same topic produces identical sub-questions.
- **No real embeddings** — `MockEmbeddingStore` uses pre-defined scores from `embedding_similarity.json` for known URL pairs and falls back to Jaccard word overlap. No external model is called.

### Potential extensions

- **Real search backend** — swap `MockSearchBackend` for a live web-search adapter (Bing API, SerpAPI, Brave Search) behind the same `search(query, top_k)` interface
- **LLM-powered expansion** — replace `SubQuestionExpander` with a call to an OpenAI / Anthropic model to generate contextually grounded follow-up questions
- **Real embeddings** — replace `MockEmbeddingStore` with a `sentence-transformers` or OpenAI embeddings adapter to compute genuine semantic similarity
- **Automatic edge wiring** — use citation metadata or co-reference resolution to automatically populate `cited-by` / `supports` edges in the graph rather than requiring manual `add_edge()` calls
- **Streaming / async engine** — run hops concurrently with `asyncio` and stream partial findings to the user
- **Persistent graph storage** — serialize the `CitationGraph` to SQLite or a graph database (Neo4j) for cross-session querying
- **Web UI** — expose the pipeline through a FastAPI endpoint and a lightweight React front-end with the citation graph visualized as an interactive force-directed diagram