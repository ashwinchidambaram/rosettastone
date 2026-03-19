# RosettaStone — Phased Folder Structure

This document shows how the codebase evolves across phases. Each phase builds on the previous one — new additions are marked with `← NEW`. The structure is designed so that Phase 1 code never needs to be rewritten, only extended.

---

## Phase 1: MVP — CLI + Library (JSONL → GEPA → Markdown Report)

```
rosettastone/
├── pyproject.toml                    # Package config, optional extras [eval], [redis], [all]
├── README.md
├── LICENSE                           # MIT
├── .github/
│   └── workflows/
│       └── ci.yml                    # Linting, type checking, unit tests
│
├── src/
│   └── rosettastone/
│       ├── __init__.py               # Public API: Migrator, MigrationConfig, MigrationResult
│       ├── py.typed                  # PEP 561 marker for type checking
│       │
│       ├── config.py                 # MigrationConfig (Pydantic model): source_model, target_model,
│       │                             #   eval_strategy, optimizer, data_path, output_dir, etc.
│       │
│       ├── cli/
│       │   ├── __init__.py
│       │   └── main.py              # Typer app: `rosettastone migrate`, `rosettastone preflight`,
│       │                             #   `rosettastone evaluate` commands
│       │
│       ├── core/
│       │   ├── __init__.py
│       │   ├── migrator.py           # Migrator class — orchestrates the full pipeline:
│       │   │                         #   preflight → ingest → baseline → optimize → validate → export
│       │   ├── pipeline.py           # Pipeline step definitions and execution order
│       │   └── types.py              # Shared types: PromptPair, EvalResult, MigrationReport,
│       │                             #   ConfidenceScore, OutputType enum
│       │
│       ├── preflight/
│       │   ├── __init__.py
│       │   ├── checks.py            # Run all pre-flight checks, return PreflightReport
│       │   ├── capabilities.py      # Model capability detection via LiteLLM (tool calling,
│       │   │                         #   JSON mode, vision, context window size)
│       │   ├── token_budget.py       # Per-model token estimation, tokenizer inflation warnings,
│       │   │                         #   context window overflow detection
│       │   └── cost_estimator.py     # Estimate API spend based on dataset size, model pricing,
│       │                             #   expected GEPA rollouts. Supports --dry-run
│       │
│       ├── ingest/
│       │   ├── __init__.py
│       │   ├── base.py              # Abstract DataAdapter class
│       │   ├── jsonl.py             # JSONL/CSV file adapter (MVP data source)
│       │   ├── schema.py            # Pydantic models for universal JSONL schema validation
│       │   └── splitter.py          # Train/validation/test split logic, deduplication,
│       │                             #   output type detection (JSON/classification/free-text)
│       │
│       ├── optimize/
│       │   ├── __init__.py
│       │   ├── base.py              # Abstract Optimizer class
│       │   ├── gepa.py              # GEPA optimizer wrapper: configures dspy.GEPA,
│       │   │                         #   builds DSPy program/signature, runs optimization
│       │   ├── dspy_program.py       # DSPy module definition: signature("prompt -> response"),
│       │   │                         #   ChainOfThought predictor
│       │   └── metric.py            # DSPy metric function: computes behavioral similarity score,
│       │                             #   returns dspy.Prediction(score=..., feedback=...)
│       │
│       ├── evaluate/
│       │   ├── __init__.py
│       │   ├── base.py              # Abstract Evaluator class, EvalResult dataclass
│       │   ├── bertscore.py         # BERTScore wrapper (optional dep, graceful fallback)
│       │   ├── embedding.py         # Embedding cosine similarity via sentence-transformers
│       │   ├── exact_match.py       # Exact match for classifications
│       │   ├── json_validator.py    # JSON schema validation, basic field-level comparison
│       │   ├── composite.py         # Composite scorer: combines metrics by output type,
│       │   │                         #   computes pairwise win rate, confidence score
│       │   └── types.py             # OutputType detection logic
│       │
│       ├── report/
│       │   ├── __init__.py
│       │   ├── markdown.py          # Generate markdown migration report
│       │   └── templates/
│       │       └── report.md.jinja   # Jinja2 template for markdown report
│       │
│       └── utils/
│           ├── __init__.py
│           ├── logging.py            # Logging config (WARN default, never log prompt content)
│           └── litellm_helpers.py    # LiteLLM convenience wrappers, model info lookups
│
├── tests/
│   ├── conftest.py                   # Shared fixtures: sample prompt pairs, mock LLM responses
│   ├── test_preflight/
│   │   ├── test_capabilities.py
│   │   ├── test_token_budget.py
│   │   └── test_cost_estimator.py
│   ├── test_ingest/
│   │   ├── test_jsonl.py
│   │   ├── test_schema.py
│   │   └── test_splitter.py
│   ├── test_optimize/
│   │   ├── test_gepa.py
│   │   └── test_metric.py
│   ├── test_evaluate/
│   │   ├── test_bertscore.py
│   │   ├── test_exact_match.py
│   │   ├── test_json_validator.py
│   │   └── test_composite.py
│   └── test_cli/
│       └── test_migrate.py
│
├── examples/
│   ├── sample_data.jsonl             # 50 example prompt/response pairs for demo
│   ├── quickstart.py                 # Minimal Python usage example
│   └── migration_config.yaml         # Example config file
│
└── docs/
    └── data-flow.md                  # Documents what data goes where (privacy/security)
```

---

## Phase 2: Evaluation Depth + Redis + Safety
New files only — everything from Phase 1 remains unchanged.

```
src/rosettastone/
│
├── ingest/
│   ├── redis_adapter.py              # ← NEW: Redis ingestion with auto-format detection
│   │                                 #   (RedisVL, LangChain, LiteLLM, GPTCache key prefixes)
│   └── redis_formats.py              # ← NEW: Format-specific parsers for each Redis schema
│
├── optimize/
│   ├── mipro.py                      # ← NEW: MIPROv2 optimizer wrapper (fallback option)
│   └── feedback.py                   # ← NEW: Known-issue feedback encoding into GEPA metric
│                                     #   constraints (2× weighting, textual feedback strings)
│
├── evaluate/
│   ├── llm_judge.py                  # ← NEW: LLM-as-judge pairwise comparison evaluator
│   ├── json_structural.py            # ← NEW: Field-level JSON diff, structural comparison,
│   │                                 #   schema drift detection
│   └── output_detector.py            # ← NEW: Auto-detect output type from response content
│                                     #   (JSON, classification, short text, long text)
│
├── safety/                            # ← NEW directory
│   ├── __init__.py
│   ├── pii_scanner.py                # ← NEW: Regex-based PII detection (email, phone, SSN)
│   │                                 #   during ingestion. Warns, doesn't block.
│   └── prompt_auditor.py             # ← NEW: Post-optimization scan for memorized training
│                                     #   data in compiled prompts (verbatim string matching)
│
└── cli/
    └── main.py                       # Updated: --local-only flag, --feedback flag,
                                      #   colored output, progress bars, summary tables

tests/
├── test_ingest/
│   ├── test_redis_adapter.py         # ← NEW
│   └── test_redis_formats.py         # ← NEW
├── test_evaluate/
│   ├── test_llm_judge.py             # ← NEW
│   └── test_json_structural.py       # ← NEW
├── test_safety/                       # ← NEW
│   ├── test_pii_scanner.py
│   └── test_prompt_auditor.py
└── test_optimize/
    ├── test_mipro.py                 # ← NEW
    └── test_feedback.py              # ← NEW
```

---

## Phase 3: Web UI + Migration Reports
New files only — Phases 1-2 remain unchanged.

```
rosettastone/
│
├── src/rosettastone/
│   │
│   ├── report/
│   │   ├── pdf_generator.py          # ← NEW: PDF export via weasyprint or puppeteer
│   │   ├── html_generator.py         # ← NEW: Interactive HTML report (self-contained)
│   │   ├── narrative.py              # ← NEW: AI-generated natural language summary
│   │   │                             #   for executive audience
│   │   └── templates/
│   │       ├── report.html.jinja     # ← NEW: HTML report template
│   │       └── executive.md.jinja    # ← NEW: Executive summary template
│   │
│   └── server/                        # ← NEW directory
│       ├── __init__.py
│       ├── app.py                    # FastAPI app: serves API + static frontend
│       ├── api/
│       │   ├── __init__.py
│       │   ├── migrations.py         # REST endpoints: list migrations, get migration detail,
│       │   │                         #   get test case, trigger new migration
│       │   ├── comparisons.py        # Endpoints: side-by-side diffs, score distributions
│       │   └── reports.py            # Endpoints: generate/download PDF, HTML reports
│       ├── models.py                 # SQLModel/SQLite schemas for persisting migration results
│       └── static/                   # Built React frontend (served by FastAPI)
│
├── web/                               # ← NEW directory (React frontend source)
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   ├── tailwind.config.js
│   └── src/
│       ├── App.tsx
│       ├── main.tsx
│       ├── api/
│       │   └── client.ts             # API client for FastAPI backend
│       ├── components/
│       │   ├── Dashboard.tsx          # Main dashboard: traffic-light cards, aggregate scores
│       │   ├── EvalGrid.tsx           # TanStack Table: filterable/sortable test case grid
│       │   ├── DiffView.tsx           # react-diff-viewer: side-by-side + unified modes
│       │   ├── ScoreCharts.tsx        # Recharts: histograms, scatter, radar charts
│       │   ├── TestCaseDetail.tsx     # Individual test case: full I/O, diff, eval trace
│       │   ├── PersonaToggle.tsx      # Engineer view ↔ Executive view switch
│       │   └── ExportButton.tsx       # PDF/HTML report download
│       ├── views/
│       │   ├── EngineerView.tsx       # Detailed view: filters, grid, drill-down
│       │   └── ExecutiveView.tsx      # Summary view: traffic lights, recommendations
│       └── types/
│           └── index.ts              # TypeScript types matching backend models
│
├── cli/
│   └── main.py                       # Updated: `rosettastone serve` command to launch web UI

tests/
├── test_server/                       # ← NEW
│   ├── test_api_migrations.py
│   └── test_api_comparisons.py
└── test_report/                       # ← NEW
    ├── test_pdf_generator.py
    └── test_html_generator.py
```

---

## Phase 4: Observability Integrations + Advanced Features
New files only.

```
src/rosettastone/
│
├── ingest/
│   ├── langsmith.py                  # ← NEW: LangSmith adapter (client.list_runs())
│   ├── braintrust.py                 # ← NEW: Braintrust adapter (BTQL queries)
│   ├── opentelemetry.py              # ← NEW: OTel adapter (gen_ai.* span attributes)
│   └── csv_adapter.py                # ← NEW: CSV/spreadsheet import
│
├── optimize/
│   └── improvement.py                # ← NEW: Behavioral cloning + improvement mode
│                                     #   (encode improvement objectives alongside equivalence)
│
├── safety/
│   └── presidio.py                   # ← NEW: Microsoft Presidio integration for PII redaction
│
├── clustering/                        # ← NEW directory
│   ├── __init__.py
│   └── prompt_clusters.py            # ← NEW: Semantic clustering of prompts,
│                                     #   per-category optimization and reporting
│
└── ci/                                # ← NEW directory
    ├── __init__.py
    └── github_action.py              # ← NEW: GitHub Action integration, PR comment generation

.github/
└── actions/
    └── rosettastone-eval/
        └── action.yml                # ← NEW: Reusable GitHub Action for CI/CD eval
```

---

## Phase 5: Pipeline Migration + Enterprise
New files only.

```
src/rosettastone/
│
├── optimize/
│   ├── pipeline_optimizer.py         # ← NEW: Multi-module DSPy pipeline optimization
│   └── teacher_student.py            # ← NEW: BetterTogether pattern (old model as teacher)
│
├── server/
│   ├── api/
│   │   ├── ab_testing.py             # ← NEW: A/B test configuration and metrics
│   │   ├── versioning.py             # ← NEW: Migration history, rollback, comparison
│   │   └── auth.py                   # ← NEW: Multi-user authentication
│   └── models.py                     # Updated: versioning, user, team schemas
│
├── web/src/
│   ├── components/
│   │   ├── ABTestDashboard.tsx        # ← NEW
│   │   ├── MigrationHistory.tsx       # ← NEW
│   │   └── AnnotationQueue.tsx        # ← NEW: Human review interface
│   └── views/
│       └── PipelineView.tsx           # ← NEW: Multi-step pipeline visualization

docker-compose.yml                     # ← NEW: Self-hosted deployment
Dockerfile                             # ← NEW
```

---

## Design Principles Behind the Structure

**1. Each phase is additive, never rewriting.**
The `base.py` abstract classes in `ingest/`, `optimize/`, and `evaluate/` define interfaces in Phase 1 that all later implementations extend. Adding a Redis adapter in Phase 2 means creating `redis_adapter.py` that implements `DataAdapter` — zero changes to existing code.

**2. Optional dependencies map to directory boundaries.**
`evaluate/bertscore.py` imports PyTorch only when called — if the user didn't install `[eval]`, the import fails gracefully with a helpful error message. The `safety/presidio.py` module (Phase 4) is only imported when Presidio is installed. Each directory can declare its own optional deps.

**3. The `core/migrator.py` orchestrator is thin.**
It calls pipeline steps in order but doesn't contain business logic. Each step (preflight, ingest, optimize, evaluate, report) is self-contained. This makes testing straightforward — each step can be unit tested in isolation.

**4. CLI and library share the same code paths.**
`cli/main.py` constructs a `MigrationConfig` from CLI args and passes it to `Migrator.run()`. The library user does the same thing in code. No separate implementation for CLI vs library.

**5. Web UI is completely decoupled.**
The `web/` directory is a standalone React app that talks to `server/` via REST API. The server reads from the same SQLite database that the CLI writes to. You can run a migration via CLI and view results in the web UI, or vice versa.
