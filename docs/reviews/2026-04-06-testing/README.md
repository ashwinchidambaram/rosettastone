# Testing Review — 2026-04-06

Service and plugin boundary testing review for RosettaStone. Plan-only: no code written, no tests modified or run.

## 1. Service & Plugin Boundary Inventory

### 1.1 Ingest Adapters & Data Pipeline

**Boundary:** Pluggable `DataAdapter` interface (`ingest/base.py`) with 6 concrete adapters: Redis (SCAN + auto-format detection via registry), LangSmith (list_runs API), Braintrust (experiment logs API), OTel (OTLP JSON file parsing), CSV/TSV, JSONL. Plus `splitter.py` (SHA-256 dedup + train/val/test split) and `schema.py` (field validation).

**External deps:** Redis server (SCAN, GET), LangSmith API (LANGCHAIN_API_KEY), Braintrust API (BRAINTRUST_API_KEY), file system (JSONL, CSV, OTel JSON files).

**Current tests:** ~100+ tests across 9 files. All adapters mocked. Good coverage of happy path, edge cases, import errors, format parsing. Redis formats tested extensively (LiteLLM, LangChain, RedisVL, GPTCache).

**Gaps:** No integration tests against real Redis. No connection resilience (timeout, auth failure, mid-SCAN drop). No cross-adapter output parity. No large dataset pagination. OTel protobuf format unsupported. No property-based tests for splitter dedup/split logic.

**Roadmap items:** T2.5 shadow deployment needs ingest working reliably against production data sources.

### 1.2 Optimization & DSPy Integration

**Boundary:** `Optimizer` base class with GEPA and MIPROv2 concretes. Both use DSPy framework + LiteLLM as backend. GEPA adds reflective optimization with feedback-driven iteration + timeout handling. Metric function (`metric.py`) scores via BERTScore/embedding/string_similarity with optional improvement objectives. `PipelineOptimizer` and `TeacherStudent` add per-module YAML-driven optimization.

**External deps:** DSPy library (>=2.6), LiteLLM (target model + reflection model), LLM API endpoints (OpenAI, Anthropic, local vLLM/Ollama).

**Current tests:** ~30 tests across 10 files. Mock-based. Cover GEPA timeout, metric scoring, MIPROv2 preset selection, pipeline config parsing, teacher-student flow.

**Gaps:** No integration test with real DSPy optimization loop. No test of iteration tracking callback firing. No property-based tests for metric scoring math. No test of GEPA intermediate result capture on timeout. Cost tracking during optimization untested.

**Roadmap items:** T2.2 multi-run eval, P0.4 cost guardrails mid-run.

### 1.3 Evaluation Strategies

**Boundary:** `Evaluator` base class with 6 strategies: BERTScore (distilbert model), Embedding (MiniLM-L6-v2), ExactMatch, JSONValidator (json parse + fence stripping), JSONStructural, LLMJudge (litellm.completion to judge model). `CompositeEvaluator` orchestrates routing by output type with per-type metric weights and win-rate thresholds.

**External deps:** bert-score package (optional), sentence-transformers (optional), LiteLLM (LLM judge, target model completions), numpy.

**Current tests:** ~40 tests across 10 files. Good unit coverage per strategy. JSON fence stripping tested. Multi-run and output type routing tested.

**Gaps:** No integration test of full composite evaluation pipeline with real models. No property-based tests for JSON validator edge cases. LLM judge bidirectional scoring not tested for position bias. Win-rate threshold math (Wilson CI) untested with realistic distributions. No test of evaluator fallback chain (BERTScore unavailable → embedding → string_similarity).

**Roadmap items:** T2.1 human-labeled validation dataset for calibrating thresholds, T2.2 multi-run eval aggregation.

### 1.4 Server HTTP & Security

**Boundary:** FastAPI application with 20+ API endpoints (migrations, comparisons, models, pipelines, costs, alerts, annotations, approvals, audit, teams, users, tasks, reports, SSE, health, versioning, AB testing, shadow, deprecation). Auth middleware: JWT + API key + CSRF. RBAC dependency. Rate limiting (threading lock). CSP nonce middleware. Security headers.

**External deps:** FastAPI/Starlette, Jinja2 templates, static files, SQLModel sessions.

**Current tests:** ~200+ tests across 25+ files. Strong coverage of API contracts, auth flows, CSRF, JWT validation, RBAC, negative/stress testing, health probes, metrics endpoint, rate limiting, SSE.

**Gaps:** P0.3 JWT default secret silently used in multi-user mode. No CORS policy tests. IDOR on audit log + migration detail (S3, blocked on P0.5 owner_id). CSP `unsafe-inline` for scripts. Playwright tests have hardcoded path blocking CI. No contract tests for API schema stability across versions.

**Roadmap items:** P0.3 security hardening, P0.5 multi-user data isolation.

### 1.5 Database, Persistence & Migrations

**Boundary:** Dual database support: SQLite (WAL mode, default) and Postgres (via `DATABASE_URL`). `database.py` manages engine singleton with safety-net column additions. Alembic migration (`c39645f955dc`) creates 16 tables with indexes and FKs. `models.py` defines SQLModel tables. On startup, orphaned `running` migrations recovered to `failed`.

**External deps:** SQLAlchemy, SQLModel, SQLite (filesystem), Postgres (network), Alembic.

**Current tests:** `test_alembic.py` exists (basic). No Postgres integration tests. Server tests use SQLite via TestClient.

**Gaps:** No Alembic upgrade/downgrade roundtrip. No schema parity test (create_all vs Alembic). No Postgres CI job (P1.2). No SQLite WAL concurrent write test. No test of `_migrate_add_columns()` safety net. No data preservation test across migrations. No disk full / corruption resilience.

**Roadmap items:** P0.1 intermediate DB writes, P0.2 DB-backed task queue, P0.5 owner_id migration, P1.2 Postgres CI.

### 1.6 Background Execution & Orchestration

**Boundary:** `task_dispatch.py` (Redis/RQ or fallback DB worker), `task_worker.py`, `pipeline_runner.py` (YAML-driven pipeline execution), `ab_runner.py` (A/B test execution with simulation and live modes, batch commits every 50), `batch.py` (sequential YAML manifest execution), `core/migrator.py` (main pipeline orchestrator: preflight→ingest→eval→optimize→eval→report), `progress.py` (SSE progress tracking).

**External deps:** Redis/RQ (optional, fallback to DB queue), LLM APIs (via pipeline), database sessions.

**Current tests:** Pipeline runner ~10 tests, task dispatch/worker tests, batch tests (10), AB runner tested via API. Migrator tested through core tests (~15).

**Gaps:** P0.1 zero intermediate state during 45-min runs. P0.2 ThreadPoolExecutor has zero persistence. No crash recovery test (server restart mid-migration). No graceful shutdown test (SIGTERM during active task). No test of RQ fallback to DB queue. No concurrent task execution test. AB runner batch-commit partial failure untested.

**Roadmap items:** P0.1 intermediate writes + checkpointing, P0.2 persistent task queue, P1.3 SSE streaming.

### 1.7 Report Generation & CLI

**Boundary:** Report generators: Markdown (`report/markdown.py`), HTML (`report/html_generator.py`), PDF (`report/pdf_generator.py`), Executive/Narrative (`report/narrative.py` + `report/executive_prompt.py`). All use Jinja2 templates from `server/templates/`. CLI via Typer (`cli/main.py`) with migrate, preflight, batch, serve commands. `cli/display.py` for Rich formatting, `cli/ci_output.py` for CI-friendly output.

**External deps:** Jinja2, wkhtmltopdf (PDF), LiteLLM (narrative generation), file system, Typer/Rich.

**Current tests:** ~15 report tests (markdown, HTML, PDF, narrative, executive prompt, rendering). ~22 CLI tests (commands, display). Playwright UI tests cover template rendering (~85 tests).

**Gaps:** HTML/PDF/executive report endpoints are 501 stubs. No test of `--output-dir` flag producing files. No batch CLI E2E test. No test of narrative LLM call failure handling. No PDF generation integration test (wkhtmltopdf dependency). Playwright hardcoded path blocks CI.

**Roadmap items:** Wire 501 report stubs, P5 Starlette deprecation fix.

### 1.8 Safety, Observability & Decision Intelligence

**Boundary:** Safety: `pii_scanner.py` (regex-based PII detection), `presidio_engine.py` (optional Presidio integration), `prompt_auditor.py` (injection detection). Observability: `server/metrics.py` (Prometheus counters/gauges/histograms, optional), `server/logging_config.py`, Sentry SDK (optional). Decision: `recommendation.py` (GO/NO_GO/CONDITIONAL with Wilson CI), `statistics.py` (per-type stats), `ab_stats.py` (A/B significance). Calibration: `calibrator.py` (ROC-based threshold calibration with scikit-learn), `collector.py`, `types.py`. Shadow: `shadow/` (config, evaluator, log_format). Clustering: `cluster/embedder.py` (zero test coverage, T2).

**External deps:** Presidio (optional), Prometheus client (optional), Sentry SDK (optional), scikit-learn (calibration), krippendorff (inter-rater), sentence-transformers (embedder).

**Current tests:** ~10 safety tests, ~10 decision tests, ~5 calibration tests, shadow log format + evaluator + proxy tests. Embedder: zero tests (T2).

**Gaps:** PII invariant not enforced systematically (X2 — no lint rule). Presidio integration untested against real engine. Prometheus metric correctness untested in integration. No Sentry integration test. Calibration ROC accuracy with realistic distributions untested. Shadow proxy not integrated with server. Embedder completely untested. Decision Wilson CI math not property-tested.

**Roadmap items:** T2.1 calibration dataset, T2.5 shadow deployment, X2 PII enforcement, T2 embedder coverage.

---

## 2. Subagent Split

8 testing areas. Each subagent owns one area, writes one report, handles its own fixture design. Cross-area coordination points are flagged in each report's section 6.

| # | Area | Slug | Boundary Count | Key Risk Theme | Estimated Report Size |
|---|------|------|---------------|----------------|----------------------|
| 1 | Ingest Adapters & Data Pipeline | `ingest-adapters` | 6 adapters + splitter + schema | Connection resilience, cross-adapter parity | Medium |
| 2 | Optimization & DSPy Integration | `optimization-engines` | 2 optimizers + metric + DSPy program | LLM API reliability, cost tracking, timeout | Medium |
| 3 | Evaluation Strategies | `evaluation-strategies` | 6 evaluators + composite + thresholds | Fallback chains, scoring accuracy, calibration | Medium |
| 4 | Server HTTP & Security | `server-http-security` | 20+ endpoints + auth + RBAC + rate limit | JWT bypass, IDOR, CORS, API schema stability | Large |
| 5 | Database, Persistence & Migrations | `database-persistence` | SQLite + Postgres + Alembic + models | Schema drift, Postgres parity, WAL conflicts | Medium |
| 6 | Background Execution & Orchestration | `background-orchestration` | 5 runners + migrator + progress | Crash recovery, persistence, graceful shutdown | Medium |
| 7 | Report Generation & CLI | `report-generation-cli` | 4 generators + templates + CLI + display | 501 stubs, output-dir, CI portability | Small-Medium |
| 8 | Safety, Observability & Decision | `safety-observability-decision` | PII + Presidio + metrics + decision + calibration + shadow + embedder | PII enforcement, calibration accuracy, shadow integration | Medium |

**Fixture ownership:**
- `ingest-adapters` owns Redis fixtures, adapter test data generators, cross-adapter parity fixtures
- `optimization-engines` owns DSPy mock fixtures, metric scoring test data
- `evaluation-strategies` owns labeled pair fixtures, evaluator strategy fixtures, golden score files
- `server-http-security` owns TestClient fixtures, auth fixtures, API schema snapshots
- `database-persistence` owns SQLite/Postgres engine fixtures, Alembic runner fixtures
- `background-orchestration` owns task dispatch fixtures, progress tracking fixtures
- `report-generation-cli` owns template rendering fixtures, CLI subprocess fixtures
- `safety-observability-decision` owns PII test data, Prometheus registry fixtures, calibration datasets

**Coordination points (cross-subagent):**
- `database-persistence` ↔ `background-orchestration`: shared DB session fixtures, migration record lifecycle
- `evaluation-strategies` ↔ `optimization-engines`: shared metric fixtures, BERTScore/embedding availability mocking
- `ingest-adapters` ↔ `background-orchestration`: pipeline integration (ingest step in migrator)
- `server-http-security` ↔ `database-persistence`: TestClient needs engine fixtures
- `safety-observability-decision` ↔ `evaluation-strategies`: calibration depends on evaluation scores
