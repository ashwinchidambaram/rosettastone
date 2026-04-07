# RosettaStone Production Readiness Roadmap

**Consolidated from 11 production-readiness analysis reports + codebase audit**
**Generated: 2026-04-01 | Status: All 14 items user-approved**

---

## Section 1: Cross-Cutting Themes

These issues were identified independently by multiple analysis reports, which signals systemic gaps that demand extra attention.

### 1.1 No Intermediate DB Writes During Long-Running Jobs (5 reports)
Reported by: Job Queue, Checkpointing, Streaming, Observability, Cost Guardrails

The `run_migration_background` function in `server/api/tasks.py` writes to the DB exactly twice: once to set `status = "running"` and once at completion with all results. For a 45-minute migration, the DB has zero intermediate state. This blocks checkpointing, streaming progress, cost enforcement mid-run, and operational visibility. Every report's implementation plan requires the same foundational change: adding intermediate DB writes at pipeline stage boundaries inside `Migrator.run()`.

**Bar-raise note:** This is the single highest-leverage change. It unblocks four independent features simultaneously. It should be the very first task in Batch 1.

### 1.2 MigrationRecord Lacks Ownership (3 reports)
Reported by: Multi-User Isolation, Security Hardening, Cost Guardrails

`MigrationRecord`, `PipelineRecord`, and `ABTest` have no `owner_id` or `created_by` foreign key. All authenticated users can see all migrations. This blocks per-user data isolation, per-user budget tracking, and audit attribution. Every report independently recommends adding `owner_id: int | None = Field(default=None, foreign_key="users.id")` to these three tables.

### 1.3 Alembic Migration Conflicts Imminent (6 reports)
Reported by: Multi-User Isolation, Streaming, Checkpointing, Cost Guardrails, PostgreSQL, Observability

Six reports recommend adding new columns to `MigrationRecord` or new tables. If implemented separately, they will produce conflicting Alembic migrations. See Section 6 for the consolidated migration plan.

### 1.4 No Structured Logging or Correlation IDs (3 reports)
Reported by: Observability, Job Queue, Backup/Runbook

There is no request-ID middleware, no structured JSON logging, and no way to correlate a web request with its background job. `logging.getLogger(__name__)` is used everywhere with no structured format. This makes debugging production failures nearly impossible and blocks meaningful alerting.

### 1.5 PostgreSQL Never Tested (3 reports)
Reported by: PostgreSQL, CI, Backup

`database.py` has Postgres branching and `render_as_batch=True` in Alembic (SQLite-specific), but the entire 1438-test suite runs only against SQLite. The CI pipeline has no Postgres job. Docker Compose has a Postgres profile, but no integration test exercises it.

### 1.6 ThreadPoolExecutor Is a Single Point of Failure (4 reports)
Reported by: Job Queue, Streaming, Checkpointing, Observability

`app.state.executor = ThreadPoolExecutor(max_workers=1)` means: (a) only one migration at a time, (b) no job persistence across restarts, (c) no retry on transient failures, (d) orphan recovery marks everything as failed. The job queue is the user's stated P0 priority.

---

## Section 2: Priority Tiers

### P0 -- Blocking (must fix before any production deployment)

| # | Item | Rationale |
|---|------|-----------|
| 1 | **Intermediate DB writes + checkpointing** | Without this, a server restart loses 45+ minutes of work and real money. The orphan recovery marks everything as `failed` with no recovery path. |
| 2 | **DB-backed task table (job queue phase 1)** | ThreadPoolExecutor provides zero persistence. A 60-minute migration vanishes on process restart. DB-backed queue is the user's stated first priority. |
| 3 | **Security hardening (HTTP headers, CORS, JWT secret enforcement)** | `_JWT_SECRET_DEFAULT = "dev-secret-change-in-production"` is used if env var is unset -- this is a critical auth bypass in any real deployment. CSP allows `unsafe-inline`. No CORS policy exists. |
| 4 | **Cost guardrails (max_cost_usd enforcement)** | Preflight estimates cost but nothing enforces a cap. A misconfigured run can spend hundreds of dollars. Per-run `max_cost_usd` is the minimum viable safeguard. |
| 5 | **Multi-user data isolation** | All users see all migrations. In multi-user mode, this is a data privacy violation. |

### P1 -- High Value (first sprint after P0)

| # | Item | Rationale |
|---|------|-----------|
| 6 | **Observability (structured logging, metrics, error tracking)** | Cannot debug production failures without correlation IDs and structured logs. |
| 7 | **PostgreSQL validation + CI job** | Postgres is listed as supported but never tested. Any production deployment on Postgres is flying blind. |
| 8 | **Streaming / live progress (SSE)** | Users stare at a spinner for 45 minutes. SSE + progress bar requires intermediate DB writes (done in P0). |
| 9 | **Rate limiting** | No per-user rate limit on migration submission. One user can monopolize the single-worker executor. |
| 10 | **Backup strategy (Phase 1)** | SQLite WAL mode makes naive `cp` dangerous. Need `sqlite3 .backup` automation. |

### P2 -- Polish (roadmap items)

| # | Item | Rationale |
|---|------|-----------|
| 11 | **Model compatibility matrix** | Zero models formally tested E2E. Must certify Tier 1 pairs before claiming production readiness. |
| 12 | **E2E validation with Ollama** | Free local testing before spending money on API-based E2E. |
| 13 | **RQ integration (job queue phase 2)** | Upgrade from DB-backed polling to Redis Queue for better retry/scheduling. |
| 14 | **Deprecation handling** | Report pending. Model deprecation alerts and automated migration suggestions. |
| 15 | **PyPI publishing** | Last step per user request. Only after everything else is stable. |

---

## Section 3: Execution Plan

### Batch 1: Foundation (Sequential -- must complete before Batch 2)
**Duration: 5 days | Parallel tasks within batch: Yes**

```
Task 1.1 -- Consolidated Alembic Migration
Model: Sonnet
Est: 4 hours
Files: alembic/versions/xxxx_production_readiness.py, src/rosettastone/server/models.py
Description: Single Alembic migration adding ALL new columns and tables needed by P0+P1
  features. See Section 6 for the complete schema. This avoids migration conflicts from
  parallel feature work. Add owner_id to migrations/pipelines/ab_tests, checkpoint fields
  to MigrationRecord, cost budget fields, progress tracking fields, and the task_queue table.
Depends on: Nothing
```

```
Task 1.2 -- Intermediate DB Writes in Migration Runner
Model: Sonnet
Est: 8 hours
Files: src/rosettastone/server/api/tasks.py, src/rosettastone/core/migrator.py,
       src/rosettastone/server/pipeline_runner.py, src/rosettastone/server/ab_runner.py
Description: Refactor run_migration_background to write current_stage, stage_progress,
  and overall_progress to MigrationRecord at each pipeline stage boundary (preflight,
  data_load, pii_scan, baseline_eval, optimize, validation_eval, recommendation, report).
  Each write is a small targeted session commit of just the progress fields. Same treatment
  for pipeline_runner (write PipelineStageRecord per-module as it completes, not in batch
  at the end). This is the foundation for checkpointing, streaming, and cost enforcement.
Depends on: 1.1
```

```
Task 1.3 -- JWT Secret Enforcement + Security Headers Hardening
Model: Opus
Est: 6 hours
Files: src/rosettastone/server/app.py, src/rosettastone/server/api/auth.py,
       src/rosettastone/server/templates/base.html
Description: (a) Refuse to start in multi-user mode if JWT secret is the default dev value
  -- make it a hard error, not a warning. (b) Tighten CSP: replace 'unsafe-inline' for
  scripts with nonce-based CSP (generate nonce per request in SecurityHeadersMiddleware,
  pass to templates). (c) Add Permissions-Policy header. (d) Add explicit CORS middleware
  with configurable allowed origins (default: same-origin only). (e) Pin Tailwind CDN to
  SRI hash or bundle locally. Requires Opus due to security-sensitive template + middleware
  coordination.
Depends on: Nothing (parallel with 1.1/1.2)
```

```
Task 1.4 -- DB-Backed Task Queue (Phase 1)
Model: Sonnet
Est: 10 hours
Files: src/rosettastone/server/models.py (TaskQueue model already in 1.1),
       src/rosettastone/server/task_worker.py (NEW),
       src/rosettastone/server/app.py,
       src/rosettastone/server/api/tasks.py,
       src/rosettastone/server/api/migrations.py
Description: Replace ThreadPoolExecutor with a DB-backed task table. New TaskQueue model
  with columns: id, task_type, payload_json, status (queued/running/complete/failed),
  created_at, started_at, completed_at, worker_id, retry_count, max_retries, error_message.
  New task_worker.py: a background thread that polls TaskQueue every 2 seconds, claims tasks
  with SELECT ... FOR UPDATE SKIP LOCKED (Postgres) or UPDATE WHERE status='queued' (SQLite),
  and runs them. Migration submission writes to TaskQueue instead of executor.submit().
  Orphan recovery on startup marks stale running tasks as queued (retry) instead of failed.
  The worker thread is started in the lifespan context manager where the executor was.
Depends on: 1.1
```

```
Task 1.5 -- Cost Guardrail: max_cost_usd Enforcement
Model: Sonnet
Est: 6 hours
Files: src/rosettastone/config.py, src/rosettastone/core/migrator.py,
       src/rosettastone/server/api/tasks.py, src/rosettastone/server/api/migrations.py,
       src/rosettastone/server/templates/migration_create.html
Description: (a) Add max_cost_usd field to MigrationConfig (default: None = unlimited).
  (b) After preflight cost estimate, if estimate > max_cost_usd, return error immediately.
  (c) During optimization, check accumulated cost (from ctx.costs) against max_cost_usd
  after each stage boundary DB write (from Task 1.2). If exceeded, abort gracefully and
  mark as failed with reason. (d) Add max_cost_usd field to migration creation form/API.
  (e) Show estimated cost in confirmation dialog before submission.
Depends on: 1.2 (needs intermediate DB writes for mid-run checking)
```

```
Task 1.6 -- Multi-User Data Isolation
Model: Sonnet
Est: 8 hours
Files: src/rosettastone/server/api/migrations.py, src/rosettastone/server/api/pipelines.py,
       src/rosettastone/server/api/ab_testing.py, src/rosettastone/server/api/tasks.py,
       src/rosettastone/server/rbac.py
Description: (a) Use the owner_id column added in 1.1. (b) On migration/pipeline/AB-test
  creation, set owner_id = current_user.id from request.state.user. (c) On list endpoints,
  add WHERE owner_id = current_user.id for non-admin users. Admins see all. (d) On
  get/update/delete endpoints, verify owner_id matches or user is admin. (e) Handle
  single-user mode gracefully (owner_id stays NULL, no filtering applied). (f) Backfill:
  existing records with NULL owner_id are visible to admins only until reassigned.
Depends on: 1.1 (needs owner_id columns)
```

### Batch 2: Observability + Postgres (Parallel with Batch 1 tasks 1.3-1.6)
**Duration: 4 days | Fully parallel with late Batch 1 tasks**

```
Task 2.1 -- Structured Logging with Correlation IDs
Model: Sonnet
Est: 8 hours
Files: src/rosettastone/server/app.py (middleware),
       src/rosettastone/server/logging_config.py (NEW),
       src/rosettastone/server/api/tasks.py,
       src/rosettastone/server/pipeline_runner.py,
       src/rosettastone/server/ab_runner.py
Description: (a) Add structlog or JSON-formatted stdlib logging. (b) Add RequestID middleware
  that generates UUID per request, stores in request.state, and adds X-Request-ID response
  header. (c) When a background task is submitted, capture the request_id and pass it to the
  background function. Use it as correlation_id in all log entries from that task. (d) Log
  format: JSON with timestamp, level, logger, request_id, migration_id, message, and
  structured fields (duration_ms, cost_usd, stage). (e) Add log_level env var configuration.
Depends on: Nothing
```

```
Task 2.2 -- Prometheus Metrics Endpoint
Model: Sonnet
Est: 6 hours
Files: src/rosettastone/server/metrics.py (NEW),
       src/rosettastone/server/app.py,
       pyproject.toml (add prometheus-client to web extras)
Description: (a) Add /metrics endpoint using prometheus_client. (b) Key counters:
  migrations_total (by status, source_model, target_model), api_requests_total (by
  method, path, status_code). (c) Key histograms: migration_duration_seconds,
  migration_cost_usd, api_request_duration_seconds. (d) Key gauges: migrations_running,
  task_queue_depth. (e) Instrument middleware for request metrics. (f) Add helper function
  for background tasks to update migration metrics on completion.
Depends on: Nothing
```

```
Task 2.3 -- PostgreSQL CI Job
Model: Haiku
Est: 4 hours
Files: .github/workflows/ci.yml, tests/conftest.py
Description: (a) Add a new CI job `test-postgres` that spins up postgres:16-alpine as a
  service container. (b) Set DATABASE_URL env var pointing to the service. (c) Run the
  full test suite against Postgres. (d) In conftest.py, add a fixture that detects
  DATABASE_URL and uses it instead of the default SQLite. (e) Fix any SQLite-specific
  assumptions in tests (PRAGMA statements, AUTOINCREMENT vs SERIAL).
Depends on: Nothing
```

```
Task 2.4 -- Alembic Postgres Compatibility
Model: Haiku
Est: 3 hours
Files: alembic/env.py, alembic/versions/*.py
Description: (a) Make render_as_batch conditional on dialect: True for SQLite, False for
  Postgres. Modify run_migrations_online() to check engine.dialect.name. (b) Audit existing
  migration files for SQLite-specific syntax. (c) Test alembic upgrade head against Postgres
  in the new CI job.
Depends on: 2.3
```

### Batch 3: UX + Operational (After Batch 1 core completes)
**Duration: 5 days | Tasks within batch are parallel**

```
Task 3.1 -- SSE Streaming Progress
Model: Sonnet
Est: 8 hours
Files: src/rosettastone/server/api/migrations.py (SSE endpoint),
       src/rosettastone/server/progress.py (NEW),
       src/rosettastone/server/app.py (progress_queues in state),
       src/rosettastone/server/templates/migration_detail.html,
       src/rosettastone/server/templates/fragments/migration_progress.html (NEW),
       src/rosettastone/server/api/tasks.py (emit_progress calls)
Description: (a) Add asyncio.Queue per migration ID in app.state.progress_queues.
  (b) Add helper emit_progress() using loop.call_soon_threadsafe for thread-safe
  emission from background tasks. (c) Add GET /api/v1/migrations/{id}/stream SSE
  endpoint with StreamingResponse. Handle catch-up on reconnect from DB state,
  30-second keepalive comments, terminal-state fast path. (d) Add progress bar
  fragment template. (e) Add vanilla JS EventSource listener in migration_detail.html
  that gracefully degrades to existing HTMX polling. (f) CSP connect-src 'self'
  already permits same-origin EventSource -- no change needed.
Depends on: 1.2 (intermediate DB writes)
```

```
Task 3.2 -- GEPA Per-Iteration Progress
Model: Sonnet
Est: 6 hours
Files: src/rosettastone/optimize/metric.py,
       src/rosettastone/optimize/gepa.py,
       src/rosettastone/server/api/tasks.py
Description: Wrap the migration_metric function with a thread-safe counter. After every
  len(trainset) metric calls (= one GEPA iteration), emit an iteration progress event
  with current iteration number and running mean score. Use loop.call_soon_threadsafe
  for the emission since DSPy's internal thread pool calls the metric concurrently.
Depends on: 3.1 (needs SSE infrastructure)
```

```
Task 3.3 -- Rate Limiting
Model: Haiku
Est: 4 hours
Files: src/rosettastone/server/rate_limit.py (NEW),
       src/rosettastone/server/app.py,
       src/rosettastone/server/api/migrations.py,
       src/rosettastone/server/api/pipelines.py
Description: (a) Implement a simple in-memory sliding window rate limiter keyed by
  user_id (multi-user) or IP (single-user). (b) Apply to migration and pipeline
  creation endpoints. Default: 10 submissions per hour per user. Configurable via
  env var ROSETTASTONE_RATE_LIMIT. (c) Return 429 with Retry-After header. (d) For
  future: if Redis is available, use Redis-backed rate limiting for multi-instance
  deployments.
Depends on: Nothing
```

```
Task 3.4 -- Backup Automation (Phase 1)
Model: Haiku
Est: 4 hours
Files: scripts/backup.sh (NEW), scripts/restore.sh (NEW),
       docker-compose.yml (backup service),
       docs/operations/backup-restore.md (NEW -- only because user is doing ops)
Description: (a) SQLite backup script using sqlite3 .backup API (NOT naive cp -- WAL
  mode makes cp dangerous). Rotation: keep 7 daily, 4 weekly. (b) Postgres backup
  script using pg_dump. Same rotation. (c) Docker Compose backup service definition
  (runs as cron container or uses host cron with docker exec). (d) Restore scripts
  with Alembic upgrade head after restore. (e) Operational runbook with step-by-step
  procedures.
Depends on: Nothing
```

```
Task 3.5 -- Per-User Budget Tracking
Model: Sonnet
Est: 6 hours
Files: src/rosettastone/server/models.py (UserBudget table),
       src/rosettastone/server/api/migrations.py,
       src/rosettastone/server/api/tasks.py,
       src/rosettastone/server/api/costs.py,
       src/rosettastone/server/templates/user_profile.html
Description: (a) UserBudget model: user_id, monthly_limit_usd, current_month_spend_usd,
  reset_date. (b) On migration submission, check estimated cost against remaining budget.
  Reject with 402 if over budget. (c) On migration completion, update current_month_spend.
  (d) Admin API to set/view budgets. (e) Monthly reset via a startup check or cron.
Depends on: 1.5 (max_cost_usd), 1.6 (multi-user isolation)
```

### Batch 4: Validation + Hardening (After Batch 2 + 3)
**Duration: 5 days**

```
Task 4.1 -- Ollama E2E Test Suite
Model: Sonnet
Est: 10 hours
Files: tests/test_e2e/test_ollama_migration.py (NEW),
       tests/test_e2e/conftest.py,
       tests/test_e2e/test_ollama_pipeline.py (NEW),
       examples/ollama_sample_data.jsonl (NEW)
Description: (a) Create E2E tests using Ollama local models (ollama/qwen3:8b as
  source, ollama/qwen3.5:4b as target -- both LiteLLM format). (b) Test the full
  pipeline: ingest -> preflight -> baseline eval -> GEPA optimize -> validation eval
  -> report. (c) Use small datasets (10-20 pairs) for fast iteration. (d) Verify
  GO/NO_GO/CONDITIONAL recommendations are sensible. (e) Mark with @pytest.mark.e2e
  and add Ollama availability check fixture. (f) Create a matching pipeline E2E test.
Depends on: Nothing (can start when Ollama is available locally)
```

```
Task 4.2 -- Tier 1 Model Certification
Model: Sonnet
Est: 10 hours
Files: tests/test_e2e/test_model_pairs.py (NEW),
       docs/model-compatibility-matrix.md (NEW),
       src/rosettastone/preflight/checks.py
Description: (a) Define Tier 1 pairs: GPT-4o -> Claude Sonnet 4, Claude Sonnet 4 -> GPT-4o,
  GPT-4o -> Claude Haiku 4.5. (b) Run each pair through the full pipeline with real APIs
  and a canonical 20-pair dataset. (c) Record: success/failure, cost, duration,
  recommendation, confidence score. (d) Document results in compatibility matrix.
  (e) Fix any model-specific issues discovered (e.g., token counting, context window
  detection for Anthropic models).
Depends on: 4.1 (technique validated with free models first)
```

```
Task 4.3 -- Checkpointing: Resume from Last Completed Stage
Model: Sonnet
Est: 8 hours
Files: src/rosettastone/server/api/tasks.py,
       src/rosettastone/core/migrator.py,
       src/rosettastone/server/models.py,
       src/rosettastone/server/api/migrations.py
Description: (a) Use the checkpoint_stage and checkpoint_data_json fields added in 1.1.
  (b) After each stage completes, serialize its output to checkpoint_data_json. For
  baseline_eval: serialize EvalResults. For optimize: serialize optimized_prompt string.
  (c) On task resume (triggered by re-queuing a failed migration), read checkpoint_stage
  and skip completed stages by loading their outputs from checkpoint_data_json.
  (d) Add "Resume" button to migration detail UI for failed migrations that have a
  checkpoint. (e) Ensure the task queue worker (from 1.4) supports re-queuing.
Depends on: 1.2, 1.4
```

```
Task 4.4 -- Error Tracking Integration (Sentry)
Model: Haiku
Est: 3 hours
Files: src/rosettastone/server/app.py, pyproject.toml,
       docker-compose.yml (SENTRY_DSN env var)
Description: (a) Add optional sentry-sdk[fastapi] dependency. (b) In create_app,
  initialize Sentry if SENTRY_DSN env var is set. (c) Configure: traces_sample_rate=0.1,
  attach user context from request.state.user, tag with migration_id where applicable.
  (d) Lazy import with try/except ImportError fallback.
Depends on: Nothing
```

### Batch 5: Job Queue Phase 2 + Polish (After Batch 4)
**Duration: 4 days**

```
Task 5.1 -- RQ (Redis Queue) Integration
Model: Sonnet
Est: 8 hours
Files: src/rosettastone/server/rq_worker.py (NEW),
       src/rosettastone/server/task_dispatch.py (NEW),
       src/rosettastone/server/app.py,
       docker-compose.yml, pyproject.toml
Description: (a) Add rq as optional dependency in a new extras group. (b) Create
  task_dispatch.py that abstracts over DB-backed queue and RQ -- if REDIS_URL is
  set and rq is installed, use RQ; otherwise fall back to DB-backed queue. (c) Create
  rq_worker.py entry point for running `rq worker`. (d) Add rq-worker service to
  docker-compose with redis profile. (e) Support job cancellation, retry with
  exponential backoff, and failed job inspection.
Depends on: 1.4 (DB-backed queue as fallback)
```

```
Task 5.2 -- Deprecation Handling (Phase A: Quick Wins)
Model: Sonnet
Est: 8 hours
Files:
  src/rosettastone/server/api/alerts.py (line 152: add check_deprecations call)
  src/rosettastone/server/api/migrations.py (line 1163: alerts_page; line 882: models_page)
  src/rosettastone/server/api/deprecation.py (fix past-deprecation message; expand provider coverage)
  src/rosettastone/server/api/models.py (lines 38-48: _model_to_template_dict)
  src/rosettastone/preflight/checks.py (add deprecation check to preflight)
  src/rosettastone/core/deprecations.py (NEW -- extract lookup logic from server module)
Description:
  Phase A -- Quick wins (all high/critical gaps from deprecation report):
  (a) Fix POST /api/v1/alerts/generate to call check_deprecations(session) -- currently
      only generates migration-event alerts; deprecation alerts only trigger at startup.
  (b) Fix GET /ui/alerts page load to also call check_deprecations(session). Currently
      _generate_alerts() is called but check_deprecations() is not.
  (c) Fix "Retiring in -N days" message for already-deprecated models (days_until < 0).
      Show "Already deprecated (retired on DATE)" instead.
  (d) Fix GET /ui/models (line 882): replace hardcoded DUMMY_ALERTS with live DB query
      filtered to alert_type="deprecation".
  (e) Fix _model_to_template_dict() in models.py: cross-reference KNOWN_DEPRECATIONS
      to populate retirement_date and replacement fields for real registered models.
  (f) Extract _load_custom_deprecations() to rosettastone/core/deprecations.py so
      preflight can import it without the web/SQLModel stack. Add deprecation check to
      preflight run_all_checks(): warning if target_model is deprecated (< 0 days),
      blocker if target model's deprecation date has already passed.
  (g) Expand KNOWN_DEPRECATIONS to cover Anthropic and Google model sunset dates.
  (h) Add severity re-evaluation on re-check: update warning->critical as 30-day
      threshold approaches (instead of leaving original severity forever).
  Phase B (future): periodic asyncio re-check, GET /api/v1/deprecations endpoint,
      LiteLLM model-disappearance detection, Google API polling, webhook delivery.
Depends on: Nothing
```

```
Task 5.3 -- Ray Cluster E2E Testing
Model: Opus
Est: 8 hours
Files: tests/test_e2e/test_ray_migration.py (NEW),
       src/rosettastone/optimize/gepa.py (Ray-aware configuration),
       docs/ray-setup.md (NEW)
Description: (a) Configure DSPy to use Ray-served OSS models via OpenAI-compatible
  endpoint. (b) Test GEPA optimization with Ray-served models (e.g., Llama-3).
  (c) Benchmark: compare optimization time and quality between Ray models and
  commercial APIs. (d) Document Ray setup for self-hosted deployments. Requires
  Opus for complex distributed system coordination.
Depends on: 4.1 (Ollama E2E validates the test framework)
```

### Batch 6: PyPI Publishing (Final)
**Duration: 2 days**

```
Task 6.1 -- PyPI Packaging Preparation
Model: Haiku
Est: 4 hours
Files: pyproject.toml, src/rosettastone/__init__.py,
       MANIFEST.in (or hatch config), .github/workflows/publish.yml (NEW)
Description: (a) Verify/fix package metadata: description, authors, classifiers, URLs.
  (b) Ensure all non-Python files are included: templates/*.html, static/js/*.js,
  static/css/*.css, report/templates/*.md -- verify with hatch build + check wheel
  contents. (c) Set up version management (pin to pyproject.toml, expose as
  __version__). (d) Check PyPI name availability -- if "rosettastone" is taken, use
  "rosettastone-llm". (e) Create GitHub Actions publish workflow triggered on version
  tag, using Trusted Publishers (OIDC).
Depends on: Everything else
```

```
Task 6.2 -- TestPyPI Dry Run + Final Publish
Model: Haiku
Est: 2 hours
Files: .github/workflows/publish.yml
Description: (a) Publish to TestPyPI first. (b) Install from TestPyPI in a fresh venv,
  verify CLI works, verify server starts, verify templates render. (c) Publish to
  real PyPI. (d) Verify pip install rosettastone[all] works.
Depends on: 6.1
```

---

## Section 4: Risk Register

### Risk 1: DSPy GEPA API Instability (HIGH)
**Impact:** Per-iteration progress tracking (Task 3.2) depends on wrapping DSPy's metric function to count iterations. DSPy's internal threading model and GEPA callback interface are not stable public APIs and can change between versions.
**Mitigation:** Pin DSPy version in pyproject.toml. Use the metric-wrapping approach (Option A from the streaming report) which is the least invasive. Write an integration test that verifies the iteration counter fires correctly. If DSPy changes the threading model, fall back to polling a shared counter (Option C).

### Risk 2: SQLite Concurrent Write Contention (HIGH)
**Impact:** The intermediate DB writes (Task 1.2) and the DB-backed task queue polling (Task 1.4) add significantly more write traffic to SQLite. With WAL mode and a single writer, this should be fine -- but under load (multiple concurrent HTTP requests + background writer + task queue poller), SQLite's write serialization could cause `database is locked` errors with the 30-second timeout.
**Mitigation:** (a) Keep writes minimal (only progress fields, not full result data). (b) Use short transactions with immediate commits. (c) Document that PostgreSQL is recommended for production multi-user deployments. (d) Add a health check metric for DB write latency.

### Risk 3: Alembic Migration on Existing Deployments (MEDIUM)
**Impact:** The consolidated migration (Task 1.1) adds ~10 new columns and 1 new table. Existing deployments that were created before Alembic (using create_all) may not have the alembic_version table, causing `alembic upgrade head` to fail.
**Mitigation:** (a) The existing `_migrate_add_columns()` in database.py acts as a safety net. (b) Add a migration step that stamps the initial revision if alembic_version is empty. (c) Document the upgrade path for pre-Alembic deployments: `alembic stamp head` then `alembic upgrade head`.

### Risk 4: SSE Connection Management at Scale (MEDIUM)
**Impact:** Each active SSE connection holds an HTTP connection open. With the single-worker Uvicorn default and many concurrent viewers of a running migration, the server could run out of connection slots.
**Mitigation:** (a) SSE connections are lightweight async generators -- Uvicorn handles thousands on a single event loop. (b) Set a maximum SSE connections per migration (e.g., 10). (c) Add X-Accel-Buffering: no header for nginx proxy compatibility. (d) Implement 30-second keepalive comments to prevent proxy timeouts.

### Risk 5: Cost Guardrail Bypass via API (MEDIUM)
**Impact:** The cost guardrail (Task 1.5) checks estimated cost before starting. But actual cost can exceed the estimate if the model pricing changes mid-run or if GEPA runs more iterations than expected. The mid-run check at stage boundaries provides partial protection, but the optimizer stage is a single long-running call to DSPy that cannot be interrupted mid-iteration.
**Mitigation:** (a) Overestimate cost by 20% in the preflight check (add a safety margin). (b) The GEPA metric wrapper (Task 3.2) can also check accumulated cost and raise an exception to abort the optimizer early. (c) Log a warning if actual cost exceeds estimate by more than 30%.

---

## Section 5: Total Effort Summary

| Batch | Duration | Haiku Hours | Sonnet Hours | Opus Hours | Key Deliverables |
|-------|----------|-------------|--------------|------------|------------------|
| 1: Foundation | 5 days | 0 | 36 | 6 | Alembic migration, intermediate writes, task queue, cost guardrails, JWT hardening, multi-user isolation |
| 2: Observability + Postgres | 4 days | 7 | 14 | 0 | Structured logging, Prometheus metrics, Postgres CI, Alembic compatibility |
| 3: UX + Operational | 5 days | 8 | 20 | 0 | SSE streaming, GEPA progress, rate limiting, backup automation, per-user budgets |
| 4: Validation + Hardening | 5 days | 3 | 28 | 0 | Ollama E2E, Tier 1 model certification, checkpointing resume, Sentry |
| 5: Job Queue Phase 2 + Polish | 4 days | 4 | 8 | 8 | RQ integration, deprecation handling, Ray E2E |
| 6: PyPI Publishing | 2 days | 6 | 0 | 0 | Package prep, TestPyPI, production publish |
| **Total** | **~25 days** | **28 hrs** | **106 hrs** | **14 hrs** | **148 hours total** |

**Confidence level: MEDIUM-HIGH.** The estimates assume a single developer + subagent workflow. Batches 1-3 are well-scoped with clear file-level targets. Batches 4-5 have more uncertainty because E2E testing with real models can surface unexpected issues. The 25-day calendar estimate assumes Batch 2 runs in parallel with late Batch 1 tasks.

**Critical path:** Task 1.1 (migration) -> Task 1.2 (intermediate writes) -> everything else. If Task 1.1 and 1.2 slip, the entire roadmap shifts.

---

## Section 6: Alembic Migration Plan

### Consolidated Migration: `production_readiness_v1`

All schema changes from all reports consolidated into a single migration to avoid conflicts.

**New columns on `migrations` table:**

```python
# Checkpointing (from Job Queue + Checkpointing reports)
checkpoint_stage: str | None       # Last completed stage name
checkpoint_data_json: str | None   # Serialized stage output for resume

# Progress tracking (from Streaming report)
current_stage: str | None          # Currently executing stage
stage_progress: float | None       # 0.0-1.0 within current stage
overall_progress: float | None     # 0.0-1.0 across all stages

# Cost guardrails (from Cost Guardrails report)
max_cost_usd: float | None         # User-set spending cap
estimated_cost_usd: float | None   # Preflight estimate

# Ownership (from Multi-User Isolation report)
owner_id: int | None               # FK to users.id
```

**New columns on `pipelines` table:**

```python
owner_id: int | None               # FK to users.id
overall_progress: float | None     # 0.0-1.0
current_module: str | None         # Currently executing module name
```

**New columns on `ab_tests` table:**

```python
owner_id: int | None               # FK to users.id
```

**New table: `task_queue`**

```python
class TaskQueue(SQLModel, table=True):
    __tablename__ = "task_queue"

    id: int | None = Field(default=None, primary_key=True)
    task_type: str                   # "migration" | "pipeline" | "ab_test"
    resource_id: int                 # FK to migrations.id / pipelines.id / ab_tests.id
    payload_json: str = "{}"         # Serialized task parameters
    status: str = "queued"           # queued | running | complete | failed | cancelled
    priority: int = 0               # Higher = more urgent
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    worker_id: str | None = None    # Hostname/PID of claiming worker
    retry_count: int = 0
    max_retries: int = 3
    error_message: str | None = None
    correlation_id: str | None = None  # Request ID for log correlation
```

**New table: `user_budgets`** (from Cost Guardrails report)

```python
class UserBudget(SQLModel, table=True):
    __tablename__ = "user_budgets"

    id: int | None = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="users.id", unique=True)
    monthly_limit_usd: float = 100.0
    current_month_spend_usd: float = 0.0
    budget_month: str               # "2026-04" format, for reset detection
```

**Alembic migration implementation notes:**
- Must use `render_as_batch=True` for SQLite (already configured in env.py)
- All new columns are nullable or have defaults -- no data migration needed
- Foreign key `owner_id -> users.id` should NOT be enforced at DB level for backwards compatibility (users table may not exist in single-user mode). Enforce in application code only.
- The `task_queue` table needs an index on `(status, priority, created_at)` for efficient polling
- Run `alembic revision --autogenerate -m "production_readiness_v1"` after all model changes are in place
- Test with both SQLite and PostgreSQL before merging

**env.py change required:**

```python
# Make render_as_batch conditional on dialect
render_as_batch = connection.dialect.name == "sqlite"
context.configure(
    connection=connection,
    target_metadata=target_metadata,
    render_as_batch=render_as_batch,
)
```

This ensures Postgres gets native ALTER TABLE operations (which are faster and support more operations) while SQLite continues to use batch mode.
