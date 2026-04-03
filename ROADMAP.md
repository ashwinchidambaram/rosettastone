# RosettaStone — Roadmap & Remaining Work

**Last updated:** 2026-04-03
**Current state:** Phase 1-4 substantially complete. Bug fix pass done. Pending: end-to-end validation, production hardening, Tier 2 measurement science.

---

## Immediate — Validate the Fence Fix

A migration with fixed JSON evaluator (fence-stripping) has not been run yet. The v6 migration ran before the fix and still scored 0%.

**Run v7 migration** (note: use `--reflection-model` pointing to your local endpoint, since `gpt-4o` fails on 10.0.10.66):

```bash
uv run rosettastone migrate \
  --data examples/datasets/fintech_extraction/fintech_extraction_gpt4o.jsonl \
  --from openai/gpt-4o \
  --to "openai/Qwen/Qwen3.5-27B" \
  --reflection-model "openai/Qwen/Qwen3.5-27B" \
  --lm-extra-kwargs '{"api_base": "http://10.0.10.66:8123/v1", "api_key": "dummy"}' \
  --auto light \
  --output ./migration_output_gpt4o_to_qwen_v7 \
  > /tmp/migration_gpt4o_qwen_v7.log 2>&1 &
```

Expected: `json_valid > 0.0`, non-zero composite scores, valid GO/CONDITIONAL/NO_GO.

---

## Deferred Code Review Items

Surviving items from the 5-agent code review (`issues.md`). All high-priority bugs were fixed (T1–T12). These remain:

| ID | Issue | Why Deferred | Priority |
|----|-------|-------------|----------|
| B3 | Checkpoint/resume is a no-op for baseline_eval + validation_eval | Complex rewrite; current behavior re-runs (not dangerous) | Medium |
| B4 | HTML report `chart_js_source` missing in PDF path | Non-critical; PDF charts broken | Low |
| BG4 | `split_data` returns empty val set for ≤2 pairs | Only triggers with exactly 2 pairs after dedup | Low |
| S3 | IDOR on audit log + migration detail (no user scoping) | Requires owner_id on MigrationRecord (see P0 below) | High — blocked on P0.5 |
| T2 | `cluster/embedder.py` has zero test coverage | Deferred; embedder is used but untested | Medium |
| T4 | Auth middleware multi-user + API-key simultaneous mode untested | Test gap | Low |
| X2 | PII invariant not enforced systematically (no lint rule) | Should add test assertion or CI check | Medium |
| C3 | task_worker TOCTOU race | Agent found it's likely atomic via SQLAlchemy; verify before fixing | Low |

---

## Production Readiness — P0 (Blocking)

From `docs/production-roadmap.md`. These must be done before any real deployment.

### P0.1 — Intermediate DB Writes + Checkpointing
**Files:** `src/rosettastone/server/api/tasks.py`, `src/rosettastone/core/migrator.py`, `src/rosettastone/server/pipeline_runner.py`

`run_migration_background` writes to DB exactly twice (start + end). For a 45-minute migration, zero intermediate state exists. A server restart loses all progress.

**Fix:** Write `current_stage`, `stage_progress`, and `overall_progress` to `MigrationRecord` at each pipeline stage boundary. This unblocks SSE streaming, checkpointing, and cost enforcement mid-run.

### P0.2 — DB-Backed Task Queue
**Files:** `src/rosettastone/server/task_dispatch.py`, new `task_queue` table

`ThreadPoolExecutor(max_workers=1)` provides zero persistence. Process restart = silent job loss.

**Fix:** Add a `task_queue` DB table. Polling worker picks up pending tasks. Survives restarts. (RQ as phase 2 upgrade after this works.)

### P0.3 — Security Hardening
**Files:** `src/rosettastone/server/app.py`, `src/rosettastone/server/api/auth.py`

- `_JWT_SECRET_DEFAULT = "dev-secret-change-in-production"` is used silently in multi-user mode — critical auth bypass
- CSP allows `unsafe-inline` for scripts
- No CORS policy

**Fix:** Hard error if JWT secret is default in multi-user mode. Nonce-based CSP. Explicit CORS origins.

### P0.4 — Cost Guardrails Mid-Run
**Files:** `src/rosettastone/core/migrator.py`, `src/rosettastone/server/api/tasks.py`

Preflight estimates cost but nothing enforces a cap once the migration is running. A misconfigured run can spend hundreds of dollars.

**Fix:** Track live cost via LiteLLM callbacks, abort if `max_cost_usd` exceeded mid-run.

### P0.5 — Multi-User Data Isolation
**Files:** `src/rosettastone/server/models.py`, all API endpoints

`MigrationRecord`, `PipelineRecord`, `ABTest` have no `owner_id`. All authenticated users see all migrations. Fixes S3 (IDOR) as a side effect.

**Fix:** Add `owner_id: int | None` FK to these tables. Scope all list/detail queries to session user.

> **Note:** P0.1–P0.5 all touch `MigrationRecord`. Do a single consolidated Alembic migration adding all new columns at once to avoid conflicts.

---

## Production Readiness — P1 (High Value)

### P1.1 — Structured Logging + Correlation IDs
No request-ID middleware, no structured JSON logging. Debugging production failures is nearly impossible.

### P1.2 — PostgreSQL Validation + CI Job
Postgres branching exists in `database.py` but the full 1663-test suite runs only against SQLite. No CI job exercises Postgres.

### P1.3 — Streaming / Live Progress (SSE)
Users stare at a spinner for 45+ minutes. Requires P0.1 (intermediate DB writes) first.

### P1.4 — Rate Limiting (per-user)
Threading lock added (T9). Still missing: per-user rate limit on migration submission. One user can monopolize the single executor.

### P1.5 — Backup Strategy
SQLite WAL mode makes naive `cp` dangerous. Need `sqlite3 .backup` automation + runbook.

---

## Web UI Wiring (Phase 3 Remaining)

From `docs/plans/phase3-next-steps.md`:

| Priority | Task | Status |
|----------|------|--------|
| P0 | Wire HTML report endpoint (`GET /migrations/{id}/report/html`) | 501 stub → `report/html_generator.py` |
| P0 | Wire PDF report endpoint (`GET /migrations/{id}/report/pdf`) | 501 stub → `report/pdf_generator.py` |
| P0 | Wire executive report API | 501 stub → `report/narrative.py` |
| P1 | `RegisteredModel` table + models backend | Landing page shows hardcoded dummy data |
| P2 | Cost tracking aggregation | `cost_usd` exists per-migration, not aggregated |
| P3 | Alert system (`Alert` table, deprecation/price detection) | Pure dummy data |
| P4 | Migration trigger from UI | "New migration" button is non-functional |
| P5 | Starlette TemplateResponse signature fix | Deprecation warning |
| P5 | Consolidate Jinja2Templates instances | Multiple instances across server |
| P5 | Mobile responsive nav | Not in current Stitch designs |

---

## Tier 2 — Measurement Science

From `docs/tier2-plan.md`. Requires P0 production foundation first.

### T2.1 — Human-Labeled Validation Dataset
The four hardcoded win thresholds (`json: 0.95`, `classification: 0.90`, `short_text: 0.80`, `long_text: 0.75`) were set by engineering intuition, never validated against ground truth.

**Work:** Build label schema (binary PRODUCTION_SAFE + diagnostic Likert dimensions), collect ~500 labeled pairs across output types, run ROC calibration to replace hardcoded thresholds with data-driven ones. Infrastructure in `src/rosettastone/calibration/` is mostly in place.

### T2.2 — Multi-Run Evaluation
`eval_runs` config field and aggregation strategy are wired in `config.py` but the pipeline only runs evaluation once. Need to actually run N times and aggregate (median/mean/p25).

### T2.3 — Per-Prompt Regression Report
Current report shows aggregate metrics. Should show per-pair pass/fail with drill-down to identify which prompts regress.

### T2.4 — Actual Cost Tracking
Cost is estimated via preflight but actual cost per migration isn't reliably tracked end-to-end.

### T2.5 — Shadow Deployment (complete the proxy)
`src/rosettastone/shadow/` has logger, evaluator, config. `scripts/shadow_proxy.py` exists but isn't wired into the main server or documented. Needs: integration with server routes, documentation, and production-hardening.

---

## P2 — Polish & PyPI

| Item | Notes |
|------|-------|
| Model compatibility matrix | Zero model pairs formally E2E certified |
| E2E validation with Ollama | Free local testing path |
| RQ integration (job queue phase 2) | Upgrade from DB-backed polling |
| Deprecation handling | Alerts when source/target model deprecated |
| PyPI publishing | Last step; only after everything else stable |

---

## Recommendation: Order of Attack

1. **Run v7 migration** (background, 3-5h) — validates the fence fix is working
2. **P0 production hardening** — single consolidated Alembic migration + intermediate DB writes + task queue; this is the highest-leverage batch
3. **Wire report 501 stubs** — quickest wins for UI usability
4. **T2.5 shadow proxy** — complete the shadow deployment story
5. **T2.1 calibration dataset** — label collection + threshold calibration
6. **Models backend + cost tracking** — rounds out the UI
7. **P1 items** — SSE streaming, Postgres CI, structured logging
8. **PyPI** — last
