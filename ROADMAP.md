# RosettaStone — Roadmap & Remaining Work

**Last updated:** 2026-04-17
**Current state:** Phase 1-4 substantially complete. P0 production hardening complete. Pending: production validation, P1 hardening, Tier 2 measurement science.

---

## Completed

The following items were previously listed as outstanding but are now fully implemented.

### Production Hardening (P0)

| Item | What Was Done |
|------|---------------|
| P0.1 Intermediate DB Writes + Checkpointing | `_make_progress_writer()` in `tasks.py` writes `current_stage`, `stage_progress`, `overall_progress` at each stage boundary. `_make_checkpoint_writer()` persists stage output. Resume logic in `migrator.py`. |
| P0.2 DB-Backed Task Queue | `TaskQueue` table in `models.py`. `TaskWorker` in `task_worker.py`. `TaskDispatcher` in `task_dispatch.py` with RQ fallback. Stale task recovery on startup. |
| P0.3 Security Hardening | `RuntimeError` if JWT secret is default in multi-user mode. Nonce-based CSP for scripts and styles. CORS via `ROSETTASTONE_CORS_ORIGINS` env var. CSRF double-submit cookie. |
| P0.4 Cost Guardrails Mid-Run | `CostLimitExceeded` exception. Pre-run cap check in `migrator.py`. Mid-run enforcement via LiteLLM success callback. Per-user budget tracking in `costs.py`. |
| P0.5 Multi-User Data Isolation | `owner_id` on `MigrationRecord`, `PipelineRecord`, `ABTest`. List endpoints filter by owner. Detail endpoints use `check_resource_owner()`. Admin bypass. Audit log scoped. All 14 migration sub-endpoints secured. |

### Deferred Code Review Items (Resolved)

| ID | Resolution |
|----|------------|
| B3 | Checkpoint/resume is fully implemented for `baseline_eval` and `validation_eval` — both have restore logic. |
| B4 | `chart.min.js` inlined from static bundle via `Markup()`. PDF path uses weasyprint. |
| BG4 | Edge case handled with `max(1, ...)` and pop-from-val fallback. Well-tested. |
| S3 | Audit log scoped to user. All migration endpoints use `check_resource_owner()`. Fixed as part of P0.5. |
| T2 | 31 tests, 654 lines in `test_embedder.py`. |
| T4 | `test_auth_jwt.py`, `test_auth_csrf.py`, `test_auth_utils.py` all exist. |
| X2 | Scanner → safety_warnings → make_recommendation → NO_GO chain is correctly wired. |
| C3 | `with_for_update(skip_locked=True)` added to `_claim_next_task()`. |

### Web UI — Previously Marked as "501 Stubs"

All report endpoints are fully implemented. The prior "501 stub" label was misleading — these are graceful degradation paths for optional deps (`weasyprint`, `sentry`), not unimplemented stubs.

| Endpoint | Implementation |
|----------|----------------|
| HTML report | `report/html_generator.py` |
| PDF report | `report/pdf_generator.py` (requires weasyprint) |
| Executive report | `report/narrative.py` |
| Markdown report | Fully implemented |
| `RegisteredModel` CRUD | `api/models.py` — list, create, delete, info, import-from-migrations |
| Alert system | `api/alerts.py` — list, generate, mark-read, delete |
| Per-prompt regression | `PromptRegression` type in `types.py`, regressions endpoint in `migrations.py` |
| Dashboard data | `DUMMY_MODELS`/`DUMMY_ALERTS` removed — dashboard queries real DB |

---

## Immediate — Validate the Fence Fix

A migration with the fixed JSON evaluator (fence-stripping) has not been run yet. The v6 migration ran before the fix and still scored 0%.

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

## Production Readiness — P1 (High Value)

### P1.1 — Structured Logging + Correlation IDs
`RequestIDMiddleware` exists but there is no structured JSON logging. Debugging production failures requires structured log output and correlation IDs threaded through request context.

### P1.2 — PostgreSQL Validation + CI Job
Postgres branching exists in `database.py` but the full test suite runs only against SQLite. No CI job exercises Postgres end-to-end.

### P1.3 — Streaming / Live Progress (SSE)
SSE endpoint and progress writer both exist but are not fully wired together. Users still see a spinner for long migrations. P0.1 (intermediate DB writes) is now complete, so this is unblocked.

### P1.4 — Rate Limiting (per-user)
Threading lock added (T9 fix). Still missing: per-user rate limit on migration submission. One user can still monopolize the task queue.

### P1.5 — SQLite Backup Automation
SQLite WAL mode makes naive `cp` dangerous. Need `sqlite3 .backup` automation and a runbook.

---

## Web UI — Remaining Wiring

| Priority | Task | Status |
|----------|------|--------|
| P2 | Cost tracking aggregation | `cost_usd` exists per-migration; full aggregation across users/time TBD |
| P4 | Migration trigger from UI | "New migration" button is non-functional |
| P5 | Starlette TemplateResponse signature fix | Deprecation warning still present |
| P5 | Consolidate Jinja2Templates instances | Multiple instances across server |
| P5 | Mobile responsive nav | Not in current designs |

---

## Tier 2 — Measurement Science

Requires P1 production foundation to be meaningful in production.

### T2.1 — Human-Labeled Validation Dataset
The four hardcoded win thresholds (`json: 0.95`, `classification: 0.90`, `short_text: 0.80`, `long_text: 0.75`) were set by engineering intuition, not validated against ground truth. Infrastructure in `src/rosettastone/calibration/` is in place; no labeled data exists yet.

**Work:** Build label schema (binary PRODUCTION_SAFE + diagnostic Likert dimensions), collect ~500 labeled pairs across output types, run ROC calibration to replace hardcoded thresholds.

### T2.2 — Multi-Run Evaluation
`eval_runs` config field and aggregation strategy are wired in `config.py` but the pipeline only runs evaluation once. Need to actually run N times and aggregate (median/mean/p25).

### T2.4 — Actual End-to-End Cost Tracking
Cost is estimated via preflight and tracked mid-run via LiteLLM callbacks, but reliable per-migration actual cost (not estimated) is not yet surfaced in reports or the UI.

### T2.5 — Shadow Deployment Proxy
`src/rosettastone/shadow/` has logger, evaluator, and config. `scripts/shadow_proxy.py` exists but is not wired into the main server. Needs: server route integration, documentation, and production-hardening.

---

## P2 — Polish & PyPI

| Item | Notes |
|------|-------|
| Model compatibility matrix | No model pairs formally E2E certified |
| E2E validation with Ollama | Free local testing path; unblocked now |
| RQ integration (job queue phase 2) | Upgrade from DB-backed polling to RQ |
| Deprecation handling | Alerts when source/target model deprecated |
| PyPI publishing | Last step; only after everything else stable |

---

## Recommendation: Order of Attack

1. **Run v7 migration** (background, 3-5h) — validates the fence fix is working
2. **P1.3 SSE live progress** — now unblocked by P0.1; highest user-visible win
3. **T2.5 shadow proxy** — complete the shadow deployment story; low-risk wiring
4. **T2.1 calibration dataset** — label collection + threshold calibration
5. **P1.1 structured logging** — necessary before any real production usage
6. **P1.2 Postgres CI job** — catches SQLite-only regressions
7. **Web UI wiring** — cost aggregation, migration trigger button
8. **T2.2 multi-run eval + T2.4 actual cost tracking** — measurement quality
9. **P1.4 rate limiting + P1.5 backup** — operational hygiene
10. **PyPI** — last
