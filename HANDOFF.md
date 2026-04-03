# RosettaStone — Handoff Prompt

Paste this entire file as your first message to a new Claude Code session.

---

## Project

RosettaStone is an automated LLM model migration tool. It ingests production prompt/response pairs, optimizes prompts for a target model via DSPy/GEPA, validates behavioral equivalence, and generates migration reports. Full spec in `docs/rosettastone-build-document.md`.

## Current State (as of 2026-04-03)

- **Branch:** `main`, clean, pushed to origin
- **Tests:** 1663 passing, 2 skipped, ruff clean
- **Phase 1–4** features are substantially implemented (see `ROADMAP.md` for full inventory)
- **Bug fix pass complete:** 12 issues from a 5-agent code review were fixed and committed (`965841b`)
  - JSON fence stripping in evaluator (fixes 0% scores when Qwen wraps responses in ```json fences)
  - lm_extra_kwargs propagated to all optimizer paths
  - /metrics auth, XML injection protection, threading lock, ROC calibrator, etc.

## What Needs To Be Done

See `ROADMAP.md` for the full prioritized list. Summary:

### 1. Validate the fence fix (immediate)

No migration with the fixed code has been run yet. The last run (v6) completed with 0% scores because it started before the fix. Run a v7 migration:

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

Expected: `json_valid > 0.0`. If still 0%, diagnose before proceeding.

Note: `--reflection-model` must point to your local endpoint. `gpt-4o` fails on `10.0.10.66:8123`.

### 2. Production P0 — must complete before any real deployment

These five items are all interconnected. Do a single consolidated Alembic migration for all schema changes first, then implement the features.

**P0.1 — Intermediate DB writes + checkpointing**
- `src/rosettastone/server/api/tasks.py`, `src/rosettastone/core/migrator.py`
- `run_migration_background` writes to DB only at start and end. For a 45-min migration, zero intermediate state.
- Fix: write `current_stage` + `stage_progress` + `overall_progress` to `MigrationRecord` at each pipeline stage boundary (preflight, ingest, pii_scan, baseline_eval, optimize, validation_eval, recommendation, report)
- This unblocks SSE streaming, checkpointing, and cost enforcement

**P0.2 — DB-backed task queue**
- `src/rosettastone/server/task_dispatch.py`
- `ThreadPoolExecutor(max_workers=1)` provides zero persistence — process restart = silent job loss
- Fix: add `task_queue` table, polling worker picks up pending tasks, survives restarts

**P0.3 — Security hardening**
- `src/rosettastone/server/app.py`, `src/rosettastone/server/api/auth.py`
- `_JWT_SECRET_DEFAULT = "dev-secret-change-in-production"` is used silently — hard error if unset in multi-user mode
- CSP allows `unsafe-inline` for scripts — replace with nonce-based CSP
- No explicit CORS policy

**P0.4 — Cost guardrails mid-run**
- Preflight estimates cost but nothing enforces a cap once running
- Track live cost via LiteLLM callbacks, abort if `max_cost_usd` exceeded

**P0.5 — Multi-user data isolation**
- `src/rosettastone/server/models.py` + all API endpoints
- `MigrationRecord`, `PipelineRecord`, `ABTest` have no `owner_id`
- Fix: add `owner_id: int | None` FK, scope all queries to session user
- Also fixes the IDOR security issue (S3 from code review)

Reference: `docs/production-roadmap.md` has the full detailed spec for each item including exact file lists, Alembic schema, and implementation notes.

### 3. Web UI — wire the 501 stubs

These are quick wins once P0 is done:

- `GET /api/v1/migrations/{id}/report/html` → connect `src/rosettastone/report/html_generator.py`
- `GET /api/v1/migrations/{id}/report/pdf` → connect `src/rosettastone/report/pdf_generator.py`
- `GET /api/v1/migrations/{id}/report/executive` → connect `src/rosettastone/report/narrative.py`
- Models landing page shows hardcoded dummy data → build `RegisteredModel` table + real backend
- Costs page shows dummy data → aggregate `cost_usd` across migrations
- "New migration" button is non-functional → wire `POST /ui/migrations/new` to background task

Reference: `docs/plans/` has been removed but the work was in `phase3-next-steps.md` (now deleted — content in `ROADMAP.md`)

### 4. Shadow deployment — complete the proxy

`src/rosettastone/shadow/` has logger, evaluator, config. `scripts/shadow_proxy.py` exists but isn't integrated.

- Wire shadow proxy into the main server
- Add documentation and integration tests
- The shadow config YAML is already generated per migration (`shadow_config.yaml` in output dir)

### 5. Tier 2 — calibration + measurement

Reference: `docs/tier2-plan.md` for the full spec.

- The four win thresholds (`json: 0.95`, `classification: 0.90`, `short_text: 0.80`, `long_text: 0.75`) are engineering intuition, never validated
- `src/rosettastone/calibration/` infrastructure is mostly built (types, collector, calibrator with ROC)
- Needs: human label collection pipeline, labeled dataset, ROC calibration run to replace hardcoded thresholds
- Also: multi-run eval (config field exists but pipeline only runs once), per-prompt regression report

### 6. Remaining deferred code review items

From the 5-agent review. Lower priority:

- **B3:** Checkpoint/resume is a no-op for `baseline_eval` and `validation_eval` stages — `migrator.py:218–336`
- **T2:** `cluster/embedder.py` has zero test coverage
- **X2:** PII invariant (never log prompt content) has no lint rule or test assertion enforcing it

### 7. P1 polish

- SSE streaming for live migration progress (requires P0.1)
- PostgreSQL validation + CI job (entire test suite runs against SQLite only)
- Structured logging + request correlation IDs
- Backup automation for SQLite WAL

### 8. PyPI (last)

Only after everything above is stable.

## Key Files to Know

```
src/rosettastone/
├── config.py              — MigrationConfig (all settings, Pydantic v2)
├── core/migrator.py       — Main pipeline orchestrator
├── core/pipeline.py       — Pipeline step definitions
├── evaluate/              — BERTScore, embedding, exact, json_validator, composite
├── optimize/gepa.py       — GEPA optimizer (DSPy)
├── calibration/           — ROC calibrator, types, collector
├── server/
│   ├── app.py             — FastAPI factory
│   ├── models.py          — SQLModel DB tables
│   ├── api/               — JSON API routes
│   ├── routes/ui.py       — UI routes
│   └── templates/         — Jinja2/HTMX templates
├── shadow/                — Shadow deployment (logger, evaluator, config)
└── report/                — Markdown, HTML, PDF report generators

docs/
├── ROADMAP.md             — Full prioritized remaining work (START HERE)
├── production-roadmap.md  — Detailed P0-P2 specs with Alembic schema
├── tier2-plan.md          — Detailed Tier 2 measurement science spec
└── rosettastone-build-document.md — Original full build spec
```

## Dev Commands

```bash
uv sync --dev --all-extras          # install
uv run pytest tests/ -v             # run tests (skip e2e: --ignore=tests/test_e2e)
uv run ruff check src/ tests/       # lint
uv run mypy src/rosettastone/       # type check (35 pre-existing errors, all pre-existing)
uv run uvicorn rosettastone.server.app:create_app --factory  # run server
```

## Execution Strategy

For multi-task implementation work, use `superpowers:subagent-driven-development` skill. Mechanical tasks → Haiku. Security/math/multi-file → Sonnet. Always run `uv run pytest tests/ -q --ignore=tests/test_e2e && uv run ruff check src/ tests/` as gate after each task.
