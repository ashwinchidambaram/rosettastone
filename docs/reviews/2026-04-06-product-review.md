# RosettaStone — Product & Engineering Review

**Date:** 2026-04-06
**Branch:** review/product-2026-04-06
**Reviewer:** Claude (autonomous, plan-only)
**Worklog:** docs/reviews/2026-04-06-worklog.md

---

## Pass 0 — Target User Definition

### Positioning vs Reality

The README positions RosettaStone as an enterprise-grade platform: teams, RBAC, approval workflows, annotations, audit logs, Docker + Postgres deployment, multi-user JWT auth, A/B testing. The README's roadmap shows 7 phases "complete."

The reality is a solo-authored project by an AI engineer running migrations against local vLLM clusters from an M3 MacBook Air. No team, no production deployment, no external users. Benchmark result tables are entirely empty (all dashes). Several enterprise features are UI shells with dummy data or 501 stubs behind them.

This isn't a criticism — it's a scope observation. The core migration pipeline is genuinely sophisticated (GEPA integration, multi-strategy evaluation, output-type-aware scoring). The enterprise scaffolding grew ahead of the core validation story.

### Default User

A solo AI/ML engineer or tech lead migrating production prompts between LLM providers. They have:

- **Goal:** Run a migration, get a GO/NO_GO answer, see the optimized prompt, understand the cost. Do this in one command.
- **Environment:** M-series Mac or Linux workstation. One of: cloud API keys (OpenAI/Anthropic), a local vLLM/Ollama endpoint, or both.
- **Data:** 20–400 JSONL prompt/response pairs from their production system.
- **Constraints:** Cost-sensitive (local endpoints preferred). Time-sensitive (wants to iterate, not wait 90 minutes for a single run). Working alone or on a 2–3 person team where "collaboration" means sharing a report link, not RBAC.
- **Doesn't care about:** Multi-user auth, team management, approval workflows, annotations queue, audit logs, Docker Compose profiles, Postgres. They'll use SQLite and the CLI.
- **Cares deeply about:** Is the GO/NO_GO answer trustworthy? Are the evaluation metrics meaningful? Does the optimized prompt actually work? Can I reproduce the results?

### Superuser

An ML platform team deploying RosettaStone as an internal migration service. Opt-in capabilities:

- Multi-user auth (JWT + RBAC) via `ROSETTASTONE_MULTI_USER=true`
- Postgres backend via `DATABASE_URL`
- Docker deployment
- Approval workflows, audit log
- A/B testing across migration versions
- Cost aggregation dashboards

These should be gated behind env vars, optional extras, or separate subcommands. The default user should never encounter them.

### Non-User

- Someone looking for a general-purpose prompt engineering IDE (this is a migration tool with a specific workflow)
- Someone wanting a production LLM router or proxy (shadow deployment exists but isn't wired)
- Someone expecting guaranteed results without human judgment (the tool gives a recommendation, the human decides)
- Teams that need certified compliance workflows (the tool has no formal audit trail beyond its own audit log table)

### Conflict Resolution

The README sells the superuser experience. The code is closest to serving the default user, but hasn't finished validating the core pipeline (empty benchmarks, unvalidated fence fix, hardcoded thresholds). The review will evaluate from the **default user's** perspective first, flagging superuser features as "gate" or "cut" candidates.

**Decision:** Prioritize the default user. An ML engineer who can't trust the GO/NO_GO answer won't care about approval workflows.

---

## Pass 1 — Bloat & Cleanup Inventory

### 1.1 Design Mockups — `assets/stitch/` (SAFE_DELETE)

| Field | Value |
|:---|:---|
| Path | `assets/stitch/` (26 files: 13 PNG screenshots + 13 HTML mockups) |
| Size | ~3.5 MB |
| Justification | Stitch MCP design artifacts from UI development. Not referenced by any source, test, or doc file. |
| Grep evidence | `grep -rn "assets/stitch" src/ tests/ docs/ pyproject.toml` → 0 results |
| Action | **SAFE_DELETE** |
| Risk | Low |

### 1.2 Parallel Server — `src/rosettastone/server_stitch/` (SAFE_DELETE)

| Field | Value |
|:---|:---|
| Path | `src/rosettastone/server_stitch/` (13 files, 2,325 lines) |
| Size | ~140 KB |
| Justification | Earlier/parallel server implementation. Only self-reference found (docstring in `app.py:9`). Zero imports from src/, tests/, or pyproject.toml. |
| Grep evidence | `grep -rn "server_stitch" src/ tests/ pyproject.toml` → 1 hit (self-reference only) |
| Action | **SAFE_DELETE** |
| Risk | Low — completely isolated, no tests, no imports |

### 1.3 One-Off Script — `scripts/fix_sql_quality.py` (SAFE_DELETE)

| Field | Value |
|:---|:---|
| Path | `scripts/fix_sql_quality.py` |
| Justification | One-time data cleanup script. Not referenced by any other file. |
| Grep evidence | `grep -rn "fix_sql_quality"` → 0 results outside the file |
| Action | **SAFE_DELETE** |
| Risk | Low |

### 1.4 Empty Package Init — `scripts/__init__.py` (SAFE_DELETE)

| Field | Value |
|:---|:---|
| Path | `scripts/__init__.py` |
| Justification | Empty `__init__.py` in a standalone scripts directory. Nothing imports `scripts` as a package. |
| Action | **SAFE_DELETE** |
| Risk | Low |

### 1.5 Stale Migration Outputs (already gitignored — no action)

Paths: `migration_output_gpt4o_to_qwen/`, `migration_output_gpt4o_to_qwen_v3/`, `migration_output_gpt4o_to_qwen_v7_fixed/`, `outputs/`. On disk but not tracked. `.gitignore` handles them.

### 1.6 Build Artifacts (already gitignored — no action)

Paths: `.venv/`, `.mypy_cache/`, `.pytest_cache/`, `.ruff_cache/`, `dist/`, `.worktrees/`. All gitignored, not tracked.

### 1.7 `docs/` Gitignore Policy (NEEDS_HUMAN_REVIEW)

| Field | Value |
|:---|:---|
| Path | `docs/` (10 markdown files on disk, gitignored at line 55) |
| Files | `model-compatibility-matrix.md`, `path_to_prod.md`, `production-roadmap.md`, `rosettastone-blog-source.md`, `rosettastone-build-document.md`, `rosettastone-folder-structure.md`, `rosettastone-spec-v2.md`, `test-plan-security-enterprise.md`, `test-plan.md`, `tier2-plan.md` |
| Issue | ROADMAP.md and HANDOFF.md reference `docs/production-roadmap.md` and `docs/tier2-plan.md` by path. A fresh clone won't have them. |
| Action | **NEEDS_HUMAN_REVIEW** — decide which docs to un-gitignore. At minimum, files referenced by tracked docs should be tracked. |
| Risk | Medium — broken references confuse onboarding |

### 1.8 Potentially Stale Docs (NEEDS_HUMAN_REVIEW)

| Field | Value |
|:---|:---|
| Paths | `docs/rosettastone-blog-source.md`, `docs/rosettastone-spec-v2.md`, `docs/rosettastone-folder-structure.md` |
| Justification | Blog draft, old spec version, and folder structure doc. Not referenced by code or tracked docs. |
| Action | **NEEDS_HUMAN_REVIEW** — keep or delete |
| Risk | Low |

### 1.9 Example Datasets — 6.2 MB (NEEDS_HUMAN_REVIEW)

| Field | Value |
|:---|:---|
| Path | `examples/datasets/` (5 datasets, 10 JSONL files + metadata) |
| Size | 6.2 MB |
| Justification | Valuable for onboarding and README examples. But 6.2 MB in git bloats every clone. |
| Action | **NEEDS_HUMAN_REVIEW** — consider Git LFS, or accept size given onboarding value |
| Risk | Low |

### 1.10 Misc OK items

- `__pycache__`, `.env` — not tracked, gitignore handles correctly
- `data/meridian_knowledge_base/` — 100 KB, referenced by README and scripts, no action
- Alembic migrations — 10 versions, standard. Squash before PyPI release.

### 1.11 Dependency Audit

All optional dependencies have at least one lazy import. **No unused dependencies found.**

| Extra | Package | Import Location |
|:---|:---|:---|
| eval | bert-score, sentence-transformers | `evaluate/` |
| redis | redis | `ingest/redis_adapter.py`, `server/app.py` |
| rq | rq | `server/task_dispatch.py` |
| braintrust | braintrust | `ingest/braintrust_adapter.py` |
| langsmith | langsmith | `ingest/langsmith_adapter.py` |
| safety | presidio-analyzer/anonymizer | `safety/presidio_engine.py` |
| clustering | scikit-learn | `cluster/embedder.py` |
| postgres | psycopg | `server/database.py` |
| metrics | prometheus-client | `server/metrics.py` |
| sentry | sentry-sdk | `server/app.py` |
| web | fastapi, uvicorn, sqlmodel, httpx | `server/` |
| auth | passlib, bcrypt, pyjwt | `server/auth_utils.py` |
| calibration | scikit-learn, krippendorff | `calibration/calibrator.py` |

### Pass 1 Totals

| Category | Count | Est. Savings |
|:---|:---:|:---|
| SAFE_DELETE | 4 items | ~2,350 lines, ~3.6 MB, 40 tracked files |
| NEEDS_HUMAN_REVIEW | 3 items | Policy decisions |
| No action | 5 items | Already handled correctly |
| Monitor | 1 item | Future cleanup (Alembic squash) |
