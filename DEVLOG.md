# Dev Log

Progress notes, design decisions, and things learned along the way.

---

## 2026-03-19 — Project Scaffolding

**What happened:** Scaffolded the full Phase 1 project structure. All core modules are in place — migrator orchestrator, GEPA optimization wrapper, multi-strategy evaluation, CLI, pre-flight checks, JSONL ingestion, and markdown report generation. Dependencies installed, CI configured, repo is live.

**Key decisions made:**
- Went with `uv` over pip for dependency management — faster, better lockfile support
- Used Hatchling as the build backend to keep `pyproject.toml` simple
- GEPA defaults to `auto="light"` (~560 metric calls) for the MVP. Users can escalate to `medium` or `heavy` if they need better results and are willing to spend more.
- Optional deps (bert-score, sentence-transformers, redis) are lazy-imported with fallback chains. If you don't install `[eval]`, it falls back to basic string similarity.
- The reflection model (used by GEPA internally) defaults to GPT-4o regardless of the migration target. Using the target model as its own reflection model would defeat the purpose.

**What's next:** Write unit tests for the ingest and evaluation modules, then wire up a real end-to-end migration run with the sample data.
