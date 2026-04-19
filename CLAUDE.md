# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RosettaStone is an automated LLM model migration tool. It ingests production prompt/response pairs (JSONL), optimizes prompts for a target model via DSPy/GEPA, validates behavioral equivalence with multi-strategy evaluation, and generates migration reports.

## Commands

```bash
# Install (development)
uv sync --dev --all-extras

# Run tests
uv run pytest tests/ -v
uv run pytest tests/test_ingest/test_jsonl.py -v  # single test file
uv run pytest tests/ -k "test_name" -v             # single test by name

# Build Tailwind CSS (run after changing templates or utility classes)
make css

# Watch mode (auto-rebuild on template changes)
make css-watch

# Lint & format
uv run ruff check src/ tests/
uv run ruff format src/ tests/

# Type check
uv run mypy src/rosettastone/

# CLI usage
uv run rosettastone migrate --data examples/sample_data.jsonl --from openai/gpt-4o --to anthropic/claude-sonnet-4
uv run rosettastone preflight --data examples/sample_data.jsonl --from openai/gpt-4o --to anthropic/claude-sonnet-4
uv run rosettastone migrate --data examples/sample_data.jsonl --from openai/gpt-4o --to anthropic/claude-sonnet-4 --dry-run
```

## Architecture

**Pipeline:** `preflight → ingest → baseline eval → GEPA optimize → validation eval → report`

- **`src/rosettastone/core/migrator.py`** — Thin orchestrator. Accepts `MigrationConfig`, delegates to pipeline steps, returns `MigrationResult`.
- **`src/rosettastone/core/pipeline.py`** — Pipeline step definitions wiring subsystems together.
- **`src/rosettastone/config.py`** — `MigrationConfig` Pydantic model. CLI and library both construct this.
- **`src/rosettastone/core/types.py`** — Shared types: `PromptPair`, `EvalResult`, `MigrationResult`, `OutputType`.

**Subsystems (each has `base.py` abstract class for extensibility):**
- **`preflight/`** — Model capability detection (LiteLLM), token budget checks, cost estimation. Returns warnings/blockers.
- **`ingest/`** — `DataAdapter` pattern. JSONL adapter for MVP. `splitter.py` handles dedup + train/val/test splits.
- **`optimize/`** — DSPy/GEPA wrapper. `metric.py` returns `dspy.Prediction(score, feedback)` for GEPA's reflective optimization. `dspy_program.py` defines the DSPy `Module`.
- **`evaluate/`** — Strategy pattern. BERTScore, embedding sim, exact match, JSON validator. `CompositeEvaluator` auto-selects by output type.
- **`report/`** — Jinja2-templated markdown reports.

**`claude_agents/`** — Claude subagent system/persona prompts with persistent memory for each subagent.

## Key Conventions

- **Never log prompt content** at any level — production data may contain PII. Use structural logging only (token counts, scores, timing).
- **Optional deps** (bert-score, sentence-transformers, redis) are imported lazily with `try/except ImportError` fallbacks.
- **DSPy GEPA metric** must return `dspy.Prediction(score=..., feedback=...)` — the feedback string drives GEPA's reflective optimization.
- **CLI and library share the same code path** — CLI constructs `MigrationConfig` from args, calls `Migrator.run()`.
- **Pydantic v2 required** — DSPy uses it internally. No v1 compat mode.
- **Rate limiting** handled by LiteLLM/DSPy retry logic — don't implement custom retry/backoff.
- **`--from` CLI flag** uses Typer Option alias pattern since `from` is a Python keyword.
- **ruff** for linting + formatting, line length 100, target Python 3.11.

## Authentication

Optional API key auth controlled by `ROSETTASTONE_API_KEY` env var. When unset, auth is disabled.

**Test credentials:** Set `ROSETTASTONE_API_KEY=test` to enable auth with API key `test`.

```bash
# Run server with auth enabled (test key)
ROSETTASTONE_API_KEY=test uv run uvicorn rosettastone.server.app:create_app --factory

# API usage with auth
curl -H "Authorization: Bearer test" http://localhost:8000/api/v1/health

# UI: navigate to /ui/login and enter "test" as the API key
```

## Project Status

Phase 1 (MVP) — scaffolded, implementing core pipeline. See `docs/rosettastone-build-document.md` for full spec and phased roadmap.
