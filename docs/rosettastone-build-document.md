# RosettaStone — Build Document

**One-line summary:** An automated LLM migration tool that uses production prompt/response pairs as training data to optimize prompts for new models via DSPy/GEPA, validated by a multi-strategy evaluation framework.

**Problem Statement:** When companies upgrade LLM versions or switch providers, their prompts break — different models interpret prompts differently, producing different formatting, reasoning, and output structures. Migration currently requires manual prompt re-engineering consuming 20-50% of original development time. No existing tool automates the full loop: ingest production data → optimize prompts → validate equivalence → report results.

**Vision:** A one-command migration: point RosettaStone at your production cache and your target model, get back an optimized prompt with a confidence score and migration report. AWS showed the approach works. We made it a one-liner.

---

# Product Requirements (PRD)

## Goals & Success Metrics

**Primary goal:** Reduce LLM model migration time from weeks to hours by automating prompt optimization using production data.

**Success metrics:**
- A migration run on 100 prompt/response pairs completes in under 60 minutes
- Optimized prompts achieve ≥85% pairwise win rate against source model behavior on held-out test set
- API cost per migration run stays under $20 for standard models
- Users can go from `pip install` to first migration result in under 15 minutes (with sample data)
- Pre-flight checks catch ≥90% of impossible migrations before burning API credits

## Primary User

**AI/ML engineers** responsible for production LLM systems. Technically fluent — familiar with APIs, prompt engineering, caching infrastructure, and model evaluation. They're the ones who get the ticket "we need to migrate from GPT-4o to Claude by end of quarter" and currently spend weeks on it.

**Secondary user:** Engineering leadership who needs a go/no-go decision on model migrations. They care about cost savings, quality consistency, and risk — not prompt text.

## Core Features (MVP — Phase 1)

1. **JSONL data ingestion** — Load prompt/response pairs from JSONL files with Pydantic schema validation, deduplication, output type detection (JSON/classification/free-text), and train/validation/test splitting
2. **Pre-flight checks** — Capability detection (tool calling, JSON mode, vision, context window), token budget calculation with tokenizer inflation warnings (15-20% SentencePiece overhead), cost estimation, and `--dry-run` mode
3. **GEPA prompt optimization** — DSPy integration using GEPA optimizer with behavioral equivalence as the objective function. Textual feedback in metric function enables targeted improvements
4. **Multi-strategy evaluation** — BERTScore + embedding cosine similarity for free text, exact match for classifications, JSON schema validation + field-level comparison for structured outputs. Composite confidence score reported as pairwise win rate
5. **Markdown migration report** — Before/after scores, per-category breakdown, sample comparisons, worst regressions, confidence score, cost summary
6. **CLI interface** — `rosettastone migrate`, `rosettastone preflight`, `rosettastone evaluate` commands via Typer
7. **Python library** — `Migrator` class with `MigrationConfig` for programmatic usage

## Explicitly Out of Scope (MVP)

- Redis integration (Phase 2)
- Web UI (Phase 3)
- LLM-as-judge evaluation (Phase 2)
- PII scanning/auditing (Phase 2)
- Observability platform adapters — LangSmith, Braintrust, OpenTelemetry (Phase 4)
- Multi-step pipeline migration (Phase 5)
- Behavioral cloning + improvement mode (Phase 4)
- Multi-turn conversation prompt support
- Fine-tuning or weight optimization — prompt optimization only
- Model hosting or serving

## Open Questions

- Exact GEPA `auto` mode configuration for best default experience (light vs medium vs heavy)
- Whether to ship `examples/sample_data.jsonl` with synthetic data or real anonymized examples
- Whether the Jinja2 report template or a pure Python string builder is simpler for Phase 1

---

# System Architecture

## High-Level Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     USER INTERFACE LAYER                     │
│  ┌──────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │   CLI    │  │ Python Lib   │  │   Web UI (Phase 3+)    │ │
│  │  (Typer) │  │  (Migrator)  │  │  (FastAPI + React)     │ │
│  └────┬─────┘  └──────┬───────┘  └───────────┬────────────┘ │
│       └───────────────┬┘                      │              │
└───────────────────────┼───────────────────────┘              │
                        ▼                                      │
┌─────────────────────────────────────────────────────────────┐│
│                   CORE ORCHESTRATION                         ││
│  ┌──────────────────────────────────────────────────────┐   ││
│  │                 Migrator (core/migrator.py)           │   ││
│  │  Executes pipeline: preflight → ingest → baseline    │   ││
│  │  → optimize → validate → export                      │   ││
│  └──────────────────────────────────────────────────────┘   ││
└─────────────────────────────────────────────────────────────┘│
                        │                                      │
        ┌───────────────┼──────────────┐                       │
        ▼               ▼              ▼                       │
┌──────────────┐ ┌────────────┐ ┌─────────────┐               │
│  PREFLIGHT   │ │   INGEST   │ │  OPTIMIZE   │               │
│              │ │            │ │             │               │
│ Capabilities │ │ JSONL      │ │ GEPA        │               │
│ Token Budget │ │ Redis (P2) │ │ MIPROv2 (P2)│               │
│ Cost Est.    │ │ LangSmith  │ │ DSPy Program│               │
│              │ │ (P4)       │ │ Metric Fn   │               │
└──────────────┘ └────────────┘ └──────┬──────┘               │
                                       │                       │
                        ┌──────────────┘                       │
                        ▼                                      │
              ┌──────────────────┐    ┌───────────────┐        │
              │    EVALUATE      │    │    REPORT     │        │
              │                  │    │               │        │
              │ BERTScore        │───▶│ Markdown      │        │
              │ Embedding Sim    │    │ PDF (P3)      │        │
              │ Exact Match      │    │ HTML (P3)     │        │
              │ JSON Validator   │    │ Narrative (P3)│        │
              │ LLM Judge (P2)  │    └───────────────┘        │
              │ Composite Score  │                             │
              └──────────────────┘                             │
                                                               │
┌─────────────────────────────────────────────────────────────┐│
│                   EXTERNAL SERVICES                          ││
│  ┌─────────┐  ┌─────────┐  ┌──────────┐  ┌──────────────┐  ││
│  │ LiteLLM │  │  DSPy   │  │  Redis   │  │ BERTScore    │  ││
│  │ (100+   │  │  (GEPA/ │  │  (cache  │  │ (local       │  ││
│  │ models) │  │  MIPROv2│  │  source) │  │  inference)  │  ││
│  └─────────┘  └─────────┘  └──────────┘  └──────────────┘  ││
└─────────────────────────────────────────────────────────────┘│
```

## Key Components and Their Responsibilities

**Migrator (`core/migrator.py`):** Thin orchestrator that executes pipeline steps in order. No business logic — delegates to each subsystem. Accepts `MigrationConfig`, returns `MigrationResult`.

**Pre-flight (`preflight/`):** Guards against wasted API spend. Checks model capabilities via LiteLLM metadata, calculates token budgets accounting for tokenizer differences, estimates total cost. Returns `PreflightReport` with warnings/blockers.

**Ingest (`ingest/`):** Adapter pattern. `DataAdapter` abstract base class with `load() -> list[PromptPair]`. JSONL adapter for MVP, Redis adapter (Phase 2) auto-detects cache format by key prefix. `Splitter` handles deduplication, output type detection, and train/val/test splits.

**Optimize (`optimize/`):** Wrapper around DSPy optimizers. `GEPAOptimizer` configures `dspy.GEPA`, builds a DSPy `Module` with the migration signature, defines the metric function (behavioral similarity + textual feedback), and runs optimization. Returns optimized prompt text.

**Evaluate (`evaluate/`):** Strategy pattern. Each metric (BERTScore, embedding sim, exact match, JSON validator) implements `Evaluator` base class. `CompositeEvaluator` auto-selects metrics by output type, combines scores with configurable weights, computes pairwise win rate.

**Report (`report/`):** Takes `MigrationResult` and produces formatted output. Markdown for MVP, PDF/HTML/narrative for Phase 3. Jinja2 templates for structured formatting.

## Data Flow

```
User data (JSONL/Redis/LangSmith)
  │
  ▼
[Ingest] → list[PromptPair]  ← Pydantic-validated, deduplicated
  │
  ├──→ train_set (for optimization)
  ├──→ val_set (for GEPA's internal validation)
  └──→ test_set (for final validation — never seen during optimization)
         │
         ├──→ [Baseline] run test_set through NEW model, unoptimized
         │       │
         │       ▼
         │    baseline_results: list[EvalResult]  ← the "before" snapshot
         │
  train + val → [Optimize via GEPA]
                  │
                  ▼
               optimized_prompt: str
                  │
                  ▼
         [Validate] run test_set through NEW model WITH optimized prompt
                  │
                  ▼
               validation_results: list[EvalResult]  ← the "after" snapshot
                  │
                  ▼
         [Report] compare baseline vs validation → MigrationReport
                  │
                  ▼
               Output: optimized_prompt.txt + report.md + config.yaml
```

## External Integrations / APIs / Services

| Service | Purpose | When Called | Data Sent |
|---|---|---|---|
| **Target LLM (via LiteLLM)** | Generate responses on new model | Baseline eval, GEPA optimization, final validation | Prompt content from training data |
| **Reflection LLM (GPT-4o recommended)** | GEPA's internal reflection model | During optimization loop | Execution traces, prompt candidates, metric feedback |
| **BERTScore model (local)** | Compute semantic similarity | Evaluation steps | Response text (stays local, no external API) |
| **Redis** (Phase 2) | Read cached prompt/response pairs | Data ingestion | Connection only — reads data, sends nothing |
| **LangSmith API** (Phase 4) | Pull production traces | Data ingestion | Authentication token — reads data |

## Tech Stack

| Layer | Technology | Version/Notes |
|---|---|---|
| Language | Python | 3.11 or 3.12 (DSPy requires ≥3.10, <3.14) |
| Optimization | DSPy + GEPA | Pin to tested version; GEPA via `gepa` package |
| LLM Access | LiteLLM | Provider-agnostic (100+ models) |
| CLI | Typer | Modern CLI framework with auto-generated help |
| Config | Pydantic + YAML | Pydantic models, YAML config files via PyYAML |
| Evaluation | bert-score, sentence-transformers | Optional `[eval]` extra; distilbert default |
| Redis | redis-py + hiredis | Optional `[redis]` extra |
| Templates | Jinja2 | Migration report templates |
| Testing | pytest + pytest-asyncio | Unit + integration tests |
| Type Checking | mypy or pyright | PEP 561 typed package |
| Linting | ruff | Fast Python linter + formatter |
| Web Backend (P3) | FastAPI + SQLModel + SQLite | Lightweight, no external DB required |
| Web Frontend (P3) | React + TypeScript + Vite + Tailwind | TanStack Table, react-diff-viewer, Recharts |

---

# Build Roadmap

## Phase 1: MVP — Single Prompt Migration (CLI + Library)
**What ships:** A working tool that takes JSONL prompt/response pairs and a target model, runs GEPA optimization, and outputs an optimized prompt with a quality report.

**Why this first:** Proves the core optimization loop works. Everything else is incremental on top of a working migration pipeline.

**Deliverables:**
- `pip install rosettastone` with `[eval]` optional extra
- `rosettastone migrate --data pairs.jsonl --from gpt-4o --to claude-sonnet-4`
- `from rosettastone import Migrator, MigrationConfig`
- Pre-flight checks with `--dry-run`
- Markdown migration report with confidence score
- `examples/sample_data.jsonl` with 50 demo pairs
- README, LICENSE (MIT), CI/CD via GitHub Actions
- Unit tests for all core modules

**Exit criteria:** A user can run a migration on sample data, get a confidence score > 80%, and the optimized prompt demonstrably outperforms the unoptimized prompt on the test set.

## Phase 2: Evaluation Depth + Redis + Safety
**What ships:** Production-grade evaluation, automated Redis ingestion, PII guardrails.

**Why this second:** Phase 1's eval is good enough to prove the concept but not good enough for production decisions. Redis integration is the first "real" data source. PII handling is a trust requirement.

**Deliverables:**
- LLM-as-judge evaluator (pairwise comparison mode)
- Field-level JSON structural diff evaluator
- Output type auto-detection
- Redis adapter with auto-format detection (RedisVL, LangChain, LiteLLM, GPTCache)
- MIPROv2 fallback optimizer
- Known-issue feedback mechanism (2× weighted constraints)
- PII detection warnings during ingestion
- Post-optimization prompt audit for memorized training data
- `--local-only` mode (BERTScore only, local model endpoint)
- Colored CLI output with progress bars and summary tables

**Exit criteria:** A user can point the tool at a production Redis cache and get a trustworthy go/no-go migration decision.

## Phase 3: Web UI + Migration Reports
**What ships:** Visual migration dashboard and auto-generated reports.

**Why this third:** This is where the portfolio impact lives. Side-by-side diffs and traffic-light dashboards are what gets screenshot and shared. Also critical for the "show leadership" use case.

**Deliverables:**
- FastAPI backend serving migration results
- React dashboard with traffic-light summary cards
- Side-by-side diff view with word-level highlighting
- Filterable evaluation grid (TanStack Table)
- Score distribution charts (Recharts)
- Individual test case drill-down
- Engineer/Executive persona toggle
- Auto-generated PDF and interactive HTML reports
- AI-generated natural language summary
- `rosettastone serve` CLI command

**Exit criteria:** An engineer can run migration, open the web UI, identify and debug regressions, export a PDF report, and share it with leadership.

## Phase 4: Observability Integrations + Advanced Features
**What ships:** Direct integrations with LangSmith, Braintrust, OpenTelemetry. Migration-and-improve workflows. CI/CD integration.

## Phase 5: Pipeline Migration + Enterprise
**What ships:** Multi-step pipeline migration, A/B testing, versioning, team collaboration, Docker self-hosted deployment.

---

# Claude Code Spec

This section contains everything needed to begin building Phase 1 immediately with no additional context.

## Project Scaffolding

```bash
# Create project directory
mkdir rosettastone && cd rosettastone

# Initialize with uv (recommended) or pip
uv init --lib --name rosettastone
# OR: mkdir -p src/rosettastone && touch src/rosettastone/__init__.py

# Create directory structure
mkdir -p src/rosettastone/{cli,core,preflight,ingest,optimize,evaluate,report/templates,utils}
mkdir -p tests/{test_preflight,test_ingest,test_optimize,test_evaluate,test_cli}
mkdir -p examples docs .github/workflows

# Create __init__.py files
find src/rosettastone -type d -exec touch {}/__init__.py \;

# Create marker files
touch src/rosettastone/py.typed
touch LICENSE  # MIT

# Install core dependencies
uv add dspy litellm pydantic pyyaml typer jinja2 rich
# OR: pip install dspy litellm pydantic pyyaml typer jinja2 rich

# Install optional eval dependencies
uv add --optional eval bert-score sentence-transformers
# OR: pip install bert-score sentence-transformers

# Install dev dependencies
uv add --dev pytest pytest-asyncio mypy ruff
```

## pyproject.toml

```toml
[project]
name = "rosettastone"
version = "0.1.0"
description = "Automated LLM model migration using production data and GEPA optimization"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.11,<3.14"
dependencies = [
    "dspy>=2.6",
    "litellm>=1.50",
    "pydantic>=2.0",
    "pyyaml>=6.0",
    "typer>=0.12",
    "jinja2>=3.1",
    "rich>=13.0",
    "gepa>=0.1",
]

[project.optional-dependencies]
eval = [
    "bert-score>=0.3.13",
    "sentence-transformers>=3.0",
]
redis = [
    "redis[hiredis]>=5.0",
    "redisvl>=0.3",
]
all = [
    "rosettastone[eval,redis]",
]

[project.scripts]
rosettastone = "rosettastone.cli.main:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/rosettastone"]

[tool.ruff]
target-version = "py311"
line-length = 100

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"

[tool.mypy]
python_version = "3.11"
strict = true
```

## Key Implementation Notes Per Component

### `src/rosettastone/__init__.py`
```python
from rosettastone.core.migrator import Migrator
from rosettastone.config import MigrationConfig
from rosettastone.core.types import MigrationResult

__all__ = ["Migrator", "MigrationConfig", "MigrationResult"]
```

### `src/rosettastone/config.py`
Pydantic model for all migration configuration. CLI parses args into this; library users construct it directly.

```python
from pydantic import BaseModel, Field
from enum import Enum
from pathlib import Path
from typing import Optional

class EvalStrategy(str, Enum):
    AUTO = "auto"          # Auto-detect by output type
    BERTSCORE = "bertscore"
    EMBEDDING = "embedding"
    EXACT = "exact"
    JSON = "json"

class OptimizerChoice(str, Enum):
    GEPA = "gepa"
    MIPRO = "mipro"        # Phase 2

class MigrationConfig(BaseModel):
    # Required
    source_model: str              # e.g. "openai/gpt-4o"
    target_model: str              # e.g. "anthropic/claude-sonnet-4"
    data_path: Path                # Path to JSONL file

    # Optional with smart defaults
    output_dir: Path = Path("./migration_output")
    eval_strategy: EvalStrategy = EvalStrategy.AUTO
    optimizer: OptimizerChoice = OptimizerChoice.GEPA
    train_split: float = Field(default=0.2, ge=0.1, le=0.5)
    val_split: float = Field(default=0.8, ge=0.5, le=0.9)
    min_pairs: int = 20
    recommended_pairs: int = 100

    # GEPA configuration
    gepa_auto: str = "light"       # "light", "medium", "heavy"
    reflection_model: str = "openai/gpt-4o"
    num_threads: int = 4

    # Pre-flight
    dry_run: bool = False
    skip_preflight: bool = False
    max_context_usage: float = 0.75  # Warn at 75% context window usage
```

### `src/rosettastone/core/types.py`
Shared types used across all modules.

```python
from pydantic import BaseModel
from enum import Enum
from typing import Optional, Any

class OutputType(str, Enum):
    JSON = "json"
    CLASSIFICATION = "classification"
    SHORT_TEXT = "short_text"
    LONG_TEXT = "long_text"

class PromptPair(BaseModel):
    prompt: str | list[dict]       # Plain text or OpenAI messages array
    response: str
    source_model: str
    metadata: dict[str, Any] = {}
    feedback: Optional[str] = None  # Known issues with this response
    output_type: Optional[OutputType] = None

class EvalResult(BaseModel):
    prompt_pair: PromptPair
    new_response: str
    scores: dict[str, float]       # e.g. {"bertscore_f1": 0.87, "embedding_sim": 0.91}
    composite_score: float
    is_win: bool                   # New model matches or exceeds old
    details: dict[str, Any] = {}   # Metric-specific details

class MigrationResult(BaseModel):
    config: dict                   # Serialized MigrationConfig
    optimized_prompt: str
    baseline_results: list[EvalResult]
    validation_results: list[EvalResult]
    confidence_score: float        # Pairwise win rate
    baseline_score: float          # Pre-optimization win rate
    improvement: float             # Delta
    cost_usd: float
    duration_seconds: float
    warnings: list[str]
```

### `src/rosettastone/core/migrator.py`
Thin orchestrator — the main entry point.

```python
class Migrator:
    def __init__(self, config: MigrationConfig):
        self.config = config

    def run(self) -> MigrationResult:
        # Step 0: Pre-flight
        if not self.config.skip_preflight:
            preflight_report = run_preflight_checks(self.config)
            if preflight_report.has_blockers:
                raise MigrationBlockedError(preflight_report)
            if self.config.dry_run:
                return preflight_report.as_dry_run_result()

        # Step 1: Ingest
        pairs = load_data(self.config)
        train, val, test = split_data(pairs, self.config)

        # Step 2: Baseline
        baseline = evaluate_baseline(test, self.config)

        # Step 3: Optimize
        optimized_prompt = optimize_prompt(train, val, self.config)

        # Step 4: Validate
        validation = evaluate_optimized(test, optimized_prompt, self.config)

        # Step 5: Report
        result = build_result(self.config, optimized_prompt, baseline, validation)
        generate_report(result, self.config.output_dir)

        return result
```

### `src/rosettastone/optimize/gepa.py`
The GEPA wrapper — the most critical implementation file.

Key implementation notes:
- Configure `dspy.LM()` for both the target model and the reflection model
- Build a DSPy `Module` with signature `"prompt -> response"` using `dspy.ChainOfThought`
- The metric function MUST return `dspy.Prediction(score=..., feedback=...)` for GEPA — the `feedback` string is what drives GEPA's reflective optimization
- Use `dspy.GEPA(metric, auto="light")` for default configuration
- The optimization result is a compiled DSPy program — extract the optimized instructions from it
- DSPy caches all LM calls automatically; no need for custom caching

```python
import dspy
from rosettastone.optimize.metric import build_migration_metric

class GEPAOptimizer:
    def optimize(self, train_set, val_set, config):
        # Configure LMs
        target_lm = dspy.LM(config.target_model)
        reflection_lm = dspy.LM(config.reflection_model, temperature=1.0, max_tokens=16000)

        # Build DSPy program
        program = MigrationProgram()  # from dspy_program.py
        program.set_lm(target_lm)

        # Build metric
        metric = build_migration_metric(config)

        # Convert to DSPy Examples
        trainset = [
            dspy.Example(prompt=p.prompt, expected_response=p.response).with_inputs('prompt')
            for p in train_set
        ]

        # Run GEPA
        optimizer = dspy.GEPA(
            metric=metric,
            auto=config.gepa_auto,  # "light", "medium", "heavy"
            reflection_lm=reflection_lm,
            num_threads=config.num_threads,
        )
        compiled = optimizer.compile(program, trainset=trainset)

        # Extract optimized prompt
        return extract_optimized_instructions(compiled)
```

### `src/rosettastone/optimize/metric.py`
The metric function is where behavioral equivalence is defined.

Key implementation notes:
- Must return `dspy.Prediction(score=..., feedback=...)` for GEPA
- The `feedback` string should be diagnostic — explain WHY the score is what it is
- Import BERTScore lazily (optional dep)
- For classification/JSON outputs, use deterministic checks first
- Score range should be 0.0-1.0

```python
def build_migration_metric(config):
    def migration_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        expected = gold.expected_response
        actual = pred.response
        feedback_parts = []

        # Semantic similarity (always available — fallback to basic string similarity)
        try:
            from rosettastone.evaluate.bertscore import compute_bertscore
            sem_score = compute_bertscore(expected, actual)
        except ImportError:
            from rosettastone.evaluate.embedding import compute_embedding_sim
            sem_score = compute_embedding_sim(expected, actual)

        score = sem_score

        if sem_score < 0.7:
            feedback_parts.append(
                f"Response diverges significantly from expected (similarity: {sem_score:.2f}). "
                f"Expected style/content: '{expected[:200]}...'"
            )
        elif sem_score < 0.85:
            feedback_parts.append(
                f"Response partially matches (similarity: {sem_score:.2f}). "
                f"Check formatting and completeness."
            )
        else:
            feedback_parts.append(f"Good match (similarity: {sem_score:.2f})")

        feedback = "\n".join(feedback_parts) if feedback_parts else ""
        return dspy.Prediction(score=min(score, 1.0), feedback=feedback)

    return migration_metric
```

### `src/rosettastone/evaluate/composite.py`
Combines multiple metrics and computes the final confidence score.

Key implementation notes:
- Auto-detect output type if not specified
- Select metrics based on output type
- Compute pairwise win rate: `(wins + 0.5 * ties) / total`
- A "win" is when composite score ≥ threshold (default 0.8)

### `src/rosettastone/preflight/capabilities.py`
Model capability detection.

Key implementation notes:
- Use `litellm.get_model_info(model)` to get context window size, supported features
- Check for `supports_function_calling`, `supports_vision`, `supports_response_schema`
- If source model uses features the target doesn't support, issue WARNING (not blocker — GEPA might work around it)
- If prompts exceed 75% of target context window, issue WARNING
- If prompts exceed 100% of target context window, issue BLOCKER

### `src/rosettastone/preflight/token_budget.py`
Token counting with cross-tokenizer awareness.

Key implementation notes:
- Use `litellm.token_counter(model, text)` for per-model token counts
- When source and target use different tokenizer families, apply inflation factor:
  - tiktoken → SentencePiece: multiply by 1.2 (20% inflation)
  - Any → CJK heavy content: warn about per-character tokenization
- Report: "Prompt X uses N tokens on source model, estimated M tokens on target (Y% of context window)"

### `src/rosettastone/cli/main.py`
Typer CLI application.

```python
import typer
from rich.console import Console
from rich.progress import Progress

app = typer.Typer(name="rosettastone", help="Automated LLM model migration")
console = Console()

@app.command()
def migrate(
    data: Path = typer.Option(..., "--data", "-d", help="Path to JSONL file"),
    source: str = typer.Option(..., "--from", help="Source model (e.g. openai/gpt-4o)"),
    target: str = typer.Option(..., "--to", help="Target model (e.g. anthropic/claude-sonnet-4)"),
    output: Path = typer.Option("./migration_output", "--output", "-o"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Estimate cost without running"),
    optimizer: str = typer.Option("gepa", "--optimizer"),
    auto_mode: str = typer.Option("light", "--auto", help="GEPA intensity: light/medium/heavy"),
):
    config = MigrationConfig(
        source_model=source, target_model=target, data_path=data,
        output_dir=output, dry_run=dry_run, gepa_auto=auto_mode,
    )
    migrator = Migrator(config)
    result = migrator.run()
    display_result(console, result)

@app.command()
def preflight(
    data: Path = typer.Option(..., "--data", "-d"),
    source: str = typer.Option(..., "--from"),
    target: str = typer.Option(..., "--to"),
):
    """Run pre-flight checks only."""
    ...
```

### `src/rosettastone/ingest/schema.py`
Pydantic models for the universal JSONL schema.

```python
from pydantic import BaseModel, model_validator
from typing import Optional, Any

class PromptPairInput(BaseModel):
    """Schema for a single line in the JSONL input file."""
    prompt: str | list[dict]       # Plain text or OpenAI messages format
    response: str | dict           # Plain text or message object
    source_model: str

    # Optional enrichment
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    timestamp: Optional[str] = None
    metadata: dict[str, Any] = {}
    feedback: Optional[str] = None

    @model_validator(mode="after")
    def normalize_response(self):
        """Ensure response is always a string."""
        if isinstance(self.response, dict):
            self.response = self.response.get("content", str(self.response))
        return self
```

## Environment Variables

```bash
# Required — at least one LLM provider key
OPENAI_API_KEY=sk-...           # For OpenAI models + default GEPA reflection
ANTHROPIC_API_KEY=sk-ant-...    # For Anthropic models
GOOGLE_API_KEY=...              # For Gemini models

# Optional — override defaults
ROSETTASTONE_REFLECTION_MODEL=openai/gpt-4o  # Model for GEPA reflection
ROSETTASTONE_LOG_LEVEL=WARN                   # Never logs prompt content
ROSETTASTONE_CACHE_DIR=~/.rosettastone/cache  # DSPy cache location
```

## Known Gotchas and Non-Obvious Decisions

1. **DSPy pins LiteLLM versions aggressively.** Check `dspy`'s `pyproject.toml` for the pinned LiteLLM range and ensure compatibility. Known conflict issues exist on GitHub (#7806, #6644). Pin DSPy to a specific tested version in our `pyproject.toml`.

2. **BERTScore downloads a model on first run.** The default `distilbert-base-uncased` is ~250MB. The first call to `compute_bertscore()` will block while downloading. Add a progress indicator or run download at install time.

3. **DSPy caches LM calls to `~/.dspy_cache/` by default.** This is great for re-runs but can surprise users with stale results. Document this and expose cache clearing.

4. **GEPA's `auto` parameter determines optimization intensity:**
   - `"light"`: ~560 metric calls, ~20 min, cheapest
   - `"medium"`: ~2000 metric calls, ~1 hr, moderate
   - `"heavy"`: ~5000+ metric calls, ~2+ hrs, most thorough
   - Default to `"light"` for MVP — users can escalate.

5. **The reflection LM should be a strong reasoning model.** Always use GPT-4o or equivalent regardless of what the target model is. Don't use the target model as its own reflection model — that defeats the purpose.

6. **BERTScore is CPU-only by default.** PyTorch GPU is not required. The `distilbert-base-uncased` model runs fine on CPU. Don't add CUDA as a dependency.

7. **GEPA returns a compiled DSPy program, not a string.** You need to extract the optimized instructions from the compiled program's `__dict__` or by inspecting the `Predict` module's `signature.instructions`. This extraction step is non-obvious.

8. **Pydantic v2 is required.** DSPy uses Pydantic v2 internally. Do not use Pydantic v1 compatibility mode.

9. **The `--from` CLI flag conflicts with Python's `from` keyword.** Typer handles this via the Option alias pattern: `source: str = typer.Option(..., "--from")` — Typer maps `--from` to the `source` Python variable.

10. **Rate limiting is handled by LiteLLM and DSPy's built-in retry logic.** Don't implement custom retry/backoff — it will conflict. Set `num_threads` conservatively (4 default) to avoid hitting provider rate limits.

11. **Never log prompt content at any log level.** Production prompt data may contain PII. The logging module should redact any string that looks like prompt content. Use structural logging (token counts, scores, timing) only.

## Commands to Run the Project

```bash
# Install from source (development)
git clone https://github.com/YOUR_USERNAME/rosettastone.git
cd rosettastone
pip install -e ".[eval,dev]"

# Run a migration
rosettastone migrate \
  --data examples/sample_data.jsonl \
  --from openai/gpt-4o \
  --to anthropic/claude-sonnet-4 \
  --output ./my_migration

# Dry run (estimate cost only)
rosettastone migrate \
  --data examples/sample_data.jsonl \
  --from openai/gpt-4o \
  --to anthropic/claude-sonnet-4 \
  --dry-run

# Pre-flight checks only
rosettastone preflight \
  --data examples/sample_data.jsonl \
  --from openai/gpt-4o \
  --to anthropic/claude-sonnet-4

# Run tests
pytest tests/ -v

# Type checking
mypy src/rosettastone/

# Lint
ruff check src/ tests/
ruff format src/ tests/
```

---

# GitHub README

```markdown
# 🪨 RosettaStone

**Automated LLM model migration. Your production data is the training signal.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/rosettastone.svg)](https://pypi.org/project/rosettastone/)

---

Switching LLM providers shouldn't mean weeks of prompt re-engineering.
RosettaStone takes your existing prompt/response pairs — from Redis caches,
JSONL files, or observability platforms — and automatically optimizes your
prompts for any new model. Powered by
[GEPA](https://arxiv.org/abs/2507.19457) (ICLR 2026 Oral) and
[DSPy](https://dspy.ai), it achieves behavioral equivalence in minutes
instead of weeks.

**The core idea:** your production cache data *is* the behavioral spec.
The old model's actual outputs are the ground truth for what the new model
should reproduce. RosettaStone turns that data into automated migration.

## Features

- **One-command migration** — `rosettastone migrate --from gpt-4o --to claude-sonnet-4`
- **Provider-agnostic** — supports 100+ models via LiteLLM (OpenAI, Anthropic, Google, AWS, Azure, Ollama, and more)
- **GEPA-powered optimization** — reflective prompt evolution that works with as few as 3 examples, using 35× fewer API calls than traditional methods
- **Multi-strategy evaluation** — BERTScore, embedding similarity, JSON schema validation, and LLM-as-judge, auto-selected by output type
- **Pre-flight safety checks** — capability detection, token budget calculation (catches 15-20% tokenizer inflation), and cost estimation before spending a dollar
- **Confidence scoring** — pairwise win rate tells you exactly: "The new model matches or exceeds the old 92% of the time"
- **Migration reports** — before/after comparisons, per-category breakdowns, worst regressions with diffs

## How It Works

```
Production Data → Pre-flight → Baseline → GEPA Optimization → Validation → Report
 (JSONL/Redis)    Checks        Eval                            Eval
```

RosettaStone ingests your prompt/response pairs, establishes a baseline by
running your existing prompts on the new model (measuring the "migration gap"),
then uses GEPA's reflective optimization to evolve your prompt instructions
until the new model's behavior matches the old. A held-out test set validates
the result, and you get a confidence score and detailed migration report.

## Quick Start

### Install

```bash
pip install rosettastone

# With local evaluation (BERTScore + sentence-transformers)
pip install "rosettastone[eval]"

# With Redis support
pip install "rosettastone[redis]"

# Everything
pip install "rosettastone[all]"
```

### Set up API keys

```bash
export OPENAI_API_KEY=sk-...        # For GEPA reflection model
export ANTHROPIC_API_KEY=sk-ant-... # For target model (if using Anthropic)
```

### Run your first migration

```bash
# Using the included sample data
rosettastone migrate \
  --data examples/sample_data.jsonl \
  --from openai/gpt-4o \
  --to anthropic/claude-sonnet-4

# Estimate cost first
rosettastone migrate \
  --data your_data.jsonl \
  --from openai/gpt-4o \
  --to anthropic/claude-sonnet-4 \
  --dry-run
```

### Or use the Python library

```python
from rosettastone import Migrator, MigrationConfig

config = MigrationConfig(
    source_model="openai/gpt-4o",
    target_model="anthropic/claude-sonnet-4",
    data_path="production_pairs.jsonl",
)

migrator = Migrator(config)
result = migrator.run()

print(f"Confidence: {result.confidence_score:.0%}")
print(f"Improvement over baseline: +{result.improvement:.0%}")
print(f"Cost: ${result.cost_usd:.2f}")
```

## Data Format

RosettaStone accepts JSONL files with one prompt/response pair per line:

```json
{"prompt": "Summarize this article: ...", "response": "The article discusses...", "source_model": "openai/gpt-4o"}
{"prompt": [{"role": "system", "content": "You are a helpful assistant"}, {"role": "user", "content": "..."}], "response": "...", "source_model": "openai/gpt-4o"}
```

Required fields: `prompt`, `response`, `source_model`. Optional: `metadata`,
`feedback` (for known issues), `input_tokens`, `output_tokens`, `timestamp`.

## Architecture

RosettaStone uses a pipeline architecture with pluggable components:

- **Ingest** — Adapter pattern for data sources (JSONL, Redis, LangSmith)
- **Pre-flight** — Capability detection, token budget, cost estimation
- **Optimize** — DSPy + GEPA reflective prompt evolution
- **Evaluate** — Layered metrics auto-selected by output type
- **Report** — Markdown/PDF/HTML migration reports

The optimization uses [GEPA](https://arxiv.org/abs/2507.19457) (Genetic-Pareto),
a reflective optimizer that analyzes *why* the new model's outputs diverge
and proposes targeted prompt improvements. It outperforms MIPROv2 by 10%+
while using 35× fewer API calls (~$2-20 per migration run).

## Expected Costs & Performance

| Target Model | Cost (100 examples) | Time |
|---|---|---|
| GPT-4o-mini | $0.50–$2 | 5–15 min |
| Claude Haiku 4.5 | $2–$6 | 10–25 min |
| GPT-4o | $5–$15 | 15–45 min |
| Claude Sonnet 4.5 | $8–$20 | 20–60 min |

Minimum dataset: 20 pairs. Recommended: 50-200 pairs.

## Roadmap

- [x] **Phase 1** — CLI + Library, JSONL ingestion, GEPA optimization, BERTScore evaluation
- [ ] **Phase 2** — Redis integration, LLM-as-judge, PII detection, known-issue feedback
- [ ] **Phase 3** — Web UI with side-by-side diffs, migration reports, executive dashboard
- [ ] **Phase 4** — LangSmith/Braintrust/OpenTelemetry adapters, CI/CD integration
- [ ] **Phase 5** — Multi-step pipeline migration, A/B testing, enterprise features

## Inspired By

- [Dropbox's DSPy + GEPA migration](https://dropbox.tech/machine-learning/optimizing-dropbox-dash-relevance-judge-with-dspy) — production validation of this exact pattern
- [AWS prompt migration architecture](https://aws.amazon.com/blogs/machine-learning/improve-amazon-nova-migration-performance-with-data-aware-prompt-optimization/) — reference architecture using DSPy MIPROv2
- [GEPA paper](https://arxiv.org/abs/2507.19457) (ICLR 2026 Oral) — reflective prompt evolution
- Apple's Rosetta — binary translation that made software built for one platform run on another

## License

MIT
```

---

# Companion Documents

This Build Document is supported by:

- **`rosettastone-spec-v2.md`** — Complete project specification with all architecture decisions, research validation, phased roadmap, data safety considerations, and technical rationale
- **`rosettastone-folder-structure.md`** — Phased codebase evolution showing exact files added at each phase, with design principles
- **`rosettastone-blog-source.md`** — Compiled narrative material for blog post: problem framing, key insight, approach, data points, and suggested structure
