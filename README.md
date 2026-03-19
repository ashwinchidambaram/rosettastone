# RosettaStone

**Automated LLM model migration. Your production data is the training signal.**

---

Switching LLM providers shouldn't mean weeks of prompt re-engineering.
RosettaStone takes your existing prompt/response pairs — from JSONL files,
Redis caches, or observability platforms — and automatically optimizes your
prompts for any new model. Powered by
[GEPA](https://arxiv.org/abs/2507.19457) (ICLR 2026 Oral) and
[DSPy](https://dspy.ai), it achieves behavioral equivalence in minutes
instead of weeks.

**The core idea:** your production cache data *is* the behavioral spec.
The old model's actual outputs are the ground truth for what the new model
should reproduce. RosettaStone turns that data into automated migration.

## Features

- **One-command migration** — `rosettastone migrate --from gpt-4o --to claude-sonnet-4`
- **Provider-agnostic** — supports 100+ models via LiteLLM
- **GEPA-powered optimization** — reflective prompt evolution using 35x fewer API calls than traditional methods
- **Multi-strategy evaluation** — BERTScore, embedding similarity, JSON validation, exact match, auto-selected by output type
- **Pre-flight safety checks** — capability detection, token budget calculation, cost estimation
- **Confidence scoring** — pairwise win rate tells you exactly how well the migration worked
- **Migration reports** — before/after comparisons, per-category breakdowns, worst regressions

## Quick Start

### Install

```bash
pip install rosettastone

# With local evaluation (BERTScore + sentence-transformers)
pip install "rosettastone[eval]"

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

### Python library

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

JSONL files with one prompt/response pair per line:

```json
{"prompt": "Summarize this article: ...", "response": "The article discusses...", "source_model": "openai/gpt-4o"}
{"prompt": [{"role": "system", "content": "You are a helpful assistant"}, {"role": "user", "content": "..."}], "response": "...", "source_model": "openai/gpt-4o"}
```

Required fields: `prompt`, `response`, `source_model`. Optional: `metadata`, `feedback`, `input_tokens`, `output_tokens`, `timestamp`.

## Architecture

Pipeline architecture with pluggable components:

- **Ingest** — Adapter pattern for data sources (JSONL, Redis, LangSmith)
- **Pre-flight** — Capability detection, token budget, cost estimation
- **Optimize** — DSPy + GEPA reflective prompt evolution
- **Evaluate** — Layered metrics auto-selected by output type
- **Report** — Markdown/PDF/HTML migration reports

## Development

```bash
git clone https://github.com/YOUR_USERNAME/rosettastone.git
cd rosettastone
uv sync --dev --all-extras

# Run tests
uv run pytest tests/ -v

# Lint
uv run ruff check src/ tests/
uv run ruff format src/ tests/

# Type check
uv run mypy src/rosettastone/
```

## License

MIT
