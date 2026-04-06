# Unified Synthetic Data Strategy

## Principles

1. **Zero API cost by default.** All fixtures in the fast test tier are handwritten, seeded-RNG-generated, or property-based. Real API calls are reserved for cost-gated E2E tests behind `@pytest.mark.e2e`.
2. **Determinism first.** Every fixture is either a checked-in file or produced by a seeded generator. No test depends on network state, clock time, or random output.
3. **One source of truth per fixture type.** Shared generators live in `tests/fixtures/generators/`. Shared Hypothesis strategies live in `tests/conftest_strategies.py`. No duplicate fixture factories across test files.
4. **Rot detection built in.** Generator scripts have a `--check` mode run in CI. Golden files are loaded by at least one test. Schema changes break strategy definitions via Pydantic.

## Tiered Fixture Catalog

### Tier 1: Fast/Frozen (checked in, zero cost, sub-second)

| Fixture | Location | Owner | Used By |
|---|---|---|---|
| `valid_simple.jsonl` | `tests/fixtures/ingest/` | ingest-adapters | JSONL adapter, pipeline integration |
| `valid_multi_turn.jsonl` | `tests/fixtures/ingest/` | ingest-adapters | JSONL adapter multi-turn path |
| `valid_unicode.jsonl` | `tests/fixtures/ingest/` | ingest-adapters | Encoding edge cases |
| `edge_cases.jsonl` | `tests/fixtures/ingest/` | ingest-adapters | Empty metadata, long prompts |
| `standard.csv`, `standard.tsv` | `tests/fixtures/ingest/` | ingest-adapters | CSV/TSV adapter |
| `bom_encoded.csv` | `tests/fixtures/ingest/` | ingest-adapters | BOM handling |
| `quoted_multiline.csv` | `tests/fixtures/ingest/` | ingest-adapters | RFC 4180 edge cases |
| `otlp_mixed.json` | `tests/fixtures/ingest/` | ingest-adapters | OTel adapter |
| `redis_*.json` (5 files) | `tests/fixtures/ingest/` | ingest-adapters | Redis format detection |
| `pii_test_patterns.py` | `tests/fixtures/safety/` | safety-observability | PII scanner, constants module |
| `calibration_synthetic.json` | `tests/fixtures/calibration/` | safety-observability | Calibrator tests |
| `score_distributions.py` | `tests/fixtures/evaluation/` | evaluation-strategies | Wilson CI, recommendation tests |
| `api_schema_snapshots/v1/*.json` | `tests/fixtures/server/` | server-http-security | API contract tests |
| `migration_result_factory.py` | `tests/fixtures/shared/` | ALL | Report, CLI, server, evaluation |

### Tier 2: Generated on Demand (seeded, zero cost, seconds)

| Generator | Location | Output | Trigger |
|---|---|---|---|
| `generate_redis_fixtures.py` | `tests/fixtures/generators/` | Redis format JSON files | Manual or CI on ingest changes |
| `generate_calibration_data.py` | `tests/fixtures/generators/` | Calibration datasets with controlled agreement rates | Manual or CI on calibration changes |
| `generate_migration_db.py` | `tests/fixtures/generators/` | Pre-populated SQLite DB for integration tests | Manual or CI on model changes |

### Tier 3: Cost-Gated (real APIs, behind markers)

| Fixture Source | Marker | Cost Per Run | Owner |
|---|---|---|---|
| VCR cassettes (LLM judge responses) | `@pytest.mark.vcr` | $0 (replay) / $0.05 (record) | evaluation-strategies |
| VCR cassettes (GEPA optimization) | `@pytest.mark.vcr` | $0 (replay) / $3 (record) | optimization-engines |
| VCR cassettes (narrative generation) | `@pytest.mark.vcr` | $0 (replay) / $0.02 (record) | report-generation-cli |
| Ollama integration tests | `@pytest.mark.ollama` | $0 (local) | optimization-engines |
| LangSmith/Braintrust API tests | `@pytest.mark.e2e` | $0 (free tier) | ingest-adapters |

## Shared Infrastructure

### `migration_result_factory` (highest priority shared fixture)

Currently 6+ test files duplicate `_make_result()` / `_base_result()` helpers with subtle differences. This causes drift and makes tests fragile to `MigrationResult` schema changes.

**Proposed:** A single `migration_result_factory()` function in `tests/fixtures/shared/migration_result_factory.py` that:
- Returns a valid `MigrationResult` with all required fields populated
- Accepts `**overrides` for any field
- Includes convenience presets: `factory(preset="go")`, `factory(preset="no_go")`, `factory(preset="conditional")`
- Used by: report tests, CLI tests, server API tests, evaluation integration tests

### Hypothesis Strategies (`tests/conftest_strategies.py`)

Shared property-based generators:

| Strategy | Produces | Used By |
|---|---|---|
| `prompt_pair_strategy()` | `PromptPair` with text or list-of-messages prompts | ingest, evaluate, optimize |
| `jsonl_line_strategy()` | Valid JSONL line dicts | ingest |
| `redis_entry_strategy(format)` | `(key, value)` tuples per Redis format | ingest |
| `eval_score_strategy()` | `float` in [0.0, 1.0] | evaluation, decision |
| `migration_config_strategy()` | Valid `MigrationConfig` | optimize, pipeline |
| `pii_text_strategy()` | Text with embedded PII patterns | safety |

### VCR Cassette Infrastructure

**Recommended tool:** `pytest-recording` (wrapper around `vcrpy`).

**Why not respx:** respx is httpx-only. LiteLLM uses both httpx and requests depending on provider. vcrpy intercepts at the socket level, catching both.

**Cassette organization:**
```
tests/cassettes/
  evaluate/
    llm_judge_gpt4o_mini_score_3.yaml
    llm_judge_gpt4o_mini_score_5.yaml
  optimize/
    gepa_gpt4o_mini_3_iterations.yaml
    mipro_gpt4o_mini_zero_shot.yaml
  report/
    narrative_gpt4o_mini.yaml
```

**Recording protocol:**
1. Run with `--record-mode=new_episodes` once, manually review cassettes
2. Strip API keys and PII from cassettes before committing
3. CI runs with `--record-mode=none` (replay only)
4. Re-record when: API response format changes, model version bumped, new test scenario added

## Directory Layout

```
tests/
  fixtures/
    shared/
      migration_result_factory.py    # Shared MigrationResult builder
    ingest/
      valid_simple.jsonl
      valid_multi_turn.jsonl
      valid_unicode.jsonl
      edge_cases.jsonl
      standard.csv
      standard.tsv
      bom_encoded.csv
      quoted_multiline.csv
      otlp_mixed.json
      otlp_events_only.json
      redis_litellm.json
      redis_langchain.json
      redis_redisvl.json
      redis_gptcache.json
      redis_mixed.json
    safety/
      pii_test_patterns.py           # Constants: fake emails, SSNs, phones
    calibration/
      calibration_synthetic.json
    evaluation/
      score_distributions.py         # Named distributions for property tests
    server/
      api_schema_snapshots/
        v1/
          migrations_list.json
          migration_detail.json
          health.json
    generators/
      generate_redis_fixtures.py
      generate_calibration_data.py
      generate_migration_db.py
  cassettes/
    evaluate/
    optimize/
    report/
  conftest_strategies.py             # Shared Hypothesis strategies
```

## Rot Prevention Plan

| Mechanism | Scope | Trigger |
|---|---|---|
| Golden fixture loaded by test | All Tier 1 files | Every test run |
| Generator `--check` mode | All generators | CI on source changes to matching boundary |
| Hypothesis strategies built on Pydantic models | All property-based tests | Pydantic schema change breaks strategy |
| VCR cassette replay | All cassette tests | CI on every run (replay mode) |
| `migration_result_factory` type checks | Shared factory | `MigrationResult` schema change |
| API schema snapshot diff | Server contract tests | CI on every PR touching server/ |

## Inconsistency Resolution

Two areas where subagent reports proposed different approaches:

1. **VCR tool:** Evaluation-strategies proposed `vcrpy` directly; optimization-engines proposed `pytest-recording`. **Resolution:** Use `pytest-recording` (wrapper around vcrpy) for both. It integrates with pytest fixtures cleanly and handles cassette naming automatically.

2. **PII test data:** Safety-observability proposed constants in a Python module; ingest-adapters proposed inline in test files. **Resolution:** Use the constants module (`tests/fixtures/safety/pii_test_patterns.py`) and import from there in both safety and ingest tests. Single source of truth for fake PII.
