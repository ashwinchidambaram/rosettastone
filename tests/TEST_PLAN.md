# RosettaStone Comprehensive Test Plan

> E2E, Infrastructure, Observability, and Resilience Testing
> Last updated: 2026-03-31

---

## Table of Contents

1. [E2E Test Scenario Analysis](#1-e2e-test-scenario-analysis)
2. [Playwright UI Test Gap Analysis](#2-playwright-ui-test-gap-analysis)
3. [Alembic Migration Tests](#3-alembic-migration-tests)
4. [Adapter Integration Tests](#4-adapter-integration-tests)
5. [Chaos and Resilience Scenarios](#5-chaos-and-resilience-scenarios)
6. [New E2E Scenarios to Implement](#6-new-e2e-scenarios-to-implement)
7. [Priority-Ranked Missing Tests (Top 20)](#7-priority-ranked-missing-tests-top-20)

---

## 1. E2E Test Scenario Analysis

### 1.1 Existing E2E Tests: Coverage Assessment

**File: `tests/test_e2e/test_smoke.py`**

| Test | What It Asserts | Sufficient? | Gaps |
|---|---|---|---|
| `test_smoke_full_pipeline` | Pipeline completes, confidence 0.0-1.0, recommendation in {GO, CONDITIONAL, NO_GO}, confidence >= scenario min | Partially | Does not assert report generation, does not check `per_type_scores`, does not validate output directory contents |
| `test_smoke_also_works_via_jsonl` | Same as above but via JSONL instead of Redis | Partially | Same gaps; does not assert that JSONL and Redis paths produce comparable results |

**Missing assertions for smoke tests:**
- `result.cost_usd > 0` (cost tracking)
- `result.per_type_scores` is populated
- `result.warnings` is a list (even if empty)
- `result.baseline_score` and `result.improvement` are valid floats
- Output directory contains a report file when `output_dir` is set
- The `result.config` field captures the original configuration

**File: `tests/test_e2e/test_cross_provider.py`**

| Test | What It Asserts | Sufficient? | Gaps |
|---|---|---|---|
| `test_migration_completes` | Layers 1-3 + recommendation in expected set | Good | No timing assertions (e.g., max duration), no cost assertion |
| `test_per_type_scores_populated` | `per_type_scores` is dict with known output types | Good | Does not check that each type has a score > 0 |

**Missing cross-provider variants:**
- OpenAI to Anthropic (e.g., `gpt-4o` -> `claude-sonnet-4`) -- the most common production migration
- OpenAI to Gemini
- Multi-turn conversation prompts (not just single-turn)
- Prompts with system messages
- Very long prompts (near token limit)

**File: `tests/test_e2e/test_model_upgrade.py`**

| Test | What It Asserts | Sufficient? | Gaps |
|---|---|---|---|
| `test_upgrade_completes` | Layers 1-3 | Adequate | No assertion on improvement direction (upgrades should generally improve) |
| `test_upgrade_recommendation` | Recommendation in expected set | Good | Could check that confidence_score >= baseline_score for upgrades |

**Missing upgrade variants:**
- GPT-4o-mini to GPT-4o (OpenAI family)
- Multi-generation upgrades (e.g., GPT-3.5-turbo to GPT-4o)

**File: `tests/test_e2e/test_model_downgrade.py`**

| Test | What It Asserts | Sufficient? | Gaps |
|---|---|---|---|
| `test_downgrade_completes` | Layers 1-2 (no min confidence) | Adequate | Correct that no min confidence is required |
| `test_downgrade_captures_regression` | `per_type_scores` populated, recommendation not None | Adequate | Could check that at least one output type shows regression (score < 1.0) |

**Missing downgrade variants:**
- OpenAI GPT-4o to GPT-4o-mini (most common cost-optimization scenario)
- Sonnet to Haiku with structured output (JSON) -- to test schema degradation detection

### 1.2 E2E conftest.py Assessment

The `conftest.py` provides:
- `redis_url` fixture (DB 15 for isolation)
- `redis_client` with graceful skip on unavailability
- `clean_redis` fixture with flush before/after
- `generated_data_cache` to avoid redundant data generation

**Gaps:**
- No fixture for Postgres-backed E2E tests
- No fixture for file system cleanup (temp directories)
- No fixture for monitoring test duration (would catch performance regressions)
- No fixture for capturing and asserting on structured logs

### 1.3 Scenario Configuration Assessment

The `ScenarioConfig` dataclass and defined scenarios (A1-A3, B1-B2, C1-C2) cover the core migration directions well. However:

**Missing scenario types:**
- **D: Same model, different version** (e.g., `gpt-4o-0613` -> `gpt-4o`)
- **E: Cost-optimization migration** (e.g., `gpt-4o` -> `gpt-4o-mini` with quality threshold)
- **F: Multi-modal model migration** (text-only to vision-capable, or vice versa)
- **G: Output-type-specific scenario** (e.g., JSON-only workload, code-only workload)

---

## 2. Playwright UI Test Gap Analysis

### 2.1 Test Plan vs Implementation Comparison

The Playwright test plan (`PLAYWRIGHT_TEST_PLAN.md`) specifies 87 test functions across 18 sections. The implementation (`test_playwright_ui.py`) maps closely to this plan. Below is a section-by-section gap analysis.

#### Section 1: Models Dashboard -- IMPLEMENTED
All 10 planned tests are implemented. Visual/styling checks (CSS color assertions) are present.

#### Section 2: Models Empty State -- IMPLEMENTED
All 4 planned tests are implemented including input text acceptance and edge case.

#### Section 3: Migrations List -- IMPLEMENTED
All planned tests are implemented including card data, links, pagination, and badge colors.

#### Section 4: Migration Detail (Safe to Ship) -- IMPLEMENTED
All planned tests are implemented including collapsible config, regressions, and export link.

#### Section 5: Migration Detail (Needs Review) -- IMPLEMENTED
All 6 planned tests are implemented including edge cases (no per_type, no regressions).

#### Section 6: Migration Detail (Do Not Ship) -- IMPLEMENTED
All 7 planned tests are implemented.

#### Section 7: Costs Page -- IMPLEMENTED
All 6 planned tests are implemented.

#### Section 8: Alerts Page -- IMPLEMENTED
All 6 planned tests including checkbox toggling are implemented.

#### Section 9: Executive Report -- IMPLEMENTED
All 12 planned tests are implemented including edge case variants (do_not_ship, needs_review, no_regressions).

#### Section 10: Diff Fragment -- IMPLEMENTED
All 5 planned tests are implemented.

#### Section 11: Navigation Bar -- IMPLEMENTED
All 7 planned tests are implemented.

#### Section 12: Theme Toggle -- IMPLEMENTED
All 4 planned tests are implemented using `dark_mode_page` fixture.

#### Section 13: Footer -- IMPLEMENTED
All 4 planned tests are implemented.

#### Section 14: Error Handling (404) -- IMPLEMENTED
4 of 5 planned 404 tests are implemented (missing: negative ID test).

#### Section 15: Slideout Panel Integration -- IMPLEMENTED
All 8 planned lifecycle tests are implemented.

#### Section 16: Do Not Ship Slideout (Hover-to-Reveal) -- NOT IMPLEMENTED

### 2.2 Specific Missing Tests

| Test Plan Section | Missing Test | Plan Reference | Risk |
|---|---|---|---|
| 14 | `test_migration_detail_negative_id_404` | Section 14.3, row for `/ui/migrations/-1` | Low |
| 16 | `test_do_not_ship_hover_reveal_diff_link` | Section 16.1, hover-to-reveal + click pattern | Medium |
| 1.6 | Visual styling: deprecated card gold left border CSS assertion | Section 1.6 | Low |
| 1.6 | Visual styling: active status dots green color CSS assertion | Section 1.6 | Low |
| 1.6 | Visual styling: match % progress bar width assertions | Section 1.6 | Low |
| 3.6 | Visual styling: badge background-color assertions for each recommendation | Section 3.6 | Low |
| 4.5 | HTMX "View diff" slideout content verification (not just panel open, but `#diff-content` contains diff data for specific `tc_id`) | Section 4.5 | Medium |
| 4.7 | Edge case: per_type is empty on safe-to-ship variant | Section 4.7 | Low |
| 4.7 | Edge case: regressions is empty on safe-to-ship variant | Section 4.7 | Low |
| 7.3 | Visual: bar width assertions (`width: 66%`, `width: 25%`, `width: 9%`) | Section 7.3 | Low |
| 8.5 | Forward arrow visibility on hover for alert cards | Section 8.5 | Low |
| 10.7 | Edge case: "Content not stored" message when diff content is null | Section 10.7 | Medium |
| 12.5 | Visual: body background color change in light mode | Section 12.5 | Low |
| 12.6 | Theme toggle on multiple pages (only tested on `/ui/` and `/ui/costs`) | Section 12.6 | Low |

### 2.3 Structural Observations

1. **Test file location**: Tests are in `tests/test_e2e/test_playwright_ui.py` but the plan specifies `tests/test_server/test_playwright_ui.py`. The actual location is correct for E2E scope.

2. **Hardcoded server path**: The `server` fixture uses `cwd="/Users/ashwinchidambaram/dev/projects/rosettastone"` which will fail in CI or on other developer machines. Should use `pathlib.Path(__file__).resolve().parents[2]` or similar.

3. **Port collision**: Using port 8765 is fine for isolation, but `_kill_port` using `lsof` + `kill -9` is aggressive and platform-specific. A more portable approach would be to use a random available port.

4. **No screenshot capture on failure**: The Playwright fixtures do not capture screenshots on test failure, which makes debugging CI failures much harder.

---

## 3. Alembic Migration Tests

### 3.1 Current State

- One migration exists: `c39645f955dc_initial_schema.py` (creates 16 tables with indexes and foreign keys)
- `alembic/env.py` uses `get_engine()` which respects `DATABASE_URL` / `ROSETTASTONE_DB_PATH` env vars
- `render_as_batch=True` for SQLite compatibility
- Docker CMD runs `alembic upgrade head` before starting the server
- No existing Alembic-specific tests exist

### 3.2 Tests to Implement

Create `tests/test_alembic/test_migrations.py`:

#### Test 1: Fresh upgrade to head (SQLite)
```
Scenario: alembic upgrade head on a fresh SQLite database
Setup: tmp_path with clean SQLite file, set ROSETTASTONE_DB_PATH env var
Steps: Run alembic upgrade head via subprocess
Assert:
  - Exit code 0
  - All 16 tables exist (query sqlite_master)
  - alembic_version table has exactly one row with revision c39645f955dc
  - All indexes exist (ix_users_username, ix_users_api_key, etc.)
  - All foreign keys exist (alerts.migration_id -> migrations.id, etc.)
```

#### Test 2: Idempotent upgrade
```
Scenario: Running alembic upgrade head twice produces no error
Setup: Same as Test 1
Steps: Run alembic upgrade head, then run it again
Assert:
  - Both commands exit 0
  - Schema is identical after both runs
  - alembic_version still has exactly one row
```

#### Test 3: Downgrade to base
```
Scenario: alembic downgrade base removes all tables
Setup: Run upgrade head first
Steps: Run alembic downgrade base
Assert:
  - Exit code 0
  - All 16 tables removed
  - alembic_version table shows no current revision (or is empty)
```

#### Test 4: Upgrade-downgrade-upgrade roundtrip
```
Scenario: Full roundtrip preserves schema integrity
Steps: upgrade head -> downgrade base -> upgrade head
Assert:
  - Final schema matches original upgrade head schema
  - No orphaned tables or indexes
```

#### Test 5: Schema parity between create_all() and alembic upgrade head
```
Scenario: SQLModel.metadata.create_all() and alembic upgrade head produce identical schemas
Setup: Two separate SQLite databases
Steps:
  1. DB-A: SQLModel.metadata.create_all(engine_a)
  2. DB-B: alembic upgrade head (with engine_b)
Assert:
  - Same table names
  - Same column names, types, nullable, and defaults per table
  - Same indexes
  - Same foreign key constraints
  - (Note: column order may differ -- ignore ordering)
```

#### Test 6: Postgres compatibility
```
Scenario: alembic upgrade head works on PostgreSQL
Setup: PostgreSQL container or testcontainers
Steps: Run alembic upgrade head against Postgres
Assert:
  - Exit code 0
  - All tables, indexes, and FKs exist
  - Schema types are Postgres-native (not SQLite workarounds)
```

#### Test 7: Data preservation across migration
```
Scenario: Existing data survives a no-op upgrade
Setup: upgrade head, insert sample migration record + related records
Steps: Run upgrade head again
Assert:
  - All inserted data is intact
  - No data loss or corruption
```

---

## 4. Adapter Integration Tests

### 4.1 Existing Adapter Test Coverage

| Adapter | File | Test Count | Coverage |
|---|---|---|---|
| Redis | `test_redis_adapter.py` | 11 | Happy path, scan pagination, unparseable entries, empty Redis, import error, expired keys |
| Redis Formats | `test_redis_formats.py` | ~20+ | LiteLLM, LangChain, GPTCache, RedisVL format parsing |
| LangSmith | `test_langsmith_adapter.py` | 25 | Load, filtering, edge cases, imports, prompt formats |
| Braintrust | `test_braintrust_adapter.py` | 25 | Load, edge cases, imports, field mapping |
| OTel | `test_otel_adapter.py` | 22 | Load, span parsing, edge cases, file discovery |
| CSV | `test_csv_adapter.py` | ~20+ | Standard CSV, TSV, column mapping |

### 4.2 Missing Integration Scenarios

#### Redis Adapter Gaps

| Scenario | Risk | Description |
|---|---|---|
| **Connection timeout handling** | HIGH | What happens when `redis.from_url()` connects but `scan()` times out mid-iteration? |
| **Large dataset pagination** | MEDIUM | Test with 10,000+ keys to verify memory behavior and SCAN cursor correctness |
| **Authentication failure** | MEDIUM | Redis with `requirepass` set but wrong/no password provided |
| **Redis Cluster mode** | LOW | Adapter may not work with Redis Cluster topology |
| **Key pattern filtering** | MEDIUM | Test that `scan(match=...)` patterns work correctly to filter non-LLM-cache keys |
| **Concurrent access** | LOW | Two adapters reading from the same Redis instance simultaneously |
| **Binary/compressed values** | MEDIUM | Redis entries compressed with gzip or msgpack |
| **TTL expiry during load** | MEDIUM | Keys expiring between format detection pass and data collection pass (partially covered by `test_none_get_result_skipped_gracefully` but not for format detection) |

#### LangSmith Adapter Gaps

| Scenario | Risk | Description |
|---|---|---|
| **API rate limiting** | HIGH | `list_runs()` returns 429 -- adapter should retry or fail gracefully |
| **Network timeout** | HIGH | `list_runs()` hangs -- adapter should respect timeout |
| **Very large projects** | MEDIUM | Project with 100,000+ runs -- memory and pagination behavior |
| **Streaming responses** | MEDIUM | Runs with streaming output (partial `generations` entries) |
| **System message extraction** | LOW | Runs where `inputs['messages']` includes system messages -- preservation check |
| **Nested chain runs** | MEDIUM | Runs with `execution_order > 1` that should be filtered out |
| **API key rotation** | LOW | What happens when API key becomes invalid mid-iteration |

#### Braintrust Adapter Gaps

| Scenario | Risk | Description |
|---|---|---|
| **API rate limiting** | HIGH | Same as LangSmith |
| **Project not found** | HIGH | `projects.retrieve()` raises 404 -- should produce clear error |
| **Large log entries** | MEDIUM | Individual log entries with very large input/output (>100KB) |
| **Paginated logs** | MEDIUM | Projects with 50,000+ log entries |
| **Concurrent reads** | LOW | Multiple adapters reading same project |
| **Dict input normalization** | MEDIUM | Input is a dict (not string or list) -- how is it stored in PromptPair? |

#### OTel Adapter Gaps

| Scenario | Risk | Description |
|---|---|---|
| **Protobuf format** | HIGH | OTLP exports can be in protobuf format, not just JSON -- adapter only reads JSON |
| **Very large files** | MEDIUM | Single trace file > 1GB -- memory behavior |
| **Gzipped JSON** | MEDIUM | `.json.gz` files common in OTel exports |
| **NDJSON format** | MEDIUM | Some OTel exporters produce newline-delimited JSON |
| **Missing resourceSpans wrapper** | MEDIUM | Files that just contain `{"scopeSpans": [...]}` without outer wrapper |
| **Span links and span events with exceptions** | LOW | Spans with error events alongside gen_ai events |

### 4.3 Cross-Adapter Integration Tests (Missing Entirely)

These tests should verify that different adapters produce equivalent `PromptPair` objects for the same logical data:

| Test | Description |
|---|---|
| **JSONL vs Redis parity** | Same prompt/response data loaded via JSONL and Redis should produce identical `PromptPair` lists |
| **Adapter output normalization** | All adapters normalize `prompt` field consistently (string for single-turn, list[dict] for multi-turn) |
| **Metadata preservation across adapters** | Each adapter preserves source-specific metadata in `PromptPair.metadata` without leaking adapter internals |

---

## 5. Chaos and Resilience Scenarios

### 5.1 Database Unavailability

| Scenario | Expected Behavior | Test Approach |
|---|---|---|
| DB file deleted mid-migration (SQLite) | Background task logs error, returns failed status, does not crash server | Start migration, delete SQLite file, verify task status becomes "failed" |
| Postgres connection drops during background task | Task retries or fails gracefully; server remains healthy | Use testcontainers to stop Postgres mid-task |
| DB disk full (SQLite) | `OperationalError` caught, task marked failed, no corruption | Mount tmpfs with 1MB limit, fill it |
| DB connection pool exhausted | New requests get 503 or queue; existing tasks complete | Flood with concurrent requests while pool is at max |
| DB locked (SQLite WAL conflict) | Write retries or fails with clear error | Two concurrent writers to same SQLite file |

### 5.2 LLM API Failures

| Scenario | Expected Behavior | Test Approach |
|---|---|---|
| Rate limit (429) from target model during evaluation | LiteLLM/DSPy retry logic kicks in; pipeline eventually completes or fails with clear message | Mock LiteLLM to return 429 for first N calls |
| API key invalid/expired | Clear error message in migration result; does not retry infinitely | Set invalid API key, run migration |
| Model not found (404) | Preflight catches this; if skipped, clear error during baseline eval | Use non-existent model ID |
| Response timeout (30s+) | LiteLLM timeout, retry, then fail; pipeline marks affected test cases | Mock with `time.sleep(60)` |
| Malformed API response (invalid JSON) | Pipeline skips affected pair, logs structural warning | Mock LiteLLM to return garbage |
| API returns empty response | Pipeline handles gracefully, marks as 0.0 score for that pair | Mock empty string response |
| Context window exceeded | Preflight should catch; if skipped, clear truncation or error | Use prompt larger than model's context window |

### 5.3 Redis Failures

| Scenario | Expected Behavior | Test Approach |
|---|---|---|
| Redis connection drops mid-SCAN | `ConnectionError` caught, adapter raises with count of successfully loaded pairs | Kill Redis container during test |
| Redis OOM (maxmemory reached) | Connection works but GET returns errors; adapter handles gracefully | Set `maxmemory 1mb` on test Redis |
| Redis AUTH failure | Clear `AuthenticationError` with message about credentials | Use wrong password |
| Redis returns corrupted data | Adapter skips corrupted entries, logs warning, continues | Manually write binary garbage to keys |
| Redis Sentinel failover during load | Connection to new primary; potential data inconsistency | Requires Sentinel setup |

### 5.4 Docker and Infrastructure

| Scenario | Expected Behavior | Test Approach |
|---|---|---|
| Container disk full | Server returns 503, does not crash | Run in container with limited tmpfs |
| Alembic fails on startup | Container exits with non-zero code, clear error in logs | Corrupt alembic_version table |
| Health check timeout | Docker marks container unhealthy; orchestrator restarts | Overload server to prevent /health response |
| Graceful shutdown during active migration | SIGTERM handler waits for in-progress task, then exits | Send SIGTERM, verify task completes |
| Port already in use | Server fails to bind, clear error message | Start two containers on same port |

### 5.5 File System Failures

| Scenario | Expected Behavior | Test Approach |
|---|---|---|
| JSONL file deleted after ingest starts | Pipeline uses already-loaded data; no mid-pipeline file reads | Delete file after `load_and_split_data` returns |
| Output directory not writable | Report generation fails gracefully; migration result still returned | Use read-only tmpdir |
| Output directory exceeds quota | Partial report written; clear error | Mount small tmpfs |
| Manifest YAML with circular references | Pydantic validation catches before execution | Write YAML with anchors/aliases that create cycles |

---

## 6. New E2E Scenarios to Implement

### 6.1 E2E Pipeline with Postgres

**File:** `tests/test_e2e/test_postgres_pipeline.py`

```
Scenario: Full migration pipeline using Postgres as the backend database
Marker: @pytest.mark.e2e, @pytest.mark.postgres
Setup:
  - Start Postgres via docker-compose --profile postgres (or testcontainers)
  - Set DATABASE_URL=postgresql://rosettastone:rosettastone@localhost:5432/rosettastone
  - Run alembic upgrade head
Steps:
  1. POST /api/v1/migrations with valid config
  2. Poll task status until complete
  3. GET /api/v1/migrations/{id} to retrieve result
Assert:
  - Migration record persisted in Postgres
  - All related records (test_cases, warnings, etc.) created
  - Alembic version table intact
  - Query performance acceptable (<1s for migration detail)
Cleanup:
  - Drop all tables or use a test-scoped database
```

### 6.2 E2E Docker Compose (Build, Start, Health, Migrate, Verify)

**File:** `tests/test_e2e/test_docker_compose.py`

```
Scenario: Full lifecycle via Docker Compose
Marker: @pytest.mark.e2e, @pytest.mark.docker
Preconditions:
  - Docker daemon running
  - No port conflicts on 8000, 5432, 6379
Steps:
  1. docker compose build (assert exit code 0)
  2. docker compose --profile postgres --profile redis up -d (assert exit code 0)
  3. Wait for health check: curl http://localhost:8000/api/v1/health (max 90s)
  4. Assert health response: {"status": "ok"}
  5. POST /api/v1/migrations with minimal config (dry_run=true to avoid needing API keys)
  6. Verify response 200/201
  7. GET /ui/ -- assert 200 with HTML content
  8. docker compose logs app -- assert no ERROR-level log lines
  9. docker compose down -v (cleanup)
Assert:
  - Build completes without error
  - All services become healthy
  - API and UI are reachable
  - alembic upgrade head ran successfully (check logs for "Running upgrade")
  - Non-root user is active (check docker compose exec app whoami == rosettastone)
```

### 6.3 E2E Batch Migration

**File:** `tests/test_e2e/test_batch_e2e.py`

```
Scenario A: Multi-entry manifest, all succeed
Setup: YAML manifest with 3 entries (use cheap models)
Steps: run_batch(manifest, output_dir)
Assert:
  - 3 BatchResult objects returned
  - All status="complete"
  - Summary table includes all 3 names
  - Output files created for each entry

Scenario B: Partial failure handling
Setup: YAML manifest with 3 entries; second entry uses invalid model
Steps: run_batch(manifest, output_dir)
Assert:
  - 3 BatchResult objects returned
  - Entry 1: status="complete"
  - Entry 2: status="failed" or "blocked", error message present
  - Entry 3: status="complete" (continues past failure)
  - Summary shows "1 failed" in aggregate

Scenario C: Dry-run batch
Setup: YAML manifest with dry_run=true on all entries
Steps: run_batch(manifest, output_dir)
Assert:
  - All entries complete quickly (no LLM calls)
  - No output files generated (or minimal)
  - Cost is 0

Scenario D: Defaults merging in batch
Setup: Manifest with defaults.gepa_auto="heavy", one entry overrides to "light"
Steps: Load manifest, run batch
Assert:
  - Default entry uses "heavy"
  - Override entry uses "light"
```

### 6.4 E2E CLI Commands

**File:** `tests/test_e2e/test_cli_e2e.py`

```
Scenario A: Full migrate command with real output
Steps: uv run rosettastone migrate --data examples/sample_data.jsonl --from openai/gpt-4o --to anthropic/claude-sonnet-4 --dry-run
Assert:
  - Exit code 0
  - Output contains "Migration complete"
  - Output contains confidence score

Scenario B: Preflight command
Steps: uv run rosettastone preflight --data examples/sample_data.jsonl --from openai/gpt-4o --to anthropic/claude-sonnet-4
Assert:
  - Exit code 0
  - Output contains "Pre-flight"

Scenario C: Serve command starts and stops
Steps: Start 'uv run rosettastone serve --port 8799' in background, health check, kill
Assert:
  - Health endpoint returns 200
  - Process terminates cleanly on SIGTERM

Scenario D: Missing required arguments
Steps: uv run rosettastone migrate --from openai/gpt-4o (missing --data and --to)
Assert:
  - Exit code != 0
  - Error message mentions missing arguments

Scenario E: Invalid model identifier
Steps: uv run rosettastone migrate --data examples/sample_data.jsonl --from not/a-model --to also/not-real --dry-run
Assert:
  - Either preflight catches it or clear error message

Scenario F: Batch command
Steps: uv run rosettastone batch --manifest examples/batch_manifest.yaml --output-dir /tmp/batch-test
Assert:
  - Exit code 0
  - Summary table printed
  - Output directory populated

Scenario G: --output-dir flag
Steps: uv run rosettastone migrate --data ... --from ... --to ... --output-dir /tmp/test-out --dry-run
Assert:
  - Output directory created
  - Report file written to specified directory
```

---

## 7. Priority-Ranked Missing Tests (Top 20)

Ranked by: (probability of occurring in production) x (severity of impact) x (difficulty to debug without the test)

| Rank | Test | Category | Risk Justification |
|---|---|---|---|
| **1** | **Alembic upgrade head on fresh DB + schema parity with create_all()** | Alembic | Schema drift between ORM and migrations is a silent killer. Any divergence causes production data loss or startup failures. Only one migration exists, so this is easy to validate now but critical to establish as a regression gate. |
| **2** | **E2E Docker Compose build-start-health-migrate cycle** | Docker | The Dockerfile runs `alembic upgrade head && uvicorn ...` -- if either fails, users get a broken container with no clear error. This is the first thing a new user or CI pipeline will try. |
| **3** | **LLM API rate limit (429) handling in pipeline** | Resilience | Rate limiting is the most common production failure mode. Without a test proving retry/backoff works, every high-traffic deployment is at risk of silent pipeline failures. |
| **4** | **Redis connection drop mid-SCAN during ingest** | Resilience | Redis is the primary production data source. SCAN pagination means a connection drop mid-iteration could produce partial data or crash the pipeline. |
| **5** | **Alembic upgrade idempotency** | Alembic | Docker CMD runs `alembic upgrade head` on every container start. If this is not truly idempotent, container restarts corrupt the database. |
| **6** | **E2E batch migration with partial failure handling** | E2E | Batch is how teams will run migrations at scale. If one failure kills the entire batch (instead of continuing), trust in the tool is destroyed. |
| **7** | **Cross-adapter output parity (JSONL vs Redis for same data)** | Integration | If different ingestion paths produce subtly different PromptPair objects, evaluation scores will be inconsistent and debugging will be extremely difficult. |
| **8** | **Database unavailable during background task** | Resilience | Background tasks write results to the DB. If the DB is unreachable when a task completes, the migration result is lost with no recovery path. |
| **9** | **OTel adapter protobuf format support** | Adapter | Many OTel exporters output protobuf by default. If the adapter only handles JSON, a large class of users will get silent empty results. |
| **10** | **E2E pipeline with Postgres backend** | E2E | Docker Compose includes Postgres and the README implies production use. But all current E2E tests use SQLite. Postgres-specific behavior (e.g., enum types, concurrent writes) is untested. |
| **11** | **Alembic downgrade and roundtrip** | Alembic | If the downgrade path is broken, there is no rollback for failed deployments. This is a safety net that must work. |
| **12** | **CLI --output-dir flag creates report files** | CLI | Users expect `--output-dir` to produce reports. If the flag is accepted but no files are written, it is a usability bug. |
| **13** | **Playwright: Do Not Ship hover-to-reveal diff link** | UI | This is the only planned Playwright test that is not implemented. It tests a specific UX interaction pattern (opacity-0 on hover) that could break without being noticed. |
| **14** | **E2E smoke test asserts per_type_scores populated** | E2E | The smoke test does not check `per_type_scores`. If type detection silently breaks, all migrations still "pass" but produce meaningless results. |
| **15** | **Graceful shutdown during active migration** | Docker | SIGTERM while a migration is running could leave the database in an inconsistent state (migration record with status="running" forever). |
| **16** | **Redis AUTH failure produces clear error** | Adapter | Production Redis instances typically require authentication. A cryptic error on auth failure wastes hours of debugging. |
| **17** | **LangSmith adapter API timeout handling** | Adapter | LangSmith API can be slow under load. Without timeout handling, the ingest step hangs indefinitely, blocking the entire pipeline. |
| **18** | **Braintrust adapter project not found error** | Adapter | Typo in project name gives 404. Without a clear error, users blame the tool instead of their config. |
| **19** | **Playwright hardcoded server path** | UI | `cwd="/Users/ashwinchidambaram/dev/projects/rosettastone"` in the Playwright server fixture will fail on any other machine or in CI. Not a test per se, but a blocking issue for test portability. |
| **20** | **E2E CLI with invalid model identifier** | CLI | Users will typo model IDs. The error path should be tested: does preflight catch it, or does the pipeline crash with an inscrutable LiteLLM error? |

---

## Appendix A: Test Infrastructure Improvements

### A.1 Required Fixtures

| Fixture | Scope | Purpose |
|---|---|---|
| `postgres_engine` | session | Testcontainers or docker-compose Postgres for integration tests |
| `alembic_runner` | function | Subprocess runner for alembic commands with isolated DB |
| `docker_compose_up` | session | Brings up docker-compose, yields health-checked URLs, tears down |
| `screenshot_on_failure` | function | Captures Playwright screenshots on test failure for CI debugging |
| `portable_server` | session | Playwright server fixture with dynamic port and relative path |

### A.2 Pytest Markers to Add

```python
# conftest.py
def pytest_configure(config):
    config.addinivalue_line("markers", "e2e: End-to-end tests requiring real APIs or infrastructure")
    config.addinivalue_line("markers", "postgres: Tests requiring a PostgreSQL instance")
    config.addinivalue_line("markers", "docker: Tests requiring Docker daemon")
    config.addinivalue_line("markers", "playwright: Playwright browser tests")
    config.addinivalue_line("markers", "alembic: Alembic migration tests")
    config.addinivalue_line("markers", "resilience: Chaos/resilience tests")
```

### A.3 CI Pipeline Considerations

- **Tier 1 (every PR):** Unit tests, adapter tests with mocks, CLI tests (~2 min)
- **Tier 2 (merge to main):** Playwright UI tests, Alembic tests, batch tests (~10 min)
- **Tier 3 (nightly):** E2E with real LLM APIs, Docker compose tests, Postgres E2E (~30 min)
- **Tier 4 (weekly):** Resilience/chaos tests, cross-provider E2E with all 7 scenarios (~60 min)

---

## Appendix B: Existing Test File Index

| Category | File | Test Count (approx) | Notes |
|---|---|---|---|
| E2E Pipeline | `test_e2e/test_smoke.py` | 2 | Smoke with Redis + JSONL |
| E2E Pipeline | `test_e2e/test_cross_provider.py` | 2 x 3 scenarios | Parametrized A1-A3 |
| E2E Pipeline | `test_e2e/test_model_upgrade.py` | 2 x 2 scenarios | Parametrized B1-B2 |
| E2E Pipeline | `test_e2e/test_model_downgrade.py` | 2 x 2 scenarios | Parametrized C1-C2 |
| E2E UI | `test_e2e/test_playwright_ui.py` | ~85 | Comprehensive Playwright |
| Integration | `test_integration/test_phase2_pipeline.py` | ~10 | Pipeline routing |
| Server | `test_server/test_negative_stress.py` | ~30 | Adversarial API tests |
| Server | `test_server/test_pipeline_runner.py` | ~10 | Background task runner |
| Server | `test_server/test_auth_csrf.py` | ~10 | Auth and CSRF |
| Server | `test_server/conftest.py` | - | Shared server fixtures |
| Adapter | `test_ingest/test_redis_adapter.py` | 11 | Redis with mocks |
| Adapter | `test_ingest/test_redis_formats.py` | ~20 | Format parsers |
| Adapter | `test_ingest/test_langsmith_adapter.py` | 25 | LangSmith with mocks |
| Adapter | `test_ingest/test_braintrust_adapter.py` | 25 | Braintrust with mocks |
| Adapter | `test_ingest/test_otel_adapter.py` | 22 | OTel with files |
| Adapter | `test_ingest/test_csv_adapter.py` | ~20 | CSV/TSV |
| Adapter | `test_ingest/test_jsonl.py` | ~10 | JSONL |
| CLI | `test_cli/test_commands.py` | 17 | Typer CLI tests |
| CLI | `test_cli/test_display.py` | ~5 | Display formatting |
| Batch | `test_batch.py` | 10 | Batch runner |
| Evaluate | `test_evaluate/*` | ~40 | All evaluator strategies |
| Optimize | `test_optimize/*` | ~30 | DSPy, GEPA, MIPROv2 |
| Preflight | `test_preflight/*` | ~15 | Capabilities, token budget, cost |
| Report | `test_report/*` | ~15 | Markdown, HTML, PDF, narrative |
| Safety | `test_safety/*` | ~10 | PII scanner, prompt auditor |
| Decision | `test_decision/*` | ~10 | Recommendation engine, stats |
| Core | `test_core/test_context.py` | ~5 | Pipeline context |
