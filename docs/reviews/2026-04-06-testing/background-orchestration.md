# Background Execution & Orchestration -- Testing Review

**Scope:** task_dispatch.py, task_worker.py, pipeline_runner.py, ab_runner.py, batch.py, core/migrator.py, progress.py
**Reviewer:** Testing Lead (Background Execution & Orchestration)
**Date:** 2026-04-06

---

## 1. Boundary Map

```
INSIDE (we test it)                        OUTSIDE (we mock/stub)
-------------------------------            ---------------------------------
TaskDispatcher                             Redis / RQ  (redis, rq packages)
  .setup(), .enqueue()                     LLM APIs    (litellm)
  .start(), .stop()                        File system (JSONL uploads, output dirs)
  RQ vs DB fallback logic
                                           DB engine   (SQLite in-memory for tests,
TaskWorker                                              Postgres in prod)
  .enqueue(), ._claim_next_task()          DSPy / GEPA optimizer
  ._execute(), ._mark_*()                  BERTScore / embedding evaluators
  .recover_stale_tasks()                   PII scanner
  Retry logic                              Prompt auditor
                                           TeacherStudentOptimizer
pipeline_runner.run_pipeline_background()  litellm model pricing / latency
  YAML parsing, status transitions         SSE event loop (asyncio)
  Stage record persistence                 Metrics recording (Prometheus)

ab_runner
  _run_simulation(), _run_live()
  _commit_results() batch commit
  _conclude_test(), _determine_winner()
  _mark_failed()

batch.py
  load_manifest(), run_batch()
  format_batch_summary()
  Path containment validation

Migrator.run()
  Stage ordering & checkpoint resume
  Progress callback dispatch
  Cost cap enforcement
  GEPA timeout handling
  Regression warning

progress.py
  emit_progress(), register/unregister
  Thread-safe queue fanout

ON THE FENCE
  run_migration_background (api/tasks.py) -- owned by this boundary for the
    background execution path; the DB persistence of TestCaseRecords,
    WarningRecords, and latency sampling straddles database-persistence.
  compute_ab_significance() -- decision module, but called from ab_runner.
```

**External dependencies:**

| Dependency | Where used | Mock strategy |
|---|---|---|
| Redis + RQ | task_dispatch.py | Fake `redis`/`rq` modules via monkeypatch |
| SQLAlchemy / SQLModel | task_worker, pipeline_runner, ab_runner, tasks.py | In-memory SQLite + StaticPool |
| LLM APIs (litellm) | Migrator pipeline, ab_runner live mode, tasks.py latency | patch litellm.completion |
| DSPy / GEPA | optimize_prompt | patch GEPAOptimizer / TeacherStudentOptimizer |
| asyncio event loop | progress.py | Test with real loop or skip (no-loop guard) |
| File system | batch.py data_path, output_dir; tasks.py upload cleanup | tmp_path fixture |
| YAML parser | pipeline_runner, batch.py | Real PyYAML on synthetic inputs |

---

## 2. Current Coverage Audit

### tests/test_server/test_task_dispatch.py

**Covers:**
- DB fallback when REDIS_URL is unset (4 tests: setup, enqueue, start, stop delegation)
- RQ path when Redis is available (5 tests: setup, enqueue, fallback on error, start/stop no-op)

**Misses:**
- `_rq_task_handler` function (the RQ worker entrypoint) is never tested directly. If an unknown task_type arrives, it logs a warning but there is no test for that branch.
- No test for the `"pipeline"` task_type branch in `_rq_task_handler`.
- No test for concurrent enqueue behavior.
- No test for what happens when `db_worker` is None and RQ is also unavailable.

**Brittle:** None. Tests use monkeypatch cleanly.

**Dead tests:** None.

### tests/test_server/test_task_worker.py

**Covers:**
- Enqueue creates queued row
- Task lifecycle: queued -> running -> complete
- Failed task writes error_message
- completed_at timestamp set
- Stale task recovery (3 tests)
- Retry logic: retry-then-succeed, permanent failure after max_retries
- Task dispatch by type (captures task_type in calls list)

**Misses:**
- Priority ordering: no test that a priority=10 task is claimed before a priority=0 task.
- FIFO ordering within same priority: no test.
- `_execute()` real dispatch path (all tests mock `_execute`). The actual branching to `run_migration_background` or `run_pipeline_background` inside `_execute` is never exercised.
- Concurrent claim safety: no test that two workers don't claim the same task.
- Worker stop while task is in-flight: no test for graceful shutdown during execution.
- `correlation_id` parameter: accepted but never tested.
- Metric recording (`record_task_queue_wait`): called in `_claim_next_task` but not verified.
- Unknown task_type raising ValueError in `_execute` is not directly tested.

**Brittle:** The polling tests use `time.sleep(0.05)` loops with 5-10s deadlines. These can flake under CI load. Consider using events instead.

**False-confidence:** `TestTaskDispatch.test_migration_task_calls_run_migration_background` -- despite the name, it patches out `_execute` entirely and just checks the task_type string was captured. It does NOT verify that `run_migration_background` is actually called.

### tests/test_server/test_pipeline_runner.py

**Covers:**
- No data_path -> empty train set
- Stage records created per module
- Optimized prompt persisted per stage
- Duration set on stage records
- Status transitions: complete, failed, not-found

**Misses:**
- Data loading from file (data_path set): the JSONLAdapter path is never tested.
- YAML parsing errors: malformed config_yaml is not tested.
- Partial success: what if optimizer returns results for only some modules?
- Error during stage record persistence (DB write failure).
- Status transition from "pending" -> "running" is verified (the source sets it), but there is no assertion that the pipeline was "running" mid-execution.
- Pipeline with zero modules in YAML.

**Dead tests:** None.

### tests/test_server/test_api_tasks.py (run_migration_background)

**Covers:**
- Success -> status="complete" with result fields persisted
- Failure -> status="failed" with error message
- Blocked -> status="blocked"
- Dry run -> status="dry_run_complete"
- Progress callback passed to Migrator
- Running status set before Migrator.run()
- Orphaned migration recovery (server restart)
- TestCaseRecord creation for validation phase

**Misses:**
- Baseline TestCaseRecords: tests only check validation phase; baseline phase persistence is never verified.
- WarningRecord creation.
- Latency sampling path (`_sample_latency`).
- Cost projection path (`_estimate_per_call_cost`).
- `store_prompt_content` flag: no test that prompt_text is stored on TestCaseRecord.
- Version creation (`create_version`): tested indirectly but not asserted.
- Audit log creation during migration completion.
- Temp file cleanup in the `finally` block.
- Budget tracking (`record_spend`).
- CostLimitExceeded exception handling path.
- Resume parameters (`_resume_from`, `_checkpoint_data`) being extracted from config_dict and passed to Migrator.
- SSE `emit_progress` calls at completion and failure.

**False-confidence:** `test_submit_creates_record_and_redirects` asserts `client.app.state.task_worker.enqueue.called` but does not verify the arguments. The task_worker is fully mocked, so this only proves the endpoint calls `.enqueue()` -- not that the right task_type/payload reaches the worker.

### tests/test_batch.py

**Covers:**
- Manifest loading: valid, invalid YAML, missing required key
- Defaults merging and override
- BatchEntry defaults
- run_batch: success, blocked, failed, mixed results, continues-on-failure
- format_batch_summary: basic, aggregates, empty
- Manifest version default
- output_dir path containment (inside and outside base)

**Misses:**
- Manifest with zero migrations (empty list).
- Manifest with extra/unknown fields (strict mode behavior).
- BatchEntry with all optional fields populated.
- `redis_url`, `reflection_model`, `judge_model` fields being passed through to MigrationConfig.
- Name sanitization: special characters, unicode, empty string.
- run_batch with very large number of entries (resource usage).
- run_batch ordering guarantee (results in same order as manifest entries).

**Dead tests:** None. Coverage is good for the happy path.

### tests/test_core/test_migrator.py

**Covers:**
- Dry run with and without skip_preflight
- MigrationBlockedError raised on blockers
- Warning accumulation through pipeline
- Result shape verification
- Optimized prompt stored in result

**Misses (within orchestration scope):**
- Checkpoint resume: `_already_done()` logic, stage skipping on resume.
- GEPA timeout handling (`GEPATimeoutWithResult`, `TimeoutError`).
- Cost cap enforcement (`max_cost_usd` vs estimated cost).
- Regression warning (optimized score < baseline).
- Progress callback `_emit()` behavior.
- Cost callback wiring with litellm.success_callback.
- `_persist_preflight_estimate()`.
- `_update_migration_failed()`.
- `_checkpoint()` serialization with non-serializable data.

**Note:** Checkpoint callback integration is tested in `test_checkpointing.py` (good).

### tests/test_server/test_checkpointing.py

**Covers:**
- Checkpoint writer saves stage and data
- Writer overwrites previous checkpoint
- Missing record does not raise
- Resume endpoint: failed migration, wrong status (409), no checkpoint (409), complete (409), not found (404)
- Migrator accepts checkpoint/resume params
- Checkpoint callback called after pipeline stages
- Callback exception does not abort migration

**Misses:**
- Resume actually re-running from a checkpoint (end-to-end with mocked pipeline stages).
- Corrupt checkpoint_data_json handling.
- Resume with an invalid/unrecognized stage name.

### tests/test_server/test_api_sse.py

**Covers:**
- Stream endpoint returns 200 with correct content-type
- 404 for unknown migration
- Catch-up event on connect for terminal migration
- Nginx no-buffer and no-cache headers
- Catch-up includes stage progress fields

**Misses:**
- Live progress streaming (non-terminal migration with events pushed).
- Multiple concurrent SSE clients.
- Client disconnect / unregister behavior.
- Queue overflow (maxsize=100) behavior.
- Thread-safety of `emit_progress` across worker and asyncio loop.

### tests/test_server/test_api_ab_testing.py

**Covers:**
- CRUD: create, list, get, start, conclude
- Wrong-status transitions
- Empty metrics
- Metrics cache: hit for concluded, miss for draft, invalidation on conclude, TTL for running
- Stable hash assignment (md5-based)
- Winner determination via mean_diff

**Misses:**
- `run_ab_test_background()` function: never called in tests. All lifecycle tests go through the API layer, which transitions status but does NOT invoke the background runner.
- `_run_simulation()` batch commit every 50 rows.
- `_run_live()` with mocked LLM evaluation.
- `_run_live()` failing when `--store-prompt-content` was not used.
- `_commit_results()` partial failure mid-batch.
- `_conclude_test()` with actual ABTestResult rows (significance computation).
- `_mark_failed()` error handling.
- Empty test cases scenario (no validation TestCaseRecords).

### tests/test_server/test_gepa_iterations.py

**Covers:**
- GEPAIterationRecord persistence
- Multiple iterations ordering
- Callback writes record to DB
- Callback swallows DB errors
- Optimizer history endpoint

**No misses relevant to this boundary.** Good coverage.

### tests/test_integration/test_phase2_pipeline.py

**Covers:**
- Data source routing (Redis vs JSONL)
- Optimizer routing (GEPA vs MIPROv2)
- PII scan pipeline
- Prompt audit pipeline
- Recommendation pipeline
- build_result with/without context
- Cluster-based deduplication

**Misses (within orchestration scope):**
- Full pipeline run with all stages mocked (end-to-end integration).
- Pipeline with checkpoint resume.

---

## 3. Risk Ranking

| # | Failure Mode | Likelihood | Blast Radius | Risk | Existing Tests Catch It? |
|---|---|---|---|---|---|
| 1 | **Task stuck in "running" after worker crash** -- worker dies mid-task, no recovery runs. Task is invisible; migration appears to hang forever. | HIGH | HIGH (lost work, stuck UI) | **CRITICAL** | PARTIAL: `recover_stale_tasks` is tested, but nothing tests that recovery actually runs at server startup. `_recover_orphaned_migrations` is tested in test_api_tasks.py for MigrationRecord, but the TaskQueue-level recovery is not wired into app startup tests. |
| 2 | **Silent data corruption in A/B batch commits** -- `_commit_results` crashes mid-batch (e.g., DB constraint violation on row 30 of 50). First 0-49 results are lost because session.commit() is all-or-nothing per batch. Partial results accepted but significance computed on incomplete data. | MEDIUM | HIGH (wrong winner decision, silent) | **CRITICAL** | NO. No test for partial batch failure. |
| 3 | **Concurrent task claiming (race condition)** -- Two TaskWorkers claim the same queued row. No row-level lock or SELECT ... FOR UPDATE. SQLite hides this because it serializes writes; Postgres would expose it. | MEDIUM | HIGH (duplicate work, cost overrun, corrupt state) | **HIGH** | NO. No concurrent test exists. The `_claim_next_task` method does a read-then-write without any locking. |
| 4 | **GEPA timeout during optimization drops pipeline** -- `TimeoutError` propagates up from GEPA, `_update_migration_failed` is called, but the checkpoint was never written for the optimize stage. Resume cannot recover; must restart from scratch. Long-running migrations lose all progress. | MEDIUM | HIGH (wasted cost, hours of time) | **HIGH** | NO. Timeout path is not tested in Migrator. `GEPATimeoutWithResult` is partially handled but `TimeoutError` (no partial result) is not tested. |
| 5 | **Batch run early-termination on uncaught exception** -- `run_batch` catches `Exception` per entry, but if `MigrationConfig(**config_dict)` raises a `TypeError` or Pydantic `ValidationError`, it is caught. However, if the `Migrator` import itself fails, the entire batch stops. | LOW | HIGH (all remaining migrations skipped) | **HIGH** | PARTIAL: Mixed-results test exists but doesn't test import-level failures. |
| 6 | **SSE progress drops under back-pressure** -- `emit_progress` silently drops events when `asyncio.QueueFull` is raised (maxsize=100). If the optimization stage emits many GEPA iteration events faster than the SSE client consumes, the UI shows stale progress. | HIGH | LOW (cosmetic, no data loss) | **MEDIUM** | NO. No test for queue overflow behavior. |
| 7 | **Pipeline runner YAML injection** -- Malformed or adversarial `config_yaml` in PipelineRecord could cause `yaml.safe_load` to produce unexpected structures. `PipelineConfig(**raw)` may then fail with an opaque error or succeed with wrong values. | LOW | MEDIUM (failed pipeline, potential bad optimization) | **MEDIUM** | NO. No malformed YAML test for pipeline_runner. |
| 8 | **Cost cap checked only at preflight, not during GEPA** -- `max_cost_usd` is checked against the pre-flight estimate, but the actual GEPA cost callback raises `CostLimitExceeded` which is caught and handled. However, the test coverage for this path is zero. | MEDIUM | MEDIUM (cost overrun) | **MEDIUM** | NO. `CostLimitExceeded` path in `run_migration_background` is not tested. |
| 9 | **ab_runner live mode with no prompt content** -- `_run_live` raises `ValueError` when test cases lack `prompt_text`. This is caught by the outer `except Exception`, which calls `_mark_failed`. But the test never runs `run_ab_test_background` at all. | LOW | MEDIUM (A/B test fails unexpectedly) | **MEDIUM** | NO. |
| 10 | **Checkpoint resume skips preflight but misses cost check** -- On resume from baseline_eval, preflight is marked `_already_done`. If the original run's cost estimate was below cap but actual costs now exceed it, the resumed run proceeds without rechecking. | LOW | LOW (minor cost overrun) | **LOW** | NO. |

---

## 4. Test Plan by Tier

### Unit Tests

| Test | File | Status | Assertions | Size |
|---|---|---|---|---|
| TaskDispatcher: unknown task_type in _rq_task_handler logs warning | test_task_dispatch.py | MISSING | logger.warning called with "Unknown task type" | S |
| TaskDispatcher: pipeline task_type routes correctly in _rq_task_handler | test_task_dispatch.py | MISSING | run_pipeline_background called with pipeline_id | S |
| TaskWorker: priority ordering (higher priority claimed first) | test_task_worker.py | MISSING | Two tasks enqueued, higher priority one executed first | S |
| TaskWorker: FIFO within same priority | test_task_worker.py | MISSING | Two tasks same priority, older one claimed first | S |
| TaskWorker: unknown task_type raises ValueError in _execute | test_task_worker.py | MISSING | _mark_failed called with ValueError message | S |
| Migrator: _already_done returns correct booleans for each stage | test_migrator.py | MISSING | Unit test of stage ordering logic | S |
| Migrator: regression warning when optimized < baseline | test_migrator.py | MISSING | ctx.warnings includes regression message | S |
| Migrator: _checkpoint with non-serializable data falls back | test_migrator.py | MISSING | No exception, checkpoint_callback still called | S |
| Migrator: GEPA timeout with partial result uses exc.instructions | test_migrator.py | MISSING | optimized_prompt == exc.instructions, warning added | M |
| Migrator: GEPA timeout without partial result re-raises | test_migrator.py | MISSING | TimeoutError propagates, warning added to ctx | M |
| Migrator: cost cap blocks when estimate exceeds max_cost_usd | test_migrator.py | MISSING | ValueError raised with cost message | S |
| Migrator: _persist_preflight_estimate writes to DB | test_migrator.py | MISSING | DB record updated with estimated_cost_usd | S |
| ab_runner: _determine_winner edge cases | test_api_ab_testing.py | EXISTS | -- | -- |
| ab_runner: _evaluate_with_prompt fallback to 0.5 | test_server/test_ab_runner.py | MISSING | ImportError returns 0.5 | S |
| batch.py: empty migrations list | test_batch.py | MISSING | Returns empty results list | S |
| batch.py: name sanitization (special chars, empty) | test_batch.py | MISSING | Sanitized path has no dangerous chars | S |
| batch.py: optional fields (redis_url, reflection_model, judge_model) passed through | test_batch.py | MISSING | MigrationConfig receives these kwargs | S |
| progress.py: register/unregister client lifecycle | test_progress.py | MISSING | Queue added/removed from _queues dict | S |
| progress.py: emit_progress with no registered clients is no-op | test_progress.py | MISSING | No exception, silent skip | S |
| progress.py: emit_progress drops event on QueueFull | test_progress.py | MISSING | No exception when queue is full | S |

### Contract Tests

| Test | File | Status | Assertions | Size |
|---|---|---|---|---|
| TaskDispatcher.enqueue contract: same signature whether RQ or DB | test_task_dispatch.py | PARTIAL (both paths tested but not verified to be interchangeable) | Both paths accept (task_type, resource_id, payload) | S |
| TaskWorker._execute dispatches to correct handler by task_type | test_task_worker.py | MISSING | "migration" calls run_migration_background, "pipeline" calls run_pipeline_background | M |
| run_migration_background: result fields contract with MigrationRecord columns | test_api_tasks.py | PARTIAL | Verify all MigrationResult fields are persisted to correct columns | M |
| Migrator callback contracts: progress_callback(stage, stage_pct, overall_pct) | test_migrator.py | MISSING | Verify signature matches what tasks.py provides | S |
| Migrator callback contracts: checkpoint_callback(stage, data_json) | test_checkpointing.py | EXISTS | -- | -- |

### Integration Tests

| Test | File | Status | Assertions | Size |
|---|---|---|---|---|
| Full pipeline with mocked LLM: preflight -> ingest -> eval -> optimize -> eval -> report | test_integration/ | PARTIAL (test_phase2_pipeline tests subsystems individually) | All stages execute in order, result is valid MigrationResult | L |
| run_migration_background end-to-end with mocked Migrator | test_api_tasks.py | EXISTS | Status transitions, result persistence | -- |
| run_pipeline_background end-to-end with mocked optimizer | test_pipeline_runner.py | EXISTS | Status transitions, stage records | -- |
| run_ab_test_background simulation mode with test data | test_server/ | MISSING | ABTestResult rows created, significance computed, winner set | L |
| run_ab_test_background live mode with mocked evaluator | test_server/ | MISSING | Scores computed per test case, batch committed | L |
| TaskWorker claims and executes real run_migration_background (mocked Migrator) | test_task_worker.py | MISSING | End-to-end: enqueue -> claim -> execute -> complete | L |
| Batch run with 3 entries (success, blocked, failed) through full stack | test_batch.py | EXISTS (test_run_batch_mixed_results) | -- | -- |
| Resume from checkpoint: re-run from baseline_eval stage | test_integration/ | MISSING | Preflight skipped, baseline restored from checkpoint, optimize runs | XL |
| Pipeline runner with data_path set, loading JSONL | test_pipeline_runner.py | MISSING | Train set populated from file | M |

### Property-Based Tests

| Test | File | Status | Assertions | Size |
|---|---|---|---|---|
| BatchManifest: arbitrary valid YAML round-trips through load_manifest | test_batch.py | MISSING | Hypothesis generates BatchManifest, dumps to YAML, loads back, fields match | M |
| PipelineConfig: arbitrary valid YAML round-trips | test_pipeline_runner.py | MISSING | Hypothesis generates config, YAML round-trip produces equivalent config | M |
| ab_runner: hash assignment is deterministic for any (tc_id, traffic_split) | test_api_ab_testing.py | PARTIAL (tested with fixed values) | Hypothesis with random tc_id/split, same input always same output | S |
| TaskQueue: any valid task_type/payload round-trips through enqueue -> claim | test_task_worker.py | MISSING | Hypothesis generates payloads, verify round-trip integrity | M |
| Migrator: _already_done monotonicity -- if stage X is done, all stages before X are done | test_migrator.py | MISSING | For any valid resume_stage, earlier stages return True | S |

### End-to-End Tests (Cost-Gated)

| Test | File | Status | Assertions | Size |
|---|---|---|---|---|
| Real pipeline run with live LLM (tiny dataset, 3 pairs) | test_e2e/ | MISSING | Full result with real scores, report generated | XL |
| Real A/B test with live evaluation | test_e2e/ | MISSING | Real scores, significance computed | XL |

**Cost gate:** These require `ROSETTASTONE_E2E=1` env var and consume API credits. Skip in CI by default.

---

## 5. Synthetic Data Generation Strategy

### Scenarios needed

1. **Long-running task states:** TaskQueue rows in each status (queued, running, complete, failed, cancelled) with realistic timestamps. Used for worker recovery, priority ordering, and metric tests.

2. **Crash-recovery states:** MigrationRecord with status="running", checkpoint_stage="baseline_eval", checkpoint_data_json containing serialized eval results. Simulates a worker crash mid-pipeline.

3. **Partial A/B results:** 75 ABTestResult rows (1.5 batches) to test batch commit boundary behavior. Mix of winners (a, b, tie) with realistic score distributions.

4. **Pipeline YAML configs:** Valid and invalid YAML strings for PipelineConfig. Include edge cases: zero modules, circular depends_on, missing required fields, extra fields.

5. **Batch manifests:** YAML manifests with 1, 5, and 50 entries. Include entries that will succeed, block, and fail.

### Generation approach

- **Factories over fixtures:** Use `factory_boy` or simple factory functions (like the existing `_insert_migration` helper) that accept overrides. Avoid complex fixture trees.
- **Deterministic seeds:** For hash-based A/B assignment tests, use fixed tc_ids with known md5 buckets.
- **Hypothesis strategies:** For property-based tests on BatchManifest and PipelineConfig, define Hypothesis strategies that generate valid Pydantic models.

### Stability and rot prevention

- Factory functions live alongside tests (or in a shared `tests/factories.py`).
- No external data files -- everything is generated in-process.
- Hypothesis database cached in `.hypothesis/` (gitignored).
- Cost: zero. All synthetic, no LLM calls.

---

## 6. Fixtures, Fakes, and Mocks

### New fixtures needed

| Fixture | Scope | Description | Shared? |
|---|---|---|---|
| `mem_engine` | function | In-memory SQLite + StaticPool. Already duplicated across 4+ test files. | YES -- **cross-subagent with database-persistence**. Should be a single conftest fixture. |
| `task_worker` | function | TaskWorker(mem_engine, poll_interval=0.05) with teardown. Already in test_task_worker.py. | No, local to background tests. |
| `sample_ab_test` | function | Creates MigrationRecord + 2 MigrationVersions + ABTest in draft status. | Local. |
| `sample_test_cases_for_ab` | function | Creates 100 TestCaseRecords with validation phase and composite scores. | Local. |
| `sample_pipeline_record` | function | PipelineRecord with valid config_yaml. Already exists in test_pipeline_runner.py as a helper. Promote to fixture. | Local. |

### New fakes

| Fake | Purpose |
|---|---|
| `FakeMigrator` | Accepts all Migrator constructor args, `.run()` returns a canned MigrationResult. Already exists in test_api_tasks.py as an inline class. Extract to shared location. |
| `FakeTeacherStudentOptimizer` | Returns a dict of module_name -> "optimized prompt". Already done via MagicMock in test_pipeline_runner.py. |
| `FakeEvaluator` | Returns canned EvalResult list. Needed for ab_runner live mode tests. |

### New mocks

| Mock target | Tests that need it |
|---|---|
| `rosettastone.server.progress.emit_progress` | All background runner tests that don't care about SSE. Already patched in test_gepa_iterations.py. |
| `rosettastone.server.metrics.record_task_queue_wait` | TaskWorker claim tests if we want to verify metric emission. |
| `rosettastone.evaluate.exact_match.string_similarity` | ab_runner live mode tests. |
| `rosettastone.decision.ab_stats.compute_ab_significance` | ab_runner conclude tests with controlled p-values. |

### Cross-subagent shared fixtures

- **`mem_engine` / `engine`**: Duplicated in test_task_worker.py, test_pipeline_runner.py, test_api_tasks.py, test_gepa_iterations.py, and tests/test_server/conftest.py. All use the same pattern. Should consolidate into a single conftest.py fixture at tests/ root or tests/test_server/conftest.py. **Coordinate with database-persistence subagent.**

- **`sample_migration` / `_insert_migration`**: Used by both this boundary and database-persistence tests. The conftest version includes full JSON fields; the task_worker version is minimal. Both are valid for their contexts. **No change needed** unless we want to reduce duplication.

- **`FakeMigrator` / mock pipeline steps**: Could be shared with ingest-adapters tests if they need to test the full pipeline path. Currently not needed. **Flag for future.**

---

## 7. Gaps You Can't Close

1. **Redis/RQ in-process testing.** The RQ path creates a real `rq.Queue` and enqueues jobs for a separate `rq-worker` process. We cannot test this end-to-end without a running Redis instance and an rq-worker. The fake modules test the dispatch logic, but the actual RQ serialization/deserialization of the `_rq_task_handler` function and its arguments is untested. **Mitigation:** Manual smoke test with `docker compose up redis` + `rq worker`.

2. **Race conditions in task claiming.** SQLite serializes all writes, so the `_claim_next_task` read-then-write pattern works fine in tests. Under Postgres with concurrent workers, this is a real TOCTOU race. We cannot reproduce this without Postgres + multiple threads/processes. **Mitigation:** Add `SELECT ... FOR UPDATE` or advisory locks in source code (not a test fix). NEEDS_HUMAN_REVIEW: Is this planned for P0.2?

3. **SSE integration with real asyncio loop + background thread.** The `emit_progress` function uses `loop.call_soon_threadsafe()`. Testing this requires a running asyncio event loop and a separate thread calling `emit_progress`. The TestClient does not provide a real running loop for streaming endpoints. **Mitigation:** The current catch-up tests are adequate for the SSE endpoint; the thread-safety of `emit_progress` should be tested with a manual asyncio test harness.

4. **Actual LLM cost tracking accuracy.** The `_make_gepa_cost_callback` accumulates `response_cost` from litellm's success_callback. We cannot verify that litellm provides accurate cost data without real API calls. **Mitigation:** Trust litellm's cost reporting; test the accumulation logic only.

5. **A/B test live mode with real LLM evaluation.** `_evaluate_with_prompt` currently falls back to `string_similarity` which is a crude approximation. Testing whether this produces meaningful scores requires real model calls. **Mitigation:** Cost-gated E2E test only.

---

## 8. Cost and Time Estimate

| Tier | New Tests | Estimated Write Time | Run Time |
|---|---|---|---|
| Unit | ~20 tests | 4-6 hours | <10 seconds |
| Contract | ~5 tests | 2-3 hours | <5 seconds |
| Integration | ~6 tests | 6-8 hours | <30 seconds |
| Property-based | ~5 tests | 3-4 hours | <15 seconds |
| End-to-end | ~2 tests | 2-3 hours | 2-5 minutes (cost-gated) |
| Fixture consolidation | -- | 2 hours | -- |
| **Total** | **~38 tests** | **19-26 hours** | **<1 minute (non-E2E)** |

LLM API cost for E2E tests: ~$0.10-0.50 per run (3-5 pairs, tiny models).

---

## 9. Path to Production

### Current readiness level

**P0.1 (intermediate writes):** PARTIAL. Checkpointing is implemented and tested. The `_checkpoint()` method is resilient to callback failures. Resume endpoint exists with proper guard rails. But: no integration test proves that a resumed migration actually skips stages correctly and produces a valid result.

**P0.2 (persistent task queue):** MOSTLY DONE. TaskWorker with DB-backed queue is implemented. Retry logic exists. Recovery of stale tasks exists. But: no concurrent-safety guarantee (see Risk #3), and the recovery is not wired into an app startup test.

### Gap to production-hardened

1. **Concurrent task safety** -- The biggest gap. The `_claim_next_task` method is not safe under concurrent Postgres workers. Fix: `SELECT ... FOR UPDATE SKIP LOCKED` in `_claim_next_task`. Test: run with Postgres in CI.
2. **A/B runner background path** -- Completely untested. The `run_ab_test_background` function is never called from any test. This is the #1 missing coverage area.
3. **Pipeline runner data loading** -- The JSONL file loading path is never tested in pipeline_runner.
4. **Cost limit enforcement** -- The `CostLimitExceeded` path through `run_migration_background` is not tested.

### Gates

| Gate | Status | Blocking? |
|---|---|---|
| All unit tests pass | Not runnable (no new tests yet) | YES |
| Integration test: full pipeline with mocked LLM | MISSING | YES |
| Integration test: A/B runner simulation mode | MISSING | YES |
| Concurrent claim safety (Postgres) | Not addressable in tests alone; needs source fix | YES for multi-worker deploy |
| E2E test with real LLM | MISSING | NO (nice-to-have) |
| Fixture consolidation | Not started | NO (quality of life) |

### Ordered sequence

1. **Write unit tests for Migrator timeout/cost/resume logic** (highest risk-to-effort ratio, 4h)
2. **Write ab_runner integration tests** (simulation mode + live mode with mocks, 4h)
3. **Write TaskWorker priority ordering + _execute dispatch tests** (2h)
4. **Write progress.py unit tests** (register/unregister/emit/overflow, 2h)
5. **Write pipeline_runner data loading + malformed YAML tests** (2h)
6. **Write property-based tests for BatchManifest + PipelineConfig** (3h)
7. **Consolidate engine/fixture duplication** (2h)
8. **Add CostLimitExceeded path test in test_api_tasks.py** (1h)
9. **Add concurrent claim test (requires Postgres in CI)** (3h)
10. **Add cost-gated E2E tests** (3h)

### Smallest next slice

Write 4 tests for `run_ab_test_background` simulation mode:
- Happy path with 10 test cases
- Batch commit boundary (>50 test cases)
- Empty test cases returns early
- Conclude sets winner from significance

This takes ~3 hours and closes the largest single coverage gap (Risk #2 + the entirely untested ab_runner background path).

### Dependencies on other boundaries

- **database-persistence:** Shares `mem_engine` fixture. MigrationRecord, TestCaseRecord, ABTest, ABTestResult models are defined there. Any schema changes break these tests.
- **ingest-adapters:** Pipeline runner loads data via JSONLAdapter. If the adapter interface changes, pipeline_runner tests need updating.
- **decision module:** ab_runner calls `compute_ab_significance`. If the return type of `ABSignificanceResult` changes, `_determine_winner` and `_conclude_test` break.
- **evaluate module:** ab_runner live mode calls `string_similarity`. Migrator calls `CompositeEvaluator`. Interface changes affect integration tests.
