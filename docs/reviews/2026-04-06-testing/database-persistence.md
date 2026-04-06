# Database, Persistence & Migrations — Test Review

**Date:** 2026-04-06
**Scope:** database.py, models.py, api/migrations.py (DB write paths), Alembic migration chain, schema parity, data preservation
**Status:** Report only. No code changes.

---

## 1. Boundary Map

```
INSIDE (we test directly)
├── database.py
│   ├── get_engine()          — singleton, env-driven SQLite/Postgres switch
│   ├── init_db()             — create_all + _migrate_add_columns
│   ├── _migrate_add_columns  — safety-net ALTER TABLE for 9 columns + 1 index + 1 table
│   ├── _is_postgres()        — dialect check
│   ├── get_session()         — FastAPI DI generator
│   └── reset_engine()        — test teardown
├── models.py
│   ├── 21 SQLModel table classes (RegisteredModel through DatasetGenerationRun)
│   ├── Field defaults, nullable flags, foreign keys, unique constraints
│   └── Relationships (MigrationRecord → TestCaseRecord, WarningRecord, GEPAIterationRecord)
├── Alembic migration chain
│   ├── c39645f955dc  — initial schema (16 base tables)
│   ├── 1a58d561a346  — unique constraint on approvals(workflow_id, user_id)
│   ├── a1b2c3d4e5f6  — production readiness (task_queue, user_budgets, 8 migration cols, pipeline/ab_test cols)
│   ├── b7c8d9e0f1a2  — dataset_generation_runs table
│   ├── d4e5f6a7b8c9  — total_tokens + token_breakdown_json columns
│   ├── e5f6a7b8c9d0  — optimization_score_history_json column
│   ├── c1d2e3f4a5b6  — gepa_iterations table
│   ├── d2e3f4a5b6c7  — failure_reason column on test_cases
│   └── fc47a11b96ba  — merge head (monitoring + observability branches)
├── alembic/env.py            — online/offline runner, render_as_batch for SQLite
├── _recover_orphaned_migrations() in app.py — startup orphan marking
└── DB write paths in api/migrations.py, api/tasks.py (checkpoint writer, GEPA callback)

ON THE FENCE (integration against real local instance)
├── SQLite WAL mode — tested implicitly via get_engine(), never directly asserted
├── Postgres via DATABASE_URL — test fixture supports it but no CI environment yet
└── Concurrent write behavior — tested at HTTP level (10 rapid POSTs) but not at DB level

OUTSIDE (we mock/stub)
├── SQLAlchemy Engine internals — we use it, don't test it
├── SQLModel ORM behavior — we trust column mapping works
├── Alembic runtime (ScriptDirectory, MigrationContext) — tested via command interface
├── LiteLLM / DSPy — completely outside this boundary
├── Redis / RQ — outside, TaskWorker tested separately
└── FastAPI dependency injection — tested via TestClient override pattern
```

**External dependencies:**
- SQLAlchemy 2.x (Engine, Inspector, dialect detection)
- SQLModel (table definitions, Session, metadata)
- SQLite (via sqlite3 driver, WAL mode, check_same_thread=False)
- PostgreSQL (via psycopg2/asyncpg when DATABASE_URL set)
- Alembic (migration framework, Config, command.upgrade/downgrade/check)

---

## 2. Current Coverage Audit

### `tests/test_alembic.py` — 8 tests

| Test | Covers | Misses | Notes |
|------|--------|--------|-------|
| `test_alembic_upgrade_head_on_fresh_db` | Table creation via Alembic on empty DB | Column-level schema correctness (only checks table names) | Solid |
| `test_alembic_upgrade_is_idempotent` | Running upgrade twice doesn't break | Doesn't verify column count/types preserved | Solid |
| `test_schema_parity_create_all_vs_alembic` | **Table-level** parity between create_all and Alembic | **Column-level parity completely missing** — two DBs could have same table names but different columns/types/indexes. This is a false-confidence test. | **CRITICAL GAP** |
| `test_alembic_downgrade_and_upgrade_roundtrip` | Round-trip survives at table level | Gracefully skips if downgrade fails (via `pytest.skip`), masking real bugs | **Brittle**: skip-on-failure hides broken downgrades |
| `test_migrations_table_exists_after_upgrade` | alembic_version row matches computed head | N/A | Solid |
| `test_observability_columns_exist_after_upgrade` | failure_reason, gepa_iterations columns exist | Only checks specific subset of columns | Solid |
| `test_new_migrations_columns_exist_after_upgrade` | 8 production-readiness columns exist | Doesn't check column types or defaults | Solid |
| `test_no_pending_migrations` | No unapplied revisions | Falls back to manual check on older Alembic | Solid |

**Key miss:** The parity test (`test_schema_parity_create_all_vs_alembic`) only compares table name sets. It does NOT compare columns, types, indexes, constraints, or defaults. This means a column added to `models.py` but missing from an Alembic migration would pass this test silently. This is the single biggest false-confidence test in the DB boundary.

### `tests/test_server/conftest.py` — Shared fixtures

The `engine` fixture creates an in-memory SQLite with `StaticPool`, uses `create_all` to set up tables, and tears down with `drop_all`. This bypasses Alembic entirely — tests that use this fixture validate ORM behavior against `create_all` schema, not the Alembic-migrated schema. This is correct for unit/integration tests but creates a gap: **no test proves that the Alembic-migrated schema works with the API endpoints**.

**Multiple test files redefine their own `engine` fixture** (test_negative_stress.py, test_gepa_iterations.py, test_task_worker.py, test_api_isolation.py). These are all identical in-memory SQLite setups. Not a bug, but creates fixture sprawl and makes shared changes harder.

### `tests/test_server/test_api_migrations.py` — ~15 tests

Covers: list pagination, detail retrieval, create via JSON API, test case listing/filtering, config field stripping (lm_extra_kwargs). All use in-memory SQLite via create_all.

Misses:
- No test for DELETE endpoint (if one exists)
- No test for concurrent writes to the same migration record
- No test for migration status transitions at the DB level (pending→running→complete)
- No test for what happens when config_json or per_type_scores_json contain invalid JSON

### `tests/test_server/test_negative_stress.py` — ~15 tests

Covers: file upload abuse, rapid submissions, orphan recovery, SQL injection attempts, XSS attempts, state machine violations.

Relevant DB coverage:
- `test_startup_recovery_marks_running_as_failed` — tests `_recover_orphaned_migrations()` marking running→failed
- `test_ten_rapid_submissions_all_queued` — verifies 10 records created without collision
- `test_multiple_rapid_uploads` — verifies 5 records unique

Misses:
- Uses mocked task_worker, so actual DB queue behavior not tested here
- No concurrent writer test (threading)

### `tests/test_server/test_checkpointing.py` — 9 tests

Covers: checkpoint writer saves/overwrites stage+data, missing record doesn't raise, resume endpoint status checks. Good coverage of checkpoint DB writes.

Misses:
- No test for checkpoint data exceeding TEXT column limits
- No test for concurrent checkpoint writes

### `tests/test_server/test_gepa_iterations.py` — 7 tests

Covers: write+read GEPAIterationRecord, multiple iterations ordered, callback persistence, callback swallows DB errors, API endpoint returns sorted list.

**Redefines its own engine/session/client/sample_migration fixtures** that duplicate conftest.py. Not harmful but creates maintenance burden.

### `tests/test_server/test_task_worker.py` — 8 tests

Covers: TaskQueue lifecycle (queued→running→complete/failed), restart recovery (running→queued), retry logic, completed_at timestamp.

Good DB coverage for task queue table specifically.

### `tests/test_server/test_dataset_runs.py` — 7 tests

Covers: DatasetGenerationRun CRUD, cost recalculation, list ordering.

### `tests/test_server/test_api_isolation.py` — 10 tests

Covers: owner_id filtering on migrations and pipelines in multi-user mode. Tests the owner_id column functionality.

### Other test files with incidental DB coverage

- `test_api_annotations.py`, `test_api_approvals.py`, `test_api_versioning.py`, `test_api_ab_testing.py`, `test_api_teams.py`, `test_api_users.py` — each tests CRUD for their respective tables via API. These validate model definitions work but don't test migration correctness.
- `test_api_comparisons.py` — tests distributions and diff endpoints, writes TestCaseRecord data.

---

## 3. Risk Ranking

| # | Risk | L x B | Rating | Existing Tests Catch It? |
|---|------|-------|--------|-------------------------|
| 1 | **Schema drift: models.py adds column, Alembic migration missing** | HIGH x HIGH | **CRITICAL** | **NO.** Parity test only compares table names, not columns. A new column in models.py but absent from Alembic would pass all tests. Production Alembic upgrade would leave the column missing, causing 500s on write. |
| 2 | **_migrate_add_columns SQL injection / type mismatch** | MEDIUM x HIGH | **HIGH** | **NO.** The 9 column additions in `_migrate_add_columns` use f-string SQL with hardcoded values (not user input, so no injection risk from external actors), but the column types (REAL, INTEGER, TEXT) are string literals that could mismatch what SQLModel/Alembic declares (Float vs REAL). No test validates type consistency. |
| 3 | **Global engine singleton leak across tests** | HIGH x MEDIUM | **HIGH** | **PARTIAL.** `test_alembic.py` has `_reset_engine` autouse fixture. `test_server/` tests override `get_session` but don't call `reset_engine`. In-memory SQLite with StaticPool avoids the singleton, but any test that calls `get_engine()` directly would poison the global. |
| 4 | **SQLite WAL + concurrent writes: database locked** | MEDIUM x MEDIUM | **MEDIUM** | **NO.** WAL mode is enabled in `get_engine()` but never tested under concurrent load. The 30-second timeout is set but never stressed. A real deployment with multiple workers writing simultaneously could hit "database is locked" errors. |
| 5 | **Alembic downgrade data loss** | MEDIUM x HIGH | **HIGH** | **PARTIAL.** Round-trip test exists but `pytest.skip`s if downgrade fails, and it only checks table presence, not data preservation. A downgrade that drops data silently would pass. |
| 6 | **`_migrate_add_columns` dataset_generation_runs table diverges from model** | MEDIUM x MEDIUM | **MEDIUM** | **NO.** The raw-SQL CREATE TABLE in `_migrate_add_columns` for Postgres has `SERIAL PRIMARY KEY` and `TIMESTAMPTZ` while the SQLModel uses `Integer` + `DateTime`. These type differences are untested. The SQLite version has `INTEGER PRIMARY KEY AUTOINCREMENT` vs SQLModel's default `INTEGER`. |
| 7 | **Merge migration (fc47a11b96ba) creates dual-parent head** | LOW x HIGH | **MEDIUM** | **YES.** The `test_migrations_table_exists_after_upgrade` dynamically discovers the head, so it handles the merge correctly. But no test verifies the merge doesn't break the dependency chain. |
| 8 | **Foreign key enforcement varies by database** | MEDIUM x MEDIUM | **MEDIUM** | **NO.** SQLite has FKs disabled by default (requires `PRAGMA foreign_keys=ON`). The codebase never enables this pragma. An orphaned test_case pointing to a deleted migration would be allowed in SQLite but rejected in Postgres. |
| 9 | **JSON column corruption (invalid JSON in _json fields)** | LOW x MEDIUM | **LOW** | **NO.** All `_json` fields are plain TEXT/VARCHAR. Nothing prevents storing `"not-valid-json"`. The API deserializes with `json.loads` and would 500 on corrupt data. |
| 10 | **`reset_engine()` called in production** | LOW x HIGH | **MEDIUM** | **NO.** `reset_engine()` disposes the engine and sets `_engine = None`. If called in production (e.g., via import side-effect), all subsequent sessions would create a new engine. No guard against production use. |

---

## 4. Test Plan by Tier

### Tier 1: Unit — Pure Logic, Model Validation

| Test ID | Description | Assertions | Write Time | Status |
|---------|-------------|------------|------------|--------|
| U1 | MigrationRecord default values | status="pending", cost_usd=0.0, total_tokens=0, all _json fields have valid JSON defaults | S | MISSING |
| U2 | All 21 model classes instantiate without error | Construct each with minimal required fields, verify no exception | S | MISSING |
| U3 | `_is_postgres()` returns correct dialect flag | Mock engine with dialect.name="postgresql" vs "sqlite" | S | MISSING |
| U4 | `_get_db_path()` respects ROSETTASTONE_DB_PATH env | Set env, verify returned path; unset env, verify default ~/.rosettastone/migrations.db | S | MISSING |
| U5 | Foreign key fields have correct references | Inspect model metadata for FK targets (test_cases.migration_id → migrations.id, etc.) | S | MISSING |
| U6 | UniqueConstraint on Approval(workflow_id, user_id) present in model metadata | Inspect __table_args__ | S | MISSING |
| U7 | `_format_recommendation()` maps all DB values correctly | Test GO, NO_GO, CONDITIONAL, None | S | EXISTS (in test_api_migrations.py UI tests) |

### Tier 2: Contract — Schema Parity

| Test ID | Description | Assertions | Write Time | Status |
|---------|-------------|------------|------------|--------|
| C1 | **Column-level parity: create_all vs Alembic** | For each table: compare column names, types, nullable, defaults between create_all DB and Alembic DB | M | **MISSING — HIGHEST PRIORITY** |
| C2 | **Index parity: create_all vs Alembic** | Compare index names and columns on both schemas | M | MISSING |
| C3 | **Foreign key parity** | Compare FK constraints between both schemas | M | MISSING |
| C4 | **Unique constraint parity** | Verify approvals(workflow_id, user_id) unique constraint exists in both paths | S | MISSING |
| C5 | `_migrate_add_columns` column types match model types | For each of the 9 columns: verify the SQL type string (REAL, INTEGER, TEXT) maps to the SQLModel Field type | M | MISSING |
| C6 | `_migrate_add_columns` dataset_generation_runs CREATE TABLE matches model | Compare columns in raw SQL against DatasetGenerationRun model fields | M | MISSING |

### Tier 3: Integration — Real DB

| Test ID | Description | Assertions | Write Time | Status |
|---------|-------------|------------|------------|--------|
| I1 | **SQLite WAL mode enabled after get_engine()** | Create engine pointing to tmp file, query `PRAGMA journal_mode`, assert "wal" | S | MISSING |
| I2 | **Alembic upgrade on pre-existing DB with missing columns** | Create DB with initial schema only, run upgrade head, verify new columns present | M | MISSING |
| I3 | **Alembic upgrade preserves existing data** | Insert rows into initial-schema DB, run upgrade head, verify rows still present with correct values | L | MISSING |
| I4 | **Alembic downgrade then upgrade preserves data structure** | Upgrade head, insert data, downgrade to intermediate rev, upgrade head again, verify schema correct | L | PARTIAL (exists but doesn't check data) |
| I5 | **init_db() on fresh SQLite creates all tables + adds safety-net columns** | Call init_db() on empty path, inspect tables and columns | M | MISSING |
| I6 | **init_db() on existing DB is idempotent** | Call init_db() twice on same DB file, verify no errors, schema unchanged | S | MISSING |
| I7 | **Postgres integration: Alembic upgrade head** | Docker Postgres, run upgrade head, verify tables/columns | L | MISSING |
| I8 | **Postgres integration: _migrate_add_columns uses IF NOT EXISTS** | Run _migrate_add_columns twice on Postgres, no error | M | MISSING |
| I9 | **get_engine() singleton returns same object on repeated calls** | Call get_engine() twice, assert `is` identity | S | MISSING |
| I10 | **reset_engine() followed by get_engine() creates new engine** | Call reset_engine(), then get_engine(), assert new engine | S | MISSING |
| I11 | **_recover_orphaned_migrations marks all running records as failed** | Insert 3 running + 2 complete + 1 pending, call recovery, assert only running→failed | S | EXISTS (in test_negative_stress.py) |
| I12 | **Session generator yields and closes** | Call get_session(), verify it yields a Session, verify engine not disposed after | S | MISSING |
| I13 | **SQLite concurrent read during write (WAL)** | Two threads: one writing, one reading; verify reader gets consistent snapshot | L | MISSING |
| I14 | **SQLite file-based: engine respects ROSETTASTONE_DB_PATH** | Set env to tmp path, call get_engine, verify file created | S | MISSING |

### Tier 4: Property-Based

| Test ID | Description | Assertions | Write Time | Status |
|---------|-------------|------------|------------|--------|
| P1 | Model field constraints: MigrationRecord status only accepts known values | Hypothesis: generate arbitrary strings for status, verify DB accepts (no constraint at DB level, only app-level) | M | MISSING |
| P2 | `_migrate_add_columns` idempotent under repeated application | Run _migrate_add_columns N times on same DB, schema identical each time | S | MISSING |
| P3 | Round-trip: write model → read model preserves all field values | Hypothesis: generate random valid field values, write MigrationRecord, read back, assert equality | M | MISSING |
| P4 | JSON field round-trip: arbitrary valid JSON survives write→read | Hypothesis: generate random JSON structures, store in config_json, read back, assert json.loads round-trips | M | MISSING |

### Tier 5: End-to-End — Full Server Lifecycle

| Test ID | Description | Assertions | Write Time | Status |
|---------|-------------|------------|------------|--------|
| E1 | Create migration via API → verify record in DB → retrieve via API | POST, check 201, GET detail, verify fields match | S | EXISTS (test_api_migrations.py) |
| E2 | Migration lifecycle: pending → running → complete, all DB writes correct | Create via API, mock background task, verify status transitions, verify test_cases written | L | MISSING |
| E3 | Full server startup with file-based SQLite | Start server (not in-memory), verify DB file created, tables exist, WAL enabled | M | MISSING |
| E4 | Server restart recovery: orphaned migrations + stale tasks | Insert running migration + running task, restart (lifespan), verify both recovered | M | PARTIAL (orphan recovery tested, stale task recovery tested separately, not together) |

---

## 5. Synthetic Data Generation Strategy

### Realistic DB states needed:

1. **Empty DB** — fresh install, no tables. Already available via tmp_path in test_alembic.py.

2. **Populated DB with complete migration history** — 5-10 migrations with test_cases, warnings, GEPA iterations, versions, annotations.
   - **Generator:** Factory function `make_full_migration(session, overrides={})` that creates a MigrationRecord + 20 TestCaseRecords + 3 WarningRecords + 5 GEPAIterationRecords + 1 MigrationVersion.
   - Use deterministic IDs and timestamps for reproducibility.

3. **Orphaned records** — TestCaseRecords pointing to deleted migration_id. Requires inserting records then deleting the parent without cascade.
   - **Generator:** `make_orphaned_test_cases(session, count=5)` — insert migration, add test_cases, raw-delete migration row.
   - **Caveat:** Only testable on SQLite since FK enforcement is off by default. On Postgres, this requires disabling FK checks or using `SET CONSTRAINTS ... DEFERRED`.

4. **Partial migration (mid-run crash)** — MigrationRecord with status="running", checkpoint_stage set, some test_cases written.
   - **Generator:** `make_crashed_migration(session, stage="baseline_eval")`.

5. **Pre-Alembic DB (safety-net migration target)** — DB created by old create_all() missing the 9 safety-net columns and gepa_iterations/dataset_generation_runs tables.
   - **Generator:** Create DB with initial schema Alembic revision only, then test _migrate_add_columns against it.

6. **Concurrent write scenario** — Two sessions writing to the same table simultaneously.
   - **Generator:** threading.Thread + barrier pattern.

### Stability / rot prevention:
- Tie factory functions to model imports — if a required field is added to a model, the factory breaks at compile time.
- Pin test data shapes to EXPECTED_APP_TABLES and a parallel EXPECTED_COLUMNS dict.
- No external data files — everything generated in-process.

### Cost profile:
- All SQLite-based synthetic data is free (in-memory or tmp_path).
- Postgres tests require Docker, ~5s container startup per test session.

---

## 6. Fixtures, Fakes, and Mocks

### New fixtures needed:

| Fixture | Scope | Description | Cross-Area? |
|---------|-------|-------------|-------------|
| `full_migration_factory` | function | Creates MigrationRecord + children, returns record | Yes — shared with **server-http-security** (they need populated migrations for auth tests) |
| `alembic_migrated_engine` | function | Runs Alembic upgrade head on tmp SQLite, returns engine | No — specific to this boundary |
| `create_all_engine` | function | Runs create_all on tmp SQLite, returns engine | Already exists as `engine` in conftest.py |
| `postgres_engine` | session | Docker Postgres engine (skip if not available) | Yes — shared with **background-orchestration** for task queue tests |
| `pre_alembic_db` | function | DB with initial schema only, missing later columns | No |
| `orphaned_records_db` | function | DB with orphaned child records | No |
| `crashed_migration` | function | MigrationRecord in running state with checkpoint | Exists in test_checkpointing.py but inline |

### Cross-subagent shared fixtures:

1. **`engine` fixture** — Currently defined identically in 5+ test files. Should be consolidated into `tests/test_server/conftest.py` (already exists there). The **server-http-security** boundary uses the same pattern. **Action:** Both areas should use the shared conftest.py fixture; duplicates in test_gepa_iterations.py, test_negative_stress.py, test_task_worker.py, test_api_isolation.py should be removed.

2. **`client` fixture** — Same duplication pattern. The override pattern (dependency_overrides[get_session]) is identical across files.

3. **`sample_migration` fixture** — Defined in conftest.py and redefined in test_gepa_iterations.py (slightly different: no cost/score fields). Should be parameterized or merged.

4. **TaskWorker engine** — The **background-orchestration** boundary defines its own engine fixture in test_task_worker.py. If Postgres CI is added, both areas need the same conditional-database-url logic.

### Mocks:

| Mock | Purpose | Notes |
|------|---------|-------|
| `MagicMock(spec=Engine)` | Test _is_postgres with mock dialect | Simple, no cross-area concern |
| `monkeypatch.setenv("DATABASE_URL", ...)` | Force Postgres path in get_engine | Needs real Postgres or mock; affects singleton |
| `patch("rosettastone.server.app.get_engine")` | Isolate _recover_orphaned_migrations | Already used in test_negative_stress.py |

---

## 7. Gaps You Can't Close

1. **Production Postgres behavior without Postgres CI.** We can test against Docker Postgres locally, but without a CI Postgres service, we can't guarantee parity in the pipeline. The `tests/test_server/conftest.py` engine fixture already has the `DATABASE_URL` conditional, but no CI config enables it. **NEEDS_HUMAN_REVIEW:** Is there a CI plan for Postgres? Without it, Postgres-specific bugs (type coercion, TIMESTAMPTZ vs DATETIME, SERIAL vs INTEGER) ship untested.

2. **Alembic autogenerate drift detection.** There's no automated check that `alembic revision --autogenerate` produces an empty migration (meaning models.py and the Alembic chain are in sync). This is the standard way to detect column-level schema drift. We can write a test for it, but it requires a running DB context with the full migration chain applied, then comparing against models — doable but fragile across Alembic versions.

3. **WAL mode under real concurrent load.** SQLite WAL behavior under genuine multi-process (not multi-thread) concurrency can't be tested in-process. A real multi-process test would require subprocess spawning. The 30-second timeout is the safety valve, but we can't test that it fires correctly without a real lock contention scenario.

4. **Long-running migration data preservation.** A real migration that runs for hours, writing test_cases incrementally, can't be simulated without the full pipeline. We can only test the DB write paths in isolation.

5. **`_migrate_add_columns` on a truly ancient DB.** The safety-net code handles DBs created before Alembic was added. We can simulate this, but the exact historical schema is undocumented — we'd be guessing which columns were present in "v0."

---

## 8. Cost and Time Estimate

| Tier | Tests to Write | Effort | Calendar Time (1 dev) |
|------|---------------|--------|----------------------|
| Unit (U1-U6) | 6 | S | 0.5 day |
| Contract (C1-C6) | 6 | M | 1.5 days |
| Integration - SQLite (I1-I6, I9-I12, I14) | 11 | M | 2 days |
| Integration - Postgres (I7-I8) | 2 | L | 1 day (includes Docker setup) |
| Integration - Concurrency (I13) | 1 | L | 0.5 day |
| Property-based (P1-P4) | 4 | M | 1 day |
| E2E (E2-E4) | 3 | L-XL | 1.5 days |
| Fixture consolidation | — | M | 0.5 day |
| **Total** | **33 new tests** | | **~8 days** |

The C1 test (column-level schema parity) alone provides ~40% of the risk reduction. If time is constrained, do C1 first.

---

## 9. Path to Production

### Current readiness level: **Development / Alpha**

The DB layer works for single-user SQLite with create_all(). Alembic migrations exist and pass table-level tests. No column-level schema parity test. No Postgres CI. No concurrent-write testing. The safety-net migration code (`_migrate_add_columns`) duplicates logic that should be in Alembic only.

### Gap to production-hardened:

| Milestone | What's Needed | Depends On |
|-----------|---------------|------------|
| **P0.1: Column-level schema parity test** | Write C1 test, fix any drift found | Nothing |
| **P0.2: Consolidate fixture sprawl** | Deduplicate engine/client/sample_migration across 5 test files | Nothing |
| **P0.3: Foreign key enforcement test** | Add `PRAGMA foreign_keys=ON` test and decide if app should enable it | NEEDS_HUMAN_REVIEW: Enabling FK enforcement in SQLite may break existing code that relies on lenient FK behavior |
| **P0.5: owner_id + multi-user DB isolation** | Verified by test_api_isolation.py, but no DB-level test that queries can't leak across owners without the API layer | server-http-security boundary |
| **P1.0: Remove _migrate_add_columns** | Once all deployments are on Alembic, remove the safety-net code — it's a second source of truth that can drift | Requires migration documentation for operators |
| **P1.2: Postgres CI** | Add Docker Postgres to CI, run test suite with DATABASE_URL | CI infrastructure decision |
| **P1.5: Concurrent write testing** | Write I13 (threaded SQLite test) and Postgres equivalent | P1.2 for Postgres version |

### Gates:
1. **C1 passes** (column-level parity) — prerequisite for any production deployment
2. **Alembic upgrade roundtrip preserves data** (I3) — prerequisite for production upgrades
3. **Postgres CI green** — prerequisite for Postgres production use

### Ordered sequence (smallest next slice first):
1. Write C1 (column-level parity test). ~2 hours. Highest ROI.
2. Write U1-U6 (model defaults and FK validation). ~3 hours.
3. Write I1 (WAL mode assertion). ~30 minutes.
4. Write I3 (data preservation across Alembic upgrade). ~2 hours.
5. Consolidate fixtures. ~3 hours.
6. Write I5-I6 (init_db idempotency). ~1 hour.
7. Add Postgres CI. ~4 hours (infra work).
8. Write C5-C6 (_migrate_add_columns type validation). ~2 hours.
9. Write P3-P4 (property-based round-trip). ~3 hours.

### Dependencies on other boundaries:
- **server-http-security:** Shares engine/client fixtures. Fixture consolidation should be coordinated.
- **background-orchestration:** TaskWorker and task dispatch tests write to task_queue table. If Postgres CI is added, both need the same engine fixture.
- **ui-template-rendering:** UI endpoints read from DB but don't write (except form-based migration creation). No fixture dependency.

### NEEDS_HUMAN_REVIEW items:
1. Should `PRAGMA foreign_keys=ON` be enabled in production SQLite? Currently off. Enabling it is a breaking change if orphaned records exist.
2. Is there a plan for Postgres CI? The conftest.py fixture supports it, but no CI workflow enables it.
3. Should `_migrate_add_columns` be deprecated now that Alembic covers all schema changes? It's a second source of truth.
4. The `dataset_generation_runs` raw-SQL CREATE TABLE in `_migrate_add_columns` uses Postgres-specific types (SERIAL, TIMESTAMPTZ) but the same table is created by Alembic (b7c8d9e0f1a2) with standard SA types. Which path runs first in production? If Alembic runs first, the raw SQL is dead code. If not, the types diverge.
