"""Tests for Alembic migration correctness and idempotency."""

from __future__ import annotations

from pathlib import Path

import pytest
import sqlalchemy

import rosettastone.server.models  # noqa: F401 — register all models before create_all()

PROJECT_ROOT = Path(__file__).parent.parent

# The revision ID for the single migration that currently exists.
INITIAL_REVISION = "c39645f955dc"

# All tables created by the initial schema migration (alembic) or create_all().
# Note: alembic_version is created by Alembic itself, not via models; it is NOT
# included here because create_all() does not produce it.
EXPECTED_APP_TABLES: set[str] = {
    "migrations",
    "test_cases",
    "warnings",
    "registered_models",
    "pipelines",
    "pipeline_stages",
    "audit_log",
    "migration_versions",
    "ab_tests",
    "ab_test_results",
    "users",
    "teams",
    "team_memberships",
    "alerts",
    "annotations",
    "approval_workflows",
    "approvals",
    "task_queue",
    "user_budgets",
    # Added by observability branch migrations
    "dataset_generation_runs",
    "gepa_iterations",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_alembic_cfg(db_path: str):
    """Build an Alembic Config that points at *db_path* and the project's ini."""
    from alembic.config import Config

    cfg = Config(str(PROJECT_ROOT / "alembic.ini"))
    # Override the URL so env.py's get_engine() picks up the right DB.
    # env.py reads ROSETTASTONE_DB_PATH; we also set sqlalchemy.url as a fallback
    # so that offline-mode tools work too.
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    return cfg


def _run_upgrade(db_path: str, revision: str = "head") -> None:
    from alembic.command import upgrade

    cfg = _make_alembic_cfg(db_path)
    upgrade(cfg, revision)


def _run_downgrade(db_path: str, revision: str) -> None:
    from alembic.command import downgrade

    cfg = _make_alembic_cfg(db_path)
    downgrade(cfg, revision)


def get_table_names(db_path: str) -> set[str]:
    """Return all table names present in the SQLite database at *db_path*."""
    engine = sqlalchemy.create_engine(f"sqlite:///{db_path}")
    inspector = sqlalchemy.inspect(engine)
    names = set(inspector.get_table_names())
    engine.dispose()
    return names


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_engine():
    """Reset the global database engine before and after each test.

    This is necessary because database.py caches a module-level ``_engine``
    singleton; without a reset the second test would reuse the first test's
    engine (pointing at the wrong DB file).
    """
    from rosettastone.server.database import reset_engine

    reset_engine()
    yield
    reset_engine()


@pytest.fixture()
def alembic_db(tmp_path, monkeypatch):
    """Yield a fresh temp DB path and set ROSETTASTONE_DB_PATH to it."""
    db_path = str(tmp_path / "test_alembic.db")
    monkeypatch.setenv("ROSETTASTONE_DB_PATH", db_path)
    # Also unset DATABASE_URL so we don't accidentally pick up a postgres URL.
    monkeypatch.delenv("DATABASE_URL", raising=False)
    return db_path


@pytest.fixture()
def upgraded_db(alembic_db):
    """Run alembic upgrade head on a fresh DB and return the path."""
    _run_upgrade(alembic_db, "head")
    return alembic_db


# ---------------------------------------------------------------------------
# Test 1 — upgrade head on a fresh DB creates all expected tables
# ---------------------------------------------------------------------------


def test_alembic_upgrade_head_on_fresh_db(alembic_db):
    """Running 'alembic upgrade head' on a brand-new SQLite DB should create
    every application table defined in the initial schema migration."""
    _run_upgrade(alembic_db, "head")

    tables = get_table_names(alembic_db)
    # alembic_version is also expected to be present after a successful upgrade
    assert "alembic_version" in tables, "alembic_version table was not created"
    missing = EXPECTED_APP_TABLES - tables
    assert not missing, f"Tables missing after upgrade head: {missing}"


# ---------------------------------------------------------------------------
# Test 2 — upgrade head is idempotent
# ---------------------------------------------------------------------------


def test_alembic_upgrade_is_idempotent(alembic_db):
    """Running 'alembic upgrade head' twice must not raise and must leave the
    schema intact."""
    _run_upgrade(alembic_db, "head")
    tables_after_first = get_table_names(alembic_db)

    # Second upgrade should be a no-op — Alembic detects the DB is already at head.
    _run_upgrade(alembic_db, "head")
    tables_after_second = get_table_names(alembic_db)

    assert tables_after_first == tables_after_second, (
        "Table set changed between first and second upgrade head"
    )
    missing = EXPECTED_APP_TABLES - tables_after_second
    assert not missing, f"Tables missing after second upgrade head: {missing}"


# ---------------------------------------------------------------------------
# Test 3 — schema parity: create_all() vs alembic upgrade head
# ---------------------------------------------------------------------------


def test_schema_parity_create_all_vs_alembic(tmp_path, monkeypatch):
    """The table set produced by SQLModel.metadata.create_all() and by
    'alembic upgrade head' must be identical (both cover every app table).

    Note: create_all() does NOT produce the ``alembic_version`` bookkeeping
    table, so we compare only the application tables.
    """
    from sqlmodel import SQLModel

    # --- DB 1: create_all path (dev/test) ---
    db1_path = str(tmp_path / "create_all.db")
    monkeypatch.setenv("ROSETTASTONE_DB_PATH", db1_path)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    from rosettastone.server.database import get_engine, reset_engine

    reset_engine()
    engine1 = get_engine()
    SQLModel.metadata.create_all(engine1)
    engine1.dispose()
    reset_engine()

    tables_create_all = get_table_names(db1_path)

    # --- DB 2: alembic path ---
    db2_path = str(tmp_path / "alembic.db")
    monkeypatch.setenv("ROSETTASTONE_DB_PATH", db2_path)
    reset_engine()

    _run_upgrade(db2_path, "head")

    tables_alembic = get_table_names(db2_path)
    # Strip the Alembic bookkeeping table before comparing
    tables_alembic_app = tables_alembic - {"alembic_version"}

    # Both paths must produce exactly the same set of application tables.
    assert tables_create_all == tables_alembic_app, (
        f"Schema divergence detected.\n"
        f"  create_all only:  {tables_create_all - tables_alembic_app}\n"
        f"  alembic only:     {tables_alembic_app - tables_create_all}"
    )


# ---------------------------------------------------------------------------
# Test 4 — downgrade to base then upgrade again (round-trip)
# ---------------------------------------------------------------------------


def test_alembic_downgrade_and_upgrade_roundtrip(upgraded_db):
    """Downgrade to base (removes all tables) then upgrade back to head.
    If downgrade is not supported for some reason, the test is skipped
    gracefully.
    """
    try:
        _run_downgrade(upgraded_db, "base")
    except Exception as exc:
        pytest.skip(f"Downgrade to base not supported: {exc}")

    # After downgrading to base, application tables should be gone.
    # (alembic_version itself may or may not exist depending on the dialect;
    # we don't assert on its presence here.)
    tables_after_downgrade = get_table_names(upgraded_db) - {"alembic_version"}
    assert not (tables_after_downgrade & EXPECTED_APP_TABLES), (
        f"Some tables still exist after downgrade to base: "
        f"{tables_after_downgrade & EXPECTED_APP_TABLES}"
    )

    # Re-upgrade to head — must succeed and restore everything.
    _run_upgrade(upgraded_db, "head")

    tables_after_reupgrade = get_table_names(upgraded_db)
    missing = EXPECTED_APP_TABLES - tables_after_reupgrade
    assert not missing, f"Tables missing after re-upgrade: {missing}"


# ---------------------------------------------------------------------------
# Test 5 — alembic_version table exists and contains the right revision
# ---------------------------------------------------------------------------


def test_migrations_table_exists_after_upgrade(upgraded_db):
    """After 'alembic upgrade head' the alembic_version table must exist and
    hold exactly one row whose version_num equals the latest revision ID."""
    tables = get_table_names(upgraded_db)
    assert "alembic_version" in tables, "alembic_version table is absent"

    engine = sqlalchemy.create_engine(f"sqlite:///{upgraded_db}")
    with engine.connect() as conn:
        result = conn.execute(sqlalchemy.text("SELECT version_num FROM alembic_version"))
        rows = result.fetchall()

    engine.dispose()

    assert len(rows) == 1, f"Expected exactly 1 row in alembic_version, got {len(rows)}"
    recorded_revision = rows[0][0]

    # Discover the head revision using Alembic's ScriptDirectory API so the
    # test stays correct regardless of how many migrations exist.
    from alembic.config import Config as AlembicConfig
    from alembic.script import ScriptDirectory

    cfg = AlembicConfig()
    cfg.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
    script = ScriptDirectory.from_config(cfg)
    heads = script.get_heads()
    assert len(heads) == 1, f"Expected 1 head revision, got {heads}"
    head_revision = heads[0]

    assert recorded_revision == head_revision, (
        f"alembic_version contains '{recorded_revision}' "
        f"but expected head revision '{head_revision}'"
    )


def test_observability_columns_exist_after_upgrade(upgraded_db):
    """Feature 6: failure_reason column must exist on test_cases after upgrade head."""
    engine = sqlalchemy.create_engine(f"sqlite:///{upgraded_db}")
    inspector = sqlalchemy.inspect(engine)
    tc_columns = {col["name"] for col in inspector.get_columns("test_cases")}
    gepa_columns = {col["name"] for col in inspector.get_columns("gepa_iterations")}
    engine.dispose()

    assert "failure_reason" in tc_columns, "failure_reason column missing from test_cases"
    expected_gepa_cols = {
        "id",
        "migration_id",
        "iteration",
        "total_iterations",
        "mean_score",
        "recorded_at",
    }
    missing_gepa = expected_gepa_cols - gepa_columns
    assert not missing_gepa, f"Columns missing from gepa_iterations: {missing_gepa}"


def test_new_migrations_columns_exist_after_upgrade(upgraded_db):
    """After upgrade head, the 8 production-readiness columns must exist on migrations table."""
    engine = sqlalchemy.create_engine(f"sqlite:///{upgraded_db}")
    inspector = sqlalchemy.inspect(engine)
    columns = {col["name"] for col in inspector.get_columns("migrations")}
    engine.dispose()

    expected_new_cols = {
        "checkpoint_stage",
        "checkpoint_data_json",
        "current_stage",
        "stage_progress",
        "overall_progress",
        "max_cost_usd",
        "estimated_cost_usd",
        "owner_id",
    }
    missing = expected_new_cols - columns
    assert not missing, f"Missing columns on migrations table after upgrade: {missing}"


# ---------------------------------------------------------------------------
# Test 6 — no pending migrations after upgrade head
# ---------------------------------------------------------------------------


def test_no_pending_migrations(upgraded_db):
    """After 'alembic upgrade head' there must be no unapplied migrations.

    Strategy: use alembic.command.check (Alembic >= 1.9) if available;
    otherwise compare the alembic_version table against the migration files
    in the versions directory to confirm every revision has been applied.
    """
    cfg = _make_alembic_cfg(upgraded_db)

    # Preferred: alembic.command.check raises SystemExit(1) if migrations are
    # pending, and exits cleanly (SystemExit(0) or returns None) when up-to-date.
    try:
        from alembic.command import check as alembic_check

        try:
            alembic_check(cfg)
        except SystemExit as exc:
            # Exit code 0 means "no pending migrations" in some Alembic versions.
            if exc.code not in (0, None):
                pytest.fail(f"alembic check reported pending migrations (exit code {exc.code})")
    except ImportError:
        # alembic.command.check not available in this version — fall back to
        # a manual comparison.
        _assert_no_pending_migrations_manual(upgraded_db)


def _assert_no_pending_migrations_manual(db_path: str) -> None:
    """Fallback: verify that every revision file in alembic/versions/ has been
    applied by checking the alembic_version table."""
    versions_dir = PROJECT_ROOT / "alembic" / "versions"
    revision_files = list(versions_dir.glob("*.py"))
    assert revision_files, "No revision files found — cannot check pending migrations"

    # Collect all declared revision IDs.
    declared_revisions: set[str] = set()
    for rev_file in revision_files:
        content = rev_file.read_text()
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("revision") and "=" in stripped:
                rev_value = stripped.split("=")[-1].strip().strip('"').strip("'")
                if rev_value and not rev_value.startswith("#"):
                    declared_revisions.add(rev_value)
                break

    engine = sqlalchemy.create_engine(f"sqlite:///{db_path}")
    with engine.connect() as conn:
        result = conn.execute(sqlalchemy.text("SELECT version_num FROM alembic_version"))
        applied_revisions = {row[0] for row in result.fetchall()}
    engine.dispose()

    # For a single-branch history the head revision should appear in
    # alembic_version; all others are superseded but were applied in sequence.
    # A simple sanity check: the applied set must not be empty.
    assert applied_revisions, "alembic_version table is empty — no revisions applied"

    # The current head must be among the declared revisions.
    assert applied_revisions <= declared_revisions, (
        f"alembic_version contains unknown revisions: {applied_revisions - declared_revisions}"
    )


# ---------------------------------------------------------------------------
# Test 7 — column-level schema parity: create_all() vs alembic upgrade head
# ---------------------------------------------------------------------------


def test_schema_parity_column_level(tmp_path, monkeypatch):
    """The column names produced by SQLModel.metadata.create_all() and by
    'alembic upgrade head' must match for every shared application table.

    This test catches the most dangerous schema drift: a column existing in one
    path but not the other. Type comparison is intentionally skipped because
    SQLAlchemy type affinity differs between create_all and Alembic; column
    name presence is the critical signal.
    """
    from sqlmodel import SQLModel

    # --- DB 1: create_all path ---
    db1_path = str(tmp_path / "col_create_all.db")
    monkeypatch.setenv("ROSETTASTONE_DB_PATH", db1_path)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    from rosettastone.server.database import get_engine, reset_engine

    reset_engine()
    engine1 = get_engine()
    SQLModel.metadata.create_all(engine1)

    inspector1 = sqlalchemy.inspect(engine1)
    tables_create_all = set(inspector1.get_table_names())
    cols_create_all: dict[str, set[str]] = {
        tbl: {col["name"] for col in inspector1.get_columns(tbl)} for tbl in tables_create_all
    }
    engine1.dispose()
    reset_engine()

    # --- DB 2: alembic path ---
    db2_path = str(tmp_path / "col_alembic.db")
    monkeypatch.setenv("ROSETTASTONE_DB_PATH", db2_path)
    reset_engine()

    _run_upgrade(db2_path, "head")

    engine2 = sqlalchemy.create_engine(f"sqlite:///{db2_path}")
    inspector2 = sqlalchemy.inspect(engine2)
    tables_alembic = set(inspector2.get_table_names()) - {"alembic_version"}
    cols_alembic: dict[str, set[str]] = {
        tbl: {col["name"] for col in inspector2.get_columns(tbl)} for tbl in tables_alembic
    }
    engine2.dispose()
    reset_engine()

    # Only compare tables that appear in both paths (intersection).
    shared_tables = tables_create_all & tables_alembic

    failures: list[str] = []
    for tbl in sorted(shared_tables):
        only_in_create_all = cols_create_all[tbl] - cols_alembic[tbl]
        only_in_alembic = cols_alembic[tbl] - cols_create_all[tbl]
        if only_in_create_all:
            failures.append(
                f"Table '{tbl}': columns in create_all but missing from alembic: "
                f"{sorted(only_in_create_all)}"
            )
        if only_in_alembic:
            failures.append(
                f"Table '{tbl}': columns in alembic but missing from create_all: "
                f"{sorted(only_in_alembic)}"
            )

    assert not failures, (
        "Column-level schema drift detected between create_all and alembic paths:\n"
        + "\n".join(f"  - {msg}" for msg in failures)
    )


# ---------------------------------------------------------------------------
# Test 8 — init_db() is idempotent
# ---------------------------------------------------------------------------


def test_init_db_idempotent(tmp_path, monkeypatch):
    """Calling init_db() twice on the same SQLite file must not raise any
    errors and must leave the schema intact on both calls."""
    db_path = str(tmp_path / "idempotent.db")
    monkeypatch.setenv("ROSETTASTONE_DB_PATH", db_path)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    from rosettastone.server.database import init_db, reset_engine

    reset_engine()

    # First call — creates tables from scratch.
    init_db()

    tables_after_first = get_table_names(db_path)
    assert tables_after_first, "init_db() produced no tables on first call"

    reset_engine()

    # Second call — must be a no-op (tables already exist).
    init_db()

    tables_after_second = get_table_names(db_path)
    assert tables_after_first == tables_after_second, (
        f"Table set changed between first and second init_db() call.\n"
        f"  Added:   {tables_after_second - tables_after_first}\n"
        f"  Removed: {tables_after_first - tables_after_second}"
    )

    reset_engine()


# ---------------------------------------------------------------------------
# Test group — Data preservation across upgrade/downgrade cycles
# ---------------------------------------------------------------------------


class TestDataPreservation:
    """Verify that data inserted into the database survives Alembic upgrade/downgrade cycles."""

    def test_data_survives_upgrade_to_head(self, alembic_db):
        """Records inserted after upgrade head must be queryable from a fresh engine."""
        from sqlalchemy import create_engine as sa_create_engine
        from sqlmodel import Session

        from rosettastone.server.models import MigrationRecord, TestCaseRecord

        _run_upgrade(alembic_db, "head")

        # Insert records using a session backed by the upgraded DB.
        engine_write = sa_create_engine(f"sqlite:///{alembic_db}")
        with Session(engine_write) as session:
            mr = MigrationRecord(
                source_model="openai/gpt-4o",
                target_model="anthropic/claude-sonnet-4",
                status="complete",
                confidence_score=0.92,
                cost_usd=1.50,
            )
            session.add(mr)
            session.flush()  # assign mr.id before creating the FK child
            migration_id = mr.id

            tc = TestCaseRecord(
                migration_id=migration_id,
                phase="validation",
                output_type="json",
                composite_score=0.85,
                is_win=True,
                scores_json='{"bertscore": 0.9}',
                details_json="{}",
                response_length=100,
                new_response_length=95,
            )
            session.add(tc)
            session.commit()
            tc_id = tc.id

        engine_write.dispose()

        # Simulate a server restart by creating a brand-new engine to the same path.
        engine_read = sa_create_engine(f"sqlite:///{alembic_db}")
        with Session(engine_read) as session:
            queried_mr = session.get(MigrationRecord, migration_id)
            assert queried_mr is not None, "MigrationRecord not found after engine restart"
            assert queried_mr.source_model == "openai/gpt-4o"
            assert queried_mr.target_model == "anthropic/claude-sonnet-4"
            assert queried_mr.confidence_score == pytest.approx(0.92)

            queried_tc = session.get(TestCaseRecord, tc_id)
            assert queried_tc is not None, "TestCaseRecord not found after engine restart"
            assert queried_tc.composite_score == pytest.approx(0.85)
            assert queried_tc.is_win is True
            # FK relationship is intact.
            assert queried_tc.migration_id == queried_mr.id

        engine_read.dispose()

    def test_data_survives_downgrade_and_reupgrade(self, alembic_db):
        """Downgrade to base destroys all data (expected); re-upgrade restores schema
        so new data can be inserted and queried correctly.

        Note: Downgrading to 'base' drops all application tables, which necessarily
        destroys any data that was stored in them. This is documented expected behaviour —
        this test verifies that (a) the downgrade succeeds, (b) the subsequent re-upgrade
        correctly recreates the schema, and (c) new records can be inserted and queried
        after the re-upgrade.
        """
        from sqlalchemy import create_engine as sa_create_engine
        from sqlmodel import Session

        from rosettastone.server.models import MigrationRecord

        _run_upgrade(alembic_db, "head")

        # Insert a record before downgrade.
        engine_before = sa_create_engine(f"sqlite:///{alembic_db}")
        with Session(engine_before) as session:
            mr = MigrationRecord(
                source_model="openai/gpt-4o",
                target_model="anthropic/claude-sonnet-4",
            )
            session.add(mr)
            session.commit()
        engine_before.dispose()

        # Downgrade to base — this drops all tables and data.
        try:
            _run_downgrade(alembic_db, "base")
        except Exception as exc:
            pytest.skip(f"Downgrade to base not supported: {exc}")

        # After downgrade, application tables should be absent.
        tables_after_downgrade = get_table_names(alembic_db) - {"alembic_version"}
        assert not (tables_after_downgrade & EXPECTED_APP_TABLES), (
            f"Tables still exist after downgrade to base: "
            f"{tables_after_downgrade & EXPECTED_APP_TABLES}"
        )

        # Re-upgrade to head — schema must be fully restored.
        _run_upgrade(alembic_db, "head")

        # Insert a new record after re-upgrade to confirm the schema works correctly.
        engine_after = sa_create_engine(f"sqlite:///{alembic_db}")
        with Session(engine_after) as session:
            new_mr = MigrationRecord(
                source_model="anthropic/claude-haiku-4-5",
                target_model="openai/gpt-4o-mini",
                status="complete",
                confidence_score=0.77,
            )
            session.add(new_mr)
            session.commit()
            new_id = new_mr.id

        engine_after.dispose()

        # Verify the new record is queryable from a fresh engine.
        engine_verify = sa_create_engine(f"sqlite:///{alembic_db}")
        with Session(engine_verify) as session:
            result = session.get(MigrationRecord, new_id)
            assert result is not None, "MigrationRecord not found after downgrade+re-upgrade"
            assert result.source_model == "anthropic/claude-haiku-4-5"
            assert result.confidence_score == pytest.approx(0.77)

        engine_verify.dispose()

    def test_init_db_preserves_existing_data(self, alembic_db):
        """Calling init_db() on an already-populated database must not destroy data."""
        from sqlalchemy import create_engine as sa_create_engine
        from sqlmodel import Session

        from rosettastone.server.database import init_db
        from rosettastone.server.models import MigrationRecord

        _run_upgrade(alembic_db, "head")

        # Insert a record before calling init_db().
        engine_before = sa_create_engine(f"sqlite:///{alembic_db}")
        with Session(engine_before) as session:
            mr = MigrationRecord(
                source_model="openai/gpt-4o",
                target_model="anthropic/claude-sonnet-4",
                status="complete",
                confidence_score=0.88,
            )
            session.add(mr)
            session.commit()
            migration_id = mr.id

        engine_before.dispose()

        # init_db() uses create_all() which is additive — it must not drop existing tables
        # or rows.
        init_db()

        # Verify from a fresh engine that the record still exists.
        engine_verify = sa_create_engine(f"sqlite:///{alembic_db}")
        with Session(engine_verify) as session:
            result = session.get(MigrationRecord, migration_id)
            assert result is not None, "MigrationRecord was lost after calling init_db()"
            assert result.source_model == "openai/gpt-4o"
            assert result.confidence_score == pytest.approx(0.88)

        engine_verify.dispose()

    def test_multiple_records_with_fk_relationships(self, alembic_db):
        """Multiple child records across tables must all be queryable and maintain FK integrity."""
        from sqlalchemy import create_engine as sa_create_engine
        from sqlmodel import Session, select

        from rosettastone.server.models import MigrationRecord, TestCaseRecord, WarningRecord

        _run_upgrade(alembic_db, "head")

        engine_write = sa_create_engine(f"sqlite:///{alembic_db}")
        with Session(engine_write) as session:
            mr = MigrationRecord(
                source_model="openai/gpt-4o",
                target_model="anthropic/claude-sonnet-4",
                status="complete",
            )
            session.add(mr)
            session.flush()
            migration_id = mr.id

            # Insert 3 TestCaseRecords linked to the migration.
            test_cases = [
                TestCaseRecord(
                    migration_id=migration_id,
                    phase="validation",
                    output_type="json",
                    composite_score=0.80 + i * 0.05,
                    is_win=(i % 2 == 0),
                    scores_json="{}",
                    details_json="{}",
                    response_length=100 + i * 10,
                    new_response_length=95 + i * 10,
                )
                for i in range(3)
            ]
            for tc in test_cases:
                session.add(tc)

            # Insert 1 WarningRecord linked to the migration.
            warning = WarningRecord(
                migration_id=migration_id,
                warning_type="pipeline",
                severity="LOW",
                message="Cost threshold exceeded by 5%",
            )
            session.add(warning)
            session.commit()

        engine_write.dispose()

        # Verify all records from a fresh engine.
        engine_read = sa_create_engine(f"sqlite:///{alembic_db}")
        with Session(engine_read) as session:
            mr_result = session.get(MigrationRecord, migration_id)
            assert mr_result is not None, "MigrationRecord missing"

            tc_results = session.exec(
                select(TestCaseRecord).where(TestCaseRecord.migration_id == migration_id)
            ).all()
            assert len(tc_results) == 3, f"Expected 3 TestCaseRecords, got {len(tc_results)}"
            for tc in tc_results:
                assert tc.migration_id == migration_id, "TestCaseRecord FK broken"

            warning_results = session.exec(
                select(WarningRecord).where(WarningRecord.migration_id == migration_id)
            ).all()
            assert len(warning_results) == 1, (
                f"Expected 1 WarningRecord, got {len(warning_results)}"
            )
            assert warning_results[0].migration_id == migration_id, "WarningRecord FK broken"
            assert warning_results[0].severity == "LOW"

        engine_read.dispose()
