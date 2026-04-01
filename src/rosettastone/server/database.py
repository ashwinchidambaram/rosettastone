"""Database engine, session factory, and initialization."""

from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path

from sqlmodel import Session, SQLModel, create_engine

_DEFAULT_DB_DIR = Path.home() / ".rosettastone"
_engine = None


def _get_db_path() -> str:
    """Resolve the database path from environment or default."""
    env_path = os.environ.get("ROSETTASTONE_DB_PATH")
    if env_path:
        return env_path
    _DEFAULT_DB_DIR.mkdir(parents=True, exist_ok=True)
    return str(_DEFAULT_DB_DIR / "migrations.db")


def _is_postgres(engine) -> bool:
    """Return True if the engine is connected to PostgreSQL."""
    return engine.dialect.name == "postgresql"


def get_engine():
    """Get or create the database engine.

    Uses PostgreSQL if DATABASE_URL is set and starts with 'postgresql://'.
    Falls back to SQLite otherwise.
    """
    global _engine
    if _engine is None:
        database_url = os.environ.get("DATABASE_URL", "")
        if database_url.startswith(("postgresql://", "postgresql+")):
            _engine = create_engine(database_url, echo=False)
        else:
            db_path = _get_db_path()
            _engine = create_engine(
                f"sqlite:///{db_path}",
                echo=False,
                connect_args={"check_same_thread": False, "timeout": 30},
            )
            # Enable WAL mode for concurrent reads (SQLite only)
            with _engine.connect() as conn:
                conn.exec_driver_sql("PRAGMA journal_mode=WAL")
                conn.commit()
    return _engine


def init_db() -> None:
    """Create all tables if they don't exist, and migrate schema for new columns."""
    import rosettastone.server.models  # noqa: F401 — ensure models registered

    engine = get_engine()
    SQLModel.metadata.create_all(engine)

    # Lightweight schema migration: add columns that may be missing on existing DBs
    _migrate_add_columns(engine)


def _migrate_add_columns(engine) -> None:
    """Add any new columns to existing tables (idempotent)."""
    new_columns = [
        ("migrations", "source_latency_p50", "REAL"),
        ("migrations", "source_latency_p95", "REAL"),
        ("migrations", "target_latency_p50", "REAL"),
        ("migrations", "target_latency_p95", "REAL"),
        ("migrations", "projected_source_cost_per_call", "REAL"),
        ("migrations", "projected_target_cost_per_call", "REAL"),
    ]
    with engine.connect() as conn:
        if _is_postgres(engine):
            # Postgres 9.6+ supports ADD COLUMN IF NOT EXISTS — no try/except needed
            for table, column, col_type in new_columns:
                conn.exec_driver_sql(
                    f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} {col_type}"
                )
        else:
            # SQLite does not support IF NOT EXISTS for columns; use try/except
            for table, column, col_type in new_columns:
                try:
                    conn.exec_driver_sql(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
                except Exception:
                    pass  # Column already exists
        conn.commit()


def get_session() -> Generator[Session, None, None]:
    """Yield a database session (for FastAPI Depends)."""
    with Session(get_engine()) as session:
        yield session


def reset_engine() -> None:
    """Reset the engine (for testing)."""
    global _engine
    if _engine is not None:
        _engine.dispose()
    _engine = None
