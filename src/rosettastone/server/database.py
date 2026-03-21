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


def get_engine():
    """Get or create the SQLite engine (WAL mode)."""
    global _engine
    if _engine is None:
        db_path = _get_db_path()
        _engine = create_engine(
            f"sqlite:///{db_path}",
            echo=False,
            connect_args={"check_same_thread": False},
        )
        # Enable WAL mode for concurrent reads
        with _engine.connect() as conn:
            conn.exec_driver_sql("PRAGMA journal_mode=WAL")
            conn.commit()
    return _engine


def init_db() -> None:
    """Create all tables if they don't exist."""
    import rosettastone.server.models  # noqa: F401 — ensure models registered

    SQLModel.metadata.create_all(get_engine())


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
