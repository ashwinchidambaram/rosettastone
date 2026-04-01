"""Tests for backup/restore scripts."""
from __future__ import annotations

import os
import sqlite3
import subprocess
from pathlib import Path

import pytest

# Only run on platforms that have bash + sqlite3
pytestmark = pytest.mark.skipif(
    subprocess.run(["which", "sqlite3"], capture_output=True).returncode != 0,
    reason="sqlite3 not available",
)


def _make_test_db(path: Path) -> None:
    """Create a minimal SQLite DB for testing."""
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE test_data (id INTEGER PRIMARY KEY, value TEXT)")
    conn.execute("INSERT INTO test_data VALUES (1, 'hello')")
    conn.commit()
    conn.close()


def test_backup_creates_backup_file(tmp_path: Path) -> None:
    """backup.sh must create a backup file in the daily directory."""
    db = tmp_path / "rosettastone.db"
    backup_dir = tmp_path / "backups"
    _make_test_db(db)

    result = subprocess.run(
        ["bash", "scripts/backup.sh", str(db), str(backup_dir)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"backup.sh failed: {result.stderr}"

    daily_backups = list((backup_dir / "daily").glob("*.db"))
    assert len(daily_backups) == 1, f"Expected 1 backup, got {daily_backups}"


def test_backup_missing_database(tmp_path: Path) -> None:
    """backup.sh must fail gracefully if database does not exist."""
    db = tmp_path / "nonexistent.db"
    backup_dir = tmp_path / "backups"

    result = subprocess.run(
        ["bash", "scripts/backup.sh", str(db), str(backup_dir)],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "ERROR: Database not found" in result.stderr


def test_restore_restores_database(tmp_path: Path) -> None:
    """restore.sh must copy backup to target location."""
    db = tmp_path / "original.db"
    backup_dir = tmp_path / "backups"
    target_db = tmp_path / "restored.db"
    _make_test_db(db)

    # Create backup first
    subprocess.run(
        ["bash", "scripts/backup.sh", str(db), str(backup_dir)], check=True
    )
    backup_file = next((backup_dir / "daily").glob("*.db"))

    # Use a minimal PATH that has bash/sqlite3 but not uv — triggers the "uv not found" warning path
    env = os.environ.copy()
    env["PATH"] = "/bin:/usr/bin"

    result = subprocess.run(
        ["/bin/bash", "scripts/restore.sh", str(backup_file), str(target_db)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, f"restore.sh failed: {result.stderr}"
    assert target_db.exists(), "Restored database file not found"

    # Verify data integrity
    conn = sqlite3.connect(str(target_db))
    rows = conn.execute("SELECT value FROM test_data WHERE id=1").fetchall()
    conn.close()
    assert rows == [("hello",)], f"Data integrity check failed: {rows}"


def test_restore_missing_backup_file(tmp_path: Path) -> None:
    """restore.sh must fail gracefully if backup file does not exist."""
    backup_file = tmp_path / "nonexistent.db"
    target_db = tmp_path / "restored.db"

    result = subprocess.run(
        ["bash", "scripts/restore.sh", str(backup_file), str(target_db)],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "ERROR: Backup file not found" in result.stderr


def test_restore_no_arguments() -> None:
    """restore.sh must fail if no backup file argument provided."""
    result = subprocess.run(
        ["bash", "scripts/restore.sh"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "Usage:" in result.stderr
