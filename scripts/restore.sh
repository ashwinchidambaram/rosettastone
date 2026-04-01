#!/usr/bin/env bash
# restore.sh — Restore a RosettaStone SQLite backup
# Usage: bash scripts/restore.sh <backup_file> [target_db_path]

set -euo pipefail

BACKUP_FILE="${1:-}"
TARGET_DB="${2:-${ROSETTASTONE_DB_PATH:-$HOME/.rosettastone/rosettastone.db}}"

if [[ -z "$BACKUP_FILE" ]]; then
    echo "Usage: $0 <backup_file> [target_db_path]" >&2
    exit 1
fi

if [[ ! -f "$BACKUP_FILE" ]]; then
    echo "ERROR: Backup file not found: $BACKUP_FILE" >&2
    exit 1
fi

# Confirm destination
echo "Restoring $BACKUP_FILE -> $TARGET_DB"

TARGET_DIR=$(dirname "$TARGET_DB")
mkdir -p "$TARGET_DIR"

# Copy the backup to the target location
sqlite3 "$BACKUP_FILE" ".backup '$TARGET_DB'"
echo "Database restored to $TARGET_DB"

# Run Alembic upgrade after restore to ensure schema is current
if command -v uv &>/dev/null; then
    echo "Running alembic upgrade head..."
    uv run alembic upgrade head
    echo "Schema upgrade complete."
else
    echo "WARNING: 'uv' not found, skipping alembic upgrade head"
fi

echo "Restore complete."
