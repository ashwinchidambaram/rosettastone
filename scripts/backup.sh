#!/usr/bin/env bash
# backup.sh — Safe SQLite backup for RosettaStone
# Uses sqlite3's .backup command (WAL-safe) not cp.
# Creates 7-day daily backups and 4-week weekly backups.
# Usage: bash scripts/backup.sh [db_path] [backup_dir]

set -euo pipefail

DB_PATH="${1:-${ROSETTASTONE_DB_PATH:-$HOME/.rosettastone/rosettastone.db}}"
BACKUP_DIR="${2:-${ROSETTASTONE_BACKUP_DIR:-$HOME/.rosettastone/backups}}"

if [[ ! -f "$DB_PATH" ]]; then
    echo "ERROR: Database not found at $DB_PATH" >&2
    exit 1
fi

mkdir -p "$BACKUP_DIR/daily" "$BACKUP_DIR/weekly"

DATE=$(date +%Y-%m-%d)
DAY_OF_WEEK=$(date +%u)  # 1=Monday, 7=Sunday

DAILY_FILE="$BACKUP_DIR/daily/rosettastone-$DATE.db"
sqlite3 "$DB_PATH" ".backup '$DAILY_FILE'"
echo "Daily backup created: $DAILY_FILE"

# Weekly backup on Sundays (day 7)
if [[ "$DAY_OF_WEEK" == "7" ]]; then
    WEEK=$(date +%Y-W%V)
    WEEKLY_FILE="$BACKUP_DIR/weekly/rosettastone-$WEEK.db"
    sqlite3 "$DB_PATH" ".backup '$WEEKLY_FILE'"
    echo "Weekly backup created: $WEEKLY_FILE"
fi

# Prune: keep only last 7 daily backups (sort -r newest-first, skip first 7, delete the rest)
find "$BACKUP_DIR/daily" -name "rosettastone-*.db" -type f | sort -r | tail -n +8 | xargs rm -f
echo "Pruned old daily backups (keeping last 7)"

# Prune: keep only last 4 weekly backups
find "$BACKUP_DIR/weekly" -name "rosettastone-*.db" -type f | sort -r | tail -n +5 | xargs rm -f
echo "Pruned old weekly backups (keeping last 4)"

echo "Backup complete."
