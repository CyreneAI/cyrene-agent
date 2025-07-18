#!/bin/sh
set -e

: "${SQLITE_DB_PATH:=/app/data/agents.db}"

# Ensure parent directory and file exist
mkdir -p "$(dirname "$SQLITE_DB_PATH")"
touch "$SQLITE_DB_PATH"

# === DEBUG: print tree under /app ===
# echo "===== /app filesystem ====="
# ls -lR /app
# echo "===== /app/data filesystem ====="
# ls -lR /app/data
# echo "===== End filesystem dump ====="

# Hand off to Uvicorn
exec "$@"
