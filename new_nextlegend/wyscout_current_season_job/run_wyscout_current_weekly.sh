#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-10}"
FRESH_START="${FRESH_START:-1}"
ATTEMPT_IDLE_TIMEOUT_SECONDS="${ATTEMPT_IDLE_TIMEOUT_SECONDS:-180}"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

mkdir -p data/seasons final_data

echo "[1/2] Scraping calendars 2025/2026 then 2026 (fallback)..."
SCRAPE_CMD=(
  "$PYTHON_BIN" -u scripts/run_wyscout_resumable.py
  --selected-file data/leagues.txt
  --download-dir data/seasons
  --output-csv-name wyscout_players_2025_2026.csv
  --state-file data/seasons/.resume_2025_2026.json
  --max-attempts "$MAX_ATTEMPTS"
  --attempt-idle-timeout-seconds "$ATTEMPT_IDLE_TIMEOUT_SECONDS"
  --calendar-preferences 2025/2026 2026
)
if [[ "$FRESH_START" == "1" ]]; then
  SCRAPE_CMD+=(--fresh-start)
fi
SCRAPE_CMD+=(--passthrough --headless --auto-login --auto-open-advanced-search --auto-select-all-columns)

PYTHONUNBUFFERED=1 "${SCRAPE_CMD[@]}"

echo "[2/2] Renaming columns..."
PYTHONUNBUFFERED=1 "$PYTHON_BIN" -u scripts/rename_columns.py \
  --input data/seasons/wyscout_players_2025_2026.csv \
  --output final_data/wyscout_players_2025_2026_final.csv

echo "Done. Output: final_data/wyscout_players_2025_2026_final.csv"
