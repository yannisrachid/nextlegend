#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FM_ROOT="${FM26_DATA_ROOT:-$HOME/fm26-data}"
test -f "$FM_ROOT/processed/players.jsonl"
test -f "$FM_ROOT/processed/manifest.json"
mkdir -p "$FM_ROOT/reports"
# Explicit local Docker DEV service; no SSH, no VPS credentials.
docker compose --env-file "$ROOT/.env" -f "$ROOT/infra/compose/docker-compose.yml" run --rm --no-deps \
  -v "$FM_ROOT/processed:/fm-input:ro" \
  -v "$FM_ROOT/reports:/fm-reports" \
  --entrypoint python api scripts/import_fm26.py \
  --input /fm-input/players.jsonl --manifest /fm-input/manifest.json \
  --report /fm-reports/nextlegend_import.json
