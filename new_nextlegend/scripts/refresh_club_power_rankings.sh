#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCKER_ENV_FILE="${DOCKER_ENV_FILE:-$ROOT_DIR/.env}"
DOCKER_COMPOSE_FILE="${DOCKER_COMPOSE_FILE:-$ROOT_DIR/infra/compose/docker-compose.yml}"
GENDER="${OPTA_POWER_GENDER:-mens}"
SOURCE_URL="${OPTA_POWER_SOURCE_URL:-https://dataviz.theanalyst.com/opta-power-rankings/}"
CSV_OUTPUT="${OPTA_POWER_CSV_OUTPUT:-}"

cd "$ROOT_DIR"

args=(
  python
  scripts/refresh_opta_power_rankings.py
  --gender "$GENDER"
  --source-url "$SOURCE_URL"
)

if [[ -n "$CSV_OUTPUT" ]]; then
  args+=(--csv-output "$CSV_OUTPUT")
fi

docker compose --env-file "$DOCKER_ENV_FILE" -f "$DOCKER_COMPOSE_FILE" run --rm api "${args[@]}"
