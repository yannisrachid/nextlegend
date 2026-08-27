#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCKER_ENV_FILE="${DOCKER_ENV_FILE:-$ROOT_DIR/.env}"
DOCKER_COMPOSE_FILE="${DOCKER_COMPOSE_FILE:-$ROOT_DIR/infra/compose/docker-compose.yml}"
INPUT_URI="${PIPELINE_INPUT_URI:-/data/wyscout_players_2026_2027_cleaned.csv}"
SIMILARITY_PREFIX="${SIMILARITY_PREFIX:-/data/current_season_similarity}"
EXPECTED_CALENDARS="${DATA_FRESHNESS_EXPECT_CALENDARS:-2026/2027 2026}"
MIN_ROWS="${DATA_QUALITY_MIN_ROWS:-30000}"
SIM_TOPK="${SIM_TOPK:-10}"

cd "$ROOT_DIR"

docker compose --env-file "$DOCKER_ENV_FILE" -f "$DOCKER_COMPOSE_FILE" run --rm \
  -e PIPELINE_INPUT_KIND=enriched \
  -e PIPELINE_REPLACE_TABLES=0 \
  -e PIPELINE_REPLACE_INPUT_SLICES=0 \
  -e PIPELINE_REPLACE_SIMILARITY=0 \
  -e PIPELINE_COPY_SIMILARITY=0 \
  -e SIM_TOPK="$SIM_TOPK" \
  -e DATA_QUALITY_SIM_TOPK="$SIM_TOPK" \
  -e DATA_FRESHNESS_EXPECT_CALENDARS="$EXPECTED_CALENDARS" \
  -e DATA_QUALITY_MIN_ROWS="$MIN_ROWS" \
  pipeline \
  --input-uri "$INPUT_URI" \
  --input-kind enriched \
  --similarity-prefix "$SIMILARITY_PREFIX"
