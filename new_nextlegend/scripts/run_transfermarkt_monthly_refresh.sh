#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCKER_ENV_FILE="${DOCKER_ENV_FILE:-$ROOT_DIR/.env}"
DOCKER_COMPOSE_FILE="${DOCKER_COMPOSE_FILE:-$ROOT_DIR/infra/compose/docker-compose.yml}"
TM_PROFILES_PATH="${TM_PROFILES_PATH:-/helpers/csv/transfermarkt_profiles.csv}"
TM_PLAYER_MAP_PATH="${TM_PLAYER_MAP_PATH:-/helpers/csv/player_matching_reference.csv}"
SNAPSHOT_DATE="${TM_SNAPSHOT_DATE:-$(date -u +%F)}"
SEASON_LABEL="${TM_REFRESH_SEASON_LABEL:-2026/2027}"
REVIEW_OUTPUT="${TM_MATCH_REVIEW_OUTPUT:-/data/transfermarkt_match_reviews/}"

cd "$ROOT_DIR"

docker compose --env-file "$DOCKER_ENV_FILE" -f "$DOCKER_COMPOSE_FILE" run --rm \
  -e TM_PROFILES_PATH="$TM_PROFILES_PATH" \
  -e TM_PLAYER_MAP_PATH="$TM_PLAYER_MAP_PATH" \
  -e TM_SNAPSHOT_DATE="$SNAPSHOT_DATE" \
  -e TM_REFRESH_SEASON_LABEL="$SEASON_LABEL" \
  -e TM_MATCH_REVIEW_OUTPUT="$REVIEW_OUTPUT" \
  transfermarkt-refresh \
    --tm-profiles "$TM_PROFILES_PATH" \
    --player-map "$TM_PLAYER_MAP_PATH" \
    --snapshot-date "$SNAPSHOT_DATE" \
    --season-label "$SEASON_LABEL"
