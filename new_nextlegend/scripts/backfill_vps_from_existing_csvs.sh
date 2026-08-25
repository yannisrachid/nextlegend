#!/usr/bin/env bash
set -Eeuo pipefail

# Backfill DB from existing local CSV files (no scraping).
# Intended for VPS after git pull.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCKER_ENV_FILE="${DOCKER_ENV_FILE:-$ROOT_DIR/.env}"
DOCKER_COMPOSE_FILE="${DOCKER_COMPOSE_FILE:-$ROOT_DIR/infra/compose/docker-compose-prod.yml}"

HISTORICAL_SIM_TOPK="${HISTORICAL_SIM_TOPK:-0}"
CURRENT_SIM_TOPK="${CURRENT_SIM_TOPK:-10}"
RUN_CURRENT_SEASON="${RUN_CURRENT_SEASON:-1}"
REPLACE_CURRENT_SEASON_SIMILARITY="${REPLACE_CURRENT_SEASON_SIMILARITY:-1}"
DRY_RUN="${DRY_RUN:-0}"

run_pipeline() {
  local input_uri="$1"
  local sim_topk="$2"
  local replace_similarity="$3"
  echo "[RUN] input=${input_uri} sim_topk=${sim_topk} replace_similarity=${replace_similarity}"
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  (
    cd "$ROOT_DIR"
    docker compose --env-file "$DOCKER_ENV_FILE" -f "$DOCKER_COMPOSE_FILE" run --rm \
      -e PIPELINE_REPLACE_TABLES=0 \
      -e PIPELINE_REPLACE_INPUT_SLICES=1 \
      -e PIPELINE_REPLACE_SIMILARITY="$replace_similarity" \
      -e SIM_TOPK="$sim_topk" \
      pipeline \
      --input-uri "$input_uri" \
      --input-kind raw
  )
}

for file in \
  /data/wyscout_players_2022_2023_final.csv \
  /data/wyscout_players_2023_2024_final.csv \
  /data/wyscout_players_2024_2025_final.csv \
  /data/wyscout_players_2025_final.csv
do
  run_pipeline "$file" "$HISTORICAL_SIM_TOPK" "0"
done

if [[ "$RUN_CURRENT_SEASON" == "1" ]]; then
  run_pipeline \
    /data/wyscout_players_2025_2026_final.csv \
    "$CURRENT_SIM_TOPK" \
    "$REPLACE_CURRENT_SEASON_SIMILARITY"
fi

echo "[DONE] VPS backfill from existing CSVs finished."
