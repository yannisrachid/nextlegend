#!/usr/bin/env bash
set -Eeuo pipefail

APP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRAPER_ROOT="${WYSCOUT_SCRAPER_ROOT:-$(cd "$APP_ROOT/.." && pwd)/nextlegend-wyscout-scraper}"
SCRAPER_RUNNER="${WYSCOUT_SCRAPER_RUNNER:-docker}" # docker|host
SCRAPER_DOCKER_BUILD="${WYSCOUT_SCRAPER_DOCKER_BUILD:-0}"
CHECK_ONLY="${CHECK_ONLY:-0}"

SEASON_LABEL="${SEASON_LABEL:-2026_2027}"
OUTPUT_CSV_NAME="${OUTPUT_CSV_NAME:-wyscout_players_${SEASON_LABEL}.csv}"
CALENDAR_PREFERENCES="${CALENDAR_PREFERENCES:-2026/2027 2026}"
SCRAPER_CLEANED="${SCRAPER_CLEANED:-$SCRAPER_ROOT/final_data/${OUTPUT_CSV_NAME%.csv}_cleaned.csv}"
APP_CLEANED="${APP_CLEANED:-$APP_ROOT/data/wyscout_players_2026_2027_cleaned.csv}"

FRESH_START="${FRESH_START:-1}"
HEADLESS="${HEADLESS:-1}"
SIM_TOPK="${SIM_TOPK:-10}"
DATA_QUALITY_MIN_ROWS="${DATA_QUALITY_MIN_ROWS:-30000}"
MIN_AVAILABLE_MEM_MB="${MIN_AVAILABLE_MEM_MB:-1800}"
MIN_AVAILABLE_MEM_MB_NO_SWAP="${MIN_AVAILABLE_MEM_MB_NO_SWAP:-2800}"
MIN_AVAILABLE_DISK_GB="${MIN_AVAILABLE_DISK_GB:-10}"

DOCKER_ENV_FILE="${DOCKER_ENV_FILE:-$APP_ROOT/.env}"
DOCKER_COMPOSE_FILE="${DOCKER_COMPOSE_FILE:-$APP_ROOT/infra/compose/docker-compose-prod.yml}"

RUN_STAMP="$(date -u +%Y%m%d_%H%M%S)"

log() {
  printf '[weekly-wyscout-chain] %s\n' "$*"
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    log "missing required file: $path"
    exit 1
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    log "missing required directory: $path"
    exit 1
  fi
}

available_mem_mb() {
  awk '/MemAvailable/ {print int($2 / 1024)}' /proc/meminfo 2>/dev/null || echo 0
}

available_disk_gb() {
  df -BG "$APP_ROOT" | awk 'NR == 2 {gsub(/G/, "", $4); print int($4)}' 2>/dev/null || echo 0
}

swap_total_mb() {
  awk '/SwapTotal/ {print int($2 / 1024)}' /proc/meminfo 2>/dev/null || echo 0
}

preflight_capacity() {
  local mem_mb disk_gb swap_mb required_mem_mb
  mem_mb="$(available_mem_mb)"
  disk_gb="$(available_disk_gb)"
  swap_mb="$(swap_total_mb)"
  required_mem_mb="$MIN_AVAILABLE_MEM_MB"
  if [[ "$swap_mb" -lt 1024 ]]; then
    required_mem_mb="$MIN_AVAILABLE_MEM_MB_NO_SWAP"
  fi
  log "capacity available_mem=${mem_mb}MiB swap=${swap_mb}MiB available_disk=${disk_gb}GiB"
  if [[ "$mem_mb" -lt "$required_mem_mb" ]]; then
    log "available memory below required=${required_mem_mb}MiB; aborting before scraper"
    exit 1
  fi
  if [[ "$disk_gb" -lt "$MIN_AVAILABLE_DISK_GB" ]]; then
    log "available disk below MIN_AVAILABLE_DISK_GB=$MIN_AVAILABLE_DISK_GB; aborting before scraper"
    exit 1
  fi
}

row_count() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo 0
    return
  fi
  local lines
  lines="$(wc -l < "$path" | tr -d ' ')"
  if [[ "$lines" -le 0 ]]; then
    echo 0
  else
    echo $((lines - 1))
  fi
}

run_scraper() {
  require_dir "$SCRAPER_ROOT"
  require_file "$SCRAPER_ROOT/run_scraper.sh"
  require_file "$SCRAPER_ROOT/data/leagues.txt"
  require_file "$SCRAPER_ROOT/.env"

  log "scraper root: $SCRAPER_ROOT"
  log "scraper runner: $SCRAPER_RUNNER"
  log "calendars: $CALENDAR_PREFERENCES"

  if [[ "$SCRAPER_RUNNER" == "docker" ]]; then
    require_file "$SCRAPER_ROOT/docker-compose.yml"
    (
      cd "$SCRAPER_ROOT"
      mkdir -p data/seasons final_data logs artifacts
      if [[ "$SCRAPER_DOCKER_BUILD" == "1" ]]; then
        docker compose build scraper
      fi
      SEASON_LABEL="$SEASON_LABEL" \
      OUTPUT_CSV_NAME="$OUTPUT_CSV_NAME" \
      CALENDAR_PREFERENCES="$CALENDAR_PREFERENCES" \
      FRESH_START="$FRESH_START" \
      HEADLESS="$HEADLESS" \
      RUN_PROCESSING=1 \
      SIM_TOPK="$SIM_TOPK" \
      docker compose run --rm \
        -e SEASON_LABEL="$SEASON_LABEL" \
        -e OUTPUT_CSV_NAME="$OUTPUT_CSV_NAME" \
        -e CALENDAR_PREFERENCES="$CALENDAR_PREFERENCES" \
        -e FRESH_START="$FRESH_START" \
        -e HEADLESS="$HEADLESS" \
        -e RUN_PROCESSING=1 \
        -e SIM_TOPK="$SIM_TOPK" \
        scraper
    )
  elif [[ "$SCRAPER_RUNNER" == "host" ]]; then
    (
      cd "$SCRAPER_ROOT"
      SEASON_LABEL="$SEASON_LABEL" \
      OUTPUT_CSV_NAME="$OUTPUT_CSV_NAME" \
      CALENDAR_PREFERENCES="$CALENDAR_PREFERENCES" \
      FRESH_START="$FRESH_START" \
      HEADLESS="$HEADLESS" \
      RUN_PROCESSING=1 \
      SIM_TOPK="$SIM_TOPK" \
      ./run_scraper.sh
    )
  else
    log "unsupported WYSCOUT_SCRAPER_RUNNER=$SCRAPER_RUNNER"
    exit 1
  fi
}

handoff_to_app() {
  require_file "$SCRAPER_CLEANED"
  local rows
  rows="$(row_count "$SCRAPER_CLEANED")"
  log "scraper cleaned rows: $rows"
  if [[ "$rows" -lt "$DATA_QUALITY_MIN_ROWS" ]]; then
    log "row count below DATA_QUALITY_MIN_ROWS=$DATA_QUALITY_MIN_ROWS; aborting DB load"
    exit 1
  fi

  mkdir -p "$APP_ROOT/data" "$APP_ROOT/data/backups"
  if [[ -f "$APP_CLEANED" ]]; then
    cp "$APP_CLEANED" "$APP_ROOT/data/backups/wyscout_players_2026_2027_cleaned_before_weekly_${RUN_STAMP}.csv"
  fi
  cp "$SCRAPER_CLEANED" "$APP_CLEANED"
  gzip -f -k "$APP_CLEANED"
  log "handoff copied to: $APP_CLEANED"
}

load_database() {
  require_file "$APP_ROOT/scripts/load_current_season_enriched.sh"
  require_file "$DOCKER_ENV_FILE"
  require_file "$DOCKER_COMPOSE_FILE"

  (
    cd "$APP_ROOT"
    DOCKER_ENV_FILE="$DOCKER_ENV_FILE" \
    DOCKER_COMPOSE_FILE="$DOCKER_COMPOSE_FILE" \
    PIPELINE_INPUT_URI="/data/$(basename "$APP_CLEANED")" \
    DATA_FRESHNESS_EXPECT_CALENDARS="2026/2027 2026" \
    DATA_QUALITY_MIN_ROWS="$DATA_QUALITY_MIN_ROWS" \
    SIM_TOPK="$SIM_TOPK" \
    SCORE_SNAPSHOT_ENABLED=1 \
    SCORE_SNAPSHOT_SEASONS="2026/2027,2026" \
    SCORE_SNAPSHOT_CADENCE=biweekly \
    PIPELINE_REPLACE_TABLES=0 \
    PIPELINE_REPLACE_INPUT_SLICES=0 \
    PIPELINE_REPLACE_SIMILARITY=0 \
    ./scripts/load_current_season_enriched.sh
  )
}

validate_database() {
  (
    cd "$APP_ROOT"
    docker compose --env-file "$DOCKER_ENV_FILE" -f "$DOCKER_COMPOSE_FILE" exec -T db \
      sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT run_id, status, rows_processed, ended_at FROM pipeline_runs ORDER BY id DESC LIMIT 3;"'
    docker compose --env-file "$DOCKER_ENV_FILE" -f "$DOCKER_COMPOSE_FILE" exec -T db \
      sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT snapshot_key, season_label, rows_snapshotted, metric_rows_snapshotted, scoring_model_version FROM scoring_snapshot_runs ORDER BY id DESC LIMIT 5;"'
  )
}

check_only() {
  require_dir "$SCRAPER_ROOT"
  require_file "$SCRAPER_ROOT/run_scraper.sh"
  require_file "$SCRAPER_ROOT/docker-compose.yml"
  require_file "$SCRAPER_ROOT/data/leagues.txt"
  require_file "$SCRAPER_ROOT/.env"
  require_file "$APP_ROOT/scripts/load_current_season_enriched.sh"
  require_file "$DOCKER_ENV_FILE"
  require_file "$DOCKER_COMPOSE_FILE"
  log "CHECK_ONLY ok"
  log "scraper output expected: $SCRAPER_CLEANED"
  log "app handoff expected: $APP_CLEANED"
}

main() {
  log "start run_stamp=$RUN_STAMP"
  if [[ "$CHECK_ONLY" == "1" ]]; then
    check_only
    return
  fi
  preflight_capacity
  run_scraper
  handoff_to_app
  load_database
  validate_database
  log "done"
}

main "$@"
