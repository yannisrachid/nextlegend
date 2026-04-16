#!/usr/bin/env bash
set -Eeuo pipefail

JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$JOB_DIR/.." && pwd)"

# Load local .env if present (Wyscout + SMTP credentials for cron usage).
if [[ -f "$JOB_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$JOB_DIR/.env"
  set +a
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRAPER_SCRIPT="${SCRAPER_SCRIPT:-$JOB_DIR/run_wyscout_current_weekly.sh}"
SCRAPER_FINAL_CSV="${SCRAPER_FINAL_CSV:-$JOB_DIR/final_data/wyscout_players_2025_2026_final.csv}"
TARGET_FINAL_CSV="${TARGET_FINAL_CSV:-$REPO_ROOT/data/wyscout_players_2025_2026_final.csv}"

DOCKER_ENV_FILE="${DOCKER_ENV_FILE:-$REPO_ROOT/.env}"
DOCKER_COMPOSE_FILE="${DOCKER_COMPOSE_FILE:-$REPO_ROOT/infra/compose/docker-compose.yml}"
PIPELINE_INPUT_URI="${PIPELINE_INPUT_URI:-/data/wyscout_players_2025_2026_final.csv}"
PIPELINE_INPUT_KIND="${PIPELINE_INPUT_KIND:-raw}"
PIPELINE_RUNNER="${PIPELINE_RUNNER:-docker}" # docker|python

PIPELINE_REPLACE_TABLES="${PIPELINE_REPLACE_TABLES:-0}"
PIPELINE_REPLACE_INPUT_SLICES="${PIPELINE_REPLACE_INPUT_SLICES:-1}"
PIPELINE_REPLACE_SIMILARITY="${PIPELINE_REPLACE_SIMILARITY:-0}"
SIM_TOPK="${SIM_TOPK:-30}"
VERIFY_PIPELINE_RUN="${VERIFY_PIPELINE_RUN:-1}"

SKIP_SCRAPE="${SKIP_SCRAPE:-0}"
SKIP_PIPELINE="${SKIP_PIPELINE:-0}"
SKIP_EMAIL_ALERTS="${SKIP_EMAIL_ALERTS:-0}"
REQUIRE_SMTP_ALERTS="${REQUIRE_SMTP_ALERTS:-0}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-10}"

EMAIL_LOG_LINES="${EMAIL_LOG_LINES:-200}"
EMAIL_SUBJECT_PREFIX="${EMAIL_SUBJECT_PREFIX:-[NextLegend][CurrentSeason]}"
MIN_FREE_MB_HARD="${MIN_FREE_MB_HARD:-512}"
MIN_FREE_MB_WARN="${MIN_FREE_MB_WARN:-2048}"

timestamp="$(date +"%Y%m%d_%H%M%S")"
LOG_DIR="${LOG_DIR:-$JOB_DIR/logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/current_season_e2e_${timestamp}.log}"

mkdir -p "$LOG_DIR"
mkdir -p "$(dirname "$TARGET_FINAL_CSV")"

if [[ -f "$JOB_DIR/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$JOB_DIR/.venv/bin/activate"
  PYTHON_BIN="python"
fi

if [[ "${TEE_TO_STDOUT:-0}" == "1" ]]; then
  if exec > >(tee -a "$LOG_FILE") 2>&1; then
    :
  else
    exec >>"$LOG_FILE" 2>&1
  fi
else
  exec >>"$LOG_FILE" 2>&1
fi

RUN_STARTED_AT="$(date -Iseconds)"
CURRENT_PHASE="init"
PIPELINE_RUN_ID=""

check_command() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "[ERROR] missing command: $cmd"
    exit 1
  fi
}

check_free_space_mb() {
  local path="$1"
  df -Pm "$path" | awk 'NR==2 {print $4}'
}

check_smtp_requirements() {
  if [[ "$SKIP_EMAIL_ALERTS" == "1" || "$REQUIRE_SMTP_ALERTS" != "1" ]]; then
    return 0
  fi
  local missing=()
  [[ -z "${SMTP_HOST:-}" ]] && missing+=("SMTP_HOST")
  if [[ -z "${SMTP_FROM:-}" && -z "${SMTP_USERNAME:-}" ]]; then
    missing+=("SMTP_FROM_or_SMTP_USERNAME")
  fi
  [[ -z "${SMTP_TO:-}" ]] && missing+=("SMTP_TO")
  if (( ${#missing[@]} > 0 )); then
    echo "[ERROR] REQUIRE_SMTP_ALERTS=1 but missing SMTP settings: ${missing[*]}"
    return 1
  fi
  return 0
}

verify_pipeline_run_status() {
  if [[ -z "$PIPELINE_RUN_ID" ]]; then
    echo "[WARN] unable to verify pipeline run in DB (run_id missing)"
    return 0
  fi
  local status
  if [[ "$PIPELINE_RUNNER" == "python" ]]; then
    status="$(
      PIPELINE_RUN_ID="$PIPELINE_RUN_ID" DATABASE_URL="${DATABASE_URL:-}" "$PYTHON_BIN" - <<'PY'
import os
import re
import psycopg

run_id = os.getenv("PIPELINE_RUN_ID", "").strip()
db_url = os.getenv("DATABASE_URL", "").strip()
if not run_id or not db_url:
    raise SystemExit(0)

db_url = re.sub(r"^postgresql\+[^:]+://", "postgresql://", db_url)
with psycopg.connect(db_url) as conn:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT status FROM pipeline_runs WHERE run_id=%s ORDER BY id DESC LIMIT 1;",
            (run_id,),
        )
        row = cur.fetchone()
        if row and row[0]:
            print(str(row[0]).strip())
PY
    )"
  else
    local pg_user="${POSTGRES_USER:-nextlegend}"
    local pg_db="${POSTGRES_DB:-nextlegend}"
    status="$(
      cd "$REPO_ROOT" && docker compose --env-file "$DOCKER_ENV_FILE" -f "$DOCKER_COMPOSE_FILE" exec -T db \
        psql -U "$pg_user" -d "$pg_db" -Atc \
        "SELECT status FROM pipeline_runs WHERE run_id='${PIPELINE_RUN_ID}' ORDER BY id DESC LIMIT 1;"
    )"
  fi
  status="$(echo "$status" | tr -d '[:space:]')"
  if [[ "$status" != "success" ]]; then
    echo "[ERROR] pipeline run verification failed: run_id=${PIPELINE_RUN_ID} status='${status}'"
    return 1
  fi
  echo "[STEP] pipeline run verified in DB: run_id=${PIPELINE_RUN_ID} status=${status}"
  return 0
}

send_status_email() {
  local rc="$1"
  local status="FAILED"
  if [[ "$rc" -eq 0 ]]; then
    status="SUCCESS"
  fi

  if [[ "$SKIP_EMAIL_ALERTS" == "1" ]]; then
    echo "[EMAIL] skip (SKIP_EMAIL_ALERTS=1)"
    return 0
  fi

  local body_file
  body_file="$(mktemp)"
  {
    echo "Current season job: ${status}"
    echo "Started at: ${RUN_STARTED_AT}"
    echo "Finished at: $(date -Iseconds)"
    echo "Phase at exit: ${CURRENT_PHASE}"
    echo "Exit code: ${rc}"
    if [[ -n "$PIPELINE_RUN_ID" ]]; then
      echo "Pipeline run_id: ${PIPELINE_RUN_ID}"
    fi
    echo "Log file: ${LOG_FILE}"
    echo "Target CSV: ${TARGET_FINAL_CSV}"
  } >"$body_file"

  local subject="${EMAIL_SUBJECT_PREFIX} ${status} $(date +'%Y-%m-%d %H:%M:%S')"
  if "$PYTHON_BIN" "$JOB_DIR/scripts/send_job_email.py" \
    --subject "$subject" \
    --body-file "$body_file" \
    --log-file "$LOG_FILE" \
    --max-log-lines "$EMAIL_LOG_LINES"; then
    rm -f "$body_file"
    return 0
  fi

  rm -f "$body_file"
  if [[ "$REQUIRE_SMTP_ALERTS" == "1" ]]; then
    echo "[ERROR] email alert failed (REQUIRE_SMTP_ALERTS=1)"
    return 1
  fi
  echo "[WARN] email alert failed (non-blocking)"
  return 0
}

on_exit() {
  local rc=$?
  local final_rc="$rc"
  echo "[END] rc=${rc} phase=${CURRENT_PHASE} finished_at=$(date -Iseconds)"
  if ! send_status_email "$rc"; then
    if [[ "$rc" -eq 0 ]]; then
      final_rc=1
    fi
  fi
  trap - EXIT
  exit "$final_rc"
}
trap on_exit EXIT

echo "[START] run_current_season_e2e started_at=${RUN_STARTED_AT}"
echo "[INFO] log_file=${LOG_FILE}"
echo "[INFO] repo_root=${REPO_ROOT}"

check_command awk
check_command df
if [[ "$PIPELINE_RUNNER" == "docker" ]]; then
  check_command docker
fi
check_smtp_requirements
free_mb="$(check_free_space_mb "$REPO_ROOT" || echo 0)"
if [[ "$free_mb" =~ ^[0-9]+$ ]]; then
  if (( free_mb < MIN_FREE_MB_HARD )); then
    echo "[ERROR] free disk too low: ${free_mb}MB (< ${MIN_FREE_MB_HARD}MB)"
    exit 1
  fi
  if (( free_mb < MIN_FREE_MB_WARN )); then
    echo "[WARN] low free disk: ${free_mb}MB (< ${MIN_FREE_MB_WARN}MB recommended)"
  fi
else
  echo "[WARN] unable to determine free disk space"
fi

if [[ "$PIPELINE_RUNNER" == "docker" ]]; then
  (
    cd "$REPO_ROOT"
    docker compose --env-file "$DOCKER_ENV_FILE" -f "$DOCKER_COMPOSE_FILE" ps >/dev/null
  )
elif [[ "$PIPELINE_RUNNER" != "python" ]]; then
  echo "[ERROR] unsupported PIPELINE_RUNNER=${PIPELINE_RUNNER} (expected: docker|python)"
  exit 1
fi

if [[ "$SKIP_SCRAPE" != "1" ]]; then
  CURRENT_PHASE="scrape"
  echo "[STEP] scrape + rename (MAX_ATTEMPTS=${MAX_ATTEMPTS})"
  PYTHONUNBUFFERED=1 "$PYTHON_BIN" -u -c "import playwright.sync_api" >/dev/null 2>&1 || {
    echo "[ERROR] playwright python package unavailable for $PYTHON_BIN"
    exit 1
  }
  MAX_ATTEMPTS="$MAX_ATTEMPTS" "$SCRAPER_SCRIPT"
else
  echo "[STEP] scrape skipped (SKIP_SCRAPE=1)"
fi

CURRENT_PHASE="prepare_csv"
if [[ ! -f "$SCRAPER_FINAL_CSV" ]]; then
  echo "[ERROR] missing scraper final csv: ${SCRAPER_FINAL_CSV}"
  exit 1
fi

src_abs="$(cd "$(dirname "$SCRAPER_FINAL_CSV")" && pwd)/$(basename "$SCRAPER_FINAL_CSV")"
dst_abs="$(cd "$(dirname "$TARGET_FINAL_CSV")" && pwd)/$(basename "$TARGET_FINAL_CSV")"
if [[ "$src_abs" == "$dst_abs" ]]; then
  echo "[STEP] final csv already at target path -> ${TARGET_FINAL_CSV}"
else
  cp "$SCRAPER_FINAL_CSV" "$TARGET_FINAL_CSV"
  echo "[STEP] copied final csv -> ${TARGET_FINAL_CSV}"
fi
wc -l "$TARGET_FINAL_CSV" || true
csv_lines="$(wc -l < "$TARGET_FINAL_CSV" | tr -d '[:space:]')"
if [[ "$csv_lines" =~ ^[0-9]+$ ]] && (( csv_lines <= 1 )); then
  echo "[ERROR] target csv has no data rows: ${TARGET_FINAL_CSV}"
  exit 1
fi

if [[ "$SKIP_PIPELINE" != "1" ]]; then
  CURRENT_PHASE="pipeline"
  echo "[STEP] pipeline upsert current season"
  if [[ "$PIPELINE_RUNNER" == "docker" ]]; then
    (
      cd "$REPO_ROOT"
      docker compose --env-file "$DOCKER_ENV_FILE" -f "$DOCKER_COMPOSE_FILE" run --rm \
        -e PIPELINE_REPLACE_TABLES="$PIPELINE_REPLACE_TABLES" \
        -e PIPELINE_REPLACE_INPUT_SLICES="$PIPELINE_REPLACE_INPUT_SLICES" \
        -e PIPELINE_REPLACE_SIMILARITY="$PIPELINE_REPLACE_SIMILARITY" \
        -e SIM_TOPK="$SIM_TOPK" \
        pipeline \
        --input-uri "$PIPELINE_INPUT_URI" \
        --input-kind "$PIPELINE_INPUT_KIND"
    )
  else
    (
      cd "$REPO_ROOT/jobs/pipeline"
      PIPELINE_REPLACE_TABLES="$PIPELINE_REPLACE_TABLES" \
      PIPELINE_REPLACE_INPUT_SLICES="$PIPELINE_REPLACE_INPUT_SLICES" \
      PIPELINE_REPLACE_SIMILARITY="$PIPELINE_REPLACE_SIMILARITY" \
      SIM_TOPK="$SIM_TOPK" \
      "$PYTHON_BIN" -m pipeline.run \
        --input-uri "$PIPELINE_INPUT_URI" \
        --input-kind "$PIPELINE_INPUT_KIND"
    )
  fi

  PIPELINE_RUN_ID="$(grep -Eo 'run_id=[0-9a-f-]+' "$LOG_FILE" | tail -n1 | cut -d'=' -f2 || true)"
  if [[ -n "$PIPELINE_RUN_ID" ]]; then
    echo "[STEP] detected pipeline run_id=${PIPELINE_RUN_ID}"
  fi
  if [[ "$VERIFY_PIPELINE_RUN" == "1" ]]; then
    verify_pipeline_run_status
  fi
else
  echo "[STEP] pipeline skipped (SKIP_PIPELINE=1)"
fi

CURRENT_PHASE="done"
echo "[DONE] current season end-to-end flow completed"
