# Runbook (Prod + Maintenance)

Operational checklist for local and VPS operations.

## 1) Production bringup
1) DNS
- Create A records for `app.nextlegend.fr` and `api.nextlegend.fr` pointing to the VPS IP.

2) Caddy (HTTPS)
`/etc/caddy/Caddyfile`
```txt
app.nextlegend.fr {
  encode zstd gzip
  reverse_proxy 127.0.0.1:3000
}

api.nextlegend.fr {
  encode zstd gzip
  reverse_proxy 127.0.0.1:8000
}
```
Reload:
```bash
sudo systemctl reload caddy
```

3) Env (prod)
- `NEXT_PUBLIC_API_BASE_URL=https://api.nextlegend.fr`
- `API_BASE_URL=https://api.nextlegend.fr`
- `CORS_ORIGINS=["https://app.nextlegend.fr"]`
- `AUTH_COOKIE_SECURE=true`

4) Start containers
```bash
cd ~/nextlegend/new_nextlegend
sudo docker compose --env-file .env -f infra/compose/docker-compose-prod.yml up -d
```

5) Health checks
```bash
curl -I https://api.nextlegend.fr/
curl -I https://api.nextlegend.fr/health
```

## 2) VPS path convention
Use this repo path on VPS:
- `~/nextlegend/new_nextlegend`

Do not use:
- `~/new_nextlegend`

## 3) Steady-state operations
Current production mode is local-compute, PRD-load:
- the full Wyscout current-season job runs locally on Yannis' machine,
- heavy pipeline work runs locally (`TM`, scores, role scores, similarity),
- final CSVs and Parquet artifacts are copied to the VPS/MinIO,
- the VPS only loads artifacts into PRD Postgres and serves the app.

Do not run the full current-season pipeline directly on the current VPS. The VPS
has about 3.7 GiB RAM and no swap; full runs with `SIM_TOPK=30` were OOM-killed
with exit code `137`.

One-time historical backfill was completed during VPS initialization.
Keep `scripts/backfill_vps_from_existing_csvs.sh` only for exceptional recovery scenarios.

## 4) Weekly scheduling policy
Do not install the weekly current-season cron on the VPS while the current VPS
size is kept.

If automation is needed, schedule it on:
- Yannis' local machine, or
- a separate worker/VM with enough RAM.

That scheduled job must follow the local-compute, PRD-load flow below.

Remove any old VPS cron entry that calls:
- `run_current_season_e2e.sh`
- `run_current_season_e2e_via_docker.sh`
- `current-season-job`

Check VPS cron:
```bash
crontab -l
```

## 5) Weekly refresh procedure
1) Run the end-to-end job locally.

```bash
cd /Users/yannis/ylfc/new_nextlegend
DOCKER_COMPOSE_FILE=infra/compose/docker-compose.yml \
DOCKER_ENV_FILE=.env \
SKIP_EMAIL_ALERTS=1 \
PIPELINE_REPLACE_INPUT_SLICES=1 \
PIPELINE_REPLACE_SIMILARITY=1 \
./wyscout_current_season_job/run_current_season_e2e_via_docker.sh
```

Expected local outputs:
- `data/wyscout_players_final.csv`
- local `pipeline_runs` row with `status=success`
- local DB updated for the current-season slices

2) Publish final CSVs to PRD MinIO.

Required PRD object:
- `s3://nextlegend/data/wyscout_players_final.csv`

Recommended PRD object for traceability:
- `s3://nextlegend/data/wyscout_players_enriched_tm_scores_2025_2026_current.csv`

3) Export local pipeline artifacts as Parquet.

Required artifact set:
- `competitions.parquet`
- `seasons.parquet`
- `players.parquet`
- `clubs.parquet`
- `player_seasons.parquet`
- `player_metrics.parquet`
- `role_scores.parquet`
- `player_similarity.parquet`

Copy the completed artifact directory to the VPS:

```text
~/nextlegend/new_nextlegend/data/prd_upsert_artifacts_current/
```

4) Run the PRD loader from those artifacts.

The PRD loader must:
- keep `PIPELINE_REPLACE_TABLES=0`,
- use `PIPELINE_REPLACE_INPUT_SLICES=1`,
- use `PIPELINE_REPLACE_SIMILARITY=1`,
- load artifacts into Postgres from the VPS container,
- insert a `pipeline_runs` success row.

This preserves historical seasons and replaces only the current-season slices
plus the similarity table.

5) Restart app services after `.env` or image changes.

```bash
cd ~/nextlegend/new_nextlegend
docker compose --env-file .env -f infra/compose/docker-compose-prod.yml up -d --no-deps api frontend
```

6) Validate PRD.

```bash
cd ~/nextlegend/new_nextlegend
set -a && . ./.env && set +a

docker compose --env-file .env -f infra/compose/docker-compose-prod.yml exec -T db \
  psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "
    SELECT
      (SELECT count(*) FROM player_seasons) AS player_seasons,
      (SELECT count(*) FROM player_metrics) AS player_metrics,
      (SELECT count(*) FROM role_scores) AS role_scores,
      (SELECT count(*) FROM player_similarity) AS player_similarity;

    SELECT id, run_id, status, rows_processed, source_uri, started_at
    FROM pipeline_runs
    ORDER BY id DESC
    LIMIT 3;
  "

curl -sk https://api.nextlegend.fr/health
curl -sk -o /tmp/front.out -w "%{http_code}\n" https://app.nextlegend.fr
```

Reference validation from the last successful PRD load:

```text
run_id=b4199ef4-b653-4618-b3fe-056451f77043
rows_processed=43429
source_uri=s3://nextlegend/data/wyscout_players_final.csv

player_seasons     190546
player_metrics     190239
role_scores        6669110
player_similarity  1302650
```

Future automation target:
- `export_prd_artifacts_from_local.sh`
- `upload_prd_minio_inputs.sh`
- `load_prd_artifacts_on_vps.sh`
- `validate_prd_refresh.sh`

## 6) Monitoring (no SMTP required)
Monitoring is done in Home page:
- Frontend Home shows `Pipeline status`.
- It also indicates whether current season (`2025/2026` or `2026`) is loaded.

Operational logs:
- `wyscout_current_season_job/logs/current_season_e2e_*.log`
- `wyscout_current_season_job/logs/cron_current_season.log`

Manual local run:
```bash
cd /Users/yannis/ylfc/new_nextlegend
DOCKER_COMPOSE_FILE=infra/compose/docker-compose.yml \
DOCKER_ENV_FILE=.env \
SKIP_EMAIL_ALERTS=1 \
PIPELINE_REPLACE_INPUT_SLICES=1 \
PIPELINE_REPLACE_SIMILARITY=1 \
./wyscout_current_season_job/run_current_season_e2e_via_docker.sh
```

## 7) Maintenance
1) Update code + rebuild
```bash
cd ~/nextlegend/new_nextlegend
git pull
sudo docker compose --env-file .env -f infra/compose/docker-compose-prod.yml up -d --build
```

2) Restart specific services
```bash
cd ~/nextlegend/new_nextlegend
sudo docker compose --env-file .env -f infra/compose/docker-compose-prod.yml restart api frontend
```

3) Logs
```bash
cd ~/nextlegend/new_nextlegend
sudo docker compose --env-file .env -f infra/compose/docker-compose-prod.yml logs -f api
sudo docker compose --env-file .env -f infra/compose/docker-compose-prod.yml logs -f frontend
sudo docker compose --env-file .env -f infra/compose/docker-compose-prod.yml logs -f pipeline-refresh
```

4) Postgres backup (manual)
```bash
cd ~/nextlegend/new_nextlegend
sudo docker compose --env-file .env -f infra/compose/docker-compose-prod.yml exec db \
  pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" > backup.sql
```

5) Postgres restore (manual)
```bash
cd ~/nextlegend/new_nextlegend
cat backup.sql | sudo docker compose --env-file .env -f infra/compose/docker-compose-prod.yml exec -T db \
  psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"
```

6) Rebuild similarity table
- Do not rebuild similarity directly on the current VPS.
- Rebuild it through the local-compute, PRD-load flow above.

7) Pipeline OOM mitigation
- Do not retry the full raw pipeline on the current VPS.
- Use the local-compute, PRD-load flow above.
- If the business decision changes, upgrade the VPS or add swap before enabling a VPS cron.
