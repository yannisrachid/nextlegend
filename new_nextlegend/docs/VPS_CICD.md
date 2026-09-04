# VPS And CI/CD

This document is the compact production operations guide.

## Production Runtime
- VPS repo path: `~/nextlegend/new_nextlegend`.
- Frontend: `https://app.nextlegend.fr`.
- API: `https://api.nextlegend.fr`.
- Reverse proxy: Caddy.
- Runtime: Docker Compose prod file `infra/compose/docker-compose-prod.yml`.
- Database: Postgres service `db`.
- Object storage: MinIO service `minio`.

Prod compose intentionally exposes only Caddy, MinIO, and internal service ports needed by the stack. Postgres is not publicly exposed by default.

## DNS And Caddy
DNS A records:
- `app.nextlegend.fr` -> VPS IP.
- `api.nextlegend.fr` -> VPS IP.

Caddy config:
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

## Production Environment
Required prod values:
```text
NEXT_PUBLIC_API_BASE_URL=https://api.nextlegend.fr
API_BASE_URL=https://api.nextlegend.fr
CORS_ORIGINS=["https://app.nextlegend.fr"]
AUTH_COOKIE_SECURE=true
```

Do not commit `.env` values or credentials.

## Branch Policy
Production is the source of truth until the repository has been reconciled.

After reconciliation, all work must follow this branch model:
- `main`: production branch. Only deploy this branch to the VPS.
- `dev`: integration branch. Merge validated work here before promoting to main.
- `feature/<short-name>`: new product or technical feature.
- `bugfix/<short-name>`: non-urgent bug fix.
- `hotfix/<short-name>`: urgent production fix.

Required flow:
1. Fetch latest remote refs and create `feature/*`, `bugfix/*`, or `hotfix/*` from `origin/main`.
2. Merge the branch into `dev` with `git merge --no-ff`.
3. Validate `dev`.
4. Merge `dev` into `main` with `git merge --no-ff` only when ready for production.
5. Deploy `main` to the VPS.

Never develop directly on `dev` or `main`. `dev` is only for integration, and `main` is only for production.

All branch integrations must keep an explicit merge commit. Do not use fast-forward merges for `feature/*`, `bugfix/*`, `hotfix/*`, `dev`, or `main` promotion merges.

Hotfix rule:
- create `hotfix/*` from `origin/main`;
- validate and merge into `main` with `git merge --no-ff`;
- deploy immediately;
- merge `main` back into `dev` with `git merge --no-ff` after deploy so branches do not diverge.

Do not commit directly on the VPS after reconciliation. Emergency VPS edits must be captured immediately into `hotfix/*`, merged back through `main` and `dev`, then redeployed.

## Reconcile Current Production With Git
The current VPS state works and must be preserved. Do not run `git reset`, `git checkout .`, or a blind `git pull` on the VPS while it has local changes.

Safe reconciliation plan:
1. Freeze deployments and feature work until reconciliation is complete.
2. Back up production:
   - Postgres dump;
   - copy current `.env` outside Git;
   - create a filesystem archive or snapshot of `~/nextlegend`.
3. On the VPS, create a branch from the current working tree:
   ```bash
   cd ~/nextlegend
   git checkout -b reconcile/prod-state-YYYYMMDD
   ```
4. Review untracked files and exclude runtime/secrets:
   - exclude `.env`, `.env.*`, backups, DB dumps, logs, build caches, generated runtime folders;
   - keep source code, config templates, docs, committed helper data, scripts, and intentional static assets.
5. Stage the production source of truth:
   ```bash
   git add -A
   git status --short
   ```
6. Commit it:
   ```bash
   git commit -m "Capture current production state"
   ```
7. Push it:
   ```bash
   git push origin reconcile/prod-state-YYYYMMDD
   ```
8. Locally, fetch and inspect the captured branch:
   ```bash
   git fetch origin
   git diff --stat origin/main..origin/reconcile/prod-state-YYYYMMDD
   ```
9. Create or update `main` from that captured branch:
   ```bash
   git checkout -B main origin/reconcile/prod-state-YYYYMMDD
   git push -u origin main
   ```
   If `main` already exists, merge the reconciliation branch into `main` with review and `git merge --no-ff`.
10. Create or update `dev` from `main`:
    ```bash
    git checkout -B dev main
    git push -u origin dev
    ```
    If `dev` already exists, merge `main` into `dev` with `git merge --no-ff` after resolving conflicts.
11. On the VPS, switch to the clean production branch only after the pushed `main` branch matches the working production state:
    ```bash
    cd ~/nextlegend
    git fetch origin
    git checkout main
    git status --short
    ```
12. Rebuild and smoke-test:
    ```bash
    cd ~/nextlegend/new_nextlegend
    docker compose --env-file .env -f infra/compose/docker-compose-prod.yml up -d --build db minio api frontend caddy
    curl -I https://api.nextlegend.fr/
    curl -s -o /tmp/api-health.out -w "%{http_code}\n" https://api.nextlegend.fr/health
    curl -sk -o /tmp/front.out -w "%{http_code}\n" https://app.nextlegend.fr
    ```

Rollback option:
- keep the pre-reconciliation archive and DB backup until the new `main` branch has been deployed and verified;
- if deployment fails, restore the previous filesystem snapshot and restart the existing containers.

## Deploy Current Code
Current deploy mode is manual deploy on the VPS. There is no `.github/` workflow in this repo yet.

After reconciliation, deploy only from `main`:

```bash
cd ~/nextlegend/new_nextlegend
git fetch origin
git checkout main
git pull --ff-only origin main
docker compose --env-file .env -f infra/compose/docker-compose-prod.yml up -d --build db minio api frontend caddy
```

`git pull --ff-only` is only for synchronizing the VPS checkout with the already-reviewed `main` branch. It is not used for branch integration.

Batch services (`pipeline`, `pipeline-refresh`, `transfermarkt-refresh`, `current-season-job`) are behind the `jobs` profile and must never be started by a generic production deploy command.

Restart only API and frontend:
```bash
cd ~/nextlegend/new_nextlegend
docker compose --env-file .env -f infra/compose/docker-compose-prod.yml up -d --no-deps api frontend
```

Logs:
```bash
docker compose --env-file .env -f infra/compose/docker-compose-prod.yml logs -f api
docker compose --env-file .env -f infra/compose/docker-compose-prod.yml logs -f frontend
docker compose --env-file .env -f infra/compose/docker-compose-prod.yml logs -f pipeline-refresh
docker compose --env-file .env -f infra/compose/docker-compose-prod.yml logs -f transfermarkt-refresh
```

## Production Performance Requirements
- Frontend prod image must run prebuilt Next assets with `next start`.
- API prod image must run Uvicorn without reload.
- Do not bind-mount `apps/frontend` or `apps/api` over prod images.
- If prod logs show `next dev` or Uvicorn reload, fix the compose/image before debugging page performance.

## Health Checks
```bash
curl -I https://api.nextlegend.fr/
curl -s -o /tmp/api-health.out -w "%{http_code}\n" https://api.nextlegend.fr/health
curl -sk -o /tmp/front.out -w "%{http_code}\n" https://app.nextlegend.fr
```

Expected:
- API root returns 200.
- API health returns 200.
- API health returns 200.
- Frontend returns 200.
- Login succeeds and persists after refresh.

## Data Refresh Policy
Current production mode is local-compute, PRD-load:
1. Run Wyscout current-season scraping and heavy pipeline locally or on a separate worker.
2. Export final CSV and Parquet artifacts.
3. Copy artifacts to the VPS/MinIO.
4. Load artifacts into PRD Postgres.
5. Restart API/frontend only if code or env changed.

Do not run the full raw pipeline directly on the current VPS. The VPS has limited RAM and previous full runs with high similarity top-k were OOM-killed with exit code 137.

Local current-season command:
```bash
cd /Users/yannis/ylfc/new_nextlegend
./scripts/load_current_season_enriched.sh
```

Expected local outputs:
- `data/wyscout_players_2026_2027_cleaned.csv`.
- `data/current_season_similarity/player_similarity.csv`.
- local `pipeline_runs.status = success`.

Required PRD artifact tables:
- `competitions`
- `seasons`
- `players`
- `clubs`
- `player_seasons`
- `player_metrics`
- `player_metric_percentiles_global`
- `player_metric_percentiles_league`
- `role_scores`
- `player_similarity`

PRD loader rules:
- keep `PIPELINE_REPLACE_TABLES=0`;
- use `PIPELINE_REPLACE_INPUT_SLICES=0` for routine weekly refreshes;
- use `PIPELINE_REPLACE_SIMILARITY=0`; similarity is merged by unique edge then pruned to `SIM_TOPK`;
- keep `SIM_TOPK=10`;
- enforce freshness with `DATA_FRESHNESS_EXPECT_CALENDARS='2026/2027 2026'` for current-season loads;
- keep `SCORE_SNAPSHOT_ENABLED=1` for current-season loads;
- keep `SCORE_SNAPSHOT_SEASONS='2026/2027,2026'` unless the active competitions change;
- default snapshot cadence is `SCORE_SNAPSHOT_CADENCE=biweekly`; use `monthly` only if product decides to reduce granularity;
- preserve historical seasons.

Score snapshot behavior:
- snapshots store the visible score plus the scoring metrics/percentiles used by the model;
- each snapshot stores the scoring model version and hash;
- rerunning a job inside the same biweekly/monthly bucket updates that bucket instead of creating duplicates;
- full fact-table replacement truncates snapshot tables because player-season IDs are rebuilt.

Next current-season run requirement:
1. Deploy `main` first so the API/pipeline code contains the snapshot schema and writer.
2. Confirm the three snapshot tables exist:
   ```bash
   cd ~/nextlegend/new_nextlegend
   docker compose --env-file .env -f infra/compose/docker-compose-prod.yml exec -T db \
     sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Atc "SELECT to_regclass('\''public.scoring_snapshot_runs'\''), to_regclass('\''public.player_score_snapshots'\''), to_regclass('\''public.player_metric_snapshots'\'');"'
   ```
3. Run the current-season enriched load with pure incremental upsert:
   ```bash
   cd ~/nextlegend/new_nextlegend
   DOCKER_COMPOSE_FILE=infra/compose/docker-compose-prod.yml \
   DATA_FRESHNESS_EXPECT_CALENDARS="2026/2027 2026" \
   SCORE_SNAPSHOT_ENABLED=1 \
   SCORE_SNAPSHOT_SEASONS="2026/2027,2026" \
   SCORE_SNAPSHOT_CADENCE=biweekly \
   PIPELINE_REPLACE_INPUT_SLICES=0 \
   ./scripts/load_current_season_enriched.sh
   ```
4. Validate that a snapshot run was written:
   ```bash
   docker compose --env-file .env -f infra/compose/docker-compose-prod.yml exec -T db \
     sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT snapshot_key, season_label, rows_snapshotted, metric_rows_snapshotted, scoring_model_version FROM scoring_snapshot_runs ORDER BY id DESC LIMIT 5;"'
   ```

Do not use `PIPELINE_REPLACE_INPUT_SLICES=1` for routine current-season refreshes when the product goal is to preserve players already visible in the app but absent from the latest provider export. The current-season upsert key is:

```text
player_id + competition_id + season_id + club_id
```

PRD current-season enriched load:

```bash
cd ~/nextlegend/new_nextlegend
DOCKER_COMPOSE_FILE=infra/compose/docker-compose-prod.yml \
./scripts/load_current_season_enriched.sh
```

## Transfermarkt Monthly Refresh

Transfermarkt scraping is produced outside this repo by `../transfermarkt-api`.
The matcher reference is `../players-matcher`, but Next Legend stores its own auditable matching and snapshot tables.

Expected CSV in this repo:

```text
helpers/csv/transfermarkt_profiles.csv
```

Generate a fresh local roster export from the Transfermarkt scraper before the monthly refresh:

```bash
cd /Users/yannis/ylfc/new_nextlegend
docker build -t nextlegend-transfermarkt-api ../transfermarkt-api
docker run --rm \
  -v /Users/yannis/ylfc/new_nextlegend:/workspace \
  -v /Users/yannis/ylfc/transfermarkt-api:/tm-api \
  -w /workspace \
  nextlegend-transfermarkt-api \
  python scripts/scrape_transfermarkt_api_profiles.py \
    --api-dir /tm-api \
    --season-id current \
    --club-workers 4 \
    --profile-workers 4 \
    --delay 0.25 \
    --skip-profiles \
    --output helpers/csv/transfermarkt_profiles.csv \
    --roster-output helpers/csv/transfermarkt_club_rosters.csv \
    --errors-output helpers/csv/transfermarkt_scrape_errors.csv \
    --cache-dir data/transfermarkt_api_cache
PYTHONPATH=jobs/pipeline python scripts/build_transfermarkt_scope_configs.py
```

The profile enrichment path exists in the same script, but individual player profile pages can return `403` under aggressive parallelism. For the monthly value refresh, the club roster endpoint is the reliable source because it already exposes Transfermarkt ID, current club, position, date of birth, age, nationality, contract, and market value.

Optional slow profile enrichment, only after roster refresh:

```bash
docker run --rm \
  -v /Users/yannis/ylfc/new_nextlegend:/workspace \
  -v /Users/yannis/ylfc/transfermarkt-api:/tm-api \
  -w /workspace \
  nextlegend-transfermarkt-api \
  python scripts/enrich_transfermarkt_profiles_slow.py \
    --api-dir /tm-api \
    --input helpers/csv/transfermarkt_profiles.csv \
    --output helpers/csv/transfermarkt_profiles.csv \
    --errors-output helpers/csv/transfermarkt_profile_enrichment_errors.csv \
    --blocked-output helpers/csv/transfermarkt_profile_enrichment_blocked.csv \
    --cache-dir data/transfermarkt_api_cache \
    --workers 1 \
    --delay 10 \
    --retries 2 \
    --checkpoint-every 25 \
    --batch-size 10 \
    --cooldown-min-sample 10 \
    --cooldown-forbidden-rate 0.2 \
    --cooldown-seconds 1800 \
    --stop-after-forbidden 30
```

`403 Forbidden` profile responses are written to `transfermarkt_profile_enrichment_blocked.csv` and skipped on later runs. Retry them only with `--retry-blocked` after the browser/IP is no longer blocked.

Local dry-run before a new export format or large rematch:

```bash
cd /Users/yannis/ylfc/new_nextlegend
docker compose --env-file .env -f infra/compose/docker-compose.yml run --rm \
  -e TM_REFRESH_DRY_RUN=1 \
  -e TM_MATCH_REVIEW_OUTPUT=/data/transfermarkt_match_reviews/ \
  transfermarkt-refresh
```

Local write:

```bash
cd /Users/yannis/ylfc/new_nextlegend
./scripts/run_transfermarkt_monthly_refresh.sh
```

Production write after `main` has been deployed:

```bash
cd ~/nextlegend/new_nextlegend
docker compose --env-file .env -f infra/compose/docker-compose-prod.yml run --rm transfermarkt-refresh
```

Validation:

```bash
docker compose --env-file .env -f infra/compose/docker-compose-prod.yml exec -T db \
  sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT snapshot_date, COUNT(*) FROM transfermarkt_market_value_snapshots GROUP BY snapshot_date ORDER BY snapshot_date DESC LIMIT 5;"'
docker compose --env-file .env -f infra/compose/docker-compose-prod.yml exec -T db \
  sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT status, COUNT(*) FROM player_transfermarkt_matches GROUP BY status ORDER BY status;"'
```

## Monitoring
Primary monitoring is inside the Home page via `Pipeline status`.

Operational logs:
- `wyscout_current_season_job/logs/current_season_e2e_*.log`
- `wyscout_current_season_job/logs/cron_current_season.log`

SMTP alerts are optional. Prod operation should not depend on SMTP.

## Database Access For Audit
Default state: Postgres is internal-only.

Preferred temporary audit approach:
1. Create a read-only PostgreSQL user.
2. Open access only for the auditor source IP.
3. Close the port immediately after the audit.
4. Rotate or drop the audit user.

Never publish the application DB user/password for external audits.

## Backup And Restore
Backup:
```bash
cd ~/nextlegend/new_nextlegend
docker compose --env-file .env -f infra/compose/docker-compose-prod.yml exec -T db \
  sh -c 'pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB"' \
  > ~/nextlegend/backups/nextlegend_prd_$(date +%Y%m%d_%H%M%S).sql
```

Restore:
```bash
cd ~/nextlegend/new_nextlegend
docker compose --env-file .env -f infra/compose/docker-compose-prod.yml exec -T db \
  sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"' \
  < backup.sql
```

## One-Off DEV Database Promotion To PROD
Use this only for a validated release where DEV is the intended full production dataset.

Full procedure, including rollback, is documented in:

```text
docs/SCORING_MODEL_V2_IMPLEMENTATION.md
```

Required safety rule:
- create a PROD backup before restoring the DEV dump;
- keep the backup until API, frontend, report, ranking, comparison, and login have been verified in production.

## CI/CD Target
Future automation should add:
- lint and syntax checks for API, pipeline, and frontend;
- frontend build;
- Docker image build;
- controlled VPS deploy over SSH;
- post-deploy health checks.

Until that exists, treat the manual deploy commands in this file as the release process.
