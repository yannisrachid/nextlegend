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
- `prod`: production branch. Only deploy this branch to the VPS.
- `dev`: integration branch. Merge validated work here before promoting to prod.
- `feature/<short-name>`: new product or technical feature.
- `bugfix/<short-name>`: non-urgent bug fix.
- `hotfix/<short-name>`: urgent production fix.

Required flow:
1. Create `feature/*`, `bugfix/*`, or `hotfix/*` from the appropriate base.
2. Merge the branch into `dev`.
3. Validate `dev`.
4. Merge `dev` into `prod` only when ready for production.
5. Deploy `prod` to the VPS.

Hotfix rule:
- create `hotfix/*` from `prod`;
- validate and merge into `prod`;
- deploy immediately;
- merge `prod` back into `dev` after deploy so branches do not diverge.

Do not commit directly on the VPS after reconciliation. Emergency VPS edits must be captured immediately into `hotfix/*`, merged back through `prod` and `dev`, then redeployed.

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
   git switch -c reconcile/prod-state-YYYYMMDD
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
   git diff --stat origin/prod..origin/reconcile/prod-state-YYYYMMDD
   ```
9. Create or update `prod` from that captured branch:
   ```bash
   git switch -c prod origin/reconcile/prod-state-YYYYMMDD
   git push -u origin prod
   ```
   If `prod` already exists, merge the reconciliation branch into `prod` with review.
10. Create or update `dev` from `prod`:
    ```bash
    git switch -c dev prod
    git push -u origin dev
    ```
    If `dev` already exists, merge `prod` into `dev` after resolving conflicts.
11. On the VPS, switch to the clean production branch only after the pushed `prod` branch matches the working production state:
    ```bash
    cd ~/nextlegend
    git fetch origin
    git switch prod
    git status --short
    ```
12. Rebuild and smoke-test:
    ```bash
    cd ~/nextlegend/new_nextlegend
    docker compose --env-file .env -f infra/compose/docker-compose-prod.yml up -d --build db minio api frontend caddy
    curl -I https://api.nextlegend.fr/
    curl -I https://api.nextlegend.fr/health
    curl -sk -o /tmp/front.out -w "%{http_code}\n" https://app.nextlegend.fr
    ```

Rollback option:
- keep the pre-reconciliation archive and DB backup until the new `prod` branch has been deployed and verified;
- if deployment fails, restore the previous filesystem snapshot and restart the existing containers.

## Deploy Current Code
Current deploy mode is manual deploy on the VPS. There is no `.github/` workflow in this repo yet.

After reconciliation, deploy only from `prod`:

```bash
cd ~/nextlegend/new_nextlegend
git fetch origin
git switch prod
git pull --ff-only origin prod
docker compose --env-file .env -f infra/compose/docker-compose-prod.yml up -d --build db minio api frontend caddy
```

Batch services (`pipeline`, `pipeline-refresh`, `current-season-job`) are behind the `jobs` profile and must never be started by a generic production deploy command.

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
```

## Production Performance Requirements
- Frontend prod image must run prebuilt Next assets with `next start`.
- API prod image must run Uvicorn without reload.
- Do not bind-mount `apps/frontend` or `apps/api` over prod images.
- If prod logs show `next dev` or Uvicorn reload, fix the compose/image before debugging page performance.

## Health Checks
```bash
curl -I https://api.nextlegend.fr/
curl -I https://api.nextlegend.fr/health
curl -sk -o /tmp/front.out -w "%{http_code}\n" https://app.nextlegend.fr
```

Expected:
- API root returns 200.
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

Do not run the full raw pipeline directly on the current VPS. The VPS has limited RAM and previous full runs with `SIM_TOPK=30` were OOM-killed with exit code 137.

Local current-season command:
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
- `data/wyscout_players_final.csv`.
- enriched pipeline artifacts.
- local `pipeline_runs.status = success`.

Required PRD artifact tables:
- `competitions`
- `seasons`
- `players`
- `clubs`
- `player_seasons`
- `player_metrics`
- `role_scores`
- `player_similarity`

PRD loader rules:
- keep `PIPELINE_REPLACE_TABLES=0`;
- use `PIPELINE_REPLACE_INPUT_SLICES=1`;
- use `PIPELINE_REPLACE_SIMILARITY=1`;
- preserve historical seasons.

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

## CI/CD Target
Future automation should add:
- lint and syntax checks for API, pipeline, and frontend;
- frontend build;
- Docker image build;
- controlled VPS deploy over SSH;
- post-deploy health checks.

Until that exists, treat the manual deploy commands in this file as the release process.
