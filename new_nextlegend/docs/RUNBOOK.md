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

## 3) Steady-state operations (no manual CSV copy)
Current production mode is steady-state:
- no recurring manual CSV transfer,
- no recurring historical backfill.

One-time historical backfill was completed during VPS initialization.
Keep `scripts/backfill_vps_from_existing_csvs.sh` only for exceptional recovery scenarios.

## 4) Weekly cron for current season (2025/2026 only)
The current-season pipeline runs weekly on Monday at 07:00 server time.

Cron entry:
```cron
0 7 * * 1 cd /home/yannis/nextlegend/new_nextlegend/wyscout_current_season_job && /usr/bin/flock -n /tmp/nextlegend_current_season.lock DOCKER_COMPOSE_FILE=/home/yannis/nextlegend/new_nextlegend/infra/compose/docker-compose-prod.yml SKIP_EMAIL_ALERTS=1 PIPELINE_REPLACE_INPUT_SLICES=1 PIPELINE_REPLACE_SIMILARITY=1 ./run_current_season_e2e.sh >> logs/cron_current_season.log 2>&1
```

Install/update without duplicates:
```bash
(crontab -l 2>/dev/null | grep -v 'nextlegend_current_season.lock'; echo '0 7 * * 1 cd /home/yannis/nextlegend/new_nextlegend/wyscout_current_season_job && /usr/bin/flock -n /tmp/nextlegend_current_season.lock DOCKER_COMPOSE_FILE=/home/yannis/nextlegend/new_nextlegend/infra/compose/docker-compose-prod.yml SKIP_EMAIL_ALERTS=1 PIPELINE_REPLACE_INPUT_SLICES=1 PIPELINE_REPLACE_SIMILARITY=1 ./run_current_season_e2e.sh >> logs/cron_current_season.log 2>&1') | crontab -
```

Verify:
```bash
crontab -l
```

Important:
- Use `flock -n` (with a space).
- `flock-n` is invalid and must be removed if present.

## 5) Monitoring (no SMTP required)
Monitoring is done in Home page:
- Frontend Home shows `Pipeline status`.
- It also indicates whether current season (`2025/2026` or `2026`) is loaded.

Operational logs:
- `wyscout_current_season_job/logs/current_season_e2e_*.log`
- `wyscout_current_season_job/logs/cron_current_season.log`

Manual run on VPS:
```bash
cd ~/nextlegend/new_nextlegend/wyscout_current_season_job
DOCKER_COMPOSE_FILE=/home/yannis/nextlegend/new_nextlegend/infra/compose/docker-compose-prod.yml SKIP_EMAIL_ALERTS=1 PIPELINE_REPLACE_INPUT_SLICES=1 PIPELINE_REPLACE_SIMILARITY=1 ./run_current_season_e2e.sh
```

## 6) Maintenance
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
```bash
cd ~/nextlegend/new_nextlegend
sudo docker compose --env-file .env -f infra/compose/docker-compose-prod.yml exec db \
  psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "TRUNCATE player_similarity;"

sudo docker compose --env-file .env -f infra/compose/docker-compose-prod.yml run --rm \
  -e PIPELINE_REPLACE_SIMILARITY=1 pipeline-refresh
```

7) Pipeline OOM mitigation
- Stop `api` and `frontend`, add swap, rerun pipeline.
- Restart `api` and `frontend` after pipeline success.
