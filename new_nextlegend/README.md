# NextLegend v2

This repository contains the production-ready NextLegend v2 stack:
- Frontend: Next.js
- Backend: FastAPI
- Database: Postgres
- Batch pipeline: Docker job
- Object storage: external S3

The docs are written to be self-contained for Codex: you should be able to understand and operate the project by reading this file + `apps/api/README.md`.

## High-level architecture
- `apps/frontend`: Next.js UI
- `apps/api`: FastAPI API
- `jobs/pipeline`: batch ingestion + enrichment job
- `infra/compose`: docker-compose definitions
- `helpers/`: reference files (Transfermarkt, league mappings, etc.)

Runtime:
- API and frontend are served on different subdomains.
- Auth uses a HttpOnly session cookie (`nl_session`).
- The frontend runs a client-side auth guard with explicit states (loading/authenticated/unauthenticated).

## Quick start (dev)
```bash
cp .env.example .env
sudo docker compose --env-file .env -f infra/compose/docker-compose.yml up --build
```
- Frontend: http://localhost:3000
- API: http://localhost:8000/health
- Postgres: localhost:${POSTGRES_PORT:-5432}

## Environment variables
Database:
- `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_PORT`
- `DATABASE_URL`

S3 (external):
- `S3_ENDPOINT`, `S3_BUCKET`, `S3_ACCESS_KEY`, `S3_SECRET_KEY`

API / Front:
- `API_BASE_URL`
- `NEXT_PUBLIC_API_BASE_URL`
- `CORS_ORIGINS`

Auth:
- `AUTH_SESSION_DAYS`
- `AUTH_COOKIE_SECURE`
- `AUTH_USERS_JSON` or `AUTH_USERNAME`/`AUTH_PASSWORD` (bootstrap)

AI:
- `OPENAI_API_KEY`

## Auth model (frontend + backend)
- Backend: global auth middleware protects all routes except `GET /`, `GET /health`, `POST /auth/login`, `POST /auth/logout`, `GET /auth/me`.
- Frontend: client-side guard waits for `/auth/me` before redirecting. No edge middleware redirect.
- Session cookie is HttpOnly and must be scoped to the correct domain.

## API (FastAPI)
- `GET /` : public, returns `{ "status": "ok" }`.
- `GET /health` : public healthcheck.
- All other endpoints require an active session (`nl_session`).

## Pipeline (batch)
Pipeline flow:
1) Download raw CSV from S3.
2) Normalize + clean.
3) Transfermarkt enrichment (club mapping + player mapping + fuzzy fallback).
4) Scores + percentiles + similarities.
5) Archive artifacts to S3 and upsert Postgres.

Run:
```bash
sudo docker compose --env-file .env -f infra/compose/docker-compose.yml run --rm pipeline-refresh
```

Important pipeline flags:
- `SIM_TOPK` : top-k similarities per profile (default 30)
- `PIPELINE_INPUT_URI` : input CSV (S3 or local)
- `PIPELINE_INPUT_KIND=raw` (default) or `enriched`
- `PIPELINE_REPLACE_SIMILARITY=1` : replace similarity table
- `TM_SKIP_ENRICH=1` : skip TM (debug)
- `TM_ENABLE_FUZZY=0/1` : toggle fuzzy TM
- `TM_CLUB_LOG_EVERY`, `TM_FUZZY_LOG_EVERY` : progress logs

Transfermarkt reference files (`helpers/csv`):
- `transfermarkt_profiles.csv`
- `player_matching_reference.csv`
- `club_mapping_dict.py`
- `club_matching_reference.csv`
- `tm_clubs_reference.csv`

## S3 archives
- `s3://$S3_BUCKET/new_nextlegend/enriched/<run_id>_<timestamp>/...`
- Artifacts: `raw`, `enriched`, `competitions`, `seasons`, `players`, `clubs`, `player_seasons`, `player_metrics`, `role_scores`, `player_similarity`

## Database (Postgres)
Core tables:
- `competitions`, `seasons`, `clubs`, `players`
- `player_seasons` (fact)
- `player_metrics` (metrics + percentiles)
- `role_scores`, `player_similarity`
- `pipeline_runs`

App tables:
- `prospects`, `club_needs`, `club_need_players`
- `ai_conversations`, `ai_messages`
- `auth_users`, `auth_sessions`

## Operations / checks
- `GET /health` (API)
- `GET /` (API root public)
- `docker compose ... logs api|frontend|pipeline`

---

# Production Runbook (detailed)

This section is the canonical prod setup for `app.nextlegend.fr` (frontend) and `api.nextlegend.fr` (API) behind Caddy.

## 1) Domain + DNS
- Buy a domain (OVH, Gandi, Infomaniak, Cloudflare Registrar).
- Create A records:
  - `app.nextlegend.fr` -> VPS IP
  - `api.nextlegend.fr` -> VPS IP

## 2) Caddy (HTTPS)
Install Caddy on the VPS and configure:

`/etc/caddy/Caddyfile`
```
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

Caddy will automatically provision TLS certs for both subdomains.

## 3) Docker compose (prod)
Make sure containers are running:
```bash
sudo docker compose --env-file .env -f infra/compose/docker-compose.yml up -d
```

## 4) Env for prod
Recommended `.env` values:
- `NEXT_PUBLIC_API_BASE_URL=https://api.nextlegend.fr`
- `API_BASE_URL=https://api.nextlegend.fr`
- `CORS_ORIGINS=["https://app.nextlegend.fr"]`
- `AUTH_COOKIE_SECURE=true`

Optional:
- `AUTH_SESSION_DAYS=365`

## 5) Auth cookie scope
The session cookie is set by the API domain. This is expected because the frontend communicates with the API via `https://api.nextlegend.fr`.
Do not attempt to scope it to `app.`.

## 6) Health checks
```bash
curl -I https://api.nextlegend.fr/
curl -I https://api.nextlegend.fr/health
```
Both should return 200.

## 7) Login check
- Open `https://app.nextlegend.fr/login`
- Login -> land on `/` and stay (no reload loop)
- Refresh -> stay logged in

## 8) Weekly data refresh
```bash
sudo docker compose --env-file .env -f infra/compose/docker-compose.yml run --rm pipeline-refresh
```

## 9) Optional: stop MinIO (if using external S3)
MinIO is not needed for prod if you use external S3.
```bash
sudo docker compose --env-file .env -f infra/compose/docker-compose.yml stop minio
sudo docker compose --env-file .env -f infra/compose/docker-compose.yml rm -f minio
```

---

# Maintenance Runbook

## 1) Update code
```bash
git pull
sudo docker compose --env-file .env -f infra/compose/docker-compose.yml up -d --build
```

## 2) Restart specific services
```bash
sudo docker compose --env-file .env -f infra/compose/docker-compose.yml restart api frontend
```

## 3) Logs
```bash
sudo docker compose --env-file .env -f infra/compose/docker-compose.yml logs -f api
sudo docker compose --env-file .env -f infra/compose/docker-compose.yml logs -f frontend
sudo docker compose --env-file .env -f infra/compose/docker-compose.yml logs -f pipeline
```

## 4) Postgres backup (manual)
```bash
sudo docker compose --env-file .env -f infra/compose/docker-compose.yml exec db \
  pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" > backup.sql
```

## 5) Postgres restore (manual)
```bash
cat backup.sql | sudo docker compose --env-file .env -f infra/compose/docker-compose.yml exec -T db \
  psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"
```

## 6) Clear and rebuild similarity table
```bash
sudo docker compose --env-file .env -f infra/compose/docker-compose.yml exec db \
  psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "TRUNCATE player_similarity;"

sudo docker compose --env-file .env -f infra/compose/docker-compose.yml run --rm \
  -e PIPELINE_REPLACE_SIMILARITY=1 pipeline-refresh
```

## 7) Health + auth checks
```bash
curl -I https://api.nextlegend.fr/
curl -I https://api.nextlegend.fr/health
```

## 8) Pipeline troubleshooting
- If the pipeline is OOM killed: stop `api` + `frontend` temporarily.
- Enable swap if needed.
- Use TM progress logs: `TM_CLUB_LOG_EVERY`, `TM_FUZZY_LOG_EVERY`.

## 9) Caddy reload
```bash
sudo systemctl reload caddy
```

## Documentation
- `README.md` (this file)
- `apps/api/README.md`
