# NextLegend v2 Docs (Codex)

This folder is the primary entry point for new Codex sessions.

Read order:
1) `docs/README.md`
2) `docs/HANDOFF_2026-03-11.md` (latest local + VPS handoff)
3) `docs/RUNBOOK.md`
4) `docs/PIPELINE_PLAN.md`
5) `docs/DATA_MODEL.md`
6) `docs/NEXTLEGEND_V2_UX_UI.md`
7) `docs/AWS_DEPLOYMENT.md`
8) `docs/NextLegend_v2_Migration_Guide.md` (history and migration notes)

Project snapshot:
- Frontend: Next.js (`apps/frontend`) served at `app.nextlegend.fr`
- API: FastAPI (`apps/api`) served at `api.nextlegend.fr`
- DB: Postgres
- Batch pipeline: `jobs/pipeline` (loads S3 CSV -> DB)
- Object storage: external S3 (MinIO only for local dev if needed)

Key invariants:
- API root `/` is public and returns 200.
- Auth uses HttpOnly cookie `nl_session`; `GET /auth/me` returns 200 when authenticated.
- Frontend auth guard waits for `/auth/me`; no redirect while loading.
- Pipeline writes `player_metrics` and `player_similarity` and stores TM fields (`tm_*`) on `player_seasons`.

Quick commands:
```bash
# start dev stack
sudo docker compose --env-file .env -f infra/compose/docker-compose.yml up -d

# run pipeline (dev, local CSV at data/wyscout_players_final.csv)
sudo docker compose --env-file .env -f infra/compose/docker-compose.yml run --rm pipeline-refresh

# run pipeline (prod, S3 CSV)
sudo docker compose --env-file .env -f infra/compose/docker-compose-prod.yml build pipeline-refresh
sudo docker compose --env-file .env -f infra/compose/docker-compose-prod.yml run --rm -e PIPELINE_REPLACE_TABLES=1 pipeline-refresh
sudo docker compose --env-file .env -f infra/compose/docker-compose-prod.yml up -d --build api frontend

# health checks
curl -I https://api.nextlegend.fr/
curl -I https://api.nextlegend.fr/health
```

Where to look:
- Pipeline logic: `jobs/pipeline/pipeline/processing.py`
- DB upserts: `jobs/pipeline/pipeline/db.py`
- API routes: `apps/api/main.py`
- Front auth guard: `apps/frontend/lib/auth.js`, `apps/frontend/pages/_app.js`
