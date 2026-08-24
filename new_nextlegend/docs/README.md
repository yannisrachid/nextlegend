# NextLegend Docs

This folder is the compact project documentation for Codex sessions and human maintenance.

Read order:
1. `docs/skill.MD` - code conventions, project invariants, and implementation rules.
2. `docs/DATA_MODEL.md` - serving database model and table ownership.
3. `docs/SCORING_MODEL_WORKSHOP.md` - workshop template for rebuilding position-group scoring.
4. `docs/VPS_CICD.md` - VPS, Docker, deployment, refresh, and CI/CD policy.
5. `docs/CRM_INTEGRATION.md` - CRM model, Neon migration, local and prod verification.
6. `docs/PROJECT_HISTORY.md` - useful project history and current product context.

Project snapshot:
- Frontend: Next.js in `apps/frontend`, served at `app.nextlegend.fr`.
- API: FastAPI in `apps/api`, served at `api.nextlegend.fr`.
- Database: Postgres.
- Batch pipeline: `jobs/pipeline`.
- Object storage: MinIO, S3-compatible.
- Current product: Next Legend for HD Sports scouting and agency operations.
- Integrated CRM: `/crm`, backed by `crm_*` tables and documented in `docs/CRM_INTEGRATION.md`.

Key invariants:
- API root `/` and `/health` are public and must return 200.
- Auth uses the HttpOnly cookie `nl_session`.
- Frontend auth waits for `GET /auth/me`; do not redirect while auth is loading.
- Pipeline writes `player_seasons`, `player_metrics`, `role_scores`, `player_similarity`, and `pipeline_runs`.
- Transfermarkt fields are stored as `tm_*` columns on `player_seasons` and `tm_id` / `tm_profile_url` on `players`.
- Do not run the full raw pipeline on the current VPS; use local-compute then PRD-load.

Quick commands:
```bash
# dev stack
docker compose --env-file .env -f infra/compose/docker-compose.yml up -d

# dev pipeline refresh from data/wyscout_players_final.csv
docker compose --env-file .env -f infra/compose/docker-compose.yml run --rm pipeline-refresh

# prod health checks
curl -I https://api.nextlegend.fr/
curl -I https://api.nextlegend.fr/health
```

Main code entry points:
- API routes and lazy schemas: `apps/api/main.py`.
- API settings: `apps/api/settings.py`.
- Front auth guard: `apps/frontend/lib/auth.js`, `apps/frontend/pages/_app.js`.
- Pipeline processing: `jobs/pipeline/pipeline/processing.py`.
- Pipeline DB upserts: `jobs/pipeline/pipeline/db.py`.
