# NextLegend Docs

This folder is the compact project documentation for Codex sessions and human maintenance.

Read order:
1. `docs/skill.MD` - code conventions, project invariants, and implementation rules.
2. `docs/DATA_MODEL.md` - serving database model and table ownership.
3. `docs/SCORING_MODEL_WORKSHOP.md` - workshop template for rebuilding position-group scoring.
4. `docs/SCORING_MODEL_V2_IMPLEMENTATION.md` - implemented scoring v2, DB cleanup, local/prod rollout, and refresh policy.
5. `docs/TRANSFERMARKT_REFRESH.md` - monthly Transfermarkt refresh, matching, snapshots, and review flow.
6. `docs/VPS_CICD.md` - VPS, Docker, deployment, refresh, and CI/CD policy.
7. `docs/CRM_INTEGRATION.md` - CRM model, Neon migration, local and prod verification.
8. `docs/PROJECT_HISTORY.md` - useful project history and current product context.

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
- Scoring v2 uses position groups, not tactical role assignment. Legacy `assigned_role` fields now contain the position group for API compatibility.
- Transfermarkt identity, market-value snapshots, and matching decisions are stored in dedicated `transfermarkt_*` / `player_transfermarkt_matches` tables, then propagated to `tm_*` compatibility fields on `player_seasons` and `players`.
- Do not run the full raw pipeline on the current VPS; use local-compute then PRD-load.

Quick commands:
```bash
# dev stack
docker compose --env-file .env -f infra/compose/docker-compose.yml up -d

# dev pipeline refresh from data/wyscout_players_final.csv
docker compose --env-file .env -f infra/compose/docker-compose.yml run --rm pipeline-refresh

# monthly Transfermarkt refresh from helpers/csv/transfermarkt_profiles.csv
./scripts/run_transfermarkt_monthly_refresh.sh

# prod health checks
curl -I https://api.nextlegend.fr/
curl -s -o /tmp/api-health.out -w "%{http_code}\n" https://api.nextlegend.fr/health
```

Main code entry points:
- API routes and lazy schemas: `apps/api/main.py`.
- API settings: `apps/api/settings.py`.
- Front auth guard: `apps/frontend/lib/auth.js`, `apps/frontend/pages/_app.js`.
- Pipeline processing: `jobs/pipeline/pipeline/processing.py`.
- Pipeline DB upserts: `jobs/pipeline/pipeline/db.py`.
