# NextLegend v2 API (FastAPI)

This document is the canonical backend reference for Codex.

## Run locally (Docker)
```bash
cd new_nextlegend
sudo docker compose --env-file .env -f infra/compose/docker-compose.yml up --build api
```
API listens on `http://localhost:8000`.

## Public routes
- `GET /` : `{ "status": "ok" }`
- `GET /health`
- `POST /auth/login`
- `POST /auth/logout`
- `GET /auth/me`

These paths are explicitly excluded from the auth middleware.
All other routes require a valid session cookie.

## Auth model
- Cookie: `nl_session` (HttpOnly)
- Sessions stored in `auth_sessions` table
- User records in `auth_users`

## Key env vars
- `DATABASE_URL`
- `CORS_ORIGINS`
- `OPENAI_API_KEY`
- `AUTH_COOKIE_SECURE`, `AUTH_SESSION_DAYS`

## Core endpoints
- `GET /ranking`, `GET /ranking/page`
- `GET /players/{id}`
- `GET /players/{id}/report`
- `GET /players/{id}/similarities` (max 10 returned rows)
- `GET /meta/positions`, `GET /meta/teams`, `GET /meta/seasons`, `GET /meta/competitions`
- `GET /meta/stats-research/metrics`
- `GET /ops/metrics`
- `POST /ai/scout`, `POST /ai/player-report`
- `GET/POST/PATCH /ai/conversations`
- `GET/POST/PATCH/DELETE /admin/users`

## DB expectations
Tables populated by the pipeline:
- `competitions`, `seasons`, `players`, `player_seasons`, `player_metrics`, `player_metric_percentiles_global`, `player_metric_percentiles_league`, `role_scores`, `player_similarity`, `pipeline_runs`

## Notes
- Root `/` must remain public to avoid auth loops on frontend.
- If `/` returns 405, make sure the route supports both GET and HEAD.
