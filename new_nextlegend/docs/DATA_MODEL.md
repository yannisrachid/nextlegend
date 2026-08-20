# Data Model

The Postgres database is the serving layer for the API and frontend. The pipeline owns football data; the API owns application and workflow tables.

## Core Football Tables
- `competitions`: competition dimension.
- `seasons`: season dimension.
- `clubs`: club dimension.
- `players`: player dimension, including `tm_id` and `tm_profile_url`.
- `player_seasons`: central fact table for one player in one club/competition/season context.
- `player_metrics`: wide metrics table keyed by `player_season_id`.
- `role_scores`: long role-fit table, one row per `player_season_id` and role profile.
- `player_similarity`: top-k neighbors per profile.
- `pipeline_runs`: batch run tracking and monitoring.

High-level relationships:
```text
competitions ┐
seasons      ├─ player_seasons ─ player_metrics
clubs        ┘          │
players ────────────────┘
                         ├─ role_scores
                         └─ player_similarity(player_a_season_id/player_b_season_id)
```

## Application Tables
- `auth_users`, `auth_sessions`: login and HttpOnly session model.
- `prospects`: selected players.
- `club_needs`: recruitment needs.
- `club_need_players`: players attached to needs.
- `ai_conversations`, `ai_messages`: AI assistant history and payloads.
- `hq_priority_items`: agency HQ Kanban/calendar priorities.
- `hd_players`: represented HD Sports players and operational notes.
- `hd_player_documents`: document metadata attached to HD players.
- `player_transfer_history`: Wyscout transfer history imported for player reports and rooms.

Several application tables are created lazily by `apps/api/main.py` on first route use. The production DB user must be allowed to `CREATE TABLE`, `ALTER TABLE`, and `CREATE INDEX`.

## Key Fields
`player_seasons` is the main API surface. It includes:
- identity/context: `player_id`, `club_id`, `competition_id`, `season_id`;
- football context: position, age, minutes, team/league labels;
- scoring: `assigned_role`, `global_score_adjusted`, `assigned_role_pct_global`, `assigned_role_pct_league`;
- Transfermarkt: `tm_*` fields such as profile, value, contract, image, nationality, and external links.

`player_metrics` is intentionally wide. It stores raw Wyscout metrics plus derived percentiles. Add metric columns here only when they are needed by ranking, report, projection, comparison, visualization, AI filtering, or future scoring.

`role_scores` is the source of truth for role fit lists. For the assigned role, `role_scores.pct_global` must align with `player_seasons.global_score_adjusted`.

`player_similarity` is rebuilt by the pipeline. Use `PIPELINE_REPLACE_SIMILARITY=1` when regenerating it to avoid duplicates.

## Data Ownership
- Pipeline owns: dimensions, `player_seasons`, `player_metrics`, `role_scores`, `player_similarity`, `pipeline_runs`.
- API owns: auth, prospects, needs, AI history, HQ/HD workspace tables, transfer history import tables.
- Frontend must not invent persisted state that bypasses the API.

## Data Model Improvement Rules
- Prefer explicit foreign keys and indexes for API query paths.
- Keep historical seasons. Do not run destructive table replacement in production unless the goal is a full rebuild.
- Avoid storing duplicated display labels when a dimension join is reliable; keep labels only when they preserve imported source context or simplify hot API paths.
- Add columns with migrations or lazy schema code that is idempotent.
- For workflow tables, prefer `archived_at` or status fields over hard deletes when users need operational history.
- For auditability, preserve `created_at`, `updated_at`, and user/author fields on user-managed entities.
- For external data, keep both stable IDs and source URLs where available.

## Production Checks
```sql
SELECT
  (SELECT count(*) FROM player_seasons) AS player_seasons,
  (SELECT count(*) FROM player_metrics) AS player_metrics,
  (SELECT count(*) FROM role_scores) AS role_scores,
  (SELECT count(*) FROM player_similarity) AS player_similarity;

SELECT id, run_id, status, rows_processed, source_uri, started_at, ended_at
FROM pipeline_runs
ORDER BY id DESC
LIMIT 5;
```

Expected invariants:
- `player_seasons`, `player_metrics`, and `role_scores` are non-empty after a successful load.
- `player_similarity` is non-empty when `SIM_TOPK > 0`.
- Latest `pipeline_runs.status` should be `success` after a refresh.
- Auth-required API endpoints return 401 without `nl_session`.
