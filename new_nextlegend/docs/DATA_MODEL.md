# Data Model

The Postgres database is the serving layer for the API and frontend. The pipeline owns football data; the API owns application and workflow tables.

## Core Football Tables
- `competitions`: competition dimension.
- `seasons`: season dimension.
- `clubs`: club dimension.
- `players`: player dimension, including `tm_id` and `tm_profile_url`.
- `player_seasons`: central fact table for one player in one club/competition/season context.
- `player_metrics`: wide metrics table keyed by `player_season_id`.
- `player_metric_percentiles_global`: long-format metric percentiles versus the same season and position group across all competitions.
- `player_metric_percentiles_league`: long-format metric percentiles versus the same season, competition, and position group.
- `role_scores`: legacy compatibility table for v2 position-group scores, one row per `player_season_id` and position group.
- `player_similarity`: top-10 neighbors per profile.
- `pipeline_runs`: batch run tracking and monitoring.

High-level relationships:
```text
competitions ┐
seasons      ├─ player_seasons ─ player_metrics
clubs        ┘          │
players ────────────────┘
                         ├─ role_scores
                         ├─ player_metric_percentiles_global
                         ├─ player_metric_percentiles_league
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
- `crm_clubs`, `crm_players`, `crm_contacts`, `crm_prospects`: integrated football CRM imported from `findyourlegend`; see `docs/CRM_INTEGRATION.md`.

Several application tables are created lazily by `apps/api/main.py` on first route use. The production DB user must be allowed to `CREATE TABLE`, `ALTER TABLE`, and `CREATE INDEX`.

## Key Fields
`player_seasons` is the main API surface. It includes:
- identity/context: `player_id`, `club_id`, `competition_id`, `season_id`;
- football context: position, age, minutes, team/league labels;
- scoring: `assigned_role`, `global_score_adjusted`, `assigned_role_pct_global`, `assigned_role_pct_league`;
- Transfermarkt: `tm_*` fields such as profile, value, contract, image, nationality, and external links.

`player_metrics` stores clean Wyscout metrics plus the compact v2 scoring breakdown. It must not accumulate stale role-score or percentile explosion columns. The pipeline prunes obsolete metric columns by default with `PIPELINE_PRUNE_METRIC_COLUMNS=1`.

Metric percentiles are stored outside `player_metrics` to keep the fact table compact:
- `player_metric_percentiles_global`: same season, same position group, all competitions.
- `player_metric_percentiles_league`: same season, same competition, same position group.
- Both tables are keyed by `player_season_id + metric_key`.
- Lower-is-better metrics, such as cards and fouls, are inverted before percentile ranking and flagged with `lower_is_better=true`.

`role_scores` keeps its historical name for API compatibility. In scoring v2, `role_scores.profile` is the position group. For the assigned position group, `role_scores.pct_global_adjusted` must align with `player_seasons.global_score_adjusted`.

`player_similarity` is rebuilt by the pipeline. Use `PIPELINE_REPLACE_SIMILARITY=1` when regenerating it to avoid duplicates. Dev and prod should store only the top 10 most similar players per player-season.

The effective natural key of `player_seasons` is `player_id + competition_id + season_id + club_id`. `player_metrics` is keyed by `player_season_id`. Weekly refreshes should upsert changed rows and preserve player-season rows that are temporarily missing from a scraper run. Percentile recalculation requires the refreshed current-season slice to be complete enough for the target season; do not compute season-wide percentiles from a partial changed-row-only input.

## Data Ownership
- Pipeline owns: dimensions, `player_seasons`, `player_metrics`, `player_metric_percentiles_global`, `player_metric_percentiles_league`, `role_scores`, `player_similarity`, `pipeline_runs`.
- API owns: auth, prospects, needs, AI history, HQ/HD workspace tables, transfer history import tables.
- API owns CRM tables: `crm_clubs`, `crm_players`, `crm_contacts`, `crm_prospects`.
- Frontend must not invent persisted state that bypasses the API.

## Data Model Improvement Rules
- Prefer explicit foreign keys and indexes for API query paths.
- Keep historical seasons. Do not run destructive table replacement in production unless the goal is a full rebuild.
- Keep weekly refreshes incremental: default `PIPELINE_REPLACE_TABLES=0` and `PIPELINE_REPLACE_INPUT_SLICES=0`.
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
  (SELECT count(*) FROM player_metric_percentiles_global) AS metric_pct_global,
  (SELECT count(*) FROM player_metric_percentiles_league) AS metric_pct_league,
  (SELECT count(*) FROM role_scores) AS role_scores,
  (SELECT count(*) FROM player_similarity) AS player_similarity;

SELECT id, run_id, status, rows_processed, source_uri, started_at, ended_at
FROM pipeline_runs
ORDER BY id DESC
LIMIT 5;
```

Expected invariants:
- `player_seasons`, `player_metrics`, `player_metric_percentiles_global`, `player_metric_percentiles_league`, and `role_scores` are non-empty after a successful load.
- `player_similarity` is non-empty when `SIM_TOPK > 0`, with at most 10 rows per source player-season.
- Latest `pipeline_runs.status` should be `success` after a refresh.
- Auth-required API endpoints return 401 without `nl_session`.
