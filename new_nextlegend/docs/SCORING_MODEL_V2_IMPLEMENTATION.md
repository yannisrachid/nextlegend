# Scoring Model v2 Implementation

This document describes the implemented scoring v2 change. The source design is `docs/Next_Legend_Scoring_Model_v2.md`.

## Objective

The old model scored many tactical roles and then selected an `assigned_role`. The v2 model scores one modern position group per player-season:

- Goalkeepers
- Centre Backs
- Left Backs
- Right Backs
- Defensive Midfielders
- Central Midfielders
- Attacking Midfielders
- Left Wingers
- Right Wingers
- Centre Forwards

The product keeps the legacy API field names for compatibility:

- `player_seasons.assigned_role` now stores the v2 position group display name.
- `player_seasons.global_score_adjusted` now stores the v2 final score.
- `player_seasons.assigned_role_pct_global` mirrors the v2 final score for legacy UI/API consumers.
- `player_seasons.assigned_role_pct_league` stores the player percentile inside the same competition and season.
- `role_scores.profile` stores the v2 position group, one row per player-season when the player has a mapped group.

## Algorithm

Implemented in `jobs/pipeline/pipeline/scoring_v2.py`.

Pipeline:

```text
metric_score = weighted_position_metrics(context_adjusted_metrics)
local_score = clamp(metric_score + profile_bonus + profile_penalty, 0, 100)
reliable_score = 50 + minutes_confidence * (local_score - 50)
projected_score = 50 + league_slope * (reliable_score - 50) + league_shift
scout_score = 50 + 1.15 * (projected_score - 45)
final_score = clamp(
  scout_score
  + competition_modifier
  + club_strength_modifier
  + minutes_regularity_modifier
  + production_bonus
  + discipline_modifier
  + previous_season_modifier,
  50,
  competition_cap
)
```

Key choices:

- Metrics are normalized against the relevant position group, not against all players.
- Normalization is robust: P2/P98 winsorization, median/MAD z-score, logistic conversion to 0-100.
- Volume metrics are adjusted by team context:
  - attacking volume is lightly penalized in dominant teams;
  - build-up volume is moderately penalized in dominant possession teams;
  - defensive volume is boosted in dominant teams because they face fewer defensive actions.
- Minutes are a confidence coefficient, not a skill.
- Early-season confidence also uses the maximum minutes observed in the same competition and season. A player with a high share of available minutes in August/September can be trusted earlier than a fixed full-season threshold would allow.
- League translation uses the existing `mercato_league_levels.json` as an initial slope/shift/cap source. Top reference leagues can reach 99; lower levels have a lower final cap.
- Goals and assists produce a cross-position `production_bonus`. This is intentionally strong, but capped, because end product matters for every modern role.
- `club_strength_modifier` rewards performance inside strong team contexts after volume metrics have already been context-adjusted.
- `minutes_regularity_modifier` rewards sustained availability and usage. Regularity is treated as a quality signal.
- `previous_season_modifier` uses the previous season score when the previous season is present in the processed dataset.
- Bonus/malus is capped and only applies to elite traits or critical weaknesses. It does not recreate tactical roles.

## Metric Hygiene

The scoring implementation only uses clean snake_case metrics. Examples:

- `xg` means total xG.
- `xg_per_90` means xG per 90 minutes.
- `xa` means total xA.
- `xa_per_90` means xA per 90 minutes.
- Percent fields use `_percent`, for example `accurate_passes_percent`.
- Rate fields use `_per_90`, for example `progressive_passes_per_90`.

`player_metrics` is pruned during refresh by default with `PIPELINE_PRUNE_METRIC_COLUMNS=1`. This removes stale role-score and percentile explosion columns from the old model and keeps only current raw/scoring/reporting fields.

Metric percentiles are not stored as extra `player_metrics` columns. They are stored in two long-format tables:

- `player_metric_percentiles_global`: same season, same position group, all competitions.
- `player_metric_percentiles_league`: same season, same competition, same position group.

Each row stores `player_season_id`, `metric_key`, `raw_value`, `percentile`, `position_group`, `sample_size`, and `lower_is_better`. This keeps `player_metrics` compact while allowing the report page to load both league and global centiles for every displayed metric.

## Database Policy

`player_seasons` is the central fact table. The effective natural key is:

```text
player_id + competition_id + season_id + club_id
```

`player_metrics` is keyed by:

```text
player_season_id
```

Metric percentile tables are keyed by:

```text
player_season_id + metric_key
```

Weekly refreshes must be incremental:

- Do not truncate football tables by default.
- Upsert changed player-seasons and metrics.
- Keep existing player-season rows if a scraper refresh omits a player temporarily.
- Use `PIPELINE_REPLACE_INPUT_SLICES=0` by default for normal historical preservation.
- Use `PIPELINE_REPLACE_TABLES=0` by default.
- Only use destructive replacement for explicit full rebuilds.
- For current-season percentile quality, the weekly input must include the complete current-season slice, not only changed rows, because global percentiles are season-wide rankings.

## 2026/2027 Early-Season Detection

The model is designed to identify performers quickly when the new season has limited minutes.

Absolute confidence still protects against tiny samples:

- 90-179 minutes remains fragile.
- 180-449 minutes remains early evidence.
- 450+ minutes becomes increasingly reliable.

But v2 also computes `minutes / max_minutes_in_competition_season`:

- `>= 30%` of available minutes can reach `0.68` confidence.
- `>= 50%` can reach `0.78`.
- `>= 70%` can reach `0.86`.
- `>= 85%` can reach `0.92`.

This relative confidence is capped by the actual progress of the season:

- max competition minutes `< 90`: no relative boost.
- `>= 90`: cap `0.75`.
- `>= 180`: cap `0.84`.
- `>= 270`: cap `0.90`.
- `>= 450`: cap `0.96`.
- `>= 900`: cap `1.00`.

Final confidence is the maximum of absolute confidence and capped relative confidence. This means a player who has played almost everything available in early 2026/2027 can surface quickly, while a player with one short cameo cannot.

## Lower-Is-Better Metrics

`MetricSpec` supports `lower_is_better=True` for metrics where low values are positive. The current v2 core avoids over-weighting very contextual defensive/GK negatives such as goals conceded, xGA, and clean sheets.

Discipline is handled as a separate lower-is-better modifier:

- high `fouls_per_90` penalizes the final score;
- high `yellow_cards_per_90` penalizes the final score;
- high `red_cards_per_90` penalizes the final score more strongly;
- total discipline impact is capped at `-3`.

The local v2.2 cleanup reduced `player_metrics` from hundreds of historical derived columns to 150 columns in the dev DB.

## Audit Findings And v2.2 Calibration

Audit before v2.2:

- Scores were too compressed around `46-48`.
- Many strong-club players had good raw/projected scores but product scores were too low.
- Top defenders and goalkeepers were capped too low because the product layer did not re-expand projected scores after league/context translation.
- A first recalibration over-corrected: too many top-club players saturated at 99.

v2.2 fixes:

- Final score is a product/scout note bounded between `50` and `99`.
- `projected_score` remains stored for audit as the translated statistical score before product calibration.
- `competition_cap` prevents medium/weak leagues from reaching 99 only through volume and production.
- Production bonus is strong but capped at `+7`.
- Club strength, minutes regularity, and competition modifiers are explicit and stored in `player_metrics`.
- `role_scores` is rebuilt after final calibration, so percentiles match the visible score.

## Local Recalculation

Fast scoring-only historical recalculation, preserving existing Transfermarkt fields:

```bash
docker compose --env-file .env -f infra/compose/docker-compose.yml exec -T db \
  psql -U nextlegend -d nextlegend -c "TRUNCATE TABLE role_scores, player_similarity;"

for f in \
  data/wyscout_players_2022_2023_final.csv \
  data/wyscout_players_2023_2024_final.csv \
  data/wyscout_players_2024_2025_final.csv \
  data/wyscout_players_2025_final.csv \
  data/wyscout_players_final.csv
do
  docker compose --env-file .env -f infra/compose/docker-compose.yml run --rm \
    -e TM_SKIP_ENRICH=1 \
    -e SIM_TOPK=0 \
    -e PIPELINE_REPLACE_TABLES=0 \
    -e PIPELINE_REPLACE_INPUT_SLICES=0 \
    -e PIPELINE_REPLACE_SIMILARITY=0 \
    -e PIPELINE_PRUNE_METRIC_COLUMNS=1 \
    pipeline --input-uri "/$f" --bucket ""
done
```

Then rebuild similarity separately when scoring has been validated:

```bash
docker compose --env-file .env -f infra/compose/docker-compose.yml run --rm \
  -e TM_SKIP_ENRICH=1 \
  -e PIPELINE_REPLACE_TABLES=0 \
  -e PIPELINE_REPLACE_INPUT_SLICES=0 \
  -e PIPELINE_REPLACE_SIMILARITY=1 \
  pipeline --input-uri "/data/wyscout_players_final.csv" --bucket ""
```

## Verification Queries

```sql
SELECT
  (SELECT count(*) FROM player_seasons) AS player_seasons,
  (SELECT count(*) FROM player_metrics) AS player_metrics,
  (SELECT count(*) FROM player_metric_percentiles_global) AS metric_pct_global,
  (SELECT count(*) FROM player_metric_percentiles_league) AS metric_pct_league,
  (SELECT count(*) FROM role_scores) AS position_group_scores,
  (SELECT count(*) FROM player_similarity) AS player_similarity;

SELECT count(*) AS player_metrics_columns
FROM information_schema.columns
WHERE table_schema = 'public'
  AND table_name = 'player_metrics';

SELECT assigned_role, count(*)
FROM player_seasons
GROUP BY assigned_role
ORDER BY count DESC;

SELECT
  min(global_score_adjusted) AS min_score,
  round(avg(global_score_adjusted)::numeric, 2) AS avg_score,
  max(global_score_adjusted) AS max_score
FROM player_seasons
WHERE global_score_adjusted IS NOT NULL;

SELECT metric_key, count(*) AS rows, min(percentile), max(percentile)
FROM player_metric_percentiles_global
GROUP BY metric_key
ORDER BY rows DESC
LIMIT 20;
```

Local dev result after migration:

```text
player_seasons: 192842
player_metrics: 192529
metric_pct_global: 5942359
metric_pct_league: 5942359
position_group_scores: 192679
player_similarity: 457970
player_metrics_columns: 150
unmapped rows: 163, all with position=<NA>
score distribution after final v2.2 pass:
  min: 50
  avg: 55.86
  p50: 53.13
  p90: 66.09
  p99: 79.16
  max: 99
  score_99_count: 35 / 192842 player-seasons
competition caps:
  top reference leagues: 99
  Netherlands / Portugal / Belgium top divisions: 91.70
  Brazil Serie A / MLS: 89.84
  Championship: 87.98
  Ligue 2 / Serie B / Segunda: 85.19
```

`player_similarity` was rebuilt with `SIM_TOPK=10`. Local validation confirms `max_similarities_per_source=10` and `sources_over_10=0`.

`previous_season_score` was available for 2015 local metric rows. For future 2026/2027 refreshes, if the scraper CSV contains only the current season, the pipeline should load the previous score from the existing DB before upsert to make this signal more complete.

## Production Rollout

Recommended production sequence:

1. Deploy code with `scoring_v2.py`, incremental refresh defaults, and frontend label changes.
2. Back up the production database.
3. Run the historical scoring-only recalculation with `TM_SKIP_ENRICH=1` and `SIM_TOPK=0`.
4. Run the verification queries above.
5. Validate top players per position group in the UI.
6. Rebuild `player_similarity` after validation with `SIM_TOPK=10`.
7. Run the normal weekly current-season Docker job with default incremental flags.

Do not run `PIPELINE_REPLACE_TABLES=1` or `PIPELINE_REPLACE_INPUT_SLICES=1` in production unless a full rebuild is explicitly planned.

## Report UI And PNG Exports

The player `Report` page now consumes the v2 position-group model directly.

Frontend entry points:

- `apps/frontend/pages/report.js`
- `apps/frontend/components/report/PlayerReportComponents.js`
- `apps/frontend/lib/reportMetrics.js`
- `apps/frontend/lib/percentileColors.js`

Report principles:

- The main rating is `player_seasons.global_score_adjusted`; the frontend must not recalculate it.
- Percentiles come from the API payload built from `player_metric_percentiles_global` and `player_metric_percentiles_league`.
- `global` context means same season, same position group, all competitions.
- `league` context means same season, same competition, same position group.
- Strengths and weaknesses are derived only from metrics relevant to the player's position group.
- Missing values are displayed as `-` and are never coerced to zero.
- Similar players are limited to the top 10 statistical neighbors and displayed without similarity percentage because the ranking is already constrained server-side.

The page supports two PNG exports:

- `Scout PNG`: short external-facing snapshot for a scout. It highlights context, score, KPI, position, strongest relevant traits, risk check, radar benchmark, and market context.
- `Full PNG`: extended version that adds more metrics, statistical neighbors, Transfermarkt context, and transfer history.

The exports are generated client-side with canvas from the already-loaded report data. They are not screenshots, so they remain stable and shareable even if the responsive layout changes.

## One-Off DEV DB Promotion To PROD

For this release, a full DEV database copy to PROD is acceptable because DEV contains the validated scoring v2 data, metric percentile tables, CRM data, Transfermarkt enrichments, and report dependencies.

This is destructive for the production database. Always keep a timestamped PROD backup before restore.

Local DEV dump:

```bash
cd /Users/yannis/ylfc/new_nextlegend
mkdir -p tmp/db_dumps
docker compose --env-file .env -f infra/compose/docker-compose.yml exec -T db \
  sh -c 'pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" --clean --if-exists --no-owner --no-privileges' \
  > tmp/db_dumps/nextlegend_dev_for_prod_$(date +%Y%m%d_%H%M%S).sql
```

Copy dump to VPS:

```bash
rsync -avz tmp/db_dumps/nextlegend_dev_for_prod_*.sql yannis@nextlegend-prod:~/nextlegend/backups/
```

On the VPS, back up PROD before restore:

```bash
ssh yannis@nextlegend-prod
cd ~/nextlegend/new_nextlegend
mkdir -p ~/nextlegend/backups
docker compose --env-file .env -f infra/compose/docker-compose-prod.yml exec -T db \
  sh -c 'pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" --no-owner --no-privileges' \
  > ~/nextlegend/backups/nextlegend_prod_before_scoring_v2_$(date +%Y%m%d_%H%M%S).sql
```

Restore DEV dump into PROD:

```bash
cd ~/nextlegend/new_nextlegend
LATEST_DEV_DUMP=$(ls -1t ~/nextlegend/backups/nextlegend_dev_for_prod_*.sql | head -n 1)
docker compose --env-file .env -f infra/compose/docker-compose-prod.yml exec -T db \
  sh -c 'psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB"' \
  < "$LATEST_DEV_DUMP"
```

Deploy code and restart runtime:

```bash
cd ~/nextlegend/new_nextlegend
git fetch origin
git checkout main
git pull --ff-only origin main
docker compose --env-file .env -f infra/compose/docker-compose-prod.yml up -d --build db minio api frontend caddy
```

Post-restore checks:

```bash
docker compose --env-file .env -f infra/compose/docker-compose-prod.yml exec -T db \
  psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "
SELECT
  (SELECT count(*) FROM player_seasons) AS player_seasons,
  (SELECT count(*) FROM player_metrics) AS player_metrics,
  (SELECT count(*) FROM player_metric_percentiles_global) AS metric_pct_global,
  (SELECT count(*) FROM player_metric_percentiles_league) AS metric_pct_league,
  (SELECT count(*) FROM role_scores) AS role_scores,
  (SELECT count(*) FROM player_similarity) AS player_similarity;
"

curl -I https://api.nextlegend.fr/health
curl -sk -o /tmp/front.out -w "%{http_code}\n" https://app.nextlegend.fr
```

Rollback:

```bash
cd ~/nextlegend/new_nextlegend
LATEST_PROD_BACKUP=$(ls -1t ~/nextlegend/backups/nextlegend_prod_before_scoring_v2_*.sql | head -n 1)
docker compose --env-file .env -f infra/compose/docker-compose-prod.yml exec -T db \
  sh -c 'psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB"' \
  < "$LATEST_PROD_BACKUP"
docker compose --env-file .env -f infra/compose/docker-compose-prod.yml restart api frontend
```

## Scraping Job Split

When the scraping job is moved to a separate repository, keep this contract:

- Output one clean CSV with stable snake_case columns.
- Preserve `wyscout_id`, `competition_name`, `calendar`, and `team_in_selected_period`.
- Keep totals and per-90 metrics separate.
- Do not generate scoring columns in the scraper repository.
- The main NextLegend pipeline remains responsible for DB upsert, scoring, similarity, and serving artifacts.
