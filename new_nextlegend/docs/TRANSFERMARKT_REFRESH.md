# Transfermarkt Refresh And Matching

This document describes how Next Legend refreshes Transfermarkt identities and market values.

## External Repos

- `../transfermarkt-api`: scraper/API used to collect Transfermarkt profiles and market values.
- `../players-matcher`: reference implementation for robust player matching. Next Legend uses the same core idea: normalized names plus date-of-birth evidence and conservative bidirectional validation.

Do not vendor either repo into `new_nextlegend`. They are external producers/reference code.

## Transfermarkt API Surface

Swagger checked on September 3, 2026:

- `GET /competitions/search/{competition_name}`
- `GET /competitions/{competition_id}/clubs?season_id=...`
- `GET /clubs/search/{club_name}`
- `GET /clubs/{club_id}/profile`
- `GET /clubs/{club_id}/players?season_id=...`
- `GET /players/search/{player_name}`
- `GET /players/{player_id}/profile`
- `GET /players/{player_id}/market_value`
- `GET /players/{player_id}/transfers`
- `GET /players/{player_id}/stats`
- `GET /players/{player_id}/injuries`
- `GET /players/{player_id}/achievements`

For monthly valuation refreshes, prefer club/competition driven scraping:

1. map Wyscout competitions to Transfermarkt competition IDs;
2. scrape clubs for each mapped competition and season;
3. scrape club players;
4. enrich missing details through player profile/market-value endpoints.

## Current Mapping Problem

The previous flow wrote Transfermarkt fields directly into `players` and `player_seasons`:

- `players.tm_id`
- `players.tm_profile_url`
- dynamic `player_seasons.tm_*` columns

This made monthly market-value refreshes hard to audit because a changed match could silently overwrite visible app data.

## New Serving Tables

`transfermarkt_players`

Stable Transfermarkt identity table keyed by `tm_player_id`.

Important fields:
- `name`
- `profile_url`
- `profile_image_url`
- `birth_date`
- `club_id`
- `club_name`
- `position_main`
- `citizenship`
- `agent_name`
- `market_value_eur`
- `raw_payload`

`transfermarkt_market_value_snapshots`

Monthly market-value history.

Natural key:

```text
tm_player_id + snapshot_date
```

`player_transfermarkt_matches`

Auditable Wyscout to Transfermarkt decision table.

Important fields:
- `player_id`
- `wyscout_id`
- `tm_player_id`
- `confidence_score`
- `score_margin`
- `method`
- `status`: `accepted`, `review`, or `rejected`
- `is_primary`
- `evidence`

Only `accepted` + `is_primary=true` rows are propagated to `players` and `player_seasons`.

## Matching Rules

The matcher scores:

- normalized player name;
- exact date of birth when available;
- birth year when Wyscout only exposes a year/age label;
- age compatibility as fallback;
- club-name compatibility;
- country/citizenship compatibility;
- position compatibility.

Automatic acceptance is intentionally conservative:

- best Wyscout candidate for a Transfermarkt player;
- best Transfermarkt candidate for a Wyscout player;
- score above `TM_AUTO_ACCEPT_SCORE`, default `0.90`;
- margin above `TM_AUTO_ACCEPT_MARGIN`, default `0.035`;
- existing accepted links are preserved when the Transfermarkt ID still exists in the new file.

Anything plausible but not safe enough remains `review`.

## Expected Transfermarkt CSV

Default path:

```text
helpers/csv/transfermarkt_profiles.csv
```

Curated player mapping, when available:

```text
helpers/csv/player_matching_reference.csv
```

The loader accepts common column variants, but this format is preferred:

```text
player_id,player_name,profile_url,profile_image_url,birth_date,age,club_id,club_name,position_main,citizenship,market_value,agent_name,fetched_at
```

The existing profile export with `profile_description` is also supported; birth date and age are parsed from it when needed.

Wyscout often exposes only a compact age label such as:

```text
'97 (28)
```

The pipeline preserves the leading year as `birth_year` before converting `age` to a number. If only numeric age is available, the matcher derives a narrow three-year candidate window from the active calendar, for example age `28` in `2026` means `1997`, `1998`, or `1999`.

## Scope And Mapping Config Files

Generated/auditable files:

- `helpers/csv/transfermarkt_competitions_scope.csv`: Wyscout competitions, mapped Transfermarkt club coverage, scrape flag, and mapping status.
- `helpers/csv/transfermarkt_clubs_scope.csv`: Wyscout clubs with Transfermarkt club IDs and scraped player counts.
- `helpers/csv/transfermarkt_players_scope.csv`: prepared Transfermarkt player scope with parsed birth date and numeric market value.
- `helpers/csv/wyscout_transfermarkt_competition_mapping.csv`: editable competition mapping file between Wyscout and Transfermarkt.

Regenerate them after changing `transfermarkt_profiles.csv`, `club_matching_reference.csv`, or `player_matching_reference.csv`:

```bash
python scripts/build_transfermarkt_scope_configs.py
```

`wyscout_transfermarkt_competition_mapping.csv` is intentionally editable. Fill:

- `tm_competition_id`;
- `tm_competition_name`;
- `tm_season_id`;
- `scrape_enabled`;
- `mapping_status`.

Default matching scope is the active current-season calendars:

```text
TM_REFRESH_SEASON_LABEL=2026/2027,2026
```

This covers European-season competitions and calendar-year competitions in the same monthly Transfermarkt refresh.

Latest local dry-run on September 3, 2026, with `TM_REFRESH_SEASON_LABEL=2026/2027,2026`:

- Wyscout scope: `34,985` players locally;
- Transfermarkt scope: `38,896` profiles;
- automatic matches: `15,354`;
- review candidates: `28,792`.

The main quality ceiling is Wyscout identity depth. More automatic matches require either reliable `birth_year` backfill in `player_metrics` or additional curated competition/club/player mappings.

## Local Dry Run

Use a dry run first when validating a new Transfermarkt export:

```bash
docker compose --env-file .env -f infra/compose/docker-compose.yml run --rm \
  -e TM_REFRESH_DRY_RUN=1 \
  -e TM_MATCH_REVIEW_OUTPUT=/data/transfermarkt_match_reviews/ \
  transfermarkt-refresh
```

The review CSV is written under:

```text
data/transfermarkt_match_reviews/
```

## Local DB Refresh

```bash
./scripts/run_transfermarkt_monthly_refresh.sh
```

Optional overrides:

```bash
TM_PROFILES_PATH=/helpers/csv/transfermarkt_profiles.csv \
TM_PLAYER_MAP_PATH=/helpers/csv/player_matching_reference.csv \
TM_SNAPSHOT_DATE=2026-09-03 \
TM_REFRESH_SEASON_LABEL=2026/2027,2026 \
./scripts/run_transfermarkt_monthly_refresh.sh
```

## Production Refresh

Deploy the code on `main` first. Then run on the VPS:

```bash
cd ~/nextlegend/new_nextlegend
docker compose --env-file .env -f infra/compose/docker-compose-prod.yml run --rm transfermarkt-refresh
```

For a first production run after a new scraper export, run with `TM_REFRESH_DRY_RUN=1`, inspect the review CSV, then run without dry-run.

## App Compatibility

The app still reads Transfermarkt fields through existing `tm_fields`.

After a successful refresh, accepted matches update:

- `players.tm_id`
- `players.tm_profile_url`
- `player_seasons.tm_player_id`
- `player_seasons.tm_profile_url`
- `player_seasons.tm_profile_image_url`
- `player_seasons.tm_birth_date`
- `player_seasons.tm_age`
- `player_seasons.tm_club_id`
- `player_seasons.tm_club_name`
- `player_seasons.tm_position_main`
- `player_seasons.tm_citizenship`
- `player_seasons.tm_market_value`
- `player_seasons.tm_market_value_eur`
- `player_seasons.tm_agent_name`
