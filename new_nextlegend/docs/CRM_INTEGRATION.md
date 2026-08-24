# CRM Integration And Migration

This document covers the integrated football CRM imported from the former `findyourlegend` standalone app.

## Scope

The CRM is integrated into the main NextLegend app, behind the existing app authentication.

Frontend entry point:

- `/crm`
- Navigation label: `NETWORK`

Backend API prefix:

- `/crm/*`

The standalone CRM authentication, layout and localStorage behavior are not used.

## Data Model

The main app already has non-CRM `clubs`, `players` and `prospects` tables. To avoid collisions and regressions, the integrated CRM uses dedicated tables:

- `crm_clubs`
- `crm_players`
- `crm_contacts`
- `crm_prospects`

Source IDs from `findyourlegend` are preserved as primary keys and also copied to `source_id` with `source = 'findyourlegend'`.

Relations:

- `crm_players.club_id -> crm_clubs.id`
- `crm_contacts.club_id -> crm_clubs.id`, nullable
- `crm_contacts.player_id -> crm_players.id`, nullable
- `crm_prospects.contact_id -> crm_contacts.id`

Important source-data behavior preserved:

- `crm_contacts.type = 'PLAYER'` is a category, not proof of a linked player row.
- Contacts with no linked club or player are valid and must remain visible.
- `crm_clubs.country` is kept as raw source data because the source contains non-country labels.
- `crm_players.age = 0` and empty `lastName` values are accepted.

## Expected Source Volumes

The Neon source observed for `findyourlegend` contains:

- `clubs`: 950
- `players`: 530
- `contacts`: 3948
- `prospects`: 1

The migration command prints source, migrated and target counts. A successful initial migration should return these same counts for `crm_*` tables, unless the source DB changed since the handoff.

## Required Environment Variables

Target DB:

- `DATABASE_URL`: already used by the main NextLegend API.

Source CRM DB:

- `CRM_SOURCE_DATABASE_URL` preferred, or `CRM_NEON_DATABASE_URL`.

The source variable must contain the former CRM Neon PostgreSQL URL with SSL enabled. Do not commit it to Git.

CRM email campaigns are intentionally not integrated in NextLegend. No SMTP variables are required.

## Local Dev Migration

From the repo root:

```bash
cd /Users/yannis/ylfc/new_nextlegend
export CRM_SOURCE_DATABASE_URL='postgresql://USER:PASSWORD@HOST:5432/neondb?sslmode=require'
docker compose --env-file .env -f infra/compose/docker-compose.yml run --rm \
  -e CRM_SOURCE_DATABASE_URL="$CRM_SOURCE_DATABASE_URL" \
  api python scripts/migrate_crm_from_neon.py
```

Expected output shape:

```text
{'source': {'clubs': 950, 'players': 530, 'contacts': 3948, 'prospects': 1}, 'migrated': {'clubs': 950, 'players': 530, 'contacts': 3948, 'prospects': 1}, 'target': {'clubs': 950, 'players': 530, 'contacts': 3948, 'prospects': 1}}
```

Then verify through the app:

```bash
docker compose --env-file .env -f infra/compose/docker-compose.yml up -d db api frontend
open http://localhost:3000/crm
```

If the frontend is already running, refresh `/crm` after the migration.

## CRM Location Cleanup

The Neon CRM source contains dirty `crm_clubs.city/country` values such as numeric tiers (`1`, `2`, `3`) or section headers. The map must not place those labels as if they were cities.

Location cleanup is explicit and conservative:

- `apps/api/crm_location_data.py` stores curated club-to-city fixes and city coordinates.
- `apps/api/scripts/clean_crm_locations.py` applies those fixes to `crm_clubs.city/country`.
- The script is dry-run by default.

Local dry-run:

```bash
docker compose --env-file .env -f infra/compose/docker-compose.yml exec -T api \
  python scripts/clean_crm_locations.py
```

Local apply:

```bash
docker compose --env-file .env -f infra/compose/docker-compose.yml exec -T api \
  python scripts/clean_crm_locations.py --apply
docker compose --env-file .env -f infra/compose/docker-compose.yml restart api
```

Current local result after the curated cleanup:

- mapped clubs: 610 / 950
- unmapped clubs: 340 / 950
- mapped locations: 499

The remaining unmapped clubs are mostly still in dirty source groups or require manual verification before correction. Do not bulk-geocode those groups blindly.

## CRM Data Cleanup

The CRM source also contains import artifacts that should not be shown as real CRM data:

- placeholder clubs created from country/tier section headers;
- player positions with inconsistent spelling;
- player nationalities with mixed languages or typos;
- contact notes filled with import metadata such as `region`, `country`, `tm_id`, `website`, `tier` and `status`.

The cleanup is implemented in `apps/api/scripts/clean_crm_data.py`.

It is conservative:

- it deletes only placeholder clubs matching the known import artifact shape, with `website = 'web'` and no linked players;
- linked contacts are preserved because `crm_contacts.club_id` uses `ON DELETE SET NULL`;
- it normalizes player positions, player nationalities and contact roles;
- it removes metadata-only notes and keeps useful free-text remnants;
- it does not invent missing ages, missing last names, missing clubs or missing player relations.

Local dry-run:

```bash
docker compose --env-file .env -f infra/compose/docker-compose.yml exec -T api \
  python scripts/clean_crm_data.py
```

Local apply:

```bash
docker compose --env-file .env -f infra/compose/docker-compose.yml exec -T api \
  python scripts/clean_crm_data.py --apply
docker compose --env-file .env -f infra/compose/docker-compose.yml restart api
```

Current local result after location cleanup plus data cleanup:

- `crm_clubs`: 902, after removing 48 placeholder header clubs;
- `crm_players`: 530;
- `crm_contacts`: 3948;
- contacts with notes: 489, down from 3833 metadata-heavy notes;
- unlinked contacts: 927, valid because source CRM supports contacts without club/player relation;
- mapped clubs: 610;
- unmapped clubs: 292;
- mapped locations: 499.

`crm_players.age = 0` is kept in the database because it means unknown in the source data. The frontend displays it as `Unknown age` instead of `0 yrs`.

## Local DB Verification

Check counts directly in the local Docker DB:

```bash
docker compose --env-file .env -f infra/compose/docker-compose.yml exec -T db \
  psql -U nextlegend -d nextlegend -c "\
SELECT 'crm_clubs' AS table_name, COUNT(*) FROM crm_clubs UNION ALL
SELECT 'crm_players', COUNT(*) FROM crm_players UNION ALL
SELECT 'crm_contacts', COUNT(*) FROM crm_contacts UNION ALL
SELECT 'crm_prospects', COUNT(*) FROM crm_prospects;"
```

Check contacts with missing relations:

```bash
docker compose --env-file .env -f infra/compose/docker-compose.yml exec -T db \
  psql -U nextlegend -d nextlegend -c "\
SELECT COUNT(*) AS unlinked_contacts
FROM crm_contacts
WHERE club_id IS NULL AND player_id IS NULL;"
```

## Production Migration

Only run this after the branch is merged/deployed and the API image includes `scripts/migrate_crm_from_neon.py`.

### Production Readiness

The CRM is ready for production deployment when these conditions are true:

- the deployed frontend image includes `/crm` and the navigation label `NETWORK`;
- the deployed API image includes the CRM endpoints, `scripts/migrate_crm_from_neon.py`, `scripts/clean_crm_locations.py`, `scripts/clean_crm_data.py` and `crm_location_data.py`;
- production `.env` contains the normal NextLegend `DATABASE_URL`;
- the CRM source secret is available only at runtime as `CRM_SOURCE_DATABASE_URL`;
- a production DB backup has been created before the migration;
- the first production run is executed manually, not by cron;
- post-run counts and CRM UI checks pass before exposing the feature broadly.

Current local validation status:

- local CRM migration from Neon: done;
- location cleanup: done;
- data cleanup: done;
- local CRM tables after cleanup: `902` clubs, `530` players, `3948` contacts, `1` prospect;
- local contacts with useful notes after cleanup: `489`;
- local map coverage after cleanup: `610` mapped clubs, `292` unmapped clubs, `499` mapped locations;
- Python syntax validation for CRM API/scripts: passed;
- frontend production build after CRM UI changes: passed.

Known acceptable residuals before prod:

- `292` clubs remain unmapped because source locations are still ambiguous and should not be guessed blindly;
- `crm_players.age = 0` remains stored for unknown ages and is displayed as `Unknown age`;
- `927` contacts are unlinked after cleanup because the source CRM allows contacts without club/player relation;
- email campaign/SMTP functionality is intentionally excluded.

Go/no-go rule:

- Go if the commands below complete without errors and the production counts match the expected post-cleanup range.
- No-go if migration counts are materially different from the source counts, API health fails, `/crm` is inaccessible after login, or CRM tables are empty after migration.

On the VPS:

```bash
cd /home/yannis/nextlegend/new_nextlegend
```

Ensure the source secret exists for the shell session, without writing it to Git:

```bash
export CRM_SOURCE_DATABASE_URL='postgresql://USER:PASSWORD@HOST:5432/neondb?sslmode=require'
```

Back up the current production DB before migration:

```bash
mkdir -p backups
stamp=$(date +%Y%m%d_%H%M%S)
docker compose --env-file .env -f infra/compose/docker-compose-prod.yml exec -T db \
  pg_dump -U nextlegend -d nextlegend > "backups/pre_crm_migration_${stamp}.sql"
```

Run the migration inside the production API image:

```bash
docker compose --env-file .env -f infra/compose/docker-compose-prod.yml run --rm \
  -e CRM_SOURCE_DATABASE_URL="$CRM_SOURCE_DATABASE_URL" \
  api python scripts/migrate_crm_from_neon.py
```

Verify counts:

```bash
docker compose --env-file .env -f infra/compose/docker-compose-prod.yml exec -T db \
  psql -U nextlegend -d nextlegend -c "\
SELECT 'crm_clubs' AS table_name, COUNT(*) FROM crm_clubs UNION ALL
SELECT 'crm_players', COUNT(*) FROM crm_players UNION ALL
SELECT 'crm_contacts', COUNT(*) FROM crm_contacts UNION ALL
SELECT 'crm_prospects', COUNT(*) FROM crm_prospects;"
```

Expected initial counts are approximately:

- `crm_clubs`: 950
- `crm_players`: 530
- `crm_contacts`: 3948
- `crm_prospects`: 1

If source data changed, trust the `source` counts printed by the migration script.

At this stage, before cleanup, the CRM UI can show dirty imported locations and metadata notes. Continue with both cleanup scripts before final validation.

After production migration and before validating the CRM map, apply the curated location cleanup:

```bash
docker compose --env-file .env -f infra/compose/docker-compose-prod.yml exec -T api \
  python scripts/clean_crm_locations.py

docker compose --env-file .env -f infra/compose/docker-compose-prod.yml exec -T api \
  python scripts/clean_crm_locations.py --apply

docker compose --env-file .env -f infra/compose/docker-compose-prod.yml restart api
```

Then apply the CRM data cleanup:

```bash
docker compose --env-file .env -f infra/compose/docker-compose-prod.yml exec -T api \
  python scripts/clean_crm_data.py

docker compose --env-file .env -f infra/compose/docker-compose-prod.yml exec -T api \
  python scripts/clean_crm_data.py --apply

docker compose --env-file .env -f infra/compose/docker-compose-prod.yml restart api
```

Final production verification:

```bash
docker compose --env-file .env -f infra/compose/docker-compose-prod.yml exec -T db \
  psql -U nextlegend -d nextlegend -c "\
SELECT 'crm_clubs' AS table_name, COUNT(*) FROM crm_clubs UNION ALL
SELECT 'crm_players', COUNT(*) FROM crm_players UNION ALL
SELECT 'crm_contacts', COUNT(*) FROM crm_contacts UNION ALL
SELECT 'crm_prospects', COUNT(*) FROM crm_prospects UNION ALL
SELECT 'crm_contacts_with_notes', COUNT(*) FROM crm_contacts WHERE NULLIF(TRIM(notes), '') IS NOT NULL UNION ALL
SELECT 'crm_unlinked_contacts', COUNT(*) FROM crm_contacts WHERE club_id IS NULL AND player_id IS NULL;"
```

Expected post-cleanup counts from the validated local run:

- `crm_clubs`: 902;
- `crm_players`: 530;
- `crm_contacts`: 3948;
- `crm_prospects`: 1;
- `crm_contacts_with_notes`: around 489;
- `crm_unlinked_contacts`: around 927.

Validate services:

```bash
curl -I https://api.nextlegend.fr/health
curl -I https://app.nextlegend.fr/
```

Validate manually in the app after login:

- `/crm` opens from the `NETWORK` tab;
- clubs list shows logos and clean cards;
- club/player cards are clickable and open modals above the page;
- search updates live for clubs, players and prospect contact selection;
- map loads, zoom works, logo markers render, and the search zooms to clubs/cities such as `Marseille`;
- dirty placeholder clubs like tier/header rows are no longer visible;
- player age `0` is shown as `Unknown age`.

Recommended first-prod sequence:

```bash
cd /home/yannis/nextlegend/new_nextlegend

docker compose --env-file .env -f infra/compose/docker-compose-prod.yml pull
docker compose --env-file .env -f infra/compose/docker-compose-prod.yml up -d

curl -I https://api.nextlegend.fr/health

stamp=$(date +%Y%m%d_%H%M%S)
mkdir -p backups
docker compose --env-file .env -f infra/compose/docker-compose-prod.yml exec -T db \
  pg_dump -U nextlegend -d nextlegend > "backups/pre_crm_migration_${stamp}.sql"

docker compose --env-file .env -f infra/compose/docker-compose-prod.yml run --rm \
  -e CRM_SOURCE_DATABASE_URL="$CRM_SOURCE_DATABASE_URL" \
  api python scripts/migrate_crm_from_neon.py

docker compose --env-file .env -f infra/compose/docker-compose-prod.yml exec -T api \
  python scripts/clean_crm_locations.py --apply

docker compose --env-file .env -f infra/compose/docker-compose-prod.yml exec -T api \
  python scripts/clean_crm_data.py --apply

docker compose --env-file .env -f infra/compose/docker-compose-prod.yml restart api frontend
```

## Idempotency

The migration is safe to rerun:

- It uses `ON CONFLICT (id) DO UPDATE`.
- It preserves source IDs.
- It updates changed CRM rows from Neon.
- It does not delete target CRM rows that are absent from the source.

If a destructive re-sync is required later, do not truncate production tables without a backup and explicit approval.

## Rollback

If the migration must be rolled back immediately and no CRM data should remain:

```bash
docker compose --env-file .env -f infra/compose/docker-compose-prod.yml exec -T db \
  psql -U nextlegend -d nextlegend -c "\
DROP TABLE IF EXISTS crm_prospects;
DROP TABLE IF EXISTS crm_contacts;
DROP TABLE IF EXISTS crm_players;
DROP TABLE IF EXISTS crm_clubs;"
```

For a full production DB rollback, restore the `pg_dump` backup created before migration.

## Notes

- The CRM map uses Leaflet with OpenStreetMap tiles through `/crm/map-clusters`. Only known clean `city,country` pairs are mapped; dirty source locations are counted as unmapped instead of being placed incorrectly.
- CRM email campaign functionality is intentionally excluded from the integrated app.
- Do not expose or commit the Neon password.
