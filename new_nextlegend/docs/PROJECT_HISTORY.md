# Project History

This file keeps only the project history that is still useful for future work.

## Origin
NextLegend started as a scouting product built around Wyscout data. The legacy Streamlit v1 implementation was used as a parity reference during migration and has now been removed from this repo.

This repo, `new_nextlegend`, is the active v2 product:
- Next.js frontend.
- FastAPI backend.
- Postgres serving database.
- Dockerized Pandas pipeline.
- MinIO object storage.

Do not reintroduce v1 code unless explicitly requested.

## Migration To v2
The v2 migration established these principles:
- all new product work happens in this repo;
- UI features must be backed by API endpoints and serving DB tables;
- football data flows through the pipeline, not through ad hoc frontend files;
- parity checks against v1 are useful only when migrating old behavior.

Main migrated surfaces:
- ranking;
- player report;
- comparison;
- projection;
- stats research;
- visualization;
- prospects;
- AI scouting assistant.

## Scoring And Data Pipeline Milestones
The pipeline now:
1. ingests Wyscout CSV;
2. normalizes and deduplicates player-season rows;
3. enriches with Transfermarkt using curated mappings and conservative fuzzy matching;
4. computes role scores;
5. converts scores to league/global percentiles;
6. applies league-strength adjustment;
7. computes player similarity;
8. upserts Postgres;
9. archives artifacts to MinIO.

Important scoring invariant:
- the assigned role score shown by the product must align with `player_seasons.global_score_adjusted`.

Important Transfermarkt lesson:
- weak fuzzy token overlap can create false matches between players at the same club. Keep fuzzy matching conservative and reject ambiguous candidates.

## MinIO Migration
The project moved away from AWS S3 to MinIO because the AWS account was no longer available.

Current object storage assumptions:
- MinIO is S3-compatible.
- Pipeline archives go under the configured `S3_BUCKET` and `S3_PREFIX`.
- Local and prod env files should use MinIO endpoint and credentials, not AWS-specific config.

## VPS Operating Model
The VPS serves production but is not sized for heavy full-pipeline compute.

Important historical finding:
- full raw pipeline runs with similarity generation on the VPS were OOM-killed with exit code 137.

Current policy:
- run heavy current-season compute locally or on a separate worker;
- copy final artifacts to the VPS/MinIO;
- load artifacts into PRD Postgres;
- preserve historical seasons.

The one-time historical backfill has already been completed. `scripts/backfill_vps_from_existing_csvs.sh` is retained only for exceptional recovery.

## HD Workspace Redesign
The product evolved from pure scouting screens into an agency workspace.

Current top-level product areas:
- `HQ`: priorities and operational cockpit.
- `HD PLAYERS`: represented players, notes, documents, contacts, objectives.
- `MERCATO 2026`: workbook-style needs, candidates, assignments, and export.
- `SCOUTING`: legacy scouting features grouped under a scouting shell.

Related API-managed tables:
- `hq_priority_items`;
- `hd_players`;
- `hd_player_documents`;
- `club_needs`;
- `club_need_players`.

## Transfer History And Club Logos
Club logos are generated from the Wyscout club source into JSON files used by both frontend and API exports.

Transfer history is imported from Wyscout transfer data into `player_transfer_history`. Mapping should stay conservative:
- prefer stable IDs plus normalized names;
- fallback only when normalized player name and known club history make the link credible.

## AI Layer
The AI assistant uses a controlled flow:
1. LLM extracts structured filters.
2. Backend executes deterministic SQL.
3. LLM writes scout-facing prioritization and explanation.

Do not let the LLM generate arbitrary SQL for production execution.

## Current Documentation Policy
Old session handoffs and interview notes were intentionally removed from `docs/`.

Keep docs focused on:
- project conventions;
- data model;
- VPS/deployment/refresh;
- useful product history.

If a detail is no longer operationally useful, do not keep it in `docs/`.
