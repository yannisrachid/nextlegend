# NextLegend Project Skill

Use this file as the project-specific implementation guide for Codex and contributors.

## Product Context
- The active product is Next Legend for HD Sports.
- v2 in this repo is the source of truth.
- The legacy Streamlit v1 folder has been removed from this repo.
- User-facing copy should say `Next Legend` for the product and `HD Sports` for the company/team.
- Avoid prototype or implementation-led text such as `v2`, `current app`, `condensed workspace`, or `prototype` in UI copy.

## Stack
- Frontend: Next.js 16, React 19, Tailwind CSS, SWR, Recharts, Plotly.
- API: FastAPI, SQLAlchemy 2, Pydantic 2, Postgres.
- Pipeline: Python, Pandas, PyArrow, SQLAlchemy, Boto3, MinIO S3-compatible storage.
- Runtime: Docker Compose for local and VPS.

## Code Ownership
- API routes, auth, lazy app schemas: `apps/api/main.py`.
- API response models: `apps/api/models.py`.
- AI scouting orchestration: `apps/api/agentic.py`.
- Mercato matching logic: `apps/api/mercato_logic.py`.
- Frontend pages: `apps/frontend/pages`.
- Frontend auth and API helpers: `apps/frontend/lib`.
- Pipeline processing: `jobs/pipeline/pipeline/processing.py`.
- Pipeline upserts: `jobs/pipeline/pipeline/db.py`.

## Backend Conventions
- Keep API behavior deterministic; LLMs may generate structured filters or prose, but SQL execution must stay controlled by backend code.
- Use SQLAlchemy sessions from the existing `get_session` / `SessionLocal` patterns.
- Keep schema creation idempotent when using lazy API-managed tables.
- Preserve public endpoints: `GET /`, `GET /health`, `POST /auth/login`, `POST /auth/logout`, `GET /auth/me`, `/image-proxy`, `/docs`, `/openapi`.
- All other endpoints require `nl_session`.
- Auth failures must return JSON 401 responses that include CORS headers for allowed origins.
- Do not expose DB credentials in docs, logs, or frontend code.

## Frontend Conventions
- Use the existing Next.js pages router.
- Keep the auth guard behavior: wait for `/auth/me`; redirect only after unauthenticated state is confirmed.
- Fetch API data through the existing frontend helpers and `NEXT_PUBLIC_API_BASE_URL`.
- Keep operational workspace UI dense, clear, and task-focused.
- Use existing shared components when present, especially `ClubLogo` and scouting shell/navigation components.
- Player photos should prefer application-provided image fields; canvas/export flows must use `/image-proxy` when cross-origin images need safe rendering.
- Current UI direction is dark premium SaaS: near-black page background, anthracite surfaces, subtle borders, restrained muted green accents, dense professional layouts, and no generic admin-template look.
- Main app navigation is on the left and must remain collapsible/expandable with a visible mid-rail toggle. Do not move it back to the right unless explicitly requested.
- Keep `Mercato 2026` hidden from normal navigation until the user asks to re-enable it. Do not delete the module.
- Use the `nl-field` pattern for text inputs/selects/filters to avoid browser-autofill blue backgrounds and inconsistent dark-mode styling.
- Scouting filter UX should preserve user selections when possible; default ranking filters are Big 5 Leagues, season `2026/2027`, all positions, 90+ minutes, age 15-45.
- Player search/autocomplete results should be ordered by club strength/ranking where the backend provides enough context.
- AI Assistant layout: keep a two-column desktop composition with `Conversation memory`, `Context lock`, admin-only `Usage`, and brief-input guidance in the left column; the chat/composer stays in the main column. Avoid a third right column because it breaks when the navigation is expanded.
- AI conversations are user-scoped server-side. The API must use the authenticated session user, not a client-provided `user_id`, as the source of truth for conversation history.
- AI usage/admin user tooling must remain visible only to admin `yrachid`.

## Pipeline Conventions
- The pipeline is the only writer for football serving tables.
- Input can be raw Wyscout CSV or enriched data via `PIPELINE_INPUT_KIND=raw|enriched`.
- Transfermarkt matching uses curated references in `helpers/csv` before fuzzy fallback.
- Keep fuzzy matching conservative: reject ambiguous weak-token matches.
- Monthly Transfermarkt refreshes use `pipeline.tm_refresh` and `docs/TRANSFERMARKT_REFRESH.md`.
- Transfermarkt refreshes must first run as dry-run/review when a new scraper export format is introduced.
- Store monthly market-value history in `transfermarkt_market_value_snapshots`; do not overwrite market values without snapshotting.
- Store Wyscout/Transfermarkt decisions in `player_transfermarkt_matches`; only `accepted` + `is_primary=true` links can update visible `tm_*` fields.
- Preserve Wyscout `birth_year` from compact age labels such as `'97 (28)` before converting `age` to numeric values; Transfermarkt matching relies on it.
- The sibling repos `../players-matcher` and `../transfermarkt-api` are references/producers, not vendored application code.
- Similarities are top-k per position group; default `SIM_TOPK=10`.
- Use `PIPELINE_REPLACE_SIMILARITY=1` when rebuilding `player_similarity`.
- Current-season score snapshots are pipeline-owned and enabled through `SCORE_SNAPSHOT_ENABLED=1`.
- Snapshot default scope is seasons `2026/2027` and `2026`, controlled by `SCORE_SNAPSHOT_SEASONS`.
- Snapshot default cadence is biweekly, controlled by `SCORE_SNAPSHOT_CADENCE=biweekly`; monthly is allowed if the product decision changes.
- Store both the visible score and the important scoring metrics in snapshot tables so future scoring-model changes remain auditable.
- Each snapshot stores `scoring_model_version` and a scoring model hash; do not remove those fields when editing scoring.
- Before the next current-season run, deploy `main` containing the snapshot schema/writer, then run the enriched load with `PIPELINE_REPLACE_INPUT_SLICES=0` and `SCORE_SNAPSHOT_ENABLED=1`.
- Routine current-season loads are pure upserts on `player_id + competition_id + season_id + club_id`; do not purge slices when the goal is to preserve players absent from the latest provider export.
- In production, do not run the full raw pipeline on the current VPS. Use local-compute then PRD-load.

## Data Model Rules
- Keep `player_seasons` as the central fact table.
- Keep metrics in `player_metrics` and role fits in `role_scores`.
- Keep in-season score tracking in `scoring_snapshot_runs`, `player_score_snapshots`, and `player_metric_snapshots`.
- Store Transfermarkt snapshots/matches in dedicated tables, then propagate accepted primary links to compatibility `tm_*` fields on `player_seasons`; keep stable TM IDs/URLs on `players`.
- Add indexes for new API query paths.
- Prefer idempotent schema changes and additive migrations.
- Do not delete historical seasons during routine refreshes.

## Validation
Before finishing code changes, run the narrowest useful checks:
```bash
python -m py_compile apps/api/main.py apps/api/agentic.py jobs/pipeline/pipeline/processing.py jobs/pipeline/pipeline/db.py
node --check apps/frontend/pages/<changed-page>.js
npm --prefix apps/frontend run build
```

Use the checks that match the changed surface. If Docker or network access is required and unavailable, say that explicitly.

Production smoke checks:
```bash
curl -I https://api.nextlegend.fr/
curl -s -o /tmp/api-health.out -w "%{http_code}\n" https://api.nextlegend.fr/health
curl -sk -o /tmp/front.out -w "%{http_code}\n" https://app.nextlegend.fr
```

## Git And Editing Rules
- Branch flow after production reconciliation: `feature/*`, `bugfix/*`, or `hotfix/*` -> `dev` -> `main`.
- Always create work branches from the latest `origin/main`.
- Do not develop directly on `dev` or `main`; all code changes must happen on `feature/*`, `bugfix/*`, or `hotfix/*`.
- Integrate branches with explicit 3-way merge commits: use `git merge --no-ff`, even when a fast-forward is possible.
- Deploy only `main` to the VPS.
- Do not commit directly on the VPS after reconciliation; capture emergencies as `hotfix/*`.
- Keep changes scoped to the requested behavior.
- Do not revert user changes unless explicitly asked.
- Do not modify generated data, logs, or local secrets unless the task requires it.
- Do not commit credentials.
- Prefer small, reviewable edits over broad rewrites.
- Update docs when changing deployment, data refresh, auth, or schema behavior.
