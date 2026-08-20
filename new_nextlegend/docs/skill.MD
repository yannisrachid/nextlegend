# NextLegend Project Skill

Use this file as the project-specific implementation guide for Codex and contributors.

## Product Context
- The active product is Next Legend for HD Sports.
- v2 in this repo is the source of truth.
- The legacy Streamlit v1 folder has been removed from this repo.
- User-facing copy should say `Next Legend` for the product and `HD Sports` for the company/team.
- Avoid prototype or implementation-led text such as `v2`, `current app`, `condensed workspace`, or `prototype` in UI copy.

## Stack
- Frontend: Next.js 14, React 18, Tailwind CSS, SWR, Recharts, Plotly.
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

## Pipeline Conventions
- The pipeline is the only writer for football serving tables.
- Input can be raw Wyscout CSV or enriched data via `PIPELINE_INPUT_KIND=raw|enriched`.
- Transfermarkt matching uses curated references in `helpers/csv` before fuzzy fallback.
- Keep fuzzy matching conservative: reject ambiguous weak-token matches.
- Similarities are top-k per role profile; default `SIM_TOPK=30`.
- Use `PIPELINE_REPLACE_SIMILARITY=1` when rebuilding `player_similarity`.
- In production, do not run the full raw pipeline on the current VPS. Use local-compute then PRD-load.

## Data Model Rules
- Keep `player_seasons` as the central fact table.
- Keep metrics in `player_metrics` and role fits in `role_scores`.
- Store Transfermarkt fields on `player_seasons` as `tm_*`; keep stable TM IDs/URLs on `players`.
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
curl -I https://api.nextlegend.fr/health
curl -sk -o /tmp/front.out -w "%{http_code}\n" https://app.nextlegend.fr
```

## Git And Editing Rules
- Branch flow after production reconciliation: `feature/*`, `bugfix/*`, or `hotfix/*` -> `dev` -> `prod`.
- Integrate branches with explicit 3-way merge commits: use `git merge --no-ff`, even when a fast-forward is possible.
- Deploy only `prod` to the VPS.
- Do not commit directly on the VPS after reconciliation; capture emergencies as `hotfix/*`.
- Keep changes scoped to the requested behavior.
- Do not revert user changes unless explicitly asked.
- Do not modify generated data, logs, or local secrets unless the task requires it.
- Do not commit credentials.
- Prefer small, reviewable edits over broad rewrites.
- Update docs when changing deployment, data refresh, auth, or schema behavior.
