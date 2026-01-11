# NextLegend v2 — Documentation complete

Cette arborescence contient la nouvelle application NextLegend (frontend Next.js + API FastAPI + Postgres + pipeline batch + stockage objet S3/MinIO). Objectif : remplacer progressivement la v1 sans la casser, avec un flux data hebdomadaire automatise et des pages front completes.

## Vue d'ensemble
- Frontend Next.js : pages Ranking, Report, Comparison, Projection, Stats Research, Vizualisation, Prospect, AI.
- API FastAPI : endpoints de recherche/filtrage, rapports joueurs, similarites, meta (competitions/saisons), AI agentique, auth + admin.
- Postgres : serving DB (joueurs, saisons, metrics, roles, similarites, prospects, AI conversations, auth).
- Pipeline batch (Docker job) : ingestion CSV, enrichissements, scores, percentiles, similarites, transfermarkt, upsert DB, archivage S3.

## Architecture
- `apps/frontend` : interface utilisateur (Next.js)
- `apps/api` : API FastAPI
- `jobs/pipeline` : job d'ingestion/enrichissement
- `infra/compose` : docker-compose (local/dev)
- `helpers/` : fichiers de reference (glossaires, coefficients ligue, aliases)
- `docs/` : documentation technique complementaire

## Demarrage rapide (dev)
1. Depuis `new_nextlegend/` :
   ```bash
   cp .env.example .env
   docker compose -f infra/compose/docker-compose.yml up --build
   ```
2. Services :
   - Frontend : http://localhost:3000
   - API : http://localhost:8000/health
   - Postgres : localhost:${POSTGRES_PORT:-5432}
   - MinIO (optionnel) : http://localhost:9000 + console http://localhost:9001

## Variables d'environnement
Fichier `.env` (voir `.env.example`).

Base de donnees:
- `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_PORT`
- `DATABASE_URL` (utilise par l'API)

Stockage objet:
- `S3_ENDPOINT`, `S3_BUCKET`, `S3_ACCESS_KEY`, `S3_SECRET_KEY`
- MinIO local : `S3_ENDPOINT=http://minio:9000` + identifiants MinIO

API / Front:
- `API_BASE_URL` (api container)
- `NEXT_PUBLIC_API_BASE_URL` (frontend)

IA:
- `OPENAI_API_KEY`

Auth (optionnel):
- `AUTH_SESSION_DAYS` (defaut 365)
- `AUTH_COOKIE_SECURE` (true en prod https)
- `AUTH_USERS_JSON` ou `AUTH_USERNAME`/`AUTH_PASSWORD` (bootstrap si besoin)

## Auth + Admin
- Auth par session en base (cookie HttpOnly `nl_session`).
- Table `auth_users` en base, roles `admin`/`user`.
- Le bootstrap lit `config/credentials.toml` si la table est vide.
- Le user `yrachid` est force en admin.
- Portail admin : `/admin` (visible uniquement pour `yrachid`)
  - Lister, creer, editer, supprimer des users
  - Bouton d'import `credentials.toml` pour resynchroniser la base

## Donnees / Pipeline hebdo
Le flux data est un job Docker qui :
1. Telecharge le CSV brut depuis S3
2. Applique nettoyage + enrichissement (scores, percentiles, similarites, transfermarkt)
3. Archive les artefacts sur S3
4. Upsert en Postgres

### Commandes utiles
- Pipeline local (dev):
  ```bash
  docker compose -f infra/compose/docker-compose.yml run --rm pipeline \
    python -m pipeline.run --input-uri /helpers/csv/wyscout_players_final.csv \
    --bucket $S3_BUCKET --prefix new_nextlegend
  ```

- Pipeline refresh (hebdo, S3 -> DB):
  ```bash
  docker compose -f infra/compose/docker-compose.yml run --rm pipeline-refresh
  ```

### Archivage S3
- `s3://$S3_BUCKET/new_nextlegend/enriched/<run_id>_<timestamp>/...`
- Artefacts : raw, enriched, competitions, seasons, players, clubs, player_seasons, player_metrics, role_scores, player_similarity

## Base de donnees (Postgres)
Tables principales (serving):
- `competitions`, `seasons`, `clubs`, `players`
- `player_seasons` (fact principale)
- `player_metrics` (metrics + percentiles)
- `role_scores`, `player_similarity`

Tables applicatives:
- `prospects`, `club_needs`, `club_need_players`
- `ai_conversations`, `ai_messages`
- `auth_users`, `auth_sessions`

Voir schema detaille : `docs/DATA_MODEL.md`.

## API (FastAPI)
Endpoints principaux:
- `GET /health`
- `GET /ranking` (filtre + pagination)
- `GET /report/{player_id}` (profil complet)
- `GET /similar-players/{player_id}`
- `GET /players` (recherche joueur)
- `GET /meta/seasons`, `GET /meta/competitions`, `GET /meta/stats-research/metrics`
- `GET /stats-research` (scatter + table)
- `POST /ai/scout`, `POST /ai/player-report`
- `GET /ai/conversations`, `POST /ai/conversations`, `PATCH /ai/conversations/{id}`
- `POST /auth/login`, `POST /auth/logout`, `GET /auth/me`
- `GET/POST/PATCH/DELETE /admin/users` (admin)

## Frontend (Next.js)
Pages clefs:
- `/ranking` : classement par role/ligue/saison, tri, pagination
- `/report` : profil joueur complet (radar, roles, metrics, TM, similarites)
- `/comparison` : comparaison multi-joueurs
- `/projection` : projection score selon league translation
- `/stats-research` : scatterplot + table filtrable
- `/vizualisation` : pizza chart (percentiles)
- `/prospect` : shortlist + kanban club needs
- `/ai` : agents IA, conversations, usage
- `/admin` : gestion users (admin only)

## IA / Agentic
- Mode conversation type ChatGPT
- Extraction d'intentions (poste/role/league/age/minutes etc.)
- Requetes base pour proposer des joueurs
- Historique en DB (`ai_conversations`, `ai_messages`)
- Tracking d'usage OpenAI (tokens + cout estime)

## Operations / checks
- Health API : `GET /health`
- DB ready : `pg_isready` (compose)
- Logs : `docker compose ... logs api|frontend|pipeline`

## Documentation complementaire
- UX/UI : `docs/NEXTLEGEND_V2_UX_UI.md`
- Migration : `docs/NextLegend_v2_Migration_Guide.md`
- Pipeline : `docs/PIPELINE_PLAN.md`
- Data model : `docs/DATA_MODEL.md`

## Notes
- La v1 n'est pas modifiee (`../nextlegend`).
- Tout changement data passe par le pipeline batch hebdo.
- En prod, activer `AUTH_COOKIE_SECURE=true` et HTTPS.
