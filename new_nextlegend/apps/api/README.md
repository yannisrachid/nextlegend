# NextLegend v2 API (FastAPI)

## Démarrer en local (Docker)
```bash
cd new_nextlegend
docker compose --env-file .env -f infra/compose/docker-compose.yml up --build api
```
L’API écoute sur `http://localhost:8000`.

## Variables clés
- `DATABASE_URL` (ex: `postgresql+psycopg://nextlegend:nextlegend_dev_password@db:5432/nextlegend`)
- `CORS_ORIGINS` via `.env` (dans `settings.py`, par défaut `*`).
- `OPENAI_API_KEY` pour activer les endpoints IA.

## Endpoints implémentés
- `GET /health` : ping.
- `GET /ranking` : top N (def 30) trié sur `global_score_adjusted`.
  - Params : `role`, `competition`, `season`, `position`, `team`, `age_min`, `age_max`, `min_minutes` (def 270), `limit`.
- `GET /ranking/page` : pagination avec `total`.
- `GET /meta/positions` : liste des positions (filtres `competition`, `season`).
- `GET /meta/teams` : liste des clubs/teams (filtres `competition`, `season`).
- `GET /meta/seasons` : liste des saisons.
- `GET /meta/players` : liste des joueurs (filtres `competition`, `season`, `team`).
- `GET /players/{id}` : carte joueur (player_season le plus récent).
- `GET /players/{id}/report` : fiche player (player_season le plus récent) + metrics wide + role_scores + summary_*.
- `GET /players/{id}/similarities` : top voisins par profil (param `profile` optionnel).
- `GET /ops/metrics` : compteurs + dernier run pipeline.
- `POST /ai/scout` : brief → filtres + shortlist + watchlist IA.
- `POST /ai/player-report` : rapport rédigé pour un joueur.
- `GET /ai/conversations` : liste des conversations IA (par user_id).
- `POST /ai/conversations` : crée une conversation IA.
- `GET /ai/conversations/{id}` : messages d'une conversation IA.
- `POST /ai/conversations/{id}/messages` : envoie un message et récupère la réponse IA.
- `PATCH /ai/conversations/{id}` : met à jour le titre de la conversation.

## Data attendue en DB
Tables alimentées par le pipeline v2 : `competitions`, `seasons`, `players`, `player_seasons`, `player_metrics`, `role_scores`, `player_similarity`, `pipeline_runs`.

## Notes
- Le pipeline écrit en S3 sous le préfixe `new_nextlegend` et upsert en DB.
- Warning fuzzywuzzy bénin (option d’installer `python-Levenshtein` si besoin).
