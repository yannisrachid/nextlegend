# NextLegend v2 (new stack)

Cette arborescence contient la future version web/infra de NextLegend (Next.js + FastAPI + Postgres + MinIO), séparée de l’app Streamlit v1 (`../nextlegend`). L’objectif : démarrer l’infra Docker rapidement sans casser la v1.

## Structure courante
```
new_nextlegend/
├─ apps/
│  ├─ frontend/   # Next.js (placeholder)
│  └─ api/        # FastAPI (placeholder)
├─ jobs/
│  └─ pipeline/   # pipeline v2 (à remplir)
├─ infra/
│  ├─ compose/    # docker-compose.yml
│  └─ docker/     # Dockerfiles complémentaires si besoin
├─ docs/          # Guides de migration/UX
└─ .env.example   # variables d’environnement à copier
```

## Démarrage rapide (dev)
1. Depuis `new_nextlegend/` :
   ```bash
   cd new_nextlegend
   cp .env.example .env
   docker compose --env-file .env -f infra/compose/docker-compose.yml up --build
   ```
2. Les services attendus :
   - Frontend : http://localhost:3000 (placeholder Next app)
   - API : http://localhost:8000/health (FastAPI)
   - Postgres : localhost:${POSTGRES_PORT:-5432}
   - S3 :
     * Par défaut on utilise AWS S3 (voir variables `S3_*` dans `.env`).
     * MinIO reste dans le compose pour les tests locaux (API http://localhost:9000, console http://localhost:9001). Si vous ne l’utilisez pas, vous pouvez le laisser arrêté.

### Variables S3 (AWS vs MinIO)
- AWS (recommandé) : `S3_ENDPOINT=https://s3.<region>.amazonaws.com`, `S3_BUCKET`, `S3_ACCESS_KEY`, `S3_SECRET_KEY`.
- MinIO local : `S3_ENDPOINT=http://minio:9000`, `S3_ACCESS_KEY=MINIO_ROOT_USER`, `S3_SECRET_KEY=MINIO_ROOT_PASSWORD`.

## Pipeline batch (squelette)
- Code : `jobs/pipeline/` (`python -m pipeline.run --help`).
- Compose service : `pipeline` (lancé à la demande, pas en boucle).
- Exemple (dev) :
  ```bash
  cd new_nextlegend
  docker compose --env-file .env -f infra/compose/docker-compose.yml run --rm pipeline \
    --input-uri /data/wyscout_players_final.csv \
    --bucket nextlegend \
    --prefix new_nextlegend \
    --dry-run
  ```
- Voir `docs/PIPELINE_PLAN.md` pour le flux S3 → enrichissement → DB (upsert à venir).

### Drop local CSV puis lancer le batch
- Placer le fichier brut dans `./data/wyscout_players_final.csv` (à la racine du repo) **ou** dans `./new_nextlegend/helpers/csv/wyscout_players_final.csv` (monté sous `/helpers/csv`).
- Le fichier `nextlegend/player_profiles.json` est monté dans le container pipeline (`/app/player_profiles.json`) pour réutiliser les profils v1.
- Lancer : 
  ```bash
  docker compose --env-file .env -f infra/compose/docker-compose.yml run --rm pipeline \
    python -m pipeline.run --input-uri /helpers/csv/wyscout_players_final.csv --bucket $S3_BUCKET --prefix new_nextlegend
  ```
- Le job :
  - upload le brut sur S3 (`new_nextlegend/raw/<run_id>_YYYYMMDD-HHMMSS.csv`),
  - produit les artefacts enrichis (scores/percentiles/similarités + enrich TM) sous `new_nextlegend/enriched/...`,
  - upsert en DB (schema auto).

## Notes
- Aucun fichier de la v1 n’est modifié.
- Les Dockerfiles sont minimaux : ils servent de base pour ajouter le vrai code (frontend Next.js + API FastAPI + pipeline).
- Variables d’environnement : ajuster `.env` selon vos ports/mots de passe avant de lancer.
