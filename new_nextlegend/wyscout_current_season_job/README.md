# Wyscout Current Season Weekly Job (Portable)

Ce dossier est autonome pour relancer chaque semaine le scraping Wyscout de la saison actuelle:
- `2025/2026`
- `2026`

Le dossier contient 2 niveaux de job:

1. `run_wyscout_current_weekly.sh`
- scrape (resumable) `2025/2026` puis fallback `2026`
- harmonise les colonnes (`rename_columns.py`)

2. `run_current_season_e2e.sh` (recommandé)
- lance le scraping/renaming ci-dessus
- copie le CSV final vers `../data/wyscout_players_2025_2026_final.csv`
- déclenche le pipeline Docker `new_nextlegend` (cleaning + scores + similarités + TM + upsert DB)
- envoie un email de statut (SUCCESS/FAILED) avec tail de log

## Contenu

- `run_wyscout_current_weekly.sh`: scraping + renommage uniquement
- `run_current_season_e2e.sh`: scraping + pipeline + alerting (cron recommandé)
- `scripts/`: scripts Python requis (Playwright + reprise + renommage)
- `wyscout_scraper/`: package scraper
- `data/leagues.txt`: liste des compétitions à scraper
- `.env.example`: variables pour auto-login
- `requirements.txt`: dépendance Python

## Installation (VPS)

Depuis ce dossier:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m playwright install chromium
cp .env.example .env
```

Prérequis machine conseillé:
- au moins `2-3 GB` libres pour le téléchargement/install de Chromium Playwright.

Puis renseigner dans `.env`:
- `WYSCOUT_EMAIL`
- `WYSCOUT_PASSWORD`

Pour les alertes email (via SMTP), exporter aussi dans l'environnement (ou dans un wrapper cron):
- `SMTP_HOST`
- `SMTP_PORT` (default `587`)
- `SMTP_USERNAME`
- `SMTP_PASSWORD`
- `SMTP_FROM`
- `SMTP_TO` (ex: `you@example.com` ou `a@x.com,b@y.com`)
- `SMTP_USE_TLS` (`1` par défaut)

Le monitoring principal peut se faire via la Home de l'application (`Pipeline status`).
Les emails SMTP sont optionnels.

## Test manuel

```bash
./run_wyscout_current_weekly.sh
```

Job complet (recommandé):

```bash
./run_current_season_e2e.sh
```

## Cron hebdo (exemple)

Tous les lundis à 07:00 avec lock anti-chevauchement:

```cron
0 7 * * 1 cd /path/to/new_nextlegend/wyscout_current_season_job && /usr/bin/flock -n /tmp/nextlegend_current_season.lock DOCKER_COMPOSE_FILE=/path/to/new_nextlegend/infra/compose/docker-compose-prod.yml SKIP_EMAIL_ALERTS=1 PIPELINE_REPLACE_INPUT_SLICES=1 PIPELINE_REPLACE_SIMILARITY=1 ./run_current_season_e2e.sh >> logs/cron_current_season.log 2>&1
```

## Outputs

- `data/seasons/wyscout_players_2025_2026.csv`
- `final_data/wyscout_players_2025_2026_final.csv` (fichier final consolidé pour le job)
- `logs/current_season_e2e_*.log` (log horodaté du job complet)

## Paramètres utiles

Variables d'environnement optionnelles:
- `MAX_ATTEMPTS` (défaut: `10`)
- `PYTHON_BIN` (défaut: `python3`)
- `SKIP_SCRAPE=1` (test pipeline sans scraping)
- `SKIP_PIPELINE=1` (test scraping sans DB upsert)
- `SKIP_EMAIL_ALERTS=1` (désactive email pour test local)
- `DOCKER_COMPOSE_FILE` (ex: `../infra/compose/docker-compose-prod.yml` sur VPS)
- `SIM_TOPK` (défaut `30`, similarités pipeline)

Exemple:

```bash
MAX_ATTEMPTS=15 ./run_wyscout_current_weekly.sh
```

Test local rapide (sans scraping + sans email):

```bash
SKIP_SCRAPE=1 \
SCRAPER_FINAL_CSV=../data/wyscout_players_2025_2026_final.csv \
SIM_TOPK=0 \
SKIP_EMAIL_ALERTS=1 \
./run_current_season_e2e.sh
```
