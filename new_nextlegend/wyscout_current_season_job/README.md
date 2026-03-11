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
- journalise le statut et met à jour les données visibles sur la Home (`Pipeline status`)

## Contenu

- `run_wyscout_current_weekly.sh`: scraping + renommage uniquement
- `run_current_season_e2e.sh`: scraping + pipeline + logs (cron recommandé)
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

Monitoring principal:
- Home de l'application (`Pipeline status`)
- logs `wyscout_current_season_job/logs/*`

SMTP reste optionnel et n'est pas requis en prod si `SKIP_EMAIL_ALERTS=1`.

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
- `SKIP_EMAIL_ALERTS=1` (recommandé en prod actuelle)
- `DOCKER_COMPOSE_FILE` (ex: `../infra/compose/docker-compose-prod.yml` sur VPS)
- `SIM_TOPK` (défaut `30`, similarités pipeline)
- `REQUIRE_SMTP_ALERTS=1` (optionnel, force échec si SMTP non configuré)

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
