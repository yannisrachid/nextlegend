from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wyscout_scraper import ScraperConfig, WyscoutScraper


def load_dotenv_file(path: str | None, *, override: bool = False) -> int:
    if not path:
        return 0
    dotenv_path = Path(path).expanduser()
    if not dotenv_path.exists():
        return 0
    loaded = 0
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if not override and key in os.environ:
            continue
        os.environ[key] = value
        loaded += 1
    return loaded


def build_config_from_args(args: argparse.Namespace) -> ScraperConfig:
    if args.config:
        raw = Path(args.config).read_text(encoding="utf-8")
        config = ScraperConfig.from_json(raw)
    else:
        config = ScraperConfig()
    if args.download_dir:
        config.download_dir = Path(args.download_dir)
    if args.output_csv_name:
        config.output_csv_name = args.output_csv_name
    if args.headless:
        config.headless = True
    if args.wait_timeout:
        config.wait_timeout = args.wait_timeout
    if args.login_timeout:
        config.wait_for_login_timeout = args.login_timeout
    if args.chunk_terms:
        config.chunk_search_terms = args.chunk_terms
    if args.calendar_preferences:
        config.calendar_preferences = args.calendar_preferences
    if args.skip:
        config.competition_blacklist = tuple({*config.competition_blacklist, *args.skip})
    if args.selected_file:
        config.selected_competitions_file = Path(args.selected_file)
    if args.auto_open_advanced_search:
        config.auto_open_advanced_search = True
    if args.auto_select_all_columns:
        config.auto_select_all_columns = True
    if args.auto_set_current_season:
        config.auto_set_current_season = True
    config.ensure_directories()
    return config


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export automatisé des données Wyscout.")
    parser.add_argument(
        "--dotenv",
        default=".env",
        help="Chemin vers un fichier .env (défaut: .env s'il existe).",
    )
    parser.add_argument(
        "--auto-login",
        action="store_true",
        help="Tente un login automatique via variables d'environnement (ex: WYSCOUT_EMAIL/WYSCOUT_PASSWORD).",
    )
    parser.add_argument("--config", help="Chemin vers un fichier JSON de configuration personnalisé.")
    parser.add_argument("--download-dir", help="Dossier de téléchargement (défaut: data).")
    parser.add_argument(
        "--output-csv-name",
        help="Nom du CSV de sortie dans download-dir (défaut: wyscout_players.csv).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Active le mode headless de Chromium (Playwright).",
    )
    parser.add_argument("--wait-timeout", type=int, help="Durée d'attente (secondes) pour les éléments.")
    parser.add_argument(
        "--login-timeout",
        type=int,
        help="Durée maximale (secondes) pour détecter la connexion manuelle.",
    )
    parser.add_argument(
        "--chunk-terms",
        nargs="+",
        help="Liste des filtres rapides utilisés pour contourner la limite d'export (ex: A B C ...).",
    )
    parser.add_argument(
        "--calendar-preferences",
        nargs="+",
        help="Calendriers acceptés, dans l'ordre de préférence (ex: 2024/2025 2024).",
    )
    parser.add_argument(
        "--auto-open-advanced-search",
        action="store_true",
        help="Laisser le script ouvrir lui-même l'application Advanced Search.",
    )
    parser.add_argument(
        "--auto-select-all-columns",
        action="store_true",
        help="Laisser le script activer l'option 'Toutes les colonnes'.",
    )
    parser.add_argument(
        "--auto-set-current-season",
        action="store_true",
        help="Laisser le script forcer la saison en cours.",
    )
    parser.add_argument(
        "--selected-file",
        help="Fichier contenant la liste des compétitions à exporter (une par ligne).",
    )
    parser.add_argument(
        "--list-competitions",
        action="store_true",
        help="Affiche toutes les compétitions détectées puis quitte sans exporter.",
    )
    parser.add_argument(
        "--list-output",
        help="Chemin de fichier où enregistrer la liste des compétitions (optionnel).",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        help="Limiter l'export à une liste de compétitions (noms exacts).",
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        help="Ignorer des compétitions spécifiques (noms exacts).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    loaded_env = load_dotenv_file(args.dotenv)
    if loaded_env:
        print(f"Chargement .env: {loaded_env} variable(s) depuis {Path(args.dotenv).expanduser()}")
    config = build_config_from_args(args)
    scraper = WyscoutScraper(config)
    try:
        scraper.run(
            only=args.only,
            skip=args.skip,
            list_only=args.list_competitions,
            list_output=args.list_output,
            auto_login=args.auto_login,
        )
    except KeyboardInterrupt:
        print("Exécution interrompue par l'utilisateur.")
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"Erreur : {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
