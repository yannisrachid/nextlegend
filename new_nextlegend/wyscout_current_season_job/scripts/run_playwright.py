from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap Playwright et lancer le scraper.")
    parser.add_argument(
        "--install-browsers",
        action="store_true",
        help="Force l'installation des navigateurs Playwright (chromium).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Lance Chromium en mode headless après installation.",
    )
    parser.add_argument(
        "--passthrough",
        nargs=argparse.REMAINDER,
        help="Arguments supplémentaires passés au script run_wyscout_scraper.py.",
    )
    return parser.parse_args(argv)


def ensure_browsers(force_install: bool) -> None:
    try:
        from playwright.sync_api import Error as PlaywrightError, sync_playwright
    except ModuleNotFoundError as exc:
        print(
            "Playwright n'est pas installé. Exécutez `pip install -r requirements.txt` puis `python -m playwright install chromium`."
        )
        raise SystemExit(1) from exc

    if force_install:
        subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=True)
        return
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            browser.close()
    except PlaywrightError as exc:  # pragma: no cover - dépend de l'environnement
        message = str(exc)
        if "Executable doesn't exist" in message or "browserType.launch" in message and "Executable" in message:
            print("Chromium n'est pas installé pour Playwright. Exécutez: python -m playwright install chromium")
        else:
            print("Échec du lancement de Chromium via Playwright (ce n'est pas forcément un problème d'installation).")
            print(f"Détail: {exc}")
        raise SystemExit(1) from exc


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    ensure_browsers(force_install=args.install_browsers)
    passthrough = args.passthrough or []
    if args.headless and "--headless" not in passthrough:
        passthrough = ["--headless", *passthrough]
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    runner = PROJECT_ROOT / "scripts" / "run_wyscout_scraper.py"
    cmd = [sys.executable, str(runner), *passthrough]
    result = subprocess.run(cmd, check=False)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
