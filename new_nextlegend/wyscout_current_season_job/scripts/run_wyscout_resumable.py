from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
import tempfile
import time
from collections import OrderedDict
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


TERMINAL_STATUSES = {"success", "empty", "skipped_calendar"}
CSV_KEY_FIELDS = ("competition_name", "calendar", "row_number")
STATE_VERSION = 2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the resumable single-CSV scraping workflow."""
    parser = argparse.ArgumentParser(
        description="Lance le scraper Wyscout avec reprise automatique sur un CSV unique."
    )
    parser.add_argument(
        "--selected-file",
        required=True,
        help="Fichier source des competitions a traiter (une par ligne).",
    )
    parser.add_argument(
        "--download-dir",
        default="data",
        help="Dossier de travail du scraper. Defaut: data",
    )
    parser.add_argument(
        "--output-csv-name",
        default="wyscout_players.csv",
        help="Nom du CSV final dans download-dir. Defaut: wyscout_players.csv",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=10,
        help="Nombre max de tentatives pour cette execution. Defaut: 10",
    )
    parser.add_argument(
        "--state-file",
        help="Chemin du fichier d'etat de reprise. Defaut: <download-dir>/.wyscout_resume_state.json",
    )
    parser.add_argument(
        "--fresh-start",
        action="store_true",
        help="Ignore l'etat precedent et redemarre un scraping complet depuis zero.",
    )
    parser.add_argument(
        "--calendar-preferences",
        nargs="+",
        help="Calendriers acceptes, dans l'ordre de preference (ex: 2024/2025 2024).",
    )
    parser.add_argument(
        "--passthrough",
        nargs=argparse.REMAINDER,
        help="Arguments passes a scripts/run_playwright.py --passthrough ...",
    )
    return parser.parse_args(argv)


def now_iso() -> str:
    """Return the current local timestamp in a stable ISO-like format."""
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def resolve_path(raw: str) -> Path:
    """Resolve a user-provided path relative to the repository root when needed."""
    path = Path(raw)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def resolve_state_path(raw: str | None, download_dir: Path) -> Path:
    """Resolve the persisted resume-state file location for the current run."""
    if raw:
        return resolve_path(raw)
    return download_dir / ".wyscout_resume_state.json"


def read_competitions(path: Path) -> list[str]:
    """Load the competition list from disk while preserving order and removing duplicates."""
    raw = path.read_text(encoding="utf-8")
    deduped = OrderedDict()
    for line in raw.splitlines():
        value = line.strip()
        if value:
            deduped[value] = None
    return list(deduped.keys())


def load_json(path: Path) -> dict:
    """Best-effort JSON loader used for resume state and scraper reports."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_json(path: Path, payload: dict) -> None:
    """Atomically write a JSON payload to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def start_new_state(selected_file: Path, output_csv: Path, competitions: list[str]) -> dict:
    """Create the initial resume state for a fresh scraping session."""
    return {
        "version": STATE_VERSION,
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "selected_file": str(selected_file),
        "output_csv": str(output_csv),
        "requested_competitions": competitions,
        "competition_status": {},
        "attempts_used": 0,
        "last_exit_code": None,
    }


def state_matches_request(state: dict, selected_file: Path, output_csv: Path, competitions: list[str]) -> bool:
    """Check whether an existing state file can be safely reused for the current command."""
    if not state:
        return False
    if int(state.get("version", 0)) != STATE_VERSION:
        return False
    if str(state.get("selected_file", "")) != str(selected_file):
        return False
    if str(state.get("output_csv", "")) != str(output_csv):
        return False
    requested = state.get("requested_competitions")
    return isinstance(requested, list) and requested == competitions


def cleanup_path(path: Path) -> None:
    """Delete a file if it exists and ignore missing-file errors."""
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def csv_row_key(fieldnames: list[str], row: dict[str, str]) -> tuple:
    """Build a stable deduplication key for a CSV row."""
    if all(field in fieldnames for field in CSV_KEY_FIELDS):
        return tuple(row.get(field, "") for field in CSV_KEY_FIELDS)
    return tuple((field, row.get(field, "")) for field in fieldnames)


def rewrite_csv_without_competitions(csv_path: Path, competitions_to_drop: set[str]) -> None:
    """Rewrite the final CSV after removing incomplete competitions and collapsing duplicates."""
    if not csv_path.exists():
        return

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            cleanup_path(csv_path)
            return

        deduped_rows: OrderedDict[tuple, dict[str, str]] = OrderedDict()
        for row in reader:
            competition_name = (row.get("competition_name") or "").strip()
            if competition_name and competition_name in competitions_to_drop:
                continue
            deduped_rows[csv_row_key(fieldnames, row)] = row

    tmp_handle = tempfile.NamedTemporaryFile(
        "w",
        newline="",
        encoding="utf-8",
        dir=csv_path.parent,
        prefix=f"{csv_path.stem}_",
        suffix=".tmp",
        delete=False,
    )
    tmp_path = Path(tmp_handle.name)
    try:
        with tmp_handle:
            writer = csv.DictWriter(tmp_handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in deduped_rows.values():
                writer.writerow({field: row.get(field, "") for field in fieldnames})
        tmp_path.replace(csv_path)
    finally:
        cleanup_path(tmp_path)


def pending_competitions(state: dict) -> list[str]:
    """Return the subset of requested competitions that still need to be scraped."""
    requested = state.get("requested_competitions") or []
    statuses = state.get("competition_status") or {}
    remaining: list[str] = []
    for competition in requested:
        info = statuses.get(competition)
        status = ""
        if isinstance(info, dict):
            status = str(info.get("status", "")).strip().lower()
        if status not in TERMINAL_STATUSES:
            remaining.append(competition)
    return remaining


def merge_report_into_state(state: dict, report: dict) -> None:
    """Merge one scraper attempt report into the persisted resume state."""
    statuses = report.get("competition_status") or {}
    if not isinstance(statuses, dict):
        return
    state_statuses = state.setdefault("competition_status", {})
    for competition, info in statuses.items():
        if not isinstance(info, dict):
            continue
        merged = {
            "status": str(info.get("status", "")).strip().lower(),
            "updated_at": now_iso(),
        }
        detail = info.get("detail")
        if detail:
            merged["detail"] = str(detail)
        state_statuses[competition] = merged


def build_attempt_command(
    passthrough: list[str],
    download_dir: Path,
    selected_file: Path,
    output_csv_name: str,
    calendar_preferences: list[str] | None,
) -> list[str]:
    """Build the underlying scraper command for one attempt."""
    forbidden = {"--selected-file", "--output-csv-name", "--download-dir", "--calendar-preferences"}
    if any(arg in forbidden for arg in passthrough):
        raise SystemExit(
            "Ne passez pas --selected-file/--output-csv-name/--download-dir/--calendar-preferences dans --passthrough: "
            "ils sont geres par le wrapper de reprise."
        )

    runner = PROJECT_ROOT / "scripts" / "run_playwright.py"
    scraper_args = [
        *passthrough,
        "--download-dir",
        str(download_dir),
        "--selected-file",
        str(selected_file),
        "--output-csv-name",
        output_csv_name,
    ]
    if calendar_preferences:
        scraper_args.extend(["--calendar-preferences", *calendar_preferences])
    return [sys.executable, str(runner), "--passthrough", *scraper_args]


def run_attempt(
    attempt_idx: int,
    competitions: list[str],
    passthrough: list[str],
    download_dir: Path,
    output_csv_name: str,
    calendar_preferences: list[str] | None,
) -> int:
    """Run one scraper attempt on the current subset of remaining competitions."""
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        prefix="wyscout_remaining_",
        suffix=".txt",
        delete=False,
    ) as handle:
        handle.write("\n".join(competitions))
        selected_path = Path(handle.name)

    try:
        cmd = build_attempt_command(
            passthrough=passthrough,
            download_dir=download_dir,
            selected_file=selected_path,
            output_csv_name=output_csv_name,
            calendar_preferences=calendar_preferences,
        )
        print(f"\n=== Tentative {attempt_idx} | competitions restantes: {len(competitions)} ===")
        print(f"Commande: {shlex.join(cmd)}")
        return subprocess.run(cmd, cwd=PROJECT_ROOT, check=False).returncode
    finally:
        cleanup_path(selected_path)


def initialise_or_resume_state(
    *,
    state_path: Path,
    selected_file: Path,
    output_csv: Path,
    report_path: Path,
    competitions: list[str],
    fresh_start: bool,
) -> dict:
    """Load an existing resume state or initialise a new one for this request."""
    if fresh_start:
        cleanup_path(state_path)
        cleanup_path(output_csv)
        cleanup_path(report_path)
        return start_new_state(selected_file, output_csv, competitions)

    state = load_json(state_path)
    if not state_matches_request(state, selected_file, output_csv, competitions):
        cleanup_path(state_path)
        cleanup_path(output_csv)
        cleanup_path(report_path)
        return start_new_state(selected_file, output_csv, competitions)

    if not output_csv.exists():
        cleanup_path(state_path)
        cleanup_path(report_path)
        return start_new_state(selected_file, output_csv, competitions)

    state["updated_at"] = now_iso()
    return state


def main(argv: list[str] | None = None) -> int:
    """Execute the resumable single-CSV scraping loop until completion or retry exhaustion."""
    args = parse_args(argv)
    selected_file = resolve_path(args.selected_file)
    download_dir = resolve_path(args.download_dir)
    state_path = resolve_state_path(args.state_file, download_dir)
    report_path = download_dir / "run_report.json"
    output_csv = download_dir / args.output_csv_name
    passthrough = list(args.passthrough or [])
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    if not selected_file.exists():
        raise SystemExit(f"Fichier introuvable: {selected_file}")

    competitions = read_competitions(selected_file)
    if not competitions:
        raise SystemExit("Aucune competition dans le fichier selected-file.")

    download_dir.mkdir(parents=True, exist_ok=True)
    state = initialise_or_resume_state(
        state_path=state_path,
        selected_file=selected_file,
        output_csv=output_csv,
        report_path=report_path,
        competitions=competitions,
        fresh_start=bool(args.fresh_start),
    )
    save_json(state_path, state)

    for _ in range(max(1, args.max_attempts)):
        remaining = pending_competitions(state)
        if not remaining:
            rewrite_csv_without_competitions(output_csv, set())
            cleanup_path(report_path)
            cleanup_path(state_path)
            print(f"Scraping termine -> {output_csv}")
            return 0

        rewrite_csv_without_competitions(output_csv, set(remaining))
        state["attempts_used"] = int(state.get("attempts_used", 0)) + 1
        state["updated_at"] = now_iso()
        save_json(state_path, state)

        rc = run_attempt(
            attempt_idx=int(state["attempts_used"]),
            competitions=remaining,
            passthrough=passthrough,
            download_dir=download_dir,
            output_csv_name=args.output_csv_name,
            calendar_preferences=args.calendar_preferences,
        )

        report = load_json(report_path)
        merge_report_into_state(state, report)
        state["last_exit_code"] = rc
        state["updated_at"] = now_iso()
        save_json(state_path, state)
        cleanup_path(report_path)

    remaining = pending_competitions(state)
    rewrite_csv_without_competitions(output_csv, set(remaining))
    print(
        "Execution incomplete apres "
        f"{args.max_attempts} tentative(s). Competitions restantes: {len(remaining)}."
    )
    print(f"Etat de reprise conserve dans {state_path}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
