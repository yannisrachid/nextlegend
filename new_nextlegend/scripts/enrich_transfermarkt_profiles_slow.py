#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from scrape_transfermarkt_api_profiles import (  # noqa: E402
    atomic_write_csv,
    call_with_retries,
    clean_id,
    first_non_blank,
    load_services,
    normalize_market_value,
    normalize_profile,
    read_json_cache,
    write_json_cache,
)


PROFILE_COLUMNS = [
    "profile_updated_at",
    "profile_url",
    "profile_description",
    "profile_name",
    "profile_full_name",
    "profile_name_in_home_country",
    "profile_image_url",
    "birth_city",
    "birth_country",
    "date_of_birth",
    "age",
    "height",
    "citizenship",
    "is_retired",
    "position_main",
    "position_other",
    "foot",
    "shirt_number",
    "club_id",
    "club_name",
    "club_joined",
    "club_contract_expires",
    "club_contract_option",
    "last_club_id",
    "last_club_name",
    "most_games_for",
    "market_value",
    "market_value_history",
    "market_value_ranking",
    "agent_name",
    "agent_url",
    "outfitter",
    "social_media",
    "trainer_profile",
    "relatives",
]


def read_csv_rows(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        columns = list(reader.fieldnames or [])
    return rows, columns


def write_errors(errors: list[dict[str, Any]], path: Path) -> None:
    atomic_write_csv(errors, path, ["scope", "id", "name", "status", "error"])


def load_blocked_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as fh:
        return {
            player_id
            for row in csv.DictReader(fh)
            if (player_id := clean_id(row.get("id"))) and str(row.get("status") or "").lower() == "blocked"
        }


def is_forbidden_error(error: str | None) -> bool:
    if not error:
        return False
    lowered = error.lower()
    return "403" in lowered or "forbidden" in lowered


def has_profile_data(row: dict[str, Any]) -> bool:
    return bool(
        first_non_blank(
            row.get("profile_updated_at"),
            row.get("profile_description"),
            row.get("profile_image_url"),
            row.get("agent_name"),
        )
    )


def build_target_indexes(
    rows: list[dict[str, Any]],
    *,
    only_missing: bool,
    player_ids: set[str] | None,
    blocked_ids: set[str],
    max_players: int | None,
) -> list[int]:
    indexes: list[int] = []
    for idx, row in enumerate(rows):
        player_id = clean_id(row.get("player_id"))
        if not player_id:
            continue
        if player_ids and player_id not in player_ids:
            continue
        if player_id in blocked_ids:
            continue
        if only_missing and has_profile_data(row):
            continue
        indexes.append(idx)
        if max_players and len(indexes) >= max_players:
            break
    return indexes


def enrich_one(args: tuple[Any, ...]) -> tuple[str, dict[str, Any] | None, dict[str, Any] | None]:
    TransfermarktPlayerProfile, TransfermarktPlayerMarketValue, row, cache_dir, retries, delay, refresh_cache, include_market_history = args
    player_id = clean_id(row.get("player_id"))
    if not player_id:
        return "", None, {"scope": "profile", "id": "", "name": row.get("player_name"), "error": "missing player_id"}

    fetched_at = dt.datetime.now(dt.timezone.utc).isoformat()
    profile_cache = cache_dir / "players" / f"{player_id}.json"
    profile = None if refresh_cache else read_json_cache(profile_cache)
    error = None
    if profile is None:
        profile, error = call_with_retries(
            f"profile={player_id}",
            lambda: TransfermarktPlayerProfile(player_id=player_id).get_player_profile(),
            retries=retries,
            delay=delay,
            no_retry_patterns=["403", "Forbidden"],
        )
        if profile:
            write_json_cache(profile_cache, profile)
    if not profile:
        return player_id, None, {
            "scope": "profile",
            "id": player_id,
            "name": row.get("player_name"),
            "status": "blocked" if is_forbidden_error(error) else "error",
            "error": error or "empty profile payload",
        }

    enriched = normalize_profile(profile, row, fetched_at)
    if include_market_history:
        market_cache = cache_dir / "market_values" / f"{player_id}.json"
        market = None if refresh_cache else read_json_cache(market_cache)
        market_error = None
        if market is None:
            market, market_error = call_with_retries(
                f"market_value={player_id}",
                lambda: TransfermarktPlayerMarketValue(player_id=player_id).get_player_market_value(),
                retries=retries,
                delay=delay,
                no_retry_patterns=["403", "Forbidden"],
            )
            if market:
                write_json_cache(market_cache, market)
        if market:
            enriched.update(normalize_market_value(market))
        elif market_error:
            return player_id, enriched, {
                "scope": "market_value",
                "id": player_id,
                "name": row.get("player_name"),
                "status": "blocked" if is_forbidden_error(market_error) else "error",
                "error": market_error,
            }
    return player_id, enriched, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Slow, resumable Transfermarkt player-profile enrichment.")
    parser.add_argument("--api-dir", default="../transfermarkt-api")
    parser.add_argument("--input", default="helpers/csv/transfermarkt_profiles.csv")
    parser.add_argument("--output", default="helpers/csv/transfermarkt_profiles.csv")
    parser.add_argument("--errors-output", default="helpers/csv/transfermarkt_profile_enrichment_errors.csv")
    parser.add_argument("--blocked-output", default="helpers/csv/transfermarkt_profile_enrichment_blocked.csv")
    parser.add_argument("--cache-dir", default="data/transfermarkt_api_cache")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--delay", type=float, default=2.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--cooldown-seconds", type=int, default=900)
    parser.add_argument("--cooldown-forbidden-rate", type=float, default=0.2)
    parser.add_argument("--cooldown-min-sample", type=int, default=10)
    parser.add_argument("--stop-after-forbidden", type=int, default=50)
    parser.add_argument("--max-players", type=int, default=None)
    parser.add_argument("--player-ids", default="")
    parser.add_argument("--retry-blocked", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--include-market-history", action="store_true")
    parser.add_argument("--all", action="store_true", help="Enrich already-enriched rows too.")
    args = parser.parse_args()

    api_dir = Path(args.api_dir)
    input_path = Path(args.input)
    output_path = Path(args.output)
    errors_path = Path(args.errors_output)
    blocked_path = Path(args.blocked_output)
    cache_dir = Path(args.cache_dir)

    if not api_dir.exists():
        raise FileNotFoundError(f"transfermarkt-api repo not found: {api_dir}")
    if not input_path.exists():
        raise FileNotFoundError(f"input CSV not found: {input_path}")

    rows, columns = read_csv_rows(input_path)
    for column in PROFILE_COLUMNS:
        if column not in columns:
            columns.append(column)
    for row in rows:
        for column in columns:
            row.setdefault(column, None)

    player_ids = {clean_id(part) for part in args.player_ids.replace(";", ",").split(",") if clean_id(part)}
    blocked_ids = set() if args.retry_blocked else load_blocked_ids(blocked_path)
    targets = build_target_indexes(
        rows,
        only_missing=not args.all,
        player_ids=player_ids or None,
        blocked_ids=blocked_ids,
        max_players=args.max_players,
    )
    print(
        f"[TM profile enrich] input_rows={len(rows)} targets={len(targets)} "
        f"workers={args.workers} delay={args.delay}s cache={cache_dir} skipped_blocked={len(blocked_ids)}",
        flush=True,
    )
    if not targets:
        atomic_write_csv(rows, output_path, columns)
        write_errors([], errors_path)
        print(f"[TM profile enrich] nothing to do; output written: {output_path}", flush=True)
        return

    _, TransfermarktPlayerProfile, TransfermarktPlayerMarketValue = load_services(api_dir)
    errors: list[dict[str, Any]] = []
    blocked_errors: list[dict[str, Any]] = [
        {"scope": "profile", "id": blocked_id, "name": "", "status": "blocked", "error": "previously blocked"}
        for blocked_id in sorted(blocked_ids)
    ]
    completed = 0
    tasks = [
        (
            TransfermarktPlayerProfile,
            TransfermarktPlayerMarketValue,
            rows[idx],
            cache_dir,
            args.retries,
            args.delay,
            args.refresh_cache,
            args.include_market_history,
        )
        for idx in targets
    ]
    index_by_player_id = {clean_id(rows[idx].get("player_id")): idx for idx in targets}

    batch_size = max(1, args.batch_size)
    workers = max(1, min(args.workers, batch_size))
    for batch_start in range(0, len(tasks), batch_size):
        batch = tasks[batch_start : batch_start + batch_size]
        batch_forbidden = 0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(enrich_one, task) for task in batch]
            for future in as_completed(futures):
                player_id, enriched, error = future.result()
                if enriched and player_id in index_by_player_id:
                    idx = index_by_player_id[player_id]
                    rows[idx].update(enriched)
                if error:
                    errors.append(error)
                    if error.get("status") == "blocked":
                        blocked_errors.append(error)
                        batch_forbidden += 1
                completed += 1
                if completed == len(targets) or completed % max(1, args.checkpoint_every) == 0:
                    atomic_write_csv(rows, output_path, columns)
                    write_errors(errors, errors_path)
                    write_errors(blocked_errors, blocked_path)
                    print(
                        f"[TM profile enrich] {completed}/{len(targets)} done "
                        f"errors={len(errors)} blocked={len(blocked_errors)} checkpoint={output_path}",
                        flush=True,
                    )

        batch_count = len(batch)
        forbidden_rate = batch_forbidden / batch_count if batch_count else 0
        if len(blocked_errors) >= args.stop_after_forbidden:
            atomic_write_csv(rows, output_path, columns)
            write_errors(errors, errors_path)
            write_errors(blocked_errors, blocked_path)
            print(
                f"[TM profile enrich] stop: blocked={len(blocked_errors)} "
                f"reached --stop-after-forbidden={args.stop_after_forbidden}",
                flush=True,
            )
            break
        if batch_count >= args.cooldown_min_sample and forbidden_rate >= args.cooldown_forbidden_rate:
            atomic_write_csv(rows, output_path, columns)
            write_errors(errors, errors_path)
            write_errors(blocked_errors, blocked_path)
            print(
                f"[TM profile enrich] cooldown: batch_forbidden={batch_forbidden}/{batch_count} "
                f"rate={forbidden_rate:.0%}; sleeping {args.cooldown_seconds}s",
                flush=True,
            )
            time.sleep(max(0, args.cooldown_seconds))

    atomic_write_csv(rows, output_path, columns)
    write_errors(errors, errors_path)
    write_errors(blocked_errors, blocked_path)
    print(
        f"[TM profile enrich] done output={output_path} errors={errors_path} "
        f"errors_count={len(errors)} blocked_count={len(blocked_errors)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
