#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


NULL_STRINGS = {"", "nan", "none", "null", "<na>"}


def clean_id(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    text = str(value).strip().replace(".0", "")
    if not text or text.lower() in NULL_STRINGS or text == "0":
        return None
    return text


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    return value


def json_text(value: Any) -> str:
    return json.dumps(json_safe(value), ensure_ascii=True, allow_nan=False, default=str)


def join_list(value: Any) -> str | None:
    if isinstance(value, list):
        return "; ".join(str(item).strip() for item in value if str(item).strip())
    if value is None:
        return None
    return str(value)


def first_non_blank(*values: Any) -> Any:
    for value in values:
        if clean_id(value) is not None:
            return value
        if isinstance(value, str) and value.strip() and value.strip().lower() not in NULL_STRINGS:
            return value
        if value not in (None, ""):
            if not (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
                return value
    return None


def atomic_write_csv(rows: list[dict[str, Any]], path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col) for col in columns})
    tmp_path.replace(path)


def load_services(api_dir: Path):
    sys.path.insert(0, str(api_dir.resolve()))
    from app.services.clubs.players import TransfermarktClubPlayers  # type: ignore
    from app.services.players.market_value import TransfermarktPlayerMarketValue  # type: ignore
    from app.services.players.profile import TransfermarktPlayerProfile  # type: ignore

    return TransfermarktClubPlayers, TransfermarktPlayerProfile, TransfermarktPlayerMarketValue


def call_with_retries(
    label: str,
    func,
    *,
    retries: int,
    delay: float,
    no_retry_patterns: list[str] | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    last_error = None
    no_retry_patterns = no_retry_patterns or []
    for attempt in range(1, retries + 1):
        if delay > 0:
            time.sleep(delay + random.random() * delay)
        try:
            return func(), None
        except Exception as exc:  # noqa: BLE001 - scraper keeps going and records failures.
            last_error = str(exc)
            if any(pattern in last_error for pattern in no_retry_patterns):
                print(f"[TM scrape] blocked {label}: {last_error}", flush=True)
                return None, last_error
            sleep_for = min(20.0, (1.5**attempt) + random.random())
            print(f"[TM scrape] retry {attempt}/{retries} {label}: {last_error}", flush=True)
            time.sleep(sleep_for)
    return None, last_error


def read_json_cache(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_json_cache(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json_text(payload), encoding="utf-8")
    tmp.replace(path)


def build_club_scope(path: Path, max_clubs: int | None = None) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as fh:
        clubs = list(csv.DictReader(fh))
    for club in clubs:
        club["tm_club_id"] = clean_id(club.get("tm_club_id"))
        try:
            club["_tm_scraped_players_sort"] = int(float(club.get("tm_scraped_players") or 0))
        except ValueError:
            club["_tm_scraped_players_sort"] = 0
    clubs = [club for club in clubs if club.get("tm_club_id")]
    clubs = sorted(
        clubs,
        key=lambda club: (
            -int(club.get("_tm_scraped_players_sort") or 0),
            str(club.get("competition_name") or ""),
            str(club.get("team") or ""),
        ),
    )
    unique = {}
    for club in clubs:
        unique.setdefault(str(club["tm_club_id"]), club)
    clubs = list(unique.values())
    if max_clubs:
        clubs = clubs[:max_clubs]
    for club in clubs:
        club.pop("_tm_scraped_players_sort", None)
    return clubs


def normalize_club_player(player: dict[str, Any], club: dict[str, Any], season_id: str | None, fetched_at: str) -> dict[str, Any]:
    player_id = clean_id(player.get("id"))
    nationality = join_list(player.get("nationality"))
    club_id = clean_id(club.get("tm_club_id"))
    return {
        "player_id": player_id,
        "player_name": player.get("name"),
        "fetched_at": fetched_at,
        "source": "transfermarkt-api",
        "season_id": season_id or "current",
        "competition_name": club.get("competition_name"),
        "wyscout_team": club.get("team"),
        "club_id": club_id,
        "club_name": first_non_blank(player.get("currentClub"), club.get("tm_club_name"), club.get("team")),
        "roster_club_id": club_id,
        "roster_club_name": club.get("tm_club_name"),
        "profile_url": f"https://www.transfermarkt.com/-/profil/spieler/{player_id}" if player_id else None,
        "profile_description": None,
        "profile_name": player.get("name"),
        "profile_name_in_home_country": None,
        "profile_image_url": None,
        "birth_city": None,
        "birth_country": None,
        "date_of_birth": player.get("dateOfBirth"),
        "age": player.get("age"),
        "height": player.get("height"),
        "citizenship": nationality,
        "is_retired": None,
        "position_main": player.get("position"),
        "position_other": None,
        "foot": player.get("foot"),
        "shirt_number": None,
        "club_joined": player.get("joinedOn"),
        "club_contract_expires": player.get("contract"),
        "market_value": player.get("marketValue"),
        "agent_name": None,
        "agent_url": None,
        "outfitter": None,
        "social_media": None,
        "trainer_profile": None,
        "relatives": None,
        "youth_clubs": None,
        "joined": player.get("joined"),
        "signed_from": player.get("signedFrom"),
        "status": player.get("status"),
    }


def normalize_profile(profile: dict[str, Any], roster_row: dict[str, Any], fetched_at: str) -> dict[str, Any]:
    place = profile.get("placeOfBirth") or {}
    position = profile.get("position") or {}
    club = profile.get("club") or {}
    agent = profile.get("agent") or {}
    profile_id = clean_id(profile.get("id")) or roster_row.get("player_id")
    return {
        **roster_row,
        "player_id": profile_id,
        "player_name": first_non_blank(profile.get("name"), roster_row.get("player_name")),
        "profile_updated_at": fetched_at,
        "profile_url": first_non_blank(profile.get("url"), roster_row.get("profile_url")),
        "profile_description": profile.get("description"),
        "profile_name": profile.get("name"),
        "profile_full_name": profile.get("fullName"),
        "profile_name_in_home_country": profile.get("nameInHomeCountry"),
        "profile_image_url": profile.get("imageUrl"),
        "birth_city": place.get("city"),
        "birth_country": place.get("country"),
        "date_of_birth": first_non_blank(profile.get("dateOfBirth"), roster_row.get("date_of_birth")),
        "age": first_non_blank(profile.get("age"), roster_row.get("age")),
        "height": first_non_blank(profile.get("height"), roster_row.get("height")),
        "citizenship": join_list(first_non_blank(profile.get("citizenship"), roster_row.get("citizenship"))),
        "is_retired": profile.get("isRetired"),
        "position_main": first_non_blank(position.get("main"), roster_row.get("position_main")),
        "position_other": join_list(position.get("other")),
        "foot": first_non_blank(profile.get("foot"), roster_row.get("foot")),
        "shirt_number": profile.get("shirtNumber"),
        "club_id": clean_id(first_non_blank(club.get("id"), roster_row.get("club_id"))),
        "club_name": first_non_blank(club.get("name"), roster_row.get("club_name")),
        "club_joined": first_non_blank(club.get("joined"), roster_row.get("club_joined")),
        "club_contract_expires": first_non_blank(club.get("contractExpires"), roster_row.get("club_contract_expires")),
        "club_contract_option": club.get("contractOption"),
        "last_club_id": clean_id(club.get("lastClubId")),
        "last_club_name": club.get("lastClubName"),
        "most_games_for": club.get("mostGamesFor"),
        "market_value": first_non_blank(profile.get("marketValue"), roster_row.get("market_value")),
        "agent_name": agent.get("name"),
        "agent_url": agent.get("url"),
        "outfitter": profile.get("outfitter"),
        "social_media": json_text(profile.get("socialMedia") or []),
        "trainer_profile": json_text(profile.get("trainerProfile") or {}),
        "relatives": json_text(profile.get("relatives") or []),
    }


def normalize_market_value(market: dict[str, Any]) -> dict[str, Any]:
    return {
        "market_value": market.get("marketValue"),
        "market_value_history": json_text(market.get("marketValueHistory") or []),
        "market_value_ranking": json_text(market.get("ranking") or {}),
    }


def scrape_club_task(args: tuple[Any, ...]) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    TransfermarktClubPlayers, club, season_id, cache_dir, retries, delay, refresh_cache = args
    club_id = clean_id(club.get("tm_club_id"))
    cache_season = season_id or "current"
    fetched_at = dt.datetime.now(dt.timezone.utc).isoformat()
    cache_path = cache_dir / "clubs" / f"{cache_season}_{club_id}.json"
    payload = None if refresh_cache else read_json_cache(cache_path)
    error = None
    if payload is None:
        payload, error = call_with_retries(
            f"club={club_id}",
            lambda: TransfermarktClubPlayers(club_id=club_id, season_id=season_id).get_club_players(),
            retries=retries,
            delay=delay,
        )
        if payload:
            write_json_cache(cache_path, payload)
    if not payload:
        return [], {"scope": "club", "id": club_id, "name": club.get("tm_club_name"), "error": error or "empty payload"}
    rows = [
        normalize_club_player(player, club, season_id, fetched_at)
        for player in payload.get("players", [])
        if clean_id(player.get("id"))
    ]
    return rows, None


def scrape_profile_task(args: tuple[Any, ...]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    TransfermarktPlayerProfile, TransfermarktPlayerMarketValue, row, cache_dir, retries, delay, refresh_cache, include_market_history = args
    player_id = clean_id(row.get("player_id"))
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
        )
        if profile:
            write_json_cache(profile_cache, profile)
    if not profile:
        return row, {"scope": "profile", "id": player_id, "name": row.get("player_name"), "error": error or "empty payload"}

    merged = normalize_profile(profile, row, fetched_at)
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
            )
            if market:
                write_json_cache(market_cache, market)
        if market:
            merged.update(normalize_market_value(market))
        elif market_error:
            return merged, {"scope": "market_value", "id": player_id, "name": row.get("player_name"), "error": market_error}
    return merged, None


def run_parallel(label: str, tasks: list[tuple[Any, ...]], worker_count: int, fn) -> tuple[list[Any], list[dict[str, Any]]]:
    outputs: list[Any] = []
    errors: list[dict[str, Any]] = []
    total = len(tasks)
    print(f"[TM scrape] {label}: tasks={total} workers={worker_count}", flush=True)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(fn, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), start=1):
            result, error = future.result()
            if isinstance(result, list):
                outputs.extend(result)
            elif result:
                outputs.append(result)
            if error:
                errors.append(error)
            if done == total or done % 25 == 0:
                print(f"[TM scrape] {label}: {done}/{total} done rows={len(outputs)} errors={len(errors)}", flush=True)
    return outputs, errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape Transfermarkt clubs, players, profiles, and market values.")
    parser.add_argument("--api-dir", default=os.getenv("TRANSFERMARKT_API_DIR", "../transfermarkt-api"))
    parser.add_argument("--clubs-scope", default="helpers/csv/transfermarkt_clubs_scope.csv")
    parser.add_argument("--output", default="helpers/csv/transfermarkt_profiles.csv")
    parser.add_argument("--roster-output", default="helpers/csv/transfermarkt_club_rosters.csv")
    parser.add_argument("--errors-output", default="helpers/csv/transfermarkt_scrape_errors.csv")
    parser.add_argument("--cache-dir", default="data/transfermarkt_api_cache")
    parser.add_argument("--season-id", default=os.getenv("TM_SCRAPE_SEASON_ID", "current"))
    parser.add_argument("--club-workers", type=int, default=int(os.getenv("TM_SCRAPE_CLUB_WORKERS", "10")))
    parser.add_argument("--profile-workers", type=int, default=int(os.getenv("TM_SCRAPE_PROFILE_WORKERS", "12")))
    parser.add_argument("--retries", type=int, default=int(os.getenv("TM_SCRAPE_RETRIES", "3")))
    parser.add_argument("--delay", type=float, default=float(os.getenv("TM_SCRAPE_DELAY", "0.05")))
    parser.add_argument("--max-clubs", type=int, default=None)
    parser.add_argument("--max-players", type=int, default=None)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--skip-profiles", action="store_true")
    parser.add_argument("--include-market-history", action="store_true")
    args = parser.parse_args()

    api_dir = Path(args.api_dir)
    if not api_dir.exists():
        raise FileNotFoundError(f"transfermarkt-api repo not found: {api_dir}")

    TransfermarktClubPlayers, TransfermarktPlayerProfile, TransfermarktPlayerMarketValue = load_services(api_dir)
    cache_dir = Path(args.cache_dir)
    clubs = build_club_scope(Path(args.clubs_scope), args.max_clubs)
    season_id = None if str(args.season_id).strip().lower() in {"", "current", "none", "null"} else str(args.season_id).strip()
    club_tasks = [
        (TransfermarktClubPlayers, club, season_id, cache_dir, args.retries, args.delay, args.refresh_cache)
        for club in clubs
    ]
    roster_rows, errors = run_parallel("club rosters", club_tasks, args.club_workers, scrape_club_task)

    roster_rows = [row for row in roster_rows if clean_id(row.get("player_id"))]
    unique_roster_rows = {}
    for row in roster_rows:
        unique_roster_rows[str(row["player_id"])] = row
    roster_rows = sorted(
        unique_roster_rows.values(),
        key=lambda row: (str(row.get("club_name") or ""), str(row.get("player_name") or "")),
    )
    if args.max_players:
        roster_rows = roster_rows[: args.max_players]

    columns = [
        "player_id",
        "player_name",
        "fetched_at",
        "profile_updated_at",
        "source",
        "season_id",
        "competition_name",
        "wyscout_team",
        "club_id",
        "club_name",
        "roster_club_id",
        "roster_club_name",
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
        "youth_clubs",
        "joined",
        "signed_from",
        "status",
    ]
    atomic_write_csv(roster_rows, Path(args.roster_output), columns)
    print(f"[TM scrape] roster written: {args.roster_output} rows={len(roster_rows)}", flush=True)

    profile_rows = roster_rows
    if not args.skip_profiles and roster_rows:
        profile_tasks = [
            (
                TransfermarktPlayerProfile,
                TransfermarktPlayerMarketValue,
                row,
                cache_dir,
                args.retries,
                args.delay,
                args.refresh_cache,
                args.include_market_history,
            )
            for row in roster_rows
        ]
        profile_rows, profile_errors = run_parallel("player profiles", profile_tasks, args.profile_workers, scrape_profile_task)
        errors.extend(profile_errors)

    atomic_write_csv(profile_rows, Path(args.output), columns)
    atomic_write_csv(errors, Path(args.errors_output), ["scope", "id", "name", "error"])
    print(f"[TM scrape] profiles written: {args.output} rows={len(profile_rows)} errors={len(errors)}", flush=True)


if __name__ == "__main__":
    main()
