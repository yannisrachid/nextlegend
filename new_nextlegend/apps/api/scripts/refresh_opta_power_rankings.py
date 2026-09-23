from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import unicodedata
import urllib.parse
import urllib.request
from ast import literal_eval
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from settings import settings


SOURCE_URL = "https://dataviz.theanalyst.com/opta-power-rankings/"
DEFAULT_MAPPING_CANDIDATES = (
    [Path("/helpers/csv/opta_power_club_mapping.csv")]
    + [parent / "helpers" / "csv" / "opta_power_club_mapping.csv" for parent in Path(__file__).resolve().parents]
)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS opta_power_ranking_runs (
    id SERIAL PRIMARY KEY,
    source_url TEXT NOT NULL,
    source_bundle_url TEXT,
    source_last_updated TEXT,
    source_last_updated_date DATE,
    gender TEXT NOT NULL DEFAULT 'mens',
    bundle_hash TEXT,
    rows_imported INT NOT NULL DEFAULT 0,
    matched_clubs INT NOT NULL DEFAULT 0,
    scraped_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(gender, source_last_updated, bundle_hash)
);
CREATE INDEX IF NOT EXISTS opta_power_ranking_runs_gender_date_idx
    ON opta_power_ranking_runs(gender, source_last_updated_date DESC, scraped_at DESC);

CREATE TABLE IF NOT EXISTS opta_power_rankings (
    id SERIAL PRIMARY KEY,
    run_id INT NOT NULL REFERENCES opta_power_ranking_runs(id) ON DELETE CASCADE,
    gender TEXT NOT NULL DEFAULT 'mens',
    opta_contestant_id TEXT NOT NULL,
    opta_id TEXT,
    rank INT,
    team TEXT NOT NULL,
    contestant_club_name TEXT,
    contestant_short_name TEXT,
    rating DOUBLE PRECISION,
    ranking_change_7_days INT,
    country TEXT,
    country_id TEXT,
    confederation TEXT,
    confederation_id TEXT,
    domestic_league_name TEXT,
    domestic_league_id TEXT,
    season_average_rating DOUBLE PRECISION,
    highest_season_rating DOUBLE PRECISION,
    lowest_season_rating DOUBLE PRECISION,
    raw JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(run_id, opta_contestant_id)
);
CREATE INDEX IF NOT EXISTS opta_power_rankings_run_rank_idx
    ON opta_power_rankings(run_id, rank);
CREATE INDEX IF NOT EXISTS opta_power_rankings_team_idx
    ON opta_power_rankings(team);

CREATE TABLE IF NOT EXISTS opta_power_ranking_club_mappings (
    id SERIAL PRIMARY KEY,
    run_id INT NOT NULL REFERENCES opta_power_ranking_runs(id) ON DELETE CASCADE,
    club_id INT NOT NULL REFERENCES clubs(id) ON DELETE CASCADE,
    opta_ranking_id INT NOT NULL REFERENCES opta_power_rankings(id) ON DELETE CASCADE,
    wyscout_team TEXT NOT NULL,
    wyscout_competition TEXT,
    opta_team TEXT NOT NULL,
    opta_country TEXT,
    opta_league TEXT,
    match_method TEXT NOT NULL,
    match_score DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(run_id, club_id)
);
CREATE INDEX IF NOT EXISTS opta_power_ranking_club_mappings_club_idx
    ON opta_power_ranking_club_mappings(club_id);
CREATE INDEX IF NOT EXISTS opta_power_ranking_club_mappings_run_idx
    ON opta_power_ranking_club_mappings(run_id);
"""


def fetch_text(url: str, timeout: int = 60) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 NextLegend club-power-refresh/1.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8")


def resolve_bundle_url(source_url: str) -> str:
    html = fetch_text(source_url)
    match = re.search(r'<script[^>]+type=["\']module["\'][^>]+src=["\']([^"\']+)["\']', html)
    if not match:
        match = re.search(r'<script[^>]+src=["\']([^"\']*index\.js[^"\']*)["\']', html)
    if not match:
        raise RuntimeError("Cannot locate Opta Power Rankings bundle script.")
    return urllib.parse.urljoin(source_url, match.group(1))


def parse_js_json_var(bundle: str, variable: str) -> list[dict[str, Any]]:
    match = re.search(rf"{re.escape(variable)}=JSON\.parse\(`(.*?)`\)", bundle, re.S)
    if not match:
        raise RuntimeError(f"Cannot locate JSON.parse block for {variable}.")
    raw = match.group(1)
    candidates = [
        raw,
        raw.replace("\\\\\"", "\\\""),
    ]
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise RuntimeError(f"Cannot decode JSON block for {variable}.")


def parse_last_updated(bundle: str) -> tuple[str | None, str | None]:
    match = re.search(r'const\s+N0="([^"]+)"', bundle)
    if not match:
        return None, None
    label = match.group(1).strip()
    try:
        parsed = datetime.strptime(label, "%b %d, %Y").date().isoformat()
    except ValueError:
        parsed = None
    return label, parsed


def safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def ranking_change(row: dict[str, Any]) -> int:
    current = safe_int(row.get("currentGlobalRank"))
    last_week = safe_int(row.get("lastWeekGlobalRank"))
    if current is None or last_week is None:
        return 0
    return last_week - current


def primary_competition(row: dict[str, Any]) -> str | None:
    existing = row.get("domesticLeagueName")
    if existing:
        return str(existing)
    raw = row.get("comps")
    if not raw:
        return None
    try:
        parsed = literal_eval(str(raw))
        if parsed and isinstance(parsed, list):
            return parsed[0].get("competitionName")
    except Exception:
        return None
    return None


def strip_accents(value: str) -> str:
    return "".join(
        char
        for char in unicodedata.normalize("NFKD", value)
        if not unicodedata.combining(char)
    )


LEGAL_SUFFIXES = {
    "fc",
    "cf",
    "sc",
    "ac",
    "afc",
    "cd",
    "sd",
    "rc",
    "rsc",
    "fk",
    "sk",
    "bk",
    "if",
    "nk",
    "ud",
    "us",
    "as",
    "sv",
    "vfb",
    "vfl",
    "club",
}


def normalize_name(value: Any, *, strip_suffixes: bool = False) -> str:
    text_value = strip_accents(str(value or "")).lower()
    text_value = text_value.replace("&", " and ")
    text_value = re.sub(r"[^a-z0-9]+", " ", text_value)
    tokens = [token for token in text_value.split() if token]
    if strip_suffixes:
        tokens = [token for token in tokens if token not in LEGAL_SUFFIXES]
    return " ".join(tokens)


def club_country_from_competition(competition: str | None) -> str | None:
    if not competition or "." not in competition:
        return None
    country = competition.split(".", 1)[0].strip()
    return country or None


def build_alias_index(rankings: list[dict[str, Any]], ranking_ids: dict[str, int]) -> dict[str, list[dict[str, Any]]]:
    aliases: dict[str, list[dict[str, Any]]] = {}
    for row in rankings:
        contestant_id = str(row.get("contestantId") or "")
        ranking_id = ranking_ids.get(contestant_id)
        if not ranking_id:
            continue
        alias_values = [
            row.get("contestantName"),
            row.get("contestantClubName"),
            row.get("contestantShortName"),
        ]
        for alias in alias_values:
            for stripped in (False, True):
                normalized = normalize_name(alias, strip_suffixes=stripped)
                if not normalized:
                    continue
                aliases.setdefault(normalized, []).append({**row, "_ranking_id": ranking_id})
    return aliases


def resolve_mapping_file(path: str | None) -> Path | None:
    if path:
        candidate = Path(path)
        return candidate if candidate.exists() else None
    return next((candidate for candidate in DEFAULT_MAPPING_CANDIDATES if candidate.exists()), None)


def load_mapping_overrides(path: str | None) -> dict[tuple[str, str], dict[str, str]]:
    mapping_path = resolve_mapping_file(path)
    if mapping_path is None:
        return {}
    overrides: dict[tuple[str, str], dict[str, str]] = {}
    with mapping_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            team = (row.get("wyscout_team") or "").strip()
            competition = (row.get("wyscout_competition") or "").strip()
            if not team:
                continue
            overrides[(team, competition)] = {key: (value or "").strip() for key, value in row.items()}
    return overrides


def best_match_for_club(
    club: dict[str, Any],
    rankings: list[dict[str, Any]],
    alias_index: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, Any], str, float] | None:
    names = [
        normalize_name(club["name"]),
        normalize_name(club["name"], strip_suffixes=True),
    ]
    country = club_country_from_competition(club.get("competition_name"))
    for name in names:
        candidates = alias_index.get(name) or []
        if candidates:
            country_candidates = [
                row for row in candidates if not country or str(row.get("country") or "").lower() == country.lower()
            ]
            selected = country_candidates[0] if country_candidates else candidates[0]
            return selected, "exact_name", 100.0

    search_space = rankings
    if country:
        country_matches = [row for row in rankings if str(row.get("country") or "").lower() == country.lower()]
        if country_matches:
            search_space = country_matches

    best: tuple[dict[str, Any], str, float] | None = None
    club_normalized = names[-1] or names[0]
    if not club_normalized:
        return None
    for row in search_space:
        ranking_id = row.get("_ranking_id")
        if not ranking_id:
            continue
        opta_names = [
            normalize_name(row.get("contestantName"), strip_suffixes=True),
            normalize_name(row.get("contestantClubName"), strip_suffixes=True),
            normalize_name(row.get("contestantShortName"), strip_suffixes=True),
        ]
        score = max(SequenceMatcher(None, club_normalized, name).ratio() for name in opta_names if name)
        match_score = round(score * 100.0, 2)
        if best is None or match_score > best[2]:
            best = (row, "fuzzy_country" if country else "fuzzy_global", match_score)
    if not best:
        return None
    threshold = 86.0 if country else 93.0
    return best if best[2] >= threshold else None


def ensure_schema(conn) -> None:
    for statement in [chunk.strip() for chunk in SCHEMA_SQL.split(";") if chunk.strip()]:
        conn.execute(text(statement))


def upsert_run(conn, *, source_url: str, bundle_url: str, last_updated: str | None, last_updated_date: str | None, gender: str, bundle_hash: str) -> int:
    row = conn.execute(
        text(
            """
            INSERT INTO opta_power_ranking_runs (
              source_url,
              source_bundle_url,
              source_last_updated,
              source_last_updated_date,
              gender,
              bundle_hash,
              scraped_at,
              updated_at
            )
            VALUES (
              :source_url,
              :source_bundle_url,
              :source_last_updated,
              CAST(:source_last_updated_date AS DATE),
              :gender,
              :bundle_hash,
              NOW(),
              NOW()
            )
            ON CONFLICT (gender, source_last_updated, bundle_hash)
            DO UPDATE SET
              source_bundle_url = EXCLUDED.source_bundle_url,
              scraped_at = NOW(),
              updated_at = NOW()
            RETURNING id
            """
        ),
        {
            "source_url": source_url,
            "source_bundle_url": bundle_url,
            "source_last_updated": last_updated,
            "source_last_updated_date": last_updated_date,
            "gender": gender,
            "bundle_hash": bundle_hash,
        },
    ).mappings().one()
    return int(row["id"])


def upsert_rankings(conn, *, run_id: int, gender: str, rankings: list[dict[str, Any]]) -> dict[str, int]:
    ranking_ids: dict[str, int] = {}
    for row in rankings:
        contestant_id = str(row.get("contestantId") or "").strip()
        team = str(row.get("contestantName") or "").strip()
        if not contestant_id or not team:
            continue
        inserted = conn.execute(
            text(
                """
                INSERT INTO opta_power_rankings (
                  run_id,
                  gender,
                  opta_contestant_id,
                  opta_id,
                  rank,
                  team,
                  contestant_club_name,
                  contestant_short_name,
                  rating,
                  ranking_change_7_days,
                  country,
                  country_id,
                  confederation,
                  confederation_id,
                  domestic_league_name,
                  domestic_league_id,
                  season_average_rating,
                  highest_season_rating,
                  lowest_season_rating,
                  raw,
                  updated_at
                )
                VALUES (
                  :run_id,
                  :gender,
                  :opta_contestant_id,
                  :opta_id,
                  :rank,
                  :team,
                  :contestant_club_name,
                  :contestant_short_name,
                  :rating,
                  :ranking_change_7_days,
                  :country,
                  :country_id,
                  :confederation,
                  :confederation_id,
                  :domestic_league_name,
                  :domestic_league_id,
                  :season_average_rating,
                  :highest_season_rating,
                  :lowest_season_rating,
                  CAST(:raw AS JSONB),
                  NOW()
                )
                ON CONFLICT (run_id, opta_contestant_id)
                DO UPDATE SET
                  opta_id = EXCLUDED.opta_id,
                  rank = EXCLUDED.rank,
                  team = EXCLUDED.team,
                  contestant_club_name = EXCLUDED.contestant_club_name,
                  contestant_short_name = EXCLUDED.contestant_short_name,
                  rating = EXCLUDED.rating,
                  ranking_change_7_days = EXCLUDED.ranking_change_7_days,
                  country = EXCLUDED.country,
                  country_id = EXCLUDED.country_id,
                  confederation = EXCLUDED.confederation,
                  confederation_id = EXCLUDED.confederation_id,
                  domestic_league_name = EXCLUDED.domestic_league_name,
                  domestic_league_id = EXCLUDED.domestic_league_id,
                  season_average_rating = EXCLUDED.season_average_rating,
                  highest_season_rating = EXCLUDED.highest_season_rating,
                  lowest_season_rating = EXCLUDED.lowest_season_rating,
                  raw = EXCLUDED.raw,
                  updated_at = NOW()
                RETURNING id
                """
            ),
            {
                "run_id": run_id,
                "gender": gender,
                "opta_contestant_id": contestant_id,
                "opta_id": str(row.get("optaId") or "") or None,
                "rank": safe_int(row.get("currentGlobalRank") or row.get("rank")),
                "team": team,
                "contestant_club_name": row.get("contestantClubName"),
                "contestant_short_name": row.get("contestantShortName"),
                "rating": safe_float(row.get("currentRating")),
                "ranking_change_7_days": ranking_change(row),
                "country": row.get("country"),
                "country_id": row.get("countryId"),
                "confederation": row.get("confederation"),
                "confederation_id": row.get("confederationId"),
                "domestic_league_name": primary_competition(row),
                "domestic_league_id": row.get("domesticLeagueId"),
                "season_average_rating": safe_float(row.get("seasonAverageRating")),
                "highest_season_rating": safe_float(row.get("highestSeasonRating")),
                "lowest_season_rating": safe_float(row.get("lowestSeasonRating")),
                "raw": json.dumps(row, ensure_ascii=False),
            },
        ).mappings().one()
        ranking_ids[contestant_id] = int(inserted["id"])
    return ranking_ids


def map_wyscout_clubs(
    conn,
    *,
    run_id: int,
    rankings: list[dict[str, Any]],
    ranking_ids: dict[str, int],
    mapping_file: str | None = None,
) -> int:
    ranking_by_id = []
    ranking_by_contestant_id: dict[str, dict[str, Any]] = {}
    for row in rankings:
        contestant_id = str(row.get("contestantId") or "")
        ranking_id = ranking_ids.get(contestant_id)
        if ranking_id:
            hydrated = {**row, "_ranking_id": ranking_id}
            ranking_by_id.append(hydrated)
            ranking_by_contestant_id[contestant_id] = hydrated
    alias_index = build_alias_index(rankings, ranking_ids)
    overrides = load_mapping_overrides(mapping_file)
    clubs = conn.execute(
        text(
            """
            SELECT cl.id, cl.name, co.name AS competition_name
            FROM clubs cl
            LEFT JOIN competitions co ON co.id = cl.competition_id
            WHERE cl.name IS NOT NULL
            """
        )
    ).mappings().all()
    matched = 0
    for club in clubs:
        club_dict = dict(club)
        override = overrides.get(((club_dict.get("name") or "").strip(), (club_dict.get("competition_name") or "").strip()))
        selected = None
        method = None
        score = None
        if override is not None:
            status = (override.get("review_status") or "").strip().lower()
            contestant_id = (override.get("opta_contestant_id") or "").strip()
            if status in {"unmapped", "rejected", "ignore"}:
                continue
            if contestant_id:
                row = ranking_by_contestant_id.get(contestant_id)
                if row:
                    selected = (row, f"mapping_file:{status or 'confirmed'}", 100.0)
        if selected is None:
            selected = best_match_for_club(club_dict, ranking_by_id, alias_index)
        if not selected:
            continue
        row, method, score = selected
        ranking_id = int(row["_ranking_id"])
        conn.execute(
            text(
                """
                INSERT INTO opta_power_ranking_club_mappings (
                  run_id,
                  club_id,
                  opta_ranking_id,
                  wyscout_team,
                  wyscout_competition,
                  opta_team,
                  opta_country,
                  opta_league,
                  match_method,
                  match_score,
                  updated_at
                )
                VALUES (
                  :run_id,
                  :club_id,
                  :opta_ranking_id,
                  :wyscout_team,
                  :wyscout_competition,
                  :opta_team,
                  :opta_country,
                  :opta_league,
                  :match_method,
                  :match_score,
                  NOW()
                )
                ON CONFLICT (run_id, club_id)
                DO UPDATE SET
                  opta_ranking_id = EXCLUDED.opta_ranking_id,
                  wyscout_team = EXCLUDED.wyscout_team,
                  wyscout_competition = EXCLUDED.wyscout_competition,
                  opta_team = EXCLUDED.opta_team,
                  opta_country = EXCLUDED.opta_country,
                  opta_league = EXCLUDED.opta_league,
                  match_method = EXCLUDED.match_method,
                  match_score = EXCLUDED.match_score,
                  updated_at = NOW()
                """
            ),
            {
                "run_id": run_id,
                "club_id": club["id"],
                "opta_ranking_id": ranking_id,
                "wyscout_team": club["name"],
                "wyscout_competition": club["competition_name"],
                "opta_team": row.get("contestantName"),
                "opta_country": row.get("country"),
                "opta_league": primary_competition(row),
                "match_method": method,
                "match_score": score,
            },
        )
        matched += 1
    return matched


def write_helper_csv(path: Path, rankings: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rankings, key=lambda row: safe_int(row.get("currentGlobalRank") or row.get("rank")) or 999999)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rank", "team", "rating", "ranking_change_7_days"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "rank": safe_int(row.get("currentGlobalRank") or row.get("rank")),
                    "team": row.get("contestantName"),
                    "rating": safe_float(row.get("currentRating")),
                    "ranking_change_7_days": ranking_change(row),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh Opta Power Rankings into PostgreSQL.")
    parser.add_argument("--source-url", default=SOURCE_URL)
    parser.add_argument("--gender", choices=["mens", "womens"], default="mens")
    parser.add_argument("--csv-output", default="")
    parser.add_argument("--mapping-file", default="")
    parser.add_argument("--skip-club-mapping", action="store_true")
    args = parser.parse_args()

    bundle_url = resolve_bundle_url(args.source_url)
    bundle = fetch_text(bundle_url)
    bundle_hash = hashlib.sha256(bundle.encode("utf-8")).hexdigest()
    last_updated, last_updated_date = parse_last_updated(bundle)
    variable = "f6" if args.gender == "mens" else "C0"
    rankings = parse_js_json_var(bundle, variable)
    if not rankings:
        raise RuntimeError("No Opta Power Ranking rows found.")

    engine = create_engine(settings.database_url, future=True, connect_args={"prepare_threshold": 0})
    with engine.begin() as conn:
        ensure_schema(conn)
        run_id = upsert_run(
            conn,
            source_url=args.source_url,
            bundle_url=bundle_url,
            last_updated=last_updated,
            last_updated_date=last_updated_date,
            gender=args.gender,
            bundle_hash=bundle_hash,
        )
        ranking_ids = upsert_rankings(conn, run_id=run_id, gender=args.gender, rankings=rankings)
        matched = 0
        if not args.skip_club_mapping:
            matched = map_wyscout_clubs(
                conn,
                run_id=run_id,
                rankings=rankings,
                ranking_ids=ranking_ids,
                mapping_file=args.mapping_file or None,
            )
        conn.execute(
            text(
                """
                UPDATE opta_power_ranking_runs
                SET rows_imported = :rows_imported,
                    matched_clubs = :matched_clubs,
                    updated_at = NOW()
                WHERE id = :run_id
                """
            ),
            {"run_id": run_id, "rows_imported": len(ranking_ids), "matched_clubs": matched},
        )

    if args.csv_output:
        write_helper_csv(Path(args.csv_output), rankings)

    print(
        json.dumps(
            {
                "run_id": run_id,
                "gender": args.gender,
                "source_last_updated": last_updated,
                "source_last_updated_date": last_updated_date,
                "bundle_url": bundle_url,
                "bundle_hash": bundle_hash[:12],
                "rankings": len(ranking_ids),
                "matched_clubs": matched,
                "scraped_at": datetime.now(timezone.utc).isoformat(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
