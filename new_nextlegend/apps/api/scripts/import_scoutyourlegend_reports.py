#!/usr/bin/env python3
"""Import legacy scoutyourlegend CSV scouting reports into PostgreSQL.

The CSV files are migration inputs only. Runtime reads and writes must use
PostgreSQL tables created by main._ensure_youth_schema().
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMPORT_DIR = ROOT / "imports" / "scoutyourlegend"


def ensure_api_import_path() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def sa_text(statement: str):
    from sqlalchemy import text

    return text(statement)


def ensure_youth_schema(session) -> None:
    ensure_api_import_path()
    import main

    main._ensure_youth_schema(session)


def session_local():
    ensure_api_import_path()
    from db import SessionLocal

    return SessionLocal


def blank_to_none(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return value


def to_uuid(value: Any) -> str:
    raw = str(value or "").strip()
    if raw:
        try:
            return str(uuid.UUID(raw))
        except ValueError:
            pass
    return str(uuid.uuid4())


def to_int(value: Any) -> int | None:
    raw = blank_to_none(value)
    if raw is None:
        return None
    try:
        numeric = int(float(str(raw)))
    except (TypeError, ValueError):
        return None
    return numeric


def to_float(value: Any) -> float | None:
    raw = blank_to_none(value)
    if raw is None:
        return None
    try:
        numeric = float(str(raw))
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def extract_eyeball_player_id(value: Any) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    match = re.search(r"(?:/player/|playerId=|player_id=)(\d+)", raw, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    if re.fullmatch(r"\d+", raw):
        return raw
    return None


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def parse_match_number(value: Any) -> float | None:
    numeric = to_float(value)
    return numeric


def normalize_matches(value: Any) -> list[dict[str, Any]]:
    raw = str(value or "").strip()
    if not raw:
        return []
    if raw.startswith("["):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            output = []
            for item in parsed:
                if isinstance(item, str):
                    output.append(
                        {
                            "team_a": item.strip(),
                            "score_a": None,
                            "score_b": None,
                            "team_b": "",
                            "competition": "",
                            "match_date": "",
                            "player_rating": None,
                        }
                    )
                    continue
                if not isinstance(item, dict):
                    continue
                normalized = {
                    "team_a": str(item.get("team_a") or "").strip(),
                    "score_a": parse_match_number(item.get("score_a")),
                    "score_b": parse_match_number(item.get("score_b")),
                    "team_b": str(item.get("team_b") or "").strip(),
                    "competition": str(item.get("competition") or "").strip(),
                    "match_date": str(item.get("match_date") or "").strip(),
                    "player_rating": parse_match_number(item.get("player_rating")),
                }
                if "minutes_played" in item:
                    normalized["minutes_played"] = parse_match_number(item.get("minutes_played"))
                if "position" in item:
                    normalized["position"] = str(item.get("position") or "").strip()
                if "observations" in item:
                    normalized["observations"] = str(item.get("observations") or "").strip()
                if "qualitative_tags" in item:
                    raw_tags = item.get("qualitative_tags")
                    normalized["qualitative_tags"] = raw_tags if isinstance(raw_tags, list) else []
                output.append(normalized)
            return output
    return [
        {
            "team_a": item.strip(),
            "score_a": None,
            "score_b": None,
            "team_b": "",
            "competition": "",
            "match_date": "",
            "player_rating": None,
        }
        for item in raw.split("|")
        if item.strip()
    ]


def normalize_star_rating(row: dict[str, str]) -> float | None:
    star = to_float(row.get("star_rating"))
    if star is not None:
        return max(1, min(5, round(star)))
    overall = to_float(row.get("overall_rating"))
    if overall is not None and overall > 0:
        return max(1, min(5, round(overall / 2)))
    ratings = [
        to_float(row.get("technical_rating")),
        to_float(row.get("physical_rating")),
        to_float(row.get("tactical_rating")),
        to_float(row.get("mental_rating")),
        to_float(row.get("potential_rating")),
    ]
    present = [item for item in ratings if item is not None and item > 0]
    if present:
        return max(1, min(5, round((sum(present) / len(present)) / 2)))
    return None


def resolve_youth_ranking_id(session, eyeball_player_id: str | None, row: dict[str, Any]) -> int | None:
    if eyeball_player_id:
        found = session.execute(
            sa_text(
                """
                SELECT id
                FROM youth_player_rankings
                WHERE provider = 'eyeball'
                  AND provider_player_id = :provider_player_id
                ORDER BY season DESC, minutes_played DESC NULLS LAST, score DESC NULLS LAST
                LIMIT 1
                """
            ),
            {"provider_player_id": eyeball_player_id},
        ).fetchone()
        if found:
            return int(found.id)
    name = blank_to_none(row.get("player_name") or row.get("name"))
    if not name:
        return None
    found = session.execute(
        sa_text(
            """
            SELECT id
            FROM youth_player_rankings
            WHERE LOWER(display_name) = LOWER(:name)
              AND COALESCE(birth_year, 0) = COALESCE(:birth_year, 0)
              AND LOWER(COALESCE(club_name, '')) = LOWER(COALESCE(:club, ''))
            ORDER BY season DESC, minutes_played DESC NULLS LAST, score DESC NULLS LAST
            LIMIT 1
            """
        ),
        {
            "name": name,
            "birth_year": to_int(row.get("year_of_birth")),
            "club": blank_to_none(row.get("club")),
        },
    ).fetchone()
    return int(found.id) if found else None


def upsert_player(session, row: dict[str, str], *, portal_url: str | None = None) -> str:
    player_id = to_uuid(row.get("id") or row.get("player_id"))
    name = blank_to_none(row.get("name") or row.get("player_name"))
    if not name:
        raise ValueError("player name is required")
    eyeball_player_id = extract_eyeball_player_id(portal_url)
    session.execute(
        sa_text(
            """
            INSERT INTO scouting_players (
              id, source, source_player_id, eyeball_player_id, portal_url, name, club,
              year_of_birth, position, nationality, photo_key, created_at, updated_at
            )
            VALUES (
              :id, 'scoutyourlegend', :source_player_id, :eyeball_player_id, :portal_url, :name,
              :club, :year_of_birth, :position, :nationality, :photo_key, NOW(), NOW()
            )
            ON CONFLICT (id) DO UPDATE SET
              source_player_id = COALESCE(EXCLUDED.source_player_id, scouting_players.source_player_id),
              eyeball_player_id = COALESCE(EXCLUDED.eyeball_player_id, scouting_players.eyeball_player_id),
              portal_url = COALESCE(EXCLUDED.portal_url, scouting_players.portal_url),
              name = EXCLUDED.name,
              club = EXCLUDED.club,
              year_of_birth = EXCLUDED.year_of_birth,
              position = EXCLUDED.position,
              nationality = EXCLUDED.nationality,
              photo_key = COALESCE(EXCLUDED.photo_key, scouting_players.photo_key),
              updated_at = NOW()
            """
        ),
        {
            "id": player_id,
            "source_player_id": player_id,
            "eyeball_player_id": eyeball_player_id,
            "portal_url": portal_url,
            "name": name,
            "club": blank_to_none(row.get("club")),
            "year_of_birth": to_int(row.get("year_of_birth")),
            "position": blank_to_none(row.get("position")),
            "nationality": blank_to_none(row.get("nationality")),
            "photo_key": blank_to_none(row.get("photo_key")),
        },
    )
    return player_id


def upsert_report(session, row: dict[str, str], players_by_id: dict[str, dict[str, str]]) -> str:
    report_id = to_uuid(row.get("id"))
    player_id = to_uuid(row.get("player_id"))
    base_player = players_by_id.get(player_id, {})
    merged_player = {
        "id": player_id,
        "name": row.get("player_name") or base_player.get("name"),
        "club": row.get("club") or base_player.get("club"),
        "year_of_birth": row.get("year_of_birth") or base_player.get("year_of_birth"),
        "position": row.get("position") or base_player.get("position"),
        "nationality": row.get("nationality") or base_player.get("nationality"),
        "photo_key": row.get("photo_key") or base_player.get("photo_key"),
    }
    portal_url = blank_to_none(row.get("portal_url"))
    player_id = upsert_player(session, merged_player, portal_url=portal_url)
    eyeball_player_id = extract_eyeball_player_id(portal_url)
    linked_youth_ranking_id = resolve_youth_ranking_id(session, eyeball_player_id, row)
    star_rating = normalize_star_rating(row)
    overall_rating = to_float(row.get("overall_rating"))
    if overall_rating is None and star_rating is not None:
        overall_rating = star_rating * 2
    session.execute(
        sa_text(
            """
            INSERT INTO scouting_reports (
              id, player_id, linked_youth_ranking_id, eyeball_player_id, portal_url,
              player_name, club, year_of_birth, position, nationality, scout,
              created_at, updated_at, matches_observed,
              technical_notes, physical_notes, tactical_notes, mental_notes, game_intelligence,
              strengths, weaknesses, development_projection, comparison, overall_comments,
              technical_rating, physical_rating, tactical_rating, mental_rating, potential_rating,
              overall_rating, star_rating, potential_star_rating, photo_key, source, raw_payload
            )
            VALUES (
              :id, :player_id, :linked_youth_ranking_id, :eyeball_player_id, :portal_url,
              :player_name, :club, :year_of_birth, :position, :nationality, :scout,
              COALESCE(CAST(:created_at AS timestamptz), NOW()), NOW(), CAST(:matches_observed AS jsonb),
              :technical_notes, :physical_notes, :tactical_notes, :mental_notes, :game_intelligence,
              :strengths, :weaknesses, :development_projection, :comparison, :overall_comments,
              :technical_rating, :physical_rating, :tactical_rating, :mental_rating, :potential_rating,
              :overall_rating, :star_rating, :potential_star_rating, :photo_key, 'scoutyourlegend', CAST(:raw_payload AS jsonb)
            )
            ON CONFLICT (id) DO UPDATE SET
              player_id = EXCLUDED.player_id,
              linked_youth_ranking_id = COALESCE(EXCLUDED.linked_youth_ranking_id, scouting_reports.linked_youth_ranking_id),
              eyeball_player_id = COALESCE(EXCLUDED.eyeball_player_id, scouting_reports.eyeball_player_id),
              portal_url = COALESCE(EXCLUDED.portal_url, scouting_reports.portal_url),
              player_name = EXCLUDED.player_name,
              club = EXCLUDED.club,
              year_of_birth = EXCLUDED.year_of_birth,
              position = EXCLUDED.position,
              nationality = EXCLUDED.nationality,
              scout = EXCLUDED.scout,
              matches_observed = EXCLUDED.matches_observed,
              technical_notes = EXCLUDED.technical_notes,
              physical_notes = EXCLUDED.physical_notes,
              tactical_notes = EXCLUDED.tactical_notes,
              mental_notes = EXCLUDED.mental_notes,
              game_intelligence = EXCLUDED.game_intelligence,
              strengths = EXCLUDED.strengths,
              weaknesses = EXCLUDED.weaknesses,
              development_projection = EXCLUDED.development_projection,
              comparison = EXCLUDED.comparison,
              overall_comments = EXCLUDED.overall_comments,
              technical_rating = EXCLUDED.technical_rating,
              physical_rating = EXCLUDED.physical_rating,
              tactical_rating = EXCLUDED.tactical_rating,
              mental_rating = EXCLUDED.mental_rating,
              potential_rating = EXCLUDED.potential_rating,
              overall_rating = EXCLUDED.overall_rating,
              star_rating = EXCLUDED.star_rating,
              potential_star_rating = COALESCE(EXCLUDED.potential_star_rating, scouting_reports.potential_star_rating),
              photo_key = COALESCE(EXCLUDED.photo_key, scouting_reports.photo_key),
              raw_payload = EXCLUDED.raw_payload,
              updated_at = NOW()
            """
        ),
        {
            "id": report_id,
            "player_id": player_id,
            "linked_youth_ranking_id": linked_youth_ranking_id,
            "eyeball_player_id": eyeball_player_id,
            "portal_url": portal_url,
            "player_name": blank_to_none(row.get("player_name")) or blank_to_none(merged_player.get("name")),
            "club": blank_to_none(row.get("club")) or blank_to_none(merged_player.get("club")),
            "year_of_birth": to_int(row.get("year_of_birth") or merged_player.get("year_of_birth")),
            "position": blank_to_none(row.get("position")) or blank_to_none(merged_player.get("position")),
            "nationality": blank_to_none(row.get("nationality")) or blank_to_none(merged_player.get("nationality")),
            "scout": blank_to_none(row.get("scout")),
            "created_at": blank_to_none(row.get("created_at")),
            "matches_observed": json.dumps(normalize_matches(row.get("matches_observed"))),
            "technical_notes": blank_to_none(row.get("technical_notes")),
            "physical_notes": blank_to_none(row.get("physical_notes")),
            "tactical_notes": blank_to_none(row.get("tactical_notes")),
            "mental_notes": blank_to_none(row.get("mental_notes")),
            "game_intelligence": blank_to_none(row.get("game_intelligence")),
            "strengths": blank_to_none(row.get("strengths")),
            "weaknesses": blank_to_none(row.get("weaknesses")),
            "development_projection": blank_to_none(row.get("development_projection")),
            "comparison": blank_to_none(row.get("comparison")),
            "overall_comments": blank_to_none(row.get("overall_comments")),
            "technical_rating": to_float(row.get("technical_rating")),
            "physical_rating": to_float(row.get("physical_rating")),
            "tactical_rating": to_float(row.get("tactical_rating")),
            "mental_rating": to_float(row.get("mental_rating")),
            "potential_rating": to_float(row.get("potential_rating")),
            "overall_rating": overall_rating,
            "star_rating": star_rating,
            "potential_star_rating": to_float(row.get("potential_star_rating")),
            "photo_key": blank_to_none(row.get("photo_key")) or blank_to_none(merged_player.get("photo_key")),
            "raw_payload": json.dumps(row, ensure_ascii=False),
        },
    )
    return report_id


def run_import(players_path: Path, reports_path: Path, dry_run: bool = False) -> dict[str, int]:
    players = read_csv(players_path)
    reports = read_csv(reports_path)
    players_by_id = {to_uuid(row.get("id")): row for row in players if row.get("id")}
    eyeball_ids = [extract_eyeball_player_id(row.get("portal_url")) for row in reports]
    summary = {
        "players_csv": len(players),
        "reports_csv": len(reports),
        "reports_with_eyeball_id": sum(1 for item in eyeball_ids if item),
        "players_imported": 0,
        "reports_imported": 0,
        "reports_linked_to_youth": 0,
    }
    if dry_run:
        return summary
    with session_local()() as session:
        ensure_youth_schema(session)
        for row in players:
            if not blank_to_none(row.get("name")):
                continue
            upsert_player(session, row)
            summary["players_imported"] += 1
        for row in reports:
            if not blank_to_none(row.get("player_name")):
                continue
            report_id = upsert_report(session, row, players_by_id)
            linked = session.execute(
                sa_text("SELECT linked_youth_ranking_id FROM scouting_reports WHERE id = :id"),
                {"id": report_id},
            ).fetchone()
            if linked and linked.linked_youth_ranking_id:
                summary["reports_linked_to_youth"] += 1
            summary["reports_imported"] += 1
        session.commit()
    return summary


def main_cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--players", type=Path, default=DEFAULT_IMPORT_DIR / "players.csv")
    parser.add_argument("--reports", type=Path, default=DEFAULT_IMPORT_DIR / "reports.csv")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    summary = run_import(args.players, args.reports, dry_run=args.dry_run)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main_cli()
