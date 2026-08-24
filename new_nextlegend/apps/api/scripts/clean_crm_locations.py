#!/usr/bin/env python3
"""Clean CRM club city/country values using curated football location fixes.

Default mode is dry-run. Use --apply to update the database.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
import unicodedata

from sqlalchemy import text

API_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

import main  # noqa: E402


def normalize_key(value: str | None) -> str:
    raw = str(value or "").strip().lower()
    normalized = unicodedata.normalize("NFKD", raw)
    ascii_only = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", " ", ascii_only).strip()


def main_cli() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Persist updates. Without it, only prints the planned changes.")
    args = parser.parse_args()

    fixes = main.CRM_CLUB_LOCATION_FIXES
    planned: list[tuple[str, str, str, str, str, str]] = []
    skipped = 0

    with main.SessionLocal() as session:
        main._ensure_crm_schema(session)
        rows = session.execute(text("SELECT id, name, city, country FROM crm_clubs ORDER BY LOWER(name)")).fetchall()
        for row in rows:
            item = dict(row._mapping)
            normalized_name = normalize_key(item["name"])
            location = fixes.get(normalized_name)
            if not location:
                skipped += 1
                continue
            next_city, next_country = location
            if item.get("city") == next_city and item.get("country") == next_country:
                continue
            planned.append((item["id"], item["name"], item.get("city") or "", item.get("country") or "", next_city, next_country))

        for club_id, _, _, _, next_city, next_country in planned:
            if args.apply:
                session.execute(
                    text(
                        """
                        UPDATE crm_clubs
                        SET city = :city,
                            country = :country,
                            updated_at = NOW()
                        WHERE id = :id
                        """
                    ),
                    {"id": club_id, "city": next_city, "country": next_country},
                )
        if args.apply:
            session.commit()

    mode = "APPLIED" if args.apply else "DRY-RUN"
    print(f"[{mode}] planned_updates={len(planned)} skipped_without_fix={skipped}")
    for _, name, old_city, old_country, next_city, next_country in planned:
        print(f"{name}: {old_city} / {old_country} -> {next_city} / {next_country}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
