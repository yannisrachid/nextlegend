#!/usr/bin/env python3
"""Clean imported CRM clubs, players and notes.

Default mode is dry-run. Use --apply to persist changes.
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


PLACEHOLDER_VALUES = {"", "-", "n/a", "na", "none", "null", "nan", "undefined"}

POSITION_FIXES = {
    "cen midfield": "Central Midfield",
    "cen midfield": "Central Midfield",
    "cen midfiled": "Central Midfield",
    "cen.midfield": "Central Midfield",
    "central-midfield": "Central Midfield",
    "def midfield": "Defensive Midfield",
    "def midfield": "Defensive Midfield",
    "def.midfield": "Defensive Midfield",
    "def. midfield": "Defensive Midfield",
    "def.midfield": "Defensive Midfield",
    "leftback": "Left Back",
    "left back": "Left Back",
    "left-midfield": "Left Midfield",
    "left midfield": "Left Midfield",
    "left-wing": "Left Wing",
    "leftwing": "Left Wing",
    "om": "Attacking Midfield",
    "om midfield": "Attacking Midfield",
    "om.midfield": "Attacking Midfield",
    "of midfield": "Attacking Midfield",
    "of.midfield": "Attacking Midfield",
    "off.midfield": "Attacking Midfield",
    "rightback": "Right Back",
    "right back": "Right Back",
    "rightwing": "Right Wing",
    "rigth wing": "Right Wing",
    "centreback": "Centre Back",
    "centre back": "Centre Back",
    "centre-back": "Centre Back",
    "centr back": "Centre Back",
}

NATIONALITY_FIXES = {
    "aserbaidschan": "Azerbaijan",
    "belgiumn": "Belgium",
    "brasilia": "Brazil",
    "croatian": "Croatia",
    "czech": "Czech Republic",
    "czech rep": "Czech Republic",
    "equador": "Ecuador",
    "french": "France",
    "kasachstan": "Kazakhstan",
    "lettland": "Latvia",
    "lituaen": "Lithuania",
    "marocco": "Morocco",
    "mazedonia": "North Macedonia",
    "mexiko": "Mexico",
    "moldawia": "Moldova",
    "montengro": "Montenegro",
    "slovenia": "Slovenia",
    "slowenia": "Slovenia",
    "spanish": "Spain",
    "swiss": "Switzerland",
    "ukraina": "Ukraine",
}

COUNTRY_NAME_FIXES = {
    "arzebajan": "Azerbaijan",
    "fyr macedonia": "North Macedonia",
    "moldavia": "Moldova",
    "the netherlands": "Netherlands",
}

NOTE_METADATA_RE = re.compile(r"^(region|country|tm_id|website|tier|status)\s*:", re.I)
CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")


def normalize_key(value: str | None) -> str:
    raw = str(value or "").strip().lower()
    normalized = unicodedata.normalize("NFKD", raw)
    ascii_only = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", " ", ascii_only).strip()


def clean_text(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = CONTROL_CHARS_RE.sub(" ", str(value))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return None if cleaned.lower() in PLACEHOLDER_VALUES else cleaned


def clean_optional_url(value: str | None) -> str | None:
    cleaned = clean_text(value)
    if not cleaned or cleaned.lower() == "web":
        return None
    if cleaned.startswith("www."):
        return f"https://{cleaned}"
    return cleaned


def clean_position(value: str | None) -> str:
    cleaned = clean_text(value) or "Player"
    normalized = normalize_key(cleaned)
    return POSITION_FIXES.get(normalized, cleaned)


def clean_nationality(value: str | None) -> str:
    cleaned = clean_text(value) or "Unknown"
    normalized = normalize_key(cleaned)
    return NATIONALITY_FIXES.get(normalized, cleaned)


def clean_club_name(value: str | None) -> str:
    cleaned = clean_text(value) or ""
    cleaned = re.sub(r"\s+:", ":", cleaned)
    return COUNTRY_NAME_FIXES.get(normalize_key(cleaned.rstrip(":")), cleaned)


def clean_note(value: str | None) -> str | None:
    cleaned = clean_text(value)
    if not cleaned:
        return None
    parts = [part.strip() for part in cleaned.split("|")]
    kept = []
    for part in parts:
        if not part:
            continue
        if part.lower() in PLACEHOLDER_VALUES:
            continue
        if NOTE_METADATA_RE.match(part):
            continue
        kept.append(part)
    if not kept:
        return None
    return " | ".join(kept)


def is_placeholder_club(row: dict) -> bool:
    name = str(row.get("name") or "")
    city = str(row.get("city") or "")
    country = str(row.get("country") or "")
    website = str(row.get("website") or "").strip().lower()
    player_count = int(row.get("player_count") or 0)
    if player_count:
        return False
    has_header_name = ":" in name or any("\U0001F1E6" <= ch <= "\U0001F1FF" for ch in name)
    has_tier_location = city.strip().lower() == "tier" or country.strip().lower() == "tier"
    return website == "web" and (has_header_name or has_tier_location)


def main_cli() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Persist updates. Without it, only prints a dry-run summary.")
    args = parser.parse_args()

    club_updates = []
    player_updates = []
    contact_note_updates = []
    contact_role_updates = []
    prospect_note_updates = []
    placeholder_club_deletes = []

    with main.SessionLocal() as session:
        main._ensure_crm_schema(session)

        club_rows = session.execute(
            text(
                """
                SELECT c.*,
                  (SELECT COUNT(*) FROM crm_players p WHERE p.club_id = c.id) AS player_count,
                  (SELECT COUNT(*) FROM crm_contacts ct WHERE ct.club_id = c.id) AS contact_count
                FROM crm_clubs c
                ORDER BY LOWER(c.name)
                """
            )
        ).fetchall()
        for row in club_rows:
            item = dict(row._mapping)
            if is_placeholder_club(item):
                placeholder_club_deletes.append(item)
                continue
            next_values = {
                "name": clean_club_name(item.get("name")) or item.get("name"),
                "city": clean_text(item.get("city")) or item.get("city"),
                "country": clean_text(item.get("country")) or item.get("country"),
                "logo": clean_optional_url(item.get("logo")),
                "email": clean_text(item.get("email")),
                "phone": clean_text(item.get("phone")),
                "website": clean_optional_url(item.get("website")),
            }
            if any(item.get(key) != value for key, value in next_values.items()):
                club_updates.append((item, next_values))

        player_rows = session.execute(text("SELECT * FROM crm_players ORDER BY LOWER(first_name), LOWER(last_name)")).fetchall()
        for row in player_rows:
            item = dict(row._mapping)
            next_values = {
                "first_name": clean_text(item.get("first_name")) or item.get("first_name"),
                "last_name": clean_text(item.get("last_name")) or "",
                "position": clean_position(item.get("position")),
                "nationality": clean_nationality(item.get("nationality")),
                "photo": clean_optional_url(item.get("photo")),
                "email": clean_text(item.get("email")),
                "phone": clean_text(item.get("phone")),
            }
            if any(item.get(key) != value for key, value in next_values.items()):
                player_updates.append((item, next_values))

        contact_rows = session.execute(text("SELECT id, role, notes FROM crm_contacts")).fetchall()
        for row in contact_rows:
            item = dict(row._mapping)
            next_role = clean_position(item.get("role"))
            if item.get("role") != next_role:
                contact_role_updates.append((item, next_role))
            next_note = clean_note(item.get("notes"))
            if item.get("notes") != next_note:
                contact_note_updates.append((item, next_note))

        prospect_rows = session.execute(text("SELECT id, notes FROM crm_prospects WHERE notes IS NOT NULL")).fetchall()
        for row in prospect_rows:
            item = dict(row._mapping)
            next_note = clean_note(item.get("notes"))
            if item.get("notes") != next_note:
                prospect_note_updates.append((item, next_note))

        if args.apply:
            for item, next_values in club_updates:
                session.execute(
                    text(
                        """
                        UPDATE crm_clubs
                        SET name=:name, city=:city, country=:country, logo=:logo, email=:email,
                            phone=:phone, website=:website, updated_at=NOW()
                        WHERE id=:id
                        """
                    ),
                    {"id": item["id"], **next_values},
                )
            for item, next_values in player_updates:
                session.execute(
                    text(
                        """
                        UPDATE crm_players
                        SET first_name=:first_name, last_name=:last_name, position=:position,
                            nationality=:nationality, photo=:photo, email=:email, phone=:phone,
                            updated_at=NOW()
                        WHERE id=:id
                        """
                    ),
                    {"id": item["id"], **next_values},
                )
            for item, next_note in contact_note_updates:
                session.execute(text("UPDATE crm_contacts SET notes=:notes, updated_at=NOW() WHERE id=:id"), {"id": item["id"], "notes": next_note})
            for item, next_role in contact_role_updates:
                session.execute(text("UPDATE crm_contacts SET role=:role, updated_at=NOW() WHERE id=:id"), {"id": item["id"], "role": next_role})
            for item, next_note in prospect_note_updates:
                session.execute(text("UPDATE crm_prospects SET notes=:notes, updated_at=NOW() WHERE id=:id"), {"id": item["id"], "notes": next_note})
            for item in placeholder_club_deletes:
                session.execute(text("DELETE FROM crm_clubs WHERE id=:id"), {"id": item["id"]})
            session.commit()

    mode = "APPLIED" if args.apply else "DRY-RUN"
    print(f"[{mode}] club_updates={len(club_updates)}")
    print(f"[{mode}] player_updates={len(player_updates)}")
    print(f"[{mode}] contact_note_updates={len(contact_note_updates)}")
    print(f"[{mode}] contact_role_updates={len(contact_role_updates)}")
    print(f"[{mode}] prospect_note_updates={len(prospect_note_updates)}")
    print(f"[{mode}] placeholder_club_deletes={len(placeholder_club_deletes)}")
    print("\nPlaceholder clubs to delete/unlink contacts:")
    for item in placeholder_club_deletes[:80]:
        print(f"- {item['name']} ({item['city']} / {item['country']}) contacts={item.get('contact_count') or 0}")
    print("\nPlayer update samples:")
    for item, next_values in player_updates[:40]:
        print(f"- {item['first_name']} {item['last_name']}: position {item['position']!r}->{next_values['position']!r}, nationality {item['nationality']!r}->{next_values['nationality']!r}")
    print("\nContact note update samples:")
    for item, next_note in contact_note_updates[:30]:
        before = re.sub(r"\s+", " ", str(item.get("notes") or ""))[:160]
        after = re.sub(r"\s+", " ", str(next_note or ""))[:160]
        print(f"- {item['id']}: {before!r} -> {after!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
