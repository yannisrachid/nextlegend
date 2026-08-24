from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from typing import Any

import psycopg
from psycopg.rows import dict_row


CRM_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS crm_clubs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    city TEXT NOT NULL,
    country TEXT NOT NULL,
    logo TEXT,
    email TEXT,
    phone TEXT,
    website TEXT,
    source TEXT NOT NULL DEFAULT 'nextlegend',
    source_id TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS crm_players (
    id TEXT PRIMARY KEY,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    age INT NOT NULL DEFAULT 0,
    position TEXT NOT NULL,
    nationality TEXT NOT NULL,
    photo TEXT,
    email TEXT,
    phone TEXT,
    club_id TEXT NOT NULL REFERENCES crm_clubs(id) ON DELETE CASCADE,
    source TEXT NOT NULL DEFAULT 'nextlegend',
    source_id TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS crm_contacts (
    id TEXT PRIMARY KEY,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    role TEXT NOT NULL,
    email TEXT,
    phone TEXT,
    type TEXT NOT NULL DEFAULT 'CLUB',
    notes TEXT,
    club_id TEXT REFERENCES crm_clubs(id) ON DELETE SET NULL,
    player_id TEXT REFERENCES crm_players(id) ON DELETE SET NULL,
    source TEXT NOT NULL DEFAULT 'nextlegend',
    source_id TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT crm_contacts_type_check CHECK (type IN ('CLUB', 'PLAYER'))
);

CREATE TABLE IF NOT EXISTS crm_prospects (
    id TEXT PRIMARY KEY,
    stage TEXT NOT NULL DEFAULT 'prequalification',
    notes TEXT,
    contact_id TEXT NOT NULL REFERENCES crm_contacts(id) ON DELETE CASCADE,
    source TEXT NOT NULL DEFAULT 'nextlegend',
    source_id TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT crm_prospects_stage_check CHECK (stage IN ('prequalification', 'relance1', 'relance2', 'relance3'))
);

CREATE INDEX IF NOT EXISTS crm_clubs_search_idx ON crm_clubs (LOWER(name), LOWER(city), LOWER(country));
CREATE INDEX IF NOT EXISTS crm_players_club_idx ON crm_players (club_id);
CREATE INDEX IF NOT EXISTS crm_contacts_club_idx ON crm_contacts (club_id);
CREATE INDEX IF NOT EXISTS crm_contacts_player_idx ON crm_contacts (player_id);
CREATE INDEX IF NOT EXISTS crm_contacts_type_idx ON crm_contacts (type);
CREATE INDEX IF NOT EXISTS crm_prospects_contact_idx ON crm_prospects (contact_id);
CREATE INDEX IF NOT EXISTS crm_prospects_stage_idx ON crm_prospects (stage);
"""


def normalize_url(value: str) -> str:
    return re.sub(r"^postgresql\+[^:]+://", "postgresql://", value.strip())


def load_dotenv() -> None:
    try:
        from dotenv import load_dotenv as load
    except Exception:
        return
    load()


def clean(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else None
    return value


def ts(value: Any) -> Any:
    return value or datetime.now(timezone.utc)


def ensure_schema(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        for statement in [chunk.strip() for chunk in CRM_SCHEMA_SQL.split(";") if chunk.strip()]:
            cur.execute(statement)
    conn.commit()


def fetch_all(conn: psycopg.Connection, sql: str) -> list[dict[str, Any]]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql)
        return list(cur.fetchall())


def migrate_clubs(source: psycopg.Connection, target: psycopg.Connection) -> int:
    rows = fetch_all(source, 'SELECT id, name, city, country, logo, email, phone, website, "createdAt", "updatedAt" FROM clubs ORDER BY id')
    with target.cursor() as cur:
        for row in rows:
            cur.execute(
                """
                INSERT INTO crm_clubs (id, name, city, country, logo, email, phone, website, source, source_id, created_at, updated_at)
                VALUES (%(id)s, %(name)s, %(city)s, %(country)s, %(logo)s, %(email)s, %(phone)s, %(website)s, 'findyourlegend', %(id)s, %(createdAt)s, %(updatedAt)s)
                ON CONFLICT (id) DO UPDATE SET
                  name = EXCLUDED.name,
                  city = EXCLUDED.city,
                  country = EXCLUDED.country,
                  logo = EXCLUDED.logo,
                  email = EXCLUDED.email,
                  phone = EXCLUDED.phone,
                  website = EXCLUDED.website,
                  updated_at = EXCLUDED.updated_at
                """,
                {**{key: clean(value) for key, value in row.items()}, "createdAt": ts(row["createdAt"]), "updatedAt": ts(row["updatedAt"])},
            )
    target.commit()
    return len(rows)


def migrate_players(source: psycopg.Connection, target: psycopg.Connection) -> int:
    rows = fetch_all(source, 'SELECT id, "firstName", "lastName", age, position, nationality, photo, email, phone, "clubId", "createdAt", "updatedAt" FROM players ORDER BY id')
    with target.cursor() as cur:
        for row in rows:
            cur.execute(
                """
                INSERT INTO crm_players (id, first_name, last_name, age, position, nationality, photo, email, phone, club_id, source, source_id, created_at, updated_at)
                VALUES (%(id)s, %(firstName)s, %(lastName)s, %(age)s, %(position)s, %(nationality)s, %(photo)s, %(email)s, %(phone)s, %(clubId)s, 'findyourlegend', %(id)s, %(createdAt)s, %(updatedAt)s)
                ON CONFLICT (id) DO UPDATE SET
                  first_name = EXCLUDED.first_name,
                  last_name = EXCLUDED.last_name,
                  age = EXCLUDED.age,
                  position = EXCLUDED.position,
                  nationality = EXCLUDED.nationality,
                  photo = EXCLUDED.photo,
                  email = EXCLUDED.email,
                  phone = EXCLUDED.phone,
                  club_id = EXCLUDED.club_id,
                  updated_at = EXCLUDED.updated_at
                """,
                {**{key: clean(value) for key, value in row.items()}, "age": int(row["age"] or 0), "lastName": clean(row["lastName"]) or "", "createdAt": ts(row["createdAt"]), "updatedAt": ts(row["updatedAt"])},
            )
    target.commit()
    return len(rows)


def migrate_contacts(source: psycopg.Connection, target: psycopg.Connection) -> int:
    rows = fetch_all(source, 'SELECT id, "firstName", "lastName", role, email, phone, type, notes, "clubId", "playerId", "createdAt", "updatedAt" FROM contacts ORDER BY id')
    with target.cursor() as cur:
        for row in rows:
            contact_type = row["type"] if row["type"] in ("CLUB", "PLAYER") else "CLUB"
            cur.execute(
                """
                INSERT INTO crm_contacts (id, first_name, last_name, role, email, phone, type, notes, club_id, player_id, source, source_id, created_at, updated_at)
                VALUES (%(id)s, %(firstName)s, %(lastName)s, %(role)s, %(email)s, %(phone)s, %(type)s, %(notes)s, %(clubId)s, %(playerId)s, 'findyourlegend', %(id)s, %(createdAt)s, %(updatedAt)s)
                ON CONFLICT (id) DO UPDATE SET
                  first_name = EXCLUDED.first_name,
                  last_name = EXCLUDED.last_name,
                  role = EXCLUDED.role,
                  email = EXCLUDED.email,
                  phone = EXCLUDED.phone,
                  type = EXCLUDED.type,
                  notes = EXCLUDED.notes,
                  club_id = EXCLUDED.club_id,
                  player_id = EXCLUDED.player_id,
                  updated_at = EXCLUDED.updated_at
                """,
                {
                    **{key: clean(value) for key, value in row.items()},
                    "firstName": clean(row["firstName"]) or "",
                    "lastName": clean(row["lastName"]) or "",
                    "role": clean(row["role"]) or "Unknown",
                    "type": contact_type,
                    "createdAt": ts(row["createdAt"]),
                    "updatedAt": ts(row["updatedAt"]),
                },
            )
    target.commit()
    return len(rows)


def migrate_prospects(source: psycopg.Connection, target: psycopg.Connection) -> int:
    rows = fetch_all(source, 'SELECT id, stage, notes, "contactId", "createdAt", "updatedAt" FROM prospects ORDER BY id')
    with target.cursor() as cur:
        for row in rows:
            cur.execute(
                """
                INSERT INTO crm_prospects (id, stage, notes, contact_id, source, source_id, created_at, updated_at)
                VALUES (%(id)s, %(stage)s, %(notes)s, %(contactId)s, 'findyourlegend', %(id)s, %(createdAt)s, %(updatedAt)s)
                ON CONFLICT (id) DO UPDATE SET
                  stage = EXCLUDED.stage,
                  notes = EXCLUDED.notes,
                  contact_id = EXCLUDED.contact_id,
                  updated_at = EXCLUDED.updated_at
                """,
                {**{key: clean(value) for key, value in row.items()}, "createdAt": ts(row["createdAt"]), "updatedAt": ts(row["updatedAt"])},
            )
    target.commit()
    return len(rows)


def table_counts(conn: psycopg.Connection, prefix: str = "") -> dict[str, int]:
    tables = {
        "clubs": f"{prefix}clubs",
        "players": f"{prefix}players",
        "contacts": f"{prefix}contacts",
        "prospects": f"{prefix}prospects",
    }
    counts: dict[str, int] = {}
    with conn.cursor() as cur:
        for key, table_name in tables.items():
            cur.execute(f"SELECT COUNT(*) FROM {table_name}")
            row = cur.fetchone()
            value = next(iter(row.values())) if isinstance(row, dict) else row[0]
            counts[key] = int(value)
    return counts


def main() -> int:
    load_dotenv()
    source_url = os.getenv("CRM_SOURCE_DATABASE_URL") or os.getenv("CRM_NEON_DATABASE_URL")
    target_url = os.getenv("DATABASE_URL")
    if not source_url:
        raise SystemExit("Set CRM_SOURCE_DATABASE_URL or CRM_NEON_DATABASE_URL.")
    if not target_url:
        raise SystemExit("Set DATABASE_URL for the target NextLegend database.")

    with psycopg.connect(normalize_url(source_url), row_factory=dict_row) as source:
        with psycopg.connect(normalize_url(target_url)) as target:
            ensure_schema(target)
            source_counts = table_counts(source)
            counts = {
                "clubs": migrate_clubs(source, target),
                "players": migrate_players(source, target),
                "contacts": migrate_contacts(source, target),
                "prospects": migrate_prospects(source, target),
            }
            target_counts = table_counts(target, "crm_")
    print({"source": source_counts, "migrated": counts, "target": target_counts})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
