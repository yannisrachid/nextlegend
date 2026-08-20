#!/usr/bin/env python3
"""Build a Postgres import script for Wyscout transfer history XLSX files."""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import unicodedata
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path
from xml.etree import ElementTree as ET


NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
HEADERS = [
    "league_id",
    "league_name",
    "team_id_context",
    "team_name_context",
    "player_id",
    "player_name",
    "transfer_date",
    "transfer_type",
    "transfer_fee",
    "team_in_id",
    "team_in_name",
    "team_out_id",
    "team_out_name",
    "transfer_date_dt",
]
STAGING_COLUMNS = [
    "source",
    "source_player_id",
    "normalized_player_name",
    "player_name",
    "league_id",
    "league_name",
    "team_id_context",
    "team_name_context",
    "transfer_date",
    "transfer_type",
    "transfer_fee",
    "team_in_id",
    "team_in_name",
    "team_out_id",
    "team_out_name",
    "transfer_date_serial",
    "raw_payload",
]


def normalize_phrase(value: str) -> str:
    cleaned = unicodedata.normalize("NFKD", value or "")
    cleaned = "".join(ch for ch in cleaned if not unicodedata.combining(ch))
    cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned.lower())
    return " ".join(cleaned.split())


def int_text(value: str) -> str:
    clean = str(value or "").strip()
    if not clean:
        return ""
    try:
        return str(int(float(clean)))
    except ValueError:
        return ""


def float_text(value: str) -> str:
    clean = str(value or "").strip()
    if not clean:
        return ""
    try:
        return str(float(clean))
    except ValueError:
        return ""


def excel_date(serial: str) -> str:
    clean = str(serial or "").strip()
    if not clean:
        return ""
    try:
        value = float(clean)
    except ValueError:
        return ""
    return (date(1899, 12, 30) + timedelta(days=int(value))).isoformat()


def iso_date(value: str, serial: str) -> str:
    clean = str(value or "").strip()
    if clean:
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
            try:
                return datetime.strptime(clean[:10], fmt).date().isoformat()
            except ValueError:
                pass
    return excel_date(serial)


def load_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    values: list[str] = []
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    for item in root.findall("x:si", NS):
        parts = [node.text or "" for node in item.findall(".//x:t", NS)]
        values.append("".join(parts))
    return values


def cell_value(cell: ET.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(node.text or "" for node in cell.findall(".//x:t", NS)).strip()
    value_node = cell.find("x:v", NS)
    if value_node is None or value_node.text is None:
        return ""
    raw = value_node.text.strip()
    if cell_type == "s":
        try:
            return shared_strings[int(raw)].strip()
        except (IndexError, ValueError):
            return ""
    return raw


def iter_sheet_rows(path: Path):
    with zipfile.ZipFile(path) as zf:
        shared_strings = load_shared_strings(zf)
        sheet_name = "xl/worksheets/sheet1.xml"
        with zf.open(sheet_name) as handle:
            for _, row in ET.iterparse(handle, events=("end",)):
                if row.tag != f"{{{NS['x']}}}row":
                    continue
                cells = [cell_value(cell, shared_strings) for cell in row.findall("x:c", NS)]
                yield cells
                row.clear()


def build_rows(path: Path, limit: int | None = None) -> list[list[str]]:
    rows: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    iterator = iter(iter_sheet_rows(path))
    headers = next(iterator, [])
    if headers[: len(HEADERS)] != HEADERS:
        raise SystemExit(f"Unexpected XLSX headers: {headers}")
    for raw in iterator:
        values = dict(zip(HEADERS, raw + [""] * (len(HEADERS) - len(raw))))
        player_name = values.get("player_name", "").strip()
        if not player_name:
            continue
        transfer_date = iso_date(values.get("transfer_date", ""), values.get("transfer_date_dt", ""))
        row_key = (
            int_text(values.get("player_id", "")),
            player_name,
            transfer_date,
            values.get("transfer_type", "").strip(),
            values.get("team_in_name", "").strip(),
            values.get("team_out_name", "").strip(),
            values.get("transfer_fee", "").strip(),
        )
        if row_key in seen:
            continue
        seen.add(row_key)
        raw_payload = {key: values.get(key, "") for key in HEADERS}
        rows.append(
            [
                "transferts.xlsx",
                int_text(values.get("player_id", "")),
                normalize_phrase(player_name),
                player_name,
                int_text(values.get("league_id", "")),
                values.get("league_name", "").strip(),
                int_text(values.get("team_id_context", "")),
                values.get("team_name_context", "").strip(),
                transfer_date,
                values.get("transfer_type", "").strip(),
                values.get("transfer_fee", "").strip(),
                int_text(values.get("team_in_id", "")),
                values.get("team_in_name", "").strip(),
                int_text(values.get("team_out_id", "")),
                values.get("team_out_name", "").strip(),
                float_text(values.get("transfer_date_dt", "")),
                json.dumps(raw_payload, ensure_ascii=False, separators=(",", ":")),
            ]
        )
        if limit and len(rows) >= limit:
            break
    return rows


def csv_payload(rows: list[list[str]]) -> str:
    output = io.StringIO()
    writer = csv.writer(output, delimiter="\t", lineterminator="\n")
    for row in rows:
        writer.writerow([value if value != "" else r"\N" for value in row])
    return output.getvalue()


def build_sql(rows: list[list[str]]) -> str:
    columns = ", ".join(STAGING_COLUMNS)
    staging_defs = ",\n    ".join(f"{column} TEXT" for column in STAGING_COLUMNS)
    return f"""BEGIN;

CREATE TEMP TABLE transfer_history_import (
    {staging_defs}
) ON COMMIT DROP;

\\copy transfer_history_import ({columns}) FROM STDIN WITH (FORMAT csv, DELIMITER E'\\t', NULL '\\N')
{csv_payload(rows)}\\.

INSERT INTO player_transfer_history (
    source,
    source_player_id,
    linked_player_id,
    normalized_player_name,
    player_name,
    league_id,
    league_name,
    team_id_context,
    team_name_context,
    transfer_date,
    transfer_type,
    transfer_fee,
    team_in_id,
    team_in_name,
    team_out_id,
    team_out_name,
    transfer_date_serial,
    raw_payload
)
SELECT
    source,
    NULLIF(source_player_id, '')::INT,
    NULL,
    normalized_player_name,
    player_name,
    NULLIF(league_id, '')::INT,
    league_name,
    NULLIF(team_id_context, '')::INT,
    team_name_context,
    NULLIF(transfer_date, '')::DATE,
    transfer_type,
    transfer_fee,
    NULLIF(team_in_id, '')::INT,
    team_in_name,
    NULLIF(team_out_id, '')::INT,
    team_out_name,
    NULLIF(transfer_date_serial, '')::DOUBLE PRECISION,
    raw_payload::JSONB
FROM transfer_history_import
ON CONFLICT DO NOTHING;

UPDATE player_transfer_history
SET linked_player_id = NULL
WHERE source = 'transferts.xlsx';

UPDATE player_transfer_history history
SET linked_player_id = players.id
FROM players
WHERE history.linked_player_id IS NULL
  AND history.source_player_id = players.id
  AND history.normalized_player_name = TRIM(
      REGEXP_REPLACE(
          REGEXP_REPLACE(LOWER(players.name), '[^a-z0-9]+', ' ', 'g'),
          '\\s+',
          ' ',
          'g'
      )
  );

WITH player_club_names AS (
    SELECT DISTINCT
        players.id AS player_id,
        TRIM(
            REGEXP_REPLACE(
                REGEXP_REPLACE(LOWER(players.name), '[^a-z0-9]+', ' ', 'g'),
                '\\s+',
                ' ',
                'g'
            )
        ) AS player_norm,
        TRIM(
            REGEXP_REPLACE(
                REGEXP_REPLACE(LOWER(clubs.name), '[^a-z0-9]+', ' ', 'g'),
                '\\s+',
                ' ',
                'g'
            )
        ) AS club_norm
    FROM players
    JOIN player_seasons ON player_seasons.player_id = players.id
    JOIN clubs ON clubs.id = player_seasons.club_id
),
transfer_candidates AS (
    SELECT
        history.id AS history_id,
        MIN(player_club_names.player_id) AS player_id,
        COUNT(DISTINCT player_club_names.player_id) AS player_count
    FROM player_transfer_history history
    JOIN player_club_names
      ON player_club_names.player_norm = history.normalized_player_name
     AND player_club_names.club_norm <> ''
     AND player_club_names.club_norm IN (
        TRIM(REGEXP_REPLACE(REGEXP_REPLACE(LOWER(COALESCE(history.team_name_context, '')), '[^a-z0-9]+', ' ', 'g'), '\\s+', ' ', 'g')),
        TRIM(REGEXP_REPLACE(REGEXP_REPLACE(LOWER(COALESCE(history.team_in_name, '')), '[^a-z0-9]+', ' ', 'g'), '\\s+', ' ', 'g')),
        TRIM(REGEXP_REPLACE(REGEXP_REPLACE(LOWER(COALESCE(history.team_out_name, '')), '[^a-z0-9]+', ' ', 'g'), '\\s+', ' ', 'g'))
     )
    WHERE history.linked_player_id IS NULL
    GROUP BY history.id
    HAVING COUNT(DISTINCT player_club_names.player_id) = 1
)
UPDATE player_transfer_history history
SET linked_player_id = transfer_candidates.player_id
FROM transfer_candidates
WHERE history.id = transfer_candidates.history_id;

COMMIT;
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xlsx", default=".codex_tmp/transferts.xlsx", help="Source transfer XLSX path.")
    parser.add_argument("--out", default="/tmp/nextlegend_transfer_history_import.sql", help="Output SQL path.")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for smoke tests.")
    args = parser.parse_args()

    xlsx_path = Path(args.xlsx).resolve()
    out_path = Path(args.out).resolve()
    rows = build_rows(xlsx_path, args.limit)
    out_path.write_text(build_sql(rows), encoding="utf-8")
    print(f"Wrote {len(rows)} transfer rows to {out_path}")


if __name__ == "__main__":
    main()
