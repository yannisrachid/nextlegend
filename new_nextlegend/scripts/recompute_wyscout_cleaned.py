#!/usr/bin/env python3
"""Recompute current-season scoring and matching from a cleaned Wyscout export."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "jobs" / "pipeline"))

from pipeline import processing  # noqa: E402


def normalize_transfermarkt_numeric_fields(enriched: pd.DataFrame) -> pd.DataFrame:
    result = enriched.copy()
    if "tm_height" in result:
        raw = result["tm_height"].astype("string").str.strip()
        meter_values = raw.where(raw.str.fullmatch(r"\d{1,2}[,.]\d{1,2}m"))
        meters_cm = pd.to_numeric(meter_values.str.removesuffix("m").str.replace(",", ".", regex=False), errors="coerce") * 100
        numeric = pd.to_numeric(raw, errors="coerce")
        numeric_cm = numeric.where(numeric >= 100, numeric * 100)
        height_cm = numeric_cm.combine_first(meters_cm)
        invalid = raw.notna() & raw.ne("-") & ~height_cm.between(100, 250)
        if invalid.any():
            raise ValueError(f"Unrecognized Transfermarkt heights: {raw[invalid].drop_duplicates().head().tolist()}")
        result["tm_height"] = height_cm
    if "tm_market_value" in result:
        raw = result["tm_market_value"].astype("string").str.strip()
        money = raw.str.replace(r"^[€£$]\s*", "", regex=True).str.extract(r"^([\d.,]+)\s*([kmbnKMBN]?)$")
        amount = pd.to_numeric(money[0].str.replace(",", ".", regex=False), errors="coerce")
        factor = money[1].str.lower().map({"": 1, "k": 1_000, "m": 1_000_000, "b": 1_000_000_000, "n": 1_000_000_000})
        value = pd.to_numeric(raw, errors="coerce").combine_first(amount * factor)
        invalid = raw.notna() & raw.ne("-") & value.isna()
        if invalid.any():
            raise ValueError(f"Unrecognized Transfermarkt market values: {raw[invalid].drop_duplicates().head().tolist()}")
        result["tm_market_value"] = value
    return result


def local_database_url(env_file: Path) -> str:
    values: dict[str, str] = {}
    for line in env_file.read_text(encoding="utf-8").splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"\'')
    required = {"POSTGRES_DB", "POSTGRES_USER", "POSTGRES_PASSWORD"}
    if not required.issubset(values):
        raise ValueError(f"Missing local PostgreSQL settings: {sorted(required.difference(values))}")
    url = URL.create(
        "postgresql+psycopg",
        username=values["POSTGRES_USER"],
        password=values["POSTGRES_PASSWORD"],
        host="127.0.0.1",
        port=int(os.getenv("POSTGRES_PORT", values.get("POSTGRES_PORT", "5432"))),
        database=values["POSTGRES_DB"],
    )
    return url.render_as_string(hide_password=False)


def preserve_known_transfermarkt(
    enriched: pd.DataFrame,
    input_path: Path,
    fallback_path: Path | None,
) -> pd.DataFrame:
    keys = ["player_id", "competition_name", "calendar", "team_in_selected_period"]
    tm_columns = [column for column in enriched.columns if column.startswith("tm_")]
    if "tm_player_id" not in tm_columns:
        return enriched
    source_columns = set(keys + tm_columns + ["profile_url"])
    sources = []
    for path in (input_path, fallback_path):
        if path is None:
            continue
        with path.open(newline="", encoding="utf-8-sig") as handle:
            available = set(next(csv.reader(handle)))
        usecols = sorted(source_columns.intersection(available))
        source = pd.read_csv(
            path,
            usecols=usecols,
            dtype={"player_id": "string", "tm_player_id": "string"},
            low_memory=False,
        )
        source["tm_player_id"] = source["tm_player_id"].str.replace(r"\.0$", "", regex=True)
        sources.append(source)
    if not sources:
        return enriched

    result = enriched.copy()
    result["player_id"] = result["player_id"].astype("string")
    result["tm_player_id"] = result["tm_player_id"].astype("string").str.replace(r"\.0$", "", regex=True)
    original_matches = int(result["tm_player_id"].notna().sum())
    original_agents = int(result["tm_agent_name"].notna().sum()) if "tm_agent_name" in result else 0

    known = pd.concat(sources, ignore_index=True)
    known_by_key = known.drop_duplicates(subset=keys, keep="first").set_index(keys)
    result = result.set_index(keys)
    prior_id = known_by_key["tm_player_id"].reindex(result.index)
    conflicts = int((result["tm_player_id"].notna() & prior_id.notna() & (result["tm_player_id"] != prior_id)).sum())
    result["tm_player_id"] = result["tm_player_id"].combine_first(prior_id)

    profiles = known.dropna(subset=["tm_player_id"]).groupby("tm_player_id", sort=False).first()
    for column in tm_columns + ["profile_url"]:
        if column == "tm_player_id" or column not in profiles.columns:
            continue
        if column not in result.columns:
            result[column] = pd.NA
        result[column] = result[column].combine_first(result["tm_player_id"].map(profiles[column]))

    result = result.reset_index()
    print(
        f"[RECOMPUTE] TM preserved matched={original_matches}->{result['tm_player_id'].notna().sum()} "
        f"agents={original_agents}->{result['tm_agent_name'].notna().sum()} "
        f"conflicting_known_ids={conflicts}",
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--similarity-output", type=Path, required=True)
    parser.add_argument("--profile-fallback", type=Path)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    with args.input.open(newline="", encoding="utf-8-sig") as handle:
        columns = next(csv.reader(handle))
    if "penalty_conversion_percent" not in columns or "scoring_model_version" not in columns:
        raise ValueError("Input is not a supported scored Wyscout export")
    raw_columns = columns[: columns.index("penalty_conversion_percent") + 1]
    required = {"player_id", "competition_name", "calendar", "team_in_selected_period", "minutes_played"}
    if not required.issubset(raw_columns):
        raise ValueError(f"Missing source columns: {sorted(required.difference(raw_columns))}")

    os.environ["DATABASE_URL"] = local_database_url(args.env_file)
    os.environ["TRANSFERMARKT_SOURCES_DIR"] = str(ROOT / "helpers" / "csv")
    os.environ["MERCATO_LEAGUE_LEVELS_PATH"] = str(ROOT / "config" / "mercato_league_levels.json")
    os.environ["SIM_TOPK"] = "10"
    with create_engine(os.environ["DATABASE_URL"]).connect() as connection:
        latest_opta = connection.execute(
            text("SELECT source_last_updated_date, matched_clubs FROM opta_power_ranking_runs WHERE gender = 'mens' ORDER BY source_last_updated_date DESC NULLS LAST LIMIT 1")
        ).one_or_none()
    if not latest_opta or not latest_opta.source_last_updated_date or latest_opta.matched_clubs < 1000:
        raise ValueError("Local Opta rankings are missing or have insufficient club mappings")
    print(f"[RECOMPUTE] opta_date={latest_opta.source_last_updated_date} matched_clubs={latest_opta.matched_clubs}", flush=True)
    frame = pd.read_csv(args.input, usecols=raw_columns, low_memory=False)
    if args.limit:
        frame = frame.head(args.limit)
    elif len(frame) < 30_000:
        raise ValueError(f"Only {len(frame)} Wyscout rows; expected at least 30,000")
    print(f"[RECOMPUTE] input_rows={len(frame)} raw_columns={len(raw_columns)}", flush=True)

    artifacts = processing.build_artifacts(frame)
    enriched = normalize_transfermarkt_numeric_fields(
        preserve_known_transfermarkt(artifacts["enriched"], args.input, args.profile_fallback)
    )
    similarity = artifacts["player_similarity"]
    if enriched.empty or not {"2026/2027", "2026"}.issubset(set(enriched["calendar"].dropna())):
        raise ValueError("Recomputed export is empty or missing an active calendar")
    if enriched.columns.duplicated().any():
        raise ValueError("Recomputed export contains duplicate column names")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.similarity_output.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(args.output, index=False)
    similarity.to_csv(args.similarity_output, index=False)
    print(
        f"[RECOMPUTE] output_rows={len(enriched)} similarity_rows={len(similarity)} "
        f"tm_matched={enriched.get('tm_player_id', pd.Series(dtype='object')).notna().sum()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
