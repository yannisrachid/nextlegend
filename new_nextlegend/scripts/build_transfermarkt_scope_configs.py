#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
HELPERS = ROOT / "helpers" / "csv"
sys.path.insert(0, str(ROOT / "jobs" / "pipeline"))

from pipeline.transfermarkt_matching import prepare_transfermarkt_profiles  # noqa: E402


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def clean_id(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)


def build_competitions_scope(club_map: pd.DataFrame, league_meta: pd.DataFrame, competition_mapping: pd.DataFrame) -> pd.DataFrame:
    if club_map.empty:
        return pd.DataFrame()
    clubs = club_map.copy()
    clubs["tm_club_id"] = clean_id(clubs["tm_club_id"]) if "tm_club_id" in clubs.columns else pd.NA
    grouped = (
        clubs.groupby(["competition_name", "competition_country"], dropna=False)
        .agg(
            wyscout_clubs=("team", "nunique"),
            mapped_tm_clubs=("tm_club_id", lambda value: int(value.notna().sum())),
            mapping_methods=("method", lambda value: "|".join(sorted(set(map(str, value.dropna()))))),
        )
        .reset_index()
    )
    grouped["mapped_tm_club_rate"] = (grouped["mapped_tm_clubs"] / grouped["wyscout_clubs"]).round(4)

    if not league_meta.empty and "competition" in league_meta.columns:
        grouped = grouped.merge(
            league_meta.rename(columns={"competition": "competition_name"}),
            on="competition_name",
            how="left",
        )

    if not competition_mapping.empty:
        grouped = grouped.merge(competition_mapping, on="competition_name", how="left")

    for col in ["tm_competition_id", "tm_competition_name", "tm_country", "tm_season_id", "scrape_enabled", "mapping_status"]:
        if col not in grouped.columns:
            grouped[col] = pd.NA

    grouped["scrape_enabled"] = grouped["scrape_enabled"].fillna(False)
    grouped["mapping_status"] = grouped["mapping_status"].fillna("needs_transfermarkt_competition_id")

    cols = [
        "scrape_enabled",
        "competition_name",
        "competition_country",
        "tm_competition_id",
        "tm_competition_name",
        "tm_country",
        "tm_season_id",
        "wyscout_clubs",
        "mapped_tm_clubs",
        "mapped_tm_club_rate",
        "difficulty",
        "intensity",
        "mapping_status",
        "mapping_methods",
    ]
    return grouped[[col for col in cols if col in grouped.columns]].sort_values(
        ["scrape_enabled", "mapped_tm_club_rate", "competition_name"],
        ascending=[False, False, True],
        kind="stable",
    )


def build_clubs_scope(club_map: pd.DataFrame, tm_profiles: pd.DataFrame) -> pd.DataFrame:
    if club_map.empty:
        return pd.DataFrame()
    clubs = club_map.copy()
    clubs["tm_club_id"] = clean_id(clubs["tm_club_id"]) if "tm_club_id" in clubs.columns else pd.NA

    if not tm_profiles.empty:
        tm = tm_profiles.copy()
        tm.columns = [str(col).strip().lower() for col in tm.columns]
        if {"club_id", "player_id"}.issubset(tm.columns):
            tm["tm_club_id"] = clean_id(tm["club_id"])
            player_counts = (
                tm.groupby("tm_club_id", dropna=False)
                .agg(tm_scraped_players=("player_id", "nunique"))
                .reset_index()
            )
            clubs = clubs.merge(player_counts, on="tm_club_id", how="left")

    clubs["tm_scraped_players"] = clubs.get("tm_scraped_players", 0).fillna(0).astype(int)
    clubs["scrape_status"] = clubs.apply(
        lambda row: "scraped" if pd.notna(row.get("tm_club_id")) and row.get("tm_scraped_players", 0) > 0 else row.get("method", "unknown"),
        axis=1,
    )
    cols = [
        "competition_name",
        "competition_country",
        "team",
        "tm_club_id",
        "tm_club_name",
        "method",
        "score",
        "tm_scraped_players",
        "scrape_status",
    ]
    return clubs[[col for col in cols if col in clubs.columns]].sort_values(
        ["competition_name", "team"],
        kind="stable",
    )


def build_players_scope(tm_profiles: pd.DataFrame, player_map: pd.DataFrame) -> pd.DataFrame:
    if tm_profiles.empty:
        return pd.DataFrame()
    tm = prepare_transfermarkt_profiles(tm_profiles)

    mapped_ids: set[str] = set()
    if not player_map.empty and "tm_player_id" in player_map.columns:
        mapped_ids = set(clean_id(player_map["tm_player_id"]).dropna().astype(str))

    tm_birth_year = pd.to_numeric(tm.get("tm_birth_year"), errors="coerce").astype("Int64")
    output = pd.DataFrame(
        {
            "tm_player_id": tm["tm_player_id"],
            "tm_player_name": tm.get("tm_player_name"),
            "tm_club_id": tm.get("tm_club_id"),
            "tm_club_name": tm.get("tm_club_name"),
            "tm_position_main": tm.get("tm_position_main"),
            "tm_birth_date": tm.get("tm_birth_date"),
            "tm_birth_year": tm_birth_year,
            "tm_citizenship": tm.get("tm_citizenship"),
            "tm_market_value": tm.get("tm_market_value"),
            "tm_market_value_eur": tm.get("tm_market_value_eur"),
            "tm_profile_url": tm.get("tm_profile_url"),
            "already_in_player_mapping": tm["tm_player_id"].astype(str).isin(mapped_ids),
        }
    )
    return output.drop_duplicates(subset=["tm_player_id"], keep="last").sort_values(
        ["tm_club_name", "tm_player_name"],
        kind="stable",
    )


def build_competition_mapping_template(club_map: pd.DataFrame, existing: pd.DataFrame) -> pd.DataFrame:
    if not existing.empty:
        return existing
    if club_map.empty:
        return pd.DataFrame(
            columns=[
                "competition_name",
                "tm_competition_id",
                "tm_competition_name",
                "tm_country",
                "tm_season_id",
                "scrape_enabled",
                "mapping_status",
                "notes",
            ]
        )
    competitions = (
        club_map[["competition_name", "competition_country"]]
        .drop_duplicates()
        .sort_values("competition_name", kind="stable")
    )
    competitions["tm_competition_id"] = ""
    competitions["tm_competition_name"] = ""
    competitions["tm_country"] = competitions["competition_country"]
    competitions["tm_season_id"] = ""
    competitions["scrape_enabled"] = False
    competitions["mapping_status"] = "needs_transfermarkt_competition_id"
    competitions["notes"] = ""
    return competitions.drop(columns=["competition_country"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build auditable Transfermarkt scope config CSVs.")
    parser.add_argument("--helpers-dir", default=str(HELPERS))
    args = parser.parse_args()

    helpers = Path(args.helpers_dir)
    helpers.mkdir(parents=True, exist_ok=True)

    club_map = read_csv(helpers / "club_matching_reference.csv")
    player_map = read_csv(helpers / "player_matching_reference.csv")
    tm_profiles = read_csv(helpers / "transfermarkt_profiles.csv")
    league_meta = read_csv(helpers / "league_translation_meta.csv")
    competition_mapping_path = helpers / "wyscout_transfermarkt_competition_mapping.csv"
    existing_competition_mapping = read_csv(competition_mapping_path)

    competition_mapping = build_competition_mapping_template(club_map, existing_competition_mapping)
    competition_mapping.to_csv(competition_mapping_path, index=False)

    build_competitions_scope(club_map, league_meta, competition_mapping).to_csv(
        helpers / "transfermarkt_competitions_scope.csv",
        index=False,
    )
    build_clubs_scope(club_map, tm_profiles).to_csv(
        helpers / "transfermarkt_clubs_scope.csv",
        index=False,
    )
    build_players_scope(tm_profiles, player_map).to_csv(
        helpers / "transfermarkt_players_scope.csv",
        index=False,
    )

    print(f"[TM-CONFIG] wrote scope configs to {helpers}")


if __name__ == "__main__":
    main()
