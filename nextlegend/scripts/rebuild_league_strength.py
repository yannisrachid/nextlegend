#!/usr/bin/env python3
"""Rebuild league strength coefficients from club ratings."""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
REPO_ROOT = PROJECT_ROOT.parent
for candidate in (REPO_ROOT, PROJECT_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from nextlegend.s3_utils import write_csv_to_s3  # type: ignore

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"

DEFAULT_CLUBS = DATA_DIR / "clubs_rankings.csv"
DEFAULT_CLEANED = DATA_DIR / "wyscout_players_cleaned.csv"
DEFAULT_META = DATA_DIR / "league_translation_meta.csv"
DEFAULT_MATRIX = DATA_DIR / "league_translation_matrix.csv"


logger = logging.getLogger("rebuild_league_strength")


def load_club_ratings(path: Path) -> Dict[str, float]:
    df = pd.read_csv(path)
    required_cols = {"team_wyscout", "rating"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"{path} must contain columns {required_cols}")
    df = df.dropna(subset=["team_wyscout", "rating"])
    # Some clubs peuvent apparaître plusieurs fois (réserves); garder le max.
    rating_map = df.groupby("team_wyscout")["rating"].max().to_dict()
    logger.info("Loaded %d club ratings from %s", len(rating_map), path)
    return rating_map


def canonical_team_name(name: str) -> str:
    tokens = name.split()
    filtered = []
    blocklist = {"B", "II", "III", "IV", "V", "VI", "RESERVES", "RESERVE", "ACADEMY", "WOMEN"}
    for token in tokens:
        upper = token.upper()
        if upper in blocklist:
            continue
        if upper.startswith("U") and upper[1:].isdigit():
            continue
        filtered.append(token)
    return " ".join(filtered) if filtered else name


def load_competition_membership(path: Path) -> Dict[str, set[str]]:
    membership: Dict[str, set[str]] = defaultdict(set)
    required_cols = {"team_in_selected_period", "competition_name"}
    chunk_iter = pd.read_csv(path, usecols=list(required_cols), chunksize=50_000)
    total_rows = 0
    for chunk in chunk_iter:
        chunk = chunk.dropna(subset=["team_in_selected_period", "competition_name"])
        total_rows += len(chunk)
        for team, competition in zip(chunk["team_in_selected_period"], chunk["competition_name"]):
            normalized_team = canonical_team_name(str(team))
            membership[str(competition)].add(normalized_team)
    logger.info(
        "Processed %d rows from %s and found %d competitions",
        total_rows,
        path,
        len(membership),
    )
    return membership


def compute_competition_strength(
    membership: Dict[str, set[str]],
    club_ratings: Dict[str, float],
    min_clubs: int = 2,
) -> pd.DataFrame:
    rows = []
    missing = 0
    for competition, teams in membership.items():
        deduped = sorted(set(teams))
        if competition == "England. Premier League":
            logger.info("Premier League unique clubs (%d): %s", len(deduped), deduped)
        if len(deduped) != len(teams):
            logger.debug("Removed duplicates for %s: %d -> %d clubs", competition, len(teams), len(deduped))
        ratings = [club_ratings[team] for team in deduped if team in club_ratings]
        if len(ratings) < min_clubs:
            missing += 1
            continue
        mean_rating = sum(ratings) / len(ratings)
        strength = mean_rating / 10.0  # ramener dans l'échelle ~5-10
        rows.append(
            {
                "competition": competition,
                "difficulty": round(strength, 3),
                "intensity": round(strength, 3),
                "clubs_count": len(ratings),
            }
        )
    logger.info(
        "Computed strength for %d competitions (skipped %d lacking ratings)",
        len(rows),
        missing,
    )
    meta_df = pd.DataFrame(rows).sort_values(by="difficulty", ascending=False)
    return meta_df


def build_translation_matrix(meta_df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, src_row in meta_df.iterrows():
        for _, tgt_row in meta_df.iterrows():
            src_comp = src_row["competition"]
            tgt_comp = tgt_row["competition"]
            src_diff = src_row["difficulty"]
            tgt_diff = tgt_row["difficulty"]
            src_int = src_row["intensity"]
            tgt_int = tgt_row["intensity"]
            if tgt_diff == 0 or tgt_int == 0:
                continue
            difficulty_coeff = round(src_diff / tgt_diff, 5)
            intensity_coeff = round(src_int / tgt_int, 5)
            overall = round(difficulty_coeff, 5)
            records.append(
                {
                    "source_competition": src_comp,
                    "target_competition": tgt_comp,
                    "difficulty_coeff": difficulty_coeff,
                    "intensity_coeff": intensity_coeff,
                    "overall_coeff": overall,
                }
            )
    matrix_df = pd.DataFrame(records)
    return matrix_df


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recompute league strength coefficients from club ratings.")
    parser.add_argument("--clubs", default=str(DEFAULT_CLUBS), help="Path to clubs_rankings.csv")
    parser.add_argument("--cleaned", default=str(DEFAULT_CLEANED), help="Path to wyscout_players_cleaned.csv")
    parser.add_argument("--out-meta", default=str(DEFAULT_META), help="Output path for league_translation_meta.csv (local)")
    parser.add_argument("--out-matrix", default=str(DEFAULT_MATRIX), help="Output path for league_translation_matrix.csv (local)")
    parser.add_argument("--out-meta-s3", default="data/league_translation_meta.csv", help="S3 key for meta file")
    parser.add_argument("--out-matrix-s3", default="data/league_translation_matrix.csv", help="S3 key for matrix file")
    parser.add_argument("--min-clubs", type=int, default=2, help="Minimum number of rated clubs per competition")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = parse_args()
    args = parser.parse_args()

    clubs_path = Path(args.clubs).expanduser()
    cleaned_path = Path(args.cleaned).expanduser()

    for path in (clubs_path, cleaned_path):
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

    club_ratings = load_club_ratings(clubs_path)
    membership = load_competition_membership(cleaned_path)
    meta_df = compute_competition_strength(membership, club_ratings, args.min_clubs)
    meta_path = Path(args.out_meta)
    matrix_path = Path(args.out_matrix)
    meta_df.to_csv(meta_path, index=False)
    matrix_df = build_translation_matrix(meta_df)
    matrix_df.to_csv(matrix_path, index=False)
    write_csv_to_s3(meta_df, args.out_meta_s3, index=False)
    write_csv_to_s3(matrix_df, args.out_matrix_s3, index=False)
    logger.info(
        "Wrote local files %s (%d rows) & %s (%d rows) and uploaded to S3 (%s, %s)",
        meta_path,
        len(meta_df),
        matrix_path,
        len(matrix_df),
        args.out_meta_s3,
        args.out_matrix_s3,
    )


if __name__ == "__main__":
    main()
