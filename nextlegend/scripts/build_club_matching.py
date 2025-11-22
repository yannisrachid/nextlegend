#!/usr/bin/env python3
"""Generate a Wyscout ↔ Opta club mapping with fuzzy/TF-IDF suggestions."""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np
import pandas as pd
from rapidfuzz import process
from rapidfuzz.fuzz import WRatio
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

if str(Path(__file__).resolve().parents[1].parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent))

from nextlegend.s3_utils import read_csv_from_s3  # type: ignore

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"

DEFAULT_OPTA_CSV = DATA_DIR / "opta_power_rankings.csv"
DEFAULT_OUT_MATCHING = DATA_DIR / "opta_wyscout_club_matching.csv"
DEFAULT_OUT_REVIEW = DATA_DIR / "opta_wyscout_club_review.csv"
DEFAULT_OUT_RANKINGS = DATA_DIR / "clubs_rankings.csv"
DEFAULT_WYSCOUT_S3 = "data/wyscout_players_final.csv"


logger = logging.getLogger("build_club_matching")


def normalise_label(value: str) -> str:
    text = unicodedata.normalize("NFKD", value)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    cleaned = [
        ch if ch.isalnum() else " "
        for ch in text
    ]
    return " ".join("".join(cleaned).split())


def load_distinct_wyscout_teams(source: str) -> list[str]:
    if source.startswith("s3://") or not Path(source).exists():
        logger.info("Loading Wyscout teams from S3: %s", source)
        df = read_csv_from_s3(source, usecols=["team"])
        teams = df["team"].dropna().astype(str).unique().tolist()
        logger.info("Loaded %d teams from S3", len(teams))
        return sorted(teams)
    unique: set[str] = set()
    logger.info("Loading Wyscout teams from local file: %s", source)
    for idx, chunk in enumerate(pd.read_csv(source, usecols=["team"], chunksize=20000), start=1):
        unique.update(chunk["team"].dropna().astype(str))
        if idx % 20 == 0:
            logger.info("Processed %d chunks (%d unique teams so far)", idx, len(unique))
    logger.info("Loaded %d teams from local file", len(unique))
    return sorted(unique)


def load_opta_teams(path: Path) -> list[str]:
    logger.info("Loading Opta teams from %s", path)
    df = pd.read_csv(path)
    if "team" not in df.columns:
        raise KeyError("Expected column `team` in opta_power_rankings.csv")
    teams = sorted(set(df["team"].dropna().astype(str)))
    logger.info("Loaded %d Opta teams", len(teams))
    return teams


@dataclass
class MatchResult:
    wyscout_team: str
    opta_team: str | None
    score: float
    method: str
    status: str  # AUTO/REVIEW


def build_tfidf_matrices(
    w_norms: Sequence[str],
    o_norms: Sequence[str],
) -> Tuple:
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
    vectorizer.fit(list(w_norms) + list(o_norms))
    w_matrix = vectorizer.transform(w_norms)
    o_matrix = vectorizer.transform(o_norms)
    return w_matrix, o_matrix


def fuzzy_candidates(query: str, choices: Sequence[str], limit: int = 5) -> list[tuple[str, float]]:
    results = process.extract(query, choices, scorer=WRatio, limit=limit)
    return [(candidate, float(score)) for candidate, score, _ in results]


def combine_scores(
    w_names: Sequence[str],
    o_names: Sequence[str],
    threshold: float,
    top_review: int = 3,
) -> tuple[list[MatchResult], list[dict[str, object]]]:
    w_norms = [normalise_label(name) for name in w_names]
    o_norms = [normalise_label(name) for name in o_names]
    logger.info("Building TF-IDF matrices (%d Wyscout teams, %d Opta teams)", len(w_names), len(o_names))
    w_matrix, o_matrix = build_tfidf_matrices(w_norms, o_norms)
    similarity = cosine_similarity(w_matrix, o_matrix, dense_output=False)

    matches: list[MatchResult] = []
    review_rows: list[dict[str, object]] = []
    auto_counter = 0

    for idx, w in enumerate(w_names, start=1):
        row = similarity.getrow(idx - 1).toarray().ravel()
        best_candidate = None
        best_score_pct = 0.0
        if row.size > 0:
            best_idx = int(row.argmax())
            max_value = float(row[best_idx])
            if max_value > 0:
                best_candidate = o_names[best_idx]
                best_score_pct = max_value * 100.0

        if best_candidate and best_score_pct >= threshold:
            matches.append(MatchResult(w, best_candidate, round(best_score_pct, 2), "tfidf", "AUTO"))
            auto_counter += 1
        else:
            matches.append(MatchResult(w, best_candidate, round(best_score_pct, 2), "tfidf", "AUTO" if best_candidate else "REVIEW"))

        if idx % 100 == 0:
            logger.info("Processed %d teams (%d auto-matched)", idx, auto_counter)

    logger.info(
        "Matching complete: %d auto, %d review",
        sum(1 for m in matches if m.status == "AUTO"),
        sum(1 for m in matches if m.status != "AUTO"),
    )
    return matches, review_rows


def write_matches(path: Path, rows: Iterable[MatchResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["wyscout_team", "opta_team", "score", "method", "status"])
        for row in rows:
            writer.writerow([row.wyscout_team, row.opta_team or "", f"{row.score:.2f}", row.method, row.status])


def write_review(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def write_club_rankings(opta_path: Path, matching_path: Path, output_path: Path) -> None:
    matching_df = pd.read_csv(matching_path)
    opta_df = pd.read_csv(opta_path)
    merged = matching_df.merge(
        opta_df,
        left_on="opta_team",
        right_on="team",
        how="left",
    )
    merged.rename(
        columns={
            "team": "team_opta",
            "wyscout_team": "team_wyscout",
        },
        inplace=True,
    )
    if "rating" in merged.columns:
        merged.sort_values(by="rating", ascending=False, inplace=True)
    merged.to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a Wyscout ↔ Opta club mapping dictionary.")
    parser.add_argument("--wyscout", default=str(DEFAULT_WYSCOUT_S3), help="S3 key or local path to Wyscout CSV")
    parser.add_argument("--opta", default=str(DEFAULT_OPTA_CSV))
    parser.add_argument("--out-matching", default=str(DEFAULT_OUT_MATCHING))
    parser.add_argument("--out-review", default=str(DEFAULT_OUT_REVIEW))
    parser.add_argument("--out-rankings", default=str(DEFAULT_OUT_RANKINGS))
    parser.add_argument("--threshold", type=float, default=70.0, help="Minimum TF-IDF score for auto match (0-100)")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    wyscout_source = args.wyscout
    opta_path = Path(args.opta).expanduser()
    if not opta_path.exists():
        raise FileNotFoundError(f"Opta CSV not found: {opta_path}")

    wyscout_teams = load_distinct_wyscout_teams(wyscout_source)
    opta_teams = load_opta_teams(opta_path)

    matches, review_rows = combine_scores(wyscout_teams, opta_teams, args.threshold)
    write_matches(Path(args.out_matching), matches)
    write_review(Path(args.out_review), review_rows)
    write_club_rankings(Path(args.opta), Path(args.out_matching), Path(args.out_rankings))

    total = len(matches)
    auto = sum(1 for row in matches if row.status == "AUTO")
    review = total - auto
    print(f"Auto matches: {auto}/{total}")
    if review_rows:
        print(f"Manual review required for {review} clubs → {args.out_review}")
    else:
        print("No manual review required.")


if __name__ == "__main__":
    main()
