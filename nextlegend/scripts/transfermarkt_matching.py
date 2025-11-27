#!/usr/bin/env python3
"""Match Transfermarkt metadata with Wyscout data and enrich the cleaned dataset.

Steps implemented:
1. Parse Transfermarkt profile descriptions to extract age/birth date.
2. Apply the curated club mapping from ``nextlegend/data/club_mapping_dict.py`` (no fuzzy matching).
3. Within each mapped club, match players using names + fuzzy fallback + age/birth checks.
4. Merge Transfermarkt columns (prefixed with ``tm_``) into the Wyscout dataset.
5. Emit reference CSVs so manual adjustments remain possible (without overwriting the curated mapping).
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import re
import sys
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz, process

sys.path.append(str(Path(__file__).resolve().parents[1]))
from s3_utils import (  # noqa: E402
    S3ConfigurationError,
    read_csv_from_s3,
    write_csv_to_s3,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT_DIR.parent

DEFAULT_WYSCOUT_KEY = "data/wyscout_players_cleaned.csv"
DEFAULT_WYSCOUT_LOCAL = PROJECT_ROOT / "data" / "wyscout_players_final.csv"
DEFAULT_TM_PATH = ROOT_DIR / "data" / "transfermarkt_profiles.csv"
DEFAULT_OUTPUT_LOCAL: Optional[Path] = None
DEFAULT_CLUB_MAPPING = ROOT_DIR / "data" / "club_matching_reference.csv"
DEFAULT_PLAYER_MAPPING = ROOT_DIR / "data" / "player_matching_reference.csv"
DEFAULT_TM_CLUBS = ROOT_DIR / "data" / "tm_clubs_reference.csv"
DEFAULT_CLUB_MAPPING_PY = ROOT_DIR / "data" / "club_mapping_dict.py"

BREST_LOG_PATH = PROJECT_ROOT / "transfermarkt_brest_debug.log"


logger = logging.getLogger("transfermarkt_matching")


def normalise_name(value: str | float | None) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_tm_description(text: str | float | None) -> tuple[Optional[int], Optional[str]]:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return None, None
    value = str(text)
    age = None
    birth_date = None

    age_match = re.search(r",\s*(\d{1,2})\s*,\s*from", value)
    if age_match:
        age = int(age_match.group(1))

    date_match = re.search(r"\*\s*(\d{2}/\d{2}/\d{4})", value)
    if date_match:
        day, month, year = date_match.group(1).split("/")
        birth_date = f"{year}-{month}-{day}"
    return age, birth_date


def parse_wyscout_age(text: str | float | None) -> Optional[int]:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return None
    value = str(text)
    match = re.search(r"\((\d{1,2})\)", value)
    if match:
        return int(match.group(1))
    digits = re.findall(r"\d{2}", value)
    if digits:
        return int(digits[-1])
    return None


def parse_birth_date(value: str | float | None) -> Optional[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    for dayfirst in (False, True):
        try:
            parsed = pd.to_datetime(text, errors="raise", dayfirst=dayfirst)
        except (ValueError, TypeError):
            continue
        if pd.isna(parsed):
            continue
        return parsed.strftime("%Y-%m-%d")
    match = re.search(r"(\d{2})/(\d{2})/(\d{4})", text)
    if match:
        day, month, year = match.groups()
        return f"{year}-{month}-{day}"
    return None


def clean_wyscout_player_name(text: str | float | None) -> str:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    value = str(text)
    if ";" in value:
        value = value.split(";")[0]
    return value.strip()


def add_tm_enrichments(tm_df: pd.DataFrame) -> pd.DataFrame:
    ages = []
    birth_dates = []
    for raw in tm_df.get("profile_description", []):
        age, birth = parse_tm_description(raw)
        ages.append(age)
        birth_dates.append(birth)
    df = tm_df.copy()
    df["tm_age"] = ages
    df["tm_birth_date"] = birth_dates
    return df


def competition_country(value: str | float | None) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value)
    return text.split(".")[0].strip()


def is_brest(name: str | float | None) -> bool:
    norm = normalise_name(name)
    return "brest" in norm.split() or norm == "brest" or norm.startswith("brest")


def log_brest(message: str) -> None:
    try:
        with BREST_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(message + "\n")
    except Exception:
        logger.debug("Unable to write Brest debug log.")


def load_manual_club_mapping(path: Path) -> dict[tuple[str, str], tuple[int, str]]:
    """Load the curated club mapping dictionary from a Python file."""
    mapping: dict[tuple[str, str], tuple[int, str]] = {}
    if not path.exists():
        logger.warning("Club mapping dict not found at %s; clubs without mapping will stay unmatched.", path)
        return mapping
    spec = importlib.util.spec_from_file_location("club_mapping_dict", path)
    if spec is None or spec.loader is None:
        logger.warning("Unable to load club mapping module from %s", path)
        return mapping
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    raw = getattr(module, "CLUB_MAPPING", {})
    if not isinstance(raw, dict):
        logger.warning("CLUB_MAPPING in %s is not a dictionary; skipping.", path)
        return mapping
    for key, value in raw.items():
        try:
            team_key, comp_key = key
            tm_id_raw, tm_name_raw = value
        except (ValueError, TypeError):
            continue
        team = str(team_key).strip()
        competition = str(comp_key).strip()
        try:
            tm_id = int(tm_id_raw)
        except (ValueError, TypeError):
            tm_id = 0
        tm_name = "" if tm_name_raw is None else str(tm_name_raw).strip()
        mapping[(team, competition)] = (tm_id, tm_name)
    logger.info("Loaded manual club mapping: %s entries from %s", len(mapping), path)
    return mapping


STOPWORDS = {
    "fc",
    "sc",
    "ac",
    "cf",
    "cd",
    "sd",
    "ud",
    "udinese",
    "club",
    "clubs",
    "de",
    "da",
    "di",
    "do",
    "del",
    "la",
    "el",
    "the",
    "ssc",
    "as",
    "ss",
    "us",
    "fk",
    "nk",
    "pfk",
    "a",
    "b",
    "c",
}

TOKEN_SYNONYMS = {
    "atl": "atletico",
    "ath": "athletic",
    "athl": "athletic",
    "athletic": "athletic",
    "dep": "deportivo",
    "rcd": "deportivo",
    "int": "internacional",
    "intl": "internacional",
    "inter": "internazionale",
    "interm": "internazionale",
    "sport": "sporting",
    "sp": "sporting",
    "vit": "vitoria",
    "benf": "benfica",
    "juv": "juventus",
    "juve": "juventus",
    "saint": "saint",
    "st": "saint",
    "stl": "saint",
    "stet": "saint",
    "real": "real",
    "cf": "",
    "cd": "",
    "sd": "",
    "ac": "",
    "as": "",
    "fc": "",
    "sc": "",
}

SCORING_STEPS: list[tuple[str, Callable[[str, str], int], int]] = [
    ("token_sort_strict", fuzz.token_sort_ratio, 92),
    ("token_set_strict", fuzz.token_set_ratio, 90),
    ("wr_strict", fuzz.WRatio, 88),
    ("partial_relaxed", fuzz.partial_ratio, 82),
]


def load_existing_club_overrides(path: Path | None) -> dict[tuple[str, str], tuple[str, int]]:
    overrides: dict[tuple[str, str], tuple[str, int]] = {}
    if not path or not path.exists():
        return overrides
    try:
        df = pd.read_csv(path)
    except Exception:
        return overrides
    for _, row in df.iterrows():
        team = str(row.get("team_in_selected_period", row.get("team", ""))).strip()
        comp = str(row.get("competition_country", "")).strip()
        name = row.get("tm_club_name")
        club_id = row.get("tm_club_id")
        if not team or pd.isna(name) or pd.isna(club_id):
            continue
        try:
            overrides[(team, comp)] = (str(name), int(float(club_id)))
        except (TypeError, ValueError):
            continue
    return overrides


def tokenize_team_name(name: str | float | None) -> list[str]:
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return []
    norm = normalise_name(name)
    base_tokens = []
    for raw in norm.split():
        token = raw.strip(".")
        if not token or token in STOPWORDS:
            continue
        token = TOKEN_SYNONYMS.get(token, token)
        if not token:
            continue
        base_tokens.append(token)
    tokens: set[str] = set()
    for token in base_tokens:
        tokens.add(token)
        if len(token) >= 4:
            tokens.add(token[:3])
        if len(token) >= 5:
            tokens.add(token[:4])
        if len(token) >= 7:
            tokens.add(token[:6])
    return sorted(tokens)


def prepare_tm_club_records(tm_df: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, set[int]]]:
    clubs = (
        tm_df[["club_id", "club_name"]]
        .dropna(subset=["club_id", "club_name"])
        .drop_duplicates(subset=["club_id"])
        .reset_index(drop=True)
    )
    records: list[dict[str, Any]] = []
    token_index: dict[str, set[int]] = defaultdict(set)
    for idx, row in clubs.iterrows():
        club_id = int(row["club_id"])
        club_name = str(row["club_name"])
        tokens = tokenize_team_name(club_name)
        record = {
            "club_id": club_id,
            "club_name": club_name,
            "norm_name": normalise_name(club_name),
            "tokens": tokens,
        }
        records.append(record)
        for token in tokens:
            token_index[token].add(idx)
    return records, token_index


def evaluate_candidates(
    target: str,
    candidate_indices: set[int],
    records: list[dict[str, Any]],
    scorer: Callable[[str, str], int],
) -> tuple[Optional[int], int, int]:
    if not candidate_indices:
        return None, 0, 0
    best_idx: Optional[int] = None
    best_score = -1
    second_score = -1
    for idx in candidate_indices:
        candidate = records[idx]["norm_name"]
        score = scorer(target, candidate)
        if score > best_score:
            second_score = best_score
            best_score = score
            best_idx = idx
        elif score > second_score:
            second_score = score
    return best_idx, best_score, second_score


def smart_match_club_name(
    name: str,
    records: list[dict[str, Any]],
    token_index: dict[str, set[int]],
    threshold_anchor: int,
) -> tuple[Optional[int], Optional[str], str, int]:
    norm = normalise_name(name)
    if not norm:
        return None, None, "empty_name", 0

    tokens = tokenize_team_name(name)
    candidate_indices: set[int] = set()
    for token in tokens:
        candidate_indices.update(token_index.get(token, set()))
    candidate_scope = "token" if candidate_indices else "global"
    if not candidate_indices:
        candidate_indices = set(range(len(records)))

    shift = max(-15, min(10, threshold_anchor - 90))

    for label, scorer, threshold in SCORING_STEPS:
        base_threshold = max(70, min(99, threshold + shift))
        idx, score, runner = evaluate_candidates(norm, candidate_indices, records, scorer)
        if idx is None:
            continue
        effective_threshold = base_threshold
        if candidate_scope == "token":
            if len(candidate_indices) <= 5:
                effective_threshold -= 7
            elif len(candidate_indices) <= 12:
                effective_threshold -= 4
        if score >= effective_threshold or (score >= base_threshold - 10 and score - runner >= 12):
            record = records[idx]
            return record["club_id"], record["club_name"], label, int(score)
    return None, None, "unmatched", 0


@dataclass
class ClubMatch:
    team: str
    competition_name: str
    competition_country: str
    tm_club_name: Optional[str]
    tm_club_id: Optional[int]
    method: str
    score: int


@dataclass
class PlayerMatch:
    player: str
    player_short: str
    team: str
    competition_name: str
    calendar: str
    age: Optional[int]
    birth_date: Optional[str]
    tm_player_id: Optional[int]
    tm_player_name: Optional[str]
    tm_club_id: Optional[int]
    method: str
    score: int
    age_diff: Optional[int]


def build_club_mapping(
    wyscout_df: pd.DataFrame,
    manual_mapping: dict[tuple[str, str], tuple[int, str]],
) -> tuple[pd.DataFrame, dict[tuple[str, str], int]]:
    """Apply the curated club mapping without fuzzy matching."""
    if "team_in_selected_period" not in wyscout_df.columns:
        wyscout_df = wyscout_df.copy()
        wyscout_df["team_in_selected_period"] = wyscout_df["team"]

    rows: list[ClubMatch] = []
    lookup: dict[tuple[str, str], int] = {}

    clubs = (
        wyscout_df[["team_in_selected_period", "competition_name"]]
        .dropna(subset=["team_in_selected_period"])
        .drop_duplicates()
    )

    for _, row in clubs.iterrows():
        team_main = str(row["team_in_selected_period"]).strip()
        comp = str(row.get("competition_name", "")).strip()
        comp_country = competition_country(comp)

        tm_id: Optional[int] = None
        tm_name: Optional[str] = None
        method = "dict"
        score = 100

        candidate_keys = [
            (team_main, comp),
            (team_main, comp_country),
            (team_main, ""),
        ]

        chosen_key: Optional[tuple[str, str]] = None
        for key in candidate_keys:
            if key not in manual_mapping:
                continue
            tm_id_raw, tm_name_raw = manual_mapping[key]
            tm_id_value = 0
            try:
                tm_id_value = int(tm_id_raw)
            except (TypeError, ValueError):
                tm_id_value = 0
            tm_name_value = "" if tm_name_raw is None else str(tm_name_raw).strip()
            if tm_id_value > 0:
                chosen_key = key
                tm_id = tm_id_value
                tm_name = tm_name_value
                break
            else:
                chosen_key = key
                tm_id = None
                tm_name = tm_name_value or None
                method = "dict_unmatched"
                score = 0
                break

        if chosen_key is None:
            method = "no_mapping"
            tm_id = None
            tm_name = None
            score = 0

        if is_brest(team_main):
            log_brest(
                "[club] team=%r comp=%r chosen_key=%r tm_id=%s tm_name=%r method=%s score=%s"
                % (team_main, comp, chosen_key, tm_id, tm_name, method, score)
            )

        if tm_id is not None:
            lookup[(team_main, comp_country)] = tm_id
            lookup[(team_main, "")] = tm_id

        rows.append(
            ClubMatch(
                team=team_main,
                competition_name=comp,
                competition_country=comp_country,
                tm_club_name=tm_name,
                tm_club_id=tm_id,
                method=method,
                score=score,
            )
        )

    df = pd.DataFrame([asdict(row) for row in rows])
    matched = df[df["tm_club_id"].notna()]
    unmatched = len(df) - len(matched)
    coverage = (len(matched) / len(df)) * 100 if len(df) else 0
    logger.info(
        "Club mapping summary (manual dict): total=%s matched=%s (%.1f%%) unmatched=%s",
        len(df),
        len(matched),
        coverage,
        unmatched,
    )
    return df, lookup


def match_player_to_tm(
    player_row: pd.Series,
    tm_players: pd.DataFrame,
    threshold: int,
) -> tuple[Optional[int], Optional[str], str, int]:
    if tm_players.empty:
        return None, None, "no_club_match", 0

    player_short = clean_wyscout_player_name(player_row["player"])
    norm = normalise_name(player_short)
    tm_players = tm_players.copy()
    tm_players["tm_clean"] = tm_players["player_name"].apply(normalise_name)

    exact = tm_players[tm_players["tm_clean"] == norm]
    if not exact.empty:
        rec = exact.iloc[0]
        return int(rec["player_id"]), rec["player_name"], "exact", 100

    names = tm_players["player_name"].tolist()
    best = process.extractOne(player_short, names, scorer=fuzz.token_sort_ratio)
    if best and best[1] >= threshold:
        rec = tm_players.loc[tm_players["player_name"] == best[0]].iloc[0]
        return int(rec["player_id"]), rec["player_name"], "fuzzy_name", int(best[1])

    wyscout_age = player_row.get("age_value")
    age_threshold = max(70, threshold - 10)
    if pd.notna(wyscout_age):
        allowed = tm_players[pd.notna(tm_players["tm_age"])]
        allowed = allowed[allowed["tm_age"].between(wyscout_age - 1, wyscout_age + 1)]
        if not allowed.empty:
            best = process.extractOne(player_short, allowed["player_name"].tolist(), scorer=fuzz.WRatio)
            if best and best[1] >= age_threshold:
                rec = allowed.loc[allowed["player_name"] == best[0]].iloc[0]
                score = int(best[1])
                return int(rec["player_id"]), rec["player_name"], "age_assisted", score

    birth_date = player_row.get("birth_date_value")
    if birth_date:
        candidates = tm_players[tm_players["tm_birth_date"] == birth_date]
        if not candidates.empty:
            best = process.extractOne(
                player_short,
                candidates["player_name"].tolist(),
                scorer=fuzz.token_sort_ratio,
            )
            if best and best[1] >= age_threshold:
                rec = candidates.loc[candidates["player_name"] == best[0]].iloc[0]
                return int(rec["player_id"]), rec["player_name"], "birthdate_match", int(best[1])
            rec = candidates.iloc[0]
            return int(rec["player_id"]), rec["player_name"], "birthdate_match", 100

    return None, None, "unmatched", 0


def build_player_mapping(
    wyscout_df: pd.DataFrame,
    tm_df: pd.DataFrame,
    club_lookup: dict[tuple[str, str], int],
    threshold: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    wyscout_df = wyscout_df.copy()
    wyscout_df["age_value"] = wyscout_df["age"].apply(parse_wyscout_age)
    wyscout_df["player_short"] = wyscout_df["player"].apply(clean_wyscout_player_name)
    birth_series = wyscout_df["birth_date"] if "birth_date" in wyscout_df.columns else pd.Series([None] * len(wyscout_df), index=wyscout_df.index)
    wyscout_df["birth_date_value"] = birth_series.apply(parse_birth_date)
    wyscout_df["competition_country"] = wyscout_df["competition_name"].apply(competition_country)

    tm_grouped = tm_df.groupby("club_id")

    rows: list[PlayerMatch] = []
    mapping_records: list[dict[str, object]] = []

    for _, player_row in wyscout_df.iterrows():
        team_value = player_row.get("team_in_selected_period") or player_row.get("team")
        team = str(team_value).strip()
        key = (team, player_row["competition_country"])
        tm_club_id = club_lookup.get(key) or club_lookup.get((team, ""))
        if not tm_club_id:
            if is_brest(team):
                log_brest(
                    "[player] team=%r comp=%r calendar=%r -> no_club_match"
                    % (
                        team,
                        player_row.get("competition_name", ""),
                        player_row.get("calendar", ""),
                    )
                )
            match = PlayerMatch(
                player=player_row["player"],
                player_short=player_row["player_short"],
                team=team,
                competition_name=player_row.get("competition_name", ""),
                calendar=player_row.get("calendar", ""),
                age=player_row.get("age_value"),
                birth_date=player_row.get("birth_date_value"),
                tm_player_id=None,
                tm_player_name=None,
                tm_club_id=None,
                method="no_club_match",
                score=0,
                age_diff=None,
            )
            rows.append(match)
            continue

        tm_players = tm_grouped.get_group(tm_club_id) if tm_club_id in tm_grouped.groups else pd.DataFrame()
        tm_player_id, tm_player_name, method, score = match_player_to_tm(player_row, tm_players, threshold)

        age_diff = None
        if tm_player_id and pd.notna(player_row.get("age_value")):
            tm_age = tm_players.loc[tm_players["player_id"] == tm_player_id, "tm_age"]
            if not tm_age.empty and pd.notna(tm_age.iloc[0]):
                age_diff = int(tm_age.iloc[0]) - int(player_row["age_value"])

        if is_brest(team):
            log_brest(
                "[player] team=%r comp=%r calendar=%r candidates=%s matched_id=%s matched_name=%r method=%s score=%s age_diff=%s"
                % (
                    team,
                    player_row.get("competition_name", ""),
                    player_row.get("calendar", ""),
                    len(tm_players),
                    tm_player_id,
                    tm_player_name,
                    method,
                    score,
                    age_diff,
                )
            )

        rows.append(
            PlayerMatch(
                player=player_row["player"],
                player_short=player_row["player_short"],
                team=team,
                competition_name=player_row.get("competition_name", ""),
                calendar=player_row.get("calendar", ""),
                age=player_row.get("age_value"),
                birth_date=player_row.get("birth_date_value"),
                tm_player_id=tm_player_id,
                tm_player_name=tm_player_name,
                tm_club_id=tm_club_id,
                method=method,
                score=score,
                age_diff=age_diff,
            )
        )

        if tm_player_id:
            mapping_records.append(
                {
                    "player": player_row["player"],
                    "team_in_selected_period": team,
                    "competition_name": player_row.get("competition_name", ""),
                    "calendar": player_row.get("calendar", ""),
                    "tm_player_id": tm_player_id,
                }
            )

    mapping_df = (
        pd.DataFrame(mapping_records)
        .drop_duplicates(subset=["player", "team_in_selected_period", "competition_name", "calendar"])
    )
    player_reference_df = pd.DataFrame([asdict(row) for row in rows])
    stats = player_reference_df["method"].value_counts().to_dict()
    logger.info(
        "Player matching summary: total=%s %s",
        len(player_reference_df),
        ", ".join(f"{key}={val}" for key, val in stats.items()),
    )
    return player_reference_df, mapping_df


def read_wyscout_dataset(key: str, fallback: Path) -> pd.DataFrame:
    try:
        logger.info("Reading Wyscout dataset from S3 key '%s'", key)
        return read_csv_from_s3(key)
    except (FileNotFoundError, S3ConfigurationError, OSError) as exc:
        logger.warning("Unable to load S3 data (%s). Falling back to %s", exc, fallback)
        if not fallback.exists():
            raise FileNotFoundError(f"Wyscout dataset not found at {fallback}") from exc
        return pd.read_csv(fallback)


def write_wyscout_dataset(
    df: pd.DataFrame,
    output_key: str,
    output_local: Optional[Path],
    dry_run: bool,
) -> None:
    if output_local:
        output_local.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_local, index=False)
        logger.info("Wrote enriched dataset locally: %s", output_local)
    if dry_run:
        logger.info("Dry-run enabled: skipping S3 upload.")
        return
    try:
        write_csv_to_s3(df, output_key, index=False)
        logger.info("Uploaded enriched dataset to S3 key '%s'", output_key)
    except S3ConfigurationError as exc:
        logger.warning("Skipping S3 upload (configuration error): %s", exc)


def rename_tm_columns(tm_df: pd.DataFrame) -> pd.DataFrame:
    df = tm_df.copy()
    df = df.sort_values("profile_updated_at", na_position="first")
    df = df.drop_duplicates(subset=["player_id"], keep="last")
    df = df.rename(columns={"player_id": "tm_player_id"})
    rename_map = {
        col: f"tm_{col}"
        for col in df.columns
        if col not in {"tm_player_id"} and not col.startswith("tm_")
    }
    return df.rename(columns=rename_map)


def merge_datasets(
    wyscout_df: pd.DataFrame,
    player_mapping_df: pd.DataFrame,
    tm_df: pd.DataFrame,
) -> pd.DataFrame:
    merged = wyscout_df.merge(
        player_mapping_df,
        on=["player", "team_in_selected_period", "competition_name", "calendar"],
        how="left",
    )
    merged_tm = rename_tm_columns(tm_df)
    merged = merged.merge(merged_tm, on="tm_player_id", how="left")
    logger.info(
        "Merged dataset: %s rows, %s enriched with Transfermarkt data",
        len(merged),
        merged["tm_player_id"].notna().sum(),
    )
    return merged


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Match Transfermarkt metadata with Wyscout data.")
    parser.add_argument("--wyscout-key", default=DEFAULT_WYSCOUT_KEY, help="S3 key of the cleaned Wyscout dataset.")
    parser.add_argument(
        "--wyscout-local",
        default=str(DEFAULT_WYSCOUT_LOCAL),
        help="Local fallback CSV containing Wyscout data.",
    )
    parser.add_argument(
        "--transfermarkt-path",
        default=str(DEFAULT_TM_PATH),
        help="Path to transfermarkt_profiles.csv",
    )
    parser.add_argument(
        "--output-local",
        default=str(DEFAULT_OUTPUT_LOCAL) if DEFAULT_OUTPUT_LOCAL else None,
        help="Optional local path where the enriched dataset will be written. "
        "Omit to skip local writes.",
    )
    parser.add_argument(
        "--output-key",
        default=DEFAULT_WYSCOUT_KEY,
        help="S3 key for the enriched dataset (default overwrites cleaned file).",
    )
    parser.add_argument(
        "--club-mapping",
        default=str(DEFAULT_CLUB_MAPPING),
        help="CSV output capturing club correspondences.",
    )
    parser.add_argument(
        "--player-mapping",
        default=str(DEFAULT_PLAYER_MAPPING),
        help="CSV output capturing player correspondences.",
    )
    parser.add_argument(
        "--tm-clubs-out",
        default=str(DEFAULT_TM_CLUBS),
        help="CSV output listing all Transfermarkt clubs (for manual review).",
    )
    parser.add_argument(
        "--club-threshold",
        type=int,
        default=90,
        help="Legacy arg (ignored): club mapping now uses the curated dictionary only.",
    )
    parser.add_argument("--player-threshold", type=int, default=92, help="Fuzzy ratio threshold for players.")
    parser.add_argument(
        "--club-mapping-py",
        default=str(DEFAULT_CLUB_MAPPING_PY),
        help="Python file containing the curated club mapping dictionary (wyscout -> tm).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Skip S3 upload.")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        BREST_LOG_PATH.write_text("Brest debug log\n", encoding="utf-8")
    except Exception:
        logger.debug("Unable to reset Brest debug log file.")

    wyscout_path = Path(args.wyscout_local)
    tm_path = Path(args.transfermarkt_path)
    output_local = Path(args.output_local) if args.output_local else None
    club_mapping_path = Path(args.club_mapping)
    player_mapping_path = Path(args.player_mapping)
    tm_clubs_out_path = Path(args.tm_clubs_out)
    club_mapping_py_path = Path(args.club_mapping_py)

    if not tm_path.exists():
        raise FileNotFoundError(f"Transfermarkt file not found at {tm_path}")

    wyscout_df = read_wyscout_dataset(args.wyscout_key, wyscout_path)
    tm_df = pd.read_csv(tm_path)
    tm_df = add_tm_enrichments(tm_df)
    logger.info("Loaded Transfermarkt profiles: %s rows", len(tm_df))

    manual_club_mapping = load_manual_club_mapping(club_mapping_py_path)
    if args.club_threshold != 90:
        logger.info("Club threshold argument is ignored when using the manual mapping dict.")

    club_mapping_df, club_lookup = build_club_mapping(
        wyscout_df,
        manual_club_mapping,
    )

    player_mapping_reference_df, player_mapping_df = build_player_mapping(
        wyscout_df,
        tm_df,
        club_lookup,
        threshold=args.player_threshold,
    )

    enriched_df = merge_datasets(wyscout_df, player_mapping_df, tm_df)

    club_mapping_path.parent.mkdir(parents=True, exist_ok=True)
    player_mapping_path.parent.mkdir(parents=True, exist_ok=True)
    tm_clubs_out_path.parent.mkdir(parents=True, exist_ok=True)
    club_mapping_py_path.parent.mkdir(parents=True, exist_ok=True)

    club_mapping_df.to_csv(club_mapping_path, index=False)
    player_mapping_reference_df.to_csv(player_mapping_path, index=False)
    tm_clubs_catalog = (
        tm_df[["club_id", "club_name"]]
        .dropna(subset=["club_id", "club_name"])
        .drop_duplicates(subset=["club_id"])
        .sort_values("club_name")
    )
    tm_clubs_catalog.to_csv(tm_clubs_out_path, index=False)
    logger.info("Wrote reference mappings:\n  Clubs: %s\n  Players: %s", club_mapping_path, player_mapping_path)
    logger.info("Wrote Transfermarkt clubs catalog: %s", tm_clubs_out_path)

    write_wyscout_dataset(
        enriched_df,
        output_key=args.output_key,
        output_local=output_local,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
