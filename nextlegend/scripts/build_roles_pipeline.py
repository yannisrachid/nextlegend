"""NextLegend roles and similarity pipeline.

This script enriches the Wyscout players dataset with profile scores, percentiles,
role assignment, and similarity exports as defined in codex/Algo.MD.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
import re
from typing import Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = PACKAGE_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nextlegend.s3_utils import read_csv_from_s3, write_csv_to_s3


DEFAULT_RAW_INPUT = "data/wyscout_players_final.csv"
DEFAULT_INPUT = "data/wyscout_players_cleaned.csv"
DEFAULT_PROFILES = PACKAGE_ROOT / "player_profiles.json"
DEFAULT_OUT_ENRICHED = "data/wyscout_players_cleaned.csv"
DEFAULT_OUT_SCORES = "players_scores.csv"
DEFAULT_OUT_LEAGUE = "roles_scores_league.csv"
DEFAULT_OUT_GLOBAL = "roles_scores_global.csv"
SIMILARITY_PREFIX = "data/similarity"
LEAGUE_META_KEY = "data/league_translation_meta.csv"
LEAGUE_FACTOR_COL = "league_strength_factor"
PCT_SUFFIX_GLOBAL = "_pct_global"

SUMMARY_DEFINITIONS: dict[str, tuple[str, ...]] = {
    "summary_finishing": (
        "goals_per_90",
        "shots_per_90",
        "shots_on_target_percent",
        "goal_conversion_rate",
        "xg_per_90",
        "touches_in_penalty_area_per_90",
    ),
    "summary_aerial": (
        "aerial_duels_per_90",
        "aerial_duels_won_percent",
        "headed_goals_per_90",
    ),
    "summary_defense": (
        "successful_def_actions_per_90",
        "def_duels_won_percent",
        "interceptions_per_90",
        "sliding_tackles_per_90",
        "blocked_shots_per_90",
    ),
    "summary_technique": (
        "successful_dribbles_percent",
        "dribbles_per_90",
        "progressive_runs_per_90",
        "touches_in_penalty_area_per_90",
    ),
    "summary_creation": (
        "assists_per_90",
        "xa_per_90",
        "key_passes_per_90",
        "smart_passes_per_90",
        "passes_to_penalty_area_per_90",
        "deep_completions_per_90",
    ),
    "summary_construction": (
        "passes_per_90",
        "progressive_passes_per_90",
        "passes_to_final_third_per_90",
        "through_passes_per_90",
        "accurate_passes_percent",
    ),
}

_ZSCORE_CACHE: Dict[str, pd.Series] = {}
PLAYER_PATTERN = re.compile(r"\(([-\d]+)\)")
AGE_PATTERN = re.compile(r"'?(\d{2})(?:\s*\((\d{1,2})\))?")


def _use_local_path(path: str | os.PathLike[str]) -> bool:
    if isinstance(path, Path):
        return True
    text = str(path)
    return os.path.isabs(text) or text.startswith("./") or text.startswith("../")


def _read_csv_any(path: str | os.PathLike[str]) -> pd.DataFrame:
    if _use_local_path(path):
        return pd.read_csv(Path(path))
    return read_csv_from_s3(str(path))


def _write_csv_any(df: pd.DataFrame, path: str | os.PathLike[str], *, index: bool = False) -> None:
    if _use_local_path(path):
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(destination, index=index)
        return
    write_csv_to_s3(df, str(path), index=index)


def _split_player_cell(cell: object) -> tuple[str, Optional[str]]:
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return "", None
    text = str(cell)
    base_name = text.split(";", 1)[0].strip()
    match = PLAYER_PATTERN.search(text)
    identifier = match.group(1) if match else None
    return base_name, identifier


def _interpret_age_cell(cell: object) -> tuple[Optional[int], Optional[int]]:
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return None, None
    text = str(cell).strip()
    if not text:
        return None, None
    match = AGE_PATTERN.search(text)
    if not match:
        return None, None
    two_digit_year = match.group(1)
    age_value = match.group(2)
    year_int = int(two_digit_year)
    birth_year = 2000 + year_int if year_int <= 24 else 1900 + year_int
    age = int(age_value) if age_value else None
    return birth_year, age


def clean_players_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    if "player" in working.columns:
        tuples = working["player"].apply(_split_player_cell).tolist()
        names = [name for name, _ in tuples]
        identifiers = [identifier for _, identifier in tuples]
        working["player"] = names
        if "player_id" in working.columns:
            working = working.drop(columns=["player_id"])
        insert_pos = working.columns.get_loc("player") + 1
        working.insert(insert_pos, "player_id", identifiers)
    if "age" in working.columns:
        birth_years: list[Optional[int]] = []
        ages: list[Optional[int]] = []
        for value in working["age"]:
            birth, age = _interpret_age_cell(value)
            birth_years.append(birth)
            ages.append(age)
        working["age"] = ages
        if "birth_year" in working.columns:
            working = working.drop(columns=["birth_year"])
        insert_pos = working.columns.get_loc("age")
        working.insert(insert_pos, "birth_year", birth_years)
    return working


def aggregate_player_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    key_cols: list[str] = []
    if "player_id" in df.columns and df["player_id"].notna().any():
        key_cols.append("player_id")
    elif "player" in df.columns:
        key_cols.append("player")

    for extra in ("competition_name", "calendar"):
        if extra in df.columns:
            key_cols.append(extra)

    if not key_cols:
        return df

    working = df.copy()
    working["_agg_minutes"] = pd.to_numeric(working.get("minutes_played"), errors="coerce").fillna(-1)
    working["_agg_complete"] = working.notna().sum(axis=1)
    working["_agg_order"] = np.arange(len(working))

    sort_cols = key_cols + ["_agg_minutes", "_agg_complete", "_agg_order"]
    ascending = [True] * len(key_cols) + [False, False, True]

    reduced = (
        working.sort_values(sort_cols, ascending=ascending)
        .drop_duplicates(subset=key_cols, keep="first")
        .drop(columns=["_agg_minutes", "_agg_complete", "_agg_order"])
        .reset_index(drop=True)
    )
    return reduced


def load_league_strength_factors(path: str | os.PathLike[str]) -> dict[str, float]:
    try:
        meta_df = _read_csv_any(path)
    except FileNotFoundError:
        return {}

    if "competition" not in meta_df.columns or "difficulty" not in meta_df.columns:
        return {}

    difficulty = pd.to_numeric(meta_df["difficulty"], errors="coerce")
    if difficulty.isna().all():
        return {}

    mean_val = difficulty.mean(skipna=True)
    if not np.isfinite(mean_val) or mean_val == 0:
        return {}

    normalized = (difficulty / mean_val).clip(lower=0.5, upper=1.5)
    return {
        str(comp): float(norm) if np.isfinite(norm) else 1.0
        for comp, norm in zip(meta_df["competition"], normalized)
    }


def load_profiles(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_positions(cell: str) -> list[str]:
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    if not isinstance(cell, str):
        cell = str(cell)
    return [token.strip() for token in cell.split(",") if token.strip()]


def split_positions_cols(df: pd.DataFrame, position_col: str = "position") -> pd.DataFrame:
    result = df.copy()
    if position_col not in result.columns:
        result[position_col] = ""
    tokens = result[position_col].apply(parse_positions)
    primary = tokens.apply(lambda lst: lst[0] if lst else "")
    secondary = tokens.apply(lambda lst: ", ".join(lst[1:]) if len(lst) > 1 else "")
    result[position_col] = primary
    result["second_position"] = secondary
    return result


def _coerce_numeric(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        return pd.to_numeric(series.replace({"-": np.nan, "": np.nan}), errors="coerce")
    return pd.to_numeric(series, errors="coerce")


def _zscore(series: pd.Series) -> pd.Series:
    numeric = _coerce_numeric(series)
    mean = numeric.mean()
    std = numeric.std(ddof=0)
    if std == 0 or np.isnan(std):
        z = pd.Series(0.0, index=series.index, dtype=float)
    else:
        z = (numeric - mean) / std
    return z.fillna(0.0)


def _positions_set(df: pd.DataFrame) -> pd.Series:
    primary = df.get("position", pd.Series("", index=df.index)).fillna("")
    secondary = df.get("second_position", pd.Series("", index=df.index)).fillna("")
    sets = []
    for prim, sec in zip(primary, secondary):
        pos_set = set()
        if prim:
            pos_set.add(str(prim).strip())
        for token in parse_positions(sec):
            pos_set.add(token)
        sets.append(pos_set)
    return pd.Series(sets, index=df.index)


def compute_raw_scores(
    df: pd.DataFrame,
    profiles: Mapping[str, Mapping[str, object]],
    minutes_col: str = "minutes_played",
) -> pd.DataFrame:
    global _ZSCORE_CACHE
    working = df.copy()
    if minutes_col not in working.columns:
        working[minutes_col] = 0
    minutes = _coerce_numeric(working[minutes_col]).fillna(0)

    required_metrics = set()
    inverse_metrics = set()
    for profile in profiles.values():
        weights = profile.get("weights", {}) or {}
        required_metrics.update(weights.keys())
        inverse_metrics.update(profile.get("lower_is_better", []) or [])

    for metric in required_metrics.union(inverse_metrics):
        if metric not in working.columns:
            working[metric] = np.nan

    _ZSCORE_CACHE = {}
    metric_reference = required_metrics.union(inverse_metrics)
    for metric in metric_reference:
        _ZSCORE_CACHE[metric] = _zscore(working[metric])

    position_sets = _positions_set(working)

    raw_scores: dict[str, pd.Series] = {}
    for profile_name, profile in profiles.items():
        weights = {k: float(v) for k, v in (profile.get("weights", {}) or {}).items()}
        lower_is_better = set(profile.get("lower_is_better", []) or [])
        if not weights:
            raw_scores[profile_name] = pd.Series(np.nan, index=working.index, dtype=float)
            continue

        weight_sum = sum(weights.values())
        normalized = {metric: (weight / weight_sum) if weight_sum else 0.0 for metric, weight in weights.items()}

        score = pd.Series(0.0, index=working.index, dtype=float)
        for metric, weight in normalized.items():
            z_series = _ZSCORE_CACHE.get(metric)
            if z_series is None:
                z_series = _zscore(working[metric])
                _ZSCORE_CACHE[metric] = z_series
            if metric in lower_is_better:
                z_series = -z_series
            score = score.add(z_series * weight, fill_value=0.0)

        pos_groups = set(profile.get("position_groups", []) or [])
        if pos_groups:
            eligible_mask = position_sets.apply(lambda s: bool(s & pos_groups))
            score = score.mask(~eligible_mask, other=np.nan)

        min_minutes = float(profile.get("min_minutes", 0) or 0)
        score = score.mask(minutes < min_minutes, other=np.nan)

        raw_scores[profile_name] = score

    return pd.DataFrame(raw_scores)


def compute_scores_percentiles(raw_scores: pd.DataFrame) -> pd.DataFrame:
    pct = raw_scores.copy()
    for col in pct.columns:
        pct[col] = percentiles_by_group(raw_scores[col], None)
    return pct


def percentiles_by_group(series: pd.Series, group: Optional[pd.Series]) -> pd.Series:
    result = pd.Series(np.nan, index=series.index, dtype=float)
    mask = series.notna()
    if not mask.any():
        return result
    if group is None:
        ranks = series[mask].rank(pct=True, method="average") * 100
        result.loc[mask] = ranks
        return result
    aligned_group = group.reindex(series.index)
    ranks = (
        series[mask]
        .groupby(aligned_group[mask])
        .rank(pct=True, method="average")
        * 100
    )
    result.loc[ranks.index] = ranks
    return result


def assign_role(
    df: pd.DataFrame,
    scores_pct: pd.DataFrame,
    profiles: Mapping[str, Mapping[str, object]],
    position_col: str = "position",
    second_position_col: str = "second_position",
) -> pd.Series:
    def eligible_profiles_for_row(row: pd.Series) -> list[str]:
        primary = str(row.get(position_col, "") or "").strip()
        secondary = str(row.get(second_position_col, "") or "")
        tokens = set(parse_positions(secondary))
        if primary:
            tokens.add(primary)
        candidates = []
        for name, profile in profiles.items():
            groups = set(profile.get("position_groups", []) or [])
            if not groups or (tokens and groups.intersection(tokens)):
                candidates.append(name)
        if not candidates:
            candidates = list(profiles.keys())
        return candidates

    assignments = []
    for idx, row in df.iterrows():
        candidates = eligible_profiles_for_row(row)
        row_scores = scores_pct.loc[idx, candidates]
        if row_scores.dropna().empty:
            assignments.append("")
        else:
            assignments.append(row_scores.idxmax())
    return pd.Series(assignments, index=df.index, dtype="object")


def estimate_minutes_possible(df: pd.DataFrame) -> pd.Series:
    comp = df.get("competition_name", pd.Series(index=df.index, dtype="object")).fillna("GLOBAL").astype(str)
    cal = df.get("calendar", pd.Series(index=df.index, dtype="object")).fillna("GLOBAL").astype(str)
    key = comp + "||" + cal
    matches = _coerce_numeric(df.get("matches_played", pd.Series(index=df.index, dtype=float))).fillna(0)
    minutes_possible = matches.groupby(key).transform("max") * 90
    return minutes_possible.reindex(df.index).fillna(0)


def eligibility_league(df: pd.DataFrame, minutes_possible: pd.Series, frac: float = 0.33) -> pd.Series:
    minutes_played = _coerce_numeric(df.get("minutes_played", pd.Series(index=df.index, dtype=float))).fillna(0)
    threshold = minutes_possible * frac
    threshold = threshold.fillna(np.inf)
    return minutes_played >= threshold


def eligibility_global(df: pd.DataFrame, min_minutes: float = 500) -> pd.Series:
    minutes_played = _coerce_numeric(df.get("minutes_played", pd.Series(index=df.index, dtype=float))).fillna(0)
    return minutes_played >= float(min_minutes)


def _league_group(df: pd.DataFrame) -> pd.Series:
    comp = df.get("competition_name", pd.Series(index=df.index, dtype="object")).fillna("GLOBAL").astype(str)
    cal = df.get("calendar", pd.Series(index=df.index, dtype="object")).fillna("GLOBAL").astype(str)
    return comp + "||" + cal


def roles_league_percentiles(
    df: pd.DataFrame,
    raw_scores: pd.DataFrame,
    assigned_role: pd.Series,
    profiles: Mapping[str, Mapping[str, object]],
) -> pd.DataFrame:
    group = _league_group(df)
    elig = df.get("_elig_league", pd.Series(False, index=df.index))
    result = {}
    for profile_name in profiles.keys():
        series = pd.Series(np.nan, index=df.index, dtype=float)
        mask = (assigned_role == profile_name) & elig & raw_scores[profile_name].notna()
        series.loc[mask] = raw_scores.loc[mask, profile_name]
        result[profile_name] = percentiles_by_group(series, group)
    return pd.DataFrame(result, index=df.index)


def roles_global_percentiles(
    df: pd.DataFrame,
    raw_scores: pd.DataFrame,
    assigned_role: pd.Series,
    profiles: Mapping[str, Mapping[str, object]],
    min_minutes: float = 500,
) -> pd.DataFrame:
    elig = eligibility_global(df, min_minutes=min_minutes)
    result = {}
    for profile_name in profiles.keys():
        series = pd.Series(np.nan, index=df.index, dtype=float)
        mask = (assigned_role == profile_name) & elig & raw_scores[profile_name].notna()
        series.loc[mask] = raw_scores.loc[mask, profile_name]
        result[profile_name] = percentiles_by_group(series, None)
    return pd.DataFrame(result, index=df.index)


def slugify_profile(name: str) -> str:
    allowed = []
    for char in name.lower():
        if char.isalnum():
            allowed.append(char)
        elif char in {" ", "-", "/", "_"}:
            allowed.append("-")
    slug = "".join(allowed)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-") or "profile"


def _foot_numeric(value: object) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    text = str(value).strip().lower()
    if text in {"right", "droite", "r"}:
        return 1.0
    if text in {"left", "gauche", "l"}:
        return -1.0
    if text in {"both", "ambidextrous", "ambidextre"}:
        return 0.0
    return 0.0


def profile_similarity(
    df: pd.DataFrame,
    profiles: Mapping[str, Mapping[str, object]],
    assigned_role: pd.Series,
    profile_name: str,
    topk: int = 10,
) -> pd.DataFrame:
    profile = profiles[profile_name]
    weights = profile.get("weights", {}) or {}
    if not weights:
        return pd.DataFrame(columns=[
            "player_a",
            "team_a",
            "competition_name_a",
            "player_b",
            "team_b",
            "competition_name_b",
            "profile",
            "similarity",
        ])

    weight_sum = sum(weights.values())
    normalized = {metric: (float(weight) / weight_sum) if weight_sum else 0.0 for metric, weight in weights.items()}
    lower_is_better = set(profile.get("lower_is_better", []) or [])

    mask_profile = assigned_role == profile_name
    mask_minutes = eligibility_global(df, min_minutes=500)
    mask = mask_profile & mask_minutes
    if not mask.any():
        return pd.DataFrame(columns=[
            "player_a",
            "team_a",
            "competition_name_a",
            "player_b",
            "team_b",
            "competition_name_b",
            "profile",
            "similarity",
        ])

    indices = df.index[mask]
    feature_blocks = []
    for metric in normalized.keys():
        zvalues = _ZSCORE_CACHE.get(metric)
        if zvalues is None:
            zvalues = _zscore(df[metric])
            _ZSCORE_CACHE[metric] = zvalues
        if metric in lower_is_better:
            zvalues = -zvalues
        feature_blocks.append(zvalues.loc[indices].to_numpy(dtype=float))

    # Additional contextual features (age removed per updated requirements)
    height_base = df.get("height", df.get("height_cm", pd.Series(np.nan, index=df.index)))
    weight_base = df.get("weight", df.get("weight_kg", pd.Series(np.nan, index=df.index)))
    height_z = _zscore(height_base) if isinstance(height_base, pd.Series) else pd.Series(0.0, index=df.index)
    weight_z = _zscore(weight_base) if isinstance(weight_base, pd.Series) else pd.Series(0.0, index=df.index)
    foot_series = df.get("foot", pd.Series(np.nan, index=df.index)).apply(_foot_numeric).fillna(0.0)

    feature_blocks.append(height_z.loc[indices].to_numpy(dtype=float))
    feature_blocks.append(weight_z.loc[indices].to_numpy(dtype=float))
    feature_blocks.append(foot_series.loc[indices].to_numpy(dtype=float))

    if not feature_blocks:
        return pd.DataFrame(columns=[
            "player_a",
            "team_a",
            "competition_name_a",
            "player_b",
            "team_b",
            "competition_name_b",
            "profile",
            "similarity",
        ])

    matrix = np.stack(feature_blocks, axis=1)
    weights_vector = np.array([normalized[m] for m in normalized.keys()], dtype=float)
    if weights_vector.size:
        weights_vector = np.sqrt(weights_vector)
        metric_dim = weights_vector.size
        matrix[:, :metric_dim] *= weights_vector

    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(matrix, axis=1)
    valid_mask = norms > 0
    if valid_mask.sum() < 2:
        return pd.DataFrame(columns=[
            "player_a",
            "team_a",
            "competition_name_a",
            "player_b",
            "team_b",
            "competition_name_b",
            "profile",
            "similarity",
        ])

    matrix = matrix[valid_mask]
    norms = norms[valid_mask]
    normed = matrix / norms[:, None]
    indices = indices[valid_mask]

    similarities = normed @ normed.T
    np.fill_diagonal(similarities, -np.inf)

    player_ids = df.get("player_id", pd.Series(index=df.index, dtype="object"))

    records = []
    for row_pos, player_idx in enumerate(indices):
        row = similarities[row_pos]
        if not np.isfinite(row).any():
            continue
        order = np.argsort(-row)
        neighbours_added = 0
        raw_id_a = player_ids.at[player_idx] if player_idx in player_ids.index else None
        player_id_a = raw_id_a if pd.notna(raw_id_a) else None
        player_name_a = df.at[player_idx, "player"] if "player" in df.columns else player_idx
        seen_indices: set[int] = set()
        seen_ids: set[object] = set()
        seen_names: set[str] = set()
        for col_pos in order:
            if neighbours_added >= topk:
                break
            value = row[col_pos]
            if not np.isfinite(value) or value <= -np.inf:
                continue
            neighbour_idx = indices[col_pos]
            if neighbour_idx == player_idx:
                continue
            if neighbour_idx in seen_indices:
                continue
            raw_id_b = player_ids.at[neighbour_idx] if neighbour_idx in player_ids.index else None
            player_id_b = raw_id_b if pd.notna(raw_id_b) else None
            if (
                player_id_a is not None
                and player_id_b is not None
                and player_id_a == player_id_b
            ):
                continue
            player_name_b = df.at[neighbour_idx, "player"] if "player" in df.columns else neighbour_idx
            if (
                (player_id_a is None or player_id_b is None)
                and str(player_name_b) == str(player_name_a)
            ):
                continue
            if player_id_b is not None:
                if player_id_b in seen_ids:
                    continue
            else:
                if str(player_name_b) in seen_names:
                    continue

            records.append(
                {
                    "player_a": player_name_a,
                    "team_a": df.at[player_idx, "team"] if "team" in df.columns else "",
                    "competition_name_a": df.at[player_idx, "competition_name"] if "competition_name" in df.columns else "",
                    "player_b": player_name_b,
                    "team_b": df.at[neighbour_idx, "team"] if "team" in df.columns else "",
                    "competition_name_b": df.at[neighbour_idx, "competition_name"] if "competition_name" in df.columns else "",
                    "profile": profile_name,
                    "similarity": float(value),
                }
            )
            neighbours_added += 1
            seen_indices.add(neighbour_idx)
            if player_id_b is not None:
                seen_ids.add(player_id_b)
            else:
                seen_names.add(str(player_name_b))
    return pd.DataFrame(records)


def _numeric_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    numeric = {col: _coerce_numeric(df[col]) for col in df.columns}
    numeric_df = pd.DataFrame(numeric, index=df.index)
    return numeric_df.dropna(axis=1, how="all")


def compute_metric_percentiles(df: pd.DataFrame, group: Optional[pd.Series]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    numeric_df = _numeric_dataframe(df)

    global_columns = {
        f"{col}_pct_global": percentiles_by_group(numeric_df[col], None)
        for col in numeric_df.columns
    }
    global_pct = pd.DataFrame(global_columns, index=df.index)

    if group is None:
        league_pct = pd.DataFrame(index=df.index)
    else:
        league_columns = {
            f"{col}_pct_league": percentiles_by_group(numeric_df[col], group)
            for col in numeric_df.columns
        }
        league_pct = pd.DataFrame(league_columns, index=df.index)

    return league_pct, global_pct


def compute_summary_scores(metrics_global_pct: pd.DataFrame) -> pd.DataFrame:
    summary_frames = {}
    for column, metric_tuple in SUMMARY_DEFINITIONS.items():
        candidate_columns = [f"{metric}{PCT_SUFFIX_GLOBAL}" for metric in metric_tuple]
        existing = [metrics_global_pct[col] for col in candidate_columns if col in metrics_global_pct]
        if not existing:
            continue
        combined = pd.concat(existing, axis=1)
        summary_frames[column] = combined.mean(axis=1, skipna=True)
    if not summary_frames:
        return pd.DataFrame(index=metrics_global_pct.index)
    return pd.DataFrame(summary_frames, index=metrics_global_pct.index)


def _run_tests() -> None:
    df_pos = pd.DataFrame({"position": ["RCB, RB", "CF"]})
    res_pos = split_positions_cols(df_pos)
    assert res_pos.loc[0, "position"] == "RCB"
    assert res_pos.loc[0, "second_position"] == "RB"
    assert res_pos.loc[1, "second_position"] == ""

    df_minutes = pd.DataFrame(
        {
            "competition_name": ["A", "A", "A"],
            "calendar": ["2024", "2024", "2024"],
            "matches_played": [34, 20, 12],
        }
    )
    minutes_possible = estimate_minutes_possible(df_minutes)
    assert minutes_possible.iloc[0] == 34 * 90

    elig = eligibility_league(
        pd.DataFrame({"minutes_played": [1100]}),
        pd.Series([3000])
    )
    assert bool(elig.iloc[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build NextLegend roles pipeline.")
    parser.add_argument("--raw_in", dest="raw_input", default=DEFAULT_RAW_INPUT, type=str)
    parser.add_argument("--in", dest="input_path", default=DEFAULT_INPUT, type=str)
    parser.add_argument("--profiles", dest="profiles_path", default=DEFAULT_PROFILES, type=Path)
    parser.add_argument("--out_enriched", dest="out_enriched", default=DEFAULT_OUT_ENRICHED, type=str)
    parser.add_argument("--out_scores", dest="out_scores", default=DEFAULT_OUT_SCORES, type=str)
    parser.add_argument("--out_league", dest="out_league", default=DEFAULT_OUT_LEAGUE, type=str)
    parser.add_argument("--out_global", dest="out_global", default=DEFAULT_OUT_GLOBAL, type=str)
    parser.add_argument("--sim_topk", dest="sim_topk", default=30, type=int)
    args = parser.parse_args()

    raw_key = (args.raw_input or "").strip()
    if raw_key:
        print("RAW_LOAD")
        raw_df = _read_csv_any(raw_key)
        print("RAW_CLEAN")
        cleaned_df = clean_players_dataframe(raw_df)
        cleaned_df = aggregate_player_rows(cleaned_df)
        print("RAW_WRITE")
        _write_csv_any(cleaned_df, args.input_path, index=False)
        df = cleaned_df.copy()
        print("LOAD (from cleaned in-memory)")
    else:
        print("LOAD")
        df = _read_csv_any(args.input_path)
        df = aggregate_player_rows(df)

    league_strength = load_league_strength_factors(LEAGUE_META_KEY)
    if league_strength:
        comp_series = None
        if "competition_name" in df.columns:
            comp_series = df["competition_name"].astype(str)
        elif "league" in df.columns:
            comp_series = df["league"].astype(str)

        if comp_series is not None:
            factors = comp_series.map(league_strength).fillna(1.0)
        else:
            factors = pd.Series(1.0, index=df.index, dtype=float)
        df[LEAGUE_FACTOR_COL] = pd.to_numeric(factors, errors="coerce").fillna(1.0)
    else:
        df[LEAGUE_FACTOR_COL] = 1.0
    profiles = load_profiles(args.profiles_path)

    print("SPLIT")
    df = split_positions_cols(df)

    print("SCORES_RAW")
    raw_scores = compute_raw_scores(df, profiles)
    if LEAGUE_FACTOR_COL in df.columns:
        raw_scores = raw_scores.mul(df[LEAGUE_FACTOR_COL], axis=0)
    scores_pct = compute_scores_percentiles(raw_scores)

    print("ASSIGN")
    assigned_role = assign_role(df, scores_pct, profiles)

    print("ELIG")
    minutes_possible = estimate_minutes_possible(df)
    elig_league = eligibility_league(df, minutes_possible, frac=0.15)
    df["_elig_league"] = elig_league

    print("LEAGUE_PCT")
    roles_scores_league = roles_league_percentiles(df, raw_scores, assigned_role, profiles)

    print("GLOBAL_PCT")
    roles_scores_global = roles_global_percentiles(df, raw_scores, assigned_role, profiles, min_minutes=270)

    print("SIM_OUT")
    for profile_name in profiles.keys():
        sim_df = profile_similarity(df, profiles, assigned_role, profile_name, topk=args.sim_topk)
        sim_key = f"{SIMILARITY_PREFIX}/similarity_{slugify_profile(profile_name)}.csv"
        _write_csv_any(sim_df, sim_key, index=False)

    print("WRITE")
    scores_pct_out = scores_pct.copy()
    if "player" in df.columns:
        scores_pct_out.insert(0, "player", df["player"])
    _write_csv_any(scores_pct_out, args.out_scores, index=False)

    league_group = _league_group(df)
    metrics_base = df.drop(columns=["_elig_league"], errors="ignore")
    metrics_league_pct, metrics_global_pct = compute_metric_percentiles(metrics_base, league_group)
    scores_league_pct, scores_global_pct = compute_metric_percentiles(raw_scores, league_group)

    summary_scores = compute_summary_scores(metrics_global_pct)

    enriched = df.copy()
    enriched = enriched.drop(columns=["_elig_league"], errors="ignore")
    enriched["assigned_role"] = assigned_role

    # Add per-profile percentiles and aggregate role-based percentiles
    role_league_pct = pd.Series(np.nan, index=df.index, dtype=float)
    role_global_pct = pd.Series(np.nan, index=df.index, dtype=float)
    for profile_name in profiles.keys():
        mask = assigned_role == profile_name
        if mask.any():
            role_league_pct.loc[mask] = roles_scores_league.loc[mask, profile_name]
            role_global_pct.loc[mask] = roles_scores_global.loc[mask, profile_name]

    enriched_extra = pd.concat(
        [
            scores_pct,
            metrics_league_pct,
            metrics_global_pct,
            scores_league_pct,
            scores_global_pct,
            role_league_pct.rename("assigned_role_pct_league"),
            role_global_pct.rename("assigned_role_pct_global"),
            summary_scores,
        ],
        axis=1,
    )
    overlapping = set(enriched.columns).intersection(enriched_extra.columns)
    enriched_extra = enriched_extra.drop(columns=list(overlapping), errors="ignore")
    enriched = pd.concat([enriched, enriched_extra], axis=1)
    _write_csv_any(enriched, args.out_enriched, index=False)

    roles_scores_league_out = pd.concat(
        [
            df.get("player", pd.Series(index=df.index)),
            assigned_role.rename("assigned_role"),
            roles_scores_league,
        ],
        axis=1,
    )
    roles_scores_league_out = roles_scores_league_out.dropna(subset=list(profiles.keys()), how="all")
    _write_csv_any(roles_scores_league_out, args.out_league, index=False)

    roles_scores_global_out = pd.concat(
        [
            df.get("player", pd.Series(index=df.index)),
            assigned_role.rename("assigned_role"),
            roles_scores_global,
        ],
        axis=1,
    )
    roles_scores_global_out = roles_scores_global_out.dropna(subset=list(profiles.keys()), how="all")
    _write_csv_any(roles_scores_global_out, args.out_global, index=False)

    print("DONE")


if __name__ == "__main__":
    _run_tests()
    main()
