"""
Processing utilities to convert the raw Wyscout CSV into structured artifacts
aligned with DATA_MODEL.md, en reprenant la logique v1 (scores, percentiles,
assigned_role, global_score_adjusted, summary_*, similarités).
"""

from __future__ import annotations

import json
import hashlib
import math
import time
import os
import re
import unicodedata
from pathlib import Path
from typing import Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz

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

_ZSCORE_CACHE: dict[str, pd.Series] = {}
SCORE_BAND_MIN = float(os.getenv("SCORE_BAND_MIN", "50") or "50")
SCORE_BAND_MAX = float(os.getenv("SCORE_BAND_MAX", "95") or "95")
if not np.isfinite(SCORE_BAND_MIN) or not np.isfinite(SCORE_BAND_MAX) or SCORE_BAND_MIN >= SCORE_BAND_MAX:
    SCORE_BAND_MIN = 50.0
    SCORE_BAND_MAX = 95.0


# --- Helpers v1 (extraits) ----------------------------------------------------


def parse_positions(cell: str) -> list[str]:
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
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


def _scale_score_series(series: pd.Series, skip_if_already_scaled: bool = False) -> pd.Series:
    numeric = _coerce_numeric(series)
    scaled = pd.Series(np.nan, index=series.index, dtype=float)
    mask = numeric.notna()
    if not mask.any():
        return scaled

    values = numeric.loc[mask]
    if skip_if_already_scaled:
        min_val = float(values.min())
        max_val = float(values.max())
        if min_val >= SCORE_BAND_MIN and max_val <= SCORE_BAND_MAX:
            scaled.loc[mask] = values
            return scaled

    values = values.clip(lower=0, upper=100)
    band_width = SCORE_BAND_MAX - SCORE_BAND_MIN
    scaled.loc[mask] = SCORE_BAND_MIN + (values / 100.0) * band_width
    return scaled


def _scale_score_frame(frame: pd.DataFrame, skip_if_already_scaled: bool = False) -> pd.DataFrame:
    if frame.empty:
        return frame
    scaled_cols = {
        col: _scale_score_series(frame[col], skip_if_already_scaled=skip_if_already_scaled)
        for col in frame.columns
    }
    return pd.DataFrame(scaled_cols, index=frame.index)


def compute_scores_percentiles(raw_scores: pd.DataFrame) -> pd.DataFrame:
    pct = raw_scores.copy()
    for col in pct.columns:
        pct[col] = percentiles_by_group(raw_scores[col], None)
    return pct


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
            assignments.append(candidates[0] if candidates else "")
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


def eligibility_league(df: pd.DataFrame, minutes_possible: pd.Series, frac: float = 0.15) -> pd.Series:
    minutes_played = _coerce_numeric(df.get("minutes_played", pd.Series(index=df.index, dtype=float))).fillna(0)
    threshold = minutes_possible * frac
    threshold = threshold.fillna(np.inf)
    return minutes_played >= threshold


def eligibility_global(df: pd.DataFrame, min_minutes: float = 270) -> pd.Series:
    minutes_played = _coerce_numeric(df.get("minutes_played", pd.Series(index=df.index, dtype=float))).fillna(0)
    return minutes_played >= float(min_minutes)


def _league_group(df: pd.DataFrame) -> pd.Series:
    comp = df.get("competition_name", pd.Series(index=df.index, dtype="object")).fillna("GLOBAL").astype(str)
    cal = df.get("calendar", pd.Series(index=df.index, dtype="object")).fillna("GLOBAL").astype(str)
    return comp + "||" + cal


def _league_strength_from_meta(df: pd.DataFrame) -> pd.Series:
    league_meta_path = Path("/helpers/csv/league_translation_meta.csv")
    if not league_meta_path.exists() or "competition_name" not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    try:
        meta_df = pd.read_csv(league_meta_path)
        if not {"competition", "difficulty"}.issubset(meta_df.columns):
            return pd.Series(np.nan, index=df.index, dtype=float)
        difficulty = pd.to_numeric(meta_df["difficulty"], errors="coerce")
        mean_val = difficulty.mean(skipna=True)
        if not np.isfinite(mean_val) or mean_val == 0:
            return pd.Series(np.nan, index=df.index, dtype=float)
        normalized = (difficulty / mean_val).clip(lower=0.8, upper=1.2)
        strength_map = {
            str(comp): float(val)
            for comp, val in zip(meta_df["competition"], normalized)
            if pd.notna(comp) and np.isfinite(val)
        }
        return df["competition_name"].map(strength_map).astype(float)
    except Exception:
        return pd.Series(np.nan, index=df.index, dtype=float)


def _resolve_league_strength_factors(df: pd.DataFrame) -> pd.Series:
    meta_series = _league_strength_from_meta(df)
    if "league_strength_factor" in df.columns:
        provided = _coerce_numeric(df["league_strength_factor"])
        merged = meta_series.where(meta_series.notna(), provided)
    else:
        merged = meta_series
    return _coerce_numeric(merged).fillna(1.0).clip(lower=0.8, upper=1.2)


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
    min_minutes: float = 270,
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
    empty_cols = [
        "player_a_id",
        "player_a",
        "team_a",
        "competition_name_a",
        "calendar_a",
        "player_b_id",
        "player_b",
        "team_b",
        "competition_name_b",
        "calendar_b",
        "profile",
        "similarity",
    ]

    profile = profiles[profile_name]
    weights = profile.get("weights", {}) or {}
    if not weights:
        return pd.DataFrame(columns=empty_cols)

    weight_sum = sum(weights.values())
    normalized = {metric: (float(weight) / weight_sum) if weight_sum else 0.0 for metric, weight in weights.items()}
    lower_is_better = set(profile.get("lower_is_better", []) or [])

    mask_profile = assigned_role == profile_name
    mask_minutes = eligibility_global(df, min_minutes=270)
    if not mask_profile.any():
        return pd.DataFrame(columns=empty_cols)

    # Keep robust neighbours (>=270 minutes) but include low-minute players as seeds.
    candidate_mask = mask_profile & mask_minutes
    if candidate_mask.sum() < 2:
        candidate_mask = mask_profile

    numeric_vectors = []
    for metric in normalized.keys():
        values = _coerce_numeric(df[metric]).fillna(0.0)
        if metric in lower_is_better:
            values = -values
        weight_scale = math.sqrt(float(normalized[metric])) if normalized[metric] else 0.0
        numeric_vectors.append(values * weight_scale)
    numeric_vectors.append(df.get("foot", pd.Series(index=df.index)).apply(_foot_numeric))
    numeric_vectors.append(_coerce_numeric(df.get("height", pd.Series(index=df.index))))
    numeric_vectors.append(_coerce_numeric(df.get("weight", pd.Series(index=df.index))))
    vectors = pd.concat(numeric_vectors, axis=1)
    vectors = vectors.apply(pd.to_numeric, errors="coerce").loc[mask_profile].fillna(0.0)
    if vectors.empty:
        return pd.DataFrame(columns=empty_cols)

    players_idx = vectors.index.tolist()
    candidate_idx = vectors.index[candidate_mask.loc[vectors.index]].tolist()
    if not candidate_idx:
        return pd.DataFrame(columns=empty_cols)

    seed_values = vectors.to_numpy(dtype=float)
    candidate_values = vectors.loc[candidate_idx].to_numpy(dtype=float)

    seed_norms = np.linalg.norm(seed_values, axis=1)
    seed_norms[seed_norms == 0] = 1e-9
    normalized_seeds = seed_values / seed_norms[:, None]

    candidate_norms = np.linalg.norm(candidate_values, axis=1)
    candidate_norms[candidate_norms == 0] = 1e-9
    normalized_candidates = candidate_values / candidate_norms[:, None]

    similarity_matrix = normalized_seeds @ normalized_candidates.T

    players_id = df.loc[players_idx, "player_id"] if "player_id" in df.columns else pd.Series(index=players_idx, dtype="object")
    players_name = df.loc[players_idx, "player"] if "player" in df.columns else pd.Series(index=players_idx, dtype="object")
    # Use the same club key as fact tables to maximize player_season_id mapping downstream.
    team_col = "team_in_selected_period" if "team_in_selected_period" in df.columns else "team" if "team" in df.columns else None

    records = []
    for i, player_idx in enumerate(players_idx):
        sims = similarity_matrix[i]
        top_indices = np.argsort(sims)[::-1]
        neighbours_added = 0
        seen_indices = set()
        seen_ids = set()
        seen_names = set()
        player_id_a = players_id.loc[player_idx] if not players_id.empty else None
        player_id_a = str(player_id_a).strip() if pd.notna(player_id_a) else None
        if player_id_a in {"nan", "None", "<NA>", ""}:
            player_id_a = None
        player_name_a = players_name.loc[player_idx] if not players_name.empty else ""
        for j in top_indices:
            if neighbours_added >= topk:
                break
            neighbour_idx = candidate_idx[j]
            if neighbour_idx == player_idx:
                continue
            if neighbour_idx in seen_indices:
                continue
            value = sims[j]
            player_id_b = players_id.loc[neighbour_idx] if not players_id.empty else None
            player_id_b = str(player_id_b).strip() if pd.notna(player_id_b) else None
            if player_id_b in {"nan", "None", "<NA>", ""}:
                player_id_b = None
            player_name_b = players_name.loc[neighbour_idx] if not players_name.empty else ""
            if player_id_b is not None and player_id_b in seen_ids:
                continue
            if player_name_b and player_name_b in seen_names:
                continue
            records.append(
                {
                    "player_a_id": player_id_a,
                    "player_a": player_name_a,
                    "team_a": df.at[player_idx, team_col] if team_col else "",
                    "competition_name_a": df.at[player_idx, "competition_name"] if "competition_name" in df.columns else "",
                    "player_b_id": player_id_b,
                    "player_b": player_name_b,
                    "team_b": df.at[neighbour_idx, team_col] if team_col else "",
                    "competition_name_b": df.at[neighbour_idx, "competition_name"] if "competition_name" in df.columns else "",
                    "calendar_a": df.at[player_idx, "calendar"] if "calendar" in df.columns else "",
                    "calendar_b": df.at[neighbour_idx, "calendar"] if "calendar" in df.columns else "",
                    "profile": profile_name,
                    "similarity": float(max(value, 0.0)),
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


def load_profiles(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_profiles_from_env() -> dict:
    path = _load_profiles_path()
    if path.exists():
        return load_profiles(path)
    return {}


def load_transfermarkt_sources(base_dir: Path = Path("/helpers/csv")) -> dict[str, pd.DataFrame]:
    """
    Charge les fichiers Transfermarkt si disponibles dans /helpers/csv.
    Attendu :
      - transfermarkt_profiles.csv
      - club_matching_reference.csv
      - player_matching_reference.csv
      - tm_clubs_reference.csv (optionnel)
      - club_mapping_dict.py (curé)
    """
    sources = {}
    tm_path = base_dir / "transfermarkt_profiles.csv"
    club_map_path = base_dir / "club_matching_reference.csv"
    player_map_path = base_dir / "player_matching_reference.csv"
    tm_clubs_path = base_dir / "tm_clubs_reference.csv"
    club_dict_path = base_dir / "club_mapping_dict.py"
    if tm_path.exists():
        try:
            sources["tm_profiles"] = pd.read_csv(tm_path)
        except Exception:
            print("[WARN] Impossible de lire transfermarkt_profiles.csv")
    if club_map_path.exists():
        try:
            sources["tm_club_map"] = pd.read_csv(club_map_path)
        except Exception:
            print("[WARN] Impossible de lire club_matching_reference.csv")
    if player_map_path.exists():
        try:
            sources["tm_player_map"] = pd.read_csv(player_map_path)
        except Exception:
            print("[WARN] Impossible de lire player_matching_reference.csv")
    if tm_clubs_path.exists():
        try:
            sources["tm_clubs"] = pd.read_csv(tm_clubs_path)
        except Exception:
            print("[WARN] Impossible de lire tm_clubs_reference.csv")
    if club_dict_path.exists():
        try:
            import importlib.util

            spec = importlib.util.spec_from_file_location("club_mapping_dict", club_dict_path)
            module = importlib.util.module_from_spec(spec)
            assert spec.loader
            spec.loader.exec_module(module)
            sources["tm_club_mapping_dict"] = getattr(module, "CLUB_MAPPING", {})
        except Exception:
            print("[WARN] Impossible de charger club_mapping_dict.py")
    return sources


def _normalise_name(value: str | float | None) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip().lower()
    text = (
        text.replace("á", "a")
        .replace("à", "a")
        .replace("â", "a")
        .replace("ä", "a")
        .replace("ã", "a")
        .replace("é", "e")
        .replace("è", "e")
        .replace("ê", "e")
        .replace("ë", "e")
        .replace("í", "i")
        .replace("ì", "i")
        .replace("î", "i")
        .replace("ï", "i")
        .replace("ó", "o")
        .replace("ò", "o")
        .replace("ô", "o")
        .replace("ö", "o")
        .replace("õ", "o")
        .replace("ú", "u")
        .replace("ù", "u")
        .replace("û", "u")
        .replace("ü", "u")
        .replace("ç", "c")
    )
    text = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in text)
    return " ".join(text.split())


def _apply_club_mapping(df: pd.DataFrame, club_mapping: dict) -> pd.Series:
    if not club_mapping:
        return pd.Series(np.nan, index=df.index)
    log_every = int(os.getenv("TM_CLUB_LOG_EVERY", "5000") or "5000")
    print(f"[TM] club map log_every={log_every}", flush=True)
    start = time.time()
    tm_ids = []
    mapped_count = 0
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        team = str(row.get("team_in_selected_period") or row.get("team") or "").strip()
        comp = str(row.get("competition_name") or "").strip()
        key = (team, comp)
        mapping = club_mapping.get(key)
        if mapping:
            tm_id, _tm_name = mapping
            mapped_count += 1
            tm_ids.append(tm_id if tm_id else np.nan)
        else:
            tm_ids.append(np.nan)
        if log_every and i % log_every == 0:
            elapsed = time.time() - start
            rate = i / elapsed if elapsed > 0 else 0
            print(f"[TM] club map progress rows={i} mapped={mapped_count} rate={rate:.1f}/s", flush=True)
    elapsed = time.time() - start
    rate = len(df) / elapsed if elapsed > 0 else 0
    print(f"[TM] club map done rows={len(df)} mapped={mapped_count} rate={rate:.1f}/s", flush=True)
    return pd.Series(tm_ids, index=df.index)


def _safe_age(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        text = str(value)
        match = re.search(r"\((\d{1,2})\)", text)
        if match:
            return float(match.group(1))
        nums = re.findall(r"\d{1,2}", text)
        if nums:
            return float(nums[-1])
    return None


def _fuzzy_match_within_club(club_profiles: pd.DataFrame, target_name: str, target_age: Optional[float]) -> Optional[str]:
    if club_profiles.empty:
        return None
    target_norm = _normalise_name(target_name)
    if not target_norm:
        return None
    target_tokens = target_norm.split()
    target_token_set = set(target_tokens)

    candidates: list[dict[str, object]] = []
    for row in club_profiles.itertuples(index=False):
        candidate_name = getattr(row, "tm_player_name", None)
        candidate_id = getattr(row, "tm_player_id", None)
        if not candidate_name or pd.isna(candidate_name) or pd.isna(candidate_id):
            continue
        candidate_norm = _normalise_name(str(candidate_name))
        if not candidate_norm:
            continue
        candidate_tokens = candidate_norm.split()
        candidate_token_set = set(candidate_tokens)
        overlap = target_token_set.intersection(candidate_token_set)
        sort_score = fuzz.token_sort_ratio(target_norm, candidate_norm)
        set_score = fuzz.token_set_ratio(target_norm, candidate_norm)
        score = max(sort_score, set_score)

        age_diff = None
        target_age_value = _safe_age(target_age)
        tm_age_value = _safe_age(getattr(row, "tm_age", None))
        if target_age_value is not None and tm_age_value is not None:
            age_diff = abs(tm_age_value - target_age_value)
            if age_diff > 3:
                continue

        compatible = False
        if len(overlap) >= 2:
            compatible = score >= 75
        elif len(overlap) == 1:
            target_remaining = [tok for tok in target_tokens if tok not in overlap]
            candidate_remaining = [tok for tok in candidate_tokens if tok not in overlap]
            if target_remaining and candidate_remaining:
                t = target_remaining[0]
                c = candidate_remaining[0]
                initial_ok = (len(t) == 1 and c.startswith(t)) or (len(c) == 1 and t.startswith(c))
                exact_ok = t == c
                if exact_ok or initial_ok:
                    compatible = score >= 75
                else:
                    compatible = score >= 92
            else:
                compatible = score >= 92
            if compatible and age_diff is not None and age_diff > 1:
                compatible = False
        else:
            compatible = score >= 96

        if not compatible:
            continue

        candidates.append(
            {
                "name": str(candidate_name),
                "tm_player_id": str(candidate_id),
                "score": float(score),
                "overlap": len(overlap),
                "age_diff": age_diff if age_diff is not None else 99.0,
            }
        )

    if not candidates:
        return None

    candidates.sort(key=lambda item: (-item["score"], -item["overlap"], item["age_diff"], item["name"]))
    best = candidates[0]
    if len(candidates) > 1:
        second = candidates[1]
        if (best["score"] - second["score"]) <= 3 and best["overlap"] == second["overlap"]:
            return None
    return str(best["tm_player_id"])


def merge_transfermarkt(enriched: pd.DataFrame, players: pd.DataFrame, sources: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge Transfermarkt avec logique proche du script v1 :
      - mapping clubs via club_mapping_dict.py (pas de fuzzy sur clubs)
      - mapping joueurs via player_matching_reference (overrides)
      - fallback fuzzy intra-club (token_sort_ratio) si tm_club_id connu
      - enrichit tm_age / tm_birth_date depuis profile_description si dispo
    """
    if not sources:
        return enriched, players
    tm_profiles = sources.get("tm_profiles")
    tm_player_map = sources.get("tm_player_map")
    club_mapping_dict = sources.get("tm_club_mapping_dict", {})
    if tm_profiles is None:
        return enriched, players
    tm_profiles = tm_profiles.copy()
    tm_profiles.columns = [c.lower() for c in tm_profiles.columns]
    print(f"[TM] tm_profiles rows={len(tm_profiles)} cols={len(tm_profiles.columns)}")

    # parse tm_age / tm_birth_date si profil disponible
    if "profile_description" in tm_profiles.columns:
        ages = []
        birth_dates = []
        for raw in tm_profiles["profile_description"]:
            age, birth = None, None
            if pd.notna(raw):
                text = str(raw)
                age_match = pd.Series(text).str.extract(r",\s*(\d{1,2})\s*,", expand=False).iloc[0]
                if pd.notna(age_match):
                    try:
                        age = int(age_match)
                    except Exception:
                        age = None
                date_match = pd.Series(text).str.extract(r"\*\s*(\d{2}/\d{2}/\d{4})", expand=False).iloc[0]
                if pd.notna(date_match):
                    day, month, year = date_match.split("/")
                    birth = f"{year}-{month}-{day}"
            ages.append(age)
            birth_dates.append(birth)
        tm_profiles["tm_age"] = ages
        tm_profiles["tm_birth_date"] = birth_dates

    # Normalise colonnes tm
    tm_id_col = None
    for candidate in ["tm_player_id", "player_id", "id"]:
        if candidate in tm_profiles.columns:
            tm_id_col = candidate
            break
    if tm_id_col is None:
        print("[WARN] Aucune colonne tm_id identifiée dans transfermarkt_profiles.csv")
        return enriched, players
    tm_profiles = tm_profiles.rename(columns={tm_id_col: "tm_player_id"})
    tm_profiles["tm_player_id"] = (
        tm_profiles["tm_player_id"]
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

    # Override mapping joueurs
    player_map = pd.DataFrame(columns=["wyscout_player_id", "tm_player_id"])
    if tm_player_map is not None:
        tm_player_map = tm_player_map.copy()
        tm_player_map.columns = [c.lower() for c in tm_player_map.columns]
        if "wyscout_player_id" in tm_player_map.columns and "tm_player_id" in tm_player_map.columns:
            player_map = tm_player_map[["wyscout_player_id", "tm_player_id"]].dropna()
            player_map["wyscout_player_id"] = player_map["wyscout_player_id"].astype(str).str.strip()
            player_map["tm_player_id"] = (
                player_map["tm_player_id"]
                .astype("string")
                .str.strip()
                .str.replace(r"\.0$", "", regex=True)
            )
        else:
            print(f"[TM] player_matching_reference columns={tm_player_map.columns.tolist()}")

    enriched = enriched.copy()
    enriched["player_id"] = enriched["player_id"].astype(str).str.strip()

    # mapping clubs (curated dict)
    if club_mapping_dict:
        print(f"[TM] club mapping entries={len(club_mapping_dict)}")
        tm_club_ids = _apply_club_mapping(enriched, club_mapping_dict)
        enriched["tm_club_id"] = (
            tm_club_ids.astype("string")
            .str.strip()
            .str.replace(r"\.0$", "", regex=True)
        )
    else:
        enriched["tm_club_id"] = np.nan
    print(f"[TM] tm_club_id mapped={enriched['tm_club_id'].notna().sum()}")

    # Merge overrides joueurs
    enriched = enriched.merge(player_map, left_on="player_id", right_on="wyscout_player_id", how="left")
    if "tm_player_id" not in enriched.columns:
        enriched["tm_player_id"] = np.nan
    enriched["tm_player_id"] = (
        enriched["tm_player_id"]
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )
    enriched.loc[enriched["tm_player_id"].isin(["nan", "None", "<NA>", ""]), "tm_player_id"] = np.nan
    print(f"[TM] tm_player_id assigned after direct map={enriched['tm_player_id'].notna().sum()}")

    # Alternative mapping via player_matching_reference if no wyscout_player_id is available
    if tm_player_map is not None and "tm_player_id" in tm_player_map.columns:
        map_cols = []
        if "player" in tm_player_map.columns:
            map_cols.append("player")
        if "team" in tm_player_map.columns:
            map_cols.append("team")
        if "competition_name" in tm_player_map.columns:
            map_cols.append("competition_name")
        if "calendar" in tm_player_map.columns:
            map_cols.append("calendar")
        if map_cols:
            map_df = tm_player_map[map_cols + ["tm_player_id"]].dropna()
            map_df["tm_player_id"] = (
                map_df["tm_player_id"]
                .astype("string")
                .str.strip()
                .str.replace(r"\.0$", "", regex=True)
            )
            map_df = map_df.drop_duplicates()
            join_cols = map_cols.copy()
            if "team" in join_cols:
                join_cols[join_cols.index("team")] = "team_in_selected_period"
                map_df = map_df.rename(columns={"team": "team_in_selected_period"})
            print(f"[TM] mapping by columns={join_cols} rows={len(map_df)}")
            enriched = enriched.merge(map_df, on=join_cols, how="left", suffixes=("", "_map"))
            if "tm_player_id_map" in enriched.columns:
                enriched["tm_player_id"] = enriched["tm_player_id"].fillna(enriched["tm_player_id_map"])
                enriched = enriched.drop(columns=["tm_player_id_map"])
    print(f"[TM] tm_player_id assigned after alt map={enriched['tm_player_id'].notna().sum()}")

    # Fuzzy fallback intra club si tm_player_id toujours vide
    enable_fuzzy = os.getenv("TM_ENABLE_FUZZY", "1").lower() not in {"0", "false", "no"}
    if not enable_fuzzy:
        print("[TM] fuzzy matching disabled")
    else:
        def _first_series(df: pd.DataFrame, col: str) -> Optional[pd.Series]:
            if col not in df.columns:
                return None
            series_or_frame = df[col]
            if isinstance(series_or_frame, pd.DataFrame):
                return series_or_frame.iloc[:, 0]
            return series_or_frame

        player_name_series = _first_series(tm_profiles, "player_name")
        if player_name_series is None:
            player_name_series = _first_series(tm_profiles, "name")
        if player_name_series is None:
            player_name_series = _first_series(tm_profiles, "tm_player_name")
        tm_profiles["tm_player_name"] = player_name_series if player_name_series is not None else ""
        tm_profiles["tm_player_name_norm"] = tm_profiles["tm_player_name"].apply(_normalise_name)
        club_series = _first_series(tm_profiles, "club_id")
        if club_series is None:
            club_series = _first_series(tm_profiles, "tm_club_id")
        tm_profiles["tm_club_id"] = club_series if club_series is not None else np.nan
        tm_profiles["tm_club_id"] = (
            tm_profiles["tm_club_id"]
            .astype("string")
            .str.strip()
            .str.replace(r"\.0$", "", regex=True)
        )
        tm_profiles_by_club = {
            club_id: group
            for club_id, group in tm_profiles.dropna(subset=["tm_club_id"]).groupby("tm_club_id")
        }

        missing_mask = enriched["tm_player_id"].isna()
        missing_total = int(missing_mask.sum())
        missing_with_club = int(enriched.loc[missing_mask, "tm_club_id"].notna().sum())
        print(f"[TM] fuzzy candidates with club={missing_with_club}/{missing_total}", flush=True)
        log_every = int(os.getenv("TM_FUZZY_LOG_EVERY", "1000") or "1000")
        start = time.time()
        checked = 0
        matched = 0
        for idx, row in enriched[missing_mask].iterrows():
            tm_club_id = row.get("tm_club_id")
            if pd.isna(tm_club_id):
                continue
            club_profiles = tm_profiles_by_club.get(str(tm_club_id))
            if club_profiles is None:
                continue
            candidate = _fuzzy_match_within_club(club_profiles, row.get("player", ""), row.get("age"))
            checked += 1
            if candidate:
                enriched.at[idx, "tm_player_id"] = candidate
                matched += 1
            if log_every and checked % log_every == 0:
                elapsed = time.time() - start
                rate = checked / elapsed if elapsed > 0 else 0
                print(f"[TM] fuzzy progress checked={checked} matched={matched} rate={rate:.1f}/s", flush=True)
        elapsed = time.time() - start
        rate = checked / elapsed if elapsed > 0 else 0
        print(f"[TM] fuzzy done checked={checked} matched={matched} rate={rate:.1f}/s", flush=True)

    enriched = enriched.merge(tm_profiles, on="tm_player_id", how="left", suffixes=("", "_tm"))
    print(f"[TM] tm_player_id assigned={enriched['tm_player_id'].notna().sum()}")
    if "profile_url" in enriched.columns:
        print(f"[TM] profile_url filled={enriched['profile_url'].notna().sum()}")
    elif "profile_url_tm" in enriched.columns:
        print(f"[TM] profile_url_tm filled={enriched['profile_url_tm'].notna().sum()}")

    tm_source_cols = [col for col in tm_profiles.columns if col != "tm_player_id"]
    for col in tm_source_cols:
        if col.startswith("tm_"):
            continue
        tm_col = f"tm_{col}"
        if tm_col in enriched.columns:
            continue
        if col in enriched.columns:
            enriched[tm_col] = enriched[col]

    players = players.copy()
    players["wyscout_id"] = players["wyscout_id"].astype(str).str.strip()
    players = players.merge(player_map, left_on="wyscout_id", right_on="wyscout_player_id", how="left")
    if "tm_player_id" in players.columns:
        players["tm_player_id"] = players["tm_player_id"].astype(str).str.strip()
        players.loc[players["tm_player_id"].isin(["nan", "None", "<NA>", ""]), "tm_player_id"] = np.nan
    players = players.merge(tm_profiles[["tm_player_id"]], on="tm_player_id", how="left")

    return enriched, players


def clean_players_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    if "player" in working.columns:
        existing_player_id = working["player_id"] if "player_id" in working.columns else None
        parts = working["player"].str.extract(r"(?P<name>.*?)(?:\((?P<id>-?\d+)\))?$")
        if "player_id" in working.columns:
            working = working.drop(columns=["player_id"])
        name_series = parts["name"].astype("string")
        name_series = name_series.str.split(";").str[-1].str.strip()
        working["player"] = name_series
        working.insert(working.columns.get_loc("player") + 1, "player_id", parts["id"].astype("string"))
        if existing_player_id is not None:
            existing_player_id = existing_player_id.astype("string")
            missing = working["player_id"].isna() | (working["player_id"].str.strip() == "")
            working.loc[missing, "player_id"] = existing_player_id[missing]
    if "age" in working.columns:
        pass
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


# --- Pipeline build -----------------------------------------------------------

def _normalize_raw(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace("-", pd.NA)
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype("string").str.strip()
    string_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    if string_cols:
        match_counts = pd.Series(0, index=df.index, dtype="int64")
        for col in string_cols:
            match_counts = match_counts.add(df[col].eq(col).fillna(False).astype("int64"), fill_value=0)
        embedded_headers = match_counts >= 3
        if embedded_headers.any():
            print(f"[PIPELINE] drop embedded header rows={int(embedded_headers.sum())}")
            df = df.loc[~embedded_headers].copy()
    if "age" in df.columns:
        df["age"] = df["age"].apply(_safe_age)
    for col in ("competition_name", "calendar", "team", "team_in_selected_period"):
        if col not in df.columns:
            continue
        series = df[col].astype("string").str.strip()
        if col == "calendar":
            series = series.str.replace(r"\.0$", "", regex=True)
        df[col] = series.replace({"": pd.NA})
    return df


def _canonical_player_name(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _stable_generated_player_id(row: pd.Series, fallback_idx: int) -> str:
    key_cols = [
        "player",
        "birth_date",
        "age",
        "country",
        "calendar",
        "competition_name",
        "team_in_selected_period",
        "team",
    ]
    parts: list[str] = []
    for col in key_cols:
        if col not in row.index:
            continue
        val = row[col]
        if pd.isna(val):
            continue
        sval = str(val).strip()
        if sval:
            parts.append(f"{col}={sval}")
    if not parts:
        parts = [f"row={fallback_idx}"]
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:12]
    return f"gen_{digest}"


def _resolve_player_id_conflicts(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    if "player_id" not in df.columns or "player" not in df.columns:
        return df, 0, 0

    working = df.copy()
    player_ids = working["player_id"].astype("string")
    name_keys = working["player"].apply(_canonical_player_name).astype("string")
    valid = player_ids.notna() & (player_ids.str.strip() != "") & name_keys.notna() & (name_keys.str.strip() != "")
    if not valid.any():
        return working, 0, 0

    conflict_counts = (
        pd.DataFrame({"player_id": player_ids[valid], "name_key": name_keys[valid]})
        .groupby("player_id")["name_key"]
        .nunique()
    )
    conflict_ids = conflict_counts[conflict_counts > 1].index.tolist()
    if not conflict_ids:
        return working, 0, 0

    minutes = (
        _coerce_numeric(working.get("minutes_played", pd.Series(0.0, index=working.index)))
        .reindex(working.index)
        .fillna(0.0)
    )
    detail = pd.DataFrame(
        {
            "player_id": player_ids[valid],
            "name_key": name_keys[valid],
            "minutes": minutes[valid],
        }
    )
    detail = detail[detail["player_id"].isin(conflict_ids)]
    stats = (
        detail.groupby(["player_id", "name_key"], as_index=False)
        .agg(rows=("name_key", "size"), minutes=("minutes", "sum"))
        .sort_values(
            ["player_id", "rows", "minutes", "name_key"],
            ascending=[True, False, False, True],
        )
    )
    keep_name_by_id = stats.drop_duplicates(subset=["player_id"]).set_index("player_id")["name_key"].to_dict()

    remap: dict[tuple[str, str], str] = {}
    for row in stats.itertuples(index=False):
        pid = str(row.player_id)
        name_key = str(row.name_key)
        if keep_name_by_id.get(pid) == name_key:
            continue
        digest = hashlib.sha1(f"{pid}|{name_key}".encode("utf-8")).hexdigest()[:10]
        remap[(pid, name_key)] = f"{pid}__alt_{digest}"

    if not remap:
        return working, len(conflict_ids), 0

    apply_mask = valid & player_ids.isin(conflict_ids)
    apply_index = working.index[apply_mask]
    updated_values = []
    changed_rows = 0
    for idx in apply_index:
        pid = str(player_ids.at[idx])
        name_key = str(name_keys.at[idx])
        new_pid = remap.get((pid, name_key), pid)
        if new_pid != pid:
            changed_rows += 1
        updated_values.append(new_pid)

    working.loc[apply_index, "player_id"] = pd.Series(updated_values, index=apply_index, dtype="string")
    return working, len(conflict_ids), changed_rows


def _ensure_player_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "player_id" not in df.columns:
        df["player_id"] = pd.Series(pd.NA, index=df.index, dtype="string")
    else:
        player_id = df["player_id"].astype("string")
        missing = player_id.isna() | (player_id.str.strip() == "")
        if missing.any():
            missing_index = list(df.index[missing])
            generated = [
                _stable_generated_player_id(df.loc[idx], int(idx) if isinstance(idx, (int, np.integer)) else pos + 1)
                for pos, idx in enumerate(missing_index)
            ]
            df.loc[missing_index, "player_id"] = pd.Series(generated, index=missing_index, dtype="string")

    def _clean_pid(val):
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return pd.NA
        sval = str(val).strip()
        if not sval:
            return pd.NA
        try:
            if re.fullmatch(r"-?\d+(?:\.0+)?", sval):
                return str(int(float(sval)))
        except Exception:  # noqa: BLE001
            pass
        return sval

    df["player_id"] = df["player_id"].apply(_clean_pid).astype("string")
    df, conflict_ids, remapped_rows = _resolve_player_id_conflicts(df)
    if conflict_ids:
        print(f"[PIPELINE] player_id conflicts resolved={conflict_ids} remapped_rows={remapped_rows}")
    return df


def _coerce_fact_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    fact_df = df.copy()
    for col in (
        "minutes_played",
        "matches_played",
        "assigned_role_pct_league",
        "assigned_role_pct_global",
        "global_score_adjusted",
        "league_strength_factor",
    ):
        if col in fact_df.columns:
            fact_df[col] = pd.to_numeric(fact_df[col], errors="coerce")
    return fact_df


def _load_profiles_path() -> Path:
    env_path = os.getenv("PROFILES_PATH")
    if env_path:
        return Path(env_path)
    # Essayer le montage dédié /profiles/player_profiles.json, sinon fallback helpers ou /app
    candidates = [
        Path("/profiles/player_profiles.json"),
        Path("/helpers/csv/player_profiles.json"),
        Path("/app/player_profiles.json"),
    ]
    for cand in candidates:
        if cand.exists() and cand.is_file():
            return cand
    return candidates[-1]


def build_artifacts(df_raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    print("[PIPELINE] normalize + clean players")
    df = _normalize_raw(df_raw)
    print(f"[PIPELINE] input rows={len(df)} cols={len(df.columns)}")
    df = clean_players_dataframe(df)
    df = _ensure_player_id(df)
    if "player_id" in df.columns:
        missing_ids = df["player_id"].isna().sum()
        print(f"[PIPELINE] player_id missing={missing_ids}")
    if "calendar" in df.columns:
        calendar_sample = df["calendar"].dropna().astype(str).head(3).tolist()
        print(f"[PIPELINE] calendar dtype={df['calendar'].dtype} sample={calendar_sample}")
    df = aggregate_player_rows(df)
    print(f"[PIPELINE] rows after dedupe={len(df)}")

    profiles_path = _load_profiles_path()
    profiles = load_profiles(profiles_path) if profiles_path.exists() else {}
    if not profiles:
        print("[WARN] Aucun profil chargé; sorties minimales.")
    print(f"[PIPELINE] profils chargés: {len(profiles)}")

    print("[PIPELINE] split positions")
    df = split_positions_cols(df)
    row_ids = pd.Series(np.arange(len(df)), index=df.index, name="_row_id")

    player_cols = ["player_id", "player"]
    if "tm_player_id" in df.columns:
        player_cols.append("tm_player_id")
    if "tm_profile_url" in df.columns:
        player_cols.append("tm_profile_url")
    players = (
        df[player_cols]
        .drop_duplicates()
        .rename(
            columns={
                "player": "name",
                "player_id": "wyscout_id",
                "tm_player_id": "tm_id",
            }
        )
    )
    if "tm_id" in players.columns:
        players["tm_id"] = players["tm_id"].astype(str).str.strip()

    tm_enriched = None
    skip_tm = os.getenv("TM_SKIP_ENRICH", "0").lower() in {"1", "true", "yes"}
    if skip_tm:
        print("[PIPELINE] skip Transfermarkt enrich (TM_SKIP_ENRICH=1)")
    else:
        print("[PIPELINE] enrich Transfermarkt (si sources présentes)")
        tm_sources = load_transfermarkt_sources()
        tm_base_cols = ["player_id", "player", "competition_name", "calendar"]
        if "team_in_selected_period" in df.columns:
            tm_base_cols.append("team_in_selected_period")
        elif "team" in df.columns:
            tm_base_cols.append("team")
        if "age" in df.columns:
            tm_base_cols.append("age")
        tm_input = df[tm_base_cols].copy()
        tm_input["_row_id"] = row_ids
        tm_enriched, players = merge_transfermarkt(tm_input, players, tm_sources)

    print("[PIPELINE] scores bruts")
    raw_scores = compute_raw_scores(df, profiles)
    scores_pct = compute_scores_percentiles(raw_scores)

    print("[PIPELINE] assignation rôle")
    assigned_role = assign_role(df, scores_pct, profiles)
    scores_pct = _scale_score_frame(scores_pct)

    print("[PIPELINE] éligibilités")
    minutes_possible = estimate_minutes_possible(df)
    elig_league = eligibility_league(df, minutes_possible, frac=0.15)
    df["_elig_league"] = elig_league
    elig_global = eligibility_global(df, min_minutes=270)

    print("[PIPELINE] percentiles rôle ligue/global")
    roles_scores_league = roles_league_percentiles(df, raw_scores, assigned_role, profiles)
    roles_scores_global = roles_global_percentiles(df, raw_scores, assigned_role, profiles, min_minutes=270)
    roles_scores_league = _scale_score_frame(roles_scores_league)
    roles_scores_global = _scale_score_frame(roles_scores_global)

    print("[PIPELINE] similarités")
    sim_topk = int(os.getenv("SIM_TOPK", "30") or "30")
    similarity_frames = []
    for profile_name in profiles.keys():
        sim_df = profile_similarity(df, profiles, assigned_role, profile_name, topk=sim_topk)
        if not sim_df.empty:
            similarity_frames.append(sim_df)
    similarity_df = pd.concat(similarity_frames, ignore_index=True) if similarity_frames else pd.DataFrame(columns=[
        "player_a",
        "team_a",
        "competition_name_a",
        "player_b",
        "team_b",
        "competition_name_b",
        "profile",
        "similarity",
    ])
    print(f"[PIPELINE] similarity rows={len(similarity_df)}")

    # Dimensions (avant TM pour réutiliser team_col/fact)
    competitions = (
        df[["competition_name"]]
        .dropna()
        .drop_duplicates()
        .rename(columns={"competition_name": "name"})
    )
    seasons = (
        df[["calendar"]]
        .dropna()
        .drop_duplicates()
        .rename(columns={"calendar": "label"})
    )
    team_col = None
    if "team_in_selected_period" in df.columns:
        team_col = "team_in_selected_period"
    elif "team" in df.columns:
        team_col = "team"
    if team_col:
        clubs = (
            df[[team_col, "competition_name"]]
            .dropna()
            .drop_duplicates()
            .rename(columns={team_col: "name", "competition_name": "competition_name"})
        )
    else:
        clubs = pd.DataFrame(columns=["name", "competition_name"])
    print(
        "[PIPELINE] dims counts:"
        f" competitions={len(competitions)} seasons={len(seasons)}"
        f" players={len(players)} clubs={len(clubs)}"
    )

    print("[PIPELINE] percentiles métriques + summary")
    league_group = _league_group(df)
    metrics_base = df.drop(columns=["_elig_league"], errors="ignore")
    metrics_league_pct, metrics_global_pct = compute_metric_percentiles(metrics_base, league_group)
    scores_league_pct, scores_global_pct = compute_metric_percentiles(raw_scores, league_group)
    scores_league_pct = _scale_score_frame(scores_league_pct)
    scores_global_pct = _scale_score_frame(scores_global_pct)
    summary_scores = compute_summary_scores(metrics_global_pct)

    factor_series = _resolve_league_strength_factors(df)
    df["league_strength_factor"] = factor_series
    strength_non_default = int((factor_series != 1.0).sum())
    print(f"[PIPELINE] league strength factors non-default={strength_non_default}/{len(factor_series)}")

    baseline_strength = factor_series.mean(skipna=True)
    if not np.isfinite(baseline_strength) or baseline_strength == 0:
        baseline_strength = 1.0
    ratio = (factor_series / baseline_strength).clip(lower=0.8, upper=1.2)
    blended_multiplier = np.power(ratio, 0.5)

    global_score_adjusted_series = pd.Series(np.nan, index=df.index, dtype=float)
    for profile_name in profiles.keys():
        mask = (assigned_role == profile_name) & elig_global
        if not mask.any():
            continue
        adj_raw = raw_scores.loc[mask, profile_name] * blended_multiplier.loc[mask]
        valid_count = adj_raw.notna().sum()
        if valid_count == 0:
            continue
        if valid_count == 1:
            pct_vals = pd.Series(100.0, index=adj_raw.index)
        else:
            ranks = adj_raw.rank(method="first", ascending=False, na_option="keep")
            pct_vals = (valid_count - ranks) / (valid_count - 1) * 100
        global_score_adjusted_series.loc[mask] = pct_vals.clip(lower=0, upper=100)
    global_score_adjusted_series = _scale_score_series(global_score_adjusted_series)

    role_league_pct = pd.Series(np.nan, index=df.index, dtype=float)
    role_global_pct = pd.Series(np.nan, index=df.index, dtype=float)
    for profile_name in profiles.keys():
        mask = assigned_role == profile_name
        if mask.any():
            if profile_name in roles_scores_league:
                role_league_pct.loc[mask] = roles_scores_league.loc[mask, profile_name]
            # Keep a single source of truth: assigned role global pct mirrors adjusted global score.
            role_global_pct.loc[mask] = global_score_adjusted_series.loc[mask]

    enriched_extra = pd.concat(
        [
            scores_pct,
            metrics_league_pct,
            metrics_global_pct,
            scores_league_pct,
            scores_global_pct,
            roles_scores_league.reindex(df.index),
            roles_scores_global.reindex(df.index),
            role_league_pct.rename("assigned_role_pct_league"),
            role_global_pct.rename("assigned_role_pct_global"),
            global_score_adjusted_series.rename("global_score_adjusted"),
            summary_scores,
        ],
        axis=1,
    )
    enriched = df.copy()
    enriched = enriched.drop(columns=["_elig_league"], errors="ignore")
    enriched["assigned_role"] = assigned_role
    enriched = pd.concat([enriched, enriched_extra], axis=1)
    if tm_enriched is not None and "_row_id" in tm_enriched.columns:
        tm_enriched = (
            tm_enriched.drop_duplicates(subset=["_row_id"], keep="first")
            .set_index("_row_id")
        )
        tm_cols = [col for col in tm_enriched.columns if col.startswith("tm_") or col in ("profile_url", "profile_url_tm")]
        row_index = row_ids.to_numpy()
        for col in tm_cols:
            if col in enriched.columns:
                continue
            enriched[col] = tm_enriched[col].reindex(row_index).to_numpy()
        players = _build_players_from_enriched(enriched, players)

    if enriched.columns.duplicated().any():
        dupes = enriched.columns[enriched.columns.duplicated()].tolist()
        print(f"[WARN] duplicate columns in enriched: {dupes}")
        enriched = enriched.loc[:, ~enriched.columns.duplicated()]

    # Fact player_seasons (one row per player/comp/calendar/team)
    group_cols = ["wyscout_id", "competition_name", "calendar"]
    if team_col:
        group_cols.append(team_col)
    fact_aggs = {
        "minutes_played": ("minutes_played", "max") if "minutes_played" in enriched.columns else ("player_id", "size"),
        "matches_played": ("matches_played", "max") if "matches_played" in enriched.columns else ("player_id", "size"),
        "assigned_role": ("assigned_role", "first") if "assigned_role" in enriched.columns else ("player_id", "first"),
        "assigned_role_pct_league": ("assigned_role_pct_league", "first") if "assigned_role_pct_league" in enriched.columns else ("player_id", "size"),
        "assigned_role_pct_global": ("assigned_role_pct_global", "max") if "assigned_role_pct_global" in enriched.columns else ("player_id", "size"),
        "global_score_adjusted": ("global_score_adjusted", "max") if "global_score_adjusted" in enriched.columns else ("player_id", "size"),
        "position": ("position", "first"),
        "second_position": ("second_position", "first"),
        "league_strength_factor": ("league_strength_factor", "first") if "league_strength_factor" in enriched.columns else ("player_id", "size"),
    }
    tm_cols = [col for col in enriched.columns if col.startswith("tm_")]
    for col in tm_cols:
        if col not in fact_aggs:
            fact_aggs[col] = (col, "first")
    fact_source = _coerce_fact_numeric_columns(enriched)
    fact = (
        fact_source.assign(wyscout_id=fact_source["player_id"])
        .groupby(group_cols, dropna=False)
        .agg(**fact_aggs)
        .reset_index()
    )
    fact.rename(columns={team_col: "team_in_selected_period"}, inplace=True)
    print(f"[PIPELINE] player_seasons rows={len(fact)}")

    # Player metrics wide (numeric only, from enriched)
    metrics_group_cols = ["wyscout_id", "competition_name", "calendar"]
    if team_col:
        metrics_group_cols.append(team_col)
    metrics_source = enriched.copy()
    skip_cols = set(metrics_group_cols)
    for col in metrics_source.columns:
        if col in skip_cols:
            continue
        metrics_source[col] = pd.to_numeric(metrics_source[col], errors="coerce")
    numeric_cols = metrics_source.select_dtypes(include=["number"]).columns
    metrics_cols = [c for c in numeric_cols if c not in skip_cols and c != "player_id"]
    metrics = (
        metrics_source.assign(wyscout_id=metrics_source["player_id"])
        .groupby(metrics_group_cols, dropna=False)[metrics_cols]
        .max()
        .reset_index()
    )
    print(f"[PIPELINE] player_metrics rows={len(metrics)} cols={len(metrics_cols)}")
    if not metrics_cols:
        print("[WARN] no numeric metrics extracted for player_metrics")
    if team_col and team_col != "team_in_selected_period":
        metrics.rename(columns={team_col: "team_in_selected_period"}, inplace=True)

    # Role scores long
    role_records = []
    for idx, row in raw_scores.iterrows():
        for profile_name in profiles.keys():
            raw_val = row.get(profile_name)
            pct_league_col = f"{profile_name}_pct_league"
            pct_global_col = f"{profile_name}_pct_global"
            if pct_league_col in scores_league_pct.columns:
                pct_league = scores_league_pct.loc[idx, pct_league_col]
            elif profile_name in scores_league_pct.columns:
                # Backward-compatible fallback for legacy column naming.
                pct_league = scores_league_pct.loc[idx, profile_name]
            else:
                pct_league = np.nan
            if pct_global_col in scores_global_pct.columns:
                pct_global = scores_global_pct.loc[idx, pct_global_col]
            elif profile_name in scores_global_pct.columns:
                # Backward-compatible fallback for legacy column naming.
                pct_global = scores_global_pct.loc[idx, profile_name]
            else:
                pct_global = np.nan
            pct_global_adjusted = (
                global_score_adjusted_series.loc[idx] if assigned_role.loc[idx] == profile_name else np.nan
            )
            role_records.append(
                {
                    "wyscout_id": df.at[idx, "player_id"],
                    "competition_name": df.at[idx, "competition_name"],
                    "calendar": df.at[idx, "calendar"],
                    "team_in_selected_period": df.at[idx, "team_in_selected_period"] if "team_in_selected_period" in df.columns else df.at[idx, "team"] if "team" in df.columns else None,
                    "profile": profile_name,
                    "raw_score": raw_val,
                    "pct_league": pct_league,
                    "pct_global": pct_global,
                    "pct_global_adjusted": pct_global_adjusted,
                }
            )
    role_scores = pd.DataFrame(role_records)

    # Player similarity mapping
    if similarity_df.empty:
        similarity_mapped = similarity_df
    else:
        if "player_a_id" in similarity_df.columns and "player_b_id" in similarity_df.columns:
            similarity_df["player_a_id"] = similarity_df["player_a_id"].astype("string").str.strip()
            similarity_df["player_b_id"] = similarity_df["player_b_id"].astype("string").str.strip()
            similarity_df.loc[similarity_df["player_a_id"].isin(["nan", "None", "<NA>", ""]), "player_a_id"] = np.nan
            similarity_df.loc[similarity_df["player_b_id"].isin(["nan", "None", "<NA>", ""]), "player_b_id"] = np.nan
            mapped = (similarity_df["player_a_id"].notna() & similarity_df["player_b_id"].notna()).sum()
            print(f"[PIPELINE] similarity mapped rows (native ids)={mapped}/{len(similarity_df)}")
        else:
            name_map = players.drop_duplicates(subset=["name"]).set_index("name")["wyscout_id"]
            similarity_df["player_a_id"] = similarity_df["player_a"].map(name_map)
            similarity_df["player_b_id"] = similarity_df["player_b"].map(name_map)
            mapped = (similarity_df["player_a_id"].notna() & similarity_df["player_b_id"].notna()).sum()
            print(f"[PIPELINE] similarity mapped rows (name fallback)={mapped}/{len(similarity_df)}")
        similarity_df.rename(columns={"competition_name_a": "competition_a", "competition_name_b": "competition_b"}, inplace=True)
        similarity_mapped = similarity_df

    return {
        "raw": df_raw,
        "enriched": enriched,
        "players_scores": scores_pct,
        "competitions": competitions,
        "seasons": seasons,
        "players": players,
        "clubs": clubs,
        "player_seasons": fact,
        "player_metrics": metrics,
        "role_scores": role_scores,
        "player_similarity": similarity_mapped,
    }


def _resolve_team_column(df: pd.DataFrame) -> Optional[str]:
    if "team_in_selected_period" in df.columns:
        return "team_in_selected_period"
    if "team" in df.columns:
        return "team"
    return None


def _aggregate_for_roles(df: pd.DataFrame, group_cols: list[str], role_cols: list[str], extra_cols: list[str]) -> pd.DataFrame:
    subset_cols = group_cols + role_cols + extra_cols
    subset_cols = [c for c in subset_cols if c in df.columns]
    subset = df[subset_cols].copy()
    numeric_cols = subset.select_dtypes(include=["number"]).columns.tolist()
    agg_map = {}
    for col in subset_cols:
        if col in group_cols:
            continue
        agg_map[col] = "max" if col in numeric_cols else "first"
    grouped = subset.groupby(group_cols, dropna=False).agg(agg_map).reset_index()
    return grouped


def _build_players_from_enriched(enriched: pd.DataFrame, fallback: pd.DataFrame) -> pd.DataFrame:
    if "player_id" not in enriched.columns:
        return fallback
    tm_id_col = "tm_player_id" if "tm_player_id" in enriched.columns else "tm_id"
    profile_url_col = None
    for candidate in ("profile_url", "profile_url_tm", "tm_profile_url"):
        if candidate in enriched.columns:
            profile_url_col = candidate
            break
    data = {
        "player_id": enriched["player_id"],
        "player": enriched.get("player", pd.Series(pd.NA, index=enriched.index)),
        "tm_player_id": enriched.get(tm_id_col, pd.Series(pd.NA, index=enriched.index)),
        "profile_url": enriched.get(profile_url_col, pd.Series(pd.NA, index=enriched.index)),
    }
    players = pd.DataFrame(data).drop_duplicates()
    players = players.rename(
        columns={
            "player_id": "wyscout_id",
            "player": "name",
            "tm_player_id": "tm_id",
            "profile_url": "tm_profile_url",
        }
    )
    return players


def _build_role_scores_from_enriched(
    df: pd.DataFrame,
    profiles: Mapping[str, Mapping[str, object]],
    team_col: Optional[str],
) -> pd.DataFrame:
    profile_names = list(profiles.keys())
    if not profile_names:
        return pd.DataFrame(columns=["wyscout_id", "competition_name", "calendar", "team_in_selected_period", "profile", "raw_score", "pct_league", "pct_global", "pct_global_adjusted"])

    role_cols = []
    for profile in profile_names:
        for suffix in ("", "_pct_league", "_pct_global"):
            col = f"{profile}{suffix}"
            if col in df.columns:
                role_cols.append(col)

    extra_cols = ["assigned_role", "global_score_adjusted"]
    group_cols = ["player_id", "competition_name", "calendar"]
    if team_col:
        group_cols.append(team_col)

    grouped = _aggregate_for_roles(df, group_cols, role_cols, extra_cols)
    grouped = grouped.rename(columns={"player_id": "wyscout_id"})
    if team_col and team_col != "team_in_selected_period":
        grouped = grouped.rename(columns={team_col: "team_in_selected_period"})

    records = []
    for _, row in grouped.iterrows():
        for profile in profile_names:
            pct_league = row.get(f"{profile}_pct_league")
            pct_global = row.get(f"{profile}_pct_global")
            if pd.isna(pct_global):
                pct_global = row.get(profile)
            assigned_role = row.get("assigned_role")
            pct_global_adjusted = row.get("global_score_adjusted") if pd.notna(assigned_role) and assigned_role == profile else np.nan
            if pd.isna(pct_league) and pd.isna(pct_global) and pd.isna(pct_global_adjusted):
                continue
            records.append(
                {
                    "wyscout_id": row.get("wyscout_id"),
                    "competition_name": row.get("competition_name"),
                    "calendar": row.get("calendar"),
                    "team_in_selected_period": row.get("team_in_selected_period"),
                    "profile": profile,
                    "raw_score": np.nan,
                    "pct_league": pct_league,
                    "pct_global": pct_global,
                    "pct_global_adjusted": pct_global_adjusted,
                }
            )
    return pd.DataFrame(records)


def _map_similarity_ids(similarity: pd.DataFrame, enriched: pd.DataFrame, team_col: Optional[str]) -> pd.DataFrame:
    if similarity is None or similarity.empty:
        return pd.DataFrame(columns=[
            "player_a",
            "team_a",
            "competition_a",
            "player_b",
            "team_b",
            "competition_b",
            "calendar_a",
            "calendar_b",
            "profile",
            "similarity",
        ])

    sim = similarity.copy()
    if "competition_name_a" in sim.columns and "competition_a" not in sim.columns:
        sim = sim.rename(columns={"competition_name_a": "competition_a"})
    if "competition_name_b" in sim.columns and "competition_b" not in sim.columns:
        sim = sim.rename(columns={"competition_name_b": "competition_b"})

    key_cols = ["player_id", "player", "competition_name", "calendar"]
    if team_col:
        key_cols.append(team_col)
    lookup = enriched[key_cols].dropna(subset=["player_id"]).drop_duplicates()

    lookup_a = lookup.rename(
        columns={
            "player_id": "player_a_id",
            "player": "player_a_name",
            "competition_name": "competition_a_name",
            "calendar": "calendar_a",
            team_col or "team": "team_a_name",
        }
    )
    left_on = ["player_a", "competition_a"]
    right_on = ["player_a_name", "competition_a_name"]
    if team_col:
        left_on.insert(1, "team_a")
        right_on.insert(1, "team_a_name")
    sim = sim.merge(lookup_a, left_on=left_on, right_on=right_on, how="left")

    lookup_b = lookup.rename(
        columns={
            "player_id": "player_b_id",
            "player": "player_b_name",
            "competition_name": "competition_b_name",
            "calendar": "calendar_b",
            team_col or "team": "team_b_name",
        }
    )
    left_on = ["player_b", "competition_b"]
    right_on = ["player_b_name", "competition_b_name"]
    if team_col:
        left_on.insert(1, "team_b")
        right_on.insert(1, "team_b_name")
    sim = sim.merge(lookup_b, left_on=left_on, right_on=right_on, how="left")

    drop_cols = [
        "player_a_name",
        "competition_a_name",
        "team_a_name" if team_col else None,
        "player_b_name",
        "competition_b_name",
        "team_b_name" if team_col else None,
    ]
    drop_cols = [c for c in drop_cols if c and c in sim.columns]
    if drop_cols:
        sim = sim.drop(columns=drop_cols, errors="ignore")

    return sim


def build_artifacts_from_enriched(
    df_raw: pd.DataFrame,
    similarity_df: Optional[pd.DataFrame] = None,
) -> dict[str, pd.DataFrame]:
    print("[PIPELINE] normalize enriched dataset")
    df = _normalize_raw(df_raw)
    df = _ensure_player_id(df)
    df["wyscout_id"] = df["player_id"]
    if "second_position" not in df.columns:
        df = split_positions_cols(df)

    profiles = load_profiles_from_env()
    print(f"[PIPELINE] profils chargés: {len(profiles)}")
    df["league_strength_factor"] = _resolve_league_strength_factors(df)

    score_columns = {
        "assigned_role_pct_league",
        "assigned_role_pct_global",
        "global_score_adjusted",
    }
    for profile_name in profiles.keys():
        score_columns.add(profile_name)
        score_columns.add(f"{profile_name}_pct_league")
        score_columns.add(f"{profile_name}_pct_global")
    for column in score_columns:
        if column in df.columns:
            df[column] = _scale_score_series(df[column], skip_if_already_scaled=True)
    if "global_score_adjusted" in df.columns and "assigned_role_pct_global" in df.columns:
        mask = _coerce_numeric(df["global_score_adjusted"]).notna()
        df.loc[mask, "assigned_role_pct_global"] = df.loc[mask, "global_score_adjusted"]

    team_col = _resolve_team_column(df)
    if team_col:
        clubs = (
            df[[team_col, "competition_name"]]
            .dropna()
            .drop_duplicates()
            .rename(columns={team_col: "name", "competition_name": "competition_name"})
        )
    else:
        clubs = pd.DataFrame(columns=["name", "competition_name"])

    competitions = (
        df[["competition_name"]]
        .dropna()
        .drop_duplicates()
        .rename(columns={"competition_name": "name"})
    )
    seasons = (
        df[["calendar"]]
        .dropna()
        .drop_duplicates()
        .rename(columns={"calendar": "label"})
    )
    players = (
        df[["player_id", "player"]]
        .drop_duplicates()
        .rename(columns={"player": "name", "player_id": "wyscout_id"})
    )

    group_cols = ["wyscout_id", "competition_name", "calendar"]
    if team_col:
        group_cols.append(team_col)

    agg_map = {
        "minutes_played": ("minutes_played", "max") if "minutes_played" in df.columns else ("player_id", "size"),
        "matches_played": ("matches_played", "max") if "matches_played" in df.columns else ("player_id", "size"),
        "assigned_role": ("assigned_role", "first") if "assigned_role" in df.columns else ("player_id", "first"),
        "assigned_role_pct_league": ("assigned_role_pct_league", "first") if "assigned_role_pct_league" in df.columns else ("player_id", "size"),
        "assigned_role_pct_global": ("assigned_role_pct_global", "max") if "assigned_role_pct_global" in df.columns else ("player_id", "size"),
        "global_score_adjusted": ("global_score_adjusted", "max") if "global_score_adjusted" in df.columns else ("player_id", "size"),
        "position": ("position", "first"),
        "second_position": ("second_position", "first"),
        "league_strength_factor": ("league_strength_factor", "first") if "league_strength_factor" in df.columns else ("player_id", "size"),
    }

    tm_cols = [c for c in df.columns if c.startswith("tm_")]
    for col in tm_cols:
        agg_map[col] = (col, "first")

    fact_source = _coerce_fact_numeric_columns(df)
    fact = fact_source.groupby(group_cols, dropna=False).agg(**agg_map).reset_index()
    if team_col and team_col != "team_in_selected_period":
        fact = fact.rename(columns={team_col: "team_in_selected_period"})

    numeric_cols = df.select_dtypes(include=["number"]).columns
    metrics_cols = [c for c in numeric_cols if c != "player_id"]
    metrics_group_cols = ["wyscout_id", "competition_name", "calendar"]
    if team_col:
        metrics_group_cols.append(team_col)

    metrics_source = df.copy()
    skip_cols = set(metrics_group_cols)
    for col in metrics_source.columns:
        if col in skip_cols:
            continue
        metrics_source[col] = pd.to_numeric(metrics_source[col], errors="coerce")

    numeric_cols = metrics_source.select_dtypes(include=["number"]).columns
    metrics_cols = [c for c in numeric_cols if c not in skip_cols]
    metrics = (
        metrics_source.groupby(metrics_group_cols, dropna=False)[metrics_cols]
        .max()
        .reset_index()
    )

    role_scores = _build_role_scores_from_enriched(df, profiles, team_col)
    similarity_mapped = _map_similarity_ids(similarity_df, df, team_col)

    return {
        "raw": df_raw,
        "enriched": df,
        "competitions": competitions,
        "seasons": seasons,
        "players": players,
        "clubs": clubs,
        "player_seasons": fact,
        "player_metrics": metrics,
        "role_scores": role_scores,
        "player_similarity": similarity_mapped,
    }
