"""NextLegend roles and similarity pipeline.

This script enriches the Wyscout players dataset with profile scores, percentiles,
role assignment, and similarity exports as defined in codex/Algo.MD.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT_DIR / "data" / "wyscout_players_cleaned.csv"
DEFAULT_PROFILES = ROOT_DIR / "player_profiles.json"
DEFAULT_OUT_ENRICHED = ROOT_DIR / "data" / "wyscout_players_cleaned.csv"
DEFAULT_OUT_SCORES = ROOT_DIR / "players_scores.csv"
DEFAULT_OUT_LEAGUE = ROOT_DIR / "roles_scores_league.csv"
DEFAULT_OUT_GLOBAL = ROOT_DIR / "roles_scores_global.csv"
SIMILARITY_DIR = ROOT_DIR / "data" / "similarity"

_ZSCORE_CACHE: Dict[str, pd.Series] = {}


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

    # Additional contextual features
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

    records = []
    for row_pos, player_idx in enumerate(indices):
        row = similarities[row_pos]
        if not np.isfinite(row).any():
            continue
        top = np.argpartition(-row, kth=min(topk, len(row) - 1))[:topk]
        top = top[np.argsort(-row[top])]
        for col_pos in top:
            value = row[col_pos]
            if not np.isfinite(value) or value <= -np.inf:
                continue
            neighbour_idx = indices[col_pos]
            records.append(
                {
                    "player_a": df.at[player_idx, "player"] if "player" in df.columns else player_idx,
                    "team_a": df.at[player_idx, "team"] if "team" in df.columns else "",
                    "competition_name_a": df.at[player_idx, "competition_name"] if "competition_name" in df.columns else "",
                    "player_b": df.at[neighbour_idx, "player"] if "player" in df.columns else neighbour_idx,
                    "team_b": df.at[neighbour_idx, "team"] if "team" in df.columns else "",
                    "competition_name_b": df.at[neighbour_idx, "competition_name"] if "competition_name" in df.columns else "",
                    "profile": profile_name,
                    "similarity": float(value),
                }
            )
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
    parser.add_argument("--in", dest="input_path", default=DEFAULT_INPUT, type=Path)
    parser.add_argument("--profiles", dest="profiles_path", default=DEFAULT_PROFILES, type=Path)
    parser.add_argument("--out_enriched", dest="out_enriched", default=DEFAULT_OUT_ENRICHED, type=Path)
    parser.add_argument("--out_scores", dest="out_scores", default=DEFAULT_OUT_SCORES, type=Path)
    parser.add_argument("--out_league", dest="out_league", default=DEFAULT_OUT_LEAGUE, type=Path)
    parser.add_argument("--out_global", dest="out_global", default=DEFAULT_OUT_GLOBAL, type=Path)
    parser.add_argument("--sim_topk", dest="sim_topk", default=10, type=int)
    args = parser.parse_args()

    print("LOAD")
    df = pd.read_csv(args.input_path)
    profiles = load_profiles(args.profiles_path)

    print("SPLIT")
    df = split_positions_cols(df)

    print("SCORES_RAW")
    raw_scores = compute_raw_scores(df, profiles)
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
    SIMILARITY_DIR.mkdir(parents=True, exist_ok=True)
    for profile_name in profiles.keys():
        sim_df = profile_similarity(df, profiles, assigned_role, profile_name, topk=args.sim_topk)
        sim_path = SIMILARITY_DIR / f"similarity_{slugify_profile(profile_name)}.csv"
        sim_df.to_csv(sim_path, index=False)

    print("WRITE")
    args.out_scores.parent.mkdir(parents=True, exist_ok=True)
    args.out_enriched.parent.mkdir(parents=True, exist_ok=True)
    args.out_league.parent.mkdir(parents=True, exist_ok=True)
    args.out_global.parent.mkdir(parents=True, exist_ok=True)

    scores_pct_out = scores_pct.copy()
    if "player" in df.columns:
        scores_pct_out.insert(0, "player", df["player"])
    scores_pct_out.to_csv(args.out_scores, index=False)

    league_group = _league_group(df)
    metrics_base = df.drop(columns=["_elig_league"], errors="ignore")
    metrics_league_pct, metrics_global_pct = compute_metric_percentiles(metrics_base, league_group)
    scores_league_pct, scores_global_pct = compute_metric_percentiles(raw_scores, league_group)

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
        ],
        axis=1,
    )
    overlapping = set(enriched.columns).intersection(enriched_extra.columns)
    enriched_extra = enriched_extra.drop(columns=list(overlapping), errors="ignore")
    enriched = pd.concat([enriched, enriched_extra], axis=1)
    enriched.to_csv(args.out_enriched, index=False)

    roles_scores_league_out = pd.concat(
        [
            df.get("player", pd.Series(index=df.index)),
            assigned_role.rename("assigned_role"),
            roles_scores_league,
        ],
        axis=1,
    )
    roles_scores_league_out = roles_scores_league_out.dropna(subset=list(profiles.keys()), how="all")
    roles_scores_league_out.to_csv(args.out_league, index=False)

    roles_scores_global_out = pd.concat(
        [
            df.get("player", pd.Series(index=df.index)),
            assigned_role.rename("assigned_role"),
            roles_scores_global,
        ],
        axis=1,
    )
    roles_scores_global_out = roles_scores_global_out.dropna(subset=list(profiles.keys()), how="all")
    roles_scores_global_out.to_csv(args.out_global, index=False)

    print("DONE")


if __name__ == "__main__":
    _run_tests()
    main()
