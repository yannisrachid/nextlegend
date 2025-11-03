"""Core analytical utilities for NextLegend."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

DEFAULT_GROUPBY = ("position",)


FEATURE_GROUPS: dict[str, list[str]] = {
    "Finition": [
        "goals_per_90",
        "shots_per_90",
        "shots_on_target_percent",
        "goal_conversion_rate",
        "xg_per_90",
    ],
    "Création": [
        "assists_per_90",
        "xa_per_90",
        "key_passes_per_90",
        "smart_passes_per_90",
        "passes_to_penalty_area_per_90",
        "deep_completions_per_90",
    ],
    "Offensif": [
        "dribbles_per_90",
        "successful_dribbles_percent",
        "progressive_runs_per_90",
        "accelerations_per_90",
        "touches_in_penalty_area_per_90",
    ],
    "Construction": [
        "passes_per_90",
        "progressive_passes_per_90",
        "passes_to_final_third_per_90",
        "through_passes_per_90",
        "shot_assists_per_90",
    ],
    "Précision": [
        "accurate_passes_percent",
        "accurate_forward_passes_percent",
        "accurate_backward_passes_percent",
        "accurate_lateral_passes_percent",
        "accurate_long_passes_percent",
    ],
    "Défense": [
        "successful_def_actions_per_90",
        "def_duels_per_90",
        "def_duels_won_percent",
        "interceptions_per_90",
        "sliding_tackles_per_90",
        "blocked_shots_per_90",
    ],
    "Aérien": [
        "aerial_duels_per_90",
        "aerial_duels_won_percent",
        "headed_goals_per_90",
    ],
}


CATEGORY_PRIORITIES = list(FEATURE_GROUPS.keys())
FEATURE_SET = tuple({metric for group in FEATURE_GROUPS.values() for metric in group})

logger = logging.getLogger("nextlegend.analytics")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


@dataclass
class PercentileResult:
    frame: pd.DataFrame
    long: pd.DataFrame


def _available_metrics(df: pd.DataFrame, metrics: Iterable[str]) -> list[str]:
    return [metric for metric in metrics if metric in df.columns]


def compute_percentiles(
    df: pd.DataFrame,
    metrics: Sequence[str],
    groupby: Sequence[str] = DEFAULT_GROUPBY,
) -> PercentileResult:
    logger.info("Computing percentiles on %s records", len(df))
    working = df.copy()
    group_cols = [col for col in groupby if col in working.columns]
    if not group_cols:
        group_cols = None

    percentile_columns: dict[str, str] = {}
    for metric in metrics:
        if metric not in working.columns:
            continue
        target = f"{metric}_pct"
        percentile_columns[metric] = target
        series = working[metric]
        if group_cols:
            ranks = (
                working.groupby(group_cols)[metric]
                .transform(lambda s: s.rank(pct=True, method="average"))
                * 100
            )
        else:
            ranks = series.rank(pct=True, method="average") * 100
        working[target] = ranks.clip(0, 100)

    long_rows = []
    for category, category_metrics in FEATURE_GROUPS.items():
        for metric in _available_metrics(working, category_metrics):
            pct_col = percentile_columns.get(metric)
            if not pct_col:
                continue
            long_rows.append(
                working[["player_id", "player_name", metric, pct_col]]
                .assign(metric=metric, category=category)
                .rename(columns={pct_col: "percentile", metric: "value"})
            )

    long_df = (
        pd.concat(long_rows, ignore_index=True)
        if long_rows
        else pd.DataFrame(columns=["player_id", "player_name", "metric", "category", "value", "percentile"])
    )
    logger.info("Percentiles computed for %s metrics", len(percentile_columns))
    return PercentileResult(frame=working, long=long_df)


def performance_index(
    percentiles: pd.DataFrame,
    weights: Mapping[str, float],
) -> pd.Series:
    adjusted_weights = {cat: weights.get(cat, 0.0) for cat in CATEGORY_PRIORITIES}
    weight_sum = sum(adjusted_weights.values()) or 1.0

    score = np.zeros(len(percentiles))
    for category, metrics in FEATURE_GROUPS.items():
        weight = adjusted_weights.get(category, 0.0) / weight_sum
        if weight <= 0:
            continue
        available_metrics = _available_metrics(percentiles, [f"{metric}_pct" for metric in metrics])
        if not available_metrics:
            continue
        category_centile = percentiles[available_metrics].mean(axis=1)
        score += category_centile * weight
    return pd.Series(score, index=percentiles.index, name="nextlegend_index")


def summarise_player_profile(
    df: pd.DataFrame,
    player_id: str,
    percentiles_long: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    base = df[df["player_id"] == player_id]
    if base.empty:
        return {}
    profile = {
        "identity": base.iloc[0].to_dict(),
        "centiles": percentiles_long[percentiles_long["player_id"] == player_id],
    }
    return profile


def build_enriched_dataset(
    df: pd.DataFrame,
    *,
    weights: Mapping[str, float],
    groupby: Sequence[str] = ("position",),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("Building enriched dataset (groupby=%s)", groupby)
    metrics = [metric for metric in FEATURE_SET if metric in df.columns]
    percentile_result = compute_percentiles(df, metrics, groupby=groupby)
    enriched = percentile_result.frame.copy()
    index_series = performance_index(enriched, weights)
    enriched[index_series.name] = index_series
    logger.info("Enriched dataset computed (index column: %s)", index_series.name)
    return enriched, percentile_result.long
