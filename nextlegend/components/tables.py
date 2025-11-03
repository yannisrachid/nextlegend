"""Tabular helpers for Streamlit displays."""

from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd


def ranking_table(
    df: pd.DataFrame,
    sort_metric: str,
    *,
    ascending: bool = False,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    if sort_metric not in df.columns:
        raise ValueError(f"Metric {sort_metric} unavailable for ranking.")

    selection = df.dropna(subset=[sort_metric])
    ordered = selection.sort_values(sort_metric, ascending=ascending)
    display_cols = list(columns) if columns else [
        "player_name",
        "team",
        "league",
        "position",
        sort_metric,
    ]
    available = [col for col in display_cols if col in ordered.columns]
    return ordered[available].reset_index(drop=True)


def comparison_table(df_players: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
    ids = ["player_name", "season", "team"]
    available_features = [feature for feature in features if feature in df_players.columns]
    selection = df_players[ids + available_features] if available_features else df_players[ids]
    return selection.set_index(ids)
