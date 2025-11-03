"""Similarity computations for scouting suggestions."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


def standardize(df: pd.DataFrame, features: Sequence[str]) -> tuple[pd.DataFrame, StandardScaler]:
    available = [feature for feature in features if feature in df.columns]
    if not available:
        raise ValueError("No features available for standardisation.")

    scaler = StandardScaler()
    matrix = df[available].fillna(0.0).values
    transformed = scaler.fit_transform(matrix)
    standardized = pd.DataFrame(transformed, columns=available, index=df.index)
    return standardized, scaler


def cosine_sim_matrix(df_std: pd.DataFrame) -> pd.DataFrame:
    matrix = cosine_similarity(df_std.values)
    return pd.DataFrame(matrix, index=df_std.index, columns=df_std.index)


def top_similar(
    df_std: pd.DataFrame,
    df_meta: pd.DataFrame,
    ref_player_id: str,
    features: Sequence[str],
    *,
    top_n: int = 20,
    same_foot: bool | None = None,
    same_position: bool | None = None,
    metric_weights: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    if ref_player_id not in df_meta["player_id"].values:
        raise ValueError(f"Reference player {ref_player_id} not found.")

    available = [feature for feature in features if feature in df_std.columns]
    if not available:
        raise ValueError("Provided features are not present in standardized dataframe.")

    weighted = df_std[available].copy()
    if metric_weights:
        for metric, weight in metric_weights.items():
            if metric in weighted.columns:
                weighted[metric] = weighted[metric] * float(weight)

    similarities = cosine_similarity(weighted.values)
    sim_df = pd.DataFrame(
        similarities,
        index=df_meta["player_id"],
        columns=df_meta["player_id"],
    )

    ref_scores = sim_df.loc[ref_player_id].drop(ref_player_id, errors="ignore")

    merged = (
        df_meta.set_index("player_id")
        .join(ref_scores.rename("similarity"))
        .dropna(subset=["similarity"])
        .sort_values("similarity", ascending=False)
    )

    if same_foot and "strong_foot" in merged.columns:
        ref_foot = (
            df_meta.loc[df_meta["player_id"] == ref_player_id, "strong_foot"]
            .dropna()
            .astype(str)
            .str.strip()
            .str.lower()
            .head(1)
            .tolist()
        )
        if ref_foot:
            merged = merged[
                merged["strong_foot"].astype(str).str.strip().str.lower() == ref_foot[0]
            ]

    if same_position and "position" in merged.columns:
        ref_pos = (
            df_meta.loc[df_meta["player_id"] == ref_player_id, "position"]
            .dropna()
            .astype(str)
            .str.strip()
            .head(1)
            .tolist()
        )
        if ref_pos:
            merged = merged[
                merged["position"].astype(str).str.strip() == ref_pos[0]
            ]

    return merged.head(top_n).reset_index()
