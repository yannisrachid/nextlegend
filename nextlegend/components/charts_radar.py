"""Radar chart generation for NextLegend."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

DEFAULT_COLORS = [
    "#7BD389",
    "#FF7A59",
    "#4C6EF5",
    "#FEC601",
]


def _prepare_axes(feature_groups: Mapping[str, Sequence[str]]) -> list[str]:
    axes: list[str] = []
    for metrics in feature_groups.values():
        axes.extend(metrics)
    return axes


def make_radar(
    df_centiles_long: pd.DataFrame,
    entities: Sequence[Mapping[str, str]],
    feature_groups: Mapping[str, Sequence[str]],
    mode: str = "centile",
) -> go.Figure:
    """Create a radar chart comparing entities across configured metrics.

    Parameters
    ----------
    df_centiles_long:
        DataFrame expected to contain columns: player_id, metric, category, value, percentile.
    entities:
        Iterable of dictionaries with at minimum keys ``player_id`` and ``label``.
    feature_groups:
        Ordered mapping of category -> list of metric column names.
    mode:
        Either ``centile`` (default) to display percentile scores, or ``value`` to show raw per90 metrics.
    """

    if df_centiles_long.empty or not entities:
        return go.Figure()

    axes = _prepare_axes(feature_groups)
    angle_labels = [metric.replace("_", " ").title() for metric in axes]
    fig = go.Figure()

    for idx, entity in enumerate(entities):
        player_id = entity.get("player_id") or entity.get("id")
        label = entity.get("label") or entity.get("player_name") or str(player_id)
        color = entity.get("color") or DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]

        player_slice = df_centiles_long[df_centiles_long["player_id"] == player_id]
        if player_slice.empty:
            continue

        values = []
        for metric in axes:
            metric_slice = player_slice[player_slice["metric"] == metric]
            if metric_slice.empty:
                values.append(np.nan)
                continue
            if mode == "value":
                values.append(metric_slice["value"].iloc[0])
            else:
                values.append(metric_slice["percentile"].iloc[0])

        # Close loop by repeating first point
        values.append(values[0])
        labels = angle_labels + [angle_labels[0]]

        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=labels,
                fill="toself",
                name=label,
                opacity=0.65,
                line=dict(color=color, width=2),
                hovertemplate="%{theta}<br>%{r:.1f}<extra>%s</extra>" % label,
            )
        )

    radial_range = [0, 100] if mode == "centile" else None

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=radial_range,
                showline=False,
                gridcolor="rgba(221, 224, 227, 0.18)",
            ),
            angularaxis=dict(
                rotation=90,
                direction="clockwise",
                gridcolor="rgba(221, 224, 227, 0.08)",
            ),
        ),
        template="plotly_dark",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        margin=dict(l=30, r=30, t=30, b=60),
    )

    return fig
