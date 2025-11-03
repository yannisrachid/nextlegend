"""Horizontal category bar charts for NextLegend centiles."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping

import pandas as pd
import plotly.graph_objects as go


def bars_by_category(
    centiles_dict: Mapping[str, Mapping[str, float]],
    *,
    highlight: Iterable[str] | None = None,
) -> list[go.Figure]:
    """Generate horizontal bar figures for each category of centiles."""

    figures: list[go.Figure] = []
    highlight_set = {metric for metric in (highlight or [])}

    for category, metrics in centiles_dict.items():
        if not metrics:
            continue

        frame = (
            pd.DataFrame(metrics.items(), columns=["metric", "percentile"])
            .sort_values("percentile", ascending=True)
        )
        frame["label"] = frame["metric"].str.replace("_", " ").str.title()
        frame["is_highlight"] = frame["metric"].isin(highlight_set)

        fig = go.Figure(
            go.Bar(
                x=frame["percentile"],
                y=frame["label"],
                orientation="h",
                marker=dict(
                    color=[
                        "#7BD389" if flag else "rgba(148, 163, 184, 0.35)"
                        for flag in frame["is_highlight"]
                    ]
                ),
                hovertemplate="%{y}: %{x:.1f}<extra></extra>",
                text=[f"{value:.0f}" for value in frame["percentile"]],
                textposition="outside",
            )
        )

        fig.update_layout(
            title=dict(text=category, font=dict(size=16)),
            margin=dict(l=90, r=30, t=40, b=20),
            xaxis=dict(range=[0, 100], showgrid=True, gridcolor="rgba(255, 255, 255, 0.08)"),
            yaxis=dict(showgrid=False),
            template="plotly_dark",
            height=max(240, 60 * len(frame)),
        )
        figures.append(fig)

    return figures
