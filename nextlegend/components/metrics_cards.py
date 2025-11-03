"""KPI metric cards for players."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import streamlit as st


def _format(value: float | int | None, suffix: str = "") -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    if isinstance(value, float):
        return f"{value:,.2f}{suffix}".replace(",", " ")
    return f"{value:,}{suffix}".replace(",", " ")


def render_metrics(player_row: Mapping[str, float | int | str]) -> None:
    """Render key KPIs for a player."""

    metrics = [
        ("Matches", player_row.get("matches")),
        ("Minutes", player_row.get("minutes")),
        ("Goals", player_row.get("goals")),
        ("Assists", player_row.get("assists")),
        ("xG", player_row.get("xg")),
        ("xA", player_row.get("xa")),
        ("Dribbles réussis/90", player_row.get("dribbles_per_90")),
        ("Prog. runs/90", player_row.get("progressive_runs_per_90")),
        ("Def. duels win %", player_row.get("def_duels_won_percent")),
        ("Aerial win %", player_row.get("aerial_duels_won_percent")),
        ("NextLegend Index", player_row.get("nextlegend_index")),
    ]

    stride = 3
    for offset in range(0, len(metrics), stride):
        cols = st.columns(stride)
        for idx, (label, value) in enumerate(metrics[offset : offset + stride]):
            display = _format(value)
            cols[idx].metric(label, display)
