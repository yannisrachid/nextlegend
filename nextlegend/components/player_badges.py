"""Visual badges for player identity metadata."""

from __future__ import annotations

from typing import Mapping

import streamlit as st


def render_badges(player_row: Mapping[str, str | int | float]) -> None:
    """Render player identity badges in a horizontal layout."""

    badges: list[str] = []
    if player_row.get("age"):
        badges.append(f"Âge: {player_row.get('age')}")
    if player_row.get("position"):
        badges.append(f"Poste: {player_row.get('position')}")
    if player_row.get("team"):
        badges.append(f"Club: {player_row.get('team')}")
    if player_row.get("league"):
        badges.append(f"Ligue: {player_row.get('league')}")
    if player_row.get("strong_foot"):
        badges.append(f"Pied: {player_row.get('strong_foot')}")

    if not badges:
        return

    pills = " ".join(f"<span class='nextlegend-pill'>{badge}</span>" for badge in badges)
    st.markdown(f"<div style='display:flex; flex-wrap:wrap; gap:0.5rem;'>{pills}</div>", unsafe_allow_html=True)
