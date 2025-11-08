"""Prospect page for NextLegend."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from auth import render_account_controls, require_authentication
from components.sidebar import render_sidebar_logo
from scripts.positions_glossary import positions_glossary
from s3_utils import read_csv_from_s3
from utils import load_prospects_csv, save_prospects_csv

PLAYERS_DATA_KEY = "data/wyscout_players_cleaned.csv"
PLACEHOLDER_IMG = "https://placehold.co/160x160?text=No+Photo"
PCT_SUFFIX_LEAGUE = "_pct_league"
PCT_SUFFIX_GLOBAL = "_pct_global"


def normalize_label(value: object) -> Optional[str]:
    if value is None:
        return None
    label = str(value).strip()
    lowered = label.lower()
    if not label or lowered in {"nan", "none", "<na>"}:
        return None
    return label


def safe_percentile_value(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return float(number) if np.isfinite(number) else None


def get_role_percentiles(player_row: pd.Series, role_name: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    role_label = normalize_label(role_name)
    if not role_label:
        return None, None
    league_value = safe_percentile_value(player_row.get(f"{role_label}{PCT_SUFFIX_LEAGUE}"))
    global_value = safe_percentile_value(player_row.get(f"{role_label}{PCT_SUFFIX_GLOBAL}"))
    return league_value, global_value


def render_percentile_group(
    container: DeltaGenerator,
    title: str,
    role_label: Optional[str],
    league_value: Optional[float],
    global_value: Optional[float],
) -> None:
    role_text = role_label or "N/A"
    container.markdown(
        f"<p style='font-size:0.95rem;color:#94A3B8;margin-bottom:0.2rem;'>{title}: "
        f"<strong style='color:#E2E8F0;'>{role_text}</strong></p>",
        unsafe_allow_html=True,
    )
    value_cols = container.columns(2)

    def render_value(column, label: str, value: Optional[float], accent: str) -> None:
        display = "N/A" if value is None else f"{value:.1f}"
        column.markdown(
            f"""
            <div style="background-color:#0F172A;padding:10px 12px;border-radius:10px;">
                <div style="font-size:0.8rem;color:#94A3B8;text-transform:uppercase;letter-spacing:0.05em;">{label}</div>
                <div style="font-size:1.8rem;font-weight:700;color:{accent};margin-top:4px;">{display}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_value(value_cols[0], "League", league_value, "#38BDF8")
    render_value(value_cols[1], "Global", global_value, "#34D399")


@st.cache_data(show_spinner=False)
def load_players() -> pd.DataFrame:
    try:
        df = read_csv_from_s3(PLAYERS_DATA_KEY)
    except FileNotFoundError:
        local_path = Path("data") / "wyscout_players_cleaned.csv"
        if not local_path.exists():
            raise
        df = pd.read_csv(local_path)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .replace({"": np.nan, "-": np.nan})
            )
            df[col] = pd.to_numeric(df[col], errors="ignore")
    return df


st.set_page_config(page_title="Prospect", layout="wide", initial_sidebar_state="collapsed")
require_authentication()
render_sidebar_logo()
render_account_controls()
st.title("Prospect List")

players_df = load_players()
prospects_df = load_prospects_csv()

filter_col1, filter_col2 = st.columns(2)
filter_type = filter_col1.selectbox("Filter type", options=["Role", "Poste"])

if prospects_df.empty:
    st.info("No prospects recorded yet. Use the form below to add one.")

if filter_type == "Role":
    role_options = ["All"] + sorted(prospects_df["role"].dropna().unique().tolist())
    selected_role_label = filter_col2.selectbox("Select role", options=role_options)
    current_filter = selected_role_label if selected_role_label != "All" else None
else:
    stored_positions = sorted({pos for pos in prospects_df["position"].dropna().unique().tolist() if isinstance(pos, str) and pos.strip()})
    display_options = ["All"] + [positions_glossary.get(pos, pos) for pos in stored_positions]
    selected_position_label = filter_col2.selectbox("Select position", options=display_options)
    if selected_position_label == "All":
        current_filter = None
    else:
        reverse_lookup = {positions_glossary.get(pos, pos): pos for pos in stored_positions}
        current_filter = reverse_lookup.get(selected_position_label, selected_position_label)

st.markdown("### Add a prospect")
with st.expander("Add a new prospect to the list"):
    add_col1, add_col2, add_col3, add_col4 = st.columns(4)

    selected_league = add_col1.selectbox(
        "League",
        options=sorted(players_df.get("competition_name", pd.Series(dtype=str)).dropna().unique().tolist()),
        key="prospect_league",
    )

    league_filtered = players_df[players_df.get("competition_name").astype(str) == str(selected_league)]
    team_options = sorted({str(team) for team in league_filtered.get("team_in_selected_period", players_df.get("team")).dropna().unique()})
    selected_team = add_col2.selectbox(
        "Team",
        options=team_options,
        key="prospect_team",
    )

    team_filtered = league_filtered[league_filtered.get("team_in_selected_period", players_df.get("team")).astype(str) == str(selected_team)]
    player_options = sorted({str(name) for name in team_filtered.get("player", pd.Series(dtype=str)).dropna().unique()})
    selected_player = add_col3.selectbox("Player", options=player_options, key="prospect_player")

    if add_col4.button("Add"):
        player_info = team_filtered[team_filtered["player"] == selected_player]
        assigned_role = player_info["assigned_role"].iloc[0] if not player_info.empty else ""
        primary_position = player_info["position"].iloc[0] if not player_info.empty else ""

        new_entry = {
            "player": selected_player,
            "team": selected_team,
            "competition_name": selected_league,
            "position": primary_position,
            "role": assigned_role,
        }
        exists = not prospects_df[
            (prospects_df["player"] == selected_player)
            & (prospects_df["team"] == selected_team)
            & (prospects_df["competition_name"] == selected_league)
        ].empty
        if exists:
            st.info("This player is already in the prospect list.")
        else:
            updated_df = pd.concat([prospects_df, pd.DataFrame([new_entry])], ignore_index=True)
            save_prospects_csv(updated_df)
            st.rerun()

if prospects_df.empty:
    filtered_prospects = prospects_df
else:
    if filter_type == "Role":
        if current_filter is None:
            filtered_prospects = prospects_df.copy()
        else:
            filtered_prospects = prospects_df[prospects_df["role"] == current_filter].copy()
    else:
        if current_filter is None:
            filtered_prospects = prospects_df.copy()
        else:
            filtered_prospects = prospects_df[prospects_df["position"] == current_filter].copy()

if filtered_prospects.empty:
    st.info("No prospects recorded for this filter.")
    st.stop()

st.divider()

if filter_type == "Role":
    filter_label = selected_role_label if prospects_df.size else "All"
else:
    filter_label = selected_position_label if prospects_df.size else "All"

st.write(f"{len(filtered_prospects)} prospect(s) recorded for {filter_type}: **{filter_label}**")

for idx, row in filtered_prospects.iterrows():
    player_name = row["player"]
    team_name = row["team"]
    league_name = row["competition_name"]

    player_data = players_df[
        (players_df["player"] == player_name)
        & (players_df.get("team_in_selected_period", players_df.get("team")) == team_name)
        & (players_df["competition_name"] == league_name)
    ]
    if player_data.empty:
        player_data = players_df[players_df["player"] == player_name]
    if player_data.empty:
        st.warning(f"Player data not found for {player_name}.")
        continue

    player_row = player_data.sort_values(by="minutes_played", ascending=False).iloc[0]

    card = st.container()
    with card:
        header_col, button_col = st.columns([3, 1])
        with header_col:
            st.markdown(f"### {player_name} — {team_name}")
            st.caption(f"{league_name}")
        with button_col:
            if st.button("Delete", key=f"delete_{idx}"):
                updated_df = prospects_df.drop(index=row.name)
                save_prospects_csv(updated_df)
                st.rerun()

        info_cols = st.columns([1.2, 1, 1])
        with info_cols[0]:
            st.image(PLACEHOLDER_IMG, width=120)
            st.markdown(f"**Role:** {player_row.get('assigned_role', 'N/A')}")
            st.markdown(f"Top position: {player_row.get('position', 'N/A')}")
        with info_cols[1]:
            st.markdown("**Player info**")
            st.markdown(f"Age: {player_row.get('age', 'N/A')}")
            st.markdown(f"Minutes: {player_row.get('minutes_played', 'N/A')}")
            st.markdown(f"Matches: {player_row.get('matches_played', 'N/A')}")
            st.markdown(f"Goals: {player_row.get('goals', 'N/A')}")
            st.markdown(f"Assists: {player_row.get('assists', 'N/A')}")
        percentile_column = info_cols[2].container()
        assigned_role_label = normalize_label(player_row.get("assigned_role"))
        assigned_league_pct = safe_percentile_value(player_row.get("assigned_role_pct_league"))
        assigned_global_pct = safe_percentile_value(player_row.get("assigned_role_pct_global"))

        percentile_column.markdown("**Percentile scores**")
        render_percentile_group(
            percentile_column,
            "Assigned role percentile",
            assigned_role_label,
            assigned_league_pct,
            assigned_global_pct,
        )

        if filter_type == "Role":
            focus_role_label = normalize_label(current_filter) or normalize_label(row.get("role"))
            focus_title = "Filtered role percentile" if current_filter else "Prospect focus role percentile"
        else:
            focus_role_label = normalize_label(row.get("role"))
            focus_title = "Prospect focus role percentile"

        focus_league_pct, focus_global_pct = get_role_percentiles(player_row, focus_role_label)
        if focus_role_label:
            percentile_column.markdown("<div style='margin:8px 0;'></div>", unsafe_allow_html=True)
            render_percentile_group(
                percentile_column,
                focus_title,
                focus_role_label,
                focus_league_pct,
                focus_global_pct,
            )

        st.divider()
