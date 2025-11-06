"""Prospect page for NextLegend."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from scripts.positions_glossary import positions_glossary
from s3_utils import read_csv_from_s3
from utils import load_prospects_csv, save_prospects_csv

PLAYERS_DATA_KEY = "data/wyscout_players_cleaned.csv"
PLACEHOLDER_IMG = "https://placehold.co/160x160?text=No+Photo"


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


players_df = load_players()
prospects_df = load_prospects_csv()

st.set_page_config(page_title="Prospect", layout="wide", initial_sidebar_state="collapsed")
st.title("Prospect List")

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

        info_cols = st.columns([1.2, 1, 1.8])
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

        st.divider()
