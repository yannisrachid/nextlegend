"""Ranking page for NextLegend."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from mplsoccer import Radar

from auth import render_account_controls, require_authentication
from components.sidebar import render_sidebar_logo
from s3_utils import read_csv_from_s3
from scripts.positions_glossary import positions_glossary

PAGES_DIR = Path(__file__).resolve().parent
ROOT_DIR = PAGES_DIR.parent
DATA_KEY = "data/wyscout_players_cleaned.csv"
PLACEHOLDER_IMG = "https://placehold.co/160x160?text=No+Photo"
PCT_SUFFIX_LEAGUE = "_pct_league"
PCT_SUFFIX_GLOBAL = "_pct_global"
TRANSFERMARKT_BASE_URL = "https://www.transfermarkt.com"

RADAR_STATS = {
    "goals_per_90": "Goals",
    "xa_per_90": "xA",
    "accurate_passes_percent": "Passing accuracy (%)",
    "passes_to_penalty_area_per_90": "Passes to penalty area",
    "progressive_passes_per_90": "Progressive passes",
    "progressive_runs_per_90": "Progressive runs",
    "successful_dribbles_percent": "Successful dribbles (%)",
    "def_duels_won_percent": "Defensive duels won (%)",
    "interceptions_padj": "Possession-adjusted interceptions",
    "aerial_duels_won_percent": "Aerial duels won (%)",
}


def safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def safe_int(value: object) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return int(round(number))


@st.cache_data(show_spinner=False)
def load_players() -> pd.DataFrame:
    try:
        df = read_csv_from_s3(DATA_KEY)
    except FileNotFoundError:
        local_path = ROOT_DIR / "data" / "wyscout_players_cleaned.csv"
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


@st.cache_data(show_spinner=False)
def load_roles_options(df: pd.DataFrame) -> List[str]:
    column_candidates = [col for col in df.columns if col.endswith(PCT_SUFFIX_LEAGUE)]
    roles = []
    for column in column_candidates:
        if column in {"assigned_role_pct_league", "assigned_role_pct_global"}:
            continue
        if column.endswith(PCT_SUFFIX_LEAGUE):
            roles.append(column.replace(PCT_SUFFIX_LEAGUE, ""))
    roles.extend(df.get("assigned_role", pd.Series(dtype=str)).dropna().unique().tolist())
    clean_roles = sorted({role for role in roles if isinstance(role, str) and role.strip()})
    return clean_roles


def display_value(value: Optional[float]) -> str:
    if value is None:
        return "—"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "—"
    if np.isnan(numeric):
        return "—"
    if numeric.is_integer():
        return f"{numeric:.0f}"
    if abs(numeric) >= 100:
        return f"{value:.0f}"
    return f"{numeric:.1f}"


@st.cache_data(show_spinner=False)
def load_role_metrics() -> tuple[dict[str, List[str]], dict[str, str]]:
    path = ROOT_DIR / "roles_metrics.json"
    if not path.exists():
        return {}, {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}, {}
    roles = data.get("roles", {}) or {}
    labels = data.get("label_map_en", {}) or {}
    return roles, labels


def metric_display_name(metric: str, label_map: dict[str, str]) -> str:
    if metric in label_map:
        return label_map[metric]
    if metric in RADAR_STATS:
        return RADAR_STATS[metric]
    display = metric.replace("_per_90", " per 90").replace("_pct", " pct").replace("_", " ")
    display = display.replace(" per 90", " /90").replace(" pct", " (%)")
    return display.title()


def render_radar(
    row: pd.Series,
    container: st.delta_generator.DeltaGenerator,
    dataset: pd.DataFrame,
    role_name: Optional[str],
    use_percentiles: bool,
    context: str,
    player_league: Optional[str],
) -> None:
    roles_map, label_map = load_role_metrics()
    metrics = roles_map.get(role_name, []) if role_name else []
    if metrics:
        metric_pairs = [(metric, metric_display_name(metric, label_map)) for metric in metrics]
    else:
        metric_pairs = list(RADAR_STATS.items())

    preferred_suffix = PCT_SUFFIX_LEAGUE if context == "League" else PCT_SUFFIX_GLOBAL
    fallback_suffix = PCT_SUFFIX_GLOBAL if preferred_suffix == PCT_SUFFIX_LEAGUE else PCT_SUFFIX_LEAGUE

    comp_col = None
    if "competition_name" in dataset.columns:
        comp_col = "competition_name"
    elif "league" in dataset.columns:
        comp_col = "league"

    selected_labels: List[str] = []
    selected_values: List[float] = []
    min_range: List[float] = []
    max_range: List[float] = []
    round_flags: List[bool] = []

    if use_percentiles:
        for metric_key, display in metric_pairs:
            value = safe_float(row.get(f"{metric_key}{preferred_suffix}"))
            if value is None:
                value = safe_float(row.get(f"{metric_key}{fallback_suffix}"))
            if value is None:
                continue
            selected_labels.append(display)
            selected_values.append(np.clip(value, 0, 100))
        min_range = [0.0] * len(selected_labels)
        max_range = [100.0] * len(selected_labels)
        round_flags = [True] * len(selected_labels)
    else:
        source_subset = dataset
        if context == "League" and comp_col and player_league:
            league_filtered = dataset[dataset[comp_col].astype(str) == str(player_league)]
            if not league_filtered.empty:
                source_subset = league_filtered

        for metric_key, display in metric_pairs:
            raw_value = safe_float(row.get(metric_key))
            if raw_value is None:
                continue
            if metric_key.endswith("percent") or metric_key.endswith("_pct"):
                raw_value = float(np.clip(raw_value, 0.0, 100.0))
            selected_labels.append(display)
            selected_values.append(float(raw_value))

            series = pd.to_numeric(source_subset.get(metric_key, pd.Series(dtype=float)), errors="coerce")
            if metric_key.endswith("percent") or metric_key.endswith("_pct"):
                series = series.clip(lower=0.0, upper=100.0)
            series = series.dropna()
            if not series.empty:
                min_val = float(series.min())
                max_val = float(series.max())
            else:
                min_val = float(raw_value)
                max_val = float(raw_value)
            if max_val == min_val:
                delta = abs(max_val) * 0.1 if max_val != 0 else 1.0
                max_val += delta
                min_val -= delta
            span = max_val - min_val
            padding = span * 0.05
            if padding <= 0:
                padding = abs(max_val) * 0.05 or 1.0
            min_adj = min_val - padding
            max_adj = max_val + padding
            if metric_key.endswith("percent") or metric_key.endswith("_pct"):
                min_adj = max(0.0, min_adj)
                max_adj = min(100.0, max_adj)
                if max_adj - min_adj < 1e-6:
                    max_adj = min(100.0, min_adj + 1.0)
            min_range.append(min_adj)
            max_range.append(max_adj)
        round_flags = [False] * len(selected_labels)

    if len(selected_labels) < 3:
        container.info("Not enough metrics to render the radar chart.")
        return

    radar = Radar(
        params=selected_labels,
        min_range=min_range if not use_percentiles else [0] * len(selected_labels),
        max_range=max_range if not use_percentiles else [100] * len(selected_labels),
        round_int=round_flags if not use_percentiles else [True] * len(selected_labels),
    )

    fig = plt.figure(figsize=(4, 4), facecolor="#0F172A")
    ax = fig.add_subplot(111)
    radar.setup_axis(
        ax=ax,
        facecolor="#0F172A",
        title=dict(
            title=f"{'Percentiles' if use_percentiles else 'Raw values'} ({context})",
            color="#E2E8F0",
            size=11,
        ),
        subtitle=dict(title="", color="#E2E8F0"),
    )
    radar.draw_circles(ax=ax, facecolor="#1E293B", edgecolor="#475569")
    radar.draw_radar(
        selected_values,
        ax=ax,
        kwargs_radar={"facecolor": "#7BD389", "edgecolor": "#448361", "alpha": 0.55},
        kwargs_rings={"facecolor": "#7BD389", "alpha": 0.1},
    )
    radar.draw_range_labels(ax=ax, fontsize=8, color="#94A3B8")
    radar.draw_param_labels(ax=ax, fontsize=8, color="#E2E8F0")

    container.pyplot(fig, use_container_width=True)
    plt.close(fig)


def pick_value(*candidates: object) -> Optional[str]:
    for value in candidates:
        if value is None:
            continue
        if isinstance(value, str):
            text = value.strip()
            if text and text.lower() != "nan":
                return text
        elif isinstance(value, (int, float, np.number)) and not np.isnan(value):
            return str(value)
    return None


def ensure_absolute_url(value: object, *, default_base: Optional[str] = None) -> Optional[str]:
    text = pick_value(value)
    if not text:
        return None
    if text.startswith(("http://", "https://")):
        return text
    if text.startswith("//"):
        return f"https:{text}"
    if text.startswith("/"):
        base = default_base or TRANSFERMARKT_BASE_URL
        return f"{base.rstrip('/')}{text}"
    return f"https://{text}"


st.set_page_config(page_title="Ranking", layout="wide", initial_sidebar_state="collapsed")
require_authentication()
render_sidebar_logo()
render_account_controls()

df_players = load_players()
roles_options = load_roles_options(df_players)

if df_players.empty:
    st.warning("Player dataset is empty. Run the pipeline first.")
    st.stop()

st.title("Ranking")

comp_column = "competition_name" if "competition_name" in df_players.columns else None
if comp_column is None:
    st.error("Competition column not found in dataset.")
    st.stop()

# Big 5 leagues
BIG5_LEAGUES = {
    "England. Premier League",
    "France. Ligue 1",
    "Germany. Bundesliga",
    "Italy. Serie A",
    "Spain. La Liga",
}

available_leagues = ["All competitions", "Big 5 Leagues"] + sorted(df_players[comp_column].dropna().unique().tolist())
if not available_leagues:
    st.info("No leagues available in the dataset.")
    st.stop()

filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)

selected_league = filter_col1.selectbox("Select a league", options=available_leagues)

mode_choice = filter_col2.selectbox("Select by", options=["Poste", "Role"])

if mode_choice == "Poste":
    positions = sorted({pos for pos in df_players.get("position", pd.Series(dtype=str)).dropna().unique().tolist() if isinstance(pos, str) and pos.strip()})
    if not positions:
        st.info("No positions found in dataset.")
        st.stop()
    display_options = [positions_glossary.get(pos, pos) for pos in positions]
    selected_position_label = filter_col2.selectbox("Select position", options=display_options, key="ranking_position")
    reverse_lookup = {display: original for display, original in zip(display_options, positions)}
    selected_value = reverse_lookup[selected_position_label]
else:
    selected_value = filter_col2.selectbox("Select role", options=roles_options, key="ranking_role")

age_series = pd.to_numeric(df_players.get("age"), errors="coerce")
min_age = int(age_series.min(skipna=True) or 16)
max_age = int(age_series.max(skipna=True) or 40)
selected_age = filter_col3.slider("Select max age", min_value=min_age, max_value=max_age, value=max_age)

percentile_view = filter_col4.checkbox("Percentile view", value=True)
context_choice = filter_col4.selectbox("Context", options=["League", "Global"], index=0)

if selected_league == "All competitions":
    filtered = df_players.copy()
elif selected_league == "Big 5 Leagues":
    filtered = df_players[df_players[comp_column].astype(str).isin(BIG5_LEAGUES)].copy()
else:
    filtered = df_players[df_players[comp_column].astype(str) == str(selected_league)].copy()
if filtered.empty:
    st.info("No players found for this league.")
    st.stop()

filtered = filtered[pd.to_numeric(filtered.get("age"), errors="coerce") <= float(selected_age)]

if mode_choice == "Poste":
    filtered = filtered[filtered.get("position").astype(str) == str(selected_value)]
    if selected_league in {"All competitions", "Big 5 Leagues"}:
        ranking_column = "assigned_role_pct_global"
    else:
        ranking_column = "assigned_role_pct_league"
else:
    filtered = filtered[filtered.get("assigned_role").astype(str) == str(selected_value)]
    if selected_league in {"All competitions", "Big 5 Leagues"}:
        ranking_column = "assigned_role_pct_global"
    else:
        candidate_column = f"{selected_value}{PCT_SUFFIX_LEAGUE}"
        ranking_column = candidate_column if candidate_column in filtered.columns else "assigned_role_pct_league"

if filtered.empty:
    st.info("No players match the selected filters.")
    st.stop()

filtered = filtered.assign(_ranking_score=pd.to_numeric(filtered.get(ranking_column), errors="coerce"))
filtered = filtered.dropna(subset=["_ranking_score"])

filtered = filtered.sort_values("_ranking_score", ascending=False).head(30).reset_index(drop=True)

if filtered.empty:
    st.info("No ranked players available after applying filters.")
    st.stop()

st.divider()

st.write(f"Displaying top {len(filtered)} players")

for idx, player_row in filtered.iterrows():
    rank_position = idx + 1
    player_name = player_row.get("player", "Unknown player")
    team_name = player_row.get("team", "")
    display_name = f"{player_name} - {team_name}" if team_name else player_name

    league_score = player_row.get(ranking_column)
    if ranking_column.endswith(PCT_SUFFIX_LEAGUE):
        companion = ranking_column.replace(PCT_SUFFIX_LEAGUE, PCT_SUFFIX_GLOBAL)
        global_score = player_row.get(companion) if companion in player_row.index else player_row.get("assigned_role_pct_global")
    else:
        global_score = player_row.get("assigned_role_pct_global")

    age_value = safe_int(player_row.get("age"))
    minutes_value = safe_int(player_row.get("minutes_played"))
    matches_value = safe_int(player_row.get("matches_played"))
    goals_value = safe_int(player_row.get("goals"))
    assists_value = safe_int(player_row.get("assists"))

    card = st.container()
    with card:
        left, middle, right = st.columns([1.2, 1, 1.8])

        with left:
            st.markdown(f"### #{rank_position}")
            photo_url = ensure_absolute_url(
                pick_value(
                    player_row.get("tm_profile_image_url"),
                    player_row.get("profile_image_url"),
                )
            ) or PLACEHOLDER_IMG
            st.image(photo_url, width=120)
            st.markdown(f"**{display_name}**")
            tm_profile = ensure_absolute_url(player_row.get("tm_profile_url"), default_base=TRANSFERMARKT_BASE_URL)
            if tm_profile:
                st.markdown(f"[Transfermarkt profile]({tm_profile})", unsafe_allow_html=False)
            st.caption(
                f"League score: {display_value(league_score)} | Global score: {display_value(global_score)}"
            )

        with middle:
            st.markdown("**Player info**")
            st.markdown(f"Age: {display_value(age_value)}")
            st.markdown(f"Minutes: {display_value(minutes_value)}")
            st.markdown(f"Matches: {display_value(matches_value)}")
            st.markdown(f"Goals: {display_value(goals_value)}")
            st.markdown(f"Assists: {display_value(assists_value)}")
            agent_name = pick_value(player_row.get("tm_agent_name"))
            agent_url = ensure_absolute_url(
                player_row.get("tm_agent_url"),
                default_base=TRANSFERMARKT_BASE_URL,
            )
            if agent_name:
                if agent_url:
                    st.markdown(f"Agent: [{agent_name}]({agent_url})")
                else:
                    st.markdown(f"Agent: {agent_name}")

        with right:
            render_radar(
                player_row,
                right,
                df_players,
                player_row.get("assigned_role"),
                percentile_view,
                context_choice,
                player_row.get("competition_name"),
            )

    st.divider()
