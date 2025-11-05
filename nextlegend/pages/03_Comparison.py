"""Comparison page for NextLegend."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from mplsoccer import Radar

from scripts.positions_glossary import positions_glossary
from s3_utils import read_csv_from_s3

PAGES_DIR = Path(__file__).resolve().parent
ROOT_DIR = PAGES_DIR.parent
DATA_KEY = "data/wyscout_players_cleaned.csv"
PLACEHOLDER_IMG = "https://placehold.co/120x120?text=No+Photo"
PCT_SUFFIX_LEAGUE = "_pct_league"
PCT_SUFFIX_GLOBAL = "_pct_global"
PLAYER_COLORS = ["#DB2777", "#2563EB", "#F59E0B"]

RADAR_FALLBACK_METRICS = [
    "goals_per_90",
    "xa_per_90",
    "accurate_passes_percent",
    "passes_to_penalty_area_per_90",
    "progressive_passes_per_90",
    "progressive_runs_per_90",
    "successful_dribbles_percent",
    "def_duels_won_percent",
    "interceptions_padj",
    "aerial_duels_won_percent",
]

SUMMARY_DEFINITIONS: Dict[str, tuple[str, ...]] = {
    "summary_finishing": (
        "goals_per_90",
        "shots_per_90",
        "shots_on_target_percent",
        "goal_conversion_rate",
        "xg_per_90",
        "touches_in_penalty_area_per_90",
    ),
    "summary_aerial": (
        "aerial_duels_per_90",
        "aerial_duels_won_percent",
        "headed_goals_per_90",
    ),
    "summary_defense": (
        "successful_def_actions_per_90",
        "def_duels_won_percent",
        "interceptions_per_90",
        "sliding_tackles_per_90",
        "blocked_shots_per_90",
    ),
    "summary_technique": (
        "successful_dribbles_percent",
        "dribbles_per_90",
        "progressive_runs_per_90",
        "touches_in_penalty_area_per_90",
    ),
    "summary_creation": (
        "assists_per_90",
        "xa_per_90",
        "key_passes_per_90",
        "smart_passes_per_90",
        "passes_to_penalty_area_per_90",
        "deep_completions_per_90",
    ),
    "summary_construction": (
        "passes_per_90",
        "progressive_passes_per_90",
        "passes_to_final_third_per_90",
        "through_passes_per_90",
        "accurate_passes_percent",
    ),
}

SUMMARY_LABELS = {
    "summary_finishing": "Finishing",
    "summary_aerial": "Aerial",
    "summary_defense": "Defence",
    "summary_technique": "Technique",
    "summary_creation": "Creation",
    "summary_construction": "Construction",
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
        return f"{numeric:.0f}"
    return f"{numeric:.1f}"


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
    core = metric.replace("_per_90", " per 90").replace("_pct", " pct").replace("_", " ")
    core = core.replace(" per 90", " /90").replace(" pct", " (%)")
    return core.title()


def select_comparison_metrics(
    players: List[pd.Series],
    dataset: pd.DataFrame,
    roles_map: dict[str, List[str]],
) -> List[str]:
    roles = {row.get("assigned_role") for row in players if isinstance(row.get("assigned_role"), str)}
    metrics: List[str] = []

    if len(roles) == 1:
        role_name = roles.pop()
        role_metrics = roles_map.get(role_name)
        if role_metrics:
            metrics = [metric for metric in role_metrics if metric in dataset.columns][:10]

    if not metrics:
        for summaries in SUMMARY_DEFINITIONS.values():
            for metric in summaries:
                if metric in dataset.columns and metric not in metrics:
                    metrics.append(metric)
                    break

    if len(metrics) < 10:
        for metric in RADAR_FALLBACK_METRICS:
            if metric in dataset.columns and metric not in metrics:
                metrics.append(metric)
            if len(metrics) >= 10:
                break

    return metrics[:10]


def render_comparison_radar(
    players: List[pd.Series],
    dataset: pd.DataFrame,
    metrics: List[str],
    label_map: dict[str, str],
    use_percentiles: bool,
    context: str,
) -> Optional[plt.Figure]:
    if len(metrics) < 3:
        return None

    preferred_suffix = PCT_SUFFIX_LEAGUE if context == "League" else PCT_SUFFIX_GLOBAL
    fallback_suffix = PCT_SUFFIX_GLOBAL if preferred_suffix == PCT_SUFFIX_LEAGUE else PCT_SUFFIX_LEAGUE

    comp_col = None
    if "competition_name" in dataset.columns:
        comp_col = "competition_name"
    elif "league" in dataset.columns:
        comp_col = "league"

    min_range: List[float] = []
    max_range: List[float] = []
    round_flags: List[bool] = []
    labels = [metric_display_name(metric, label_map) for metric in metrics]

    if use_percentiles:
        min_range = [0.0] * len(metrics)
        max_range = [100.0] * len(metrics)
        round_flags = [True] * len(metrics)
    else:
        for metric in metrics:
            series = pd.to_numeric(dataset.get(metric, pd.Series(dtype=float)), errors="coerce")
            if metric.endswith("percent") or metric.endswith("_pct"):
                series = series.clip(lower=0.0, upper=100.0)
            series = series.dropna()
            if series.empty:
                min_val = 0.0
                max_val = 1.0
            else:
                min_val = float(series.min())
                max_val = float(series.max())
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
            if metric.endswith("percent") or metric.endswith("_pct"):
                min_adj = max(0.0, min_adj)
                max_adj = min(100.0, max_adj)
                if max_adj - min_adj < 1e-6:
                    max_adj = min(100.0, min_adj + 1.0)
            min_range.append(min_adj)
            max_range.append(max_adj)
            round_flags.append(False)

    radar = Radar(
        params=labels,
        min_range=min_range,
        max_range=max_range,
        round_int=round_flags,
    )

    fig = plt.figure(figsize=(5.5, 5.5), facecolor="#0F172A")
    ax = fig.add_subplot(111)
    radar.setup_axis(
        ax=ax,
        facecolor="#0F172A",
        title=dict(
            title=f"{'Percentiles' if use_percentiles else 'Raw values'} ({context})",
            color="#E2E8F0",
            size=12,
        ),
        subtitle=dict(title="", color="#E2E8F0"),
    )
    radar.draw_circles(ax=ax, facecolor="#1E293B", edgecolor="#475569")

    for idx, player_row in enumerate(players):
        player_color = PLAYER_COLORS[idx % len(PLAYER_COLORS)]
        values: List[float] = []
        player_league = player_row.get("competition_name")

        if context == "League" and comp_col and player_league:
            subset = dataset[dataset[comp_col].astype(str) == str(player_league)]
            if subset.empty:
                subset = dataset
        else:
            subset = dataset

        for metric in metrics:
            if use_percentiles:
                value = safe_float(player_row.get(f"{metric}{preferred_suffix}"))
                if value is None:
                    value = safe_float(player_row.get(f"{metric}{fallback_suffix}"))
                values.append(np.clip(value, 0, 100) if value is not None else np.nan)
            else:
                raw_value = safe_float(player_row.get(metric))
                if metric.endswith("percent") or metric.endswith("_pct"):
                    raw_value = float(np.clip(raw_value, 0.0, 100.0)) if raw_value is not None else raw_value
                values.append(raw_value if raw_value is not None else np.nan)

        if not any(np.isfinite(v) for v in values):
            continue

        radar.draw_radar(
            values,
            ax=ax,
            kwargs_radar={"facecolor": "none", "edgecolor": player_color, "linewidth": 2.2},
            kwargs_rings={"facecolor": "none", "edgecolor": "none"},
        )

    radar.draw_range_labels(ax=ax, fontsize=8, color="#94A3B8")
    radar.draw_param_labels(ax=ax, fontsize=9, color="#E2E8F0")

    return fig


st.set_page_config(page_title="Comparison", layout="wide", initial_sidebar_state="collapsed")

df_players = load_players()
if df_players.empty:
    st.warning("Player dataset is empty. Run the pipeline first.")
    st.stop()

roles_map, label_map = load_role_metrics()

st.title("Comparison")

comp_column = "competition_name" if "competition_name" in df_players.columns else None
if comp_column is None:
    st.error("Competition column not found in dataset.")
    st.stop()

league_options = ["Select league", "All competitions"] + sorted({str(val) for val in df_players[comp_column].dropna().unique()})

team_selection_column = "team_in_selected_period" if "team_in_selected_period" in df_players.columns else "team"

selector_cols = st.columns(3)
player_selectors = []
for idx, col in enumerate(selector_cols):
    with col:
        selected_league = col.selectbox(
            f"Player {idx + 1} - League",
            options=league_options,
            key=f"league_{idx}",
        )

        if selected_league == "Select league":
            player_selectors.append({"league": None, "team": None, "player": None})
            continue

        if selected_league == "All competitions":
            league_filtered = df_players
        else:
            league_filtered = df_players[df_players[comp_column].astype(str) == str(selected_league)]

        team_options = sorted({
            str(team)
            for team in league_filtered.get(team_selection_column, pd.Series(dtype=str)).dropna().unique()
        })
        if not team_options:
            col.info("No clubs available.")
            player_selectors.append({"league": selected_league, "team": None, "player": None})
            continue

        selected_team = col.selectbox(
            f"Player {idx + 1} - Club",
            options=team_options,
            key=f"club_{idx}",
        )

        player_filtered = league_filtered[league_filtered.get(team_selection_column).astype(str) == str(selected_team)]
        player_options = sorted({str(name) for name in player_filtered.get("player", pd.Series(dtype=str)).dropna().unique()})
        if not player_options:
            col.info("No players available.")
            player_selectors.append({"league": selected_league, "team": selected_team, "player": None})
            continue

        default_label = "Select player" if idx < 2 else "None"
        player_options = [default_label] + player_options
        selected_player = col.selectbox(
            f"Player {idx + 1}",
            options=player_options,
            key=f"player_{idx}",
        )

        if selected_player == default_label:
            selected_player = None

        player_selectors.append({"league": selected_league, "team": selected_team, "player": selected_player})

percentile_view = st.checkbox("Percentile view", value=True)
context_choice = st.selectbox("Context", options=["League", "Global"], index=0)

compare_button = st.button("Compare")

if not compare_button:
    st.stop()

selected_rows: List[pd.Series] = []
for selector in player_selectors:
    name = selector.get("player")
    if not name:
        continue
    row = df_players[df_players.get("player").astype(str) == str(name)].copy()
    if row.empty:
        continue
    # If multiple entries remain (e.g. different seasons), take the latest minute total.
    row = row.sort_values(by=["minutes_played"], ascending=False).iloc[0]
    selected_rows.append(row)

if len(selected_rows) < 2:
    st.warning("Please select at least two players before comparing.")
    st.stop()

comparison_colors = PLAYER_COLORS
selected_metrics = select_comparison_metrics(selected_rows, df_players, roles_map)

layout_left, layout_right = st.columns([0.3, 0.7])

with layout_left:
    st.subheader("Players")
    for idx, player_row in enumerate(selected_rows):
        color = comparison_colors[idx % len(comparison_colors)]
        player_name = player_row.get("player", "Unknown")
        team_name = player_row.get(team_selection_column, "")
        position = positions_glossary.get(str(player_row.get("position")), player_row.get("position"))
        age_value = safe_int(player_row.get("age"))
        minutes_value = safe_int(player_row.get("minutes_played"))
        league_score = display_value(player_row.get("assigned_role_pct_league"))
        global_score = display_value(player_row.get("assigned_role_pct_global"))

        st.markdown(
            f"""
            <div style="border: 1px solid {color}; border-radius: 8px; padding: 0.75rem; margin-bottom: 0.75rem;">
                <div style="font-weight:600; color:{color};">{player_name}</div>
                <div style="font-size:0.9rem; color:#E2E8F0;">{team_name}</div>
                <div style="font-size:0.85rem; color:#94A3B8;">{position}</div>
                <div style="margin-top:0.4rem; font-size:0.85rem; color:#CBD5F5;">
                    Age: {display_value(age_value)} • Minutes: {display_value(minutes_value)}
                </div>
                <div style="font-size:0.85rem; color:#CBD5F5;">
                    League score: {league_score} • Global score: {global_score}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with layout_right:
    st.subheader("Radar")
    radar_fig = render_comparison_radar(selected_rows, df_players, selected_metrics, label_map, percentile_view, context_choice)
    if radar_fig is None:
        st.info("Not enough metrics available to render the radar chart.")
    else:
        st.pyplot(radar_fig, use_container_width=True)

st.divider()

# Comparison table
metric_rows = []
for metric in selected_metrics:
    metric_label = metric_display_name(metric, label_map)
    row_dict = {"Metric": metric_label}
    for player_row, color in zip(selected_rows, comparison_colors):
        player_name = player_row.get("player", "Player")
        row_dict[player_name] = display_value(player_row.get(metric))
    metric_rows.append(row_dict)

if metric_rows:
    df_comparison = pd.DataFrame(metric_rows)
    st.subheader("Metric comparison")
    st.dataframe(df_comparison.set_index("Metric"), use_container_width=True)

st.divider()

# Summary scores comparison
summary_rows = []
for key, label in SUMMARY_LABELS.items():
    row_dict = {"Summary": label}
    for player_row in selected_rows:
        player_name = player_row.get("player", "Player")
        row_dict[player_name] = display_value(player_row.get(key))
    summary_rows.append(row_dict)

if summary_rows:
    df_summary = pd.DataFrame(summary_rows)
    st.subheader("Aggregated summary scores")
    styled = df_summary.set_index("Summary").style.format(display_value)
    st.dataframe(styled, use_container_width=True)
