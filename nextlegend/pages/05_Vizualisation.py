"""Vizualisation page for NextLegend."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.font_manager import FontProperties
from mplsoccer import PyPizza

from auth import render_account_controls, require_authentication
from components.sidebar import render_sidebar_logo
from s3_utils import read_csv_from_s3
from scripts.positions_glossary import positions_glossary

PAGES_DIR = Path(__file__).resolve().parent
ROOT_DIR = PAGES_DIR.parent
DATA_KEY = "data/wyscout_players_cleaned.csv"
PCT_SUFFIX_LEAGUE = "_pct_league"
PCT_SUFFIX_GLOBAL = "_pct_global"

GLOBAL_HIGHLIGHT_THRESHOLD = 75.0

EXTENDED_SECTIONS: Sequence[Dict[str, object]] = (
    {
        "key": "goal_creation",
        "title": "Goal Creation",
        "metrics": [
            "xa_per_90",
            "key_passes_per_90",
            "smart_passes_per_90",
            "shot_assists_per_90",
            "passes_to_penalty_area_per_90",
            "deep_completions_per_90",
            "through_passes_per_90",
            "progressive_passes_per_90",
        ],
    },
    {
        "key": "attacking_threat",
        "title": "Attacking Threat",
        "metrics": [
            "goals_per_90",
            "xg_per_90",
            "shots_per_90",
            "shots_on_target_percent",
            "touches_in_penalty_area_per_90",
            "progressive_runs_per_90",
            "accelerations_per_90",
            "non_penalty_goals_per_90",
        ],
    },
    {
        "key": "crossing_delivery",
        "title": "Crossing & Delivery",
        "metrics": [
            "crosses_per_90",
            "accurate_crosses_percent",
            "deep_crosses_per_90",
            "crosses_to_goalkeeper_per_90",
            "crosses_to_box_per_90",
            "crosses_to_penalty_area_per_90",
        ],
    },
    {
        "key": "defensive_contribution",
        "title": "Defensive Contribution",
        "metrics": [
            "successful_def_actions_per_90",
            "def_duels_per_90",
            "def_duels_won_percent",
            "interceptions_per_90",
            "sliding_tackles_per_90",
            "aerial_duels_won_percent",
            "blocked_shots_per_90",
            "recoveries_per_90",
        ],
    },
    {
        "key": "pressing_activity",
        "title": "Pressing & Work Rate",
        "metrics": [
            "offensive_duels_per_90",
            "duels_per_90",
            "pressures_per_90",
            "fouls_per_90",
            "counterpressing_recoveries_per_90",
            "ball_recoveries_in_final_third_per_90",
        ],
    },
    {
        "key": "build_up_play",
        "title": "Build-Up & Progression",
        "metrics": [
            "passes_per_90",
            "accurate_passes_percent",
            "progressive_passes_per_90",
            "passes_to_final_third_per_90",
            "passes_to_penalty_area_per_90",
            "progressive_carries_per_90",
            "long_passes_per_90",
            "accurate_long_passes_percent",
        ],
    },
    {
        "key": "possession_retention",
        "title": "Possession Retention",
        "metrics": [
            "dribbles_per_90",
            "successful_dribbles_percent",
            "ball_losses_per_90",
            "dispossessed_per_90",
            "miscontrols_per_90",
            "passes_received_per_90",
        ],
    },
)


def safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


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
def load_metric_labels() -> dict[str, str]:
    path = ROOT_DIR / "roles_metrics.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data.get("label_map_en", {}) or {}


def parse_positions(cell: object) -> List[str]:
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    text = str(cell)
    return [token.strip() for token in text.split(",") if token.strip()]


def extract_position_tokens(row: pd.Series) -> List[str]:
    primary = parse_positions(row.get("position"))
    secondary = parse_positions(row.get("second_position"))
    tokens = primary + secondary
    return [token for token in tokens if token]


def unique_positions(df: pd.DataFrame) -> List[str]:
    tokens: set[str] = set()
    for _, row in df.iterrows():
        tokens.update(extract_position_tokens(row))
    cleaned = {token for token in tokens if token and token.lower() != "nan"}
    return sorted(cleaned)


def display_label_for_position(code: str) -> str:
    return positions_glossary.get(code, code)


def filter_dataset_for_context(
    df: pd.DataFrame,
    *,
    positions: Sequence[str],
    min_minutes: int,
    context: str,
    competition: Optional[str],
    player_row: pd.Series,
) -> pd.DataFrame:
    filtered = df.copy()
    if min_minutes > 0 and "minutes_played" in filtered.columns:
        filtered = filtered[pd.to_numeric(filtered["minutes_played"], errors="coerce") >= min_minutes]

    if positions:
        selected = set(positions)

        def has_position(row: pd.Series) -> bool:
            tokens = set(extract_position_tokens(row))
            return bool(tokens & selected)

        filtered = filtered[filtered.apply(has_position, axis=1)]

    if context == "League" and competition:
        filtered = filtered[
            filtered.get("competition_name", pd.Series(dtype=str)).astype(str) == str(competition)
        ]

    if player_row.name not in filtered.index:
        filtered = pd.concat([filtered, player_row.to_frame().T], ignore_index=True)

    return filtered


def compute_percentiles(
    player_row: pd.Series,
    comparison_df: pd.DataFrame,
    metrics: Iterable[str],
) -> Dict[str, float]:
    percentiles: Dict[str, float] = {}
    for metric in metrics:
        player_value = safe_float(player_row.get(metric))
        if player_value is None:
            continue

        if metric not in comparison_df.columns:
            continue

        series = pd.to_numeric(comparison_df[metric], errors="coerce").dropna()
        if series.empty:
            continue

        values = np.sort(np.append(series.values, player_value))
        position = np.searchsorted(values, player_value, side="right")
        percentile = (position / len(values)) * 100.0
        percentiles[metric] = float(np.clip(percentile, 0.0, 100.0))
    return percentiles


def get_metric_label(metric: str, label_map: dict[str, str]) -> str:
    if metric in label_map:
        return label_map[metric]
    label = (
        metric.replace("_per_90", " per 90")
        .replace("_pct", " (%)")
        .replace("_percent", " (%)")
        .replace("_", " ")
    )
    return label.title()


def build_pizza_chart(
    labels: Sequence[str],
    values: Sequence[float],
    player_name: str,
    subtitle: str,
) -> plt.Figure:

    font_normal = FontProperties(family="DejaVu Sans", weight="normal")
    font_bold = FontProperties(family="DejaVu Sans", weight="bold")
    font_italic = FontProperties(family="DejaVu Sans", style="italic")

    slice_colors = [
        "#01ca0a" if value >= GLOBAL_HIGHLIGHT_THRESHOLD else "#4ecc54"
        for value in values
    ]
    text_colors = ["#000000"] * len(values)

    baker = PyPizza(
        params=list(labels),
        background_color="#0F172A",
        straight_line_color="#111111",
        straight_line_lw=1,
        last_circle_color="#111111",
        last_circle_lw=1,
        other_circle_lw=0,
        inner_circle_size=20,
    )

    fig, ax = baker.make_pizza(
        values,
        figsize=(7, 8),
        color_blank_space="same",
        slice_colors=slice_colors,
        value_colors=text_colors,
        value_bck_colors=slice_colors,
        blank_alpha=0.4,
        kwargs_slices=dict(edgecolor="#000000", linewidth=0.8, zorder=2),
        kwargs_params=dict(
            color="#E2E8F0",
            fontsize=11,
            fontproperties=font_normal,
            va="center",
        ),
        kwargs_values=dict(
            color="#000000",
            fontsize=10,
            fontproperties=font_normal,
            zorder=3,
            bbox=dict(
                edgecolor="#000000",
                facecolor="#F2F2F2",
                boxstyle="round,pad=0.2",
                lw=0.8,
            ),
        ),
    )

    fig.subplots_adjust(top=0.74, bottom=0.12)

    fig.text(
        0.5,
        0.995,
        player_name,
        size=16,
        ha="center",
        fontproperties=font_bold,
        color="#F2F2F2",
    )
    fig.text(
        0.5,
        0.965,
        subtitle,
        size=12,
        ha="center",
        fontproperties=font_bold,
        color="#F2F2F2",
    )
    fig.text(
        0.98,
        0.015,
        "Data: StatsBomb\nYour Legend",
        size=8,
        fontproperties=font_italic,
        color="#F2F2F2",
        ha="right",
    )
    fig.text(
        0.02,
        0.015,
        "Statistics scaled per 90 minutes",
        size=8,
        fontproperties=font_italic,
        color="#F2F2F2",
        ha="left",
    )

    plt.tight_layout()
    return fig


def render_section(
    *,
    section: Dict[str, object],
    percentiles: Dict[str, float],
    label_map: dict[str, str],
    default_metrics: Sequence[str],
    default_show_top: bool,
    player_label: str,
    subtitle_suffix: str,
    min_minutes: int,
) -> None:
    section_key = str(section["key"])
    title = str(section["title"])
    section_metrics = [metric for metric in default_metrics if metric in percentiles]

    st.divider()
    st.subheader(title)

    col_chart, col_filters = st.columns(2)

    with col_filters:
        options = [metric for metric in default_metrics if metric in percentiles]
        if not options:
            st.info("No metrics available for this section with the current filters.")
            return
        default_selection = section_metrics if section_metrics else options[: min(6, len(options))]
        selected_metrics = st.multiselect(
            "Metrics",
            options=options,
            default=default_selection,
            key=f"viz_metrics_{section_key}",
            help="Select the metrics to display on the Pizza Chart.",
        )
        show_only_top = st.checkbox(
            "Show only +75 percentile values",
            value=default_show_top,
            key=f"viz_top_{section_key}",
        )

    metrics_to_plot = []
    for metric in selected_metrics:
        value = percentiles.get(metric)
        if value is None:
            continue
        if show_only_top and value < GLOBAL_HIGHLIGHT_THRESHOLD:
            continue
        metrics_to_plot.append((metric, round(value)))

    if not metrics_to_plot:
        col_chart.info("No metrics to display with the current selections.")
        return

    labels = [get_metric_label(metric, label_map) for metric, _ in metrics_to_plot]
    values = [float(val) for _, val in metrics_to_plot]
    subtitle = f"{title} • {subtitle_suffix} (min {min_minutes} mins)"

    fig = build_pizza_chart(labels, values, player_label, subtitle)
    with col_chart:
        st.pyplot(fig, use_container_width=True, clear_figure=True)
    plt.close(fig)


def generate_dynamic_sections(
    percentiles: Dict[str, float],
    threshold: float,
) -> List[Dict[str, object]]:
    dynamic_sections: List[Dict[str, object]] = []
    for section in EXTENDED_SECTIONS:
        metrics = [
            metric
            for metric in section["metrics"]  # type: ignore[index]
            if percentiles.get(metric, 0.0) >= threshold
        ]
        if len(metrics) >= 2:
            dynamic_sections.append(
                {
                    "key": section["key"],
                    "title": section["title"],
                    "metrics": metrics,
                }
            )

    if dynamic_sections:
        return dynamic_sections

    top_metrics = sorted(
        ((metric, value) for metric, value in percentiles.items() if value >= threshold),
        key=lambda item: item[1],
        reverse=True,
    )[:6]
    if not top_metrics:
        return []

    dynamic_sections.append(
        {
            "key": "high_impact",
            "title": "High Impact Metrics",
            "metrics": [metric for metric, _ in top_metrics],
        }
    )
    return dynamic_sections


def scout_analyst_engine(
    percentiles: Dict[str, float],
    *,
    threshold: float,
    enforce_top: bool,
) -> List[Dict[str, object]]:
    if enforce_top:
        sections = generate_dynamic_sections(percentiles, threshold)
        reason = (
            "Focus on metrics where the player ranks above the 75th percentile. "
            "Sections have been filtered to highlight the most decisive contributions."
        )
    else:
        sections = list(EXTENDED_SECTIONS)
        reason = (
            "Full analytical breakdown retained. Sections are prioritised based on positional relevance."
        )

    titles = ", ".join(section["title"] for section in sections) if sections else "None"
    st.caption(
        f"🧠 *Scout Analyst Engine*: {reason} — Active focus: {titles}"
    )
    return sections


st.set_page_config(
    page_title="Vizualisation",
    layout="wide",
    initial_sidebar_state="collapsed",
)
require_authentication()
render_sidebar_logo()
render_account_controls()

df_players = load_players()
label_map = load_metric_labels()

if df_players.empty:
    st.warning("Player dataset is empty. Run the pipeline first.")
    st.stop()

st.title("Vizualisation")

competition_col = "competition_name" if "competition_name" in df_players.columns else None
if competition_col is None:
    st.error("Competition column not found in dataset.")
    st.stop()

league_options = sorted(
    {str(val) for val in df_players[competition_col].dropna().unique()}
)

position_choices = unique_positions(df_players)
display_to_code = {display_label_for_position(code): code for code in position_choices}
code_to_display = {code: display_label_for_position(code) for code in position_choices}
display_options = sorted(display_to_code.keys())

col_league, col_team, col_player = st.columns(3)
selected_league = col_league.selectbox("Select league", options=league_options)

league_df = df_players[df_players[competition_col].astype(str) == str(selected_league)]

team_column = "team_in_selected_period" if "team_in_selected_period" in df_players.columns else "team"
team_options = sorted(
    {
        str(team)
        for team in league_df.get(team_column, pd.Series(dtype=str)).dropna().unique()
    }
)

selected_team = col_team.selectbox("Select club", options=team_options)
team_df = league_df[league_df.get(team_column).astype(str) == str(selected_team)]

player_options = sorted(
    {str(name) for name in team_df.get("player", pd.Series(dtype=str)).dropna().unique()}
)
selected_player = col_player.selectbox("Select player", options=player_options)

if not selected_player:
    st.info("Select a player to display visualisations.")
    st.stop()

player_rows = team_df[team_df.get("player").astype(str) == str(selected_player)].copy()
if player_rows.empty:
    st.warning("No data found for the selected player.")
    st.stop()

player_rows = player_rows.sort_values(by="minutes_played", ascending=False, na_position="last")
player_row = player_rows.iloc[0]

player_positions = extract_position_tokens(player_row)
if not player_positions:
    if isinstance(player_row.get("position"), str) and player_row.get("position"):
        player_positions = [str(player_row.get("position"))]
default_positions = [pos for pos in player_positions if pos in position_choices] or []

filter_col_context, filter_col_positions, filter_col_minutes, filter_col_threshold = st.columns(
    (1.2, 1.6, 1, 1)
)
context_option = filter_col_context.selectbox(
    "Context",
    options=("League", "Global"),
    index=0,
    help="Choose the percentile comparison cohort.",
)
selected_positions = filter_col_positions.multiselect(
    "Compare against positions",
    options=display_options,
    default=[code_to_display[pos] for pos in default_positions] if default_positions else [],
    help="Add or remove positional groups to shape the comparison peer group.",
)
selected_position_codes = [display_to_code[label] for label in selected_positions]
min_minutes = filter_col_minutes.number_input(
    "Minimum minutes played",
    min_value=0,
    max_value=5000,
    value=270,
    step=30,
)
show_only_top_global = filter_col_threshold.checkbox(
    "Generate with only +75 percentile values",
    value=False,
    help="Generate thematic visuals focusing only on metrics where the player exceeds the 75th percentile.",
)

comparison_df = filter_dataset_for_context(
    df_players,
    positions=selected_position_codes,
    min_minutes=int(min_minutes),
    context=context_option,
    competition=selected_league if context_option == "League" else None,
    player_row=player_row,
)

if comparison_df.empty:
    st.warning("The comparison cohort is empty with the current filters.")
    st.stop()

all_metrics: set[str] = set()
for section in EXTENDED_SECTIONS:
    all_metrics.update(section["metrics"])  # type: ignore[arg-type]

percentiles = compute_percentiles(player_row, comparison_df, all_metrics)

player_label = f"{selected_player} — {selected_team}"

comparison_label = (
    f"{selected_league} peer group"
    if context_option == "League"
    else "global peer group"
)

subtitle_suffix = f"Percentile vs {comparison_label}"

active_sections = scout_analyst_engine(
    percentiles=percentiles,
    threshold=GLOBAL_HIGHLIGHT_THRESHOLD,
    enforce_top=show_only_top_global,
)

if not active_sections:
    st.info("No relevant metrics identified with the current configuration.")
    st.stop()

for section in active_sections:
    default_metrics = section["metrics"]  # type: ignore[assignment]
    render_section(
        section=section,
        percentiles=percentiles,
        label_map=label_map,
        default_metrics=default_metrics,
        default_show_top=show_only_top_global,
        player_label=player_label,
        subtitle_suffix=subtitle_suffix,
        min_minutes=int(min_minutes),
    )
