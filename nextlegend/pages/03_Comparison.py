"""Comparison page for NextLegend."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import random
from collections import deque
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
metric_rows_display: List[Dict[str, object]] = []
metric_rows_numeric: List[Dict[str, Optional[float]]] = []
for metric in selected_metrics:
    metric_label = metric_display_name(metric, label_map)
    row_display: Dict[str, object] = {"Metric": metric_label}
    row_numeric: Dict[str, Optional[float]] = {"Metric": metric_label}
    for player_row, color in zip(selected_rows, comparison_colors):
        player_name = player_row.get("player", "Player")
        raw_value = safe_float(player_row.get(metric))
        row_display[player_name] = display_value(player_row.get(metric))
        row_numeric[player_name] = raw_value
    metric_rows_display.append(row_display)
    metric_rows_numeric.append(row_numeric)

df_comparison_numeric = (
    pd.DataFrame(metric_rows_numeric).set_index("Metric")
    if metric_rows_numeric
    else pd.DataFrame()
)

if metric_rows_display:
    df_comparison = pd.DataFrame(metric_rows_display)
    st.subheader("Metric comparison")

    def highlight_max(row: pd.Series) -> list[str]:
        try:
            numeric = pd.to_numeric(row, errors="coerce")
        except Exception:
            numeric = row
        max_val = numeric.max(skipna=True) if hasattr(numeric, "max") else None
        styles = []
        for value in numeric:
            if max_val is not None and pd.notna(value) and value == max_val:
                styles.append("background-color: rgba(123, 211, 137, 0.45); color: #0F172A; font-weight:600;")
            else:
                styles.append("")
        return styles

    styled = (
        df_comparison.set_index("Metric")
        .style
        .apply(highlight_max, axis=1)
    )
    st.dataframe(styled, use_container_width=True)

st.divider()

# Summary scores comparison
summary_rows_display: List[Dict[str, object]] = []
summary_rows_numeric: List[Dict[str, Optional[float]]] = []
for key, label in SUMMARY_LABELS.items():
    row_display: Dict[str, object] = {"Summary": label}
    row_numeric: Dict[str, Optional[float]] = {"Summary": label}
    for player_row in selected_rows:
        player_name = player_row.get("player", "Player")
        raw_value = safe_float(player_row.get(key))
        row_display[player_name] = display_value(player_row.get(key))
        row_numeric[player_name] = raw_value
    summary_rows_display.append(row_display)
    summary_rows_numeric.append(row_numeric)

df_summary_numeric = (
    pd.DataFrame(summary_rows_numeric).set_index("Summary")
    if summary_rows_numeric
    else pd.DataFrame()
)

if summary_rows_display:
    df_summary = pd.DataFrame(summary_rows_display)
    st.subheader("Aggregated summary scores")
    def highlight_summary_max(row: pd.Series) -> list[str]:
        try:
            numeric = pd.to_numeric(row, errors="coerce")
        except Exception:
            numeric = row
        max_val = numeric.max(skipna=True) if hasattr(numeric, "max") else None
        styles = []
        for value in numeric:
            if max_val is not None and pd.notna(value) and value == max_val:
                styles.append("background-color: rgba(123, 211, 137, 0.45); color: #0F172A; font-weight:600;")
            else:
                styles.append("")
        return styles

    styled = (
        df_summary.set_index("Summary")
        .style
        .format(display_value)
        .apply(highlight_summary_max, axis=1)
    )
    st.dataframe(styled, use_container_width=True)

st.divider()

if "comparison_language" not in st.session_state:
    st.session_state["comparison_language"] = "English"

language_default = st.session_state.get("comparison_language", "English")
language_choice = st.radio(
    "Interpretation language",
    options=("English", "Français"),
    horizontal=True,
    index=("English", "Français").index(language_default),
    key="comparison_language",
)


st.subheader("Interpretation")


def build_metric_highlights(table: pd.DataFrame) -> Dict[str, List[Dict[str, object]]]:
    highlights: Dict[str, List[Dict[str, object]]] = {}
    if table is None or table.empty:
        return highlights
    for metric, row in table.iterrows():
        row = row.dropna()
        if row.empty:
            continue
        sorted_vals = row.sort_values(ascending=False)
        best_player = sorted_vals.index[0]
        best_value = float(sorted_vals.iloc[0])
        runner_name = sorted_vals.index[1] if len(sorted_vals) > 1 else None
        gap = float(best_value - sorted_vals.iloc[1]) if runner_name is not None else None
        highlights.setdefault(best_player, []).append(
            {
                "metric": metric,
                "value": best_value,
                "runner": runner_name,
                "gap": gap,
            }
        )
    for player, items in highlights.items():
        highlights[player] = sorted(
            items,
            key=lambda item: item["gap"] if item["gap"] is not None else 0.0,
            reverse=True,
        )[:2]
    return highlights


metric_highlights = build_metric_highlights(df_comparison_numeric)


def scout_analyst_summary(
    players: List[pd.Series],
    metric_highlights: Dict[str, List[Dict[str, object]]],
    context: str,
    language: str,
) -> str:
    if not players:
        return "No players selected to compare."

    player_names = [player.get("player", f"Player {idx + 1}") for idx, player in enumerate(players)]
    lines: List[str] = []

    if language == "Français":
        lines.append(
            "Coach, l'analyse statistique dans le contexte {} met en lumière les axes suivants pour {} :".format(
                context.lower(),
                ", ".join(player_names),
            )
        )
    else:
        lines.append(
            "Coach, within the {} frame the numbers carve out the following edges for {}:".format(
                context.lower(),
                ", ".join(player_names),
            )
        )

    if language == "Français":
        openings_pool = deque([
            "impose son influence via",
            "prend l'ascendant grâce à",
            "se démarque par",
            "met en évidence une constance supérieure sur",
            "affiche un volume stabilisé sur",
        ])
        summary_pool = deque([
            "L'indicateur composite **{}** reste orienté vers {} ({:.1f}).",
            "Sur l'agrégat **{}**, {} affiche toujours la note la plus élevée ({:.1f}).",
            "Le score combiné **{}** confirme la supériorité de {} ({:.1f}).",
            "La mesure consolidée **{}** maintient {} en tête ({:.1f}).",
            "La synthèse globale **{}** situe {} un cran au-dessus ({:.1f}).",
        ])
    else:
        openings_pool = deque([
            "dictates the tempo through",
            "continues to separate himself with",
            "builds his advantage around",
            "shows a steadier level on",
            "keeps an elevated profile across",
        ])
        summary_pool = deque([
            "Composite indicator **{}** stays tilted toward {} ({:.1f}).",
            "The **{}** aggregate continues to favour {} ({:.1f}).",
            "The combined score **{}** underlines {}'s stronger output ({:.1f}).",
            "The consolidated index **{}** keeps pointing to {} ({:.1f}).",
            "The blended summary **{}** still sets {} above the rest ({:.1f}).",
        ])

    openings_fallback = openings_pool[-1] if openings_pool else ""
    summary_fallback = summary_pool[-1] if summary_pool else ""

    for player_name in player_names:
        strengths = metric_highlights.get(player_name, [])
        if not strengths:
            continue
        descriptors = []
        for item in strengths:
            metric = item["metric"]
            value = item["value"]
            runner = item["runner"]
            gap = item["gap"]
            if language == "Français":
                if runner and gap is not None:
                    descriptors.append(
                        f"{metric} ({value:.1f}, +{gap:.1f} vs {runner})"
                    )
                else:
                    descriptors.append(f"{metric} ({value:.1f})")
            else:
                if runner and gap is not None:
                    descriptors.append(
                        f"{metric} ({value:.1f}, +{gap:.1f} vs {runner})"
                    )
                else:
                    descriptors.append(f"{metric} ({value:.1f})")
        if descriptors:
            phrase = openings_pool.popleft() if openings_pool else openings_fallback
            lines.append(f"- **{player_name}** {phrase} {', '.join(descriptors)}.")

    if not df_summary_numeric.empty:
        for summary_label, row in df_summary_numeric.iterrows():
            row = row.dropna()
            if row.empty:
                continue
            leader = row.idxmax()
            best_value = row[leader]
            sentence_template = summary_pool.popleft() if summary_pool else summary_fallback
            sentence = sentence_template.format(summary_label, leader, best_value)
            lines.append(f"- {sentence}")

    if language == "Français":
        lines.append(
            "Ces observations reposent sur les données disponibles ; il reste essentiel de les croiser avec le contexte tactique, la qualité de l'opposition et l'analyse vidéo pour confirmer la pertinence du profil."
        )
    else:
        lines.append(
            "These takeaways stem from the available data; blending them with tactical context, opposition level, and video remains essential before validating fit."
        )

    return "\n".join(lines)


interpretation_text = scout_analyst_summary(
    selected_rows,
    metric_highlights,
    context_choice,
    language_choice,
)
st.markdown(interpretation_text)


def scouting_report_paragraph(
    players: List[pd.Series],
    language: str,
    context: str,
    league: Optional[str],
    metric_highlights: Dict[str, List[Dict[str, object]]],
) -> str:
    names = [player.get("player", f"Player {idx + 1}") for idx, player in enumerate(players)]
    fragments: List[str] = []

    if language == "Français":
        lead_pool = deque([
            "affiche actuellement une maîtrise notable sur",
            "présente des repères supérieurs sur",
            "se situe à un niveau intéressant sur",
            "maintient un rendement solide sur",
            "témoigne d'une constance appréciable sur",
        ])
        summary_pool = deque([
            "sur l'agrégat {}, {} demeure le point de repère ({:.1f})",
            "sur l'indicateur composite {}, {} affiche pour l'instant la meilleure note ({:.1f})",
            "sur la synthèse statistique {}, {} se stabilise au-dessus de la concurrence ({:.1f})",
            "sur l'index consolidé {}, {} conserve un niveau supérieur ({:.1f})",
            "sur la moyenne agrégée {}, {} se maintient en tête ({:.1f})",
        ])
    else:
        lead_pool = deque([
            "currently sustains higher levels on",
            "produces reassuring signals across",
            "operates above the peer baseline on",
            "delivers steady outputs across",
            "keeps a dependable profile on",
        ])
        summary_pool = deque([
            "on the {} aggregate, {} remains the current reference ({:.1f})",
            "for the composite metric {}, {} presently holds the highest mark ({:.1f})",
            "across the summary score {}, {} stabilises above the comparison set ({:.1f})",
            "on the merged index {}, {} maintains the top tier ({:.1f})",
            "out of the blended benchmark {}, {} retains the upper profile ({:.1f})",
        ])

    lead_fallback = lead_pool[-1] if lead_pool else ""
    summary_fallback = summary_pool[-1] if summary_pool else ""

    for player_name in names:
        strengths = metric_highlights.get(player_name, [])
        if not strengths:
            continue
        descriptors = []
        for item in strengths:
            metric = item["metric"]
            value = item["value"]
            runner = item["runner"]
            gap = item["gap"]
            if runner and gap is not None:
                descriptors.append(f"{metric} ({value:.1f}, +{gap:.1f} vs {runner})")
            else:
                descriptors.append(f"{metric} ({value:.1f})")
        if descriptors:
            phrase = lead_pool.popleft() if lead_pool else lead_fallback
            fragments.append(f"{player_name} {phrase} {', '.join(descriptors)}")

    if not df_summary_numeric.empty:
        for summary_label, row in df_summary_numeric.iterrows():
            row = row.dropna()
            if row.empty:
                continue
            leader = row.idxmax()
            best_value = row[leader]
            sentence = summary_pool.popleft() if summary_pool else summary_fallback
            fragments.append(sentence.format(summary_label, leader, best_value))

    if language == "Français":
        intro_templates = [
            "Dans le contexte {context}{league}, {players} offrent un panorama nuancé des indicateurs de performance.",
            "À l'échelle {context}{league}, {players} dessinent un profil statistique révélateur pour affiner la prise de décision.",
            "En se concentrant sur le contexte {context}{league}, {players} proposent des repères chiffrés utiles à la préparation du projet de jeu.",
        ]
        intro = random.choice(intro_templates).format(
            context=context.lower(),
            league=f' ({league})' if league else "",
            players=", ".join(names),
        )
        closing = (
            "Les statistiques confirment les instincts de lecture mais ne doivent pas supplanter l'analyse vidéo. "
            "Un suivi complémentaire est recommandé pour valider la répétabilité de ces performances et leur adéquation au projet de jeu."
        )
    else:
        intro_templates = [
            "Within the {context}{league} frame, {players} deliver a layered statistical picture for elite planning.",
            "Focusing on the {context}{league} angle, {players} reveal data-driven cues that sharpen tactical foresight.",
            "In the {context}{league} context, {players} outline a set of metrics worth integrating into the scouting narrative.",
        ]
        intro = random.choice(intro_templates).format(
            context=context.lower(),
            league=f' ({league})' if league else "",
            players=", ".join(names),
        )
        closing = (
            "The data supports the scouting impressions yet remains a reference point rather than a final verdict. "
            "Further live/video assessment is advised to confirm sustainability and tactical fit."
        )

    body = ""
    if fragments:
        if language == "Français":
            linkers = ["D'un point de vue chiffré,", "À la lecture des tendances,", "En synthèse,"]
        else:
            linkers = ["From a statistical angle,", "Interpreting the trends,", "In summary,"]
        body = f"{random.choice(linkers)} " + " ".join(f"{frag}." for frag in fragments)

    narrative = intro
    if body:
        narrative += "\n\n" + body
    narrative += "\n\n" + closing
    return narrative


report_paragraph = scouting_report_paragraph(
    selected_rows,
    language_choice,
    context_choice,
    selected_rows[0].get("competition_name") if selected_rows else None,
    metric_highlights,
)
st.markdown(report_paragraph)
