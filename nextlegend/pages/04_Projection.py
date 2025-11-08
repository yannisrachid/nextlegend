"""Projection page for NextLegend."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from mplsoccer import Radar

from auth import render_account_controls, require_authentication
from components.sidebar import render_sidebar_logo
from scripts.positions_glossary import positions_glossary
from s3_utils import read_csv_from_s3

PAGES_DIR = Path(__file__).resolve().parent
ROOT_DIR = PAGES_DIR.parent
DATA_KEY = "data/wyscout_players_cleaned.csv"
SIMILARITY_KEY = "data/similarity"
LEAGUE_META_PATH = ROOT_DIR / "data" / "league_translation_meta.csv"
LEAGUE_MATRIX_PATH = ROOT_DIR / "data" / "league_translation_matrix.csv"
PLACEHOLDER_IMG = "https://placehold.co/120x120?text=No+Photo"
PCT_SUFFIX_LEAGUE = "_pct_league"
PCT_SUFFIX_GLOBAL = "_pct_global"
SUMMARY_LABELS = {
    "summary_finishing": "Finishing",
    "summary_aerial": "Aerial",
    "summary_defense": "Defence",
    "summary_technique": "Technique",
    "summary_creation": "Creation",
    "summary_construction": "Construction",
}

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


def load_csv(path: Path, columns: Optional[List[str]] = None) -> pd.DataFrame:
    if not path.exists():
        try:
            return read_csv_from_s3(str(path.relative_to(ROOT_DIR)))
        except FileNotFoundError:
            return pd.DataFrame(columns=columns if columns else [])
    return pd.read_csv(path)


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
    display = metric.replace("_per_90", " per 90").replace("_pct", " pct").replace("_", " ")
    display = display.replace(" per 90", " /90").replace(" pct", " (%)")
    return display.title()


def select_radar_metrics(role_name: Optional[str], dataset: pd.DataFrame, roles_map: dict[str, List[str]]) -> List[str]:
    metrics = roles_map.get(role_name, []) if role_name else []
    if metrics:
        filtered = [metric for metric in metrics if metric in dataset.columns]
        if len(filtered) >= 3:
            return filtered[:10]
    fallback = [metric for metric in RADAR_FALLBACK_METRICS if metric in dataset.columns]
    return fallback[:10]


def gather_available_roles(row: pd.Series, roles_map: dict[str, List[str]]) -> List[str]:
    roles: list[str] = []
    assigned = str(row.get("assigned_role") or "").strip()
    if assigned and assigned.lower() != "nan":
        roles.append(assigned)

    def role_has_data(role: str) -> bool:
        candidates = [
            safe_float(row.get(role)),
            safe_float(row.get(f"{role}{PCT_SUFFIX_LEAGUE}")),
            safe_float(row.get(f"{role}{PCT_SUFFIX_GLOBAL}")),
        ]
        return any(value is not None for value in candidates)

    for role_name in roles_map.keys():
        if role_name in roles:
            continue
        if role_has_data(role_name):
            roles.append(role_name)
    return roles


def percentile_rank(series: pd.Series, value: Optional[float]) -> Optional[float]:
    """Compute percentile rank (0-100) of a value within a numeric series."""
    numeric_value = safe_float(value)
    if numeric_value is None:
        return None
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return None
    sorted_vals = np.sort(clean.to_numpy(dtype=float))
    position = np.searchsorted(sorted_vals, float(numeric_value), side="right")
    percentile = (position / len(sorted_vals)) * 100.0
    return float(np.clip(percentile, 0.0, 100.0))


def inject_projected_percentiles(
    projected_row: pd.Series,
    metrics: List[str],
    dataset: pd.DataFrame,
    target_league: Optional[str],
) -> pd.Series:
    """Update percentile columns of the projected row based on projected raw metrics."""
    comp_col = None
    if "competition_name" in dataset.columns:
        comp_col = "competition_name"
    elif "league" in dataset.columns:
        comp_col = "league"

    league_mask = None
    if comp_col and target_league:
        league_mask = dataset[comp_col].astype(str) == str(target_league)

    for metric in metrics:
        if metric not in dataset.columns:
            continue
        projected_value = safe_float(projected_row.get(metric))
        if projected_value is None:
            continue

        league_pct = None
        if league_mask is not None and league_mask.any():
            league_pct = percentile_rank(dataset.loc[league_mask, metric], projected_value)
        global_pct = percentile_rank(dataset[metric], projected_value)

        if league_pct is not None:
            projected_row[f"{metric}{PCT_SUFFIX_LEAGUE}"] = league_pct
        if global_pct is not None:
            projected_row[f"{metric}{PCT_SUFFIX_GLOBAL}"] = global_pct

    return projected_row


def project_percentile(base_percentile: Optional[float], coeff: Optional[float]) -> Optional[float]:
    value = safe_float(base_percentile)
    factor = safe_float(coeff)
    if value is None or factor is None:
        return None
    projected = value * factor
    return float(np.clip(projected, 0.0, 100.0))


def render_radar(
    row: pd.Series,
    container: st.delta_generator.DeltaGenerator,
    dataset: pd.DataFrame,
    metrics: List[str],
    label_map: dict[str, str],
    use_percentiles: bool,
    context: str,
    player_league: Optional[str],
) -> None:
    if len(metrics) < 3:
        container.info("Not enough metrics to render the radar chart.")
        return

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
        for metric in metrics:
            display = metric_display_name(metric, label_map)
            value = safe_float(row.get(f"{metric}{preferred_suffix}"))
            if value is None:
                value = safe_float(row.get(f"{metric}{fallback_suffix}"))
            if value is None:
                continue
            selected_labels.append(display)
            selected_values.append(np.clip(value, 0, 100))
        min_range = [0.0] * len(selected_labels)
        max_range = [100.0] * len(selected_labels)
        round_flags = [True] * len(selected_labels)
    else:
        for metric in metrics:
            display = metric_display_name(metric, label_map)
            raw_value = safe_float(row.get(metric))
            if raw_value is None:
                continue
            if metric.endswith("percent") or metric.endswith("_pct"):
                raw_value = float(np.clip(raw_value, 0.0, 100.0))
            selected_labels.append(display)
            selected_values.append(raw_value)

            series = pd.to_numeric(dataset.get(metric, pd.Series(dtype=float)), errors="coerce")
            if metric.endswith("percent") or metric.endswith("_pct"):
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
            if metric.endswith("percent") or metric.endswith("_pct"):
                min_adj = max(0.0, min_adj)
                max_adj = min(100.0, max_adj)
                if max_adj - min_adj < 1e-6:
                    max_adj = min(100.0, min_adj + 1.0)
            min_range.append(min_adj)
            max_range.append(max_adj)
            round_flags.append(False)

    if len(selected_labels) < 3:
        container.info("Not enough metrics to render the radar chart.")
        return

    radar = Radar(
        params=selected_labels,
        min_range=min_range if not use_percentiles else [0] * len(selected_labels),
        max_range=max_range if not use_percentiles else [100] * len(selected_labels),
        round_int=round_flags if not use_percentiles else [True] * len(selected_labels),
    )

    fig = plt.figure(figsize=(4.5, 4.5), facecolor="#0F172A")
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
    radar.draw_radar(
        selected_values,
        ax=ax,
        kwargs_radar={"facecolor": "#7BD389" if use_percentiles else "none", "edgecolor": "#448361", "alpha": 0.55},
        kwargs_rings={"facecolor": "#7BD389" if use_percentiles else "none", "alpha": 0.1},
    )
    radar.draw_range_labels(ax=ax, fontsize=8, color="#94A3B8")
    radar.draw_param_labels(ax=ax, fontsize=8, color="#E2E8F0")

    container.pyplot(fig, use_container_width=True)
    plt.close(fig)


def load_league_meta() -> pd.DataFrame:
    meta = load_csv(LEAGUE_META_PATH, columns=["competition", "difficulty", "intensity"])
    return meta


def load_league_matrix() -> pd.DataFrame:
    matrix = load_csv(
        LEAGUE_MATRIX_PATH,
        columns=["source_competition", "target_competition", "difficulty_coeff", "intensity_coeff", "overall_coeff"],
    )
    return matrix


def project_metric(value: Optional[float], coeff: Optional[float]) -> Optional[float]:
    if value is None or coeff is None:
        return None
    return value * coeff


def get_translation_coeff(
    matrix: pd.DataFrame,
    source_league: Optional[str],
    target_league: Optional[str],
) -> float:
    if not source_league or not target_league:
        return 1.0
    subset = matrix[
        (matrix["source_competition"].astype(str) == str(source_league))
        & (matrix["target_competition"].astype(str) == str(target_league))
    ]
    coeff = safe_float(subset.iloc[0].get("overall_coeff")) if not subset.empty else None
    return coeff if coeff is not None else 1.0


def project_player_metrics(
    player_row: pd.Series,
    target_league: str,
    matrix: pd.DataFrame,
    metrics: List[str],
) -> Dict[str, Optional[float]]:
    source_league = player_row.get("competition_name")
    if not source_league:
        return {metric: None for metric in metrics}

    coeff = get_translation_coeff(matrix, source_league, target_league)

    projected = {}
    for metric in metrics:
        projected[metric] = project_metric(safe_float(player_row.get(metric)), coeff)
    return projected


def project_summary_scores(
    player_row: pd.Series,
    target_league: str,
    matrix: pd.DataFrame,
) -> Dict[str, Optional[float]]:
    source_league = player_row.get("competition_name")
    if not source_league:
        return {key: None for key in SUMMARY_LABELS.keys()}

    coeff = get_translation_coeff(matrix, source_league, target_league)

    projected = {}
    for key in SUMMARY_LABELS.keys():
        projected[key] = project_metric(safe_float(player_row.get(key)), coeff)
    return projected


st.set_page_config(page_title="Projection", layout="wide", initial_sidebar_state="collapsed")
require_authentication()
render_sidebar_logo()
render_account_controls()

df_players = load_players()
if df_players.empty:
    st.warning("Player dataset is empty. Run the pipeline first.")
    st.stop()

roles_map, label_map = load_role_metrics()
meta_df = load_league_meta()
matrix_df = load_league_matrix()

st.title("Projection")

comp_column = "competition_name" if "competition_name" in df_players.columns else None
if comp_column is None:
    st.error("Competition column not found in dataset.")
    st.stop()

league_options = sorted({str(val) for val in df_players[comp_column].dropna().unique()})
team_column = "team_in_selected_period" if "team_in_selected_period" in df_players.columns else "team"

col_left, col_right = st.columns(2)

with col_left:
    selected_league = st.selectbox("Select league", options=league_options)
    league_filtered = df_players[df_players[comp_column].astype(str) == str(selected_league)]

    team_options = sorted({str(team) for team in league_filtered.get(team_column, pd.Series(dtype=str)).dropna().unique()})
    selected_team = st.selectbox("Select club", options=team_options)

    player_filtered = league_filtered[league_filtered.get(team_column).astype(str) == str(selected_team)]
    player_options = sorted({str(name) for name in player_filtered.get("player", pd.Series(dtype=str)).dropna().unique()})
    selected_player = st.selectbox("Select player", options=player_options)

with col_right:
    target_leagues = [league for league in league_options if league != selected_league]
    target_league = st.selectbox("Projection league", options=target_leagues)

percentile_view = st.checkbox("Percentile view", value=True)

if not selected_player:
    st.stop()

player_row = player_filtered[player_filtered.get("player").astype(str) == str(selected_player)]
if player_row.empty:
    st.warning("Player could not be found in the dataset.")
    st.stop()

player_row = player_row.sort_values(by=["minutes_played"], ascending=False).iloc[0]

available_roles = gather_available_roles(player_row, roles_map)
role_select_key = "projection_role_choice"

if st.session_state.get("projection_prev_player") != selected_player:
    st.session_state["projection_prev_player"] = selected_player
    st.session_state.pop(role_select_key, None)

player_header = st.container()
with player_header:
    st.markdown(f"### {selected_player}")
    if available_roles:
        default_role = available_roles[0]
        if role_select_key not in st.session_state:
            st.session_state[role_select_key] = default_role
        selected_role = st.selectbox(
            "Role for analysis",
            available_roles,
            key=role_select_key,
            help="Choose the role context used for projections and radar metrics.",
        )
        st.caption(f"Analysing projections for role: **{selected_role}**")
    else:
        selected_role = str(player_row.get("assigned_role") or "").strip() or None
        st.info("No alternative role data available; projections use the default role.")

metrics = select_radar_metrics(selected_role, df_players, roles_map)

translation_coeff = get_translation_coeff(
    matrix_df,
    player_row.get("competition_name"),
    target_league,
)

projected_metrics = project_player_metrics(player_row, target_league, matrix_df, metrics)

layout_main = st.columns([0.45, 0.1, 0.45])

with layout_main[0]:
    st.subheader(selected_league)
    render_radar(
        player_row,
        layout_main[0],
        df_players,
        metrics,
        label_map,
        percentile_view,
        "League" if percentile_view else "Raw",
        player_row.get("competition_name"),
    )

with layout_main[1]:
    st.markdown(
        "<div style='text-align:center; font-size:3rem; color:#7BD389; padding-top:3rem;'>→</div>",
        unsafe_allow_html=True,
    )

with layout_main[2]:
    st.subheader(target_league)
    projected_series = player_row.copy()
    for metric, value in projected_metrics.items():
        if value is None:
            continue
        projected_series[metric] = value
    projected_series = inject_projected_percentiles(
        projected_series,
        metrics,
        df_players,
        target_league,
    )
    if selected_role:
        role_league_col = f"{selected_role}{PCT_SUFFIX_LEAGUE}"
        role_global_col = f"{selected_role}{PCT_SUFFIX_GLOBAL}"
        projected_role_league_pct = project_percentile(
            player_row.get(role_league_col),
            translation_coeff,
        )
        projected_role_global_pct = project_percentile(
            player_row.get(role_global_col),
            translation_coeff,
        )
        if projected_role_league_pct is not None:
            projected_series[role_league_col] = projected_role_league_pct
        if projected_role_global_pct is not None:
            projected_series[role_global_col] = projected_role_global_pct
    render_radar(
        projected_series,
        layout_main[2],
        df_players,
        metrics,
        label_map,
        percentile_view,
        "League" if percentile_view else "Raw",
        target_league,
    )

st.divider()

summary_actual = {key: safe_float(player_row.get(key)) for key in SUMMARY_LABELS.keys()}
summary_projected = project_summary_scores(player_row, target_league, matrix_df)

summary_rows = []
for key, label in SUMMARY_LABELS.items():
    summary_rows.append(
        {
            "Summary": label,
            selected_league: display_value(summary_actual.get(key)),
            target_league: display_value(summary_projected.get(key)),
        }
    )

if summary_rows:
    df_summary = pd.DataFrame(summary_rows)
    st.subheader("Aggregated summary scores")
    st.dataframe(df_summary.set_index("Summary"), use_container_width=True)
    st.divider()
    if selected_role:
        st.subheader(f"Projected percentiles (estimated) — {selected_role}")

        role_league_col = f"{selected_role}{PCT_SUFFIX_LEAGUE}"
        role_global_col = f"{selected_role}{PCT_SUFFIX_GLOBAL}"

        league_pct_base = safe_float(player_row.get(role_league_col))
        league_pct_projected = safe_float(projected_series.get(role_league_col))

        if league_pct_base is not None and league_pct_projected is not None:
            col_l1, col_l2, col_l3 = st.columns([1, 0.2, 1])
            col_l1.metric(
                f"{selected_league} league percentile",
                f"{league_pct_base:.0f}",
                help=f"Current percentile in {selected_league} for role {selected_role}.",
            )
            col_l2.markdown(
                "<div style='text-align:center; font-size:3rem; color:#7BD389; padding-top:0.6rem;'>➜</div>",
                unsafe_allow_html=True,
            )
            col_l3.metric(
                f"{target_league} league percentile (estimated)",
                f"{league_pct_projected:.0f}",
                help=f"Projected percentile in {target_league} using translation adjustments.",
            )

        global_pct_base = safe_float(player_row.get(role_global_col))
        global_pct_projected = safe_float(projected_series.get(role_global_col))

        if global_pct_base is not None and global_pct_projected is not None:
            col_g1, col_g2, col_g3 = st.columns([1, 0.2, 1])
            col_g1.metric(
                "Global percentile",
                f"{global_pct_base:.0f}",
                help=f"Current global percentile for role {selected_role}.",
            )
            col_g2.markdown(
                "<div style='text-align:center; font-size:3rem; color:#7BD389; padding-top:0.6rem;'>➜</div>",
                unsafe_allow_html=True,
            )
            col_g3.metric(
                "Global percentile (projected)",
                f"{global_pct_projected:.0f}",
                help="Projected global percentile after applying the translation coefficient.",
            )
