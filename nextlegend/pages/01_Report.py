"""Streamlit Report page for NextLegend."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from mplsoccer import Radar

PAGES_DIR = Path(__file__).resolve().parent
ROOT_DIR = PAGES_DIR.parent
DATA_PATH = ROOT_DIR / "data" / "wyscout_players_cleaned.csv"
SCORES_PATH = ROOT_DIR / "players_scores.csv"
PLACEHOLDER_IMG = "https://placehold.co/300x400?text=No+Photo"
PCT_SUFFIX_LEAGUE = "_pct_league"
PCT_SUFFIX_GLOBAL = "_pct_global"
DEFAULT_METRICS = [
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
BLACKLIST_METRICS = {
    "yellow_cards",
    "yellow_cards_per_90",
    "red_cards",
    "red_cards_per_90",
    "fouls",
    "fouls_per_90",
    "conversion_rate",
}
KEY_COLUMNS = ["player", "team", "competition_name", "calendar"]


def safe_float(value: object) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


@st.cache_data(show_spinner=False)
def load_players() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col].replace({"-": np.nan, "": np.nan}), errors="ignore")
    return df


@st.cache_data(show_spinner=False)
def load_profile_scores() -> Optional[pd.DataFrame]:
    try:
        scores = pd.read_csv(SCORES_PATH)
    except FileNotFoundError:
        return None
    for col in scores.select_dtypes(include="object").columns:
        scores[col] = scores[col].astype(str).str.strip()
    return scores


def prepare_dataset() -> tuple[pd.DataFrame, List[str]]:
    df = load_players()
    scores = load_profile_scores()
    score_columns: List[str] = []

    if scores is not None:
        score_columns = [col for col in scores.columns if col not in KEY_COLUMNS]
        missing_cols = [col for col in score_columns if col not in df.columns]
        if missing_cols:
            join_cols = [col for col in KEY_COLUMNS if col in df.columns and col in scores.columns]
            if join_cols:
                df = df.merge(scores[join_cols + score_columns], on=join_cols, how="left")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    heuristic_cols = [col for col in numeric_cols if " - " in col or col in score_columns]
    role_columns = sorted(set(score_columns) | set(heuristic_cols))
    return df, role_columns


def get_percentile_columns(df: pd.DataFrame, suffix: str) -> List[str]:
    return [col for col in df.columns if col.endswith(suffix)]


def pick_top_k_percentiles(
    row: pd.Series,
    cols: Sequence[str],
    k: int = 8,
    blacklist: Optional[Iterable[str]] = None,
) -> tuple[List[str], List[float]]:
    blacklist = set(blacklist or [])
    scored: list[tuple[str, float]] = []
    for col in cols:
        if any(term in col for term in blacklist):
            continue
        value = safe_float(row.get(col))
        if value is None:
            continue
        scored.append((col, value))
    scored.sort(key=lambda item: item[1], reverse=True)
    top = scored[:k]
    return [item[0] for item in top], [item[1] for item in top]


def render_player_header(row: pd.Series, container: st.delta_generator.DeltaGenerator) -> None:
    container.image(PLACEHOLDER_IMG, width=140)
    container.markdown(f"### {row.get('player', 'Unknown player')}")

    info_pairs = [
        ("Club", row.get("team")),
        ("Country", row.get("birth_country")),
        ("Age", row.get("age")),
        ("Birth year", row.get("birth_year")),
        (
            "Position",
            ", ".join(
                str(val)
                for val in [row.get("position"), row.get("second_position")]
                if val not in (None, "", np.nan)
            )
            or "—",
        ),
        ("Matches", row.get("matches_played")),
        ("Minutes", row.get("minutes_played")),
    ]

    cols = container.columns(2)
    for idx, (label, value) in enumerate(info_pairs):
        maybe = value if value not in (None, "", np.nan) else "—"
        cols[idx % 2].markdown(f"**{label}:** {maybe}")


def format_metric_label(metric: str) -> str:
    radar_stats_map = {
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

    base_metric = metric.replace(PCT_SUFFIX_LEAGUE, "").replace(PCT_SUFFIX_GLOBAL, "")
    label = radar_stats_map.get(base_metric)
    if label:
        return label
    label = base_metric.replace("_per_90", "/90").replace("_percent", "%")
    label = label.replace("_", " ")
    return label.title()


def render_radar(row: pd.Series, container: st.delta_generator.DeltaGenerator, df: pd.DataFrame) -> None:
    percentile_cols_league = get_percentile_columns(df, PCT_SUFFIX_LEAGUE)
    percentile_cols_global = get_percentile_columns(df, PCT_SUFFIX_GLOBAL)

    selected_cols = []
    selected_vals = []
    title_suffix = "league"

    for metric in DEFAULT_METRICS:
        league_col = f"{metric}{PCT_SUFFIX_LEAGUE}" if metric + PCT_SUFFIX_LEAGUE in df.columns else None
        global_col = f"{metric}{PCT_SUFFIX_GLOBAL}" if metric + PCT_SUFFIX_GLOBAL in df.columns else None

        value = None
        if league_col in percentile_cols_league:
            value = safe_float(row.get(league_col))
            title_suffix = "league"
        if value is None and global_col in percentile_cols_global:
            value = safe_float(row.get(global_col))
            if value is not None:
                title_suffix = "global"

        if value is None:
            continue

        selected_cols.append(metric)
        selected_vals.append(np.clip(value, 0, 100))

    if not selected_cols:
        container.info("No percentile data available to build the radar chart.")
        return

    labels = [format_metric_label(col) for col in selected_cols]
    values = selected_vals

    radar = Radar(params=labels, min_range=[0] * len(labels), max_range=[100] * len(labels), round_int=[True] * len(labels))

    fig = plt.figure(figsize=(6, 6), facecolor="#0F172A")
    ax = fig.add_subplot(111)
    radar.setup_axis(
        ax=ax,
        facecolor="#0F172A",
        title=dict(
            title=f"Percentiles ({title_suffix}) — {row.get('player', 'Player')}",
            color="#E2E8F0",
            size=14,
        ),
        subtitle=dict(title="", color="#E2E8F0"),
    )

    radar.draw_circles(ax=ax, facecolor="#1E293B", edgecolor="#475569")
    radar.draw_radar(
        values,
        ax=ax,
        kwargs_radar={"facecolor": "#7BD389", "edgecolor": "#448361", "alpha": 0.55},
        kwargs_rings={"facecolor": "#7BD389", "alpha": 0.1},
    )
    radar.draw_range_labels(ax=ax, fontsize=8, color="#94A3B8")
    radar.draw_param_labels(ax=ax, fontsize=9, color="#E2E8F0")

    container.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_role_ranking(
    row: pd.Series,
    role_cols: Sequence[str],
    assigned_role: Optional[str] = None,
    role_pct_league: Optional[float] = None,
    role_pct_global: Optional[float] = None,
) -> None:
    role_cols = [col for col in role_cols if col in row.index]
    if not role_cols:
        st.info("No role score columns found.")
        return

    rows: list[tuple[str, float]] = []
    for col in role_cols:
        value = safe_float(row.get(col))
        if value is None:
            continue
        rows.append((col, value))

    if not rows:
        st.info("No role scores available for this player.")
        return

    rows.sort(key=lambda item: item[1], reverse=True)
    top_rows = rows[:10]

    data = pd.DataFrame(top_rows, columns=["Role", "Score"])
    data["Role"] = data["Role"].str.replace("_", " ")

    st.markdown("### Role fit (profiles)")
    league_display = f"{role_pct_league:.1f}" if role_pct_league is not None else "—"
    global_display = f"{role_pct_global:.1f}" if role_pct_global is not None else "—"
    st.caption(f"Assigned role percentiles — League: {league_display} | Global: {global_display}")

    assigned_display = assigned_role.replace("_", " ") if assigned_role else None
    if assigned_display:
        st.markdown(f"✅ **Assigned role**: {assigned_display}")

    style = data.style.format({"Score": "{:.1f}"})
    if assigned_display:
        def highlight_role(val: str) -> str:
            return "font-weight: bold; color: #7BD389" if val == assigned_display else ""
        style = style.applymap(highlight_role, subset=["Role"])

    st.dataframe(style, hide_index=True, use_container_width=True)


st.set_page_config(page_title="Report", layout="wide", initial_sidebar_state="collapsed")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
df, role_columns = prepare_dataset()
if df.empty:
    st.warning("Player dataset is empty. Please refresh the data pipeline.")
    st.stop()

# ---------------------------------------------------------------------------
# Header filters (league → team → player)
# ---------------------------------------------------------------------------
st.session_state.setdefault("report_prev_league", None)
st.session_state.setdefault("report_team", None)
st.session_state.setdefault("report_player", None)

leagues = sorted(df["competition_name"].dropna().unique())
if not leagues:
    st.info("No leagues available in the dataset.")
    st.stop()

col_l, col_t, col_p = st.columns(3)
selected_league = col_l.selectbox("Select a league", options=leagues)

if st.session_state["report_prev_league"] != selected_league:
    st.session_state["report_team"] = None
    st.session_state["report_player"] = None
st.session_state["report_prev_league"] = selected_league

league_mask = df["competition_name"] == selected_league
teams = sorted(df.loc[league_mask, "team"].dropna().unique())
if not teams:
    st.info("No teams available for this league.")
    st.stop()

current_team = st.session_state.get("report_team")
if current_team not in teams:
    current_team = teams[0]
selected_team = col_t.selectbox("Select a team", options=teams, index=teams.index(current_team))
if selected_team != st.session_state.get("report_team"):
    st.session_state["report_team"] = selected_team
    st.session_state["report_player"] = None

team_mask = league_mask & (df["team"] == selected_team)
players = sorted(df.loc[team_mask, "player"].dropna().unique())
if not players:
    st.info("No players available for this team.")
    st.stop()

current_player = st.session_state.get("report_player")
if current_player not in players:
    current_player = players[0]
selected_player = col_p.selectbox("Select a player", options=players, index=players.index(current_player))
st.session_state["report_player"] = selected_player

if not selected_player:
    st.info("Select a player to display the scouting report.")
    st.stop()

# ---------------------------------------------------------------------------
# Player selection
# ---------------------------------------------------------------------------
player_rows = df[team_mask & (df["player"] == selected_player)].copy()
if player_rows.empty:
    st.info("No data found for the selected player.")
    st.stop()

if "calendar" in player_rows.columns:
    player_rows = player_rows.sort_values(by="calendar", ascending=False, na_position="last")
row = player_rows.iloc[0]

assigned_role = row.get("assigned_role")
role_pct_league = safe_float(row.get("assigned_role_pct_league"))
role_pct_global = safe_float(row.get("assigned_role_pct_global"))

left, right = st.columns([1, 1.2])
render_player_header(row, left)
render_radar(row, right, df)

st.divider()
render_role_ranking(
    row,
    role_columns,
    assigned_role=assigned_role,
    role_pct_league=role_pct_league,
    role_pct_global=role_pct_global,
)
