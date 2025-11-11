"""Stats Research page for NextLegend."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from auth import render_account_controls, require_authentication
from components.sidebar import render_sidebar_logo
from scripts.positions_glossary import positions_glossary
from s3_utils import read_csv_from_s3

PAGES_DIR = Path(__file__).resolve().parent
ROOT_DIR = PAGES_DIR.parent
DATA_KEY = "data/wyscout_players_cleaned.csv"
BIG5_PATH = ROOT_DIR / "big_5_leagues.txt"
ROLES_METRICS_PATH = ROOT_DIR / "roles_metrics.json"
PLAYER_PROFILES_PATH = ROOT_DIR / "player_profiles.json"


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
def load_big5_leagues() -> List[str]:
    if not BIG5_PATH.exists():
        return []
    return [
        line.strip()
        for line in BIG5_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@st.cache_data(show_spinner=False)
def load_metric_labels() -> dict[str, str]:
    if not ROLES_METRICS_PATH.exists():
        return {}
    try:
        data = json.loads(ROLES_METRICS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data.get("label_map_en", {}) or {}


@st.cache_data(show_spinner=False)
def load_lower_is_better_metrics() -> set[str]:
    path = PLAYER_PROFILES_PATH
    metrics: set[str] = set()
    if not path.exists():
        return metrics
    try:
        profiles = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return metrics
    for spec in profiles.values():
        metrics.update(spec.get("lower_is_better", []) or [])
    return metrics


def extract_positions(tokens: str | float | None) -> List[str]:
    if tokens is None or (isinstance(tokens, float) and np.isnan(tokens)):
        return []
    raw = str(tokens)
    return [part.strip() for part in raw.split(",") if part.strip() and part.strip().lower() != "nan"]


def add_position_tokens(df: pd.DataFrame) -> pd.DataFrame:
    positions_primary = df.get("position", pd.Series("", index=df.index))
    positions_secondary = df.get("second_position", pd.Series("", index=df.index))

    combined = []
    for prim, sec in zip(positions_primary, positions_secondary):
        tokens = extract_positions(prim) + extract_positions(sec)
        combined.append(list(dict.fromkeys(tokens)))
    df = df.copy()
    df["position_tokens"] = combined
    return df


def human_readable_position_options(df: pd.DataFrame) -> Tuple[Dict[str, str], List[str]]:
    tokens = set()
    for entry in df["position_tokens"]:
        tokens.update(entry)
    filtered = sorted({token for token in tokens if token and token.lower() != "nan"})
    mapping = {token: positions_glossary.get(token, token) for token in filtered}
    return mapping, [mapping[token] for token in filtered]


def filter_by_league(df: pd.DataFrame, selection: str, big5: List[str]) -> pd.DataFrame:
    if selection == "All leagues":
        return df
    if selection == "Big 5 Leagues":
        return df[df.get("competition_name", pd.Series(dtype=str)).isin(big5)]
    return df[df.get("competition_name", pd.Series(dtype=str)) == selection]


st.set_page_config(
    page_title="Stats Research",
    layout="wide",
    initial_sidebar_state="collapsed",
)
require_authentication()
render_sidebar_logo()
render_account_controls()

df_players = load_players()
if df_players.empty:
    st.warning("Player dataset is empty. Run the pipeline first.")
    st.stop()

df_players = add_position_tokens(df_players)
position_label_map, position_labels = human_readable_position_options(df_players)
position_options = ["All"] + position_labels
league_options = ["All leagues", "Big 5 Leagues"] + sorted(
    {str(val) for val in df_players.get("competition_name", pd.Series(dtype=str)).dropna().unique()}
)
big5_leagues = load_big5_leagues()
metric_label_map = load_metric_labels()
lower_is_better_metrics = load_lower_is_better_metrics()
metric_whitelist = lower_is_better_metrics.copy()
metric_whitelist.update(metric_label_map.keys())

try:
    profiles_raw = json.loads(PLAYER_PROFILES_PATH.read_text(encoding="utf-8"))
except (FileNotFoundError, json.JSONDecodeError):
    profiles_raw = {}

for spec in profiles_raw.values():
    metric_whitelist.update((spec.get("weights") or {}).keys())

st.title("Stats Research")

filter_row1 = st.columns(4)
selected_league = filter_row1[0].selectbox("Select league", options=league_options)
selected_positions = filter_row1[1].multiselect(
    "Positions",
    options=position_options,
    default=["All"],
)
valid_ages = pd.to_numeric(df_players.get("age", pd.Series(dtype=float)), errors="coerce")
if valid_ages.dropna().empty:
    age_range = filter_row1[2].slider(
        "Age range",
        min_value=15,
        max_value=40,
        value=(18, 32),
        step=1,
    )
else:
    min_age = int(np.floor(valid_ages.min()))
    max_age = int(np.ceil(valid_ages.max()))
    default_min = max(min_age, 18)
    default_max = min(max_age, 32) if default_min < min(max_age, 32) else max_age
    age_range = filter_row1[2].slider(
        "Age range",
        min_value=min_age,
        max_value=max_age,
        value=(default_min, default_max),
        step=1,
    )
min_minutes = filter_row1[3].number_input(
    "Minimum minutes played",
    min_value=0,
    max_value=5000,
    value=270,
    step=30,
)

numeric_columns = [
    col
    for col in df_players.columns
    if df_players[col].dtype in (np.float64, np.float32, np.int64, np.int32)
    and col in metric_whitelist
]


def metric_display_name(metric: str) -> str:
    if metric in metric_label_map:
        return metric_label_map[metric]
    cleaned = (
        metric.replace("_per_90", " per 90")
        .replace("_pct", " (%)")
        .replace("_percent", " (%)")
        .replace("_pct_league", " league percentile")
        .replace("_pct_global", " global percentile")
        .replace("_", " ")
    )
    return cleaned.title()


display_to_metric = {metric_display_name(col): col for col in numeric_columns}
sorted_display_metrics = sorted(display_to_metric.keys())

filter_row2 = st.columns(2)
metric_x_display = filter_row2[0].selectbox("Axis X", options=sorted_display_metrics, index=0)
metric_x = display_to_metric[metric_x_display]
metric_y_options_display = [label for label in sorted_display_metrics if display_to_metric[label] != metric_x]
metric_y_display = filter_row2[1].selectbox("Axis Y", options=metric_y_options_display, index=0)
metric_y = display_to_metric[metric_y_display]


def apply_filters(
    df: pd.DataFrame,
    league_choice: str,
    positions_choice: List[str],
    minutes_threshold: int,
    age_bounds: Tuple[int, int],
) -> pd.DataFrame:
    filtered = filter_by_league(df, league_choice, big5_leagues)
    if minutes_threshold > 0 and "minutes_played" in filtered.columns:
        filtered = filtered[pd.to_numeric(filtered["minutes_played"], errors="coerce") >= minutes_threshold]

    if "age" in filtered.columns and age_bounds:
        ages = pd.to_numeric(filtered["age"], errors="coerce")
        filtered = filtered[(ages >= age_bounds[0]) & (ages <= age_bounds[1])]

    if positions_choice and "All" not in positions_choice:
        selected_codes = {
            code for code, label in position_label_map.items() if label in positions_choice
        }

        def has_position(tokens: List[str]) -> bool:
            return any(token in selected_codes for token in tokens)

        filtered = filtered[filtered["position_tokens"].apply(has_position)]

    filtered = filtered.dropna(subset=[metric_x, metric_y])
    filtered = filtered[(filtered[metric_x] > 0) & (filtered[metric_y] > 0)]
    return filtered


filtered_df = apply_filters(
    df_players,
    selected_league,
    selected_positions,
    int(min_minutes),
    tuple(age_range),
)

if filtered_df.empty:
    st.info("No players found with the current filters.")
    st.stop()


def compute_highlighted_subset(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if metric_x in lower_is_better_metrics or metric_y in lower_is_better_metrics:
        return df.head(0)  # avoid highlighting when one metric rewards low values

    ranks_x = df[metric_x].rank(pct=True)
    ranks_y = df[metric_y].rank(pct=True)

    mask = (ranks_x >= 0.85) & (ranks_y >= 0.5)
    highlight = df[mask].copy()

    idx_max_x = ranks_x.idxmax()
    if pd.notna(idx_max_x) and idx_max_x not in highlight.index:
        highlight = pd.concat([highlight, df.loc[[idx_max_x]]], axis=0)

    if highlight.empty:
        return highlight

    highlight = highlight.drop_duplicates(subset=["player", "team", "competition_name"])
    highlight = highlight.sort_values(by=[metric_x, metric_y], ascending=False)
    return highlight.head(15)


highlight_df = compute_highlighted_subset(filtered_df)
highlight_keys = {
    (row.player, row.get("team"), row.get("competition_name"))
    for _, row in highlight_df.iterrows()
}

chart_title = f"{metric_x_display} vs {metric_y_display} — {selected_league}"
if selected_positions and "All" not in selected_positions:
    chart_title += f" | Positions: {', '.join(selected_positions)}"
chart_title += f" | Min {int(min_minutes)} mins"


def build_scatter() -> px.scatter:
    fig = px.scatter(
        filtered_df,
        x=metric_x,
        y=metric_y,
        opacity=0.3,
        color_discrete_sequence=["#60A5FA"],
        hover_data={
            "player": True,
            "team": True,
            "competition_name": True,
            metric_x: ":.2f",
            metric_y: ":.2f",
        },
        labels={
            metric_x: metric_x_display,
            metric_y: metric_y_display,
            "player": "Player",
            "team": "Team",
            "competition_name": "League",
        },
        title=chart_title,
    )
    fig.update_traces(marker=dict(size=9))

    if not highlight_df.empty:
        highlight_trace = px.scatter(
            highlight_df,
            x=metric_x,
            y=metric_y,
            text="player",
            color_discrete_sequence=["#34D399"],
            hover_data={
                "player": True,
                "team": True,
                "competition_name": True,
                metric_x: ":.2f",
                metric_y: ":.2f",
            },
        )
        highlight_trace.update_traces(
            marker=dict(size=12, opacity=0.95, line=dict(color="#064E3B", width=1.4)),
            textposition="top center",
            textfont=dict(color="#F8FAFC", size=11),
            showlegend=False,
        )
        fig.add_traces(highlight_trace.data)

    fig.update_layout(
        plot_bgcolor="#0F172A",
        paper_bgcolor="#0F172A",
        font=dict(color="#E2E8F0"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
    )
    return fig


scatter_fig = build_scatter()
st.plotly_chart(scatter_fig, use_container_width=True)

st.divider()

table_df = filtered_df.copy()
table_df["Key"] = list(
    zip(
        table_df["player"].fillna("Unknown"),
        table_df.get("team", pd.Series("Unknown", index=table_df.index)),
        table_df.get("competition_name", pd.Series("Unknown", index=table_df.index)),
    )
)
table_df["Player — Club — League"] = (
    table_df["player"].fillna("Unknown")
    + " — "
    + table_df.get("team", pd.Series("Unknown", index=table_df.index))
    + " — "
    + table_df.get("competition_name", pd.Series("Unknown", index=table_df.index))
)

display_table = (
    table_df.sort_values(by=[metric_y, metric_x], ascending=False)[
        ["Player — Club — League", metric_x, metric_y, "Key"]
    ]
)
display_table = display_table.rename(columns={metric_x: metric_x_display, metric_y: metric_y_display})
display_table = display_table.drop(columns=["Key"])


def highlight_table_row(row: pd.Series) -> List[str]:
    key = (
        row["Player — Club — League"].split(" — ")
        if isinstance(row["Player — Club — League"], str)
        else []
    )
    key_tuple: Optional[Tuple[str, str, str]] = None
    if len(key) == 3:
        key_tuple = (key[0], key[1], key[2])
    if key_tuple in highlight_keys:
        styles = [""] * len(row)
        styles[0] = "background-color: rgba(52, 211, 153, 0.35); font-weight:600; color:#0F172A;"
        return styles
    return [""] * len(row)


styled_table = (
    display_table.style
    .format({metric_x_display: "{:.2f}", metric_y_display: "{:.2f}"})
    .apply(highlight_table_row, axis=1)
)

st.dataframe(styled_table, use_container_width=True, hide_index=True)
