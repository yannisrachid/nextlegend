"""Streamlit Report page for NextLegend."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from mplsoccer import Radar

from s3_utils import read_csv_from_s3

PAGES_DIR = Path(__file__).resolve().parent
ROOT_DIR = PAGES_DIR.parent
DATA_KEY = "data/wyscout_players_cleaned.csv"
SCORES_KEY = "players_scores.csv"
SIM_PREFIX = "data/similarity"
BIG5_PATH = ROOT_DIR / "big_5_leagues.txt"
ROLE_METRICS_PATH = ROOT_DIR / "roles_metrics.json"
PLACEHOLDER_IMG = "https://placehold.co/160x160?text=No+Photo"
PCT_SUFFIX_LEAGUE = "_pct_league"
PCT_SUFFIX_GLOBAL = "_pct_global"

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

KEY_COLUMNS = ["player", "team", "team_in_selected_period", "competition_name", "calendar"]
ROLE_SCORE_SUFFIXES = ("_pct_global", "_pct_league")

CATEGORY_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("Goalkeeping", ("save", "conceded", "shots_against", "goals_prevented", "back_passes_to_gk", "aerial_duels_gk", "exits", "xg_against")),
    ("Shooting & Finishing", ("goal", "shot", "xg", "touches_in_penalty_area", "conversion")),
    ("Chance Creation", ("assist", "xa", "key_pass", "smart_pass", "deep_completion", "through_pass")),
    ("Crossing", ("cross",)),
    ("Possession & Progression", ("progressive", "carry", "dribble", "touch", "passes_received")),
    ("Passing", ("pass",)),
    ("Physical & Aerial", ("aerial", "headed", "height", "weight")),
    ("Pressing & Duels", ("press", "duel")),
    ("Defensive Actions", ("interception", "tackle", "clear", "block", "def_", "successful_def_actions", "marking")),
    ("Discipline", ("card", "foul")),
]

CATEGORY_ORDER = [name for name, _ in CATEGORY_RULES] + ["Other"]


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
def load_profile_scores() -> Optional[pd.DataFrame]:
    try:
        scores = read_csv_from_s3(SCORES_KEY)
    except FileNotFoundError:
        local_path = ROOT_DIR / "players_scores.csv"
        if not local_path.exists():
            return None
        scores = pd.read_csv(local_path)
    for col in scores.select_dtypes(include="object"):
        scores[col] = scores[col].astype(str).str.strip()
    return scores


@st.cache_data(show_spinner=False)
def load_big5_competitions() -> Set[str]:
    if not BIG5_PATH.exists():
        return set()
    entries = {
        line.strip()
        for line in BIG5_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    return entries


@st.cache_data(show_spinner=False)
def load_role_metrics() -> tuple[dict[str, List[str]], dict[str, str]]:
    if not ROLE_METRICS_PATH.exists():
        return {}, {}
    try:
        data = json.loads(ROLE_METRICS_PATH.read_text(encoding="utf-8"))
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
    display = metric.replace("_per_90", " per 90").replace("_pct", " pct").replace("_percent", " percent")
    display = display.replace("_", " ")
    display = display.replace(" per 90", " /90")
    display = display.replace(" pct", " (%)")
    return display.title()


def format_metric_value(value: Optional[float]) -> str:
    if value is None:
        return "—"
    if abs(value) >= 100 or float(value).is_integer():
        return f"{value:.0f}"
    return f"{value:.2f}"


def format_percentile(value: Optional[float]) -> str:
    if value is None:
        return "—"
    return f"{value:.1f}"


def assign_category(metric: str) -> str:
    lowered = metric.lower()
    for name, keywords in CATEGORY_RULES:
        if any(keyword in lowered for keyword in keywords):
            return name
    return "Other"


@st.cache_data(show_spinner=False)
def load_similarity(profile_slug: str) -> Optional[pd.DataFrame]:
    key = f"{SIM_PREFIX}/similarity_{profile_slug}.csv"
    try:
        df = read_csv_from_s3(key)
    except FileNotFoundError:
        local_path = ROOT_DIR / "data" / "similarity" / f"similarity_{profile_slug}.csv"
        if not local_path.exists():
            return None
        df = pd.read_csv(local_path)
    for col in df.select_dtypes(include="object"):
        df[col] = df[col].astype(str).str.strip()
    return df


def slugify(name: str) -> str:
    allowed = []
    for char in name.lower():
        if char.isalnum():
            allowed.append(char)
        elif char in {" ", "-", "/", "_"}:
            allowed.append("-")
    slug = "".join(allowed)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-") or "profile"


def format_positions(row: pd.Series) -> str:
    positions: list[str] = []
    primary = row.get("position")
    if isinstance(primary, str):
        primary = primary.strip()
    if primary and not pd.isna(primary) and str(primary).lower() != "nan":
        positions.append(str(primary))

    secondary = row.get("second_position")
    if isinstance(secondary, str):
        tokens = [token.strip() for token in secondary.split(",")]
    else:
        tokens = [str(secondary).strip()] if secondary and not pd.isna(secondary) else []
    for token in tokens:
        if token and token.lower() != "nan":
            positions.append(token)

    if not positions:
        return "—"
    # Preserve order while removing duplicates
    seen = []
    for pos in positions:
        if pos not in seen:
            seen.append(pos)
    return ", ".join(seen)


def prepare_dataset() -> tuple[pd.DataFrame, List[str]]:
    df = load_players()
    scores = load_profile_scores()

    if scores is not None:
        score_columns = [col for col in scores.columns if col not in KEY_COLUMNS]
        missing_cols = [col for col in score_columns if col not in df.columns]
        if missing_cols:
            join_cols = [col for col in KEY_COLUMNS if col in df.columns and col in scores.columns]
            if join_cols:
                df = df.merge(scores[join_cols + score_columns], on=join_cols, how="left")
    role_cols = sorted(
        {
            col
            for col in df.columns
            if " - " in str(col)
            and not col.endswith(PCT_SUFFIX_GLOBAL)
            and not col.endswith(PCT_SUFFIX_LEAGUE)
        }
    )
    return df, role_cols


def select_top_profiles(row: pd.Series, role_cols: Sequence[str], top_n: Optional[int] = None) -> List[dict]:
    entries: list[dict] = []
    for col in role_cols:
        value = safe_float(row.get(col))
        if value is None:
            continue
        label = str(col).strip()
        entries.append(
            {
                "base": str(col),
                "label": label,
                "value": value,
                "league_pct": safe_float(row.get(f"{col}{PCT_SUFFIX_LEAGUE}")),
                "global_pct": safe_float(row.get(f"{col}{PCT_SUFFIX_GLOBAL}")),
            }
        )

    sorted_entries = sorted(entries, key=lambda item: item["value"], reverse=True)
    if top_n is not None:
        return sorted_entries[:top_n]
    return sorted_entries


def render_player_header(row: pd.Series, container: st.delta_generator.DeltaGenerator) -> None:
    container.image(PLACEHOLDER_IMG, width=120)
    container.markdown(f"### {row.get('player', 'Unknown player')}")

    position_text = format_positions(row)
    age_value = safe_int(row.get("age"))
    birth_year_value = safe_int(row.get("birth_year"))
    matches_value = safe_int(row.get("matches_played"))
    minutes_value = safe_int(row.get("minutes_played"))

    info_pairs = [
        ("Club", row.get("team")),
        ("Birth Country", row.get("birth_country")),
        ("Age", age_value),
        ("Birth year", birth_year_value),
        ("Position", position_text),
        ("Matches", matches_value),
        ("Minutes", minutes_value),
    ]

    cols = container.columns(2)
    for idx, (label, value) in enumerate(info_pairs):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            display = "—"
        else:
            display = value
        cols[idx % 2].markdown(f"**{label}:** {display}")


def render_radar(
    row: pd.Series,
    container: st.delta_generator.DeltaGenerator,
    role_name: Optional[str],
    roles_map: dict[str, List[str]],
    label_map: dict[str, str],
    dataset: pd.DataFrame,
    use_percentiles: bool,
    context: str,
    player_league: Optional[str],
) -> None:
    metrics = roles_map.get(role_name, []) if role_name else []
    if metrics:
        metric_pairs = [(metric, metric_display_name(metric, label_map)) for metric in metrics]
    else:
        metric_pairs = list(RADAR_STATS.items())

    selected_labels: List[str] = []
    selected_values: List[float] = []
    source_label = "league"
    min_range: List[float] = []
    max_range: List[float] = []

    round_flags: List[bool] = []

    preferred_suffix = PCT_SUFFIX_LEAGUE if context == "League" else PCT_SUFFIX_GLOBAL
    fallback_suffix = PCT_SUFFIX_GLOBAL if preferred_suffix == PCT_SUFFIX_LEAGUE else PCT_SUFFIX_LEAGUE

    comp_col = None
    if "competition_name" in dataset.columns:
        comp_col = "competition_name"
    elif "league" in dataset.columns:
        comp_col = "league"

    if use_percentiles:
        for metric_key, display in metric_pairs:
            value = safe_float(row.get(f"{metric_key}{preferred_suffix}"))
            source_label = context.lower()
            if value is None:
                value = safe_float(row.get(f"{metric_key}{fallback_suffix}"))
                if value is not None:
                    source_label = "league" if fallback_suffix == PCT_SUFFIX_LEAGUE else "global"
            if value is None:
                continue
            selected_labels.append(display)
            selected_values.append(np.clip(value, 0, 100))
        min_range = [0.0] * len(selected_labels)
        max_range = [100.0] * len(selected_labels)
        round_flags = [True] * len(selected_labels)
    else:
        source_label = context.lower()
        subset = dataset
        if context == "League" and comp_col and player_league:
            filtered = dataset[dataset[comp_col].astype(str) == str(player_league)]
            if not filtered.empty:
                subset = filtered
        for metric_key, display in metric_pairs:
            raw_value = safe_float(row.get(metric_key))
            if raw_value is None:
                continue

            is_percent_metric = metric_key.endswith("percent") or metric_key.endswith("_pct")
            if is_percent_metric:
                raw_value = float(np.clip(raw_value, 0.0, 100.0))

            selected_labels.append(display)
            selected_values.append(float(raw_value))

            series = pd.to_numeric(subset.get(metric_key, pd.Series(dtype=float)), errors="coerce")
            if is_percent_metric:
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
            if is_percent_metric:
                min_adj = max(0.0, min_adj)
                max_adj = min(100.0, max_adj)
                if max_adj - min_adj < 1e-6:
                    max_adj = min(100.0, min_adj + 1.0)
            min_range.append(min_adj)
            max_range.append(max_adj)
        round_flags = [False] * len(selected_labels)

    if not selected_labels:
        container.info("No percentile data available to build the radar chart.")
        return

    radar = Radar(
        params=selected_labels,
        min_range=min_range,
        max_range=max_range,
        round_int=round_flags,
    )

    fig = plt.figure(figsize=(6, 6), facecolor="#0F172A")
    ax = fig.add_subplot(111)
    radar.setup_axis(
        ax=ax,
        facecolor="#0F172A",
        title=dict(
            title=f"{'Percentiles' if use_percentiles else 'Raw values'} ({source_label}) — {row.get('player', 'Player')}",
            color="#E2E8F0",
            size=14,
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
    radar.draw_param_labels(ax=ax, fontsize=9, color="#E2E8F0")

    container.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_role_cards(
    row: pd.Series,
    role_cols: Sequence[str],
    assigned_role: Optional[str] = None,
) -> None:
    all_profiles = select_top_profiles(row, role_cols)
    if not all_profiles:
        st.info("No role score columns found.")
        return

    st.markdown("### Role fit (profiles)")
    ordered_profiles: list[dict] = []
    if assigned_role:
        assigned_entry = next((item for item in all_profiles if item["base"] == assigned_role), None)
        if assigned_entry:
            ordered_profiles.append(assigned_entry)
    ordered_profiles.extend(item for item in all_profiles if item["base"] != assigned_role)

    top_profiles = ordered_profiles[:3]

    cols = st.columns(len(top_profiles))
    for col, profile in zip(cols, top_profiles):
        base = profile["base"]
        score = profile["value"]
        league_pct = profile.get("league_pct")
        global_pct = profile.get("global_pct")
        display_name = profile["label"]

        with col:
            card = st.container(border=True)
            card.markdown(f"**{display_name}**")
            if assigned_role and base == assigned_role:
                card.caption("Assigned role")

            global_score_display = f"{score:.1f}" if score is not None else "—"
            card.metric("Global score", global_score_display)

            league_display = f"{league_pct:.1f}" if league_pct is not None else "—"
            card.metric("League percentile", league_display)

            # if global_pct is not None:
            #     card.caption(f"Global percentile: {global_pct:.1f}")


def render_role_metrics_section(
    row: pd.Series,
    role_name: Optional[str],
    roles_map: dict[str, List[str]],
    label_map: dict[str, str],
) -> None:
    if not role_name:
        return
    metrics = roles_map.get(role_name)
    if not metrics:
        return

    records = []
    for metric in metrics:
        value = safe_float(row.get(metric))
        league_pct = safe_float(row.get(f"{metric}{PCT_SUFFIX_LEAGUE}"))
        global_pct = safe_float(row.get(f"{metric}{PCT_SUFFIX_GLOBAL}"))
        if all(item is None for item in (value, league_pct, global_pct)):
            continue
        records.append(
            {
                "Metric": metric_display_name(metric, label_map),
                "Value": value if value is not None else np.nan,
                "League percentile": league_pct if league_pct is not None else np.nan,
                "Global percentile": global_pct if global_pct is not None else np.nan,
            }
        )

    if not records:
        return

    st.markdown("### Role metrics focus")
    df_records = pd.DataFrame(records)
    styled = (
        df_records.style.format(
            {
                "Value": lambda x: "—" if pd.isna(x) else format_metric_value(float(x)),
                "League percentile": lambda x: "—" if pd.isna(x) else format_percentile(float(x)),
                "Global percentile": lambda x: "—" if pd.isna(x) else format_percentile(float(x)),
            }
        )
        .background_gradient(
            cmap="RdYlGn",
            subset=["League percentile", "Global percentile"],
            vmin=0,
            vmax=100,
        )
        .hide(axis="index")
    )
    st.dataframe(styled, use_container_width=True)


def render_percentile_table(
    row: pd.Series,
    roles_map: dict[str, List[str]],
    label_map: dict[str, str],
) -> None:
    metrics_with_data: Set[str] = set()
    for col in row.index:
        if col.endswith(PCT_SUFFIX_LEAGUE):
            metrics_with_data.add(col[: -len(PCT_SUFFIX_LEAGUE)])
        elif col.endswith(PCT_SUFFIX_GLOBAL):
            metrics_with_data.add(col[: -len(PCT_SUFFIX_GLOBAL)])

    excluded_prefixes = (
        "calendar",
        "page_number",
        "row_number",
        "player",
        "player_id",
        "birth_year",
        "age",
        "matches_played",
        "minutes_played",
        "team",
        "competition",
        "assigned_role",
    )

    records = []
    seen: Set[str] = set()
    for metric in sorted(metrics_with_data):
        if metric.startswith(excluded_prefixes):
            continue
        if metric in seen:
            continue
        seen.add(metric)

        value = safe_float(row.get(metric))
        league_pct = safe_float(row.get(f"{metric}{PCT_SUFFIX_LEAGUE}"))
        global_pct = safe_float(row.get(f"{metric}{PCT_SUFFIX_GLOBAL}"))
        if all(item is None for item in (value, league_pct, global_pct)):
            continue
        records.append(
            {
                "Category": assign_category(metric),
                "Metric": metric_display_name(metric, label_map),
                "Value": value if value is not None else np.nan,
                "League percentile": league_pct if league_pct is not None else np.nan,
                "Global percentile": global_pct if global_pct is not None else np.nan,
            }
        )

    st.markdown("### All metrics")

    if not records:
        st.info("No percentile data available.")
        return

    df_display = pd.DataFrame(records)
    df_display["Category"] = pd.Categorical(df_display["Category"], categories=CATEGORY_ORDER, ordered=True)
    df_display = df_display.sort_values(["Category", "Metric"])
    styled = (
        df_display.style.format(
            {
                "Value": lambda x: "—" if pd.isna(x) else format_metric_value(float(x)),
                "League percentile": lambda x: "—" if pd.isna(x) else format_percentile(float(x)),
                "Global percentile": lambda x: "—" if pd.isna(x) else format_percentile(float(x)),
            }
        )
        .background_gradient(
            cmap="RdYlGn",
            subset=["League percentile", "Global percentile"],
            vmin=0,
            vmax=100,
        )
        .hide(axis="index")
    )
    st.dataframe(styled, use_container_width=True)


def render_similar_players(
    row: pd.Series,
    assigned_role: Optional[str],
    dataset: pd.DataFrame,
    big5_competitions: Optional[Set[str]] = None,
) -> None:
    st.markdown("### Similar players")
    if not assigned_role:
        st.info("No assigned role found for similarity lookup.")
        return

    slug = slugify(assigned_role)
    sim_df = load_similarity(slug)
    if sim_df is None or sim_df.empty:
        st.info("No similarity data available for this role.")
        return

    current_player = row.get("player")
    matches = sim_df[sim_df["player_a"] == current_player].copy()
    if matches.empty:
        st.info("No similar players found for this profile.")
        return

    matches = matches.sort_values(by="similarity", ascending=False).head(10)

    details: list[dict] = []
    for _, sim_row in matches.iterrows():
        player_b = sim_row.get("player_b")
        team_b = sim_row.get("team_b")
        competition_b = sim_row.get("competition_name_b")

        if str(player_b) == str(current_player):
            continue

        candidate_mask = dataset["player"] == player_b
        if "team" in dataset.columns and isinstance(team_b, str) and team_b:
            candidate_mask &= dataset["team"] == team_b
        if "competition_name" in dataset.columns and isinstance(competition_b, str) and competition_b:
            candidate_mask &= dataset["competition_name"] == competition_b

        candidates = dataset[candidate_mask].copy()
        if candidates.empty:
            candidates = dataset[dataset["player"] == player_b].copy()
        if candidates.empty:
            continue
        if "calendar" in candidates.columns:
            candidates = candidates.sort_values(by="calendar", ascending=False, na_position="last")
        candidate_row = candidates.iloc[0]

        age_value = safe_int(candidate_row.get("age"))
        global_score = safe_float(candidate_row.get(f"{assigned_role}{PCT_SUFFIX_GLOBAL}"))
        league_score = safe_float(candidate_row.get(f"{assigned_role}{PCT_SUFFIX_LEAGUE}"))

        details.append(
            {
                "Player": player_b,
                "Team": team_b,
                "Competition": competition_b,
                "Age": age_value,
                "Similarity": float(sim_row.get("similarity", 0.0)),
                "Global score": global_score,
                "League score": league_score,
            }
        )

    if not details:
        st.info("No similar players found for this profile.")
        return

    slider_container = st
    checkbox_container = None
    if big5_competitions:
        slider_container, checkbox_container = st.columns(2)

    age_values = [item["Age"] for item in details if item["Age"] is not None]
    filtered_details = details
    if age_values:
        min_age, max_age = min(age_values), max(age_values)
        if min_age < max_age:
            age_min, age_max = slider_container.slider(
                "Filter by age",
                min_value=min_age,
                max_value=max_age,
                value=(min_age, max_age),
            )
            filtered_details = [
                item
                for item in details
                if item["Age"] is None or age_min <= item["Age"] <= age_max
            ]
        else:
            slider_container.caption(f"Similar players age: {min_age} years.")

    if big5_competitions:
        toggle_container = checkbox_container or st
        only_big5 = toggle_container.checkbox(
            "Big 5 Leagues",
            value=False,
            help="Show only similar players competing in the Big 5 leagues.",
        )
        if only_big5:
            filtered_details = [
                item
                for item in filtered_details
                if str(item.get("Competition")) in big5_competitions
            ]

    if not filtered_details:
        st.info("No similar players in the selected age range.")
        return

    table = pd.DataFrame(filtered_details)
    table["Similarity"] = table["Similarity"].apply(lambda x: f"{float(x):.3f}")
    table["Global score"] = table["Global score"].apply(lambda x: f"{x:.1f}" if x is not None else "—")
    table["League score"] = table["League score"].apply(lambda x: f"{x:.1f}" if x is not None else "—")
    table["Age"] = table["Age"].apply(lambda x: int(x) if x is not None else "—")

    st.dataframe(
        table[["Player", "Team", "Competition", "Age", "Similarity", "Global score", "League score"]],
        hide_index=True,
        use_container_width=True,
    )


st.set_page_config(page_title="Report", layout="wide", initial_sidebar_state="collapsed")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
df, role_columns = prepare_dataset()
if df.empty:
    st.warning("Player dataset is empty. Please refresh the data pipeline.")
    st.stop()

big5_set = load_big5_competitions()
role_metrics_map, role_label_map = load_role_metrics()

# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
st.session_state.setdefault("report_prev_league", None)
st.session_state.setdefault("report_team", None)
st.session_state.setdefault("report_player", None)

competition_series = (
    df.get("competition_name", pd.Series(index=df.index, dtype="object"))
    .fillna("")
    .astype(str)
    .str.strip()
)
leagues = sorted({comp for comp in competition_series if comp})
if not leagues:
    st.info("No leagues available in the dataset.")
    st.stop()

col_l, col_t, col_p = st.columns(3)
selected_league = col_l.selectbox("Select a league", options=leagues)

if st.session_state["report_prev_league"] != selected_league:
    st.session_state["report_team"] = None
    st.session_state["report_player"] = None
st.session_state["report_prev_league"] = selected_league

league_mask = competition_series == selected_league
team_column = "team_in_selected_period" if "team_in_selected_period" in df.columns else "team"
team_series = (
    df.get(team_column, pd.Series(index=df.index, dtype="object"))
    .fillna("")
    .astype(str)
    .str.strip()
)
teams = sorted({team for team in team_series[league_mask] if team})
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

team_mask = league_mask & (team_series == selected_team)
player_series = (
    df.get("player", pd.Series(index=df.index, dtype="object"))
    .fillna("")
    .astype(str)
    .str.strip()
)
players = sorted({player for player in player_series[team_mask] if player})
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
player_rows = df[team_mask & (player_series == selected_player)].copy()
if player_rows.empty:
    st.info("No data found for the selected player.")
    st.stop()

if "calendar" in player_rows.columns:
    player_rows = player_rows.sort_values(by="calendar", ascending=False, na_position="last")
row = player_rows.iloc[0]

assigned_role = row.get("assigned_role")

left, right = st.columns([1, 1.2])
render_player_header(row, left)
with right:
    percentile_view = st.checkbox(
        "Percentile View",
        value=True,
        key="report_percentile_view",
        help="Toggle between percentile or raw value view for the radar.",
    )
    context_choice = st.selectbox(
        "Context",
        ("League", "Global"),
        index=0,
        key="report_radar_context",
        help="Choose whether to compare the player within their league or across all leagues.",
    )
    render_radar(
        row,
        right,
        assigned_role,
        role_metrics_map,
        role_label_map,
        dataset=df,
        use_percentiles=percentile_view,
        context=context_choice,
        player_league=row.get("competition_name"),
    )

st.divider()
render_role_cards(row, role_columns, assigned_role=assigned_role)
render_role_metrics_section(row, assigned_role, role_metrics_map, role_label_map)
st.divider()
render_percentile_table(row, role_metrics_map, role_label_map)
st.divider()
render_similar_players(row, assigned_role, df, big5_competitions=big5_set)
