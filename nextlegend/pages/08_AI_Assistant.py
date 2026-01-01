"""Agentic AI Assistant page for NextLegend."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

from ai import (  # noqa: E402
    PlayerFilters,
    build_column_catalog,
    detect_language,
    filter_candidates,
    get_llm,
    prepare_scout_payload,
    run_data_scientist,
    run_player_agent,
    run_scout_agent,
    summarise_player_row,
)
from auth import render_account_controls, require_authentication  # noqa: E402
from components.sidebar import render_sidebar_logo  # noqa: E402
from s3_utils import read_csv_from_s3  # noqa: E402

DATA_KEY = "data/wyscout_players_cleaned.csv"
LOCAL_DATA_PATH = _PROJECT_ROOT / "data" / "wyscout_players_cleaned.csv"


@st.cache_data(show_spinner=False)
def load_ai_dataset() -> pd.DataFrame:
    """Load the enriched dataset from session, local disk or S3."""

    if "enriched_dataset" in st.session_state:
        df = st.session_state["enriched_dataset"]
    elif "dataset" in st.session_state:
        df = st.session_state["dataset"]
    else:
        try:
            df = read_csv_from_s3(DATA_KEY)
        except FileNotFoundError:
            if not LOCAL_DATA_PATH.exists():
                raise
            df = pd.read_csv(LOCAL_DATA_PATH)

    working = df.copy()
    if "player" not in working.columns and "player_name" in working.columns:
        working["player"] = working["player_name"]
    if "competition_name" not in working.columns and "league" in working.columns:
        working["competition_name"] = working["league"]
    if "minutes_played" not in working.columns and "minutes" in working.columns:
        working["minutes_played"] = working["minutes"]
    return working


def _get_options(df: pd.DataFrame, column: str) -> list[str]:
    if column not in df.columns:
        return []
    return sorted({str(val) for val in df[column].dropna().unique()})


def _render_manual_filters(df: pd.DataFrame) -> Dict[str, Optional[str | int]]:
    league_col = "competition_name" if "competition_name" in df.columns else "league"
    role_col = "assigned_role" if "assigned_role" in df.columns else None
    position_col = "position" if "position" in df.columns else None

    leagues = _get_options(df, league_col) if league_col else []
    roles = _get_options(df, role_col) if role_col else []
    positions = _get_options(df, position_col) if position_col else []

    col1, col2, col3 = st.columns(3)
    with col1:
        league = st.selectbox(
            "League",
            options=["All"] + leagues,
            index=0,
        )
        league_value = None if league == "All" else league
    with col2:
        role = (
            st.selectbox("Role", options=["All"] + roles, index=0)
            if roles
            else "All"
        )
        role_value = None if role == "All" else role
    with col3:
        position = (
            st.selectbox("Position", options=["All"] + positions, index=0)
            if positions
            else "All"
        )
        position_value = None if position == "All" else position

    col4, col5 = st.columns(2)
    with col4:
        max_age_input = st.number_input(
            "Max age (optional, 0 to ignore)",
            min_value=0,
            max_value=50,
            value=0,
            step=1,
        )
        max_age_value = int(max_age_input) if max_age_input > 0 else None
    with col5:
        min_minutes_input = st.number_input(
            "Minimum minutes (optional, 0 to ignore)",
            min_value=0,
            max_value=6000,
            value=0,
            step=90,
        )
        min_minutes_value = int(min_minutes_input) if min_minutes_input > 0 else None

    return {
        "league": league_value,
        "role": role_value,
        "position": position_value,
        "max_age": max_age_value,
        "min_minutes": min_minutes_value,
    }


def _display_watchlist(df_filtered: pd.DataFrame, response_json: dict) -> None:
    candidates = response_json.get("candidates", []) if isinstance(response_json, dict) else []
    if not candidates:
        st.info("The scout agent did not return any candidates.")
        return

    result_df = pd.DataFrame(candidates)
    match_col = "player" if "player" in df_filtered.columns else "player_name"
    if match_col not in df_filtered.columns:
        match_col = None
    league_col = "competition_name" if "competition_name" in df_filtered.columns else "league"
    minutes_col = "minutes_played" if "minutes_played" in df_filtered.columns else "minutes"

    if match_col:
        extra_cols = [
            col
            for col in (match_col, league_col, "assigned_role", minutes_col, "global_score_adjusted")
            if col in df_filtered.columns
        ]
        joined = result_df.merge(
            df_filtered[extra_cols],
            how="left",
            left_on="player_name",
            right_on=match_col,
            suffixes=("", "_data"),
        )
    else:
        joined = result_df.copy()
    joined = joined.sort_values(by="priority")

    priority_palette = {1: "#22c55e", 2: "#eab308", 3: "#38bdf8"}

    def _style_priority(val: int) -> str:
        return f"background-color: {priority_palette.get(val, '#1f2937')}; color: #0F172A; font-weight:600;"

    styled = joined[["player_name", "priority", "reason", "role_summary"]]
    st.subheader("Watchlist")
    st.dataframe(
        styled.style.applymap(
            _style_priority,
            subset=pd.IndexSlice[:, ["priority"]],
        ),
        use_container_width=True,
    )

    csv_data = styled.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download watchlist (CSV)",
        data=csv_data,
        file_name="nextlegend_watchlist.csv",
        mime="text/csv",
    )


def _display_player_report(report: str, player_context: dict) -> None:
    st.subheader("Player report")
    st.markdown(report)

    with st.expander("Context used for the report"):
        st.json(player_context, expanded=False)


def main() -> None:
    st.set_page_config(
        page_title="AI Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    require_authentication()
    render_sidebar_logo()
    render_account_controls()

    st.title("AI Assistant – NextLegend")
    st.caption("Multi-agent assistant for scouting queries. Data stays local; only compact summaries are sent to the model.")

    df = load_ai_dataset()
    if df.empty:
        st.warning("Dataset is empty. Run the pipeline first.")
        st.stop()

    mode = st.selectbox(
        "Assistant mode",
        options=["Scout Advisor", "Player Agent / Report"],
        help="Scout Advisor builds a watchlist. Player Agent writes a narrative report.",
    )

    user_prompt = st.text_area(
        "What do you need?",
        placeholder="Example: Un jeune buteur <23 ans, beaucoup de buts et xG, altruiste (xA élevés), minutes solides.",
        height=160,
    )

    language_choice = st.radio(
        "Output language",
        options=["Auto", "English", "Français"],
        horizontal=True,
    )
    manual_filters = _render_manual_filters(df)

    run_button = st.button("Lancer l’assistant IA", type="primary")
    if not run_button:
        st.stop()

    if not user_prompt.strip():
        st.warning("Please provide a short brief for the assistant.")
        st.stop()

    language = (
        "en"
        if language_choice == "English"
        else "fr"
        if language_choice == "Français"
        else detect_language(user_prompt)
    )

    try:
        llm = get_llm()
    except Exception as exc:
        st.error(f"Unable to initialise OpenAI client: {exc}")
        st.stop()

    column_catalog = build_column_catalog(df)
    with st.spinner("Translating your brief into structured filters..."):
        try:
            filters = run_data_scientist(
                user_prompt,
                column_catalog=column_catalog,
                overrides=manual_filters,
                llm=llm,
            )
        except Exception as exc:
            st.error(f"Data Scientist agent failed: {exc}")
            st.stop()

    # Enforce manual overrides after the agent decision
    filters = PlayerFilters.parse_obj(
        {
            **filters.dict(),
            **{k: v for k, v in manual_filters.items() if v is not None},
        }
    )

    with st.expander("Applied filters (AI + manual)"):
        st.json(filters.dict())

    with st.spinner("Filtering candidates locally..."):
        filtered = filter_candidates(df, filters)

    if filtered.empty:
        relaxed_payload = {
            **filters.dict(),
            "min_metrics": {},
            "max_age": None,
            "min_minutes": None,
        }
        relaxed_filters = PlayerFilters.parse_obj(relaxed_payload)
        relaxed_filtered = filter_candidates(df, relaxed_filters)
        if relaxed_filtered.empty:
            st.warning("No players match the request. Try relaxing the constraints.")
            st.stop()
        st.info("No players matched the strict filters. Showing a relaxed search without age/minutes/metric thresholds.")
        filters = relaxed_filters
        filtered = relaxed_filtered

    st.success(f"{len(filtered)} players matched the request (showing up to {filters.limit}).")

    if mode == "Scout Advisor":
        payload = prepare_scout_payload(filtered)
        if not payload:
            st.warning("No usable data to send to the scout agent.")
            st.stop()
        with st.spinner("Ranking the shortlist with the scout agent..."):
            try:
                scout_response = run_scout_agent(
                    user_text=user_prompt,
                    players=payload,
                    language=language,
                    llm=llm,
                )
                scout_dict = scout_response.dict()
            except Exception as exc:
                st.error(f"Scout agent failed: {exc}")
                st.stop()
        _display_watchlist(filtered, scout_dict)
    else:
        player_col = "player" if "player" in filtered.columns else "player_name"
        options = [str(name) for name in filtered[player_col].head(10)]
        selected_name = st.selectbox("Select the player for the report", options=options)
        target_row = filtered[filtered[player_col].astype(str) == selected_name].iloc[0]
        player_context = summarise_player_row(target_row)
        with st.spinner("Generating the player report..."):
            try:
                report_text = run_player_agent(
                    user_text=user_prompt,
                    player=player_context,
                    language=language,
                    llm=llm,
                )
            except Exception as exc:
                st.error(f"Player agent failed: {exc}")
                st.stop()
        _display_player_report(report_text, player_context)


if __name__ == "__main__":
    main()
