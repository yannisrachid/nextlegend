from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
import streamlit as st

from analytics import FEATURE_GROUPS, build_enriched_dataset
from data_utils import (
    SchemaMappingResult,
    load_index_weights,
    load_player_dataset,
    load_settings,
)


@st.cache_resource(show_spinner=False)
def _get_dataset() -> tuple[pd.DataFrame, SchemaMappingResult]:
    return load_player_dataset()


@st.cache_resource(show_spinner=False)
def _get_settings() -> dict[str, Any]:
    return load_settings()


def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### NextLegend")
        st.caption("Player intelligence powered by Your Legend.")


DEFAULT_FILTERS = {"season": None, "leagues": [], "min_minutes": 0}


def _prepare_session_state(
    df_enriched: pd.DataFrame,
    mapping: SchemaMappingResult,
    filters: Dict[str, Any],
    settings: dict[str, Any],
) -> None:
    st.session_state["dataset"] = df_enriched
    st.session_state["filtered_dataset"] = df_enriched
    st.session_state["schema_mapping"] = mapping
    st.session_state["global_filters"] = filters
    st.session_state["settings"] = settings


def main() -> None:
    st.set_page_config(
        page_title="NextLegend — Home",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _render_sidebar()

    df, mapping = _get_dataset()
    settings = _get_settings()
    weights = load_index_weights()

    weights_signature = tuple(sorted(weights.items()))

    if (
        "enriched_dataset" not in st.session_state
        or st.session_state.get("enriched_signature") != weights_signature
    ):
        with st.spinner("Loading data..."):
            enriched_df, centiles_long = build_enriched_dataset(
                df,
                weights=weights,
                groupby=("position", "league"),
            )
        st.session_state["enriched_dataset"] = enriched_df
        st.session_state["centiles_long_full"] = centiles_long
        st.session_state["enriched_signature"] = weights_signature
    else:
        enriched_df = st.session_state["enriched_dataset"]
        centiles_long = st.session_state.get("centiles_long_full", pd.DataFrame())

    filters = DEFAULT_FILTERS.copy()
    _prepare_session_state(enriched_df, mapping, filters, settings)
    st.session_state["enriched_dataset"] = enriched_df
    st.session_state["centiles_long"] = centiles_long
    st.session_state["centiles_long_full"] = centiles_long
    st.session_state["feature_groups"] = FEATURE_GROUPS
    st.session_state["index_weights"] = weights

    st.markdown(
        """
        <div style="text-align:center; padding: 2rem 0 1.5rem;">
            <h1 style="font-size:3rem; margin-bottom:0.2rem;">NextLegend by Your Legend</h1>
            <h3 style="margin-bottom:0.4rem; color:#7BD389;">Scout with Intelligence with NextLegend</h3>
            <p style="font-size:1.05rem; color:#cbd5f5;">A tool designed by Your Legend for scouting</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    base_df = df.copy()
    players_count = int(len(base_df))
    metric_columns = base_df.select_dtypes(include=[np.number]).columns
    metrics_count = len(metric_columns)
    competition_field = "league" if "league" in base_df.columns else "competition_name"
    competitions_count = int(
        base_df[competition_field].nunique()
    ) if competition_field in base_df.columns else 0

    stats = [
        ("Players in database", f"{players_count:,}"),
        ("Metrics tracked", f"{metrics_count:,}"),
        ("Competitions covered", f"{competitions_count:,}"),
    ]

    cols = st.columns(len(stats), gap="large")
    for col, (label, value) in zip(cols, stats):
        col.markdown(
            f"""
            <div class="nextlegend-card" style="text-align:center; padding:1.8rem 1.2rem;">
                <div style="font-size:2.4rem; font-weight:700; color:#7BD389;">{value}</div>
                <div style="margin-top:0.4rem; font-size:1rem; letter-spacing:0.05em; text-transform:uppercase; color:#cbd5f5;">{label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("Tools for Advanced Analysis")

    tools = [
        {
            "title": "Scouting Report",
            "description": "Generate comprehensive player dossiers enriched with performance metrics and contextual insights.",
        },
        {
            "title": "Comparison",
            "description": "Assess up to four players side by side across seasons and competitions to support informed decisions.",
        },
        {
            "title": "Performance Score",
            "description": "Leverage the NextLegend index to highlight individual strengths within positional cohorts.",
        },
        {
            "title": "Similarity Score",
            "description": "Surface profiles comparable to your reference players using feature-weighted similarity models.",
        },
        {
            "title": "Projection",
            "description": "Project trajectories combining performance indicators, playing time trends and contextual data.",
        },
    ]

    card_columns = st.columns(5, gap="large")
    for col, tool in zip(card_columns, tools):
        col.markdown(
            f"""
            <div class="nextlegend-card" style="min-height:160px;">
                <h4 style="margin-bottom:0.6rem; color:#7BD389;">{tool['title']}</h4>
                <p style="margin:0; color:#e2e8f0; font-size:0.95rem;">{tool['description']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        "<div style='text-align:center; margin-top:3rem; font-size:0.9rem; color:#94a3b8;'>© 2025 YOUR LEGEND — All Rights Reserved.</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
