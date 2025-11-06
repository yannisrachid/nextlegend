from __future__ import annotations

from pathlib import Path

import streamlit as st

ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"
LOGO_PATH = ASSETS_DIR / "ylfc_logo.png"

__all__ = ["render_sidebar_logo", "LOGO_PATH"]


def render_sidebar_logo() -> None:
    """Display the YLFC logo and product tagline at the top of the sidebar."""
    container = st.sidebar.container()
    if LOGO_PATH.exists():
        container.image(str(LOGO_PATH), use_container_width=True)
    else:
        container.markdown("### Your Legend FC")
    container.markdown("### NextLegend")
    container.caption("Player intelligence powered by Your Legend.")
    container.markdown("<div style='margin-bottom:0.5rem;'></div>", unsafe_allow_html=True)
