"""
UI helper for MPCIM dashboard

Provides a simple function to load the global stylesheet and small helpers
for consistent headers or small UI utilities.
"""
from pathlib import Path
import streamlit as st


def apply_styles():
    """Load global CSS from `app/styles.css` and inject into the page."""
    try:
        app_root = Path(__file__).resolve().parent
        css_path = app_root / "styles.css"
        if css_path.exists():
            with open(css_path, "r", encoding="utf-8") as f:
                css = f.read()
            st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except Exception:
        # Fail silently — styling is optional
        pass


def page_header(title: str, subtitle: str = None, icon: str = "📊"):
    """Render a consistent header for pages."""
    st.markdown(f'<div class="main-header">{icon} {title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="sub-header">{subtitle}</div>', unsafe_allow_html=True)
