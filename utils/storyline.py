"""
Page navigation and progress tracking for the modeling lab.

Note: Insight management has moved to utils/insight_ledger.py (InsightLedger).
This module now only provides UI navigation helpers.
"""
from typing import List, Optional, Tuple
import streamlit as st

# Page order for breadcrumbs and navigation (page_id, display_name, switch_path)
PAGE_ORDER = [
    ("Home", "Home", "app.py"),
    ("01_Upload_and_Audit", "Upload & Audit", "pages/01_Upload_and_Audit.py"),
    ("02_EDA", "EDA", "pages/02_EDA.py"),
    ("03_Feature_Engineering", "Feature Engineering", "pages/03_Feature_Engineering.py"),
    ("04_Feature_Selection", "Feature Selection", "pages/04_Feature_Selection.py"),
    ("05_Preprocess", "Preprocess", "pages/05_Preprocess.py"),
    ("06_Train_and_Compare", "Train & Compare", "pages/06_Train_and_Compare.py"),
    ("07_Explainability", "Explainability", "pages/07_Explainability.py"),
    ("08_Sensitivity_Analysis", "Sensitivity Analysis", "pages/08_Sensitivity_Analysis.py"),
    ("09_Hypothesis_Testing", "Statistical Validation", "pages/09_Hypothesis_Testing.py"),
    ("10_Report_Export", "Report Export", "pages/10_Report_Export.py"),
]

RECOMMENDED_PAGE_IDS = [
    "Home",
    "01_Upload_and_Audit",
    "02_EDA",
    "04_Feature_Selection",
    "05_Preprocess",
    "06_Train_and_Compare",
    "07_Explainability",
    "10_Report_Export",
]


def render_breadcrumb(current_page: str, step_label: Optional[str] = None) -> None:
    """Render breadcrumb at top of page: Upload & Audit > Step 4: Data Audit"""
    parts = []
    for page_id, display, _ in PAGE_ORDER:
        if page_id == current_page:
            parts.append(display)
            break
        parts.append(display)
    breadcrumb = " > ".join(parts)
    if step_label:
        breadcrumb += f" > {step_label}"
    st.caption(f"**{breadcrumb}**")


def _get_navigation_order(current_page: str):
    """Return page order for navigation.

    Recommended mode keeps the core workflow spine intact and leaves advanced pages
    available without making them part of the default next/previous flow.
    """
    workflow_mode = st.session_state.get("workflow_mode", "quick")
    if workflow_mode == "quick" and current_page in RECOMMENDED_PAGE_IDS:
        return [p for p in PAGE_ORDER if p[0] in RECOMMENDED_PAGE_IDS]
    return PAGE_ORDER


def get_prev_next_pages(current_page: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Return (prev_path, prev_label, next_path, next_label) for navigation buttons."""
    navigation_order = _get_navigation_order(current_page)
    pages = [p[0] for p in navigation_order]
    if current_page not in pages:
        return None, None, None, None
    idx = pages.index(current_page)
    prev = navigation_order[idx - 1] if idx > 0 else None
    next_ = navigation_order[idx + 1] if idx < len(navigation_order) - 1 else None
    prev_path, prev_label = (prev[2], prev[1]) if prev else (None, None)
    next_path, next_label = (next_[2], next_[1]) if next_ else (None, None)
    return prev_path, prev_label, next_path, next_label


def render_page_navigation(current_page: str) -> None:
    """Render Previous / Next page buttons. Uses st.switch_page when available."""
    prev_path, prev_label, next_path, next_label = get_prev_next_pages(current_page)
    if not prev_path and not next_path:
        return
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if prev_path and prev_label:
            if st.button(f"← {prev_label}", key="nav_prev"):
                try:
                    st.switch_page(prev_path)
                except AttributeError:
                    st.info("Use the sidebar to navigate to " + prev_label)
    with col3:
        if next_path and next_label:
            if st.button(f"{next_label} →", key="nav_next"):
                try:
                    st.switch_page(next_path)
                except AttributeError:
                    st.info("Use the sidebar to navigate to " + next_label)
