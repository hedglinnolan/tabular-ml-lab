"""
Storyline and progress tracking for the educational modeling lab.
"""
from typing import List, Dict, Optional, Tuple
import streamlit as st

# Page order for breadcrumbs and navigation (page_id, display_name, switch_path)
PAGE_ORDER = [
    ("Home", "Home", "app.py"),
    ("01_Upload_and_Audit", "Upload & Audit", "pages/01_Upload_and_Audit.py"),
    ("02_EDA", "EDA", "pages/02_EDA.py"),
    ("03_Feature_Selection", "Feature Selection", "pages/03_Feature_Selection.py"),
    ("04_Preprocess", "Preprocess", "pages/04_Preprocess.py"),
    ("05_Train_and_Compare", "Train & Compare", "pages/05_Train_and_Compare.py"),
    ("06_Explainability", "Explainability", "pages/06_Explainability.py"),
    ("07_Sensitivity_Analysis", "Sensitivity Analysis", "pages/07_Sensitivity_Analysis.py"),
    ("08_Hypothesis_Testing", "Hypothesis Testing", "pages/08_Hypothesis_Testing.py"),
    ("09_Report_Export", "Report Export", "pages/09_Report_Export.py"),
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


def get_prev_next_pages(current_page: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Return (prev_path, prev_label, next_path, next_label) for navigation buttons."""
    pages = [p[0] for p in PAGE_ORDER]
    if current_page not in pages:
        return None, None, None, None
    idx = pages.index(current_page)
    prev = PAGE_ORDER[idx - 1] if idx > 0 else None
    next_ = PAGE_ORDER[idx + 1] if idx < len(PAGE_ORDER) - 1 else None
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


class StorylinePhase:
    """Represents a phase in the modeling lab storyline."""
    def __init__(self, id: str, name: str, description: str, page: str):
        self.id = id
        self.name = name
        self.description = description
        self.page = page


PHASES = [
    StorylinePhase("data_loaded", "Data Loaded", "Dataset uploaded and basic info confirmed", "01_Upload_and_Audit"),
    StorylinePhase("target_confirmed", "Target & Task Confirmed", "Target variable and task type (regression/classification) set", "01_Upload_and_Audit"),
    StorylinePhase("cohort_confirmed", "Cohort Structure Confirmed", "Cross-sectional vs longitudinal structure identified", "01_Upload_and_Audit"),
    StorylinePhase("eda_insights", "EDA Insights Gathered", "Key patterns and relationships explored", "02_EDA"),
    StorylinePhase("preprocessing", "Preprocessing Configured", "Data transformation pipeline built", "04_Preprocess"),
    StorylinePhase("models_trained", "Models Trained & Compared", "Models trained and performance evaluated", "05_Train_and_Compare"),
    StorylinePhase("explainability", "Explainability Completed", "Model interpretations and feature importance analyzed", "06_Explainability"),
    StorylinePhase("report_exported", "Report Exported", "Comprehensive report generated and downloaded", "09_Report_Export"),
]


def get_completed_phases() -> List[str]:
    """Get list of completed phase IDs from session state."""
    completed = []
    
    # Check each phase
    if st.session_state.get('raw_data') is not None:
        completed.append("data_loaded")
    
    data_config = st.session_state.get('data_config')
    if data_config and data_config.target_col:
        completed.append("target_confirmed")
    
    cohort_detection = st.session_state.get('cohort_structure_detection')
    if cohort_detection and cohort_detection.final:
        completed.append("cohort_confirmed")
    
    if st.session_state.get('eda_insights'):
        completed.append("eda_insights")
    
    if st.session_state.get('preprocessing_pipeline'):
        completed.append("preprocessing")
    
    if st.session_state.get('trained_models'):
        completed.append("models_trained")
    
    if st.session_state.get('permutation_importance') or st.session_state.get('partial_dependence'):
        completed.append("explainability")
    
    if st.session_state.get('report_data'):
        completed.append("report_exported")
    
    return completed


def render_progress_indicator(current_page: str):
    """Render the progress indicator showing where user is in the lab."""
    completed = get_completed_phases()
    
    st.sidebar.header("📍 Modeling Lab Progress")
    
    for phase in PHASES:
        is_completed = phase.id in completed
        is_current = phase.page == current_page
        
        if is_completed:
            icon = "✅"
            status = "Completed"
        elif is_current:
            icon = "🔄"
            status = "Current"
        else:
            icon = "⏳"
            status = "Pending"
        
        st.sidebar.markdown(f"{icon} **{phase.name}**")
        if is_current:
            st.sidebar.caption(phase.description)
    
    # Progress percentage
    progress_pct = len(completed) / len(PHASES) * 100
    st.sidebar.progress(progress_pct / 100)
    st.sidebar.caption(f"{len(completed)}/{len(PHASES)} phases complete ({progress_pct:.0f}%)")


def add_insight(insight_id: str, finding: str, implication: str, category: str = "general"):
    """Add an EDA insight to session state."""
    if 'eda_insights' not in st.session_state:
        st.session_state.eda_insights = []
    
    insight = {
        'id': insight_id,
        'finding': finding,
        'implication': implication,
        'category': category
    }
    
    # Avoid duplicates
    existing_ids = [i['id'] for i in st.session_state.eda_insights]
    if insight_id not in existing_ids:
        st.session_state.eda_insights.append(insight)


def get_insights_by_category(category: Optional[str] = None) -> List[Dict]:
    """Get EDA insights, optionally filtered by category."""
    insights = st.session_state.get('eda_insights', [])
    if category:
        return [i for i in insights if i.get('category') == category]
    return insights
