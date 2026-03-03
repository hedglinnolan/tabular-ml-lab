"""
Tabular ML Lab — Publication-grade machine learning for tabular research data.

A guided, interactive platform for researchers working with tabular data
who need defensible methodology and publication-ready outputs.
"""
import streamlit as st

from utils.session_state import get_data, init_session_state
from utils.llm_ui import render_llm_settings_sidebar
from utils.theme import inject_custom_css, render_info_card, render_guidance, render_step_indicator, render_sidebar_workflow

# Initialize session state
init_session_state()

# Page config
st.set_page_config(
    page_title="Tabular ML Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Material Design theme
inject_custom_css()

# Sidebar: LLM settings
render_llm_settings_sidebar()

# Sidebar: Workflow Progress
render_sidebar_workflow(current_page="")

# ============================================================================
# Main Landing Page
# ============================================================================

# Hero section
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">Open Source · 18 Models · 9-Step Workflow</div>
    <h1>Tabular ML Lab</h1>
    <p class="hero-sub">
        From raw data to publication-ready results.<br/>
        Built for researchers who need defensible methodology.
    </p>
</div>
""", unsafe_allow_html=True)

# Three value props
col1, col2, col3 = st.columns(3)
with col1:
    render_info_card(
        "🎯 Guided Workflow",
        "Step-by-step from upload to publication. Smart defaults with advanced options when you need them."
    )
with col2:
    render_info_card(
        "📊 Publication Ready",
        "Table 1, bootstrap CIs, TRIPOD checklists, methods sections — straight into your paper."
    )
with col3:
    render_info_card(
        "🛡️ Reviewer Proof",
        "Baselines, calibration, sensitivity checks — anticipate and address reviewer concerns."
    )

st.markdown("<br/>", unsafe_allow_html=True)

# Workflow steps
st.markdown("### How It Works")
st.markdown("Follow the pages in the sidebar, in order. Each step builds on the previous one.")

steps = [
    ("1", "Upload & Audit", "Load your data, configure target variable and features, review data quality.", "📂"),
    ("2", "Explore (EDA)", "Distributions, correlations, Table 1, missing data analysis, AI-powered insights.", "📈"),
    ("3", "Feature Selection", "LASSO path, RFE-CV, stability selection — find the most informative predictors.", "🎯"),
    ("4", "Preprocess", "Build per-model preprocessing pipelines: imputation, scaling, encoding, outlier treatment.", "⚙️"),
    ("5", "Train & Compare", "Multiple model families with bootstrap CIs, baseline comparison, and calibration analysis.", "🧠"),
    ("6", "Explain & Validate", "SHAP values, permutation importance, external validation, subgroup analysis.", "🔬"),
    ("7", "Sensitivity Analysis", "Test robustness: random seed sensitivity and feature dropout analysis.", "🔬"),
    ("8", "Hypothesis Testing", "Statistical tests without ML — t-tests, ANOVA, chi-square, correlation.", "📐"),
    ("9", "Export Report", "Methods section, TRIPOD checklist, flow diagrams, publication-quality figures & tables.", "📄"),
]

for num, title, desc, icon in steps:
    st.markdown(f"""
    <div class="step-item">
        <div class="step-num">{num}</div>
        <div class="step-content">
            <strong>{icon} {title}</strong><br/>
            <span>{desc}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)

# Quick start CTA
col_qs1, col_qs2 = st.columns([2, 1])
with col_qs1:
    render_guidance(
        "<strong>First time?</strong> Click <strong>Upload & Audit</strong> in the sidebar to get started. "
        "Upload a CSV or Excel file, select your target variable, and the app will guide you from there.",
        icon="👋"
    )
with col_qs2:
    st.markdown("""
    **Also available:**
    - **Hypothesis Testing** — Statistical tests without ML (t-tests, ANOVA, chi-square)
    """)

# Capabilities (collapsed)
with st.expander("📋 Full Capabilities List", expanded=False):
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        **Models:**
        - Linear (Ridge, Lasso, ElasticNet)
        - Trees (Random Forest, ExtraTrees, HistGBT)
        - KNN · SVM · Naive Bayes · LDA
        - Neural Networks (PyTorch, configurable)
        - Automatic baseline comparisons

        **Evaluation:**
        - Bootstrap 95% CIs (BCa, 1000 resamples)
        - Calibration (Brier, ECE, reliability diagrams)
        - Decision curve analysis
        - Subgroup analysis with forest plots
        - Cross-validation with paired tests
        """)
    with col_b:
        st.markdown("""
        **Publication Tools:**
        - Table 1 generator (stratified, p-values, SMD)
        - Auto-generated methods section
        - TRIPOD checklist tracker
        - CONSORT-style flow diagrams
        - LaTeX/Word table export

        **Feature Selection:**
        - LASSO path visualization
        - RFE-CV · Stability selection
        - Univariate screening (FDR-corrected)
        - Consensus across methods

        **AI Interpretation:**
        - Ollama / OpenAI / Anthropic
        """)

# Footer
st.markdown("---")
st.caption("Tabular ML Lab · Built for researchers, by researchers · [GitHub](https://github.com/hedglinnolan/tabular-ml-lab)")

# Debug
if st.sidebar.checkbox("Show Session State", value=False):
    st.sidebar.json({
        'has_data': get_data() is not None,
        'task_mode': st.session_state.get('task_mode'),
        'n_datasets': len(st.session_state.get('datasets_registry', {})),
        'configured': st.session_state.get('data_config') is not None,
        'pipeline': st.session_state.get('preprocessing_pipeline') is not None,
        'splits': st.session_state.get('X_train') is not None,
        'n_models': len(st.session_state.get('trained_models', {})),
    })
