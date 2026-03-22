"""
📖 Theory Reference — Statistical foundations behind every page of Tabular ML Lab.

Structured as a browsable reference: selectbox picks the chapter, tabs organize
parallel content (e.g., model families), and expanders offer optional deep dives.
"""

import streamlit as st
from utils.theme import inject_custom_css, render_sidebar_workflow

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Theory Reference | Tabular ML Lab",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_custom_css()
render_sidebar_workflow(current_page="11_Theory")

# ── Local styles ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Chapter intro card */
.theory-intro {
    background: linear-gradient(135deg, rgba(102,126,234,0.08), rgba(118,75,162,0.08));
    border-left: 3px solid #667eea;
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1.5rem;
    color: #cbd5e1;
    font-size: 0.97rem;
    line-height: 1.65;
}
.theory-intro strong { color: #e2e8f0; }

/* Section heading inside chapters */
.theory-section-head {
    font-size: 1.08rem;
    font-weight: 700;
    color: #e2e8f0;
    margin: 1.6rem 0 0.6rem 0;
    letter-spacing: -0.01em;
}

/* Citation badge */
.cite {
    display: inline-block;
    background: rgba(102,126,234,0.15);
    color: #93a3f8;
    font-size: 0.78rem;
    padding: 0.15rem 0.55rem;
    border-radius: 4px;
    margin-left: 0.3rem;
    vertical-align: middle;
    font-weight: 500;
}

/* Formula block */
.formula-block {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(102,126,234,0.2);
    border-radius: 6px;
    padding: 0.9rem 1.2rem;
    margin: 0.8rem 0;
    font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
    font-size: 0.92rem;
    color: #e2e8f0;
    line-height: 1.6;
    overflow-x: auto;
}

/* App connection callout */
.app-callout {
    background: rgba(34,197,94,0.08);
    border-left: 3px solid #22c55e;
    border-radius: 6px;
    padding: 0.8rem 1.1rem;
    margin: 1rem 0;
    font-size: 0.9rem;
    color: #a7f3d0;
    line-height: 1.55;
}
.app-callout strong { color: #bbf7d0; }

/* Key takeaway box */
.key-takeaway {
    background: rgba(251,191,36,0.08);
    border-left: 3px solid #fbbf24;
    border-radius: 6px;
    padding: 0.8rem 1.1rem;
    margin: 1rem 0;
    font-size: 0.9rem;
    color: #fde68a;
    line-height: 1.55;
}
.key-takeaway strong { color: #fef3c7; }

/* Page hero */
.theory-hero {
    text-align: center;
    padding: 1.5rem 0 1rem 0;
}
.theory-hero h1 {
    font-size: 1.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #667eea 0%, #a78bfa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.theory-hero .subtitle {
    color: #94a3b8;
    font-size: 0.95rem;
    max-width: 640px;
    margin: 0 auto;
    line-height: 1.5;
}
</style>
""", unsafe_allow_html=True)


# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="theory-hero">
    <h1>📖 Theory Reference</h1>
    <p class="subtitle">
        The statistical reasoning behind every recommendation, coaching prompt, and
        model assumption in Tabular ML Lab — written for researchers who want to
        understand <em>why</em>, not just <em>what</em>.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Chapter selector ─────────────────────────────────────────────────────────
CHAPTERS = [
    "Data Quality & Assumptions",
    "Feature Engineering & Selection",
    "Model Families",
    "Preprocessing Theory",
    "Evaluation & Validation",
    "Statistical Testing",
    "Sensitivity & Robustness",
    "Reporting Standards (TRIPOD)",
]

chapter = st.selectbox(
    "Choose a chapter",
    CHAPTERS,
    index=0,
    help="Each chapter covers the theory behind one or more pages of the app.",
)

st.markdown("<div style='margin-top: 0.5rem'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Helper functions for consistent rendering
# ══════════════════════════════════════════════════════════════════════════════

def intro(text: str):
    """Render a chapter introduction card."""
    st.markdown(f'<div class="theory-intro">{text}</div>', unsafe_allow_html=True)


def section(title: str):
    """Render a section heading within a chapter."""
    st.markdown(f'<div class="theory-section-head">{title}</div>', unsafe_allow_html=True)


def formula(text: str):
    """Render a formula/code block."""
    st.markdown(f'<div class="formula-block">{text}</div>', unsafe_allow_html=True)


def app_connection(text: str):
    """Render a callout linking theory to the app."""
    st.markdown(f'<div class="app-callout">🔬 <strong>In the app:</strong> {text}</div>', unsafe_allow_html=True)


def cite(text: str):
    """Return a citation badge HTML string for inline use."""
    return f'<span class="cite">{text}</span>'


def takeaway(text: str):
    """Render a key takeaway box."""
    st.markdown(f'<div class="key-takeaway">💡 <strong>Key takeaway:</strong> {text}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Chapter 1: Data Quality & Assumptions
# ══════════════════════════════════════════════════════════════════════════════

def render_data_quality():
    intro(
        "Before any model is trained, the quality and structure of your data "
        "determines what methods are appropriate and what conclusions are defensible. "
        "This chapter covers the diagnostic checks the app performs during "
        "<strong>Upload & Audit</strong> and <strong>EDA</strong>, and the theory behind them."
    )

    tabs = st.tabs([
        "Missing Data",
        "Distributional Shape",
        "Outliers",
        "Collinearity",
        "Sample Size",
    ])

    # ── Missing Data ─────────────────────────────────────────────────────────
    with tabs[0]:
        st.markdown("*Content will be populated in the next pass.*")

    # ── Distributional Shape ─────────────────────────────────────────────────
    with tabs[1]:
        st.markdown("*Content will be populated in the next pass.*")

    # ── Outliers ─────────────────────────────────────────────────────────────
    with tabs[2]:
        st.markdown("*Content will be populated in the next pass.*")

    # ── Collinearity ─────────────────────────────────────────────────────────
    with tabs[3]:
        st.markdown("*Content will be populated in the next pass.*")

    # ── Sample Size ──────────────────────────────────────────────────────────
    with tabs[4]:
        st.markdown("*Content will be populated in the next pass.*")


# ══════════════════════════════════════════════════════════════════════════════
# Chapter 2: Feature Engineering & Selection
# ══════════════════════════════════════════════════════════════════════════════

def render_feature_engineering():
    intro(
        "Feature engineering transforms raw variables into representations that "
        "better expose the underlying signal to your models. Feature selection then "
        "prunes the input space to reduce noise, improve interpretability, and avoid "
        "overfitting. This chapter covers the theory behind the "
        "<strong>Feature Engineering</strong> and <strong>Feature Selection</strong> pages."
    )

    tabs = st.tabs([
        "Transformations",
        "Encoding Categoricals",
        "Selection Methods",
        "Information Leakage",
    ])

    with tabs[0]:
        st.markdown("*Content will be populated in the next pass.*")

    with tabs[1]:
        st.markdown("*Content will be populated in the next pass.*")

    with tabs[2]:
        st.markdown("*Content will be populated in the next pass.*")

    with tabs[3]:
        st.markdown("*Content will be populated in the next pass.*")


# ══════════════════════════════════════════════════════════════════════════════
# Chapter 3: Model Families
# ══════════════════════════════════════════════════════════════════════════════

def render_model_families():
    intro(
        "Tabular ML Lab organizes models into six families based on their "
        "mathematical assumptions. Understanding these families is the key to "
        "understanding <em>why</em> the app recommends different preprocessing "
        "for different models — and why a choice that helps one model can hurt another."
    )

    tabs = st.tabs([
        "Linear",
        "Tree-Based",
        "Neural Network",
        "Distance-Based",
        "Margin-Based",
        "Probabilistic",
    ])

    with tabs[0]:
        st.markdown("*Content will be populated in the next pass.*")

    with tabs[1]:
        st.markdown("*Content will be populated in the next pass.*")

    with tabs[2]:
        st.markdown("*Content will be populated in the next pass.*")

    with tabs[3]:
        st.markdown("*Content will be populated in the next pass.*")

    with tabs[4]:
        st.markdown("*Content will be populated in the next pass.*")

    with tabs[5]:
        st.markdown("*Content will be populated in the next pass.*")


# ══════════════════════════════════════════════════════════════════════════════
# Chapter 4: Preprocessing Theory
# ══════════════════════════════════════════════════════════════════════════════

def render_preprocessing():
    intro(
        "Preprocessing is where theory meets practice. The choices you make here — "
        "how to scale, impute, and encode — are not neutral. Each decision carries "
        "assumptions that align with some models and conflict with others. This is why "
        "Tabular ML Lab builds <strong>per-model pipelines</strong>: the preprocessing "
        "that serves a Ridge regression well can actively harm a Random Forest."
    )

    tabs = st.tabs([
        "Per-Model Pipelines",
        "Scaling Methods",
        "Imputation Strategies",
        "Train-Test Splitting",
    ])

    with tabs[0]:
        st.markdown("*Content will be populated in the next pass.*")

    with tabs[1]:
        st.markdown("*Content will be populated in the next pass.*")

    with tabs[2]:
        st.markdown("*Content will be populated in the next pass.*")

    with tabs[3]:
        st.markdown("*Content will be populated in the next pass.*")


# ══════════════════════════════════════════════════════════════════════════════
# Chapter 5: Evaluation & Validation
# ══════════════════════════════════════════════════════════════════════════════

def render_evaluation():
    intro(
        "A model is only as good as the evidence you can marshal for its performance. "
        "Choosing the right metrics, validating properly, and explaining predictions "
        "are what separate a publishable result from a notebook experiment. This chapter "
        "covers the theory behind <strong>Train & Compare</strong> and "
        "<strong>Explainability</strong>."
    )

    tabs = st.tabs([
        "Classification Metrics",
        "Regression Metrics",
        "Cross-Validation",
        "Calibration",
        "SHAP & Feature Importance",
    ])

    with tabs[0]:
        st.markdown("*Content will be populated in the next pass.*")

    with tabs[1]:
        st.markdown("*Content will be populated in the next pass.*")

    with tabs[2]:
        st.markdown("*Content will be populated in the next pass.*")

    with tabs[3]:
        st.markdown("*Content will be populated in the next pass.*")

    with tabs[4]:
        st.markdown("*Content will be populated in the next pass.*")


# ══════════════════════════════════════════════════════════════════════════════
# Chapter 6: Statistical Testing
# ══════════════════════════════════════════════════════════════════════════════

def render_statistical_testing():
    intro(
        "Statistical tests give your results formal rigor. But a test is only meaningful "
        "when its assumptions are met and its results are correctly interpreted. "
        "This chapter covers the tests available in the <strong>Hypothesis Testing</strong> "
        "page and the theory that governs when each is appropriate."
    )

    tabs = st.tabs([
        "Hypothesis Testing Fundamentals",
        "Table 1 & Descriptive Tests",
        "Model Comparison Tests",
        "Goodness of Fit",
        "Bootstrap Methods",
    ])

    with tabs[0]:
        st.markdown("*Content will be populated in the next pass.*")

    with tabs[1]:
        st.markdown("*Content will be populated in the next pass.*")

    with tabs[2]:
        st.markdown("*Content will be populated in the next pass.*")

    with tabs[3]:
        st.markdown("*Content will be populated in the next pass.*")

    with tabs[4]:
        st.markdown("*Content will be populated in the next pass.*")


# ══════════════════════════════════════════════════════════════════════════════
# Chapter 7: Sensitivity & Robustness
# ══════════════════════════════════════════════════════════════════════════════

def render_sensitivity():
    intro(
        "A result that changes dramatically with a different random seed or a slightly "
        "different sample is not a result — it's an accident. Sensitivity analysis "
        "quantifies how stable your findings are and where fragility lurks. "
        "This covers the theory behind the <strong>Sensitivity Analysis</strong> page."
    )

    tabs = st.tabs([
        "Seed Sensitivity",
        "Bootstrap Stability",
    ])

    with tabs[0]:
        st.markdown("*Content will be populated in the next pass.*")

    with tabs[1]:
        st.markdown("*Content will be populated in the next pass.*")


# ══════════════════════════════════════════════════════════════════════════════
# Chapter 8: Reporting Standards (TRIPOD)
# ══════════════════════════════════════════════════════════════════════════════

def render_reporting():
    intro(
        "The ultimate output of this app is a defensible, reproducible manuscript. "
        "Reporting standards like TRIPOD exist because reviewers and readers need "
        "to assess <em>how</em> you built your model, not just <em>how well</em> it "
        "performed. This chapter explains the standards the "
        "<strong>Report Export</strong> page helps you meet."
    )

    tabs = st.tabs([
        "TRIPOD Guidelines",
        "Reproducibility Checklist",
    ])

    with tabs[0]:
        st.markdown("*Content will be populated in the next pass.*")

    with tabs[1]:
        st.markdown("*Content will be populated in the next pass.*")


# ══════════════════════════════════════════════════════════════════════════════
# Dispatch
# ══════════════════════════════════════════════════════════════════════════════

CHAPTER_RENDERERS = {
    "Data Quality & Assumptions": render_data_quality,
    "Feature Engineering & Selection": render_feature_engineering,
    "Model Families": render_model_families,
    "Preprocessing Theory": render_preprocessing,
    "Evaluation & Validation": render_evaluation,
    "Statistical Testing": render_statistical_testing,
    "Sensitivity & Robustness": render_sensitivity,
    "Reporting Standards (TRIPOD)": render_reporting,
}

CHAPTER_RENDERERS[chapter]()

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.82rem; padding: 0.5rem 0 1rem 0;">
    This reference is part of <strong style="color: #94a3b8;">Tabular ML Lab</strong>.
    Content is written for researchers and reviewers — if something is unclear or
    incomplete, that's a bug worth reporting.
</div>
""", unsafe_allow_html=True)
