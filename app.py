"""
Tabular ML Lab — Publication-grade machine learning for tabular research data.

A guided, interactive platform for researchers working with tabular data
who need defensible methodology and publication-ready outputs.
"""
import streamlit as st

from utils.session_state import get_data, init_session_state
# LLM settings now rendered by render_sidebar_workflow in theme.py
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
# Sidebar: Workflow Progress
render_sidebar_workflow(current_page="")

# ============================================================================
# Main Landing Page
# ============================================================================

# Hero section
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">Open Source · Guided Default Path · Advanced Options When Needed</div>
    <h1>Tabular Machine Learning Lab</h1>
    <p class="hero-sub">
        Interactive machine learning workbench for researchers.<br/>
        Start with the core guided workflow, then add advanced analyses only when they help your study.
    </p>
</div>
""", unsafe_allow_html=True)

# Who this is for
st.markdown("### 👥 Who This Is For")

col_audience1, col_audience2 = st.columns(2)
with col_audience1:
    st.markdown("""
    **✅ This tool is designed for:**
    - Clinical researchers analyzing trial data
    - Biomedical scientists working with experimental results
    - Graduate students writing methods sections
    - Anyone who needs to build and document a prediction model
    
    **📊 Works best with:**
    - Tabular data (CSV, Excel)
    - 50-10,000 rows
    - Supervised learning (classification or regression)
    - Research that requires transparent, reproducible methods
    """)
with col_audience2:
    st.markdown("""
    **❌ Not designed for:**
    - Production ML deployment
    - Time series forecasting
    - Image, text, or audio data
    - Extremely large datasets (>100K rows)
    
    **📝 What you need to know:**
    - **No coding required** — point-and-click interface
    - **Basic stats helpful** — p-values, confidence intervals
    - **Domain expertise required** — you interpret the results
    """)

st.markdown("---")

# Problem/Solution
st.markdown("### 🎯 What Problem Does This Solve?")

col_prob, col_sol = st.columns(2)
with col_prob:
    st.markdown("""
    <div class="info-card">
        <h3>❌ The Manual Workflow Problem</h3>
        <p>Building a defensible ML model for publication typically requires:</p>
        <ul>
            <li>15+ separate Python/R scripts</li>
            <li>Tracking dozens of preprocessing decisions</li>
            <li>Generating Table 1, calibration plots, SHAP values</li>
            <li>Writing methods sections from scratch</li>
            <li>Manually creating TRIPOD checklists</li>
        </ul>
        <p><strong>Result:</strong> Weeks of work, inconsistent methodology, reproducibility issues.</p>
    </div>
    """, unsafe_allow_html=True)
with col_sol:
    st.markdown("""
    <div class="info-card">
        <h3>✅ The Tabular ML Lab Solution</h3>
        <p>This app provides a <strong>guided default workflow</strong> that:</p>
        <ul>
            <li>Gets you from upload → baseline model → export without requiring every advanced step</li>
            <li>Tracks your choices automatically as you work</li>
            <li>Generates draft manuscript materials as you go</li>
            <li>Applies best practices by default</li>
            <li>Keeps advanced options available when your study needs them</li>
        </ul>
        <p><strong>Result:</strong> A manuscript-ready starting point in 30-60 minutes, with advanced analyses available when you want to go deeper.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Workflow steps
st.markdown("### 📋 How It Works")
st.markdown("""
For a first pass, follow the **recommended workflow**: **Upload → EDA → Feature Selection → Preprocess → Train → Explain → Export**.  
Advanced analyses are available when they strengthen your study, but they should not slow down your first run through the product.
""")

col_flow1, col_flow2 = st.columns([1.5, 1])
with col_flow1:
    st.markdown("#### Recommended first run")
    recommended_steps = [
        ("1", "Upload & Audit", "Load your data, set the target, and confirm the dataset is usable."),
        ("2", "Explore (EDA)", "Understand distributions, missingness, and relationships before modeling."),
        ("3", "Feature Selection", "Narrow the predictor set to the variables most worth carrying forward."),
        ("4", "Preprocess", "Build the transformations each model needs without leaking information."),
        ("5", "Train & Compare", "Fit models, compare performance, and establish a baseline result."),
        ("6", "Explainability", "Interpret what the best models learned and whether the result is defensible."),
        ("7", "Export Report", "Download draft methods, tables, and figures as a manuscript starting point."),
    ]

    for num, title, desc in recommended_steps:
        st.markdown(f"""
        <div class="step-item">
            <div class="step-num">{num}</div>
            <div class="step-content">
                <strong>{title}</strong><br/>
                <span>{desc}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

with col_flow2:
    render_info_card(
        "Navigation modes",
        "**Quick workflow** keeps the default research spine front and center.\n\n"
        "**Advanced workflow** keeps the complete page sequence in the main navigation without changing how data flows through the app."
    )
    render_info_card(
        "Advanced workflow",
        "Use these when your first pass shows you need more depth, stronger robustness claims, or a specific statistical narrative.\n\n"
        "**Feature Engineering** — Add richer derived features if baseline models underperform.\n\n"
        "**Sensitivity Analysis** — Show that results are not fragile to seeds or feature dropout.\n\n"
        "**Statistical Validation** — Add classical tests when you need complementary p-values or Table 1 support."
    )
    render_guidance(
        "**Recommendation:** reach Export once on the quick workflow before deciding whether an advanced workflow step is actually necessary. You can switch workflow modes at any time from the sidebar.",
        icon="🧭"
    )

st.markdown("---")

# What you get
st.markdown("### 📦 What You'll Get")

col_out1, col_out2, col_out3 = st.columns(3)
with col_out1:
    st.markdown("""
    <div class="info-card">
        <h3>📊 Publication Tables</h3>
        <ul>
            <li>Table 1 (characteristics + p-values)</li>
            <li>Model comparison with bootstrap CIs</li>
            <li>Calibration metrics</li>
            <li>CSV and LaTeX formats</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
with col_out2:
    st.markdown("""
    <div class="info-card">
        <h3>📈 Publication Figures</h3>
        <ul>
            <li>Calibration curves</li>
            <li>SHAP importance plots</li>
            <li>ROC/Precision-Recall curves</li>
            <li>Forest plots for subgroups</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
with col_out3:
    st.markdown("""
    <div class="info-card">
        <h3>📝 Draft Methods Section</h3>
        <ul>
            <li>Auto-generated from your choices</li>
            <li>TRIPOD checklist tracker</li>
            <li>Flow diagram (CONSORT-style)</li>
            <li>Ready to edit for your paper</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
**⚠️ Important:** This app generates a **manuscript-ready starting point** that you must review and adapt for your specific study. 
You are responsible for interpreting results, ensuring scientific validity, and adding the domain-specific context no tool can infer for you.
""")

st.markdown("---")

# Quick start CTA
st.markdown("### 🚀 Getting Started")

col_qs1, col_qs2 = st.columns([2, 1])
with col_qs1:
    render_guidance(
        "<strong>Ready to start?</strong> Click <strong>📂 Upload & Audit</strong> in the sidebar. "
        "For most users, the best first pass is: upload data, review EDA, select features, preprocess, train, explain, and export. "
        "Advanced steps stay available when you need them.",
        icon="👋"
    )
with col_qs2:
    st.markdown("""
    **⏱️ Time commitment:**
    - Quick exploration: 10 minutes
    - Full workflow: 30-60 minutes
    - Iterative refinement: 2-4 hours
    
    **💾 Save & resume:**
    - Download `.pkl` session file anytime
    - Resume later from same point
    """)

# Capabilities (collapsed)
with st.expander("🔍 Full Capabilities & Technical Details", expanded=False):
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        **18 Model Families:**
        - Linear: Ridge, Lasso, ElasticNet, Logistic Regression, GLM, Huber
        - Trees: Random Forest, ExtraTrees, HistGradientBoosting
        - Other: KNN, SVM, Naive Bayes, LDA, Neural Networks (PyTorch)
        - Automatic baseline comparisons (dummy classifiers/regressors)

        **Evaluation Metrics:**
        - Bootstrap 95% CIs (BCa method, 1000 resamples)
        - Calibration: Brier score, ECE, reliability diagrams
        - Decision curve analysis for clinical utility
        - Subgroup analysis with forest plots
        - Cross-validation with statistical comparisons
        """)
    with col_b:
        st.markdown("""
        **Feature Engineering:**
        - Polynomial features (degree 2-3)
        - Mathematical transforms (log, sqrt, reciprocal)
        - Ratio features (pairwise divisions)
        - Binning (quantile, K-means, equal-width)
        - Topological Data Analysis (persistent homology)
        - PCA/UMAP dimensionality reduction

        **Publication Tools:**
        - Table 1 generator (stratified, automatic tests)
        - Statistical validation (t-test, ANOVA, chi-square, correlation)
        - Auto-generated methods section reflecting your workflow
        - TRIPOD checklist tracking
        - CONSORT-style flow diagrams
        - LaTeX and CSV table export

        **AI Assistance (Optional):**
        - LLM-powered data insights (Ollama, OpenAI, or Anthropic)
        - Preprocessing recommendations
        - Not required to use the app — all features work without AI
        """)

# FAQ
with st.expander("❓ Frequently Asked Questions", expanded=False):
    st.markdown("""
    **Q: Do I need to know Python or R?**  
    A: No. This is a point-and-click web app — no coding required.
    
    **Q: Is my data private?**  
    A: All data processing happens on the server running this application. No data is sent to third-party services (unless you enable optional AI features that use external APIs). If you are running this locally, your data never leaves your machine. On a shared or university server, your data is processed on that server — check with your administrator about data handling policies.
    
    **Q: How long does a complete workflow take?**  
    A: 30-60 minutes for a full analysis. You can save and resume at any point.
    
    **Q: What file formats are supported?**  
    A: CSV and Excel (.xlsx, .xls). You can also merge multiple files.
    
    **Q: Can I use this for my thesis/dissertation?**  
    A: Yes. The app generates draft methods sections and TRIPOD checklists. You'll need to review, interpret, and adapt the outputs for your specific study.
    
    **Q: What if my models perform poorly?**  
    A: The app includes a diagnostic assistant that explains possible causes (weak features, class imbalance, insufficient data, high missingness) and suggests next steps.
    
    **Q: Can I deploy models trained here to production?**  
    A: No. This tool is designed for research and publication, not production deployment. You can export trained models, but deploying them safely requires additional engineering work.
    
    **Q: What makes this different from scikit-learn?**  
    A: scikit-learn is a Python library — you write code. Tabular ML Lab is a guided UI that walks you through best practices and generates publication materials automatically.
    
    **Q: Is this peer-reviewed or validated?**  
    A: The statistical methods used (bootstrap CIs, calibration, SHAP) are standard in the field. However, **you are responsible** for ensuring your analysis is scientifically sound and appropriate for your research question.
    """)

# ── Quick Start ────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🚀 Quick Start")
_qs_col1, _qs_col2 = st.columns(2)
with _qs_col1:
    st.markdown("""
    <div class="info-card">
        <h3>📂 Your Data</h3>
        <p>Bring your own CSV, Excel, or Parquet file. The guided workflow starts at Upload &amp; Audit.</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("→ Go to Upload & Audit", type="primary", key="cta_upload"):
        st.switch_page("pages/01_Upload_and_Audit.py")

with _qs_col2:
    st.markdown("""
    <div class="info-card">
        <h3>🧪 Demo Dataset</h3>
        <p>No data yet? Try a built-in sample dataset on the Upload &amp; Audit page — look for the <em>"Need a practice dataset?"</em> expander.</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("→ Go to Upload & Audit (Demo)", type="secondary", key="cta_demo"):
        st.switch_page("pages/01_Upload_and_Audit.py")

# Theory Reference callout
st.markdown("---")
st.markdown("""
<div style="
    background: linear-gradient(135deg, rgba(102,126,234,0.9), rgba(118,75,162,0.9));
    border-radius: 10px;
    padding: 1.4rem 1.6rem;
    margin: 1rem 0;
    text-align: center;
    box-shadow: 0 4px 16px rgba(102, 126, 234, 0.2);
">
    <div style="font-size: 1.6rem; margin-bottom: 0.4rem;">📖</div>
    <div style="font-size: 1.05rem; font-weight: 700; color: #fff; margin-bottom: 0.4rem;">
        Want to understand why the app makes a recommendation?
    </div>
    <div style="font-size: 0.9rem; color: rgba(255,255,255,0.88); line-height: 1.55; max-width: 560px; margin: 0 auto;">
        The <strong style="color: #fff;">Theory Reference</strong> explains the statistical
        reasoning behind every coaching prompt, preprocessing choice, and model assumption — from
        missing data mechanisms to SHAP values to TRIPOD reporting standards.
        Open it from the sidebar anytime.
    </div>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("""
**Tabular ML Lab** · Open source research tool · Not for clinical decision-making  
[GitHub](https://github.com/hedglinnolan/tabular-ml-lab) · [Report Issues](https://github.com/hedglinnolan/tabular-ml-lab/issues) · [University Deployment Guide](https://github.com/hedglinnolan/tabular-ml-lab/tree/university-docker)
""")

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
