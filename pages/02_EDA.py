"""
Page 02: Exploratory Data Analysis
Shows summary stats, distributions, correlations, and target analysis.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional, Dict, Any, List

from utils.session_state import (
    init_session_state, get_data, DataConfig,
    TaskTypeDetection, CohortStructureDetection
)
from utils.storyline import add_insight, get_insights_by_category, render_breadcrumb, render_page_navigation
from data_processor import get_numeric_columns
from utils.theme import inject_custom_css, render_guidance, render_reviewer_concern, render_step_indicator, render_sidebar_workflow
from utils.table_export import table
from ml.eda_recommender import compute_dataset_signals, recommend_eda, DatasetSignals
from ml import eda_actions
from ml.plot_narrative import (
    narrative_eda_linearity,
    narrative_eda_residuals,
    narrative_eda_influence,
    narrative_eda_normality,
    narrative_eda_sufficiency,
    narrative_eda_scaling,
    narrative_eda_multicollinearity,
)

init_session_state()

st.set_page_config(page_title="EDA", page_icon="📈", layout="wide")
inject_custom_css()
render_sidebar_workflow(current_page="02_EDA")
render_step_indicator(2, "Exploratory Data Analysis")
st.title("📈 Exploratory Data Analysis")
render_breadcrumb("02_EDA")
render_page_navigation("02_EDA")

# Progress indicator

df = get_data()
if df is None:
    st.warning("Please upload data in the Upload & Audit page first")
    st.stop()
if len(df) == 0 or len(df.columns) == 0:
    st.warning("Your dataset is empty. Please upload data with at least one row and one column.")
    st.stop()

# Check task mode - EDA works for both but some features are prediction-specific
task_mode = st.session_state.get('task_mode')
if task_mode == 'hypothesis_testing':
    st.info("🔬 **Hypothesis Testing Mode**: EDA is available, but some prediction-specific features may be limited.")
elif task_mode != 'prediction':
    st.warning("Please select a task mode (Prediction or Hypothesis Testing) in the Upload & Audit page")
    st.stop()

data_config: Optional[DataConfig] = st.session_state.get('data_config')
# For prediction mode, require target/features; for hypothesis testing, allow without
if task_mode == 'prediction' and (data_config is None or not data_config.target_col):
    st.warning("Please select target and features in the Upload & Audit page first")
    st.stop()

target_col = data_config.target_col if data_config else None
feature_cols = data_config.feature_cols if data_config and data_config.feature_cols else [c for c in df.columns if c != target_col]
_has_target = target_col is not None and target_col in df.columns

# Get final detection values
task_type_detection: TaskTypeDetection = st.session_state.get('task_type_detection', TaskTypeDetection())
cohort_structure_detection: CohortStructureDetection = st.session_state.get('cohort_structure_detection', CohortStructureDetection())

task_type_final = task_type_detection.final if task_type_detection.final else data_config.task_type
cohort_type_final = cohort_structure_detection.final if cohort_structure_detection.final else 'cross_sectional'
entity_id_final = cohort_structure_detection.entity_id_final

# EDA settings
with st.expander("EDA Settings", expanded=False):
    outlier_method = st.selectbox(
        "Outlier detection method",
        ["iqr", "mad", "zscore", "percentile"],
        index=0,
        key="eda_outlier_method",
        help="Choose how outliers are defined for EDA metrics and plots."
    )
    method_descriptions = {
        "iqr": "IQR: Flags points outside Q1−1.5×IQR or Q3+1.5×IQR.",
        "mad": "MAD: Uses median absolute deviation with modified z-score threshold (robust).",
        "zscore": "Z-score: Flags points with |z| > 3 (assumes near-normal).",
        "percentile": "Percentile: Flags points outside the 1st–99th percentiles."
    }
    st.caption(method_descriptions.get(outlier_method, ""))

# ============================================================================
# DATASET PROFILE - Compute comprehensive profile for intelligent coaching
# ============================================================================
@st.cache_data
def compute_profile_cached(_df: pd.DataFrame, target: str, features: List[str], task_type: str, outlier_method: str):
    """Compute dataset profile with caching."""
    from ml.dataset_profile import compute_dataset_profile
    return compute_dataset_profile(_df, target, features, task_type, outlier_method)

# Compute the dataset profile
if _has_target:
    profile = compute_profile_cached(df, target_col, feature_cols, task_type_final, outlier_method)
else:
    profile = compute_profile_cached(df, feature_cols[0] if feature_cols else df.columns[0], feature_cols, 'regression', outlier_method)

st.session_state['dataset_profile'] = profile  # Store for other pages

# ============================================================================
# DATASET OVERVIEW PANEL
# ============================================================================
st.markdown("---")

# Quick stats in a clean row
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Samples", f"{profile.n_rows:,}")
with col2:
    st.metric("Features", f"{profile.n_features}")
with col3:
    st.metric("Numeric", f"{profile.n_numeric}")
with col4:
    st.metric("Categorical", f"{profile.n_categorical}")
with col5:
    st.metric("Data Sufficiency", profile.data_sufficiency.value.title())

# Data sufficiency narrative
with st.expander("Data Sufficiency Analysis", expanded=True):
    st.markdown(f"**{profile.sufficiency_narrative}**")
    
    # Show feature-to-sample ratio context (guard against numpy types and div-by-zero)
    p_n = float(profile.p_n_ratio)
    ratio_str = f"{1/p_n:.0f}" if 0 < p_n < float('inf') else "N/A"
    n_feat = max(1, int(profile.n_features))
    pct_num = (int(profile.n_numeric) / n_feat * 100) if n_feat else 0
    pct_cat = (int(profile.n_categorical) / n_feat * 100) if n_feat else 0
    st.markdown(f"""
    **What this means for your models:**
    - **Feature-to-sample ratio:** {profile.p_n_ratio:.3f} (1 feature per {ratio_str} samples)
    - **Numeric features:** {profile.n_numeric} ({pct_num:.0f}% of total)
    - **Categorical features:** {profile.n_categorical} ({pct_cat:.0f}% of total)
    """)
    
    if profile.target_profile and profile.target_profile.task_type == 'classification':
        if profile.events_per_variable:
            st.markdown(f"- **Events per variable:** {profile.events_per_variable:.1f} "
                       f"(minority class has {profile.target_profile.minority_class_size:,} samples)")

# Warnings panel
if profile.warnings:
    st.markdown("### Data Warnings")
    
    # Group warnings by severity
    critical = [w for w in profile.warnings if w.level.value == 'critical']
    warnings = [w for w in profile.warnings if w.level.value == 'warning']
    cautions = [w for w in profile.warnings if w.level.value == 'caution']
    
    if critical:
        for w in critical:
            st.error(f"**{w.short_message}:** {w.detailed_message}")
            if w.suggested_actions:
                st.markdown("**Suggested actions:**")
                for action in w.suggested_actions:
                    st.markdown(f"  • {action}")
    
    if warnings:
        for w in warnings:
            st.warning(f"**{w.short_message}:** {w.detailed_message}")
            with st.expander("Suggested actions"):
                for action in w.suggested_actions:
                    st.markdown(f"• {action}")
    
    if cautions:
        with st.expander(f"{len(cautions)} additional caution(s)"):
            for w in cautions:
                st.info(f"**{w.short_message}:** {w.detailed_message}")

# ============================================================================
# Compute signals for EDA actions
# ============================================================================
@st.cache_data
def compute_signals_cached(_df: pd.DataFrame, target: str, task_type: Optional[str], 
                          cohort_type: Optional[str], entity_id: Optional[str], outlier_method: str):
    """Cached signal computation."""
    return compute_dataset_signals(
        _df, target, task_type, cohort_type, entity_id, outlier_method=outlier_method
    )

try:
    signals = compute_signals_cached(
        df, target_col, task_type_final, cohort_type_final, entity_id_final, outlier_method
    )
except Exception as e:
    st.warning(f"Some signal computations were skipped due to data issues: {str(e)[:100]}")
    from ml.eda_recommender import DatasetSignals
    signals = DatasetSignals(
        n_rows=len(df),
        n_cols=len(df.columns),
        target_name=target_col,
        task_type_final=task_type_final,
        cohort_type_final=cohort_type_final,
        entity_id_final=entity_id_final
    )

# ============================================================================
# KEY INSIGHTS PANEL
# ============================================================================
st.markdown("---")
insights = get_insights_by_category()
if insights:
    with st.expander("Key Insights So Far", expanded=False):
        st.markdown("**Insights collected from EDA analyses:**")
        for insight in insights:
            # Safety check: ensure insight is a dict
            if isinstance(insight, dict):
                st.markdown(f"**{insight.get('category', 'General').title()}:** {insight['finding']}")
                st.caption(f"→ Implication: {insight['implication']}")
else:
    st.info("Run EDA analyses below to collect insights that will guide model selection and preprocessing.")

# ============================================================================
# EDA: UPFRONT (non–model-specific)
# ============================================================================
if 'eda_results' not in st.session_state:
    st.session_state.eda_results = {}

st.header("Non–Model-Specific EDA")
st.caption("Run these first. Plausibility and collinearity inform model choice and preprocessing.")

def _run_and_show(action_id: str, title: str, run_action: str):
    from utils.llm_ui import build_llm_context, build_eda_full_results_context, render_interpretation_with_llm_button
    key_run = f"upfront_run_{action_id}"
    if st.button(f"Run {title}", key=key_run, type="primary"):
        try:
            action_func = getattr(eda_actions, run_action, None)
            if action_func:
                with st.spinner(f"Running {title}..."):
                    result = action_func(df, target_col, feature_cols, signals, st.session_state)
                    st.session_state.eda_results[action_id] = result
                    st.rerun()
            else:
                st.error(f"Action '{run_action}' not found")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    if action_id in st.session_state.eda_results:
        result = st.session_state.eda_results[action_id]
        for w in result.get('warnings', []):
            st.warning(w)
        findings = result.get('findings', [])[:2]
        interp = "; ".join(findings) if findings else None
        for idx, (fig_type, fig_data) in enumerate(result.get('figures', [])):
            if fig_type == 'plotly':
                st.plotly_chart(fig_data, width="stretch", key=f"upfront_plot_{action_id}_{idx}")
            elif fig_type == 'table':
                table(fig_data, width="stretch", key=f"upfront_table_{action_id}_{idx}")
        if interp:
            st.markdown(f"**Interpretation:** {interp}")
            stats_summary = build_eda_full_results_context(result, action_id)
            ctx = build_llm_context(action_id, stats_summary, existing=interp, feature_names=feature_cols, sample_size=len(df) if df is not None else None, task_type=task_type_final if task_type_final else None)
            render_interpretation_with_llm_button(
                ctx, key=f"llm_upfront_{action_id}", result_session_key=f"llm_result_upfront_{action_id}",
            )

col_plaus, col_coll = st.columns(2)
with col_plaus:
    st.subheader("Physiologic Plausibility Check")
    _run_and_show("plausibility_check", "Physiologic Plausibility", "plausibility_check")
with col_coll:
    st.subheader("Collinearity Heatmap")
    _run_and_show("collinearity_map", "Collinearity Heatmap", "collinearity_map")

st.markdown("---")

# ============================================================================
# EDA: MODEL-FAMILY–SPECIFIC (primary)
# ============================================================================
# (description, action_id) per family; each task runs and shows “what am I looking at”
FAMILY_TASKS: Dict[str, List[tuple]] = {
    "Linear Models": [
        ("Check linearity: scatter plots of features vs target", "linearity_scatter"),
        ("Residual analysis: look for patterns in residuals", "residual_analysis"),
        ("Multicollinearity check: correlation matrix, VIF if available", "multicollinearity_vif"),
        ("Influence diagnostics: identify high-leverage points", "influence_diagnostics"),
        ("Normality of residuals (for inference, not prediction)", "normality_residuals"),
    ],
    "Tree-Based Models": [
        ("Feature interactions: look for non-additive effects", "interaction_analysis"),
        ("Nonlinearity indicators and monotonic trends: binned averages by feature", "dose_response_trends"),
        ("Outlier influence on target", "outlier_influence"),
        ("Target profile", "target_profile"),
    ],
    "Neural Networks": [
        ("Data sufficiency check: at least 20× samples per feature recommended", "data_sufficiency_check"),
        ("Feature scaling necessity: check feature value ranges", "feature_scaling_check"),
        ("Leakage detection: features too correlated with target", "leakage_scan"),
        ("Target profile", "target_profile"),
        ("Missingness scan", "missingness_scan"),
    ],
    "Boosting": [
        ("Target profile", "target_profile"),
        ("Outlier influence on target", "outlier_influence"),
        ("Interaction detection: tree-based interaction tests", "interaction_analysis"),
        ("Nonlinearity indicators: binned averages by feature", "dose_response_trends"),
    ],
}

ACTION_NARRATIVE = {
    "linearity_scatter": narrative_eda_linearity,
    "residual_analysis": narrative_eda_residuals,
    "influence_diagnostics": narrative_eda_influence,
    "normality_residuals": narrative_eda_normality,
    "multicollinearity_vif": narrative_eda_multicollinearity,
    "data_sufficiency_check": narrative_eda_sufficiency,
    "feature_scaling_check": narrative_eda_scaling,
}

st.header("Model-Family–Specific EDA")
st.caption("Run all analyses for a family, or run individual tasks. Results include a short “What am I looking at” narrative.")

for family, tasks in FAMILY_TASKS.items():
    with st.expander(f"**{family}**", expanded=False):
        for desc, action_id in tasks:
            st.markdown(f"• {desc}")
        run_list = [(d, a) for d, a in tasks if getattr(eda_actions, a, None) is not None]
        if not run_list:
            st.caption("No runnable actions for this family.")
            continue
        run_all_key = f"run_all_{family.replace(' ', '_')}"
        if st.button("Run All", key=run_all_key, type="primary"):
            for _desc, act in run_list:
                try:
                    action_func = getattr(eda_actions, act, None)
                    if action_func:
                        result = action_func(df, target_col, feature_cols, signals, st.session_state)
                        st.session_state.eda_results[f"family_{family}_{act}"] = result
                except Exception as e:
                    st.session_state.eda_results[f"family_{family}_{act}"] = {
                        "findings": [], "warnings": [str(e)], "figures": [], "stats": {}
                    }
            st.rerun()
        for desc, act in run_list:
            fkey = f"family_{family}_{act}"
            if fkey not in st.session_state.eda_results:
                continue
            result = st.session_state.eda_results[fkey]
            st.markdown(f"**{desc}**")
            for w in result.get("warnings", []):
                st.warning(w)
            findings = result.get("findings", [])
            stats = result.get("stats", {})
            nar_fn = ACTION_NARRATIVE.get(act)
            if nar_fn:
                interp = nar_fn(stats, findings)
            else:
                interp = "; ".join(findings[:2]) if findings else None
            for idx, (fig_type, fig_data) in enumerate(result.get("figures", [])):
                if fig_type == "plotly":
                    st.plotly_chart(fig_data, width="stretch", key=f"eda_plot_{fkey}_{idx}")
                elif fig_type == "table":
                    table(fig_data, width="stretch", key=f"eda_table_{fkey}_{idx}")
            if interp:
                st.markdown(f"**Interpretation:** {interp}")
            elif findings and result.get("figures"):
                st.markdown(f"**Interpretation:** {'; '.join(findings[:2])}")
            if (interp or findings) and result.get("figures"):
                from utils.llm_ui import build_llm_context, build_eda_full_results_context, render_interpretation_with_llm_button
                stats_summary = build_eda_full_results_context(result, act)
                ctx = build_llm_context(act, stats_summary, existing=interp or "; ".join(findings[:2]) if findings else "", feature_names=feature_cols, sample_size=len(df) if df is not None else None, task_type=task_type_final or None)
                render_interpretation_with_llm_button(
                    ctx, key=f"llm_family_{family}_{act}", result_session_key=f"llm_result_family_{family}_{act}",
                )
            st.markdown("---")

st.markdown("---")

# ============================================================================
# EDA: OTHER ADVANCED (dropdown)
# ============================================================================
upfront_and_family = {"plausibility_check", "collinearity_map"}
for _tasks in FAMILY_TASKS.values():
    for _d, aid in _tasks:
        upfront_and_family.add(aid)
OTHER_ACTIONS = [a for a in [
    "missingness_scan", "cohort_split_guidance", "leakage_scan", "quick_probe_baselines"
] if a not in upfront_and_family and getattr(eda_actions, a, None) is not None]

st.header("Other Advanced Analyses")
if OTHER_ACTIONS:
    other_select = st.selectbox("Select analysis to run", OTHER_ACTIONS, key="eda_other_select")
    if st.button("Run Selected", key="eda_other_run"):
        try:
            action_func = getattr(eda_actions, other_select, None)
            if action_func:
                with st.spinner(f"Running {other_select}..."):
                    result = action_func(df, target_col, feature_cols, signals, st.session_state)
                    st.session_state.eda_results[f"other_{other_select}"] = result
                    st.rerun()
        except Exception as e:
            st.error(f"Error: {str(e)}")
    other_key = f"other_{other_select}"
    if other_key in st.session_state.eda_results:
        result = st.session_state.eda_results[other_key]
        for w in result.get('warnings', []):
            st.warning(w)
        findings = result.get('findings', [])[:2]
        interp = "; ".join(findings) if findings else None
        for idx, (fig_type, fig_data) in enumerate(result.get('figures', [])):
            if fig_type == 'plotly':
                st.plotly_chart(fig_data, width="stretch", key=f"other_plot_{other_select}_{idx}")
            elif fig_type == 'table':
                table(fig_data, width="stretch", key=f"other_table_{other_select}_{idx}")
        if interp:
            st.markdown(f"**Interpretation:** {interp}")
            from utils.llm_ui import build_llm_context, build_eda_full_results_context, render_interpretation_with_llm_button
            stats_summary = build_eda_full_results_context(result, other_select)
            ctx = build_llm_context(other_select, stats_summary, existing=interp, feature_names=feature_cols, sample_size=len(df) if df is not None else None, task_type=task_type_final or None)
            render_interpretation_with_llm_button(
                ctx, key=f"llm_other_{other_select}", result_session_key=f"llm_result_other_{other_select}",
            )
else:
    st.caption("No additional analyses in this category.")

st.markdown("---")

# ============================================================================
# DATASET SIGNALS EXPLAINER
# ============================================================================
with st.expander("Dataset Signals Detail"):
    st.markdown("**Dataset Summary:**")
    st.write(f"• Rows: {signals.n_rows:,}")
    st.write(f"• Columns: {signals.n_cols}")
    st.write(f"• Numeric columns: {len(signals.numeric_cols)}")
    st.write(f"• Categorical columns: {len(signals.categorical_cols)}")
    st.write(f"• High missing columns (>5%): {len(signals.high_missing_cols)}")
    st.write(f"• Duplicate row rate: {signals.duplicate_row_rate:.1%}")
    st.write(f"• Outlier method: {outlier_method.upper()}")
    
    if signals.target_stats:
        st.markdown("**Target Statistics:**")
        for key, value in signals.target_stats.items():
            if isinstance(value, (int, float)):
                st.write(f"• {key}: {value:.3f}")
            else:
                st.write(f"• {key}: {value}")
    
    if signals.collinearity_summary:
        st.markdown("**Collinearity:**")
        st.write(f"• Max correlation: {signals.collinearity_summary.get('max_corr', 0):.3f}")
    
    if signals.physio_plausibility_flags:
        st.markdown("**Physiologic Plausibility Flags (NHANES):**")
        for flag in signals.physio_plausibility_flags:
            st.write(f"• {flag}")

st.markdown("---")

# ============================================================================
# STANDARD EDA VIEWS
# ============================================================================
st.header("Standard EDA Views")

# Summary statistics
st.subheader("Summary Statistics")
_summary_cols = (feature_cols + [target_col]) if _has_target else feature_cols
_summary_cols = [c for c in _summary_cols if c in df.columns]
table(df[_summary_cols].describe(), width="stretch")

# Distribution plots
st.subheader("Distributions")

# Target distribution
if _has_target:
    st.markdown(f"**Target Distribution: {target_col}**")
    col1, col2 = st.columns(2)

    with col1:
        fig_hist = px.histogram(df, x=target_col, nbins=30, title=f"Distribution of {target_col}")
        st.plotly_chart(fig_hist, width="stretch")

    with col2:
        fig_box = px.box(df, y=target_col, title=f"Box Plot of {target_col}")
        st.plotly_chart(fig_box, width="stretch")
else:
    st.info("No target variable selected. Showing feature distributions only.")

# Classification: class balance
if _has_target and task_type_final == 'classification':
    st.subheader("Class Balance")
    class_counts = df[target_col].value_counts().sort_index()
    fig_bar = px.bar(x=class_counts.index.astype(str), y=class_counts.values,
                     title="Class Distribution", labels={'x': 'Class', 'y': 'Count'})
    st.plotly_chart(fig_bar, width="stretch")
    st.info(f"Classes: {len(class_counts)} | Imbalance ratio: {class_counts.max()/class_counts.min():.2f}")

# Feature distributions (top 6)
st.subheader("Feature Distributions")
n_features_show = min(6, len(feature_cols))
cols_per_row = 3

for i in range(0, n_features_show, cols_per_row):
    cols = st.columns(cols_per_row)
    for j, col in enumerate(cols):
        if i + j < n_features_show:
            feat = feature_cols[i + j]
            with col:
                fig = px.histogram(df, x=feat, nbins=20, title=feat)
                st.plotly_chart(fig, width="stretch")

# Target vs feature plots (collinearity heatmap is upfront)
if not _has_target:
    st.header("Feature Distributions")
    st.info("Select a target variable to see target-vs-feature plots.")

if _has_target and task_type_final == 'regression':
    n_plots = min(6, len(feature_cols))
    cols_per_row = 3
    
    for i in range(0, n_plots, cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            if i + j < n_plots:
                feat = feature_cols[i + j]
                with col:
                    fig = px.scatter(df, x=feat, y=target_col, title=f"{target_col} vs {feat}")
                    st.plotly_chart(fig, width="stretch")

# Classification: box plots
elif _has_target:
    n_plots = min(6, len(feature_cols))
    cols_per_row = 3
    
    for i in range(0, n_plots, cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            if i + j < n_plots:
                feat = feature_cols[i + j]
                with col:
                    fig = px.box(df, x=target_col, y=feat, title=f"{feat} by {target_col}")
                    st.plotly_chart(fig, width="stretch")

# ============================================================================
# TABLE 1: Characteristics of Study Population
# ============================================================================
st.header("📋 Table 1: Study Population Characteristics")
st.markdown("""
**Why this matters:** Table 1 is the first table in nearly every clinical/research paper.
It summarizes your study population, stratified by groups, with appropriate statistical tests.

💡 **Note:** This generates Table 1 with automatic p-values. You can add custom statistical tests 
later in **Statistical Validation** (page 9) if you need specific comparisons beyond the defaults.
""")

with st.expander("Generate Table 1", expanded=False):
    from ml.table_one import Table1Config, generate_table1, table1_to_csv, table1_to_latex
    from data_processor import get_categorical_columns

    all_numeric = get_numeric_columns(df)
    all_categorical = get_categorical_columns(df)

    # Grouping variable
    possible_groups = [c for c in all_categorical if c != target_col and df[c].nunique() <= 10]
    grouping_var = st.selectbox(
        "Stratify by (grouping variable)",
        options=["None"] + possible_groups,
        index=0,
        help="Select a categorical variable to stratify the table. Leave 'None' for overall summary only.",
        key="table1_group",
    )

    # Variable selection
    # Clear stale Table 1 widget state if columns changed
    for _wk in ("table1_continuous", "table1_categorical", "table1_group"):
        _old = st.session_state.get(_wk)
        if isinstance(_old, list):
            st.session_state[_wk] = [v for v in _old if v in df.columns]
        elif isinstance(_old, str) and _old not in ("None",) and _old not in df.columns:
            st.session_state.pop(_wk, None)

    _t1_cont_options = [c for c in all_numeric if c != target_col]
    _t1_cont_default = [c for c in feature_cols if c in all_numeric and c in _t1_cont_options][:10]
    t1_continuous = st.multiselect(
        "Continuous variables",
        options=_t1_cont_options,
        default=_t1_cont_default,
        key="table1_continuous",
    )
    _t1_cat_options = [c for c in all_categorical if c != target_col and c != grouping_var]
    _t1_cat_default = [c for c in feature_cols if c in all_categorical and c in _t1_cat_options][:5]
    t1_categorical = st.multiselect(
        "Categorical variables",
        options=_t1_cat_options,
        default=_t1_cat_default,
        key="table1_categorical",
    )

    # Options
    col_t1a, col_t1b, col_t1c = st.columns(3)
    with col_t1a:
        show_pvalues = st.checkbox("Show p-values", value=True, key="table1_pval")
    with col_t1b:
        show_smd = st.checkbox("Show SMD", value=False, key="table1_smd",
                               help="Standardized Mean Difference — requires exactly 2 groups in grouping variable")
    with col_t1c:
        show_missing = st.checkbox("Show missing counts", value=True, key="table1_miss",
                                    help="Shows count and % missing for each variable — only appears for variables with missing data")

    if st.button("Generate Table 1", key="gen_table1", type="primary"):
        config = Table1Config(
            grouping_var=grouping_var if grouping_var != "None" else None,
            continuous_vars=t1_continuous,
            categorical_vars=t1_categorical,
            show_pvalues=show_pvalues,
            show_smd=show_smd,
            show_missing=show_missing,
        )
        table1_df, table1_metadata = generate_table1(df, config)
        st.session_state["table1_df"] = table1_df
        st.session_state["table1_metadata"] = table1_metadata

    if st.session_state.get("table1_df") is not None:
        table1_df = st.session_state["table1_df"]
        table(table1_df)  # No key - CSV/LaTeX downloads provided below

        # Test info
        table1_metadata = st.session_state.get("table1_metadata", {})
        if table1_metadata.get("tests_used"):
            st.caption("**Tests used:** " + ", ".join(
                f"{var}: {test}" for var, test in table1_metadata["tests_used"].items()
            ))

        # Export options
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            csv_data = table1_to_csv(table1_df)
            st.download_button("📥 Download CSV", csv_data, "table1.csv", "text/csv", key="dl_table1_csv")
        with col_exp2:
            latex_data = table1_to_latex(table1_df)
            st.download_button("📥 Download LaTeX", latex_data, "table1.tex", "text/plain", key="dl_table1_latex")

# ============================================================================
# STORE EDA INSIGHTS FOR FEATURE ENGINEERING PAGE
# ============================================================================
# Store insights in session state for downstream pages (used by Feature Engineering page)
feature_engineering_hints = {
    'skewed_features': [],  # Features with abs(skewness) > 1.0
    'high_corr_pairs': [],  # Pairs with correlation > 0.9
    'has_missing': False,
    'numeric_features': [],
}

# Compute numeric columns for analysis
numeric_cols_check = df.select_dtypes(include=[np.number]).columns
feature_engineering_hints['numeric_features'] = list(numeric_cols_check)

# Detect skewed features
for col in numeric_cols_check:
    try:
        skew_val = df[col].skew()
        if pd.notna(skew_val) and abs(skew_val) > 1.0:
            feature_engineering_hints['skewed_features'].append({
                'name': col,
                'skewness': round(float(skew_val), 2)
            })
    except:
        pass  # Skip features that can't compute skewness

# Detect high correlations
correlation_matrix = df[numeric_cols_check].corr() if len(numeric_cols_check) > 1 else None
if correlation_matrix is not None and len(correlation_matrix.columns) > 1:
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            try:
                corr_val = correlation_matrix.iloc[i, j]
                if pd.notna(corr_val) and abs(corr_val) > 0.9:
                    feature_engineering_hints['high_corr_pairs'].append({
                        'feature1': correlation_matrix.columns[i],
                        'feature2': correlation_matrix.columns[j],
                        'correlation': round(float(corr_val), 3)
                    })
            except:
                pass

# Check missing data
feature_engineering_hints['has_missing'] = (df.isnull().sum() > 0).any()

# Store in session state for Feature Engineering page
st.session_state['feature_engineering_hints'] = feature_engineering_hints

# ============================================================================
# NEXT STEPS BASED ON YOUR DATA
# ============================================================================
st.markdown("---")
st.markdown("### 📊 Next Steps Based on Your Data")

# Check for common patterns
has_missing = (df.isnull().sum() > 0).any()
numeric_cols_check = df.select_dtypes(include=[np.number]).columns
correlation_matrix = df[numeric_cols_check].corr() if len(numeric_cols_check) > 1 else None
high_corr = False
if correlation_matrix is not None:
    high_corr = ((correlation_matrix.abs() > 0.9) & (correlation_matrix != 1.0)).any().any()

recommendations = []
if has_missing:
    recommendations.append("⚠️ **Missing data detected** → Preprocessing will handle imputation")
if high_corr:
    recommendations.append("✅ **High correlation detected** → Feature Selection will help choose which features to keep")

if recommendations:
    for rec in recommendations:
        st.markdown(f"- {rec}")
else:
    st.markdown("✅ Your data looks clean. Proceed to Feature Engineering (optional) or Feature Selection.")

st.markdown("👉 **Recommended:** Feature Engineering (optional) or Feature Selection")

st.success("EDA complete. Proceed to Feature Selection or Preprocessing.")
