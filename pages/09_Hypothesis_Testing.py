"""
Page 09: Statistical Validation
Generate traditional statistical tests to validate ML findings and populate Table 1.
These tests provide p-values and effect sizes required for publication.

AUDIT NOTE (Data Flow):
- get_data() returns: df_engineered (if FE applied) > filtered_data > raw_data
- Works in both prediction and hypothesis_testing modes
- Methodology logging: Added for all statistical tests (correlation, t-test, ANOVA, chi-square, normality, paired)
- Custom test results stored in session state for Table 1 export
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
import logging

from utils.session_state import init_session_state, get_data, log_methodology
from utils.storyline import render_breadcrumb, render_page_navigation
from utils.theme import inject_custom_css, render_guidance, render_sidebar_workflow
from utils.table_export import table
from data_processor import get_numeric_columns, get_categorical_columns
from ml.table_one import format_pvalue
from ml.stats_tests import (
    correlation_test,
    two_sample_location_test,
    k_sample_location_test,
    categorical_association_test,
    normality_check,
    paired_location_test
)

logger = logging.getLogger(__name__)

# ── Which family of test the DATA asks for ────────────────────────────────
# The parametric/non-parametric switch used to default to parametric whatever
# the distribution looked like, so an unattended run reported a t-test on
# skewed data. ml/table_one.py already chooses per-variable by Shapiro-Wilk for
# every Table 1 row; the same rule (and the same statistic, from
# ml.stats_tests.normality_check) decides the default here.
_NORMALITY_MIN_N = 8      # ml/table_one._is_normal's floor: below it Shapiro-Wilk decides nothing
_NORMALITY_ALPHA = 0.05


def _parametric_default(samples: Dict[str, np.ndarray]) -> Tuple[bool, str]:
    """Default the parametric choice from a normality pre-check.

    Parametric only when EVERY sample is compatible with normality at
    `_NORMALITY_ALPHA`. Too few observations to test is not evidence of
    normality, so it falls to the non-parametric test and says which sample
    made it fall.

    Returns (use_parametric, reason) — the reason is shown on screen and
    recorded with the result.
    """
    verdicts: List[str] = []
    parametric = True
    for label, values in samples.items():
        x = np.asarray(values, dtype=float)
        x = x[~np.isnan(x)]
        if len(x) < _NORMALITY_MIN_N:
            parametric = False
            verdicts.append(f"{label}: n={len(x)}, too few to test normality")
            continue
        _, p, test_label = normality_check(x)
        if np.isnan(p):
            parametric = False
            verdicts.append(f"{label}: {test_label} could not be computed")
            continue
        if p <= _NORMALITY_ALPHA:
            parametric = False
        verdicts.append(f"{label}: {test_label} p={p:.4g}")
    return parametric, "; ".join(verdicts)


def _parametric_choice(
    samples: Dict[str, np.ndarray],
    key_prefix: str,
    scope: str,
    checkbox_label: str,
    parametric_name: str,
    nonparametric_name: str,
    help_text: str,
) -> Tuple[bool, bool, str]:
    """Render the pre-check, its verdict, and the override box.

    The widget key carries `scope` (the columns the pre-check ran on): a key
    that outlived the selection would answer for the previous columns, which is
    the failure this replaces.

    Returns (use_parametric, default, reason).
    """
    default, reason = _parametric_default(samples)
    chosen = parametric_name if default else nonparametric_name
    st.caption(
        f"**Assumption check:** {reason} → defaulting to **{chosen}** "
        f"({'normality not rejected' if default else 'normality rejected or untestable'} "
        f"at α={_NORMALITY_ALPHA}). You can override below; whichever test runs "
        f"is recorded with the result."
    )
    use_parametric = st.checkbox(
        checkbox_label,
        value=default,
        key=f"{key_prefix}::{scope}",
        help=help_text,
    )
    if use_parametric != default:
        st.warning(
            f"⚠️ Overriding the assumption check: running the "
            f"**{parametric_name if use_parametric else nonparametric_name}** "
            f"where the pre-check selected the "
            f"**{nonparametric_name if use_parametric else parametric_name}**. "
            f"The override is recorded with the result."
        )
    return use_parametric, default, reason


init_session_state()

st.set_page_config(page_title="Statistical Validation", page_icon="📊", layout="wide")
inject_custom_css()
render_sidebar_workflow(current_page="09_Hypothesis_Testing")
st.title("📊 Statistical Validation")
st.caption("Use this when you need classical tests to support the story coming out of EDA and model explainability.")
render_breadcrumb("09_Hypothesis_Testing")
from utils.test_lockbox import render_lockbox_status
render_lockbox_status("Tests on this page run on the full cohort (including locked test rows) — appropriate for Table 1 and descriptive claims, not model-performance claims.")
render_page_navigation("09_Hypothesis_Testing")

from utils.coaching_ui import render_page_coaching
render_page_coaching("09_Hypothesis_Testing")

if st.session_state.get("workflow_mode", "quick") == "quick":
    st.info("""
    🧭 **Advanced workflow step** — Return here after the quick workflow when a manuscript or reviewer needs targeted classical tests in addition to your ML result.
    """)

with st.expander("📖 Why Statistical Validation?", expanded=False):
    st.markdown("""
Not required for every project. Use when you need classical statistics to complement ML results.

- **Add targeted confirmatory tests** for features or comparisons you care about
- **Populate Table 1** with custom p-values when automatic outputs aren't enough
- **ML says:** "Glucose is an important predictor" · **Statistics says:** "Glucose differs significantly between groups"
""")

with st.expander("📄 How this fits the workflow", expanded=False):
    st.markdown("""
1. Build a baseline result → 2. Explain & check interpretability → **3. Add targeted tests (this page)** → 4. Export

EDA already generates automatic descriptive statistics. This page is for **targeted additions**, not repeating what you already ran. Custom test results merge into the Export page.
""")

# Progress indicator

# Check prerequisites — allow both prediction and hypothesis_testing modes
task_mode = st.session_state.get('task_mode')
if task_mode not in ('hypothesis_testing', 'prediction'):
    st.warning("⚠️ **Please select a task mode first.**")
    st.info("Go to the **Upload & Audit** page and select either **Prediction** or **Hypothesis Testing** as your task mode.")
    st.stop()

# Show context-appropriate guidance
if task_mode == 'prediction':
    pass  # Coaching layer handles context
    # Cross-reference EDA to warn about duplicate tests
    eda_results = st.session_state.get('eda_results', {})
    if eda_results:
        eda_test_types = [k for k in eda_results.keys() if 'test' in str(k).lower() or 'correlation' in str(k).lower()]
        if eda_test_types:
            st.warning(f"""
            ⚠️ **Note:** You already ran {len(eda_test_types)} analysis/test(s) in EDA. 
            Check that you're not duplicating those tests here. 
            Tests from EDA: {', '.join(str(t) for t in eda_test_types[:5])}
            """)

df = get_data()
if df is None:
    st.warning("Please upload data first in the Upload & Audit page")
    st.stop()
if len(df) == 0 or len(df.columns) == 0:
    st.warning("Your dataset is empty. Please upload data with at least one row and one column.")
    st.stop()

# Get column types
numeric_cols = get_numeric_columns(df)
categorical_cols = get_categorical_columns(df)

# ============================================================================
# DATA OVERVIEW - Show available variables FIRST
# ============================================================================
st.header("Your Data")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Rows", f"{len(df):,}")
with col2:
    st.metric("Numeric Variables", len(numeric_cols))
with col3:
    st.metric("Categorical Variables", len(categorical_cols))

with st.expander("View Available Variables", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Numeric Variables:**")
        if numeric_cols:
            for col in numeric_cols[:20]:  # Limit display
                st.caption(f"• {col}")
            if len(numeric_cols) > 20:
                st.caption(f"... and {len(numeric_cols) - 20} more")
        else:
            st.warning("No numeric variables detected")
    
    with col2:
        st.markdown("**Categorical Variables:**")
        if categorical_cols:
            for col in categorical_cols[:20]:
                n_unique = df[col].nunique()
                st.caption(f"• {col} ({n_unique} levels)")
            if len(categorical_cols) > 20:
                st.caption(f"... and {len(categorical_cols) - 20} more")
        else:
            st.warning("No categorical variables detected")

if not numeric_cols and not categorical_cols:
    st.error("No selectable columns found in the dataset. Check that your data has been loaded correctly.")
    st.stop()

st.markdown("---")

# ============================================================================
# CONFIGURATION
# ============================================================================
with st.sidebar:
    st.header("Test Settings")
    
    alpha_level = st.selectbox(
        "Significance Level (α)",
        options=[0.01, 0.05, 0.10],
        index=1,
        key="alpha_level",
        help="Threshold for statistical significance"
    )
    
    show_effect_size = st.checkbox(
        "Show Effect Size",
        value=True,
        key="show_effect_size",
        help="Display effect size measures where applicable"
    )
    
    st.divider()
    
    st.markdown("**Quick Reference:**")
    st.caption("• p < α: Reject null hypothesis")
    st.caption("• p ≥ α: Fail to reject null hypothesis")
    st.caption(f"• Current α = {alpha_level}")

# Test selection
st.header("Run Statistical Tests")

test_type = st.selectbox(
    "What do you want to test?",
    options=[
        "Correlation (two numeric variables)",
        "Two-sample comparison (numeric variable, two groups)",
        "Multi-group comparison (numeric variable, multiple groups)",
        "Categorical association (two categorical variables)",
        "Normality test (one numeric variable)",
        "Paired comparison (numeric variable, before/after)"
    ],
    key="test_type_selection",
    help="Select the type of statistical test you want to perform"
)

st.markdown("---")

# Test-specific UI and execution
if test_type == "Correlation (two numeric variables)":
    st.subheader("Correlation Test")
    render_guidance(
        "Correlation measures the strength and direction of association between two numeric variables. "
        "<strong>Pearson</strong> detects linear relationships (assumes normality). "
        "<strong>Spearman</strong> detects monotonic relationships (rank-based, robust to outliers). "
        "<strong>Kendall</strong> is also rank-based and works well for small samples.",
        icon="📊"
    )
    
    if len(numeric_cols) < 2:
        st.error(f"""
        **Need at least 2 numeric variables for correlation test.**
        
        Currently detected: {len(numeric_cols)} numeric variable(s).
        
        Available numeric columns: {', '.join(numeric_cols) if numeric_cols else 'None'}
        
        **Tip:** If you expected more numeric columns, check if they contain non-numeric data 
        or are being detected as categorical due to too few unique values.
        """)
        st.stop()
    
    st.markdown("**Select Variables to Correlate:**")
    col1, col2 = st.columns(2)
    with col1:
        var1 = st.selectbox("Variable 1 (X)", options=numeric_cols, key="corr_var1")
    with col2:
        var2_options = [c for c in numeric_cols if c != var1]
        if not var2_options:
            st.error("Need at least 2 different numeric variables")
            st.stop()
        var2 = st.selectbox("Variable 2 (Y)", options=var2_options, key="corr_var2")
    
    st.markdown("**Configuration:**")
    method = st.radio(
        "Correlation method",
        options=["Pearson", "Spearman", "Kendall"],
        key="corr_method",
        horizontal=True,
        help="Pearson: linear relationship (assumes normality). Spearman: monotonic relationship (rank-based, robust). Kendall: rank-based, good for small samples."
    )
    
    # Show sample size
    valid_data = df[[var1, var2]].dropna()
    st.caption(f"Sample size: {len(valid_data)} valid pairs (after removing missing values)")
    
    if st.button("Run Correlation Test", type="primary", key="run_corr"):
        with st.spinner("Calculating correlation..."):
            x = valid_data[var1].values
            y = valid_data[var2].values
            
            method_map = {"Pearson": "pearson", "Spearman": "spearman", "Kendall": "kendall"}
            r, p, test_name = correlation_test(
                x, y,
                method=method_map[method]
            )
            
            # Calculate effect size (r^2 for correlation)
            r_squared = r ** 2
            
            # Store results
            st.session_state.hypothesis_test_results = {
                'test_type': 'correlation',
                'var1': var1,
                'var2': var2,
                'method': method,
                'r': r,
                'r_squared': r_squared,
                'p': p,
                'test_name': test_name,
                'n': len(valid_data),
                'alpha': alpha_level
            }
            log_methodology(step='Statistical Validation', action=f'{method} correlation test', details={
                'var1': var1,
                'var2': var2,
                'test': test_name,
                'p_value': p,
                'r': r
            })
            try:
                from utils.workflow_provenance import get_provenance
                get_provenance().record_statistical_test(
                    test_name=test_name,
                    variable=f"{var1} ~ {var2}",
                    statistic=float(r) if r is not None else None,
                    p_value=float(p) if p is not None else None,
                )
            except Exception:
                pass  # Provenance recording should never break the workflow
            st.rerun()

    # Display results
    if st.session_state.get('hypothesis_test_results') and st.session_state.hypothesis_test_results.get('test_type') == 'correlation':
        results = st.session_state.hypothesis_test_results
        st.subheader("Results")
        
        is_significant = results['p'] < alpha_level
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Correlation (r)", f"{results['r']:.4f}")
        with col2:
            st.metric("p-value", format_pvalue(results['p']))
        with col3:
            sig = "Significant" if is_significant else "Not significant"
            st.metric(f"At α={alpha_level}", sig)
        with col4:
            if show_effect_size:
                st.metric("R² (effect)", f"{results['r_squared']:.4f}")
        
        # Interpretation
        abs_r = abs(results['r'])
        if abs_r < 0.1:
            strength = "negligible"
        elif abs_r < 0.3:
            strength = "weak"
        elif abs_r < 0.5:
            strength = "moderate"
        elif abs_r < 0.7:
            strength = "strong"
        else:
            strength = "very strong"
        
        direction = "positive" if results['r'] > 0 else "negative"
        
        if is_significant:
            st.success(f"""
            **Summary:**
            - {results['method']} correlation: **r = {results['r']:.4f}**
            - This indicates a **{strength} {direction}** relationship
            - **Statistically significant** (p = {format_pvalue(results['p'])} < α = {alpha_level})
            - R² = {results['r_squared']:.4f} ({results['r_squared']*100:.1f}% of variance explained)
            - Sample size: n = {results['n']}
            """)
        else:
            st.warning(f"""
            **Summary:**
            - {results['method']} correlation: **r = {results['r']:.4f}**
            - This indicates a **{strength} {direction}** relationship
            - **Not statistically significant** (p = {format_pvalue(results['p'])} ≥ α = {alpha_level})
            - Sample size: n = {results['n']}
            """)
        
        # Export to Table 1 button
        if st.button("📋 Add to Table 1", key="export_corr_table1"):
            # Store results in session state for Table 1 merging
            if 'custom_table1_tests' not in st.session_state:
                st.session_state['custom_table1_tests'] = []
            
            st.session_state['custom_table1_tests'].append({
                'variable': f"{results['var1']} vs {results['var2']}",
                'test': results['test_name'],
                'statistic': f"r = {results['r']:.3f}",
                'p_value': results['p'],
                'note': f"{results['method']} correlation"
            })
            st.success(f"✅ Test result saved! Will be added to Table 1 in Export page. ({len(st.session_state['custom_table1_tests'])} custom tests total)")
        
        # Scatter plot (trendline requires statsmodels; skip if unavailable)
        try:
            fig = px.scatter(
                valid_data, x=var1, y=var2,
                title=f"Scatter Plot: {var1} vs {var2}",
                trendline="ols"
            )
        except (ImportError, ZeroDivisionError):
            fig = px.scatter(
                valid_data, x=var1, y=var2,
                title=f"Scatter Plot: {var1} vs {var2}"
            )
        fig.add_annotation(
            text=f"r = {results['r']:.3f}, p = {format_pvalue(results['p'])}, n = {results['n']}",
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            align="left",
            bgcolor="white"
        )
        st.plotly_chart(fig, width="stretch")

        # LLM interpretation for correlation
        from utils.llm_ui import build_llm_context, render_interpretation_with_llm_button, gather_session_context
        _bg_corr = gather_session_context()
        _corr_summary = (f"method={results['method']}; r={results['r']:.4f}; p={results['p']:.3e}; "
                         f"r_squared={results['r_squared']:.4f}; n={results['n']}; "
                         f"alpha={alpha_level}; significant={'yes' if is_significant else 'no'}; "
                         f"strength={strength}; direction={direction}")
        ctx_corr = build_llm_context(
            "correlation", _corr_summary,
            where=f"Correlation: {results['var1']} vs {results['var2']}",
            sample_size=_bg_corr.pop("sample_size", results['n']),
            task_type=_bg_corr.pop("task_type", None),
            feature_names=_bg_corr.pop("feature_names", None),
            **_bg_corr,
        )
        render_interpretation_with_llm_button(ctx_corr, key="llm_corr", result_session_key="llm_result_corr", plot_type="correlation")

elif test_type == "Two-sample comparison (numeric variable, two groups)":
    st.subheader("Two-Sample Comparison")
    render_guidance(
        "Compare means between two independent groups. <strong>t-test</strong> (parametric) assumes normality. "
        "<strong>Mann-Whitney U</strong> (non-parametric) is robust when data is skewed or has outliers.",
        icon="📊"
    )
    
    if len(numeric_cols) == 0:
        st.error("Need at least 1 numeric variable for two-sample test")
        st.stop()
    if len(categorical_cols) == 0:
        st.error("Need at least 1 categorical variable to define groups")
        st.stop()
    
    numeric_var = st.selectbox("Numeric Variable", options=numeric_cols, key="two_sample_numeric")
    group_var = st.selectbox("Group Variable (categorical)", options=categorical_cols, key="two_sample_group")
    
    # Check group variable has exactly 2 groups
    unique_groups = df[group_var].dropna().unique()
    if len(unique_groups) != 2:
        st.warning(f"Group variable has {len(unique_groups)} groups. Two-sample test requires exactly 2 groups.")
        st.info(f"Groups found: {', '.join(map(str, unique_groups))}")
        st.stop()
    
    # Parametric vs non-parametric — defaulted by the distributions themselves
    group1_name, group2_name = unique_groups[0], unique_groups[1]
    group1_data = df[df[group_var] == group1_name][numeric_var].dropna().values
    group2_data = df[df[group_var] == group2_name][numeric_var].dropna().values
    use_parametric, parametric_default, assumption_basis = _parametric_choice(
        {str(group1_name): group1_data, str(group2_name): group2_data},
        key_prefix="two_sample_parametric",
        scope=f"{numeric_var}|{group_var}",
        checkbox_label="Use parametric test (t-test)",
        parametric_name="t-test",
        nonparametric_name="Mann-Whitney U",
        help_text="Uncheck to use Mann-Whitney U (non-parametric). The box is pre-set from the Shapiro-Wilk result above.",
    )

    if st.button("Run Two-Sample Test", type="primary", key="run_two_sample"):
        with st.spinner("Running test..."):

            stat, p, test_name = two_sample_location_test(
                group1_data, group2_data,
                parametric=use_parametric
            )
            
            st.session_state.hypothesis_test_results = {
                'test_type': 'two_sample',
                'numeric_var': numeric_var,
                'group_var': group_var,
                'group1': group1_name,
                'group2': group2_name,
                'group1_mean': float(np.mean(group1_data)),
                'group2_mean': float(np.mean(group2_data)),
                'stat': stat,
                'p': p,
                'test_name': test_name,
                'parametric': use_parametric,
                'parametric_default': parametric_default,
                'assumption_basis': assumption_basis,
                'assumption_overridden': use_parametric != parametric_default,
            }
            log_methodology(step='Statistical Validation', action=test_name, details={
                'numeric_var': numeric_var,
                'group_var': group_var,
                'groups': [str(group1_name), str(group2_name)],
                'p_value': p,
                'parametric': use_parametric,
                'assumption_basis': assumption_basis,
                'assumption_overridden': use_parametric != parametric_default,
            })
            try:
                from utils.workflow_provenance import get_provenance
                get_provenance().record_statistical_test(
                    test_name=test_name,
                    variable=f"{numeric_var} by {group_var}",
                    statistic=float(stat) if stat is not None else None,
                    p_value=float(p) if p is not None else None,
                    details={'parametric': use_parametric,
                             'assumption_basis': assumption_basis,
                             'assumption_overridden': use_parametric != parametric_default},
                )
            except Exception:
                pass  # Provenance recording should never break the workflow
            st.rerun()

    # Display results
    if st.session_state.get('hypothesis_test_results') and st.session_state.hypothesis_test_results.get('test_type') == 'two_sample':
        results = st.session_state.hypothesis_test_results
        st.subheader("Results")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(f"Mean ({results['group1']})", f"{results['group1_mean']:.4f}")
        with col2:
            st.metric(f"Mean ({results['group2']})", f"{results['group2_mean']:.4f}")
        with col3:
            st.metric("Test Statistic", f"{results['stat']:.4f}")
        with col4:
            st.metric("p-value", format_pvalue(results['p']))
        
        st.info(f"""
        **Summary:**
        - Test: **{results['test_name']}**
        - Mean difference: **{results['group1_mean'] - results['group2_mean']:.4f}**
        - p-value: **{format_pvalue(results['p'])}** ({'statistically significant' if results['p'] < alpha_level else 'not statistically significant'} at α={alpha_level})
        - This {'suggests' if results['p'] < alpha_level else 'does not suggest'} a significant difference between {results['group1']} and {results['group2']}
        """)
        if results.get('assumption_basis'):
            st.caption(
                f"Test selection: {results['assumption_basis']} → "
                f"{'author override' if results.get('assumption_overridden') else 'assumption check'} "
                f"chose the {'parametric' if results['parametric'] else 'non-parametric'} test."
            )
        
        # Export to Table 1 button
        if st.button("📋 Add to Table 1", key="export_ttest_table1"):
            if 'custom_table1_tests' not in st.session_state:
                st.session_state['custom_table1_tests'] = []
            
            st.session_state['custom_table1_tests'].append({
                'variable': results['numeric_var'],
                'test': results['test_name'],
                'statistic': f"Δ = {results['group1_mean'] - results['group2_mean']:.3f}",
                'p_value': results['p'],
                'note': f"Comparing {results['group1']} vs {results['group2']}"
            })
            st.success(f"✅ Test result saved! Will be added to Table 1 in Export page. ({len(st.session_state['custom_table1_tests'])} custom tests total)")
        
        # Box plot
        plot_df = pd.DataFrame({
            numeric_var: np.concatenate([
                df[df[group_var] == results['group1']][numeric_var].dropna().values,
                df[df[group_var] == results['group2']][numeric_var].dropna().values
            ]),
            group_var: [results['group1']] * len(df[df[group_var] == results['group1']][numeric_var].dropna()) +
                      [results['group2']] * len(df[df[group_var] == results['group2']][numeric_var].dropna())
        })
        fig = px.box(plot_df, x=group_var, y=numeric_var, title=f"Distribution: {numeric_var} by {group_var}")
        st.plotly_chart(fig, width="stretch")

        # LLM interpretation for two-sample comparison
        from utils.llm_ui import build_llm_context, render_interpretation_with_llm_button, gather_session_context
        _bg_ts = gather_session_context()
        _mean_diff = results['group1_mean'] - results['group2_mean']
        _ts_summary = (f"test={results['test_name']}; stat={results['stat']:.4f}; p={results['p']:.3e}; "
                       f"mean_{results['group1']}={results['group1_mean']:.4f}; "
                       f"mean_{results['group2']}={results['group2_mean']:.4f}; "
                       f"mean_diff={_mean_diff:.4f}; n1={results.get('n1', '?')}; n2={results.get('n2', '?')}")
        ctx_ts = build_llm_context(
            "two_group_comparison", _ts_summary,
            where=f"Two-sample: {results.get('numeric_var', '')} by {results['group1']} vs {results['group2']}",
            sample_size=_bg_ts.pop("sample_size", None),
            task_type=_bg_ts.pop("task_type", None),
            feature_names=_bg_ts.pop("feature_names", None),
            **_bg_ts,
        )
        render_interpretation_with_llm_button(ctx_ts, key="llm_twosamp", result_session_key="llm_result_twosamp", plot_type="two_group_comparison")

elif test_type == "Multi-group comparison (numeric variable, multiple groups)":
    st.subheader("Multi-Group Comparison")
    render_guidance(
        "Compare means across 3+ groups. <strong>ANOVA</strong> (parametric) assumes normality and equal variances. "
        "<strong>Kruskal-Wallis</strong> (non-parametric) is robust to violations. "
        "If significant, follow up with post-hoc tests to identify which groups differ.",
        icon="📊"
    )
    
    if len(numeric_cols) == 0:
        st.error("Need at least 1 numeric variable for multi-group test")
        st.stop()
    if len(categorical_cols) == 0:
        st.error("Need at least 1 categorical variable to define groups")
        st.stop()
    
    numeric_var = st.selectbox("Numeric Variable", options=numeric_cols, key="multi_group_numeric")
    group_var = st.selectbox("Group Variable (categorical)", options=categorical_cols, key="multi_group_group")
    
    unique_groups = df[group_var].dropna().unique()
    if len(unique_groups) < 2:
        st.error("Need at least 2 groups for multi-group comparison")
        st.stop()
    
    st.info(f"Groups found: {', '.join(map(str, unique_groups))}")
    
    groups_data = [
        df[df[group_var] == group][numeric_var].dropna().values
        for group in unique_groups
    ]
    use_parametric, parametric_default, assumption_basis = _parametric_choice(
        {str(g): d for g, d in zip(unique_groups, groups_data)},
        key_prefix="multi_group_parametric",
        scope=f"{numeric_var}|{group_var}",
        checkbox_label="Use parametric test (ANOVA)",
        parametric_name="ANOVA",
        nonparametric_name="Kruskal-Wallis",
        help_text="Uncheck to use Kruskal-Wallis (non-parametric). The box is pre-set from the Shapiro-Wilk results above.",
    )

    if st.button("Run Multi-Group Test", type="primary", key="run_multi_group"):
        with st.spinner("Running test..."):

            stat, p, test_name = k_sample_location_test(
                groups_data,
                parametric=use_parametric
            )
            
            group_means = {str(group): float(np.mean(df[df[group_var] == group][numeric_var].dropna())) for group in unique_groups}
            
            st.session_state.hypothesis_test_results = {
                'test_type': 'multi_group',
                'numeric_var': numeric_var,
                'group_var': group_var,
                'groups': [str(g) for g in unique_groups],
                'group_means': group_means,
                'stat': stat,
                'p': p,
                'test_name': test_name,
                'parametric': use_parametric,
                'parametric_default': parametric_default,
                'assumption_basis': assumption_basis,
                'assumption_overridden': use_parametric != parametric_default,
            }
            log_methodology(step='Statistical Validation', action=test_name, details={
                'numeric_var': numeric_var,
                'group_var': group_var,
                'n_groups': len(unique_groups),
                'p_value': p,
                'parametric': use_parametric,
                'assumption_basis': assumption_basis,
                'assumption_overridden': use_parametric != parametric_default,
            })
            try:
                from utils.workflow_provenance import get_provenance
                get_provenance().record_statistical_test(
                    test_name=test_name,
                    variable=f"{numeric_var} by {group_var}",
                    statistic=float(stat) if stat is not None else None,
                    p_value=float(p) if p is not None else None,
                    details={'parametric': use_parametric,
                             'assumption_basis': assumption_basis,
                             'assumption_overridden': use_parametric != parametric_default},
                )
            except Exception:
                pass  # Provenance recording should never break the workflow
            st.rerun()

    # Display results
    if st.session_state.get('hypothesis_test_results') and st.session_state.hypothesis_test_results.get('test_type') == 'multi_group':
        results = st.session_state.hypothesis_test_results
        st.subheader("Results")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Test Statistic", f"{results['stat']:.4f}")
        with col2:
            st.metric("p-value", format_pvalue(results['p']))
        
        st.write("**Group Means:**")
        means_df = pd.DataFrame([
            {'Group': group, 'Mean': mean}
            for group, mean in results['group_means'].items()
        ])
        table(means_df, key="group_means", width="stretch", hide_index=True)
        
        st.info(f"""
        **Summary:**
        - Test: **{results['test_name']}**
        - p-value: **{format_pvalue(results['p'])}** ({'statistically significant' if results['p'] < alpha_level else 'not statistically significant'} at α={alpha_level})
        - This {'suggests' if results['p'] < alpha_level else 'does not suggest'} a significant difference among groups
        - Note: If significant, consider post-hoc tests to identify which groups differ
        """)
        if results.get('assumption_basis'):
            st.caption(
                f"Test selection: {results['assumption_basis']} → "
                f"{'author override' if results.get('assumption_overridden') else 'assumption check'} "
                f"chose the {'parametric' if results['parametric'] else 'non-parametric'} test."
            )
        
        # Export to Table 1 button
        if st.button("📋 Add to Table 1", key="export_anova_table1"):
            if 'custom_table1_tests' not in st.session_state:
                st.session_state['custom_table1_tests'] = []
            
            st.session_state['custom_table1_tests'].append({
                'variable': results['numeric_var'],
                'test': results['test_name'],
                'statistic': f"F = {results['stat']:.3f}",
                'p_value': results['p'],
                'note': f"{len(results['group_means'])} groups compared"
            })
            st.success(f"✅ Test result saved! Will be added to Table 1 in Export page. ({len(st.session_state['custom_table1_tests'])} custom tests total)")
        
        # Box plot
        plot_df = df[[numeric_var, group_var]].dropna()
        fig = px.box(plot_df, x=group_var, y=numeric_var, title=f"Distribution: {numeric_var} by {group_var}")
        st.plotly_chart(fig, width="stretch")

        # LLM interpretation for multi-group comparison
        from utils.llm_ui import build_llm_context, render_interpretation_with_llm_button, gather_session_context
        _bg_mg = gather_session_context()
        _group_means_str = "; ".join(f"{g}={m:.4f}" for g, m in results.get('group_means', {}).items())
        _mg_summary = (f"test={results['test_name']}; stat={results['stat']:.4f}; p={results['p']:.3e}; "
                       f"n_groups={len(results.get('group_means', {}))}; group_means: {_group_means_str}")
        ctx_mg = build_llm_context(
            "multi_group_comparison", _mg_summary,
            where=f"Multi-group: {results.get('numeric_var', '')} by {group_var}",
            sample_size=_bg_mg.pop("sample_size", None),
            task_type=_bg_mg.pop("task_type", None),
            feature_names=_bg_mg.pop("feature_names", None),
            **_bg_mg,
        )
        render_interpretation_with_llm_button(ctx_mg, key="llm_multigrp", result_session_key="llm_result_multigrp", plot_type="multi_group_comparison")

elif test_type == "Categorical association (two categorical variables)":
    st.subheader("Categorical Association Test")
    render_guidance(
        "Test whether two categorical variables are associated. <strong>Chi-square</strong> works for most cases. "
        "<strong>Fisher's exact test</strong> is more accurate for 2×2 tables with small sample sizes (&lt;5 expected counts in any cell).",
        icon="📊"
    )
    
    if len(categorical_cols) < 2:
        st.error("Need at least 2 categorical variables for association test")
        st.stop()
    
    var1 = st.selectbox("Categorical Variable 1", options=categorical_cols, key="cat_var1")
    var2 = st.selectbox("Categorical Variable 2", options=[c for c in categorical_cols if c != var1], key="cat_var2")
    
    use_fisher = st.checkbox(
        "Use Fisher's exact test (for 2x2 tables)",
        value=False,
        key="use_fisher",
        help="Fisher's exact test is more accurate for small sample sizes in 2x2 tables"
    )
    
    if st.button("Run Association Test", type="primary", key="run_cat_assoc"):
        with st.spinner("Running test..."):
            contingency = pd.crosstab(df[var1], df[var2])
            
            stat, p, test_name = categorical_association_test(
                contingency.values,
                use_fisher=use_fisher
            )
            
            st.session_state.hypothesis_test_results = {
                'test_type': 'categorical_assoc',
                'var1': var1,
                'var2': var2,
                'contingency': contingency.to_dict(),
                'stat': stat,
                'p': p,
                'test_name': test_name
            }
            log_methodology(step='Statistical Validation', action=test_name, details={
                'var1': var1,
                'var2': var2,
                'p_value': p
            })
            try:
                from utils.workflow_provenance import get_provenance
                get_provenance().record_statistical_test(
                    test_name=test_name,
                    variable=f"{var1} ~ {var2}",
                    statistic=float(stat) if stat is not None else None,
                    p_value=float(p) if p is not None else None,
                )
            except Exception:
                pass  # Provenance recording should never break the workflow
            st.rerun()

    # Display results
    if st.session_state.get('hypothesis_test_results') and st.session_state.hypothesis_test_results.get('test_type') == 'categorical_assoc':
        results = st.session_state.hypothesis_test_results
        st.subheader("Results")
        
        # Contingency table
        contingency = pd.DataFrame(results['contingency'])
        st.write("**Contingency Table:**")
        table(contingency, key="contingency_table", width="stretch")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Test Statistic", f"{results['stat']:.4f}")
        with col2:
            st.metric("p-value", format_pvalue(results['p']))
        
        st.info(f"""
        **Summary:**
        - Test: **{results['test_name']}**
        - p-value: **{format_pvalue(results['p'])}** ({'statistically significant' if results['p'] < alpha_level else 'not statistically significant'} at α={alpha_level})
        - This {'suggests' if results['p'] < alpha_level else 'does not suggest'} an association between {var1} and {var2}
        """)
        
        # Export to Table 1 button
        if st.button("📋 Add to Table 1", key="export_chi_table1"):
            if 'custom_table1_tests' not in st.session_state:
                st.session_state['custom_table1_tests'] = []
            
            st.session_state['custom_table1_tests'].append({
                'variable': f"{results['var1']} vs {results['var2']}",
                'test': results['test_name'],
                # Fisher's exact returns an odds ratio as its statistic, not χ²
                'statistic': (f"OR = {results['stat']:.3f}" if 'fisher' in str(results['test_name']).lower()
                              else f"χ² = {results['stat']:.3f}"),
                'p_value': results['p'],
                'note': 'Categorical association'
            })
            st.success(f"✅ Test result saved! Will be added to Table 1 in Export page. ({len(st.session_state['custom_table1_tests'])} custom tests total)")
        
        # Heatmap
        fig = px.imshow(
            contingency,
            text_auto=True,
            aspect="auto",
            title=f"Contingency Table: {var1} vs {var2}",
            labels=dict(x=var2, y=var1, color="Count")
        )
        st.plotly_chart(fig, width="stretch")

        # LLM interpretation for chi-squared
        from utils.llm_ui import build_llm_context, render_interpretation_with_llm_button, gather_session_context
        _bg_chi = gather_session_context()
        _chi_summary = (f"test={results['test_name']}; chi2={results['stat']:.4f}; p={results['p']:.3e}; "
                        f"df={results.get('dof', '?')}; cramers_v={results.get('cramers_v', 'N/A')}; "
                        f"min_expected_count={results.get('min_expected', 'N/A')}")
        ctx_chi = build_llm_context(
            "chi_squared", _chi_summary,
            where=f"Chi-squared: {results.get('var1', '')} vs {results.get('var2', '')}",
            sample_size=_bg_chi.pop("sample_size", None),
            task_type=_bg_chi.pop("task_type", None),
            feature_names=_bg_chi.pop("feature_names", None),
            **_bg_chi,
        )
        render_interpretation_with_llm_button(ctx_chi, key="llm_chi2", result_session_key="llm_result_chi2", plot_type="chi_squared")

elif test_type == "Normality test (one numeric variable)":
    st.subheader("Normality Test")
    render_guidance(
        "Test whether a variable follows a normal (Gaussian) distribution. "
        "Many parametric tests (t-test, ANOVA, linear regression) assume normality. "
        "<strong>Shapiro-Wilk</strong> is sensitive to deviations and works well for small-to-medium samples.",
        icon="📊"
    )
    
    if len(numeric_cols) == 0:
        st.error("Need at least 1 numeric variable for normality test")
        st.stop()
    
    numeric_var = st.selectbox("Numeric Variable", options=numeric_cols, key="normality_var")
    
    if st.button("Run Normality Test", type="primary", key="run_normality"):
        with st.spinner("Running test..."):
            data = df[numeric_var].dropna().values
            
            stat, p, test_name = normality_check(data)
            
            st.session_state.hypothesis_test_results = {
                'test_type': 'normality',
                'var': numeric_var,
                'stat': stat,
                'p': p,
                'test_name': test_name,
                'mean': float(np.mean(data)),
                'std': float(np.std(data)),
                'n': len(data)
            }
            log_methodology(step='Statistical Validation', action=test_name, details={
                'var': numeric_var,
                'p_value': p,
                'n': len(data)
            })
            try:
                from utils.workflow_provenance import get_provenance
                get_provenance().record_statistical_test(
                    test_name=test_name,
                    variable=numeric_var,
                    statistic=float(stat) if stat is not None else None,
                    p_value=float(p) if p is not None else None,
                )
            except Exception:
                pass  # Provenance recording should never break the workflow
            st.rerun()

    # Display results
    if st.session_state.get('hypothesis_test_results') and st.session_state.hypothesis_test_results.get('test_type') == 'normality':
        results = st.session_state.hypothesis_test_results
        st.subheader("Results")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Test Statistic", f"{results['stat']:.4f}")
        with col2:
            st.metric("p-value", format_pvalue(results['p']))
        with col3:
            st.metric("Mean", f"{results['mean']:.4f}")
        with col4:
            st.metric("Std Dev", f"{results['std']:.4f}")
        
        is_normal = results['p'] >= 0.05
        st.info(f"""
        **Summary:**
        - Test: **{results['test_name']}**
        - p-value: **{format_pvalue(results['p'])}**
        - The data appears to be {'normally distributed' if is_normal else 'NOT normally distributed'} (p {'≥' if is_normal else '<'} 0.05)
        - Sample size: **{results['n']}**
        """)
        
        # Histogram with normal overlay
        fig = px.histogram(
            df, x=numeric_var,
            nbins=30,
            title=f"Distribution: {numeric_var}",
            labels={numeric_var: numeric_var, 'count': 'Frequency'}
        )
        st.plotly_chart(fig, width="stretch")

elif test_type == "Paired comparison (numeric variable, before/after)":
    st.subheader("Paired Comparison Test")
    render_guidance(
        "Compare two measurements on the same subjects (e.g., before/after treatment). "
        "<strong>Paired t-test</strong> (parametric) assumes differences are normally distributed. "
        "<strong>Wilcoxon signed-rank</strong> (non-parametric) is robust to non-normality.",
        icon="📊"
    )
    
    if len(numeric_cols) < 2:
        st.error("Need at least 2 numeric variables for paired comparison (before/after)")
        st.stop()
    
    st.info("Select two numeric variables representing paired measurements (e.g., before/after)")
    
    var_before = st.selectbox("Before/Time 1 Variable", options=numeric_cols, key="paired_before")
    var_after = st.selectbox("After/Time 2 Variable", options=[c for c in numeric_cols if c != var_before], key="paired_after")
    
    # The paired t-test's assumption is on the DIFFERENCES, so that is what the
    # pre-check tests — not the two measurement columns.
    paired_df = df[[var_before, var_after]].dropna()
    differences = (paired_df[var_after] - paired_df[var_before]).values
    use_parametric, parametric_default, assumption_basis = _parametric_choice(
        {f"{var_after} − {var_before}": differences},
        key_prefix="paired_parametric",
        scope=f"{var_before}|{var_after}",
        checkbox_label="Use parametric test (paired t-test)",
        parametric_name="paired t-test",
        nonparametric_name="Wilcoxon signed-rank",
        help_text="Uncheck to use the Wilcoxon signed-rank test (non-parametric). The box is pre-set from the Shapiro-Wilk result on the paired differences.",
    )

    if st.button("Run Paired Test", type="primary", key="run_paired"):
        with st.spinner("Running test..."):

            stat, p, test_name = paired_location_test(
                differences,
                parametric=use_parametric
            )
            
            st.session_state.hypothesis_test_results = {
                'test_type': 'paired',
                'var_before': var_before,
                'var_after': var_after,
                'before_mean': float(np.mean(paired_df[var_before])),
                'after_mean': float(np.mean(paired_df[var_after])),
                'mean_diff': float(np.mean(differences)),
                'stat': stat,
                'p': p,
                'test_name': test_name,
                'n_pairs': len(paired_df),
                'parametric': use_parametric,
                'parametric_default': parametric_default,
                'assumption_basis': assumption_basis,
                'assumption_overridden': use_parametric != parametric_default,
            }
            log_methodology(step='Statistical Validation', action=test_name, details={
                'var_before': var_before,
                'var_after': var_after,
                'n_pairs': len(paired_df),
                'p_value': p,
                'parametric': use_parametric,
                'assumption_basis': assumption_basis,
                'assumption_overridden': use_parametric != parametric_default,
            })
            try:
                from utils.workflow_provenance import get_provenance
                get_provenance().record_statistical_test(
                    test_name=test_name,
                    variable=f"{var_before} vs {var_after}",
                    statistic=float(stat) if stat is not None else None,
                    p_value=float(p) if p is not None else None,
                    details={'parametric': use_parametric,
                             'assumption_basis': assumption_basis,
                             'assumption_overridden': use_parametric != parametric_default},
                )
            except Exception:
                pass  # Provenance recording should never break the workflow
            st.rerun()

    # Display results
    if st.session_state.get('hypothesis_test_results') and st.session_state.hypothesis_test_results.get('test_type') == 'paired':
        results = st.session_state.hypothesis_test_results
        st.subheader("Results")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(f"Mean ({results['var_before']})", f"{results['before_mean']:.4f}")
        with col2:
            st.metric(f"Mean ({results['var_after']})", f"{results['after_mean']:.4f}")
        with col3:
            st.metric("Mean Difference", f"{results['mean_diff']:.4f}")
        with col4:
            st.metric("p-value", format_pvalue(results['p']))
        
        st.info(f"""
        **Summary:**
        - Test: **{results['test_name']}**
        - Mean difference: **{results['mean_diff']:.4f}**
        - p-value: **{format_pvalue(results['p'])}** ({'statistically significant' if results['p'] < alpha_level else 'not statistically significant'} at α={alpha_level})
        - Number of pairs: **{results['n_pairs']}**
        - This {'suggests' if results['p'] < alpha_level else 'does not suggest'} a significant change from {results['var_before']} to {results['var_after']}
        """)
        if results.get('assumption_basis'):
            st.caption(
                f"Test selection: {results['assumption_basis']} → "
                f"{'author override' if results.get('assumption_overridden') else 'assumption check'} "
                f"chose the {'parametric' if results['parametric'] else 'non-parametric'} test."
            )
        
        # Export to Table 1 button
        if st.button("📋 Add to Table 1", key="export_paired_table1"):
            if 'custom_table1_tests' not in st.session_state:
                st.session_state['custom_table1_tests'] = []
            
            st.session_state['custom_table1_tests'].append({
                'variable': f"{results['var_before']} → {results['var_after']}",
                'test': results['test_name'],
                'statistic': f"Δ = {results['mean_diff']:.3f}",
                'p_value': results['p'],
                'note': f"Paired comparison (n={results['n_pairs']})"
            })
            st.success(f"✅ Test result saved! Will be added to Table 1 in Export page. ({len(st.session_state['custom_table1_tests'])} custom tests total)")
        
        # Before/after plot
        plot_df = pd.DataFrame({
            'Value': np.concatenate([
                df[results['var_before']].dropna().values,
                df[results['var_after']].dropna().values
            ]),
            'Time': [results['var_before']] * len(df[results['var_before']].dropna()) +
                   [results['var_after']] * len(df[results['var_after']].dropna())
        })
        fig = px.box(plot_df, x='Time', y='Value', title=f"Comparison: {results['var_before']} vs {results['var_after']}")
        st.plotly_chart(fig, width="stretch")

# Family-wise error rate warning
#
# `DRIVE8-20`. The burden is the number of QUESTIONS asked of the data, and a
# comparison re-run under an author override is the same question with a
# different estimator. Counting rows counted the override twice, and the same
# double-count reached the Methods draft's multiplicity sentence. The identity
# of a comparison is the variables it is about, not the test that answered it.
_custom_tests = st.session_state.get('custom_table1_tests', [])
_comparisons = {(str(t.get('variable', '')), str(t.get('note', ''))): t
                for t in _custom_tests if isinstance(t, dict)}
if len(_comparisons) > 1:
    _n = len(_comparisons)
    _fwer = 1 - (1 - 0.05) ** _n
    _bonf = 0.05 / _n
    st.markdown("---")
    st.warning(
        f"⚠️ **Multiple comparisons:** You have run **{_n} distinct comparisons** in this session. "
        f"Without correction, the probability of at least one false positive is approximately "
        f"**{_fwer:.0%}** (family-wise error rate). "
        f"Consider Bonferroni-adjusted α = 0.05/{_n} = **{_bonf:.4f}** when interpreting results, "
        f"or use FDR correction if you have many planned comparisons."
    )

    # AUDIT-001. The warning above has been here all along and the MANUSCRIPT
    # did not carry it: the draft reported how many tests reached a raw
    # p < 0.05 with no correction named. The draft now declines to count an
    # uncorrected family — which is right, and it leaves the author no way to
    # get a corrected count either. This is that way.
    #
    # An ACT rather than a default, because Benjamini-Hochberg over "every test
    # I ran today" is a decision about what the family IS, and the app does not
    # get to make it silently. Pressing it records the correction against the
    # family, and the Methods draft then reports the corrected count naming the
    # method and the threshold.
    from utils.workflow_provenance import get_provenance
    _recorded = getattr(get_provenance().statistical_validation, "tests_run", []) \
        if get_provenance().statistical_validation else []
    if _recorded:
        if st.button("Apply Benjamini–Hochberg FDR correction to these tests",
                     key="apply_fdr_correction"):
            _summary = get_provenance().apply_multiplicity_correction(
                method="fdr_bh", alpha=0.05)
            st.success(
                f"Corrected {_summary['n_adjusted']} test(s): "
                f"**{_summary['n_significant']}** remain significant at "
                f"q < 0.05. At an uncorrected α = 0.05 about "
                f"{_summary['expected_by_chance']:.0f} of "
                f"{_summary['n_adjusted']} would clear the line by chance "
                f"alone. The Methods draft now reports the corrected count."
            )
        st.caption(
            "Until a correction is applied, the Methods draft states that these "
            "tests are uncorrected and does not report a count of significant "
            "ones — an uncorrected count on a wide table is the number a "
            "reviewer would object to."
        )

# Export results
if st.session_state.get('hypothesis_test_results'):
    st.markdown("---")
    st.subheader("Export Results")
    
    results = st.session_state.hypothesis_test_results
    results_df = pd.DataFrame([results])
    
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name="hypothesis_test_results.csv",
        mime="text/csv",
        key="download_results"
    )
