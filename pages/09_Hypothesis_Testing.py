"""
Page 09: Statistical Validation
Generate traditional statistical tests to validate ML findings and populate Table 1.
These tests provide p-values and effect sizes required for publication.
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
import logging

from utils.session_state import init_session_state, get_data
from utils.storyline import render_breadcrumb, render_page_navigation
from utils.theme import inject_custom_css, render_guidance, render_sidebar_workflow
from utils.table_export import table
from data_processor import get_numeric_columns, get_categorical_columns
from ml.stats_tests import (
    correlation_test,
    two_sample_location_test,
    k_sample_location_test,
    categorical_association_test,
    normality_check,
    paired_location_test
)

logger = logging.getLogger(__name__)

init_session_state()

st.set_page_config(page_title="Statistical Validation", page_icon="📊", layout="wide")
inject_custom_css()
render_sidebar_workflow(current_page="08_Hypothesis")
st.title("📊 Statistical Validation")
render_breadcrumb("09_Hypothesis_Testing")
render_page_navigation("09_Hypothesis_Testing")

st.markdown("""
### Why Statistical Validation?

Your ML model is trained and performing well. But reviewers will ask:

**"Did you test your features statistically?"**

This page generates traditional statistical tests that:
1. ✅ **Validate ML findings** — Confirm important features differ between groups
2. ✅ **Populate Table 1** — Generate p-values for baseline characteristics table
3. ✅ **Strengthen your paper** — Show both ML and statistical evidence

**ML vs Statistics:**
- **ML:** "Glucose is the most important predictor (SHAP value 0.42)"
- **Statistics:** "Glucose differs significantly between groups (t-test p<0.001, Cohen's d=0.85)"

**For publication:** Report both. ML shows prediction power, stats show group differences.
""")

st.markdown("---")

st.markdown("""
### 📄 How This Fits Your Publication

**Your workflow:**
1. ✅ Built ML model (pages 1-6)
2. ✅ Explained predictions (page 7)
3. ✅ Validated robustness (page 8)
4. **NOW:** Generate statistics for Table 1 and Methods section
5. **NEXT:** Export everything (page 10)

**These tests will appear in:**
- **Table 1:** Baseline characteristics with p-values
- **Methods:** "Univariate associations were tested using..."
- **Results:** "Features X, Y, Z differed significantly between groups"
""")

st.markdown("---")

# Progress indicator

# Check prerequisites
task_mode = st.session_state.get('task_mode')
if task_mode != 'hypothesis_testing':
    st.warning("⚠️ **Statistical Validation mode not selected.**")
    st.info("""
    Please go to the **Upload & Audit** page and select **Hypothesis Testing** as your task mode.
    
    Alternatively, you can use this page in "exploration mode" by clicking below.
    """)
    if st.button("Enable Statistical Validation for This Session", key="enable_hyp_test"):
        st.session_state.task_mode = 'hypothesis_testing'
        st.rerun()
    st.stop()

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
st.header("Generate Table 1 Statistics")

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
            st.metric("p-value", f"{results['p']:.4f}")
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
            **Interpretation:**
            - {results['method']} correlation: **r = {results['r']:.4f}**
            - This indicates a **{strength} {direction}** relationship
            - **Statistically significant** (p = {results['p']:.4f} < α = {alpha_level})
            - R² = {results['r_squared']:.4f} ({results['r_squared']*100:.1f}% of variance explained)
            - Sample size: n = {results['n']}
            """)
        else:
            st.warning(f"""
            **Interpretation:**
            - {results['method']} correlation: **r = {results['r']:.4f}**
            - This indicates a **{strength} {direction}** relationship
            - **Not statistically significant** (p = {results['p']:.4f} ≥ α = {alpha_level})
            - Sample size: n = {results['n']}
            """)
        
        # Export to Table 1 button
        if st.button("📋 Copy p-values to Table 1", key="export_corr_table1"):
            # Store results in session state for Table 1 generation
            if 'table1_pvalues' not in st.session_state:
                st.session_state['table1_pvalues'] = {}
            st.session_state['table1_pvalues'][f"{results['var1']} vs {results['var2']}"] = {
                'test': results['test_name'],
                'statistic': results['r'],
                'p_value': results['p'],
                'method': results['method']
            }
            st.success("✅ p-values saved! These will appear in your Table 1 (Export page)")
        
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
            text=f"r = {results['r']:.3f}, p = {results['p']:.4f}, n = {results['n']}",
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            align="left",
            bgcolor="white"
        )
        st.plotly_chart(fig, width="stretch")

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
    
    # Parametric vs non-parametric
    use_parametric = st.checkbox(
        "Use parametric test (t-test)",
        value=True,
        key="two_sample_parametric",
        help="Uncheck to use Mann-Whitney U (non-parametric). Use parametric if data is normally distributed."
    )
    
    if st.button("Run Two-Sample Test", type="primary", key="run_two_sample"):
        with st.spinner("Running test..."):
            group1_name, group2_name = unique_groups[0], unique_groups[1]
            group1_data = df[df[group_var] == group1_name][numeric_var].dropna().values
            group2_data = df[df[group_var] == group2_name][numeric_var].dropna().values
            
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
                'parametric': use_parametric
            }
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
            st.metric("p-value", f"{results['p']:.4f}")
        
        st.info(f"""
        **Interpretation:**
        - Test: **{results['test_name']}**
        - Mean difference: **{results['group1_mean'] - results['group2_mean']:.4f}**
        - p-value: **{results['p']:.4f}** ({'statistically significant' if results['p'] < 0.05 else 'not statistically significant'} at α=0.05)
        - This {'suggests' if results['p'] < 0.05 else 'does not suggest'} a significant difference between {results['group1']} and {results['group2']}
        """)
        
        # Export to Table 1 button
        if st.button("📋 Copy p-values to Table 1", key="export_ttest_table1"):
            if 'table1_pvalues' not in st.session_state:
                st.session_state['table1_pvalues'] = {}
            st.session_state['table1_pvalues'][f"{results['numeric_var']} by {results['group_var']}"] = {
                'test': results['test_name'],
                'statistic': results['stat'],
                'p_value': results['p'],
                'group1_mean': results['group1_mean'],
                'group2_mean': results['group2_mean']
            }
            st.success("✅ p-values saved! These will appear in your Table 1 (Export page)")
        
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
    
    use_parametric = st.checkbox(
        "Use parametric test (ANOVA)",
        value=True,
        key="multi_group_parametric",
        help="Uncheck to use Kruskal-Wallis (non-parametric)."
    )
    
    if st.button("Run Multi-Group Test", type="primary", key="run_multi_group"):
        with st.spinner("Running test..."):
            groups_data = [
                df[df[group_var] == group][numeric_var].dropna().values
                for group in unique_groups
            ]
            
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
                'parametric': use_parametric
            }
            st.rerun()
    
    # Display results
    if st.session_state.get('hypothesis_test_results') and st.session_state.hypothesis_test_results.get('test_type') == 'multi_group':
        results = st.session_state.hypothesis_test_results
        st.subheader("Results")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Test Statistic", f"{results['stat']:.4f}")
        with col2:
            st.metric("p-value", f"{results['p']:.4f}")
        
        st.write("**Group Means:**")
        means_df = pd.DataFrame([
            {'Group': group, 'Mean': mean}
            for group, mean in results['group_means'].items()
        ])
        table(means_df, key="group_means", width="stretch", hide_index=True)
        
        st.info(f"""
        **Interpretation:**
        - Test: **{results['test_name']}**
        - p-value: **{results['p']:.4f}** ({'statistically significant' if results['p'] < 0.05 else 'not statistically significant'} at α=0.05)
        - This {'suggests' if results['p'] < 0.05 else 'does not suggest'} a significant difference among groups
        - Note: If significant, consider post-hoc tests to identify which groups differ
        """)
        
        # Export to Table 1 button
        if st.button("📋 Copy p-values to Table 1", key="export_anova_table1"):
            if 'table1_pvalues' not in st.session_state:
                st.session_state['table1_pvalues'] = {}
            st.session_state['table1_pvalues'][f"{results['numeric_var']} by {results['group_var']}"] = {
                'test': results['test_name'],
                'statistic': results['stat'],
                'p_value': results['p'],
                'group_means': results['group_means']
            }
            st.success("✅ p-values saved! These will appear in your Table 1 (Export page)")
        
        # Box plot
        plot_df = df[[numeric_var, group_var]].dropna()
        fig = px.box(plot_df, x=group_var, y=numeric_var, title=f"Distribution: {numeric_var} by {group_var}")
        st.plotly_chart(fig, width="stretch")

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
            st.metric("p-value", f"{results['p']:.4f}")
        
        st.info(f"""
        **Interpretation:**
        - Test: **{results['test_name']}**
        - p-value: **{results['p']:.4f}** ({'statistically significant' if results['p'] < 0.05 else 'not statistically significant'} at α=0.05)
        - This {'suggests' if results['p'] < 0.05 else 'does not suggest'} an association between {var1} and {var2}
        """)
        
        # Export to Table 1 button
        if st.button("📋 Copy p-values to Table 1", key="export_chi_table1"):
            if 'table1_pvalues' not in st.session_state:
                st.session_state['table1_pvalues'] = {}
            st.session_state['table1_pvalues'][f"{results['var1']} vs {results['var2']}"] = {
                'test': results['test_name'],
                'statistic': results['stat'],
                'p_value': results['p']
            }
            st.success("✅ p-values saved! These will appear in your Table 1 (Export page)")
        
        # Heatmap
        fig = px.imshow(
            contingency,
            text_auto=True,
            aspect="auto",
            title=f"Contingency Table: {var1} vs {var2}",
            labels=dict(x=var2, y=var1, color="Count")
        )
        st.plotly_chart(fig, width="stretch")

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
            st.rerun()
    
    # Display results
    if st.session_state.get('hypothesis_test_results') and st.session_state.hypothesis_test_results.get('test_type') == 'normality':
        results = st.session_state.hypothesis_test_results
        st.subheader("Results")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Test Statistic", f"{results['stat']:.4f}")
        with col2:
            st.metric("p-value", f"{results['p']:.4f}")
        with col3:
            st.metric("Mean", f"{results['mean']:.4f}")
        with col4:
            st.metric("Std Dev", f"{results['std']:.4f}")
        
        is_normal = results['p'] >= 0.05
        st.info(f"""
        **Interpretation:**
        - Test: **{results['test_name']}**
        - p-value: **{results['p']:.4f}**
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
    
    use_parametric = st.checkbox(
        "Use parametric test (paired t-test)",
        value=True,
        key="paired_parametric",
        help="Uncheck to use Wilcoxon signed-rank test (non-parametric)"
    )
    
    if st.button("Run Paired Test", type="primary", key="run_paired"):
        with st.spinner("Running test..."):
            # Get paired data (drop rows where either is missing)
            paired_df = df[[var_before, var_after]].dropna()
            differences = (paired_df[var_after] - paired_df[var_before]).values
            
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
                'parametric': use_parametric
            }
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
            st.metric("p-value", f"{results['p']:.4f}")
        
        st.info(f"""
        **Interpretation:**
        - Test: **{results['test_name']}**
        - Mean difference: **{results['mean_diff']:.4f}**
        - p-value: **{results['p']:.4f}** ({'statistically significant' if results['p'] < 0.05 else 'not statistically significant'} at α=0.05)
        - Number of pairs: **{results['n_pairs']}**
        - This {'suggests' if results['p'] < 0.05 else 'does not suggest'} a significant change from {results['var_before']} to {results['var_after']}
        """)
        
        # Export to Table 1 button
        if st.button("📋 Copy p-values to Table 1", key="export_paired_table1"):
            if 'table1_pvalues' not in st.session_state:
                st.session_state['table1_pvalues'] = {}
            st.session_state['table1_pvalues'][f"{results['var_before']} vs {results['var_after']} (paired)"] = {
                'test': results['test_name'],
                'statistic': results['stat'],
                'p_value': results['p'],
                'mean_difference': results['mean_diff'],
                'before_mean': results['before_mean'],
                'after_mean': results['after_mean']
            }
            st.success("✅ p-values saved! These will appear in your Table 1 (Export page)")
        
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
