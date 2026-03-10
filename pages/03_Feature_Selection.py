"""
Page 03: Feature Selection
LASSO path, RFE-CV, univariate screening, stability selection.
Results feed into preprocessing for recommended feature sets.
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict

from utils.session_state import init_session_state, get_data, DataConfig
from utils.storyline import render_breadcrumb, render_page_navigation
from utils.theme import inject_custom_css, render_guidance, render_reviewer_concern, render_step_indicator, render_sidebar_workflow
from utils.table_export import table
from data_processor import get_numeric_columns

init_session_state()

st.set_page_config(page_title="Feature Selection", page_icon="🎯", layout="wide")
inject_custom_css()
render_sidebar_workflow(current_page="03_Feature")
render_step_indicator(3, "Feature Selection")
st.title("🎯 Feature Selection")
render_breadcrumb("03_Feature_Selection")
render_page_navigation("03_Feature_Selection")

# Prerequisites
df = get_data()
if df is None:
    st.warning("Please upload data in the Upload & Audit page first.")
    st.stop()

data_config: DataConfig = st.session_state.get('data_config')
if data_config is None or not data_config.target_col:
    st.warning("Please select target and features in the Upload & Audit page first.")
    st.stop()

task_mode = st.session_state.get('task_mode')
if task_mode != 'prediction':
    st.warning("⚠️ Feature Selection is available in Prediction mode only.")
    st.stop()

# ============================================================================
# WHY FEATURE SELECTION?
# ============================================================================
st.markdown("""
### Why Feature Selection?

After uploading and exploring your data, you likely have many features (predictors). 
Feature selection helps you:

1. **Remove redundant features** (e.g., BMI and Weight are highly correlated — keep one)
2. **Identify the most predictive variables** (focus your analysis)  
3. **Reduce overfitting** (fewer features = simpler, more generalizable models)
4. **Improve interpretability** (explain 5 key predictors vs. explaining 50)

This step uses multiple methods (LASSO, RFE-CV, Stability Selection) to find consensus features.
""")

# If feature engineering was applied, add note
if st.session_state.get('feature_engineering_applied'):
    n_engineered = len(st.session_state.get('engineered_feature_names', []))
    st.info(f"""
    💡 **Note:** You created {n_engineered} engineered features in the previous step. 
    Many may be redundant or unhelpful — feature selection will filter them.
    """)

# Get feature info
all_features = data_config.feature_cols
target_col = data_config.target_col
numeric_cols = get_numeric_columns(df)
numeric_features = [f for f in all_features if f in numeric_cols]

if len(numeric_features) < 2:
    st.warning("Feature selection requires at least 2 numeric features.")
    st.stop()

task_type = data_config.task_type or "regression"

render_guidance(
    "<strong>Why feature selection matters:</strong> Identifying the most informative predictors reduces overfitting, "
    "improves interpretability, and gives peer reviewers confidence that your variable selection is robust. "
    "Running multiple methods and checking <strong>consensus</strong> is best practice."
)
render_reviewer_concern(
    "Reviewers often ask: 'How did you select your variables?' Having multiple methods agree gives you a defensible answer."
)

st.info(f"📊 **{len(numeric_features)} numeric features** available for selection · Target: `{target_col}` ({task_type})")

# Prepare data (drop missing target)
mask = df[target_col].notna()
X = df.loc[mask, numeric_features].values
y = df.loc[mask, target_col].values

# Handle NaN in features (simple imputation for feature selection)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)

# ============================================================================
# Method selection
# ============================================================================

st.header("Select Methods")
st.caption("Run multiple methods and compare which features are consistently selected.")

col1, col2 = st.columns(2)
with col1:
    run_lasso = st.checkbox("LASSO Path", value=True,
                            help="Shows how features enter/leave the model as regularization changes. Best for identifying the strongest linear predictors.")
    run_rfe = st.checkbox("RFE-CV (Recursive Feature Elimination)", value=True,
                          help="Iteratively removes least important features. Finds the optimal subset size via cross-validation.")
with col2:
    run_univariate = st.checkbox("Univariate Screening (FDR-corrected)", value=True,
                                 help="Tests each feature individually against the target. FDR correction controls false discovery rate.")
    run_stability = st.checkbox("Stability Selection", value=False,
                                help="Runs LASSO on many random subsamples. Features selected consistently are most robust. Slower but very reliable.")

# Advanced settings
with st.expander("⚙️ Advanced Settings", expanded=False):
    cv_folds = st.slider("Cross-validation folds", 3, 10, 5, key="fs_cv_folds")
    fdr_alpha = st.number_input("FDR significance level (α)", 0.01, 0.20, 0.05, 0.01, key="fs_alpha")
    stability_threshold = st.slider("Stability selection threshold", 0.3, 0.9, 0.6, 0.05, key="fs_stability_thresh")
    n_stability_bootstrap = st.slider("Stability bootstrap resamples", 50, 200, 100, key="fs_n_bootstrap")
    random_seed = st.session_state.get("random_seed", 42)

# ============================================================================
# Run feature selection
# ============================================================================

# Warn about wide datasets
n_features = len(numeric_features)
n_samples = len(X)
if n_features > 200:
    st.warning(
        f"⚠️ **Wide dataset detected:** {n_features} features × {n_samples} samples. "
        f"Feature selection may take several minutes. "
        f"{'**RFE is especially slow with this many features — consider disabling it.**' if n_features > 500 else ''}"
    )
    if n_features > n_samples:
        st.info(
            "💡 **p >> n scenario** (more features than samples). "
            "LASSO and univariate screening are best suited here. "
            "RFE and stability selection may be unreliable with so few samples."
        )

if st.button("🔍 Run Feature Selection", type="primary"):
    import signal, functools
    from ml.feature_selection import (
        lasso_path_selection, rfe_cv_selection,
        univariate_screening, stability_selection, consensus_features,
    )

    results = []
    progress = st.progress(0)
    status = st.empty()

    methods_to_run = []
    if run_lasso:
        methods_to_run.append("lasso")
    if run_rfe:
        methods_to_run.append("rfe")
    if run_univariate:
        methods_to_run.append("univariate")
    if run_stability:
        methods_to_run.append("stability")

    for i, method in enumerate(methods_to_run):
        pct = (i + 1) / len(methods_to_run)

        if method == "lasso":
            status.text(f"Running LASSO path analysis ({n_features} features)...")
            try:
                result = lasso_path_selection(
                    X, y, numeric_features, task_type,
                    cv_folds=cv_folds, random_state=random_seed,
                )
                results.append(result)
            except Exception as e:
                st.warning(f"⚠️ LASSO failed: {e}")

        elif method == "rfe":
            if n_features > 500:
                status.text(f"Running RFE-CV ({n_features} features — this will be slow)...")
            else:
                status.text("Running Recursive Feature Elimination (CV)...")
            try:
                result = rfe_cv_selection(
                    X, y, numeric_features, task_type,
                    cv_folds=cv_folds, random_state=random_seed,
                )
                results.append(result)
            except Exception as e:
                st.warning(f"⚠️ RFE failed: {e}")

        elif method == "univariate":
            status.text("Running univariate screening with FDR correction...")
            try:
                result = univariate_screening(
                    X, y, numeric_features, task_type,
                    alpha=fdr_alpha, correction="fdr_bh",
                )
                results.append(result)
            except Exception as e:
                st.warning(f"⚠️ Univariate screening failed: {e}")

        elif method == "stability":
            status.text(f"Running stability selection ({n_stability_bootstrap} bootstraps × {n_features} features)...")
            try:
                result = stability_selection(
                    X, y, numeric_features, task_type,
                    n_bootstrap=n_stability_bootstrap,
                    threshold=stability_threshold,
                    random_state=random_seed,
                )
                results.append(result)
            except Exception as e:
                st.warning(f"⚠️ Stability selection failed: {e}")

        progress.progress(pct)

    status.text("Done!")
    st.session_state["feature_selection_results"] = results

    # Consensus
    consensus = consensus_features(results, min_methods=max(1, len(results) // 2))
    st.session_state["consensus_features"] = consensus

    st.success(f"Feature selection complete! {len(results)} methods run.")

# ============================================================================
# Display results
# ============================================================================

results = st.session_state.get("feature_selection_results", [])
if results:
    st.header("Results")

    # Per-method results
    for result in results:
        with st.expander(f"**{result.method}** — {len(result.selected_features)}/{len(result.all_features)} features selected", expanded=True):
            st.markdown(result.description)

            # Score table
            scores_df = pd.DataFrame([
                {"Feature": f, "Score": s, "Selected": "✅" if f in result.selected_features else ""}
                for f, s in sorted(result.scores.items(), key=lambda x: -x[1])
            ])
            # Use method name in key to avoid duplicates in loop
            safe_method = result.method.replace(" ", "_").replace("-", "_").lower()
            table(scores_df, key=f"feature_scores_{safe_method}", use_container_width=True, hide_index=True)

            # LASSO-specific: coefficient path plot
            if result.method == "LASSO" and "path_coefs" in result.details and result.details.get("alphas"):
                try:
                    import plotly.graph_objects as go
                    alphas = np.array(result.details["alphas"])
                    coefs = np.array(result.details["path_coefs"])
                    fig = go.Figure()
                    for j, fname in enumerate(numeric_features):
                        fig.add_trace(go.Scatter(
                            x=np.log10(alphas), y=coefs[j, :],
                            mode='lines', name=fname,
                        ))
                    fig.add_vline(
                        x=np.log10(result.details["optimal_alpha"]),
                        line_dash="dash", line_color="red",
                        annotation_text="Optimal α",
                    )
                    fig.update_layout(
                        title="LASSO Coefficient Path",
                        xaxis_title="log₁₀(α)",
                        yaxis_title="Coefficient",
                        height=400,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.caption(f"Could not render LASSO path plot: {e}")

            # Stability-specific: selection probability bar chart
            if result.method == "Stability Selection" and "selection_probabilities" in result.details:
                try:
                    import plotly.express as px
                    probs = result.details["selection_probabilities"]
                    prob_df = pd.DataFrame([
                        {"Feature": f, "Selection Probability": p}
                        for f, p in sorted(probs.items(), key=lambda x: -x[1])
                    ])
                    fig = px.bar(
                        prob_df, x="Feature", y="Selection Probability",
                        title="Stability Selection Probabilities",
                    )
                    fig.add_hline(
                        y=stability_threshold, line_dash="dash", line_color="red",
                        annotation_text=f"Threshold ({stability_threshold})",
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    pass

    # Consensus features
    st.header("Consensus Features")
    consensus = st.session_state.get("consensus_features", [])

    if consensus:
        st.success(f"**{len(consensus)} features** selected by multiple methods:")

        # Build consensus matrix
        matrix_data = []
        for f in numeric_features:
            row = {"Feature": f}
            count = 0
            for result in results:
                selected = f in result.selected_features
                row[result.method] = "✅" if selected else ""
                if selected:
                    count += 1
            row["Count"] = count
            matrix_data.append(row)

        matrix_df = pd.DataFrame(matrix_data).sort_values("Count", ascending=False)
        table(matrix_df, key="consensus_matrix", use_container_width=True, hide_index=True)

        # Apply to data config
        st.markdown("---")
        if st.button("📋 Use consensus features for modeling", type="primary"):
            data_config.feature_cols = consensus
            st.session_state['data_config'] = data_config
            st.success(f"Updated feature set to {len(consensus)} consensus features. Proceed to Preprocessing.")
    else:
        st.warning("No consensus features found. Try lowering the threshold or running more methods.")

    # Option to manually select
    with st.expander("🔧 Manual feature selection", expanded=False):
        st.caption("Override the automatic selection by manually choosing features.")
        manual_selection = st.multiselect(
            "Select features",
            options=numeric_features,
            default=consensus if consensus else numeric_features,
            key="manual_feature_selection",
        )
        if st.button("Apply manual selection"):
            data_config.feature_cols = manual_selection
            st.session_state['data_config'] = data_config
            st.success(f"Updated feature set to {len(manual_selection)} features.")
