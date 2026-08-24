"""
Page 04: Feature Selection
LASSO path, RFE-CV, univariate screening, stability selection.
Results feed into preprocessing for recommended feature sets.

AUDIT NOTE (Data Flow):
- get_data() returns: df_engineered (if FE applied) > filtered_data > raw_data
- Operates on: data_config.feature_cols (if FE applied, includes engineered features)
- Methodology logging: Added for running analyses (already existed) AND for Apply actions (consensus/manual)
- Applying selection updates data_config.feature_cols, which downstream pages use
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict

from utils.session_state import init_session_state, get_data, DataConfig, log_methodology, reset_downstream_results
from utils.storyline import render_breadcrumb, render_page_navigation
from utils.theme import inject_custom_css, render_guidance, render_reviewer_concern, render_step_indicator, render_sidebar_workflow
from utils.table_export import table
from data_processor import get_numeric_columns

init_session_state()

st.set_page_config(page_title="Feature Selection", page_icon="🎯", layout="wide")
inject_custom_css()
render_sidebar_workflow(current_page="04_Feature_Selection")
render_step_indicator(4, "Feature Selection")
st.title("🎯 Feature Selection")
st.caption("Recommended workflow: use this step to simplify the modeling problem before you start tuning preprocessing or training multiple models.")
render_breadcrumb("04_Feature_Selection")
from utils.test_lockbox import render_lockbox_status as _render_lockbox_chip
_render_lockbox_chip("Selectors on this page are fit on training rows only.")
render_page_navigation("04_Feature_Selection")

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
# COACHING COMPANION
# ============================================================================
from utils.coaching_ui import render_page_coaching
render_page_coaching("04_Feature_Selection")

# ============================================================================
# WHY FEATURE SELECTION?
# ============================================================================
with st.expander("📖 Why feature selection?", expanded=False):
    st.markdown("""

After uploading and exploring your data, you likely have many features (predictors). 
Feature selection helps you:

1. **Remove redundant features** (e.g., BMI and Weight are highly correlated — keep one)
2. **Identify the most predictive variables** (focus your analysis)  
3. **Reduce model complexity** (fewer features = simpler models — though choosing them from these same rows adds optimism of its own, which is why selection belongs inside the validation loop)
4. **Improve interpretability** (explain 5 key predictors vs. explaining 50)

This step uses multiple methods (LASSO, RFE-CV, Stability Selection) to find consensus features.

**What the clinical-prediction literature says about selecting predictors from your own data.**
Univariable pre-screening by p-value is contraindicated: it is one of PROBAST's explicit
high-risk-of-bias signals, it discards variables that matter only in combination, and it
invalidates the p-values of the model you fit on the survivors. Stepwise selection draws the same
objection — unstable variable sets, biased coefficients, and confidence intervals with wrong
coverage. The preferred routes are pre-specifying predictors on clinical grounds, or a penalized
fit (LASSO, ridge, elastic net) that shrinks rather than testing and dropping — and LASSO is
itself unstable in small samples, which is what Stability Selection is here to show you.
(PROBAST: Wolff et al., *Ann Intern Med* 2019. Harrell, *Regression Modeling Strategies*, 2nd ed.)

Nothing here is taken away by that: these methods are the right tool for high-dimensional
discovery, where what survives is a set of hypotheses rather than a final predictor set. What
changed is that univariate screening is no longer ticked for you.
""")

# ============================================================================
# Data Source Indicator
# ============================================================================
if st.session_state.get('feature_engineering_applied'):
    n_engineered = len(st.session_state.get('engineered_feature_names', []))
    original_count = len(df.columns) - n_engineered - 1  # -1 for target
    total_features = len(df.columns) - 1
    
    st.success(
        f"📊 **Working Dataset:** Engineered Data\n\n"
        f"• Original features: {original_count}\n\n"
        f"• Engineered features: {n_engineered}\n\n"
        f"• Total features: {total_features}\n\n"
        f"💡 Feature selection will help identify which engineered features are actually useful."
    )

# Get feature info
target_col = data_config.target_col

# If feature engineering was applied, use ALL columns from df (except target)
# Otherwise use configured feature_cols
if st.session_state.get('feature_engineering_applied'):
    all_features = [col for col in df.columns if col != target_col]
else:
    all_features = data_config.feature_cols

numeric_cols = get_numeric_columns(df)
numeric_features = [f for f in all_features if f in numeric_cols]

if len(numeric_features) < 2:
    st.warning("Feature selection requires at least 2 numeric features.")
    st.stop()

task_type = data_config.task_type or "regression"

categorical_excluded = [f for f in all_features if f not in numeric_features]
st.caption(f"📊 {len(numeric_features)} numeric features available · Target: `{target_col}` ({task_type})")

if categorical_excluded:
    st.info(
        f"ℹ️ **{len(categorical_excluded)} non-numeric feature(s)** "
        f"({', '.join(categorical_excluded[:5])}"
        f"{'...' if len(categorical_excluded) > 5 else ''}) "
        f"are excluded from ranking — selection methods require numeric inputs. "
        f"They are carried through into the modeling feature set when you apply "
        f"a selection below, and the manual selector lets you drop them by hand."
    )


# `CONTRACT-014`. The caption above promises the non-ranked columns survive,
# and both Apply buttons used to write a numeric-only list into
# `data_config.feature_cols` — the very list pages/05 splits its categorical
# branch out of, so sex, smoking status and every other categorical predictor
# left the model by the same click that promised to keep them. Ranking stays
# numeric-only; APPLYING a ranking unions the non-ranked columns back in.
def _with_carried_categoricals(selected: List[str]) -> List[str]:
    """Ranked picks plus the non-ranked columns the caption promised to keep."""
    return list(selected) + [c for c in categorical_excluded if c not in selected]

# Prepare data: drop missing targets AND quarantine the locked test rows.
# Running selectors on all rows would let the test set vote on which
# predictors enter the model — the classic feature-selection leakage
# (Ambroise & McLachlan 2002; ESL §7.10.2).
from utils.test_lockbox import train_row_mask, is_exploratory, quarantine_is_active

mask = df[target_col].notna() & train_row_mask(df.index)
X = df.loc[mask, numeric_features].values
y = df.loc[mask, target_col].values
# The caption states what the MASK did, which is not the same question as
# whether exploratory mode is off. With no lockbox `train_row_mask` returns
# all-True and this said "held-out test rows are excluded" over a selection
# that had just voted with every row in the study (`MINE-005`). The chip
# rendered above says why there is no lockbox; this says what it cost here.
if quarantine_is_active():
    st.caption(
        f"Selection methods see n={int(mask.sum())} training rows; "
        f"held-out test rows are excluded to prevent selection leakage."
    )
elif not is_exploratory():
    st.warning(
        f"⚠️ No held-out test set is in force, so selection ran on all "
        f"n={int(mask.sum())} rows with a value for the outcome — including any "
        f"rows you later evaluate on. Predictors chosen this way carry selection "
        f"leakage (Ambroise & McLachlan 2002), and performance measured on those "
        f"rows afterwards is not held-out performance. Seal a test set on "
        f"**Upload & Audit** and re-run selection before reporting."
    )

# Handle NaN in features (simple imputation for feature selection)
# Note: This temporary imputation does not affect the modeling pipeline
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)

# Disclose imputation
# `AUDIT-007` · CLINICAL_SURVEY_PACK.md §A2 anti-patterns 2 and 3, both
# [SETTLED]. The fill above is median, and the outcome `y` is not in the
# imputation model. BEFORE, the two captions here said only "Results may be
# affected" and "(does not affect modeling data)" — true, and silent about
# what the fill costs the ranking a user is about to read. AFTER names the
# cost: §A2.2 "Understates variance, destroys the distribution" and §A2.3
# "Imputing with the outcome excluded from the imputation model. Biases
# associations toward the null." Nothing is removed; the scope disclaimer
# that was already true is kept beside it.
_high_missing = [f for f in numeric_features if df[f].isna().mean() > 0.2]
_IMPUTATION_COST = (
    "Filling with the median shrinks that column's variance and its association "
    "with the target is biased toward the null, because the outcome is not in "
    "this fill — so a column with many blanks can rank lower here than it "
    "deserves."
)
if _high_missing:
    st.caption(
        f"⚠️ Missing values temporarily filled with column medians for selection. "
        f"Features with >20% missing: {', '.join(_high_missing[:5])}. "
        f"{_IMPUTATION_COST} Preprocessing handles imputation separately during "
        f"training, where multiple imputation (MICE) is available under Advanced "
        f"(full control)."
    )
elif df.loc[mask, numeric_features].isna().any().any():
    st.caption(
        f"Missing values temporarily filled with column medians for selection "
        f"(does not affect modeling data). {_IMPUTATION_COST}"
    )

# ============================================================================
# Method selection
# ============================================================================

st.header("Select Methods")
st.caption("Run multiple methods and compare which features are consistently selected.")
# `AUDIT-024` · CLINICAL_SURVEY_PACK.md §A5.5 [SETTLED]. Every method on this
# panel is data-driven selection, and the panel used to state only what that is
# good for. The objection is stated here once, beside the controls.
st.caption(
    "All of these choose predictors from your rows, so the set they return is unstable across "
    "resamples and the p-values and confidence intervals of a model refitted on it are not valid "
    "as printed. For a clinical prediction or association model the literature's preference is "
    "pre-specification on clinical grounds, or a penalized fit that shrinks rather than selecting "
    "abruptly; stepwise selection in particular produces unstable variable sets, biased "
    "coefficients and confidence intervals with wrong coverage (Harrell, *Regression Modeling "
    "Strategies*, 2nd ed.)."
)

# RFE with step=1 fits ~p models over shrinking feature sets — measured
# ~80s at 3000 features on this hardware, so default it off on wide data.
_RFE_AUTO_DISABLE_FEATURES = 500
_rfe_too_wide = len(numeric_features) > _RFE_AUTO_DISABLE_FEATURES

col1, col2 = st.columns(2)
with col1:
    run_lasso = st.checkbox("LASSO Path", value=True,
                            help="Shows how features enter/leave the model as regularization changes. Best for identifying the strongest linear predictors.")
    run_rfe = st.checkbox("RFE-CV (Recursive Feature Elimination)", value=not _rfe_too_wide,
                          help="Iteratively removes least important features. Finds the optimal subset size via cross-validation.")
    if _rfe_too_wide and not run_rfe:
        st.caption(f"Off by default above {_RFE_AUTO_DISABLE_FEATURES} features "
                   "— recursive elimination takes minutes at this width. "
                   "Enable it if you need it.")
with col2:
    # `AUDIT-024` · CLINICAL_SURVEY_PACK.md §A5.5 [SETTLED]: "Avoid univariable
    # pre-screening of predictors by p-value. It is one of PROBAST's explicit
    # high-risk-of-bias signals." The method is NOT removed — it is offered with
    # the objection stated and is no longer pre-ticked, on the `GUIDED-049`
    # pattern (`ml/imbalance_advice.py`): keep it, stop recommending it, say what
    # the literature says. It stays the right tool for high-dimensional
    # discovery, and `ml.feature_selection.univariate_screening` is untouched.
    run_univariate = st.checkbox("Univariate Screening (FDR-corrected)", value=False,
                                 help="Tests each feature individually against the target and keeps those surviving "
                                      "Benjamini-Hochberg FDR correction. Off by default: univariable pre-screening by "
                                      "p-value is one of PROBAST's explicit high-risk-of-bias signals for a clinical "
                                      "prediction or association model — it discards variables that matter only in "
                                      "combination, and it invalidates the p-values of the model fitted on the survivors.")
    st.caption(
        "⚠️ Univariable pre-screening by p-value is a PROBAST high-risk-of-bias signal: it discards "
        "variables that matter only in combination, and it invalidates the p-values of the model you "
        "fit afterwards. Instead: pre-specify predictors on clinical grounds, or use a penalized fit "
        "(LASSO, ridge, elastic net). It is kept here — and it is the standard tool for "
        "high-dimensional discovery, where the survivors are hypotheses rather than a predictor set — "
        "so tick it if that is what you are doing (Wolff et al., *Ann Intern Med* 2019)."
    )
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
    _rfe_note = " RFE is the slow one — expect minutes if enabled." if run_rfe and n_features > _RFE_AUTO_DISABLE_FEATURES else ""
    st.caption(f"⚠️ {n_features} features × {n_samples} samples — selection may take a few minutes.{_rfe_note}")

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

    # `MISC-104`. Which methods COMPLETED, in the same vocabulary as
    # `methods_to_run`: every branch below catches its own failure and warns,
    # so a method that raised leaves `methods_to_run` unchanged and the
    # manuscript could not tell the two lists apart.
    methods_completed = []
    _n_results_done = 0

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

        if len(results) > _n_results_done:
            methods_completed.append(method)
        _n_results_done = len(results)

        progress.progress(pct)

    status.text("Done!")
    st.session_state["feature_selection_results"] = results

    # Consensus. `max(1, len(results) // 2)` made the threshold 1 for one, two
    # or three methods, so "consensus" named the UNION of the methods' picks —
    # while the success message below and the manuscript both read it as
    # agreement. Agreement means at least two methods selected the feature.
    consensus_threshold = max(2, len(results) // 2)
    consensus = consensus_features(results, min_methods=consensus_threshold)
    st.session_state["consensus_features"] = consensus
    
    # Log methodology action
    methods_used = ", ".join(methods_to_run)
    log_methodology(
        step='Feature Selection',
        action=f"Selected {len(consensus)} features using {methods_used}",
        details={
            'methods': methods_to_run,
            # `MISC-104`. REQUESTED and COMPLETED are different lists whenever a
            # method raises, and the consensus threshold is computed from the
            # COMPLETED ones (`len(results)` above). The Methods sentence said
            # "at least T of N methods" with T from one universe and N from the
            # other; it can only be written honestly if both are recorded.
            'methods_completed': list(methods_completed),
            'n_features_before': len(numeric_features),
            'n_features_after': len(consensus),
            'selected': list(consensus),
            'consensus_threshold': consensus_threshold,
        }
    )
    try:
        from utils.workflow_provenance import get_provenance
        get_provenance().record_feature_selection(
            method='consensus',
            n_before=len(numeric_features),
            n_after=len(consensus),
            features_kept=list(consensus),
            consensus_methods=list(methods_to_run),
            # `AUDIT-023`. The screened set, by name, recorded at the moment it
            # is still knowable — the apply buttons below overwrite
            # `data_config.feature_cols` in place and §A5.4 sizes for what was
            # screened, not for what survived.
            candidates_screened=list(numeric_features),
        )
    except Exception:
        pass  # Provenance recording should never break the workflow

    if len(results) < 2:
        st.info(
            "Only one method completed, so there is no consensus to report — "
            "two methods must agree before a feature is called a consensus "
            "predictor. Run a second method, or use manual selection below."
        )
    st.success(f"Feature selection complete! {len(results)} methods run.")

# ============================================================================
# Display results
# ============================================================================

results = st.session_state.get("feature_selection_results", [])
if results:
    st.header("Results")

    # Per-method results
    _method_tabs = st.tabs([
        f"{r.method} · {len(r.selected_features)}/{len(r.all_features)} kept" for r in results
    ])
    for result, _method_tab in zip(results, _method_tabs):
        with _method_tab:
            st.markdown(result.description)

            # Score table
            scores_df = pd.DataFrame([
                {"Feature": f, "Score": s, "Selected": "✅" if f in result.selected_features else ""}
                for f, s in sorted(result.scores.items(), key=lambda x: -x[1])
            ])
            # Use method name in key to avoid duplicates in loop
            safe_method = result.method.replace(" ", "_").replace("-", "_").lower()
            table(scores_df, key=f"feature_scores_{safe_method}", hide_index=True)

            # LASSO-specific: coefficient path plot
            if result.method == "LASSO" and "path_coefs" in result.details and result.details.get("alphas"):
                try:
                    import plotly.graph_objects as go
                    alphas = np.array(result.details["alphas"])
                    coefs = np.array(result.details["path_coefs"])
                    # One trace per feature is unreadable (and a huge payload)
                    # on wide data — keep the strongest paths only.
                    _LASSO_PLOT_MAX_TRACES = 20
                    _path_order = np.argsort(-np.abs(coefs).max(axis=1))
                    _shown = _path_order[:_LASSO_PLOT_MAX_TRACES]
                    fig = go.Figure()
                    for j in _shown:
                        fig.add_trace(go.Scatter(
                            x=np.log10(alphas), y=coefs[j, :],
                            mode='lines', name=numeric_features[j],
                        ))
                    if len(numeric_features) > _LASSO_PLOT_MAX_TRACES:
                        st.caption(
                            f"Showing the {_LASSO_PLOT_MAX_TRACES} strongest coefficient "
                            f"paths of {len(numeric_features)} features (ranked by max "
                            "|coefficient| along the path)."
                        )
                    fig.add_vline(
                        x=np.log10(result.details["optimal_alpha"]),
                        line_dash="dash", line_color="red",
                        annotation_text="Optimal α",
                    )
                    fig.update_layout(
                        title="LASSO Coefficient Path",
                        xaxis_title="log₁₀(α)",
                        yaxis_title="Standardized coefficient",
                        height=400,
                    )
                    st.plotly_chart(fig)
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
                    st.plotly_chart(fig)
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
        table(matrix_df, key="consensus_matrix", hide_index=True)

        # LLM interpretation for feature selection consensus
        from utils.llm_ui import build_llm_context, render_interpretation_with_llm_button, gather_session_context
        _bg_fs = gather_session_context()
        _methods_used = ", ".join(r.method for r in results)
        _consensus_str = ", ".join(consensus[:10])
        _n_total = len(numeric_features)
        _fs_summary = (f"methods: {_methods_used}; consensus_features ({len(consensus)}/{_n_total}): {_consensus_str}"
                       + (f", +{len(consensus)-10} more" if len(consensus) > 10 else ""))
        ctx_fs = build_llm_context(
            "feature_selection", _fs_summary,
            where="Feature selection consensus",
            sample_size=_bg_fs.pop("sample_size", None),
            task_type=_bg_fs.pop("task_type", None),
            feature_names=_bg_fs.pop("feature_names", numeric_features),
            **_bg_fs,
        )
        render_interpretation_with_llm_button(ctx_fs, key="llm_feat_sel", result_session_key="llm_result_feat_sel", plot_type="feature_selection")

        # Apply to data config
        st.markdown("---")
        if categorical_excluded:
            st.caption(
                f"Applying will model on the {len(consensus)} consensus "
                f"predictor(s) plus the {len(categorical_excluded)} non-ranked "
                f"feature(s) listed above ({', '.join(categorical_excluded[:5])}"
                f"{'...' if len(categorical_excluded) > 5 else ''}). Use manual "
                f"selection below to drop any of them."
            )
        if st.button("📋 Use consensus features for modeling", type="primary"):
            applied_features = _with_carried_categoricals(consensus)
            data_config.feature_cols = applied_features
            st.session_state['data_config'] = data_config
            st.session_state['selected_features'] = list(applied_features)
            # The feature set changed: any preprocessing pipeline, split, or
            # trained model built on the old set is stale and would name dropped
            # columns at train time. Clear them (keeping this selection and its
            # record) so Preprocess rebuilds against the new set.
            reset_downstream_results(clear_feature_engineering=False,
                                     clear_feature_selection=False)
            # Retrieve consensus_threshold from the analysis log
            consensus_threshold_logged = None
            for entry in st.session_state.get('methodology_log', []):
                if entry.get('step') == 'Feature Selection':
                    consensus_threshold_logged = entry.get('details', {}).get('consensus_threshold')
                    break
            log_methodology(step='Feature Selection Applied', action='Applied consensus feature selection', details={
                'method': 'consensus',
                'n_features_selected': len(applied_features),
                'features': list(applied_features),
                'n_consensus_ranked': len(consensus),
                'carried_through_unranked': list(categorical_excluded),
                'consensus_threshold': consensus_threshold_logged,
            })
            try:
                from utils.workflow_provenance import get_provenance
                _prov = get_provenance()
                _n_before = _prov.feature_selection.n_features_before if _prov.feature_selection else 0
                _methods = _prov.feature_selection.consensus_methods if _prov.feature_selection else []
                _prov.record_feature_selection(
                    method='consensus',
                    n_before=_n_before,
                    n_after=len(applied_features),
                    features_kept=list(applied_features),
                    consensus_methods=_methods,
                )
            except Exception:
                pass  # Provenance recording should never break the workflow
            _carried = len(applied_features) - len(consensus)
            st.success(
                f"Updated feature set to {len(applied_features)} features: "
                f"{len(consensus)} consensus predictors"
                + (f" plus {_carried} non-ranked feature(s) carried through"
                   if _carried else "")
                + ". Proceed to Preprocessing."
            )
    else:
        # `MISC-105`: the threshold has a floor of 2 (agreement means two methods
        # agreed) and no control lowers it, so "try lowering the threshold" named
        # an action the page does not offer.
        st.warning(
            "No consensus features found — no feature was selected by at least "
            "two of the methods that ran. Run another selection method above, or "
            "pick features yourself under **Manual feature selection** below."
        )

    # Option to manually select
    with st.expander("🔧 Manual feature selection", expanded=False):
        st.caption(
            "Override the automatic selection by manually choosing features. "
            "Non-ranked (non-numeric) features are listed too: they were not "
            "ranked above, but they are modeling features, so dropping one has "
            "to be a choice made here rather than a side effect of applying a "
            "ranking."
        )
        # Options include categorical_excluded: with a numeric-only options list
        # a categorical predictor could not be re-added by hand once dropped.
        manual_options = numeric_features + [
            c for c in categorical_excluded if c not in numeric_features
        ]
        manual_selection = st.multiselect(
            "Select features",
            options=manual_options,
            default=_with_carried_categoricals(consensus) if consensus else manual_options,
            key="manual_feature_selection",
        )
        if st.button("Apply manual selection"):
            data_config.feature_cols = manual_selection
            st.session_state['data_config'] = data_config
            st.session_state['selected_features'] = list(manual_selection)
            # Feature set changed → clear stale pipelines/splits/models built on
            # the old set (keeping this selection and its record).
            reset_downstream_results(clear_feature_engineering=False,
                                     clear_feature_selection=False)
            log_methodology(step='Feature Selection Applied', action='Applied manual feature selection', details={
                'method': 'manual',
                'n_features_selected': len(manual_selection),
                'features': manual_selection
            })
            try:
                from utils.workflow_provenance import get_provenance
                _prov = get_provenance()
                _n_before = _prov.feature_selection.n_features_before if _prov.feature_selection else 0
                _prov.record_feature_selection(
                    method='manual',
                    n_before=_n_before,
                    n_after=len(manual_selection),
                    features_kept=list(manual_selection),
                )
            except Exception:
                pass  # Provenance recording should never break the workflow
            st.success(f"Updated feature set to {len(manual_selection)} features.")
