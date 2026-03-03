"""
Page 06: Model Explainability
Permutation importance, SHAP, partial dependence, external validation, subgroup analysis.
"""
import streamlit as st
import numpy as np
import pandas as pd
import time
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional
from sklearn.inspection import permutation_importance, partial_dependence
import logging

from utils.session_state import (
    init_session_state, get_preprocessing_pipeline, DataConfig, get_data
)
from utils.storyline import render_progress_indicator, render_breadcrumb, render_page_navigation
from ml.estimator_utils import is_estimator_fitted
from ml.model_registry import get_registry
from utils.theme import inject_custom_css, render_step_indicator, render_guidance, render_reviewer_concern, render_sidebar_workflow
from sklearn.pipeline import Pipeline as SklearnPipeline

@st.cache_resource
def _get_registry_cached():
    return get_registry()

logger = logging.getLogger(__name__)

init_session_state()

st.set_page_config(page_title="Explainability", page_icon="🔬", layout="wide")
inject_custom_css()
render_sidebar_workflow(current_page="06_Explain")
render_step_indicator(6, "Explain & Validate")

# ── Page Header ─────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom: 1.5rem;">
    <h1 style="margin-bottom: 0.25rem;">🔬 Explain & Validate</h1>
    <p style="color: var(--text-secondary, #475569); font-size: 0.95rem; margin: 0;">
        Understand <em>why</em> your models make the predictions they do. Every analysis here
        strengthens your paper's methodology section.
    </p>
</div>
""", unsafe_allow_html=True)

render_breadcrumb("06_Explainability")
render_page_navigation("06_Explainability")
render_progress_indicator("06_Explainability")

# ── Guardrails ──────────────────────────────────────────────────
task_mode = st.session_state.get('task_mode')
if task_mode != 'prediction':
    st.warning("⚠️ **Model Explainability is only available in Prediction mode.**")
    st.info("Please go to the **Upload & Audit** page and select **Prediction** as your task mode.")
    st.stop()

if not st.session_state.get('trained_models'):
    st.warning("Please train models first in the Train & Compare page")
    st.info("**Next steps:** Go to Train & Compare page, prepare splits, and train at least one model.")
    st.stop()

data_config: DataConfig = st.session_state.get('data_config')
pipeline = get_preprocessing_pipeline()
X_test = st.session_state.get('X_test')
y_test = st.session_state.get('y_test')
feature_names = st.session_state.get('feature_names', [])

if X_test is None or y_test is None:
    st.warning("Please prepare data splits first")
    st.info("**Next steps:** Go to Train & Compare page and click 'Prepare Splits'.")
    st.stop()

registry = _get_registry_cached()

# ────────────────────────────────────────────────────────────────
# HELPER: Build full pipeline for a model
# ────────────────────────────────────────────────────────────────
def _get_pipeline_and_data(name):
    """Return (full_pipeline_or_estimator, X_test_for_perm, y_test_for_perm, X_test_raw_or_processed)."""
    estimator = st.session_state.get('fitted_estimators', {}).get(name)
    if estimator is None or not is_estimator_fitted(estimator):
        return None, None, None, None

    if name in st.session_state.get('fitted_preprocessing_pipelines', {}):
        prep_pipeline = st.session_state.fitted_preprocessing_pipelines[name]
        full_pipeline = SklearnPipeline([('preprocess', prep_pipeline), ('model', estimator)])
        df_raw = get_data()
        test_indices = st.session_state.get('test_indices')
        if df_raw is not None and data_config and test_indices is not None:
            try:
                X_raw = df_raw[data_config.feature_cols].iloc[test_indices]
                y_raw = df_raw[data_config.target_col].iloc[test_indices].values
                return full_pipeline, X_raw, y_raw, X_raw
            except:
                pass
        return estimator, X_test, y_test, X_test
    return estimator, X_test, y_test, X_test


def _to_dense_numpy(arr):
    if hasattr(arr, 'toarray'):
        out = arr.toarray()
    elif isinstance(arr, pd.DataFrame):
        out = np.asarray(arr.values, dtype=float)
    else:
        out = np.asarray(arr, dtype=float)
    return np.ascontiguousarray(out)


# ════════════════════════════════════════════════════════════════
# MAIN ANALYSIS: Run Everything
# ════════════════════════════════════════════════════════════════

trained = list(st.session_state.get('trained_models', {}).keys())

# Show what will be computed
st.markdown("""
<div class="glass-card" style="padding: 1.25rem;">
    <div style="display: flex; gap: 2rem; flex-wrap: wrap;">
        <div>
            <div style="font-weight: 600; margin-bottom: 0.25rem;">📊 Permutation Importance</div>
            <div style="font-size: 0.85rem; color: var(--text-secondary, #475569);">Which features matter most to each model</div>
        </div>
        <div>
            <div style="font-weight: 600; margin-bottom: 0.25rem;">🎯 SHAP Values</div>
            <div style="font-size: 0.85rem; color: var(--text-secondary, #475569);">How each feature pushes predictions up or down</div>
        </div>
        <div>
            <div style="font-weight: 600; margin-bottom: 0.25rem;">📈 Partial Dependence</div>
            <div style="font-size: 0.85rem; color: var(--text-secondary, #475569);">Marginal effect of each feature on predictions</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Configuration
with st.expander("⚙️ Analysis Configuration", expanded=False):
    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
    with col_cfg1:
        perm_repeats = st.slider("Permutation repeats", 5, 30, 10, key="perm_repeats")
    with col_cfg2:
        shap_background = st.slider("SHAP background samples", 50, 200, 100, 10, key="shap_bg",
                                     help="Background distribution for SHAP (larger = more accurate but slower)")
    with col_cfg3:
        shap_eval_size = st.slider("SHAP evaluation samples", 100, 500, 200, 50, key="shap_eval",
                                    help="Number of samples to explain (larger = more detailed)")

# Model SHAP support summary
shap_support_info = []
for name in trained:
    spec = registry.get(name)
    if spec:
        support = spec.capabilities.supports_shap
        label = {'tree': '⚡ TreeExplainer', 'linear': '⚡ LinearExplainer',
                 'kernel': '🐢 KernelExplainer', 'none': '❌ Not supported'}.get(support, '?')
        shap_support_info.append(f"**{name.upper()}**: {label}")
if shap_support_info:
    st.caption("SHAP methods: " + " · ".join(shap_support_info))

# ── Run Button ──────────────────────────────────────────────────
if st.button("🚀 Run Full Explainability Analysis", type="primary", use_container_width=True):
    t0 = time.perf_counter()
    total_steps = len(trained) * 3  # perm + shap + pdp per model
    step_count = 0
    overall_progress = st.progress(0)
    overall_status = st.empty()

    perm_results = {}
    shap_results = {}
    pdp_results = {}
    errors = []

    for name in trained:
        full_pipe, X_perm, y_perm, X_raw = _get_pipeline_and_data(name)
        if full_pipe is None:
            errors.append(f"{name}: Fitted estimator not found or not fitted. Please retrain.")
            step_count += 3
            overall_progress.progress(min(step_count / total_steps, 1.0))
            continue

        spec = registry.get(name)

        # ── 1. Permutation Importance ───────────────────────────
        overall_status.text(f"Permutation importance: {name.upper()}...")
        try:
            pi = permutation_importance(full_pipe, X_perm, y_perm, n_repeats=perm_repeats,
                                        random_state=42, n_jobs=-1)
            fn_by_model = st.session_state.get('feature_names_by_model', {})
            n = len(pi.importances_mean)
            base = list(fn_by_model.get(name, feature_names) or [])
            fnames = (base + [f"feature_{i}" for i in range(len(base), n)])[:n]
            perm_results[name] = {
                'importances_mean': pi.importances_mean,
                'importances_std': pi.importances_std,
                'feature_names': fnames,
            }
        except Exception as e:
            errors.append(f"{name} permutation: {e}")
        step_count += 1
        overall_progress.progress(min(step_count / total_steps, 1.0))

        # ── 2. SHAP ────────────────────────────────────────────
        overall_status.text(f"SHAP values: {name.upper()}...")
        try:
            import shap
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            shap_support = spec.capabilities.supports_shap if spec else 'kernel'
            if shap_support == 'none':
                errors.append(f"{name} SHAP: not supported for this model type")
            else:
                # Transform data for SHAP
                if isinstance(full_pipe, SklearnPipeline) and 'preprocess' in getattr(full_pipe, 'named_steps', {}):
                    prep = full_pipe.named_steps['preprocess']
                    X_transformed = prep.transform(X_raw)
                    X_bg = _to_dense_numpy(X_transformed[:min(shap_background, len(X_transformed))])
                    X_ev = _to_dense_numpy(X_transformed[:min(shap_eval_size, len(X_transformed))])
                    model_step = full_pipe.named_steps['model']
                else:
                    try:
                        X_bg = _to_dense_numpy(X_raw[:min(shap_background, len(X_raw))])
                        X_ev = _to_dense_numpy(X_raw[:min(shap_eval_size, len(X_raw))])
                    except (ValueError, TypeError) as e:
                        errors.append(f"{name} SHAP: requires numeric data ({str(e)[:60]})")
                        step_count += 1
                        overall_progress.progress(min(step_count / total_steps, 1.0))
                        continue
                    model_step = full_pipe

                # Choose explainer
                if shap_support == 'tree':
                    explainer = shap.TreeExplainer(model_step)
                    shap_values = explainer.shap_values(X_ev)
                elif shap_support == 'linear':
                    explainer = shap.LinearExplainer(model_step, X_bg)
                    shap_values = explainer.shap_values(X_ev)
                else:
                    task_type = data_config.task_type if data_config else 'regression'
                    bg_small = X_bg[:min(50, len(X_bg))]
                    if task_type == 'classification' and hasattr(model_step, 'predict_proba'):
                        explainer = shap.KernelExplainer(model_step.predict_proba, bg_small)
                    else:
                        explainer = shap.KernelExplainer(model_step.predict, bg_small)
                    shap_values = explainer.shap_values(X_ev)

                # Handle multiclass / multi-output SHAP values
                sv_raw = shap_values
                if isinstance(sv_raw, list):
                    # List of arrays — one per class
                    if len(sv_raw) == 2:
                        sv_plot = np.asarray(sv_raw[1])
                        class_label = "Class 1 (Positive)"
                    else:
                        sv_plot = np.asarray(sv_raw[0])
                        class_label = "Class 0"
                else:
                    sv_plot = np.asarray(sv_raw)
                    class_label = None

                # Ensure 2D (n_samples, n_features) — some explainers return 3D
                if sv_plot.ndim == 3:
                    # (n_samples, n_features, n_classes) → take last class
                    sv_plot = sv_plot[:, :, -1]
                elif sv_plot.ndim == 1:
                    sv_plot = sv_plot.reshape(1, -1)

                # Feature names for SHAP
                fn_by_model = st.session_state.get('feature_names_by_model', {})
                fn_for_shap = fn_by_model.get(name, feature_names)
                n_cols = X_ev.shape[1]
                fn_shap = list(fn_for_shap[:n_cols]) if len(fn_for_shap) >= n_cols else [f"Feature {i}" for i in range(n_cols)]

                shap_results[name] = {
                    'shap_values': sv_plot,
                    'X_eval': X_ev,
                    'feature_names': fn_shap,
                    'class_label': class_label,
                    'all_shap_values': shap_values,  # keep for class switching
                }
        except ImportError:
            errors.append(f"{name} SHAP: shap package not installed")
        except Exception as e:
            errors.append(f"{name} SHAP: {e}")
            logger.exception(f"SHAP error for {name}: {e}")
        step_count += 1
        overall_progress.progress(min(step_count / total_steps, 1.0))

        # ── 3. Partial Dependence (top 4 features from perm) ───
        overall_status.text(f"Partial dependence: {name.upper()}...")
        try:
            if name in perm_results and spec and spec.capabilities.supports_pdp:
                pi_data = perm_results[name]
                top_idx = np.argsort(pi_data['importances_mean'])[::-1][:4]
                top_features_idx = top_idx.tolist()

                if isinstance(full_pipe, SklearnPipeline) and 'preprocess' in getattr(full_pipe, 'named_steps', {}):
                    prep = full_pipe.named_steps['preprocess']
                    X_pdp = prep.transform(X_raw)
                    if hasattr(X_pdp, 'toarray'):
                        X_pdp = X_pdp.toarray()
                    model_for_pdp = full_pipe.named_steps['model']
                else:
                    X_pdp = _to_dense_numpy(X_raw)
                    model_for_pdp = full_pipe

                pd_result = partial_dependence(model_for_pdp, X_pdp, features=top_features_idx, kind='average')
                pdp_results[name] = {
                    'pd_result': pd_result,
                    'feature_indices': top_features_idx,
                    'feature_names': pi_data['feature_names'],
                }
        except Exception as e:
            errors.append(f"{name} PDP: {e}")
            logger.exception(f"PDP error for {name}: {e}")
        step_count += 1
        overall_progress.progress(min(step_count / total_steps, 1.0))

    # Store all results
    st.session_state.permutation_importance = perm_results
    st.session_state.shap_results = shap_results
    st.session_state.pdp_results = pdp_results

    elapsed = time.perf_counter() - t0
    st.session_state.setdefault("last_timings", {})["Full Explainability"] = round(elapsed, 2)
    overall_progress.empty()
    overall_status.empty()

    if errors:
        with st.expander(f"⚠️ {len(errors)} issue(s) during analysis", expanded=False):
            for err in errors:
                st.text(err)

    st.success(f"✅ Explainability analysis complete ({elapsed:.1f}s)")

# ════════════════════════════════════════════════════════════════
# DISPLAY RESULTS
# ════════════════════════════════════════════════════════════════

perm_data = st.session_state.get('permutation_importance', {})
shap_data = st.session_state.get('shap_results', {})
pdp_data = st.session_state.get('pdp_results', {})

if perm_data or shap_data:
    # Per-model tabs
    model_tabs = st.tabs([f"📊 {name.upper()}" for name in trained if name in perm_data or name in shap_data])

    for tab, name in zip(model_tabs, [n for n in trained if n in perm_data or n in shap_data]):
        with tab:
            # Sub-tabs within each model
            analysis_tabs = st.tabs(["Permutation Importance", "SHAP Values", "Partial Dependence"])

            # ── Permutation Importance Tab ──────────────────────
            with analysis_tabs[0]:
                if name in perm_data:
                    pd_info = perm_data[name]
                    _fn = pd_info['feature_names']
                    _im = pd_info['importances_mean']
                    _is = pd_info['importances_std']
                    n = min(len(_fn), len(_im), len(_is))
                    if n == 0:
                        st.warning("Empty permutation importance data.")
                    else:
                        importance_df = pd.DataFrame({
                            'Feature': _fn[:n],
                            'Importance': np.asarray(_im)[:n],
                            'Std': np.asarray(_is)[:n],
                        }).sort_values('Importance', ascending=False)

                        top_n = min(10, len(importance_df))
                        fig = px.bar(
                            importance_df.head(top_n),
                            x='Importance', y='Feature',
                            error_x='Std', orientation='h',
                            title=f"Top {top_n} Features by Permutation Importance",
                            color='Importance',
                            color_continuous_scale='Blues',
                        )
                        fig.update_layout(
                            yaxis={'categoryorder': 'total ascending'},
                            height=max(350, top_n * 40),
                            showlegend=False,
                            coloraxis_showscale=False,
                            margin=dict(l=10, r=10, t=40, b=10),
                        )
                        st.plotly_chart(fig, use_container_width=True, key=f"perm_chart_{name}")

                        with st.expander("Full rankings table"):
                            st.dataframe(importance_df, use_container_width=True, hide_index=True)

                        from ml.plot_narrative import narrative_permutation_importance
                        nar = narrative_permutation_importance(pd_info, model_name=name)
                        if nar:
                            st.markdown(f"**Interpretation:** {nar}")
                        from utils.llm_ui import build_llm_context, render_interpretation_with_llm_button
                        stats_summary = "; ".join(f"{r['Feature']}={r['Importance']:.4f}" for _, r in importance_df.head(5).iterrows())
                        ctx = build_llm_context("permutation_importance", stats_summary, model_name=name,
                                                existing=nar or "", feature_names=_fn[:n],
                                                sample_size=X_test.shape[0] if X_test is not None else None,
                                                task_type=data_config.task_type if data_config else None)
                        render_interpretation_with_llm_button(ctx, key=f"llm_perm_{name}", result_session_key=f"llm_result_perm_{name}")
                else:
                    st.info("Permutation importance was not computed for this model.")

            # ── SHAP Tab ────────────────────────────────────────
            with analysis_tabs[1]:
                if name in shap_data:
                    import matplotlib
                    matplotlib.use('Agg')
                    import matplotlib.pyplot as plt
                    import shap

                    s = shap_data[name]
                    sv = np.asarray(s['shap_values'])
                    X_ev = np.asarray(s['X_eval'])
                    fn = s['feature_names']
                    cl = s.get('class_label')

                    # Ensure 2D
                    if sv.ndim == 3:
                        sv = sv[:, :, -1]
                    if sv.ndim == 1:
                        sv = sv.reshape(1, -1)

                    # Align columns: SHAP values and X_eval must have same n_features
                    n_cols = min(X_ev.shape[1], sv.shape[1]) if sv.ndim == 2 else X_ev.shape[1]
                    X_ev = X_ev[:, :n_cols]
                    if sv.ndim == 2:
                        sv = sv[:, :n_cols]
                    fn_plot = fn[:n_cols] if len(fn) >= n_cols else [f"Feature {i}" for i in range(n_cols)]
                    X_plot_df = pd.DataFrame(X_ev, columns=fn_plot)

                    # Summary plot
                    fig_height = max(4, min(8, n_cols * 0.4))
                    fig, ax = plt.subplots(figsize=(10, fig_height))
                    shap.summary_plot(sv, X_plot_df, feature_names=fn_plot, show=False,
                                      plot_size=(10, fig_height))
                    if cl:
                        ax.set_title(f"SHAP Values ({cl})", fontsize=11)
                    st.pyplot(fig)
                    plt.close(fig)

                    # Mean absolute SHAP bar chart
                    mean_abs = np.abs(sv).mean(axis=0)
                    # Ensure mean_abs is 1D and aligned with feature names
                    mean_abs = np.asarray(mean_abs).ravel()[:len(fn_plot)]
                    shap_df = pd.DataFrame({
                        'Feature': fn_plot[:len(mean_abs)],
                        'Mean |SHAP|': mean_abs,
                    }).sort_values('Mean |SHAP|', ascending=False)

                    fig2 = px.bar(shap_df.head(10), x='Mean |SHAP|', y='Feature', orientation='h',
                                  title="Mean Absolute SHAP Value (Global Importance)",
                                  color='Mean |SHAP|', color_continuous_scale='Purples')
                    fig2.update_layout(yaxis={'categoryorder': 'total ascending'}, height=350,
                                       showlegend=False, coloraxis_showscale=False,
                                       margin=dict(l=10, r=10, t=40, b=10))
                    st.plotly_chart(fig2, use_container_width=True, key=f"shap_bar_{name}")

                    from ml.plot_narrative import narrative_shap
                    from utils.llm_ui import build_llm_context, render_interpretation_with_llm_button
                    nar = narrative_shap(sv, fn_plot, model_name=name)
                    if nar:
                        st.markdown(f"**Interpretation:** {nar}")
                    top_idx = np.argsort(mean_abs)[::-1][:5]
                    stats_summary = "; ".join(f"{fn_plot[i]}={mean_abs[i]:.3f}" for i in top_idx if i < len(fn_plot))
                    ctx = build_llm_context("SHAP", stats_summary, model_name=name, existing=nar or "",
                                            feature_names=fn_plot, sample_size=X_ev.shape[0],
                                            task_type=data_config.task_type if data_config else None)
                    render_interpretation_with_llm_button(ctx, key=f"llm_shap_{name}", result_session_key=f"llm_result_shap_{name}")
                else:
                    st.info("SHAP was not computed for this model. Check the issues log above.")

            # ── Partial Dependence Tab ──────────────────────────
            with analysis_tabs[2]:
                if name in pdp_data:
                    pd_info = pdp_data[name]
                    pd_result = pd_info['pd_result']
                    feat_idx = pd_info['feature_indices']
                    feat_names = pd_info['feature_names']

                    cols = st.columns(2)
                    for i, fidx in enumerate(feat_idx):
                        fname = feat_names[fidx] if fidx < len(feat_names) else f"Feature {fidx}"
                        with cols[i % 2]:
                            grid = pd_result['grid_values'][i]
                            avg = pd_result['average'][i].ravel()
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=grid, y=avg, mode='lines+markers',
                                                      line=dict(color='#667eea', width=2),
                                                      marker=dict(size=4)))
                            fig.update_layout(title=f"PDP: {fname}", xaxis_title=fname,
                                              yaxis_title="Partial Dependence",
                                              height=300, margin=dict(l=10, r=10, t=40, b=10))
                            st.plotly_chart(fig, use_container_width=True, key=f"pdp_{name}_{fidx}")
                else:
                    st.info("Partial dependence not available. Model may not support it, or permutation importance wasn't computed.")

else:
    render_guidance(
        "<strong>Ready to analyze.</strong> Click the button above to compute permutation importance, "
        "SHAP values, and partial dependence for all trained models in one pass.",
        icon="👆"
    )

# ════════════════════════════════════════════════════════════════
# CROSS-MODEL COMPARISON
# ════════════════════════════════════════════════════════════════
if perm_data and len(perm_data) > 1:
    st.header("Cross-Model Feature Importance")
    st.markdown("Compare which features matter most across different models.")

    # Build comparison dataframe
    all_features = set()
    for name, pd_info in perm_data.items():
        all_features.update(pd_info['feature_names'])
    all_features = sorted(all_features)

    comparison_data = {}
    for name, pd_info in perm_data.items():
        feat_imp = dict(zip(pd_info['feature_names'], pd_info['importances_mean']))
        comparison_data[name.upper()] = [feat_imp.get(f, 0) for f in all_features]

    comp_df = pd.DataFrame(comparison_data, index=all_features)
    comp_df['Mean'] = comp_df.mean(axis=1)
    comp_df = comp_df.sort_values('Mean', ascending=False)

    top_cross = min(10, len(comp_df))
    fig = go.Figure()
    for col in comp_df.columns[:-1]:  # skip Mean
        fig.add_trace(go.Bar(name=col, x=comp_df.index[:top_cross], y=comp_df[col][:top_cross]))
    fig.update_layout(barmode='group', title=f"Top {top_cross} Features Across Models",
                      height=400, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True, key="cross_model_importance")

    # Consensus features
    render_guidance(
        f"<strong>Consensus:</strong> Features that rank highly across multiple models are more likely to be "
        f"genuinely important. Look for features that appear in the top 5 for all models.",
        icon="🎯"
    )

# ════════════════════════════════════════════════════════════════
# BLAND–ALTMAN (regression, 2+ models)
# ════════════════════════════════════════════════════════════════
st.header("Bland–Altman Plot")
with st.expander("What is a Bland–Altman Plot?", expanded=False):
    st.markdown(
        "Compares **agreement** between two measurement methods (e.g. two models' predictions). "
        "X-axis: mean of the two; Y-axis: difference. Lines show mean difference and limits of agreement (mean ± 1.96 SD). "
        "Useful to see systematic bias and spread of disagreement between models."
    )
    from ml.plot_narrative import interpretation_bland_altman
    st.caption(f"**Interpreting these numbers:** {interpretation_bland_altman()}")

mr = st.session_state.get('model_results', {})
task_det = st.session_state.get('task_type_detection')
task_final = (task_det.final if task_det and task_det.final else None) or (data_config.task_type if data_config else None)
if task_final == 'regression' and len(mr) >= 2:
    models_with_pred = [n for n in mr if 'y_test_pred' in mr[n] and mr[n]['y_test_pred'] is not None]
    if len(models_with_pred) >= 2:
        col_ba1, col_ba2 = st.columns(2)
        with col_ba1:
            ma = st.selectbox("Model A", models_with_pred, key="bland_altman_a")
        with col_ba2:
            mb = st.selectbox("Model B", [m for m in models_with_pred if m != ma], key="bland_altman_b")
        if ma and mb:
            from visualizations import plot_bland_altman
            from ml.eval import analyze_bland_altman
            from ml.plot_narrative import narrative_bland_altman
            pa = np.asarray(mr[ma]['y_test_pred']).ravel()
            pb = np.asarray(mr[mb]['y_test_pred']).ravel()
            if len(pa) == len(pb):
                fig_ba = plot_bland_altman(pa, pb, title=f"Bland–Altman: {ma.upper()} vs {mb.upper()}", label_a=ma, label_b=mb)
                st.plotly_chart(fig_ba, use_container_width=True, key=f"bland_altman_{ma}_{mb}")
                ba_stats = analyze_bland_altman(pa, pb)
                nar = narrative_bland_altman(ba_stats, label_a=ma, label_b=mb)
                if nar:
                    st.markdown(f"**Interpretation:** {nar}")
                from utils.llm_ui import build_llm_context, render_interpretation_with_llm_button
                stats_summary = f"mean_diff={ba_stats.get('mean_diff', 0):.4f}; width_loa={ba_stats.get('width_loa', 0):.4f}; pct_outside={ba_stats.get('pct_outside_loa', 0):.1%}"
                ctx = build_llm_context(
                    "bland_altman", stats_summary, where=f"Bland-Altman ({ma} vs {mb})", existing=nar or "",
                    metrics=ba_stats, sample_size=len(pa), task_type=data_config.task_type if data_config else None,
                )
                render_interpretation_with_llm_button(ctx, key="llm_bland_altman_btn", result_session_key="llm_result_bland_altman")
            else:
                st.warning("Prediction lengths differ; cannot compare.")
    else:
        st.info("At least two models with test predictions are required.")
else:
    st.info("Bland–Altman is available for **regression** tasks with at least two trained models.")

# ════════════════════════════════════════════════════════════════
# EXTERNAL VALIDATION
# ════════════════════════════════════════════════════════════════
st.header("🔗 External Validation")
render_guidance(
    "<strong>Why this matters:</strong> Internal validation (train/test split) shows how well your model works "
    "on similar data. External validation — applying the model to a completely separate dataset — "
    "is the gold standard for publication.",
    icon="📋"
)

with st.expander("Upload External Validation Dataset", expanded=False):
    ext_file = st.file_uploader("Upload external dataset (CSV/Excel)", type=["csv", "xlsx", "xls"], key="ext_val_file")

    if ext_file is not None:
        from data_processor import load_tabular_data
        try:
            ext_df = load_tabular_data(ext_file, filename=ext_file.name)
            st.success(f"Loaded external dataset: {ext_df.shape[0]} rows × {ext_df.shape[1]} columns")

            required_cols = data_config.feature_cols + [data_config.target_col]
            missing_cols = [c for c in required_cols if c not in ext_df.columns]
            if missing_cols:
                st.error(f"Missing columns in external dataset: {missing_cols}")
            else:
                if st.button("Validate on External Dataset", type="primary", key="run_ext_val"):
                    from ml.bootstrap import bootstrap_all_regression_metrics, bootstrap_all_classification_metrics, format_metric_with_ci

                    ext_y = ext_df[data_config.target_col].values
                    ext_X = ext_df[data_config.feature_cols]

                    st.subheader("External Validation Results")
                    for name in st.session_state.get('trained_models', {}):
                        model_obj = st.session_state.trained_models[name]
                        pipeline_local = st.session_state.get("fitted_preprocessing_pipelines", {}).get(name)

                        try:
                            if pipeline_local is not None:
                                ext_X_t = pipeline_local.transform(ext_X)
                            else:
                                ext_X_t = np.array(ext_X)

                            ext_pred = model_obj.predict(ext_X_t)

                            st.markdown(f"**{name.upper()}:**")
                            if data_config.task_type == "regression":
                                cis = bootstrap_all_regression_metrics(ext_y, ext_pred, n_resamples=500)
                            else:
                                cis = bootstrap_all_classification_metrics(ext_y, ext_pred, n_resamples=500)

                            for metric_name, result in cis.items():
                                st.write(f"  {metric_name}: {format_metric_with_ci(result)}")
                        except Exception as e:
                            st.warning(f"Could not validate {name}: {e}")
        except Exception as e:
            st.error(f"Error loading file: {e}")

# ════════════════════════════════════════════════════════════════
# SUBGROUP ANALYSIS
# ════════════════════════════════════════════════════════════════
st.header("📊 Subgroup Analysis")
render_guidance(
    "<strong>Reviewers often ask:</strong> \"Does your model work equally well for all subgroups?\" "
    "Subgroup analysis reveals performance disparities across demographics or clinical categories.",
    icon="🔍"
)

with st.expander("Run Subgroup Analysis", expanded=False):
    _df_sub = get_data()
    if _df_sub is not None and st.session_state.get('trained_models'):
        from data_processor import get_categorical_columns
        cat_cols = get_categorical_columns(_df_sub)
        subgroup_options = [c for c in cat_cols if c != data_config.target_col and _df_sub[c].nunique() <= 10]

        if subgroup_options:
            subgroup_var = st.selectbox("Stratify by", subgroup_options, key="subgroup_var")

            if st.button("Run Subgroup Analysis", type="primary", key="run_subgroup"):
                from ml.publication import subgroup_analysis, plot_forest_subgroups

                for name, results in st.session_state.model_results.items():
                    y_test_sub = np.array(results["y_test"])
                    y_pred_sub = np.array(results["y_test_pred"])

                    X_test_local = st.session_state.get("X_test")
                    if X_test_local is not None and subgroup_var in X_test_local.columns:
                        subgroup_labels = X_test_local[subgroup_var].values

                        st.subheader(f"{name.upper()}")
                        sub_df = subgroup_analysis(
                            y_test_sub, y_pred_sub, subgroup_labels,
                            task_type=data_config.task_type or "regression",
                            n_bootstrap=200,
                        )
                        st.dataframe(sub_df[["Subgroup", "N", sub_df.columns[2], "95% CI"]], use_container_width=True, hide_index=True)

                        fig = plot_forest_subgroups(sub_df, metric_name=sub_df.columns[2])
                        st.plotly_chart(fig, use_container_width=True, key=f"forest_{name}")
                    else:
                        st.warning(f"Subgroup variable `{subgroup_var}` not found in test data.")
        else:
            st.info("No suitable categorical variables found for subgroup analysis (need ≤10 unique values).")
    else:
        st.info("Train models first to run subgroup analysis.")

# ── State Debug ─────────────────────────────────────────────────
with st.expander("Advanced / State Debug", expanded=False):
    _df = get_data()
    st.write(f"• Data shape: {_df.shape if _df is not None else 'None'}")
    st.write(f"• Target: {data_config.target_col if data_config else 'None'}")
    st.write(f"• Features: {len(data_config.feature_cols) if data_config else 0}")
    st.write(f"• X_test shape: {X_test.shape if X_test is not None else 'None'}")
    st.write(f"• Trained models: {len(st.session_state.get('trained_models', {}))}")
    st.write(f"• Permutation importance: {len(perm_data)}")
    st.write(f"• SHAP results: {len(shap_data)}")
    st.write(f"• PDP results: {len(pdp_data)}")
    _lt = st.session_state.get("last_timings", {})
    if _lt:
        st.write("• Last timings (s):", ", ".join(f"{k}={v}s" for k, v in _lt.items()))
