"""
Page 05: Preprocessing Builder
Build sklearn Pipeline with ColumnTransformer.
Integrates coach recommendations for intelligent preprocessing suggestions.
"""
import streamlit as st
import pandas as pd
import numpy as np
import copy
import time
from typing import List, Dict, Any, Optional

from utils.session_state import (
    init_session_state, get_data, DataConfig, set_preprocessing_pipeline, set_preprocessing_pipelines,
    TaskTypeDetection, log_methodology,
)
from utils.storyline import render_breadcrumb, render_page_navigation
from ml.pipeline import (
    build_preprocessing_pipeline,
    get_pipeline_recipe,
    get_feature_names_after_transform,
    build_unit_harmonization_config,
    build_plausibility_bounds,
    apply_plausibility_filter,
)
from ml.model_registry import get_registry
from data_processor import get_numeric_columns
from utils.theme import inject_custom_css, render_step_indicator, render_guidance, render_sidebar_workflow
from utils.table_export import table

@st.cache_resource
def _get_registry_cached():
    return get_registry()
from utils.widget_helpers import safe_option_index

init_session_state()

st.set_page_config(page_title="Preprocessing", page_icon="⚙️", layout="wide")
inject_custom_css()
render_sidebar_workflow(current_page="05_Preprocess")  # Page ID correct after renumbering
render_step_indicator(5, "Preprocessing")
st.title("⚙️ Preprocess for Modeling")
st.caption("Recommended workflow: make the data model-ready here, then move directly into training and comparison.")
render_breadcrumb("05_Preprocess")
render_page_navigation("05_Preprocess")

st.markdown("""
### Why Preprocessing?

After selecting your features, you need to prepare them for machine learning.

**Where this fits in the product:**
1. ✅ Upload → EDA → Feature Selection
2. **NOW:** prepare the data for training
3. **NEXT:** train models and compare a strong baseline result

**Why this matters:**
- Different models need different preprocessing (tree-based models don't need scaling, linear models do)
- Missing data must be handled before training
- Proper preprocessing prevents "data leakage" (test set contamination)

This step creates **per-model pipelines** that will transform your data correctly during training.
""")

st.info(
    "🔒 **Execution order:** The settings you configure here are **not applied yet.** "
    "They are saved as a pipeline recipe and applied *after* your data is split on the "
    "next page (Train & Compare). Preprocessing is fit on training data only, then "
    "applied to validation and test sets — this prevents data leakage. "
    "You are configuring *what* to do; the split determines *on which data* it happens."
)

# Progress indicator

df = get_data()
if df is None:
    st.warning("Please upload data in the Upload & Audit page first")
    st.stop()
if len(df) == 0 or len(df.columns) == 0:
    st.warning("Your dataset is empty. Please upload data with at least one row and one column.")
    st.stop()

# Guardrail: Preprocessing is only for prediction mode
task_mode = st.session_state.get('task_mode')
if task_mode != 'prediction':
    st.warning("⚠️ **Preprocessing is only available in Prediction mode.**")
    st.info("""
    Please go to the **Upload & Audit** page and select **Prediction** as your task mode.
    Preprocessing pipelines are used to prepare data for machine learning models.
    """)
    st.stop()

data_config: Optional[DataConfig] = st.session_state.get('data_config')
if data_config is None or not data_config.target_col:
    st.warning("Please select target and features in the Upload & Audit page first")
    st.stop()

# Identify feature types
# If feature engineering was applied, use ALL columns except target
# Otherwise use configured feature_cols
target_col = data_config.target_col
if st.session_state.get('feature_engineering_applied'):
    all_features = [col for col in df.columns if col != target_col]
else:
    all_features = data_config.feature_cols if data_config else []

if not all_features:
    st.warning("No features selected. Please select features in the Upload & Audit page first")
    st.stop()

numeric_cols = get_numeric_columns(df)
numeric_features = [f for f in all_features if f in numeric_cols]
categorical_features = [f for f in all_features if f not in numeric_cols]

st.info(f"**Numeric features:** {len(numeric_features)} | **Categorical features:** {len(categorical_features)}")

# ── Double-transformation guardrail ─────────────────────────────
_eng_transform_map = st.session_state.get("engineered_feature_transforms", {})
_log_engineered = [f for f, t in _eng_transform_map.items() if t == "log" and f in numeric_features]
_power_engineered = [f for f, t in _eng_transform_map.items() if t == "power" and f in numeric_features]
_pca_engineered = [f for f, t in _eng_transform_map.items() if t == "pca" and f in numeric_features]

if _log_engineered or _power_engineered or _pca_engineered:
    st.warning(f"""
    ⚠️ **Double-transformation risk detected!**
    
    Some features were already transformed in Feature Engineering:
    {"- **Log-transformed:** " + ", ".join(f"`{f}`" for f in _log_engineered[:5]) + chr(10) if _log_engineered else ""}{"- **Power-transformed:** " + ", ".join(f"`{f}`" for f in _power_engineered[:5]) + chr(10) if _power_engineered else ""}{"- **PCA components:** " + ", ".join(f"`{f}`" for f in _pca_engineered[:5]) + chr(10) if _pca_engineered else ""}
    Applying log/power transforms or PCA again in preprocessing would double-transform these features.
    The preprocessing pipeline below will **auto-exclude** these from redundant transforms.
    """)

# Get profile and EDA results for recommendations
profile = st.session_state.get('dataset_profile')
eda_results = st.session_state.get('eda_results', {})

# EDA-based recommendation cues (for display next to options)
_eda_outliers = bool(profile and profile.features_with_outliers)
_eda_missing = bool(profile and profile.n_features_with_missing > 0)
_eda_high_pn = bool(profile and getattr(profile, 'p_n_ratio', 0) > 0.3)
_eda_collinearity = any('collinearity' in str(k).lower() or 'multicollinearity' in str(k).lower() for k in (eda_results or {}))

# Coaching companion
from utils.coaching_ui import render_page_coaching
render_page_coaching("05_Preprocess")

# ── Model Coach: data-aware recommendations ─────────────────────
_profile = st.session_state.get("dataset_profile")
_coach_picks = []
if _profile:
    try:
        from ml.model_coach import select_top_picks
        _coach_picks, _coach_skips = select_top_picks(_profile)

        if _coach_picks:
            with st.container(border=True):
                st.markdown("#### 🧠 Model Coach")
                for pick in _coach_picks:
                    st.markdown(f"**{pick.role}** · **{pick.model_name}** — {pick.why} · _Prep: {pick.preprocessing}_")
                if _coach_skips:
                    skip_strs = [f"{name} ({reason})" for name, reason in _coach_skips[:4]]
                    st.caption(f"**Skip unless needed:** {'; '.join(skip_strs)}")

            # Auto-select picks in session state
            if not st.session_state.get("_coach_applied"):
                for pick in _coach_picks:
                    st.session_state[f"train_model_{pick.model_key}"] = True
                st.session_state["_coach_applied"] = True

    except Exception as _coach_err:
        import logging
        logging.getLogger(__name__).debug(f"Model coach error: {_coach_err}")

# ============================================================================
# 1. MODEL SELECTION FIRST
# ============================================================================
st.markdown("---")
st.header("Select models for preprocessing")
st.caption("Select models below; these choices drive pipeline options and are used on Train & Compare.")
task_type_det = st.session_state.get("task_type_detection") or TaskTypeDetection()
task_type_final = (getattr(task_type_det, "final", None) or (data_config.task_type if data_config else None) or "regression")
if data_config:
    data_config.task_type = task_type_final

registry_prep = _get_registry_cached()
available_prep = {
    k: v for k, v in registry_prep.items()
    if (task_type_final == "regression" and v.capabilities.supports_regression)
    or (task_type_final == "classification" and v.capabilities.supports_classification)
}
model_groups_prep: Dict[str, List[tuple]] = {}
for key, spec in available_prep.items():
    g = spec.group
    if g not in model_groups_prep:
        model_groups_prep[g] = []
    model_groups_prep[g].append((key, spec))

# Render model selection as styled cards in columns
_GROUP_ICONS = {
    "Linear": "📏", "Tree-based": "🌳", "Distance-based": "📍",
    "Margin-based": "🔲", "Probabilistic": "🎲", "Neural": "🧠",
}
for group_name in sorted(model_groups_prep.keys()):
    icon = _GROUP_ICONS.get(group_name, "📦")
    st.markdown(f"#### {icon} {group_name}")
    models_in_group = model_groups_prep[group_name]
    cols = st.columns(min(len(models_in_group), 3))
    for idx, (model_key, spec) in enumerate(models_in_group):
        ck = f"train_model_{model_key}"
        with cols[idx % len(cols)]:
            is_selected = st.session_state.get(ck, False)
            border_color = "#667eea" if is_selected else "#e2e8f0"
            bg_color = "#f0f0ff" if is_selected else "#ffffff"
            notes_text = "; ".join(spec.capabilities.notes) if spec.capabilities.notes else ""
            check_icon = "✅" if is_selected else ""
            _desc_map = {
                "ridge": "L2-regularized linear model. Good baseline.",
                "lasso": "L1-regularized; performs feature selection.",
                "elasticnet": "Combines L1 + L2 penalties.",
                "logreg": "Standard linear classifier. Interpretable.",
                "glm": "Ordinary least squares or logistic regression.",
                "huber": "Robust to outliers in the target variable.",
                "knn_reg": "Predicts from nearby neighbors. No assumptions.",
                "knn_clf": "Classifies by majority vote of neighbors.",
                "rf": "Ensemble of decorrelated trees. Robust default.",
                "extratrees_reg": "Extremely randomized trees. Fast.",
                "extratrees_clf": "Extremely randomized trees. Fast.",
                "histgb_reg": "Fast gradient boosting with histogram binning.",
                "histgb_clf": "Fast gradient boosting with histogram binning.",
                "svr": "Finds optimal margin hyperplane for regression.",
                "svc": "Finds optimal margin hyperplane for classification.",
                "gaussian_nb": "Assumes feature independence. Very fast.",
                "lda": "Maximizes class separability. Linear boundaries.",
                "nn": "Multi-layer perceptron. Flexible, needs tuning.",
                "xgb_reg": "XGBoost gradient boosting. Industry standard.",
                "xgb_clf": "XGBoost gradient boosting. Industry standard.",
                "lgbm_reg": "LightGBM. Fast leaf-wise gradient boosting.",
                "lgbm_clf": "LightGBM. Fast leaf-wise gradient boosting.",
            }
            desc = _desc_map.get(model_key, notes_text)
            # Card is a toggle button — clicking it selects/deselects the model
            btn_label = f"{'✅ ' if is_selected else ''}{spec.name}"
            if st.button(
                btn_label,
                key=f"btn_{ck}",
                use_container_width=True,
                type="primary" if is_selected else "secondary",
                help=desc,
            ):
                st.session_state[ck] = not is_selected
                st.rerun()
            # Keep hidden checkbox in sync for downstream code
            if ck not in st.session_state:
                st.session_state[ck] = False

selected_models = [k.replace("train_model_", "") for k, v in st.session_state.items() if k.startswith("train_model_") and v]
if selected_models:
    st.success(f"✅ **{len(selected_models)} model(s) selected:** {', '.join(m.upper() for m in selected_models)}")
else:
    st.info("Select at least one model above. Each gets its own preprocessing pipeline.")

# ============================================================================
# 2. PREPROCESSING CONFIGURATION
# ============================================================================
st.markdown("---")
st.header("⚙️ Configure Preprocessing")

preprocessing_config = st.session_state.get("preprocessing_config", {}) or {}
st.session_state.preprocessing_config = preprocessing_config

# Simple vs Advanced mode toggle
config_mode = st.radio(
    "Configuration mode",
    ["🟢 Smart Defaults (recommended)", "🔧 Advanced (full control)"],
    index=0,
    key="preprocess_config_mode",
    horizontal=True,
    help="Smart Defaults auto-configures based on your data and EDA findings. Advanced gives full control over every option.",
)
use_smart_defaults = "Smart" in config_mode

if use_smart_defaults:
    render_guidance(
        "<strong>Smart Defaults</strong> will automatically configure preprocessing based on your data profile: "
        "missing values → median imputation + missing indicators; "
        "outliers detected → robust scaling; "
        "linear models → standard scaling; "
        "tree models → minimal preprocessing. "
        "You can switch to Advanced mode anytime to fine-tune."
    )

    # Auto-detect best settings from EDA
    _auto_scaling = "robust" if _eda_outliers else "standard"
    _auto_imputation = "median"
    _auto_missing_indicators = _eda_missing
    _auto_outlier = "none"  # Let robust scaling handle outliers rather than clipping

    st.markdown("**Auto-detected settings:**")
    auto_cols = st.columns(3)
    with auto_cols[0]:
        st.markdown(f"- Scaling: **{_auto_scaling}**" + (" *(outliers detected)*" if _eda_outliers else ""))
        st.markdown(f"- Imputation: **{_auto_imputation}**")
    with auto_cols[1]:
        st.markdown(f"- Missing indicators: **{'Yes' if _auto_missing_indicators else 'No'}**")
        st.markdown(f"- Categorical: **one-hot encoding**")
    with auto_cols[2]:
        st.markdown(f"- Outlier treatment: **{_auto_outlier}**")
        st.markdown(f"- Feature augmentation: **none**")

    st.caption("These defaults are applied to all selected models. Model-specific adjustments (e.g., enabling scaling for SVM) are handled automatically.")

# Interpretability preference (both modes)
_imode_opts = ["high", "balanced", "performance"]
_imode_stored = st.session_state.get("interpretability_mode", "balanced")
_imode_idx = _imode_opts.index(_imode_stored) if _imode_stored in _imode_opts else 1
interpretability_mode = st.selectbox(
    "Interpretability preference",
    _imode_opts,
    index=_imode_idx,
    key="interpretability_mode",
    format_func=lambda x: {"high": "🔍 High (simple pipelines, no PCA/KMeans)", "balanced": "⚖️ Balanced (recommended)", "performance": "🚀 Performance (all transforms allowed)"}[x],
    help="Controls whether advanced transforms (PCA, KMeans, log) are allowed. High keeps pipelines simple and explainable.",
)

# When using smart defaults, set session state values automatically
if use_smart_defaults:
    # Smart defaults: auto-upgrade scaling for models that need it
    for _mk in (selected_models if selected_models else ["default"]):
        _spec = registry_prep.get(_mk)
        _needs_scale = _spec and _spec.capabilities and getattr(_spec.capabilities, "requires_scaled_numeric", False)
        st.session_state[f"preprocess_{_mk}_numeric_scaling"] = "standard" if _needs_scale else _auto_scaling
        st.session_state[f"preprocess_{_mk}_numeric_imputation"] = _auto_imputation
        st.session_state[f"preprocess_{_mk}_numeric_missing_indicators"] = _auto_missing_indicators
        st.session_state[f"preprocess_{_mk}_numeric_outlier_treatment"] = _auto_outlier
        # Auto-enable power transform for linear models when skewness detected
        _is_linear = _mk in ("ridge", "lasso", "elasticnet", "glm", "logreg", "huber")
        _has_skewed_features = bool(profile and getattr(profile, "highly_skewed_features", []))
        if _has_skewed_features and _is_linear:
            st.session_state[f"preprocess_{_mk}_numeric_power_transform"] = "yeo-johnson"
        else:
            st.session_state[f"preprocess_{_mk}_numeric_power_transform"] = "none"
        st.session_state[f"preprocess_{_mk}_categorical_imputation"] = "most_frequent"
        st.session_state[f"preprocess_{_mk}_categorical_encoding"] = "onehot"
        st.session_state[f"preprocess_{_mk}_numeric_log_transform"] = False
        st.session_state[f"preprocess_{_mk}_use_pca"] = False
        st.session_state[f"preprocess_{_mk}_use_kmeans"] = False
        st.session_state[f"preprocess_{_mk}_plausibility_gating"] = False
        st.session_state[f"preprocess_{_mk}_unit_harmonization"] = False

def _interpretability_guidance(
    profile: Optional[Any],
    insights: List[Dict],
    eda_results: Dict,
    selected: List[str],
    registry: Dict,
) -> List[str]:
    bullets = []
    pn = getattr(profile, "p_n_ratio", 0) if profile else 0
    has_collinearity = any("collinearity" in str(k).lower() or "multicollinearity" in str(k).lower() for k in (eda_results or {}))
    has_outliers = bool(profile and profile.features_with_outliers)
    linear = [m for m in selected if m in ["ridge", "lasso", "elasticnet", "glm", "logreg"]]
    trees = [m for m in selected if m in ["rf", "extratrees_reg", "extratrees_clf", "histgb_reg", "histgb_clf", "xgb_reg", "xgb_clf", "lgbm_reg", "lgbm_clf"]]
    nn_only = selected and all(m == "nn" for m in selected)
    if pn > 0.3 and linear:
        bullets.append(f"High feature-to-sample ratio ({pn:.2f}) and linear models → **performance** can help accuracy; **high** keeps pipelines simple for stakeholders.")
    if has_collinearity and linear:
        bullets.append("Collinearity detected and linear models → **balanced** or **performance**; consider PCA or regularization.")
    if has_outliers and (linear or selected and "nn" in selected):
        bullets.append("Outliers present → **performance** (e.g. robust scaling) or **balanced**; **high** avoids extra transforms.")
    if trees and not linear and not (selected and "nn" in selected):
        bullets.append("Mostly tree models → interpretability preference mainly affects optional preprocessing (log, PCA, KMeans); **balanced** is a reasonable default.")
    if nn_only:
        bullets.append("Neural network only → interpretability affects only preprocessing; use **performance** if you care more about accuracy than explainability.")
    if not bullets:
        bullets.append("**Balanced** is a reasonable default. Use **high** when you need simple, explainable pipelines; **performance** when accuracy matters most.")
    return bullets[:4]

_guidance = _interpretability_guidance(profile, [], eda_results or {}, selected_models, registry_prep)
if _guidance:
    st.caption("**Interpretability guidance:**")
    for _g in _guidance:
        st.caption(f"• {_g}")

_config_keys = ["default"] if not selected_models else selected_models

def _cfg(mk: str, key: str, default: Any, from_global: bool = True) -> Any:
    k = f"preprocess_{mk}_{key}"
    v = st.session_state.get(k)
    if v is not None:
        return v
    if from_global and preprocessing_config:
        return preprocessing_config.get(key, default)
    return default

if use_smart_defaults:
    pass  # Smart defaults already set in session state above; skip manual config
else:
    st.markdown("---")
    st.subheader("Per-Model Configuration")
    st.caption("Expand each model to customize its preprocessing pipeline. Settings apply per-model so you can tailor preprocessing to each algorithm's needs.")

    # Helper: detect high-cardinality categoricals
    _high_card_feats = [f for f in categorical_features if df[f].nunique() > 10] if categorical_features else []

    for _mk in _config_keys:
        with st.expander(f"🔧 Configure {_mk.upper()}", expanded=(len(_config_keys) == 1)):

            # ── 1. 🧹 Handle Missing Data ──
            st.markdown("#### 🧹 Handle Missing Data")
            st.caption("*Why it matters:* Most models cannot handle NaN values. How you fill gaps affects both accuracy and what your results mean — reviewers will scrutinize this.")
            # Missing data warning removed — coaching layer handles this
            _c_miss1, _c_miss2 = st.columns(2)
            with _c_miss1:
                _imp_options = ["median", "mean", "iterative (MICE)", "constant"]
                _imp_help = {
                    "median": "Robust to skewed distributions. Most common default.",
                    "mean": "Assumes symmetry. Sensitive to outliers — use only if features are roughly Gaussian.",
                    "iterative (MICE)": "Gold standard for clinical research. Models each feature conditioned on others. Recommended when >5% data is missing (Rubin, 1987).",
                    "constant": "Fills with a fixed value (e.g., 0). Use when missingness has domain meaning.",
                }
                _stored_imp = _cfg(_mk, "numeric_imputation", "median")
                if _stored_imp == "iterative":
                    _stored_imp = "iterative (MICE)"
                _nim = safe_option_index(_imp_options, _stored_imp, "median")
                _sel_imp = st.selectbox(
                    "Numeric imputation",
                    _imp_options,
                    index=_nim,
                    key=f"preprocess_{_mk}_numeric_imputation_display",
                    help="How to fill missing numeric values before modeling.",
                )
                # Map display value back to internal key
                _imp_internal = "iterative" if "MICE" in _sel_imp else _sel_imp
                st.session_state[f"preprocess_{_mk}_numeric_imputation"] = _imp_internal
                st.caption(f"ℹ️ {_imp_help.get(_sel_imp, '')}")
                st.checkbox(
                    "Add missing-data indicator columns",
                    value=bool(_cfg(_mk, "numeric_missing_indicators", False)),
                    key=f"preprocess_{_mk}_numeric_missing_indicators",
                    help="Adds binary columns (feature_missing = 0/1) so the model can learn whether missingness itself is informative (MNAR pattern).",
                )
            with _c_miss2:
                _cim = safe_option_index(["most_frequent", "constant"], _cfg(_mk, "categorical_imputation", "most_frequent"), "most_frequent")
                st.selectbox(
                    "Categorical imputation",
                    ["most_frequent", "constant"],
                    index=_cim,
                    key=f"preprocess_{_mk}_categorical_imputation",
                    help="'Most frequent' fills with the mode. 'Constant' fills with a placeholder category.",
                )

            # ── 2. 📏 Scale & Transform ──
            st.markdown("#### 📏 Scale & Transform")
            st.caption("*Why it matters:* Linear models, SVMs, and neural nets are sensitive to feature scale. Tree-based models (Random Forest, XGBoost) are scale-invariant — scaling won't hurt but is unnecessary.")
            _scale_options = ["standard", "robust", "minmax", "none"]
            _scale_help = {
                "standard": "Zero-mean, unit-variance (z-score). Best when features are roughly Gaussian. Sensitive to outliers.",
                "robust": "Median/IQR-based. Resistant to outliers — recommended when EDA found outliers.",
                "minmax": "Scales to [0, 1] range. Useful for neural nets. Sensitive to outliers.",
                "none": "No scaling. Fine for tree/ensemble models. Preserves raw coefficient interpretation.",
            }
            _scl = safe_option_index(_scale_options, _cfg(_mk, "numeric_scaling", "standard"), "standard")
            _sel_scale = st.selectbox(
                "Scaling method",
                _scale_options,
                index=_scl,
                key=f"preprocess_{_mk}_numeric_scaling",
                format_func=lambda x: {"standard": "Standard (z-score)", "robust": "Robust (median/IQR)", "minmax": "Min-Max [0,1]", "none": "None (raw values)"}[x],
            )
            st.caption(f"ℹ️ {_scale_help.get(_sel_scale, '')}")
            # Contextual nudge: only show if user actively selected a suboptimal option
            if _eda_outliers and _sel_scale == "standard":
                st.caption("💡 EDA found outliers — Robust scaling may be more appropriate here.")

            _transform_options = ["none", "log1p", "yeo-johnson"]
            _transform_help = {
                "none": "No power transform applied.",
                "log1p": "log(1+x). Compresses right-skewed distributions. Requires non-negative values.",
                "yeo-johnson": "Automatically optimizes the transform parameter. Handles negative values. More general than log — preferred for publication (Box & Cox, 1964; Yeo & Johnson, 2000).",
            }
            _stored_transform = _cfg(_mk, "numeric_power_transform", "none")
            # Backward compat: old log_transform boolean → log1p
            if _stored_transform == "none" and bool(_cfg(_mk, "numeric_log_transform", False)):
                _stored_transform = "log1p"
            _tidx = safe_option_index(_transform_options, _stored_transform, "none")
            _sel_transform = st.selectbox(
                "Power transform",
                _transform_options,
                index=_tidx,
                key=f"preprocess_{_mk}_numeric_power_transform",
                format_func=lambda x: {"none": "None", "log1p": "Log (log(1+x))", "yeo-johnson": "Yeo-Johnson (auto-optimized)"}[x],
                help="Power transforms make skewed features more Gaussian, which helps linear models and neural nets.",
            )
            st.caption(f"ℹ️ {_transform_help.get(_sel_transform, '')}")
            # Map back to old log_transform key for pipeline compatibility
            st.session_state[f"preprocess_{_mk}_numeric_log_transform"] = (_sel_transform == "log1p")

            # ── 3. 🏷️ Encode Categories ──
            if categorical_features:
                st.markdown("#### 🏷️ Encode Categories")
                st.caption("*Why it matters:* Models need numeric inputs. How you encode categories affects feature count, model performance, and interpretability.")
                _enc_options = ["onehot", "target", "ordinal"]
                _enc_help = {
                    "onehot": "Creates a binary column per category. Safe and interpretable. Can cause feature explosion with high-cardinality variables (>10 levels).",
                    "target": "Encodes each category as the smoothed mean of the target variable. Prevents feature explosion. Risk: subtle data leakage if not done in-fold — our pipeline uses cross-fitting to mitigate this.",
                    "ordinal": "Maps categories to integers (0, 1, 2, …). Only appropriate when categories have a natural order (e.g., education level, severity grade). Incorrect use implies false ordering.",
                }
                _stored_enc = _cfg(_mk, "categorical_encoding", "onehot")
                _eidx = safe_option_index(_enc_options, _stored_enc, "onehot")
                _sel_enc = st.selectbox(
                    "Categorical encoding",
                    _enc_options,
                    index=_eidx,
                    key=f"preprocess_{_mk}_categorical_encoding",
                    format_func=lambda x: {"onehot": "One-Hot (binary columns)", "target": "Target Encoding (smoothed means)", "ordinal": "Ordinal (integer mapping)"}[x],
                )
                st.caption(f"ℹ️ {_enc_help.get(_sel_enc, '')}")
                if _high_card_feats and _sel_enc == "onehot":
                    st.caption(f"💡 {', '.join(_high_card_feats[:3])} have {df[_high_card_feats[0]].nunique()}+ levels — Target Encoding avoids sparse columns.")
                if _sel_enc == "ordinal":
                    st.caption("💡 Ordinal encoding assumes a meaningful order. Use One-Hot or Target Encoding for nominal categories.")
            else:
                st.session_state[f"preprocess_{_mk}_categorical_encoding"] = "onehot"

            # ── 4. ✂️ Handle Outliers ──
            st.markdown("#### ✂️ Handle Outliers")
            st.caption("*Why it matters:* Outliers can dominate loss functions (especially MSE) and distort scaling. Tree models are naturally robust; linear/neural models are not.")
            # Outlier warning removed — coaching layer handles this
            _c_out1, _c_out2 = st.columns(2)
            with _c_out1:
                if numeric_features:
                    _out_options = ["none", "percentile", "mad"]
                    _out_help = {
                        "none": "No outlier treatment. Appropriate for tree models or when outliers are real data points.",
                        "percentile": "Clips values outside specified percentiles (e.g., 1st–99th). Simple and effective.",
                        "mad": "Median Absolute Deviation. Flags values beyond k × MAD from the median. More robust than z-score.",
                    }
                    _v = _cfg(_mk, "numeric_outlier_treatment", "none")
                    _idx = safe_option_index(_out_options, _v, "none")
                    _ot = st.selectbox(
                        "Outlier treatment",
                        _out_options,
                        index=_idx,
                        key=f"preprocess_{_mk}_numeric_outlier_treatment",
                        format_func=lambda x: {"none": "None", "percentile": "Percentile clipping", "mad": "MAD-based removal"}[x],
                    )
                    st.caption(f"ℹ️ {_out_help.get(_ot, '')}")
                    if _ot == "percentile":
                        st.number_input("Lower percentile", 0.0, 0.1, 0.01, key=f"preprocess_{_mk}_outlier_lower_q")
                        st.number_input("Upper percentile", 0.9, 1.0, 0.99, key=f"preprocess_{_mk}_outlier_upper_q")
                    elif _ot == "mad":
                        st.number_input("MAD threshold (k)", 2.0, 6.0, 3.5, key=f"preprocess_{_mk}_outlier_mad_threshold", help="Values beyond k × MAD from the median are treated as outliers. 3.5 is a common default.")
                else:
                    _ot = "none"
            with _c_out2:
                _pg = st.checkbox(
                    "Domain-specific range filtering",
                    value=bool(_cfg(_mk, "plausibility_gating", False)),
                    key=f"preprocess_{_mk}_plausibility_gating",
                    help="Apply domain-specific plausible ranges (e.g., NHANES reference ranges for biomarkers). Values outside the range are clipped or filtered.",
                )
                if _pg:
                    _pm = safe_option_index(["clip", "filter"], _cfg(_mk, "plausibility_mode", "clip"), "clip")
                    st.radio(
                        "Range filter mode",
                        ["clip", "filter"],
                        index=_pm,
                        format_func=lambda x: "Clip to NaN (keep rows)" if x == "clip" else "Remove out-of-range rows",
                        key=f"preprocess_{_mk}_plausibility_mode",
                        horizontal=True,
                    )
                st.checkbox(
                    "Unit harmonization",
                    value=bool(_cfg(_mk, "unit_harmonization", False)),
                    key=f"preprocess_{_mk}_unit_harmonization",
                    help="Auto-detect and convert mixed units (e.g., mg/dL ↔ mmol/L) before modeling.",
                )

            # ── 5. 🔬 Advanced: Dimensionality Reduction & Feature Engineering ──
            st.markdown("#### 🔬 Advanced")
            st.caption("⚠️ *These options modify your feature space in ways that affect interpretability. Use only with clear justification — reviewers will ask why.*")

            # PCA — Dimensionality Reduction
            _up = bool(_cfg(_mk, "use_pca", False))
            _up = st.checkbox(
                "PCA — Dimensionality Reduction",
                value=_up,
                key=f"preprocess_{_mk}_use_pca",
                help="Replaces original features with principal components (PC1, PC2, …). Eliminates multicollinearity but DESTROYS feature interpretability.",
            )
            if _up:
                st.warning(
                    "⚠️ **Interpretability impact:** After PCA, you can no longer say 'BMI was the strongest predictor.' "
                    "SHAP values will refer to PC1, PC2, etc. Use only when: (1) severe multicollinearity, "
                    "(2) features >> samples, or (3) black-box model where only prediction accuracy matters."
                )
                if _eda_collinearity:
                    st.info("✅ Collinearity was detected in EDA — PCA may be justified here.")
                if _eda_high_pn:
                    st.info("✅ High feature-to-sample ratio detected — PCA can help reduce dimensionality.")
                _maxc = max(1, min(50, len(numeric_features) + (len(categorical_features) * 5) if categorical_features else len(numeric_features)))
                _pn = _cfg(_mk, "pca_n_components", 10)
                _fix = isinstance(_pn, (int, type(1)))
                _pmode = st.radio("PCA mode", ["Fixed Components", "Variance Threshold"], index=0 if _fix else 1, key=f"preprocess_{_mk}_pca_mode")
                if _pmode == "Fixed Components":
                    _defn = min(int(_pn), _maxc) if isinstance(_pn, (int, float)) else min(10, _maxc)
                    st.number_input("Components", 1, _maxc, _defn, key=f"preprocess_{_mk}_pca_n_components")
                else:
                    _pv = 0.95 if not isinstance(_pn, (int, float)) or _pn > 1 else float(_pn)
                    st.slider("Variance explained", 0.5, 0.99, _pv, 0.05, key=f"preprocess_{_mk}_pca_n_components", help="Retain enough components to explain this fraction of total variance.")
                st.checkbox("Whiten", value=bool(_cfg(_mk, "pca_whiten", False)), key=f"preprocess_{_mk}_pca_whiten", help="Decorrelates and normalizes components to unit variance. Useful for downstream algorithms that assume isotropic data.")

            # KMeans — Cluster-based Feature Engineering
            _uk = bool(_cfg(_mk, "use_kmeans_features", False))
            _uk = st.checkbox(
                "🧪 Cluster-based features (experimental)",
                value=_uk,
                key=f"preprocess_{_mk}_use_kmeans",
                help="Adds distance-to-centroid columns that capture nonlinear cluster structure in your data.",
            )
            if _uk:
                st.warning(
                    "⚠️ **Experimental feature.** Adds derived columns (distance to each KMeans centroid). "
                    "Increases feature count. Makes SHAP/coefficient interpretation harder — "
                    "'distance to cluster 3' is not meaningful to domain experts. "
                    "Most useful for neural nets or when you suspect latent subgroups in your data."
                )
                st.number_input("Number of clusters", 2, 20, int(_cfg(_mk, "kmeans_n_clusters", 5)), key=f"preprocess_{_mk}_kmeans_n_clusters")
                st.checkbox("Add distance features", value=bool(_cfg(_mk, "kmeans_add_distances", True)), key=f"preprocess_{_mk}_kmeans_distances")
                st.checkbox("Add one-hot cluster labels", value=bool(_cfg(_mk, "kmeans_add_onehot", False)), key=f"preprocess_{_mk}_kmeans_onehot")

# Pipeline summary before building
st.markdown("---")
if use_smart_defaults and selected_models:
    st.markdown("**📋 Pipeline Summary** — what will be built:")
    _summary_cols = st.columns(min(len(selected_models), 4))
    for _si, _sm in enumerate(selected_models):
        with _summary_cols[_si % len(_summary_cols)]:
            _spec = registry_prep.get(_sm)
            _needs_scale = _spec and _spec.capabilities and getattr(_spec.capabilities, "requires_scaled_numeric", False)
            _effective_scaling = "standard" if _needs_scale else _auto_scaling
            st.markdown(f"""
            <div style="border: 1px solid #e2e8f0; border-radius: 8px; padding: 0.6rem; margin-bottom: 0.4rem; font-size: 0.85rem;">
                <strong>{_sm.upper()}</strong><br/>
                Scale: {_effective_scaling} · Impute: {_auto_imputation}<br/>
                {"Missing indicators ✓" if _auto_missing_indicators else ""}
                {"· Scaling auto-upgraded" if _needs_scale and _auto_scaling != "standard" else ""}
            </div>
            """, unsafe_allow_html=True)

if st.button("🔨 Build Pipelines", type="primary", key="preprocess_build_button"):
    try:
        t0 = time.perf_counter()
        with st.spinner("Building pipelines..."):
            _sel = [k.replace("train_model_", "") for k, v in st.session_state.items() if k.startswith("train_model_") and v]
            registry = _get_registry_cached()
            model_keys = _sel if _sel else ["default"]

            def _get(mk: str, key: str, default: Any) -> Any:
                return st.session_state.get(f"preprocess_{mk}_{key}", default)

            any_unit = any(_get(mk, "unit_harmonization", False) for mk in model_keys)
            unit_overrides = st.session_state.get("unit_overrides", {})
            unit_config = build_unit_harmonization_config(df, numeric_features, unit_overrides) if any_unit else None
            any_plaus = any_unit and any(_get(mk, "plausibility_gating", False) for mk in model_keys)
            plausibility_bounds = build_plausibility_bounds(numeric_features, unit_config["conversion_factors"]) if (unit_config and any_plaus) else None

            def apply_interpretability_overrides(c: Dict[str, Any], imode: str) -> List[str]:
                notes = []
                if imode != "high":
                    return notes
                if c.get("numeric_log_transform") or c.get("numeric_power_transform", "none") != "none":
                    c["numeric_log_transform"] = False
                    c["numeric_power_transform"] = "none"
                    notes.append("Disabled power transform to preserve interpretability.")
                if c.get("use_pca"):
                    c["use_pca"] = False
                    notes.append("Disabled PCA for interpretability.")
                if c.get("use_kmeans_features"):
                    c["use_kmeans_features"] = False
                    notes.append("Disabled KMeans features for interpretability.")
                return notes

            def apply_model_requirements(c: Dict[str, Any], caps: Any) -> List[str]:
                notes = []
                if caps and getattr(caps, "requires_scaled_numeric", False) and c.get("numeric_scaling") == "none":
                    c["numeric_scaling"] = "standard"
                    notes.append("Enabled standard scaling (model requires scaling).")
                return notes

            pipelines_by_model = {}
            configs_by_model = {}
            any_filter = any_plaus and plausibility_bounds and any(
                _get(mk, "plausibility_mode", "clip") == "filter" and _get(mk, "plausibility_gating", False)
                for mk in model_keys
            )
            if any_filter:
                uf_list = unit_config["conversion_factors"] if unit_config else None
                filtered_df = apply_plausibility_filter(
                    df, numeric_features, plausibility_bounds, uf_list
                )
                st.session_state["filtered_data"] = filtered_df
                X_sample = filtered_df[all_features]
            else:
                st.session_state.pop("filtered_data", None)
                X_sample = df[all_features]
            imode = st.session_state.get("interpretability_mode", "balanced")

            for model_key in model_keys:
                ot = _get(model_key, "numeric_outlier_treatment", "none")
                params = {}
                if ot == "percentile":
                    params = {"lower_q": float(_get(model_key, "outlier_lower_q", 0.01)), "upper_q": float(_get(model_key, "outlier_upper_q", 0.99))}
                elif ot == "mad":
                    params = {"threshold": float(_get(model_key, "outlier_mad_threshold", 3.5))}

                use_unit = _get(model_key, "unit_harmonization", False)
                use_plaus = _get(model_key, "plausibility_gating", False)
                pca_mode = _get(model_key, "pca_mode", "Fixed Components")
                pn = _get(model_key, "pca_n_components", 10)
                pca_int = pca_mode == "Fixed Components" and (isinstance(pn, (int, float)) and (pn >= 1 and pn == int(pn)))

                model_config = {
                    "numeric_features": numeric_features,
                    "categorical_features": categorical_features,
                    "numeric_imputation": _get(model_key, "numeric_imputation", "median"),
                    "numeric_scaling": _get(model_key, "numeric_scaling", "standard"),
                    "numeric_log_transform": bool(_get(model_key, "numeric_log_transform", False)),
                    "numeric_power_transform": _get(model_key, "numeric_power_transform", "none"),
                    "numeric_missing_indicators": bool(_get(model_key, "numeric_missing_indicators", False)),
                    "numeric_outlier_treatment": ot,
                    "numeric_outlier_params": params,
                    "categorical_imputation": _get(model_key, "categorical_imputation", "most_frequent"),
                    "categorical_encoding": _get(model_key, "categorical_encoding", "onehot"),
                    "use_kmeans_features": bool(_get(model_key, "use_kmeans", False)),
                    "kmeans_n_clusters": int(_get(model_key, "kmeans_n_clusters", 5)),
                    "kmeans_add_distances": bool(_get(model_key, "kmeans_distances", True)),
                    "kmeans_add_onehot": bool(_get(model_key, "kmeans_onehot", False)),
                    "use_pca": bool(_get(model_key, "use_pca", False)),
                    "pca_n_components": int(pn) if pca_int else (float(pn) if pca_mode == "Variance Threshold" and isinstance(pn, (int, float)) else (0.95 if _get(model_key, "use_pca", False) else None)),
                    "pca_whiten": bool(_get(model_key, "pca_whiten", False)),
                    "unit_harmonization": use_unit,
                    "plausibility_gating": use_plaus,
                    "plausibility_mode": _get(model_key, "plausibility_mode", "clip"),
                    "interpretability_mode": imode,
                }
                if unit_config:
                    model_config["unit_harmonization_config"] = unit_config
                if plausibility_bounds:
                    model_config["plausibility_bounds"] = plausibility_bounds

                override_notes = []
                spec = registry.get(model_key)
                caps = spec.capabilities if spec else None
                override_notes.extend(apply_interpretability_overrides(model_config, imode))
                override_notes.extend(apply_model_requirements(model_config, caps))

                uf = unit_config["conversion_factors"] if unit_config and use_unit else None
                pb = plausibility_bounds if use_plaus and plausibility_bounds else None
                pmode = model_config["plausibility_mode"]

                temp_pipeline = build_preprocessing_pipeline(
                    numeric_features=numeric_features,
                    categorical_features=categorical_features,
                    numeric_imputation=model_config["numeric_imputation"],
                    numeric_scaling=model_config["numeric_scaling"],
                    numeric_log_transform=model_config["numeric_log_transform"],
                    numeric_power_transform=model_config.get("numeric_power_transform", "none"),
                    numeric_missing_indicators=model_config["numeric_missing_indicators"],
                    numeric_outlier_treatment=model_config["numeric_outlier_treatment"],
                    numeric_outlier_params=model_config["numeric_outlier_params"],
                    unit_harmonization_factors=uf,
                    plausibility_bounds=pb,
                    plausibility_mode=pmode,
                    categorical_imputation=model_config["categorical_imputation"],
                    categorical_encoding=model_config["categorical_encoding"],
                    use_kmeans_features=model_config["use_kmeans_features"],
                    kmeans_n_clusters=model_config["kmeans_n_clusters"],
                    kmeans_add_distances=model_config["kmeans_add_distances"],
                    kmeans_add_onehot=model_config["kmeans_add_onehot"],
                    use_pca=False,
                    random_state=st.session_state.get("random_seed", 42),
                )
                temp_pipeline.fit(X_sample)
                X_temp = temp_pipeline.transform(X_sample)
                if hasattr(X_temp, "toarray"):
                    X_temp = X_temp.toarray()
                actual_n = X_temp.shape[1]
                if model_config["use_pca"] and isinstance(model_config["pca_n_components"], int) and model_config["pca_n_components"] > actual_n:
                    model_config["pca_n_components"] = actual_n
                    override_notes.append(f"Adjusted PCA components to {actual_n} (available features).")

                # Guard against double-transforming engineered features
                _safe_log = model_config["numeric_log_transform"]
                _safe_power = model_config.get("numeric_power_transform", "none")
                _safe_pca = model_config["use_pca"]
                if _eng_transform_map:
                    if _safe_log and _log_engineered:
                        override_notes.append(f"Auto-excluded {len(_log_engineered)} already log-transformed features from log transform.")
                    if _safe_power != "none" and _power_engineered:
                        override_notes.append(f"Auto-excluded {len(_power_engineered)} already power-transformed features from power transform.")
                    if _safe_pca and _pca_engineered:
                        override_notes.append(f"Auto-excluded {len(_pca_engineered)} PCA-derived features from PCA reduction.")

                # Split numeric features: exclude already-transformed from redundant transforms
                _exclude_from_log = set(_log_engineered) if _safe_log else set()
                _exclude_from_power = set(_power_engineered) if _safe_power != "none" else set()
                _exclude_from_pca = set(_pca_engineered) if _safe_pca else set()
                _all_excluded = _exclude_from_log | _exclude_from_power | _exclude_from_pca
                numeric_features_safe = [f for f in numeric_features if f not in _all_excluded]
                numeric_features_passthrough = [f for f in numeric_features if f in _all_excluded]

                pipeline = build_preprocessing_pipeline(
                    numeric_features=numeric_features_safe,
                    categorical_features=categorical_features,
                    numeric_imputation=model_config["numeric_imputation"],
                    numeric_scaling=model_config["numeric_scaling"],
                    numeric_log_transform=_safe_log,
                    numeric_power_transform=_safe_power,
                    passthrough_numeric_features=numeric_features_passthrough,
                    numeric_missing_indicators=model_config["numeric_missing_indicators"],
                    numeric_outlier_treatment=model_config["numeric_outlier_treatment"],
                    numeric_outlier_params=model_config["numeric_outlier_params"],
                    unit_harmonization_factors=uf,
                    plausibility_bounds=pb,
                    plausibility_mode=pmode,
                    categorical_imputation=model_config["categorical_imputation"],
                    categorical_encoding=model_config["categorical_encoding"],
                    use_kmeans_features=model_config["use_kmeans_features"],
                    kmeans_n_clusters=model_config["kmeans_n_clusters"],
                    kmeans_add_distances=model_config["kmeans_add_distances"],
                    kmeans_add_onehot=model_config["kmeans_add_onehot"],
                    use_pca=model_config["use_pca"],
                    pca_n_components=model_config["pca_n_components"],
                    pca_whiten=model_config["pca_whiten"],
                    random_state=st.session_state.get("random_seed", 42),
                )
                pipeline.fit(X_sample)
                X_transformed = pipeline.transform(X_sample)
                if hasattr(X_transformed, "toarray"):
                    X_transformed = X_transformed.toarray()
                model_config["n_output_features"] = X_transformed.shape[1]
                model_config["overrides"] = override_notes
                pipelines_by_model[model_key] = pipeline
                configs_by_model[model_key] = model_config

            base_config = {"numeric_features": numeric_features, "categorical_features": categorical_features}
            set_preprocessing_pipelines(pipelines_by_model, configs_by_model, base_config)
            _built = [k for k in pipelines_by_model.keys() if k != "default"]
            st.session_state["preprocess_built_model_keys"] = _built

            # Save preprocessing summary for methods section generation
            _first_cfg = next(iter(configs_by_model.values()), {})
            _imp_method = _first_cfg.get("numeric_imputation", "median")
            _imp_label = {"median": "median imputation", "mean": "mean imputation",
                          "iterative": "multiple imputation by chained equations (MICE)",
                          "constant": "constant value imputation"}.get(_imp_method, _imp_method)
            _scale_method = _first_cfg.get("numeric_scaling", "standard")
            _scale_label = {"standard": "z-score standardization (zero mean, unit variance)",
                            "robust": "robust scaling (median and IQR)",
                            "minmax": "min-max normalization to [0, 1]",
                            "none": "no scaling applied"}.get(_scale_method, _scale_method)
            _enc_method = _first_cfg.get("categorical_encoding", "onehot")
            _enc_label = {"onehot": "one-hot encoding", "target": "target encoding",
                          "ordinal": "ordinal encoding"}.get(_enc_method, _enc_method)
            _outlier = _first_cfg.get("numeric_outlier_treatment", "none")
            _outlier_label = {"none": "no explicit outlier treatment",
                              "percentile": "percentile-based winsorization",
                              "mad": "MAD-based outlier clipping"}.get(_outlier, _outlier)
            _transform = _first_cfg.get("numeric_power_transform", "none")
            if _first_cfg.get("numeric_log_transform"):
                _transform = "log1p"
            _transform_label = {"none": "no additional transformation",
                                "yeo-johnson": "Yeo-Johnson power transformation",
                                "log1p": "log(1+x) transformation"}.get(_transform, _transform)
            st.session_state["preprocessing_summary"] = {
                "missing_data": {"method": _imp_method, "label": _imp_label,
                                 "indicators": _first_cfg.get("numeric_missing_indicators", False)},
                "scaling": {"method": _scale_method, "label": _scale_label},
                "encoding": {"method": _enc_method, "label": _enc_label},
                "outliers": {"method": _outlier, "label": _outlier_label,
                             "params": _first_cfg.get("numeric_outlier_params", {})},
                "transforms": {"method": _transform, "label": _transform_label},
                "n_numeric": len(numeric_features),
                "n_categorical": len(categorical_features),
                "models_configured": list(configs_by_model.keys()),
            }

            # Model-aware preprocessing insights for Train & Compare and Report
            high_card = bool(profile and getattr(profile, "high_cardinality_features", None))
            model_check_bullets = []
            for mk, cfg in configs_by_model.items():
                spec = registry.get(mk)
                caps = spec.capabilities if spec else None
                scaling = cfg.get("numeric_scaling", "standard")
                ov = cfg.get("overrides", [])
                parts = [f"{mk.upper()}:"]
                if caps and getattr(caps, "requires_scaled_numeric", False):
                    if scaling == "none":
                        parts.append("model requires scaling but you used none — consider enabling scaling.")
                    else:
                        parts.append(f"scaling enabled ({scaling}); appropriate for this model.")
                else:
                    if scaling != "none":
                        parts.append(f"scaling {scaling} (optional for tree models).")
                    else:
                        parts.append("no scaling; fine for tree models.")
                if any("interpretability" in str(o).lower() for o in ov):
                    parts.append("Interpretability overrides applied (e.g. PCA/KMeans disabled).")
                if high_card and cfg.get("categorical_encoding") == "onehot":
                    parts.append("High cardinality in EDA; one-hot may inflate feature count — consider alternatives.")
                model_check_bullets.append(" ".join(parts))
            finding = " ".join(model_check_bullets[:5])
            if len(model_check_bullets) > 5:
                finding += " …"
            from utils.insight_ledger import Insight, get_ledger as _get_pp_resolve_ledger
            _pp_resolve_ledger = _get_pp_resolve_ledger()
            _pp_resolve_ledger.upsert(Insight(
                id="preprocess_model_checks",
                source_page="05_Preprocess", category="methodology", severity="info",
                finding=finding,
                implication="Review that preprocessing matches each model; adjust and rebuild if needed.",
                relevant_pages=["06_Train_and_Compare"],
            ))
            # Build structured per-model provenance for the ledger
            _per_model_provenance = {}
            for _mk, _mc in configs_by_model.items():
                _prov = {
                    "imputation": _mc.get("numeric_imputation", "median"),
                    "scaling": _mc.get("numeric_scaling", "standard"),
                    "encoding": _mc.get("categorical_encoding", "onehot"),
                    "outlier_treatment": _mc.get("numeric_outlier_treatment", "none"),
                }
                _op = _mc.get("numeric_outlier_params", {})
                if _op:
                    _prov["outlier_params"] = _op
                _pt = _mc.get("numeric_power_transform", "none")
                if _pt != "none":
                    _prov["power_transform"] = _pt
                if _mc.get("numeric_log_transform"):
                    _prov["log_transform"] = True
                if _mc.get("numeric_missing_indicators"):
                    _prov["missing_indicators"] = True
                _per_model_provenance[_mk] = _prov

            _pp_resolve_ledger.upsert(Insight(
                id="preprocess_summary",
                source_page="05_Preprocess", category="methodology", severity="info",
                finding=f"Pipelines built for {len(pipelines_by_model)} model(s): {', '.join(m.upper() for m in pipelines_by_model.keys())}.",
                implication="Use Train & Compare to train models; preprocessing is applied per model.",
                relevant_pages=["06_Train_and_Compare", "10_Report_Export"],
                resolved=True,
                resolved_by=f"Built {len(pipelines_by_model)} preprocessing pipeline(s)",
                resolved_on_page="05_Preprocess",
                resolution_details={
                    "action_type": "preprocessing",
                    "method": "per_model_pipeline",
                    "models_trained": list(pipelines_by_model.keys()),
                    "per_model_config": _per_model_provenance,
                },
            ))
            # Build model-scoped resolution details
            _imp_by_model = {mk: mc.get("numeric_imputation", "median") for mk, mc in configs_by_model.items()}
            _unique_imps = set(_imp_by_model.values())
            if len(_unique_imps) == 1:
                _imp_scope_msg = f"All models: {_unique_imps.pop()}"
            else:
                _imp_scope_msg = "; ".join(f"{mk.upper()}: {m}" for mk, m in _imp_by_model.items())

            _scale_by_model = {mk: mc.get("numeric_scaling", "standard") for mk, mc in configs_by_model.items()}
            _unique_scales = set(_scale_by_model.values())
            if len(_unique_scales) == 1:
                _scale_scope_msg = f"All models: {_unique_scales.pop()}"
            else:
                _scale_scope_msg = "; ".join(f"{mk.upper()}: {m}" for mk, m in _scale_by_model.items())

            # Resolve EDA insights addressed by building pipelines
            for _resolve_id, _resolve_msg, _resolve_details in [
                ("eda_missing_severe", f"Imputation configured ({_imp_scope_msg})", {
                    "action_type": "imputation", "method": _imp_method,
                    "scope": "all models" if len(configs_by_model) == 1 else f"{len(configs_by_model)} model pipelines",
                    "per_model": _imp_by_model,
                }),
                ("eda_missing_moderate", f"Imputation configured ({_imp_scope_msg})", {
                    "action_type": "imputation", "method": _imp_method,
                    "scope": "all models" if len(configs_by_model) == 1 else f"{len(configs_by_model)} model pipelines",
                    "per_model": _imp_by_model,
                }),
                ("eda_sufficiency_insufficient", "User proceeded with preprocessing despite insufficient data", {
                    "action_type": "acknowledgment", "method": "accepted_risk",
                }),
                ("eda_sufficiency_borderline", "User proceeded with preprocessing despite borderline sufficiency", {
                    "action_type": "acknowledgment", "method": "accepted_risk",
                }),
            ]:
                _ins = _pp_resolve_ledger.get(_resolve_id)
                if _ins and not _ins.resolved:
                    _pp_resolve_ledger.resolve(
                        _resolve_id,
                        resolved_by=_resolve_msg,
                        resolved_on_page="05_Preprocess",
                        resolution_details=_resolve_details,
                    )

            # Resolve skewness insights if transforms were applied
            _transform_by_model = {mk: mc.get("numeric_power_transform", "none") for mk, mc in configs_by_model.items()}
            _models_with_transform = {mk: t for mk, t in _transform_by_model.items() if t != "none"}
            _models_without = [mk for mk, t in _transform_by_model.items() if t == "none"]
            if _models_with_transform:
                for _skew_id in ["eda_skew_individual", "eda_skew_batch", "eda_target_skew"]:
                    _ins = _pp_resolve_ledger.get(_skew_id)
                    if _ins and not _ins.resolved:
                        if _models_without:
                            _msg = (f"Transform applied: {'; '.join(f'{mk.upper()}: {t}' for mk, t in _models_with_transform.items())}. "
                                    f"Raw features retained for: {', '.join(mk.upper() for mk in _models_without)}.")
                        else:
                            _msg = f"Transform applied across all models: {list(_models_with_transform.values())[0]}"
                        _pp_resolve_ledger.resolve(_skew_id, resolved_by=_msg, resolved_on_page="05_Preprocess",
                            resolution_details={"action_type": "power_transform", "per_model": _transform_by_model})

            # Resolve outlier insights if treatment was applied
            _outlier_by_model = {mk: mc.get("numeric_outlier_treatment", "none") for mk, mc in configs_by_model.items()}
            _models_with_outlier = {mk: t for mk, t in _outlier_by_model.items() if t != "none"}
            if _models_with_outlier:
                for _out_id in ["eda_outliers"]:
                    _ins = _pp_resolve_ledger.get(_out_id)
                    if _ins and not _ins.resolved:
                        _models_no_outlier = [mk for mk, t in _outlier_by_model.items() if t == "none"]
                        if _models_no_outlier:
                            _msg = (f"Outlier treatment: {'; '.join(f'{mk.upper()}: {t}' for mk, t in _models_with_outlier.items())}. "
                                    f"No treatment for: {', '.join(mk.upper() for mk in _models_no_outlier)}.")
                        else:
                            _msg = f"Outlier treatment applied across all models: {list(_models_with_outlier.values())[0]}"
                        _pp_resolve_ledger.resolve(_out_id, resolved_by=_msg, resolved_on_page="05_Preprocess",
                            resolution_details={"action_type": "outlier_treatment", "per_model": _outlier_by_model})

        elapsed = time.perf_counter() - t0
        st.session_state.setdefault("last_timings", {})["Build Pipelines"] = round(elapsed, 2)
        
        # Log methodology action
        # Collect per-model outlier params for methodology log
        _outlier_params_by_model = {}
        for mk, mc in configs_by_model.items():
            op = mc.get("numeric_outlier_params", {})
            if op:
                _outlier_params_by_model[mk] = op

        log_methodology(
            step='Preprocessing',
            action="Configured preprocessing pipeline",
            details={
                'imputation': _imp_method,
                'scaling': _scale_method,
                'encoding': _enc_method,
                'outlier_handling': _outlier,
                'numeric_outlier_params': _outlier_params_by_model if _outlier_params_by_model else _first_cfg.get("numeric_outlier_params", {}),
                'transformation': _transform,
                'models_configured': list(configs_by_model.keys())
            }
        )
        try:
            from utils.workflow_provenance import get_provenance
            get_provenance().record_preprocessing(
                configs_by_model=configs_by_model,
                imputation_method=_imp_method,
            )
        except Exception:
            pass  # Provenance recording should never break the workflow

        # Auto-acknowledge unresolved EDA insights at the preprocessing gate.
        # The user has reviewed their data, configured pipelines, and proceeded —
        # any unresolved EDA observations are implicitly accepted.
        try:
            _pp_resolve_ledger.auto_acknowledge_gate(
                gate_name="Proceeded to preprocessing",
                source_pages=["02_EDA"],
            )
        except Exception:
            pass

        st.success("Preprocessing pipelines built successfully. Expand each model below to view recipe and transformed data.")
        
    except Exception as e:
        st.error(f"Error building pipeline: {e}")
        st.exception(e)

# Per-model expanders: recipe, overrides, show table, CSV export
pipelines_by_model = st.session_state.get("preprocessing_pipelines_by_model") or {}
configs_by_model = st.session_state.get("preprocessing_config_by_model") or {}
if pipelines_by_model:
    st.markdown("---")
    st.header("Pipelines by model")
    st.caption("Expand each model to view recipe and overrides. Use «Show transformed table» to preview values, then «Download as CSV» to export.")
    X_sample_preview = df[all_features]
    for model_key, pipeline in pipelines_by_model.items():
        _show = st.session_state.get(f"show_preview_{model_key}", False)
        with st.expander(f"Pipeline for {model_key.upper()}", expanded=(model_key == "default" or _show)):
            cfg = configs_by_model.get(model_key, {})
            recipe = get_pipeline_recipe(pipeline, plausibility_mode=cfg.get("plausibility_mode"))
            st.code(recipe, language=None)
            overrides = cfg.get("overrides", [])
            if overrides:
                st.caption("Overrides applied:")
                for note in overrides:
                    st.write(f"• {note}")
            show_table = st.checkbox("Show transformed table", value=_show, key=f"show_preview_{model_key}")
            if show_table:
                _before = X_sample_preview.head(100)
                X_t = pipeline.transform(X_sample_preview)
                if hasattr(X_t, "toarray"):
                    X_t = X_t.toarray()
                col_names = get_feature_names_after_transform(pipeline, all_features)
                if len(col_names) != X_t.shape[1]:
                    col_names = [f"feature_{i}" for i in range(X_t.shape[1])]
                preview_df = pd.DataFrame(X_t, columns=col_names)
                _ba, _aa = st.columns(2)
                with _ba:
                    st.subheader("Before")
                    table(_before, width="stretch")
                with _aa:
                    st.subheader("After")
                    table(preview_df.head(100), width="stretch")
                csv_bytes = preview_df.to_csv(index=False).encode()
                st.download_button(
                    "Download as CSV",
                    data=csv_bytes,
                    file_name=f"transformed_{model_key}.csv",
                    mime="text/csv",
                    key=f"download_preview_{model_key}",
                )
    _built_names = ", ".join(k.upper() for k in pipelines_by_model.keys())
    st.success(f"✅ **Pipelines ready for: {_built_names}**. Head to **Train & Compare** → your models and preprocessing are already synced.")
    st.page_link("pages/06_Train_and_Compare.py", label="➡️ Go to Train & Compare", icon="🏋️")

    if st.button("Rebuild Pipeline", type="secondary", key="preprocess_rebuild_button", help="Clear current pipelines and reconfigure"):
        st.session_state.preprocessing_pipeline = None
        st.session_state.preprocessing_config = None
        st.session_state.preprocessing_pipelines_by_model = {}
        st.session_state.preprocessing_config_by_model = {}
        st.rerun()

# State Debug (Advanced)
with st.expander("Advanced / State Debug", expanded=False):
    st.markdown("**Current State:**")
    st.write(f"• Data shape: {df.shape if df is not None else 'None'}")
    st.write(f"• Target: {data_config.target_col if data_config else 'None'}")
    st.write(f"• Features: {len(data_config.feature_cols) if data_config else 0}")
    st.write(f"• Preprocessing pipeline: {'Built' if st.session_state.get('preprocessing_pipeline') else 'Not built'}")
    _lt = st.session_state.get("last_timings", {})
    if _lt:
        st.write("• Last timings (s):", ", ".join(f"{k}={v}s" for k, v in _lt.items()))
    preprocessing_config = st.session_state.get('preprocessing_config')
    if preprocessing_config:
        st.write(f"• Numeric imputation: {preprocessing_config.get('numeric_imputation', 'N/A')}")
        st.write(f"• Numeric scaling: {preprocessing_config.get('numeric_scaling', 'N/A')}")
    if profile:
        st.write(f"• Dataset profile available: Yes")
        st.write(f"• Data sufficiency: {profile.data_sufficiency.value}")