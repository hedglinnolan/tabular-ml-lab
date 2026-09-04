"""
Page 07: Model Explainability
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
    init_session_state, get_preprocessing_pipeline, DataConfig, get_data,
    get_split_rows
)
from ml.splits import SplitIdentityError
from utils.storyline import render_breadcrumb, render_page_navigation
from ml.estimator_utils import is_estimator_fitted
from ml.model_registry import get_registry
from ml.pipeline import get_feature_names_after_transform
from ml.regime import (
    kernel_shap_availability, permutation_importance_availability,
    shap_result_guard,
)
from utils.insight_ledger import Insight, get_ledger
from utils.theme import inject_custom_css, render_step_indicator, render_guidance, render_reviewer_concern, render_sidebar_workflow
from utils.table_export import table
from sklearn.pipeline import Pipeline as SklearnPipeline

class _SkipAnalysis(Exception):
    """Raised to skip an analysis block when user deselected it."""
    pass


class _ShapRefused(Exception):
    """Raised when a SHAP estimator was declined rather than attempted.

    Distinct from `_SkipAnalysis` (the user unticked SHAP) because the two need
    different bookkeeping: a refusal has already been shown on screen and
    written to the ledger, and it consumed a step of the progress bar. It is
    also distinct from an exception: nothing failed, and the issues expander
    should not read as if something did.
    """
    pass

@st.cache_resource
def _get_registry_cached():
    return get_registry()

logger = logging.getLogger(__name__)

init_session_state()

st.set_page_config(page_title="Explainability", page_icon="🔬", layout="wide")
inject_custom_css()
render_sidebar_workflow(current_page="07_Explainability")
render_step_indicator(7, "Explain & Validate")

# ── Page Header ─────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom: 1.5rem;">
    <h1 style="margin-bottom: 0.25rem;">🔬 Explain & Validate</h1>
    <p style="color: var(--text-secondary, #475569); font-size: 0.95rem; margin: 0;">
        Recommended workflow: explain the baseline models you just trained, then decide whether any additional validation is actually necessary.
    </p>
</div>
""", unsafe_allow_html=True)

render_breadcrumb("07_Explainability")
from utils.test_lockbox import render_lockbox_status
render_lockbox_status("Explanations on this page are computed on the held-out test set.")
render_page_navigation("07_Explainability")

# ── Coaching companion ──
from utils.coaching_ui import render_page_coaching
render_page_coaching("07_Explainability")

# ── Feature Engineering Reminder ────────────────────────────────
# Check if feature engineering was applied
if st.session_state.get('feature_engineering_applied'):
    engineered_names = st.session_state.get('engineered_feature_names', [])
    engineering_log = st.session_state.get('engineering_log', [])
    
    if engineered_names:
        st.info(f"""
        **💡 Remember:** You created {len(engineered_names)} engineered features in Feature Engineering .
        
        When interpreting feature importance below, some features are transformations of your original data:
        """)
        
        # Show engineering log summary
        if engineering_log:
            st.markdown("**Transformations applied:**")
            for log_entry in engineering_log[:5]:  # Show first 5
                st.markdown(f"- {log_entry}")
            if len(engineering_log) > 5:
                with st.expander("Show all transformations"):
                    for log_entry in engineering_log[5:]:
                        st.markdown(f"- {log_entry}")
        
        st.markdown("""
        **For publication:** When reporting important features, explain transformations.
        
        Example: "The most important predictor was log-transformed glucose (log₁₊ₓ glucose), 
        indicating that the relationship between glucose and outcome is non-linear."
        """)
        
        st.markdown("---")

st.markdown("### 📋 Explainability Checklist")

# Three-tier priority system
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **📊 Essential** (include in paper)
    - [ ] SHAP feature importance
    - [ ] Calibration plot (classification)
    - [ ] Feature importance table
    
    **Why:** Reviewers expect SHAP as gold standard. Calibration proves your probabilities are trustworthy.
    
    **Time:** quoted beside the control when your data's width makes it worth quoting
    """)

with col2:
    st.markdown("""
    **📈 Recommended** (if asked by reviewers)
    - [ ] Permutation importance
    - [ ] Partial dependence plots (PDP)
    
    **Why:** Validates SHAP findings. Shows feature-outcome relationships.
    
    **Time:** quoted beside the control when your data's width makes it worth quoting
    """)

with col3:
    st.markdown("""
    **🔬 Advanced** (optional)
    Consider if reviewers request individual-level explanations:
    - **ICE plots** — individual PDP curves per sample
    - **LIME** — local linear explanations
    - **Interaction plots** — pairwise SHAP interactions
    
    **Why:** Deep dive for specific questions. Overkill for most papers. These are not yet built into this app — use the Python packages directly if needed.
    
    **Time:** Varies
    """)

st.markdown("---")

# ── Which group is this page about? ─────────────────────────────
# ABOVE every gate below, including the task-mode one. A researcher whose
# active branch has no models is exactly the person who needs to switch to the
# one that does, and a picker rendered after `st.stop()` is invisible to them.
# It renders nothing at all until a second branch exists.
from utils.cohort_ui import render_branch_picker
render_branch_picker("07_Explainability")

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
if not data_config:
    st.warning("Please configure your data in Upload & Audit first")
    st.stop()

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
# HELPER: the held-out rows, or a refusal
# ────────────────────────────────────────────────────────────────
_identity_refusals_shown = set()


def _report_split_identity(exc: SplitIdentityError, consequence: str) -> None:
    """Say that the held-out rows cannot be found, once per rerun per message.

    Everything on this page claims to describe the test set. When the rows it
    was drawn on are no longer in the active dataset there is nothing here to
    compute honestly, so the page says so instead of computing something.
    """
    key = (str(exc), consequence)
    if key in _identity_refusals_shown:
        return
    _identity_refusals_shown.add(key)
    st.error(f"**Held-out rows not found.** {exc}\n\n{consequence}")


def _held_out_rows(df_raw, consequence: str):
    """The test rows of `df_raw` by label, or None after reporting a refusal."""
    if not st.session_state.get("test_row_labels"):
        # No split drawn in this session — nothing to contradict. The callers'
        # own fallback is the split's stored X_test, which is the held-out
        # matrix itself, so silence here asserts nothing.
        return None
    try:
        return get_split_rows("test", df=df_raw)
    except SplitIdentityError as exc:
        _report_split_identity(exc, consequence)
        return None


def _get_pipeline_and_data(name):
    """Return (full_pipeline_or_estimator, X_test_for_perm, y_test_for_perm, X_test_raw_or_processed)."""
    estimator = st.session_state.get('fitted_estimators', {}).get(name)
    if estimator is None or not is_estimator_fitted(estimator):
        return None, None, None, None

    if name in st.session_state.get('fitted_preprocessing_pipelines', {}):
        prep_pipeline = st.session_state.fitted_preprocessing_pipelines[name]
        full_pipeline = SklearnPipeline([('preprocess', prep_pipeline), ('model', estimator)])
        feature_names_in = getattr(prep_pipeline, 'feature_names_in_', None)
        expected_cols = list(feature_names_in) if feature_names_in is not None else []

        df_raw = get_data()
        raw_candidate = None
        if df_raw is not None and data_config:
            # By label. The positional read this replaces could not tell "the
            # same rows" from "the same number of rows", and the bare except
            # below hid even the IndexError that a shorter frame raised.
            raw_candidate = _held_out_rows(
                df_raw,
                "Permutation importance and SHAP fall back to the feature "
                "matrix recorded when the split was drawn; raw-column analyses "
                "are unavailable until the split is re-run.")
        if raw_candidate is not None:
            try:
                if expected_cols and all(col in raw_candidate.columns for col in expected_cols):
                    X_raw = raw_candidate.loc[:, expected_cols].copy()
                elif isinstance(X_test, pd.DataFrame) and expected_cols and all(col in X_test.columns for col in expected_cols):
                    X_raw = X_test.loc[:, expected_cols].copy()
                elif isinstance(X_test, pd.DataFrame):
                    X_raw = X_test.copy()
                else:
                    fallback_cols = list(st.session_state.get('selected_features') or data_config.feature_cols or feature_names or [])
                    X_raw = raw_candidate.loc[:, [c for c in fallback_cols if c in raw_candidate.columns]].copy()

                y_raw = raw_candidate[data_config.target_col].values
                # Encode string labels to match what the model was trained on
                label_encoder = st.session_state.get('target_label_encoder')
                if label_encoder is not None and y_raw.dtype == object:
                    y_raw = label_encoder.transform(y_raw)

                if len(X_raw.columns) > 0:
                    return full_pipeline, X_raw, y_raw, X_raw
            except Exception:
                logger.exception(
                    "%s: held-out rows could not be re-read by label; falling "
                    "back to the split's stored test matrix", name)
        # THE PIPELINE STAYS ATTACHED ON EVERY PATH OUT OF THIS BRANCH.
        # `X_test` holds RAW columns — page 06 fits on `pipeline.transform(
        # X_test)` — and the estimator was fitted on that pipeline's OUTPUT.
        # Returning the bare estimator here handed a one-hot-encoded model a
        # frame still containing 'female', so every permutation and SHAP run
        # died on the numeric cast while the page reported success
        # (`DRIVE-065`).
        return full_pipeline, X_test, y_test, X_test
    return estimator, X_test, y_test, X_test


def _to_dense_numpy(arr):
    if hasattr(arr, 'toarray'):
        out = arr.toarray()
    elif isinstance(arr, pd.DataFrame):
        out = np.asarray(arr.values, dtype=float)
    else:
        out = np.asarray(arr, dtype=float)
    return np.ascontiguousarray(out)


# ────────────────────────────────────────────────────────────────
# HELPER: one SHAP record, normalized once  (`STATE-033`)
# ────────────────────────────────────────────────────────────────

def _shap_class_label(class_names, idx: int, n_classes: int) -> str:
    """What a SHAP matrix is *of*, said the way a figure caption must say it."""
    name = str(class_names[idx]) if idx < len(class_names) else f"Class {idx}"
    if n_classes == 2 and idx == 1:
        return f"{name} (positive class)"
    return name


def _normalize_shap_values(sv_raw, class_names=None) -> Dict:
    """Build the SHAP record ONCE, with its class attribution written down.

    The array's shape is the only thing that says whether a value belongs to a
    class, and it used to be flattened twice with nothing recorded: the compute
    path took `sv[:, :, -1]` — the LAST class — and set `class_label = None`,
    and the render path re-sliced whatever it was handed. A multiclass model
    therefore produced a summary plot and a "Mean Absolute SHAP Value (Global
    Importance)" ranking that described one arbitrary class, with no way for the
    figure or the exported report to reveal the substitution.

    So: every per-class matrix is kept, each one is named, and the render path
    is forbidden to reshape. A shape this cannot account for comes back as
    `error` — a refusal the page shows — never as a silently reduced array.
    """
    n_classes = 1
    per_class = None

    if isinstance(sv_raw, list):
        try:
            per_class = [np.asarray(a, dtype=float) for a in sv_raw]
        except Exception as exc:
            return {'error': f"SHAP returned a list this page cannot read ({exc})."}
        if not per_class:
            return {'error': "SHAP returned no values."}
        n_classes = len(per_class)
    else:
        arr = np.asarray(sv_raw)
        if arr.ndim == 3:
            # (n_samples, n_features, n_classes) — the modern shap shape for
            # tree models on a multiclass target.
            n_classes = arr.shape[2]
            per_class = [np.asarray(arr[:, :, k], dtype=float)
                         for k in range(n_classes)]
        elif arr.ndim == 2:
            values = np.asarray(arr, dtype=float)
        elif arr.ndim == 1:
            values = np.asarray(arr, dtype=float).reshape(1, -1)
        else:
            return {'error': f"SHAP returned a {arr.ndim}-dimensional array "
                             f"({arr.shape}); this page can only plot per-sample "
                             f"× per-feature values."}

    if per_class is not None:
        if any(a.ndim != 2 for a in per_class):
            return {'error': "SHAP returned per-class values that are not "
                             "sample × feature matrices."}
        names = [str(c) for c in (class_names or [])]
        if len(names) != n_classes:
            names = [f"Class {i}" for i in range(n_classes)]
        # Binary: index 1 is the positive class, which is the one a clinical
        # reader means. Multiclass: no class is privileged, so the first is
        # shown and the reader picks — with the class named either way.
        default_idx = 1 if n_classes == 2 else 0
        return {
            'shap_values': per_class[default_idx],
            'class_index': default_idx,
            'class_label': _shap_class_label(names, default_idx, n_classes),
            'class_names': names,
            'per_class': per_class,
            'n_classes': n_classes,
            'error': None,
        }

    return {
        'shap_values': values,
        'class_index': None,
        'class_label': None,
        'class_names': [],
        'per_class': None,
        'n_classes': 1,
        'error': None,
    }


def _consensus_top_features(perm_data: Dict, top_n: int = 5) -> List[str]:
    """Features in EVERY model's top `top_n` by permutation importance.

    The comparison chart used to tell the reader to "look for features that
    appear in the top 5 for all models" — a computation handed back to the
    person reading it. `consensus_features` is where this app already decides
    what consensus means; a second notion drawn on a chart would be a third
    answer to the same question.
    """
    from ml.feature_selection import FeatureSelectionResult, consensus_features

    results = []
    for name, info in (perm_data or {}).items():
        names = list(info['feature_names'])
        importances = [float(v) for v in info['importances_mean']]
        order = np.argsort(importances)[::-1][:top_n]
        picks = [names[i] for i in order if i < len(names)]
        results.append(FeatureSelectionResult(
            method=f"{name} permutation importance",
            selected_features=picks,
            all_features=names,
            scores=dict(zip(names, importances)),
            details={"top_n": top_n},
            description=f"Top {top_n} features by permutation importance for {name}.",
        ))
    if not results:
        return []
    return consensus_features(results, min_methods=len(results))


def _shap_class_names_for(model_step, n_classes: int, label_encoder=None) -> List[str]:
    """The model's own class labels, decoded back to what the user typed."""
    _declared = getattr(model_step, 'classes_', None)
    classes = list(_declared) if _declared is not None else []
    if len(classes) != n_classes:
        classes = list(range(n_classes))
    if label_encoder is not None and hasattr(label_encoder, 'inverse_transform'):
        try:
            classes = list(label_encoder.inverse_transform(np.asarray(classes)))
        except Exception:
            pass
    return [str(c) for c in classes]


def _record_kernel_shap_skip(name: str, avail: Dict, errors: List[str]) -> None:
    """Show and RECORD that the kernel estimator was declined for one model.

    Both halves are obligatory. On screen, because a user who asked for SHAP is
    owed the reason it did not appear and the alternative that does work. In the
    ledger, because this page's output is a manuscript: a refused analysis that
    reaches no record leaves the Methods section silent about a diagnostic the
    workflow offers, which is the silent-omission failure the caps exist to
    prevent. The entry is left UNRESOLVED, which is what routes it through
    `InsightLedger.discussion_points_for_manuscript()` into the Discussion
    limitations.
    """
    p = int(avail.get('n_features') or 0)
    if avail.get('policy') == 'confirm':
        # The helper's own sentence for this band is a PRICE ("about 8 minutes
        # and 1.6 GB per model"), written for the control above. At the point
        # of skipping, the user needs the outcome and the way back in, so the
        # price is restated inside a sentence that says what happened.
        msg = (
            f"SHAP was not computed for {name.upper()}: KernelExplainer at "
            f"{p:,} features costs about "
            f"{float(avail.get('estimated_minutes_per_model') or 0):,.0f} "
            f"minutes and {float(avail.get('estimated_gb_per_model') or 0):,.1f} "
            f"GB for this model, so it is not started without confirmation. "
            f"Tick 'Run KernelExplainer anyway' above to run it. Tree-based and "
            f"linear models were explained normally, and permutation importance "
            f"remains available for this one."
        )
    else:
        msg = avail.get('reason') or (
            f"SHAP was not computed for {name.upper()}: the model-agnostic "
            f"kernel estimator is not affordable at {p:,} features. "
            f"TreeExplainer models were explained normally, and permutation "
            f"importance remains available for this one."
        )
    st.warning(f"🐢 {msg}")
    errors.append(f"{name} SHAP: {msg}")
    try:
        get_ledger().upsert(Insight(
            id=f"xai_kernel_shap_skipped_{name}",
            source_page="07_Explainability",
            category="explainability",
            severity="info",
            finding=msg,
            implication=(
                f"No SHAP attribution exists for {name.upper()}. Any statement "
                f"about which features drive this model's predictions must come "
                f"from permutation importance instead."
            ),
            recommended_action=(
                "Use permutation importance for this model, compare against a "
                "tree-based model that has an exact fast explainer, or reduce "
                "the feature space on the Feature Selection page."
            ),
            # The Discussion limitation. A deferral and a refusal are different
            # facts about the study and must not be written the same way: one
            # says the analysis was priced and declined, the other that it was
            # not available at this width.
            manuscript_text=(
                (f"SHAP values were not computed for the {name.upper()} model; "
                 f"the model-agnostic kernel estimator was not run at "
                 f"{p:,} features")
                if avail.get('policy') == 'confirm' else
                (f"SHAP values were not computed for the {name.upper()} model; "
                 f"the model-agnostic kernel estimator was not affordable at "
                 f"{p:,} features")
            ),
            relevant_pages=["10_Report_Export"],
            metadata={
                'model': name,
                'n_features': p,
                'policy': avail.get('policy'),
                'refuse_above': avail.get('refuse_above'),
                'estimated_gb_per_model': round(
                    float(avail.get('estimated_gb_per_model') or 0.0), 2),
                'estimated_minutes_per_model': round(
                    float(avail.get('estimated_minutes_per_model') or 0.0), 1),
            },
        ))
    except Exception:
        logger.exception("Could not record the KernelExplainer skip for %s", name)


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

# ── Analysis Selection ──────────────────────────────────────────
_explain_tabs = st.tabs(["🎯 Feature Importance", "📉 Bland–Altman", "🔗 External Validation", "📊 Subgroup Analysis"])

with _explain_tabs[0]:
    st.markdown("### Select Analyses to Run")

    def _post_transform_width(models) -> int:
        """The widest post-preprocessing feature space among `models`.

        `feature_names_by_model` is written at train time from
        `get_feature_names_after_transform`, so it is the space the estimator
        actually sees — 20 PCA components, not 3,000 raw columns — which is the
        axis every cost model in `ml.regime` is calibrated against. The fallback
        is the RAW column count, which OVER-states p whenever preprocessing
        reduces. Over-stating is the safe direction here: it can make the app
        quote a higher price than it will pay, never make it skip work it
        should have done, because every threshold below is re-checked against
        the real width once the pipeline has been applied.
        """
        by_model = st.session_state.get('feature_names_by_model') or {}
        return max(
            (len(by_model.get(m) or []) for m in models),
            default=0,
        ) or len(st.session_state.get('feature_names') or [])

    # ── Permutation importance: default, not availability ───────────────
    # Measured with this page's own defaults (n_repeats=10, n_jobs=-1, 200 test
    # rows) on RandomForest: 26.8 s at p=200, 101.2 s at p=1,000, 200.3 s at
    # p=2,000 — PER MODEL, and this page runs every selected model. Above
    # PERM_IMPORTANCE_DEFAULT_ON_MAX_FEATURES it therefore stops being
    # something the app starts on the user's behalf. It is deliberately never a
    # refusal: this is the only model-agnostic importance the page offers, it
    # does complete, and Cancel works mid-run.
    _perm_width = _post_transform_width(trained)
    _perm_avail = permutation_importance_availability(
        _perm_width, n_models=len(trained), n_repeats=perm_repeats)
    # Streamlit honors `value=` only on the FIRST render of a session; from
    # then on `session_state["run_perm"]` wins. A width-dependent default has to
    # be seeded into session state, and seeded only when the width CLASS
    # changes — re-seeding every rerun would untick a box the user had just
    # deliberately ticked.
    if ('run_perm' not in st.session_state
            or st.session_state.get('_run_perm_width_class') != _perm_avail['default_on']):
        st.session_state['_run_perm_width_class'] = _perm_avail['default_on']
        st.session_state['run_perm'] = _perm_avail['default_on']

    sel_col1, sel_col2, sel_col3 = st.columns(3)
    with sel_col1:
        # No `value=` here: the seeding above always leaves session state
        # holding the intended default, and passing both makes Streamlit log a
        # conflict warning on every render.
        run_perm = st.checkbox("📊 Permutation Importance", key="run_perm",
                               help="Essential — which features matter most to each model")
        if _perm_avail['reason']:
            # The price, beside the control it applies to, so the choice to
            # spend it is made with the number in view.
            st.caption(f"⏱️ {_perm_avail['reason']}")
    with sel_col2:
        run_shap = st.checkbox("🎯 SHAP Values", value=True, key="run_shap",
                               help="Essential — how each feature pushes predictions up or down")
    with sel_col3:
        run_pdp = st.checkbox("📈 Partial Dependence", value=False, key="run_pdp",
                              help="Recommended — marginal effect of each feature on predictions")

    # Model selection
    st.markdown("**Models to analyze:**")
    model_selection = st.multiselect(
        "Select models",
        options=trained,
        default=trained,
        key="explain_model_selection",
        format_func=lambda x: x.upper(),
        label_visibility="collapsed",
    )

    if not any([run_perm, run_shap, run_pdp]):
        st.info("Select at least one analysis above.")

    # Wide-feature advisory: permutation importance evaluates the model
    # n_features × n_repeats times. Measured ~141s at 3000 features × 5
    # repeats — warn up front instead of letting the run look hung.
    if run_perm and model_selection:
        _max_model_feats = _post_transform_width(model_selection)
        if _max_model_feats > 500:
            st.info(
                f"⏱️ Up to **{_max_model_feats:,} features** reach your models. "
                f"Permutation importance costs features × repeats model "
                f"evaluations — expect **several minutes per model** at "
                f"{perm_repeats} repeats. To speed this up: reduce features on "
                f"the Feature Selection page, enable PCA in Preprocessing, or "
                f"lower the repeats slider. The Cancel button works mid-run."
            )

    # ── KernelExplainer: quote the price before spending it ─────────────
    # The model-agnostic estimator tiles the background to
    # (2p + 2048) · n_bg · p float64 cells and re-allocates that for EVERY
    # explained row, so it is quadratic in p where the tree and linear paths
    # are not. Measured on this page's own hard-coded 50 background / 50
    # explained rows: 4.82 s per explained row at p=200, 20.54 s and 2.60 GB at
    # p=800 — roughly 17 minutes for one model. Between
    # KERNEL_SHAP_CONFIRM_FEATURES and KERNEL_SHAP_MAX_FEATURES it does still
    # finish, so it is offered with a price attached rather than refused; it is
    # simply no longer started on the user's behalf. Above
    # KERNEL_SHAP_MAX_FEATURES the refusal itself is raised inside the run
    # loop, where the real post-preprocessing width is known and the projected
    # GB in the message is a number about this dataset rather than an estimate.
    run_kernel_confirmed = False
    _kernel_quotes = []
    if run_shap and model_selection:
        _observed_widths = st.session_state.get('kernel_shap_observed_widths') or {}
        for _km in model_selection:
            _kspec = registry.get(_km)
            if not _kspec or _kspec.capabilities.supports_shap != 'kernel':
                continue
            # A width MEASURED by a previous run wins outright; the estimate is
            # only what to say before the pipeline has ever been applied.
            #
            # This was `max(estimate, observed)`, which self-heals in one
            # direction only. `_post_transform_width` falls back to the RAW
            # column count whenever `feature_names_by_model` has no entry, so a
            # 60,000-column upload whose pipeline emits 300 components quoted
            # `refuse`, rendered no confirmation checkbox, then met the real 300
            # in the run loop, read `confirm`, found nothing ticked and recorded
            # a skip. Taking the max threw the measured 300 away, so the control
            # that would have released a five-minute job never appeared and the
            # ledger filed "was not affordable" against it in perpetuity.
            _observed_kw = int(_observed_widths.get(_km, 0))
            _kw = _observed_kw if _observed_kw > 0 else _post_transform_width([_km])
            _kq = kernel_shap_availability(_kw, n_models=1, model_label=_km.upper())
            if _kq['policy'] == 'confirm':
                _kernel_quotes.append((_km, _kq))
    if _kernel_quotes:
        for _km, _kq in _kernel_quotes:
            st.info(f"🐢 **{_km.upper()}** — {_kq['reason']}")
        run_kernel_confirmed = st.checkbox(
            "Run KernelExplainer anyway for the model(s) above",
            value=False, key="run_kernel_shap_confirm",
            help=("Leaving this unticked skips SHAP for those models only — "
                  "tree and linear models are explained either way, and "
                  "permutation importance is unaffected. The skip is recorded "
                  "so the manuscript cannot claim SHAP for a model that did "
                  "not get it."),
        )

    # Initialize cancel flag
    if 'cancel_explainability' not in st.session_state:
        st.session_state.cancel_explainability = False

    run_col, cancel_col = st.columns([3, 1])
    with run_col:
        run_button = st.button("🚀 Run Selected Analyses", type="primary", width="stretch",
                               disabled=not any([run_perm, run_shap, run_pdp]) or not model_selection)
    with cancel_col:
        if st.button("🛑 Cancel", type="secondary", key="cancel_explain_init"):
            st.session_state.cancel_explainability = True

    if run_button:
        st.session_state.cancel_explainability = False  # Reset flag
        t0 = time.perf_counter()
        analyses_per_model = sum([run_perm, run_shap, run_pdp])
        total_steps = len(model_selection) * analyses_per_model
        step_count = 0
        overall_progress = st.progress(0)
        overall_status = st.empty()
        cancel_container = st.empty()

        perm_results = {}
        shap_results = {}
        pdp_results = {}
        errors = []
        # Which models the run actually reached. The banner and the provenance
        # record below both describe outcomes per model, and `model_selection`
        # is the request — a cancel mid-run makes the two differ.
        models_attempted = []
    
        # Display cancel button during execution
        with cancel_container:
            if st.button("🛑 Skip Current Model", type="secondary", key="cancel_explain_running"):
                st.session_state.cancel_explainability = True
                st.warning("Skipping current model...")

        for name in model_selection:
            # Check if user canceled
            if st.session_state.cancel_explainability:
                st.warning(f"Analysis canceled. Results saved for completed models.")
                break
        
            step_start = time.perf_counter()
            models_attempted.append(name)
            full_pipe, X_perm, y_perm, X_raw = _get_pipeline_and_data(name)
            if full_pipe is None:
                errors.append(f"{name}: Fitted estimator not found or not fitted. Please retrain.")
                step_count += 3
                overall_progress.progress(min(step_count / total_steps, 1.0))
                continue

            spec = registry.get(name)

            # ── 1. Permutation Importance ───────────────────────────
            if run_perm:
                perm_start = time.perf_counter()
                overall_status.text(f"Permutation importance: {name.upper()}...")
                try:
                    # Transform data through preprocessing pipeline first, then
                    # compute PI on the transformed features using just the estimator.
                    # This ensures PI permutes the features the model actually sees
                    # (e.g., 20 PCA components) instead of the original feature space
                    # (e.g., 3000+ raw features), which would stall and produce
                    # misleading results.  Mirrors the SHAP approach below.
                    if isinstance(full_pipe, SklearnPipeline) and 'preprocess' in getattr(full_pipe, 'named_steps', {}):
                        prep = full_pipe.named_steps['preprocess']
                        X_perm_transformed = _to_dense_numpy(prep.transform(X_perm))
                        pi_estimator = full_pipe.named_steps['model']
                    else:
                        X_perm_transformed = _to_dense_numpy(X_perm)
                        pi_estimator = full_pipe

                    pi = permutation_importance(pi_estimator, X_perm_transformed, y_perm,
                                                n_repeats=perm_repeats,
                                                random_state=42, n_jobs=-1)
                    fn_by_model = st.session_state.get('feature_names_by_model', {})
                    n = len(pi.importances_mean)
                    base = list(fn_by_model.get(name) or [])
                    if len(base) != n and name in st.session_state.get('fitted_preprocessing_pipelines', {}):
                        try:
                            base = list(get_feature_names_after_transform(
                                st.session_state.fitted_preprocessing_pipelines[name],
                                list(getattr(X_perm, 'columns', feature_names))
                            ) or [])
                        except Exception:
                            base = list(base)
                    if len(base) != n:
                        fallback_base = list(getattr(X_perm, 'columns', feature_names))
                        base = fallback_base
                    fnames = (base + [f"feature_{i}" for i in range(len(base), n)])[:n]
                    perm_results[name] = {
                        'importances_mean': pi.importances_mean,
                        'importances_std': pi.importances_std,
                        'feature_names': fnames,
                    }
                    perm_time = time.perf_counter() - perm_start
                    if perm_time > 10:
                        st.caption(f"⏱️ {name.upper()} permutation took {perm_time:.1f}s (slow model)")
                except Exception as e:
                    errors.append(f"{name} permutation: {e}")
                step_count += 1
                overall_progress.progress(min(step_count / total_steps, 1.0))
        
            # Check cancel after slow operation
            if st.session_state.cancel_explainability:
                st.warning(f"Canceled after permutation importance for {name.upper()}")
                break

            # ── 2. SHAP ────────────────────────────────────────────
            shap_start = time.perf_counter()
            overall_status.text(f"SHAP values: {name.upper()}...")
            try:
                if not run_shap:
                    raise _SkipAnalysis()
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

                    # ── Result-size guard: rows, never features ─────────
                    # The one thing that does grow without bound here is the
                    # RESULT array, n_eval · p · n_classes float64 cells, which
                    # stays resident in session state for the report. The class
                    # count is only known for certain after the explainer
                    # returns, so it is estimated from the fitted model — the
                    # same `classes_` attribute the class-naming helper reads
                    # below. On anything but an extreme frame this is a no-op:
                    # at the default 200 evaluation rows it does not fire until
                    # p · n_classes exceeds 312,500.
                    _n_classes_budget = 1
                    _declared_classes = getattr(model_step, 'classes_', None)
                    if (data_config and data_config.task_type == 'classification'
                            and _declared_classes is not None):
                        # `classes_` is a numpy array — truth-testing it raises,
                        # so the None check has to be explicit.
                        _n_classes_budget = max(len(_declared_classes), 1)
                    _row_guard = shap_result_guard(
                        len(X_ev), int(X_ev.shape[1]), _n_classes_budget)
                    if _row_guard['reduced']:
                        X_ev = X_ev[:_row_guard['n_rows']]

                    # Choose explainer.
                    #
                    # TreeExplainer is deliberately NOT capped on the feature
                    # axis, and the asymmetry with the kernel branch below is
                    # intentional — please do not "fix" it. Measured:
                    # RandomForest at 200 explained rows took 1.297 / 1.238 /
                    # 1.311 / 1.301 s at p = 500 / 2,000 / 5,000 / 20,000, a
                    # fitted exponent of 0.00 across a 40× range, because the
                    # cost is O(trees · leaves · depth²) per explained row and
                    # depth is set by n_train, not by p. Capping it on feature
                    # count would delete information for free and force a
                    # Methods caveat onto an analysis that needed none.
                    # LinearExplainer is likewise uncapped, but for a weaker
                    # reason: it was never benchmarked, so no threshold could be
                    # defended. Measure it before capping it.
                    if shap_support == 'tree':
                        explainer = shap.TreeExplainer(model_step)
                        shap_values = explainer.shap_values(X_ev)
                    elif shap_support == 'linear':
                        explainer = shap.LinearExplainer(model_step, X_bg)
                        shap_values = explainer.shap_values(X_ev)
                    else:
                        task_type = data_config.task_type if data_config else 'regression'
                        # The real post-preprocessing width. The pre-run quote
                        # above could only use the session-state estimate; this
                        # is the number the projection must be computed from,
                        # and it is remembered so the next render can offer the
                        # confirmation for a width the estimate under-stated.
                        _kernel_p = int(X_bg.shape[1])
                        st.session_state.setdefault(
                            'kernel_shap_observed_widths', {})[name] = _kernel_p
                        _kernel = kernel_shap_availability(
                            _kernel_p, n_models=len(model_selection),
                            model_label=name.upper())
                        if _kernel['policy'] == 'refuse' or (
                                _kernel['policy'] == 'confirm'
                                and not run_kernel_confirmed):
                            _record_kernel_shap_skip(name, _kernel, errors)
                            raise _ShapRefused()
                        bg_small = X_bg[:min(50, len(X_bg))]
                        # KernelExplainer is very slow — cap eval samples at 50
                        X_ev_kernel = X_ev[:min(50, len(X_ev))]
                        if task_type == 'classification' and hasattr(model_step, 'predict_proba'):
                            explainer = shap.KernelExplainer(model_step.predict_proba, bg_small)
                        else:
                            explainer = shap.KernelExplainer(model_step.predict, bg_small)
                        overall_status.text(f"SHAP (KernelExplainer): {name.upper()}... this may take a minute")
                        shap_values = explainer.shap_values(X_ev_kernel)
                        X_ev = X_ev_kernel  # use subsampled version downstream

                    # Multiclass / multi-output SHAP: normalized once, here, with
                    # the class attribution written into the record (`STATE-033`).
                    _n_classes_guess = 1
                    if isinstance(shap_values, list):
                        _n_classes_guess = len(shap_values)
                    elif np.asarray(shap_values).ndim == 3:
                        _n_classes_guess = np.asarray(shap_values).shape[2]
                    shap_record = _normalize_shap_values(
                        shap_values,
                        class_names=_shap_class_names_for(
                            model_step, _n_classes_guess,
                            st.session_state.get('target_label_encoder'))
                        if _n_classes_guess > 1 else None,
                    )
                    if shap_record.get('error'):
                        errors.append(f"{name} SHAP: {shap_record['error']}")
                        step_count += 1
                        overall_progress.progress(min(step_count / total_steps, 1.0))
                        continue
                    sv_plot = shap_record['shap_values']

                    # Feature names for SHAP
                    fn_by_model = st.session_state.get('feature_names_by_model', {})
                    fn_for_shap = list(fn_by_model.get(name) or [])
                    n_cols = X_ev.shape[1]
                    if len(fn_for_shap) != n_cols and name in st.session_state.get('fitted_preprocessing_pipelines', {}):
                        try:
                            fn_for_shap = list(get_feature_names_after_transform(
                                st.session_state.fitted_preprocessing_pipelines[name],
                                list(getattr(X_raw, 'columns', feature_names))
                            ) or [])
                        except Exception:
                            fn_for_shap = list(fn_for_shap)
                    if len(fn_for_shap) != n_cols:
                        fallback_names = list(getattr(X_raw, 'columns', feature_names))
                        fn_for_shap = fallback_names
                    fn_shap = (fn_for_shap + [f"Feature {i}" for i in range(len(fn_for_shap), n_cols)])[:n_cols]

                    shap_results[name] = {
                        'shap_values': sv_plot,
                        'X_eval': X_ev,
                        'feature_names': fn_shap,
                        # The normalized record: which class each matrix is of,
                        # and every class kept so the render path can show one
                        # without ever reshaping (`STATE-033`).
                        'class_label': shap_record['class_label'],
                        'class_index': shap_record['class_index'],
                        'class_names': shap_record['class_names'],
                        'per_class': shap_record['per_class'],
                        'n_classes': shap_record['n_classes'],
                        'kernel_capped': shap_support == 'kernel',
                        'n_eval_samples': len(X_ev),
                        # Distinct from `kernel_capped`, which the caption below
                        # reads as "KernelExplainer was used" — this one means
                        # the stored result was trimmed to stay under the cell
                        # budget, on a path where every feature was explained.
                        'eval_rows_capped': bool(_row_guard['reduced']),
                        'eval_rows_reason': _row_guard['reason'] or "",
                    }
                    # A model that got SHAP this time must not still carry a
                    # "was not computed" entry from a run where it was declined.
                    try:
                        get_ledger().remove(f"xai_kernel_shap_skipped_{name}")
                    except Exception:
                        pass
                    if _row_guard['reduced']:
                        st.caption(f"ℹ️ {_row_guard['reason']}")
                        try:
                            get_ledger().upsert(Insight(
                                id=f"xai_shap_eval_rows_reduced_{name}",
                                source_page="07_Explainability",
                                category="explainability", severity="info",
                                finding=_row_guard['reason'],
                                implication=(
                                    "The SHAP summary describes fewer held-out "
                                    "observations than were available; feature "
                                    "attributions are unaffected."
                                ),
                                manuscript_text=(
                                    f"SHAP values were computed for "
                                    f"{_row_guard['n_rows']:,} of the "
                                    f"{_row_guard['n_rows_requested']:,} available "
                                    f"evaluation observations"
                                ),
                                relevant_pages=["10_Report_Export"],
                                metadata={
                                    'model': name,
                                    'n_rows': _row_guard['n_rows'],
                                    'n_rows_requested': _row_guard['n_rows_requested'],
                                    'n_features': _row_guard['n_features'],
                                    'n_classes': _row_guard['n_classes'],
                                },
                            ))
                        except Exception:
                            logger.exception(
                                "Could not record the SHAP row guard for %s", name)
                    shap_time = time.perf_counter() - shap_start
                    if shap_time > 10:
                        st.caption(f"⏱️ {name.upper()} SHAP took {shap_time:.1f}s (slow for this model type)")
            except _SkipAnalysis:
                pass  # User opted out of SHAP — don't increment step_count
            except _ShapRefused:
                # Already shown and already recorded by `_record_kernel_shap_skip`.
                # The step is counted because it was reached and decided, and
                # the loop continues to partial dependence for this model —
                # declining one estimator must not silently drop the rest.
                step_count += 1
                overall_progress.progress(min(step_count / total_steps, 1.0))
            except ImportError:
                errors.append(f"{name} SHAP: shap package not installed")
                step_count += 1
                overall_progress.progress(min(step_count / total_steps, 1.0))
            except Exception as e:
                errors.append(f"{name} SHAP: {e}")
                logger.exception(f"SHAP error for {name}: {e}")
                step_count += 1
                overall_progress.progress(min(step_count / total_steps, 1.0))
            else:
                step_count += 1
                overall_progress.progress(min(step_count / total_steps, 1.0))
        
            # Check cancel after SHAP
            if st.session_state.cancel_explainability:
                st.warning(f"Canceled after SHAP for {name.upper()}")
                break

            # ── 3. Partial Dependence (top 4 features from perm) ───
            overall_status.text(f"Partial dependence: {name.upper()}...")
            try:
                if not run_pdp:
                    raise _SkipAnalysis()
                if not run_perm or name not in perm_results:
                    errors.append(f"{name} PDP: requires Permutation Importance to identify top features. Enable it above.")
                    raise _SkipAnalysis()
                if name in perm_results and spec and spec.capabilities.supports_partial_dependence:
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

                    # Subsample for PDP if dataset is large (>2000 rows)
                    max_pdp_samples = 2000
                    if X_pdp.shape[0] > max_pdp_samples:
                        rng = np.random.RandomState(42)
                        idx = rng.choice(X_pdp.shape[0], max_pdp_samples, replace=False)
                        X_pdp = X_pdp[idx]

                    # Compute PDP for each feature individually (NOT as a multi-way interaction)
                    pd_per_feature = {}
                    for fidx in top_features_idx:
                        pd_per_feature[fidx] = partial_dependence(model_for_pdp, X_pdp, features=[fidx], kind='average')

                    # ── 2D PDP for top feature pairs ──
                    # Subsample more aggressively and use coarser grid for 2D
                    max_2d_samples = min(1000, X_pdp.shape[0])
                    if X_pdp.shape[0] > max_2d_samples:
                        rng_2d = np.random.RandomState(43)
                        idx_2d = rng_2d.choice(X_pdp.shape[0], max_2d_samples, replace=False)
                        X_pdp_2d = X_pdp[idx_2d]
                    else:
                        X_pdp_2d = X_pdp
                    top_pairs = []
                    for ii in range(min(len(top_features_idx), 3)):
                        for jj in range(ii + 1, min(len(top_features_idx), 3)):
                            top_pairs.append((top_features_idx[ii], top_features_idx[jj]))
                    pd_2d = {}
                    for pair in top_pairs:
                        try:
                            pd_2d[pair] = partial_dependence(
                                model_for_pdp, X_pdp_2d,
                                features=[pair], kind='average',
                                grid_resolution=15,
                            )
                        except Exception:
                            pass

                    pdp_results[name] = {
                        'pd_per_feature': pd_per_feature,
                        'pd_2d': pd_2d,
                        'feature_indices': top_features_idx,
                        'feature_names': pi_data['feature_names'],
                    }
            except _SkipAnalysis:
                pass  # User opted out of PDP — don't increment step_count
            except Exception as e:
                errors.append(f"{name} PDP: {e}")
                logger.exception(f"PDP error for {name}: {e}")
                step_count += 1
                overall_progress.progress(min(step_count / total_steps, 1.0))
            else:
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

        # ── What this run actually produced ──────────────────────────
        # Three surfaces used to speak past the results: the errors were
        # folded into a collapsed expander, the banner was an unconditional
        # `st.success`, and the provenance record was written from the
        # REQUESTED analysis list. A run where all six analyses raised
        # therefore printed "✅ Explainability analysis complete" and ticked
        # two TRIPOD items (`DRIVE-065`). All three now read the results.
        _n_attempted = len(models_attempted)
        _requested = []
        if run_perm: _requested.append(("permutation importance", perm_results))
        if run_shap: _requested.append(("SHAP", shap_results))
        if run_pdp: _requested.append(("partial dependence", pdp_results))
        _succeeded = [(label, res) for label, res in _requested if res]
        _failed = [(label, res) for label, res in _requested if not res]
        _n_units_ok = sum(len(res) for _, res in _requested)
        _n_units = _n_attempted * len(_requested)

        if errors:
            # Expanded when nothing came out: the reason is the whole result.
            with st.expander(f"⚠️ {len(errors)} issue(s) during analysis",
                             expanded=not _succeeded):
                for err in errors:
                    st.text(err)

        _ok_phrase = ", ".join(
            f"{label} ({len(res)}/{_n_attempted} models)" for label, res in _succeeded)
        _fail_phrase = ", ".join(
            f"{label} ({_n_attempted - len(res)}/{_n_attempted} models)"
            for label, res in _failed)

        # Log methodology
        analyses_run = []
        if run_perm and perm_results: analyses_run.append("permutation_importance")
        if run_shap and shap_results: analyses_run.append("shap")
        if run_pdp and pdp_results: analyses_run.append("partial_dependence")
        # The models a result exists for, not the models asked about: the
        # ledger entry this writes is what page 10 quotes beside its TRIPOD
        # ticks, and "Ran  on 3 models" described an empty run.
        models_with_results = [
            m for m in models_attempted
            if m in perm_results or m in shap_results or m in pdp_results
        ]
        # ...and which models each individual analysis reached. `models` above
        # is one flat list ORed across all three, so a model whose SHAP was
        # declined but whose permutation importance succeeded still appears in
        # it — and `ml/publication.py` joins that list into the SHAP sentence,
        # producing a Methods claim about work that did not happen. The
        # per-analysis mapping is what that sentence reads when present.
        _models_by_analysis = {
            'permutation_importance': [m for m in models_attempted if m in perm_results],
            'shap': [m for m in models_attempted if m in shap_results],
            'partial_dependence': [m for m in models_attempted if m in pdp_results],
        }
        _models_by_analysis = {k: v for k, v in _models_by_analysis.items()
                               if k in analyses_run}
        # How many held-out rows SHAP actually explained, per model. The Methods
        # sentence used to quote the full test-set size, which has never been
        # what was explained: the evaluation sample is capped at the slider
        # value, and at 50 rows on the kernel path.
        _shap_eval_rows = sorted({int(r.get('n_eval_samples') or 0)
                                  for r in shap_results.values()} - {0})
        # Permutation importance the width default turned off and the user left
        # off. A skip changes what the manuscript may claim, so it is recorded;
        # the entry is removed again as soon as a run produces results, so the
        # ledger can never carry "was not computed" beside a computed table.
        try:
            _led = get_ledger()
            if perm_results:
                _led.remove("xai_perm_importance_not_run")
            elif not run_perm and not _perm_avail['default_on']:
                _led.upsert(Insight(
                    id="xai_perm_importance_not_run",
                    source_page="07_Explainability",
                    category="explainability", severity="info",
                    finding=(
                        f"Permutation importance was left off at "
                        f"{_perm_avail['n_features']:,} features, where it is "
                        f"not enabled by default."
                    ),
                    implication=(
                        "No model-agnostic importance ranking exists for this "
                        "analysis; any feature-importance claim rests on SHAP "
                        "or on model-internal importances alone."
                    ),
                    recommended_action=(
                        "Tick Permutation Importance and re-run if a "
                        "model-agnostic ranking is needed, or reduce the "
                        "feature space first to make it cheaper."
                    ),
                    manuscript_text="permutation feature importance was not computed",
                    relevant_pages=["10_Report_Export"],
                    metadata={'n_features': _perm_avail['n_features'],
                              'limit': _perm_avail['limit']},
                ))
        except Exception:
            logger.exception("Could not record the permutation-importance skip")
        if analyses_run:
            from utils.session_state import log_methodology
            log_methodology(
                step='Explainability',
                action=(f"Ran {', '.join(analyses_run)} on "
                        f"{len(models_with_results)} models"),
                details={'analyses': analyses_run,
                         'models': list(models_with_results),
                         'models_by_analysis': _models_by_analysis,
                         'shap_n_eval_rows': _shap_eval_rows,
                         'models_requested': list(model_selection),
                         'failed_analyses': [label for label, _ in _failed]}
            )
            try:
                from utils.workflow_provenance import get_provenance
                get_provenance().record_explainability(
                    methods=analyses_run,
                    models=list(models_with_results),
                )
            except Exception:
                pass  # Provenance recording should never break the workflow

        if _n_units == 0:
            st.info(
                "No analysis ran — the run was canceled before any model was "
                "reached. Nothing was recorded."
            )
        elif not _succeeded:
            st.error(
                f"❌ Explainability produced no results ({elapsed:.1f}s): "
                f"0 of {_n_units} analyses completed — {_fail_phrase} failed. "
                f"Nothing on this page describes your models, and nothing was "
                f"recorded for the report. Open the issues above for the reason."
            )
        elif _failed or _n_units_ok < _n_units:
            st.warning(
                f"⚠️ Explainability finished with {_n_units_ok} of {_n_units} "
                f"analyses complete ({elapsed:.1f}s). Succeeded: {_ok_phrase}."
                + (f" Failed: {_fail_phrase}." if _fail_phrase else "")
                + " Only the completed analyses were recorded for the report."
            )
        else:
            st.success(
                f"✅ Explainability analysis complete ({elapsed:.1f}s) — "
                f"{_ok_phrase}."
            )

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
                analysis_tabs = st.tabs(["📈 Permutation Importance (Recommended)", "📊 SHAP Values (Essential)", "📈 Partial Dependence (Recommended)"])

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
                        
                            # Add source column to indicate engineered features
                            if st.session_state.get('feature_engineering_applied'):
                                engineered_names = st.session_state.get('engineered_feature_names', [])
                                if engineered_names:
                                    importance_df['Source'] = importance_df['Feature'].map(
                                        lambda x: '🧬 Engineered' if x in engineered_names else '📊 Original'
                                    )

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
                            st.plotly_chart(fig, key=f"perm_chart_{name}")

                            with st.expander("Full rankings table"):
                                # Show appropriate columns based on whether Source was added
                                if 'Source' in importance_df.columns:
                                    table(importance_df[['Feature', 'Importance', 'Std', 'Source']], 
                                         key=f"perm_importance_{name}", hide_index=True)
                                else:
                                    table(importance_df, key=f"perm_importance_{name}", hide_index=True)

                            from ml.plot_narrative import narrative_permutation_importance
                            nar = narrative_permutation_importance(pd_info, model_name=name)
                            if nar:
                                st.markdown(f"**Summary:** {nar}")
                            from utils.llm_ui import build_llm_context, render_interpretation_with_llm_button, gather_session_context
                            from utils.insight_ledger import MODEL_TO_FAMILY
                            stats_summary = "; ".join(f"{r['Feature']}={r['Importance']:.4f}" for _, r in importance_df.head(5).iterrows())
                            _bg_perm = gather_session_context()
                            _bg_perm.pop("feature_names", None); _bg_perm.pop("sample_size", None); _bg_perm.pop("task_type", None)
                            ctx = build_llm_context("permutation_importance", stats_summary, model_name=name,
                                                    model_family=MODEL_TO_FAMILY.get(name, ""),
                                                    existing=nar or "", feature_names=_fn[:n],
                                                    sample_size=X_test.shape[0] if X_test is not None else None,
                                                    task_type=data_config.task_type if data_config else None, **_bg_perm)
                            render_interpretation_with_llm_button(ctx, key=f"llm_perm_{name}", result_session_key=f"llm_result_perm_{name}", plot_type="permutation_importance")
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
                        X_ev = np.asarray(s['X_eval'])
                        fn = s['feature_names']

                        # One SHAP matrix per class, named. The class is CHOSEN
                        # here and never derived from a shape: this tab used to
                        # re-slice whatever it was handed, so a multiclass model
                        # was explained as one unnamed arbitrary class
                        # (`STATE-033`).
                        _per_class = s.get('per_class')
                        _class_names = list(s.get('class_names') or [])
                        _n_classes = int(s.get('n_classes') or 1)
                        if _per_class is not None and _n_classes > 2:
                            _pick = st.selectbox(
                                "SHAP explains one class at a time — which class?",
                                list(range(_n_classes)),
                                index=int(s.get('class_index') or 0),
                                format_func=lambda i, _n=_class_names, _k=_n_classes:
                                    _shap_class_label(_n, i, _k),
                                key=f"shap_class_pick_{name}",
                                help="A multiclass model produces one SHAP value "
                                     "per class per feature. Every figure below "
                                     "describes the class selected here.",
                            )
                            sv = np.asarray(_per_class[_pick])
                            cl = _shap_class_label(_class_names, _pick, _n_classes)
                        elif _per_class is not None:
                            _pick = int(s.get('class_index') or 0)
                            sv = np.asarray(_per_class[_pick])
                            cl = _shap_class_label(_class_names, _pick, _n_classes)
                        else:
                            sv = np.asarray(s['shap_values'])
                            cl = s.get('class_label')

                        if sv.ndim != 2:
                            # Reshaping here is what produced the wrong figure,
                            # so the page halts instead: a matrix of another
                            # shape cannot be plotted as sample × feature
                            # without deciding, silently, what it is of.
                            st.error(
                                f"**SHAP values for {name.upper()} are not a "
                                f"sample × feature matrix** (shape {sv.shape}), "
                                f"so no importance plot is drawn. Re-run the "
                                f"analysis for this model.")
                            st.stop()

                        if s.get('kernel_capped'):
                            _n_eval = s.get('n_eval_samples', len(X_ev))
                            _n_test = len(st.session_state.get('y_test', []))
                            st.caption(
                                f"ℹ️ SHAP values computed on {_n_eval} of {_n_test} test samples "
                                f"(KernelExplainer is computationally expensive). For full coverage, "
                                f"use tree-based or linear models which have fast exact SHAP methods."
                            )
                        elif s.get('eval_rows_capped') and s.get('eval_rows_reason'):
                            # The row guard, restated beside the figure it
                            # produced. Every feature was explained here — only
                            # the number of explained observations was reduced.
                            st.caption(f"ℹ️ {s['eval_rows_reason']}")

                        # Align columns: SHAP values and X_eval must have same n_features
                        n_cols = min(X_ev.shape[1], sv.shape[1])
                        X_ev = X_ev[:, :n_cols]
                        sv = sv[:, :n_cols]
                        fn_plot = fn[:n_cols] if len(fn) >= n_cols else [f"Feature {i}" for i in range(n_cols)]
                        X_plot_df = pd.DataFrame(X_ev, columns=fn_plot)

                        # Summary plot
                        fig_height = max(4, min(8, n_cols * 0.4))
                        fig, ax = plt.subplots(figsize=(10, fig_height))
                        shap.summary_plot(sv, X_plot_df, feature_names=fn_plot, show=False,
                                          plot_size=(10, fig_height))
                        # The title is the figure's only class attribution, and
                        # the figure is exported into the manuscript — so it is
                        # written whenever the values belong to a class, not
                        # only when a label happened to survive.
                        ax.set_title(
                            f"SHAP Values ({cl})" if cl
                            else "SHAP Values", fontsize=11)
                        st.pyplot(fig)

                        # Store figure for export
                        if 'shap_matplotlib_figs' not in st.session_state:
                            st.session_state['shap_matplotlib_figs'] = {}
                        st.session_state['shap_matplotlib_figs'][f"{name}_summary"] = fig
                    
                        plt.close(fig)

                        # Mean absolute SHAP bar chart
                        mean_abs = np.abs(sv).mean(axis=0)
                        # Ensure mean_abs is 1D and aligned with feature names
                        mean_abs = np.asarray(mean_abs).ravel()[:len(fn_plot)]
                        shap_df = pd.DataFrame({
                            'Feature': fn_plot[:len(mean_abs)],
                            'Mean |SHAP|': mean_abs,
                        }).sort_values('Mean |SHAP|', ascending=False)
                    
                        # Add source column to indicate engineered features
                        if st.session_state.get('feature_engineering_applied'):
                            engineered_names = st.session_state.get('engineered_feature_names', [])
                            if engineered_names:
                                shap_df['Source'] = shap_df['Feature'].map(
                                    lambda x: '🧬 Engineered' if x in engineered_names else '📊 Original'
                                )

                        # "Global Importance" was the claim on a one-class
                        # ranking. It is global only when there is one output.
                        _bar_title = ("Mean Absolute SHAP Value "
                                      + (f"({cl})" if cl else "(Global Importance)"))
                        fig2 = px.bar(shap_df.head(10), x='Mean |SHAP|', y='Feature', orientation='h',
                                      title=_bar_title,
                                      color='Mean |SHAP|', color_continuous_scale='Purples')
                        fig2.update_layout(yaxis={'categoryorder': 'total ascending'}, height=350,
                                           showlegend=False, coloraxis_showscale=False,
                                           margin=dict(l=10, r=10, t=40, b=10))
                        st.plotly_chart(fig2, key=f"shap_bar_{name}")
                    
                        # Show full SHAP table with source column
                        with st.expander("Full SHAP rankings table"):
                            if 'Source' in shap_df.columns:
                                table(shap_df[['Feature', 'Mean |SHAP|', 'Source']], 
                                     key=f"shap_importance_{name}", hide_index=True)
                            else:
                                table(shap_df[['Feature', 'Mean |SHAP|']], 
                                     key=f"shap_importance_{name}", hide_index=True)

                        from ml.plot_narrative import narrative_shap
                        from utils.llm_ui import build_llm_context, render_interpretation_with_llm_button, gather_session_context
                        from utils.insight_ledger import MODEL_TO_FAMILY
                        nar = narrative_shap(sv, fn_plot, model_name=name)
                        if nar:
                            st.markdown(
                                f"**Summary{f' ({cl})' if cl else ''}:** {nar}")
                        top_idx = np.argsort(mean_abs)[::-1][:5]
                        stats_summary = "; ".join(f"{fn_plot[i]}={mean_abs[i]:.3f}" for i in top_idx if i < len(fn_plot))
                        _bg_shap = gather_session_context()
                        _bg_shap.pop("feature_names", None); _bg_shap.pop("sample_size", None); _bg_shap.pop("task_type", None)
                        ctx = build_llm_context("SHAP", stats_summary, model_name=name, model_family=MODEL_TO_FAMILY.get(name, ""),
                                                existing=nar or "", feature_names=fn_plot, sample_size=X_ev.shape[0],
                                                task_type=data_config.task_type if data_config else None, **_bg_shap)
                        render_interpretation_with_llm_button(ctx, key=f"llm_shap_{name}", result_session_key=f"llm_result_shap_{name}", plot_type="SHAP")
                    
                        # Enhanced SHAP visualizations
                        st.markdown("---")
                        st.markdown("### 📊 Individual Prediction Explanations")
                    
                        st.info("""
                    **What are waterfall plots?** These show which features had the biggest impact on a single prediction. 
                    For example, if predicting diabetes risk for patient #5, you can see exactly which lab values or demographics 
                    pushed the prediction higher (green) or lower (red). This helps explain individual model decisions.
                    """)
                    
                        # Waterfall plot for individual prediction
                        try:
                            if sv.shape[0] > 0:
                                sample_idx = st.slider(f"Select sample to explain (0-{min(sv.shape[0]-1, 99)})", 
                                                       0, min(sv.shape[0]-1, 99), 0, key=f"waterfall_idx_{name}")
                            
                                fig_waterfall, ax_wf = plt.subplots(figsize=(10, 6))
                                # Ensure we handle both 1D and 2D SHAP values
                                sv_sample = sv[sample_idx] if sv.ndim > 1 else sv
                            
                                # Manually create waterfall plot data
                                shap_vals_sample = pd.DataFrame({
                                    'Feature': fn_plot[:len(sv_sample)],
                                    'SHAP Value': sv_sample
                                }).sort_values('SHAP Value', key=abs, ascending=False).head(10)
                            
                                # Create horizontal bar chart as waterfall approximation
                                colors = ['#FF6B6B' if v < 0 else '#4ECDC4' for v in shap_vals_sample['SHAP Value']]
                                ax_wf.barh(shap_vals_sample['Feature'], shap_vals_sample['SHAP Value'], color=colors)
                                ax_wf.set_xlabel('SHAP Value (impact on prediction)')
                                ax_wf.set_title(
                                    f'Top 10 Features Impacting Sample {sample_idx}'
                                    + (f' ({cl})' if cl else ''))
                                ax_wf.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
                                plt.tight_layout()
                                st.pyplot(fig_waterfall)
                                st.session_state['shap_matplotlib_figs'][f"{name}_waterfall"] = fig_waterfall
                                plt.close(fig_waterfall)
                            
                                st.caption(f"🔴 Negative SHAP values decrease the prediction • 🟢 Positive SHAP values increase the prediction")
                        except Exception as e:
                            st.info(f"Individual explanation not available: {str(e)[:100]}")
                    
                        # Dependence plots for top features
                        st.markdown("---")
                        st.info("""
                    **What are dependence plots?** These show the relationship between a feature's value and its impact on predictions. 
                    For example, you can see if higher glucose levels consistently push diabetes predictions higher (positive relationship), 
                    or if the relationship is more complex (e.g., U-shaped). Each dot is one sample in your data.
                    """)
                    
                        try:
                            # Get indices of top features (up to 3, or fewer if not enough features)
                            n_top_features = min(3, len(mean_abs), X_ev.shape[1])
                            top_feature_indices = np.argsort(mean_abs)[::-1][:n_top_features]
                        
                            for i, top_idx in enumerate(top_feature_indices):
                                if top_idx < len(fn_plot) and top_idx < X_ev.shape[1]:
                                    top_feature = fn_plot[top_idx]
                                
                                    fig_dep, ax_dep = plt.subplots(figsize=(10, 5))
                                    # Create dependence scatter
                                    feature_vals = X_ev[:, top_idx]
                                    shap_vals_feat = sv[:, top_idx] if sv.ndim > 1 else sv
                                
                                    scatter = ax_dep.scatter(feature_vals, shap_vals_feat, 
                                                            alpha=0.5, c=feature_vals, cmap='viridis', s=20)
                                    ax_dep.set_xlabel(f'{top_feature} (feature value)')
                                    ax_dep.set_ylabel(f'SHAP value for {top_feature}')
                                    ax_dep.set_title(f'Feature Dependence #{i+1}: {top_feature}')
                                    ax_dep.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
                                    plt.colorbar(scatter, ax=ax_dep, label='Feature Value')
                                    plt.tight_layout()
                                    st.pyplot(fig_dep)
                                    st.session_state['shap_matplotlib_figs'][f"{name}_dependence_{i}"] = fig_dep
                                    plt.close(fig_dep)
                                
                                    st.caption(f"Shows how **{top_feature}** values influence predictions. Higher on Y-axis = stronger positive impact.")
                        except Exception as e:
                            st.info(f"Dependence plots not available: {str(e)[:100]}")
                        
                    else:
                        st.info("SHAP was not computed for this model. Check the issues log above.")

                # ── Partial Dependence Tab ──────────────────────────
                with analysis_tabs[2]:
                    if name in pdp_data:
                        pd_info = pdp_data[name]
                        pd_per_feature = pd_info.get('pd_per_feature', {})
                        feat_idx = pd_info['feature_indices']
                        feat_names = pd_info['feature_names']

                        cols = st.columns(2)
                        for i, fidx in enumerate(feat_idx):
                            fname = feat_names[fidx] if fidx < len(feat_names) else f"Feature {fidx}"
                            with cols[i % 2]:
                                pf = pd_per_feature.get(fidx, {})
                                grid = pf['grid_values'][0]
                                avg = pf['average'][0].ravel()
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(x=grid, y=avg, mode='lines+markers',
                                                          line=dict(color='#667eea', width=2),
                                                          marker=dict(size=4)))
                                fig.update_layout(title=f"PDP: {fname}", xaxis_title=fname,
                                                  yaxis_title="Partial Dependence",
                                                  height=300, margin=dict(l=10, r=10, t=40, b=10))
                                st.plotly_chart(fig, key=f"pdp_{name}_{fidx}")

                        # ── 2D PDP: Explore Interactions ──────────────
                        pd_2d = pd_info.get('pd_2d', {})
                        if pd_2d:
                            with st.expander("🔍 Explore Interactions (2D Partial Dependence)", expanded=False):
                                st.caption(
                                    "2D partial dependence shows how the **joint effect** of two features "
                                    "influences predictions. Patterns that deviate from simple additive "
                                    "effects indicate feature interactions."
                                )
                                for pair, pf_2d in pd_2d.items():
                                    fidx_a, fidx_b = pair
                                    fname_a = feat_names[fidx_a] if fidx_a < len(feat_names) else f"Feature {fidx_a}"
                                    fname_b = feat_names[fidx_b] if fidx_b < len(feat_names) else f"Feature {fidx_b}"
                                    grid_a = pf_2d['grid_values'][0]
                                    grid_b = pf_2d['grid_values'][1]
                                    avg_2d = pf_2d['average'][0]

                                    # Interaction effect: total 2D range minus sum of 1D ranges
                                    range_2d = float(np.max(avg_2d) - np.min(avg_2d))
                                    range_1d_a = 0.0
                                    range_1d_b = 0.0
                                    if fidx_a in pd_per_feature:
                                        a1d = pd_per_feature[fidx_a]['average'][0].ravel()
                                        range_1d_a = float(np.max(a1d) - np.min(a1d))
                                    if fidx_b in pd_per_feature:
                                        b1d = pd_per_feature[fidx_b]['average'][0].ravel()
                                        range_1d_b = float(np.max(b1d) - np.min(b1d))
                                    interaction_magnitude = max(0.0, range_2d - range_1d_a - range_1d_b)

                                    fig_2d = go.Figure(data=go.Heatmap(
                                        x=grid_a, y=grid_b, z=avg_2d,
                                        colorscale='RdBu_r',
                                        colorbar=dict(title="Partial<br>Dependence"),
                                    ))
                                    fig_2d.update_layout(
                                        title=f"2D PDP: {fname_a} × {fname_b}",
                                        xaxis_title=fname_a,
                                        yaxis_title=fname_b,
                                        height=420,
                                        margin=dict(l=10, r=10, t=40, b=10),
                                    )
                                    st.plotly_chart(fig_2d, width="stretch", key=f"pdp2d_{name}_{fidx_a}_{fidx_b}")

                                    if interaction_magnitude > 1e-6:
                                        st.metric(
                                            label=f"Interaction Effect ({fname_a} × {fname_b})",
                                            value=f"{interaction_magnitude:.4f}",
                                            help="Magnitude of the interaction effect beyond additive 1D effects. "
                                                 "Larger values indicate stronger feature interactions.",
                                        )
                                    else:
                                        st.caption(f"No meaningful interaction detected between {fname_a} and {fname_b} — effects appear additive.")
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

        # The consensus the page used to ask the reader to find by eye.
        _top_n = 5
        consensus = set(_consensus_top_features(perm_data, top_n=_top_n))

        _x_labels = [f"★ {f}" if f in consensus else str(f)
                     for f in comp_df.index[:top_cross]]
        fig = go.Figure()
        for col in comp_df.columns[:-1]:  # skip Mean
            fig.add_trace(go.Bar(name=col, x=_x_labels, y=comp_df[col][:top_cross]))
        fig.update_layout(barmode='group',
                          title=f"Top {top_cross} Features Across Models "
                                f"(★ = in every model's top {_top_n})",
                          height=400, margin=dict(l=10, r=10, t=40, b=10))
        for _i, _f in enumerate(comp_df.index[:top_cross]):
            if _f in consensus:
                fig.add_vrect(x0=_i - 0.5, x1=_i + 0.5, fillcolor="#667eea",
                              opacity=0.10, line_width=0, layer="below")
        st.plotly_chart(fig, key="cross_model_importance")

        # Consensus features — named, not left as an instruction to the reader.
        _consensus_ordered = [str(f) for f in comp_df.index if f in consensus]
        if _consensus_ordered:
            render_guidance(
                f"<strong>Consensus ({len(_consensus_ordered)} of "
                f"{len(comp_df)} features):</strong> "
                f"<strong>{', '.join(_consensus_ordered)}</strong> "
                f"{'is' if len(_consensus_ordered) == 1 else 'are'} in the top "
                f"{_top_n} for all {len(perm_data)} models (★ above). Features "
                f"that rank highly across models are more likely to be "
                f"genuinely important than any single model's ranking.",
                icon="🎯"
            )
        else:
            render_guidance(
                f"<strong>No consensus:</strong> no feature is in the top "
                f"{_top_n} for all {len(perm_data)} models. The models disagree "
                f"about what matters, which is itself worth reporting.",
                icon="🎯"
            )

    # ════════════════════════════════════════════════════════════════
    # BLAND–ALTMAN (regression, 2+ models)
    # ════════════════════════════════════════════════════════════════
with _explain_tabs[1]:
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
                    st.plotly_chart(fig_ba, key=f"bland_altman_{ma}_{mb}")
                    ba_stats = analyze_bland_altman(pa, pb)
                    nar = narrative_bland_altman(ba_stats, label_a=ma, label_b=mb)
                    if nar:
                        st.markdown(f"**Summary:** {nar}")
                    from utils.llm_ui import build_llm_context, render_interpretation_with_llm_button, gather_session_context
                    stats_summary = f"mean_diff={ba_stats.get('mean_diff', 0):.4f}; width_loa={ba_stats.get('width_loa', 0):.4f}; pct_outside={ba_stats.get('pct_outside_loa', 0):.1%}"
                    _bg_ba = gather_session_context()
                    _bg_ba.pop("sample_size", None); _bg_ba.pop("task_type", None)
                    ctx = build_llm_context(
                        "bland_altman", stats_summary, where=f"Bland-Altman ({ma} vs {mb})", existing=nar or "",
                        metrics=ba_stats, sample_size=len(pa), task_type=data_config.task_type if data_config else None, **_bg_ba,
                    )
                    render_interpretation_with_llm_button(ctx, key="llm_bland_altman_btn", result_session_key="llm_result_bland_altman", plot_type="bland_altman")
                else:
                    st.warning("Prediction lengths differ; cannot compare.")
        else:
            st.info("At least two models with test predictions are required.")
    else:
        st.info("Bland–Altman is available for **regression** tasks with at least two trained models.")

    # ════════════════════════════════════════════════════════════════
    # EXTERNAL VALIDATION
    # ════════════════════════════════════════════════════════════════
with _explain_tabs[2]:
    st.header("🔗 External Validation")
    render_guidance(
        "<strong>Why this matters:</strong> Internal validation (train/test split) shows how well your model works "
        "on similar data. External validation — applying the model to a completely separate dataset — "
        "is the gold standard for publication.",
        icon="📋"
    )

    with st.expander("Upload External Validation Dataset", expanded=False):
        ext_file = st.file_uploader("Upload external dataset (CSV/Excel/JSON)",
                                    type=["csv", "xlsx", "xls", "parquet", "tsv", "json", "jsonl", "ndjson"],
                                    key="ext_val_file")

        if ext_file is not None:
            # `IMPORT-213`: this was the one uploader in the app with no front
            # door — a bare load_tabular_data and a missing-column check, on the
            # file whose numbers the banner above calls the gold standard for
            # publication. Every defect that preserves column names (a JSON
            # payload with two record lists, sentinel codes, a duplicate index)
            # went straight into predict(). Same path as page 01 now: layout
            # disclosure, records-key choice, transpose, Import Doctor.
            from data_processor import detect_file_type, inspect_json, load_tabular_data
            from ml.import_doctor import diagnose
            from utils.import_ui import applied_fixes, render_import_doctor

            ext_key = f"extval_{ext_file.name}"
            ext_type = detect_file_type(ext_file.name)
            ext_records_key = ""
            ext_df = None
            _layout_failed = False

            if ext_type in ('json', 'jsonl'):
                ext_file.seek(0)
                _layout = inspect_json(ext_file, lines=(ext_type == 'jsonl'))
                ext_file.seek(0)
                if _layout.error:
                    st.error(_layout.error)
                    _layout_failed = True
                else:
                    if _layout.candidates:
                        _default_idx = (_layout.candidates.index(_layout.chosen_key)
                                        if _layout.chosen_key in _layout.candidates else 0)
                        ext_records_key = st.selectbox(
                            "Which part of this file holds your rows?",
                            _layout.candidates, index=_default_idx,
                            key=f"records_key_{ext_key}",
                            help="This JSON wraps its table inside a key. Pick "
                                 "the one holding the external cohort's records.",
                        )
                    if _layout.note:
                        st.caption(f"ℹ️ {_layout.note}")

            if not _layout_failed:
                ext_transpose = st.checkbox(
                    "Transpose this file (rows ↔ columns)", value=False,
                    key=f"transpose_{ext_key}",
                    help="Use this if the external file has features in rows.")
                # Same crossing, same reason, as `pages/01_Upload_and_Audit.py`:
                # turning a table around refuses rather than silently merging
                # two rows that share a name.
                from turbotab.orientation import OrientationError
                try:
                    ext_file.seek(0)
                    ext_df = load_tabular_data(
                        ext_file, filename=ext_file.name,
                        transpose=ext_transpose,
                        records_key=ext_records_key or None)
                    ext_file.seek(0)
                    ext_df.columns = [str(c) for c in ext_df.columns]
                except OrientationError as exc:
                    # Not "Error loading file:" — the file loaded, and it was
                    # the transpose that refused. Its message already names the
                    # row to fix.
                    st.error(f"{exc} Or untick “Transpose this file” to load "
                             f"it the way round it arrived.")
                    ext_df = None
                except Exception as e:
                    st.error(f"Error loading file: {e}")
                    ext_df = None

            if ext_df is not None:
                st.success(f"Loaded external dataset: {ext_df.shape[0]} rows × {ext_df.shape[1]} columns")

                # Findings are DISPLAYED, with their reversible fixes, and the
                # frame that goes into predict() is the repaired one.
                ext_df = render_import_doctor(ext_df, ext_key)
                ext_blocking = [f for f in diagnose(ext_df) if f.severity == "critical"]
                ext_repairs = list(applied_fixes(ext_key))

                selected_features = st.session_state.get('selected_features') or data_config.feature_cols
                required_cols = list(selected_features) + [data_config.target_col]
                missing_cols = [c for c in required_cols if c not in ext_df.columns]
                if missing_cols:
                    st.error(f"Missing columns in external dataset: {missing_cols}")
                else:
                    ext_override = True
                    if ext_blocking:
                        st.error(
                            f"**{len(ext_blocking)} blocking structural problem"
                            f"{'s' if len(ext_blocking) != 1 else ''} in this "
                            f"file:** " +
                            "; ".join(f.title for f in ext_blocking) +
                            ". External validation is what a reviewer weighs "
                            "most heavily, so it will not run on a file the "
                            "structural review calls broken.")
                        ext_override = st.checkbox(
                            "Validate anyway — I have reviewed these and they "
                            "are not defects in this file",
                            key=f"ext_override_{ext_key}",
                            help="Anything unfixed is recorded with the results "
                                 "and reported in the manuscript.")

                    if st.button("Validate on External Dataset", type="primary",
                                 key="run_ext_val",
                                 disabled=bool(ext_blocking) and not ext_override):
                        from datetime import datetime as _dt
                        from ml.bootstrap import bootstrap_all_regression_metrics, bootstrap_all_classification_metrics, format_metric_with_ci
                        from utils.workflow_provenance import get_provenance

                        ext_y = ext_df[data_config.target_col].values
                        ext_X = ext_df[selected_features]
                        _n_boot = 500
                        ext_per_model = {}

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
                                    cis = bootstrap_all_regression_metrics(ext_y, ext_pred, n_resamples=_n_boot)
                                else:
                                    cis = bootstrap_all_classification_metrics(ext_y, ext_pred, n_resamples=_n_boot)

                                for metric_name, result in cis.items():
                                    st.write(f"  {metric_name}: {format_metric_with_ci(result)}")
                                ext_per_model[name] = {
                                    m: {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                        for k, v in r.to_dict().items()}
                                    for m, r in cis.items()
                                }
                            except Exception as e:
                                st.warning(f"Could not validate {name}: {e}")

                        # The results used to be displayed and dropped: no
                        # session write, no provenance event, so the manuscript's
                        # external-validation section could never populate. They
                        # are kept, and the record is what ml/publication reads.
                        if ext_per_model:
                            st.session_state['external_validation_results'] = {
                                'dataset_name': ext_file.name,
                                'n_rows': int(ext_df.shape[0]),
                                'n_features': len(selected_features),
                                'target_col': data_config.target_col,
                                'features': list(selected_features),
                                'task_type': data_config.task_type,
                                'per_model': ext_per_model,
                                'n_bootstrap': _n_boot,
                                'records_key': ext_records_key,
                                'transposed': bool(ext_transpose),
                                'import_repairs': ext_repairs,
                                'unresolved_findings': [f.title for f in ext_blocking],
                                'timestamp': _dt.now().isoformat(),
                            }
                            get_provenance().record_external_validation(
                                dataset_name=ext_file.name,
                                n_rows=int(ext_df.shape[0]),
                                n_features=len(selected_features),
                                models_validated=list(ext_per_model.keys()),
                                metrics=ext_per_model,
                                n_bootstrap=_n_boot,
                                import_repairs=ext_repairs,
                                structural_findings="; ".join(f.title for f in ext_blocking),
                                records_key=ext_records_key,
                            )
                            st.success(
                                "Recorded. The Methods draft now reports this "
                                "external validation, with these models and "
                                "this cohort size.")
                        else:
                            st.warning(
                                "No model could be scored on this file, so "
                                "nothing was recorded as external validation.")

    _ext_stored = st.session_state.get('external_validation_results')
    if _ext_stored and _ext_stored.get('per_model'):
        st.caption(
            f"✅ External validation on record: **{_ext_stored.get('dataset_name', '')}** "
            f"({_ext_stored.get('n_rows', 0):,} rows), "
            f"{len(_ext_stored['per_model'])} model(s), "
            f"95% CIs from {_ext_stored.get('n_bootstrap', 0):,} bootstrap resamples. "
            f"This is what the manuscript reports.")
        _ext_rows = [
            {"Model": _m.upper(), "Metric": _metric,
             "Estimate": _vals.get("estimate"),
             "95% CI": f"[{_vals.get('ci_lower'):.4f}, {_vals.get('ci_upper'):.4f}]"
                       if _vals.get('ci_lower') is not None else ""}
            for _m, _mets in _ext_stored['per_model'].items()
            for _metric, _vals in _mets.items()
        ]
        if _ext_rows:
            table(pd.DataFrame(_ext_rows), key="ext_val_recorded", hide_index=True)

    # ════════════════════════════════════════════════════════════════
    # SUBGROUP ANALYSIS
    # ════════════════════════════════════════════════════════════════
with _explain_tabs[3]:
    st.header("📊 Subgroup Analysis")
    render_guidance(
        "<strong>Reviewers often ask:</strong> \"Does your model work equally well for all subgroups?\" "
        "Subgroup analysis reveals performance disparities across demographics or clinical categories.",
        icon="🔍"
    )

    with st.expander("Run Subgroup Analysis", expanded=False):
        df_raw = get_data()
        if df_raw is not None and st.session_state.get('trained_models'):
            available_cat_cols = [c for c in df_raw.columns if df_raw[c].dtype in ('object', 'category') or df_raw[c].nunique() <= 20]
            subgroup_options = [c for c in available_cat_cols if c != data_config.target_col]

            if subgroup_options:
                subgroup_var = st.selectbox("Stratify by", subgroup_options, key="subgroup_var")

                if st.button("Run Subgroup Analysis", type="primary", key="run_subgroup"):
                    from ml.publication import subgroup_analysis, plot_forest_subgroups

                    # y_test/y_pred come from the stored run; the stratum each
                    # one belongs to must come from the SAME rows. Reading the
                    # strata positionally paired one person's prediction with
                    # another person's subgroup, and the N and 95% CI per
                    # stratum then described nobody (CONTRACT-001). Resolved
                    # once, before any model is tabulated: if the rows are gone
                    # the whole table is refused rather than drawn wrong.
                    test_rows = _held_out_rows(
                        df_raw,
                        "Subgroup analysis needs the held-out rows themselves "
                        "to stratify by, so it cannot be run on this dataset "
                        "until the split is re-run.")
                    # The rows were named and could not be found: the fallback
                    # below would answer a different question, so nothing runs.
                    refused = test_rows is None and st.session_state.get("test_row_labels")

                    for name, results in st.session_state.model_results.items():
                        if refused:
                            break
                        y_test_sub = np.array(results["y_test"])
                        y_pred_sub = np.array(results["y_test_pred"])

                        if test_rows is not None:
                            subgroup_labels = test_rows[subgroup_var].values
                        else:
                            X_test_local = st.session_state.get("X_test")
                            if X_test_local is not None and subgroup_var in X_test_local.columns:
                                subgroup_labels = X_test_local[subgroup_var].values
                            else:
                                st.warning(f"Subgroup variable `{subgroup_var}` not found in test data.")
                                continue

                        if len(subgroup_labels) != len(y_test_sub):
                            # Three vectors that must describe one person each,
                            # in one order. Unequal lengths mean they do not.
                            st.warning(
                                f"{name}: {len(y_test_sub)} stored predictions "
                                f"but {len(subgroup_labels)} subgroup values — "
                                "the results and the data are from different "
                                "splits. Re-run Prepare Splits and retrain.")
                            continue

                        st.subheader(f"{name.upper()}")
                        sub_df = subgroup_analysis(
                            y_test_sub, y_pred_sub, subgroup_labels,
                            task_type=data_config.task_type or "regression",
                            n_bootstrap=200,
                        )
                        table(sub_df[["Subgroup", "N", sub_df.columns[2], "95% CI"]], key=f"subgroup_{name}", hide_index=True)

                        fig = plot_forest_subgroups(sub_df, metric_name=sub_df.columns[2])
                        st.plotly_chart(fig, key=f"forest_{name}")
            else:
                st.info("No suitable categorical variables found for subgroup analysis (need ≤20 unique values).")
        else:
            st.info("Train models first to run subgroup analysis.")

    # ── State Debug ─────────────────────────────────────────────────
if st.session_state.get("show_debug_panel"):
  with st.expander("Advanced / State Debug", expanded=False):
    _df = get_data()
    st.write(f"• Data shape: {_df.shape if _df is not None else 'None'}")
    st.write(f"• Target: {data_config.target_col if data_config else 'None'}")
    st.write(f"• Features: {len(st.session_state.get('selected_features') or (data_config.feature_cols if data_config else []))}")
    st.write(f"• X_test shape: {X_test.shape if X_test is not None else 'None'}")
    st.write(f"• Trained models: {len(st.session_state.get('trained_models', {}))}")
    st.write(f"• Permutation importance: {len(perm_data)}")
    st.write(f"• SHAP results: {len(shap_data)}")
    st.write(f"• PDP results: {len(pdp_data)}")
    _lt = st.session_state.get("last_timings", {})
    if _lt:
        st.write("• Last timings (s):", ", ".join(f"{k}={v}s" for k, v in _lt.items()))
