"""
Page 02: Exploratory Data Analysis (Redesigned)

Architecture: Data first, coaching second.
  Section 0: At-a-Glance Header
  Section 1: Data Snapshot (interactive table + column inspector)
  Section 2: Shape of the Data (distributions, outliers, missing)
  Section 3: Relationships (correlations, target, feature explorer, clusters)
  Section 4: Macro Shape (PCA, UMAP, TDA, Mapper) — ≥16 features only
  Section 5: Coaching Layer (insight ledger summary)
  Section 6: Classical Diagnostics (plausibility, Q-Q, VIF, influence)
  Section 7: Table 1 (publication summary, collapsed)

Data flow:
  get_data() → df → detect_regime() → DatasetRegime drives all layout decisions
  compute_dataset_profile() → profile (cached)
  compute_dataset_signals() → signals (cached)
  InsightLedger: written by auto-detectors + user promotion, read by coaching layer
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional, Dict, Any, List
from math import ceil

from utils.session_state import (
    init_session_state, get_data, DataConfig,
    TaskTypeDetection, CohortStructureDetection, log_methodology
)
from utils.storyline import render_breadcrumb, render_page_navigation
from data_processor import get_numeric_columns
from utils.perf_cache import cached_numeric_summary, cached_target_correlations
from utils.theme import (
    inject_custom_css, render_guidance, render_reviewer_concern,
    render_step_indicator, render_sidebar_workflow
)
from utils.table_export import table
from utils.insight_ledger import (
    Insight, get_ledger,
    MODEL_FAMILY_LINEAR, MODEL_FAMILY_TREE, MODEL_FAMILY_NEURAL,
    MODEL_FAMILY_DISTANCE, MODEL_FAMILY_MARGIN, MODEL_FAMILY_PROBABILISTIC,
    ISSUE_MODEL_RELEVANCE,
)
from ml.regime import detect_regime
from ml.eda_recommender import compute_dataset_signals, recommend_eda, DatasetSignals, EDARecommendation
from ml import eda_actions
from ml.plot_narrative import (
    narrative_eda_influence,
    narrative_eda_normality,
    narrative_eda_multicollinearity,
)


# ============================================================================
# PAGE SETUP
# ============================================================================

init_session_state()
st.set_page_config(page_title="EDA", page_icon="📈", layout="wide")
inject_custom_css()
render_sidebar_workflow(current_page="02_EDA")
render_step_indicator(2, "Exploratory Data Analysis")
st.title("📈 Explore Your Data")
render_breadcrumb("02_EDA")
from utils.test_lockbox import render_lockbox_status
render_lockbox_status("Descriptive views below cover every row you are working with; automated selection and modeling decisions use training rows only.")
render_page_navigation("02_EDA")

# ============================================================================
# DATA LOADING + GUARDS
# ============================================================================

df = get_data()
if df is None:
    st.warning("Please upload data in the Upload & Audit page first.")
    st.stop()
if len(df) == 0 or len(df.columns) == 0:
    st.warning("Your dataset is empty.")
    st.stop()

task_mode = st.session_state.get("task_mode")
if task_mode == "hypothesis_testing":
    st.info("🔬 **Hypothesis Testing Mode**: EDA is available, but some prediction-specific features may be limited.")
elif task_mode != "prediction":
    st.warning("Please select a task mode in the Upload & Audit page.")
    st.stop()

data_config: Optional[DataConfig] = st.session_state.get("data_config")

# Staleness guard: if data_config references columns not in df, reset downstream state
if data_config and data_config.target_col and data_config.target_col not in df.columns:
    from utils.session_state import reset_data_dependent_state
    reset_data_dependent_state()
    st.rerun()

if task_mode == "prediction" and (data_config is None or not data_config.target_col):
    st.warning("Please select target and features in the Upload & Audit page first.")
    st.stop()

# Feature engineering warning
if st.session_state.get("feature_engineering_applied"):
    n_eng = len(st.session_state.get("engineered_feature_names", []))
    st.warning(
        f"⚠️ **Engineered dataset active ({n_eng} features created).** "
        "EDA now analyzes the engineered data. To revert: re-upload on Upload & Audit."
    )

target_col = data_config.target_col if data_config else None
feature_cols = (
    data_config.feature_cols
    if data_config and data_config.feature_cols
    else [c for c in df.columns if c != target_col]
)
_has_target = target_col is not None and target_col in df.columns

# ── Sealed rows, quarantined from the paths that model ───────────────────
# `utils/test_lockbox.py`'s contract: "Every target-aware step upstream of
# Train & Compare — feature-engineering fits, feature selection,
# target-association views — operates on training rows only, via
# train_row_mask()." This page did not call it. The two paths where that costs
# something real are quarantined here, the same way pages/04 does it:
#
#   - the dataset profile, because it drives the model coach's picks, and a
#     profile computed on held-out people lets the test set choose the models;
#   - quick_probe_baselines, which runs its own 80/20 split and FITS models —
#     a modeling step wearing an EDA costume, reporting a score for rows it
#     has already been shown.
#
# Descriptive views elsewhere on this page still cover every row, which is what
# the lockbox status line above states. See CONTRACT-017 for the remainder.
from utils.test_lockbox import train_row_mask

# train_row_mask already returns all-True in exploratory mode and with no
# lockbox, so the scoping claim below follows from the mask rather than from a
# second reading of the same state.
_train_mask = train_row_mask(df.index)
_train_df = df.loc[_train_mask]
# A lockbox that sealed everything would leave nothing to profile; fall back
# rather than crash, and do not claim a scoping that did not happen.
if _train_df.empty:
    _train_df = df
    _train_mask = pd.Series(True, index=df.index)
_lockbox_scoped = bool((~_train_mask).any())

# quick_probe_baselines is the one EDA action that fits models. Anything added
# here gets the masked frame and says so on screen.
_TRAIN_ONLY_ACTIONS = {"quick_probe_baselines"}


def _frame_for_action(run_action: str) -> pd.DataFrame:
    """The rows an EDA action may see: training rows only if it models."""
    return _train_df if run_action in _TRAIN_ONLY_ACTIONS else df


_TRAIN_SCOPE_CAPTION = (
    "held-out test rows are excluded to prevent selection leakage."
)


def _content_fingerprint(d):
    """Cache key for every cached computation on this page.

    Streamlit skips hashing `_`-prefixed params (like `_df`), so the key has to
    be passed as a separate non-prefixed argument. What that argument *contains*
    is the whole question.

    **Shape and column names are not enough** (`T0-LIVE-005`). Two cohort runs
    of the same study have identical row counts and identical columns and
    different rows — a median split, a 1:1 matched case-control, a balanced sex
    split. Under a shape-only key those two runs collide, and cohort A's
    correlation matrix, skew list, outlier heatmap and interaction ranking are
    served to cohort B. Cohort runs are the newest subsystem in the app, so the
    collision lands exactly where it is least expected.

    The digest makes the key follow the values. Cheap: `hash_pandas_object` is
    a vectorised row hash, and it runs once per rerun rather than per cache
    lookup.
    """
    import hashlib
    try:
        digest = int(pd.util.hash_pandas_object(d, index=True).sum())
    except Exception:
        # Unhashable cells (a list from nested JSON) must not collapse the key
        # to something stable — that would stop every cache from ever missing.
        digest = hashlib.md5(repr(d.head(50).values.tobytes()).encode()).hexdigest()
    return (len(d), len(d.columns), tuple(map(str, d.columns)), str(digest))


# One fingerprint, used by every cache on this page. There used to be two: a
# shape-only tuple for the eight older caches and a content digest for the four
# macro-shape ones added with T0-LIVE-001. Having both meant the principle was
# written down in one place and applied in the other.
_data_fingerprint = _content_fingerprint(df)
# The profile is computed on a different frame, so it needs its own key. Reusing
# the full-frame fingerprint would serve the all-rows profile to the masked call
# and put the leak straight back.
_train_fingerprint = (
    _data_fingerprint if _train_df is df else _content_fingerprint(_train_df)
)


def _macro_fp(d):
    """Kept as the name the macro-shape wrappers call; same function now."""
    return _content_fingerprint(d)

# Detection values
task_type_detection: TaskTypeDetection = st.session_state.get(
    "task_type_detection", TaskTypeDetection()
)
cohort_structure_detection: CohortStructureDetection = st.session_state.get(
    "cohort_structure_detection", CohortStructureDetection()
)
task_type_final = task_type_detection.final or (data_config.task_type if data_config else None)
cohort_type_final = cohort_structure_detection.final or "cross_sectional"
entity_id_final = cohort_structure_detection.entity_id_final

# ============================================================================
# REGIME DETECTION + PROFILE + SIGNALS
# ============================================================================

regime = detect_regime(df, feature_cols, target_col)
ledger = get_ledger()

# Dataset profile (cached)
@st.cache_data(show_spinner="Profiling dataset structure (one-time per dataset)…")
def _compute_profile(_df, target, features, task_type, outlier_method, data_id=None):
    from ml.dataset_profile import compute_dataset_profile
    return compute_dataset_profile(_df, target, features, task_type, outlier_method)

# EDA settings in sidebar
with st.sidebar:
    with st.expander("⚙️ EDA Settings", expanded=False):
        outlier_method = st.selectbox(
            "Outlier detection method",
            ["iqr", "mad", "zscore", "percentile"],
            index=0,
            key="eda_outlier_method",
        )

profile = _compute_profile(
    _train_df, target_col or feature_cols[0],
    feature_cols, task_type_final or "regression", outlier_method,
    data_id=_train_fingerprint,
)
st.session_state["dataset_profile"] = profile
# Which rows the profile describes, recorded beside it. Pages 05, 06 and 10 read
# `dataset_profile` and cannot otherwise tell: since the lockbox mask above, its
# p/n ratio, missingness rate and data-sufficiency verdict describe the training
# rows, and page 10 copies those numbers into the exported record. A number whose
# population is not stated is a number the reader will assume is about everyone.
st.session_state["dataset_profile_scope"] = {
    "rows": "training" if _lockbox_scoped else "all",
    "n_rows": int(_train_mask.sum()),
    "n_rows_total": int(len(df)),
    "reason": ("held-out test rows are excluded to prevent selection leakage"
               if _lockbox_scoped else "no rows are sealed in this analysis"),
}
if _lockbox_scoped:
    st.caption(
        f"The dataset profile and quick baselines see n={int(_train_mask.sum())} "
        f"training rows; {_TRAIN_SCOPE_CAPTION}"
    )

# Signals (cached)
@st.cache_data(show_spinner="Scanning statistical signals (one-time per dataset)…")
def _compute_signals(_df, target, task_type, cohort_type, entity_id, outlier_method, _feature_cols=None, data_id=None):
    return compute_dataset_signals(_df, target, task_type, cohort_type, entity_id, outlier_method=outlier_method, feature_cols=_feature_cols)

try:
    signals = _compute_signals(
        df, target_col, task_type_final, cohort_type_final, entity_id_final, outlier_method,
        _feature_cols=feature_cols,
        data_id=_data_fingerprint,
    )
except Exception as e:
    st.warning(f"Signal computation partially failed: {str(e)[:100]}")
    signals = DatasetSignals(
        n_rows=len(df), n_cols=len(df.columns),
        target_name=target_col, task_type_final=task_type_final,
        cohort_type_final=cohort_type_final, entity_id_final=entity_id_final,
    )

eda_recommendations = recommend_eda(signals)


# ============================================================================
# HELPER: Auto-generate insights from profile/signals
# ============================================================================

def _count_word(n: int, noun: str) -> str:
    """'1 predictor' / '3 predictors' — manuscript prose avoids '(s)'."""
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def _auto_generate_insights():
    """Write auto-detected insights to the ledger. Idempotent via upsert."""

    # Sufficiency. DataSufficiencyLevel values are abundant/adequate/limited/
    # scarce/critical — an earlier vocabulary ("insufficient"/"borderline")
    # matched nothing, so these insights never fired even on p >> n data.
    # Insight ids are kept for downstream resolution mappings.
    sufficiency = getattr(getattr(profile, "data_sufficiency", None), "value", "adequate")
    # `AUDIT-023`. THE DENOMINATOR IS THE SCREENED SET, NOT THE KEPT ONE.
    # `regime.n_features` counts the columns currently in `feature_cols`, which
    # after a selection on page 04 is what SURVIVED — and the sentences below
    # call them *candidate predictors*. §A5.4's ⚠ clause is explicit that a
    # predictor counts toward sample size even when it is later dropped,
    # because it was looked at. So a 40-candidate study that kept 8 was
    # reporting a 5x better ratio than it earned, under the word "candidate".
    #
    # `ml.candidate_predictors` is the one place that arithmetic lives; the
    # phrase it returns is quoted rather than re-composed here, and where no
    # selection was recorded it is the plain count so a reader is not sent
    # looking for a screening step that did not happen.
    from ml.candidate_predictors import candidate_count as _cand_count
    from ml.candidate_predictors import candidate_phrase as _cand_phrase
    from utils.workflow_provenance import get_provenance as _get_prov
    try:
        _prov = _get_prov()
    except Exception:
        _prov = None
    _cands = _cand_count(feature_cols, _prov)
    _cand_text = _cand_phrase(_cands)
    _suff_ratio = regime.n_rows / max(_cands.screened, 1)
    _suff_ratio_str = f"{_suff_ratio:.2f}:1" if _suff_ratio < 10 else f"{_suff_ratio:.0f}:1"
    if sufficiency == "critical":
        ledger.upsert(Insight(
            id="eda_sufficiency_insufficient",
            source_page="02_EDA", category="sufficiency", severity="blocker",
            finding=f"Sample size may be insufficient ({regime.n_rows:,} rows, {_cands.screened} candidate predictors, {_suff_ratio_str} observations per candidate)",
            implication="Complex models will likely overfit. Prefer simple baselines.",
            recommended_action="Reduce features or gather more data",
            manuscript_text=(
                f"the sample size was small relative to the number of candidate "
                f"predictors ({regime.n_rows:,} observations, {_cand_text}), "
                f"which limits statistical power and increases overfitting risk"
            ),
            relevant_pages=["04_Feature_Selection", "06_Train_and_Compare", "10_Report_Export"],
            model_scope=[MODEL_FAMILY_NEURAL],  # most affected by low sample size
        ))
    elif sufficiency in ("scarce", "limited"):
        ledger.upsert(Insight(
            id="eda_sufficiency_borderline",
            source_page="02_EDA", category="sufficiency", severity="warning",
            finding=f"Data sufficiency is {sufficiency} ({regime.n_rows:,} rows, {_cands.screened} candidate predictors)",
            implication="Prefer simpler models and tighter regularization.",
            recommended_action="Consider feature reduction before complex modeling",
            manuscript_text=(
                f"the modest ratio of observations to candidate predictors "
                f"({regime.n_rows:,} observations, {_cand_text}) constrained the "
                f"model complexity that could be reliably supported"
            ),
            relevant_pages=["04_Feature_Selection", "06_Train_and_Compare"],
            model_scope=[MODEL_FAMILY_NEURAL],  # most affected by low sample size
        ))

    # Leakage
    if signals.leakage_candidate_cols:
        for col in signals.leakage_candidate_cols:
            ledger.upsert(Insight(
                id=f"eda_leakage_{col}",
                source_page="02_EDA", category="relationship", severity="blocker",
                finding=f"Potential target leakage: {col} has >0.95 correlation with target",
                implication="Model performance will be artificially inflated",
                affected_features=[col],
                recommended_action=f"Remove {col} from feature set",
                manuscript_text=(
                    f"the predictor {col} was nearly collinear with the outcome "
                    f"(|r| > 0.95), raising the possibility of information leakage; "
                    f"results including this predictor should be interpreted with caution"
                ),
                relevant_pages=["04_Feature_Selection", "10_Report_Export"],
            ))

    # Collinearity — cluster correlated features into groups instead of per-pair
    # (high_corr_pairs already filtered to user's feature_cols at computation time)
    max_corr = signals.collinearity_summary.get("max_corr", 0)
    high_pairs = signals.collinearity_summary.get("high_corr_pairs", [])
    if high_pairs:
        # Build adjacency graph and find connected components
        from collections import defaultdict, deque
        adj = defaultdict(set)
        pair_corrs = {}
        for a, b, corr in high_pairs:
            adj[a].add(b)
            adj[b].add(a)
            pair_corrs[(a, b)] = float(corr)
            pair_corrs[(b, a)] = float(corr)

        visited = set()
        clusters = []
        for node in adj:
            if node not in visited:
                cluster = []
                queue = deque([node])
                while queue:
                    n = queue.popleft()
                    if n in visited:
                        continue
                    visited.add(n)
                    cluster.append(n)
                    for neighbor in adj[n]:
                        if neighbor not in visited:
                            queue.append(neighbor)
                if len(cluster) >= 2:
                    # Find max correlation within cluster
                    cluster_max_corr = 0
                    for i, a in enumerate(cluster):
                        for b in cluster[i+1:]:
                            c = pair_corrs.get((a, b), 0)
                            cluster_max_corr = max(cluster_max_corr, c)
                    clusters.append((cluster, cluster_max_corr))

        for cluster_features, cluster_max in clusters:
            n_feats = len(cluster_features)
            feat_list = ", ".join(cluster_features[:6])
            if n_feats > 6:
                feat_list += f" +{n_feats - 6} more"
            ledger.upsert(Insight(
                id=f"eda_corr_cluster_{'_'.join(sorted(cluster_features[:3]))}",
                source_page="02_EDA", category="relationship", severity="warning",
                finding=f"Collinearity cluster: {n_feats} features are intercorrelated (max r={cluster_max:.2f}): {feat_list}",
                implication=f"Keeping all {n_feats} may inflate variance in linear models. Consider retaining 1-2 representatives.",
                affected_features=cluster_features,
                recommended_action=f"Review in Feature Selection — consider dropping {n_feats - 1} of {n_feats}",
                manuscript_text=(
                    f"a cluster of {n_feats} intercorrelated predictors was present "
                    f"(maximum pairwise r = {cluster_max:.2f}), which can inflate the "
                    f"variance of coefficient estimates and complicate attribution of "
                    f"importance among the correlated predictors"
                ),
                relevant_pages=["04_Feature_Selection", "05_Preprocess"],
                model_scope=ISSUE_MODEL_RELEVANCE["collinearity"],  # linear only
                metadata={"max_correlation": cluster_max, "cluster_size": n_feats},
            ))

    # Missing data — synthesize into severity tiers, not per-column
    if signals.high_missing_cols:
        severe_missing = [(c, signals.missing_rate_by_col.get(c, 0)) for c in signals.high_missing_cols if signals.missing_rate_by_col.get(c, 0) > 0.3]
        moderate_missing = [(c, signals.missing_rate_by_col.get(c, 0)) for c in signals.high_missing_cols if 0.05 < signals.missing_rate_by_col.get(c, 0) <= 0.3]

        if severe_missing:
            cols_str = ", ".join(f"{c} ({r:.0%})" for c, r in severe_missing[:5])
            ledger.upsert(Insight(
                id="eda_missing_severe",
                source_page="02_EDA", category="data_quality", severity="warning",
                finding=f"{len(severe_missing)} feature(s) with >30% missing: {cols_str}",
                implication="High missingness may require column removal or advanced imputation (MICE, kNN). Simple mean imputation may distort distributions.",
                affected_features=[c for c, _ in severe_missing],
                recommended_action="Review in Preprocessing — consider dropping or advanced imputation",
                manuscript_text=(
                    f"{_count_word(len(severe_missing), 'predictor')} exhibited "
                    f"substantial missingness (>30% of values: {cols_str}), which may "
                    f"bias estimates if the missingness mechanism is not random"
                ),
                relevant_pages=["05_Preprocess", "10_Report_Export"],
                metadata={"n_features": len(severe_missing), "max_rate": max(r for _, r in severe_missing)},
            ))
        if moderate_missing:
            cols_str = ", ".join(f"{c} ({r:.0%})" for c, r in moderate_missing[:8])
            ledger.upsert(Insight(
                id="eda_missing_moderate",
                source_page="02_EDA", category="data_quality", severity="info",
                finding=f"{len(moderate_missing)} feature(s) with 5-30% missing: {cols_str}",
                # `AUDIT-007` · CLINICAL_SURVEY_PACK.md §A2 anti-pattern 2
                # [SETTLED as bad] and Cross-cutting 11. BEFORE, this read:
                # "Standard imputation (median/mode) should be sufficient.
                #  Consider adding missingness indicator features."
                # That asserted the sufficiency of the one method the registry
                # settles against — "Mean/median imputation. Understates
                # variance, destroys the distribution, indefensible in a
                # manuscript" — and recommended an indicator unconditionally,
                # where §A2 splits it: legitimate for prediction, biased and
                # not to be used for an unbiased association estimate.
                # AFTER says less and is true: it states what the method costs
                # and names the alternative that is actually on the shelf
                # ("iterative (MICE)", pages/05_Preprocess.py:600). The method
                # is not withdrawn and nothing is blurred — the same subject,
                # a weaker claim, checkable against §A2.
                #
                # The route is named because MICE is NOT on the default path:
                # pages/05_Preprocess.py:580 skips the whole per-model
                # configuration block while "Smart Defaults" is selected, so
                # the option exists only under "Advanced (full control)".
                # "MICE is offered in Preprocessing" would have been the second
                # false claim in the same sentence that corrected the first.
                implication=(
                    "Median/mode imputation understates the variance of the imputed column "
                    "and distorts its distribution — the filled values carry none of the "
                    "uncertainty of the values they replace. Multiple imputation (MICE) is "
                    "available in Preprocessing under Advanced (full control). A missingness "
                    "indicator is legitimate for a prediction model and is contraindicated "
                    "for an unbiased association estimate."
                ),
                affected_features=[c for c, _ in moderate_missing],
                recommended_action="Address in Preprocessing",
                relevant_pages=["05_Preprocess"],
                metadata={"n_features": len(moderate_missing)},
            ))

    # Target skewness
    if _has_target and task_type_final == "regression":
        skew = signals.target_stats.get("skew", 0)
        if skew and abs(skew) > 1.5:
            ledger.upsert(Insight(
                id="eda_target_skew",
                source_page="02_EDA", category="distribution", severity="warning",
                finding=f"Target is skewed (skew={skew:.2f})",
                implication="May affect loss function choice and prediction intervals",
                manuscript_text=(
                    f"the outcome distribution was skewed (skewness = {skew:.2f}), "
                    f"which can affect error-based metrics and prediction intervals"
                ),
                affected_features=[target_col],
                recommended_action="Apply a target transformation on the Train & Compare page (Log, Yeo-Johnson, or Box-Cox). Predictions are automatically back-transformed and metrics reported on the original scale. Tree-based models and Huber regression are also robust to target skew.",
                relevant_pages=["06_Train_and_Compare"],
                model_scope=ISSUE_MODEL_RELEVANCE["skewness"],  # linear, neural, distance
                metadata={"skewness": float(skew)},
            ))

    # Target outliers. ml/eda_actions.target_profile() used to detect this from
    # the Deep Dive tab; that action is delisted, so the finding is detected here
    # from the same signals.target_stats it read. Same insight id on both sides,
    # so if target_profile() is ever dispatched again the two upserts collapse
    # into one entry instead of filing the finding twice.
    if _has_target and task_type_final == "regression":
        target_outlier_rate = signals.target_stats.get("outlier_rate", 0) or 0
        if target_outlier_rate > 0.1:
            ledger.upsert(Insight(
                id="eda_target_outliers",
                source_page="02_EDA", category="distribution", severity="warning",
                finding=f"High outlier rate in target: {target_outlier_rate:.1%} of values flagged",
                implication="Squared-error losses are dominated by the extremes, so a model can be tuned almost entirely by a small tail of observations.",
                manuscript_text=(
                    f"{target_outlier_rate:.1%} of outcome values were flagged as "
                    f"outliers, which inflates squared-error losses and can allow a "
                    f"small number of observations to dominate model fitting"
                ),
                affected_features=[target_col],
                recommended_action="Use a robust loss (Huber) or tree-based models in Train & Compare, or trim the target if the extreme values are measurement artifacts.",
                relevant_pages=["06_Train_and_Compare"],
                model_scope=ISSUE_MODEL_RELEVANCE["outliers"],  # linear, neural, distance
                metadata={"outlier_rate": float(target_outlier_rate), "detection_method": outlier_method},
            ))

    # Class imbalance — affects all models
    if _has_target and task_type_final == "classification":
        imbalance = signals.target_stats.get("class_imbalance_ratio", 1.0)
        if imbalance and imbalance < 0.35:
            ledger.upsert(Insight(
                id="eda_class_imbalance",
                source_page="02_EDA", category="distribution", severity="warning",
                finding=f"Class imbalance detected (ratio={imbalance:.2f})",
                implication="Accuracy alone may be misleading. Use F1, balanced accuracy, or AUROC.",
                manuscript_text=(
                    f"the outcome classes were imbalanced (minority-to-majority "
                    f"ratio = {imbalance:.2f}), so threshold-dependent metrics "
                    f"should be interpreted alongside AUROC and F1"
                ),
                affected_features=[target_col],
                # `GUIDED-049`. Was "Use class weighting or stratified sampling" —
            # the one field that tells the user what to DO named the
            # contraindicated step, and pointed at the two pages that do it.
            recommended_action=(
                "Report PR-AUC and calibration alongside discrimination, and "
                "choose the decision threshold from the costs of the two "
                "errors. Rebalancing is contraindicated for a risk model"
            ),
                relevant_pages=["05_Preprocess", "06_Train_and_Compare"],
                # model_scope=[] → all models affected
                metadata={"imbalance_ratio": float(imbalance)},
            ))

    # Feature skewness — use cached computation
    @st.cache_data
    def _get_skewed_features(_df, _feature_cols, data_id=None):
        cols = _df[_feature_cols].select_dtypes(include=[np.number]).columns
        skewed = []
        for col in cols:
            try:
                sv = float(_df[col].skew())
                if abs(sv) > 2.0:
                    skewed.append((col, sv))
            except (TypeError, ValueError):
                pass
        return skewed

    skewed_list = _get_skewed_features(df, feature_cols, data_id=_data_fingerprint)
    if skewed_list:
        skew_names = ", ".join(f"{c} ({s:.1f})" for c, s in skewed_list[:8])
        if len(skewed_list) > 8:
            skew_names += f" +{len(skewed_list) - 8} more"
        ledger.upsert(Insight(
            id="eda_skew_group",
            source_page="02_EDA", category="distribution", severity="info",
            finding=f"{len(skewed_list)} feature(s) heavily skewed (|skew| > 2): {skew_names}",
            implication="Log or power transforms may improve linear model performance and reduce outlier influence",
            manuscript_text=(
                f"{_count_word(len(skewed_list), 'predictor')} exhibited strong "
                f"skewness (|skewness| > 2), which can increase the influence of "
                f"extreme values in scale-sensitive models"
            ),
            affected_features=[c for c, _ in skewed_list],
            recommended_action="Consider transforms in Feature Engineering or Preprocessing",
            relevant_pages=["03_Feature_Engineering", "05_Preprocess"],
            model_scope=ISSUE_MODEL_RELEVANCE["skewness"],  # linear, neural, distance
            metadata={"n_skewed": len(skewed_list), "features": {c: s for c, s in skewed_list}},
        ))

    # ------------------------------------------------------------------
    # OPPORTUNITIES — things to exploit, not just problems to fix
    # ------------------------------------------------------------------

    # Clean data opportunity
    n_issues = len([i for i in ledger.get_unresolved() if i.severity in ("blocker", "warning")])
    if n_issues == 0 and regime.n_features >= 5:
        ledger.upsert(Insight(
            id="eda_opportunity_clean_data",
            source_page="02_EDA", category="data_quality", severity="opportunity",
            finding="Dataset has no blockers or warnings — unusually clean",
            implication="You can lean into interpretable models (GLM, GAM) where coefficient interpretation is meaningful, rather than defaulting to black-box approaches",
            manuscript_text=("the dataset contained no blocking data-quality issues "
                             "(no severe missingness, leakage candidates, or "
                             "distributional anomalies)"),
            recommended_action="Consider GLM or GAM baselines in Train & Compare",
            relevant_pages=["06_Train_and_Compare"],
            resolved=True, resolved_by="Positive signal — no action needed",
            resolved_on_page="02_EDA", auto_generated=True,
        ))

    # Strong target signal opportunity
    # Both this check and the non-linearity check below need per-feature
    # correlations with the target; one cached vectorized pass replaces
    # thousands of per-column Series.corr calls on wide data.
    _corr_pearson, _corr_spearman = None, None
    if _has_target and task_type_final == "regression":
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns
        _corr_cols = tuple(c for c in numeric_cols if c != target_col)
        if _corr_cols:
            _corr_pearson, _corr_spearman = cached_target_correlations(
                df, target_col, _corr_cols)
    if _corr_pearson is not None and len(_corr_pearson) >= 2:
            top_corr = float(_corr_pearson.abs().max(skipna=True) or 0)
            if not np.isnan(top_corr) and top_corr > 0.7:
                ledger.upsert(Insight(
                    id="eda_opportunity_strong_signal",
                    source_page="02_EDA", category="relationship", severity="opportunity",
                    finding=f"Strong linear signal detected (max |r| with target = {top_corr:.2f})",
                    implication="Linear models may perform surprisingly well. Establish a strong OLS baseline before trying complex models.",
                    manuscript_text=(f"at least one predictor showed a strong linear "
                                     f"association with the outcome "
                                     f"(maximum |r| = {top_corr:.2f})"),
                    recommended_action="Run GLM baseline first in Train & Compare",
                    relevant_pages=["06_Train_and_Compare"],
                    metadata={"max_target_correlation": float(top_corr)},
                    resolved=True, resolved_by="Positive signal — no action needed",
                    resolved_on_page="02_EDA", auto_generated=True,
                ))

    # Non-linear relationship opportunity
    if _corr_pearson is not None and len(_corr_pearson) >= 3:
            # Compare Pearson vs Spearman to detect non-linearity
            _gap = (_corr_spearman.abs() - _corr_pearson.abs()).dropna()
            if len(_gap) > 0:
                avg_gap = float(_gap.mean())
                if avg_gap > 0.08:
                    ledger.upsert(Insight(
                        id="eda_opportunity_nonlinear",
                        source_page="02_EDA", category="relationship", severity="opportunity",
                        finding=f"Features show non-linear relationships with target (avg Spearman-Pearson gap = {avg_gap:.3f})",
                        implication="Tree-based models (RF, XGBoost) or GAMs may capture structure that linear models miss",
                        manuscript_text=(f"rank-based predictor–outcome associations "
                                         f"exceeded linear associations (mean "
                                         f"Spearman−Pearson gap = {avg_gap:.3f}), "
                                         f"consistent with non-linear structure"),
                        recommended_action="Include tree-based models in Train & Compare",
                        relevant_pages=["06_Train_and_Compare"],
                        metadata={"spearman_pearson_gap": float(avg_gap)},
                        resolved=True, resolved_by="Positive signal — no action needed",
                        resolved_on_page="02_EDA", auto_generated=True,
                    ))

    # High n/p ratio opportunity
    n_p_ratio = regime.n_rows / max(regime.n_features, 1)
    if n_p_ratio > 100:
        ledger.upsert(Insight(
            id="eda_opportunity_high_np",
            source_page="02_EDA", category="sufficiency", severity="opportunity",
            finding=f"Large sample-to-feature ratio ({n_p_ratio:.0f}:1) — plenty of data relative to complexity",
            implication="You can afford more complex models (deep trees, neural nets) without overfitting. Cross-validation will be reliable.",
            manuscript_text=(f"the sample size was large relative to the number of "
                             f"predictors ({n_p_ratio:.0f}:1 observations per "
                             f"predictor), supporting stable model estimation"),
            recommended_action="Consider full model suite in Train & Compare",
            relevant_pages=["06_Train_and_Compare"],
            metadata={"n_p_ratio": float(n_p_ratio)},
            resolved=True, resolved_by="Positive signal — no action needed",
            resolved_on_page="02_EDA", auto_generated=True,
        ))

    # Classification: balanced classes opportunity
    if _has_target and task_type_final == "classification":
        imbalance = signals.target_stats.get("class_imbalance_ratio", 0)
        if imbalance and imbalance > 0.7:
            ledger.upsert(Insight(
                id="eda_opportunity_balanced",
                source_page="02_EDA", category="distribution", severity="opportunity",
                finding=f"Classes are well-balanced (ratio = {imbalance:.2f})",
                implication="Accuracy is a valid metric. No need for class weighting or oversampling.",
                manuscript_text=(f"the outcome classes were well balanced "
                                 f"(minority-to-majority ratio = {imbalance:.2f})"),
                recommended_action="Standard metrics will be reliable in Train & Compare",
                relevant_pages=["06_Train_and_Compare"],
                resolved=True, resolved_by="Positive signal — no action needed",
                resolved_on_page="02_EDA", auto_generated=True,
            ))


_auto_generate_insights()


# ============================================================================
# SECTION 0: AT-A-GLANCE HEADER
# ============================================================================
# (Title renders in the header cluster at the top of the page, matching the
# title-first order of the other workflow pages.)

if regime.n_features > 500:
    st.caption(
        f"⏱️ Wide dataset ({regime.n_features:,} features): the first visit to "
        "this page computes its statistics once — the spinners below name each "
        "step — and everything afterward is served from cache."
    )

cols = st.columns([1, 1, 1, 1, 1, 1])
with cols[0]:
    st.metric("Rows", f"{regime.n_rows:,}")
with cols[1]:
    st.metric("Features", f"{regime.n_features}")
with cols[2]:
    st.metric("Numeric", f"{regime.n_numeric}")
with cols[3]:
    st.metric("Categorical", f"{regime.n_categorical}")
with cols[4]:
    _missing_by_col = df[feature_cols].isnull().mean()
    missing_pct = _missing_by_col.mean() * 100
    # The overall mean dilutes a badly-missing column across thousands of
    # complete ones — surface the worst offender in the tooltip.
    _missing_help = "Mean missingness across selected features."
    if len(_missing_by_col) > 0 and _missing_by_col.max() > 0:
        _worst_col = _missing_by_col.idxmax()
        _missing_help += f" Highest single feature: {_worst_col} ({_missing_by_col.max():.0%})."
    st.metric("Missing", f"{missing_pct:.1f}%", help=_missing_help)
with cols[5]:
    sufficiency_val = getattr(getattr(profile, "data_sufficiency", None), "value", "adequate")
    # Compact verdicts: anything longer than ~4 glyphs ('Critical',
    # '✗ Poor') truncates at the metric tile's numeral font size at
    # six-tiles-across width. Full term stays in the tooltip. Covers every
    # DataSufficiencyLevel value.
    _suff_display = {"abundant": "High", "adequate": "OK",
                     "limited": "Fair", "scarce": "Weak",
                     "critical": "Poor"}.get(sufficiency_val, sufficiency_val.title()[:4])
    st.metric("Sufficiency", _suff_display,
              help=f"Data sufficiency: {sufficiency_val} (based on samples-per-feature ratio)")

# Alert ribbon — only if blockers exist
if ledger.has_blockers():
    n_blockers = ledger.summary()["blockers"]
    st.error(f"🚨 **{n_blockers} blocker(s) detected** — resolve before modeling. See Coaching Layer below.")

if regime.show_sample_size_warning:
    st.warning(f"⚠️ Small dataset ({regime.n_rows} rows). All data points shown; be cautious about overfitting.")


# ============================================================================
# SECTION 1: DATA SNAPSHOT
# ============================================================================

st.markdown("---")
_eda_tabs = st.tabs(["📷 Data Snapshot", "📊 Distributions & Outliers", "🔗 Relationships"])

with _eda_tabs[0]:
    st.header("Data Snapshot")
    st.caption("See your data. Sort, filter, and inspect columns to build initial intuition.")

    # Interactive dataframe
    st.dataframe(
        df.head(200),
        width="stretch",
        height=350,
    )

    # Type filter pills and column inspector
    type_label = f"{regime.n_numeric} numeric · {regime.n_categorical} categorical"
    if regime.n_datetime > 0:
        type_label += f" · {regime.n_datetime} datetime"
    st.caption(type_label)

    # Column inspector
    with st.expander("🔍 Column Inspector", expanded=False):
        inspect_col = st.selectbox("Select column to inspect", df.columns, key="col_inspector")
        if inspect_col:
            col_data = df[inspect_col]
            ic1, ic2, ic3, ic4 = st.columns(4)
            with ic1:
                st.metric("Type", str(col_data.dtype))
            with ic2:
                st.metric("Unique", f"{col_data.nunique():,}")
            with ic3:
                st.metric("Missing", f"{col_data.isnull().sum():,} ({col_data.isnull().mean():.1%})")
            with ic4:
                if pd.api.types.is_numeric_dtype(col_data):
                    st.metric("Mean", f"{col_data.mean():.3f}")
                else:
                    st.metric("Top Value", str(col_data.mode().iloc[0]) if len(col_data.mode()) > 0 else "N/A")

            if pd.api.types.is_numeric_dtype(col_data):
                # Sparkline histogram
                fig = px.histogram(col_data.dropna(), nbins=30, height=200)
                fig.update_layout(
                    showlegend=False, margin=dict(l=0, r=0, t=10, b=0),
                    xaxis_title="", yaxis_title="",
                )
                st.plotly_chart(fig)

                desc = col_data.describe()
                d1, d2, d3, d4 = st.columns(4)
                d1.metric("Min", f"{desc['min']:.3f}")
                d2.metric("Median", f"{desc['50%']:.3f}")
                d3.metric("Max", f"{desc['max']:.3f}")
                d4.metric("Std", f"{desc['std']:.3f}")
            else:
                # Value counts for categorical
                vc = col_data.value_counts().head(10)
                fig = px.bar(x=vc.index.astype(str), y=vc.values, height=200)
                fig.update_layout(
                    showlegend=False, margin=dict(l=0, r=0, t=10, b=0),
                    xaxis_title="", yaxis_title="Count",
                )
                st.plotly_chart(fig)

            # Show insights for this column
            col_insights = ledger.get_for_features([inspect_col])
            if col_insights:
                for ins in col_insights:
                    if not ins.resolved:
                        icon = {"blocker": "🚨", "warning": "⚠️", "info": "ℹ️", "opportunity": "💡"}.get(ins.severity, "ℹ️")
                        st.caption(f"{icon} {ins.finding}")


    # ============================================================================
    # SECTION 2: SHAPE OF THE DATA
    # ============================================================================

    st.markdown("---")
with _eda_tabs[1]:
    st.header("Shape of the Data")
    st.caption("Distributions, outliers, and missing data patterns. Build visual intuition before analyzing relationships.")

    # -- Target Distribution --------------------------------------------------
    if _has_target:
        st.subheader(f"Target: {target_col}")
        tc1, tc2 = st.columns(2)
        with tc1:
            fig_hist = px.histogram(df, x=target_col, nbins=30, title=f"Distribution of {target_col}")
            fig_hist.update_layout(template="plotly_white", height=350)
            st.plotly_chart(fig_hist)
        with tc2:
            if task_type_final == "classification":
                class_counts = df[target_col].value_counts().sort_index()
                fig_bar = px.bar(
                    x=class_counts.index.astype(str), y=class_counts.values,
                    title="Class Distribution",
                    labels={"x": "Class", "y": "Count"},
                )
                fig_bar.update_layout(template="plotly_white", height=350)
                st.plotly_chart(fig_bar)
                imbalance = class_counts.min() / class_counts.max()
                if imbalance < 0.35:
                    st.caption(f"⚠️ Class imbalance: {imbalance:.2f} ratio. Stratified sampling recommended.")
            else:
                fig_box = px.box(df, y=target_col, title=f"Box Plot of {target_col}")
                fig_box.update_layout(template="plotly_white", height=350)
                st.plotly_chart(fig_box)
                skew = signals.target_stats.get("skew")
                if skew and abs(skew) > 1.5:
                    st.caption(f"ℹ️ Skew = {skew:.2f} — log transform may help.")

    # -- Feature Distribution Gallery -----------------------------------------
    st.subheader("Feature Distributions")

    numeric_features = [f for f in feature_cols if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]
    cat_features = [f for f in feature_cols if f in df.columns and not pd.api.types.is_numeric_dtype(df[f])]

    if regime.distribution_mode == "summary":
        # Ultra-wide: summary-of-summaries view
        st.caption(f"Dataset has {regime.n_features} features — showing summary statistics. Use Column Inspector to drill into individual features.")
        if numeric_features:
            summary_df = cached_numeric_summary(df, tuple(numeric_features))
            summary_df.index.name = "Feature"
            table(summary_df.round(3).reset_index())

            # Distribution-of-distributions: skew histogram
            skews = df[numeric_features].skew().dropna()
            if len(skews) > 1:
                fig_skew = px.histogram(skews, nbins=20, title="Distribution of Feature Skewness")
                fig_skew.update_layout(template="plotly_white", height=250, xaxis_title="Skewness", yaxis_title="Count")
                st.plotly_chart(fig_skew)
    else:
        # Gallery mode: paginated 3×3 grid
        filter_options = ["All Features"]
        if numeric_features:
            filter_options.append(f"Numeric ({len(numeric_features)})")
        if cat_features:
            filter_options.append(f"Categorical ({len(cat_features)})")

        # Detect features with notable properties for filter pills
        high_missing_features = [f for f in feature_cols if signals.missing_rate_by_col.get(f, 0) > 0.05]
        if high_missing_features:
            filter_options.append(f"High Missing ({len(high_missing_features)})")

        selected_filter = st.pills("Filter features", filter_options, default="All Features", key="dist_filter")

        if selected_filter and "Numeric" in selected_filter:
            display_features = numeric_features
        elif selected_filter and "Categorical" in selected_filter:
            display_features = cat_features
        elif selected_filter and "High Missing" in selected_filter:
            display_features = high_missing_features
        else:
            display_features = feature_cols

        page_size = regime.gallery_page_size
        total_pages = max(1, ceil(len(display_features) / page_size))

        if total_pages > 1:
            gallery_page = st.number_input(
                f"Page (1-{total_pages})", min_value=1, max_value=total_pages,
                value=1, key="dist_gallery_page"
            )
        else:
            gallery_page = 1

        page_features = display_features[(gallery_page - 1) * page_size: gallery_page * page_size]
        st.caption(f"Showing {len(page_features)} of {len(display_features)} features (page {gallery_page}/{total_pages})")

        for row_start in range(0, len(page_features), 3):
            row_cols = st.columns(3)
            for j, col_widget in enumerate(row_cols):
                idx = row_start + j
                if idx < len(page_features):
                    feat = page_features[idx]
                    with col_widget:
                        if pd.api.types.is_numeric_dtype(df[feat]):
                            fig = px.histogram(df, x=feat, nbins=30, title=feat)
                        else:
                            vc = df[feat].value_counts().head(10)
                            fig = px.bar(x=vc.index.astype(str), y=vc.values, title=feat)
                        fig.update_layout(
                            template="plotly_white", height=220,
                            margin=dict(l=10, r=10, t=35, b=10),
                            showlegend=False,
                        )
                        st.plotly_chart(fig)

                        # Inline coaching annotation
                        if pd.api.types.is_numeric_dtype(df[feat]):
                            feat_skew = df[feat].skew()
                            if abs(feat_skew) > 2.0:
                                st.caption(f"ℹ️ Skew = {feat_skew:.1f}")
                            feat_missing = df[feat].isnull().mean()
                            if feat_missing > 0.05:
                                st.caption(f"⚠️ {feat_missing:.1%} missing")

    # -- Outlier Overview ------------------------------------------------------
    st.subheader("Outlier Overview")

    if numeric_features and regime.row_regime != "tiny":
        # Skip for tiny datasets (outlier detection on <100 rows is unreliable)
        from ml.outliers import detect_outliers

        # Cap at 50 features for performance; show note if capped
        _outlier_cap = 50
        _outlier_features = numeric_features[:_outlier_cap]

        @st.cache_data
        def _compute_outlier_heatmap(_df, _numeric_feats, methods, data_id=None):
            """Cached outlier prevalence computation."""
            outlier_data = {}
            for feat in _numeric_feats:
                feat_data = _df[feat].dropna()
                if len(feat_data) < 10:
                    continue
                row = {}
                for method in methods:
                    try:
                        mask, _ = detect_outliers(feat_data, method=method)
                        row[method.upper()] = float(mask.sum() / len(feat_data) * 100)
                    except Exception:
                        row[method.upper()] = 0.0
                outlier_data[feat] = row
            return outlier_data

        outlier_data = _compute_outlier_heatmap(df, _outlier_features, ["iqr", "zscore"], data_id=_data_fingerprint)
        if len(numeric_features) > _outlier_cap:
            st.caption(f"Showing {_outlier_cap} of {len(numeric_features)} features. Use Column Inspector for individual features.")

        if outlier_data:
            outlier_df = pd.DataFrame(outlier_data).T
            outlier_df = outlier_df.sort_values(outlier_df.columns[0], ascending=False)

            fig_outlier = go.Figure(data=go.Heatmap(
                z=outlier_df.values,
                x=outlier_df.columns.tolist(),
                y=outlier_df.index.tolist(),
                colorscale=[[0, "white"], [0.05, "#fef3c7"], [0.15, "#fbbf24"], [0.3, "#ef4444"]],
                zmin=0, zmax=max(20, outlier_df.values.max()),
                text=np.round(outlier_df.values, 1),
                texttemplate="%{text}%",
                hovertemplate="Feature: %{y}<br>Method: %{x}<br>Outlier %: %{z:.1f}%<extra></extra>",
            ))
            fig_outlier.update_layout(
                title="Outlier Prevalence by Feature × Method",
                template="plotly_white",
                height=max(300, len(outlier_data) * 22 + 80),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig_outlier)
            st.caption(f"Primary method for downstream: **{outlier_method.upper()}**. Change in sidebar settings.")
    elif not numeric_features:
        st.info("No numeric features for outlier analysis.")
    elif regime.row_regime == "tiny":
        st.caption(f"Outlier detection skipped — only {regime.n_rows} rows. Statistical outlier methods are unreliable at this sample size.")

    # -- Missing Data ----------------------------------------------------------
    total_missing = df[feature_cols].isnull().sum().sum()
    if total_missing > 0:
        st.subheader("Missing Data")
        missing_by_col = df[feature_cols].isnull().mean().sort_values(ascending=False)
        missing_cols = missing_by_col[missing_by_col > 0]

        fig_missing = px.bar(
            x=missing_cols.values * 100,
            y=missing_cols.index,
            orientation="h",
            title=f"Missing Data ({len(missing_cols)} columns with gaps)",
            labels={"x": "Missing %", "y": "Column"},
        )
        fig_missing.update_layout(template="plotly_white", height=max(250, len(missing_cols) * 25 + 60))
        st.plotly_chart(fig_missing)

        # Co-missingness pattern matrix (if meaningful)
        n_high_missing = sum(1 for v in missing_cols.values if v > 0.05)
        if n_high_missing >= 2:
            with st.expander("Co-missingness pattern matrix"):
                missing_matrix = df[missing_cols.index[:30]].isnull().astype(int)
                # Sample rows for visualization
                if len(missing_matrix) > 200:
                    missing_matrix = missing_matrix.sample(200, random_state=42).sort_index()
                fig_pattern = go.Figure(data=go.Heatmap(
                    z=missing_matrix.values.T,
                    x=list(range(len(missing_matrix))),
                    y=missing_matrix.columns.tolist(),
                    colorscale=[[0, "white"], [1, "#667eea"]],
                    showscale=False,
                ))
                fig_pattern.update_layout(
                    title="Missingness Pattern (white=present, blue=missing)",
                    template="plotly_white",
                    height=max(250, len(missing_matrix.columns) * 20 + 60),
                    xaxis_title="Sample index",
                )
                st.plotly_chart(fig_pattern)

        # -- Is that missingness informative? ----------------------------------
        # A gap can itself be data: if the target differs between the rows where
        # a column is missing and the rows where it is present, the missingness
        # carries signal and deserves an indicator column, not a silent median
        # fill. Nothing else in the app asks this question.
        if _has_target and task_type_final in ("regression", "classification"):

            # `cols` is hashed, not underscore-skipped: changing the selected
            # features must invalidate this, and the dataset fingerprint alone
            # would not notice.
            @st.cache_data(show_spinner=False)
            def _missingness_target_assoc(_df, target, task_type, cols, data_id=None):
                return eda_actions.missingness_target_association(
                    _df, target, task_type, candidate_cols=list(cols)
                )

            _mt = _missingness_target_assoc(
                df, target_col, task_type_final, tuple(feature_cols), data_id=_data_fingerprint
            )
            _mt_table = _mt.get("table")
            if _mt_table is not None and len(_mt_table) > 0:
                _mt_sig = _mt["n_significant"]
                with st.expander(
                    (f"Informative missingness — {_mt_sig} of {_mt['n_tested']} columns track {target_col}"
                     if _mt_sig else
                     f"Informative missingness — none of the {_mt['n_tested']} columns tested track {target_col}"),
                    expanded=bool(_mt_sig),
                ):
                    st.caption(
                        f"Each row compares **{target_col}** between the rows where that column is "
                        f"missing and the rows where it is present. A real gap means the values are "
                        f"not missing at random, and being missing is itself a predictor. q-values are "
                        f"Benjamini–Hochberg adjusted across the {_mt['n_tested']} tests — read those, "
                        f"not the raw p-values."
                    )
                    _mt_show = _mt_table.copy()
                    _mt_show["Missing %"] = (_mt_show["Missing %"] * 100).round(1)
                    for _mt_col in ("p-value", "q-value (BH)"):
                        _mt_show[_mt_col] = _mt_show[_mt_col].map(
                            lambda v: "—" if pd.isna(v) else (f"{v:.2e}" if v < 1e-4 else f"{v:.4f}")
                        )
                    for _mt_col in _mt_show.columns:
                        if pd.api.types.is_float_dtype(_mt_show[_mt_col]):
                            _mt_show[_mt_col] = _mt_show[_mt_col].round(3)
                    table(_mt_show, key="eda_missingness_association")
                    if _mt["skipped_low_n"]:
                        _mt_skipped = _mt["skipped_low_n"]
                        st.caption(
                            f"Not tested: {len(_mt_skipped)} column(s) have fewer than 20 rows on one "
                            f"side of the missing/present split, too few for the result to mean "
                            f"anything — {', '.join(map(str, _mt_skipped[:6]))}"
                            f"{f' +{len(_mt_skipped) - 6} more' if len(_mt_skipped) > 6 else ''}."
                        )

                if _mt_sig:
                    _mt_top = _mt_table.iloc[0]
                    ledger.upsert(Insight(
                        id="eda_missing_informative",
                        source_page="02_EDA", category="data_quality", severity="warning",
                        finding=(
                            f"Missingness is associated with {target_col} in "
                            f"{_count_word(_mt_sig, 'column')} — strongest is {_mt_top['Column']} "
                            f"({_mt_top['Test']}, BH q={_mt_top['q-value (BH)']:.3g})"
                        ),
                        implication=(
                            "The values are not missing at random. Imputing these columns without "
                            "recording that the value was absent discards signal and can bias estimates."
                        ),
                        affected_features=list(_mt["significant_cols"]),
                        recommended_action=(
                            "Add missing-indicator columns in Preprocessing, or train a model that takes "
                            "NaN natively (HistGradientBoosting, LightGBM, XGBoost)."
                        ),
                        manuscript_text=(
                            f"missingness was associated with the outcome for "
                            f"{_count_word(_mt_sig, 'predictor')} (Benjamini-Hochberg adjusted q < 0.05), "
                            f"indicating the values were not missing completely at random"
                        ),
                        relevant_pages=["05_Preprocess"],
                        metadata={
                            "n_significant": int(_mt_sig),
                            "n_tested": int(_mt["n_tested"]),
                            "top_column": str(_mt_top["Column"]),
                            "top_q": float(_mt_top["q-value (BH)"]),
                        },
                    ))


    # ============================================================================
    # SECTION 3: RELATIONSHIPS
    # ============================================================================

    st.markdown("---")
with _eda_tabs[2]:
    st.header("Relationships")
    st.caption("How features relate to each other and to the target.")

    # -- Correlation Matrix / Top Pairs ----------------------------------------
    st.subheader("Feature Correlations")

    if len(numeric_features) >= 2:
        corr_method = st.pills("Method", ["Pearson", "Spearman"], default="Pearson", key="corr_method")
        method_name = corr_method.lower() if corr_method else "pearson"

        # Include numeric target so the matrix surfaces feature↔target relationships, not just feature↔feature.
        corr_cols = list(numeric_features)
        if _has_target and pd.api.types.is_numeric_dtype(df[target_col]) and target_col not in corr_cols:
            corr_cols.append(target_col)

        @st.cache_data
        def _compute_corr(_df, _features, method, data_id=None):
            return _df[_features].corr(method=method).round(3)

        if regime.show_full_corr_matrix:
            # Full heatmap for narrow/medium datasets
            corr_matrix = _compute_corr(df, corr_cols, method_name, data_id=_data_fingerprint)
            threshold = st.slider("Highlight threshold", 0.0, 1.0, 0.8, 0.05, key="corr_threshold")

            fig_corr = px.imshow(
                corr_matrix,
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1,
                title=f"{corr_method} Correlation Matrix",
                aspect="auto",
            )
            fig_corr.update_layout(template="plotly_white", height=max(400, len(corr_cols) * 18 + 100))
            st.plotly_chart(fig_corr)

            # List pairs above threshold (numpy-based)
            corr_vals = corr_matrix.values
            idx_upper = np.triu_indices_from(corr_vals, k=1)
            upper_vals = corr_vals[idx_upper]
            mask = np.abs(upper_vals) >= threshold
            if mask.any():
                cols_list = corr_matrix.columns.tolist()
                pairs_above = pd.DataFrame({
                    "Feature A": [cols_list[idx_upper[0][i]] for i in np.where(mask)[0]],
                    "Feature B": [cols_list[idx_upper[1][i]] for i in np.where(mask)[0]],
                    "Correlation": [round(float(upper_vals[i]), 3) for i in np.where(mask)[0]],
                }).sort_values("Correlation", key=abs, ascending=False)
                st.caption(f"{len(pairs_above)} pairs above |r| ≥ {threshold}")
                table(pairs_above)
        else:
            # Wide/ultra-wide: top-N pairs via numpy (avoids O(n²) Python loop)
            top_n = regime.corr_top_n

            @st.cache_data
            def _top_corr_pairs(_df, _features, method, n, data_id=None):
                corr = _df[_features].corr(method=method).values
                cols = _features
                idx_upper = np.triu_indices_from(corr, k=1)
                vals = corr[idx_upper]
                # Get top N by absolute value
                top_idx = np.argsort(np.abs(vals))[-n:][::-1]
                return pd.DataFrame([
                    {
                        "Feature A": cols[idx_upper[0][i]],
                        "Feature B": cols[idx_upper[1][i]],
                        "Correlation": round(float(vals[i]), 3),
                    }
                    for i in top_idx
                ])

            pairs_df = _top_corr_pairs(df, corr_cols, method_name, top_n, data_id=_data_fingerprint)
            n_total = len(corr_cols) * (len(corr_cols) - 1) // 2
            st.caption(f"Top {top_n} correlated pairs ({method_name}) out of {n_total:,} total")
            table(pairs_df)
    else:
        st.info("Need at least 2 numeric features for correlation analysis.")

    # -- Target Relationship Gallery -------------------------------------------
    if _has_target:
        st.subheader("Features vs Target")

        target_features = [f for f in numeric_features if f != target_col]
        # Sort by absolute correlation with target (cached)
        if target_features and task_type_final == "regression":
            @st.cache_data
            def _sort_by_target_corr(_df, _features, _target, data_id=None):
                corrs = _df[_features].corrwith(_df[_target]).abs().fillna(0)
                return corrs.sort_values(ascending=False).index.tolist()

            target_features = _sort_by_target_corr(df, target_features, target_col, data_id=_data_fingerprint)

        if regime.target_relationship_top_n > 0:
            target_features = target_features[:regime.target_relationship_top_n]
            st.caption(f"Showing top {len(target_features)} features by correlation with target. Use Feature Explorer for others.")

        t_page_size = 9
        t_total_pages = max(1, ceil(len(target_features) / t_page_size))
        if t_total_pages > 1:
            t_page = st.number_input(
                f"Page (1-{t_total_pages})", min_value=1, max_value=t_total_pages,
                value=1, key="target_gallery_page",
            )
        else:
            t_page = 1

        t_page_features = target_features[(t_page - 1) * t_page_size: t_page * t_page_size]

        for row_start in range(0, len(t_page_features), 3):
            row_cols = st.columns(3)
            for j, col_widget in enumerate(row_cols):
                idx = row_start + j
                if idx < len(t_page_features):
                    feat = t_page_features[idx]
                    with col_widget:
                        if task_type_final == "regression":
                            sample_df = df[[feat, target_col]].dropna()
                            if regime.needs_sampling and len(sample_df) > regime.sample_size:
                                sample_df = sample_df.sample(regime.sample_size, random_state=42)
                            fig = px.scatter(
                                sample_df, x=feat, y=target_col,
                                title=feat, trendline="lowess" if len(sample_df) > 20 else None,
                                opacity=0.4,
                            )
                        else:
                            fig = px.violin(
                                df, x=target_col, y=feat, title=feat,
                                box=True, points=False,
                            )
                        fig.update_layout(
                            template="plotly_white", height=250,
                            margin=dict(l=10, r=10, t=35, b=10),
                            showlegend=False,
                        )
                        st.plotly_chart(fig)

    # -- Feature Explorer (interactive scatter) --------------------------------
    st.subheader("Feature Explorer")
    st.caption("Pick any two features and color the points by a third — by default, the target.")

    if len(feature_cols) >= 2:
        fe_col1, fe_col2, fe_col3 = st.columns(3)
        with fe_col1:
            feat_x = st.selectbox("X axis", feature_cols, index=0, key="fe_x")
        with fe_col2:
            default_y = 1 if len(feature_cols) > 1 else 0
            feat_y = st.selectbox("Y axis", feature_cols, index=default_y, key="fe_y")
        with fe_col3:
            color_options = ["None"] + ([target_col] if _has_target else []) + feature_cols
            # Default to the target: seeing where the outcome sits in a feature
            # pair is the reason this control exists.
            color_by = st.selectbox(
                "Color by", color_options,
                index=1 if _has_target else 0, key="fe_color",
            )

        # Build column-by-column: df[[x, y]] duplicates the column when x == y,
        # which plotly express rejects.
        plot_df = pd.DataFrame({"__x": df[feat_x], "__y": df[feat_y]}, index=df.index)
        _color_active = color_by != "None" and color_by in df.columns
        if _color_active:
            plot_df["__color"] = df[color_by]

        plot_df = plot_df.dropna(subset=["__x", "__y"])
        _n_before_color_drop = len(plot_df)
        if _color_active:
            plot_df = plot_df.dropna(subset=["__color"])
        _n_full = len(plot_df)
        if regime.needs_sampling and len(plot_df) > regime.sample_size:
            plot_df = plot_df.sample(regime.sample_size, random_state=42)

        # Continuous vs categorical color. Decided by column semantics, not
        # dtype — a classification target coded 0/1 is numeric and would
        # otherwise render as two shades of one colorbar.
        _color_is_discrete = False
        if _color_active:
            from utils.column_utils import color_by_category
            _color_is_discrete = color_by_category(
                plot_df["__color"],
                is_classification_target=(
                    color_by == target_col and task_type_final == "classification"
                ),
            )
            if _color_is_discrete:
                plot_df["__color"] = plot_df["__color"].astype(str)

        _scatter_kwargs = dict(
            x="__x", y="__y", opacity=0.5,
            title=f"{feat_x} vs {feat_y}" + (f", colored by {color_by}" if _color_active else ""),
            labels={"__x": feat_x, "__y": feat_y, "__color": color_by if _color_active else ""},
        )
        if _color_active:
            _scatter_kwargs["color"] = "__color"
            if _color_is_discrete:
                _scatter_kwargs["color_discrete_sequence"] = px.colors.qualitative.Set2
            else:
                _scatter_kwargs["color_continuous_scale"] = "Viridis"

        fig_explorer = px.scatter(plot_df, **_scatter_kwargs)
        fig_explorer.update_layout(template="plotly_white", height=450)
        if _color_active:
            fig_explorer.update_layout(legend_title_text=color_by)
        st.plotly_chart(fig_explorer)

        # Say what was left out, rather than quietly plotting a subset.
        _notes = []
        if len(plot_df) < _n_full:
            _notes.append(f"showing a random {len(plot_df):,} of {_n_full:,} rows")
        if _color_active and _n_full < _n_before_color_drop:
            _notes.append(f"{_n_before_color_drop - _n_full:,} rows dropped for missing {color_by}")
        if _notes:
            st.caption(" · ".join(_notes).capitalize())

        # Show correlation for this pair
        if (
            feat_x != feat_y
            and pd.api.types.is_numeric_dtype(df[feat_x])
            and pd.api.types.is_numeric_dtype(df[feat_y])
        ):
            r = df[feat_x].corr(df[feat_y])
            if not np.isnan(r):
                st.caption(f"Pearson r = {r:.3f}")

    # -- Suggested Interactions ------------------------------------------------
    if _has_target and len(numeric_features) >= 4:
        with st.expander("💡 Suggested Interactions (auto-detected)", expanded=False):
            st.caption("Top feature pairs by mutual information with target. Click to explore.")

            @st.cache_data
            def _compute_interactions(_df, _features, _target, _task_type, max_pairs=5, data_id=None):
                """Compute top interaction pairs by MI gain."""
                from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
                sample = _df.sample(min(1000, len(_df)), random_state=42) if len(_df) > 1000 else _df
                feats = [f for f in _features if f in sample.columns][:30]
                X = sample[feats].fillna(sample[feats].median())
                y = sample[_target]
                valid = ~y.isnull()
                X, y = X[valid], y[valid]
                if len(X) < 20:
                    return []

                mi_func = mutual_info_regression if _task_type == "regression" else mutual_info_classif
                base_mi = mi_func(X, y, random_state=42)

                # Check top pairs for MI gain from interaction
                top_singles = np.argsort(-base_mi)[:10]
                results = []
                for i in range(len(top_singles)):
                    for j in range(i + 1, len(top_singles)):
                        fi, fj = feats[top_singles[i]], feats[top_singles[j]]
                        interaction = (X[fi] * X[fj]).values.reshape(-1, 1)
                        mi_inter = mi_func(interaction, y, random_state=42)[0]
                        mi_sum = base_mi[top_singles[i]] + base_mi[top_singles[j]]
                        gain = mi_inter - max(base_mi[top_singles[i]], base_mi[top_singles[j]])
                        if gain > 0:
                            results.append((fi, fj, float(gain)))
                results.sort(key=lambda x: x[2], reverse=True)
                return results[:max_pairs]

            try:
                interactions = _compute_interactions(df, numeric_features, target_col, task_type_final, data_id=_data_fingerprint)
                if interactions:
                    for a, b, gain in interactions:
                        st.markdown(f"- **{a} × {b}** (MI gain: {gain:.4f})")
                else:
                    st.caption("No strong interaction effects detected.")
            except Exception as e:
                st.caption(f"Interaction detection skipped: {str(e)[:80]}")

    # -- Cluster Structure (k-means) -------------------------------------------
    st.markdown("---")
    st.subheader("Cluster Structure")
    st.caption(
        "Does this cohort divide into subgroups, or is it one population? "
        "k-means always returns the number of clusters you ask for, so every result here "
        "is scored against the same pipeline run on shuffled data with no structure in it."
    )

    from ml import clustering as _clus

    _km_candidates = [f for f in feature_cols if f in df.columns]
    if len(_km_candidates) < 2 or len(df) < 30:
        st.info("Clustering needs at least 2 features and 30 rows.")
    else:
        _km_numeric = [f for f in _km_candidates if pd.api.types.is_numeric_dtype(df[f])]
        _km_default = (_km_numeric or _km_candidates)[: min(15, _clus.MAX_CLUSTER_FEATURES)]

        km_feats = st.multiselect(
            "Features to cluster on",
            options=_km_candidates,
            default=_km_default,
            key="eda_km_feats",
            help=(
                "Every column you include is a vote, and correlated columns vote as a bloc — "
                "eight correlated labs will outweigh one comorbidity flag. Choose deliberately."
            ),
        )

        kc1, kc2 = st.columns(2)
        with kc1:
            km_scaler = st.pills(
                "Scaling", ["Standard", "Robust"], default="Standard", key="eda_km_scaler",
                help=(
                    "Scaling is not optional for k-means: the objective is not invariant to "
                    "rescaling, so unscaled data clusters on your unit choices. Robust "
                    "(median/IQR) resists extreme values."
                ),
            )
        _km_has_cat = any(not pd.api.types.is_numeric_dtype(df[f]) for f in km_feats)
        with kc2:
            if _km_has_cat:
                km_cat_weight = st.slider(
                    "Categorical weight", 0.0, 2.0, 1.0, 0.25, key="eda_km_catw",
                    help=(
                        "At 1.0 each categorical variable carries about the weight of one "
                        "standardized numeric feature. Raise it to let categories drive the "
                        "split, lower it to make the split mostly numeric."
                    ),
                )
            else:
                km_cat_weight = 1.0
                st.caption("All selected features are numeric.")

        if len(km_feats) > _clus.MAX_CLUSTER_FEATURES:
            st.caption(
                f"Clustering on the first {_clus.MAX_CLUSTER_FEATURES} of {len(km_feats)} selected "
                "features — beyond that, Euclidean distance stops discriminating and every row "
                "looks equidistant from every other."
            )
            km_feats = km_feats[: _clus.MAX_CLUSTER_FEATURES]

        _km_config = (tuple(km_feats), (km_scaler or "Standard").lower(), float(km_cat_weight))

        if st.button("Explore cluster structure", key="eda_km_run", type="primary"):
            if len(km_feats) < 2:
                st.warning("Select at least 2 features.")
            else:
                st.session_state["eda_km_config"] = _km_config

        if st.session_state.get("eda_km_config") == _km_config and len(km_feats) >= 2:
            with st.spinner("Clustering and comparing against a no-structure baseline…"):
                _prep = _clus.prepare_cluster_matrix(
                    df, km_feats, scaler=_km_config[1],
                    categorical_weight=_km_config[2], data_id=_data_fingerprint,
                )

            if "error" in _prep:
                st.warning(_prep["error"])
            else:
                _X = _prep["X"]
                _prep_notes = [
                    f"{_prep['n_rows']:,} rows × {_prep['effective_p']} columns after encoding",
                    f"{_km_config[1]} scaling",
                ]
                if _prep["sampled"]:
                    _prep_notes.append(f"random subsample of {_prep['n_source_rows']:,} rows")
                if _prep["skew_transformed"]:
                    _prep_notes.append(
                        f"Yeo-Johnson applied to {len(_prep['skew_transformed'])} heavily skewed "
                        f"{'column' if len(_prep['skew_transformed']) == 1 else 'columns'}"
                    )
                if _prep["categorical_variance_share"] is not None:
                    _prep_notes.append(
                        f"categoricals carry {_prep['categorical_variance_share']:.0%} of the variance"
                    )
                st.caption(" · ".join(_prep_notes))
                if _prep["dropped"]:
                    st.caption("Dropped: " + ", ".join(_prep["dropped"][:6]) + (
                        f" (+{len(_prep['dropped']) - 6} more)" if len(_prep["dropped"]) > 6 else ""))

                _k_max = _clus.max_supported_k(_prep["n_rows"])
                _k_values = tuple(range(2, _k_max + 1))

                with st.spinner("Sweeping k against the shuffled baseline…"):
                    _sweep = _clus.sweep_k(_X, _k_values, _prep["variable_spans"])

                if "error" in _sweep:
                    st.warning(_sweep["error"])
                else:
                    st.plotly_chart(_clus.plot_k_sweep(_sweep), key="fig_eda_kmeans_sweep")

                    _sweep_df = pd.DataFrame([{
                        "k": r["k"],
                        "Silhouette": round(r["silhouette"], 3),
                        "Shuffled baseline": round(r["null_silhouette"], 3),
                        "Excess over baseline": round(r["excess"], 3),
                        "p (vs shuffled)": round(r["p_value"], 3),
                        "Calinski-Harabasz": round(r["calinski_harabasz"], 1),
                        "Davies-Bouldin": round(r["davies_bouldin"], 3),
                        "Smallest cluster": r["min_cluster_size"],
                    } for r in _sweep["table"]])
                    with st.expander("All k values, scored", expanded=False):
                        table(_sweep_df, key="eda_kmeans_sweep")
                        st.caption(
                            "Inertia is deliberately absent: it falls monotonically with k by "
                            "construction, so the \"elbow\" is an artifact of the plot's aspect "
                            "ratio rather than a method for choosing k."
                        )

                    _rec_k = _sweep["recommended_k"]
                    if _rec_k is None:
                        st.warning(
                            "**No evidence of cluster structure.** At every k tried, this data scored "
                            "no better than shuffled copies of itself — copies with identical column "
                            "distributions but every relationship between columns destroyed. "
                            "k-means will still hand you clusters below; treat them as a partition of "
                            "one population, not as discovered subgroups."
                        )
                        st.caption(
                            "One blind spot worth knowing: because the shuffled copies keep each "
                            "column's own distribution, this comparison cannot see structure that "
                            "lives inside a single column — a column with three distinct humps stays "
                            "three-humped after shuffling. The dominance check under **Cluster "
                            "profile** is what catches that case."
                        )
                    else:
                        _rec_row = next(r for r in _sweep["table"] if r["k"] == _rec_k)
                        st.success(
                            f"**Strongest structure at k = {_rec_k}** — silhouette "
                            f"{_rec_row['silhouette']:.3f} versus {_rec_row['null_silhouette']:.3f} on "
                            f"shuffled data (p = {_rec_row['p_value']:.2f}). That is "
                            f"{_clus.silhouette_reading(_rec_row['silhouette'])}."
                        )

                    _k_choice = st.selectbox(
                        "Number of clusters to inspect",
                        options=[r["k"] for r in _sweep["table"]],
                        index=[r["k"] for r in _sweep["table"]].index(_rec_k) if _rec_k else 0,
                        key="eda_km_k",
                    )

                    _fit = _clus.fit_clusters(_X, int(_k_choice))
                    if "error" in _fit:
                        st.warning(_fit["error"])
                    else:
                        _labels = _fit["labels"]
                        _stability = _clus.seed_stability(_X, int(_k_choice))

                        # -- Sizes and stability, above the profile ------------
                        _size_rows = []
                        for _cid, _size in enumerate(_fit["sizes"]):
                            _mask = _labels[_fit["silhouette_index"]] == _cid
                            _cluster_sil = float(np.nanmean(_fit["silhouette_samples"][_mask])) if _mask.any() else float("nan")
                            _size_rows.append({
                                "Cluster": _cid,
                                "n": int(_size),
                                "% of rows": f"{_size / max(1, _prep['n_rows']):.1%}",
                                "Mean silhouette": round(_cluster_sil, 3),
                                "Underpowered": "yes" if _size < max(20, 0.02 * _prep["n_rows"]) else "",
                            })
                        table(pd.DataFrame(_size_rows), key="eda_kmeans_sizes")

                        if "error" not in _stability:
                            _verdict_icon = {"stable": "✅", "suggestive": "⚠️", "unstable": "🚨"}[_stability["verdict"]]
                            st.caption(
                                f"{_verdict_icon} **Seed stability:** refitting under "
                                f"{_stability['n_seeds']} random starts gives mean adjusted Rand "
                                f"{_stability['mean_ari']:.2f} (worst pair {_stability['min_ari']:.2f}) — "
                                f"{_stability['verdict']}. A fixed seed buys reproducibility, not stability."
                            )

                        _km_tabs = st.tabs(["Projection", "Per-row silhouette", "Cluster profile", "Against the target"])

                        with _km_tabs[0]:
                            _proj = _clus.project_for_display(_X)
                            if "error" in _proj:
                                st.caption(_proj["error"])
                            else:
                                st.plotly_chart(
                                    _clus.plot_cluster_scatter(_proj["coords"], _labels, _proj["explained"]),
                                    key="fig_eda_kmeans_scatter",
                                )
                                _var2 = sum(_proj["explained"][:2])
                                if not np.isfinite(_var2):
                                    # A zero-variance matrix (every feature
                                    # categorical with the weight slider at 0)
                                    # makes PCA's ratio NaN. Left alone it
                                    # printed "nan%" and then, since nan < 0.40
                                    # is False, the reassuring branch.
                                    st.caption(
                                        "This matrix has no variance left to project — every column is "
                                        "constant once scaled. Add a feature, or raise the categorical weight."
                                    )
                                else:
                                    st.caption(
                                        f"These two components carry {_var2:.0%} of the variance. "
                                        + ("At that level the picture is a rough sketch — clusters that look "
                                           "merged here may be separated in directions this projection drops."
                                           if _var2 < 0.40 else
                                           "The projection was chosen without reference to the cluster labels, "
                                           "so separation you see here is not manufactured by the plot.")
                                    )

                        with _km_tabs[1]:
                            st.plotly_chart(
                                _clus.plot_silhouette_knife(
                                    _fit["silhouette_samples"],
                                    _labels[_fit["silhouette_index"]],
                                    _fit["silhouette"],
                                ),
                                key="fig_eda_kmeans_knife",
                            )
                            st.caption(
                                "Rows below zero sit closer to a different cluster than their own — "
                                "they are the rows that would move under a different seed. Clusters of "
                                "wildly different thickness, or clusters sitting mostly below the mean "
                                "line, mean this k is not describing the data well."
                            )

                        with _km_tabs[2]:
                            _profile = _clus.cluster_profile(
                                df, _labels, _prep["row_pos"], _prep["numeric_cols"],
                                data_id=_data_fingerprint,
                            )
                            if "error" in _profile:
                                st.caption(_profile["error"])
                            else:
                                st.plotly_chart(
                                    _clus.plot_cluster_profile(_profile["centroids_z"]),
                                    key="fig_eda_kmeans_profile",
                                )
                                st.caption(
                                    "Columns are ranked by how much of their variance the split explains. "
                                    "A centroid at +1.5 SD is only meaningful if the cluster is tighter "
                                    "than that — check the spread before naming a group."
                                )
                                table(
                                    _profile["centroids_z"].reset_index().rename(columns={"_cluster": "Cluster"}),
                                    key="eda_kmeans_profile",
                                )

                            # Score only the columns that actually reached the
                            # matrix — a column dropped during preparation
                            # cannot be what drove the split.
                            _dominance = _clus.feature_dominance(
                                df, _labels, _prep["row_pos"],
                                _prep["numeric_cols"] + _prep["categorical_cols"],
                                data_id=_data_fingerprint,
                            )
                            if "error" not in _dominance:
                                if _dominance["dominant"]:
                                    st.warning(
                                        f"**{_dominance['dominant']} alone explains this split** "
                                        f"({_dominance['dominant_score']:.0%} of that column is accounted "
                                        "for by the cluster label). This partition is a re-labeling of a "
                                        "column you already have, not a discovered subgroup."
                                    )
                                else:
                                    st.caption(
                                        "Share of each column accounted for by the split: "
                                        + ", ".join(
                                            f"{r['Feature']} {r['Explained by split']:.0%}"
                                            for r in _dominance["table"][:3]
                                        )
                                        + " — no single column explains it on its own."
                                    )

                        with _km_tabs[3]:
                            if not _has_target:
                                st.caption("No target selected, so there is nothing held out to test against.")
                            else:
                                _assoc = _clus.target_association(
                                    df, _labels, _prep["row_pos"], target_col,
                                    task_type_final or "regression", data_id=_data_fingerprint,
                                )
                                if "error" in _assoc:
                                    st.caption(_assoc["error"])
                                else:
                                    st.markdown(
                                        f"**{target_col} was held out of the clustering**, so this is the one "
                                        "comparison here that is not circular. Testing the clusters on the "
                                        "features they were built from would return p ≈ 0 by construction."
                                    )
                                    table(_assoc["table"].round(3), key="eda_kmeans_target")
                                    _eff = _assoc["effect"]
                                    # The threshold comes from the result, not
                                    # from here: Cramer's V and eta-squared are
                                    # different scales, and 0.06 on V is below
                                    # "small" — it called pure noise worth
                                    # following up on most random draws.
                                    _eff_thr = _assoc.get("effect_threshold", 0.06)
                                    _eff_real = (
                                        np.isfinite(_eff) and _eff >= _eff_thr
                                        and _assoc["p_value"] < 0.05
                                    )
                                    st.caption(
                                        f"{_assoc['effect_name']} = {_eff:.3f}, p = {_assoc['p_value']:.2g}. "
                                        + ("The clusters differ on the target by an amount worth following up."
                                           if _eff_real else
                                           "The effect is small — the clusters barely distinguish the outcome.")
                                    )
                                    if _assoc["kind"] == "regression":
                                        _tgt_plot_df = _assoc["values"].copy()
                                        _tgt_plot_df["cluster"] = _tgt_plot_df["cluster"].astype(str)
                                        _fig_tgt = px.box(
                                            _tgt_plot_df, x="cluster", y="target", color="cluster",
                                            color_discrete_sequence=px.colors.qualitative.Set2,
                                            labels={"cluster": "Cluster", "target": target_col},
                                        )
                                        _fig_tgt.update_layout(
                                            template="plotly_white", height=380, showlegend=False,
                                            title=f"{target_col} by cluster",
                                        )
                                        st.plotly_chart(_fig_tgt, key="fig_eda_kmeans_target")

                        # -- Record the run, and the finding if there is one ----
                        # Streamlit re-executes the whole script on any widget
                        # touch anywhere on the page, so this block re-renders
                        # constantly. Log once per distinct (config, k): the
                        # ledger de-duplicates by id, but methodology_log does
                        # not, and it is what the manuscript methods read.
                        _km_logged = (_km_config, int(_k_choice))
                        if st.session_state.get("eda_km_logged") != _km_logged:
                            st.session_state["eda_km_logged"] = _km_logged
                            log_methodology(step="EDA", action="Ran k-means cluster exploration", details={
                                "k": int(_k_choice),
                                "n_features": len(km_feats),
                                "n_rows_clustered": _prep["n_rows"],
                                "scaling": _km_config[1],
                                "silhouette": round(_fit["silhouette"], 3),
                                "shuffled_baseline": round(
                                    next(r["null_silhouette"] for r in _sweep["table"] if r["k"] == _k_choice), 3
                                ),
                                "seed_stability_ari": round(_stability.get("mean_ari", float("nan")), 3),
                            })

                        if (
                            _rec_k == _k_choice
                            and _stability.get("verdict") == "stable"
                            and not _dominance.get("dominant")
                        ):
                            ledger.upsert(Insight(
                                id="eda_kmeans_structure",
                                source_page="02_EDA", category="topology", severity="opportunity",
                                finding=(
                                    f"{_k_choice} stable subgroups detected across {len(km_feats)} features "
                                    f"(silhouette {_fit['silhouette']:.2f} vs "
                                    f"{next(r['null_silhouette'] for r in _sweep['table'] if r['k'] == _k_choice):.2f} "
                                    f"on shuffled data)"
                                ),
                                implication=(
                                    "A single global model averages over these subgroups; linear models "
                                    "in particular fit one slope where the data may hold several"
                                ),
                                recommended_action=(
                                    "Consider k-means cluster features in Preprocess, or report "
                                    "performance stratified by subgroup"
                                ),
                                relevant_pages=["05_Preprocess", "06_Train_and_Compare"],
                                model_scope=[MODEL_FAMILY_LINEAR],
                                metadata={
                                    "k": int(_k_choice),
                                    "silhouette": round(_fit["silhouette"], 3),
                                    "n_features": len(km_feats),
                                    "seed_stability_ari": round(_stability["mean_ari"], 3),
                                },
                                manuscript_text=(
                                    f"k-means clustering across {len(km_feats)} features identified "
                                    f"{_k_choice} subgroups whose separation exceeded that obtained on "
                                    f"permuted data with matched marginal distributions"
                                ),
                                resolved=True, resolved_by="Positive signal — no action needed",
                                resolved_on_page="02_EDA", auto_generated=True,
                            ))

                        try:
                            from utils.llm_ui import (
                                build_llm_context, render_interpretation_with_llm_button,
                                gather_session_context,
                            )
                            _bg_km = gather_session_context()
                            _km_summary = (
                                f"k={_k_choice}; silhouette={_fit['silhouette']:.3f}; "
                                f"shuffled_baseline={next(r['null_silhouette'] for r in _sweep['table'] if r['k'] == _k_choice):.3f}; "
                                f"cluster_sizes={_fit['sizes']}; "
                                f"seed_stability_ari={_stability.get('mean_ari', float('nan')):.3f}; "
                                f"features={len(km_feats)}"
                            )
                            _ctx_km = build_llm_context(
                                "kmeans_clusters", _km_summary,
                                where="EDA cluster structure",
                                sample_size=_bg_km.pop("sample_size", _prep["n_rows"]),
                                task_type=_bg_km.pop("task_type", task_type_final),
                                feature_names=_bg_km.pop("feature_names", km_feats),
                                **_bg_km,
                            )
                            render_interpretation_with_llm_button(
                                _ctx_km, key="llm_eda_kmeans",
                                result_session_key="llm_result_eda_kmeans",
                                plot_type="kmeans_clusters",
                            )
                        except Exception:
                            pass  # Interpretation is optional; it should never break the page.


    # ============================================================================
# SECTION 4: MACRO SHAPE (≥16 features only)
# ============================================================================

if regime.show_macro_shape and numeric_features:
    st.markdown("---")
    st.header("Macro Shape")
    st.caption("How your data looks in reduced dimensions. Each view reveals something the others hide.")

    from ml.macro_shape import (
        compute_pca as _compute_pca, plot_scree, plot_pca_biplot,
        compute_umap as _compute_umap, plot_umap,
        compute_persistence as _compute_persistence, plot_persistence_diagram, plot_persistence_barcode,
        compute_mapper as _compute_mapper, plot_mapper,
    )

    df_numeric = df[numeric_features].dropna()

    # Color values for embeddings
    if _has_target and target_col in df.columns:
        color_vals = df.loc[df_numeric.index, target_col].values
        color_label = target_col
    else:
        color_vals = None
        color_label = ""

    # Available tiers based on regime
    available_views = []
    tier_labels = {
        "pca": "PCA Biplot",
        "umap": "UMAP",
        "persistence": "Persistence Diagram",
        "mapper": "Mapper Graph",
    }
    for tier in regime.macro_shape_tiers:
        available_views.append(tier_labels[tier])

    # Variance profile (always first)
    st.subheader("Variance Profile")
    # Cached here rather than in ml/macro_shape.py. The engine's caches keyed on
    # nothing — their only argument is underscore-prefixed, which Streamlit does
    # not hash, so one dataset's PCA was served to every later dataset and, in
    # the shared deployment, to every later user (T0-LIVE-001). The host is what
    # knows when the dataset changed, so the host is what caches.
    @st.cache_data(show_spinner=False)
    def _macro_cached(_kind: str, _df: pd.DataFrame, fingerprint):
        return {"pca": _compute_pca, "umap": _compute_umap,
                "persistence": _compute_persistence, "mapper": _compute_mapper}[_kind](_df)

    def compute_pca(d):          return _macro_cached("pca", d, _macro_fp(d))
    def compute_umap(d):         return _macro_cached("umap", d, _macro_fp(d))
    def compute_persistence(d):  return _macro_cached("persistence", d, _macro_fp(d))
    def compute_mapper(d):       return _macro_cached("mapper", d, _macro_fp(d))

    pca_result = compute_pca(df_numeric)
    if "error" not in pca_result:
        fig_scree = plot_scree(pca_result)
        st.plotly_chart(fig_scree)
        n_90 = pca_result["n_components_90"]
        total_var = pca_result["total_variance_explained"]
        n_used = len(pca_result["feature_names"])
        cap_note = f" (computed on {n_used} of {len(numeric_features)})" if n_used < len(numeric_features) else ""
        st.caption(
            f"**{n_90} components** explain 90% of variance across {len(numeric_features)} features{cap_note}. "
            f"Top {min(len(pca_result['explained_variance_ratio']), 5)} components capture {total_var:.1%} total."
        )

        # Auto-insight: effective dimensionality
        if n_90 <= 3 and len(numeric_features) > 10:
            ledger.upsert(Insight(
                id="eda_low_dimensionality",
                source_page="02_EDA", category="topology", severity="opportunity",
                finding=f"Data is effectively {n_90}-dimensional despite {len(numeric_features)} features",
                implication="Dimensionality reduction (PCA) could simplify models with minimal information loss",
                recommended_action="Consider PCA preprocessing or feature selection",
                relevant_pages=["04_Feature_Selection", "05_Preprocess"],
                model_scope=ISSUE_MODEL_RELEVANCE["high_dimensionality"],  # linear, distance, margin
                metadata={"n_components_90": n_90, "n_features": len(numeric_features)},
                resolved=True, resolved_by="Positive signal — no action needed",
                resolved_on_page="02_EDA", auto_generated=True,
            ))
    else:
        st.warning(pca_result["error"])

    # Embedding views
    if available_views:
        selected_view = st.pills("Embedding", available_views, default=available_views[0], key="macro_view")

        if selected_view == "PCA Biplot" and "error" not in pca_result:
            fig_biplot = plot_pca_biplot(pca_result, color_vals, color_label)
            st.plotly_chart(fig_biplot)
            st.caption("Arrows show feature loadings — longer arrows have more influence on this projection. "
                      "This view preserves global variance structure but hides non-linear patterns.")

        elif selected_view == "UMAP":
            with st.spinner("Computing UMAP embedding..."):
                umap_result = compute_umap(df_numeric)
            if "error" not in umap_result:
                # Align color values to sampled indices
                if color_vals is not None:
                    umap_colors = df.loc[
                        df_numeric.index[df_numeric.index.isin(
                            pd.Index(umap_result["sample_indices"])
                        )],
                        target_col,
                    ].values[:len(umap_result["embedding"])]
                else:
                    umap_colors = None
                fig_umap = plot_umap(umap_result, umap_colors, color_label)
                st.plotly_chart(fig_umap)
                st.caption("UMAP preserves local neighborhood structure — nearby points are genuinely similar. "
                          "Cluster sizes and inter-cluster distances are NOT meaningful.")
            else:
                st.warning(umap_result["error"])

        elif selected_view == "Persistence Diagram":
            with st.spinner("Computing persistent homology (this may take a moment)..."):
                tda_result = compute_persistence(df_numeric)
            if "error" not in tda_result:
                diag_tab, barcode_tab = st.tabs(["Diagram", "Barcode"])
                with diag_tab:
                    fig_diag = plot_persistence_diagram(tda_result)
                    st.plotly_chart(fig_diag)
                with barcode_tab:
                    fig_barcode = plot_persistence_barcode(tda_result)
                    st.plotly_chart(fig_barcode)

                # Summary
                for dim, info in tda_result["features_by_dim"].items():
                    dim_name = {0: "H₀ (connected components)", 1: "H₁ (loops)", 2: "H₂ (voids)"}.get(dim, f"H{dim}")
                    st.caption(
                        f"**{dim_name}:** {info['n_features']} features, "
                        f"max persistence = {info['max_persistence']:.3f}"
                    )

                st.caption("Points far from the diagonal are topologically significant (persist across scales). "
                          "H₀ counts clusters; H₁ counts loops/holes in the data manifold.")

                # Auto-insight for notable topology
                h1_info = tda_result["features_by_dim"].get(1, {})
                if h1_info.get("n_features", 0) > 0 and h1_info.get("max_persistence", 0) > 0.5:
                    ledger.upsert(Insight(
                        id="eda_tda_loops",
                        source_page="02_EDA", category="topology", severity="opportunity",
                        finding=f"Persistent loops detected in data manifold (H₁ max persistence = {h1_info['max_persistence']:.3f})",
                        implication="Data has non-trivial topological structure that linear models cannot capture",
                        recommended_action="Consider TDA features in Feature Engineering",
                        relevant_pages=["03_Feature_Engineering"],
                        resolved=True, resolved_by="Positive signal — no action needed",
                        resolved_on_page="02_EDA", auto_generated=True,
                    ))
            else:
                st.warning(tda_result["error"])

        elif selected_view == "Mapper Graph":
            with st.spinner("Computing Mapper graph..."):
                mapper_result = compute_mapper(df_numeric)
            if "error" not in mapper_result:
                mapper_colors = color_vals[:len(df_numeric)] if color_vals is not None else None
                fig_mapper = plot_mapper(mapper_result, mapper_colors, color_label)
                st.plotly_chart(fig_mapper)
                st.caption(
                    f"Mapper approximates the data manifold as a graph ({mapper_result['n_nodes']} nodes, "
                    f"{mapper_result['n_edges']} edges). Branching reveals subpopulations; "
                    "loops reveal circular structure. Node size = sample count."
                )
            else:
                st.warning(mapper_result["error"])


# ============================================================================
# SECTION 5: COACHING LAYER
# ============================================================================

st.markdown("---")
st.header("What Your Data Is Telling You")

summary = ledger.summary()

# Compact severity bar
sc1, sc2, sc3, sc4 = st.columns(4)
sc1.metric("🚨 Blockers", summary["blockers"])
sc2.metric("⚠️ Warnings", summary["warnings"])
sc3.metric("ℹ️ Info", summary["info"])
sc4.metric("💡 Opportunities", summary["opportunities"])

unresolved = ledger.get_unresolved()

if not unresolved:
    st.success("No issues detected. Your data looks ready for modeling.")
else:
    # ------------------------------------------------------------------
    # Synthesized view: group insights by category, then render each group
    # as a single narrative block with expandable details
    # ------------------------------------------------------------------
    from collections import defaultdict as _dd

    # Category display config
    _cat_meta = {
        "relationship": ("🔗 Relationships", "How features relate to each other and the target"),
        "distribution": ("📊 Distributions", "Shape and spread of your features"),
        "data_quality": ("🧹 Data Quality", "Missing values, plausibility, and integrity"),
        "sufficiency": ("📏 Sample Size", "Whether you have enough data for reliable modeling"),
        "topology": ("🌐 Data Geometry", "High-dimensional structure of your data"),
        "methodology": ("📐 Methodology", "Modeling strategy considerations"),
    }

    # Group by category
    _groups = _dd(list)
    for ins in unresolved:
        _groups[ins.category].append(ins)

    # Render each group
    for cat in ["relationship", "distribution", "data_quality", "sufficiency", "topology", "methodology"]:
        group = _groups.get(cat, [])
        if not group:
            continue

        cat_title, cat_desc = _cat_meta.get(cat, (cat.title(), ""))
        blockers_in_group = [i for i in group if i.severity == "blocker"]
        warnings_in_group = [i for i in group if i.severity == "warning"]
        info_in_group = [i for i in group if i.severity == "info"]
        opps_in_group = [i for i in group if i.severity == "opportunity"]

        # Severity indicator for the group header
        if blockers_in_group:
            group_icon = "🚨"
        elif warnings_in_group:
            group_icon = "⚠️"
        elif opps_in_group:
            group_icon = "💡"
        else:
            group_icon = "ℹ️"

        with st.container(border=True):
            st.markdown(f"### {group_icon} {cat_title}")

            # Synthesis: one-sentence summary of the group
            if cat == "relationship":
                corr_insights = [i for i in group if "corr" in i.id or "collinear" in i.id.lower()]
                leakage_insights = [i for i in group if "leakage" in i.id]
                other_rel = [i for i in group if i not in corr_insights and i not in leakage_insights]

                if leakage_insights:
                    st.error(f"**Target leakage detected** in {len(leakage_insights)} feature(s). This must be resolved.")
                    for ins in leakage_insights:
                        st.caption(f"  🚨 {ins.finding}")

                if corr_insights:
                    total_affected = len(set(f for i in corr_insights for f in i.affected_features))
                    st.markdown(f"**{len(corr_insights)} collinearity cluster(s)** affecting {total_affected} features total.")
                    for ins in corr_insights:
                        st.caption(f"  ⚠️ {ins.finding}")
                    st.caption(f"→ {corr_insights[0].recommended_action}")

                for ins in other_rel:
                    _ic = {"blocker": "🚨", "warning": "⚠️", "info": "ℹ️", "opportunity": "💡"}.get(ins.severity, "ℹ️")
                    st.markdown(f"{_ic} {ins.finding}")
                    st.caption(f"→ {ins.implication}")

            elif cat == "distribution":
                # Separate problems from opportunities
                problems = warnings_in_group + info_in_group
                if problems:
                    for ins in problems:
                        st.markdown(f"{'⚠️' if ins.severity == 'warning' else 'ℹ️'} {ins.finding}")
                        st.caption(f"→ {ins.implication}")
                if opps_in_group:
                    for ins in opps_in_group:
                        st.markdown(f"💡 {ins.finding}")
                        st.caption(f"→ {ins.implication}")

            elif cat == "data_quality":
                for ins in group:
                    _ic = "⚠️" if ins.severity == "warning" else "ℹ️"
                    st.markdown(f"{_ic} {ins.finding}")
                    st.caption(f"→ {ins.implication}")

            else:
                # Generic rendering for other categories
                for ins in group:
                    _ic = {"blocker": "🚨", "warning": "⚠️", "info": "ℹ️", "opportunity": "💡"}.get(ins.severity, "ℹ️")
                    st.markdown(f"{_ic} {ins.finding}")
                    st.caption(f"→ {ins.implication}")

# Reviewer risks
# Reviewer risks and next steps removed — coaching layer on each downstream page handles this


# ============================================================================
# SECTION 6: CLASSICAL DIAGNOSTICS
# ============================================================================
#
# These four were the survivors of the old "Deep Dive Diagnostics" tab strip.
# The other eleven analyses it offered had been overtaken by the always-on
# sections above — the target histogram, class bar, missingness bar,
# correlation matrix, outlier heatmap, feature-vs-target gallery and
# interaction detector all moved into Sections 1-3, and the InsightLedger now
# states the verbal conclusions (skew, imbalance, leakage, sufficiency,
# scaling) those analyses used to restate as findings.
#
# What is left is the classical-statistics layer nothing else in the app
# implements: NHANES unit and reference-range checking, a Q-Q plot and
# Shapiro-Wilk on residuals, variance inflation, and leverage / Cook's D.
# Each is now its own section rather than a button inside a tab, because a
# diagnostic nobody can find is a diagnostic nobody runs.

from ml.physiology_reference import load_reference_bundle, match_variable_key
_ref_bundle = load_reference_bundle()
_nhanes_ref = _ref_bundle.get("nhanes", {}) if _ref_bundle else {}
# Scan every column, not just the features: a dataset whose only NHANES-matching
# column is the target (predicting HbA1c, say) still deserves the unit check.
_physio_scan_cols = list(dict.fromkeys(list(feature_cols) + ([target_col] if _has_target else [])))
_has_physio_matches = any(match_variable_key(f, _nhanes_ref) for f in _physio_scan_cols) if _nhanes_ref else False

if "eda_results" not in st.session_state:
    st.session_state.eda_results = {}


# Map recommendation panel action IDs to the ledger insight IDs they speak to.
# `AUDIT-032`: the map and the recorder both live in `ml/eda_actions.py` now —
# a Streamlit page is not importable, so the only test of this wiring had to
# keep a hand-copy of both, and the copy is where the pre-fix behavior was
# still asserted. The comment there carries the before/after.
from ml.eda_actions import (
    _ACTION_TO_INSIGHT_MAP,
    diagnostic_disclosure,
    record_diagnostic_on_insights,
)


def _resolve_insights_from_eda_result(action_id: str, result: dict, title: str) -> str:
    """Record a completed recommended analysis against the insights it speaks to.

    `AUDIT-032`. BEFORE this resolved every matching insight, so pressing **Run
    Leakage Detection** — which re-reads `signals.leakage_candidate_cols` and
    removes nothing — moved a `blocker` into the report's *"N were addressed
    during the modeling workflow"* count and dropped its manuscript caveat.
    AFTER it attaches the findings and leaves the observation open, and returns
    the sentence the page shows so the user is told which of the two happened.

    Returns "" when nothing was recorded. Never raises: the workflow must not
    break on a bookkeeping step.
    """
    try:
        touched = record_diagnostic_on_insights(ledger, action_id, result, title)
        if not _ACTION_TO_INSIGHT_MAP.get(action_id):
            return ""
        return diagnostic_disclosure(title, len(touched))
    except Exception:
        return ""  # Never break the workflow


ACTION_NEXT_STEPS = {
    'multicollinearity_vif': '→ **Next:** Go to Feature Selection to use LASSO/RFE to resolve collinearity, or apply Ridge/ElasticNet regularization in training.',
    'influence_diagnostics': '→ **Next:** Review flagged high-influence points. Consider target trimming on the Train page, or use robust regression (Huber).',
    'normality_residuals': '→ **Next:** If residuals depart from normality, prefer bootstrap confidence intervals over parametric ones, or use a robust loss on the Train page.',
    # `sibling-of: AUDIT-025` · found by the §08 check-5 sweep ("what would the
    # same lens find one surface over?") while AUDIT-025 was blocked on a file
    # this chunk does not own. Same defect: a page names a capability on a page
    # that does not have it. BEFORE this read "Apply target trimming or filter
    # rows in Upload & Audit." — `pages/01_Upload_and_Audit.py` contains zero
    # occurrences of "trim" and zero of "plausib", so NEITHER control is there.
    # Target trimming is `pages/06_Train_and_Compare.py:296` ("Enable target
    # trimming before split") and plausibility filtering is
    # `pages/05_Preprocess.py:744` — inside the per-model block that :580 skips
    # while Smart Defaults is selected, hence the mode is named. The app's own
    # `influence_diagnostics` entry above already said "on the Train page",
    # which is how the wrong destination here was visible at all.
    #
    # MERGE NOTE: main's restructure delisted leakage_scan, target_profile,
    # feature_scaling_check and linearity_scatter from this page, so their
    # entries are dropped with them; the AUDIT-025 sibling correction to the
    # plausibility_check line is kept, since that section survived.
    'plausibility_check': '→ **Next:** Review flagged implausible values. Neither control is on Upload & Audit: target trimming is on Train & Compare, applied before the split, and plausibility filtering is on Preprocess under Advanced (full control).',
}


def _run_and_show(action_id: str, title: str, run_action: str):
    """Run a diagnostic and display its figures, narrative, and next step."""
    from utils.llm_ui import build_llm_context, build_eda_full_results_context, render_interpretation_with_llm_button, gather_session_context

    # MERGE NOTE: `tab_key` went with main's tab strip, so the key prefix is
    # just the action id now. The lockbox scoping below is TurboTab's and is
    # load-bearing: `_action_df` is what the action is actually run on.
    _action_df = _frame_for_action(run_action)
    if _lockbox_scoped and run_action in _TRAIN_ONLY_ACTIONS:
        st.caption(
            f"Fits models, so it sees n={len(_action_df)} training rows; "
            f"{_TRAIN_SCOPE_CAPTION}"
        )
    if st.button(f"Run {title}", key=f"run_{action_id}", type="primary"):
        try:
            action_func = getattr(eda_actions, run_action, None)
            if action_func:
                with st.spinner(f"Running {title}..."):
                    result = action_func(_action_df, target_col, feature_cols, signals, st.session_state)
                    st.session_state.eda_results[action_id] = result
                    log_methodology(step="EDA", action=f"Ran {title}", details={"analysis": run_action})
                    # MERGE NOTE: three things happen here and all three are
                    # fixes. (1) main's upsert — actions now RETURN their
                    # insights (utils.storyline.add_insight is gone), and
                    # without this the plausibility finding reaches nothing.
                    # (2) main's VIF resolver — VIF is the answer to the
                    # pairwise-correlation clusters the page itself raised,
                    # and nothing else in the app closes eda_corr_cluster_*.
                    # (3) TurboTab's AUDIT-032 recorder — a diagnostic is
                    # evidence, not an action, so it annotates and discloses
                    # rather than resolving. (2) runs BEFORE (3) so the
                    # disclosure counts only what is still open; the recorder
                    # skips resolved insights.
                    for _insight in result.get("insights", []) or []:
                        try:
                            ledger.upsert(_insight)
                        except Exception:
                            pass  # A malformed insight must not lose the analysis.
                    # Running VIF answers the collinearity clusters the page
                    # detected from pairwise correlation, so it closes them.
                    # Nothing else in the app resolves eda_corr_cluster_*, and
                    # left open they reach the manuscript as a limitation the
                    # user has in fact already investigated.
                    if action_id == "multicollinearity_vif":
                        _vif_summary = (result.get("findings") or [title])[0]
                        for _ins in list(ledger.insights):
                            if _ins.resolved or not _ins.id.startswith("eda_corr_cluster_"):
                                continue
                            try:
                                ledger.resolve(
                                    _ins.id,
                                    resolved_by=f"{title}: {_vif_summary}",
                                    resolved_on_page="02_EDA",
                                    resolution_details={
                                        "action_type": "diagnostic_analysis",
                                        "method": action_id,
                                        "stats": result.get("stats", {}),
                                    },
                                )
                            except Exception:
                                pass
                    # Stored rather than written straight out: `st.rerun()`
                    # below discards everything this block renders, which is
                    # `AGENT_ONBOARD.md` §07 trap 6 in its purest form — the
                    # server composes the sentence and nobody ever sees it.
                    _disclosure = _resolve_insights_from_eda_result(action_id, result, title)
                    if _disclosure:
                        st.session_state.setdefault("eda_diagnostic_disclosure", {})[action_id] = _disclosure
                    try:
                        from utils.workflow_provenance import get_provenance
                        get_provenance().record_eda_analysis(title)
                    except Exception:
                        pass  # Provenance recording should never break the workflow
                    st.rerun()
            else:
                st.error(f"Action '{run_action}' not found")
        except Exception as e:
            st.error(f"Error: {str(e)}")

    if action_id in st.session_state.eda_results:
        result = st.session_state.eda_results[action_id]
        _disclosure = (st.session_state.get("eda_diagnostic_disclosure") or {}).get(action_id)
        if _disclosure:
            st.info(_disclosure, icon="🔎")
        for w in result.get("warnings", []):
            st.warning(w)

        ACTION_NARRATIVE = {
            "influence_diagnostics": narrative_eda_influence,
            "normality_residuals": narrative_eda_normality,
            "multicollinearity_vif": narrative_eda_multicollinearity,
        }
        findings = result.get("findings", [])[:2]
        stats = result.get("stats", {})
        nar_fn = ACTION_NARRATIVE.get(run_action)
        interp = nar_fn(stats, findings) if nar_fn else ("; ".join(findings) if findings else None)

        for idx, (fig_type, fig_data) in enumerate(result.get("figures", [])):
            if fig_type == "plotly":
                st.plotly_chart(fig_data, key=f"fig_{action_id}_{idx}")
            elif fig_type == "table":
                table(fig_data, key=f"tbl_{action_id}_{idx}")

        if interp:
            st.markdown(f"**Summary:** {interp}")
            stats_summary = build_eda_full_results_context(result, action_id)
            bg = gather_session_context()
            ctx = build_llm_context(
                action_id, stats_summary, existing=interp,
                feature_names=bg.pop("feature_names", feature_cols),
                sample_size=bg.pop("sample_size", len(df)),
                task_type=bg.pop("task_type", task_type_final),
                **bg,
            )
            render_interpretation_with_llm_button(
                ctx, key=f"llm_{action_id}",
                result_session_key=f"llm_result_{action_id}",
                plot_type=action_id,
            )
        next_step = ACTION_NEXT_STEPS.get(action_id)
        if next_step:
            st.markdown(next_step)


# -- Physiologic Plausibility ------------------------------------------------

st.markdown("---")
st.header("Physiologic Plausibility")
st.caption(
    "Reads each column against NHANES reference ranges and clinical guideline bands, "
    "after inferring its units. This is the check that catches a glucose column recorded "
    "in mmol/L being read as mg/dL — a mix-up that produces no statistical outliers at all."
)
if _has_physio_matches:
    _run_and_show("plausibility_check", "Physiologic Plausibility", "plausibility_check")
else:
    st.caption(
        "No columns matched a known biomedical variable, so there are no reference "
        "ranges to check against."
    )


# -- Residual Normality ------------------------------------------------------

st.markdown("---")
st.header("Residual Normality")
st.caption(
    "Q-Q plot and Shapiro-Wilk on the residuals of a quick OLS fit. Residual normality, "
    "not the normality of any single column, is what governs whether parametric confidence "
    "intervals and p-values from a linear model can be trusted."
)
if task_type_final == 'regression':
    st.info(
        "💡 This fits a **simple OLS regression as a diagnostic proxy** — standard EDA "
        "practice. The OLS here is an instrument, not your final model."
    )
    _run_and_show("normality_residuals", "Normality of Residuals", "normality_residuals")
else:
    st.caption("Regression only — residual normality is not a classification assumption.")


# -- Multicollinearity (VIF) -------------------------------------------------

st.markdown("---")
st.header("Multicollinearity (VIF)")
st.caption(
    "Variance inflation factor per feature. This is not the same question the correlation "
    "matrix above answers: a feature that is a linear combination of several others can be "
    "invisible to every pairwise correlation and still make a linear fit unstable."
)
_collinear_pairs = signals.collinearity_summary.get('high_corr_pairs') or []
if _collinear_pairs:
    st.warning(
        f"{len(_collinear_pairs)} feature pairs are already correlated above the flagging "
        "threshold. VIF will show whether the problem is wider than those pairs."
    )
_run_and_show("multicollinearity_vif", "VIF (Multicollinearity)", "multicollinearity_vif")


# -- Influence Diagnostics ---------------------------------------------------

st.markdown("---")
st.header("Influence Diagnostics")
st.caption(
    "Leverage and Cook's distance per observation. Univariate outlier flags find rows with an "
    "extreme value in some column; these find rows that move the fitted surface, which is a "
    "different set — a point can have high leverage with no extreme value anywhere."
)
if task_type_final == 'regression':
    if len(df) > 20_000:
        st.info(
            f"Skipped: this builds an n×n hat matrix and would need roughly "
            f"{(len(df) ** 2 * 8) / 1e9:.0f} GB at {len(df):,} rows. Run it on a subset if you need it."
        )
    else:
        _run_and_show("influence_diagnostics", "Influence Diagnostics", "influence_diagnostics")
else:
    st.caption("Regression only — leverage and Cook's distance are defined for a least-squares fit.")

# ============================================================================
# SECTION 7: TABLE 1 — PUBLICATION SUMMARY
# ============================================================================

st.markdown("---")
with st.expander("📄 Table 1 — Publication Summary", expanded=False):
    st.caption("Academic/publication-oriented cohort summary with CSV and LaTeX export.")

    from ml.table_one import Table1Config, generate_table1, table1_to_csv, table1_to_latex
    from data_processor import get_categorical_columns

    all_numeric = get_numeric_columns(df)
    all_categorical = get_categorical_columns(df)

    possible_groups = [c for c in all_categorical if c != target_col and df[c].nunique() <= 10]
    grouping_var = st.selectbox(
        "Stratify by",
        options=["None"] + possible_groups,
        index=0,
        key="table1_group",
    )

    # Clear stale widget state
    for _wk in ("table1_continuous", "table1_categorical", "table1_group"):
        _old = st.session_state.get(_wk)
        if isinstance(_old, list):
            st.session_state[_wk] = [v for v in _old if v in df.columns]
        elif isinstance(_old, str) and _old not in ("None",) and _old not in df.columns:
            st.session_state.pop(_wk, None)

    _t1_cont_options = [c for c in all_numeric if c != target_col]
    _t1_cont_default = [c for c in feature_cols if c in all_numeric and c in _t1_cont_options][:10]
    t1_continuous = st.multiselect("Continuous variables", options=_t1_cont_options, default=_t1_cont_default, key="table1_continuous")

    _t1_cat_options = [c for c in all_categorical if c != target_col and c != grouping_var]
    _t1_cat_default = [c for c in feature_cols if c in all_categorical and c in _t1_cat_options][:5]
    t1_categorical = st.multiselect("Categorical variables", options=_t1_cat_options, default=_t1_cat_default, key="table1_categorical")

    ct1, ct2, ct3 = st.columns(3)
    with ct1:
        show_pvalues = st.checkbox("Show p-values", value=True, key="table1_pval")
    with ct2:
        show_smd = st.checkbox("Show SMD", value=False, key="table1_smd")
    with ct3:
        show_missing = st.checkbox("Show missing counts", value=True, key="table1_miss")

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
        log_methodology(step="EDA", action="Generated exploratory Table 1", details={
            "grouping_var": grouping_var if grouping_var != "None" else None,
            "n_continuous": len(t1_continuous),
            "n_categorical": len(t1_categorical),
        })
        try:
            from utils.workflow_provenance import get_provenance
            get_provenance().record_table1()
        except Exception:
            pass  # Provenance recording should never break the workflow

    if st.session_state.get("table1_df") is not None:
        table1_df = st.session_state["table1_df"]
        table(table1_df)

        table1_metadata = st.session_state.get("table1_metadata", {})
        if table1_metadata.get("tests_used"):
            st.caption("**Tests used:** " + ", ".join(
                f"{var}: {test}" for var, test in table1_metadata["tests_used"].items()
            ))

        ex1, ex2 = st.columns(2)
        with ex1:
            csv_data = table1_to_csv(table1_df)
            st.download_button("📥 CSV", csv_data, "table1.csv", "text/csv", key="dl_table1_csv")
        with ex2:
            latex_data = table1_to_latex(table1_df)
            st.download_button("📥 LaTeX", latex_data, "table1.tex", "text/plain", key="dl_table1_latex")


# ============================================================================
# SYNC BACKWARD COMPAT + STORE HINTS FOR DOWNSTREAM PAGES
# ============================================================================



# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
if ledger.has_blockers():
    st.warning("Resolve blocker insights before treating downstream model results as defensible.")
else:
    st.success("EDA complete. Proceed to Feature Selection or Preprocessing.")
