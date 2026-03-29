"""
Page 02: Exploratory Data Analysis (Redesigned)

Architecture: Data first, coaching second.
  Section 0: At-a-Glance Header
  Section 1: Data Snapshot (interactive table + column inspector)
  Section 2: Shape of the Data (distributions, outliers, missing)
  Section 3: Relationships (correlations, target, feature explorer)
  Section 4: Macro Shape (PCA, UMAP, TDA, Mapper) — ≥16 features only
  Section 5: Coaching Layer (insight ledger summary)
  Section 6: Deep Dive Diagnostics (tabbed, intent-based)
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
    narrative_eda_linearity,
    narrative_eda_residuals,
    narrative_eda_influence,
    narrative_eda_normality,
    narrative_eda_sufficiency,
    narrative_eda_scaling,
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
render_breadcrumb("02_EDA")
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
@st.cache_data
def _compute_profile(_df, target, features, task_type, outlier_method):
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
    df, target_col or feature_cols[0],
    feature_cols, task_type_final or "regression", outlier_method
)
st.session_state["dataset_profile"] = profile

# Signals (cached)
@st.cache_data
def _compute_signals(_df, target, task_type, cohort_type, entity_id, outlier_method, _feature_cols=None):
    return compute_dataset_signals(_df, target, task_type, cohort_type, entity_id, outlier_method=outlier_method, feature_cols=_feature_cols)

try:
    signals = _compute_signals(
        df, target_col, task_type_final, cohort_type_final, entity_id_final, outlier_method,
        _feature_cols=feature_cols,
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

def _auto_generate_insights():
    """Write auto-detected insights to the ledger. Idempotent via upsert."""

    # Sufficiency
    sufficiency = getattr(getattr(profile, "data_sufficiency", None), "value", "adequate")
    if sufficiency == "insufficient":
        ledger.upsert(Insight(
            id="eda_sufficiency_insufficient",
            source_page="02_EDA", category="sufficiency", severity="blocker",
            finding=f"Sample size may be insufficient ({regime.n_rows:,} rows, {regime.n_features} features, ratio {regime.n_rows / max(regime.n_features, 1):.0f}:1)",
            implication="Complex models will likely overfit. Prefer simple baselines.",
            recommended_action="Reduce features or gather more data",
            relevant_pages=["04_Feature_Selection", "06_Train_and_Compare", "10_Report_Export"],
            model_scope=[MODEL_FAMILY_NEURAL],  # most affected by low sample size
        ))
    elif sufficiency == "borderline":
        ledger.upsert(Insight(
            id="eda_sufficiency_borderline",
            source_page="02_EDA", category="sufficiency", severity="warning",
            finding=f"Data sufficiency is borderline ({regime.n_rows:,} rows, {regime.n_features} features)",
            implication="Prefer simpler models and tighter regularization.",
            recommended_action="Consider feature reduction before complex modeling",
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
                relevant_pages=["05_Preprocess", "10_Report_Export"],
                metadata={"n_features": len(severe_missing), "max_rate": max(r for _, r in severe_missing)},
            ))
        if moderate_missing:
            cols_str = ", ".join(f"{c} ({r:.0%})" for c, r in moderate_missing[:8])
            ledger.upsert(Insight(
                id="eda_missing_moderate",
                source_page="02_EDA", category="data_quality", severity="info",
                finding=f"{len(moderate_missing)} feature(s) with 5-30% missing: {cols_str}",
                implication="Standard imputation (median/mode) should be sufficient. Consider adding missingness indicator features.",
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
                affected_features=[target_col],
                recommended_action="Apply a target transformation on the Train & Compare page (Log, Yeo-Johnson, or Box-Cox). Predictions are automatically back-transformed and metrics reported on the original scale. Tree-based models and Huber regression are also robust to target skew.",
                relevant_pages=["06_Train_and_Compare"],
                model_scope=ISSUE_MODEL_RELEVANCE["skewness"],  # linear, neural, distance
                metadata={"skewness": float(skew)},
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
                affected_features=[target_col],
                recommended_action="Use class weighting or stratified sampling",
                relevant_pages=["05_Preprocess", "06_Train_and_Compare"],
                # model_scope=[] → all models affected
                metadata={"imbalance_ratio": float(imbalance)},
            ))

    # Feature skewness — use cached computation
    @st.cache_data
    def _get_skewed_features(_df, _feature_cols):
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

    skewed_list = _get_skewed_features(df, feature_cols)
    if skewed_list:
        skew_names = ", ".join(f"{c} ({s:.1f})" for c, s in skewed_list[:8])
        if len(skewed_list) > 8:
            skew_names += f" +{len(skewed_list) - 8} more"
        ledger.upsert(Insight(
            id="eda_skew_group",
            source_page="02_EDA", category="distribution", severity="info",
            finding=f"{len(skewed_list)} feature(s) heavily skewed (|skew| > 2): {skew_names}",
            implication="Log or power transforms may improve linear model performance and reduce outlier influence",
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
            recommended_action="Consider GLM or GAM baselines in Train & Compare",
            relevant_pages=["06_Train_and_Compare"],
            resolved=True, resolved_by="Positive signal — no action needed",
            resolved_on_page="02_EDA", auto_generated=True,
        ))

    # Strong target signal opportunity
    if _has_target and task_type_final == "regression":
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            top_corr = max(
                (abs(df[c].corr(df[target_col])) for c in numeric_cols if c != target_col),
                default=0,
            )
            if top_corr > 0.7:
                ledger.upsert(Insight(
                    id="eda_opportunity_strong_signal",
                    source_page="02_EDA", category="relationship", severity="opportunity",
                    finding=f"Strong linear signal detected (max |r| with target = {top_corr:.2f})",
                    implication="Linear models may perform surprisingly well. Establish a strong OLS baseline before trying complex models.",
                    recommended_action="Run GLM baseline first in Train & Compare",
                    relevant_pages=["06_Train_and_Compare"],
                    metadata={"max_target_correlation": float(top_corr)},
                    resolved=True, resolved_by="Positive signal — no action needed",
                    resolved_on_page="02_EDA", auto_generated=True,
                ))

    # Non-linear relationship opportunity
    if _has_target and task_type_final == "regression":
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 3:
            # Compare Pearson vs Spearman to detect non-linearity
            pearson_corrs = [abs(df[c].corr(df[target_col])) for c in numeric_cols if c != target_col]
            spearman_corrs = [abs(df[c].corr(df[target_col], method="spearman")) for c in numeric_cols if c != target_col]
            if pearson_corrs and spearman_corrs:
                avg_gap = np.mean([s - p for p, s in zip(pearson_corrs, spearman_corrs) if not (np.isnan(p) or np.isnan(s))])
                if avg_gap > 0.08:
                    ledger.upsert(Insight(
                        id="eda_opportunity_nonlinear",
                        source_page="02_EDA", category="relationship", severity="opportunity",
                        finding=f"Features show non-linear relationships with target (avg Spearman-Pearson gap = {avg_gap:.3f})",
                        implication="Tree-based models (RF, XGBoost) or GAMs may capture structure that linear models miss",
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
                recommended_action="Standard metrics will be reliable in Train & Compare",
                relevant_pages=["06_Train_and_Compare"],
                resolved=True, resolved_by="Positive signal — no action needed",
                resolved_on_page="02_EDA", auto_generated=True,
            ))


_auto_generate_insights()


# ============================================================================
# SECTION 0: AT-A-GLANCE HEADER
# ============================================================================

st.title("📈 Explore Your Data")

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
    missing_pct = df[feature_cols].isnull().mean().mean() * 100
    st.metric("Missing", f"{missing_pct:.1f}%")
with cols[5]:
    sufficiency_val = getattr(getattr(profile, "data_sufficiency", None), "value", "adequate")
    st.metric("Sufficiency", sufficiency_val.title())

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
st.header("Data Snapshot")
st.caption("See your data. Sort, filter, and inspect columns to build initial intuition.")

# Interactive dataframe
st.dataframe(
    df.head(200),
    use_container_width=True,
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
        summary_df = df[numeric_features].describe().T
        summary_df["skew"] = df[numeric_features].skew()
        summary_df["missing_%"] = df[numeric_features].isnull().mean() * 100
        table(summary_df.round(3))

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
    def _compute_outlier_heatmap(_df, _numeric_feats, methods):
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

    outlier_data = _compute_outlier_heatmap(df, _outlier_features, ["iqr", "zscore"])
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


# ============================================================================
# SECTION 3: RELATIONSHIPS
# ============================================================================

st.markdown("---")
st.header("Relationships")
st.caption("How features relate to each other and to the target.")

# -- Correlation Matrix / Top Pairs ----------------------------------------
st.subheader("Feature Correlations")

if len(numeric_features) >= 2:
    corr_method = st.pills("Method", ["Pearson", "Spearman"], default="Pearson", key="corr_method")
    method_name = corr_method.lower() if corr_method else "pearson"

    @st.cache_data
    def _compute_corr(_df, _features, method):
        return _df[_features].corr(method=method).round(3)

    if regime.show_full_corr_matrix:
        # Full heatmap for narrow/medium datasets
        corr_matrix = _compute_corr(df, numeric_features, method_name)
        threshold = st.slider("Highlight threshold", 0.0, 1.0, 0.8, 0.05, key="corr_threshold")

        fig_corr = px.imshow(
            corr_matrix,
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            title=f"{corr_method} Correlation Matrix",
            aspect="auto",
        )
        fig_corr.update_layout(template="plotly_white", height=max(400, len(numeric_features) * 18 + 100))
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
        def _top_corr_pairs(_df, _features, method, n):
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

        pairs_df = _top_corr_pairs(df, numeric_features, method_name, top_n)
        n_total = len(numeric_features) * (len(numeric_features) - 1) // 2
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
        def _sort_by_target_corr(_df, _features, _target):
            corrs = _df[_features].corrwith(_df[_target]).abs().fillna(0)
            return corrs.sort_values(ascending=False).index.tolist()

        target_features = _sort_by_target_corr(df, target_features, target_col)

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
st.caption("Pick any two features to visualize their relationship.")

if len(feature_cols) >= 2:
    fe_col1, fe_col2, fe_col3 = st.columns(3)
    with fe_col1:
        feat_x = st.selectbox("X axis", feature_cols, index=0, key="fe_x")
    with fe_col2:
        default_y = 1 if len(feature_cols) > 1 else 0
        feat_y = st.selectbox("Y axis", feature_cols, index=default_y, key="fe_y")
    with fe_col3:
        color_options = ["None"] + ([target_col] if _has_target else []) + feature_cols
        color_by = st.selectbox("Color by", color_options, index=0, key="fe_color")

    plot_df = df[[feat_x, feat_y]].copy()
    if color_by != "None" and color_by in df.columns:
        plot_df[color_by] = df[color_by]

    plot_df = plot_df.dropna(subset=[feat_x, feat_y])
    if regime.needs_sampling and len(plot_df) > regime.sample_size:
        plot_df = plot_df.sample(regime.sample_size, random_state=42)

    fig_explorer = px.scatter(
        plot_df, x=feat_x, y=feat_y,
        color=color_by if color_by != "None" else None,
        opacity=0.5,
        title=f"{feat_x} vs {feat_y}",
    )
    fig_explorer.update_layout(template="plotly_white", height=450)
    st.plotly_chart(fig_explorer)

    # Show correlation for this pair
    if pd.api.types.is_numeric_dtype(df[feat_x]) and pd.api.types.is_numeric_dtype(df[feat_y]):
        r = df[feat_x].corr(df[feat_y])
        if not np.isnan(r):
            st.caption(f"Pearson r = {r:.3f}")

# -- Suggested Interactions ------------------------------------------------
if _has_target and len(numeric_features) >= 4:
    with st.expander("💡 Suggested Interactions (auto-detected)", expanded=False):
        st.caption("Top feature pairs by mutual information with target. Click to explore.")

        @st.cache_data
        def _compute_interactions(_df, _features, _target, _task_type, max_pairs=5):
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
            interactions = _compute_interactions(df, numeric_features, target_col, task_type_final)
            if interactions:
                for a, b, gain in interactions:
                    st.markdown(f"- **{a} × {b}** (MI gain: {gain:.4f})")
            else:
                st.caption("No strong interaction effects detected.")
        except Exception as e:
            st.caption(f"Interaction detection skipped: {str(e)[:80]}")


# ============================================================================
# SECTION 4: MACRO SHAPE (≥16 features only)
# ============================================================================

if regime.show_macro_shape and numeric_features:
    st.markdown("---")
    st.header("Macro Shape")
    st.caption("How your data looks in reduced dimensions. Each view reveals something the others hide.")

    from ml.macro_shape import (
        compute_pca, plot_scree, plot_pca_biplot,
        compute_umap, plot_umap,
        compute_persistence, plot_persistence_diagram, plot_persistence_barcode,
        compute_mapper, plot_mapper,
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
# SECTION 6: DEEP DIVE DIAGNOSTICS
# ============================================================================

st.markdown("---")
st.header("Deep Dive Diagnostics")
st.caption("Specialized analyses organized by intent. Run what's relevant, skip what isn't.")

from ml.physiology_reference import load_reference_bundle, match_variable_key
_ref_bundle = load_reference_bundle()
_nhanes_ref = _ref_bundle.get("nhanes", {}) if _ref_bundle else {}
_has_physio_matches = any(match_variable_key(f, _nhanes_ref) for f in feature_cols) if _nhanes_ref else False

if "eda_results" not in st.session_state:
    st.session_state.eda_results = {}


ACTION_NEXT_STEPS = {
    'multicollinearity_vif': '→ **Next:** Go to Feature Selection to use LASSO/RFE to resolve collinearity, or apply Ridge/ElasticNet regularization in training.',
    'missingness_scan': '→ **Next:** Configure imputation strategy in Preprocess. Consider whether missingness is informative (MNAR) before choosing a method.',
    'data_sufficiency_check': '→ **Next:** Consider reducing features via Feature Selection, or use simpler models (Ridge, LASSO) that handle high dimensionality better.',
    'influence_diagnostics': '→ **Next:** Review flagged high-influence points. Consider target trimming on the Train page, or use robust regression (Huber).',
    'leakage_scan': '→ **Next:** Remove suspected leakage columns in Upload & Audit before proceeding to modeling.',
    'target_profile': '→ **Next:** Apply target transformation (Log, Yeo-Johnson, Box-Cox) on the Train & Compare page.',
    'feature_scaling_check': '→ **Next:** Configure scaling in Preprocess. StandardScaler for normal features, RobustScaler if outliers are present.',
    'linearity_scatter': '→ **Next:** If non-linear patterns are visible, consider tree-based models or polynomial features in Feature Engineering.',
    'plausibility_check': '→ **Next:** Review flagged implausible values. Apply target trimming or filter rows in Upload & Audit.',
}


def _run_and_show(action_id: str, title: str, run_action: str, tab_key: str = ""):
    """Run an EDA action and display results with optional LLM interpretation."""
    from utils.llm_ui import build_llm_context, build_eda_full_results_context, render_interpretation_with_llm_button, gather_session_context

    key_prefix = f"{tab_key}_{action_id}" if tab_key else action_id
    if st.button(f"Run {title}", key=f"run_{key_prefix}", type="primary"):
        try:
            action_func = getattr(eda_actions, run_action, None)
            if action_func:
                with st.spinner(f"Running {title}..."):
                    result = action_func(df, target_col, feature_cols, signals, st.session_state)
                    st.session_state.eda_results[action_id] = result
                    log_methodology(step="EDA", action=f"Ran {title}", details={"analysis": run_action})
                    st.rerun()
            else:
                st.error(f"Action '{run_action}' not found")
        except Exception as e:
            st.error(f"Error: {str(e)}")

    if action_id in st.session_state.eda_results:
        result = st.session_state.eda_results[action_id]
        for w in result.get("warnings", []):
            st.warning(w)

        # Narrative interpretation
        ACTION_NARRATIVE = {
            "linearity_scatter": narrative_eda_linearity,
            "residual_analysis": narrative_eda_residuals,
            "influence_diagnostics": narrative_eda_influence,
            "normality_residuals": narrative_eda_normality,
            "multicollinearity_vif": narrative_eda_multicollinearity,
            "data_sufficiency_check": narrative_eda_sufficiency,
            "feature_scaling_check": narrative_eda_scaling,
        }
        findings = result.get("findings", [])[:2]
        stats = result.get("stats", {})
        nar_fn = ACTION_NARRATIVE.get(run_action)
        interp = nar_fn(stats, findings) if nar_fn else ("; ".join(findings) if findings else None)

        for idx, (fig_type, fig_data) in enumerate(result.get("figures", [])):
            if fig_type == "plotly":
                st.plotly_chart(fig_data, key=f"fig_{key_prefix}_{idx}")
            elif fig_type == "table":
                table(fig_data, key=f"tbl_{key_prefix}_{idx}")

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
                ctx, key=f"llm_{key_prefix}",
                result_session_key=f"llm_result_{key_prefix}",
                plot_type=action_id,
            )
        next_step = ACTION_NEXT_STEPS.get(action_id)
        if next_step:
            st.markdown(next_step)


_recommendations = []
if signals.high_missing_cols:
    _recommendations.append(('missingness_scan', 'Missingness Deep Dive', f'{len(signals.high_missing_cols)} columns have >20% missing data'))
if signals.collinearity_summary.get('high_pairs'):
    _recommendations.append(('multicollinearity_vif', 'VIF (Multicollinearity)', f'High correlation detected between feature pairs'))
if signals.target_stats.get('skew') and abs(signals.target_stats['skew']) > 1.5:
    _recommendations.append(('target_profile', 'Target Profile', f'Target is skewed (skew={signals.target_stats["skew"]:.1f})'))
if hasattr(signals, 'leakage_flags') and signals.leakage_flags:
    _recommendations.append(('leakage_scan', 'Leakage Detection', f'{len(signals.leakage_flags)} potential leakage flags'))
n_p_ratio = len(df) / max(1, len(feature_cols))
if n_p_ratio < 20:
    _recommendations.append(('data_sufficiency_check', 'Data Sufficiency', f'n/p ratio is {n_p_ratio:.1f} (< 20)'))

_recommendations = [(aid, title, reason) for aid, title, reason in _recommendations if aid not in st.session_state.get('eda_results', {})]

if _recommendations:
    st.markdown('#### 🎯 Recommended for Your Data')
    for aid, title, reason in _recommendations[:3]:
        col_rec_1, col_rec_2 = st.columns([3, 1])
        with col_rec_1:
            st.markdown(f'**{title}** — {reason}')
        with col_rec_2:
            if st.button(f'Run', key=f'rec_run_{aid}', type='primary'):
                action_func = getattr(eda_actions, aid, None)
                if action_func:
                    with st.spinner(f'Running {title}...'):
                        result = action_func(df, target_col, feature_cols, signals, st.session_state)
                        st.session_state.eda_results[aid] = result
                        log_methodology(step='EDA', action=f'Ran {title}', details={'analysis': aid})
                        st.rerun()
    st.markdown('---')

tab_readiness, tab_quality, tab_advanced = st.tabs(
    ["Model Readiness", "Feature Quality", "Advanced"]
)

with tab_readiness:
    st.caption("Check assumptions for linear and parametric models.")
    st.info(
        "💡 These diagnostics fit a **simple OLS regression as a quick proxy** to check linear modeling "
        "assumptions. This is standard EDA practice — the OLS here is a diagnostic tool, not your final "
        "model. Results help you decide whether linear models are appropriate and identify influential "
        "observations before committing to a full training run."
    )
    _run_and_show("linearity_scatter", "Linearity Check", "linearity_scatter", "readiness")
    st.markdown("---")
    if task_type_final == 'regression':
        _run_and_show("residual_analysis", "Residual Analysis", "residual_analysis", "readiness")
    else:
        st.caption("Regression only — not applicable for classification tasks.")
    st.markdown("---")
    if task_type_final == 'regression':
        _run_and_show("normality_residuals", "Normality of Residuals", "normality_residuals", "readiness")
    else:
        st.caption("Regression only — not applicable for classification tasks.")
    st.markdown("---")
    _run_and_show("multicollinearity_vif", "VIF (Multicollinearity)", "multicollinearity_vif", "readiness")
    st.markdown("---")
    if task_type_final == 'regression':
        _run_and_show("influence_diagnostics", "Influence Diagnostics", "influence_diagnostics", "readiness")
    else:
        st.caption("Regression only — not applicable for classification tasks.")

with tab_quality:
    st.caption("Validate data integrity and detect problems.")
    if _has_physio_matches:
        _run_and_show("plausibility_check", "Physiologic Plausibility", "plausibility_check", "quality")
    else:
        st.caption("Physiologic plausibility checks require biomedical variables with known reference ranges.")
    st.markdown("---")
    _run_and_show("leakage_scan", "Leakage Detection", "leakage_scan", "quality")
    st.markdown("---")
    _run_and_show("missingness_scan", "Missingness Deep Dive", "missingness_scan", "quality")
    st.markdown("---")
    _run_and_show("data_sufficiency_check", "Data Sufficiency", "data_sufficiency_check", "quality")
    st.markdown("---")
    _run_and_show("feature_scaling_check", "Feature Scaling", "feature_scaling_check", "quality")

with tab_advanced:
    st.caption("Interaction effects, dose-response, and quick baselines.")
    _run_and_show("interaction_analysis", "Interaction Detection", "interaction_analysis", "advanced")
    st.markdown("---")
    if task_type_final == 'regression':
        _run_and_show("dose_response_trends", "Dose-Response Trends", "dose_response_trends", "advanced")
    else:
        st.caption("Regression only — not applicable for classification tasks.")
    st.markdown("---")
    _run_and_show("outlier_influence", "Outlier Influence", "outlier_influence", "advanced")
    st.markdown("---")
    _run_and_show("target_profile", "Target Profile", "target_profile", "advanced")
    st.markdown("---")
    _run_and_show("quick_probe_baselines", "Quick Baselines", "quick_probe_baselines", "advanced")


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
        log_methodology(step="EDA", action="Generated Table 1", details={
            "grouping_var": grouping_var if grouping_var != "None" else None,
            "n_continuous": len(t1_continuous),
            "n_categorical": len(t1_categorical),
        })

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
