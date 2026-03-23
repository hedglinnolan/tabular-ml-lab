"""
Unified Insight Ledger — single source of truth for the analysis lifecycle.

One system, four views:
  1. Page coaching:         filtered entries relevant to current page
  2. Progress summary:      counts of detected/resolved/open
  3. Methodology audit:     chronological record of all decisions
  4. Publication narrative:  manuscript-ready prose grouped by workflow phase

Lifecycle: Detect → Recommend → Act → Narrate

Replaces:
  - eda_insights[] (storyline.add_insight)
  - feature_engineering_hints{} (session state dict)
  - coach_output (model_coach standalone object)
  - methodology_log (append-only list)
  - TRIPODTracker (manual checklist)
  - eda_decision_hub{}

Usage:
    ledger = get_ledger()
    ledger.add(Insight(
        id="eda_skew_BMI",
        source_page="02_EDA",
        category="distribution",
        severity="warning",
        finding="BMI is heavily right-skewed (skewness=3.2)",
        implication="Linear models assume normality; skew may degrade performance",
        recommended_action="Apply log or Box-Cox transform",
        relevant_pages=["03_Feature_Engineering", "05_Preprocess"],
        tripod_keys=["predictor_handling"],
        affected_features=["BMI"],
    ))
    ledger.resolve(
        "eda_skew_BMI",
        resolved_by="Applied log1p transform",
        resolved_on_page="05_Preprocess",
        resolution_details={"method": "log1p", "columns": ["BMI"]},
    )
    narrative = ledger.to_manuscript_narrative()
"""
from dataclasses import dataclass, field
from typing import List, Optional, Literal, Dict, Any, Tuple
from datetime import datetime
import streamlit as st


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEVERITY_ORDER = {"blocker": 0, "warning": 1, "info": 2, "opportunity": 3}

# ---------------------------------------------------------------------------
# Model families — used for model_scope on insights
# ---------------------------------------------------------------------------

MODEL_FAMILY_LINEAR = "linear"       # ridge, lasso, elasticnet, logreg, glm, huber
MODEL_FAMILY_TREE = "tree"           # rf, extratrees, histgb, xgb, lgbm
MODEL_FAMILY_NEURAL = "neural"       # nn (PyTorch MLP)
MODEL_FAMILY_DISTANCE = "distance"   # knn_reg, knn_clf
MODEL_FAMILY_MARGIN = "margin"       # svm (SVR/SVC)
MODEL_FAMILY_PROBABILISTIC = "prob"  # naive_bayes, lda

MODEL_FAMILIES = (
    MODEL_FAMILY_LINEAR, MODEL_FAMILY_TREE, MODEL_FAMILY_NEURAL,
    MODEL_FAMILY_DISTANCE, MODEL_FAMILY_MARGIN, MODEL_FAMILY_PROBABILISTIC,
)

# Map individual model keys → family
MODEL_TO_FAMILY = {
    "ridge": MODEL_FAMILY_LINEAR, "lasso": MODEL_FAMILY_LINEAR,
    "elasticnet": MODEL_FAMILY_LINEAR, "logreg": MODEL_FAMILY_LINEAR,
    "glm": MODEL_FAMILY_LINEAR, "huber": MODEL_FAMILY_LINEAR,
    "rf": MODEL_FAMILY_TREE, "extratrees_reg": MODEL_FAMILY_TREE,
    "extratrees_clf": MODEL_FAMILY_TREE, "histgb_reg": MODEL_FAMILY_TREE,
    "histgb_clf": MODEL_FAMILY_TREE,
    "nn": MODEL_FAMILY_NEURAL,
    "knn_reg": MODEL_FAMILY_DISTANCE, "knn_clf": MODEL_FAMILY_DISTANCE,
    "svm": MODEL_FAMILY_MARGIN,
    "naive_bayes": MODEL_FAMILY_PROBABILISTIC, "lda": MODEL_FAMILY_PROBABILISTIC,
}

# Human-readable model names for reports and narrative
MODEL_DISPLAY_NAMES = {
    "ridge": "Ridge Regression",
    "lasso": "LASSO",
    "elasticnet": "Elastic Net",
    "logreg": "Logistic Regression",
    "glm": "Generalized Linear Model",
    "huber": "Huber Regression",
    "rf": "Random Forest",
    "extratrees_reg": "Extra Trees (Regressor)",
    "extratrees_clf": "Extra Trees (Classifier)",
    "histgb_reg": "Histogram Gradient Boosting (Regressor)",
    "histgb_clf": "Histogram Gradient Boosting (Classifier)",
    "nn": "Neural Network (MLP)",
    "knn_reg": "k-Nearest Neighbors (Regressor)",
    "knn_clf": "k-Nearest Neighbors (Classifier)",
    "svm": "Support Vector Machine",
    "naive_bayes": "Naïve Bayes",
    "lda": "Linear Discriminant Analysis",
}


def model_display_name(key: str) -> str:
    """Human-readable name for a model key. Falls back to UPPER if unknown."""
    return MODEL_DISPLAY_NAMES.get(key.lower(), key.upper())


# Human-readable family names for coaching UI
FAMILY_DISPLAY_NAMES = {
    MODEL_FAMILY_LINEAR: "Linear Models",
    MODEL_FAMILY_TREE: "Tree-Based Models",
    MODEL_FAMILY_NEURAL: "Neural Networks",
    MODEL_FAMILY_DISTANCE: "Distance-Based Models",
    MODEL_FAMILY_MARGIN: "Margin-Based Models",
    MODEL_FAMILY_PROBABILISTIC: "Probabilistic Models",
}

# Which families are affected by common data issues?
# Empty list = applies to all families
ISSUE_MODEL_RELEVANCE = {
    "skewness": [MODEL_FAMILY_LINEAR, MODEL_FAMILY_NEURAL, MODEL_FAMILY_DISTANCE],
    "outliers": [MODEL_FAMILY_LINEAR, MODEL_FAMILY_NEURAL, MODEL_FAMILY_DISTANCE],
    "collinearity": [MODEL_FAMILY_LINEAR],
    "missing_data": [],       # all models affected
    "class_imbalance": [],    # all models affected
    "high_dimensionality": [MODEL_FAMILY_LINEAR, MODEL_FAMILY_DISTANCE, MODEL_FAMILY_MARGIN],
    "low_sample_size": [MODEL_FAMILY_NEURAL],  # most affected
    "non_normality": [MODEL_FAMILY_LINEAR, MODEL_FAMILY_PROBABILISTIC],
    "feature_scale": [MODEL_FAMILY_LINEAR, MODEL_FAMILY_NEURAL, MODEL_FAMILY_DISTANCE, MODEL_FAMILY_MARGIN],
}


# ---------------------------------------------------------------------------
# Resolution Details Schema — structured provenance for report generation
# ---------------------------------------------------------------------------
# Every resolution_details dict should include these standard fields when
# applicable. The narrative renderer uses these to produce publication prose.
# Freeform keys are still allowed for backward compat.

RESOLUTION_SCHEMA_FIELDS = {
    # What action was taken
    "action_type": "str — e.g., imputation, scaling, outlier_treatment, transform, feature_selection, encoding, training",
    "method": "str — specific technique: median, log1p, yeo_johnson, percentile_clip, standard, robust, onehot, etc.",
    "params": "dict — method-specific parameters: {lower: 4, upper: 96}, {alpha: 0.01}, {n_components: 5}",
    # What it was applied to
    "scope": "str — pipeline scope: 'all models', 'Ridge pipeline', 'Ridge, MLP pipelines'",
    "columns_affected": "list[str] — which columns/features were affected",
    "n_columns_affected": "int — count when listing all columns is unwieldy",
    # What happened as a result
    "result": "dict — outcome metrics: {rows_clipped: 47, pct_affected: '6.1%'}, {n_features_selected: 12}",
    # Context
    "models_trained": "list[str] — which models this applies to (for per-model pipeline provenance)",
    "triggered_by": "str — insight ID that prompted this action",
}


def format_resolution_detail(detail: Dict[str, Any], model_scope: Optional[List[str]] = None) -> str:
    """Render a resolution_details dict into publication-quality prose.

    Handles both structured (schema-conformant) and legacy (freeform) dicts.

    Examples:
        {"action_type": "outlier_treatment", "method": "percentile_clip",
         "params": {"lower": 4, "upper": 96}, "scope": "Ridge pipeline",
         "columns_affected": ["BMI", "Insulin"], "result": {"rows_clipped": 47}}
        → "Outliers were treated via percentile clipping (4th–96th percentile)
           on BMI and Insulin in the Ridge pipeline (47 rows clipped)."
    """
    if not detail:
        return ""

    action_type = detail.get("action_type", "")
    method = detail.get("method", "")
    params = detail.get("params", {})
    scope = detail.get("scope", "")
    columns = detail.get("columns_affected", [])
    n_cols = detail.get("n_columns_affected", len(columns) if columns else 0)
    result = detail.get("result", {})

    # --- Structured rendering by action_type ---
    if action_type:
        parts = []

        # Method description with params (pass full detail for templates that need top-level keys)
        method_str = _format_method(action_type, method, params, full_detail=detail)
        if method_str:
            parts.append(method_str)

        # Columns
        if columns and len(columns) <= 5:
            parts.append(f"on {_join_list(columns)}")
        elif n_cols:
            parts.append(f"on {n_cols} features")

        # Scope
        if scope and scope != "all models":
            parts.append(f"in the {scope}")

        # Result
        result_str = _format_result(result)
        if result_str:
            parts.append(f"({result_str})")

        # Model scope
        if model_scope:
            scope_names = [FAMILY_DISPLAY_NAMES.get(f, f) for f in model_scope]
            parts.append(f"[applicable to {_join_list(scope_names)}]")

        return " ".join(parts)

    # --- Legacy fallback: format freeform dict as key=value ---
    legacy_parts = []
    for k, v in detail.items():
        if k in ("finding", "category", "handled_by"):
            continue  # skip meta fields
        if isinstance(v, list) and v:
            legacy_parts.append(f"{k}: {', '.join(str(x) for x in v[:8])}")
        elif isinstance(v, dict) and v:
            inner = ", ".join(f"{ik}={iv}" for ik, iv in list(v.items())[:5])
            legacy_parts.append(f"{k}: {inner}")
        elif v is not None and v != "":
            legacy_parts.append(f"{k}: {v}")
    return "; ".join(legacy_parts) if legacy_parts else ""


def _format_method(action_type: str, method: str, params: Dict, full_detail: Optional[Dict] = None) -> str:
    """Format a method + params into readable prose.

    Args:
        action_type: The type of action (e.g., "outlier_treatment")
        method: The specific method (e.g., "percentile_clip")
        params: The params sub-dict (method-specific parameters)
        full_detail: The complete resolution_details dict — used by templates
            that need top-level keys like models_trained, result, etc.
    """
    # Merge params with full_detail for template access — params takes precedence
    merged = dict(full_detail or {})
    merged.update(params)

    # Specific formatting by action type
    templates = {
        ("outlier_treatment", "percentile_clip"): lambda p: (
            f"Outliers were treated via percentile clipping "
            f"({p.get('lower', '?')}th–{p.get('upper', '?')}th percentile)"
        ),
        ("outlier_treatment", "iqr"): lambda p: (
            f"Outliers were capped using the IQR method "
            f"(multiplier: {p.get('multiplier', p.get('iqr_multiplier', 1.5))})"
        ),
        ("outlier_treatment", "zscore"): lambda p: (
            f"Outliers were removed using z-score threshold "
            f"(|z| > {p.get('threshold', 3)})"
        ),
        ("imputation", "median"): lambda _: "Missing values were imputed with column medians",
        ("imputation", "mean"): lambda _: "Missing values were imputed with column means",
        ("imputation", "knn"): lambda p: (
            f"Missing values were imputed using k-nearest neighbors "
            f"(k={p.get('n_neighbors', p.get('k', 5))})"
        ),
        ("imputation", "most_frequent"): lambda _: "Missing categorical values were imputed with mode",
        ("scaling", "standard"): lambda _: "Features were standardized (zero mean, unit variance)",
        ("scaling", "robust"): lambda _: "Features were scaled using robust scaling (median/IQR)",
        ("transform", "log1p"): lambda _: "Log(1+x) transform was applied",
        ("transform", "yeo_johnson"): lambda _: "Yeo-Johnson power transform was applied",
        ("transform", "box_cox"): lambda _: "Box-Cox power transform was applied",
        ("encoding", "onehot"): lambda _: "Categorical features were one-hot encoded",
        ("encoding", "ordinal"): lambda _: "Categorical features were ordinal encoded",
        ("feature_selection", "consensus"): lambda p: (
            f"Features were selected by consensus across methods "
            f"(threshold: {p.get('threshold', '?')})"
        ),
        ("feature_selection", "manual"): lambda _: "Features were manually selected",
        ("dimensionality_reduction", "pca"): lambda p: (
            f"PCA was applied (n_components={p.get('n_components', '?')})"
        ),
        ("preprocessing", "per_model_pipeline"): lambda p: (
            f"Per-model preprocessing pipelines were configured for "
            f"{len(p.get('models_trained', []))} model(s)"
            if p.get("models_trained")
            else "Preprocessing pipelines were configured"
        ),
        ("training", "model_comparison"): lambda p: (
            f"{len(p.get('models_trained', []))} models were trained and compared"
            + (f"; best: {p['result']['best_model'].upper()}" if p.get("result", {}).get("best_model") else "")
            if p.get("models_trained") else "Models were trained"
        ),
        ("acknowledgment", "accepted_risk"): lambda _: (
            "Acknowledged and accepted as a study limitation"
        ),
        ("data_setup", None): lambda _: "Dataset configured for analysis",
        ("data_cleaning", None): lambda _: "Data cleaning operations applied",
    }

    key = (action_type, method)
    if key in templates:
        return templates[key](merged)

    # Try action_type with None method (for data_setup, data_cleaning, etc.)
    key_none = (action_type, None)
    if key_none in templates:
        return templates[key_none](merged)

    # Generic fallback
    if method and method != "none":
        param_str = ""
        if params:
            param_str = " (" + ", ".join(f"{k}={v}" for k, v in params.items()) + ")"
        action_label = action_type.replace("_", " ") if action_type else "Processing"
        return f"{action_label.capitalize()} via {method}{param_str}"

    if action_type:
        return action_type.replace("_", " ").capitalize()

    return ""


def _format_result(result: Dict) -> str:
    """Format result dict into a parenthetical summary."""
    if not result:
        return ""
    parts = []
    for k, v in result.items():
        label = k.replace("_", " ")
        if isinstance(v, float):
            parts.append(f"{label}: {v:.1%}" if v < 1 else f"{label}: {v:.2f}")
        else:
            parts.append(f"{label}: {v}")
    return "; ".join(parts)


def _join_list(items: list, conjunction: str = "and") -> str:
    """Join list items with Oxford comma."""
    if len(items) == 0:
        return ""
    if len(items) == 1:
        return str(items[0])
    if len(items) == 2:
        return f"{items[0]} {conjunction} {items[1]}"
    return ", ".join(str(i) for i in items[:-1]) + f", {conjunction} {items[-1]}"


def models_to_families(model_keys: List[str]) -> List[str]:
    """Convert a list of model keys to unique family names."""
    return list(dict.fromkeys(
        MODEL_TO_FAMILY.get(k, MODEL_FAMILY_LINEAR) for k in model_keys
    ))


def families_display(families: List[str]) -> str:
    """Human-readable string of family names."""
    return ", ".join(FAMILY_DISPLAY_NAMES.get(f, f) for f in families)


CATEGORIES = (
    "data_quality",    # missing data, duplicates, implausible values
    "distribution",    # skewness, outliers, target shape
    "relationship",    # correlations, interactions, leakage
    "topology",        # PCA structure, UMAP clusters, persistence features
    "sufficiency",     # sample size, events per variable
    "methodology",     # split strategy, model assumptions, preprocessing choices
    "model_selection", # model recommendations, training observations
    "explainability",  # SHAP findings, feature importance, calibration
    "sensitivity",     # seed stability, feature dropout robustness
    "validation",      # statistical test results, hypothesis testing
)

SEVERITIES = ("blocker", "warning", "info", "opportunity")

# Page identifiers used throughout the app
PAGES = (
    "01_Upload_and_Audit",
    "02_EDA",
    "03_Feature_Engineering",
    "04_Feature_Selection",
    "05_Preprocess",
    "06_Train_and_Compare",
    "07_Explainability",
    "08_Sensitivity_Analysis",
    "09_Hypothesis_Testing",
    "10_Report_Export",
)

# Workflow phase grouping for narrative generation
WORKFLOW_PHASES = {
    "Data Preparation": ["01_Upload_and_Audit", "02_EDA"],
    "Feature Engineering & Selection": ["03_Feature_Engineering", "04_Feature_Selection"],
    "Preprocessing": ["05_Preprocess"],
    "Model Training & Evaluation": ["06_Train_and_Compare"],
    "Interpretation & Validation": ["07_Explainability", "08_Sensitivity_Analysis", "09_Hypothesis_Testing"],
}

# TRIPOD auto-completion mapping: category → TRIPOD auto_keys
# When a ledger entry with this category is resolved, these TRIPOD items auto-complete
TRIPOD_CATEGORY_MAP = {
    "data_quality": ["missing_data"],
    "distribution": ["predictor_handling"],
    "relationship": ["predictor_handling"],
    "sufficiency": ["sample_size"],
    "methodology": ["model_building", "performance_measures"],
    "model_selection": ["model_building"],
    "explainability": ["full_model", "interpretation"],
    "sensitivity": ["performance_ci"],
    "validation": ["performance_measures"],
}


# ---------------------------------------------------------------------------
# Insight dataclass
# ---------------------------------------------------------------------------

@dataclass
class Insight:
    """A single lifecycle entry: observation → recommendation → resolution.

    Attributes:
        id: Deterministic unique key.
            Convention: {source_page_short}_{category_short}_{detail}
            e.g. "eda_skew_BMI", "preprocess_impute_age", "train_overfit_RF"
        source_page: Page that created this, e.g. "02_EDA"
        category: One of CATEGORIES
        severity: One of SEVERITIES
        finding: What was observed (the fact)
        implication: What it means for the analysis
        recommended_action: What to do about it
        relevant_pages: Pages where this should surface as coaching
        affected_features: Columns this concerns
        tripod_keys: TRIPOD auto_keys this entry satisfies when resolved
        model_scope: Which model families this insight applies to.
            Empty list = applies to ALL families (e.g., missing data).
            Non-empty = only relevant to listed families.
            Values from MODEL_FAMILIES: "linear", "tree", "neural", etc.
        auto_generated: True if system-detected, False if user-created

        Resolution fields (populated when user acts):
        resolved: Whether addressed
        resolved_by: Description of action taken
        resolved_on_page: Where the user acted
        resolved_at: ISO timestamp
        resolution_details: Structured metadata for report generation
            e.g. {"method": "knn", "k": 5, "columns": ["BMI", "Age"]}

        Timestamps:
        created_at: ISO timestamp of creation

        Extensibility:
        metadata: Arbitrary extra data (skew value, correlation coef, etc.)
    """
    id: str
    source_page: str
    category: str
    severity: str
    finding: str
    implication: str
    recommended_action: str = ""
    relevant_pages: List[str] = field(default_factory=list)
    affected_features: List[str] = field(default_factory=list)
    tripod_keys: List[str] = field(default_factory=list)
    model_scope: List[str] = field(default_factory=list)  # empty = all families
    auto_generated: bool = True
    # Legacy field — kept for backward compat, mapped to relevant_pages[0]
    action_page: str = ""

    # Resolution
    resolved: bool = False
    resolved_by: str = ""
    resolved_on_page: str = ""
    resolved_at: Optional[str] = None
    resolution_details: Dict[str, Any] = field(default_factory=dict)

    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Extensibility
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Migrate legacy action_page to relevant_pages
        if self.action_page and self.action_page not in self.relevant_pages:
            self.relevant_pages.append(self.action_page)
        # Auto-populate tripod_keys from category if not provided
        if not self.tripod_keys and self.category in TRIPOD_CATEGORY_MAP:
            self.tripod_keys = list(TRIPOD_CATEGORY_MAP[self.category])

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for session state / JSON export."""
        return {
            "id": self.id,
            "source_page": self.source_page,
            "category": self.category,
            "severity": self.severity,
            "finding": self.finding,
            "implication": self.implication,
            "recommended_action": self.recommended_action,
            "relevant_pages": list(self.relevant_pages),
            "affected_features": list(self.affected_features),
            "tripod_keys": list(self.tripod_keys),
            "model_scope": list(self.model_scope),
            "auto_generated": self.auto_generated,
            "action_page": self.action_page,
            "resolved": self.resolved,
            "resolved_by": self.resolved_by,
            "resolved_on_page": self.resolved_on_page,
            "resolved_at": self.resolved_at,
            "resolution_details": dict(self.resolution_details),
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Insight":
        """Deserialize from dict, ignoring unknown keys."""
        valid_keys = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# InsightLedger
# ---------------------------------------------------------------------------

class InsightLedger:
    """Central registry for the full analysis lifecycle.

    Stored in st.session_state['insight_ledger'].
    All pages read/write through get_ledger() helper.
    """

    def __init__(self):
        self._insights: List[Insight] = []

    # == Core operations ====================================================

    def add(self, insight: Insight) -> bool:
        """Add insight if not duplicate (by id). Returns True if added."""
        if any(i.id == insight.id for i in self._insights):
            return False
        self._insights.append(insight)
        return True

    def upsert(self, insight: Insight) -> bool:
        """Add or update insight by id. Returns True if newly added.

        Preserves existing non-default content fields when the new insight
        doesn't specify them (e.g., affected_features, model_scope, metadata).
        Always preserves resolution state if already resolved.
        """
        for idx, existing in enumerate(self._insights):
            if existing.id == insight.id:
                # Preserve resolution state if already resolved
                if existing.resolved and not insight.resolved:
                    insight.resolved = existing.resolved
                    insight.resolved_by = existing.resolved_by
                    insight.resolved_on_page = existing.resolved_on_page
                    insight.resolved_at = existing.resolved_at
                    insight.resolution_details = existing.resolution_details
                # Preserve non-empty content fields when new value is default/empty
                if not insight.affected_features and existing.affected_features:
                    insight.affected_features = existing.affected_features
                if not insight.model_scope and existing.model_scope:
                    insight.model_scope = existing.model_scope
                if not insight.metadata and existing.metadata:
                    insight.metadata = existing.metadata
                if not insight.tripod_keys and existing.tripod_keys:
                    insight.tripod_keys = existing.tripod_keys
                if not insight.relevant_pages and existing.relevant_pages:
                    insight.relevant_pages = existing.relevant_pages
                self._insights[idx] = insight
                return False
        self._insights.append(insight)
        return True

    def resolve(
        self,
        insight_id: str,
        resolved_by: str,
        resolved_on_page: str,
        resolution_details: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Mark an insight as resolved with structured details.

        Args:
            insight_id: The insight to resolve
            resolved_by: Human-readable description of what was done
            resolved_on_page: Page where the action was taken
            resolution_details: Structured metadata for report generation
                e.g. {"method": "knn", "k": 5, "columns": ["BMI"]}

        Returns True if found and resolved.
        """
        for i in self._insights:
            if i.id == insight_id:
                i.resolved = True
                i.resolved_by = resolved_by
                i.resolved_on_page = resolved_on_page
                i.resolved_at = datetime.now().isoformat()
                if resolution_details:
                    i.resolution_details = resolution_details
                return True
        import logging
        logging.getLogger(__name__).warning(
            f"InsightLedger.resolve: insight_id='{insight_id}' not found. "
            f"Check for typos in the resolution wiring."
        )
        return False

    def remove(self, insight_id: str) -> bool:
        """Remove an insight entirely. Returns True if found."""
        before = len(self._insights)
        self._insights = [i for i in self._insights if i.id != insight_id]
        return len(self._insights) < before

    def clear(self):
        """Remove all insights."""
        self._insights.clear()

    # == Queries ============================================================

    @property
    def insights(self) -> List[Insight]:
        """All insights, sorted by severity (blockers first)."""
        return sorted(
            self._insights,
            key=lambda i: SEVERITY_ORDER.get(i.severity, 99),
        )

    def get(self, insight_id: str) -> Optional[Insight]:
        """Get a specific insight by id."""
        for i in self._insights:
            if i.id == insight_id:
                return i
        return None

    def get_unresolved(
        self,
        severity: Optional[str] = None,
        category: Optional[str] = None,
        page: Optional[str] = None,
        source_page: Optional[str] = None,
        model_families: Optional[List[str]] = None,
    ) -> List[Insight]:
        """Get unresolved insights, optionally filtered.

        Args:
            page: Filter by relevant_pages (entries that should surface on this page)
            source_page: Filter by where the insight was created
            model_families: Filter to insights relevant to these model families.
                An insight matches if its model_scope is empty (applies to all)
                or intersects with the provided families.
        """
        results = [i for i in self._insights if not i.resolved]
        if severity:
            results = [i for i in results if i.severity == severity]
        if category:
            results = [i for i in results if i.category == category]
        if page:
            results = [
                i for i in results
                if page in i.relevant_pages or page == i.source_page
            ]
        if source_page:
            results = [i for i in results if i.source_page == source_page]
        if model_families:
            results = [
                i for i in results
                if not i.model_scope  # empty = all families
                or set(i.model_scope) & set(model_families)
            ]
        return sorted(results, key=lambda i: SEVERITY_ORDER.get(i.severity, 99))

    def get_resolved(
        self,
        page: Optional[str] = None,
        category: Optional[str] = None,
    ) -> List[Insight]:
        """Get resolved insights, optionally filtered."""
        resolved = [i for i in self._insights if i.resolved]
        if page:
            resolved = [
                i for i in resolved
                if i.resolved_on_page == page or page in i.relevant_pages
            ]
        if category:
            resolved = [i for i in resolved if i.category == category]
        return sorted(resolved, key=lambda i: i.resolved_at or "")

    def get_for_page(self, page: str) -> List[Insight]:
        """Get ALL insights relevant to a page (resolved and unresolved).

        An insight is relevant to a page if:
        - page is in relevant_pages, OR
        - page is the source_page, OR
        - page is the resolved_on_page
        """
        return [
            i for i in self._insights
            if page in i.relevant_pages
            or i.source_page == page
            or i.resolved_on_page == page
        ]

    def get_for_models(
        self,
        model_keys: List[str],
        page: Optional[str] = None,
        unresolved_only: bool = True,
    ) -> Dict[str, List[Insight]]:
        """Get insights grouped by model family for coaching display.

        Args:
            model_keys: The user's selected model keys (e.g., ["ridge", "rf", "nn"])
            page: Optional page filter
            unresolved_only: If True, only return unresolved insights

        Returns:
            Dict mapping family display name → list of relevant insights.
            Also includes "_universal" key for insights with empty model_scope.
        """
        families = models_to_families(model_keys)
        all_insights = (
            self.get_unresolved(page=page) if unresolved_only
            else self.get_for_page(page) if page
            else self._insights
        )

        result: Dict[str, List[Insight]] = {}
        universal = []

        for insight in all_insights:
            if not insight.model_scope:
                # Applies to all models
                universal.append(insight)
            else:
                # Only add to families the user has selected
                for family in families:
                    if family in insight.model_scope:
                        display = FAMILY_DISPLAY_NAMES.get(family, family)
                        result.setdefault(display, []).append(insight)

        if universal:
            result["All Models"] = universal

        return result

    def coaching_summary_for_models(
        self,
        model_keys: List[str],
        page: Optional[str] = None,
    ) -> str:
        """Generate a one-line coaching summary for the selected models.

        e.g., "3 items for Linear Models, 1 for all models. Tree-Based Models: no issues."
        """
        grouped = self.get_for_models(model_keys, page=page, unresolved_only=True)
        if not grouped:
            return "No coaching notes for your selected models."

        families = models_to_families(model_keys)
        parts = []
        for family in families:
            display = FAMILY_DISPLAY_NAMES.get(family, family)
            items = grouped.get(display, [])
            if items:
                parts.append(f"{len(items)} for {display}")

        universal = grouped.get("All Models", [])
        if universal:
            parts.append(f"{len(universal)} for all models")

        # Note families with no issues
        clean_families = [
            FAMILY_DISPLAY_NAMES.get(f, f)
            for f in families
            if FAMILY_DISPLAY_NAMES.get(f, f) not in grouped
        ]
        clean_str = ""
        if clean_families:
            clean_str = f" {', '.join(clean_families)}: no issues."

        return f"{', '.join(parts)}.{clean_str}" if parts else "No coaching notes."

    def get_for_features(self, features: List[str]) -> List[Insight]:
        """Get insights affecting any of the specified features."""
        feature_set = set(features)
        return [
            i for i in self._insights
            if feature_set.intersection(i.affected_features)
        ]

    def get_by_category(self, category: str) -> List[Insight]:
        """Get all insights in a category (resolved and unresolved)."""
        return [i for i in self._insights if i.category == category]

    # == Summary ============================================================

    def summary(self) -> Dict[str, int]:
        """Summary counts for display."""
        unresolved = [i for i in self._insights if not i.resolved]
        return {
            "total": len(self._insights),
            "unresolved": len(unresolved),
            "blockers": sum(1 for i in unresolved if i.severity == "blocker"),
            "warnings": sum(1 for i in unresolved if i.severity == "warning"),
            "info": sum(1 for i in unresolved if i.severity == "info"),
            "opportunities": sum(
                1 for i in unresolved if i.severity == "opportunity"
            ),
            "resolved": sum(1 for i in self._insights if i.resolved),
        }

    def page_summary(self, page: str) -> Dict[str, int]:
        """Summary counts for a specific page."""
        relevant = self.get_for_page(page)
        unresolved = [i for i in relevant if not i.resolved]
        return {
            "total": len(relevant),
            "unresolved": len(unresolved),
            "blockers": sum(1 for i in unresolved if i.severity == "blocker"),
            "warnings": sum(1 for i in unresolved if i.severity == "warning"),
            "resolved": sum(1 for i in relevant if i.resolved),
        }

    def has_blockers(self) -> bool:
        """Quick check: any unresolved blockers?"""
        return any(
            i.severity == "blocker" and not i.resolved for i in self._insights
        )

    # == TRIPOD auto-completion =============================================

    def get_tripod_status(self) -> Dict[str, bool]:
        """Compute TRIPOD checklist completion from resolved entries.

        Returns dict of {tripod_auto_key: completed}.
        A TRIPOD item is completed when any resolved insight has that key
        in its tripod_keys.
        """
        completed = set()
        for i in self._insights:
            if i.resolved:
                completed.update(i.tripod_keys)
        # Import TRIPOD items to get full list
        from ml.publication import TRIPOD_ITEMS
        return {
            item["auto_key"]: item["auto_key"] in completed
            for item in TRIPOD_ITEMS
        }

    def get_tripod_progress(self) -> Tuple[int, int]:
        """Returns (completed, total) for TRIPOD checklist."""
        status = self.get_tripod_status()
        return sum(status.values()), len(status)

    # == Methodology log (replaces standalone methodology_log) ==============

    def get_methodology_log(self) -> List[Dict[str, Any]]:
        """Generate methodology log from resolved insights.

        Returns a list compatible with the old methodology_log format,
        so Report Export works without changes during migration.
        """
        log = []
        for i in self.get_resolved():
            log.append({
                "step": _page_to_step_name(i.resolved_on_page or i.source_page),
                "action": i.resolved_by,
                "details": {
                    "finding": i.finding,
                    "category": i.category,
                    **i.resolution_details,
                },
                "timestamp": i.resolved_at or i.created_at,
            })
        return log

    # == Report generation ==================================================

    def _is_narrative_worthy(self, insight: Insight) -> bool:
        """Check if an insight should appear in publication narratives.

        Excludes:
        - audit_only entries (EDA activity logs like "Generated Table 1")
        - bridge entries that are just activity records with no provenance value
          (identified by implication="Logged methodology decision" AND no
          meaningful resolution_details beyond action_type)
        """
        # Explicit audit-only flag
        if insight.metadata.get("audit_only"):
            return False
        # Bridge entries that just record "I clicked a button" without
        # meaningful structured provenance (no columns_affected, no params, etc.)
        if (insight.implication == "Logged methodology decision"
                and insight.auto_generated):
            details = insight.resolution_details
            # Keep if it has substantive structured details
            has_substance = any(
                details.get(k) for k in (
                    "columns_affected", "params", "per_model_config",
                    "result", "models_trained", "scope",
                )
            )
            if not has_substance:
                return False
        return True

    def narrative_for_report(self) -> str:
        """Generate a concise methods-section narrative."""
        s = self.summary()
        resolved = [i for i in self.get_resolved() if self._is_narrative_worthy(i)]
        unresolved = self.get_unresolved()

        lines = []
        lines.append(
            f"Exploratory analysis identified {s['total']} "
            f"data observations. {s['resolved']} were addressed during the "
            f"modeling workflow; {s['unresolved']} were documented and accepted."
        )

        if resolved:
            lines.append("")
            lines.append("Addressed observations:")
            for i in resolved:
                if i.resolution_details.get("action_type"):
                    detail_prose = format_resolution_detail(
                        i.resolution_details, model_scope=i.model_scope
                    )
                    lines.append(f"  - {i.finding} → {detail_prose}")
                else:
                    lines.append(
                        f"  - {i.finding} → {i.resolved_by} ({i.resolved_on_page})"
                    )

        if unresolved:
            lines.append("")
            lines.append("Accepted/unresolved observations:")
            for i in unresolved:
                lines.append(
                    f"  - [{i.severity.upper()}] {i.finding}: {i.implication}"
                )

        return "\n".join(lines)

    def to_manuscript_narrative(self) -> Dict[str, str]:
        """Generate manuscript-ready narrative grouped by workflow phase.

        Returns dict of {phase_name: narrative_text} for direct insertion
        into the LaTeX methods section.
        """
        narratives = {}

        seen_ids = set()  # Prevent duplicate entries across phases
        for phase_name, phase_pages in WORKFLOW_PHASES.items():
            resolved_in_phase = []
            for i in self.get_resolved():
                if i.id in seen_ids:
                    continue
                if not self._is_narrative_worthy(i):
                    seen_ids.add(i.id)  # Mark as seen so it doesn't appear elsewhere
                    continue
                # Prefer resolved_on_page (where the action happened)
                primary_page = i.resolved_on_page or i.source_page
                if primary_page in phase_pages:
                    resolved_in_phase.append(i)
                    seen_ids.add(i.id)

            if not resolved_in_phase:
                continue

            sentences = []
            for i in resolved_in_phase:
                # Use structured renderer if resolution_details has action_type
                if i.resolution_details.get("action_type"):
                    detail_prose = format_resolution_detail(
                        i.resolution_details, model_scope=i.model_scope
                    )
                    sentences.append(f"{i.finding}. {detail_prose}.")
                else:
                    # Legacy fallback — resolved_by + freeform details
                    detail_str = ""
                    if i.resolution_details:
                        parts = []
                        for k, v in i.resolution_details.items():
                            if k in ("method", "strategy", "approach", "finding", "category"):
                                continue
                            if isinstance(v, list):
                                parts.append(f"{k}: {', '.join(str(x) for x in v)}")
                            else:
                                parts.append(f"{k}={v}")
                        if parts:
                            detail_str = f" ({', '.join(parts)})"

                    scope_str = ""
                    if i.model_scope:
                        scope_names = [
                            FAMILY_DISPLAY_NAMES.get(f, f) for f in i.model_scope
                        ]
                        scope_str = f" [applicable to {', '.join(scope_names)}]"

                    sentences.append(
                        f"{i.finding}. {i.resolved_by}{detail_str}.{scope_str}"
                    )

            narratives[phase_name] = " ".join(sentences)

        return narratives

    def provenance_timeline(self) -> List[Dict[str, Any]]:
        """Ordered timeline of insight creation and resolution for audit."""
        events = []
        for i in self._insights:
            events.append({
                "timestamp": i.created_at,
                "type": "created",
                "insight_id": i.id,
                "severity": i.severity,
                "finding": i.finding,
                "page": i.source_page,
            })
            if i.resolved and i.resolved_at:
                events.append({
                    "timestamp": i.resolved_at,
                    "type": "resolved",
                    "insight_id": i.id,
                    "resolved_by": i.resolved_by,
                    "resolution_details": i.resolution_details,
                    "page": i.resolved_on_page,
                })
        return sorted(events, key=lambda e: e["timestamp"])

    # == Serialization =====================================================

    def to_list(self) -> List[Dict[str, Any]]:
        """Serialize all insights."""
        return [i.to_dict() for i in self._insights]

    @classmethod
    def from_list(cls, items: List[Dict[str, Any]]) -> "InsightLedger":
        """Deserialize from list of dicts."""
        ledger = cls()
        for item in items:
            try:
                ledger.add(Insight.from_dict(item))
            except (TypeError, KeyError):
                continue  # skip malformed entries
        return ledger

    def __len__(self) -> int:
        return len(self._insights)

    def __repr__(self) -> str:
        s = self.summary()
        return (
            f"InsightLedger({s['total']} total, {s['unresolved']} unresolved, "
            f"{s['blockers']} blockers)"
        )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _page_to_step_name(page_id: str) -> str:
    """Convert page identifier to human-readable step name."""
    names = {
        "01_Upload_and_Audit": "Upload & Audit",
        "02_EDA": "Exploratory Data Analysis",
        "03_Feature_Engineering": "Feature Engineering",
        "04_Feature_Selection": "Feature Selection",
        "05_Preprocess": "Preprocessing",
        "06_Train_and_Compare": "Model Training",
        "07_Explainability": "Explainability",
        "08_Sensitivity_Analysis": "Sensitivity Analysis",
        "09_Hypothesis_Testing": "Statistical Validation",
        "10_Report_Export": "Report Export",
    }
    return names.get(page_id, page_id)


# ---------------------------------------------------------------------------
# Session-state accessor
# ---------------------------------------------------------------------------

def get_ledger() -> InsightLedger:
    """Get or create the InsightLedger from session state.

    This is the single entry point. All pages call this.
    """
    if "insight_ledger" not in st.session_state:
        st.session_state.insight_ledger = InsightLedger()
    ledger = st.session_state.insight_ledger
    if not isinstance(ledger, InsightLedger):
        # Recover from serialization edge cases
        st.session_state.insight_ledger = InsightLedger()
        ledger = st.session_state.insight_ledger
    return ledger



