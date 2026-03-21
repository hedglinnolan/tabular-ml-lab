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
        """Add or update insight by id. Returns True if newly added."""
        for idx, existing in enumerate(self._insights):
            if existing.id == insight.id:
                # Preserve resolution state if already resolved
                if existing.resolved and not insight.resolved:
                    insight.resolved = existing.resolved
                    insight.resolved_by = existing.resolved_by
                    insight.resolved_on_page = existing.resolved_on_page
                    insight.resolved_at = existing.resolved_at
                    insight.resolution_details = existing.resolution_details
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
    ) -> List[Insight]:
        """Get unresolved insights, optionally filtered.

        Args:
            page: Filter by relevant_pages (entries that should surface on this page)
            source_page: Filter by where the insight was created
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

    def narrative_for_report(self) -> str:
        """Generate a concise methods-section narrative."""
        s = self.summary()
        resolved = self.get_resolved()
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

        for phase_name, phase_pages in WORKFLOW_PHASES.items():
            resolved_in_phase = []
            for i in self.get_resolved():
                if (i.resolved_on_page in phase_pages
                        or i.source_page in phase_pages):
                    resolved_in_phase.append(i)

            if not resolved_in_phase:
                continue

            sentences = []
            for i in resolved_in_phase:
                detail_str = ""
                if i.resolution_details:
                    # Build detail string from structured metadata
                    parts = []
                    for k, v in i.resolution_details.items():
                        if k in ("method", "strategy", "approach"):
                            continue  # already in resolved_by
                        if isinstance(v, list):
                            parts.append(f"{k}: {', '.join(str(x) for x in v)}")
                        else:
                            parts.append(f"{k}={v}")
                    if parts:
                        detail_str = f" ({', '.join(parts)})"

                sentences.append(
                    f"{i.finding}. {i.resolved_by}{detail_str}."
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

    # == Backward compatibility (remove after full migration) ===============

    def to_eda_insights(self) -> List[Dict[str, Any]]:
        """Backward compat: produce the old eda_insights[] format."""
        return [
            {
                "id": i.id,
                "finding": i.finding,
                "implication": i.implication,
                "category": i.category,
            }
            for i in self._insights
        ]

    def to_feature_engineering_hints(self) -> Dict[str, Any]:
        """Backward compat: produce the old feature_engineering_hints{} format."""
        skewed = []
        high_corr = []
        has_missing = False

        for i in self._insights:
            if "skew" in i.id or (
                i.category == "distribution" and "skew" in i.finding.lower()
            ):
                skew_val = i.metadata.get("skewness", 0)
                for feat in i.affected_features:
                    skewed.append({"name": feat, "skewness": skew_val})

            if i.category == "relationship" and "corr" in i.id:
                corr_val = i.metadata.get("correlation", 0)
                feats = i.affected_features
                if len(feats) >= 2:
                    high_corr.append({
                        "feature1": feats[0],
                        "feature2": feats[1],
                        "correlation": corr_val,
                    })

            if i.category == "data_quality" and "missing" in i.finding.lower():
                has_missing = True

        return {
            "skewed_features": skewed,
            "high_corr_pairs": high_corr,
            "has_missing": has_missing,
            "numeric_features": [],  # caller must fill from df
        }

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


def sync_backward_compat(ledger: InsightLedger, df=None):
    """Update legacy session-state keys from ledger.

    Call this after ledger mutations so pages that haven't migrated yet
    still see consistent data. Remove once all pages read ledger directly.
    """
    # eda_insights (used by Report Export, Train & Compare)
    st.session_state["eda_insights"] = ledger.to_eda_insights()

    # feature_engineering_hints (used by Feature Engineering)
    hints = ledger.to_feature_engineering_hints()
    if df is not None:
        import numpy as np
        hints["numeric_features"] = list(
            df.select_dtypes(include=[np.number]).columns
        )
    st.session_state["feature_engineering_hints"] = hints

    # methodology_log — bridge from ledger resolutions
    existing_log = st.session_state.get("methodology_log", [])
    ledger_log = ledger.get_methodology_log()
    # Merge: keep existing entries not from ledger, add ledger entries
    ledger_actions = {e["action"] for e in ledger_log}
    merged = [e for e in existing_log if e.get("action") not in ledger_actions]
    merged.extend(ledger_log)
    st.session_state["methodology_log"] = merged
