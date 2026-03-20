"""
Insight Ledger — unified logical layer for cross-page insight tracking.

Three-layer architecture:
  1. Data layer:    dataset_profile (computed state, read-only)
  2. Insight layer:  InsightLedger (observations, read/write/resolve) ← THIS FILE
  3. Action layer:   methodology_log (actions taken, append-only)

Replaces: eda_insights[], feature_engineering_hints{}, eda_decision_hub{}
Bridges to: methodology_log via resolved_by field

Usage:
    ledger = get_ledger()
    ledger.add(Insight(id="eda_skew_BMI", ...))
    ledger.resolve("eda_skew_BMI", "Applied log transform", "05_Preprocess")
    unresolved = ledger.get_unresolved(severity="blocker")
"""
from dataclasses import dataclass, field
from typing import List, Optional, Literal, Dict, Any, Tuple
from datetime import datetime
import streamlit as st


# ---------------------------------------------------------------------------
# Insight severity and category constants
# ---------------------------------------------------------------------------

SEVERITY_ORDER = {"blocker": 0, "warning": 1, "info": 2, "opportunity": 3}

CATEGORIES = (
    "data_quality",    # missing data, duplicates, implausible values
    "distribution",    # skewness, outliers, target shape
    "relationship",    # correlations, interactions, leakage
    "topology",        # PCA structure, UMAP clusters, persistence features
    "sufficiency",     # sample size, events per variable
    "methodology",     # split strategy, model assumptions
)

SEVERITIES = ("blocker", "warning", "info", "opportunity")


# ---------------------------------------------------------------------------
# Insight dataclass
# ---------------------------------------------------------------------------

@dataclass
class Insight:
    """A single trackable observation from any page in the app.
    
    Attributes:
        id: Deterministic unique key, e.g. "eda_skew_BMI" or "eda_corr_BMI_Weight".
            Convention: {source_page_short}_{category_short}_{feature_or_detail}
        source_page: Page that created this insight, e.g. "02_EDA"
        category: One of CATEGORIES
        severity: One of SEVERITIES
        finding: Plain-English description of what was observed
        implication: What it means for modeling decisions
        affected_features: Which columns this insight concerns
        recommended_action: What the user should do about it
        action_page: Which page handles the recommended action
        auto_generated: True if system-detected, False if user-promoted
        resolved: Whether the user has addressed this insight
        resolved_by: Description of what action resolved it
        resolved_on_page: Which page resolved it
        resolved_at: ISO timestamp of resolution
        created_at: ISO timestamp of creation
        metadata: Arbitrary extra data (e.g. skew value, correlation coefficient)
    """
    id: str
    source_page: str
    category: Literal[
        "data_quality", "distribution", "relationship",
        "topology", "sufficiency", "methodology",
    ]
    severity: Literal["blocker", "warning", "info", "opportunity"]
    finding: str
    implication: str
    affected_features: List[str] = field(default_factory=list)
    recommended_action: str = ""
    action_page: str = ""
    auto_generated: bool = True
    resolved: bool = False
    resolved_by: str = ""
    resolved_on_page: str = ""
    resolved_at: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for session state / JSON export."""
        return {
            "id": self.id,
            "source_page": self.source_page,
            "category": self.category,
            "severity": self.severity,
            "finding": self.finding,
            "implication": self.implication,
            "affected_features": list(self.affected_features),
            "recommended_action": self.recommended_action,
            "action_page": self.action_page,
            "auto_generated": self.auto_generated,
            "resolved": self.resolved,
            "resolved_by": self.resolved_by,
            "resolved_on_page": self.resolved_on_page,
            "resolved_at": self.resolved_at,
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Insight":
        """Deserialize from dict."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# InsightLedger
# ---------------------------------------------------------------------------

class InsightLedger:
    """Central insight registry across all pages.
    
    Stored in st.session_state['insight_ledger'].
    All pages read/write through get_ledger() helper.
    """

    def __init__(self):
        self._insights: List[Insight] = []

    # -- Core operations ---------------------------------------------------

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
                self._insights[idx] = insight
                return False
        self._insights.append(insight)
        return True

    def resolve(self, insight_id: str, resolved_by: str, resolved_on_page: str) -> bool:
        """Mark an insight as resolved. Returns True if found."""
        for i in self._insights:
            if i.id == insight_id:
                i.resolved = True
                i.resolved_by = resolved_by
                i.resolved_on_page = resolved_on_page
                i.resolved_at = datetime.now().isoformat()
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

    # -- Queries -----------------------------------------------------------

    @property
    def insights(self) -> List[Insight]:
        """All insights, sorted by severity (blockers first)."""
        return sorted(self._insights, key=lambda i: SEVERITY_ORDER.get(i.severity, 99))

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
        action_page: Optional[str] = None,
        source_page: Optional[str] = None,
    ) -> List[Insight]:
        """Get unresolved insights, optionally filtered."""
        results = [i for i in self._insights if not i.resolved]
        if severity:
            results = [i for i in results if i.severity == severity]
        if category:
            results = [i for i in results if i.category == category]
        if action_page:
            results = [i for i in results if i.action_page == action_page]
        if source_page:
            results = [i for i in results if i.source_page == source_page]
        return sorted(results, key=lambda i: SEVERITY_ORDER.get(i.severity, 99))

    def get_resolved(self) -> List[Insight]:
        """Get all resolved insights, sorted by resolution time."""
        resolved = [i for i in self._insights if i.resolved]
        return sorted(resolved, key=lambda i: i.resolved_at or "")

    def get_for_features(self, features: List[str]) -> List[Insight]:
        """Get insights affecting any of the specified features."""
        feature_set = set(features)
        return [i for i in self._insights
                if feature_set.intersection(i.affected_features)]

    def get_by_category(self, category: str) -> List[Insight]:
        """Get all insights in a category (resolved and unresolved)."""
        return [i for i in self._insights if i.category == category]

    # -- Summary -----------------------------------------------------------

    def summary(self) -> Dict[str, int]:
        """Summary counts for display."""
        unresolved = [i for i in self._insights if not i.resolved]
        return {
            "total": len(self._insights),
            "unresolved": len(unresolved),
            "blockers": sum(1 for i in unresolved if i.severity == "blocker"),
            "warnings": sum(1 for i in unresolved if i.severity == "warning"),
            "info": sum(1 for i in unresolved if i.severity == "info"),
            "opportunities": sum(1 for i in unresolved if i.severity == "opportunity"),
            "resolved": sum(1 for i in self._insights if i.resolved),
        }

    def has_blockers(self) -> bool:
        """Quick check: any unresolved blockers?"""
        return any(i.severity == "blocker" and not i.resolved for i in self._insights)

    # -- Report generation -------------------------------------------------

    def narrative_for_report(self) -> str:
        """Generate a methods-section narrative for Report Export."""
        s = self.summary()
        resolved = self.get_resolved()
        unresolved = self.get_unresolved()

        lines = []
        lines.append(
            f"Exploratory analysis and preprocessing identified {s['total']} "
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
                    "page": i.resolved_on_page,
                })
        return sorted(events, key=lambda e: e["timestamp"])

    # -- Backward compatibility --------------------------------------------

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
            if "skew" in i.id or (i.category == "distribution" and "skew" in i.finding.lower()):
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

    # -- Serialization -----------------------------------------------------

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

    # eda_decision_hub — computed on-the-fly from ledger, no longer stored
    # Pages that read it should migrate to ledger.summary() + ledger.get_unresolved()
