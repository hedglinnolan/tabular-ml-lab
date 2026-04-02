"""Tests for EDA recommendation panel → InsightLedger bridge.

Tests the _resolve_insights_from_eda_result logic directly using the InsightLedger,
without importing the full EDA page (which requires Streamlit runtime).
"""

import pytest
from utils.insight_ledger import Insight, InsightLedger


# Reproduce the mapping and function from pages/02_EDA.py for testing.
# This avoids importing the page module which triggers Streamlit at load time.
_ACTION_TO_INSIGHT_MAP = {
    "multicollinearity_vif": {"prefix": "eda_corr_cluster_", "category": "collinearity"},
    "leakage_scan": {"prefix": "eda_leakage_", "category": "leakage"},
    "missingness_scan": {"prefix": "eda_missing_", "category": "missing_data"},
    "target_profile": {"exact": ["eda_target_skew"], "category": "target"},
    "data_sufficiency_check": {"exact": ["eda_sufficiency_insufficient", "eda_sufficiency_borderline"], "category": "sufficiency"},
}


def _resolve_insights_from_eda_result(ledger, action_id, result, title):
    """Mirror of the bridge function in pages/02_EDA.py."""
    mapping = _ACTION_TO_INSIGHT_MAP.get(action_id)
    if not mapping:
        return

    findings = result.get("findings", [])
    warnings = result.get("warnings", [])
    stats = result.get("stats", {})

    resolution_details = {
        "action_type": "diagnostic_analysis",
        "method": action_id,
        "findings": findings,
        "warnings": warnings,
    }
    if stats:
        resolution_details["stats"] = stats

    resolved_msg = f"Diagnostic analysis performed: {title}"
    if findings:
        resolved_msg = f"{title}: {findings[0]}"

    exact_ids = mapping.get("exact", [])
    prefix = mapping.get("prefix", "")

    for insight in ledger.insights:
        if insight.resolved:
            continue
        match = (
            insight.id in exact_ids
            or (prefix and insight.id.startswith(prefix))
        )
        if match:
            ledger.resolve(
                insight.id,
                resolved_by=resolved_msg,
                resolved_on_page="02_EDA",
                resolution_details=resolution_details,
            )


@pytest.fixture
def eda_ledger():
    """Ledger with typical EDA insights."""
    ledger = InsightLedger()

    ledger.upsert(Insight(
        id="eda_corr_cluster_bmi_weight_waist",
        source_page="02_EDA", category="collinearity", severity="warning",
        finding="High collinearity cluster: BMI, weight, waist_circumference",
        implication="Correlated predictors inflate coefficient variance",
        recommended_action="Run VIF analysis",
    ))
    ledger.upsert(Insight(
        id="eda_leakage_cholesterol",
        source_page="02_EDA", category="data_quality", severity="blocker",
        finding="Cholesterol shows r=0.94 with target — possible leakage",
        implication="Model may use future information",
    ))
    ledger.upsert(Insight(
        id="eda_missing_severe",
        source_page="02_EDA", category="data_quality", severity="warning",
        finding="3 features have >20% missing data",
        implication="May bias imputation",
    ))
    ledger.upsert(Insight(
        id="eda_target_skew",
        source_page="02_EDA", category="distribution", severity="warning",
        finding="Target is right-skewed (skew=2.3)",
        implication="Linear models may underperform",
    ))
    ledger.upsert(Insight(
        id="eda_sufficiency_borderline",
        source_page="02_EDA", category="methodology", severity="warning",
        finding="Sample size (n=180) is borderline",
        implication="Overfitting risk elevated",
    ))

    return ledger


class TestEDALedgerBridge:

    def test_vif_resolves_collinearity_insight(self, eda_ledger):
        _resolve_insights_from_eda_result(eda_ledger, "multicollinearity_vif", {
            "findings": ["VIF computed for 8 features."],
            "warnings": ["VIF > 10: BMI, weight"],
            "stats": {"vif": [("BMI", 23.4), ("weight", 18.7)]},
        }, "VIF (Multicollinearity)")

        insight = eda_ledger.get("eda_corr_cluster_bmi_weight_waist")
        assert insight.resolved is True
        assert "VIF" in insight.resolved_by
        assert insight.resolved_on_page == "02_EDA"
        assert insight.resolution_details["action_type"] == "diagnostic_analysis"
        assert insight.resolution_details["method"] == "multicollinearity_vif"
        assert "vif" in insight.resolution_details["stats"]

    def test_leakage_scan_resolves_leakage_insight(self, eda_ledger):
        _resolve_insights_from_eda_result(eda_ledger, "leakage_scan", {
            "findings": ["No obvious leakage candidates detected"],
            "warnings": [],
        }, "Leakage Detection")

        insight = eda_ledger.get("eda_leakage_cholesterol")
        assert insight.resolved is True
        assert "Leakage" in insight.resolved_by

    def test_missingness_scan_resolves_missing_insight(self, eda_ledger):
        _resolve_insights_from_eda_result(eda_ledger, "missingness_scan", {
            "findings": ["3 features have >20% missing. Pattern appears MAR."],
            "warnings": ["Consider MICE for MAR data."],
        }, "Missingness Deep Dive")

        insight = eda_ledger.get("eda_missing_severe")
        assert insight.resolved is True
        assert "Missingness" in insight.resolved_by

    def test_target_profile_resolves_skew_insight(self, eda_ledger):
        _resolve_insights_from_eda_result(eda_ledger, "target_profile", {
            "findings": ["Target is right-skewed (skew=2.3). Log transform recommended."],
            "warnings": [],
        }, "Target Profile")

        insight = eda_ledger.get("eda_target_skew")
        assert insight.resolved is True
        assert "Target Profile" in insight.resolved_by

    def test_sufficiency_check_resolves_sufficiency_insight(self, eda_ledger):
        _resolve_insights_from_eda_result(eda_ledger, "data_sufficiency_check", {
            "findings": ["n/p ratio = 9.0. Below threshold."],
            "warnings": ["Consider regularization."],
        }, "Data Sufficiency")

        insight = eda_ledger.get("eda_sufficiency_borderline")
        assert insight.resolved is True
        assert "Data Sufficiency" in insight.resolved_by

    def test_unknown_action_does_nothing(self, eda_ledger):
        _resolve_insights_from_eda_result(eda_ledger, "unknown_action", {
            "findings": ["test"],
        }, "Unknown")

        for insight in eda_ledger.insights:
            assert insight.resolved is False

    def test_already_resolved_not_overwritten(self, eda_ledger):
        # First run
        _resolve_insights_from_eda_result(eda_ledger, "multicollinearity_vif", {
            "findings": ["First run"], "warnings": [], "stats": {},
        }, "VIF Run 1")

        first_msg = eda_ledger.get("eda_corr_cluster_bmi_weight_waist").resolved_by

        # Second run
        _resolve_insights_from_eda_result(eda_ledger, "multicollinearity_vif", {
            "findings": ["Second run"], "warnings": [], "stats": {},
        }, "VIF Run 2")

        assert eda_ledger.get("eda_corr_cluster_bmi_weight_waist").resolved_by == first_msg

    def test_multiple_prefix_matches(self):
        """Multiple insights with the same prefix should all be resolved."""
        ledger = InsightLedger()
        ledger.upsert(Insight(
            id="eda_leakage_cholesterol", source_page="02_EDA",
            category="data_quality", severity="warning",
            finding="Cholesterol leakage", implication="test",
        ))
        ledger.upsert(Insight(
            id="eda_leakage_hdl", source_page="02_EDA",
            category="data_quality", severity="warning",
            finding="HDL leakage", implication="test",
        ))

        _resolve_insights_from_eda_result(ledger, "leakage_scan", {
            "findings": ["2 leakage candidates found"],
            "warnings": ["Exclude both"],
        }, "Leakage Detection")

        assert ledger.get("eda_leakage_cholesterol").resolved is True
        assert ledger.get("eda_leakage_hdl").resolved is True


class TestNarrativeWithEDAAnalyses:

    def test_vif_resolution_in_narrative(self, eda_ledger):
        _resolve_insights_from_eda_result(eda_ledger, "multicollinearity_vif", {
            "findings": ["VIF computed for 8 features."],
            "warnings": ["VIF > 10: BMI, weight"],
            "stats": {"vif": [("BMI", 23.4), ("weight", 18.7)]},
        }, "VIF (Multicollinearity)")

        # Auto-acknowledge the rest so they are available for discussion
        eda_ledger.auto_acknowledge_gate("Preprocessing", source_pages=["02_EDA"])

        narratives = eda_ledger.to_manuscript_narrative()
        all_text = " ".join(narratives.values())

        # VIF resolution should appear
        assert "collinearity" in all_text.lower() or "VIF" in all_text
        assert "diagnostic" in all_text.lower() or "analysis" in all_text.lower()

    def test_full_eda_lifecycle_narrative(self, eda_ledger):
        """All resolution types should appear in the narrative after EDA."""
        # Resolve VIF
        _resolve_insights_from_eda_result(eda_ledger, "multicollinearity_vif", {
            "findings": ["VIF computed"], "warnings": ["High VIF detected"], "stats": {},
        }, "VIF Analysis")

        # Resolve leakage
        _resolve_insights_from_eda_result(eda_ledger, "leakage_scan", {
            "findings": ["No leakage detected"], "warnings": [],
        }, "Leakage Scan")

        # Auto-acknowledge the rest (missing, skew, sufficiency)
        count = eda_ledger.auto_acknowledge_gate("Preprocessing", source_pages=["02_EDA"])
        assert count == 3  # missing_severe, target_skew, sufficiency_borderline

        narratives = eda_ledger.to_manuscript_narrative()
        all_text = " ".join(narratives.values())

        # Resolved analyses
        assert "VIF" in all_text or "collinearity" in all_text.lower()
        assert "Leakage" in all_text or "leakage" in all_text.lower()

        discussion_points = eda_ledger.discussion_points_for_manuscript()
        assert any("borderline" in text.lower() for text in discussion_points["limitations"])
