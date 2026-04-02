"""Tests for the complete insight lifecycle: created → resolved | acknowledged.

Validates that:
1. Insights can be acknowledged (reviewed and accepted without action)
2. Auto-acknowledge gates work at workflow transitions
3. to_manuscript_narrative() includes acknowledged insights and strengths
4. Resolved insights are not downgraded to acknowledged
5. The full lifecycle produces complete manuscript narratives
"""

import pytest
from utils.insight_ledger import Insight, InsightLedger


def _make_eda_insight(insight_id: str, finding: str, severity: str = "warning",
                      category: str = "data_quality", model_scope=None) -> Insight:
    """Helper to create an EDA insight."""
    return Insight(
        id=insight_id,
        source_page="02_EDA",
        category=category,
        severity=severity,
        finding=finding,
        implication="Test implication",
        recommended_action="Test action",
        relevant_pages=["05_Preprocess"],
        model_scope=model_scope or [],
    )


def _make_opportunity_insight(insight_id: str, finding: str) -> Insight:
    """Helper to create a positive observation (opportunity) insight."""
    return Insight(
        id=insight_id,
        source_page="02_EDA",
        category="data_quality",
        severity="info",
        finding=finding,
        implication="Favorable for analysis",
        recommended_action="",
        relevant_pages=[],
    )


class TestAcknowledgment:
    """Test the acknowledge() method."""

    def test_acknowledge_unresolved_insight(self):
        ledger = InsightLedger()
        ledger.upsert(_make_eda_insight("eda_small_sample", "Small sample size (n=200)"))

        result = ledger.acknowledge("eda_small_sample", "Accepted — using regularization")
        assert result is True

        insight = ledger.get("eda_small_sample")
        assert insight.acknowledged is True
        assert insight.acknowledged_by == "Accepted — using regularization"
        assert insight.acknowledged_at is not None
        assert insight.resolved is False  # Not resolved, just acknowledged

    def test_acknowledge_does_not_downgrade_resolution(self):
        """Resolved insights should not be changed by acknowledge()."""
        ledger = InsightLedger()
        ledger.upsert(_make_eda_insight("eda_skew", "BMI is skewed"))
        ledger.resolve("eda_skew", "Applied log1p", "05_Preprocess",
                       {"action_type": "power_transform", "method": "log1p"})

        # Try to acknowledge a resolved insight
        ledger.acknowledge("eda_skew", "Should not overwrite")

        insight = ledger.get("eda_skew")
        assert insight.resolved is True
        assert insight.acknowledged is False  # Should NOT be acknowledged

    def test_acknowledge_nonexistent_returns_false(self):
        ledger = InsightLedger()
        result = ledger.acknowledge("nonexistent", "test")
        assert result is False

    def test_get_acknowledged_returns_only_acknowledged(self):
        ledger = InsightLedger()
        ledger.upsert(_make_eda_insight("eda_a", "Finding A"))
        ledger.upsert(_make_eda_insight("eda_b", "Finding B"))
        ledger.upsert(_make_eda_insight("eda_c", "Finding C"))

        ledger.resolve("eda_a", "Fixed", "05_Preprocess")
        ledger.acknowledge("eda_b", "Accepted")
        # eda_c left unresolved and unacknowledged

        acknowledged = ledger.get_acknowledged()
        assert len(acknowledged) == 1
        assert acknowledged[0].id == "eda_b"


class TestAutoAcknowledgeGate:
    """Test auto_acknowledge_gate() at workflow transitions."""

    def test_gate_acknowledges_unresolved_eda_insights(self):
        ledger = InsightLedger()
        ledger.upsert(_make_eda_insight("eda_small_sample", "Small sample"))
        ledger.upsert(_make_eda_insight("eda_leakage_col1", "Potential leakage in col1"))
        ledger.upsert(_make_eda_insight("eda_skew", "BMI is skewed"))

        # Resolve one
        ledger.resolve("eda_skew", "Applied transform", "05_Preprocess")

        # Gate: preprocessing
        count = ledger.auto_acknowledge_gate(
            "Proceeded to preprocessing",
            source_pages=["02_EDA"],
        )

        assert count == 2  # small_sample and leakage (not skew — already resolved)

        small = ledger.get("eda_small_sample")
        assert small.acknowledged is True
        assert small.acknowledged_by == "Proceeded to preprocessing"

        leakage = ledger.get("eda_leakage_col1")
        assert leakage.acknowledged is True

        skew = ledger.get("eda_skew")
        assert skew.resolved is True
        assert skew.acknowledged is False  # Resolved, not acknowledged

    def test_gate_skips_already_acknowledged(self):
        ledger = InsightLedger()
        ledger.upsert(_make_eda_insight("eda_a", "Finding A"))
        ledger.acknowledge("eda_a", "Manually accepted")

        count = ledger.auto_acknowledge_gate("Gate 2")
        assert count == 0  # Already acknowledged

    def test_gate_with_source_page_filter(self):
        ledger = InsightLedger()
        ledger.upsert(_make_eda_insight("eda_a", "EDA finding"))
        ledger.upsert(Insight(
            id="preprocess_issue",
            source_page="05_Preprocess",
            category="methodology",
            severity="warning",
            finding="Preprocessing issue",
            implication="test",
        ))

        count = ledger.auto_acknowledge_gate(
            "Proceeded to training",
            source_pages=["02_EDA"],
        )

        assert count == 1  # Only EDA insight
        assert ledger.get("eda_a").acknowledged is True
        assert ledger.get("preprocess_issue").acknowledged is False

    def test_gate_without_filter_acknowledges_all(self):
        ledger = InsightLedger()
        ledger.upsert(_make_eda_insight("eda_a", "Finding A"))
        ledger.upsert(Insight(
            id="other_b",
            source_page="05_Preprocess",
            category="methodology",
            severity="warning",
            finding="Other finding",
            implication="test",
        ))

        count = ledger.auto_acknowledge_gate("Training completed")
        assert count == 2


class TestNarrativeWithAcknowledgments:
    """Test that acknowledged insights route to discussion-only helpers."""

    def test_narrative_includes_acknowledged_limitations(self):
        ledger = InsightLedger()
        ledger.upsert(_make_eda_insight(
            "eda_small_sample", "Sample size (n=200) below recommended threshold",
            severity="warning",
        ))
        ledger.acknowledge("eda_small_sample", "Proceeded with regularization")

        discussion_points = ledger.discussion_points_for_manuscript()

        assert any("Sample size" in text for text in discussion_points["limitations"]), \
            f"Acknowledged limitation missing from discussion points: {discussion_points}"
        assert ledger.to_manuscript_narrative() == {}, "Acknowledged-only insights should not populate methods narrative"

    def test_narrative_includes_strengths(self):
        ledger = InsightLedger()
        ledger.upsert(_make_opportunity_insight(
            "eda_opportunity_balanced", "Balanced class distributions observed"
        ))
        ledger.upsert(_make_opportunity_insight(
            "eda_opportunity_clean_data", "Minimal missing data (<1% overall)"
        ))
        # Auto-acknowledge at gate
        ledger.auto_acknowledge_gate("Proceeded to preprocessing", source_pages=["02_EDA"])

        discussion_points = ledger.discussion_points_for_manuscript()

        assert any("Balanced class" in text for text in discussion_points["strengths"]), \
            f"Strength details missing from discussion points: {discussion_points}"
        assert ledger.to_manuscript_narrative() == {}, "Strength-only insights should not populate methods narrative"

    def test_narrative_separates_resolved_and_acknowledged(self):
        """Resolved and acknowledged insights should both appear but with different framing."""
        ledger = InsightLedger()

        # Resolved insight
        ledger.upsert(_make_eda_insight("eda_skew", "BMI distribution is right-skewed"))
        ledger.resolve("eda_skew", "Applied log1p transform", "05_Preprocess",
                       {"action_type": "power_transform", "method": "log1p"})

        # Acknowledged limitation
        ledger.upsert(_make_eda_insight("eda_small_sample", "Small sample (n=150)"))
        ledger.acknowledge("eda_small_sample", "Accepted with regularization")

        # Strength
        ledger.upsert(_make_opportunity_insight(
            "eda_opportunity_strong_signal", "Strong univariate associations detected"
        ))
        ledger.auto_acknowledge_gate("Preprocessing", source_pages=["02_EDA"])

        narratives = ledger.to_manuscript_narrative()
        all_text = " ".join(narratives.values())
        discussion_points = ledger.discussion_points_for_manuscript()

        assert "BMI distribution" in all_text, "Resolved insight missing"
        assert any("Small sample" in text for text in discussion_points["limitations"]), "Acknowledged limitation missing"
        assert any("Strong univariate associations" in text for text in discussion_points["strengths"]), "Strength missing"

    def test_unresolved_unacknowledged_still_invisible(self):
        """Insights that are neither resolved nor acknowledged should not appear."""
        ledger = InsightLedger()
        ledger.upsert(_make_eda_insight("eda_orphan", "This was never reviewed"))

        narratives = ledger.to_manuscript_narrative()
        all_text = " ".join(narratives.values()) if narratives else ""

        assert "never reviewed" not in all_text


class TestFullLifecycleIntegration:
    """Simulate a realistic workflow and verify narrative completeness."""

    def test_realistic_eda_to_training_lifecycle(self):
        """A realistic workflow: EDA produces insights, some resolved, rest auto-acknowledged."""
        ledger = InsightLedger()

        # EDA auto-detectors produce insights
        ledger.upsert(_make_eda_insight("eda_skew_group", "3 features exhibit right skewness",
                                        severity="warning", model_scope=["linear", "neural"]))
        ledger.upsert(_make_eda_insight("eda_missing_moderate", "5 features have 2-10% missing values"))
        ledger.upsert(_make_eda_insight("eda_small_sample", "Sample size (n=180) is borderline",
                                        severity="warning"))
        ledger.upsert(_make_eda_insight("eda_leakage_cholesterol",
                                        "Cholesterol shows r=0.94 with target — possible leakage"))
        ledger.upsert(_make_opportunity_insight("eda_opportunity_balanced",
                                                "Balanced class distributions"))
        ledger.upsert(_make_opportunity_insight("eda_opportunity_clean_data",
                                                "No features exceed 10% missingness"))

        # Preprocess resolves some
        ledger.resolve("eda_skew_group", "Yeo-Johnson for Ridge/MLP; raw for RF", "05_Preprocess",
                       {"action_type": "power_transform", "per_model": {"ridge": "yeo-johnson", "rf": "none"}})
        ledger.resolve("eda_missing_moderate", "Median imputation configured", "05_Preprocess",
                       {"action_type": "imputation", "method": "median"})

        # Preprocessing gate: auto-acknowledge remaining EDA insights
        count = ledger.auto_acknowledge_gate("Proceeded to preprocessing", source_pages=["02_EDA"])
        assert count == 4  # small_sample, leakage, 2 opportunities

        # Training gate: should find nothing new to acknowledge (all from EDA already handled)
        count2 = ledger.auto_acknowledge_gate("Training completed")
        assert count2 == 0

        # Check final state
        assert ledger.get("eda_skew_group").resolved is True
        assert ledger.get("eda_missing_moderate").resolved is True
        assert ledger.get("eda_small_sample").acknowledged is True
        assert ledger.get("eda_leakage_cholesterol").acknowledged is True
        assert ledger.get("eda_opportunity_balanced").acknowledged is True

        # Methods narrative should focus on resolved workflow actions
        narratives = ledger.to_manuscript_narrative()
        all_text = " ".join(narratives.values())
        discussion_points = ledger.discussion_points_for_manuscript()

        assert "skewness" in all_text.lower() or "Yeo-Johnson" in all_text
        assert "imput" in all_text.lower() or "median" in all_text.lower()
        assert "noted and accepted" not in all_text
        assert "favorable" not in all_text.lower()
        assert any("Cholesterol" in text or "leakage" in text.lower() for text in discussion_points["limitations"])
        assert any("Balanced class" in text for text in discussion_points["strengths"])

    def test_all_resolved_no_acknowledged(self):
        """If everything is resolved, no acknowledged section appears."""
        ledger = InsightLedger()
        ledger.upsert(_make_eda_insight("eda_skew", "Skewed"))
        ledger.resolve("eda_skew", "Fixed", "05_Preprocess",
                       {"action_type": "power_transform", "method": "log1p"})

        narratives = ledger.to_manuscript_narrative()
        all_text = " ".join(narratives.values())

        assert "noted and accepted" not in all_text
        assert "favorable" not in all_text.lower()

    def test_serialization_preserves_acknowledgment(self):
        """Acknowledged state survives to_dict → from_dict round-trip."""
        ledger = InsightLedger()
        ledger.upsert(_make_eda_insight("eda_a", "Finding A"))
        ledger.acknowledge("eda_a", "Accepted")

        serialized = [i.to_dict() for i in ledger.insights]
        restored = InsightLedger.from_list(serialized)

        insight = restored.get("eda_a")
        assert insight.acknowledged is True
        assert insight.acknowledged_by == "Accepted"
        assert insight.acknowledged_at is not None
