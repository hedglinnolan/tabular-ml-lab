"""Tests for model-scoped preprocessing coaching (issue #46).

Verifies that generate_preprocessing_insights() produces correctly
scoped recommendations based on selected models and data profile.
"""
from dataclasses import dataclass, field
from typing import List
import importlib
import sys
from unittest.mock import MagicMock

# Mock streamlit before importing insight_ledger
if "streamlit" not in sys.modules:
    sys.modules["streamlit"] = MagicMock()

from ml.model_coach import generate_preprocessing_insights

# Constants copied from insight_ledger to avoid streamlit import issues
MODEL_FAMILY_LINEAR = "linear"
MODEL_FAMILY_TREE = "tree"
MODEL_FAMILY_NEURAL = "neural"
MODEL_FAMILY_DISTANCE = "distance"


@dataclass
class FakeProfile:
    highly_skewed_features: List[str] = field(default_factory=list)
    features_with_outliers: List[str] = field(default_factory=list)
    n_features_with_missing: int = 0


# ── Skewness scoping ──────────────────────────────────────────────────────

def test_skewness_scoped_to_linear_not_tree():
    """Skewness insight should target linear models but not tree models."""
    profile = FakeProfile(highly_skewed_features=["BMI", "age"])
    insights = generate_preprocessing_insights(["ridge", "rf"], profile)
    skew = [i for i in insights if i["id"] == "preprocess_skewness_transform"]
    assert len(skew) == 1
    assert MODEL_FAMILY_LINEAR in skew[0]["model_scope"]
    assert MODEL_FAMILY_TREE not in skew[0]["model_scope"]


def test_skewness_includes_neural_and_distance():
    """Neural and distance models are also affected by skewness."""
    profile = FakeProfile(highly_skewed_features=["x1"])
    insights = generate_preprocessing_insights(["nn", "knn_reg", "rf"], profile)
    skew = [i for i in insights if i["id"] == "preprocess_skewness_transform"]
    assert len(skew) == 1
    assert MODEL_FAMILY_NEURAL in skew[0]["model_scope"]
    assert MODEL_FAMILY_DISTANCE in skew[0]["model_scope"]
    assert MODEL_FAMILY_TREE not in skew[0]["model_scope"]


def test_no_skewness_insight_when_tree_only():
    """Trees-only selection should not produce skewness insight."""
    profile = FakeProfile(highly_skewed_features=["x1", "x2"])
    insights = generate_preprocessing_insights(["rf", "xgb_reg"], profile)
    skew = [i for i in insights if i["id"] == "preprocess_skewness_transform"]
    assert len(skew) == 0


def test_no_skewness_insight_when_no_skew():
    """No skewed features → no skewness insight."""
    profile = FakeProfile(highly_skewed_features=[])
    insights = generate_preprocessing_insights(["ridge"], profile)
    skew = [i for i in insights if i["id"] == "preprocess_skewness_transform"]
    assert len(skew) == 0


# ── Outlier scoping ────────────────────────────────────────────────────────

def test_outlier_scoped_correctly():
    """Outlier insight targets linear/neural/distance but not tree."""
    profile = FakeProfile(features_with_outliers=["income"])
    insights = generate_preprocessing_insights(["ridge", "rf", "nn"], profile)
    outlier = [i for i in insights if i["id"] == "preprocess_outlier_handling"]
    assert len(outlier) == 1
    assert MODEL_FAMILY_LINEAR in outlier[0]["model_scope"]
    assert MODEL_FAMILY_NEURAL in outlier[0]["model_scope"]
    assert MODEL_FAMILY_TREE not in outlier[0]["model_scope"]


def test_no_outlier_insight_when_tree_only():
    """Trees handle outliers natively → no outlier insight."""
    profile = FakeProfile(features_with_outliers=["x1"])
    insights = generate_preprocessing_insights(["rf", "lgbm_reg"], profile)
    outlier = [i for i in insights if i["id"] == "preprocess_outlier_handling"]
    assert len(outlier) == 0


# ── Feature scaling scoping ───────────────────────────────────────────────

def test_scaling_insight_for_mixed_models():
    """Scaling insight should appear when scale-sensitive models are selected."""
    profile = FakeProfile()
    insights = generate_preprocessing_insights(["ridge", "rf"], profile)
    scale = [i for i in insights if i["id"] == "preprocess_feature_scaling"]
    assert len(scale) == 1
    assert MODEL_FAMILY_LINEAR in scale[0]["model_scope"]
    assert MODEL_FAMILY_TREE not in scale[0]["model_scope"]
    # Recommendation should mention trees are scale-invariant
    assert "scale-invariant" in scale[0]["recommended_action"].lower() or "no scaling" in scale[0]["recommended_action"].lower()


def test_no_scaling_insight_for_tree_only():
    """Trees don't need scaling → no scaling insight."""
    profile = FakeProfile()
    insights = generate_preprocessing_insights(["rf", "xgb_reg"], profile)
    scale = [i for i in insights if i["id"] == "preprocess_feature_scaling"]
    assert len(scale) == 0


# ── Missing data scoping ─────────────────────────────────────────────────

def test_missing_data_tree_native_insight():
    """When trees + other models selected with missing data, mention native handling."""
    profile = FakeProfile(n_features_with_missing=5)
    insights = generate_preprocessing_insights(["ridge", "rf"], profile)
    missing = [i for i in insights if i["id"] == "preprocess_missing_tree_native"]
    assert len(missing) == 1
    assert "native" in missing[0]["recommended_action"].lower()


def test_no_tree_missing_insight_without_trees():
    """Without tree models, skip the tree-native-missing insight."""
    profile = FakeProfile(n_features_with_missing=5)
    insights = generate_preprocessing_insights(["ridge", "nn"], profile)
    missing = [i for i in insights if i["id"] == "preprocess_missing_tree_native"]
    assert len(missing) == 0


# ── Edge cases ────────────────────────────────────────────────────────────

def test_empty_models_returns_empty():
    """No selected models → no insights."""
    profile = FakeProfile(highly_skewed_features=["x1"])
    assert generate_preprocessing_insights([], profile) == []


def test_no_profile_returns_empty():
    """No profile → no insights."""
    assert generate_preprocessing_insights(["ridge"], None) == []


def test_all_issues_present():
    """Full dataset with all issues should produce all relevant insights."""
    profile = FakeProfile(
        highly_skewed_features=["x1"],
        features_with_outliers=["x2"],
        n_features_with_missing=3,
    )
    insights = generate_preprocessing_insights(["ridge", "rf", "nn"], profile)
    ids = {i["id"] for i in insights}
    assert "preprocess_skewness_transform" in ids
    assert "preprocess_outlier_handling" in ids
    assert "preprocess_feature_scaling" in ids
    assert "preprocess_missing_tree_native" in ids


def test_immune_model_mentioned_in_recommendation():
    """When tree models are immune, the recommendation should say so."""
    profile = FakeProfile(highly_skewed_features=["BMI"])
    insights = generate_preprocessing_insights(["ridge", "rf"], profile)
    skew = [i for i in insights if i["id"] == "preprocess_skewness_transform"][0]
    assert "tree" in skew["recommended_action"].lower() or "natively" in skew["recommended_action"].lower()


# ── Top-pick shape awareness (2026-07 coach assessment) ──────────────────
# The old select_top_picks reacted to whatever it checked first: feature
# outliers hijacked the pick to Huber (which addresses TARGET outliers),
# p≫n and EPV were never mentioned, and "well-calibrated" was claimed at
# any event count. These tests pin the shape-first behavior.

def _profile_for(n, p, task="regression", target_outliers=False,
                 feature_outliers=False, imbalance=None, seed=0):
    import numpy as np
    import pandas as pd
    from ml.dataset_profile import compute_dataset_profile

    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
    if feature_outliers:
        for i in range(min(3, p)):
            idx = rng.choice(n, max(2, n // 25), replace=False)
            df.iloc[idx, i] = 15.0
    if task == "regression":
        y = df["f0"] * 2 + rng.normal(0, 1, n)
        if target_outliers:
            y.iloc[rng.choice(n, max(3, n // 15), replace=False)] = y.mean() + 20 * y.std()
        df["y"] = y
    else:
        df["y"] = (rng.random(n) < (imbalance or 0.5)).astype(int)
    feats = [c for c in df.columns if c != "y"]
    return compute_dataset_profile(df, "y", feats, task)


class TestTopPicksShapeAwareness:
    def test_wide_data_gets_penalized_picks_not_huber(self):
        from ml.model_coach import select_top_picks

        profile = _profile_for(40, 200)
        picks, skips, headline = select_top_picks(profile)
        keys = {p.model_key for p in picks}
        assert keys and keys <= {"lasso", "ridge", "elasticnet"}, keys
        assert "p≫n" in headline
        assert any("Tree ensembles" in name for name, _ in skips)

    def test_feature_outliers_alone_do_not_trigger_huber(self):
        from ml.model_coach import select_top_picks

        profile = _profile_for(400, 12, feature_outliers=True)
        picks, _, _ = select_top_picks(profile)
        assert picks[0].model_key != "huber", (
            "Huber addresses target outliers; feature outliers are a "
            "preprocessing concern")

    def test_target_outliers_do_trigger_huber_with_rate_cited(self):
        from ml.model_coach import select_top_picks

        profile = _profile_for(300, 8, target_outliers=True)
        picks, _, _ = select_top_picks(profile)
        assert picks[0].model_key == "huber"
        assert "outcome" in picks[0].why.lower()
        assert "%" in picks[0].why  # cites the measured rate

    def test_low_epv_keeps_lineup_small_and_cites_numbers(self):
        from ml.model_coach import select_top_picks

        profile = _profile_for(400, 12, task="classification", imbalance=0.06)
        picks, skips, headline = select_top_picks(profile)
        assert "EPV" in headline
        keys = {p.model_key for p in picks}
        assert "histgb_clf" not in keys and "rf" not in keys, (
            "low EPV must not recommend high-capacity models")
        assert any("EPV" in reason for _, reason in skips)
        # `AUDIT-014` (L43-B). This asserted `"class weights"` — a green
        # test holding the contraindicated advice in place, which is
        # `GUIDED-145`'s class one layer over: a guard pinning a
        # behavior the research says is wrong. §A5.2 is [SETTLED] that
        # the remedy for a rare outcome is penalization and adequate
        # sample size, not reweighting, and this is a LOW-EPV profile —
        # the one place the registry is most explicit.
        assert "penalize" in picks[0].preprocessing, (
            "the coach no longer names penalization for a low-EPV "
            "profile, which is §A5.2's named remedy")
        assert "class weights" not in picks[0].preprocessing, (
            "the coach recommends reweighting again")

    def test_calibration_claim_requires_events(self):
        from ml.model_coach import select_top_picks

        few = _profile_for(120, 10, task="classification", imbalance=0.3)
        picks_few, _, _ = select_top_picks(few)
        assert "well-calibrated" not in picks_few[0].why
        assert "calibration" not in picks_few[0].why.lower() or "checkable" not in picks_few[0].why.lower()

        many = _profile_for(20000, 10, task="classification", imbalance=0.5)
        picks_many, _, _ = select_top_picks(many)
        assert "calibration" in picks_many[0].why.lower()

    def test_small_n_tree_pick_carries_cv_spread_caution(self):
        from ml.model_coach import select_top_picks

        profile = _profile_for(80, 8)
        picks, _, headline = select_top_picks(profile)
        tree = [p for p in picks if p.group == "Trees/Boosting"]
        assert tree and "CV spread" in tree[0].why
        assert "80 rows" in headline

    def test_skip_reasons_carry_dataset_numbers(self):
        from ml.model_coach import select_top_picks

        profile = _profile_for(20000, 25, task="classification")
        _, skips, _ = select_top_picks(profile)
        svm = [r for name, r in skips if "SVM" in name]
        assert svm and "20,000" in svm[0]

    def test_unconstrained_dataset_has_no_alarmist_headline(self):
        from ml.model_coach import select_top_picks

        profile = _profile_for(2000, 15)
        _, _, headline = select_top_picks(profile)
        assert headline == ""


class TestPreprocessingInsightHonesty:
    def test_scaling_insight_acknowledges_defaults(self):
        from ml.model_coach import generate_preprocessing_insights

        profile = _profile_for(300, 10)
        ins = generate_preprocessing_insights(["ridge", "rf"], profile)
        scale = [i for i in ins if i["id"] == "preprocess_feature_scaling"]
        assert scale and "default" in scale[0]["recommended_action"].lower()

    def test_small_n_outlier_flags_get_detector_caveat(self):
        from ml.model_coach import generate_preprocessing_insights

        profile = _profile_for(34, 12, feature_outliers=True)
        if not profile.features_with_outliers:
            import pytest
            pytest.skip("detector did not flag this draw")
        ins = generate_preprocessing_insights(["ridge"], profile)
        outlier = [i for i in ins if i["id"] == "preprocess_outlier_handling"]
        if len(profile.features_with_outliers) > 0.3 * profile.n_features:
            assert outlier and "inspect" in outlier[0]["recommended_action"].lower()
