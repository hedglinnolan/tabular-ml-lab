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
