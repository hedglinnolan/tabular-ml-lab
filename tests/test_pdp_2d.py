"""Tests for 2D partial dependence computation and narrative (issue #86).

These tests verify the 2D PDP integration without requiring Streamlit
or a full model training pipeline.
"""
import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import partial_dependence

from ml.plot_narrative import narrative_partial_dependence_2d


# ── Narrative function tests ───────────────────────────────────────────────

def test_narrative_2d_no_interaction():
    """Zero interaction magnitude produces additive narrative."""
    nar = narrative_partial_dependence_2d("A × B", 0.0, 5.0, model_name="Ridge")
    assert "additive" in nar.lower()
    assert "Ridge" in nar


def test_narrative_2d_strong_interaction():
    """Large interaction produces 'strong' narrative."""
    nar = narrative_partial_dependence_2d("age × waist", 3.0, 5.0)
    assert "strong" in nar.lower()
    assert "age × waist" in nar


def test_narrative_2d_moderate_interaction():
    """Moderate interaction produces 'moderate' narrative."""
    nar = narrative_partial_dependence_2d("A × B", 1.0, 5.0)
    assert "moderate" in nar.lower()


def test_narrative_2d_mild_interaction():
    """Small interaction produces 'mild' narrative."""
    nar = narrative_partial_dependence_2d("A × B", 0.3, 5.0)
    assert "mild" in nar.lower()


def test_narrative_2d_includes_magnitude():
    """Narrative should report the interaction magnitude value."""
    nar = narrative_partial_dependence_2d("X × Y", 2.5, 8.0, model_name="GBM")
    assert "2.5000" in nar
    assert "GBM" in nar


# ── sklearn 2D PDP integration tests ──────────────────────────────────────

@pytest.fixture
def trained_gbr():
    """Train a simple GBR with a known interaction (x0 * x1)."""
    rng = np.random.RandomState(42)
    X = rng.randn(200, 3)
    # Target has interaction: x0 * x1
    y = X[:, 0] * X[:, 1] + 0.5 * X[:, 2] + rng.randn(200) * 0.1
    model = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
    model.fit(X, y)
    return model, X


def test_sklearn_2d_pdp_shape(trained_gbr):
    """2D PDP should return a 2D grid of predictions."""
    model, X = trained_gbr
    result = partial_dependence(
        model, X, features=[(0, 1)], kind='average', grid_resolution=10,
    )
    assert result['average'][0].shape == (10, 10)
    assert len(result['grid_values']) == 2
    assert len(result['grid_values'][0]) == 10
    assert len(result['grid_values'][1]) == 10


def test_sklearn_2d_pdp_interaction_detected(trained_gbr):
    """The interaction x0*x1 should produce a non-additive 2D PDP surface."""
    model, X = trained_gbr
    # 1D PDPs
    pd_0 = partial_dependence(model, X, features=[0], kind='average')
    pd_1 = partial_dependence(model, X, features=[1], kind='average')
    # 2D PDP
    pd_01 = partial_dependence(model, X, features=[(0, 1)], kind='average', grid_resolution=10)

    range_2d = float(np.max(pd_01['average'][0]) - np.min(pd_01['average'][0]))
    range_1d_0 = float(np.max(pd_0['average'][0]) - np.min(pd_0['average'][0]))
    range_1d_1 = float(np.max(pd_1['average'][0]) - np.min(pd_1['average'][0]))
    interaction = max(0.0, range_2d - range_1d_0 - range_1d_1)
    # With x0*x1 interaction, there should be a non-trivial interaction effect
    assert interaction > 0.01, f"Expected interaction > 0.01, got {interaction}"


def test_sklearn_2d_pdp_no_interaction(trained_gbr):
    """Features 0 and 2 have no interaction — effect should be mostly additive."""
    model, X = trained_gbr
    pd_0 = partial_dependence(model, X, features=[0], kind='average')
    pd_2 = partial_dependence(model, X, features=[2], kind='average')
    pd_02 = partial_dependence(model, X, features=[(0, 2)], kind='average', grid_resolution=10)

    range_2d = float(np.max(pd_02['average'][0]) - np.min(pd_02['average'][0]))
    range_1d_0 = float(np.max(pd_0['average'][0]) - np.min(pd_0['average'][0]))
    range_1d_2 = float(np.max(pd_2['average'][0]) - np.min(pd_2['average'][0]))
    interaction = max(0.0, range_2d - range_1d_0 - range_1d_2)
    # Interaction for non-interacting features should be small
    # (GBR can pick up spurious interactions, so use a generous threshold)
    assert interaction < range_2d * 0.5, f"Interaction {interaction} unexpectedly large relative to 2D range {range_2d}"


def test_subsampling_produces_valid_result(trained_gbr):
    """Subsampling to fewer rows should still produce valid 2D PDP."""
    model, X = trained_gbr
    rng = np.random.RandomState(43)
    idx = rng.choice(X.shape[0], 50, replace=False)
    X_sub = X[idx]
    result = partial_dependence(
        model, X_sub, features=[(0, 1)], kind='average', grid_resolution=10,
    )
    assert result['average'][0].shape == (10, 10)
    assert np.all(np.isfinite(result['average'][0]))
