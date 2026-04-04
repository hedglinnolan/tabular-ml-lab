"""Tests for stratified residual analysis (issue #85).

These tests verify analyze_residuals_stratified() in ml/eval.py
and the narrative function in ml/plot_narrative.py.
"""
import numpy as np
from ml.eval import analyze_residuals_stratified
from ml.plot_narrative import narrative_residuals_stratified


# ── Core analysis ──────────────────────────────────────────────────────────

def test_default_quintile_bins():
    """Default 5 bins with uniform data should produce 5 entries."""
    np.random.seed(42)
    y_true = np.linspace(0, 100, 100)
    y_pred = y_true + 2  # constant over-prediction
    result = analyze_residuals_stratified(y_true, y_pred)
    assert len(result["bins"]) == 5
    for b in result["bins"]:
        assert b["n"] > 0
        assert b["bias_direction"] == "over"
        assert abs(b["mean_bias"] - 2.0) < 0.5


def test_custom_edges():
    """Custom edges override default quintile binning."""
    y_true = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=float)
    y_pred = y_true + 1
    result = analyze_residuals_stratified(y_true, y_pred, custom_edges=[0, 50, 100])
    assert len(result["bins"]) == 2
    assert result["bins"][0]["n"] + result["bins"][1]["n"] == 10


def test_over_prediction_detected():
    """Systematic over-prediction should be labelled 'over'."""
    y_true = np.arange(50, dtype=float)
    y_pred = y_true + 5
    result = analyze_residuals_stratified(y_true, y_pred, n_bins=3)
    for b in result["bins"]:
        if b["n"] > 0:
            assert b["bias_direction"] == "over"
            assert b["mean_bias"] > 0


def test_under_prediction_detected():
    """Systematic under-prediction should be labelled 'under'."""
    y_true = np.arange(50, dtype=float)
    y_pred = y_true - 5
    result = analyze_residuals_stratified(y_true, y_pred, n_bins=3)
    for b in result["bins"]:
        if b["n"] > 0:
            assert b["bias_direction"] == "under"
            assert b["mean_bias"] < 0


def test_balanced_when_unbiased():
    """Near-zero bias should be labelled 'balanced'."""
    y_true = np.arange(100, dtype=float)
    y_pred = y_true.copy()  # perfect predictions
    result = analyze_residuals_stratified(y_true, y_pred, n_bins=5)
    for b in result["bins"]:
        if b["n"] > 0:
            assert b["bias_direction"] == "balanced"
            assert abs(b["mean_bias"]) < 1e-10


def test_mae_and_rmse_correct():
    """MAE and RMSE should be computed correctly for a simple case."""
    y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    y_pred = np.array([12.0, 18.0, 33.0, 37.0, 55.0])
    result = analyze_residuals_stratified(y_true, y_pred, custom_edges=[0, 25, 55])
    bin0 = result["bins"][0]  # [0, 25): values 10, 20
    # errors: +2, -2
    assert abs(bin0["mae"] - 2.0) < 1e-6
    assert abs(bin0["mean_bias"] - 0.0) < 1e-6
    assert abs(bin0["rmse"] - 2.0) < 1e-6


def test_too_few_samples():
    """Fewer than 3 valid samples returns empty bins."""
    result = analyze_residuals_stratified(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    assert result["bins"] == []
    assert result["overall_bias_direction"] == "balanced"


def test_nan_handling():
    """NaN values should be filtered out."""
    y_true = np.array([1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    y_pred = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, np.nan], dtype=float)
    result = analyze_residuals_stratified(y_true, y_pred, n_bins=3)
    total_n = sum(b["n"] for b in result["bins"])
    assert total_n == 8  # two NaN rows removed


def test_custom_bin_count():
    """n_bins=3 should produce 3 bins."""
    y_true = np.linspace(0, 100, 60)
    y_pred = y_true + 1
    result = analyze_residuals_stratified(y_true, y_pred, n_bins=3)
    assert len(result["bins"]) == 3


def test_overall_bias_direction():
    """overall_bias_direction reflects the bin with the worst bias."""
    y_true = np.concatenate([np.ones(20) * 10, np.ones(20) * 50])
    y_pred = np.concatenate([np.ones(20) * 10, np.ones(20) * 60])  # over-predict high range
    result = analyze_residuals_stratified(y_true, y_pred, n_bins=3)
    assert result["overall_bias_direction"] == "over"


# ── Narrative ──────────────────────────────────────────────────────────────

def test_narrative_empty_stats():
    """Empty bins produce empty narrative."""
    assert narrative_residuals_stratified({"bins": []}) == ""


def test_narrative_mixed_bias():
    """Mixed bias directions should be mentioned."""
    stats = {"bins": [
        {"range": "0–50", "n": 10, "mae": 2.0, "rmse": 2.5, "mean_bias": 2.0, "bias_direction": "over"},
        {"range": "50–100", "n": 10, "mae": 3.0, "rmse": 3.5, "mean_bias": -3.0, "bias_direction": "under"},
    ]}
    nar = narrative_residuals_stratified(stats, model_name="Ridge")
    assert "Ridge" in nar
    assert "mixed bias" in nar.lower() or "over-predicting" in nar.lower()


def test_narrative_uniform_over():
    """All over-prediction bins should produce over-prediction narrative."""
    stats = {"bins": [
        {"range": "0–50", "n": 10, "mae": 2.0, "rmse": 2.5, "mean_bias": 2.0, "bias_direction": "over"},
        {"range": "50–100", "n": 10, "mae": 2.0, "rmse": 2.5, "mean_bias": 1.5, "bias_direction": "over"},
    ]}
    nar = narrative_residuals_stratified(stats)
    assert "over-predict" in nar.lower()


def test_narrative_worst_bin_mentioned():
    """Narrative should mention the bin with the largest bias."""
    stats = {"bins": [
        {"range": "0–25", "n": 10, "mae": 1.0, "rmse": 1.2, "mean_bias": 0.5, "bias_direction": "over"},
        {"range": "25–50", "n": 10, "mae": 5.0, "rmse": 6.0, "mean_bias": -5.0, "bias_direction": "under"},
    ]}
    nar = narrative_residuals_stratified(stats)
    assert "25–50" in nar or "25.00–50.00" in nar
