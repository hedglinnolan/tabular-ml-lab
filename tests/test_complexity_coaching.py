"""Tests for post-training complexity coaching (issue #87).

These tests verify the detection functions in ml/model_coach.py that surface
coaching when complex models don't meaningfully beat simple ones.
"""
from ml.model_coach import (
    run_post_training_diagnostics,
    _detect_prefer_simpler,
    _detect_low_overall_performance,
    _detect_high_cv_variance,
)


# ── Prefer Simpler ──────────────────────────────────────────────────────────

def test_prefer_simpler_detected_regression():
    """Ridge within 5% of XGBoost → should trigger prefer-simpler."""
    results = {
        'ridge': {'metrics': {'RMSE': 0.42, 'R2': 0.55}},
        'xgb_reg': {'metrics': {'RMSE': 0.41, 'R2': 0.57}},
    }
    findings = _detect_prefer_simpler(results, 'regression', tolerance=0.05)
    assert len(findings) == 1
    assert findings[0]['id'] == 'train_prefer_simpler'
    assert findings[0]['severity'] == 'warning'
    assert 'Ridge' in findings[0]['finding']
    assert 'reviewer' in findings[0]['finding'].lower()
    meta = findings[0]['metadata']
    assert meta['simple_best_model'] == 'ridge'
    assert meta['complex_best_model'] == 'xgb_reg'


def test_prefer_simpler_not_detected_regression():
    """Ridge much worse than XGBoost → should NOT trigger."""
    results = {
        'ridge': {'metrics': {'RMSE': 0.60, 'R2': 0.30}},
        'xgb_reg': {'metrics': {'RMSE': 0.41, 'R2': 0.57}},
    }
    findings = _detect_prefer_simpler(results, 'regression', tolerance=0.05)
    assert len(findings) == 0


def test_prefer_simpler_detected_classification():
    """Logistic within 5% of LightGBM → should trigger."""
    results = {
        'logreg': {'metrics': {'F1': 0.78, 'Accuracy': 0.80}},
        'lgbm_clf': {'metrics': {'F1': 0.80, 'Accuracy': 0.82}},
    }
    findings = _detect_prefer_simpler(results, 'classification', tolerance=0.05)
    assert len(findings) == 1
    assert findings[0]['id'] == 'train_prefer_simpler'
    assert 'Logistic' in findings[0]['finding']


def test_prefer_simpler_not_detected_classification():
    """Logistic much worse than LightGBM → should NOT trigger."""
    results = {
        'logreg': {'metrics': {'F1': 0.50, 'Accuracy': 0.55}},
        'lgbm_clf': {'metrics': {'F1': 0.80, 'Accuracy': 0.82}},
    }
    findings = _detect_prefer_simpler(results, 'classification', tolerance=0.05)
    assert len(findings) == 0


def test_prefer_simpler_no_simple_models():
    """Only complex models trained → no comparison possible."""
    results = {
        'xgb_reg': {'metrics': {'RMSE': 0.41}},
        'nn': {'metrics': {'RMSE': 0.42}},
    }
    findings = _detect_prefer_simpler(results, 'regression')
    assert len(findings) == 0


def test_prefer_simpler_no_complex_models():
    """Only simple models trained → no comparison possible."""
    results = {
        'ridge': {'metrics': {'RMSE': 0.42}},
        'lasso': {'metrics': {'RMSE': 0.43}},
    }
    findings = _detect_prefer_simpler(results, 'regression')
    assert len(findings) == 0


# ── Low Overall Performance ─────────────────────────────────────────────────

def test_low_performance_detected_regression():
    """Best R² < 0.15 → should trigger."""
    results = {
        'ridge': {'metrics': {'RMSE': 0.90, 'R2': 0.10}},
        'rf': {'metrics': {'RMSE': 0.88, 'R2': 0.12}},
    }
    findings = _detect_low_overall_performance(results, 'regression')
    assert len(findings) == 1
    assert findings[0]['id'] == 'train_low_performance'
    assert findings[0]['severity'] == 'opportunity'


def test_low_performance_not_detected_regression():
    """Best R² >= 0.15 → should NOT trigger."""
    results = {
        'ridge': {'metrics': {'RMSE': 0.50, 'R2': 0.55}},
    }
    findings = _detect_low_overall_performance(results, 'regression')
    assert len(findings) == 0


def test_low_performance_detected_classification():
    """Best AUC < 0.60 → should trigger."""
    results = {
        'logreg': {'metrics': {'AUC': 0.55, 'Accuracy': 0.60}},
    }
    findings = _detect_low_overall_performance(results, 'classification')
    assert len(findings) == 1
    assert findings[0]['id'] == 'train_low_performance'


def test_low_performance_not_detected_classification():
    """Best AUC >= 0.60 → should NOT trigger."""
    results = {
        'logreg': {'metrics': {'AUC': 0.75, 'Accuracy': 0.80}},
    }
    findings = _detect_low_overall_performance(results, 'classification')
    assert len(findings) == 0


# ── High CV Variance ────────────────────────────────────────────────────────

def test_cv_variance_detected():
    """CV std exceeds 50% of model performance range → should trigger."""
    results = {
        'ridge': {
            'metrics': {'RMSE': 0.42},
            'cv_results': {'mean': 0.43, 'std': 0.08},
        },
        'xgb_reg': {
            'metrics': {'RMSE': 0.37},
            'cv_results': {'mean': 0.38, 'std': 0.06},
        },
    }
    # range = 0.42 - 0.37 = 0.05; max std = 0.08 > 0.05 * 0.5 = 0.025
    findings = _detect_high_cv_variance(results, 'regression')
    assert len(findings) == 1
    assert findings[0]['id'] == 'train_cv_variance'
    assert findings[0]['severity'] == 'info'


def test_cv_variance_not_detected():
    """CV std small relative to model gap → should NOT trigger."""
    results = {
        'ridge': {
            'metrics': {'RMSE': 0.50},
            'cv_results': {'mean': 0.51, 'std': 0.01},
        },
        'xgb_reg': {
            'metrics': {'RMSE': 0.35},
            'cv_results': {'mean': 0.36, 'std': 0.01},
        },
    }
    # range = 0.50 - 0.35 = 0.15; max std = 0.01 < 0.15 * 0.5 = 0.075
    findings = _detect_high_cv_variance(results, 'regression')
    assert len(findings) == 0


def test_cv_variance_no_cv_results():
    """No CV results → should NOT trigger."""
    results = {
        'ridge': {'metrics': {'RMSE': 0.42}},
        'xgb_reg': {'metrics': {'RMSE': 0.41}},
    }
    findings = _detect_high_cv_variance(results, 'regression')
    assert len(findings) == 0


# ── Full Diagnostic Pipeline ────────────────────────────────────────────────

def test_run_post_training_diagnostics_multiple():
    """Multiple patterns detected at once."""
    results = {
        'ridge': {
            'metrics': {'RMSE': 0.92, 'R2': 0.08},
            'cv_results': {'mean': 0.93, 'std': 0.05},
        },
        'xgb_reg': {
            'metrics': {'RMSE': 0.91, 'R2': 0.09},
            'cv_results': {'mean': 0.92, 'std': 0.04},
        },
    }
    findings = run_post_training_diagnostics(results, 'regression')
    ids = {f['id'] for f in findings}
    # Should detect: prefer_simpler (Ridge ~= XGBoost) AND low_performance (R2 < 0.15)
    # CV variance: range=0.01, max_std=0.05, 0.05 > 0.01*0.5=0.005 → also triggered
    assert 'train_prefer_simpler' in ids
    assert 'train_low_performance' in ids
    assert 'train_cv_variance' in ids


def test_no_triggers_when_complex_clearly_wins():
    """Complex model clearly better, good performance, low CV noise."""
    results = {
        'ridge': {
            'metrics': {'RMSE': 0.50, 'R2': 0.55},
            'cv_results': {'mean': 0.51, 'std': 0.01},
        },
        'xgb_reg': {
            'metrics': {'RMSE': 0.30, 'R2': 0.80},
            'cv_results': {'mean': 0.31, 'std': 0.01},
        },
    }
    findings = run_post_training_diagnostics(results, 'regression')
    assert len(findings) == 0
