"""Tests for comparative context layer and overfit detection (#92, #87).

Tests cover:
- _detect_overfit() regression and classification variants
- _detect_overfit() edge cases (no train_metrics, small gap)
- Performance spread computation logic
- run_post_training_diagnostics includes overfit findings
"""
from ml.model_coach import (
    run_post_training_diagnostics,
    _detect_overfit,
    _detect_prefer_simpler,
)


# ── Overfit Detection ─────────────────────────────────────────────────────────

def test_overfit_detected_regression():
    """Train R² much higher than test R² → should trigger overfit warning."""
    results = {
        'xgb_reg': {
            'metrics': {'RMSE': 12.0, 'R2': 0.27},
            'train_metrics': {'RMSE': 3.0, 'R2': 0.95},
        },
    }
    findings = _detect_overfit(results, 'regression', gap_threshold=0.10)
    assert len(findings) == 1
    assert findings[0]['id'] == 'train_overfit_xgb_reg'
    assert findings[0]['severity'] == 'warning'
    assert 'overfitting' in findings[0]['finding'].lower()
    meta = findings[0]['metadata']
    assert meta['train_score'] == 0.95
    assert meta['test_score'] == 0.27
    assert meta['gap'] > 0.10


def test_overfit_not_detected_small_gap_regression():
    """Train R² close to test R² → should NOT trigger."""
    results = {
        'ridge': {
            'metrics': {'RMSE': 12.0, 'R2': 0.27},
            'train_metrics': {'RMSE': 11.5, 'R2': 0.30},
        },
    }
    findings = _detect_overfit(results, 'regression', gap_threshold=0.10)
    assert len(findings) == 0


def test_overfit_detected_classification():
    """Train F1 much higher than test F1 → should trigger."""
    results = {
        'lgbm_clf': {
            'metrics': {'F1': 0.60, 'Accuracy': 0.65},
            'train_metrics': {'F1': 0.98, 'Accuracy': 0.99},
        },
    }
    findings = _detect_overfit(results, 'classification', gap_threshold=0.10)
    assert len(findings) == 1
    assert findings[0]['id'] == 'train_overfit_lgbm_clf'
    assert findings[0]['metadata']['gap'] > 0.10


def test_overfit_not_detected_classification_small_gap():
    """Train F1 close to test F1 → should NOT trigger."""
    results = {
        'logreg': {
            'metrics': {'F1': 0.78, 'Accuracy': 0.80},
            'train_metrics': {'F1': 0.82, 'Accuracy': 0.83},
        },
    }
    findings = _detect_overfit(results, 'classification', gap_threshold=0.10)
    assert len(findings) == 0


def test_overfit_no_train_metrics():
    """Missing train_metrics → should NOT trigger (graceful skip)."""
    results = {
        'ridge': {
            'metrics': {'RMSE': 12.0, 'R2': 0.27},
        },
    }
    findings = _detect_overfit(results, 'regression')
    assert len(findings) == 0


def test_overfit_empty_train_metrics():
    """Empty train_metrics dict → should NOT trigger."""
    results = {
        'ridge': {
            'metrics': {'RMSE': 12.0, 'R2': 0.27},
            'train_metrics': {},
        },
    }
    findings = _detect_overfit(results, 'regression')
    assert len(findings) == 0


def test_overfit_multiple_models():
    """Multiple models, only one overfitting → one finding."""
    results = {
        'ridge': {
            'metrics': {'RMSE': 12.0, 'R2': 0.27},
            'train_metrics': {'RMSE': 11.5, 'R2': 0.30},
        },
        'xgb_reg': {
            'metrics': {'RMSE': 12.0, 'R2': 0.27},
            'train_metrics': {'RMSE': 2.0, 'R2': 0.97},
        },
    }
    findings = _detect_overfit(results, 'regression', gap_threshold=0.10)
    assert len(findings) == 1
    assert findings[0]['metadata']['model_key'] == 'xgb_reg'


def test_overfit_custom_threshold():
    """Custom gap_threshold = 0.50 → only extreme overfitting triggers."""
    results = {
        'xgb_reg': {
            'metrics': {'RMSE': 12.0, 'R2': 0.27},
            'train_metrics': {'RMSE': 3.0, 'R2': 0.60},
        },
    }
    # gap = 0.33, threshold = 0.50 → should NOT trigger
    findings = _detect_overfit(results, 'regression', gap_threshold=0.50)
    assert len(findings) == 0


# ── Integration with run_post_training_diagnostics ────────────────────────────

def test_diagnostics_includes_overfit():
    """run_post_training_diagnostics should include overfit findings."""
    results = {
        'ridge': {
            'metrics': {'RMSE': 12.0, 'R2': 0.27},
            'train_metrics': {'RMSE': 11.5, 'R2': 0.30},
        },
        'xgb_reg': {
            'metrics': {'RMSE': 12.0, 'R2': 0.27},
            'train_metrics': {'RMSE': 2.0, 'R2': 0.97},
        },
    }
    findings = run_post_training_diagnostics(results, 'regression')
    overfit_findings = [f for f in findings if f['id'].startswith('train_overfit_')]
    assert len(overfit_findings) == 1
    assert overfit_findings[0]['metadata']['model_key'] == 'xgb_reg'


def test_diagnostics_overfit_and_prefer_simpler_coexist():
    """Both overfit and prefer-simpler can trigger simultaneously."""
    results = {
        'ridge': {
            'metrics': {'RMSE': 0.42, 'R2': 0.55},
            'train_metrics': {'RMSE': 0.40, 'R2': 0.58},
        },
        'xgb_reg': {
            'metrics': {'RMSE': 0.41, 'R2': 0.57},
            'train_metrics': {'RMSE': 0.10, 'R2': 0.99},
        },
    }
    findings = run_post_training_diagnostics(results, 'regression', tolerance=0.05)
    ids = [f['id'] for f in findings]
    assert 'train_prefer_simpler' in ids
    assert 'train_overfit_xgb_reg' in ids


# ── Overfit finding metadata ─────────────────────────────────────────────────

def test_overfit_metadata_fields():
    """Overfit finding should have all required metadata fields."""
    results = {
        'histgb_reg': {
            'metrics': {'RMSE': 10.0, 'R2': 0.30},
            'train_metrics': {'RMSE': 2.0, 'R2': 0.95},
        },
    }
    findings = _detect_overfit(results, 'regression')
    assert len(findings) == 1
    meta = findings[0]['metadata']
    assert 'model_key' in meta
    assert 'model_name' in meta
    assert 'train_score' in meta
    assert 'test_score' in meta
    assert 'gap' in meta
    assert 'metric_name' in meta
    assert meta['metric_name'] == 'R²'


def test_overfit_finding_has_recommended_action():
    """Overfit finding should include actionable recommendation."""
    results = {
        'nn': {
            'metrics': {'RMSE': 15.0, 'R2': 0.20},
            'train_metrics': {'RMSE': 1.0, 'R2': 0.99},
        },
    }
    findings = _detect_overfit(results, 'regression')
    assert len(findings) == 1
    assert 'regularis' in findings[0]['recommended_action'].lower()
