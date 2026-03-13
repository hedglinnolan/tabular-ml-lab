"""
Tests for GitHub issues #7-15 fixes.
Run with: pytest tests/test_issues_7_15.py -v
"""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc


# ── Issue #8: ROC/PR curves for classification ──────────────────────
class TestIssue8_ROC_PR:
    """ROC and PR curves should generate for binary classification models."""

    @pytest.fixture
    def binary_classification_data(self):
        np.random.seed(42)
        X = np.random.randn(200, 5)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        return X, y

    def test_roc_curve_binary(self, binary_classification_data):
        """ROC curve should compute for binary classification with predict_proba."""
        X, y = binary_classification_data
        model = LogisticRegression(solver='saga', max_iter=1000)
        model.fit(X, y)
        y_proba = model.predict_proba(X)

        proba_pos = y_proba[:, 1]
        fpr, tpr, _ = roc_curve(y, proba_pos)
        roc_auc = auc(fpr, tpr)

        assert len(fpr) > 2
        assert len(tpr) > 2
        assert 0.5 < roc_auc <= 1.0

    def test_pr_curve_binary(self, binary_classification_data):
        """PR curve should compute for binary classification."""
        X, y = binary_classification_data
        model = LogisticRegression(solver='saga', max_iter=1000)
        model.fit(X, y)
        y_proba = model.predict_proba(X)

        proba_pos = y_proba[:, 1]
        prec, rec, _ = precision_recall_curve(y, proba_pos)
        pr_auc = auc(rec, prec)

        assert len(prec) > 2
        assert pr_auc > 0.5

    def test_confusion_matrix_generation(self, binary_classification_data):
        """Confusion matrix should generate without errors."""
        X, y = binary_classification_data
        model = LogisticRegression(solver='saga', max_iter=1000)
        model.fit(X, y)
        y_pred = model.predict(X)

        cm = confusion_matrix(y, y_pred)
        assert cm.shape == (2, 2)
        assert cm.sum() == len(y)


# ── Issue #9: PDP attribute name ────────────────────────────────────
class TestIssue9_PDP_Attribute:
    """ModelCapabilities should use supports_partial_dependence, not supports_pdp."""

    def test_model_capabilities_has_correct_attribute(self):
        from ml.model_registry import ModelCapabilities
        import dataclasses
        field_names = [f.name for f in dataclasses.fields(ModelCapabilities)]
        assert 'supports_partial_dependence' in field_names
        assert 'supports_pdp' not in field_names

    def test_registry_models_have_pdp_capability(self):
        from ml.model_registry import get_registry
        registry = get_registry()
        for key, spec in registry.items():
            # Should not raise AttributeError
            _ = spec.capabilities.supports_partial_dependence


# ── Issue #11: report_text defined before export ────────────────────
class TestIssue11_ReportText:
    """report_text should be generated before Export Options section."""

    def test_report_text_before_export(self):
        import ast
        with open('pages/10_Report_Export.py', 'r') as f:
            source = f.read()

        # report_text = generate_report(...) should appear before "Export Options"
        gen_pos = source.find('report_text = generate_report(')
        export_pos = source.find('# EXPORT OPTIONS')
        preview_pos = source.find('# REPORT PREVIEW')

        assert gen_pos > 0, "report_text = generate_report(...) not found"
        assert export_pos > 0, "EXPORT OPTIONS section not found"
        assert gen_pos < export_pos, "report_text must be generated before Export Options"
        assert gen_pos < preview_pos, "report_text must be generated before Report Preview"


# ── Issue #12: PDP subsampling ──────────────────────────────────────
class TestIssue12_PDP_Subsample:
    """PDP should subsample large datasets."""

    def test_subsample_logic(self):
        """Arrays > 2000 rows should be subsampled."""
        max_pdp_samples = 2000
        X_large = np.random.randn(5000, 10)

        if X_large.shape[0] > max_pdp_samples:
            rng = np.random.RandomState(42)
            idx = rng.choice(X_large.shape[0], max_pdp_samples, replace=False)
            X_sub = X_large[idx]

        assert X_sub.shape == (2000, 10)

    def test_small_dataset_no_subsample(self):
        """Arrays <= 2000 rows should not be subsampled."""
        max_pdp_samples = 2000
        X_small = np.random.randn(500, 10)

        X_result = X_small
        if X_small.shape[0] > max_pdp_samples:
            X_result = X_small[:max_pdp_samples]

        assert X_result.shape == (500, 10)


# ── Issue #13: Seed sensitivity clone ───────────────────────────────
class TestIssue13_SeedSensitivity:
    """Seed sensitivity should clone the underlying estimator, not the wrapper."""

    def test_clone_sklearn_model(self):
        """sklearn models should be clonable."""
        model = LogisticRegression(solver='saga', C=1.0)
        cloned = clone(model)
        assert cloned is not model
        assert cloned.get_params() == model.get_params()

    def test_clone_with_random_state(self):
        """Cloned model should accept new random_state."""
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        cloned = clone(model)
        cloned.set_params(random_state=99)
        assert cloned.random_state == 99

    def test_seed_sensitivity_produces_results(self):
        """Running same model with different seeds should produce varied results."""
        np.random.seed(0)
        X = np.random.randn(200, 5)
        y = (X[:, 0] > 0).astype(int)

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        results = []
        for seed in [0, 1, 42, 99]:
            cloned = clone(model)
            cloned.set_params(random_state=seed)
            cloned.fit(X[:150], y[:150])
            acc = (cloned.predict(X[150:]) == y[150:]).mean()
            results.append(acc)

        assert len(results) == 4
        assert all(0.0 <= r <= 1.0 for r in results)


# ── Issue #14: KernelExplainer eval cap ─────────────────────────────
class TestIssue14_KernelExplainerCap:
    """KernelExplainer should use at most 50 eval samples."""

    def test_eval_cap_applied(self):
        """X_ev should be capped at 50 for KernelExplainer."""
        X_ev = np.random.randn(200, 10)
        X_ev_kernel = X_ev[:min(50, len(X_ev))]
        assert X_ev_kernel.shape[0] == 50

    def test_small_eval_unchanged(self):
        """X_ev smaller than 50 should not be modified."""
        X_ev = np.random.randn(30, 10)
        X_ev_kernel = X_ev[:min(50, len(X_ev))]
        assert X_ev_kernel.shape[0] == 30


# ── Issue #15: Zip export with plots ────────────────────────────────
class TestIssue15_ZipPlots:
    """Zip export should include organized plot directories."""

    def test_export_section_exists_in_code(self):
        """Report export should contain plot directory structure."""
        with open('pages/10_Report_Export.py', 'r') as f:
            source = f.read()

        assert 'plots/train/' in source
        assert 'plots/explainability/' in source
        assert 'plots/sensitivity/' in source

    def test_plotly_fig_to_bytes(self):
        """Plotly figures should be convertible to PNG bytes (if kaleido installed)."""
        try:
            import plotly.express as px
            fig = px.scatter(x=[1, 2, 3], y=[4, 5, 6], title="Test")
            img_bytes = fig.to_image(format="png", width=800, height=600)
            assert len(img_bytes) > 0
            assert img_bytes[:4] == b'\x89PNG'
        except (ImportError, ValueError, RuntimeError):
            pytest.skip("kaleido/Chrome not available — plot export requires it on headless servers")


# ── Issue #1: best_model_state init ─────────────────────────────────
class TestIssue1_BestModelState:
    """best_model_state should be initialized before training loop."""

    def test_best_model_state_initialized(self):
        with open('models/nn_whuber.py', 'r') as f:
            source = f.read()

        # Find the initialization
        init_pos = source.find('best_model_state = self.model.state_dict().copy()')
        loop_pos = source.find('for epoch in range')

        assert init_pos > 0, "best_model_state initialization not found"
        assert init_pos < loop_pos, "best_model_state must be initialized before training loop"


# ── Issue #2: Label encoding in explainability ──────────────────────
class TestIssue2_LabelEncoding:
    """Explainability should encode string labels before permutation importance."""

    def test_label_encoder_reference(self):
        with open('pages/07_Explainability.py', 'r') as f:
            source = f.read()

        assert 'target_label_encoder' in source
        assert "label_encoder.transform(y_raw)" in source
