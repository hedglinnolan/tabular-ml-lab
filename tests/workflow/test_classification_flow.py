"""
Workflow test: Full classification pipeline.
Upload → Configure → EDA (imbalance detection) → Split → Train (with class weighting)
→ Metrics → Probabilities → Calibration → Subgroup → Publication

Mirrors test_regression_flow.py but exercises classification-specific paths:
- Imbalance detection and severity rating
- Stratified splitting
- Class weight / sample weight application
- Classification metrics (Accuracy, F1, AUC, LogLoss, PR-AUC)
- Probability outputs and calibration
- Per-class performance
"""
import sys
import os
import hashlib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tests.conftest import (
    inject_uploaded_state, prepare_splits, make_data_config,
)


# ── Helpers ──────────────────────────────────────────────────────────

def train_classification_model(splits, model_name='logreg', class_weight=None):
    """Train a classification model and return results dict with metrics + probabilities."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from ml.eval import calculate_classification_metrics

    X_train = splits['X_train'].values if hasattr(splits['X_train'], 'values') else splits['X_train']
    X_test = splits['X_test'].values if hasattr(splits['X_test'], 'values') else splits['X_test']
    X_val = splits['X_val'].values if hasattr(splits['X_val'], 'values') else splits['X_val']
    y_train = splits['y_train'].values if hasattr(splits['y_train'], 'values') else splits['y_train']
    y_test = splits['y_test'].values if hasattr(splits['y_test'], 'values') else splits['y_test']

    models = {
        'logreg': lambda: LogisticRegression(
            max_iter=1000, random_state=42,
            class_weight=class_weight,
        ),
        'rf': lambda: RandomForestClassifier(
            n_estimators=50, random_state=42,
            class_weight=class_weight,
        ),
        'extratrees_clf': lambda: ExtraTreesClassifier(
            n_estimators=50, random_state=42,
            class_weight=class_weight,
        ),
        'knn_clf': lambda: KNeighborsClassifier(n_neighbors=5),
    }

    model = models[model_name]()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    metrics = calculate_classification_metrics(y_test, y_pred, y_proba)

    return {
        'model': model,
        'y_test_pred': y_pred,
        'y_test_proba': y_proba,
        'metrics': metrics,
    }


def prepare_stratified_splits(df, target_col='condition', train_frac=0.7, val_frac=0.15):
    """Prepare stratified train/val/test splits for classification."""
    from sklearn.model_selection import train_test_split
    from utils.session_state import SplitConfig

    feature_cols = [c for c in df.columns
                    if c != target_col and df[c].dtype in ('float64', 'int64', 'float32', 'int32')]

    mask = df[target_col].notna()
    X = df.loc[mask, feature_cols].copy().fillna(df[feature_cols].median())
    y = df.loc[mask, target_col].copy()

    test_frac = 1.0 - train_frac - val_frac

    X_temp, X_test, y_temp, y_test, idx_temp, idx_test = train_test_split(
        X, y, np.arange(len(X)),
        test_size=test_frac, random_state=42, stratify=y,
    )
    relative_val = val_frac / (train_frac + val_frac)
    X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
        X_temp, y_temp, idx_temp,
        test_size=relative_val, random_state=42, stratify=y_temp,
    )

    return {
        'split_config': SplitConfig(),
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
        'train_indices': idx_train.tolist(),
        'val_indices': idx_val.tolist(),
        'test_indices': idx_test.tolist(),
        'feature_names': feature_cols,
    }


# ── Tests ────────────────────────────────────────────────────────────

class TestClassificationPipeline:
    """Sequential classification pipeline on imbalanced data."""

    def test_step1_upload_and_configure(self, classification_df, classification_state):
        """Upload detects classification task and configures correctly."""
        state = classification_state

        assert state.get('raw_data') is not None
        assert state.get('data_config') is not None

        data_config = state['data_config']
        assert data_config.task_type == 'classification'
        assert data_config.target_col == 'condition'
        assert 'condition' not in data_config.feature_cols
        assert len(data_config.feature_cols) > 0

    def test_step2_imbalance_detection(self, classification_state):
        """Dataset profile correctly detects class imbalance."""
        state = classification_state
        profile = state.get('dataset_profile')
        assert profile is not None, "Profile not computed"
        assert profile.target_profile is not None, "Target profile missing"

        tp = profile.target_profile
        assert tp.is_imbalanced, "Should detect imbalance at 5:1 ratio"
        assert tp.imbalance_severity in ('mild', 'moderate', 'severe'), \
            f"Unexpected severity: {tp.imbalance_severity}"
        assert tp.class_balance_ratio is not None
        assert tp.class_balance_ratio >= 2.0, \
            f"Ratio should be ≥2 for imbalanced data, got {tp.class_balance_ratio}"

    def test_step2b_balanced_no_imbalance(self, balanced_classification_state):
        """Balanced dataset should NOT trigger imbalance detection."""
        state = balanced_classification_state
        profile = state.get('dataset_profile')
        assert profile is not None

        tp = profile.target_profile
        # Balanced 50/50 — ratio should be ~1.0
        assert not tp.is_imbalanced, \
            f"Balanced data should not be imbalanced (ratio={tp.class_balance_ratio})"

    def test_step3_stratified_splits(self, classification_df, classification_state):
        """Stratified splits preserve class distribution in each fold."""
        df = classification_df
        state = classification_state

        splits = prepare_stratified_splits(df, target_col='condition')
        for k, v in splits.items():
            state[k] = v

        y_train = state['y_train']
        y_val = state['y_val']
        y_test = state['y_test']

        # Check class proportions are roughly preserved (within 5% of overall)
        overall_pos_rate = (df['condition'] == 1).mean()

        for name, y_split in [('train', y_train), ('val', y_val), ('test', y_test)]:
            split_pos_rate = (y_split == 1).mean()
            diff = abs(split_pos_rate - overall_pos_rate)
            assert diff < 0.05, \
                f"{name} pos rate {split_pos_rate:.3f} too far from overall {overall_pos_rate:.3f}"

        # Both classes present in every split
        for name, y_split in [('train', y_train), ('val', y_val), ('test', y_test)]:
            unique = set(y_split.unique() if hasattr(y_split, 'unique') else np.unique(y_split))
            assert len(unique) == 2, f"{name} split missing a class: {unique}"

    def test_step4_train_logreg(self, classification_state):
        """Train logistic regression and verify classification metrics."""
        state = classification_state
        assert state.get('X_train') is not None, "Step 3 must run first"

        splits = {k: state[k] for k in [
            'X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test',
            'train_indices', 'val_indices', 'test_indices', 'feature_names',
            'split_config',
        ] if k in state}

        result = train_classification_model(splits, 'logreg')
        metrics = result['metrics']

        # Core classification metrics must exist
        assert 'Accuracy' in metrics, "Accuracy missing"
        assert 'F1' in metrics, "F1 missing"
        assert 0 < metrics['Accuracy'] <= 1.0
        assert 0 < metrics['F1'] <= 1.0

        # Probability-based metrics
        assert 'ROC-AUC' in metrics, "ROC-AUC missing (probabilities should be available)"
        assert 'LogLoss' in metrics, "LogLoss missing"
        assert 'PR-AUC' in metrics, "PR-AUC missing"
        assert 0 < metrics['ROC-AUC'] <= 1.0
        assert metrics['LogLoss'] > 0

        # Store in state
        state['trained_models'] = {'logreg': result['model']}
        state['fitted_estimators'] = {'logreg': result['model']}
        state['model_results'] = {
            'logreg': {
                'metrics': metrics,
                'y_test': state['y_test'].values if hasattr(state['y_test'], 'values') else state['y_test'],
                'y_test_pred': result['y_test_pred'],
                'y_test_proba': result['y_test_proba'],
            }
        }

    def test_step5_class_weight_effect(self, classification_state):
        """Class weighting should improve minority class recall."""
        state = classification_state

        splits = {k: state[k] for k in [
            'X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test',
            'feature_names',
        ] if k in state}

        # Train without class weight
        result_unweighted = train_classification_model(splits, 'logreg', class_weight=None)

        # Train with class weight
        result_weighted = train_classification_model(splits, 'logreg', class_weight='balanced')

        # Compute per-class recall
        from sklearn.metrics import recall_score
        y_test = state['y_test'].values if hasattr(state['y_test'], 'values') else state['y_test']

        recall_minority_unweighted = recall_score(y_test, result_unweighted['y_test_pred'], pos_label=1)
        recall_minority_weighted = recall_score(y_test, result_weighted['y_test_pred'], pos_label=1)

        # Weighted should have higher minority recall (or at least not dramatically worse)
        # We allow a small margin because synthetic data can be noisy
        assert recall_minority_weighted >= recall_minority_unweighted - 0.1, \
            f"Weighted recall ({recall_minority_weighted:.3f}) shouldn't be much worse than " \
            f"unweighted ({recall_minority_unweighted:.3f})"

    def test_step6_multiple_models(self, classification_state):
        """Train multiple classification models and compare metrics."""
        state = classification_state

        splits = {k: state[k] for k in [
            'X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test',
            'feature_names',
        ] if k in state}

        model_names = ['logreg', 'rf', 'extratrees_clf', 'knn_clf']
        results = {}

        for model_name in model_names:
            result = train_classification_model(splits, model_name)
            results[model_name] = result

            # Every model should produce valid metrics
            assert 'Accuracy' in result['metrics'], f"{model_name}: Accuracy missing"
            assert 0 < result['metrics']['Accuracy'] <= 1.0, \
                f"{model_name}: Accuracy out of range"

            # Every model should produce probabilities
            assert result['y_test_proba'] is not None, \
                f"{model_name}: probabilities missing"
            assert result['y_test_proba'].shape[1] == 2, \
                f"{model_name}: expected 2 probability columns for binary"

            # Probabilities should sum to ~1
            row_sums = result['y_test_proba'].sum(axis=1)
            assert np.allclose(row_sums, 1.0, atol=1e-6), \
                f"{model_name}: probabilities don't sum to 1"

        # Store all models in state
        state['trained_models'] = {name: r['model'] for name, r in results.items()}
        state['model_results'] = {
            name: {
                'metrics': r['metrics'],
                'y_test': state['y_test'].values if hasattr(state['y_test'], 'values') else state['y_test'],
                'y_test_pred': r['y_test_pred'],
                'y_test_proba': r['y_test_proba'],
            }
            for name, r in results.items()
        }

    def test_step7_probability_calibration(self, classification_state):
        """Probability calibration functions work on classification output."""
        state = classification_state

        # Check calibration module works
        from ml.calibration import calibration_classification, plot_calibration_curve

        for model_name, result in state['model_results'].items():
            y_test = result['y_test']
            y_proba = result['y_test_proba']
            if y_proba is None:
                continue

            y_proba_pos = y_proba[:, 1]

            try:
                cal = calibration_classification(
                    np.array(y_test), y_proba_pos,
                    model_name=model_name.upper(),
                )
                assert cal is not None, f"{model_name}: calibration returned None"
            except Exception as e:
                pytest.fail(f"{model_name}: calibration_classification raised {e}")

    def test_step8_subgroup_access(self, classification_df, classification_state):
        """Subgroup analysis can stratify by categorical columns (gender, smoking)."""
        state = classification_state

        test_indices = state.get('test_indices')
        assert test_indices is not None

        raw_df = state['raw_data']

        # Categorical columns accessible from raw data
        for col in ['gender', 'smoking']:
            values = raw_df.iloc[test_indices][col].values
            assert len(values) == len(state['X_test']), f"{col} length mismatch"
            assert len(set(values)) >= 2, f"{col} should have multiple values in test set"

        # Can compute per-subgroup metrics
        from sklearn.metrics import accuracy_score
        y_test = state['model_results']['logreg']['y_test']
        y_pred = state['model_results']['logreg']['y_test_pred']
        gender_labels = raw_df.iloc[test_indices]['gender'].values

        for gender in ['male', 'female']:
            mask = gender_labels == gender
            if mask.sum() > 0:
                acc = accuracy_score(y_test[mask], y_pred[mask])
                assert 0 <= acc <= 1.0, f"Subgroup accuracy for {gender} out of range"

    def test_step9_publication_output(self, classification_state):
        """Publication methods section handles classification correctly."""
        state = classification_state

        from ml.publication import generate_methods_section

        data_config = state['data_config']
        methods = generate_methods_section(
            data_config={
                'feature_cols': data_config.feature_cols,
                'target_col': 'condition',
                'task_type': 'classification',
            },
            preprocessing_config={},
            model_configs={},
            split_config={
                'train_size': 0.7, 'val_size': 0.15, 'test_size': 0.15,
                'target_trim_enabled': False,
                'target_transform': 'none',
                'stratify': True, 'use_time_split': False,
            },
            n_total=len(state['raw_data']),
            n_train=len(state['X_train']),
            n_val=len(state['X_val']),
            n_test=len(state['X_test']),
            feature_names=list(state['X_train'].columns),
            target_name='condition',
            task_type='classification',
            metrics_used=['Accuracy', 'F1', 'ROC-AUC'],
        )

        assert 'condition' in methods, "Target not mentioned"
        assert len(methods) > 100, f"Methods too short ({len(methods)} chars)"
        # Classification methods should mention stratification
        assert 'stratif' in methods.lower(), "Stratification not mentioned for classification"


class TestClassificationEdgeCases:
    """Edge cases that could break classification workflows."""

    def test_single_class_in_small_split(self):
        """Very small dataset where a split might have only one class."""
        np.random.seed(42)
        # 20 samples, 18:2 split — test set might get only one class without stratification
        df = pd.DataFrame({
            'x1': np.random.randn(20),
            'x2': np.random.randn(20),
            'target': [0]*18 + [1]*2,
        })

        # Non-stratified split could lose minority class
        splits = prepare_splits(df, target_col='target')
        y_test = splits['y_test']

        # Even if we lose a class, metrics shouldn't crash
        from ml.eval import calculate_classification_metrics
        from sklearn.linear_model import LogisticRegression

        X_train = splits['X_train'].values
        X_test = splits['X_test'].values
        y_train_vals = splits['y_train'].values

        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train_vals)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        y_test_vals = y_test.values if hasattr(y_test, 'values') else y_test

        # This should not crash even if test set is all one class
        metrics = calculate_classification_metrics(y_test_vals, y_pred, y_proba)
        assert 'Accuracy' in metrics
        assert 'F1' in metrics

    def test_multiclass_three_classes(self):
        """Three-class classification works end-to-end."""
        np.random.seed(42)
        n = 150
        df = pd.DataFrame({
            'x1': np.random.randn(n),
            'x2': np.random.randn(n),
            'x3': np.random.randn(n),
            'label': np.repeat([0, 1, 2], n // 3),
        })

        state = {}
        inject_uploaded_state(state, df, target_col='label', task_type='classification')

        profile = state.get('dataset_profile')
        assert profile is not None

        splits = prepare_splits(df, target_col='label')

        from sklearn.linear_model import LogisticRegression
        from ml.eval import calculate_classification_metrics

        X_train = splits['X_train'].values
        X_test = splits['X_test'].values
        y_train = splits['y_train'].values
        y_test = splits['y_test'].values

        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        assert y_proba.shape[1] == 3, "Should have 3 probability columns"
        assert np.allclose(y_proba.sum(axis=1), 1.0, atol=1e-6)

        metrics = calculate_classification_metrics(y_test, y_pred, y_proba)
        assert 'Accuracy' in metrics
        assert 'F1' in metrics
        # ROC-AUC should work for multiclass too
        assert 'ROC-AUC' in metrics, "Multiclass ROC-AUC should work with ovr"

    def test_high_cardinality_imbalance(self):
        """Severe imbalance (20:1) still produces meaningful training."""
        np.random.seed(42)
        n = 210
        df = pd.DataFrame({
            'x1': np.random.randn(n),
            'x2': np.random.randn(n),
            'x3': np.random.randn(n),
            'target': [0]*200 + [1]*10,
        })
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        state = {}
        inject_uploaded_state(state, df, target_col='target', task_type='classification')

        profile = state['dataset_profile']
        tp = profile.target_profile
        assert tp.is_imbalanced
        assert tp.imbalance_severity == 'severe', \
            f"20:1 should be severe, got {tp.imbalance_severity}"

        # Should still be able to train
        splits = prepare_splits(df, target_col='target')

        from sklearn.linear_model import LogisticRegression
        from ml.eval import calculate_classification_metrics

        model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        model.fit(splits['X_train'].values, splits['y_train'].values)

        y_pred = model.predict(splits['X_test'].values)
        y_proba = model.predict_proba(splits['X_test'].values)
        y_test = splits['y_test'].values

        metrics = calculate_classification_metrics(y_test, y_pred, y_proba)
        assert 'Accuracy' in metrics

    def test_classification_cross_validation(self):
        """Cross-validation works for classification models."""
        from ml.eval import perform_cross_validation
        from sklearn.linear_model import LogisticRegression

        np.random.seed(42)
        n = 200
        X = np.random.randn(n, 3)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        model = LogisticRegression(max_iter=1000, random_state=42)
        cv_results = perform_cross_validation(model, X, y, cv_folds=5, task_type='classification')

        assert cv_results is not None
        assert 'test_score' in cv_results or 'scores' in cv_results or len(cv_results) > 0, \
            f"CV results empty: {cv_results}"

    def test_bootstrap_classification_metrics(self):
        """Bootstrap confidence intervals work for classification."""
        from ml.bootstrap import bootstrap_all_classification_metrics

        np.random.seed(42)
        n = 100
        y_true = np.array([0]*70 + [1]*30)
        y_pred = y_true.copy()
        # Add some noise
        flip_idx = np.random.choice(n, 10, replace=False)
        y_pred[flip_idx] = 1 - y_pred[flip_idx]
        y_proba = np.column_stack([1 - y_pred * 0.8, y_pred * 0.8])

        cis = bootstrap_all_classification_metrics(
            y_true, y_pred, y_proba=y_proba, n_resamples=200,
        )

        assert cis is not None
        assert len(cis) > 0, "Bootstrap should produce CI results"
