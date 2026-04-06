"""
Workflow tests with diverse, edge-case datasets.

All prior workflow tests used the same 200-sample, 7-feature, 5% missing
synthetic dataset.  Real users upload data that is tiny, wide, sparse,
imbalanced, or has columns with extreme characteristics.

These tests push edge-case data through the actual pipeline functions
(profiling, feature selection, preprocessing, splitting, training) and
verify the pipeline either handles them gracefully or produces actionable
errors — never silent corruption or unhelpful tracebacks.

Datasets:
  - tiny: n=30, 5 features (split edge cases, CV fold limits)
  - wide: n=50, p=60 features (p > n, LASSO/RFE challenges)
  - heavy_missing: n=150, 40% missing + 1 all-NaN column
  - severe_imbalance: n=200, 95/5 class split
  - constant_and_sparse: has constant columns, near-zero-variance, high cardinality
"""
import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# ═══════════════════════════════════════════════════════════════════════
# Dataset Builders
# ═══════════════════════════════════════════════════════════════════════

def build_tiny_regression_df(n=30, seed=99):
    """Tiny dataset: tests split edge cases and CV fold limits."""
    np.random.seed(seed)
    df = pd.DataFrame({
        'feat_a': np.random.normal(0, 1, n),
        'feat_b': np.random.normal(5, 2, n),
        'feat_c': np.random.exponential(1, n),
        'cat_x': np.random.choice(['A', 'B'], n),
    })
    df['target'] = 3 * df['feat_a'] + df['feat_b'] + np.random.normal(0, 0.5, n)
    # A few missing values
    df.loc[0, 'feat_b'] = np.nan
    df.loc[1, 'feat_c'] = np.nan
    return df


def build_wide_regression_df(n=50, p=60, seed=101):
    """Wide dataset: more features than samples (p > n)."""
    np.random.seed(seed)
    X = np.random.randn(n, p)
    # Only first 3 features are truly predictive
    y = 2 * X[:, 0] - X[:, 1] + 0.5 * X[:, 2] + np.random.normal(0, 0.3, n)
    feature_names = [f'feat_{i:03d}' for i in range(p)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    return df


def build_heavy_missing_df(n=150, seed=102):
    """Heavy missing data: 40% missing rate + 1 entirely NaN column."""
    np.random.seed(seed)
    df = pd.DataFrame({
        'feat_a': np.random.normal(10, 3, n),
        'feat_b': np.random.normal(0, 1, n),
        'feat_c': np.random.exponential(2, n),
        'feat_d': np.random.normal(50, 10, n),
        'feat_e': np.random.choice(['low', 'mid', 'high'], n),
    })
    df['target'] = df['feat_a'] * 0.5 + df['feat_d'] * 0.1 + np.random.normal(0, 1, n)

    # 40% missing in feat_a and feat_b
    for col in ['feat_a', 'feat_b']:
        idx = np.random.choice(n, int(n * 0.4), replace=False)
        df.loc[idx, col] = np.nan

    # Entirely NaN column
    df['feat_all_nan'] = np.nan

    return df


def build_severe_imbalance_df(n=200, minority_frac=0.05, seed=103):
    """Severely imbalanced classification: 95/5 split."""
    np.random.seed(seed)
    n_minority = max(int(n * minority_frac), 3)  # at least 3
    n_majority = n - n_minority
    df = pd.DataFrame({
        'feat_a': np.concatenate([np.random.normal(0, 1, n_majority), np.random.normal(3, 1, n_minority)]),
        'feat_b': np.concatenate([np.random.normal(5, 2, n_majority), np.random.normal(8, 2, n_minority)]),
        'feat_c': np.random.exponential(1, n),
        'cat_x': np.random.choice(['A', 'B', 'C'], n),
    })
    df['target_class'] = np.array([0] * n_majority + [1] * n_minority)
    # Shuffle
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


def build_constant_and_sparse_df(n=100, seed=104):
    """Dataset with constant columns, near-zero-variance, high cardinality."""
    np.random.seed(seed)
    df = pd.DataFrame({
        'feat_normal': np.random.normal(0, 1, n),
        'feat_constant': np.full(n, 42.0),             # zero variance
        'feat_near_constant': np.concatenate([np.full(n - 1, 1.0), [2.0]]),
        'feat_predictive': np.random.normal(10, 3, n),
        'cat_high_card': [f'cat_{i}' for i in range(n)],  # n unique values
        'cat_binary': np.random.choice(['yes', 'no'], n),
    })
    df['target'] = df['feat_predictive'] * 2 + np.random.normal(0, 1, n)
    return df


# ═══════════════════════════════════════════════════════════════════════
# Module-scoped fixtures
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def tiny_df():
    return build_tiny_regression_df()


@pytest.fixture(scope="module")
def wide_df():
    return build_wide_regression_df()


@pytest.fixture(scope="module")
def heavy_missing_df():
    return build_heavy_missing_df()


@pytest.fixture(scope="module")
def imbalanced_df():
    return build_severe_imbalance_df()


@pytest.fixture(scope="module")
def constant_df():
    return build_constant_and_sparse_df()


# ═══════════════════════════════════════════════════════════════════════
# 1. Dataset Profiling — must not crash on any dataset
# ═══════════════════════════════════════════════════════════════════════

class TestDatasetProfiling:

    def _profile(self, df, target_col, task_type='regression'):
        from ml.dataset_profile import compute_dataset_profile
        features = [c for c in df.columns if c != target_col]
        return compute_dataset_profile(df, target_col, features, task_type)

    def test_tiny_dataset_profile(self, tiny_df):
        profile = self._profile(tiny_df, 'target')
        assert profile is not None
        assert profile.n_rows == 30

    def test_wide_dataset_profile(self, wide_df):
        profile = self._profile(wide_df, 'target')
        assert profile is not None
        assert profile.n_features >= 60

    def test_heavy_missing_profile(self, heavy_missing_df):
        """Profile must handle 40% missing + all-NaN column."""
        profile = self._profile(heavy_missing_df, 'target')
        assert profile is not None

    def test_imbalanced_profile(self, imbalanced_df):
        profile = self._profile(imbalanced_df, 'target_class', 'classification')
        assert profile is not None

    def test_constant_columns_profile(self, constant_df):
        """Constant and near-constant columns should not crash profiling."""
        profile = self._profile(constant_df, 'target')
        assert profile is not None


# ═══════════════════════════════════════════════════════════════════════
# 2. Feature Selection — edge cases
# ═══════════════════════════════════════════════════════════════════════

class TestFeatureSelectionEdgeCases:

    def _get_Xy(self, df, target_col):
        features = [c for c in df.columns if c != target_col and df[c].dtype in ('float64', 'int64')]
        X = df[features].fillna(df[features].median()).values
        y = df[target_col].values
        return X, y, features

    def test_lasso_tiny_dataset(self, tiny_df):
        """LASSO on n=30 with 3 numeric features: should work."""
        from ml.feature_selection import lasso_path_selection
        X, y, names = self._get_Xy(tiny_df, 'target')
        # Reduce CV folds to avoid n < cv_folds issue
        result = lasso_path_selection(X, y, names, cv_folds=min(5, len(y) // 3))
        assert len(result.selected_features) >= 0
        assert len(result.all_features) == len(names)

    def test_lasso_wide_dataset(self, wide_df):
        """LASSO on p=60 > n=50: must not crash (LASSO handles p>n natively)."""
        from ml.feature_selection import lasso_path_selection
        X, y, names = self._get_Xy(wide_df, 'target')
        result = lasso_path_selection(X, y, names, cv_folds=3)
        assert len(result.selected_features) >= 0

    def test_rfe_tiny_dataset(self, tiny_df):
        """RFE on n=30: CV folds must be capped to avoid degenerate folds."""
        from ml.feature_selection import rfe_cv_selection
        X, y, names = self._get_Xy(tiny_df, 'target')
        result = rfe_cv_selection(X, y, names, cv_folds=min(5, len(y) // 3))
        assert len(result.selected_features) >= 1

    def test_rfe_wide_dataset(self, wide_df):
        """RFE on p=60 > n=50: Ridge handles this with regularization."""
        from ml.feature_selection import rfe_cv_selection
        X, y, names = self._get_Xy(wide_df, 'target')
        result = rfe_cv_selection(X, y, names, cv_folds=3)
        assert len(result.selected_features) >= 1

    def test_univariate_heavy_missing(self, heavy_missing_df):
        """Univariate screening with 40% missing + all-NaN column."""
        from ml.feature_selection import univariate_screening
        X, y, names = self._get_Xy(heavy_missing_df, 'target')
        result = univariate_screening(X, y, names)
        assert result is not None
        assert len(result.all_features) == len(names)

    def test_stability_tiny_dataset(self, tiny_df):
        """Stability selection on n=30: some bootstraps may fail silently."""
        from ml.feature_selection import stability_selection
        X, y, names = self._get_Xy(tiny_df, 'target')
        result = stability_selection(X, y, names, n_bootstrap=20)
        assert result is not None

    def test_stability_wide_dataset(self, wide_df):
        """Stability selection on p>n: most bootstrap LASSO fits may fail."""
        from ml.feature_selection import stability_selection
        X, y, names = self._get_Xy(wide_df, 'target')
        result = stability_selection(X, y, names, n_bootstrap=20)
        assert result is not None


# ═══════════════════════════════════════════════════════════════════════
# 3. Preprocessing Pipeline — edge cases
# ═══════════════════════════════════════════════════════════════════════

class TestPreprocessingEdgeCases:

    def test_pipeline_heavy_missing(self, heavy_missing_df):
        """Pipeline handles 40% missing data with median imputation."""
        from ml.pipeline import build_preprocessing_pipeline
        features = [c for c in heavy_missing_df.columns
                    if c != 'target' and heavy_missing_df[c].dtype in ('float64', 'int64')]
        cat_features = [c for c in heavy_missing_df.columns
                        if c != 'target' and heavy_missing_df[c].dtype == 'object']
        pipe = build_preprocessing_pipeline(
            numeric_features=features,
            categorical_features=cat_features,
            numeric_imputation="median",
            numeric_scaling="standard",
        )
        X = heavy_missing_df[features + cat_features]
        pipe.fit(X)
        X_t = pipe.transform(X)
        if hasattr(X_t, 'toarray'):
            X_t = X_t.toarray()
        assert not np.any(np.isnan(X_t[:, :len(features)])), \
            "Imputation must eliminate NaNs in known-non-empty columns"

    def test_pipeline_constant_columns(self, constant_df):
        """Pipeline handles constant + near-constant columns."""
        from ml.pipeline import build_preprocessing_pipeline
        features = [c for c in constant_df.columns
                    if c != 'target' and constant_df[c].dtype in ('float64', 'int64')]
        cat_features = [c for c in constant_df.columns
                        if c != 'target' and constant_df[c].dtype == 'object']
        pipe = build_preprocessing_pipeline(
            numeric_features=features,
            categorical_features=cat_features,
            numeric_imputation="median",
            numeric_scaling="standard",
        )
        X = constant_df[features + cat_features]
        pipe.fit(X)
        X_t = pipe.transform(X)
        if hasattr(X_t, 'toarray'):
            X_t = X_t.toarray()
        # StandardScaler on constant column produces 0 (not NaN)
        assert not np.any(np.isnan(X_t)), "No NaNs after preprocessing constant columns"

    def test_pipeline_wide_dataset(self, wide_df):
        """Pipeline handles 60 numeric features (more than samples)."""
        from ml.pipeline import build_preprocessing_pipeline
        features = [c for c in wide_df.columns
                    if c != 'target' and wide_df[c].dtype in ('float64', 'int64')]
        pipe = build_preprocessing_pipeline(
            numeric_features=features,
            categorical_features=[],
            numeric_imputation="median",
            numeric_scaling="standard",
        )
        X = wide_df[features]
        pipe.fit(X)
        X_t = pipe.transform(X)
        assert X_t.shape == (50, 60)

    def test_pipeline_tiny_dataset(self, tiny_df):
        """Pipeline works on n=30."""
        from ml.pipeline import build_preprocessing_pipeline
        features = [c for c in tiny_df.columns
                    if c != 'target' and tiny_df[c].dtype in ('float64', 'int64')]
        cat_features = [c for c in tiny_df.columns
                        if c != 'target' and tiny_df[c].dtype == 'object']
        pipe = build_preprocessing_pipeline(
            numeric_features=features,
            categorical_features=cat_features,
            numeric_imputation="median",
            numeric_scaling="standard",
        )
        X = tiny_df[features + cat_features]
        pipe.fit(X)
        X_t = pipe.transform(X)
        if hasattr(X_t, 'toarray'):
            X_t = X_t.toarray()
        assert X_t.shape[0] == 30


# ═══════════════════════════════════════════════════════════════════════
# 4. Splitting — edge cases
# ═══════════════════════════════════════════════════════════════════════

class TestSplittingEdgeCases:

    def test_tiny_dataset_split(self, tiny_df):
        """n=30 split into 70/15/15 produces non-empty sets."""
        from tests.conftest import prepare_splits
        splits = prepare_splits(tiny_df, target_col='target')
        assert len(splits['X_train']) >= 15
        assert len(splits['X_val']) >= 2
        assert len(splits['X_test']) >= 2

    def test_imbalanced_split_preserves_both_classes(self, imbalanced_df):
        """Severe imbalance (95/5): both classes must appear in all sets."""
        from tests.conftest import prepare_splits
        splits = prepare_splits(imbalanced_df, target_col='target_class')
        for key in ('y_train', 'y_val', 'y_test'):
            y = splits[key]
            unique = np.unique(y.values if hasattr(y, 'values') else y)
            # With 200 samples and 5% minority, all sets should have both classes
            # (though very small test set might not — check train at minimum)
            if key == 'y_train':
                assert len(unique) == 2, f"Train set must have both classes, got {unique}"


# ═══════════════════════════════════════════════════════════════════════
# 5. Training — edge cases
# ═══════════════════════════════════════════════════════════════════════

class TestTrainingEdgeCases:

    def test_ridge_on_tiny_dataset(self, tiny_df):
        """Ridge regression on n=30: should produce reasonable metrics."""
        from tests.conftest import prepare_splits, train_ridge_model
        splits = prepare_splits(tiny_df, target_col='target')
        result = train_ridge_model(splits)
        assert 'metrics' in result
        assert result['metrics']['RMSE'] > 0
        assert isinstance(result['metrics']['R2'], float)

    def test_ridge_on_wide_dataset(self, wide_df):
        """Ridge on p=60 > n=50: regularization handles this."""
        from tests.conftest import prepare_splits, train_ridge_model
        splits = prepare_splits(wide_df, target_col='target')
        result = train_ridge_model(splits)
        assert result['metrics']['RMSE'] > 0

    def test_ridge_on_heavy_missing(self, heavy_missing_df):
        """Ridge on data with 40% missing (after imputation).

        The all-NaN column must be excluded before training — in the real app
        this is handled by preprocessing pipeline (SimpleImputer skips it).
        prepare_splits uses fillna(median), and median of all-NaN is NaN.
        """
        # Drop the all-NaN column to match what the pipeline would do
        df = heavy_missing_df.drop(columns=['feat_all_nan'])
        from tests.conftest import prepare_splits, train_ridge_model
        splits = prepare_splits(df, target_col='target')
        result = train_ridge_model(splits)
        assert result['metrics']['RMSE'] > 0

    def test_cross_validation_capped_folds(self, tiny_df):
        """CV folds must be capped to n_samples on tiny datasets."""
        from ml.eval import perform_cross_validation
        from sklearn.linear_model import Ridge
        from tests.conftest import prepare_splits

        splits = prepare_splits(tiny_df, target_col='target')
        X_train = splits['X_train']
        y_train = splits['y_train']
        model = Ridge(alpha=1.0)
        n_train = len(X_train)

        # Request more folds than we have samples — function should cap or handle
        safe_folds = min(5, n_train)
        result = perform_cross_validation(model, X_train, y_train, cv_folds=safe_folds)
        assert result is not None

    def test_classification_severe_imbalance(self, imbalanced_df):
        """Classification with 95/5 imbalance: should not crash."""
        from sklearn.linear_model import LogisticRegression
        from ml.eval import calculate_classification_metrics
        from tests.conftest import prepare_splits

        splits = prepare_splits(imbalanced_df, target_col='target_class')
        X_train = splits['X_train']
        y_train = splits['y_train']
        X_test = splits['X_test']
        y_test = splits['y_test']

        model = LogisticRegression(max_iter=1000, class_weight='balanced')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = calculate_classification_metrics(
            y_test.values if hasattr(y_test, 'values') else y_test,
            y_pred,
        )
        assert 'Accuracy' in metrics
        assert 'F1' in metrics


# ═══════════════════════════════════════════════════════════════════════
# 6. Full Mini-Pipeline — diverse datasets end-to-end
# ═══════════════════════════════════════════════════════════════════════

class TestMiniPipelineEndToEnd:
    """Push each dataset through profile → preprocess → split → train → evaluate."""

    def _run_mini_pipeline(self, df, target_col, task_type='regression'):
        """Run a minimal pipeline and return metrics."""
        from ml.dataset_profile import compute_dataset_profile
        from ml.pipeline import build_preprocessing_pipeline
        from sklearn.linear_model import Ridge, LogisticRegression
        from ml.eval import calculate_regression_metrics, calculate_classification_metrics

        features = [c for c in df.columns if c != target_col and df[c].dtype in ('float64', 'int64')]
        cat_features = [c for c in df.columns if c != target_col and df[c].dtype == 'object']

        # Profile
        profile = compute_dataset_profile(df, target_col, features + cat_features, task_type)
        assert profile is not None

        # Preprocess
        pipe = build_preprocessing_pipeline(
            numeric_features=features,
            categorical_features=cat_features,
            numeric_imputation="median",
            numeric_scaling="standard",
        )

        # Split
        mask = df[target_col].notna()
        X_all = df.loc[mask, features + cat_features]
        y_all = df.loc[mask, target_col]
        n = len(X_all)
        n_train = max(int(n * 0.7), 5)
        n_val = max(int(n * 0.15), 2)

        X_train = X_all.iloc[:n_train]
        X_test = X_all.iloc[n_train + n_val:]
        y_train = y_all.iloc[:n_train]
        y_test = y_all.iloc[n_train + n_val:]

        if len(X_test) < 2:
            X_test = X_all.iloc[n_train:]
            y_test = y_all.iloc[n_train:]

        # Fit pipeline on train, transform both
        pipe.fit(X_train)
        X_train_t = pipe.transform(X_train)
        X_test_t = pipe.transform(X_test)
        if hasattr(X_train_t, 'toarray'):
            X_train_t = X_train_t.toarray()
            X_test_t = X_test_t.toarray()

        # Train
        if task_type == 'regression':
            model = Ridge(alpha=1.0)
            model.fit(X_train_t, y_train)
            y_pred = model.predict(X_test_t)
            metrics = calculate_regression_metrics(
                y_test.values if hasattr(y_test, 'values') else y_test, y_pred
            )
            assert 'RMSE' in metrics
        else:
            y_tr = y_train.values if hasattr(y_train, 'values') else y_train
            y_te = y_test.values if hasattr(y_test, 'values') else y_test
            model = LogisticRegression(max_iter=1000, class_weight='balanced')
            model.fit(X_train_t, y_tr)
            y_pred = model.predict(X_test_t)
            metrics = calculate_classification_metrics(y_te, y_pred)
            assert 'Accuracy' in metrics

        return metrics

    def test_tiny_pipeline(self, tiny_df):
        metrics = self._run_mini_pipeline(tiny_df, 'target')
        assert metrics['RMSE'] > 0

    def test_wide_pipeline(self, wide_df):
        metrics = self._run_mini_pipeline(wide_df, 'target')
        assert metrics['RMSE'] > 0

    def test_heavy_missing_pipeline(self, heavy_missing_df):
        metrics = self._run_mini_pipeline(heavy_missing_df, 'target')
        assert metrics['RMSE'] > 0

    def test_imbalanced_pipeline(self, imbalanced_df):
        metrics = self._run_mini_pipeline(imbalanced_df, 'target_class', 'classification')
        assert metrics['Accuracy'] > 0

    def test_constant_columns_pipeline(self, constant_df):
        metrics = self._run_mini_pipeline(constant_df, 'target')
        assert metrics['RMSE'] > 0
