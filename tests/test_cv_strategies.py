"""Cross-validation must respect the split's leakage semantics.

If the train/test split kept entities together (group split) or in time order
(time split), plain shuffled KFold would leak across folds and inflate the CV
score. perform_cross_validation adapts the fold scheme to cv_strategy.
"""
import os
import sys

import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from sklearn.linear_model import LinearRegression, LogisticRegression  # noqa: E402


def test_standard_strategy_matches_task():
    from ml.eval import perform_cross_validation
    X = np.random.RandomState(0).rand(60, 3)
    y = X[:, 0] * 2 + np.random.RandomState(1).rand(60) * 0.1
    res = perform_cross_validation(LinearRegression(), X, y, cv_folds=5,
                                   task_type='regression')
    assert res['strategy'] == 'standard'
    assert res['folds'] == 5
    assert np.isfinite(res['mean'])


def test_group_strategy_uses_group_folds_and_clamps():
    """Group CV must not leak entities across folds and must not request more
    folds than there are groups."""
    from ml.eval import perform_cross_validation
    rng = np.random.RandomState(0)
    # 4 entities, repeated measures — a random KFold would split an entity
    groups = np.repeat(np.arange(4), 10)
    X = rng.rand(40, 3)
    y = X[:, 0] + rng.rand(40) * 0.1
    res = perform_cross_validation(LinearRegression(), X, y, cv_folds=5,
                                   task_type='regression',
                                   cv_strategy='group', groups=groups)
    assert res['strategy'] == 'group'
    # 5 folds requested but only 4 groups → clamped
    assert res['folds'] == 4
    assert np.isfinite(res['mean'])


def test_group_strategy_falls_back_when_no_groups():
    from ml.eval import perform_cross_validation
    X = np.random.RandomState(0).rand(40, 3)
    y = X[:, 0] + np.random.RandomState(1).rand(40) * 0.1
    # cv_strategy='group' but groups omitted → must not crash; uses standard
    res = perform_cross_validation(LinearRegression(), X, y, cv_folds=4,
                                   task_type='regression',
                                   cv_strategy='group', groups=None)
    assert np.isfinite(res['mean'])


def test_time_strategy_uses_timeseries_split():
    from ml.eval import perform_cross_validation
    rng = np.random.RandomState(0)
    n = 60
    X = np.column_stack([np.arange(n), rng.rand(n)])  # first col is time-like
    y = np.arange(n) * 0.5 + rng.rand(n)
    res = perform_cross_validation(LinearRegression(), X, y, cv_folds=5,
                                   task_type='regression', cv_strategy='time')
    assert res['strategy'] == 'time'
    assert res['folds'] == 5
    assert np.isfinite(res['mean'])


def test_group_classification_keeps_groups_intact():
    from ml.eval import perform_cross_validation
    rng = np.random.RandomState(0)
    groups = np.repeat(np.arange(6), 10)
    X = rng.rand(60, 3)
    y = (rng.rand(60) > 0.5).astype(int)
    res = perform_cross_validation(LogisticRegression(max_iter=200), X, y,
                                   cv_folds=4, task_type='classification',
                                   cv_strategy='group', groups=groups)
    assert res['strategy'] == 'group'
    assert np.isfinite(res['mean'])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
