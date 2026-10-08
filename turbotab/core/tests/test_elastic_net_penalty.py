"""The elastic net's penalty is chosen the same way on every platform (RECIPES_AND_TUNING §4.2,
"Score" and "Choice"; §4.7, "Threads").

On macOS (Accelerate) and Linux (OpenBLAS) the WP7 prediction fixture's elastic net chose
neighboring penalties for the same rows, and its intercept moved by up to 6.4 (commit 9c349c4e).
scikit-learn's ``ElasticNetCV`` computes each point of the inner cross-validated curve by
coordinate descent to a tolerance of 10⁻⁴, and chooses the first minimum of the folds' unweighted
mean, unrounded. The family now converges its paths to 10⁻¹² and chooses on the pooled inner loss
(squared error summed over every inner validation row, over their number), rounded to 10⁻⁹ of the
smallest, a tie going to the earlier mix and the larger penalty.

**Independent reference.** The elastic net's exact solution at each penalty, mix and inner fold:
the linear system its optimality (KKT) conditions give on the active set and signs, solved
directly with iterative refinement, then checked against every condition, the inactive columns'
included. Coordinate descent only proposes the active set; the numbers come from the linear solve,
and a wrong active set fails the check.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

FOLD_SIZES = (30, 45, 60, 75, 90)  # unequal, so pooling the rows differs from averaging the folds


def _table(seed: int = 0, n: int = 300, p: int = 12):
    """Correlated predictors, four of them real, and five inner folds of unequal size."""
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, p))
    X = Z + 0.6 * Z[:, [0]]
    y = X[:, :4] @ np.array([0.5, -0.3, 0.2, 0.1]) + rng.normal(0, 1.5, n)
    tests = np.split(rng.permutation(n), np.cumsum(FOLD_SIZES)[:-1])
    folds = [(np.setdiff1d(np.arange(n), t), np.sort(t)) for t in tests]
    return X, y, folds


def _exact(X: np.ndarray, y: np.ndarray, alpha: float, mix: float, proposed: np.ndarray):
    """scikit-learn's elastic net, (1/2n)‖y − Xw − b‖² + α·mix·‖w‖₁ + α(1 − mix)/2·‖w‖², solved
    exactly on the active set and signs ``proposed`` suggests; the KKT conditions are asserted."""
    n = len(y)
    x_mean, y_mean = X.mean(axis=0), y.mean()
    Xc, yc = X - x_mean, y - y_mean
    active = np.flatnonzero(proposed)
    w = np.zeros(X.shape[1])
    if len(active):
        signs = np.sign(proposed[active])
        A = Xc[:, active].T @ Xc[:, active] / n + alpha * (1 - mix) * np.eye(len(active))
        b = Xc[:, active].T @ yc / n - alpha * mix * signs
        w_a = np.linalg.solve(A, b)
        for _ in range(3):  # iterative refinement
            w_a = w_a + np.linalg.solve(A, b - A @ w_a)
        assert np.all(np.sign(w_a) == signs), (alpha, mix)
        w[active] = w_a
    gradient = Xc.T @ (yc - Xc @ w) / n
    inactive = np.setdiff1d(np.arange(X.shape[1]), active)
    assert np.all(np.abs(gradient[inactive]) <= alpha * mix * (1 + 1e-9)), (alpha, mix)
    return w, y_mean - x_mean @ w


def _exact_curve(X, y, folds, mixes, grid) -> np.ndarray:
    """The pooled inner loss of the exact solution at every mix and penalty of ``grid``."""
    from sklearn.linear_model import enet_path

    out = np.zeros(grid.shape)
    for m, mix in enumerate(mixes):
        sse = np.zeros(grid.shape[1])
        for train, test in folds:
            Xt, yt = X[train], y[train]
            _, proposed, _ = enet_path(Xt - Xt.mean(axis=0), yt - yt.mean(), l1_ratio=mix,
                                       alphas=grid[m], tol=1e-14, max_iter=100_000)
            for k, alpha in enumerate(grid[m]):
                w, b = _exact(Xt, yt, alpha, mix, proposed[:, k])
                sse[k] += float(((y[test] - X[test] @ w - b) ** 2).sum())
        out[m] = sse / sum(len(t) for _, t in folds)
    return out


def test_the_inner_curve_is_the_exact_elastic_nets_and_the_choice_its_rounded_minimum():
    """The family's inner curve (pooled over the validation rows) equals the exact solution's to
    10⁻¹¹, two orders below the 10⁻⁹ the choice rounds to (scikit-learn's 10⁻⁴ tolerance was off
    by 10⁻⁵ to 10⁻³ on the WP7 fixture). The chosen mix and penalty are the exact curve's lowest
    after rounding to 10⁻⁹ of its smallest, the first of any tie; where that differs from averaging
    the folds (as on these unequal folds), the coefficients are refit at it: they equal the exact
    solution on every row to 10⁻⁹ of the largest."""
    from turbotab.core.models.elastic_net import ELASTIC_NET, L1_RATIOS

    X, y, folds = _table()
    model = ELASTIC_NET.build("regression", "prediction", len(y), X.shape[1]).set_params(cv=folds)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)
    mixes = list(L1_RATIOS)
    grid = np.atleast_2d(model.alphas_)
    sizes = np.array([len(t) for _, t in folds], dtype=float)
    errors = np.asarray(model.mse_path_).reshape(len(mixes), -1, len(folds))
    pooled = (errors * sizes).sum(axis=2) / sizes.sum()
    exact = _exact_curve(X, y, folds, mixes, grid)
    worst = float(np.max(np.abs(pooled - exact) / exact))
    assert worst <= 1e-11, f"the inner curve is {worst:.2g} from the exact solution's"

    rounded = np.round(exact / (1e-9 * exact.min()))
    m, k = np.unravel_index(int(np.argmin(rounded)), exact.shape)
    assert (model.l1_ratio_, model.alpha_) == (mixes[m], grid[m, k])
    averaged = np.unravel_index(int(np.argmin(errors.mean(axis=2))), exact.shape)
    assert averaged != (m, k)  # the fixture exercises the refit
    w, b = _exact(X, y, grid[m, k], mixes[m], model.coef_)
    scale = float(np.max(np.abs(w)))
    assert np.max(np.abs(model.coef_ - w)) <= 1e-9 * scale
    assert abs(model.intercept_ - b) <= 1e-9 * max(scale, abs(b))


def test_a_near_tie_goes_to_the_larger_penalty_and_noise_cannot_move_the_choice():
    """Losses closer than 10⁻⁹ of the smallest tie, and a tie goes to the earlier candidate (on a
    path, the larger penalty), whichever of them floating point makes smaller. Relative noise of
    10⁻¹² on every point of a fitted curve (200 draws) never moves its choice."""
    from turbotab.core.models.elastic_net import (L1_RATIOS, SOLVER_MAX_ITER, SOLVER_TOL,
                                                  PooledElasticNetCV, lowest_rounded)

    near = [2.0, 2.0 * (1 - 1e-12), 3.0]
    assert int(np.argmin(near)) == 1  # unrounded, the argmin follows the noise
    assert lowest_rounded(near) == 0
    assert lowest_rounded([2.0 * (1 + 1e-12), 2.0, 3.0]) == 0
    assert lowest_rounded([2.0 * (1 + 3e-9), 2.0, 3.0]) == 1  # a real difference stands
    assert lowest_rounded([np.nan, 3.0, 2.0]) == 2 and lowest_rounded([np.inf, 2.0]) == 1

    X, y, folds = _table()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = PooledElasticNetCV(l1_ratio=list(L1_RATIOS), cv=folds,
                                   max_iter=SOLVER_MAX_ITER, tol=SOLVER_TOL).fit(X, y)
    curve = model.pooled_loss_
    chosen = lowest_rounded(curve)
    m, k = np.unravel_index(chosen, curve.shape)
    assert (model.l1_ratio_, model.alpha_) == (L1_RATIOS[m], np.atleast_2d(model.alphas_)[m, k])
    rng = np.random.default_rng(0)
    for _ in range(200):
        assert lowest_rounded(curve * (1 + 1e-12 * rng.standard_normal(curve.shape))) == chosen
    tied = curve.ravel().copy()
    tied[chosen + 1] = tied[chosen] * (1 - 1e-12)  # the next candidate, a hair lower
    assert lowest_rounded(tied) == chosen


def test_a_difference_in_the_last_bit_draws_the_same_inner_folds():
    """The inner folds are drawn from a key per row computed from its values
    (``inner_cv.row_keys``). The WP7 fixture's simulated values differ in their last bit between
    macOS and Linux (``bp_di`` in the first row: 58.24442394575788 against …787); hashed exactly,
    every key and so every inner fold differed, and the elastic net tuned its penalty on other rows
    (the Linux runner's diagnostic on ci/elastic-net-determinism-diag). Moving every value and the
    outcome one unit in the last place, either way, now leaves every key and fold as it was, and a
    real difference still moves its row's key and no other."""
    import pandas as pd

    from turbotab.core.models.inner_cv import inner_splits, row_keys
    from turbotab.core.tests import modeling_fixtures as mf

    frame = mf.nhanes_like(400, seed=707)
    X = frame.drop(columns=["SEQN", "cycle_begin_year", "glucose"])
    y = frame["glucose"].to_numpy()
    keys = row_keys(X, y)
    folds = inner_splits(None, 5, 0, keys=keys)
    for direction in (np.inf, -np.inf):
        moved = X.copy()
        for c in moved.columns:
            if pd.api.types.is_float_dtype(moved[c]):
                moved[c] = np.nextafter(moved[c].to_numpy(), direction)
        again = row_keys(moved, np.nextafter(y, direction))
        assert np.array_equal(again, keys), f"{int((again != keys).sum())} of {len(keys)} keys moved"
        for (train, test), (train2, test2) in zip(folds, inner_splits(None, 5, 0, keys=again)):
            assert np.array_equal(train, train2) and np.array_equal(test, test2)
    changed = X.copy()
    changed.loc[5, "bmi"] += 0.01
    assert np.flatnonzero(row_keys(changed, y) != keys).tolist() == [5]


@pytest.mark.parametrize("threads", [2, 4])
def test_the_choice_is_the_same_at_any_thread_count(threads):
    """One BLAS thread and one path at a time, or ``threads`` of each: the same mix and penalty,
    the inner curve to 10⁻¹², and the coefficients to 10⁻¹⁰ of the largest."""
    from threadpoolctl import threadpool_limits

    from turbotab.core.models.elastic_net import (L1_RATIOS, SOLVER_MAX_ITER, SOLVER_TOL,
                                                  PooledElasticNetCV)

    rng = np.random.default_rng(3)
    n, p = 3000, 40
    Z = rng.normal(size=(n, p))
    X = Z + 0.5 * Z[:, [0]] + 0.3 * Z[:, [1]]
    y = X[:, :6] @ rng.normal(0, 0.3, 6) + rng.normal(0, 2.0, n)
    fits = []
    for blas, jobs in ((1, 1), (threads, threads)):
        with threadpool_limits(limits=blas), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fits.append(PooledElasticNetCV(l1_ratio=list(L1_RATIOS), cv=5,
                                           max_iter=SOLVER_MAX_ITER, tol=SOLVER_TOL,
                                           n_jobs=jobs).fit(X, y))
    one, many = fits
    assert (one.l1_ratio_, one.alpha_) == (many.l1_ratio_, many.alpha_)
    assert np.max(np.abs(one.pooled_loss_ - many.pooled_loss_) / one.pooled_loss_) <= 1e-12
    assert np.max(np.abs(one.coef_ - many.coef_)) <= 1e-10 * np.max(np.abs(one.coef_))
