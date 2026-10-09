"""The elastic net's penalty is chosen the same way on every platform (RECIPES_AND_TUNING §4.2,
"Score" and "Choice"; §4.7, "Threads").

On macOS (Accelerate) and Linux (OpenBLAS) the WP7 prediction fixture's elastic net chose
neighboring penalties for the same rows, and its intercept moved by up to 6.4 (commit 9c349c4e).
scikit-learn's ``ElasticNetCV`` computes each point of the inner cross-validated curve by
coordinate descent to a tolerance of 10⁻⁴, and chooses the first minimum of the folds' unweighted
mean, unrounded. The family now solves its paths exactly (``exact_path``) and chooses on the pooled
inner loss (squared error summed over every inner validation row, over their number), rounded to
10⁻⁹ of the smallest, a tie going to the earlier mix and the larger penalty.

The yes/no and class outcomes had kept scikit-learn's ``LogisticRegressionCV`` (``saga`` at a
tolerance of 10⁻³, the unrounded mean of the folds' log losses): its inner curve was 2 × 10⁻³ off
the exact one, and a penalty that zeroed every coefficient stopped it with the intercept unfitted.
The same ruling applies to them now (:class:`PooledLogisticRegressionCV`), and to the selection
step's elastic net on a yes/no outcome, whose penalty was chosen by accuracy.

**Independent reference.** The elastic net's exact solution at each penalty, mix and inner fold:
the linear system its optimality (KKT) conditions give on the active set and signs, solved
directly with iterative refinement, then checked against every condition, the inactive columns'
included. For the logistic loss the system is not linear, so Newton's method solves it on the
active set and signs, from a proposal by scikit-learn's own ``saga``, until the step is below
rounding; then every condition is checked the same way. The solver only proposes the active set;
the numbers come from the solve, and a wrong active set fails the check.
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


def _exact_curve(X, y, folds, mixes, grid, tol: float = 1e-14) -> np.ndarray:
    """The pooled inner loss of the exact solution at every mix and penalty of ``grid``, each
    proposed by scikit-learn's coordinate descent at ``tol``."""
    from sklearn.linear_model import enet_path

    out = np.zeros(grid.shape)
    for m, mix in enumerate(mixes):
        sse = np.zeros(grid.shape[1])
        for train, test in folds:
            Xt, yt = X[train], y[train]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # a proposal need not converge: the solve checks
                _, proposed, _ = enet_path(Xt - Xt.mean(axis=0), yt - yt.mean(), l1_ratio=mix,
                                           alphas=grid[m], tol=tol, max_iter=100_000)
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


def test_a_table_with_nearly_as_many_columns_as_rows_is_solved_exactly_and_quickly():
    """The verifier's cost finding: converging coordinate descent to 10⁻¹² made the family four to
    twelve times slower than scikit-learn's default, and on near-square correlated tables it still
    stopped short (ConvergenceWarnings, reported to users as "optimizer stopped before
    converging"). On 120 rows × 100 correlated columns (inner training folds of 96 rows) it took
    18.6 s and its curve was 3 × 10⁻¹¹ off. The exact path takes about a second, raises no warning,
    and its curve is the exact solution's to 10⁻¹² at every penalty of the lasso end and of an even
    mix."""
    import time

    from sklearn.exceptions import ConvergenceWarning

    from turbotab.core.models.elastic_net import SOLVER_MAX_ITER, SOLVER_TOL, PooledElasticNetCV

    rng = np.random.default_rng(11)
    n, p = 120, 100
    F, L = rng.normal(size=(n, 5)), rng.normal(size=(5, p))
    X = 0.8 * F @ L / np.sqrt(5) + 0.6 * rng.normal(size=(n, p))
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    y = X[:, :8] @ rng.normal(0, 0.5, 8) + rng.normal(size=n)
    tests = np.array_split(rng.permutation(n), 5)
    folds = [(np.setdiff1d(np.arange(n), t), np.sort(t)) for t in tests]
    mixes = [0.5, 1.0]
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        started = time.perf_counter()
        model = PooledElasticNetCV(l1_ratio=mixes, cv=folds, max_iter=SOLVER_MAX_ITER,
                                   tol=SOLVER_TOL).fit(X, y)
        took = time.perf_counter() - started
    assert took < 60, f"{took:.1f} s"  # a loose guard (about 1 s idle): CI runs four at once
    grid = np.atleast_2d(model.alphas_)
    exact = _exact_curve(X, y, folds, mixes, grid, tol=1e-6)
    worst = float(np.max(np.abs(model.pooled_loss_ - exact) / exact))
    assert worst <= 1e-12, f"the inner curve is {worst:.2g} from the exact solution's"


# ── the logistic loss: yes/no and classes ────────────────────────────────────


def _logistic_table(classes: int, seed: int = 0, n: int = 300, p: int = 10):
    """Correlated predictors, three of them real, an outcome drawn from a logistic model (two
    classes, or ordered thirds of a latent score), and five inner folds of unequal size."""
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, p))
    X = Z + 0.6 * Z[:, [0]]
    score = X[:, :3] @ np.array([0.8, -0.6, 0.4])
    if classes == 2:
        y = (rng.random(n) < 1 / (1 + np.exp(-score))).astype(int)
    else:
        latent = score + rng.logistic(size=n)
        y = np.digitize(latent, np.quantile(latent, [1 / 3, 2 / 3]))
    X = (X - X.mean(axis=0)) / X.std(axis=0)  # as the family's pipeline scales its columns
    sizes = np.round(np.asarray(FOLD_SIZES) * n / sum(FOLD_SIZES)).astype(int)
    tests = np.split(rng.permutation(n), np.cumsum(sizes)[:-1])
    folds = [(np.setdiff1d(np.arange(n), t), np.sort(t)) for t in tests]
    return X, y, folds


def _logistic_eta(X, W, b):
    """The linear predictors, a column of zeros first for two classes (the first class's)."""
    eta = X @ W.T + b
    return np.column_stack([np.zeros(len(X)), eta]) if W.shape[0] == 1 else eta


def _logistic_loss_rows(X, y, W, b):
    from scipy.special import logsumexp

    eta = _logistic_eta(X, W, b)
    return logsumexp(eta, axis=1) - eta[np.arange(len(y)), y]


def _exact_logistic(X, y, C, mix, proposal):
    """scikit-learn's penalized logistic regression, ``C Σ ℓᵢ + mix‖W‖₁ + (1 − mix)/2 ‖W‖²``
    (one row of ``W`` for two classes, one per class otherwise; the intercepts free), solved by
    Newton's method on the active set and signs ``proposal`` suggests, a coefficient dropped where
    its sign flips and added where its condition fails, until every KKT condition holds:
    ``(W, b)``, a multinomial's intercepts centered."""
    from scipy.special import softmax

    n, p = X.shape
    K = int(y.max()) + 1
    rows = 1 if K == 2 else K
    lam = 1.0 / (C * n)
    Y = np.eye(K)[y]
    k_of = slice(1, None) if rows == 1 else slice(None)  # the rows' columns of the class matrix
    n_b = 1 if K == 2 else K - 1  # the free intercepts: the last class's is held at 0
    W = np.where(np.abs(np.asarray(proposal, dtype=float)) > 1e-8, proposal, 0.0).reshape(rows, p)
    b = np.zeros(n_b)
    for _ in range(60):
        active = list(zip(*np.nonzero(W)))
        signs = np.array([np.sign(W[a]) for a in active])
        na = len(active)

        def unpack(t, active=active):
            V = np.zeros((rows, p))
            for i, a in enumerate(active):
                V[a] = t[i]
            return V, np.concatenate([t[len(active):], [0.0]])[:rows]

        def parts(t, active=active, signs=signs, na=na, unpack=unpack):
            V, c = unpack(t)
            eta = _logistic_eta(X, V, c)
            P = softmax(eta, axis=1)
            R = (P - Y)[:, k_of] / n  # the loss's gradient in each row's linear predictor
            g = np.array([R[:, r] @ X[:, j] for r, j in active] + [R[:, r].sum() for r in range(n_b)])
            g[:na] += lam * (mix * signs + (1 - mix) * t[:na])
            cols = [(r, X[:, j]) for r, j in active] + [(r, np.ones(n)) for r in range(n_b)]
            Pk = P[:, k_of]
            H = np.empty((len(cols), len(cols)))
            for i, (ri, xi) in enumerate(cols):
                for m_, (rm, xm) in enumerate(cols[i:], start=i):
                    H[i, m_] = H[m_, i] = (Pk[:, ri] * ((ri == rm) - Pk[:, rm]) / n * xi) @ xm
            H[:na, :na] += lam * (1 - mix) * np.eye(na)
            top = eta.max(axis=1)
            loss = float((np.log(np.exp(eta - top[:, None]).sum(axis=1)) + top
                          - eta[np.arange(n), y]).sum() / n)
            pen = (mix * float(signs @ t[:na]) + (1 - mix) / 2 * float(t[:na] @ t[:na])) if na else 0.0
            return g, H, loss + lam * pen

        theta = np.concatenate([[W[a] for a in active], b]) if active else b.copy()
        for _ in range(100):
            g, H, value = parts(theta)
            step = np.linalg.solve(H, g)
            t = 1.0
            while parts(theta - t * step)[2] > value - 1e-4 * t * (g @ step) and t > 1e-12:
                t /= 2
            theta = theta - t * step
            if np.max(np.abs(t * step)) <= 1e-15 * max(1.0, np.max(np.abs(theta))):
                break
        for _ in range(2):  # refinement
            g, H, _ = parts(theta)
            theta = theta - np.linalg.solve(H, g)
        W, full = unpack(theta)
        b = full[:n_b]
        flipped = [a for a, s in zip(active, signs) if np.sign(W[a]) != s]
        if flipped:
            for a in flipped:
                W[a] = 0.0
            continue
        G = ((softmax(_logistic_eta(X, W, full), axis=1) - Y)[:, k_of].T @ X) / n  # rows × p
        excess = np.where(W == 0, np.abs(G) - lam * mix, -np.inf)
        worst = np.unravel_index(int(np.argmax(excess)), excess.shape)
        if excess[worst] > 1e-9 * lam * max(mix, 1e-300):
            W[worst] = -np.sign(G[worst]) * 1e-12  # joins with the sign that descends
            continue
        return W, (full - full.mean() if rows > 1 else full)
    raise AssertionError(f"no active set settled at C={C}, mix={mix}")


def _exact_logistic_curve(X, y, folds, mixes, Cs):
    """The pooled inner log loss of the exact solution at every mix and C, each proposed by
    scikit-learn's ``saga``."""
    from sklearn.linear_model import LogisticRegression

    out = np.zeros((len(mixes), len(Cs)))
    for m, mix in enumerate(mixes):
        for train, test in folds:
            for c, C in enumerate(Cs):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    proposed = LogisticRegression(C=C, l1_ratio=mix, solver="saga", tol=1e-8,
                                                  max_iter=100_000, random_state=0
                                                  ).fit(X[train], y[train]).coef_
                W, b = _exact_logistic(X[train], y[train], C, mix, proposed)
                out[m, c] += _logistic_loss_rows(X[test], y[test], W, b).sum()
    return out / sum(len(t) for _, t in folds)


def _pooled(model, folds) -> np.ndarray:
    """The inner log loss pooled over every validation row, from the folds' means (``scores_``,
    negated, one per fold, mix and C as scikit-learn lays them out): mixes × Cs."""
    sizes = np.array([len(t) for _, t in folds], dtype=float)
    return -(np.asarray(model.scores_) * sizes[:, None, None]).sum(axis=0) / sizes.sum()


@pytest.mark.parametrize("classes", [2, 3], ids=["yes-no", "classes"])
def test_the_logistic_inner_curve_is_the_exact_solutions_and_the_choice_its_rounded_minimum(classes):
    """The family's yes/no and class outcomes: its inner curve (the log loss pooled over the
    validation rows) equals the exact solution's to 10⁻¹¹, where ``saga`` at the family's 10⁻³ was
    10⁻³ off; at the strongest penalty, which zeros every coefficient, the intercepts are the
    training rows' log odds, where ``saga`` stopped with them unfitted. The chosen mix and C are
    the exact curve's lowest after rounding to 10⁻⁹ of its smallest, the first of any tie, and the
    refit's coefficients and probabilities are the exact solution's at them to 10⁻⁹."""
    from scipy.special import softmax

    from turbotab.core.models.elastic_net import ELASTIC_NET, LOGISTIC_CS, LOGISTIC_L1_RATIOS

    X, y, folds = _logistic_table(classes)
    task = "binary" if classes == 2 else "multiclass"
    model = ELASTIC_NET.build(task, "prediction", len(y), X.shape[1]).set_params(cv=folds)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)
    mixes, Cs = list(LOGISTIC_L1_RATIOS), list(LOGISTIC_CS)
    pooled = _pooled(model, folds)
    exact = _exact_logistic_curve(X, y, folds, mixes, Cs)
    worst = float(np.max(np.abs(pooled - exact) / exact))
    assert worst <= 1e-11, f"the inner curve is {worst:.2g} from the exact solution's"

    train = folds[0][0]
    shares = np.bincount(y[train], minlength=classes) / len(train)
    null = (np.array([np.log(shares[1] / shares[0])]) if classes == 2
            else np.log(shares) - np.log(shares).mean())
    W, b = _exact_logistic(X[train], y[train], Cs[0], mixes[0],
                           np.zeros((1 if classes == 2 else classes, X.shape[1])))
    assert not W.any() and np.allclose(b, null, rtol=0, atol=1e-12)  # the null model
    path = np.asarray(model.coefs_paths_)[0, 0, 0]  # fold 0, the first mix, the smallest C
    assert not path[:, :-1].any() and np.allclose(path[:, -1], null, rtol=0, atol=1e-12)

    rounded = np.round(exact / (1e-9 * exact.min()))
    m, c = np.unravel_index(int(np.argmin(rounded)), exact.shape)
    assert (model.l1_ratio_, model.C_) == (mixes[m], Cs[c])
    W, b = _exact_logistic(X, y, Cs[c], mixes[m], np.asarray(model.coef_))
    scale = float(np.max(np.abs(W)))
    assert np.max(np.abs(np.atleast_2d(model.coef_) - W)) <= 1e-9 * scale
    assert np.max(np.abs(np.atleast_1d(model.intercept_) - b)) <= 1e-9 * max(scale, 1.0)
    proba = softmax(_logistic_eta(X, W, b), axis=1)
    assert np.max(np.abs(model.predict_proba(X) - proba)) <= 1e-9


def test_a_last_bit_difference_in_the_matrix_moves_no_logistic_choice_or_curve():
    """What differs between macOS and Linux with the folds held: the matrix's last bit, and the
    order of a sum. Every value moved one unit in the last place either way, or the columns
    permuted, leaves the chosen C and mix as they were, the curve to 10⁻¹³ and the coefficients to
    10⁻¹¹ of the largest, for two classes and three. (An invariance guard: ``saga``, deterministic
    for its seed, also held it; what it did not hold is the exact curve, above.)"""
    from sklearn.base import clone

    from turbotab.core.models.elastic_net import ELASTIC_NET

    for classes in (2, 3):
        X, y, folds = _logistic_table(classes, seed=4)
        task = "binary" if classes == 2 else "multiclass"
        proto = ELASTIC_NET.build(task, "prediction", len(y), X.shape[1]).set_params(cv=folds)
        rng = np.random.default_rng(classes)
        nudged = np.where(rng.random(X.shape) < 0.5, np.nextafter(X, np.inf),
                          np.nextafter(X, -np.inf))
        cols = rng.permutation(X.shape[1])
        base, *others = [clone(proto).fit(matrix, y) for matrix in (X, nudged, X[:, cols])]
        for i, other in enumerate(others):
            assert (other.C_, other.l1_ratio_) == (base.C_, base.l1_ratio_), (classes, i)
            moved = np.max(np.abs(_pooled(other, folds) - _pooled(base, folds))
                           / _pooled(base, folds))
            assert moved <= 1e-13, (classes, i, moved)
            coef = np.atleast_2d(other.coef_)
            if i == 1:  # the permuted columns, put back in their order
                coef = coef[:, np.argsort(cols)]
            assert np.max(np.abs(coef - np.atleast_2d(base.coef_))) <= 1e-11 * np.max(np.abs(base.coef_))


# The same table on every platform: what macOS computed (2026-10-08: numpy 2.4.6, scikit-learn
# 1.9.0, Accelerate), which CI's Linux (OpenBLAS) must give again; losses to 12 significant digits.
LOGISTIC_FROZEN = {
    2: {"C": 0.19306977288832497, "mix": 1.0,
        "loss": [0.531962574952, 0.533852080874, 0.536289013137, 0.537409917079]},
    3: {"C": 0.19306977288832497, "mix": 1.0,
        "loss": [0.921962519313, 0.930128154047, 0.930542685067, 0.940733429653]},
}


def _nhanes_logistic(classes: int):
    """The WP7-like NHANES table through the family's own pipeline step and inner folds."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from turbotab.core.models.elastic_net import ELASTIC_NET
    from turbotab.core.models.inner_cv import fit_pipeline
    from turbotab.core.tests import modeling_fixtures as mf

    frame = mf.nhanes_like(400, seed=707)
    X = frame.drop(columns=["SEQN", "cycle_begin_year", "glucose", "gender"]).astype(float)
    X["male"] = (frame["gender"] == "male").astype(float)
    g = frame["glucose"].to_numpy()
    y = ((g > np.median(g)).astype(int) if classes == 2
         else np.digitize(g, np.quantile(g, [1 / 3, 2 / 3])))
    task = "binary" if classes == 2 else "multiclass"
    pipe = Pipeline([("scale", StandardScaler()),
                     ("model", ELASTIC_NET.build(task, "prediction", len(y), X.shape[1]))])
    return fit_pipeline(pipe, X, y, seed=0)[-1]


@pytest.mark.parametrize("classes", [2, 3], ids=["yes-no", "classes"])
def test_the_logistic_choice_and_curve_are_the_ones_every_platform_computes(classes):
    """The WP7-like NHANES table (``nhanes_like``, 400 rows, seed 707): glucose above its median,
    or its thirds, from every other column, through the family's pipeline step and its inner folds
    (``inner_cv.fit_pipeline``: row keys hashed in single precision). The chosen C and mix, and the
    pooled curve's four lowest values, are those frozen above, on every platform."""
    frozen = LOGISTIC_FROZEN[classes]
    model = _nhanes_logistic(classes)
    lowest = [float(f"{v:.12g}") for v in np.sort(_pooled(model, model.cv).ravel())[:4]]
    assert (model.C_, model.l1_ratio_) == (frozen["C"], frozen["mix"]), (model.C_, model.l1_ratio_)
    assert np.allclose(lowest, frozen["loss"], rtol=1e-11, atol=0), lowest


def test_the_selection_steps_yes_no_penalty_is_the_familys_choice_on_the_log_loss():
    """The selection step's elastic net on a yes/no outcome (mixing 0.5, ``C`` over scikit-learn's
    ten from 10⁻⁴ to 10⁴) chose its penalty by accuracy, scikit-learn's default score, which is
    not a proper score: where the predictions are all one class the accuracies tie, and the first,
    the strongest penalty, wins. It keeps now what the exact solution keeps at the C the pooled log
    loss chooses, rounded."""
    from turbotab.core.models.variable_selection import elastic_net_support

    X, y, folds = _logistic_table(2, seed=0, n=150, p=12)  # by accuracy, all 12 were kept
    Cs = list(np.logspace(-4, 4, 10))
    exact = _exact_logistic_curve(X, y, folds, [0.5], Cs)
    C = Cs[int(np.argmin(np.round(exact / (1e-9 * exact.min()))))]
    W, _ = _exact_logistic(X, y, C, 0.5, np.ones((1, X.shape[1])))
    kept = elastic_net_support(X, y.astype(float), "binary", folds, seed=0)
    assert kept.tolist() == (np.abs(W[0]) > 1e-10).tolist()
    assert 0 < int(kept.sum()) < X.shape[1]
