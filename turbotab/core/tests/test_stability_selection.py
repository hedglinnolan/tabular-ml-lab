"""Stability selection's base procedure on a full-coded category (HANDOFF of 2026-10-11, "Known
red"; ``variable_selection.lasso_first``).

The elastic net and ridge read a category in full coding: a two-level ``sex`` becomes ``sex_F``
and ``sex_M``, which sum to one on every row, so on the fold's standardized matrix
``z_F = −z_M`` exactly. The lasso on the two columns is the lasso on one of them: the fit reads
only ``b_M − b_F``, and the smallest penalty ``|b_F| + |b_M|`` for a given ``b_M − b_F = d`` is
``|d|`` (the triangle inequality), so the solutions, and the variables they make active, are the
reference-coded design's at every penalty. LARS does not see it: once one twin is active the other
ties with the active set's correlation at every step, the path takes steps of length zero, and the
"first q to enter" are whatever its degenerate handling admits, which rounding decides (the fast
tier's three held-out cases with ``select_stability`` failed intermittently on CI's runners).

**Independent reference.** scikit-learn's own ``lars_path`` on the reference-coded matrix
(``sex_M`` alone), which has full column rank, standardized here by hand (ddof = 1) as the module
documents, over the complementary halves rebuilt here from the documented rule (half b is
``rng.permutation(n)[:⌊n/2⌋]``, its complement the next ⌊n/2⌋ positions).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

N, PAIRS, Q = 120, 10, 3
OTHERS = [f"x{j}" for j in range(6)]


def _full_coded(seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    """A two-level sex that predicts strongly, in full coding first (as the encoder emits it), two
    numbers that predict less, and four that do not."""
    rng = np.random.default_rng(seed)
    male = rng.integers(0, 2, N).astype(float)
    X = pd.DataFrame({"sex_F": 1.0 - male, "sex_M": male})
    for c in OTHERS:
        X[c] = rng.standard_normal(N)
    y = 1.2 * male + 0.6 * X["x0"] + 0.4 * X["x1"] + rng.standard_normal(N)
    return X, y.to_numpy()


def _reference_frequencies(X: pd.DataFrame, y: np.ndarray, seed: int) -> dict[str, float]:
    """Each variable's proportion of the 2B halves in which it is among the first ``Q`` to enter
    the lasso path of the reference-coded matrix (module docstring)."""
    from sklearn.linear_model import lars_path

    terms = ["sex", *OTHERS]
    R = X[["sex_M", *OTHERS]].to_numpy(dtype=float)
    Z = (R - R.mean(axis=0)) / R.std(axis=0, ddof=1)
    rng = np.random.default_rng(seed)
    half = N // 2
    counts = dict.fromkeys(terms, 0)
    for _ in range(PAIRS):
        order = rng.permutation(N)
        for rows in (order[:half], order[half: 2 * half]):
            yc = y[rows] - y[rows].mean()
            with warnings.catch_warnings():
                warnings.simplefilter("error")  # a full-rank path needs no degenerate handling
                _, _, coefs = lars_path(Z[rows], yc, method="lasso")
            first = next(j for j in range(coefs.shape[1]) if (coefs[:, j] != 0).sum() >= Q)
            for t, on in zip(terms, coefs[:, first] != 0):
                counts[t] += bool(on)
    return {t: counts[t] / (2.0 * PAIRS) for t in terms}


@pytest.mark.parametrize("seed", range(6))
def test_a_full_coded_category_enters_the_lasso_path_as_one_variable(seed):
    """The selection proportions on the full-coded matrix are the reference-coded matrix's, for
    every variable (module docstring), and the kept panel maps back to whole variables: a kept
    ``sex`` keeps both of its columns."""
    from turbotab.core.models.variable_selection import Selector

    X, y = _full_coded(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        step = Selector("stability", "regression", q=Q, pairs=PAIRS, seed=seed,
                        inputs=["sex", *OTHERS]).fit(X, y)
    expected = _reference_frequencies(X, y, seed)
    columns = list(X.columns)
    freq = np.nan_to_num(np.asarray(step.frequencies_, dtype=float), nan=0.0)
    found = {t: float(max(freq[columns.index(c)] for c in cols))
             for t, cols in step.terms_.items()}
    assert found == pytest.approx(expected, abs=1e-12)
    kept = {t for t, f in expected.items() if f >= 0.6}
    assert set(step.selected_) == kept
    assert ({"sex_F", "sex_M"} <= set(step.kept_)) == ("sex" in kept)


ORDER_N, ORDER_PAIRS, ORDER_SEEDS = 150, 20, 10


def _category(seed: int, levels: int) -> tuple[pd.DataFrame, np.ndarray, str]:
    """A category of ``levels`` levels (shares drawn from a Dirichlet(4)) in full coding, whose
    signal sits in one level drawn at random, beside two numbers that predict less and four that
    do not (the stage-1 reviewer's design)."""
    rng = np.random.default_rng(seed)
    level = rng.choice(levels, ORDER_N, p=rng.dirichlet(np.full(levels, 4.0)))
    signal = int(rng.integers(levels))
    X = pd.DataFrame({f"g_{k}": (level == k).astype(float) for k in range(levels)})
    for c in OTHERS:
        X[c] = rng.standard_normal(ORDER_N)
    y = 1.0 * (level == signal) + 0.6 * X["x0"] + 0.4 * X["x1"] + rng.standard_normal(ORDER_N)
    return X, y.to_numpy(), f"g_{signal}"


@pytest.mark.parametrize("levels", [3, 4])
def test_a_category_of_three_or_more_levels_is_selected_whatever_its_level_order(levels):
    """The lasso's penalty and fit are blind to the order of the columns, so where its solution is
    unique (a full coding of three or more levels whose shares do not split into equal halves), the
    path and the variables it makes active are the same whichever level the encoder lists last.
    Each level's selection proportion is the same under either order, and so is whether the
    category is kept."""
    from turbotab.core.models.variable_selection import Selector

    kept = {"first": 0, "last": 0}
    for seed in range(ORDER_SEEDS):
        X, y, signal = _category(seed, levels)
        rest = [c for c in X.columns if c.startswith("g_") and c != signal]
        found = {}
        for where, order in (("first", [signal, *rest]), ("last", [*rest, signal])):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                step = Selector("stability", "regression", q=Q, pairs=ORDER_PAIRS, seed=seed,
                                inputs=["g", *OTHERS]).fit(X[[*order, *OTHERS]], y)
            found[where] = dict(zip([*order, *OTHERS], np.asarray(step.frequencies_, float)))
            kept[where] += "g" in step.selected_
        assert found["last"] == pytest.approx(found["first"], abs=1e-12), seed
    assert kept["last"] == kept["first"]


def _stalled_coding(case: str, seed: int) -> tuple[np.ndarray, np.ndarray, int, int]:
    """(Z, y, q, spare) on which scikit-learn's path sometimes ties: a two-level category's twins
    (``_full_coded``), or a four-level category in exactly equal counts whose levels split two
    up and two down, where the lasso's split of the category among its levels is not unique.
    ``spare`` is the column a fallback leaves off: the twin, or the last level."""
    if case == "twins":
        X, y = _full_coded(seed)
        R, q, spare = X.to_numpy(dtype=float), Q, 1
    else:
        rng = np.random.default_rng(seed)
        level = rng.permutation(np.repeat(np.arange(4), 15))
        D = np.column_stack([(level == k).astype(float) for k in range(4)])
        R = np.column_stack([D, rng.standard_normal((60, 4))])
        y = D @ np.array([1.0, 1.0, -1.0, -1.0]) + 0.5 * R[:, 4] + 0.5 * rng.standard_normal(60)
        q, spare = 4, 3
    return (R - R.mean(axis=0)) / R.std(axis=0, ddof=1), np.asarray(y, dtype=float), q, spare


def _first_q(Z: np.ndarray, yc: np.ndarray, q: int) -> tuple[np.ndarray, bool]:
    """scikit-learn's path to its first knot with ``q`` nonzero, and whether it tied on the way:
    a ConvergenceWarning, or a step of the penalty under 10⁻⁸ of its start."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import lars_path

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        alphas, _, coefs = lars_path(Z, yc, method="lasso", max_iter=max(4 * q, 50))
    knot = next((j for j in range(coefs.shape[1]) if (coefs[:, j] != 0).sum() >= q),
                coefs.shape[1] - 1)
    steps = -np.diff(alphas[: knot + 1])
    tied = (any(issubclass(w.category, ConvergenceWarning) for w in caught)
            or bool(steps.size and steps.min() <= 1e-8 * alphas[0]))
    return coefs[:, knot] != 0, tied


@pytest.mark.parametrize("case", ["twins", "balanced four"])
def test_the_lasso_path_leaves_off_a_spare_level_only_where_it_ties(case):
    """Where scikit-learn's path on the full coding ties before its first q (it warns, or the
    penalty takes a step of length zero), ``lasso_first`` reads the coding less its spare column,
    and its first q are that path's; for twins it is the reference-coded lasso, the same problem
    (module docstring). Where the path does not tie, its first q are the full coding's own."""
    from turbotab.core.models.variable_selection import lasso_first

    seen = {True: 0, False: 0}
    for seed in range(40):
        Z, y, q, spare = _stalled_coding(case, seed)
        yc = y - y.mean()
        full, tied = _first_q(Z, yc, q)
        expected = full
        if tied:
            rest = [j for j in range(Z.shape[1]) if j != spare]
            expected = np.zeros(Z.shape[1], dtype=bool)
            expected[rest] = _first_q(Z[:, rest], yc, q)[0]
        seen[tied] += 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            got = lasso_first(Z, y, q, spare=[spare])
        assert got.tolist() == expected.tolist(), seed
    assert seen[True] and seen[False]
