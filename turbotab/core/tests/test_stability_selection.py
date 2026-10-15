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
