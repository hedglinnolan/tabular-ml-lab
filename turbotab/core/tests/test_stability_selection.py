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

import time
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


RARE_N, RARE_PAIRS, RARE_SEEDS = 120, 20, 20
FLAGS = [f"f{j}" for j in range(12)]


def _beside_rare_flags(seed: int) -> tuple[pd.DataFrame, np.ndarray, str]:
    """A yes/no outcome whose risk rises in one level (drawn at random) of a four-level category
    (shares drawn from a Dirichlet(4)), in full coding beside twelve flags of 3% prevalence that do
    not predict (the stage-1 re-review's design)."""
    rng = np.random.default_rng(seed)
    level = rng.choice(4, RARE_N, p=rng.dirichlet(np.full(4, 4.0)))
    signal = int(rng.integers(4))
    X = pd.DataFrame({f"g_{k}": (level == k).astype(float) for k in range(4)})
    for c in FLAGS:
        X[c] = (rng.random(RARE_N) < 0.03).astype(float)
    risk = 1.0 / (1.0 + np.exp(0.5 - 1.0 * (level == signal)))
    return X, (rng.random(RARE_N) < risk).astype(int), f"g_{signal}"


def test_a_category_beside_rare_flags_is_selected_whatever_its_level_order():
    """Rare flags make a half's path tie for reasons that are not the category's: a flag with no
    ones on the half, or two flags with the same rows on it. The category's own coding does not tie
    (its shares are unbalanced), and the lasso is blind to the order of the columns, so in every
    dataset each column's selection proportion, and whether the category is kept, are the same
    whichever level the encoder lists last. A flag with no ones on the fold has no spread and no
    proportion."""
    from turbotab.core.models.variable_selection import Selector

    for seed in range(RARE_SEEDS):
        X, y, signal = _beside_rare_flags(seed)
        rest = [c for c in X.columns if c.startswith("g_") and c != signal]
        found, kept = {}, {}
        for where, order in (("first", [signal, *rest]), ("last", [*rest, signal])):
            columns = [*order, *FLAGS]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                step = Selector("stability", "binary", q=4, pairs=RARE_PAIRS, seed=seed,
                                inputs=["g", *FLAGS]).fit(X[columns], y)
            usable = [c for c in columns if X[c].std() > 0]
            found[where] = dict(zip(usable, np.asarray(step.frequencies_, float)))
            kept[where] = "g" in step.selected_
        assert found["last"] == pytest.approx(found["first"], abs=1e-12, nan_ok=True), seed
        assert kept["last"] == kept["first"], seed


def _stalled_coding(case: str, seed: int) -> tuple[np.ndarray, np.ndarray, int, int]:
    """(Z, y, q, spare): a four-level coding (the first four columns) or a two-level one (the first
    two) whose spare column is ``spare``, standardized (ddof = 1).

    * twins: a two-level category (``_full_coded``), whose twins tie as soon as it enters;
    * balanced four: a four-level category in exactly equal counts whose levels split two up and
      two down, where the lasso's split of the category among its levels is not unique;
    * rare copy: a four-level category in unequal counts (a Dirichlet(4)) whose signal sits in its
      last level, beside a flag that predicts and a copy of it on these rows: the copies tie, and
      the coding does not."""
    rng = np.random.default_rng(seed)
    if case == "twins":
        X, y = _full_coded(seed)
        R, q, spare = X.to_numpy(dtype=float), Q, 1
    elif case == "balanced four":
        level = rng.permutation(np.repeat(np.arange(4), 15))
        D = np.column_stack([(level == k).astype(float) for k in range(4)])
        R = np.column_stack([D, rng.standard_normal((60, 4))])
        y = D @ np.array([1.0, 1.0, -1.0, -1.0]) + 0.5 * R[:, 4] + 0.5 * rng.standard_normal(60)
        q, spare = 4, 3
    else:
        level = rng.choice(4, 60, p=rng.dirichlet(np.full(4, 4.0)))
        D = np.column_stack([(level == k).astype(float) for k in range(4)])
        flag = np.zeros(60)
        flag[rng.choice(60, 6, replace=False)] = 1.0
        R = np.column_stack([D, flag, flag, rng.standard_normal((60, 3))])
        y = 1.0 * D[:, 3] + 1.5 * flag + 0.5 * R[:, 6] + rng.standard_normal(60)
        q, spare = 4, 3
    return (R - R.mean(axis=0)) / R.std(axis=0, ddof=1), np.asarray(y, dtype=float), q, spare


def _lars(Z: np.ndarray, yc: np.ndarray, q: int) -> tuple[np.ndarray, np.ndarray, list]:
    """scikit-learn's path to its first knot with ``q`` nonzero: (that knot's nonzero mask, the
    coefficients at the knots before it, or through the last knot where it stops short of q, and
    its warnings)."""
    from sklearn.linear_model import lars_path

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _, _, coefs = lars_path(Z, yc, method="lasso", max_iter=max(4 * q, 50))
    count = (coefs != 0).sum(axis=0)
    reached = np.flatnonzero(count >= q)
    knot = int(reached[0]) if reached.size else coefs.shape[1] - 1
    before = coefs[:, :knot] if reached.size else coefs
    return coefs[:, knot] != 0, before, [(w.category, str(w.message)) for w in caught]


def _coding_ties(Z: np.ndarray, yc: np.ndarray, before: np.ndarray, coding: list[int]) -> bool:
    """Whether, at some knot of ``before``, every column of the coding is as correlated with the
    residual as the most correlated column (to 10⁻⁸ of it): its levels enter together."""
    corr = np.abs(Z.T @ (yc[:, None] - Z @ before))
    top = corr.max(axis=0)
    return bool(np.any((corr[coding] >= (1 - 1e-8) * top).all(axis=0) & (top > 1e-8 * top[0])))


@pytest.mark.parametrize("case", ["twins", "balanced four", "rare copy"])
def test_the_lasso_path_leaves_off_a_spare_level_only_where_its_coding_ties(case):
    """``lasso_first`` reads a full coding less its spare column only where the coding's own levels
    tie. For twins that is whenever the category enters, and the answer is the reference-coded
    lasso, the same problem (module docstring), on every seed. For a balanced coding it is where, at
    a knot before the q-th variable enters, scikit-learn's path on the full coding has every level
    at the residual's largest correlation, and the answer is that path less the spare level. Two
    copies of a flag tie for a reason that is not the coding's (scikit-learn warns on some seeds,
    and on others admits both): the answer is the full path's own first q, with its warnings passed
    on unchanged, though the signal sits in the spare level."""
    from turbotab.core.models.variable_selection import lasso_first

    reran = warned = 0
    for seed in range(40):
        Z, y, q, spare = _stalled_coding(case, seed)
        coding = [0, 1] if case == "twins" else [0, 1, 2, 3]
        yc = y - y.mean()
        rest = [j for j in range(Z.shape[1]) if j != spare]
        less = np.zeros(Z.shape[1], dtype=bool)
        less[rest], _, quiet = _lars(Z[:, rest], yc, q)
        full, before, said = _lars(Z, yc, q)
        if case == "twins":
            assert not quiet  # the reference coding has full rank: no degenerate step
            expected, rerun = less, bool(less[0])
        elif case == "balanced four":
            rerun = _coding_ties(Z, yc, before, coding)
            expected = less if rerun else full
        else:
            assert not _coding_ties(Z, yc, before, coding)
            expected, rerun = full, False
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            got = lasso_first(Z, y, q, [(coding, [spare])])
        assert got.tolist() == expected.tolist(), seed
        if case == "rare copy":
            assert [(w.category, str(w.message)) for w in caught] == said, seed
        reran += rerun
        warned += bool(said)
    if case == "rare copy":
        assert warned and not reran  # the copies' tie warns on some seeds
    else:
        assert reran


def test_a_term_wider_than_the_fold_is_read_whole():
    """A term of 80 columns on a fold of 50 rows, beside a predictor, fits. Centered on the fold the
    term's columns span at most 49 dimensions, so each column that does not raise the rank of the
    term's columns up to it (numpy's ``matrix_rank``, column by column) is spare; the 80 never tie
    at once, so each column's selection proportion is scikit-learn's path's own over the halves
    rebuilt from the documented rule."""
    from sklearn.linear_model import lars_path

    from turbotab.core.models.variable_selection import Selector, path_columns

    n, wide, q, pairs = 50, 80, 3, 10
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((n, wide)), columns=[f"w_{k}" for k in range(wide)])
    X["x0"] = rng.standard_normal(n)
    y = 2.0 * X["x0"].to_numpy() + rng.standard_normal(n)
    R = X.to_numpy()
    Z = (R - R.mean(axis=0)) / R.std(axis=0, ddof=1)
    ranks = [0, *(np.linalg.matrix_rank(Z[:, : j + 1]) for j in range(wide))]
    spare = [j for j in range(wide) if ranks[j + 1] == ranks[j]]
    path, codings = path_columns(Z, [list(range(wide)), [wide]])
    assert path.tolist() == list(range(wide + 1))
    assert codings == [(list(range(wide)), spare)] and spare == list(range(n - 1, wide))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        step = Selector("stability", "regression", q=q, pairs=pairs, seed=0,
                        inputs=["w", "x0"]).fit(X, y)
    rng = np.random.default_rng(0)
    counts = np.zeros(wide + 1)
    for _ in range(pairs):
        order = rng.permutation(n)
        for rows in (order[: n // 2], order[n // 2: 2 * (n // 2)]):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, _, coefs = lars_path(Z[rows], y[rows] - y[rows].mean(), method="lasso")
            counts += coefs[:, next(j for j in range(coefs.shape[1])
                                    if (coefs[:, j] != 0).sum() >= q)] != 0
    expected = counts / (2.0 * pairs)
    assert np.asarray(step.frequencies_, float) == pytest.approx(expected, abs=1e-12)
    assert step.selected_ == [t for t, f in (("w", expected[:wide].max()), ("x0", expected[wide]))
                              if f >= 0.6]


def test_a_class_of_copies_is_found_in_time_linear_in_its_size():
    """30,000 single-row indicators on 20 rows, each a term of its own, fall into 20 classes of
    about 1,500 exact copies: two standardized single-row indicators are ± copies exactly when they
    mark the same row. The path reads the first indicator of each row, found in under 5 s (about
    0.1 s idle; comparing every pair inside a class takes about 17 s)."""
    from turbotab.core.models.variable_selection import path_columns

    n, p = 20, 30_000
    row = np.random.default_rng(0).integers(0, n, p)
    D = np.zeros((n, p))
    D[row, np.arange(p)] = 1.0
    Z = (D - D.mean(axis=0)) / D.std(axis=0, ddof=1)
    started = time.perf_counter()
    path, codings = path_columns(Z, [[j] for j in range(p)])
    took = time.perf_counter() - started
    assert path.tolist() == sorted(int(np.flatnonzero(row == i)[0]) for i in range(n))
    assert not codings
    assert took < 5, f"{took:.1f} s"  # a loose guard: CI runs four at once
