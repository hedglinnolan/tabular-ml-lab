"""Under complete separation a Firth table serves its penalized estimates and refuses its profile
intervals and p-values (T5aV2-1; the methods lead's ruling of 2026-10-28).

Where some direction puts every row strictly on its own side (complete separation, Albert &
Anderson 1984), Firth's penalized likelihood with a coefficient held at a value can have more than
one maximum, and the profile's warm-started fits (``inference.firth_profile``) can stop on a lower
one: the drop is overstated and the bound is found inside the interval. On the base, at
``_separated(0)`` on 120 rows, `xs`'s upper bound was served at 16.5254, where a point with `xs`
held there has a penalized likelihood ratio of 3.023, under χ²₁(0.95) = 3.841 (its bound is near
19.4), and at ``_separated(4)`` on 120 rows `x4`, an ordinary covariate, had its lower bound
served at −2.8115 (ratio 3.603; the bound is near −3.05); 7 of the 14 tables the base served on
these 20 designs had such a bound. The fit cannot certify which maximum it found, so each such
table keeps its estimates and reports no interval or p-value, in words that name the separating
columns and say why. Where the separation is only quasi-complete (no direction is strict on every
row: an indicator every one of whose rows has the event), the profile is served as before, and
right.

Truth: complete or quasi-complete by construction (``xs = sign(y)·(0.3 + |d|)`` puts every row at
least 0.3 on its own side; a row that repeats another's columns with the other outcome is on
neither side strictly for any direction). The estimates, standard errors and profiles are Firth's
penalized log-likelihood ``l(β) + ½ log|XᵀWX|`` (Firth 1993) and its gradient, the modified score
``Xᵀ(y − μ + h(½ − μ))``, written out here with numpy (a dense inverse of the information), and
maximized by scipy's BFGS; a profile point is the highest of several starts' maxima with the
coefficient held there, and a bound the root of its ratio's crossing of χ²₁(0.95).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import optimize, stats
from scipy.special import expit

from turbotab.core.models import inference as I
from turbotab.core.models.inference import INDEPENDENT, inference_table

CRIT = float(stats.chi2.ppf(0.95, 1))
COLUMNS = ["x1", "xs", "x4"]
NAMES = ["(intercept)", *COLUMNS]
# The refusal names `xs` alone, the column of a separating direction none of whose columns can be
# left out (`x1` and `x4` are the outcome model's covariates), and ends at what is not reported:
# other surfaces quote it where no estimate is reported either (V5c-4, V5c-5).
WORDS = ("`xs` separates the outcome completely: a cut on it puts every row of each outcome on its "
         "own side. With a coefficient held at a value, Firth's penalized likelihood can then have "
         "more than one maximum, and the fit cannot certify that it found the highest, so no "
         "profile interval or p-value is reported.")


def _separated(seed: int, n: int) -> pd.DataFrame:
    """``n`` rows whose `xs` separates the outcome completely: ``xs = sign(y)·(0.3 + |d|)``
    (``test_floor_stops_fire_only_at_rounding._separated`` at any number of rows)."""
    rng = np.random.default_rng(100 + seed)
    x1, x4, d = rng.normal(size=(3, n))
    y = (rng.random(n) < 1 / (1 + np.exp(-(0.5 * x1 - 0.4 * x4)))).astype(int)
    xs = np.where(y == 1, 1.0, -1.0) * (0.3 + np.abs(d))
    return pd.DataFrame({"y": y, "x1": x1, "xs": xs, "x4": x4})


def _tied(seed: int, n: int) -> pd.DataFrame:
    """:func:`_separated` with one row more: the columns of the event row nearest the boundary
    (its least `xs`) with no event. That pair's margins are ``m`` and ``−m`` along any direction,
    so no direction is strict on both, while ``(−xs_a, 0, 1, 0)`` leaves every other row strictly
    on its side and the pair on the boundary: quasi-complete, by one tied row."""
    rows = _separated(seed, n)
    nearest = rows.loc[rows["y"] == 1, "xs"].idxmin()
    return pd.concat([rows, rows.loc[[nearest]].assign(y=0)], ignore_index=True)


def _quasi(seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    """360 rows (``test_iterative_tables_served_at_their_maximum._data``'s construction at
    another seed): `f` marks every ninth row, each of which has the event, and the other rows have
    both outcomes, so `f`'s maximum-likelihood log-odds is infinite and no direction is strict on
    the rows `f` leaves at zero."""
    n = 360
    rng = np.random.default_rng(seed)
    x1, x2, x4 = rng.normal(size=(3, n))
    d = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(0.4 * x1 - 0.3 * x4 + 0.25 * x2)))).astype(int)
    f = (np.arange(n) % 9 == 0).astype(float)
    y[f == 1] = 1
    return pd.DataFrame({"x1": x1, "x2": d, "x4": x4, "f": f}), y


def _design(matrix: pd.DataFrame) -> np.ndarray:
    return np.column_stack([np.ones(len(matrix)), matrix.to_numpy(dtype=float)])


def _penalized(beta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    eta = X @ beta
    mu = expit(eta)
    sign, logdet = np.linalg.slogdet(X.T @ (X * (mu * (1 - mu))[:, None]))
    if sign <= 0:
        return -np.inf
    return float(np.sum(y * eta - np.logaddexp(0.0, eta)) + 0.5 * logdet)


def _score(beta: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    mu = expit(X @ beta)
    w = mu * (1 - mu)
    h = w * np.einsum("ij,jk,ik->i", X, np.linalg.inv(X.T @ (X * w[:, None])), X)
    return X.T @ (y - mu + h * (0.5 - mu))


def _maximum(X: np.ndarray, y: np.ndarray, start: np.ndarray, held: dict[int, float] | None = None
             ) -> tuple[np.ndarray, float]:
    """The penalized maximum from ``start`` by BFGS, with the coefficients ``held`` fixed."""
    held = held or {}
    free = np.array([j for j in range(X.shape[1]) if j not in held], dtype=int)
    base = np.array(start, dtype=float)
    for j, value in held.items():
        base[j] = value

    def full(z: np.ndarray) -> np.ndarray:
        beta = base.copy()
        beta[free] = z
        return beta

    def loss(z: np.ndarray) -> float:
        value = _penalized(full(z), X, y)
        return -value if np.isfinite(value) else 1e300

    found = optimize.minimize(loss, base[free], jac=lambda z: -_score(full(z), X, y)[free],
                              method="BFGS", options={"gtol": 1e-10, "maxiter": 20_000})
    again = optimize.minimize(loss, found.x, jac=lambda z: -_score(full(z), X, y)[free],
                              method="BFGS", options={"gtol": 1e-10, "maxiter": 20_000})
    best = again if again.fun <= found.fun else found
    return full(best.x), -float(best.fun)


def _starts(beta: np.ndarray) -> list[np.ndarray]:
    """Several starts for a profile point: the maximum, zero, half and twice the maximum, and
    eight draws about it on its own scale (seeded)."""
    rng = np.random.default_rng(0)
    scale = np.maximum(np.abs(beta), 1.0)
    return [beta, np.zeros_like(beta), 0.5 * beta, 2.0 * beta,
            *(beta + rng.normal(size=len(beta)) * scale for _ in range(8))]


def _profile(X: np.ndarray, y: np.ndarray, j: int, b: float, starts: list[np.ndarray]) -> float:
    """The highest penalized log-likelihood with coefficient j held at b, over the starts."""
    return max(_maximum(X, y, s, {j: b})[1] for s in starts)


def _reference(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    beta, top = _maximum(X, y, np.zeros(X.shape[1]))
    mu = expit(X @ beta)
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ (X * (mu * (1 - mu))[:, None]))))
    return beta, top, se


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("n", [120, 300])
def test_a_completely_separated_firth_table_serves_its_estimates_and_no_profile(n, seed):
    """Every one of the 20 tables is refused for its intervals and p-values in words that name
    the separating column, `xs`, and say why, the intercept's row too, with the separation's exit
    for `xs` and its concern; each estimate is the penalized maximum's within 10⁻⁶ of a standard
    error. On the base 14 were served with profile intervals, 7 of them with a bound inside the
    interval, and 6 refused in other words (a profile fit that did not converge or a bound not
    crossed)."""
    rows = _separated(seed, n)
    table = inference_table("binary", rows[COLUMNS], rows["y"].to_numpy(), [0, 1], INDEPENDENT,
                            study=["x1"])
    assert table.info["estimator"] == I.FIRTH
    assert table.info["refused"] == WORDS
    assert table.info["separated"] == COLUMNS
    assert table.info["exits"] == [{"label": "Leave `xs` out, or merge its rare levels",
                                    "decision": None}]
    assert table.concerns[0].startswith("`x1`, `xs` and `x4` separate the outcome (the "
                                        "maximum-likelihood log-odds is infinite)")
    assert WORDS in table.concerns
    beta, _, se = _reference(_design(rows[COLUMNS]), rows["y"].to_numpy(dtype=float))
    served = {r["feature"]: r for r in table.rows}
    assert list(served) == NAMES
    for j, name in enumerate(NAMES):
        row = served[name]
        assert all(row[k] is None for k in ("se", "df", "ci_low", "ci_high", "p")), row
        assert abs(row["estimate"] - beta[j]) <= 1e-6 * se[j], (name, row["estimate"], beta[j])


@pytest.mark.parametrize("seed", [11, 8, 21])
def test_a_quasi_separated_firth_table_keeps_the_multistart_profile(seed):
    """The silent side: `f` separates the outcome quasi-completely, and the table is served with
    profile intervals and p-values, every bound (the intercept's too) within 10⁻⁶ of the root of
    the multi-start profile's ratio at χ²₁(0.95), every p-value its ratio at zero to 10⁻⁶, and every
    estimate the penalized maximum's within 10⁻⁶ of a standard error."""
    matrix, y = _quasi(seed)
    table = inference_table("binary", matrix, y, [0, 1], INDEPENDENT, study=["x4"])
    assert table.info["estimator"] == I.FIRTH and table.info["separated"] == ["f"]
    assert table.info["refused"] is None and table.info["covariance"] == "profile"
    X, y = _design(matrix), y.astype(float)
    beta, top, se = _reference(X, y)
    starts = _starts(beta)
    served = {r["feature"]: r for r in table.rows}
    for j, name in enumerate(["(intercept)", *matrix.columns]):
        row = served[name]
        assert abs(row["estimate"] - beta[j]) <= 1e-6 * se[j], name

        def ratio(b: float) -> float:
            return 2.0 * (top - _profile(X, y, j, b, [s.copy() for s in starts])) - CRIT

        for key in ("ci_low", "ci_high"):
            bound, near = row[key], 1e-3 * se[j]
            assert ratio(bound - near) * ratio(bound + near) < 0, (name, key, bound)
            root = optimize.brentq(ratio, bound - near, bound + near, xtol=1e-12, rtol=1e-12)
            assert abs(bound - root) <= 1e-6, (name, key, bound, root)
        p = float(stats.chi2.sf(max(0.0, ratio(0.0) + CRIT), 1))
        assert row["p"] == pytest.approx(p, rel=1e-6, abs=1e-12), name


def test_complete_separation_is_told_from_quasi_complete_separation():
    """Both ways, on designs whose kind the construction decides: complete where `xs` puts every
    row 0.3 or more on its side (at 120 and 300 rows); quasi-complete, though separated (the
    maximum-likelihood estimate infinite), where one row repeats another's columns with the other
    outcome, or where an indicator's rows all have the event; neither where nothing separates."""
    for n, seed in ((120, 0), (120, 4), (300, 3), (300, 9)):
        rows = _separated(seed, n)
        X, y = _design(rows[COLUMNS]), rows["y"].to_numpy(dtype=float)
        assert I.completely_separated(X, y), (n, seed)
        tied = _tied(seed, n)
        X, y = _design(tied[COLUMNS]), tied["y"].to_numpy(dtype=float)
        assert I.separated_columns(X, y) and not I.completely_separated(X, y), (n, seed)
    for seed in (11, 8, 21):
        matrix, y = _quasi(seed)
        X = _design(matrix)
        assert 4 in I.separated_columns(X, y) and not I.completely_separated(X, y), seed
    rng = np.random.default_rng(5)
    x = rng.normal(size=(200, 2))
    y = (rng.random(200) < expit(0.5 * x[:, 0])).astype(float)
    X = np.column_stack([np.ones(200), x])
    assert I.separated_columns(X, y) == [] and not I.completely_separated(X, y)


def _strictly_separate(X: np.ndarray, y: np.ndarray) -> bool:
    """Whether some direction puts every row at least 10⁻⁶ on its own side, ``s_i x_iᵀβ ≥ 10⁻⁶``
    with β boxed in [−10⁶, 10⁶] (scipy's linear program, written here apart from the engine's)."""
    s = np.where(y > 0.5, 1.0, -1.0)
    found = optimize.linprog(np.zeros(X.shape[1]), A_ub=-(s[:, None] * X),
                             b_ub=np.full(len(y), -1e-6), bounds=[(-1e6, 1e6)] * X.shape[1],
                             method="highs")
    return found.status == 0


def _sum_separates(seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    """200 rows whose outcome is ``x1 + x2 > 0`` with every row at least 0.3 from the cut, beside a
    noise column `x3`: the sum separates the outcome completely, and neither `x1` nor `x2` does
    alone (each one's events and non-events overlap, checked where it is used)."""
    rng = np.random.default_rng(300 + seed)
    x1, x2, x3 = rng.normal(size=(3, 600))
    keep = np.abs(x1 + x2) > 0.3
    x1, x2, x3 = x1[keep][:200], x2[keep][:200], x3[keep][:200]
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3}), (x1 + x2 > 0).astype(int)


@pytest.mark.parametrize("seed", range(4))
def test_the_words_name_the_columns_that_separate_and_no_other(seed):
    """Both ways (V5c-5): under complete separation every coefficient can move along some
    separating direction, so the separation check names every column; the words and the exit name
    the columns of a separating direction none of which can be left out. Where `xs` separates
    alone, `xs` alone, not the covariates `x1` and `x4` beside it; where only `x1 + x2` separates,
    both, and not the noise column `x3` beside them. Truth: by construction, and checked by scipy's
    linear program written here (each column alone, and each with `x3`, does not separate)."""
    rows = _separated(seed, 120)
    table = inference_table("binary", rows[COLUMNS], rows["y"].to_numpy(), [0, 1], INDEPENDENT,
                            study=["x1"])
    assert table.info["separated"] == COLUMNS
    assert table.info["refused"].startswith("`xs` separates the outcome completely: a cut on it "
                                            "puts every row of each outcome on its own side.")
    matrix, y = _sum_separates(seed)
    ones = np.ones((len(y), 1))
    for alone in (["x1"], ["x2"], ["x1", "x3"], ["x2", "x3"]):
        assert not _strictly_separate(np.hstack([ones, matrix[alone].to_numpy()]), y), alone
    assert _strictly_separate(np.hstack([ones, matrix[["x1", "x2"]].to_numpy()]), y)
    table = inference_table("binary", matrix, y, [0, 1], INDEPENDENT, study=["x1"])
    assert table.info["estimator"] == I.FIRTH
    assert table.info["refused"].startswith("`x1` and `x2` separate the outcome completely: a cut "
                                            "on a combination of them puts every row of each "
                                            "outcome on its own side. With a coefficient held")
    assert table.info["exits"] == [{"label": "Leave `x1` and `x2` out, or merge its rare levels",
                                    "decision": None}]
    assert all(r["ci_low"] is None and r["p"] is None for r in table.rows)
