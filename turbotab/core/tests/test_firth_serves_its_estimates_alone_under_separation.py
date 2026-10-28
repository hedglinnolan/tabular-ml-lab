"""Wherever the data separate, a Firth table serves its penalized estimates and refuses every
profile quantity (V5c2-1; the methods lead's ruling of 2026-10-28, after T5aV2-1).

Held at a value, Firth's penalized likelihood on separated rows can have more than one maximum, and
the profile's warm-started fits (``inference.firth_profile``) can stop on a lower one, so a bound is
found inside the interval and a p-value overstated. T5c refused the profile under complete
separation only; under quasi-complete separation the base still served it, and wrong: on ``_tied(0,
120)`` `xs`'s upper bound was served at 19.0387, where a multi-start fit with `xs` held there has a
penalized likelihood ratio of 3.276 (under χ²₁(0.95) = 3.841), and on ``_tied(4, 120)`` `x4`'s
lower bound at −3.7631 (ratio 3.325). So every profile quantity is refused wherever
``separated_columns`` finds the data separated, complete or quasi: the table's intervals and
p-values, the intercept's, a combination's (``firth_combination``), the penalized likelihood-ratio
form tests (``exposure_form.firth_lr_test``) and the modification's (``methods.interaction``), in
words that name the separated columns (the cone ``separated_columns`` finds: leaving all of them
out ends the separation, V5c2-4) and say why. Where nothing separates, a Firth profile is served,
and right.

Truth: separated or not by construction (``xs = sign(y)·(0.3 + |d|)`` puts every row at least 0.3
on its own side; a row that repeats another's columns with the other outcome is on neither side
for any direction; an indicator every one of whose rows has the event), each checked by scipy's
linear program written here apart from the engine's. The estimates, standard errors and profiles
are Firth's penalized log-likelihood ``l(β) + ½ log|XᵀWX|`` (Firth 1993) and its gradient, the
modified score ``Xᵀ(y − μ + h(½ − μ))``, written out with numpy (a dense inverse of the
information) and maximized by scipy's BFGS from several starts (zero and seeded draws, never the
engine's estimate); a profile point is the highest of several starts' maxima with the coefficient
held there, and a bound the root of its ratio's crossing of χ²₁(0.95).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import linalg, optimize, stats
from scipy.special import expit

from turbotab.core.methods import exposure_form as ef
from turbotab.core.models import inference as I
from turbotab.core.models.inference import INDEPENDENT, inference_table

CRIT = float(stats.chi2.ppf(0.95, 1))
COLUMNS = ["x1", "xs", "x4"]
NAMES = ["(intercept)", *COLUMNS]
HELD = ("Firth's penalized likelihood can then have more than one maximum, and the fit cannot "
        "certify that it found the highest")


def _listed(names: list[str]) -> str:
    quoted = [f"`{n}`" for n in names]
    return quoted[0] if len(quoted) == 1 else f"{', '.join(quoted[:-1])} and {quoted[-1]}"


def words(names: list[str], held: str = "a coefficient held at a value",
          unreported: str = "profile interval or p-value") -> str:
    """The refusal a separated Firth table, or a form test on one, states: the separated columns
    and why nothing profiled is reported."""
    one = len(names) == 1
    return (f"{_listed(names)} {'separates' if one else 'separate'} the outcome: "
            f"{'a cut on it' if one else 'a cut on a combination of them'} puts every row on its "
            f"outcome's side or on the cut itself. With {held}, {HELD}, so no {unreported} is "
            f"reported.")


# ── the designs ──────────────────────────────────────────────────────────────


def _separated(seed: int, n: int) -> pd.DataFrame:
    """``n`` rows whose `xs` separates the outcome completely: ``xs = sign(y)·(0.3 + |d|)``."""
    rng = np.random.default_rng(100 + seed)
    x1, x4, d = rng.normal(size=(3, n))
    y = (rng.random(n) < 1 / (1 + np.exp(-(0.5 * x1 - 0.4 * x4)))).astype(int)
    xs = np.where(y == 1, 1.0, -1.0) * (0.3 + np.abs(d))
    return pd.DataFrame({"y": y, "x1": x1, "xs": xs, "x4": x4})


def _tied(seed: int, n: int) -> pd.DataFrame:
    """:func:`_separated` with one row more, the columns of the event row of least `xs` with no
    event: that pair's margins are ``m`` and ``−m`` along any direction, so no direction is strict
    on both, and the data separate quasi-completely."""
    rows = _separated(seed, n)
    nearest = rows.loc[rows["y"] == 1, "xs"].idxmin()
    return pd.concat([rows, rows.loc[[nearest]].assign(y=0)], ignore_index=True)


def _indicator(seed: int, force: bool = True) -> tuple[pd.DataFrame, np.ndarray]:
    """360 rows; `f` marks every ninth row. With ``force`` each of them has the event (the other
    rows hold both outcomes), so `f`'s maximum-likelihood log-odds is infinite: quasi-complete
    separation; without, nothing separates (checked where it is used)."""
    n = 360
    rng = np.random.default_rng(seed)
    x1, x2, x4 = rng.normal(size=(3, n))
    d = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(0.4 * x1 - 0.3 * x4 + 0.25 * x2)))).astype(int)
    f = (np.arange(n) % 9 == 0).astype(float)
    if force:
        y[f == 1] = 1
    return pd.DataFrame({"x1": x1, "x2": d, "x4": x4, "f": f}), y


def _design(matrix: pd.DataFrame) -> np.ndarray:
    return np.column_stack([np.ones(len(matrix)), matrix.to_numpy(dtype=float)])


DESIGNS: dict[str, Any] = {
    **{f"complete {seed}, {n} rows": (lambda s=seed, k=n: _frame_of(_separated(s, k)))
       for n in (120, 300) for seed in range(10)},
    **{f"tied {seed}": (lambda s=seed: _frame_of(_tied(s, 120))) for seed in (0, 4, 1, 2)},
    **{f"indicator {seed}": (lambda s=seed: _indicator(s)) for seed in (11, 8, 21)},
}


def _frame_of(rows: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    return rows[COLUMNS], rows["y"].to_numpy()


# ── the reference: scipy, written apart from the engine ─────────────────────


def _cone(X: np.ndarray, y: np.ndarray) -> list[int]:
    """The coefficients some separating direction moves (Konis 2007: ``s_i x_iᵀβ ≥ 0`` on every
    row, some row strict), each by scipy's linear program maximizing ``±β_j`` with β boxed in
    [−1, 1] on the columns scaled to at most one; a direction is believed only where its margins
    hold on the rows."""
    s = np.where(y > 0.5, 1.0, -1.0)
    scale = np.abs(X).max(axis=0)
    Xs = X / np.where(scale > 0, scale, 1.0)
    moved = []
    for j in range(X.shape[1]):
        for sign in (1.0, -1.0):
            c = np.zeros(X.shape[1])
            c[j] = -sign
            got = optimize.linprog(c, A_ub=-(s[:, None] * Xs), b_ub=np.zeros(len(y)),
                                   bounds=[(-1.0, 1.0)] * X.shape[1], method="highs")
            if got.status == 0 and sign * got.x[j] > 1e-7:
                margins = s * (Xs @ got.x)
                if margins.min() >= -1e-9 and margins.max() > 1e-7:
                    moved.append(j)
                    break
    return moved


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
    try:
        inverse = np.linalg.inv(X.T @ (X * w[:, None]))
    except np.linalg.LinAlgError:  # where the information is singular the loss is 10³⁰⁰
        return np.zeros(X.shape[1])
    h = w * np.einsum("ij,jk,ik->i", X, inverse, X)
    return X.T @ (y - mu + h * (0.5 - mu))


def _maximum(X: np.ndarray, y: np.ndarray, start: np.ndarray,
             held: dict[int, float] | None = None) -> tuple[np.ndarray, float]:
    """The penalized maximum from ``start`` by BFGS (twice), the coefficients ``held`` fixed."""
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

    def grad(z: np.ndarray) -> np.ndarray:
        return -_score(full(z), X, y)[free]

    found = optimize.minimize(loss, base[free], jac=grad, method="BFGS",
                              options={"gtol": 1e-10, "maxiter": 20_000})
    again = optimize.minimize(loss, found.x, jac=grad, method="BFGS",
                              options={"gtol": 1e-10, "maxiter": 20_000})
    best = again if again.fun <= found.fun else found
    return full(best.x), -float(best.fun)


def _starts(X: np.ndarray) -> list[np.ndarray]:
    """Zero and six seeded draws, each coefficient's on a scale of two over its column's largest
    value: never the engine's estimate."""
    rng = np.random.default_rng(0)
    scale = 2.0 / np.maximum(np.abs(X).max(axis=0), 1e-12)
    return [np.zeros(X.shape[1]), *(scale * rng.normal(size=X.shape[1]) for _ in range(6))]


def _reference(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """The highest penalized maximum over the starts, its value and its standard errors (the
    dense inverse of the information there)."""
    beta, top = max((_maximum(X, y, s) for s in _starts(X)), key=lambda r: r[1])
    mu = expit(X @ beta)
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ (X * (mu * (1 - mu))[:, None]))))
    return beta, top, se


def _held_top(X: np.ndarray, y: np.ndarray, held: dict[int, float], beta: np.ndarray) -> float:
    """The highest penalized log-likelihood with ``held`` fixed, from the starts and from the
    reference maximum (a start, not a value expected)."""
    return max(_maximum(X, y, s, held)[1] for s in [beta, *_starts(X)])


# ── V5c2-1: every profile quantity refused where the data separate ──────────


@pytest.mark.parametrize("design", list(DESIGNS))
def test_a_separated_firth_table_serves_its_estimates_and_refuses_every_profile_quantity(design):
    """Complete (`xs` 0.3 or more from the cut on every row), quasi-complete by one tied row, and
    quasi-complete by an indicator whose rows all have the event: each table is Firth's, refused
    for every interval and p-value (the intercept's too) in words that name the columns the
    separation check finds (the cone; under complete separation every column, under the tie
    every column, beside the indicator `f` alone) with their exit, and serves each estimate
    within 10⁻⁶ of a standard error of the independent maximum. On the base the tied and the
    indicator tables were served with profile intervals (`xs`'s bound at 19.0387 on ``tied 0``,
    inside the interval) and the complete ones named `xs` alone."""
    matrix, y = DESIGNS[design]()
    X = _design(matrix)
    cone = _cone(X, y.astype(float))
    named = [str(matrix.columns[j - 1]) for j in cone if j > 0]
    expected = list(matrix.columns) if not design.startswith("indicator") else ["f"]
    assert named == expected, (design, named)  # the construction's cone, by scipy's program
    table = inference_table("binary", matrix, y, [0, 1], INDEPENDENT, study=[matrix.columns[0]])
    assert table.info["estimator"] == I.FIRTH
    assert table.info["separated"] == expected
    assert table.info["refused"] == words(expected)
    assert table.info["covariance"] == "none"
    assert table.info["exits"] == [{"label": f"Leave {_listed(expected)} out, or merge its rare "
                                             f"levels", "decision": None}]
    assert table.concerns[0].startswith(f"{_listed(expected)} separate")
    assert words(expected) in table.concerns
    beta, _, se = _reference(X, y.astype(float))
    served = {r["feature"]: r for r in table.rows}
    assert list(served) == ["(intercept)", *matrix.columns]
    for j, name in enumerate(served):
        row = served[name]
        assert all(row[k] is None for k in ("se", "df", "ci_low", "ci_high", "p")), (name, row)
        assert abs(row["estimate"] - beta[j]) <= 1e-6 * se[j], (design, name, row["estimate"],
                                                               beta[j])


def test_held_at_its_served_bound_a_tied_tables_ratio_is_under_the_critical_value():
    """The premise, as a certificate from the independent fits: at the bound the base served for
    `xs` on ``tied 0`` (19.03870) and for `x4` on ``tied 4`` (−3.76311), a point with the
    coefficient held there has a penalized likelihood ratio under χ²₁(0.95), so those bounds were
    inside the interval; the engine's own fit held there from the free maximum finds the same
    lower ratio on ``tied 0``, and a lower maximum, at the critical value, on ``tied 4``."""
    for seed, j, bound in ((0, 2, 19.03869554765375), (4, 3, -3.76310697713731)):
        rows = _tied(seed, 120)
        X, y = _design(rows[COLUMNS]), rows["y"].to_numpy(dtype=float)
        beta, top, _ = _reference(X, y)
        ratio = 2.0 * (top - _held_top(X, y, {j: bound}, beta))
        assert ratio < CRIT - 0.4, (seed, ratio)


# ── the guard where it lives: the profile, a combination, the form test ─────


def _unseparated() -> tuple[np.ndarray, np.ndarray]:
    matrix, y = _indicator(11, force=False)
    X, y = _design(matrix), y.astype(float)
    assert _cone(X, y) == []  # nothing separates, by scipy's program
    return X, y


def test_on_separated_rows_every_profile_quantity_is_a_typed_refusal():
    """``firth_profile`` (strict or not, so for every caller), ``firth_combination`` with its
    profile and ``exposure_form.firth_lr_test`` each refuse on separated rows, in typed words
    (``FirthProfileSeparated``, a ``FirthProfileFailed``) that say why, rather than serve a bound,
    a p-value or a statistic. On the base each returned numbers."""
    for matrix, y in (_indicator(8), _frame_of(_tied(0, 120))):
        X, y = _design(matrix), y.astype(float)
        fit = I.firth_fit(X, y)
        assert fit.converged
        for strict in (True, False):
            with pytest.raises(I.FirthProfileSeparated) as caught:
                I.firth_profile(X, y, fit, 1, strict=strict, name="x1")
            assert isinstance(caught.value, I.FirthProfileFailed)
            assert str(caught.value) == (
                f"Firth's profile of `x1` is not computed: the data separate the outcome. With a "
                f"coefficient held at a value, {HELD}, so no interval or p-value is reported.")
        c = np.zeros(X.shape[1])
        c[1], c[3] = 1.0, 1.0
        with pytest.raises(I.FirthProfileSeparated):
            I.firth_combination(X, y, fit, c, profile=True, strict=True)
        with pytest.raises(I.FirthProfileSeparated) as caught:
            ef.firth_lr_test(X, y, [1, 3], fit)
        assert str(caught.value) == (
            f"The penalized likelihood-ratio test is not computed: the data separate the outcome. "
            f"With coefficients held at zero, {HELD}, so no test is reported.")
        # Without the profile a combination's estimate and standard error are still served.
        fields = I.firth_combination(X, y, fit, c, profile=False)
        assert fields["se"] == pytest.approx(math.sqrt(c @ fit.cov @ c), rel=1e-12)


def test_where_nothing_separates_firths_profile_is_served_and_right():
    """The silent side: on rows nothing separates, each coefficient's profile bounds are within
    10⁻⁶ of the root of the multi-start profile's ratio at χ²₁(0.95) and its p-value is the
    ratio's at zero to 10⁻⁶, and a combination's (`x1` + `x4`, held by its own
    parametrization) is too."""
    X, y = _unseparated()
    fit = I.firth_fit(X, y)
    assert fit.converged
    beta, top, se = _reference(X, y)
    assert np.all(np.abs(fit.beta - beta) <= 1e-6 * se)

    def check(low: float, high: float, p: float, ratio: Any, scale: float, what: str) -> None:
        for bound in (low, high):
            near = 1e-3 * scale
            assert ratio(bound - near) * ratio(bound + near) < 0, (what, bound)
            root = optimize.brentq(ratio, bound - near, bound + near, xtol=1e-12, rtol=1e-12)
            assert abs(bound - root) <= 1e-6, (what, bound, root)
        assert p == pytest.approx(float(stats.chi2.sf(max(0.0, ratio(0.0) + CRIT), 1)),
                                  rel=1e-6, abs=1e-12), what

    for j in (1, 3):
        low, high, p = I.firth_profile(X, y, fit, j, strict=True, name="x")
        check(low, high, p, lambda b, j=j: 2.0 * (top - _held_top(X, y, {j: b}, beta)) - CRIT,
              se[j], f"coefficient {j}")
    c = np.zeros(X.shape[1])
    c[1], c[3] = 1.0, 1.0
    N = linalg.null_space(c[None, :])

    def combination_top(b: float) -> float:
        # c'β = b: β = b·c/|c|² + N·δ with δ free, an independent parametrization of the constraint
        at = b * c / float(c @ c)

        def loss(z: np.ndarray) -> float:
            value = _penalized(at + N @ z, X, y)
            return -value if np.isfinite(value) else 1e300

        def grad(z: np.ndarray) -> np.ndarray:
            return -(N.T @ _score(at + N @ z, X, y))

        best = -np.inf
        for start in [N.T @ beta, *_starts(X @ N)]:
            found = optimize.minimize(loss, start, jac=grad, method="BFGS",
                                      options={"gtol": 1e-10, "maxiter": 20_000})
            found = optimize.minimize(loss, found.x, jac=grad, method="BFGS",
                                      options={"gtol": 1e-10, "maxiter": 20_000})
            best = max(best, -float(found.fun))
        return best

    fields = I.firth_combination(X, y, fit, c, profile=True, strict=True)
    assert fields["infinite"] is False
    mu = expit(X @ beta)
    spread = math.sqrt(c @ np.linalg.inv(X.T @ (X * (mu * (1 - mu))[:, None])) @ c)
    check(fields["ci_low"], fields["ci_high"], fields["p"],
          lambda b: 2.0 * (top - combination_top(b)) - CRIT, spread, "x1 + x4")


def test_where_nothing_separates_the_penalized_likelihood_ratio_test_is_served_and_right():
    """``firth_lr_test`` on rows nothing separates is ``2(l*(β̂) − max_{β₁ = β₃ = 0} l*(β))`` on
    χ²(2), the restricted maximum the highest of the starts', to 10⁻⁶."""
    X, y = _unseparated()
    fit = I.firth_fit(X, y)
    beta, top, _ = _reference(X, y)
    got = ef.firth_lr_test(X, y, [1, 3], fit)
    statistic = 2.0 * (top - _held_top(X, y, {1: 0.0, 3: 0.0}, beta))
    assert got["statistic"] == pytest.approx(statistic, rel=1e-6, abs=1e-9)
    assert (got["df_num"], got["distribution"]) == (2, "chi2")
    assert got["p"] == pytest.approx(float(stats.chi2.sf(statistic, 2)), rel=1e-6, abs=1e-12)


# ── V5c2-4: the words name the cone, and leaving it out ends the separation ──


def _two_indicators() -> tuple[pd.DataFrame, np.ndarray]:
    """:func:`_indicator`'s rows with a second indicator `g` on every ninth row from the fifth,
    each of which has the event too: `f` and `g` each separate the outcome (quasi-completely)."""
    matrix, y = _indicator(11)
    g = (np.arange(len(y)) % 9 == 4).astype(float)
    y = np.where(g == 1, 1, y)
    return matrix.assign(g=g), y


def test_the_words_and_the_exit_name_every_separated_column_and_leaving_them_out_ends_it():
    """Both ways (V5c2-4). Where `f` and `g` each separate the outcome on their own, the refusal
    and its exit name both (T5c named one set the separation could keep, and leaving it out was
    refused again naming the other); leaving `f` alone out is refused naming `g`, and leaving both
    out is served as a maximum-likelihood table, as statsmodels fits it. Where `xs` and `xt =
    2·xs + 0.1·sign(xs)·U` each separate completely, the cone is every column, and every column
    is named. Truth: by construction, checked by scipy's linear program written here."""
    import statsmodels.api as sm

    matrix, y = _two_indicators()
    X = _design(matrix)
    assert [matrix.columns[j - 1] for j in _cone(X, y.astype(float))] == ["f", "g"]
    table = inference_table("binary", matrix, y, [0, 1], INDEPENDENT, study=["x4"])
    assert table.info["refused"] == words(["f", "g"])
    assert table.info["exits"] == [{"label": "Leave `f` and `g` out, or merge its rare levels",
                                    "decision": None}]
    without_f = inference_table("binary", matrix.drop(columns="f"), y, [0, 1], INDEPENDENT,
                                study=["x4"])
    assert without_f.info["refused"] == words(["g"])
    rest = matrix.drop(columns=["f", "g"])
    assert _cone(_design(rest), y.astype(float)) == []
    served = inference_table("binary", rest, y, [0, 1], INDEPENDENT, study=["x4"])
    assert served.info["refused"] is None and served.info["covariance"] == "model"
    ml = sm.Logit(y, sm.add_constant(rest)).fit(disp=0)
    for row, b, s in zip(served.rows, ml.params, ml.bse):
        assert row["estimate"] == pytest.approx(b, abs=1e-6 * s) and row["se"] == pytest.approx(
            s, rel=1e-6)

    rows = _separated(2, 120)
    rng = np.random.default_rng(9)
    rows["xt"] = 2.0 * rows["xs"] + 0.1 * np.sign(rows["xs"]) * rng.random(len(rows))
    matrix = rows[["x1", "xs", "xt", "x4"]]
    table = inference_table("binary", matrix, rows["y"].to_numpy(), [0, 1], INDEPENDENT,
                            study=["x1"])
    assert table.info["refused"] == words(["x1", "xs", "xt", "x4"])
    assert table.info["exits"][0]["label"] == ("Leave `x1`, `xs`, `xt` and `x4` out, or merge its "
                                               "rare levels")
