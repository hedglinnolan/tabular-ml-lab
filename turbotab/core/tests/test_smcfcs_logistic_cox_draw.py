"""SMC-FCS's logistic and Cox outcome draws (the Sullivan 2021 crash sweep, B1; the verifier's
second-round problems).

The linear draw had been made from the QR decomposition of the design with unit-length columns
(``test_smcfcs_aliased_design``); the logistic and Cox draws still took the Cholesky factor of a
pseudo-inverse of the information plus 1e−12 I. That jitter is absolute: on income in dollars
beside a number (10,000 rows) the income coefficient's drawn variance was 1.9 times its inverse
information under the logistic model and 2.7 times under the Cox model, and on a code's indicators
beside an energy in thousandths of a kcal and the code's near recode the logistic factor raised
"Matrix is not positive definite". Both are now drawn as the linear draw is: the logistic from the
QR decomposition of its weighted design with unit-length columns, the Cox from the Cholesky factor
of its information on the standardized columns its fit uses, each mapped back, with no
pseudo-inverse and no jitter.

The Cox draw took its baseline hazard at η = Mβ*, multiplying by e^(−max η), which overflowed where
every η was below −709: a column on a large offset (100,000 plus a standard normal) with a negative
log hazard ratio, or a column near a combination of the others drawn far out along it. It is now
taken at η − max η, and each row's hazard as H e^(η − max η): the same product.

**Truth.** statsmodels' ``Logit`` and ``PHReg`` (Efron ties), fit on the design with unit-length
(logistic) or standardized (Cox) columns and mapped back: the model on XD⁻¹ with γ = Dβ is the same
model, so β̂ = D⁻¹γ̂ and the lower Cholesky factor of the inverse information is D⁻¹ times γ's (D
diagonal and positive); the Cox partial likelihood does not change when a column is centered. The
reference draw is β̂ + Lz with z the generator's normals, as the engine takes them. The cumulative
hazard of each row, H₀(t)e^η, does not change when a constant is added to every row's η (the
partial likelihood's risk-set sums scale by its exponential, and H₀ by its inverse), so the
reference hazard is Efron's sum written out here at η less its mean.

**Refused.** A Cox fit that did not converge has no maximum to center the draw on. Where the
partial likelihood has none (a group with no events: a monotone likelihood) the draw is refused in
those words; on the near-alias construction the engine's Cox fit stops at its first Newton step (a
coefficient 939 standard deviations out against its bound of 20) and the draw is refused as a fit
that found no stable maximum, as the analysis model's own Cox table refuses it. These are the words
where the design's columns are not named; named, the words name the terms that fail
(``test_smcfcs_cox_refusal``).
"""
from __future__ import annotations

import numpy as np
import pytest
import statsmodels.api as sm

from turbotab.core.methods import smcfcs
from turbotab.core.methods.imputation import ImputationRefused
from turbotab.core.methods.smcfcs import Outcome, _draw_outcome
from turbotab.core.tests import test_smcfcs_aliased_design as aliased


def _design(name: str, seed: int) -> tuple[np.ndarray, np.random.Generator]:
    """``"unit scale"``: two correlated numbers, a third, an indicator (400 rows). ``"kcal scale"``:
    a six-level code's five indicators, an energy in kcal, a number (400). ``"income in dollars"``:
    a number beside income ~ N(50,000, 20,000) (2,000). ``"offset"``: a number beside a column at
    100,000 plus a standard normal (400)."""
    rng = np.random.default_rng(seed)
    if name == "unit scale":
        z = rng.normal(size=(400, 3))
        return np.column_stack([z[:, 0], 0.5 * z[:, 0] + z[:, 1], z[:, 2],
                                (rng.random(400) < 0.3).astype(float)]), rng
    if name == "kcal scale":
        code = rng.integers(0, 6, 400)
        return np.column_stack([*[(code == k).astype(float) for k in range(1, 6)],
                                rng.normal(2000.0, 900.0, 400), rng.normal(size=400)]), rng
    if name == "income in dollars":
        return np.column_stack([rng.normal(size=2000), rng.normal(50_000.0, 20_000.0, 2000)]), rng
    return np.column_stack([rng.normal(size=400), 100_000.0 + rng.normal(size=400)]), rng


def _outcome(kind: str, M: np.ndarray, rng: np.random.Generator) -> Outcome:
    """A binary or survival outcome on the centered columns, log odds or log hazard ratios per SD
    from 0.3 down to −0.2 (the offset column's, −0.2 per unit, puts Mβ near −20,000 on every row)."""
    per_sd = np.linspace(0.3, -0.2, M.shape[1])
    eta = (M - M.mean(axis=0)) @ (per_sd / M.std(axis=0))
    if kind == "logistic":
        return Outcome("logistic", y=(rng.random(len(M)) < 1.0 / (1.0 + np.exp(-eta))).astype(float))
    time = rng.exponential(1.0 / (0.1 * np.exp(eta)))
    censor = rng.uniform(0.0, 20.0, len(M))
    return Outcome("cox", time=np.minimum(time, censor), event=(time <= censor).astype(float),
                   entry=np.zeros(len(M)))


def _logit_reference(M: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """statsmodels' Logit on [1, M] with unit-length columns, mapped back: (β̂ + Lz, the posterior
    SD of each coefficient), L the lower Cholesky factor of the inverse information."""
    X = np.column_stack([np.ones(len(M)), M])
    D = np.linalg.norm(X, axis=0)
    fit = sm.Logit(y, X / D).fit(disp=0, method="newton", tol=1e-12, maxiter=100)
    assert fit.mle_retvals["converged"]
    V = np.asarray(fit.cov_params())
    return (fit.params + np.linalg.cholesky(V) @ z) / D, np.sqrt(np.diag(V)) / D


def _cox_reference(M: np.ndarray, out: Outcome, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """statsmodels' PHReg (Efron ties) on M's standardized columns, mapped back, as
    :func:`_logit_reference`."""
    D = M.std(axis=0)
    fit = sm.PHReg(out.time, (M - M.mean(axis=0)) / D, status=out.event, ties="efron").fit(
        method="newton", tol=1e-12, maxiter=100)
    V = np.asarray(fit.cov_params())
    score = fit.model.score(fit.params)
    assert score @ V @ score < 1e-16  # the Newton decrement: at the maximum
    return (fit.params + np.linalg.cholesky(V) @ z) / D, np.sqrt(np.diag(V)) / D


def _efron_hazard(eta: np.ndarray, out: Outcome) -> np.ndarray:
    """Each row's cumulative hazard H₀(time)e^η − H₀(entry)e^η, Efron's sum written out: H₀(t) =
    Σ over event times τ ≤ t of Σ_{k<d} 1/(R(τ) − (k/d)D(τ)), R the risk set's Σe^η (entry < τ ≤
    time), D the events' at τ, d their count. Taken at η less its mean: the product is the same."""
    c = eta - eta.mean()
    r = np.exp(c)
    times = np.unique(out.time[out.event > 0])
    steps = np.zeros(len(times))
    for i, tau in enumerate(times):
        at = (out.entry < tau) & (out.time >= tau)
        ev = (out.time == tau) & (out.event > 0)
        d = int(ev.sum())
        steps[i] = sum(1.0 / (r[at].sum() - k / d * r[ev].sum()) for k in range(d))
    cum = np.concatenate([[0.0], np.cumsum(steps)])

    def H0(t: np.ndarray) -> np.ndarray:
        return cum[np.searchsorted(times, t, side="right")]

    return (H0(out.time) - H0(out.entry)) * r


@pytest.mark.parametrize("design", ["unit scale", "kcal scale", "income in dollars"])
@pytest.mark.parametrize("seed", range(6))
def test_the_logistic_draw_is_statsmodels_posterior_draw_from_the_same_generator(design, seed):
    """Per seed: β* is statsmodels' β̂ + Lz from the same generator state, to 2e−5 of each
    coefficient's posterior SD (the engine's ridge of 1e−8 per row moves it by at most 3.4e−6 on
    these). At the base commit the kcal-scale and income draws were off by up to 9.5e−5 and 0.18 SD
    (the absolute jitter)."""
    M, rng = _design(design, seed)
    out = _outcome("logistic", M, rng)
    got = _draw_outcome(M, out, np.random.default_rng(70 + seed)).beta
    z = np.random.default_rng(70 + seed).standard_normal(M.shape[1] + 1)
    star, sd = _logit_reference(M, out.y, z)
    assert np.max(np.abs(got - star) / sd) < 2e-5


@pytest.mark.parametrize("design", ["unit scale", "kcal scale", "income in dollars", "offset"])
@pytest.mark.parametrize("seed", range(6))
def test_the_cox_draw_is_phreg_s_posterior_draw_from_the_same_generator(design, seed):
    """Per seed: β* is statsmodels PHReg's β̂ + Lz from the same generator state, to 1e−8 of each
    coefficient's posterior SD (7.3e−10 at most on these). At the base commit the kcal-scale and
    income draws were off by up to 1.8e−4 and 0.20 SD (the absolute jitter), and the offset design's
    raised OverflowError."""
    M, rng = _design(design, seed)
    out = _outcome("cox", M, rng)
    got = _draw_outcome(M, out, np.random.default_rng(70 + seed)).beta
    star, sd = _cox_reference(M, out, np.random.default_rng(70 + seed).standard_normal(M.shape[1]))
    assert np.max(np.abs(got - star) / sd) < 1e-8


@pytest.mark.parametrize("design", ["offset", "kcal scale"])
@pytest.mark.parametrize("seed", range(10))
def test_each_row_s_cox_hazard_is_efron_s_at_the_draw(design, seed):
    """Each row's hazard as the sampler reads it, H e^(η − shift), is Efron's cumulative hazard at
    the drawn coefficients (:func:`_efron_hazard`), to 1e−9 of itself, on a design whose every η is
    below −709 (the offset column) and on one whose η is near zero. At the base commit the offset
    design's draw raised OverflowError."""
    M, rng = _design(design, seed)
    out = _outcome("cox", M, rng)
    fit = _draw_outcome(M, out, np.random.default_rng(90 + seed))
    eta = M @ fit.beta
    if design == "offset":
        assert eta.max() < -709  # e^(−max η) overflows
    got = fit.H * np.exp(smcfcs._eta(fit, M, "cox"))
    np.testing.assert_allclose(got, _efron_hazard(eta, out), rtol=1e-9)


@pytest.mark.parametrize("kind", ["logistic", "cox"])
def test_a_large_unit_s_drawn_variance_is_its_inverse_information(kind):
    """The verifier's construction: income in dollars beside a number, 10,000 rows. Over 1,000
    draws each coefficient's drawn variance is statsmodels' inverse information (Logit; PHReg with
    Efron ties) within 15% (the sample variance of 1,000 draws has a relative standard error of
    √(2/1000) = 4.5%). At the base commit the income coefficient's was 1.9 times it (logistic) and
    2.7 times (Cox)."""
    rng = np.random.default_rng(0)
    n = 10_000
    x = rng.normal(size=n)
    income = rng.normal(50_000.0, 20_000.0, n)
    M = np.column_stack([x, income])
    eta = -0.5 + 0.5 * x + 1e-5 * (income - 50_000.0)
    if kind == "logistic":
        out = Outcome("logistic", y=(rng.random(n) < 1.0 / (1.0 + np.exp(-eta))).astype(float))
        fit = sm.Logit(out.y, np.column_stack([np.ones(n), M])).fit(disp=0)
    else:
        time = rng.exponential(1.0 / (0.1 * np.exp(eta)))
        censor = rng.uniform(0.0, 20.0, n)
        out = Outcome("cox", time=np.minimum(time, censor), event=(time <= censor).astype(float),
                      entry=np.zeros(n))
        fit = sm.PHReg(out.time, M, status=out.event, ties="efron").fit()
    V = np.asarray(fit.cov_params())
    score = fit.model.score(fit.params)
    assert score @ V @ score < 1e-16  # the Newton decrement: at the maximum
    expected = np.diag(V)
    draws = np.asarray([_draw_outcome(M, out, np.random.default_rng(20_000 + k)).beta
                        for k in range(1000)])
    ratio = draws.var(axis=0) / expected
    assert np.all(np.abs(ratio - 1.0) < 0.15), ratio


def _spline_design(seed: int, normalize: bool) -> tuple[np.ndarray, np.ndarray]:
    """The verifier's Sullivan-like design (3,000 rows): a code's indicators, a recorded day of
    energy and a second near it, their mean recorded to 1e−3, a skewed minutes column and its
    restricted cubic spline terms (five knots at its 5th to 95th percentiles), unnormalized (cubes
    of minutes, up to 1e13) or divided by (k₅ − k₁)² as Harrell normalizes them. Returns the design
    and the spline columns' scale (1 unnormalized)."""
    rng = np.random.default_rng(seed)
    n = 3000
    code = rng.integers(0, 6, n)
    day1 = rng.gamma(4.0, 500.0, n)
    day2 = day1 * rng.lognormal(0.0, 0.25, n)
    mean = (day1 + day2) / 2 + rng.normal(0.0, 1e-3, n)
    minutes = rng.exponential(60.0, n) ** 1.5
    k = np.quantile(minutes, [0.05, 0.275, 0.5, 0.725, 0.95])

    def cube(v: np.ndarray) -> np.ndarray:
        return np.maximum(v, 0.0) ** 3

    gap = k[4] - k[3]
    terms = np.column_stack([cube(minutes - k[j]) - cube(minutes - k[3]) * (k[4] - k[j]) / gap
                             + cube(minutes - k[4]) * (k[3] - k[j]) / gap for j in range(3)])
    scale = (k[4] - k[0]) ** 2 if normalize else 1.0
    M = np.column_stack([*[(code == j).astype(float) for j in range(1, 6)], day1, day2, mean,
                         minutes, terms / scale])
    return M, np.full(3, scale)


@pytest.mark.parametrize("design", ["near alias", "spline terms"])
@pytest.mark.parametrize("seed", range(3))
def test_the_logistic_draw_does_not_depend_on_a_column_s_unit(design, seed):
    """Truth by derivation: the coefficient of a column multiplied by c is the original's divided by
    c, every other coefficient the same. The near-alias construction (a code's indicators, an energy,
    a number, the code's recode 1e−5 off their span) with the energy in kcal and in thousandths of a
    kcal: over ten draws each, the draws agree to 1e−6 of each coefficient. The Sullivan-like spline
    design with its cubic terms normalized and not: to 1e−4 of each coefficient's spread over the
    ten draws (1.4e−5 at most here), since the fit's ridge, 1e−8 per row on each coefficient in its
    column's own unit, is a share of the information along the spline terms' near-collinear
    direction that differs between the two scalings. At the base commit the logistic factor raised
    "Matrix is not positive definite" on the near-alias design in thousandths of a kcal, and drew
    the unnormalized cubes' coefficients with the jitter's variance, 1e−12, many times theirs."""
    if design == "near alias":
        M, _, rng = aliased._columns(seed, 300, "near", unit=1.0)
        big = M.copy()
        big[:, 5] *= 1000.0
        factor = np.ones(M.shape[1] + 1)
        factor[6] = 1000.0  # the energy's coefficient, behind β₀
        small, large = M, big
    else:
        small, scale = _spline_design(seed, normalize=True)
        large, _ = _spline_design(seed, normalize=False)
        factor = np.ones(small.shape[1] + 1)
        factor[-3:] = scale  # the unnormalized cubes are the normalized times the scale
        rng = np.random.default_rng(seed)
    eta = (small - small.mean(axis=0)) @ (np.linspace(0.3, -0.2, small.shape[1]) / small.std(axis=0))
    out = Outcome("logistic", y=(rng.random(len(small)) < 1.0 / (1.0 + np.exp(-eta))).astype(float))
    a = np.asarray([_draw_outcome(small, out, np.random.default_rng(s)).beta for s in range(10)])
    b = np.asarray([_draw_outcome(large, out, np.random.default_rng(s)).beta for s in range(10)])
    assert np.all(np.isfinite(b))
    if design == "near alias":
        np.testing.assert_allclose(b * factor, a, rtol=1e-6, atol=0)
    else:
        assert np.max(np.abs(b * factor - a) / a.std(axis=0)) < 1e-4


def test_a_cox_model_with_no_maximum_is_refused_in_words():
    """A group (a quarter of the rows) none of whose rows has the event: at every event time the
    rows with the event have the highest value of the group's indicator's negative, so its hazard
    ratio runs to zero and the partial likelihood has no maximum. The draw is refused in those
    words. The guard is silent on every design the PHReg test draws. At the base commit the draw
    went on from the Newton fit's last step."""
    rng = np.random.default_rng(4)
    n = 400
    group = (rng.random(n) < 0.25).astype(float)
    x = rng.normal(size=n)
    time = rng.exponential(1.0 / (0.1 * np.exp(0.3 * x)))
    censor = rng.uniform(0.0, 20.0, n)
    event = ((time <= censor) & (group == 0)).astype(float)
    out = Outcome("cox", time=np.minimum(time, censor), event=event, entry=np.zeros(n))
    with pytest.raises(ImputationRefused) as refused:
        _draw_outcome(np.column_stack([x, group]), out, np.random.default_rng(0))
    assert str(refused.value) == (
        "The Cox model's coefficients cannot be drawn for the imputations compatible with it, since "
        "its partial likelihood has no maximum on a completed copy of the data: at every event "
        "time the rows with the event have the highest value of one fixed combination of its "
        "columns among the rows at risk, so its hazard ratios run to zero or infinity.")


def near_columns(seed: int, n: int, share: float) -> tuple[np.ndarray, np.random.Generator]:
    """:func:`aliased._columns`' near-alias design with its recode moved off the span of the
    other columns by ``share`` of its length (1e−5 there), and the generator after it."""
    rng = np.random.default_rng(seed)
    code = rng.integers(0, aliased.LEVELS, n)
    indicators = np.column_stack([(code == k).astype(float) for k in range(1, aliased.LEVELS)])
    base = np.column_stack([indicators, rng.normal(2000.0, 900.0, n), rng.normal(size=n)])
    extra = aliased.RECODE[code]
    span = np.column_stack([np.ones(n), base])
    noise = rng.normal(size=n)
    noise -= span @ np.linalg.lstsq(span, noise, rcond=None)[0]
    extra = extra + noise * (share * np.linalg.norm(extra) / np.linalg.norm(noise))
    return np.column_stack([base, extra]), rng


@pytest.mark.parametrize("seed", range(20))
def test_the_cox_draw_on_the_near_alias_design_is_refused_only_without_a_stable_maximum(seed):
    """The near-alias construction with its own survival outcome: no column is aliased (R's lm
    keeps all). With the recode 1e−5 of its length off the others' span (:func:`aliased._columns`)
    the Cox likelihood has a maximum, whose standardized coefficients run to some 350 along the
    near-alias at a linear predictor of SD 0.9 (R's ``coxph`` converges to it, survival 3.8.6), so
    the draw is made, every coefficient finite, on every generator seed; on the tail ruling's base
    the fit stopped at its first Newton step, a coefficient past its bound of 20, and the draw was
    refused. With the recode 1e−6 of its length off the span the information at the estimate is
    singular to working precision (its smallest eigenvalue some 10⁻¹³ of its largest), so the draw
    is refused as a fit that found no stable maximum, never an untyped error. Before package B the
    draw went on from the first step's estimate, and raised OverflowError on 6 of these 20 seeds
    (e^(−max η), every η below −709)."""
    M, _, rng = aliased._columns(1, 300, "near", unit=1.0)
    out = aliased._outcome("cox", M, rng)
    drawn = _draw_outcome(M, out, np.random.default_rng(seed))
    assert np.all(np.isfinite(drawn.beta)) and np.all(np.isfinite(drawn.H))
    M, rng = near_columns(1, 300, 1e-6)
    out = aliased._outcome("cox", M, rng)
    with pytest.raises(ImputationRefused) as refused:
        _draw_outcome(M, out, np.random.default_rng(seed))
    assert str(refused.value) == (
        "The Cox model's coefficients cannot be drawn for the imputations compatible with it, since "
        "the Newton fit of its partial likelihood to a completed copy of the data found no stable "
        "maximum.")
