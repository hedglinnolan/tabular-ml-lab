"""Acceptance: the NCI usual-intake method (V2 definition of done, "Dietary, extended"; wave 1,
package NCI).

    (1) Amount-only: Box-Cox transform chosen on the data, a mixed model with a person random effect
        and nuisance covariates (sequence of recall, weekend), back-transformation by Monte Carlo
        (or adaptive quadrature); on a simulation with a known usual-intake distribution the
        10th/50th/90th percentiles and the share below a cut-off are recovered within a stated bound
        (bias and coverage over ≥ 200 reps).
    (2) Two-part model for an episodic food: probability (logistic mixed) × amount (Box-Cox mixed)
        with correlated person effects; same recovery criteria; the correlation recovered.
    (3) Variance by balanced repeated replication or bootstrap (survey weights when present);
        against simulation coverage.
    (4) Routing: the dietary lens with ≥ 2 recalls on a subset offers the usual-intake distribution
        as its own estimand, distinct from association (regression calibration); consumers-only vs
        whole population stated (STROBE-nut nut-14).
    (5) The methods sentence names the model, the transform, the covariates and the variance method
        (asserted verbatim).

**No R package implements the NCI macros**, so for the distribution simulation truth is the
reference: each truth below is computed in this file from the data-generating parameters by its own
quadrature (``scipy.integrate.quad``, or NumPy's Gauss–Hermite nodes at a different order than the
engine's), never by the engine. Where an independent implementation exists it is used: statsmodels
``MixedLM`` and R's ``lme4::lmer`` (ML) for the amount-only model at a fixed λ; R's ``survey``
(``svydesign``, ``as.svrepdesign(type = "Fay", fay.rho = 0.3)``) and the two-PSU textbook variance for
balanced repeated replication; ``scipy.integrate.quad`` over the person effect, with SciPy's
multivariate normal density, for the two-part likelihood. R tests skip where ``Rscript`` is absent.

**Source checks** (read 2026-10-05; the module docstrings quote them at length):

* Tooze JA et al. *J Am Diet Assoc* 2006;106:1575 (PMC2517157): "the NCI method fits both parts
  simultaneously ... the two person-specific effects are modeled as correlated random variables";
  "the Box-Cox transformation parameter is also estimated as part of the likelihood maximization
  procedure"; "we generate 100 pseudo-persons for each individual in the sample".
* Tooze JA et al. *Stat Med* 2010;29:2857 (PMC3865776): "g(R_ij, λ) = X′_i β + u_i + e_ij"; "we
  propose the use of the nine-point approximation used by the ISU method"; "Standard errors for the
  NCI method were estimated using a simple bootstrap with 200 replications"; "Six recall days had 0
  reported intake of vitamin A; these were set to one-half of the lowest non-zero value."
* Kipnis V et al. *Biometrics* 2009;65:1003 (PMC2881223): the two-part model's equations; "T_i ≡
  E(T_ij | i) = p_i A_i"; "Standard errors (SEs) ... were estimated using the 'balanced repeated
  replication method' (Wolter, 1995)"; "Our two-part model assumes that each food is ultimately
  consumed by all individuals, so that T_i > 0."
* The NCI MIXTRAN and DISTRIB macros v2.1 (epi.grants.cancer.gov/diet/usualintakes): the weekend
  indicator ("A value of 1 represents a Fri.-Sun. record"), DISTRIB's "default weights for weekdays
  and weekend days are 4/7 and 3/7", the nine-point arrays ``cj``/``wj``, and the half-minimum
  floor ``mc_a = max(predcda,(.5*min_amt))``.

Monte Carlo error is stated beside every simulated bound.
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, optimize, special, stats

from turbotab.core.methods import usual_intake as nci

RSCRIPT = shutil.which("Rscript")
needs_r = pytest.mark.skipif(RSCRIPT is None, reason="Rscript is not installed")


# ── independent helpers (the test's own, never the engine's) ─────────────────


def ginv(y, lam):
    """The Box-Cox inverse, written out: (λy + 1)^{1/λ}, exp at λ = 0."""
    y = np.asarray(y, dtype=float)
    if lam == 0:
        return np.exp(y)
    return np.maximum(lam * y + 1.0, 0.0) ** (1.0 / lam)


def boxcox_ref(r, lam):
    r = np.asarray(r, dtype=float)
    return np.log(r) if lam == 0 else (r ** lam - 1.0) / lam


def amount_given(v, lam, se):
    """E g⁻¹(v + ε), ε ~ N(0, se²), by adaptive quadrature (scipy.integrate.quad)."""
    f = lambda e: float(ginv(v + e, lam)) * stats.norm.pdf(e, scale=se)
    return integrate.quad(f, -12 * se, 12 * se, limit=200, epsabs=1e-12, epsrel=1e-11)[0]


def gh(n):
    z, w = np.polynomial.hermite_e.hermegauss(n)
    return z, w / math.sqrt(2 * math.pi)


def amount_vector(v, lam, se, n=80):
    """E g⁻¹(v + ε) for an array ``v``, by NumPy's 80-node Gauss–Hermite rule (the engine uses
    32 nodes; this is a different rule, checked against ``amount_given`` below)."""
    z, w = gh(n)
    return ginv(np.asarray(v)[..., None] + se * z, lam) @ w


def run_r(tmp_path: Path, script: str, data: pd.DataFrame | None = None) -> dict:
    """Run ``script`` with Rscript; it reads ``data.csv`` and prints one JSON object."""
    if data is not None:
        data.to_csv(tmp_path / "data.csv", index=False)
    (tmp_path / "script.R").write_text(script)
    out = subprocess.run([RSCRIPT, "--vanilla", str(tmp_path / "script.R")], cwd=tmp_path,
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


# ── the amount-only simulation (known truth) ─────────────────────────────────

LAM, MU, SU, SE, SEQ, WK = 0.3, 8.0, 1.0, 1.4, -0.3, 0.4
CUT = 50.0  # near the 20th percentile of the true distribution


def simulate_amount(rng: np.random.Generator, n: int = 1000, p2: float = 0.7, *, su: float = SU,
                    u: np.ndarray | None = None) -> nci.Recalls:
    """``n`` people, 70% with two recalls: on the Box-Cox scale (λ = 0.3) intake is μ + u_i + a
    repeat-recall shift −0.3 + a weekend shift 0.4 + day-to-day error (sd 1.4), u_i ~ N(0, 1).
    A Friday–Sunday recall has probability 3/7."""
    k = np.where(rng.random(n) < p2, 2, 1)
    person = np.repeat(np.arange(n), k)
    later = np.concatenate([[0] + [1] * (kk - 1) for kk in k]).astype(float)
    weekend = (rng.random(len(person)) < 3 / 7).astype(float)
    u = rng.normal(0, su, n) if u is None else u
    y = MU + u[person] + SEQ * later + WK * weekend + rng.normal(0, SE, len(person))
    return nci.Recalls.of(ginv(y, LAM), person, n, later=later, weekend=weekend)


def usual_of(u: float, su_shift: float = 0.0) -> float:
    """The true usual intake of a person with effect ``u``: at a first recall, 4/7 weekday and 3/7
    weekend (DISTRIB's convention), each the conditional mean over the day-to-day error."""
    return 4 / 7 * amount_given(MU + u, LAM, SE) + 3 / 7 * amount_given(MU + WK + u, LAM, SE)


def amount_truth(percentiles=(10, 50, 90), cutoff=CUT, su=SU) -> tuple[dict[int, float], float]:
    """Usual intake rises with u, so its p-th percentile is T(σ_u z_p) exactly and the share below
    c is Φ(u_c/σ_u) with T(u_c) = c (brentq)."""
    truth = {p: usual_of(su * stats.norm.ppf(p / 100)) for p in percentiles}
    uc = optimize.brentq(lambda u: usual_of(u) - cutoff, -8 * su, 8 * su, xtol=1e-12)
    return truth, float(stats.norm.cdf(uc / su))


# ── 1 · the amount-only model ────────────────────────────────────────────────


def _mixedlm(y, later, weekend, person):
    import statsmodels.formula.api as smf

    long = pd.DataFrame({"y": y, "later": later, "weekend": weekend, "g": person})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return smf.mixedlm("y ~ later + weekend", long, groups=long["g"]).fit(
            reml=False, method="bfgs", gtol=1e-10)


@pytest.mark.parametrize("lam", [0.0, 0.3, 0.7])
def test_1a_at_a_fixed_lambda_the_fit_is_statsmodels_maximum_likelihood(lam):
    """At a fixed λ the amount-only model is a linear mixed model of the Box-Cox recalls on the
    nuisance covariates with a person random intercept. statsmodels ``MixedLM`` (ML) on the same
    transformed recalls gives the same fixed effects, σ²_u and σ²_e (to 10⁻⁵, the BFGS optimum's
    precision), and its log-likelihood plus the Box-Cox Jacobian (λ − 1) Σ log R is the engine's."""
    rec = simulate_amount(np.random.default_rng(21), n=600)
    X, names = rec.design()
    fit = nci.fit_amount(rec.amount, rec.person, X, rec.n_persons, np.ones((1, rec.n_persons)),
                         names, lam=lam)
    m = _mixedlm(boxcox_ref(rec.amount, lam), rec.later, rec.weekend, rec.person)
    assert names == ["intercept", "repeat recall", "weekend"]
    assert fit.beta[0] == pytest.approx(m.fe_params.to_numpy(), rel=1e-5, abs=1e-6)
    assert fit.sigma2_u[0] == pytest.approx(float(m.cov_re.iloc[0, 0]), rel=1e-5)
    assert fit.sigma2_e[0] == pytest.approx(float(m.scale), rel=1e-5)
    assert fit.loglik[0] == pytest.approx(m.llf + (lam - 1) * np.log(rec.amount).sum(), abs=1e-6)


def test_1a_lambda_is_the_profile_maximum_of_the_likelihood():
    """λ is chosen on the data by maximum likelihood (Tooze et al. 2006: "estimated as part of the
    likelihood maximization procedure"): statsmodels' ML log-likelihood plus the Jacobian, computed
    by statsmodels alone, equals the engine's at λ̂ and is lower at λ̂ ± 0.03 and ± 0.1."""
    rec = simulate_amount(np.random.default_rng(22), n=500)
    X, names = rec.design()
    fit = nci.fit_amount(rec.amount, rec.person, X, rec.n_persons, np.ones((1, rec.n_persons)), names)
    lam = float(fit.lam[0])

    def profile(g: float) -> float:
        return (_mixedlm(boxcox_ref(rec.amount, g), rec.later, rec.weekend, rec.person).llf
                + (g - 1) * np.log(rec.amount).sum())

    at = profile(lam)
    assert fit.loglik[0] == pytest.approx(at, abs=1e-6)
    for step in (-0.1, -0.03, 0.03, 0.1):
        assert profile(lam + step) < at, step


def test_1a_integer_weights_are_people_counted_that_many_times():
    """Weights enter as a pseudo-likelihood: a person of weight 3 counts as three people. statsmodels
    on the data with each person repeated (as separate people) gives the weighted fit."""
    rng = np.random.default_rng(23)
    rec = simulate_amount(rng, n=300)
    w = rng.integers(1, 4, rec.n_persons).astype(float)
    X, names = rec.design()
    fit = nci.fit_amount(rec.amount, rec.person, X, rec.n_persons, w[None, :], names, lam=0.3)
    rows = np.concatenate([np.flatnonzero(rec.person == i) for i in range(rec.n_persons)
                           for _ in range(int(w[i]))])
    copies = np.concatenate([np.full(int((rec.person == i).sum()), c)
                             for c, i in enumerate(i for i in range(rec.n_persons)
                                                   for _ in range(int(w[i])))])
    m = _mixedlm(boxcox_ref(rec.amount[rows], 0.3), rec.later[rows], rec.weekend[rows], copies)
    assert fit.beta[0] == pytest.approx(m.fe_params.to_numpy(), rel=1e-5, abs=1e-6)
    assert fit.sigma2_u[0] == pytest.approx(float(m.cov_re.iloc[0, 0]), rel=1e-5)
    assert fit.sigma2_e[0] == pytest.approx(float(m.scale), rel=1e-5)


@needs_r
def test_1a_r_lme4_agrees_at_a_fixed_lambda(tmp_path):
    """R's ``lme4::lmer(y ~ later + weekend + (1 | person), REML = FALSE)`` on the Box-Cox
    recalls: the same fixed effects, variance components and log-likelihood (to 10⁻⁵)."""
    rec = simulate_amount(np.random.default_rng(24), n=500)
    X, names = rec.design()
    fit = nci.fit_amount(rec.amount, rec.person, X, rec.n_persons, np.ones((1, rec.n_persons)),
                         names, lam=0.3)
    data = pd.DataFrame({"y": boxcox_ref(rec.amount, 0.3), "later": rec.later,
                         "weekend": rec.weekend, "person": rec.person})
    r = run_r(tmp_path, """
suppressMessages(library(lme4))
d <- read.csv("data.csv"); d$person <- factor(d$person)
m <- lmer(y ~ later + weekend + (1 | person), data = d, REML = FALSE,
          control = lmerControl(optimizer = "bobyqa", optCtrl = list(rhoend = 1e-12)))
vc <- as.data.frame(VarCorr(m))
cat(sprintf('{"beta": [%.12g, %.12g, %.12g], "s2u": %.12g, "s2e": %.12g, "ll": %.12g}\\n',
    fixef(m)[1], fixef(m)[2], fixef(m)[3], vc$vcov[1], vc$vcov[2], as.numeric(logLik(m))))
""", data)
    assert fit.beta[0] == pytest.approx(r["beta"], rel=1e-5, abs=1e-6)
    assert fit.sigma2_u[0] == pytest.approx(r["s2u"], rel=1e-4)
    assert fit.sigma2_e[0] == pytest.approx(r["s2e"], rel=1e-5)
    assert fit.loglik[0] == pytest.approx(r["ll"] + (0.3 - 1) * np.log(rec.amount).sum(), abs=1e-5)


@pytest.mark.parametrize("lam", [0.0, 0.15, 0.5, 1.0])
def test_1b_the_back_transformation_is_the_conditional_mean(lam):
    """A usual consumption-day amount is E g⁻¹(v + ε). The engine's integral equals adaptive
    quadrature (``scipy.integrate.quad``) to 10⁻⁷ relative, including where g⁻¹'s kink at zero sits
    within a standard deviation of v (λ = 0.5, v = −1), and at λ = 0 it is the lognormal mean
    exp(v + σ²/2) that DISTRIB itself uses there (``predcda = exp(mc_bca&wk + a_var_e&ve / 2)``).
    For λ > 0 DISTRIB's nine-point rule (its documented arrays) agrees within 10⁻³ (measured: 10⁻⁶
    to 4·10⁻⁴, the largest at λ = 1 where g⁻¹'s kink at zero lies inside the nodes)."""
    se = 1.1
    v = np.array([-1.0, 2.0, 5.0, 9.0]) if lam else np.array([-1.0, 0.5, 2.0, 3.5])
    engine = nci.back_transform(v, lam, se, floor=0.0)
    ref = np.array([amount_given(x, lam, se) for x in v])
    assert engine == pytest.approx(ref, rel=1e-7, abs=1e-12)
    if lam == 0:
        assert engine == pytest.approx(np.exp(v + se ** 2 / 2), rel=1e-12)
        return
    nine = np.array([(ginv(x + nci.NINE_POINT_C * se, lam) * nci.NINE_POINT_W).sum() for x in v])
    keep = ref > 1.0
    assert np.max(np.abs(nine[keep] / ref[keep] - 1)) < 1e-3


def test_1c_the_integrated_distribution_is_a_monte_carlo_of_the_same_model():
    """DISTRIB simulates pseudo-persons; the engine integrates. For one fitted model, two million
    random pseudo-persons (each person effect drawn, each usual intake by the test's 80-node rule,
    4/7 weekday and 3/7 weekend at a first recall) give the same percentiles, mean and share below
    the cut-off within their Monte Carlo error (percentiles to 0.3%, share to 0.002)."""
    rec = simulate_amount(np.random.default_rng(25))
    res = nci.usual_intake(rec, "amount_only", cutoff=CUT)
    p = res.parameters
    u = np.random.default_rng(26).normal(0, math.sqrt(p["sigma2_u"]), 2_000_000)
    b0, bw, lam, se = p["beta_intercept"], p["beta_weekend"], p["lambda"], math.sqrt(p["sigma2_e"])
    T = (4 / 7 * np.maximum(amount_vector(b0 + u, lam, se), res.floor)
         + 3 / 7 * np.maximum(amount_vector(b0 + bw + u, lam, se), res.floor))
    for q, e in res.percentiles.items():
        assert e.value == pytest.approx(np.percentile(T, q), rel=0.003), q
    assert res.mean.value == pytest.approx(T.mean(), rel=0.002)
    assert res.below.value == pytest.approx(np.mean(T < CUT), abs=0.002)


REPS = 200


@pytest.fixture(scope="module")
def amount_runs():
    """200 datasets of 1,000 people (70% with two recalls), each fitted with a 50-replicate person
    bootstrap (the engine's default is 200, Tooze et al. 2010; 50 keeps the study to about a
    minute, and a coarser bootstrap only widens the intervals' own noise)."""
    rng = np.random.default_rng(2026)
    out = {"p": {q: [] for q in (10, 50, 90)}, "cov": {q: 0 for q in (10, 50, 90)},
           "below": [], "cov_below": 0, "naive10": [], "day10": [], "lam": []}
    truth, share = amount_truth()
    for r in range(REPS):
        rec = simulate_amount(rng)
        res = nci.usual_intake(rec, "amount_only", cutoff=CUT, percentiles=(10, 50, 90),
                               replication=nci.person_bootstrap(rec.n_persons, 50, seed=r))
        for q in (10, 50, 90):
            e = res.percentiles[q]
            out["p"][q].append(e.value)
            out["cov"][q] += e.ci_low <= truth[q] <= e.ci_high
        out["below"].append(res.below.value)
        out["cov_below"] += res.below.ci_low <= share <= res.below.ci_high
        out["naive10"].append(res.mean_of_days[10])
        out["day10"].append(res.day_one[10])
        out["lam"].append(res.lam)
    out["truth"], out["share"] = truth, share
    return out


def test_1d_the_simulation_recovers_the_known_usual_intake_distribution(amount_runs):
    """Bias over 200 datasets: each of the 10th, 50th and 90th percentiles within 1.5% of the truth
    and within 4 Monte Carlo standard errors of it (observed: about 0.5%), and the share below 50
    within 0.006 of the truth. λ averages within 0.02 of 0.3."""
    out = amount_runs
    lines = []
    for q in (10, 50, 90):
        est = np.array(out["p"][q])
        bias = est.mean() - out["truth"][q]
        mcse = est.std(ddof=1) / math.sqrt(REPS)
        lines.append(f"p{q}: truth {out['truth'][q]:.2f} bias {bias:+.3f} (MC se {mcse:.3f})")
        assert abs(bias) < 0.015 * out["truth"][q]
        assert abs(bias) < 4 * mcse
    below = np.array(out["below"])
    bias_b = below.mean() - out["share"]
    lines.append(f"share<50: truth {out['share']:.4f} bias {bias_b:+.4f} "
                 f"(MC se {below.std(ddof=1) / math.sqrt(REPS):.4f})")
    print("\n" + "\n".join(lines))
    assert abs(bias_b) < 0.006
    assert abs(np.mean(out["lam"]) - LAM) < 0.02


def test_1d_the_bootstrap_intervals_cover(amount_runs):
    """Coverage of the 95% intervals (estimate ± 1.96 bootstrap SE) over 200 datasets: at least 0.90
    for each percentile and for the share (the binomial Monte Carlo error at 0.95 is 0.015)."""
    out = amount_runs
    cover = {**{f"p{q}": out["cov"][q] / REPS for q in (10, 50, 90)}, "share": out["cov_below"] / REPS}
    print("\ncoverage", cover)
    assert all(v >= 0.90 for v in cover.values()), cover


def test_1d_single_days_and_means_of_days_are_not_usual_intake(amount_runs):
    """Why the method exists (NUTRITION_PACK §03: "the tails are too fat"): the 10th percentile of
    the mean of each person's recalls sits more than 15% below the true usual-intake 10th percentile,
    and a single day's lower still; the NCI estimate does not."""
    out = amount_runs
    t10 = out["truth"][10]
    assert np.mean(out["naive10"]) < 0.85 * t10
    assert np.mean(out["day10"]) < np.mean(out["naive10"])
    assert abs(np.mean(out["p"][10]) / t10 - 1) < 0.015


def test_1e_zero_days_become_half_the_smallest_amount_and_are_counted():
    """MIXTRAN: "for amount models change 0 intake to half the smallest amount actually eaten"."""
    rec = simulate_amount(np.random.default_rng(27), n=400)
    amount = rec.amount.copy()
    amount[:5] = 0.0
    rec0 = nci.Recalls(amount, rec.person, rec.n_persons, rec.later, rec.weekend)
    res = nci.usual_intake(rec0, "amount_only")
    assert res.zeros_replaced == 5
    assert res.floor == pytest.approx(0.5 * amount[amount > 0].min())


def test_1e_without_two_people_of_two_recalls_there_is_nothing_to_separate():
    rng = np.random.default_rng(28)
    rec = simulate_amount(rng, n=300, p2=0.0)
    with pytest.raises(nci.UsualIntakeRefused, match="at least two positive recalls"):
        nci.usual_intake(rec, "amount_only")


# ── 2 · the two-part model ───────────────────────────────────────────────────

LAM2, A0, B0, S1, S2, RHO, SE2 = 0.25, -0.2, 4.0, 1.0, 0.6, 0.5, 0.9
AWK, BWK, ASEQ, BSEQ = 0.3, 0.2, -0.2, -0.1
CUT2 = 10.0


def simulate_two_part(rng: np.random.Generator, n: int = 2000, p2: float = 0.85) -> nci.Recalls:
    """An episodically consumed food: on each recall day a person eats it with probability
    expit(−0.2 + 0.3·weekend − 0.2·repeat + u₁) and then eats g⁻¹(4 + 0.2·weekend − 0.1·repeat + u₂ + ε)
    (λ = 0.25, ε sd 0.9), with (u₁, u₂) normal, sd 1 and 0.6, correlation 0.5. About half the
    recalls are zero. 85% of people have two recalls."""
    k = np.where(rng.random(n) < p2, 2, 1)
    person = np.repeat(np.arange(n), k)
    later = np.concatenate([[0] + [1] * (kk - 1) for kk in k]).astype(float)
    weekend = (rng.random(len(person)) < 3 / 7).astype(float)
    z1, z2 = rng.normal(size=n), rng.normal(size=n)
    u1, u2 = S1 * z1, S2 * (RHO * z1 + math.sqrt(1 - RHO ** 2) * z2)
    p = special.expit(A0 + AWK * weekend + ASEQ * later + u1[person])
    eat = rng.random(len(person)) < p
    y = B0 + BWK * weekend + BSEQ * later + u2[person] + rng.normal(0, SE2, len(person))
    return nci.Recalls.of(np.where(eat, ginv(y, LAM2), 0.0), person, n, later=later, weekend=weekend)


def amount_curve(lam: float, se: float, lo: float, hi: float, n: int = 1201):
    """E g⁻¹(v + ε) on a grid of v by adaptive quadrature, for interpolation."""
    v = np.linspace(lo, hi, n)
    return v, np.array([amount_given(x, lam, se) for x in v])


def two_part_monte_carlo(alpha, beta, s1, s2, rho, lam, se, floor=0.0, draws=4_000_000, seed=31):
    """Usual intake of ``draws`` random people: T = 4/7 expit(a₀ + u₁) A(b₀ + u₂) + 3/7 (weekend),
    at a first recall, A by adaptive quadrature on a grid (DISTRIB's pseudo-persons, more of them)."""
    r = np.random.default_rng(seed)
    z1, z2 = r.normal(size=draws), r.normal(size=draws)
    u1, u2 = s1 * z1, s2 * (rho * z1 + math.sqrt(1 - rho ** 2) * z2)
    v, a = amount_curve(lam, se, beta[0] - 7 * s2, beta[0] + beta[2] + 7 * s2)
    T = np.zeros(draws)
    for share, wk in ((4 / 7, 0.0), (3 / 7, 1.0)):
        amount = np.maximum(np.interp(beta[0] + wk * beta[2] + u2, v, a), floor)
        T += share * special.expit(alpha[0] + wk * alpha[2] + u1) * amount
    return T


@pytest.fixture(scope="module")
def two_part_truth():
    T = two_part_monte_carlo([A0, ASEQ, AWK], [B0, BSEQ, BWK], S1, S2, RHO, LAM2, SE2)
    return {q: float(np.percentile(T, q)) for q in (10, 50, 90)}, float(np.mean(T < CUT2))


def test_2a_the_likelihood_integrates_the_person_effect_correctly():
    """Each person's likelihood is ∫ Π_j p_j(z)^I (1 − p_j(z))^{1−I} × f(amounts | z) × Π R^{λ−1}
    φ(z) dz, the amounts jointly normal given u₁ = σ₁z (mean shifted by ρσ₂z, covariance σ²_e I +
    σ²₂(1 − ρ²) J). Written here with SciPy's multivariate normal density and integrated by
    ``scipy.integrate.quad``, it equals the engine's 30-node Gauss–Hermite value to 10⁻⁶ for 40 people
    of every kind (none, one or two consumption days)."""
    rec = simulate_two_part(np.random.default_rng(41), n=1500)
    fit = nci.fit_two_part(rec)
    engine = nci.person_loglik(rec, fit)
    a, b = fit.alpha, fit.beta
    names = fit.names
    assert names == ["intercept", "repeat recall", "weekend"]
    rng = np.random.default_rng(42)
    for i in rng.choice(rec.n_persons, 40, replace=False):
        rows = rec.person == i
        R, lt, wk = rec.amount[rows], rec.later[rows], rec.weekend[rows]
        eaten = R > 0

        def integrand(z: float) -> float:
            p = special.expit(a[0] + a[1] * lt + a[2] * wk + fit.sigma_u1 * z)
            lik = float(np.prod(np.where(eaten, p, 1 - p)))
            if eaten.any():
                k = int(eaten.sum())
                mean = b[0] + b[1] * lt[eaten] + b[2] * wk[eaten] + fit.rho * fit.sigma_u2 * z
                cov = (fit.sigma_e ** 2 * np.eye(k)
                       + fit.sigma_u2 ** 2 * (1 - fit.rho ** 2) * np.ones((k, k)))
                lik *= stats.multivariate_normal.pdf(boxcox_ref(R[eaten], fit.lam), mean, cov)
                lik *= float(np.prod(R[eaten] ** (fit.lam - 1)))
            return lik * stats.norm.pdf(z)

        ref = math.log(integrate.quad(integrand, -12, 12, limit=200, epsabs=0, epsrel=1e-11)[0])
        assert engine[i] == pytest.approx(ref, abs=1e-6), (i, int(eaten.sum()))


def test_2a_the_maximum_is_a_maximum():
    """At the fit the log-likelihood's analytic gradient vanishes, and stepping any parameter by
    ±10⁻³ lowers the likelihood (computed by the per-person sums above)."""
    rec = simulate_two_part(np.random.default_rng(43), n=1200)
    fit = nci.fit_two_part(rec)
    at = nci.person_loglik(rec, fit).sum()
    theta = fit.theta()
    for j in range(len(theta)):
        for step in (-1e-3, 1e-3):
            t = theta.copy()
            t[j] += step
            moved = nci._two_part_loglik(t, nci._TwoPartData(rec), np.ones(rec.n_persons),
                                         gradient=False)[0]
            assert moved < at, (j, step)


def test_2c_the_integrated_distribution_is_a_monte_carlo_of_the_same_model():
    """For one fitted two-part model, four million random pseudo-persons (the test's own adaptive
    quadrature for each consumption-day amount) give the same percentiles (0.5%), mean (0.3%) and
    share below 10 (0.002) as the engine's integration."""
    rec = simulate_two_part(np.random.default_rng(44))
    fit = nci.fit_two_part(rec)
    floor = nci.half_minimum(rec)
    dist = nci.two_part_distribution(fit, floor, CUT2)
    T = two_part_monte_carlo(fit.alpha, fit.beta, fit.sigma_u1, fit.sigma_u2, fit.rho, fit.lam,
                             fit.sigma_e, floor)
    for q, value in dist.percentiles.items():
        assert value == pytest.approx(np.percentile(T, q), rel=0.005), q
    assert dist.mean == pytest.approx(T.mean(), rel=0.003)
    assert dist.below == pytest.approx(np.mean(T < CUT2), abs=0.002)


REPS2 = 200


@pytest.fixture(scope="module")
def two_part_runs(two_part_truth):
    """Bias and the correlation: 200 datasets of 2,000 people, each fitted once."""
    truth, share = two_part_truth
    rng = np.random.default_rng(7)
    out = {"p": {q: [] for q in (10, 50, 90)}, "below": [], "rho": [], "lam": [], "zero": []}
    for _ in range(REPS2):
        rec = simulate_two_part(rng)
        res = nci.usual_intake(rec, "two_part", cutoff=CUT2, percentiles=(10, 50, 90))
        for q in (10, 50, 90):
            out["p"][q].append(res.percentiles[q].value)
        out["below"].append(res.below.value)
        out["rho"].append(res.parameters["rho"])
        out["lam"].append(res.lam)
        out["zero"].append(res.zero_share)
    out["truth"], out["share"] = truth, share
    return out


@pytest.fixture(scope="module")
def two_part_coverage(two_part_truth):
    """Coverage: 200 datasets of 1,000 people, each with a 20-replicate person bootstrap (every
    replicate refits the whole two-part model, λ included). Smaller than the bias study's samples so
    that 4,200 two-part fits take about a minute; a coarser bootstrap only adds noise to each
    interval's width, which lowers coverage if anything."""
    truth, share = two_part_truth
    rng = np.random.default_rng(8)
    out = {"cov": {q: 0 for q in (10, 50, 90)}, "cov_below": 0}
    for r in range(REPS2):
        rec = simulate_two_part(rng, n=1000)
        res = nci.usual_intake(rec, "two_part", cutoff=CUT2, percentiles=(10, 50, 90),
                               replication=nci.person_bootstrap(rec.n_persons, 20, seed=r))
        for q in (10, 50, 90):
            e = res.percentiles[q]
            out["cov"][q] += e.ci_low <= truth[q] <= e.ci_high
        out["cov_below"] += res.below.ci_low <= share <= res.below.ci_high
    return out


def test_2b_the_simulation_recovers_the_episodic_food_distribution(two_part_runs):
    """Bias over 200 datasets: each of the 10th, 50th and 90th percentiles within 3% of the truth and
    within 4 Monte Carlo standard errors (observed: within 0.3%), the share below 10 within 0.01.
    (The truth is from four million simulated people; its own Monte Carlo error is under 0.2%.)"""
    out = two_part_runs
    lines = [f"zero recalls {np.mean(out['zero']):.2f}"]
    for q in (10, 50, 90):
        est = np.array(out["p"][q])
        bias = est.mean() - out["truth"][q]
        mcse = est.std(ddof=1) / math.sqrt(REPS2)
        lines.append(f"p{q}: truth {out['truth'][q]:.3f} bias {bias:+.4f} (MC se {mcse:.4f})")
        assert abs(bias) < 0.03 * out["truth"][q]
        assert abs(bias) < 4 * mcse
    below = np.array(out["below"])
    lines.append(f"share<10: truth {out['share']:.4f} bias {below.mean() - out['share']:+.4f}")
    print("\n" + "\n".join(lines))
    assert abs(below.mean() - out["share"]) < 0.01


def test_2b_the_correlation_of_the_person_effects_is_recovered(two_part_runs):
    """ρ = 0.5 (and λ = 0.25) recovered: the mean estimate within 0.05 of ρ (its Monte Carlo error
    is about 0.01; observed 0.504) and λ within 0.02."""
    rho = np.array(two_part_runs["rho"])
    print(f"\nrho mean {rho.mean():.3f} sd {rho.std(ddof=1):.3f}; "
          f"lambda mean {np.mean(two_part_runs['lam']):.3f}")
    assert abs(rho.mean() - RHO) < 0.05
    assert abs(np.mean(two_part_runs["lam"]) - LAM2) < 0.02


def test_2b_the_bootstrap_intervals_cover(two_part_coverage):
    """Coverage of the 95% intervals over 200 datasets at least 0.90 for each percentile and the
    share (the binomial Monte Carlo error at 0.95 is 0.015)."""
    out = two_part_coverage
    cover = {**{f"p{q}": out["cov"][q] / REPS2 for q in (10, 50, 90)},
             "share": out["cov_below"] / REPS2}
    print("\ncoverage", cover)
    assert all(v >= 0.90 for v in cover.values()), cover


def test_2d_the_two_part_model_needs_zero_days_and_repeat_consumers():
    """No zero day: the probability part has nothing to estimate (refused, amount-only named);
    fewer than two people with two consumption days: MIXTRAN's rule (refused)."""
    rec = simulate_amount(np.random.default_rng(45), n=300)
    with pytest.raises(nci.UsualIntakeRefused, match="amount-only model is the one"):
        nci.usual_intake(rec, "two_part")
    rare = simulate_two_part(np.random.default_rng(46), n=300, p2=0.0)
    with pytest.raises(nci.UsualIntakeRefused, match="two or more recalls"):
        nci.usual_intake(rare, "two_part")


# ── 3 · variance: balanced repeated replication and the bootstrap ────────────


def _design(rng, strata=15, per_psu=(20, 40)):
    rows = []
    for h in range(strata):
        for j in (1, 2):
            for _ in range(int(rng.integers(*per_psu))):
                rows.append((h + 1, j))
    d = pd.DataFrame(rows, columns=["stratum", "psu"])
    d["w"] = rng.uniform(500, 3000, len(d))
    d["y"] = rng.normal(50, 12, len(d)) + 3 * d["stratum"] + 2 * (d["psu"] == 2)
    return d


def test_3a_fay_brr_is_the_two_psu_variance_for_a_total():
    """For a weighted total, BRR with full orthogonal balance reproduces the with-replacement
    two-PSU variance Σ_h (t_h1 − t_h2)² exactly (Wolter 2007, ch. 3), Fay's coefficient included.
    Fifteen strata take a Hadamard matrix of order 16: 16 replicates, each column summing to zero,
    factors 1.7 and 0.3."""
    d = _design(np.random.default_rng(51))
    rep = nci.brr(d["stratum"], d["psu"])
    assert rep.factors.shape == (16, len(d)) and rep.df == 15 and rep.fay == 0.3
    assert set(np.unique(rep.factors)) == {0.3, 1.7}
    first = (d["psu"] == 1).to_numpy()
    for h in range(1, 16):
        col = rep.factors[:, (d["stratum"] == h).to_numpy() & first][:, 0]
        assert np.sum(col == 1.7) == np.sum(col == 0.3) == 8  # balanced
    total = float((d["w"] * d["y"]).sum())
    reps = rep.factors @ (d["w"] * d["y"]).to_numpy()
    t = d.assign(t=d["w"] * d["y"]).groupby(["stratum", "psu"])["t"].sum().unstack()
    textbook = float(((t[1] - t[2]) ** 2).sum())
    assert rep.variance(total, reps) == pytest.approx(textbook, rel=1e-10)


@needs_r
def test_3a_r_survey_agrees_on_the_fay_brr_standard_error(tmp_path):
    """R's survey package: ``as.svrepdesign(svydesign(...), type = "Fay", fay.rho = 0.3)`` and
    ``svytotal`` give the same standard error for the weighted total (and so does the Taylor
    linearization of the original design, the identity above)."""
    d = _design(np.random.default_rng(52))
    rep = nci.brr(d["stratum"], d["psu"])
    total = float((d["w"] * d["y"]).sum())
    se = math.sqrt(rep.variance(total, rep.factors @ (d["w"] * d["y"]).to_numpy()))
    r = run_r(tmp_path, """
suppressMessages(library(survey))
d <- read.csv("data.csv")
des <- svydesign(ids = ~psu, strata = ~stratum, weights = ~w, data = d, nest = TRUE)
fay <- as.svrepdesign(des, type = "Fay", fay.rho = 0.3)
cat(sprintf('{"fay": %.12g, "taylor": %.12g}\\n', SE(svytotal(~y, fay)), SE(svytotal(~y, des))))
""", d)
    assert se == pytest.approx(r["fay"], rel=1e-8)
    assert se == pytest.approx(r["taylor"], rel=1e-8)


def test_3a_the_psu_bootstrap_rescales_within_strata():
    """Rao and Wu's rescaling bootstrap with n_h − 1 PSUs drawn: every replicate's weights sum, per
    stratum, to n_h/(n_h − 1) × (draws) units, and averaged over replicates each factor is 1."""
    rng = np.random.default_rng(53)
    strata = np.repeat(np.arange(6), 30)
    psu = np.tile(np.repeat(np.arange(3), 10), 6)
    rep = nci.psu_bootstrap(strata, psu, n_boot=4000, seed=1)
    assert rep.df == 18 - 6 and rep.n_psu == 18
    assert rep.factors.mean() == pytest.approx(1.0, abs=0.02)
    assert set(np.unique(np.round(rep.factors, 6))) <= {0.0, 1.5, 3.0}


SURVEY_STRATA, PER_PSU, SIGMA_V = 15, 64, 0.35


def simulate_survey(rng: np.random.Generator):
    """A stratified two-stage sample: 15 strata, two PSUs each, a PSU effect on usual intake (sd 0.35
    of the person effect's 1), and people drawn within each PSU with probability rising as their
    usual intake falls (informative: unweighted, the low tail is over-represented). The population's
    person effects are N(0, 1), so the amount-only truth above is the population's."""
    persons, weights, strata, psu = [], [], [], []
    for h in range(SURVEY_STRATA):
        for j in (0, 1):
            v = rng.normal(0, SIGMA_V)
            pool = v + rng.normal(0, math.sqrt(1 - SIGMA_V ** 2), 6 * PER_PSU)
            prob = 0.32 * special.expit(-0.9 * pool)
            take = rng.random(len(pool)) < prob
            persons.append(pool[take])
            weights.append(1 / prob[take])
            strata += [h] * int(take.sum())
            psu += [2 * h + j] * int(take.sum())
    u = np.concatenate(persons)
    rec = simulate_amount(rng, n=len(u), u=u)
    return rec, np.concatenate(weights), np.array(strata), np.array(psu)


REPS3 = 200


@pytest.fixture(scope="module")
def survey_runs():
    truth, share = amount_truth()
    rng = np.random.default_rng(99)
    out = {"w": {q: [] for q in (10, 50, 90)}, "u": {q: [] for q in (10, 50, 90)},
           "cov": {q: 0 for q in (10, 50, 90)}, "se": {q: [] for q in (10, 50, 90)},
           "below": [], "cov_below": 0, "se_below": []}
    for r in range(REPS3):
        rec, w, strata, psu = simulate_survey(rng)
        rep = nci.brr(strata, psu)
        res = nci.usual_intake(rec, "amount_only", weights=w, replication=rep, cutoff=CUT,
                               percentiles=(10, 50, 90))
        flat = nci.usual_intake(rec, "amount_only", cutoff=CUT, percentiles=(10, 50, 90))
        for q in (10, 50, 90):
            e = res.percentiles[q]
            out["w"][q].append(e.value)
            out["u"][q].append(flat.percentiles[q].value)
            out["cov"][q] += e.ci_low <= truth[q] <= e.ci_high
            out["se"][q].append(e.se)
        out["below"].append(res.below.value)
        out["cov_below"] += res.below.ci_low <= share <= res.below.ci_high
        out["se_below"].append(res.below.se)
    out["truth"], out["share"] = truth, share
    return out


def test_3b_under_a_survey_design_the_weights_recover_the_population(survey_runs):
    """Weighted (the fit a pseudo-likelihood, the distribution weighted), each percentile is within
    1.5% of the population's truth and the share within 0.01; unweighted, the sample's
    over-represented low tail pulls each percentile down by more than 4% (the weights matter)."""
    out = survey_runs
    lines = []
    for q in (10, 50, 90):
        w, u = np.mean(out["w"][q]), np.mean(out["u"][q])
        t = out["truth"][q]
        lines.append(f"p{q}: truth {t:.2f} weighted {w - t:+.3f} unweighted {u - t:+.3f}")
        assert abs(w / t - 1) < 0.015
        assert u / t - 1 < -0.04
    print("\n" + "\n".join(lines))
    assert abs(np.mean(out["below"]) - out["share"]) < 0.01


def test_3b_fay_brr_intervals_cover_under_the_design(survey_runs):
    """With PSU-correlated intakes, Fay's BRR (16 replicates, t on 15 df) covers at 0.88 or more
    for each percentile and the share over 200 samples (the binomial Monte Carlo error at 0.95 is
    0.015), and its standard error is within 20% of the estimates' spread across samples."""
    out = survey_runs
    cover = {**{f"p{q}": out["cov"][q] / REPS3 for q in (10, 50, 90)},
             "share": out["cov_below"] / REPS3}
    ratio = {**{f"p{q}": np.mean(out["se"][q]) / np.std(out["w"][q], ddof=1) for q in (10, 50, 90)},
             "share": np.mean(out["se_below"]) / np.std(out["below"], ddof=1)}
    print("\ncoverage", cover, "\nSE / spread", {k: round(v, 3) for k, v in ratio.items()})
    assert all(v >= 0.88 for v in cover.values()), cover
    assert all(0.8 < v < 1.2 for v in ratio.values()), ratio


# ── 4 · routing, through the real server ─────────────────────────────────────

from turbotab.core.survey import ATTESTATION  # noqa: E402
from turbotab.core.tests.acceptance.server_drive import Truth, local_server, open_project  # noqa: E402


def recall_table(seed: int = 61, n: int = 500) -> pd.DataFrame:
    """A long table of 24-hour recalls: 60% of people have two (``recall`` 1 and 2), the rest one;
    ``weekend`` marks a Friday–Sunday recall. Protein is eaten every day (lognormal around each
    person's level); fish is episodic, and a fifth of people never eat it (``fish_ever`` = 0). LDL is
    a person's outcome."""
    rng = np.random.default_rng(seed)
    age = rng.integers(25, 75, n).astype(float)
    sex = rng.choice(["F", "M"], n)
    u = rng.normal(0, 0.45, n)
    ever = (rng.random(n) >= 0.2).astype(int)
    u1 = rng.normal(0, 1.0, n)
    u2 = 0.5 * u1 + rng.normal(0, 0.5, n)
    ldl = 120 + 10 * u + rng.normal(0, 15, n)
    rows = []
    for i in range(n):
        for j in range(2 if rng.random() < 0.6 else 1):
            wk = int(rng.random() < 3 / 7)
            protein = math.exp(4.2 + u[i] + 0.1 * wk - 0.05 * j + rng.normal(0, 0.45))
            eats = ever[i] and rng.random() < special.expit(-0.5 + u1[i] + 0.2 * wk)
            fish = math.exp(3.8 + u2[i] + rng.normal(0, 0.6)) if eats else 0.0
            rows.append({"participant_id": f"P{i:04d}", "recall": j + 1, "weekend": wk,
                         "age": age[i], "sex": sex[i],
                         "energy_kcal": round(2000 + 300 * u[i] + rng.normal(0, 400), 1),
                         "protein_g": round(protein, 2), "fish_g": round(fish, 1),
                         "fish_ever": int(ever[i]), "ldl": round(ldl[i], 1)})
    return pd.DataFrame(rows)


def recall_truth() -> Truth:
    """The table's truth for the readings the server asks (BLUEPRINT §14.3): ages and recall numbers
    are amounts, the weekend flag a code; energy is one day's kcal."""
    return Truth({"code_or_count:age": "amount", "code_or_count:recall": "amount",
                  "code_or_count:weekend": "code", "unit:energy_kcal": "kcal",
                  "day_count:energy_kcal": "1"}, fixture="recall_table")


RECALL_ROLES = {"participant_id": "identifier", "age": "covariate", "sex": "covariate",
                "energy_kcal": "energy", "protein_g": "exposure", "fish_g": "exposure"}


def opening(d, purpose: str = "inference", lenses=("dietary",)) -> None:
    d.decide({"kind": "set_lens", "lenses": list(lenses)})
    d.reach("target")
    d.decide({"kind": "set_target", "column": "ldl"})
    d.answer("task", {"kind": "set_task", "column": "ldl", "task": "regression"})
    d.reach("purpose")
    d.decide({"kind": "set_purpose", "purpose": purpose})
    d.reach("grain")
    d.decide({"kind": "set_grain", "grain": "repeated", "id_column": "participant_id"})
    d.answer("repeat_kind", {"kind": "set_repeat_kind", "repeat_kind": "repeats"})
    d.answer("unit", {"kind": "set_unit", "unit": "unit"})
    d.answer("aggregation", {"kind": "set_aggregation", "method": "mean"})
    d.answer("temporal", {"kind": "set_temporal", "temporal": False})
    d.reach("roles")
    d.decide_roles(RECALL_ROLES)


PROTEIN = {"kind": "set_usual_intake", "nutrient": "protein_g", "model": "amount_only",
           "order_column": "recall", "weekend": ["weekend"], "cutoff": 46, "cutoff_kind": "EAR",
           "n_boot": 100}
FISH = {"kind": "set_usual_intake", "nutrient": "fish_g", "model": "two_part",
        "order_column": "recall", "weekend": ["weekend"], "n_boot": 100}


def nhanes_table(seed: int = 21, n: int = 900) -> pd.DataFrame:
    """An NHANES-shaped wide table: one row per ``SEQN``, day-1 and day-2 total protein (``DR1TPROT``,
    ``DR2TPROT``; day 2 missing for a fifth), NHANES's days of the week (``DR1DAY``: 1 Sunday … 7
    Saturday), 15 strata of two PSUs (``SDMVSTRA``, ``SDMVPSU``) and the day-one dietary weight."""
    rng = np.random.default_rng(seed)
    strata = np.repeat(np.arange(1, 16), n // 15)
    psu = np.tile([1, 2], n // 2)[: len(strata)]
    n = len(strata)
    u = rng.normal(0, 0.45, n) + 0.05 * (strata - 8) / 7
    rows = []
    for i in range(n):
        has2 = rng.random() < 0.8
        rows.append({"SEQN": 83732 + i, "RIAGENDR": int(rng.integers(1, 3)),
                     "RIDAGEYR": int(rng.integers(20, 80)), "SDMVSTRA": int(strata[i]) + 118,
                     "SDMVPSU": int(psu[i]), "WTDRD1": round(float(rng.uniform(5000, 90000)), 2),
                     "DR1TPROT": round(math.exp(4.2 + u[i] + rng.normal(0, 0.5)), 2),
                     "DR2TPROT": round(math.exp(4.1 + u[i] + rng.normal(0, 0.5)), 2) if has2 else None,
                     "DR1DAY": int(rng.integers(1, 8)), "DR2DAY": int(rng.integers(1, 8)) if has2 else None,
                     "LBXTC": round(190 + 15 * u[i] + rng.normal(0, 30), 0)})
    return pd.DataFrame(rows)


NHANES_PROTEIN = {"kind": "set_usual_intake", "nutrient": "DRxTPROT", "model": "amount_only",
                  "days": ["DR1TPROT", "DR2TPROT"], "weekend": ["DR1DAY", "DR2DAY"],
                  "weekend_coding": "nhanes_day", "cutoff": 46, "cutoff_kind": "EAR"}


@pytest.fixture(scope="module")
def journeys(tmp_path_factory):
    """Four projects through one local server. A: the long recall table under inference, every answer
    and refusal the routing has. B: the same table under prediction, and under the clinical lens.
    C: the NHANES-shaped wide table under its survey design, then as these participants."""
    root = tmp_path_factory.mktemp("nci_routing")
    long_path, wide_path = root / "recalls.csv", root / "nhanes.csv"
    recall_table().to_csv(long_path, index=False)
    nhanes_table().to_csv(wide_path, index=False)
    out: dict = {}
    with local_server(root / "home") as client:
        d = open_project(client, long_path, recall_truth())
        opening(d)
        a = {"offer": d.artifact("usual_intake")["offer"], "refused": {}}
        for name, body in (
                ("consumers", {**FISH, "population": "consumers"}),
                ("ai", {**PROTEIN, "cutoff_kind": "AI"}),
                ("energy", {**PROTEIN, "nutrient": "energy_kcal"}),
                ("columns", {**PROTEIN, "nutrient": "protein_mg"})):
            r = d.post(body)
            a["refused"][name] = (r.status_code, r.json().get("error"))
        d.decide(PROTEIN)
        d.decide(FISH)
        art = d.artifact("usual_intake")
        a["protein"], a["fish_whole"] = art["analyses"]
        a["sentences"] = [r["sentence"] for r in d.view()["decisions"][-2:]]
        d.decide({**FISH, "model": "amount_only"})
        a["fish_amount"] = d.artifact("usual_intake")["analyses"][1]
        d.decide({**FISH, "population": "consumers", "consumer_column": "fish_ever"})
        a["fish_consumers"] = d.artifact("usual_intake")["analyses"][1]
        d.decide({"kind": "set_measurement_error", "method": "none"})
        a["calibration_kept_apart"] = d.artifact("usual_intake")["analyses"][0]["methods"]
        r = d.post({"kind": "set_repeat_kind", "repeat_kind": "time_points"})
        a["time_points_status"] = r.status_code
        if r.status_code == 200:
            after = d.artifact("usual_intake")
            a["after_time_points"] = after
        out["A"] = a

        b = {}
        d = open_project(client, long_path, recall_truth())
        opening(d, purpose="prediction")
        b["offer"] = d.artifact("usual_intake")["offer"]
        r = d.post(PROTEIN)
        b["refused"] = (r.status_code, r.json().get("error"))
        d = open_project(client, long_path, recall_truth())
        d.decide({"kind": "set_lens", "lenses": ["clinical"]})
        d.reach("target")
        d.decide({"kind": "set_target", "column": "ldl"})
        d.answer("task", {"kind": "set_task", "column": "ldl", "task": "regression"})
        d.reach("purpose")
        d.decide({"kind": "set_purpose", "purpose": "inference"})
        b["clinical_offer"] = d.artifact("usual_intake")["offer"]
        r = d.post(PROTEIN)
        b["clinical_refused"] = (r.status_code, r.json().get("error"))
        out["B"] = b

        c = {}
        d = open_project(client, wide_path, Truth({"code_or_count:RIDAGEYR": "amount",
                                                   "code_or_count:RIAGENDR": "code",
                                                   "code_or_count:DR1DAY": "code",
                                                   "code_or_count:DR2DAY": "code"},
                                                  fixture="nhanes_table"))
        d.decide({"kind": "set_lens", "lenses": ["dietary"]})
        d.reach("target")
        d.decide({"kind": "set_target", "column": "LBXTC"})
        d.answer("task", {"kind": "set_task", "column": "LBXTC", "task": "regression"})
        d.reach("purpose")
        d.decide({"kind": "set_purpose", "purpose": "inference"})
        d.reach("roles")
        d.decide_roles({"SEQN": "identifier", "RIAGENDR": "covariate", "RIDAGEYR": "covariate",
                        "SDMVSTRA": "design", "SDMVPSU": "design", "WTDRD1": "design",
                        "DR1TPROT": "exposure"})
        d.decide(NHANES_PROTEIN)
        c["unanswered"] = d.artifact("usual_intake")["analyses"][0]
        d.answer("survey", {"kind": "set_survey", "estimand": "population", "weight": "WTDRD1",
                            "strata": "SDMVSTRA", "psu": "SDMVPSU"})
        art = d.artifact("usual_intake")
        c["offer"], c["population"] = art["offer"], art["analyses"][0]
        d.decide({"kind": "set_survey", "estimand": "sample"})
        c["sample"] = d.artifact("usual_intake")["analyses"][0]
        out["C"] = c
    out["long"], out["wide"] = pd.read_csv(long_path), pd.read_csv(wide_path)
    return out


def _long_recalls(frame: pd.DataFrame, column: str) -> nci.Recalls:
    """The recalls read from the CSV with pandas: each person's rows ordered by ``recall``."""
    f = frame.sort_values(["participant_id", "recall"], kind="stable")
    codes, _ = pd.factorize(f["participant_id"], sort=True)
    return nci.Recalls.of(f[column].to_numpy(float), codes, int(codes.max()) + 1,
                          later=(f["recall"] > 1).to_numpy(float), weekend=f["weekend"].to_numpy(float))


def test_4_the_recalls_are_ordered_only_by_a_settled_or_named_column():
    """BLUEPRINT §14.1: the sequence covariate changes a number, so a long table's recalls are
    ordered by the answer's own column, else by a time column the ledger holds settled; a time
    column the repeats reading only proposes orders nothing (no sequence effect is modeled)."""
    from turbotab.core import decisions as d
    from turbotab.core.stages.usual_intake import recall_order

    def folded(*decisions):
        return d.fold([d.DecisionRecord(id=f"r{i}", seq=i, at="2026-10-05T00:00:00Z", decision=x)
                       for i, x in enumerate(decisions, start=1)])

    structure = {"repeats": {"replicate_index": "visit"}}
    plain = d.UsualIntakeSpec(model="amount_only")
    proposed = folded(d.SetTarget(column="y"))
    confirmed = folded(d.SetTarget(column="y"),
                       d.ConfirmReading(reading="time_column", column="visit", value="orders"))
    assert recall_order(proposed, structure, plain) is None
    assert recall_order(confirmed, structure, plain) == "visit"
    named = d.UsualIntakeSpec(model="amount_only", order_column="recall")
    assert recall_order(proposed, structure, named) == "recall"


def test_4_the_offer_names_the_estimand_and_ranks_a_model_for_each_component(journeys):
    """The dietary lens, inference, and two recalls for 60% of people: the usual-intake distribution
    is offered as its own estimand. The offer counts the recalls as pandas does, ranks the amount-only
    model first for protein and energy (no zero day) and the two-part model for fish (its zero share
    as pandas counts it), and says that association is regression calibration's question and that
    nut-14's population must be stated."""
    offer, frame = journeys["A"]["offer"], journeys["long"]
    per = frame.groupby("participant_id").size()
    assert offer["offered"] and offer["format"] == "long" and offer["reason"] is None
    assert offer["recalls"] == {str(k): int(v) for k, v in per.value_counts().sort_index().items()}
    assert offer["n_repeat"] == int((per >= 2).sum())
    cands = {c["column"]: c for c in offer["candidates"]}
    assert cands["protein_g"]["suggested"] == "amount_only"
    assert cands["energy_kcal"]["suggested"] == "amount_only"
    assert cands["fish_g"]["suggested"] == "two_part"
    assert cands["fish_g"]["zero_share"] == pytest.approx(float((frame["fish_g"] == 0).mean()))
    assert "regression calibration" in offer["association"]
    assert "set_measurement_error" in offer["association"]
    assert "nut-14" in offer["population_question"]


def test_4_it_is_not_offered_under_prediction_or_without_the_dietary_lens(journeys):
    """Prediction: not offered, and the answer refused with "change the purpose" and "leave it out"
    as exits. The clinical lens alone: not offered, refused with "add the dietary lens"."""
    from turbotab.core import usual_intake as ui

    b = journeys["B"]
    assert not b["offer"]["offered"] and b["offer"]["reason"] == ui.PREDICTION
    status, error = b["refused"]
    assert status == 409 and error["code"] == "not_for_prediction"
    assert [e["label"] for e in error["exits"]] == ["Change the purpose to inference",
                                                     "Leave usual intake out"]
    assert error["exits"][1]["decision"]["model"] == "none"
    assert not b["clinical_offer"]["offered"] and b["clinical_offer"]["reason"] == ui.NOT_DIETARY
    status, error = b["clinical_refused"]
    assert status == 409 and error["code"] == "not_dietary"


def test_4_the_amount_only_analysis_is_an_independent_mixed_model_of_the_csv(journeys):
    """Protein through the server, against the CSV read by pandas: at the engine's λ, statsmodels'
    ML mixed model of the Box-Cox recalls (repeat-recall and weekend indicators) gives the same
    coefficients and variances; from them the test's own quadrature gives the same percentiles and
    share below the EAR (46); the recall counts are pandas'."""
    a, frame = journeys["A"]["protein"], journeys["long"]
    rec = _long_recalls(frame, "protein_g")
    p = a["parameters"]
    lam = p["lambda"]
    m = _mixedlm(boxcox_ref(rec.amount, lam), rec.later, rec.weekend, rec.person)
    assert [p["beta_intercept"], p["beta_repeat recall"], p["beta_weekend"]] == pytest.approx(
        m.fe_params.to_numpy(), rel=1e-5, abs=1e-6)
    assert p["sigma2_u"] == pytest.approx(float(m.cov_re.iloc[0, 0]), rel=1e-4)
    assert p["sigma2_e"] == pytest.approx(float(m.scale), rel=1e-5)
    su, se = math.sqrt(p["sigma2_u"]), math.sqrt(p["sigma2_e"])

    def T(u: float) -> float:
        return (4 / 7 * amount_given(p["beta_intercept"] + u, lam, se)
                + 3 / 7 * amount_given(p["beta_intercept"] + p["beta_weekend"] + u, lam, se))

    for q, e in a["percentiles"].items():
        assert e["value"] == pytest.approx(T(su * stats.norm.ppf(int(q) / 100)), rel=1e-6), q
    uc = optimize.brentq(lambda u: T(u) - 46, -10 * su, 10 * su, xtol=1e-12)
    assert a["share"]["value"] == pytest.approx(stats.norm.cdf(uc / su), abs=1e-6)
    per = frame.groupby("participant_id").size()
    assert a["recalls"] == {str(k): int(v) for k, v in per.value_counts().sort_index().items()}
    assert a["variance"] == {"method": "bootstrap", "replicates": 100, "ok": 100, "df": None,
                             "fay": None, "n_strata": None, "n_psu": None}
    assert a["share"]["ci_low"] < a["share"]["value"] < a["share"]["ci_high"]


def test_4_the_amount_only_methods_sentence_is_verbatim(journeys):
    """(5) The sentence names the model, the transform (with λ), the nuisance covariates and the
    variance method, word for word."""
    a = journeys["A"]["protein"]
    n, n2 = a["n_persons"], a["n_repeat"]
    assert a["methods"] == (
        f"Usual intake of `protein_g` was estimated by the NCI method (Tooze et al. 2006; Tooze et "
        f"al. 2010) from 24-hour recalls of {n:,} participants (1 to 2 recalls each; {n2:,} with two "
        f"or more): an amount-only model, a linear mixed model with a person-specific random "
        f"intercept fitted by maximum likelihood to Box-Cox-transformed intakes "
        f"(λ = {a['parameters']['lambda']:.2f}, estimated with the model), with an indicator of a "
        f"repeat recall and a weekend (Friday–Sunday) indicator as nuisance covariates; the "
        f"distribution was predicted for a first recall, weighted 4/7 weekday and 3/7 weekend, "
        f"back-transformed by numerical integration over the within-person error, and describes "
        f"the whole population. Standard errors came from 100 bootstrap resamples of participants, "
        f"each refitting the whole model. The share below the EAR (46) is the EAR cut-point "
        f"estimate of the prevalence of inadequacy (Institute of Medicine 2000).")
    assert a["population_statement"] == ("Results describe the whole population (STROBE-nut nut-14: "
                                         "total population, not consumers only).")
    assert a["share_label"].startswith("share below the EAR (46)")


def test_4_the_two_part_analysis_and_its_sentence(journeys):
    """Fish through the server: the two-part model, its fit the engine's own on the CSV's recalls
    (pandas extraction; the model's arithmetic is sections 2's), its sentence verbatim with ρ and λ,
    its population the whole one with Kipnis et al.'s assumption stated."""
    a, frame = journeys["A"]["fish_whole"], journeys["long"]
    rec = _long_recalls(frame, "fish_g")
    fit = nci.fit_two_part(rec)
    assert a["parameters"]["rho"] == pytest.approx(fit.rho, abs=1e-4)
    assert a["parameters"]["lambda"] == pytest.approx(fit.lam, abs=1e-4)
    assert a["zero_share"] == pytest.approx(float((frame["fish_g"] == 0).mean()))
    n, n2 = a["n_persons"], a["n_repeat"]
    assert a["methods"] == (
        f"Usual intake of `fish_g` was estimated by the NCI method (Tooze et al. 2006; Tooze et al. "
        f"2010) from 24-hour recalls of {n:,} participants (1 to 2 recalls each; {n2:,} with two or "
        f"more): a two-part model for an episodically consumed food (Kipnis et al. 2009), the "
        f"probability of consumption on a recall day by a logistic mixed model and the "
        f"consumption-day amount by a linear mixed model on Box-Cox-transformed amounts "
        f"(λ = {a['parameters']['lambda']:.2f}, estimated with the model), with correlated "
        f"person-specific random effects (ρ = {a['parameters']['rho']:.2f}) and an indicator of a "
        f"repeat recall and a weekend (Friday–Sunday) indicator as nuisance covariates in both parts, "
        f"fitted jointly by maximum likelihood; the distribution was predicted for a first recall, "
        f"weighted 4/7 weekday and 3/7 weekend, integrated numerically over the within-person error "
        f"and the person effects, usual intake being the probability times the consumption-day "
        f"amount, and describes the whole population, everyone taken to consume it on some days "
        f"(Kipnis et al. 2009). Standard errors came from {a['variance']['ok']:,} bootstrap resamples "
        f"of participants, each refitting the whole model.")
    assert any("ultimately consumed by all" in s for s in a["assumptions"])
    assert a["variance"]["ok"] >= 95


def test_4_the_decisions_record_their_own_sentences(journeys):
    """Each answer's sentence in the record (its replication follows the survey answer, so the
    analysis' methods sentence states that)."""
    assert journeys["A"]["sentences"] == [
        "The usual-intake distribution of `protein_g` was estimated by the NCI method with an "
        "amount-only model, for the whole population, with the share below the EAR (46).",
        "The usual-intake distribution of `fish_g` was estimated by the NCI method with a two-part "
        "model for an episodically consumed food, for the whole population."]


def test_4_an_episodic_food_under_the_amount_only_model_is_recorded_with_its_concern(journeys):
    """Block and record (MODELING_SEQUENCE §2): the answer is recorded and run, and the analysis says
    that its zero days, as many as pandas counts, became half the smallest amount."""
    a, frame = journeys["A"]["fish_amount"], journeys["long"]
    zeros = int((frame["fish_g"] == 0).sum())
    assert a["applies"] and a["model"] == "amount_only" and a["zeros_replaced"] == zeros
    assert any("for which the two-part model ranks first" in c and f"those {zeros:,} zero days" in c
               for c in a["concerns"])
    assert f"{zeros:,} zero-intake recalls were set to half the smallest reported amount" in a["methods"]


def test_4_consumers_only_needs_a_column_and_is_then_stated(journeys):
    """nut-14: consumers only is refused without a column that says who consumes the food (exits:
    the whole population, or name the column); with ``fish_ever`` the analysis describes the people
    it marks, as many as pandas counts, and says so."""
    status, error = journeys["A"]["refused"]["consumers"]
    assert status == 409 and error["code"] == "consumers_unnamed"
    assert error["exits"][0]["decision"]["population"] == "whole"
    a, frame = journeys["A"]["fish_consumers"], journeys["long"]
    consumers = frame.groupby("participant_id")["fish_ever"].first()
    assert a["applies"] and a["n_persons"] == int((consumers == 1).sum())
    assert a["population_statement"] == ("Results describe consumers only, the participants "
                                         "`fish_ever` marks (STROBE-nut nut-14).")
    assert "describes consumers only (the participants `fish_ever` marks)" in a["methods"]


def test_4_a_prevalence_needs_an_ear_and_energy_has_none(journeys):
    """An AI yields no prevalence of inadequacy; total energy has no EAR. Both refused, each with the
    exits "no cut-off" and "a plain share"; a column the table lacks is refused by name."""
    refused = journeys["A"]["refused"]
    for name, code in (("ai", "ai_no_prevalence"), ("energy", "energy_no_ear")):
        status, error = refused[name]
        assert status == 409 and error["code"] == code, name
        assert [e["decision"]["cutoff_kind"] for e in error["exits"]] == [None, "other"]
    status, error = refused["columns"]
    assert status == 409 and error["code"] == "unknown_column" and "`protein_mg`" in error["message"]


def test_4_a_changed_repeats_answer_is_not_silently_kept(journeys):
    """Recorded after the analyses, "the rows are time points" makes the recalls visits: each
    recorded usual-intake answer is no longer applied, and says why."""
    from turbotab.core import usual_intake as ui

    a = journeys["A"]
    assert a["time_points_status"] == 200
    after = a["after_time_points"]
    assert not after["offer"]["offered"] and after["offer"]["reason"] == ui.TIME_POINTS
    assert [x["refused"] for x in after["analyses"]] == [ui.TIME_POINTS, ui.TIME_POINTS]
    assert all(not x["applies"] for x in after["analyses"])


def test_4_under_the_survey_design_the_estimate_is_weighted_and_fay_brr_gives_its_errors(journeys):
    """NHANES-shaped: the day columns proposed from their names; until the survey question is
    answered nothing is estimated; answered "the surveyed population", the fit is weighted by
    ``WTDRD1`` and the standard errors are Fay's BRR (16 replicates over 15 strata, t on 15 df),
    the weekend read from NHANES's day codes (1, 6, 7). The point estimates are the engine's own on
    the CSV read by pandas with those weights (sections 1 and 3 verify the arithmetic)."""
    from turbotab.core.methods.survey import UNANSWERED

    c, wide = journeys["C"], journeys["wide"]
    assert c["offer"]["format"] == "wide"
    assert c["offer"]["candidates"][0]["days"] == ["DR1TPROT", "DR2TPROT"]
    assert c["unanswered"]["refused"] == UNANSWERED
    a = c["population"]
    assert a["variance"] == {"method": "brr", "replicates": 16, "ok": 16, "df": 15, "fay": 0.3,
                             "n_strata": 15, "n_psu": 30}
    assert a["weight"] == "WTDRD1"
    amount = np.concatenate([wide["DR1TPROT"].to_numpy(float), wide["DR2TPROT"].to_numpy(float)])
    person = np.tile(np.arange(len(wide)), 2)
    later = np.repeat([0.0, 1.0], len(wide))
    day = np.concatenate([wide["DR1DAY"].to_numpy(float), wide["DR2DAY"].to_numpy(float)])
    weekend = np.where(np.isfinite(day), np.isin(day, [1, 6, 7]).astype(float), np.nan)
    rec = nci.Recalls.of(amount, person, len(wide), later=later, weekend=weekend)
    res = nci.usual_intake(rec, "amount_only", weights=wide["WTDRD1"].to_numpy(float), cutoff=46)
    flat = nci.usual_intake(rec, "amount_only", cutoff=46)
    for q, e in a["percentiles"].items():
        assert e["value"] == pytest.approx(res.percentiles[int(q)].value, rel=1e-6), q
    assert a["percentiles"]["50"]["value"] != pytest.approx(flat.percentiles[50].value, rel=1e-3)
    n, n2 = a["n_persons"], a["n_repeat"]
    assert a["methods"] == (
        f"Usual intake of `DRxTPROT` was estimated by the NCI method (Tooze et al. 2006; Tooze et "
        f"al. 2010) from 24-hour recalls of {n:,} participants (1 to 2 recalls each; {n2:,} with two "
        f"or more): an amount-only model, a linear mixed model with a person-specific random "
        f"intercept fitted by maximum likelihood to Box-Cox-transformed intakes "
        f"(λ = {a['parameters']['lambda']:.2f}, estimated with the model), each participant "
        f"weighted by `WTDRD1`, with an indicator of a repeat recall and a weekend (Friday–Sunday) "
        f"indicator as nuisance covariates; the distribution was predicted for a first recall, "
        f"weighted 4/7 weekday and 3/7 weekend, back-transformed by numerical integration over the "
        f"within-person error, and describes the whole population. Standard errors came from Fay's "
        f"balanced repeated replication (16 of 16 replicates over 15 strata of two PSUs, Fay "
        f"coefficient 0.3), each refitting the whole model, with t intervals on 15 degrees of "
        f"freedom. The share below the EAR (46) is the EAR cut-point estimate of the prevalence of "
        f"inadequacy (Institute of Medicine 2000).")


def test_4_these_participants_are_unweighted_with_the_attestation(journeys):
    """Answered "these participants": unweighted, a bootstrap over people, and the survey answer's
    attestation word for word."""
    s = journeys["C"]["sample"]
    assert s["weight"] is None and s["variance"]["method"] == "bootstrap"
    assert s["variance"]["replicates"] == 200
    assert s["methods"].endswith(f" Survey design: {ATTESTATION}.")
    assert "Standard errors came from 200 bootstrap resamples of participants, each refitting the " \
           "whole model." in s["methods"]


# ── 5 · the method contract and its chain ────────────────────────────────────


def test_5_the_contract_declares_every_part():
    """BLUEPRINT §13: slot, data scope, needs, routing (question, place in the sequence, options with
    both labels per purpose, the leash per purpose), storyboard, sentence, relations, each conflict
    with its exit."""
    from turbotab.core.contracts import contract

    c = contract("nci_usual_intake")
    assert c.slot == "model" and c.scope == "training_fold" and c.needs
    assert c.decision_kind == "set_usual_intake" and c.stage == "usual_intake"
    assert c.leash == {"inference": "offer", "prediction": "not_offered"}
    assert [o.key for o in c.options] == ["amount_only", "two_part", "mean_of_days"]
    assert all(o.customary and set(o.sound) == {"inference", "prediction"} for o in c.options)
    assert len(c.storyboard) == 5 and "step 2" in c.sequence_step
    assert all(r.exit for r in c.relations if r.kind == "conflicts")
    assert {r.kind for r in c.relations} >= {"implies", "enables", "conflicts", "invalidates"}


def test_5_every_relation_the_contract_declares_fires_in_the_chain(journeys):
    """The chain test (BLUEPRINT §13): each declared relation's consequence, observed in the
    journeys above. A relation added to the contract without a check here fails."""
    from turbotab.core import usual_intake as ui
    from turbotab.core.contracts import contract

    A, B, C = journeys["A"], journeys["B"], journeys["C"]
    checks = {
        "repeats_offer": lambda: A["offer"]["offered"] and C["offer"]["offered"],
        "association_is_calibration": lambda: (
            "regression calibration" in A["offer"]["association"]
            and all(not isinstance(v, list) or len(v) < 50 for v in A["protein"].values())
            and A["calibration_kept_apart"] == A["protein"]["methods"]),
        "zeros_two_part": lambda: (
            {c["column"]: c["suggested"] for c in A["offer"]["candidates"]}["fish_g"] == "two_part"
            and any("two-part model ranks first" in c for c in A["fish_amount"]["concerns"])),
        "no_zero_no_two_part": lambda: _two_part_refused_without_zeros(),
        "population_design": lambda: (C["population"]["variance"]["method"] == "brr"
                                      and "Fay's balanced repeated replication" in C["population"]["methods"]
                                      and "weighted by `WTDRD1`" in C["population"]["methods"]),
        "sample_attestation": lambda: C["sample"]["methods"].endswith(f"{ATTESTATION}."),
        "consumers_need_a_column": lambda: (A["refused"]["consumers"][1]["code"] == "consumers_unnamed"
                                            and "consumers only" in A["fish_consumers"]["methods"]),
        "prediction_not_offered": lambda: (B["offer"]["reason"] == ui.PREDICTION
                                           and B["refused"][1]["code"] == "not_for_prediction"),
        "time_points_not_recalls": lambda: A["after_time_points"]["offer"]["reason"] == ui.TIME_POINTS,
        "ai_no_prevalence": lambda: A["refused"]["ai"][1]["code"] == "ai_no_prevalence",
        "energy_no_ear": lambda: A["refused"]["energy"][1]["code"] == "energy_no_ear",
        "structure_invalidates": lambda: all(x["refused"] == ui.TIME_POINTS
                                             for x in A["after_time_points"]["analyses"]),
    }
    declared = {r.id for r in contract("nci_usual_intake").relations}
    assert set(checks) == declared, set(checks) ^ declared
    failed = [rid for rid, check in checks.items() if not check()]
    assert failed == [], failed


def _two_part_refused_without_zeros() -> bool:
    rec = simulate_amount(np.random.default_rng(47), n=200)
    try:
        nci.usual_intake(rec, "two_part")
    except nci.UsualIntakeRefused as refused:
        return "amount-only model" in str(refused)
    return False
