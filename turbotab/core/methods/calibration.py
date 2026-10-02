"""Univariate regression calibration of an energy-adjusted intake from repeated recalls (WP12, IN-22).

The mean of a person's recalls measures their usual intake with day-to-day error, and a coefficient
on it is biased. With one error-prone exposure in the model the bias is toward zero; with several
"estimated relative risks may become attenuated, inflated, or can even change direction"
(Freedman et al. 2011, *J Natl Cancer Inst* 103:1086). Freedman et al. recommend univariate
regression calibration for energy-adjusted intakes: "our recommendation refers to energy-adjusted
intake variables used in the density or residual models. Univariate adjustment for the unadjusted
intakes used in the standard and partition models is inappropriate because the attenuation factor
for the nutrient would be too small; the multivariate adjustment is recommended in this case." And:
"the attenuation and contamination factors should be estimated from the validation study after
adjustment for the (exactly measured) confounders included in the disease model."

**The calibration** (replicate data; Carroll, Ruppert, Stefanski & Crainiceanu 2006, *Measurement
Error in Nonlinear Models*, 2nd ed., §4.4; Rosner, Spiegelman & Willett 1990, *Am J Epidemiol*
132:734). Person i has k_i recalls W_ij = X_i + U_ij of the energy-adjusted intake, with classical
error U (mean zero, variance σ²_u, independent of X, of the other covariates Z and across days).
W̄_i is their mean. Then:

* σ²_u is the pooled within-person variance, Σ_i Σ_j (W_ij − W̄_i)² / Σ_i (k_i − 1), over people with
  two or more recalls (the one-way analysis-of-variance estimator);
* W̄ is regressed on Z (the other columns of the model's matrix, with an intercept) by least squares;
  its residual variance s² = RSS / (n − p) estimates σ²_{x|z} + σ²_u · mean(1/k), so
  σ²_{x|z} = s² − σ²_u · mean(1/k_i);
* each person's attenuation factor is λ_i = σ²_{x|z} / (σ²_{x|z} + σ²_u / k_i), and the calibrated
  exposure is E[X | W̄, Z] = fitted_i + λ_i · residual_i;
* the outcome model is refit with the calibrated exposure in place of W̄.

With every person on the same number of recalls and a least-squares outcome model, the refit
coefficient is exactly the uncorrected one divided by λ, Freedman et al.'s univariate method
("division of the unadjusted relative risk estimate by the 24-hour recall–based attenuation factor
for that variable"); for a logistic model the substitution is the usual approximation (Carroll et
al. §4.2). With balanced recalls σ²_u and σ²_{x|z} are the REML variance components of a
random-intercept model of the recalls on Z, which the acceptance tests check against statsmodels.

**Intervals** come from refitting on people drawn with replacement (each with all their recalls):
σ²_u, the calibration and the outcome model are re-estimated in every refit; the interval is the
estimate ± 1.96 bootstrap standard errors. Freedman et al.: "For a single mismeasured exposure in the
disease model, the usual statistical test of the null hypothesis (no exposure effect) remains
theoretically valid even though the estimated relative risk is attenuated", so the test of no
association is the uncorrected one's; the correction changes the estimate's size and widens its
interval.

**What it assumes**, stated with every result: errors independent across a person's recalls and of
their true intake. Recalls share person-specific reporting bias, which this reads as true intake;
Freedman et al.'s Table 2 shows recall-based attenuation factors above and below the biomarker-based
ones, so the corrected estimate can still be off in either direction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

Z_95 = 1.959963984540054  # the standard normal's 97.5th percentile


class CalibrationRefused(ValueError):
    """The data cannot support the calibration; the message says why."""


@dataclass(frozen=True)
class Replicates:
    """Each person's recalls: ``values[m]`` belongs to person ``person[m]`` (0 … n − 1)."""

    values: np.ndarray
    person: np.ndarray
    n_persons: int

    @classmethod
    def of(cls, values: Any, person: Any, n_persons: int) -> "Replicates":
        v = np.asarray(values, dtype=float)
        p = np.asarray(person, dtype=np.int64)
        keep = np.isfinite(v)
        return cls(v[keep], p[keep], int(n_persons))

    def counts(self) -> np.ndarray:
        return np.bincount(self.person, minlength=self.n_persons).astype(float)

    def means(self) -> np.ndarray:
        k = self.counts()
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.bincount(self.person, weights=self.values, minlength=self.n_persons) / k


@dataclass(frozen=True)
class WithinPerson:
    sigma2_u: float  # pooled within-person (day-to-day) variance
    df: int  # Σ (k_i − 1)
    n_repeat: int  # people with two or more recalls


def within_person_variance(rep: Replicates) -> WithinPerson:
    """The pooled within-person variance of the recalls (one-way ANOVA)."""
    k = rep.counts()
    means = rep.means()
    df = int(np.sum(np.maximum(k - 1, 0)))
    n_repeat = int(np.sum(k >= 2))
    if df == 0:
        raise CalibrationRefused("No one has two or more recalls with a value, so the day-to-day "
                                 "variance cannot be estimated.")
    resid = rep.values - means[rep.person]
    return WithinPerson(float(np.sum(resid ** 2) / df), df, n_repeat)


@dataclass(frozen=True)
class Calibration:
    calibrated: np.ndarray  # E[X | W̄, Z] per person
    attenuation: np.ndarray  # λ_i per person
    sigma2_u: float
    sigma2_xz: float  # var(X | Z)
    residual_variance: float  # s² of W̄ on Z
    within: WithinPerson


def calibrate(rep: Replicates, Z: np.ndarray | None) -> Calibration:
    """Calibrated exposure for every person with at least one recall (others NaN)."""
    within = within_person_variance(rep)
    k = rep.counts()
    w_bar = rep.means()
    have = k >= 1
    n = int(have.sum())
    Zh = np.empty((n, 0)) if Z is None else np.asarray(Z, dtype=float)[have]
    design = np.column_stack([np.ones(n), Zh])
    coef, *_ = np.linalg.lstsq(design, w_bar[have], rcond=None)
    fitted = design @ coef
    resid = w_bar[have] - fitted
    rank = int(np.linalg.matrix_rank(design))
    if n - rank <= 0:
        raise CalibrationRefused("There are no more people than columns in the calibration "
                                 "regression.")
    s2 = float(resid @ resid / (n - rank))
    sigma2_xz = s2 - within.sigma2_u * float(np.mean(1.0 / k[have]))
    if not sigma2_xz > 0:
        raise CalibrationRefused(
            "The recalls vary as much within people as the people's means vary between them, "
            "once the other columns are allowed for: the true intakes' variance estimate is not "
            "positive, so no attenuation factor exists.")
    lam = np.full(rep.n_persons, np.nan)
    lam[have] = sigma2_xz / (sigma2_xz + within.sigma2_u / k[have])
    calibrated = np.full(rep.n_persons, np.nan)
    calibrated[have] = fitted + lam[have] * resid
    return Calibration(calibrated, lam, within.sigma2_u, float(sigma2_xz), s2, within)


Fit = Callable[[np.ndarray, np.ndarray], np.ndarray]


def ols_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least-squares coefficients of ``y`` on ``X`` with an intercept (the intercept dropped)."""
    design = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return coef[1:]


def logistic_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Unpenalized logistic regression coefficients (Newton–Raphson), the intercept dropped."""
    design = np.column_stack([np.ones(len(X)), X])
    beta = np.zeros(design.shape[1])
    for _ in range(100):
        eta = np.clip(design @ beta, -35, 35)
        mu = 1.0 / (1.0 + np.exp(-eta))
        grad = design.T @ (y - mu)
        hess = design.T @ (design * (mu * (1 - mu))[:, None])
        step = np.linalg.lstsq(hess, grad, rcond=None)[0]
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    return beta[1:]


@dataclass
class Result:
    """One exposure's calibration: the uncorrected and the calibrated coefficient, with the parts."""

    naive: float
    estimate: float
    se: float | None
    ci_low: float | None
    ci_high: float | None
    attenuation: float  # λ for the most common number of recalls
    attenuation_mean: float
    attenuation_se: float | None
    sigma2_u: float
    sigma2_xz: float
    n_persons: int
    n_repeat: int
    recalls: dict[int, int]  # number of recalls -> people
    n_boot: int
    n_boot_ok: int
    boot: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))


def _bootstrap_indices(rep: Replicates, draw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The recalls of the people ``draw`` names (with repeats), as a new ``Replicates``'s arrays."""
    order = np.argsort(rep.person, kind="stable")
    k = rep.counts().astype(np.int64)
    starts = np.concatenate([[0], np.cumsum(k)[:-1]])
    kb = k[draw]
    total = int(kb.sum())
    offsets = np.arange(total) - np.repeat(np.cumsum(kb) - kb, kb)
    rows = order[np.repeat(starts[draw], kb) + offsets]
    return rep.values[rows], np.repeat(np.arange(len(draw)), kb)


def regression_calibration(rep: Replicates, X: np.ndarray, j: int, y: np.ndarray, *, fit: Fit = ols_fit,
                           n_boot: int = 200, seed: int = 0) -> Result:
    """Calibrate column ``j`` of the person-level matrix ``X`` from its recalls ``rep``.

    Column ``j`` of ``X`` is replaced by each person's mean recall for the uncorrected fit and by the
    calibrated exposure for the corrected one; every other column of ``X`` is a covariate of the
    calibration (Freedman et al. 2011). ``fit(X, y)`` returns the outcome model's coefficients
    (intercept dropped) in ``X``'s column order. People with no recall are left out.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if rep.n_persons != len(X) or len(y) != len(X):
        raise ValueError("the recalls, the matrix and the outcome must describe the same people")
    k = rep.counts()
    have = k >= 1
    if not have.all():
        keep = np.flatnonzero(have)
        remap = np.full(rep.n_persons, -1)
        remap[keep] = np.arange(len(keep))
        rep = Replicates(rep.values, remap[rep.person], len(keep))
        X, y, k = X[keep], y[keep], k[keep]
    others = np.delete(X, j, axis=1)

    def one(r: Replicates, Xo: np.ndarray, yy: np.ndarray) -> tuple[float, float, Calibration]:
        cal = calibrate(r, Xo)
        Xw = np.insert(Xo, j, r.means(), axis=1)
        Xc = np.insert(Xo, j, cal.calibrated, axis=1)
        return float(fit(Xw, yy)[j]), float(fit(Xc, yy)[j]), cal

    naive, estimate, cal = one(rep, others, y)
    rng = np.random.default_rng(seed)
    boots, lams = [], []
    modal = int(np.bincount(k.astype(np.int64)).argmax())
    for _ in range(int(n_boot)):
        draw = rng.integers(0, rep.n_persons, rep.n_persons)
        values, person = _bootstrap_indices(rep, draw)
        try:
            _, b, c = one(Replicates(values, person, rep.n_persons), others[draw], y[draw])
        except (CalibrationRefused, np.linalg.LinAlgError):
            continue
        if np.isfinite(b):
            boots.append(b)
            lams.append(c.sigma2_xz / (c.sigma2_xz + c.sigma2_u / modal))
    boot = np.asarray(boots)
    se = float(np.std(boot, ddof=1)) if len(boot) >= 2 else None
    lam_modal = cal.sigma2_xz / (cal.sigma2_xz + cal.sigma2_u / modal)
    counts = np.bincount(k.astype(np.int64))
    return Result(
        naive=naive, estimate=estimate, se=se,
        ci_low=None if se is None else estimate - Z_95 * se,
        ci_high=None if se is None else estimate + Z_95 * se,
        attenuation=float(lam_modal), attenuation_mean=float(np.nanmean(cal.attenuation)),
        attenuation_se=float(np.std(lams, ddof=1)) if len(lams) >= 2 else None,
        sigma2_u=cal.sigma2_u, sigma2_xz=cal.sigma2_xz, n_persons=int(rep.n_persons),
        n_repeat=cal.within.n_repeat,
        recalls={int(i): int(c) for i, c in enumerate(counts) if c and i},
        n_boot=int(n_boot), n_boot_ok=int(len(boot)), boot=boot)


__all__ = [
    "Calibration", "CalibrationRefused", "Replicates", "Result", "WithinPerson", "Z_95", "calibrate",
    "logistic_fit", "ols_fit", "regression_calibration", "within_person_variance",
]
