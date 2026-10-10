"""Regression calibration of intakes measured by repeated 24-hour recalls (MS5; WP12, IN-22).

The mean of a person's recalls measures their usual intake with day-to-day error, and a coefficient
on it is biased. With one error-prone exposure in the model the bias is toward zero; with several
"estimated relative risks may become attenuated, inflated, or can even change direction"
(Freedman et al. 2011, *J Natl Cancer Inst* 103:1086), and "if several risk factors for disease are
considered in the same multiple logistic regression model, and some of these risk factors are
measured with error, the point and interval estimates of relative risk corresponding to any of
these factors may be biased either toward or away from the null value" (Rosner, Spiegelman &
Willett 1990, *Am J Epidemiol* 132:734). So every error-prone intake of the outcome model is
calibrated **jointly** (multivariate regression calibration), never one nutrient at a time
(MODELING_SEQUENCE §0 ruling 7, §2); with one error-prone intake this is the univariate method of
Rosner, Willett & Spiegelman 1989 (*Stat Med* 8:1051), the corrected coefficient the uncorrected
one divided by the attenuation factor.

**The model** (replicate data). Person i has k_i recall days W_ij = X_i + U_ij of p intakes (a
vector), with classical error U (mean zero, covariance Σ_uu, independent of X, of the covariates Z
and across days); W̄_i is their mean. Z is **every other column of the outcome model** and never the
outcome: "the attenuation and contamination factors should be estimated from the validation study
after adjustment for the (exactly measured) confounders included in the disease model" (Freedman
et al. 2011), "the calibration equation should include all confounders included in the outcome
model" (Boe et al. 2023, *Am J Epidemiol* 192:1406).

**The estimator** is the best linear approximation of Carroll, Ruppert, Stefanski & Crainiceanu
(2006, *Measurement Error in Nonlinear Models*, 2nd ed., §4.4, replication data), people with no
recall left out:

* Σ̂_uu = Σ_i Σ_j (W_ij − W̄_i)(W_ij − W̄_i)ᵀ / Σ_i (k_i − 1), the pooled within-person covariance;
* μ̂_w = Σ k_i W̄_i / Σ k_i;  Z̄ = mean of Z;  Σ̂_zz = Σ (Z_i − Z̄)(Z_i − Z̄)ᵀ / (n − 1);
* ν = Σ k_i − Σ k_i² / Σ k_i;  Σ̂_xz = Σ k_i (W̄_i − μ̂_w)(Z_i − Z̄)ᵀ / ν;
* Σ̂_xx = [Σ k_i (W̄_i − μ̂_w)(W̄_i − μ̂_w)ᵀ − (n − 1) Σ̂_uu] / ν;
* E[X_i | W̄_i, Z_i] = μ̂_w + [Σ̂_xx  Σ̂_xz] V_i⁻¹ (W̄_i − μ̂_w; Z_i − Z̄), V_i = [[Σ̂_xx + Σ̂_uu/k_i,
  Σ̂_xz], [Σ̂_zx, Σ̂_zz]]; its slope on W̄ is Γ_k = Σ̂_x|z (Σ̂_x|z + Σ̂_uu/k)⁻¹, Σ̂_x|z = Σ̂_xx −
  Σ̂_xz Σ̂_zz⁻¹ Σ̂_zx (the attenuation matrix; with one intake, λ).

With every person on k recalls, ν = k(n − 1) and V is the (n − 1) covariance matrix of (W̄, Z): the
calibration is R ``mecor``'s ``MeasErrorRandom(W̄, variance = σ̂²_u/k)`` exactly, and for a
least-squares outcome model the refit coefficient vector is the closed form β = (Γᵀ)⁻¹ β̃ of Rosner
et al. (β̃ the uncorrected one). With unequal k the refit substitutes each person's own E[X | W̄, Z]
(Carroll §4.4); for a logistic model the substitution is the usual approximation (§4.2).

**Survey weights** (the surveyed-population answer). Every sum above is weighted by w̃_i, the
survey weights scaled to sum to n (so unit weights give the formulas as written), and the outcome
model is fit by weighted least squares or the weighted logistic likelihood (pseudo-maximum
likelihood, as R's ``survey::svyglm``): design-consistent estimates of the population's calibration
and coefficients.

**The delta-method covariance** (Rosner, Spiegelman & Willett 1990's, in this replicate design):
β = A⁻¹ b with A = I − S⁻¹ Σ̂_uu / k, b the uncorrected coefficients and S = Σ̂_x|z + Σ̂_uu/k (the
(n − 1) covariance of W̄ given Z). Its parts are independent under normality (the within-person
and between-person sums of squares; the outcome model's coefficients given the recalls), with
Cov(Σ̂_uu,ab, Σ̂_uu,cd) = (σ_ac σ_bd + σ_ad σ_bc)/Σ(k_i − 1), Cov(S_ab, S_cd) = (S_ac S_bd + S_ad
S_bc)/(n − r) (r the columns of [1, Z]), and Cov(b) the outcome model's own. Defined for k recalls
each, unweighted. It is a check beside the interval, never the interval: under multiple imputation,
a survey design or clusters it knows nothing of them.

**Intervals** come from a bootstrap over the **whole chain**: each replicate redraws the people
(whole PSUs within strata under a survey design, Rao & Wu 1988's rescaling bootstrap with n_h − 1
PSUs, a stratum with a single PSU by :data:`LONELY_RULE`; whole clusters under repeated units, never
fewer than the cluster floor; people otherwise, each with all their recalls) and
repeats every step that learns from the data: the imputations, the energy model, the within-person
covariance, the calibration and the outcome model. Under multiple imputation this is Schomaker &
Heumann's (2018, *Stat Med* 37:2252) "Boot MI": "B bootstrap samples D*_b (including missing data)
are drawn, and each of them is imputed M times … applying (3.1) to the estimates of each bootstrap
sample yields B point estimates … The set of ordered estimates … can then be used to construct the
1 − 2α% confidence interval", one of the three approaches they show valid ("Both Boot MI and MI Boot
are probably the best options … the former may be preferred for small M or large imputation
uncertainty"; "Boot MI may perform well even for M < 5"). The interval is that percentile interval
(their eq. 3.5), each replicate imputed :data:`BOOT_COPIES` times; the point estimate is the mean
over the analysis's own m imputations. "One cannot use the usual model standard errors from the
outcome regression model when performing RC, as these will be too small" (Keogh, Shaw & Gustafson
2020, STRATOS, *Stat Med* 39:2197, §6.1.2), so model-based intervals, and Rubin's rules over them,
are refused for a calibrated coefficient (:func:`interval_refusal`).

**What it assumes**, stated with every result (:data:`LABEL`): recalls unbiased for usual intake.
Recalls share a person's own reporting bias, which this reads as true intake; "Using the 24HR as a
reference instrument can seriously underestimate true attenuation (up to 60% for energy-adjusted
protein)" (Kipnis et al. 2003, *Am J Epidemiol* 158:14). It corrects within-person random error
only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

LEVEL = 0.95
# Each bootstrap replicate's own imputations under multiple imputation (Schomaker & Heumann 2018:
# "Boot MI may perform well even for M < 5"; for M = 1 its coverage was "too large").
BOOT_COPIES = 5
LABEL = ("corrects only within-person random error, assuming recalls are unbiased for usual intake "
         "(customary; sound under that assumption)")
ROSNER_1989 = "Rosner, Willett & Spiegelman 1989, Stat Med 8:1051"
ROSNER_1990 = "Rosner, Spiegelman & Willett 1990, Am J Epidemiol 132:734"
CARROLL = "Carroll, Ruppert, Stefanski & Crainiceanu 2006, Measurement Error in Nonlinear Models, §4.4"
FREEDMAN = "Freedman et al. 2011, J Natl Cancer Inst 103:1086"
SCHOMAKER = "Schomaker & Heumann 2018, Stat Med 37:2252"
RAO_WU = "Rao & Wu 1988, J Am Stat Assoc 83:231"
KEOGH = "Keogh, Shaw & Gustafson 2020, Stat Med 39:2197"
# A stratum with a single PSU (a lonely PSU) in the bootstrap by PSU within strata. The primary's
# design-based table follows R survey's ``lonely.psu = "adjust"`` (``models.survey.LONELY_RULE``):
# the lonely PSU's total is centered at the grand mean, which for an estimating equation (scores
# summing to zero) is zero, and its n_h/(n_h − 1) is taken as 1, so it adds its total's square to
# the variance. Rao & Wu's draw has nothing to draw from in such a stratum (R's ``subbootweights``
# gives it no finite weight), so the bootstrap gives the lonely PSU the weight multiplier that adds
# the same: it is drawn twice or not at all, each with chance 1/2 (mean 1, variance 1; the
# generalized bootstrap's matching of a variance estimator's quadratic form, Beaumont & Patak 2012,
# Int Stat Rev 80:127). Kept whole in every replicate instead, it would add no variance at all.
LONELY_RULE = ("drawn twice or not at all, each with chance 1/2, so its PSU total varies about "
               "zero as R survey's lonely.psu \"adjust\" centers it")
# The way past any refusal of the data's: no correction, which keeps the primary as it is.
NO_CALIBRATION = {"label": "Record no calibration",
                  "decision": {"kind": "set_measurement_error", "method": "none"}}


class CalibrationRefused(ValueError):
    """The data cannot support the calibration; the message says why, ``exits`` the ways forward
    (no correction unless another is given)."""

    def __init__(self, message: str, exits: Sequence[Mapping[str, Any]] = (NO_CALIBRATION,)):
        super().__init__(message)
        self.exits = [dict(e) for e in exits]


# ── the recalls ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Recalls:
    """Each person's recall days of p intakes: row m of ``values`` (N × p) is a day of person
    ``person[m]`` (0 … n − 1). A day counts only when every one of the p intakes is recorded."""

    values: np.ndarray
    person: np.ndarray
    n_persons: int

    @classmethod
    def of(cls, values: Any, person: Any, n_persons: int) -> "Recalls":
        v = np.asarray(values, dtype=float)
        v = v.reshape(len(v), -1)
        p = np.asarray(person, dtype=np.int64)
        keep = np.isfinite(v).all(axis=1) & (p >= 0)
        return cls(v[keep], p[keep], int(n_persons))

    @property
    def p(self) -> int:
        return int(self.values.shape[1])

    def counts(self) -> np.ndarray:
        return np.bincount(self.person, minlength=self.n_persons).astype(float)

    def means(self) -> np.ndarray:
        """Each person's mean recall (n × p; NaN for a person with none)."""
        k = self.counts()
        out = np.full((self.n_persons, self.p), np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            for j in range(self.p):
                out[:, j] = np.bincount(self.person, weights=self.values[:, j],
                                        minlength=self.n_persons) / k
        return out

    def take(self, draw: np.ndarray) -> "Recalls":
        """The recalls of the people ``draw`` names (with repeats), renumbered 0 … len(draw) − 1."""
        draw = np.asarray(draw, dtype=np.int64)
        order = np.argsort(self.person, kind="stable")
        k = self.counts().astype(np.int64)
        starts = np.concatenate([[0], np.cumsum(k)[:-1]])
        kb = k[draw]
        total = int(kb.sum())
        offsets = np.arange(total) - np.repeat(np.cumsum(kb) - kb, kb)
        rows = order[np.repeat(starts[draw], kb) + offsets]
        return Recalls(self.values[rows], np.repeat(np.arange(len(draw)), kb), len(draw))

    def subset(self, keep: np.ndarray) -> "Recalls":
        """The recalls of the people ``keep`` (a boolean mask over them), renumbered in order."""
        keep = np.asarray(keep, dtype=bool)
        remap = np.full(self.n_persons, -1)
        remap[keep] = np.arange(int(keep.sum()))
        new = remap[self.person]
        on = new >= 0
        return Recalls(self.values[on], new[on], int(keep.sum()))


@dataclass(frozen=True)
class Replicates(Recalls):
    """One intake's recalls (``values`` one-dimensional): the univariate form of :class:`Recalls`."""

    @classmethod
    def of(cls, values: Any, person: Any, n_persons: int) -> "Replicates":
        v = np.asarray(values, dtype=float).ravel()
        p = np.asarray(person, dtype=np.int64)
        keep = np.isfinite(v) & (p >= 0)
        return cls(v[keep], p[keep], int(n_persons))

    @property
    def p(self) -> int:
        return 1

    def means(self) -> np.ndarray:  # type: ignore[override]
        k = self.counts()
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.bincount(self.person, weights=self.values, minlength=self.n_persons) / k

    def joint(self) -> Recalls:
        return Recalls(self.values.reshape(-1, 1), self.person, self.n_persons)


def _as_recalls(rec: Recalls) -> Recalls:
    return rec.joint() if isinstance(rec, Replicates) else rec


# ── the within-person covariance and the calibration ─────────────────────────


@dataclass(frozen=True)
class WithinPerson:
    covariance: np.ndarray  # Σ_uu, p × p: the pooled within-person (day-to-day) covariance
    df: int  # Σ (k_i − 1)
    n_repeat: int  # people with two or more recalls

    @property
    def sigma2_u(self) -> float:
        return float(self.covariance[0, 0])


def within_person(rec: Recalls, weights: np.ndarray | None = None) -> WithinPerson:
    """Σ̂_uu = Σ_i w_i Σ_j (W_ij − W̄_i)(W_ij − W̄_i)ᵀ / Σ_i w_i (k_i − 1) (w ≡ 1 without weights)."""
    rec = _as_recalls(rec)
    k = rec.counts()
    df = int(np.sum(np.maximum(k - 1, 0)))
    n_repeat = int(np.sum(k >= 2))
    if df == 0:
        raise CalibrationRefused("No one has two or more recalls with a value, so the day-to-day "
                                 "variance cannot be estimated.")
    means = rec.means()
    dev = rec.values - means[rec.person]
    w = np.ones(rec.n_persons) if weights is None else np.asarray(weights, dtype=float)
    wd = w[rec.person]
    cov = (dev * wd[:, None]).T @ dev / float(np.sum(w * np.maximum(k - 1, 0)))
    return WithinPerson((cov + cov.T) / 2, df, n_repeat)


def within_person_variance(rep: Recalls) -> WithinPerson:
    """The univariate form of :func:`within_person` (its ``sigma2_u``)."""
    return within_person(rep)


@dataclass(frozen=True)
class Calibration:
    """E[X | W̄, Z] for every person with a recall (NaN otherwise), and its parts."""

    calibrated: np.ndarray  # n × p
    within: WithinPerson
    mu: np.ndarray  # μ̂_w
    sigma_xx: np.ndarray  # p × p
    sigma_xz: np.ndarray  # p × q
    sigma_zz: np.ndarray  # q × q
    conditional: np.ndarray  # Σ̂_x|z, p × p
    counts: np.ndarray  # k_i per person
    n: int  # people with a recall
    rank_z: int  # columns of [1, Z]
    weighted: bool

    @property
    def sigma_uu(self) -> np.ndarray:
        return self.within.covariance

    def slope(self, k: float) -> np.ndarray:
        """Γ_k = Σ̂_x|z (Σ̂_x|z + Σ̂_uu/k)⁻¹: the slope of E[X | W̄, Z] on W̄ for k recalls."""
        C = self.conditional
        return np.linalg.solve((C + self.sigma_uu / float(k)).T, C.T).T

    @property
    def modal_k(self) -> int:
        k = self.counts[self.counts >= 1].astype(np.int64)
        return int(np.bincount(k).argmax())

    @property
    def balanced(self) -> bool:
        k = self.counts[self.counts >= 1]
        return bool(len(k)) and bool(np.all(k == k[0]))

    # the univariate names
    @property
    def sigma2_u(self) -> float:
        return float(self.sigma_uu[0, 0])

    @property
    def sigma2_xz(self) -> float:
        return float(self.conditional[0, 0])

    @property
    def attenuation(self) -> np.ndarray:
        """λ_i of the first intake for each person (NaN without a recall)."""
        out = np.full(len(self.counts), np.nan)
        for k in np.unique(self.counts[self.counts >= 1]):
            out[self.counts == k] = self.slope(float(k))[0, 0]
        return out


def _normalized(weights: np.ndarray | None, n: int) -> np.ndarray:
    if weights is None:
        return np.ones(n)
    w = np.asarray(weights, dtype=float)
    if not (np.all(np.isfinite(w)) and np.all(w > 0)):
        raise ValueError("calibration weights must be positive and finite")
    return w * n / float(w.sum())


def calibrate(rec: Recalls, Z: np.ndarray | None, weights: np.ndarray | None = None) -> Calibration:
    """Carroll et al.'s best linear approximation of E[X | W̄, Z] (module docstring), for every
    person with at least one recall. ``Z`` (n × q) holds the calibration's covariates: every other
    column of the outcome model, never the outcome. ``weights``: survey weights."""
    rec = _as_recalls(rec)
    k_all = rec.counts()
    have = k_all >= 1
    n = int(have.sum())
    p = rec.p
    if n < 3:
        raise CalibrationRefused("Fewer than three people have a recall, so no calibration exists.")
    w_all = None if weights is None else np.asarray(weights, dtype=float)
    within = within_person(rec, w_all)
    k = k_all[have]
    Wb = rec.means()[have]
    Zh = (np.empty((n, 0)) if Z is None else np.asarray(Z, dtype=float).reshape(len(k_all), -1)[have])
    q = Zh.shape[1]
    w = _normalized(None if w_all is None else w_all[have], n)
    wk = w * k
    mu = (wk[:, None] * Wb).sum(axis=0) / float(wk.sum())
    z_bar = (w[:, None] * Zh).sum(axis=0) / float(w.sum()) if q else np.zeros(0)
    dW = Wb - mu
    dZ = Zh - z_bar
    nu = float(wk.sum() - (wk * k).sum() / wk.sum())
    if not nu > 0:
        raise CalibrationRefused("Too few people have recalls for the calibration's variances.")
    sigma_zz = (dZ * w[:, None]).T @ dZ / (n - 1) if q else np.zeros((0, 0))
    sigma_xz = (dW * wk[:, None]).T @ dZ / nu if q else np.zeros((p, 0))
    sigma_xx = ((dW * wk[:, None]).T @ dW - (n - 1) * within.covariance) / nu
    sigma_xx = (sigma_xx + sigma_xx.T) / 2
    rank_z = 1 + (int(np.linalg.matrix_rank(Zh - z_bar)) if q else 0)
    if n - rank_z <= 0:
        raise CalibrationRefused("There are no more people than columns in the calibration "
                                 "regression.")
    zz_inv = np.linalg.pinv(sigma_zz) if q else np.zeros((0, 0))
    conditional = sigma_xx - sigma_xz @ zz_inv @ sigma_xz.T if q else sigma_xx.copy()
    conditional = (conditional + conditional.T) / 2
    if not np.linalg.eigvalsh(conditional).min() > 0:
        raise CalibrationRefused(
            "The recalls vary as much within people as the people's means vary between them, once "
            "the other columns are allowed for: the true intakes' variance estimate is not "
            "positive, so no attenuation factor exists.")
    calibrated = np.full((len(k_all), p), np.nan)
    rows = np.flatnonzero(have)
    top = np.hstack([sigma_xx, sigma_xz])  # p × (p + q)
    for kk in np.unique(k):
        V = np.block([[sigma_xx + within.covariance / float(kk), sigma_xz],
                      [sigma_xz.T, sigma_zz]])
        M = top @ np.linalg.pinv(V)
        on = k == kk
        d = np.hstack([dW[on], dZ[on]])
        calibrated[rows[on]] = mu + d @ M.T
    return Calibration(calibrated, within, mu, sigma_xx, sigma_xz, sigma_zz, conditional, k_all, n,
                       rank_z, weights is not None)


# ── the outcome model ────────────────────────────────────────────────────────


Fit = Callable[..., np.ndarray]


def _design(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return np.column_stack([np.ones(len(X)), X.reshape(len(X), -1)])


def ols_fit(X: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """(Weighted) least-squares coefficients of ``y`` on ``X`` with an intercept (the intercept
    dropped)."""
    D = _design(X)
    yv = np.asarray(y, dtype=float)
    if weights is not None:
        root = np.sqrt(np.asarray(weights, dtype=float))
        D, yv = D * root[:, None], yv * root
    coef, *_ = np.linalg.lstsq(D, yv, rcond=None)
    return coef[1:]


def _logistic(X: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None
              ) -> tuple[np.ndarray, np.ndarray]:
    D = _design(X)
    yv = np.asarray(y, dtype=float)
    w = np.ones(len(yv)) if weights is None else np.asarray(weights, dtype=float)
    beta = np.zeros(D.shape[1])
    info = np.eye(D.shape[1])
    for _ in range(100):
        eta = np.clip(D @ beta, -35, 35)
        mu = 1.0 / (1.0 + np.exp(-eta))
        grad = D.T @ (w * (yv - mu))
        info = D.T @ (D * (w * mu * (1 - mu))[:, None])
        step = np.linalg.lstsq(info, grad, rcond=None)[0]
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    return beta, info


def logistic_fit(X: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Unpenalized (weighted) logistic regression coefficients (Newton–Raphson), intercept dropped."""
    return _logistic(X, y, weights)[0][1:]


def _call(fit: Fit, X: np.ndarray, y: np.ndarray, weights: np.ndarray | None) -> np.ndarray:
    return np.asarray(fit(X, y) if weights is None else fit(X, y, weights), dtype=float)


def model_covariance(fit: Fit, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The outcome model's own covariance of its coefficients (intercept dropped): σ̂²(DᵀD)⁻¹ for
    least squares, the inverse information for the logistic model."""
    D = _design(X)
    yv = np.asarray(y, dtype=float)
    if fit is logistic_fit:
        _, info = _logistic(X, yv)
        return np.linalg.inv(info)[1:, 1:]
    coef, *_ = np.linalg.lstsq(D, yv, rcond=None)
    resid = yv - D @ coef
    s2 = float(resid @ resid) / (len(yv) - D.shape[1])
    return s2 * np.linalg.inv(D.T @ D)[1:, 1:]


# ── one calibration of the outcome model ─────────────────────────────────────


@dataclass
class Corrected:
    """The outcome model's coefficients (intercept dropped) with each person's mean recall in the
    error-prone columns (``naive``) and with E[X | W̄, Z] in them (``estimate``)."""

    naive: np.ndarray
    estimate: np.ndarray
    calibration: Calibration
    columns: list[int]  # the error-prone columns of X
    covariates: list[int]  # the calibration's covariates: every other column of X
    rows: np.ndarray  # the people analyzed (each with a recall), as indices into X
    W: np.ndarray  # their mean recalls, n × p
    X: np.ndarray  # their matrix with the mean recalls in place


def correct(X: np.ndarray, columns: Sequence[int], rec: Recalls, y: Any, *, fit: Fit = ols_fit,
            weights: np.ndarray | None = None) -> Corrected:
    """Calibrate columns ``columns`` of the person-level matrix ``X`` jointly from their recalls
    ``rec`` (one column of ``rec`` per error-prone column, in that order). Every other column of
    ``X`` is a covariate of the calibration; the outcome ``y`` is never in it. People with no
    recall are left out. ``fit(X, y[, weights])`` returns the outcome model's coefficients
    (intercept dropped) in ``X``'s column order."""
    X = np.asarray(X, dtype=float)
    yv = np.asarray(y)
    rec = _as_recalls(rec)
    J = [int(j) for j in columns]
    if rec.n_persons != len(X) or len(yv) != len(X):
        raise ValueError("the recalls, the matrix and the outcome must describe the same people")
    if rec.p != len(J):
        raise ValueError("one column of recalls per error-prone column")
    have = rec.counts() >= 1
    rows = np.flatnonzero(have)
    if not have.all():
        rec = rec.subset(have)
        X, yv = X[have], yv[have]
        weights = None if weights is None else np.asarray(weights, dtype=float)[have]
    others = [c for c in range(X.shape[1]) if c not in set(J)]
    cal = calibrate(rec, X[:, others] if others else None, weights)
    W = rec.means()
    Xw = X.copy()
    Xw[:, J] = W
    Xc = X.copy()
    Xc[:, J] = cal.calibrated
    naive = _call(fit, Xw, yv, weights)
    estimate = _call(fit, Xc, yv, weights)
    return Corrected(naive, estimate, cal, J, others, rows, W, Xw)


def delta_covariance(corrected: Corrected, naive_cov: np.ndarray) -> np.ndarray | None:
    """Rosner et al.'s delta-method covariance of the calibrated coefficients of the error-prone
    columns (module docstring), from ``naive_cov`` (the outcome model's covariance of their
    uncorrected coefficients). None unless every person has the same number of recalls and the fit
    is unweighted: the closed form is defined there."""
    cal = corrected.calibration
    if cal.weighted or not cal.balanced:
        return None
    k = float(cal.modal_k)
    U = cal.sigma_uu
    S = cal.conditional + U / k
    p = len(U)
    b = corrected.naive[corrected.columns]
    S_inv = np.linalg.inv(S)
    A = np.eye(p) - S_inv @ U / k
    A_inv = np.linalg.inv(A)
    beta = A_inv @ b
    pairs = [(a, c) for a in range(p) for c in range(a, p)]

    def unit(a: int, c: int) -> np.ndarray:
        E = np.zeros((p, p))
        E[a, c] = E[c, a] = 1.0
        return E

    J_u = np.column_stack([A_inv @ S_inv @ unit(a, c) @ beta / k for a, c in pairs])
    J_s = np.column_stack([-A_inv @ S_inv @ unit(a, c) @ S_inv @ U @ beta / k for a, c in pairs])

    def wishart(M: np.ndarray, df: float) -> np.ndarray:
        return np.array([[(M[a, e] * M[c, f] + M[a, f] * M[c, e]) / df for e, f in pairs]
                         for a, c in pairs])

    V_u = wishart(U, float(cal.within.df))
    V_s = wishart(S, float(cal.n - cal.rank_z))
    V = A_inv @ np.asarray(naive_cov, dtype=float) @ A_inv.T + J_u @ V_u @ J_u.T + J_s @ V_s @ J_s.T
    return (V + V.T) / 2


# ── the whole-chain bootstrap ────────────────────────────────────────────────


@dataclass(frozen=True)
class Draw:
    """One bootstrap replicate: the people it holds (``rows``, indices with repeats), each one's
    weight multiplier (Rao–Wu's n_h/(n_h − 1); 1 otherwise), the drawn unit each belongs to
    (a unit drawn twice is two units) and its stratum."""

    rows: np.ndarray
    factor: np.ndarray
    unit: np.ndarray
    stratum: np.ndarray


def person_draw(n: int, rng: np.random.Generator) -> Draw:
    rows = rng.integers(0, n, n)
    return Draw(rows, np.ones(n), np.arange(n), np.zeros(n, dtype=np.int64))


def _members(codes: np.ndarray, G: int) -> list[np.ndarray]:
    order = np.argsort(codes, kind="stable")
    return np.split(order, np.cumsum(np.bincount(codes, minlength=G))[:-1])


def cluster_draw(codes: np.ndarray, rng: np.random.Generator) -> Draw:
    """G clusters drawn with replacement (``codes``: each person's cluster, 0 … G − 1), each with
    all its people."""
    codes = np.asarray(codes, dtype=np.int64)
    G = int(codes.max()) + 1 if len(codes) else 0
    members = _members(codes, G)
    drawn = rng.integers(0, G, G)
    parts = [members[g] for g in drawn]
    rows = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)
    unit = np.repeat(np.arange(G), [len(m) for m in parts])
    return Draw(rows, np.ones(len(rows)), unit, np.zeros(len(rows), dtype=np.int64))


def psu_draw(stratum: np.ndarray, psu: np.ndarray, rng: np.random.Generator,
             design_psus: Mapping[int, int] | None = None) -> Draw:
    """Rao & Wu's (1988) rescaling bootstrap with n_h − 1 PSUs: in each stratum with n_h ≥ 2 PSUs,
    n_h − 1 are drawn with replacement and each drawn person's weight is multiplied by
    n_h/(n_h − 1) (R ``survey``'s ``subbootstrap``). ``stratum`` and ``psu`` are each person's;
    ``design_psus`` maps every PSU of the design (those with no analyzed person too) to its stratum,
    so n_h counts them. A stratum with a single PSU is :data:`LONELY_RULE`: its PSU is drawn twice
    or not at all, each with chance 1/2, its weight unchanged."""
    stratum = np.asarray(stratum, dtype=np.int64)
    psu = np.asarray(psu, dtype=np.int64)
    every = dict(design_psus) if design_psus is not None else {}
    for s, u in zip(stratum, psu):
        every.setdefault(int(u), int(s))
    G = max(every) + 1 if every else 0
    members = _members(psu, max(G, int(psu.max()) + 1 if len(psu) else 0))
    by_stratum: dict[int, list[int]] = {}
    for u, s in sorted(every.items()):
        by_stratum.setdefault(s, []).append(u)
    rows, factor, unit, strata = [], [], [], []
    label = 0
    for s in sorted(by_stratum):
        units = by_stratum[s]
        n_h = len(units)
        if n_h >= 2:
            drawn = [units[i] for i in rng.integers(0, n_h, n_h - 1)]
            scale = n_h / (n_h - 1)
        else:  # a lonely PSU: twice or not at all, each with chance 1/2 (LONELY_RULE)
            drawn, scale = (units * 2 if rng.random() < 0.5 else []), 1.0
        for u in drawn:
            m = members[u] if u < len(members) else np.zeros(0, dtype=np.int64)
            rows.append(m)
            factor.append(np.full(len(m), scale))
            unit.append(np.full(len(m), label))
            strata.append(np.full(len(m), s))
            label += 1
    cat = (lambda parts, dtype: np.concatenate(parts).astype(dtype) if parts else np.zeros(0, dtype))
    return Draw(cat(rows, np.int64), cat(factor, float), cat(unit, np.int64), cat(strata, np.int64))


@dataclass(frozen=True)
class Resampling:
    """How the whole chain is redrawn: ``kind`` is ``"persons"``, ``"clusters"`` or
    ``"psu_within_strata"``."""

    kind: str
    n: int
    codes: np.ndarray | None = None  # clusters: each person's cluster
    stratum: np.ndarray | None = None  # PSUs: each person's stratum and PSU
    psu: np.ndarray | None = None
    design_psus: Mapping[int, int] | None = None
    lonely: int = 0  # strata with a single PSU, each drawn by LONELY_RULE

    def draw(self, rng: np.random.Generator) -> Draw:
        if self.kind == "psu_within_strata":
            return psu_draw(self.stratum, self.psu, rng, self.design_psus)  # type: ignore[arg-type]
        if self.kind == "clusters":
            return cluster_draw(self.codes, rng)  # type: ignore[arg-type]
        return person_draw(self.n, rng)

    @property
    def words(self) -> str:
        return {"psu_within_strata": "resampling PSUs within strata",
                "clusters": "resampling whole clusters",
                "persons": "resampling participants"}[self.kind]


@dataclass
class Bootstrap:
    """The replicates' estimates (``values``: replicates kept × quantities)."""

    values: np.ndarray
    n_boot: int
    failures: dict[str, int] = field(default_factory=dict)

    @property
    def n_ok(self) -> int:
        return int(len(self.values))

    def interval(self, level: float = LEVEL) -> tuple[np.ndarray, np.ndarray]:
        """The percentile interval of each quantity (Schomaker & Heumann's eq. 3.5)."""
        a = (1 - level) / 2
        if self.n_ok < 2:
            nan = np.full(self.values.shape[1] if self.values.ndim == 2 else 0, np.nan)
            return nan, nan
        return (np.quantile(self.values, a, axis=0), np.quantile(self.values, 1 - a, axis=0))

    def se(self) -> np.ndarray:
        if self.n_ok < 2:
            return np.full(self.values.shape[1] if self.values.ndim == 2 else 0, np.nan)
        return np.std(self.values, axis=0, ddof=1)


def whole_chain(estimate: Callable[[Draw, int], np.ndarray], resampling: Resampling, n_boot: int,
                seed: int, *, progress: Callable[[int, int], None] | None = None,
                cancelled: Callable[[], bool] | None = None) -> Bootstrap:
    """``n_boot`` replicates of ``estimate(draw, b)``: each redraws the people as ``resampling``
    says and repeats the whole chain on them. A replicate the data cannot carry (its calibration
    refused, its imputation model or fit singular) is counted under its reason and left out."""
    rng = np.random.default_rng(seed)
    kept: list[np.ndarray] = []
    failures: dict[str, int] = {}
    for b in range(int(n_boot)):
        if cancelled is not None and cancelled():
            from turbotab.core.graph import Cancelled

            raise Cancelled()
        draw = resampling.draw(rng)
        try:
            values = np.asarray(estimate(draw, b), dtype=float)
        except (CalibrationRefused, np.linalg.LinAlgError, ValueError) as refused:
            key = type(refused).__name__
            failures[key] = failures.get(key, 0) + 1
            continue
        if np.all(np.isfinite(values)):
            kept.append(values)
        else:
            failures["not finite"] = failures.get("not finite", 0) + 1
        if progress is not None:
            progress(b + 1, int(n_boot))
    width = len(kept[0]) if kept else 0
    return Bootstrap(np.asarray(kept).reshape(len(kept), width), int(n_boot), failures)


# ── the interval's leash ─────────────────────────────────────────────────────

INTERVALS = ("whole_chain_bootstrap", "model_based", "rubin_only")
INTERVAL_REFUSALS = {
    "model_based": (
        "A calibrated coefficient's model-based interval treats E[X | W̄, Z] as measured: "
        f"\"one cannot use the usual model standard errors from the outcome regression model when "
        f"performing RC, as these will be too small\" ({KEOGH}, §6.1.2). Its interval comes from "
        "a bootstrap over the whole chain."),
    "rubin_only": (
        "Rubin's rules over each imputed copy's model-based variance carry the same defect: each "
        "copy's variance ignores the calibration's own uncertainty. The calibrated coefficient's "
        "interval comes from a bootstrap that repeats the imputation, the calibration and the "
        f"outcome model (Boot MI, {SCHOMAKER})."),
}


def interval_refusal(kind: str) -> str | None:
    """Why an interval of ``kind`` is refused for a calibrated coefficient; None when it is the
    whole-chain bootstrap."""
    return INTERVAL_REFUSALS.get(kind)


# ── the univariate form, as WP12 wrote it ────────────────────────────────────


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


def regression_calibration(rep: Recalls, X: np.ndarray, j: int, y: np.ndarray, *, fit: Fit = ols_fit,
                           n_boot: int = 200, seed: int = 0) -> Result:
    """Calibrate column ``j`` of the person-level matrix ``X`` alone from its recalls ``rep``
    (:func:`correct` with one error-prone column), its interval the percentile interval of
    ``n_boot`` refits on people drawn with replacement, each with all their recalls."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    rec = _as_recalls(rep)
    if rec.n_persons != len(X) or len(y) != len(X):
        raise ValueError("the recalls, the matrix and the outcome must describe the same people")
    have = rec.counts() >= 1
    if not have.all():
        rec, X, y = rec.subset(have), X[have], y[have]
    first = correct(X, [j], rec, y, fit=fit)
    cal = first.calibration
    modal = cal.modal_k
    rng = np.random.default_rng(seed)
    boots, lams = [], []
    for _ in range(int(n_boot)):
        draw = rng.integers(0, rec.n_persons, rec.n_persons)
        try:
            again = correct(X[draw], [j], rec.take(draw), y[draw], fit=fit)
        except (CalibrationRefused, np.linalg.LinAlgError):
            continue
        b = float(again.estimate[j])
        if np.isfinite(b):
            boots.append(b)
            lams.append(float(again.calibration.slope(modal)[0, 0]))
    boot = np.asarray(boots)
    se = float(np.std(boot, ddof=1)) if len(boot) >= 2 else None
    low, high = ((float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975)))
                 if len(boot) >= 2 else (None, None))
    k = cal.counts.astype(np.int64)
    counts = np.bincount(k)
    return Result(
        naive=float(first.naive[j]), estimate=float(first.estimate[j]), se=se, ci_low=low,
        ci_high=high, attenuation=float(cal.slope(modal)[0, 0]),
        attenuation_mean=float(np.nanmean(cal.attenuation)),
        attenuation_se=float(np.std(lams, ddof=1)) if len(lams) >= 2 else None,
        sigma2_u=cal.sigma2_u, sigma2_xz=cal.sigma2_xz, n_persons=int(rec.n_persons),
        n_repeat=cal.within.n_repeat,
        recalls={int(i): int(c) for i, c in enumerate(counts) if c and i},
        n_boot=int(n_boot), n_boot_ok=int(len(boot)), boot=boot)


# ── the method contract (BLUEPRINT §13), in the one registry ─────────────────

RC_CONTRACT = "regression_calibration"
_STAGE = "turbotab.core.stages.calibration"
_SCOPE = ("Training rows: it learns the within-person covariance and the calibration from every "
          "analyzed participant's recalls and covariates, never from the outcome (the scope test "
          "holds it there); under inference there is no held-out fold, so it reads every analyzed "
          "row, inside each imputed copy.")


def _clause(run: Mapping[str, Any]) -> str | None:
    from turbotab.core.stages.calibration import main_clause

    return main_clause(run)


def _register_contract() -> None:
    from turbotab.core.contracts import (CONTRACTS as REGISTRY, ContractOption, MethodContract,
                                         Relation, register_contract)

    if RC_CONTRACT in REGISTRY:
        return
    inference = ("inference",)
    not_for_prediction = ("Not for prediction: the model is used on recalls measured the same way "
                          "as these, so its predictions need no correction.")

    def option(key: str, label: str, customary: str, sound: str, rung: str) -> ContractOption:
        return ContractOption(key, label, customary,
                              {"inference": sound, "prediction": not_for_prediction},
                              {"inference": rung, "prediction": "refused"})  # type: ignore[dict-item]

    register_contract(MethodContract(
        key=RC_CONTRACT, label="Regression calibration from repeated 24-hour recalls",
        slot="in_fold", scope="training_fold", scope_note=_SCOPE, run_order=0.75,
        needs=("two or more recall days for some participants, combined by the mean",
               "the outcome model's covariates (its adjustment set)",
               "the energy model's terms the recalls measure",
               "the survey design or the clusters, for the bootstrap"),
        question="Correct the coefficients for day-to-day error in the recalls (regression "
                 "calibration, declared as a secondary analysis)?",
        options=(
            option("multivariate",
                   "Every error-prone intake calibrated jointly (multivariate regression "
                   "calibration)",
                   f"{FREEDMAN}: multivariate calibration for the standard and partition models; "
                   f"{ROSNER_1990}",
                   f"Sound under its assumption ({LABEL}): with several error-prone intakes only "
                   f"the joint calibration removes the bias, which can go either way",
                   "recommended"),
            option("univariate",
                   "One error-prone intake calibrated (an energy-adjusted intake with energy out "
                   "of the model, or one study factor)",
                   f"{FREEDMAN}: univariate calibration of energy-adjusted intakes; {ROSNER_1989}",
                   f"Sound under its assumption ({LABEL}) when the model holds one error-prone "
                   f"intake; with several, the joint calibration is implied", "recommended"),
            option("one_at_a_time",
                   "Each intake calibrated on its own while the others stay uncorrected",
                   "Seen in cohort reports that apply a univariate attenuation factor per nutrient",
                   f"Unsound with several error-prone intakes: the attenuation factor is wrong and "
                   f"the bias can go either way ({ROSNER_1990}); never offered",
                   "refused"),
            option("none", "No correction: the mean of the recalls, uncorrected",
                   "Most reports; STROBE-nut nut-12.3 asks that any adjustment be reported",
                   "Sound as the primary: its test of no association stays valid; its estimate "
                   "is attenuated", "available"),
        ),
        storyboard=("Read each person's recall days of every intake the outcome model holds",
                    "Adjust energy on each recall day with the copy's own energy model",
                    "Estimate the day-to-day covariance from people with two or more recalls",
                    "Calibrate every error-prone intake jointly, given every other covariate",
                    "Refit the outcome model on the calibrated intakes, beside the uncorrected one",
                    "Redraw the whole chain (PSUs within strata, clusters or people) for its "
                    "interval"),
        relations=(
            Relation("implies", "every outcome-model covariate in the calibration model",
                     "The calibration equation holds every other column of the outcome model, and "
                     "never the outcome (Boe et al. 2023).", purposes=inference,
                     condition="regression calibration declared",
                     enforced_by="turbotab.core.methods.calibration:correct",
                     id="covariates_in_calibration"),
            Relation("implies", "multivariate calibration",
                     "Every error-prone intake is calibrated jointly, never one nutrient at a time "
                     f"({ROSNER_1990}).", purposes=inference,
                     condition="two or more error-prone intakes in the outcome model (total energy "
                               "beside an adjusted nutrient; every source of the all-components "
                               "model)",
                     enforced_by=f"{_STAGE}:error_prone", id="multivariate"),
            Relation("enables", "the all-components model calibrated",
                     "The all-components model's energy sources are calibrated jointly, so the "
                     "calibration and the all-components substitution coexist (MODELING_SEQUENCE "
                     "§0 ruling 7).", purposes=inference,
                     condition="the all-components or partition energy model",
                     enforced_by=f"{_STAGE}:error_prone", id="all_components"),
            Relation("implies", "energy adjusted per recall day before calibration",
                     "Energy is adjusted on each recall day with the copy's own energy model, then "
                     "calibrated.", purposes=inference,
                     condition="the residual or density energy model",
                     enforced_by=f"{_STAGE}:recall_matrix", id="energy_per_day_first"),
            Relation("implies", "a whole-chain bootstrap interval",
                     "Its interval comes from a bootstrap that repeats the imputation, the energy "
                     "model, the calibration and the outcome model, resampling PSUs within strata "
                     "under a survey design and clusters under repeated units.", purposes=inference,
                     condition="regression calibration declared",
                     enforced_by="turbotab.core.methods.calibration:whole_chain",
                     id="whole_chain_bootstrap"),
            Relation("conflicts", "model-based or Rubin-only intervals",
                     "Refused for a calibrated coefficient: the outcome model's own standard "
                     f"errors are too small ({KEOGH}, §6.1.2).", purposes=inference,
                     rung="refused",
                     exits=("the whole-chain bootstrap interval", "no correction"),
                     enforced_by="turbotab.core.decisions:_calibrated_interval_is_the_whole_chain",
                     id="model_or_rubin_intervals"),
            Relation("implies", "multiple_imputation_compatible",
                     "The calibration runs inside each completed copy, after the copy's energy "
                     "model, and each bootstrap resample is imputed again (Boot MI, "
                     f"{SCHOMAKER}).", purposes=inference,
                     condition="multiple imputation under inference",
                     enforced_by=f"{_STAGE}:calibration_stage", id="inside_each_copy"),
            Relation("implies", "survey_population",
                     "The calibration and the outcome model are survey-weighted, and PSUs are "
                     f"resampled within strata ({RAO_WU}); with no stratum of two PSUs it is "
                     "blocked and recorded, the sample-only attestation its exit.",
                     purposes=inference, condition="the surveyed-population answer",
                     enforced_by=f"{_STAGE}:resampling_of", id="psu_within_strata"),
            Relation("implies", "a lonely PSU resampled as the primary centers it",
                     f"A stratum with a single PSU is {LONELY_RULE}, so the calibrated interval "
                     "treats it as the primary's design-based table does, never as a stratum "
                     "with no variance.", purposes=inference,
                     condition="the surveyed-population answer, a stratum with a single PSU",
                     enforced_by="turbotab.core.methods.calibration:psu_draw", id="lonely_psu"),
            Relation("conflicts", "fewer clusters than the floor",
                     "Refused: with fewer whole clusters than cluster-robust intervals require, a "
                     "bootstrap of clusters cannot carry the calibrated coefficient's interval, "
                     "as the fit reports none for the uncorrected one.", purposes=inference,
                     rung="refused", exits=("no correction",),
                     condition="repeated units below the cluster floor",
                     enforced_by=f"{_STAGE}:cluster_floor", id="cluster_floor"),
            Relation("implies", "a declared secondary analysis beside the uncorrected estimate",
                     "The calibrated estimate sits beside the uncorrected one, whose estimate, "
                     f"interval and test of no association are the primary's; it {LABEL}.",
                     purposes=inference, condition="regression calibration declared",
                     enforced_by=f"{_STAGE}:calibration_stage", id="secondary_beside_uncorrected"),
            Relation("implies", "the calibration in the manuscript bundle",
                     "The calibrated estimates, their label and the methods paragraph reach the "
                     "export; a declared calibration that was not run is said to be blocked, "
                     "there and in the record.", purposes=inference,
                     condition="regression calibration declared",
                     enforced_by="turbotab.core.export.tables:calibration_table",
                     id="in_the_export"),
            Relation("invalidates", "the calibration's declaration",
                     "Declared again under the current adjustment set, never silently kept; "
                     "until then the record says it is re-asked and the export waits for it.",
                     purposes=inference, condition="a change to the adjustment set",
                     enforced_by=f"{_STAGE}:current_calibration", id="adjustment_set_invalidates"),
        ),
        sources=(ROSNER_1989, ROSNER_1990, CARROLL, FREEDMAN, SCHOMAKER, RAO_WU, KEOGH,
                 "Boe et al. 2023, Am J Epidemiol 192:1406",
                 "R survey 4.5, lonely.psu \"adjust\" (onestage, onestrat)",
                 "Beaumont & Patak 2012, Int Stat Rev 80:127"),
        decision="set_measurement_error", stage="calibration",
        place="MODELING_SEQUENCE §1 step 4 (planned), run as a declared secondary analysis (row 11)",
        leash={"inference": "recommended", "prediction": "refused"},
        short="regression calibration", clause=_clause,
        sentence=f"{_STAGE}:methods_sentence", package="RC"))


_register_contract()


__all__ = [
    "BOOT_COPIES", "Bootstrap", "CARROLL", "Calibration", "CalibrationRefused", "Corrected", "Draw",
    "FREEDMAN", "INTERVALS", "INTERVAL_REFUSALS", "KEOGH", "LABEL", "LEVEL", "LONELY_RULE",
    "NO_CALIBRATION", "RAO_WU", "ROSNER_1989", "ROSNER_1990", "Recalls", "Replicates",
    "Resampling", "Result", "SCHOMAKER", "WithinPerson",
    "calibrate", "cluster_draw", "correct", "delta_covariance", "interval_refusal",
    "logistic_fit", "model_covariance", "ols_fit", "person_draw", "psu_draw",
    "regression_calibration", "whole_chain", "within_person", "within_person_variance",
]
