"""The NCI method for the distribution of usual intake (V2 definition of done, "Dietary, extended").

A person's *usual intake* is their long-run average daily intake. One 24-hour recall measures it
with a day's worth of within-person variation, so the distribution of single days (or of a mean of
two days) is wider than the distribution of usual intake, and a share below a cut-off read from it
is wrong in both tails (NUTRITION_PACK §03: "Averaging 2 days leaves too much within-person variance
in the distribution; the tails are too fat"). The NCI method removes the within-person variance with
a measurement-error model fitted to repeated recalls on at least a subset of people.

**The models** (each quoted from its primary source, read 2026-10-05):

* *Amount-only*, for a nutrient nearly everyone consumes every day. Tooze et al. 2010 (*Stat Med*
  29:2857, PMC3865776, §3.1): "g(R_ij, λ) = X′_i β + u_i + e_ij", with "u_i ~ N(0, σ²_u)" and
  "e_ij ~ N(0, σ²_e)", where g is "a Box–Cox transformation, which includes the logarithmic
  transformation as a limiting case ... when λ = 0, the natural log transformation is used."
* *Two-part*, for an episodically consumed food. Kipnis et al. 2009 (*Biometrics* 65:1003,
  PMC2881223): "P(R_ij > 0 | i) ≡ p_i = H(β_10 + β′_X1 X_1i + u_1i)" (H the logistic function) and
  "g(R_ij, λ_R | R_ij > 0) = β_20 + β′_X2 X_2i + u_2i + ε_ij", with "u_i = (u_1i, u_2i)ᵗ =
  Normal(0; Σ_u)" and usual intake "T_i ≡ E(T_ij | i) = p_i A_i", "the product of the probability to
  consume and the usual amount on consumption days." Tooze et al. 2006 (*J Am Diet Assoc* 106:1575,
  PMC2517157): "the NCI method fits both parts simultaneously ... First, the two person-specific
  effects are modeled as correlated random variables." Its assumption, stated with every result:
  "Our two-part model assumes that each food is ultimately consumed by all individuals, so that
  T_i > 0" (Kipnis et al. 2009).

**How it is fit.** By maximum likelihood with λ estimated with the model: Tooze et al. 2006, "the
Box-Cox transformation parameter is also estimated as part of the likelihood maximization
procedure"; the NCI MIXTRAN macro (v2.1) starts λ from a grid "lambda=0.05 to 1 by 0.05" and frees
it in PROC NLMIXED with the bound "A_LAMBDA>0.01". Here λ ranges over [0, 1.5] (0 is the log).

* The amount-only likelihood is a linear mixed model's: for each λ the fixed effects and σ²_e are
  profiled out in closed form (generalized least squares under compound symmetry), the variance
  ratio τ = σ²_u/σ²_e and λ are found by golden-section search, and the likelihood carries the
  Box-Cox Jacobian (λ − 1) Σ log R (MIXTRAN's ``ll2 = ... + (&amtlambda-1)*log(&response)``).
* The two-part likelihood integrates the amount part's person effect analytically (given u₁, the
  consumption-day amounts are jointly normal) and u₁ by Gauss–Hermite quadrature (MIXTRAN integrates
  both in NLMIXED by adaptive Gaussian quadrature: Tooze et al. 2010, "quasi-Newton optimization of
  a likelihood approximated by the adaptive Gaussian quadrature"); the gradient is analytic and the
  optimizer is L-BFGS-B.

**Zeros in the amount-only model.** MIXTRAN: "for amount models change 0 intake to half the smallest
amount actually eaten" (and Tooze et al. 2010 §4: "Six recall days had 0 reported intake of
vitamin A; these were set to one-half of the lowest non-zero value"). So here too, and the count is
reported. Back-transformed amounts are floored at that half-minimum as DISTRIB does
(``mc_a = max(predcda,(.5*min_amt))``).

**Nuisance covariates.** Tooze et al. 2006: covariates "adjust for temporal effects, such as
seasonality and day-of-week effects, and the reduction in mean levels of intake that can occur with
repeat 24HRs, the time-in-sample effect ... by including an indicator variable into the statistical
model as a covariate to indicate that the second recall is being modeled." Here: an indicator of a
repeat recall (any recall after the person's first), and a weekend indicator, MIXTRAN's
"weekend (Fri.-Sun.) indicator variable ... A value of 1 represents a Fri.-Sun. record, and a value
of 0 represents a Mon.-Thurs. record." Usual intake is predicted for a first recall (MIXTRAN deletes
"the sequence variables" from the linear predictor DISTRIB uses) and as the weekday/weekend mixture
DISTRIB forms: "The default weights for weekdays and weekend days are 4/7 and 3/7 respectively."

**The distribution** is integrated, not simulated. DISTRIB draws 100 random pseudo-persons per person
(Tooze et al. 2010: "We use k = 100 simulated values per person") and back-transforms each with "the
nine-point approximation used by the ISU method". With no person-level covariates in the model every
person shares one linear predictor, so the integrals are one- and two-dimensional and are done here
by quadrature (Gauss–Legendre over the within-person error above the Box-Cox inverse's kink at
zero, exact inversion over the person effect for the amount-only model, Gauss–Hermite over u₁ with
exact conditional inversion over u₂ for the two-part model). That is the "adaptive quadrature" route the V2 definition allows in place of Monte
Carlo, and it carries no simulation noise. The acceptance tests check it against a brute-force
Monte Carlo of the same fitted model and against DISTRIB's nine-point rule.

**Variance.** Tooze et al. 2010: "Standard errors for the NCI method were estimated using a simple
bootstrap with 200 replications"; Kipnis et al. 2009 (NHANES): "Standard errors ... were estimated
using the 'balanced repeated replication method' (Wolter, 1995)". Every replicate refits the whole
model (λ included) and re-integrates the distribution:

* no survey design, or the "these participants" answer: a bootstrap over people (each with all their
  recalls), 200 resamples by default;
* a survey design with two PSUs in every stratum (NHANES's masked variance units): Fay's balanced
  repeated replication, coefficient 0.3, over a Hadamard matrix's columns (full orthogonal balance);
  the variance is Σ(θ_r − θ̂)² / (R(1 − K)²) and intervals use t on the design's degrees of
  freedom (strata);
* any other design: a bootstrap of PSUs within strata, rescaled (Rao & Wu 1988), intervals on
  #PSU − #strata degrees of freedom.

The fit under a survey design is a pseudo-likelihood weighted by the person's analysis weight, and
the distribution is weighted by it (MIXTRAN passes the weight to NLMIXED's ``replicate`` statement;
DISTRIB divides "the individual weight by the number of repetitions").
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Sequence

import numpy as np
from scipy import optimize, special, stats

TOOZE_2006 = "Tooze et al. 2006"
TOOZE_2010 = "Tooze et al. 2010"
KIPNIS_2009 = "Kipnis et al. 2009"
IOM_2000 = "Institute of Medicine 2000"

Model = Literal["amount_only", "two_part"]
Variance = Literal["bootstrap", "brr", "psu_bootstrap"]

PERCENTILES: tuple[int, ...] = (5, 10, 25, 50, 75, 90, 95)
WEEKEND_SHARE = 3.0 / 7.0  # DISTRIB's default ``wkend_prop`` (Fri.–Sun.)
FAY = 0.3
N_BOOT = 200  # Tooze et al. 2010: "a simple bootstrap with 200 replications"
LAMBDA_RANGE = (0.0, 1.5)
Z_95 = 1.959963984540054

# DISTRIB v2.1's nine-point back-transformation (``array cj`` / ``array wj``), for the comparison in
# the acceptance tests; this module integrates numerically instead (``back_transform``).
NINE_POINT_C = np.array([-2.1, -1.3, -0.8, -0.5, 0.0, 0.5, 0.8, 1.3, 2.1])
NINE_POINT_W = np.array([0.063345, 0.080255, 0.070458, 0.159698, 0.252489, 0.159698, 0.070458,
                         0.080255, 0.063345])


class UsualIntakeRefused(ValueError):
    """The recalls cannot support the model; the message says why."""


def _hermite(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and weights with Σ w f(z) ≈ E f(Z), Z ~ N(0, 1)."""
    z, w = np.polynomial.hermite_e.hermegauss(n)
    return z, w / math.sqrt(2.0 * math.pi)


GH_PERSON = _hermite(64)  # over a person effect, in the distribution
GH_FIT = _hermite(30)  # over u₁, in the two-part likelihood


# ── the Box-Cox transformation ───────────────────────────────────────────────


def boxcox(r: np.ndarray, lam: Any) -> np.ndarray:
    """(R^λ − 1)/λ, the log at λ = 0 (``lam`` broadcasts against ``r``)."""
    r = np.asarray(r, dtype=float)
    lam = np.asarray(lam, dtype=float)
    logr = np.log(r)
    small = np.abs(lam) < 1e-8
    safe = np.where(small, 1.0, lam)
    return np.where(small, logr, np.expm1(safe * logr) / safe)


def boxcox_dlam(r: np.ndarray, lam: float) -> np.ndarray:
    """∂g(R; λ)/∂λ = (R^λ(λ log R − 1) + 1)/λ², (log R)²/2 at λ = 0."""
    logr = np.log(np.asarray(r, dtype=float))
    if abs(lam) < 1e-5:
        return logr ** 2 / 2 + lam * logr ** 3 / 3
    return (np.exp(lam * logr) * (lam * logr - 1.0) + 1.0) / lam ** 2


def boxcox_inverse(z: np.ndarray, lam: Any) -> np.ndarray:
    """g⁻¹: (max(0, λz + 1))^{1/λ}, exp(z) at λ = 0 (DISTRIB's ``(max(0,xbt[j]*a_lambda+1))**(1/a_lambda)``)."""
    z = np.asarray(z, dtype=float)
    lam = np.asarray(lam, dtype=float)
    small = np.abs(lam) < 1e-8
    safe = np.where(small, 1.0, lam)
    base = np.maximum(0.0, safe * z + 1.0)
    with np.errstate(divide="ignore"):
        powered = np.exp(np.log(base) / safe)
    return np.where(small, np.exp(np.minimum(z, 700.0)), np.where(base > 0, powered, 0.0))


_GL_X, _GL_W = np.polynomial.legendre.leggauss(24)
_PANELS = 6
_REACH = 10.0  # the error's range integrated, in standard deviations


def back_transform(v: np.ndarray, lam: Any, sigma_e: Any, floor: float) -> np.ndarray:
    """A person's usual consumption-day amount at linear predictor ``v``: E_ε g⁻¹(v + ε), ε ~
    N(0, σ²_e), floored at half the smallest reported amount.

    At λ = 0 it is the lognormal mean exp(v + σ²_e/2), as DISTRIB computes it. For λ > 0,
    g⁻¹(z) = (λz + 1)^{1/λ} is zero below z = −1/λ and has a kink there, which a Gauss–Hermite rule
    integrates poorly (1% off when the kink sits within a standard deviation); so the integral is
    taken over the error's range above the kink, ±10σ_e, by six panels of 24-point Gauss–Legendre
    (within 10⁻⁷ of adaptive quadrature for λ ≤ 1, the acceptance tests check)."""
    v = np.asarray(v, dtype=float)
    lam = np.broadcast_to(np.asarray(lam, dtype=float), np.broadcast_shapes(v.shape, np.shape(lam)))
    se = np.broadcast_to(np.asarray(sigma_e, dtype=float), lam.shape)
    v = np.broadcast_to(v, lam.shape)
    small = np.abs(lam) < 1e-8
    lognormal = np.exp(np.minimum(v + se * se / 2.0, 700.0))
    safe = np.where(small, 1.0, lam)
    lo = np.maximum(-(v + 1.0 / safe), -_REACH * se)
    hi = _REACH * se
    width = np.maximum(hi - lo, 0.0) / _PANELS
    total = np.zeros(lam.shape)
    for k in range(_PANELS):
        a = lo + k * width
        e = a[..., None] + width[..., None] * (_GL_X + 1.0) / 2.0
        base = np.maximum(safe[..., None] * (v[..., None] + e) + 1.0, 0.0)
        with np.errstate(divide="ignore"):
            power = np.where(base > 0, np.exp(np.log(np.where(base > 0, base, 1.0)) / safe[..., None]), 0.0)
        density = np.exp(-0.5 * (e / se[..., None]) ** 2) / (se[..., None] * math.sqrt(2.0 * math.pi))
        total = total + width / 2.0 * ((power * density) @ _GL_W)
    return np.maximum(np.where(small, lognormal, total), floor)


# ── the recalls ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Recalls:
    """One dietary component's recalls: ``amount[m]`` is recall ``m`` of person ``person[m]``
    (0 … n − 1), sorted by person. ``later`` is 1 on any recall after the person's first (the
    sequence indicator); ``weekend`` 1 on a Friday–Sunday recall, or None when not recorded."""

    amount: np.ndarray
    person: np.ndarray
    n_persons: int
    later: np.ndarray
    weekend: np.ndarray | None = None

    @classmethod
    def of(cls, amount: Any, person: Any, n_persons: int | None = None, *, later: Any = None,
           weekend: Any = None) -> "Recalls":
        """Recalls with a value (and a weekend value, when given), sorted by person. ``later``
        defaults to "not the person's first recall in the order given"."""
        a = np.asarray(amount, dtype=float)
        p = np.asarray(person, dtype=np.int64)
        if later is None:
            seen: set[int] = set()
            first = np.zeros(len(p), dtype=bool)
            for m, i in enumerate(p.tolist()):
                if i not in seen and np.isfinite(a[m]):
                    first[m] = True
                    seen.add(i)
            lt = (~first).astype(float)
        else:
            lt = np.asarray(later, dtype=float)
        wk = None if weekend is None else np.asarray(weekend, dtype=float)
        keep = np.isfinite(a) & np.isfinite(lt) & (a >= 0)
        if wk is not None:
            keep &= np.isfinite(wk)
        order = np.argsort(p[keep], kind="stable")
        n = int(n_persons) if n_persons is not None else (int(p.max()) + 1 if len(p) else 0)
        return cls(a[keep][order], p[keep][order], n, lt[keep][order],
                   None if wk is None else wk[keep][order])

    def counts(self) -> np.ndarray:
        return np.bincount(self.person, minlength=self.n_persons)

    def positive_counts(self) -> np.ndarray:
        return np.bincount(self.person[self.amount > 0], minlength=self.n_persons)

    def zero_share(self) -> float:
        return float(np.mean(self.amount <= 0)) if len(self.amount) else float("nan")

    def has_weekend(self) -> bool:
        return self.weekend is not None and 0 < float(np.mean(self.weekend)) < 1

    def has_later(self) -> bool:
        return 0 < float(np.mean(self.later)) < 1

    def design(self, rows: np.ndarray | None = None) -> tuple[np.ndarray, list[str]]:
        """The nuisance design matrix ``[1, later, weekend]`` (columns that vary) and its names."""
        rows = slice(None) if rows is None else rows
        cols, names = [np.ones(len(self.amount))[rows]], ["intercept"]
        if self.has_later():
            cols.append(self.later[rows])
            names.append("repeat recall")
        if self.has_weekend():
            cols.append(self.weekend[rows])  # type: ignore[index]
            names.append("weekend")
        return np.column_stack(cols), names


def half_minimum(rec: Recalls) -> float:
    """Half the smallest reported positive amount (MIXTRAN's ``min_amt``·0.5)."""
    positive = rec.amount[rec.amount > 0]
    if not len(positive):
        raise UsualIntakeRefused("No recall reports any amount, so there is no intake to model.")
    return 0.5 * float(positive.min())


# ── lock-step golden-section search (one search per weighting, all at once) ─


_GOLD = (math.sqrt(5.0) - 1.0) / 2.0


def _maximize(f: Callable[[np.ndarray], np.ndarray], lo: float, hi: float, n: int, *,
              grid: int = 16, iters: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """For ``n`` independent problems, the argmax of ``f`` on [lo, hi] and its value.

    ``f`` maps an array of ``n`` points (one per problem) to their ``n`` values. A grid scan finds
    each problem's best grid point; golden-section search then refines inside its two neighbors.
    """
    points = np.linspace(lo, hi, grid)
    values = np.stack([f(np.full(n, x)) for x in points])  # grid × n
    values = np.where(np.isfinite(values), values, -np.inf)
    best = np.argmax(values, axis=0)
    a = points[np.maximum(best - 1, 0)]
    b = points[np.minimum(best + 1, grid - 1)]
    c = b - _GOLD * (b - a)
    d = a + _GOLD * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(iters):
        fc = np.where(np.isfinite(fc), fc, -np.inf)
        fd = np.where(np.isfinite(fd), fd, -np.inf)
        left = fc >= fd  # the maximum is in [a, d]
        b = np.where(left, d, b)
        a = np.where(left, a, c)
        nc = np.where(left, b - _GOLD * (b - a), d)
        nd = np.where(left, c, a + _GOLD * (b - a))
        new = np.where(left, nc, nd)
        fnew = f(new)
        fc, fd = np.where(left, fnew, fd), np.where(left, fc, fnew)
        c, d = nc, nd
    x = (a + b) / 2
    fx = f(x)
    grid_best = values[best, np.arange(n)]
    better = grid_best > fx
    return np.where(better, points[best], x), np.where(better, grid_best, fx)


# ── the amount-only model: a Box-Cox linear mixed model, fit for many weightings at once ──


@dataclass
class AmountFit:
    """Maximum-likelihood estimates, one row per weighting (row 0 the analysis' own)."""

    lam: np.ndarray
    beta: np.ndarray  # weightings × columns of the nuisance design
    sigma2_u: np.ndarray
    sigma2_e: np.ndarray
    loglik: np.ndarray
    names: list[str]


class _AmountData:
    """Per-person sufficient statistics of the recalls for the profile likelihood."""

    def __init__(self, amount: np.ndarray, person: np.ndarray, X: np.ndarray, n_persons: int):
        self.amount = amount
        self.logr = np.log(amount)
        order = np.argsort(person, kind="stable")
        self.amount, self.logr, X = amount[order], self.logr[order], X[order]
        self.person = person[order]
        self.k = np.bincount(self.person, minlength=n_persons).astype(float)
        present = self.k > 0
        self.present = present
        self.starts = np.concatenate([[0], np.cumsum(self.k[present])[:-1]]).astype(np.int64)
        self.X = X
        p = X.shape[1]
        self.p = p
        ks = np.unique(self.k[present])
        self.ks = ks
        self.onehot = (self.k[present][:, None] == ks[None, :]).astype(float)  # persons × K
        sx = np.add.reduceat(X, self.starts, axis=0)  # persons × p
        sxx = np.add.reduceat(X[:, :, None] * X[:, None, :], self.starts, axis=0)
        self.sx = sx
        self.sxx_flat = sxx.reshape(len(sx), p * p)
        self.sxsx_flat = (sx[:, :, None] * sx[:, None, :]).reshape(len(sx), p * p)
        self.slog = np.add.reduceat(self.logr, self.starts)
        self.kp = self.k[present]

    def weighted(self, W: np.ndarray) -> dict[str, np.ndarray]:
        """The λ-free weighted sums per recall count k, for weightings ``W`` (R × persons)."""
        Wp = W[:, self.present]
        R, p = len(W), self.p
        asx = np.swapaxes(np.swapaxes(Wp[:, :, None] * self.sxsx_flat[None], 1, 2) @ self.onehot, 1, 2)
        return {
            "n": Wp @ self.onehot,  # R × K
            "nrec": Wp @ self.kp,  # R
            "slog": Wp @ self.slog,  # R
            "axx_total": (Wp @ self.sxx_flat).reshape(R, p, p),
            "asx_flat": asx,  # R × K × p²
            "W": Wp,
        }


def _amount_y_sums(data: _AmountData, sums: dict[str, np.ndarray], lam: np.ndarray) -> dict[str, np.ndarray]:
    """The λ-dependent weighted sums for one λ per weighting."""
    Y = boxcox(data.amount[None, :], lam[:, None])  # R × recalls
    sy = np.add.reduceat(Y, data.starts, axis=1)  # R × persons
    syy = np.add.reduceat(Y * Y, data.starts, axis=1)
    sxy = np.add.reduceat(Y[:, :, None] * data.X[None, :, :], data.starts, axis=1)  # R × persons × p
    Wp = sums["W"]
    wsy = Wp * sy
    # per-person terms summed within each recall count k, by matrix products with the k indicator
    return {
        "bxy": np.swapaxes(np.swapaxes(Wp[:, :, None] * sxy, 1, 2) @ data.onehot, 1, 2),
        "bsxy": np.swapaxes(np.swapaxes(wsy[:, :, None] * data.sx[None, :, :], 1, 2) @ data.onehot,
                            1, 2),
        "cyy": (Wp * syy) @ data.onehot,
        "csy": (wsy * sy) @ data.onehot,
    }


def _amount_profile(data: _AmountData, sums: dict[str, np.ndarray], ysums: dict[str, np.ndarray],
                    lam: np.ndarray, log_tau: np.ndarray, full: bool = False) -> Any:
    """The log-likelihood profiled over β and σ²_e at (λ, τ), one per weighting."""
    tau = np.exp(log_tau)
    ks = data.ks
    R, p = len(tau), data.p
    c = tau[:, None] / (1.0 + ks[None, :] * tau[:, None])  # R × K
    A = sums["axx_total"] - (c[:, None, :] @ sums["asx_flat"]).reshape(R, p, p)
    b = ysums["bxy"].sum(axis=1) - (c[:, None, :] @ ysums["bsxy"])[:, 0, :]
    C = ysums["cyy"].sum(axis=1) - (c * ysums["csy"]).sum(axis=1)
    try:
        beta = np.linalg.solve(A, b[:, :, None])[:, :, 0]
    except np.linalg.LinAlgError:
        beta = np.stack([np.linalg.lstsq(A[r], b[r], rcond=None)[0] for r in range(len(A))])
    Q = C - np.einsum("ra,ra->r", b, beta)
    nrec = sums["nrec"]
    with np.errstate(divide="ignore", invalid="ignore"):
        s2 = Q / nrec
        ll = (-0.5 * (nrec * (math.log(2 * math.pi) + np.log(s2) + 1.0)
                      + (sums["n"] * np.log1p(ks[None, :] * tau[:, None])).sum(axis=1))
              + (lam - 1.0) * sums["slog"])
    ll = np.where(np.isfinite(ll) & (s2 > 0), ll, -np.inf)
    if full:
        return ll, beta, s2
    return ll


def fit_amount(amount: np.ndarray, person: np.ndarray, X: np.ndarray, n_persons: int,
               W: np.ndarray, names: Sequence[str], *, lam: float | None = None) -> AmountFit:
    """Maximum likelihood of ``g(R; λ) = Xβ + u + e`` for every row of ``W`` (persons' weights).

    ``amount`` must be positive. ``lam`` fixes λ (the acceptance tests' reference check)."""
    data = _AmountData(np.asarray(amount, dtype=float), np.asarray(person, dtype=np.int64),
                       np.asarray(X, dtype=float), n_persons)
    W = np.atleast_2d(np.asarray(W, dtype=float))
    sums = data.weighted(W)
    R = len(W)

    def inner(lam_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ys = _amount_y_sums(data, sums, lam_vec)
        return _maximize(lambda lt: _amount_profile(data, sums, ys, lam_vec, lt), -14.0, 8.0, R,
                         grid=12, iters=34)

    if lam is None:
        lam_hat, _ = _maximize(lambda lv: inner(lv)[1], LAMBDA_RANGE[0], LAMBDA_RANGE[1], R,
                               grid=11, iters=26)
    else:
        lam_hat = np.full(R, float(lam))
    log_tau, _ = inner(lam_hat)
    ys = _amount_y_sums(data, sums, lam_hat)
    ll, beta, s2 = _amount_profile(data, sums, ys, lam_hat, log_tau, full=True)
    return AmountFit(lam=lam_hat, beta=beta, sigma2_u=np.exp(log_tau) * s2, sigma2_e=s2, loglik=ll,
                     names=list(names))


# ── the two-part model ───────────────────────────────────────────────────────


@dataclass
class TwoPartFit:
    """Maximum-likelihood estimates of the two-part model for one weighting."""

    alpha: np.ndarray  # the probability part, on the nuisance design
    beta: np.ndarray  # the amount part, on the nuisance design
    sigma_u1: float
    sigma_u2: float
    rho: float
    sigma_e: float
    lam: float
    loglik: float
    converged: bool
    iterations: int
    names: list[str]
    logc: float = 0.0  # log of the scale constant the fit normalizes the Box-Cox scale by

    def theta(self) -> np.ndarray:
        """The optimizer's parameters: the amount part on the normalized scale g(R; λ)·c^{1−λ}."""
        shift = (1.0 - self.lam) * self.logc
        return np.concatenate([self.alpha, self.beta * math.exp(shift),
                               [math.log(self.sigma_u1), math.log(self.sigma_u2) + shift,
                                math.atanh(self.rho), math.log(self.sigma_e) + shift, self.lam]])


class _TwoPartData:
    def __init__(self, rec: Recalls, n_nodes: int = len(GH_FIT[0])):
        X, names = rec.design()
        self.names = names
        p = X.shape[1]
        self.p = p
        n = rec.n_persons
        self.n = n
        consumed = rec.amount > 0
        # The probability part, by pattern of the nuisance covariates (at most 4 patterns).
        Xu, pattern = np.unique(X, axis=0, return_inverse=True)
        pattern = pattern.reshape(-1)
        self.Xu = Xu
        U = len(Xu)
        self.C = np.zeros((n, U))
        np.add.at(self.C, (rec.person, pattern), 1.0)
        self.Cpos = np.zeros((n, U))
        np.add.at(self.Cpos, (rec.person[consumed], pattern[consumed]), 1.0)
        self.npos = self.Cpos.sum(axis=1)
        self.ipx = self.Cpos @ Xu  # Σ_j I_ij x_ij, persons × p
        # The amount part, on consumption days.
        self.pos_person = rec.person[consumed]
        self.pos_amount = rec.amount[consumed]
        self.pos_logr = np.log(self.pos_amount)
        self.pos_X = X[consumed]
        self.k = np.bincount(self.pos_person, minlength=n).astype(float)
        # The fit runs on g(R; λ)·c^{1−λ} with c the geometric mean of the consumption-day amounts:
        # the same model, reparameterized so that λ no longer moves the scale of the intercept and
        # the variances with it (their estimates correlate 0.95–0.97 with λ's on the raw scale).
        self.logc = float(np.mean(self.pos_logr)) if len(self.pos_logr) else 0.0
        self.slog = np.bincount(self.pos_person, weights=self.pos_logr - self.logc, minlength=n)
        self.sx = np.stack([np.bincount(self.pos_person, weights=self.pos_X[:, a], minlength=n)
                            for a in range(p)], axis=1)
        self.z, self.w = _hermite(n_nodes)
        self.logw = np.log(self.w)

    def unpack(self, theta: np.ndarray) -> tuple[Any, ...]:
        p = self.p
        alpha, beta = theta[:p], theta[p:2 * p]
        ls1, ls2, eta, lse, lam = theta[2 * p:2 * p + 5]
        return alpha, beta, math.exp(ls1), math.exp(ls2), math.tanh(eta), math.exp(lse), lam


def _two_part_loglik(theta: np.ndarray, d: _TwoPartData, w: np.ndarray,
                     gradient: bool = True) -> tuple[float, np.ndarray]:
    """Σ_i w_i log L_i and its gradient in θ = (α, β, log σ₁, log σ₂, atanh ρ, log σ_e, λ)."""
    alpha, beta, s1, s2, rho, se, lam = d.unpack(theta)
    z = d.z
    se2 = se * se
    c1 = rho * s2  # E(u₂ | u₁ = σ₁z) = ρσ₂z
    v2 = s2 * s2 * (1.0 - rho * rho)  # var(u₂ | u₁)
    tau = v2 / se2
    # the probability part: B[i, q]
    a_u = d.Xu @ alpha  # U
    eta = a_u[:, None] + s1 * z[None, :]  # U × Q
    sp = np.logaddexp(0.0, eta)
    E = special.expit(eta)
    B = (d.Cpos @ a_u)[:, None] + s1 * z[None, :] * d.npos[:, None] - d.C @ sp
    # the amount part: A[i, q]
    scale = math.exp((1.0 - lam) * d.logc)
    g = boxcox(d.pos_amount, lam)
    y = g * scale
    r0 = y - d.pos_X @ beta
    n = d.n
    S1 = np.bincount(d.pos_person, weights=r0, minlength=n)
    S2 = np.bincount(d.pos_person, weights=r0 * r0, minlength=n)
    k = d.k
    D = 1.0 + k * tau
    Sr = S1[:, None] - (k * c1)[:, None] * z[None, :]
    Srr = S2[:, None] - 2.0 * c1 * z[None, :] * S1[:, None] + (k * c1 * c1)[:, None] * (z * z)[None, :]
    A = -0.5 * ((k * (math.log(2 * math.pi) + math.log(se2)) + np.log(D))[:, None]
                + (Srr - (tau / D)[:, None] * Sr * Sr) / se2)
    L = d.logw[None, :] + B + A
    top = L.max(axis=1, keepdims=True)
    ex = np.exp(L - top)
    tot = ex.sum(axis=1)
    ll_i = top[:, 0] + np.log(tot) + (lam - 1.0) * d.slog
    value = float(w @ ll_i)
    if not gradient:
        return value, ll_i
    pi = ex / tot[:, None]  # posterior weights over the nodes
    m = pi @ z  # E(z | data)
    v = pi @ (z * z)  # E(z² | data)
    p = d.p
    # α and log σ₁
    g_alpha = (w @ d.ipx) - d.Xu.T @ (((w[:, None] * d.C) * (pi @ E.T)).sum(axis=0))
    g_ls1 = s1 * (float((w * d.npos) @ m) - float(((w[:, None] * d.C) * (pi @ (E * z[None, :]).T)).sum()))
    # β
    Sxr0 = np.stack([np.bincount(d.pos_person, weights=d.pos_X[:, a] * r0, minlength=n)
                     for a in range(p)], axis=1)
    ESr = S1 - k * c1 * m
    g_beta = w @ ((Sxr0 - d.sx * (c1 * m + (tau / D) * ESr)[:, None]) / se2)
    # c1, τ, σ²_e (τ held), λ
    dc1 = (S1 * m - k * c1 * v) / (se2 * D)
    ESr2 = S1 * S1 - 2.0 * S1 * k * c1 * m + (k * c1) ** 2 * v
    dtau = -0.5 * (k / D - ESr2 / (se2 * D * D))
    ESrr = S2 - 2.0 * c1 * m * S1 + k * c1 * c1 * v
    dse2 = -0.5 * (k / se2 - (ESrr - (tau / D) * ESr2) / (se2 * se2))
    yl = (boxcox_dlam(d.pos_amount, lam) - g * d.logc) * scale
    Syl = np.bincount(d.pos_person, weights=yl, minlength=n)
    Sr0yl = np.bincount(d.pos_person, weights=r0 * yl, minlength=n)
    dlam = -(Sr0yl - c1 * m * Syl - (tau / D) * ESr * Syl) / se2 + d.slog
    g_ls2 = w @ (dc1 * c1 + dtau * 2.0 * tau)
    g_eta = w @ (dc1 * s2 * (1.0 - rho * rho) + dtau * (-2.0 * rho * tau))
    g_lse = w @ (dse2 * 2.0 * se2 + dtau * (-2.0 * tau))
    g_lam = float(w @ dlam)
    grad = np.concatenate([g_alpha, g_beta, [g_ls1, g_ls2, g_eta, g_lse, g_lam]])
    return value, grad


def _logistic(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    beta = np.zeros(X.shape[1])
    for _ in range(50):
        mu = special.expit(X @ beta)
        grad = X.T @ (w * (y - mu))
        hess = X.T @ (X * (w * mu * (1 - mu))[:, None])
        step = np.linalg.lstsq(hess, grad, rcond=None)[0]
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    return beta


def _two_part_start(rec: Recalls, w: np.ndarray) -> np.ndarray:
    """Starting values: the amount part's own mixed model on consumption days, and a logistic
    regression for the probability part scaled to the conditional scale of σ₁ = 1 (Zeger, Liang &
    Albert 1988: β_marginal ≈ β_conditional / √(1 + 0.346σ²))."""
    X, _ = rec.design()
    consumed = rec.amount > 0
    wd = w[rec.person]
    alpha = _logistic(X, consumed.astype(float), wd) * math.sqrt(1.0 + 0.346)
    keep = rec.positive_counts() > 0
    Wfit = np.where(keep, w, 0.0)[None, :]
    fit = fit_amount(rec.amount[consumed], rec.person[consumed], X[consumed], rec.n_persons, Wfit,
                     [])
    s2u = max(float(fit.sigma2_u[0]), 1e-4)
    lam = float(fit.lam[0])
    shift = (1.0 - lam) * float(np.mean(np.log(rec.amount[consumed])))  # to the fit's scale
    return np.concatenate([alpha, fit.beta[0] * math.exp(shift),
                           [0.0, 0.5 * math.log(s2u) + shift, 0.0,
                            0.5 * math.log(float(fit.sigma2_e[0])) + shift, lam]])


def fit_two_part(rec: Recalls, w: np.ndarray | None = None, *, start: np.ndarray | None = None,
                 lam: float | None = None) -> TwoPartFit:
    """Maximum likelihood of the two-part model, with person weights ``w`` (default 1)."""
    w = np.ones(rec.n_persons) if w is None else np.asarray(w, dtype=float)
    d = _TwoPartData(rec)
    total = float(w.sum())
    theta0 = _two_part_start(rec, w) if start is None else np.asarray(start, dtype=float).copy()
    p = d.p
    bounds = ([(None, None)] * (2 * p)
              + [(-8.0, 4.0), (-8.0, 4.0), (-3.8, 3.8), (-8.0, 4.0), LAMBDA_RANGE])
    if lam is not None:
        theta0[-1] = lam
        bounds[-1] = (lam, lam)
    theta0[-1] = min(max(theta0[-1], LAMBDA_RANGE[0]), LAMBDA_RANGE[1])

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        value, grad = _two_part_loglik(theta, d, w)
        if not np.isfinite(value):
            return 1e300, np.zeros_like(theta)
        return -value / total, -grad / total

    res = optimize.minimize(objective, theta0, jac=True, method="L-BFGS-B", bounds=bounds,
                            options={"maxiter": 500, "ftol": 1e-13, "gtol": 1e-7, "maxcor": 30})
    return _two_part_result(d, res.x, -float(res.fun) * total, bool(res.success), int(res.nit))


def _two_part_result(d: _TwoPartData, theta: np.ndarray, loglik: float, converged: bool,
                     iterations: int) -> TwoPartFit:
    alpha, beta, s1, s2, rho, se, lam_hat = d.unpack(theta)
    back = math.exp((float(lam_hat) - 1.0) * d.logc)  # to the plain Box-Cox scale
    return TwoPartFit(alpha=np.asarray(alpha), beta=np.asarray(beta) * back, sigma_u1=s1,
                      sigma_u2=s2 * back, rho=rho, sigma_e=se * back, lam=float(lam_hat),
                      loglik=loglik, converged=converged, iterations=iterations, names=d.names,
                      logc=d.logc)


def person_loglik(rec: Recalls, fit: TwoPartFit) -> np.ndarray:
    """Each person's log-likelihood of their recalls under ``fit`` (the Box-Cox Jacobian included):
    the terms the two-part fit maximizes, integrated over u₁ by Gauss–Hermite quadrature."""
    d = _TwoPartData(rec)
    return _two_part_loglik(fit.theta(), d, np.ones(rec.n_persons), gradient=False)[1]


def two_part_information(rec: Recalls, fit: TwoPartFit, w: np.ndarray | None = None,
                         step: float = 1e-5) -> np.ndarray:
    """The observed information of the mean log-likelihood at ``fit`` (central differences of the
    analytic gradient), symmetrized."""
    w = np.ones(rec.n_persons) if w is None else np.asarray(w, dtype=float)
    d = _TwoPartData(rec)
    theta = fit.theta()
    total = float(w.sum())
    H = np.empty((len(theta), len(theta)))
    for b in range(len(theta)):
        e = np.zeros(len(theta))
        e[b] = step
        H[:, b] = -(_two_part_loglik(theta + e, d, w)[1] - _two_part_loglik(theta - e, d, w)[1]) / (2 * step * total)
    return (H + H.T) / 2


def refit_two_part(rec: Recalls, w: np.ndarray, start: TwoPartFit, information: np.ndarray, *,
                   max_iter: int = 40, tol: float = 1e-8) -> TwoPartFit:
    """The two-part fit under replicate weights ``w``, from the full fit: BFGS whose first inverse
    Hessian is the full sample's (each replicate's differs from it by O(n^-1/2)), so it starts as
    Newton's method would. A fit that leaves λ's range is redone with L-BFGS-B's bounds."""
    d = _TwoPartData(rec)
    total = float(w.sum())

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        if not (LAMBDA_RANGE[0] <= theta[-1] <= LAMBDA_RANGE[1]):
            return 1e300, np.zeros_like(theta)
        value, grad = _two_part_loglik(theta, d, w)
        if not np.isfinite(value):
            return 1e300, np.zeros_like(theta)
        return -value / total, -grad / total

    try:
        inverse = np.linalg.inv(information)
        res = optimize.minimize(objective, start.theta(), jac=True, method="BFGS",
                                options={"hess_inv0": (inverse + inverse.T) / 2, "gtol": 1e-7,
                                         "maxiter": max_iter * 5})
        theta = res.x
        ok = bool(res.success) or np.max(np.abs(res.jac)) < 1e-5
    except (np.linalg.LinAlgError, ValueError):
        ok, theta = False, start.theta()
    if ok and LAMBDA_RANGE[0] <= theta[-1] <= LAMBDA_RANGE[1] and np.all(np.isfinite(theta)):
        return _two_part_result(d, theta, -float(res.fun) * total, True, int(res.nit))
    return fit_two_part(rec, w, start=start.theta())


# ── the distribution of usual intake ─────────────────────────────────────────


def _weekend_mix(names: Sequence[str]) -> list[tuple[float, float]]:
    """``[(share, weekend value)]``: DISTRIB's 4/7 weekday and 3/7 weekend, or one term."""
    if "weekend" in names:
        return [(1.0 - WEEKEND_SHARE, 0.0), (WEEKEND_SHARE, 1.0)]
    return [(1.0, 0.0)]


def _reference(coef: np.ndarray, names: Sequence[str], weekend: float) -> np.ndarray:
    """The linear predictor at a first recall, on a weekday (0) or a weekend day (1)."""
    eta = coef[..., names.index("intercept")]
    if "weekend" in names:
        eta = eta + weekend * coef[..., names.index("weekend")]
    return eta


@dataclass
class Distribution:
    """Usual intake's distribution in one weighting: percentiles, mean, share below a cut-off."""

    percentiles: dict[int, float]
    mean: float
    below: float | None


def _bisect(f: Callable[[np.ndarray], np.ndarray], lo: np.ndarray, hi: np.ndarray, target: np.ndarray,
            iters: int = 80) -> np.ndarray:
    """The x in [lo, hi] with f(x) = target, f increasing (vectorized)."""
    lo, hi = np.array(lo, dtype=float), np.array(hi, dtype=float)
    for _ in range(iters):
        mid = (lo + hi) / 2
        up = f(mid) < target
        lo = np.where(up, mid, lo)
        hi = np.where(up, hi, mid)
    return (lo + hi) / 2


def amount_distribution(beta: np.ndarray, sigma2_u: np.ndarray, sigma2_e: np.ndarray, lam: np.ndarray,
                        names: Sequence[str], floor: float, cutoff: float | None,
                        percentiles: Sequence[int] = PERCENTILES) -> list[Distribution]:
    """Usual intake T(u) = Σ_w π_w A(η_w + u), u ~ N(0, σ²_u), one distribution per weighting.

    T is increasing in u, so its p-th percentile is T(σ_u z_p) exactly, and the share below c is
    Φ(u_c/σ_u) with T(u_c) = c."""
    beta = np.atleast_2d(beta)
    R = len(beta)
    su = np.sqrt(np.asarray(sigma2_u, dtype=float))
    se = np.sqrt(np.asarray(sigma2_e, dtype=float))
    lam = np.asarray(lam, dtype=float)
    mix = _weekend_mix(names)
    etas = [(share, _reference(beta, names, wk)) for share, wk in mix]

    def T(u: np.ndarray) -> np.ndarray:  # u: R × m
        out = np.zeros_like(u)
        for share, eta in etas:
            out = out + share * back_transform(eta[:, None] + u, lam[:, None], se[:, None], floor)
        return out

    zp = stats.norm.ppf(np.asarray(percentiles, dtype=float) / 100.0)
    values = T(su[:, None] * zp[None, :])
    zg, wg = GH_PERSON
    means = T(su[:, None] * zg[None, :]) @ wg
    below = None
    if cutoff is not None:
        lo, hi = np.full(R, -40.0), np.full(R, 40.0)
        zc = _bisect(lambda z: T((su * z)[:, None])[:, 0], lo, hi, np.full(R, float(cutoff)))
        below = stats.norm.cdf(zc)
        below = np.where(T((su * 40.0)[:, None])[:, 0] < cutoff, 1.0, below)
        below = np.where(T((su * -40.0)[:, None])[:, 0] >= cutoff, 0.0, below)
    return [Distribution({int(p): float(values[r, j]) for j, p in enumerate(percentiles)},
                         float(means[r]), None if below is None else float(below[r]))
            for r in range(R)]


def two_part_distribution(fit: TwoPartFit, floor: float, cutoff: float | None,
                          percentiles: Sequence[int] = PERCENTILES,
                          grid: int = 801, amount_grid: int = 1201) -> Distribution:
    """Usual intake T = Σ_w π_w expit(a_w + u₁) A(b_w + u₂), integrated: Gauss–Hermite over u₁ and,
    given u₁, exact inversion in u₂ (T rises with u₂), so F(t) = Σ_q ω_q Φ(z₂*(t | u₁ = σ₁z_q))."""
    names = fit.names
    z1, w1 = GH_PERSON
    s1, s2, rho = fit.sigma_u1, fit.sigma_u2, fit.rho
    cond = s2 * math.sqrt(max(1.0 - rho * rho, 1e-12))
    zgrid = np.linspace(-9.0, 9.0, grid)
    mix = [(share, float(_reference(fit.alpha, names, wk)), float(_reference(fit.beta, names, wk)))
           for share, wk in _weekend_mix(names)]
    # The consumption-day amount A(v) on a fine grid of its linear predictor, interpolated after.
    reach = abs(rho) * s2 * float(np.max(np.abs(z1))) + cond * 9.0
    vgrid = np.linspace(min(b for _, _, b in mix) - reach, max(b for _, _, b in mix) + reach, amount_grid)
    agrid = back_transform(vgrid, fit.lam, fit.sigma_e, floor)

    def amount(v: np.ndarray) -> np.ndarray:
        return np.interp(v, vgrid, agrid)

    table = np.zeros((len(z1), grid))  # T at (z₁ node, z₂ grid)
    for share, a, b in mix:
        prob = special.expit(a + s1 * z1)
        table += share * prob[:, None] * amount(b + rho * s2 * z1[:, None] + cond * zgrid[None, :])
    table = np.maximum.accumulate(table, axis=1)  # monotone in z₂ (guards rounding)

    def F(t: np.ndarray) -> np.ndarray:
        out = np.zeros(t.shape, dtype=float)
        for q in range(len(z1)):
            zs = np.interp(t, table[q], zgrid, left=-np.inf, right=np.inf)
            out += w1[q] * special.ndtr(zs)
        return out

    targets = np.asarray(percentiles, dtype=float) / 100.0
    # Invert F on log t (an episodic food's usual intake spans orders of magnitude): a coarse grid
    # brackets each percentile, a fine grid inside the bracket locates it.
    lo, hi = math.log(max(float(table.min()), 1e-300)), math.log(float(table.max()))
    coarse = np.linspace(lo, hi, 513)
    fc = F(np.exp(coarse))
    at = np.clip(np.searchsorted(fc, targets), 1, len(coarse) - 1)
    fine = coarse[at - 1][:, None] + (coarse[1] - coarse[0]) * np.linspace(0, 1, 129)[None, :]
    ff = F(np.exp(fine))
    logt = np.array([np.interp(p, np.maximum.accumulate(ff[i]), fine[i]) for i, p in enumerate(targets)])
    # the mean: two-dimensional Gauss–Hermite
    z2, w2 = GH_PERSON
    mean = 0.0
    for share, a, b in mix:
        prob = special.expit(a + s1 * z1)
        values = amount(b + rho * s2 * z1[:, None] + cond * z2[None, :])
        mean += share * float(w1 @ (prob[:, None] * values) @ w2)
    below = None if cutoff is None else float(F(np.array([float(cutoff)]))[0])
    return Distribution({int(p): float(math.exp(v)) for p, v in zip(percentiles, logt)}, mean, below)


# ── replicate weights ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Replication:
    """How the variance is estimated: replicate multipliers of the persons' weights."""

    method: Variance
    factors: np.ndarray  # replicates × persons
    df: int | None  # the interval's degrees of freedom (None: normal)
    fay: float | None = None
    n_strata: int | None = None
    n_psu: int | None = None

    def variance(self, estimate: float, replicates: np.ndarray) -> float:
        reps = np.asarray(replicates, dtype=float)
        reps = reps[np.isfinite(reps)]
        if len(reps) < 2:
            return float("nan")
        if self.method == "brr":
            return float(np.sum((reps - estimate) ** 2) / (len(reps) * (1.0 - (self.fay or 0.0)) ** 2))
        return float(np.var(reps, ddof=1))

    def critical(self) -> float:
        return Z_95 if self.df is None else float(stats.t.ppf(0.975, self.df))


def person_bootstrap(n_persons: int, n_boot: int = N_BOOT, seed: int = 0) -> Replication:
    """Resampling people with replacement (each with all their recalls), as count multipliers."""
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_persons, size=(n_boot, n_persons))
    factors = np.stack([np.bincount(d, minlength=n_persons) for d in draws]).astype(float)
    return Replication("bootstrap", factors, None)


def _hadamard(order: int) -> np.ndarray:
    from scipy.linalg import hadamard

    return hadamard(order)


def brr(strata: Sequence[Any], psu: Sequence[Any], fay: float = FAY) -> Replication:
    """Fay's balanced repeated replication over strata of exactly two PSUs each.

    Replicate r keeps stratum h's first PSU at weight factor 2 − K and its second at K when column
    h of a Hadamard matrix reads +1 on row r, the other way round on −1. The columns are 2 … L + 1 of
    a Hadamard matrix of order ≥ L + 1 (full orthogonal balance: each column sums to zero)."""
    s = np.asarray([str(v) for v in strata])
    c = np.asarray([str(v) for v in psu])
    levels = sorted(set(s.tolist()))
    L = len(levels)
    order = 4
    while order < L + 1:
        order *= 2
    H = _hadamard(order)[:, 1:L + 1]
    factors = np.ones((order, len(s)))
    for h, level in enumerate(levels):
        rows = s == level
        units = sorted(set(c[rows].tolist()))
        if len(units) != 2:
            raise UsualIntakeRefused(
                f"Stratum `{level}` has {len(units)} PSUs; balanced repeated replication needs "
                f"exactly two in every stratum.")
        first = rows & (c == units[0])
        second = rows & (c == units[1])
        plus = H[:, h] > 0
        factors[np.ix_(plus, first)] = 2.0 - fay
        factors[np.ix_(plus, second)] = fay
        factors[np.ix_(~plus, first)] = fay
        factors[np.ix_(~plus, second)] = 2.0 - fay
    return Replication("brr", factors, L, fay=fay, n_strata=L, n_psu=2 * L)


def psu_bootstrap(strata: Sequence[Any], psu: Sequence[Any], n_boot: int = N_BOOT,
                  seed: int = 0) -> Replication:
    """The rescaling bootstrap of PSUs within strata (Rao & Wu 1988, m_h = n_h − 1): replicate
    factor n_h/(n_h − 1) × (times the PSU was drawn)."""
    s = np.asarray([str(v) for v in strata])
    c = np.asarray([str(v) for v in psu])
    rng = np.random.default_rng(seed)
    factors = np.zeros((n_boot, len(s)))
    levels = sorted(set(s.tolist()))
    n_psu = 0
    for level in levels:
        rows = np.flatnonzero(s == level)
        units = sorted(set(c[rows].tolist()))
        nh = len(units)
        n_psu += nh
        if nh < 2:
            raise UsualIntakeRefused(f"Stratum `{level}` has one PSU, so its sampling variance "
                                     f"cannot be estimated.")
        index = {u: i for i, u in enumerate(units)}
        unit_of = np.array([index[v] for v in c[rows]])
        draws = rng.integers(0, nh, size=(n_boot, nh - 1))
        times = np.stack([np.bincount(d, minlength=nh) for d in draws]).astype(float)
        factors[:, rows] = (nh / (nh - 1.0)) * times[:, unit_of]
    return Replication("psu_bootstrap", factors, n_psu - len(levels), n_strata=len(levels),
                       n_psu=n_psu)


# ── the whole method ─────────────────────────────────────────────────────────


@dataclass
class Estimate:
    value: float
    se: float | None
    ci_low: float | None
    ci_high: float | None


@dataclass
class Result:
    """One dietary component's usual-intake distribution, with everything its sentence states."""

    model: Model
    names: list[str]  # the nuisance design's columns
    lam: float
    parameters: dict[str, float]
    percentiles: dict[int, Estimate]
    mean: Estimate
    below: Estimate | None
    cutoff: float | None
    n_persons: int
    n_recalls: int
    recalls: dict[int, int]  # recalls per person -> people
    n_repeat: int  # people with two or more recalls
    zero_share: float
    zeros_replaced: int
    floor: float
    replication: Replication | None
    n_replicates_ok: int
    replicate_values: dict[str, np.ndarray] = field(repr=False, default_factory=dict)
    day_one: dict[int, float] = field(default_factory=dict)  # single-day percentiles (not usual intake)
    mean_of_days: dict[int, float] = field(default_factory=dict)  # each person's mean of recalls
    concerns: list[str] = field(default_factory=list)
    converged: bool = True


def _weighted_percentiles(values: np.ndarray, weights: np.ndarray, ps: Sequence[int]) -> dict[int, float]:
    keep = np.isfinite(values) & (weights > 0)
    v, w = values[keep], weights[keep]
    if not len(v):
        return {}
    order = np.argsort(v)
    v, w = v[order], w[order]
    cum = (np.cumsum(w) - 0.5 * w) / w.sum()
    return {int(p): float(np.interp(p / 100.0, cum, v)) for p in ps}


def _estimates(point: Distribution, reps: list[Distribution | None], replication: Replication | None,
               percentiles: Sequence[int]) -> tuple[dict[int, Estimate], Estimate, Estimate | None,
                                                    dict[str, np.ndarray]]:
    ok = [r for r in reps if r is not None]

    def one(value: float, rep_values: np.ndarray) -> Estimate:
        if replication is None or len(rep_values) < 2:
            return Estimate(value, None, None, None)
        se = math.sqrt(replication.variance(value, rep_values))
        crit = replication.critical()
        return Estimate(value, se, value - crit * se, value + crit * se)

    store: dict[str, np.ndarray] = {}
    pct = {}
    for p in percentiles:
        vals = np.array([r.percentiles[p] for r in ok])
        store[f"p{p}"] = vals
        pct[p] = one(point.percentiles[p], vals)
    mvals = np.array([r.mean for r in ok])
    store["mean"] = mvals
    mean = one(point.mean, mvals)
    below = None
    if point.below is not None:
        bvals = np.array([r.below for r in ok], dtype=float)
        store["below"] = bvals
        below = one(point.below, bvals)
        if below.ci_low is not None:
            below = Estimate(below.value, below.se, max(0.0, below.ci_low), min(1.0, below.ci_high))
    return pct, mean, below, store


def usual_intake(rec: Recalls, model: Model, *, weights: np.ndarray | None = None,
                 domain: np.ndarray | None = None, replication: Replication | None = None,
                 cutoff: float | None = None, percentiles: Sequence[int] = PERCENTILES,
                 progress: Callable[[float, str], None] | None = None) -> Result:
    """The NCI method for one dietary component: fit, integrate, and replicate.

    ``weights`` are the persons' analysis weights (None: equal). ``domain`` marks the persons the
    distribution describes (a consumers-only domain); the replicate factors are formed over every
    person and then restricted, so the domain's size varies over replicates as it should.
    """
    n = rec.n_persons
    base = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    if domain is not None:
        base = base * np.asarray(domain, dtype=float)
    base = np.where(np.isfinite(base) & (base > 0), base, 0.0)
    have = (rec.counts() > 0) & (base > 0)
    if not have.any():
        raise UsualIntakeRefused("No one in the analysis has a recall with a value.")
    counts = rec.counts()[have]
    n_repeat = int(np.sum(counts >= 2))
    concerns: list[str] = []
    in_analysis = have[rec.person]
    floor = half_minimum(Recalls(rec.amount[in_analysis], rec.person[in_analysis], n,
                                 rec.later[in_analysis]))  # over the analysis' own recalls
    zeros_replaced = 0

    if model == "amount_only":
        pos_k = rec.positive_counts()[have]
        if int(np.sum(pos_k >= 2)) < 2:
            raise UsualIntakeRefused(
                "Fewer than two people have two or more recalls with an amount, so the day-to-day "
                "variance cannot be separated from the differences between people (MIXTRAN: \"there "
                "must be at least two subjects with at least two positive recalls\").")
        amount = np.where(rec.amount > 0, rec.amount, floor)
        zeros_replaced = int(np.sum((rec.amount <= 0) & in_analysis))
        X, names = rec.design()
        W = base[None, :]
        if replication is not None:
            W = np.vstack([W, replication.factors * base[None, :]])
        if progress:
            progress(0.1, f"Fitting the amount-only model{' and its replicates' if replication else ''}")
        keep = in_analysis
        fit = fit_amount(amount[keep], rec.person[keep], X[keep], n, W, names)
        dists = amount_distribution(fit.beta, fit.sigma2_u, fit.sigma2_e, fit.lam, names, floor,
                                    cutoff, percentiles)
        point, reps = dists[0], list(dists[1:])
        bad = ~(np.isfinite(fit.loglik[1:]) & np.isfinite(fit.sigma2_e[1:]))
        reps = [None if b else r for r, b in zip(reps, bad)]
        lam = float(fit.lam[0])
        params = {"lambda": lam, "sigma2_u": float(fit.sigma2_u[0]),
                  "sigma2_e": float(fit.sigma2_e[0]),
                  **{f"beta_{nm}": float(b) for nm, b in zip(names, fit.beta[0])}}
        converged = bool(np.isfinite(fit.loglik[0]))
        if lam >= LAMBDA_RANGE[1] - 1e-6:
            concerns.append(f"The Box-Cox λ reached the top of the range searched ({lam:.2f}); the "
                            f"intakes are less skewed than any transform in it straightens.")
    else:
        if not np.any((rec.amount <= 0) & in_analysis):
            raise UsualIntakeRefused(
                "No recall reports a day without it, so the probability part of the two-part model "
                "has nothing to estimate; the amount-only model is the one for an intake reported "
                "every day.")
        pos_k = rec.positive_counts()[have]
        if int(np.sum(pos_k >= 2)) < 2:
            raise UsualIntakeRefused(
                "Fewer than two people report it on two or more recalls, so the consumption-day "
                "amount's day-to-day variance cannot be estimated (MIXTRAN: \"In the amount model, "
                "there must be at least two subjects with at least two positive recalls\").")
        sub = Recalls(rec.amount[in_analysis], rec.person[in_analysis], n, rec.later[in_analysis],
                      None if rec.weekend is None else rec.weekend[in_analysis])
        if progress:
            progress(0.1, "Fitting the two-part model")
        fit2 = fit_two_part(sub, base)
        names = fit2.names
        point = two_part_distribution(fit2, floor, cutoff, percentiles)
        reps = []
        if replication is not None:
            try:
                information = two_part_information(sub, fit2, base)
                np.linalg.cholesky(information)
            except np.linalg.LinAlgError:
                information = None
            R = len(replication.factors)
            for r in range(R):
                if progress and r % 10 == 0:
                    progress(0.15 + 0.8 * r / R, f"Refitting replicate {r + 1:,} of {R:,}")
                wr = replication.factors[r] * base
                try:
                    fr = (refit_two_part(sub, wr, fit2, information) if information is not None
                          else fit_two_part(sub, wr, start=fit2.theta()))
                    reps.append(two_part_distribution(fr, floor, cutoff, percentiles)
                                if np.isfinite(fr.loglik) else None)
                except (np.linalg.LinAlgError, ValueError, FloatingPointError):
                    reps.append(None)
        lam = fit2.lam
        params = {"lambda": lam, "sigma_u1": fit2.sigma_u1, "sigma_u2": fit2.sigma_u2,
                  "rho": fit2.rho, "sigma2_e": fit2.sigma_e ** 2,
                  **{f"alpha_{nm}": float(a) for nm, a in zip(names, fit2.alpha)},
                  **{f"beta_{nm}": float(b) for nm, b in zip(names, fit2.beta)}}
        converged = fit2.converged
        if not converged:
            concerns.append("The two-part model's optimizer stopped before its convergence "
                            "criterion; the estimates are the best it reached.")
        if abs(fit2.rho) > 0.99:
            concerns.append(f"The correlation of the person effects reached {fit2.rho:.2f}, near "
                            f"its bound.")
    pct, mean, below, store = _estimates(point, reps, replication, percentiles)
    n_ok = sum(r is not None for r in reps)
    if replication is not None and n_ok < len(reps):
        concerns.append(f"{n_ok:,} of {len(reps):,} replicates could be refit; the standard errors "
                        f"rest on those.")
    # the shrinkage comparison: single days and means of days (neither is usual intake)
    first = (rec.later == 0) & in_analysis
    w_rec = base[rec.person]
    day_one = _weighted_percentiles(rec.amount[first], w_rec[first], percentiles)
    sums = np.bincount(rec.person[in_analysis], weights=rec.amount[in_analysis], minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = sums / rec.counts()
    mean_of_days = _weighted_percentiles(means[have], base[have], percentiles)
    values, nper = np.unique(counts, return_counts=True)
    return Result(model=model, names=list(names), lam=lam, parameters=params, percentiles=pct,
                  mean=mean, below=below, cutoff=cutoff, n_persons=int(have.sum()),
                  n_recalls=int(in_analysis.sum()),
                  recalls={int(v): int(c) for v, c in zip(values, nper)}, n_repeat=n_repeat,
                  zero_share=float(np.mean(rec.amount[in_analysis] <= 0)),
                  zeros_replaced=zeros_replaced, floor=floor, replication=replication,
                  n_replicates_ok=n_ok, replicate_values=store, day_one=day_one,
                  mean_of_days=mean_of_days, concerns=concerns, converged=converged)


# ── the methods sentence ─────────────────────────────────────────────────────

CutoffKind = Literal["EAR", "AI", "UL", "other"]


def days_phrase(recalls: dict[int, int]) -> str:
    ks = sorted(int(k) for k in recalls)
    if not ks:
        return "no recall"
    if len(ks) == 1:
        return f"{ks[0]} {'recall' if ks[0] == 1 else 'recalls'} each"
    return f"{ks[0]} to {ks[-1]} recalls each"


def _number(x: float) -> str:
    return f"{x:,.6g}"


def covariates_phrase(names: Sequence[str]) -> str:
    """The nuisance covariates the model carried, as a clause."""
    parts = []
    if "repeat recall" in names:
        parts.append("an indicator of a repeat recall")
    if "weekend" in names:
        parts.append("a weekend (Friday–Sunday) indicator")
    if not parts:
        return "no nuisance covariates"
    return " and ".join(parts) + " as nuisance covariates"


def reference_phrase(names: Sequence[str]) -> str:
    """Where the distribution is predicted for the nuisance covariates (DISTRIB's convention)."""
    parts = []
    if "repeat recall" in names:
        parts.append("predicted for a first recall")
    if "weekend" in names:
        parts.append("weighted 4/7 weekday and 3/7 weekend")
    return ", ".join(parts) + ", " if parts else ""


def variance_phrase(replication: Replication | None, n_ok: int, weight: str | None) -> str:
    if replication is None:
        return "No standard errors were computed."
    every = "each refitting the whole model"
    if replication.method == "brr":
        return (f"Standard errors came from Fay's balanced repeated replication ({n_ok:,} of "
                f"{len(replication.factors):,} replicates over {replication.n_strata:,} strata of two "
                f"PSUs, Fay coefficient {replication.fay:g}), {every}, with t intervals on "
                f"{replication.df:,} degrees of freedom.")
    if replication.method == "psu_bootstrap":
        return (f"Standard errors came from {n_ok:,} bootstrap resamples of PSUs within strata "
                f"(Rao and Wu's rescaling, {replication.n_psu:,} PSUs in {replication.n_strata:,} "
                f"strata), {every}, with t intervals on {replication.df:,} degrees of freedom.")
    whom = "participants" if weight is None else f"participants, carrying `{weight}`"
    return f"Standard errors came from {n_ok:,} bootstrap resamples of {whom}, {every}."


def share_phrase(cutoff: float | None, kind: CutoffKind | None) -> str:
    if cutoff is None:
        return ""
    value = _number(cutoff)
    if kind == "EAR":
        return (f" The share below the EAR ({value}) is the EAR cut-point estimate of the prevalence "
                f"of inadequacy ({IOM_2000}).")
    if kind == "UL":
        return (f" The share above the UL ({value}) is read as a share at risk of excess only if "
                f"the recalls include supplements, since the UL applies to total intake.")
    return f" The share below {value} is reported as a share of the distribution, not a prevalence of inadequacy."


def methods_sentence(*, label: str, model: Model, names: Sequence[str], lam: float, n_persons: int,
                     recalls: dict[int, int], n_repeat: int, zeros_replaced: int, population: str,
                     consumer_column: str | None, weight: str | None,
                     replication: Replication | None, n_ok: int, cutoff: float | None,
                     cutoff_kind: CutoffKind | None, rho: float | None = None,
                     attestation: str | None = None) -> str:
    """The methods sentence: the model, the transform, the covariates, the variance method."""
    head = (f"Usual intake of `{label}` was estimated by the NCI method ({TOOZE_2006}; {TOOZE_2010}) "
            f"from 24-hour recalls of {n_persons:,} participants ({days_phrase(recalls)}; "
            f"{n_repeat:,} with two or more)")
    weighted = f", each participant weighted by `{weight}`" if weight else ""
    covs = covariates_phrase(names)
    if model == "amount_only":
        body = (f": an amount-only model, a linear mixed model with a person-specific random "
                f"intercept fitted by maximum likelihood to Box-Cox-transformed intakes "
                f"(λ = {lam:.2f}, estimated with the model){weighted}, with {covs}")
        if zeros_replaced:
            body += (f"; {zeros_replaced:,} zero-intake {'recall was' if zeros_replaced == 1 else 'recalls were'} "
                     f"set to half the smallest reported amount, as the NCI MIXTRAN macro does")
        integrate = "back-transformed by numerical integration over the within-person error"
    else:
        body = (f": a two-part model for an episodically consumed food ({KIPNIS_2009}), the "
                f"probability of consumption on a recall day by a logistic mixed model and the "
                f"consumption-day amount by a linear mixed model on Box-Cox-transformed amounts "
                f"(λ = {lam:.2f}, estimated with the model), with correlated person-specific random "
                f"effects (ρ = {rho:.2f}) and "
                f"{'no nuisance covariates in either part' if covs.startswith('no ') else covs + ' in both parts'}"
                f", fitted jointly by maximum "
                f"likelihood{weighted}")
        integrate = ("integrated numerically over the within-person error and the person "
                     "effects, usual intake being the probability times the consumption-day amount")
    if population == "consumers" and consumer_column:
        whom = f"consumers only (the participants `{consumer_column}` marks)"
    else:
        whom = "the whole population"
        if model == "two_part":
            whom += f", everyone taken to consume it on some days ({KIPNIS_2009})"
    sentence = (f"{head}{body}; the distribution was {reference_phrase(names)}{integrate}, and "
                f"describes {whom}. {variance_phrase(replication, n_ok, weight)}"
                f"{share_phrase(cutoff, cutoff_kind)}")
    if attestation:
        sentence += f" Survey design: {attestation}."
    return sentence


__all__ = [
    "AmountFit", "CutoffKind", "Distribution", "covariates_phrase", "days_phrase", "methods_sentence",
    "reference_phrase", "share_phrase", "variance_phrase", "Estimate", "FAY", "IOM_2000", "KIPNIS_2009", "LAMBDA_RANGE",
    "N_BOOT", "NINE_POINT_C", "NINE_POINT_W", "PERCENTILES", "Recalls", "Replication", "Result",
    "TOOZE_2006", "TOOZE_2010", "TwoPartFit", "UsualIntakeRefused", "WEEKEND_SHARE", "Z_95",
    "amount_distribution", "back_transform", "boxcox", "boxcox_inverse", "brr", "fit_amount",
    "fit_two_part", "half_minimum", "person_bootstrap", "person_loglik", "refit_two_part",
    "two_part_information", "psu_bootstrap", "two_part_distribution",
    "usual_intake",
]
