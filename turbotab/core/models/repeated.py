"""Families for rows that repeat within a unit (AUDIT_REPORT §5 WP12: the exits WP2 left for MA-01).

Two families, each a plug-in on the shelf:

* ``mixed`` — the **random-intercept linear mixed model**, fit by restricted maximum likelihood
  (REML), with t intervals on **Satterthwaite degrees of freedom** (Giesbrecht & Burns 1985, the
  method lmerTest implements: Kuznetsova, Brockhoff & Christensen 2017, *J Stat Softw* 82(13)):
  for a coefficient's variance C_jj(θ), ν = 2 C_jj² / Var(Ĉ_jj), the variance by the delta method
  from the inverse of the REML log-likelihood's negative Hessian in the variance parameters θ.
  Its variance is model-based: it does not rest on unit-level residuals, so it is the exit when
  there are too few units for a cluster-robust interval. Luke (2017, *Behav Res Methods*
  49:1494) found "Type 1 error rates are closest to .05 when models are fitted using REML and
  p-values are derived using the Kenward-Roger or Satterthwaite approximations, as these
  approximations both produced acceptable Type 1 error rates even for smaller samples", where
  "applying the z distribution to the Wald t values from the model output (t-as-z)" was
  "somewhat anti-conservative, especially for smaller sample sizes".
* ``gee`` — **generalized estimating equations** with an exchangeable working correlation
  (Liang & Zeger 1986, *Biometrika* 73:13), for a continuous or a binary outcome: population-
  average effects. Its intervals are the WP2 CR2 engine (:func:`~turbotab.core.models.inference.cr2`)
  on the GEE's whitened working model, which is McCaffrey & Bell's bias-reduced linearization
  for GEE (2006, *Stat Med* 25:4081: "We propose tests that use bias-reduced linearization, BRL,
  to adjust the sandwich estimator and Satterthwaite or saddlepoint approximations for the
  reference distribution of resulting Wald t-tests"). Being a sandwich, it keeps the unit floor
  the linear family's cluster-robust intervals keep.

The REML fit is computed here in closed form for one variance component (each unit's rows share
an intercept): with ``V_g = σ²_e I + σ²_u J``, every quantity, the Satterthwaite df's information
among them, is a sum over units of each unit's sums of the columns of Q, the design's orthonormal
basis from one QR factorization, and of the triangle of a second, of Q's and the outcome's
within-unit deviations, so a fit costs those factorizations and a few passes over the units
whatever their sizes, the design's condition number is not squared, and at a large ratio of the
variances the directions a unit's rows share are not left to the rounding of differences near 1.
γ = σ²_u/σ²_e is searched up to :data:`GAMMA_GRID`'s top; where REML's likelihood still rises
there, the table is refused (:func:`beyond_words`). The GEE estimates
are statsmodels' (``statsmodels.genmod.generalized_estimating_equations.GEE``); only the
variance is computed here.

Each family's sklearn estimator takes the unit of every row through its ``units`` parameter (a
Series of unit labels indexed by row id, which the fit stage sets from the rows' identifier) and
predicts a new unit from its fixed effects alone.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from turbotab.core.decisions import Purpose, Task
from turbotab.core.models.base import (
    Assessment,
    FamilyBase,
    Identity,
    InferenceDecl,
    Situation,
    coefficient_rows,
    register_family,
)

# The REML profile over γ = σ²_u / σ²_e is searched on this log10 grid, then refined by Brent's
# method around the best grid point; γ = 0 (no between-unit variance) is checked on its own.
GAMMA_GRID = tuple(10.0 ** k for k in range(-8, 9))
FEW_UNITS_MIXED = 5  # below this, a between-unit variance is barely estimable: the shelf says so
EPV_RULE = 10


# ── which unit each row is in ────────────────────────────────────────────────


def unit_codes(units: Any, index: Any, n: int) -> np.ndarray | None:
    """One integer per row naming its unit, from ``units`` (a Series of unit labels indexed by
    row id) looked up by ``index`` (the rows' ids); None when nothing names a unit. A row whose
    unit is unknown is a unit of its own, as the split maps a missing identifier."""
    if units is None or index is None:
        return None
    labels = pd.Series(units).reindex(pd.Index(index)).to_numpy(dtype=object)
    missing = pd.isna(labels)
    labels = labels.copy()
    labels[missing] = [f"__row_{i}" for i in np.flatnonzero(missing)]
    codes = pd.factorize(pd.Series(labels, dtype=object))[0].astype(np.int64)
    return codes if len(codes) == n else None


def repeats(codes: np.ndarray | None) -> bool:
    return codes is not None and len(codes) > 0 and int(codes.max()) + 1 < len(codes)


def _matrix(X: Any) -> tuple[np.ndarray, list[str], Any]:
    if isinstance(X, pd.DataFrame):
        return X.to_numpy(dtype=float), [str(c) for c in X.columns], X.index
    M = np.asarray(X, dtype=float)
    return M, [f"x{j}" for j in range(M.shape[1])], None


# ── the random-intercept model by REML ───────────────────────────────────────


@dataclass(frozen=True)
class _Sums:
    """Each unit's sufficient statistics on an orthonormal basis of the design's columns: with
    ``X = QT`` (a Householder QR factorization, Q's columns orthonormal, T upper triangular),
    every REML quantity below is a function of these. ``XᵀX`` is never formed, so the design's
    condition number is not squared: a design whose columns are nearly, not exactly, dependent
    (:func:`dependent`) is factored to working precision. Nor is ``QᵀH⁻¹Q`` formed as ``I`` less
    the units' sums, which at a large γ leaves the directions a unit's rows share (its smallest
    eigenvalue, 1/(1 + γ m_g)) to the rounding of numbers near 1: each fit (:func:`_at`) factors
    the within-unit part's triangle ``R`` with each unit's sums of Q and y appended."""

    T: np.ndarray  # (P, P): X = QT
    logT: float  # log |det T|: log|XᵀH⁻¹X| = 2 logT + log|QᵀH⁻¹Q|
    z: np.ndarray  # (P,): Qᵀy
    yty: float
    S: np.ndarray  # (G, P): each unit's sums of Q's columns
    t: np.ndarray  # (G,): each unit's outcome sum
    m: np.ndarray  # (G,): each unit's row count
    N: int
    P: int
    # (P + 1, P + 1): the upper triangle of a QR factorization of [Q y] less each unit's means,
    # so ``RᵀR`` holds the within-unit cross-products of Q's columns and y, computed from the
    # deviations themselves.
    R: np.ndarray


# A column whose share of the dependencies' directions (the null space of the scaled design) is
# above this is in one of them; a column in none has a share of rounding, some 10⁻¹³.
IN_DEPENDENCY = 1e-6


def dependencies(X: np.ndarray) -> tuple[int, np.ndarray]:
    """How many dependencies the design's columns have to working precision, and the columns in
    any of them. With every column scaled to unit length (so units do not count; a column of zeros
    is a dependency of its own), a singular value at most the largest times max(N, P) machine
    epsilons (``numpy.linalg.matrix_rank``'s rule) is a dependency, and its right singular vectors
    span them: a column with weight in that span (:data:`IN_DEPENDENCY`) is a fixed combination of
    the others. REML has no fit where there is one (``log|XᵀV⁻¹X|`` is −∞); a factorization of the
    design would let rounding decide instead, the pivot an exact dependency leaves being a few ulps,
    or zero."""
    if X.shape[1] == 0:
        return 0, np.arange(0)
    _, vt, null = _spectrum(X)
    weight = np.sqrt((vt[null] ** 2).sum(axis=0))
    return int(null.sum()), np.flatnonzero(weight > IN_DEPENDENCY)


def _spectrum(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The singular values and right singular vectors of ``X`` with every column scaled to unit
    length, and which of them are dependencies (:func:`dependencies`' rule). With fewer rows than
    columns, each direction beyond the rows' count is one (a singular value of zero)."""
    N, P = X.shape
    norms = np.linalg.norm(X, axis=0)
    _, s, vt = np.linalg.svd(X / np.where(norms > 0, norms, 1.0), full_matrices=N < P)
    s = np.concatenate([s, np.zeros(P - len(s))])
    null = s <= s[0] * max(X.shape) * np.finfo(float).eps if s[0] > 0 else np.ones(P, bool)
    return s, vt, null


def basis(X: np.ndarray) -> tuple[int, np.ndarray, list[int] | None]:
    """The design's dependencies and the columns in any (:func:`dependencies`), and the columns a
    fit on it keeps: every column in no dependency, and of those in one each that is not, to the
    same working precision, a combination of the columns kept before it, so a copy's first column
    is kept and the copy left out, as R's ``lm`` leaves out an aliased column. A coefficient in no
    dependency has the same estimate and standard error whichever of them a fit keeps; one in a
    dependency has none. The kept columns are None where they do not number P less the
    dependencies, where working precision cannot tell which to keep."""
    P = X.shape[1]
    if P == 0:
        return 0, np.arange(0), []
    s, vt, null = _spectrum(X)
    count = int(null.sum())
    columns = np.flatnonzero(np.sqrt((vt[null] ** 2).sum(axis=0)) > IN_DEPENDENCY)
    if count == 0:
        return 0, columns, list(range(P))
    # The scaled design is U·(s Vᵀ) with U's columns orthonormal, so any of its columns have the
    # singular values of the same columns of s Vᵀ (P × P): no pass over the rows.
    tol = s[0] * max(X.shape) * np.finfo(float).eps
    small = s[:, None] * vt
    kept = [j for j in range(P) if j not in set(columns.tolist())]
    for j in columns:
        trial = sorted([*kept, int(j)])
        if np.linalg.svd(small[:, trial], compute_uv=False)[-1] > tol:
            kept = trial
    return count, columns, kept if len(kept) == P - count else None


@dataclass(frozen=True)
class Aliased:
    """What :func:`aliased` finds in ``[1, values]`` (column 0 the constant, column j the j-th of
    ``values``)."""

    count: int  # the dependencies: the columns set aside as aliased
    columns: np.ndarray  # every column in a dependency, sorted
    kept: list[int]  # the columns a fit keeps, in order
    exact: bool  # every aliased column is a combination of those before it to working precision


def aliased(values: np.ndarray) -> Aliased:
    """Which columns of ``[1, values]`` a coefficient table sets aside as aliased, and which are in
    a dependency, at R's tolerance (``effects.ALIAS_TOL``, ``lm``'s ``qr(x, tol = 1e-07)``) on the
    columns centered at their means (the intercept left alone, and first).

    Each column in turn is aliased where its part outside the span of the columns kept before it
    is under the tolerance times its own length (R's limited pivoting, ``effects.column_basis``),
    centered so that a column far from its origin is not a near copy of the intercept. Its
    dependency is the column less its least-squares fit on the kept columns, which these rows
    cannot tell from zero at that precision; carried back to the columns as they stand, each
    scaled to unit length (a constant column's dependency then holds the intercept), these
    directions span the dependencies, and a column with weight in that span above
    :data:`IN_DEPENDENCY` is in one, as :func:`dependencies` names an exact one. A coefficient in
    none has, to that precision, the same estimate whichever of the dependent columns a fit
    leaves out. ``exact`` where every aliased column's part was at most max(N, P) machine epsilons
    of its length (:func:`dependencies`' rule), a dependency to working precision: at R's
    tolerance alone, `x` beside ``x + 10⁻⁹·d`` is not one, and a fit keeping both served standard
    errors whose rounding was the design's condition number squared (+2.6 times `v`'s HC3 one)."""
    from turbotab.core.models.effects import ALIAS_TOL

    values = np.asarray(values, dtype=float)
    N, P = values.shape
    with np.errstate(invalid="ignore", over="ignore"):
        mean = np.mean(values, axis=0) if N else np.zeros(P)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    design = np.column_stack([np.ones(N), values - mean])
    floor = max(N, P + 1) * np.finfo(float).eps
    kept: list[int] = []
    Q = np.zeros((N, 0))
    exact = True
    for j in range(P + 1):
        x = design[:, j]
        length = float(np.linalg.norm(x))
        if length > 0.0:
            r = x - Q @ (Q.T @ x)
            r = r - Q @ (Q.T @ r)
            rest = float(np.linalg.norm(r))
            if rest > ALIAS_TOL * length:
                kept.append(j)
                Q = np.column_stack([Q, r / rest])
                continue
            exact = exact and rest <= floor * length
    out = [j for j in range(P + 1) if j not in set(kept)]
    if not out:
        return Aliased(0, np.arange(0), kept, True)
    # Each aliased column's direction n (design @ n ≈ 0) on the centered columns, then on the
    # columns as they stand, ``[1, values] = design · M`` with ``M`` moving the means into the
    # intercept, so a direction n there is ``M⁻¹``'s image: its constant entry less mean · n.
    fit = np.linalg.lstsq(design[:, kept], design[:, out], rcond=None)[0]
    null = np.zeros((P + 1, len(out)))
    null[out, np.arange(len(out))] = 1.0
    null[kept] = -fit
    null[0] -= mean @ null[1:]
    stood = np.column_stack([np.ones(N), values])
    norms = np.linalg.norm(stood, axis=0)
    basis_ = np.linalg.qr(null * np.where(norms > 0, norms, 1.0)[:, None])[0]
    weight = np.sqrt((basis_ ** 2).sum(axis=1))
    columns = np.array(sorted({*np.flatnonzero(weight > IN_DEPENDENCY).tolist(), *out}))
    return Aliased(len(out), columns, kept, exact)


def dependent(X: np.ndarray) -> bool:
    """Whether the design's columns are dependent to working precision (:func:`dependencies`)."""
    return dependencies(X)[0] > 0


class DependentColumns(np.linalg.LinAlgError):
    """REML has no fit: the design's columns are dependent to working precision (:func:`dependent`)."""


def _sums(X: np.ndarray, y: np.ndarray, codes: np.ndarray) -> _Sums:
    G = int(codes.max()) + 1
    Q, T = np.linalg.qr(X)
    S = np.zeros((G, X.shape[1]))
    np.add.at(S, codes, Q)
    t = np.bincount(codes, weights=y, minlength=G)
    m = np.bincount(codes, minlength=G).astype(float)
    with np.errstate(divide="ignore"):
        logT = float(np.log(np.abs(np.diag(T))).sum())
    per_row = np.divide(1.0, m, out=np.zeros_like(m), where=m > 0)
    within = np.column_stack([Q, y]) - (np.column_stack([S, t]) * per_row[:, None])[codes]
    return _Sums(T=T, logT=logT, z=Q.T @ y, yty=float(y @ y), S=S, t=t, m=m, N=len(y),
                 P=X.shape[1], R=np.linalg.qr(within, mode="r"))


def _between(s: _Sums, weight: np.ndarray) -> np.ndarray:
    """``Σ_g weight_g b_g b_gᵀ``, with ``b_g`` unit g's sums of Q's columns."""
    return (s.S * weight[:, None]).T @ s.S


@dataclass(frozen=True)
class _At:
    """The GLS fit at one γ = σ²_u/σ²_e, ``H_g = I + γJ``, on the basis Q (:class:`_Sums`)."""

    root: np.ndarray  # the lower Cholesky factor of K = QᵀH⁻¹Q
    w: np.ndarray  # the GLS coefficients on Q: Tβ
    beta: np.ndarray
    rss: float  # (y − Xβ)ᵀH⁻¹(y − Xβ)
    logH: float  # Σ log(1 + m_g γ) = log|H|
    logA: float  # log|XᵀH⁻¹X|


def _at(gamma: float, s: _Sums) -> _At:
    """The GLS fit at γ, from one QR factorization of at most P + 1 + G rows. Since
    ``H_g⁻¹ = I − c_g J`` with ``c_g = γ/(1 + m_g γ)``, ``[Q y]ᵀH⁻¹[Q y]`` is the within-unit
    cross-products plus ``Σ_g d_g (b_g, t_g)(b_g, t_g)ᵀ``, with
    ``d_g = 1/(m_g(1 + m_g γ)) = 1/m_g − c_g``: the Gram matrix of the within-unit triangle ``R``
    (:class:`_Sums`) stacked on the rows ``√d_g (b_g, t_g)``. That stack's triangle
    ``[[U, g], [0, ρ]]`` gives ``K = QᵀH⁻¹Q = UᵀU``, whose eigenvalues lie in
    [1/(1 + γ max m_g), 1], to working precision whatever γ and the units' sizes; ``Tβ = U⁻¹g``;
    the residual sum of squares ``ρ²``, never below zero; and
    ``log|XᵀH⁻¹X| = 2 log|det T| + log|K|``. Forming K as ``I − Σ_g c_g b_g b_gᵀ`` left its
    smallest eigenvalue, some 10⁻¹³ at γ = 10⁸ on units of 50,000 rows, to the rounding of a
    difference of numbers near 1, and the factorization failed."""
    from scipy.linalg import solve_triangular

    has = s.m > 0  # a code no row has adds nothing (its sums are zero), and no row here
    m = s.m[has]
    rows = np.sqrt(1.0 / (m * (1.0 + m * gamma)))[:, None] * np.column_stack([s.S, s.t])[has]
    full = np.linalg.qr(np.vstack([s.R, rows]), mode="r")
    upper, g, rho = full[:s.P, :s.P], full[:s.P, s.P], float(full[s.P, s.P])
    sign = np.where(np.diag(upper) < 0, -1.0, 1.0)
    root = (upper * sign[:, None]).T
    w = solve_triangular(upper, g)
    return _At(root=root, w=w, beta=solve_triangular(s.T, w), rss=rho ** 2,
               logH=float(np.log1p(s.m * gamma).sum()),
               logA=2.0 * s.logT + 2.0 * float(np.log(np.diag(root)).sum()))


def _profile(gamma: float, s: _Sums) -> float:
    """−2 × the REML log-likelihood with σ²_e profiled out, up to a constant."""
    at = _at(gamma, s)
    nu = s.N - s.P
    return nu * math.log(max(at.rss, 1e-300) / nu) + at.logH + at.logA


def reml_loglik(sigma2_u: float, sigma2_e: float, s: _Sums) -> float:
    """The REML log-likelihood ``−½[log|V| + log|XᵀV⁻¹X| + rᵀV⁻¹r]`` (no constant), with
    ``V = σ²_e H(γ)``: ``−½[(N − P) log σ²_e + Σ log(1 + m_g γ) + log|A| + RSS/σ²_e]``."""
    at = _at(sigma2_u / sigma2_e, s)
    return -0.5 * ((s.N - s.P) * math.log(sigma2_e) + at.logH + at.logA + at.rss / sigma2_e)


def _slope_at_zero(s: _Sums) -> float:
    """The profile's derivative in γ at γ = 0 (:func:`_profile`), in closed form:
    ``N − tr((XᵀX)⁻¹ SᵀS) − (N − P) Σ_g r_g² / RSS``, with ``r_g`` the sum of unit g's
    least-squares residuals. At γ = 0, ``d log(1 + m_g γ)/dγ = m_g``, ``dA/dγ = −SᵀS`` and, β being
    optimal (the envelope theorem), ``dRSS/dγ = −Σ_g r_g²``. On Q the least-squares coefficients
    are ``Qᵀy`` and ``tr((XᵀX)⁻¹SᵀS)`` is the sum of the squares of Q's unit sums."""
    rss = s.yty - float(s.z @ s.z)
    r = s.t - s.S @ s.z
    lever = float((s.S ** 2).sum())
    if rss <= 0:  # the rows are fit exactly at every γ: the profile's RSS term (clipped) is fixed
        return s.N - lever
    return s.N - lever - (s.N - s.P) * float(r @ r) / rss


def _slope(gamma: float, s: _Sums) -> float:
    """The profile's derivative in log γ at γ > 0 (:func:`_profile`), in closed form:
    ``γ[Σ_g m_g/(1 + m_g γ) − Σ_g b_gᵀK⁻¹b_g/(1 + m_g γ)² − (N − P) Σ_g r_g²/((1 + m_g γ)² RSS)]``,
    with ``r_g`` the sum of unit g's GLS residuals and ``b_g`` its sums of Q's columns: since
    ``dc_g/dγ = 1/(1 + m_g γ)²``, ``dK/dγ = −Σ_g b_g b_gᵀ/(1 + m_g γ)²`` and, β being optimal
    (the envelope theorem), ``dRSS/dγ = −Σ_g r_g²/(1 + m_g γ)²``. As γ grows it tends to the
    units, less the design's directions that units share, less N − P times the share of the RSS
    between units: below zero where the outcome barely varies within a unit beyond what the
    predictors explain."""
    from scipy.linalg import solve_triangular

    at = _at(gamma, s)
    one = 1.0 + s.m * gamma
    lever = (solve_triangular(at.root, s.S.T, lower=True) ** 2).sum(axis=0)
    slope = float((s.m / one).sum() - (lever / one ** 2).sum())
    if at.rss > 0:  # else the rows are fit exactly at every γ: the RSS term (clipped) is fixed
        r = s.t - s.S @ at.w
        slope -= (s.N - s.P) * float((r ** 2 / one ** 2).sum()) / at.rss
    return gamma * slope


def _best_gamma(s: _Sums) -> float:
    """γ̂, REML's σ̂²_u/σ̂²_e: 0 at the boundary, and ∞ where the profile is lowest at the grid's
    top and still falls there (:func:`_slope`), REML's ratio lying beyond what is searched."""
    from scipy.optimize import minimize_scalar

    values = [(_profile(g, s), g) for g in GAMMA_GRID]
    best, gamma = min(values)
    at_zero = _profile(0.0, s)
    i = GAMMA_GRID.index(gamma)
    if i == len(GAMMA_GRID) - 1 and _slope(gamma, s) < 0:
        return math.inf
    if i == 0 and at_zero <= best:
        # The profile is lowest toward γ = 0. Where it rises from γ = 0 the minimum is the
        # boundary itself, and its derivative there says so: comparing the profile at 0 with a
        # bounded search's point a few 10⁻¹³ from it compares two numbers that differ by less
        # than their rounding, so which way that fell depended on the arithmetic.
        if _slope_at_zero(s) >= 0:
            return 0.0
        # It falls from γ = 0: refine on [0, the first grid point].
        found = minimize_scalar(lambda g: _profile(g, s), bounds=(0.0, GAMMA_GRID[1]),
                                method="bounded", options={"xatol": 1e-14})
        return 0.0 if at_zero <= found.fun else float(found.x)
    lo = math.log(GAMMA_GRID[max(i - 1, 0)])
    hi = math.log(GAMMA_GRID[min(i + 1, len(GAMMA_GRID) - 1)])
    found = minimize_scalar(lambda z: _profile(math.exp(z), s), bounds=(lo, hi), method="bounded",
                            options={"xatol": 1e-10})
    return float(math.exp(found.x)) if found.fun <= best else gamma


@dataclass
class RandomInterceptFit:
    """A random-intercept model fit by REML on the design ``[1, X]`` (or whatever design given)."""

    beta: np.ndarray
    cov: np.ndarray  # C(θ̂) = (XᵀV̂⁻¹X)⁻¹
    sigma2_u: float  # between-unit variance
    sigma2_e: float  # within-unit (residual) variance
    df: np.ndarray | None  # Satterthwaite degrees of freedom per coefficient
    loglik: float  # the REML log-likelihood (no constant)
    boundary: bool  # σ̂²_u = 0: the units add nothing; the fit is least squares
    n_units: int
    # REML's γ̂ lies beyond the grid's top (:func:`_best_gamma`): the fit is the GLS one there, its
    # variances not REML's, and no df are computed
    beyond: bool = False

    @property
    def icc(self) -> float:
        total = self.sigma2_u + self.sigma2_e
        return self.sigma2_u / total if total > 0 else float("nan")


def fit_random_intercept(X: np.ndarray, y: np.ndarray, codes: np.ndarray, *,
                         satterthwaite: bool = True) -> RandomInterceptFit:
    """REML for ``y = Xβ + u_g + e``, ``u_g ~ N(0, σ²_u)``, ``e ~ N(0, σ²_e)``.

    γ = σ²_u/σ²_e is found by minimizing the profile (:func:`_profile`); then σ̂²_e = RSS/(N − P)
    (zero where the rows are fit exactly) and ``C = σ̂²_e A⁻¹``. With ``satterthwaite``, each
    coefficient's df are ``2 f² / (∇fᵀ 𝒜 ∇f)`` with ``f = C_jj``; ``∇f`` is analytic
    (``∂C/∂θ_k = C XᵀV⁻¹(∂V/∂θ_k)V⁻¹X C``) and ``𝒜`` is the inverse of the observed information,
    the negative Hessian of :func:`reml_loglik` in θ = (σ²_u, σ²_e), in closed form
    (:func:`satterthwaite_df`; the df are the same in any parameterization at the optimum). At the
    boundary σ̂²_u = 0 the between-unit variance is held at zero, and the df are N − P. Where the
    profile still falls at the grid's top (``beyond``, :func:`_best_gamma`), REML's γ̂ lies past
    what is searched: the fit is the GLS one at the top, whose fixed effects the larger γ̂ barely
    moves but whose variances are not REML's, and no df are computed (a table refuses it,
    :func:`mixed_table`). Where the design's columns are dependent (:func:`dependent`) REML has
    no fit, and :class:`DependentColumns` says the model matrix is singular.
    """
    from scipy.linalg import cho_solve, solve_triangular

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    codes = np.asarray(codes, dtype=np.int64)
    N, P = X.shape
    if N <= P:
        raise ValueError(f"{N:,} rows for {P:,} coefficients: the model cannot be fit.")
    if dependent(X):
        raise DependentColumns("the model matrix is singular")
    s = _sums(X, y, codes)
    gamma = _best_gamma(s)
    beyond = math.isinf(gamma)
    if beyond:
        gamma = GAMMA_GRID[-1]
    at = _at(gamma, s)
    sigma2_e = max(at.rss, 0.0) / (s.N - s.P)
    sigma2_u = gamma * sigma2_e
    inverse_T = solve_triangular(s.T, np.eye(P))
    cov = sigma2_e * inverse_T @ cho_solve((at.root, True), np.eye(P)) @ inverse_T.T
    cov = (cov + cov.T) / 2
    boundary = gamma == 0.0
    df = (satterthwaite_df(s, at, sigma2_u, sigma2_e, inverse_T, cov, boundary)
          if satterthwaite and not beyond else None)
    return RandomInterceptFit(beta=at.beta, cov=cov, sigma2_u=sigma2_u, sigma2_e=sigma2_e, df=df,
                              loglik=reml_loglik(sigma2_u, sigma2_e, s) if sigma2_e > 0 else float("nan"),
                              boundary=boundary, n_units=len(s.m), beyond=beyond)


def satterthwaite_df(s: _Sums, at: _At, sigma2_u: float, sigma2_e: float, inverse_T: np.ndarray,
                     cov: np.ndarray, boundary: bool) -> np.ndarray:
    """Each coefficient's Satterthwaite df (see :func:`fit_random_intercept`), from the observed
    REML information in θ = (σ²_u, σ²_e), ``I_kl = yᵀPV_kPV_lPy − ½ tr(PV_kPV_l)`` with
    ``P = V⁻¹ − V⁻¹XCXᵀV⁻¹``, ``V_u = blockdiag(J)`` and ``V_e = I`` (Harville 1977), in closed
    form: each unit's V has eigenvalue ``λ_g = σ²_e + m_g σ²_u`` on its rows' mean and σ²_e on the
    rest, so every trace and quadratic form is a sum over units of Q's and the residuals' unit sums,
    with Q's within-unit cross-products ``W`` (``I − Σ_g b_g b_gᵀ/m_g``) and the within-unit
    residual sum of squares, both from the within-unit triangle (:class:`_Sums`), not as those
    differences, which on large units leave a direction the units share to rounding. Taking the
    Hessian by differences of the log-likelihood instead lost it to rounding near the boundary,
    where its curvature in σ²_u is of the size of the rounding."""
    P = s.P
    if boundary:
        # σ²_u held at zero: only σ²_e varies, C = σ²_e(XᵀX)⁻¹ is proportional to it and
        # −ℓ'' = (N − P)/(2σ⁴), so ν = 2C²/((C/σ²_e)² · 2σ⁴/(N − P)) = N − P.
        return np.full(P, float(s.N - s.P))
    from scipy.linalg import cho_solve

    gamma = sigma2_u / sigma2_e
    lam = sigma2_e + s.m * sigma2_u
    # 1/m_g, and 0 for a code no row has (whose sums are zero; it adds nothing below)
    per_row = np.divide(1.0, s.m, out=np.zeros_like(s.m), where=s.m > 0)
    C = sigma2_e * cho_solve((at.root, True), np.eye(P))  # (QᵀV⁻¹Q)⁻¹
    W = s.R[:P, :P].T @ s.R[:P, :P]
    # QᵀV⁻¹V_kV⁻¹Q and QᵀV⁻¹V_kV⁻¹V_lV⁻¹Q
    M = {"u": _between(s, 1.0 / lam ** 2),
         "e": W / sigma2_e ** 2 + _between(s, per_row / lam ** 2)}
    MM = {("u", "u"): _between(s, s.m / lam ** 3), ("u", "e"): _between(s, 1.0 / lam ** 3),
          ("e", "e"): W / sigma2_e ** 3 + _between(s, per_row / lam ** 3)}
    # tr(V⁻¹V_kV⁻¹V_l), over the units that have rows
    units = int((s.m > 0).sum())
    VV = {("u", "u"): float((s.m ** 2 / lam ** 2).sum()), ("u", "e"): float((s.m / lam ** 2).sum()),
          ("e", "e"): (s.N - units) / sigma2_e ** 2 + float((1.0 / lam[s.m > 0] ** 2).sum())}
    # The residuals: r_g their sum on unit g, and Py = V⁻¹r. Then yᵀPV_kV⁻¹V_lPy, and
    # QᵀV⁻¹V_kPy, which for V_e follows from the GLS equations QᵀV⁻¹r = 0 as −γ times V_u's.
    r = s.t - s.S @ at.w
    within = float(np.sum((s.R @ np.append(-at.w, 1.0)) ** 2))
    q = {("u", "u"): float((s.m * r ** 2 / lam ** 3).sum()), ("u", "e"): float((r ** 2 / lam ** 3).sum()),
         ("e", "e"): within / sigma2_e ** 3 + float((r ** 2 * per_row / lam ** 3).sum())}
    v = {"u": s.S.T @ (r / lam ** 2)}
    v["e"] = -gamma * v["u"]
    info = np.zeros((2, 2))
    for i, k in enumerate("ue"):
        for j, l in enumerate("ue"):
            pair = (k, l) if (k, l) in VV else (l, k)
            trace = (VV[pair] - 2.0 * float(np.sum(C * MM[pair]))
                     + float(np.sum((C @ M[k]) * (C @ M[l]).T)))
            info[i, j] = q[pair] - float(v[k] @ C @ v[l]) - 0.5 * trace
    try:
        A = np.linalg.inv(info)
    except np.linalg.LinAlgError:  # no curvature in some direction: no df to give
        return np.full(P, np.nan)
    # ∂C/∂θ_k = C XᵀV⁻¹V_kV⁻¹X C, mapped from Q's basis to X's by T⁻¹.
    grad = np.stack([np.einsum("jk,kl,jl->j", inverse_T, C @ M[k] @ C, inverse_T) for k in "ue"],
                    axis=1)
    f = np.diag(cov)
    var_f = np.einsum("ja,ab,jb->j", grad, A, grad)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(var_f > 0, 2.0 * f ** 2 / var_f, np.nan)


# ── GEE: statsmodels' estimates, the WP2 CR2 engine for the variance ──────────


@dataclass
class GEEFit:
    beta: np.ndarray
    mu: np.ndarray  # fitted means
    alpha: float  # the exchangeable working correlation
    converged: bool


def fit_gee(X: np.ndarray, y: np.ndarray, codes: np.ndarray, task: str) -> GEEFit:
    """GEE with an exchangeable working correlation (identity link for a quantity, logit for a
    binary outcome coded 0/1), by statsmodels."""
    import warnings

    import statsmodels.api as sm

    family = sm.families.Gaussian() if task == "regression" else sm.families.Binomial()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = sm.GEE(np.asarray(y, dtype=float), np.asarray(X, dtype=float), groups=codes,
                       family=family, cov_struct=sm.cov_struct.Exchangeable())
        result = model.fit(maxiter=200, ctol=1e-10)
    converged = not any("converge" in str(w.message).lower() for w in caught)
    alpha = float(np.atleast_1d(result.cov_struct.dep_params)[0])
    return GEEFit(beta=np.asarray(result.params, dtype=float),
                  mu=np.asarray(result.fittedvalues, dtype=float), alpha=alpha, converged=converged)


def gee_working(X: np.ndarray, y: np.ndarray, mu: np.ndarray, codes: np.ndarray, alpha: float,
                task: str) -> tuple[np.ndarray, np.ndarray]:
    """The GEE's working linear model at its estimate: ``W_g = R_g^(−½) A_g^(½) X_g`` and
    ``r_g = R_g^(−½) A_g^(−½)(y_g − μ_g)``, with ``A`` the variance function (1, or μ(1 − μ)) and
    ``R_g = (1 − α)I + αJ``. Then ``WᵀW = Σ D_gᵀV_g⁻¹D_g`` (the GEE's bread, inverted) and
    ``Σ W_gᵀ r_g r_gᵀ W_g`` its meat, so :func:`~turbotab.core.models.inference.cr2` with
    ``adjust=False`` is the GEE sandwich, and with ``adjust=True`` McCaffrey & Bell's BRL (CR2 is
    unchanged by which square root of V⁻¹ whitens a unit, since any two differ by a rotation).
    ``R_g^(−½) = a I + (λ^(−½) − a) J/m`` with ``a = (1 − α)^(−½)`` and ``λ = 1 + (m − 1)α``."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    v = np.ones_like(mu) if task == "regression" else np.clip(mu * (1.0 - mu), 1e-300, None)
    root = np.sqrt(v)
    W0 = X * root[:, None]
    r0 = (y - mu) / root
    G = int(codes.max()) + 1
    m = np.bincount(codes, minlength=G).astype(float)
    lam = 1.0 + (m - 1.0) * alpha
    if alpha >= 1.0 or np.any(lam <= 0):
        raise ValueError(f"The working correlation ({alpha:.3f}) is not a valid correlation for "
                         f"units of these sizes.")
    a = 1.0 / math.sqrt(1.0 - alpha)
    b = (1.0 / np.sqrt(lam) - a) / m  # per unit: R^(−½) = aI + b_g J
    sumW = np.zeros((G, X.shape[1]))
    np.add.at(sumW, codes, W0)
    sumr = np.bincount(codes, weights=r0, minlength=G)
    W = a * W0 + b[codes][:, None] * sumW[codes]
    r = a * r0 + b[codes] * sumr[codes]
    return W, r


# ── the sklearn estimators ───────────────────────────────────────────────────


class RandomInterceptRegressor(RegressorMixin, BaseEstimator):
    """A random-intercept linear mixed model (REML); predicts from the fixed effects alone.

    ``units``: each row's unit label, indexed by row id (set by the fit stage). Without units that
    repeat, the fit is ordinary least squares, which is what the model reduces to. Where units
    repeat and the model matrix's columns are dependent, REML has no fit on them, so it is fit on
    the columns that identify the coefficients in no dependency (:func:`basis`: each dependent
    column that is a combination of those kept before it is left out, as ``lme4::lmer`` drops a
    rank-deficient design's columns), whose predictions are the same whichever such columns are
    left out; each coefficient in a dependency, the intercept too where the constant is in one,
    has no estimate (``unidentified_``, :meth:`Mixed.coefficients`), and a left-out column's
    coefficient is zero in the prediction. The fit is refused in words that name the columns
    (:func:`singular_refusal`) only where working precision cannot tell which to keep. Where
    REML's ratio of the variances lies
    past the search's top (``RandomInterceptFit.beyond``) the fixed effects are the GLS fit's at
    the top, where a unit's mean carries 1/(1 + m_g 10⁸) of the weight it has at γ = 0, so a larger
    ratio barely moves them (on balanced units with unit-level columns, not at all): the
    prediction is served, and only the table, whose intervals rest on the variances, is refused.
    """

    def __init__(self, units: Any = None):
        self.units = units

    def fit(self, X: Any, y: Any) -> "RandomInterceptRegressor":
        M, names, index = _matrix(X)
        y = np.asarray(y, dtype=float)
        design = np.column_stack([np.ones(len(y)), M])
        codes = unit_codes(self.units, index, len(y))
        self.unidentified_: list[str] = []
        if repeats(codes):
            count, columns, kept = basis(design)
            if kept is None:
                raise singular_refusal(pd.DataFrame(M, columns=names))
            if count:
                self.unidentified_ = ["(intercept)" if j == 0 else names[j - 1] for j in columns]
            fit = fit_random_intercept(design[:, kept], y, codes, satterthwaite=False)
            beta = np.zeros(design.shape[1])
            beta[kept] = fit.beta
            self.sigma2_u_, self.sigma2_e_ = fit.sigma2_u, fit.sigma2_e
        else:
            beta = np.linalg.lstsq(design, y, rcond=None)[0]
            self.sigma2_u_, self.sigma2_e_ = 0.0, float(np.var(y - design @ beta))
        self.intercept_, self.coef_ = float(beta[0]), beta[1:]
        self.feature_names_in_ = np.asarray(names, dtype=object)
        self.n_features_in_ = len(names)
        return self

    def predict(self, X: Any) -> np.ndarray:
        return self.intercept_ + _matrix(X)[0] @ self.coef_


class GEERegressor(RegressorMixin, BaseEstimator):
    """GEE with an exchangeable working correlation, identity link; population-average predictions."""

    def __init__(self, units: Any = None):
        self.units = units

    def fit(self, X: Any, y: Any) -> "GEERegressor":
        M, names, index = _matrix(X)
        y = np.asarray(y, dtype=float)
        design = np.column_stack([np.ones(len(y)), M])
        codes = unit_codes(self.units, index, len(y))
        if repeats(codes):
            fit = fit_gee(design, y, codes, "regression")
            beta, self.alpha_ = fit.beta, fit.alpha
        else:
            beta, self.alpha_ = np.linalg.lstsq(design, y, rcond=None)[0], 0.0
        self.intercept_, self.coef_ = float(beta[0]), beta[1:]
        self.feature_names_in_ = np.asarray(names, dtype=object)
        self.n_features_in_ = len(names)
        return self

    def predict(self, X: Any) -> np.ndarray:
        return self.intercept_ + _matrix(X)[0] @ self.coef_


class GEEClassifier(ClassifierMixin, BaseEstimator):
    """GEE with an exchangeable working correlation, logit link, for a two-class outcome."""

    def __init__(self, units: Any = None):
        self.units = units

    def fit(self, X: Any, y: Any) -> "GEEClassifier":
        import warnings

        import statsmodels.api as sm

        M, names, index = _matrix(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError(f"GEE here models two classes; these rows have {len(self.classes_)}.")
        event = (y == self.classes_[1]).astype(float)
        design = np.column_stack([np.ones(len(y)), M])
        codes = unit_codes(self.units, index, len(y))
        if repeats(codes):
            fit = fit_gee(design, event, codes, "binary")
            beta, self.alpha_ = fit.beta, fit.alpha
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                beta = np.asarray(sm.GLM(event, design, family=sm.families.Binomial()).fit().params)
            self.alpha_ = 0.0
        self.intercept_, self.coef_ = np.array([beta[0]]), beta[None, 1:]
        self.feature_names_in_ = np.asarray(names, dtype=object)
        self.n_features_in_ = len(names)
        return self

    def decision_function(self, X: Any) -> np.ndarray:
        return self.intercept_[0] + _matrix(X)[0] @ self.coef_[0]

    def predict_proba(self, X: Any) -> np.ndarray:
        eta = self.decision_function(X)
        p = 0.5 * (1.0 + np.tanh(0.5 * eta))
        return np.column_stack([1.0 - p, p])

    def predict(self, X: Any) -> np.ndarray:
        return self.classes_[(self.predict_proba(X)[:, 1] > 0.5).astype(int)]


# ── the inference tables ─────────────────────────────────────────────────────


def _design(matrix: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    import statsmodels.api as sm

    exog = sm.add_constant(matrix.astype(float), has_constant="add")
    return exog, ["(intercept)" if c == "const" else str(c) for c in exog.columns]


def _no_units(names: Sequence[str], est: np.ndarray, estimator: str, clusters: Any,
              family_label: str) -> Any:
    """The table when no unit repeats: the family has nothing to model beyond the linear one."""
    from turbotab.core.models.inference import _refused

    if clusters.refusal:
        return _refused(names, est, estimator, clusters, clusters.refusal, clusters.exits)
    if clusters.clustered and clusters.n_clusters < 2:
        reason = (f"Every row is in one `{clusters.column}` unit, so there is no between-unit "
                  f"variation for {family_label} to describe.")
    else:
        reason = (f"No identifier repeats in these rows, so there is no unit for {family_label} "
                  f"to describe; the linear model's table is the one to read.")
    return _refused(names, est, estimator, clusters, reason,
                    ({"label": "Use the linear model", "decision": {"kind": "select_models",
                                                                     "models": ["linear"]}},))


def _singular(matrix: pd.DataFrame) -> str | None:
    from turbotab.core.models.linear import collinearity_concern

    try:
        return collinearity_concern(matrix)
    except Exception:  # noqa: BLE001 - a diagnostic that cannot run is not a verdict
        return None


def dependency_words(matrix: pd.DataFrame, *, beside: str = "the intercept"
                     ) -> tuple[str, str] | None:
    """What is dependent in ``matrix`` (its columns without the intercept) with the intercept beside
    it, to working precision (:func:`dependencies`), and how to undo it; None where nothing is.
    The columns named are those in a dependency and the count is the dependencies'; no condition
    number is quoted, since at an exact dependency the one computed is rounding. ``beside`` names
    the constant the columns are combined with (the intercept, or, for a model whose cut-points
    take its place, a constant)."""
    from turbotab.core.models.inference import _terms

    X = np.column_stack([np.ones(len(matrix)), matrix.to_numpy(dtype=float)])
    count, columns = dependencies(X)
    if count == 0:
        return None
    named = [str(matrix.columns[j - 1]) for j in columns if j > 0]
    ways = "" if count <= 1 else f", in {count:,} ways,"
    if len(named) == 1:
        what = (f"{_terms(named)} is a fixed combination of the other columns and {beside}, so "
                f"its coefficient is not identified")
    elif named:
        what = (f"{_terms(named)} are{ways} fixed combinations of the other columns and "
                f"{beside}, so their coefficients are not identified")
    else:
        what = (f"its columns and {beside} are dependent, so their coefficients are not "
                f"identified")
    leave = ("leave one of the dependent columns out" if count <= 1 else
             f"leave out one dependent column for each of the {count:,} ways")
    return what, leave


def singular_words(matrix: pd.DataFrame) -> str:
    """Why REML has no fit on ``matrix`` (its columns without the intercept), where its columns,
    with the intercept, are dependent to working precision (:func:`dependent`), and REML needs
    every coefficient identified, since its likelihood holds ``log|XᵀV⁻¹X|``: in the words of
    :func:`dependency_words`."""
    what, leave = dependency_words(matrix) or (
        "its columns and the intercept are dependent, so their coefficients are not identified",
        "leave one of the dependent columns out")
    return (f"On these {len(matrix):,} rows the model matrix is singular: {what}. The "
            f"random-intercept mixed model is fit by REML, which needs every coefficient "
            f"identified, so it is not fit; {leave}.")


def singular_refusal(matrix: pd.DataFrame) -> Any:
    """The mixed model's fit refused where REML has none (:func:`singular_words`), typed as a
    fit-time refusal (SWARM_PLAN §8)."""
    from turbotab.core.graph import StageRefusal

    return StageRefusal("singular_model_matrix", singular_words(matrix), exits=[
        {"label": "Leave one of the dependent columns out of the model", "decision": None}])


def beyond_words(column: str, n_units: int) -> str:
    """Why the mixed model's table is refused where REML's γ̂ lies beyond the grid's top
    (:func:`_best_gamma`): its intervals rest on the two variances, which the fit there does not
    estimate."""
    top = GAMMA_GRID[-1]
    return (f"The random-intercept model's REML likelihood by `{column}` on these {n_units:,} "
            f"units is still rising where the between-unit variance is {top:,.0f} times the "
            f"within-unit variance, the most the fit searches: within a `{column}` unit the "
            f"outcome varies so little beyond what the predictors explain that REML puts the "
            f"within-unit SD below {1 / math.sqrt(top):.3g} times the between-unit SD. The fit "
            f"does not estimate variances that far apart, so no interval or p-value is reported.")


def beyond_exits(column: str) -> tuple[dict[str, Any], ...]:
    return ({"label": f"Combine each `{column}`'s rows into one (the unit question, before the "
                      f"seal)", "decision": None},
            {"label": "Use the linear model", "decision": {"kind": "select_models",
                                                           "models": ["linear"]}})


MIXED_ESTIMATOR = "random-intercept linear mixed model (REML)"


def mixed_table(matrix: pd.DataFrame, y: Any, clusters: Any, *,
                study: Sequence[str] | None = None) -> Any:
    """The random-intercept model's coefficient table on the matrix the model saw, by the
    clusters :func:`~turbotab.core.models.inference.resolve_clusters` read from the rows; on no
    more rows than its coefficients, refused with no estimate, before any other refusal that
    would keep one (``inference._no_spare_rows``). Fit on its columns as they are (REML's QR is
    unmoved by a column's unit or origin), and where its columns are dependent at R's tolerance
    (:func:`aliased`), REML having no fit on an exact dependency (:func:`dependent`), on the
    columns that identify the coefficients in no dependency, each one in a dependency refused by
    row, before any other refusal: the whole table where ``study`` is in one or is None
    (``inference.by_identified``). Refused, its estimates kept, where no unit repeats, where the
    matrix is nearly singular (the linear family's concern, on the columns centered), where
    REML's ratio of the variances lies beyond what the fit searches (:func:`beyond_words`), and
    where the fit gives some coefficient no standard error or Satterthwaite degrees of freedom
    (``inference.no_interval``)."""
    from turbotab.core.models.inference import _no_spare_rows, _nothing, by_identified

    spare = _no_spare_rows("regression", len(matrix), matrix.shape[1] + 1, clusters,
                           estimator=MIXED_ESTIMATOR)
    if spare is not None:
        return spare
    # Its columns are not scaled: REML factors the design by QR, whose triangle carries a column's
    # unit, so the fit is already invariant to it (to 10⁻¹⁰ with a date in epoch seconds), and
    # scaling would only round γ̂ again, which the profile on 50,000-row units fixes to some 10⁻⁶.
    return by_identified(matrix, lambda m: _mixed_table(m, y, clusters), layout="intercept",
                         study=study, scaled=False,
                         refuse=lambda reason: _nothing(MIXED_ESTIMATOR, clusters, reason))


def _mixed_table(matrix: pd.DataFrame, y: Any, clusters: Any) -> Any:
    """:func:`mixed_table` on columns that are not dependent."""
    from turbotab.core.models.inference import (FEW_CLUSTERS, InferenceTable, _info, _refused,
                                                _t_rows, no_interval)

    exog, names = _design(matrix)
    X = exog.to_numpy(dtype=float)
    y = np.asarray(y, dtype=float)
    estimator = MIXED_ESTIMATOR
    # Least squares on the columns scaled to unit length: its columns are not scaled for REML
    # (:func:`mixed_table`), and on them a date in epoch seconds beside the intercept fell below
    # lstsq's cutoff, so the estimates a refusal keeps were the pseudo-inverse's.
    norms = np.linalg.norm(X, axis=0)
    norms = np.where(norms > 0, norms, 1.0)
    ols = np.linalg.lstsq(X / norms, y, rcond=None)[0] / norms
    if not clusters.clustered or clusters.n_clusters < 2:
        return _no_units(names, ols, estimator, clusters, "a random intercept")
    singular = _singular(matrix)
    if singular:
        return _refused(names, ols, estimator, clusters, singular, ())
    if len(clusters.codes) != len(y):
        raise ValueError(f"The clusters cover {len(clusters.codes):,} rows but the model matrix "
                         f"has {len(y):,}.")
    try:
        fit = fit_random_intercept(X, y, clusters.codes)
    except DependentColumns:  # dependent columns the concern did not name (``dependent``)
        return _refused(names, ols, estimator, clusters, singular_words(matrix), ())
    se = np.sqrt(np.clip(np.diag(fit.cov), 0, None))
    G = fit.n_units
    column = clusters.column
    if fit.beyond:
        return _refused(names, fit.beta, estimator, clusters, beyond_words(column, G),
                        beyond_exits(column))
    reason = no_interval(names, se, fit.df, fit=f"The random-intercept model's REML fit by "
                                                f"`{column}` on these {G:,} units",
                         df_kind="Satterthwaite")
    if reason:
        return _refused(names, fit.beta, estimator, clusters, reason, ())
    rows = _t_rows(names, fit.beta, se, fit.df)
    missing = (f", {clusters.n_missing:,} rows with no `{column}` counted as units of their own"
               if clusters.n_missing else "")
    caption = (f"95% intervals from a random-intercept model by `{column}` fit by REML, G = {G:,} "
               f"units{missing}, on t with Satterthwaite degrees of freedom; between-unit SD "
               f"{math.sqrt(fit.sigma2_u):.3g}, within-unit SD {math.sqrt(fit.sigma2_e):.3g}.")
    concerns = [clusters.note] if clusters.note else []
    if fit.boundary:
        concerns.append(f"The between-`{column}` variance is estimated at zero, so the random "
                        f"intercept adds nothing: the estimates are least squares, and the "
                        f"intervals treat the rows as independent.")
    elif G < FEW_CLUSTERS:
        concerns.append(f"Only {G} `{column}` units: the between-unit variance rests on {G}, and "
                        f"the Satterthwaite degrees of freedom widen the intervals to allow for "
                        f"it (Luke 2017). The intervals assume each unit's own level is unrelated "
                        f"to the predictors.")
    return InferenceTable(rows, _info(estimator, "model", caption, clusters), concerns)


def gee_table(matrix: pd.DataFrame, y: Any, clusters: Any, task: str,
              classes: Sequence[Any] | None = None, *,
              study: Sequence[str] | None = None) -> Any:
    """The GEE's coefficient table: statsmodels' estimates, CR2 (BRL) intervals with Bell–McCaffrey
    degrees of freedom on its whitened working model, refused below the unit floor; on no more rows
    than its coefficients, refused with no estimate, before any other refusal that would keep one
    (``inference._no_spare_rows``). Fit on its columns centered and scaled, its slopes scaled back
    and its intercept mapped back to the columns' origin, and where its columns are dependent, on
    the columns that identify the coefficients in no dependency, each one in a dependency refused
    by row, before any other refusal: the whole table where ``study`` is in one or is None
    (``inference.by_identified``). Refused where no unit repeats, where the matrix is nearly
    singular (the linear family's concern) and where a column separates a binary outcome, before
    the fit: the identity link's table keeps the least-squares estimates (its own where no unit
    repeats), and the logit link's keeps none (least squares on a 0/1 outcome is no log-odds)."""
    from turbotab.core.models.inference import _no_spare_rows, _nothing, by_identified

    if task == "binary":
        if classes is None or len(classes) != 2:
            raise ValueError("A binary outcome needs exactly two classes.")
        y = (np.asarray(y) == classes[1]).astype(float)
    else:
        y = np.asarray(y, dtype=float)
    estimator = ("GEE, exchangeable working correlation" +
                 (" (logit link)" if task == "binary" else " (identity link)"))
    spare = _no_spare_rows(task, len(matrix), matrix.shape[1] + 1, clusters, estimator=estimator)
    if spare is not None:
        return spare
    return by_identified(matrix, lambda m: _gee_table(m, y, clusters, task, estimator),
                         layout="intercept", study=study,
                         refuse=lambda reason: _nothing(estimator, clusters, reason))


def _gee_table(matrix: pd.DataFrame, y: np.ndarray, clusters: Any, task: str,
               estimator: str) -> Any:
    """:func:`gee_table` on columns that are not dependent, ``y`` coded 0/1 for a binary outcome."""
    from turbotab.core.models.inference import (_and, _clustered, _named, _refused, floor_refusal,
                                                separated_columns)

    exog, names = _design(matrix)
    X = exog.to_numpy(dtype=float)
    # What a refusal before the fit keeps: the identity link's least-squares estimates (its own
    # where no unit repeats); on a 0/1 outcome least squares is no log-odds, so the logit link's
    # table keeps none, where it kept them and called them log-odds.
    start = (np.linalg.lstsq(X, y, rcond=None)[0] if task == "regression" else
             np.full(X.shape[1], np.nan))
    if not clusters.clustered or clusters.n_clusters < 2:
        return _no_units(names, start, estimator, clusters, "an exchangeable working correlation")
    singular = _singular(matrix)
    if singular:
        return _refused(names, start, estimator, clusters, singular, ())
    if task == "binary":
        separated = _named(names, separated_columns(X, y))
        if separated:
            verb = "separates" if len(separated) == 1 else "separate"
            reason = (f"{_and(separated)} {verb} the outcome, so the GEE's estimate is infinite "
                      f"and no interval or p-value can be reported.")
            return _refused(names, start, estimator, clusters, reason,
                            ({"label": f"Leave {_and(separated)} out, or merge its rare levels",
                              "decision": None},))
    fit = fit_gee(X, y, clusters.codes, task)
    refusal = floor_refusal(clusters, task=task)
    if refusal:
        return _refused(names, fit.beta, estimator, clusters, *refusal)
    W, r = gee_working(X, y, fit.mu, clusters.codes, fit.alpha, task)
    table = _clustered(names, fit.beta, estimator, W, r, clusters.codes, clusters)
    if table.info["refused"]:  # no standard error from its sandwich (``inference.no_interval``)
        return table
    table.info["caption"] = (table.info["caption"].rstrip(".") +
                             f"; exchangeable working correlation {fit.alpha:.2f}.")
    if not fit.converged:
        table.concerns.append("The GEE fit stopped before converging; treat these numbers with care.")
    return table


# ── the families ─────────────────────────────────────────────────────────────


def _prediction_concern(label: str) -> str:
    return (f"For prediction, {label} predicts a new unit from its population-level effects, much "
            f"as the linear model does; it earns its place in inference.")


class Mixed(FamilyBase):
    key = "mixed"
    label = "Random-intercept mixed model"
    tasks: tuple[Task, ...] = ("regression",)
    inductive_bias = ("Straight-line effects, plus a level of its own for each unit whose rows "
                      "repeat.")
    strengths = (
        "Intervals that allow for a unit's rows being alike, even with few units.",
        "Uses variation within and between units, weighted by how alike a unit's rows are.",
    )
    cautions = (
        "Assumes each unit's own level is unrelated to the predictors; otherwise within- and "
        "between-unit effects are blended.",
        "Adds nothing when no unit repeats, and predicts a new unit from its fixed effects alone.",
    )
    needs_scaling = False
    handles_missing = False
    # MODEL_FAMILY_CONTRACT §1 (§3.1's row for it). It declares no attribution yet: MC-7 draws
    # curves for every predicting family, and until then the explanations do not reach it.
    identity = Identity(kind="estimator", library="turbotab", estimator="RandomInterceptRegressor")
    purposes = ("prediction", "inference")
    predicts = True
    flexible = False
    bootstrap_optimism = True
    inference_decl = InferenceDecl(table="intervals", intervals=("Satterthwaite",),
                                   matrix_table=True, words="the random-intercept mixed model")
    invariances = ("linear_maps",)  # its fixed effects, with random intercepts
    curve_shape = "straight"
    diagnostics = ("boundary",)
    raw_scale = {"regression": "value"}
    review_lenses = ("shared",)
    consequence = ("A random intercept per unit: model-based intervals when rows repeat, even within "
                   "few units.")
    card_label = "Mixed model"

    def methods_label(self, task: Task | None) -> str:
        return "a random-intercept mixed model"

    def build(self, task: Task, purpose: Purpose | None, n_rows: int, n_features: int) -> Any:
        return RandomInterceptRegressor()

    def describe(self, task: Task, purpose: Purpose | None) -> tuple[str, str]:
        detail = "Fits straight-line effects with an intercept of its own for each unit, by REML."
        if purpose == "inference":
            detail += " Intervals are on t with Satterthwaite degrees of freedom."
        return "Random-intercept mixed model", detail

    def coefficients(self, pipeline: Any, X: Any, y: Any, *, task: Task,
                     purpose: Purpose | None, groups: Any = None) -> list[dict[str, Any]] | None:
        """The fit's fixed effects, each in a dependency of the model matrix's columns with no
        estimate (``RandomInterceptRegressor.unidentified_``)."""
        model = pipeline[-1]
        rows = coefficient_rows([str(f) for f in model.feature_names_in_], model.coef_,
                                intercept=model.intercept_)
        unidentified = set(getattr(model, "unidentified_", []))
        for row in rows:
            if row["feature"] in unidentified:
                row["estimate"] = None
        return rows

    def inference(self, pipeline: Any, X: Any, y: Any, *, task: Task, clusters: Any,
                  outcome: Any = None, rows: Any = None, study: Any = None) -> Any:
        """The mixed model's table, its scale declared (a difference in the mean outcome; WP8);
        ``study`` names the model matrix's columns of what is studied
        (``inference.by_identified``)."""
        from turbotab.core.models.inference import _on_rows, _on_scale
        from turbotab.core.models.linear import model_matrix

        table = mixed_table(model_matrix(pipeline, X), y, clusters, study=study)
        if not table.rows:  # refused with no estimate (``mixed_table``): on no scale, its rows said
            table.info.update(n_rows=int(len(X)), rows=rows)
            return table
        return _on_rows(_on_scale(table, task, None, outcome), len(X), rows)

    def assess(self, s: Situation) -> Assessment:
        from turbotab.core.models.inference import FEW_CLUSTERS

        if s.n_features >= s.n_rows:
            return Assessment(0.0, "poor", (f"{s.n_features:,} predictors for {s.n_rows:,} rows: "
                                            f"the model has no unique solution.",))
        units = getattr(s, "n_units", None)
        if units is None:
            return Assessment(0.25, "poor", ("No identifier repeats in these rows, so there is no "
                                             "unit for a random intercept; it is the linear "
                                             "model's fit.",))
        if s.purpose != "inference":
            return Assessment(1.0, "fair", (_prediction_concern("the mixed model"),))
        concerns: list[str] = []
        fit = "good"
        if units < FEW_UNITS_MIXED:
            concerns.append(f"Rows repeat within only {units} units: the between-unit variance "
                            f"rests on {units}, so its intervals are wide.")
            fit = "fair"
        # Few units: the model-based intervals do not rest on unit-level residuals, so it leads.
        score = 3.5 if units < FEW_CLUSTERS else 2.5
        if s.n_rows < 10 * s.n_features:
            concerns.append(f"{s.n_rows:,} rows for {s.n_features:,} predictors: unpenalized "
                            f"estimates will be unstable.")
            fit, score = "fair", score - 1.0
        return Assessment(score, fit, tuple(concerns))


class GEE(FamilyBase):
    key = "gee"
    label = "GEE (exchangeable)"
    tasks: tuple[Task, ...] = ("regression", "binary")
    inductive_bias = ("Population-average straight-line or log-odds effects; a unit's rows share "
                      "one correlation.")
    strengths = (
        "Population-average effects, with intervals robust to how a unit's rows correlate.",
        "Weights rows by their estimated within-unit correlation, which can sharpen estimates.",
    )
    cautions = (
        "Its intervals rest on unit-level residuals, so they are refused below the unit floor.",
        "Adds nothing when no unit repeats, and predicts from population-average effects.",
    )
    needs_scaling = False
    handles_missing = False
    # MODEL_FAMILY_CONTRACT §1 (§3.1's row for it). No attribution yet, as the mixed model's.
    identity = Identity(kind="estimator", library="statsmodels",
                        estimator="GEERegressor; GEEClassifier (exchangeable)")
    purposes = ("prediction", "inference")
    predicts = True
    flexible = False
    bootstrap_optimism = True
    inference_decl = InferenceDecl(table="intervals", intervals=("CR2",), matrix_table=True,
                                   words="the GEE model")
    invariances = ("linear_maps",)
    curve_shape = "straight"
    diagnostics = ("convergence",)
    output = "margin"
    raw_scale = {"regression": "value", "binary": "margin"}
    review_lenses = ("shared",)
    consequence = ("Population-average effects, with intervals robust to how a unit's repeated rows "
                   "correlate.")
    card_label = "GEE"

    def methods_label(self, task: Task | None) -> str:
        return "generalized estimating equations"

    def build(self, task: Task, purpose: Purpose | None, n_rows: int, n_features: int) -> Any:
        return GEERegressor() if task == "regression" else GEEClassifier()

    def describe(self, task: Task, purpose: Purpose | None) -> tuple[str, str]:
        link = "straight-line" if task == "regression" else "log-odds"
        detail = (f"Fits population-average {link} effects, with an exchangeable correlation "
                  f"among each unit's rows.")
        if purpose == "inference":
            detail += " Intervals are CR2 cluster-robust with Bell–McCaffrey degrees of freedom."
        return "Generalized estimating equations", detail

    def coefficients(self, pipeline: Any, X: Any, y: Any, *, task: Task,
                     purpose: Purpose | None, groups: Any = None) -> list[dict[str, Any]] | None:
        model = pipeline[-1]
        return coefficient_rows([str(f) for f in model.feature_names_in_], model.coef_,
                                intercept=model.intercept_)

    def inference(self, pipeline: Any, X: Any, y: Any, *, task: Task, clusters: Any,
                  outcome: Any = None, rows: Any = None, study: Any = None) -> Any:
        """The GEE table on the outcome's scale (WP8's ``_on_scale``, closing ME-07 for this
        family): log-odds with odds ratios, the event and reference named, for a yes/no outcome;
        a difference in the mean for a number. ``study`` names the model matrix's columns of what
        is studied (``inference.by_identified``)."""
        from turbotab.core.models.inference import _on_rows, _on_scale
        from turbotab.core.models.linear import model_matrix

        classes = list(getattr(pipeline[-1], "classes_", [])) or None
        table = gee_table(model_matrix(pipeline, X), y, clusters, task, classes, study=study)
        if not table.rows:  # refused with no estimate (``gee_table``): on no scale, its rows said
            table.info.update(n_rows=int(len(X)), rows=rows)
            return table
        return _on_rows(_on_scale(table, task, classes, outcome), len(X), rows)

    def assess(self, s: Situation) -> Assessment:
        from turbotab.core.models.inference import FEW_CLUSTERS, min_clusters

        if s.n_features >= s.n_rows:
            return Assessment(0.0, "poor", (f"{s.n_features:,} predictors for {s.n_rows:,} rows: "
                                            f"the model has no unique solution.",))
        units = getattr(s, "n_units", None)
        if units is None:
            return Assessment(0.25, "poor", ("No identifier repeats in these rows, so there is no "
                                             "within-unit correlation to model; it is the linear "
                                             "model's fit.",))
        if s.purpose != "inference":
            return Assessment(1.0, "fair", (_prediction_concern("GEE"),))
        concerns: list[str] = []
        if units < min_clusters():
            other = ("a random-intercept mixed model can" if s.task == "regression"
                     else "combining each unit's rows can")
            return Assessment(0.5, "poor", (f"Rows repeat within only {units} units, fewer than "
                                            f"the {min_clusters()} a sandwich interval needs; "
                                            f"{other}.",))
        fit, score = ("fair", 2.0) if units < FEW_CLUSTERS else ("good", 2.75)
        if units < FEW_CLUSTERS:
            concerns.append(f"{units} units: the intervals use the CR2 small-sample correction with "
                            f"Bell–McCaffrey degrees of freedom (McCaffrey & Bell 2006).")
        if s.task == "binary" and s.n_events is not None and s.n_features:
            epv = s.n_events / s.n_features
            if epv < EPV_RULE:
                concerns.append(f"{s.n_events:,} events for {s.n_features:,} predictors ({epv:.1f} "
                                f"each); a common rule of thumb asks for {EPV_RULE}.")
                fit, score = ("poor", 0.5) if epv < EPV_RULE / 2 else ("fair", score - 1.0)
        return Assessment(score, fit, tuple(concerns))


MIXED = register_family(Mixed())
GEE_FAMILY = register_family(GEE())

__all__ = [
    "GEE", "GEEClassifier", "GEEFit", "GEERegressor", "GEE_FAMILY", "MIXED", "Mixed",
    "RandomInterceptFit", "RandomInterceptRegressor", "fit_gee", "fit_random_intercept",
    "gee_table", "gee_working", "mixed_table", "reml_loglik", "repeats", "satterthwaite_df",
    "unit_codes",
]
