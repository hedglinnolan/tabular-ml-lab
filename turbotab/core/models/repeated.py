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
an intercept): with ``V_g = σ²_e I + σ²_u J``, every quantity is a sum over units of each unit's
column sums, so a fit costs a few passes over the units whatever their sizes. The GEE estimates
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
from turbotab.core.models.base import Assessment, FamilyBase, Situation, coefficient_rows, register_family

# The REML profile over γ = σ²_u / σ²_e is searched on this log10 grid, then refined by Brent's
# method around the best grid point; γ = 0 (no between-unit variance) is checked on its own.
GAMMA_GRID = tuple(10.0 ** k for k in range(-8, 9))
HESSIAN_STEP = 1e-4  # central differences of the REML log-likelihood, in log variance units
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
    """Each unit's sufficient statistics: what every REML quantity below is a function of."""

    XtX: np.ndarray  # (P, P)
    Xty: np.ndarray  # (P,)
    yty: float
    S: np.ndarray  # (G, P): each unit's column sums
    t: np.ndarray  # (G,): each unit's outcome sum
    m: np.ndarray  # (G,): each unit's row count
    N: int
    P: int


def _sums(X: np.ndarray, y: np.ndarray, codes: np.ndarray) -> _Sums:
    G = int(codes.max()) + 1
    S = np.zeros((G, X.shape[1]))
    np.add.at(S, codes, X)
    t = np.bincount(codes, weights=y, minlength=G)
    m = np.bincount(codes, minlength=G).astype(float)
    return _Sums(XtX=X.T @ X, Xty=X.T @ y, yty=float(y @ y), S=S, t=t, m=m, N=len(y), P=X.shape[1])


def _at(gamma: float, s: _Sums) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """For γ = σ²_u/σ²_e and ``H_g = I + γJ``: ``A = XᵀH⁻¹X``, the GLS ``β``, the residual sum of
    squares ``(y − Xβ)ᵀH⁻¹(y − Xβ)``, ``Σ log(1 + m_g γ)`` and ``log |A|``."""
    c = gamma / (1.0 + s.m * gamma)  # H_g⁻¹ = I − c_g J
    A = s.XtX - (s.S * c[:, None]).T @ s.S
    b = s.Xty - s.S.T @ (c * s.t)
    yHy = s.yty - float(c @ (s.t ** 2))
    A = (A + A.T) / 2
    beta = np.linalg.solve(A, b)
    rss = max(yHy - float(beta @ b), 1e-300)
    sign, logdet = np.linalg.slogdet(A)
    if sign <= 0:
        raise np.linalg.LinAlgError("the model matrix is singular")
    return A, beta, rss, float(np.log1p(s.m * gamma).sum()), float(logdet)


def _profile(gamma: float, s: _Sums) -> float:
    """−2 × the REML log-likelihood with σ²_e profiled out, up to a constant."""
    _, _, rss, logH, logA = _at(gamma, s)
    nu = s.N - s.P
    return nu * math.log(rss / nu) + logH + logA


def reml_loglik(sigma2_u: float, sigma2_e: float, s: _Sums) -> float:
    """The REML log-likelihood ``−½[log|V| + log|XᵀV⁻¹X| + rᵀV⁻¹r]`` (no constant), with
    ``V = σ²_e H(γ)``: ``−½[(N − P) log σ²_e + Σ log(1 + m_g γ) + log|A| + RSS/σ²_e]``."""
    gamma = sigma2_u / sigma2_e
    _, _, rss, logH, logA = _at(gamma, s)
    return -0.5 * ((s.N - s.P) * math.log(sigma2_e) + logH + logA + rss / sigma2_e)


def _best_gamma(s: _Sums) -> float:
    from scipy.optimize import minimize_scalar

    values = [(_profile(g, s), g) for g in GAMMA_GRID]
    best, gamma = min(values)
    at_zero = _profile(0.0, s)
    i = GAMMA_GRID.index(gamma)
    if i == 0 and at_zero <= best:
        # The profile still falls toward γ = 0: refine on [0, the first grid point], where the
        # minimum may be at the boundary itself.
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

    @property
    def icc(self) -> float:
        total = self.sigma2_u + self.sigma2_e
        return self.sigma2_u / total if total > 0 else float("nan")


def fit_random_intercept(X: np.ndarray, y: np.ndarray, codes: np.ndarray, *,
                         satterthwaite: bool = True) -> RandomInterceptFit:
    """REML for ``y = Xβ + u_g + e``, ``u_g ~ N(0, σ²_u)``, ``e ~ N(0, σ²_e)``.

    γ = σ²_u/σ²_e is found by minimizing the profile (:func:`_profile`); then σ̂²_e = RSS/(N − P)
    and ``C = σ̂²_e A⁻¹``. With ``satterthwaite``, each coefficient's df are
    ``2 f² / (∇fᵀ 𝒜 ∇f)`` with ``f = C_jj``; ``∇f`` is analytic (``∂C/∂θ_k = C XᵀV⁻¹(∂V/∂θ_k)V⁻¹X
    C``) and ``𝒜`` is the inverse of the negative Hessian of :func:`reml_loglik`, by central
    differences in log σ² (the df are the same in any parameterization at the optimum). At the
    boundary σ̂²_u = 0 the between-unit variance is held at zero, and the df are N − P.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    codes = np.asarray(codes, dtype=np.int64)
    s = _sums(X, y, codes)
    if s.N <= s.P:
        raise ValueError(f"{s.N:,} rows for {s.P:,} coefficients: the model cannot be fit.")
    gamma = _best_gamma(s)
    A, beta, rss, _, _ = _at(gamma, s)
    sigma2_e = rss / (s.N - s.P)
    sigma2_u = gamma * sigma2_e
    cov = sigma2_e * np.linalg.inv(A)
    cov = (cov + cov.T) / 2
    boundary = gamma == 0.0
    df = satterthwaite_df(s, sigma2_u, sigma2_e, cov, boundary) if satterthwaite else None
    return RandomInterceptFit(beta=beta, cov=cov, sigma2_u=sigma2_u, sigma2_e=sigma2_e, df=df,
                              loglik=reml_loglik(sigma2_u, sigma2_e, s) if sigma2_e > 0 else float("nan"),
                              boundary=boundary, n_units=len(s.m))


def satterthwaite_df(s: _Sums, sigma2_u: float, sigma2_e: float, cov: np.ndarray,
                     boundary: bool) -> np.ndarray:
    """Each coefficient's Satterthwaite df (see :func:`fit_random_intercept`)."""
    denom = sigma2_e + s.m * sigma2_u
    c = sigma2_u / denom
    a = s.S / denom[:, None]  # X_gᵀ V_g⁻¹ 1
    G_u = a.T @ a  # Σ X_gᵀV_g⁻¹ J V_g⁻¹X_g
    G_e = (s.XtX - (s.S * (2 * c - c * c * s.m)[:, None]).T @ s.S) / sigma2_e ** 2  # Σ X_gᵀV_g⁻²X_g
    dC_u = cov @ G_u @ cov
    dC_e = cov @ G_e @ cov
    f = np.diag(cov)
    if boundary:
        # σ²_u held at zero: only σ²_e varies, and −ℓ'' = (N − P)/(2σ⁴) at its estimate.
        var_e = 2.0 * sigma2_e ** 2 / (s.N - s.P)
        with np.errstate(divide="ignore", invalid="ignore"):
            return 2.0 * f ** 2 / (np.diag(dC_e) ** 2 * var_e)
    theta = np.array([sigma2_u, sigma2_e])
    H = _hessian(lambda phi: reml_loglik(math.exp(phi[0]), math.exp(phi[1]), s), np.log(theta))
    info = -H
    try:
        A = np.linalg.inv(info)
    except np.linalg.LinAlgError:
        A = np.linalg.pinv(info)
    grad = np.stack([np.diag(dC_u), np.diag(dC_e)], axis=1) * theta[None, :]  # ∂f/∂ log θ
    var_f = np.einsum("ja,ab,jb->j", grad, A, grad)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(var_f > 0, 2.0 * f ** 2 / var_f, np.nan)


def _hessian(fn: Any, x: np.ndarray, h: float = HESSIAN_STEP) -> np.ndarray:
    """The Hessian of ``fn`` at ``x`` by central differences."""
    k = len(x)
    H = np.zeros((k, k))
    f0 = fn(x)
    for i in range(k):
        e_i = np.zeros(k)
        e_i[i] = h
        H[i, i] = (fn(x + e_i) - 2 * f0 + fn(x - e_i)) / h ** 2
        for j in range(i + 1, k):
            e_j = np.zeros(k)
            e_j[j] = h
            H[i, j] = H[j, i] = (fn(x + e_i + e_j) - fn(x + e_i - e_j) - fn(x - e_i + e_j)
                                 + fn(x - e_i - e_j)) / (4 * h * h)
    return H


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
    repeat, the fit is ordinary least squares, which is what the model reduces to.
    """

    def __init__(self, units: Any = None):
        self.units = units

    def fit(self, X: Any, y: Any) -> "RandomInterceptRegressor":
        M, names, index = _matrix(X)
        y = np.asarray(y, dtype=float)
        design = np.column_stack([np.ones(len(y)), M])
        codes = unit_codes(self.units, index, len(y))
        if repeats(codes):
            fit = fit_random_intercept(design, y, codes, satterthwaite=False)
            beta, self.sigma2_u_, self.sigma2_e_ = fit.beta, fit.sigma2_u, fit.sigma2_e
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


def mixed_table(matrix: pd.DataFrame, y: Any, clusters: Any) -> Any:
    """The random-intercept model's coefficient table on the matrix the model saw, by the
    clusters :func:`~turbotab.core.models.inference.resolve_clusters` read from the rows."""
    from turbotab.core.models.inference import FEW_CLUSTERS, InferenceTable, _info, _refused, _t_rows

    exog, names = _design(matrix)
    X = exog.to_numpy(dtype=float)
    y = np.asarray(y, dtype=float)
    estimator = "random-intercept linear mixed model (REML)"
    ols = np.linalg.lstsq(X, y, rcond=None)[0]
    if not clusters.clustered or clusters.n_clusters < 2:
        return _no_units(names, ols, estimator, clusters, "a random intercept")
    singular = _singular(matrix)
    if singular:
        return _refused(names, ols, estimator, clusters, singular, ())
    if len(clusters.codes) != len(y):
        raise ValueError(f"The clusters cover {len(clusters.codes):,} rows but the model matrix "
                         f"has {len(y):,}.")
    fit = fit_random_intercept(X, y, clusters.codes)
    se = np.sqrt(np.clip(np.diag(fit.cov), 0, None))
    rows = _t_rows(names, fit.beta, se, fit.df)
    G = fit.n_units
    column = clusters.column
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
              classes: Sequence[Any] | None = None) -> Any:
    """The GEE's coefficient table: statsmodels' estimates, CR2 (BRL) intervals with Bell–McCaffrey
    degrees of freedom on its whitened working model, refused below the unit floor."""
    from turbotab.core.models.inference import (_and, _clustered, _named, _refused, floor_refusal,
                                                separated_columns)

    exog, names = _design(matrix)
    X = exog.to_numpy(dtype=float)
    if task == "binary":
        if classes is None or len(classes) != 2:
            raise ValueError("A binary outcome needs exactly two classes.")
        y = (np.asarray(y) == classes[1]).astype(float)
    else:
        y = np.asarray(y, dtype=float)
    estimator = ("GEE, exchangeable working correlation" +
                 (" (logit link)" if task == "binary" else " (identity link)"))
    start = np.linalg.lstsq(X, y, rcond=None)[0]
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

    def build(self, task: Task, purpose: Purpose | None, n_rows: int, n_features: int) -> Any:
        return RandomInterceptRegressor()

    def describe(self, task: Task, purpose: Purpose | None) -> tuple[str, str]:
        detail = "Fits straight-line effects with an intercept of its own for each unit, by REML."
        if purpose == "inference":
            detail += " Intervals are on t with Satterthwaite degrees of freedom."
        return "Random-intercept mixed model", detail

    def coefficients(self, pipeline: Any, X: Any, y: Any, *, task: Task,
                     purpose: Purpose | None, groups: Any = None) -> list[dict[str, Any]] | None:
        model = pipeline[-1]
        return coefficient_rows([str(f) for f in model.feature_names_in_], model.coef_,
                                intercept=model.intercept_)

    def inference(self, pipeline: Any, X: Any, y: Any, *, task: Task, clusters: Any) -> Any:
        from turbotab.core.models.linear import model_matrix

        return mixed_table(model_matrix(pipeline, X), y, clusters)

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

    def inference(self, pipeline: Any, X: Any, y: Any, *, task: Task, clusters: Any) -> Any:
        from turbotab.core.models.linear import model_matrix

        classes = list(getattr(pipeline[-1], "classes_", [])) or None
        return gee_table(model_matrix(pipeline, X), y, clusters, task, classes)

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
