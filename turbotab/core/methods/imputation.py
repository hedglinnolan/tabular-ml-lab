"""Multiple imputation by chained equations, and Rubin's rules (AUDIT_REPORT §5 WP7; ME-01, ME-08).

Under inference a missing predictor is filled ``m`` times from its conditional distribution given
every other variable of the analysis, **the outcome and total energy included** (BLUEPRINT §12
ruling 4), each completed table is analyzed as if it were complete, and the ``m`` answers are
combined by Rubin's rules. Harrell (*Regression Modeling Strategies*, §3.8, citing Moons et al.
2006, *J Clin Epidemiol* 59:1092): "multiple imputation can and should use the response variable
for imputing predictors". Under prediction nothing here runs: the pipeline imputes in-fold without
the outcome, so the fitted pipeline can be deployed (Sisk et al. 2023, *Stat Methods Med Res*
32:1461).

**The chained equations** (van Buuren & Groothuis-Oudshoorn 2011, *J Stat Softw* 45(3), the
``mice`` algorithm). Each incomplete variable gets its own model given all the others; the
variables are visited in turn, from the least to the most incomplete, for :data:`ITERATIONS`
sweeps, and ``m`` independent chains give the ``m`` completed tables. A chain starts from a random
draw of each column's observed values. Every model is *proper*: its parameters are drawn from their
approximate posterior before each imputation, so the spread between imputations carries the
uncertainty of the imputation model itself (Rubin 1987, *Multiple Imputation for Nonresponse in
Surveys*, §4.2). By kind:

* **numbers** — Bayesian linear regression (``mice``'s ``norm``): σ*² = RSS/χ²(n_obs − p), β* ~
  N(β̂, σ*²(XᵀX)⁻¹) with ``mice``'s ridge of 10⁻⁵ on the diagonal (``.norm.draw``), and each blank
  drawn as X β* + σ* ε. Not predictive mean matching: with a confounder 40% missing at random given
  the exposure (the audit's fixture), matching to observed donors could not reach the high values
  the blanks hide and gave 95% intervals that covered the truth 0.905 of the time over 400 runs,
  against 0.93–0.95 for these draws. Nor transformed or truncated for a skewed intake: von Hippel
  (2013, *Sociol Methods Res* 42:105) found that "if one has to impute a skewed variable under a
  normal model, it is usually safest to do so without modifications–unless you are more interested
  in estimating percentiles and shape than in estimating means, variances, and regressions". An
  imputed value can therefore lie outside the observed range; the estimand is a coefficient, not a
  value to display.
* **yes/no** — logistic regression (Newton, with a small ridge so separation stays finite; ``mice``
  augments the data to the same end, White, Daniel & Royston 2010), β* ~ N(β̂, H⁻¹), then a
  Bernoulli draw.
* **categories** — multinomial logistic regression refit on a bootstrap sample of the observed
  rows (the bootstrap makes the draw proper), then a draw from the predicted probabilities.
* **values below a detection limit** (left-censored, ``censored``) — a censored-normal (Tobit) model
  of the column, on its own or the log scale (whichever fits, :func:`censoring_of`): detected values
  are exact and a blank is known only to lie below the limit. Its maximum-likelihood parameters are
  drawn from N(θ̂, I(θ̂)⁻¹), and each blank from the normal truncated above at the limit. Lubin et al.
  2004 (*Environ Health Perspect* 112:1691): "Truncated data methods (e.g., Tobit regression) and
  multiple imputation offer two unbiased approaches for analyzing measurement data with detection
  limits … multiple imputation produces unbiased estimates and nominal confidence intervals unless
  the proportion of missing data is extreme."

**Rubin's rules** (van Buuren, *Flexible Imputation of Missing Data*, 2nd ed., §2.3, eq. 2.16–2.32):
Q̄ = mean of the m estimates, Ū = mean of their variances, B = variance between them, T = Ū +
(1 + 1/m)B; λ = (1 + 1/m)B/T; the reference is t on Barnard & Rubin's (1999, *Biometrika* 86:948)
degrees of freedom ν = ν_old ν_obs/(ν_old + ν_obs), ν_old = (m − 1)/λ², ν_obs = (ν_com + 1)/(ν_com
+ 3) ν_com (1 − λ), with ν_com each table's own complete-data degrees of freedom (ν = ν_old when the
complete-data reference is normal). The fraction of missing information γ = (r + 2/(ν + 3))/(1 + r),
r = (1 + 1/m)B/Ū. A test of several coefficients at once is Li, Raghunathan & Rubin's (1991) D1:
D1 = Q̄ᵀ Ū⁻¹ Q̄ / (k(1 + r1)), r1 = (1 + 1/m) tr(BŪ⁻¹)/k, on F(k, ν1) with ν1 = 4 + (t − 4)[1 + (1 −
2/t)/r1]² for t = k(m − 1) > 4, else t(1 + 1/k)(1 + 1/r1)²/2 (van Buuren §5.3.2).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

M_DEFAULT = 20  # BLUEPRINT §12 ruling 4 and the audit's WP7: m ≥ 20
M_MAX = 200
ITERATIONS = 10  # sweeps per chain (van Buuren §6.5: 5–20 is usually enough)
RIDGE = 1e-5  # mice's ridge on XᵀX's diagonal, for collinear imputation models
LOGIT_RIDGE = 1e-3  # on standardized predictors: keeps a separated yes/no model finite
# The most variables (after categories become indicators) an imputation model may hold: beyond
# this, each incomplete column's model is a regression on more columns than chained equations
# estimate reliably in the rows a nutrition study has, and the run time grows with the square.
MAX_MODEL_COLUMNS = 80
LOG_GAIN = 2.0  # a censored column is modeled on the log scale when that gains this much log-likelihood
OUTCOME_PREFIX = "__outcome"  # the outcome's terms in the imputation model never collide with a column

Kind = Literal["numeric", "binary", "categorical", "censored"]


@dataclass(frozen=True)
class Censoring:
    """A left-censored column: a blank lies below ``limit`` (the smallest detected value, unless
    the user declared one), modeled on the log scale when ``log``."""

    limit: float
    log: bool


@dataclass
class Imputations:
    """``m`` completed copies of the input columns, and what was imputed."""

    frames: list[pd.DataFrame]
    m: int
    iterations: int
    imputed: dict[str, int]  # column -> cells filled
    n_incomplete_rows: int
    variables: list[str]  # the imputation model's variables (outcome terms named as such)
    kinds: dict[str, str]
    censored: dict[str, Censoring] = field(default_factory=dict)
    seed: int = 0


class ImputationRefused(ValueError):
    """The data cannot carry a chained-equations model; the message says why."""


# ── the variables ────────────────────────────────────────────────────────────


def column_kind(series: pd.Series, categorical: bool = False) -> Kind:
    """How a column is imputed: categories (text, or declared), yes/no (two values 0/1, or any
    number with exactly two values, declared a code or not: BLUEPRINT §14.3 reads such a column as
    one indicator either way, so both readings must impute it alike, a draw of one of its two
    values), numbers."""
    present = series.dropna()
    numeric = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
    if numeric and present.nunique() == 2:
        return "binary"
    if categorical or not pd.api.types.is_numeric_dtype(series):
        return "binary" if present.nunique() == 2 and set(present.astype(str)) <= {"0", "1", "0.0", "1.0"} \
            else "categorical"
    if len(present) and set(np.unique(present.to_numpy(dtype=float))) <= {0.0, 1.0}:
        return "binary"
    return "numeric"


def censoring_of(series: pd.Series, limit: float | None = None) -> Censoring:
    """A left-censored column's limit (the smallest detected value unless given) and its scale.

    The scale is the one under which a censored normal fits the column better: the column itself,
    or its logarithm (raw intensities are close to log-normal; values already on a log scale are
    close to normal). The two maximized likelihoods are compared on the column's own scale (the
    log model's density carries its Jacobian, 1/x, for each detected value; a blank contributes a
    probability on either scale): a Box–Cox choice between λ = 1 and λ = 0, the log taken only when
    it raises the maximized log-likelihood by more than :data:`LOG_GAIN` (a likelihood ratio above
    e² ≈ 7.4). At 60% censored, assuming the log scale for values already logged (N(10, 1)) put the
    pooled slope 11% high in the audit's simulation (0.557 against 0.5; the column's own scale gave
    0.513). Over 100 draws of 300 rows the rule took the log scale for none of the N(10, 1) columns
    at 20% or 60% censored, and for every log-normal column with σ = 1."""
    x = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    detected = x[np.isfinite(x)]
    if not len(detected):
        raise ImputationRefused(f"`{series.name}` has no detected value, so nothing says where its "
                                f"detection limit is.")
    lod = float(np.min(detected)) if limit is None else float(limit)
    return Censoring(limit=lod, log=log_scale_fits_better(x, lod))


def censored_loglik(z: np.ndarray, censored: np.ndarray, limit: float) -> float:
    """The maximized log-likelihood of a censored normal with a mean and a spread alone."""
    from scipy import stats

    theta, _ = tobit_fit(np.ones((len(z), 1)), np.where(censored, limit, z), censored, limit)
    mu, sigma = float(theta[0]), math.exp(float(theta[1]))
    det = ~censored
    return float(stats.norm.logpdf(z[det], mu, sigma).sum()
                 + stats.norm.logcdf((limit - mu) / sigma) * int(censored.sum()))


def log_scale_fits_better(x: np.ndarray, limit: float) -> bool:
    """True when a censored normal on log x fits better than one on x (see :func:`censoring_of`)."""
    x = np.asarray(x, dtype=float)
    blank = ~np.isfinite(x)
    detected = x[~blank]
    if not (np.all(detected > 0) and limit > 0) or len(detected) < 3 or np.ptp(detected) == 0:
        return False
    if not blank.any():
        return True
    linear = censored_loglik(np.where(blank, limit, x), blank, limit)
    logged = (censored_loglik(np.log(np.where(blank, limit, x)), blank, math.log(limit))
              - float(np.log(detected).sum()))
    return logged - linear > LOG_GAIN


def outcome_terms(task: str, y: Any, *, event_time: Any = None) -> pd.DataFrame:
    """The outcome as the imputation model holds it.

    A number or an ordered outcome's codes as one term; a yes/no outcome as its 0/1 event; several
    unordered classes as indicators against the first. A time-to-event outcome enters as its event
    indicator and the Nelson–Aalen cumulative hazard at the row's own time (White & Royston 2009,
    *Stat Med* 28:1982: "we recommend … including the event indicator and the Nelson–Aalen estimate
    of the cumulative hazard").
    """
    if task == "time_to_event":
        values = np.asarray(y)
        event = np.asarray(values["event"], dtype=float)
        time = np.asarray(values["time"], dtype=float)
        return pd.DataFrame({f"{OUTCOME_PREFIX}_event": event,
                             f"{OUTCOME_PREFIX}_hazard": nelson_aalen(time, event)})
    if task == "multiclass":
        codes = pd.Categorical(np.asarray(y))
        out = {f"{OUTCOME_PREFIX}_{k}": (codes.codes == k).astype(float)
               for k in range(1, len(codes.categories))}
        return pd.DataFrame(out)
    return pd.DataFrame({OUTCOME_PREFIX: np.asarray(y, dtype=float)})


def nelson_aalen(time: np.ndarray, event: np.ndarray) -> np.ndarray:
    """The Nelson–Aalen cumulative hazard H(t) at each row's own time (ties: all events at t
    count, the risk set is everyone still followed at t)."""
    order = np.argsort(time, kind="mergesort")
    t, d = time[order], event[order]
    unique, start = np.unique(t, return_index=True)
    deaths = np.add.reduceat(d, start)
    at_risk = len(t) - start
    H = np.cumsum(deaths / at_risk)
    out = np.empty(len(time))
    out[order] = H[np.searchsorted(unique, t)]
    return out


# ── one variable's model ─────────────────────────────────────────────────────


def _standardize(block: np.ndarray) -> np.ndarray:
    mu = np.nanmean(block, axis=0)
    sd = np.nanstd(block, axis=0)
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    return (block - mu) / sd


class _Design:
    """Every variable's current values as standardized model columns, so the design for one
    variable is all the others' blocks beside an intercept."""

    def __init__(self, data: pd.DataFrame, kinds: Mapping[str, str], censored: Mapping[str, Censoring]):
        self.kinds = dict(kinds)
        self.censored = dict(censored)
        self.levels: dict[str, list[Any]] = {}
        self.values: dict[str, np.ndarray] = {}
        for c in data.columns:
            if kinds[c] == "categorical":
                self.levels[c] = sorted(pd.unique(data[c].dropna()), key=str)
                self.values[c] = data[c].to_numpy(dtype=object)
            else:
                self.values[c] = pd.to_numeric(data[c], errors="coerce").to_numpy(dtype=float)
        self.blocks: dict[str, np.ndarray] = {}

    def refresh(self, column: str) -> None:
        v = self.values[column]
        kind = self.kinds[column]
        if kind == "categorical":
            levels = self.levels[column]
            block = np.column_stack([(v == lv).astype(float) for lv in levels[1:]]) if len(levels) > 1 \
                else np.zeros((len(v), 0))
        else:
            x = v.astype(float)
            c = self.censored.get(column)
            if c is not None and c.log:
                x = np.log(np.clip(x, 1e-300, None))
            block = _standardize(x.reshape(-1, 1))
        self.blocks[column] = block

    def matrix(self, without: str) -> np.ndarray:
        parts = [self.blocks[c] for c in self.blocks if c != without]
        n = len(next(iter(self.values.values())))
        return np.column_stack([np.ones(n), *parts]) if parts else np.ones((n, 1))


def _norm_draw(X: np.ndarray, y: np.ndarray, rng: np.random.Generator
               ) -> tuple[np.ndarray, np.ndarray, float]:
    """(β̂, β*, σ*): least squares and a draw from its posterior (mice's ``.norm.draw``)."""
    XtX = X.T @ X
    XtX = XtX + np.diag(RIDGE * np.maximum(np.diag(XtX), 1e-12))
    inv = np.linalg.pinv(XtX)
    beta = inv @ (X.T @ y)
    resid = y - X @ beta
    df = max(len(y) - X.shape[1], 1)
    sigma = math.sqrt(float(resid @ resid) / rng.chisquare(df))
    root = np.linalg.cholesky(inv + np.eye(len(inv)) * 1e-12)
    star = beta + sigma * (root @ rng.standard_normal(len(beta)))
    return beta, star, sigma


def _norm(X: np.ndarray, y: np.ndarray, miss: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Bayesian linear-regression imputation (mice's ``norm``): X_mis β* + σ* ε."""
    obs = ~miss
    beta, star, sigma = _norm_draw(X[obs], y[obs], rng)
    return X[miss] @ star + sigma * rng.standard_normal(int(miss.sum()))


def _expit(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def _logistic(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ridge-penalized logistic regression by Newton: (β̂, the inverse penalized information)."""
    p = X.shape[1]
    penalty = np.full(p, LOGIT_RIDGE * len(y))
    penalty[0] = 0.0
    beta = np.zeros(p)
    for _ in range(50):
        mu = _expit(X @ beta)
        w = mu * (1 - mu)
        H = (X * w[:, None]).T @ X + np.diag(penalty)
        g = X.T @ (y - mu) - penalty * beta
        step = np.linalg.solve(H + np.eye(p) * 1e-10, g)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-8:
            break
    mu = _expit(X @ beta)
    H = (X * (mu * (1 - mu))[:, None]).T @ X + np.diag(penalty)
    return beta, np.linalg.pinv(H)


def _binary(X: np.ndarray, y: np.ndarray, miss: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    obs = ~miss
    beta, V = _logistic(X[obs], y[obs])
    root = np.linalg.cholesky(V + np.eye(len(V)) * 1e-12)
    star = beta + root @ rng.standard_normal(len(beta))
    return (rng.random(int(miss.sum())) < _expit(X[miss] @ star)).astype(float)


def _categorical(X: np.ndarray, values: np.ndarray, levels: Sequence[Any], miss: np.ndarray,
                 rng: np.random.Generator) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression

    obs = np.flatnonzero(~miss)
    boot = rng.choice(obs, size=len(obs), replace=True)
    labels = values[boot]
    present = sorted(pd.unique(labels), key=str)
    if len(present) == 1:
        return np.full(int(miss.sum()), present[0], dtype=object)
    model = LogisticRegression(C=1.0 / LOGIT_RIDGE, max_iter=500)
    model.fit(X[boot][:, 1:], labels.astype(str))
    probs = model.predict_proba(X[miss][:, 1:])
    cum = np.cumsum(probs, axis=1)
    draw = (rng.random(len(probs))[:, None] > cum).sum(axis=1).clip(0, probs.shape[1] - 1)
    by_name = {str(lv): lv for lv in levels}
    return np.asarray([by_name.get(model.classes_[k], model.classes_[k]) for k in draw], dtype=object)


# The censored-normal (Tobit) model ─────────────────────────────────────────


def tobit_fit(X: np.ndarray, z: np.ndarray, censored: np.ndarray, limit: float
              ) -> tuple[np.ndarray, np.ndarray]:
    """Maximum likelihood for z ~ N(Xβ, σ²) where ``censored`` rows are known only to lie below
    ``limit``: (θ̂ = (β, log σ), the inverse observed information at θ̂)."""
    from scipy import optimize, stats

    det = ~censored
    Xd, zd, Xc = X[det], z[det], X[censored]

    def nll(theta: np.ndarray) -> tuple[float, np.ndarray]:
        beta, tau = theta[:-1], theta[-1]
        sigma = math.exp(tau)
        e = (zd - Xd @ beta) / sigma
        a = (limit - Xc @ beta) / sigma
        logcdf = stats.norm.logcdf(a)
        ll = float(-len(zd) * tau - 0.5 * e @ e + logcdf.sum())
        lam = np.exp(stats.norm.logpdf(a) - logcdf)
        g_beta = Xd.T @ e / sigma - Xc.T @ lam / sigma
        g_tau = -len(zd) + float(e @ e) - float(lam @ a)
        return -ll, -np.concatenate([g_beta, [g_tau]])

    start_beta = np.linalg.lstsq(Xd, zd, rcond=None)[0] if len(zd) >= X.shape[1] else np.zeros(X.shape[1])
    resid = zd - Xd @ start_beta
    start = np.concatenate([start_beta, [math.log(max(float(np.std(resid)), 1e-6))]])
    fit = optimize.minimize(nll, start, jac=True, method="BFGS",
                            options={"gtol": 1e-8, "maxiter": 1000})
    theta = fit.x
    # The observed information by central differences of the analytic gradient.
    k = len(theta)
    H = np.empty((k, k))
    for j in range(k):
        h = 1e-5 * max(1.0, abs(theta[j]))
        up, down = theta.copy(), theta.copy()
        up[j] += h
        down[j] -= h
        H[:, j] = (nll(up)[1] - nll(down)[1]) / (2 * h)
    H = 0.5 * (H + H.T)
    return theta, np.linalg.pinv(H)


def tobit_expectation(mu: float, sigma: float, limit: float) -> float:
    """E[Z | Z < limit] for Z ~ N(mu, σ²): mu − σ φ(a)/Φ(a), a = (limit − mu)/σ."""
    from scipy import stats

    a = (limit - mu) / sigma
    return float(mu - sigma * math.exp(stats.norm.logpdf(a) - stats.norm.logcdf(a)))


def _censored(X: np.ndarray, x: np.ndarray, miss: np.ndarray, c: Censoring,
              rng: np.random.Generator) -> np.ndarray:
    from scipy import stats

    z = np.log(np.clip(x, 1e-300, None)) if c.log else x.astype(float)
    limit = math.log(c.limit) if c.log else c.limit
    theta, V = tobit_fit(X, np.where(miss, limit, z), miss, limit)
    root = np.linalg.cholesky(V + np.eye(len(V)) * 1e-12)
    star = theta + root @ rng.standard_normal(len(theta))
    mu, sigma = X[miss] @ star[:-1], math.exp(star[-1])
    upper = (limit - mu) / sigma
    draws = stats.truncnorm.rvs(-np.inf, upper, loc=mu, scale=sigma, random_state=rng)
    draws = np.minimum(np.asarray(draws, dtype=float), limit)
    return np.exp(draws) if c.log else draws


# ── the chained equations ────────────────────────────────────────────────────


def chained_equations(data: pd.DataFrame, *, impute: Sequence[str] | None = None,
                      kinds: Mapping[str, str] | None = None,
                      censored: Mapping[str, Censoring] | None = None,
                      m: int = M_DEFAULT, iterations: int = ITERATIONS, seed: int = 0,
                      progress: Callable[[int, int], None] | None = None,
                      cancelled: Callable[[], bool] | None = None) -> Imputations:
    """``m`` completed copies of ``data`` (every column of the imputation model).

    ``impute``: the columns whose blanks are filled (default: every column with a blank); the
    others are only predictors (the outcome's terms, which are never missing in the analysis rows).
    ``kinds``: each column's kind (default :func:`column_kind`). ``censored``: the left-censored
    columns and their limits. Deterministic for a given ``seed``.
    """
    censored = dict(censored or {})
    kinds = {c: (kinds or {}).get(c) or ("censored" if c in censored else column_kind(data[c]))
             for c in data.columns}
    for c in censored:
        kinds[c] = "censored"
    targets = [c for c in (impute if impute is not None else data.columns)
               if c in data.columns and data[c].isna().any()]
    width = sum(max(1, data[c].nunique(dropna=True) - 1) if kinds[c] == "categorical" else 1
                for c in data.columns)
    if width > MAX_MODEL_COLUMNS:
        raise ImputationRefused(
            f"The imputation model would hold {width:,} columns (categories as indicators), more "
            f"than the {MAX_MODEL_COLUMNS} chained equations are run with here.")
    for c in targets:
        if data[c].notna().sum() < 2 and kinds[c] != "categorical":
            raise ImputationRefused(f"`{c}` has fewer than two observed values, so it cannot be "
                                    f"imputed.")
    miss = {c: data[c].isna().to_numpy() for c in targets}
    n_incomplete = int(np.any(np.column_stack([miss[c] for c in targets]), axis=1).sum()) if targets else 0
    order = sorted(targets, key=lambda c: (int(miss[c].sum()), list(data.columns).index(c)))
    sweeps = iterations if len(order) > 1 else 1  # one incomplete column: the chain is monotone
    rng = np.random.default_rng(seed)
    frames: list[pd.DataFrame] = []
    total = m * sweeps * max(1, len(order))
    done = 0
    for _k in range(m):
        design = _Design(data, kinds, censored)
        for c in order:  # start each chain from random draws of the observed values
            observed = design.values[c][~miss[c]]
            fill = observed[rng.integers(0, len(observed), int(miss[c].sum()))]
            if c in censored:  # a non-detect starts below its limit
                spread = float(np.nanstd(observed.astype(float))) or 1.0
                fill = np.full(int(miss[c].sum()), censored[c].limit / 2 if censored[c].log
                               else censored[c].limit - spread)
            design.values[c] = design.values[c].copy()
            design.values[c][miss[c]] = fill
        for c in data.columns:
            design.refresh(c)
        for _ in range(sweeps):
            for c in order:
                if cancelled is not None and cancelled():
                    from turbotab.core.jobs import Cancelled

                    raise Cancelled()
                X = design.matrix(c)
                kind = kinds[c]
                if kind == "numeric":
                    new = _norm(X, design.values[c].astype(float), miss[c], rng)
                elif kind == "binary":
                    new = _binary(X, _as01(design.values[c], data[c]), miss[c], rng)
                    new = _from01(new, data[c])
                elif kind == "categorical":
                    new = _categorical(X, design.values[c], design.levels[c], miss[c], rng)
                else:
                    new = _censored(X, design.values[c].astype(float), miss[c], censored[c], rng)
                design.values[c] = design.values[c].copy()
                design.values[c][miss[c]] = new
                design.refresh(c)
                done += 1
                if progress is not None:
                    progress(done, total)
        completed = data.copy()
        for c in order:
            col = design.values[c]
            if kinds[c] == "categorical" or not pd.api.types.is_numeric_dtype(data[c]):
                filled = pd.Series(col, index=data.index, dtype=object)
                if pd.api.types.is_numeric_dtype(data[c]):  # a declared category of number codes
                    filled = filled.astype(float)
            else:
                filled = pd.Series(col.astype(float), index=data.index)
            completed[c] = filled
        frames.append(completed)
    return Imputations(frames=frames, m=m, iterations=sweeps,
                       imputed={c: int(miss[c].sum()) for c in order},
                       n_incomplete_rows=n_incomplete, variables=list(data.columns), kinds=kinds,
                       censored=censored, seed=seed)


def _two_levels(original: pd.Series | None) -> tuple[float, float] | None:
    """A numeric yes/no column's two values, low then high, when they are not 0 and 1."""
    if original is None or not pd.api.types.is_numeric_dtype(original) \
            or pd.api.types.is_bool_dtype(original):
        return None
    levels = sorted(float(v) for v in pd.unique(original.dropna()))
    if len(levels) != 2 or levels == [0.0, 1.0]:
        return None
    return levels[0], levels[1]


def _as01(values: np.ndarray, original: pd.Series | None = None) -> np.ndarray:
    """A yes/no column's current values as 0/1 floats (text levels: the second sorted is 1; two
    other numbers, 1 and 2: the higher is 1)."""
    two = _two_levels(original)
    if two is not None:
        return (np.asarray(values, dtype=float) == two[1]).astype(float)
    try:
        return np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        labels = sorted(pd.unique(pd.Series(values).dropna()), key=str)
        return (pd.Series(values) == labels[-1]).to_numpy(dtype=float)


def _from01(draws: np.ndarray, original: pd.Series) -> np.ndarray:
    two = _two_levels(original)
    if two is not None:
        return np.where(draws > 0.5, two[1], two[0])
    if pd.api.types.is_numeric_dtype(original):
        return draws
    labels = sorted(pd.unique(original.dropna()), key=str)
    return np.where(draws > 0.5, labels[-1], labels[0]).astype(object)


# ── Rubin's rules ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Pooled:
    """One quantity pooled over m imputations."""

    estimate: float
    within: float  # Ū
    between: float  # B
    total: float  # T
    df: float | None  # Barnard–Rubin ν (None: no variance to pool)
    fmi: float | None  # γ, the fraction of missing information
    ci_low: float | None
    ci_high: float | None
    p: float | None


def barnard_rubin(m: int, within: float, between: float, df_com: float | None
                  ) -> tuple[float, float, float]:
    """(T, ν, γ) for one quantity: Rubin's total variance, Barnard & Rubin's degrees of freedom
    (``df_com`` None: an infinite complete-data df, Rubin's 1987 ν_old), the fraction of missing
    information."""
    total = within + (1.0 + 1.0 / m) * between
    if total <= 0:
        return total, float("inf") if df_com is None else float(df_com), 0.0
    lam = (1.0 + 1.0 / m) * between / total
    if lam <= 1e-12:
        nu = float("inf") if df_com is None else float(df_com)
    else:
        nu_old = (m - 1) / lam ** 2
        if df_com is None or not math.isfinite(df_com):
            nu = nu_old
        else:
            nu_obs = (df_com + 1) / (df_com + 3) * df_com * (1 - lam)
            nu = nu_old * nu_obs / (nu_old + nu_obs)
    r = (1.0 + 1.0 / m) * between / within if within > 0 else float("inf")
    if math.isinf(r):
        gamma = 1.0
    elif math.isinf(nu):
        gamma = r / (1 + r)
    else:
        gamma = (r + 2.0 / (nu + 3.0)) / (1.0 + r)
    return total, nu, gamma


def pool_scalar(estimates: Sequence[float], variances: Sequence[float], df_com: float | None = None,
                level: float = 0.95) -> Pooled:
    from scipy import stats

    q = np.asarray(estimates, dtype=float)
    u = np.asarray(variances, dtype=float)
    m = len(q)
    est = float(q.mean())
    within = float(u.mean())
    between = float(q.var(ddof=1)) if m > 1 else 0.0
    total, nu, gamma = barnard_rubin(m, within, between, df_com)
    if not (total > 0 and math.isfinite(total)):
        return Pooled(est, within, between, total, None, None, None, None, None)
    se = math.sqrt(total)
    ref = stats.norm if math.isinf(nu) else stats.t(nu)
    half = float(ref.ppf(0.5 + level / 2)) * se
    p = float(2 * ref.sf(abs(est) / se))
    return Pooled(est, within, between, total, None if math.isinf(nu) else nu, gamma,
                  est - half, est + half, p)


def pooled_wald(estimates: np.ndarray, covariances: np.ndarray) -> dict[str, float] | None:
    """Li, Raghunathan & Rubin's D1 that every coefficient is zero: ``estimates`` (m × k),
    ``covariances`` (m × k × k). None when the within-imputation covariance is singular."""
    from scipy import stats

    Q = np.asarray(estimates, dtype=float)
    U = np.asarray(covariances, dtype=float)
    m, k = Q.shape
    qbar = Q.mean(axis=0)
    ubar = U.mean(axis=0)
    if not np.all(np.isfinite(ubar)) or np.linalg.matrix_rank(ubar) < k:
        return None
    B = np.cov(Q, rowvar=False, ddof=1).reshape(k, k) if m > 1 else np.zeros((k, k))
    inv = np.linalg.inv(ubar)
    r1 = (1 + 1 / m) * float(np.trace(B @ inv)) / k
    D1 = float(qbar @ inv @ qbar) / (k * (1 + r1))
    t = k * (m - 1)
    if r1 <= 1e-12:
        nu1 = float("inf")
    elif t > 4:
        nu1 = 4 + (t - 4) * (1 + (1 - 2 / t) / r1) ** 2
    else:
        nu1 = t * (1 + 1 / k) * (1 + 1 / r1) ** 2 / 2
    p = float(stats.chi2.sf(D1 * k, k)) if math.isinf(nu1) else float(stats.f.sf(D1, k, nu1))
    return {"statistic": D1, "df_num": k, "df_den": None if math.isinf(nu1) else nu1, "p": p,
            "r1": r1}


def pool_rows(tables: Sequence[Sequence[Mapping[str, Any]]], level: float = 0.95
              ) -> list[dict[str, Any]]:
    """Coefficient rows pooled feature by feature over the imputations' tables (in the first
    table's order). A row is pooled from its ``estimate`` and ``se`` with its own ``df`` as the
    complete-data degrees of freedom; a row with no standard error is averaged and carries no
    interval. A ratio (odds or relative-risk ratio) is exp of the pooled estimate and its ends."""
    m = len(tables)
    first = list(tables[0])
    by_name = [{str(r["feature"]): r for r in t} for t in tables]
    out: list[dict[str, Any]] = []
    for row in first:
        name = str(row["feature"])
        rows = [b.get(name) for b in by_name]
        if any(r is None for r in rows):
            continue
        est = [r.get("estimate") for r in rows]
        if any(e is None for e in est):
            out.append({**row, "estimate": None, "ci_low": None, "ci_high": None, "p": None,
                        "se": None, "df": None, "fmi": None})
            continue
        se = [r.get("se") for r in rows]
        new = dict(row)
        if any(s is None for s in se):
            new.update(estimate=float(np.mean(est)), ci_low=None, ci_high=None, p=None, se=None,
                       df=None, fmi=None)
        else:
            dfs = [r.get("df") for r in rows]
            df_com = None if any(d is None for d in dfs) else float(np.mean(dfs))
            pooled = pool_scalar(est, [float(s) ** 2 for s in se], df_com, level)
            new.update(estimate=pooled.estimate, ci_low=pooled.ci_low, ci_high=pooled.ci_high,
                       p=pooled.p, se=math.sqrt(pooled.total) if pooled.total > 0 else None,
                       df=pooled.df, fmi=pooled.fmi)
        if "ratio" in row:
            ratio = row.get("ratio") is not None or any(r.get("ratio") is not None for r in rows)
            new.update(ratio=_exp(new["estimate"]) if ratio else None,
                       ratio_low=_exp(new["ci_low"]) if ratio else None,
                       ratio_high=_exp(new["ci_high"]) if ratio else None)
        if any("q" in r for r in rows):
            new["q"] = None  # recomputed over the pooled p-values by the caller
        out.append(new)
    if any("q" in r for r in out):
        _bh(out)
    return out


def _exp(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = math.exp(float(value))
    except OverflowError:
        return None
    return out if math.isfinite(out) else None


def _bh(rows: list[dict[str, Any]]) -> None:
    """Benjamini–Hochberg adjusted p-values over the rows that carried one (a feature-wise table)."""
    idx = [i for i, r in enumerate(rows) if "q" in r and r.get("p") is not None]
    if not idx:
        return
    p = np.asarray([rows[i]["p"] for i in idx], dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1.0)
    out = np.empty(n)
    out[order] = q
    for i, v in zip(idx, out):
        rows[i]["q"] = float(v)


def pooled_cov(estimates: np.ndarray, covariances: np.ndarray) -> np.ndarray:
    """Rubin's total covariance T = Ū + (1 + 1/m)B of a vector of coefficients."""
    Q = np.asarray(estimates, dtype=float)
    U = np.asarray(covariances, dtype=float)
    m = Q.shape[0]
    B = np.cov(Q, rowvar=False, ddof=1) if m > 1 else np.zeros(U.shape[1:])
    return U.mean(axis=0) + (1 + 1 / m) * np.atleast_2d(B)


__all__ = [
    "Censoring", "ITERATIONS", "ImputationRefused", "Imputations", "M_DEFAULT", "M_MAX",
    "censored_loglik", "log_scale_fits_better",
    "MAX_MODEL_COLUMNS", "OUTCOME_PREFIX", "Pooled", "barnard_rubin", "censoring_of",
    "chained_equations", "column_kind", "nelson_aalen", "outcome_terms", "pool_rows", "pool_scalar",
    "pooled_cov", "pooled_wald", "tobit_expectation", "tobit_fit",
]
