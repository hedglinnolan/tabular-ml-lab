"""The exposure's effect as the estimand declares it, and what it rests on (MODELING_SEQUENCE §0
rulings 9 and 10; §1 rows 2, 11 and 12; the package ESTIMAND).

Pure numerics and display helpers, no project state: the ``effects`` stage
(``turbotab/core/stages/effects.py``) calls them on the analysis rows. Four parts.

**Marginal standardization (g-computation).** A logistic model's odds ratio is conditional on its
covariates and non-collapsible: it changes when a covariate that predicts the outcome is added,
even without confounding (Daniel, Zhang & Farewell 2021, *Biom J* 63:528: "the words conditional
and adjusted (likewise marginal and unadjusted) should not be used interchangeably"). The marginal
risk difference and risk ratio average each row's predicted risk under two settings of the
exposure: ``r1 = mean_i expit(x_i(1)ᵀβ)`` and ``r0 = mean_i expit(x_i(0)ᵀβ)``, ``RD = r1 − r0``,
``RR = r1 / r0`` (standardization over the analyzed rows' own covariates). A two-valued exposure is
set to each of its values; a continuous one is the observed value against the observed value plus
one unit (every person's exposure one unit higher). The model is the maximum-likelihood logistic
regression on the pipeline's model matrix, an aliased column left out as R's ``glm`` leaves it
(:func:`logistic_fit`); the intervals are percentiles of a nonparametric bootstrap that refits the
whole pipeline (energy model, forms, fill) and the model on each resample, resampling whole units
when rows repeat (:func:`g_computation`). A resample whose outcome happens to be separated has no
finite estimate, but its fitted risks have a limit (0 or 1 for the separated rows, the rest the
maximum-likelihood fit of the others; :func:`separated_rows`), the fit R's ``glm`` approaches as
its coefficients run away: it is kept at that limit and counted, since leaving it out would cut
the interval's most extreme resamples.

**The Table 2 display.** Westreich & Greenland (2013, *Am J Epidemiol* 177:292): "Presentation of
exposure and confounder effect estimates from a single model may lead to several interpretative
difficulties, inviting confusion of direct-effect estimates with total-effect estimates for
covariates in the model." So only the exposure's rows are effect estimates; every other row is an
adjustment term, listed apart under :data:`APPENDIX_TITLE` (:func:`split_rows`), and a declared
modifier's main effect is never one.

**Diagnostics** are reported and never acted on silently:

* proportional hazards by Schoenfeld residuals: the score test of ``γ = 0`` for each term's
  covariates multiplied by a function of time, ``g(t) = 1 − KM(t−)`` centered over the events,
  with ``u_β = 0`` and the expanded model's full information, as R's ``survival::cox.zph``
  computes it since version 3.0 (Grambsch & Therneau 1994, *Biometrika* 81:515), Efron's handling
  of ties and delayed entry included (:func:`cox_zph`);
* influence: each row's leverage and Cook's distance, as R's ``hatvalues`` and
  ``cooks.distance`` give them for a least-squares or a logistic model (:func:`ols_influence`,
  :func:`logistic_influence`), against the median of F(p, n − p) (Cook 1977, *Technometrics*
  19:15), the reference R's ``plot.lm`` draws its contours from. As in R, an aliased column is
  left out and p is the rank (:func:`column_basis`), and a row with leverage 1 (the fit passes
  through it, as through a level held by one row) has no distance (NaN).

**Sensitivity to unmeasured confounding** (MODELING_SEQUENCE §0 ruling 10), offered for every
inference result and required in the causal lane (:func:`unmeasured_confounding`, the function
the causal package calls):

* the E-value (VanderWeele & Ding 2017, *Ann Intern Med* 167:268) for the estimate and for the
  confidence limit nearer the null, with the R package ``EValue``'s conversions (an odds ratio of a
  common outcome by its square root, a hazard ratio by ``(1 − 0.5^√HR) / (1 − 0.5^√(1/HR))``, a
  linear coefficient by ``exp(0.91 d)``); never with a pass/fail threshold (Ioannidis, Tan & Blum
  2019, *Ann Intern Med* 170:108: "No general rule can exist about what is a 'small enough'
  E-value");
* for a linear outcome, the Cinelli–Hazlett robustness value and partial R² (Cinelli & Hazlett
  2020, *J R Stat Soc B* 82:39), benchmarked against each named measured covariate ("an unmeasured
  confounder as strong as `smoking`"), as R's ``sensemakr`` computes them; ranked first, because it
  is anchored to the study's own covariates. What rests on the point estimate alone (RV_q, the
  bounds, the adjusted estimates) is algebraic; the interval forms rest on the classical standard
  error, so they are reported only when the interval shown is the classical one.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

LEVEL = 0.95
BOOT = 1_000  # bootstrap resamples for a percentile interval (Efron & Tibshirani 1993, §13.3)
APPENDIX_TITLE = "adjustment terms, not effect estimates"
WESTREICH = "Westreich & Greenland 2013, Am J Epidemiol 177:292"
VANDERWEELE_DING = "VanderWeele & Ding 2017, Ann Intern Med 167:268"
CINELLI_HAZLETT = "Cinelli & Hazlett 2020, J R Stat Soc B 82:39"
GRAMBSCH_THERNEAU = "Grambsch & Therneau 1994, Biometrika 81:515"
DANIEL = "Daniel, Zhang & Farewell 2021, Biom J 63:528"
ZHANG_YU = "Zhang & Yu 1998, JAMA 280:1690"
COOK = "Cook 1977, Technometrics 19:15"
# VanderWeele & Ding (2017): an odds ratio or hazard ratio approximates the risk ratio when the
# outcome is rare, "(e.g., <15% at the end of follow-up)"; above it EValue converts it.
RARE_OUTCOME = 0.15
# Zhang & Yu (1998): the odds ratio overstates the risk ratio once the outcome's incidence passes
# about 10%; above it the marginal measures rank first (ruling 9).
COMMON_OUTCOME = 0.10
PH_ALPHA = 0.05  # the proportional-hazards check of the exposure's own term, at the customary level


class Unestimable(ValueError):
    """The quantity cannot be estimated on these rows (separation, no events, a singular matrix)."""


class Separated(Unestimable):
    """The logistic likelihood has no finite maximum: the covariates separate an outcome level."""


# R's ``qr(x, tol = 1e-07)``, the rank rule of ``lm`` (LINPACK ``dqrdc2``): a column whose part
# outside the span of the columns before it is below this share of its own length is aliased.
ALIAS_TOL = 1e-7
# R's ``lm.influence`` sets a leverage above ``1 − 10ε`` to exactly 1, and ``cooks.distance`` then
# reports NaN for that row (the fit passes through it). The engine's QR is not R's bit for bit, so
# a leverage within 1e-10 of 1 is read as 1: a row the fit passes through, such as the one row of
# a categorical level held by one row, has a structural leverage of exactly 1.
LEVERAGE_ONE = 1e-10


def column_basis(X: np.ndarray, tol: float = ALIAS_TOL) -> list[int]:
    """The columns ``lm`` keeps, in order: each column is kept unless its part outside the span of
    the kept columns before it is below ``tol`` of its own length (R's limited pivoting, which
    moves an aliased column to the end and reports its coefficient as NA). Two passes of
    Gram–Schmidt against the kept columns make the residual exact to rounding."""
    X = np.asarray(X, dtype=float)
    kept: list[int] = []
    Q = np.zeros((X.shape[0], 0))
    for j in range(X.shape[1]):
        x = X[:, j]
        length = float(np.linalg.norm(x))
        if length == 0.0:
            continue
        r = x - Q @ (Q.T @ x)
        r = r - Q @ (Q.T @ r)
        rest = float(np.linalg.norm(r))
        if rest <= tol * length:
            continue
        kept.append(j)
        Q = np.column_stack([Q, r / rest])
    return kept


# ── marginal standardization (g-computation) ─────────────────────────────────


def _expit(eta: np.ndarray) -> np.ndarray:
    out = np.empty_like(eta, dtype=float)
    pos = eta >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-eta[pos]))
    z = np.exp(eta[~pos])
    out[~pos] = z / (1.0 + z)
    return out


def logistic_mle(X: np.ndarray, y: np.ndarray, *, tol: float = 1e-12, max_iter: int = 100) -> np.ndarray:
    """The logistic model's maximum-likelihood coefficients by Newton–Raphson, halving a step that
    lowers the likelihood; ``X`` carries its own intercept column. Converged when the largest step
    is below ``tol`` times the coefficients' scale. Raises :class:`Unestimable` when the likelihood
    has no maximum (separation: a coefficient runs away) or the information is singular."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2 or len(X) != len(y):
        raise ValueError("X and y must have one row per observation.")
    beta = np.zeros(X.shape[1])
    # A coefficient running past this many log-odds per standard deviation of its column is an
    # infinite one (separation), as the Cox fit reads it (``survival.DIVERGED``); a constant column
    # (the intercept) is read per unit.
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0

    def loglik(b: np.ndarray) -> float:
        eta = X @ b
        return float(np.sum(y * eta - np.logaddexp(0.0, eta)))

    ll = loglik(beta)
    for _ in range(max_iter):
        mu = _expit(X @ beta)
        w = mu * (1.0 - mu)
        info = (X.T * w) @ X
        try:
            step = np.linalg.solve(info, X.T @ (y - mu))
        except np.linalg.LinAlgError as exc:
            raise Unestimable("the logistic model's information is singular") from exc
        for _half in range(40):
            trial = beta + step
            new = loglik(trial)
            if new >= ll - 1e-12 * max(1.0, abs(ll)):
                break
            step = step / 2.0
        beta, ll = trial, new
        if not np.all(np.isfinite(beta)) or np.max(np.abs(beta) * sd) > 30:
            raise Separated("the logistic likelihood has no maximum (an outcome level is "
                            "separated by the covariates)")
        if np.max(np.abs(step)) < tol * max(1.0, float(np.max(np.abs(beta)))):
            return beta
    raise Unestimable("the logistic model did not converge")


def _null_relations(X: np.ndarray, kept: Sequence[int]) -> np.ndarray:
    """For each column left out of ``kept``, its coefficients on the kept columns (rows of the
    result, one per left-out column, ``len(kept)`` wide): the linear relation that aliases it."""
    out = [j for j in range(X.shape[1]) if j not in set(kept)]
    if not out:
        return np.zeros((0, len(kept)))
    A, *_ = np.linalg.lstsq(X[:, list(kept)], X[:, out], rcond=None)
    return A.T


def separated_rows(X: np.ndarray, y: np.ndarray, tol: float = 1e-7) -> tuple[np.ndarray, np.ndarray] | None:
    """The rows whose fitted probability is 0 or 1 at the logistic likelihood's supremum, and a
    direction that separates them; None when nothing is separated.

    Konis (2007): the data are separated when some ``d`` has ``s_i x_iᵀd ≥ 0`` for every row
    (``s_i = ±1`` for an event or not) and ``> 0`` for some. The set of rows any such direction
    separates is found by linear programs, each maximizing the margins of the rows not yet found
    (with the columns scaled to at most one and ``d`` boxed in [−1, 1]); the sum of the directions
    found separates them all at once, and the rows it leaves at a zero margin are separated by
    none (every other direction would separate one more)."""
    from scipy.optimize import linprog

    X = np.asarray(X, dtype=float)
    s = np.where(np.asarray(y, dtype=float) > 0.5, 1.0, -1.0)
    n, P = X.shape
    scale = np.abs(X).max(axis=0)
    scale[scale == 0] = 1.0
    Xs = X / scale
    A = -(s[:, None] * Xs)
    found = np.zeros(n, dtype=bool)
    total = np.zeros(P)
    for _ in range(n):
        c = -(s[~found] @ Xs[~found])
        result = linprog(c, A_ub=A, b_ub=np.zeros(n), bounds=[(-1.0, 1.0)] * P, method="highs")
        if result.status != 0 or result.x is None:
            break
        d = np.where(np.abs(result.x) < 1e-12, 0.0, result.x)
        margins = s * (Xs @ d)
        if margins.min() < -tol:
            break
        new = (margins > tol) & ~found
        if not new.any():
            break
        found |= new
        total += d
    if not found.any():
        return None
    margins = s * (Xs @ total)
    if margins[found].min() <= tol or margins.min() < -tol:
        return None
    return found, total / scale


@dataclass
class LogisticFit:
    """A logistic model's fitted probabilities, for any rows (:meth:`predict`).

    ``kept`` are the columns R's ``glm`` keeps (an aliased column's coefficient is NA there), and a
    row is predicted from them when it obeys the relations that alias the others, as every row of
    the data does; otherwise its prediction is not identified. At the likelihood's supremum
    (``limit``: separation), the separated direction ``direction`` sends a row's probability to 1
    or 0 by the sign of its ``xᵀd``, and the rest of the model is the maximum-likelihood fit on the
    rows it does not separate (on ``basis``, an orthonormal basis of their span). A row outside
    that span is decided by ``d`` only when the span leaves one direction free; otherwise its limit
    depends on the path, and it is refused as not identified."""

    kept: list[int]
    relations: np.ndarray
    beta: np.ndarray  # on the kept columns (the finite fit); on ``basis`` coordinates at the limit
    limit: bool = False
    basis: np.ndarray | None = None  # kept-columns × r: the span of the unseparated rows
    direction: np.ndarray | None = None  # on the kept columns
    n_separated: int = 0

    def _check(self, X: np.ndarray, Z: np.ndarray) -> None:
        if not len(self.relations):
            return
        out = [j for j in range(X.shape[1]) if j not in set(self.kept)]
        gap = X[:, out] - Z @ self.relations.T
        size = np.maximum(np.abs(X[:, out]), np.abs(Z).max(axis=1, initial=0.0)[:, None])
        if np.any(np.abs(gap) > 1e-8 * np.maximum(size, 1.0)):
            raise Unestimable("a row is outside the span of the rows the model was fit on, so its "
                              "prediction is not identified")

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        Z = X[:, self.kept]
        self._check(X, Z)
        if not self.limit:
            return _expit(Z @ self.beta)
        assert self.basis is not None and self.direction is not None
        inside = Z @ self.basis
        # The part of each row the unseparated rows do not span decides it, if anything does.
        outside = Z - inside @ self.basis.T
        lean = Z @ self.direction
        size = np.maximum(np.abs(Z).max(axis=1, initial=0.0), 1.0)
        spanned = np.abs(outside).max(axis=1, initial=0.0) <= 1e-8 * size
        p = np.empty(len(Z))
        p[spanned] = _expit(inside[spanned] @ self.beta)
        rest = ~spanned
        if rest.any():
            if self.basis.shape[1] + 1 != self.basis.shape[0]:
                raise Unestimable("the separated likelihood's limit is not identified for the "
                                  "rows set here")
            tilt = lean[rest]
            if np.any(np.abs(tilt) <= 1e-8 * size[rest] * np.abs(self.direction).max()):
                raise Unestimable("the separated likelihood's limit is not identified for the "
                                  "rows set here")
            p[rest] = (tilt > 0).astype(float)
        return p


def logistic_fit(X: np.ndarray, y: np.ndarray, *, limit: bool = False) -> LogisticFit:
    """The logistic model's maximum-likelihood fit on the columns ``lm``'s rule keeps
    (:func:`column_basis`), as R's ``glm`` fits an aliased design. Under separation it raises
    :class:`Separated`, unless ``limit``: then the fit is the likelihood's supremum (the separated
    rows' probabilities 0 or 1, :func:`separated_rows`), the fit R's ``glm`` approaches as its
    coefficients run away, for a bootstrap resample whose outcome happens to be separated."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    kept = column_basis(X)
    relations = _null_relations(X, kept)
    Z = X[:, kept]
    try:
        return LogisticFit(kept=kept, relations=relations, beta=logistic_mle(Z, y))
    except Unestimable as exc:
        # A runaway coefficient, or an information made singular by one, is separation only if a
        # separating direction exists; anything else is not.
        if not limit:
            raise
        found = separated_rows(Z, y)
        if found is None:
            raise exc
    separated, direction = found
    rest = ~separated
    if rest.any():
        Zr = Z[rest]
        _, sv, vt = np.linalg.svd(Zr, full_matrices=False)
        rank = int(np.sum(sv > 1e-10 * sv[0])) if len(sv) else 0
        basis = vt[:rank].T
        gamma = logistic_mle(Zr @ basis, y[rest]) if rank else np.zeros(0)
    else:
        basis, gamma = np.zeros((Z.shape[1], 0)), np.zeros(0)
    return LogisticFit(kept=kept, relations=relations, beta=gamma, limit=True, basis=basis,
                       direction=direction, n_separated=int(separated.sum()))


def standardized_risks(beta: np.ndarray | LogisticFit, X0: np.ndarray,
                       X1: np.ndarray) -> tuple[float, float]:
    """The mean predicted risk with every row set to the first setting and to the second."""
    if isinstance(beta, LogisticFit):
        return float(np.mean(beta.predict(X0))), float(np.mean(beta.predict(X1)))
    return (float(np.mean(_expit(np.asarray(X0, float) @ beta))),
            float(np.mean(_expit(np.asarray(X1, float) @ beta))))


@dataclass(frozen=True)
class Setting:
    """One contrast of the exposure: the values every row is set to (``low`` → ``high``), or a
    shift of every row's own value by ``shift`` units (``low``/``high`` then None)."""

    label: str  # "one unit higher than observed" | "`1` against `0`" | "`b` against `a`"
    low: Any = None
    high: Any = None
    shift: float | None = None


def settings_of(values: pd.Series, exposure: str) -> list[Setting]:
    """The contrasts g-computation reports for an exposure, from its values: each level against
    the first for a categorical one; the higher value against the lower for a two-valued number;
    one unit higher than observed for a continuous one (the per-unit scale of its odds ratio)."""
    present = values.dropna()
    if present.empty:
        raise Unestimable(f"`{exposure}` has no recorded value")
    numeric = pd.api.types.is_numeric_dtype(present) and not pd.api.types.is_bool_dtype(present)
    if not numeric:
        levels = sorted(present.astype(str).unique())
        if len(levels) < 2:
            raise Unestimable(f"`{exposure}` takes one value only")
        return [Setting(f"`{lv}` against `{levels[0]}`", levels[0], lv) for lv in levels[1:]]
    distinct = np.unique(present.to_numpy(dtype=float))
    if len(distinct) == 2:
        lo, hi = float(distinct[0]), float(distinct[1])
        return [Setting(f"`{_num(hi)}` against `{_num(lo)}`", lo, hi)]
    return [Setting("one unit higher than observed", shift=1.0)]


def _num(x: float) -> str:
    return f"{x:g}"


def counterfactual(X: pd.DataFrame, exposure: str, setting: Setting, *, side: str,
                   energy: tuple[str, float] | None = None) -> pd.DataFrame:
    """``X`` with the exposure set as ``setting``'s ``side`` ("low" or "high") says. ``energy``
    (total energy's column, kcal per unit of the exposure) moves total energy with a shifted
    exposure: an addition's calories, where the energy model derives other energy from the total
    (the partition model)."""
    out = X.copy()
    if setting.shift is not None:
        if side == "high":
            out[exposure] = out[exposure].astype(float) + setting.shift
            if energy is not None:
                column, factor = energy
                out[column] = out[column].astype(float) + factor * setting.shift
        return out
    value = setting.high if side == "high" else setting.low
    if pd.api.types.is_numeric_dtype(X[exposure]) and not isinstance(value, str):
        out[exposure] = float(value)
    else:
        out[exposure] = pd.Series([value] * len(out), index=out.index, dtype=X[exposure].dtype)
    return out


@dataclass
class Standardized:
    """The marginal risk difference and risk ratio of one contrast, with bootstrap intervals."""

    setting: str
    risk_low: float  # the mean predicted risk with every row at the contrast's first setting
    risk_high: float  # … and at its second
    rd: float
    rr: float
    rd_low: float | None = None
    rd_high: float | None = None
    rr_low: float | None = None
    rr_high: float | None = None  # +inf when the low risk was 0 in more than 2.5% of resamples
    n_boot: int = 0
    # Resamples whose outcome the covariates separated: kept, at the likelihood's limit (the
    # separated rows' risks 0 or 1), the fit R's ``glm`` approaches; leaving them out would cut
    # the extreme resamples from the interval.
    n_limit: int = 0
    n_failed: int = 0  # resamples whose model could not be fit at all, left out (and said so)
    failed_reason: str | None = None  # the first such resample's reason
    n_rr_infinite: int = 0  # resamples whose low risk was 0 (the separated limit): an infinite ratio
    n_rr_undefined: int = 0  # resamples with both risks 0: no risk ratio, left out of its interval
    by_unit: str | None = None  # the column whose units the bootstrap resampled whole

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


Fitter = Callable[[pd.DataFrame, np.ndarray], Callable[[pd.DataFrame], np.ndarray]]


def resample_indices(n: int, n_boot: int, seed: int, units: np.ndarray | None = None) -> list[np.ndarray]:
    """The bootstrap's resamples (row positions): rows drawn with replacement, or whole units
    (every row of a drawn unit, as often as the unit is drawn) when ``units`` is given. One
    ``numpy.random.default_rng(seed)`` stream, so the resamples replay."""
    rng = np.random.default_rng(seed)
    if units is None:
        return [rng.integers(0, n, n) for _ in range(n_boot)]
    codes, uniques = pd.factorize(pd.Series(units), use_na_sentinel=False)
    groups = [np.flatnonzero(codes == g) for g in range(len(uniques))]
    G = len(groups)
    return [np.concatenate([groups[g] for g in rng.integers(0, G, G)]) for _ in range(n_boot)]


def percentile(values: Sequence[float], level: float = LEVEL) -> tuple[float | None, float | None]:
    """The percentile interval (R's ``quantile`` type 7, numpy's default). An infinite value (a
    risk ratio over a zero risk) is an order statistic like any other: a limit that falls on or
    interpolates toward it is infinite. Undefined values (NaN) are left out."""
    v = np.sort(np.asarray([x for x in values if x is not None and not math.isnan(x)], dtype=float))
    if len(v) < 2:
        return None, None
    a = (1.0 - level) / 2.0
    if np.all(np.isfinite(v)):
        return float(np.quantile(v, a)), float(np.quantile(v, 1.0 - a))

    def type7(p: float) -> float:
        h = (len(v) - 1) * p
        lo = int(math.floor(h))
        hi = min(lo + 1, len(v) - 1)
        frac = h - lo
        if frac == 0 or v[lo] == v[hi]:
            return float(v[lo])
        if not (math.isfinite(v[lo]) and math.isfinite(v[hi])):
            return float(v[hi]) if math.isinf(v[hi]) else float(v[lo])
        return float(v[lo] + frac * (v[hi] - v[lo]))

    return type7(a), type7(1.0 - a)


def g_computation(fit: Fitter, X: pd.DataFrame, y: np.ndarray, exposure: str, setting: Setting, *,
                  n_boot: int = BOOT, seed: int = 0, units: np.ndarray | None = None,
                  unit_column: str | None = None, energy: tuple[str, float] | None = None,
                  progress: Callable[[int, int], None] | None = None,
                  cancelled: Callable[[], bool] | None = None) -> Standardized:
    """Marginal standardization of a logistic model over the analyzed rows ``X``.

    ``fit(X, y)`` fits the whole pipeline on the rows given and returns ``matrix_of``: the model
    matrix (with its intercept column) of any rows set as they like, through the steps fitted on
    those rows. The point estimate standardizes over ``X`` (separated rows are refused: the
    logistic model has no estimate). Each bootstrap resample refits the whole chain on the
    resample and standardizes over it; a resample whose outcome happens to be separated is kept
    at the likelihood's limit (:func:`logistic_fit`), and one that cannot be fit at all is left
    out and counted. ``n_boot = 0``: the point estimate alone."""
    def estimate(rows_X: pd.DataFrame, rows_y: np.ndarray, limit: bool) -> tuple[float, float, bool]:
        matrix_of = fit(rows_X, rows_y)
        model = logistic_fit(matrix_of(rows_X), rows_y, limit=limit)
        low = matrix_of(counterfactual(rows_X, exposure, setting, side="low", energy=energy))
        high = matrix_of(counterfactual(rows_X, exposure, setting, side="high", energy=energy))
        return (*standardized_risks(model, low, high), model.limit)

    y = np.asarray(y, dtype=float)
    r0, r1, _ = estimate(X, y, False)
    out = Standardized(setting=setting.label, risk_low=r0, risk_high=r1, rd=r1 - r0,
                       rr=r1 / r0 if r0 > 0 else float("nan"), by_unit=unit_column)
    if n_boot <= 0:
        return out
    rds: list[float] = []
    rrs: list[float] = []
    failed = at_limit = undefined = infinite = 0
    reason = None
    draws = resample_indices(len(X), n_boot, seed, units)
    for b, idx in enumerate(draws):
        if cancelled is not None and cancelled():
            raise InterruptedError("cancelled")
        try:
            b0, b1, limit = estimate(X.iloc[idx].reset_index(drop=True), y[idx], True)
        except (Unestimable, ValueError, np.linalg.LinAlgError) as exc:
            failed += 1
            reason = reason or str(exc)
            continue
        at_limit += int(limit)
        rds.append(b1 - b0)
        if b0 > 0:
            rrs.append(b1 / b0)
        elif b1 > 0:
            rrs.append(math.inf)
            infinite += 1
        else:
            undefined += 1
        if progress is not None:
            progress(b + 1, n_boot)
    out.rd_low, out.rd_high = percentile(rds)
    out.rr_low, out.rr_high = percentile(rrs)
    out.n_boot, out.n_failed, out.n_limit = n_boot, failed, at_limit
    out.failed_reason, out.n_rr_undefined, out.n_rr_infinite = reason, undefined, infinite
    return out


# ── the Table 2 display ──────────────────────────────────────────────────────


def split_rows(rows: Iterable[Mapping[str, Any]], effects: Iterable[str],
               modifiers: Iterable[str] = ()) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(the exposure's rows, the adjustment terms). ``effects`` are the model-matrix columns that
    carry the exposure's effect (its own terms and its spline or quintile terms); every other row
    is an adjustment term, the intercept and each declared modifier's main effect included, each
    with why it is no effect estimate (Westreich & Greenland 2013)."""
    wanted = {str(f) for f in effects}
    modifying = {str(m) for m in modifiers}
    shown: list[dict[str, Any]] = []
    appendix: list[dict[str, Any]] = []
    for row in rows:
        feature = str(row.get("feature"))
        if feature in wanted:
            shown.append(dict(row))
            continue
        if feature.startswith("(intercept)"):
            why = "the model's baseline, not an effect"
        elif any(feature == m or feature.startswith(f"{m}_") for m in modifying):
            why = "a modifier's main effect: the effect of what you study is read within its levels"
        else:
            why = ("an adjustment term: a direct effect at best, and possibly confounded even "
                   "where the estimate of what you study is not")
        appendix.append({**dict(row), "why": why})
    return shown, appendix


# ── diagnostics: proportional hazards ────────────────────────────────────────


def time_scale(y: np.ndarray, transform: str = "km") -> np.ndarray:
    """Each row's time on the scale the proportional-hazards test uses: ``km``, one minus the
    left-continuous Kaplan–Meier estimate at its time (R's ``cox.zph`` default, counting-process
    risk sets under delayed entry); ``rank`` (average ranks of every row's time); ``identity``;
    ``log``."""
    from scipy.stats import rankdata

    t, e, s = y["time"], y["event"].astype(bool), y["entry"]
    if transform == "identity":
        return t.astype(float)
    if transform == "log":
        return np.log(t.astype(float))
    if transform == "rank":
        return rankdata(t, method="average")
    if transform != "km":
        raise ValueError(f"unknown time transform {transform!r}")
    times = np.unique(t[e])
    t_sorted, s_sorted = np.sort(t), np.sort(s)
    at_risk = (len(t) - np.searchsorted(t_sorted, times, side="left")) - (
        len(s) - np.searchsorted(s_sorted, times, side="left"))
    deaths = np.bincount(np.searchsorted(times, t[e]), minlength=len(times)).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        step = np.where(at_risk > 0, 1.0 - deaths / at_risk, 1.0)
    surv = np.cumprod(step)
    before = np.searchsorted(times, t, side="left")  # event times strictly before each row's time
    return 1.0 - np.concatenate([[1.0], surv])[before]


def cox_zph(X: np.ndarray, y: np.ndarray, beta: np.ndarray,
            terms: Mapping[str, Sequence[int]] | None = None, transform: str = "km") -> dict[str, Any]:
    """The proportional-hazards score tests at the fitted ``beta`` (Efron ties), per term and
    global, as R's ``survival::cox.zph(fit, transform)`` computes them: for the expanded model with
    covariates ``(x, x·g(t))``, ``u = (0, Σ_events g_i r_i)`` (``r_i`` the Schoenfeld residuals),
    the information ``I`` of the expanded model at ``(β̂, 0)``, and ``u_kᵀ I_kk⁻¹ u_k`` over all of
    β and the term's γ (the global test over every γ). ``terms`` maps a term's name to its columns
    (default: one term per column)."""
    from scipy.stats import chi2

    from turbotab.core.models.survival import _at_risk_sums, _risk, schoenfeld_residuals

    X = np.asarray(X, dtype=float)
    n, P = X.shape
    if not np.any(y["event"]):
        raise Unestimable("no row has the event")
    rk = _risk(X, y, np.asarray(beta, dtype=float))
    scale = time_scale(y, transform)
    events = y["event"].astype(bool)
    g = scale - float(np.mean(scale[events]))
    g_slot = g[rk.order]
    eta = X @ beta
    r = np.exp(eta - (float(eta.max()) if len(eta) else 0.0))
    xbar = rk.S1 / rk.S0[:, None]

    def information(w: np.ndarray) -> np.ndarray:
        c = _at_risk_sums(y, rk, w / rk.S0, w * (1.0 - rk.frac) / rk.S0)
        out = (X * (r * c)[:, None]).T @ X - (xbar * w[:, None]).T @ xbar
        return (out + out.T) / 2.0

    I_bb = information(np.ones(len(rk.order)))
    I_bg = information(g_slot)
    I_gg = information(g_slot ** 2)
    imat = np.block([[I_bb, I_bg], [I_bg, I_gg]])
    resid, order = schoenfeld_residuals(X, y, beta)
    U = (resid * g[order][:, None]).sum(axis=0)
    if terms is None:
        terms = {str(j): [j] for j in range(P)}
    rows = []
    for name, cols in terms.items():
        cols = [int(c) for c in cols]
        kk = list(range(P)) + [P + c for c in cols]
        u = np.concatenate([np.zeros(P), U[cols]])
        stat = float(np.linalg.solve(imat[np.ix_(kk, kk)], u) @ u)
        rows.append({"term": str(name), "chisq": stat, "df": len(cols),
                     "p": float(chi2.sf(stat, len(cols)))})
    u = np.concatenate([np.zeros(P), U])
    total = float(np.linalg.solve(imat, u) @ u)
    return {"terms": rows, "global": {"term": "GLOBAL", "chisq": total, "df": P,
                                      "p": float(chi2.sf(total, P))},
            "transform": transform, "n_events": int(events.sum())}


def split_follow_up(y: np.ndarray, cut: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Episode splitting at ``cut`` (R's ``survSplit``): a row followed past ``cut`` becomes two,
    ``(entry, cut]`` without the event and ``(max(entry, cut), time]`` with its own; returns
    (the split outcome, each new row's source row, whether it is in the later period)."""
    from turbotab.core.models.survival import survival_outcome

    t, e, s = y["time"], y["event"], y["entry"]
    early = t <= cut
    late_only = s >= cut
    both = ~early & ~late_only
    src = np.concatenate([np.flatnonzero(~both), np.flatnonzero(both), np.flatnonzero(both)])
    later = np.concatenate([late_only[~both], np.zeros(both.sum(), bool), np.ones(both.sum(), bool)])
    time = np.concatenate([t[~both], np.full(both.sum(), cut), t[both]])
    entry = np.concatenate([s[~both], s[both], np.full(both.sum(), cut)])
    event = np.concatenate([e[~both], np.zeros(both.sum(), bool), e[both]])
    order = np.argsort(src, kind="stable")
    return survival_outcome(event[order], time[order], entry[order]), src[order], later[order]


def period_hazard_ratios(X: np.ndarray, y: np.ndarray, exposure: int, cut: float) -> dict[str, Any]:
    """The exposure's hazard ratio before and after ``cut``: the Cox model refit on the follow-up
    split at ``cut`` with the exposure's product with the later period added (a time-varying
    coefficient, two steps), each period's log hazard ratio and its Wald interval from the
    model's information. The split rows of one person never overlap in time, so the partial
    likelihood is the time-varying-covariate one and needs no clustering."""
    from scipy import stats

    from turbotab.core.models.survival import cox_fit

    X = np.asarray(X, dtype=float)
    split, src, later = split_follow_up(y, cut)
    Xs = X[src]
    product = Xs[:, exposure] * later
    design = np.column_stack([Xs, product])
    fit = cox_fit(design, split)
    if not fit.converged:
        raise Unestimable("the split model's likelihood has no maximum")
    P = X.shape[1]
    q = float(stats.norm.ppf(0.5 + LEVEL / 2))
    out = []
    for label, c in (("before", np.eye(P + 1)[exposure]),
                     ("after", np.eye(P + 1)[exposure] + np.eye(P + 1)[P])):
        b = float(c @ fit.beta)
        se = float(math.sqrt(max(c @ fit.cov @ c, 0.0)))
        out.append({"period": label, "estimate": b, "se": se, "ci_low": b - q * se,
                    "ci_high": b + q * se, "ratio": math.exp(b), "ratio_low": math.exp(b - q * se),
                    "ratio_high": math.exp(b + q * se)})
    return {"cut": float(cut), "periods": out, "n_split_rows": int(len(split)),
            "n_events": int(fit.n_events)}


# ── diagnostics: influence ───────────────────────────────────────────────────


def _hat(Xw: np.ndarray, kept: Sequence[int] | None = None) -> np.ndarray:
    """The hat matrix's diagonal: each row's squared length in an orthonormal basis of the span of
    the columns ``lm`` keeps (:func:`column_basis`), so an aliased column, wherever it sits, leaves
    the span (and the leverage) as R's ``hatvalues`` has it; a leverage within
    :data:`LEVERAGE_ONE` of 1 is 1, as R's ``lm.influence`` rounds it."""
    Xw = np.asarray(Xw, dtype=float)
    kept = column_basis(Xw) if kept is None else list(kept)
    Q, _ = np.linalg.qr(Xw[:, kept])
    h = np.einsum("ij,ij->i", Q, Q)
    h[h > 1.0 - LEVERAGE_ONE] = 1.0
    return h


def _cooks(scaled: np.ndarray, h: np.ndarray, p: int) -> np.ndarray:
    """``(r_i / (1 − h_i))² h_i / p`` for the scaled residuals ``r_i``; NaN where the leverage is
    1, as R's ``cooks.distance`` reports it (``res[is.infinite(res)] <- NaN``)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        D = (scaled / (1.0 - h)) ** 2 * h / p
    D[h >= 1.0] = np.nan
    return D


def ols_influence(X: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray | float]:
    """Each row's leverage ``h_i`` and Cook's distance ``D_i = e_i² h_i / (p s² (1 − h_i)²)``,
    ``s²`` the residual variance on n − p and p the rank, as R's ``hatvalues`` and
    ``cooks.distance`` give them for ``lm``; ``X`` carries its intercept column. A row with
    leverage 1 has no distance (NaN), as in R."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    kept = column_basis(X)
    h = _hat(X, kept)
    p = len(kept)
    beta = np.linalg.lstsq(X[:, kept], y, rcond=None)[0]
    e = y - X[:, kept] @ beta
    s2 = float(e @ e) / (len(y) - p)
    D = _cooks(e / np.sqrt(s2), h, p)
    return {"leverage": h, "cooks": D, "p": p, "n": len(y),
            "leverage_one": int(np.sum(h >= 1.0)), "aliased": X.shape[1] - p}


def logistic_influence(X: np.ndarray, y: np.ndarray, beta: np.ndarray | None = None) -> dict[str, Any]:
    """A logistic model's leverage (the diagonal of ``W½ X (XᵀWX)⁻¹ Xᵀ W½``) and Cook's distance
    ``D_i = (r_i / (1 − h_i))² h_i / p`` from the Pearson residuals ``r_i`` (dispersion 1), as R's
    ``hatvalues`` and ``cooks.distance`` give them for a binomial ``glm``: an aliased column is
    left out of the fit (its coefficient NA in R), p is the rank, and a row with leverage 1 has no
    distance (NaN)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    kept = column_basis(X)
    Z = X[:, kept]
    mu = _expit(Z @ logistic_mle(Z, y)) if beta is None else _expit(X @ np.asarray(beta, float))
    w = mu * (1.0 - mu)
    h = _hat(Z * np.sqrt(w)[:, None], list(range(Z.shape[1])))
    p = len(kept)
    pearson = (y - mu) / np.sqrt(w)
    D = _cooks(pearson, h, p)
    return {"leverage": h, "cooks": D, "p": p, "n": len(y),
            "leverage_one": int(np.sum(h >= 1.0)), "aliased": X.shape[1] - p}


def cook_threshold(p: int, n: int) -> float:
    """The median of F(p, n − p): Cook's (1977) reference for a distance that moves the estimate to
    the edge of its 50% confidence region."""
    from scipy.stats import f

    return float(f.ppf(0.5, p, max(n - p, 1)))


# ── sensitivity to unmeasured confounding ────────────────────────────────────

Measure = Literal["RR", "OR", "HR", "OLS"]


def e_value_rr(rr: float, true: float = 1.0) -> float:
    """VanderWeele & Ding's E-value of a risk ratio: ``RR + √(RR (RR − 1))`` for RR ≥ 1, of 1/RR
    below it (EValue's ``threshold``)."""
    x, t = float(rr), float(true)
    if not math.isfinite(x) or x <= 0:
        return float("nan")
    if x <= 1:
        x, t = 1.0 / x, 1.0 / t
    if t <= x:
        return (x + math.sqrt(x * (x - t))) / t
    ratio = t / x
    return ratio + math.sqrt(ratio * (ratio - 1.0))


def to_risk_ratio(value: float, measure: Measure, *, rare: bool | None = None,
                  sd: float | None = None, delta: float = 1.0) -> float:
    """EValue's conversion to the risk-ratio scale: an odds ratio of a common outcome by its square
    root; a hazard ratio of a common outcome by ``(1 − 0.5^√HR) / (1 − 0.5^√(1/HR))``; a linear
    coefficient as the standardized difference ``d = β δ / sd`` by ``exp(0.91 d)``; a rare outcome's
    ratio as it is."""
    v = float(value)
    if measure == "RR":
        return v
    if measure == "OR":
        if rare is None:
            raise ValueError("say whether the outcome is rare")
        return v if rare else math.sqrt(v)
    if measure == "HR":
        if rare is None:
            raise ValueError("say whether the outcome is rare")
        return v if rare else (1.0 - 0.5 ** math.sqrt(v)) / (1.0 - 0.5 ** math.sqrt(1.0 / v))
    if measure == "OLS":
        if sd is None or not sd > 0:
            raise ValueError("a linear coefficient needs the outcome's standard deviation")
        return math.exp(0.91 * v * delta / sd)
    raise ValueError(f"unknown measure {measure!r}")


def e_values(estimate: float, low: float | None = None, high: float | None = None, *,
             measure: Measure, rare: bool | None = None, sd: float | None = None,
             se: float | None = None, delta: float = 1.0) -> dict[str, Any]:
    """The E-value for the estimate and for the confidence limit nearer the null (1 when the
    interval includes the null), as R's ``EValue::evalues.RR/OR/HR/OLS`` report them. For a linear
    coefficient the interval is ``exp(0.91 d ± 1.78 se_d)`` from its standard error ``se``."""
    if measure == "OLS":
        if sd is None or not sd > 0:
            raise ValueError("a linear coefficient needs the outcome's standard deviation")
        d = float(estimate) * delta / sd
        rr = math.exp(0.91 * d)
        lo = hi = None
        if se is not None and math.isfinite(se):
            sd_d = float(se) * delta / sd
            lo, hi = math.exp(0.91 * d - 1.78 * sd_d), math.exp(0.91 * d + 1.78 * sd_d)
    else:
        rr = to_risk_ratio(estimate, measure, rare=rare)
        lo = to_risk_ratio(low, measure, rare=rare) if low is not None else None
        hi = to_risk_ratio(high, measure, rare=rare) if high is not None else None
    point = e_value_rr(rr)
    limit = None
    crosses = None
    if rr > 1 and lo is not None:
        crosses = lo < 1
        limit = 1.0 if crosses else e_value_rr(lo)
    elif rr < 1 and hi is not None:
        crosses = hi > 1
        limit = 1.0 if crosses else e_value_rr(hi)
    elif rr == 1:
        limit, crosses = 1.0, True
    return {"measure": measure, "rr": rr, "rr_low": lo, "rr_high": hi, "point": point,
            "limit": limit, "interval_includes_null": crosses, "rare": rare,
            "converted": measure != "RR" and not (measure in ("OR", "HR") and rare)}


def outcome_sd(values: np.ndarray, weights: np.ndarray | None = None) -> float:
    """The outcome's standard deviation an E-value of a difference standardizes by
    (MODELING_SEQUENCE §0 ruling 14: the SD the estimand speaks of, the same in the effects stage,
    the causal lane and the time-varying lane): :func:`estimand_sd`'s value without its basis.
    With ``weights`` (the surveyed-population answer's design weights at the analyzed rows) the
    population's, as R survey's ``svyvar`` estimates it; else the rows' (n − 1)."""
    return estimand_sd(values, weights)[0]


def partial_r2(t: float, dof: float) -> float:
    """The partial R² of a coefficient with the outcome from its t-statistic, ``t² / (t² + dof)``."""
    return float(t) ** 2 / (float(t) ** 2 + float(dof))


def group_partial_r2(F: float, p: int, dof: float) -> float:
    """A group of p coefficients' partial R² from their Wald F, ``F p / (F p + dof)``."""
    return float(F) * p / (float(F) * p + float(dof))


def robustness_value(t: float, dof: float, *, q: float = 1.0, alpha: float = 1.0) -> float:
    """Cinelli & Hazlett's robustness value ``RV_{q,α}``: the partial R² that a confounder must have
    with both the exposure and the outcome to bring the estimate to (1 − q) of itself (α = 1), or
    its interval to include it (α < 1); sensemakr's ``robustness_value``, its extreme-value branch
    included."""
    from scipy import stats

    fq = q * abs(float(t) / math.sqrt(dof))
    f_crit = abs(float(stats.t.ppf(alpha / 2.0, dof - 1))) / math.sqrt(dof - 1) if alpha < 1 else 0.0
    fqa = fq - f_crit
    if fqa < 0:
        return 0.0
    if fqa == 0:
        return 0.0
    rv = 2.0 / (1.0 + math.sqrt(1.0 + 4.0 / fqa ** 2))
    if f_crit > 0 and fq > 1.0 / f_crit:
        xrv = (fq ** 2 - f_crit ** 2) / (1.0 + fq ** 2)
        return max(xrv, 0.0)
    return rv


def benchmark_bounds(r2dxj: float, r2yxj: float, *, kd: float = 1.0,
                     ky: float | None = None) -> tuple[float, float]:
    """The bounds on a confounder's partial R² with the exposure (``r2dz.x``) and with the outcome
    (``r2yz.dx``) if it were ``kd`` (``ky``) times as strong as the benchmark covariate, from the
    benchmark's own partial R² with the exposure (``r2dxj``) and with the outcome (``r2yxj``):
    sensemakr's ``ovb_partial_r2_bound``."""
    ky = kd if ky is None else ky
    r2dz = kd * (r2dxj / (1.0 - r2dxj))
    if r2dz >= 1:
        raise Unestimable("a common cause that strong would explain all of what you study")
    r2zxj = kd * r2dxj ** 2 / ((1.0 - kd * r2dxj) * (1.0 - r2dxj))
    if r2zxj >= 1:
        raise Unestimable("a common cause that strong is impossible here")
    r2yz = ((math.sqrt(ky) + math.sqrt(r2zxj)) / math.sqrt(1.0 - r2zxj)) ** 2 * (r2yxj / (1.0 - r2yxj))
    return float(r2dz), float(min(r2yz, 1.0))


def adjusted_for_confounder(estimate: float, se: float, dof: float, r2dz: float, r2yz: float, *,
                            alpha: float = 0.05) -> dict[str, float]:
    """The estimate, standard error, t and interval once a confounder with these partial R² values
    is adjusted for, its bias taken toward zero (sensemakr's ``adjusted_estimate``,
    ``adjusted_se``, ``adjusted_t`` and ``ovb_bounds``' interval on t(dof))."""
    from scipy import stats

    bias = math.sqrt(r2yz * r2dz / (1.0 - r2dz)) * se * math.sqrt(dof)
    new = math.copysign(1.0, estimate) * (abs(estimate) - bias)
    new_se = math.sqrt(1.0 - r2yz) / math.sqrt(1.0 - r2dz) * se * math.sqrt(dof / (dof - 1.0))
    q = float(stats.t.ppf(1.0 - alpha / 2.0, dof))
    # A bound clipped at r2yz = 1 (sensemakr warns and sets it to 1) leaves no residual variance:
    # the adjusted standard error is 0 and the interval is the point, as sensemakr reports it.
    t = new / new_se if new_se > 0 else math.copysign(math.inf, new)
    return {"estimate": new, "se": new_se, "t": t, "ci_low": new - q * new_se,
            "ci_high": new + q * new_se}


@dataclass
class Benchmark:
    covariate: str
    r2dxj: float  # the benchmark's partial R² with the exposure, given the other covariates
    r2yxj: float  # … with the outcome, given the exposure and the other covariates
    r2dz: float  # the bound on a confounder k times as strong: with the exposure
    r2yz: float  # … and with the outcome
    estimate: float  # the estimate once such a confounder is adjusted for (algebraic: no se)
    # The adjusted standard error and interval rest on the classical standard error; None when the
    # interval reported is not the classical one (:data:`CLASSICAL`).
    se: float | None
    ci_low: float | None
    ci_high: float | None
    kd: float = 1.0


# The covariances under which the reported interval is the classical least-squares one that
# sensemakr's interval algebra assumes: ``lm``'s own (``model`` is the inference tables' name for
# it, as the feature-wise family reports it).
CLASSICAL = ("classical", "model")


@dataclass
class LinearSensitivity:
    """The Cinelli–Hazlett analysis of one least-squares coefficient.

    What rests on the point estimate alone is algebraic, whatever standard error the reported
    interval uses: the partial R², RV_q, and each benchmark's bounds and adjusted estimate (the
    omitted-variable bias of the least-squares coefficient; Cinelli & Hazlett 2020, §4). RV_{q,α}
    and the benchmarks' adjusted intervals rest on the classical standard error; when the reported
    interval is not classical (HC3, cluster-robust), they would describe an interval nobody was
    shown, so they are not reported and ``interval_note`` says why."""

    exposure: str
    estimate: float
    se: float  # classical, as ``lm`` reports it: the algebra's, not necessarily the one shown
    t: float
    dof: float
    partial_r2: float  # the exposure's partial R² with the outcome
    rv: float  # RV_q (q = 1): enough to bring the estimate to zero
    rv_alpha: float | None  # RV_{q,α} (α = 0.05): enough to bring the interval to include zero
    benchmarks: list[Benchmark] = field(default_factory=list)
    alpha: float = 0.05
    covariance: str = "classical"  # the reported interval's covariance
    interval_note: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    dof = float(len(y) - np.linalg.matrix_rank(X))
    s2 = float(e @ e) / dof
    cov = s2 * np.linalg.pinv(X.T @ X)
    return beta, cov, dof


def linear_sensitivity(matrix: pd.DataFrame, y: np.ndarray, exposure: str,
                       benchmarks: Mapping[str, Sequence[str]] | None = None, *,
                       kd: float = 1.0, alpha: float = 0.05, covariance: str = "classical",
                       shown_se: float | None = None) -> LinearSensitivity:
    """The robustness value and partial R² of the coefficient of ``exposure`` (one column of the
    model matrix, which carries no intercept: one is added) in the least-squares model of ``y``,
    and for each benchmark (a name and the matrix columns it is made of: one, or a categorical
    covariate's indicators) the bounds and adjusted estimate for a confounder ``kd`` times as
    strong, as R's ``sensemakr(model, treatment, benchmark_covariates, kd)`` reports them.

    ``covariance`` is the reported interval's (``shown_se`` its standard error): unless it is the
    classical one (:data:`CLASSICAL`), RV_{q,α} and the benchmarks' adjusted intervals are not
    reported (:class:`LinearSensitivity`)."""
    cols = [str(c) for c in matrix.columns]
    if exposure not in cols:
        raise ValueError(f"`{exposure}` is not a column of the model matrix")
    X = np.column_stack([np.ones(len(matrix)), matrix.to_numpy(dtype=float)])
    names = ["(intercept)", *cols]
    # An aliased column is left out, as ``lm`` reports its coefficient NA; a benchmark made only
    # of aliased columns has no coefficient to benchmark against, and the exposure must have one.
    kept = column_basis(X)
    if names.index(exposure) not in kept:
        raise Unestimable(f"`{exposure}` is aliased with the covariates, so it has no coefficient")
    X, names = X[:, kept], [names[k] for k in kept]
    y = np.asarray(y, dtype=float)
    beta, cov, dof = _ols(X, y)
    j = names.index(exposure)
    est, se = float(beta[j]), float(math.sqrt(cov[j, j]))
    t = est / se
    classical = covariance in CLASSICAL
    note = None
    if not classical:
        shown = f", {shown_se:.3g}" if shown_se is not None else ""
        note = (f"The robustness value for the 95% interval and the benchmarks' adjusted intervals "
                f"rest on the classical least-squares standard error ({se:.3g}), as sensemakr "
                f"computes them; the interval reported uses {covariance} standard errors{shown}, so "
                f"they are not reported: they would describe an interval that is not the one "
                f"shown.")
    out = LinearSensitivity(exposure=exposure, estimate=est, se=se, t=t, dof=dof,
                            partial_r2=partial_r2(t, dof), rv=robustness_value(t, dof, alpha=1.0),
                            rv_alpha=robustness_value(t, dof, alpha=alpha) if classical else None,
                            alpha=alpha, covariance=covariance, interval_note=note)
    if not benchmarks:
        return out
    others = [k for k in range(X.shape[1]) if k != j]
    d = X[:, j]
    Xd = X[:, others]
    gamma, cov_d, dof_d = _ols(Xd, d)
    other_names = [names[k] for k in others]
    for label, members in benchmarks.items():
        members = [str(m) for m in members if str(m) in names and str(m) != exposure]
        if not members:
            continue
        ky_idx = [names.index(m) for m in members]
        kd_idx = [other_names.index(m) for m in members]
        r2y = _group_r2(beta, cov, ky_idx, dof)
        r2d = _group_r2(gamma, cov_d, kd_idx, dof_d)
        try:
            r2dz, r2yz = benchmark_bounds(r2d, r2y, kd=kd)
        except Unestimable:
            continue
        adj = adjusted_for_confounder(est, se, dof, r2dz, r2yz, alpha=alpha)
        out.benchmarks.append(Benchmark(
            covariate=str(label), r2dxj=r2d, r2yxj=r2y, r2dz=r2dz, r2yz=r2yz,
            estimate=adj["estimate"], se=adj["se"] if classical else None,
            ci_low=adj["ci_low"] if classical else None,
            ci_high=adj["ci_high"] if classical else None, kd=kd))
    return out


def _group_r2(coef: np.ndarray, cov: np.ndarray, idx: Sequence[int], dof: float) -> float:
    b = coef[list(idx)]
    V = cov[np.ix_(list(idx), list(idx))]
    if len(idx) == 1:
        return partial_r2(float(b[0]) / math.sqrt(float(V[0, 0])), dof)
    F = float(b @ np.linalg.solve(V, b)) / len(idx)
    return group_partial_r2(F, len(idx), dof)


SD_BASES = ("design_weighted", "sample")


def estimand_sd(y: Any, weights: Any = None) -> tuple[float, str]:
    """The standard deviation the E-value of a difference standardizes by, and its basis
    (MODELING_SEQUENCE §0 ruling 14: "the SD the estimand speaks of"; VanderWeele & Ding 2017's
    ``d = β / SD``, ``RR ≈ exp(0.91 d)``).

    Under the surveyed-population answer (``weights``, the analysis weights of the analyzed rows)
    it is the population's: R survey's ``svyvar``, ``n / (n − 1) · Σ w (y − ȳ_w)² / Σ w`` over the
    rows with a positive weight (``ȳ_w`` the weighted mean, ``n`` those rows' count), whose square
    root estimates the outcome's standard deviation in the surveyed population. Otherwise it is
    the analyzed rows' own (``ddof = 1``). The effects stage and the causal lane both call this."""
    values = np.asarray(y, dtype=float).ravel()
    if weights is None:
        return float(np.std(values, ddof=1)), "sample"
    w = np.asarray(weights, dtype=float).ravel()
    if len(w) != len(values):
        raise ValueError("the outcome and its weights must cover the same rows")
    keep = np.isfinite(w) & (w > 0) & np.isfinite(values)
    v, w = values[keep], w[keep]
    n = len(v)
    if n < 2:
        return float("nan"), "design_weighted"
    mean = float(np.sum(w * v) / np.sum(w))
    var = n / (n - 1) * float(np.sum(w * (v - mean) ** 2) / np.sum(w))
    return math.sqrt(var), "design_weighted"


def unmeasured_confounding(*, measure: str, estimate: float, ci_low: float | None,
                           ci_high: float | None, se: float | None = None,
                           outcome_share: float | None = None, outcome_sd: float | None = None,
                           matrix: pd.DataFrame | None = None, y: np.ndarray | None = None,
                           exposure_column: str | None = None,
                           benchmarks: Mapping[str, Sequence[str]] | None = None,
                           covariance: str = "classical",
                           sd_basis: str | None = None) -> dict[str, Any]:
    """Sensitivity to unmeasured confounding for one inference estimate (MODELING_SEQUENCE §0
    ruling 10): the function every inference result calls, and the one the causal lane must.

    ``measure`` is the estimate's scale: ``mean_difference`` (a linear coefficient), ``odds_ratio``,
    ``hazard_ratio`` or ``risk_ratio`` (ratios on their own scale, not logged), ``risk_difference``
    (no E-value without the risks: pass its risk ratio instead). ``outcome_share`` (the event's
    share) decides whether an odds or hazard ratio is read as a risk ratio (rare: below
    :data:`RARE_OUTCOME`) or converted. For a linear outcome, ``matrix``/``y``/``exposure_column``
    give the robustness value and ``benchmarks`` its named comparisons; it ranks first.
    ``covariance`` is the reported interval's (:func:`linear_sensitivity`).

    Returns ``{"methods": [...], "e_value": {...} | None, "robustness": {...} | None}``, the
    analyses in rank order, never a pass/fail verdict."""
    rare = None if outcome_share is None else bool(outcome_share < RARE_OUTCOME)
    out: dict[str, Any] = {"methods": [], "e_value": None, "robustness": None, "rare": rare}
    if measure == "mean_difference":
        if matrix is not None and y is not None and exposure_column is not None:
            out["robustness"] = linear_sensitivity(matrix, y, exposure_column, benchmarks,
                                                   covariance=covariance, shown_se=se).as_dict()
            out["methods"].append("robustness_value")
        if outcome_sd is not None and outcome_sd > 0:
            out["e_value"] = {**e_values(estimate, measure="OLS", sd=outcome_sd, se=se),
                              "sd": float(outcome_sd), "sd_basis": sd_basis or "sample"}
            out["methods"].append("e_value")
        return out
    by = {"odds_ratio": "OR", "hazard_ratio": "HR", "risk_ratio": "RR"}.get(measure)
    if by is None:
        raise ValueError(f"no E-value is defined here for a {measure.replace('_', ' ')}")
    if by in ("OR", "HR") and rare is None:
        raise ValueError("an odds or hazard ratio's E-value needs the outcome's share")
    out["e_value"] = e_values(estimate, ci_low, ci_high, measure=by, rare=rare)  # type: ignore[arg-type]
    out["methods"].append("e_value")
    return out


__all__ = [
    "ALIAS_TOL", "APPENDIX_TITLE", "BOOT", "Benchmark", "CLASSICAL", "COMMON_OUTCOME",
    "LEVERAGE_ONE", "LinearSensitivity", "LogisticFit", "PH_ALPHA", "RARE_OUTCOME", "SD_BASES",
    "Separated", "Setting", "Standardized", "Unestimable", "adjusted_for_confounder",
    "benchmark_bounds", "column_basis", "cook_threshold", "counterfactual", "cox_zph",
    "e_value_rr", "e_values", "estimand_sd", "g_computation", "group_partial_r2",
    "linear_sensitivity", "logistic_fit", "logistic_influence", "logistic_mle", "ols_influence",
    "outcome_sd", "partial_r2", "percentile", "period_hazard_ratios", "resample_indices",
    "robustness_value", "separated_rows", "settings_of", "split_follow_up", "split_rows",
    "standardized_risks", "time_scale", "to_risk_ratio", "unmeasured_confounding",
]
