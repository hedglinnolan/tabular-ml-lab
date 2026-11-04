"""Multiple imputation by chained equations, and Rubin's rules (AUDIT_REPORT §5 WP7; ME-01, ME-08).

Under inference a missing predictor is filled ``m`` times from its conditional distribution given
every other variable of the analysis, **the outcome and total energy included** (BLUEPRINT §12
ruling 4), each completed table is analyzed as if it were complete, and the ``m`` answers are
combined by Rubin's rules. Harrell (*Regression Modeling Strategies*, §3.8, citing Moons et al.
2006, *J Clin Epidemiol* 59:1092): "multiple imputation can and should use the response variable
for imputing predictors". Under prediction nothing here runs: the pipeline imputes in-fold without
the outcome, so the fitted pipeline can be deployed (Sisk et al. 2023, *Stat Methods Med Res*
32:1461).

**Where the copies are drawn** (MS1–MS2, MODELING_SEQUENCE §1.1): ``turbotab.core.methods.smcfcs``
runs the chained equations below and, where the analysis model is not linear in the imputed values
(a spline, a log, a ratio, a logistic or Cox outcome), substantive-model-compatible FCS; it adds the
log scale, the energy identity, the survey design and clustered rows. This module holds each
variable's model and draw, and the pooling: Rubin's rules with their Monte Carlo error, m by
:func:`m_rule`, and D1 (with Reiter's df on a finite complete-data df, :func:`reiter_df`).

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
# The widest imputation model (after categories become indicators) drawn whatever its rows, as
# before the rows rule: a wider one is drawn where each incomplete column's model has the rows per
# parameter the engine's rule asks for on the rows that record it (:func:`model_size`), and refused
# only where one has not (``smcfcs.ModelWidth.refused``), so the rule lifts this cap where the rows
# support it and removes nothing it drew. Its run time grows with its width (each sweep fits one
# least-squares or Newton model of n rows on p columns per incomplete column, about n·p²
# operations), a cost of the fit, not a reason its draws are unreliable.
MAX_MODEL_COLUMNS = 80
LOG_GAIN = 2.0  # a censored column is modeled on the log scale when that gains this much log-likelihood
OUTCOME_PREFIX = "__outcome"  # the outcome's terms in the imputation model never collide with a column

Kind = Literal["numeric", "binary", "categorical", "censored"]
# :class:`CensoredFitRefused`'s words where its caller names no column.
CENSORED_FIT_WORDS = ("A censored-normal (Tobit) model could not be fit on these rows: no maximum "
                      "of its likelihood could be confirmed.")


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
    # MS1–MS2 (``turbotab.core.methods.smcfcs``): how the copies were drawn, and what the
    # imputation model held beyond the columns (``plan``: the record ``missing.imputation_record``
    # writes from)
    method: str = "chained_equations"  # or "smcfcs", or "supplied" (copies the data carried)
    rejection_failures: int = 0  # SMC-FCS draws no candidate was kept for (smcfcs's rjFailCount)
    identity_infeasible: int = 0  # rows whose recorded sources already reach their recorded total
    plan: dict[str, Any] = field(default_factory=dict)
    # Imputed-outcome copies (NHANES DXA): each copy's own outcome, aligned with its frame
    outcomes: list[Any] | None = None


class ImputationRefused(ValueError):
    """The data cannot carry a chained-equations model; the message says why."""


class CensoredFitRefused(ImputationRefused):
    """:func:`tobit_fit`'s one failure: no maximum of the censored-normal likelihood could be
    certified on these rows (the solver did not reach a point where the score is zero and the
    information positive definite, or its arithmetic failed on the way). An
    :class:`ImputationRefused`, so every caller that refuses an imputation in words refuses this
    one too; its message is words, and ``why`` the solver's own account (never shown). A caller
    that knows the column says so (``columns``)."""

    def __init__(self, message: str = CENSORED_FIT_WORDS, why: str = ""):
        super().__init__(message)
        self.why = str(why)

    def of(self, column: str, words: str | None = None) -> "CensoredFitRefused":
        """This failure as ``column``'s: its words name the column (``words``, or that its values
        below detection cannot be drawn), and leaving it out lets the imputation run
        (``columns``)."""
        named = CensoredFitRefused(words or censored_fit_words(column), why=self.why)
        named.columns = [column]
        return named


def censored_fit_words(column: str) -> str:
    """The refusal of ``column``'s draws below detection where :func:`tobit_fit` certified no
    maximum (:class:`CensoredFitRefused`)."""
    return (f"`{column}`'s censored-normal model could not be fit on these rows: no maximum of its "
            f"likelihood could be confirmed, so its values below detection cannot be drawn.")


# ── how many rows an imputation model needs ──────────────────────────────────


@dataclass(frozen=True)
class ModelSize:
    """One incomplete column's imputation model against the rows-per-parameter rule
    (:func:`model_size`)."""

    column: str
    kind: str  # how it is drawn: "numeric", "censored", "binary" or "categorical"
    predictors: int  # its design's predictor columns, the intercept not counted
    rows: int  # the rows (the units, for a unit-level variable) that record it
    counted: float  # what the rule counts: those rows, the rarer value's (two levels), or their
    # effective size (three or more)
    needed: int  # what the rule asks per parameter
    levels: int = 0  # its observed levels (a yes/no or categorical model)
    units: bool = False  # imputed once per unit, from a model of the units

    @property
    def parameters(self) -> int:
        """The slopes the model fits: one per predictor column, and a multinomial model of k ≥ 3
        levels one per predictor column for each level beyond the first, (k − 1)·p."""
        return self.predictors * (self.levels - 1) if self.levels > 2 else self.predictors

    @property
    def held(self) -> bool:
        """Whether the rule applies: not to a yes/no or categorical model whose recorded values
        hold one level, which has no second level to count events or an effective size over. The
        chained equations draw such a column's blanks as its one level without fitting a model
        (``_categorical``, ``_binary``), as a substantive-model-compatible draw does a category's;
        such a draw of a 0/1 column fits its logistic model all the same (``smcfcs``'s covariate
        model), and its draws can be the other level."""
        return self.kind not in ("binary", "categorical") or self.levels > 1

    @property
    def per(self) -> float:
        return self.counted / self.parameters if self.parameters else math.inf

    @property
    def short(self) -> bool:
        return self.held and self.parameters > 0 and self.counted < self.needed * self.parameters


def model_size(column: str, kind: str, predictors: int, recorded: pd.Series, *,
               units: bool = False) -> ModelSize:
    """``column``'s imputation model (``predictors`` design columns, fit on its ``recorded``
    values: one per row, or per unit for a unit-level variable) against the rule the engine holds
    an inference model's coefficients to:

    * **a number** (least squares, ``mice``'s ``norm``), and a value below a detection limit (a
      censored normal, counted on its detected values): about
      :data:`~turbotab.core.models.sample_size.SPV_INFERENCE` rows per parameter (Austin &
      Steyerberg 2015, *J Clin Epidemiol* 68:627; ``models.sample_size``, as ``models.linear``
      states it);
    * **two levels** (yes/no, or a category with two recorded levels: a logistic model):
      :data:`~turbotab.core.models.sample_size.EPV_INFERENCE` events per parameter, the events
      being the rows of its rarer value (Peduzzi et al. 1996, the rule ``models.linear`` states
      for logistic coefficients and ``models.sample_size`` counts on the rarer class);
    * **three or more levels** (a multinomial model, which fits a slope per predictor column for
      each level beyond the first, (k − 1)·p parameters):
      :data:`~turbotab.core.models.ordinal.PER_PARAMETER` per parameter of the effective sample
      size n − Σn_k³/n² over the recorded levels (Whitehead's, as ``models.ordinal`` counts it
      for a proportional-odds model's predictors).

    A yes/no or categorical model whose recorded values hold one level is not held to the rule
    (:attr:`ModelSize.held`). The rule judges reliability; the width's run time is a cost the rule
    does not count."""
    from turbotab.core.models.ordinal import PER_PARAMETER, effective_rows
    from turbotab.core.models.sample_size import EPV_INFERENCE, SPV_INFERENCE

    values = recorded.dropna()
    n = int(len(values))
    if kind in ("binary", "categorical"):
        counts = [int(k) for k in values.astype(str).value_counts().to_numpy()]
        if len(counts) == 2:  # a logistic model: its events are the rarer value's rows
            return ModelSize(column, kind, int(predictors), n, float(min(counts)), EPV_INFERENCE,
                             levels=2, units=units)
        return ModelSize(column, kind, int(predictors), n, float(effective_rows(counts)),
                         PER_PARAMETER, levels=len(counts), units=units)
    return ModelSize(column, kind, int(predictors), n, float(n), SPV_INFERENCE, units=units)


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


NORM_DRAW_WORDS = "The model that imputes a number's blanks could not be computed on these rows."


class NormDrawRefused(ImputationRefused, np.linalg.LinAlgError):
    """:func:`_norm_draw`'s one refusal: numpy could not compute the pseudo-inverse or the
    eigendecomposition of the model's cross-product (an SVD or eigendecomposition that did not
    converge). An :class:`ImputationRefused`, so every caller that refuses an imputation in words
    refuses this one too, and a LinAlgError, as numpy's failure was, so a caller that refuses a
    chained model's linear-algebra failure in its own words naming the column
    (``featurewise_mi.draw_copies``) still does. Its message is words, and ``why`` numpy's own
    account (never shown)."""

    def __init__(self, message: str = NORM_DRAW_WORDS, why: str = ""):
        super().__init__(message)
        self.why = str(why)


def _norm_draw(X: np.ndarray, y: np.ndarray, rng: np.random.Generator
               ) -> tuple[np.ndarray, np.ndarray, float]:
    """(β̂, β*, σ*): least squares and a draw from its posterior (mice's ``.norm.draw``), β* = β̂ +
    σ* R z with R Rᵀ = V, V the pseudo-inverse of XᵀX with mice's ridge.

    R is V's Cholesky factor, with 10⁻¹² on its diagonal. Where that factor does not exist in
    float64, R is V's root by its eigendecomposition, V symmetrized first as mice's ``sym`` does
    and its negative eigenvalues, which only rounding gives a pseudo-inverse of a positive
    semidefinite matrix, set to 0; the draw takes the same z. The factor can fail where a predictor
    is zero on the rows the model is fit to (the indicator of a level none of them hold, or a
    column of one value, which standardizes to 0), so V and the predictor's coefficient are zero
    along it, as mice's ``remove.lindep`` drops a predictor whose variance on those rows is at most
    10⁻⁴, and V's other directions are large enough (as where the model has more columns than
    rows) that its rounding outweighs the 10⁻¹². A cross-product numpy cannot factor (an SVD or
    eigendecomposition that does not converge) is refused in words (:class:`NormDrawRefused`)."""
    XtX = X.T @ X
    XtX = XtX + np.diag(RIDGE * np.maximum(np.diag(XtX), 1e-12))
    try:
        inv = np.linalg.pinv(XtX)
    except np.linalg.LinAlgError as exc:
        raise NormDrawRefused(why=f"{type(exc).__name__}: {exc}") from None
    beta = inv @ (X.T @ y)
    resid = y - X @ beta
    df = max(len(y) - X.shape[1], 1)
    sigma = math.sqrt(float(resid @ resid) / rng.chisquare(df))
    try:
        root = np.linalg.cholesky(inv + np.eye(len(inv)) * 1e-12)
    except np.linalg.LinAlgError:
        try:
            lam, Q = np.linalg.eigh((inv + inv.T) / 2)
        except np.linalg.LinAlgError as exc:
            raise NormDrawRefused(why=f"{type(exc).__name__}: {exc}") from None
        root = Q * np.sqrt(np.clip(lam, 0.0, None))
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
    """A yes/no column's blanks: β* ~ N(β̂, H⁻¹) from its penalized logistic model (``_logistic``),
    then a Bernoulli draw for each. Where its recorded values hold one level, that model's
    intercept has no finite maximum (the fit stops far out, with a variance far larger, so a drawn
    intercept can take either sign), and every blank is drawn as that level, as ``_categorical``
    draws a category of one level; the model's random numbers are still taken, so every later draw
    in the copy takes the random numbers it took before."""
    obs = ~miss
    recorded = np.unique(y[obs])
    if len(recorded) == 1:
        rng.standard_normal(X.shape[1])  # β*'s
        rng.random(int(miss.sum()))  # each blank's
        return np.full(int(miss.sum()), recorded[0], dtype=float)
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


# The Newton steps the fit takes at most; it reaches its maximum in 5 to 15 from its start.
TOBIT_STEPS = 200
# The fit is certified at its maximum where the Newton decrement gᵀ(−H)⁻¹g is at most this: twice
# the log-likelihood a further Newton step would gain, and the squared distance to the maximum in
# standard errors (1e-12: within 10⁻⁶ of a standard error, far above rounding, ~1e-25 on 100 rows).
TOBIT_DECREMENT = 1e-12
# Below this decrement the fit takes Newton's full step, no line search (the quadratic region).
TOBIT_NEWTON = 1e-8
# ... and its information positive definite: the smallest eigenvalue of the information scaled to a
# unit diagonal (Jacobi's scaling, which no parameter's units move) above this, survreg's
# ``toler.chol`` (1e-10). Over the verifier's 362 designs with a maximum it was at least 3.5e-4.
TOBIT_SINGULAR = 1e-10
# ... and in every direction of the coefficients at least this share of the information the same
# rows would carry were every value detected (:func:`_design_share`). The unit diagonal hides a
# direction whose curvature vanishes with its own diagonal, as where a 0/1 column's ones are all
# below detection and its coefficient runs to −∞; this share does not, whatever the columns'
# coding. A row below detection carries w = λ(a)(a + λ(a)) of a detected row's information, under
# 1e-10 only where its fitted mean lies more than 6.93 σ below the limit, its log Φ(a) then within
# 2e-12 of its bound 0: a direction only such rows carry is flat to the log-likelihood's rounding,
# and a maximum along it cannot be told from a bound approached at infinity. Over the verifier's
# 362 designs with a maximum and the 417 of its 600 fuzz designs the solver fits, the share was at
# least 1.6e-6; at the 280 points the base solver returned where a 0/1 column's ones were all below
# detection (its probe's 200 and 80 fuzz designs), at most 2.6e-16.
TOBIT_SHARE = 1e-10


def _below_weights(a: np.ndarray, log_cdf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For rows below detection at a = t L − xᵀγ, log Φ(a) given: (λ = φ(a)/Φ(a), w = λ(a + λ)),
    w ∈ (0, 1) each row's share of a detected row's information on γ."""
    lam = np.exp(-0.5 * a * a - 0.5 * math.log(2.0 * math.pi) - log_cdf)
    return lam, lam * (a + lam)


def _design_share(Xk: np.ndarray, censored: np.ndarray, w: np.ndarray) -> float:
    """The smallest share of its information that a direction of the coefficients keeps at the
    fit: the least eigenvalue of Qᵀ diag(r) Q, Q an orthonormal basis of the design's columns
    (Householder QR) and r 1 on a detected row, w on a row below detection. The information on γ is
    Xᵀ diag(r) X and every value detected gives XᵀX, so this is their least generalized eigenvalue,
    in (0, 1], moved by no change of the columns' coding (X → XT), and read from Q and w, never
    from the information's own diagonal."""
    Q, _ = np.linalg.qr(Xk)
    r = np.ones(len(Xk))
    r[censored] = w
    return float(np.linalg.eigvalsh((Q * r[:, None]).T @ Q).min())


def _olsen(Xd: np.ndarray, zd: np.ndarray, Xc: np.ndarray, limit: float, phi: np.ndarray,
           derivatives: bool = True) -> tuple[float, np.ndarray | None, np.ndarray | None]:
    """The censored-normal log-likelihood in Olsen's (1978) parametrization φ = (γ, t) = (β/σ,
    1/σ), with its score and Hessian: Σ_detected [log t − (t z − xᵀγ)²/2] + Σ_below log Φ(t L −
    xᵀγ), concave in φ (Olsen, *Econometrica* 46:1211), the constant −½ log 2π of each detected
    row left out. With λ = φ(a)/Φ(a) and w = λ(a + λ) ∈ (0, 1) at a = t L − xᵀγ: the score is
    (Σ_d e x − Σ_c λ x, n_d/t − Σ_d e z + L Σ_c λ), e = t z − xᵀγ, and the Hessian
    [[−Σ_d x xᵀ − Σ_c w x xᵀ, Σ_d z x + L Σ_c w x], [·, −n_d/t² − Σ_d z² − L² Σ_c w]]."""
    from scipy import special

    gamma, t = phi[:-1], float(phi[-1])
    e = t * zd - Xd @ gamma
    a = t * limit - Xc @ gamma
    log_cdf = special.log_ndtr(a)
    ll = float(len(zd) * math.log(t) - 0.5 * float(e @ e) + float(log_cdf.sum()))
    if not derivatives:
        return ll, None, None
    lam, w = _below_weights(a, log_cdf)
    g = np.concatenate([Xd.T @ e - Xc.T @ lam,
                        [len(zd) / t - float(zd @ e) + limit * float(lam.sum())]])
    k = len(phi)
    H = np.empty((k, k))
    H[:-1, :-1] = -(Xd.T @ Xd) - (Xc * w[:, None]).T @ Xc
    H[:-1, -1] = H[-1, :-1] = Xd.T @ zd + limit * (Xc.T @ w)
    H[-1, -1] = -len(zd) / t ** 2 - float(zd @ zd) - limit ** 2 * float(w.sum())
    return ll, g, H


def _unit_diagonal(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(A scaled to a unit diagonal, the scale); raises where a diagonal entry is not positive."""
    d = np.sqrt(np.diag(A))
    if not (np.all(np.isfinite(d)) and np.all(d > 0)):
        raise CensoredFitRefused(why="the information has a diagonal entry that is not positive")
    return A / np.outer(d, d), d


def tobit_fit(X: np.ndarray, z: np.ndarray, censored: np.ndarray, limit: float
              ) -> tuple[np.ndarray, np.ndarray]:
    """Maximum likelihood for z ~ N(Xβ, σ²) where ``censored`` rows are known only to lie below
    ``limit``: (θ̂ = (β, log σ), the inverse observed information at θ̂), θ̂ a certified maximum,
    or :class:`CensoredFitRefused`.

    Fitted by Newton's method with a backtracking (Armijo) line search in Olsen's concave
    parametrization (:func:`_olsen`), full steps once the decrement is below
    :data:`TOBIT_NEWTON`, on z centered on its detected mean (where a column of X is constant, an
    intercept) and scaled by its root mean square, so neither the column's units nor its offset
    move the fit. A column the earlier ones span is aliased (``models.effects.column_basis``, R's
    ``qr`` at 1e-7, as ``lm`` reads it): its coefficient is 0 and its row of the inverse
    information 0, as survreg reports it (NA, a zero row of its covariance). The point returned is
    certified the maximum: its Newton decrement at most :data:`TOBIT_DECREMENT` (the score zero),
    its information, scaled to a unit diagonal, with no eigenvalue at or below
    :data:`TOBIT_SINGULAR` (a negative-definite Hessian), and no direction of the coefficients
    keeping :data:`TOBIT_SHARE` or less of the information the same rows would carry with every
    value detected (:func:`_design_share`, whatever the columns' coding). Where any fails, or the
    arithmetic does on the way (an overflow, a value that is not finite), it raises
    :class:`CensoredFitRefused`, never a point short of the maximum: a likelihood with no finite
    maximum (a column detected only in rows of one outcome, a 0/1 column whose ones or whose zeros
    are all below detection, raw or standardized, or values fitted exactly with every value below
    detection under the limit) is refused so. The inverse information is that of (β, log σ), the
    inverse of Jᵀ(−H)J with J = ∂φ/∂θ, exact at the maximum (the score's term vanishes there)."""
    try:
        with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
            return _tobit_certified(np.asarray(X, dtype=float), np.asarray(z, dtype=float),
                                    np.asarray(censored, dtype=bool), float(limit))
    except CensoredFitRefused:
        raise
    except (ArithmeticError, np.linalg.LinAlgError, ValueError) as exc:
        raise CensoredFitRefused(why=f"{type(exc).__name__}: {exc}") from exc


def _tobit_certified(X: np.ndarray, z: np.ndarray, censored: np.ndarray, limit: float
                     ) -> tuple[np.ndarray, np.ndarray]:
    n, p = X.shape
    det = ~censored
    if not det.any():
        raise CensoredFitRefused(why="no detected value")
    if not (np.all(np.isfinite(X)) and np.all(np.isfinite(z[det])) and math.isfinite(limit)):
        raise CensoredFitRefused(why="a value that is not finite")
    from scipy import special

    from turbotab.core.models.effects import column_basis

    keep = column_basis(X)
    Xk = X[:, keep]
    y = np.where(censored, limit, z)
    constant = [j for j in range(Xk.shape[1]) if np.ptp(Xk[:, j]) == 0 and Xk[0, j] != 0]
    center = float(np.mean(z[det])) if constant else 0.0
    scale = float(np.sqrt(np.mean((y - center) ** 2)))
    if not (math.isfinite(scale) and scale > 0):
        scale = 1.0
    ys, L = (y - center) / scale, (limit - center) / scale
    Xd, zd, Xc = Xk[det], ys[det], Xk[censored]
    # The start: least squares on every row (those below detection at the limit), its spread at
    # least a tenth of the column's.
    start = np.linalg.lstsq(Xk, ys, rcond=None)[0]
    spread = max(float(np.sqrt(np.mean((ys - Xk @ start) ** 2))), 0.1)
    phi = np.concatenate([start / spread, [1.0 / spread]])
    eye = np.eye(len(phi))
    previous = math.inf
    for _ in range(TOBIT_STEPS):
        ll, g, H = _olsen(Xd, zd, Xc, L, phi)
        if not (math.isfinite(ll) and np.all(np.isfinite(g)) and np.all(np.isfinite(H))):
            raise CensoredFitRefused(why="a value that is not finite on the way to the maximum")
        S, d = _unit_diagonal(-H)
        damp = 0.0  # Levenberg's: only where the scaled information is not positive definite
        while True:
            try:
                root = np.linalg.cholesky(S + damp * eye)
                break
            except np.linalg.LinAlgError:
                damp = 1e-10 if damp == 0 else damp * 10
                if damp > 1e6:
                    raise CensoredFitRefused(why="no direction of ascent") from None
        step = np.linalg.solve(root.T, np.linalg.solve(root, g / d)) / d
        decrement = float(g @ step)
        if damp == 0 and decrement <= TOBIT_DECREMENT * 1e-8:
            break
        if damp == 0 and decrement <= TOBIT_NEWTON:
            # Near the maximum Newton's full step squares the decrement; a gain this small is
            # below the log-likelihood's rounding, so no line search could see it. Stop where a
            # step no longer halves it (rounding).
            if decrement > 0.5 * previous:
                break
            previous = decrement
            phi = phi + step
            continue
        alpha, moved = 1.0, False
        while alpha > 1e-14:
            trial = phi + alpha * step
            if trial[-1] > 0:
                gained = _olsen(Xd, zd, Xc, L, trial, derivatives=False)[0]
                if math.isfinite(gained) and gained >= ll + 1e-4 * alpha * decrement:
                    phi, moved = trial, True
                    break
            alpha *= 0.5
        if not moved:
            break  # no step gains: the maximum within rounding, or none (the certificate says)
    _, g, H = _olsen(Xd, zd, Xc, L, phi)
    A = -H
    S, _d = _unit_diagonal(A)
    if not (np.all(np.isfinite(S)) and np.all(np.isfinite(g))):
        raise CensoredFitRefused(why="a value that is not finite at the maximum")
    smallest = float(np.linalg.eigvalsh(S).min())
    if not smallest > TOBIT_SINGULAR:
        raise CensoredFitRefused(why=f"the information is singular (scaled eigenvalue "
                                     f"{smallest:.3g})")
    gamma, t = phi[:-1], float(phi[-1])
    a = t * L - Xc @ gamma
    share = _design_share(Xk, censored, _below_weights(a, special.log_ndtr(a))[1])
    if not share > TOBIT_SHARE:
        raise CensoredFitRefused(why=f"a direction of the coefficients keeps {share:.3g} of the "
                                     f"information its rows would carry with every value detected")
    decrement = float(g @ np.linalg.solve(A, g))
    if not decrement <= TOBIT_DECREMENT:
        raise CensoredFitRefused(why=f"the score is not zero (Newton decrement {decrement:.3g})")
    k = len(phi)
    J = np.zeros((k, k))  # ∂(γ, t)/∂(β, log σ) on the scaled column
    J[:-1, :-1] = t * np.eye(k - 1)
    J[:-1, -1] = -gamma
    J[-1, -1] = -t
    inverse = np.linalg.inv(J.T @ A @ J)
    inverse = 0.5 * (inverse + inverse.T)
    beta = scale * gamma / t
    if constant:
        beta[constant[0]] += center / Xk[0, constant[0]]
    units = np.r_[np.full(k - 1, scale), 1.0]
    theta = np.zeros(p + 1)
    theta[keep] = beta
    theta[-1] = math.log(scale / t)
    at = [*keep, p]
    V = np.zeros((p + 1, p + 1))
    V[np.ix_(at, at)] = inverse * np.outer(units, units)
    if not (np.all(np.isfinite(theta)) and np.all(np.isfinite(V))):
        raise CensoredFitRefused(why="a value that is not finite at the maximum")
    return theta, V


def tobit_expectation(mu: float, sigma: float, limit: float) -> float:
    """E[Z | Z < limit] for Z ~ N(mu, σ²): mu − σ φ(a)/Φ(a), a = (limit − mu)/σ."""
    from scipy import stats

    a = (limit - mu) / sigma
    return float(mu - sigma * math.exp(stats.norm.logpdf(a) - stats.norm.logcdf(a)))


# A draw's square root raises each diagonal entry of the inverse information by this share of
# itself (rounding's margin for the Cholesky factor; relative, so no column's units move it).
TOBIT_ROOT_JITTER = 1e-12


def tobit_root(V: np.ndarray) -> np.ndarray:
    """A square root R (R Rᵀ = V) of :func:`tobit_fit`'s inverse information, for the parameter
    draws θ* = θ̂ + R ε: the Cholesky factor of the block of the columns the fit kept, each diagonal
    entry raised by :data:`TOBIT_ROOT_JITTER` of itself, and zero in an aliased column's row and
    column (its coefficient stays 0, as survreg reports it NA). It moves with the columns' units
    (R of D V D is D R), where an absolute jitter would swamp a coefficient's variance in small
    units. Raises :class:`CensoredFitRefused` where the block has no Cholesky factor."""
    V = np.asarray(V, dtype=float)
    kept = np.flatnonzero(np.diag(V) > 0)
    block = V[np.ix_(kept, kept)]
    root = np.zeros_like(V)
    try:
        root[np.ix_(kept, kept)] = np.linalg.cholesky(
            block + np.diag(np.diag(block) * TOBIT_ROOT_JITTER))
    except np.linalg.LinAlgError as exc:
        raise CensoredFitRefused(why=f"{type(exc).__name__}: {exc}") from exc
    return root


def _censored(X: np.ndarray, x: np.ndarray, miss: np.ndarray, c: Censoring,
              rng: np.random.Generator) -> np.ndarray:
    from scipy import stats

    z = np.log(np.clip(x, 1e-300, None)) if c.log else x.astype(float)
    limit = math.log(c.limit) if c.log else c.limit
    theta, V = tobit_fit(X, np.where(miss, limit, z), miss, limit)
    root = tobit_root(V)
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
    """``m`` completed copies of ``data`` (every column of the imputation model) by chained
    equations, ``mice``'s algorithm: :func:`turbotab.core.methods.smcfcs.impute` in its ``"fcs"``
    mode, the one implementation (the module that also draws SMC-FCS, the log scale, the energy
    identity and clustered rows).

    ``impute``: the columns whose blanks are filled (default: every column with a blank); the
    others are only predictors (the outcome's terms, which are never missing in the analysis rows).
    ``kinds``: each column's kind (default :func:`column_kind`). ``censored``: the left-censored
    columns and their limits. Deterministic for a given ``seed``.
    """
    from turbotab.core.methods.smcfcs import Variable
    from turbotab.core.methods.smcfcs import impute as draw

    censored = dict(censored or {})
    kinds = {c: (kinds or {}).get(c) or ("censored" if c in censored else column_kind(data[c]))
             for c in data.columns}
    for c in censored:
        kinds[c] = "censored"
    names = [c for c in (impute if impute is not None else data.columns)
             if c in data.columns and data[c].isna().any()]
    variables = [Variable(c, kinds[c], censoring=censored.get(c)) for c in names]
    return draw(data, variables, mode="fcs", m=m, iterations=iterations, seed=seed,
                kinds=kinds, progress=progress, cancelled=cancelled)


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
    # Monte Carlo error of the pooled estimate, √(B/m): how much it would move with another set of
    # m imputations (White, Royston & Wood 2011, Stat Med 30:377, §7).
    mc_se: float | None = None


def m_rule(n_incomplete: int, n_rows: int, asked: int = M_DEFAULT) -> int:
    """The number of imputations: at least ``asked`` (20 by default, the WP7 floor) and at least the
    percentage of rows with any imputed value (MODELING_SEQUENCE §1 step 6; White, Royston & Wood
    2011, as White, Pandis & Pham put it: "the number of imputations should be at least equal to the
    percentage of incomplete cases"), at most :data:`M_MAX`."""
    share = 100.0 * n_incomplete / n_rows if n_rows else 0.0
    return int(min(M_MAX, max(int(asked), M_DEFAULT, math.ceil(share - 1e-9))))


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
    mc = math.sqrt(between / m) if m > 1 else None
    if not (total > 0 and math.isfinite(total)):
        return Pooled(est, within, between, total, None, None, None, None, None, mc)
    se = math.sqrt(total)
    ref = stats.norm if math.isinf(nu) else stats.t(nu)
    half = float(ref.ppf(0.5 + level / 2)) * se
    p = float(2 * ref.sf(abs(est) / se))
    return Pooled(est, within, between, total, None if math.isinf(nu) else nu, gamma,
                  est - half, est + half, p, mc)


def reiter_df(r: float, k: int, m: int, df_com: float) -> float:
    """The denominator degrees of freedom of D1 with a finite complete-data df: Reiter's (2007,
    *Biometrika* 94:502) small-sample ν, exactly as R's ``mitml`` (0.4.5, ``mitml:::.D1``) computes
    it from r1, t = k(m − 1) and ν* = (ν_com + 1)/(ν_com + 3) ν_com."""
    t = k * (m - 1)
    a = r * t / (t - 2)
    vstar = (df_com + 1) / (df_com + 3) * df_com
    c0 = 1 / (t - 4)
    c1 = vstar - 2 * (1 + a)
    c2 = vstar - 4 * (1 + a)
    z = (1 / c2 + c0 * (a ** 2 * c1 / ((1 + a) ** 2 * c2))
         + c0 * (8 * a ** 2 * c1 / ((1 + a) * c2 ** 2) + 4 * a ** 2 / ((1 + a) * c2))
         + c0 * (4 * a ** 2 / (c2 * c1) + 16 * a ** 2 * c1 / c2 ** 3) + c0 * (8 * a ** 2 / c2 ** 2))
    return 4 + 1 / z


def pooled_wald(estimates: np.ndarray, covariances: np.ndarray,
                df_com: float | None = None) -> dict[str, float] | None:
    """Li, Raghunathan & Rubin's D1 that every coefficient is zero: ``estimates`` (m × k),
    ``covariances`` (m × k × k). With ``df_com`` (each copy's complete-data df: the design df under a
    survey) the denominator df is Reiter's (:func:`reiter_df`), as ``mitml::testModels`` and
    ``testConstraints`` take ``df.com``; without it, Li et al.'s. None when the within-imputation
    covariance is singular."""
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
    if df_com is not None and math.isfinite(df_com) and t > 4 and r1 > 1e-12:
        nu1 = reiter_df(r1, k, m, float(df_com))
    elif r1 <= 1e-12:
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
                       df=None, fmi=None,
                       mc_se=float(np.std(est, ddof=1) / math.sqrt(m)) if m > 1 else None)
        else:
            dfs = [r.get("df") for r in rows]
            df_com = None if any(d is None for d in dfs) else float(np.mean(dfs))
            pooled = pool_scalar(est, [float(s) ** 2 for s in se], df_com, level)
            new.update(estimate=pooled.estimate, ci_low=pooled.ci_low, ci_high=pooled.ci_high,
                       p=pooled.p, se=math.sqrt(pooled.total) if pooled.total > 0 else None,
                       df=pooled.df, fmi=pooled.fmi, mc_se=pooled.mc_se)
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
    "CENSORED_FIT_WORDS", "Censoring", "CensoredFitRefused", "ITERATIONS", "ImputationRefused",
    "Imputations", "M_DEFAULT", "M_MAX", "TOBIT_DECREMENT", "TOBIT_SINGULAR",
    "censored_fit_words", "censored_loglik", "log_scale_fits_better",
    "MAX_MODEL_COLUMNS", "ModelSize", "OUTCOME_PREFIX", "Pooled", "barnard_rubin", "censoring_of",
    "chained_equations", "column_kind", "m_rule", "model_size", "nelson_aalen", "outcome_terms",
    "pool_rows",
    "pool_scalar", "pooled_cov", "pooled_wald", "reiter_df", "tobit_expectation", "tobit_fit",
]
