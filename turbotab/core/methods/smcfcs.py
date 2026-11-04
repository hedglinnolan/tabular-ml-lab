"""Multiple imputation compatible with the analysis model (MODELING_SEQUENCE §0 ruling 5, §1.1;
MS1 and MS2 of §5).

**Why.** The engine used to impute every incomplete column from a model linear in the others and in
the outcome, on the column's own scale, and then derive logs, residuals, splines and products in
each completed copy ("passive" imputation). Bartlett, Seaman, White & Carpenter 2015 (*Stat Methods
Med Res* 24:462): "X2 is then passively imputed by squaring the imputed X values. In this case,
estimates of the parameters of the substantive model from multiply imputed datasets will be biased
(unless the quadratic coefficient is in truth zero)"; their Table 2 (normal X, MCAR) gives 0.696
(SD 0.041, coverage 0%) for a true quadratic coefficient of 1 after linear passive imputation, and
0.998 (0.038, 93.9%) after substantive-model-compatible FCS. Seaman, Bartlett & White 2012 (*BMC
Med Res Methodol* 12:46): just another variable "should not be used for logistic regression".

**Two modes, one sampler** (:func:`impute`):

* ``"fcs"`` — chained equations as ``mice`` runs them (van Buuren & Groothuis-Oudshoorn 2011): each
  incomplete variable from a model on every other variable and the outcome's terms, fit on its
  observed rows, its parameters drawn from their posterior before each draw. Compatible with a
  linear outcome model that is linear in the imputed variables on the scale they are imputed on.
* ``"smcfcs"`` — Bartlett et al.'s algorithm as R's ``smcfcs`` (2.0.2) runs it
  (``smcfcs:::smcfcs.core``): the outcome is never a predictor in a covariate model; each
  variable's covariate model is fit to every row with its current values and its parameters drawn;
  the substantive model is fit to the current completed data and its parameters drawn (an aliased
  column of its design left out and its coefficient zero, as R's ``lm`` leaves it:
  :func:`_draw_outcome`; a Cox model whose fit does not converge refused: :func:`_cox_draw`, in
  words that name the terms that fail and whether they are the analysis model's or the survey
  design's: :func:`_cox_refusal`);
  then a binary or categorical variable is drawn directly from
  p(x | z) f(y | x, z), and a number by
  rejection sampling: a candidate from its covariate model is kept with probability
  f(y | x*, z)/max_x f(y | x, z) — exp(−(y − η)²/2σ²) for a linear outcome, p^y(1 − p)^(1 − y) for a
  logistic one, and for a Cox model exp(−H₀(t)e^η) when censored and H₀(t)e^η exp(1 − H₀(t)e^η) for an
  event, with H₀ the cumulative baseline hazard at the drawn coefficients as smcfcs takes it
  (``survival::basehaz`` of the Efron-ties fit with its coefficients replaced by the draw: Efron's
  tie-corrected increments, :func:`efron`, which equal Breslow's where no event time is shared).
  Up to 1,024 candidates per draw, smcfcs's
  budget (24 vectorized first tries, then ``rjlimit`` = 1,000 for each draw still waiting), sent
  through the design in batches and tried in order, so the first kept is exactly what one-at-a-time
  rejection sampling keeps; a draw none of them is kept for holds its last candidate and is counted
  (``rejection_failures``), as ``smcfcs`` counts ``rjFailCount``. The substantive
  model's terms come from ``Substantive.design``, any function of the raw values that is row-local
  once fit (the analysis pipeline's own steps), so a spline with fixed knots, a product, a log
  residual or a density enter the imputation exactly as the analysis model holds them.

**What both modes add** (MODELING_SEQUENCE §1.1, inference, step 1):

* **Logged quantities on the log scale** (``Variable.log``): the variable is modeled and drawn on
  log x and returned as exp, so no copy holds a non-positive value of a column the analysis logs.
* **The energy identity** (:class:`Identity`): the energy sources and "other" are the imputed
  variables and total energy is derived as their sum, E = Σ fⱼ Nⱼ + other. Where E is recorded and
  a source is not, the source is drawn truncated so that other = E − Σ fⱼ Nⱼ stays above zero; where
  E is blank, other is drawn (on the log scale when every recorded other is positive, else
  truncated at zero) and E follows. A Gibbs step truncated this way keeps a row inside the
  identity only if the row starts inside it, so each chain starts every row with a recorded total
  jointly inside it (:meth:`_Chain.feasible_start`: where the random starts of a row's blank
  sources together leave no room, they are scaled to half the room their recorded total leaves).
  Before that start, two blank sources on one row could hold each other above the total forever
  (each one's bound read the other's start, found no room and kept its own), and on the log scale
  other fell to −1,700 kcal. Other is therefore never negative in a copy, except where the recorded
  sources alone already reach the recorded total (counted in ``identity_infeasible``; a logged
  source there starts at, and keeps, its smallest recorded value). An energy source's covariate
  model reads the total in place of "other" (:meth:`_Chain.energy_terms`): where the total is
  recorded, other is a function of the source itself, and reading it held each draw near the
  source's last value (imputed protein −4 to −5.5 simulation SE low with two sources blank on the
  log scale, −1 to −2 SE with one; within ±1.4 SE with the total read instead).
* **Clustered rows** (ruling 12; ``units``): a ``unit_level`` variable (one the user confirmed as
  one value per unit, through the readings ledger: ``missing.time_invariant_columns``; its
  recorded value already carried to the unit's blank rows, ``missing.carry_within_units``) is
  imputed once per unit, from a model of the units on the other unit-level variables and
  the unit means of the row-level ones (the outcome's too, in ``"fcs"``), and copied to every row of
  the unit; a row-level variable's model adds the unit means of the other row-level variables and
  its own mean over the row's other rows in the unit (the unit's level of it: without it, in this
  package's simulation, the completed copies' intraclass correlation was 0.43 against a true 0.56,
  no better than single-level imputation's 0.42; with it, 0.57). Under
  ``"smcfcs"`` a unit's candidate is kept with probability Π f(y_ij | x*, z_ij)/max, each factor the
  row's normalized density. Full two-level FCS is not built (ruling 12: INBOX).
* **Design variables** and any other complete column are predictors in every covariate model.
  Under ``"smcfcs"`` a design variable is in the substantive model too (``Substantive.design``
  holds it beside the analysis model's terms: ``missing.engine_substantive``), since SMC-FCS
  assumes an auxiliary variable is independent of the outcome given the substantive model's
  covariates; with the design in the covariate models only, an outcome that differs by stratum
  biased the pooled design-based estimates (MS2 repair: z −0.06, about 7 Monte Carlo SE).

**How wide the model may be** (:func:`model_width`, :meth:`ModelWidth.refused`). Up to
``imputation.MAX_MODEL_COLUMNS`` columns, categories as indicators, it is drawn whatever its rows,
as before; wider, each incomplete column's model is held to the rows-per-parameter rule the
engine holds an inference model's coefficients to (``imputation.model_size``: about 2 rows per
parameter for least squares, 10 events, its rarer value's rows, per parameter for a model of two
levels, and 10 per parameter of the effective sample size for a multinomial model, whose
parameters are a slope per predictor column for each level beyond the first) on the rows that
record it, and the model is refused only where one falls short. The rule lifts the fixed cap where
the rows support it and removes nothing the cap drew (the orchestrator's ruling, 2026-11-02). The
chain fits a numeric code the user declared a category as its indicators, as the rule counts it.
Before this rule every model wider than 80 columns was refused whatever its rows, and only at Fit,
after the plan's lock: Sullivan 2021's runs 22 and 26 locked imputations of 89 and 84 columns on
7,937 rows, the survey design's strata most of them. The same function refuses here and at the
answer, before the lock (:func:`imputation_at`, registered for every answer and for the lock); its
ways forward are complete cases, the sample answer where the design's terms are most of the width,
and leaving out a column the others already span (a recode beside its code), of the columns an
exit may leave out (the answer's settled covariates; the fit, given none, names none).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from turbotab.core.methods.imputation import (
    ITERATIONS,
    M_DEFAULT,
    MAX_MODEL_COLUMNS,
    Censoring,
    ImputationRefused,
    Imputations,
    ModelSize,
    _as01,
    _expit,
    _from01,
    _logistic,
    _norm_draw,
    _standardize,
)

Mode = Literal["fcs", "smcfcs"]
Kind = Literal["numeric", "binary", "categorical", "censored"]
RJLIMIT = 1000  # smcfcs's default rejection limit per row
FIRST_TRIES = 24  # smcfcs's vectorized first tries (``j`` from 1 while below ``firstTryLimit`` = 25)
BATCH_ROWS = 4_000  # rows of candidates sent through the design at once
OTHER_FLOOR = 1e-6  # "other" kept at least this share of total energy above zero (log scale)
START_SHARE = 0.5  # a row's blank sources that start beyond its total start at this share of the room
# A design whose columns, each scaled to unit length, have a Gram matrix with every eigenvalue
# above this has no aliased column: each column's part outside the span of the others is then at
# least 1e−4 of its length (the square root), a thousand times R's ``lm`` tolerance (1e−7), and
# far above the rounding of the eigenvalue itself. Below it, R's rule decides column by column.
NO_ALIAS_EIGENVALUE = 1e-8


@dataclass(frozen=True)
class Variable:
    """One incomplete variable of the imputation model and how it is drawn."""

    name: str
    kind: Kind
    log: bool = False  # modeled and drawn on log x (every recorded value must be positive)
    lower: float | None = None  # drawn at or above this on its own scale (not used with ``log``)
    unit_level: bool = False  # constant within each unit: imputed once per unit
    censoring: Censoring | None = None  # left-censored: drawn below its limit


@dataclass(frozen=True)
class Identity:
    """Total energy as the sum of its sources and "other" (MODELING_SEQUENCE §1.1).

    ``energy``: the total-energy column (raw kcal; blank where missing). ``other``: the column of
    the data frame holding E − Σ fⱼ Nⱼ (blank wherever E or a source is blank), one of the
    variables. ``factors``: kcal per unit of each source column."""

    energy: str
    other: str
    factors: Mapping[str, float]


@dataclass
class Outcome:
    """The substantive model's outcome: ``linear`` (``y``), ``logistic`` (``y`` 0/1) or ``cox``
    (``time``, ``event``, ``entry``)."""

    kind: Literal["linear", "logistic", "cox"]
    y: np.ndarray | None = None
    time: np.ndarray | None = None
    event: np.ndarray | None = None
    entry: np.ndarray | None = None


@dataclass(frozen=True)
class Term:
    """One column of the substantive model's design as a refusal names it (``Substantive.terms``).

    ``label``: the words for it (the analysis table's name in backticks, or a design term's
    description, "the indicator of stratum 107 of `SDMVSTRA`"). ``column``: the analysis input it
    is made from, which leaving out of the analysis drops. ``design``: a survey design's term, which
    the imputation's model holds and the analysis model does not (``missing.engine_substantive``).
    ``group``: the rows a design indicator marks, in words ("stratum 107"). ``leave_out``: for a
    design term, the sentence that says how the copies are drawn without it, and ``way`` the way
    forward that does it, an exit (``{label, decision}``) the stage offers. ``level``: the design
    variable a design term is made of ("stratum", "psu"; None for the weight), and ``order`` its
    place in the design's own order (a stratum's code, its own values sorted), so a refusal lists
    several in that order."""

    label: str
    column: str | None = None
    design: bool = False
    group: str | None = None
    leave_out: str | None = None
    way: Mapping[str, Any] | None = None
    level: str | None = None
    order: tuple[int, ...] | None = None


@dataclass
class Substantive:
    """The analysis model the imputations must be compatible with.

    ``design(raw)`` maps raw rows (the data frame's columns, every value filled) to the model
    matrix without an intercept; it must be row-local (each output row depends only on its own
    input row). ``refresh(raw)``, when given, refits what the design learns from all the rows
    (an energy residual's line) on the current completed data, once per iteration. ``terms()``,
    when given, names the columns of the design's last matrix, in order (:class:`Term`), so a
    refusal of the outcome draw names them (:func:`_cox_refusal`). ``quiet(event)``, when given,
    names the survey design's groups none of whose rows has the event (``event``: one value per
    row), counted on every row, the reference stratum among them, as that refusal names them."""

    outcome: Outcome
    design: Callable[[pd.DataFrame], np.ndarray]
    refresh: Callable[[pd.DataFrame], None] | None = None
    terms: Callable[[], Sequence[Term]] | None = None
    quiet: Callable[[np.ndarray], Sequence[str]] | None = None


@dataclass
class _Fit:
    """The substantive model's drawn parameters for one variable's update."""

    beta: np.ndarray
    sigma2: float | None = None  # linear
    # cox: the cumulative baseline hazard of the linear predictor less ``shift``, H₀(t)e^shift, at
    # each row's time less at its entry; the hazard of a row is H e^(η − shift) (:func:`_eta`)
    H: np.ndarray | None = None
    shift: float = 0.0  # cox: the largest linear predictor over the rows the model is fit to


# ── the outcome model ────────────────────────────────────────────────────────


def _linear_draw(M: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> _Fit:
    """smcfcs's lm draw: σ*² = σ̂² df/χ²(df), β* ~ N(β̂, σ*²(XᵀX)⁻¹), X = [1, M], every column
    of which is identified (:func:`_draw_outcome` leaves an aliased one out), so df = n − p.

    β* = β̂ + σ* L z with L the lower Cholesky factor of (XᵀX)⁻¹, taken without forming XᵀX or its
    inverse: with Xs = XD⁻¹ (each column scaled to unit length, D the lengths) and R the triangle of
    the QR decomposition of Xs's columns in reverse order (P the reversal, diag R > 0), PRᵀRP =
    XsᵀXs, so (XsᵀXs)⁻¹ = (PR⁻¹P)(PR⁻¹P)ᵀ with PR⁻¹P lower triangular, and L = D⁻¹PR⁻¹P; β̂ comes
    from the same decomposition. Its rounding grows with the condition number of Xs, not of its
    square. Before, σ* L was the Cholesky factor of σ*²((XᵀX)⁺ + 1e−12 I), the same draw in exact
    arithmetic but for that jitter on the diagonal, which was absolute: it inflated the drawn variance
    of a coefficient whose column is on a large scale (income in dollars over 10,000 rows: 5
    times), and it did not cover the rounding of (XᵀX)⁺ where the Sullivan 2021 sweep's model
    further adjusted for 12 covariates (kcal beside indicators, a recorded two-day mean beside its
    two days, near-collinear spline terms) raised "Matrix is not positive definite": the condition
    number of XᵀX was 1.6e13, and the inverse of even the scaled design's cross-products was
    indefinite by rounding at a condition number of 4e9."""
    from scipy.linalg import solve_triangular

    X = np.column_stack([np.ones(len(y)), M])
    n, p = X.shape
    length = np.linalg.norm(X, axis=0)
    length[length == 0] = 1.0
    Q, R = np.linalg.qr((X / length)[:, ::-1])
    sign = np.where(np.diag(R) < 0, -1.0, 1.0)
    Q, R = Q * sign, R * sign[:, None]
    beta = solve_triangular(R, Q.T @ y)[::-1]
    resid = y - (X / length) @ beta
    df = max(n - p, 1)
    sigma2 = float(resid @ resid) / df
    new = sigma2 * df / rng.chisquare(df)
    star = beta + math.sqrt(new) * (_root_of_inverse(R) @ rng.standard_normal(p))
    return _Fit(beta=star / length, sigma2=new)


def _logistic_mle(X: np.ndarray, y: np.ndarray, penalty: np.ndarray) -> np.ndarray:
    """β̂ by Newton with step-halving, ``penalty`` the ridge on each coefficient of ``X``'s columns
    (it keeps a separated fit finite)."""
    p = X.shape[1]
    beta = np.zeros(p)

    def loglik(b: np.ndarray) -> float:
        eta = X @ b
        return float(y @ eta - np.logaddexp(0.0, eta).sum() - 0.5 * penalty @ (b * b))

    ll = loglik(beta)
    for _ in range(100):
        mu = _expit(X @ beta)
        H = (X * (mu * (1 - mu))[:, None]).T @ X + np.diag(penalty)
        g = X.T @ (y - mu) - penalty * beta
        step = np.linalg.lstsq(H, g, rcond=None)[0]
        for _ in range(30):
            trial = beta + step
            new = loglik(trial)
            if new >= ll - 1e-12 * max(1.0, abs(ll)):
                break
            step = step / 2
        beta, ll = trial, new
        if float(np.max(np.abs(step))) < 1e-10:
            break
    return beta


def _root_of_inverse(R: np.ndarray) -> np.ndarray:
    """L, lower triangular with a positive diagonal and L Lᵀ = (P Rᵀ R P)⁻¹, from R upper triangular
    with a positive diagonal (P the reversal of the columns): (PRᵀRP)⁻¹ = (PR⁻¹P)(PR⁻¹P)ᵀ with
    PR⁻¹P lower triangular, so L = PR⁻¹P, taken without inverting PRᵀRP."""
    from scipy.linalg import solve_triangular

    return solve_triangular(R, np.eye(len(R)))[::-1, ::-1]


def _logistic_draw(M: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> _Fit:
    """smcfcs's glm draw: β* ~ N(β̂, I(β̂)⁻¹), X = [1, M], every column of which is identified
    (:func:`_draw_outcome`), I = XᵀWX + Λ the information at the maximum, W the fitted p(1 − p) and
    Λ a ridge of 1e−8 per row on each coefficient but β₀ (it keeps a separated fit finite).

    As the linear draw (:func:`_linear_draw`): with Xs = XD⁻¹ (each column scaled to unit length)
    the fit is made on Xs, its ridge D⁻¹ΛD⁻¹ the same ridge on the coefficients of X, and with R the
    triangle of the QR decomposition of [W^½Xs; (D⁻¹ΛD⁻¹)^½] with its columns in reverse order
    (diag R > 0), the lower Cholesky factor of I⁻¹ is D⁻¹PR⁻¹P. Before, it was the Cholesky factor
    of I⁺ + 1e−12 I: the jitter, absolute, inflated a large-unit coefficient's drawn variance
    (income in dollars over 10,000 rows: 1.9 times its inverse information), and the pseudo-inverse
    of a design's ill-conditioned information was indefinite by rounding (kcal × 1,000 beside a
    code's indicators and its near recode: "Matrix is not positive definite")."""
    X = np.column_stack([np.ones(len(y)), M])
    p = X.shape[1]
    length = np.linalg.norm(X, axis=0)
    length[length == 0] = 1.0
    Xs = X / length
    penalty = np.full(p, 1e-8 * len(y)) / (length * length)
    penalty[0] = 0.0
    beta = _logistic_mle(Xs, y, penalty)
    mu = _expit(Xs @ beta)
    weighted = np.vstack([Xs * np.sqrt(mu * (1 - mu))[:, None], np.diag(np.sqrt(penalty))])
    R = np.linalg.qr(weighted[:, ::-1], mode="r")
    R = R * np.where(np.diag(R) < 0, -1.0, 1.0)[:, None]
    star = beta + _root_of_inverse(R) @ rng.standard_normal(p)
    return _Fit(beta=star / length)


def breslow(eta: np.ndarray, time: np.ndarray, event: np.ndarray, entry: np.ndarray
            ) -> Callable[[np.ndarray], np.ndarray]:
    """Breslow's cumulative baseline hazard H₀ at linear predictors ``eta`` (risk set at τ: entry <
    τ ≤ time), as a function of time: H₀(t) = Σ_{event times τ ≤ t} d_τ / Σ_{at risk at τ} e^{η}."""
    r = np.exp(eta - float(np.max(eta)))
    times = np.unique(time[event > 0])
    by_t = np.argsort(time, kind="stable")
    by_s = np.argsort(entry, kind="stable")
    ct = np.concatenate([np.cumsum(r[by_t][::-1])[::-1], [0.0]])
    cs = np.concatenate([np.cumsum(r[by_s][::-1])[::-1], [0.0]])
    at_risk = (ct[np.searchsorted(time[by_t], times, side="left")]
               - cs[np.searchsorted(entry[by_s], times, side="left")])
    d = np.bincount(np.searchsorted(times, time[event > 0]), minlength=len(times)).astype(float)
    steps = np.cumsum(d / at_risk) * math.exp(-float(np.max(eta)))

    def H0(t: np.ndarray) -> np.ndarray:
        at = np.searchsorted(times, np.asarray(t, dtype=float), side="right")
        return np.concatenate([[0.0], steps])[at]

    return H0


def efron(eta: np.ndarray, time: np.ndarray, event: np.ndarray, entry: np.ndarray
          ) -> Callable[[np.ndarray], np.ndarray]:
    """The Efron-corrected cumulative baseline hazard at linear predictors ``eta`` (risk set at τ:
    entry < τ ≤ time), as a function of time: H₀(t) = Σ_{event times τ ≤ t} Σ_{k=0}^{d−1} 1/(R(τ) −
    (k/d) D(τ)), with d the events at τ, R(τ) = Σ_{at risk} e^{η} and D(τ) = Σ_{events at τ} e^{η}.
    It is what R's ``survival::basehaz(fit, centered = FALSE)`` returns for a Cox fit with Efron
    ties (``survfit.coxph``'s ``ctype`` 2), which ``smcfcs`` calls after replacing the fit's
    coefficients by the draw; with no shared event time it is Breslow's (:func:`breslow`)."""
    shift = float(np.max(eta))
    r = np.exp(eta - shift)
    times = np.unique(time[event > 0])
    by_t = np.argsort(time, kind="stable")
    by_s = np.argsort(entry, kind="stable")
    ct = np.concatenate([np.cumsum(r[by_t][::-1])[::-1], [0.0]])
    cs = np.concatenate([np.cumsum(r[by_s][::-1])[::-1], [0.0]])
    at_risk = (ct[np.searchsorted(time[by_t], times, side="left")]
               - cs[np.searchsorted(entry[by_s], times, side="left")])
    which = np.searchsorted(times, time[event > 0])
    d = np.bincount(which, minlength=len(times))
    tied = np.bincount(which, weights=r[event > 0], minlength=len(times))
    j = np.repeat(np.arange(len(times)), d)  # one term per event: k = 0 … d − 1 at its time
    k = np.arange(len(j)) - np.repeat(np.cumsum(d) - d, d)
    terms = 1.0 / (at_risk[j] - k / d[j] * tied[j])
    steps = np.cumsum(np.bincount(j, weights=terms, minlength=len(times))) * math.exp(-shift)

    def H0(t: np.ndarray) -> np.ndarray:
        at = np.searchsorted(times, np.asarray(t, dtype=float), side="right")
        return np.concatenate([[0.0], steps])[at]

    return H0


class _NoMaximum(ImputationRefused):
    """A Cox fit of the substantive model that did not converge (:func:`_cox_draw`), ``fit`` on the
    columns it was given. Its words name no column: they are the refusal's where the design's
    columns are not named (:func:`_draw_outcome`)."""

    def __init__(self, fit: Any):
        if fit.ordering is not None:
            why = ("its partial likelihood has no maximum on a completed copy of the data: at every "
                   "event time the rows with the event have the highest value of one fixed "
                   "combination of its columns among the rows at risk, so its hazard ratios run "
                   "to zero or infinity")
        else:
            why = ("the Newton fit of its partial likelihood to a completed copy of the data found "
                   "no stable maximum")
        super().__init__(f"The Cox model's coefficients cannot be drawn for the imputations "
                         f"compatible with it, since {why}.")
        self.fit = fit


def _listed(items: Sequence[str]) -> str:
    """"a", "a and b", "a, b and c"."""
    items = list(items)
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"


def _failing(fit: Any, X: np.ndarray) -> tuple[list[int], bool | None]:
    """The columns of ``X`` a Cox fit that did not converge fails along, as the analysis table's
    :func:`~turbotab.core.models.survival.no_maximum` takes them: along its ordering (each column
    whose weight in it is above 1e−6 of the largest), with which way it runs where that is one
    column (True: the rows with the event have its highest value); else the columns whose
    coefficient is past :data:`~turbotab.core.models.survival.DIVERGED` of their SDs (none: [])."""
    from turbotab.core.models.survival import DIVERGED

    d = fit.ordering
    if d is not None:
        weight = np.abs(d) / np.abs(d).max()
        at = [i for i, w in enumerate(weight) if w > 1e-6]
        return at, (bool(d[at[0]] > 0) if len(at) == 1 else None)
    sd = X.std(axis=0)
    return [i for i, (b, s) in enumerate(zip(fit.beta, sd)) if abs(b * s) > DIVERGED], None


def _cox_refusal(M: np.ndarray, out: Outcome, fit: Any, kept: Sequence[int],
                 terms: Sequence[Term], quiet: Callable[[np.ndarray], Sequence[str]] | None = None,
                 complete: np.ndarray | None = None) -> ImputationRefused:
    """The refusal of a Cox fit of the substantive model that did not converge (``fit``, on the
    columns ``kept`` of the design ``M``), naming the design's columns by ``terms``.

    Where the model holds a survey design's terms (``Term.design``) and the analysis model's own
    terms, fit alone to the same completed copy, converge, the model that fails is the
    imputation's, not the analysis model: the words name the design terms the fit fails along
    (several in the design's own order, ``Term.order``), say so, and end with the terms' own words
    for drawing the copies without them (``Term.leave_out``); ``exits`` holds the ways forward
    those words name (``Term.way``: the sample answer), ``design_terms`` those terms' labels, and
    ``columns`` is unset, since leaving an input out would not help. Where the words name no
    stratum by its own indicator (several terms, or a PSU's indicator), they name the design's
    groups no row with the event is in, counted on every row (``quiet``: the reference stratum,
    which no indicator marks, among them; the verifier's T2-B-V4-1). Otherwise the analysis
    model's own terms fail on the copy, and the words name them as the analysis table's
    :func:`~turbotab.core.models.survival.no_maximum` does (:func:`_failing`, else every term), and
    ``columns`` holds the inputs the terms it fails along are made from (none where it names every
    term), which the stage offers to leave out (``stages.modeling._missing_for_table``). Either
    way ``complete`` (the rows every imputed variable records) gives ``complete_cases``
    (:func:`_complete_cases`)."""
    from turbotab.core.models.survival import cox_fit, survival_outcome

    kept = list(kept)
    design = [j for j in kept if terms[j].design]
    own = [j for j in kept if not terms[j].design]
    analysis = list(own)
    if design and own:
        alone = cox_fit(np.ascontiguousarray(M[:, own]),
                        survival_outcome(out.event, out.time, out.entry))
        if not alone.converged:
            fit, kept, design = alone, own, []
    at, up = _failing(fit, M[:, kept])
    along = [kept[i] for i in at]
    ordered = fit.ordering is not None
    if design:
        named = [j for j in along if terms[j].design] or design
        listed = sorted(named, key=lambda j: (terms[j].order is None, terms[j].order or ()))
        one = len(named) == 1
        groups = list(quiet(np.asarray(out.event, dtype=float) > 0.5)) if quiet is not None else []
        counted = f" (no row of {_listed(groups)} has the event)" if groups else ""
        if ordered and along == named and one:
            t = terms[named[0]]
            if counted and t.level != "stratum":
                example = counted  # its own group's words name no stratum: the count does
            else:
                example = ("" if t.group is None else f" (as when every row with the event is in "
                           f"{t.group})" if up else f" (as when no row of {t.group} has the event)")
            how = (f"at every event time the rows with the event have the "
                   f"{'highest' if up else 'lowest'} value of it among the rows at risk{example}, "
                   f"so its hazard ratio runs to {'infinity' if up else 'zero'}")
        elif ordered:
            which = "them" if along == named else _listed([terms[j].label for j in along])
            how = (f"at every event time the rows with the event have the highest value of one "
                   f"fixed combination of {which} among the rows at risk{counted}, so their "
                   f"hazard ratios run to zero or infinity")
        else:
            how = (f"the Newton fit of its partial likelihood found no stable maximum for "
                   f"{'it' if one else 'them'}")
        ways = "".join(f" {w}" for w in dict.fromkeys(terms[j].leave_out for j in named) if w)
        refused = ImputationRefused(
            f"The Cox model the imputations are drawn from has no maximum on a completed copy of "
            f"the data, so its coefficients cannot be drawn: to carry the survey design into the "
            f"copies, it holds {_listed([terms[j].label for j in listed])} beside the analysis "
            f"model's terms, and {how}. The analysis model itself is not the one that fails: "
            f"without the survey design's terms, its partial likelihood has a maximum on the same "
            f"copy.{ways}")
        refused.design_terms = [terms[j].label for j in listed]
        # the ways forward those words name, as the stage offers them (the sample answer)
        offered = {str(terms[j].way.get("label")): dict(terms[j].way) for j in listed
                   if terms[j].way is not None}
        refused.exits = list(offered.values())
        return _complete_cases(refused, M, out, analysis, complete)
    labels = _listed([terms[j].label for j in along or kept])
    if ordered and len(along) == 1:
        why = (f"its partial likelihood has no maximum for {labels} on a completed copy of the "
               f"data: at every event time the rows with the event have the "
               f"{'highest' if up else 'lowest'} {labels} of the rows at risk (as when a group has "
               f"no events), so its hazard ratio runs to {'infinity' if up else 'zero'}")
    elif ordered:
        why = (f"its partial likelihood has no maximum for {labels} on a completed copy of the "
               f"data: at every event time the rows with the event have the highest value of one "
               f"fixed combination of them among the rows at risk, so their hazard ratios run to "
               f"zero or infinity")
    else:
        why = (f"the Newton fit of its partial likelihood to a completed copy of the data found no "
               f"stable maximum for {labels}")
    refused = ImputationRefused(f"The Cox model's coefficients cannot be drawn for the imputations "
                                f"compatible with it, since {why}.")
    # what leaving out would let the imputation run without: the inputs of the terms that fail
    refused.columns = list(dict.fromkeys(terms[j].column for j in along if terms[j].column))
    return _complete_cases(refused, M, out, analysis, complete)


def _complete_cases(refused: ImputationRefused, M: np.ndarray, out: Outcome,
                    columns: Sequence[int], complete: np.ndarray | None) -> ImputationRefused:
    """``refused`` with ``complete_cases``, where the rows every imputed variable records are known
    (``complete``): whether the analysis model's own terms (``columns`` of ``M``) have a maximum
    on those rows alone, so the stage offers complete cases only where they give a table (the
    orchestrator's ruling (d) on the swarm's tail: no woman dies, and complete cases reached the
    analysis table's own refusal)."""
    if complete is not None:
        refused.complete_cases = _has_maximum(M, out, columns, complete)
    return refused


def _has_maximum(M: np.ndarray, out: Outcome, columns: Sequence[int], rows: np.ndarray) -> bool:
    """Whether the Cox model on the design's ``columns`` has a maximum on ``rows`` alone: its
    partial likelihood's fit converges there (``survival.cox_fit``), a column constant or aliased
    on those rows left out as :func:`_draw_outcome` leaves one out. With no row with the event
    among them there is none."""
    from turbotab.core.models.survival import cox_fit, survival_outcome

    rows = np.asarray(rows, dtype=bool)
    event = np.asarray(out.event, dtype=float)[rows]
    if not (event > 0.5).any():
        return False
    X = np.asarray(M, dtype=float)[rows][:, list(columns)]
    kept = [j - 1 for j in identified_columns(np.column_stack([np.ones(len(X)), X])) if j > 0]
    if not kept:
        return True  # no coefficient to have a maximum for
    entry = None if out.entry is None else np.asarray(out.entry, dtype=float)[rows]
    fit = cox_fit(np.ascontiguousarray(X[:, kept]),
                  survival_outcome(event, np.asarray(out.time, dtype=float)[rows], entry))
    return bool(fit.converged)


def _cox_draw(M: np.ndarray, out: Outcome, rng: np.random.Generator) -> _Fit:
    """smcfcs's coxph draw: β* ~ N(β̂, I⁻¹) from the partial likelihood (Efron ties; delayed entry),
    then the Efron-corrected H₀ at β* (:func:`efron`, as smcfcs's ``basehaz`` of the fit holding
    β*) at each row's own times.

    The factor is taken as the linear and logistic draws take theirs: on the columns standardized
    as :func:`~turbotab.core.models.survival.cox_fit` fits them (Z = (M − mean)/SD, every column
    identified: :func:`_draw_outcome`), with L the lower Cholesky factor of PIP (I the information
    at β̂ on Z, P the reversal), the lower Cholesky factor of I⁻¹ is PL⁻ᵀP, and β* = (SD β̂ + PL⁻ᵀP z)
    / SD. Before, it was the Cholesky factor of cox_fit's covariance (the pseudo-inverse of I mapped
    to M's units) plus 1e−12 I, whose absolute jitter inflated a large-unit coefficient's drawn
    variance (income in dollars over 10,000 rows: 2.7 times its inverse information).

    H₀ is taken at η − max η over the rows (``_Fit.shift``) and each row's hazard as H e^(η − shift):
    the same product, but at η itself e^(−max η) overflowed where every η was below −709 (a column
    on a large offset with a negative coefficient, 100,000 plus a standard normal at −0.2 per unit;
    or a column near a combination of the intercept and the others, drawn far out along it).

    A fit that did not converge (cox_fit: a monotone likelihood, a coefficient past its divergence
    bound, or an information singular at the estimate) has no maximum to center the draw on, and the
    draw is refused (:class:`_NoMaximum`), in words that name the design's columns where the
    substantive model names them (:func:`_cox_refusal`)."""
    from turbotab.core.models.survival import _score_info, cox_fit, survival_outcome

    y = survival_outcome(out.event, out.time, out.entry)
    fit = cox_fit(M, y)
    if not fit.converged:
        raise _NoMaximum(fit)
    sd = M.std(axis=0)
    sd[sd == 0] = 1.0
    info = _score_info((M - M.mean(axis=0)) / sd, y, fit.beta * sd)[2]
    L = np.linalg.cholesky(info[::-1, ::-1])
    beta = (fit.beta * sd + _root_of_inverse(L.T) @ rng.standard_normal(len(sd))) / sd
    eta = M @ beta
    shift = float(eta.max())
    H0 = efron(eta - shift, out.time, out.event, out.entry)
    return _Fit(beta=beta, H=H0(out.time) - H0(out.entry), shift=shift)


def identified_columns(X: np.ndarray) -> list[int]:
    """The columns of the design ``X`` (the intercept first) whose coefficients R's ``lm`` reports:
    a column whose part outside the span of the kept columns before it is below 1e−7 of its length
    is aliased and left out (R's ``qr`` rule, :func:`~turbotab.core.models.effects.column_basis`).
    ``glm`` leaves the same columns out, and ``coxph``, which centers its columns, leaves out a
    column constant on every row as aliased with its baseline hazard. One eigenvalue problem first:
    where the scaled Gram matrix shows that no column can be aliased (:data:`NO_ALIAS_EIGENVALUE`),
    every column is kept without the column-by-column pass."""
    from turbotab.core.models.effects import column_basis

    X = np.asarray(X, dtype=float)
    norms = np.linalg.norm(X, axis=0)
    if X.shape[0] >= X.shape[1] and np.all(norms > 0):
        scaled = X / norms
        if float(np.linalg.eigvalsh(scaled.T @ scaled)[0]) > NO_ALIAS_EIGENVALUE:
            return list(range(X.shape[1]))
    return column_basis(X)


def _draw_on(M: np.ndarray, out: Outcome, rng: np.random.Generator) -> _Fit:
    if out.kind == "linear":
        return _linear_draw(M, np.asarray(out.y, dtype=float), rng)
    if out.kind == "logistic":
        return _logistic_draw(M, np.asarray(out.y, dtype=float), rng)
    return _cox_draw(M, out, rng)


def _draw_outcome(M: np.ndarray, out: Outcome, rng: np.random.Generator,
                  terms: Callable[[], Sequence[Term]] | None = None,
                  quiet: Callable[[np.ndarray], Sequence[str]] | None = None,
                  complete: np.ndarray | None = None) -> _Fit:
    """The substantive model's parameters drawn on its design ``M`` (no intercept column).

    A Cox fit that does not converge is refused (:func:`_cox_draw`); where ``terms`` (the
    substantive model's ``terms``) names every column of ``M``, in words that name them
    (:func:`_cox_refusal`, with the substantive model's ``quiet`` and the complete rows,
    ``complete``), else in words that name none.

    An aliased column (:func:`identified_columns`: R's ``lm`` reports its coefficient NA) is left
    out of the fit and its coefficient drawn as zero. The model without it is the same model (the
    same column space, so the same linear predictor for every row while the column stays aliased),
    and the draw on the kept columns is that model's posterior draw: the same random numbers, in
    the same order, as on a design that never held the column. With the column in, (XᵀX)⁺ was
    indefinite to rounding and the linear draw's Cholesky factor failed: the Sullivan 2021 sweep's
    model further adjusted for `race5` beside `RIDRETH3`'s indicators (a recode of them), and its
    exclusion that left one level of `RIAGENDR` (a column constant on every analyzed row). The Cox
    draw's coefficients along the aliased direction could overflow its baseline hazard, and the
    logistic draw put a coefficient on the column."""
    kept = identified_columns(np.column_stack([np.ones(len(M)), M]))
    columns = [j - 1 for j in kept if j > 0]
    try:
        if len(kept) == M.shape[1] + 1:
            return _draw_on(M, out, rng)
        # row-major, as a design built without the column is: BLAS and LAPACK round by the layout
        fit = _draw_on(np.ascontiguousarray(M[:, columns]), out, rng)
    except _NoMaximum as exc:
        named = list(terms()) if terms is not None else []
        if len(named) != M.shape[1]:
            raise
        raise _cox_refusal(M, out, exc.fit, columns, named, quiet, complete) from None
    beta = np.zeros(M.shape[1] + (0 if out.kind == "cox" else 1))
    beta[columns if out.kind == "cox" else kept] = fit.beta
    fit.beta = beta
    return fit


def _eta(fit: _Fit, M: np.ndarray, kind: str) -> np.ndarray:
    if kind == "cox":
        return M @ fit.beta - fit.shift  # the scale of fit.H (``_Fit``)
    return fit.beta[0] + M @ fit.beta[1:]


def outcome_logdensity(fit: _Fit, out: Outcome, eta: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """log f(y | x) for each of ``rows`` (positions in the data), normalized so it is at most 0:
    the log of smcfcs's acceptance probability."""
    if out.kind == "linear":
        dev = np.asarray(out.y, dtype=float)[rows] - eta
        return -(dev * dev) / (2.0 * float(fit.sigma2))
    if out.kind == "logistic":
        y = np.asarray(out.y, dtype=float)[rows]
        return np.where(y > 0.5, -np.logaddexp(0.0, -eta), -np.logaddexp(0.0, eta))
    H = np.asarray(fit.H, dtype=float)[rows]
    d = np.asarray(out.event, dtype=float)[rows]
    hz = H * np.exp(np.clip(eta, -700, 700))
    with np.errstate(divide="ignore"):
        event = 1.0 + np.log(np.where(H > 0, hz, 1.0)) - hz
    return np.where(d > 0.5, np.where(H > 0, event, 0.0), -hz)


# ── the sampler ──────────────────────────────────────────────────────────────


@dataclass
class _State:
    """One chain's current values: imputation-scale columns (logs taken), their predictor blocks."""

    values: dict[str, np.ndarray]
    levels: dict[str, list[Any]]
    kinds: dict[str, str]
    blocks: dict[str, np.ndarray] = field(default_factory=dict)

    def refresh(self, column: str) -> None:
        v = self.values[column]
        if self.kinds.get(column) == "categorical":
            levels = self.levels[column]
            block = (np.column_stack([(v == lv).astype(float) for lv in levels[1:]])
                     if len(levels) > 1 else np.zeros((len(v), 0)))
        else:
            block = _standardize(v.astype(float).reshape(-1, 1))
        self.blocks[column] = block

    def predictors(self, without: str, skip: Sequence[str] = ()) -> np.ndarray:
        gone = {without, *skip}
        parts = [b for c, b in self.blocks.items() if c not in gone and b.shape[1]]
        n = len(next(iter(self.values.values())))
        return np.column_stack([np.ones(n), *parts]) if parts else np.ones((n, 1))


def _kind_of(series: pd.Series) -> str:
    from turbotab.core.methods.imputation import column_kind

    return column_kind(series)


def _truncnorm(mu: np.ndarray, sigma: float | np.ndarray, lo: np.ndarray, hi: np.ndarray,
               rng: np.random.Generator) -> np.ndarray:
    """Draws of N(mu, σ²) truncated to [lo, hi] (either may be infinite), by the inverse CDF."""
    from scipy import stats

    mu = np.asarray(mu, dtype=float)
    sigma = np.broadcast_to(np.asarray(sigma, dtype=float), mu.shape)
    lo = np.broadcast_to(np.asarray(lo, dtype=float), mu.shape)
    hi = np.broadcast_to(np.asarray(hi, dtype=float), mu.shape)
    plain = ~np.isfinite(lo) & ~np.isfinite(hi)
    out = np.empty_like(mu)
    if plain.any():
        out[plain] = mu[plain] + sigma[plain] * rng.standard_normal(int(plain.sum()))
    rest = ~plain
    if rest.any():
        a = (lo[rest] - mu[rest]) / sigma[rest]
        b = (hi[rest] - mu[rest]) / sigma[rest]
        draws = stats.truncnorm.rvs(a, b, loc=mu[rest], scale=sigma[rest], random_state=rng)
        out[rest] = np.clip(draws, lo[rest], hi[rest])
    return out


class _Chain:
    """One imputation's chain over ``data`` (see :func:`impute`)."""

    def __init__(self, data: pd.DataFrame, variables: Sequence[Variable], mode: Mode,
                 substantive: Substantive | None, identity: Identity | None,
                 units: np.ndarray | None, rng: np.random.Generator, rjlimit: int,
                 kinds: Mapping[str, str] | None = None):
        self.data = data
        self.n = len(data)
        self.vars = {v.name: v for v in variables}
        self.mode = mode
        self.sub = substantive
        self.identity = identity
        self.rng = rng
        self.rjlimit = rjlimit
        self.failures = 0
        # ``recorded``: the values the data holds; ``miss``: the cells this chain draws
        self.recorded = {v.name: data[v.name].notna().to_numpy() for v in variables}
        self.miss = {name: ~rec for name, rec in self.recorded.items()}
        for s in (identity.factors if identity is not None else {}):
            self.recorded.setdefault(s, data[s].notna().to_numpy())  # a complete source
        # how each column enters the models: an imputed variable by its kind, a complete column as
        # ``kinds`` declares it (a numeric code the user declared a category is its indicators, as
        # :func:`model_width` counts it), else by its values
        declared = dict(kinds or {})
        modeled: dict[str, str] = {}
        levels: dict[str, list[Any]] = {}
        values: dict[str, np.ndarray] = {}
        energy = identity.energy if identity is not None else None
        for c in data.columns:
            if c == energy:
                continue
            var = self.vars.get(c)
            kind = var.kind if var is not None else (declared.get(c) or _kind_of(data[c]))
            modeled[c] = "numeric" if kind == "censored" else kind
            if kind == "categorical":
                levels[c] = sorted(pd.unique(data[c].dropna()), key=str)
                values[c] = data[c].to_numpy(dtype=object).copy()
            else:
                x = pd.to_numeric(data[c], errors="coerce").to_numpy(dtype=float).copy()
                values[c] = self.scaled(c, x)
        self.st = _State(values=values, levels=levels, kinds=modeled)
        self.units = None if units is None else np.asarray(units, dtype=np.int64)
        if self.units is not None:
            self.G = int(self.units.max()) + 1 if len(self.units) else 0
            order = np.argsort(self.units, kind="stable")
            starts = np.flatnonzero(np.r_[True, np.diff(self.units[order]) != 0])
            self.first = np.zeros(self.G, dtype=np.int64)
            self.first[self.units[order[starts]]] = order[starts]
            self.unit_miss = {}
            for name, v in self.vars.items():
                if v.unit_level:
                    umiss = np.ones(self.G, dtype=bool)
                    umiss[np.unique(self.units[self.recorded[name]])] = False
                    self.unit_miss[name] = umiss
        if identity is not None:
            self.E = pd.to_numeric(data[identity.energy], errors="coerce").to_numpy(dtype=float)
            self.E_obs = np.isfinite(self.E)
            self.derived = self.E_obs & ~self.recorded[identity.other]  # other follows E
            self.miss[identity.other] = ~self.E_obs  # other is drawn where E is blank

    # -- scales ---------------------------------------------------------------

    def _logged(self, name: str) -> bool:
        var = self.vars.get(name)
        return var is not None and (var.log or (var.censoring is not None and var.censoring.log))

    def raw(self, name: str, x: np.ndarray) -> np.ndarray:
        return np.exp(x) if self._logged(name) else x

    def scaled(self, name: str, x: np.ndarray) -> np.ndarray:
        if not self._logged(name):
            return x
        with np.errstate(divide="ignore", invalid="ignore"):
            if self.vars[name].log:
                return np.log(np.where(np.isfinite(x) & (x > 0), x, np.nan))
            return np.log(np.clip(x, 1e-300, None))

    def raw_frame(self, rows: np.ndarray | None = None,
                  override: Mapping[str, np.ndarray] | None = None) -> pd.DataFrame:
        """The raw values of ``rows`` (default every row), ``override`` giving columns'
        imputation-scale values for those rows; under the identity total energy is derived where it
        is blank, and other follows a recorded total."""
        idx = np.arange(self.n) if rows is None else np.asarray(rows)
        out: dict[str, Any] = {}
        for c in self.data.columns:
            if self.identity is not None and c == self.identity.energy:
                continue
            v = override[c] if override is not None and c in override else self.st.values[c][idx]
            out[c] = v if self.st.kinds[c] == "categorical" else self.raw(c, v)
        if self.identity is not None:
            ident = self.identity
            parts = sum(float(f) * out[s] for s, f in ident.factors.items())
            recorded = self.E_obs[idx]
            out[ident.energy] = np.where(recorded, self.E[idx], parts + out[ident.other])
            out[ident.other] = np.where(recorded, self.E[idx] - parts, out[ident.other])
        return pd.DataFrame({c: out[c] for c in self.data.columns}, index=self.data.index[idx])

    def derive_other(self) -> None:
        """Other on the rows where it follows a recorded total, from the current sources."""
        ident = self.identity
        if ident is None or not self.derived.any():
            return
        rows = self.derived
        parts = sum(float(f) * self.raw(s, self.st.values[s][rows]) for s, f in ident.factors.items())
        other = self.E[rows] - parts
        if self.vars[ident.other].log:  # only rows whose recorded sources reach E hit the floor
            other = np.maximum(other, OTHER_FLOOR * np.abs(self.E[rows]))
        v = self.st.values[ident.other].copy()
        v[rows] = self.scaled(ident.other, other)
        self.st.values[ident.other] = v
        self.st.refresh(ident.other)

    def infeasible_rows(self) -> int:
        """Rows with a recorded total whose recorded sources alone already reach it (less the
        floor a logged "other" keeps), so that no draw of their missing sources leaves other above
        zero."""
        ident = self.identity
        if ident is None:
            return 0
        known = np.zeros(self.n)
        missing_source = np.zeros(self.n, dtype=bool)
        for s, f in ident.factors.items():
            rec = self.recorded[s]
            x = pd.to_numeric(self.data[s], errors="coerce").to_numpy(dtype=float)
            known += np.where(rec, float(f) * np.nan_to_num(x), 0.0)
            missing_source |= ~rec
        with np.errstate(invalid="ignore"):
            return int(np.sum(self.E_obs & missing_source & (self.E - known - self._floor() <= 0)))

    def bounds(self, name: str, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The imputation-scale interval each of ``rows`` is drawn in; ``hi`` NaN where a recorded
        total leaves no room."""
        var = self.vars[name]
        lo = np.full(len(rows), -np.inf)
        hi = np.full(len(rows), np.inf)
        if var.censoring is not None:
            hi[:] = math.log(var.censoring.limit) if var.censoring.log else var.censoring.limit
        if var.lower is not None and not var.log:
            lo[:] = var.lower
        ident = self.identity
        if ident is not None and name in ident.factors:
            if not var.log:
                lo = np.maximum(lo, 0.0)
            recorded = self.E_obs[rows]
            if recorded.any():
                r = rows[recorded]
                rest = sum((float(f) * self.raw(s, self.st.values[s][r])
                            for s, f in ident.factors.items() if s != name), np.zeros(len(r)))
                floor = OTHER_FLOOR * np.abs(self.E[r]) if self.vars[ident.other].log else 0.0
                room = (self.E[r] - rest - floor) / float(ident.factors[name])
                top = np.full(len(r), np.nan)
                ok = room > 0
                top[ok] = np.log(room[ok]) if var.log else room[ok]
                hi[recorded] = np.where(np.isnan(top), np.nan, np.minimum(hi[recorded], top))
        if ident is not None and name == ident.other and not var.log:
            lo = np.maximum(lo, 0.0)
        return lo, hi

    # -- start ------------------------------------------------------------------

    def start(self) -> None:
        """Every chain starts from random draws of each variable's recorded values (per unit for a
        unit-level one), a non-detect below its limit, a source within a recorded total."""
        rng = self.rng
        for name, var in self.vars.items():
            miss = self.miss[name]
            if not miss.any():
                continue
            v = self.st.values[name].copy()
            rec = self.recorded[name]
            observed = v[rec]
            if not len(observed):
                refused = ImputationRefused(f"`{name}` has no recorded value to start its "
                                            f"imputations from.")
                refused.columns = [name]  # what leaving out would let the imputation run without
                raise refused
            if var.unit_level and self.units is not None:
                umiss = self.unit_miss[name]
                per_unit = self._unit_values(name, v)
                per_unit[umiss] = observed[rng.integers(0, len(observed), int(umiss.sum()))]
                v[miss] = per_unit[self.units[miss]]
            else:
                v[miss] = observed[rng.integers(0, len(observed), int(miss.sum()))]
            if var.censoring is not None:
                limit = math.log(var.censoring.limit) if var.censoring.log else var.censoring.limit
                spread = float(np.nanstd(observed.astype(float))) or 1.0
                v[miss] = limit - (math.log(2.0) if var.censoring.log else spread)
            self.st.values[name] = v
        if self.identity is not None:
            self.feasible_start()
        for c in self.st.values:
            self.st.refresh(c)
        self.derive_other()

    def _drawn(self, name: str) -> np.ndarray:
        """The rows whose ``name`` this chain draws: its blanks, or for a unit-level variable the
        rows of the units that record it nowhere (the rest hold their unit's recorded value)."""
        if name not in self.vars:
            return np.zeros(self.n, dtype=bool)
        if self.vars[name].unit_level and self.units is not None:
            return self.unit_miss[name][self.units]
        return self.miss[name].copy()

    def _floor(self) -> np.ndarray | float:
        ident = self.identity
        return OTHER_FLOOR * np.abs(self.E) if self.vars[ident.other].log else 0.0

    def room(self) -> tuple[np.ndarray, np.ndarray]:
        """Per row, under a recorded total: the kcal it leaves the drawn sources (E less the
        sources it records and the floor), and the drawn sources' current kcal."""
        ident = self.identity
        known = np.zeros(self.n)
        drawn = np.zeros(self.n)
        for s, f in ident.factors.items():
            x = self.raw(s, np.asarray(self.st.values[s], dtype=float))
            mine = self._drawn(s)
            known += np.where(mine, 0.0, float(f) * np.nan_to_num(x))
            drawn += np.where(mine, float(f) * np.nan_to_num(x), 0.0)
        with np.errstate(invalid="ignore"):
            return self.E - known - self._floor(), drawn

    def feasible_start(self) -> None:
        """Every row with a recorded total starts inside it, jointly over its blank sources: where
        their random starts together leave no room, each is scaled by the share that puts their
        kcal at :data:`START_SHARE` of the room (a unit-level source by its unit's smallest share,
        so it stays one value per unit). A row whose recorded sources alone reach the total has no
        room: a source drawn on its own scale starts at zero, a logged one at its smallest recorded
        value. Every truncated draw after this keeps the row inside (its interval holds its current
        value), so the identity holds in every copy wherever it can."""
        ident = self.identity
        room, drawn = self.room()
        with np.errstate(invalid="ignore", divide="ignore"):
            over = self.E_obs & (room > 0) & (drawn > 0) & (drawn >= room)
            share = np.where(over, START_SHARE * room / np.where(drawn > 0, drawn, 1.0), 1.0)
            none = self.E_obs & ~(room > 0)
        for s in ident.factors:
            mine = self._drawn(s)
            if not mine.any():
                continue
            var = self.vars[s]
            factor = share
            if var.unit_level and self.units is not None:
                per_unit = np.ones(self.G)
                np.minimum.at(per_unit, self.units[mine], share[mine])
                factor = per_unit[self.units]
            v = np.asarray(self.st.values[s], dtype=float).copy()
            scale = mine & (factor < 1.0)
            if scale.any():
                v[scale] = self.scaled(s, self.raw(s, v[scale]) * factor[scale])
            stuck = mine & none
            if stuck.any():
                recorded = v[self.recorded[s]]
                floor = (float(np.min(recorded)) if self._logged(s) and len(recorded)
                         else (-np.inf if self._logged(s) else 0.0))
                if np.isfinite(floor):
                    if var.unit_level and self.units is not None:
                        units = np.zeros(self.G, dtype=bool)
                        units[self.units[stuck]] = True
                        stuck = mine & units[self.units]
                    v[stuck] = floor
            self.st.values[s] = v

    def _unit_values(self, name: str, v: np.ndarray) -> np.ndarray:
        """Each unit's value of a unit-level variable: a recorded one where the unit has it, else
        the unit's first row's current value."""
        out = v[self.first].copy()
        rows = np.flatnonzero(self.recorded[name])
        out[self.units[rows]] = v[rows]
        return out

    # -- the predictors ---------------------------------------------------------

    def _cluster_means(self, without: str, skip: Sequence[str]) -> list[np.ndarray]:
        """Unit means (G rows) of every other row-level column's block."""
        if self.units is None:
            return []
        counts = np.bincount(self.units, minlength=self.G).astype(float)
        counts[counts == 0] = 1.0
        out = []
        for c, block in self.st.blocks.items():
            var = self.vars.get(c)
            if c == without or c in skip or (var is not None and var.unit_level) or not block.shape[1]:
                continue
            out.append(np.column_stack([np.bincount(self.units, weights=block[:, j], minlength=self.G)
                                        for j in range(block.shape[1])]) / counts[:, None])
        return out

    def _unit_means(self, block: np.ndarray) -> np.ndarray:
        counts = np.bincount(self.units, minlength=self.G).astype(float)
        counts[counts == 0] = 1.0
        return np.column_stack([np.bincount(self.units, weights=block[:, j], minlength=self.G)
                                for j in range(block.shape[1])]) / counts[:, None]

    def energy_terms(self, target: str) -> np.ndarray | None:
        """An energy source's covariate model reads the total in place of "other" (None for any
        other variable, or with no recorded total): on a row whose total is recorded, other is
        E − Σ fⱼNⱼ and so a function of the source itself, and conditioning the source's draw on
        its own last value through it held each draw near its start (the repair's simulation:
        imputed protein −4 to −5.5 SE low with two sources blank on the log scale, −1 to −2 SE with
        one; with the total in its place, within ±1.4 SE). So the model's terms are the recorded
        total on its rows (log E for a logged source) and, on a row whose total is blank, the
        drawn "other" (there a free variable, the total derived from it) with an indicator of the
        blank total; each slot standardized over its own rows, zero elsewhere."""
        ident = self.identity
        if ident is None or target not in ident.factors or not self.E_obs.any():
            return None
        E = np.asarray(self.E, dtype=float)
        rec = self.E_obs
        logged = self._logged(target) and bool(np.all(E[rec] > 0))
        e = np.zeros(self.n)
        e[rec] = np.log(E[rec]) if logged else E[rec]
        cols = [self._slot(e, rec)]
        blank = ~rec
        if blank.any():
            other = np.asarray(self.st.values[ident.other], dtype=float)
            cols += [self._slot(other, blank), blank.astype(float)]
        return np.column_stack(cols)

    @staticmethod
    def _slot(values: np.ndarray, rows: np.ndarray) -> np.ndarray:
        out = np.zeros(len(values))
        v = values[rows]
        sd = float(np.std(v)) if len(v) else 0.0
        out[rows] = (v - float(np.mean(v))) / sd if sd > 0 else 0.0
        return out

    def design_rows(self, target: str, skip: Sequence[str]) -> np.ndarray:
        """A row-level covariate model's design: the intercept, every other column's block and,
        with units, the unit means of the other row-level columns and the mean of the target over
        the row's other rows in its unit (:meth:`_own_mean`). An energy source reads the total in
        place of "other" (:meth:`energy_terms`)."""
        energy = self.energy_terms(target)
        if energy is not None:
            skip = [*skip, self.identity.other]
        X = self.st.predictors(target, skip)
        means = self._cluster_means(target, skip)
        if means:
            X = np.column_stack([X, *[_standardize(mm[self.units]) for mm in means]])
        if energy is not None:
            X = np.column_stack([X, energy])
            if self.units is not None:
                X = np.column_stack([X, _standardize(self._unit_means(energy)[self.units])])
        own = self._own_mean(target) if self.units is not None else None
        if own is not None:
            X = np.column_stack([X, own])
        return X

    def _own_mean(self, target: str) -> np.ndarray | None:
        """The mean of ``target`` over each row's other rows in its unit (standardized; a unit of
        one row reads the overall mean), the unit's own level of the variable."""
        block = self.st.blocks.get(target)
        if block is None or block.shape[1] != 1:
            return None
        v = block[:, 0]
        total = np.bincount(self.units, weights=v, minlength=self.G)
        size = np.bincount(self.units, minlength=self.G).astype(float)
        others = size[self.units] - 1
        loo = np.where(others > 0, (total[self.units] - v) / np.maximum(others, 1), float(v.mean()))
        return _standardize(loo.reshape(-1, 1))[:, 0]

    def design_units(self, target: str, skip: Sequence[str]) -> np.ndarray:
        """A unit-level covariate model's design, one row per unit: the intercept, the other
        unit-level columns and the unit means of the row-level ones."""
        energy = self.energy_terms(target)
        if energy is not None:
            skip = [*skip, self.identity.other]
        parts = [np.ones(self.G)]
        for c, block in self.st.blocks.items():
            var = self.vars.get(c)
            if c == target or c in skip or not block.shape[1] or not (var is not None and var.unit_level):
                continue
            parts.append(block[self.first])
        parts.extend(_standardize(mm) for mm in self._cluster_means(target, skip))
        if energy is not None:
            parts.append(_standardize(self._unit_means(energy)))
        return np.column_stack(parts)

    # -- one variable's update --------------------------------------------------

    def update(self, name: str, skip: Sequence[str]) -> None:
        var = self.vars[name]
        if self.units is not None and var.unit_level:
            self._update_units(name, var, skip)
        elif self.mode == "fcs":
            self._update_fcs(name, var, skip)
        else:
            self._update_smc(name, var, skip)
        if self.identity is not None and name in self.identity.factors:
            self.derive_other()

    def _set(self, name: str, rows: np.ndarray, values: np.ndarray) -> None:
        v = self.st.values[name].copy()
        v[rows] = values
        self.st.values[name] = v
        self.st.refresh(name)

    def _bounded(self, mu: np.ndarray, sigma: float, lo: np.ndarray, hi: np.ndarray,
                 current: np.ndarray) -> np.ndarray:
        """Truncated draws; a row with no room (``hi`` NaN) takes its floor, else keeps its value."""
        out = np.array(current, dtype=float, copy=True)
        ok = ~np.isnan(hi)
        if ok.any():
            out[ok] = _truncnorm(mu[ok], sigma, lo[ok], hi[ok], self.rng)
        stuck = ~ok
        out[stuck] = np.where(np.isfinite(lo[stuck]), lo[stuck], out[stuck])
        return out

    def _update_fcs(self, name: str, var: Variable, skip: Sequence[str]) -> None:
        """mice's step: the model on the recorded rows, its parameters drawn, then each blank."""
        from turbotab.core.methods.imputation import (CensoredFitRefused, _binary, _categorical,
                                                      _censored)

        X = self.design_rows(name, skip)
        miss = self.miss[name]
        rows = np.flatnonzero(miss)
        cur = self.st.values[name]
        if var.kind == "categorical":
            new = _categorical(X, cur, self.st.levels[name], miss, self.rng)
        elif var.kind == "binary":
            new = _from01(_binary(X, _as01(cur, self.data[name]), miss, self.rng), self.data[name])
        elif var.kind == "censored":
            try:
                new = self.scaled(name, _censored(X, self.raw(name, cur.astype(float)), miss,
                                                  var.censoring, self.rng))
            except CensoredFitRefused as exc:
                # A censored-normal fit with no certified maximum (``imputation.tobit_fit``), or a
                # draw from it whose arithmetic fails, is refused in words naming the column,
                # never drawn.
                raise exc.of(name) from None
            except (np.linalg.LinAlgError, ArithmeticError) as exc:
                raise CensoredFitRefused(why=f"{type(exc).__name__}: {exc}").of(name) from None
        else:
            rec = self.recorded[name]
            _, star, sigma = _norm_draw(X[rec], cur[rec].astype(float), self.rng)
            lo, hi = self.bounds(name, rows)
            new = self._bounded(X[rows] @ star, sigma, lo, hi, cur[rows])
        self._set(name, rows, new)

    # SMC-FCS ---------------------------------------------------------------------

    def _outcome(self) -> _Fit:
        M = np.asarray(self.sub.design(self.raw_frame()), dtype=float)
        # the complete cases: the rows every imputed variable records
        complete = np.logical_and.reduce([self.recorded[v] for v in self.vars], initial=True)
        return _draw_outcome(M, self.sub.outcome, self.rng, self.sub.terms, self.sub.quiet,
                             np.broadcast_to(complete, (self.n,)))

    def _covariate_draw(self, name: str, var: Variable, X: np.ndarray, y: np.ndarray
                        ) -> tuple[np.ndarray, float | None]:
        """smcfcs's covariate-model draw, fit on every row of ``X`` with ``y`` (current values):
        the fitted means (a number, with σ*) or level probabilities over the rows of ``X``."""
        if var.kind == "binary":
            beta, V = _logistic(X, _as01(y, self.data[name]))
            root = np.linalg.cholesky(V + np.eye(len(V)) * 1e-12)
            p1 = _expit(X @ (beta + root @ self.rng.standard_normal(len(beta))))
            return np.column_stack([1 - p1, p1]), None
        if var.kind == "categorical":
            from sklearn.linear_model import LogisticRegression

            from turbotab.core.methods.imputation import LOGIT_RIDGE

            levels = self.st.levels[name]
            boot = self.rng.integers(0, len(X), len(X))
            labels = np.asarray([str(v) for v in y[boot]], dtype=object)
            probs = np.zeros((len(X), len(levels)))
            at = {str(lv): j for j, lv in enumerate(levels)}
            present = sorted(set(labels))
            if len(present) == 1:
                probs[:, at[present[0]]] = 1.0
                return probs, None
            model = LogisticRegression(C=1.0 / LOGIT_RIDGE, max_iter=500)
            model.fit(X[boot][:, 1:], labels)
            got = model.predict_proba(X[:, 1:])
            for k, cls in enumerate(model.classes_):
                probs[:, at[cls]] = got[:, k]
            return probs, None
        _, star, sigma = _norm_draw(X, y.astype(float), self.rng)
        return X @ star, sigma

    def _update_smc(self, name: str, var: Variable, skip: Sequence[str]) -> None:
        X = self.design_rows(name, skip)
        fitted, sigma = self._covariate_draw(name, var, X, self.st.values[name])
        fit = self._outcome()
        rows = np.flatnonzero(self.miss[name])
        if var.kind in ("binary", "categorical"):
            self._direct(name, var, rows, fitted[rows], fit)
            return
        lo, hi = self.bounds(name, rows)
        self._reject(name, rows, fitted[rows], float(sigma), lo, hi, fit, groups=None)

    def _level_values(self, name: str, var: Variable, j: int, count: int) -> np.ndarray:
        if var.kind == "binary":
            return _from01(np.full(count, float(j)), self.data[name])
        return np.full(count, self.st.levels[name][j], dtype=object)

    def _direct(self, name: str, var: Variable, rows: np.ndarray, probs: np.ndarray, fit: _Fit,
                groups: np.ndarray | None = None) -> None:
        """A binary or categorical variable drawn from p(x | z) f(y | x, z) over its levels;
        ``probs`` per draw (a row, or with ``groups`` a unit whose rows' densities multiply)."""
        out = self.sub.outcome
        n_levels = probs.shape[1]
        count = len(probs)
        logw = np.zeros((count, n_levels))
        for j in range(n_levels):
            frame = self.raw_frame(rows, {name: self._level_values(name, var, j, len(rows))})
            eta = _eta(fit, np.asarray(self.sub.design(frame), dtype=float), out.kind)
            dens = outcome_logdensity(fit, out, eta, rows)
            if groups is not None:
                dens = np.bincount(groups, weights=dens, minlength=count)
            logw[:, j] = dens + np.log(np.clip(probs[:, j], 1e-300, None))
        logw -= logw.max(axis=1, keepdims=True)
        w = np.exp(logw)
        w /= w.sum(axis=1, keepdims=True)
        pick = (self.rng.random(count)[:, None] > np.cumsum(w, axis=1)).sum(axis=1)
        pick = pick.clip(0, n_levels - 1)
        if var.kind == "binary":
            chosen = _from01(pick.astype(float), self.data[name])
        else:
            chosen = np.asarray([self.st.levels[name][k] for k in pick], dtype=object)
        self._set(name, rows, chosen if groups is None else chosen[groups])

    def _logp(self, name: str, rows: np.ndarray, values: np.ndarray, fit: _Fit,
              groups: np.ndarray | None, count: int) -> np.ndarray:
        """log f(y | x) of each draw: a row's, or the sum over a group's rows."""
        frame = self.raw_frame(rows, {name: values})
        eta = _eta(fit, np.asarray(self.sub.design(frame), dtype=float), self.sub.outcome.kind)
        per = outcome_logdensity(fit, self.sub.outcome, eta, rows)
        return per if groups is None else np.bincount(groups, weights=per, minlength=count)

    def _reject(self, name: str, rows: np.ndarray, mu: np.ndarray, sigma: float, lo: np.ndarray,
                hi: np.ndarray, fit: _Fit, groups: np.ndarray | None) -> None:
        """Rejection sampling of a number's draws: one per row of ``rows``, or with ``groups``
        (each row's draw) one per group, kept by the product of its rows' densities."""
        count = len(mu)
        current = self.st.values[name][rows].astype(float)
        if groups is None:
            draw = current.copy()
        else:
            draw = np.full(count, np.nan)
            draw[groups] = current
        stuck = np.isnan(hi)
        draw[stuck] = np.where(np.isfinite(lo[stuck]), lo[stuck], draw[stuck])
        need = np.flatnonzero(~stuck)
        tried = np.zeros(count, dtype=np.int64)
        while len(need):
            # Each remaining draw gets K candidates at once, tried in order: the first kept is the
            # draw, exactly as one candidate at a time would give (a batch only saves design calls).
            if groups is None:
                at, local = need, np.arange(len(need))
                per_candidate = len(need)
            else:
                at = np.flatnonzero(np.isin(groups, need))
                local = np.searchsorted(need, groups[at])
                per_candidate = len(at)
            budget = self.rjlimit + FIRST_TRIES
            K = int(max(1, min(BATCH_ROWS // max(per_candidate, 1),
                               budget - int(tried[need].max()))))
            cand = _truncnorm(np.repeat(mu[need], K), sigma, np.repeat(lo[need], K),
                              np.repeat(hi[need], K), self.rng).reshape(len(need), K)
            row_idx = np.repeat(at if groups is not None else need, K)
            draw_idx = np.repeat(local, K)
            cand_idx = np.tile(np.arange(K), len(at) if groups is not None else len(need))
            values = cand[draw_idx, cand_idx]
            logp = self._logp(name, rows[row_idx], values, fit, draw_idx * K + cand_idx,
                              len(need) * K).reshape(len(need), K)
            kept = np.log(self.rng.random((len(need), K))) <= logp
            hit = kept.any(axis=1)
            first = np.argmax(kept, axis=1)
            draw[need[hit]] = cand[hit, first[hit]]
            draw[need[~hit]] = cand[~hit, -1]  # a draw no candidate was kept for: its last
            tried[need] += K
            left = need[~hit]
            self.failures += int(np.sum(tried[left] >= budget))
            need = left[tried[left] < budget]
        self._set(name, rows, draw if groups is None else draw[groups])

    # unit level ------------------------------------------------------------------

    def _update_units(self, name: str, var: Variable, skip: Sequence[str]) -> None:
        """A time-invariant variable, imputed once per unit and copied to each of its rows."""
        from turbotab.core.methods.imputation import _binary, _categorical

        umiss = self.unit_miss[name]
        if not umiss.any():
            return
        U = self.design_units(name, skip)
        per_unit = self._unit_values(name, self.st.values[name])
        missing_units = np.flatnonzero(umiss)
        rows = np.flatnonzero(np.isin(self.units, missing_units))
        lo = np.full(len(missing_units), var.lower if var.lower is not None and not var.log
                     else -np.inf)
        hi = np.full(len(missing_units), np.inf)
        if self.identity is not None and name in self.identity.factors:
            # one value for the unit, inside the recorded total of each of its rows
            local = np.searchsorted(missing_units, self.units[rows])
            row_lo, row_hi = self.bounds(name, rows)
            np.maximum.at(lo, local, row_lo)
            np.minimum.at(hi, local, np.where(np.isnan(row_hi), -np.inf, row_hi))
            hi = np.where(np.isneginf(hi), np.nan, hi)
        if self.mode == "fcs":
            if var.kind == "categorical":
                new = _categorical(U, per_unit, self.st.levels[name], umiss, self.rng)
            elif var.kind == "binary":
                new = _from01(_binary(U, _as01(per_unit, self.data[name]), umiss, self.rng),
                              self.data[name])
            else:
                _, star, sigma = _norm_draw(U[~umiss], per_unit[~umiss].astype(float), self.rng)
                new = self._bounded(U[umiss] @ star, sigma, lo, hi, per_unit[umiss])
            filled = per_unit.copy()
            filled[umiss] = new
            self._set(name, rows, filled[self.units[rows]])
            return
        fitted, sigma = self._covariate_draw(name, var, U, per_unit)
        fit = self._outcome()
        local = np.searchsorted(missing_units, self.units[rows])
        if var.kind in ("binary", "categorical"):
            self._direct(name, var, rows, fitted[missing_units], fit, groups=local)
            return
        self._reject(name, rows, fitted[missing_units], float(sigma), lo, hi, fit, groups=local)

    # -- the chain ----------------------------------------------------------------

    def run(self, order: Sequence[str], sweeps: int, skip: Sequence[str],
            tick: Callable[[], None]) -> pd.DataFrame:
        self.start()
        for _ in range(sweeps):
            if self.sub is not None and self.sub.refresh is not None:
                self.sub.refresh(self.raw_frame())
            for name in order:
                tick()
                self.update(name, skip)
        return self.raw_frame()


# ── how wide the imputation model may be ─────────────────────────────────────

# The survey design's columns among the imputation model's (``missing.design_variables``): its
# strata and its PSUs within them as categories, and its weight.
DESIGN_TERMS = ("__stratum", "__psu", "__weight")
SHOWN_MODELS = 3  # incomplete columns a refusal names after the first; the rest are counted
WIDTH_REFUSAL = "imputation_too_wide"
# The lead of the imputation's refusal, which the answer's (:func:`_imputation_has_its_rows`) and
# the fit's table's (``stages.modeling._missing_for_table``) both read.
REFUSED_LEAD = "Multiple imputation cannot run on these data: "


@dataclass
class ModelWidth:
    """The imputation model's width and each incomplete column's model against the
    rows-per-parameter rule (:func:`model_width`)."""

    width: int  # every model column of its predictors: R's ncol(model.matrix(~ ., data)) − 1
    columns: dict[str, int]  # each predictor's model columns (a category's levels less one)
    models: list[ModelSize]  # each column the chain draws, as its model is fit
    design: int = 0  # of ``width``, the survey design's terms

    def short(self) -> list[ModelSize]:
        """The models with fewer rows per parameter than the rule asks, the furthest short first."""
        return sorted((m for m in self.models if m.short), key=lambda m: m.per / m.needed)

    def refused(self) -> bool:
        """Whether the sampler refuses it: wider than
        :data:`~turbotab.core.methods.imputation.MAX_MODEL_COLUMNS` columns with a model short of
        the rule. Up to that width it is drawn whatever its rows, as before the rule, which so
        removes nothing the fixed cap drew (the orchestrator's ruling, 2026-11-02)."""
        return self.width > MAX_MODEL_COLUMNS and bool(self.short())


def chain_kinds(data: pd.DataFrame, variables: Sequence[Variable],
                kinds: Mapping[str, str] | None = None) -> dict[str, str]:
    """How each column of ``data`` enters the chain's models: an imputed variable by its kind, a
    complete column as ``kinds`` declares it (a numeric code column the user declared a category)
    or else by its values."""
    declared = dict(kinds or {})
    own = {v.name: v.kind for v in variables}
    return {c: own.get(c) or declared.get(c) or _kind_of(data[c]) for c in data.columns}


def _blocks(data: pd.DataFrame, kinds: Mapping[str, str], energy: str | None) -> dict[str, int]:
    """Each column's model columns as the chain encodes it (``_State.refresh``): a category's
    levels but the first as indicators, any other column one."""
    return {c: (max(len(pd.unique(data[c].dropna())) - 1, 0) if kinds[c] == "categorical" else 1)
            for c in data.columns if c != energy}


def model_width(data: pd.DataFrame, variables: Sequence[Variable], *,
                identity: Identity | None = None, units: np.ndarray | None = None,
                kinds: Mapping[str, str] | None = None, exclude: Sequence[str] = ()) -> ModelWidth:
    """The imputation model's width, and each incomplete column's model against the rule
    (:func:`~turbotab.core.methods.imputation.model_size`), as :func:`impute` fits them on
    ``data``: the one width function, which the sampler refuses by (:meth:`ModelWidth.refused`)
    and the check at the answer runs (:func:`imputation_at`).

    The width counts every predictor column of ``data`` as R's ``model.matrix`` does, a category as
    its levels less one (total energy under ``identity``, and ``exclude``, are never predictors). A
    column the chain draws is held to the rule with its own design's columns
    (:meth:`_Chain.design_rows`, :meth:`_Chain.design_units`, which encode each column as
    :func:`chain_kinds` reads it, a declared numeric code as its indicators): the other columns',
    an energy source's total-energy terms in place of "other", and with ``units`` the other
    columns' unit means and its own; on the rows that record it (the units, for a unit-level
    variable). A multinomial model fits those columns once for each level beyond the first
    (:attr:`~turbotab.core.methods.imputation.ModelSize.parameters`). Under ``"smcfcs"`` a
    covariate model is fit on every row with its current values, but what it learns from are the
    recorded ones, so they are counted under either mode."""
    from turbotab.core.methods.imputation import model_size

    variables = [v for v in variables if v.name in data.columns]
    resolved = chain_kinds(data, variables, kinds)
    energy = identity.energy if identity is not None else None
    skip = {*exclude, *([energy] if energy is not None else [])}
    blocks = _blocks(data, resolved, energy)
    predictors = {c: w for c, w in blocks.items() if c not in skip}
    e_obs = (pd.to_numeric(data[energy], errors="coerce").notna().to_numpy()
             if energy is not None else np.zeros(len(data), dtype=bool))
    clustered = units is not None
    once = {v.name for v in variables if v.unit_level} if clustered else set()
    models: list[ModelSize] = []
    for v in variables:
        drawn = (bool((~e_obs).any()) if identity is not None and v.name == identity.other
                 else bool(data[v.name].isna().any()))
        if not drawn:
            continue
        gone = {v.name, *skip}
        energy_terms = 0
        if identity is not None and v.name in identity.factors and e_obs.any():
            gone.add(identity.other)  # read the total in place of "other" (:meth:`energy_terms`)
            energy_terms = 1 + (2 if (~e_obs).any() else 0)
        others = {c: w for c, w in blocks.items() if c not in gone and w}
        row_level = sum(w for c, w in others.items() if c not in once)
        recorded = data[v.name].notna().to_numpy()
        if v.name in once:
            columns = sum(w for c, w in others.items() if c in once) + row_level + energy_terms
            codes = np.asarray(units)[recorded]
            values = pd.Series(data[v.name].to_numpy()[recorded]).groupby(codes).last()
        else:
            columns = sum(others.values()) + energy_terms
            if clustered:  # the unit means, and the column's own over the row's unit
                columns += row_level + energy_terms + (1 if blocks.get(v.name) == 1 else 0)
            values = data[v.name][recorded]
        models.append(model_size(v.name, v.kind, columns, values, units=v.name in once))
    return ModelWidth(sum(predictors.values()), predictors, models,
                      sum(predictors.get(c, 0) for c in DESIGN_TERMS))


SMALL = 0.05  # below this, one decimal would read a figure as 0.0


def _below(value: float, limit: float, first: int = 1) -> str:
    """``value`` in words, with the fewest digits from ``first`` that keep it below ``limit``; a
    value above 0 and below :data:`SMALL`, which one decimal would read as 0.0, to two significant
    digits (1/90 as 0.011)."""
    if 0 < value < SMALL:
        said = f"{value:,.{max(first, 1 - math.floor(math.log10(value)))}f}"
        if float(said.replace(",", "")) < limit:
            return said
    for digits in range(first, 4):
        said = f"{value:,.{digits}f}"
        if float(said.replace(",", "")) < limit:
            return said
    return f"{math.floor(value * 1000) / 1000:,.3f}"


def _model_words(m: ModelSize) -> str:
    """How far one imputation model falls short of the rule, as a refusal says it: least squares
    in rows, a model of two levels in events (its rarer value's rows), a multinomial model in its
    effective sample size over its (k − 1)·p parameters (:class:`~turbotab.core.methods.
    imputation.ModelSize`)."""
    from turbotab.core.models.sample_size import SOURCE_SPV

    per = _below(m.per, m.needed)
    on = (f"fit on the {_counted(m.rows, 'units' if m.units else 'rows')} that "
          f"record{'s' if m.rows == 1 else ''} it")
    if m.levels > 2:
        size = _below(m.counted, m.needed * m.parameters, 0)
        return (f"the chained equations would impute `{m.column}` from a multinomial model of "
                f"{m.parameters:,} parameters ({m.predictors:,} predictor parameters, categories "
                f"as indicators, for each of its {m.levels - 1:,} levels beyond the first) {on}, "
                f"an effective sample size of {size} (n − Σn³/n² over its {m.levels} levels), "
                f"{per} per parameter, where a common rule of thumb asks for {m.needed}")
    lead = (f"the chained equations would impute `{m.column}` from a model of {m.predictors:,} "
            f"predictor parameters (categories as indicators) {on}")
    if m.levels == 2:
        return (f"{lead}, {int(m.counted):,} of them its rarer value, {per} events per "
                f"parameter, where a common rule for logistic coefficients asks for {m.needed} "
                f"(Peduzzi et al. 1996)")
    return (f"{lead}, {per} per parameter, where {SOURCE_SPV} found about {m.needed} per "
            f"parameter enough for least-squares coefficients, standard errors and intervals")


def _short_words(m: ModelSize) -> str:
    """One more model short of its own rule, as a refusal names it after the first
    (:func:`_model_words` says the first in full): its own count, parameters and rule."""
    per = _below(m.per, m.needed)
    rows = "units" if m.units else "rows"
    if m.levels > 2:
        size = _below(m.counted, m.needed * m.parameters, 0)
        return (f"an effective sample size of {size} on {m.parameters:,} parameters, {per} per "
                f"parameter where its rule asks for {m.needed}")
    if m.levels == 2:
        return (f"{_counted(int(m.counted), rows)} of its rarer value on {m.parameters:,} "
                f"parameters, {per} events per parameter where its rule asks for {m.needed}")
    return (f"{_counted(m.rows, rows)} on {m.parameters:,} parameters, {per} per parameter where "
            f"its rule asks for about {m.needed}")


def _counted(k: int, things: str) -> str:
    """``k`` rows (or units) in words, a count of one in the singular."""
    return f"{k:,} {things if k != 1 else things[:-1]}"


def _indicators(k: int, of: str) -> str:
    return f"one indicator of {of}" if k == 1 else f"{k:,} indicators of {of}"


def redundant_columns(data: pd.DataFrame, kinds: Mapping[str, str], candidates: Sequence[str],
                      skip: Sequence[str] = ()) -> tuple[list[str], int]:
    """Of ``candidates``, the columns of ``data`` the others already span on the rows that record
    every predictor (as a recode spans nothing its code does not): each one whose model columns R's
    ``lm`` would report as aliased (``NA``) were it entered after the rest
    (:func:`identified_columns`), tried from the last column back, so of several that span each
    other the last is left out and the rest kept. With the number of those rows; none where the
    rows are too few to tell a column's span apart from the rows' own shortage."""
    gone = set(skip)
    blocks = _blocks(data, kinds, None)
    cols = [c for c in data.columns if c not in gone and blocks[c]]
    complete = data[cols].notna().all(axis=1).to_numpy()
    n = int(complete.sum())
    parts: dict[str, np.ndarray] = {}
    for c in cols:
        if kinds[c] == "categorical":
            levels = sorted(pd.unique(data[c].dropna()), key=str)
            v = data[c].to_numpy(dtype=object)[complete]
            parts[c] = np.column_stack([(v == lv).astype(float) for lv in levels[1:]])
        else:
            x = pd.to_numeric(data[c], errors="coerce").to_numpy(dtype=float)[complete]
            parts[c] = _standardize(x.reshape(-1, 1))
    ones = np.ones((n, 1))
    full = np.column_stack([ones, *parts.values()]) if parts else ones
    if n <= full.shape[1] or len(identified_columns(full)) == full.shape[1]:
        return [], n
    kept = list(cols)
    out: list[str] = []
    for c in reversed(cols):
        if c not in candidates:
            continue
        rest = [x for x in kept if x != c]
        M = np.column_stack([ones, *[parts[x] for x in rest], parts[c]])
        first = M.shape[1] - parts[c].shape[1]
        if not any(j >= first for j in identified_columns(M)):
            out.append(c)
            kept = rest
    return out[::-1], n


def width_refusal(sized: ModelWidth, data: pd.DataFrame, variables: Sequence[Variable], *,
                  identity: Identity | None = None, units: np.ndarray | None = None,
                  kinds: Mapping[str, str] | None = None, exclude: Sequence[str] = (),
                  leavable: Sequence[str] = ()) -> ImputationRefused:
    """The refusal of an imputation model the sampler refuses (:meth:`ModelWidth.refused`: wider
    than :data:`~turbotab.core.methods.imputation.MAX_MODEL_COLUMNS` columns with a model short of
    its rows), with its ways forward on it, which the fit's table offers as exits
    (``stages.modeling._missing_for_table``) and the check at the answer as its own
    (:func:`imputation_at`):

    * the width and both conditions, interpolated from it and the cap; then the models that fall
      short, the furthest first, the first in full and each after it in its own count and rule;
    * where the survey design's terms are most of the width, how many they are; and where the
      model without them is drawn, the sample answer (``exits``: the words
      :data:`~turbotab.core.methods.missing.WITHOUT_THE_DESIGN` name it);
    * a column the others already span (:func:`redundant_columns`), of ``leavable`` only: the
      columns an exit may leave out (the answer's settled covariates, never what the analysis
      studies, which the answer and the fit's table both pass; none given, none named), never the
      outcome's, the survey design's or the energy identity's, where the model without it is drawn
      (``columns``, which the exits offer to leave out), in words that say which condition then
      holds;
    * complete cases stay a way forward (``complete_cases`` is left unset)."""
    short = sized.short()
    first = _model_words(short[0])
    words = [f"The imputation model would hold {sized.width:,} columns (categories as indicators): "
             f"up to {MAX_MODEL_COLUMNS} it is drawn whatever its rows, and wider only where each "
             f"incomplete column's model has the rows per parameter its rule asks for.",
             first[0].upper() + first[1:] + "."]
    rest = short[1:]
    if len(rest) == 1:
        words.append(f"The model of `{rest[0].column}` falls short of its own rule too: "
                     f"{_short_words(rest[0])}.")
    elif rest:
        named = [f"`{m.column}` ({_short_words(m)})" for m in rest[:SHOWN_MODELS]]
        if len(rest) > len(named):
            named.append(f"{len(rest) - len(named):,} more")
        words.append(f"The models of {_listed(named)} fall short of their own rules too.")
    exits: list[dict[str, Any]] = []
    columns: list[str] = []
    variables = list(variables)
    if sized.design and 2 * sized.design > sized.width:
        parts = []
        if sized.columns.get("__stratum"):
            parts.append(_indicators(sized.columns["__stratum"], "its strata"))
        if sized.columns.get("__psu"):
            parts.append(_indicators(sized.columns["__psu"], "its PSUs within their strata"))
        if "__weight" in sized.columns:
            parts.append("its weight")
        words.append(f"Of the imputation model's {sized.width:,} columns, {sized.design:,} are "
                     f"the survey design's: {_listed(parts)}.")
        without = model_width(data.drop(columns=[c for c in DESIGN_TERMS if c in data.columns]),
                              variables, identity=identity, units=units, kinds=kinds,
                              exclude=exclude)
        if not without.refused():
            from turbotab.core.methods.missing import WITHOUT_THE_DESIGN, sample_way

            words.append(WITHOUT_THE_DESIGN)
            exits.append(sample_way())
    energy = identity.energy if identity is not None else None
    structural = {*exclude, *([energy, identity.other, *identity.factors] if identity else [])}
    # never the outcome's terms, the design's or "other", whose names ``missing`` starts with "__"
    candidates = [c for c in leavable
                  if c in data.columns and c not in structural and not str(c).startswith("__")]
    if candidates:
        resolved = chain_kinds(data, variables, kinds)
        skip = [c for c in data.columns if c == energy or c in exclude]
        gone, n = redundant_columns(data, resolved, candidates, skip)
        if gone:
            narrower = model_width(data.drop(columns=gone),
                                   [v for v in variables if v.name not in gone],
                                   identity=identity, units=units, kinds=kinds, exclude=exclude)
            if not narrower.refused():
                one = len(gone) == 1
                drawn = (f"the imputation model would hold {narrower.width:,} columns, no more "
                         f"than the {MAX_MODEL_COLUMNS} drawn whatever the rows"
                         if narrower.short() else "each incomplete column's model has the rows "
                                                  "per parameter its rule asks for")
                words.append(f"{_listed([f'`{c}`' for c in gone])} {'adds' if one else 'add'} "
                             f"nothing the other columns do not hold on the {n:,} rows that record "
                             f"them all (as a recode adds nothing beside the column it recodes), "
                             f"and without {'it' if one else 'them'} {drawn}.")
                columns = gone
    refused = ImputationRefused(" ".join(words))
    if exits:
        refused.exits = exits  # the sample answer, which the words name
    if columns:
        refused.columns = columns  # what leaving out lets the imputation be drawn without
    return refused


def _targets(data: pd.DataFrame, variables: Sequence[Variable],
             identity: Identity | None) -> tuple[list[Variable], np.ndarray]:
    """The variables the chain draws (each with a blank; "other" where total energy is blank), and
    the rows whose total energy is blank."""
    energy_blank = (pd.to_numeric(data[identity.energy], errors="coerce").isna().to_numpy()
                    if identity is not None else np.zeros(len(data), dtype=bool))

    def drawn(v: Variable) -> bool:
        if identity is not None and v.name == identity.other:
            return bool(energy_blank.any())
        return bool(data[v.name].isna().any())

    return [v for v in variables if drawn(v)], energy_blank


def refusal_before_drawing(data: pd.DataFrame, variables: Sequence[Variable], *,
                           identity: Identity | None = None, units: np.ndarray | None = None,
                           kinds: Mapping[str, str] | None = None, exclude: Sequence[str] = (),
                           leavable: Sequence[str] = (), sized: ModelWidth | None = None
                           ) -> ImputationRefused | None:
    """What :func:`impute` refuses before it draws a copy, the first in its order, or None where it
    draws: a variable analyzed on the log scale recorded at zero or below (``zeros``); a column
    the chain draws, other than a category, with fewer than two recorded values (``columns``);
    then a model the width rule refuses (:meth:`ModelWidth.refused`, :func:`width_refusal`, which
    ``leavable`` reaches). The sampler raises it, and the check at the answer says it with the
    fit's ways forward (:func:`_imputation_has_its_rows`), so both say the same. ``sized``: the
    model's width where it is measured already (the answer's, :func:`imputation_at`), measured
    here otherwise. The refusals that may name a column to leave out (fewer than two values, the
    width) carry ``width_refused``: whether the width rule refuses the model, so the check at the
    answer, which runs only where it does, refuses the plan in these words; the fit's table then
    offers to leave out what the answer offers, of ``leavable`` only
    (``stages.modeling._missing_for_table``)."""
    variables = [v for v in variables if v.name in data.columns]
    for v in variables:
        if v.log:
            x = pd.to_numeric(data[v.name], errors="coerce")
            bad = x.notna() & (x <= 0)
            if bad.any():
                refused = ImputationRefused(
                    f"`{v.name}` is analyzed on the log scale and is zero or negative on "
                    f"{_counted(int(bad.sum()), 'rows')}, so it cannot be imputed on that scale; a "
                    f"zero a log would turn into a blank goes to the detection-limit question, "
                    f"never to a fill.")
                refused.zeros = [v.name]  # the columns whose zeros the analysis cannot log
                return refused
    targets, _ = _targets(data, variables, identity)
    for v in targets:
        if data[v.name].notna().sum() < 2 and v.kind != "categorical":
            refused = ImputationRefused(f"`{v.name}` has fewer than two observed values, so it "
                                        f"cannot be imputed.")
            refused.columns = [v.name]  # what leaving out would let the imputation run without
            refused.width_refused = _measured(sized, data, variables, identity, units, kinds,
                                              exclude).refused()
            return refused
    # Wider than the cap, each incomplete column's model against the rows-per-parameter rule, on
    # the rows that record it (:func:`model_width`, :meth:`ModelWidth.refused`): the check at the
    # answer runs this before the plan's lock (:func:`imputation_at`), so a plan refused here was
    # refused there, where it could be.
    sized = _measured(sized, data, variables, identity, units, kinds, exclude)
    if sized.refused():
        refused = width_refusal(sized, data, variables, identity=identity, units=units,
                                kinds=kinds, exclude=exclude, leavable=leavable)
        refused.width_refused = True
        return refused
    return None


def _measured(sized: ModelWidth | None, data: pd.DataFrame, variables: Sequence[Variable],
              identity: Identity | None, units: np.ndarray | None,
              kinds: Mapping[str, str] | None, exclude: Sequence[str]) -> ModelWidth:
    """``sized`` where it is given, else the model's width measured (:func:`model_width`)."""
    if sized is not None:
        return sized
    return model_width(data, variables, identity=identity, units=units, kinds=kinds,
                       exclude=exclude)


def zero_exits(column: str, adjustment: Any) -> list[dict[str, Any]]:
    """The ways forward from ``column``'s zeros that the analysis logs (``zeros``), as the fit's
    table offers them (``stages.modeling._missing_for_table``): no fill or row choice resolves
    them, only how the zeros are read, or, where ``adjustment`` (the energy adjustment) logs that
    column, an energy model without the log (MODELING_SEQUENCE §2, zeros)."""
    exits: list[dict[str, Any]] = []
    if (adjustment is not None and adjustment.log_transform
            and column in (*adjustment.nutrients, adjustment.energy_column)):
        exits.append({"label": "Adjust for energy without the log",
                      "decision": {"kind": "set_energy_adjustment", **adjustment.model_dump(),
                                   "log_transform": False}})
    exits.append({"label": f"Say how the zeros of `{column}` are read: values below a detection "
                           f"limit, or true zeros", "decision": None})
    return exits


def impute(data: pd.DataFrame, variables: Sequence[Variable], *, mode: Mode = "fcs",
           substantive: Substantive | None = None, identity: Identity | None = None,
           units: np.ndarray | None = None, m: int = M_DEFAULT, iterations: int = ITERATIONS,
           seed: int = 0, rjlimit: int = RJLIMIT, exclude: Sequence[str] = (),
           kinds: Mapping[str, str] | None = None,
           progress: Callable[[int, int], None] | None = None,
           cancelled: Callable[[], bool] | None = None,
           leavable: Sequence[str] = ()) -> Imputations:
    """``m`` completed copies of ``data``.

    ``data`` holds every column of the imputation model on its raw scale: the ``variables`` (blank
    where missing) and complete predictors (design variables, the outcome's terms under ``"fcs"``).
    Under ``identity`` it also holds total energy (raw, blank where missing), which is never a
    predictor (it is the sum of the sources and other), and other (``Identity.other``, a variable).
    ``substantive`` is required under ``"smcfcs"``. ``units``: integer unit codes per row
    (clustered rows). ``exclude``: columns carried in ``data`` that are never predictors.
    ``kinds``: how each complete column enters the models (a numeric code column the user declared
    categorical is a category, whatever its values look like); by its values otherwise.
    ``leavable``: the columns a refusal of the model's width may name to leave out (the answer's
    settled covariates, :func:`imputation_at`, which the fit's table passes too through
    ``missing.impute_for_inference``); none given, it names none, so a column the analysis
    studies is never named. Before any copy is drawn, refused as :func:`refusal_before_drawing`
    says. Deterministic for a given ``seed``.
    """
    if mode == "smcfcs" and substantive is None:
        raise ValueError("SMC-FCS needs the substantive model.")
    variables = [v for v in variables if v.name in data.columns]
    refused = refusal_before_drawing(data, variables, identity=identity, units=units, kinds=kinds,
                                     exclude=exclude, leavable=leavable)
    if refused is not None:
        raise refused
    targets, energy_blank = _targets(data, variables, identity)
    kinds = chain_kinds(data, variables, kinds)
    miss_any = energy_blank.copy()
    for v in variables:
        if not (identity is not None and v.name == identity.other):
            miss_any |= data[v.name].isna().to_numpy()
    order = [v.name for v in sorted(targets, key=lambda v: (int(data[v.name].isna().sum()),
                                                             list(data.columns).index(v.name)))]
    skip = [*exclude, *([identity.energy] if identity is not None else [])]
    plain = mode == "fcs" and len(order) <= 1 and units is None and identity is None
    sweeps = 1 if plain else iterations  # one incomplete column alone: the chain is monotone
    rng = np.random.default_rng(seed)
    total = m * sweeps * max(1, len(order))
    done = [0]

    def tick() -> None:
        if cancelled is not None and cancelled():
            from turbotab.core.jobs import Cancelled

            raise Cancelled()
        done[0] += 1
        if progress is not None:
            progress(done[0], total)

    frames: list[pd.DataFrame] = []
    failures = 0
    infeasible = 0
    for _k in range(m):
        chain = _Chain(data, variables, mode, substantive, identity, units, rng, rjlimit, kinds)
        infeasible = chain.infeasible_rows()
        raw = chain.run(order, sweeps, skip, tick)
        completed = data.copy()
        for c in data.columns:
            col = raw[c]
            if kinds[c] == "categorical" or not pd.api.types.is_numeric_dtype(data[c]):
                filled = pd.Series(col.to_numpy(), index=data.index, dtype=object)
                if pd.api.types.is_numeric_dtype(data[c]):  # a declared category of number codes
                    filled = filled.astype(float)
            else:
                filled = pd.Series(col.to_numpy(dtype=float), index=data.index)
                if c in chain.vars or (identity is not None and c == identity.energy):
                    # a recorded value comes back exactly as recorded (exp(log x) may not)
                    original = pd.to_numeric(data[c], errors="coerce")
                    filled = filled.where(original.isna(), original)
            completed[c] = filled
        frames.append(completed)
        failures += chain.failures
    imputed = {v.name: int(data[v.name].isna().sum()) for v in targets
               if not (identity is not None and v.name == identity.other)}
    if identity is not None and energy_blank.any():
        imputed[identity.energy] = int(energy_blank.sum())
    censored = {v.name: v.censoring for v in variables if v.censoring is not None}
    return Imputations(frames=frames, m=m, iterations=sweeps, imputed=imputed,
                       n_incomplete_rows=int(miss_any.sum()), variables=list(data.columns),
                       kinds=kinds, censored=censored, seed=seed,
                       method="smcfcs" if mode == "smcfcs" else "chained_equations",
                       rejection_failures=failures, identity_infeasible=infeasible)


# ── the same rule at the answer, before the plan's lock ──────────────────────


@dataclass
class AtTheAnswer:
    """What the fit would impute under a state (:func:`imputation_at`): the sampler's inputs
    (``missing.ImputationPlan``), its width (:func:`model_width`), the columns an exit may leave
    out (the settled covariates, never what the analysis studies), and the design stage's spec the
    plan is built from (its energy adjustment says the ways forward from a logged zero)."""

    plan: Any
    sized: ModelWidth
    leavable: list[str]
    spec: Any = None


def imputation_at(state: Any, ctx: Any) -> AtTheAnswer | None:
    """The imputation model the fit would draw under ``state``, built over the table the context
    opens as the fit builds it: the cohort's rows under ``state`` (every analyzed row under
    inference, ruling 3), the design stage's spec, the outcome as the fit codes it, the clusters,
    the survey design under the population answer, the parts of totals and the energy identity's
    factors, then ``missing.imputation_plan`` and :func:`model_width` on its frame.

    None where the fit would draw no copies (not inference, not multiple imputation, no blank, a
    block the fit holds first, the data's own copies, no family that pools), or where what it would
    draw cannot be foreseen here (no table to open, a reading the fit asks first): the fit's own
    refusal says the same words then (:func:`impute`)."""
    missing = getattr(state, "missing", None)
    if (getattr(state, "purpose", None) != "inference" or missing is None
            or missing.strategy != "multiple_imputation" or not state.target or not state.roles):
        return None
    from turbotab.core.decisions import _store_of
    from turbotab.core.stages.modeling import imputed_copies_column

    if imputed_copies_column(state) is not None:
        return None  # the data's own copies are pooled, and none is drawn
    store = _store_of(ctx)
    if store is None:
        return None
    try:
        return _imputation_at(state, ctx, store)
    except Exception:  # noqa: BLE001 - what cannot be foreseen here: the fit says why, in its words
        return None


def _imputation_at(state: Any, ctx: Any, store: Any) -> AtTheAnswer | None:
    from turbotab.core.decisions import _ctx, left_out
    from turbotab.core.methods.batch import batch_inputs
    from turbotab.core.methods.energy import PARTITION_METHODS
    from turbotab.core.methods.missing import (imputation_plan, missing_block, nonlinear_terms,
                                               time_invariant_columns)
    from turbotab.core.models import get_family
    from turbotab.core.models.inference import cluster_columns, resolve_clusters
    from turbotab.core.models.pipeline import (design_spec, input_columns, model_predictors,
                                               modeling_frame)
    from turbotab.core.readings import (energy_source_factors, factor_verdicts, kcal_per_unit,
                                        nesting, predictors_or_ask, settled_roles)
    from turbotab.core.stages.modeling import _leavable, coded_outcome
    from turbotab.core.stages.rows import compute_cohort

    infos = list(store.info().columns)
    # The fit's predictor set once every reading it rests on is settled; while one is not, the fit
    # asks first (``readings.predictors_or_ask`` raises), and nothing is foreseen here.
    predictors_or_ask(state, {c.name: {"dtype": c.dtype, "n_unique": c.n_unique} for c in infos},
                      drop=left_out(state), store=store)
    predictors = model_predictors(state)
    task = getattr(state, "task", None) or _ctx(ctx, "task")
    if not predictors or not task:
        return None
    bundle = _ctx(ctx, "bundle")
    working = bundle("working") if callable(bundle) else None
    split = bundle("split") if callable(bundle) else None
    ingest = dict(getattr(working, "data", None) or {"columns": [c.to_dict() for c in infos]})
    _, kept, _ = compute_cohort(store, state, ingest)  # the analyzed rows: the cohort's, ruling 3
    rows = np.asarray(kept, dtype=np.int64)
    if not len(rows):
        return None
    target = state.target
    adj = state.energy_adjustment
    grouped_by = (getattr(split, "data", None) or {}).get("grouped_by") if split is not None else None
    unit_columns = cluster_columns(state, store.columns, [grouped_by])
    follow = []
    if task == "time_to_event":
        from turbotab.core.models.survival import follow_up_columns

        follow = follow_up_columns(state)
    inputs = list(dict.fromkeys([*input_columns(predictors, adj), *batch_inputs(state)]))
    frame = modeling_frame(store, list(dict.fromkeys([*inputs, target, *unit_columns, *follow])),
                           rows, outcome=target)
    factors = None
    if adj is not None and adj.method in PARTITION_METHODS:
        factors = energy_source_factors(state, list(adj.nutrients), store=store)
    spec = design_spec(state, frame[[c for c in inputs if c in frame.columns]], predictors,
                       column_info={c.name: c for c in infos}, energy_factors=factors)
    X = frame[list(spec.inputs)]
    if task == "ordinal":
        from turbotab.core.models.ordinal import ordinal_outcome

        coded, _ = ordinal_outcome(frame[target].to_numpy(), state.outcome_order, column=target)
    else:
        coded = coded_outcome(task, frame[target].to_numpy(), state.event)
    if task == "time_to_event":
        from turbotab.core.models.survival import time_to_event_outcome

        y = time_to_event_outcome(state, frame, coded)
    else:
        y = np.asarray(coded)
    answer = spec.missing or {}
    levels = [c for c in spec.levels if c in X.columns and X[c].isna().any()]
    gaps = [c for c in spec.inputs if c not in levels and X[c].isna().any()]
    if not gaps:
        return None
    clusters = resolve_clusters(state, frame[list(unit_columns)], [grouped_by])
    clustered = clusters is not None and getattr(clusters, "clustered", False)
    # What the fit's table holds before any copy is drawn (``stages.modeling._missing_for_table``):
    # a passive or single-level imputation it blocks, a chosen set of families none of which pools
    # the copies, one whose tables all draw their own copies one study factor at a time (the
    # feature-wise tests, ``methods.featurewise_mi``, which no joint copy serves), and an
    # unconfirmed one-value-per-unit column it asks about.
    if missing_block(answer, "inference", levels, nonlinear=nonlinear_terms(spec, task),
                     passive=task in ("ordinal", "multiclass"), clustered=clustered) is not None:
        return None
    pools = [k for k in (state.models or []) if get_family(k).inference is not None]
    if state.models and (not pools or not any(getattr(get_family(k), "pools_imputations", True)
                                              for k in pools)):
        return None
    if pools and all(getattr(get_family(k), "draws_per_feature", False) for k in pools):
        return None  # the feature-wise tests draw their own copies, one study factor at a time
    once: list[str] = []
    if clustered and answer.get("imputation_levels") != "single_level":
        once, ask = time_invariant_columns(state, X, clusters.codes, gaps)
        if ask:
            return None
    survey = None
    from turbotab.core.survey import reading_of

    if state.survey is not None or reading_of(state).present:
        from turbotab.core.methods.survey import for_fit

        found = for_fit(state, store, clusters.column if clustered else None)
        if found is not None and found.refusal:
            return None  # the survey question is the fit's refusal first
        if found is not None and found.answer == "population":
            survey = found.design
    numeric = [c for c in spec.predictors if c in set(spec.numeric)]
    nested = nesting(state, nesting(state, frame=X, columns=numeric), columns=spec.inputs)
    settled: dict[str, float] = {}
    nutrients = list(adj.nutrients) if adj is not None else []
    if nutrients:  # the factors the identity may compute with (``modeling._settled_factors``)
        verdicts = factor_verdicts(state, store, nutrients)
        for c in nutrients:
            reading = kcal_per_unit(state, c, verdict=verdicts.get(c))
            if reading.settled and reading.factor is not None:
                settled[c] = float(reading.factor)
    plan = imputation_plan(spec, X, y, task, survey=survey,
                           clusters=clusters if clustered else None, nested=nested,
                           factors=settled, time_invariant=once)
    sized = model_width(plan.data, plan.variables, identity=plan.identity, units=plan.units,
                        kinds=plan.kinds)
    covariates = [c for c, r in settled_roles(state).items() if r == "covariate"]
    return AtTheAnswer(plan, sized, _leavable(state, covariates), spec)


@lru_cache(maxsize=1)
def _frame_reads() -> tuple[str, ...]:
    """Every slot the fit's imputation frame is built from: the fit's and its upstream stages'."""
    from turbotab.core.graph import load_graph
    from turbotab.core.stages import GRAPH_FACTORY

    graph = load_graph(GRAPH_FACTORY)
    names = graph.upstream("fit") | {"fit"}
    return tuple(sorted({slot for name in names for slot in graph[name].reads}))


def _imputation_has_its_rows(decision: Any, ctx: Any) -> None:
    """The rule at the answer (:func:`model_width` on the frame the fit would impute,
    :func:`imputation_at`): an answer that leaves multiple imputation under inference with a model
    the sampler refuses (:meth:`ModelWidth.refused`: wider than
    :data:`~turbotab.core.methods.imputation.MAX_MODEL_COLUMNS` columns, with a model short of the
    rows that record its column) is refused when it is given, in the words the sampler refuses it
    with, the first in its order (:func:`refusal_before_drawing`: a logged column's zeros, a column
    with fewer than two recorded values, then the width), and with the ways forward the fit's table
    offers for them (``stages.modeling._missing_for_table``): for the zeros, how they are read,
    and the energy model without the log where the record accepts it (:func:`zero_exits`; the
    fit's table offers that one without asking the record); otherwise complete cases, the width
    refusal's own (the sample answer), and leaving out a column the refusal names, of the settled
    covariates only. Each missing-values answer among them is offered where the record accepts it,
    as the fit's table counts it (``row_floor.stage_exits``).
    Registered for every kind that writes a slot, so the answer refused is the one that makes the
    model too wide (multiple imputation itself, a role, the survey answer, an exclusion), and for
    the plan's lock, which the server records when Fit is pressed: a plan the rows cannot carry is
    refused before it is locked, never after."""
    from turbotab.core.decisions import (Refusal, SetMissing, _records_in,
                                         _roles_record_what_rode_along, _state, fold_onto,
                                         state_after)
    from turbotab.core.methods.missing import _COMPLETE_CASES_EXIT
    from turbotab.core.row_floor import _PROBING, accepted_exit

    now = _state(ctx)
    if now is None:
        return
    if decision.kind == "lock_plan":
        after = now  # the plan as it is locked
    else:
        if decision.kind == "set_roles":  # what rides along unconfirmed, as it will be recorded
            decision = _roles_record_what_rode_along(decision, ctx)
        after = fold_onto(now, decision) if _records_in(ctx) is None else state_after(decision, ctx)
        if after is None or all(getattr(now, s, None) == getattr(after, s, None)
                                for s in _frame_reads()):
            return
    found = imputation_at(after, ctx)
    if found is None or not found.sized.refused():
        return
    plan = found.plan
    refused = refusal_before_drawing(plan.data, plan.variables, identity=plan.identity,
                                     units=plan.units, kinds=plan.kinds, leavable=found.leavable,
                                     sized=found.sized)
    if refused is None:  # the width rule refuses it (above), so the sampler does
        return
    if _PROBING.get():  # an exit tried for another refusal: only whether it is refused is read
        raise Refusal(WIDTH_REFUSAL, f"{REFUSED_LEAD}{refused}")
    base = after.missing.model_dump(mode="json")
    zeros = list(getattr(refused, "zeros", None) or [])
    if zeros:
        # Each where taking it is accepted: the energy model without the log is as wide (the log
        # enters no column's count), so where the width rule refuses this one, it is refused too.
        exits = [e for e in zero_exits(zeros[0], found.spec.energy_adjustment() if found.spec
                                       else None)
                 if e["decision"] is None or accepted_exit(e["decision"], ctx) is not None]
    else:
        exits = [{"label": _COMPLETE_CASES_EXIT,
                  "decision": SetMissing(**{**base, "strategy": "complete_case",
                                            "acknowledged": False})}]
        exits.extend(dict(e) for e in getattr(refused, "exits", None) or [])
        left = [c for c in getattr(refused, "columns", None) or []
                if c in found.leavable and c not in base["drop_columns"]]
        if left:
            exits.append({"label": f"Leave {', '.join(f'`{c}`' for c in left)} out",
                          "decision": SetMissing(**{**base, "drop_columns": [
                              *base["drop_columns"], *left]})})
    raise Refusal(WIDTH_REFUSAL, f"{REFUSED_LEAD}{refused}",
                  exits=[e for e in exits if not isinstance(e["decision"], SetMissing)
                         or accepted_exit(e["decision"], ctx) is not None])


def _register() -> None:
    """Importing this module registers the check, for every kind the decisions know by then (as
    ``row_floor`` registers its own) and for revert."""
    from turbotab.core.decisions import SLOTS, register_validator

    for kind in [*SLOTS, "revert"]:
        register_validator(kind, _imputation_has_its_rows)


_register()

__all__ = ["DESIGN_TERMS", "FIRST_TRIES", "Identity", "Mode", "ModelWidth", "Outcome",
           "REFUSED_LEAD", "RJLIMIT", "START_SHARE", "Substantive", "Term", "Variable",
           "WIDTH_REFUSAL", "breslow", "chain_kinds", "efron", "imputation_at", "impute",
           "model_width", "outcome_logdensity", "redundant_columns", "refusal_before_drawing",
           "width_refusal", "zero_exits"]
