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
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from turbotab.core.methods.imputation import (
    ITERATIONS,
    M_DEFAULT,
    Censoring,
    ImputationRefused,
    Imputations,
    MAX_MODEL_COLUMNS,
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
        kinds: dict[str, str] = {}
        levels: dict[str, list[Any]] = {}
        values: dict[str, np.ndarray] = {}
        energy = identity.energy if identity is not None else None
        for c in data.columns:
            if c == energy:
                continue
            var = self.vars.get(c)
            kind = var.kind if var is not None else ((kinds or {}).get(c) or _kind_of(data[c]))
            kinds[c] = "numeric" if kind == "censored" else kind
            if kind == "categorical":
                levels[c] = sorted(pd.unique(data[c].dropna()), key=str)
                values[c] = data[c].to_numpy(dtype=object).copy()
            else:
                x = pd.to_numeric(data[c], errors="coerce").to_numpy(dtype=float).copy()
                values[c] = self.scaled(c, x)
        self.st = _State(values=values, levels=levels, kinds=kinds)
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
        from turbotab.core.methods.imputation import _binary, _categorical, _censored

        X = self.design_rows(name, skip)
        miss = self.miss[name]
        rows = np.flatnonzero(miss)
        cur = self.st.values[name]
        if var.kind == "categorical":
            new = _categorical(X, cur, self.st.levels[name], miss, self.rng)
        elif var.kind == "binary":
            new = _from01(_binary(X, _as01(cur, self.data[name]), miss, self.rng), self.data[name])
        elif var.kind == "censored":
            new = self.scaled(name, _censored(X, self.raw(name, cur.astype(float)), miss,
                                              var.censoring, self.rng))
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


def impute(data: pd.DataFrame, variables: Sequence[Variable], *, mode: Mode = "fcs",
           substantive: Substantive | None = None, identity: Identity | None = None,
           units: np.ndarray | None = None, m: int = M_DEFAULT, iterations: int = ITERATIONS,
           seed: int = 0, rjlimit: int = RJLIMIT, exclude: Sequence[str] = (),
           kinds: Mapping[str, str] | None = None,
           progress: Callable[[int, int], None] | None = None,
           cancelled: Callable[[], bool] | None = None) -> Imputations:
    """``m`` completed copies of ``data``.

    ``data`` holds every column of the imputation model on its raw scale: the ``variables`` (blank
    where missing) and complete predictors (design variables, the outcome's terms under ``"fcs"``).
    Under ``identity`` it also holds total energy (raw, blank where missing), which is never a
    predictor (it is the sum of the sources and other), and other (``Identity.other``, a variable).
    ``substantive`` is required under ``"smcfcs"``. ``units``: integer unit codes per row
    (clustered rows). ``exclude``: columns carried in ``data`` that are never predictors.
    ``kinds``: how each complete column enters the models (a numeric code column the user declared
    categorical is a category, whatever its values look like); by its values otherwise.
    Deterministic for a given ``seed``.
    """
    if mode == "smcfcs" and substantive is None:
        raise ValueError("SMC-FCS needs the substantive model.")
    variables = [v for v in variables if v.name in data.columns]
    for v in variables:
        if v.log:
            x = pd.to_numeric(data[v.name], errors="coerce")
            bad = x.notna() & (x <= 0)
            if bad.any():
                refused = ImputationRefused(
                    f"`{v.name}` is analyzed on the log scale and is zero or negative on "
                    f"{int(bad.sum()):,} rows, so it cannot be imputed on that scale; a zero a log "
                    f"would turn into a blank goes to the detection-limit question, never to a fill.")
                refused.zeros = [v.name]  # the columns whose zeros the analysis cannot log
                raise refused
    energy_blank = (pd.to_numeric(data[identity.energy], errors="coerce").isna().to_numpy()
                    if identity is not None else np.zeros(len(data), dtype=bool))

    def drawn(v: Variable) -> bool:
        if identity is not None and v.name == identity.other:
            return bool(energy_blank.any())
        return bool(data[v.name].isna().any())

    targets = [v for v in variables if drawn(v)]
    width = 0
    declared = dict(kinds or {})
    kinds = {c: (next((v.kind for v in variables if v.name == c), None) or declared.get(c)
                 or _kind_of(data[c])) for c in data.columns}
    for c in data.columns:
        if identity is not None and c == identity.energy:
            continue
        width += max(1, data[c].nunique(dropna=True) - 1) if kinds[c] == "categorical" else 1
    if width > MAX_MODEL_COLUMNS:
        raise ImputationRefused(
            f"The imputation model would hold {width:,} columns (categories as indicators), more "
            f"than the {MAX_MODEL_COLUMNS} chained equations are run with here.")
    for v in targets:
        if data[v.name].notna().sum() < 2 and v.kind != "categorical":
            refused = ImputationRefused(f"`{v.name}` has fewer than two observed values, so it "
                                        f"cannot be imputed.")
            refused.columns = [v.name]  # what leaving out would let the imputation run without
            raise refused
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


__all__ = ["FIRST_TRIES", "Identity", "Mode", "Outcome", "RJLIMIT", "START_SHARE", "Substantive",
           "Term", "Variable", "breslow", "efron", "impute", "outcome_logdensity"]
