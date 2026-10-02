"""How many rows an unpenalized regression needs, by purpose (AUDIT_REPORT §5 WP8: A21, E12).

**Prediction: Riley's criteria.** A prediction model's sample size is set by how much it will
overfit, which depends on the number of candidate predictor parameters, the outcome proportion and
how much of the outcome the model can explain, not on a fixed number of events per variable (van
Smeden et al. 2019, *Stat Methods Med Res* 28:2455, found events per variable "not an appropriate
criterion" for prediction). Riley et al. (*BMJ* 2020;368:m441; *Stat Med* 2019;38:1262 and
38:1276) give the minimum as the largest of several criteria, implemented here as their own
``pmsampsize`` package implements them:

* **Binary** (box 1, B1, B3, B4): the overall proportion within ±0.05,
  ``n = (1.96 / 0.05)² φ(1 − φ)``; an expected uniform shrinkage of at least 0.9,
  ``n = p / ((S − 1) ln(1 − R²_CS / S))``; and an optimism of at most 0.05 in the apparent
  Nagelkerke R², the same formula at ``S = R²_CS / (R²_CS + 0.05 max(R²_CS))``, where
  ``max(R²_CS) = 1 − (φ^φ (1 − φ)^(1 − φ))²``. B2 (mean absolute prediction error) needs van
  Smeden's simulation-based formula and, as in ``pmsampsize``, is not computed.
* **Continuous** (C1–C4): the intercept within a multiplicative margin of 1.1 (it needs the
  outcome's mean, positive, and its SD); the residual SD within a 10% margin, ``n = 234 + p``; an
  expected shrinkage of at least 0.9, the smallest ``n ≥ p + 2`` with
  ``1 + (p − 2) / (n ln(1 − R²_app)) ≥ 0.9``, ``R²_app = (R² (n − p − 1) + p) / (n − 1)``; and an
  optimism of at most 0.05 in R², ``n = 1 + p (1 − R²) / 0.05``.

The model's R² is anticipated before any fit, as Riley et al. do when nothing better is known:
"If we assume, conservatively, that the new model will explain 15% of the variability, the
anticipated R2 cs value is 0.15×0.33=0.05" (*BMJ* 2020, example 1). So the anticipated Cox–Snell
R² is 15% of its maximum: ``0.15 max(R²_CS)`` for a binary outcome, 0.15 for a continuous one
(whose maximum is 1). A multinomial outcome's criteria (Pate et al. 2023) are not computed.

**Inference: Austin & Steyerberg.** For the coefficients of a linear regression, "Linear
regression models require only two SPV [subjects per variable] for adequate estimation of
regression coefficients, standard errors, and confidence intervals" (Austin & Steyerberg 2015,
*J Clin Epidemiol* 68:627, abstract). Logistic coefficients keep the events-per-variable rule
(Peduzzi et al. 1996, *J Clin Epidemiol* 49:1373), which the audit found closer to justified there.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

SHRINKAGE = 0.9  # Riley et al.: target an expected uniform shrinkage of at least 0.9
OPTIMISM = 0.05  # …and an optimism of at most 0.05 in the apparent (Nagelkerke) R²
MARGIN = 0.05  # the overall outcome proportion within ±0.05
Z = 1.96  # as pmsampsize writes it
R2_SHARE = 0.15  # the anticipated R²_CS: 15% of its maximum when nothing better is known
RESIDUAL_ROWS = 234  # C2: n = 234 + p gives a residual SD within a 10% multiplicative margin
INTERCEPT_MMOE = 1.1  # C1: the intercept's interval within a multiplicative margin of 1.1
SPV_INFERENCE = 2  # Austin & Steyerberg 2015: subjects per variable for OLS coefficients
EPV_INFERENCE = 10  # Peduzzi et al. 1996: events per variable for logistic coefficients
SOURCE_RILEY = "Riley et al. 2020"
SOURCE_SPV = "Austin & Steyerberg 2015"


@dataclass(frozen=True)
class Criterion:
    key: str  # Riley et al.'s label: B1, B3, B4, C1–C4
    what: str
    n: int


@dataclass(frozen=True)
class Minimum:
    """The smallest sample that meets every criterion, and what each one asks for."""

    n: int
    criteria: tuple[Criterion, ...]
    parameters: int
    r2_cs: float  # the anticipated Cox–Snell R² the criteria assume
    max_r2_cs: float  # its maximum (1 for a continuous outcome)
    prevalence: float | None = None

    @property
    def binding(self) -> Criterion:
        return max(self.criteria, key=lambda c: c.n)


def max_r2_cs(prevalence: float) -> float:
    """The largest Cox–Snell R² a binary outcome with this proportion allows (Riley et al. 2020,
    supplementary S5): ``1 − exp(2 lnL_null / n)`` with ``lnL_null / n = φ ln φ + (1 − φ) ln(1 − φ)``."""
    phi = float(prevalence)
    per_row = phi * math.log(phi) + (1.0 - phi) * math.log(1.0 - phi)
    return 1.0 - math.exp(2.0 * per_row)


def _shrinkage_n(parameters: int, r2: float, shrinkage: float) -> int:
    """``p / ((S − 1) ln(1 − R²/S))``, rounded up (Riley et al. 2020, figure 3)."""
    return math.ceil(parameters / ((shrinkage - 1.0) * math.log(1.0 - r2 / shrinkage)))


def binary_minimum(parameters: int, prevalence: float, r2_cs: float | None = None) -> Minimum:
    """Riley et al.'s minimum for a logistic prediction model (B1, B3, B4)."""
    p = int(parameters)
    phi = float(prevalence)
    if not (0.0 < phi < 1.0) or p < 1:
        raise ValueError("The criteria need an outcome proportion strictly between 0 and 1 and at "
                         "least one predictor parameter.")
    top = max_r2_cs(phi)
    r2 = R2_SHARE * top if r2_cs is None else float(r2_cs)
    s_optimism = r2 / (r2 + OPTIMISM * top)
    criteria = (
        Criterion("B1", "the overall outcome proportion within ±0.05",
                  math.ceil((Z / MARGIN) ** 2 * phi * (1.0 - phi))),
        Criterion("B3", "an expected shrinkage of at least 0.9", _shrinkage_n(p, r2, SHRINKAGE)),
        Criterion("B4", "an optimism of at most 0.05 in the apparent R²",
                  _shrinkage_n(p, r2, s_optimism)),
    )
    return Minimum(n=max(c.n for c in criteria), criteria=criteria, parameters=p, r2_cs=r2,
                   max_r2_cs=top, prevalence=phi)


def expected_shrinkage(n: np.ndarray | float, parameters: int, r2: float) -> np.ndarray:
    """The expected uniform shrinkage of a linear model on ``n`` rows (pmsampsize, criterion 1 for
    a continuous outcome): ``1 + (p − 2) / (n ln(1 − R²_app))`` with the apparent R² that an
    adjusted R² of ``r2`` implies, ``R²_app = (R² (n − p − 1) + p) / (n − 1)``."""
    n = np.asarray(n, dtype=float)
    p = float(parameters)
    with np.errstate(divide="ignore", invalid="ignore"):
        apparent = (r2 * (n - p - 1.0) + p) / (n - 1.0)
        return 1.0 + (p - 2.0) / (n * np.log(1.0 - apparent))


def _continuous_shrinkage_n(parameters: int, r2: float) -> int:
    """The smallest ``n ≥ p + 2`` whose expected shrinkage reaches 0.9, searched upward as
    pmsampsize searches (the curve dips before it rises, so the first crossing is the answer)."""
    start = parameters + 2
    first = expected_shrinkage(start, parameters, r2)
    if np.isfinite(first) and first > SHRINKAGE:
        return start
    lo, width = start + 1, 1024
    while True:
        n = np.arange(lo, lo + width)
        es = expected_shrinkage(n, parameters, r2)
        hit = np.flatnonzero(np.isfinite(es) & (es >= SHRINKAGE))
        if hit.size:
            return int(n[hit[0]])
        lo, width = lo + width, width * 2


def continuous_minimum(parameters: int, r2: float | None = None, *, mean: float | None = None,
                       sd: float | None = None) -> Minimum:
    """Riley et al.'s minimum for a linear prediction model (C1–C4; C1 only with a positive
    outcome mean and its SD, since its margin is relative to the mean)."""
    from scipy import stats

    p = int(parameters)
    if p < 1:
        raise ValueError("The criteria need at least one predictor parameter.")
    r2 = R2_SHARE if r2 is None else float(r2)
    criteria = [
        Criterion("C2", "the residual SD within a 10% margin", RESIDUAL_ROWS + p),
        Criterion("C3", "an expected shrinkage of at least 0.9", _continuous_shrinkage_n(p, r2)),
        Criterion("C4", "an optimism of at most 0.05 in the apparent R²",
                  math.ceil(1.0 + p * (1.0 - r2) / OPTIMISM)),
    ]
    if mean is not None and sd is not None and mean > 0 and sd > 0:
        n = max(c.n for c in criteria)
        while True:
            half = stats.t.ppf(0.975, n - p - 1) * math.sqrt(sd * sd * (1.0 - r2) / n)
            if (mean + half) / mean <= INTERCEPT_MMOE:
                break
            n += 1
        criteria.insert(0, Criterion("C1", "the mean outcome within a 10% margin", n))
    return Minimum(n=max(c.n for c in criteria), criteria=tuple(criteria), parameters=p, r2_cs=r2,
                   max_r2_cs=1.0)


def prediction_minimum(task: str, parameters: int, *, n_rows: int, n_events: int | None = None,
                       outcome_mean: float | None = None,
                       outcome_sd: float | None = None) -> Minimum | None:
    """Riley et al.'s minimum for this situation, or None where no criterion is computed (a
    multiclass outcome, a binary one whose class counts are unknown, no predictors)."""
    if parameters < 1:
        return None
    if task == "regression":
        return continuous_minimum(parameters, mean=outcome_mean, sd=outcome_sd)
    if task == "binary" and n_events and n_rows and 0 < n_events < n_rows:
        return binary_minimum(parameters, n_events / n_rows)
    return None


def prediction_concern(minimum: Minimum, n_rows: int) -> str | None:
    """The shelf's sentence when the rows fall short of Riley et al.'s minimum, else None."""
    if n_rows >= minimum.n:
        return None
    if minimum.prevalence is not None:
        assumed = (f"an outcome proportion of {minimum.prevalence:.2f} and a model explaining 15% "
                   f"of the most it could (Cox–Snell R² {minimum.r2_cs:.3f})")
    else:
        assumed = f"a model explaining 15% of the variance (R² {minimum.r2_cs:.2f})"
    return (f"{n_rows:,} rows for {minimum.parameters:,} predictor parameters: {SOURCE_RILEY} ask "
            f"for at least {minimum.n:,} to fit them without a penalty, assuming {assumed}; "
            f"the binding criterion is {minimum.binding.what}.")


def inference_concern(task: str, parameters: int, n_rows: int,
                      n_events: int | None = None) -> tuple[str, bool] | None:
    """The shelf's sentence when there are too few rows (or events) for the coefficients, and
    whether the shortfall is severe (under half the rule); None when the rule is met."""
    if parameters < 1:
        return None
    if task == "regression":
        per = n_rows / parameters
        if per >= SPV_INFERENCE:
            return None
        return (f"{n_rows:,} rows for {parameters:,} predictor parameters ({per:.1f} each): "
                f"{SOURCE_SPV} found about {SPV_INFERENCE} per parameter enough for least-squares "
                f"coefficients, standard errors and intervals.", per < SPV_INFERENCE / 2)
    if task == "binary" and n_events is not None:
        epv = n_events / parameters
        if epv >= EPV_INFERENCE:
            return None
        return (f"{n_events:,} events for {parameters:,} predictor parameters ({epv:.1f} each); "
                f"a common rule for logistic coefficients asks for {EPV_INFERENCE} (Peduzzi et "
                f"al. 1996).", epv < EPV_INFERENCE / 2)
    return None


__all__ = [
    "Criterion", "EPV_INFERENCE", "Minimum", "R2_SHARE", "SHRINKAGE", "SPV_INFERENCE",
    "binary_minimum", "continuous_minimum", "expected_shrinkage", "inference_concern",
    "max_r2_cs", "prediction_concern", "prediction_minimum",
]
