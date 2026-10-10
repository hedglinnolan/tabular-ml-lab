"""The complexity formulas a family's knobs name (MODEL_FAMILY_CONTRACT C7).

A :class:`~turbotab.core.models.base.Knob` names a key of :data:`FORMULAS`, and
``register_family`` refuses a key that is not here. Each callable takes the estimator's own
parameters, as the fit records them, so a setting is converted here and nowhere else.

**Ridge's shrinkage** (Hastie, Tibshirani & Friedman 2009, ESL §3.4.1, eqs. 3.47 and 3.50). On the
centered matrix Z = UDVᵀ, ridge with penalty κ on the summed loss, ½‖y − Zβ‖² + ½κ‖β‖², keeps
dⱼ²/(dⱼ² + κ) of the least-squares fit along principal pattern j, so it shrinks the patterns of
least spread most. Its effective degrees of freedom, the trace of its hat matrix, are
df(κ) = Σⱼ dⱼ²/(dⱼ² + κ): rank(Z) at κ = 0, falling toward 0 as κ grows. ESL writes κ as λ;
scikit-learn's ``Ridge`` calls it ``alpha``; per row it is nλ.

**The elastic net's ridge part.** scikit-learn's ``ElasticNet`` minimizes
(1/2n)‖y − Zβ‖² + αρ‖β‖₁ + ½α(1 − ρ)‖β‖², with ``alpha`` α and ``l1_ratio`` ρ. Times n, its ridge
part is ½nα(1 − ρ)‖β‖², so κ = nα(1 − ρ). At ρ = 0 that is the whole fit. Otherwise the lasso part
also drops columns, which the shrinkage path shows, and these factors are the ridge part's alone
(MODEL_FAMILY_CONTRACT §2.3). Rows are unweighted.

**Ridge's knob** (``ridge``, RT-5b). The ridge family tunes λ per row; its least-squares estimator,
scikit-learn's ``Ridge``, records ``alpha`` = nλ, which is κ itself (``Ridge`` minimizes
‖y − Zβ‖² + α‖β‖², the module's loss doubled). Logistic ridge's ``LogisticRegression`` records
``C`` = 1/(nλ) and minimizes C·Σℓᵢ + ½‖β‖², the same as Σℓᵢ + ½κ‖β‖² at κ = 1/C. Its factors
(MODEL_FAMILY_CONTRACT C7, a convention) take the log-likelihood's Hessian at the fitted
probabilities pᵢ, the weighted matrix W^½Z with wᵢ = pᵢ(1 − pᵢ) (ESL §4.4.1, the iteratively
reweighted least-squares step), centered by its weighted mean as the unpenalized intercept profiles
it out; the penalized fit there is a weighted ridge, so the same dⱼ²/(dⱼ² + κ) holds for that
linearized step. They are an approximation and say so (``Shrinkage.approximate``). The contract
gives them for one coefficient vector, a yes/no outcome; K classes have K coupled vectors, and no
factors are drawn for them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class Shrinkage:
    """A penalty's effect on each principal pattern of the scaled matrix, widest first."""

    spread: np.ndarray  # dⱼ²: each pattern's spread
    factors: np.ndarray  # dⱼ²/(dⱼ² + κ): the share of the fit along it that the penalty keeps
    df: float  # their sum: the effective degrees of freedom
    approximate: bool = False  # True for logistic ridge's linearized step (the module docstring)


def ridge_shrinkage(Z: Any, penalty: float, weights: Any = None) -> Shrinkage:
    """Ridge's shrinkage of each pattern of ``Z`` (rows × columns, centered here as an unpenalized
    intercept centers it) under ``penalty`` κ on the summed loss (the module docstring). With
    ``weights`` wᵢ, the patterns are those of W^½(Z − z̄_w), z̄_w the weighted mean, for the loss
    ½Σwᵢ(yᵢ − zᵢβ)². Patterns with no spread, beyond the matrix's rank, keep nothing."""
    Z = np.asarray(Z, dtype=float)
    if weights is None:
        d = np.linalg.svd(Z - Z.mean(axis=0), compute_uv=False)
    else:
        w = np.asarray(weights, dtype=float)
        d = np.linalg.svd(np.sqrt(w)[:, None] * (Z - w @ Z / w.sum()), compute_uv=False)
    # numpy's matrix_rank cut-off: singular values below it are rounding, not spread
    spread = np.where(d > d.max(initial=0.0) * max(Z.shape) * np.finfo(float).eps, d ** 2, 0.0)
    factors = np.divide(spread, spread + float(penalty), out=np.zeros_like(spread),
                        where=spread > 0)
    return Shrinkage(spread=spread, factors=factors, df=float(factors.sum()))


def elastic_net_ridge_part(Z: Any, alpha: float, l1_ratio: float) -> Shrinkage:
    """The ridge part of scikit-learn's elastic net at its ``alpha`` and ``l1_ratio``, on the
    scaled matrix ``Z`` it was fit on: κ = nα(1 − ρ) (the module docstring)."""
    n = np.asarray(Z).shape[0]
    return ridge_shrinkage(Z, n * float(alpha) * (1.0 - float(l1_ratio)))


def ridge(Z: Any, alpha: float | None = None, *, C: float | None = None,
          probabilities: Any = None) -> Shrinkage:
    """Ridge's shrinkage on the scaled matrix ``Z`` it was fit on (the module docstring), from the
    estimator's own parameter, one of the two:

    * ``alpha``, scikit-learn's ``Ridge``: κ = α, so df = Σⱼ dⱼ²/(dⱼ² + nλ) at the family's λ;
    * ``C``, its ``LogisticRegression`` for a yes/no outcome, with ``probabilities``, the fit's
      P(class 1) on those rows (or ``predict_proba``'s two columns): κ = 1/C on the matrix weighted
      by pᵢ(1 − pᵢ), an approximation, labeled so."""
    if (alpha is None) == (C is None):
        raise ValueError("ridge's knob formula takes one of Ridge's alpha and "
                         "LogisticRegression's C")
    if alpha is not None:
        return ridge_shrinkage(Z, float(alpha))
    if probabilities is None:
        raise ValueError("logistic ridge's factors need the fitted probabilities on the rows of Z")
    prob = np.asarray(probabilities, dtype=float)
    if prob.ndim == 2:
        if prob.shape[1] != 2:
            raise ValueError(f"logistic ridge's factors are drawn for a yes/no outcome only, not "
                             f"{prob.shape[1]} classes (the module docstring)")
        prob = prob[:, 1]
    weighted = ridge_shrinkage(Z, 1.0 / float(C), weights=prob * (1.0 - prob))
    return Shrinkage(spread=weighted.spread, factors=weighted.factors, df=weighted.df,
                     approximate=True)


FORMULAS: dict[str, Callable[..., Shrinkage]] = {
    "elastic_net_ridge_part": elastic_net_ridge_part,
    "ridge": ridge,
}

__all__ = ["FORMULAS", "Shrinkage", "elastic_net_ridge_part", "ridge", "ridge_shrinkage"]
