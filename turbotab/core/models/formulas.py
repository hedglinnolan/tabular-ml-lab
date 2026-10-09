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


def ridge_shrinkage(Z: Any, penalty: float) -> Shrinkage:
    """Ridge's shrinkage of each pattern of ``Z`` (rows × columns, centered here as an unpenalized
    intercept centers it) under ``penalty`` κ on the summed loss (the module docstring). Patterns
    with no spread, beyond the matrix's rank, keep nothing."""
    Z = np.asarray(Z, dtype=float)
    d = np.linalg.svd(Z - Z.mean(axis=0), compute_uv=False)
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


FORMULAS: dict[str, Callable[..., Shrinkage]] = {
    "elastic_net_ridge_part": elastic_net_ridge_part,
}

__all__ = ["FORMULAS", "Shrinkage", "elastic_net_ridge_part", "ridge_shrinkage"]
