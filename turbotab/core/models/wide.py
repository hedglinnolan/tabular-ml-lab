"""Penalized fits on wide matrices — more predictors than training rows (M2_CONTRACT §5).

At 20,000 columns an elastic-net fit is coordinate descent over every column, for every penalty
on the path, for every inner split and every lasso-ridge mix: about four minutes per fit in double
precision, and the fit stage makes six fits (five folds and the refit). Two things change the
cost and not the answer:

* **Single precision.** The inner loop is memory-bound, and float32 halves the bytes it reads:
  ~3× faster. The matrix is standardized first, so no column needs float64's range. On the
  benchmark and in test_wide_data it chooses the same penalty, mix and genes as float64.
* **Threads.** One split's paths, one per lasso-ridge mix, are independent; up to
  :data:`MAX_THREADS` run at once (scikit-learn's coordinate descent releases the interpreter's
  lock), as the boosted trees already use every core.

The tolerance stays sklearn's (:data:`WIDE_TOL`, with :data:`WIDE_MAX_ITER` sweeps): a looser one
(1e-3) is ~5× faster again, but it moves the chosen penalty a grid step or more and changes which
genes are kept. The narrow fit's exact path (``exact_path``) solves a linear system per step,
which does not reach these widths. The search and its choice are the family's (the path search,
``elastic_net``: each split's own λ_max, the pooled inner loss, rounded); losses computed in single
precision carry its rounding (about 10⁻⁷), so a wide fit's penalty is not promised to be the same on
every platform. Narrow matrices, and wide ones the exact path reaches
(``elastic_net.EXACT_MAX_COLUMNS`` columns or fewer), keep the exact path; a matrix past the exact
path with no more columns than rows runs coordinate descent in double precision to
``elastic_net.SOLVER_TOL``.

:class:`Float32ElasticNetCV` is the same arithmetic for ``elastic_net.PooledElasticNetCV``, the
cross-validated estimator the selection step keeps.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np

from turbotab.core.models.elastic_net import (EXACT_MAX_COLUMNS, SOLVER_MAX_ITER, SOLVER_TOL,
                                              ExactElasticNet, PooledElasticNetCV)

MAX_THREADS = 4
WIDE_TOL = 1e-4  # scikit-learn's own
WIDE_MAX_ITER = 5000


def is_wide(n_rows: int, n_features: int) -> bool:
    return int(n_features) > int(n_rows)


def fit_threads() -> int:
    """Threads for one fit: half the cores, at most MAX_THREADS (other jobs run beside it)."""
    return max(1, min(MAX_THREADS, (os.cpu_count() or 2) // 2))


class Float32ElasticNetCV(PooledElasticNetCV):
    """:class:`PooledElasticNetCV` computed in single precision by coordinate descent (the exact
    path's linear solves do not reach 20,000 columns); the choice is the same."""

    exact = False

    def fit(self, X: Any, y: Any, sample_weight: Any = None, **params: Any) -> "Float32ElasticNetCV":
        X = X.astype(np.float32) if hasattr(X, "astype") else np.asarray(X, dtype=np.float32)
        return super().fit(X, np.asarray(y, dtype=np.float32), sample_weight, **params)


class WideElasticNet(ExactElasticNet):
    """The family's refit as built for a table with more columns than rows (:func:`for_wide`).
    It fits as :class:`~turbotab.core.models.elastic_net.ExactElasticNet` does, by the matrix it
    is handed: a screen that leaves the screened net within the exact path's reach is fit exactly,
    as its inner splits' paths are; past that reach it runs :func:`coordinate_path`, in single
    precision at :data:`WIDE_TOL` while the matrix stays wide. Its own class marks a wide build
    in the family's identity."""


def coordinate_path(Z: np.ndarray, y: np.ndarray, alphas: np.ndarray, l1_ratio: float, *,
                    weights: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """One mix's least-squares path past the exact path's reach: ``(coefs (G, p), intercepts
    (G,))``, scikit-learn's ``enet_path`` on the (weighted-)centered rows, warm-started from the
    strongest penalty, in single precision at :data:`WIDE_TOL` when the matrix is wide and in
    double precision at ``SOLVER_TOL`` otherwise."""
    import warnings

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import enet_path

    Z = np.asarray(Z, dtype=float)
    y = np.asarray(y, dtype=float)
    n = Z.shape[0]
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    zbar, ybar = w @ Z / w.sum(), float(w @ y / w.sum())
    root = np.sqrt(w / w.mean())
    Zc, yc = root[:, None] * (Z - zbar), root * (y - ybar)
    wide = is_wide(*Z.shape)
    dtype = np.float32 if wide else np.float64
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        _, coefs, _ = enet_path(np.asfortranarray(Zc, dtype=dtype), yc.astype(dtype),
                                l1_ratio=float(l1_ratio), alphas=np.asarray(alphas, dtype=float),
                                tol=WIDE_TOL if wide else SOLVER_TOL,
                                max_iter=WIDE_MAX_ITER if wide else SOLVER_MAX_ITER,
                                precompute=False)
    coefs = np.asarray(coefs, dtype=float).T  # (G, p)
    return coefs, ybar - coefs @ zbar


def for_wide(model: Any, n_rows: int, n_features: int) -> Any:
    """``model`` (the family's ``ExactElasticNet``, or the selection step's PooledElasticNetCV)
    as it should be built for a matrix of this shape: itself when narrow, or when the exact path
    reaches its columns; otherwise its wide twin (:class:`WideElasticNet`, which still fits the
    matrix it is handed, or :class:`Float32ElasticNetCV` at :data:`WIDE_TOL`)."""
    if not is_wide(n_rows, n_features) or int(n_features) <= EXACT_MAX_COLUMNS:
        return model
    if type(model) is ExactElasticNet:
        return WideElasticNet(**model.get_params()).set_params(tol=WIDE_TOL,
                                                               max_iter=WIDE_MAX_ITER)
    if type(model) is PooledElasticNetCV:
        wide = Float32ElasticNetCV(**model.get_params())
        return wide.set_params(n_jobs=model.n_jobs or fit_threads(), tol=WIDE_TOL,
                               max_iter=WIDE_MAX_ITER)
    return model


__all__ = ["Float32ElasticNetCV", "MAX_THREADS", "WIDE_MAX_ITER", "WIDE_TOL",
           "WideElasticNet", "coordinate_path", "fit_threads", "for_wide", "is_wide"]
