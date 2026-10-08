"""Penalized fits on wide matrices — more predictors than training rows (M2_CONTRACT §5).

At 20,000 columns an elastic-net fit is coordinate descent over every column, for every penalty
on the path, for every inner fold and every lasso-ridge mix: about four minutes per fit in double
precision, and the fit stage makes six fits (five folds and the refit). Two things change the
cost and not the answer:

* **Single precision.** The inner loop is memory-bound, and float32 halves the bytes it reads:
  ~3× faster. The matrix is standardized first, so no column needs float64's range. On the
  benchmark and in test_wide_data it chooses the same penalty, mix and genes as float64.
* **Threads.** The inner cross-validation's paths (6 mixes × 5 folds) are independent; up to
  :data:`MAX_THREADS` run at once, as the boosted trees already use every core.

The tolerance stays sklearn's (:data:`WIDE_TOL`): a looser one (1e-3) is ~5× faster again, but it
moves the chosen penalty a grid step or more and changes which genes are kept. The narrow fit's
10⁻¹² (``elastic_net.SOLVER_TOL``) is below single precision's resolution, so it would never stop.
The choice is the narrow fit's rule (the pooled inner loss, rounded); losses computed in single
precision carry its rounding (about 10⁻⁷), so a wide fit's penalty is not promised to be the same
on every platform. Narrow matrices keep every default.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np

from turbotab.core.models.elastic_net import PooledElasticNetCV

MAX_THREADS = 4
WIDE_TOL = 1e-4  # scikit-learn's own


def is_wide(n_rows: int, n_features: int) -> bool:
    return int(n_features) > int(n_rows)


def fit_threads() -> int:
    """Threads for one fit: half the cores, at most MAX_THREADS (other jobs run beside it)."""
    return max(1, min(MAX_THREADS, (os.cpu_count() or 2) // 2))


class Float32ElasticNetCV(PooledElasticNetCV):
    """:class:`PooledElasticNetCV` computed in single precision; everything else is the same."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None, **params: Any) -> "Float32ElasticNetCV":
        X = X.astype(np.float32) if hasattr(X, "astype") else np.asarray(X, dtype=np.float32)
        return super().fit(X, np.asarray(y, dtype=np.float32), sample_weight, **params)


def for_wide(model: Any, n_rows: int, n_features: int) -> Any:
    """``model`` (a PooledElasticNetCV) as it should fit a matrix of this shape: itself when
    narrow."""
    if not is_wide(n_rows, n_features) or type(model) is not PooledElasticNetCV:
        return model
    wide = Float32ElasticNetCV(**model.get_params())
    return wide.set_params(n_jobs=model.n_jobs or fit_threads(), tol=WIDE_TOL)


__all__ = ["Float32ElasticNetCV", "MAX_THREADS", "WIDE_TOL", "fit_threads", "for_wide", "is_wide"]
