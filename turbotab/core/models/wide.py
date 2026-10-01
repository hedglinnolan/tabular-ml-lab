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

The tolerance stays sklearn's: a looser one (1e-3) is ~5× faster again, but it moves the chosen
penalty a grid step or more and changes which genes are kept. Narrow matrices keep every default.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
from sklearn.linear_model import ElasticNetCV

MAX_THREADS = 4


def is_wide(n_rows: int, n_features: int) -> bool:
    return int(n_features) > int(n_rows)


def fit_threads() -> int:
    """Threads for one fit: half the cores, at most MAX_THREADS (other jobs run beside it)."""
    return max(1, min(MAX_THREADS, (os.cpu_count() or 2) // 2))


class Float32ElasticNetCV(ElasticNetCV):
    """``ElasticNetCV`` computed in single precision; everything else is sklearn's own."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None, **params: Any) -> "Float32ElasticNetCV":
        X = X.astype(np.float32) if hasattr(X, "astype") else np.asarray(X, dtype=np.float32)
        return super().fit(X, np.asarray(y, dtype=np.float32), sample_weight, **params)


def for_wide(model: Any, n_rows: int, n_features: int) -> Any:
    """``model`` (an ElasticNetCV) as it should fit a matrix of this shape: itself when narrow."""
    if not is_wide(n_rows, n_features) or type(model) is not ElasticNetCV:
        return model
    wide = Float32ElasticNetCV(**model.get_params())
    return wide.set_params(n_jobs=model.n_jobs or fit_threads())


__all__ = ["Float32ElasticNetCV", "MAX_THREADS", "fit_threads", "for_wide", "is_wide"]
