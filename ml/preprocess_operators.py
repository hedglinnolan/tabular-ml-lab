"""
Custom preprocessing operators for unit harmonization and plausibility gating.
"""
from __future__ import annotations

from typing import Optional, Sequence, Dict
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


def _bounds_array(bounds: Sequence[Optional[float]]) -> np.ndarray:
    """Coerce a bounds sequence to a float array, `None` becoming NaN.

    This runs at fit time, never in `__init__`. `sklearn.base.clone`
    reconstructs an estimator from `get_params()` and then asserts that each
    parameter is the *same object* it passed in, so a constructor that coerces
    what it was handed cannot be cloned — and every refit path in this app
    clones (`reconcile_pipeline_columns` before each training run,
    `make_cv_pipeline` once per fold, seed sensitivity, feature dropout).
    """
    return np.array(
        [np.nan if v is None else float(v) for v in bounds], dtype=float
    )


class UnitHarmonizer(BaseEstimator, TransformerMixin):
    """Convert numeric features to canonical units using per-feature factors.
    Store conversion_factors by reference (no copy) so sklearn clone works."""

    def __init__(self, conversion_factors: Sequence[float]):
        self.conversion_factors = conversion_factors

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=float)
        factors = np.asarray(self.conversion_factors, dtype=float)
        return X_arr * factors


def plausibility_row_mask(
    X: np.ndarray,
    lower_bounds: Sequence[Optional[float]],
    upper_bounds: Sequence[Optional[float]],
) -> np.ndarray:
    """Compute a row-level mask: True = keep row (all gated cols in range)."""
    X_arr = np.asarray(X, dtype=float)
    n_cols = min(X_arr.shape[1], len(lower_bounds), len(upper_bounds))
    keep = np.ones(X_arr.shape[0], dtype=bool)
    for idx in range(n_cols):
        low = lower_bounds[idx] if idx < len(lower_bounds) else None
        high = upper_bounds[idx] if idx < len(upper_bounds) else None
        if low is not None and not (isinstance(low, float) and np.isnan(low)):
            keep &= X_arr[:, idx] >= float(low)
        if high is not None and not (isinstance(high, float) and np.isnan(high)):
            keep &= X_arr[:, idx] <= float(high)
    return keep


class PlausibilityGate(BaseEstimator, TransformerMixin):
    """Set values outside empirical plausibility bounds to NaN.

    Constructor arguments are stored by reference, exactly as UnitHarmonizer
    documents; the coercion to float arrays happens in `fit` and lands on
    `lower_bounds_` / `upper_bounds_`. Doing it in `__init__` made this class
    unclonable, which silently disabled cross-validation and turned the
    pipeline-drift reconciler from self-heal-loudly into crash-quietly for
    every project with plausibility bounds configured.
    """

    def __init__(self, lower_bounds: Sequence[Optional[float]], upper_bounds: Sequence[Optional[float]]):
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

    def fit(self, X, y=None):
        self.lower_bounds_ = _bounds_array(self.lower_bounds)
        self.upper_bounds_ = _bounds_array(self.upper_bounds)
        return self

    def transform(self, X):
        check_is_fitted(self, ("lower_bounds_", "upper_bounds_"))
        X_arr = np.asarray(X, dtype=float).copy()
        # `STATE-003`: the bounds are POSITIONAL, so a matrix that is not the
        # column set they were built for means bound j lands on column j' != j —
        # an NHANES band for one biomarker applied to another, every value
        # outside it turned to NaN, and the very next pipeline step imputing
        # those NaNs to the median. `min()` used to absorb the mismatch in
        # silence. `UnitHarmonizer` fails loudly on the same mistake; so does
        # this now.
        if len(self.lower_bounds_) != X_arr.shape[1] or len(self.upper_bounds_) != X_arr.shape[1]:
            raise ValueError(
                f"PlausibilityGate was fitted with {len(self.lower_bounds_)} "
                f"lower and {len(self.upper_bounds_)} upper bound(s) but was "
                f"handed a matrix with {X_arr.shape[1]} column(s). The bounds "
                f"are positional, so applying them here would gate columns "
                f"against another column's reference range.")
        n_cols = X_arr.shape[1]
        for idx in range(n_cols):
            lower = self.lower_bounds_[idx]
            upper = self.upper_bounds_[idx]
            if not np.isnan(lower):
                mask_lo = X_arr[:, idx] < lower
                X_arr[mask_lo, idx] = np.nan
            if not np.isnan(upper):
                mask_hi = X_arr[:, idx] > upper
                X_arr[mask_hi, idx] = np.nan
        return X_arr


class OutlierCapping(BaseEstimator, TransformerMixin):
    """Cap outliers based on percentile or MAD bounds computed at fit time.

    `params` is stored unmodified — `params or {}` returned a *different*
    empty dict than the one it was handed, which broke `clone` for exactly the
    configuration `ml/pipeline.py` builds when no outlier parameters are set.
    The fitted bounds are not created until `fit`, so a rebuilt-but-unfitted
    capper refuses to transform rather than passing uncapped data through
    while the recorded configuration says capping is on.
    """

    def __init__(self, method: str = "percentile", params: Optional[Dict] = None):
        self.method = method
        self.params = params

    def fit(self, X, y=None):
        X_arr = np.asarray(X, dtype=float)
        params = self.params or {}
        if self.method == "mad":
            threshold = float(params.get("threshold", 3.5))
            med = np.nanmedian(X_arr, axis=0)
            mad = np.nanmedian(np.abs(X_arr - med), axis=0)
            scale = 1.4826 * mad
            self.lower_bounds_ = med - threshold * scale
            self.upper_bounds_ = med + threshold * scale
            # `TEST-018`: mad == 0 whenever at least half the values equal the
            # median — every binary flag with prevalence != 50%, every integer
            # code with a dominant level, every mostly-zero count. The bounds
            # then collapse to [median, median] and the clip below forces the
            # whole column to that constant: the predictor is destroyed inside
            # the pipeline, its coefficient and SHAP value are zero for a
            # mechanical reason, and the recipe still says "MAD capping". A
            # capping step may trim extremes; it may not delete a variable. Such
            # a column is left uncapped and named on the fitted step, so the
            # recipe can say which ones.
            degenerate = ~np.isfinite(scale) | (scale <= 0)
            self.lower_bounds_ = np.where(degenerate, -np.inf, self.lower_bounds_)
            self.upper_bounds_ = np.where(degenerate, np.inf, self.upper_bounds_)
            self.uncapped_columns_ = np.flatnonzero(degenerate).tolist()
        else:
            lower_q = float(params.get("lower_q", 0.01))
            upper_q = float(params.get("upper_q", 0.99))
            self.lower_bounds_ = np.nanpercentile(X_arr, lower_q * 100, axis=0)
            self.upper_bounds_ = np.nanpercentile(X_arr, upper_q * 100, axis=0)
            self.uncapped_columns_ = []
        return self

    def transform(self, X):
        check_is_fitted(self, ("lower_bounds_", "upper_bounds_"))
        X_arr = np.asarray(X, dtype=float)
        return np.clip(X_arr, self.lower_bounds_, self.upper_bounds_)
