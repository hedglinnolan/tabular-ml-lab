"""
Bootstrap confidence intervals for model evaluation metrics.

Provides BCa (bias-corrected and accelerated) bootstrap CIs for all
standard regression and classification metrics.
"""
import numpy as np
from typing import Dict, Tuple, Optional, Callable, List
from dataclasses import dataclass


@dataclass
class BootstrapResult:
    """Result of a bootstrap CI calculation."""
    estimate: float
    ci_lower: float
    ci_upper: float
    ci_level: float
    n_resamples: int
    metric_name: str

    def __str__(self):
        return f"{self.estimate:.4f} [{self.ci_lower:.4f}, {self.ci_upper:.4f}]"

    def to_dict(self):
        return {
            "estimate": self.estimate,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "ci_level": self.ci_level,
            "formatted": str(self),
        }


def _bca_ci(
    data_stat: float,
    boot_stats: np.ndarray,
    jackknife_stats: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """Compute BCa (bias-corrected and accelerated) confidence interval.

    Args:
        data_stat: Statistic computed on original data
        boot_stats: Array of bootstrap replicate statistics
        jackknife_stats: Array of leave-one-out jackknife statistics
        alpha: Significance level (default 0.05 for 95% CI)

    Returns:
        (lower, upper) bounds of the CI
    """
    from scipy.stats import norm

    # Bias correction
    z0 = norm.ppf(np.mean(boot_stats < data_stat))
    if np.isinf(z0):
        z0 = 0.0

    # Acceleration
    jack_mean = np.mean(jackknife_stats)
    diffs = jack_mean - jackknife_stats
    num = np.sum(diffs ** 3)
    denom = 6.0 * (np.sum(diffs ** 2)) ** 1.5
    a = num / denom if denom != 0 else 0.0

    # Adjusted percentiles
    z_alpha = norm.ppf(alpha / 2)
    z_1alpha = norm.ppf(1 - alpha / 2)

    def _adj(z):
        num = z0 + z
        denom = 1 - a * num
        if denom == 0:
            return 0.5
        return norm.cdf(z0 + num / denom)

    p_lower = _adj(z_alpha)
    p_upper = _adj(z_1alpha)

    # Clamp
    p_lower = np.clip(p_lower, 0.001, 0.999)
    p_upper = np.clip(p_upper, 0.001, 0.999)

    ci_lower = np.percentile(boot_stats, p_lower * 100)
    ci_upper = np.percentile(boot_stats, p_upper * 100)

    return ci_lower, ci_upper


def bootstrap_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable,
    n_resamples: int = 1000,
    ci_level: float = 0.95,
    metric_name: str = "metric",
    random_state: int = 42,
    y_proba: Optional[np.ndarray] = None,
) -> BootstrapResult:
    """Compute a metric with BCa bootstrap confidence interval.

    Args:
        y_true: True labels/values
        y_pred: Predicted labels/values
        metric_fn: Function(y_true, y_pred) -> float, or Function(y_true, y_proba) -> float
        n_resamples: Number of bootstrap resamples
        ci_level: Confidence level (default 0.95)
        metric_name: Name of the metric
        random_state: Random seed
        y_proba: Predicted probabilities (for AUC etc.)

    Returns:
        BootstrapResult with estimate and CI
    """
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    alpha = 1 - ci_level

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    pred_input = y_proba if y_proba is not None else y_pred

    # Original statistic
    try:
        original_stat = metric_fn(y_true, pred_input)
    except Exception:
        return BootstrapResult(
            estimate=float('nan'), ci_lower=float('nan'), ci_upper=float('nan'),
            ci_level=ci_level, n_resamples=0, metric_name=metric_name
        )

    # Bootstrap resamples. Degenerate resamples (e.g. a single-class resample
    # for AUC) are left as NaN and dropped, rather than substituted — substitution
    # with the point estimate artificially narrows the interval, and a single
    # NaN would otherwise propagate through np.percentile into both CI bounds.
    boot_stats = np.full(n_resamples, np.nan)
    for i in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        try:
            boot_stats[i] = metric_fn(y_true[idx], pred_input[idx])
        except Exception:
            pass

    valid_boot = boot_stats[np.isfinite(boot_stats)]
    if len(valid_boot) < max(50, n_resamples // 2):
        return BootstrapResult(
            estimate=original_stat, ci_lower=float('nan'), ci_upper=float('nan'),
            ci_level=ci_level, n_resamples=len(valid_boot), metric_name=metric_name,
        )

    # Jackknife for BCa
    n_jack = min(n, 200)  # Cap jackknife at 200 for performance
    jack_stats = np.full(n, np.nan)
    for i in range(n_jack):
        idx = np.concatenate([np.arange(i), np.arange(i + 1, n)])
        try:
            jack_stats[i] = metric_fn(y_true[idx], pred_input[idx])
        except Exception:
            pass

    computed = jack_stats[:n_jack]
    jack_fill = np.nanmean(computed) if np.isfinite(computed).any() else original_stat
    jack_stats[~np.isfinite(jack_stats)] = jack_fill

    # BCa interval on the valid resamples only
    ci_lower, ci_upper = _bca_ci(original_stat, valid_boot, jack_stats, alpha)

    return BootstrapResult(
        estimate=original_stat,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_level=ci_level,
        n_resamples=len(valid_boot),
        metric_name=metric_name,
    )


def bootstrap_all_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_resamples: int = 1000,
    ci_level: float = 0.95,
    random_state: int = 42,
) -> Dict[str, BootstrapResult]:
    """Bootstrap CIs for all standard regression metrics."""
    from sklearn.metrics import (
        mean_squared_error, mean_absolute_error, r2_score, median_absolute_error
    )

    def rmse(yt, yp):
        return np.sqrt(mean_squared_error(yt, yp))

    metrics = {
        "RMSE": rmse,
        "MAE": mean_absolute_error,
        "R2": r2_score,
        "MedianAE": median_absolute_error,
    }

    results = {}
    for name, fn in metrics.items():
        results[name] = bootstrap_metric(
            y_true, y_pred, fn,
            n_resamples=n_resamples,
            ci_level=ci_level,
            metric_name=name,
            random_state=random_state,
        )
    return results


def bootstrap_all_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    n_resamples: int = 1000,
    ci_level: float = 0.95,
    random_state: int = 42,
) -> Dict[str, BootstrapResult]:
    """Bootstrap CIs for all standard classification metrics."""
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss

    def acc(yt, yp):
        return accuracy_score(yt, yp)

    def f1(yt, yp):
        return f1_score(yt, yp, average='weighted', zero_division=0)

    results = {}
    results["Accuracy"] = bootstrap_metric(
        y_true, y_pred, acc,
        n_resamples=n_resamples, ci_level=ci_level,
        metric_name="Accuracy", random_state=random_state,
    )
    results["F1"] = bootstrap_metric(
        y_true, y_pred, f1,
        n_resamples=n_resamples, ci_level=ci_level,
        metric_name="F1", random_state=random_state,
    )

    if y_proba is not None:
        try:
            def auc(yt, yp):
                if len(np.unique(yt)) < 2:
                    return float('nan')
                return roc_auc_score(yt, yp)

            results["AUC"] = bootstrap_metric(
                y_true, y_pred, auc,
                n_resamples=n_resamples, ci_level=ci_level,
                metric_name="AUC", random_state=random_state,
                y_proba=y_proba,
            )
        except Exception:
            pass

    return results


def format_metric_with_ci(result: BootstrapResult, decimal_places: int = 4) -> str:
    """Format a metric with its CI for display."""
    dp = decimal_places
    return f"{result.estimate:.{dp}f} [{result.ci_lower:.{dp}f}, {result.ci_upper:.{dp}f}]"
