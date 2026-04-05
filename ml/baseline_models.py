"""
Baseline/null model comparison.

Automatically trains trivial baselines (mean predictor, majority class,
simple linear/logistic regression) and compares against user models.
"""
import logging
import numpy as np
from typing import Dict, Optional, Tuple
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, median_absolute_error,
    accuracy_score, f1_score, roc_auc_score,
)
from ml.bootstrap import bootstrap_all_regression_metrics, bootstrap_all_classification_metrics

logger = logging.getLogger(__name__)


def train_baseline_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task_type: str = "regression",
    random_state: int = 42,
    n_bootstrap: int = 1000,
) -> Dict[str, Dict]:
    """Train baseline models and compute metrics with bootstrap CIs.

    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        task_type: 'regression' or 'classification'
        random_state: Random seed
        n_bootstrap: Number of bootstrap resamples for CIs

    Returns:
        Dict of {model_name: {metrics, y_pred, bootstrap_cis, description}}
    """
    results = {}

    if task_type == "regression":
        # Mean predictor (null model)
        dummy = DummyRegressor(strategy="mean")
        dummy.fit(X_train, y_train)
        y_pred_dummy = dummy.predict(X_test)

        dummy_metrics = {
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_dummy)),
            "MAE": mean_absolute_error(y_test, y_pred_dummy),
            "R2": r2_score(y_test, y_pred_dummy),
            "MedianAE": median_absolute_error(y_test, y_pred_dummy),
        }
        dummy_cis = bootstrap_all_regression_metrics(
            y_test, y_pred_dummy, n_resamples=n_bootstrap, random_state=random_state
        )
        results["Baseline: Mean"] = {
            "metrics": dummy_metrics,
            "y_pred": y_pred_dummy,
            "bootstrap_cis": dummy_cis,
            "description": "Predicts the training set mean for all samples. Any useful model must beat this.",
            "model": dummy,
        }

        # Simple linear regression
        try:
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            y_pred_lr = lr.predict(X_test)

            lr_metrics = {
                "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_lr)),
                "MAE": mean_absolute_error(y_test, y_pred_lr),
                "R2": r2_score(y_test, y_pred_lr),
                "MedianAE": median_absolute_error(y_test, y_pred_lr),
            }
            lr_cis = bootstrap_all_regression_metrics(
                y_test, y_pred_lr, n_resamples=n_bootstrap, random_state=random_state
            )
            results["Baseline: Linear Regression"] = {
                "metrics": lr_metrics,
                "y_pred": y_pred_lr,
                "bootstrap_cis": lr_cis,
                "description": "Ordinary least squares. The simplest useful model — your model should improve on this.",
                "model": lr,
            }
        except Exception as e:
            logger.warning("Baseline linear regression failed: %s", e)

    else:  # classification
        # Majority class predictor
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(X_train, y_train)
        y_pred_dummy = dummy.predict(X_test)

        dummy_metrics = {
            "Accuracy": accuracy_score(y_test, y_pred_dummy),
            "F1": f1_score(y_test, y_pred_dummy, average='weighted', zero_division=0),
        }
        dummy_cis = bootstrap_all_classification_metrics(
            y_test, y_pred_dummy, n_resamples=n_bootstrap, random_state=random_state
        )
        results["Baseline: Majority Class"] = {
            "metrics": dummy_metrics,
            "y_pred": y_pred_dummy,
            "bootstrap_cis": dummy_cis,
            "description": "Always predicts the most common class. Any useful model must beat this.",
            "model": dummy,
        }

        # Simple logistic regression
        try:
            log_reg = LogisticRegression(
                random_state=random_state, max_iter=1000, C=1.0
            )
            log_reg.fit(X_train, y_train)
            y_pred_log = log_reg.predict(X_test)
            y_proba_log = log_reg.predict_proba(X_test) if hasattr(log_reg, 'predict_proba') else None

            log_metrics = {
                "Accuracy": accuracy_score(y_test, y_pred_log),
                "F1": f1_score(y_test, y_pred_log, average='weighted', zero_division=0),
            }
            if y_proba_log is not None and len(np.unique(y_test)) == 2:
                try:
                    log_metrics["AUC"] = roc_auc_score(y_test, y_proba_log[:, 1])
                except Exception as e:
                    logger.debug("Baseline AUC computation failed: %s", e)

            log_cis = bootstrap_all_classification_metrics(
                y_test, y_pred_log, y_proba=y_proba_log[:, 1] if y_proba_log is not None and y_proba_log.shape[1] == 2 else None,
                n_resamples=n_bootstrap, random_state=random_state
            )
            results["Baseline: Logistic Regression"] = {
                "metrics": log_metrics,
                "y_pred": y_pred_log,
                "y_proba": y_proba_log,
                "bootstrap_cis": log_cis,
                "description": "Simple logistic regression with L2 penalty. A solid baseline for classification.",
                "model": log_reg,
            }
        except Exception as e:
            logger.warning("Baseline logistic regression failed: %s", e)

    return results


def format_comparison_table(
    baseline_results: Dict[str, Dict],
    model_results: Dict[str, Dict],
    task_type: str = "regression",
) -> "pd.DataFrame":
    """Create a comparison DataFrame with baselines and user models.

    Returns a DataFrame with metrics and bootstrap CIs formatted for display.
    """
    import pandas as pd

    rows = []

    # Add baselines first
    for name, res in baseline_results.items():
        row = {"Model": name, "_is_baseline": True}
        cis = res.get("bootstrap_cis", {})
        for metric_name, value in res["metrics"].items():
            ci = cis.get(metric_name)
            if ci:
                row[metric_name] = f"{value:.4f} [{ci.ci_lower:.4f}, {ci.ci_upper:.4f}]"
            else:
                row[metric_name] = f"{value:.4f}"
        rows.append(row)

    # Add user models
    for name, res in model_results.items():
        row = {"Model": name.upper(), "_is_baseline": False}
        cis = res.get("bootstrap_cis", {})
        for metric_name, value in res["metrics"].items():
            ci = cis.get(metric_name)
            if ci:
                row[metric_name] = f"{value:.4f} [{ci.ci_lower:.4f}, {ci.ci_upper:.4f}]"
            else:
                row[metric_name] = f"{value:.4f}"
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.drop(columns=["_is_baseline"], errors="ignore")
    return df
