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


# ─────────────────────────────────────────────────────────────────────────────
# `MODELS-009`. THE COMPARATOR GETS ITS OWN PREPROCESSING.
#
# The baselines used to be transformed through
# `fitted_preprocessing_pipelines[first_model]` — whichever per-model pipeline
# happened to be first in dict insertion order, i.e. the order the analyst
# ticked the model checkboxes. Per-model pipelines are the product's
# differentiator precisely because they differ, so training {ridge, rf} and
# {rf, ridge} gave the linear baseline a different feature matrix and a
# different R², and the headline claim "our model beats the linear baseline by
# X" moved with the checkbox order. A bare `except Exception` then fell back to
# raw arrays with no message at all.
#
# The baseline's preprocessing is now a fixed, stated recipe that belongs to
# the baseline: it does not inherit anyone's pipeline, and it is named in the
# record next to the numbers it produced.
# ─────────────────────────────────────────────────────────────────────────────
BASELINE_PREPROCESSING_DESCRIPTION = (
    "median imputation and z-score standardization for numeric columns; "
    "most-frequent imputation and one-hot encoding for categorical columns "
    "(fitted on the training rows only)"
)


def build_baseline_preprocessor(X_train):
    """The baselines' own preprocessor — independent of any user model."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric_block = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_block = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                  drop="if_binary")),
    ])

    if not hasattr(X_train, "columns"):
        # A plain array carries no dtypes to route on; it is numeric by then.
        return numeric_block

    numeric_cols = list(X_train.select_dtypes(include=[np.number]).columns)
    categorical_cols = [c for c in X_train.columns if c not in numeric_cols]
    transformers = []
    if numeric_cols:
        transformers.append(("numeric", numeric_block, numeric_cols))
    if categorical_cols:
        transformers.append(("categorical", categorical_block, categorical_cols))
    if not transformers:
        raise ValueError("No columns to build a baseline preprocessor from.")
    return ColumnTransformer(transformers=transformers, remainder="drop",
                             verbose_feature_names_out=False)


def prepare_baseline_matrices(X_train, X_test) -> Tuple[np.ndarray, np.ndarray, str]:
    """Fit the baselines' own preprocessor on train and apply it to both.

    Raises rather than falling back to raw arrays: a baseline computed on
    untransformed data that may hold NaNs or strings is not the comparator the
    manuscript says it is, and the caller has to be able to say so.
    """
    pre = build_baseline_preprocessor(X_train)
    X_train_t = pre.fit_transform(X_train)
    X_test_t = pre.transform(X_test)
    if hasattr(X_train_t, "toarray"):
        X_train_t = X_train_t.toarray()
    if hasattr(X_test_t, "toarray"):
        X_test_t = X_test_t.toarray()
    return np.asarray(X_train_t), np.asarray(X_test_t), BASELINE_PREPROCESSING_DESCRIPTION


def train_baseline_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task_type: str = "regression",
    random_state: int = 42,
    n_bootstrap: int = 1000,
    preprocessing_description: str = "",
) -> Dict[str, Dict]:
    """Train baseline models and compute metrics with bootstrap CIs.

    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        task_type: 'regression' or 'classification'
        random_state: Random seed
        n_bootstrap: Number of bootstrap resamples for CIs
        preprocessing_description: What the baseline features went through —
            recorded on every result so the comparison can be described.

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

    # The comparator's preprocessing travels with the comparator's numbers.
    for res in results.values():
        res["preprocessing"] = preprocessing_description

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
