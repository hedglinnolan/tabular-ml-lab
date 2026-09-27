"""Performance metrics by task (M1_CONTRACT §3 "fit").

Regression: R², RMSE, MAE. Binary: AUC, Brier score, log loss (the positive class is the
second of the sorted classes, as scikit-learn orders them). Multiclass: accuracy, macro-F1,
log loss. Each is scikit-learn's own function, so a number here is the number anyone gets from
the same predictions.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from turbotab.core.decisions import Task

METRICS: dict[str, tuple[str, ...]] = {
    "regression": ("r2", "rmse", "mae"),
    "binary": ("auc", "brier", "log_loss"),
    "multiclass": ("accuracy", "macro_f1", "log_loss"),
}
PRIMARY: dict[str, str] = {"regression": "r2", "binary": "auc", "multiclass": "macro_f1"}
LABELS: dict[str, str] = {
    "r2": "R²", "rmse": "RMSE", "mae": "MAE", "auc": "AUC", "brier": "Brier score",
    "log_loss": "Log loss", "accuracy": "Accuracy", "macro_f1": "Macro-F1",
}


def metric_labels(task: Task) -> dict[str, str]:
    return {m: LABELS[m] for m in METRICS[task]}


def score(task: Task, model: Any, X: Any, y: Any) -> dict[str, float]:
    """Every metric for ``task`` of a fitted model on ``X``, ``y``."""
    from sklearn import metrics as m

    y = np.asarray(y)
    if task == "regression":
        pred = model.predict(X)
        return {
            "r2": float(m.r2_score(y, pred)),
            "rmse": float(m.root_mean_squared_error(y, pred)),
            "mae": float(m.mean_absolute_error(y, pred)),
        }
    classes = list(model.classes_)
    proba = model.predict_proba(X)
    if task == "binary":
        positive = y == classes[1]
        return {
            "auc": float(m.roc_auc_score(positive, proba[:, 1])),
            "brier": float(m.brier_score_loss(positive, proba[:, 1])),
            "log_loss": float(m.log_loss(y, proba, labels=classes)),
        }
    pred = model.predict(X)
    return {
        "accuracy": float(m.accuracy_score(y, pred)),
        "macro_f1": float(m.f1_score(y, pred, average="macro", labels=classes)),
        "log_loss": float(m.log_loss(y, proba, labels=classes)),
    }


def summarize(task: Task, per_fold: list[dict[str, float]]) -> dict[str, dict[str, Any]]:
    """``{metric: {mean, sd, folds}}``; ``sd`` is the sample SD over folds (null for one fold)."""
    out: dict[str, dict[str, Any]] = {}
    for metric in METRICS[task]:
        values = [f[metric] for f in per_fold]
        arr = np.asarray(values, dtype=float)
        out[metric] = {
            "mean": float(arr.mean()),
            "sd": float(arr.std(ddof=1)) if arr.size > 1 else None,
            "folds": [float(v) for v in values],
        }
    return out


__all__ = ["LABELS", "METRICS", "PRIMARY", "metric_labels", "score", "summarize"]
