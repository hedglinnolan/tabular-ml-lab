"""``linear``: ordinary least squares and unpenalized logistic regression.

Predictions and cross-validation use scikit-learn. When the purpose is inference, the
coefficient table comes from statsmodels (OLS, Logit or MNLogit) fit on the same training
matrix, with cluster-robust intervals when rows repeat within a participant.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from turbotab.core.decisions import Purpose, Task
from turbotab.core.models.base import (
    Assessment,
    FamilyBase,
    Situation,
    coefficient_rows,
    register_family,
)

EPV_RULE = 10  # events per predictor: a common rule of thumb (Peduzzi 1996), not a law
# Belsley, Kuh & Welsch (1980): the condition number of the design matrix with its intercept, each
# column scaled to unit length. Above 30 is the usual "moderate" mark; 1,000 is far past "strong"
# and means some columns are close to an exact linear combination of others (shares that sum to
# 100%, a total beside all of its parts), so their separate coefficients are not identified.
NEAR_SINGULAR = 1_000.0


def model_matrix(pipeline: Any, X: pd.DataFrame) -> pd.DataFrame:
    """What the model step saw: X through every step but the last."""
    if len(pipeline.steps) == 1:
        return X
    matrix = pipeline[:-1].transform(X)
    if not isinstance(matrix, pd.DataFrame):
        names = pipeline[:-1].get_feature_names_out()
        matrix = pd.DataFrame(matrix, index=X.index, columns=names)
    return matrix


def collinearity_concern(matrix: pd.DataFrame) -> str | None:
    """A plain concern when the model matrix (with its intercept) is nearly singular, else None.

    The scaled condition number (Belsley): the intercept and every column, each scaled to unit
    length, so units do not count. The columns named are those the near-dependency involves.
    """
    values = matrix.to_numpy(dtype=float, na_value=np.nan)
    values = values[np.isfinite(values).all(axis=1)]
    if values.shape[0] <= values.shape[1] or values.shape[1] == 0:
        return None
    X = np.column_stack([np.ones(len(values)), values])
    norms = np.linalg.norm(X, axis=0)
    norms[norms == 0] = 1.0
    _, s, vt = np.linalg.svd(X / norms, full_matrices=False)
    condition = float(s[0] / s[-1]) if s[-1] > 0 else float("inf")
    if condition < NEAR_SINGULAR:
        return None
    # Belsley's variance-decomposition proportions: the share of each coefficient's variance that
    # comes from the smallest singular value. Above one half, the column is in the dependency.
    with np.errstate(divide="ignore", invalid="ignore"):
        phi = (vt.T ** 2) / np.where(s > 0, s, np.finfo(float).tiny) ** 2
        proportion = phi[:, -1] / phi.sum(axis=1)
    names = ["the intercept", *[str(c) for c in matrix.columns]]
    order = np.argsort(-np.nan_to_num(proportion))
    involved = [names[j] for j in order if proportion[j] >= 0.5 and names[j] != "the intercept"]
    columns = involved[:6] or [str(c) for c in matrix.columns][:5]
    listed = columns[0] if len(columns) == 1 else f"{', '.join(columns[:-1])} and {columns[-1]}"
    size = "too large to compute" if not np.isfinite(condition) else f"{condition:,.0f}"
    return (f"The model matrix is nearly singular (scaled condition number {size}): {listed} are "
            f"close to a fixed combination of each other, so their separate coefficients are not "
            f"identified; leave one of them out.")


def statsmodels_fit(task: Task, matrix: pd.DataFrame, y: Any, classes: Any, groups: Any = None) -> Any:
    """The statsmodels fit the inference table reports; cluster-robust by ``groups`` when given."""
    import statsmodels.api as sm

    exog = sm.add_constant(matrix.astype(float), has_constant="add")
    cov: dict[str, Any] = {}
    if groups is not None:
        cov = {"cov_type": "cluster", "cov_kwds": {"groups": pd.factorize(np.asarray(groups))[0]}}
    y = np.asarray(y)
    if task == "regression":
        return sm.OLS(y.astype(float), exog).fit(**cov)
    if task == "binary":
        return sm.Logit((y == classes[1]).astype(float), exog).fit(disp=0, maxiter=200, **cov)
    codes = pd.Categorical(y, categories=list(classes)).codes
    return sm.MNLogit(codes, exog).fit(disp=0, maxiter=200, **cov)


class Linear(FamilyBase):
    key = "linear"
    label = "Linear model"
    inductive_bias = ("Each predictor adds a straight-line effect; effects add up, with no "
                      "interactions unless you build them.")
    strengths = (
        "Coefficients read as effects per unit, with confidence intervals.",
        "Stable and transparent with many rows per predictor.",
    )
    cautions = (
        "Misses curves and interactions it is not given.",
        "Unstable when predictors are many or strongly correlated.",
    )
    needs_scaling = False
    handles_missing = False

    def build(self, task: Task, purpose: Purpose | None, n_rows: int, n_features: int) -> Any:
        if task == "regression":
            from sklearn.linear_model import LinearRegression

            return LinearRegression()
        from sklearn.linear_model import LogisticRegression

        # C = inf is no penalty (sklearn 1.8+); Newton-Cholesky copes with unscaled columns.
        return LogisticRegression(C=np.inf, solver="newton-cholesky", max_iter=1000)

    def describe(self, task: Task, purpose: Purpose | None) -> tuple[str, str]:
        if task == "regression":
            label, detail = "Ordinary least squares", "Fits the straight-line effect of every column."
        else:
            label = "Logistic regression"
            detail = "Fits the log-odds effect of every column, without a penalty."
        if purpose == "inference":
            detail += " Confidence intervals come from statsmodels on the training rows."
        return label, detail

    def coefficients(self, pipeline: Any, X: Any, y: Any, *, task: Task,
                     purpose: Purpose | None, groups: Any = None) -> list[dict[str, Any]] | None:
        model = pipeline[-1]
        features = [str(f) for f in model.feature_names_in_]
        classes = list(getattr(model, "classes_", [])) or None
        if purpose != "inference":
            return coefficient_rows(features, model.coef_, intercept=model.intercept_,
                                    classes=classes if classes and len(classes) > 2 else None)
        matrix = model_matrix(pipeline, X)
        result = statsmodels_fit(task, matrix, y, classes, groups)
        return statsmodels_rows(task, result, classes)

    def assess(self, s: Situation) -> Assessment:
        concerns: list[str] = []
        if s.n_features >= s.n_rows:
            concerns.append(f"{s.n_features:,} predictors for {s.n_rows:,} rows: least squares has "
                            f"no unique solution.")
            return Assessment(0.0, "poor", tuple(concerns))
        fit = "good"
        if s.n_rows < 10 * s.n_features:
            concerns.append(f"{s.n_rows:,} rows for {s.n_features:,} predictors: unpenalized "
                            f"estimates will be unstable.")
            fit = "fair"
        if s.task == "binary" and s.n_events is not None and s.n_features:
            epv = s.n_events / s.n_features
            if epv < EPV_RULE:
                concerns.append(f"{s.n_events:,} events for {s.n_features:,} predictors ({epv:.1f} "
                                f"each); a common rule of thumb asks for {EPV_RULE}.")
                fit = "poor" if epv < EPV_RULE / 2 else "fair"
        score = 3.0 if s.purpose == "inference" else 1.5
        if fit == "fair":
            score -= 1.0
        elif fit == "poor":
            score = 0.5
        return Assessment(score, fit, tuple(concerns))


def statsmodels_rows(task: Task, result: Any, classes: list[Any] | None) -> list[dict[str, Any]]:
    """Coefficient rows (intercept first) from a statsmodels OLS, Logit or MNLogit result."""
    params = result.params
    ci = np.asarray(result.conf_int(0.05))
    pvalues = result.pvalues
    if task != "multiclass":
        names = list(params.index)
        rows = []
        for j, name in enumerate(names):
            rows.append({"feature": "(intercept)" if name == "const" else str(name)})
        est, p = np.asarray(params, dtype=float), np.asarray(pvalues, dtype=float)
        return [
            {**row, **_numbers(est[j], ci[j, 0], ci[j, 1], p[j])} for j, row in enumerate(rows)
        ]
    # MNLogit: params is (features, K-1) against the first class; conf_int stacks classes.
    names = [("(intercept)" if n == "const" else str(n)) for n in params.index]
    est = np.asarray(params, dtype=float)
    p = np.asarray(pvalues, dtype=float)
    n_features = len(names)
    rows = []
    for k in range(est.shape[1]):
        label = classes[k + 1] if classes else k + 1
        for j, name in enumerate(names):
            lo, hi = ci[k * n_features + j]
            rows.append({"feature": f"{name} [{label}]", **_numbers(est[j, k], lo, hi, p[j, k])})
    return rows


def _numbers(estimate: float, low: float, high: float, p: float) -> dict[str, float | None]:
    def clean(v: float) -> float | None:
        v = float(v)
        return v if np.isfinite(v) else None

    return {"estimate": clean(estimate), "ci_low": clean(low), "ci_high": clean(high), "p": clean(p)}


LINEAR = register_family(Linear())
