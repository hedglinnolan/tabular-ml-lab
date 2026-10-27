"""``linear``: ordinary least squares and unpenalized logistic regression.

Predictions and cross-validation use scikit-learn. When the purpose is inference, the
coefficient table is made by :mod:`turbotab.core.models.inference` on the matrix of every analyzed
row (the fit stage refits the pipeline on all of them; BLUEPRINT §12 ruling 3): HC3 intervals for
least squares on independent rows, CR2 cluster-robust intervals with Bell–McCaffrey degrees of
freedom when a unit's rows repeat (refused below the unit floor), and Firth's penalized likelihood
when a column separates a binary outcome. Binary and multinomial rows carry their odds ratios or
relative-risk ratios, with the event and the reference level named.

The shelf judges the sample size by purpose (:mod:`turbotab.core.models.sample_size`): Riley et
al.'s criteria under prediction; under inference about two rows per parameter for least squares
(Austin & Steyerberg 2015) and ten events per parameter for logistic coefficients (Peduzzi 1996).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from turbotab.core.decisions import Purpose, Task
from turbotab.core.models.base import (
    CLASS_SCALES,
    Assessment,
    FamilyBase,
    Identity,
    InferenceDecl,
    Situation,
    coefficient_rows,
    register_family,
)

ROWS_PER_PREDICTOR = 10  # multiclass only: no published criterion is computed for it yet
# What the methods text calls it, by the outcome it models (``voice``'s select_models sentence).
METHODS_LABELS = {
    "regression": "linear regression",
    "binary": "logistic regression",
    "multiclass": "multinomial logistic regression",
    "ordinal": "multinomial logistic regression, which ignores the levels' order",
}
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


def as_clusters(groups: Any) -> Any:
    """``groups`` as :class:`~turbotab.core.models.inference.Clusters`: given as such, or as one
    unit label per row (a missing label is its own unit, as the split maps it)."""
    from turbotab.core.models.inference import INDEPENDENT, Clusters

    if groups is None:
        return INDEPENDENT
    if isinstance(groups, Clusters):
        return groups
    values = pd.Series(np.asarray(groups, dtype=object))
    missing = values.isna().to_numpy()
    keys = values.to_numpy(dtype=object).copy()
    keys[missing] = [f"__missing_{i}" for i in np.flatnonzero(missing)]
    codes = pd.factorize(pd.Series(keys, dtype=object))[0].astype(np.int64)
    if len(codes) == 0 or codes.max() + 1 == len(codes):
        return INDEPENDENT
    return Clusters(column="unit", codes=codes, n_clusters=int(codes.max()) + 1,
                    n_missing=int(missing.sum()))


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
    linear_in_values = True
    # MODEL_FAMILY_CONTRACT §1 (§3.1's row for it).
    identity = Identity(kind="estimator", library="scikit-learn",
                        estimator="LinearRegression; LogisticRegression (no penalty)")
    purposes = ("prediction", "inference")
    predicts = True
    flexible = False
    bootstrap_optimism = True
    inference_decl = InferenceDecl(table="intervals",
                                   intervals=("model", "HC3", "CR2", "profile", "Taylor"),
                                   design_based=True, product_terms=True, matrix_table=True,
                                   default_for=("regression", "binary"),
                                   words={"regression": "least squares",
                                          "binary": "logistic regression (pseudo-maximum "
                                                    "likelihood)",
                                          "multiclass": "multinomial logistic regression "
                                                        "(pseudo-maximum likelihood)",
                                          "ordinal": "multinomial logistic regression "
                                                     "(pseudo-maximum likelihood)"})
    invariances = ("linear_maps",)
    curve_shape = "straight"
    diagnostics = ("separation", "collinearity", "residual_spread", "influence")
    output = "margin"
    updating = ("shrinkage",)
    raw_scale = CLASS_SCALES
    attribution = "linear"
    architecture = ("equation",)
    review_lenses = ("shared",)
    consequence = ("OLS or logistic regression: one reportable coefficient per predictor, with "
                   "intervals for inference.")
    cost_model = "cross_product"  # least squares and Newton-Cholesky factor the p × p product

    def methods_label(self, task: Task | None) -> str:
        return METHODS_LABELS.get(task, "a linear model")

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
            if purpose == "inference":
                detail += (" The coefficient table uses every analyzed row; intervals use HC3 "
                           "robust standard errors, or CR2 cluster-robust ones when a unit's rows "
                           "repeat, or Taylor linearization over a survey design.")
        else:
            label = "Logistic regression"
            detail = "Fits the log-odds effect of every column, without a penalty."
            if purpose == "inference":
                detail += (" The coefficient table uses every analyzed row and reports odds ratios "
                           "(relative-risk ratios for several classes); intervals are "
                           "cluster-robust (CR2) when a unit's rows repeat; a column that separates "
                           "the outcome gets Firth's penalized fit.")
        return label, detail

    def coefficients(self, pipeline: Any, X: Any, y: Any, *, task: Task,
                     purpose: Purpose | None, groups: Any = None) -> list[dict[str, Any]] | None:
        model = pipeline[-1]
        features = [str(f) for f in model.feature_names_in_]
        classes = list(getattr(model, "classes_", [])) or None
        if purpose != "inference":
            return coefficient_rows(features, model.coef_, intercept=model.intercept_,
                                    classes=classes if classes and len(classes) > 2 else None)
        return self.inference(pipeline, X, y, task=task, clusters=as_clusters(groups)).rows

    def inference(self, pipeline: Any, X: Any, y: Any, *, task: Task, clusters: Any,
                  outcome: Any = None, rows: Any = None, survey: Any = None) -> Any:
        """The inference table (:class:`~turbotab.core.models.inference.InferenceTable`): rows,
        how their intervals were made, and the concerns to state, on the matrix the model saw.

        ``outcome`` (:class:`~turbotab.core.models.inference.Outcome`) names the outcome and its
        levels for the odds or relative-risk ratios; ``rows`` says which rows ``X`` holds. With
        ``survey`` (a :class:`~turbotab.core.models.survey.SurveyDesign`, the "surveyed population"
        answer) the table is design-based: weighted, with Taylor-linearized intervals over the
        design's strata and PSUs, ``X``'s rows its domain (audit §5 WP10)."""
        classes = list(getattr(pipeline[-1], "classes_", [])) or None
        return self.inference_matrix(model_matrix(pipeline, X), y, task=task, classes=classes,
                                     clusters=clusters, outcome=outcome, rows=rows, survey=survey)

    def inference_matrix(self, matrix: pd.DataFrame, y: Any, *, task: Task, classes: Any,
                         clusters: Any, outcome: Any = None, rows: Any = None,
                         survey: Any = None) -> Any:
        """The inference table on a given model matrix (the all-components contrasts and a
        quintile trend test refit on one): design-based under ``survey`` (WP10), model-based or
        robust otherwise
        (``models/inference.py``), and either way on the outcome's scale (WP8)."""
        from turbotab.core.models.inference import _on_rows, _on_scale, inference_table

        if survey is None:
            return inference_table(task, matrix, y, classes, clusters, outcome=outcome, rows=rows)
        from turbotab.core.models.survey import survey_table

        table = survey_table(task, matrix, y, classes, survey)
        if table.rows:  # a blocked or refused design leaves no rows to put on a scale
            table = _on_scale(table, task, classes, outcome)
        n_domain = (table.info.get("survey") or {}).get("n_domain")
        if n_domain is not None and int(n_domain) < len(matrix):
            # Rows with no positive weight or no place in the design are outside the domain.
            table.info.update(n_rows=int(n_domain), rows=rows)
            table.info["caption"] += (f" Estimated from the {int(n_domain):,} of the "
                                      f"{len(matrix):,} analyzed rows with a positive weight and a "
                                      f"place in the design.")
            return table
        return _on_rows(table, len(matrix), rows)

    def assess(self, s: Situation) -> Assessment:
        from turbotab.core.models.sample_size import (
            inference_concern,
            prediction_concern,
            prediction_minimum,
        )

        concerns: list[str] = []
        if s.n_features >= s.n_rows:
            concerns.append(f"{s.n_features:,} predictors for {s.n_rows:,} rows: least squares has "
                            f"no unique solution.")
            return Assessment(0.0, "poor", tuple(concerns))
        fit = "good"
        parameters = s.n_parameters if s.n_parameters is not None else s.n_features
        if s.purpose == "inference":
            short = inference_concern(s.task, parameters, s.n_rows, s.n_events)
            if short is not None:
                concerns.append(short[0])
                fit = "poor" if short[1] else "fair"
        else:  # prediction, or a purpose not declared yet: the model is judged by how it predicts
            minimum = prediction_minimum(s.task, parameters, n_rows=s.n_rows, n_events=s.n_events,
                                         outcome_mean=s.outcome_mean, outcome_sd=s.outcome_sd)
            said = prediction_concern(minimum, s.n_rows) if minimum is not None else None
            if said:
                concerns.append(said)
                fit = "fair"
        if s.task == "multiclass" and s.n_rows < ROWS_PER_PREDICTOR * parameters:
            concerns.append(f"{s.n_rows:,} rows for {parameters:,} predictor parameters: "
                            f"unpenalized estimates will be unstable.")
            fit = "fair" if fit == "good" else fit
        units = getattr(s, "n_units", None)
        if s.purpose == "inference" and units is not None:
            from turbotab.core.models.inference import min_clusters

            if units < min_clusters():
                # Below the unit floor its table reports no interval (``inference.floor_refusal``).
                other = ("; a random-intercept mixed model can" if s.task == "regression" else "")
                concerns.append(f"Rows repeat within only {units} units, fewer than the "
                                f"{min_clusters()} a cluster-robust interval needs, so no interval "
                                f"is reported{other}.")
                fit = "poor"
        score = 3.0 if s.purpose == "inference" else 1.5
        if fit == "fair":
            score -= 1.0
        elif fit == "poor":
            score = 0.5
        return Assessment(score, fit, tuple(concerns))


LINEAR = register_family(Linear())
