"""``elastic_net``: a penalized linear model whose penalty is tuned by an inner cross-validation.

The inner cross-validation is the estimator's own (``ElasticNetCV`` / ``LogisticRegressionCV``),
so it only ever splits the rows the estimator is fit on: inside an outer fold, that is the
training part of the fold and nothing else.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from turbotab.core.decisions import Purpose, Task
from turbotab.core.models.base import Assessment, FamilyBase, Situation, coefficient_rows, register_family

L1_RATIOS = (0.1, 0.5, 0.7, 0.9, 0.95, 1.0)
# saga is slow on the full default grid (≈ 150 s per multiclass fit on 17,000 NHANES rows); this
# grid stops short of the nearly unpenalized end, which the linear family already covers.
LOGISTIC_L1_RATIOS = (0.2, 0.6, 1.0)
LOGISTIC_CS = tuple(float(c) for c in np.logspace(-3, 1, 8))


def inner_folds(n_rows: int) -> int:
    return 5 if 100 <= n_rows <= 5000 else 3


class ElasticNet(FamilyBase):
    key = "elastic_net"
    label = "Elastic net"
    inductive_bias = ("Straight-line effects shrunk toward zero; few predictors matter, correlated "
                      "ones share weight.")
    strengths = (
        "Works with more predictors than rows.",
        "Tunes its own penalty by cross-validation on training rows.",
    )
    cautions = (
        "Shrunk coefficients have no confidence intervals.",
        "Which of several correlated predictors it keeps is arbitrary.",
    )
    needs_scaling = True
    handles_missing = False

    def build(self, task: Task, purpose: Purpose | None, n_rows: int, n_features: int) -> Any:
        cv = inner_folds(n_rows)
        if task == "regression":
            from sklearn.linear_model import ElasticNetCV

            from turbotab.core.models.wide import for_wide  # p > n: float32 and threads

            return for_wide(ElasticNetCV(l1_ratio=list(L1_RATIOS), cv=cv, max_iter=5000),
                            n_rows, n_features)
        from sklearn.linear_model import LogisticRegressionCV

        return LogisticRegressionCV(
            Cs=list(LOGISTIC_CS), l1_ratios=LOGISTIC_L1_RATIOS, solver="saga", cv=cv,
            scoring="neg_log_loss", max_iter=1000, tol=1e-3, use_legacy_attributes=False,
            random_state=0,
        )

    def describe(self, task: Task, purpose: Purpose | None) -> tuple[str, str]:
        label = "Elastic net regression" if task == "regression" else "Penalized logistic regression"
        return label, ("Chooses the penalty's strength and its lasso-ridge mix by an inner "
                       "cross-validation within the training rows it is given.")

    def coefficients(self, pipeline: Any, X: Any, y: Any, *, task: Task,
                     purpose: Purpose | None, groups: Any = None) -> list[dict[str, Any]] | None:
        """Coefficients per unit of each matrix column (the scaling undone); no intervals."""
        model = pipeline[-1]
        features = [str(f) for f in model.feature_names_in_]
        coef = np.atleast_2d(np.asarray(model.coef_, dtype=float))
        intercept = np.atleast_1d(np.asarray(model.intercept_, dtype=float))
        scaler = pipeline.named_steps.get("scale")
        if scaler is not None:
            coef = coef / scaler.scale_
            intercept = intercept - coef @ scaler.mean_
        classes = list(getattr(model, "classes_", [])) or None
        return coefficient_rows(features, coef, intercept=intercept,
                                classes=classes if classes and len(classes) > 2 else None)

    def assess(self, s: Situation) -> Assessment:
        concerns: list[str] = []
        if s.n_features >= s.n_rows:
            concerns.append(f"{s.n_features:,} predictors for {s.n_rows:,} rows: the predictors it "
                            f"keeps will change from sample to sample.")
            return Assessment(4.0, "good", tuple(concerns))
        fit = "good"
        score = 2.5
        if s.purpose == "inference":
            concerns.append("Penalized coefficients are shrunk and carry no confidence intervals.")
            fit, score = "fair", 2.0
        if s.n_rows < 50:
            concerns.append(f"With {s.n_rows:,} rows the inner cross-validation picks the penalty "
                            f"from very few rows.")
            fit = "fair"
        return Assessment(score, fit, tuple(concerns))


ELASTIC_NET = register_family(ElasticNet())
