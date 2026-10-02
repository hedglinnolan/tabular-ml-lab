"""Performance metrics by task, and how a cross-validated score is estimated (M1_CONTRACT §3 "fit").

Regression: R², RMSE, MAE. Binary: AUC, Brier score, log loss (the positive class is the second
of the sorted classes, as scikit-learn orders them). Multiclass: accuracy, macro-F1, log loss.
Ordinal (classes are the codes 0…K − 1 of the declared order): Harrell's C, the ranked probability
score, log loss and the mean absolute error in levels (:func:`ordinal_scores`).

Time to event: Harrell's concordance index of the model's risk score
(:func:`turbotab.core.models.survival.concordance`; Harrell et al. 1996, *Stat Med* 15:361), the
time-to-event analogue of the AUC, with the outcome a structured array of event, time and entry
(``survival.survival_outcome``). Pairs are ordered by follow-up time, not by entry.

**R² is measured against the training rows' mean** — in every fold, in the pooled estimate and on
the held-out rows. R² compares the model with the model that predicts without predictors; that
model is the mean of the rows it was fit on, so its error on new rows is measured against that mean,
not against the new rows' own (which no prediction could know). Staerk et al. 2024 call the
training-mean definition the one that "may generally be preferable based on theoretical reasons".
Scored against the held-out rows' own mean, a 60-row holdout read 0.155 where the training mean
read 0.175, against a target of 0.18 (audit MA-09).

**Cross-validated R², RMSE and MAE are pooled** over every out-of-fold prediction (audit MA-09):

    R²_CV = 1 − Σ_k Σ_{i ∈ k} (y_i − ŷ_i)²  /  Σ_k Σ_{i ∈ k} (y_i − ȳ_{−k})²

where ȳ_{−k} is the mean of the rows fold k's model was fit on. The denominator is the out-of-fold
squared error of the no-predictor model, so this is Hawinkel, Waegeman & Maere's pooling estimator
(Am Stat 2024; arXiv:2302.05131): "The pooling R² estimator, which separately estimates the squared
error losses of the null and prediction models and only then combines them into a final estimate
R², is unbiased. Hence this pooling estimator should be preferred to averaging estimators that
calculate R² values in every cross-validation fold separately and then average over the folds,
which suffer from bias." Averaged fold by fold against each fold's own mean, OLS with 5 predictors
and a population R² of 0.20 read 0.05 at n = 100 against a target of 0.13–0.14. The per-fold values
are kept to show the spread. Classification scores stay the mean over folds: AUC is not pooled
across fold models, whose scores are not on one scale (Forman & Scholz 2010).

**Every cross-validated score carries a standard error** (audit ME-10; the construction is
:mod:`turbotab.core.models.performance`'s): for a fold mean, LeDell et al.'s (2015) — the square
root of the sum of the folds' own variances (DeLong's for the AUC), over the number of folds; for a
pooled score, the delta method over every out-of-fold row. Rows of one unit are summed before they
are squared. Under repeated k-fold (audit ME-11) each repeat is estimated as above, the reported
score is the mean over repeats, its SE the root mean of the repeats' squared SEs, and the spread of
the repeats' estimates is kept as ``repeat_sd``.

**The primary metric** is the one a family is compared with its baseline on and the banner names:
R² for regression; AUC for a binary outcome, said as "the highest AUC" because AUC ranks risks and
says nothing of calibration (Van Calster et al. 2019); log loss for a multiclass outcome, a proper
scoring rule where macro-F1, the earlier primary, is not (audit ME-10). Log loss, Brier score, RMSE
and MAE are better when lower (:data:`LOWER_IS_BETTER`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from turbotab.core.decisions import Task

METRICS: dict[str, tuple[str, ...]] = {
    "regression": ("r2", "rmse", "mae"),
    "binary": ("auc", "brier", "log_loss"),
    "multiclass": ("log_loss", "accuracy", "macro_f1"),
    "ordinal": ("c_index", "rps", "log_loss", "mae_levels"),
    "time_to_event": ("c_index",),
}
# Log loss is a proper scoring rule; macro-F1 is not (audit ME-10). An ordered outcome is ranked on
# its concordance (WP12a), and so is a time-to-event outcome, by Harrell's C of its risk (WP12b).
PRIMARY: dict[str, str] = {"regression": "r2", "binary": "auc", "multiclass": "log_loss",
                           "ordinal": "c_index", "time_to_event": "c_index"}
LOWER_IS_BETTER = frozenset({"rmse", "mae", "brier", "log_loss", "rps", "mae_levels"})
LABELS: dict[str, str] = {
    "r2": "R²", "rmse": "RMSE", "mae": "MAE", "auc": "AUC", "brier": "Brier score",
    "log_loss": "Log loss", "accuracy": "Accuracy", "macro_f1": "Macro-F1",
    "c_index": "C-index", "rps": "Ranked probability score", "mae_levels": "MAE (levels)",
}
POOLED = ("r2", "rmse", "mae")  # estimated over every out-of-fold prediction, not fold by fold
CV_DEFINITION = {
    "regression": ("Cross-validated R², RMSE and MAE pool every out-of-fold prediction; R² is "
                   "measured against the mean of the rows each fold's model was fit on, as the "
                   "held-out R² is against the training rows' mean. The fold values show the spread."),
    "binary": "Cross-validated scores are the mean over folds; the fold values show the spread.",
    "multiclass": "Cross-validated scores are the mean over folds; the fold values show the spread.",
    "ordinal": ("Cross-validated scores are the mean over folds; the fold values show the spread. "
                "C is the share of pairs of rows at different levels whose predicted mean level is "
                "in the same order (ties count one half)."),
    "time_to_event": ("Cross-validated scores are the mean over folds of Harrell's C, the share of "
                      "comparable pairs whose order of events the risk score gets right; the fold "
                      "values show the spread."),
}
SE_DEFINITION = ("Each standard error counts which rows were scored, with each fold's model as "
                 "fitted (LeDell et al. 2015; DeLong's for the AUC); it leaves out how the models "
                 "would vary with other training rows.")


def higher_is_better(metric: str) -> bool:
    return metric not in LOWER_IS_BETTER


def metric_labels(task: Task) -> dict[str, str]:
    return {m: LABELS[m] for m in METRICS[task]}


def r2_against(y: Any, pred: Any, reference: float) -> float:
    """1 − Σ(y − ŷ)² / Σ(y − reference)²: R² against a reference mean known before scoring."""
    y = np.asarray(y, dtype=float)
    sst = float(((y - reference) ** 2).sum())
    if sst == 0:
        return float("nan")
    return 1.0 - float(((y - np.asarray(pred, dtype=float)) ** 2).sum()) / sst


def score(task: Task, model: Any, X: Any, y: Any, *, reference: float | None = None) -> dict[str, float]:
    """Every metric for ``task`` of a fitted model on ``X``, ``y``.

    ``reference``: for regression, the mean of the rows the model was fit on; R² is measured
    against it. Without one, R² is scikit-learn's (against ``y``'s own mean).
    """
    from sklearn import metrics as m

    y = np.asarray(y)
    if task == "time_to_event":
        from turbotab.core.models.survival import concordance

        return {"c_index": concordance(y["time"], y["event"], model.predict(X))}
    if task == "regression":
        pred = model.predict(X)
        r2 = (float(m.r2_score(y, pred)) if reference is None
              else r2_against(y, pred, float(reference)))
        return {
            "r2": r2,
            "rmse": float(m.root_mean_squared_error(y, pred)),
            "mae": float(m.mean_absolute_error(y, pred)),
        }
    classes = list(model.classes_)
    proba = model.predict_proba(X)
    if task == "ordinal":
        return ordinal_scores(y, proba, classes)
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


def concordance(levels: Any, score: Any) -> float:
    """Harrell's C for an ordered outcome: over every pair of rows at different levels, the share
    in which the higher level has the higher score, a tied score counting one half. Each pair is
    counted once, from its lower row's level; O(K n log n)."""
    levels = np.asarray(levels)
    score = np.asarray(score, dtype=float)
    concordant = 0.0
    pairs = 0.0
    for level in np.unique(levels)[:-1]:
        lower = np.sort(score[levels == level])
        higher = score[levels > level]
        below = np.searchsorted(lower, higher, side="left")
        at_or_below = np.searchsorted(lower, higher, side="right")
        concordant += float(below.sum() + 0.5 * (at_or_below - below).sum())
        pairs += float(len(lower) * len(higher))
    return concordant / pairs if pairs else float("nan")


def ordinal_scores(y: Any, proba: Any, classes: Sequence[Any]) -> dict[str, float]:
    """C, the ranked probability score, log loss and the absolute error in levels.

    ``classes`` are in the outcome's order (the codes of the declared order). The score C ranks is
    the predicted mean level, Σ k·p_k. The ranked probability score is the mean over rows of
    (1/(K − 1)) Σ_k (F̂_k − 1{y ≤ k})² over the K − 1 cut-points (Epstein 1969, *J Appl Meteorol*
    8:985): a proper score that counts how far off in order a forecast is. The error in levels is
    against the predictive median, the level that minimizes the expected absolute error.
    """
    from sklearn import metrics as m

    classes = list(classes)
    y = np.asarray(y)
    position = {c: i for i, c in enumerate(classes)}
    unseen = sorted({str(v) for v in y if v not in position})
    if unseen:
        raise ValueError(f"Level {', '.join(unseen)} is not among the levels the model was fit on.")
    codes = np.array([position[v] for v in y])
    proba = np.asarray(proba, dtype=float)
    K = len(classes)
    expected = proba @ np.arange(K)
    cumulative = np.cumsum(proba, axis=1)[:, :-1]
    observed = (codes[:, None] <= np.arange(K - 1)[None, :]).astype(float)
    rps = float(np.mean(np.sum((cumulative - observed) ** 2, axis=1) / max(K - 1, 1)))
    median = np.argmax(np.cumsum(proba, axis=1) >= 0.5 - 1e-12, axis=1)
    return {
        "c_index": concordance(codes, expected),
        "rps": rps,
        "log_loss": float(m.log_loss(y, proba, labels=classes)),
        "mae_levels": float(np.mean(np.abs(median - codes))),
    }


@dataclass
class FoldPart:
    """What pooling needs from one regression fold: sums over its scored rows."""

    sse: float  # Σ (y − ŷ)²
    sst: float  # Σ (y − ȳ_train)²: the no-predictor model's squared error
    sae: float  # Σ |y − ŷ|
    n: int


def fold_part(model: Any, X: Any, y: Any, reference: float) -> FoldPart:
    y = np.asarray(y, dtype=float)
    e = y - np.asarray(model.predict(X), dtype=float)
    return FoldPart(sse=float((e ** 2).sum()), sst=float(((y - reference) ** 2).sum()),
                    sae=float(np.abs(e).sum()), n=int(len(y)))


def pooled(parts: Sequence[FoldPart]) -> dict[str, float]:
    """R², RMSE and MAE over every out-of-fold prediction (the module docstring's estimator)."""
    sse = sum(p.sse for p in parts)
    sst = sum(p.sst for p in parts)
    sae = sum(p.sae for p in parts)
    n = sum(p.n for p in parts)
    if not n:
        return {m: float("nan") for m in POOLED}
    return {"r2": 1.0 - sse / sst if sst > 0 else float("nan"), "rmse": float(np.sqrt(sse / n)),
            "mae": sae / n}


def summarize(task: Task, per_fold: list[dict[str, float]],
              parts: Sequence[FoldPart] | None = None, *,
              repeat_of: Sequence[int] | None = None,
              errors: Mapping[str, float | None] | None = None) -> dict[str, dict[str, Any]]:
    """``{metric: {estimate, estimator, mean, sd, folds, se, ci_low, ci_high, repeats, repeat_sd}}``.

    ``estimate`` is the cross-validated score the app reports: pooled over every out-of-fold
    prediction for R², RMSE and MAE when ``parts`` are given, else the mean over folds; with
    ``repeat_of`` (each fold's repeat), each repeat is estimated so and the estimate is their mean.
    ``mean`` is always the mean over folds and ``sd`` their sample SD (null for one fold).
    ``errors``: each metric's standard error (:meth:`CrossValidated.standard_errors`).
    """
    from turbotab.core.models.performance import z_value

    repeat_of = list(repeat_of) if repeat_of is not None else [0] * len(per_fold)
    repeats = sorted(set(repeat_of))
    pooled_by_repeat: list[dict[str, float]] = []
    if parts and task == "regression":
        for r in repeats:
            pooled_by_repeat.append(pooled([p for p, rr in zip(parts, repeat_of) if rr == r]))
    out: dict[str, dict[str, Any]] = {}
    z = z_value()
    for metric in METRICS[task]:
        values = [f[metric] for f in per_fold]
        arr = np.asarray(values, dtype=float)
        mean = float(arr.mean()) if arr.size else float("nan")
        if pooled_by_repeat and metric in POOLED:
            each = [float(p[metric]) for p in pooled_by_repeat]
            estimator = "pooled"
        else:
            each = [float(np.mean([v for v, rr in zip(arr, repeat_of) if rr == r])) for r in repeats]
            estimator = "fold_mean"
        estimate = float(np.mean(each)) if each else float("nan")
        se = (errors or {}).get(metric)
        lo = hi = None
        if se is not None and np.isfinite(se) and np.isfinite(estimate):
            lo, hi = estimate - z * se, estimate + z * se
            if metric in ("auc", "accuracy", "macro_f1", "brier"):
                lo, hi = max(lo, 0.0), min(hi, 1.0)
            elif metric == "r2":
                hi = min(hi, 1.0)
            elif metric in LOWER_IS_BETTER:
                lo = max(lo, 0.0)
        out[metric] = {
            "estimate": estimate,
            "estimator": estimator,
            "mean": mean,
            "sd": float(arr.std(ddof=1)) if arr.size > 1 else None,
            "folds": [float(v) for v in values],
            "se": float(se) if se is not None and np.isfinite(se) else None,
            "ci_low": lo,
            "ci_high": hi,
            "repeats": len(repeats),
            "repeat_sd": float(np.std(each, ddof=1)) if len(each) > 1 else None,
        }
    return out


# ── cross-validation over the split's folds ──────────────────────────────────


def fold_pairs(folds: Any, scheme: str = "random") -> list[tuple[int, np.ndarray, np.ndarray]]:
    """``(fold, fit rows, scored rows)`` as boolean masks over the training rows.

    ``random``: each fold is scored by a model fit on every other fold. ``time_ordered``: fold
    ``j`` is scored by a model fit on the folds before it, and fold 0 only trains
    (:mod:`turbotab.core.models.folds`).
    """
    folds = np.asarray(folds).astype(int)
    keys = sorted(set(folds.tolist()))
    if scheme == "time_ordered":
        return [(k, folds < k, folds == k) for k in keys if k > keys[0]]
    return [(k, folds != k, folds == k) for k in keys]


def repeated_pairs(fold_columns: Sequence[Any], scheme: str = "random"
                   ) -> tuple[list[tuple[int, np.ndarray, np.ndarray]], list[int]]:
    """Every repeat's :func:`fold_pairs`, in order, and each pair's repeat (0, 1, …)."""
    pairs: list[tuple[int, np.ndarray, np.ndarray]] = []
    repeat_of: list[int] = []
    for r, folds in enumerate(fold_columns):
        these = fold_pairs(folds, scheme)
        pairs.extend(these)
        repeat_of.extend([r] * len(these))
    return pairs, repeat_of


@dataclass
class FoldPrediction:
    """One fold's scored rows and what the fold's model predicted for them."""

    rows: np.ndarray  # positions of the scored rows among the training rows
    y: np.ndarray
    prediction: np.ndarray  # ŷ (regression) or the class-probability matrix
    reference: float | None  # regression: the mean of the rows the fold's model was fit on
    classes: list[Any] | None = None


@dataclass
class CrossValidated:
    per_fold: list[dict[str, float]]
    parts: list[FoldPart] = field(default_factory=list)  # regression only
    sizes: list[tuple[int, int]] = field(default_factory=list)  # (rows fit, rows scored) per fold
    predictions: list[FoldPrediction] = field(default_factory=list)
    repeat_of: list[int] = field(default_factory=list)  # each fold's repeat; empty: one run

    def repeats(self) -> list[int]:
        return sorted(set(self.repeat_of)) if self.repeat_of else [0]

    def _repeat(self) -> list[int]:
        return self.repeat_of if self.repeat_of else [0] * len(self.per_fold)

    def summary(self, task: Task, *, groups: Any = None, unit: str | None = None
                ) -> dict[str, dict[str, Any]]:
        """Each metric's cross-validated estimate (:func:`summarize`), with its standard error
        when the folds' predictions were kept. ``groups``: each training row's unit."""
        errors = self.standard_errors(task, groups=groups, unit=unit) if self.predictions else None
        return summarize(task, self.per_fold, self.parts, repeat_of=self._repeat(), errors=errors)

    def standard_errors(self, task: Task, *, groups: Any = None, unit: str | None = None
                        ) -> dict[str, float | None]:
        """Each metric's SE (module docstring): per repeat, then the root mean square."""
        from turbotab.core.models import performance as perf

        units = None if groups is None else np.asarray(groups, dtype=object)
        by_repeat: dict[str, list[float]] = {m: [] for m in METRICS[task]}
        repeat_of = self._repeat()
        for r in self.repeats():
            folds = [p for p, rr in zip(self.predictions, repeat_of) if rr == r]
            if not folds:
                continue
            if task == "regression":
                e2, d2, ae, rows = [], [], [], []
                for f in folds:
                    a, b, c = perf.regression_parts(f.y, f.prediction, f.reference)
                    e2.append(a)
                    d2.append(b)
                    ae.append(c)
                    rows.append(f.rows)
                at = np.concatenate(rows)
                codes = None if units is None else perf.unit_codes(units[at], len(at))
                found = perf.regression_intervals(np.concatenate(e2), np.concatenate(d2),
                                                  np.concatenate(ae), codes, "")
                for m in METRICS[task]:
                    if found.get(m) is not None and found[m].se is not None:
                        by_repeat[m].append(found[m].se ** 2)
                continue
            for m in METRICS[task]:
                variances: list[float] | None = []
                for f in folds:
                    v = perf.fold_variance(task, m, f.y, f.prediction, classes=f.classes or [],
                                           groups=None if units is None else units[f.rows])
                    if v is None:
                        variances = None
                        break
                    variances.append(v)
                if variances:
                    by_repeat[m].append(float(sum(variances)) / len(variances) ** 2)
        return {m: (float(np.sqrt(np.mean(v))) if v else None) for m, v in by_repeat.items()}

    def out_of_fold(self, repeat: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(row positions, y, prediction) over every scored row of one repeat, in row order."""
        repeat_of = self._repeat()
        folds = [p for p, rr in zip(self.predictions, repeat_of) if rr == repeat]
        if not folds:
            return np.zeros(0, dtype=np.int64), np.zeros(0), np.zeros(0)
        rows = np.concatenate([f.rows for f in folds])
        y = np.concatenate([f.y for f in folds])
        pred = np.concatenate([f.prediction for f in folds])
        order = np.argsort(rows, kind="stable")
        return rows[order], y[order], pred[order]

    @property
    def test_share(self) -> float | None:
        """The mean of rows scored over rows fit, per fold (Nadeau & Bengio's n₂/n₁)."""
        shares = [s / f for f, s in self.sizes if f]
        return float(np.mean(shares)) if shares else None


def _rows(data: Any, mask: np.ndarray) -> Any:
    import pandas as pd

    if isinstance(data, (pd.DataFrame, pd.Series)):
        return data.iloc[np.flatnonzero(mask)]
    return np.asarray(data)[mask]


CLASSIFIED = ("binary", "multiclass", "ordinal")  # tasks whose models predict class probabilities


def predict(task: Task, model: Any, X: Any) -> np.ndarray:
    """ŷ for regression, the risk score for a time-to-event outcome (WP12b), and the
    class-probability matrix (columns in ``model.classes_``) otherwise."""
    if task in ("regression", "time_to_event"):
        return np.asarray(model.predict(X), dtype=float)
    return np.asarray(model.predict_proba(X), dtype=float)


def classes_of(task: Task, model: Any) -> list[Any] | None:
    """A fitted model's classes when the task has them; None for a number or a time to event."""
    return list(model.classes_) if task in CLASSIFIED else None


def cross_validate(task: Task, make: Callable[[], Any], X: Any, y: Any,
                   pairs: Sequence[tuple[int, np.ndarray, np.ndarray]], *,
                   fit: Callable[[Any, Any, Any, np.ndarray], Any] | None = None,
                   before_fold: Callable[[int, int], None] | None = None,
                   repeat_of: Sequence[int] | None = None,
                   keep_predictions: bool = False) -> CrossValidated:
    """Fit a fresh model (``make()``) on each pair's fit rows and score it on its scored rows.

    ``fit(model, X_fit, y_fit, fit_mask)`` fits a model (default: ``model.fit``); the fit stage
    passes one that draws the model's inner splits as the outer folds are. ``before_fold(i, n)``
    runs before each fold (progress, cancellation). ``repeat_of`` names each pair's repeat
    (:func:`repeated_pairs`). ``keep_predictions`` keeps every fold's predictions, for the
    standard errors and the out-of-fold calibration.
    """
    y = np.asarray(y)
    out = CrossValidated(per_fold=[], repeat_of=list(repeat_of) if repeat_of is not None else [])
    for i, (_, fit_rows, test_rows) in enumerate(pairs):
        if before_fold is not None:
            before_fold(i, len(pairs))
        X_fit, y_fit = _rows(X, fit_rows), y[fit_rows]
        X_test, y_test = _rows(X, test_rows), y[test_rows]
        model = make()
        model = fit(model, X_fit, y_fit, fit_rows) if fit is not None else model.fit(X_fit, y_fit)
        reference = float(np.mean(y_fit.astype(float))) if task == "regression" else None
        out.per_fold.append(score(task, model, X_test, y_test, reference=reference))
        if task == "regression":
            out.parts.append(fold_part(model, X_test, y_test, reference))
        out.sizes.append((int(fit_rows.sum()), int(test_rows.sum())))
        if keep_predictions:
            classes = classes_of(task, model)
            out.predictions.append(FoldPrediction(
                rows=np.flatnonzero(test_rows), y=y_test, prediction=predict(task, model, X_test),
                reference=reference, classes=classes))
    return out


__all__ = ["CV_DEFINITION", "CrossValidated", "FoldPart", "FoldPrediction", "LABELS",
           "CLASSIFIED", "LOWER_IS_BETTER", "METRICS", "POOLED", "PRIMARY", "SE_DEFINITION",
           "classes_of", "concordance",
           "cross_validate", "fold_part", "fold_pairs", "higher_is_better", "metric_labels",
           "ordinal_scores", "pooled", "predict", "r2_against", "repeated_pairs", "score",
           "summarize"]
