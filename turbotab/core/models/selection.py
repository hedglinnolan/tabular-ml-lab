"""One declared model: choosing among families, and what the choice costs (AUDIT_REPORT §5 WP8: ME-13).

Two rules, one for each side of the seal.

**Before the seal is opened, the choice is made on cross-validation and its optimism is estimated.**
The family with the best cross-validated score was chosen *because* that score was best, so it
flatters the family (Varma & Simon 2006, *BMC Bioinformatics* 7:91: "Using CV to compute an error
estimate for a classifier that has itself been tuned using CV gives a significantly biased estimate
of the true error"). On null binary data the CV-best of the three families averaged an AUC of 0.536
against a true 0.50 (audit A7).

:func:`selection_optimism` estimates it by bootstrap bias-corrected cross-validation, BBC-CV
(Tsamardinos, Greasidou & Borboudakis 2018, *Mach Learn* 107:1895, algorithm 5): "BBC-CV's main
idea is to bootstrap the whole process of selecting the best-performing configuration on the
out-of-sample predictions of each configuration, without additional training of models." The
out-of-fold predictions of every family (:class:`OutOfFold`, the paper's matrix Π, kept as
cross-validation fits each fold) are resampled by row, B times; each time the family with the best
score on the resampled rows is chosen and scored on the rows the resample left out, and the mean of
those scores is the corrected estimate, with the 2.5th and 97.5th percentiles as its interval (the
paper's percentile interval, "accurate albeit somewhat conservative"). The optimism is the best
family's reported CV score less that estimate. Scores are computed on the pooled out-of-fold
predictions, as the paper computes them; for AUC that pools scores from different fold models,
which the paper notes must be "comparable (in the same scale)", true of the predicted probabilities
the families give. On the audit's null scenario (150 rows, ten noise predictors, three families,
five folds) it recovers the optimism (+0.039 measured, +0.036 estimated over 100 datasets), where
Tibshirani & Tibshirani's fold-wise correction (2009), tried first, recovered about two thirds of it
(+0.023 of +0.035 over 100 other datasets). With a real signal (a log-odds slope of 0.5 on one
predictor, 60 datasets) the corrected AUC averaged 0.567 against a true 0.580: conservative, as the
paper found, partly because each fold's model learns from four fifths of the rows.

**Opening the seal needs the final family declared** (:func:`_a_final_model_is_declared`), chosen
on cross-validation before any held-out score is seen; with one family fitted it is that one. Once
opened, the served fit marks the declared family ``final`` (its held-out score is the reported
result) and the others ``secondary`` (:func:`mark_final`): the best of several held-out scores,
picked after seeing them, is optimistic in turn (+0.023 AUC on 150-row holdouts, audit A7).

**With no held-out rows, the result is the selection-corrected estimate** (MODELING_SEQUENCE §1
row 12 (b), ruling 4, §4; MS6): :func:`declared_result`. Only a family declared before any score
was seen may report its own corrected score, and here that means the only family fitted. The
winner's own score as "the result" is refused, which covers its back door too: under prediction
with no rows held out, dropping families whose scores were compared would leave the winner alone
and its own score reported as the result, so :func:`_compared_families_stay` refuses it with the
exit that keeps them. BBC-CV resamples whole units when rows repeat (MODELING_SEQUENCE §2), so no
unit's rows sit both in a resample and among the rows it is scored on.

Importing this module registers the ``open_seal`` validator and completion, and the
``select_models`` validator.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict

from turbotab.core import decisions
from turbotab.core.decisions import OpenSeal, Refusal, SelectModels

from turbotab.core.models.metrics import LOWER_IS_BETTER  # noqa: E402 - one list for every score
METHOD = "bootstrap bias-corrected cross-validation (Tsamardinos et al. 2018)"
REPLICATES = 1000  # the paper's B
LARGE = 10_000  # above this many scored rows, fewer resamples: each costs more, the optimism is small
REPLICATES_LARGE = 200
SEED = 0


# ── the optimism of choosing on cross-validation ─────────────────────────────


def _rows(data: Any, mask: np.ndarray) -> Any:
    import pandas as pd

    if isinstance(data, (pd.DataFrame, pd.Series)):
        return data.iloc[np.flatnonzero(mask)]
    return np.asarray(data)[mask]


class OutOfFold:
    """Each family's out-of-fold predictions on the training rows (Tsamardinos et al.'s Π).

    :meth:`wrap` turns the fit stage's fit function into one that also predicts the rows its fold
    scores, for :func:`~turbotab.core.models.metrics.cross_validate`, which passes each pair's fit
    mask as it is. Regression keeps the predicted values and, for R², the mean of the rows each
    fold's model was fit on; classification keeps the predicted probabilities, in the order of the
    outcome's sorted classes. Rows no fold scores (the first block of time-ordered folds) stay
    missing and are never resampled.

    With repeated k-fold (WP9) the fit stage hands it the first repeat's pairs only, so every row
    is predicted once, as Π is defined; a wrapped fit of another repeat's fold is passed through
    without keeping its predictions.
    """

    def __init__(self, task: str, X: Any, y: Any, pairs: Sequence[tuple[int, np.ndarray, np.ndarray]],
                 *, units: Any = None, horizon: float | None = None):
        self.task = task
        self.X = X
        self.y = np.asarray(y)
        self.pairs = list(pairs)
        self.units = units  # each row's unit: the resamples take or leave a unit's rows together
        self.horizon = horizon  # a time to event's: its predictions are [risk score, risk by then]
        n = len(self.y)
        self.scored = np.zeros(n, dtype=bool)
        self.reference = np.full(n, np.nan)
        for _, fit_rows, test_rows in self.pairs:
            self.scored |= np.asarray(test_rows, dtype=bool)
            if task == "regression":
                self.reference[test_rows] = float(np.mean(self.y[fit_rows].astype(float)))
        self.classes: list[Any] = ([] if task in ("regression", "time_to_event")
                                   else np.unique(self.y).tolist())
        self.predictions: dict[str, np.ndarray] = {}

    def wrap(self, key: str, fit: Callable[..., Any]) -> Callable[..., Any]:
        def fit_and_keep(model: Any, X_fit: Any, y_fit: Any, rows: Any) -> Any:
            fitted = fit(model, X_fit, y_fit, rows)
            test_rows = self._scored_by(rows)
            if test_rows is not None:
                if self.task == "time_to_event" and getattr(fitted, "survival_baseline_", None) is None:
                    from turbotab.core.models.metrics import survival_baseline

                    survival_baseline(fitted, X_fit, y_fit)
                self.keep(key, fitted, test_rows)
            return fitted
        return fit_and_keep

    def _scored_by(self, fit_rows: Any) -> np.ndarray | None:
        """The rows the pair fit on ``fit_rows`` scores; None for a fold not among the pairs (another
        repeat's)."""
        for _, f, t in self.pairs:
            if f is fit_rows:
                return t
        for _, f, t in self.pairs:
            if np.array_equal(f, fit_rows):
                return t
        return None

    def keep(self, key: str, fitted: Any, test_rows: np.ndarray) -> None:
        """Store ``fitted``'s predictions for the rows ``test_rows`` marks."""
        X_test = _rows(self.X, test_rows)
        n = len(self.y)
        if self.task == "regression":
            out = self.predictions.setdefault(key, np.full(n, np.nan))
            out[test_rows] = np.asarray(fitted.predict(X_test), dtype=float)
            return
        if self.task == "time_to_event":
            from turbotab.core.models.metrics import predict

            out = self.predictions.setdefault(key, np.full((n, 2), np.nan))
            out[test_rows] = predict("time_to_event", fitted, X_test, horizon=self.horizon)
            return
        proba = np.asarray(fitted.predict_proba(X_test), dtype=float)
        block = np.zeros((proba.shape[0], len(self.classes)))
        for j, c in enumerate(fitted.classes_):
            block[:, self.classes.index(c)] = proba[:, j]
        out = self.predictions.setdefault(key, np.full((n, len(self.classes)), np.nan))
        out[test_rows] = block

    def score(self, metric: str, key: str, rows: np.ndarray) -> float:
        """``metric`` of family ``key`` on the out-of-fold predictions of ``rows`` (indices, with
        repeats), pooled as Tsamardinos et al. pool them; NaN where it is undefined."""
        return pooled_score(self.task, metric, self.y[rows], self.predictions[key][rows],
                            self.reference[rows], self.classes, horizon=self.horizon)


def _auc(positive: np.ndarray, p: np.ndarray) -> float:
    """The area under the ROC curve by the Mann–Whitney statistic, ties counted half (the trapezoid
    rule scikit-learn uses)."""
    from scipy.stats import rankdata

    n1 = int(positive.sum())
    n0 = len(positive) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = rankdata(p)
    return float((ranks[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def pooled_score(task: str, metric: str, y: np.ndarray, pred: np.ndarray, reference: np.ndarray,
                 classes: Sequence[Any], *, horizon: float | None = None) -> float:
    """A metric of ``models/metrics.py`` on pooled out-of-fold predictions."""
    from sklearn import metrics as m

    if task == "regression":
        y = y.astype(float)
        e = y - pred
        if metric == "r2":
            sst = float(((y - reference) ** 2).sum())
            return 1.0 - float((e ** 2).sum()) / sst if sst > 0 else float("nan")
        if metric == "mse":
            return float(np.mean(e ** 2))
        if metric == "rmse":
            return float(np.sqrt(np.mean(e ** 2)))
        if metric == "mae":
            return float(np.mean(np.abs(e)))
        raise KeyError(metric)
    if task == "binary":
        positive = y == classes[1]
        p = pred[:, 1]
        if metric == "auc":
            return _auc(positive, p)
        if metric == "brier":
            return float(np.mean((positive - p) ** 2))
        if metric == "log_loss":
            return float(m.log_loss(positive, p, labels=[False, True]))
        raise KeyError(metric)
    if task == "time_to_event":  # Harrell's C of the pooled risk scores (WP12b); the Brier score
        from turbotab.core.models.performance import brier_at  # at the horizon of their risks (MS6)
        from turbotab.core.models.survival import concordance

        if metric == "c_index":
            return float(concordance(y["time"], y["event"], pred[:, 0]))
        if metric == "brier_t":
            return brier_at(y, pred[:, 1], horizon)
        raise KeyError(metric)
    if task == "ordinal":  # WP12a: the ordered outcome's scores, on the pooled predictions
        from turbotab.core.models.metrics import ordinal_scores

        return float(ordinal_scores(y, pred, classes)[metric])
    labels = np.asarray(classes, dtype=object)[np.argmax(pred, axis=1)]
    if metric == "accuracy":
        return float(np.mean(labels == y))
    if metric == "macro_f1":
        return float(m.f1_score(y, labels, average="macro", labels=list(classes), zero_division=0))
    if metric == "log_loss":
        return float(m.log_loss(y, pred, labels=list(classes)))
    raise KeyError(metric)


def selection_optimism(task: str, metric: str, results: Mapping[str, Any], oof: OutOfFold,
                       labels: Mapping[str, str] | None = None, metric_label: str | None = None,
                       replicates: int | None = None, seed: int = SEED,
                       extras: Sequence[str] = ()) -> dict[str, Any] | None:
    """How much the best family's cross-validated ``metric`` flatters it, by BBC-CV over the
    families' out-of-fold predictions; None with fewer than two families or nothing to resample.

    ``results`` maps each family to its :class:`~turbotab.core.models.metrics.CrossValidated`.
    Each resample draws whole units (``oof.units``; every row its own unit without them) with
    ``numpy.random.default_rng(seed).integers(0, U, U)``: the family with the best ``metric`` on the
    drawn units' rows is chosen and scored on the rows of the units the draw left out. ``extras``
    are scored on the same chosen family and left-out rows (the customary headline, R²), so they are
    corrected for the same choice. Returns the ``selection`` artifact field: the best family, its CV
    score, the corrected estimate with its 95% percentile interval, the optimism (the CV score less
    the corrected estimate, in the score's units, signed so that positive flatters), how often each
    family won a resample, the extras' corrected values, and the sentence that states it.
    """
    from turbotab.core.models.folds import draw_units, unit_rows

    keys = [k for k in results if k in oof.predictions]
    if len(keys) < 2:
        return None
    estimates = {k: results[k].summary(task)[metric]["estimate"] for k in keys}
    if not all(e is not None and math.isfinite(e) for e in estimates.values()):
        return None
    lower = metric in LOWER_IS_BETTER

    def pick(scores: Mapping[str, float]) -> str:
        return (min if lower else max)(scores, key=scores.get)  # ties: the first family listed

    best = pick(estimates)
    ok = oof.scored.copy()
    for k in keys:
        values = oof.predictions[k]
        ok &= np.isfinite(values if values.ndim == 1 else values.sum(axis=1))
    rows = np.flatnonzero(ok)
    n = len(rows)
    if n < 2:
        return None
    # Each scored row's unit (0 … U − 1, in order of first appearance; every row its own unit
    # without ``oof.units``): a draw's count of each unit repeats that unit's rows as many times.
    codes, members = unit_rows(None if oof.units is None
                               else np.asarray(oof.units, dtype=object)[rows], n)
    U = len(members)
    B = replicates or (REPLICATES if n <= LARGE else REPLICATES_LARGE)
    rng = np.random.default_rng(seed)
    scores: list[float] = []
    extra_scores: dict[str, list[float]] = {m: [] for m in extras}
    wins = {k: 0 for k in keys}
    for _ in range(B):
        draw = draw_units(rng, U)
        times = np.bincount(draw, minlength=U)[codes]  # how often each scored row was drawn
        if times.all():
            continue
        drawn = np.repeat(rows, times)
        out = rows[times == 0]
        inbag = {k: oof.score(metric, k, drawn) for k in keys}
        inbag = {k: v for k, v in inbag.items() if math.isfinite(v)}
        if not inbag or not len(out):
            continue
        chosen = pick(inbag)
        value = oof.score(metric, chosen, out)
        if math.isfinite(value):
            scores.append(value)
            wins[chosen] += 1
            for m in extras:
                extra_scores[m].append(oof.score(m, chosen, out))
    if len(scores) < B / 2:
        return None
    corrected = float(np.mean(scores))
    low, high = (float(v) for v in np.percentile(scores, [2.5, 97.5]))
    cv = float(estimates[best])
    optimism = corrected - cv if lower else cv - corrected
    names = labels or {}
    label = metric_label or metric
    family = names.get(best, best)
    flatter = (f"{family}'s CV {label} of {_num(cv)} is {'low' if lower else 'high'} by about "
               f"{_num(optimism)}" if optimism > 0 else
               f"{family}'s CV {label} of {_num(cv)} shows no optimism from the choice")
    by = (f"{len(scores):,} resamples of the out-of-fold predictions"
          + (", by whole unit" if oof.units is not None else ""))
    text = (f"Choosing the best of {len(keys)} families by cross-validated {label} flatters the "
            f"winner: {flatter}; corrected for the choice, {_num(corrected)} (95% interval "
            f"{_num(low)} to {_num(high)}) is the expected performance of this modeling procedure "
            f"at this sample size ({METHOD}, {by}). The held-out score of a family declared "
            f"before the seal is opened carries no such optimism.")
    extra_out: dict[str, dict[str, float | None]] = {}
    for m, values in extra_scores.items():
        finite = [v for v in values if math.isfinite(v)]
        extra_out[m] = ({"corrected": float(np.mean(finite)),
                         "corrected_low": float(np.percentile(finite, 2.5)),
                         "corrected_high": float(np.percentile(finite, 97.5))}
                        if finite else {"corrected": None, "corrected_low": None,
                                        "corrected_high": None})
    return {"metric": metric, "families": keys, "best": best, "cv": cv, "optimism": optimism,
            "corrected": corrected, "corrected_low": low, "corrected_high": high,
            "replicates": len(scores), "wins": wins, "method": METHOD, "text": text,
            "by_unit": oof.units is not None, "extras": extra_out}


def _num(value: float) -> str:
    return f"{value:.3f}".replace("-", "−")


# ── the declared result (MODELING_SEQUENCE §1 row 12, ruling 4; MS6) ─────────


class DeclaredResult(BaseModel):
    """What the fit reports as the result, and on what basis (module docstring)."""

    model_config = ConfigDict(extra="forbid", json_schema_serialization_defaults_required=True)

    basis: Literal["holdout", "own_score", "selection_corrected", "not_declared"]
    family: str | None  # the family whose fit is deployed (the best by CV when chosen among several)
    metric: str
    estimate: float | None
    ci_low: float | None = None
    ci_high: float | None = None
    how: str | None = None  # how the estimate was made, in words
    narrow: str | None = None  # why its interval is labeled likely too narrow, when it is
    sentence: str


def declared_result(task: str, metric: str, *, n_holdout: int, families: Sequence[str],
                    labels: Mapping[str, str], summaries: Mapping[str, Any],
                    selection: Mapping[str, Any] | None, optimism: Mapping[str, Any] | None = None,
                    narrow: str | None = None, nested: Mapping[str, Any] | None = None,
                    how_cv: str = "cross-validation") -> DeclaredResult:
    """The declared result (MODELING_SEQUENCE §1 row 12):

    * held-out rows drawn: the held-out score of the family declared before they are opened
      (``open_seal``; :func:`mark_final`);
    * one family, declared before any score was seen: its own score (bootstrap optimism-corrected
      when the split asked for it, else cross-validated), or the nested cross-validation interval
      when it ran;
    * several families and no held-out rows: the selection-corrected estimate (BBC-CV). The
      winner's own score as the result is refused (MODELING_SEQUENCE §4); with no corrected
      estimate nothing is declared.
    """
    from turbotab.core.models.validation import performance_sentence, score_words

    label = score_words(metric)
    if n_holdout:
        return DeclaredResult(basis="holdout", family=None, metric=metric, estimate=None,
                              sentence=(f"The held-out rows score the final model, declared on "
                                        f"cross-validation before they are opened; its held-out "
                                        f"{label} will be the result."))
    if len(families) == 1:
        key = families[0]
        name = labels.get(key, key)
        if nested and nested.get("estimate") is not None:
            how = (f"nested cross-validation ({nested['reps']} repetitions of {nested['folds']} "
                   f"folds; Bates, Hastie & Tibshirani 2023)")
            said = performance_sentence(label, nested["estimate"], nested.get("ci_low"),
                                        nested.get("ci_high"), how=how)
            return DeclaredResult(basis="own_score", family=key, metric=metric,
                                  estimate=nested["estimate"], ci_low=nested.get("ci_low"),
                                  ci_high=nested.get("ci_high"), how=how,
                                  sentence=(f"{name} was the only family fitted, declared before any "
                                            f"score was seen, so its own score is the result: {said}"))
        corrected = ((optimism or {}).get("estimates") or {}).get(metric) or {}
        if corrected.get("corrected") is not None and not (optimism or {}).get("refused"):
            how = (f"Harrell's bootstrap optimism correction ({optimism['n_ok']:,} resamples)")
            said = performance_sentence(f"optimism-corrected {label}", corrected["corrected"],
                                        None, None, how=how)
            return DeclaredResult(basis="own_score", family=key, metric=metric,
                                  estimate=corrected["corrected"], how=how,
                                  sentence=(f"{name} was the only family fitted, declared before any "
                                            f"score was seen, so its own score is the result: {said}"))
        entry = (summaries.get(key) or {}).get(metric) or {}
        said = performance_sentence(f"cross-validated {label}", entry.get("estimate"),
                                    entry.get("ci_low"), entry.get("ci_high"), how=how_cv,
                                    narrow=narrow)
        return DeclaredResult(basis="own_score", family=key, metric=metric,
                              estimate=entry.get("estimate"), ci_low=entry.get("ci_low"),
                              ci_high=entry.get("ci_high"), how=how_cv, narrow=narrow,
                              sentence=(f"{name} was the only family fitted, declared before any "
                                        f"score was seen, so its own score is the result: {said}"))
    if not selection:
        return DeclaredResult(basis="not_declared", family=None, metric=metric, estimate=None,
                              sentence=(f"No selection-corrected estimate could be computed, so no "
                                        f"result is declared: the best of {len(families)} families' "
                                        f"own {label} would flatter the choice."))
    best = str(selection["best"])
    how = (f"bootstrap bias-corrected cross-validation over {len(families)} families "
           f"(Tsamardinos et al. 2018; {selection['replicates']:,} resamples"
           + (" by whole unit" if selection.get("by_unit") else "") + ")")
    said = performance_sentence(f"selection-corrected {label}", selection["corrected"],
                                selection["corrected_low"], selection["corrected_high"], how=how,
                                narrow=narrow)
    return DeclaredResult(
        basis="selection_corrected", family=best, metric=metric, estimate=selection["corrected"],
        ci_low=selection["corrected_low"], ci_high=selection["corrected_high"], how=how,
        narrow=narrow,
        sentence=(f"{labels.get(best, best)} was chosen among {len(families)} families on "
                  f"cross-validation with no rows held out, so the result is the "
                  f"selection-corrected estimate, not its own score: {said}"))


# ── declaring the final family before the seal is opened ─────────────────────


def _ctx(ctx: Any, name: str) -> Any:
    if ctx is None:
        return None
    if isinstance(ctx, Mapping):
        return ctx.get(name)
    return getattr(ctx, name, None)


def _openable_fit(ctx: Any) -> Mapping[str, Any] | None:
    """The fit the seal would open on, when the seal can be opened at all; else None (the seal's
    own validator says why it cannot)."""
    state = _ctx(ctx, "state")
    if state is None or getattr(state, "seal_opened", None):
        return None
    split = getattr(state, "split", None)
    if split is None or float(getattr(split, "holdout", 0) or 0) == 0:
        return None
    artifact = _ctx(ctx, "artifact")
    if not callable(artifact):
        return None
    try:
        fit = artifact("fit")
    except Exception:  # noqa: BLE001 - no fit: the seal's validator refuses
        return None
    if not isinstance(fit, Mapping) or not int(fit.get("n_holdout") or 0):
        return None
    return fit


def _cv_of(fit: Mapping[str, Any], model: Mapping[str, Any]) -> float | None:
    """A family's cross-validated primary as the families are compared on it: the comparison
    substrate's (MS6: every repeat of the repeated k-fold BBC-CV chooses on), else the headline's."""
    metric = fit.get("primary_metric")
    entry = model.get("compared_on") or (model.get("cv") or {}).get(metric) or {}
    value = entry.get("estimate", entry.get("mean"))
    return float(value) if value is not None and math.isfinite(float(value)) else None


def _a_final_model_is_declared(decision: Any, ctx: Any) -> None:
    fit = _openable_fit(ctx)
    if fit is None:
        return
    models = [m for m in fit.get("models") or [] if m.get("family")]
    fitted = [str(m["family"]) for m in models]
    metric = str(fit.get("primary_metric") or "")
    label = (fit.get("metric_labels") or {}).get(metric, metric)
    lower = metric in LOWER_IS_BETTER
    ranked = sorted(models, key=lambda m: (_cv_of(fit, m) is None,
                                           (1 if lower else -1) * (_cv_of(fit, m) or 0.0)))
    exits = []
    for i, m in enumerate(ranked):
        cv = _cv_of(fit, m)
        said = f" (CV {label} {_num(cv)}{', the best' if i == 0 and len(ranked) > 1 else ''})" \
            if cv is not None else ""
        exits.append({"label": f"Open with {m.get('label') or m['family']} as the final model{said}",
                      "decision": OpenSeal(family=str(m["family"]))})
    family = getattr(decision, "family", None)
    if family is not None and family not in fitted:
        raise Refusal("not_a_fitted_family",
                      f"`{family}` is not one of the fitted families, so it cannot be the final "
                      f"model.", exits=exits)
    if family is None and len(fitted) > 1:
        chosen = fit.get("selection") or {}
        cost = (f" Choosing on cross-validation flatters the best family's CV {label} too, by about "
                f"{_num(chosen['optimism'])} here (bootstrap bias-corrected)."
                if chosen.get("optimism") is not None and chosen["optimism"] > 0 else "")
        raise Refusal(
            "final_model_needed",
            f"Name the final model before the held-out rows are opened, choosing it on "
            f"cross-validation: its held-out {label} is then the result. The best of "
            f"{len(fitted)} held-out scores, picked after seeing them, would flatter itself.{cost}",
            exits=exits)


def _the_only_family_is_the_final_one(decision: Any, ctx: Any) -> Any:
    """With one family fitted, the opening declares it (nothing else could be the final model)."""
    if getattr(decision, "family", None) is not None:
        return decision
    fit = _openable_fit(ctx)
    if fit is None:
        return decision
    fitted = [str(m["family"]) for m in fit.get("models") or [] if m.get("family")]
    # Every other field the server fills (the seal's scores at the opening, audit WP16) is kept.
    return decision.model_copy(update={"family": fitted[0]}) if len(fitted) == 1 else decision


decisions.register_validator("open_seal", _a_final_model_is_declared)
decisions.register_completion("open_seal", _the_only_family_is_the_final_one)


# ── the winner's own score as the result, by dropping the others (MS6) ───────


def _live(records: Sequence[Any]) -> list[Any]:
    ordered = sorted(records, key=lambda r: r.seq)
    try:
        cancelled = decisions.reverted(ordered)
    except Refusal:
        return ordered
    return [r for r in ordered if r.id not in cancelled]


def compared_families(records: Sequence[Any] | None, shown: Any = None) -> list[str]:
    """The families whose cross-validated scores were compared on the current seal and outcome:
    those named together in a live ``select_models`` since the latest live ``set_target`` and
    ``set_split``, and, when the newest fit a client was shown is given, scored in it."""
    if not records:
        return []
    live = _live(records)
    since = max((r.seq for r in live if r.decision.kind in ("set_target", "set_split")), default=0)
    named: list[str] = []
    for r in live:
        if r.seq > since and r.decision.kind == "select_models" and len(r.decision.models) > 1:
            named += [k for k in r.decision.models if k not in named]
    if isinstance(shown, Mapping):
        scored = {str(m.get("family")) for m in shown.get("models") or [] if m.get("cv")}
        named = [k for k in named if k in scored]
    return named if len(named) > 1 else []


def _compared_families_stay(decision: Any, ctx: Any) -> None:
    """MODELING_SEQUENCE §4: "Winner's own corrected score as 'the result' (no holdout): refuse;
    report the selection-corrected estimate". Under prediction with no rows held out, dropping a
    family whose score was compared would report the survivors' own score as the result, so it is
    refused, with the exit that keeps the compared families (the result is then corrected for the
    choice)."""
    state = _ctx(ctx, "state")
    if state is None or getattr(state, "purpose", None) == "inference":
        return
    split = getattr(state, "split", None)
    if split is not None and float(getattr(split, "holdout", 0) or 0) > 0:
        return  # the held-out rows score the family declared before they are opened
    records = _ctx(ctx, "records")
    if callable(records):
        try:
            records = records()
        except Exception:  # noqa: BLE001 - no record: nothing was compared
            return
    shown = _ctx(ctx, "shown")
    if callable(shown):
        try:
            shown = shown("fit")
        except Exception:  # noqa: BLE001 - nothing shown: the record alone says what was compared
            shown = None
    compared = compared_families(records, shown)
    dropped = [k for k in compared if k not in decision.models]
    if not dropped:
        return
    keep = list(decision.models) + [k for k in compared if k not in decision.models]
    names = ", ".join(f"`{k}`" for k in dropped)
    raise Refusal(
        "compared_families_stay",
        f"These families' scores were compared on these rows with none held out: {names}. "
        f"Dropping them would make the remaining family's own score the result, which flatters "
        f"the choice (Tsamardinos et al. 2018). Keep them: the result is then the "
        f"selection-corrected estimate, and the best family is still the one deployed.",
        exits=[{"label": "Keep the compared families", "decision": SelectModels(models=keep)}])


decisions.register_validator("select_models", _compared_families_stay)


# ── the served fit: final and secondary ──────────────────────────────────────


def declared_family(records: Sequence[Any]) -> str | None:
    """The family the opening declared final, or None (not opened, or opened before a final model
    had to be declared)."""
    from turbotab.core.seal import opening

    record = opening(records)
    return getattr(record.decision, "family", None) if record is not None else None


def mark_final(out: dict[str, Any], *, opened: bool, family: str | None) -> dict[str, Any]:
    """Mark the served fit's declared final family and its secondary ones (in place; returned).

    Before the opening nothing is final: ``final_model`` and every ``role`` are null.
    """
    models = out.get("models") or []
    for m in models:
        m["role"] = None
    out["final_model"] = None
    out["final_note"] = None
    if not opened or not int(out.get("n_holdout") or 0):
        return out
    metric = str(out.get("primary_metric") or "")
    label = (out.get("metric_labels") or {}).get(metric, metric)
    if family is None:
        out["final_note"] = (f"The seal was opened before a final model had to be declared, so no "
                             f"held-out {label} is the reported result, and the best of them, "
                             f"picked after seeing them, is optimistic.")
        return out
    named = next((m for m in models if m.get("family") == family), None)
    if named is None:
        for m in models:
            m["role"] = "secondary"
        out["final_note"] = (f"`{family}` was declared the final model when the seal was opened, "
                             f"but it is not among the families fitted now, so every held-out "
                             f"score here is secondary.")
        return out
    out["final_model"] = family
    for m in models:
        m["role"] = "final" if m is named else "secondary"
    others = len(models) - 1
    rest = ("; the other family's held-out score is secondary" if others == 1 else
            f"; the other {others} families' held-out scores are secondary" if others else "")
    out["final_note"] = (f"{named.get('label') or family} was declared the final model on "
                         f"cross-validation before the held-out rows were opened, so its held-out "
                         f"{label} is the reported result{rest}.")
    return out


__all__ = ["DeclaredResult", "LOWER_IS_BETTER", "METHOD", "OutOfFold", "compared_families",
           "declared_family", "declared_result", "mark_final", "pooled_score", "selection_optimism"]
