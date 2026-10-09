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
was seen may report its own corrected score. The fit stage cannot know that (it reads the answers,
not what was shown), so the record decides it: every fit served to a client notes, beside the
project, the families whose cross-validated scores it showed for its outcome
(:func:`note_seen`, :data:`SEEN_FILE`), and the served fit's result is checked against that
(:func:`vouch`). The only family fitted is declared before any score was seen when no other
family's score was ever shown for the outcome; then, and only then, its own score is the result.
Otherwise nothing is declared, with the exit that fits the compared families together.

**The winner's own score as "the result" is refused** (§4), with every back door: a family whose
score was shown for this outcome stays among the families fitted while no rows are held out, so
dropping it (:func:`_compared_families_stay`), undoing the selection that fitted it, or holding no
rows out after it was dropped under a holdout is refused, with the exit that keeps it. The scores
shown are never forgotten: a revert cannot unsee them, and a new seed, a new fold count or the
nested cross-validation offer draws the same rows (no rows held out means every row is scored),
so none of them starts the comparison afresh. A new outcome does. BBC-CV resamples whole units
when rows repeat (MODELING_SEQUENCE §2), so no unit's rows sit both in a resample and among the
rows it is scored on.

Importing this module registers the ``open_seal`` validator and completion, and the
``select_models``, ``set_split`` and ``revert`` validators.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict

from turbotab.core import decisions
from turbotab.core.decisions import OpenSeal, Refusal, SelectModels, SplitSpec

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
    # The family as the record names it ("Logistic regression"), and the performance clause: what
    # the served fit restates once the record has said whether the family was declared before any
    # score was seen (:func:`vouch`).
    name: str | None = None
    said: str | None = None
    # Whether the record says the only family fitted was declared before any score was seen: None
    # until the served fit checks it (the fit stage cannot know), then True or False.
    vouched: bool | None = None
    # Nothing declared because families whose scores were shown are not fitted now: the decision
    # that fits them together ({"label", "decision"}).
    exit: dict[str, Any] | None = None


def family_name(key: str, task: str | None, label: str | None = None) -> str:
    """A family as the record's sentences name it, capitalized to open a sentence: the linear family
    by its task ("Logistic regression" for a yes/no outcome), every other by the record's word."""
    from turbotab.core import voice

    try:
        said = voice._family_label(key, task, {})
    except Exception:  # noqa: BLE001 - a name never fails a fit; the family's own label stands
        said = None
    said = str(said or label or key).strip("`")
    return said[0].upper() + said[1:] if said else key


def _only_family(name: str) -> str:
    """How the result names the only family fitted, before the record has vouched for it."""
    return f"{name} is the only family fitted, so its own score is the result"


def _declared_first(name: str) -> str:
    """... and once the record says no other family's score was shown for the outcome."""
    return (f"{name} was the only family fitted, declared before any score was seen, so its own "
            f"score is the result")


SELECTION_NOT_NESTED = ("nested cross-validation widened each family's own interval, but none exists "
                        "for a choice among families")


def declared_result(task: str, metric: str, *, n_holdout: int, families: Sequence[str],
                    labels: Mapping[str, str], summaries: Mapping[str, Any],
                    selection: Mapping[str, Any] | None, optimism: Mapping[str, Any] | None = None,
                    narrow: str | None = None, nested: Mapping[str, Any] | None = None,
                    how_cv: str = "cross-validation", nested_ran: bool = False,
                    by_row: bool = False) -> DeclaredResult:
    """The declared result (MODELING_SEQUENCE §1 row 12):

    * held-out rows drawn: the held-out score of the family declared before they are opened
      (``open_seal``; :func:`mark_final`);
    * one family: its own score (bootstrap optimism-corrected when the split asked for it, else
      cross-validated), or the nested cross-validation interval when it ran. The served fit then
      says whether the record shows it declared before any score was seen, or declares nothing
      (:func:`vouch`);
    * several families and no held-out rows: the selection-corrected estimate (BBC-CV). The
      winner's own score as the result is refused (MODELING_SEQUENCE §4); with no corrected
      estimate nothing is declared. At p ≫ n its interval keeps the label "likely too narrow" even
      after nested cross-validation ran (``nested_ran``), which widens each family's own interval
      and no choice among them, and says so.

    ``labels`` are the families' own labels; the sentence names each as the record does
    (:func:`family_name`). ``by_row``: the folds could not keep a unit's rows together, so BBC-CV
    resampled rows and the estimate is within-unit performance (``validation.unit_spans``);
    ``how_cv`` says it of a family's own score.
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
        name = family_name(key, task, labels.get(key))

        def own(estimate: Any, said: str, *, how: str, low: Any = None, high: Any = None,
                said_narrow: str | None = None) -> DeclaredResult:
            return DeclaredResult(basis="own_score", family=key, metric=metric, estimate=estimate,
                                  ci_low=low, ci_high=high, how=how, narrow=said_narrow, name=name,
                                  said=said, sentence=f"{_only_family(name)}: {said}")

        if nested and nested.get("estimate") is not None:
            how = (f"nested cross-validation ({nested['reps']} repetitions of {nested['folds']} "
                   f"folds; Bates, Hastie & Tibshirani 2023)")
            said = performance_sentence(label, nested["estimate"], nested.get("ci_low"),
                                        nested.get("ci_high"), how=how)
            return own(nested["estimate"], said, how=how, low=nested.get("ci_low"),
                       high=nested.get("ci_high"))
        corrected = ((optimism or {}).get("estimates") or {}).get(metric) or {}
        if corrected.get("corrected") is not None and not (optimism or {}).get("refused"):
            how = (f"Harrell's bootstrap optimism correction ({optimism['n_ok']:,} resamples)")
            said = performance_sentence(f"optimism-corrected {label}", corrected["corrected"],
                                        None, None, how=how)
            return own(corrected["corrected"], said, how=how)
        entry = (summaries.get(key) or {}).get(metric) or {}
        said = performance_sentence(f"cross-validated {label}", entry.get("estimate"),
                                    entry.get("ci_low"), entry.get("ci_high"), how=how_cv,
                                    narrow=narrow)
        return own(entry.get("estimate"), said, how=how_cv, low=entry.get("ci_low"),
                   high=entry.get("ci_high"), said_narrow=narrow)
    if not selection:
        return DeclaredResult(basis="not_declared", family=None, metric=metric, estimate=None,
                              sentence=(f"No selection-corrected estimate could be computed, so no "
                                        f"result is declared: the best of {len(families)} families' "
                                        f"own {label} would flatter the choice."))
    best = str(selection["best"])
    how = (f"bootstrap bias-corrected cross-validation over {len(families)} families "
           f"(Tsamardinos et al. 2018; {selection['replicates']:,} resamples"
           + (" by whole unit" if selection.get("by_unit") else "") + ")"
           + (" over rows, not units (within-unit performance)" if by_row else ""))
    if narrow and nested_ran:
        narrow = f"{narrow}; {SELECTION_NOT_NESTED}"
    said = performance_sentence(f"selection-corrected {label}", selection["corrected"],
                                selection["corrected_low"], selection["corrected_high"], how=how,
                                narrow=narrow)
    name = family_name(best, task, labels.get(best))
    return DeclaredResult(
        basis="selection_corrected", family=best, metric=metric, estimate=selection["corrected"],
        ci_low=selection["corrected_low"], ci_high=selection["corrected_high"], how=how,
        narrow=narrow, name=name, said=said,
        sentence=(f"{name} was chosen among {len(families)} families on cross-validation with no "
                  f"rows held out, so the result is the selection-corrected estimate, not its own "
                  f"score: {said}"))


# ── the scores a client was shown, and the result the record vouches for (MS6) ─

SEEN_FILE = "scores_seen.json"  # beside the project: {"targets": {outcome: [family, …]}}
_SEEN_LOCK = threading.Lock()


def scored_in(fit: Any) -> list[str]:
    """The families whose cross-validated scores a served fit shows."""
    if not isinstance(fit, Mapping):
        return []
    return [str(m["family"]) for m in fit.get("models") or []
            if isinstance(m, Mapping) and m.get("family") and m.get("cv")]


def explained_in(explanation: Any) -> list[str]:
    """The families whose cross-validated scores a served explanation shows: each explained
    family's floor reads its score against the no-predictor baseline's (``models.explain.floor_of``),
    so an explanation served before the fit shows the comparison as much as the fit does."""
    if not isinstance(explanation, Mapping):
        return []
    out = []
    for f in explanation.get("families") or []:
        floor = f.get("floor") if isinstance(f, Mapping) else None
        scored = isinstance(floor, Mapping) and (floor.get("model") is not None
                                                 or floor.get("verdict") not in (None, "unscored"))
        if scored and f.get("family"):
            out.append(str(f["family"]))
    return out


def read_seen(project_dir: str | os.PathLike[str] | None) -> dict[str, list[str]]:
    """Each outcome's families whose cross-validated scores a client was served ({} for none)."""
    if not project_dir:
        return {}
    try:
        data = json.loads((Path(project_dir) / SEEN_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    targets = data.get("targets") if isinstance(data, Mapping) else None
    return {str(k): [str(f) for f in v] for k, v in (targets or {}).items() if isinstance(v, list)}


def note_seen(project_dir: str | os.PathLike[str] | None, target: str | None,
              families: Sequence[str]) -> None:
    """Keep beside the project that ``families``' scores for ``target`` were served (append-only:
    nothing undoes a score seen)."""
    if not project_dir or not target or not families:
        return
    folder = Path(project_dir)
    with _SEEN_LOCK:
        seen = read_seen(folder)
        known = seen.get(target, [])
        new = [f for f in families if f not in known]
        if not new:
            return
        seen[target] = [*known, *new]
        handle, tmp = tempfile.mkstemp(dir=folder, prefix=".seen-", suffix=".json")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as out:
                json.dump({"targets": seen}, out, ensure_ascii=False, indent=1)
            os.replace(tmp, folder / SEEN_FILE)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise


def _names(keys: Sequence[str]) -> str:
    return ", ".join(f"`{k}`" for k in keys)


def vouch(out: dict[str, Any], seen: Sequence[str], target: str | None) -> dict[str, Any]:
    """The served fit's declared result, checked against the record of the scores shown for its
    outcome (``seen``, :func:`read_seen`); ``out`` is changed in place and returned.

    With every family whose score was shown among the families fitted, the result stands, and the
    only family fitted is said to be declared before any score was seen. Otherwise nothing is
    declared (MODELING_SEQUENCE §4: the winner's own score is refused, and a selection-corrected
    estimate that leaves a compared family out under-corrects the choice), with the exit that fits
    them together, and the chain says why.
    """
    result = out.get("result")
    if not isinstance(result, dict) or result.get("basis") not in ("own_score", "selection_corrected"):
        return out
    fitted = scored_in(out)
    missing = [k for k in seen if k not in fitted]
    if not missing:
        if result["basis"] == "own_score" and result.get("name") and result.get("said"):
            out["result"] = {**result, "vouched": True,
                             "sentence": f"{_declared_first(result['name'])}: {result['said']}"}
        return out
    outcome = f" for `{target}`" if target else ""
    names = _names(missing)
    if result["basis"] == "own_score":
        why = (f"{result.get('name') or result.get('family')} was fitted alone after the "
               f"cross-validated scores of {names} were shown{outcome} with no rows held out, so it "
               f"was not declared before any score was seen: its own score would flatter the "
               f"choice (Tsamardinos et al. 2018), and no result is declared. Fit them together: "
               f"the result is then the selection-corrected estimate.")
    else:
        why = (f"The cross-validated scores of {names} were also shown{outcome} with no rows held "
               f"out, but they are not among the {len(fitted)} families fitted now, so the "
               f"selection-corrected estimate would leave part of the choice out (Tsamardinos et "
               f"al. 2018), and no result is declared. Fit them together: the result is then "
               f"corrected for every family compared.")
    keep = [*fitted, *missing]
    out["result"] = {**result, "basis": "not_declared", "estimate": None, "ci_low": None,
                     "ci_high": None, "vouched": False, "sentence": why,
                     "exit": {"label": "Fit the compared families together",
                              "decision": {"kind": "select_models", "models": keep}}}
    chain = [c for c in out.get("chain") or [] if c.get("relation") != "no_holdout_declares_corrected"]
    chain.append({"relation": "no_holdout_declares_corrected",
                  "because": f"the scores of {names} were shown{outcome} with no rows held out",
                  "then": ("no result is declared: the winner's own score is refused, and fitting "
                           "the compared families together declares the selection-corrected "
                           "estimate")})
    out["chain"] = chain
    return out


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


# ── the winner's own score as the result, by any back door (MS6) ─────────────


def seen_for(ctx: Any, target: str | None) -> list[str]:
    """The families whose cross-validated scores a client was served for ``target``: from the
    context's ``seen`` (an outcome → families mapping, or a callable returning one) when given,
    else the record beside the project (``project_dir``, :func:`read_seen`)."""
    if not target:
        return []
    given = _ctx(ctx, "seen")
    if callable(given):
        try:
            given = given()
        except Exception:  # noqa: BLE001 - nothing readable: the project's record is read instead
            given = None
    if isinstance(given, Mapping):
        return [str(k) for k in given.get(target) or []]
    return read_seen(_ctx(ctx, "project_dir")).get(target, [])


def compared_families(state: Any, seen: Sequence[str]) -> list[str]:
    """The families whose scores were shown for the outcome (``seen``) that ``state`` leaves out of
    the families fitted, under prediction with no rows held out: each would make the remaining
    family's own score the result. Empty under inference, with rows held out (the held-out rows
    score the family declared before they are opened), or with no family selected."""
    if state is None or getattr(state, "purpose", None) == "inference":
        return []
    split = getattr(state, "split", None)
    if split is not None and float(getattr(split, "holdout", 0) or 0) > 0:
        return []
    models = list(getattr(state, "models", None) or [])
    if not models:
        return []
    return [k for k in seen if k not in models]


def _probe(records: Sequence[Any], decision: Any) -> list[Any]:
    from datetime import datetime, timezone

    seq = max((r.seq for r in records), default=0) + 1
    return [*records, decisions.DecisionRecord(id="__probe__", seq=seq,
                                               at=datetime.now(timezone.utc), decision=decision)]


def _after(decision: Any, state: Any, ctx: Any) -> Any:
    """The state this answer would leave (None: the log refuses it itself, or cannot be read)."""
    if decision.kind == "select_models":
        return state.model_copy(update={"models": list(decision.models)})
    if decision.kind == "set_split":
        return state.model_copy(update={"split": SplitSpec(**decision.model_dump(exclude={"kind"}))})
    records = _ctx(ctx, "records")
    if callable(records):
        try:
            records = records()
        except Exception:  # noqa: BLE001 - no record: nothing to undo
            return None
    if not records:
        return None
    try:
        return decisions.fold(_probe(list(records), decision))
    except Refusal:
        return None  # the log refuses an unknown revert with its own reason


def _compared_families_stay(decision: Any, ctx: Any) -> None:
    """MODELING_SEQUENCE §4: "Winner's own corrected score as 'the result' (no holdout): refuse;
    report the selection-corrected estimate". Under prediction with no rows held out, a family whose
    score was shown for this outcome stays among the families fitted, whatever answer would leave
    it out: a selection that drops it, a revert of the selection that fitted it (a revert cannot
    unsee a score), or no rows held out after it was dropped under a holdout. Each is refused with
    the exit that keeps the compared families (the result is then corrected for the choice). A new
    seed, fold count or nested cross-validation draws the same rows, so it starts nothing afresh."""
    state = _ctx(ctx, "state")
    if state is None:
        return
    after = _after(decision, state, ctx)
    if after is None:
        return
    target = getattr(after, "target", None)
    seen = seen_for(ctx, target)
    left = compared_families(after, seen)
    if not left:
        return
    if decision.kind != "select_models":
        now = compared_families(state, seen if getattr(state, "target", None) == target
                                else seen_for(ctx, getattr(state, "target", None)))
        if set(left) <= set(now):
            return  # this answer leaves out nothing the state did not already leave out
    keep = [*after.models, *left]
    names = _names(left)
    outcome = f" for `{target}`" if target else ""
    shown = f"These families' cross-validated scores were shown{outcome} with no rows held out: {names}."
    flatters = "which flatters the choice (Tsamardinos et al. 2018)"
    if decision.kind == "select_models":
        said = (f"{shown} Dropping them would make the remaining family's own score the result, "
                f"{flatters}. Keep them: the result is then the selection-corrected estimate, and "
                f"the best family is still the one deployed.")
    elif decision.kind == "set_split":
        said = (f"{shown} They are not among the families selected now, so with no rows held out "
                f"the remaining family's own score would be the result, {flatters}. Keep the "
                f"compared families first: the result is then the selection-corrected estimate.")
    else:
        said = (f"{shown} Undoing that answer would leave them out of the families fitted, so the "
                f"remaining family's own score would be the result, {flatters}; a revert cannot "
                f"unsee a score. Keep the compared families: the result is then the "
                f"selection-corrected estimate.")
    raise Refusal("compared_families_stay", said,
                  exits=[{"label": "Keep the compared families",
                          "decision": SelectModels(models=keep)}])


decisions.register_validator("select_models", _compared_families_stay)
decisions.register_validator("set_split", _compared_families_stay)
decisions.register_validator("revert", _compared_families_stay)


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


# ── the shelf at this size: Riley's minimum first (MODELING_SEQUENCE §1 row 9; EXPLORE) ──────────
#
# The prediction review: "No validated rule predicts which family will perform best. The evidence is
# about data hunger and stability". Riley's minimum sample size (``models/sample_size.py``; TRIPOD+AI
# item 10) is computed for the candidate predictor parameters before the shelf is ranked. Below it
# the regression families rank first and every flexible learner after them, with the stated reason:
# van der Ploeg, Austin & Steyerberg (BMC Med Res Methodol 2014;14:137) found that support vector
# machines, neural networks and random forests "may need over 10 times as many events per variable
# to achieve a stable AUC and a small optimism" as logistic regression. A flexible learner is one
# that declares ``flexible`` (``models/base.py``; MODEL_FAMILY_CONTRACT C11).

FLEXIBLE_REASON = ("below Riley et al.'s minimum sample size, a flexible learner may need over 10 "
                   "times as many events per variable as a regression to reach a stable score "
                   "(van der Ploeg, Austin & Steyerberg 2014), so the regression families rank "
                   "first")


class SampleSizeCriterion(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_serialization_defaults_required=True)

    key: str
    what: str
    n: int


class SampleSize(BaseModel):
    """Riley et al.'s minimum sample size, computed before the shelf is ranked (TRIPOD+AI 10)."""

    model_config = ConfigDict(extra="forbid", json_schema_serialization_defaults_required=True)

    minimum: int | None
    n_rows: int
    parameters: int
    criteria: list[SampleSizeCriterion] = []
    binding: str | None = None
    below: bool = False
    r2_cs: float | None = None
    sentence: str


def is_flexible(family: Any) -> bool:
    """A flexible learner (the comment above), as the family declares it."""
    return bool(family.flexible)


def shelf_order(ranked: Sequence[tuple[Any, Any]], situation: Any
                ) -> tuple[list[tuple[Any, Any]], SampleSize | None]:
    """The shelf under prediction with Riley's minimum computed first (the comment above): below it
    the regression families first and the flexible learners after them, each flexible family
    carrying the reason; otherwise the order as ranked. Under inference the shelf is unchanged and
    no minimum is computed (its own per-family concerns judge the coefficients)."""
    from turbotab.core.models.base import Assessment
    from turbotab.core.models.sample_size import prediction_minimum

    ranked = list(ranked)
    if getattr(situation, "purpose", None) == "inference":
        return ranked, None
    n = int(situation.n_rows)
    parameters = int(situation.n_parameters if situation.n_parameters is not None
                     else situation.n_features)
    try:
        minimum = prediction_minimum(situation.task, parameters, n_rows=n,
                                     n_events=situation.n_events,
                                     outcome_mean=situation.outcome_mean,
                                     outcome_sd=situation.outcome_sd)
    except ValueError:
        minimum = None
    if minimum is None:
        said = (f"Riley et al.'s minimum sample size is not computed for a "
                f"{str(situation.task).replace('_', '-')} outcome here, so the shelf keeps its "
                f"order by the families' own judgments." if parameters >= 1 else
                "No candidate predictor, so no minimum sample size applies.")
        return ranked, SampleSize(minimum=None, n_rows=n, parameters=parameters, sentence=said)
    below = n < minimum.n
    criteria = [SampleSizeCriterion(key=c.key, what=c.what, n=c.n) for c in minimum.criteria]
    head = (f"Riley et al.'s minimum sample size for {parameters:,} candidate predictor "
            f"parameters is {minimum.n:,} rows (the binding criterion: {minimum.binding.what})")
    if below:
        flexible = [(f, a) for f, a in ranked if is_flexible(f)]
        regression = [(f, a) for f, a in ranked if not is_flexible(f)]
        moved = [(f, Assessment(a.score, a.fit, (f"{n:,} rows: {FLEXIBLE_REASON}.", *a.concerns)))
                 for f, a in flexible]
        ranked = regression + moved
        said = (f"{head}; these {n:,} rows fall short of it, so {FLEXIBLE_REASON}; fewer candidate "
                f"parameters (fewer knots, outcome-blind data reduction) would lower the minimum.")
    else:
        said = f"{head}; these {n:,} rows meet it."
    return ranked, SampleSize(minimum=minimum.n, n_rows=n, parameters=parameters,
                              criteria=criteria, binding=minimum.binding.key, below=below,
                              r2_cs=float(minimum.r2_cs), sentence=said)


# ── what the interpretable model costs or gains (MODELING_SEQUENCE §5; EXPLORE) ───────────────
#
# Renamed from "the price of explainability": Rudin (Nat Mach Intell 2019;1:206): "It is a myth that
# there is necessarily a trade-off between accuracy and interpretability", and "One can always create
# an artificial trade-off … by removing parts of a more complex model to reduce accuracy". So the
# interpretable side is a properly specified regression with splines on the same in-fold
# preprocessing (the benchmark the ``evaluation`` stage always fits), never a post-hoc explanation of
# the flexible model. The difference is signed so that a positive value favors the interpretable
# model: on the strictly proper primary, and on calibration as the distance of the calibration slope
# from 1. Picking "the best flexible family" flatters it, so the comparison is corrected by BBC-CV over
# the flexible set: each resample of the out-of-fold predictions (whole units together) chooses the
# best flexible family on the drawn rows and scores both on the rows left out. When the interval
# spans 0 the record says "no measurable cost on these rows".

NO_COST = "no measurable cost on these rows"


def _slope(task: str, y: np.ndarray, pred: np.ndarray, classes: Sequence[Any]) -> float | None:
    from turbotab.core.models.validation import _calibration_points

    try:
        found = _calibration_points(task, y, pred, list(classes) if classes else None)
    except Exception:  # noqa: BLE001 - a resample with one class has no slope
        return None
    value = found.get("calibration_slope")
    return float(value) if value is not None and math.isfinite(value) else None


def interpretable_cost(task: str, metric: str, oof: OutOfFold, interpretable: str,
                       flexible: Sequence[str], *, labels: Mapping[str, str] | None = None,
                       metric_label: str | None = None, replicates: int | None = None,
                       seed: int = SEED) -> dict[str, Any] | None:
    """The signed paired difference between the interpretable model and the best flexible family,
    corrected for choosing the best of the flexible set (the comment above). ``oof`` holds every
    family's out-of-fold predictions; draws are ``draw_units`` from
    ``numpy.random.default_rng(seed)``, as :func:`selection_optimism` draws them. None without a
    flexible family or with too few rows."""
    from turbotab.core.models.folds import draw_units, unit_rows

    flexible = [k for k in flexible if k in oof.predictions and k != interpretable]
    if interpretable not in oof.predictions or not flexible:
        return None
    lower = metric in LOWER_IS_BETTER
    keys = [interpretable, *flexible]
    ok = oof.scored.copy()
    for k in keys:
        values = oof.predictions[k]
        ok &= np.isfinite(values if values.ndim == 1 else values.sum(axis=1))
    rows = np.flatnonzero(ok)
    n = len(rows)
    if n < 10:
        return None
    codes, members = unit_rows(None if oof.units is None
                               else np.asarray(oof.units, dtype=object)[rows], n)
    U = len(members)
    B = replicates or (REPLICATES if n <= LARGE else REPLICATES_LARGE)
    rng = np.random.default_rng(seed)
    calibrated = task in ("regression", "binary")
    diffs: list[float] = []
    cal: list[float] = []
    wins = {k: 0 for k in flexible}
    for _ in range(B):
        draw = draw_units(rng, U)
        times = np.bincount(draw, minlength=U)[codes]
        if times.all():
            continue
        drawn = np.repeat(rows, times)
        out = rows[times == 0]
        inbag = {k: oof.score(metric, k, drawn) for k in flexible}
        inbag = {k: v for k, v in inbag.items() if math.isfinite(v)}
        if not inbag or len(out) < 2:
            continue
        chosen = (min if lower else max)(inbag, key=inbag.get)
        a, b = oof.score(metric, interpretable, out), oof.score(metric, chosen, out)
        if not (math.isfinite(a) and math.isfinite(b)):
            continue
        diffs.append(b - a if lower else a - b)
        wins[chosen] += 1
        if calibrated:
            y = oof.y[out]
            sa = _slope(task, y, oof.predictions[interpretable][out], oof.classes)
            sb = _slope(task, y, oof.predictions[chosen][out], oof.classes)
            if sa is not None and sb is not None:
                cal.append(abs(1.0 - sb) - abs(1.0 - sa))
    if len(diffs) < B / 2:
        return None
    names = labels or {}
    label = metric_label or metric

    def summary(values: Sequence[float]) -> dict[str, float | None]:
        if not values:
            return {"difference": None, "ci_low": None, "ci_high": None}
        return {"difference": float(np.mean(values)),
                "ci_low": float(np.percentile(values, 2.5)),
                "ci_high": float(np.percentile(values, 97.5))}

    score_part = summary(diffs)
    cal_part = summary(cal)

    def verdict(part: Mapping[str, float | None]) -> str | None:
        lo, hi = part["ci_low"], part["ci_high"]
        if lo is None or hi is None:
            return None
        if lo <= 0 <= hi:
            return NO_COST
        return "the interpretable model gains" if lo > 0 else "the interpretable model costs"

    who = names.get(interpretable, interpretable)
    flex = ", ".join(names.get(k, k) for k in flexible)

    def clause(what: str, part: Mapping[str, float | None]) -> str:
        said = verdict(part)
        return (f"on {what} the difference is {_num(part['difference'])} (95% interval "
                f"{_num(part['ci_low'])} to {_num(part['ci_high'])}): {said}")

    text = (f"What the interpretable model costs or gains: {who} against the best of {flex}, "
            f"chosen anew in each of {len(diffs):,} resamples of the out-of-fold predictions "
            f"(BBC-CV over the flexible set; positive favors {who}); "
            + clause(label, score_part)
            + (f"; {clause('calibration (the slope’s distance from 1)', cal_part)}"
               if cal_part["difference"] is not None else "") + ".")
    return {"interpretable": interpretable, "flexible": list(flexible), "metric": metric,
            "score": {**score_part, "verdict": verdict(score_part)},
            "calibration": ({**cal_part, "verdict": verdict(cal_part)}
                            if cal_part["difference"] is not None else None),
            "replicates": len(diffs), "wins": wins, "by_unit": oof.units is not None,
            "text": text}


__all__ = ["DeclaredResult", "FLEXIBLE_REASON", "LOWER_IS_BETTER", "METHOD", "NO_COST",
           "OutOfFold", "SEEN_FILE", "SELECTION_NOT_NESTED", "SampleSize", "SampleSizeCriterion",
           "compared_families", "declared_family", "declared_result", "explained_in", "family_name",
           "interpretable_cost", "is_flexible", "mark_final", "note_seen", "pooled_score",
           "read_seen", "scored_in", "seen_for", "selection_optimism", "shelf_order", "vouch"]
