"""Design-based cross-validation under the surveyed-population answer (MODELING_SEQUENCE §0 ruling
13; Wieczorek, Guerin & McMahon, *Stat* 2022;11:e454).

Under prediction with the population answer, the performance estimated is the population's, so the
cross-validation follows the design:

* **Folds keep whole PSUs together within strata** (:func:`design_folds`). Within each stratum its
  PSUs are shuffled and dealt to the folds in turn, so every fold holds PSUs from every stratum, as
  the sample itself does. Wieczorek et al.: with a stratified cluster sample, "each fold is formed by
  sampling clusters within each stratum". A fold must hold at least one PSU of every stratum, so the
  number of folds is the smaller of the folds asked for and the fewest PSUs any stratum holds
  (NHANES releases two masked PSUs per stratum, so two folds), and the draw is repeated to steady
  the estimate. Without a PSU column each row (or each unit) is its own PSU; without strata the
  design has one.
* **Every loss is survey-weighted.** A fold's score is the weighted mean of its rows' losses,
  Σ wᵢℓᵢ / Σ wᵢ over the fold, and the cross-validated estimate is the mean of the folds' scores over
  every fold of every repeat, as ``surveyCV::cv.svy`` averages the folds' design-based means. The
  pooled out-of-fold weighted mean of one repeat carries the design's linearized standard error,
  as ``survey::svymean`` computes it (PSUs with replacement within strata; module ``models/survey.py``).
* **Calibration is survey-weighted**: the weighted logistic recalibration of the out-of-fold
  log-odds (``survey::svyglm``'s point estimates are the weighted maximum likelihood), or the
  weighted least-squares slope of the outcome on the prediction.
* **Comparisons are survey-weighted**: each family against the baseline and against each other, by
  the corrected repeated k-fold t (Nadeau & Bengio 2003) over the paired weighted fold scores; the
  choice among families by bootstrap bias-corrected cross-validation (Tsamardinos et al. 2018) with
  PSUs resampled within strata and every score weighted.

The record labels these "design-based cross-validation". The fold's models are fitted as the
procedure fits them; only the scoring follows the design. Under the sample answer the scores stay
unweighted, labeled as the procedure's performance on these rows; under inference no
cross-validated score is shown (MODELING_SEQUENCE §1 row 11).
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

LABEL = "design-based cross-validation"
SOURCE = "Wieczorek, Guerin & McMahon, Stat 2022;11:e454"
REPEATS = 10
BBC_REPLICATES = 1000
LOWER_IS_BETTER_LOSSES = ("mse", "log_loss", "brier", "rps")


def codes(strata: Any, psu: Any, n: int) -> tuple[np.ndarray, np.ndarray]:
    """(stratum code, PSU code) per row: one stratum without strata; each row its own PSU without
    a PSU column; a PSU is its label within its stratum (``nest = TRUE``)."""
    s = np.zeros(n, dtype=np.int64) if strata is None else pd.factorize(
        pd.Series(np.asarray(strata, dtype=object)).astype(str), sort=True)[0]
    p = np.arange(n) if psu is None else pd.factorize(pd.MultiIndex.from_arrays(
        [s, pd.Series(np.asarray(psu, dtype=object)).astype(str).to_numpy()]), sort=True)[0]
    return np.asarray(s, dtype=np.int64), np.asarray(p, dtype=np.int64)


def design_folds(strata: Any, psu: Any, n: int, folds: int, seed: int = 0
                 ) -> tuple[np.ndarray, int, str | None]:
    """A fold per row (``0 … K − 1``): each stratum's PSUs (in order of their codes) permuted by
    ``numpy.random.default_rng(seed).permutation``, stratum by stratum in order, and dealt to the
    folds in turn (module docstring); K, and a note when it is fewer than ``folds``."""
    s, p = codes(strata, psu, n)
    psus_per_stratum = pd.Series(p).groupby(s).nunique()
    k = int(max(2, min(int(folds), int(psus_per_stratum.min()))))
    note = None
    if k < int(folds):
        note = (f"{k} folds, not {int(folds)}: the fewest PSUs in a stratum is "
                f"{int(psus_per_stratum.min())}, and every fold keeps a PSU of every stratum.")
    rng = np.random.default_rng(int(seed))
    fold_of_psu: dict[int, int] = {}
    for stratum in sorted(set(s.tolist())):
        units = np.unique(p[s == stratum])
        units = units[rng.permutation(len(units))]
        for i, u in enumerate(units):
            fold_of_psu[int(u)] = i % k
    return np.asarray([fold_of_psu[int(u)] for u in p], dtype=np.int64), k, note


def fold_pairs(fold: np.ndarray) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """(fold, fit mask, test mask) for each fold of one draw."""
    return [(int(f), fold != f, fold == f) for f in sorted(set(fold.tolist()))]


def weighted_mean(values: Any, weights: Any) -> float:
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    return float((w * v).sum() / w.sum())


def fold_scores(losses: np.ndarray, weights: np.ndarray, folds: Sequence[np.ndarray]) -> np.ndarray:
    """Each fold's weighted mean loss, for the test masks ``folds``."""
    return np.asarray([weighted_mean(losses[m], weights[m]) for m in folds], dtype=float)


def svymean_se(values: Any, weights: Any, strata: Any, psu: Any) -> float:
    """The linearized standard error of the weighted mean, PSUs with replacement within strata:
    ``survey::svymean`` with ``svydesign(ids = ~psu, strata = ~strata, weights = ~w, nest = TRUE)``
    over these rows (every row in the domain)."""
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    total = float(w.sum())
    mean = float((w * v).sum() / total)
    z = w * (v - mean) / total
    s, p = codes(strata, psu, len(v))
    frame = pd.DataFrame({"s": s, "p": p, "z": z})
    totals = frame.groupby(["s", "p"])["z"].sum().reset_index()
    var = 0.0
    for _, g in totals.groupby("s"):
        nh = len(g)
        if nh < 2:
            continue
        dev = g["z"].to_numpy() - g["z"].mean()
        var += nh / (nh - 1) * float((dev ** 2).sum())
    return math.sqrt(var)


def weighted_calibration(task: str, y: Any, prediction: Any, weights: Any,
                         classes: Sequence[Any] | None = None) -> dict[str, float | None]:
    """The weighted calibration intercept and slope (module docstring): a yes/no outcome's
    weighted logistic recalibration ``logit P(y) = a + b·logit p̂`` (slope) and with the slope fixed
    at 1 (intercept, calibration-in-the-large); a numeric outcome's weighted least squares of y on
    ŷ (slope) and the weighted mean of y − ŷ (intercept)."""
    w = np.asarray(weights, dtype=float)
    if task == "regression":
        yy = np.asarray(y, dtype=float)
        yhat = np.asarray(prediction, dtype=float)
        X = np.column_stack([np.ones(len(yy)), yhat])
        WX = X * w[:, None]
        beta = np.linalg.solve(X.T @ WX, WX.T @ yy)
        return {"intercept": weighted_mean(yy - yhat, w), "slope": float(beta[1])}
    P = np.asarray(prediction, dtype=float)
    event = (np.asarray(y) == list(classes or [0, 1])[1]).astype(float)
    lp = np.log(np.clip(P[:, 1], 1e-12, 1 - 1e-12) / np.clip(1 - P[:, 1], 1e-12, 1))
    slope = _weighted_logistic(np.column_stack([np.ones(len(lp)), lp]), event, w)
    intercept = _weighted_logistic(np.ones((len(lp), 1)), event, w, offset=lp)
    return {"intercept": None if intercept is None else float(intercept[0]),
            "slope": None if slope is None else float(slope[1])}


def _weighted_logistic(X: np.ndarray, y: np.ndarray, w: np.ndarray,
                       offset: np.ndarray | None = None) -> np.ndarray | None:
    off = np.zeros(len(y)) if offset is None else offset
    beta = np.zeros(X.shape[1])
    for _ in range(100):
        mu = 1.0 / (1.0 + np.exp(-(off + X @ beta)))
        info = X.T @ (X * (w * mu * (1 - mu))[:, None])
        try:
            step = np.linalg.solve(info, X.T @ (w * (y - mu)))
        except np.linalg.LinAlgError:
            return None
        beta = beta + step
        if np.max(np.abs(step)) < 1e-12:
            return beta
    return None


def paired(a: Sequence[float], b: Sequence[float], *, test_share: float | None, repeats: int,
           higher_is_better: bool = False) -> Any:
    """Two families' weighted fold scores, paired fold by fold, by the corrected repeated k-fold t
    (``validation.corrected_t``)."""
    from turbotab.core.models.validation import corrected_t

    return corrected_t(list(a), list(b), test_share=test_share, repeats=repeats,
                       higher_is_better=higher_is_better)


def design_bbc(losses: Mapping[str, np.ndarray], weights: np.ndarray, strata: Any, psu: Any, *,
               replicates: int = BBC_REPLICATES, seed: int = 0) -> dict[str, Any] | None:
    """Bootstrap bias-corrected cross-validation with the design: each resample draws, within each
    stratum, as many of its PSUs as it holds, with replacement; the family with the lowest weighted
    mean loss over the drawn PSUs' rows (counted as often as drawn) is chosen and scored by its
    weighted mean loss over the rows of the PSUs left out. ``losses``: each family's out-of-fold
    per-row loss (one repeat)."""
    keys = list(losses)
    if len(keys) < 2:
        return None
    s, p = codes(strata, psu, len(weights))
    by_stratum = {h: np.unique(p[s == h]) for h in sorted(set(s.tolist()))}
    rng = np.random.default_rng(int(seed))
    L = np.vstack([losses[k] for k in keys])
    w = np.asarray(weights, dtype=float)
    scores, wins = [], {k: 0 for k in keys}
    n_psu = int(p.max()) + 1
    for _ in range(int(replicates)):
        times = np.zeros(n_psu)
        for units in by_stratum.values():
            drawn = units[rng.integers(0, len(units), len(units))]
            np.add.at(times, drawn, 1.0)
        row_times = times[p]
        out = row_times == 0
        if not out.any():
            continue
        inbag = (L * (w * row_times)[None, :]).sum(axis=1) / float((w * row_times).sum())
        chosen = int(np.argmin(inbag))
        scores.append(float((L[chosen, out] * w[out]).sum() / w[out].sum()))
        wins[keys[chosen]] += 1
    if not scores:
        return None
    return {"corrected": float(np.mean(scores)), "corrected_low": float(np.percentile(scores, 2.5)),
            "corrected_high": float(np.percentile(scores, 97.5)), "replicates": len(scores),
            "wins": wins, "families": keys}


def _register_contract() -> None:
    from turbotab.core.contracts import (CONTRACTS, ContractOption, MethodContract, Relation,
                                        register_contract)

    if "design_based_cv" in CONTRACTS:
        return
    here = "turbotab.core.models.design_cv"
    register_contract(MethodContract(
        key="design_based_cv", label="Design-based cross-validation", slot="evaluation",
        scope="training_fold", package="EXPLORE", decision="set_survey", stage="evaluation",
        place="11 · Tuning and comparison", run_order=8.0,
        scope_note=("Each fold's model learns from the training rows as the procedure does; the "
                    "design only places whole PSUs in the folds and weighs every score."),
        needs=("the surveyed-population answer", "the weight, and the strata and PSUs if named"),
        question="(stated by the survey answer: the population's performance, or these rows')",
        options=(
            ContractOption("population", "Design-based cross-validation: whole PSUs within strata, "
                                         "every score weighted",
                           "Unweighted cross-validation is the habit even on survey data",
                           {"prediction": "Sound for performance in the surveyed population "
                                          "(Wieczorek et al. 2022)",
                            "inference": "Not shown: under inference no cross-validated score is "
                                         "reported"},
                           {"prediction": "recommended", "inference": "not_offered"}),
            ContractOption("sample", "Unweighted scores, labeled as these rows' performance",
                           "The field's habit",
                           {"prediction": "Sound for the procedure's performance on these rows, and "
                                          "labeled so",
                            "inference": "Not shown: under inference no cross-validated score is "
                                         "reported"},
                           {"prediction": "available", "inference": "not_offered"})),
        storyboard=("deal each stratum's PSUs to the folds", "fit on the other folds",
                    "weigh each held-out row's loss by its survey weight",
                    "average the folds' weighted means"),
        relations=(
            Relation("implies", "design_folds",
                     "folds keep whole PSUs together within strata, and every loss, calibration "
                     "and comparison is survey-weighted, labeled design-based cross-validation",
                     purposes=("prediction",), when=("population",),
                     condition="the surveyed-population answer under prediction",
                     enforced_by=f"{here}:design_folds", id="design_based_scores"),
            Relation("implies", "no_cv_under_inference",
                     "no cross-validated score is shown under inference",
                     purposes=("inference",), enforced_by="turbotab.core.stages.evaluation:"
                                                          "withhold_scores",
                     id="no_cv_score_under_inference")),
        sources=(SOURCE, "MODELING_SEQUENCE §0 ruling 13"),
        sentence="turbotab.core.stages.evaluation:design_sentence"))


_register_contract()


__all__ = ["BBC_REPLICATES", "LABEL", "REPEATS", "SOURCE", "codes", "design_bbc", "design_folds",
           "fold_pairs", "fold_scores", "paired", "svymean_se", "weighted_calibration",
           "weighted_mean"]
