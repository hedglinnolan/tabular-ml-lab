"""Intended use → the decision curve, subgroup performance and model updating (MODELING_SEQUENCE §1
rows 2 and 11, prediction; TRIPOD+AI items 12e, 12f, 14, 15, 23a and 24).

**Intended use** (``set_intended_use``) asks whether the model will support a decision. Decision
support gates the decision curve and the threshold question (TRIPOD+AI 15: "Provide details and
rationale for any classification and how the thresholds were identified").

**Net benefit** (Vickers & Elkin, Med Decis Making 2006;26:565), at a threshold probability t: the
true positives per row less the false positives per row weighted by the odds at the threshold,

    NB(t) = TP/n − FP/n · t/(1 − t),

a row "positive" when its predicted risk is at least t, exactly as R's ``dcurves::dca`` counts it
(``risk >= threshold``); treating everyone is NB = π − (1 − π)·t/(1 − t), treating no one 0. With
survey weights each count is a weighted total over the weights' total. The curve is drawn on the
out-of-fold predictions, so no row is scored by a model that saw it.

**The threshold is declared, or chosen in-fold.** A threshold probability is a statement about the
decision's harms and benefits (Vickers & Elkin: "the threshold probability … reflects the relative
harms of false-positive and false-negative results"), so the soundest one is declared from them
before any score is seen (``threshold``), and every fold uses it. When none is declared, the
customary data-driven rule, Youden's J (sensitivity + specificity − 1; Youden 1950), picks it within
the declared range on each outer fold's own training rows, and the sensitivity, specificity and net
benefit reported are that threshold's on the fold the model never saw, pooled over the folds. The
choice is part of the procedure the resampling repeats, so its optimism is in the estimate.
(Maximizing net benefit across thresholds is no rule: each threshold's net benefit weighs false
positives by its own odds.)

**Subgroup performance** (TRIPOD+AI 23a: "Report model performance estimates with confidence
intervals, including for any key subgroups (eg, sociodemographic)"; item 14, the fairness approach,
recorded even when it is none): each named column's groups (its own levels, quoted verbatim; thirds
of a continuous column, which reads nothing into its unit) scored on the out-of-fold predictions with
the intervals of ``models/performance.py``.

**Model updating** (TRIPOD+AI 12f; Collins et al., BMJ 2024: "the value of optimism corrected
calibration slope can be used to adjust the model from any overfitting by applying it as shrinkage
factor to the original regression coefficients"): the regression's coefficients multiplied by the
slope s and the intercept re-estimated with the shrunk linear predictor as an offset (Steyerberg,
*Clinical Prediction Models* 2nd ed. 2019, §13.2; Van Houwelingen & le Cessie, Stat Med 1990;9:1303).
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

DEFAULT_RANGE = (0.05, 0.50)  # the threshold range when the answer names none (a convention)
GRID_STEP = 0.01  # dcurves' default thresholds: seq(0, 0.99, by = 0.01)
MIN_GROUP_ROWS = 10
MIN_GROUP_EVENTS = 5
SOURCES = {
    "vickers": "Vickers & Elkin, Med Decis Making 2006;26:565",
    "tripod": "Collins et al., TRIPOD+AI, BMJ 2024;385:e078378",
    "collins": "Collins et al., BMJ 2024;384:e074819",
    "steyerberg": "Steyerberg, Clinical Prediction Models 2nd ed. 2019, §13.2",
    "goorbergh": "van den Goorbergh et al., JAMIA 2022;29:1525",
    "stratos": "Van Calster et al., STRATOS TG6, arXiv:2412.10288",
}


# ── net benefit ──────────────────────────────────────────────────────────────


def grid(low: float, high: float, step: float = GRID_STEP) -> np.ndarray:
    """Thresholds from ``low`` to ``high`` by ``step`` (rounded to the step's decimals)."""
    count = int(round((high - low) / step)) + 1
    return np.round(low + step * np.arange(count), 10)


def net_benefit(event: Any, risk: Any, thresholds: Any, weights: Any = None) -> dict[str, np.ndarray]:
    """``{model, all, tp_rate, fp_rate}`` at each threshold (module docstring); ``event`` 0/1."""
    e = np.asarray(event, dtype=float)
    r = np.asarray(risk, dtype=float)
    t = np.asarray(thresholds, dtype=float)
    w = np.ones(len(e)) if weights is None else np.asarray(weights, dtype=float)
    total = float(w.sum())
    positive = r[None, :] >= t[:, None]
    tp = (positive * (w * e)[None, :]).sum(axis=1) / total
    fp = (positive * (w * (1 - e))[None, :]).sum(axis=1) / total
    odds = t / (1 - t)
    prevalence = float((w * e).sum() / total)
    return {"model": tp - fp * odds, "all": prevalence - (1 - prevalence) * odds,
            "tp_rate": tp, "fp_rate": fp}


def classified(event: Any, risk: Any, threshold: float) -> dict[str, float]:
    """Sensitivity, specificity and net benefit of classifying a row positive when its risk is at
    least ``threshold``."""
    e = np.asarray(event, dtype=float)
    positive = np.asarray(risk, dtype=float) >= threshold
    events, others = float(e.sum()), float((1 - e).sum())
    nb = net_benefit(e, risk, [threshold])
    return {"sensitivity": float((positive * e).sum() / events) if events else float("nan"),
            "specificity": float((~positive * (1 - e)).sum() / others) if others else float("nan"),
            "net_benefit": float(nb["model"][0]), "treat_all": float(nb["all"][0])}


def youden_threshold(event: Any, risk: Any, thresholds: Sequence[float]) -> float:
    """The threshold in ``thresholds`` with the largest Youden's J = sensitivity + specificity − 1
    (Youden 1950; the customary data-driven rule), the lowest among ties."""
    e = np.asarray(event, dtype=float)
    r = np.asarray(risk, dtype=float)
    t = np.asarray(thresholds, dtype=float)
    positive = r[None, :] >= t[:, None]
    sens = (positive * e[None, :]).sum(axis=1) / max(float(e.sum()), 1.0)
    spec = (~positive * (1 - e)[None, :]).sum(axis=1) / max(float((1 - e).sum()), 1.0)
    return float(t[int(np.argmax(sens + spec - 1.0))])


def threshold_in_fold(folds: Sequence[Mapping[str, Any]], thresholds: Sequence[float],
                      declared: float | None = None) -> dict[str, Any]:
    """The threshold (module docstring). ``folds``: each outer fold's ``train_event``,
    ``train_risk`` (its model's predictions on its own training rows), ``test_event`` and
    ``test_risk``. With a ``declared`` threshold (set from the decision's harms and benefits before
    any score was seen) every fold uses it; otherwise each fold chooses by Youden's J on its own
    training rows within ``thresholds``. Returns each fold's threshold and held-out sensitivity,
    specificity and net benefit, and those pooled over the folds (each fold weighted by its rows)."""
    rows = []
    for f in folds:
        t = float(declared) if declared is not None else youden_threshold(
            f["train_event"], f["train_risk"], thresholds)
        held = classified(f["test_event"], f["test_risk"], t)
        rows.append({"threshold": t, "n": len(np.asarray(f["test_event"])), **held})
    n = float(sum(r["n"] for r in rows))

    def pooled(key: str) -> float | None:
        values = [(r["n"], r[key]) for r in rows if math.isfinite(r[key])]
        return float(sum(k * v for k, v in values) / sum(k for k, _ in values)) if values else None

    chosen = [r["threshold"] for r in rows]
    return {"folds": rows, "declared": declared is not None, "thresholds": chosen,
            "median_threshold": float(np.median(chosen)) if chosen else None,
            "net_benefit": pooled("net_benefit") if n else None,
            "sensitivity": pooled("sensitivity") if n else None,
            "specificity": pooled("specificity") if n else None}


def decision_curve(event: Any, risks: Mapping[str, Any], thresholds: Sequence[float],
                   weights: Any = None) -> list[dict[str, Any]]:
    """One row per threshold: each named model's net benefit, treat all and treat none."""
    out = []
    computed = {name: net_benefit(event, r, thresholds, weights)["model"] for name, r in risks.items()}
    every = net_benefit(event, np.ones(len(np.asarray(event))), thresholds, weights)["all"]
    for i, t in enumerate(thresholds):
        out.append({"threshold": float(t), "treat_all": float(every[i]), "treat_none": 0.0,
                    "models": {name: float(v[i]) for name, v in computed.items()}})
    return out


def useful_range(rows: Sequence[Mapping[str, Any]], model: str) -> tuple[float, float] | None:
    """The thresholds where the model's net benefit beats both treating everyone and no one."""
    useful = [r["threshold"] for r in rows
              if r["models"][model] > max(r["treat_all"], 0.0) + 1e-12]
    return (min(useful), max(useful)) if useful else None


# ── subgroup performance ─────────────────────────────────────────────────────


def subgroup_labels(values: Any, *, max_levels: int = 10) -> tuple[np.ndarray, str]:
    """Each row's group and how the groups were made: the column's own levels (as text) when it
    has at most ``max_levels``, else its thirds by its 1/3 and 2/3 quantiles (R type 7). Blank rows
    are a group of their own, ``(blank)``."""
    s = pd.Series(np.asarray(values, dtype=object))
    present = s.dropna()
    numeric = pd.to_numeric(present, errors="coerce")
    if len(present.unique()) > max_levels and numeric.notna().all():
        x = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
        cuts = np.quantile(x[np.isfinite(x)], [1 / 3, 2 / 3])
        group = np.searchsorted(cuts, x, side="left")
        names = np.asarray([f"≤ {cuts[0]:.4g}", f"{cuts[0]:.4g}–{cuts[1]:.4g}", f"> {cuts[1]:.4g}"],
                           dtype=object)
        labels = np.where(np.isfinite(x), names[np.clip(group, 0, 2)], "(blank)")
        return labels.astype(object), "thirds"
    labels = s.map(lambda v: "(blank)" if pd.isna(v) else _plain(v)).to_numpy(dtype=object)
    return labels, "levels"


def _plain(v: Any) -> str:
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def subgroup_performance(task: str, y: Any, prediction: Any, columns: Mapping[str, Any], *,
                         classes: Sequence[Any] | None = None, metrics: Sequence[str],
                         reference: Any = None) -> list[dict[str, Any]]:
    """Each named column's groups scored with intervals (module docstring). ``columns``: name →
    each scored row's value; ``prediction`` as ``performance.score_intervals`` takes it."""
    from turbotab.core.models.performance import score_intervals

    y = np.asarray(y)
    P = np.asarray(prediction)
    out = []
    for name, values in columns.items():
        labels, how = subgroup_labels(values)
        groups = []
        for level in sorted(set(labels.tolist()), key=lambda v: (v == "(blank)", v)):
            rows = labels == level
            n = int(rows.sum())
            entry: dict[str, Any] = {"group": level, "n": n, "scores": {}, "note": None}
            if task == "binary":
                events = int((y[rows] == list(classes or [0, 1])[1]).sum())
                entry["events"] = events
                if events < MIN_GROUP_EVENTS or n - events < MIN_GROUP_EVENTS:
                    entry["note"] = (f"Too few rows of one class to score this group (fewer than "
                                     f"{MIN_GROUP_EVENTS}).")
                    groups.append(entry)
                    continue
            if n < MIN_GROUP_ROWS:
                entry["note"] = f"Fewer than {MIN_GROUP_ROWS} rows: not scored."
                groups.append(entry)
                continue
            ref = (None if reference is None else
                   (float(np.mean(np.asarray(reference)[rows])) if np.ndim(reference) else reference))
            found = score_intervals(task, y[rows], P[rows], classes=classes, reference=ref)
            entry["scores"] = {m: found[m].model_dump(mode="json") for m in metrics if m in found}
            groups.append(entry)
        out.append({"column": name, "grouped_by": how, "groups": groups})
    return out


# ── model updating ───────────────────────────────────────────────────────────


def shrunk_intercept(task: str, linear_predictor: Any, y: Any, slope: float) -> float:
    """The intercept re-estimated with ``slope`` × the linear predictor (without its intercept) as
    an offset: least squares (the mean residual) for a numeric outcome, the logistic maximum
    likelihood for a yes/no one (``glm(y ~ 1 + offset(s·lp), family = binomial)``)."""
    lp = np.asarray(linear_predictor, dtype=float)
    yy = np.asarray(y, dtype=float)
    if task == "regression":
        return float(np.mean(yy - slope * lp))
    from turbotab.core.models.performance import logistic_fit

    fitted = logistic_fit(np.ones((len(yy), 1)), yy, offset=slope * lp)
    if fitted is None:
        raise ValueError("The intercept could not be re-estimated (the outcome is separated).")
    return float(fitted[0][0])


def shrinkage(task: str, matrix: pd.DataFrame, y: Any, coefficients: Sequence[float],
              slope: float) -> dict[str, Any]:
    """Uniform shrinkage of a regression (module docstring): each coefficient times ``slope``, and
    the intercept re-estimated."""
    beta = np.asarray(coefficients, dtype=float)
    lp = matrix.to_numpy(dtype=float) @ beta
    intercept = shrunk_intercept(task, lp, y, slope)
    return {"factor": float(slope), "intercept": intercept,
            "coefficients": [{"feature": str(c), "estimate": float(b), "shrunk": float(slope * b)}
                             for c, b in zip(matrix.columns, beta)]}


def slope_of(model: Mapping[str, Any]) -> tuple[float | None, str | None]:
    """The fit's calibration slope for a family, and how it was made: the bootstrap's
    optimism-corrected slope when the split asked for it, else the out-of-fold slope."""
    optimism = ((model.get("optimism") or {}).get("estimates") or {}).get("calibration_slope") or {}
    if optimism.get("corrected") is not None:
        return float(optimism["corrected"]), "the optimism-corrected calibration slope (Harrell's bootstrap)"
    slope = ((model.get("calibration") or {}).get("slope") or {}).get("estimate")
    if slope is not None:
        return float(slope), "the out-of-fold calibration slope (cross-validation)"
    return None, None


# ── the contract, the leash and the sentences ────────────────────────────────


def _register_contract() -> None:
    from turbotab.core.contracts import (CONTRACTS, ContractOption, MethodContract, Relation,
                                        register_contract)

    if "intended_use" in CONTRACTS:
        return
    here = "turbotab.core.models.decision_curve"
    prediction = ("prediction",)

    def option(key: str, label: str, customary: str, says: str, rung: str, order: int
               ) -> ContractOption:
        return ContractOption(key, label, customary,
                              {"prediction": says, "inference": "Not asked: no score is reported "
                                                                "under inference"},
                              {"prediction": rung, "inference": "not_offered"},
                              {"prediction": order, "inference": order})

    register_contract(MethodContract(
        key="intended_use", label="Intended use, the decision curve and the threshold",
        slot="evaluation", scope="training_fold", package="EXPLORE", decision="set_intended_use",
        stage="evaluation", place="2 · Exposure and estimand (prediction: intended use)",
        run_order=7.0,
        scope_note=("The curve reads the out-of-fold predictions, and a named threshold is chosen "
                    "on each training fold and scored on the fold its model never saw."),
        needs=("a yes/no outcome", "out-of-fold predictions"),
        question="Will the model support a decision at a risk threshold?",
        options=(
            option("decision_support", "Decision support: the decision curve, a threshold chosen "
                   "in-fold", "Decision curves are recommended but still rare in nutrition papers",
                   "Sound: net benefit over the declared threshold range (Vickers & Elkin 2006; "
                   "STRATOS TG6 lists it as essential)", "recommended", 0),
            option("risk_estimation", "Risk estimation only: no threshold",
                   "Most prediction papers report discrimination and calibration only",
                   "Sound when no decision rests on a cut-off; the curve is not drawn",
                   "available", 1),
        ),
        storyboard=("each row's out-of-fold risk", "count true and false positives at each "
                    "threshold", "weigh false positives by the threshold's odds",
                    "against treating everyone and no one"),
        relations=(
            Relation("enables", "decision_curve",
                     "the decision curve is drawn over the declared threshold range, beside treat "
                     "all and treat none", purposes=prediction, when=("decision_support",),
                     condition="decision support with a yes/no outcome",
                     enforced_by=f"{here}:decision_curve", id="decision_support_curve"),
            Relation("implies", "threshold_in_fold",
                     "a named threshold is chosen in each training fold and scored on the fold its "
                     "model never saw", purposes=prediction, when=("decision_support",),
                     condition="decision support", enforced_by=f"{here}:threshold_in_fold",
                     id="threshold_chosen_in_fold"),
            Relation("implies", "subgroup_performance",
                     "performance with intervals in each named sociodemographic subgroup "
                     "(TRIPOD+AI 23a), the fairness approach recorded (item 14)",
                     purposes=prediction, condition="subgroup columns named",
                     enforced_by=f"{here}:subgroup_performance", id="subgroups_scored"),
            Relation("enables", "model_updating",
                     "uniform shrinkage by the calibration slope is offered as model updating",
                     purposes=prediction, condition="an unpenalized regression family",
                     enforced_by=f"{here}:shrinkage", id="shrinkage_offered"),
        ),
        sources=(SOURCES["vickers"], SOURCES["stratos"], SOURCES["tripod"], SOURCES["collins"],
                 SOURCES["steyerberg"]),
        sentence=f"{here}:intended_use_sentence"))


_register_contract()


def _prediction_only(kind_words: str) -> Any:
    def check(decision: Any, ctx: Any) -> None:
        from turbotab.core.decisions import Refusal, SetPurpose, _state

        state = _state(ctx)
        if state is None or getattr(state, "purpose", None) != "inference":
            return
        raise Refusal(
            "not_prediction",
            f"{kind_words} is a prediction question: under inference no cross-validated score is "
            f"reported, so there is nothing to weigh at a threshold or to update.",
            exits=[{"label": "Make the purpose prediction",
                    "decision": SetPurpose(purpose="prediction")}])
    return check


def _intended_use_fits(decision: Any, ctx: Any) -> None:
    """Decision support needs a yes/no outcome; a threshold range runs low to high; the subgroup
    columns are the table's."""
    from turbotab.core.decisions import Refusal, SetIntendedUse, _columns_of, _ctx, _state

    state = _state(ctx)
    task = getattr(state, "task", None) if state is not None else None
    task = task or _ctx(ctx, "detected_task")
    if decision.use == "decision_support" and task is not None and task != "binary":
        raise Refusal(
            "decision_support_task",
            f"A decision curve weighs true and false positives of a yes/no outcome; a "
            f"{str(task).replace('_', '-')} outcome's model is used for risk estimation here.",
            exits=[{"label": "Risk estimation only",
                    "decision": decision.model_copy(update={"use": "risk_estimation",
                                                            "threshold_low": None,
                                                            "threshold_high": None})}])
    low, high = decision.threshold_low, decision.threshold_high
    if low is not None and high is not None and low >= high:
        raise Refusal("threshold_range", "The threshold range runs from its lower to its higher "
                                         "probability.",
                      exits=[{"label": f"From {high:g} to {low:g}",
                              "decision": decision.model_copy(update={"threshold_low": high,
                                                                      "threshold_high": low})}])
    columns = _columns_of(ctx)
    if columns is not None:
        unknown = [c for c in decision.subgroups if c not in columns]
        if unknown:
            kept = [c for c in decision.subgroups if c in columns]
            raise Refusal("unknown_column", f"This dataset has no column named `{unknown[0]}`.",
                          exits=[{"label": "Keep the subgroups that are columns",
                                  "decision": decision.model_copy(update={"subgroups": kept})}])
    if state is not None and getattr(state, "target", None) in decision.subgroups:
        raise Refusal("subgroup_is_outcome", "The outcome cannot group its own performance.",
                      exits=[{"label": "Leave the outcome out of the subgroups",
                              "decision": SetIntendedUse(**{
                                  **decision.model_dump(exclude={"kind"}),
                                  "subgroups": [c for c in decision.subgroups
                                                if c != state.target]})}])


def _register_validators() -> None:
    from turbotab.core.decisions import register_validator

    register_validator("set_intended_use", _prediction_only("Intended use"))
    register_validator("set_intended_use", _intended_use_fits)
    register_validator("set_updating", _prediction_only("Model updating"))


_register_validators()


def intended_use_sentence(d: Any, state: Any) -> str:
    """The ``set_intended_use`` record's sentence."""
    from turbotab.core.voice import listing, tick

    if d.use == "decision_support":
        low = d.threshold_low if d.threshold_low is not None else DEFAULT_RANGE[0]
        high = d.threshold_high if d.threshold_high is not None else DEFAULT_RANGE[1]
        chosen = (f"the decision threshold {tick(f'{d.threshold:g}')} was declared from the "
                  f"decision's harms and benefits before any score was seen"
                  if d.threshold is not None else
                  "the decision threshold was chosen within each training fold by Youden's J "
                  "(customary) and scored on the fold its model never saw")
        head = (f"The model was declared for decision support: net benefit was assessed by decision "
                f"curve analysis over threshold probabilities from {tick(f'{low:g}')} to "
                f"{tick(f'{high:g}')} (Vickers & Elkin 2006), and {chosen}")
    else:
        head = "The model was declared for risk estimation: no classification threshold was set"
    if d.subgroups:
        tail = (f"; performance was reported with 95% intervals within the groups of "
                f"{listing(d.subgroups)} (TRIPOD+AI 23a)")
    else:
        tail = "; no subgroup was named for performance"
    fair = ("; no fairness method beyond that report was applied (TRIPOD+AI 14)"
            if d.fairness == "subgroup_performance" else
            "; no fairness approach was applied (TRIPOD+AI 14)")
    return head + tail + fair


def updating_sentence(d: Any, state: Any) -> str:
    """The ``set_updating`` record's sentence."""
    if d.method == "shrinkage":
        return ("The regression's coefficients were shrunk by the calibration slope as model "
                "updating, and its intercept re-estimated with the shrunk linear predictor as an "
                "offset (TRIPOD+AI 12f; Steyerberg 2019).")
    return "No model updating was applied (TRIPOD+AI 12f)."


__all__ = ["DEFAULT_RANGE", "GRID_STEP", "classified", "decision_curve", "grid", "youden_threshold",
           "intended_use_sentence", "net_benefit", "shrinkage", "shrunk_intercept", "slope_of",
           "subgroup_labels", "subgroup_performance", "threshold_in_fold", "updating_sentence",
           "useful_range"]
