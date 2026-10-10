"""The substitution stage for a multiclass outcome: one curve per class, per fitted family.

V2 definition of done §2 ("Dietary, extended": multiclass substitution curves, one per class). The
substitution stage (:func:`turbotab.core.stages.modeling.substitution_stage`) reads its rows, its
fits, its kcal per unit and its shift as it does for any outcome, then hands each family here when
the outcome's classes are unordered. The math is :mod:`turbotab.core.methods.substitution`'s
(:func:`~turbotab.core.methods.substitution.class_curves` and its band, pooling and design-based
twins); this module chooses among them as the substitution stage chooses for one curve:

* **one fit** (prediction's training fit, or inference's fit on every analyzed row): each class's
  curve through that fit, and with ``n_boot`` each class's band from refits on bootstrap resamples
  (whole units when rows repeat);
* **multiple imputation** under inference (MS3): each completed copy's class curves through that
  copy's own fit, pooled at each k by Rubin's rules, each copy's bootstrap (or design-based)
  variance its within-copy variance;
* **the surveyed population** under inference (MS4): the linear family's survey-weighted
  multinomial fit, each curve the population's weighted mean and its band the design's
  linearization; a family with no design-based estimator draws no curve (blocked and recorded,
  with its exits).

**The linear family's fit, solved to convergence.** The linear family fits an unpenalized
multinomial logit with scikit-learn, which stops at a gradient tolerance. Its curve follows the
same model's maximum-likelihood estimate on the same rows, refined by Newton–Raphson until the
Newton decrement vanishes (:func:`class_predictor`), so a curve agrees with any other exact solver
of the same likelihood. The refinement never replaces the fit: where it would move any row's
probability by more than :data:`POLISH_AGREEMENT`, or the fit is penalized or weighted, or Newton
does not converge (a separating column), the curve follows scikit-learn's own fit.
"""
from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

# The refinement must stay within the solver's own tolerance of the fit it refines (scikit-learn
# stops when the largest gradient of the mean log loss is below 1e-4); a larger move means the rows
# or the estimator differ, and the fit stands as it was made.
POLISH_AGREEMENT = 1e-4


@dataclass
class ClassDraw:
    """One family's class curves, as the substitution artifact takes them.

    ``entries`` are its ``SubstitutionModel`` entries (one per class; one blocked entry when the
    family draws none); ``curve`` the first :func:`class_curves` dict (support counts and the
    note); ``band`` the refit band's record for the caption; ``design`` the design-based band's
    (``variance``, ``df``) under the surveyed population; ``m`` the copies pooled."""

    entries: list[dict[str, Any]]
    curve: dict[str, Any] | None = None
    band: dict[str, Any] | None = None
    design: dict[str, Any] | None = None
    m: int | None = None
    seconds: float = 0.0
    failed: int = 0
    estimate: float = 0.0


def _plain_multinomial(family: Any, model: Any) -> bool:
    """Whether ``model``, ``family``'s fitted model step, is an unpenalized, unweighted multinomial
    logit with an intercept: the family declares a weighted sum of the values as given
    (``linear_in_values``) read out as a margin (``output``, the log-odds), and the fit's own
    settings hold no penalty, no class weights and an intercept, over three classes or more.

    A fit that does not state its penalty is read as penalized: its settings must hold the
    logit's inverse penalty ``C``, and either name no penalty or set ``C`` infinite. A ridge-type
    classifier (``alpha``, no ``C``) or a wrapper that states no settings is never refined."""
    if family is None or not family.linear_in_values or family.output != "margin":
        return False
    params = model.get_params(deep=False)
    if "C" not in params:
        return False
    unpenalized = (params.get("penalty", "unstated") in (None, "none")
                   or not math.isfinite(float(params["C"])))
    return (unpenalized and params.get("class_weight") is None
            and bool(params.get("fit_intercept", True)) and len(getattr(model, "classes_", ())) >= 3)


def class_predictor(pipeline: Any, X_rows: pd.DataFrame, y_rows: Any,
                    classes: Sequence[Any],
                    family: Any = None) -> Callable[[pd.DataFrame], np.ndarray]:
    """``pipeline``'s probability of each of ``classes`` (in that order) for the rows of a frame.

    ``X_rows`` and ``y_rows`` are the rows the pipeline was fit on, and ``family`` the family that
    fit it. For an unpenalized multinomial logit (:func:`_plain_multinomial`: the linear family's)
    the probabilities are its maximum-likelihood estimate on those rows, refined to convergence
    (module docstring); for any other family, or with no family named, they are its own."""
    from turbotab.core.models.linear import model_matrix
    from turbotab.core.models.survey import _softmax, weighted_multinomial

    classes = list(classes)
    order = list(getattr(pipeline, "classes_", classes))
    if sorted(map(str, order)) != sorted(map(str, classes)) or len(order) != len(classes):
        raise ValueError("The fit does not hold every class of the outcome (a resample without one "
                         "of them).")
    index = [order.index(c) for c in classes]

    def fitted(frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(pipeline.predict_proba(frame), dtype=float)[:, index]

    model = pipeline.steps[-1][1] if hasattr(pipeline, "steps") else pipeline
    if not _plain_multinomial(family, model):
        return fitted
    try:
        values = model_matrix(pipeline, X_rows).to_numpy(dtype=float)
        if not np.isfinite(values).all():
            return fitted
        design = np.column_stack([np.ones(len(values)), values])
        codes = pd.Categorical(pd.Series(np.asarray(y_rows, dtype=object)),
                               categories=order).codes.astype(np.int64)
        if (codes < 0).any() or len(codes) != len(design):
            return fitted
        K = len(order)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            fit = weighted_multinomial(design, codes, K, np.ones(len(design)))
        if not fit.converged:
            return fitted
        B = fit.estimate.reshape(K - 1, design.shape[1]).T
        moved = float(np.max(np.abs(_softmax(design @ B)
                                    - np.asarray(pipeline.predict_proba(X_rows), dtype=float))))
        if not moved <= POLISH_AGREEMENT:
            return fitted
    except (ValueError, np.linalg.LinAlgError):
        return fitted

    def refined(frame: pd.DataFrame) -> np.ndarray:
        if len(frame) == 0:
            return np.empty((0, len(classes)))
        matrix = model_matrix(pipeline, frame).to_numpy(dtype=float)
        return _softmax(np.column_stack([np.ones(len(matrix)), matrix]) @ B)[:, index]

    return refined


def population_blocked(family: Any, task: str, models: Sequence[str]) -> tuple[str, list[dict]]:
    """(why, exits) for a family with no design-based estimator under the surveyed population: no
    class curves (MODELING_SEQUENCE §4, blocked and recorded); its exits put the design-based family
    in its place, every other chosen family kept, or take the sample-only attestation."""
    from turbotab.core.models.survey import no_design_estimator

    table = no_design_estimator(family, task, models, what="substitution curves")
    return str(table.info["refused"]), [dict(e) for e in table.info["exits"]]


def _blocked_entry(key: str, family: Any, n_k: int, refused: str,
                   exits: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {"family": key, "label": family.label, "level": None, "delta": [None] * n_k,
            "ci_low": None, "ci_high": None, "on_support_fraction": [0.0] * n_k,
            "stopped_at": None, "effect_label": None, "fixed_delta": [None] * n_k,
            "fixed_ci_low": None, "fixed_ci_high": None, "band_ok": None, "refused": refused,
            "exits": list(exits)}


def _support_only(X: pd.DataFrame, weights: Any, curve_args: Mapping[str, Any]) -> dict[str, Any]:
    """The support's counts on ``X`` with no model (a blocked family's: they are the data's)."""
    from turbotab.core.methods.substitution import substitution_curve

    return substitution_curve(lambda frame: np.zeros(len(frame)), X, weights=weights, **curve_args)


def _label(value: float | None, chosen: float | None, name: str, scale: str) -> str | None:
    from turbotab.core.methods.substitution import PER_UNIT, _amount, _plain, _signed

    if value is None or chosen is None:
        return None
    per = PER_UNIT[scale]
    return (f"{_signed(value * per / chosen)} in the probability of {name} per {_amount(per, scale)} "
            f"at k = {_plain(chosen)}")


def _split_seed(state: Any) -> int:
    """The recorded split's seed (0 when no split is recorded)."""
    split = getattr(state, "split", None)
    return int(getattr(split, "seed", 0) or 0) if split is not None else 0


def class_family_entries(key: str, family: Any, *, task: str, pipeline: Any, template: Any,
                         X_fit: pd.DataFrame, y_fit: Any, X: pd.DataFrame, curve_rows: Any,
                         shift: Any, ks: Sequence[float], sub: Any, state: Any,
                         kcal_per_unit: Mapping[str, float], nested: Mapping[str, str],
                         total_energy: str | None, scale: str, percent: Sequence[str],
                         n_boot: int, groups: Any, group_of: Any, interval: str,
                         imputed: Mapping[str, Any] | None, train_ids: Any,
                         survey_design: Any = None, domain: Any = None,
                         models: Sequence[str] = (),
                         progress: Callable[[float, str], None] | None = None,
                         seed: int | None = None, designs: Any = None,
                         cancelled: Callable[[], bool] | None = None) -> ClassDraw:
    """``key``'s class curves as the substitution stage draws them (module docstring).

    ``pipeline`` is the family's fit on ``X_fit``/``y_fit`` (every analyzed row under inference,
    the training rows under prediction; None when the family must be refit on them), ``template``
    its unfitted pipeline for the band's refits, ``X`` the curve's rows (``curve_rows``: their
    positions in ``X_fit``) and ``shift`` the move on them. ``imputed`` is the fit's multiple
    imputations (each copy's frame and fit) under inference, ``survey_design`` and ``domain`` the
    surveyed population's, ``models`` every chosen family (a blocked curve's exit keeps the rest).

    F15 (RECIPES §4.3): every refit here is fit at the split's ``seed`` (the recorded split's when
    None), with its rows' survey design under the population answer (``designs``,
    ``stages.modeling.fit_designs``, by row id), inside a cancel scope asking ``cancelled``.
    """
    from sklearn.base import clone

    from turbotab.core.stages.modeling import BAND_BOOT, BAND_ROWS, fit_with, pinned_to_full_fit

    seed = _split_seed(state) if seed is None else int(seed)

    curve_args = dict(donor=sub.donor, recipient=sub.recipient, kcal_per_unit=kcal_per_unit,
                      ks=ks, total_kind="variable", nested=nested, total=total_energy,
                      scale=scale)
    say = progress or (lambda fraction, message: None)
    if imputed is not None and ((imputed.get("fits") or {}).get(key) or template is not None):
        # MS3: a family whose table drew no copy fits of its own (boosted trees, a penalized
        # family) is refit on each copy, never drawn on one fill beside the pooled ones.
        return _pooled(key, family, task=task, imputed=imputed, train_ids=train_ids, state=state,
                       sub=sub, ks=ks, kcal_per_unit=kcal_per_unit, nested=nested,
                       total_energy=total_energy, scale=scale, percent=percent, n_boot=n_boot,
                       template=template, y_fit=y_fit, group_of=group_of,
                       survey_design=survey_design, models=models, progress=say, seed=seed,
                       designs=designs, cancelled=cancelled)
    if survey_design is not None:
        return _population(key, family, task=task, pipeline=pipeline, X=X,
                           y=np.asarray(y_fit)[np.asarray(curve_rows)], shift=shift,
                           design=survey_design, domain=domain, models=models,
                           curve_args=curve_args, ks=ks)
    if pipeline is None:
        # Under inference a family with no coefficient table (boosted trees) was fit on the
        # training rows only; its curves read its refit on every analyzed row.
        say(0.0, f"{family.label}: refitting on every analyzed row")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline = fit_with(clone(template), X_fit, y_fit, groups=groups, designs=designs,
                                seed=seed, cancelled=cancelled)
    from turbotab.core.methods.substitution import class_curves, class_refit_band

    classes = list(pipeline.classes_)
    predict = class_predictor(pipeline, X_fit, y_fit, classes, family)
    say(0.05, f"{family.label}: moving energy")
    curves = class_curves(predict, X, classes=classes, shift=shift, **curve_args)
    draw = ClassDraw(entries=_entries(key, family, curves), curve=curves)
    if template is None:
        return draw

    def refit(Xb: pd.DataFrame, yb: Any, _full: Any = pipeline) -> Any:
        # Every copy of a resampled row (every row of a unit) is one inner unit, as the one-curve
        # band's refits keep it.
        inner = (group_of.loc[Xb.index].to_numpy() if group_of is not None
                 else Xb.index.to_numpy())
        pipe = pinned_to_full_fit(clone(template), _full)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            refitted = fit_with(pipe, Xb, yb, groups=inner, designs=designs, seed=seed,
                                cancelled=cancelled)
        return class_predictor(refitted, Xb, yb, classes, family)

    common = dict(classes=classes, shift=shift, ks=ks, live=curves["live"], groups=groups,
                  random_state=0, centers=[c["delta"] for c in curves["classes"]],
                  fixed_centers=[c["fixed_delta"] for c in curves["classes"]],
                  curve_rows=curve_rows, max_rows=BAND_ROWS, interval=interval)
    if n_boot:
        def counted(done: int, total: int) -> None:
            say(0.1 + 0.9 * done / total, f"{family.label}: refit {done} of {total}")

        band = class_refit_band(refit, X_fit, y_fit, n_boot=n_boot, progress=counted, **common)
        for entry, drawn in zip(draw.entries, band["classes"]):
            entry.update(ci_low=drawn["ci_low"], ci_high=drawn["ci_high"],
                         fixed_ci_low=drawn["fixed_ci_low"], fixed_ci_high=drawn["fixed_ci_high"],
                         band_ok=band["n_ok"])
        draw.band, draw.seconds, draw.failed = band, band["seconds"], band["failed"]
    else:
        say(0.5, f"{family.label}: timing one refit for the band")
        timed = class_refit_band(refit, X_fit, y_fit, n_boot=1, **common)
        draw.estimate = timed["seconds"] * BAND_BOOT
    return draw


def _entries(key: str, family: Any, curves: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One ``SubstitutionModel`` entry per class of a :func:`class_curves` dict, with no band."""
    return [{"family": key, "label": f"{family.label}: {c['name']}", "level": c["name"],
             "delta": c["delta"], "ci_low": None, "ci_high": None,
             "on_support_fraction": curves["on_support_fraction"],
             "stopped_at": curves["stopped_at"], "effect_label": c["effect_label"],
             "fixed_delta": c["fixed_delta"], "fixed_ci_low": None, "fixed_ci_high": None,
             "band_ok": None} for c in curves["classes"]]


def _population(key: str, family: Any, *, task: str, pipeline: Any, X: pd.DataFrame, y: Any,
                shift: Any, design: Any, domain: Any, models: Sequence[str],
                curve_args: Mapping[str, Any], ks: Sequence[float]) -> ClassDraw:
    """The class curves over the surveyed population: the linear family's survey-weighted
    multinomial fit and its linearized band, or the block (MS4)."""
    from turbotab.core.methods.substitution import design_class_curves
    from turbotab.core.models.linear import model_matrix
    from turbotab.core.models.survey import SAMPLE_EXIT, has_design_estimator

    if not has_design_estimator(family, task) or pipeline is None:
        refused, exits = population_blocked(family, task, models)
        support = _support_only(X, domain.raw, {**curve_args, "shift": shift})
        return ClassDraw(entries=[_blocked_entry(key, family, len(ks), refused, exits)],
                         curve=support)
    classes = list(pipeline.classes_)
    try:
        drawn = design_class_curves(lambda frame: model_matrix(pipeline, frame), X, y, classes,
                                    design, domain, shift=shift, **curve_args)
    except ValueError as exc:
        support = _support_only(X, domain.raw, {**curve_args, "shift": shift})
        exits = [{"label": SAMPLE_EXIT, "decision": {"kind": "set_survey", "estimand": "sample"}}]
        return ClassDraw(entries=[_blocked_entry(key, family, len(ks),
                                                 f"No design-based curves: {exc}", exits)],
                         curve=support)
    curves, band = drawn["curves"], drawn["band"]
    entries = _entries(key, family, curves)
    for entry, c in zip(entries, band["classes"]):
        entry.update(ci_low=c["ci_low"], ci_high=c["ci_high"], fixed_ci_low=c["fixed_ci_low"],
                     fixed_ci_high=c["fixed_ci_high"])
    return ClassDraw(entries=entries, curve=curves, design=band)


def _pooled(key: str, family: Any, *, task: str, imputed: Mapping[str, Any], train_ids: Any,
            state: Any, sub: Any, ks: Sequence[float], kcal_per_unit: Mapping[str, float],
            nested: Mapping[str, str], total_energy: str | None, scale: str,
            percent: Sequence[str], n_boot: int, template: Any, y_fit: Any, group_of: Any,
            survey_design: Any, models: Sequence[str],
            progress: Callable[[float, str], None], seed: int = 0, designs: Any = None,
            cancelled: Callable[[], bool] | None = None) -> ClassDraw:
    """MS3: each completed copy's class curves through that copy's own fit, pooled at each k.

    Each copy's curves average over that copy's own rows (the curve's rows, or every row of a copy
    the data supplied; under the surveyed population every row of the copy in the design's domain,
    weighted). Each copy's within-copy variance at each k is its bootstrap's (``n_boot`` refits
    split over the copies; Schomaker & Heumann 2018's MI-then-bootstrap) or, under the surveyed
    population, its design-based linearization's, with the design's df as the complete-data df;
    Rubin's rules pool them (:func:`~turbotab.core.methods.substitution.pool_class_curves`)."""
    from sklearn.base import clone

    from turbotab.core.methods.substitution import (class_curves, class_refit_band,
                                                    design_class_curves, pool_class_curves)
    from turbotab.core.models.linear import model_matrix
    from turbotab.core.models.survey import SAMPLE_EXIT, domain_of, has_design_estimator
    from turbotab.core.stages.modeling import (BAND_ROWS, SUBSTITUTION_ROWS, fit_with,
                                               pinned_to_full_fit,
                                               shift_for)

    frames = list(imputed["frames"])
    given = list((imputed.get("fits") or {}).get(key) or [])
    outcomes = imputed.get("outcomes")
    m = len(given) or len(frames)
    supplied = outcomes is not None
    curve_args = dict(donor=sub.donor, recipient=sub.recipient, kcal_per_unit=kcal_per_unit,
                      ks=ks, total_kind="variable", nested=nested, total=total_energy,
                      scale=scale)

    def copy_fit(j: int, X_all: pd.DataFrame, y_all: Any) -> Any:
        """Copy j's fit: the table's own, or (a family with no table of its own) this family
        refit on the completed copy, as its single fit is on every analyzed row."""
        if given:
            return given[j]
        progress(0.6 * j / max(m, 1), f"{family.label}: fitting copy {j + 1} of {m}")
        units = (group_of.reindex(X_all.index).to_numpy() if group_of is not None
                 and not supplied else None)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return fit_with(clone(template), X_all, y_all, groups=units, designs=designs,
                            seed=seed, cancelled=cancelled)

    curve_ids = set(int(i) for i in np.asarray(train_ids))
    curves: list[dict[str, Any]] = []
    shifts, rows_of, ys, fits = [], [], [], []
    classes: list[Any] | None = None
    design_bands: list[dict[str, Any]] = []
    started = time.perf_counter()
    for j, X_all in enumerate(frames[:m]):
        progress(0.6 * j / max(m, 1), f"{family.label}: copy {j + 1} of {m}")
        y_all = np.asarray(outcomes[j]) if outcomes is not None else np.asarray(y_fit)
        if survey_design is not None:
            domain = domain_of(X_all.index, survey_design)
            rows = list(np.flatnonzero(domain.keep))
            X_k = X_all.iloc[rows]
            shift = shift_for(state, X_k, donor=sub.donor, recipient=sub.recipient,
                              kcal_per_unit=kcal_per_unit, design_nested=nested,
                              total=total_energy, scale=scale, percent=percent)
            if not has_design_estimator(family, task):
                refused, exits = population_blocked(family, task, models)
                support = _support_only(X_k, domain.raw, {**curve_args, "shift": shift})
                return ClassDraw(entries=[_blocked_entry(key, family, len(ks), refused, exits)],
                                 curve=support, m=m)
        fitted = copy_fit(j, X_all, y_all)
        if classes is None:
            classes = list(fitted.classes_)
        elif list(fitted.classes_) != classes:
            raise ValueError("The imputed copies' fits do not hold the same classes.")
        fits.append(fitted)
        if survey_design is not None:
            try:
                drawn = design_class_curves(lambda frame, _f=fitted: model_matrix(_f, frame), X_k,
                                            y_all[rows], classes, survey_design, domain,
                                            shift=shift, **curve_args)
            except ValueError as exc:
                support = _support_only(X_k, domain.raw, {**curve_args, "shift": shift})
                exits = [{"label": SAMPLE_EXIT,
                          "decision": {"kind": "set_survey", "estimand": "sample"}}]
                return ClassDraw(entries=[_blocked_entry(key, family, len(ks),
                                                         f"No design-based curves: {exc}", exits)],
                                 curve=support, m=m)
            curves.append(drawn["curves"])
            design_bands.append(drawn["band"])
        else:
            rows = [i for i, rid in enumerate(X_all.index) if int(rid) in curve_ids]
            if outcomes is not None or not rows:  # a supplied copy: its own rows
                rows = list(range(min(len(X_all), SUBSTITUTION_ROWS)))
            X_k = X_all.iloc[rows]
            shift = shift_for(state, X_k, donor=sub.donor, recipient=sub.recipient,
                              kcal_per_unit=kcal_per_unit, design_nested=nested,
                              total=total_energy, scale=scale, percent=percent)
            predict = class_predictor(fitted, X_all, y_all, classes, family)
            curves.append(class_curves(predict, X_k, classes=classes, shift=shift, **curve_args))
        shifts.append(shift)
        rows_of.append(rows)
        ys.append(y_all)
    n_k = len(curves[0]["ks"])
    live = [all(c["classes"][0]["delta"][i] is not None for c in curves) for i in range(n_k)]
    se = fixed_se = None
    df_com = None
    band: dict[str, Any] | None = None
    if design_bands:
        se = [[c["se"] for c in b["classes"]] for b in design_bands]
        fixed_se = [[c["fixed_se"] for c in b["classes"]] for b in design_bands]
        df_com = float(min(b["df"] for b in design_bands))
    elif n_boot and template is not None:
        per_copy = max(10, math.ceil(n_boot / max(m, 1)))
        se, fixed_se = [], []
        n_ok = failed = 0
        for j, (X_all, fitted) in enumerate(zip(frames, fits)):
            progress(0.6 + 0.4 * j / max(m, 1), f"{family.label}: copy {j + 1}'s refits")
            supplied = outcomes is not None
            unit = (group_of.reindex(X_all.index).to_numpy() if group_of is not None
                    and not supplied else None)

            def refit(Xb: pd.DataFrame, yb: Any, _full: Any = fitted) -> Any:
                inner = (group_of.reindex(Xb.index).to_numpy() if group_of is not None
                         and not supplied else Xb.index.to_numpy())
                pipe = pinned_to_full_fit(clone(template), _full)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    refitted = fit_with(pipe, Xb, yb, groups=inner, designs=designs,
                                        seed=seed, cancelled=cancelled)
                return class_predictor(refitted, Xb, yb, classes, family)

            drawn = class_refit_band(
                refit, X_all, ys[j], classes=classes, shift=shifts[j], ks=ks, live=live,
                n_boot=per_copy, random_state=j,
                centers=[c["delta"] for c in curves[j]["classes"]],
                fixed_centers=[c["fixed_delta"] for c in curves[j]["classes"]],
                curve_rows=rows_of[j], max_rows=BAND_ROWS, interval="normal", groups=unit)
            n_ok += drawn["n_ok"]
            failed += drawn["failed"]
            se.append([c["se"] for c in drawn["classes"]])
            fixed_se.append([c["fixed_se"] for c in drawn["classes"]])
        band = {"per_copy": per_copy, "n_ok": n_ok, "failed": failed}
    pooled = pool_class_curves(curves, se, fixed_se, df_com)
    first = curves[0]
    chosen = first["label_k"]
    ks_out = [float(k) for k in first["ks"]]
    stops = [c["stopped_at"] for c in curves if c["stopped_at"] is not None]
    on_support = [float(np.mean([c["on_support_fraction"][i] for c in curves])) for i in range(n_k)]
    banded = se is not None
    entries = []
    for c, mine in zip(first["classes"], pooled["classes"]):
        value = (mine["delta"][ks_out.index(chosen)] if chosen is not None and chosen in ks_out
                 else None)
        entries.append({
            "family": key, "label": f"{family.label}: {c['name']}", "level": c["name"],
            "delta": mine["delta"], "ci_low": mine["ci_low"] if banded else None,
            "ci_high": mine["ci_high"] if banded else None, "on_support_fraction": on_support,
            "stopped_at": min(stops) if stops else None,
            "effect_label": _label(value, chosen, c["name"], scale),
            "fixed_delta": mine["fixed_delta"],
            "fixed_ci_low": mine["fixed_ci_low"] if banded else None,
            "fixed_ci_high": mine["fixed_ci_high"] if banded else None,
            "band_ok": band["n_ok"] if band else None, "pooled": "per_k",
            "df": mine["df"] if banded else None})
    draw = ClassDraw(entries=entries, curve=first, m=m,
                     seconds=round(time.perf_counter() - started, 3),
                     design=design_bands[0] if design_bands else None)
    if band is not None:
        draw.failed = band["failed"]
    return draw


def pooled_note(m: int, n_boot: int, supplied: bool, design: bool = False) -> str:
    """The substitution note's sentence on how a multiclass outcome's curves were pooled over the
    copies (MS3); with ``design``, each copy's curves and variances are the surveyed population's
    (MS4)."""
    what = f"the data's {m} imputed copies" if supplied else f"the {m} imputations"
    if design:
        how = ("each copy's survey-weighted class curves over the surveyed population, pooled at "
               "each k by Rubin's rules with each copy's Taylor-linearized variance as its "
               "within-copy variance and the design's degrees of freedom as the complete-data df")
    elif n_boot:
        how = ("each copy's class curves, pooled at each k by Rubin's rules with each copy's "
               f"bootstrap variance as its within-copy variance ({n_boot:,} refits split over the "
               "copies; Schomaker & Heumann 2018)")
    else:
        how = ("the mean at each k of each copy's class curves, with no band until one is asked "
               "for")
    return f"Each class's curve is pooled over {what}: {how}."


__all__ = ["ClassDraw", "POLISH_AGREEMENT", "class_family_entries", "class_predictor",
           "pooled_note", "population_blocked"]
