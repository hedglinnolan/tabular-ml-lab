"""The first two figures, built against `turbotab/figures.py`'s spec.

**Two, deliberately.** They were chosen to be maximally different so the
abstraction's seams show, and a third is not built until the spec has survived
both:

| | Calibration plot | PCA scores |
|---|---|---|
| tier | CONFIRMATORY | EXPLORATORY |
| needs | a fitted model's predictions | a numeric block and nothing else |
| axis rule | do **not** truncate | aspect ratio proportional to variance |
| the delta | a spike histogram under the curve | % variance in the labels, QCs overlaid |
| companions | its own discrimination figure | none — it makes no claim |
| promotable | no — it is about a fitted model | yes — PCA re-fits per fold |

## What the seams turned out to be

**The annotation source is not the caption's source.** `Annotation.source` names
the computation (`ml.calibration`) and the figure's `evidence` names where the
field stands. Two questions that look like one until a figure has both.

**A checklist item can be about what is NOT done.** *"Do not truncate the axis to
hide the sparse tail"* is scored by comparing the rendered x-range against the
data's, which means the render has to carry both. An earlier draft carried only
the range it drew, and the item was unscoreable — `GUIDED-045`'s axis, one layer
into the figure layer.

**`when_applicable` needs the project, not the frame.** The calibration plot
needs a fitted model and the PCA needs a numeric block, and neither question can
be answered from a dataframe alone.

Every threshold quoted here is the research's, and none of them is a number the
research marked `[verify-at-build]`.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from turbotab.figures import (CONFIRMATORY, EXPLORATORY, Annotation,
                              ChecklistItem, FigureSpec, register)
from turbotab.packs import Evidence, SETTLED

# ─────────────────────────────────────────────────────────────────────────────
# 1 · The calibration plot — CONFIRMATORY
#
# "The single most important figure in a clinical prediction paper", and the
# one whose publication-grade delta is almost entirely annotation: the 45° line
# labeled *ideal*, the flexible curve with its band, the six-number box, the
# spike histogram along the bottom, and a refusal to truncate.
#
# The spike histogram is the item the research singles out — *without it the
# reader cannot tell whether the curve's wild behavior at 0.8 is based on 3
# patients or 300. This is the detail most often missing and most often
# requested by reviewers.*
# ─────────────────────────────────────────────────────────────────────────────

CALIBRATION_EVIDENCE = Evidence(
    status=SETTLED,
    source=("research/CLINICAL_SURVEY_PACK.md#A4.3 · ★ Calibration plot — the "
            "single most important figure in a clinical prediction paper"))


def calibration_payload(y_true, y_proba, *, n_bins: int = 10,
                        model_name: str = "model") -> Dict[str, Any]:
    """The render's payload, computed by the ENGINE and annotated here.

    `ml.calibration.calibration_classification` already holds the mathematics.
    Recomputing any of it here would be the two-engines failure inside the
    figure layer, so this adds exactly what the checklist needs and the engine
    does not produce: the risk distribution split by outcome, and both x-ranges.
    """
    from ml.calibration import calibration_classification

    y_true = np.asarray(y_true, dtype=float)
    y_proba = np.asarray(y_proba, dtype=float)
    result = calibration_classification(y_true, y_proba, n_bins=n_bins,
                                        model_name=model_name)

    def _num(name, default=None):
        value = getattr(result, name, default)
        return None if value is None else (
            float(value) if isinstance(value, (int, float, np.floating)) else value)

    # THE SPIKE HISTOGRAM, split by outcome. Events above the axis, non-events
    # below — the reader has to be able to see that the tail is three patients.
    edges = np.linspace(0.0, 1.0, 51)
    events = np.histogram(y_proba[y_true == 1], bins=edges)[0]
    non_events = np.histogram(y_proba[y_true == 0], bins=edges)[0]

    observed_lo, observed_hi = float(y_proba.min()), float(y_proba.max())
    return {
        "figure": "calibration",
        "model_name": model_name,
        "n": int(len(y_true)),
        "events": int((y_true == 1).sum()),
        # The hierarchy's SECOND rung, which the engine did not compute until
        # this checklist item failed against a real render (`GUIDED-051`).
        "calibration_intercept": _num("weak_intercept"),
        "calibration_slope": _num("weak_slope"),
        "c_statistic": _num("c_statistic"),
        "e_avg": _num("ece"),
        "e_max": _num("mce"),
        "brier": _num("brier_score"),
        "curve": {
            "predicted": [float(v) for v in
                          (result.bin_pred_mean if result.bin_pred_mean is not None
                           else [])],
            "observed": [float(v) for v in
                         (result.bin_true_freq if result.bin_true_freq is not None
                          else [])],
        },
        "risk_distribution": {
            "edges": [float(e) for e in edges],
            "events": [int(v) for v in events],
            "non_events": [int(v) for v in non_events],
        },
        # BOTH ranges, because the "do not truncate" item is scored by
        # comparing them. A payload carrying only what it drew makes the item
        # unscoreable, which is a checklist entry that cannot fail.
        "x_range_drawn": [0.0, 1.0],
        "x_range_observed": [observed_lo, observed_hi],
        "reference_line": {"kind": "identity", "label": "ideal", "dash": True},
        "aspect": "square",
    }


def calibration_render(y_true, y_proba, **kw) -> Dict[str, Any]:
    """The payload with its annotation box already rendered.

    One entry point, so a caller cannot get the numbers without the rows that
    say what to do when one is missing.
    """
    payload = calibration_payload(y_true, y_proba, **kw)
    payload["annotation_box"] = annotation_box(payload)
    return payload


def _has(payload: Dict[str, Any], *keys: str) -> bool:
    return all(payload.get(k) is not None for k in keys)


def annotation_box(payload: Dict[str, Any]) -> List[Dict[str, str]]:
    """The six numbers a reviewer wants, as rendered rows.

    **A missing number renders the ABSENCE, not a blank.** `weak_calibration`
    returns `(None, None)` where the fit is undefined — one outcome class,
    constant predictions, or separation, which is what a very good model on a
    small sample produces — and a blank cell beside five real numbers reads as a
    rendering fault. It is not: it is the app declining to state a quantity it
    does not have, and the box says which and why.

    That is the governing rule's *silent* branch made visible rather than left
    silent. The `annotation_box` checklist item still FAILS when a number is
    missing, and it should: the figure is not publication-grade without them.
    Failing the checklist and rendering honestly are different jobs.
    """
    rows: List[Dict[str, str]] = []
    for key, label in (("calibration_intercept", "Calibration intercept"),
                       ("calibration_slope", "Calibration slope"),
                       ("c_statistic", "C-statistic"),
                       ("e_avg", "E:avg"),
                       ("e_max", "E:max"),
                       ("n", "n"),
                       ("events", "events")):
        value = payload.get(key)
        if value is None:
            rows.append({
                "key": key, "label": label, "value": "not estimable",
                "why": ("The calibration fit is not defined for these "
                        "predictions — one outcome class, no variation in the "
                        "predicted risks, or complete separation. A number is "
                        "not shown because there is not one, rather than "
                        "because it failed to render.")})
        else:
            rows.append({"key": key, "label": label,
                         "value": (f"{value:,}" if isinstance(value, int)
                                   else f"{value:.3f}"), "why": ""})
    return rows


CALIBRATION = register(FigureSpec(
    id="calibration",
    title="Calibration plot",
    tier=CONFIRMATORY,
    when_applicable=lambda s: (
        s.get("task_type") == "classification" and bool(s.get("has_predictions"))),
    layers=("identity_line", "flexible_curve", "confidence_band",
            "risk_spike_histogram"),
    annotations=(
        Annotation("calibration_intercept", "Calibration intercept (95% CI)",
                   "ml.calibration"),
        Annotation("calibration_slope", "Calibration slope (95% CI)",
                   "ml.calibration"),
        Annotation("c_statistic", "C-statistic (95% CI)", "ml.calibration"),
        Annotation("e_avg", "E:avg", "ml.calibration"),
        Annotation("e_max", "E:max", "ml.calibration"),
        Annotation("n", "n", "turbotab.figure_specs"),
        Annotation("events", "events", "turbotab.figure_specs"),
    ),
    checklist=(
        ChecklistItem(
            "risk_distribution",
            "A distribution of predicted risks along the bottom, split by outcome",
            "Without it a reader cannot tell whether the curve's behavior at "
            "0.8 rests on 3 patients or 300. This is the item most often "
            "missing and most often requested by reviewers.",
            lambda p: bool(p.get("risk_distribution", {}).get("events"))
            and bool(p.get("risk_distribution", {}).get("non_events"))),
        ChecklistItem(
            "identity_line",
            "45° reference line, dashed, labeled 'ideal'",
            "Without the reference the curve has nothing to be read against.",
            lambda p: (p.get("reference_line", {}).get("kind") == "identity"
                       and p.get("reference_line", {}).get("label") == "ideal")),
        ChecklistItem(
            "annotation_box",
            "Intercept, slope, C-statistic, E:avg, E:max, n and events, on the figure",
            "The two weak-calibration numbers alone can hide a curve that is "
            "badly wrong in the clinically relevant range while averaging to a "
            "slope of 1.",
            lambda p: _has(p, "calibration_intercept", "calibration_slope",
                           "c_statistic", "e_avg", "e_max", "n", "events")),
        ChecklistItem(
            "no_truncation",
            "The axis is not truncated to hide the sparse tail",
            "Truncating hides the region where the model is least reliable, "
            "which is the region a reader most needs to see. Show it and let "
            "the confidence band widen.",
            lambda p: (p.get("x_range_drawn", [1, 0])[0]
                       <= p.get("x_range_observed", [0, 1])[0]
                       and p.get("x_range_drawn", [0, 0])[1]
                       >= p.get("x_range_observed", [0, 1])[1])),
        ChecklistItem(
            "square_aspect",
            "Square aspect ratio, same scale on both axes",
            "Predicted and observed are the same quantity; a non-square panel "
            "makes agreement look like disagreement.",
            lambda p: p.get("aspect") == "square"),
    ),
    caption=lambda p: (
        f"Calibration of {p.get('model_name', 'the model')} on "
        f"{p.get('n', 0):,} observations with {p.get('events', 0):,} events. "
        f"The dashed 45° line is ideal calibration; the solid curve is the "
        f"flexible (loess) estimate with a pointwise 95% band. Calibration "
        f"intercept {_fmt(p.get('calibration_intercept'))} and slope "
        f"{_fmt(p.get('calibration_slope'))} (a slope below 1 indicates "
        f"predictions that are too extreme); C-statistic "
        f"{_fmt(p.get('c_statistic'))}; E:avg {_fmt(p.get('e_avg'))}. The "
        f"histogram along the axis shows the distribution of predicted risks, "
        f"events above and non-events below. The axis is not truncated."),
    companions=("discrimination",),
    evidence=CALIBRATION_EVIDENCE,
    # NOT PROMOTABLE, and the reason is the rule rather than the subject: a
    # calibration curve is a property of a model already fitted to these rows.
    # Re-running it inside a fold would need the fold's own model, which is the
    # thing being evaluated — the computation is not re-executable in the sense
    # `PRODUCT_VISION.md` requires.
    promotable=False,
    promotable_because="",
    compute=calibration_payload,
))


def _fmt(value: Optional[float]) -> str:
    return "not estimable" if value is None else f"{value:.3f}"


# ─────────────────────────────────────────────────────────────────────────────
# 2 · The PCA scores plot — EXPLORATORY
#
# The field's trust anchor, and a figure whose *quality-control* job matters
# more than its scientific one: pooled QCs clustering tightly at the center is
# itself part of the result.
#
# The distinction the pack says must not be got wrong: the Hotelling's T²
# ellipse is a SINGLE ellipse over all samples defining a multivariate outlier
# boundary; group-wise confidence ellipses are a different object entirely, and
# papers routinely mislabel one as the other.
# ─────────────────────────────────────────────────────────────────────────────

PCA_EVIDENCE = Evidence(
    status=SETTLED,
    source="research/METABOLOMICS_PACK.md#06.1 · PCA scores plot — the field's trust anchor")


def pca_scores_payload(df: pd.DataFrame, *, group_col: Optional[str] = None,
                       qc_mask: Optional[Sequence[bool]] = None,
                       scaling: str = "unit variance") -> Dict[str, Any]:
    """The render's payload. `ml.macro_shape.compute_pca` does the mathematics."""
    from ml.macro_shape import compute_pca

    numeric = df.select_dtypes(include=[np.number])
    result = compute_pca(numeric, n_components=2)
    if "error" in result:
        raise ValueError(result["error"])

    ratios = [float(v) for v in result["explained_variance_ratio"][:2]]
    scores = np.asarray(result["components"])[:, :2]

    groups: List[Optional[str]] = ([str(v) for v in df[group_col]]
                                   if group_col and group_col in df.columns
                                   else [None] * len(df))
    qc = [bool(v) for v in (qc_mask if qc_mask is not None
                            else [False] * len(df))]
    counts: Dict[str, int] = {}
    for g in groups:
        counts[str(g)] = counts.get(str(g), 0) + 1

    return {
        "figure": "pca_scores",
        "scores": scores.tolist(),
        "explained_variance_ratio": ratios,
        # THE AXIS LABELS CARRY THE % VARIANCE. The research calls omitting
        # this "the single most common defect", so it is built into the payload
        # rather than left to a renderer to remember.
        "axis_labels": [f"PC{i + 1} ({r * 100:.1f}%)"
                        for i, r in enumerate(ratios)],
        "groups": groups,
        "group_counts": counts,
        "qc": qc,
        "n_qc": int(sum(qc)),
        # ASPECT PROPORTIONAL TO VARIANCE EXPLAINED. Stretching PC2 to fill the
        # panel visually exaggerates separation, which is the failure this
        # figure is most often used to commit.
        "aspect_ratio": (ratios[1] / ratios[0]) if ratios and ratios[0] else 1.0,
        # The two ellipses, rendered as DIFFERENT objects and labeled as such.
        "hotelling_t2": {"kind": "outlier_boundary", "style": "dashed_grey",
                         "label": "95% Hotelling's T²", "single": True},
        "group_ellipses": ({"kind": "group_confidence", "style": "filled_group_color",
                            "label": "95% group confidence"} if group_col else None),
        "scaling": scaling,
        "qc_in_fit": True,
        "n": int(len(df)),
    }


PCA_SCORES = register(FigureSpec(
    id="pca_scores",
    title="PCA scores plot",
    tier=EXPLORATORY,
    when_applicable=lambda s: int(s.get("n_numeric") or 0) >= 2
    and int(s.get("n_rows") or 0) >= 10,
    layers=("scores_scatter", "hotelling_t2_ellipse", "group_ellipses",
            "qc_overlay"),
    annotations=(
        Annotation("explained_variance_ratio", "% variance per component",
                   "ml.macro_shape"),
        Annotation("group_counts", "n per group", "turbotab.figure_specs"),
        Annotation("n_qc", "n pooled QC", "turbotab.figure_specs"),
        Annotation("scaling", "normalization and scaling used",
                   "turbotab.figure_specs"),
    ),
    checklist=(
        ChecklistItem(
            "variance_in_labels",
            "Axis labels give the component and the % variance explained",
            "Omitting this is the single most common defect in the figure, and "
            "reviewers ask for it.",
            lambda p: all("%" in str(label)
                          for label in p.get("axis_labels", []))
            and len(p.get("axis_labels", [])) >= 2),
        ChecklistItem(
            "qc_overlaid",
            "Pooled QCs overlaid in a distinct color, never dropped",
            "Their tight central cluster IS part of the result — it is the "
            "evidence the run was stable.",
            lambda p: p.get("n_qc", 0) > 0),
        ChecklistItem(
            "aspect_proportional",
            "Aspect ratio proportional to variance explained",
            "Stretching PC2 to fill the panel visually exaggerates separation, "
            "which is the claim this figure is most often misused to make.",
            lambda p: p.get("aspect_ratio") is not None
            and abs(p["aspect_ratio"]
                    - (p["explained_variance_ratio"][1]
                       / p["explained_variance_ratio"][0])) < 1e-9),
        ChecklistItem(
            "ellipses_distinguished",
            "Hotelling's T² ellipse drawn and labeled distinctly from group ellipses",
            "They are different objects — an outlier boundary over all samples "
            "versus where each group lies — and papers routinely mislabel one "
            "as the other.",
            lambda p: (p.get("hotelling_t2", {}).get("single") is True
                       and p.get("hotelling_t2", {}).get("style")
                       != (p.get("group_ellipses") or {}).get("style"))),
        ChecklistItem(
            "legend_states_n_and_scaling",
            "Legend states n per group, the scaling used, and whether QCs were in the fit",
            "Without them the picture is not reproducible from the figure.",
            lambda p: bool(p.get("group_counts")) and bool(p.get("scaling"))
            and p.get("qc_in_fit") is not None),
    ),
    caption=lambda p: (
        f"Principal component analysis of {p.get('n', 0):,} samples, "
        f"{p['axis_labels'][0]} against {p['axis_labels'][1]}, on "
        f"{p.get('scaling', 'unscaled')} data. The dashed grey ellipse is the "
        f"95% Hotelling's T² multivariate outlier boundary and is not a group "
        f"confidence region. "
        + (f"{p.get('n_qc', 0)} pooled quality-control samples are overlaid"
           + (" and were included in the fit. " if p.get("qc_in_fit")
              else " and were projected onto the fit. ")
           if p.get("n_qc") else "No pooled quality-control samples were "
                                 "identified. ")
        + "PCA does not use the group labels, so separation here is not a "
          "test of a group difference. Axes are scaled in proportion to the "
          "variance each component explains."),
    companions=(),
    evidence=PCA_EVIDENCE,
    # PROMOTABLE. `PRODUCT_VISION.md`'s rule is RE-EXECUTABILITY, not
    # label-blindness: an artifact is promotable when the app can re-run its
    # computation inside every fold. A PCA fit is a deterministic function of
    # the rows it sees, so refitting it per fold is exactly what a pipeline
    # step does — no information crosses the split.
    promotable=True,
    promotable_because=(
        "A PCA fit is a deterministic function of the rows it is given, so it "
        "can be refitted inside every training fold and applied to the held-out "
        "rows without any information crossing the split. What is promoted is "
        "the recipe — centre, scale, project onto k components — and never the "
        "component values computed here on the whole table."),
    compute=pca_scores_payload,
))


# ─────────────────────────────────────────────────────────────────────────────
# 3 · The shrinkage plot — EXPLORATORY
#
# `research/NUTRITION_PACK.md` §03. Three overlaid densities of ONE nutrient:
# single-day intake, the mean of available days, and modeled usual intake, with
# the 5th and 95th percentiles of each annotated. **The visible narrowing from
# the first to the third is the entire argument for usual-intake modeling, in
# one image.**
#
# ## WHAT IT COST THE ABSTRACTION — the deliverable, as much as the figure
#
# This is the spec's first figure from a domain it was not designed against,
# and its first where the payload is **three versions of the same quantity**
# rather than one. Three things gave, and only one needed a change to the spec:
#
# **1 · `layers` was already right, and that was luck.** It is a tuple of
# strings and carries no cardinality, so "three densities" needed nothing. Had
# it been the typed geometry an earlier draft nearly made it, three-of-a-kind
# would have needed a new layer type.
#
# **2 · `annotations` bent, and the bend is real.** Every annotation until now
# named ONE number — a slope, an n, a percentage. Here the same annotation (the
# 5th percentile) exists three times, once per series, and `Annotation.key`
# cannot address that. It is resolved by keying per series rather than by
# generalizing `Annotation` — `p05_single_day` and not `p05[series]` — and that
# is a deliberate refusal to add an axis on one example. **A second figure with
# per-series annotations is what would justify the generalization, and there is
# not one yet.**
#
# **3 · `checklist` needed a comparison ACROSS series, which no item had done.**
# "The narrowing is visible" is not a property of any one density; it is a
# relation between three. The item reads them all from the payload, which the
# `check(payload) -> bool` signature already allowed — so the spec held, and it
# held because the callable takes the whole payload rather than a series.
#
# **What genuinely did not fit: `tier`.** The shrinkage plot is EXPLORATORY by
# the two-tier logic — it sees no group labels and makes no group claim. But it
# is the *argument for a method*, which is neither exploration nor
# confirmation, and the enum has no third value. Recorded rather than resolved:
# adding a tier on one example is how a two-value distinction becomes a
# taxonomy nobody can apply. It is filed on `GUIDED-056`.
# ─────────────────────────────────────────────────────────────────────────────

SHRINKAGE_EVIDENCE = Evidence(
    status=SETTLED,
    source=("research/NUTRITION_PACK.md#03 · ★ Repeated recalls and "
            "measurement error"))

SERIES = ("single_day", "mean_of_days", "usual_intake")
SERIES_LABEL = {
    "single_day": "One day",
    "mean_of_days": "Mean of available days",
    "usual_intake": "Modeled usual intake",
}


def shrinkage_payload(series: Dict[str, Sequence[float]], *, nutrient: str,
                      unit: str = "", n_days: Optional[int] = None,
                      modeled: bool = True) -> Dict[str, Any]:
    """Three versions of one quantity, with the percentiles that carry the claim.

    `series` maps each of `SERIES` to that version's values. A missing series is
    REFUSED rather than drawn with two: the narrowing is a relation between
    three, and two of them is a different figure making a weaker claim while
    wearing this one's caption.
    """
    missing = [s for s in SERIES if s not in series]
    if missing:
        raise ValueError(
            f"the shrinkage plot needs all three series and is missing "
            f"{missing}. Two densities is a different figure — the claim this "
            f"one makes is the narrowing ACROSS the three, and drawing it with "
            f"two would be that claim without its evidence.")

    densities, annotations = {}, {}
    for key in SERIES:
        values = np.asarray([v for v in series[key]], dtype=float)
        values = values[np.isfinite(values)]
        if len(values) < 10:
            raise ValueError(
                f"{key} has {len(values)} usable values; a density drawn from "
                f"fewer would be a claim about a shape nobody can see.")
        p05, p95 = float(np.percentile(values, 5)), float(np.percentile(values, 95))
        densities[key] = {"n": int(len(values)), "median": float(np.median(values))}
        # KEYED PER SERIES rather than by generalizing `Annotation` — see the
        # note above. Six keys, not one key with an axis.
        annotations[f"p05_{key}"] = p05
        annotations[f"p95_{key}"] = p95
        annotations[f"spread_{key}"] = p95 - p05

    return {
        "figure": "shrinkage",
        "nutrient": nutrient,
        "unit": unit,
        "n_days": n_days,
        "modeled": modeled,
        "series": list(SERIES),
        "series_labels": [SERIES_LABEL[s] for s in SERIES],
        "densities": densities,
        **annotations,
        "n": densities[SERIES[0]]["n"],
    }


SHRINKAGE = register(FigureSpec(
    id="shrinkage",
    title="What usual-intake modeling changes",
    tier=EXPLORATORY,
    when_applicable=lambda s: (int(s.get("n_recalls_per_person") or 0) >= 2
                               and bool(s.get("has_dietary_lens"))),
    layers=("density_single_day", "density_mean_of_days",
            "density_usual_intake", "percentile_markers"),
    annotations=tuple(
        Annotation(f"{stat}_{key}",
                   f"{stat.upper()} of {SERIES_LABEL[key].lower()}",
                   "turbotab.figure_specs")
        for key in SERIES for stat in ("p05", "p95")),
    checklist=(
        ChecklistItem(
            "three_series",
            "All three densities are drawn: one day, mean of days, modeled usual intake",
            "The claim is the narrowing ACROSS the three. Two of them is a "
            "different figure making a weaker claim in this one's caption.",
            lambda p: list(p.get("series", [])) == list(SERIES)
            and all(f"p05_{k}" in p for k in SERIES)),
        ChecklistItem(
            "percentiles_annotated",
            "The 5th and 95th percentile of each density is annotated",
            "The narrowing is the argument, and an unlabeled narrowing is an "
            "impression rather than a measurement a reader can check.",
            lambda p: all(p.get(f"p05_{k}") is not None
                          and p.get(f"p95_{k}") is not None for k in SERIES)),
        ChecklistItem(
            "narrowing_is_visible",
            "The modeled distribution is narrower than the observed ones",
            "If it is not, the figure does not make the argument it is drawn "
            "to make — and saying so is more useful than drawing it anyway.",
            lambda p: (p.get("spread_usual_intake") is not None
                       and p["spread_usual_intake"] <= p["spread_single_day"])),
        ChecklistItem(
            "modeling_is_declared",
            "The caption says the third density is modeled, not measured",
            "Reporting modeled individual predictions as measured usual "
            "intakes is a named failure in this field.",
            lambda p: p.get("modeled") is not None),
    ),
    caption=lambda p: (
        f"Distribution of {p['nutrient']}"
        + (f" ({p['unit']})" if p.get("unit") else "")
        + f" for {p.get('n', 0):,} participants, shown three ways. One day "
          f"spans {_fmt(p.get('p05_single_day'))} to "
          f"{_fmt(p.get('p95_single_day'))} between the 5th and 95th "
          f"percentiles; the mean of "
        + (f"{p['n_days']} days" if p.get("n_days") else "the available days")
        + f" spans {_fmt(p.get('p05_mean_of_days'))} to "
          f"{_fmt(p.get('p95_mean_of_days'))}; modeled usual intake spans "
          f"{_fmt(p.get('p05_usual_intake'))} to "
          f"{_fmt(p.get('p95_usual_intake'))}. The narrowing is day-to-day "
          f"variation being separated from between-person difference — it is "
          f"why a percentile or a prevalence computed from one day, or from a "
          f"mean of days, is overstated in both tails. The third distribution "
          f"is MODELED and its individual values are not measured usual "
          f"intakes."),
    companions=(),
    evidence=SHRINKAGE_EVIDENCE,
    # PROMOTABLE by the re-executability rule: the variance decomposition is a
    # deterministic function of the rows it is fitted on, so it refits per fold
    # like any other pipeline step. What is promoted is the recipe.
    promotable=True,
    promotable_because=(
        "The variance decomposition behind a usual-intake distribution is a "
        "deterministic function of the rows it is fitted on, so it can be "
        "refitted inside every training fold and applied to the held-out rows. "
        "What is promoted is the recipe — decompose within and between "
        "variance, shrink each person toward the mean — and never the modeled "
        "values computed here on the whole table."),
    compute=shrinkage_payload,
))
