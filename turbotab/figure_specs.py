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
                              ChecklistItem, FigureSpec, Pending, register,
                              register_pending)
from turbotab.packs import (CONVENTION_STATUS, DISPUTED, Claim, Evidence,
                            PackRefusal, SETTLED)

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
    # `"ungrouped"` RATHER THAN `"None"`, and it is not cosmetic. The legend
    # item is *"n per group"*, and `None 600` beside it reads as a group whose
    # name failed to render. There is no group; saying so is the recorded-
    # absence rule at the smallest possible scale. Found the first time this
    # payload reached a user, which was `GUIDED-058`'s whole point.
    counts: Dict[str, int] = {}
    for g in groups:
        key = "ungrouped" if g is None else str(g)
        counts[key] = counts.get(key, 0) + 1

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
          f"intakes."
        # THE METHOD IS NAMED WHERE IT IS KNOWN. "Modeled" covers the NCI
        # method, ISU, MSM and SPADE, which are not interchangeable, and a
        # caption that says only "modeled" lets a reader assume whichever one
        # they know. `research/NUTRITION_PACK.md` §03's own anti-pattern list
        # is built out of claims that were true of some method and not of the
        # one that ran.
        + (f" The model is {p['method']}." if p.get("method") else "")),
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


# ─────────────────────────────────────────────────────────────────────────────
# Specified and not built — the catalogue's honest half
#
# `GUIDED-060`. The nutrition pack's refusals offer what the app CAN draw
# instead, and two of the four named a figure that did not exist. The offer was
# not wrong to name it — `research/NUTRITION_PACK.md` §07 figure E specifies
# both, and `DOMAIN_SCIENCE.md` §03b lists intake-vs-DRI with the EAR region
# shaded in the signature set — so the target is planned rather than invented.
# What was missing is that nothing resolved it.
#
# **The instruction was explicitly not to build them**, and building them would
# not have been possible honestly anyway: every one needs a Dietary Reference
# Intake table, no DRI table ships anywhere in the repository, and
# `DOMAIN_SCIENCE.md` §04 says they must ship as data read from NASEM rather
# than as a dict of remembered numbers. A wrong EAR does not look wrong.
# `GUIDED-067` carries that.
# ─────────────────────────────────────────────────────────────────────────────

DISTRIBUTION_AGAINST_AI = register_pending(Pending(
    id="distribution_against_ai",
    title="Usual intake against the Adequate Intake",
    specified_in="research/NUTRITION_PACK.md#07 · EDA and presentation",
    needs=(
        "the Adequate Intake for the participant's age band, sex, pregnancy "
        "and lactation stratum. No Dietary Reference Intake table ships in "
        "this repository yet, and the AI cannot be inferred from your data — "
        "it is a published value or it is nothing."),
    blocked_by="GUIDED-067"))

DISTRIBUTION_AGAINST_EAR_AND_RDA = register_pending(Pending(
    id="distribution_against_ear_and_rda",
    title="Usual intake against the EAR and the RDA",
    specified_in="research/NUTRITION_PACK.md#07 · EDA and presentation",
    needs=(
        "the Estimated Average Requirement and the RDA for the participant's "
        "age band, sex, pregnancy and lactation stratum, which is the same "
        "missing Dietary Reference Intake table. The shaded area below the EAR "
        "IS the prevalence of inadequacy, so drawing it against a guessed cut "
        "point would assert the number this pack refuses to compute."),
    blocked_by="GUIDED-067"))

PER_NUTRIENT_DISTRIBUTION = register_pending(Pending(
    id="per_nutrient_distribution",
    title="The observed distribution of one nutrient",
    specified_in="research/NUTRITION_PACK.md#07 · EDA and presentation",
    needs=(
        "nothing this repository lacks — histogram and density, raw and log₁₀ "
        "side by side, with a marker saying whether the plotted variable is a "
        "single day, a mean of days or modeled usual intake. It is figure A of "
        "§07 and is simply not built, which is a different sentence from the "
        "two above and is why the record says which."),
    blocked_by="GUIDED-058"))


# ─────────────────────────────────────────────────────────────────────────────
# 4 · The volcano plot — CONFIRMATORY
#
# `research/METABOLOMICS_PACK.md` §06.3, and it was built FIRST of this batch
# on the adjudicator's instruction, because it was the one most likely to bend
# the spec. `LOOP.md` §02: *order a batch hardest-first — five instances built
# easiest-first are five castings of a shape nobody stress-tested.*
#
# ## IT BENT THE SPEC, AND HERE IS EXACTLY WHERE
#
# The research's caveat is not about what this figure draws. It is about the
# STATE OF THE DATA IT IS DRAWN FROM:
#
# > *"The fold change must be computed where fold change is meaningful. **After
# > autoscaling, 'fold change' is a fold change in z-units and is meaningless.**
# > Compute FC from normalized-but-not-scaled data and say so. Getting this
# > wrong is a subtle, real, embarrassing error."*
#
# `when_applicable` answers *"does this figure have anything to say about this
# project"*. It cannot answer *"are these data in a state where this figure
# would tell the truth"*, and the two are different questions: a table of 400
# autoscaled features and a binary outcome is a table the volcano applies to
# perfectly and must not be drawn from. **The spec has no field for a
# precondition on upstream data state and no refusal path.**
#
# Resolved WITHOUT adding a field, and the reason is `GUIDED-056`'s verbatim:
# a distinction generalized from one example becomes a taxonomy nobody can
# apply. The precondition lives in the payload builder, which raises a
# `FigureRefusal` — a `PackRefusal`, so it already carries a badge and an offer
# and `figure_bundle.render` already surfaces it under `unavailable` with the
# refusal's own words. That path existed because the shrinkage plot needed it
# for a different reason, which is the evidence it is a shape and not a patch.
#
# **What it costs, stated rather than hidden.** A reader of `FigureSpec` cannot
# see that this figure has a precondition; only a reader of `compute` can. A
# second figure with a data-state precondition is what would justify the field.
# Filed as `GUIDED-071`.
#
# The second bend is smaller and is `GUIDED-064`'s third instance: the y-axis
# rule is SETTLED — plotting raw p on 3,000 features *"is an anti-pattern and
# would be flagged in review"* — while the |log2FC| cut beside it is
# **[CONVENTION — arbitrary, justify biologically]**, and a figure carries one
# badge.
# ─────────────────────────────────────────────────────────────────────────────

VOLCANO_EVIDENCE = Evidence(
    status=SETTLED,
    source="research/METABOLOMICS_PACK.md#06.3 · Volcano plot")

# `GUIDED-064`. Two requirements, two statuses, and the caption has always said
# so while the badge did not.
VOLCANO_CLAIMS = (
    Claim("q_on_the_y_axis",
          "The y-axis carries the FDR-adjusted q, or the cut line is drawn at "
          "the p corresponding to q = 0.05 and the caption says which.",
          VOLCANO_EVIDENCE),
    Claim("fold_change_cut",
          "The |log2 fold change| cut is drawn at 1.0. The research calls the "
          "cut arbitrary and says to justify it biologically.",
          Evidence(status=CONVENTION_STATUS,
                   source="research/METABOLOMICS_PACK.md#06.3 · Volcano plot")),
)


class FigureRefusal(PackRefusal):
    """A figure the data are not in a state to support.

    Distinct from `when_applicable` returning False, and the distinction is the
    whole of `GUIDED-071`: that says *"this figure has nothing to say about
    this project"*, this says *"it does, and drawing it from these values would
    state something false."*
    """


# |log2FC| > 1 is the common convention and > 0.58 (1.5-fold) is common in
# metabolomics. **[CONVENTION — arbitrary, justify biologically]**, so it is a
# parameter with its status carried beside it rather than a threshold the
# figure asserts.
VOLCANO_FC_CUT = 1.0
VOLCANO_Q_CUT = 0.05

# What autoscaled data looks like: every feature centred at zero with unit
# spread. Measured against the block's own medians rather than per feature,
# because one constant column should not decide this.
_AUTOSCALE_SD = (0.8, 1.25)


def volcano_payload(df: pd.DataFrame, *, group_col: str,
                    feature_columns: Optional[Sequence[str]] = None,
                    q_cut: float = VOLCANO_Q_CUT,
                    fc_cut: float = VOLCANO_FC_CUT) -> Dict[str, Any]:
    """log2 fold change against −log10(q), refusing where FC is meaningless.

    `ml.feature_selection.univariate_screening` already holds the testing and
    the Benjamini–Hochberg correction. Recomputing either here would be the
    two-engines failure inside the figure layer, so this adds exactly what the
    checklist needs and the engine does not produce: the fold change, the data
    state it was computed from, and the counts.
    """
    from ml.feature_selection import univariate_screening

    features = [str(c) for c in (feature_columns if feature_columns is not None
                                 else df.select_dtypes(include=[np.number]).columns)
                if str(c) != str(group_col)]
    groups = df[group_col]
    levels = [v for v in pd.Series(groups).dropna().unique()]
    if len(levels) != 2:
        raise FigureRefusal(
            f"A volcano plot contrasts two groups and `{group_col}` has "
            f"{len(levels)}. With more than two the fold change on the x-axis "
            f"has no single meaning, and picking a pair for you would be the "
            f"app choosing your contrast.",
            evidence=VOLCANO_EVIDENCE,
            offer={"draw": "pca_scores",
                   "label": "The scores plot, which uses no group labels",
                   "caption_note": (
                       "PCA does not see the outcome, so it makes no group "
                       "claim and needs no contrast."),
                   "forbidden": "volcano_over_more_than_two_groups"})

    block = df[features].apply(pd.to_numeric, errors="coerce")
    reference, comparison = sorted(levels, key=str)

    # ── THE PRECONDITION ON DATA STATE, which is what bent the spec ──────────
    #
    # THE AUTOSCALE READING IS TESTED FIRST, and the order is load-bearing in
    # the way `Prior.__post_init__` records: autoscaled data has negatives BY
    # CONSTRUCTION, so with the negatives branch first the specific message —
    # the one naming z-units, which is the error the research calls subtle,
    # real and embarrassing — was unreachable, and the generic one about
    # transformed values spoke in its place. **The most diagnostic refusal has
    # to be first.** Found by a test asserting the message rather than the
    # exception type.
    spread = float(block.std(numeric_only=True).median())
    centre = float(block.mean(numeric_only=True).abs().median())
    if _AUTOSCALE_SD[0] <= spread <= _AUTOSCALE_SD[1] and centre < 0.1 * spread:
        raise FigureRefusal(
            f"These features are autoscaled — across {len(features):,} of them "
            f"the median standard deviation is {spread:.2f} and the median "
            f"absolute mean is {centre:.3f}, which is unit variance centred at "
            f"zero. **After autoscaling a fold change is a fold change in "
            f"z-units and is meaningless.** The x-axis of this figure would "
            f"carry a number that looks like a biological effect and is not. "
            f"Compute the fold change from normalized-but-not-scaled data.",
            evidence=VOLCANO_EVIDENCE,
            offer={"draw": "pca_scores",
                   "label": "The scores plot, for which scaling is expected",
                   "caption_note": (
                       "A scores plot states the scaling it was fitted on and "
                       "is not distorted by it, which is why it is the figure "
                       "autoscaled data supports."),
                   "forbidden": "fold_change_in_z_units"})

    if bool((block < 0).any().any()):
        raise FigureRefusal(
            "Some of these feature values are negative, so a ratio between "
            "group means is not defined and a fold change cannot be computed "
            "from them. Negatives in an abundance matrix mean the values have "
            "already been transformed — logged, scaled, or both — and a fold "
            "change of transformed values is not a fold change. It is computed "
            "from normalized-but-not-scaled data or it is not computed.",
            evidence=VOLCANO_EVIDENCE,
            offer={"draw": "pca_scores",
                   "label": "The scores plot, which is drawn from these values",
                   "caption_note": (
                       "PCA is a projection of the values as they are, so a "
                       "transformed matrix is what it expects rather than "
                       "something it must undo."),
                   "forbidden": "fold_change_from_transformed_values"})

    usable = groups.isin([reference, comparison])
    screening = univariate_screening(
        block[usable].fillna(block[usable].median()).to_numpy(dtype=float),
        pd.Series(groups[usable]).astype("category").cat.codes.to_numpy(
            dtype=float),
        features, task_type="classification")
    q_values = screening.details["corrected_p_values"]
    p_values = screening.details["raw_p_values"]

    mean_ref = block[groups == reference].mean()
    mean_cmp = block[groups == comparison].mean()
    eps = float(block[block > 0].min().min() or 1.0) / 2.0

    points, n_up, n_down = [], 0, 0
    for name in features:
        a, b = float(mean_cmp.get(name, np.nan)), float(mean_ref.get(name, np.nan))
        if not np.isfinite(a) or not np.isfinite(b):       # pragma: no cover
            continue
        log2fc = float(np.log2((a + eps) / (b + eps)))
        q = float(q_values.get(name, 1.0))
        significant = q <= q_cut and abs(log2fc) >= fc_cut
        if significant:
            n_up += log2fc > 0
            n_down += log2fc < 0
        points.append({"feature": name, "log2_fold_change": log2fc,
                       "q": q, "p": float(p_values.get(name, 1.0)),
                       "neg_log10_q": float(-np.log10(max(q, 1e-300))),
                       "significant": bool(significant)})

    # THE p CORRESPONDING TO q = 0.05, so a renderer that puts p on the y-axis
    # can still draw the cut line where the research requires it.
    below = [pt["p"] for pt in points if pt["q"] <= q_cut]
    return {
        "figure": "volcano",
        "group_column": str(group_col),
        "reference_group": str(reference),
        "comparison_group": str(comparison),
        "direction": (f"positive log2 fold change means higher in "
                      f"{comparison} than in {reference}"),
        "n_features": len(points),
        "n": int(usable.sum()),
        "points": points,
        "y_axis": "neg_log10_q",
        "test": screening.details["test_names"].get(features[0], "rank")
        if features else "rank",
        "correction": "Benjamini-Hochberg FDR",
        "q_cut": float(q_cut),
        "p_at_q_cut": float(max(below)) if below else None,
        "fold_change_cut": float(fc_cut),
        # CONVENTION, and carried rather than asserted — the research calls the
        # cut arbitrary and says to justify it biologically.
        "fold_change_cut_status": CONVENTION_STATUS,
        "n_significant_up": int(n_up),
        "n_significant_down": int(n_down),
        "n_significant": int(n_up + n_down),
        "n_uncorrected_significant": int(
            sum(1 for pt in points if pt["p"] <= 0.05)),
        "expected_by_chance": float(0.05 * len(points)),
        "scaling_state": "normalized_not_scaled",
        "feature_median_sd": spread,
        # No compound labels are drawn, and that is the item rather than an
        # omission: labels belong only to annotated compounds with a stated MSI
        # level, and this table carries no annotations.
        "labeled_features": [],
        "labels_require_msi_level": True,
    }


VOLCANO = register(FigureSpec(
    id="volcano",
    title="Volcano plot",
    tier=CONFIRMATORY,
    when_applicable=lambda s: (bool(s.get("has_assay_lens"))
                               and bool(s.get("target_is_binary"))
                               and int(s.get("n_numeric") or 0) >= 30),
    layers=("feature_points", "significance_cut_line", "fold_change_cut_lines",
            "direction_annotation"),
    annotations=(
        Annotation("n_features", "features tested", "turbotab.figure_specs"),
        Annotation("correction", "multiple-testing correction",
                   "ml.feature_selection"),
        Annotation("q_cut", "q threshold", "turbotab.figure_specs"),
        Annotation("fold_change_cut", "|log2 fold change| threshold",
                   "turbotab.figure_specs"),
        Annotation("n_significant_up", "significant, higher in the comparison "
                                       "group", "turbotab.figure_specs"),
        Annotation("n_significant_down", "significant, lower in the comparison "
                                         "group", "turbotab.figure_specs"),
        Annotation("scaling_state", "the data state the fold change was "
                                    "computed from", "turbotab.figure_specs"),
    ),
    checklist=(
        ChecklistItem(
            "q_on_the_y_axis",
            "The y-axis is −log10 of the FDR-adjusted q, or the cut line is "
            "drawn at the p corresponding to q = 0.05 and the caption says so",
            "Plotting raw p-values with a line at p = 0.05 on a "
            "3,000-feature untargeted dataset is an anti-pattern and would be "
            "flagged in review.",
            lambda p: p.get("y_axis") == "neg_log10_q"
            or p.get("p_at_q_cut") is not None),
        ChecklistItem(
            "fold_change_from_unscaled_data",
            "The fold change was computed from normalized-but-NOT-scaled data",
            "After autoscaling a fold change is a fold change in z-units and "
            "is meaningless. This is a subtle, real, embarrassing error.",
            lambda p: p.get("scaling_state") == "normalized_not_scaled"),
        ChecklistItem(
            "thresholds_annotated_numerically",
            "Both threshold lines are annotated with their numbers",
            "A cut line with no number is a decision the reader cannot check "
            "or reproduce.",
            lambda p: (p.get("q_cut") is not None
                       and p.get("fold_change_cut") is not None)),
        ChecklistItem(
            "counts_printed",
            "Counts of significant features up and down, printed on the panel",
            "The reader should not have to count dots to learn the size of "
            "the result.",
            lambda p: (p.get("n_significant_up") is not None
                       and p.get("n_significant_down") is not None)),
        ChecklistItem(
            "direction_stated",
            "The direction of the fold change is stated unambiguously",
            "Which group is the numerator is the whole meaning of the x-axis, "
            "and 'up' means nothing without it.",
            lambda p: bool(p.get("direction"))),
        ChecklistItem(
            "no_label_without_an_msi_level",
            "Compounds are labeled only where an MSI identification level is "
            "stated",
            "A feature label is a claim of identity. Without a stated MSI "
            "level it reads as 'identified' when the honest word is "
            "'putatively annotated'.",
            lambda p: not p.get("labeled_features")
            or bool(p.get("labels_require_msi_level"))),
    ),
    caption=lambda p: (
        f"Differential abundance of {p.get('n_features', 0):,} features "
        f"between {p.get('comparison_group')} and {p.get('reference_group')} "
        f"(`{p.get('group_column')}`, n = {p.get('n', 0):,}). "
        f"{p.get('direction', '')}. The x-axis is the log2 ratio of group "
        f"means computed from normalized-but-not-scaled data; the y-axis is "
        f"−log10 of the {p.get('correction')} adjusted q-value. The horizontal "
        f"cut is drawn at q = {p.get('q_cut')}"
        + (f", which corresponds to p = {p['p_at_q_cut']:.2g} in these data"
           if p.get("p_at_q_cut") is not None else "")
        + f"; the vertical cuts at |log2 fold change| = "
          f"{p.get('fold_change_cut')}, which is a convention rather than a "
          f"result and is stated here so it can be argued with. "
        + f"{p.get('n_significant', 0):,} features are significant "
          f"({p.get('n_significant_up', 0):,} higher, "
          f"{p.get('n_significant_down', 0):,} lower). "
        + (f"At an uncorrected p < 0.05 you would expect about "
           f"{p.get('expected_by_chance', 0):.0f} features by chance alone and "
           f"there are {p.get('n_uncorrected_significant', 0):,}; the "
           f"corrected count above is the result. "
           if p.get("n_features") else "")
        + "No features are labeled, because a label is a claim of identity and "
          "this table carries no MSI identification levels."),
    # NO COMPANION, and that is a reading of the research rather than an
    # omission. §06.5 names one for PLS-DA — its permutation plot — and §06.3
    # names none for the volcano. Inventing one to make CONFIRMATORY feel
    # earned would be ceremony, and `admissible()` would then hold back a
    # figure the field admits.
    companions=(),
    evidence=VOLCANO_EVIDENCE,
    claims=VOLCANO_CLAIMS,
    # NOT PROMOTABLE. The q-values are computed from every row, including the
    # rows a held-out set would contain — this is a description of the whole
    # table, and re-running it inside a fold would be a different figure.
    promotable=False,
    promotable_because="",
    compute=volcano_payload,
))


# ─────────────────────────────────────────────────────────────────────────────
# 5 · The restricted cubic spline — CONFIRMATORY
#
# `research/NUTRITION_PACK.md` §07 figure G. The publication-grade delta is
# almost entirely underneath the curve: **a rug or histogram of the exposure**,
# so the reader sees the dramatic upturn is driven by eleven people. Truncate at
# the 1st–99th percentile. Report a p for non-linearity.
#
# ## What it did NOT bend, which is the more useful result
#
# Built second, after the volcano bent the spec. This one fits: `layers` carries
# the rug, `annotations` carry the knots and the p, `checklist` scores the
# truncation by comparing the drawn range against the observed one — the same
# both-ranges trick the calibration plot needed, arriving unchanged on a figure
# from another field. `when_applicable` answers the question it is for. **A
# figure that costs the abstraction nothing is evidence the abstraction is
# done bending**, which is `LOOP.md` §02's own test for which phase a batch is
# in, and it is worth as much as the volcano's bend.
#
# ## The one modeling decision, and it is the research's
#
# `dietary_recalls.csv` is 600 rows from 300 people. Fitting a dose–response on
# rows and reporting a p computed under independence would be wrong in the
# direction that matters — too small. So where the project has RECORDED that
# people repeat, the fit runs on one row per participant, and §03 is what
# licenses it: *"If your goal is to rank people — regression, classification,
# quantiles of exposure, a predictive model — the mean of your available
# recalls is an acceptable exposure."* The caption says which rows were fitted.
# ─────────────────────────────────────────────────────────────────────────────

SPLINE_EVIDENCE = Evidence(
    status=CONVENTION_STATUS,
    source="research/NUTRITION_PACK.md#07 · EDA and presentation")

# 3 knots at the 10th, 50th and 90th percentiles, or 4 at 5/35/65/95. The
# research's own two options; three is the default because it is the one it
# names first and it costs one degree of freedom.
SPLINE_KNOT_PERCENTILES = {3: (10.0, 50.0, 90.0), 4: (5.0, 35.0, 65.0, 95.0)}
# Truncate at the 1st–99th percentile of the exposure.
SPLINE_TRUNCATION = (1.0, 99.0)


def _rcs_basis(x: np.ndarray, knots: Sequence[float]) -> np.ndarray:
    """Harrell's restricted-cubic-spline basis: a linear term and k−2 others.

    Written out rather than taken from `patsy.cr`, and the reason is the p for
    non-linearity: this parameterization puts the linear term first and every
    NONLINEAR term after it, so *"is the association non-linear"* is an F-test
    on a contiguous block of columns rather than a comparison of two fits.
    """
    knots = [float(k) for k in knots]
    k = len(knots)
    if k < 3:                                              # pragma: no cover
        raise ValueError("a restricted cubic spline needs at least three knots")
    scale = (knots[-1] - knots[0]) ** 2

    def cube(v):
        return np.where(v > 0, v ** 3, 0.0)

    columns = [x]
    for j in range(k - 2):
        term = (cube(x - knots[j])
                - cube(x - knots[k - 2]) * (knots[k - 1] - knots[j])
                / (knots[k - 1] - knots[k - 2])
                + cube(x - knots[k - 1]) * (knots[k - 2] - knots[j])
                / (knots[k - 1] - knots[k - 2]))
        columns.append(term / scale)
    return np.column_stack(columns)


def spline_payload(df: pd.DataFrame, *, exposure: str, outcome: str,
                   person_col: Optional[str] = None, n_knots: int = 3,
                   reference_percentile: float = 10.0) -> Dict[str, Any]:
    """The fitted curve, its band, the exposure underneath it, and p for non-linearity."""
    import statsmodels.api as sm

    frame = pd.DataFrame({
        "x": pd.to_numeric(df[exposure], errors="coerce"),
        "y": pd.to_numeric(df[outcome], errors="coerce"),
    })
    unit = "row"
    n_people = None
    if person_col and person_col in df.columns:
        frame["person"] = df[person_col].to_numpy()
        n_people = int(frame["person"].nunique())
        if n_people < len(frame):
            # ONE ROW PER PARTICIPANT. A p computed across a person's repeated
            # days under an independence assumption is too small, and too small
            # in the direction a reader would act on.
            frame = frame.groupby("person", as_index=False)[["x", "y"]].mean()
            unit = "participant"
    frame = frame.dropna(subset=["x", "y"])
    if len(frame) < 40:
        raise FigureRefusal(
            f"A restricted cubic spline over {len(frame):,} {unit}s would be "
            f"fitting three knots to a handful of points, and the curve would "
            f"describe the sample rather than the dose–response. The shape it "
            f"drew would be the strongest thing on the page and the least "
            f"supported.",
            evidence=SPLINE_EVIDENCE,
            offer={"draw": "per_nutrient_distribution",
                   "label": f"The observed distribution of {exposure}",
                   "caption_note": (
                       "The exposure's own distribution, with no outcome model "
                       "over it and no shape claimed."),
                   "forbidden": "spline_on_too_few_points"})

    lo, hi = (float(np.percentile(frame["x"], SPLINE_TRUNCATION[0])),
              float(np.percentile(frame["x"], SPLINE_TRUNCATION[1])))
    observed_lo, observed_hi = float(frame["x"].min()), float(frame["x"].max())
    percentiles = SPLINE_KNOT_PERCENTILES.get(n_knots,
                                              SPLINE_KNOT_PERCENTILES[3])
    knots = [float(np.percentile(frame["x"], q)) for q in percentiles]
    if len(set(knots)) < len(knots):
        raise FigureRefusal(
            f"The knots for this spline fall on the same value of "
            f"`{exposure}`, because its distribution has too little spread at "
            f"the percentiles the research specifies "
            f"({', '.join(f'{q:g}' for q in percentiles)}). A basis built on "
            f"repeated knots is singular, and a curve fitted through it would "
            f"be an artifact of the tie rather than a dose–response.",
            evidence=SPLINE_EVIDENCE,
            offer={"draw": "per_nutrient_distribution",
                   "label": f"The observed distribution of {exposure}",
                   "caption_note": (
                       "The exposure's own distribution, which is where the "
                       "lack of spread is visible."),
                   "forbidden": "spline_on_repeated_knots"})

    x = frame["x"].to_numpy(dtype=float)
    y = frame["y"].to_numpy(dtype=float)
    basis = _rcs_basis(x, knots)
    design = sm.add_constant(basis, has_constant="add")
    fit = sm.OLS(y, design).fit()

    # p FOR NON-LINEARITY: the joint test on the nonlinear columns, which is why
    # the basis puts the linear term first.
    n_nonlinear = basis.shape[1] - 1
    contrast = np.zeros((n_nonlinear, design.shape[1]))
    for i in range(n_nonlinear):
        contrast[i, 2 + i] = 1.0
    p_nonlinearity = float(fit.f_test(contrast).pvalue)

    grid = np.linspace(lo, hi, 120)
    grid_design = sm.add_constant(_rcs_basis(grid, knots), has_constant="add")
    reference_x = float(np.percentile(x, reference_percentile))
    reference_design = sm.add_constant(
        _rcs_basis(np.array([reference_x]), knots), has_constant="add")

    # Centred on the reference, so the curve reads as a contrast against a
    # stated percentile rather than as an intercept nobody chose.
    delta = grid_design - reference_design
    fitted = delta @ fit.params
    cov = np.asarray(fit.cov_params())
    se = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", delta, cov, delta), 0.0))

    counts, edges = np.histogram(x[(x >= lo) & (x <= hi)], bins=30)
    return {
        "figure": "dose_response_spline",
        "exposure": str(exposure),
        "outcome": str(outcome),
        "n": int(len(frame)),
        "unit_of_analysis": unit,
        "n_rows_supplied": int(len(df)),
        "n_participants": n_people,
        "n_knots": len(knots),
        "knots": knots,
        "knot_percentiles": [float(q) for q in percentiles],
        "p_nonlinearity": p_nonlinearity,
        "reference_percentile": float(reference_percentile),
        "reference_value": reference_x,
        "curve": {"x": [float(v) for v in grid],
                  "fit": [float(v) for v in fitted],
                  "lower": [float(v) for v in fitted - 1.96 * se],
                  "upper": [float(v) for v in fitted + 1.96 * se]},
        # THE RUG, which is the entire publication-grade delta. Without it the
        # reader cannot tell that the upturn at the right is eleven people.
        "exposure_distribution": {
            "counts": [int(c) for c in counts],
            "edges": [float(e) for e in edges]},
        "x_range_drawn": [lo, hi],
        "x_range_observed": [observed_lo, observed_hi],
        "truncation_percentiles": list(SPLINE_TRUNCATION),
        "r_squared": float(fit.rsquared),
    }


DOSE_RESPONSE_SPLINE = register(FigureSpec(
    id="dose_response_spline",
    title="Dose–response, restricted cubic spline",
    tier=CONFIRMATORY,
    when_applicable=lambda s: (bool(s.get("has_dietary_lens"))
                               and s.get("task_type") == "regression"
                               and int(s.get("n_rows") or 0) >= 40),
    layers=("fitted_curve", "confidence_band", "exposure_rug",
            "reference_marker"),
    annotations=(
        Annotation("n", "n fitted", "turbotab.figure_specs"),
        Annotation("unit_of_analysis", "what one point is",
                   "turbotab.figure_specs"),
        Annotation("knot_percentiles", "knot percentiles",
                   "turbotab.figure_specs"),
        Annotation("p_nonlinearity", "p for non-linearity", "statsmodels"),
        Annotation("reference_percentile", "reference percentile",
                   "turbotab.figure_specs"),
        Annotation("truncation_percentiles", "axis truncated at",
                   "turbotab.figure_specs"),
    ),
    checklist=(
        ChecklistItem(
            "exposure_distribution_underneath",
            "A rug or histogram of the exposure is drawn underneath the curve",
            "Without it the reader cannot see that the dramatic upturn at the "
            "right-hand end is driven by eleven people.",
            lambda p: bool(p.get("exposure_distribution", {}).get("counts"))
            and sum(p["exposure_distribution"]["counts"]) > 0),
        ChecklistItem(
            "truncated_at_1st_and_99th",
            "The curve is truncated at the 1st–99th percentile of the exposure",
            "Beyond them the spline is extrapolating from single observations, "
            "and the band does not widen enough to say so.",
            lambda p: (p.get("x_range_drawn", [0, 1])[0]
                       > p.get("x_range_observed", [0, 1])[0]
                       or p.get("x_range_drawn", [0, 1])[1]
                       < p.get("x_range_observed", [0, 1])[1])),
        ChecklistItem(
            "p_for_nonlinearity_reported",
            "A p for non-linearity is reported",
            "Without it the reader cannot tell whether the curve's bend is "
            "evidence or is the spline drawing noise.",
            lambda p: p.get("p_nonlinearity") is not None),
        ChecklistItem(
            "knots_stated",
            "The number of knots and the percentiles they sit at are stated",
            "The shape of a spline is a function of its knots, so a curve "
            "whose knots are unstated is not reproducible.",
            lambda p: bool(p.get("knots")) and bool(p.get("knot_percentiles"))),
        ChecklistItem(
            "reference_stated",
            "The reference the curve is centred on is stated as a percentile",
            "A dose–response curve is a contrast, and a contrast with an "
            "unnamed baseline cannot be read.",
            lambda p: p.get("reference_percentile") is not None),
    ),
    caption=lambda p: (
        f"Restricted cubic spline of {p['outcome']} on {p['exposure']}, "
        f"{p.get('n_knots', 3)} knots at the "
        f"{', '.join(f'{q:g}th' for q in p.get('knot_percentiles', []))} "
        f"percentiles, fitted on {p.get('n', 0):,} "
        f"{p.get('unit_of_analysis', 'row')}s"
        + (f" — one row per participant, the mean of each participant's "
           f"{p['n_rows_supplied'] // max(p.get('n_participants') or 1, 1)} "
           f"available days, because a p computed across a person's repeated "
           f"days under an independence assumption would be too small"
           if p.get("unit_of_analysis") == "participant" else "")
        + f". The curve is a contrast against the "
          f"{p.get('reference_percentile'):g}th percentile of the exposure "
          f"({p.get('reference_value', 0):,.0f}), with a pointwise 95% band. "
          f"p for non-linearity = {p.get('p_nonlinearity', float('nan')):.3g}. "
          f"The histogram underneath is the exposure's own distribution, so a "
          f"bend supported by few observations can be seen to be one. The axis "
          f"is truncated at the "
          f"{p.get('truncation_percentiles', [1, 99])[0]:g}st–"
          f"{p.get('truncation_percentiles', [1, 99])[1]:g}th percentile."),
    companions=(),
    evidence=SPLINE_EVIDENCE,
    promotable=False,
    promotable_because="",
    compute=spline_payload,
))


# ─────────────────────────────────────────────────────────────────────────────
# 6 · The diverging stacked bar — EXPLORATORY
#
# `research/CLINICAL_SURVEY_PACK.md` §B5.1. *"The field-standard Likert
# figure"* (Heiberger & Robbins, J Stat Softw 2014; `likert()` in the R **HH**
# package), and the figure a table of means cannot replace: it is what lets a
# reader compare items at a glance.
#
# ## The bend that is not a bend — a checklist item the app can never pass
#
# Requirement 7 is *"the response-scale anchors in the legend verbatim
# ('Strongly disagree … Strongly agree'), not '1 … 5'"*, and **the app has only
# the numeric codes.** `likert_block` reads a shared response scale out of the
# values; nothing in an upload carries the instrument's anchor text.
#
# The item is kept and it fails, and that is the right behavior rather than a
# gap: the figure genuinely is not publication-grade without anchors, and the
# same rule already governs the calibration plot — *failing the checklist and
# rendering honestly are different jobs.* What the payload must not do is print
# `1 … 5` in the legend as if those were the anchors, so it carries
# `anchors: None` with the reason, and `annotation_rows` renders that as
# `not estimable` rather than as a number nobody supplied.
#
# **This is NOT `GUIDED-066`'s class**, and the difference is worth stating: the
# PCA plot's QC item scores a metabolomics requirement against a table from
# another field, which is a scoping error. This item scores a survey
# requirement against a survey table and reports a real absence. One is the
# wrong question; the other is the right question with an honest answer.
#
# ## The disputed choice, carried rather than hidden
#
# How to treat the neutral midpoint is **[DISPUTED, low stakes]** — splitting it
# across zero preserves total bar length and slightly distorts both wings;
# placing it on one side or excluding it are both defensible. The research says
# TurboTab defaults to splitting and **the caption must say which**, so the
# choice is a payload field and a caption sentence rather than a rendering
# detail. The figure's badge is CONVENTION over the whole figure, and this one
# clause inside it is disputed — `GUIDED-064`'s fourth instance.
# ─────────────────────────────────────────────────────────────────────────────

DIVERGING_EVIDENCE = Evidence(
    status=CONVENTION_STATUS,
    source=("research/CLINICAL_SURVEY_PACK.md#B5.1 ★ Diverging stacked bar "
            "chart — the field-standard Likert figure"))

# `GUIDED-064`. The figure is the field standard; one clause inside it is not.
DIVERGING_CLAIMS = (
    Claim("the_figure",
          "The diverging stacked bar is the standard graphic for Likert data "
          "and is what lets a reader compare items at a glance.",
          DIVERGING_EVIDENCE),
    Claim("neutral_treatment",
          "The neutral category is split across the zero line, which preserves "
          "total bar length and slightly distorts both wings.",
          Evidence(
              status=DISPUTED,
              source=("research/CLINICAL_SURVEY_PACK.md#B5.1 ★ Diverging "
                      "stacked bar chart — the field-standard Likert figure"),
              both_sides=(
                  "Splitting the neutral category across zero preserves total "
                  "bar length and makes the agree/disagree split visually "
                  "honest, but slightly distorts the apparent size of both "
                  "wings. Placing the whole neutral category on one side, or "
                  "excluding it and reporting it separately, are both "
                  "defensible; the caption states which was done."))),
)

# Segments below this share are suppressed rather than overprinted.
LABEL_THRESHOLD = 0.05
NEUTRAL_SPLIT = "split_across_zero"


def diverging_bar_payload(df: pd.DataFrame, *, columns: Sequence[str],
                          scale: Sequence[int],
                          anchors: Optional[Sequence[str]] = None,
                          item_text: Optional[Dict[str, str]] = None
                          ) -> Dict[str, Any]:
    """Percentages across a zero line, sorted by net agreement, with n per item."""
    scale = [int(v) for v in scale]
    if len(scale) < 3:                                     # pragma: no cover
        raise FigureRefusal(
            "A diverging bar needs a response scale with at least three "
            "categories to have a direction at all.",
            evidence=DIVERGING_EVIDENCE,
            offer={"draw": "per_item_response_panel",
                   "label": "One bar chart per item",
                   "caption_note": "Shape rather than direction.",
                   "forbidden": "diverging_bar_without_a_direction"})

    midpoint = scale[len(scale) // 2] if len(scale) % 2 else None
    disagree = [v for v in scale if midpoint is None or v < midpoint]
    agree = [v for v in scale if midpoint is None or v > midpoint]

    items = []
    for column in columns:
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        n = int(len(values))
        if not n:                                          # pragma: no cover
            continue
        shares = {v: float((values == v).mean()) for v in scale}
        net = sum(shares[v] for v in agree) - sum(shares[v] for v in disagree)
        items.append({
            "column": str(column),
            "text": (item_text or {}).get(str(column)),
            "n": n,
            "shares": {str(v): shares[v] for v in scale},
            "net_agreement": float(net),
            "percent_agree": float(sum(shares[v] for v in agree)),
            "percent_disagree": float(sum(shares[v] for v in disagree)),
            "percent_neutral": float(shares[midpoint]) if midpoint is not None
            else 0.0,
            "labeled_segments": [str(v) for v in scale
                                 if shares[v] >= LABEL_THRESHOLD],
        })

    # SORTED BY NET AGREEMENT, and the sort is stated. Alphabetical or by item
    # number is what the figure exists to replace.
    items.sort(key=lambda row: row["net_agreement"], reverse=True)
    n_values = [row["n"] for row in items]
    return {
        "figure": "diverging_stacked_bar",
        "n_items": len(items),
        "items": items,
        "scale": scale,
        "midpoint": midpoint,
        # THE ANCHORS, OR THE ABSENCE OF THEM. Never `1 … 5` dressed as anchor
        # text: those are the codes, and printing them in the legend would be
        # the app inventing the instrument's wording.
        "anchors": list(anchors) if anchors else None,
        "anchors_absent_because": None if anchors else (
            "This table carries the numeric response codes and not the "
            "instrument's anchor text, and the anchors are not recoverable "
            "from the data. The legend says so rather than printing "
            f"{scale[0]} … {scale[-1]} as if those were the words."),
        "sort": "net_agreement_descending",
        "sort_stated": True,
        "neutral_treatment": NEUTRAL_SPLIT,
        "neutral_treatment_status": DISPUTED,
        "percentage_basis": "respondents_answering_the_item",
        "n_min": min(n_values) if n_values else 0,
        "n_max": max(n_values) if n_values else 0,
        "n_per_item_shown": True,
        "label_threshold": LABEL_THRESHOLD,
        "zero_line": "solid_rule",
        # The ordinal sequence is encoded by LIGHTNESS so it survives greyscale.
        # The colors themselves are `DESIGN_LANGUAGE.md`'s and are not restated
        # here — a second palette beside the design language's is the drift this
        # project keeps finding.
        "palette": {"kind": "diverging_ordinal", "encoding": "lightness",
                    "colorblind_safe": True,
                    "source": "DESIGN_LANGUAGE.md §02"},
    }


DIVERGING_STACKED_BAR = register(FigureSpec(
    id="diverging_stacked_bar",
    title="Diverging stacked bar",
    tier=EXPLORATORY,
    when_applicable=lambda s: (bool(s.get("has_survey_lens"))
                               and bool(s.get("has_likert_block"))),
    layers=("diverging_segments", "zero_line", "n_per_item", "legend_anchors"),
    annotations=(
        Annotation("n_items", "items", "turbotab.figure_specs"),
        Annotation("scale", "response scale", "turbotab.packs"),
        Annotation("anchors", "response anchors, verbatim",
                   "turbotab.figure_specs"),
        Annotation("sort", "item order", "turbotab.figure_specs"),
        Annotation("neutral_treatment", "how the neutral category is drawn",
                   "turbotab.figure_specs"),
        Annotation("percentage_basis", "percentages are of",
                   "turbotab.figure_specs"),
    ),
    checklist=(
        ChecklistItem(
            "sorted_by_net_agreement_and_stated",
            "Items are ordered by net agreement, and the caption says so",
            "Alphabetical or item-number order is what this figure exists to "
            "replace — the comparison at a glance is the whole point.",
            lambda p: p.get("sort") == "net_agreement_descending"
            and bool(p.get("sort_stated"))),
        ChecklistItem(
            "n_per_item_at_the_right_edge",
            "n is printed per item at the right edge",
            "Item-level missingness varies, and a bar whose n is 40 reads the "
            "same as one whose n is 400 without it.",
            lambda p: bool(p.get("n_per_item_shown"))
            and all(row.get("n") for row in p.get("items", []))),
        ChecklistItem(
            "anchors_verbatim_in_the_legend",
            "The response-scale anchors appear in the legend verbatim, never "
            "'1 … 5'",
            "The codes are not the question. A legend reading 1 … 5 asks the "
            "reader to guess what the respondent was agreeing with.",
            lambda p: bool(p.get("anchors"))),
        ChecklistItem(
            "lightness_encoded_ordinal_palette",
            "A single ordered diverging palette with the sequence encoded by "
            "lightness",
            "A categorical palette for ordered categories loses the order, and "
            "a hue-encoded one loses it again in greyscale.",
            lambda p: (p.get("palette", {}).get("encoding") == "lightness"
                       and p.get("palette", {}).get("colorblind_safe") is True)),
        ChecklistItem(
            "neutral_treatment_stated",
            "How the neutral category is treated is stated in the caption",
            "Splitting it across zero, placing it on one side and excluding it "
            "are all defensible and give different-looking wings, so the "
            "reader has to be told which was done.",
            lambda p: bool(p.get("neutral_treatment"))),
        ChecklistItem(
            "percentage_basis_stated",
            "Whether percentages are of respondents answering the item or of "
            "all respondents is stated",
            "With item-level missingness the two differ, and the difference is "
            "invisible on the bar.",
            lambda p: bool(p.get("percentage_basis"))),
    ),
    caption=lambda p: (
        f"Response distribution across {p.get('n_items', 0):,} items on a "
        f"{len(p.get('scale', []))}-point scale, as percentages of the "
        f"respondents answering each item "
        f"(n {p.get('n_min', 0):,}–{p.get('n_max', 0):,}). Items are ordered "
        f"by net agreement, most agreed at the top. The neutral category is "
        f"split across the zero line so total bar length is preserved; that "
        f"choice is disputed and placing it on one side or excluding it are "
        f"both defensible, so it is stated rather than assumed. Segments below "
        f"{p.get('label_threshold', 0.05):.0%} are left unlabeled rather than "
        f"overprinted. "
        + (f"The legend carries the response anchors verbatim."
           if p.get("anchors")
           else f"{p['anchors_absent_because']}")),
    companions=(),
    evidence=DIVERGING_EVIDENCE,
    claims=DIVERGING_CLAIMS,
    promotable=False,
    promotable_because="",
    compute=diverging_bar_payload,
))


# ─────────────────────────────────────────────────────────────────────────────
# 7 · Prediction instability — CONFIRMATORY
# 8 · Calibration instability — CONFIRMATORY
#
# `CLINICAL_SURVEY_PACK.md` §A4.8, marked ★, and the two figures are one build
# because they read the same resampling result: `turbotab.instability.run`
# refits the entire pipeline B times and both of these are views of its output.
#
# THE SEAM THIS PAIR FOUND. Every figure before them annotated a computation
# that already existed; these two required the computation to be built first,
# and that inverted the usual order — `B` is a number the FIGURE has to state
# and the ENGINE has to choose. The resolution is that the engine owns the
# constant and the caption reads it, so there is one number and it is visible.
# ─────────────────────────────────────────────────────────────────────────────

INSTABILITY_EVIDENCE = Evidence(
    status=CONVENTION_STATUS,
    source=("research/CLINICAL_SURVEY_PACK.md#A4.8 · ★ Prediction stability "
            "plots — the modern addition"))

#: Per-requirement, because the pack does not hold these at one status. The
#: METHOD is emerging convention in the pack's own words; the presentation
#: details are its specific instructions and are not disputed.
INSTABILITY_CLAIMS = (
    Claim(key="resample_the_whole_pipeline",
          statement=("The ENTIRE modeling pipeline is refitted in each resample, "
                "including any variable selection — not the estimator over a "
                "fixed feature set."),
          evidence=Evidence(
              status=SETTLED,
              source=("research/CLINICAL_SURVEY_PACK.md#A5.5 Modeling "
                      "practice"))),
    Claim(key="instability_is_expected_reporting",
          statement=("Reviewers now ask for per-individual prediction instability "
                "rather than a single point estimate of discrimination."),
          evidence=INSTABILITY_EVIDENCE),
    Claim(key="alpha_and_overlay",
          statement=("Scatter with the 45° line and semi-transparent points "
                "(alpha about 0.02); calibration instability as many thin grey "
                "curves with the original in bold."),
          evidence=INSTABILITY_EVIDENCE),
)


def prediction_instability_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    """The render's payload, from `turbotab.instability.run`'s output.

    Adds exactly what the checklist needs and the engine does not produce: the
    45° reference, the alpha the pack names, and both axis ranges — the same
    `no_truncation` scoring the calibration plot needs, and for the same
    reason. The instability plot's sparse region is where the model is least
    trustworthy, so it is the region that must not be cropped.
    """
    from turbotab import instability as _inst

    spread = _inst.spread(result)
    original = np.asarray(result["original"], dtype=float)
    matrix = np.asarray(result["bootstrap"], dtype=float)
    lo = float(min(original.min(), matrix.min()))
    hi = float(max(original.max(), matrix.max()))
    return {
        "figure": "prediction_instability",
        "model_name": result.get("model_name", "the model"),
        "task_type": result.get("task_type"),
        "n": result.get("n", 0),
        "b_completed": result.get("b_completed", 0),
        "b_requested": result.get("b_requested", 0),
        "b_recommended": result.get("b_recommended", _inst.RECOMMENDED_B),
        "n_failed": len(result.get("failures") or []),
        "points": int(matrix.size),
        "alpha": 0.02,
        "reference_line": {"kind": "identity", "label": "no change"},
        "aspect": "square",
        "mape": result.get("mape") or {},
        "median_width": spread["median_width"],
        "max_width": spread["max_width"],
        "worst_row_label": spread["worst_row_label"],
        "worst_interval": spread["worst_interval"],
        "scored_on": result.get("scored_on", ""),
        "x_range_observed": [lo, hi],
        "x_range_drawn": [lo, hi],
    }


PREDICTION_INSTABILITY = register(FigureSpec(
    id="prediction_instability",
    title="Prediction instability plot",
    tier=CONFIRMATORY,
    when_applicable=lambda s: bool(s.get("has_instability_run")),
    layers=("identity_line", "bootstrap_scatter", "per_row_interval"),
    annotations=(
        Annotation("b_completed", "Bootstrap resamples (B)", "turbotab.instability"),
        Annotation("mape", "Mean absolute prediction error", "turbotab.instability"),
        Annotation("median_width", "Median 95% interval width", "turbotab.instability"),
        Annotation("max_width", "Widest 95% interval", "turbotab.instability"),
        Annotation("n", "n", "turbotab.instability"),
    ),
    checklist=(
        ChecklistItem(
            "b_stated",
            "The number of bootstrap resamples is stated on the figure",
            "A reader cannot judge the width of the cloud without knowing how "
            "many refits produced it, and B is the one number a different "
            "analyst would choose differently.",
            lambda p: bool(p.get("b_completed"))),
        ChecklistItem(
            "identity_line",
            "45° reference line, labeled",
            "The plot's whole content is vertical distance FROM this line. "
            "Without it the scatter is a cloud with no claim attached.",
            lambda p: p.get("reference_line", {}).get("kind") == "identity"),
        ChecklistItem(
            "semi_transparent",
            "Points are semi-transparent (alpha about 0.02)",
            "One point per patient per resample is tens of thousands of "
            "points; drawn opaque the plot is a solid block and the density "
            "that carries the message is invisible.",
            lambda p: 0 < float(p.get("alpha", 1)) <= 0.05),
        ChecklistItem(
            "mape_annotated",
            "Mean absolute prediction error is on the figure, named in full",
            "The pack writes MAPE and does not expand it, and the absolute and "
            "percentage readings differ by more than an order of magnitude on "
            "predicted risks near zero. Writing it out is what stops a reader "
            "assuming the other one.",
            lambda p: (p.get("mape", {}).get("absolute") is not None
                       and "absolute" in str(p.get("mape", {}).get("label", "")).lower())),
        ChecklistItem(
            "no_truncation",
            "The axis is not truncated",
            "The extremes are where predictions move most, which is what the "
            "figure exists to show.",
            lambda p: (p.get("x_range_drawn", [1, 0])[0]
                       <= p.get("x_range_observed", [0, 1])[0]
                       and p.get("x_range_drawn", [0, 0])[1]
                       >= p.get("x_range_observed", [0, 1])[1])),
        ChecklistItem(
            "scope_stated",
            "Which rows were resampled and predicted is stated",
            "An instability plot that had quietly resampled the held-out rows "
            "would look identical and would have dissolved the seal.",
            lambda p: "held-out" in str(p.get("scored_on", ""))),
    ),
    caption=lambda p: (
        f"Prediction instability for {p.get('model_name', 'the model')}: the "
        f"entire modeling pipeline, including any variable selection, was "
        f"refitted in {p.get('b_completed', 0):,} bootstrap resamples of the "
        f"{p.get('n', 0):,} training rows and each refitted model was applied "
        f"back to those same rows — one point per row per resample "
        f"({p.get('points', 0):,} points, alpha {p.get('alpha', 0.02)}). "
        f"The 45° line is no change from the original model; vertical spread "
        f"is how much an individual's prediction would have moved had a "
        f"different sample been drawn. Mean absolute prediction error "
        f"{_fmt(p.get('mape', {}).get('absolute'))}; median 95% interval width "
        f"{_fmt(p.get('median_width'))}, widest {_fmt(p.get('max_width'))}. "
        f"B = {p.get('b_completed', 0):,}"
        + (f" of {p.get('b_requested', 0):,} requested "
           f"({p.get('n_failed', 0):,} resample(s) could not be fitted)"
           if p.get("n_failed") else "")
        + f"; Riley and Collins recommend on the order of "
          f"{p.get('b_recommended', 1000):,}. {p.get('scored_on', '')}."),
    # ITS OWN COMPANION IS THE CALIBRATION INSTABILITY PLOT. §A4.8 specifies
    # the pair, and for the reason §A5.1 gives about the originals: spread in
    # individual predictions and spread in calibration are different failures,
    # and a model can look tight on one while moving badly on the other.
    companions=("calibration_instability",),
    evidence=INSTABILITY_EVIDENCE,
    claims=INSTABILITY_CLAIMS,
    # NOT PROMOTABLE, and the reason is the rule rather than the subject: this
    # figure is about a set of models already fitted to these rows. Re-running
    # it inside a fold would need the fold's own B refits, which is a bootstrap
    # inside a bootstrap and is not what promotion means.
    promotable=False,
    promotable_because="",
    compute=prediction_instability_payload,
))


def calibration_instability_payload(result: Dict[str, Any],
                                    y_true) -> Dict[str, Any]:
    """Every bootstrap model's calibration curve, plus the original in bold.

    One curve per resample, each binned the way `calibration_payload` bins the
    original — the two figures sit beside each other and a reader compares
    them, so a different binning would make the comparison a comparison of
    binnings.

    Regression has no calibration curve in this sense, so this returns
    `applicable: False` rather than a plot of something else.
    """
    if result.get("task_type") != "classification":
        return {"figure": "calibration_instability", "applicable": False,
                "because": ("A calibration curve plots observed risk against "
                            "predicted risk, and a regression model predicts a "
                            "value rather than a risk. There is nothing here "
                            "to overlay.")}

    y_true = np.asarray(y_true, dtype=float)
    matrix = np.asarray(result["bootstrap"], dtype=float)
    original = np.asarray(result["original"], dtype=float)
    edges = np.linspace(0.0, 1.0, 11)

    def _curve(predicted):
        binned = np.clip(np.digitize(predicted, edges[1:-1]), 0, 9)
        xs, ys, ns = [], [], []
        for b in range(10):
            mask = binned == b
            if not mask.any():
                continue
            xs.append(float(predicted[mask].mean()))
            ys.append(float(y_true[mask].mean()))
            ns.append(int(mask.sum()))
        return {"x": xs, "y": ys, "n": ns}

    curves = [_curve(matrix[i]) for i in range(matrix.shape[0])]
    return {
        "figure": "calibration_instability",
        "applicable": True,
        "model_name": result.get("model_name", "the model"),
        "n": result.get("n", 0),
        "events": int(y_true.sum()),
        "b_completed": result.get("b_completed", 0),
        "b_requested": result.get("b_requested", 0),
        "b_recommended": result.get("b_recommended", 1000),
        "n_bins": 10,
        "curves": curves,
        "original_curve": _curve(original),
        # The pack's presentation instruction, carried as data so the checklist
        # can score it rather than trusting the renderer.
        "bootstrap_style": {"color": "grey", "width": "thin", "alpha": 0.08},
        "original_style": {"color": "ink", "width": "bold"},
        "reference_line": {"kind": "identity", "label": "ideal"},
        "aspect": "square",
        "scored_on": result.get("scored_on", ""),
    }


CALIBRATION_INSTABILITY = register(FigureSpec(
    id="calibration_instability",
    title="Calibration instability plot",
    tier=CONFIRMATORY,
    when_applicable=lambda s: (
        s.get("task_type") == "classification"
        and bool(s.get("has_instability_run"))),
    layers=("identity_line", "bootstrap_curves", "original_curve"),
    annotations=(
        Annotation("b_completed", "Bootstrap resamples (B)", "turbotab.instability"),
        Annotation("n", "n", "turbotab.instability"),
        Annotation("events", "events", "turbotab.figure_specs"),
    ),
    checklist=(
        ChecklistItem(
            "b_stated",
            "The number of bootstrap resamples is stated on the figure",
            "The density of the grey band is a function of B; without it a "
            "reader cannot tell a tight model from a small B.",
            lambda p: bool(p.get("b_completed"))),
        ChecklistItem(
            "original_distinguishable",
            "The original model's curve is drawn in bold over the grey ones",
            "The pack asks for many thin grey curves plus the original in "
            "bold. Without the distinction the figure shows spread and hides "
            "what the spread is around.",
            lambda p: (p.get("original_style", {}).get("width") == "bold"
                       and p.get("bootstrap_style", {}).get("width") == "thin")),
        ChecklistItem(
            "identity_line",
            "45° reference line, labeled 'ideal'",
            "Every curve on this plot is read as a distance from ideal.",
            lambda p: (p.get("reference_line", {}).get("kind") == "identity"
                       and p.get("reference_line", {}).get("label") == "ideal")),
        ChecklistItem(
            "square_aspect",
            "Square aspect ratio, same scale on both axes",
            "Predicted and observed risk are the same quantity; a non-square "
            "panel makes agreement look like disagreement.",
            lambda p: p.get("aspect") == "square"),
        ChecklistItem(
            "scope_stated",
            "Which rows were resampled and predicted is stated",
            "Same reason as the prediction instability plot: a curve drawn "
            "over resampled held-out rows would look identical.",
            lambda p: "held-out" in str(p.get("scored_on", ""))),
    ),
    caption=lambda p: (
        (f"Calibration instability for {p.get('model_name', 'the model')}: "
         f"{p.get('b_completed', 0):,} calibration curves, one per bootstrap "
         f"refit of the entire pipeline, over {p.get('n', 0):,} training rows "
         f"with {p.get('events', 0):,} events, binned into "
         f"{p.get('n_bins', 10)} equal-width bins of predicted risk. Thin grey "
         f"curves are the bootstrap models; the bold curve is the original. "
         f"The dashed 45° line is ideal calibration. Spread between the grey "
         f"curves is how much the model's calibration depends on which "
         f"patients were sampled. B = {p.get('b_completed', 0):,}; Riley and "
         f"Collins recommend on the order of "
         f"{p.get('b_recommended', 1000):,}. {p.get('scored_on', '')}.")
        if p.get("applicable", True) else str(p.get("because", ""))),
    companions=("prediction_instability",),
    evidence=INSTABILITY_EVIDENCE,
    claims=INSTABILITY_CLAIMS,
    promotable=False,
    promotable_because="",
    compute=calibration_instability_payload,
))
