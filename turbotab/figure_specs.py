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


def _has(payload: Dict[str, Any], *keys: str) -> bool:
    return all(payload.get(k) is not None for k in keys)


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
    return "not computed" if value is None else f"{value:.3f}"


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
