"""The figure layer's first consumer — a project, drawn.

`GUIDED-058` and `DRIVE-009`. Three figures were specified, tested and
registered, and `figures.applicable()` and `figures.bundle()` had **zero callers
anywhere in the repository**. A module reachable only from its own tests is a
specification, and from inside the loop that built it it looks finished.

This is the part that was missing. It answers three questions the spec
deliberately does not:

1. **What does this project look like?** `state()` — the dict
   `when_applicable` reads. Every key in it is derived from a RECORDED answer or
   from the working table, never guessed.
2. **Where do this figure's numbers come from?** `SOURCES` — one adapter per
   figure id. The spec says what a figure *is*; the adapter says where this
   project's version of it comes from. They are different jobs and keeping them
   apart is what lets a pack change which figure is drawn without touching what
   the figure means (`DOMAIN_PACKS.md` §08).
3. **Why is a figure NOT drawn?** `not_drawn` — and this is the half that would
   have been easy to skip. A figure silently absent is indistinguishable from a
   figure the app does not have, which is `DESIGN_LANGUAGE.md` §09's
   recorded-absence rule pointed at the figure layer.

## The pack reaches the figure here, and that is DRIVE-009's own sentence

*"Per-domain figure selection through the pack mechanism."* Two places it is
literal rather than aspirational:

* The shrinkage plot fires **only** under the dietary lens, because
  `has_dietary_lens` is in its `when_applicable` and this module is what writes
  that key.
* The PCA scores plot overlays pooled QCs **only** when the metabolomics pack's
  `pooled_qc` detector fired, and takes the QC column and value from that
  finding's own `params`. The checklist item *"pooled QCs overlaid, never
  dropped"* then passes or fails on the pack's reading rather than on a
  renderer's guess.

## What cannot be drawn from a project today, stated rather than omitted

**The calibration plot.** Its `when_applicable` needs `has_predictions`, and
**TurboTab has no training step** — there is no fitted model, no held-out
prediction, and no question that records a column of predicted risks. So
`has_predictions` is `False` for every project that can exist, and the figure the
clinical pack calls *"the single most important figure in a clinical prediction
paper"* is reachable from `figure_specs.calibration_render()` and from nowhere a
user can stand. That is filed rather than papered over: the endpoint names it in
`not_drawn` with that reason, and `GUIDED-065` carries it.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from turbotab import figures

# Registers the three specs. Imported for the side effect, which is the same
# shape `recipes` uses and is why `register()` refuses a duplicate id.
from turbotab import figure_specs  # noqa: F401

_DIETARY = "dietary"
_METABOLOMICS = "metabolomics"
_GENOMICS = "genomics"
_SURVEY = "survey"


# ─────────────────────────────────────────────────────────────────────────────
# The state dict — every key derived from a recorded answer
# ─────────────────────────────────────────────────────────────────────────────

def _numeric_columns(df: pd.DataFrame) -> List[str]:
    return [str(c) for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c])
            and not pd.api.types.is_bool_dtype(df[c])]


def recalls_per_person(project) -> Tuple[int, str]:
    """`(n, why)` — how many recalls a person has, and how that was decided.

    **The key `GUIDED-058` says is written nowhere**, and the honest derivation
    turned out to need three recorded answers rather than one:

    * **grain** must say *people repeat*, with a grouping column. Without it the
      app does not know that two rows are one person, and counting rows would
      make a 600-row one-row-per-person table look like 600 recalls.
    * **repeat_kind** must say *repeats* rather than *time points*. A person's
      rows being twelve months apart is a longitudinal series, and averaging
      them is a different operation with a different meaning — question 4 exists
      precisely because *"which of the two it is decides whether averaging a
      person's rows is correct"*.
    * **the lens** must say dietary, which `when_applicable` checks separately.

    Returns the MEDIAN rows per person, not the maximum: NHANES-shaped data has
    people with one recall beside people with two, and the maximum would let one
    two-day participant turn on a figure about the sample.

    The `why` string is returned even on success, because the endpoint reports
    it either way. A figure drawn on a basis nobody can see is the same defect
    as a figure absent for a reason nobody can see.
    """
    grain = project.grain or {}
    if grain.get("answer") != "people_repeat":
        return 0, (
            "The grain question has not been answered *people repeat*, so "
            "nothing in this project says that two rows are one person's two "
            "days. Rows are not recalls until something records that they are."
            if grain else
            "The grain question has not been asked yet, and until it is, the "
            "number of recalls a person has is not something the app knows.")
    group_col = grain.get("group_col")
    if not group_col or group_col not in project.df.columns:
        return 0, (
            "The grain answer is *people repeat* and names no grouping column, "
            "so a person's rows cannot be identified.")

    kind = (project.repeat_kind or {}).get("kind")
    if kind is None:
        return 0, (
            f"A person's rows in `{group_col}` have not been recorded as "
            f"repeated measurements or as different time points. Averaging "
            f"them is correct for one and not the other, so the app does not "
            f"decide it here.")
    if kind != "repeats":
        return 0, (
            f"A person's rows in `{group_col}` were recorded as different time "
            f"points rather than repeated measurements of the same quantity. "
            f"A usual-intake distribution is drawn from replicate days; a "
            f"longitudinal series is a different figure.")

    sizes = project.working_table.groupby(group_col).size()
    if sizes.empty:                                        # pragma: no cover
        return 0, "There are no rows to count."
    n = int(sizes.median())
    return n, (
        f"`{group_col}` identifies {len(sizes):,} people with a median of {n} "
        f"row(s) each, recorded as repeated measurements of the same quantity.")


def target_levels(project) -> int:
    """How many levels the recorded target has. `0` where none is recorded.

    A count rather than a boolean, because two different figures want two
    different readings of it — the volcano needs exactly two, and a scores
    plot colors by anything up to a legend's worth.
    """
    table = project.working_table
    if not project.target or project.target not in table.columns:
        return 0
    return int(table[project.target].nunique(dropna=True))


def likert_block(project):
    """The declared response block, from the pack's own detector.

    Through `packs.likert_block`, which is what the survey pack's
    `_ordinal_declared` detector reads — so the figure and the finding cannot
    disagree about what the instrument is.
    """
    from turbotab import packs

    try:
        return packs.likert_block(project.working_table)
    except Exception:                                      # pragma: no cover
        return None


def state(project) -> Dict[str, Any]:
    """What `when_applicable` reads. Recorded answers and measured shape only."""
    table = project.working_table
    numeric = _numeric_columns(table)
    lens = list(project.lens or [])
    n_recalls, recalls_why = recalls_per_person(project)
    n_levels = target_levels(project)
    block = likert_block(project)
    return {
        "task_type": project.task_type,
        # L40-C. The new figures read arity and item count, so `state` supplies
        # them — `when_applicable` may only read recorded answers and measured
        # shape, and both of these are the second.
        "n_classes": n_levels if project.task_type == "classification" else None,
        "n_items": len(block["columns"]) if block else 0,
        "has_instability_run": bool(
            getattr(project, "instability_runs", None) or {}),
        # A FOREST PLOT IS ABOUT COEFFICIENTS, so a project whose only fitted
        # models are trees has nothing to plot — and that is a different
        # sentence from "you have not trained yet".
        "has_coefficients": bool(_coefficients_for(project)),
        # NO LONGER FALSE FOR EVERY PROJECT (`GUIDED-065`). It is a fact about
        # this project now: whether a classification run has been fitted and
        # scored on the held-out rows.
        "has_predictions": predictions_for(project) is not None,
        "has_predictions_because": _no_predictions_because(project),
        "n_numeric": len(numeric),
        "n_rows": int(len(table)),
        "n_recalls_per_person": n_recalls,
        "n_recalls_because": recalls_why,
        "has_dietary_lens": _DIETARY in lens,
        # THE LENS DECIDES WHICH FIGURES EXIST FOR THIS PROJECT, which is
        # `DOMAIN_PACKS.md` §08 and `DRIVE-009`'s `act` field. An assay lens is
        # metabolomics or genomics: both packs specify the volcano, and both
        # mean the same thing by it.
        "has_assay_lens": bool({_METABOLOMICS, _GENOMICS} & set(lens)),
        "has_survey_lens": _SURVEY in lens,
        "n_target_levels": n_levels,
        "target_is_binary": n_levels == 2,
        "has_likert_block": block is not None,
        "likert_columns": list(block["columns"]) if block else [],
        "lens": lens,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Where each figure's numbers come from
# ─────────────────────────────────────────────────────────────────────────────

class FigureUnavailable(Exception):
    """This project cannot supply this figure's numbers, and here is why.

    Distinct from `when_applicable` returning False. That says *"this figure has
    nothing to say about this project"*; this says *"it does, and the numbers
    are not there"* — and the two read differently to a user, so they are
    reported separately.
    """


def _pca_payload(project, **_: Any) -> Dict[str, Any]:
    """PCA scores, with the metabolomics pack deciding what is overlaid.

    `DOMAIN_PACKS.md` §08 made executable: the QC overlay comes from the pack's
    own `pooled_qc` finding, so the checklist item about QCs scores against a
    detector's reading rather than against a renderer's guess.
    """
    table = project.working_table
    columns = [c for c in _numeric_columns(table) if c != project.target]
    if len(columns) < 2:
        raise FigureUnavailable(
            "fewer than two numeric columns remain once the target is set "
            "aside, and a scores plot needs two components")
    frame = table[columns].copy()

    qc_mask = None
    for finding in project.pack_findings():
        if finding["id"] != "pack::metabolomics::pooled_qc":
            continue
        params = finding["params"]
        column, value = params.get("column"), params.get("qc_value")
        if column in table.columns:
            qc_mask = [str(v) == str(value) for v in table[column]]
        break

    # COLORED BY THE TARGET AND NOT FITTED ON IT. Putting the outcome into the
    # matrix and then coloring by it is the circular-figure family's own shape
    # (`DOMAIN_SCIENCE.md` §01.6) — separation you built in. Carried as strings
    # so `select_dtypes` cannot pull it back into the fit.
    group_col = None
    if project.target and project.target in table.columns:
        if table[project.target].nunique(dropna=True) <= 8:
            group_col = "__group__"
            # `DRIVE-040`. THE LEVEL NAMES, WHERE THE RECORD HAS THEM. Run 5
            # read the annotation as `<NA> 15,552, 1 5,527, 0 770` — a legend
            # whose group names are the encoding rather than anything in the
            # user's column. `outcome_level_names` is empty where the record
            # cannot say, and then this is `str(v)` exactly as it was.
            from turbotab import training as _training

            names = _training.outcome_level_names(project)
            frame[group_col] = [names.get(v, str(v))
                                for v in table[project.target]]

    return figure_specs.pca_scores_payload(
        frame, group_col=group_col, qc_mask=qc_mask)


def _shrinkage_payload(project, *, nutrient: Optional[str] = None,
                       **_: Any) -> Dict[str, Any]:
    """The three densities, from this project's own repeated recalls."""
    from turbotab import nutrition

    table = project.working_table
    group_col = (project.grain or {}).get("group_col")
    if not group_col or group_col not in table.columns:    # pragma: no cover
        raise FigureUnavailable(
            "the grain answer names no column identifying a person's rows")
    column = nutrient or _default_nutrient(table, group_col)
    if column is None:
        raise FigureUnavailable(
            "no numeric nutrient column was found to draw this for")
    if column not in table.columns:
        raise FigureUnavailable(f"there is no column named '{column}'")

    built = nutrition.usual_intake_series(
        table, person_col=group_col, value_col=column)
    components = built["components"]
    payload = figure_specs.shrinkage_payload(
        built["series"], nutrient=column, n_days=int(built["n_days"]),
        modeled=True)
    # The variance-components table §03 says reviewers in this field ask for,
    # carried beside the figure rather than recomputed by whoever wants it.
    payload["method"] = built["method"]
    payload["variance_components"] = {
        "within": round(components.within, 4),
        "between": round(components.between, 4),
        "ratio": None if components.ratio is None else round(components.ratio, 3),
        "icc": None if components.icc is None else round(components.icc, 3),
        "lambda_observed": components.lambda_at(components.days_median),
        "n_people": components.n_people,
        "n_rows": components.n_rows,
    }
    return payload


def _default_nutrient(df: pd.DataFrame, group_col: Optional[str]) -> Optional[str]:
    """Total energy where the engine's reference matcher recognizes it.

    Through `packs._reference_column`, which goes through
    `physiology_reference.match_variable_key` — exact against the key or a
    declared alias, never a substring. Borrowing the vetted matcher is the
    opposite of adding a fifth name list.
    """
    from turbotab import packs

    energy = packs._reference_column(df, "kcal")
    if energy:
        return energy
    for column in _numeric_columns(df):
        if column != group_col and df[column].nunique(dropna=True) > 10:
            return column
    return None


def _volcano_payload(project, **_: Any) -> Dict[str, Any]:
    """The differential-abundance contrast, over the assay block only.

    The target is the contrast and is set aside from the features, for the
    reason the scores plot sets it aside from the matrix: a feature that is the
    outcome would be the most significant point on the panel and would mean
    nothing.
    """
    table = project.working_table
    features = [c for c in _numeric_columns(table) if c != project.target]
    if not project.target or project.target not in table.columns:
        raise FigureUnavailable(
            "a volcano plot contrasts two groups and no target is recorded, so "
            "there is nothing to contrast")
    if len(features) < 2:                                  # pragma: no cover
        raise FigureUnavailable("fewer than two features to test")
    return figure_specs.volcano_payload(
        table, group_col=project.target, feature_columns=features)


def _spline_payload(project, *, nutrient: Optional[str] = None,
                    **_: Any) -> Dict[str, Any]:
    """The dose–response, on the exposure the pack recognizes."""
    table = project.working_table
    if not project.target or project.target not in table.columns:
        raise FigureUnavailable(
            "a dose–response needs an outcome and no target is recorded")
    exposure = nutrient or _default_nutrient(table, project.target)
    if exposure is None or exposure not in table.columns:
        raise FigureUnavailable(
            "no numeric exposure column was found to draw this against")
    if exposure == project.target:                         # pragma: no cover
        raise FigureUnavailable(
            "the exposure and the outcome are the same column")
    grain = project.grain or {}
    person_col = (grain.get("group_col")
                  if grain.get("answer") == "people_repeat" else None)
    return figure_specs.spline_payload(
        table, exposure=exposure, outcome=project.target,
        person_col=person_col)


def _diverging_payload(project, **_: Any) -> Dict[str, Any]:
    """The Likert block, read by the same detector the survey pack uses."""
    from turbotab import packs

    table = project.working_table
    block = packs.likert_block(table)
    if block is None:                                      # pragma: no cover
        raise FigureUnavailable(
            "no block of columns sharing one declared response scale was found")
    return figure_specs.diverging_bar_payload(
        table, columns=block["columns"], scale=block["scale"])


def _calibration_payload(project, **_: Any) -> Dict[str, Any]:
    """`GUIDED-065`, closed. This used to be unconditionally unavailable, and
    the reason it gave was a fact about the APP rather than about the table:
    there was no training step, so no project could hold predictions.

    There is one now. The predictions come from the held-out rows and nowhere
    else — a calibration curve drawn on training predictions is a picture of a
    model grading its own homework, and it looks better the more it overfits.
    """
    from turbotab import figure_specs as _specs, training as _training

    run = predictions_for(project)
    if run is None:
        raise FigureUnavailable(
            "this project holds no model predictions to calibrate")
    y_true, y_proba, model_name, event = run
    payload = _specs.calibration_render(y_true, y_proba)
    payload["model"] = model_name
    payload["scored_on"] = "held-out rows only"
    # WHICH EVENT the curve is about, carried rather than assumed. On a 0/1
    # target this reads `1`; on `responder` / `non-responder` it is the
    # difference between a calibration plot and its mirror image.
    #
    # **`DRIVE-040`. THE LEVEL THE USER NAMED, NOT THE VALUE IT BECAME.** After
    # `L60-A` no binary classification reaches this line without a recorded
    # answer, and recording the answer ENCODES it — so `best.positive_label` is
    # `1` on every such run and the figure said `event: "1.0"`. Correct, and not
    # a name: a reader could not tell which level `1` is without opening the
    # transcript. The app stopped asserting something false and became less
    # legible in the same move; this is the rest of the move.
    #
    # Both travel. The encoded value is what the binarization above actually
    # used, and dropping it would leave the figure naming a level with nothing
    # tying it to the vector that was drawn.
    level = _training.chosen_event_level(project)
    payload["event"] = level if level is not None else str(event)
    payload["event_value"] = str(event)
    payload["event_named"] = level is not None
    return payload


#: Set by the training layer, read here. A function rather than an import so
#: the figure layer does not depend on where the run is stored — it asks the
#: project, and a project with no run answers `None` rather than raising.
def predictions_for(project):
    """`(y_true, y_proba, model_name, event)` for the best-calibratable run.

    `None` where there is nothing to calibrate — which is a different sentence
    from a failure, and the caller renders it as one.

    **The outcome is binarized against the class the probabilities are ABOUT**
    (`GUIDED-093`). `predict_proba`'s second column is `classes_[1]`; with a
    0/1 target that is `1` and nobody had to ask, and with `responder` /
    `non-responder` it is whichever sorts second. A curve drawn against the
    other class is a picture of the complementary event, drawn confidently, and
    that is the governing rule's *assert something false* branch. The event's
    name travels with the payload so the figure can say which.
    """
    run = getattr(project, "training_run", None)
    if run is None or getattr(run, "task_type", None) != "classification":
        return None
    from turbotab import training as _training

    scored = [r for r in run.results if r.probabilities]
    if not scored:
        return None
    y_true = _training.y_true_for(project)
    best = scored[0]
    if len(y_true) != len(best.probabilities):
        return None
    event = best.positive_label
    if event is None:
        # The run did not record which class the column is about. Refusing
        # here is the honest branch: guessing `1` would be right on a 0/1
        # target and silently wrong on every other one.
        return None
    binary = [1 if value == event else 0 for value in y_true]
    return binary, best.probabilities, best.name, event


def _risks_or_refuse(project):
    """`(y_true, {model_name: risks})` from the held-out rows, or a refusal.

    Every clinical figure added at L40 reads the same three things, so they
    read them in one place — four copies of *find the predictions* is four
    chances to disagree about which rows they came from, and the answer is
    always the same: the held-out rows and nowhere else.
    """
    run = predictions_for(project)
    if run is None:
        raise FigureUnavailable(
            "this project holds no model predictions on the held-out rows")
    y_true, y_proba, model_name, _event = run
    return y_true, {model_name: y_proba}


def _decision_curve_payload(project, **_: Any) -> Dict[str, Any]:
    y_true, risks = _risks_or_refuse(project)
    return figure_specs.decision_curve_payload(y_true, risks)


def _roc_payload(project, **_: Any) -> Dict[str, Any]:
    y_true, risks = _risks_or_refuse(project)
    return figure_specs.roc_payload(y_true, risks)


def _forest_payload(project, **_: Any) -> Dict[str, Any]:
    """Coefficients from a fitted LINEAR model, or a refusal that says why.

    §A4.7 is about coefficients, so a project whose only fitted models are
    trees has nothing to plot — and that is a different sentence from *you
    have not trained yet*, which is why it is said rather than left blank.
    """
    import numpy as np

    coefficients = _coefficients_for(project)
    if not coefficients:
        raise FigureUnavailable(
            "no fitted model in this project exposes coefficients — a forest "
            "plot is about coefficients, and a tree ensemble has none")
    return figure_specs.forest_payload(coefficients)


def _coefficients_for(project):
    """`[{name, estimate, low, high}]` from a fitted linear model.

    The interval is the coefficient plus or minus 1.96 standard errors where
    the estimator exposes them, and **the coefficient alone with no interval
    where it does not** — a forest plot of bare points is thinner than one
    with intervals and is not false, whereas an interval invented from nothing
    would be.
    """
    import numpy as np

    from turbotab import pipeline_plan as _plan_mod, training as _training
    from ml.model_registry import get_registry

    run = getattr(project, "training_run", None)
    if run is None or not getattr(project, "lockbox", None):
        return []

    # THE RUN DOES NOT RETAIN ITS FITTED ESTIMATORS — `ModelResult` carries the
    # PLAN as a dict and the predictions, which is the right thing to persist
    # and leaves nothing to read a coefficient off. So the recorded plan is
    # recomposed and refitted on the training rows with the run's own seed,
    # which is the same composition path and the same rows and therefore the
    # same model. Not a second engine: `pipeline_plan.compose` is the only
    # thing that builds a pipeline in this app.
    rows = project.training_rows
    target = str(project.target)
    rows = rows[rows[target].notna()]
    if rows.empty:
        return []
    X = _training.feature_frame(project, rows)
    y = rows[target]
    registry = get_registry()

    for result in getattr(run, "results", []) or []:
        if result.error or result.key not in registry:
            continue
        spec = registry[result.key]
        try:
            plan = _plan_mod.compose(project, result.key, X, seed=42)
            pipe = plan.build(spec.factory(project.task_type or "regression", 42))
            pipe.fit(X, y)
            estimator = pipe.named_steps.get("model")
        except Exception:
            continue
        if estimator is None or not hasattr(estimator, "coef_"):
            continue
        coef = np.ravel(np.asarray(estimator.coef_, dtype=float))
        try:
            names = [str(c) for c in pipe[:-1].get_feature_names_out()]
        except Exception:
            names = []
        if len(names) != len(coef):
            names = [f"predictor {i + 1}" for i in range(len(coef))]
        return [{"name": n, "estimate": float(c), "low": None, "high": None}
                for n, c in zip(names, coef)]
    return []


def _survey_frame(project):
    from turbotab import packs

    table = project.working_table
    block = packs.likert_block(table)
    if block is None:
        raise FigureUnavailable(
            "no block of columns sharing one declared response scale was found")
    return table[block["columns"]], block


def _scree_payload(project, **_: Any) -> Dict[str, Any]:
    frame, _block = _survey_frame(project)
    return figure_specs.scree_payload(frame)


def _item_correlations_payload(project, **_: Any) -> Dict[str, Any]:
    frame, _block = _survey_frame(project)
    return figure_specs.item_correlations_payload(frame)


def _floor_ceiling_payload(project, **_: Any) -> Dict[str, Any]:
    frame, block = _survey_frame(project)
    scale = list(block.get("scale") or [])
    # THE THEORETICAL LIMITS, from the DECLARED scale rather than the observed
    # values — §B5.3's rule, and the whole difference between "nobody is at the
    # ceiling" and "everybody is".
    low = high = None
    numeric = [v for v in scale if isinstance(v, (int, float))]
    if len(numeric) >= 2:
        low = float(min(numeric)) * len(frame.columns)
        high = float(max(numeric)) * len(frame.columns)
    return figure_specs.floor_ceiling_payload(frame, scale_min=low,
                                              scale_max=high)


def _item_panel_payload(project, **_: Any) -> Dict[str, Any]:
    frame, block = _survey_frame(project)
    return figure_specs.item_panel_payload(frame, scale=block.get("scale"))


SOURCES: Dict[str, Callable[..., Dict[str, Any]]] = {
    "pca_scores": _pca_payload,
    "shrinkage": _shrinkage_payload,
    "volcano": _volcano_payload,
    "dose_response_spline": _spline_payload,
    "diverging_stacked_bar": _diverging_payload,
    "calibration": _calibration_payload,
    # L40-C. Every one of the eight reaches `/figures`, which is `LOOP.md`
    # §05's rule and `GUIDED-058`'s whole subject: a figure registered and
    # unreachable is a specification with a passing test that no user can see.
    "decision_curve": _decision_curve_payload,
    "roc": _roc_payload,
    "forest": _forest_payload,
    "scree": _scree_payload,
    "item_correlations": _item_correlations_payload,
    "floor_ceiling": _floor_ceiling_payload,
    "item_panel": _item_panel_payload,
}


# ─────────────────────────────────────────────────────────────────────────────
# Why a figure is not drawn — stated, never silently omitted
# ─────────────────────────────────────────────────────────────────────────────

def _no_predictions_because(project) -> str:
    """WHICH of the reasons applies, rather than one sentence for all of them.

    A user whose regression project cannot draw a calibration plot and a user
    who simply has not trained yet are owed different sentences, and the second
    one is an instruction.
    """
    run = getattr(project, "training_run", None)
    if run is None:
        if not (project.lockbox and project.lockbox.get("labels")):
            return ("No model has been fitted yet, and none can be until the "
                    "held-out set is sealed — a calibration curve drawn on "
                    "rows the model was fitted on is a model grading its own "
                    "homework.")
        return ("No model has been fitted yet. Choose models in Train and the "
                "curve is drawn from the held-out predictions.")
    if getattr(run, "task_type", None) != "classification":
        return (f"Calibration is a claim about predicted PROBABILITIES, and "
                f"this is a {run.task_type} task — there are no probabilities "
                f"to compare against observed frequencies.")
    if not any(r.probabilities for r in run.results):
        return ("The models that were fitted do not produce probabilities, so "
                "there is nothing to calibrate.")
    return "There are predictions and the curve should be drawn."


def _why_not(figure_id: str, project_state: Dict[str, Any]) -> str:
    if figure_id == "calibration":
        return project_state["has_predictions_because"]
    if figure_id == "shrinkage":
        if not project_state["has_dietary_lens"]:
            return (
                "The lens question has not been answered *dietary intake*, and "
                "a usual-intake distribution is a claim about diet. The app "
                "does not infer the field from the column names.")
        return project_state["n_recalls_because"]
    if figure_id == "pca_scores":
        return (
            f"A scores plot needs at least two numeric columns and ten rows; "
            f"this table has {project_state['n_numeric']} and "
            f"{project_state['n_rows']}.")
    if figure_id == "volcano":
        if not project_state["has_assay_lens"]:
            return (
                "The lens question has not been answered metabolomics or "
                "genomics. A volcano plot is a differential-abundance claim "
                "over an assay panel, and the app does not infer the assay "
                "from the column count.")
        if not project_state["target_is_binary"]:
            return (
                f"A volcano contrasts exactly two groups and the recorded "
                f"target has {project_state['n_target_levels']}. With more "
                f"than two the fold change on the x-axis has no single "
                f"meaning, and picking a pair would be the app choosing your "
                f"contrast.")
        return (
            f"A volcano needs an assay panel; this table has "
            f"{project_state['n_numeric']} numeric columns.")
    if figure_id == "dose_response_spline":
        if not project_state["has_dietary_lens"]:
            return (
                "The lens question has not been answered dietary intake, and "
                "a dose–response curve of an outcome on an intake is a claim "
                "about diet.")
        if project_state["task_type"] != "regression":
            return (
                f"A dose–response curve needs a continuous outcome and this "
                f"project's task is {project_state['task_type'] or 'not yet '
                'recorded'}.")
        return (
            f"A spline with three knots needs more than "
            f"{project_state['n_rows']} rows to describe a dose–response "
            f"rather than a sample.")
    if figure_id == "diverging_stacked_bar":
        if not project_state["has_survey_lens"]:
            return (
                "The lens question has not been answered survey or "
                "questionnaire instruments. A block of small integers is a "
                "scale only where an instrument says it is.")
        return (
            "No block of columns sharing one declared response scale was "
            "found, so there is no instrument to lay across a zero line.")
    return "This figure does not apply to this project."   # pragma: no cover


# ─────────────────────────────────────────────────────────────────────────────
# The bundle
# ─────────────────────────────────────────────────────────────────────────────

def render(project, *, nutrient: Optional[str] = None) -> Dict[str, Any]:
    """Every figure this project can carry, drawn, scored and captioned.

    Three outcomes, and they are kept apart because they mean different things:

    * **admitted / held** — `figures.bundle()`'s own split. A CONFIRMATORY
      figure whose companion is absent is held, not captioned with a caveat a
      reader can skip.
    * **unavailable** — the figure applies and the numbers are not there. The
      refusal's own words are carried, so *"the within-person variance is not
      identifiable from these data"* reaches the user instead of an empty panel.
    * **not_drawn** — the figure does not apply, with the reason.
    """
    from turbotab.packs import PackRefusal

    project_state = state(project)
    applicable = figures.applicable(project_state)

    payloads: Dict[str, Dict[str, Any]] = {}
    unavailable: List[Dict[str, Any]] = []
    for spec in applicable:
        source = SOURCES.get(spec.id)
        if source is None:                                 # pragma: no cover
            unavailable.append({
                "id": spec.id, "title": spec.title,
                "why": "no source is registered for this figure"})
            continue
        try:
            payloads[spec.id] = source(project, nutrient=nutrient)
        except PackRefusal as refusal:
            # A REFUSAL, NOT AN ERROR. It carries a badge and an offer, and
            # dropping either at this boundary would be `DRIVE-001`'s class:
            # computed on the server, correct, and unreachable by a reader.
            unavailable.append({"id": spec.id, "title": spec.title,
                                "why": str(refusal), **refusal.to_dict()})
        except (FigureUnavailable, ValueError, KeyError) as exc:
            unavailable.append({"id": spec.id, "title": spec.title,
                                "why": str(exc)})

    out = figures.bundle(payloads)
    for row in out["admitted"] + out["held"]:
        row["payload"] = payloads[row["id"]]
    out["unavailable"] = unavailable
    out["not_drawn"] = [
        {"id": spec.id, "title": spec.title, "tier": spec.tier,
         "why": _why_not(spec.id, project_state)}
        for spec in figures.REGISTRY.values()
        if spec not in applicable]
    out["state"] = project_state
    return out
