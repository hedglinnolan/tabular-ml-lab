"""The ``target_info`` stage: what kind of outcome the chosen target is."""
from __future__ import annotations

from typing import Any

import pandas as pd

from turbotab.core.graph import StageContext
from turbotab.core.stages.data import open_store
from turbotab.core.stages.working import table_info

HISTOGRAM_BINS = 30
MAX_CLASSES = 100
NUMERIC = ("numeric", "integer")
_CONFIDENCE = {"high": "high", "med": "medium", "medium": "medium", "low": "low"}


def task_reason(series: pd.Series, detection: dict[str, Any], detected_task: str) -> str:
    """Why the column reads as its task, in the app's voice: one sentence, its data first.

    ``Continuous, with 4,213 distinct values — read as a regression outcome.`` The branches
    follow ``ml.triage.detect_task_type``, so the sentence states the evidence the detection used.
    """
    from turbotab.core.voice import finish, number, tick

    values = series.dropna()
    n_rows = int(len(series))
    k = int(values.nunique())
    read = f"read as a {detected_task} outcome"
    kind = series.dtype
    if k == 0:
        return finish(f"No value is recorded, so there is nothing to go on — {read}")
    levels = sorted(values.unique(), key=lambda v: (str(type(v)), str(v)))
    if pd.api.types.is_bool_dtype(kind):
        return finish(f"True or false — {read}")
    if (pd.api.types.is_object_dtype(kind) or pd.api.types.is_string_dtype(kind)
            or isinstance(kind, pd.CategoricalDtype)):
        if k == 1:
            return finish(f"Only one value, {tick(levels[0])}, so there is nothing to tell apart "
                          f"— {read}")
        if k == 2:
            return finish(f"Text with two values, {tick(levels[0])} and {tick(levels[1])} — {read}")
        return finish(f"Text with {k:,} distinct values — {read}")
    if {number(v) for v in levels} == {"0", "1"}:
        blanks = ", with blanks" if n_rows > len(values) else ""
        return finish(f"Only {tick(0)} and {tick(1)}{blanks} — {read}")
    numeric = pd.api.types.is_numeric_dtype(kind)
    whole = numeric and (pd.api.types.is_integer_dtype(kind) or bool((values == values.round()).all()))
    if detection.get("detected") == "classification":
        if whole and k <= 10:
            return finish(f"Whole numbers with {k:,} distinct values; class codes, counts and "
                          f"ordinal scores all look like this — {read}")
        if k <= 10:
            return finish(f"Numbers with only {k:,} distinct values — {read}")
        return finish(f"Numbers, but only {k:,} distinct values across {n_rows:,} rows — {read}")
    if not numeric:
        return finish(f"Its type gives little to go on — {read}")
    if whole:
        return finish(f"Whole numbers with {k:,} distinct values — {read}")
    return finish(f"Continuous, with {k:,} distinct values — {read}")


def target_info_stage(ctx: StageContext) -> dict[str, Any]:
    """Detect the task from the target column; the recorded ``task`` overrides it.

    Detection is ``turbotab.engine.detect_task_type`` (``ml.triage``). It
    answers regression or classification; classification is split into binary
    or multiclass by the number of distinct values. ``reason`` restates the
    detection's evidence in the app's voice (:func:`task_reason`).
    """
    target = ctx.state.target
    columns = {c["name"]: c for c in table_info(ctx)["columns"]}
    column = columns.get(target)
    if column is None:
        raise ValueError(f"There is no column named {target!r} in this dataset.")

    from turbotab import engine
    from turbotab.core.datastore import json_safe

    with open_store(ctx) as store:
        frame = store.materialize([target]).reset_index(drop=True)
        ctx.progress(0.3, "Detecting the task")
        detection = engine.detect_task_type(frame, target)
        series = frame[target]
        n_classes = int(series.nunique(dropna=True))
        numeric = column["dtype"] in NUMERIC
        from turbotab.core.decisions import MAX_CLASSES_MULTICLASS, task_fits

        # Audit RO-10 (WP18): the tasks the explicit answer accepts, by the one rule both share.
        fits = task_fits(n_classes, numeric)
        if detection.get("detected") == "classification":
            detected_task = "binary" if n_classes <= 2 else "multiclass"
        else:
            detected_task = "regression"
        confidence = _CONFIDENCE.get(str(detection.get("confidence")), "low")
        too_many = detected_task == "multiclass" and n_classes > MAX_CLASSES_MULTICLASS
        if too_many and "regression" in fits:
            # More classes than multiclass takes: a number with that many values is a quantity
            # read on a coarse grid, proposed (never skipped) as regression.
            detected_task, confidence = "regression", "low"
        # BLUEPRINT §14.1 (the readings ledger): the task is skipped only on a settled reading. Two
        # levels or a continuous number leave no doubt; three or more labels may be ordered (none,
        # mild, severe: an ordinal outcome) or not, which no dtype says, so it is asked.
        from turbotab.core.readings import task_reading

        found = task_reading(target, detected_task, confidence)
        unordered_doubt = found.confidence != confidence
        confidence = found.confidence
        if detected_task not in fits:
            confidence = "low"  # a task the answer would refuse is never stated (the 20-class rule)
        task = ctx.state.task or detected_task
        from turbotab.core.structural import log_outcome, order_question, scale_question

        levels_question = None
        if detected_task == "multiclass" or task in ("multiclass", "ordinal"):
            listed = sorted(series.dropna().unique(), key=lambda v: (str(type(v)), str(v)))
            levels_question = order_question([json_safe(v) for v in listed], target,
                                             numeric)
        # A positive, markedly skewed outcome is asked its scale (RO-10); the log of one already
        # chosen is not asked again.
        scale = (scale_question(series, target) if task == "regression" and numeric
                 and log_outcome(ctx.state) is None else None)

        histogram = None
        classes = None
        if task == "regression" and column["dtype"] in NUMERIC:
            histogram = store.histogram(target, bins=HISTOGRAM_BINS)
        else:
            counts = series.dropna().value_counts(sort=False)
            ranked = sorted(counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))
            classes = [
                {"value": json_safe(value), "count": int(count)}
                for value, count in ranked[:MAX_CLASSES]
            ]

    from turbotab.core.units import outcome_unit, proposed_unit, recorded_unit

    # The outcome's unit, for every outcome quantity the app shows (M2_CONTRACT §6). A class
    # label has none. Stated only as recorded (BLUEPRINT §14.3: a header's letters are a name);
    # the name's letters and the clinical pack's reading are proposals for ``set_outcome_unit``,
    # never stated (audit IN-05).
    regression = task == "regression"
    unit, unit_source = (outcome_unit(target, recorded=recorded_unit(ctx.state, target))
                         if regression else (None, None))
    proposal = None
    if regression and unit is None:
        # The name's own letters first (the header's unit, never checked against the values),
        # then the clinical pack's reading: each a proposal, never stated.
        from turbotab.core.units import from_name

        from turbotab.core.readings import BARE_AMOUNTS

        named = from_name(target)
        pack = proposed_unit(target, series)
        if named in BARE_AMOUNTS and (pack or {}).get("candidates") \
                and named not in pack["candidates"]:
            named = None  # ``ldl_mg``: a bare amount the quantity does not take (mg/dL, mmol/L)
        if named is not None:
            candidates = [named, *[u for u in (pack or {}).get("candidates") or [] if u != named]]
            proposal = {"unit": named, "candidates": candidates, "source": "name"}
        else:
            proposal = pack
    reason = task_reason(series, detection, detected_task)
    if too_many:
        reason += (f" Multiclass takes at most {MAX_CLASSES_MULTICLASS} classes, so "
                   + ("it is proposed as a number on a coarse grid." if "regression" in fits else
                      "it cannot be modeled as it is: group its levels, or choose another "
                      "outcome."))
    elif unordered_doubt:
        reason += " Whether its levels are ordered (an ordinal outcome) is yours to say."
    if scale is not None:
        reason += f" {scale['evidence']} Its scale is yours to choose."
    return {
        "column": target,
        "task": task,
        "detected_task": detected_task,
        "confidence": confidence,
        "reason": reason,
        "fits": fits,
        "order_question": levels_question,
        "scale_question": scale,
        "histogram": histogram,
        "classes": classes,
        "unit": unit,
        "unit_source": unit_source,
        # Proposed for the user's decision when nothing states it; never in a sentence.
        "proposed_unit": proposal["unit"] if proposal else None,
        "unit_candidates": list(proposal["candidates"]) if proposal else [],
    }
