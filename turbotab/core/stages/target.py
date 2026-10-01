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
        if detection.get("detected") == "classification":
            detected_task = "binary" if n_classes <= 2 else "multiclass"
        else:
            detected_task = "regression"
        confidence = _CONFIDENCE.get(str(detection.get("confidence")), "low")
        task = ctx.state.task or detected_task

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

    return {
        "column": target,
        "task": task,
        "detected_task": detected_task,
        "confidence": confidence,
        "reason": task_reason(series, detection, detected_task),
        "histogram": histogram,
        "classes": classes,
    }
