"""The ``target_info`` stage: what kind of outcome the chosen target is."""
from __future__ import annotations

from typing import Any

from turbotab.core.graph import StageContext
from turbotab.core.stages.data import open_store

HISTOGRAM_BINS = 30
MAX_CLASSES = 100
NUMERIC = ("numeric", "integer")
_CONFIDENCE = {"high": "high", "med": "medium", "medium": "medium", "low": "low"}


def target_info_stage(ctx: StageContext) -> dict[str, Any]:
    """Detect the task from the target column; the recorded ``task`` overrides it.

    Detection is ``turbotab.engine.detect_task_type`` (``ml.triage``). It
    answers regression or classification; classification is split into binary
    or multiclass by the number of distinct values.
    """
    target = ctx.state.target
    columns = {c["name"]: c for c in ctx.inputs["ingest"]["columns"]}
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
        reasons = [str(r) for r in detection.get("reasons") or []]
        if detection.get("detected") == "classification":
            detected_task = "binary" if n_classes <= 2 else "multiclass"
            reasons.append(
                f"It has {n_classes:,} distinct value{'s' if n_classes != 1 else ''}, "
                f"so it reads as {detected_task} classification."
            )
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
        "reason": " ".join(reasons),
        "histogram": histogram,
        "classes": classes,
    }
