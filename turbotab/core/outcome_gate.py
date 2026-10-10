"""The outcome's gates on the data routes (CROSSWALK disagreement 2 and question 1, ruled
2026-10-08; "the outcome beside a column"; calm/FOUNDATION §3 rule 6).

Two views of the outcome open at their own gates, and the server enforces both, as the explore
stage's relationship points already wait (``fit_press.relationships_served``, disagreement 4):

* **The outcome alone** (its distribution, blanks and event count; O1 in the brief's §6.1). Under
  Estimate and Describe (the engine's ``inference``) it opens after Who's in, which ends with the
  split TurboTab records for the person (``default:split_under_inference``), on the rows kept so
  far. Under Predict it opens once the person draws the held-out rows under Predict, on the
  training rows. With the goal unanswered it follows Estimate, the strictest case. Before its gate
  the outcome's summary keeps only what the outcome card shows (CROSSWALK, "The outcome card"):
  the rows that record it, its levels and their counts, and its range.
* **The outcome beside another column** (O3). Under Estimate and Describe, and with the goal
  unanswered, it opens with the plan's lock, recorded when Fit is pressed, on the rows analyzed
  and labeled exploratory; under Predict once the held-out rows are drawn, on the training rows
  only. A row window holds the outcome beside every other column it shows, so a window leaves the
  outcome out until this gate opens. Rows in file order are the outcome beside every column
  whichever columns one window holds, so a window of the outcome alone waits too.

Under Predict the gate stays shut while the current draw's held-out rows are not known (the split
recomputing), and a split TurboTab recorded under Estimate, or one recorded before the goal was
Predict, is no draw: the person has not decided which rows to hold out (:func:`drawn`).

Every view is served only once it is recorded (CROSSWALK, "The outcome alone: ... recorded";
``decision:view_outcome``). Reading records nothing, so a route that would serve an outcome view
not yet recorded for the rows it reads says so and hands back the record to post; once posted the
same request serves it. The record carries the rows it was drawn on (``rows_key``), so a changed
answer that moves those rows (an exclusion re-answered) needs a new record, and each look is in the
log as a forking path.

Every other route serves the outcome's own values only through these views: a stage's column
sample leaves the outcome out, the profile keeps only the card's fields for it, and a finding's
evidence leaves the outcome's distribution and its values by row out (:func:`served_stage`,
:func:`served_evidence`).
"""
from __future__ import annotations

import hashlib
from typing import Any, Literal, Mapping, Sequence

Rows = Literal["training", "analyzed"]

ALONE_AFTER_WHOS_IN = "The outcome · opens after Who's in"
BESIDE_AFTER_FIT = "The outcome beside another column · opens after Fit"
AFTER_THE_DRAW = "Opens once you decide which rows to hold out"
HELD_OUT_SEALED = "The outcome · its held-out rows stay sealed"
NOT_ANALYZED = "The outcome · shown only on the rows analyzed"
ROWS_BEING_COUNTED = "The outcome · opens once the rows kept so far are counted"
BEING_DRAWN = "The outcome · opens once the held-out rows are drawn"
NOT_RECORDED = "The outcome · opening it is recorded as looked at"
OWN_VIEW = "The outcome · its distribution is shown in its own view"
EXPLORATORY = "Exploratory · looked at after the plan was locked"

CARD_FIELDS = ("name", "dtype", "n", "n_missing", "n_unique", "min", "max", "top", "n_infinite")
SAMPLED_STAGES = ("ingest", "oriented", "working")


def _get(state: Any, name: str) -> Any:
    if state is None:
        return None
    if isinstance(state, Mapping):
        return state.get(name)
    return getattr(state, name, None)


def predicting(state: Any) -> bool:
    return _get(state, "purpose") == "prediction"


def drawn(records: Sequence[Any] | None) -> bool:
    """Whether the person drew the held-out rows under Predict: the live ``set_split`` holding the
    split now was recorded by the person while the goal was Predict. A split TurboTab recorded
    (``default:split_under_inference``), or one recorded under Estimate before the goal changed,
    is not a draw under Predict."""
    from turbotab.core import decisions
    from turbotab.core.seal import split_writer

    if not records:
        return False
    writer_id = split_writer(records)
    if writer_id is None:
        return False
    ordered = sorted(records, key=lambda r: r.seq)
    writer = next(r for r in ordered if r.id == writer_id)
    if getattr(writer, "recorded_by", "you") != "you":
        return False
    try:
        before = decisions.fold([r for r in ordered if r.seq < writer.seq])
    except Exception:  # noqa: BLE001 - a log the fold refuses draws nothing
        return False
    return before.purpose == "prediction"


def alone_gate(state: Any, drew: bool) -> dict[str, Any] | None:
    """Why the outcome alone is not served now, as a refusal reads it (``line``, ``exits``); None
    when its gate is open, or when no outcome is chosen. ``drew``: :func:`drawn`."""
    if not _get(state, "target"):
        return None
    if predicting(state):
        if drew:
            return None
        return {"line": AFTER_THE_DRAW, "exits": [{"label": "Decide which rows to hold out",
                                                    "decision": None}]}
    if _get(state, "split") is not None and _get(state, "purpose") is not None:
        return None
    if _get(state, "purpose") is None:  # an unanswered goal is the strictest case: Estimate
        return {"line": ALONE_AFTER_WHOS_IN, "exits": [{"label": "Choose the goal",
                                                         "decision": None}]}
    return {"line": ALONE_AFTER_WHOS_IN, "exits": [{"label": "Answer Who's in", "decision": None}]}


def beside_gate(state: Any, pressed: bool, drew: bool) -> dict[str, Any] | None:
    """Why the outcome is not served beside another column now; None when its gate is open, or
    when no outcome is chosen. ``pressed``: Fit was pressed for this outcome
    (``fit_press.pressed_for``); ``drew``: :func:`drawn`."""
    from turbotab.core.fit_press import serving_gate

    if not _get(state, "target"):
        return None
    if predicting(state):
        if drew:
            return None
        return {"line": AFTER_THE_DRAW, "exits": [{"label": "Decide which rows to hold out",
                                                    "decision": None}]}
    if _get(state, "purpose") is not None and serving_gate(state, pressed) is None:
        return None
    label = "Choose the goal" if _get(state, "purpose") is None else "Press Fit"
    return {"line": BESIDE_AFTER_FIT, "exits": [{"label": label, "decision": None}]}


def rows_read(state: Any) -> Rows:
    """Which rows a view of the outcome reads once open: the training rows under Predict, the rows
    analyzed (kept so far) otherwise."""
    return "training" if predicting(state) else "analyzed"


def rows_key(row_ids: Any) -> str:
    """The rows a view reads, as a short digest its record carries."""
    import numpy as np

    ids = np.sort(np.asarray(row_ids, dtype=np.int64))
    return hashlib.sha256(ids.tobytes()).hexdigest()[:16]


def view_record(view: str, columns: Sequence[str], key: str) -> dict[str, Any]:
    """The ``view_outcome`` a client records on opening an outcome view; the server fills the
    outcome, the rows and the levers (``stages.explore._view_filled``). A row window of the outcome
    beside other columns is the outcome's relationship with each of them."""
    return {"kind": "view_outcome", "view": view, "columns": list(columns), "rows_key": key}


def recorded(state: Any, view: str, columns: Sequence[str], key: str) -> bool:
    """Whether the outcome view is recorded for the rows it reads now: each of its entries names
    the current outcome and these rows."""
    target = _get(state, "target")
    views = _get(state, "outcome_views") or {}
    names = list(columns) if view == "relationship" else [target]
    for c in names:
        spec = views.get(f"{view}:{c}")
        if spec is None or _get(spec, "target") != target or _get(spec, "rows_key") != key:
            return False
    return True


def card_summary(summary: Mapping[str, Any], line: str) -> dict[str, Any]:
    """The outcome's column summary before its view is served: only what the outcome card shows,
    its distribution's statistics left out and the line saying when they open."""
    out = {k: (summary.get(k) if k in CARD_FIELDS else None) for k in summary}
    out["withheld"] = line
    return out


def served_stage(stage: str, artifact: Any, state: Any, line: str) -> Any:
    """A stage's artifact with the outcome's own values left to its views: no column sample of the
    outcome (a sample is the first rows, beside every other column's), and the profile's summary
    of it kept to the outcome card's fields with ``line``. ``artifact`` is never changed in
    place."""
    target = _get(state, "target")
    if not target or not isinstance(artifact, Mapping) or (
            stage not in SAMPLED_STAGES and stage != "profile"):
        return artifact
    columns = artifact.get("columns")
    if not isinstance(columns, list):
        return artifact
    out = []
    for c in columns:
        if isinstance(c, Mapping) and c.get("name") == target:
            if stage == "profile":
                c = card_summary(c, line)
            elif "sample" in c:
                c = {**c, "sample": []}
        out.append(c)
    return {**artifact, "columns": out}


def _mentions(text: Any, target: str) -> bool:
    return isinstance(text, str) and (f"`{target}`" in text or text == target)


def served_evidence(result: Any, state: Any) -> Any:
    """A finding's evidence (``consequences.PreviewResult``) with the outcome's own values left to
    its views: its values by row taken out of every table, and its distribution and any scatter
    against it dropped. A view left with nothing is dropped; with no view left, the note says
    where the outcome is shown."""
    target = _get(state, "target")
    if not target:
        return result
    views = []
    for v in result.views:
        kind = getattr(v, "kind", None)
        if kind == "table_focus" and (target in v.columns_before or target in v.columns_after):
            before = [c for c in v.columns_before if c != target]
            after = [c for c in v.columns_after if c != target]
            if not before and not after:
                continue
            rows = [r.model_copy(update={
                "before": {k: x for k, x in r.before.items() if k != target},
                "after": {k: x for k, x in r.after.items() if k != target}}) for r in v.rows]
            v = v.model_copy(update={
                "columns_before": before, "columns_after": after, "rows": rows, "story": [],
                "changed": [cell for cell in v.changed if cell[1] != target],
                "emphasis": [c for c in v.emphasis if c != target]})
        elif kind == "distribution" and v.column == target:
            continue
        elif kind == "relationship" and any(_mentions(t, target) for t in (
                v.x_label, v.y_label_before, v.y_label_after)):
            continue
        views.append(v)
    if len(views) == len(result.views) and all(a is b for a, b in zip(views, result.views)):
        return result
    note = result.note if views else OWN_VIEW
    return result.model_copy(update={"views": views, "note": note})
