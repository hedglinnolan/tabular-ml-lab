"""The outcome's gates on the data routes (CROSSWALK disagreement 2 and question 1, ruled
2026-10-08; "the outcome beside a column"; calm/FOUNDATION §3 rule 6).

Two views of the outcome open at their own gates, and the server enforces both, as the explore
stage's relationship points already wait (``fit_press.relationships_served``, disagreement 4):

* **The outcome alone** (its distribution, blanks and event count; O1 in the brief's §6.1). Under
  Estimate and Describe (the engine's ``inference``) it opens after Who's in, which ends with the
  split TurboTab records for the person (``default:split_under_inference``), on the rows kept so
  far. Under Predict it opens once the held-out rows are drawn, on the training rows. With the goal
  unanswered it follows Estimate, the strictest case. Before its gate the outcome's summary keeps
  only what the outcome card shows: the rows that record it, its levels and their counts, and its
  range (the card's unit and impossible-value questions read it).
* **The outcome beside another column** (O3). Under Estimate and Describe, and with the goal
  unanswered, it opens with the plan's lock, recorded when Fit is pressed; under Predict once the
  held-out rows are drawn, on the training rows only. A row window holds the outcome beside every
  other column it shows, so a window leaves the outcome out until this gate opens. Rows in file
  order are the outcome beside every column whichever columns one window holds, so a window of the
  outcome alone waits too.

Every view of the outcome alone that is served carries the ``view_outcome`` record a client posts
on opening it (``decision:view_outcome``), as the explore stage's outcome views do; reading a route
records nothing.
"""
from __future__ import annotations

from typing import Any, Literal, Mapping

Rows = Literal["training", "analyzed"]

ALONE_AFTER_WHOS_IN = "The outcome · opens after Who's in"
BESIDE_AFTER_FIT = "The outcome beside another column · opens after Fit"
AFTER_THE_DRAW = "Opens once you decide which rows to hold out"
HELD_OUT_SEALED = "The outcome · its held-out rows stay sealed"
ROWS_BEING_COUNTED = "The outcome · opens once the rows kept so far are counted"

CARD_FIELDS = ("name", "dtype", "n", "n_missing", "n_unique", "min", "max", "top", "n_infinite")


def _get(state: Any, name: str) -> Any:
    if state is None:
        return None
    if isinstance(state, Mapping):
        return state.get(name)
    return getattr(state, name, None)


def predicting(state: Any) -> bool:
    return _get(state, "purpose") == "prediction"


def alone_gate(state: Any) -> dict[str, Any] | None:
    """Why the outcome alone is not served now, as a refusal reads it (``line``, ``exits``); None
    when it is, or when no outcome is chosen."""
    if not _get(state, "target") or _get(state, "split") is not None:
        return None
    if predicting(state):
        return {"line": AFTER_THE_DRAW, "exits": [{"label": "Decide which rows to hold out",
                                                    "decision": None}]}
    if _get(state, "purpose") is None:  # an unanswered goal is the strictest case: Estimate
        return {"line": ALONE_AFTER_WHOS_IN, "exits": [{"label": "Choose the goal",
                                                         "decision": None}]}
    return {"line": ALONE_AFTER_WHOS_IN, "exits": [{"label": "Answer Who's in", "decision": None}]}


def beside_gate(state: Any, pressed: bool) -> dict[str, Any] | None:
    """Why the outcome is not served beside another column now; None when it is, or when no
    outcome is chosen. ``pressed``: Fit was pressed for this outcome (``fit_press.pressed_for``)."""
    from turbotab.core.fit_press import serving_gate

    if not _get(state, "target"):
        return None
    if predicting(state):
        if _get(state, "split") is not None:
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


def view_record() -> dict[str, Any]:
    """The ``view_outcome`` a client records on opening the outcome alone; the server fills the
    outcome, the rows and the levers (``stages.explore._view_filled``)."""
    return {"kind": "view_outcome", "view": "distribution", "columns": []}


def card_summary(summary: Mapping[str, Any], line: str) -> dict[str, Any]:
    """The outcome's column summary before its gate: only what the outcome card shows, its
    distribution's statistics left out and the line saying when they open."""
    out = {k: (summary.get(k) if k in CARD_FIELDS else None) for k in summary}
    out["withheld"] = line
    return out
