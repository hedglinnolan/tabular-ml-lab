"""Fit, the hold and a visible lock (SIZING P0.8; RECIPES_AND_TUNING §4.4; CROSSWALK "Settled
here", "Fit, and when estimates appear"; calm/FOUNDATION §7).

Fit is pressed on the analysis flowchart, at the end of Models. Pressing it is a job command, not a
decision: it enters the Record only as the plan's lock, under Estimate and Describe.

* **What is served** (:func:`serving_gate`, :func:`served`). Computing stays live; what waits is
  what is shown.
  - With no purpose answered, no estimate stage is served: an unanswered purpose is the strictest
    case (guarantee test 2; EXTERNAL_AUDIT_2026-10-09 §3.2). Nothing computes either: every
    estimate stage requires the purpose (``stages.estimates_wait_for_the_purpose``).
  - Under Estimate and Describe (the engine's ``inference``), no estimate stage
    (``estimand.ESTIMATE_STAGES``) is served before the plan is locked, and pressing Fit records
    the lock: the existing system record ``lock_plan``, so no new decision kind.
  - Under Predict nothing locks, and until Fit is pressed for the outcome no estimate stage is
    served, its cross-validated scores included, so no score is marked seen
    (``models.selection.note_seen``) before Models' Confirm sweep is done (disagreement 12).
  - A question the estimates rest on that is still open speaks first (:func:`gate`), but never
    lets through a score the press withholds.
  - Under Estimate and Describe, and with no purpose, the explore stage's outcome relationships
    wait for the lock too (:func:`relationships_served`; disagreement 4).
* **The hold** (:func:`holds`; RECIPES §4.4, RT-8). A fit expected to take over about 2 minutes (a
  convention, :data:`HOLD_SECONDS`) waits for Fit in the scheduler; shorter fits compute live and
  may be done when Fit is pressed. The hold re-arms when a new estimate exceeds 1.5 times the last
  one confirmed (:data:`REARM`). Every estimate stage that fits the chosen families (one that
  reads ``models``) waits with the fit. The hold is the scheduler's, not a stage requirement, so
  stages stay pure functions of the log and a replay bypasses it.
* **The press** is kept beside the project (:data:`PRESS_FILE`: the outcome it was pressed for, the
  estimate it confirmed and the last record before it), as the scores seen are
  (``selection.SEEN_FILE``): it is not a decision and never enters the log. Cancel, or a change to
  the plan, before anything was shown marks it withdrawn (:func:`withdraw_press`), and with it the
  lock (``plan_lock.current_lock``; calm/FOUNDATION §7), so Fit is asked for again.
* **The lock is visible** (:func:`fit_lock`): the quest log says whether the plan is locked, when,
  its SHA-256 and why, in plain words. The lock is never called prespecified or preregistered
  (``plan_lock.NEVER_SAID``).
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, ConfigDict

# A fit expected to take over about 2 minutes does not start on its own (the ruling of
# 2026-10-06, a convention); the hold re-arms when a new estimate exceeds 1.5 times the last one
# confirmed (RECIPES §4.4).
HOLD_SECONDS = 120.0
REARM = 1.5
PRESS_FILE = "fit_pressed.json"

PURPOSE_FIRST = ("No estimate is shown, and nothing estimated is computed, until the goal is "
                 "chosen: whether the analysis estimates an effect, describes or predicts decides "
                 "which estimates there are and whether the plan locks.")
LOCK_FIRST = ("No estimate is shown until Fit is pressed: pressing it locks the analysis plan, so "
              "the plan is declared before any estimate is seen.")
FIT_FIRST = ("No score or estimate is shown until Fit is pressed, so none is seen before the "
             "models' settings are confirmed.")
PRESS_FIT = "Press Fit on the analysis flowchart"


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


# ── what is served ───────────────────────────────────────────────────────────


def serving_gate(state: Any, pressed: bool) -> dict[str, Any] | None:
    """Why no estimate stage may be served now, as ``estimand.withhold`` reads a gate; None when
    one may. ``pressed``: Fit was pressed for the state's outcome (:func:`pressed_for`)."""
    purpose = _get(state, "purpose")
    if purpose is None:
        return {"question": "purpose", "reason": PURPOSE_FIRST, "purpose": None, "scores": True,
                "exits": [{"label": "Choose the goal", "decision": None}]}
    if purpose == "prediction":
        if pressed:
            return None
        return {"question": "fit", "reason": FIT_FIRST, "purpose": purpose, "scores": True,
                "exits": [{"label": PRESS_FIT, "decision": None}]}
    # Estimate and Describe, and any goal not known here: the strict rule, the lock.
    if _get(state, "plan_locked"):
        return None
    return {"question": "fit", "reason": LOCK_FIRST, "purpose": purpose, "scores": True,
            "exits": [{"label": PRESS_FIT, "decision": None}]}


def gate(question: Mapping[str, Any] | None, state: Any, pressed: bool) -> dict[str, Any] | None:
    """The gate an estimate stage is served under: a question it rests on (``question``,
    ``estimand.served_gate``) and Fit (:func:`serving_gate`), composed so that neither lets through
    what the other withholds. With no purpose, the purpose's gate speaks (the strictest case);
    otherwise an open question speaks first, its reason and exits, and while Fit is not pressed
    every score waits with the estimates. None when both are clear."""
    fit = serving_gate(state, pressed)
    if fit is None:
        return dict(question) if question is not None else None
    if question is None or _get(state, "purpose") is None:
        return fit
    return {**question, "scores": True}


def served(stage: str, artifact: Any, state: Any, *, pressed: bool,
           question: Mapping[str, Any] | None = None) -> Any:
    """``artifact`` as an estimate stage may be served now: whole, or with every estimate (and
    every score) withheld and the reason first (:func:`gate`)."""
    from turbotab.core.estimand import withhold

    found = gate(question, state, pressed)
    return artifact if found is None else withhold(stage, artifact, found)


RELATIONSHIPS_AFTER_LOCK = ("How the outcome moves with this column is shown once Fit is pressed: "
                            "pressing it locks the analysis plan, so the plan is declared before "
                            "the outcome is read beside any column.")


def relationships_served(artifact: Any, state: Any, pressed: bool) -> Any:
    """The explore artifact as it may be served now (CROSSWALK, "the outcome beside a column";
    SIZING P0.8, disagreement 4): under Estimate and Describe, and with no purpose, each outcome
    relationship's points wait for the plan's lock with the estimates, and nothing is offered to
    record as viewed, since nothing is shown. Under prediction they are served after the draw, on
    the training rows. ``artifact`` is never changed in place."""
    if not isinstance(artifact, Mapping) or _get(state, "purpose") == "prediction":
        return artifact
    gate = serving_gate(state, pressed)
    if gate is None:
        return artifact
    reason = RELATIONSHIPS_AFTER_LOCK if _get(state, "purpose") is not None else PURPOSE_FIRST
    findings = []
    for f in artifact.get("findings") or []:
        if isinstance(f, Mapping) and f.get("kind") == "outcome_relationship":
            f = {**f, "points": [], "detail": reason, "record": None}
        findings.append(f)
    return {**artifact, "findings": findings}


# ── the press, and the hold ──────────────────────────────────────────────────


def read_press(project_dir: str | Path) -> dict[str, Any] | None:
    """The last press of Fit kept beside the project, or None."""
    try:
        found = json.loads((Path(project_dir) / PRESS_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return found if isinstance(found, dict) else None


def record_press(project_dir: str | Path, *, target: str | None, seconds: float | None,
                 seq: int | None) -> dict[str, Any]:
    """Keep a press of Fit beside the project: the outcome, the estimate it confirmed, the last
    record before it. Written atomically; the newest press replaces the last."""
    press = {"target": target, "seconds": seconds, "seq": seq}
    _write(project_dir, press)
    return press


def _write(project_dir: str | Path, press: Mapping[str, Any]) -> None:
    path = Path(project_dir) / PRESS_FILE
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(dict(press), f, sort_keys=True)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def withdraw_press(project_dir: str | Path) -> dict[str, Any] | None:
    """Cancel, or a change to the plan, before anything was shown: the press is kept, marked
    withdrawn, so Fit is asked for again and the hold re-arms (calm/FOUNDATION §7). Returns the
    press as kept, or None when there was none."""
    press = read_press(project_dir)
    if press is None or press.get("withdrawn"):
        return press
    kept = {**press, "withdrawn": True}
    _write(project_dir, kept)
    return kept


def pressed_for(press: Mapping[str, Any] | None, target: str | None) -> bool:
    """Whether Fit was pressed for this outcome, and not withdrawn since."""
    return press is not None and press.get("target") == target and not press.get("withdrawn")


def fit_estimate(shelf: Any, models: Sequence[str] | None) -> float | None:
    """What fitting the chosen families will take: the shelf's measured estimates, summed
    (``models.cost``; the models card says the same). None when none of them was measured."""
    shelf = _get(shelf, "data", shelf)
    if not isinstance(shelf, Mapping):
        return None
    timed = {f.get("key"): f.get("estimate_seconds") for f in shelf.get("families") or []
             if isinstance(f, Mapping)}
    found = [float(timed[k]) for k in models or () if timed.get(k) is not None]
    return float(sum(found)) if found else None


def holds(estimate: float | None, press: Mapping[str, Any] | None, target: str | None) -> bool:
    """Whether the scheduler holds the fit until Fit is pressed: its estimate exceeds about 2
    minutes, and no press for this outcome confirmed an estimate it is within 1.5 times of. An
    estimate not measured holds nothing: computing stays live."""
    if estimate is None or estimate <= HOLD_SECONDS:
        return False
    if not pressed_for(press, target):
        return True
    confirmed = (press or {}).get("seconds")
    return confirmed is not None and estimate > REARM * float(confirmed)


# ── the lock, as the quest log shows it ──────────────────────────────────────


class FitLock(BaseModel):
    """Fit and the plan's lock, as the quest log shows them (FOUNDATION §7: "Plan fixed at 14:02",
    with the plan's SHA-256 as its quiet label)."""

    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    purpose: str | None
    locks: bool  # whether this goal locks a plan (Estimate and Describe)
    locked: bool
    at: datetime | None  # when the lock was recorded (UTC)
    sha256: str | None  # the locked plan's SHA-256, as the lock records it
    pressed: bool  # Fit was pressed for this outcome
    # Results was opened by a press of Fit not withdrawn since, for this outcome and goal or
    # earlier ones: it stays reached when its results drop back, so it can say why (quest log)
    opened: bool
    held: bool  # the fit waits for Fit: its estimate exceeds about 2 minutes
    estimate_seconds: float | None  # what fitting the chosen families will take
    reason: str  # why, in plain words


def _when(at: datetime) -> str:
    return at.astimezone(timezone.utc).strftime("%Y-%m-%d at %H:%M UTC")


def fit_lock(state: Any, records: Sequence[Any], *, pressed: bool, held: bool,
             estimate: float | None, opened: bool = False) -> FitLock:
    """Whether the plan is locked, when, its fingerprint, and why, in plain words. ``opened``:
    a press of Fit not withdrawn since is kept, for any outcome (:func:`opened_by`)."""
    from turbotab.core.models.cost import duration
    from turbotab.core.plan_lock import current_lock

    purpose = _get(state, "purpose")
    lock = current_lock(records)
    locked = bool(_get(state, "plan_locked")) and lock is not None
    waits = (f" The fit waits for it: it is expected to take {duration(estimate)}."
             if held and estimate is not None else "")
    if purpose is None:
        reason = ("Nothing is locked and no estimate is shown: the goal is not chosen yet, and it "
                  "decides whether the plan locks.")
    elif purpose == "prediction":
        reason = ("Under prediction nothing locks: the held-out rows are the seal. "
                  + ("Fit was pressed, so the scores are shown." if pressed else
                     "Pressing Fit opens Results; until then no score or estimate is shown.")
                  + waits)
    elif locked:
        if getattr(lock.decision, "seen", None) == "prediction":
            reason = (f"The plan was locked on {_when(lock.at)}, when the goal became estimating: "
                      f"estimates had already been shown under prediction, so it was declared "
                      f"after they were seen. Every change since is marked so.")
        else:
            reason = (f"The plan was locked on {_when(lock.at)}, before any estimate was shown. "
                      f"Every change since is marked as made after the estimates were seen.")
    else:
        reason = ("The plan is not locked yet. Pressing Fit locks it and shows the first "
                  "estimates; until then no estimate is shown." + waits)
    return FitLock(
        purpose=purpose, locks=purpose not in (None, "prediction"), locked=locked,
        at=lock.at if locked else None,
        sha256=getattr(lock.decision, "digest", None) if locked else None,
        pressed=pressed, opened=bool(opened or pressed), held=held, estimate_seconds=estimate,
        reason=reason)


def opened_by(press: Mapping[str, Any] | None) -> bool:
    """Whether a press of Fit is kept and not withdrawn, whatever outcome it was pressed for."""
    return press is not None and not press.get("withdrawn")


__all__ = ["FIT_FIRST", "FitLock", "HOLD_SECONDS", "LOCK_FIRST", "PRESS_FILE", "PRESS_FIT",
           "PURPOSE_FIRST", "RELATIONSHIPS_AFTER_LOCK", "REARM", "fit_estimate", "fit_lock", "gate",
           "holds", "opened_by", "pressed_for", "read_press", "record_press", "relationships_served",
           "served", "serving_gate", "withdraw_press"]
