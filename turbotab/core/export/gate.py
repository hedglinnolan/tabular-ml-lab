"""When the export refuses, and what it names (V2 definition of done §1: export ends every
journey; §3.6).

A manuscript bundle states a finished analysis: its methods are the record's sentences as they
stand, its tables the declared result. So the export refuses — HTTP 409, never a partial bundle —
while anything it would report is not settled yet, and the refusal names each thing missing, with
a way forward for each (BLUEPRINT §3's refusal shape):

* **an input file changed** since TurboTab read it: no record could name the data the results
  came from (its bytes no longer match the fingerprint recorded when it was read);
* **a required question is unanswered**: any question the Router has open or waiting, except the
  ones that only add a display nothing declared depends on (:data:`OPTIONAL_QUESTIONS`: the
  substitution curve, drawn or not at the analyst's choice) and the opening of the held-out rows,
  which declares the result and is named below as the open plan (:data:`PLAN_QUESTIONS`); a
  regression calibration declared under another adjustment set is re-asked, so it counts too;
* **the plan is open**:
  - under inference, the analysis plan is not locked: it locks the first time an estimate is
    displayed (``plan_lock``), so a bundle exported before that would report estimates no plan was
    declared for;
  - under prediction, no result is declared: the held-out rows are drawn and still sealed, so the
    final model is not declared and no held-out score is the result yet; or nothing could be
    declared (``selection.declared_result``'s "not_declared");
* **a result is not computed**: a stage the bundle reports (:func:`result_stages`) is still being
  computed, waits for an answer, or failed;
* **the estimates are withheld** while a question they rest on is unanswered (WP17's gate).

A sentence of the record that counts rows (the exclusions', the complete cases') is not a reason to
refuse when a later answer changed those rows: the methods text restates its counts on the rows as
they stand (``voice.restated_counts``), so it agrees with the participant flow beside it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from turbotab.core.decisions import Refusal

# The Router's questions whose answer only adds a display the declared analysis never reads.
OPTIONAL_QUESTIONS = ("substitution",)
# The question that declares the result under prediction: an open plan, named as such below.
PLAN_QUESTIONS = ("open_seal",)
# What the refusal calls each stage the bundle reports.
STAGE_NAMES = {"cohort": "The participant flow", "design": "The model matrix",
               "fit": "The fit", "effects": "Table 2 (the declared models)",
               "calibration": "The regression calibration (the declared secondary analysis)"}
OPEN = ("open", "waiting")


@dataclass
class Missing:
    """One thing the export waits for: its code, what it is in a sentence, and its ways forward."""

    code: str
    message: str
    exits: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        from turbotab.core.voice import finish

        self.message = finish(self.message)


def calibration_declared(state: Any) -> bool:
    """Regression calibration is declared as a secondary analysis under inference (MS5)."""
    spec = getattr(state, "measurement_error", None)
    return (getattr(state, "purpose", None) == "inference" and spec is not None
            and getattr(spec, "method", "none") != "none")


def result_stages(state: Any) -> tuple[str, ...]:
    """The stages whose artifacts the bundle reports: the participant flow, the design (the lineage
    and the model matrix), the fit, and under inference the declared model sequence (Table 2) and
    a declared regression calibration (REPAIR-RC: the declared secondary reaches the bundle)."""
    stages = ("cohort", "design", "fit")
    if getattr(state, "purpose", None) != "inference":
        return stages
    return (*stages, "effects", *(("calibration",) if calibration_declared(state) else ()))


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _question_name(key: str) -> str:
    from turbotab.core.voice import question_name

    return question_name(key)


def _listed(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def missing(source: Any) -> list[Missing]:
    """Everything the export waits for, most fundamental first (module docstring)."""
    out: list[Missing] = []
    for f in source.inputs:
        if f.missing:
            out.append(Missing("input_missing", (
                f"The {f.role} `{f.name}` is no longer at {f.path}, so its hash cannot be recorded "
                f"for the provenance record."),
                [{"label": f"Put `{f.name}` back where it was read", "decision": None}]))
        elif f.changed:
            out.append(Missing("input_changed", (
                f"The {f.role} `{f.name}` changed since TurboTab read it: its bytes no longer match "
                f"the copy the analysis was made from, so no record could name the data the results "
                f"came from."),
                [{"label": f"Restore `{f.name}` as it was read, or open the new file as a new "
                           f"project", "decision": None}]))
    open_steps = [s for s in source.interview if _get(s, "status") in OPEN
                  and _get(s, "key") not in (*OPTIONAL_QUESTIONS, *PLAN_QUESTIONS)]
    if open_steps:
        names = [_question_name(str(_get(s, "key"))) for s in open_steps]
        out.append(Missing("unanswered_questions", (
            f"{_listed(names)[:1].upper()}{_listed(names)[1:]} "
            f"{'is' if len(names) == 1 else 'are'} not answered yet; the methods would leave "
            f"{'it' if len(names) == 1 else 'them'} out."),
            [{"label": f"Answer {name}", "decision": None} for name in names]))
    state = source.state
    purpose = getattr(state, "purpose", None)
    fit = source.artifact("fit")
    if calibration_declared(state):
        # REPAIR-RC (MODELING_SEQUENCE §2): a calibration declared under another adjustment set is
        # re-asked, so it is an unanswered question until it is declared again or recorded as none.
        from turbotab.core.stages.calibration import current_calibration, invalidated

        spec, recorded = current_calibration(state)
        if recorded is not None:
            reason, exits = invalidated(spec, recorded, state)
            out.append(Missing("unanswered_questions", reason, exits))
    if purpose == "inference" and not getattr(state, "plan_locked", None):
        out.append(Missing("plan_open", (
            "The analysis plan is still open: under inference it is locked the first time an "
            "estimate is displayed, and the export reports only estimates of a declared plan."),
            [{"label": "Show the estimates; the first one shown locks the plan",
              "decision": None}]))
    elif purpose == "prediction" and isinstance(fit, dict):
        result = fit.get("result") or {}
        basis = result.get("basis")
        if basis == "holdout" and not getattr(state, "seal_opened", None):
            labels = {m["family"]: m.get("label") or m["family"] for m in fit.get("models") or []}
            out.append(Missing("plan_open", (
                "The final model is not declared: the held-out rows are still sealed, so no "
                "held-out score is the result yet. Declare the final model on cross-validation, "
                "then open them once."),
                [{"label": f"Declare {label} final and open the held-out rows",
                  "decision": {"kind": "open_seal", "family": family}}
                 for family, label in labels.items()]))
        elif basis == "not_declared":
            out.append(Missing("plan_open", (
                f"No result is declared: {result.get('sentence') or 'the fit declared none.'}"),
                [{"label": "Choose the model families again", "decision": None}]))
    for stage in result_stages(state):
        status = source.status(stage)
        name = str(_get(status, "status") or "idle")
        if name == "fresh":
            continue
        label = STAGE_NAMES.get(stage, f"The {stage} stage")
        if name == "error":
            out.append(Missing("result_failed", (
                f"{label} failed, so its result cannot be reported: "
                f"{_get(status, 'error') or 'no reason was given.'}"),
                [{"label": f"Compute {label[:1].lower()}{label[1:]} again", "decision": None}]))
        elif name == "blocked":
            waits = [_question_name(str(m)) for m in _get(status, "missing") or []]
            out.append(Missing("result_not_ready", (
                f"{label} waits for {_listed(waits) or 'an answer'} before it can run."),
                [{"label": "Answer the question it waits for", "decision": None}]))
        else:
            out.append(Missing("result_not_ready", (
                f"{label} is still being computed for the current answers."),
                [{"label": "Export again once it is done", "decision": None}]))
    withheld = fit.get("withheld") if isinstance(fit, dict) else None
    if withheld and not open_steps:
        out.append(Missing("estimates_withheld", f"The estimates are withheld: {withheld}",
                           [{"label": "Answer the question they rest on", "decision": None}]))
    return out


def refusal(found: list[Missing]) -> Refusal:
    """One refusal naming everything missing, its exits in that order (duplicates once)."""
    message = " ".join(m.message for m in found)
    exits: list[dict[str, Any]] = []
    seen: set[str] = set()
    for m in found:
        for e in m.exits:
            if e["label"] not in seen:
                seen.add(e["label"])
                exits.append(e)
    return Refusal(found[0].code, message, exits=exits)


def check(source: Any) -> None:
    """Raise the export's refusal when anything is missing (module docstring)."""
    found = missing(source)
    if found:
        raise refusal(found)


__all__ = ["Missing", "OPTIONAL_QUESTIONS", "PLAN_QUESTIONS", "calibration_declared", "check",
           "missing", "refusal", "result_stages"]
