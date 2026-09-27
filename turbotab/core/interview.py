"""The interview Router (M1_CONTRACT.md §1): which question is asked now, and why the others wait.

A pure function of the recorded state, the stage statuses and a few small artifacts. The client
renders what it says and never decides the order itself.

Rules:

* Questions are asked in :data:`QUESTION_KEYS` order. A question whose slot holds an answer is
  ``answered`` (with the record that wrote it); changing an earlier answer never reopens a later
  one — it stays answered and its stages go stale.
* ``task`` is ``skipped`` while detection is high-confidence for the current outcome and no
  answer overrides it; ``reason`` is the detection. The client's "Ask me anyway" reopens it.
* ``energy_adjustment`` is ``not_applicable`` unless the lens includes ``dietary``, a column has
  the ``energy`` role and an ``exposure`` carries energy; ``reason`` names what is missing.
  ``substitution`` is ``not_applicable`` with fewer than two energy-bearing exposures.
* At most one question is ``open``: the first applicable unanswered one. It is ``waiting``
  instead while a stage it needs is still being computed (``waiting_on`` names the stage) —
  ``substitution`` waits until ``fit`` is fresh. Every later unanswered question is ``waiting``
  on the earliest unanswered question before it (plus any stage of its own still computing).
  A stage that failed or was stopped does not hold a question back: the question opens and the
  pipeline panel shows the failure.
"""
from __future__ import annotations

from typing import Any, Callable, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict

QuestionKey = Literal[
    "lens", "target", "task", "purpose", "roles", "exclusions", "missing", "split",
    "energy_adjustment", "models", "substitution",
]
QUESTION_KEYS: tuple[str, ...] = (
    "lens", "target", "task", "purpose", "roles", "exclusions", "missing", "split",
    "energy_adjustment", "models", "substitution",
)
StepStatus = Literal["answered", "open", "waiting", "skipped", "not_applicable"]

# The stage a question needs before it can be shown (its options come from it).
NEEDS: dict[str, tuple[str, ...]] = {
    "lens": ("ingest",),
    "target": ("ingest",),
    "task": ("target_info",),
    "purpose": (),
    "roles": ("roles",),
    "exclusions": ("proposals",),
    "missing": (),
    "split": (),
    "energy_adjustment": ("proposals",),
    "models": ("shelf",),
    "substitution": ("fit",),
}
MUST_BE_FRESH = {"substitution": "fit"}


class InterviewStep(BaseModel):
    model_config = ConfigDict(json_schema_serialization_defaults_required=True)

    key: QuestionKey
    status: StepStatus
    decision_id: str | None = None
    reason: str | None = None
    waiting_on: list[str] = []


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _deps() -> dict[str, tuple[str, ...]]:
    try:
        from turbotab.core.graph import load_graph
        from turbotab.core.stages import GRAPH_FACTORY

        return {s.name: s.deps for s in load_graph(GRAPH_FACTORY).stages()}
    except Exception:  # pragma: no cover - a graph that cannot load has nothing pending
        return {}


def pending_stages(stages: Mapping[str, Any], deps: Mapping[str, Sequence[str]] | None = None) -> set[str]:
    """Stages whose work for the current answers is under way or about to start.

    Queued or running; or idle/stale (not stopped by request) while a stage it depends on is
    pending — it starts when that one finishes.
    """
    deps = _deps() if deps is None else deps
    memo: dict[str, bool] = {}

    def pending(name: str) -> bool:
        if name in memo:
            return memo[name]
        memo[name] = False  # guards a (malformed) cycle
        status = stages.get(name)
        state = _get(status, "status")
        result = state in ("queued", "running") or (
            state in ("idle", "stale")
            and not _get(status, "cancelled", False)
            and any(pending(d) for d in deps.get(name, ()))
        )
        memo[name] = result
        return result

    return {name for name in stages if pending(name)}


def _live_writer(records: Sequence[Any], state: Any) -> dict[str, str]:
    """Slot -> id of the live record that wrote its current value."""
    from turbotab.core.decisions import SLOTS, Revert, reverted

    ordered = sorted(records, key=lambda r: r.seq)
    try:
        cancelled = reverted(ordered)
    except Exception:
        cancelled = {}
    out: dict[str, str] = {}
    for record in ordered:
        decision = record.decision
        if record.id in cancelled or isinstance(decision, Revert) or decision.kind not in SLOTS:
            continue
        slot = SLOTS[decision.kind]
        if decision.kind == "set_task" and (
            decision.column != state.target or decision.task != state.task
        ):
            continue
        out[slot] = record.id
    return out


def _energy_applicability(state: Any, bearing: Callable[[str], bool]) -> str | None:
    """Why energy adjustment does not apply, or None when it does (or cannot tell yet)."""
    if state.lens is not None and "dietary" not in state.lens:
        return "The dietary lens is off, so energy adjustment does not apply."
    if state.roles is None:
        return None
    roles = state.roles
    if not any(r == "energy" for r in roles.values()):
        return "No column has the energy role, so there is no total energy to adjust against."
    if not any(r == "exposure" and bearing(c) for c, r in roles.items()):
        return "No exposure is a nutrient that carries energy, so there is nothing to adjust."
    return None


def _substitution_applicability(state: Any, bearing: Callable[[str], bool]) -> str | None:
    if state.roles is None:
        return None
    n = sum(1 for c, r in state.roles.items() if r == "exposure" and bearing(c))
    if n < 2:
        return (f"Only {n} exposure carries energy; a substitution swaps kcal between two."
                if n == 1 else "No exposure carries energy; a substitution swaps kcal between two.")
    return None


def route(
    state: Any,
    stages: Mapping[str, Any],
    artifacts: Mapping[str, Any] | None = None,
    records: Sequence[Any] = (),
    *,
    deps: Mapping[str, Sequence[str]] | None = None,
    energy_bearing: Callable[[str], bool] | None = None,
) -> list[InterviewStep]:
    """The interview, in asking order.

    ``stages``: stage name -> StageStatus (or its dict). ``artifacts``: the fresh public artifacts
    the Router reads — ``target_info`` (for the task skip). ``records``: the decision log, for
    each answered step's ``decision_id``.
    """
    if energy_bearing is None:
        from turbotab.core.stages.rows import energy_bearing as bearing
    else:
        bearing = energy_bearing
    artifacts = artifacts or {}
    pending = pending_stages(stages, deps)
    writers = _live_writer(records, state)
    target_info = artifacts.get("target_info")

    steps: list[InterviewStep] = []
    first_unanswered: str | None = None
    for key in QUESTION_KEYS:
        value = getattr(state, key, None)
        decision_id = writers.get(key)
        not_applicable = None
        if key == "energy_adjustment":
            not_applicable = _energy_applicability(state, bearing)
        elif key == "substitution":
            not_applicable = _substitution_applicability(state, bearing)
        if not_applicable is not None:
            steps.append(InterviewStep(key=key, status="not_applicable", reason=not_applicable,
                                       decision_id=decision_id if value is not None else None))
            continue
        if value is not None:
            steps.append(InterviewStep(key=key, status="answered", decision_id=decision_id))
            continue
        if (key == "task" and state.target is not None and target_info is not None
                and _get(target_info, "column") == state.target
                and _get(target_info, "confidence") == "high"):
            steps.append(InterviewStep(key=key, status="skipped", reason=_get(target_info, "reason")))
            continue
        own = [s for s in NEEDS.get(key, ()) if s in pending]
        fresh_stage = MUST_BE_FRESH.get(key)
        if fresh_stage and _get(stages.get(fresh_stage), "status") != "fresh" and fresh_stage not in own:
            own.append(fresh_stage)
        if first_unanswered is None:
            first_unanswered = key
            status = "waiting" if own else "open"
            steps.append(InterviewStep(key=key, status=status, waiting_on=own))
        else:
            steps.append(InterviewStep(key=key, status="waiting", waiting_on=[first_unanswered, *own]))
    return steps


__all__ = ["InterviewStep", "NEEDS", "QUESTION_KEYS", "QuestionKey", "pending_stages", "route"]
