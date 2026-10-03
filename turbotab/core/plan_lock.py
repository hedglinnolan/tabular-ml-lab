"""The inference analysis-plan lock (audit WP16, RO-12; MODELING_SEQUENCE §1 row 12).

Under inference the coefficients and p-values were live while the plan was still being changed:
every change of roles, exclusions, missing values or energy model refitted and re-served them, and
nothing recorded that the analyst had seen them (RO-12). Gelman & Loken (2013, "The garden of
forking paths", abstract): "Researcher degrees of freedom can lead to a multiple comparisons
problem, even in settings where researchers perform only a single analysis on their data. The
problem is there can be a large number of potential comparisons when the details of data analysis
are highly contingent on data". The forks they name (§1.2): "choices of control variables in a
regression, transformations, and data coding and excluding rules".

So the plan is locked the first time inference estimates are displayed:

* **When.** The server records ``lock_plan`` the first time a client is served an estimate under
  inference: a coefficient or an inference table in the fit, a substitution curve, a sensitivity
  analysis's estimate or a calibrated one (:func:`shows_estimates`). A user may also record it
  earlier. It is recorded once and never undone.
* **What.** The plan is every slot the estimates read, as it stood (:func:`plan_of`): the outcome,
  the exposures and the adjustment set (the roles), the exclusions, the missing-data plan, the
  energy model, the exposure forms, the families, the secondaries, and the data coding (repairs,
  confirmed readings). It is derived from the stage graph, so a slot a later stage adds is covered
  without a list to keep. Its SHA-256 (:func:`digest`) is recorded with it, for external
  registration.
* **After.** Every later decision is marked ``after_estimates`` and its sentence says "After the
  estimates were seen, …" (``decisions.disclose``); the methods text keeps the plan as declared and
  every change after it (``turbotab.core.provenance``).
* **Wording.** The lock says what was declared in the software before estimates were displayed.
  It is never called prespecified or preregistered (MODELING_SEQUENCE §1 row 12): the analyst may
  have seen the data, and nothing was registered outside the software.

Importing this module registers the validators and the completion.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from typing import Any, Mapping

from turbotab.core import decisions
from turbotab.core.decisions import Refusal

# The stages whose artifacts carry estimates of the outcome model.
ESTIMATE_STAGES = ("fit", "substitution", "sensitivity", "calibration")


@lru_cache(maxsize=1)
def plan_slots() -> tuple[str, ...]:
    """Every slot the estimates read: those of the estimate stages and of everything upstream."""
    from turbotab.core.graph import load_graph
    from turbotab.core.stages import GRAPH_FACTORY

    graph = load_graph(GRAPH_FACTORY)
    names: set[str] = set()
    for stage in ESTIMATE_STAGES:
        if stage in graph:
            names |= graph.upstream(stage) | {stage}
    return tuple(sorted({slot for name in names for slot in graph[name].reads}))


def plan_of(state: Any) -> dict[str, Any]:
    """The plan as ``state`` holds it: each answered slot the estimates read, as JSON."""
    values = state.model_dump(mode="json")
    return {slot: values[slot] for slot in plan_slots() if values.get(slot) is not None}


def digest(plan: Mapping[str, Any]) -> str:
    """The plan's SHA-256 over its canonical JSON (keys sorted, no spaces, UTF-8)."""
    text = json.dumps(plan, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def shows_estimates(stage: str, artifact: Any) -> bool:
    """Whether a stage's artifact, as served, puts an estimate of the outcome model in view."""
    if not isinstance(artifact, Mapping):
        return False
    if stage == "fit":
        return any(m.get("coefficients") or m.get("inference")
                   for m in artifact.get("models") or [] if isinstance(m, Mapping))
    if stage == "substitution":
        return any(any(v is not None for v in m.get("delta") or [])
                   for m in artifact.get("models") or [] if isinstance(m, Mapping))
    if stage == "sensitivity":
        return any(f.get("coefficients") or f.get("inference")
                   for family in artifact.get("families") or [] if isinstance(family, Mapping)
                   for f in family.get("fits") or [] if isinstance(f, Mapping))
    if stage == "calibration":
        return any(e.get("estimate") is not None or e.get("naive") is not None
                   for e in artifact.get("exposures") or [] if isinstance(e, Mapping))
    return False


def _state(ctx: Any) -> Any:
    if ctx is None:
        return None
    return ctx.get("state") if isinstance(ctx, Mapping) else getattr(ctx, "state", None)


def _records(ctx: Any) -> list[Any] | None:
    found = ctx.get("records") if isinstance(ctx, Mapping) else getattr(ctx, "records", None)
    if callable(found):
        try:
            found = found()
        except Exception:  # noqa: BLE001 - no records: nothing is checked against them
            return None
    return list(found) if found is not None else None


def _locked_once_under_inference(decision: Any, ctx: Any) -> None:
    state = _state(ctx)
    if state is None:
        return
    if state.plan_locked:
        raise Refusal(
            "plan_already_locked",
            "The analysis plan was locked when the estimates were first displayed; every change "
            "since is marked as made after the estimates were seen.",
        )
    if state.purpose != "inference":
        raise Refusal(
            "not_inference",
            "The analysis-plan lock is for inference, where the model is not chosen from the data "
            "it reports on; under prediction the held-out rows are the seal.",
            exits=[{"label": "Declare the analysis for inference first", "decision": None}],
        )


def _the_lock_records_the_plan(decision: Any, ctx: Any) -> Any:
    """Fill the plan and its digest from the state, never from the client."""
    state = _state(ctx)
    if state is None:
        return decision.model_copy(update={"plan": None, "digest": None})
    plan = plan_of(state)
    return decision.model_copy(update={"plan": plan, "digest": digest(plan)})


def _the_lock_stays(decision: Any, ctx: Any) -> None:
    records = _records(ctx)
    target = next((r for r in records or [] if r.id == decision.decision_id), None)
    if target is not None and target.decision.kind == "lock_plan":
        raise Refusal(
            "plan_stays_locked",
            "The estimates were displayed with this plan, and that cannot be undone: the lock "
            "stays in the record, and later changes stay marked as made after the estimates were "
            "seen.",
        )


decisions.register_validator("lock_plan", _locked_once_under_inference)
decisions.register_completion("lock_plan", _the_lock_records_the_plan)
decisions.register_validator("revert", _the_lock_stays, first=True)

__all__ = ["ESTIMATE_STAGES", "digest", "plan_of", "plan_slots", "shows_estimates"]
