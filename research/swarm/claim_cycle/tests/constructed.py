"""Proposals built by hand for the claim cycle's tests: one constructed to pass every guard of the
validator, so a test changes one thing and names what must fire. Every marker name is the
vocabulary's, every decision kind, metric and refusal code the engine's; no number here is an
engine's output."""
from __future__ import annotations

import copy
from typing import Any

GOOD: dict[str, Any] = {
    "id": "p1",
    "author": "theorist A",
    "decision": "set_missing",
    "region": [{"marker": "complete_case_share", "op": "below", "threshold": 0.8}],
    "direction": "material",
    "size": 0.02,
    "loss": {"purpose": "prediction", "metric": "log_loss"},
    "refits": {"refits": 10, "refused": 2, "share": 0.2, "codes": {"stepwise_needs_rows": 2},
               "one_region": True,
               "region": [{"marker": "predictors_per_row", "op": "above", "threshold": 0.5}]},
    "evidence": {"datasets": ["dataset_1", "dataset_2"], "runs": ["run_01-a", "run_02-b"]},
    "basis": "The fill moved log loss in the records whose complete-case share was below 0.8.",
}


def proposal(**over: Any) -> dict[str, Any]:
    """:data:`GOOD` with the top-level fields in ``over`` put in its place."""
    out = copy.deepcopy(GOOD)
    out.update(copy.deepcopy(over))
    return out


def keyed(pid: str, decision: str, markers: list[str], direction: str = "material",
          op: str = "below", threshold: float = 0.5) -> dict[str, Any]:
    """A proposal with this decision, region (each marker under ``op`` and ``threshold``) and
    direction: the three things the overlap matches on, the rest :data:`GOOD`'s."""
    region = [{"marker": m, "op": op, "threshold": threshold} for m in markers]
    scoped = ({"scope": {"datasets": list(GOOD["evidence"]["datasets"])}}
              if decision == "set_selection" else {})
    return proposal(id=pid, decision=decision, region=region, direction=direction,
                    refits={"refits": 0, "refused": 0, "share": None}, **scoped)


def selection(**over: Any) -> dict[str, Any]:
    """A claim about variable selection, scoped to the two datasets its evidence comes from."""
    return proposal(**{"decision": "set_selection",
                       "scope": {"datasets": ["dataset_1", "dataset_2"]}, **over})
