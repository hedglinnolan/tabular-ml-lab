"""Proposals built by hand for the claim cycle's tests: one constructed to pass every guard of the
validator, so a test changes one thing and names what must fire. Every marker name is the
vocabulary's, every decision kind, metric and refusal code the engine's; no number here is an
engine's output.

:data:`GOOD` is a proposal as the theorist's tools take one: its refits not counted (``null``),
since the tools read no sweeps table. :func:`counted` gives it refits counted as a sweeps table
would count them, and :func:`sweeps_of` the table that agrees with what it states, so a test of a
count's own guards reaches them past the check against the sweeps."""
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
    "refits": None,
    "evidence": {"datasets": ["dataset_1", "dataset_2"], "runs": ["run_01-a", "run_02-b"]},
    "basis": "The fill moved log loss in the records whose complete-case share was below 0.8.",
}
# Ten refits in the region, two refused for backward elimination short of rows (the method's own
# limit), both where the predictors outnumber half the rows.
COUNTED: dict[str, Any] = {
    "refits": 10, "refused": 2, "share": 0.2, "codes": {"stepwise_needs_rows": 2},
    "one_region": True,
    "region": [{"marker": "predictors_per_row", "op": "above", "threshold": 0.5}]}


def proposal(**over: Any) -> dict[str, Any]:
    """:data:`GOOD` with the top-level fields in ``over`` put in its place."""
    out = copy.deepcopy(GOOD)
    out.update(copy.deepcopy(over))
    return out


def counted(**over: Any) -> dict[str, Any]:
    """:data:`GOOD` with :data:`COUNTED`'s refits, the top-level fields in ``over`` put in place
    (``refits`` among them, whole)."""
    return proposal(**{"refits": copy.deepcopy(COUNTED), **over})


def sweeps_of(*proposals: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """A sweeps table that counts each proposal's refits as it states them."""
    out = {}
    for p in proposals:
        r = p.get("refits")
        if isinstance(r, dict) and r.get("refits") is not None:
            out[p["id"]] = {"refits": r["refits"], "refused": r.get("refused"),
                            "codes": dict(r.get("codes") or {})}
    return out


def keyed(pid: str, decision: str, markers: list[str], direction: str = "material",
          op: str = "below", threshold: float = 0.5) -> dict[str, Any]:
    """A proposal with this decision, region (each marker under ``op`` and ``threshold``) and
    direction: the three things the overlap matches on, the rest :data:`GOOD`'s."""
    region = [{"marker": m, "op": op, "threshold": threshold} for m in markers]
    scoped = ({"scope": {"datasets": list(GOOD["evidence"]["datasets"])}}
              if decision == "set_selection" else {})
    return proposal(id=pid, decision=decision, region=region, direction=direction, **scoped)


def selection(**over: Any) -> dict[str, Any]:
    """A claim about variable selection, scoped to the two datasets its evidence comes from."""
    return proposal(**{"decision": "set_selection",
                       "scope": {"datasets": ["dataset_1", "dataset_2"]}, **over})
