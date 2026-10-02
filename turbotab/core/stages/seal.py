"""The ``seal_plan`` stage: what the split question can offer on this table (M2_CONTRACT §3).

Before the seal is drawn the split question needs three things the data decides: the basis the
seal would carry (grouped by which column, or why not), whether the answers ask for a
chronological draw and whether it can be drawn, and what a held-out score on each offered size
could measure, ordered so cross-validation alone leads below the floor. The statistics are
``turbotab.core.seal``'s; this stage reads the rows they need and shapes the artifact.

It reads the identifier, time and outcome columns over every row with the outcome measured: the
same rows the seal is drawn over, which is what defining the held-out rows means. Nothing here
informs a model.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from turbotab.core.graph import Bundle, StageContext
from turbotab.core.stages.data import open_store

# The slots the seal's basis and its chronological draw read besides the roles and the task.
SEAL_READS: tuple[str, ...] = ("grain", "unit", "aggregation", "temporal", "repeat_kind")


def seal_plan_stage(ctx: StageContext) -> dict[str, Any]:
    from turbotab.core.seal import plan

    cohort = ctx.inputs["cohort"]
    frames = cohort.frames if isinstance(cohort, Bundle) else {}
    empty = np.zeros(0, dtype=np.int64)
    rows = frames["rows"]["row_id"].to_numpy(dtype=np.int64) if "rows" in frames else empty
    measured = frames.get("measured")
    universe = measured["row_id"].to_numpy(dtype=np.int64) if measured is not None else rows
    task = ctx.state.task or ctx.inputs["target_info"].get("task")
    structure = ctx.inputs.get("structure")  # the grain stated from a unique identifier
    ctx.progress(0.2, "Reading the identifier, the outcome and the time column")
    with open_store(ctx) as store:
        return plan(ctx.state, universe, store, task, analyzed=rows,
                    structure=getattr(structure, "data", structure))


__all__ = ["SEAL_READS", "seal_plan_stage"]
