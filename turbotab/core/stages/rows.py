"""M1 row-side stages: column roles, the cohort (participant flow), the split.

Owner: the M1 "rows" agent. Contract: docs/turbotab-next/M1_CONTRACT.md.
"""
from __future__ import annotations

from typing import Any

from turbotab.core.graph import StageContext


def roles_stage(ctx: StageContext) -> Any:
    raise NotImplementedError("M1: the roles stage is not built yet")


def cohort_stage(ctx: StageContext) -> Any:
    raise NotImplementedError("M1: the cohort stage is not built yet")


def split_stage(ctx: StageContext) -> Any:
    raise NotImplementedError("M1: the split stage is not built yet")
