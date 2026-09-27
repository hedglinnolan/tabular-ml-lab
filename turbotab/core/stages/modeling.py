"""M1 model-side stages: the shelf, the design (pipelines + column lineage), the
fit, and substitution curves.

Owner: the M1 "modeling" agent. Contract: docs/turbotab-next/M1_CONTRACT.md.
"""
from __future__ import annotations

from typing import Any

from turbotab.core.graph import StageContext


def shelf_stage(ctx: StageContext) -> Any:
    raise NotImplementedError("M1: the shelf stage is not built yet")


def design_stage(ctx: StageContext) -> Any:
    raise NotImplementedError("M1: the design stage is not built yet")


def fit_stage(ctx: StageContext) -> Any:
    raise NotImplementedError("M1: the fit stage is not built yet")


def substitution_stage(ctx: StageContext) -> Any:
    raise NotImplementedError("M1: the substitution stage is not built yet")
