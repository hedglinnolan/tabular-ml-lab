"""M1 domain proposals: the pack-sourced exclusion rules and energy-adjustment
reading the exclusions and energy questions offer (never pre-selected).

Owner: the M1 "voice" agent. Contract: docs/turbotab-next/M1_CONTRACT.md.
"""
from __future__ import annotations

from typing import Any

from turbotab.core.graph import StageContext


def proposals_stage(ctx: StageContext) -> Any:
    raise NotImplementedError("M1: the proposals stage is not built yet")
