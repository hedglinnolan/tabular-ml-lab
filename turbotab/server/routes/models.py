"""``/models``: the model-family registry (M1_CONTRACT §7)."""
from __future__ import annotations

from fastapi import APIRouter

from turbotab.core.models import FamilyInfo, families, info

router = APIRouter(tags=["models"])


@router.get("/models", response_model=list[FamilyInfo])
def list_models() -> list[FamilyInfo]:
    """Every registered model family: what it assumes, what it is good at, what to watch for."""
    return [info(family) for family in families()]
