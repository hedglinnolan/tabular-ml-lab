"""``/teaching``: what each interview question teaches (M1_CONTRACT §5). The same for every project."""
from __future__ import annotations

from fastapi import APIRouter, Response

from turbotab.core import teaching
from turbotab.server.schemas import TeachingEntry

router = APIRouter(tags=["teaching"])


@router.get("/teaching", response_model=list[TeachingEntry])
def get_teaching(response: Response) -> list[TeachingEntry]:
    """One entry per question, in asking order: the question, its one line, each option's
    consequence, the terms it uses, and the pack content behind it with evidence badges."""
    response.headers["Cache-Control"] = "max-age=300"
    return teaching.entries()
