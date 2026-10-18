"""``/teaching``: what each interview question teaches (M1_CONTRACT §5). The same for every project
with the same goal."""
from __future__ import annotations

from fastapi import APIRouter, Response

from turbotab.core import teaching
from turbotab.core.decisions import Purpose
from turbotab.server.schemas import TeachingEntry

router = APIRouter(tags=["teaching"])


@router.get("/teaching", response_model=list[TeachingEntry])
def get_teaching(response: Response, purpose: Purpose | None = None) -> list[TeachingEntry]:
    """One entry per question, in asking order: the question, its one line, each option's
    consequence, the terms it uses, and the pack content behind it with evidence badges.
    ``purpose``: the project's declared goal; under ``prediction`` the roles card says
    "predictors" and names no study factor. Without it, the cards as under Estimate."""
    # W1-R round 2 (the walkthrough's finding 3c): the roles card by the goal.
    response.headers["Cache-Control"] = "max-age=300"
    return teaching.entries(purpose)
