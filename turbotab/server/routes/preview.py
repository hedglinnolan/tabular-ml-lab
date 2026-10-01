"""Consequence previews (what an option would do to the user's own data, before it is recorded) and
finding evidence (why a finding was raised), in one vocabulary of views."""
from __future__ import annotations

from fastapi import APIRouter, Request

from turbotab.core.consequences import PreviewResult
from turbotab.server.routes import get_service, refusal
from turbotab.server.schemas import Decision

router = APIRouter(tags=["previews"])


@router.post(
    "/projects/{pid}/preview",
    response_model=PreviewResult,
    responses={
        409: refusal("The decision would be refused, or the table is not read yet"),
        404: refusal("No such project"),
    },
)
def preview(request: Request, pid: str, decision: Decision) -> PreviewResult:
    """The views that show what recording ``decision`` would change. Nothing is recorded.

    At most three views, the first primary. Once the split exists, held-out rows are never read.
    """
    return get_service(request).preview(pid, decision)


@router.get(
    "/projects/{pid}/findings/{fid}/evidence",
    response_model=PreviewResult,
    responses={
        409: refusal("The findings or the table are not read yet"),
        404: refusal("No such project, or no such finding now"),
    },
)
def finding_evidence(request: Request, pid: str, fid: str) -> PreviewResult:
    """The views that show why finding ``fid`` was raised, in the preview's vocabulary.

    ``kind`` is the finding's family. Data-quality evidence reads every row, as the finding did;
    evidence for a modeling choice never reads the held-out rows.
    """
    return get_service(request).evidence(pid, fid)
