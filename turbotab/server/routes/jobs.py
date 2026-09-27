"""Jobs: watch one, or stop it."""
from __future__ import annotations

from fastapi import APIRouter, Request

from turbotab.server.routes import get_service, refusal
from turbotab.server.schemas import JobView

router = APIRouter(tags=["jobs"])


@router.get("/projects/{pid}/jobs/{jid}", response_model=JobView, responses={404: refusal("No such job")})
def get_job(request: Request, pid: str, jid: str) -> JobView:
    return get_service(request).job(pid, jid)


@router.post(
    "/projects/{pid}/jobs/{jid}/cancel",
    response_model=JobView,
    responses={404: refusal("No such job")},
)
def cancel_job(request: Request, pid: str, jid: str) -> JobView:
    """Stop a job. It is not rerun by itself; the returned view may still say running for ~1 s."""
    return get_service(request).cancel_job(pid, jid)
