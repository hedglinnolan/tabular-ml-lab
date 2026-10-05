"""Projects, decisions and stage results."""
from __future__ import annotations

from fastapi import APIRouter, Request
from starlette.concurrency import run_in_threadpool

from turbotab.server.routes import get_service, get_settings, refusal, require_local
from turbotab.server.schemas import (
    CreateProject,
    Decision,
    ProjectSummary,
    ProjectView,
    ReadingsCard,
    StageResult,
    StageStatus,
)
from turbotab.server.uploads import FIELD, receive_upload

router = APIRouter(tags=["projects"])

_UPLOAD_BODY = {
    "requestBody": {
        "required": True,
        "content": {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "required": [FIELD],
                    "properties": {FIELD: {"type": "string", "format": "binary"}},
                }
            }
        },
    }
}


@router.get("/projects", response_model=list[ProjectSummary])
def list_projects(request: Request) -> list[dict]:
    """Every project in the workspace, newest first."""
    return get_service(request).list()


@router.post(
    "/projects",
    response_model=ProjectSummary,
    responses={
        400: refusal("Not a readable data file"),
        403: refusal("Server mode: open files by uploading them"),
        404: refusal("No file at that path"),
    },
)
def create_project(request: Request, body: CreateProject) -> dict:
    """Open a file on this machine by its path (local mode only). Ingest starts as a job."""
    require_local(
        get_settings(request),
        "Opening a file by its path",
        [{"label": "Upload the file instead", "decision": None}],
    )
    service = get_service(request)
    meta = service.create_from_path(body.path)
    return service.summary(meta)


@router.post(
    "/projects/upload",
    response_model=ProjectSummary,
    responses={400: refusal("No file, or not a readable data file")},
    openapi_extra=_UPLOAD_BODY,
)
async def upload_project(request: Request) -> dict:
    """Upload a data file (multipart field ``file``). It is streamed to disk as it arrives."""
    service = get_service(request)
    staged = await receive_upload(request, service.workspace.uploads_dir())
    meta = await run_in_threadpool(
        service.create_from_upload, staged.path, staged.client_name, staged.fingerprint
    )
    return await run_in_threadpool(service.summary, meta)


@router.get("/projects/{pid}", response_model=ProjectView, responses={404: refusal("No such project")})
def get_project(request: Request, pid: str) -> dict:
    return get_service(request).view(pid)


@router.post(
    "/projects/{pid}/decisions",
    response_model=ProjectView,
    responses={409: refusal("The decision was refused and not recorded"), 404: refusal("No such project")},
)
def decide(request: Request, pid: str, decision: Decision) -> dict:
    """Record a decision. Stages downstream of the slot it writes recompute."""
    return get_service(request).decide(pid, decision)


@router.get(
    "/projects/{pid}/readings",
    response_model=ReadingsCard,
    responses={404: refusal("No such project")},
)
def readings_card(request: Request, pid: str) -> dict:
    """The readings the values settled with no question asked, each with its evidence and the
    answers that change it ("read from your data"; BLUEPRINT §14.3). Nothing waits on them."""
    return get_service(request).readings(pid)


@router.get(
    "/projects/{pid}/stages/{stage}",
    response_model=StageResult,
    responses={404: refusal("No such project or stage")},
)
def stage_result(request: Request, pid: str, stage: str) -> dict:
    """A stage's artifact: the current one (``fresh``), else the newest older one, else null."""
    return get_service(request).stage_result(pid, stage)


@router.post(
    "/projects/{pid}/stages/{stage}/run",
    response_model=StageStatus,
    responses={404: refusal("No such project or stage")},
)
def run_stage(request: Request, pid: str, stage: str) -> StageStatus:
    """Compute a stage for the current answers again, after it failed or was cancelled.

    Stages it waits on that failed or were cancelled are retried too. A stage
    that is fresh, in flight or blocked is left as it is. Returns its status.
    """
    return get_service(request).run_stage(pid, stage)
