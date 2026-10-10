"""Projects, decisions and stage results."""
from __future__ import annotations

from fastapi import APIRouter, Request, Response
from starlette.concurrency import run_in_threadpool

from turbotab.core.fit_press import FitLock
from turbotab.core.materiality import Ledger
from turbotab.core.plan_lock import PlanExport
from turbotab.core.provenance import MethodsText
from turbotab.core.quest import QuestLog
from turbotab.core.sweep import ForTheRecord, Triage
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
def decide(request: Request, pid: str, decision: Decision, decide_now: bool = False) -> dict:
    """Record a decision. Stages downstream of the slot it writes recompute. ``decide_now``
    answers a later question ahead of the Router, only where its card is computed and the earlier
    answers it reads are in (crosswalk disagreement 20); otherwise it is refused with what it is
    waiting for, and the record names the question it was decided ahead of."""
    return get_service(request).decide(pid, decision, early=decide_now)


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
    "/projects/{pid}/quest",
    response_model=QuestLog,
    responses={404: refusal("No such project")},
)
def quest(request: Request, pid: str) -> QuestLog:
    """The quest log's seven stages, in order (SIZING P0.4): each stage's lines under Decide,
    Confirm and For the record, with "Waiting for" on a line whose earlier answers are missing;
    progress as answered over required, null for a stage not reached, with ``complete`` once every
    objective is answered (a reached stage that asks nothing, 0 of 0, is complete: its segment is
    full; except Results under Estimate and Describe, which counts the exhibits the engine serves
    after Fit, so it reads 0 of N until they are placed, and Write-up is not reached until Results
    is placed); and, when an answer decided in another stage asked its questions again or left its
    results out of date, or another answer in the stage itself asked its questions again
    (``within``), why, in plain words. ``version`` changes when the shape or the meaning of this
    log does."""
    return get_service(request).quest(pid)


@router.post(
    "/projects/{pid}/fit",
    response_model=FitLock,
    responses={409: refusal("Nothing to fit yet, or a question the estimates rest on is open"),
               404: refusal("No such project")},
)
def press_fit(request: Request, pid: str) -> FitLock:
    """Fit, pressed on the analysis flowchart (SIZING P0.8; RECIPES_AND_TUNING §4.4): a job
    command, not a decision. It releases a fit the scheduler holds (one expected to take over
    about 2 minutes) and opens Results. Under Estimate and Describe it locks the analysis plan, the
    system's ``lock_plan`` record, once; no estimate is served before it. Under Predict nothing
    locks, and no score or estimate is served before it. Refused while nothing is chosen to fit or
    a question the estimates rest on is open. Returns the lock as the quest log shows it."""
    return get_service(request).press_fit(pid)


@router.post(
    "/projects/{pid}/fit/cancel",
    response_model=FitLock,
    responses={404: refusal("No such project")},
)
def cancel_fit(request: Request, pid: str) -> FitLock:
    """Cancel, where Fit was on the analysis flowchart (calm/FOUNDATION §7): the estimates' work
    still running stops. Before any estimate is served the press is withdrawn, and under Estimate
    and Describe the plan's lock with it (kept in the record as withdrawn), so Fit is asked for
    again; once an estimate has been served the lock stands. Returns the lock as the quest log
    shows it."""
    return get_service(request).cancel_fit(pid)


@router.get(
    "/projects/{pid}/record",
    response_model=ForTheRecord,
    responses={404: refusal("No such project")},
)
def for_the_record(request: Request, pid: str) -> ForTheRecord:
    """Each stage's For the record lines (SIZING P0.5), collapsed in the quest log and never
    counted: what was read (the ingest's facts and warnings, the profile's basis), why a question
    was not asked, the defaults no other choice changes a number on here (with why), and the
    answers TurboTab recorded itself. A stage not reached yet has none."""
    return get_service(request).record(pid)


@router.get(
    "/projects/{pid}/materiality",
    response_model=Ledger,
    responses={404: refusal("No such project")},
)
def materiality_ledger(request: Request, pid: str) -> Ledger:
    """The materiality ledger (SURFACING_POLICY §2.2): for each noticing measured on the table,
    how far its alternative is predicted to move the numbers before the plan is fixed (outcome-
    blind), the triage's recommendation and the disposition recorded, what was done about it,
    and once the plan is fixed its realized movement and whether it matched. Derived from the
    record each time, so a replay gives the same rows."""
    return get_service(request).materiality(pid)


@router.get(
    "/projects/{pid}/triage",
    response_model=Triage,
    responses={404: refusal("No such project")},
)
def triage(request: Request, pid: str) -> Triage:
    """The triage of the open noticings at the gate (SIZING P0.5; calm/FOUNDATION §7): before the
    plan is fixed under Estimate and Describe (and while the goal is unanswered), before the
    held-out rows open under Predict. Each open noticing carries the recommended disposition
    ("doesn't change your numbers here", "could bias the estimate", "act on it") and its reason;
    blockers come first and must be resolved. ``POST …/decisions`` with ``{"kind":
    "confirm_sweep", "stage": <the gate's stage>, "sweep": "noticings"}`` records every
    disposition, with any the person changed in ``lines``."""
    return get_service(request).triage(pid)


@router.get(
    "/projects/{pid}/methods",
    response_model=MethodsText,
    responses={404: refusal("No such project")},
)
def methods(request: Request, pid: str) -> MethodsText:
    """The methods text built from the decision log: the sentences in force, with decisions
    superseded before anything was seen folded out, and every change made after the estimates
    were seen or the held-out rows were opened kept and marked (audit WP16)."""
    return get_service(request).methods(pid)


@router.get(
    "/projects/{pid}/plan",
    response_class=Response,
    responses={200: {"model": PlanExport, "content": {"application/json": {}}},
               404: refusal("No such project")},
)
def plan(request: Request, pid: str) -> Response:
    """The analysis plan for external registration (MODELING_SEQUENCE §1 row 12): the plan as
    declared, its timestamp and its SHA-256 over canonical JSON, as canonical JSON bytes that are
    the same for the same decision log."""
    return Response(content=get_service(request).plan(pid), media_type="application/json")


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
