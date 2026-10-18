"""Projects, decisions and stage results."""
from __future__ import annotations

from fastapi import APIRouter, Query, Request, Response
from starlette.concurrency import run_in_threadpool

from turbotab.core.decisions import NOTE_MAX_CHARS
from turbotab.core.export.methods import MethodsSections
from turbotab.core.fit_press import FitLock
from turbotab.core.materiality import Ledger
from turbotab.core.mattered import Mattered
from turbotab.core.plan_lock import PlanExport
from turbotab.core.provenance import MethodsText
from turbotab.core.quest import QuestLog
from turbotab.core.sweep import ForTheRecord, Triage
from turbotab.core.workspace import PROVENANCE_MAX_CHARS, source_provenance
from turbotab.server.errors import ApiError
from turbotab.server.routes import get_service, get_settings, refusal, require_local
from turbotab.server.schemas import (
    AnalystLog,
    CreateProject,
    Decision,
    FitsLog,
    ProjectSummary,
    ProjectView,
    ReadingsCard,
    Results,
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
    responses={400: refusal("No file, not a readable data file, or a field of where it came from "
                            "that does not read")},
    openapi_extra=_UPLOAD_BODY,
    description=(
        "Upload a data file (multipart field ``file``). It is streamed to disk as it arrives. "
        "``url``, ``release_date`` and ``license`` (each optional, each at most "
        f"{PROVENANCE_MAX_CHARS:,} characters) say where the file came from: kept on the "
        "project's meta and echoed into the export's ``provenance.json`` inputs beside the file's "
        "SHA-256 (SWARM_ENGINE_PLAN E5). One that does not read refuses the upload before any of "
        "it is kept."),
)
async def upload_project(
    request: Request,
    url: str | None = Query(None, description="Where the file came from, in full (https://…)"),
    release_date: str | None = Query(None, description="When it was released, YYYY-MM-DD"),
    license: str | None = Query(None, description="The license it was released under"),
) -> dict:
    """Upload a data file, with where it came from (the route's description)."""
    try:
        provenance = source_provenance(url=url, release_date=release_date, license=license)
    except ValueError as exc:
        raise ApiError(400, "bad_provenance", str(exc)) from None
    service = get_service(request)
    staged = await receive_upload(request, service.workspace.uploads_dir())
    meta = await run_in_threadpool(
        service.create_from_upload, staged.path, staged.client_name, staged.fingerprint, provenance
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
def decide(
    request: Request, pid: str, decision: Decision, decide_now: bool = False,
    note: str | None = Query(None, description=(
        f"Why: the decision's reason, at most {NOTE_MAX_CHARS:,} characters, kept on its record "
        f"and served in the decision log, never written into the methods")),
) -> dict:
    """Record a decision. Stages downstream of the slot it writes recompute. ``decide_now``
    answers a later question ahead of the Router, only where its card is computed and the earlier
    answers it reads are in (crosswalk disagreement 20); otherwise it is refused with what it is
    waiting for, and the record names the question it was decided ahead of. ``note``: the reason,
    in any mode (SWARM_ENGINE_PLAN E4); one past its limit is refused (``note_too_long``) and
    nothing is recorded."""
    return get_service(request).decide(pid, decision, early=decide_now, note=note)


@router.get(
    "/projects/{pid}/analyst-log",
    response_model=AnalystLog,
    responses={403: refusal("TurboTab is not in analyst mode"), 404: refusal("No such project")},
)
def analyst_log(request: Request, pid: str) -> dict:
    """What the analyst saw, in analyst mode only (``--analyst``, or ``TURBOTAB_ANALYST=1``;
    SWARM_ENGINE_PLAN E4), oldest first: each claim rendered for the project (the claim asked for,
    its register, frozen status and grade, the claim that rendered in its place outside its
    regime, and why a render was refused), which is logged in every mode; and each preview asked
    for in analyst mode, a decision's (``…/preview``) or a join's (``…/join-preview``, as the
    ``join_files`` decision it previews), with the refusal's code when it was refused."""
    return get_service(request).analyst_log(pid)


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
    "/projects/{pid}/fits",
    response_model=FitsLog,
    responses={404: refusal("No such project")},
)
def fits(request: Request, pid: str) -> dict:
    """Every fit that completed, one record per fit key in the order the keys turned fresh, each
    written once and never rewritten (SWARM_ENGINE_PLAN E1): its purpose, outcome and key, the
    plan's digest and the records in force it was computed from, the press of Fit in force, its
    families, the seconds its job ran and the server's workers, the rows its split held out, and
    per family the cross-validated summary (Predict) or the exposure's rows with their intervals
    (Estimate). Each number is the fit's as ``GET /stages/fit`` would serve that fit now, read for
    the record's own scope: none under another goal than the goal now; under Predict none before
    Fit is pressed for the record's outcome; under Estimate none before the plan's lock, nor for a
    fit computed before the lock that is not the fit the lock was recorded on (the plan's digest
    is the plan when the fit completed, so the lock can record another for the same fit), unless
    it is the fit the stage route serves now; a held-out score only once the record's outcome's
    seal is opened, only where the fit was scored on the rows that opening opened, and only for
    the fit the opening was recorded on, a fit computed after it or the current fit, with what the
    stage route says beside them: whether the fit is not the opened one, the scores the opening
    recorded (the reported result) and whether those are the scores served, and the final
    model's note. ``withheld`` and ``holdout_why`` say why a number is not served. Under Predict,
    a record scored on the same
    folds as the one before it carries each family's paired movement against it; otherwise
    ``movement_why`` says why it has none. Under Estimate, ``locked`` marks the record of the fit
    the plan's lock in force was recorded on (its key is the one the answers at the lock give)."""
    return get_service(request).fits(pid)


@router.get(
    "/projects/{pid}/results",
    response_model=Results,
    responses={404: refusal("No such project")},
)
def results(request: Request, pid: str) -> dict:
    """What the Results screens show, joined into one document, each field the served artifact's
    own value and gated as the stage route gates it (SWARM_ENGINE_PLAN E2). Under Predict: each
    family's cross-validated summary, the declared result and, once the seal is opened, its
    held-out scores with their intervals and calibration, with what the screen says beside them:
    the scores the opening recorded (the reported result) and whether those are the scores served,
    the final model's note, whether the fit changed after the opening and the post-seal decisions
    that changed it, and the split's basis, which can mark the held-out scores exploratory. Under
    Estimate: the estimand, each family's rows of the exposure from the fit, and from the effects
    stage its sequence fits, marginal contrast and E-values. The cross-validated scores served
    count as seen for the outcome of the fit that holds them, as on the stage route, and also
    while that fit is not the current one."""
    return get_service(request).results(pid)


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
    "/projects/{pid}/mattered",
    response_model=Mattered,
    responses={409: refusal("Not offered for this goal or effect measure, the plan not locked "
                            "yet, a question the estimates rest on open, or the curve cannot be "
                            "drawn"),
               404: refusal("No such project")},
)
def mattered(request: Request, pid: str) -> Mattered:
    """Which decisions mattered, under Estimate (SWARM_ENGINE_PLAN E3): the exposure's estimate
    refit with each recorded choice changed, one at a time (each exclusion rule dropped, each
    adjusted covariate dropped, each covariate the adjustment answers left out added), a
    specification curve the ``mattered`` stage computes after the plan's lock. Its primary is the
    reported estimate, which stays the result whatever the curve shows. Refused under a goal or an
    effect measure that does not offer it (a marginal risk difference or ratio), and before the
    lock with the line saying when it opens; while the stage computes, its status and no curve; a
    curve that cannot be drawn is refused with why, quoting no estimate. Serving a curve marks its
    estimates as shown."""
    return get_service(request).mattered(pid)


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
    "/projects/{pid}/methods/sections",
    response_model=MethodsSections,
    responses={404: refusal("No such project")},
)
def methods_sections(request: Request, pid: str) -> MethodsSections:
    """The methods in the sections and words of the export bundle's methods (P8): TRIPOD+AI's
    sections under prediction, STROBE's with STROBE-nut's otherwise, each line with its decision
    record's id (null for a paragraph the analysis wrote and for the export's own sentence), and
    how many lines there are. Built as the bundle builds them, from the decision log and the fresh
    stages as they stand, without its refusal; reading it records nothing."""
    return get_service(request).methods_sections(pid)


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
