"""
turbotab.api — FastAPI over the project.

Four endpoints, as specified in `docs/turbotab/LOOP.md` §"Loop 3":

    POST /project                    upload a table, get a diagnosis
    GET  /project/{id}               what is currently true
    POST /project/{id}/decision      record one answer
    GET  /project/{id}/findings      the ranked findings

plus the frontend, served from `turbotab/web/`.

This layer orchestrates and stores. It computes nothing: every number it returns
came from `turbotab.engine`, which in turn is a pass-through to `ml/`. There is
no job queue here and no training — anything over a second is out of scope for
the skeleton, and pretending otherwise is exactly the lie the job queue exists
to stop telling.

Run:  turbotab/.venv/Scripts/python -m uvicorn turbotab.api:app --reload
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from turbotab import engine
from turbotab.project import AnalysisProject, ProjectError, ProjectStore

WEB_DIR = Path(__file__).resolve().parent / "web"

app = FastAPI(
    title="TurboTab — walking skeleton",
    description="Upload a CSV, get a real structural diagnosis, a real profile, "
                "real ranked findings, and a record of what you decided.",
    version="0.1.0",
)

STORE = ProjectStore()

# Upload ceiling. The frame is held in memory and nothing is written to disk
# (ARCHITECTURE.md §02), so an unbounded upload is an unbounded resident set.
MAX_UPLOAD_BYTES = 64 * 1024 * 1024


class DecisionIn(BaseModel):
    """One answer from the interview.

    `kind` is the shared vocabulary between the record and the frontend:
    ``set_target`` · ``defer`` · ``dismiss`` · ``undismiss`` · ``flag``
    · ``unflag`` · ``note``.
    """
    kind: str
    subject: str = ""
    text: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _recompute(project: AnalysisProject) -> None:
    """Re-run diagnosis and profile against the project's current answers.

    Called on upload and whenever the target changes. Both engine calls are pure
    and read-only, so this is safe to repeat.
    """
    structural = engine.diagnose(project.df)
    prof = None
    try:
        prof = engine.profile(project.df, project.target, project.task_type)
    except ValueError:
        # `compute_dataset_profile` raises on a frame it cannot profile. The
        # structural findings are still real and still worth showing — reporting
        # nothing here would present an unprofiled file as a clean one.
        prof = None
    project.set_findings(
        engine.rank_findings(structural, prof),
        engine.profile_to_dict(prof) if prof is not None else None,
    )


def _payload(project: AnalysisProject) -> Dict[str, Any]:
    body = project.to_dict()
    body["sample"] = project.head(8)
    return body


def _sentence(project: AnalysisProject, d: DecisionIn) -> str:
    """The transcript sentence for a decision that did not bring its own.

    Built from the finding's own title so the record quotes the engine rather
    than paraphrasing it.
    """
    if d.text:
        return d.text
    title = d.subject
    try:
        title = project.finding(d.subject)["title"]
    except ProjectError:
        pass
    return {
        "defer":     f"{title} — deferred to the step where it belongs.",
        "dismiss":   f"{title} — dismissed; kept in the record.",
        "undismiss": f"{title} — brought back.",
        "flag":      f"{title} — marked for the manuscript.",
        "unflag":    f"{title} — unmarked.",
    }.get(d.kind, f"{title} — {d.kind}.")


# ─────────────────────────────────────────────────────────────────────────────
# POST /project — upload
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/project")
async def create_project(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Upload a table. Diagnosis runs immediately.

    "The profiler runs the moment data lands, and the section opens already
    answered" — `PRODUCT_VISION.md` §04. The user is never asked what they would
    like to look at.
    """
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"{file.filename} is larger than "
                                 f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    if not raw:
        raise HTTPException(400, f"{file.filename} is empty.")
    try:
        df = engine.read_table(raw, file.filename or "upload.csv")
        project = AnalysisProject.from_dataframe(df, file.filename or "upload.csv")
    except engine.EngineRefusal as exc:
        raise HTTPException(400, str(exc)) from exc
    except ProjectError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        # A parse failure is the user's file being unreadable, not a server
        # fault; say which file and why rather than returning a bare 500.
        raise HTTPException(400, f"Could not read {file.filename}: {exc}") from exc

    STORE.add(project)
    project.record("note", f"{project.name} was loaded: {len(project.df)} rows, "
                           f"{len(project.df.columns)} columns.", subject=project.name)
    _recompute(project)
    return _payload(project)


# ─────────────────────────────────────────────────────────────────────────────
# GET /project/{id}
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/project/{project_id}")
async def get_project(project_id: str) -> Dict[str, Any]:
    try:
        return _payload(STORE.get(project_id))
    except ProjectError as exc:
        raise HTTPException(404, str(exc)) from exc


# ─────────────────────────────────────────────────────────────────────────────
# POST /project/{id}/decision
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/project/{project_id}/decision")
async def add_decision(project_id: str, decision: DecisionIn) -> Dict[str, Any]:
    """Record one answer.

    Decisions append. Choosing a target a second time does not edit the first
    choice, it adds a second one and marks the findings computed under the old
    target stale — visible and recoverable, never silently dropped
    (`PRODUCT_VISION.md` §07.4).
    """
    try:
        project = STORE.get(project_id)
    except ProjectError as exc:
        raise HTTPException(404, str(exc)) from exc

    if decision.kind == "set_target":
        column = decision.payload.get("column") or decision.subject
        if not column:
            raise HTTPException(400, "set_target needs a column.")
        try:
            task = engine.detect_task_type(project.df, column)
            project.set_target(column, task["detected"], task["confidence"],
                               task.get("reasons", []))
        except engine.EngineRefusal as exc:
            raise HTTPException(400, str(exc)) from exc
        except ProjectError as exc:
            raise HTTPException(400, str(exc)) from exc
        _recompute(project)
        return _payload(project)

    if decision.kind not in {"defer", "dismiss", "undismiss", "flag", "unflag", "note"}:
        raise HTTPException(400, f"Unknown decision kind '{decision.kind}'.")

    project.record(decision.kind, _sentence(project, decision),
                   subject=decision.subject, payload=decision.payload)
    return _payload(project)


# ─────────────────────────────────────────────────────────────────────────────
# GET /project/{id}/findings
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/project/{project_id}/findings")
async def get_findings(project_id: str) -> Dict[str, Any]:
    """The ranked findings, plus enough of the profile to caption them."""
    try:
        project = STORE.get(project_id)
    except ProjectError as exc:
        raise HTTPException(404, str(exc)) from exc

    prof = project.profile or {}
    return {
        "project_id": project.id,
        "target": project.target,
        "task_type": project.task_type,
        "task_confidence": project.task_confidence,
        "stale": project.findings_stale,
        "count": len(project.findings),
        "findings": project.findings,
        "profile_summary": {
            "n_rows": prof.get("n_rows"),
            "n_features": prof.get("n_features"),
            "n_numeric": prof.get("n_numeric"),
            "n_categorical": prof.get("n_categorical"),
            "total_missing_rate": prof.get("total_missing_rate"),
            "data_sufficiency": prof.get("data_sufficiency"),
            "sufficiency_narrative": prof.get("sufficiency_narrative"),
            "target_profile": prof.get("target_profile"),
        } if prof else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# The frontend
#
# Mounted last: a mount at "/" swallows anything registered after it.
# ─────────────────────────────────────────────────────────────────────────────

if WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
else:  # pragma: no cover - only before step 5 lands
    @app.get("/")
    async def _no_frontend() -> Dict[str, str]:
        return {"detail": "turbotab/web/ is not built yet; the API is up."}
