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

from turbotab import draft, engine
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

    if decision.kind == "set_task_type":
        want = decision.payload.get("task_type") or decision.subject
        try:
            project.override_task_type(str(want))
        except ProjectError as exc:
            raise HTTPException(400, str(exc)) from exc
        _recompute(project)
        return _payload(project)

    if decision.kind == "apply":
        # The only endpoint in this service that changes the working table, and
        # it is reached only by asking for it by name. A preview never lands
        # here: it computes on a copy and throws the copy away.
        try:
            live = engine.find_shape_finding(engine.diagnose(project.df), decision.subject)
            # The identity barrier (T0-ID-001). Refused here rather than
            # detected afterwards: once the lockbox is sealed there is no way to
            # recover which rows its labels meant.
            project.check_repair_allowed(live.fix_kind)
            prev = engine.preview_fix(project.df, live)
            if not prev.get("applicable"):
                raise HTTPException(
                    400, "That finding has no automatic repair — it needs a human decision.")
            new_df, description = engine.apply_fix(project.df, live)
            project.apply_fix(new_df, live.id, live.title, description,
                              prev["row_identity_preserved"])
        except engine.EngineRefusal as exc:
            raise HTTPException(400, str(exc)) from exc
        except ProjectError as exc:
            raise HTTPException(400, str(exc)) from exc
        _recompute(project)
        return _payload(project)

    if decision.kind == "revert":
        try:
            project.revert_last_fix()
        except ProjectError as exc:
            raise HTTPException(400, str(exc)) from exc
        _recompute(project)
        return _payload(project)

    if decision.kind == "resolve_blocker":
        # The blocker's other terminal exit: the CHOICE it spawns, carried out.
        # Dropping a column is `import_doctor`'s own `drop_columns` repair, so
        # this composes the engine rather than adding a repair of its own — and
        # it goes through the same identity barrier as every other apply.
        column = decision.payload.get("column") or ""
        if not column:
            raise HTTPException(400, "Resolving a blocker needs the column it is about.")
        if column not in list(project.df.columns):
            raise HTTPException(400, f"No column named '{column}' in this table.")
        from ml.import_doctor import ShapeFinding
        drop = ShapeFinding(
            id=f"blocker_drop__{column}",
            severity="critical",
            title=f"Drop '{column}'",
            detail=f"'{column}' was dropped to resolve a question of consequence.",
            why_it_matters="",
            fix_label=f"Drop '{column}'",
            fix_kind="drop_columns",
            confidence="high",
            params={"columns": [column]},
            affected_columns=[column],
        )
        try:
            project.check_repair_allowed(drop.fix_kind)
            prev = engine.preview_fix(project.df, drop)
            new_df, description = engine.apply_fix(project.df, drop)
            project.apply_fix(new_df, drop.id, drop.title, description,
                              prev["row_identity_preserved"])
        except engine.EngineRefusal as exc:
            raise HTTPException(400, str(exc)) from exc
        except ProjectError as exc:
            raise HTTPException(400, str(exc)) from exc
        project.record("resolve_blocker",
                       f"`{column}` was dropped from the analysis because it may "
                       f"encode the outcome.",
                       subject=decision.subject,
                       payload={**decision.payload, "column": column,
                                "resolved": True})
        _recompute(project)
        return _payload(project)

    if decision.kind == "acknowledge_blocker":
        # The constitution's third clause. The tool does not refuse the user's
        # judgment — the flagged column may be legitimate — it refuses silence.
        # This is what the manuscript carries as a stated limitation.
        text = decision.text or decision.payload.get("acknowledgment")
        if not text:
            raise HTTPException(
                400, "An acknowledgment must say what is being accepted; that "
                     "sentence is what the manuscript carries as a limitation.")
        project.record("acknowledge_blocker", str(text), subject=decision.subject,
                       payload={**decision.payload, "unresolved": True})
        return _payload(project)

    if decision.kind not in {"defer", "dismiss", "undismiss", "flag", "unflag", "note"}:
        raise HTTPException(400, f"Unknown decision kind '{decision.kind}'.")

    payload = dict(decision.payload)
    if decision.kind == "defer" and "target_step" not in payload:
        # A deferral without a target is a discard with manners. The Router
        # refuses one, so the record must carry it.
        payload["target_step"] = "explore"
    project.record(decision.kind, _sentence(project, decision),
                   subject=decision.subject, payload=payload)
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
# GET /project/{id}/finding/{fid}/preview
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/project/{project_id}/finding/{finding_id}/preview")
async def preview_finding(project_id: str, finding_id: str) -> Dict[str, Any]:
    """What this fix would change, without changing it.

    A `GET`, because it is a question. It runs `import_doctor.apply_fix` on a
    deep copy, describes the difference, and discards the result — the project's
    frame is not touched, which `test_declining_a_preview_leaves_the_project_
    byte_identical` asserts against a content hash rather than trusting this
    paragraph.
    """
    try:
        project = STORE.get(project_id)
    except ProjectError as exc:
        raise HTTPException(404, str(exc)) from exc
    try:
        live = engine.find_shape_finding(engine.diagnose(project.df), finding_id)
        return engine.preview_fix(project.df, live)
    except engine.EngineRefusal as exc:
        raise HTTPException(404, str(exc)) from exc


# ─────────────────────────────────────────────────────────────────────────────
# Evidence
#
# "Every finding card names its objects in mono and embeds its evidence (rows or
# plot). 'The engine refuses to guess' is fine; refusing to show is not."
# (GUIDED-003). These endpoints are the show.
# ─────────────────────────────────────────────────────────────────────────────

# Which pull affordances actually run. The frontend renders a chip as live only
# when it appears here as `built`, and as a visibly disabled affordance
# otherwise — a solid-bordered control that no-ops asserts a capability that
# does not exist (GUIDED-006). Adding an analysis means adding it here, so the
# interface cannot drift ahead of the engine.
PULL_CAPABILITIES: Dict[str, Dict[str, Any]] = {
    "look::r1_plausibility": {
        "built": True, "endpoint": "plausibility",
        "label": "Physiologic plausibility",
    },
    "look::r2_missingness": {
        "built": True, "endpoint": "missingness",
        "label": "Missingness by feature",
    },
    "look::r8_collinearity": {
        "built": True, "endpoint": "correlations",
        "label": "Correlation matrix",
    },
    "histogram_pager": {
        "built": True, "endpoint": "histograms",
        "label": "Distribution of each feature",
    },
}

# Named, not counted: the reason a chip is dark is shown on the chip. Anything
# the Router offers that is not in PULL_CAPABILITIES falls back to this.
NOT_BUILT_REASON = ("Not in this build. The engine has the analysis; the Guided "
                    "door has not been wired to it yet, and a control that "
                    "silently does nothing is worse than one that says so.")


def _project(project_id: str) -> AnalysisProject:
    try:
        return STORE.get(project_id)
    except ProjectError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/project/{project_id}/evidence/plausibility")
async def evidence_plausibility(project_id: str) -> Dict[str, Any]:
    """Impossible and improbable entries, in two tiers, with the rows named.

    The impossible tier carries a repair proposal; the improbable tier stays
    advisory. A diastolic pressure of ~0 is an entry error, not a rare patient
    (GUIDED-004).
    """
    return engine.plausibility(_project(project_id).working_table)


@app.get("/project/{project_id}/evidence/histograms")
async def evidence_histograms(project_id: str, page: int = 0,
                              per_page: int = 6) -> Dict[str, Any]:
    return engine.histograms(_project(project_id).working_table,
                             page=page, per_page=per_page)


@app.get("/project/{project_id}/evidence/histogram/{column}")
async def evidence_one_histogram(project_id: str, column: str) -> Dict[str, Any]:
    try:
        h = engine.column_histogram(_project(project_id).working_table, column)
    except engine.EngineRefusal as exc:
        raise HTTPException(404, str(exc)) from exc
    if h is None:
        raise HTTPException(
            404, f"'{column}' has no distribution to draw — it is constant or "
                 f"has fewer than two usable values.")
    return {"column": column, **h}


@app.get("/project/{project_id}/evidence/correlations")
async def evidence_correlations(project_id: str) -> Dict[str, Any]:
    return engine.correlations(_project(project_id).working_table)


@app.get("/project/{project_id}/evidence/missingness")
async def evidence_missingness(project_id: str) -> Dict[str, Any]:
    """Dtype-routed missingness decisions, each naming its own column.

    The action-timing ruling is carried on every option: structural repairs run
    now, statistical transforms are recorded and fitted inside the per-model
    pipeline on training folds. Stated as methods prose in the decision
    sentence, never as a note about the software (GUIDED-002).
    """
    return {"cards": engine.missingness(_project(project_id).working_table)}


@app.get("/project/{project_id}/evidence/imputation/{column}")
async def evidence_imputation(project_id: str, column: str,
                              strategy: str = "impute_median") -> Dict[str, Any]:
    try:
        preview = engine.imputation_preview(
            _project(project_id).working_table, column, strategy)
    except engine.EngineRefusal as exc:
        raise HTTPException(404, str(exc)) from exc
    if preview is None:
        raise HTTPException(
            400, f"'{strategy}' has no before/after to show for '{column}'.")
    return preview


@app.get("/project/{project_id}/draft")
async def get_draft(project_id: str) -> Dict[str, Any]:
    """The decisions so far, as draft methods prose with the gaps visible."""
    return draft.draft(_project(project_id).to_dict())


@app.get("/capabilities")
async def get_capabilities() -> Dict[str, Any]:
    """Which pull affordances are wired, and what the unwired ones say.

    Served rather than hard-coded in the page so the interface cannot claim a
    capability the server does not have.
    """
    return {"pulls": PULL_CAPABILITIES, "not_built_reason": NOT_BUILT_REASON}


# ─────────────────────────────────────────────────────────────────────────────
# GET /project/{id}/interview — the Router's plan
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/project/{project_id}/interview")
async def get_interview(project_id: str, step: str = "data") -> Dict[str, Any]:
    """What the interview asks next, and what it offers alongside.

    The Router is a pure function of the record, so this endpoint holds no state
    of its own: `answered` and `deferred` are folded out of the project's
    decisions, which is the same fold the frontend does for finding dispositions.
    Two readers of one record cannot drift.
    """
    try:
        project = STORE.get(project_id)
    except ProjectError as exc:
        raise HTTPException(404, str(exc)) from exc

    from ml import router

    answered, deferred = [], {}
    for d in project.decisions:
        if d.kind == "set_target":
            answered.append("choose_target")
        elif d.kind == "set_task_type":
            answered.append("confirm_task_type")
        elif d.kind in ("apply", "dismiss"):
            answered.append(f"repair::{d.subject}")
        elif d.kind == "acknowledge_blocker":
            # A terminal state is guaranteed (DESIGN_LANGUAGE §09). A blocker
            # never re-fires on the same facts after acknowledgment: a flag that
            # cannot be satisfied teaches contempt for all flags. The
            # acknowledgment stays in the record and surfaces afterwards as its
            # own --stop-flagged artifact — never green, never gone.
            answered.append(d.subject)
        elif d.kind == "resolve_blocker":
            answered.append(d.subject)
        elif d.kind == "defer":
            deferred[f"repair::{d.subject}"] = d.payload.get("target_step", "explore")
        elif d.kind == "undismiss":
            key = f"repair::{d.subject}"
            if key in answered:
                answered.remove(key)

    detection = None
    if project.target and project.task_type:
        detection = {"detected": project.task_type,
                     "confidence": project.task_confidence,
                     "reasons": list(project.task_reasons)}

    structural = [f for f in project.findings if f.get("source") == "structure"]

    # The pull palette, from the recommender that was already engine code — and
    # the signals that carry blocker severity, which are questions rather than
    # offers (the constitution's third clause).
    recommendations, signals = [], None
    if step == "explore" and project.target:
        try:
            from ml.eda_recommender import compute_dataset_signals, recommend_eda
            signals = compute_dataset_signals(
                project.working_table, project.target, project.task_type,
                "cross_sectional", None)
            recommendations = recommend_eda(signals)
        except Exception:
            # The palette is an offer, not a promise. Losing it must not take
            # the interview's questions with it. A blocker is different: if the
            # signals could not be computed there is no blocker to hide, and the
            # next branch reports none rather than claiming none exist.
            recommendations, signals = [], None

    try:
        questions = router.plan(structural, target=project.target,
                                detection=detection, step=step,
                                deferred=deferred, answered=answered,
                                recommendations=recommendations, signals=signals)
        router.audit(questions)
    except router.RouterError as exc:
        # A plan that breaks a governing rule is not rendered at all.
        raise HTTPException(500, f"The interview broke a governing rule: {exc}") from exc

    open_blockers = router.unresolved_blockers(questions, answered)

    # Every pull affordance says whether it runs. A chip that is offered in live
    # styling and silently no-ops asserts a capability that does not exist —
    # the exact "assert something false" the governing rule forbids — so the
    # server, which knows, tells the page rather than the page guessing
    # (GUIDED-006).
    rendered = []
    for q in questions:
        d = q.to_dict()
        if d["mode"] == "pull":
            cap = PULL_CAPABILITIES.get(d["key"])
            d["built"] = bool(cap and cap.get("built"))
            d["endpoint"] = (cap or {}).get("endpoint")
            d["not_built_reason"] = None if d["built"] else NOT_BUILT_REASON
        rendered.append(d)

    return {
        "project_id": project.id,
        "step": step,
        "steps": list(router.STEPS),
        "questions": rendered,
        # Leaving the step with one of these open is allowed, and is itself a
        # decision. The frontend shows the sentence the record will carry.
        "unresolved_blockers": [q.key for q in open_blockers],
        "acknowledgment_required": router.acknowledgment_required(questions, answered),
        "n_asked": sum(1 for q in questions if q.mode == "push" and q.status == "asked"),
        "n_offered": sum(1 for q in questions if q.mode == "pull"),
        "next": next((q.to_dict() for q in questions
                      if q.mode == "push" and q.status == "asked"), None),
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
