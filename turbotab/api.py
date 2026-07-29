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

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from turbotab import (
    devchecks, draft, engine, features as feat_mod, grain as grain_mod,
    selection as sel_mod,
)
from turbotab.project import (
    AnalysisProject, GrainContradiction, ProjectError, ProjectStore,
)

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


# ─────────────────────────────────────────────────────────────────────────────
# The dev harness — off unless TURBOTAB_DEV_CHECKS=1
#
# A RAW ASGI middleware rather than `BaseHTTPMiddleware`, and the reason is
# specific: to capture a request body under `BaseHTTPMiddleware` you have to
# `await request.body()`, which CONSUMES the receive channel — the upload
# endpoint downstream then reads an empty multipart stream and every drive
# begins with a broken upload. Instrumentation that breaks the thing it
# instruments is worse than none. This wraps `receive` and `send` and OBSERVES
# the bytes going past instead of standing in the way of them.
# ─────────────────────────────────────────────────────────────────────────────

_PROJECT_PATH = re.compile(r"/project/([0-9a-f]{6,})")


class DevCapture:
    """Record the wire and the state around every action, then run the checks."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):        # noqa: C901
        if scope.get("type") != "http" or not devchecks.enabled():
            return await self.app(scope, receive, send)
        path = scope.get("path", "")
        if not (path.startswith("/project") or path.startswith("/capabilities")):
            return await self.app(scope, receive, send)
        # Armed on the first request rather than at startup: `on_event` is
        # deprecated, a `lifespan=` argument would have to be threaded through
        # the app's construction for a dev flag, and this is idempotent and
        # costs one boolean check per request.
        devchecks.start_listening()

        chunks: List[bytes] = []

        async def observing_receive():
            message = await receive()
            if message.get("type") == "http.request":
                body = message.get("body") or b""
                if sum(len(c) for c in chunks) < 256 * 1024:
                    chunks.append(body)
            return message

        captured: Dict[str, Any] = {"status": None, "body": bytearray()}

        async def observing_send(message):
            if message.get("type") == "http.response.start":
                captured["status"] = message.get("status")
            elif message.get("type") == "http.response.body":
                if len(captured["body"]) < 512 * 1024:
                    captured["body"].extend(message.get("body") or b"")
            await send(message)

        project_id = None
        m = _PROJECT_PATH.search(path)
        if m:
            project_id = m.group(1)
        before = _dev_state(project_id)

        await self.app(scope, observing_receive, observing_send)
        # Everything from here is instrumentation, and the response has already
        # been sent. `safely` is what keeps a harness bug from ending the drive
        # it exists to record — the first draft raised inside the upload
        # endpoint and took the whole drive with it.
        devchecks.safely(self._record, scope, path, project_id, before,
                         chunks, captured)

    def _record(self, scope, path, project_id, before, chunks, captured) -> None:
        after = _dev_state(project_id)
        raw = b"".join(chunks)
        request_body: Any = None
        upload_filename = None
        content_type = ""
        for key, value in scope.get("headers") or []:
            if key.lower() == b"content-type":
                content_type = value.decode("latin-1", "replace")
        if "multipart/form-data" in content_type:
            fm = re.search(rb'filename="([^"]+)"', raw[:4096])
            upload_filename = fm.group(1).decode("utf-8", "replace") if fm else "upload"
        elif raw:
            try:
                request_body = json.loads(raw.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                request_body = {"_unparsed_bytes": len(raw)}

        try:
            response_body = json.loads(bytes(captured["body"]).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            response_body = {"_bytes": len(captured["body"])}

        action = {
            "method": scope.get("method"),
            "path": path,
            "query": (scope.get("query_string") or b"").decode("latin-1"),
            "project_id": project_id,
            "kind": (request_body or {}).get("kind") if isinstance(request_body, dict) else None,
            "status": captured["status"],
            "upload_filename": upload_filename,
            "request_body": request_body,
            "response_body": _truncate(response_body),
        }
        devchecks.capture_action(action, before, after)

        # The upload has no `before`, so the battery runs against `after` alone
        # and the transition checks that need both simply find nothing to
        # compare — which is correct rather than skipped.
        project = None
        if project_id:
            try:
                project = STORE.get(project_id)
            except ProjectError:
                project = None
        # THE CHECKS RUN ON ACCEPTED ACTIONS ONLY, and this is a correction
        # rather than a convenience. A 400 means the app REFUSED — and the
        # refusal branch is the governing rule working. Running "every action
        # records a decision" against a request the app declined reports the
        # refusal as a defect, which would fill a drive with violations that are
        # the app being right. The request is still captured; only the
        # invariants that presuppose the action happened are skipped.
        status = captured["status"] or 0
        if after is not None and 200 <= status < 300:
            devchecks.check_transition(project, before, after, action)
        devchecks.write_index()


# Which pull affordances a pack prior can gate, and the question whose prior
# gates them. One entry, and the mapping is explicit rather than derived from a
# naming convention: a gate that attached itself by string similarity would
# attach to the wrong figure the first time somebody renamed one.
PULL_GATES: Dict[str, str] = {
    "look::r8_collinearity": "collinearity_figure",
}


def _pull_gate(project: AnalysisProject, key: str) -> Optional[Dict[str, Any]]:
    """The gate on one pull affordance, or `None`."""
    question = PULL_GATES.get(key)
    if not question or not project.lens:
        return None
    from turbotab import packs as _packs
    found = _packs.priors(project.lens, question, project.df)
    if not found:
        return None
    return {"packs": [g["pack"] for g in found],
            "columns": sorted({c for g in found for c in g.get("columns", [])}),
            "reason": found[0]["reason"], "draw": found[0].get("gate")}


def _dev_state(project_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """The resolved project, as the interface would receive it. Never raises.

    Taken through a JSON round trip, which is what makes it a SNAPSHOT rather
    than a view. `to_dict` now copies its containers (`STATE-111`), but a
    snapshot the harness holds across an action has to be independent all the
    way down or the diff compares a thing against itself — and the harness must
    not depend on the project model getting that right, since noticing when it
    does not is part of the job.
    """
    if not project_id:
        return None
    try:
        return json.loads(json.dumps(_payload(STORE.get(project_id)), default=str))
    except Exception:                                      # pragma: no cover
        return None


def _truncate(body: Any, limit: int = 400) -> Any:
    """Findings and columns dominate a response and are already in `state/`."""
    if not isinstance(body, dict):
        return body
    out = dict(body)
    for key in ("findings", "columns", "decisions"):
        value = out.get(key)
        if isinstance(value, list) and len(value) > 8:
            out[key] = value[:8] + [{"_truncated": len(value) - 8}]
    text = json.dumps(out, default=str)
    if len(text) > limit * 200:
        return {"_truncated_response_bytes": len(text)}
    return out


app.add_middleware(DevCapture)


class DecisionIn(BaseModel):
    """One answer from the interview.

    `kind` is the shared vocabulary between the record and the frontend:
    ``set_target`` · ``set_grain`` · ``seal`` · ``defer`` · ``dismiss``
    · ``undismiss`` · ``flag`` · ``unflag`` · ``note``.
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
    structural = engine.diagnose(project.df, target=project.target)
    prof = None
    try:
        prof = engine.profile(project.df, project.target, project.task_type)
    except ValueError as exc:
        # `compute_dataset_profile` raises on a frame it cannot profile. The
        # structural findings are still real and still worth showing — reporting
        # nothing here would present an unprofiled file as a clean one.
        devchecks.swallowed(
            "api._recompute::profile", exc,
            "every profile-derived finding is absent from this render, and the "
            "file presents as one the profiler had nothing to say about")
        prof = None

    # THE LENS IS A PARAMETER OF THE DIAGNOSIS, not something applied to it
    # afterwards here. `rank_findings` is the one function that produces the
    # finding list the app presents, and nothing reaches a user except through
    # it — which is what makes "the lens comes before the diagnosis" a property
    # of the code rather than of the order of two statements in this function.
    project.set_findings(
        engine.rank_findings(structural, prof, lens=project.lens or [],
                             df=project.df),
        engine.profile_to_dict(prof) if prof is not None else None,
    )


def _disclosures(project: AnalysisProject) -> Dict[str, Any]:
    """The sentences the user reads about the grain answer and the seal.

    Served rather than composed in the page, for the same reason
    `/capabilities` is: an interface that writes its own disclosure can drift
    from what the record says, and the disclosure is the record's claim about
    itself.
    """
    out: Dict[str, Any] = {"grain": None, "eligibility": None, "seal": None,
                           "preprocess": None, "exploratory": False}
    if project.grain:
        out["grain"] = grain_mod.answer_disclosure(
            project.grain["answer"], project.grain.get("group_col"))
        if project.grain.get("contradiction_acknowledged"):
            out["attested"] = (
                "You confirmed this answer against the shape of the data. The "
                "disagreement is recorded and carries into the methods section "
                "as a stated limitation.")
    if project.eligibility:
        from turbotab import eligibility as _elig
        out["eligibility"] = _elig.disclosure(project.eligibility)
    if project.missingness or project.preprocess_settled:
        from turbotab import missingness as _miss
        out["preprocess"] = _miss.plan_receipt(
            project.missingness, len(_miss.survey(project.df, project.target)))
    if project.lockbox:
        out["seal"] = grain_mod.seal_disclosure(project.lockbox)
        out["exploratory"] = grain_mod.is_exploratory_basis(
            project.lockbox.get("seal_basis"))
        if (project.grain or {}).get("contradiction_acknowledged"):
            # §09: the attestation flows into the record so the manuscript can
            # carry it as a limitation. The seal is where that matters, because
            # the seal's disclosure is what a reader takes the held-out number
            # to mean. Without this the sentence reads as a clean split, which
            # is not false but is not the whole of what the app knows.
            out["seal"] += (
                " Note: this split rests on your answer, which disagreed with "
                "the shape of the data. That disagreement is on the record and "
                "belongs in the methods section.")
            out["exploratory"] = True
    return out


def _payload(project: AnalysisProject) -> Dict[str, Any]:
    body = project.to_dict()
    body["sample"] = project.head(8)
    body["disclosures"] = _disclosures(project)
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
    except ProjectError as exc:
        # The transcript then names a finding by its ID rather than its title —
        # `binary_text__sex — deferred` instead of the sentence a reader can
        # follow. Not wrong, and not what the record is for.
        devchecks.swallowed(
            "api._sentence::finding-title", exc,
            f"the transcript will name {d.subject!r} by id rather than by title")
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

    if decision.kind == "set_lens":
        keys = decision.payload.get("lens")
        if keys is None:
            keys = [k for k in (decision.subject or "").split(",") if k]
        try:
            project.set_lens(keys)
        except ProjectError as exc:
            raise HTTPException(400, str(exc)) from exc
        # The diagnosis is read under the lens, so it is recomputed here rather
        # than left to go stale — clause §01 puts the lens BEFORE the structural
        # diagnosis, and a lens that changed nothing on screen would be a
        # question the user could not see the point of.
        _recompute(project)
        return _payload(project)

    if decision.kind == "set_reverse_coding":
        # Asked, never inferred. The list may legitimately be EMPTY — "none of
        # them are reverse-coded" is a recorded answer, and the difference
        # between that and never having asked is the whole recorded-absence
        # rule. So the empty list is stored, not treated as no answer.
        columns = list(decision.payload.get("columns") or [])
        unknown = [c for c in columns if c not in list(project.df.columns)]
        if unknown:
            raise HTTPException(
                400, f"No column named '{unknown[0]}' in this table.")
        project.record(
            "set_reverse_coding", subject=",".join(columns),
            text=(f"{len(columns)} item(s) were declared reverse-coded and will "
                  f"be flipped before the scale is scored: "
                  + ", ".join(f"`{c}`" for c in columns) + "."
                  if columns else
                  "No items were declared reverse-coded; the scale is scored "
                  "with every item in the direction it was recorded."),
            payload={"columns": columns, "source": "declared"})
        return _payload(project)

    if decision.kind == "set_grain":
        answer = decision.payload.get("answer") or decision.subject
        group_col = decision.payload.get("group_col") or None
        try:
            project.set_grain(
                str(answer), group_col,
                inherited=bool(decision.payload.get("inherited")),
                acknowledged_contradiction=bool(
                    decision.payload.get("acknowledge_contradiction")))
        except GrainContradiction as exc:
            # 409, not 400: the request is well-formed and the state disagrees
            # with it. The evidence travels with the refusal so the interruption
            # can show what it saw rather than assert that it saw something.
            # The exits travel WITH the refusal, so an interface cannot render
            # the interruption without also rendering its way out. §09: a
            # CONSEQUENCE resolves or is attested, never a dead end.
            raise HTTPException(409, {"message": str(exc),
                                      "contradiction": exc.detail,
                                      "exits": exc.detail.get("exits", [])}) from exc
        except ProjectError as exc:
            raise HTTPException(400, str(exc)) from exc
        return _payload(project)

    if decision.kind in ("set_repeat_kind", "set_unit_of_analysis",
                         "set_aggregation", "set_temporal_prediction"):
        payload = decision.payload or {}
        try:
            if decision.kind == "set_repeat_kind":
                project.set_repeat_kind(
                    str(payload.get("kind") or decision.subject),
                    overturned=bool(payload.get("overturned")))
            elif decision.kind == "set_unit_of_analysis":
                project.set_unit_of_analysis(
                    str(payload.get("unit") or decision.subject))
            elif decision.kind == "set_aggregation":
                project.set_aggregation(
                    str(payload.get("method") or decision.subject))
            else:
                project.set_temporal_prediction(bool(payload.get("temporal")))
        except ProjectError as exc:
            raise HTTPException(400, str(exc)) from exc
        # Aggregation rewrote the table, so the diagnosis is of a different
        # table. The other three change no value and recompute anyway, because
        # recomputing a diagnosis is cheap and reasoning about which of four
        # branches needs it is how one of them stops.
        _recompute(project)
        return _payload(project)

    if decision.kind == "set_eligibility":
        try:
            project.set_eligibility(
                str(decision.payload.get("answer") or decision.subject),
                column=decision.payload.get("column"),
                minimum=decision.payload.get("minimum"),
                maximum=decision.payload.get("maximum"),
                keep_values=decision.payload.get("keep_values"),
                reason=str(decision.payload.get("reason") or ""))
        except ProjectError as exc:
            raise HTTPException(400, str(exc)) from exc
        _recompute(project)
        return _payload(project)

    if decision.kind == "eligibility_evidence":
        # A GET-shaped read served through the decision endpoint so it stays
        # beside the question it belongs to. Bounded by clause §04: this answers
        # "is this data corrupted?" and cannot answer "where should I cut?".
        from turbotab import eligibility as _elig
        try:
            return _elig.permitted_evidence(
                project.df, str(decision.payload.get("column") or decision.subject))
        except _elig.EligibilityRefusal as exc:
            raise HTTPException(400, str(exc)) from exc

    if decision.kind == "select_models":
        try:
            project.select_models(decision.payload.get("models") or [])
        except ProjectError as exc:
            raise HTTPException(400, str(exc)) from exc
        return _payload(project)

    if decision.kind == "set_preparation_mode":
        try:
            project.set_preparation_mode(
                str(decision.payload.get("mode") or decision.subject))
        except ProjectError as exc:
            raise HTTPException(400, str(exc)) from exc
        return _payload(project)

    if decision.kind == "set_model_recipe":
        try:
            project.set_model_recipe(
                str(decision.payload.get("model") or ""),
                str(decision.payload.get("operation") or ""),
                str(decision.payload.get("variant") or ""))
        except ProjectError as exc:
            raise HTTPException(400, str(exc)) from exc
        return _payload(project)

    if decision.kind == "route_missingness":
        from turbotab import missingness as _miss
        col = decision.payload.get("column") or decision.subject
        mech = str(decision.payload.get("mechanism") or "")
        strat = str(decision.payload.get("strategy") or "")
        # The CONSEQUENCE is surfaced BEFORE the refusal is raised, so the
        # interface gets the interruption with both exits attached rather than
        # a 400 it has to interpret. §09: resolves or is attested.
        try:
            n_missing = int(project.df[col].isna().sum()) if col in project.df.columns else 0
        except Exception:
            n_missing = 0
        if (_miss.blocks(mech, strat)
                and not decision.payload.get("acknowledge_signal_loss")):
            raise HTTPException(409, _miss.blocker(col, mech, strat, n_missing))
        try:
            project.route_missingness(
                col, mech, strat,
                uses_columns=decision.payload.get("uses_columns"),
                acknowledged=bool(decision.payload.get("acknowledge_signal_loss")))
        except ProjectError as exc:
            raise HTTPException(400, str(exc)) from exc
        _recompute(project)
        return _payload(project)

    if decision.kind == "route_missingness_bulk":
        from turbotab import missingness as _miss
        payload = decision.payload or {}
        columns = [str(c) for c in payload.get("columns") or []]
        branch = str(payload.get("branch") or "")
        mech = str(payload.get("mechanism") or "")
        strat = str(payload.get("strategy") or "")
        # The CONSEQUENCE is surfaced before the refusal, exactly as it is for
        # one column — and it names the COUNT, because "you are about to impute
        # over an informatively-missing column" reads very differently when it
        # is 294 of them.
        if _miss.blocks(mech, strat) and not payload.get("acknowledge_signal_loss"):
            total = sum(int(project.df[c].isna().sum())
                        for c in columns if c in project.df.columns)
            block = _miss.blocker(f"{len(columns):,} {branch} column(s)",
                                  mech, strat, total)
            block["n_columns"] = len(columns)
            raise HTTPException(409, block)
        try:
            project.route_missingness_bulk(
                branch, mech, strat, columns,
                uses_columns=payload.get("uses_columns"),
                acknowledged=bool(payload.get("acknowledge_signal_loss")))
        except ProjectError as exc:
            raise HTTPException(400, str(exc)) from exc
        _recompute(project)
        return _payload(project)

    if decision.kind == "except_from_bulk":
        # Editing the RULE, not the members: a column pulled out of the set
        # rejoins the individually-asked ones and the rule's sentence says so.
        column = str(decision.payload.get("column") or decision.subject)
        try:
            project.except_from_bulk(column)
        except ProjectError as exc:
            raise HTTPException(400, str(exc)) from exc
        return _payload(project)

    if decision.kind == "settle_preprocess":
        try:
            project.settle_preprocess(bool(decision.payload.get("skipped")))
        except ProjectError as exc:
            raise HTTPException(400, str(exc)) from exc
        return _payload(project)

    if decision.kind == "trim_training_rows":
        try:
            project.trim_training_rows(
                decision.payload.get("column") or decision.subject,
                minimum=decision.payload.get("minimum"),
                maximum=decision.payload.get("maximum"),
                reason=str(decision.payload.get("reason") or ""))
        except ProjectError as exc:
            raise HTTPException(400, str(exc)) from exc
        _recompute(project)
        return _payload(project)

    if decision.kind == "seal":
        if project.target is None:
            raise HTTPException(400, "The held-out set is drawn against the "
                                     "outcome, so the target comes first.")
        if project.grain is None:
            raise HTTPException(
                400, "The grain question comes before the seal: whether one "
                     "person can appear in more than one row decides how the "
                     "held-out rows are chosen.")
        # Clause §01's bracketed steps, read from the project rather than
        # restated here. They sit BETWEEN the grain and eligibility, so this
        # check does too — the first version put it after and told a driver who
        # had not said what one row means to answer a question two steps on.
        gap = project.repeat_chain_gap()
        if gap:
            raise HTTPException(400, gap)
        if project.eligibility is None:
            raise HTTPException(
                400, "The eligibility question comes before the seal: whether "
                     "your study is restricted to part of this data decides "
                     "which rows the held-out set is drawn from. Answering "
                     "'the study is about everyone here' settles it.")
        try:
            drawn = engine.draw_holdout(
                project.df, project.target, project.task_type or "regression",
                project.grain,
                fraction=float(decision.payload.get("fraction", 0.15)),
                seed=int(decision.payload.get("seed", 42)))
            project.seal_lockbox(drawn["labels"], **drawn["disclosure"])
        except engine.EngineRefusal as exc:
            raise HTTPException(400, str(exc)) from exc
        except ProjectError as exc:
            raise HTTPException(400, str(exc)) from exc
        return _payload(project)

    if decision.kind == "add_feature":
        try:
            project.add_feature(decision.payload.get("transform") or decision.subject,
                                decision.payload.get("columns") or [],
                                decision.payload.get("params") or {})
        except ProjectError as exc:
            raise HTTPException(400, str(exc)) from exc
        _recompute(project)
        return _payload(project)

    if decision.kind == "remove_feature":
        try:
            project.remove_feature(decision.payload.get("column") or decision.subject)
        except ProjectError as exc:
            raise HTTPException(400, str(exc)) from exc
        _recompute(project)
        return _payload(project)

    if decision.kind == "defer_feature":
        try:
            project.defer_feature(decision.payload.get("transform") or decision.subject,
                                  decision.payload.get("columns") or [],
                                  decision.payload.get("params") or {})
        except ProjectError as exc:
            raise HTTPException(400, str(exc)) from exc
        return _payload(project)

    if decision.kind == "set_selection":
        payload = decision.payload or {}
        if not payload.get("method"):
            try:
                project.set_selection(None)          # "use every column"
            except ProjectError as exc:
                raise HTTPException(400, str(exc)) from exc
            return _payload(project)
        try:
            spec = sel_mod.declare(
                payload["method"], project.target or "",
                payload.get("candidates") or [],
                n_features=payload.get("n_features"),
                consensus_min_methods=payload.get("consensus_min_methods"),
                scope=payload.get("scope", sel_mod.TRAIN_FOLDS))
            project.set_selection(spec)
        except sel_mod.SelectionRefusal as exc:
            raise HTTPException(400, str(exc)) from exc
        except ProjectError as exc:
            raise HTTPException(400, str(exc)) from exc
        return _payload(project)

    if decision.kind == "settle_features":
        project.settle_features(skipped=bool(decision.payload.get("skipped")))
        return _payload(project)

    if decision.kind == "apply":
        # The only endpoint in this service that changes the working table, and
        # it is reached only by asking for it by name. A preview never lands
        # here: it computes on a copy and throws the copy away.
        try:
            live = engine.find_shape_finding(
                engine.diagnose(project.df, target=project.target), decision.subject)
            # The identity barrier (T0-ID-001). Refused here rather than
            # detected afterwards: once the lockbox is sealed there is no way to
            # recover which rows its labels meant.
            project.check_repair_allowed(live.fix_kind)
            # An answer the finding cannot supply itself. Today that is which
            # level of the outcome is the event — never defaulted, at any
            # confidence, because it is the research question rather than a
            # property of the data.
            choice = decision.payload.get("choice") or decision.payload.get("event")
            if live.fix_kind == "set_positive_class" and not choice:
                raise HTTPException(
                    400, "Setting the event needs the level being predicted. "
                         "There is no default: whether the event is (say) death "
                         "or survival is the research question, not something "
                         "the file can say.")
            prev = engine.preview_fix(project.df, live, choice=choice)
            if not prev.get("applicable"):
                raise HTTPException(
                    400, "That finding has no automatic repair — it needs a human decision.")
            new_df, description = engine.apply_fix(project.df, live, choice=choice)
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
async def preview_finding(project_id: str, finding_id: str,
                          choice: Optional[str] = None) -> Dict[str, Any]:
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
        live = engine.find_shape_finding(
            engine.diagnose(project.df, target=project.target), finding_id)
        if live.fix_kind == "set_positive_class" and not choice:
            # A preview of "set the event" with no event named would have to
            # pick one, and picking one is the thing this question forbids.
            return {
                "finding_id": live.id, "fix_kind": live.fix_kind,
                "fix_label": live.fix_label, "applicable": False,
                "description": (
                    "Choose which level is the event and this will show what "
                    "changes. There is no default — the choice sets the sign of "
                    "every effect estimate."),
                "choices": (live.params or {}).get("spellings", {}),
                "suggested": (live.params or {}).get("suggested"),
                "suggested_reason": (live.params or {}).get("suggested_reason"),
            }
        return engine.preview_fix(project.df, live, choice=choice)
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
    """The correlation matrix, GATED where the columns are parts of a whole.

    `params["gates"] = "collinearity_figure"` on the compositional finding used
    to be a field a test asserted the existence of and nothing read. It reads
    here: correlation between parts of a whole is negatively biased BY
    CONSTRUCTION — raising one necessarily lowers another — so a matrix drawn
    over them is not a figure with a caveat, it is a figure that cannot be read.

    The gate ANNOTATES rather than withholds, for the same reason reframing
    annotates rather than deletes: the other columns' correlations are real and
    a user who asked to see the matrix is entitled to it. What the gate adds is
    which cells cannot be interpreted and why.
    """
    from turbotab import packs as _packs
    project = _project(project_id)
    out = engine.correlations(project.working_table)
    gate = _packs.priors(project.lens or [], "collinearity_figure", project.df)
    if gate:
        out["gated"] = True
        out["gates"] = [{"pack": g["pack"], "columns": g.get("columns", []),
                         "reason": g["reason"], "draw": g.get("gate")}
                        for g in gate]
    return out


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


@app.get("/project/{project_id}/grain")
async def get_grain(project_id: str) -> Dict[str, Any]:
    """The grain question's material: the suggestion, and what was answered.

    A `GET`, because it is a question. The suggestion is offered under "yes,
    people repeat" and is never the answer — constitution §02 demotes the
    heuristics to exactly this role. `evidence` is the shape-only reading, so a
    reviewer can see WHY a column was suggested rather than trusting that it
    was.
    """
    project = _project(project_id)
    return {
        "question": "Can one person appear in more than one row?",
        "why": ("This decides how your held-out rows are chosen. If the same "
                "person lands on both sides, your held-out numbers will look "
                "better than the model is."),
        "options": [
            {"answer": grain_mod.ONE_ROW_PER_PERSON,
             "label": "No, one row per person"},
            {"answer": grain_mod.PEOPLE_REPEAT,
             "label": "Yes, people repeat",
             "follow_up": "which column identifies the person?"},
            {"answer": grain_mod.NOT_SURE, "label": "I'm not sure"},
        ],
        "suggestion": grain_mod.suggestion(project.df),
        "answered": project.grain,
    }


@app.get("/project/{project_id}/repeats")
async def get_repeats(project_id: str) -> Dict[str, Any]:
    """Questions 4 to 7's material: the reading, its evidence, and the menu.

    A `GET`, because they are questions. `applies: false` is the ordinary
    answer — most datasets never reach any of these, which is
    `OPENING_SEQUENCE.md` §02's whole claim about the count tracking the shape
    of the study.
    """
    from turbotab import repeats as _rep
    project = _project(project_id)
    grain = project.grain or {}
    if grain.get("answer") != "people_repeat":
        return {"applies": False,
                "why_not": ("These questions only mean something when a person "
                            "can appear in more than one row. The grain answer "
                            "says they cannot.")}
    reading = _rep.read(project.df, grain.get("group_col"))
    kind = (project.repeat_kind or {}).get("kind")
    return {
        "applies": True,
        "group_col": grain.get("group_col"),
        "reading": reading,
        "reopen": _rep.REOPEN,
        "answered": {
            "repeat_kind": project.repeat_kind,
            "unit_of_analysis": project.unit_of_analysis,
            "aggregation": project.aggregation,
            "temporal_prediction": project.temporal_prediction,
        },
        "menu": _rep.menu(kind, project.lens or []) if kind else None,
        "temporal": {
            "applies": bool(kind == _rep.TIME_POINTS
                            and project.unit_of_analysis == _rep.UNIT_RECORD),
            "why": _rep.TEMPORAL_WHY,
            "consumer": _rep.TEMPORAL_CONSUMER,
        },
    }


@app.get("/project/{project_id}/lens")
async def get_lens(project_id: str) -> Dict[str, Any]:
    """The lens question's material: the options, the suggestion, the answer.

    A `GET`, because it is a question. The suggestion is offered beside the
    options and is never the answer — the same demotion constitution §02 applies
    to the grouping heuristics, for the same reason: a pack that fires on the
    wrong data asserts something false authoritatively, which is harder to catch
    than an ordinary bug.
    """
    from turbotab import packs as _packs
    project = _project(project_id)
    return {
        **_packs.question(_packs.suggest(project.df)),
        "answered": project.lens,
        "methods_sentence": (_packs.methods_sentence(project.lens)
                             if project.lens else None),
        "contradiction": _packs.contradiction(project.df, project.lens or []),
    }


@app.get("/project/{project_id}/models")
async def get_models(project_id: str) -> Dict[str, Any]:
    """The shelf: every model this task can use, ordered and never filtered.

    Three groups always returned, including empty ones — "nothing is recommended
    for this data" is a real state and a renderer that only sees two groups
    cannot say it.
    """
    from turbotab import models as _models
    project = _project(project_id)
    if not project.barrier_raised:
        raise HTTPException(
            400, "The shelf is ordered by the shape of your data, so it is "
                 "offered after the seal — the shape it reads must be the "
                 "shape the models will be fitted on.")
    entries = project.model_shelf()
    from turbotab import packs as _packs
    return {
        "disclosure": _models.SHELF_DISCLOSURE,
        # Dataset-scoped, and legitimately so: p much greater than n is a
        # property of the shape rather than of any column. The shelf is ORDERED
        # by this and never filtered by it — a competent researcher can have a
        # reason for a tree ensemble at p >> n.
        "priors": _packs.priors(project.lens or [], "model_ranking", project.df),
        "groups": _models.grouped(entries),
        "selected": project.selected_models,
        "n_available": len(entries),
        "concern_note": _models.selection_note(entries, project.selected_models),
    }


@app.get("/project/{project_id}/recipes")
async def get_recipes(project_id: str) -> Dict[str, Any]:
    """Per-model preprocessing: what the table resolves, and what to ask.

    Two axes, and they are not the same axis:

    * **Determinacy** decides whether the row is a question at all. A FACT is
      pre-selected with a rendered skip; a CHOICE is asked, and no confidence in
      the engine makes a judgment about this data moot.
    * **Divergence** decides whether a pre-selected fact's VARIANT is worth
      raising. *Whether* a model gets scaled inputs is the model's property and
      the table settles it. *Which* scaling, once it happens, is a judgment
      about this data — so standard and robust are put against each other, and
      the question surfaces only when they would put the columns on different
      relative footings.

    `n_choices_suppressed` counts the second: variant questions that were
    derived, compared and found not to change the answer. A row with no pushed
    alternative is not counted — there was no question to suppress.
    """
    from turbotab import packs as _packs, recipes as _rec
    project = _project(project_id)
    # THE RECIPE TABLE IS CANONICAL for a pack's variant preferences
    # (`GUIDED-025`), so the packs are loaded INTO it here and resolution reads
    # one structure. Idempotent.
    _packs.load(project.lens or [])
    numeric = [str(c) for c in project.df.columns
               if pd.api.types.is_numeric_dtype(project.df[c])
               and str(c) != (project.target or "")]
    resolved = project.resolved_recipes()
    suppressed = 0
    for rows in resolved.values():
        for row in rows:
            r = _rec.resolve(row["model"], row["operation"])
            raise_variant, div = _rec.worth_asking(project.df, numeric, r)
            row["divergence"] = div.to_dict() if div else None

            if row["may_be_preselected"]:
                # The fact is pre-selected either way; the variant question is
                # the only thing the divergence check can add or remove.
                row["ask"] = False
                row["skip_reason"] = row["reason"]
                row["variant_worth_raising"] = bool(raise_variant)
                if raise_variant and div is not None:
                    row["variant_prompt"] = (
                        f"{div.b} instead of {div.a}? {div.evidence}")
                elif div is not None:
                    suppressed += 1
                    row["variant_skipped_because"] = div.evidence
                continue

            # A CHOICE stays asked. The divergence, when known, travels beside
            # it as evidence — it informs the answer, it does not replace it.
            row["ask"] = True
            row["variant_worth_raising"] = bool(raise_variant)
    return {
        "mode": project.preparation_mode,
        "models": resolved,
        "operations": [{"key": o.key, "label": o.label,
                        "determinacy": o.determinacy, "scope": o.scope,
                        "because": o.because, "origin": o.origin,
                        "variants": list(o.variants),
                        "pushed_alternatives": [list(p)
                                                for p in o.pushed_alternatives]}
                       for o in _rec.operations()],
        "n_choices_suppressed": suppressed,
        # Read back out of the recipe table rather than mirrored from the pack,
        # because a second copy of what a pack registered is the drift
        # `GUIDED-025` names, one level down.
        "pack_defaults": _packs.recipe_origins(project.lens or []),
        "normalization": _packs.priors(project.lens or [], "normalization",
                                       project.df),
    }


@app.get("/project/{project_id}/preprocess")
async def get_preprocess(project_id: str) -> Dict[str, Any]:
    """The Preprocess step's state: which columns have blanks, what was decided.

    Each strategy carries `because` — clause §06's litmus answer in words — and
    `defers`, so an interface can tell the user WHY a choice changes nothing on
    screen instead of leaving them to conclude the app did nothing.
    """
    from turbotab import missingness as _miss, packs as _packs
    project = _project(project_id)
    survey = project.missingness_survey()
    return {
        "columns": survey,
        "strategies": {
            "numeric": [_miss.strategy(k) for k in _miss.NUMERIC_STRATEGIES],
            "categorical": [_miss.strategy(k) for k in _miss.CATEGORICAL_STRATEGIES],
        },
        "mechanism_question": {
            "why": _miss.MECHANISM_WHY,
            "consumer": _miss.MECHANISM_CONSUMER,
            "options": list(_miss.MECHANISM_OPTIONS),
            # One entry per column, because the prior is a fact about the
            # column and not about the table. Where two packs disagree BOTH
            # appear, named — a mixed table gets the assay reading on its
            # features and the clinical reading on its labs, and where those
            # collide the user is shown the collision rather than one of them.
            "priors": {
                r["column"]: _packs.prior_for_column(
                    project.lens or [], "missingness_direction",
                    r["column"], project.df)
                for r in survey} if project.lens else {},
        },
        "declared": project.missingness,
        "settled": project.preprocess_settled,
        "receipt": _miss.plan_receipt(project.missingness, len(survey)),
    }


@app.get("/project/{project_id}/features")
async def get_features(project_id: str) -> Dict[str, Any]:
    """The transform catalogue, split by clause §06, and what has been decided.

    `row_local` executes immediately and posts a receipt. `deferred` is
    recorded now and fitted inside each training fold. Each entry carries
    `because` — the litmus answer in words — so the interface can show the
    reasoning rather than assert the classification.
    """
    project = _project(project_id)
    numeric = [str(c) for c in project.df.columns
               if pd.api.types.is_numeric_dtype(project.df[c])
               and str(c) != (project.target or "")]
    return {
        "row_local": [feat_mod.get(k).to_dict() for k in feat_mod.row_local_keys()],
        "deferred": [feat_mod.get(k).to_dict() for k in feat_mod.deferred_keys()],
        "numeric_columns": numeric,
        "all_columns": [str(c) for c in project.df.columns],
        "engineered": project.engineered,
        "deferred_transforms": project.deferred_transforms,
        "selection": project.selection_spec,
        "settled": project.features_settled,
        "selection_methods": [
            {"key": k, "label": m.label,
             "explainability_cost": m.explainability_cost}
            for k, m in sel_mod.METHODS.items()],
    }


@app.get("/project/{project_id}/feature/preview")
async def preview_feature(project_id: str, transform: str, columns: str,
                          params: str = "") -> Dict[str, Any]:
    """Before/after for one transform, computed on a copy and thrown away.

    A `GET`, because it is a question. A CHOICE gets a preview
    (`DESIGN_LANGUAGE.md` §09) and the preview is the REAL computation rather
    than a description of one — a description is a claim about what would
    happen, which is the class of thing this project keeps finding to be wrong.
    """
    import json as _json
    project = _project(project_id)
    cols = [c for c in columns.split(",") if c]
    try:
        parsed = _json.loads(params) if params else {}
    except ValueError as exc:
        raise HTTPException(400, f"params is not valid JSON: {exc}") from exc
    try:
        return feat_mod.preview(project.df, transform, cols, parsed)
    except feat_mod.FeatureRefusal as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/project/{project_id}/selection/evidence")
async def selection_evidence(project_id: str) -> Dict[str, Any]:
    """What a selection CHOICE is shown beside — ranked on training rows only.

    Ranks, does not choose. Nothing is stored and the response is marked
    `preview_not_applied`, the same distinction clause §06 draws for a deferred
    transform's preview.
    """
    project = _project(project_id)
    if not project.target:
        raise HTTPException(400, "Ranking features needs the outcome first.")
    candidates = [str(c) for c in project.df.columns
                  if str(c) != project.target
                  and pd.api.types.is_numeric_dtype(project.df[c])]
    mask = None
    if project.lockbox and project.lockbox.get("labels"):
        sealed = set(project.lockbox["labels"])
        mask = pd.Series([i not in sealed for i in project.df.index],
                         index=project.df.index)
    try:
        return sel_mod.evidence(project.df, project.target, candidates, mask)
    except sel_mod.SelectionRefusal as exc:
        raise HTTPException(400, str(exc)) from exc


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

    missing_columns = [r["column"] for r in project.missingness_survey()]
    # The priors that apply to each column with blanks, resolved against the
    # frame here because `ml/router.py` takes no dataframe. Column by column
    # (`GUIDED-027`): a dataset-level "below the detection limit" would be
    # wrong for most columns of an NHANES-shaped table.
    from turbotab import bulk as _bulk, packs as _packs
    missingness_priors = {
        col: _packs.prior_for_column(project.lens or [], "missingness_direction",
                                     col, project.df)
        for col in missing_columns} if project.lens else {}

    # THE GROUPS, resolved against the frame here because the Router takes no
    # dataframe (`GUIDED-029`). A group is what REMAINS after pack priors settle
    # their columns: a bulk question stating a count the user cannot reconcile
    # with what they are being shown is worse than no bulk question.
    survey_rows = project.missingness_survey()
    settled = {}
    for row in survey_rows:
        priors = missingness_priors.get(row["column"]) or []
        if any(p.get("marker") == "derived" for p in priors) and len(priors) == 1:
            settled.setdefault(row["branch"], []).append(row["column"])
    groups = _bulk.group_columns(survey_rows, settled=settled,
                                 excepted=project.bulk_exceptions())
    missingness_groups = [g.to_dict() for g in groups]
    missingness_settled = _bulk.settled_groups(survey_rows, missingness_priors)

    # And the exceptions, computed against the ANSWER the user already gave —
    # so they exist only after a bulk decision, which is when a disagreement
    # with it is a thing that can be raised.
    missingness_exceptions = {}
    for decision in project.decisions:
        if decision.kind != "route_missingness_bulk":
            continue
        branch = decision.payload.get("branch")
        group = next((g for g in groups if g.branch == branch), None)
        if group is None:
            group = _bulk.Group(question="missingness", branch=branch,
                                members=tuple(decision.payload.get("columns") or ()))
        detail = _bulk.exceptions(project.df, group,
                                  decision.payload.get("mechanism") or "",
                                  project.target)
        if detail.get("columns"):
            missingness_exceptions[branch] = detail
    answered, deferred = [], {}
    for d in project.decisions:
        if d.kind == "set_lens":
            answered.append("state_lens")
        elif d.kind == "set_reverse_coding":
            answered.append("state_reverse_coding")
        elif d.kind == "set_target":
            answered.append("choose_target")
        elif d.kind == "set_grain":
            answered.append("state_grain")
        elif d.kind == "set_repeat_kind":
            answered.append("state_repeat_kind")
        elif d.kind == "set_unit_of_analysis":
            answered.append("state_unit_of_analysis")
        elif d.kind == "set_aggregation":
            answered.append("state_aggregation")
        elif d.kind == "set_temporal_prediction":
            answered.append("state_temporal_prediction")
        elif d.kind == "set_eligibility":
            answered.append("state_eligibility")
        elif d.kind == "route_missingness":
            answered.append(f"missingness::{d.subject}")
        elif d.kind == "route_missingness_bulk":
            answered.append(f"missingness_bulk::{d.subject}")
            for column in d.payload.get("columns") or []:
                answered.append(f"missingness::{column}")
        elif d.kind == "resolve_missingness_exceptions":
            answered.append(f"missingness_exceptions::{d.subject}")
        elif d.kind == "select_models":
            answered.append("choose_models")
        elif d.kind == "set_preparation_mode":
            answered.append("choose_preparation_mode")
        elif d.kind == "settle_features":
            answered.append("choose_features")
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
        except Exception as exc:
            # The palette is an offer, not a promise. Losing it must not take
            # the interview's questions with it. A blocker is different: if the
            # signals could not be computed there is no blocker to hide, and the
            # next branch reports none rather than claiming none exist.
            #
            # It is still the deepest well on the Guided path: a blocker that
            # would have fired does not, and the step renders as though the data
            # raised nothing. Legitimate to continue, never legitimate to be
            # quiet about it.
            devchecks.swallowed(
                "api.get_interview::eda-signals", exc,
                "the pull palette is empty and NO BLOCKER can fire this render; "
                "the step renders as though the data raised nothing")
            recommendations, signals = [], None

    # The survey pack's detector, resolved HERE and passed in: `ml/router.py` is
    # headless and takes no dataframe, which is what keeps `plan()` a pure
    # function of the record. Gated on the lens, so a Likert block in a table
    # nobody called a survey asks nothing.
    lens_block = None
    if project.lens and "survey" in project.lens:
        from turbotab import packs as _packs
        lens_block = _packs.likert_block(project.df)

    # Questions 4 to 7's state, resolved against the frame HERE for the same
    # reason: the Router takes no dataframe. `None` when the grain says people
    # do not repeat, and then none of the four is in the plan — which is how the
    # question count tracks the shape of the study rather than the pipeline's.
    repeats_state = None
    if (project.grain or {}).get("answer") == "people_repeat":
        from turbotab import repeats as _rep
        reading = _rep.read(project.df, project.grain.get("group_col"))
        kind = (project.repeat_kind or {}).get("kind")
        repeats_state = {
            "reading": reading["reading"],
            "sentence": reading["sentence"],
            "confidence": reading["confidence"],
            "kind": kind,
            "unit": project.unit_of_analysis,
            "menu": (_rep.menu(kind, project.lens or []) if kind else None),
        }

    try:
        questions = router.plan(structural, target=project.target,
                                detection=detection, step=step,
                                deferred=deferred, answered=answered,
                                recommendations=recommendations, signals=signals,
                                missing_columns=missing_columns,
                                lens_block=lens_block, repeats=repeats_state,
                                missingness_priors=missingness_priors,
                                missingness_groups=missingness_groups,
                                missingness_exceptions=missingness_exceptions,
                                missingness_settled=missingness_settled)
        router.audit(questions)
    except router.RouterError as exc:                      # noqa: B902
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
            # A gated figure says so ON THE CHIP, before it is opened. A caveat
            # discovered after looking is a caveat applied to a reading the user
            # has already taken.
            gate = _pull_gate(project, d["key"])
            if gate:
                d["gated"] = True
                d["gate"] = gate
        rendered.append(d)

    # The audit is re-run against the SAME list the interface receives, so what
    # was audited and what is shown cannot be two different things. Recorded, not
    # raised: the plan already passed above, and a harness that turns a passing
    # render into a 500 has broken the drive it is instrumenting.
    devchecks.record_violations(
        devchecks.router_audit_passed_before_this_render(questions, rendered),
        {"kind": "render_interview", "path": f"/project/{project.id}/interview",
         "step": step})

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
# The dev endpoints — the half of the capture only the browser can supply
#
# Registered before the static mount, like everything else. `/dev/status` exists
# so the page does not have to guess: the harness is a server-side flag and the
# page asks rather than assumes, which is the same reason `/capabilities` is
# served rather than hard-coded.
# ─────────────────────────────────────────────────────────────────────────────

class DomIn(BaseModel):
    step: str = "render"
    html: str = ""


class ConsoleIn(BaseModel):
    level: str = "error"
    message: str = ""
    stack: str = ""
    url: str = ""


@app.get("/dev/status")
async def dev_status() -> Dict[str, Any]:
    s = devchecks.session()
    return {"enabled": devchecks.enabled(),
            "session": str(s.root) if s else None,
            "flag": devchecks.ENV_FLAG}


@app.post("/dev/dom")
async def dev_dom(snapshot: DomIn) -> Dict[str, Any]:
    """One DOM snapshot per render, styles inline.

    `index.html` carries its whole stylesheet in `<style>` blocks, so
    `outerHTML` is already self-contained and opens in a browser with nothing
    beside it. If that ever stops being true — an external stylesheet, a font
    link — this endpoint is where the inlining would have to happen, and the
    snapshot would silently lose its appearance until it did.
    """
    if not devchecks.enabled():
        raise HTTPException(404, "The dev harness is off.")
    name = devchecks.capture_dom(snapshot.step, snapshot.html)
    devchecks.write_index()
    return {"written": name}


@app.post("/dev/console")
async def dev_console(entry: ConsoleIn) -> Dict[str, Any]:
    if not devchecks.enabled():
        raise HTTPException(404, "The dev harness is off.")
    devchecks.capture_console(entry.level, entry.message, entry.stack, entry.url)
    devchecks.write_index()
    return {"recorded": True}


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
