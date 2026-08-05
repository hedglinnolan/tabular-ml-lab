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
    jobs as jobs_mod, selection as sel_mod,
)
from turbotab.project import (
    AnalysisProject, GrainContradiction, LensContradiction, ProjectError,
    ProjectStore, PurposeContraindication,
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

    # EVERY FINDING STATES ITS SUBJECT (`GUIDED-053`). Computed here, once, so
    # the page has one question to ask and no rule of its own — including for a
    # finding whose subject is the cohort, where an empty chip row would read as
    # a card that failed to load.
    #
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
        # `GUIDED-102`. Read off the lockbox rather than recomputed, so the
        # page cannot show a resolution for a cohort other than the sealed one.
        # The CITATIONS are merged from the module at read time instead: they
        # are static content about a research file, and a copy frozen into
        # every project record would go stale one project at a time.
        recorded = project.lockbox.get("resolution")
        if recorded:
            from turbotab import resolution as _res
            recorded = {**recorded, "sources": _res.SOURCES}
        out["resolution"] = recorded
        out["exploratory"] = grain_mod.is_exploratory_basis(
            project.lockbox.get("seal_basis"))
        if (project.grain or {}).get("design_not_described"):
            # The basis may be `grouped` and honest, and the app still cannot
            # vouch that grouping is the right treatment for a design it was not
            # told. Exploratory for a reason the basis cannot express, and the
            # sentence says which.
            out["exploratory"] = True
            out["seal"] += (
                " Note: you told us none of the offered shapes describes this "
                "study, so this split is the most conservative treatment "
                "available rather than one verified against your design. Treat "
                "these numbers as exploratory, and the methods section carries "
                "an [AUTHOR REQUIRED] gap where the design would be described.")
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
        # `GUIDED-143`. **A TEMPORAL VALIDATION THAT WAS ASKED FOR AND NOT
        # DRAWN IS NOT A CLEAN LOCK**, and this is the third note on exactly the
        # two precedents above rather than a new mechanism.
        #
        # `IMPORT-020`'s asymmetry is why it flips `exploratory` rather than
        # only appending a sentence: leaking and disclosing is the *refuse*
        # branch, and leaking behind a lock icon is the *assert something false*
        # branch. A held-out score whose split trains on rows from after the
        # rows it is scored on is optimistic, and a band reading `sealed` over
        # it tells the reader the opposite of what the app knows.
        if project.lockbox.get("temporal_honored") is False:
            out["seal"] += " " + (project.lockbox.get("temporal_sentence") or "")
            out["exploratory"] = True
    return out


def _payload(project: AnalysisProject) -> Dict[str, Any]:
    from turbotab import packs as _packs

    body = project.to_dict()
    body["sample"] = project.head(8)
    body["disclosures"] = _disclosures(project)
    # `METABOLOMICS_PACK.md` §11 — where the selected pack declines to be
    # confident. HERE rather than on `/lens`, and that is trap #6 avoided by
    # looking: the page never fetches `/lens` at all, so a hedge block served
    # there would be composed correctly, correct on the wire, and invisible to
    # every person who ever used the app. This payload is what the page holds as
    # `P` after every load and every decision.
    #
    # `None` when no selected pack has any, so the key is absent-shaped rather
    # than an empty block asserting that a pack has nothing to hedge.
    body["pack_hedges"] = _packs.hedges(project.lens or [])
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
            project.set_lens(
                keys,
                acknowledged_contradiction=bool(
                    decision.payload.get("acknowledge_contradiction")))
        except LensContradiction as exc:
            # 409, not 400: the request is well-formed and the state disagrees
            # with it. The same shape the grain contradiction uses, and the
            # exits travel WITH the refusal so an interface cannot render the
            # interruption without also rendering its way out.
            raise HTTPException(409, {"message": str(exc),
                                      "contradiction": exc.detail,
                                      "exits": exc.detail.get("exits", [])}) from exc
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
            text=(f"{len(columns)} item(s) were declared reverse-coded, per the "
                  f"scoring key: "
                  + ", ".join(f"`{c}`" for c in columns) + ". "
                  "The reverse-coding audit recomputes each item's correlation "
                  "with the rest of its scale with the reversal applied. This "
                  "app computes no scale score, so the declaration is not "
                  "applied to the table any other analysis reads."
                  if columns else
                  "No items were declared reverse-coded, per the scoring key. "
                  "This app computes no scale score; the declaration is "
                  "recorded so the methods section can state it."),
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

    if decision.kind == "set_time_column":
        try:
            project.set_time_column(
                str((decision.payload or {}).get("column") or decision.subject))
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

    if decision.kind == "eligibility_candidates":
        # Candidate criteria a pack has offered, resolved against the frame.
        # A GET-shaped read served here so it stays beside the question it
        # belongs to, exactly as `eligibility_evidence` is.
        #
        # `GUIDED-033`: the metabolomics pack states at derived confidence that
        # pooled QC rows are not participants, and nothing acted on it — so the
        # app modeled them while the record said it should not. Offered here
        # rather than applied, because an exclusion changes N.
        from turbotab import packs as _packs
        out: List[Dict[str, Any]] = []
        for prior in _packs.priors(project.lens or [], "qc_rows_excluded",
                                   project.df):
            found = next((f for f in project.pack_findings()
                          if f["id"] == prior.get("detector")), None)
            if not found:
                continue
            params = found["params"]
            column, keep = params["column"], params["qc_value"]
            keep_values = [v for v in project.df[column].dropna().unique()
                           if str(v) != str(keep)]
            out.append({
                "source": prior["pack"], "marker": prior["marker"],
                "column": column, "keep_values": [str(v) for v in keep_values],
                "n_excluded": int((project.df[column] == keep).sum()),
                "reason": prior["reason"],
                "criterion_reason": (
                    f"{int((project.df[column] == keep).sum())} pooled "
                    f"quality-control injections are not participants and were "
                    f"excluded from modeling."),
            })
        return {"candidates": out,
                "note": ("Offered, never applied. An exclusion changes N and is "
                         "reported in participant flow, so it is a criterion "
                         "you state.")}

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

    if decision.kind in ("set_impossible_missing", "keep_impossible"):
        # `GUIDED-165`. Both of these used to post `kind="note"`, which falls
        # through to the generic tail at the bottom of this function: it records
        # a sentence, calls no engine function and does not `_recompute`. So the
        # transcript said *"entries were set to missing"* and the plausibility
        # endpoint reported the same count before and after — `AUDIT-001`'s shape
        # at the decision layer, escalated into the manuscript by the draft.
        #
        # TWO KINDS RATHER THAN ONE, because with one the record could not tell
        # them apart: both buttons posted the same `kind` and the same `subject`,
        # and only the free-text prose differed. A consumer had to string-match a
        # sentence to know whether a repair happened.
        column = str(decision.payload.get("column") or decision.subject or "")
        try:
            if decision.kind == "set_impossible_missing":
                project.set_impossible_missing(column)
            else:
                project.keep_impossible(column)
        except ProjectError as exc:
            raise HTTPException(400, str(exc)) from exc
        _recompute(project)
        return _payload(project)

    if decision.kind == "route_missingness":
        from turbotab import missingness as _miss
        col = decision.payload.get("column") or decision.subject
        mech = str(decision.payload.get("mechanism") or "")
        strat = str(decision.payload.get("strategy") or "")
        # `DRIVE-008`. The missingness panel names its options in the CARD's
        # vocabulary (`explicit_missing`, `indicator_and_impute`) and the record
        # keeps them in the declaration's (`explicit_category`, `indicator`).
        # The panel used to bridge that by posting a free-text `note`, so
        # pressing "Record this" wrote a sentence and routed nothing. The join
        # is `turbotab/missingness.CARD_STRATEGY`, and an option with no
        # declaration behind it is refused rather than defaulted.
        if decision.payload.get("card_option"):
            try:
                strat = _miss.strategy_for_card_option(
                    str(decision.payload["card_option"]))
            except _miss.MissingnessRefusal as exc:
                raise HTTPException(400, str(exc)) from exc
        # The CONSEQUENCE is surfaced BEFORE the refusal is raised, so the
        # interface gets the interruption with both exits attached rather than
        # a 400 it has to interpret. §09: resolves or is attested.
        try:
            n_missing = int(project.df[col].isna().sum()) if col in project.df.columns else 0
        except Exception:
            n_missing = 0
        if (_miss.blocks(mech, strat)
                and not decision.payload.get("acknowledge_signal_loss")):
            # The branch travels with the blocker, because the way through
            # depends on it: `explicit_category` keeps the signal in a
            # categorical column and turns a numeric one into text.
            branch = "categorical"
            if col in project.df.columns:
                branch = ("numeric"
                          if pd.api.types.is_numeric_dtype(project.df[col])
                          else "categorical")
            raise HTTPException(
                409, _miss.blocker(col, mech, strat, n_missing, branch=branch))
        try:
            project.route_missingness(
                col, mech, strat,
                uses_columns=decision.payload.get("uses_columns"),
                acknowledged=bool(decision.payload.get("acknowledge_signal_loss")))
        except PurposeContraindication as exc:
            # 409, not 400: the request is well-formed and the recorded purpose
            # disagrees with it. Both exits travel with the refusal.
            raise HTTPException(409, exc.detail) from exc
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

    if decision.kind == "earmark":
        payload = decision.payload or {}
        try:
            project.earmark(str(payload.get("key") or decision.subject),
                            str(payload.get("target_step") or ""),
                            str(payload.get("label") or decision.subject),
                            subject=decision.subject)
        except ProjectError as exc:
            raise HTTPException(400, str(exc)) from exc
        return _payload(project)

    if decision.kind == "set_purpose":
        answer = decision.payload.get("answer") or decision.subject
        try:
            project.set_purpose(str(answer))
        except ProjectError as exc:
            raise HTTPException(400, str(exc)) from exc
        return _payload(project)

    if decision.kind == "set_orientation":
        answer = decision.payload.get("answer") or decision.subject
        try:
            project.set_orientation(str(answer))
        except ProjectError as exc:
            raise HTTPException(400, str(exc)) from exc
        # RECOMPUTED, not marked and left. The whole claim of question 1.5 is
        # that it acts BEFORE the diagnosis; a turned-around table whose
        # findings were still the old ones would be the clause stated and not
        # kept, and the findings are what the user acts on next.
        _recompute(project)
        return _payload(project)

    if decision.kind == "unskip":
        # `GUIDED-041`. The question comes back ASKED. This endpoint records
        # nothing about the answer, deliberately — the previous implementation's
        # whole defect was that reopening wrote an answer.
        payload = decision.payload or {}
        try:
            project.unskip(str(payload.get("key") or decision.subject),
                           str(payload.get("title") or ""))
        except ProjectError as exc:
            raise HTTPException(400, str(exc)) from exc
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
            # `GUIDED-143`, Part C. The recorded temporal objective reaches
            # the draw. It did not before — that gap is the whole of this row,
            # and `repeats.split_strategy` had exactly one caller, the setter
            # that wrote the sentence.
            _temporal = bool((project.temporal_prediction or {}).get("temporal"))
            drawn = engine.draw_holdout(
                project.df, project.target, project.task_type or "regression",
                project.grain,
                fraction=float(decision.payload.get("fraction", 0.15)),
                seed=int(decision.payload.get("seed", 42)),
                time_col=project.time_column, temporal=_temporal)
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
                # THE SCOPE THIS DOOR ACTUALLY FITS (`GUIDED-104`).
                #
                # `selection.declare`'s own docstring says scope *"is explicit
                # and has no default that hides the weaker option"*, and that
                # `TRAIN_ROWS` exists so a door that inherits Classic's
                # behavior can SAY so rather than imply the stronger claim.
                # This door took the `TRAIN_FOLDS` default, fitted train-rows-
                # once, and repaired the difference in a run note — so the
                # record asserted the stronger claim and a note in a different
                # object with a different lifetime retracted it. The archive,
                # the methods sentence and any future parity check read the
                # record.
                #
                # `train_rows` is what happens: `training.train` fits each
                # model once, on the training partition. `TRAIN_FOLDS` becomes
                # reachable the day `GUIDED-103`'s resampling policy lands, and
                # is then chosen by what will happen rather than by a default —
                # which is why the caller may still ask for it explicitly and
                # `pipeline_plan` still states the divergence if it cannot be
                # honored.
                scope=payload.get("scope", sel_mod.TRAIN_ROWS))
            project.set_selection(spec)
        except sel_mod.SelectionRefusal as exc:
            raise HTTPException(400, str(exc)) from exc
        except ProjectError as exc:
            raise HTTPException(400, str(exc)) from exc
        return _payload(project)

    if decision.kind == "keep_identifier":
        # `GUIDED-108`. The way back. Refuses a column the app never set
        # aside, rather than recording an exception to a rule that did not
        # apply — an allow-list entry for a column nobody excluded is a
        # decision about nothing that outlives the reason it was made.
        try:
            project.keep_identifier(
                str(decision.payload.get("column") or decision.subject or ""),
                keep=bool(decision.payload.get("kept", True)))
        except ProjectError as exc:
            raise HTTPException(400, str(exc)) from exc
        return _payload(project)

    if decision.kind == "promote_figure":
        # `GUIDED-107`. The consumer `promotable` has been waiting for since
        # L26. It does NOT consult the figure's tier — the ruling is that a
        # marked figure is promoted as the author marked it, and a route that
        # refused an EXPLORATORY one would be the app overruling the author in
        # their own document.
        project.promote_figure(
            str(decision.payload.get("figure_id") or decision.subject or ""),
            promoted=bool(decision.payload.get("promoted", True)))
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
            # `live.fix_kind` travels so the record can say WHICH repair blanked
            # a cell. `recode_missing` and `coerce_numeric` both do — measured
            # across the fixture set at 189 and 55 cells — and both arrive here
            # as `kind="apply"`, which names nine operations at once.
            project.apply_fix(new_df, live.id, live.title, description,
                              prev["row_identity_preserved"],
                              fix_kind=live.fix_kind)
        except engine.EngineRefusal as exc:
            raise HTTPException(400, str(exc)) from exc
        except ProjectError as exc:
            raise HTTPException(400, str(exc)) from exc
        _recompute(project)
        return _payload(project)

    if decision.kind == "decline_bulk":
        # "Leave all N as they are" — its own kind rather than `apply_bulk`
        # with an empty list, because the record should say what happened in
        # its own words. §09's recorded-absence rule: *nothing to do here* is
        # an answer and it gets a sentence, or a group the user considered and
        # declined is indistinguishable from a group nobody reached.
        from turbotab import repairs as _repairs
        found = [g for g in _repairs.group(project.findings)
                 if g.fix_kind == (decision.subject or "")]
        if not found:
            raise HTTPException(404, f"No repair group '{decision.subject}'.")
        the_group = found[0]
        project.record(
            kind="decline_bulk", subject=decision.subject or "",
            text=_repairs.sentence(the_group.label, [], the_group.columns),
            payload={"fix_kind": decision.subject, "label": the_group.label,
                     "declined": [m["id"] for m in the_group.findings],
                     "declined_columns": the_group.columns,
                     "n_offered": the_group.n})
        return _payload(project)

    if decision.kind == "apply_bulk":
        # Imported locally, as every other branch in this function does. A
        # module-level alias would be shadowed for the WHOLE function body by
        # the `route_missingness` branch's local import of the same name — the
        # binding is function-scoped, so the shadow reaches backwards.
        from turbotab import missingness as _miss
        # `DRIVE-002`. One preview, a selectable set, ONE apply, ONE decision
        # covering N features.
        #
        # Every selected column gets its OWN diff computed and applied — the
        # card shows one worked example because nine before/after tables are
        # unreadable, and a preview of one column standing in for the apply of
        # nine would be exactly the blind consent the preview exists to end.
        from turbotab import repairs as _repairs
        wanted = [str(i) for i in (decision.payload.get("findings") or [])]
        if not wanted:
            raise HTTPException(
                400, "A bulk repair needs at least one feature selected. "
                     "Applying it to none is not a repair, and leaving every "
                     "column alone is what 'dismiss' records.")
        structural = engine.diagnose(project.df, target=project.target)
        offered = [f for f in _repairs.group(project.findings)
                   if f.fix_kind == (decision.subject or "")]
        if not offered:
            raise HTTPException(
                404, f"No repair group '{decision.subject}' in this table. "
                     f"Findings are recomputed after every change, so a group "
                     f"from before a fix may no longer exist.")
        the_group = offered[0]
        n_offered = the_group.n
        label = the_group.label

        applied_columns: List[str] = []
        # WHICH ORIGINAL VALUE BECAME 1, per column (`GUIDED-157`). Read off the
        # frame the apply is about to run on, so the record and the rewrite are
        # the same plan rather than two readings of it — and read BEFORE the
        # apply, because afterwards the column is 0/1 and the levels it came
        # from are gone from the table entirely.
        encodings: Dict[str, Dict[str, Any]] = {}
        # THE BLANKS, ACCUMULATED ACROSS THE N APPLIES THAT SHARE ONE DECISION.
        # `apply_fix_quietly` records nothing by design, so its provenance has
        # to be carried up here or it is lost — which is what `GUIDED-191`
        # looks like on this path: nine frames installed, one sentence, and no
        # receipt for any cell any of them blanked.
        bulk_blanks: List[Dict[str, Any]] = []
        bulk_opaque: List[Dict[str, Any]] = []
        try:
            for finding_id in wanted:
                # RE-DIAGNOSED PER COLUMN, because each apply replaces the
                # frame and the next finding must be located in the table that
                # now exists rather than in the one that did.
                structural = engine.diagnose(project.df, target=project.target)
                live = engine.find_shape_finding(structural, finding_id)
                project.check_repair_allowed(live.fix_kind)
                if live.fix_kind != the_group.fix_kind:
                    raise HTTPException(
                        400, f"'{finding_id}' is a {live.fix_kind} repair and "
                             f"this group applies {the_group.fix_kind}. One "
                             f"control may not stand for two operations.")
                prev = engine.preview_fix(project.df, live)
                if not prev.get("applicable"):
                    raise HTTPException(
                        400, f"'{finding_id}' has no automatic repair — it "
                             f"needs a human decision, so it cannot travel "
                             f"inside a bulk one.")
                encoding = engine.fix_encoding(project.df, live)
                new_df, _description = engine.apply_fix(project.df, live)
                made = project.apply_fix_quietly(
                    new_df, live.id, prev["row_identity_preserved"],
                    fix_kind=live.fix_kind)
                bulk_blanks.extend(made.get(_miss.BLANKS_MADE) or [])
                bulk_opaque.extend(made.get(_miss.BLANKS_UNATTRIBUTABLE) or [])
                applied_columns.extend(str(c) for c in (live.affected_columns or []))
                if encoding:
                    encodings[encoding["column"]] = encoding
        except engine.EngineRefusal as exc:
            raise HTTPException(400, str(exc)) from exc
        except ProjectError as exc:
            raise HTTPException(400, str(exc)) from exc

        # THE DECLINED MEMBERS ARE PART OF THE DECISION, not left open.
        #
        # "One decision covering N features" is the finding's own words, and N
        # is the set that was OFFERED. A user who applied seven of nine decided
        # about all nine — and the first version of this recorded "2 others were
        # deliberately left as recorded" in the transcript while the interview
        # re-asked them on their own cards. The record and the draw disagreed,
        # which is precisely the failure this loop is sweeping for.
        #
        # Reopening one is still free: `undismiss` already removes
        # `repair::<id>` from the answered set, and it is the affordance a
        # declined member's row carries.
        declined = [m for m in the_group.findings if m["id"] not in wanted]
        declined_columns = [str(c) for m in declined
                            for c in (m.get("affected_columns") or [])]
        project.record(
            kind="apply_bulk", subject=decision.subject or "",
            text=_repairs.sentence(label, applied_columns, declined_columns,
                                   encodings),
            payload={"fix_kind": decision.subject, "label": label,
                     "findings": wanted, "columns": applied_columns,
                     # THE MAPPING, MACHINE-READABLE, beside the sentence that
                     # states it. `GUIDED-157`: the payload is what the draft,
                     # the manuscript and the archive read, and a payload that
                     # names the columns and not the direction is the structured
                     # form lossier than the prose — trap #7, and here the prose
                     # did not carry it either.
                     "encodings": encodings,
                     "declined": [m["id"] for m in declined],
                     "declined_columns": declined_columns,
                     "n_selected": len(wanted), "n_offered": n_offered,
                     # The same two keys every other blank writer files, in the
                     # same shape, so `blanks_the_app_made` needs no branch for
                     # the bulk path. Absent entirely when nothing was blanked.
                     **({_miss.MADE_BLANKS: True,
                         _miss.BLANKS_MADE: bulk_blanks} if bulk_blanks else {}),
                     **({_miss.BLANKS_UNATTRIBUTABLE: bulk_opaque}
                        if bulk_opaque else {})})
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
                              prev["row_identity_preserved"],
                              fix_kind=drop.fix_kind)
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
        "title": "Physiologic plausibility",
        "why": "Entries outside what a living person can produce, named.",
    },
    "look::r2_missingness": {
        "built": True, "endpoint": "missingness",
        "label": "Missingness by feature",
        "title": "Missingness by feature",
        "why": "Which columns are blank, how often, and in which rows.",
        # `GUIDED-168`. THE TITLE THIS CHIP CARRIED WAS THE CORE'S, AND IT
        # PROMISED A DIFFERENT ANALYSIS. `ml/eda_recommender.py:292` names its
        # card *Missingness Pattern Analysis* and states three deliverables;
        # `ml/router.py:399` builds the palette straight off those
        # recommendations — `title=get("title") or rid` — so the core's title
        # arrived on a Guided chip that opens per-column blank rates and a
        # question. The label above was already on the wire and already right.
        #
        # UNROUTED AND ABSENT ARE DIFFERENT AND BOTH ARE HERE, so `MISC-014`'s
        # distinction is drawn item by item rather than gestured at:
        #   · the rate per column IS delivered here;
        #   · the association-with-target test IS BUILT IN THE CORE —
        #     `ml/eda_actions.py:255-330` runs a two-sample location test or a
        #     categorical association test per high-missing column, reached
        #     from `pages/02_EDA.py:1771` — and is UNROUTED in this door, which
        #     asks the user instead of computing it;
        #   · the MCAR/MAR/MNAR reading is ABSENT FROM BOTH DOORS. Nothing
        #     under `ml/` computes one; `pages/11_Theory_Reference.py` explains
        #     Little's test and no module runs it.
        # So the sentence says *computed by neither door*, which is a claim a
        # reader can check, rather than *coming soon*, which is not.
        "instead_of": {
            "core_title": "Missingness Pattern Analysis",
            "core_source": "ml/eda_recommender.py:292",
            "delivered_here": [
                "Which columns have missing data and at what rate"],
            "asked_here_not_computed": [
                "Whether missingness is associated with target "
                "(informative missingness)"],
            "built_in_core_unrouted_here": [
                "ml/eda_actions.py:217 missingness_scan — a two-sample "
                "location test or a categorical association test of the "
                "target against each high-missing column's blank mask"],
            "absent_from_both_doors": [
                "Patterns suggesting MCAR (Missing Completely At Random) vs "
                "MAR (Missing At Random) vs MNAR (Missing Not At Random)"],
            "sentence": (
                "This is not the core's Missingness Pattern Analysis: it "
                "names each column's blank rate and asks you whether the "
                "blanks mean something. The association-with-target test that "
                "title promises is built in Classic and not wired here; the "
                "MCAR/MAR/MNAR reading it promises is computed by neither "
                "door."),
        },
    },
    "look::r8_collinearity": {
        "built": True, "endpoint": "correlations",
        "label": "Correlation matrix",
        "title": "Correlation matrix",
        "why": "Pairwise correlations across your numeric features.",
    },
    "histogram_pager": {
        "built": True, "endpoint": "histograms",
        "label": "Distribution of each feature",
        "title": "Distribution of each feature",
        "why": "One page of histograms at a time, drawn from your table.",
    },
    # `GUIDED-136`. The consumer `set_reverse_coding` has been waiting for since
    # the survey pack shipped. It is a PULL rather than a pushed question
    # because it reports on an answer the user has already given — the pushed
    # question is `state_reverse_coding`, and a second card asking the same
    # thing in the other direction would be the interview arguing with its own
    # record.
    "look::reverse_coding": {
        "built": True, "endpoint": "reverse-coding",
        "label": "Reverse-coding audit",
        "title": "Reverse-coding audit",
        "why": ("Each item's correlation with the rest of its scale, before "
                "and after the reversals you declared."),
    },
}

# Named, not counted: the reason a chip is dark is shown on the chip. Anything
# the Router offers that is not in PULL_CAPABILITIES falls back to this.
NOT_BUILT_REASON = ("Not in this build. The engine has the analysis; the Guided "
                    "door has not been wired to it yet, and a control that "
                    "silently does nothing is worse than one that says so.")


@app.get("/project/{project_id}/repair_group/{fix_kind}")
async def repair_group(project_id: str, fix_kind: str) -> Dict[str, Any]:
    """One worked example, and the set it could be run on. `DRIVE-002`.

    The worked example is the group's **first** member rather than a
    representative chosen by any cleverness — the findings arrive ranked, so the
    first is the one the engine put first, and a "representative" picked on some
    other basis would be the card quietly deciding which column is typical.

    Every member carries its own row count, because *"nine features"* is not the
    number a user needs to decide with; *"`sex`: 400 rows, `batch`: 400 rows,
    `qc_flag`: 12 rows"* is.

    **And every member carries its own encoding** (`GUIDED-157`). The worked
    example is the FIRST member's, so before this the card said which value
    became 1 for one column out of N and said nothing about the other N−1 — the
    user selected `sex`, `site` and `batch` and could see the direction of one
    of them. `encoding` is `None` for a kind that has no mapping, which is every
    kind except `read_as_binary`.
    """
    from turbotab import repairs as _repairs

    project = _project(project_id)
    found = [g for g in _repairs.group(project.findings) if g.fix_kind == fix_kind]
    if not found:
        raise HTTPException(
            404, f"No repair group '{fix_kind}' in this table. Findings are "
                 f"recomputed after every change, so a group from before a fix "
                 f"may no longer exist.")
    the_group = found[0]
    payload = the_group.to_dict()

    structural = engine.diagnose(project.df, target=project.target)
    example, error = None, None
    try:
        live = engine.find_shape_finding(structural, the_group.findings[0]["id"])
        example = engine.preview_fix(project.df, live)
        example["finding_id"] = live.id
    except engine.EngineRefusal as exc:
        # Reported, never swallowed. A group whose worked example cannot be
        # computed must not render as a group with nothing to show — the user
        # would press Apply on a preview they never saw.
        devchecks.swallowed(
            "api.repair_group::preview", exc,
            "the bulk card has no worked example and would offer an apply "
            "over a diff nobody looked at")
        error = str(exc)
    payload["example"] = example
    payload["example_error"] = error

    encodings: Dict[str, Dict[str, Any]] = {}
    for member in payload["members"]:
        try:
            live = engine.find_shape_finding(structural, member["id"])
        except engine.EngineRefusal:
            member["encoding"] = None
            continue
        member["encoding"] = engine.fix_encoding(project.df, live)
        if member["encoding"]:
            encodings[member["encoding"]["column"]] = member["encoding"]
    payload["encodings"] = encodings
    payload["effect"] = _repairs.sentence(the_group.label, the_group.columns,
                                          encodings=encodings)
    return payload


def _identifiers_mod():
    """Imported lazily, like the other engine modules this file reaches for:
    `turbotab.identifiers` imports `ml.dataset_profile`, and a module-level
    import would put that cost on every process that imports the API."""
    from turbotab import identifiers as _ids
    return _ids


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

    **And it carries all three routes, not one** (`GUIDED-166`). Setting the
    entries to missing is clause §06's row-local repair; excluding the rows is
    clause §04's eligibility criterion and is a different object with different
    rules; marking the column untrustworthy is `GUIDED-096`'s split and is not
    built. Each route travels with the decision that takes it, so a client
    holding this payload can act. Composed by `turbotab.eligibility`, which
    owns §04's rules, rather than restated here.
    """
    from turbotab import eligibility as _elig
    report = engine.plausibility(_project(project_id).working_table)
    for block in report.get("impossible", []):
        block.update(_elig.routes_from_impossible(block))
    return report


@app.get("/project/{project_id}/evidence/reverse-coding")
async def evidence_reverse_coding(project_id: str) -> Dict[str, Any]:
    """§B1.2's reverse-coding audit table. **`GUIDED-136`.**

    The app has asked *which of these items are reverse-coded* since the survey
    pack shipped — `set_reverse_coding` is dispatched, the `reverse_coding`
    prior is the one deliberate exception to `DOMAIN_PACKS.md`'s guard #1, and
    the question renders. **Nothing scored it.** A recorded decision with no
    consumer is `AGENT_ONBOARD.md` §07's first trap, on the one question this
    pack is allowed to add.

    **Recomputed per request, from the record as it stands now**, which is what
    §B1.2 means by *"re-rendered after every declared change"* and what makes
    this an audit rather than a report. There is no cached table to go stale
    and no invalidation edge to forget: the declaration is read at the top of
    this function and the numbers below it are a function of it.

    It reports and never proposes. §B1.2's central sentence is SETTLED that
    correlations cannot distinguish the four causes of a negative item–rest
    correlation, so the status vocabulary has no `should_be_reversed` in it.
    """
    from turbotab import survey as _survey

    project = _project(project_id)
    table = project.working_table
    declared: List[str] = []
    # THE LAST DECLARATION WINS, and the past is editable. Folding forward
    # rather than taking the first means a user who corrects their codebook
    # gets an audit of the correction rather than of their first answer.
    for decision in project.decisions:
        if decision.kind == "set_reverse_coding":
            declared = list(decision.payload.get("columns") or [])

    out = _survey.audit(table, declared=declared)
    if out is None:
        return {"available": False,
                "because": _survey.unavailable_because(table),
                "rows": [], "declared_reversed": list(declared)}
    return out


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


@app.get("/project/{project_id}/figures")
async def project_figures(project_id: str,
                          nutrient: Optional[str] = None) -> Dict[str, Any]:
    """Every figure this project can carry, drawn — and every one it cannot, named.

    **The figure layer's first consumer** (`GUIDED-058`, `DRIVE-009`).
    `figures.applicable()` and `figures.bundle()` had no callers anywhere; three
    figures were specifications with passing tests that no user could reach.

    Beside the `evidence/*` family rather than inside it, because those draw one
    geometry each ad hoc and this resolves WHICH figures a project supports
    through the pack mechanism — which is `DRIVE-009`'s own `act` field.

    Four lists, and the last two are the ones that would have been easy to drop:
    `admitted`, `held` (a confirmatory figure whose companion is absent),
    `unavailable` (it applies and the numbers are not there, carrying the
    refusal's own words and badge) and `not_drawn` (it does not apply, with the
    reason). A figure silently missing is indistinguishable from a figure the
    app does not have.
    """
    from turbotab import figure_bundle
    return figure_bundle.render(_project(project_id), nutrient=nutrient)


@app.get("/project/{project_id}/nutrition/prevalence")
async def nutrition_prevalence(project_id: str, nutrient: str,
                               basis: str = "usual_intake",
                               reference_kind: str = "EAR",
                               stratum: Optional[str] = None) -> Dict[str, Any]:
    """A prevalence of inadequacy, or the refusal — with what it CAN draw, drawn.

    **`GUIDED-060`, probed from outside.** `prevalence_of_inadequacy` refuses in
    four cases and each refusal offers a figure instead, because *a refusal that
    offers nothing is indistinguishable from a missing feature.* Until now the
    function was reachable only from its own test, and two of the four offers
    named a figure that does not exist — so the principle was stated and the
    path that would have tested it did not run.

    A refusal is **200 with a payload**, not a 4xx. It is an answer: the app
    knows what you asked, it is telling you the question cannot be answered from
    these data and by whom, and it is drawing the thing that can. An error code
    would say the request was malformed, and it was not.

    Every offer's `draw` target is resolved through `figures.resolve`, so it
    comes back as a registered figure or as a declared pending one with what it
    needs and the ledger row blocking it — never as a bare string nobody
    follows. Where the target is registered and this project can draw it, the
    rendered figure comes back in the same response.
    """
    from turbotab import figure_bundle, figures, nutrition
    from turbotab.packs import DIETARY

    project = _project(project_id)
    if DIETARY not in (project.lens or []):
        raise HTTPException(
            409,
            "A prevalence of inadequacy is a claim about diet, and this "
            "project's lens does not say the measurements are dietary intake. "
            "The app does not infer the field from column names — answer the "
            "lens question and the nutrition pack's reference logic applies.")
    if basis not in (nutrition.USUAL_INTAKE, nutrition.SINGLE_DAY,
                     nutrition.NAIVE_MEAN):
        raise HTTPException(
            400,
            f"'{basis}' is not one of {[nutrition.USUAL_INTAKE, nutrition.SINGLE_DAY, nutrition.NAIVE_MEAN]}. "
            f"Which of the three the distribution is decides whether a "
            f"prevalence can be computed from it at all, so it is stated "
            f"rather than assumed.")

    try:
        return {"refused": False, "nutrient": nutrient,
                **nutrition.prevalence_of_inadequacy(
                    nutrient, basis=basis, reference_kind=reference_kind,
                    stratum=stratum)}
    except nutrition.PrevalenceRefusal as refusal:
        payload = refusal.to_dict()
        payload["nutrient"] = nutrient
        try:
            payload["offer"] = figures.resolve_offer(refusal.offer)
        except figures.FigureError as exc:
            # THE FAILURE `GUIDED-060` NAMED, if it ever returns: an offer whose
            # target resolves to nothing. Surfaced rather than swallowed —
            # promising a picture nobody can draw is worse than offering
            # nothing, because it reads as a feature.
            devchecks.swallowed(
                "api.nutrition_prevalence::offer", exc,
                "the refusal offered a figure that is neither registered nor "
                "declared pending, and the user would have been shown a "
                "target nothing can draw")
            payload["offer"] = {**refusal.offer, "unresolvable": str(exc)}
            return payload

        resolved = payload["offer"]["resolved"]
        if resolved["status"] != figures.REGISTERED_STATUS:
            return payload
        # The figure is built. Draw it, so the refusal arrives with the picture
        # rather than with the name of one.
        drawn = figure_bundle.render(project)
        for row in drawn["admitted"] + drawn["held"]:
            if row["id"] == resolved["id"]:
                payload["figure"] = row
                return payload
        for row in drawn["unavailable"] + drawn["not_drawn"]:
            if row["id"] == resolved["id"]:
                payload["figure_unavailable"] = row
                return payload
        return payload                                     # pragma: no cover


@app.get("/project/{project_id}/genomics/data_type")
async def genomics_data_type(project_id: str) -> Dict[str, Any]:
    """§02's *"what your numbers are"* card — the genomics pack's own words for
    what this matrix is, and what that closes off.

    `research/GENOMICS_PACK.md` §02 calls this **the highest-leverage diagnostic
    in the pack** and the card **the single most valuable artifact in it**, for
    one reason: the classification decides what is legal downstream, and getting
    it wrong is the commonest real failure. TPM handed to a count model runs
    cleanly and reports p-values that are wrong.

    **Offered under the genomics lens only**, and a 409 otherwise, for the same
    reason the prevalence route refuses outside the dietary lens: the app does
    not infer the field from column names. `wide_assay.csv` is 45 continuous
    columns centred on zero and there is nothing in the numbers that says
    whether they are expression, spectra or sensor readings. The user's answer
    to the lens question is what licenses the sentence.

    **Three answers and no fourth**, which is what stops this rounding to the
    nearest row:

    * `read: True` — the matrix matched, with the classification, the evidence,
      and the capability matrix;
    * `read: False` — the matrix was read and matched none of the nine shapes,
      which is an ANSWER and says so;
    * a 409 for the lens, or a 404 for a table too narrow to be an expression
      matrix at all.

    Every list in the payload is uncut and says how many it holds
    (`GUIDED-209`), and every capability row carries the whole sentence rather
    than a key standing for one (`GUIDED-207`) — a page that had to translate
    `disabled_because: "count_model"` into prose would be holding a second copy
    of the research.
    """
    from turbotab import packs as _packs

    project = _project(project_id)
    if _packs.GENOMICS not in (project.lens or []):
        raise HTTPException(
            409,
            "What a number in a matrix IS — a count, a CPM, a variance-"
            "stabilized value — is a claim about the assay that produced it, "
            "and this project's lens does not say the measurements are "
            "genomic. The app does not infer the field from column names: "
            "answer the lens question and this reading applies.")
    card = _packs.data_type_card(project.working_table)
    if card is None:
        raise HTTPException(
            404,
            "This table is not wide enough to be an expression matrix. The "
            "reading needs a block of measurement columns to read; with fewer "
            "than that there is nothing here the research's nine signatures "
            "describe.")
    return card


@app.get("/project/{project_id}/evidence/missingness")
async def evidence_missingness(project_id: str) -> Dict[str, Any]:
    """Dtype-routed missingness decisions, each naming its own column.

    The action-timing ruling is carried on every option: structural repairs run
    now, statistical transforms are recorded and fitted inside the per-model
    pipeline on training folds. Stated as methods prose in the decision
    sentence, never as a note about the software (GUIDED-002).

    Every option also says whether clause §07 would refuse it, and the ones it
    can refuse are ordered after the ones it cannot (`GUIDED-163`). The
    mechanism the app already has on the record travels with the request, so a
    column whose question has been answered gets the definite sentence rather
    than the conditional one — and a column that has not been asked gets
    `None`, which is what `mechanism` has always meant here.
    """
    project = _project(project_id)
    return {"cards": engine.missingness(
        project.working_table,
        mechanisms={str(d["column"]): d["mechanism"]
                    for d in (project.missingness or [])
                    if d.get("mechanism")},
        provenance=project.blank_provenance())}


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


def offer_caption(defers: bool) -> str:
    """What an offer's preview panel says a press would do. `GUIDED-162`.

    It used to say *"this is what pressing apply would do"* for a non-deferring
    offer, **and there is no apply to press.** `data-offer-key` appears exactly
    twice in the page — emitted on the button, read to build the preview URL —
    and the `apply` decision is keyed to a structural finding's `fix_kind`, a
    different mechanism entirely, on a branch `openPanel` reaches only when
    `fix_kind` is absent or `"none"`. The caption named a control that cannot
    exist by construction.

    **The caption was wrong, not the panel.** The preview is real and computed,
    and the shelf is never shortened — so the affordance and its content stay,
    and the sentence stops promising a button and starts naming the exit that
    does exist. Both branches now have the same shape: what this would do, and
    where you take it.

    Its own function because a caption asserted in one place and tested in
    another is two strings, and the first version of the test read this module's
    SOURCE and matched the comment explaining the old wording — which is
    `LOOP.md` trap #5 in miniature, a grep answering the wrong question.
    """
    return ("preview, not applied — this one is fitted inside each training fold"
            if defers else
            "preview, not applied — earmark it to record it")


@app.get("/project/{project_id}/finding/{finding_id}/offers")
async def finding_offers(project_id: str, finding_id: str) -> Dict[str, Any]:
    """What this finding offers to DO — options and earmarks (`GUIDED-031`).

    The endpoint behind the branch that used to print `suggested_actions` as
    em-dash paragraphs and stop. Every entry is classified: an OPTION is
    something the app does and previews, an EARMARK goes to a person or to the
    step that owns the decision.

    A `GET`, because it is a question. Nothing is recorded by asking.
    """
    from turbotab import actions as _actions
    project = _project(project_id)
    try:
        finding = project.finding(finding_id)
    except ProjectError as exc:
        raise HTTPException(404, str(exc)) from exc
    return {"finding_id": finding_id, "title": finding.get("title"),
            **_actions.offers(finding, project.df, project.target)}


@app.get("/project/{project_id}/finding/{finding_id}/offer/{option}/preview")
async def offer_preview(project_id: str, finding_id: str,
                        option: str) -> Dict[str, Any]:
    """What one option would do to the data, without doing it.

    A deferred option still previews — clause §06 permits exactly one override,
    *a read-only preview not persisted to the modeling table, labeled preview,
    not applied* — and `features.preview` already computes those on TRAINING
    ROWS ONLY, so a preview cannot show a picture of the held-out data.
    """
    from turbotab import actions as _actions, features as _feat
    project = _project(project_id)
    try:
        finding = project.finding(finding_id)
    except ProjectError as exc:
        raise HTTPException(404, str(exc)) from exc

    found = next((o for o in _actions.offers(finding, project.df,
                                             project.target)["options"]
                  if o["key"] == option), None)
    if found is None:
        raise HTTPException(
            404, f"'{option}' is not one of the options this finding offers.")
    if not found["columns"]:
        raise HTTPException(
            400, "This finding does not name the columns the option would act "
                 "on, so there is nothing to preview against.")

    body = {k: found[k] for k in ("key", "label", "sentence", "because",
                                  "defers", "columns", "catalogue")}
    body["applied"] = False
    body["label_note"] = offer_caption(bool(found["defers"]))

    # A FEATURE binding previews through the catalogue that owns it, which
    # already draws the before/after and already refuses to fit a deferred
    # transform on anything but training rows.
    if found["catalogue"] == _actions.FEATURE:
        try:
            body["preview"] = _feat.preview(project.df, found["binding"],
                                            found["columns"][:1])
        except _feat.FeatureRefusal as exc:
            raise HTTPException(400, str(exc)) from exc
        return body

    # Everything else is a distribution the operation would learn. The honest
    # preview is the distribution itself and what the operation would move —
    # a before/after of two numbers, computed on a copy and thrown away.
    body["preview"] = engine.offer_simulation(
        project.working_table, found["columns"], found["binding"],
        found.get("variant"))
    return body


@app.get("/project/{project_id}/gaps")
async def get_gaps(project_id: str) -> Dict[str, Any]:
    """The `[AUTHOR REQUIRED]` gaps this record carries, and where each sits.

    The escape hatch's other half. Recording *"my design isn't described here"*
    and then writing a methods section that describes it anyway would be the
    governing rule broken by the mechanism built to honor it — so the gap is a
    first-class thing the export reads, placed at the point the app cannot
    describe rather than appended at the end.
    """
    from turbotab import repeats as _rep
    project = _project(project_id)
    gaps: List[Dict[str, Any]] = []
    if (project.grain or {}).get("design_not_described"):
        gaps.append({"where": "study_design", "after": "participants",
                     "question": "state_grain", "text": _rep.DESIGN_GAP})
    if project.unit_of_analysis == _rep.UNIT_NOT_DESCRIBED:
        gaps.append({"where": "unit_of_analysis", "after": "study_design",
                     "question": "state_unit_of_analysis",
                     "text": _rep.UNIT_GAP})
    return {"gaps": gaps, "n": len(gaps),
            "marker": _rep.AUTHOR_REQUIRED,
            "note": ("Each gap sits at the point the app cannot describe. "
                     "Nothing here is generated prose about your design; the "
                     "app is naming what it does not know.")}


@app.get("/project/{project_id}/teaching/{question}")
async def get_teaching(project_id: str, question: str) -> Dict[str, Any]:
    """Layer 3 for one question — computed on this table, never prose.

    `DESIGN_LANGUAGE.md` §10's third layer, and the product owner's ruling on
    what it is: the preview mechanic pointed at interview questions rather than
    at repairs. Teaching means showing consequences, and consequences are
    computable — so nothing here parses natural language in either direction and
    nothing here explains a concept in general.

    A `GET`, because it is a question about a question. Nothing is recorded by
    opening it.
    """
    from turbotab import teaching as _teaching
    project = _project(project_id)
    found = _teaching.panel(question, project.df, project.grain, project.target)
    if found is None:
        raise HTTPException(
            404, f"No teaching panel for '{question}'. The three hardest "
                 f"questions carry one: {', '.join(_teaching.TAUGHT)}.")
    return found


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
    entries, ranked_on = project.model_shelf_ranked()
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
        # `GUIDED-088`/`GUIDED-092`: the rows the ORDER was computed on, taken
        # OUT OF THE PROFILE THE SHELF RANKED ON rather than re-derived beside
        # it. A count computed separately would keep saying "training rows"
        # after somebody reverted the mask — which is a served number that is
        # true about a computation nobody performed.
        "n_rows_seen": int(ranked_on.n_rows),
        "n_rows_withheld": int(len(project.df) - int(ranked_on.n_rows)),
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
    # `GUIDED-092`. The divergence measure decides whether a variant question is
    # PUT to the user, and it decides it by measuring how the columns would be
    # rescaled relative to one another in the fit. `select_models` states the
    # requirement in its own refusal — *the shape it reads must be the shape the
    # models will actually be fitted on* — and this read the whole table,
    # sealed rows included, so a question a user answers was raised or
    # suppressed partly by rows the models will never see.
    ranking_frame = project.training_rows
    # Scoped to THIS project's lens: `packs.load` never unloads, so the table
    # accumulates every pack any project in this process has selected.
    origins = _rec.allowed_origins(project.lens or [])
    resolved = project.resolved_recipes()
    suppressed = 0
    for rows in resolved.values():
        for row in rows:
            r = _rec.resolve(row["model"], row["operation"], origins=origins)
            raise_variant, div = _rec.worth_asking(ranking_frame, numeric, r)
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
        # The row count the variant questions were measured on, served so a
        # reader — and `test_every_ranking_is_computed_on_the_training_rows` —
        # can check it rather than take it on trust.
        "n_rows_seen": int(len(ranking_frame)),
        "n_rows_withheld": int(len(project.df) - len(ranking_frame)),
        "models": resolved,
        "operations": [{"key": o.key, "label": o.label,
                        "determinacy": o.determinacy, "scope": o.scope,
                        "because": o.because, "origin": o.origin,
                        "variants": list(o.variants),
                        "pushed_alternatives": [list(p)
                                                for p in o.pushed_alternatives]}
                       for o in _rec.operations()],
        # GUIDED-074. The reasoning `resolve` discards: every default that
        # matched each cell, ranked, with the winner marked. Served rather than
        # re-derived in the interface — the ranking rule and the tie-break are
        # `resolve`'s, and a second implementation of them in JavaScript is a
        # second thing to drift.
        "candidates": {
            f"{model_key}::{row['operation']}":
                _rec.candidates(model_key, row["operation"], origins=origins)
            for model_key, rows in resolved.items() for row in rows},
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
        # `GUIDED-198`. `ordinal_declared` needs an `order`, and the legitimate
        # values of an order are the chosen column's own distinct levels — so
        # the parameter has no renderable control until they are served. Every
        # column carries either its levels or the sentence saying why an order
        # cannot be stated over it; `features.column_levels` owns both, because
        # a page that decided which columns are orderable would be holding a
        # second copy of a rule that lives in the engine.
        "column_levels": feat_mod.column_levels(
            project.df, exclude=[project.target] if project.target else []),
        "engineered": project.engineered,
        "deferred_transforms": project.deferred_transforms,
        "selection": project.selection_spec,
        # The recorded pool names columns; one added or removed since makes it
        # describe a table that no longer exists (`GUIDED-094`).
        "selection_stale": project.stale_since(
            (project.selection_spec or {}).get("mark")),
        "settled": project.features_settled,
        "selection_methods": [
            {"key": k, "label": m.label,
             "explainability_cost": m.explainability_cost}
            for k, m in sel_mod.METHODS.items()],
        # `GUIDED-108`. Columns that name a row rather than describe one, left
        # out of the models WITH A RECEIPT. `None` where there are none — a
        # labeled empty region would read as a finding of nothing.
        "identifiers": _identifiers_mod().receipt(project),
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
async def selection_evidence(project_id: str, method: str = "") -> Dict[str, Any]:
    """What a selection CHOICE is shown beside — ranked on training rows only.

    Ranks, does not choose. Nothing is stored and the response is marked
    `preview_not_applied`, the same distinction clause §06 draws for a deferred
    transform's preview.

    `method` is the choice this is evidence FOR (`GUIDED-177`). The page sends
    the method sitting in its own dropdown; a project that has already recorded
    a spec falls back to that, so the preview cannot show a correlation ranking
    under a sentence reading *by mutual information*. Omitted on both, the
    table is a plain correlation and says so.
    """
    project = _project(project_id)
    if not project.target:
        raise HTTPException(400, "Ranking features needs the outcome first.")
    candidates = [str(c) for c in project.df.columns
                  if str(c) != project.target
                  and pd.api.types.is_numeric_dtype(project.df[c])]
    chosen = method or str((project.selection_spec or {}).get("method") or "")
    try:
        return sel_mod.evidence(project.df, project.target, candidates,
                                project.training_mask,
                                method=chosen or None,
                                task_type=project.task_type)
    except sel_mod.SelectionRefusal as exc:
        raise HTTPException(400, str(exc)) from exc


# ─────────────────────────────────────────────────────────────────────────────
# Training — the first number this door computes rather than reads
# ─────────────────────────────────────────────────────────────────────────────
#
# `PRODUCT_VISION.md` §05: nothing owning long work is the absence that caused
# the migration, and §04 requires anything over about a second to be an
# OBSERVABLE JOB — a name in plain language, progress, and a cancel that stops
# it. `turbotab/jobs.py` was built at L7 for exactly this and had a consumer in
# Classic and none here.
_QUEUE = jobs_mod.JobQueue(max_workers=2)
_RUNS: Dict[str, Dict[str, Any]] = {}


def _train_worker(ctx, project_id: str, model_keys: List[str]):
    from turbotab import training as _training

    project = STORE.get(project_id)
    run = _training.train(project, model_keys, ctx=ctx,
                          seed=int(ctx.rng.integers(0, 2**31 - 1)))
    _RUNS[project_id] = {"run": run}
    # Held ON THE PROJECT as well, because the figure layer asks the project
    # rather than the API: a figure that had to know where the web layer keeps
    # its runs would be a second place to look.
    project.training_run = run
    return run.to_dict()


@app.post("/project/{project_id}/train")
async def start_training(project_id: str,
                         body: Dict[str, Any]) -> Dict[str, Any]:
    """Submit a training run. Returns the JOB, not the answer.

    A `POST` that blocked until the models were fitted would be the spinner
    §04 forbids wearing an HTTP costume: the user could not see progress, could
    not cancel, and could not tell a slow fit from a hung one.
    """
    from turbotab import training as _training

    project = _project(project_id)
    keys = [str(k) for k in (body.get("models") or [])] or list(
        project.selected_models or [])
    try:
        # Refused HERE rather than inside the worker, so a request that cannot
        # produce a number fails as a refusal the caller can read instead of as
        # a job that goes away and comes back empty.
        _training.check(project, keys)
    except _training.TrainingRefusal as exc:
        raise HTTPException(400, str(exc)) from exc
    handle = _QUEUE.submit(
        f"Training {len(keys)} model(s) on the held-out split",
        _train_worker, project_id, keys)
    return handle.to_dict()


@app.get("/job/{job_id}")
async def get_job(job_id: str) -> Dict[str, Any]:
    try:
        handle = _QUEUE.get(job_id)
    except KeyError as exc:
        raise HTTPException(404, f"no job {job_id!r}") from exc
    out = handle.to_dict()
    out["result"] = handle.result if handle.status is jobs_mod.JobStatus.DONE \
        else None
    return out


@app.post("/job/{job_id}/cancel")
async def cancel_job(job_id: str) -> Dict[str, Any]:
    """`T0-LIVE-002`: the Classic cancel sets a flag nothing reads. This one
    sets the token the worker checks between models, and the queue reports a
    job that ignored it as `finished` rather than claiming a stop it did not
    make."""
    try:
        return _QUEUE.cancel(job_id).to_dict()
    except KeyError as exc:
        raise HTTPException(404, f"no job {job_id!r}") from exc


@app.get("/project/{project_id}/explain")
async def get_explain(project_id: str, model: Optional[str] = None) -> Dict[str, Any]:
    """Why the model predicts what it predicts, on the held-out rows.

    **The step that has no research behind it, scoped to what this repository
    can defend** (`GUIDED-101`). Permutation importance from
    `sklearn.inspection`, computed on the sealed rows — the choice with a
    leakage consequence — with the interpretation prose read out of
    `ml/plot_narrative.py` rather than written here.

    `blocked_by` says WHICH of the states applies rather than returning an
    empty object, the same rule the training and prevalence surfaces follow: a
    step that has not happened and a step that produced nothing are different
    sentences.
    """
    from turbotab import explain as _explain, training as _training

    project = _project(project_id)
    run = getattr(project, "training_run", None)
    fitted = [r for r in (run.results if run else []) if r.metrics]
    if not fitted:
        return {"run": None, "shap": _explain.unavailable(),
                "costly_decisions": _explain.costly_decisions(project),
                "blocked_by": (
                    "No model has been fitted yet. Permutation importance is a "
                    "drop in a metric when a column is shuffled, so there has "
                    "to be a metric first — choose models in Train."
                    if run is None else
                    "Every model in the last run reported a reason instead of "
                    "a score, so there is no metric to permute against.")}

    chosen = model or fitted[0].key
    if chosen not in {r.key for r in fitted}:
        raise HTTPException(
            400, f"{chosen!r} did not produce a score in the last run, so "
                 f"there is no metric for shuffling a column to move.")
    try:
        payload = _explain.importance(project, chosen)
    except _explain.ExplainRefusal as exc:
        return {"run": None, "shap": _explain.unavailable(),
                "costly_decisions": _explain.costly_decisions(project),
                "blocked_by": str(exc)}
    stale = project.stale_since(getattr(run, "mark", None))
    return {
        "run": payload,
        "models": [{"key": r.key, "name": r.name} for r in fitted],
        # THE RANKING IS RECOMPUTED FROM THE CURRENT RECORD on every request,
        # so it is never stale itself — and the metrics it sits beside are the
        # stored run's, which can be. Presenting a fresh ranking next to a
        # stale accuracy without saying so would be two numbers about two
        # different analyses (`GUIDED-094`). Said, rather than resolved by
        # refusing: principle 4 is visible and recoverable, not blocked.
        "stale": stale,
        "recomputed_note": (
            "This ranking was computed just now, from the analysis as it "
            "stands. The held-out scores above it are from the last training "
            "run, which does not account for what changed since."
            if stale else None),
        # THE PROMISE THE REGISTER ALREADY MADE. Every transform carries
        # `explainability_cost` and `FEATURE_REGISTER.md`'s `prep-pca` row
        # states the consequence in words; until now nothing delivered it.
        "costly_decisions": _explain.costly_decisions(project),
        "shap": _explain.unavailable(),
        "blocked_by": None,
    }


@app.get("/project/{project_id}/training")
async def get_training(project_id: str) -> Dict[str, Any]:
    """The last run, or what is missing before there can be one."""
    project = _project(project_id)
    from turbotab import training as _training

    held = _RUNS.get(project_id)
    if held:
        # WHAT CHANGED SINCE THESE NUMBERS WERE PRODUCED (`GUIDED-094`). The
        # project has recorded it all along and nothing read it, so a held-out
        # accuracy stood unchanged over a table that had moved beneath it —
        # `PRODUCT_VISION.md` principle 4's *visible, veiled, recoverable*
        # failing at the visible step. Never a recompute and never a clear: the
        # run stays, and it says what it no longer accounts for.
        run = held["run"]
        return {"run": run.to_dict(), "blocked_by": None,
                "stale": project.stale_since(getattr(run, "mark", None))}
    try:
        _training.check(project, list(project.selected_models or []))
    except _training.TrainingRefusal as exc:
        # WHAT IT NEEDS, not an empty object. The same rule the prevalence
        # surface follows: a step that has not happened and a step that
        # produced nothing are different sentences.
        return {"run": None, "blocked_by": str(exc), "stale": []}
    return {"run": None, "blocked_by": None, "stale": []}


def _instability_worker(ctx, project_id: str, model_key: str, b: int):
    from turbotab import instability as _inst

    project = STORE.get(project_id)
    result = _inst.run(project, model_key, b=b, ctx=ctx,
                       seed=int(ctx.rng.integers(0, 2 ** 31 - 1)))
    # HELD ON THE PROJECT, same reason the training run is: the figure layer
    # asks the project rather than the API, so a figure that had to know where
    # the web layer keeps its results would be a second place to look.
    runs = getattr(project, "instability_runs", None)
    if runs is None:
        runs = {}
        project.instability_runs = runs
    runs[model_key] = result
    return {k: v for k, v in result.items() if k != "bootstrap"}


@app.post("/project/{project_id}/instability")
async def start_instability(project_id: str,
                            body: Dict[str, Any]) -> Dict[str, Any]:
    """Submit a resampling run. Returns the JOB, not the answer.

    `PRODUCT_VISION.md` §04: anything over about a second is an observable job
    with a name in plain language, progress and a cancel. B refits of a full
    pipeline is the longest thing this app does, and it is the one place where
    a spinner would be least forgivable — a researcher cannot tell a slow
    bootstrap from a hung one by looking.
    """
    from turbotab import instability as _inst

    project = _project(project_id)
    model_key = str(body.get("model") or "")
    b = int(body.get("b") or _inst.B_RESAMPLES)
    if not model_key:
        raise HTTPException(400, "Name the model to resample.")
    handle = _QUEUE.submit(
        f"Refitting the whole pipeline in {b:,} bootstrap resamples",
        _instability_worker, project_id, model_key, b)
    return handle.to_dict()


@app.get("/project/{project_id}/manuscript")
async def get_manuscript(project_id: str) -> Dict[str, Any]:
    """`GUIDED-107`. The draft as data, rendered, and CHECKED.

    The validation report is served beside the document rather than instead of
    it: the ruling is that the author gets the document they asked for, and a
    separate honest list of what a reviewer will notice.
    """
    from turbotab import manuscript as _ms

    project = _project(project_id)
    held = _RUNS.get(project_id)
    # `GUIDED-131`. THE WHOLE REGISTRY, and now also what this project can
    # actually draw. The list stays registry-wide because `promoted_figures`
    # can name any figure and a row missing from here is a promotion the
    # document cannot see. What was missing is the second half: this route
    # built the list and **never read `figures.bundle`**, so the companion rule
    # — stated as admissibility, enforced on `/figures` — had no consumer at
    # the boundary it was written for, which is the artifact that leaves the
    # building.
    #
    # `drawn` is what separates *promote the companion too* from *this project
    # cannot draw it at all*, and only the bundle knows which.
    drawn = _drawable(project)
    promoted_ids = set(getattr(project, "promoted_figures", []) or [])
    figures = [
        {"id": f.id, "title": f.title, "tier": f.tier,
         "promoted": f.id in promoted_ids,
         "drawn": None if drawn is None else (f.id in drawn)}
        for f in _figure_specs_all()
    ]
    # EVERYTHING THE APP HOLDS, not just the run. The first version passed the
    # project and the run and dropped the importance ranking, the sensitivity
    # fork and the resampling results — three analyses the app had already
    # done, absent from the document that leaves the building.
    from turbotab import explain as _explain
    from turbotab import instability as _inst

    extra: Dict[str, Any] = {}
    try:
        extra["explain"] = {"run": _explain.importance(
            project, (project.selected_models or [None])[0])} \
            if project.selected_models else None
    except Exception:
        extra["explain"] = None
    try:
        extra["sensitivity"] = await get_sensitivity(project_id)
    except Exception:
        extra["sensitivity"] = None
    runs = getattr(project, "instability_runs", None) or {}
    if runs:
        try:
            extra["instability"] = await get_instability(project_id)
        except Exception:
            extra["instability"] = None

    built = _ms.table_one(project)
    # `GUIDED-123`. The nutrition checklist, on a dietary project. The lens is
    # a RECORDED answer, so this is not an inference from column names — a
    # table of numbers does not know it is food.
    strobe = None
    from turbotab.packs import DIETARY as _DIETARY
    if _DIETARY in (project.lens or []):
        from turbotab import strobe_nut as _sn
        strobe = _sn.checklist(project)
    return _ms.validate(project.to_dict(),
                        table1=built[0] if built else None,
                        strobe_nut=strobe,
                        run=held["run"].to_dict() if held else None,
                        figures=figures,
                        explain=extra.get("explain"),
                        sensitivity=(extra.get("sensitivity") or {}).get("result"),
                        instability=extra.get("instability"))


def _figure_specs_all():
    """Every registered figure. Imported for the side effect, which is how the
    registry is populated — a module that lists them here instead would be a
    second list that goes stale."""
    from turbotab import figures as _figs
    import turbotab.figure_specs                            # noqa: F401 — registers
    return list(_figs.REGISTRY.values())


def _drawable(project) -> Optional[set]:
    """Which figures this project can draw at all — `admitted` plus `held`.

    **`held` counts as drawn**, and the distinction is the point. A figure the
    bundle held still has a payload, a caption and a checklist; what it lacks is
    its companion. Treating it as undrawable here would tell the author *this
    project cannot produce it* when the truth is *its companion is missing* —
    which is the same sentence the cross-section is trying to make, arriving one
    level down and wrong.

    `None` rather than an empty set when the bundle cannot be built, because
    empty is a claim (*nothing is drawable*) and this is an absence of one. The
    cross-section reads `None` as "unknown" and drops the extra clause instead
    of asserting the stronger sentence from a failure.
    """
    try:
        from turbotab import figure_bundle as _fb
        bundle = _fb.render(project)
    except Exception:
        from turbotab import devchecks
        import sys
        devchecks.swallowed(
            "api.get_manuscript::_drawable", sys.exc_info()[1] or Exception(),
            "the manuscript's companion cross-section cannot say whether a "
            "missing companion was drawable, so it reports the gap without "
            "the clause that would tell the author what to do about it")
        return None
    return {row["id"] for row in bundle["admitted"] + bundle["held"]}


@app.get("/project/{project_id}/instability")
async def get_instability(project_id: str) -> Dict[str, Any]:
    """The resampling results held for this project, and what they say.

    The bootstrap MATRIX is not served — B × n floats is megabytes and the page
    draws from the figure payload rather than from the raw draws. What is
    served is the figure's own payload plus the two sentences the run supports.
    """
    from turbotab import instability as _inst
    from turbotab import figure_specs as _specs

    project = _project(project_id)
    runs = getattr(project, "instability_runs", None) or {}
    if not runs:
        return {"runs": {}, "blocked_by": (
            "No resampling has been run yet. It refits the entire pipeline "
            f"{_inst.B_RESAMPLES:,} times, so it is a job you start rather "
            "than something computed on the way past.")}
    out: Dict[str, Any] = {"runs": {}, "blocked_by": None,
                           "b_default": _inst.B_RESAMPLES,
                           "b_recommended": _inst.RECOMMENDED_B}
    for key, result in runs.items():
        payload = _specs.prediction_instability_payload(result)
        entry = {
            "prediction_instability": payload,
            "prediction_caption": _specs.PREDICTION_INSTABILITY.caption(payload),
            "selection": _inst.selection_moved(result),
            "spread": {k: v for k, v in _inst.spread(result).items()
                       if k != "per_row"},
        }
        if result.get("task_type") == "classification":
            rows = project.training_rows
            rows = rows[rows[str(project.target)].notna()]
            positive = sorted(rows[str(project.target)].dropna().unique())[-1]
            y = (rows[str(project.target)] == positive).astype(float)
            calib = _specs.calibration_instability_payload(result, y)
            entry["calibration_instability"] = calib
            entry["calibration_caption"] = \
                _specs.CALIBRATION_INSTABILITY.caption(calib)
            # L40-C2. The two §A4.8 named and L38 deferred because the decision
            # curve did not exist. It does now, so all four instability plots
            # come from one resampling run rather than from two mechanisms.
            classification = _specs.classification_instability_payload(result)
            entry["classification_instability"] = classification
            entry["classification_caption"] = \
                _specs.CLASSIFICATION_INSTABILITY.caption(classification)
            dca = _specs.decision_curve_instability_payload(result, y)
            entry["decision_curve_instability"] = dca
            entry["decision_curve_caption"] = \
                _specs.DECISION_CURVE_INSTABILITY.caption(dca)
        out["runs"][key] = entry
    return out


@app.get("/project/{project_id}/sensitivity")
async def get_sensitivity(project_id: str) -> Dict[str, Any]:
    """`MISC-014`. The recorded plan run the other way, over one axis.

    A GET rather than a job because the fork is bounded by the models that were
    already chosen and it fits each of them twice — expensive, but not the
    open-ended work `/train` exists to make watchable. If that stops being true
    the answer is the job queue, not a longer timeout.

    Three distinct empty answers, and they are three different sentences for
    the same reason the training route has three: *nothing was recorded to
    fork*, *the arms were not comparable*, and *there is no run to compare
    against* are not the same fact, and a client that could not tell them apart
    would render one of them as the others.
    """
    project = _project(project_id)
    from turbotab import sensitivity as _sens

    spec = _sens.fork(project)
    if spec is None:
        return {"available": False, "result": None, "because": (
            "No missing-value decision on this project has an alternative that "
            "can be fitted the same way, so there is nothing to run both "
            "ways.")}

    models = list(project.selected_models or [])
    if not models:
        return {"available": False, "result": None, "fork": spec, "because": (
            "No models have been chosen yet, so there is no result to check "
            "against the other handling.")}

    result = _sens.run(project, models)
    if result is None or result.get("unavailable"):
        return {"available": False, "result": None, "fork": spec,
                "because": (result or {}).get("unavailable") or (
                    "The comparison could not be made.")}
    return {"available": True, "result": result,
            "methods_sentence": _sens.methods_sentence(result)}


@app.get("/capabilities")
async def get_capabilities() -> Dict[str, Any]:
    """Which pull affordances are wired, and what the unwired ones say.

    Served rather than hard-coded in the page so the interface cannot claim a
    capability the server does not have. **Dataset-independent** — whether the
    Guided door has been wired to an analysis at all. Whether it can run on
    THIS table is `/project/{id}/capabilities`.
    """
    return {"pulls": PULL_CAPABILITIES, "not_built_reason": NOT_BUILT_REASON}


#: Pull affordances whose availability is a property of THIS table rather than
#: of the build. Keyed to the gate in `ml.card_evidence`, which is where
#: `GUIDED-005` put `MAX_FEATURES_FOR_GALLERY` so the page and the server could
#: not disagree about it.
_PER_PROJECT_GATES = {
    "histogram_pager": "gallery_availability",
    "look::r8_collinearity": "matrix_availability",
}


@app.get("/project/{project_id}/capabilities")
async def get_project_capabilities(project_id: str) -> Dict[str, Any]:
    """Which pull affordances run **on this table**, and why not where not.

    `GUIDED-084`, and the ruling is worth restating because it is not a new
    decision. `/capabilities` exists so the interface cannot claim a capability
    the server does not have; the page then computed `built` itself from
    `P.profile.n_numeric` and wrote its own not-built sentences beside it. The
    gate was never the page's argument — `GUIDED-005` put the constant in the
    engine *precisely so the page and the server cannot disagree about it* —
    so there is nothing here for the server to learn. It held the constant and
    simply was not applying it per project.

    Two reasons this matters, and the second is the one that is easy to miss.
    A page-composed verdict can drift from the endpoint that serves the
    analysis, so a live-looking chip can open onto a refusal. And **a sentence
    a user reads that no server composed cannot be reviewed in
    `COPY_DECK.md`**, which is how copy gets reviewed without running the app.
    """
    return _project_capabilities(_project(project_id))


def _project_capabilities(project: AnalysisProject) -> Dict[str, Any]:
    """The per-project capability table. One composer, two readers.

    Read by `/project/{id}/capabilities` and by the interview's pull loop, so
    a chip the Router offers and the same chip in the capability table cannot
    give different answers — which is the failure this whole row is about, one
    surface along.
    """
    frame = project.working_table
    n_numeric = int(sum(1 for c in frame.columns
                        if pd.api.types.is_numeric_dtype(frame[c])
                        and str(c) != (project.target or "")))
    from ml import card_evidence as _card_evidence
    gates = {"gallery_availability": _card_evidence.gallery_availability,
             "matrix_availability": _card_evidence.matrix_availability}

    pulls: Dict[str, Any] = {}
    for key, cap in PULL_CAPABILITIES.items():
        entry = dict(cap)
        # `GUIDED-136`. The one gate that is about the TABLE rather than about a
        # feature count: reverse-coding is a property of an instrument, and a
        # table with no block of items sharing a response scale has no scale to
        # audit. Dark with the reason on the chip rather than absent, per
        # `GUIDED-006` — a control that silently does nothing is worse than one
        # that says so, and so is a control that is simply not there.
        if key == "look::reverse_coding":
            from turbotab import survey as _survey
            because = _survey.unavailable_because(frame)
            entry["built"] = not because
            entry["not_built_reason"] = because or None
            pulls[key] = entry
            continue
        gate_name = _PER_PROJECT_GATES.get(key)
        if gate_name and entry.get("built"):
            gate = gates[gate_name](n_numeric)
            entry["built"] = bool(gate["available"])
            entry["not_built_reason"] = gate["reason"]
            entry["limit"] = gate["limit"]
            entry["n_features"] = gate["n_features"]
        elif not entry.get("built"):
            entry["not_built_reason"] = NOT_BUILT_REASON
        else:
            entry["not_built_reason"] = None
        pulls[key] = entry
    return {"pulls": pulls, "not_built_reason": NOT_BUILT_REASON,
            "n_numeric": n_numeric}


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
        elif d.kind == "set_orientation":
            answered.append("state_orientation")
        elif d.kind == "set_purpose":
            answered.append("state_purpose")
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
        elif d.kind in ("apply_bulk", "decline_bulk"):
            # The group is answered, and so is every member — a member left out
            # of the selection was left out ON PURPOSE, and re-asking it on its
            # own would turn a recorded decision back into an open question.
            answered.append(f"repair_bulk::{d.subject}")
            for finding_id in ((d.payload.get("findings") or [])
                               + (d.payload.get("declined") or [])):
                answered.append(f"repair::{finding_id}")
        elif d.kind == "acknowledge_blocker":
            # A terminal state is guaranteed (DESIGN_LANGUAGE §09). A blocker
            # never re-fires on the same facts after acknowledgment: a flag that
            # cannot be satisfied teaches contempt for all flags. The
            # acknowledgment stays in the record and surfaces afterwards as its
            # own --stop-flagged artifact — never green, never gone.
            answered.append(d.subject)
        elif d.kind == "resolve_blocker":
            answered.append(d.subject)
        elif d.kind == "earmark":
            # An earmark resurfaces where it was sent, exactly as a deferral
            # does — the two are the same disposition applied to different
            # objects, so they share the mechanism rather than growing a second
            # one beside it.
            deferred[f"earmark::{d.payload.get('key') or d.subject}"] = \
                d.payload.get("target_step", "explore")
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

    # Question 1.5's evidence, resolved here for the same reason as everything
    # else on this list: `ml/router.py` takes no dataframe. `None` unless BOTH
    # conditions hold — an assay lens and a feature-major shape — so the
    # question is absent from the plan on every table it does not describe,
    # which is guard #2 expressed as a precondition rather than as restraint.
    orientation_state = None
    if project.lens and project.orientation is None:
        from turbotab import orientation as _orient
        reading = _orient.read(project.df)
        if _orient.fires(project.lens, reading):
            orientation_state = reading

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
                                lens_block=lens_block,
                                orientation=orientation_state,
                                repeats=repeats_state,
                                missingness_priors=missingness_priors,
                                missingness_groups=missingness_groups,
                                missingness_exceptions=missingness_exceptions,
                                missingness_settled=missingness_settled,
                                # Folded out of the record like `answered` and
                                # `deferred` beside it, so the three cannot
                                # drift (`GUIDED-041`).
                                unskipped=project.unskipped())
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
    # PER PROJECT, not per build (`GUIDED-084`). A correlation chip on a
    # 400-column table is not built *for this table*, and the reason belongs on
    # the chip before it is pressed rather than inside the refusal it opens.
    capabilities = _project_capabilities(project)["pulls"]
    rendered = []
    for q in questions:
        d = q.to_dict()
        if d["mode"] == "pull":
            cap = capabilities.get(d["key"])
            d["built"] = bool(cap and cap.get("built"))
            d["endpoint"] = (cap or {}).get("endpoint")
            d["not_built_reason"] = (
                None if d["built"]
                else (cap or {}).get("not_built_reason") or NOT_BUILT_REASON)
            # `GUIDED-168`. THE CHIP IS TITLED BY THIS DOOR'S OWN CAPABILITY
            # TABLE, not by the core recommendation it was built from.
            #
            # `ml/router.py:399` titles a palette entry `get("title") or rid`,
            # reading `ml.eda_recommender`'s card — and the core's title
            # describes the CORE's analysis. Measured on `clinic_visits.csv`:
            # three of the five built chips carried a title the capability
            # table disagreed with, and one of the three, *Missingness Pattern
            # Analysis*, names an analysis with an MCAR/MAR/MNAR deliverable
            # over an endpoint that returns per-column blank rates.
            #
            # The same correction `GUIDED-084` made one surface along: the
            # server already serves an accurate account of each affordance, so
            # nothing downstream should compose its own. The core keeps its
            # title for the Classic door; `core_title` keeps the borrowed one
            # on the record rather than dropping it.
            if cap and cap.get("title") and cap["title"] != d.get("title"):
                d["core_title"] = d.get("title")
                d["title"] = cap["title"]
            instead = (cap or {}).get("instead_of")
            if instead and instead.get("sentence"):
                # ON THE CHIP, not only in the payload. `why` is what the page
                # renders as the chip's `data-tip`; a difference recorded in a
                # field nothing draws is trap #6, which is how the label got
                # away with it for as long as it did.
                d["why"] = ((d.get("why") or "").rstrip(". ") + ". "
                            + instead["sentence"]).strip()
                d["instead_of"] = instead
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
