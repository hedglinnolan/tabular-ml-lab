"""The HTTP contract, one route family at a time (Tier B, BLUEPRINT §8)."""
from __future__ import annotations

import json

import anyio
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from turbotab.core import decisions
from turbotab.core.config import Settings
from turbotab.server import openapi, schemas
from turbotab.server.app import create_app
from turbotab.server.service import DecisionContext
from turbotab.server.tests.conftest import DIETARY, SAMPLES, open_by_path, wait_for


def decide(client: TestClient, pid: str, decision: dict):
    return client.post(f"/api/projects/{pid}/decisions", json=decision)


# ── system ───────────────────────────────────────────────────────────────────


def test_health_says_which_mode_and_how_many_workers(client):
    body = client.get("/api/health").json()
    assert schemas.Health.model_validate(body)
    assert body["mode"] == "local" and body["workers"] == 2


def test_local_mode_lists_a_folder(client):
    body = client.get("/api/fs/list", params={"path": str(SAMPLES)}).json()
    listing = schemas.FsListing.model_validate(body)
    assert listing.path == str(SAMPLES) and listing.parent == str(SAMPLES.parent)
    entry = next(e for e in listing.entries if e.name == DIETARY.name)
    assert entry.path == str(DIETARY) and not entry.is_dir and entry.size == DIETARY.stat().st_size
    assert client.get("/api/fs/list", params={"path": str(SAMPLES / "missing")}).status_code == 404


def test_the_host_guard_turns_away_other_hostnames(client):
    refused = client.get("/api/health", headers={"host": "evil.example"})
    assert refused.status_code == 403
    assert refused.json()["error"]["code"] == "host_not_allowed"
    assert client.get("/api/health", headers={"host": "localhost:8787"}).status_code == 200


def test_the_guard_refuses_changes_sent_by_a_page_on_another_site(client):
    """A form post is sent cross-site without a preflight, with Host 127.0.0.1."""
    before = len(client.get("/api/projects").json())
    evil = {"origin": "https://evil.example", "sec-fetch-site": "cross-site"}
    upload = client.post(
        "/api/projects/upload", files={"file": ("x.csv", b"a,b\n1,2\n", "text/csv")}, headers=evil
    )
    assert upload.status_code == 403 and upload.json()["error"]["code"] == "cross_site"
    assert len(client.get("/api/projects").json()) == before  # nothing was created

    cancel = "/api/projects/p0000000000/jobs/j0/cancel"
    for headers in (
        {"origin": "https://evil.example"},
        {"sec-fetch-site": "cross-site"},
        {"origin": "null"},  # a sandboxed frame or a file:// page
        {"origin": "http://127.0.0.1.evil.example:8787"},
    ):
        refused = client.post(cancel, headers=headers)
        assert (refused.status_code, refused.json()["error"]["code"]) == (403, "cross_site"), headers

    # The app's own pages, the Vite dev server, and clients that name no page get through.
    for headers in (
        {"origin": "http://127.0.0.1:8787", "sec-fetch-site": "same-origin"},
        {"origin": "http://localhost:5173"},
        {},
    ):
        assert client.post(cancel, headers=headers).json()["error"]["code"] == "unknown_project"
    # Reads are left alone: without CORS headers the page cannot see the answer.
    assert client.get("/api/health", headers=evil).status_code == 200


def test_server_mode_refuses_paths_and_browsing_but_takes_uploads(server_client):
    refused = server_client.post("/api/projects", json={"path": str(DIETARY)})
    assert refused.status_code == 403
    assert refused.json()["error"]["code"] == "local_only"
    assert refused.json()["error"]["exits"][0]["label"] == "Upload the file instead"
    assert server_client.get("/api/fs/list").status_code == 403

    with open(DIETARY, "rb") as fh:
        response = server_client.post("/api/projects/upload", files={"file": (DIETARY.name, fh, "text/csv")})
    assert response.status_code == 200, response.text
    summary = response.json()
    assert summary["source_kind"] == "upload" and summary["source_name"] == DIETARY.name
    view = wait_for(server_client, summary["id"], {"ingest": "fresh"})
    assert (view["summary"]["n_rows"], view["summary"]["n_cols"]) == (600, 17)

    unsupported = server_client.post("/api/projects/upload", files={"file": ("notes.docx", b"x", "text/plain")})
    assert unsupported.status_code == 400
    assert unsupported.json()["error"]["code"] == "unsupported_file"


def test_the_frontend_route_answers_without_a_build(client):
    page = client.get("/")
    assert page.status_code == 200 and "npm run build" in page.text
    missing = client.get("/api/no-such-route")
    assert missing.status_code == 404 and missing.json()["error"]["code"] == "not_found"


def test_a_built_frontend_is_served_with_a_single_page_fallback(tmp_path):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>app</title>")
    (dist / "assets" / "app.js").write_text("console.log(1)")
    app = create_app(Settings(home=tmp_path, mode="server", workers=1, memory_budget_bytes=1 << 30), frontend_dist=dist)
    plain = TestClient(app)  # no lifespan: the frontend needs no engine
    assert plain.get("/assets/app.js").text == "console.log(1)"
    assert "<title>app</title>" in plain.get("/projects/p0123456789").text
    assert "<title>app</title>" in plain.get("/../../etc/passwd").text


def test_the_committed_openapi_document_matches_the_app():
    committed = openapi.OPENAPI_JSON.read_text("utf-8")
    assert committed == openapi.render(), "run: venv/bin/python -m turbotab.server.openapi --write"


# ── projects and stages ──────────────────────────────────────────────────────


def test_a_file_opened_by_path_is_ingested_and_profiled(client, dietary):
    view = client.get(f"/api/projects/{dietary}").json()
    schemas.ProjectView.model_validate(view)
    summary = view["summary"]
    assert summary["source_kind"] == "path" and summary["source_name"] == DIETARY.name
    assert (summary["n_rows"], summary["n_cols"]) == (600, 17)
    assert dietary in [p["id"] for p in client.get("/api/projects").json()]

    ingest = client.get(f"/api/projects/{dietary}/stages/ingest").json()
    assert ingest["fresh"] and ingest["status"] == "fresh"
    info = schemas.DatasetInfo.model_validate(ingest["artifact"])
    assert info.n_rows == 600 and "hba1c" in [c.name for c in info.columns]

    profile = schemas.ProfileArtifact.model_validate(
        client.get(f"/api/projects/{dietary}/stages/profile").json()["artifact"]
    )
    assert len(profile.columns) == 17
    assert "dietary" in [h.lens for h in profile.lens_hints]
    assert "all 600 rows" in profile.basis

    assert client.get(f"/api/projects/{dietary}/stages/nope").status_code == 404
    assert client.get("/api/projects/p0000000000").status_code == 404
    assert client.get("/api/projects/..%2F..%2Fetc").status_code == 404


def test_path_creation_refuses_what_it_cannot_read(client):
    assert client.post("/api/projects", json={"path": "relative.csv"}).json()["error"]["code"] == "relative_path"
    assert client.post("/api/projects", json={"path": str(SAMPLES / "nope.csv")}).status_code == 404
    assert client.post("/api/projects", json={"path": str(SAMPLES)}).json()["error"]["code"] == "is_a_folder"
    assert client.post("/api/projects", json={"path": str(SAMPLES / "clinical_labs.md")}).status_code == 400


def test_decisions_are_refused_recorded_and_reverted(client):
    pid = open_by_path(client)
    wait_for(client, pid, {"ingest": "fresh"})

    refused = decide(client, pid, {"kind": "set_target", "column": "nope"})
    assert refused.status_code == 409
    assert refused.json()["error"]["code"] == "unknown_column"
    schemas.Refusal.model_validate(refused.json())
    assert decide(client, pid, {"kind": "set_lens", "lenses": []}).status_code == 422
    assert client.get(f"/api/projects/{pid}").json()["decisions"] == []  # refusals are not recorded

    target = decide(client, pid, {"kind": "set_target", "column": "hba1c"})
    assert target.status_code == 200 and target.json()["state"]["target"] == "hba1c"
    target_id = target.json()["decisions"][-1]["id"]
    wait_for(client, pid, {"target_info": "fresh"})
    info = schemas.TargetInfo.model_validate(client.get(f"/api/projects/{pid}/stages/target_info").json()["artifact"])
    assert (info.task, info.detected_task, info.confidence) == ("regression", "regression", "high")
    assert info.histogram is not None and sum(info.histogram.counts) == 600 and info.classes is None

    task = decide(client, pid, {"kind": "set_task", "column": "hba1c", "task": "binary"}).json()
    wait_for(client, pid, {"target_info": "fresh"})
    info = schemas.TargetInfo.model_validate(client.get(f"/api/projects/{pid}/stages/target_info").json()["artifact"])
    assert (info.task, info.detected_task) == ("binary", "regression")  # the answer overrides detection
    assert info.classes and info.histogram is None

    reverted = decide(client, pid, {"kind": "revert", "decision_id": task["decisions"][-1]["id"]}).json()
    assert reverted["state"]["task"] is None and reverted["state"]["target"] == "hba1c"
    unknown = decide(client, pid, {"kind": "revert", "decision_id": "no-such-decision"})
    assert unknown.status_code == 409 and unknown.json()["error"]["code"] == "unknown_decision"

    view = decide(client, pid, {"kind": "revert", "decision_id": target_id}).json()
    assert view["state"]["target"] is None
    assert view["stages"]["target_info"]["status"] == "blocked"
    assert view["stages"]["target_info"]["missing"] == ["target"]
    assert [d["seq"] for d in view["decisions"]] == [1, 2, 3, 4]


def test_a_task_answer_belongs_to_the_column_it_was_given_for(client):
    pid = open_by_path(client)
    wait_for(client, pid, {"ingest": "fresh"})
    assert decide(client, pid, {"kind": "set_target", "column": "sex"}).status_code == 200
    answered = decide(client, pid, {"kind": "set_task", "column": "sex", "task": "multiclass"})
    assert answered.json()["state"]["task"] == "multiclass"

    view = decide(client, pid, {"kind": "set_target", "column": "energy_kcal"}).json()
    assert (view["state"]["target"], view["state"]["task"]) == ("energy_kcal", None)
    wait_for(client, pid, {"target_info": "fresh"})
    info = schemas.TargetInfo.model_validate(client.get(f"/api/projects/{pid}/stages/target_info").json()["artifact"])
    assert (info.column, info.task, info.detected_task) == ("energy_kcal", "regression", "regression")
    assert info.histogram is not None and info.classes is None

    stale = decide(client, pid, {"kind": "set_task", "column": "sex", "task": "binary"})
    assert stale.status_code == 409 and stale.json()["error"]["code"] == "not_the_target"


def test_the_target_cannot_be_chosen_before_the_columns_are_known():
    ctx = DecisionContext(columns=None, ingest_status="running")
    with pytest.raises(decisions.Refusal) as refused:
        decisions.validate({"kind": "set_target", "column": "hba1c"}, ctx)
    assert refused.value.code == "table_not_ready"


def test_findings_follow_the_lens(client, dietary):
    blocked = client.get(f"/api/projects/{dietary}").json()["stages"]["findings"]
    assert blocked["status"] == "blocked" and blocked["missing"] == ["lens"]

    assert decide(client, dietary, {"kind": "set_lens", "lenses": ["dietary"]}).status_code == 200
    wait_for(client, dietary, {"findings": "fresh"})
    artifact = schemas.FindingsArtifact.model_validate(
        client.get(f"/api/projects/{dietary}/stages/findings").json()["artifact"]
    )
    assert artifact.findings and "dietary lens" in artifact.basis
    from_pack = [f for f in artifact.findings if f.source == "pack"]
    assert from_pack and all(f.lens == "dietary" and f.evidence for f in from_pack)
    assert {f.evidence.status for f in from_pack} <= {"SETTLED", "CONVENTION", "DISPUTED"}
    assert any(f.source == "structural" for f in artifact.findings)
    assert len({f.id for f in artifact.findings}) == len(artifact.findings)


# ── data reads ───────────────────────────────────────────────────────────────


def test_a_table_window_is_rows_in_file_order(client, dietary):
    body = client.get(
        f"/api/projects/{dietary}/table", params={"offset": 10, "limit": 5, "columns": "participant_id,hba1c"}
    ).json()
    window = schemas.TableWindow.model_validate(body)
    expected = pd.read_csv(DIETARY)[["participant_id", "hba1c"]].iloc[10:15].values.tolist()
    assert window.rows == expected
    assert (window.columns, window.total_rows, window.offset) == (["participant_id", "hba1c"], 600, 10)

    missing = client.get(f"/api/projects/{dietary}/table", params={"columns": "hba1c,nope"})
    assert missing.status_code == 404 and missing.json()["error"]["code"] == "unknown_column"


def test_column_summaries_and_histograms_are_queries(client, dietary):
    summaries = [schemas.ColumnSummary.model_validate(s) for s in client.get(f"/api/projects/{dietary}/columns").json()]
    assert len(summaries) == 17
    hba1c = next(s for s in summaries if s.name == "hba1c")
    assert hba1c.mean == pytest.approx(pd.read_csv(DIETARY)["hba1c"].mean())

    hist = schemas.Histogram.model_validate(
        client.get(f"/api/projects/{dietary}/columns/hba1c/histogram", params={"bins": 10}).json()
    )
    assert len(hist.edges) == 11 and sum(hist.counts) + hist.n_missing == 600
    text = client.get(f"/api/projects/{dietary}/columns/participant_id/histogram")
    assert text.status_code == 400 and text.json()["error"]["code"] == "not_numeric"


# ── events and jobs ──────────────────────────────────────────────────────────


def read_events(client: TestClient, pid: str, trigger, until, timeout: float = 30.0) -> list[tuple[str, dict]]:
    """Read the project's SSE stream until ``until(events)``, calling ``trigger()`` once subscribed.

    TestClient buffers whole responses, so an endless stream is driven as raw
    ASGI on the client's own event loop instead.
    """

    async def run() -> list[tuple[str, dict]]:
        events: list[tuple[str, dict]] = []
        buffer = b""
        subscribed, finished = anyio.Event(), anyio.Event()

        async def receive() -> dict:
            await finished.wait()
            return {"type": "http.disconnect"}

        async def send(message: dict) -> None:
            nonlocal buffer
            if message["type"] == "http.response.start":
                assert message["status"] == 200
            if message["type"] != "http.response.body":
                return
            buffer += message.get("body", b"")
            while b"\n\n" in buffer:
                block, buffer = buffer.split(b"\n\n", 1)
                fields = dict(line.split(": ", 1) for line in block.decode().splitlines())
                events.append((fields["event"], json.loads(fields["data"])))
                if fields["event"] == "resync":
                    subscribed.set()
                if until(events):
                    finished.set()

        path = f"/api/projects/{pid}/events"
        scope = {
            "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "GET",
            "scheme": "http", "path": path, "raw_path": path.encode(), "root_path": "",
            "query_string": b"", "headers": [(b"host", b"127.0.0.1")],
            "client": ("127.0.0.1", 50000), "server": ("127.0.0.1", 80),
        }
        async with anyio.create_task_group() as tg:
            tg.start_soon(client.app, scope, receive, send)
            with anyio.fail_after(timeout):
                await subscribed.wait()
                await anyio.to_thread.run_sync(trigger)
                await finished.wait()
            tg.cancel_scope.cancel()
        return events

    return client.portal.call(run)


def test_the_event_stream_reports_a_decision_and_what_it_set_off(client):
    pid = open_by_path(client)
    wait_for(client, pid, {"ingest": "fresh"})

    def done(events):
        kinds = {kind for kind, _ in events}
        return {"decision", "job"} <= kinds and any(
            kind == "stage" and data["stage"] == "findings" and data["status"] == "fresh" for kind, data in events
        )

    events = read_events(client, pid, lambda: decide(client, pid, {"kind": "set_lens", "lenses": ["dietary"]}), done)
    assert events[0] == ("resync", {})
    decision = next(data for kind, data in events if kind == "decision")
    assert decisions.DecisionRecord.model_validate(decision).decision.kind == "set_lens"
    findings = [data for kind, data in events if kind == "stage" and data["stage"] == "findings"]
    assert [s["status"] for s in findings][-1] == "fresh"

    job = next(data for kind, data in events if kind == "job" and data["stage"] == "findings")
    view = client.get(f"/api/projects/{pid}/jobs/{job['job_id']}").json()
    assert schemas.JobView.model_validate(view).stage == "findings"
    assert client.post(f"/api/projects/{pid}/jobs/{job['job_id']}/cancel").json()["state"] == "done"
    other = open_by_path(client)
    assert client.get(f"/api/projects/{other}/jobs/{job['job_id']}").status_code == 404


# ── the memory budget ────────────────────────────────────────────────────────


def test_over_the_memory_budget_hints_sample_and_findings_refuse(tmp_path):
    """dietary_recalls needs ~205 kB to materialize whole; this budget holds about 300 rows."""
    settings = Settings(home=tmp_path, mode="local", workers=1, memory_budget_bytes=150_000)
    with TestClient(create_app(settings), base_url="http://127.0.0.1") as small:
        pid = open_by_path(small)
        wait_for(small, pid, {"ingest": "fresh", "profile": "fresh"})
        basis = small.get(f"/api/projects/{pid}/stages/profile").json()["artifact"]["basis"]
        assert "all 600 rows" in basis and "fixed sample of 300 rows" in basis

        decide(small, pid, {"kind": "set_lens", "lenses": ["dietary"]})
        view = wait_for(small, pid, {"findings": "error"})
        assert "run TurboTab on a server" in view["stages"]["findings"]["error"]

        # Trying again runs it again (and fails the same way).
        again = small.post(f"/api/projects/{pid}/stages/findings/run")
        assert again.status_code == 200
        assert schemas.StageStatus.model_validate(again.json()).status in {"queued", "running"}
        wait_for(small, pid, {"findings": "error"})
        assert small.post(f"/api/projects/{pid}/stages/nope/run").status_code == 404


def test_a_file_that_cannot_be_read_says_why_and_can_be_tried_again(client, tmp_path):
    ragged = tmp_path / "ragged.csv"
    ragged.write_text('a,b\n1,2\n3\n"unterminated,4\n')
    pid = open_by_path(client, ragged)
    view = wait_for(client, pid, {"ingest": "error"})
    error = view["stages"]["ingest"]["error"]
    assert "unterminated quote" in error
    assert view["summary"]["ingest"]["status"] == "error" and view["summary"]["n_rows"] is None
    listed = next(p for p in client.get("/api/projects").json() if p["id"] == pid)
    assert listed["ingest"]["error"] == error

    refused = decide(client, pid, {"kind": "set_target", "column": "a"})
    assert refused.status_code == 409 and refused.json()["error"]["code"] == "table_unreadable"

    retried = client.post(f"/api/projects/{pid}/stages/ingest/run").json()
    assert retried["status"] in {"queued", "running"} and retried["error"] is None
    assert wait_for(client, pid, {"ingest": "error"})["stages"]["ingest"]["error"] == error
