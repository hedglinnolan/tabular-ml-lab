"""Analyst mode, a decision's note, and where each file came from, through the HTTP API
(SWARM_ENGINE_PLAN E4 and E5).

An agent driving the API in a person's place runs the server in analyst mode (``--analyst``, or
``TURBOTAB_ANALYST=1``). There a re-seal is refused with no exit offering one, and once the
held-out rows are opened no refusal offers to withdraw them and draw them again; each decision and
join preview is logged, and ``GET …/analyst-log`` serves the previews and the claims rendered. The
claims log is written in every mode (R1_PLAN R1-M reads it), by the stages in the job workers and
by the requests alike; for a person nothing else changes. In every mode a
decision takes an optional note (its reason, at most 2,000 characters), kept in the decision log and
never in the methods; an upload takes the file's URL, release date and license, kept on the
project's meta and echoed into the export's inputs; the ingest serves the SHA-256 of the bytes.

The references: the SHA-256 is hashlib's over the bytes uploaded; the note is the text posted; a
person's server is the shared ``client`` fixture, the analyst's is the same app with only
``analyst`` set.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from turbotab.core.config import Settings
from turbotab.server import schemas
from turbotab.server.app import create_app
from turbotab.server.tests.conftest import DIETARY, wait_for

NOTE = "Kept HbA1c as the outcome: the paper's primary endpoint, named in its abstract."
NHANES_GLU = {"url": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/GLU_J.xpt",
              "release_date": "2019-11-01",
              "license": "Public domain: a work of the U.S. Government (17 U.S.C. 105)"}


@pytest.fixture(scope="module")
def analyst(tmp_path_factory):
    """A local-mode server in analyst mode, two workers."""
    home = tmp_path_factory.mktemp("analyst")
    settings = Settings(home=home, mode="local", workers=2, memory_budget_bytes=2 << 30,
                        analyst=True)
    with TestClient(create_app(settings, frontend_dist=home / "no-frontend-here"),
                    base_url="http://127.0.0.1") as c:
        yield c


def upload(client: TestClient, data: bytes, name: str = DIETARY.name, **params: str):
    return client.post("/api/projects/upload", params=params,
                       files={"file": (name, data, "text/csv")})


def uploaded(client: TestClient, data: bytes | None = None, **params: str) -> str:
    response = upload(client, DIETARY.read_bytes() if data is None else data, **params)
    assert response.status_code == 200, response.text
    pid = response.json()["id"]
    wait_for(client, pid, {"ingest": "fresh"})
    return pid


def project_dir(client: TestClient, pid: str) -> Path:
    return client.app.state.service.workspace.project_dir(pid)


def decide(client: TestClient, pid: str, body: dict, **params: str):
    return client.post(f"/api/projects/{pid}/decisions", json=body, params=params)


# ── the switch ───────────────────────────────────────────────────────────────


def test_the_flag_and_the_environment_switch_analyst_mode_on_and_it_is_off_by_default(tmp_path):
    from turbotab.server.__main__ import build_parser, settings_for

    env = {"TURBOTAB_HOME": str(tmp_path)}
    parse = build_parser().parse_args
    assert settings_for(parse(["--analyst"]), env).analyst is True
    assert settings_for(parse([]), {**env, "TURBOTAB_ANALYST": "1"}).analyst is True
    assert settings_for(parse([]), env).analyst is False
    assert settings_for(parse(["--mode", "server"]), env).mode == "server"


def test_the_analyst_log_is_served_in_analyst_mode_and_refused_for_a_person(client, analyst):
    mine = uploaded(analyst)
    served = analyst.get(f"/api/projects/{mine}/analyst-log")
    assert served.status_code == 200, served.text
    log = schemas.AnalystLog.model_validate(served.json())
    assert log.previews == []

    theirs = uploaded(client)
    refused = client.get(f"/api/projects/{theirs}/analyst-log")
    assert refused.status_code == 403
    assert refused.json()["error"]["code"] == "analyst_mode_only"
    assert analyst.get("/api/projects/p0000000000/analyst-log").status_code == 404


# A built claim with no regime, which nothing shipped renders at run time (``designs`` fills it
# once, as it is imported, and an import's renders are never logged).
PLANTED = "designs.observational_default"


def claims_logged(client: TestClient, pid: str, id: str = PLANTED) -> list[dict]:
    """The project's claims render log's lines for ``id``, read from its file (either mode)."""
    path = client.app.state.service.workspace.claims_log_path(pid)
    if not path.exists():
        return []
    return [line for line in map(json.loads, path.read_text("utf-8").splitlines())
            if line["id"] == id]


def _ingest_as_a_job_worker(spec: dict, out: str) -> None:
    """A spawned process doing what a job worker does for a heavy stage (``graph.run_stage_job``,
    which loads the graph by the factory's name), its job's progress reports each rendering a
    shipped claim: a render made inside the stage's function as it reads the file."""
    from turbotab.core import claims, graph

    renders: list[str] = []

    class Job:
        def progress(self, fraction: float, message: str = "") -> None:
            renders.append(claims.card(PLANTED))

        def cancelled(self) -> bool:
            return False

    graph.current_job = lambda: Job()
    meta = graph.run_stage_job(spec)
    Path(out).write_text(json.dumps({"renders": len(renders), "stage": meta["stage"]}), "utf-8")


@pytest.mark.parametrize("mode", ["person", "analyst"])
def test_a_claim_rendered_inside_the_server_s_stage_in_a_job_worker_lands_in_its_log(
        mode, client, analyst, tmp_path):
    """The graph the server's engine hands its job workers, by its own name
    (``service.engine.graph_factory``), loaded in a spawned process as a worker loads it, runs the
    real ingest stage over the project's file inside that project's claims log: each claim
    rendered as the stage reports its progress is a line of the project's log (the count is the
    child's own), in either mode, and analyst mode serves them. The stage writes its table to a
    scratch path, so the project's own is untouched."""
    import multiprocessing

    from turbotab.core.graph import load_graph

    c = client if mode == "person" else analyst
    service = c.app.state.service
    pid = uploaded(c)
    factory = service.engine.graph_factory
    stage = load_graph(factory)["ingest"]
    assert stage.heavy  # the engine sends it to a job worker
    ctx = service._project_ctx(pid)
    (tmp_path / "data").mkdir()
    spec = {"graph_factory": factory, "stage": "ingest", "version": stage.version, "key": "probe",
            "dep_keys": {}, "cache_root": str(tmp_path / "cache"), "project_id": pid,
            "state": ctx.state, "paths": {**ctx.paths, "data": str(tmp_path / "data" / "raw.parquet")},
            "settings": dict(ctx.settings)}
    out = tmp_path / "child.json"
    child = multiprocessing.get_context("spawn").Process(target=_ingest_as_a_job_worker,
                                                         args=(spec, str(out)))
    child.start()
    child.join(120)
    assert child.exitcode == 0
    done = json.loads(out.read_text("utf-8"))
    assert done["stage"] == "ingest" and done["renders"] >= 1
    logged = claims_logged(c, pid)
    assert len(logged) == done["renders"]
    assert {(x["register"], x["status"], x["fallback"], x["refused"]) for x in logged} == {
        ("card", "built", None, None)}
    if mode == "analyst":
        served = [x for x in schemas.AnalystLog.model_validate(
            analyst.get(f"/api/projects/{pid}/analyst-log").json()).claims if x.id == PLANTED]
        assert len(served) == done["renders"]
        assert {(x.sentence_register, x.status, x.fallback, x.refused) for x in served} == {
            ("card", "built", None, None)}


@pytest.mark.parametrize("mode", ["person", "analyst"])
def test_a_claim_rendered_on_the_request_path_is_logged_beside_its_project(
        mode, client, analyst, monkeypatch):
    """A planted check on the lens answer renders a shipped claim each time it runs, and refuses
    in its words for the lens ``other``; a planted reading of the questions renders it as TurboTab
    looks for what it records itself. Each render made while a decision is recorded, refused or
    previewed, or a completion is looked for, is a line of that project's log, in either mode:
    the count is the planted check's own."""
    from turbotab.core import claims
    from turbotab.core import decisions as d

    c = client if mode == "person" else analyst
    service = c.app.state.service
    pid = uploaded(c)
    renders: list[str] = []

    def planted(decision, ctx) -> None:
        said = claims.card(PLANTED)
        renders.append(said)
        if decision.lenses == ["other"]:
            raise d.Refusal("planted", f"Planted: {said}")

    monkeypatch.setitem(d._VALIDATORS, "set_lens", [*d._VALIDATORS.get("set_lens", []), planted])
    assert claims_logged(c, pid) == []
    assert decide(c, pid, {"kind": "set_lens", "lenses": ["dietary"]}).status_code == 200
    assert len(renders) >= 1 and len(claims_logged(c, pid)) == len(renders)
    refused = decide(c, pid, {"kind": "set_lens", "lenses": ["other"]})
    assert refused.status_code == 409 and refused.json()["error"]["code"] == "planted"
    assert renders[-1] in refused.json()["error"]["message"]
    assert len(claims_logged(c, pid)) == len(renders)
    previewed = c.post(f"/api/projects/{pid}/preview", json={"kind": "set_lens", "lenses": ["clinical"]})
    assert previewed.status_code == 200, previewed.text[:300]
    assert len(claims_logged(c, pid)) == len(renders)

    # The completer: it reads the questions (``interview``) to find what is due.
    asked: list[str] = []
    real = service.interview

    def interview(project, *args, **kwargs):
        if project == pid:
            asked.append(claims.card(PLANTED))
        return real(project, *args, **kwargs)

    before = len(claims_logged(c, pid))
    with monkeypatch.context() as m:
        m.setattr(service, "interview", interview)
        service._complete(pid)
    assert len(asked) >= 1 and len(claims_logged(c, pid)) == before + len(asked)


# ── E4: no re-seal, and what the analyst saw ─────────────────────────────────


def test_a_reseal_is_refused_in_analyst_mode_and_meets_the_seal_s_own_checks_for_a_person(
        client, analyst):
    """The guard reads the server's setting: in analyst mode a re-seal is refused for that reason,
    with no exit offering one, and nothing is recorded; a person's re-seal on the same fresh
    project passes it and meets the seal's own checks (the Router's order, or nothing opened)."""
    mine = uploaded(analyst)
    refused = decide(analyst, mine, {"kind": "reseal"})
    assert refused.status_code == 409
    error = refused.json()["error"]
    assert error["code"] == "no_reseal_in_analyst_mode"
    assert all((e["decision"] or {}).get("kind") != "reseal" for e in error["exits"])
    # Nothing is drawn on a fresh project, and the one exit says none was.
    assert error["exits"] == [{"label": "Keep the held-out rows as they are", "decision": None}]
    assert analyst.get(f"/api/projects/{mine}").json()["decisions"] == []
    previewed = analyst.post(f"/api/projects/{mine}/preview", json={"kind": "reseal"})
    assert previewed.status_code == 409
    assert previewed.json()["error"]["code"] == "no_reseal_in_analyst_mode"

    theirs = uploaded(client)
    person = decide(client, theirs, {"kind": "reseal"})
    assert person.status_code == 409
    assert person.json()["error"]["code"] != "no_reseal_in_analyst_mode"


def test_each_preview_is_logged_in_analyst_mode_and_not_for_a_person(client, analyst):
    lens = {"kind": "set_lens", "lenses": ["dietary"]}
    mine = uploaded(analyst)
    started = time.time()
    assert analyst.post(f"/api/projects/{mine}/preview", json=lens).status_code == 200
    assert analyst.post(f"/api/projects/{mine}/preview", json={"kind": "reseal"}).status_code == 409
    previews = schemas.AnalystLog.model_validate(
        analyst.get(f"/api/projects/{mine}/analyst-log").json()).previews
    assert [(p.decision.kind, p.refused) for p in previews] == [
        ("set_lens", None), ("reseal", "no_reseal_in_analyst_mode")]
    assert previews[0].decision.model_dump(mode="json") == lens
    assert all(started - 1 <= p.at.timestamp() <= time.time() + 1 for p in previews)

    theirs = uploaded(client)
    assert client.post(f"/api/projects/{theirs}/preview", json=lens).status_code == 200
    assert not (project_dir(client, theirs) / "previews.jsonl").exists()


def added_file(client: TestClient, pid: str, name: str, frame) -> str:
    """``frame`` uploaded to the project as a file to join (``POST …/files/upload``); its id."""
    added = client.post(f"/api/projects/{pid}/files/upload",
                        files={"file": (name, frame.to_csv(index=False).encode(), "text/csv")})
    assert added.status_code == 200, added.text[:300]
    return added.json()["id"]


def test_each_join_preview_is_logged_in_analyst_mode_and_not_for_a_person(client, analyst):
    """A join preview previews a ``join_files`` decision: in analyst mode each is logged with the
    previews as that decision, the refusal's code beside it when it was refused (many-to-many,
    which the preview carries; an unknown file, which it raises); a join whose file is not written
    as a file's id is logged with no decision, so the log still serves. Reference: the table
    repeats ``participant_id`` (DIETARY), so a file with one row per identifier joins many to one,
    and one with two rows per identifier is many to many."""
    import pandas as pd

    ids = pd.read_csv(DIETARY)["participant_id"].drop_duplicates()
    once = pd.DataFrame({"participant_id": ids,
                         "fasting_glucose": [90 + i % 17 for i in range(len(ids))]})
    twice = pd.concat([once, once], ignore_index=True)
    assert ids.size < pd.read_csv(DIETARY).shape[0]  # the table's identifier repeats

    mine = uploaded(analyst)
    one, two = added_file(analyst, mine, "GLU_J.csv", once), added_file(analyst, mine, "x.csv", twice)
    started = time.time()
    asks = [({"file": one, "on": "participant_id"}, 200, None),
            ({"file": two, "on": "participant_id", "how": "inner"}, 200, "many_to_many"),
            ({"file": "f0000000000", "on": "participant_id"}, 404, "unknown_file")]
    for body, status, code in asks:
        answer = analyst.post(f"/api/projects/{mine}/join-preview", json=body)
        assert answer.status_code == status, answer.text[:300]
        if status == 200:
            assert (answer.json()["refusal"] or {}).get("code") == code
    malformed = analyst.post(f"/api/projects/{mine}/join-preview",
                             json={"file": "GLU_J.csv", "on": "participant_id"})
    assert malformed.status_code == 404 and malformed.json()["error"]["code"] == "unknown_file"
    served = analyst.get(f"/api/projects/{mine}/analyst-log")
    assert served.status_code == 200, served.text[:300]
    *previews, last = schemas.AnalystLog.model_validate(served.json()).previews
    assert [(p.decision.model_dump(mode="json", exclude={"name", "counts"}), p.refused)
            for p in previews] == [
        ({"kind": "join_files", "right_on": None, "how": "left", **body}, code)
        for body, _, code in asks]
    assert (last.decision, last.refused) == (None, "unknown_file")
    assert all(started - 1 <= p.at.timestamp() <= time.time() + 1 for p in [*previews, last])

    theirs = uploaded(client)
    fid = added_file(client, theirs, "GLU_J.csv", once)
    answer = client.post(f"/api/projects/{theirs}/join-preview",
                         json={"file": fid, "on": "participant_id"})
    assert answer.status_code == 200 and answer.json()["refusal"] is None
    assert not (project_dir(client, theirs) / "previews.jsonl").exists()


# ── the note, in every mode ──────────────────────────────────────────────────


@pytest.mark.parametrize("mode", ["person", "analyst"])
def test_a_decision_s_note_round_trips_and_never_reaches_the_methods(mode, client, analyst):
    c = client if mode == "person" else analyst
    pid = uploaded(c)
    view = decide(c, pid, {"kind": "set_lens", "lenses": ["dietary"]}, note=NOTE)
    assert view.status_code == 200, view.text
    [record] = view.json()["decisions"]
    assert record["note"] == NOTE
    assert c.get(f"/api/projects/{pid}").json()["decisions"][0]["note"] == NOTE
    line = json.loads((project_dir(c, pid) / "decisions.jsonl").read_text("utf-8").splitlines()[0])
    assert line["note"] == NOTE
    methods = c.get(f"/api/projects/{pid}/methods")
    sections = c.get(f"/api/projects/{pid}/methods/sections")
    assert methods.status_code == sections.status_code == 200
    assert NOTE not in methods.text and NOTE not in sections.text
    assert "paper's primary endpoint" not in methods.text + sections.text
    unnoted = decide(c, pid, {"kind": "set_lens", "lenses": ["dietary"]})
    assert unnoted.json()["decisions"][-1]["note"] is None


def test_a_note_past_2000_characters_is_refused_and_records_nothing(client):
    pid = uploaded(client)
    whole = ("because " * 250)[:-1] + "."
    assert len(whole) == 2000
    kept = decide(client, pid, {"kind": "set_lens", "lenses": ["dietary"]}, note=whole)
    assert kept.status_code == 200 and kept.json()["decisions"][-1]["note"] == whole
    refused = decide(client, pid, {"kind": "set_lens", "lenses": ["dietary"]}, note=whole + "!")
    assert refused.status_code == 409
    error = refused.json()["error"]
    assert error["code"] == "note_too_long" and "2,000" in error["message"]
    assert len(client.get(f"/api/projects/{pid}").json()["decisions"]) == 1


# ── E5: where each file came from ────────────────────────────────────────────


def test_an_upload_keeps_where_its_file_came_from_and_the_export_echoes_it(client):
    service = client.app.state.service
    pid = uploaded(client, **NHANES_GLU)
    meta = json.loads((project_dir(client, pid) / "project.json").read_text("utf-8"))
    assert meta["provenance"] == NHANES_GLU
    state = service.log(pid).state()
    [table] = service._export_inputs(pid, service.workspace.get(pid), state)
    entry = table.to_json()
    assert {k: entry[k] for k in NHANES_GLU} == NHANES_GLU
    assert entry["sha256"] == hashlib.sha256(DIETARY.read_bytes()).hexdigest()

    plain = uploaded(client)
    assert "provenance" not in json.loads(
        (project_dir(client, plain) / "project.json").read_text("utf-8"))
    [bare] = service._export_inputs(plain, service.workspace.get(plain), service.log(plain).state())
    assert not set(NHANES_GLU) & set(bare.to_json())


def test_a_joined_file_keeps_its_sha256_and_where_it_came_from_and_the_export_echoes_both(client):
    """A file added to join (DATAIN) is served with the SHA-256 of its bytes beside its
    fingerprint, and keeps where it came from when that is given; once joined, the export's
    inputs echo it beside the table's. The service is called as the upload route calls it
    (``routes/assembly.py``, whose query fields are not wired yet)."""
    import pandas as pd

    from turbotab.core.workspace import source_provenance

    service = client.app.state.service
    pid = uploaded(client)
    assert decide(client, pid, {"kind": "set_lens", "lenses": ["dietary"]}).status_code == 200
    ids = pd.read_csv(DIETARY)["participant_id"].drop_duplicates()  # one row each: many to one
    extra = pd.DataFrame({"participant_id": ids,
                          "fasting_glucose": [90 + i % 17 for i in range(len(ids))]})
    staged = service.workspace.uploads_dir() / "u_joined" / "GLU_J.csv"
    staged.parent.mkdir(parents=True, exist_ok=True)
    extra.to_csv(staged, index=False)
    data = staged.read_bytes()
    added = service.add_file_from_upload(pid, staged, "GLU_J.csv", source_provenance(**NHANES_GLU))
    [listed] = client.get(f"/api/projects/{pid}/files").json()
    assert listed["sha256"] == hashlib.sha256(data).hexdigest()
    assert listed["fingerprint"] == hashlib.blake2b(data, digest_size=16).hexdigest()
    joined = decide(client, pid, {"kind": "join_files", "file": added["id"], "on": "participant_id"})
    assert joined.status_code == 200, joined.text[:400]
    table, glu = (f.to_json() for f in service._export_inputs(
        pid, service.workspace.get(pid), service.log(pid).state()))
    assert glu["role"] == "joined file" and glu["sha256"] == hashlib.sha256(data).hexdigest()
    assert {k: glu[k] for k in NHANES_GLU} == NHANES_GLU
    assert not set(NHANES_GLU) & set(table)  # the table was uploaded with none


@pytest.mark.parametrize("field, value", [("release_date", "November 2019"),
                                          ("url", "wwwn.cdc.gov/Nchs/GLU_J.xpt")])
def test_a_provenance_field_that_does_not_read_refuses_the_upload_and_makes_no_project(
        client, field, value):
    before = {p["id"] for p in client.get("/api/projects").json()}
    refused = upload(client, DIETARY.read_bytes(), **{**NHANES_GLU, field: value})
    assert refused.status_code == 400
    assert refused.json()["error"]["code"] == "bad_provenance"
    assert {p["id"] for p in client.get("/api/projects").json()} == before


def test_the_ingest_serves_the_sha256_of_the_uploaded_bytes_beside_its_fingerprint(client):
    data = DIETARY.read_bytes().replace(b"\n", b"\r\n")  # bytes no sample file on disk has
    pid = uploaded(client, data)
    artifact = client.get(f"/api/projects/{pid}/stages/ingest").json()["artifact"]
    schemas.DatasetInfo.model_validate(artifact)
    assert artifact["source_sha256"] == hashlib.sha256(data).hexdigest()
    assert artifact["fingerprint"] == hashlib.blake2b(data, digest_size=16).hexdigest()
    assert artifact["source_sha256"] != hashlib.sha256(DIETARY.read_bytes()).hexdigest()


# ── the export carries both, and replays as before ───────────────────────────


@pytest.mark.slow
def test_the_export_echoes_the_file_s_provenance_keeps_the_note_out_of_the_methods_and_replays(
        analyst, tmp_path):
    """The synthetic diet under Predict (the acceptance export's journey), uploaded with where it
    came from and one decision noted, through Fit and the opening of the held-out rows, in analyst
    mode. The bundle's input is the uploaded bytes (hashlib's SHA-256) with the three fields as
    given; the note is in the decision log and in no other file; and a fresh-home replay of the
    bundle reproduces it as one without them does (``test_export``'s replay). Once the held-out
    rows are opened, a change to what a row is is refused with no exit that withdraws them or
    draws them again (``test_seal`` has a person's, which offers it)."""
    from turbotab.core.export import replay
    from turbotab.core.tests.acceptance import test_export as journey
    from turbotab.core.tests.acceptance.server_drive import Drive
    from turbotab.core.tests.truths import Truth

    csv = tmp_path / "diet.csv"
    journey._diet().to_csv(csv, index=False)
    data = csv.read_bytes()
    made = upload(analyst, data, name=csv.name, **NHANES_GLU)
    assert made.status_code == 200, made.text
    drive = Drive(analyst, made.json()["id"], Truth(journey.DIET_TRUTH, fixture="the synthetic diet"))
    drive.artifact("ingest")
    drive.decide({"kind": "set_lens", "lenses": ["dietary"]})
    drive.reach("target")
    drive.decide({"kind": "set_target", "column": "dm"})
    drive.answer("event", {"kind": "set_event", "column": "dm", "level": "True"})
    drive.reach("purpose")
    noted = decide(analyst, drive.pid, {"kind": "set_purpose", "purpose": "prediction"}, note=NOTE)
    assert noted.status_code == 200, noted.text
    drive.answer("grain", {"kind": "set_grain", "grain": "one_row_per_unit", "id_column": "pid"})
    drive.reach("roles")
    drive.decide_roles(journey.DIET_ROLES)
    drive.answer("exclusions", {"kind": "set_exclusions", "rules": []})
    drive.answer("missing", {"kind": "set_missing", "strategy": "complete_case"})
    drive.answer("split", {"kind": "set_split", "holdout": 0.2, "seed": 0, "folds": 5})
    drive.answer("models", {"kind": "select_models", "models": ["linear"]})
    journey._wait_fresh(drive, ("cohort", "design", "fit"))
    assert drive.press_fit()
    exits = journey._export(drive).json()["error"]["exits"]
    drive.decide(next(e["decision"] for e in exits
                      if (e["decision"] or {}).get("family") == "linear"))
    journey._wait_fresh(drive, ("cohort", "design", "fit"))
    exported = journey._export(drive)
    assert exported.status_code == 200, exported.text[:600]

    files = journey._files(exported.content)
    [table] = json.loads(files["provenance.json"])["inputs"]
    assert table["sha256"] == hashlib.sha256(data).hexdigest() and table["name"] == csv.name
    assert {k: table[k] for k in NHANES_GLU} == NHANES_GLU
    holding = sorted(name for name, body in files.items() if NOTE.encode() in body)
    assert holding == ["decisions.jsonl", "provenance.json"]  # the log, and the record of it
    assert "paper's primary endpoint" not in files["methods.md"].decode("utf-8")
    noted_lines = [json.loads(line) for line in files["decisions.jsonl"].decode().splitlines()
                   if NOTE in line]
    assert [r["decision"]["kind"] for r in noted_lines] == ["set_purpose"]

    bundle = tmp_path / "bundle.zip"
    bundle.write_bytes(exported.content)
    report = replay.run(bundle, data=str(csv), workers=2)
    assert report.reproduced is True, report.model_dump()
    assert report.files_different == []

    assert drive.view()["state"]["seal_opened"] is True
    n_records = len(drive.view()["decisions"])
    regrain = decide(analyst, drive.pid, {"kind": "set_grain", "grain": "repeated", "id_column": "pid"})
    assert regrain.status_code == 409, regrain.text[:400]
    error = regrain.json()["error"]
    assert error["code"] == "sealed" and "in analyst mode they are not drawn again" in error["message"]
    assert error["exits"] == [{"label": "Keep the held-out rows as they are", "decision": None}]
    assert len(drive.view()["decisions"]) == n_records
