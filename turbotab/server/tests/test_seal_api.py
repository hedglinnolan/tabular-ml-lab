"""Tier A over HTTP: the seal (M2_CONTRACT §3).

* No held-out score reaches any API response before ``open_seal``: every route the app mounts is
  called (each stage, every preview kind, every finding's evidence, the data reads, the jobs, the
  file browser, a refused and an accepted decision), and the event stream is spied on at the bus.
* After ``open_seal`` the scores appear, identical to the ones the fit computed; the seal opens once
  and is never reverted.
* Decision A waits for a re-seal, and the re-seal path its refusal names works.
* Every decision after the opening is marked post-seal, and the Results say the fit changed.
"""
from __future__ import annotations

import json
from typing import Any, Iterator

import pytest

from turbotab.core import seal
from turbotab.server.service import ProjectService
from turbotab.server.tests.conftest import DIETARY, open_by_path, wait_for

FAMILIES = ["linear", "elastic_net", "boosted_trees"]
# The event stream never ends, so it is spied on where it starts: every event the bus publishes.
EXEMPT = {"/api/projects/{pid}/events"}


def decide(client, pid, decision, status=200):
    response = client.post(f"/api/projects/{pid}/decisions", json=decision)
    assert response.status_code == status, response.text
    return response.json()


def sealed_project(client) -> str:
    """dietary_recalls.csv up to a fresh fit, people stated to repeat, 25% held out."""
    pid = open_by_path(client)
    wait_for(client, pid, {"ingest": "fresh", "profile": "fresh"})
    decide(client, pid, {"kind": "set_lens", "lenses": ["dietary"]})
    decide(client, pid, {"kind": "set_target", "column": "hba1c"})
    decide(client, pid, {"kind": "set_grain", "grain": "repeated", "id_column": "participant_id"})
    decide(client, pid, {"kind": "set_unit", "unit": "row"})
    wait_for(client, pid, {"roles": "fresh", "target_info": "fresh", "cohort": "fresh"})
    roles = client.get(f"/api/projects/{pid}/stages/roles").json()["artifact"]
    decide(client, pid, {"kind": "set_roles", "roles": {c["column"]: c["proposed"] for c in roles["columns"]}})
    decide(client, pid, {"kind": "set_missing", "strategy": "complete_case"})
    decide(client, pid, {"kind": "set_split", "holdout": 0.25, "seed": 0, "folds": 5})
    return pid


def numbers(obj: Any) -> Iterator[float]:
    if isinstance(obj, bool):
        return
    if isinstance(obj, float):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from numbers(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from numbers(v)


def strings(obj: Any) -> Iterator[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k)
            yield from strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from strings(v)


def score_dicts(obj: Any) -> Iterator[dict]:
    """Every ``holdout`` that holds scores (a mapping) anywhere in a response."""
    if isinstance(obj, dict):
        if isinstance(obj.get("holdout"), dict):
            yield obj["holdout"]
        for v in obj.values():
            yield from score_dicts(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from score_dicts(v)


def body(response) -> Any:
    try:
        return response.json()
    except ValueError:
        return response.text


def representative_decisions(pid_roles: dict, finding_id: str | None) -> list[dict]:
    fid = finding_id or "no_such_finding"
    return [
        {"kind": "set_lens", "lenses": ["dietary", "clinical"]},
        {"kind": "set_target", "column": "energy_kcal"},
        {"kind": "set_task", "column": "hba1c", "task": "regression"},
        {"kind": "set_purpose", "purpose": "inference"},
        {"kind": "revert", "decision_id": "0" * 32},
        {"kind": "set_roles", "roles": pid_roles},
        {"kind": "set_energy_adjustment", "method": "residual", "energy_column": "energy_kcal",
         "nutrients": ["protein_g", "fat_g"]},
        {"kind": "set_exclusions", "rules": [{"column": "energy_kcal", "low": 500, "high": 5000,
                                              "reason": "implausible intake"}]},
        {"kind": "set_missing", "strategy": "impute"},
        {"kind": "set_split", "holdout": 0.2, "seed": 1, "folds": 5},
        {"kind": "set_split", "holdout": 0.0, "seed": 0, "folds": 5},
        {"kind": "select_models", "models": ["linear"]},
        {"kind": "set_substitution", "donor": "fat_g", "recipient": "protein_g"},
        {"kind": "set_orientation", "orientation": "sample_major"},
        {"kind": "set_event", "column": "hba1c", "level": "1"},
        {"kind": "set_grain", "grain": "one_row_per_unit"},
        {"kind": "set_repeat_kind", "repeat_kind": "repeats"},
        {"kind": "set_unit", "unit": "unit"},
        {"kind": "set_aggregation", "method": "mean"},
        {"kind": "set_temporal", "temporal": False},
        {"kind": "open_seal"},
        {"kind": "apply_repair", "finding_id": fid, "option": "set_missing"},
        {"kind": "defer_finding", "finding_id": fid, "to": "exclusions"},
        {"kind": "dismiss_finding", "finding_id": fid},
    ]


def mounted(client) -> dict[str, set[str]]:
    """Every API path the app serves and its methods, from the app's own OpenAPI document."""
    paths = client.app.openapi()["paths"]
    return {path: {m.upper() for m in item} for path, item in paths.items() if path.startswith("/api")}


def sweep(client, pid: str, service: ProjectService, tmp_path,
          job_ids: list[str]) -> tuple[list[Any], set[str]]:
    """Call every API route the app mounts; returns every response body and the paths called."""
    stages = [s.name for s in service.engine.graph.stages()]
    findings = client.get(f"/api/projects/{pid}/stages/findings").json()["artifact"]["findings"]
    assert findings, "the sweep needs a finding to ask for evidence"
    roles = client.get(f"/api/projects/{pid}").json()["state"]["roles"]
    values = {"pid": [pid], "stage": stages, "name": ["hba1c"], "fid": [f["id"] for f in findings],
              "jid": job_ids[-1:] or ["no-such-job"]}
    posts: dict[str, list[dict] | None] = {
        "/api/projects": [{"path": str(DIETARY)}],
        "/api/projects/{pid}/decisions": [
            {"kind": "set_target", "column": "no_such_column"},  # refused: 409
            {"kind": "set_missing", "strategy": "complete_case"},  # the same answer again
        ],
        "/api/projects/{pid}/preview": representative_decisions(
            roles, findings[0]["id"] if findings else None),
    }
    queries = {"/api/fs/list": {"path": str(tmp_path)}}
    bodies: list[Any] = []
    called: set[str] = set()
    for template, methods in mounted(client).items():
        if template in EXEMPT:
            continue
        paths = [template]
        for name, options in values.items():
            if "{" + name + "}" in template:
                paths = [p.replace("{" + name + "}", str(v)) for p in paths for v in options]
        assert "{" not in paths[0], f"the sweep does not know how to fill {template}"
        for method in sorted(methods):
            for path in paths:
                if method == "GET":
                    responses = [client.get(path, params=queries.get(template))]
                elif template == "/api/projects/upload":
                    with open(DIETARY, "rb") as fh:
                        responses = [client.post(path, files={"file": (DIETARY.name, fh, "text/csv")})]
                elif template in posts:
                    responses = [client.post(path, json=b) for b in posts[template]]
                else:
                    responses = [client.post(path)]
                for response in responses:
                    assert response.status_code < 500, (path, response.text)
                    bodies.append(body(response))
                called.add(template)
    return bodies, called


def assert_nothing_held_out_in(bodies: list[Any], sealed: dict[str, dict[str, float]]) -> None:
    hidden = [v for scores in sealed.values() for v in scores.values() if v is not None]
    assert hidden, "the fit computed no held-out scores to hide"
    for b in bodies:
        assert not list(score_dicts(b)), json.dumps(b)[:400]
        seen = set(numbers(b))
        for v in hidden:
            assert v not in seen, (v, json.dumps(b)[:400])
        text = " ".join(strings(b))
        for v in hidden:
            for shown in (f"{v:.4f}", f"{v:.4f}".replace("-", "−")):
                assert shown not in text, (shown, text[:400])


@pytest.fixture
def bus_spy(client, monkeypatch):
    """Every event the bus publishes while the test runs (the SSE stream's whole content)."""
    service: ProjectService = client.app.state.service
    published: list[tuple[str, str, dict]] = []
    original = service.bus.publish

    def spy(pid, event_type, data):
        published.append((pid, event_type, json.loads(json.dumps(data, default=str))))
        return original(pid, event_type, data)

    monkeypatch.setattr(service.bus, "publish", spy)
    return published


def test_no_held_out_score_reaches_any_response_until_the_seal_is_opened(client, bus_spy, tmp_path):
    service: ProjectService = client.app.state.service
    pid = sealed_project(client)
    refused = client.post(f"/api/projects/{pid}/decisions", json={"kind": "open_seal"})
    assert refused.status_code == 409  # no fit yet: nothing to open
    decide(client, pid, {"kind": "select_models", "models": FAMILIES})
    wait_for(client, pid, {"fit": "fresh", "split": "fresh", "seal_plan": "fresh"}, timeout=180)
    key = service.engine.status(pid)["fit"].key
    sealed = seal.read_sealed_scores(service.workspace.cache_dir(pid), key)
    assert set(sealed) == set(FAMILIES)

    fit = client.get(f"/api/projects/{pid}/stages/fit").json()["artifact"]
    assert fit["holdout_sealed"] is True and fit["n_holdout"] > 0
    assert all(m["holdout"] is None for m in fit["models"])
    split = client.get(f"/api/projects/{pid}/stages/split").json()["artifact"]
    assert split["basis"]["state"] == "grouped" and split["basis"]["column"] == "participant_id"

    job_ids = [e["job_id"] for p, kind, e in bus_spy if p == pid and kind == "job" and e.get("job_id")]
    assert job_ids
    bodies, called = sweep(client, pid, service, tmp_path, job_ids)
    every = set(mounted(client)) - EXEMPT
    assert len(every) >= 15 and called == every, every - called  # every route but the stream
    fits = [b for b in bodies if isinstance(b, dict) and b.get("stage") == "fit"]
    assert fits and fits[0]["artifact"]["holdout_sealed"] is True  # the sweep did read the fit
    events = [data for p, _, data in bus_spy if p == pid]
    assert any(e.get("stage") == "fit" for e in events)
    assert_nothing_held_out_in(bodies + events, sealed)

    # Opened once: the scores appear, exactly the ones the fit computed.
    view = decide(client, pid, {"kind": "open_seal"})
    assert view["state"]["seal_opened"] is True
    opened = client.get(f"/api/projects/{pid}/stages/fit").json()["artifact"]
    assert opened["holdout_sealed"] is False and opened["changed_after_seal"] is False
    assert {m["family"]: m["holdout"] for m in opened["models"]} == sealed
    again = client.post(f"/api/projects/{pid}/decisions", json={"kind": "open_seal"})
    assert again.status_code == 409 and again.json()["error"]["code"] == "seal_already_open"
    opening = next(r for r in view["decisions"] if r["decision"]["kind"] == "open_seal")
    undo = client.post(f"/api/projects/{pid}/decisions", json={"kind": "revert", "decision_id": opening["id"]})
    assert undo.status_code == 409 and undo.json()["error"]["code"] == "seal_stays_open"
    # The keys the seal compares are the engine's own.
    from turbotab.core.decisions import fold

    state = fold(service.log(pid).records())
    assert seal.keys_for(service.engine.graph, state, service.fingerprint(pid)) == service.engine.keys(pid)


def test_a_change_after_the_opening_is_marked_post_seal_and_the_results_say_so(client):
    service: ProjectService = client.app.state.service
    pid = sealed_project(client)
    decide(client, pid, {"kind": "select_models", "models": ["linear", "boosted_trees"]})
    wait_for(client, pid, {"fit": "fresh"}, timeout=180)
    view = decide(client, pid, {"kind": "open_seal"})
    assert not any(r["post_seal"] for r in view["decisions"])  # nothing before it, nor the opening

    view = decide(client, pid, {"kind": "select_models", "models": ["linear"]})
    changed = view["decisions"][-1]
    assert changed["post_seal"] is True
    assert changed["sentence"].startswith("After the held-out rows were opened, ")
    wait_for(client, pid, {"fit": "fresh"}, timeout=180)
    fit = client.get(f"/api/projects/{pid}/stages/fit").json()["artifact"]
    assert fit["changed_after_seal"] is True and fit["post_seal_decisions"] == [changed["id"]]
    assert all(m["holdout"] is not None for m in fit["models"])  # still opened: still shown
    key = service.engine.status(pid)["fit"].key
    assert {m["family"]: m["holdout"] for m in fit["models"]} == seal.read_sealed_scores(
        service.workspace.cache_dir(pid), key)

    # Changing it back is a post-seal decision too, and the fit is the one the seal opened on.
    view = decide(client, pid, {"kind": "revert", "decision_id": changed["id"]})
    assert view["decisions"][-1]["post_seal"] is True
    wait_for(client, pid, {"fit": "fresh"}, timeout=180)
    assert client.get(f"/api/projects/{pid}/stages/fit").json()["artifact"]["changed_after_seal"] is False


def test_the_split_question_says_what_a_holdout_of_each_size_can_measure(client):
    pid = sealed_project(client)
    wait_for(client, pid, {"seal_plan": "fresh", "split": "fresh"})
    plan = client.get(f"/api/projects/{pid}/stages/seal_plan").json()["artifact"]
    assert plan["n_measured"] == 600 and plan["basis"]["state"] == "grouped"
    assert sorted(o["holdout"] for o in plan["options"]) == [0.0, 0.1, 0.2, 0.3]
    assert plan["floor"]["n"] == 100 and plan["floor"]["convention"] is True  # stated as such
    small = client.post(f"/api/projects/{pid}/preview",
                        json={"kind": "set_split", "holdout": 0.1, "seed": 0, "folds": 5}).json()
    assert "R² known to about ±" in small["views"][0]["caption"]
    assert small["caution"]["exits"][0]["decision"]["holdout"] == 0.0  # never refused: offered
    usual = client.post(f"/api/projects/{pid}/preview",
                        json={"kind": "set_split", "holdout": 0.3, "seed": 0, "folds": 5}).json()
    assert usual["caution"] is None and "grouped by `participant_id`" in usual["views"][0]["caption"]


def test_decision_a_waits_for_a_reseal_and_the_reseal_path_works(client):
    pid = sealed_project(client)
    view = client.get(f"/api/projects/{pid}").json()
    writer = [r for r in view["decisions"] if r["decision"]["kind"] == "set_split"][-1]["id"]
    for decision in ({"kind": "set_grain", "grain": "one_row_per_unit"},
                     {"kind": "set_unit", "unit": "unit"}):
        refused = client.post(f"/api/projects/{pid}/decisions", json=decision)
        assert refused.status_code == 409, refused.text
        error = refused.json()["error"]
        assert error["code"] == "sealed"
        assert error["exits"][0]["decision"] == {"kind": "revert", "decision_id": writer}
    for decision in ({"kind": "set_orientation", "orientation": "feature_major"},
                     {"kind": "set_aggregation", "method": "mean"}):
        assert client.post(f"/api/projects/{pid}/decisions", json=decision).status_code == 409
    grain = next(r for r in view["decisions"] if r["decision"]["kind"] == "set_grain")
    undo = client.post(f"/api/projects/{pid}/decisions", json={"kind": "revert", "decision_id": grain["id"]})
    assert undo.status_code == 409 and undo.json()["error"]["code"] == "sealed"

    # The exit: withdraw the seal, change the grain, draw the seal again.
    decide(client, pid, {"kind": "revert", "decision_id": writer})
    decide(client, pid, {"kind": "set_grain", "grain": "one_row_per_unit"})
    decide(client, pid, {"kind": "set_split", "holdout": 0.25, "seed": 0, "folds": 5})
    wait_for(client, pid, {"split": "fresh"})
    split = client.get(f"/api/projects/{pid}/stages/split").json()["artifact"]
    # One row per unit was answered while `participant_id` repeats: drawn by row, and said so.
    assert split["basis"]["state"] == "abandoned" and split["exploratory"] is True
