"""The seal reads the fit it was opened on by the engine's own keys (``graph.stage_keys``).

The seal said a fit had "changed after the opening" when no decision after it changed what the fit
reads: its keys took each slot's raw value, while the engine keys the ``findings`` slot by its
applied repairs alone (``graph.KEY_VIEWS``), so a noticing dismissed before the opening made
``changed_after_seal`` true at the opening itself, and Predict Results said the held-out scores
came "from a fit changed after the opening". The triage of the noticings is recorded before the
opening (and refused after it) and writes only the ``sweeps`` slot, which no stage reads.

A Predict project on dietary_recalls.csv is driven through the real server. The references are
what the server served, read before the seal compares anything: the key of the fit the engine
served when the opening was recorded, and the held-out scores the opening's record kept. The fit
served afterwards is the one the seal was opened on exactly when its key is that key; a new
exclusion after the opening changes the rows the fit reads, so it is a changed fit.
"""
from __future__ import annotations

import pytest

from turbotab.server.tests.conftest import make_client, prepare, wait_for
from turbotab.server.tests.test_fit_lock_repair import press
from turbotab.server.tests.test_seal_api import decide, sealed_project

OPEN = {"kind": "open_seal", "family": "elastic_net"}
CHANGED = "from a fit changed after it"


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    with make_client(tmp_path_factory.mktemp("set_aside"), "local", 2, "http://127.0.0.1") as c:
        yield c


def served_fit(client, pid: str) -> tuple[str, dict]:
    """The fresh fit's key and its artifact as served."""
    view = wait_for(client, pid, {"fit": "fresh"}, timeout=240)
    fit = client.get(f"/api/projects/{pid}/stages/fit").json()["artifact"]
    return view["stages"]["fit"]["key"], fit


def accepted(client, pid: str, decision: dict) -> dict:
    response = client.post(f"/api/projects/{pid}/decisions", json=decision)
    assert response.status_code == 200, response.text
    return response.json()["decisions"][-1]


def settle_the_triage_and_open(client, pid: str) -> dict:
    """The opening, through the triage gate's own exits: the recommended dispositions kept, then
    each noticing to act on recorded as a limitation (``test_opening_gate_api``)."""
    url = f"/api/projects/{pid}/decisions"
    refused = client.post(url, json=OPEN)
    assert refused.status_code == 409 and refused.json()["error"]["code"] == "triage_open"
    accepted(client, pid, next(e["decision"] for e in refused.json()["error"]["exits"] if e["decision"]))
    waiting = client.post(url, json=OPEN)
    if waiting.status_code == 409:
        for e in [e for e in waiting.json()["error"]["exits"] if e["decision"]]:
            accepted(client, pid, e["decision"])
    return accepted(client, pid, OPEN)


def test_a_noticing_set_aside_is_no_change_and_a_new_exclusion_is(client):
    pid = sealed_project(client)
    decide(client, pid, {"kind": "select_models", "models": ["linear", "elastic_net"]})
    wait_for(client, pid, {"fit": "fresh", "split": "fresh", "seal_plan": "fresh"}, timeout=240)
    prepare(client, pid, {"kind": "open_seal"})
    press(client, pid)
    assert client.get(f"/api/projects/{pid}").json()["state"]["purpose"] == "prediction"
    noticings = [f["id"] for f in
                 client.get(f"/api/projects/{pid}/stages/findings").json()["artifact"]["findings"]]
    assert len(noticings) >= 2, noticings

    # Before the opening: one noticing dismissed. The fit it leaves is the fit the seal opens on.
    accepted(client, pid, {"kind": "dismiss_finding", "finding_id": noticings[0]})
    key, _ = served_fit(client, pid)
    opening = settle_the_triage_and_open(client, pid)
    now, fit = served_fit(client, pid)
    assert now == key  # nothing the fit reads changed: the fit served is the one opened on
    assert {m["family"]: m["holdout"] for m in fit["models"]} == opening["decision"]["scores"]
    assert fit["changed_after_seal"] is False and fit["post_seal_decisions"] == []
    assert fit["at_opening"]["current"] is True
    assert fit["at_opening"]["note"].endswith(": the reported result.")
    assert CHANGED not in (fit.get("final_note") or "")

    # After it: another noticing dismissed. The record is post-seal; the fit is the same one.
    after = accepted(client, pid, {"kind": "dismiss_finding", "finding_id": noticings[1]})
    assert after["post_seal"] is True
    now, fit = served_fit(client, pid)
    assert now == key
    assert fit["changed_after_seal"] is False and fit["at_opening"]["current"] is True
    assert CHANGED not in fit["at_opening"]["note"] and CHANGED not in (fit.get("final_note") or "")

    # The triage was recorded before the opening; after it, it is refused and changes nothing.
    triage = client.post(f"/api/projects/{pid}/decisions", json={
        "kind": "confirm_sweep", "stage": "results", "sweep": "noticings", "lines": []})
    assert triage.status_code == 409 and triage.json()["error"]["code"] == "gate_passed"

    # A new exclusion after it changes the rows the fit reads: a changed fit, listed and said.
    rule = {"kind": "set_exclusions", "rules": [{"column": "energy_kcal", "low": 500, "high": 5000,
                                                 "reason": "implausible intake"}]}
    decide(client, pid, rule)
    excluded = next(r for r in reversed(client.get(f"/api/projects/{pid}").json()["decisions"])
                    if r["decision"]["kind"] == "set_exclusions")
    assert excluded["post_seal"] is True
    now, fit = served_fit(client, pid)
    assert now != key
    assert fit["changed_after_seal"] is True
    assert excluded["id"] in fit["post_seal_decisions"]
    assert fit["at_opening"]["current"] is False and CHANGED in fit["at_opening"]["note"]
