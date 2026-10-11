"""W1-H · the opening gate over HTTP (WALKABLE_PLAN §3 P2): on a sealed project with Fit pressed,
``open_seal`` is refused while the triage (``GET /triage``) holds open noticings, naming each; the
one-click "Keep the recommended dispositions" records them but does not open the rows while one
recorded "act on it" is still open; each such noticing has its own way through (decide it, or record
it as a limitation); once every one is settled the rows open. No held-out score is served before.

The reference for what the refusal must name is the server's own triage route, read independently
of the refusal, and the dispositions recorded are read back from the decision log.
"""
from __future__ import annotations

from turbotab.server.tests.conftest import prepare, wait_for
from turbotab.server.tests.test_fit_lock_repair import press, stage
from turbotab.server.tests.test_seal_api import decide, sealed_project

OPEN = {"kind": "open_seal", "family": "elastic_net"}


def _said(summary: str) -> str:
    return summary.strip().rstrip(".")


def test_the_opening_waits_until_every_noticing_is_settled_and_a_kept_act_on_it_is_not_enough(client):
    pid = sealed_project(client)
    decide(client, pid, {"kind": "select_models", "models": ["linear", "elastic_net"]})
    wait_for(client, pid, {"fit": "fresh", "split": "fresh", "seal_plan": "fresh"}, timeout=180)
    prepare(client, pid, {"kind": "open_seal"})
    press(client, pid)
    url = f"/api/projects/{pid}/decisions"

    # 1. Refused, naming every open noticing the triage route lists, each with its disposition.
    triage = client.get(f"/api/projects/{pid}/triage").json()
    assert triage["items"] and not triage["blockers"] and not triage["answered"]
    first = client.post(url, json=OPEN)
    assert first.status_code == 409, first.text
    error = first.json()["error"]
    assert error["code"] == "triage_open"
    for item in triage["items"]:
        assert f"{_said(item['summary'])} ({item['label'].lower()})" in error["message"]
    keep = next(e["decision"] for e in error["exits"] if e["decision"])
    assert keep == {"kind": "confirm_sweep", "stage": "results", "sweep": "noticings", "lines": []}

    # 2. The one click records the recommended dispositions, and the rows stay sealed while a
    #    noticing recorded "act on it" is not acted on.
    assert client.post(url, json=keep).status_code == 200
    triage = client.get(f"/api/projects/{pid}/triage").json()
    to_act = [i for i in triage["items"] if i["recorded"] == "act_on_it"]
    assert to_act, "the dietary noticings are recommended to act on"
    second = client.post(url, json=OPEN)
    assert second.status_code == 409, second.text
    error = second.json()["error"]
    assert error["code"] == "triage_open" and "not acted on yet" in error["message"]
    limits = [e for e in error["exits"] if e["decision"]]
    assert [e["label"] for e in limits] == [
        f"Record as a limitation: {i['summary']}" for i in to_act]
    assert [e["label"] for e in error["exits"] if not e["decision"]] == [
        f"Decide: {i['summary']}" for i in to_act]
    fit = stage(client, pid, "fit")
    assert fit["holdout_sealed"] is True  # nothing held out is served
    assert all(m["holdout"] is None for m in fit["models"])

    # 3. Each recorded as a limitation, by its own exit: then the rows open.
    for e in limits:
        assert client.post(url, json=e["decision"]).status_code == 200, e
    opened = client.post(url, json=OPEN)
    assert opened.status_code == 200, opened.text
    records = client.get(f"/api/projects/{pid}").json()["decisions"]
    sweep = [r for r in records if r["decision"]["kind"] == "confirm_sweep"
             and r["decision"]["sweep"] == "noticings"][-1]
    settled = {l["key"]: l["value"] for l in sweep["decision"]["lines"]}
    assert {i["id"] for i in to_act} <= set(settled)
    assert all(settled[i["id"]] == "could_bias" for i in to_act)
    assert records[-1]["decision"]["kind"] == "open_seal"
    assert stage(client, pid, "fit")["holdout_sealed"] is False
