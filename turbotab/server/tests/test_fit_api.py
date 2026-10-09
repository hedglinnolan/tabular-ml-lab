"""P0.8 over HTTP: Fit, the hold and a visible lock (SIZING P0.8; RECIPES_AND_TUNING §4.4;
CROSSWALK "Settled here"; guarantee test 2 of EXTERNAL_AUDIT_2026-10-09).

* Under Predict, no estimate stage and no cross-validated score is served before Fit is pressed,
  so no family is marked seen (``models.selection.read_seen``); pressing Fit records nothing in the
  decision log, and the scores then served are the fit's own.
* With the purpose withdrawn, an estimate computed earlier is not served.
* A fit expected to take longer than the hold waits for Fit in the scheduler, and starts when Fit is
  pressed.
* Under Estimate, pressing Fit records the plan's lock (``lock_plan``, the existing system record),
  only once the questions the estimates rest on are answered, and the quest log shows the lock
  with its time and the plan's SHA-256, computed here from the lock's own plan.
"""
from __future__ import annotations

import hashlib
import json
import time

from turbotab.core.models.selection import read_seen
from turbotab.core.plan_lock import shows_estimates
from turbotab.server.service import ProjectService
from turbotab.server.tests.conftest import wait_for
from turbotab.server.tests.test_seal_api import decide, sealed_project


def stage(client, pid: str, name: str) -> dict:
    return client.get(f"/api/projects/{pid}/stages/{name}").json()["artifact"]


def scores(fit: dict) -> dict:
    return {m["family"]: m["cv"] for m in fit["models"] if m.get("cv")}


def press(client, pid: str, status: int = 200) -> dict:
    response = client.post(f"/api/projects/{pid}/fit")
    assert response.status_code == status, response.text
    return response.json()


def test_under_prediction_nothing_is_served_and_no_score_is_seen_before_fit(client):
    from turbotab.core import fit_press

    service: ProjectService = client.app.state.service
    pid = sealed_project(client)
    pdir = service.workspace.project_dir(pid)
    decide(client, pid, {"kind": "select_models", "models": ["linear"]})
    wait_for(client, pid, {"fit": "fresh"}, timeout=180)
    assert client.get(f"/api/projects/{pid}").json()["state"]["purpose"] == "prediction"

    before = stage(client, pid, "fit")
    assert scores(before) == {} and before["withheld_for"] == "fit"
    assert not shows_estimates("fit", before)
    assert read_seen(pdir) == {}  # nothing served, nothing marked seen
    quest = client.get(f"/api/projects/{pid}/quest").json()
    assert quest["fit"]["locks"] is False and quest["fit"]["pressed"] is False
    assert not next(s for s in quest["stages"] if s["key"] == "results")["reached"]

    records = len(client.get(f"/api/projects/{pid}").json()["decisions"])
    report = press(client, pid)
    assert report["pressed"] is True and report["locked"] is False
    view = client.get(f"/api/projects/{pid}").json()
    assert len(view["decisions"]) == records  # a job command, not a decision
    assert view["state"]["plan_locked"] is None
    assert fit_press.read_press(pdir)["target"] == "hba1c"

    after = stage(client, pid, "fit")
    raw = service.engine.get(pid, "fit").artifact  # the fit as computed, before any gate
    assert scores(after) == {m["family"]: m["cv"] for m in raw["models"] if m.get("cv")}
    assert set(scores(after)) == {"linear"}
    assert after.get("withheld") is None and after.get("withheld_for") is None
    assert read_seen(pdir) == {"hba1c": ["linear"]}
    quest = client.get(f"/api/projects/{pid}/quest").json()
    assert quest["fit"]["pressed"] is True
    assert next(s for s in quest["stages"] if s["key"] == "results")["reached"]

    # The purpose withdrawn: the fit computed under it is not served (guarantee test 2).
    purpose = next(r for r in view["decisions"] if r["decision"]["kind"] == "set_purpose")
    undo = client.post(f"/api/projects/{pid}/decisions",
                       json={"kind": "revert", "decision_id": purpose["id"]})
    assert undo.status_code == 200, undo.text
    assert undo.json()["state"]["purpose"] is None
    for name in ("fit", "explain", "evaluation"):
        served = client.get(f"/api/projects/{pid}/stages/{name}").json()["artifact"]
        assert not shows_estimates(name, served), name
    withdrawn = stage(client, pid, "fit")
    assert withdrawn["withheld_for"] == "purpose" and scores(withdrawn) == {}


def test_a_fit_longer_than_the_hold_waits_for_fit(client, monkeypatch):
    from turbotab.core import fit_press

    monkeypatch.setattr(fit_press, "HOLD_SECONDS", 0.0)  # every measured fit is a long one
    pid = sealed_project(client)
    decide(client, pid, {"kind": "select_models", "models": ["linear"]})
    wait_for(client, pid, {"shelf": "fresh"}, timeout=180)
    end = time.monotonic() + 30
    while True:
        fit = client.get(f"/api/projects/{pid}").json()["stages"]["fit"]
        if fit["held"] == "fit":
            break
        assert fit["status"] in ("idle", "stale", "queued", "running"), fit
        assert fit["status"] in ("idle", "stale") or time.monotonic() < end, fit
        time.sleep(0.1)
    time.sleep(1.0)  # held, not merely slow to start
    fit = client.get(f"/api/projects/{pid}").json()["stages"]["fit"]
    assert fit["status"] in ("idle", "stale") and fit["held"] == "fit" and fit["job_id"] is None
    quest = client.get(f"/api/projects/{pid}/quest").json()["fit"]
    assert quest["held"] is True and quest["estimate_seconds"] > 0

    press(client, pid)
    wait_for(client, pid, {"fit": "fresh"}, timeout=180)
    assert client.get(f"/api/projects/{pid}").json()["stages"]["fit"]["held"] is None


def test_fit_is_refused_while_nothing_is_chosen_to_fit_and_the_lock_is_never_posted(client):
    pid = sealed_project(client)
    refused = press(client, pid, status=409)  # nothing chosen to fit yet
    assert refused["error"]["code"] == "fit_not_yet"
    posted = client.post(f"/api/projects/{pid}/decisions", json={"kind": "lock_plan"})
    assert posted.status_code == 409  # under prediction nothing locks


def test_the_lock_is_shown_with_its_time_and_the_plan_s_fingerprint(client):
    from turbotab.core.tests.acceptance.server_drive import answer_plan, open_project
    from turbotab.core.tests.stage_harness import NHANES
    from turbotab.core.tests.truths import fixture_truth

    drive = open_project(client, NHANES, fixture_truth("_tt_tmp_nhanes.csv"))
    drive.decide({"kind": "set_lens", "lenses": ["dietary"]})
    drive.reach("target")
    drive.decide({"kind": "set_target", "column": "glucose"})
    drive.reach("purpose")
    drive.decide({"kind": "set_purpose", "purpose": "inference"})
    drive.answer("grain", {"kind": "set_grain", "grain": "one_row_per_unit"})
    drive.reach("roles")
    from turbotab.server.tests.test_quest_api import ROLES, rule

    drive.decide_roles(ROLES)
    drive.answer("exclusions", rule(500, 5000))
    drive.answer("missing", {"kind": "set_missing", "strategy": "complete_case"})
    drive.answer("split", {"kind": "set_split", "holdout": 0.0, "seed": 0, "folds": 5})
    answer_plan(drive, "sugar")
    drive.answer("energy_adjustment", {"kind": "set_energy_adjustment", "method": "standard",
                                       "energy_column": "kcal", "nutrients": ["sugar"]})
    drive.decide({"kind": "select_models", "models": ["linear"]})
    wait_for(client, drive.pid, {"fit": "fresh"}, timeout=240)

    before = stage(client, drive.pid, "fit")
    assert before["withheld_for"] == "fit" and not shows_estimates("fit", before)
    assert drive.view()["state"]["plan_locked"] is None  # served, withheld, nothing locked
    unlocked = client.get(f"/api/projects/{drive.pid}/quest").json()["fit"]
    assert unlocked["locks"] and not unlocked["locked"] and unlocked["sha256"] is None
    posted = drive.post({"kind": "lock_plan"})  # a client never posts the lock: Fit records it
    assert posted.status_code == 409 and posted.json()["error"]["code"] == "plan_locks_itself"
    assert "pressing Fit" in posted.json()["error"]["message"]

    report = press(client, drive.pid)
    view = drive.view()
    lock = view["decisions"][-1]
    assert lock["decision"]["kind"] == "lock_plan" and view["state"]["plan_locked"] is True
    plan = lock["decision"]["plan"]
    sha = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False).encode("utf-8")).hexdigest()
    assert report["locked"] and report["sha256"] == sha
    quest = client.get(f"/api/projects/{drive.pid}/quest").json()["fit"]
    assert quest["locked"] and quest["sha256"] == sha and quest["at"].startswith(lock["at"][:19])
    assert "before any estimate was shown" in quest["reason"]
    assert shows_estimates("fit", stage(client, drive.pid, "fit"))

    # Pressed again, nothing more is recorded: the lock is recorded once.
    press(client, drive.pid)
    assert len(drive.view()["decisions"]) == len(view["decisions"])
