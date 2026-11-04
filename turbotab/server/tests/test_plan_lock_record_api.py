"""The plan lock's record is true, over HTTP (INBOX 1044, 1045 and 1016; calm/FOUNDATION §7; SIZING
P0.8; ``plan_lock``).

* A client's revert of TurboTab's withdrawal of a lock is refused, so no lock comes back over a
  plan changed since: the plan GET /plan declares and the lock a press of Fit then records hold the
  models in force, and the fit serves those models. Pressing Fit is offered as the way forward
  only while the plan is unlocked, where a press records a lock.
* Down an older engine's chain of reverts, a client's revert that would withdraw the lock in force
  is refused in words that do not say an estimate was displayed, one the log refuses hears the log's
  refusal without a refused exit, and one that moves no lock is recorded.
* A change the log refuses leaves no withdrawal behind: the lock is withdrawn only for a change the
  log then records.
* A revert that makes the goal inference again, after estimates were shown under prediction, locks
  the plan then, as a new answer of the goal does; one made before any estimate was shown locks
  nothing.
* A press of Fit a lock check refuses keeps no press: Fit stays unpressed, so nothing reads as
  pressed for a plan that was never locked, and the press accepted later records both.

Each expectation is read from the record (the log's own records, the lock's own plan, hashed here
with ``hashlib``), the request as made, or the fixture's own construction, never from the code
under test.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import pytest

from turbotab.core import decisions, fit_press
from turbotab.core.decisions import Refusal, Revert, SetMissing
from turbotab.server.tests.conftest import declare, open_by_path, prepare, wait_for
from turbotab.server.tests.test_fit_lock_repair import _inference_drive, press, revert, stage


def view_of(client, pid: str) -> dict:
    return client.get(f"/api/projects/{pid}").json()


def canonical_sha(plan: dict) -> str:
    text = json.dumps(plan, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def families(fit: dict) -> list[str]:
    return [m["family"] for m in fit["models"]]


# ── 1. a revert of TurboTab's withdrawal is refused (INBOX 1044) ─────────────


@pytest.mark.slow
def test_a_revert_of_turbotabs_withdrawal_is_refused_and_the_plan_and_fit_agree(client):
    drive = _inference_drive(client)
    pid = drive.pid
    drive.decide({"kind": "select_models", "models": ["linear"]})
    wait_for(client, pid, {"fit": "fresh"}, timeout=240)
    press(client, pid)
    lock = view_of(client, pid)["decisions"][-1]
    assert lock["decision"]["kind"] == "lock_plan" and lock["decision"]["plan"]["models"] == [
        "linear"]
    drive.decide({"kind": "select_models", "models": ["linear", "elastic_net"]})
    v = view_of(client, pid)
    withdrawal = v["decisions"][-2]
    assert withdrawal["decision"] == {"kind": "revert", "decision_id": lock["id"]}
    assert withdrawal["recorded_by"] == "turbotab" and v["state"]["plan_locked"] is None
    count = len(v["decisions"])

    refused = client.post(f"/api/projects/{pid}/decisions",
                          json={"kind": "revert", "decision_id": withdrawal["id"]})
    assert refused.status_code == 409, refused.text
    error = refused.json()["error"]
    assert error["code"] == "lock_stays_withdrawn"
    assert [e["label"] for e in error["exits"]] == [fit_press.PRESS_FIT]
    v = view_of(client, pid)
    assert len(v["decisions"]) == count and v["state"]["plan_locked"] is None
    assert v["state"]["models"] == ["linear", "elastic_net"]
    plan = client.get(f"/api/projects/{pid}/plan").json()
    assert plan["status"] == "declared" and plan["plan"]["models"] == v["state"]["models"]

    # The guard is the lock's alone: the person's revert of their own revert is recorded.
    change = v["decisions"][-1]
    assert change["decision"]["kind"] == "select_models"
    undone = revert(client, pid, change)
    assert undone["state"]["models"] == ["linear"]
    revert(client, pid, view_of(client, pid)["decisions"][-1])
    v = view_of(client, pid)
    assert v["state"]["models"] == ["linear", "elastic_net"] and v["state"]["plan_locked"] is None

    # Fit pressed: the lock it records holds the models in force, and the plan says the same.
    wait_for(client, pid, {"fit": "fresh"}, timeout=240)
    report = press(client, pid)
    relocked = view_of(client, pid)["decisions"][-1]
    assert report["locked"] is True and relocked["decision"]["kind"] == "lock_plan"
    assert relocked["id"] != lock["id"]
    assert relocked["decision"]["plan"]["models"] == ["linear", "elastic_net"]
    assert report["sha256"] == canonical_sha(relocked["decision"]["plan"])
    plan = client.get(f"/api/projects/{pid}/plan").json()
    assert plan["status"] == "locked" and plan["plan"]["models"] == ["linear", "elastic_net"]
    assert plan["plan_sha256"] == report["sha256"]
    assert families(stage(client, pid, "fit")) == ["linear", "elastic_net"]

    # With that newer lock in force the revert of the withdrawal is still refused, and pressing Fit
    # is no longer offered: a press records a lock only while the plan is unlocked.
    count = len(view_of(client, pid)["decisions"])
    refused = client.post(f"/api/projects/{pid}/decisions",
                          json={"kind": "revert", "decision_id": withdrawal["id"]})
    assert refused.status_code == 409, refused.text
    error = refused.json()["error"]
    assert error["code"] == "lock_stays_withdrawn" and error["exits"] == []
    v = view_of(client, pid)
    assert len(v["decisions"]) == count and v["state"]["plan_locked"] is True


@pytest.mark.slow
def test_down_an_older_engines_chain_a_client_revert_moves_no_lock(client):
    """An older engine accepted a client's revert of TurboTab's withdrawal (the reinstatement), so
    its log holds the lock in force again over the models changed since, no estimate shown under
    it. The client's revert of the reinstatement would withdraw that lock: it is refused, in words
    that do not say an estimate was displayed, and a second revert of the withdrawal hears the log's
    own refusal, without the refused revert as its exit. Cancel, which the words name, withdraws
    the lock (TurboTab's record); then no lock is in force either way, so the same revert of the
    reinstatement is the log's exit and is recorded, and no lock comes back."""
    drive = _inference_drive(client)
    pid = drive.pid
    service = client.app.state.service
    drive.decide({"kind": "select_models", "models": ["linear"]})
    wait_for(client, pid, {"fit": "fresh"}, timeout=240)
    press(client, pid)
    lock = view_of(client, pid)["decisions"][-1]
    assert lock["decision"]["kind"] == "lock_plan"
    drive.decide({"kind": "select_models", "models": ["linear", "elastic_net"]})
    withdrawal = view_of(client, pid)["decisions"][-2]
    assert withdrawal["decision"] == {"kind": "revert", "decision_id": lock["id"]}
    assert withdrawal["recorded_by"] == "turbotab"
    reinstated = service.log(pid).append(Revert(decision_id=withdrawal["id"]))  # an older engine's
    v = view_of(client, pid)
    assert v["state"]["plan_locked"] is True and v["state"]["models"] == ["linear", "elastic_net"]
    count = len(v["decisions"])

    def posted(target: str) -> Any:
        return client.post(f"/api/projects/{pid}/decisions",
                           json={"kind": "revert", "decision_id": target})

    def logs_refusal(target: str) -> Refusal:
        """The log's own refusal of a revert of ``target`` (``DecisionLog.append``'s check)."""
        with pytest.raises(Refusal) as refused:
            decisions._check_revert(service.log(pid).records(), Revert(decision_id=target))
        return refused.value

    refused = posted(reinstated.id)
    assert refused.status_code == 409, refused.text
    error = refused.json()["error"]
    assert error["code"] == "lock_stays_in_force" and error["exits"] == []
    assert "displayed" not in error["message"] and "were seen" not in error["message"]
    assert "when Fit is canceled or the plan is changed" in error["message"]

    logs = logs_refusal(withdrawal["id"])
    assert logs.code == "already_reverted"
    assert [e["decision"] for e in logs.exits] == [
        {"kind": "revert", "decision_id": reinstated.id}]  # the revert just refused
    refused = posted(withdrawal["id"])
    assert refused.status_code == 409, refused.text
    error = refused.json()["error"]
    assert (error["code"], error["message"], error["exits"]) == (logs.code, logs.message, [])
    v = view_of(client, pid)
    assert len(v["decisions"]) == count and v["state"]["plan_locked"] is True

    canceled = client.post(f"/api/projects/{pid}/fit/cancel")
    assert canceled.status_code == 200, canceled.text
    v = view_of(client, pid)
    assert [(r["decision"], r["recorded_by"]) for r in v["decisions"][count:]] == [
        ({"kind": "revert", "decision_id": lock["id"]}, "turbotab")]
    assert v["state"]["plan_locked"] is None
    count = len(v["decisions"])

    logs = logs_refusal(withdrawal["id"])
    refused = posted(withdrawal["id"])
    assert refused.status_code == 409, refused.text
    error = refused.json()["error"]
    assert (error["code"], error["message"]) == ("already_reverted", logs.message)
    assert [e["decision"] for e in error["exits"]] == [
        {"kind": "revert", "decision_id": reinstated.id}]
    v = revert(client, pid, {"id": reinstated.id})
    assert len(v["decisions"]) == count + 1 and v["state"]["plan_locked"] is None
    assert v["decisions"][-1]["decision"] == {"kind": "revert", "decision_id": reinstated.id}
    assert v["state"]["models"] == ["linear", "elastic_net"]
    plan = client.get(f"/api/projects/{pid}/plan").json()
    assert plan["status"] == "declared" and plan["plan"]["models"] == ["linear", "elastic_net"]


# ── 2. a change the log refuses leaves no withdrawal (INBOX 1045) ────────────


@pytest.mark.slow
def test_a_change_the_log_refuses_leaves_no_withdrawal_behind(client):
    """An older engine accepted a revert of TurboTab's withdrawal, so its log holds a lock in force
    over a plan changed since, no estimate yet shown under it. A change then is refused by the log
    itself (a revert of an answer already reverted): the lock is not withdrawn for a change that
    is never recorded, so the log is as it was."""
    drive = _inference_drive(client)
    pid = drive.pid
    service = client.app.state.service
    drive.decide({"kind": "select_models", "models": ["linear"]})
    # an answer given again and reverted, so a second revert of it is the log's to refuse
    drive.decide({"kind": "set_missing", "strategy": "complete_case"})
    again = view_of(client, pid)["decisions"][-1]
    assert again["decision"]["kind"] == "set_missing"
    revert(client, pid, again)
    wait_for(client, pid, {"fit": "fresh"}, timeout=240)
    press(client, pid)
    lock = view_of(client, pid)["decisions"][-1]
    assert lock["decision"]["kind"] == "lock_plan"
    drive.decide({"kind": "select_models", "models": ["linear", "elastic_net"]})
    withdrawal = view_of(client, pid)["decisions"][-2]
    assert withdrawal["decision"] == {"kind": "revert", "decision_id": lock["id"]}
    service.log(pid).append(Revert(decision_id=withdrawal["id"]))  # as the older engine took it
    v = view_of(client, pid)
    assert v["state"]["plan_locked"] is True and v["state"]["models"] == ["linear", "elastic_net"]
    count = len(v["decisions"])

    refused = client.post(f"/api/projects/{pid}/decisions",
                          json={"kind": "revert", "decision_id": again["id"]})
    assert refused.status_code == 409, refused.text
    assert refused.json()["error"]["code"] == "already_reverted"
    v = view_of(client, pid)
    assert len(v["decisions"]) == count  # no withdrawal recorded for the refused change
    assert v["state"]["plan_locked"] is True


# ── 3. a revert back to inference after estimates under prediction locks (INBOX 1045) ──


def test_a_revert_back_to_inference_after_estimates_under_prediction_locks_the_plan(
        client, tmp_path):
    """The goal declared inference, then changed to prediction; reverting that change makes it
    inference again. Before any estimate was shown nothing locks; after the fit's coefficients were
    shown under prediction the plan locks at once, its record saying where they were shown, as a
    new answer of the goal does (``test_routing_gate_repair``'s test 3)."""
    from turbotab.core.tests.acceptance.server_drive import open_project
    from turbotab.core.tests.acceptance.test_routing_gate_repair import (
        SODIUM_ROLES, SODIUM_TRUTH, _sodium)
    from turbotab.core.tests.truths import Truth

    path = tmp_path / "purpose_revert.csv"
    _sodium().to_csv(path, index=False)
    drive = open_project(client, path, Truth(SODIUM_TRUTH, fixture="the sodium table"))
    pid = drive.pid
    drive.decide({"kind": "set_lens", "lenses": ["clinical"]})
    drive.reach("target")
    drive.decide({"kind": "set_target", "column": "sbp"})
    drive.reach("purpose")
    drive.decide({"kind": "set_purpose", "purpose": "inference"})
    drive.decide({"kind": "set_purpose", "purpose": "prediction"})
    drive.answer("grain", {"kind": "set_grain", "grain": "one_row_per_unit"})
    drive.reach("roles")
    drive.decide_roles(SODIUM_ROLES)
    drive.answer("exclusions", {"kind": "set_exclusions", "rules": []})
    drive.answer("missing", {"kind": "set_missing", "strategy": "complete_case"})
    drive.answer("split", {"kind": "set_split", "holdout": 0.0, "seed": 0, "folds": 5})
    drive.reach("models")
    drive.decide({"kind": "select_models", "models": ["linear"]})

    def goal_records() -> list[dict]:
        return [r for r in view_of(client, pid)["decisions"]
                if r["decision"]["kind"] == "set_purpose"]

    # Silent: nothing was shown under prediction, so the revert back to inference locks nothing.
    undone = revert(client, pid, goal_records()[-1])
    assert undone["state"]["purpose"] == "inference" and undone["state"]["plan_locked"] is None
    assert "lock_plan" not in [r["decision"]["kind"] for r in undone["decisions"]]
    drive.decide({"kind": "set_purpose", "purpose": "prediction"})

    # The fit's coefficients shown under prediction (Fit pressed, the fit read).
    drive.artifact("fit")
    fit = client.get(f"/api/projects/{pid}/stages/fit").json()["artifact"]
    linear = next(m for m in fit["models"] if m["family"] == "linear")
    assert any(r.get("estimate") is not None for r in linear["coefficients"] or [])
    before = len(view_of(client, pid)["decisions"])

    goal = goal_records()[-1]
    revert(client, pid, goal)
    added = view_of(client, pid)["decisions"][before:]
    assert [r["decision"]["kind"] for r in added] == ["revert", "lock_plan"]
    switch, lock = added
    assert switch["decision"]["decision_id"] == goal["id"]
    assert lock["decision"]["seen"] == "prediction"
    assert lock["decision"]["seen_target"] == "sbp" and lock["recorded_by"] == "turbotab"
    state = view_of(client, pid)["state"]
    assert state["purpose"] == "inference" and state["plan_locked"] is True


# ── 4. a press a lock check refuses keeps no press (INBOX 1016) ──────────────


def _sodium_to_models(client, tmp_path, name: str) -> Any:
    """The sodium table of ``test_routing_gate_repair``, under inference, up to its models."""
    from turbotab.core.tests.acceptance.server_drive import open_project
    from turbotab.core.tests.acceptance.test_routing_gate_repair import (
        SODIUM_TRUTH, _sodium, _to_models)
    from turbotab.core.tests.truths import Truth

    path = tmp_path / f"{name}.csv"
    _sodium().to_csv(path, index=False)
    drive = open_project(client, path, Truth(SODIUM_TRUTH, fixture="the sodium table"))
    _to_models(drive, "inference")
    drive.answer("exclusions", {"kind": "set_exclusions", "rules": []})
    drive.answer("missing", {"kind": "set_missing", "strategy": "complete_case"})
    drive.answer("split", {"kind": "set_split", "holdout": 0.0, "seed": 0, "folds": 5})
    drive.answer_plan("sodium_g")
    drive.reach("models")
    drive.decide({"kind": "select_models", "models": ["linear"]})
    return drive


def test_a_press_any_lock_check_refuses_keeps_no_press_and_an_accepted_one_keeps_both(
        client, tmp_path, monkeypatch):
    """Whichever check of the lock refuses it (here one constructed for the test, registered
    beside the lock's own), the press is refused and kept nowhere; with the check gone, the press
    is accepted and records both the press and the lock."""
    drive = _sodium_to_models(client, tmp_path, "press_refused")
    pid = drive.pid
    pdir = client.app.state.service.workspace.project_dir(pid)

    def refuse(decision: Any, ctx: Any) -> None:
        raise Refusal("constructed_lock_refusal", "Refused by the test's own check.")

    monkeypatch.setitem(decisions._VALIDATORS, "lock_plan",
                        [*decisions._VALIDATORS["lock_plan"], refuse])
    pressed = client.post(f"/api/projects/{pid}/fit")
    assert pressed.status_code == 409, pressed.text[:600]
    assert pressed.json()["error"]["code"] == "constructed_lock_refusal"
    assert fit_press.read_press(pdir) is None
    v = view_of(client, pid)
    assert v["state"]["plan_locked"] is None
    assert "lock_plan" not in [r["decision"]["kind"] for r in v["decisions"]]
    quest = client.get(f"/api/projects/{pid}/quest").json()["fit"]
    assert quest["pressed"] is False and quest["opened"] is False

    monkeypatch.undo()
    report = press(client, pid)
    assert report["pressed"] is True and report["locked"] is True
    assert fit_press.read_press(pdir)["target"] == "sbp"
    locks = [r for r in view_of(client, pid)["decisions"] if r["decision"]["kind"] == "lock_plan"]
    assert len(locks) == 1 and locks[0]["recorded_by"] == "turbotab"


@pytest.mark.slow
def test_a_press_a_lock_check_refuses_keeps_no_press(client, tmp_path):
    """The 150-row wide table of ``core/tests/test_imputation_width.py`` (about 90 model columns),
    under inference, with multiple imputation written to the log as a log made before the width
    check holds it (Sullivan 2021's runs 22 and 26): the press is refused by the lock's width
    check, and no press is kept, so Fit reads as not pressed. Complete cases taken, the press is
    accepted and records both the press and the lock."""
    from turbotab.core.tests.test_imputation_width import wide  # imports smcfcs: its check
    from turbotab.server.tests.test_imputation_width_api import ROLES

    service = client.app.state.service
    path = tmp_path / "wide.csv"
    wide(150).to_csv(path, index=False)
    pid = open_by_path(client, path)
    declare(pid, {"exposure:y": "x", **{f"adjust:{k}": "yes,yes,no" for k, r in ROLES.items()
                                        if r == "covariate"}}, fixture="wide.csv")
    wait_for(client, pid, {"ingest": "fresh"})

    def post(body: dict[str, Any], walk: bool = True) -> Any:
        if walk:
            prepare(client, pid, body, timeout=300)
        answered = client.post(f"/api/projects/{pid}/decisions", json=body)
        assert answered.status_code == 200, (body["kind"], answered.text[:600])
        return answered

    for body in ({"kind": "set_lens", "lenses": ["clinical"]},
                 {"kind": "set_target", "column": "y"},
                 {"kind": "set_purpose", "purpose": "inference"},
                 {"kind": "set_roles", "roles": ROLES}):
        post(body)
    for col in ("x", "g", "smoke"):  # proposed below high confidence: one press each
        post({"kind": "confirm_readings", "items": [
            {"reading": "role", "column": col, "value": ROLES[col]}]}, walk=False)
    post({"kind": "set_missing", "strategy": "complete_case"})
    post({"kind": "select_models", "models": ["linear"]})
    service.log(pid).append(SetMissing(strategy="multiple_imputation"))  # as runs 22 and 26
    service.engine.on_decision(pid)
    end = time.monotonic() + 120
    while any(s["status"] in ("open", "waiting") for s in view_of(client, pid)["interview"]):
        assert time.monotonic() < end, "the questions never settled after the log's answer"
        time.sleep(0.05)
    pdir = service.workspace.project_dir(pid)
    assert fit_press.read_press(pdir) is None  # nothing pressed yet

    pressed = client.post(f"/api/projects/{pid}/fit")
    assert pressed.status_code == 409, pressed.text[:600]
    assert pressed.json()["error"]["code"] == "imputation_too_wide"
    assert fit_press.read_press(pdir) is None
    assert "lock_plan" not in [r.decision.kind for r in service.log(pid).records()]
    quest = client.get(f"/api/projects/{pid}/quest").json()["fit"]
    assert quest["pressed"] is False and quest["opened"] is False and quest["locked"] is False

    post({"kind": "set_missing", "strategy": "complete_case"}, walk=False)  # the way forward
    quest = client.get(f"/api/projects/{pid}/quest").json()["fit"]
    assert quest["pressed"] is False and quest["opened"] is False
    report = press(client, pid)
    assert report["pressed"] is True and report["locked"] is True
    assert fit_press.read_press(pdir)["target"] == "y"
    assert [r.decision.kind for r in service.log(pid).records()].count("lock_plan") == 1
