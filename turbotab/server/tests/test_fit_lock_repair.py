"""P0.8's repair over HTTP: what no response may carry before Fit, and a lock that says only what
happened (SIZING P0.8; RECIPES_AND_TUNING §4.4; calm/FOUNDATION §7 and §10; guarantee test 2 of
EXTERNAL_AUDIT_2026-10-09).

* A question gate under prediction (the follow-up) and the gate of an unanswered purpose withhold
  every cross-validated score, as the gate before Fit does: whichever gate speaks first, no score
  is served before Fit.
* Under prediction the held-out rows cannot be opened before Fit, and a refusal quotes no
  cross-validated score before it; once a refusal quotes them, they count as seen.
* Under prediction a preview draws nothing read from the outcome model before Fit.
* Under Estimate the explore stage's relationship points wait for the plan's lock.
* A lock that no estimate was shown under is withdrawn when the plan changes, and by Cancel, the
  withdrawal recorded by TurboTab with its reason and what an accepted press of Fit does (under
  prediction, after a change of the goal, nothing locks; a press refused after it records no
  lock), and the change is not marked as made after the estimates were seen.
* Results stays reached when the goal is withdrawn after Fit, and says why it dropped back.

Each expectation is read from the data or the record independently of the code under test: the
scores are the fit's own as computed (``engine.get``), the families quoted are those named in the
refusal's exits, the plan is the lock's own record.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from turbotab.server.tests.conftest import answer_settled, prepare, wait_for
from turbotab.server.tests.test_seal_api import decide, sealed_project


def stage(client, pid: str, name: str) -> dict:
    return client.get(f"/api/projects/{pid}/stages/{name}").json()["artifact"]


def press(client, pid: str, status: int = 200) -> dict:
    response = client.post(f"/api/projects/{pid}/fit")
    assert response.status_code == status, response.text
    return response.json()


def numbers_under(obj: Any, names: set[str]) -> list[float]:
    """Every number held under a key in ``names``, anywhere in ``obj``."""
    out: list[float] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in names and v is not None:
                out += [x for x in _numbers(v)]
            else:
                out += numbers_under(v, names)
    elif isinstance(obj, list):
        for v in obj:
            out += numbers_under(v, names)
    return out


def _numbers(obj: Any) -> list[float]:
    if isinstance(obj, bool):
        return []
    if isinstance(obj, (int, float)):
        return [float(obj)]
    if isinstance(obj, dict):
        return [x for v in obj.values() for x in _numbers(v)]
    if isinstance(obj, list):
        return [x for v in obj for x in _numbers(v)]
    return []


def record_of(client, pid: str, kind: str) -> dict:
    return next(r for r in client.get(f"/api/projects/{pid}").json()["decisions"]
                if r["decision"]["kind"] == kind)


def revert(client, pid: str, record: dict) -> dict:
    r = client.post(f"/api/projects/{pid}/decisions",
                    json={"kind": "revert", "decision_id": record["id"]})
    assert r.status_code == 200, r.text
    return r.json()


# ── 1. a question gate under prediction withholds the scores too ─────────────


def _staggered_csv(folder: Path) -> Path:
    from turbotab.core.tests.acceptance.test_wp12b_cox_mixed_gee import _staggered_entry_cohort

    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "cohort_tte_null.csv"
    _staggered_entry_cohort().to_csv(path, index=False)
    return path


def test_no_cross_validated_score_is_served_behind_a_question_gate_before_fit(tmp_path):
    """The staggered-entry cohort under prediction, a Cox fit computed and Fit never pressed. The
    follow-up answer reverted, the follow-up question's gate speaks first; the purpose reverted, the
    purpose's. Neither serves a cross-validated score, and nothing is marked seen."""
    from turbotab.core.models.selection import read_seen
    from turbotab.core.tests.acceptance.server_drive import local_server, open_project

    path = _staggered_csv(tmp_path / "data")
    with local_server(tmp_path / "home") as client:
        drive = open_project(client, path)
        pid = drive.pid
        drive.decide({"kind": "set_lens", "lenses": ["clinical"]})
        drive.reach("target")
        drive.decide({"kind": "set_target", "column": "cvd_event"})
        drive.answer("event", {"kind": "set_event", "column": "cvd_event", "level": "1"})
        drive.reach("follow_up")
        drive.decide({"kind": "set_task", "column": "cvd_event", "task": "time_to_event"})
        drive.decide({"kind": "set_follow_up", "column": "cvd_event",
                      "time_column": "followup_years"})
        drive.decide({"kind": "set_purpose", "purpose": "prediction"})
        drive.answer("grain", {"kind": "set_grain", "grain": "one_row_per_unit"})
        drive.reach("roles")
        drive.decide_roles({"participant_id": "identifier", "fiber_g": "covariate",
                            "age": "covariate", "followup_years": "time"})
        drive.answer("exclusions", {"kind": "set_exclusions", "rules": []})
        drive.answer("missing", {"kind": "set_missing", "strategy": "complete_case"})
        drive.answer("split", {"kind": "set_split", "holdout": 0.25, "seed": 0, "folds": 5})
        drive.reach("models")
        drive.decide({"kind": "select_models", "models": ["cox"]})
        wait_for(client, pid, {"fit": "fresh"}, timeout=240)
        raw = client.app.state.service.engine.get(pid, "fit").artifact
        computed = [m["cv"] for m in raw["models"] if m.get("cv")]
        assert computed, "the fit computed no cross-validated score to withhold"

        revert(client, pid, record_of(client, pid, "set_follow_up"))
        held = stage(client, pid, "fit")
        assert held["withheld_for"] == "follow_up"  # the question speaks first
        assert all(not m.get("cv") for m in held["models"]), held["models"]
        assert numbers_under(held, {"cv", "c_index", "brier_t"}) == []
        for name in ("evaluation", "explain"):
            served = stage(client, pid, name) or {}
            assert numbers_under(served, {"cv", "benchmark", "decision_curve", "subgroups",
                                          "c_index"}) == [], name

        revert(client, pid, record_of(client, pid, "set_purpose"))
        assert client.get(f"/api/projects/{pid}").json()["state"]["purpose"] is None
        held = stage(client, pid, "fit")
        assert held["withheld_for"] == "purpose"  # the strictest case speaks first
        assert numbers_under(held, {"cv", "c_index", "brier_t"}) == []
        assert read_seen(client.app.state.service.workspace.project_dir(pid)) == {}


# ── 2. the held-out rows open after Fit, and a quoted score counts as seen ───


def test_the_held_out_rows_open_only_after_fit_and_a_quoted_score_counts_as_seen(client):
    from turbotab.core.fit_press import read_press
    from turbotab.core.models.selection import read_seen

    service = client.app.state.service
    pid = sealed_project(client)
    pdir = service.workspace.project_dir(pid)
    families = ["linear", "elastic_net"]
    decide(client, pid, {"kind": "select_models", "models": families})
    wait_for(client, pid, {"fit": "fresh", "split": "fresh", "seal_plan": "fresh"}, timeout=180)
    prepare(client, pid, {"kind": "open_seal"})  # the Router's questions before the opening
    assert read_press(pdir) is None  # answering them pressed nothing
    records = len(client.get(f"/api/projects/{pid}").json()["decisions"])

    for body in ({"kind": "open_seal"}, {"kind": "open_seal", "family": "elastic_net"}):
        refused = client.post(f"/api/projects/{pid}/decisions", json=body)
        assert refused.status_code == 409, refused.text
        error = refused.json()["error"]
        assert error["code"] == "fit_not_yet", error
        assert "CV" not in refused.text and "MSE" not in refused.text
        assert any("Press Fit" in e["label"] for e in error["exits"])
        previewed = client.post(f"/api/projects/{pid}/preview", json=body)
        assert previewed.status_code == 409 and "CV" not in previewed.text
    assert len(client.get(f"/api/projects/{pid}").json()["decisions"]) == records
    assert read_seen(pdir) == {}

    press(client, pid)
    assert read_seen(pdir) == {}  # pressed, nothing served yet
    refused = client.post(f"/api/projects/{pid}/decisions", json={"kind": "open_seal"})
    assert refused.status_code == 409 and refused.json()["error"]["code"] == "final_model_needed"
    quoted = sorted(e["decision"]["family"] for e in refused.json()["error"]["exits"]
                    if "(CV " in e["label"])
    assert quoted == sorted(families)
    assert sorted(read_seen(pdir).get("hba1c", [])) == quoted  # quoted, so seen

    # The open noticings are triaged as the opening offers (``seal``'s gate, tested in
    # ``test_opening_gate_api``), then the rows open.
    opened = answer_settled(client, pid, None, {"kind": "open_seal", "family": "elastic_net"})
    assert opened.status_code == 200, opened.text


# ── 3. under prediction a preview draws nothing from the outcome model before Fit ──


def test_under_prediction_a_preview_reads_nothing_from_the_outcome_model_before_fit(client):
    pid = sealed_project(client)
    decide(client, pid, {"kind": "select_models", "models": ["linear"]})
    wait_for(client, pid, {"fit": "fresh"}, timeout=180)
    body = {"kind": "set_updating", "method": "shrinkage"}
    before = client.post(f"/api/projects/{pid}/preview", json=body)
    assert before.status_code == 200, before.text
    assert before.json()["views"] == []
    assert "Fit is pressed" in before.text and "answered with prediction" not in before.text

    press(client, pid)
    after = client.post(f"/api/projects/{pid}/preview", json=body)
    assert after.status_code == 200 and after.json()["views"] != []


# ── 4. under Estimate the relationship points wait for the lock ──────────────


def _inference_drive(client) -> Any:
    from turbotab.core.tests.acceptance.server_drive import answer_plan, open_project
    from turbotab.core.tests.stage_harness import NHANES
    from turbotab.core.tests.truths import fixture_truth
    from turbotab.server.tests.test_quest_api import ROLES, rule

    drive = open_project(client, NHANES, fixture_truth("_tt_tmp_nhanes.csv"))
    drive.decide({"kind": "set_lens", "lenses": ["dietary"]})
    drive.reach("target")
    drive.decide({"kind": "set_target", "column": "glucose"})
    drive.reach("purpose")
    drive.decide({"kind": "set_purpose", "purpose": "inference"})
    drive.answer("grain", {"kind": "set_grain", "grain": "one_row_per_unit"})
    drive.reach("roles")
    drive.decide_roles(ROLES)
    drive.answer("exclusions", rule(500, 5000))
    drive.answer("missing", {"kind": "set_missing", "strategy": "complete_case"})
    drive.answer("split", {"kind": "set_split", "holdout": 0.0, "seed": 0, "folds": 5})
    answer_plan(drive, "sugar")
    drive.answer("energy_adjustment", {"kind": "set_energy_adjustment", "method": "standard",
                                       "energy_column": "kcal", "nutrients": ["sugar"]})
    return drive


def relationships(explore: dict) -> list[dict]:
    return [f for f in explore.get("findings") or [] if f["kind"] == "outcome_relationship"]


def test_under_estimate_the_relationship_points_wait_for_the_lock(client):
    drive = _inference_drive(client)
    wait_for(client, drive.pid, {"explore": "fresh"}, timeout=240)
    raw = client.app.state.service.engine.get(drive.pid, "explore").artifact
    computed = relationships(raw.data if hasattr(raw, "data") else raw)
    assert computed and all(f["points"] for f in computed)  # the stage computed them

    before = relationships(stage(client, drive.pid, "explore"))
    assert [f["id"] for f in before] == [f["id"] for f in computed]
    assert all(f["points"] == [] for f in before)
    assert all("Fit" in (f.get("detail") or "") for f in before)

    drive.decide({"kind": "select_models", "models": ["linear"]})
    wait_for(client, drive.pid, {"fit": "fresh"}, timeout=240)
    press(client, drive.pid)
    after = relationships(stage(client, drive.pid, "explore"))
    assert [f["points"] for f in after] == [f["points"] for f in computed]


# ── 5. a lock nothing was shown under is withdrawn ───────────────────────────

# What the withdrawal's note says Fit does under inference: press_fit records the lock on a press
# its gates accept when the plan is unlocked, and on no other press (a refused press, or one with
# a lock in force: the goal's return from prediction locks the plan without a press).
INFERENCE_THEN = ("under inference, a press of Fit that is accepted while the plan is unlocked "
                  "locks the plan as it then stands.")


def test_a_lock_no_estimate_was_shown_under_is_withdrawn_by_a_change_and_by_cancel(client):
    drive = _inference_drive(client)
    pid = drive.pid
    drive.decide({"kind": "select_models", "models": ["linear"]})
    wait_for(client, pid, {"fit": "fresh"}, timeout=240)

    def view() -> dict:
        return client.get(f"/api/projects/{pid}").json()

    # Pressed, then Cancel before any estimate is served: the lock is withdrawn, kept in the
    # record, and Fit is asked for again.
    report = press(client, pid)
    assert report["locked"] is True
    first = view()["decisions"][-1]
    assert first["decision"]["kind"] == "lock_plan"
    cancelled = client.post(f"/api/projects/{pid}/fit/cancel")
    assert cancelled.status_code == 200, cancelled.text
    assert cancelled.json()["locked"] is False and cancelled.json()["pressed"] is False
    v = view()
    assert v["state"]["plan_locked"] is None
    withdrawal = v["decisions"][-1]
    assert withdrawal["decision"] == {"kind": "revert", "decision_id": first["id"]}
    assert withdrawal["after_estimates"] is False
    # The server records the withdrawal (plan_lock's rule; no client can revert a lock), and says
    # why in its note: Cancel caused it, and no estimate shown under the lock allowed it (said
    # once). The press's own report was read under this lock above, so the note claims only that
    # no estimate was shown, never that nothing was seen; and it says what an accepted press of
    # Fit does under inference, as the press below does.
    assert withdrawal["recorded_by"] == "turbotab" and (withdrawal["note"] or "").strip()
    assert withdrawal["note"].startswith(
        "Fit was canceled before any estimate was shown under this lock, so TurboTab withdrew it;")
    assert withdrawal["note"].count("before any estimate was shown") == 1
    assert "nothing was seen" not in withdrawal["note"]
    assert INFERENCE_THEN in withdrawal["note"]
    assert any(r["id"] == first["id"] for r in v["decisions"])  # the history stays
    assert stage(client, pid, "fit")["withheld_for"] == "fit"

    # Pressed again, then a change to the plan before any estimate is served: the lock is
    # withdrawn first, and the change is an ordinary one.
    press(client, pid)
    second = view()["decisions"][-1]
    assert second["decision"]["kind"] == "lock_plan" and second["id"] != first["id"]
    drive.decide({"kind": "select_models", "models": ["linear", "elastic_net"]})
    v = view()
    assert v["state"]["plan_locked"] is None
    withdrawn = v["decisions"][-2]
    assert withdrawn["decision"] == {"kind": "revert", "decision_id": second["id"]}
    change = v["decisions"][-1]
    assert change["decision"]["kind"] == "select_models"
    assert change["after_estimates"] is False and withdrawn["after_estimates"] is False
    # The server's withdrawal is its own, with its own reason; the change stays the person's.
    assert withdrawn["recorded_by"] == "turbotab" and (withdrawn["note"] or "").strip()
    assert withdrawn["note"] != withdrawal["note"]
    assert withdrawn["note"].startswith(
        "The plan was changed before any estimate was shown under this lock, so TurboTab "
        "withdrew it;")
    assert withdrawn["note"].count("before any estimate was shown") == 1
    assert "nothing was seen" not in withdrawn["note"]
    assert INFERENCE_THEN in withdrawn["note"]  # the third press below
    assert change["recorded_by"] == "you" and change["note"] is None
    plan = client.get(f"/api/projects/{pid}/plan").json()
    assert plan["status"] == "declared" and plan["after_estimates"] == []

    # Pressed a third time and an estimate served: now the lock stands. Cancel stops the work
    # only, and a change is marked as made after the estimates were seen.
    wait_for(client, pid, {"fit": "fresh"}, timeout=240)
    press(client, pid)
    third = view()["decisions"][-1]
    assert third["decision"]["kind"] == "lock_plan" and third["id"] != second["id"]
    from turbotab.core.plan_lock import shows_estimates

    assert shows_estimates("fit", stage(client, pid, "fit"))
    kept = client.post(f"/api/projects/{pid}/fit/cancel")
    assert kept.status_code == 200 and kept.json()["locked"] is True
    drive.decide({"kind": "select_models", "models": ["linear"]})
    v = view()
    assert v["state"]["plan_locked"] is True and v["decisions"][-1]["after_estimates"] is True
    assert v["decisions"][-2]["id"] == third["id"]  # nothing withdrawn, so the server recorded none
    plan = client.get(f"/api/projects/{pid}/plan").json()
    assert plan["status"] == "locked" and plan["through_record"] == third["seq"]
    assert [r["kind"] for r in plan["after_estimates"]] == ["select_models"]


@pytest.mark.slow
def test_a_change_of_the_goal_to_prediction_withdraws_the_lock_and_its_note_says_nothing_locks(
        client):
    """The goal is part of the plan, so changing it to prediction under a lock nothing was shown
    under withdraws the lock; the note then says what the next press does under prediction, which
    is to lock nothing (the press below records no lock), not to lock the plan anew."""
    drive = _inference_drive(client)
    pid = drive.pid
    drive.decide({"kind": "select_models", "models": ["linear"]})
    wait_for(client, pid, {"fit": "fresh"}, timeout=240)

    def view() -> dict:
        return client.get(f"/api/projects/{pid}").json()

    press(client, pid)
    lock = view()["decisions"][-1]
    assert lock["decision"]["kind"] == "lock_plan"
    changed = client.post(f"/api/projects/{pid}/decisions",
                          json={"kind": "set_purpose", "purpose": "prediction"})
    assert changed.status_code == 200, changed.text
    v = view()
    withdrawn, change = v["decisions"][-2], v["decisions"][-1]
    assert withdrawn["decision"] == {"kind": "revert", "decision_id": lock["id"]}
    assert withdrawn["recorded_by"] == "turbotab" and change["decision"]["kind"] == "set_purpose"
    note = withdrawn["note"] or ""
    assert note.startswith("The plan was changed")
    assert "under prediction nothing locks" in note and "locks the plan" not in note

    # The next press: accepted, and no lock recorded, as the note says.
    report = press(client, pid)
    v = view()
    assert report["locks"] is False and report["locked"] is False
    assert v["state"]["plan_locked"] is None
    assert v["decisions"][-1]["id"] == change["id"]


@pytest.mark.slow
def test_a_press_refused_after_a_withdrawal_records_no_lock_and_the_note_still_holds(client):
    """Reverting the only models answer under a lock nothing was shown under withdraws the lock,
    and leaves nothing to fit, so the next press is refused and records no lock. The note claims a
    lock only for a press that is accepted, so it holds: the refused press locks nothing, and the
    press accepted once models are chosen again locks the plan with those models, as it then
    stands."""
    drive = _inference_drive(client)
    pid = drive.pid
    drive.decide({"kind": "select_models", "models": ["linear"]})
    wait_for(client, pid, {"fit": "fresh"}, timeout=240)

    def view() -> dict:
        return client.get(f"/api/projects/{pid}").json()

    press(client, pid)
    lock = view()["decisions"][-1]
    assert lock["decision"]["kind"] == "lock_plan" and lock["decision"]["plan"]["models"] == [
        "linear"]
    chosen = [r for r in view()["decisions"] if r["decision"]["kind"] == "select_models"]
    assert len(chosen) == 1  # the only models answer, so its revert leaves none
    revert(client, pid, chosen[0])
    v = view()
    withdrawn, change = v["decisions"][-2], v["decisions"][-1]
    assert withdrawn["decision"] == {"kind": "revert", "decision_id": lock["id"]}
    assert withdrawn["recorded_by"] == "turbotab" and change["recorded_by"] == "you"
    assert v["state"]["plan_locked"] is None and v["state"]["models"] is None
    note = withdrawn["note"] or ""
    assert note.startswith("The plan was changed before any estimate was shown under this lock")
    assert INFERENCE_THEN in note and "the next press of Fit locks" not in note

    # Nothing to fit: the press is refused, and no lock is recorded.
    refused = client.post(f"/api/projects/{pid}/fit")
    assert refused.status_code == 409 and refused.json()["error"]["code"] == "fit_not_yet"
    v = view()
    assert v["state"]["plan_locked"] is None
    assert v["decisions"][-1]["id"] == change["id"]
    assert not any(r["decision"]["kind"] == "lock_plan" and r["seq"] > lock["seq"]
                   for r in v["decisions"])

    # Models chosen again: the press is accepted, and it locks the plan as it then stands.
    drive.decide({"kind": "select_models", "models": ["linear", "elastic_net"]})
    wait_for(client, pid, {"fit": "fresh"}, timeout=240)
    report = press(client, pid)
    relocked = view()["decisions"][-1]
    assert report["locked"] is True and relocked["decision"]["kind"] == "lock_plan"
    assert relocked["id"] != lock["id"]
    assert relocked["decision"]["plan"]["models"] == ["linear", "elastic_net"]


# ── 6. Results stays reached when the goal is withdrawn after Fit ────────────


def test_results_stays_reached_with_its_reason_when_the_goal_is_withdrawn_after_fit(client):
    pid = sealed_project(client)
    decide(client, pid, {"kind": "select_models", "models": ["linear"]})
    wait_for(client, pid, {"fit": "fresh"}, timeout=180)
    press(client, pid)
    assert stage(client, pid, "fit").get("withheld") is None

    def results() -> dict:
        quest = client.get(f"/api/projects/{pid}/quest").json()
        return next(s for s in quest["stages"] if s["key"] == "results")

    assert results()["reached"]
    revert(client, pid, record_of(client, pid, "set_purpose"))
    end = time.monotonic() + 60
    while client.get(f"/api/projects/{pid}").json()["stages"]["fit"]["status"] == "fresh":
        assert time.monotonic() < end
        time.sleep(0.05)
    dropped = results()
    assert dropped["reached"] and dropped["reopened"], dropped


# ── 7. the hold covers every estimate stage that fits the chosen families ────


def test_the_hold_covers_every_estimate_stage_that_fits_the_chosen_families(client):
    """A stage that refits the chosen families (a sensitivity analysis, the effects, the scales'
    corrections) costs at least what the fit does, so it waits for Fit with it; a stage that does
    not read them (the usual intake) is not held by their estimate."""
    from types import SimpleNamespace

    from turbotab.core.decisions import ProjectState
    from turbotab.core.estimand import ESTIMATE_STAGES

    service = client.app.state.service
    graph = service.engine.graph
    shelf = {"families": [{"key": "linear", "estimate_seconds": 600.0}]}
    view = SimpleNamespace(pid=sealed_project(client), state=ProjectState(
        target="hba1c", purpose="inference", models=["linear"]),
        artifact=lambda name: shelf if name == "shelf" else None, pending=lambda name: False)
    refit = sorted(s for s in ESTIMATE_STAGES if "models" in graph[s].reads)
    assert {"fit", "sensitivity", "effects", "secondary"} <= set(refit)
    assert {s: service._hold(s, view) for s in refit} == {s: "fit" for s in refit}
    assert service._hold("usual_intake", view) is None
    assert service._hold("profile", view) is None
