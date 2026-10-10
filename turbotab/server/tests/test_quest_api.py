"""P0.4 · the quest log on the server (``GET /projects/{pid}/quest``), and its reopen reasons on the
NHANES journey (sugar → fasting glucose, under Estimate).

The reasons are checked against changes a person makes on that journey: a column's role confirmed
in Your data asks the adjustment set again in Models, and stays the reason when an unrelated rule is
recorded next; an adjustment answer in Models changes who is in, so Who's in's participant flow is
out of date (crosswalk disagreement 7); after the fit, an eligibility rule in Who's in leaves
Models' cards and the estimates out of date, and Results stays reached to say so. A result is out
of date only until it is computed again, which can take less time than a second request, so those
read the log as it stood when the change was recorded: with the stage statuses the decision's own
response reports.
"""
from __future__ import annotations

import time
from typing import Any

import pytest

from turbotab.core.graph import StageStatus
from turbotab.core.quest import QUEST_VERSION
from turbotab.core.tests.acceptance.server_drive import (Drive, answer_plan, open_project,
                                                        settle_post)
from turbotab.core.tests.stage_harness import NHANES
from turbotab.core.tests.truths import fixture_truth
from turbotab.server.tests.conftest import open_by_path, wait_for

SEVEN = [("data", "Your data"), ("question", "Your question"), ("first_look", "First look"),
         ("whos_in", "Who's in"), ("models", "Models"), ("results", "Results"),
         ("writeup", "Write-up")]


def test_the_quest_log_is_served_versioned_with_the_seven_stages_in_order(client):
    pid = open_by_path(client)
    # Which way round the table is reads the oriented stage: while it computes, whether that
    # question is asked at all is not known, so it is not counted yet.
    wait_for(client, pid, {"ingest": "fresh", "oriented": "fresh"})
    r = client.get(f"/api/projects/{pid}/quest")
    assert r.status_code == 200, r.text
    log = r.json()
    assert log["version"] == QUEST_VERSION == 5
    assert [(s["key"], s["name"]) for s in log["stages"]] == SEVEN
    # Nothing is locked, and the log says why (P0.8): no goal is chosen yet.
    assert log["fit"]["purpose"] is None and log["fit"]["locked"] is False
    assert "goal" in log["fit"]["reason"]
    lens = next(l for l in log["stages"][0]["lines"] if l["key"] == "lens")
    assert lens["status"] == "open" and lens["id"] == "q:lens"
    # Your data counts the Router's questions there that apply (the lens, which way round the
    # table is, the roles); the offers (a join, a codebook) are listed and not counted.
    asked = [s for s in client.get(f"/api/projects/{pid}").json()["interview"]
             if s["key"] in ("lens", "orientation", "roles") and s["status"] != "not_applicable"]
    assert log["stages"][0]["reached"] and log["stages"][0]["progress"] == {
        "answered": 0, "required": len(asked), "complete": False}
    # Nothing past Your data is reached: each shows empty, never "0 of N".
    assert all(not s["reached"] and s["progress"] is None for s in log["stages"][1:])
    assert log["kinds"]["set_split"] == "whos_in" and log["kinds"]["select_models"] == "models"
    assert client.get("/api/projects/no-such-project/quest").status_code == 404


# ── the NHANES journey ───────────────────────────────────────────────────────

needs_nhanes = pytest.mark.skipif(not NHANES.is_file(),
                                  reason="the NHANES export is not on this machine")
COVARIATES = ["age", "gender", "cycle_begin_year", "weight", "height", "bmi", "waist", "bp_sys",
              "bp_di", "hdl", "meds_hbp", "meds_chol"]
NUTRIENTS = ["protein", "carb", "fat_total", "fat_sat", "fat_mon", "fat_poly"]
# Triglycerides start left out; confirming them as a covariate later is the change in Your data.
ROLES = {"SEQN": "identifier", "sugar": "exposure", "kcal": "energy",
         **{c: "exposure" for c in NUTRIENTS}, **{c: "covariate" for c in COVARIATES},
         "triglycerides": "excluded",
         **{c: "flag" for c in ("imputed_weight", "imputed_height", "imputed_bmi",
                                "imputed_waist", "imputed_bp_sys", "imputed_bp_di")}}


def settle(drive: Drive, timeout: float = 300.0) -> None:
    """Wait until no stage is computing or about to (a fit the scheduler holds for Fit is not
    about to: it waits for the press)."""
    end = time.monotonic() + timeout
    while True:
        stages = drive.view()["stages"].values()
        if not any(s["status"] in ("queued", "running", "stale") and s.get("held") != "fit"
                   for s in stages):
            return
        assert not any(s["status"] == "error" for s in stages), [s for s in stages
                                                                 if s["status"] == "error"]
        assert time.monotonic() < end, [s["stage"] for s in stages if s["status"] != "fresh"]
        time.sleep(0.1)


def quest(drive: Drive) -> dict[str, dict[str, Any]]:
    log = drive.c.get(f"/api/projects/{drive.pid}/quest").json()
    return {s["key"]: s for s in log["stages"]}


def at_the_change(drive: Drive, body: dict[str, Any], monkeypatch: Any
                  ) -> tuple[str, dict[str, dict[str, Any]]]:
    """Record ``body``; return its record id and the quest log as it stood when it was recorded,
    read by the service with the stage statuses the decision's response reports (a result is out of
    date from then until it is computed again)."""
    r = settle_post(drive.c, drive.pid, body, drive.truth)
    assert r.status_code == 200, r.text[:600]
    view = r.json()
    then = {name: StageStatus.model_validate(s) for name, s in view["stages"].items()}
    service = drive.c.app.state.service
    live = service.engine.status
    with monkeypatch.context() as m:
        m.setattr(service.engine, "status", lambda pid: then if pid == drive.pid else live(pid))
        log = service.quest(drive.pid).model_dump(mode="json")
    return view["decisions"][-1]["id"], {s["key"]: s for s in log["stages"]}


def out_of_date(changed_in: str, n: int, here: str) -> str:
    results = "result" if n == 1 else "results"
    return f"Your change to {changed_in} made {n} {results} in {here} out of date."


def rule(low: float, high: float) -> dict[str, Any]:
    return {"kind": "set_exclusions", "rules": [
        {"column": "kcal", "low": low, "high": high, "reason": "implausible intake"}]}


@pytest.mark.slow
@needs_nhanes
def test_reopen_reasons_follow_real_changes_on_the_nhanes_journey(client, monkeypatch):
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
    settle(drive)

    stages = quest(drive)
    assert all(s["reopened"] == [] for s in stages.values())  # nothing has changed since
    assert [stages[k]["reached"] for k, _ in SEVEN] == [True] * 5 + [False] * 2  # no estimate yet
    assert stages["results"]["progress"] is None
    models_before = stages["models"]["progress"]
    adjustment = next(l for l in stages["models"]["lines"] if l["key"] == "adjustment")
    assert adjustment["status"] == "answered" and adjustment["reopened_by"] is None

    # 1 · Your data → Models: the new covariate has no answers, so the adjustment set is asked
    # again, and Models drops back with the reason until it is answered.
    drive.decide({"kind": "confirm_role", "column": "triglycerides", "role": "covariate"})
    settle(drive)
    stages = quest(drive)
    adjustment = next(l for l in stages["models"]["lines"] if l["key"] == "adjustment")
    assert adjustment["status"] == "open"
    assert adjustment["reopened_by"]["stage"] == "data"
    [reason] = stages["models"]["reopened"]
    assert reason["changed_in"] == "data" and reason["questions"] == ["q:adjustment"]
    assert reason["decision_id"] == adjustment["reopened_by"]["decision_id"]
    assert reason["kind"] in ("confirm_role", "confirm_readings")
    assert reason["sentence"] == "Your change to Your data reopened 1 question in Models."
    assert stages["models"]["progress"]["answered"] == models_before["answered"] - 1
    assert all(s["reopened"] == [] for k, s in stages.items() if k != "models")

    # A rule recorded next in Who's in changes nothing the adjustment set rests on: Your data's
    # change is still the one named. (Posted as it is: the drive's ``decide`` would answer the open
    # adjustment question on the way.)
    r = settle_post(drive.c, drive.pid, rule(600, 4500), drive.truth)
    assert r.status_code == 200, r.text[:600]
    settle(drive)
    stages = quest(drive)
    assert stages["models"]["reopened"] == [reason]

    # 2 · Models → Who's in (disagreement 7): the answer changes what the participant flow reads,
    # so the flow drawn before it is out of date until it is drawn again.
    answer, stages = at_the_change(drive, {
        "kind": "set_adjustment", "exposure": "sugar", "answers": {"triglycerides": {
            "causes_exposure": "no", "causes_outcome": "yes", "after_exposure": "yes"}}},
        monkeypatch)
    [flow] = [r for r in stages["whos_in"]["reopened"] if r["decision_id"] == answer]
    assert flow["changed_in"] == "models" and "cohort" in flow["results"]
    assert flow["sentence"] == out_of_date("Models", len(flow["results"]), "Who's in")
    # Models' own answer gives Models no reason line, and it answers the question asked again.
    assert all(r["decision_id"] != answer for r in stages["models"]["reopened"])
    settle(drive)
    stages = quest(drive)
    assert all(s["reopened"] == [] for s in stages.values())
    assert stages["models"]["progress"] == models_before

    # 3 · The fit: computed, but Results opens only when Fit is pressed, which locks the plan
    # (P0.8); then Results is reached and reads 0 of 5, the exhibits the engine serves on this
    # plan, none decided (Q-c; calm/FOUNDATION §3, §8): usual intake (the dietary lens), the
    # performance table, the further-adjusted model, Table 2 and the decision curve; nothing
    # declares sensitivity, calibration, scales, the causal lane, a unit, modifiers or explain.
    # Write-up waits for Results to be placed.
    drive.answer("energy_adjustment", {"kind": "set_energy_adjustment", "method": "standard",
                                       "energy_column": "kcal", "nutrients": ["sugar"]})
    drive.decide({"kind": "select_models", "models": ["linear"]})
    settle(drive)
    stages = quest(drive)
    assert drive.view()["stages"]["fit"]["status"] == "fresh"
    assert not stages["results"]["reached"] and not stages["writeup"]["reached"]
    log = drive.c.get(f"/api/projects/{drive.pid}/quest").json()
    assert log["fit"]["locks"] is True and log["fit"]["locked"] is False
    assert "Pressing Fit locks it" in log["fit"]["reason"]
    assert drive.press_fit()
    assert drive.view()["state"]["plan_locked"] is True
    # Results opens on the estimates: once one is served the lock stands (before, a change to the
    # plan would withdraw it; calm/FOUNDATION §7).
    from turbotab.core.plan_lock import shows_estimates

    served = drive.c.get(f"/api/projects/{drive.pid}/stages/fit").json()["artifact"]
    assert shows_estimates("fit", served)
    stages = quest(drive)
    assert all(s["reopened"] == [] for s in stages.values())
    # Under Estimate Results counts the Estimate exhibits the engine serves (TRUST; the
    # crosswalk's goals): Table 2 (effects) and Model 3 (secondary). Usual intake is Describe's,
    # the performance table and the decision curve Predict's.
    assert stages["results"]["reached"] and stages["results"]["progress"] == {
        "answered": 0, "required": 2, "complete": False}
    assert not stages["writeup"]["reached"] and stages["writeup"]["progress"] is None

    # 4 · Who's in → Models and Results: an eligibility rule leaves the shelf, the cards and the
    # estimates out of date. Results stays reached and says why (with Fit held, until Fit is
    # pressed again).
    changed, stages = at_the_change(drive, rule(700, 4000), monkeypatch)
    [shelf] = [r for r in stages["models"]["reopened"] if r["decision_id"] == changed]
    assert shelf["changed_in"] == "whos_in" and "shelf" in shelf["results"]
    assert shelf["sentence"] == out_of_date("Who's in", len(shelf["results"]), "Models")
    results = stages["results"]
    assert results["reached"] and not stages["writeup"]["reached"]
    assert results["progress"] == {"answered": 0, "required": 2, "complete": False}
    [estimates] = results["reopened"]
    assert estimates["decision_id"] == changed and estimates["changed_in"] == "whos_in"
    # the reason names Results' own objectives: Table 2, never the Predict exhibits
    assert "effects" in estimates["results"]
    assert not {"fit", "evaluation", "usual_intake"} & set(estimates["results"])
    assert estimates["sentence"] == out_of_date("Who's in", len(estimates["results"]), "Results")
    settle(drive)
    stages = quest(drive)
    assert all(s["reopened"] == [] for s in stages.values())
    assert stages["results"]["reached"]
