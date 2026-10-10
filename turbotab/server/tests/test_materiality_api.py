"""SURFACING_POLICY §2 on the server: the triage carries the noticings measured on the table, and
``GET /projects/{pid}/materiality`` serves the ledger, predicted before the lock and verified after
it, on the NHANES journey (sugar → fasting glucose under Estimate, a 500–5,000 kcal screen with the
every-row analysis declared beside it, Banna et al. 2017).

What is expected comes from outside the ledger: the noticings the policy names for this journey,
the rule that no refit is read before the plan is fixed, and the screen's row count recounted by
pandas on the fixture.
"""
from __future__ import annotations

import time

import pandas as pd

from turbotab.core.tests.acceptance.server_drive import answer_plan, open_project, release
from turbotab.core.tests.stage_harness import NHANES
from turbotab.core.tests.truths import fixture_truth
from turbotab.server.tests.test_confirm_sweep_api import post
from turbotab.server.tests.test_quest_api import ROLES, needs_nhanes, rule, settle

SCREEN = "diet-implausible-reporters"
DAYS = "diet-day-to-day-variance"
ENERGY = "diet-energy-carries-the-nutrient"
COLLINEAR = "shared-collinear-predictors"


def journey(client):
    """Sugar → fasting glucose under Estimate, a 500–5,000 kcal screen, the every-row analysis
    declared beside it, settled before Fit."""
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
    drive.decide({"kind": "select_models", "models": ["linear"]})
    drive.decide({"kind": "set_sensitivity", "analyses": [{"label": "Every row", "rules": []}]})
    settle(drive)
    return drive


@needs_nhanes
def test_the_triage_and_the_ledger_carry_the_measured_noticings_through_the_lock(client):
    drive = journey(client)

    # ── the triage before the lock: the measured noticings beside the findings ──
    t = drive.c.get(f"/api/projects/{drive.pid}/triage").json()
    noticed = {i["id"]: i for i in t["items"] if i["severity"] == "noticing"}
    # Energy's decision is answered: it dropped out. The adjustment terms (kcal beside its parts)
    # are nearly collinear with sugar outside the dependency: exact by theorem, no number moves.
    assert set(noticed) == {SCREEN, DAYS, COLLINEAR}
    assert noticed[COLLINEAR]["recommended"] == "no_change" and noticed[COLLINEAR]["band"] == 0
    assert noticed[COLLINEAR]["family"] == "K5" and not noticed[COLLINEAR]["limitation"]
    screen = noticed[SCREEN]
    kcal = pd.read_csv(NHANES, usecols=["kcal"])["kcal"]
    gone = int(((kcal < 500) | (kcal > 5000)).sum())
    assert f"`{gone:,}` of" in screen["reason"]
    # Few rows leave a 500–5,000 kcal screen and they differ little on what is adjusted for
    # (rows × SMD ≈ 0.019), but the calibration cases show such screens moving the estimate: the
    # rows instrument says it could bias, never that it changes nothing. The every-row analysis
    # is declared, which is something done: no limitation sentence.
    assert screen["family"] == "S5" and screen["measure"].startswith("rows × SMD")
    assert screen["calibrated"] and screen["band"] == 1
    assert screen["recommended"] == "could_bias" and not screen["limitation"]
    assert "Banna" in screen["done"]
    assert noticed[DAYS]["limitation"] and noticed[DAYS]["label"].endswith("(not measurable here)")
    assert sum(r["count"] for r in t["rows"]) == len(t["items"]) and len(t["rows"]) <= 6

    before = drive.c.get(f"/api/projects/{drive.pid}/materiality")
    assert before.status_code == 200, before.text[:600]
    rows = {r["thread"]: r for r in before.json()["rows"]}
    assert not before.json()["locked"]
    assert rows[SCREEN]["verdict"] == "pending" and rows[SCREEN]["realized"] is None
    assert rows[ENERGY]["verdict"] == "not_graded"

    r = post(drive, {"kind": "confirm_sweep", "stage": "models", "sweep": "noticings"})
    assert r.status_code == 200, r.text[:600]

    # ── Fit locks the plan, its digest over the dispositions; the screen is then verified ──
    drive.press_fit()
    drive.artifact("sensitivity")
    state = drive.view()["state"]
    assert state["plan_locked"]
    after = drive.c.get(f"/api/projects/{drive.pid}/materiality").json()
    rows = {r["thread"]: r for r in after["rows"]}
    assert after["locked"]
    assert rows[SCREEN]["recorded"] == "could_bias"
    realized = rows[SCREEN]["realized"]
    assert realized["instrument"] == "sensitivity"
    # The every-row refit moves sugar's coefficient by more than half its half-width (about 0.64):
    # act on it, above the prediction, so the exhibit is relabeled openly.
    assert realized["band"] == 2 and 0.5 <= realized["value"] < 1.0
    assert rows[SCREEN]["verdict"] == "upgraded"
    assert rows[SCREEN]["label"].startswith("Moved more than predicted")
    assert rows[SCREEN]["exhibit"] == "sensitivity"
    assert rows[DAYS]["verdict"] == "not_verifiable" and rows[DAYS]["limitation"]


@needs_nhanes
def test_the_ledger_serves_its_refits_through_the_serving_path_and_the_lock_then_stands(client):
    """The ledger's realized rows quote the refit's estimates, so reading them is an estimate
    shown: they come through the serving path (WP17's gate, the lock), and the lock is marked as
    shown. A plan change after it is recorded as made after the estimates were seen, and the lock
    is not withdrawn as though nothing had been shown."""
    drive = journey(client)
    r = post(drive, {"kind": "confirm_sweep", "stage": "models", "sweep": "noticings"})
    assert r.status_code == 200, r.text[:600]
    assert drive.press_fit()
    end = time.monotonic() + 240
    while True:  # fresh, never served through the stage route
        status = drive.view()["stages"]["sensitivity"]
        release(drive.c, drive.pid, status)
        if status["status"] == "fresh":
            break
        assert status["status"] != "error" and time.monotonic() < end, status
        time.sleep(0.05)
    book = drive.c.get(f"/api/projects/{drive.pid}/materiality").json()
    screen = next(r for r in book["rows"] if r["thread"] == SCREEN)
    assert screen["realized"] is not None and "moved the estimate from" in screen["realized"]["words"]

    r = post(drive, rule(400, 5000))
    assert r.status_code == 200, r.text[:600]
    v = drive.view()
    assert v["state"]["plan_locked"] is True
    assert not any(d["decision"]["kind"] == "revert" for d in v["decisions"])
    change = v["decisions"][-1]
    assert change["decision"]["kind"] == "set_exclusions" and change["after_estimates"] is True
