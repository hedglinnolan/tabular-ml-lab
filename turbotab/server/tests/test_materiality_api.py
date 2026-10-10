"""SURFACING_POLICY §2 on the server: the triage carries the noticings measured on the table, and
``GET /projects/{pid}/materiality`` serves the ledger, predicted before the lock and verified after
it, on the NHANES journey (sugar → fasting glucose under Estimate, a 500–5,000 kcal screen with the
every-row analysis declared beside it, Banna et al. 2017).

What is expected comes from outside the ledger: the noticings the policy names for this journey,
the rule that no refit is read before the plan is fixed, and the screen's row count recounted by
pandas on the fixture.
"""
from __future__ import annotations

import pandas as pd

from turbotab.core.tests.acceptance.server_drive import answer_plan, open_project
from turbotab.core.tests.stage_harness import NHANES
from turbotab.core.tests.truths import fixture_truth
from turbotab.server.tests.test_confirm_sweep_api import post
from turbotab.server.tests.test_quest_api import ROLES, needs_nhanes, rule, settle

SCREEN = "diet-implausible-reporters"
DAYS = "diet-day-to-day-variance"
ENERGY = "diet-energy-carries-the-nutrient"


@needs_nhanes
def test_the_triage_and_the_ledger_carry_the_measured_noticings_through_the_lock(client):
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

    # ── the triage before the lock: the measured noticings beside the findings ──
    t = drive.c.get(f"/api/projects/{drive.pid}/triage").json()
    noticed = {i["id"]: i for i in t["items"] if i["severity"] == "noticing"}
    assert set(noticed) == {SCREEN, DAYS}  # energy's decision is answered: it dropped out
    screen = noticed[SCREEN]
    kcal = pd.read_csv(NHANES, usecols=["kcal"])["kcal"]
    gone = int(((kcal < 500) | (kcal > 5000)).sum())
    assert f"`{gone:,}` of" in screen["reason"]
    # Few rows leave a 500–5,000 kcal screen: below noise on the calibrated rows instrument, so
    # it doesn't change the numbers here, a claim about the design that the lock will check.
    assert screen["family"] == "S5" and screen["measure"].startswith("rows × SMD")
    assert screen["calibrated"] and screen["band"] == 0
    assert screen["recommended"] == "no_change" and not screen["limitation"]
    assert "checked again after the plan is fixed" in screen["reason"]
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
    assert rows[SCREEN]["recorded"] == "no_change"
    assert rows[SCREEN]["realized"]["instrument"] == "sensitivity"
    assert rows[SCREEN]["verdict"] in ("confirmed", "upgraded", "downgraded")
    assert (rows[SCREEN]["label"] is not None) == (rows[SCREEN]["verdict"] == "upgraded")
    assert rows[DAYS]["verdict"] == "not_verifiable" and rows[DAYS]["limitation"]
