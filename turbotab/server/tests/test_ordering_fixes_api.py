"""P0.6 · the ordering fixes on the server, walked on the NHANES journey (sugar → fasting glucose,
under Estimate) and the clinical journey (30-day readmission, under Predict).

The expectations are the crosswalk's (``docs/turbotab-next/crosswalk/CROSSWALK.md``, "Where the
engine order and the stage order disagree"), independent of the code under test:

* disagreement 1: once every column's role is settled in Your data (proposed high or confirmed on
  its own), TurboTab records the roles itself when the Router reaches them, and Who's in shows
  that record For the record; while one is unsettled nothing is recorded;
* disagreement 5: under Estimate TurboTab records the split itself at the end of Who's in, no rows
  held out, For the record (``default:split_under_inference``), and the person can still change
  it; under Predict the split is asked, and a changed validation scheme (``set_validation``) keeps
  the draw: the same held-out rows, and the draw's record;
* disagreement 9: the survey question is asked under Predict where a design reads;
* disagreement 10: the design is stated observational, a Confirm line in Your question;
* disagreement 20: "Decide now" answers a later question early where its card is computed and the
  answers it reads are in, names what it waits for otherwise, and the record says what it was
  decided ahead of.

No fitted number may change: the split TurboTab records under Estimate is exactly the one every
Estimate journey recorded by hand before (no rows held out, seed 0, five folds), so every stage
reads the same answers.
"""
from __future__ import annotations

from typing import Any

import pytest

from turbotab.core.decisions import SplitSpec
from turbotab.core.seal import INFERENCE_SPLIT_REASON, split_writer
from turbotab.core.tests.acceptance.server_drive import Drive, open_project, settle_post
from turbotab.core.tests.stage_harness import NHANES
from turbotab.core.tests.truths import Truth, fixture_truth
from turbotab.server.tests.conftest import SAMPLES

needs_nhanes = pytest.mark.skipif(not NHANES.is_file(),
                                  reason="the NHANES export is not on this machine")
COVARIATES = ["age", "gender", "cycle_begin_year", "weight", "height", "bmi", "waist", "bp_sys",
              "bp_di", "hdl", "meds_hbp", "meds_chol"]
NUTRIENTS = ["protein", "carb", "fat_total", "fat_sat", "fat_mon", "fat_poly"]
ROLES = {"SEQN": "identifier", "sugar": "exposure", "kcal": "energy",
         **{c: "exposure" for c in NUTRIENTS}, **{c: "covariate" for c in COVARIATES},
         "triglycerides": "excluded",
         **{c: "flag" for c in ("imputed_weight", "imputed_height", "imputed_bmi",
                                "imputed_waist", "imputed_bp_sys", "imputed_bp_di")}}
CLINICAL_ROLES = {"encounter_id": "identifier", "age": "covariate", "sex": "covariate",
                  "charlson_index": "covariate", "prior_admissions_12mo": "covariate",
                  "albumin_g_dl": "covariate", "creatinine_mg_dl": "covariate",
                  "hemoglobin_g_dl": "covariate", "sodium_mmol_l": "covariate",
                  "length_of_stay_days": "covariate"}


def quest_lines(drive: Drive) -> dict[str, list[dict[str, Any]]]:
    log = drive.c.get(f"/api/projects/{drive.pid}/quest").json()
    return {s["key"]: s["lines"] for s in log["stages"]}


def line(lines: list[dict[str, Any]], key: str) -> dict[str, Any]:
    return next(l for l in lines if l["key"] == key)


def records_of(drive: Drive, kind: str) -> list[dict[str, Any]]:
    return [r for r in drive.view()["decisions"] if r["decision"]["kind"] == kind]


def step(drive: Drive, key: str) -> dict[str, Any]:
    return next(s for s in drive.view()["interview"] if s["key"] == key)


@needs_nhanes
def test_the_nhanes_journey_records_the_roles_and_the_split_for_the_record(client):
    drive = open_project(client, NHANES, fixture_truth("_tt_tmp_nhanes.csv"))
    drive.decide({"kind": "set_lens", "lenses": ["dietary"]})
    drive.reach("target")
    drive.decide({"kind": "set_target", "column": "glucose"})
    drive.reach("purpose")

    # The design: stated observational, a Confirm line in Your question (disagreement 10).
    design = step(drive, "design")
    assert design["status"] == "skipped" and "observational" in design["reason"]
    stated = line(quest_lines(drive)["question"], "design")
    assert (stated["label"], stated["status"], stated["id"]) == (
        "Confirm", "set_for_you", "q:study-design")

    drive.decide({"kind": "set_purpose", "purpose": "inference"})

    # The roles (disagreement 1): nothing is recorded while a proposal below high is unconfirmed;
    # once each is confirmed on its own, TurboTab records the roles as confirmed.
    drive.reach("roles")
    proposals = [p for p in drive.artifact("roles")["columns"] if p["column"] != "glucose"]
    attention = [p["column"] for p in proposals if p["confidence"] in ("medium", "low")]
    assert attention, "the export has columns whose role is read below high confidence"
    assert drive.view()["state"]["roles"] is None and step(drive, "roles")["status"] == "open"

    # Decide now (disagreement 20), with the Router held at the roles. Who is kept when values are
    # blank reads no card, so it is answered ahead of the roles; the eligibility rules' card reads
    # the roles, so it waits and says so. Without "Decide now" both wait their turn.
    url = f"/api/projects/{drive.pid}/decisions"
    missing = {"kind": "set_missing", "strategy": "complete_case"}
    refused = client.post(url, json=missing)
    assert refused.status_code == 409 and refused.json()["error"]["code"] == "not_yet"
    waits = client.post(url + "?decide_now=true", json={"kind": "set_exclusions", "rules": []})
    assert waits.status_code == 409, waits.text[:400]
    assert waits.json()["error"]["code"] == "not_yet"
    assert waits.json()["error"]["message"].startswith("Waiting for")
    early = client.post(url + "?decide_now=true", json=missing)
    assert early.status_code == 200, early.text[:600]
    record = early.json()["decisions"][-1]
    assert record["decision"]["kind"] == "set_missing" and record["early"] == "roles"
    assert record["sentence"].startswith("Decided early, ahead of the column roles: ")
    assert step(drive, "missing")["status"] == "answered"
    assert drive.view()["state"]["roles"] is None and step(drive, "roles")["status"] == "open"
    chosen = {c: ROLES.get(c, next(p["proposed"] for p in proposals if p["column"] == c))
              for c in attention}
    r = settle_post(client, drive.pid, {"kind": "confirm_readings", "items": [
        {"reading": "role", "column": c, "value": v} for c, v in chosen.items()]}, drive.truth)
    assert r.status_code == 200, r.text[:600]
    expected = {p["column"]: chosen.get(p["column"], p["proposed"]) for p in proposals}
    [completed] = records_of(drive, "set_roles")
    assert completed["recorded_by"] == "turbotab"
    assert completed["decision"]["roles"] == expected
    recorded = line(quest_lines(drive)["whos_in"], "roles_recorded")
    assert (recorded["label"], recorded["status"], recorded["decision_id"]) == (
        "For the record", "set_for_you", completed["id"])
    assert recorded["name"] == "Column roles recorded as you confirmed them in Your data"
    # A way to change it: the author's own roles, recorded as theirs.
    drive.decide_roles(ROLES)
    assert drive.view()["state"]["roles"] == ROLES
    assert records_of(drive, "set_roles")[-1]["recorded_by"] == "you"

    # The split (disagreement 5): at the end of Who's in TurboTab records it under Estimate, with
    # no rows held out: exactly the split each Estimate journey recorded by hand before.
    drive.answer("exclusions", {"kind": "set_exclusions", "rules": [
        {"column": "kcal", "low": 500, "high": 5000, "reason": "implausible intake"}]})
    drive.answer("missing", {"kind": "set_missing", "strategy": "complete_case"})
    assert drive.reach("split")["status"] == "answered"
    [split] = records_of(drive, "set_split")
    assert split["recorded_by"] == "turbotab"
    assert SplitSpec(**drive.view()["state"]["split"]) == SplitSpec(holdout=0.0, seed=0, folds=5)
    whos_in = quest_lines(drive)["whos_in"]
    shown = line(whos_in, "split")
    assert (shown["id"], shown["label"], shown["status"], shown["decision_id"]) == (
        "default:split_under_inference", "For the record", "set_for_you", split["id"])
    assert shown["reason"] == INFERENCE_SPLIT_REASON and not shown["counted"]
    # With a way to change it: the person's own split is recorded as theirs.
    drive.decide({"kind": "set_split", "holdout": 0.0, "seed": 3, "folds": 5})
    assert drive.view()["state"]["split"]["seed"] == 3
    assert records_of(drive, "set_split")[-1]["recorded_by"] == "you"
    assert len(records_of(drive, "set_split")) == 2  # TurboTab never records it over the person's


def test_the_clinical_journey_asks_the_split_and_a_changed_scheme_keeps_the_draw(client):
    path = SAMPLES / "clinical_risk.csv"
    truth = fixture_truth("clinical_risk.csv")
    drive = open_project(client, path, Truth({**truth, **{f"role:{c}": r for c, r
                                                           in CLINICAL_ROLES.items()}},
                                             fixture=path.name))
    drive.decide({"kind": "set_lens", "lenses": ["clinical"]})
    drive.reach("target")
    drive.decide({"kind": "set_target", "column": "readmit_30d"})
    drive.answer("task", {"kind": "set_task", "column": "readmit_30d", "task": "binary"})
    drive.answer("event", {"kind": "set_event", "column": "readmit_30d", "level": "1"})
    drive.reach("purpose")
    assert step(drive, "design")["status"] == "skipped"
    drive.decide({"kind": "set_purpose", "purpose": "prediction"})
    drive.answer("grain", {"kind": "set_grain", "grain": "one_row_per_unit"})
    drive.reach("roles")
    drive.decide_roles(CLINICAL_ROLES)
    # No column reads as a survey design: the question does not apply, under Predict as anywhere.
    assert step(drive, "survey")["status"] == "not_applicable"
    drive.answer("exclusions", {"kind": "set_exclusions", "rules": []})
    drive.answer("missing", {"kind": "set_missing", "strategy": "complete_case"})
    # Under Predict the split is the seal: asked, never recorded for the person.
    assert drive.reach("split")["status"] == "open"
    assert records_of(drive, "set_split") == []
    split_line = line(quest_lines(drive)["whos_in"], "split")
    assert (split_line["id"], split_line["label"]) == ("q:split", "Decide")
    drive.decide({"kind": "set_split", "holdout": 0.2, "seed": 3, "folds": 5})
    held_out = drive.sealed()
    [drawn] = records_of(drive, "set_split")

    # A changed scheme is its own kind: the same held-out rows, and the draw's record stands.
    r = client.post(f"/api/projects/{drive.pid}/decisions", json={
        "kind": "set_validation", "validation": "repeated_kfold", "folds": 5, "repeats": 10})
    assert r.status_code == 200, r.text[:600]
    state = r.json()["state"]["split"]
    assert (state["holdout"], state["seed"], state["validation"], state["repeats"]) == (
        0.2, 3, "repeated_kfold", 10)
    records = client.app.state.service.log(drive.pid).records()
    assert split_writer(records) == drawn["id"]
    assert drive.sealed() == held_out
    assert step(drive, "split")["decision_id"] == drawn["id"]
    scheme = line(quest_lines(drive)["models"], "set_validation")
    assert (scheme["label"], scheme["id"]) == ("Confirm", "default:validation-scheme")


def test_the_survey_question_is_asked_under_predict_where_a_design_reads(client):
    path = SAMPLES / "nhanes_dietary.csv"
    roles = {"SEQN": "identifier", "DR1TKCAL": "energy", "DR1TCARB": "covariate",
             "DR1TTFAT": "covariate", "DR1TALCO": "covariate", "WTDRD1": "design",
             "WTMEC2YR": "design", "SDMVSTRA": "design", "SDMVPSU": "design"}
    truth = fixture_truth(path.name)
    drive = open_project(client, path, Truth({**truth, **{f"role:{c}": r for c, r
                                                           in roles.items()}}, fixture=path.name))
    drive.decide({"kind": "set_lens", "lenses": ["dietary"]})
    drive.reach("target")
    drive.decide({"kind": "set_target", "column": "DR1TPROT"})
    drive.reach("purpose")
    drive.decide({"kind": "set_purpose", "purpose": "prediction"})
    drive.answer("grain", {"kind": "set_grain", "grain": "one_row_per_unit"})
    drive.reach("roles")
    drive.decide_roles(roles)
    asked = drive.reach("survey")
    assert asked["status"] == "open", asked
    # Its card says whose performance the scores estimate (MODELING_SEQUENCE ruling 13).
    options = {o["key"]: o for o in drive.artifact("proposals")["survey"]["options"]}
    assert "design-based cross-validation" in options["population:WTDRD1"]["consequence"]
    assert "these rows" in options["sample"]["consequence"]
    # The next question waits behind it.
    r = client.post(f"/api/projects/{drive.pid}/decisions",
                    json={"kind": "set_exclusions", "rules": []})
    assert r.status_code == 409 and r.json()["error"]["code"] == "not_yet"
    drive.decide(options["sample"]["decision"])
    assert step(drive, "survey")["status"] == "answered"
