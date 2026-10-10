"""LEASH · the routing leash where the routing verifier found it wrong (wave 2; INBOX leash notes).

Each test drives the real server (or, where a stage is enough, the stage as a worker runs it) on a
generator written here, with its own seed, and asserts the rule against a reference computed outside
the app: the pack's table as the fixture's author reads it, NumPy written out, pandas counts of the
file's own cells, or R (``survey::svyvar`` and ``EValue::evalues.OLS``) where R is installed.

1. **The adjustment card at thirty covariates.** On a wide NHANES-style table under inference
   (``DR1TSUGR`` → ``LBXGLU``) every covariate leads with the packs' guess and its source:
   demographics and lifestyle are confounders, other nutrients possible confounders, body size,
   the clinical measurements and the medications possible mediators measured with the exposure
   (declared without them and, beside, with them), and HbA1c another measure of the outcome. Those
   with the same guess are one block: accepting the guesses takes four taps (at most six). A
   multi-select answer settles exactly the covariates it lists; a mediator kept in the total-effect
   set is blocked and recorded.
2. **The grouping question by structure.** Any column that can group rows (more than ten repeating
   values, several rows on each) is asked under inference with its guess, whatever its name:
   ``study_site``, an integer ``region``, ``trial_site``, ``recruitment_centre``, ``gp_practice``,
   ``hosp``, ``facility_id``, ``physician_id``, ``doctor_id`` and an unnamed integer code; a
   category of words is asked with the guess "no"; a sex, an age, a blood pressure, an unnamed
   integer measurement and an education level never are. Answered, the intervals are CR2 by it,
   equal to the definition written out.
3. **A withheld inference fit serves no fit statistics** (R², RMSE, MAE and the rest) while the plan
   is open, and serves them again once it is answered.
4. **A bulk roles answer keeps the roles confirmed one by one**, and nothing they settled reopens.
5. **Repeats over imputed copies** are blocked and recorded under inference, the Rubin's-rules
   answer the first exit (CDC's NHANES DXA guidance quoted), whichever way the answer is reached.
6. **Censored values.** The lab pack's censored-values finding leaves the values below a detection
   limit to the app's own below-detection repair (it said "No control for this yet" beside it); a
   text predictor confirmed "amount" is checked by the plausibility checks as numbers.
7. **MODELING_SEQUENCE ruling 14.** The E-value of a difference is standardized by the
   design-weighted SD under the surveyed-population answer and by the sample SD otherwise, in the
   effects stage and the causal lane, agreeing with ``EValue::evalues.OLS`` given that SD (1e-8).

**Source checks** (read 2026-10-05). Schisterman EF, Cole SR, Platt RW. Overadjustment bias and
unnecessary adjustment in epidemiologic studies. *Epidemiology* 2009;20:488–495: "We define
overadjustment bias as control for an intermediate variable (or a descending proxy for an
intermediate variable) on a causal path from exposure to outcome." Tobin MD, Sheehan NA, Scurrah KJ,
Burton PR. Adjusting for treatment effects in studies of quantitative traits: antihypertensive
therapy and systolic blood pressure. *Stat Med* 2005;24:2911–2935 (PubMed 16152135, abstract):
"three approaches that are used relatively commonly are fundamentally flawed and should not be used
at all. These are: (i) ignoring the problem altogether …; (ii) fitting a conventional regression
model with treatment as a binary covariate; and (iii) excluding treated subjects from the analysis."
CDC, NHANES 1999–2006 DXA, Multiple Imputation Details: "The extra variability due to imputation
CANNOT be incorporated by simply analyzing a SINGLE dataset as if the imputed values were true
values."
"""
from __future__ import annotations

import importlib
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from turbotab.core import decisions as d, estimand
from turbotab.core.tests.acceptance import references as ref
from turbotab.core.tests.acceptance.r_reference import needs_r, run_r
from turbotab.core.tests.acceptance.server_drive import (
    answer_estimand,
    local_server,
    open_project,
)
from turbotab.core.tests.truths import Truth

AT = "2026-10-05T00:00:00Z"
PRE = {"causes_exposure": "yes", "causes_outcome": "yes", "after_exposure": "no"}
NUTRIENT = {"causes_exposure": "unknown", "causes_outcome": "unknown", "after_exposure": "no"}
TIMING = {"causes_exposure": "unknown", "causes_outcome": "yes", "after_exposure": "unknown"}
OUTCOME_KIND = {"causes_exposure": "no", "causes_outcome": "no", "after_exposure": "yes"}
CDC_DXA = ("The extra variability due to imputation CANNOT be incorporated by simply analyzing a "
           "SINGLE dataset as if the imputed values were true values")


def _csv(frame: pd.DataFrame, folder: Path, name: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.csv"
    frame.to_csv(path, index=False)
    return path


def _error(response: Any, code: str) -> dict[str, Any]:
    assert response.status_code == 409, response.text[:900]
    error = response.json()["error"]
    assert error["code"] == code, error
    return error


def _records(drive: Any) -> list[dict[str, Any]]:
    return drive.view()["decisions"]


def _until(read: Any, done: Any, timeout: float = 240.0) -> Any:
    end = time.monotonic() + timeout
    while True:
        value = read()
        if done(value):
            return value
        assert time.monotonic() < end, value
        time.sleep(0.1)


def _served(drive: Any, stage: str) -> dict[str, Any]:
    drive.artifact(stage)
    return drive.c.get(f"/api/projects/{drive.pid}/stages/{stage}").json()["artifact"]


def _step(drive: Any, key: str) -> dict[str, Any]:
    return next(s for s in drive.view()["interview"] if s["key"] == key)


# ═════════════════════════════════════════════════════════════════════════════
# 1 · the adjustment card stays light at thirty covariates
# ═════════════════════════════════════════════════════════════════════════════

# The fixture's thirty covariates and the packs' guess for each under DR1TSUGR → LBXGLU, as the
# fixture's author reads NUTRITION_PACK §08's table ("The adjustment card's guesses"): the
# reference.
DEMOGRAPHICS = ["RIDAGEYR", "RIAGENDR", "RIDRETH3", "DMDEDUC2", "INDFMPIR"]
LIFESTYLE = ["SMQ020", "ALQ130", "PAD680", "coffee_cups"]
NUTRIENTS = ["DR1TPROT", "DR1TCARB", "DR1TTFAT", "DR1TSFAT", "DR1TFIBE", "DR1TSODI"]
BODY = ["BMXBMI", "BMXWT", "BMXWAIST"]
CLINICAL = ["LBDHDD", "LBDLDL", "LBXTR", "LBXHSCRP", "LBXSATSI", "LBXSCR", "BPXSY1", "BPXDI1"]
MEDICATIONS = ["BPQ100D", "BPQ050A", "DIQ070"]
OUTCOME_MEASURES = ["LBXGH"]
WIDE = [*DEMOGRAPHICS, *LIFESTYLE, *NUTRIENTS, *BODY, *CLINICAL, *MEDICATIONS, *OUTCOME_MEASURES]
EXPECTED_GUESS = {**{c: PRE for c in DEMOGRAPHICS + LIFESTYLE}, **{c: NUTRIENT for c in NUTRIENTS},
                  **{c: TIMING for c in BODY + CLINICAL + MEDICATIONS},
                  **{c: OUTCOME_KIND for c in OUTCOME_MEASURES}}
WIDE_WHOLE = {"RIDAGEYR": "amount", "RIAGENDR": "code", "RIDRETH3": "code", "DMDEDUC2": "code",
              "SMQ020": "code", "ALQ130": "amount", "PAD680": "amount", "coffee_cups": "amount",
              "BPQ100D": "code", "BPQ050A": "code", "DIQ070": "code", "LBDHDD": "amount",
              "LBDLDL": "amount", "LBXTR": "amount", "LBXSATSI": "amount", "BPXSY1": "amount",
              "BPXDI1": "amount", "DR1TSODI": "amount", "DR1TKCAL": "amount", "LBXGLU": "amount"}


def _wide30(n: int = 800, seed: int = 3030) -> pd.DataFrame:
    """An NHANES-shaped table: thirty covariates by their NHANES names (and a coffee count), sugar
    as the exposure, total energy, and fasting glucose as the outcome."""
    rng = np.random.default_rng(seed)
    prot = rng.gamma(9, 9, n).round(1)
    carb = rng.gamma(12, 20, n).round(1)
    fat = rng.gamma(8, 9, n).round(1)
    sugar = (carb * rng.uniform(0.2, 0.5, n)).round(1)
    age = rng.integers(20, 81, n)
    gh = rng.normal(5.6, 0.5, n).clip(4.2).round(1)
    frame = pd.DataFrame({
        "SEQN": np.arange(93703, 93703 + n), "RIDAGEYR": age, "RIAGENDR": rng.integers(1, 3, n),
        "RIDRETH3": rng.choice([1, 2, 3, 4, 6, 7], n), "DMDEDUC2": rng.integers(1, 6, n),
        "INDFMPIR": rng.uniform(0, 5, n).round(2), "SMQ020": rng.integers(1, 3, n),
        "ALQ130": rng.integers(0, 9, n), "PAD680": rng.integers(1, 30, n) * 30,
        "coffee_cups": rng.integers(0, 7, n),
        "DR1TKCAL": (4 * prot + 4 * carb + 9 * fat + rng.normal(0, 60, n)).round(0),
        "DR1TSUGR": sugar, "DR1TPROT": prot, "DR1TCARB": carb, "DR1TTFAT": fat,
        "DR1TSFAT": (fat * rng.uniform(0.25, 0.4, n)).round(1),
        "DR1TFIBE": rng.gamma(4, 4, n).round(1),
        "DR1TSODI": rng.normal(3300, 900, n).clip(600).round(0),
        "BMXBMI": rng.normal(28, 5, n).clip(16, 60).round(1),
        "BMXWT": rng.normal(80, 16, n).clip(40).round(1),
        "BMXWAIST": rng.normal(97, 14, n).clip(55).round(1),
        "LBDHDD": rng.normal(52, 14, n).clip(15).round(0),
        "LBDLDL": rng.normal(112, 32, n).clip(30).round(0),
        "LBXTR": rng.lognormal(np.log(110), 0.45, n).round(0),
        "LBXHSCRP": rng.lognormal(np.log(2), 0.9, n).round(2),
        "LBXSATSI": rng.lognormal(np.log(22), 0.4, n).round(0),
        "LBXSCR": rng.normal(0.88, 0.2, n).clip(0.3).round(2),
        "BPXSY1": rng.normal(124, 17, n).round(0), "BPXDI1": rng.normal(71, 11, n).round(0),
        "BPQ100D": rng.choice([1, 2], n, p=[0.2, 0.8]),
        "BPQ050A": rng.choice([1, 2], n, p=[0.3, 0.7]),
        "DIQ070": rng.choice([1, 2], n, p=[0.1, 0.9]), "LBXGH": gh})
    frame["LBXGLU"] = (95 + 0.05 * sugar + 0.15 * (age - 50) + 12 * (gh - 5.6)
                       + rng.normal(0, 8, n)).round(0)
    return frame


WIDE_ROLES = {"SEQN": "identifier", "DR1TSUGR": "exposure", "DR1TKCAL": "energy",
              **{c: "exposure" for c in NUTRIENTS},
              **{c: "covariate" for c in WIDE if c not in NUTRIENTS}}


def _wide_truth() -> Truth:
    return Truth({**{f"code_or_count:{c}": v for c, v in WIDE_WHOLE.items()},
                  "unit:DR1TKCAL": "kcal", "day_count:DR1TKCAL": "1",
                  "sex_coding:RIAGENDR": "female=2,male=1", "contrast:DR1TSUGR": "substitution"},
                 fixture="the wide NHANES-style table")


def _derive_place(a: dict[str, str]) -> str:
    """The criterion's place for a total effect, written out for the answers this table takes
    (VanderWeele 2019): a pre-exposure possible cause is adjusted; unknown timing is the declared
    with-and-without pair; a consequence of the exposure that causes nothing is left out."""
    if a["after_exposure"] == "no":
        return "adjusted"
    if a["after_exposure"] == "unknown":
        return "secondary"
    return "left_out"


def _adjustment_card(drive: Any, exposure: str, done: Any = None) -> dict[str, Any]:
    return _until(lambda: drive.artifact("proposals").get("adjustment") or {},
                  lambda card: card.get("exposure") == exposure and (done is None or done(card)))


def test_1_the_adjustment_card_stays_light_at_thirty_covariates(tmp_path):
    """Reference: NUTRITION_PACK §08's guess table as the fixture's author reads it
    (``EXPECTED_GUESS``) and the criterion's places written out (``_derive_place``)."""
    frame = _wide30()
    path = _csv(frame, tmp_path, "wide30")
    with local_server(tmp_path / "home") as client:
        drive = open_project(client, path, _wide_truth())
        drive.decide({"kind": "set_lens", "lenses": ["dietary"]})
        drive.reach("target")
        drive.decide({"kind": "set_target", "column": "LBXGLU"})
        drive.answer("task", {"kind": "set_task", "column": "LBXGLU", "task": "regression"})
        drive.reach("purpose")
        drive.decide({"kind": "set_purpose", "purpose": "inference"})
        drive.answer("grain", {"kind": "set_grain", "grain": "one_row_per_unit"})
        drive.reach("roles")
        drive.decide_roles(WIDE_ROLES)
        # Covariates such as sex, age or a blood pressure never trigger the grouping question.
        assert drive.reach("clusters")["status"] == "skipped"
        assert drive.artifact("proposals")["grouping"] is None
        drive.answer("exclusions", {"kind": "set_exclusions", "rules": []})
        drive.answer("missing", {"kind": "set_missing", "strategy": "complete_case"})
        drive.answer("split", {"kind": "set_split", "holdout": 0.0, "seed": 0, "folds": 5})
        drive.reach("estimand")
        answer_estimand(drive, "DR1TSUGR", contrast="substitution")
        assert drive.reach("adjustment")["status"] == "open"
        card = _adjustment_card(drive, "DR1TSUGR")

        groups = card["groups"]
        listed = [c for g in groups for c in g["columns"]]
        assert sorted(listed) == sorted(WIDE) and len(listed) == 30  # each once
        assert all(g["guess"] is not None for g in groups)  # nothing unguessed here
        for g in groups:
            assert [m["column"] for m in g["members"]] == g["columns"]
            for m in g["members"]:  # every guess is visible with its source
                assert m["answers"] == EXPECTED_GUESS[m["column"]], m
                assert m["source"] and m["reason"] and m["label"], m
            assert g["decision"] == {"kind": "set_adjustment", "exposure": "DR1TSUGR",
                                     "answers": {c: g["guess"] for c in g["columns"]}}
        # Same guess, one block: four blocks, so accepting the guesses takes four taps (≤ 6).
        taps = len([g for g in groups if g["guess"] is not None]) + len(
            [c for g in groups if g["guess"] is None for c in g["columns"]])
        assert taps == 4 and taps <= 6
        by_key = {g["key"]: g for g in groups}
        assert by_key["demographic"]["label"] == "Demographics and lifestyle"
        assert by_key["body"]["label"] == ("Body size and composition, clinical measurements and "
                                           "medications")
        assert by_key["body"]["derived"] == "timing_unknown"
        assert by_key["outcome_measure"]["columns"] == ["LBXGH"]
        member = {m["column"]: m for g in groups for m in g["members"]}
        assert "Schisterman, Cole & Platt 2009" in member["LBDHDD"]["source"]
        assert "Tobin et al. 2005" in member["DIQ070"]["source"]  # it treats the outcome's group
        assert "Tobin et al. 2005" not in member["BPQ050A"]["source"]
        assert "`LBXGLU`" in member["LBXGH"]["reason"]
        # The repair: never "a consequence of the outcome" (HDL is no consequence of LDL).
        assert "consequence of it" not in member["LBXGH"]["reason"]
        assert member["LBXGH"]["reason"].startswith("another measure of the outcome `LBXGLU`'s "
                                                    "own kind (glycemic)")

        # A possible mediator kept in a total-effect set: blocked and recorded.
        kept = drive.post({"kind": "set_adjustment", "exposure": "DR1TSUGR",
                           "answers": {"BMXBMI": {**TIMING, "keep": True}}})
        error = _error(kept, "mediator_in_total_effect")
        assert error["exits"][0]["decision"]["answers"]["BMXBMI"]["further"] is True
        assert error["exits"][1]["decision"]["answers"]["BMXBMI"]["acknowledged"] is True

        # A multi-select answer across three blocks settles exactly the covariates it lists.
        bulk = {"RIDAGEYR": PRE, "LBDHDD": PRE, "DR1TPROT": {
            "causes_exposure": "no", "causes_outcome": "yes", "after_exposure": "no"}}
        assert set(card["bulk"]["columns"]) == set(WIDE)
        r = drive.post({**card["bulk"]["decision"], "answers": bulk})
        assert r.status_code == 200, r.text[:600]
        given = {c for c, a in (drive.view()["state"]["adjustment"] or {}).items()
                 if a["exposure"] == "DR1TSUGR"}
        assert given == set(bulk)
        card = _adjustment_card(drive, "DR1TSUGR",
                                lambda c: set(c["bulk"]["columns"]) == set(WIDE) - set(bulk))
        rest = [c for g in card["groups"] for c in g["columns"]]
        assert sorted(rest) == sorted(set(WIDE) - set(bulk))
        before = len(_records(drive))
        for g in card["groups"]:  # accept every block's guess: one tap each
            r = drive.post(g["decision"])
            assert r.status_code == 200, r.text[:600]
        taps = [x for x in _records(drive)[before:] if x["decision"]["kind"] == "set_adjustment"]
        assert len(taps) == 4
        assert drive.reach("adjustment")["status"] == "answered"
        sentences = {next(iter(x["decision"]["answers"])): x["sentence"] for x in taps}
        card = _adjustment_card(drive, "DR1TSUGR", lambda c: not c["bulk"]["columns"])

    answers = {**{c: EXPECTED_GUESS[c] for c in WIDE}, **bulk}
    places = {c: _derive_place(a) for c, a in answers.items()}
    assert set(card["adjusted"]) == {c for c, p in places.items() if p == "adjusted"}
    assert set(card["secondary"]) == {c for c, p in places.items() if p == "secondary"}
    assert set(card["left_out"]) == {c for c, p in places.items() if p != "adjusted"}
    assert "LBXGH" in card["left_out"] and "LBXGH" not in card["secondary"]
    # The record says what the card said (the repair: it said "a possible collider").
    assert sentences["LBXGH"] == ("For the effect of `DR1TSUGR`, by the disjunctive cause "
                                  "criterion: `LBXGH` is another measure of the outcome's own "
                                  "kind, which the exposure could have changed as it could the "
                                  "outcome, left out.")
    assert card["answered"]["LBXGH"]["words"] == "another measure of the outcome's own kind"


def _state(exposure: str, target: str, task: str) -> d.ProjectState:
    measure = {"regression": "mean_difference", "time_to_event": "hazard_ratio"}.get(task,
                                                                                    "odds_ratio")
    return d.ProjectState(roles={exposure: "exposure"}, target=target, task=task,
                          purpose="inference", estimand=d.EstimandSpec(
                              exposure=exposure, effect="total", measure=measure))


def test_1_the_guess_follows_the_exposure_outcome_pairing():
    """The packs' table, row by row, under the pairings it distinguishes (reference: NUTRITION_PACK
    §08 as written): a dietary exposure and a glycemic outcome; a dietary exposure and an event over
    follow-up; an LDL exposure and an event; and (the repair) a baseline measurement of the
    outcome's own group beside an outcome over follow-up (an incident event, or a measurement named
    as taken at follow-up, as a change or at a time since baseline), which is the clinical row's
    pair (unknown, yes, unknown), never "another measure of the outcome" left out."""
    from turbotab.core.covariate_guesses import guess

    state = _state
    cross = state("DR1TSUGR", "LBXGLU", "regression")
    cohort = state("fiber_g", "death", "time_to_event")
    ldl = state("LBDLDL", "chd", "time_to_event")
    expect = [
        (cross, "LBXGH", OUTCOME_KIND, "another measure of the outcome `LBXGLU`'s own kind"),
        (cross, "hba1c", OUTCOME_KIND, "another measure of the outcome `LBXGLU`'s own kind"),
        (cross, "LBDHDD", TIMING, "measured at the same visit as what you study"),
        (cross, "metformin", TIMING, "Tobin et al. 2005"),
        (cross, "statin_use", TIMING, "a treatment that what you study may have led to"),
        (cross, "smoking", PRE, "habits"),
        (cross, "physical_activity", PRE, "habits"),
        (cross, "alcohol_drinks", PRE, "habits"),
        (cohort, "LBXGH", TIMING, "measured at baseline with what you study"),
        (cohort, "hs_crp", TIMING, "measured at baseline with what you study"),
        (ldl, "statin_use", TIMING, "it lowers `LBDLDL`, what you study,"),
        (ldl, "LBXGH", TIMING, "measured at baseline with what you study"),
    ]
    # The repair (the LEASH verifier's v1c_cohort): beside an incident event the baseline levels of
    # the outcome's own group take the clinical (or body) row; beside an LDL outcome at the same
    # visit HDL and triglycerides are another measure of its own kind, never its consequence.
    incident = state("fiber_g", "incident_diabetes", "time_to_event")
    htn = state("sodium_mg", "hypertension", "time_to_event")
    named_incident = state("fiber_g", "incident_t2d", "binary")
    prevalent = state("fiber_g", "diabetes", "binary")
    ldl_out = state("sat_fat_g", "ldl_mg_dl", "regression")
    obese = state("sugar_g", "obesity", "time_to_event")
    baseline = "measured at baseline with what you study, so before the event: no consequence of the event"
    # A continuous outcome named as measured over follow-up, as a change, or at a time since
    # baseline: the baseline level of its own kind is the pair too, with Glymour et al. 2005 beside.
    later = ("measured at baseline with what you study, so before the outcome's measurement: no "
             "consequence of the outcome")
    ldl_12m = state("sat_fat_g", "ldl_12m", "regression")
    change = state("sugar_g", "hba1c_change", "regression")
    weight_fu = state("kcal", "weight_followup", "regression")
    post = state("sugar_g", "postprandial_glucose", "regression")  # same visit: no follow-up word
    expect += [
        (incident, "hba1c", TIMING, baseline), (incident, "fasting_glucose", TIMING, baseline),
        (incident, "LBXGH", TIMING, baseline), (htn, "sbp", TIMING, baseline),
        (htn, "dbp", TIMING, baseline), (htn, "BPXSY1", TIMING, baseline),
        (named_incident, "hba1c", TIMING, baseline), (obese, "bmi", TIMING, baseline),
        (prevalent, "hba1c", OUTCOME_KIND, "another measure of the outcome `diabetes`'s own kind"),
        (ldl_out, "hdl", OUTCOME_KIND, "another measure of the outcome `ldl_mg_dl`'s own kind"),
        (ldl_out, "triglycerides", OUTCOME_KIND, "it measures the state the outcome measures"),
        (ldl_12m, "ldl_baseline", TIMING, later), (ldl_12m, "hdl", TIMING, later),
        (change, "hba1c", TIMING, later), (change, "fasting_glucose", TIMING, later),
        (weight_fu, "bmi", TIMING, later), (weight_fu, "waist_cm", TIMING, later),
        (post, "hba1c", OUTCOME_KIND, "another measure of the outcome `postprandial_glucose`'s "),
    ]
    for st, column, answers, said in expect:
        found = guess(column, st)
        assert found is not None and dict(found.answers) == answers, (column, found)
        assert said in found.reason or said in found.source, (column, found.reason)
        assert "consequence of it rather than a cause" not in found.reason, (column, found.reason)
        # Glymour et al. 2005 (baseline adjustment in an analysis of change) beside a measurement
        # over follow-up only, never beside an event.
        assert ("Glymour et al. 2005" in found.source) == (said == later), (column, found.source)
    assert guess("total_cholesterol", ldl) is None  # another measure of the exposure: asked
    coffee = state("coffee_cups", "LBXGLU", "regression")
    assert guess("caffeine_mg", coffee) is None  # the exposure's own habit: asked
    assert dict(guess("smoking", coffee).answers) == PRE
    assert guess("cycle_begin_year", cross) is None  # the packs have no guess


def test_1_the_record_says_what_the_card_said_of_another_measure_of_the_outcome():
    """The verifier's WORDING note: an accepted outcome-measure guess was recorded as "a
    consequence of the exposure (a possible collider)" while the card said "another measure of the
    outcome". Reference: the sentences written out from NUTRITION_PACK §08's row and the
    criterion's wording for a covariate the packs have no guess for (the same answers)."""
    from turbotab.core.voice import sentence_for

    out = d.CovariateAnswers(**OUTCOME_KIND)
    cross = _state("DR1TSUGR", "LBXGLU", "regression")
    said = sentence_for(d.SetAdjustment(exposure="DR1TSUGR", answers={
        "LBXGH": out, "fasting_insulin": out, "cycle_begin_year": out}), cross)
    assert said == ("For the effect of `DR1TSUGR`, by the disjunctive cause criterion: `LBXGH` and "
                    "`fasting_insulin` are other measures of the outcome's own kind, which the "
                    "exposure could have changed as it could the outcome, left out; "
                    "`cycle_begin_year` is a consequence of the exposure (a possible collider), "
                    "left out.")
    # Beside an incident event the packs guess the pair for HbA1c: answers the user changed to
    # "no, no, yes" are the criterion's, worded as such, never as the card's outcome measure.
    incident = _state("fiber_g", "incident_diabetes", "time_to_event")
    said = sentence_for(d.SetAdjustment(exposure="fiber_g", answers={"hba1c": out}), incident)
    assert said == ("For the effect of `fiber_g`, by the disjunctive cause criterion: `hba1c` is a "
                    "consequence of the exposure (a possible collider), left out.")


def _incident_cohort(n: int = 900, seed: int = 1717) -> pd.DataFrame:
    """The verifier's v1c_cohort shape: fiber at baseline, incident diabetes over follow-up, and
    baseline HbA1c and fasting glucose that drive the hazard (so neither is a consequence of the
    event: the simulation's truth)."""
    rng = np.random.default_rng(seed)
    age = rng.integers(40, 76, n)
    fiber = rng.gamma(4, 4, n).round(1)
    hba1c = (5.5 + 0.015 * (age - 55) - 0.01 * (fiber - 16) + rng.normal(0, 0.35, n)).round(1)
    glucose = (95 + 12 * (hba1c - 5.6) + rng.normal(0, 6, n)).round(0).astype(int)
    bmi = (27 + rng.normal(0, 4, n) - 0.05 * (fiber - 16)).round(1)
    hazard = 0.02 * np.exp(1.2 * (hba1c - 5.6) + 0.03 * (age - 55) - 0.02 * (fiber - 16)
                           + 0.04 * (bmi - 27))
    t = rng.exponential(1 / hazard)
    c = rng.uniform(2, 12, n)
    return pd.DataFrame({"pid": np.arange(1, n + 1), "age": age,
                         "sex": rng.choice(["female", "male"], n), "fiber_g": fiber,
                         "hba1c": hba1c, "fasting_glucose": glucose, "bmi": bmi,
                         "followup_years": np.minimum(t, c).round(3),
                         "incident_diabetes": (t <= c).astype(int)})


# The pack's table for this pairing, as the fixture's author reads NUTRITION_PACK §08: demographics
# are confounders; body size and the baseline clinical measurements measured with the exposure
# are the declared with-and-without pair (an event over follow-up: no "outcome's own kind" row).
COHORT_GUESS = {"age": PRE, "sex": PRE, "bmi": TIMING, "hba1c": TIMING, "fasting_glucose": TIMING}


def test_1_a_baseline_level_of_the_outcomes_kind_beside_an_incident_event_is_the_pair(tmp_path):
    """The verifier's v1c_cohort: fiber → incident diabetes over follow-up, where baseline HbA1c
    and fasting glucose took "another measure of the outcome … a consequence of it" and were left
    out as "possible colliders". Reference: ``COHORT_GUESS`` (the pack's table read by hand) and
    the places it derives, written out (``_derive_place``); the sentence verbatim."""
    frame = _incident_cohort()
    assert frame["incident_diabetes"].sum() > 60  # events enough for the pairing to be real
    path = _csv(frame, tmp_path, "incident")
    truth = Truth({"code_or_count:age": "amount", "code_or_count:fasting_glucose": "amount",
                   "code_or_count:pid": "amount", "exposure:incident_diabetes": "fiber_g"},
                  fixture="the incident-diabetes cohort")
    roles = {"pid": "identifier", "fiber_g": "exposure", "age": "covariate", "sex": "covariate",
             "hba1c": "covariate", "fasting_glucose": "covariate", "bmi": "covariate",
             "followup_years": "time"}
    with local_server(tmp_path / "home") as client:
        drive = open_project(client, path, truth)
        drive.decide({"kind": "set_lens", "lenses": ["clinical"]})
        drive.reach("target")
        drive.decide({"kind": "set_target", "column": "incident_diabetes"})
        drive.answer("event", {"kind": "set_event", "column": "incident_diabetes", "level": "1"})
        drive.decide({"kind": "set_task", "column": "incident_diabetes", "task": "time_to_event"})
        drive.decide({"kind": "set_follow_up", "column": "incident_diabetes",
                      "time_column": "followup_years"})
        drive.reach("purpose")
        drive.decide({"kind": "set_purpose", "purpose": "inference"})
        drive.answer("grain", {"kind": "set_grain", "grain": "one_row_per_unit", "id_column": "pid"})
        drive.reach("roles")
        drive.decide_roles(roles)
        drive.answer("exclusions", {"kind": "set_exclusions", "rules": []})
        drive.answer("missing", {"kind": "set_missing", "strategy": "complete_case"})
        drive.answer("split", {"kind": "set_split", "holdout": 0.0, "seed": 0, "folds": 5})
        drive.reach("estimand")
        drive.decide({"kind": "set_estimand", "exposure": "fiber_g", "measure": "hazard_ratio"})
        assert drive.reach("adjustment")["status"] == "open"
        card = _adjustment_card(drive, "fiber_g")
        member = {m["column"]: m for g in card["groups"] for m in g["members"]}
        assert {c: m["answers"] for c, m in member.items()} == COHORT_GUESS
        for c in ("hba1c", "fasting_glucose"):
            reason = member[c]["reason"]
            assert reason.startswith("a level of the outcome `incident_diabetes`'s own kind "
                                     "(glycemic) measured at baseline with what you study, so before "
                                     "the event: no consequence of the event, and it can predict "
                                     "it"), reason
            assert "NUTRITION_PACK §08" in member[c]["source"]
        assert "outcome_measure" not in {g["key"] for g in card["groups"]}
        before = len(_records(drive))
        for g in card["groups"]:  # accept every block's guess: one tap each
            r = drive.post(g["decision"])
            assert r.status_code == 200, r.text[:600]
        taps = [x for x in _records(drive)[before:] if x["decision"]["kind"] == "set_adjustment"]
        assert len(taps) == 2
        card = _adjustment_card(drive, "fiber_g", lambda c: not c["bulk"]["columns"])
    places = {c: _derive_place(a) for c, a in COHORT_GUESS.items()}
    assert set(card["adjusted"]) == {c for c, p in places.items() if p == "adjusted"}
    assert set(card["secondary"]) == {"bmi", "hba1c", "fasting_glucose"} == {
        c for c, p in places.items() if p == "secondary"}
    pair = next(x["sentence"] for x in taps if "bmi" in x["decision"]["answers"])
    assert pair == ("For the effect of `fiber_g`, by the disjunctive cause criterion: `bmi`, "
                    "`hba1c` and `fasting_glucose` are of unknown timing, left out of the primary "
                    "model and adjusted for in a declared secondary one.")
    assert "collider" not in " ".join(x["sentence"] for x in taps)


# ═════════════════════════════════════════════════════════════════════════════
# 2 · the grouping question by structure; 3 · a withheld fit serves no fit statistics
# ═════════════════════════════════════════════════════════════════════════════

GROUPED = {"study_site": "yes", "region": "yes", "trial_site": "yes", "recruitment_centre": "yes",
           "gp_practice": "yes", "hosp": "yes", "facility_id": "yes", "physician_id": "yes",
           "doctor_id": "yes", "cg": "yes", "country_of_birth": "no", "x_bell": "no"}
# The repair: an unnamed whole-number column whose counts fall away from the middle (``x_bell``) is
# asked with the guess "no", never dropped; only a name the packs read as a characteristic or a
# measured quantity, with no grouping word beside it, scopes a column out.
NEVER = ["sex", "age", "sbp", "education", "bmi"]


def _spearman_by_hand(values: pd.Series) -> float:
    """Spearman's ρ between each distinct value's count and its distance from the values' median,
    written out: average ranks by pandas, then Pearson's r of the ranks by NumPy."""
    counts = values.value_counts()
    distance = (counts.index.to_series().astype(float) - float(np.median(values))).abs()
    a = counts.rank(method="average").to_numpy(float)
    b = distance.rank(method="average").to_numpy(float)
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / math.sqrt((a @ a) * (b @ b)))


def _sizes(rng: np.random.Generator, k: int, n: int) -> np.ndarray:
    """``n`` rows over ``k`` groups of uneven sizes (a Dirichlet draw), every group at least 3."""
    p = rng.dirichlet(np.full(k, 2.0))
    counts = np.maximum(3, np.round(p * (n - 3 * k))).astype(int) + 0
    counts[0] += n - counts.sum()
    return np.repeat(np.arange(k), counts)


def _multisite(n: int = 720, seed: int = 4242) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    def labels(values: list[Any], k: int) -> np.ndarray:
        g = rng.permutation(_sizes(rng, k, n))
        return np.asarray(values, dtype=object)[g]

    cities = ["Leeds", "York", "Hull", "Bath", "Derby", "Luton", "Wells", "Ripon", "Truro", "Ely",
              "Bristol", "Exeter", "Durham", "Carlisle"]
    countries = ["Mexico", "India", "China", "Vietnam", "Cuba", "Peru", "Chile", "Ghana",
                 "Kenya", "Egypt", "Japan", "Korea", "Brazil", "Poland", "Spain"]
    frame = pd.DataFrame({
        "participant_id": [f"P{i:04d}" for i in range(n)],
        "study_site": labels([f"S{i:02d}" for i in range(1, 25)], 24),
        "region": labels(list(range(1, 16)), 15),
        "trial_site": labels([f"T{i}" for i in range(1, 19)], 18),
        "recruitment_centre": labels(cities, len(cities)),
        "gp_practice": labels(list(range(1001, 1041)), 40),
        "hosp": labels([f"H-{i:02d}" for i in range(1, 21)], 20),
        "facility_id": labels(list(range(501, 531)), 30),
        "physician_id": labels([f"DR{i:03d}" for i in range(1, 36)], 35),
        "doctor_id": labels(list(range(1, 61)), 60),
        "cg": labels(list(range(1, 26)), 25),
        "country_of_birth": labels(countries, len(countries)),
        "sex": rng.choice(["female", "male"], n),
        "age": rng.integers(20, 81, n),
        "sbp": rng.normal(124, 15, n).round(0).astype(int),
        "x_bell": rng.normal(100, 15, n).round(0).astype(int),
        "education": rng.choice(["none", "primary", "secondary", "college", "graduate"], n),
        "bmi": rng.normal(27, 4, n).round(1),
        "x": rng.normal(0, 1, n).round(3)})
    frame["y"] = (1 + 0.4 * frame["x"] + 0.01 * frame["age"] + rng.normal(0, 1, n)).round(3)
    return frame


def test_2_the_grouping_question_is_asked_of_any_column_that_can_group_rows(tmp_path):
    """Reference: pandas counts of the file's own cells (more than ten values, at least two rows on
    the median one) for every column asked, and none of the columns that must never be."""
    frame = _multisite()
    for c in GROUPED:  # the reference: each column asked can structurally group rows
        counts = frame[c].value_counts()
        assert len(counts) > 10 and counts.median() >= 2 and len(frame) / len(counts) >= 2, c
    assert _spearman_by_hand(frame["x_bell"]) <= -0.5  # a measurement's shape, asked all the same
    path = _csv(frame, tmp_path, "multisite")
    roles = {"participant_id": "identifier", "x": "exposure",
             **{c: "covariate" for c in frame.columns if c not in ("participant_id", "x", "y")}}
    truth = Truth({"code_or_count:age": "amount"}, fixture="the multi-site table")
    with local_server(tmp_path / "home") as client:
        drive = open_project(client, path, truth)
        drive.decide({"kind": "set_lens", "lenses": ["clinical"]})
        drive.reach("target")
        drive.decide({"kind": "set_target", "column": "y"})
        drive.answer("task", {"kind": "set_task", "column": "y", "task": "regression"})
        drive.reach("purpose")
        drive.decide({"kind": "set_purpose", "purpose": "inference"})
        drive.answer("grain", {"kind": "set_grain", "grain": "one_row_per_unit"})
        drive.reach("roles")
        drive.decide_roles(roles)
        assert drive.reach("clusters")["status"] == "open"
        card = _until(lambda: drive.artifact("proposals").get("grouping"), lambda c: bool(c))
        asked = {c["column"]: c["guess"] for c in card["columns"]}
        assert asked == GROUPED
        assert all(c["why"] for c in card["columns"])  # the guess is shown with its evidence
        assert not set(NEVER) & set(asked)
        # "Nothing groups them" over columns guessed to group the participants: block and record.
        error = _error(drive.post({"kind": "set_clusters", "column": None}), "grouping_reads")
        # The one column guessed from its values alone (``cg``) may be said to mark no group: its
        # reading confirmed "no" (the repair), beside the record exit for the named ones.
        assert {"label": "`cg` marks no group of participants",
                "decision": {"kind": "confirm_reading", "reading": "cluster", "column": "cg",
                             "value": "no"}} in error["exits"]
        nothing = next(e["decision"] for e in error["exits"]
                       if e["label"] == "They group nothing; record that")
        drive.decide(nothing)
        record = _records(drive)[-1]
        assert record["decision"]["none_of"] == list(GROUPED)  # in the table's column order
        # The limitation names only the columns that read as a grouping (the guess "yes"); the two
        # asked with the guess "no" were asked and answered, and are no limitation (the repair: a
        # measurement's shape now asks, so it must not inflate the stated limitation).
        read = [c for c, g in GROUPED.items() if g == "yes"]
        assert read[:2] == ["study_site", "region"] and len(read) == 10
        assert [c for c, g in GROUPED.items() if g == "no"] == ["country_of_birth", "x_bell"]
        assert record["sentence"] == (
            "Nothing groups the participants above the person; each is analyzed as independent, "
            "although `study_site`, `region` and 8 more read as possible groupings of the "
            "participants (a site, centre, household or batch, by name or by values); the answer "
            "was kept over that reading, so the intervals do not cluster by them, and it is a "
            "stated limitation; `country_of_birth` and `x_bell` were asked whether they group the "
            "participants, and the answer was that they do not.")
        state = drive.view()["state"]
        assert all(state["reading_confirmations"][f"cluster:{c}"] == "no" for c in GROUPED)

    # Under prediction the question is not widened: no column of these is asked by its values.
    from turbotab.core.groupings import candidates

    st = d.ProjectState(purpose="prediction", target="y", roles=roles)
    assert candidates(st, {"groupings": [{"column": "cg", "levels": 25, "rows": 720,
                                          "median_rows": 28, "kind": "whole_numbers",
                                          "profile": 0.0, "named": False}]}) == []


def _falling(seed: int = 99) -> pd.DataFrame:
    """The verifier's seed-99 case, built on purpose: 35 groups coded 1–35 in an unnamed whole
    number (``grp2``), larger near the median code (as clinics numbered outward from a city's
    center might be), so the counts fall away from the middle as a measurement's do; a strong group
    effect, and an exposure that shares it."""
    rng = np.random.default_rng(seed)
    codes = np.arange(1, 36)
    sizes = np.round(4 + 14 * np.exp(-((codes - 18) / 9.0) ** 2)).astype(int)
    g = np.repeat(codes, sizes)
    n = len(g)
    shared = rng.normal(0, 1.2, len(codes))[g - 1]
    x = (0.6 * shared + rng.normal(0, 1, n)).round(3)
    age = rng.normal(55, 10, n).round(1)
    y = (2 + 0.5 * x + 0.03 * age + shared + rng.normal(0, 1, n)).round(3)
    order = rng.permutation(n)
    return pd.DataFrame({"participant_id": [f"G{i:04d}" for i in range(n)], "grp2": g[order],
                         "x": x[order], "age": age[order], "y": y[order]})


def _falling_reference(frame: pd.DataFrame) -> dict[str, Any]:
    """OLS of y on x and age; CR2 by ``grp2`` with Bell–McCaffrey df and HC3, both written out
    (``references``)."""
    from scipy import stats

    X = np.column_stack([np.ones(len(frame)), frame["x"], frame["age"]])
    yv = frame["y"].to_numpy(float)
    beta = np.linalg.lstsq(X, yv, rcond=None)[0]
    e = yv - X @ beta
    V, df = ref.cr2_by_definition(X, e, pd.factorize(frame["grp2"])[0])
    q = stats.t.ppf(0.975, df[1])
    se = math.sqrt(V[1, 1])
    return {"beta": beta[1], "se": se, "df": df[1], "ci": (beta[1] - q * se, beta[1] + q * se),
            "hc3_se": math.sqrt(ref.hc3_by_definition(X, e)[1, 1])}


def test_2_a_grouping_whose_counts_fall_away_from_its_median_code_is_asked(tmp_path):
    """The verifier's seed 99: an unnamed whole-number grouping whose counts fell away from the
    median code was dropped by its value profile, the question skipped, and the intervals HC3 (SE
    0.052 against 0.117 for CR2). Asked now with the guess "no" and its evidence; answered, the
    intervals are CR2 by it. Reference: Spearman's ρ written out (pandas ranks, NumPy r), and CR2
    with Bell–McCaffrey df and HC3 by definition (``references``; the CR2 definition is checked
    against R clubSandwich below)."""
    frame = _falling()
    counts = frame["grp2"].value_counts()
    assert len(counts) == 35 and counts.median() >= 2
    rho = _spearman_by_hand(frame["grp2"])
    assert rho <= -0.5  # a measured quantity's shape, though it is a grouping
    expected = _falling_reference(frame)
    assert expected["se"] > 1.3 * expected["hc3_se"]  # what skipping the question cost
    path = _csv(frame, tmp_path, "falling")
    truth = Truth({"code_or_count:age": "amount", "code_or_count:grp2": "code",
                   "adjust:age": "no,yes,no", "exposure:y": "x"}, fixture="the falling grouping")
    roles = {"participant_id": "identifier", "grp2": "identifier", "x": "exposure",
             "age": "covariate"}
    with local_server(tmp_path / "home") as client:
        drive = open_project(client, path, truth)
        drive.decide({"kind": "set_lens", "lenses": ["clinical"]})
        drive.reach("target")
        drive.decide({"kind": "set_target", "column": "y"})
        drive.answer("task", {"kind": "set_task", "column": "y", "task": "regression"})
        drive.reach("purpose")
        drive.decide({"kind": "set_purpose", "purpose": "inference"})
        drive.decide({"kind": "set_grain", "grain": "one_row_per_unit",
                      "id_column": "participant_id"})
        drive.reach("roles")
        drive.decide_roles(roles)
        assert drive.reach("clusters")["status"] == "open"
        card = _until(lambda: drive.artifact("proposals").get("grouping"), lambda c: bool(c))
        (asked,) = card["columns"]
        assert (asked["column"], asked["guess"], asked["by"]) == ("grp2", "no", "values")
        assert f"(ρ = {rho:.2f})" in asked["why"] and "fall away from the middle" in asked["why"]
        # "Nothing groups them", even acknowledged (as a client may post it), over a column asked
        # only with the guess "no" is no block and states no limitation: it was asked and answered.
        r = drive.post({"kind": "set_clusters", "column": None, "acknowledged": True})
        assert r.status_code == 200, r.text[:600]
        assert _records(drive)[-1]["sentence"] == (
            "Nothing groups the participants above the person; each is analyzed as independent; "
            "`grp2` was asked whether it groups the participants, and the answer was that it does "
            "not.")
        # The user changes the answer: the grouping stands over it (its own confirmation, "yes").
        drive.decide({"kind": "set_clusters", "column": "grp2", "adjust": "cluster_only"})
        drive.answer("exclusions", {"kind": "set_exclusions", "rules": []})
        drive.answer("missing", {"kind": "set_missing", "strategy": "complete_case"})
        drive.answer("split", {"kind": "set_split", "holdout": 0.0, "seed": 0, "folds": 5})
        drive.reach("estimand")
        answer_estimand(drive, "x")
        drive.reach("adjustment")
        tap = drive.post({"kind": "set_adjustment", "exposure": "x",
                          "answers": {"age": {"causes_exposure": "no", "causes_outcome": "yes",
                                              "after_exposure": "no"}}})
        assert tap.status_code == 200, tap.text[:600]
        drive.reach("models")
        drive.decide({"kind": "select_models", "models": ["linear"]})
        fit = _until(lambda: _served(drive, "fit"), lambda f: not f.get("withheld"))
    model = fit["models"][0]
    row = next(r for r in model["coefficients"] if r["feature"] == "x")
    assert model["inference"]["covariance"] == "CR2"
    assert model["inference"]["grouped_by"] == "grp2"
    assert row["estimate"] == pytest.approx(expected["beta"], rel=1e-8)
    assert (row["ci_low"], row["ci_high"]) == pytest.approx(expected["ci"], abs=1e-6)
    assert row["df"] == pytest.approx(expected["df"], rel=1e-6)


CR2_R = """
suppressMessages({library(clubSandwich)})
dd <- read.csv(rows_csv)
fit <- lm(y ~ x + age, data = dd)
ct <- coef_test(fit, vcov = "CR2", cluster = dd$grp2, test = "Satterthwaite")
df <- if ("df_Satt" %in% names(ct)) ct$df_Satt[2] else ct$df[2]
out(list(se = ct$SE[2], df = df, b = unname(coef(fit)["x"])))
"""


@needs_r
def test_2_the_cr2_definition_is_clubsandwich_on_the_falling_grouping(tmp_path):
    """The reference the falling-grouping test asserts against, checked against R:
    ``clubSandwich::coef_test(vcov = "CR2", test = "Satterthwaite")`` on the same rows (1e-8)."""
    frame = _falling()
    expected = _falling_reference(frame)
    r = run_r(CR2_R, {"rows": frame}, tmp_path / "r")
    assert r["b"] == pytest.approx(expected["beta"], rel=1e-10)
    assert r["se"] == pytest.approx(expected["se"], rel=1e-8)
    assert r["df"] == pytest.approx(expected["df"], rel=1e-8)


def _named_both_ways(n: int = 900, seed: int = 5150) -> pd.DataFrame:
    """Groupings whose names carry a lifestyle word (``alcohol_clinic``, ``coffee_shop_id``) beside
    characteristics that must never be asked."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "participant_id": [f"N{i:04d}" for i in range(n)],
        "alcohol_clinic": rng.integers(101, 131, n),
        "coffee_shop_id": [f"CS-{k:02d}" for k in rng.integers(1, 26, n)],
        "sex": rng.choice(["female", "male"], n), "age": rng.integers(20, 81, n),
        "sbp": rng.normal(124, 15, n).round(0).astype(int), "x": rng.normal(0, 1, n).round(3)})
    frame["y"] = (0.4 * frame["x"] + 0.01 * frame["age"] + rng.normal(0, 1, n)).round(3)
    return frame


def _birth_years(n: int = 900, seed: int = 1946) -> pd.DataFrame:
    """A uniform whole-number measurement (a birth year) whose counts follow no order: guessed from
    its values to group the participants, which it does not."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({"participant_id": [f"B{i:04d}" for i in range(n)],
                          "birth_year": rng.integers(1940, 1990, n),
                          "sex": rng.choice(["female", "male"], n),
                          "x": rng.normal(0, 1, n).round(3)})
    frame["y"] = (0.4 * frame["x"] + rng.normal(0, 1, n)).round(3)
    return frame


def _to_clusters(drive: Any, roles: dict[str, str]) -> dict[str, Any]:
    drive.decide({"kind": "set_lens", "lenses": ["clinical"]})
    drive.reach("target")
    drive.decide({"kind": "set_target", "column": "y"})
    drive.answer("task", {"kind": "set_task", "column": "y", "task": "regression"})
    drive.reach("purpose")
    drive.decide({"kind": "set_purpose", "purpose": "inference"})
    drive.decide({"kind": "set_grain", "grain": "one_row_per_unit",
                  "id_column": "participant_id"})
    drive.reach("roles")
    drive.decide_roles(roles)
    assert drive.reach("clusters")["status"] == "open"
    card = _until(lambda: drive.artifact("proposals").get("grouping"), lambda c: bool(c))
    return {c["column"]: c for c in card["columns"]}


def test_2_a_grouping_name_with_a_measured_word_is_asked_and_a_values_guess_can_be_denied(tmp_path):
    """Two of the verifier's leash notes. (a) ``alcohol_clinic`` and ``coffee_shop_id`` were
    dropped because a lifestyle word scoped them out before the grouping name was read; asked now,
    their values leading the guess. (b) A uniform birth year guessed "groups the participants" from
    its values alone blocked "nothing groups them" with only the cluster and the limitation exits,
    and the limitation's record said it read as a possible grouping; the honest answer, its reading
    confirmed "no", is now an exit, after which "nothing" records cleanly. Reference: pandas counts
    of the file's own cells, Spearman's ρ written out, and the sentences verbatim."""
    named = _named_both_ways()
    years = _birth_years()
    for frame, c in ((named, "alcohol_clinic"), (named, "coffee_shop_id"), (years, "birth_year")):
        counts = frame[c].value_counts()
        assert len(counts) > 10 and counts.median() >= 2, c
    rho = _spearman_by_hand(named["alcohol_clinic"])
    assert rho > -0.5  # labels: its counts follow no order
    assert _spearman_by_hand(years["birth_year"]) > -0.5  # a measurement shaped as labels
    truth = Truth({"code_or_count:age": "amount", "code_or_count:alcohol_clinic": "code",
                   "code_or_count:sbp": "amount", "code_or_count:birth_year": "amount"},
                  fixture="the named-both-ways and birth-year tables")
    with local_server(tmp_path / "home") as client:
        drive = open_project(client, _csv(named, tmp_path, "named"), truth)
        asked = _to_clusters(drive, {"participant_id": "identifier", "x": "exposure",
                                     **{c: "covariate" for c in ("alcohol_clinic", "coffee_shop_id",
                                                                 "sex", "age", "sbp")}})
        assert {c: (a["guess"], a["by"]) for c, a in asked.items()} == {
            "alcohol_clinic": ("yes", "values"), "coffee_shop_id": ("yes", "values")}
        assert asked["alcohol_clinic"]["why"].startswith(
            "named like a group of participants and like a lifestyle habit, so its values lead "
            f"the guess: whole numbers that repeat as labels do, their counts follow no order "
            f"(ρ = {rho:.2f})")
        assert asked["coffee_shop_id"]["why"].startswith(
            "named like a group of participants and like a lifestyle habit, so its values lead "
            "the guess: codes written with digits that repeat")
        drive.decide({"kind": "set_clusters", "column": "alcohol_clinic", "adjust": "fixed_effects"})
        assert drive.view()["state"]["clusters"]["column"] == "alcohol_clinic"

        drive = open_project(client, _csv(years, tmp_path, "years"), truth)
        asked = _to_clusters(drive, {"participant_id": "identifier", "x": "exposure",
                                     "birth_year": "covariate", "sex": "covariate"})
        assert {c: (a["guess"], a["by"]) for c, a in asked.items()} == {
            "birth_year": ("yes", "values")}
        error = _error(drive.post({"kind": "set_clusters", "column": None}), "grouping_reads")
        assert [e["label"] for e in error["exits"]] == [
            "Adjust for `birth_year` and cluster by it", "`birth_year` marks no group of participants",
            "They group nothing; record that"]
        assert error["message"].endswith(
            "`birth_year` is guessed from its values alone: if it holds a measured value or a "
            "characteristic rather than a group's label, say so, and nothing is recorded as a "
            "limitation.")
        drive.decide(error["exits"][1]["decision"])
        said = _records(drive)[-1]
        assert said["decision"] == {"kind": "confirm_reading", "reading": "cluster",
                                    "column": "birth_year", "value": "no"}
        assert said["sentence"] == (
            "`birth_year` was confirmed as not marking rows that belong together, so the intervals "
            "do not cluster by it, on its own, after the evidence for its reading was read.")
        assert drive.reach("clusters")["status"] == "skipped"  # nothing left to ask
        r = drive.post({"kind": "set_clusters", "column": None})
        assert r.status_code == 200, r.text[:600]
        record = _records(drive)[-1]
        assert record["decision"]["none_of"] == []
        assert record["sentence"] == ("Nothing groups the participants above the person; each is "
                                      "analyzed as independent.")


def _facilities(n_fac: int = 30, seed: int = 909) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    g = _sizes(rng, n_fac, 540)
    n = len(g)
    shared = rng.normal(0, 1.2, n_fac)[g]
    x = (0.5 * shared + rng.normal(0, 1, n)).round(3)
    age = rng.normal(55, 10, n).round(1)
    y = (2 + 0.6 * x + 0.03 * age + shared + rng.normal(0, 1, n)).round(3)
    return pd.DataFrame({"participant_id": [f"Q{i:04d}" for i in range(n)],
                         "facility_id": 501 + g, "x": x, "age": age, "y": y})


SCORE_KEYS = ("r2", "rmse", "mae", "auc", "brier", "log_loss", "c_index")


def _scores_in(obj: Any) -> list[str]:
    """Every key in a served artifact that holds a fit statistic with a number."""
    out: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in SCORE_KEYS and (isinstance(v, (int, float)) or (
                    isinstance(v, dict) and any(isinstance(x, (int, float)) for x in v.values()))):
                out.append(str(k))
            out += _scores_in(v)
    elif isinstance(obj, list):
        for v in obj:
            out += _scores_in(v)
    return out


def test_2_3_a_grouping_named_by_structure_clusters_the_intervals_and_fit_statistics_wait(tmp_path):
    """The verifier's case: ``facility_id`` read as a row identifier beside the unit the grain
    names, so the grouping question was skipped and the intervals were HC3. Asked now, and answered, the
    intervals are CR2 by it with Bell–McCaffrey df (reference: both written out by definition,
    ``references.cr2_by_definition``). Then an answer the estimate rests on is taken back: the fit
    serves no R², RMSE, MAE or other score while the plan is open, and says why. Answered, it serves
    none either (MODELING_SEQUENCE §0 ruling 13, EXPLORE's ``withhold_scores``; wave 2b
    integration): under inference no cross-validated score is shown."""
    frame = _facilities()
    path = _csv(frame, tmp_path, "facilities")
    X = np.column_stack([np.ones(len(frame)), frame["x"], frame["age"]])
    yv = frame["y"].to_numpy(float)
    beta = np.linalg.lstsq(X, yv, rcond=None)[0]
    V, df = ref.cr2_by_definition(X, yv - X @ beta, pd.factorize(frame["facility_id"])[0])
    from scipy import stats

    q = stats.t.ppf(0.975, df[1])
    expected = (beta[1] - q * math.sqrt(V[1, 1]), beta[1] + q * math.sqrt(V[1, 1]))
    truth = Truth({"code_or_count:age": "amount", "adjust:age": "no,yes,no", "exposure:y": "x"},
                  fixture="the facilities table")
    roles = {"participant_id": "identifier", "facility_id": "identifier", "x": "exposure",
             "age": "covariate"}
    with local_server(tmp_path / "home") as client:
        drive = open_project(client, path, truth)
        drive.decide({"kind": "set_lens", "lenses": ["clinical"]})
        drive.reach("target")
        drive.decide({"kind": "set_target", "column": "y"})
        drive.answer("task", {"kind": "set_task", "column": "y", "task": "regression"})
        drive.reach("purpose")
        drive.decide({"kind": "set_purpose", "purpose": "inference"})
        drive.decide({"kind": "set_grain", "grain": "one_row_per_unit",
                      "id_column": "participant_id"})
        drive.reach("roles")
        drive.decide_roles(roles)
        assert drive.reach("clusters")["status"] == "open"
        card = _until(lambda: drive.artifact("proposals").get("grouping"), lambda c: bool(c))
        assert [(c["column"], c["guess"]) for c in card["columns"]] == [("facility_id", "yes")]
        drive.decide({"kind": "set_clusters", "column": "facility_id", "adjust": "cluster_only"})
        drive.answer("exclusions", {"kind": "set_exclusions", "rules": []})
        drive.answer("missing", {"kind": "set_missing", "strategy": "complete_case"})
        drive.answer("split", {"kind": "set_split", "holdout": 0.0, "seed": 0, "folds": 5})
        drive.reach("estimand")
        answer_estimand(drive, "x")
        drive.reach("adjustment")
        tap = drive.post({"kind": "set_adjustment", "exposure": "x",
                          "answers": {"age": {"causes_exposure": "no", "causes_outcome": "yes",
                                              "after_exposure": "no"}}})
        assert tap.status_code == 200, tap.text[:600]
        drive.reach("models")
        drive.decide({"kind": "select_models", "models": ["linear"]})
        fit = _served(drive, "fit")
        model = fit["models"][0]
        row = next(r for r in model["coefficients"] if r["feature"] == "x")
        assert model["inference"]["covariance"] == "CR2"
        assert model["inference"]["grouped_by"] == "facility_id"
        assert (row["ci_low"], row["ci_high"]) == pytest.approx(expected, abs=1e-6)
        assert row["df"] == pytest.approx(df[1], rel=1e-6)
        # Answered, still no score (ruling 13), and the served fit says why.
        from turbotab.core.stages.evaluation import NO_SCORE_UNDER_INFERENCE

        assert _scores_in(fit) == [] and fit["cv_definition"] == NO_SCORE_UNDER_INFERENCE
        assert model["cv"] == {} and model["baseline"]["value"] is None

        # Taking the adjustment answer back reopens the plan: no estimate and no fit statistic.
        tap_id = next(x["id"] for x in _records(drive) if x["decision"]["kind"] == "set_adjustment")
        drive.decide({"kind": "revert", "decision_id": tap_id})
        withheld = _until(lambda: _served(drive, "fit"), lambda f: bool(f.get("withheld")))
        assert _scores_in(withheld) == []
        for m in withheld["models"]:
            assert m["cv"] == {} and m["coefficients"] is None and m["baseline"]["value"] is None
            assert m["performance"] is None and m["compared_on"] is None
            assert estimand.SCORES_WITHHELD in m["concerns"]
        assert withheld["selection"] is None and withheld["result"] is None
        drive.decide({"kind": "set_adjustment", "exposure": "x",
                      "answers": {"age": {"causes_exposure": "no", "causes_outcome": "yes",
                                          "after_exposure": "no"}}})
        served = _until(lambda: _served(drive, "fit"), lambda f: not f.get("withheld"))
        assert _scores_in(served) == [] and served["models"][0]["coefficients"]  # ruling 13

    # Under prediction a held follow-up question withholds the estimates only: there the scores are
    # the result, so they stay.
    gate = {"question": "follow_up", "reason": "No estimate is shown until …", "exits": [],
            "purpose": "prediction"}
    kept = estimand.withhold("fit", {"models": [{"cv": {"r2": {"mean": 0.3}}, "coefficients": []}]},
                             gate)
    assert kept["models"][0]["cv"] == {"r2": {"mean": 0.3}}


# ═════════════════════════════════════════════════════════════════════════════
# 4 · a bulk roles answer keeps the roles confirmed one by one
# ═════════════════════════════════════════════════════════════════════════════


def _plain(n: int = 400, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    f = pd.DataFrame({"participant_id": [f"P{i:04d}" for i in range(n)],
                      "age": rng.normal(50, 10, n).round(1), "x": rng.normal(0, 1, n).round(3),
                      "w1": rng.normal(0, 1, n).round(2), "z_score": rng.normal(0, 1, n).round(2)})
    f["y"] = (0.5 * f["x"] + 0.02 * f["age"] + 0.3 * f["w1"] + rng.normal(0, 1, n)).round(3)
    return f


def test_4_a_bulk_roles_answer_keeps_the_roles_confirmed_one_by_one(tmp_path):
    """Roles proposed below high are confirmed one by one, the plan is answered, and then a bulk
    roles answer changes one column: the record lists no confirmed role as riding along, every one
    stays settled, and the adjustment set is not reopened. Reference: the ledger's own rule as
    BLUEPRINT §14.1 states it (a role is settled once confirmed with the role recorded)."""
    from turbotab.core.readings import role_reading

    path = _csv(_plain(), tmp_path, "plain")
    truth = Truth({"code_or_count:age": "amount", "exposure:y": "x"}, fixture="the plain table")
    with local_server(tmp_path / "home") as client:
        drive = open_project(client, path, truth)
        drive.decide({"kind": "set_lens", "lenses": ["clinical"]})
        drive.reach("target")
        drive.decide({"kind": "set_target", "column": "y"})
        drive.answer("task", {"kind": "set_task", "column": "y", "task": "regression"})
        drive.reach("purpose")
        drive.decide({"kind": "set_purpose", "purpose": "inference"})
        drive.answer("grain", {"kind": "set_grain", "grain": "one_row_per_unit"})
        drive.reach("roles")
        proposals = drive.artifact("roles")["columns"]
        bulk = {p["column"]: p["proposed"] for p in proposals}
        bulk.update({"x": "exposure", "w1": "covariate", "z_score": "covariate"})
        assert drive.post({"kind": "set_roles", "roles": bulk}).status_code == 200
        riding = _records(drive)[-1]["decision"]["unconfirmed"]
        assert riding  # proposals below high rode along unconfirmed
        for column in riding:  # each confirmed on its own
            r = drive.post({"kind": "confirm_role", "column": column, "role": bulk[column]})
            assert r.status_code == 200, r.text[:600]
        drive.answer("exclusions", {"kind": "set_exclusions", "rules": []})
        drive.answer("missing", {"kind": "set_missing", "strategy": "complete_case"})
        drive.answer("split", {"kind": "set_split", "holdout": 0.0, "seed": 0, "folds": 5})
        drive.reach("estimand")
        answer_estimand(drive, "x")
        drive.reach("adjustment")
        answers = {c: {"causes_exposure": "no", "causes_outcome": "yes", "after_exposure": "no"}
                   for c in ("age", "w1", "z_score")}
        assert drive.post({"kind": "set_adjustment", "exposure": "x",
                           "answers": answers}).status_code == 200
        assert drive.reach("adjustment")["status"] == "answered"
        changed = {**bulk, "z_score": "excluded"}
        assert drive.post({"kind": "set_roles", "roles": changed}).status_code == 200
        record = _records(drive)[-1]
        assert record["decision"]["unconfirmed"] == []
        state = d.ProjectState(**drive.view()["state"])
        assert all(role_reading(state, c).settled for c in changed)
        steps = {s["key"]: s["status"] for s in drive.view()["interview"]}
        assert steps["estimand"] == "answered" and steps["adjustment"] == "answered"

    # The other way a confirmation was lost: a role the user gave as their own (not the proposal),
    # re-recorded unchanged by a later bulk answer once the proposal has come to match it.
    roles = {"age": "covariate", "x": "exposure", "batch": "covariate"}
    first = d.SetRoles(roles=roles)  # the user's own answer: nothing rode along
    state = d.fold([d.DecisionRecord(id="r1", seq=1, at=AT, decision=first)])
    shifted = {"columns": [{"column": "batch", "proposed": "covariate", "confidence": "medium"},
                           {"column": "age", "proposed": "covariate", "confidence": "high"},
                           {"column": "x", "proposed": "exposure", "confidence": "medium"}]}
    again = d.validate({"kind": "set_roles", "roles": {**roles, "age": "covariate"}},
                       {"state": state, "artifact": lambda stage: shifted if stage == "roles"
                        else None, "target": "y", "columns": [*roles, "y"]})
    assert again.unconfirmed == []
    fresh = d.validate({"kind": "set_roles", "roles": roles},
                       {"state": d.ProjectState(), "artifact": lambda stage: shifted
                        if stage == "roles" else None, "target": "y", "columns": [*roles, "y"]})
    assert sorted(fresh.unconfirmed) == ["batch", "x"]  # with nothing settled, they ride along


# ═════════════════════════════════════════════════════════════════════════════
# 5 · repeats over imputed copies: block and record, Rubin's rules the first exit
# ═════════════════════════════════════════════════════════════════════════════


def _copies(seed: int = 2006) -> pd.DataFrame:
    """NHANES DXA's shape: each SEQN five times, ``_MULT_`` 1–5, a third of the people with an
    imputed percent fat that differs between copies."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(200):
        age = round(float(rng.normal(45, 12)), 1)
        fiber = round(float(rng.gamma(4, 4)), 1)
        fat = 30 + 0.1 * (age - 45) - 0.2 * (fiber - 16) + float(rng.normal(0, 4))
        imputed = rng.random() < 1 / 3
        for k in range(1, 6):
            rows.append({"SEQN": 40000 + i, "_MULT_": k, "RIDAGEYR": age, "fiber_g": fiber,
                         "DXDTOPF": round(fat + (float(rng.normal(0, 2)) if imputed else 0.0), 1)})
    return pd.DataFrame(rows)


def test_5_repeats_over_imputed_copies_are_blocked_and_recorded(tmp_path):
    """Reference: pandas over the file (every SEQN holds ``_MULT_`` 1–5 once, and the outcome
    differs between copies for some), and CDC's guidance quoted in the refusal."""
    frame = _copies()
    assert all(sorted(v) == [1, 2, 3, 4, 5] for v in frame.groupby("SEQN")["_MULT_"].apply(list))
    assert (frame.groupby("SEQN")["DXDTOPF"].nunique() > 1).any()
    path = _csv(frame, tmp_path, "dxa")
    truth = Truth({"code_or_count:_MULT_": "code"}, fixture="the DXA copies")

    def to_repeat_kind(drive: Any, purpose: str) -> None:
        drive.decide({"kind": "set_lens", "lenses": ["clinical"]})
        drive.reach("target")
        drive.decide({"kind": "set_target", "column": "DXDTOPF"})
        drive.answer("task", {"kind": "set_task", "column": "DXDTOPF", "task": "regression"})
        drive.reach("purpose")
        drive.decide({"kind": "set_purpose", "purpose": purpose})
        drive.reach("grain")
        drive.decide({"kind": "set_grain", "grain": "repeated", "id_column": "SEQN"})
        assert drive.reach("repeat_kind")["status"] == "open"
        reading = drive.artifact("structure")["repeats"]
        assert reading["reading"] == "imputed_copies" and reading["implicate_column"] == "_MULT_"

    with local_server(tmp_path / "home") as client:
        drive = open_project(client, path, truth)
        to_repeat_kind(drive, "inference")
        error = _error(drive.post({"kind": "set_repeat_kind", "repeat_kind": "repeats"}),
                       "copies_read_as_repeats")
        assert f'"{CDC_DXA}"' in error["message"] and "Rubin's rules" in error["message"]
        assert error["exits"][0]["decision"] == {
            "kind": "set_repeat_kind", "repeat_kind": "imputed_copies", "time_column": None,
            "levels": None, "implicate_column": "_MULT_", "acknowledged": False}
        assert error["exits"][1]["decision"]["acknowledged"] is True
        _error(drive.post({"kind": "set_repeat_kind", "repeat_kind": "time_points"}),
               "copies_read_as_repeats")
        drive.decide(error["exits"][1]["decision"])
        assert _records(drive)[-1]["sentence"] == (
            "Each `SEQN`'s rows were taken as repeated measurements of the same quantity, not "
            "different time points; recorded as a limitation: the rows read as imputed copies of "
            "one record, which were not pooled by Rubin's rules, and analyzing imputed copies as "
            "records, or by their mean, treats imputed values as measured, so the intervals leave "
            "out the imputation's uncertainty.")
        drive.decide(error["exits"][0]["decision"])  # the Rubin's-rules answer, taken
        said = _records(drive)[-1]["sentence"]
        assert "pooled by Rubin's rules" in said

    # The other way in: "repeats" under prediction, then the purpose made inference.
    with local_server(tmp_path / "home2") as client:
        drive = open_project(client, path, truth)
        to_repeat_kind(drive, "prediction")
        drive.decide({"kind": "set_repeat_kind", "repeat_kind": "repeats"})
        error = _error(drive.post({"kind": "set_purpose", "purpose": "inference"}),
                       "copies_read_as_repeats")
        assert error["exits"][0]["decision"]["repeat_kind"] == "imputed_copies"
        drive.decide(error["exits"][0]["decision"])
        drive.decide({"kind": "set_purpose", "purpose": "inference"})
        assert drive.view()["state"]["purpose"] == "inference"


# ═════════════════════════════════════════════════════════════════════════════
# 6 · censored values and text amounts
# ═════════════════════════════════════════════════════════════════════════════


def _labs(n: int = 300, seed: int = 66) -> tuple[pd.DataFrame, list[int]]:
    rng = np.random.default_rng(seed)
    crp = rng.lognormal(np.log(1.5), 0.8, n).round(2).astype(object)
    crp[rng.random(n) < 0.15] = "<0.3"
    wbc = rng.normal(7, 2, n).round(1).astype(object)
    wbc[:6], wbc[6:10] = "TNTC", "QNS"
    dbp = rng.normal(75, 10, n).round(0).astype(object)
    bad = [11, 120, 250]
    dbp[bad] = [999, 0, 350]
    for i in range(3, n, 47):
        if i not in bad:
            dbp[i] = "."
    frame = pd.DataFrame({"pid": [f"L{i:04d}" for i in range(n)], "age": rng.integers(20, 80, n),
                          "crp_mg_l": crp, "wbc": wbc, "dbp": dbp,
                          "y": rng.normal(0, 1, n).round(3)})
    return frame, bad


def test_6_censored_values_are_left_to_their_repair_and_text_amounts_are_checked(tmp_path):
    """Reference: pandas counts of the file's own cells. ``crp_mg_l``'s values below a detection
    limit have the app's own finding with its repair, so the lab pack's censored-values finding
    leaves them to it and speaks of ``wbc``'s TNTC and QNS alone; with no such column left it is
    not spoken at all. ``dbp`` (SAS "." beside 999, 0 and 350), confirmed "amount" through the
    ledger, is checked by the plausibility checks as numbers."""
    frame, bad = _labs()
    n_below = int((frame["crp_mg_l"].astype(str) == "<0.3").sum())
    path = _csv(frame, tmp_path, "labs")
    truth = Truth({"code_or_count:age": "amount"}, fixture="the labs table")
    with local_server(tmp_path / "home") as client:
        drive = open_project(client, path, truth)
        drive.decide({"kind": "set_lens", "lenses": ["clinical"]})
        drive.reach("target")
        drive.decide({"kind": "set_target", "column": "y"})
        findings = drive.artifact("findings")["findings"]
        below = next(f for f in findings if f["id"] == "below_detection__crp_mg_l")
        assert below["repairs"] and "No control for this yet" not in below["summary"]
        assert f"`{n_below}`" in below["title"]
        censored = [f for f in findings if f["id"].startswith("pack::clinical::censored_values")]
        assert len(censored) == 1
        c = censored[0]
        assert c["affected_columns"] == ["wbc"]
        assert "No control for this yet" not in c["summary"] or "crp_mg_l" not in c["summary"]
        assert "`crp_mg_l`" not in c["summary"]
        assert ("Values below a detection limit in `crp_mg_l` are read by its own below-detection "
                "repair (half the limit, or the limit over √2), not here." in c["detail"])
        assert "`wbc`: `6` too numerous to count" in c["detail"]
        impossible = [f for f in findings if f["id"].startswith("pack::clinical::impossible")
                      and "dbp" in f["affected_columns"]]
        assert impossible == []  # as text, dbp's numbers were never checked
        drive.decide({"kind": "confirm_reading", "reading": "code_or_count", "column": "dbp",
                      "value": "amount"})
        findings = _until(lambda: drive.artifact("findings")["findings"],
                          lambda fs: any(f["id"].startswith("pack::clinical::impossible")
                                         and "dbp" in f["affected_columns"] for f in fs))
    impossible = next(f for f in findings if f["id"].startswith("pack::clinical::impossible")
                      and "dbp" in f["affected_columns"])
    numbers = pd.to_numeric(frame["dbp"], errors="coerce")
    assert int(numbers.isna().sum()) == len(range(3, len(frame), 47))
    assert any(f"`{len(bad)}`" in str(impossible.get(k)) for k in ("title", "summary", "detail"))

    # With no TNTC, ULOQ or failure left, the censored-values finding is not spoken at all.
    only = frame.drop(columns=["wbc"])
    path = _csv(only, tmp_path, "labs_crp_only")
    with local_server(tmp_path / "home2") as client:
        drive = open_project(client, path, truth)
        drive.decide({"kind": "set_lens", "lenses": ["clinical"]})
        drive.reach("target")
        drive.decide({"kind": "set_target", "column": "y"})
        findings = drive.artifact("findings")["findings"]
    assert not [f for f in findings if f["id"].startswith("pack::clinical::censored_values")]
    assert any(f["id"] == "below_detection__crp_mg_l" and f["repairs"] for f in findings)


# ═════════════════════════════════════════════════════════════════════════════
# 7 · ruling 14: the E-value of a difference is standardized by the estimand's SD
# ═════════════════════════════════════════════════════════════════════════════


def _surveyed(seed: int = 14) -> pd.DataFrame:
    """15 strata of two PSUs of 40, informative weights; the outcome's spread differs between the
    heavily and lightly weighted strata, so the design-weighted SD differs from the sample SD."""
    rng = np.random.default_rng(seed)
    H, per = 15, 40
    n = H * 2 * per
    stratum = np.repeat(np.arange(1, H + 1), 2 * per)
    psu = np.tile(np.repeat([1, 2], per), H)
    shared = rng.normal(scale=0.5, size=H * 2)[np.repeat(np.arange(H * 2), per)]
    age = (rng.normal(50, 10, n) + 3 * shared).round(1)
    income = (rng.normal(5, 2, n) + shared).round(2)
    fiber = (15 + 0.6 * (income - 5) + 0.05 * (age - 50) + rng.normal(0, 4, n)).round(1)
    weight = (np.exp(rng.normal(scale=0.4, size=n)) * (1 + 2 * (stratum % 3)) * 1000).round(1)
    spread = 4 + 6 * (stratum % 3)
    sbp = (120 + 0.4 * (age - 50) - 1.0 * (income - 5) - 0.4 * fiber + 2 * shared
           + rng.normal(0, 1, n) * spread).round(1)
    return pd.DataFrame({"SEQN": np.arange(1, n + 1), "age": age, "income": income,
                         "fiber": fiber, "sbp": sbp, "WTMEC2YR": weight, "SDMVSTRA": stratum,
                         "SDMVPSU": psu})


EVALUE_R = """
suppressMessages({library(survey); library(EValue)})
df <- read.csv(rows_csv)
des <- svydesign(ids = ~SDMVPSU, strata = ~SDMVSTRA, weights = ~WTMEC2YR, nest = TRUE, data = df)
sd_w <- sqrt(as.numeric(coef(svyvar(~sbp, des))))
sd_s <- sd(df$sbp)
est <- read.csv(est_csv)
ev <- function(e, s, sd) { r <- suppressMessages(evalues.OLS(est = e, se = s, sd = sd))
  list(point = r[2, 1], limit = if (is.na(r[2, 2])) r[2, 3] else r[2, 2]) }
out(list(sd_w = sd_w, sd_s = sd_s,
         effects = ev(est$estimate[1], est$se[1], sd_w),
         causal = ev(est$estimate[2], est$se[2], sd_w),
         sample = ev(est$estimate[3], est$se[3], sd_s)))
"""
ALL4 = ["no_unmeasured_confounding", "positivity", "consistency", "time_ordering"]
POPULATION_CLAUSE = ("Sensitivity to unmeasured confounding is reported by the E-value for the "
                     "estimate and for the confidence limit nearer the null, the difference "
                     "standardized by the outcome's design-weighted standard deviation in the "
                     "surveyed population, never as a pass or a fail.")

LANE_POPULATION_CLAUSE = (
    "Sensitivity to unmeasured confounding is reported by the Cinelli–Hazlett robustness value of "
    "the final least-squares step, the outcome's residual on the exposure's (the form the "
    "omitted-variable bound of Chernozhukov, Cinelli, Newey, Sharma & Syrgkanis 2022, NBER w30302 "
    "takes in the partially linear model), the median over the sample splits (its form for the 95% "
    "interval is not reported: it assumes a classical standard error, and the interval reported "
    "is the estimating equation's) and by the E-value for the estimate and for the confidence "
    "limit nearer the null, the difference standardized by the outcome's design-weighted standard "
    "deviation in the surveyed population, never as a pass or a fail.")


@needs_r
def test_7_the_e_value_of_a_difference_is_standardized_by_the_estimand_sd(tmp_path):
    """Reference: R's ``survey::svyvar`` for the design-weighted variance and ``sd`` for the
    sample's, and ``EValue::evalues.OLS(est, se, sd)`` on the app's own estimates and standard
    errors, to 1e-8; the methods clause verbatim in both the effects stage and the causal lane."""
    frame = _surveyed()
    path = _csv(frame, tmp_path, "surveyed")
    truth = Truth({"adjust:age": "yes,yes,no", "adjust:income": "yes,yes,no",
                   "cluster:SEQN": "no", "exposure:sbp": "fiber"}, fixture="the surveyed cohort")
    roles = {"SEQN": "identifier", "age": "covariate", "income": "covariate", "fiber": "exposure",
             "WTMEC2YR": "design", "SDMVSTRA": "design", "SDMVPSU": "design"}
    with local_server(tmp_path / "home") as client:
        drive = open_project(client, path, truth)
        drive.decide({"kind": "set_lens", "lenses": ["clinical"]})
        drive.reach("target")
        drive.decide({"kind": "set_target", "column": "sbp"})
        drive.answer("task", {"kind": "set_task", "column": "sbp", "task": "regression"})
        drive.reach("purpose")
        drive.decide({"kind": "set_purpose", "purpose": "inference"})
        drive.answer("grain", {"kind": "set_grain", "grain": "one_row_per_unit"})
        drive.reach("roles")
        drive.decide_roles(roles)
        drive.answer("survey", {"kind": "set_survey", "estimand": "population",
                                "weight": "WTMEC2YR", "strata": "SDMVSTRA", "psu": "SDMVPSU"})
        drive.answer("exclusions", {"kind": "set_exclusions", "rules": []})
        drive.answer("missing", {"kind": "set_missing", "strategy": "complete_case"})
        drive.answer("split", {"kind": "set_split", "holdout": 0.0, "seed": 0, "folds": 5})
        drive.exposure = "fiber"
        drive.answer_plan("fiber")
        drive.reach("causal")
        drive.decide({"kind": "set_causal", "exposure": "fiber", "method": "dml_plr",
                      "learner": "linear", "repetitions": 3, "assumptions": ALL4})
        drive.decide({"kind": "select_models", "models": ["linear"]})
        art = _until(lambda: drive.artifact("causal"), lambda a: bool(a.get("estimates")))
        effects = _served(drive, "effects")
        drive.decide({"kind": "set_survey", "estimand": "sample", "weight": "WTMEC2YR",
                      "strata": "SDMVSTRA", "psu": "SDMVPSU"})
        sample = _until(lambda: _served(drive, "effects"),
                        lambda e: bool(e.get("families")) and (
                            e["families"][0]["sensitivity"][0]["e_value"] or {}).get("sd_basis")
                        == "sample")
    lane, primary = art["sensitivity"], effects["families"][0]["sensitivity"][0]
    unweighted = sample["families"][0]["sensitivity"][0]
    row = next(s for s in effects["families"][0]["sequence"] if s["key"] == "model_2")["effects"][0]
    row_s = next(s for s in sample["families"][0]["sequence"]
                 if s["key"] == "model_2")["effects"][0]
    est = pd.DataFrame({"estimate": [row["estimate"], art["estimates"][0]["estimate"],
                                     row_s["estimate"]],
                        "se": [row["se"], art["estimates"][0]["se"], row_s["se"]]})
    r = run_r(EVALUE_R, {"rows": frame, "est": est}, tmp_path / "r")
    assert r["sd_w"] != pytest.approx(r["sd_s"], rel=1e-3)  # the two SDs differ on this design
    assert lane["e_value"]["sd_basis"] == primary["e_value"]["sd_basis"] == "design_weighted"
    assert lane["e_value"]["sd"] == pytest.approx(r["sd_w"], rel=1e-10)
    assert primary["e_value"]["sd"] == pytest.approx(r["sd_w"], rel=1e-10)
    assert unweighted["e_value"]["sd_basis"] == "sample"
    assert unweighted["e_value"]["sd"] == pytest.approx(r["sd_s"], rel=1e-10)
    for found, expected in ((primary, r["effects"]), (lane, r["causal"]),
                            (unweighted, r["sample"])):
        assert found["e_value"]["point"] == pytest.approx(expected["point"], rel=1e-8, abs=1e-8)
        assert found["e_value"]["limit"] == pytest.approx(expected["limit"], rel=1e-8, abs=1e-8)
    # The partially linear model's robustness value of its final least-squares step ranks first
    # (wave 2a repairs); the E-value's clause after it names the population's SD.
    assert art["methods"].endswith(" " + LANE_POPULATION_CLAUSE)
    assert (" " + POPULATION_CLAUSE) in effects["methods"]
    assert "design-weighted" not in sample["methods"]


def test_7_the_design_weighted_sd_is_svyvar_written_out():
    """``estimand_sd`` against the survey package's definition written out by hand: ``n/(n−1)``
    times the weighted mean of squared deviations from the weighted mean, over positive weights."""
    from turbotab.core.models.effects import estimand_sd

    rng = np.random.default_rng(7)
    y = rng.normal(0, 3, 200)
    w = rng.uniform(0.5, 4, 200)
    w[:5] = 0.0  # outside the domain
    keep = w > 0
    mean = np.sum(w[keep] * y[keep]) / np.sum(w[keep])
    var = keep.sum() / (keep.sum() - 1) * np.sum(w[keep] * (y[keep] - mean) ** 2) / np.sum(w[keep])
    assert estimand_sd(y, w) == (pytest.approx(math.sqrt(var), rel=1e-12), "design_weighted")
    assert estimand_sd(y) == (pytest.approx(float(np.std(y, ddof=1)), rel=1e-12), "sample")


# ═════════════════════════════════════════════════════════════════════════════
# the chain: every LEASH relation names the test that asserts it fires
# ═════════════════════════════════════════════════════════════════════════════

HERE = "turbotab.core.tests.acceptance.test_leash::"
RELATION_TESTS = {
    "guess-pair-for-measurements": HERE + "test_1_the_adjustment_card_stays_light_at_thirty_covariates",
    "guess-outcome-measure-out": HERE + "test_1_the_adjustment_card_stays_light_at_thirty_covariates",
    "guess-baseline-outcome-kind-pair": HERE + "test_1_a_baseline_level_of_the_outcomes_kind_beside_an_incident_event_is_the_pair",
    "block-settles-listed": HERE + "test_1_the_adjustment_card_stays_light_at_thirty_covariates",
    "mediator-kept-blocked": HERE + "test_1_the_adjustment_card_stays_light_at_thirty_covariates",
    "bulk-roles-keep-confirmations": HERE + "test_4_a_bulk_roles_answer_keeps_the_roles_confirmed_one_by_one",
    "structure-asks": HERE + "test_2_the_grouping_question_is_asked_of_any_column_that_can_group_rows",
    "grouping-implies-cr2": HERE + "test_2_3_a_grouping_named_by_structure_clusters_the_intervals_and_fit_statistics_wait",
    "none-over-grouping": HERE + "test_2_the_grouping_question_is_asked_of_any_column_that_can_group_rows",
    "copies-as-repeats-blocked": HERE + "test_5_repeats_over_imputed_copies_are_blocked_and_recorded",
    "copies-imply-pooling": HERE + "test_5_repeats_over_imputed_copies_are_blocked_and_recorded",
    "evalue-population-sd": HERE + "test_7_the_e_value_of_a_difference_is_standardized_by_the_estimand_sd",
    "evalue-sample-sd": HERE + "test_7_the_e_value_of_a_difference_is_standardized_by_the_estimand_sd",
    "plan-open-withholds-scores": HERE + "test_2_3_a_grouping_named_by_structure_clusters_the_intervals_and_fit_statistics_wait",
    "censored-left-to-repair": HERE + "test_6_censored_values_are_left_to_their_repair_and_text_amounts_are_checked",
    "text-amount-checked": HERE + "test_6_censored_values_are_left_to_their_repair_and_text_amounts_are_checked",
}


def test_every_leash_relation_is_asserted_and_enters_through_the_one_registry():
    """BLUEPRINT §13: each LEASH method's contract (slot, scope, needs, routing with both labels
    and a rung per purpose, storyboard, sentence, relations) is in the one registry; every relation
    names the code that makes it fire and the test that asserts it; a conflict is blocked and
    recorded with its exits, never silent."""
    from turbotab.core import contracts as C

    mine = [c for c in C.contracts().values() if c.package == "LEASH"]
    assert {c.key for c in mine} == {"adjustment_guesses", "grouping_by_structure",
                                     "copies_not_repeats", "evalue_sd", "fit_statistics_withheld",
                                     "censored_below_detection"}
    assert {r.id for c in mine for r in c.relations} == set(RELATION_TESTS)
    for relation, where in RELATION_TESTS.items():
        module, name = where.split("::")
        assert callable(getattr(importlib.import_module(module), name, None)), (relation, where)
    for c in mine:
        assert c.slot in C.SLOTS and c.scope in C.SCOPES and c.scope_note, c.key
        assert c.needs and c.question and c.storyboard and c.options and c.sentence, c.key
        module, _, name = str(c.sentence).partition(":")
        assert callable(getattr(importlib.import_module(module), name)), c.key
        for o in c.options:
            assert o.customary and all(o.sound[p] for p in C.PURPOSES), (c.key, o.key)
            assert all(o.rung[p] in C.RUNGS for p in C.PURPOSES), (c.key, o.key)
        for r in c.relations:
            assert r.condition and r.says and r.enforced_by, (c.key, r.id)
            module, name = r.enforced_by.split(":")
            assert callable(getattr(importlib.import_module(module), name)), r.enforced_by
            assert bool(r.exits) == (r.kind == "conflicts"), (c.key, r.id)
            if r.kind == "conflicts":
                assert r.rung == "block_and_record", (c.key, r.id)
    C.run_order(list(C.contracts()))
