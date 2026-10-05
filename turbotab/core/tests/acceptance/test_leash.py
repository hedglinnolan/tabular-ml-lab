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
    assert sentences["LBXGH"] == ("For the effect of `DR1TSUGR`, by the disjunctive cause "
                                  "criterion: `LBXGH` is a consequence of the exposure (a possible "
                                  "collider), left out.")


def test_1_the_guess_follows_the_exposure_outcome_pairing():
    """The packs' table, row by row, under three pairings (reference: NUTRITION_PACK §08 as
    written): a dietary exposure and a glycemic outcome; a dietary exposure and an event over
    follow-up; an LDL exposure and an event."""
    from turbotab.core.covariate_guesses import guess

    def state(exposure: str, target: str, task: str) -> d.ProjectState:
        return d.ProjectState(roles={exposure: "exposure"}, target=target, task=task,
                              purpose="inference", estimand=d.EstimandSpec(
                                  exposure=exposure, effect="total",
                                  measure="mean_difference" if task == "regression"
                                  else "hazard_ratio"))

    cross = state("DR1TSUGR", "LBXGLU", "regression")
    cohort = state("fiber_g", "death", "time_to_event")
    ldl = state("LBDLDL", "chd", "time_to_event")
    expect = [
        (cross, "LBXGH", OUTCOME_KIND, "another measure of the outcome"),
        (cross, "hba1c", OUTCOME_KIND, "another measure of the outcome"),
        (cross, "LBDHDD", TIMING, "measured at the same visit as the exposure"),
        (cross, "metformin", TIMING, "Tobin et al. 2005"),
        (cross, "statin_use", TIMING, "a treatment the exposure may have led to"),
        (cross, "smoking", PRE, "habits"),
        (cross, "physical_activity", PRE, "habits"),
        (cross, "alcohol_drinks", PRE, "habits"),
        (cohort, "LBXGH", TIMING, "measured at baseline with the exposure"),
        (cohort, "hs_crp", TIMING, "measured at baseline with the exposure"),
        (ldl, "statin_use", TIMING, "it lowers the exposure `LBDLDL`"),
        (ldl, "LBXGH", TIMING, "measured at baseline with the exposure"),
    ]
    for st, column, answers, said in expect:
        found = guess(column, st)
        assert found is not None and dict(found.answers) == answers, (column, found)
        assert said in found.reason or said in found.source, (column, found.reason)
    assert guess("total_cholesterol", ldl) is None  # another measure of the exposure: asked
    coffee = state("coffee_cups", "LBXGLU", "regression")
    assert guess("caffeine_mg", coffee) is None  # the exposure's own habit: asked
    assert dict(guess("smoking", coffee).answers) == PRE
    assert guess("cycle_begin_year", cross) is None  # the packs have no guess


# ═════════════════════════════════════════════════════════════════════════════
# 2 · the grouping question by structure; 3 · a withheld fit serves no fit statistics
# ═════════════════════════════════════════════════════════════════════════════

GROUPED = {"study_site": "yes", "region": "yes", "trial_site": "yes", "recruitment_centre": "yes",
           "gp_practice": "yes", "hosp": "yes", "facility_id": "yes", "physician_id": "yes",
           "doctor_id": "yes", "cg": "yes", "country_of_birth": "no"}
NEVER = ["sex", "age", "sbp", "x_bell", "education", "bmi"]


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
        nothing = next(e["decision"] for e in error["exits"]
                       if e["label"] == "They group nothing; record that")
        drive.decide(nothing)
        record = _records(drive)[-1]
        assert record["decision"]["none_of"] == list(GROUPED)  # in the table's column order
        assert record["sentence"] == (
            "Nothing groups the participants above the person; each is analyzed as independent, "
            "although `study_site`, `region` and 9 more read as possible groupings of the "
            "participants (a site, centre, household or batch, by name or by values); the answer "
            "was kept over that reading, so the intervals do not cluster by them, and it is a "
            "stated limitation.")
        state = drive.view()["state"]
        assert all(state["reading_confirmations"][f"cluster:{c}"] == "no" for c in GROUPED)

    # Under prediction the question is not widened: no column of these is asked by its values.
    from turbotab.core.groupings import candidates

    st = d.ProjectState(purpose="prediction", target="y", roles=roles)
    assert candidates(st, {"groupings": [{"column": "cg", "levels": 25, "rows": 720,
                                          "median_rows": 28, "kind": "whole_numbers",
                                          "profile": 0.0, "named": False}]}) == []


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
    serves no R², RMSE, MAE or other score while the plan is open, and serves them once answered."""
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
        assert _scores_in(fit)  # once the plan is answered its scores are served

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
        assert _scores_in(served)

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
