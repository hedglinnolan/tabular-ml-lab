"""Settle by values only where the alternatives are excluded (BLUEPRINT §14.3, amendment after the
fifth gate, 2026-10-03): package LEDGER-REPAIR-2.

The fifth gate (``/private/tmp/turbotab-fix/gate5_run.json``, field ``gate5``) found value tests
that real data still fools, and three confirmations the consumers ignored:

* decimal ICD-9-CM codes (307.1, 307.51, 250.02) read as amounts and fit as one slope;
* an NHANES-style skip-pattern gate read as a missingness flag and dropped from the fit;
* a 4-day kcal total (ratio 4.00, next to kJ's 4.184) read as kJ, the screens' exits kJ only;
* an assay run date read as visit time; the outcome's task settled by its dtype;
* a role confirmed as another role, a unit and day count confirmed on their own, and a
  substitution's unit confirmed in kg, each accepted and ignored;
* the leash: total energy asked code-or-amount though the energy answer "none" removed it (its
  "codes" exit crashed the design stage), and a 1/2 sex asked though both answers fit alike.

Section A replays each not-closed item draw for draw from the gate's generators
(``gate5/p1``–``p9``, the seeds named at each), through the real server where the gate drove it;
each server replay first shows the reading asked (or refused) with no number computed, then
answers from the fixture's declared truth (``truths.Truth``), never a constant. Section D is the
exhaustive property: every confirmable kind × every registered consumer of it × every value the
validator accepts, through the decision fold. Section E posts every exit every ask offers and runs
the stage it unblocks. Section F: settlement is visible.

Expected values never come from the code under test: NumPy least squares, pandas counts, the
exact unit factors (NIST SP 811), and quoted primary sources (fetched 2026-10-03).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import pytest

from turbotab.core import decisions as d
from turbotab.core import readings as R
from turbotab.core.decisions import ProjectState
from turbotab.core.tests.acceptance.server_drive import (
    Truth, _post_when_reached, local_server, open_project,
)
from turbotab.core.tests.acceptance.test_discriminate import (
    ORDER, answers, coefficients, drive_unsettled, fit_refused_without_a_number, ols, roles_of,
    write,
)
from turbotab.core.tests.truths import ASKING, asked
from turbotab.core.tests.truths import answers as truth_answers

ROOT = Path(__file__).resolve().parents[4]

# ── primary sources, quoted (fetched 2026-10-03) ─────────────────────────────

BLUEPRINT_EXCLUDED = ("Settle by values only where the alternatives are excluded, not merely "
                      "unlikely.")
BLUEPRINT_SAME_VALUES = "Where two readings produce the same values, the kind is settled by the user."
BLUEPRINT_VISIBLE = ("Readings settled by their values appear on the card under \"read from your "
                     "data\", each with its evidence and a way to change it, and in the methods "
                     "record.")
BLUEPRINT_NEVER = "An ignored confirmation, or an offered answer that fails, is never accepted."
# ICD-9-CM, Volume 1 (2015), icd9data.com:
ICD9 = {307.1: "ICD-9-CM Diagnosis Code 307.1 : Anorexia nervosa",
        307.51: "ICD-9-CM Diagnosis Code 307.51 : Bulimia nervosa",
        250.02: "ICD-9-CM Diagnosis Code 250.02 : Diabetes mellitus without mention of "
                "complication, type II or unspecified type, uncontrolled"}
# NHANES 2017–2018 Alcohol Use (ALQ_J) codebook (wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/
# DataFiles/ALQ_J.htm): ALQ111 "Ever had a drink of any kind of alcohol", code 2 "No", 585, skip
# "End of Section"; ALQ121 code 0 "Never in the last year", 1049, skip "ALQ151"; ALQ130 "Avg #
# alcohol drinks/day - past 12 mos", 2,038 of 5,533 missing.
ALQ111 = ("In {your/SP's} entire life, {have you/has he/has she} had at least 1 drink of any kind "
          "of alcohol, not counting small tastes or sips?")
ALQ111_NO_SKIP = ("2", "No", "End of Section")
ALQ130 = "Avg # alcohol drinks/day - past 12 mos"
# NIST SP 811, Appendix B.9 (exact, boldface): calorie_th → joule 4.184 E+00; kilocalorie_th →
# joule 4.184 E+03.
KJ_PER_KCAL = 4.184
# Kroenke, Spitzer & Williams 2001, J Gen Intern Med 16:606 (PMC1495268): "the PHQ-9 score can
# range from 0 to 27, since each of the 9 items can be scored from 0 (not at all) to 3 (nearly
# every day)".
PHQ9_RANGE = (0, 27)
# The international yard and pound agreement (1959): 1 lb = 0.45359237 kg exactly.
LB_TO_KG = 0.45359237
ATWATER = {"protein": 4.0, "carbohydrate": 4.0, "fat": 9.0, "alcohol": 7.0}  # NUTRITION_PACK §01


def test_the_quoted_rules_are_the_blueprints_own():
    blueprint = " ".join((ROOT / "docs/turbotab-next/BLUEPRINT.md").read_text("utf-8").split())
    for sentence in (BLUEPRINT_EXCLUDED, BLUEPRINT_SAME_VALUES, BLUEPRINT_VISIBLE, BLUEPRINT_NEVER):
        assert sentence in blueprint, sentence


# ═════════════════════════════════════════════════════════════════════════════
# The gate's generators (gate5/p*.py), draw for draw
# ═════════════════════════════════════════════════════════════════════════════


def dsm_table(n: int = 600) -> pd.DataFrame:
    """p2_dsm_codes.py (rng 55002): DSM-IV / ICD-9-CM eating-disorder codes as a predictor."""
    rng = np.random.default_rng(55002)
    f = pd.DataFrame({"participant_id": [f"ED{i:04d}" for i in range(n)]})
    f["dsm_dx"] = rng.choice([307.1, 307.51, 307.5, 307.59], n, p=[0.3, 0.3, 0.3, 0.1])
    f["age"] = rng.normal(24, 5, n).round(1)
    f["energy_kcal"] = rng.normal(1900, 400, n).round(0)
    eff = f["dsm_dx"].map({307.1: -4.0, 307.51: 1.5, 307.5: 0.0, 307.59: -1.0})
    f["bmi"] = (21 + eff + 0.05 * (f["age"] - 24) + 0.001 * (f["energy_kcal"] - 1900)
                + rng.normal(0, 1.5, n)).round(2)
    return f


def skip_table(n: int = 800) -> pd.DataFrame:
    """p3_skip_flag.py (rng 55003): `alcohol_flag` (drank in the past year) gates `alcohol`, blank
    for non-drinkers, as NHANES ALQ111 "No" skips ALQ130."""
    rng = np.random.default_rng(55003)
    f = pd.DataFrame({"participant_id": [f"S{i:04d}" for i in range(n)]})
    f["age"] = rng.normal(50, 12, n).round(1)
    f["alcohol_flag"] = rng.choice([0, 1], n, p=[0.35, 0.65])
    drinks = rng.gamma(2.0, 8.0, n).round(1)
    f["alcohol"] = np.where(f["alcohol_flag"] == 1, drinks, np.nan)
    f["fiber_g"] = rng.gamma(4, 5, n).round(1)
    f["sbp"] = (120 + 0.3 * (f["age"] - 50) + 4.0 * f["alcohol_flag"]
                + 0.25 * np.nan_to_num(f["alcohol"]) - 0.2 * f["fiber_g"]
                + rng.normal(0, 8, n)).round(1)
    return f


SKIP_TRUTH = {"code_or_count:age": "amount", "code_or_count:alcohol_flag": "code",
              "role:alcohol_flag": "covariate", "role:alcohol": "exposure",
              "role:fiber_g": "exposure", "code_or_count:alcohol": "amount"}


def four_day_table(n: int = 500) -> pd.DataFrame:
    """p4_four_day_total.py (rng 55004): a 4-day food record's energy TOTAL beside the
    macronutrients' daily AVERAGE (reconstruction ratio 4.00, within 5% of kJ's 4.184)."""
    rng = np.random.default_rng(55004)
    f = pd.DataFrame({"participant_id": [f"R{i:04d}" for i in range(n)]})
    f["sex"] = rng.choice(["F", "M"], n)
    male = (f["sex"] == "M").to_numpy()
    f["age"] = rng.integers(20, 70, n)
    f["weight"] = np.where(male, rng.normal(84, 13, n), rng.normal(70, 13, n)).round(1)
    f["height"] = np.where(male, rng.normal(176, 7, n), rng.normal(163, 6, n)).round(1)
    daily = np.where(male, rng.normal(2400, 600, n), rng.normal(1900, 500, n)).clip(600)
    share_p = rng.normal(0.16, 0.03, n).clip(0.08)
    share_f = rng.normal(0.35, 0.05, n).clip(0.15)
    share_c = 1 - share_p - share_f
    f["protein_g"] = (daily * share_p / 4).round(1)
    f["fat_g"] = (daily * share_f / 9).round(1)
    f["carbohydrate_g"] = (daily * share_c / 4).round(1)
    f["energy_kcal_total"] = (4 * daily * rng.normal(1, 0.015, n)).round(0)
    f["hba1c"] = (5.4 + 0.01 * (f["age"] - 45) + rng.normal(0, 0.4, n)).round(2)
    return f


FOUR_DAY_TRUTH = {"unit:energy_kcal_total": "kcal", "day_count:energy_kcal_total": "4",
                  "code_or_count:age": "amount", "code_or_count:energy_kcal_total": "amount",
                  "sex_coding:sex": "female=F,male=M",
                  # WP17, the generator: sex sets the daily intake and so the protein eaten; age
                  # moves HbA1c; weight and height move nothing; the other macronutrients share
                  # the day's intake with protein, the field's default (possible confounders).
                  "exposure:hba1c": "protein_g", "adjust:sex": "yes,no,no",
                  "adjust:age": "no,yes,no", "adjust:weight": "no,no,no",
                  "adjust:height": "no,no,no", "adjust:fat_g": "unknown,unknown,no",
                  "adjust:carbohydrate_g": "unknown,unknown,no"}


def clean_table(n: int = 400) -> pd.DataFrame:
    """p6_leash.py (rng 55006): a clean 10-column dietary table."""
    rng = np.random.default_rng(55006)
    f = pd.DataFrame({"participant_id": [f"P{i:04d}" for i in range(n)]})
    f["age"] = rng.integers(25, 75, n)
    f["sex"] = rng.choice(["F", "M"], n)
    male = (f["sex"] == "M").to_numpy()
    daily = np.where(male, rng.normal(2400, 450, n), rng.normal(1900, 380, n)).clip(900)
    p = rng.normal(0.16, 0.025, n)
    fa = rng.normal(0.34, 0.05, n)
    f["protein_g"] = (daily * p / 4).round(1)
    f["fat_g"] = (daily * fa / 9).round(1)
    f["carbohydrate_g"] = (daily * (1 - p - fa) / 4).round(1)
    f["energy_kcal"] = (4 * f["protein_g"] + 4 * f["carbohydrate_g"] + 9 * f["fat_g"]
                        + rng.normal(0, 25, n)).round(0)
    f["fiber_g"] = (daily / 1000 * rng.normal(9, 2, n)).clip(2).round(1)
    f["bmi"] = rng.normal(27, 4.5, n).round(1)
    f["sbp"] = (118 + 0.4 * (f["age"] - 50) + 0.6 * (f["bmi"] - 27) - 0.3 * f["fiber_g"]
                + rng.normal(0, 10, n)).round(0)
    return f


CLEAN_TRUTH = {"code_or_count:age": "amount", "code_or_count:energy_kcal": "amount",
               "unit:energy_kcal": "kcal", "day_count:energy_kcal": "1",
               "sex_coding:sex": "female=F,male=M",
               # WP17: the question is BMI's effect on SBP (its energy answer, "none", estimates
               # no substitution, so the exposure carries no energy). The generator: age, BMI and
               # fiber move SBP, and sex sets the day's intake and so the fiber eaten. The
               # macronutrients are declared causes of BMI whose effect on SBP is unknown, as an
               # analyst answers (the generator draws BMI apart, which no analyst could know).
               "exposure:sbp": "bmi", "adjust:age": "no,yes,no", "adjust:sex": "no,yes,no",
               "adjust:fiber_g": "no,yes,no",
               **{f"adjust:{c}": "yes,unknown,no" for c in ("protein_g", "fat_g",
                                                            "carbohydrate_g")}}


def phq_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """p8_task.py (rng 55008): a PHQ-9 total (0–27) as the outcome, once as integers and once
    written as floats because one value is blank; age and fiber drawn for each, in that order."""
    rng = np.random.default_rng(55008)
    n = 600
    phq = np.clip(rng.poisson(6, n), 0, 27)
    ints = pd.DataFrame({"phq9": phq})
    floats = pd.DataFrame({"phq9": np.where(np.arange(n) == 0, np.nan, phq)})
    out = []
    for f in (ints, floats):
        out.append(f.assign(age=rng.normal(45, 12, n).round(1),
                            fiber_g=rng.gamma(4, 5, n).round(1)))
    return out[0], out[1]


def p1_assay_dates() -> tuple[pd.Series, pd.Series]:
    """p1_value_tests.py §2 (rng 55001, its first draw): 60 subjects × 3 visits, each visit's
    sample assayed on one of four batch dates."""
    rng = np.random.default_rng(55001)
    units = np.repeat(np.arange(60), 3)
    batch = pd.to_datetime(["2024-03-04", "2024-03-11", "2024-05-20", "2024-06-03"])
    assay = pd.Series(batch[rng.integers(0, 4, len(units))])
    return assay, pd.Series(units)


def p1_draws() -> dict[str, Any]:
    """p1_value_tests.py, every draw in its order (rng 55001): the assay dates, the two skip
    patterns, the protein requirement, the food table, age bands, the DSM and ICD codes."""
    rng = np.random.default_rng(55001)
    out: dict[str, Any] = {}
    units = np.repeat(np.arange(60), 3)
    batch = pd.to_datetime(["2024-03-04", "2024-03-11", "2024-05-20", "2024-06-03"])
    out["assay"] = (pd.Series(batch[rng.integers(0, 4, len(units))]), pd.Series(units))
    flag = pd.Series(rng.choice([0, 1], 400, p=[0.8, 0.2]))
    cigs = pd.Series(np.where(flag == 1, rng.integers(2, 40, 400), np.nan))
    out["smoking"] = (flag, cigs)
    drk = pd.Series(rng.choice([0, 1], 400, p=[0.35, 0.65]))
    alc = pd.Series(np.where(drk == 1, rng.gamma(2, 7, 400).round(1), np.nan))
    out["alcohol"] = (drk, alc)
    w = rng.normal(78, 16, 500).clip(45, 160)
    rng.normal(0, 380, 500)
    m = 600
    rng.gamma(1.5, 5, m), rng.gamma(0.8, 8, m), rng.gamma(1.2, 15, m), rng.normal(1, 0.03, m)
    rng.integers(1, 7, 500)
    out["dsm"] = pd.Series(rng.choice([307.1, 307.51, 307.5, 307.59], 300, p=[0.3, 0.3, 0.3, 0.1]))
    out["icd"] = pd.Series(rng.choice([250.0, 250.02, 401.9, 272.4, 414.01, 428.0], 300))
    del w
    return out


ROW_ORDER = ORDER


def _feature_level(name: str, column: str) -> float | None:
    """The level a one-hot feature stands for (``dsm_dx_307.51`` → 307.51)."""
    prefix = f"{column}_"
    if not name.startswith(prefix):
        return None
    try:
        return float(name[len(prefix):])
    except ValueError:
        return None


# ═════════════════════════════════════════════════════════════════════════════
# A · the fifth gate's not-closed items, replayed draw for draw
# ═════════════════════════════════════════════════════════════════════════════


def test_a1_decimal_diagnosis_codes_are_asked_and_fit_as_one_indicator_per_code(tmp_path):
    """Gate item 1 (p2_dsm_codes.py, rng 55002, through the server). Sources (quoted): "ICD-9-CM
    Diagnosis Code 307.1 : Anorexia nervosa", "… 307.51 : Bulimia nervosa", "… 250.02 : Diabetes
    mellitus without mention of complication, type II or unspecified type, uncontrolled". Before:
    any fractional value settled "amount", the fit never asked, and `dsm_dx` entered as one slope,
    10.8513 per code unit (the NumPy single slope). Expected: the fit asks the code-or-amount
    reading of `dsm_dx` (guessing codes), computes nothing until it is answered, and with the
    fixture's truth (codes) fits one indicator per code: the NumPy indicator fit, 307.1 the
    reference (scikit-learn's first sorted level), with age and total energy beside them."""
    frame = dsm_table()
    y = frame["bmi"].to_numpy(float)
    one = np.ones(len(frame))
    single = ols(y, np.column_stack([one, frame.dsm_dx, frame.age, frame.energy_kcal]).astype(float))
    assert round(single[1], 4) == 10.8513  # the gate's number: the slope the old fit entered
    levels = (307.5, 307.51, 307.59)
    indicator = ols(y, np.column_stack([one, *[(frame.dsm_dx == c) for c in levels], frame.age,
                                        frame.energy_kcal]).astype(float))
    # WP17, the generator: the diagnosis, age and energy are drawn apart, and each moves BMI.
    truth = Truth({"code_or_count:dsm_dx": "code", "code_or_count:energy_kcal": "amount",
                   "exposure:bmi": "dsm_dx", "adjust:age": "no,yes,no",
                   "adjust:energy_kcal": "no,yes,no"}, fixture="p2 (ICD-9-CM codes)")
    plan = answers("bmi", ["clinical"])
    with local_server(tmp_path / "home") as client:
        drive = open_project(client, write(frame, tmp_path, "dsm_codes.csv"), truth)
        drive_unsettled(drive, plan, roles={})
        error = fit_refused_without_a_number(drive, plan["models"])
        listed = asked(error["exits"])
        assert ("code_or_count", "dsm_dx") in listed, listed
        block = next(e["decision"] for e in error["exits"]
                     if (e.get("decision") or {}).get("kind") == "confirm_readings")
        assert ("dsm_dx", "code") in {(i["column"], i["value"]) for i in block["items"]}
        assert "codes written with a decimal point" in error["message"], error["message"]
        drive.decide(plan["models"])
        coef = coefficients(drive.artifact("fit", timeout=600))
    got = {_feature_level(k, "dsm_dx"): v["estimate"] for k, v in coef.items()
           if k.startswith("dsm_dx")}
    assert set(got) == set(levels), got  # one indicator per code, 307.1 the reference
    for i, level in enumerate(levels, start=1):
        assert got[level] == pytest.approx(indicator[i], abs=1e-6), level
    assert coef["age"]["estimate"] == pytest.approx(indicator[4], abs=1e-6)
    assert coef["energy_kcal"]["estimate"] == pytest.approx(indicator[5], abs=1e-9)


def test_a1_the_decimal_code_value_test_rejects_the_gates_codes_and_prints_them_whole():
    """The registry's value test on the gate's own draws (p1 §8, rng 55001): the four eating-
    disorder codes and the six ICD-9-CM codes settle nothing; the evidence prints a code's decimals
    (the gate's minor: 250.02 printed as `250`)."""
    draws = p1_draws()
    for series in (draws["dsm"], draws["icd"]):
        verdict = R.amounts_by_values(series)
        assert not verdict.settles, verdict
    icd = R.amounts_by_values(pd.Series([250.02, 401.9, 272.4] * 20))
    assert "250.02" in icd.evidence and "`250`" not in icd.evidence, icd.evidence


def test_a2_a_skip_pattern_gate_is_asked_and_the_users_answer_keeps_it_in_the_fit(tmp_path):
    """Gate item 2 (p3_skip_flag.py, rng 55003, prediction, impute; through the server). Source
    (NHANES ALQ_J, quoted): ALQ111 "Ever had a drink of any kind of alcohol", code 2 "No" → "End
    of Section", so ALQ130 "Avg # alcohol drinks/day - past 12 mos" is blank exactly where the
    gate says no. Before: `alcohol_flag` was "flag (high)" and silently left the fit (features
    age, alcohol, fiber_g). Expected: the role is proposed below high, the fit asks it with the
    skip-gate alternative offered, and with the fixture's truth (a characteristic) the drinker
    indicator is in the fit, its coefficient the NumPy fit on the training rows with `alcohol`'s
    blanks filled by its training median."""
    frame = skip_table()
    plan = answers("sbp", ["clinical"],
                   purpose={"kind": "set_purpose", "purpose": "prediction"},
                   missing={"kind": "set_missing", "strategy": "impute"})
    with local_server(tmp_path / "home") as client:
        drive = open_project(client, write(frame, tmp_path, "skip_flag.csv"),
                             Truth(SKIP_TRUTH, fixture="p3 (skip pattern)"))
        proposals = drive_unsettled(drive, plan, roles={})
        flag = next(p for p in proposals if p["column"] == "alcohol_flag")
        assert (flag["proposed"], flag["confidence"]) == ("flag", "medium"), flag
        error = fit_refused_without_a_number(drive, plan["models"])
        assert ("role", "alcohol_flag") in asked(error["exits"])
        offered = {(e["decision"]["column"], e["decision"]["value"]) for e in error["exits"]
                   if (e.get("decision") or {}).get("kind") == "confirm_reading"}
        assert {("alcohol_flag", "flag"), ("alcohol_flag", "covariate")} <= offered, offered
        drive.decide(plan["models"])
        assert drive.view()["state"]["roles"]["alcohol_flag"] == "covariate"
        coef = coefficients(drive.artifact("fit", timeout=600))
        sealed = drive.sealed()
    assert set(coef) == {"(intercept)", "age", "alcohol_flag", "alcohol", "fiber_g"}, set(coef)
    train = frame.loc[~frame.index.isin(sealed)]
    alcohol = train["alcohol"].fillna(train["alcohol"].median())
    X = np.column_stack([np.ones(len(train)), train.age, train.alcohol_flag, alcohol,
                         train.fiber_g]).astype(float)
    beta = ols(train["sbp"].to_numpy(float), X)
    for i, name in enumerate(["(intercept)", "age", "alcohol_flag", "alcohol", "fiber_g"]):
        assert coef[name]["estimate"] == pytest.approx(beta[i], abs=1e-6), name


def test_a2_the_users_role_confirmation_reaches_the_fit_through_the_server(tmp_path):
    """Gate census item (p3b_flag_confirm.py: `confirm_reading` role alcohol_flag=covariate
    returned 200 and the state kept roles[alcohol_flag] = flag). Expected: the confirmation is the
    column's role from then on (the state's roles say covariate) and the fit's features hold it."""
    plan = answers("sbp", ["clinical"],
                   purpose={"kind": "set_purpose", "purpose": "prediction"},
                   missing={"kind": "set_missing", "strategy": "impute"})
    with local_server(tmp_path / "home") as client:
        drive = open_project(client, write(skip_table(), tmp_path, "flag_confirm.csv"),
                             Truth(SKIP_TRUTH, fixture="p3b"))
        drive_unsettled(drive, plan, roles={}, stop_before="survey")
        r = drive.post({"kind": "confirm_reading", "reading": "role", "column": "alcohol_flag",
                        "value": "covariate"})
        assert r.status_code == 200, r.text[:400]
        state = drive.view()["state"]
        assert state["roles"]["alcohol_flag"] == "covariate"
        assert state["role_confirmations"] == {"alcohol_flag": "covariate"}
        for key in ORDER[ORDER.index("survey"):]:
            if drive.reach(key, timeout=300)["status"] in ("open", "waiting"):
                drive.decide(plan[key])
        coef = coefficients(drive.artifact("fit", timeout=600))
    assert "alcohol_flag" in coef, set(coef)


def test_a3_a_role_confirmed_as_another_role_sets_that_role():
    """Gate census item (p9_role_confirm.py): `confirm_reading(s)` for a role with a value other
    than the recorded one was stored and never read. Expected (from the confirmed values): the
    role is the confirmed one, settled, and the fit's predictors follow it."""
    from turbotab.core.models.pipeline import model_predictors

    def rec(i: int, dec: Any) -> d.DecisionRecord:
        return d.DecisionRecord(id=f"r{i}", seq=i, at="2026-10-03T00:00:00Z", decision=dec)

    roles = {"pid": "identifier", "x": "exposure", "flagish": "flag", "age": "covariate"}
    base = [rec(1, d.SetTarget(column="y")), rec(2, d.SetRoles(roles=roles,
                                                               unconfirmed=["x", "flagish"]))]
    for x_value, f_value in (("exposure", "flag"), ("covariate", "covariate"),
                             ("excluded", "covariate")):
        st = d.fold(base + [rec(3, d.ConfirmReadings(items=[
            {"reading": "role", "column": "x", "value": x_value},
            {"reading": "role", "column": "flagish", "value": f_value}]))])
        assert (st.roles["x"], st.roles["flagish"]) == (x_value, f_value)
        assert R.unsettled(st) == []
        want = [c for c, r in {"x": x_value, "flagish": f_value, "age": "covariate"}.items()
                if r in ("exposure", "covariate", "energy")]
        assert sorted(model_predictors(st)) == sorted(want)
        undone = d.fold(base + [rec(3, d.ConfirmReadings(items=[
            {"reading": "role", "column": "x", "value": x_value}])), rec(4, d.Revert(
                decision_id="r3"))])
        assert undone.roles == roles  # one revert restores the recorded roles


def _screens(drive: Any) -> dict[str, dict[str, Any]]:
    return {o["key"]: o for o in drive.artifact("proposals").get("exclusions") or []}


def test_a4_a_four_day_kcal_total_is_asked_with_kcal_offered_and_read_as_recorded(tmp_path):
    """Gate item 4 (p4_four_day_total.py, rng 55004, through the server). Source: NIST SP 811
    (quoted): the thermochemical kilocalorie is exactly 4.184 E+03 J, so a 4-day kcal total beside
    daily-mean macronutrients reconstructs at 4.00, within 5% of kJ's 4.184. Before: the unit was
    settled kJ, the refusal offered only kJ exits, and the 500–3,500 kcal screen counted 5 rows
    against 7 under the truth. Expected: the unit is not settled (both readings named), an exit
    offers kcal over 4 days, and once it is recorded the screen counts the rows pandas counts
    outside 2,000–14,000 kcal, and no Goldberg screen (a day's mean) is offered on a 4-day total."""
    frame = four_day_table()
    ratio = float((frame["energy_kcal_total"] / (4 * frame["protein_g"]
                                                 + 4 * frame["carbohydrate_g"]
                                                 + 9 * frame["fat_g"])).median())
    assert abs(ratio - 4.0) < 0.01 and abs(ratio - KJ_PER_KCAL) / KJ_PER_KCAL < 0.05
    e = frame["energy_kcal_total"]
    want = int(((e < 4 * 500) | (e > 4 * 3500)).sum())
    assert want == 7  # the gate's count under the truth
    plan = answers("hba1c", ["dietary"])
    with local_server(tmp_path / "home") as client:
        drive = open_project(client, write(frame, tmp_path, "four_day_total.csv"),
                             Truth(FOUR_DAY_TRUTH, fixture="p4"))
        drive_unsettled(drive, plan, roles={}, stop_before="exclusions")
        unit = drive.artifact("proposals")["energy_unit"]
        assert not unit["unit_settled"] and not unit["confirmed"], unit
        assert ["kcal", 4] in unit["candidates"] and ["kj", 1] in unit["candidates"], unit
        screen = _screens(drive)["sex_neutral_500_3500"]
        assert screen["refused"], screen
        r = drive.post({"kind": "set_exclusions", "rules": [screen["rule"]]})
        assert r.status_code == 409 and r.json()["error"]["code"] in ASKING, r.text[:400]
        exits = [x["decision"] for x in r.json()["error"]["exits"] if x.get("decision")]
        truth_exit = {"kind": "set_column_unit", "column": "energy_kcal_total", "unit": "kcal",
                      "days": 4}
        assert truth_exit in exits, exits
        assert drive.post(truth_exit).status_code == 200
        screens = _screens(drive)
        screen = screens["sex_neutral_500_3500"]
        assert screen["affected"] == want and not screen["refused"], screen
        assert "goldberg_schofield" not in screens
        assert _post_when_reached(client, f"/api/projects/{drive.pid}/decisions",
                                  {"kind": "set_exclusions",
                                   "rules": [screen["rule"]]}).status_code == 200


def test_a5_the_screens_read_a_unit_and_days_confirmed_on_their_own(tmp_path):
    """Gate census item (p4 confirm_path): `confirm_reading` unit=kcal and day_count=4 returned 200,
    the ledger and the coach read them, and the screens kept kJ over one day. Expected: one store,
    one accessor: the proposals read kcal over 4 days, settled, and the screen counts the rows
    pandas counts outside 2,000–14,000 kcal and is chosen without a question."""
    frame = four_day_table()
    e = frame["energy_kcal_total"]
    want = int(((e < 2000) | (e > 14000)).sum())
    plan = answers("hba1c", ["dietary"])
    with local_server(tmp_path / "home") as client:
        drive = open_project(client, write(frame, tmp_path, "four_day_conf.csv"),
                             Truth(FOUR_DAY_TRUTH, fixture="p4 confirm"))
        drive_unsettled(drive, plan, roles={}, stop_before="exclusions")
        for body in ({"kind": "confirm_reading", "reading": "unit", "column": "energy_kcal_total",
                      "value": "kcal"},
                     {"kind": "confirm_reading", "reading": "day_count",
                      "column": "energy_kcal_total", "value": "4"}):
            assert drive.post(body).status_code == 200
        state = ProjectState.model_validate(drive.view()["state"])
        assert R.recorded_energy(state, "energy_kcal_total") == ("kcal", 4)
        unit = drive.artifact("proposals")["energy_unit"]
        assert (unit["unit"], unit["days"], unit["confirmed"]) == ("kcal", 4, True), unit
        screen = _screens(drive)["sex_neutral_500_3500"]
        assert screen["affected"] == want and not screen["refused"], screen
        r = _post_when_reached(client, f"/api/projects/{drive.pid}/decisions",
                               {"kind": "set_exclusions", "rules": [screen["rule"]]})
        assert r.status_code == 200, r.text[:400]


def _rec(i: int, dec: Any) -> d.DecisionRecord:
    return d.DecisionRecord(id=f"r{i}", seq=i, at="2026-10-03T00:00:00Z", decision=dec)


def _folded(*decisions: Any) -> ProjectState:
    """The state the decision log folds to: every probe confirms through the fold (the write path
    the server uses), never by building the slots by hand."""
    return d.fold([_rec(i, dec) for i, dec in enumerate(decisions, start=1)])


def test_a6_the_kcal_per_unit_derives_from_the_confirmed_unit_and_never_from_a_name():
    """Gate census item (p5_factor_confirm.py): for `Protein` (an InBody export's body protein)
    each confirmed unit gave 4 kcal per unit, and `_g`/`_kcal` names set the factor unconfirmed.
    Expected (Atwater 4 kcal/g for protein, NUTRITION_PACK §01; 1 kg = 1,000 g; NIST: 1 kcal =
    4.184 kJ exactly): g → 4, kg → 4,000, kcal → 1, kJ → 1/4.184, a share of energy refused with
    its route, any other unit refused; unconfirmed, no name settles it, and the substitution's
    refusal offers every unit it may be in."""
    want = {"g": ATWATER["protein"], "kg": 1000 * ATWATER["protein"], "kcal": 1.0,
            "kj": 1 / KJ_PER_KCAL}
    for unit in R.UNIT_VALUES:
        st = _folded(d.SetTarget(column="y"),
                     d.SetRoles(roles={"Protein": "exposure", "fat_g": "exposure"}),
                     d.ConfirmReading(reading="unit", column="Protein", value=unit))
        found = R.kcal_per_unit(st, "Protein")
        if unit in want:
            assert found.settled and found.factor == pytest.approx(want[unit], rel=1e-12), unit
        else:
            assert not found.settled and found.factor is None, unit
            assert (found.route is not None) == (unit == "pct_energy"), unit
    bare = _folded(d.SetTarget(column="y"),
                   d.SetRoles(roles={"Protein": "exposure", "fat_g": "exposure",
                                     "protein_kcal": "exposure"}))
    for named in ("Protein", "fat_g", "protein_kcal"):
        assert not R.kcal_per_unit(bare, named).settled, named
    ctx = {"state": bare, "columns": ["Protein", "fat_g", "protein_kcal", "y"], "target": "y"}
    with pytest.raises(d.Refusal) as refused:
        d.validate({"kind": "set_substitution", "donor": "Protein", "recipient": "fat_g"}, ctx)
    offered = {(e["decision"]["column"], e["decision"]["value"]) for e in refused.value.exits
               if e.get("decision")}
    assert {("Protein", u) for u in want} | {("fat_g", u) for u in want} <= offered, offered


def test_a7_a_phq9_total_is_asked_its_task_whatever_its_dtype(tmp_path):
    """Gate item (p8_task.py, rng 55008, through the server). Source (Kroenke 2001, quoted): "the
    PHQ-9 score can range from 0 to 27, since each of the 9 items can be scored from 0 (not at all)
    to 3 (nearly every day)": an ordinal score, and the app offers an ordinal task. Before: as
    integers the task was asked (medium); written as floats after one blank it was settled
    "regression (high)" and skipped. Expected: asked both ways (the registry's value test rejects
    whole numbers of any type), with the ordinal alternative named; a measurement with decimals
    filling its grid (the control) is settled and skipped."""
    ints, floats = phq_tables()
    assert ints["phq9"].between(*PHQ9_RANGE).all()
    control = ints.assign(phq9=np.random.default_rng(7).normal(5.4, 0.6, len(ints)).round(2))
    with local_server(tmp_path / "home") as client:
        seen = {}
        for name, frame in (("int", ints), ("float", floats), ("control", control)):
            drive = open_project(client, write(frame, tmp_path, f"task_{name}.csv"),
                                 Truth(fixture=f"p8 {name}"))
            drive.decide({"kind": "set_lens", "lenses": ["clinical"]})
            drive.decide({"kind": "set_target", "column": "phq9"})
            info = drive.artifact("target_info")
            step = next(s for s in drive.view()["interview"] if s["key"] == "task")
            seen[name] = (step["status"], info["confidence"], info["task"])
    assert seen["int"][0] == "open" and seen["int"][1] != "high", seen
    assert seen["float"][0] == "open" and seen["float"][1] != "high", seen
    assert seen["control"] == ("skipped", "high", "regression"), seen
    verdict = R.outcome_task_by_values(floats["phq9"])
    assert not verdict.settles and "ordinal" in verdict.candidates


def test_a8_an_assay_run_date_that_changes_within_units_is_asked_never_settled_time(tmp_path):
    """Gate item (p1 §2, rng 55001: 60 subjects × 3 visits, each sample assayed on one of four
    batch dates). Before: `dates_order_rows` settled "time (high)" and the run date left the
    predictors unasked. Expected: the registry declares the run date as an alternative and no value
    test for the time role (the user settles it); the roles stage proposes it below high, so a bulk
    answer carries it unconfirmed and the fit asks it, the batch reading offered."""
    assay, units = p1_assay_dates()
    assert R.dates_order_rows(assay, units).settles  # the values cannot tell a run date apart
    rule = R.KIND_RULES["role:time"]
    assert rule.test is None and any("run" in a for a in rule.alternatives)
    rng = np.random.default_rng(9)
    frame = pd.DataFrame({"subject_id": [f"S{u:03d}" for u in units], "RunDate": assay,
                          "age": np.repeat(rng.normal(50, 10, 60).round(1), 3),
                          "crp": rng.normal(3, 1, len(units)).round(2)})
    from turbotab.core.decisions import GrainSpec

    proposals = roles_of(frame, lens=["clinical"], target="crp", folder=tmp_path,
                         grain=GrainSpec(grain="repeated", id_column="subject_id"))
    run = proposals["RunDate"]
    assert run["proposed"] == "time" and run["confidence"] == "medium" and run["attention"], run
    st = _folded(d.SetTarget(column="crp"),
                 d.SetRoles(roles={"subject_id": "identifier", "RunDate": "time",
                                   "age": "covariate"}, unconfirmed=["RunDate"]))
    with pytest.raises(R.Unsettled) as waiting:
        R.predictors_or_ask(st)
    offered = {(e["decision"]["column"], e["decision"]["value"]) for e in waiting.value.exits
               if (e.get("decision") or {}).get("kind") == "confirm_reading"}
    assert {("RunDate", "time"), ("RunDate", "covariate")} <= offered, offered


def _journey(drive: Any, plan: dict[str, Any], count: dict[str, Any], *,
             roles: Callable[[Any], dict[str, Any]] | None = None) -> None:
    """The opening sequence in the Router's order, as the gate's journey ran it (g5.journey):
    the roles answered as proposed (the truth for their role readings), every ask answered from
    the fixture's truth, each question and each ask counted with the readings it listed."""
    url = f"/api/projects/{drive.pid}/decisions"
    count.setdefault("questions", [])
    count.setdefault("asks", [])
    for key in ORDER:
        if key in ("event", "task"):
            # The outcome's questions read the target's stage: wait for it, as a client shows the
            # question once its options are read (the Router can show it open a moment before the
            # stage is queued).
            drive.artifact("target_info", timeout=600)
        step = drive.reach(key, timeout=600)
        if step["status"] not in ("open", "waiting"):
            continue
        count["questions"].append(key)
        body = plan.get(key)
        if key == "roles":
            proposals = drive.artifact("roles")["columns"]
            body = {"kind": "set_roles", "roles": {c["column"]: c["proposed"] for c in proposals}}
            for column, role in body["roles"].items():
                drive.truth.setdefault(f"role:{column}", role)
        r = _post_when_reached(drive.c, url, body)
        rounds = 0
        while r.status_code == 409 and r.json()["error"]["code"] in ASKING and rounds < 5:
            rounds += 1
            error = r.json()["error"]
            count["asks"].append({"at": key, "readings": asked(error["exits"])})
            for decision in truth_answers(error, drive.truth):
                assert drive.c.post(url, json=decision).status_code == 200, decision
            r = _post_when_reached(drive.c, url, body)
        assert r.status_code == 200, (key, r.text[:800])


def test_a9_the_leash_on_a_clean_dietary_table_asks_nothing_the_plan_does_not_use(tmp_path):
    """Gate leash item (p6_leash.py rng 55006, p7_two_codes.py, p7b_energy_code.py; through the
    server, inference). Before: `energy_kcal` was asked code-or-amount though the energy answer
    "none" takes total energy out of the model, its offered "codes" exit crashed the design stage
    ("ValueError: Some column names are not columns of the dataframe: {'energy_kcal'}"), and a 1/2
    `sex` with no blanks was asked though both answers give the same slope (0.344842) and
    interval. Expected: neither is asked; the fit runs; and with `sex` 1/2 the fit's sex slope is the
    NumPy slope, whichever way the column is coded."""
    plan = answers("sbp", ["dietary"])
    counts = {}
    base = clean_table()
    two = base.assign(sex=base["sex"].map({"M": 1, "F": 2}))
    with local_server(tmp_path / "home") as client:
        for name, frame, truth in (
                ("clean10", base, CLEAN_TRUTH),
                ("two_codes", two, {**CLEAN_TRUTH, "sex_coding:sex": "female=2,male=1"})):
            drive = open_project(client, write(frame, tmp_path, f"{name}.csv"),
                                 Truth(truth, fixture=name))
            count: dict[str, Any] = {}
            _journey(drive, plan, count)
            fit = drive.artifact("fit", timeout=600)
            assert drive.view()["stages"]["design"]["status"] == "fresh"
            listed = [tuple(x) for a in count["asks"] for x in a["readings"]]
            assert ("code_or_count", "energy_kcal") not in listed, listed
            assert ("code_or_count", "sex") not in listed, listed
            counts[name] = {"questions": len(count["questions"]), "asks": len(count["asks"]),
                            "readings": listed}
            if name == "two_codes":
                coef = coefficients(fit)
                X = two[["sex", "age", "protein_g", "fat_g", "carbohydrate_g", "fiber_g", "bmi"]]
                beta = ols(two["sbp"].to_numpy(float),
                           np.column_stack([np.ones(len(two)), X.to_numpy(float)]))
                sex_term = next(k for k in coef if k.startswith("sex"))
                assert coef[sex_term]["estimate"] == pytest.approx(beta[1], abs=1e-6)
                assert round(beta[1], 6) == 0.344842  # the gate's slope, either answer
    (tmp_path / "clean_questions.json").write_text(json.dumps(counts))
    print(f"clean dietary (inference): {counts}")


# ═════════════════════════════════════════════════════════════════════════════
# D · every confirmation is honored: CONFIRMABLE × every registered consumer × every value
# ═════════════════════════════════════════════════════════════════════════════
#
# The pairs come from the registry (``readings.CONSUMERS``: each consumer names the registry kinds
# it reads), the values from what the validator accepts for the kind (every role, every unit, every
# day count from 1 to 366, both answers of a yes/no reading), and every probe confirms through the
# decision fold (``confirm_reading``), the write path the server uses, never by setting slots by
# hand: the fifth gate's three gaps (a role confirmed as another role, a unit and days confirmed on
# their own, a substitution's unit) each passed a probe that wrote the slot the consumer read.


def _base(kind: str) -> str:
    return kind.split(":", 1)[0]


def domain(kind: str) -> tuple[str, ...]:
    """Every value the validator accepts for a confirmation of this kind. Updated by
    LEDGER-REPAIR-3 (the sixth gate): a unit includes alcohol's standard drinks, every size from 8
    to 20 g to the hundredth (``drinks:<grams>``); a column-valued kind (``readings.COLUMN_VALUED``:
    ``nested_in``) every column of its probe's table the validator accepts, and no total."""
    base = _base(kind)
    if base == "day_count":
        return tuple(str(i) for i in range(1, 367))
    if base == "unit":
        return tuple(R.UNIT_VALUES) + R.drink_units()
    if base == "sex_coding":
        return ("female=1,male=2", "female=2,male=1")
    if base == "nested_in":
        return nested_domain()
    return tuple(R.VALUES[base])


def test_d0_the_domain_is_what_the_validator_accepts():
    for kind in ("role", "cluster", "unit:energy", "day_count", "code_or_count", "time_column",
                 "sex_coding"):
        for value in domain(kind):
            d.validate({"kind": "confirm_reading", "reading": _base(kind), "column": "x",
                        "value": value}, {"columns": ["x", "y"], "target": "y"})
    for bad in ("0", "367", "two"):
        with pytest.raises(d.Refusal):
            d.validate({"kind": "confirm_reading", "reading": "day_count", "column": "x",
                        "value": bad}, {"columns": ["x", "y"], "target": "y"})
    # Standard drinks hold 8–20 g of alcohol (Kalinowski & Humphreys 2016): 1,201 sizes to the
    # hundredth of a gram, written without trailing zeros; nothing outside them.
    assert len(R.drink_units()) == 1201 and "drinks:14" in R.drink_units() \
        and "drinks:13.45" in R.drink_units()
    for bad in ("drinks:7.99", "drinks:20.01", "drinks:14.0", "drinks:", "drinks:abc"):
        with pytest.raises(d.Refusal):
            d.validate({"kind": "confirm_reading", "reading": "unit", "column": "x",
                        "value": bad}, {"columns": ["x", "y"], "target": "y"})
    # A column-valued kind: every other column of the table but the outcome, and no total, exactly.
    assert set(nested_domain()) == ({c for c in NESTED_COLUMNS if c not in ("sfa_g", "y")}
                                    | {R.NOT_NESTED})


def _confirm(*before: Any, kind: str, column: str, value: str) -> ProjectState:
    return _folded(*before, d.ConfirmReading(reading=_base(kind), column=column, value=value))


PREDICTORS = set(R.PREDICTOR_ROLES)


# ── role ──

def _role_state(value: str, **extra: Any) -> ProjectState:
    return _confirm(d.SetTarget(column="y"),
                    d.SetRoles(roles={"x": "exposure", "age": "covariate", **extra},
                               unconfirmed=["x"]),
                    kind="role", column="x", value=value)


def _probe_model_predictors(value: str) -> Any:
    from turbotab.core.models.pipeline import model_predictors

    return "x" in model_predictors(_role_state(value))


def _probe_models_validator_role(value: str) -> Any:
    ctx = {"state": _role_state(value), "column_info": {"x": {"dtype": "integer", "n_unique": 3}},
           "columns": ["x", "age", "y"], "target": "y", "task": "regression"}
    try:
        d.validate({"kind": "select_models", "models": ["linear"]}, ctx)
    except d.Refusal as refused:
        return sorted(asked(refused.exits))
    return []


def _probe_cohort_inputs(value: str) -> Any:
    from turbotab.core.stages.rows import cohort_inputs

    state = _folded(d.SetTarget(column="y"),
                    d.SetRoles(roles={"x": "exposure", "age": "covariate"}, unconfirmed=["x"]),
                    d.SetMissing(strategy="complete_case"),
                    d.ConfirmReading(reading="role", column="x", value=value))
    ingest = {"columns": [{"name": "x", "n_missing": 5}, {"name": "age", "n_missing": 0},
                          {"name": "y", "n_missing": 0}]}
    _, preds, gappy = cohort_inputs(state, ingest)
    return ("x" in preds, "x" in gappy)


def _energy_card_frame() -> pd.DataFrame:
    rng = np.random.default_rng(31)
    n = 300
    P, C, F = rng.normal(80, 20, n).clip(20), rng.normal(250, 60, n).clip(50), \
        rng.normal(75, 20, n).clip(15)
    return pd.DataFrame({"energy_kcal": (4 * P + 4 * C + 9 * F).round(0),
                         "protein_g": P.round(1), "carbohydrate_g": C.round(1), "fat_g": F.round(1),
                         "x": rng.normal(15, 4, n).round(1),
                         "y": rng.normal(120, 10, n).round(1)})


def _columns(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [{"name": c, "dtype": "numeric", "n_unique": int(frame[c].nunique()),
             "n_missing": int(frame[c].isna().sum())} for c in frame.columns]


def _probe_proposals_role(value: str) -> Any:
    """The energy card: confirming `protein_g` (recorded an exposure) as each role."""
    from turbotab.core.stages.proposals import build_proposals

    frame = _energy_card_frame()
    state = _confirm(d.SetLens(lenses=["dietary"]), d.SetTarget(column="y"),
                     d.SetRoles(roles={"energy_kcal": "energy", "protein_g": "exposure",
                                       "carbohydrate_g": "exposure", "fat_g": "exposure",
                                       "x": "covariate"}, unconfirmed=["protein_g"]),
                     kind="role", column="protein_g", value=value)
    out = build_proposals(frame, _columns(frame), lens=["dietary"], target="y", roles=state.roles,
                          settled=R.settled_columns(state), state=state)
    return "protein_g" in ((out.get("energy") or {}).get("nutrients") or [])


def _clusters_frame() -> pd.DataFrame:
    """Households of 3 (40 of them) inside 12 sites of 10 rows: two groupings, each repeating."""
    return pd.DataFrame({"hhid": np.repeat(np.arange(40), 3),
                         "site": np.repeat(np.arange(12), 10)}, index=np.arange(120))


def _probe_resolve_clusters_role(value: str) -> Any:
    from turbotab.core.models.inference import resolve_clusters

    state = _confirm(d.SetTarget(column="y"), d.SetGrain(grain="one_row_per_unit"),
                     d.SetRoles(roles={"hhid": "identifier", "site": "covariate"},
                                unconfirmed=["hhid"]),
                     kind="role", column="hhid", value=value)
    found = resolve_clusters(state, _clusters_frame())
    return found.column, found.n_clusters


# ── cluster ──

def _cluster_state(value: str, grain: str = "one_row_per_unit") -> ProjectState:
    return _confirm(d.SetTarget(column="y"), d.SetGrain(grain=grain),
                    d.SetRoles(roles={"hhid": "identifier", "site": "covariate"},
                               unconfirmed=["hhid"]),
                    kind="cluster", column="hhid", value=value)


def _probe_resolve_clusters(value: str) -> Any:
    from turbotab.core.models.inference import resolve_clusters

    found = resolve_clusters(_cluster_state(value), _clusters_frame())
    return found.column, found.n_clusters


class _Store:
    columns = ["hhid", "site", "y"]

    def materialize(self, columns: Any, ids: Any = None) -> pd.DataFrame:
        frame = _clusters_frame().assign(y=np.arange(120.0))
        return frame.loc[ids if ids is not None else frame.index, list(columns)]


def _probe_split_inputs(value: str) -> Any:
    from turbotab.core.stages.rows import split_inputs

    return split_inputs(_cluster_state(value), np.arange(120), _Store(), "regression")["grouped_by"]


def _probe_seal_inputs(value: str) -> Any:
    from turbotab.core.seal import seal_inputs

    return seal_inputs(_cluster_state(value, "repeated"), np.arange(120), _Store(), "regression",
                       holdout=0.2, seed=0).grouped_by


# ── code or amount ──

def _code_state(value: str, **extra: Any) -> ProjectState:
    return _confirm(d.SetTarget(column="y"), d.SetRoles(roles={"smoking": "covariate"}),
                    *extra.values(), kind="code_or_count", column="smoking", value=value)


def _probe_design_spec(value: str) -> Any:
    from turbotab.core.models.pipeline import design_spec

    X = pd.DataFrame({"smoking": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]})
    return "smoking" in design_spec(_code_state(value), X, ["smoking"]).categorical


def _probe_imputation(value: str) -> Any:
    from turbotab.core.methods.missing import imputation_frame
    from turbotab.core.models.pipeline import design_spec

    X = pd.DataFrame({"smoking": [1.0, 2.0, 3.0, np.nan, 2.0, 3.0, 1.0, 2.0]})
    state = _code_state(value, missing=d.SetMissing(strategy="multiple_imputation"))
    spec = design_spec(state, X, ["smoking"])
    return imputation_frame(spec, X, pd.Series(np.arange(8.0)), "regression")[2]["smoking"]


def _probe_models_validator_codes(value: str) -> Any:
    ctx = {"state": _code_state(value),
           "column_info": {"smoking": {"dtype": "integer", "n_unique": 3}},
           "columns": ["smoking", "y"], "target": "y", "task": "regression"}
    try:
        d.validate({"kind": "select_models", "models": ["linear"]}, ctx)
    except d.Refusal as refused:
        return refused.code
    return None


# ── the time column ──

def _probe_recall_order(value: str) -> Any:
    """The NCI usual-intake method orders a long table's recalls by the confirmed time column."""
    from turbotab.core.stages.usual_intake import recall_order

    state = _confirm(d.SetTarget(column="y"), kind="time_column", column="visit", value=value)
    return recall_order(state, {"repeats": {"replicate_index": "visit"}},
                        d.UsualIntakeSpec(model="amount_only"))


def _probe_time_column(value: str) -> Any:
    from turbotab.core.stages.working import time_column

    state = _confirm(d.SetTarget(column="y"), kind="time_column", column="visit", value=value)
    return time_column(state, {"repeats": {"replicate_index": "visit"}})


# ── total energy's unit and days ──

def _energy_frame() -> pd.DataFrame:
    """Total energy (no macronutrients, so no value settles its unit), sex, body measures."""
    rng = np.random.default_rng(32)
    n = 400
    sex = rng.choice(["F", "M"], n)
    male = sex == "M"
    return pd.DataFrame({
        "energy": np.where(male, rng.normal(2500, 700, n), rng.normal(1900, 600, n)).round(0),
        "sex": sex, "age": rng.integers(25, 70, n),
        "weight": np.where(male, rng.normal(84, 13, n), rng.normal(70, 13, n)).round(1),
        "height": np.where(male, rng.normal(176, 7, n), rng.normal(163, 6, n)).round(1),
        "y": rng.normal(120, 10, n).round(1)})


ENERGY_ROLES = {"energy": "energy", "sex": "covariate", "age": "covariate",
                "weight": "covariate", "height": "covariate"}


def _energy_state(kind: str, value: str, *, fixed: tuple[str, str] | None = None) -> ProjectState:
    """The roles as the user's own, the other half of the unit-and-days answer recorded (``fixed``:
    the unit when days are probed, one day when the unit is), then the probed confirmation."""
    before: list[Any] = [d.SetLens(lenses=["dietary"]), d.SetTarget(column="y"),
                         d.SetRoles(roles=ENERGY_ROLES)]
    if fixed is not None:
        before.append(d.ConfirmReading(reading=fixed[0], column="energy", value=fixed[1]))
    return _confirm(*before, kind=kind, column="energy", value=value)


def _offers(state: ProjectState, frame: pd.DataFrame | None = None) -> dict[str, Any]:
    from turbotab.core.stages.proposals import build_proposals

    frame = _energy_frame() if frame is None else frame
    columns = [{"name": c, "dtype": "text" if c == "sex" else "numeric",
                "n_unique": int(frame[c].nunique()), "n_missing": 0} for c in frame.columns]
    out = build_proposals(frame, columns, lens=["dietary"], target="y", roles=state.roles,
                          units=state.column_units or {}, settled=R.settled_columns(state),
                          state=state)
    return {o["key"]: o for o in out["exclusions"]} | {"__unit__": out["energy_unit"]}


def _screen(offers: dict[str, Any]) -> Any:
    o = offers["sex_neutral_500_3500"]
    return (o["rule"]["low"], o["rule"]["high"], o["affected"], bool(o["refused"]))


# The screens' own rule (proposals.MOST_ROWS, audit IN-07): a screen that would remove more than
# half the rows with energy is refused, the unit being the likelier error.
MOST_ROWS = 0.5


def _expected_screen(unit: str, days: int) -> Any:
    """pandas: the 500–3,500 kcal a day screen read in ``unit`` over ``days`` days; any unit that
    is no unit of energy refuses it, its bounds kept in a day's kcal; a screen removing more than
    half the rows is refused too."""
    e = _energy_frame()["energy"]
    factor = {"kcal": 1.0, "kj": KJ_PER_KCAL}.get(unit)
    lo = round(500 * (factor or 1.0) * days, 1)
    hi = round(3500 * (factor or 1.0) * days, 1)
    n = int(((e < lo) | (e > hi)).sum())
    return (lo, hi, n, factor is None or n > MOST_ROWS * len(e))


def _probe_exclusions_unit(value: str) -> Any:
    return _screen(_offers(_energy_state("unit", value, fixed=("day_count", "1"))))


def _probe_exclusions_days(value: str) -> Any:
    return _screen(_offers(_energy_state("day_count", value, fixed=("unit", "kcal"))))


def _probe_energy_reading_unit(value: str) -> Any:
    from turbotab.core.stages.proposals import energy_unit_reading, recorded_energy_unit

    state = _energy_state("unit", value, fixed=("day_count", "1"))
    r = energy_unit_reading(_energy_frame(), "energy", recorded_energy_unit(state, "energy"))
    return (r["unit"] if r["confirmed"] else None, r["days"], r["confirmed"])


def _probe_energy_reading_days(value: str) -> Any:
    from turbotab.core.stages.proposals import energy_unit_reading, recorded_energy_unit

    state = _energy_state("day_count", value, fixed=("unit", "kcal"))
    r = energy_unit_reading(_energy_frame(), "energy", recorded_energy_unit(state, "energy"))
    return (r["unit"], r["days"], r["confirmed"])


class _EnergyStore:
    def __init__(self) -> None:
        self.frame = _energy_frame()
        self.columns = list(self.frame.columns)

    def materialize(self, columns: Any, ids: Any = None) -> pd.DataFrame:
        return self.frame[list(columns)]


def _wait_for_unit(state: ProjectState) -> Any:
    rule = {"column": "energy", "low": 500, "high": 3500, "reason": "implausible intake"}
    ctx = {"state": state, "columns": list(_energy_frame().columns), "target": "y",
           "store": lambda: _EnergyStore()}
    try:
        d.validate({"kind": "set_exclusions", "rules": [rule]}, ctx)
    except d.Refusal as refused:
        return refused.code == "energy_unit_unconfirmed"
    return False


def _probe_screens_wait_unit(value: str) -> Any:
    return _wait_for_unit(_energy_state("unit", value, fixed=("day_count", "1")))


def _probe_screens_wait_days(value: str) -> Any:
    return _wait_for_unit(_energy_state("day_count", value, fixed=("unit", "kcal")))


def _restated(state: ProjectState) -> Any:
    from turbotab.core.stages.finding_words import FindingContext, restate_implausible

    frame = _energy_frame()
    e = frame["energy"]
    f = {"affected_columns": ["energy"]}
    p = {"column": "energy", "minimum": 500.0, "maximum": 5000.0,
         "n_flagged": int(((e < 500) | (e > 5000)).sum())}
    restate_implausible(f, p, FindingContext(frame=frame, lens=["dietary"], target="y",
                                             units=state.column_units or {}))
    return (p["minimum"], p["maximum"], p["n_flagged"], bool(p.get("unit_unconfirmed")))


def _expected_restated(unit: str, days: int) -> Any:
    e = _energy_frame()["energy"]
    factor = {"kcal": 1.0, "kj": KJ_PER_KCAL}.get(unit)
    lo, hi = 500.0 * (factor or 1.0) * days, 5000.0 * (factor or 1.0) * days
    return (lo, hi, int(((e < lo) | (e > hi)).sum()), factor is None)


def _probe_restate_unit(value: str) -> Any:
    return _restated(_energy_state("unit", value, fixed=("day_count", "1")))


def _probe_restate_days(value: str) -> Any:
    return _restated(_energy_state("day_count", value, fixed=("unit", "kcal")))


def _probe_coach_suffix(value: str) -> Any:
    from turbotab.core.coach import _unit_suffix

    return _unit_suffix("energy", _energy_state("unit", value))


def _probe_equation_unit(value: str) -> Any:
    """Wave 2, EXPLAIN: the fitted equation's unit for a predictor, confirmed through the fold."""
    from turbotab.core.models.explain import equation_units

    state = _confirm(d.SetTarget(column="y"), d.SetRoles(roles={"x": "exposure"}),
                     kind="unit", column="x", value=value)
    return equation_units(state, "y", ["x"])[1].get("x")


# ── the Goldberg screen's body measures, energy unit and days ──

def _goldberg(offers: dict[str, Any]) -> Any:
    """What the offer reads (its units, its height and equation) and whether it waits for the
    energy column's unit; its count is the Goldberg tests' (``test_wp12c_goldberg``)."""
    o = offers.get("goldberg_schofield")
    if o is None:
        return None
    rule = o["rule"]
    waits = str(o.get("refused") or "").startswith("Refused until `energy`'s unit")
    return (rule["energy_unit"], rule["weight_unit"], rule["height"], rule["height_unit"],
            rule["equation"], waits)


def _goldberg_state(column: str, value: str) -> ProjectState:
    before = [d.SetLens(lenses=["dietary"]), d.SetTarget(column="y"),
              d.SetRoles(roles=ENERGY_ROLES),
              d.SetColumnUnit(column="energy", unit="kcal", days=1),
              d.ConfirmReading(reading="sex_coding", column="sex", value="female=F,male=M")]
    for other, unit in (("weight", "kg"), ("height", "cm"), ("age", "years")):
        if other != column:
            before.append(d.ConfirmReading(reading="unit", column=other, value=unit))
    if column == "energy":
        before = before[:3] + before[4:] + [d.ConfirmReading(reading="day_count", column="energy",
                                                             value="1")]
    return _confirm(*before, kind="unit", column=column, value=value)


def _probe_goldberg_weight(value: str) -> Any:
    return _goldberg(_offers(_goldberg_state("weight", value)))


def _probe_goldberg_height(value: str) -> Any:
    return _goldberg(_offers(_goldberg_state("height", value)))


def _probe_goldberg_age(value: str) -> Any:
    return _goldberg(_offers(_goldberg_state("age", value)))


def _probe_goldberg_energy(value: str) -> Any:
    return _goldberg(_offers(_goldberg_state("energy", value)))


def _probe_goldberg_days(value: str) -> Any:
    state = _energy_state("day_count", value, fixed=("unit", "kcal"))
    state = _folded(*[d.SetLens(lenses=["dietary"]), d.SetTarget(column="y"),
                      d.SetRoles(roles=ENERGY_ROLES),
                      d.ConfirmReading(reading="sex_coding", column="sex", value="female=F,male=M"),
                      d.ConfirmReading(reading="unit", column="weight", value="kg"),
                      d.ConfirmReading(reading="unit", column="height", value="cm"),
                      d.ConfirmReading(reading="unit", column="age", value="years"),
                      d.ConfirmReading(reading="unit", column="energy", value="kcal"),
                      d.ConfirmReading(reading="day_count", column="energy", value=value)])
    return _goldberg(_offers(state))


GOLDBERG_OK = ("kcal", "kg", "height", "cm", "schofield_height", False)


def _expected_goldberg(column: str, value: str) -> Any:
    """What the screen reads for each recorded unit: weight in kg or lb (converted exactly), else
    not offered; height in cm or m, else the weight-only equation; age in years, else not offered;
    total energy in kcal or kJ, else refused; one day only."""
    energy, weight, height, h_unit, equation, refused = GOLDBERG_OK
    if column == "weight":
        return None if value not in ("kg", "lb") else \
            (energy, value, height, h_unit, equation, refused)
    if column == "height":
        if value in ("cm", "m"):
            return (energy, weight, height, value, equation, refused)
        return (energy, weight, None, "cm", "schofield", refused)
    if column == "age":
        return GOLDBERG_OK if value == "years" else None
    if column == "energy":
        if value in ("kcal", "kj"):
            return (value, weight, height, h_unit, equation, refused)
        return ("kcal", weight, height, h_unit, equation, True)
    if column == "days":
        return GOLDBERG_OK if value == "1" else None
    raise AssertionError(column)


# ── sex codings ──

def _probe_sex_codes(value: str) -> Any:
    from turbotab.core.detectors.plausibility import sex_codes

    state = _confirm(d.SetTarget(column="y"), kind="sex_coding", column="sex", value=value)
    coded = sex_codes(pd.DataFrame({"sex": [1, 2, 2]}), state.sex_codings)
    return None if coded is None else tuple(coded.tolist())


def _probe_sex_column(value: str) -> Any:
    from turbotab.core.stages.proposals import sex_column

    state = _confirm(d.SetTarget(column="y"), kind="sex_coding", column="sex", value=value)
    return sex_column({"sex": {"dtype": "integer"}}, pd.DataFrame({"sex": [1, 2, 1, 2]}), {},
                      state=state)


# ── an energy source's kcal per unit, and the parts of totals ──

def _probe_substitution_validator_unit(value: str) -> Any:
    state = _confirm(d.SetTarget(column="y"),
                     d.SetRoles(roles={"Protein": "exposure", "fat_g": "exposure"}),
                     d.ConfirmReading(reading="unit", column="fat_g", value="g"),
                     kind="unit", column="Protein", value=value)
    ctx = {"state": state, "columns": ["Protein", "fat_g", "y"], "target": "y"}
    try:
        d.validate({"kind": "set_substitution", "donor": "Protein", "recipient": "fat_g"}, ctx)
    except d.Refusal as refused:
        return refused.code
    return None


def _probe_substitution_stage_unit(value: str) -> Any:
    """The stage's factor: the ledger's, read by the stage itself (an AST check below holds that
    the stage reads no other), for an InBody `Protein` and, since LEDGER-REPAIR-3, an `alcohol`
    column confirmed in the same unit (standard drinks measure alcohol only)."""
    state = _confirm(d.SetTarget(column="y"),
                     d.SetRoles(roles={"Protein": "exposure", "fat_g": "exposure",
                                       "alcohol": "exposure"}),
                     d.ConfirmReading(reading="unit", column="alcohol", value=value),
                     kind="unit", column="Protein", value=value)
    out = []
    for column in ("Protein", "alcohol"):
        found = R.kcal_per_unit(state, column)
        out.append(round(found.factor, 9) if found.settled else None)
    return tuple(out)


def _probe_imputation_factors(value: str) -> Any:
    """The kcal per unit the imputation's energy identity computes with (MS1): the ledger's settled
    factor for `Protein` and `alcohol` confirmed in ``value``, and none where it is not settled."""
    from types import SimpleNamespace

    from turbotab.core.stages.modeling import _settled_factors

    state = _confirm(d.SetTarget(column="y"),
                     d.SetRoles(roles={"Protein": "exposure", "fat_g": "exposure",
                                       "alcohol": "exposure", "kcal": "energy"}),
                     d.ConfirmReading(reading="unit", column="alcohol", value=value),
                     kind="unit", column="Protein", value=value)
    spec = SimpleNamespace(energy_adjustment=lambda: d.EnergyAdjustment(
        method="residual", energy_column="kcal", nutrients=["Protein", "alcohol"]))
    found = _settled_factors(SimpleNamespace(state=state, paths={}), spec)
    return tuple(round(found[c], 9) if c in found else None for c in ("Protein", "alcohol"))


def _expected_stage_unit(value: str) -> Any:
    """Protein: 4 kcal/g, 4,000 per kg, 1 per kcal, 1/4.184 per kJ, no drinks; alcohol: 7 kcal/g
    (NUTRITION_PACK §01), 7,000 per kg, 1 per kcal, 1/4.184 per kJ, and 7 × the drink's grams per
    standard drink (a US drink of 14 g: 98 kcal)."""
    grams = R.parse_drinks(value)
    alcohol = {"g": ATWATER["alcohol"], "kg": 1000 * ATWATER["alcohol"], "kcal": 1.0,
               "kj": round(1 / KJ_PER_KCAL, 9)}
    return (FACTORS.get(value),
            round(ATWATER["alcohol"] * grams, 9) if grams is not None else alcohol.get(value))


# The routing gate's ledger residue: the all-components model (and the partition) split each
# source into kcal by its settled kcal per unit, asked at the energy question and read by the
# design, never by the name (an `alcohol_g` holding US drinks moved at 7 kcal per unit).
ENERGY_SOURCES = ("protein_g", "fat_g", "carbohydrate_g", "alcohol")


def _energy_model_state(value: str) -> ProjectState:
    return _confirm(d.SetTarget(column="y"),
                    d.SetRoles(roles={"energy_kcal": "energy", **{c: "exposure"
                                                                   for c in ENERGY_SOURCES}}),
                    d.ConfirmReadings(items=[d.ReadingItem(reading="unit", column=c, value="g")
                                             for c in ENERGY_SOURCES[:3]]),
                    kind="unit", column="alcohol", value=value)


def _probe_energy_model_validator_unit(value: str) -> Any:
    ctx = {"state": _energy_model_state(value), "target": "y",
           "columns": ["energy_kcal", *ENERGY_SOURCES, "y"]}
    try:
        d.validate({"kind": "set_energy_adjustment", "method": "all_components",
                    "energy_column": "energy_kcal", "nutrients": list(ENERGY_SOURCES)}, ctx)
    except d.Refusal as refused:
        return refused.code
    return None


def _probe_energy_model_design_unit(value: str) -> Any:
    """The design's factor for `alcohol` (``models.pipeline._energy_factors``), read from the ledger
    with no values to read: the recorded unit alone."""
    from turbotab.core.models.pipeline import _energy_factors

    state = _energy_model_state(value)
    adj = d.EnergyAdjustment(method="all_components", energy_column="energy_kcal",
                             nutrients=list(ENERGY_SOURCES))
    found = _energy_factors(state, adj, pd.DataFrame({"alcohol": [1.0, 2.0]}), None)["alcohol"]
    return None if found["factor"] is None else round(float(found["factor"]), 9)


# The nesting probes' table (LEDGER-REPAIR-3, the sixth gate's q9): `sfa_g` reads as part of
# `fat_g` by its name and values; a confirmation may name any other column but the outcome, or none.
NESTED_COLUMNS = ("sfa_g", "fat_g", "carbohydrate_g", "protein_g", "y")


def nested_domain() -> tuple[str, ...]:
    """Every value the validator accepts for `sfa_g`'s ``nested_in`` on the probes' table."""
    out = []
    for value in [*NESTED_COLUMNS, R.NOT_NESTED, "no_such_column"]:
        try:
            d.validate({"kind": "confirm_reading", "reading": "nested_in", "column": "sfa_g",
                        "value": value}, {"columns": list(NESTED_COLUMNS), "target": "y"})
        except d.Refusal:
            continue
        out.append(value)
    return tuple(out)


def _nested_state(value: str) -> ProjectState:
    return _confirm(d.SetTarget(column="y"),
                    d.SetRoles(roles={c: "exposure" for c in NESTED_COLUMNS if c != "y"}),
                    *[d.ConfirmReading(reading="unit", column=c, value="g")
                      for c in NESTED_COLUMNS if c != "y"],
                    kind="nested_in", column="sfa_g", value=value)


def _probe_substitution_nested(value: str) -> Any:
    """`sfa_g` reads as part of `fat_g` (the design's nesting); the swap `sfa_g` → `carbohydrate_g`
    with `sfa_g` confirmed as part of each column: refused as moving nothing only when it names
    the recipient, settled for every other total and for none (LEDGER-REPAIR-3: any confirmation
    settles the reading; before it, only `fat_g` did)."""
    ctx = {"state": _nested_state(value), "columns": list(NESTED_COLUMNS), "target": "y",
           "artifact": lambda stage: ({"nested": [{"column": "sfa_g", "parent": "fat_g"}]}
                                if stage == "design" else None)}
    try:
        d.validate({"kind": "set_substitution", "donor": "sfa_g", "recipient": "carbohydrate_g"},
                   ctx)
    except d.Refusal as refused:
        return refused.code
    return None


def _nested_frame() -> pd.DataFrame:
    rng = np.random.default_rng(66009)  # the sixth gate's q9 seed, its first draws
    n = 60
    P, C, F = rng.normal(80, 20, n).clip(20), rng.normal(250, 60, n).clip(50), \
        rng.normal(75, 20, n).clip(15)
    return pd.DataFrame({"sfa_g": (F * rng.uniform(0.25, 0.45, n)).round(1), "fat_g": F.round(1),
                         "carbohydrate_g": C.round(1), "protein_g": P.round(1)})


NESTED_K = 18.0  # kcal moved: 2 g of `sfa_g` (9 kcal/g) for 4.5 g of `carbohydrate_g` (4 kcal/g)


def _probe_shift_nested(value: str) -> Any:
    """The curve's move (``modeling.shift_for``, what the substitution stage draws): each column's
    values after 18 kcal move from `sfa_g` to `carbohydrate_g`, the design's guess `sfa_g` ⊂
    `fat_g` handed in, the confirmation read over it; "refused" when nothing can move."""
    from turbotab.core.stages.modeling import shift_for

    X = _nested_frame()
    try:
        shift = shift_for(_nested_state(value), X, donor="sfa_g", recipient="carbohydrate_g",
                          kcal_per_unit={"sfa_g": 9.0, "carbohydrate_g": 4.0},
                          design_nested={"sfa_g": "fat_g"}, total=None)
    except ValueError:
        return "refused"
    return {c: np.round(v - X[c].to_numpy(float), 9).tolist()
            for c, v in sorted(shift.values(X, NESTED_K).items())}


def _expected_shift(value: str) -> Any:
    """NumPy: the donor gives up k/9 g, the recipient gains k/4 g, and the total `sfa_g` is
    confirmed part of (if any) moves by the same grams as its part; part of the recipient itself,
    the move goes nowhere and is refused."""
    if value == "carbohydrate_g":
        return "refused"
    n = len(_nested_frame())
    out = {"sfa_g": [round(-NESTED_K / 9.0, 9)] * n,
           "carbohydrate_g": [round(NESTED_K / 4.0, 9)] * n}
    if value != R.NOT_NESTED:
        out[value] = [round(-NESTED_K / 9.0, 9)] * n
    return dict(sorted(out.items()))


def _probe_design_nesting(value: str) -> Any:
    """The design's nesting (``modeling.design_nesting``): what the design artifact, its pairs, its
    estimand and the omitted-energy check read."""
    from turbotab.core.stages.modeling import design_nesting

    X = _nested_frame()
    return design_nesting(_nested_state(value), X, list(X.columns))


# ── text that holds numbers (LEDGER-REPAIR-3, the sixth gate's q3c) ──

TEXT_BMI = ["27.1", ".", "31.4", "22.0", ".", "25.5", "29.9", "24.3"]


def _band_folded(value: str) -> ProjectState:
    return _confirm(d.SetTarget(column="y"), d.SetPurpose(purpose="inference"),
                    d.SetRoles(roles={"band": "exposure"}),
                    d.SetEstimand(exposure="band", effect="total", measure="mean_difference"),
                    kind="code_or_count", column="band", value=value)


BAND_FACTS = {"band": {"whole": True, "zero_one": False, "n_values": 12, "min": 1, "max": 12}}
BAND_INFO = {"band": {"dtype": "integer", "n_unique": 12}}


def _probe_form_plan(value: str) -> Any:
    from turbotab.core.methods.exposure_form import form_plan

    plan = form_plan(_band_folded(value), BAND_INFO, BAND_FACTS)
    assert plan["waiting"] == []
    return ([n["column"] for n in plan["needs"]],
            [s["column"] for s in plan["stated"] if s["why"].startswith("declared codes")])


def _probe_form_declared(value: str) -> Any:
    try:
        d.validate({"kind": "set_exposure_form", "column": "band", "form": "linear"},
                   {"state": _band_folded(value), "column_info": BAND_INFO})
    except d.Refusal as refused:
        return refused.code
    return None


def _probe_text_amounts(value: str) -> Any:
    """The working table's reading of a text BMI with SAS `.` marks (``working.text_amounts``,
    evaluated by the SQL the working table runs): numbers, or the text as recorded."""
    import tempfile

    from turbotab.core.repairs import evaluate
    from turbotab.core.stages.working import text_amounts

    state = _confirm(d.SetTarget(column="y"), d.SetRoles(roles={"bmi": "covariate"}),
                     kind="code_or_count", column="bmi", value=value)
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "oriented.parquet"
        pd.DataFrame({"bmi": TEXT_BMI, d.ROW_ID: np.arange(len(TEXT_BMI))}).to_parquet(path)
        expressions = text_amounts(state, path, ["bmi"], {})
        got = evaluate(path, ["bmi"], expressions)["bmi"]
    return [None if pd.isna(v) else v for v in got.tolist()]


def _expected_text_amounts(value: str) -> Any:
    """pandas: an amount reads every value as a number, `.` blank; codes keep the text."""
    if value == "amount":
        return [None if pd.isna(v) else float(v)
                for v in pd.to_numeric(pd.Series(TEXT_BMI), errors="coerce")]
    return list(TEXT_BMI)


FACTORS = {"g": 9.0 / 9.0 * ATWATER["protein"], "kg": 1000 * ATWATER["protein"], "kcal": 1.0,
           "kj": round(1 / KJ_PER_KCAL, 9)}

# MI repair (MODELING_SEQUENCE §0 ruling 12): the clustered imputation carries a column to a unit's
# blank rows and imputes it once per unit only where the user confirmed it as one value per unit;
# the other answer imputes it row by row, and while unanswered the fit asks. `edu` is recorded at
# baseline only, so its values agree within every person whatever the truth.
VISITS = pd.DataFrame({"pid": [1, 1, 1, 2, 2, 3, 3, 3],
                       "edu": [12.0, np.nan, np.nan, 16.0, np.nan, np.nan, np.nan, np.nan],
                       "x": [1.0, 2.0, np.nan, 0.5, 0.7, 1.1, np.nan, 0.9]})


def _probe_time_invariance(value: str) -> Any:
    """The clustered imputation's columns (``missing.time_invariant_columns``): (imputed once per
    unit, still asked), with `edu`'s time-invariance confirmed as ``value``."""
    from turbotab.core.methods.missing import time_invariant_columns

    state = _confirm(d.SetTarget(column="y"),
                     d.SetRoles(roles={"edu": "covariate", "x": "exposure"}),
                     kind="time_invariant", column="edu", value=value)
    once, ask = time_invariant_columns(state, VISITS, VISITS["pid"].to_numpy(), ["edu", "x"])
    return once, [r.column for r in ask]

# (consumer, kind) -> (probe, oracle: value -> the behavior that value must produce)
PROBES: dict[tuple[str, str], tuple[Callable[[str], Any], Callable[[str], Any]]] = {
    ("turbotab.core.models.pipeline:model_predictors", "role"):
        (_probe_model_predictors, lambda v: v in PREDICTORS),
    ("turbotab.core.decisions:_models_read_settled_readings", "role"):
        (_probe_models_validator_role,
         lambda v: [("code_or_count", "x")] if v in PREDICTORS else []),
    ("turbotab.core.stages.rows:cohort_inputs", "role"):
        (_probe_cohort_inputs, lambda v: (v in PREDICTORS, v in PREDICTORS)),
    ("turbotab.core.stages.proposals:proposals_stage", "role"):
        (_probe_proposals_role, lambda v: v == "exposure"),
    ("turbotab.core.models.inference:resolve_clusters", "role"):
        (_probe_resolve_clusters_role,
         lambda v: ("hhid", 40) if v in ("identifier", "cluster") else (None, 0)),
    ("turbotab.core.models.inference:resolve_clusters", "cluster"):
        (_probe_resolve_clusters, lambda v: ("hhid", 40) if v == "yes" else (None, 0)),
    ("turbotab.core.seal:seal_inputs", "cluster"):
        (_probe_seal_inputs, lambda v: "hhid" if v == "yes" else None),
    ("turbotab.core.stages.rows:split_inputs", "cluster"):
        (_probe_split_inputs, lambda v: "hhid" if v == "yes" else None),
    ("turbotab.core.models.pipeline:design_spec", "code_or_count"):
        (_probe_design_spec, lambda v: v == "code"),
    ("turbotab.core.methods.missing:imputation_frame", "code_or_count"):
        (_probe_imputation, lambda v: "categorical" if v == "code" else "numeric"),
    ("turbotab.core.decisions:_models_read_settled_readings", "code_or_count"):
        (_probe_models_validator_codes, lambda v: None),
    ("turbotab.core.stages.working:time_column", "time_column"):
        (_probe_time_column, lambda v: "visit"),
    ("turbotab.core.stages.usual_intake:recall_order", "time_column"):
        (_probe_recall_order, lambda v: "visit"),
    ("turbotab.core.stages.proposals:energy_unit_reading", "unit:energy"):
        (_probe_energy_reading_unit,
         lambda v: (v, 1, True) if v in ("kcal", "kj") else (None, 1, False)),
    ("turbotab.core.stages.proposals:energy_unit_reading", "day_count"):
        (_probe_energy_reading_days, lambda v: ("kcal", int(v), True)),
    ("turbotab.core.stages.proposals:exclusion_proposals", "unit:energy"):
        (_probe_exclusions_unit, lambda v: _expected_screen(v, 1)),
    ("turbotab.core.stages.proposals:exclusion_proposals", "day_count"):
        (_probe_exclusions_days, lambda v: _expected_screen("kcal", int(v))),
    ("turbotab.core.decisions:_screens_wait_for_the_unit", "unit:energy"):
        (_probe_screens_wait_unit, lambda v: v not in ("kcal", "kj")),
    ("turbotab.core.decisions:_screens_wait_for_the_unit", "day_count"):
        (_probe_screens_wait_days, lambda v: False),
    ("turbotab.core.stages.finding_words:restate_implausible", "unit:energy"):
        (_probe_restate_unit, lambda v: _expected_restated(v, 1)),
    ("turbotab.core.stages.finding_words:restate_implausible", "day_count"):
        (_probe_restate_days, lambda v: _expected_restated("kcal", int(v))),
    ("turbotab.core.coach:_unit_suffix", "unit:energy"):
        (_probe_coach_suffix, lambda v: f" {'kJ' if v == 'kj' else v}"),
    # Wave 2, EXPLAIN: the equation states the unit confirmed, in words, whichever it is.
    ("turbotab.core.models.explain:equation_units", "unit:column"):
        (_probe_equation_unit, lambda v: R.unit_words(v)),
    ("turbotab.core.stages.proposals:goldberg_proposal", "unit:weight"):
        (_probe_goldberg_weight, lambda v: _expected_goldberg("weight", v)),
    ("turbotab.core.stages.proposals:goldberg_proposal", "unit:height"):
        (_probe_goldberg_height, lambda v: _expected_goldberg("height", v)),
    ("turbotab.core.stages.proposals:goldberg_proposal", "unit:age"):
        (_probe_goldberg_age, lambda v: _expected_goldberg("age", v)),
    ("turbotab.core.stages.proposals:goldberg_proposal", "unit:energy"):
        (_probe_goldberg_energy, lambda v: _expected_goldberg("energy", v)),
    ("turbotab.core.stages.proposals:goldberg_proposal", "day_count"):
        (_probe_goldberg_days, lambda v: _expected_goldberg("days", v)),
    ("turbotab.core.detectors.plausibility:sex_codes", "sex_coding"):
        (_probe_sex_codes, lambda v: (1.0, 2.0, 2.0) if v == "female=2,male=1"
         else (2.0, 1.0, 1.0)),
    ("turbotab.core.stages.proposals:sex_column", "sex_coding"):
        (_probe_sex_column, lambda v: ("sex", R.parse_sex_coding(v))),
    ("turbotab.core.decisions:_substitution_reads_settled_readings", "unit:factor"):
        (_probe_substitution_validator_unit,
         lambda v: None if v in FACTORS else "reading_unsettled"),
    ("turbotab.core.stages.modeling:substitution_stage", "unit:factor"):
        (_probe_substitution_stage_unit, _expected_stage_unit),
    # MI repair: the clustered imputation reads `edu`'s time-invariance as confirmed (and `x`,
    # whose records differ within a person, is imputed row by row, never asked).
    ("turbotab.core.methods.missing:time_invariant_columns", "time_invariant"):
        (_probe_time_invariance, lambda v: (["edu"], []) if v == "yes" else ([], [])),
    # MS1: the imputation's energy identity reads the same settled factors, and only those.
    ("turbotab.core.stages.modeling:_settled_factors", "unit:factor"):
        (_probe_imputation_factors, _expected_stage_unit),
    ("turbotab.core.decisions:_energy_adjustment_fits_the_roles", "unit:factor"):
        (_probe_energy_model_validator_unit,
         lambda v: None if _expected_stage_unit(v)[1] is not None else "reading_unsettled"),
    ("turbotab.core.models.pipeline:_energy_factors", "unit:factor"):
        (_probe_energy_model_design_unit, lambda v: _expected_stage_unit(v)[1]),
    ("turbotab.core.decisions:_substitution_reads_settled_readings", "nested_in"):
        (_probe_substitution_nested, lambda v: "part_of_the_other" if v == "carbohydrate_g"
         else None),
    # LEDGER-REPAIR-3 (the sixth gate): the curve's move and the design read the nesting the user
    # confirmed, whichever column it names; a text column confirmed "amount" is numbers for every
    # consumer, read on the working table.
    ("turbotab.core.stages.modeling:shift_for", "nested_in"):
        (_probe_shift_nested, _expected_shift),
    ("turbotab.core.stages.modeling:design_nesting", "nested_in"):
        (_probe_design_nesting, lambda v: {} if v == R.NOT_NESTED else {"sfa_g": v}),
    ("turbotab.core.stages.working:text_amounts", "code_or_count"):
        (_probe_text_amounts, _expected_text_amounts),
    # FORM (repair round): a 1–12 `band`, the declared exposure: codes are stated (indicators,
    # no form) and a form declared on them refused; an amount is asked its form and takes one.
    ("turbotab.core.methods.exposure_form:form_plan", "code_or_count"):
        (_probe_form_plan, lambda v: ([], ["band"]) if v == "code" else (["band"], [])),
    ("turbotab.core.methods.exposure_form:_form_reads_a_settled_reading", "code_or_count"):
        (_probe_form_declared, lambda v: "codes_take_no_form" if v == "code" else None),
}


def registered_pairs() -> set[tuple[str, str]]:
    return {(c.where, k) for c in R.CONSUMERS if c.changes for k in c.kinds
            if _base(k) in R.CONFIRMABLE}


def test_d1_every_registered_consumer_of_a_confirmable_kind_has_a_probe_and_an_oracle():
    """The property is generated from the registry: each consumer that changes a number names the
    kinds it reads, and each (consumer, confirmable kind) has a probe confirming through the fold
    and an oracle for every value; no probe stands for a consumer the registry does not name."""
    pairs = registered_pairs()
    assert pairs == set(PROBES), (pairs ^ set(PROBES))
    for _, kind in pairs:
        assert _base(kind) in R.CONFIRMABLE and (kind in R.KIND_RULES or _base(kind) in R.KINDS)


@pytest.mark.parametrize("key", sorted(PROBES), ids=lambda k: f"{k[0].split(':')[1]}-{k[1]}")
def test_d2_confirming_every_value_of_every_kind_produces_its_behavior(key):
    """For each registered (consumer, kind), every value the validator accepts, confirmed through
    the fold, produces the behavior its oracle computes independently (pandas counts, the exact unit
    factors, the role lists); and the consumer's behavior is not one constant across the values,
    unless the kind has one value."""
    probe, oracle = PROBES[key]
    _, kind = key
    seen = set()
    for value in domain(kind):
        got = probe(value)
        want = oracle(value)
        assert got == want, (key, value, got, want)
        seen.add(json.dumps(got, default=str, sort_keys=True))
    assert len(seen) > 1 or len(domain(kind)) == 1 or key in CONSTANT_BY_DESIGN, (key, seen)


# The models validator's code-or-amount answer: either answer settles the reading, so the fit no
# longer waits (what each answer fits is the design_spec and imputation probes'); a recorded unit
# with days probed by the screens' wait: any day count with kcal settles the screen's bounds.
CONSTANT_BY_DESIGN = {("turbotab.core.decisions:_models_read_settled_readings", "code_or_count"),
                      ("turbotab.core.decisions:_screens_wait_for_the_unit", "day_count")}


def test_d3_the_substitution_stage_reads_its_factor_from_the_ledger_only():
    """No name sets the stage's factor: the stage calls ``readings.kcal_per_unit`` and never reads
    ``energy_factor(...).factor`` (the fifth gate: `energy_factor(c).factor` from the name alone)."""
    import ast
    import inspect

    from turbotab.core.stages import modeling

    tree = ast.parse(inspect.getsource(modeling.substitution_stage))
    calls = {n.func.id if isinstance(n.func, ast.Name) else getattr(n.func, "attr", "")
             for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert "factor_of" in calls and "energy_factor" not in calls, calls


# ═════════════════════════════════════════════════════════════════════════════
# E · every offered exit of every ask executes
# ═════════════════════════════════════════════════════════════════════════════
#
# BLUEPRINT §14.3 (amendment, quoted above): "An ignored confirmation, or an offered answer that
# fails, is never accepted." The fifth gate's "codes" exit for `energy_kcal` crashed the design
# stage. Each test below reaches an ask through the server, then for every exit that carries a
# decision: posts it, answers the rest from the fixture's truth, records the asking decision, and
# runs the stage it unblocks, which must finish without an error; then reverts what it recorded,
# so the next exit starts from the same ask.


def _records(drive: Any) -> list[dict[str, Any]]:
    return drive.view()["decisions"]


def _revert_since(drive: Any, n: int) -> None:
    """Revert every live record after the first ``n``, newest first."""
    from turbotab.core.decisions import DecisionRecord, reverted

    records = [DecisionRecord.model_validate(r) for r in _records(drive)]
    gone = reverted(records)
    for record in reversed(records[n:]):
        if record.id in gone or record.decision.kind == "revert":
            continue
        r = drive.post({"kind": "revert", "decision_id": record.id})
        assert r.status_code == 200, r.text[:400]


def _stage_done(drive: Any, stage: str, timeout: float = 600.0) -> dict[str, Any]:
    import time

    end = time.monotonic() + timeout
    while True:
        status = drive.view()["stages"][stage]
        if status["status"] in ("fresh", "error"):
            return status
        assert time.monotonic() < end, status
        time.sleep(0.05)


def _each_exit(drive: Any, body: dict[str, Any], stage: str,
               check: Callable[[Any, dict[str, Any]], None] | None = None) -> list[dict[str, Any]]:
    """Every exit of the ask ``body`` meets, each posted from the same ask: the stage it unblocks
    finishes without an error. Returns the exits tried."""
    url = f"/api/projects/{drive.pid}/decisions"
    first = _post_when_reached(drive.c, url, body)
    assert first.status_code == 409 and first.json()["error"]["code"] in ASKING, first.text[:400]
    exits = [e for e in first.json()["error"]["exits"] if e.get("decision")]
    assert exits
    n = len(_records(drive))
    for item in exits:
        r = drive.post(item["decision"])
        assert r.status_code == 200, (item["label"], r.text[:600])
        r = _post_when_reached(drive.c, url, body)
        rounds = 0
        while r.status_code == 409 and r.json()["error"]["code"] in ASKING and rounds < 5:
            rounds += 1
            for decision in truth_answers(r.json()["error"], drive.truth):
                assert drive.post(decision).status_code == 200, decision
            r = _post_when_reached(drive.c, url, body)
        assert r.status_code == 200, (item["label"], r.text[:600])
        status = _stage_done(drive, stage)
        assert status["status"] == "fresh", (item["label"], status)
        for upstream in ("design", "fit"):
            if upstream in drive.view()["stages"]:
                got = drive.view()["stages"][upstream]
                assert got["status"] != "error", (item["label"], upstream, got)
        if check is not None:
            check(item, drive.c.get(f"/api/projects/{drive.pid}/stages/{stage}").json()["artifact"])
        _revert_since(drive, n)
    return exits


def mixed_table(n: int = 300) -> pd.DataFrame:
    """A small table that meets every reading the fit asks: decimal diagnosis codes, a skip-pattern
    gate, whole-number ages, a 1/2-coded sex (exempt), total energy (left out by "none")."""
    rng = np.random.default_rng(55100)
    f = pd.DataFrame({"participant_id": [f"M{i:04d}" for i in range(n)]})
    f["age"] = rng.integers(25, 75, n)
    f["sex"] = rng.choice([1, 2], n)
    f["dsm_dx"] = rng.choice([307.1, 307.51, 307.5, 307.59], n, p=[0.3, 0.3, 0.3, 0.1])
    f["alcohol_flag"] = rng.choice([0, 1], n, p=[0.35, 0.65])
    f["alcohol"] = np.where(f["alcohol_flag"] == 1, rng.gamma(2.0, 8.0, n).round(1), np.nan)
    P, C, F = rng.normal(80, 20, n).clip(20), rng.normal(250, 60, n).clip(50), \
        rng.normal(75, 20, n).clip(15)
    f["protein_g"], f["carbohydrate_g"], f["fat_g"] = P.round(1), C.round(1), F.round(1)
    f["energy_kcal"] = (4 * P + 4 * C + 9 * F + rng.normal(0, 25, n)).round(0)
    f["sbp"] = (118 + 0.4 * (f["age"] - 50) + 3 * f["alcohol_flag"] + rng.normal(0, 10, n)).round(1)
    return f


MIXED_TRUTH = {"code_or_count:age": "amount", "code_or_count:dsm_dx": "code",
               "role:alcohol_flag": "covariate", "code_or_count:alcohol": "amount",
               "unit:energy_kcal": "kcal", "day_count:energy_kcal": "1",
               "sex_coding:sex": "female=2,male=1"}


def test_e1_every_exit_the_fits_ask_offers_runs_the_fit(tmp_path):
    """The models ask on a table that meets every reading (decimal codes, a skip gate, a whole-
    number age): the block confirmation, each role's guess and its alternatives, each code-or-
    amount answer. Each runs the design and the fit to the end, and `energy_kcal` (left out by
    "none") and the 1/2 `sex` (one indicator either way) are not asked."""
    plan = answers("sbp", ["dietary"], purpose={"kind": "set_purpose", "purpose": "prediction"},
                   missing={"kind": "set_missing", "strategy": "impute"})
    with local_server(tmp_path / "home") as client:
        drive = open_project(client, write(mixed_table(), tmp_path, "mixed.csv"),
                             Truth(MIXED_TRUTH, fixture="mixed"))
        drive_unsettled(drive, plan, roles={})
        exits = _each_exit(drive, plan["models"], "fit")
    labels = " | ".join(e["label"] for e in exits)
    listed = {(e["decision"].get("reading"), e["decision"].get("column"))
              for e in exits if e["decision"]["kind"] == "confirm_reading"}
    assert ("code_or_count", "dsm_dx") in listed and ("role", "alcohol_flag") in listed, labels
    assert ("code_or_count", "energy_kcal") not in listed and ("code_or_count", "sex") not in listed
    values = {(e["decision"]["column"], e["decision"]["value"]) for e in exits
              if e["decision"]["kind"] == "confirm_reading"}
    assert {("dsm_dx", "code"), ("dsm_dx", "amount"), ("alcohol_flag", "flag"),
            ("alcohol_flag", "covariate")} <= values, values


def test_e2_every_exit_the_energy_units_ask_offers_runs_the_screens(tmp_path):
    """The screens' ask on the 4-day total (p4): each (unit, days) the ratio fits, and the other
    answers offered, is recorded, and the proposals and the participant flow recompute without an
    error; the chosen screen is then recorded or refused with its reason, never a crash."""
    frame = four_day_table()
    plan = answers("hba1c", ["dietary"])
    with local_server(tmp_path / "home") as client:
        drive = open_project(client, write(frame, tmp_path, "four_day_exits.csv"),
                             Truth(FOUR_DAY_TRUTH, fixture="p4 exits"))
        drive_unsettled(drive, plan, roles={}, stop_before="exclusions")
        rule = _screens(drive)["sex_neutral_500_3500"]["rule"]
        url = f"/api/projects/{drive.pid}/decisions"
        first = _post_when_reached(client, url, {"kind": "set_exclusions", "rules": [rule]})
        assert first.status_code == 409 and first.json()["error"]["code"] in ASKING
        exits = [e for e in first.json()["error"]["exits"] if e.get("decision")]
        assert len(exits) >= 2
        n = len(_records(drive))
        for item in exits:
            assert drive.post(item["decision"]).status_code == 200, item
            screens = _screens(drive)
            assert _stage_done(drive, "proposals")["status"] == "fresh"
            r = _post_when_reached(client, url, {"kind": "set_exclusions",
                                                 "rules": [screens["sex_neutral_500_3500"]["rule"]]})
            assert r.status_code in (200, 409), r.text[:400]
            if r.status_code == 200:
                assert _stage_done(drive, "cohort")["status"] == "fresh", item
            else:
                assert r.json()["error"]["code"] not in ASKING, r.text[:400]
            _revert_since(drive, n)


def factor_table(n: int = 300) -> pd.DataFrame:
    """Nutrients with no total energy to reconstruct (so no value settles a unit): `Protein`
    unmarked, `fat_g`, `carbohydrate_g`, an outcome."""
    rng = np.random.default_rng(55101)
    P, F, C = rng.normal(80, 20, n).clip(20), rng.normal(75, 20, n).clip(15), \
        rng.normal(250, 60, n).clip(50)
    f = pd.DataFrame({"participant_id": [f"F{i:04d}" for i in range(n)],
                      "age": rng.normal(50, 10, n).round(1), "Protein": P.round(1),
                      "fat_g": F.round(1), "carbohydrate_g": C.round(1)})
    f["sbp"] = (110 + 0.3 * f["age"] + 0.05 * f["Protein"] + 0.08 * f["fat_g"]
                + rng.normal(0, 8, n)).round(1)
    return f


def test_e3_every_unit_the_substitutions_ask_offers_draws_its_own_curve(tmp_path):
    """The substitution's ask (gate census item: `Protein` confirmed in kg moved energy at 4 kcal a
    unit): every unit offered for each source is recorded and the curve is drawn without an error,
    each at its own kcal per unit: moving k kcal from `Protein` to `fat_g` changes the linear
    model's prediction by k·(β_fat/9 − β_Protein/f), f the Protein unit's kcal (4 per g, 4,000 per
    kg, 1 per kcal, 1/4.184 per kJ; NIST, NUTRITION_PACK §01), β from NumPy on every analyzed
    row."""
    frame = factor_table()
    beta = ols(frame["sbp"].to_numpy(float),
               np.column_stack([np.ones(len(frame)), frame.age, frame.Protein, frame.fat_g,
                                frame.carbohydrate_g]).astype(float))
    def per(column: str) -> dict[str, float]:
        """kcal per unit of ``column`` in each unit: its Atwater factor per gram (protein 4, fat
        9), 1,000 times that per kg, 1 per kcal, 1/4.184 per kJ."""
        gram = ATWATER["protein" if column == "Protein" else "fat"]
        return {"g": gram, "kg": 1000 * gram, "kcal": 1.0, "kj": 1 / KJ_PER_KCAL}

    plan = answers("sbp", ["dietary"])
    body = {"kind": "set_substitution", "donor": "Protein", "recipient": "fat_g",
            "step_kcal": 100.0, "acknowledged": True}
    # WP17: `Protein`'s effect; age moves SBP (the generator); the other macronutrients are the
    # field's default for other dietary components (possible confounders).
    truth = Truth({"unit:Protein": "g", "unit:fat_g": "g", "exposure:sbp": "Protein",
                   "adjust:age": "no,yes,no", "adjust:fat_g": "unknown,unknown,no",
                   "adjust:carbohydrate_g": "unknown,unknown,no"}, fixture="factor")

    def check(item: dict[str, Any], art: dict[str, Any]) -> None:
        decision = item["decision"]
        f_p = per("Protein")[decision["value"] if decision["column"] == "Protein" else "g"]
        f_f = per("fat_g")[decision["value"] if decision["column"] == "fat_g" else "g"]
        k = art["ks"][1]
        model = next(m for m in art["models"] if m["family"] == "linear")
        # On support (the stage's amount check, counted with pandas): a row whose shifted amounts
        # stay within each column's observed range and at or above 0; the curve stops where fewer
        # than half the rows remain (``methods.substitution``'s stated floor).
        P, F = frame["Protein"], frame["fat_g"]
        p_new, f_new = P - k / f_p, F + k / f_f
        on = ((p_new >= P.min()) & (p_new <= P.max()) & (p_new >= 0)
              & (f_new >= F.min()) & (f_new <= F.max()) & (f_new >= 0))
        if on.mean() < 0.5:
            assert model["delta"][1] is None, (decision, model)
            return
        want = k * (beta[3] / f_f - beta[2] / f_p)
        assert model["delta"][1] == pytest.approx(want, rel=1e-6, abs=1e-9), (decision, model)

    with local_server(tmp_path / "home") as client:
        drive = open_project(client, write(frame, tmp_path, "factor.csv"), truth)
        for key in ORDER:
            if key in ("event", "task"):  # the outcome's stage first, as a client
                drive.artifact("target_info", timeout=300)
            step = drive.reach(key, timeout=300)
            if step["status"] not in ("open", "waiting"):
                continue
            if key == "roles":
                drive.decide_roles({"participant_id": "identifier", "age": "covariate",
                                    "Protein": "exposure", "fat_g": "exposure",
                                    "carbohydrate_g": "exposure"})
                continue
            drive.decide(plan[key])
        drive.artifact("fit", timeout=600)
        exits = _each_exit(drive, body, "substitution", check)
    offered = {(e["decision"]["column"], e["decision"]["value"]) for e in exits}
    units = set(per("Protein"))
    assert {("Protein", u) for u in units} | {("fat_g", u) for u in units} <= offered, offered


# ═════════════════════════════════════════════════════════════════════════════
# F · settlement is visible
# ═════════════════════════════════════════════════════════════════════════════


def test_f1_readings_settled_by_their_values_are_listed_with_evidence_and_a_change(tmp_path):
    """BLUEPRINT §14.3 (amendment, quoted above): the card's API lists the readings the values
    settled, each with its evidence and a way to change it, and the methods record states them.
    On the clean dietary table (p6, rng 55006): the identifier, the age and sex covariates, the
    nutrients and total energy read by their values, the outcome's task, energy's kcal by the
    Atwater identity, the BMI's decimals an amount, and sex's labels. Each change exit is a
    decision the server records. The select_models record carries "Read from the values"."""
    plan = answers("sbp", ["dietary"])
    with local_server(tmp_path / "home") as client:
        drive = open_project(client, write(clean_table(), tmp_path, "visible.csv"),
                             Truth(CLEAN_TRUTH, fixture="clean10 visible"))
        count: dict[str, Any] = {}
        _journey(drive, plan, count)
        drive.artifact("fit", timeout=600)
        card = client.get(f"/api/projects/{drive.pid}/readings").json()
        items = {(i["kind"], i["column"]): i for i in card["read_from_data"]}
        record = next(r for r in drive.view()["decisions"]
                      if r["decision"]["kind"] == "select_models")
        for item in card["read_from_data"]:
            assert item["evidence"] and item["change"], item
            for change in item["change"]:
                r = client.post(f"/api/projects/{drive.pid}/preview", json=change["decision"])
                assert r.status_code in (200, 409), (change, r.text[:300])
                if r.status_code == 409:
                    assert r.json()["error"]["code"] not in ("validation_error",), r.text[:300]
    assert ("role", "participant_id") in items and ("role", "protein_g") in items, items.keys()
    assert ("unit", "energy_kcal") in items and items[("unit", "energy_kcal")]["value"] == "kcal"
    assert ("code_or_count", "bmi") in items
    assert ("sex_coding", "sex") in items
    assert card["sentence"].startswith("Read from the values")
    assert "Read from the values" in (record.get("sentence") or ""), record.get("sentence")
    assert "`bmi` is an amount" in record["sentence"]


def test_e4_each_answer_the_time_roles_ask_offers_builds_a_design_that_fits():
    """The run date's ask (gate item p1 §2: the time role is the user's): both offered answers,
    confirmed through the fold, give a design the pipeline fits. As the time of each row it leaves
    the predictors; as a batch kept in the model its four dates are one indicator each (the
    reference the first date), and the fit is the NumPy fit on those indicators."""
    from turbotab.core.models.pipeline import design_spec, model_predictors, shared_steps, transformer

    assay, units = p1_assay_dates()
    rng = np.random.default_rng(10)
    X = pd.DataFrame({"RunDate": assay, "age": np.repeat(rng.normal(50, 10, 60).round(1), 3)})
    shift = assay.map({d: i * 0.4 for i, d in enumerate(sorted(assay.unique()))})
    y = (2 + 0.03 * X["age"] + shift + rng.normal(0, 0.5, len(X))).to_numpy(float)
    for value, kept in (("time", False), ("covariate", True)):
        state = _folded(d.SetTarget(column="crp"),
                        d.SetRoles(roles={"subject_id": "identifier", "RunDate": "time",
                                          "age": "covariate"}, unconfirmed=["RunDate"]),
                        d.ConfirmReading(reading="role", column="RunDate", value=value))
        predictors = model_predictors(state)
        assert ("RunDate" in predictors) == kept, (value, predictors)
        spec = design_spec(state, X[predictors], predictors)
        matrix = transformer(shared_steps(spec)).fit_transform(X[predictors])
        assert len(matrix) == len(X) and np.isfinite(matrix.to_numpy(float)).all()
        if kept:
            dates = sorted(assay.unique())
            D = np.column_stack([(assay == dt).to_numpy(float) for dt in dates[1:]])
            want = ols(y, np.column_stack([np.ones(len(X)), D, X["age"].to_numpy(float)]))
            got = ols(y, np.column_stack([np.ones(len(X)), matrix.to_numpy(float)]))
            assert matrix.shape[1] == 4  # three date indicators (one-hot first), then age
            assert np.allclose(got, want, atol=1e-8), (list(matrix.columns), got, want)
