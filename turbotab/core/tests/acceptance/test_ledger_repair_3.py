"""One test per kind, text that holds numbers, and column-valued confirmations (BLUEPRINT §14.3 and
its amendment after the fifth gate): package LEDGER-REPAIR-3.

The sixth gate (``/private/tmp/turbotab-fix/resume_args.json``, field ``gate6``) closed every item
of the fifth and found four:

* **Text that holds numbers** (q3, q3b): a BMI exported from SAS with ten ``.`` for missing, and a
  CRP with fifteen ``<0.20``, arrived as text; ``amounts_by_values`` settled them "code" ("labels,
  not numbers"), the fit entered 175 and 300 indicators where one slope belonged, nothing was
  asked, and the card listed neither reading.
* **A confirmation the fit ignored** (q3c): ``code_or_count`` bmi = amount was accepted and the fit
  still one-hot encoded the text.
* **A private settler and a band that cannot see a minor source** (q1, q2): alcohol recorded in US
  standard drinks moved energy at 7 kcal a drink (grams), not 98, read by a consumer's own Atwater
  test beside the registry's; even the registry's test passed it at a ratio of 1.02.
* **A column-valued confirmation read nowhere** (q9): ``nested_in`` sfa_g = carbohydrate_g was
  accepted, and the substitution asked the same question again.

Section A replays each draw for draw (``gate6_fixtures``: the probes' own seeds and draws), through
the real server where the gate drove it; each replay first shows the reading asked with no number
computed, then answers from the fixture's declared truth (``truths.Truth``), never a constant.
Section B is the structural check that no module outside ``readings.py`` decides a registry kind.
Section C: the property test's column-valued domains. Section D: settlement is visible.

Expected values never come from the code under test: NumPy least squares, pandas parsing and
counts, the Atwater factors (NUTRITION_PACK §01) and the quoted primary sources (fetched
2026-10-04).
"""
from __future__ import annotations

import ast
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from turbotab.core import decisions as d
from turbotab.core import readings as R
from turbotab.core.decisions import ProjectState
from turbotab.core.tests.acceptance import gate6_fixtures as g6
from turbotab.core.tests.acceptance.server_drive import (
    Truth, _post_when_reached, local_server, open_project,
)
from turbotab.core.tests.acceptance.test_discriminate import (
    answers, coefficients, drive_unsettled, fit_refused_without_a_number, ols, write,
)
from turbotab.core.tests.truths import ASKING, asked
from turbotab.core.tests.truths import answers as truth_answers

ROOT = Path(__file__).resolve().parents[4]

# ── primary sources, quoted (fetched 2026-10-04) ─────────────────────────────

BLUEPRINT_ALTERNATIVES = ("A kind is settled by its values only when it declares its plausible "
                          "alternatives and a value test that rejects each of them.")
BLUEPRINT_HONORED = ("For each consumer and each kind it reads, confirming each alternative must "
                     "produce that alternative's behavior. A confirmation that changes nothing is a "
                     "defect.")
BLUEPRINT_GATE = "a value test settles a reading that a plausible alternative would also pass."
# NIAAA, "What Is A Standard Drink?" (niaaa.nih.gov): the US standard drink.
NIAAA = ("In the United States, one standard drink contains about 14 grams, or about 0.6 fluid "
         "ounces, of pure alcohol.")
US_DRINK_G = 14.0
# NHS, "Calculating alcohol units" (nhs.uk): the UK unit.
NHS = "One unit equals 10ml or 8g of pure alcohol"
# Kalinowski & Humphreys 2016, Addiction 111(7):1293-1298 (abstract, Europe PMC): the range.
KALINOWSKI = "the modal standard drink size was 10 g pure ethanol, but variation was wide (8-20 g)"
# haven (tidyverse), tagged_na: SAS's and Stata's lettered missing values.
HAVEN = ('"Tagged" missing values work exactly like regular R missing values except that they '
         'store one additional byte of information a tag, which is usually a letter ("a" to "z").')
ATWATER = {"protein": 4.0, "carbohydrate": 4.0, "fat": 9.0, "alcohol": 7.0}  # NUTRITION_PACK §01
PASS_BAND = (0.90, 1.10)  # the nutrition pack's Atwater pass band (turbotab.nutrition PASS_LOW/HIGH)


def test_the_quoted_rules_are_the_blueprints_own():
    blueprint = " ".join((ROOT / "docs/turbotab-next/BLUEPRINT.md").read_text("utf-8").split())
    for sentence in (BLUEPRINT_ALTERNATIVES, BLUEPRINT_HONORED, BLUEPRINT_GATE):
        assert sentence in blueprint, sentence
    # The offered drinks are the sources' sizes, and the accepted range is theirs.
    assert {g for g, _ in R.OFFERED_DRINKS} == {US_DRINK_G, 8.0, 10.0}
    assert R.DRINK_GRAMS == (8.0, 20.0)


# ── helpers ──────────────────────────────────────────────────────────────────


def _numbers(series: pd.Series, lod_factor: float | None = None) -> pd.Series:
    """pandas, independently of the app: a text column read as numbers, ``.`` and other text
    blank, ``<x`` at ``lod_factor`` × x."""
    text = series.astype(str).str.strip()
    out = pd.to_numeric(text, errors="coerce")
    if lod_factor is not None:
        below = text.str.fullmatch(r"<\s*([0-9]*\.?[0-9]+)")
        limits = pd.to_numeric(text.str.extract(r"^<\s*([0-9]*\.?[0-9]+)$")[0], errors="coerce")
        out = out.where(~below, limits * lod_factor)
    return out


def _fresh_after(drive: Any, stage: str, old: str | None, timeout: float = 600.0) -> dict[str, Any]:
    """The stage's artifact once it is fresh under a key other than ``old``."""
    end = time.monotonic() + timeout
    while True:
        status = drive.view()["stages"][stage]
        assert status["status"] != "error", status
        if status["status"] == "fresh" and status["key"] != old:
            return drive.c.get(f"/api/projects/{drive.pid}/stages/{stage}").json()["artifact"]
        assert time.monotonic() < end, status
        time.sleep(0.05)


def _key(drive: Any, stage: str) -> str | None:
    return drive.view()["stages"][stage].get("key")


def _asks_without_a_number(drive: Any, body: dict[str, Any]) -> dict[str, Any]:
    """:func:`fit_refused_without_a_number` once the Router has reached the question again (an
    answer that rebuilds the working table holds later questions until it is read)."""
    r = _post_when_reached(drive.c, f"/api/projects/{drive.pid}/decisions", body)
    assert r.status_code == 409, r.text[:600]
    return fit_refused_without_a_number(drive, body)


# ═════════════════════════════════════════════════════════════════════════════
# A · the sixth gate's items, replayed draw for draw
# ═════════════════════════════════════════════════════════════════════════════

# ── A1 · text that holds numbers (q3, q3b, q3c) ──


def test_a1_the_value_test_reads_numbers_written_as_text_and_settles_nothing():
    """Gate item 1 (q3, rng 66003). Before: ``amounts_by_values`` settled "code" for the BMI with ten
    SAS ``.`` and the CRP with fifteen ``<0.20``. Expected (pandas counts): neither settles; each
    names its numbers and its marks, the CRP its values below a detection limit; labels still settle
    codes; and the registry declares numbers written as text as the alternative its test rejects."""
    table = g6.text_numbers_table()
    bmi, crp = table["bmi"], table["crp"]
    assert int(pd.to_numeric(bmi, errors="coerce").notna().sum()) == 490
    assert int((bmi == ".").sum()) == 10
    assert int(pd.to_numeric(crp, errors="coerce").notna().sum()) == 485
    assert int((crp == "<0.20").sum()) == 15
    for column, numbers in (("bmi", 490), ("crp", 485)):
        verdict = R.by_values("code_or_count", table[column])
        assert not verdict.settles, verdict
        assert verdict.detail["numbers"] == numbers and verdict.detail["n_values"] == 500
        assert "numbers with missing marks" in verdict.evidence
    assert R.by_values("code_or_count", bmi).detail["marks"] == {".": 10}
    assert R.by_values("code_or_count", crp).detail["below"] == {"0.20": 15}
    labels = R.by_values("code_or_count", pd.Series(["never", "former", "current"] * 40))
    assert labels.settles and labels.value == "code"
    rule = R.KIND_RULES["code_or_count"]
    assert any("numbers written as text" in a for a in rule.alternatives)
    assert {a for a, _ in rule.rejects} == set(rule.alternatives)
    # Stata's and SAS's lettered missing values (haven: a tag "a" to "z") are marks too, and a
    # column of mostly marks with a few numbers is still numbers written as text.
    lettered = pd.Series([".a", ".B", "12.5", "13.1", "", "NA"] * 20)
    found = R.by_values("code_or_count", lettered)
    assert not found.settles and found.detail["numbers"] == 40
    assert sum(found.detail["marks"].values()) == 80


# WP17 under inference: each table's question and each covariate's causal place (causes the
# exposure, causes the outcome, changed by the exposure), from the generators (``gate6_fixtures``).
# The tests' roles name no exposure, so the question is age's effect, the estimand card's first
# offer; it carries no energy, so the plan's "none" energy answer stays coherent (a substitution or
# an addition is asked only of an energy-bearing exposure). Every other column is drawn apart from
# age. Where the generator gives a dietary column no effect but the reference fit holds it, it is
# answered with the field's default for another dietary component, a possible confounder.
TEXT_CAUSAL = {"exposure:sbp": "age", "adjust:bmi": "no,yes,no", "adjust:crp": "no,yes,no"}
DRINKS_CAUSAL = {"exposure:sbp": "age", "adjust:alcohol": "no,yes,no",
                 "adjust:carbohydrate_g": "no,yes,no", "adjust:protein_g": "unknown,unknown,no",
                 "adjust:fat_g": "unknown,unknown,no"}
NESTED_CAUSAL = {"exposure:ldl": "age", "adjust:sfa_g": "no,yes,no",
                 "adjust:carbohydrate_g": "no,yes,no", "adjust:protein_g": "unknown,unknown,no",
                 "adjust:fat_g": "unknown,unknown,no"}


def test_a1_a_text_bmi_is_asked_and_fit_as_one_slope_through_the_server(tmp_path):
    """Gate items 1 and 2 (q3b: CRP and vitamin D left out; through the server, inference). Before:
    `bmi` was proposed a covariate, nothing was asked, and the fit entered 175 indicators (177
    features); NumPy's one slope on the 490 rows whose BMI is a number is 0.3302, age 0.4459, and
    the app's age read 0.4359. Expected: the fit asks the code-or-amount reading of `bmi`, guessing
    an amount ("numbers with missing marks"), computes nothing until answered, and with the
    fixture's truth (an amount) fits one slope, the NumPy fit on those 490 rows."""
    table = g6.text_numbers_table()
    numbers = _numbers(table["bmi"])
    keep = numbers.notna()
    X = np.column_stack([np.ones(int(keep.sum())), table.loc[keep, "age"], numbers[keep]])
    beta = ols(table.loc[keep, "sbp"].to_numpy(float), X.astype(float))
    assert (int(keep.sum()), round(beta[2], 4), round(beta[1], 4)) == (490, 0.3302, 0.4459)
    plan = answers("sbp", ["clinical"])
    truth = Truth({"code_or_count:bmi": "amount", "code_or_count:age": "amount", **TEXT_CAUSAL},
                  fixture="q3b (SAS-exported BMI)")
    with local_server(tmp_path / "home") as client:
        drive = open_project(client, write(table, tmp_path, "text_numbers_fit.csv"), truth)
        drive_unsettled(drive, plan, roles={"crp": "excluded", "vitd": "excluded",
                                            "bmi": "covariate", "age": "covariate",
                                            "participant_id": "identifier"})
        error = fit_refused_without_a_number(drive, plan["models"])
        assert ("code_or_count", "bmi") in asked(error["exits"]), error
        offered = {(e["decision"]["column"], e["decision"]["value"]) for e in error["exits"]
                   if (e.get("decision") or {}).get("kind") == "confirm_reading"}
        assert {("bmi", "amount"), ("bmi", "code")} <= offered, offered
        assert "numbers with missing marks" in error["message"] and "`.`" in error["message"]
        drive.decide(plan["models"])
        coef = coefficients(drive.artifact("fit", timeout=600))
    assert set(coef) == {"(intercept)", "age", "bmi"}, set(coef)
    for i, name in enumerate(("(intercept)", "age", "bmi")):
        assert coef[name]["estimate"] == pytest.approx(beta[i], abs=1e-6), name


def test_a1_a_text_bmi_confirmed_an_amount_first_is_one_slope(tmp_path):
    """Gate item 2 (q3c): ``confirm_reading`` code_or_count bmi = amount returned 200, and the fit
    still entered 175 indicators. Expected: the confirmation converts the column on the working
    table (marks blank), so the fit asks nothing about `bmi` and fits its one slope (NumPy)."""
    table = g6.text_numbers_table()
    numbers = _numbers(table["bmi"])
    keep = numbers.notna()
    X = np.column_stack([np.ones(int(keep.sum())), table.loc[keep, "age"], numbers[keep]])
    beta = ols(table.loc[keep, "sbp"].to_numpy(float), X.astype(float))
    plan = answers("sbp", ["clinical"])
    truth = Truth({"code_or_count:bmi": "amount", "code_or_count:age": "amount", **TEXT_CAUSAL},
                  fixture="q3c")
    with local_server(tmp_path / "home") as client:
        drive = open_project(client, write(table, tmp_path, "text_amount_confirm.csv"), truth)
        drive_unsettled(drive, plan, roles={"crp": "excluded", "vitd": "excluded",
                                            "bmi": "covariate", "age": "covariate",
                                            "participant_id": "identifier"})
        r = drive.post({"kind": "confirm_reading", "reading": "code_or_count", "column": "bmi",
                        "value": "amount"})
        assert r.status_code == 200, r.text[:400]
        r = _post_when_reached(client, f"/api/projects/{drive.pid}/decisions", plan["models"])
        assert r.status_code == 200, r.text[:600]
        coef = coefficients(drive.artifact("fit", timeout=600))
        working = drive.c.get(f"/api/projects/{drive.pid}/stages/working").json()["artifact"]
    assert set(coef) == {"(intercept)", "age", "bmi"}, set(coef)
    assert coef["bmi"]["estimate"] == pytest.approx(beta[2], abs=1e-6)
    assert "bmi" in {x["column"] for x in working["repairs"]}  # read as numbers for every consumer


def test_a1_a_censored_crp_is_routed_to_the_detection_limit_question(tmp_path):
    """Gate item 1 (q3: CRP with fifteen ``<0.20``; through the server). Before: answering its role
    entered 300 indicators. Expected: the fit asks its code-or-amount reading; once "amount" is the
    answer it asks how a value below the detection limit is read (half the limit, or the limit
    over √2: the below-detection repair's offers), never reading ``<0.20`` as a blank or a level;
    each answer offered fits the NumPy slope with ``<0.20`` at that fraction of 0.20."""
    table = g6.text_numbers_table()
    plan = answers("sbp", ["clinical"])
    truth = Truth({"code_or_count:crp": "amount", "code_or_count:age": "amount",
                   "detection_limit:crp": "half_limit", **TEXT_CAUSAL},
                  fixture="q3 (censored CRP)")

    def reference(factor: float) -> np.ndarray:
        crp = _numbers(table["crp"], factor)
        assert crp.notna().all()
        X = np.column_stack([np.ones(len(table)), table["age"], crp]).astype(float)
        return ols(table["sbp"].to_numpy(float), X)

    with local_server(tmp_path / "home") as client:
        drive = open_project(client, write(table, tmp_path, "censored_crp.csv"), truth)
        drive_unsettled(drive, plan, roles={"crp": "covariate", "vitd": "excluded",
                                            "bmi": "excluded", "age": "covariate",
                                            "participant_id": "identifier"})
        error = fit_refused_without_a_number(drive, plan["models"])
        assert ("code_or_count", "crp") in asked(error["exits"]), error
        assert "below a detection limit" in error["message"]
        assert drive.post({"kind": "confirm_reading", "reading": "code_or_count", "column": "crp",
                           "value": "amount"}).status_code == 200
        error = _asks_without_a_number(drive, plan["models"])
        limits = {e["decision"]["option"]: e["decision"] for e in error["exits"]
                  if (e.get("decision") or {}).get("kind") == "apply_repair"}
        assert set(limits) == {"half_limit", "limit_root2"}, error["exits"]
        assert all(dd["finding_id"] == "below_detection__crp" for dd in limits.values())
        drive.decide(plan["models"])  # the truth's answer: half the limit
        coef = coefficients(drive.artifact("fit", timeout=600))
        assert set(coef) == {"(intercept)", "age", "crp"}, set(coef)
        assert coef["crp"]["estimate"] == pytest.approx(reference(0.5)[2], abs=1e-6)
        # The other offered answer runs too, and moves the slope as its reading does.
        old = _key(drive, "fit")
        assert drive.post(limits["limit_root2"]).status_code == 200
        coef = coefficients(_fresh_after(drive, "fit", old))
    assert coef["crp"]["estimate"] == pytest.approx(reference(1 / np.sqrt(2))[2], abs=1e-6)


def test_a1_labels_settle_codes_listed_with_evidence_and_an_amount_answer_is_refused(tmp_path):
    """Settlement is visible (the gate: "Labels settled as codes are never listed"): a text
    predictor whose values are labels is listed under "read from your data" as codes, with its
    labels counted (pandas) and a way to change it; "amount" for it is refused with its exits, since
    every value would be blank, never accepted and ignored."""
    rng = np.random.default_rng(66011)
    n = 300
    frame = pd.DataFrame({"participant_id": [f"L{i:04d}" for i in range(n)],
                          "age": rng.normal(50, 12, n).round(1),
                          "smoking": rng.choice(["never", "former", "current"], n),
                          "sbp": rng.normal(120, 12, n).round(1)})
    plan = answers("sbp", ["clinical"])
    # WP17 under inference: the roles name no exposure, so the question is age's (the card's first
    # offer). The generator draws `sbp` apart from both columns; `smoking` is answered as the
    # subject matter has it (a cause of blood pressure, not of age), which keeps it in the fit
    # whose readings the card lists.
    truth = Truth({"code_or_count:age": "amount", "exposure:sbp": "age",
                   "adjust:smoking": "no,yes,no"}, fixture="labels")
    with local_server(tmp_path / "home") as client:
        drive = open_project(client, write(frame, tmp_path, "labels.csv"), truth)
        drive_unsettled(drive, plan, roles={"smoking": "covariate", "age": "covariate",
                                            "participant_id": "identifier"})
        drive.decide(plan["models"])
        drive.artifact("fit", timeout=600)
        card = client.get(f"/api/projects/{drive.pid}/readings").json()
        r = drive.post({"kind": "confirm_reading", "reading": "code_or_count", "column": "smoking",
                        "value": "amount"})
    items = {(i["kind"], i["column"]): i for i in card["read_from_data"]}
    item = items[("code_or_count", "smoking")]
    assert item["value"] == "code" and item["change"]
    assert f"`{frame['smoking'].nunique()}` labels" in item["evidence"]
    assert r.status_code == 409 and r.json()["error"]["code"] == "not_numbers", r.text[:400]
    assert {e["decision"]["value"] for e in r.json()["error"]["exits"]} == {"code", "excluded"}


def test_a1_text_numbers_that_change_within_units_are_asked_before_combining():
    """Combining a unit's rows is a consumer of the same reading: a text BMI with SAS ``.`` that
    changes within units would be combined as a category (its most frequent spelling) where its
    mean is the amount. It is asked (``working.code_or_count_facts``, the combine scope), guessing
    an amount; labels that change within units are not."""
    from turbotab.core.stages.working import code_or_count_facts

    rng = np.random.default_rng(66012)
    units = np.repeat(np.arange(40), 3)
    bmi = [("." if i % 17 == 3 else f"{v:.1f}") for i, v in enumerate(rng.normal(27, 4, 120))]
    frame = pd.DataFrame({"pid": units, "bmi": bmi,
                          "visit": np.tile(["baseline", "month_6", "month_12"], 40)})
    facts = code_or_count_facts(frame, "pid")
    assert "bmi" in facts and "visit" not in facts, facts
    reading = R.code_or_count_reading(ProjectState(), "bmi", facts["bmi"], scope="combine")
    assert reading is not None and not reading.settled and reading.value == "amount"


# ── A2 · an energy source's unit: alcohol in standard drinks (q1, q2) ──


class _Store:
    def __init__(self, frame: pd.DataFrame):
        self.frame, self.columns = frame, list(frame.columns)

    def materialize(self, columns: Any, ids: Any = None) -> pd.DataFrame:
        return self.frame[list(columns)]


def test_a2_alcohol_in_standard_drinks_is_never_settled_by_the_atwater_identity():
    """Gate item 3 (q1, rng 66001; NIAAA, quoted: a US drink holds about 14 g of alcohol, 98 kcal
    at 7 kcal/g). Before: the registry's test passed `alcohol_g` at a ratio of 1.02, and the
    consumer's own settler settled every macronutrient total, an unmarked `alcohol` the identity
    had left out included, at 7 kcal a drink. Expected (NumPy): read as grams, the reconstruction
    sits inside the pass band, and read in drinks too; alcohol carries about 4% of the energy, so no
    name settles it, and the consumers' factor is unsettled, asked with every drink size offered;
    the major sources beside it settle grams because each other unit leaves the band."""
    for name in ("alcohol", "alcohol_drinks", "alcohol_g", "drinks_per_day", "etoh"):
        frame = g6.alcohol_factor_frame(name)
        E = frame["energy_kcal"].to_numpy(float)
        macros = (4 * frame["protein_g"] + 4 * frame["carbohydrate_g"] + 9 * frame["fat_g"]
                  ).to_numpy(float)
        drinks = frame[name].to_numpy(float)
        as_grams = float(np.median(E / (macros + ATWATER["alcohol"] * drinks)))
        as_drinks = float(np.median(E / (macros + ATWATER["alcohol"] * US_DRINK_G * drinks)))
        share = float(np.mean(ATWATER["alcohol"] * US_DRINK_G * drinks / E))
        assert PASS_BAND[0] <= as_grams <= PASS_BAND[1] and PASS_BAND[0] <= as_drinks <= PASS_BAND[1]
        assert share < 0.06, share
        verdict = R.by_values("unit:factor", frame, "energy_kcal", name)
        assert not verdict.settles, (name, verdict)
        state = d.fold([d.DecisionRecord(id="r1", seq=1, at="2026-10-04T00:00:00Z",
                                         decision=d.SetTarget(column="y")),
                        d.DecisionRecord(id="r2", seq=2, at="2026-10-04T00:00:00Z",
                                         decision=d.SetRoles(roles={
                                             "energy_kcal": "energy", "protein_g": "exposure",
                                             "fat_g": "exposure", "carbohydrate_g": "exposure",
                                             name: "exposure"}))])
        store = _Store(frame)
        found = R.kcal_per_unit(state, name, store)
        assert not found.settled and found.factor is None, (name, found)
        assert R.unsettled_factors(state, [name, "protein_g", "fat_g", "carbohydrate_g"],
                                   store) == [name]
        for major, factor in (("protein_g", 4.0), ("fat_g", 9.0), ("carbohydrate_g", 4.0)):
            kcal_major = factor * frame[major].to_numpy(float)
            others = macros + ATWATER["alcohol"] * drinks - kcal_major
            for grams_per_unit in (1000.0, 1.0 / factor, 1.0 / (4.184 * factor)):
                moved = float(np.median(E / (others + kcal_major * grams_per_unit)))
                assert not PASS_BAND[0] <= moved <= PASS_BAND[1], (major, grams_per_unit, moved)
            assert R.kcal_per_unit(state, major, store).factor == factor
    offered = {e["decision"]["value"] for e in R.factor_exits("alcohol") if e.get("decision")}
    assert {"drinks:14", "drinks:8", "drinks:10", "g", "kcal"} <= offered, offered


def test_a2_a_carbohydrate_to_alcohol_swap_asks_alcohols_unit_and_moves_drinks_at_98_kcal(
        tmp_path):
    """Gate item 3 (q2, rng 66002, through the server; energy built at 98 kcal a drink). Before:
    ``set_substitution`` carbohydrate_g → alcohol at 20 kcal returned 200 with nothing asked, and the
    linear model's change was +3.0459 (NumPy at 7 kcal a unit) where 98 kcal a drink gives
    +0.2183; no record said what a unit of alcohol moved. Expected: the swap asks alcohol's unit,
    the drinks offered, and computes nothing; with the fixture's truth (US drinks of 14 g) the curve
    moves 20/98 drinks a step and its change is the NumPy one, and its note states 98 kcal per
    unit."""
    table = g6.alcohol_substitution_table()
    X = np.column_stack([np.ones(len(table)), table[["age", "protein_g", "fat_g", "carbohydrate_g",
                                                     "alcohol"]].to_numpy(float)])
    beta = ols(table["sbp"].to_numpy(float), X)
    k = 20.0

    def change(alcohol_kcal_per_unit: float) -> float:
        return k * (beta[5] / alcohol_kcal_per_unit - beta[4] / ATWATER["carbohydrate"])

    assert (round(change(7.0), 4), round(change(98.0), 4)) == (3.0459, 0.2183)
    plan = answers("sbp", ["dietary"])
    roles = {"participant_id": "identifier", "age": "covariate", "protein_g": "exposure",
             "fat_g": "exposure", "carbohydrate_g": "exposure", "alcohol": "exposure",
             "energy_kcal": "energy"}
    truth = Truth({"code_or_count:age": "amount", "unit:energy_kcal": "kcal",
                   "day_count:energy_kcal": "1", "code_or_count:energy_kcal": "amount",
                   "unit:alcohol": R.drinks_value(US_DRINK_G), **DRINKS_CAUSAL},
                  fixture="q2 (drinks)")
    body = {"kind": "set_substitution", "donor": "carbohydrate_g", "recipient": "alcohol",
            "step_kcal": k, "acknowledged": True}
    with local_server(tmp_path / "home") as client:
        drive = open_project(client, write(table, tmp_path, "alcohol_sub.csv"), truth)
        drive_unsettled(drive, plan, roles=roles, stop_before="__none__")
        coef = coefficients(drive.artifact("fit", timeout=600))
        assert set(coef) == {"(intercept)", "age", "protein_g", "fat_g", "carbohydrate_g",
                             "alcohol"}, set(coef)
        r = _post_when_reached(client, f"/api/projects/{drive.pid}/decisions", body)
        assert r.status_code == 409 and r.json()["error"]["code"] in ASKING, r.text[:400]
        error = r.json()["error"]
        assert asked(error["exits"]) == [("unit", "alcohol")], error["exits"]
        offered = {e["decision"]["value"] for e in error["exits"] if e.get("decision")}
        assert {"drinks:14", "drinks:8", "drinks:10", "g", "kcal"} <= offered, offered
        assert not drive.c.get(f"/api/projects/{drive.pid}/stages/substitution").json().get(
            "artifact")
        drive.decide(body)
        art = drive.artifact("substitution", timeout=600)
        records = drive.view()["decisions"]
    model = next(m for m in art["models"] if m["family"] == "linear")
    assert art["ks"][1] == k
    assert model["delta"][1] == pytest.approx(change(ATWATER["alcohol"] * US_DRINK_G), abs=1e-6)
    assert "alcohol moves at 98 kcal per unit" in art["note"], art["note"]
    confirmed = [x for x in records if x["decision"]["kind"] in ("confirm_readings",
                                                                  "confirm_reading")]
    assert any("drinks:14" in str(x["decision"]) for x in confirmed)


# ── A3 · a column-valued confirmation (q9) ──


def test_a3_a_part_confirmed_in_any_total_moves_with_that_total_or_none(tmp_path):
    """Gate item 4 (q9, rng 66009, through the server). Before: ``nested_in`` sfa_g =
    carbohydrate_g returned 200, and the swap sfa_g → carbohydrate_g asked the same question again;
    only `fat_g` settled it. Expected: the swap asks the nesting with the guess and "no total"
    offered; confirmed part of the recipient, the swap is refused as moving nothing; confirmed part
    of `protein_g`, of no total, or of `fat_g`, the curve moves that total with its part by the same
    grams (or nothing), and its change is NumPy's: k·(β_carb/4 − β_sfa/9 − β_total/9)."""
    table = g6.nested_table()
    names = ["age", "protein_g", "fat_g", "carbohydrate_g", "sfa_g"]
    X = np.column_stack([np.ones(len(table)), table[names].to_numpy(float)])
    beta = dict(zip(["(intercept)", *names], ols(table["ldl"].to_numpy(float), X)))
    k = 20.0

    def change(total: str | None) -> float:
        out = k * (beta["carbohydrate_g"] / 4.0 - beta["sfa_g"] / 9.0)
        return out - (k / 9.0 * beta[total] if total else 0.0)

    plan = answers("ldl", ["dietary"])
    roles = {"participant_id": "identifier", "age": "covariate", "protein_g": "exposure",
             "fat_g": "exposure", "carbohydrate_g": "exposure", "sfa_g": "exposure",
             "energy_kcal": "energy"}
    truth = Truth({"code_or_count:age": "amount", "unit:energy_kcal": "kcal",
                   "day_count:energy_kcal": "1", "code_or_count:energy_kcal": "amount",
                   **NESTED_CAUSAL}, fixture="q9 (nested)")
    body = {"kind": "set_substitution", "donor": "sfa_g", "recipient": "carbohydrate_g",
            "step_kcal": k, "acknowledged": True}
    url = None
    seen: dict[str, Any] = {}
    with local_server(tmp_path / "home") as client:
        drive = open_project(client, write(table, tmp_path, "nested.csv"), truth)
        url = f"/api/projects/{drive.pid}/decisions"
        drive_unsettled(drive, plan, roles=roles, stop_before="__none__")
        coef = coefficients(drive.artifact("fit", timeout=600))
        assert set(coef) == {"(intercept)", *names}, set(coef)
        for c in ("sfa_g", "carbohydrate_g"):
            assert drive.post({"kind": "confirm_reading", "reading": "unit", "column": c,
                               "value": "g"}).status_code == 200
        r = _post_when_reached(client, url, body)
        assert r.status_code == 409 and r.json()["error"]["code"] == "reading_unsettled"
        offered = {e["decision"]["value"] for e in r.json()["error"]["exits"]
                   if (e.get("decision") or {}).get("reading") == "nested_in"}
        assert offered == {"fat_g", R.NOT_NESTED}, offered
        assert drive.post({"kind": "confirm_reading", "reading": "nested_in", "column": "sfa_g",
                           "value": "carbohydrate_g"}).status_code == 200
        r = _post_when_reached(client, url, body)
        assert r.status_code == 409 and r.json()["error"]["code"] == "part_of_the_other", r.text
        for total in ("protein_g", R.NOT_NESTED, "fat_g"):
            old = _key(drive, "substitution")
            assert drive.post({"kind": "confirm_reading", "reading": "nested_in",
                               "column": "sfa_g", "value": total}).status_code == 200
            r = _post_when_reached(client, url, body)
            assert r.status_code == 200, (total, r.text[:600])
            seen[total] = _fresh_after(drive, "substitution", old)
    for total, art in seen.items():
        model = next(m for m in art["models"] if m["family"] == "linear")
        parent = None if total == R.NOT_NESTED else total
        assert art["carried"] == ([parent] if parent else []), (total, art["carried"])
        assert model["delta"][1] == pytest.approx(change(parent), abs=1e-6), total


# ═════════════════════════════════════════════════════════════════════════════
# B · one test per kind, no private settler (structural)
# ═════════════════════════════════════════════════════════════════════════════

LEDGER = "turbotab.core.readings"


class _Sites:
    """Every call in turbotab/core and turbotab/server (tests aside), read with ``ast`` and
    resolved through each module's and each function's imports to ``module:function``, and every
    ``from … import``."""

    def __init__(self, sources: list[tuple[str, str]]):
        self.calls: list[tuple[str, str, int]] = []
        self.imports: list[tuple[str, str, int]] = []
        for module, source in sources:
            tree = ast.parse(source)
            top = self._bindings(tree.body)
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    self._walk(module, node.name, node, top)
                elif isinstance(node, ast.ClassDef):
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            self._walk(module, f"{node.name}.{item.name}", item, top)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    for alias in node.names:
                        self.imports.append((module, f"{node.module}:{alias.name}", node.lineno))

    @staticmethod
    def _bindings(body: Any) -> dict[str, tuple[str, str, str | None]]:
        out: dict[str, tuple[str, str, str | None]] = {}
        nodes = body if isinstance(body, list) else list(ast.walk(body))
        for node in nodes:
            if isinstance(node, ast.ImportFrom) and node.module:
                for alias in node.names:
                    out[alias.asname or alias.name] = ("from", node.module, alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    out[alias.asname or alias.name.split(".")[0]] = (
                        "module", alias.name if alias.asname else alias.name.split(".")[0], None)
        return out

    def _walk(self, module: str, name: str, fn: Any, top: dict[str, Any]) -> None:
        names = {**top, **self._bindings(fn)}
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            f, callee = node.func, None
            if isinstance(f, ast.Name):
                bound = names.get(f.id)
                if bound and bound[0] == "from":
                    callee = f"{bound[1]}:{bound[2]}"
                elif bound is None:
                    callee = f"{module}:{f.id}"
            elif isinstance(f, ast.Attribute):
                parts, v = [], f.value
                while isinstance(v, ast.Attribute):
                    parts.append(v.attr)
                    v = v.value
                bound = names.get(v.id) if isinstance(v, ast.Name) else None
                if bound and bound[0] == "module":
                    callee = ".".join([bound[1], *reversed(parts)]) + f":{f.attr}"
                elif bound and bound[0] == "from":
                    callee = ".".join([f"{bound[1]}.{bound[2]}", *reversed(parts)]) + f":{f.attr}"
            if callee:
                self.calls.append((f"{module}:{name}", callee, node.lineno))


def _repo_sources() -> list[tuple[str, str]]:
    out = []
    for base in ("turbotab/core", "turbotab/server"):
        for path in sorted((ROOT / base).rglob("*.py")):
            if "/tests/" in str(path):
                continue
            module = ".".join(path.relative_to(ROOT).with_suffix("").parts)
            out.append((module.removesuffix(".__init__"), path.read_text("utf-8")))
    return out


def _qualified(fn: Any) -> str:
    return f"{fn.__module__}:{fn.__qualname__}"


def settlers(sites: _Sites) -> list[tuple[str, str, str, int]]:
    """Where a module outside ``readings.py`` decides a registry kind: a call to (or an import
    of) a kind's value test, its table form, or a helper its test reads, other than inside a helper
    of a kind itself or at a call the registry lists as evidence only."""
    tests: set[str] = set()
    helpers: set[str] = set()
    for rule in R.KIND_RULES.values():
        tests |= {_qualified(fn) for fn in (rule.test, rule.table) if fn is not None}
        helpers |= set(rule.helpers)
    out = []
    for caller, callee, line in sites.calls:
        if caller.split(":")[0] == LEDGER:
            continue
        if callee in tests:
            out.append(("calls a kind's value test", caller, callee, line))
        elif callee in helpers and caller not in helpers \
                and (caller, callee) not in R.EVIDENCE_ONLY:
            out.append(("calls a kind's helper", caller, callee, line))
    for module, name, line in sites.imports:
        if module != LEDGER and name in tests:
            out.append(("imports a kind's value test", module, name, line))
    return out


def test_b1_no_module_outside_the_ledger_decides_a_registry_kind():
    """BLUEPRINT §14.3 (quoted above: a kind is settled by its values only through the test that
    rejects its alternatives), and the sixth gate's ``_macros_in_grams_by_values``: a consumer
    settled `unit:factor` by its own, weaker Atwater test. Every consumer settles a kind only
    through ``readings.by_values`` / ``by_values_table`` (the registry's ``KIND_RULES``): no module
    but ``readings.py`` calls or imports a kind's value test, or a helper a test reads its values
    with (the recognizers' nutrient, energy and flag checks, the Atwater check, the text-number
    parser, the nesting reader), except inside those helpers and at the calls the registry lists as
    evidence only."""
    found = settlers(_Sites(_repo_sources()))
    assert found == [], "\n".join(map(str, found))
    assert not hasattr(R, "_macros_in_grams_by_values")


def test_b1_every_kind_with_a_test_declares_the_helpers_it_reads():
    """The check is only as wide as the helpers each kind declares: the value-settled kinds whose
    tests read values through another module name those functions, and each named function
    exists."""
    import importlib

    for kind in ("role:exposure", "role:energy", "role:covariate", "role:flag", "code_or_count",
                 "unit:energy", "unit:factor", "nested_in"):
        assert R.KIND_RULES[kind].helpers, kind
    for rule in R.KIND_RULES.values():
        for helper in rule.helpers:
            module, name = helper.split(":")
            assert callable(getattr(importlib.import_module(module), name)), helper


def test_b1_the_evidence_only_calls_exist_and_change_no_number():
    """Each call the registry lists as evidence only is a call the code makes, and its caller is a
    consumer the census registers as changing no number."""
    sites = _Sites(_repo_sources())
    calls = {(caller, callee) for caller, callee, _ in sites.calls}
    by_where = {c.where: c for c in R.CONSUMERS}
    for (caller, helper), why in R.EVIDENCE_ONLY.items():
        assert (caller, helper) in calls, (caller, helper)
        assert why and caller in by_where and not by_where[caller].changes, caller


def test_b1_the_structural_check_can_fail():
    """Positive controls: a consumer calling a kind's test directly, one calling the Atwater check
    or a recognizer's intake check, one reaching a flag check through a module path, and one
    importing a test are each caught; a call through ``readings.by_values`` is not."""
    bad = [("x.consumer", "from turbotab.core.readings import names_rows\n"
                          "def f(v):\n    return names_rows(v)\n"),
           ("y.consumer", "from turbotab.core.methods.energy import atwater_check\n"
                          "def g(frame, e):\n    return atwater_check(frame, e).verdict == 'pass'\n"),
           ("z.consumer", "from turbotab.core import recognizers as rr\n"
                          "def h(a, b):\n    return rr.intake_check(a, b).by_values\n"),
           ("w.consumer", "import turbotab.core.recognizers\n"
                          "def k(a):\n    return turbotab.core.recognizers.flag_values(a)\n"),
           ("ok.consumer", "from turbotab.core.readings import by_values\n"
                           "def m(v):\n    return by_values('code_or_count', v).settles\n")]
    found = settlers(_Sites(bad))
    assert {(f[1], f[2]) for f in found if f[0] != "imports a kind's value test"} == {
        ("x.consumer:f", "turbotab.core.readings:names_rows"),
        ("y.consumer:g", "turbotab.core.methods.energy:atwater_check"),
        ("z.consumer:h", "turbotab.core.recognizers:intake_check"),
        ("w.consumer:k", "turbotab.core.recognizers:flag_values")}
    assert ("imports a kind's value test", "x.consumer", "turbotab.core.readings:names_rows", 1) \
        in found
    assert not any(f[1].startswith("ok.consumer") for f in found)


def test_b2_by_values_is_the_registry_test_and_a_user_settled_kind_has_none():
    """``by_values`` runs exactly the registry's test (a kind's verdict is one function), and a
    kind with no value test (the nesting, the detection limit, a flag) cannot be settled by values
    at all."""
    values = pd.Series([1.25, 2.5, 3.75] * 10)
    assert R.by_values("code_or_count", values) == R.KIND_RULES["code_or_count"].test(values)
    for kind in ("nested_in", "detection_limit", "role:flag", "role:time"):
        with pytest.raises(ValueError):
            R.by_values(kind, values)


# ═════════════════════════════════════════════════════════════════════════════
# C · column-valued confirmations: every eligible column
# ═════════════════════════════════════════════════════════════════════════════


def test_c1_a_column_valued_kind_accepts_every_eligible_column_and_honors_each():
    """The sixth gate: the property test's oracle encoded 'reading_unsettled' for every total but
    the one the design found. A column-valued kind (``readings.COLUMN_VALUED``: ``nested_in``)
    accepts every column of the table but the column itself and the outcome, and no total; the
    exhaustive property test (``test_ledger_repair_2`` section D) enumerates exactly that domain for
    each consumer of the kind, each against its own NumPy oracle."""
    from turbotab.core.tests.acceptance import test_ledger_repair_2 as lr2

    assert R.COLUMN_VALUED == ("nested_in",)
    columns = ["sfa_g", "fat_g", "carbohydrate_g", "protein_g", "energy_kcal", "age", "y"]
    want = {"fat_g", "carbohydrate_g", "protein_g", "energy_kcal", "age", R.NOT_NESTED}
    accepted = set()
    for value in [*columns, R.NOT_NESTED, "absent"]:
        try:
            d.validate({"kind": "confirm_reading", "reading": "nested_in", "column": "sfa_g",
                        "value": value}, {"columns": columns, "target": "y"})
        except d.Refusal:
            continue
        accepted.add(value)
    assert accepted == want == set(R.nesting_parents("sfa_g", columns, "y"))
    assert set(lr2.domain("nested_in")) == set(R.nesting_parents("sfa_g", lr2.NESTED_COLUMNS, "y"))
    consumers = {k for k in lr2.PROBES if k[1] == "nested_in"}
    assert consumers == {(c.where, "nested_in") for c in R.CONSUMERS
                         if c.changes and "nested_in" in c.kinds}
    assert len(consumers) == 3


def test_c2_a_part_cannot_be_confirmed_inside_its_own_part():
    """A cycle is refused with its reason: once `sfa_g` is part of `fat_g`, `fat_g` cannot be
    confirmed part of `sfa_g`."""
    state = d.fold([d.DecisionRecord(id="r1", seq=1, at="2026-10-04T00:00:00Z",
                                     decision=d.ConfirmReading(reading="nested_in", column="sfa_g",
                                                               value="fat_g"))])
    with pytest.raises(d.Refusal) as refused:
        d.validate({"kind": "confirm_reading", "reading": "nested_in", "column": "fat_g",
                    "value": "sfa_g"}, {"state": state, "columns": ["sfa_g", "fat_g", "y"],
                                         "target": "y"})
    assert refused.value.code == "part_of_the_other"


# ═════════════════════════════════════════════════════════════════════════════
# D · settlement is visible
# ═════════════════════════════════════════════════════════════════════════════


def test_d1_grams_read_by_the_identity_are_listed_and_alcohol_is_not(tmp_path):
    """The gate: "read_from_data has no unit:factor item". On q2's table, once total energy is the
    user's, the card lists protein, fat and carbohydrate in grams by the Atwater identity, each with
    its evidence (the ratios that reject kg, kcal and kJ) and its other units as the change; it
    never lists alcohol, whose unit is the user's."""
    plan = answers("sbp", ["dietary"])
    roles = {"participant_id": "identifier", "age": "covariate", "protein_g": "exposure",
             "fat_g": "exposure", "carbohydrate_g": "exposure", "alcohol": "exposure",
             "energy_kcal": "energy"}
    truth = Truth({"code_or_count:age": "amount", "unit:energy_kcal": "kcal",
                   "day_count:energy_kcal": "1", "code_or_count:energy_kcal": "amount",
                   **DRINKS_CAUSAL}, fixture="q2 visible")
    with local_server(tmp_path / "home") as client:
        drive = open_project(client, write(g6.alcohol_substitution_table(), tmp_path, "v.csv"),
                             truth)
        drive_unsettled(drive, plan, roles=roles, stop_before="__none__")
        drive.artifact("fit", timeout=600)
        card = client.get(f"/api/projects/{drive.pid}/readings").json()
        record = next(r for r in drive.view()["decisions"]
                      if r["decision"]["kind"] == "select_models")
    units = {i["column"]: i for i in card["read_from_data"] if i["kind"] == "unit"}
    for column in ("protein_g", "fat_g", "carbohydrate_g"):
        assert units[column]["value"] == "g", units
        assert "fails were it in kg" in units[column]["evidence"]
        assert {c["decision"]["value"] for c in units[column]["change"]} >= {"kg", "kcal", "kj"}
    assert "alcohol" not in units
    assert "`carbohydrate_g` is in grams" in (record.get("sentence") or ""), record.get("sentence")


def test_d2_the_settled_by_values_answer_flows_into_truth_driven_asks():
    """The driver answers the detection-limit question from the truth it declares, never a
    constant, and names an undeclared one."""
    exits = R.detection_exits("crp", {"below": {"0.20": 15}, "decimal": ".", "thousands": False})
    error = {"exits": exits}
    chosen = truth_answers(error, Truth({"detection_limit:crp": "limit_root2",
                                         "code_or_count:crp": "amount"}, fixture="t"))
    assert [x for x in chosen if x.get("kind") == "apply_repair"][0]["option"] == "limit_root2"
    with pytest.raises(AssertionError):
        truth_answers(error, Truth({"code_or_count:crp": "amount"}, fixture="t"))
