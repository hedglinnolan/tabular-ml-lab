"""The readings ledger (BLUEPRINT §14.1, 2026-10-03): the leash as architecture.

Recognition's leash (§14) says a recognizer may be wrong, but a wrong recognition may never
silently change a number. The third intelligence gate showed it cannot be enforced reader by
reader: after the roles were leashed, seven other heuristics still turned a name or a value into a
number-changing default without asking. So every interpretation the engine makes about the data is
a *reading* (``turbotab.core.readings``), a reading is *settled* only when its values corroborate it
at high confidence or the user confirmed it (one reading per confirmation), and every
number-changing consumer reads only settled readings or asks.

Section A replays the gate's seven failures (``leash_gate3.json``, field ``gate``) draw for draw
from the verifier's generators (``gate-intelligence-3/*.py``, seeds 31001–31022), through the real
server where the gate drove it. Each asserts the leash's criterion: the number the misreading
would have changed does not change unasked; the reading is asked, one exit per reading, and once
the user answers, the number follows the answer (NumPy/pandas references).

Section B checks the ledger's mechanics: the settled rule, the one confirmation decision (old
``confirm_role`` records stay valid), and what a bulk roles answer carries.

Section C is the structural check the gate runs: the independent census of readers and consumers
(``readings_census.json``, verbatim) against the ledger's registry of consumers, and an AST
call-graph check that every consumer that can change a number reads through the ledger.

Expected values never come from the code under test: NumPy/pandas on the fixtures, and quoted
primary sources (fetched 2026-10-03).
"""
from __future__ import annotations

import ast
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from turbotab.core import decisions as d
from turbotab.core.decisions import GrainSpec, ProjectState
from turbotab.core.tests.stage_harness import Ingested

ROOT = Path(__file__).resolve().parents[4]
CENSUS = Path(__file__).with_name("readings_census.json")

# ── primary sources, quoted ──────────────────────────────────────────────────

BLUEPRINT_14_1 = ("A reading is *settled* when its values corroborate it at high confidence or the "
                  "user confirmed it.")
# NHANES 2017–2018 Body Measures codebook (wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/
# BMX_J.htm), the variable list:
NHANES_BMX = {"BMXWT": "BMXWT - Weight (kg)", "BMXHT": "BMXHT - Standing Height (cm)",
              "BMXBMI": "BMXBMI - Body Mass Index (kg/m**2)"}
# MEPS HC-216 (2019 Full Year Consolidated File) documentation (meps.ahrq.gov/data_stats/
# download_data/pufs/h216/h216doc.shtml), §3.10.1 and the weights variable list:
MEPS_DESIGN = ("The variables VARSTR and VARPSU on this MEPS data file serve to identify the "
               "sampling strata and primary sampling units required by the variance estimation "
               "programs.")
MEPS_VARIABLES = {"PERWT19F": "Final Person Weight, 2019", "VARSTR": "Variance Estimation Stratum - 2019",
                  "VARPSU": "Variance Estimation PSU - 2019"}
MEPS_PSUS = "There are 165 variance strata with either two or three variance estimation PSUs per stratum"
# NDNS RP Years 9–11 User Guide (UK Data Service SN 6533), Table 6.1 and §6.2:
NDNS_WEIGHT = "wti_Y911 Weight for non-response by individuals to the individual questionnaire and diary"
NDNS_STRATA = "There are 5 strata altogether (called astrata1h to astrata5 in the archived dataset)"
NDNS_PSU = ("the addresses were clustered into Primary Sampling Units (PSUs), small geographical "
            "areas, based on postcode sectors")
# NUTRITION_PACK §01 (docs/turbotab-next/reference/research/NUTRITION_PACK.md:41), the one-day prior the day-count
# reading consults:
NUTRITION_PRIOR = "Median-magnitude plausibility priors** (adults/day) as a second signal: energy 1,600–2,600 kcal"
# CLINICAL_SURVEY_PACK §A1.1 (docs/turbotab-next/reference/research/CLINICAL_SURVEY_PACK.md:42-43):
CLINICAL_UNITS = "Please confirm units per analyte against the source data dictionary — TurboTab"


def test_the_quoted_sources_are_the_repositorys_own_where_they_live_there():
    """The packs' sentences these tests rest on are quoted as the repository holds them."""
    pack = (ROOT / "docs/turbotab-next/reference/research/NUTRITION_PACK.md").read_text("utf-8")
    assert NUTRITION_PRIOR in pack
    clinical = (ROOT / "docs/turbotab-next/reference/research/CLINICAL_SURVEY_PACK.md").read_text("utf-8")
    assert CLINICAL_UNITS in " ".join(clinical.replace(">", " ").split())
    blueprint = (ROOT / "docs/turbotab-next/BLUEPRINT.md").read_text("utf-8")
    assert BLUEPRINT_14_1 in " ".join(blueprint.split())


# ═════════════════════════════════════════════════════════════════════════════
# The verifier's generators (gate-intelligence-3), draw for draw
# ═════════════════════════════════════════════════════════════════════════════


def _adults(rng: np.random.Generator, n: int) -> dict[str, Any]:
    """probe.py ``adults``: energy from fat-free mass × PAL, reported with error; macronutrient and
    alcohol shares drawn per person."""
    sex = rng.choice(["F", "M"], n)
    male = sex == "M"
    age = rng.integers(25, 70, n)
    height = np.where(male, rng.normal(176, 7, n), rng.normal(163, 6.5, n))
    pbf = np.where(male, rng.normal(24, 6, n), rng.normal(34, 6, n)).clip(8, 55)
    weight = np.where(male, rng.normal(84, 14, n), rng.normal(70, 14, n)).clip(45, 160)
    ffm = weight * (1 - pbf / 100)
    eer = (500 + 22 * ffm) * rng.uniform(1.4, 1.9, n)
    kcal = eer * np.exp(rng.normal(-0.1, 0.25, n))
    p = rng.normal(0.16, 0.03, n).clip(0.08, 0.3)
    f = rng.normal(0.34, 0.06, n).clip(0.15, 0.55)
    a = rng.gamma(0.6, 0.03, n).clip(0, 0.2) * (rng.random(n) < 0.6)
    c = (1 - p - f - a).clip(0.2, 0.7)
    return dict(sex=sex, male=male, age=age, height=height.round(1), weight=weight.round(1),
                pbf=pbf, ffm=ffm, kcal=kcal, protein_g=kcal * p / 4, fat_g=kcal * f / 9,
                carb_g=kcal * c / 4, alc_g=kcal * a / 7)


def _day_totals(rng: np.random.Generator, days: int, e_name: str, suffix: str,
                n: int = 500) -> pd.DataFrame:
    """b2_days.py ``make``: each day's intake varies around the person's usual; the energy column
    and the macronutrients are totals over ``days`` days."""
    a = _adults(rng, n)
    tot = {k: np.zeros(n) for k in ("p", "f", "c", "a")}
    for _ in range(days):
        jitter = np.exp(rng.normal(0, 0.25, n))
        tot["p"] += a["protein_g"] * jitter
        tot["f"] += a["fat_g"] * jitter * np.exp(rng.normal(0, 0.1, n))
        tot["c"] += a["carb_g"] * jitter
        tot["a"] += a["alc_g"] * jitter
    kcal = 4 * tot["p"] + 9 * tot["f"] + 4 * tot["c"] + 7 * tot["a"]
    return pd.DataFrame({
        "participant_id": np.arange(1, n + 1), "sex": a["sex"], "age": a["age"],
        "weight_kg": a["weight"], "height_cm": a["height"],
        e_name: kcal.round(0), f"protein{suffix}": tot["p"].round(1),
        f"fat{suffix}": tot["f"].round(1), f"carbohydrate{suffix}": tot["c"].round(1),
        f"alcohol{suffix}": tot["a"].round(1), "ldl": rng.normal(3.2, 0.8, n).round(2)})


DAY_CASES = [  # b2_days.py ``cases``, in its order (each draws from the one generator)
    (2, "kcal_d1d2", "_d1d2"), (2, "kcal_sum_d1_d2", "_sum_d1_d2"),
    (2, "kcal_both_days", "_both_days"), (2, "kcal_2rec", "_2rec"),
    (2, "energy_kcal_2x24h", "_2x24h"), (3, "energy_3dfr", "_3dfr"), (2, "kcal_2dr", "_2dr"),
    (7, "kcal_7dd", "_7dd"), (2, "Energy (kcal) - 2 recalls", " (g) - 2 recalls"),
    (2, "TEI_2days", "_2days"),
]


def _days_cases() -> dict[str, tuple[int, pd.DataFrame]]:
    rng = np.random.default_rng(31002)
    return {e: (days, _day_totals(rng, days, e, suf)) for days, e, suf in DAY_CASES}


def _days_server() -> pd.DataFrame:
    """s1_days.py: ``energy_kcal_day1_day2`` (2-day totals, rng 31022), every energy-bearing
    column scaled by 0.8."""
    frame = _day_totals(np.random.default_rng(31022), 2, "energy_kcal_day1_day2", "_day1_day2")
    for c in frame.columns:
        if c == "energy_kcal_day1_day2" or c.startswith(("protein", "fat", "carbohydrate",
                                                          "alcohol")):
            frame[c] = (frame[c] * 0.8).round(1)
    return frame


def _strata(col: str = "stratum_id", k: int = 16, n: int = 480, seed: int = 31006) -> pd.DataFrame:
    """s2_strata.py ``frame_of``: a stratified two-arm trial, one row per participant."""
    rng = np.random.default_rng(seed)
    stratum = rng.integers(1, k + 1, n)
    arm = np.where(rng.random(n) < 0.5, "intervention", "control")
    age = rng.integers(30, 70, n)
    base = rng.normal(140, 12, n)
    shift = rng.normal(0, 4, k + 1)[stratum]
    sbp6 = base - 5 * (arm == "intervention") + shift + rng.normal(0, 10, n)
    return pd.DataFrame({"participant_id": np.arange(5001, 5001 + n), col: stratum, "arm": arm,
                         "age": age, "sbp_baseline": base.round(0), "sbp_6m": sbp6.round(0)})


def _id_traps() -> pd.DataFrame:
    """b8_ids.py (rng 31012): interviewer, randomization-block, enumeration-area and recruiter
    codes beside a unique participant id."""
    rng = np.random.default_rng(31012)
    n = 600
    return pd.DataFrame({"participant_id": np.arange(1, n + 1),
                         "interviewer_id": rng.integers(1, 25, n),
                         "rand_block_id": rng.integers(1, 40, n),
                         "EA_ID": rng.integers(1, 80, n),
                         "recruiter_id": rng.integers(1, 15, n),
                         "age": rng.integers(20, 70, n), "sbp": rng.normal(130, 15, n).round(0)})


def _hchs() -> pd.DataFrame:
    """s4_hchs.py (rng 31008): HCHS/SOL's published design names, PSU_ID nested in STRAT."""
    rng = np.random.default_rng(31008)
    n = 800
    psu = rng.integers(1, 120, n)
    frame = pd.DataFrame({
        "ID": rng.choice(np.arange(100000, 999999), n, replace=False),
        "PSU_ID": psu * 10 + rng.integers(1, 3, n), "STRAT": (psu // 4 + 1),
        "WEIGHT_FINAL_NORM_OVERALL": rng.lognormal(0, 0.6, n).round(4),
        "CENTER": rng.choice(["B", "C", "M", "S"], n), "AGE": rng.integers(18, 75, n),
        "GENDER": rng.choice(["F", "M"], n), "BMI": rng.normal(29, 5, n).round(1)})
    frame["HBA1C"] = (5.2 + 0.03 * (frame.BMI - 29) + rng.normal(0, 0.1, 120)[psu - 1]
                      + rng.normal(0, .6, n)).round(2)
    return frame


def _women_lb() -> pd.DataFrame:
    """b6b_goldberg.py (rng 31010): a US women's cohort, self-reported weight in lb and height in
    inches (the gate's ``gold_women_lb.csv``)."""
    rng = np.random.default_rng(31010)
    n = 500
    age = rng.integers(20, 60, n)
    w_kg = rng.normal(70, 13, n).clip(42, 140)
    h_cm = rng.normal(163, 6.5, n)
    pal = rng.uniform(1.5, 1.9, n)
    kcal = (8.126 * w_kg + 845.6) * pal * np.exp(rng.normal(-0.08, 0.2, n))
    p = rng.normal(.16, .03, n).clip(.08, .3)
    fsh = rng.normal(.34, .05, n).clip(.15, .5)
    c = 1 - p - fsh
    f = pd.DataFrame({"pid": np.arange(1, n + 1), "sex": "F", "age": age,
                      "weight": (w_kg * 2.20462).round(1), "height": (h_cm / 2.54).round(1),
                      "energy_kcal": kcal.round(0), "protein_g": (kcal * p / 4).round(1),
                      "fat_g": (kcal * fsh / 9).round(1), "carbohydrate_g": (kcal * c / 4).round(1),
                      "ldl": rng.normal(3.2, .8, n).round(2)})
    f.loc[:9, "sex"] = "M"
    return f


OCCASIONS = {  # b4_repeats.py ``cases``, in its order
    "month": (0, 6), "time": ("baseline", "6 months"), "trimester": ("T1", "T3"),
    "stage": ("pre", "post"), "fu": (0, 1), "year": (2018, 2020), "timepoint": ("BL", "M6"),
    "assessment": ("baseline", "follow-up"), "followup_month": (0, 6),
}


def _trial(rng: np.random.Generator, occ_name: str, occ_values: tuple, n: int = 120,
           server: bool = False) -> pd.DataFrame:
    """b4_repeats.py / s3_occasion.py ``trial``: two recalls at baseline and two at follow-up,
    numbered 1–4; the intervention arm's intake falls about 22% at follow-up."""
    rows = []
    for u in range(n):
        arm = "intervention" if u % 2 else "control"
        usual = rng.normal(2100, 350)
        if server:
            age = int(rng.integers(30, 65))
            sex = rng.choice(["F", "M"])
            y = rng.normal(130, 12)
        for k in range(4):
            fu = k >= 2
            mult = 0.78 if (fu and arm == "intervention") else 1.0
            kcal = usual * mult * np.exp(rng.normal(0, 0.22))
            row = {"participant_id": 1000 + u, "arm": arm}
            if server:
                row.update(age=age, sex=sex)
            row.update({"recall_no": k + 1, occ_name: occ_values[1] if fu else occ_values[0],
                        "energy_kcal": round(kcal), "protein_g": round(kcal * 0.16 / 4, 1),
                        "fat_g": round(kcal * 0.34 / 9, 1),
                        "carbohydrate_g": round(kcal * 0.50 / 4, 1)})
            if server:
                row["sbp"] = round(y - (4 if (fu and arm == "intervention") else 0)
                                   + rng.normal(0, 5))
            rows.append(row)
    return pd.DataFrame(rows)


def _unit_tables() -> dict[str, pd.DataFrame]:
    """b9_units.py (rng 31013): four outcomes whose names end in a bare amount."""
    rng = np.random.default_rng(31013)
    n = 300
    cases = [("glucose_mg", rng.normal(98, 12, n).round(0)), ("bmi_kg", rng.normal(27, 4, n).round(1)),
             ("hb_g", rng.normal(13.8, 1.3, n).round(1)), ("ldl_mg", rng.normal(118, 30, n).round(0))]
    return {name: pd.DataFrame({"pid": np.arange(n), "age": rng.integers(20, 70, n), name: vals})
            for name, vals in cases}


def _ldl_mg() -> pd.DataFrame:
    """s6_units.py (rng 31014): ``ldl_mg`` as the outcome of age and statin use."""
    rng = np.random.default_rng(31014)
    n = 400
    age = rng.integers(25, 75, n)
    f = pd.DataFrame({"pid": np.arange(1, n + 1), "age": age, "sex": rng.choice(["F", "M"], n),
                      "statin": rng.integers(0, 2, n)})
    f["ldl_mg"] = (100 + 0.6 * (age - 50) - 30 * f.statin + rng.normal(0, 25, n)).round(0)
    return f


def _recalls_with_counts() -> pd.DataFrame:
    """s7_combine.py (rng 31017): two 24-hour recalls per person, with per-recall counts."""
    rng = np.random.default_rng(31017)
    rows = []
    for u in range(150):
        usual = rng.normal(2000, 350)
        cof = rng.uniform(0, 4)
        occ = rng.uniform(3, 6)
        y = rng.normal(5.5, .6)
        for k in range(2):
            kc = usual * np.exp(rng.normal(0, .22))
            rows.append({"participant_id": 2000 + u, "recall_no": k + 1, "energy_kcal": round(kc),
                         "protein_g": round(kc * .16 / 4, 1), "fat_g": round(kc * .34 / 9, 1),
                         "carbohydrate_g": round(kc * .5 / 4, 1),
                         "coffee_cups": int(rng.poisson(cof)),
                         "eating_occasions": int(np.clip(rng.poisson(occ), 1, 9)),
                         "hba1c": round(y, 2)})
    return pd.DataFrame(rows)


def _ndns() -> pd.DataFrame:
    """b1_roles.py table A (rng 31001): NDNS RP person-level names (the UK Data Service user
    guide's), with the individual, nurse and blood weights, two strata and the area (PSU)."""
    rng = np.random.default_rng(31001)
    n = 600
    a = _adults(rng, n)
    serialh = rng.choice(np.arange(10000000, 10000000 + 380), n)
    seriali = serialh * 10 + rng.integers(1, 3, n)
    carb375 = a["carb_g"] * 4 / 3.75
    kcal = a["protein_g"] * 4 + a["fat_g"] * 9 + carb375 * 3.75 + a["alc_g"] * 7
    nurse = rng.random(n) < 0.7
    blood = nurse & (rng.random(n) < 0.6)
    frame = pd.DataFrame({
        "seriali": seriali, "serialh": serialh, "Sex": np.where(a["male"], 1, 2), "age": a["age"],
        "Energykcal": kcal.round(1),
        "EnergykJ": (a["protein_g"] * 17 + a["fat_g"] * 37 + carb375 * 16 + a["alc_g"] * 29).round(1),
        "Protein": a["protein_g"].round(1), "Fat": a["fat_g"].round(1),
        "Carbohydrate": carb375.round(1), "Alcohol": a["alc_g"].round(1),
        "Totalsugars": (carb375 * rng.uniform(0.25, 0.5, n)).round(1),
        "AOACFibre": (kcal / 1000 * rng.normal(9, 2, n)).clip(3).round(1),
        "Sodium": (kcal * rng.normal(1.3, 0.2, n)).round(0),
        "wti_Y911": rng.lognormal(0, 0.4, n).round(4),
        "wtn_Y911": np.where(nurse, rng.lognormal(0, 0.45, n), np.nan).round(4),
        "wtb_Y911": np.where(blood, rng.lognormal(0, 0.5, n), np.nan).round(4),
        "astrata1": rng.integers(1, 40, n), "astrata2": rng.integers(1, 20, n),
        "area": rng.integers(1, 160, n),
        "Glucose": np.where(blood, rng.normal(5.2, 0.6, n), np.nan).round(2)})
    return frame.drop_duplicates("seriali")


def _meps() -> pd.DataFrame:
    """A MEPS 2019 person-level extract, built here (the gate checked MEPS on the state alone):
    PERWT19F a positive person weight with decimals, VARSTR the variance strata numbered from
    1001, VARPSU two or three PSUs numbered within each stratum (MEPS_PSUS), and a BMI outcome."""
    rng = np.random.default_rng(2019)
    strata = np.arange(1001, 1061)
    per = rng.choice([2, 3], len(strata))
    cells = [(s, p) for s, k in zip(strata, per) for p in range(1, k + 1)]
    n = 900
    pick = rng.integers(0, len(cells), n)
    varstr = np.array([cells[i][0] for i in pick])
    varpsu = np.array([cells[i][1] for i in pick])
    age = rng.integers(18, 85, n)
    return pd.DataFrame({
        "DUPERSID": np.arange(2_460_001, 2_460_001 + n), "VARSTR": varstr, "VARPSU": varpsu,
        "PERWT19F": (rng.lognormal(9.0, 0.7, n)).round(6), "AGE19X": age,
        "SEX": rng.choice([1, 2], n), "BMINDX53": (24 + 0.05 * age + rng.normal(0, 4, n)).round(1)})


# ── helpers ──────────────────────────────────────────────────────────────────


def ADULTS_CAUSAL(outcome: str, suffix: str, weight: str, height: str) -> dict[str, str]:  # noqa: N802
    """WP17: the causal answers for a table drawn by :func:`_adults` with an outcome drawn apart
    (``ldl``), protein (``protein<suffix>``) the exposure. Sex and weight set energy needs and so
    the protein eaten; age and height move nothing; the other macronutrients share total energy
    with protein, the field's default (other dietary components are possible confounders)."""
    return {f"exposure:{outcome}": f"protein{suffix}", "adjust:sex": "yes,no,no",
            f"adjust:{weight}": "yes,no,no", "adjust:age": "no,no,no",
            f"adjust:{height}": "no,no,no",
            **{f"adjust:{n}{suffix}": "unknown,unknown,no" for n in ("fat", "carbohydrate",
                                                                    "alcohol")}}


def _write(frame: pd.DataFrame, folder: Path, name: str) -> Path:
    path = Path(folder) / name
    frame.to_csv(path, index=False)
    return path


def _table(path: Path) -> Ingested:
    return Ingested(path, Path(tempfile.mkdtemp()))


def _roles(path: Path, *, lens, target=None, purpose="inference", grain=None) -> dict:
    from turbotab.core.stages.rows import roles_stage

    return _table(path).run(roles_stage, ProjectState(lens=lens, target=target, purpose=purpose,
                                                      grain=grain))


def _bulk(artifact: dict, **slots: Any) -> ProjectState:
    """The state a bulk "confirm the roles of N columns" leaves: every proposal recorded as
    proposed, the ones below high recorded unconfirmed (BLUEPRINT §14 rule 2)."""
    roles = {p["column"]: p["proposed"] for p in artifact["columns"]}
    waiting = [p["column"] for p in artifact["columns"] if p["confidence"] != "high"]
    return ProjectState(roles=roles, roles_unconfirmed=waiting, **slots)


def _strings(x: Any) -> list[str]:
    out: list[str] = []

    def walk(v: Any) -> None:
        if isinstance(v, dict):
            for w in v.values():
                walk(w)
        elif isinstance(v, list):
            for w in v:
                walk(w)
        elif isinstance(v, str):
            out.append(v)

    walk(x)
    return out


ORDER = ["lens", "orientation", "target", "event", "task", "purpose", "grain", "repeat_kind",
         "unit", "aggregation", "temporal", "roles", "survey", "exclusions", "missing", "split",
         "energy_adjustment", "models"]


def _drive(drive, *, lens, target, grain, stop_before=None, answers=None,
           confirm_roles=True, purpose="inference") -> None:
    """Answer the Router in order. The roles are recorded exactly as proposed (the roles card's
    one-click confirm); with ``confirm_roles`` each one that rode along is then confirmed on its
    own, as a user who knows the table does, so a test can isolate the reading it is about."""
    base = {
        "lens": {"kind": "set_lens", "lenses": lens},
        "orientation": {"kind": "set_orientation", "orientation": "sample_major"},
        "target": {"kind": "set_target", "column": target},
        "task": {"kind": "set_task", "column": target, "task": "regression"},
        "purpose": {"kind": "set_purpose", "purpose": purpose},
        "grain": grain, "temporal": {"kind": "set_temporal", "temporal": False},
        "survey": {"kind": "set_survey", "estimand": "sample"},
        "exclusions": {"kind": "set_exclusions", "rules": []},
        "missing": {"kind": "set_missing", "strategy": "complete_case"},
        "split": {"kind": "set_split", "holdout": 0.2, "seed": 0, "folds": 5},
        "energy_adjustment": {"kind": "set_energy_adjustment", "method": "none"},
        "models": {"kind": "select_models", "models": ["linear"]},
        "unit": {"kind": "set_unit", "unit": "unit"},
        "aggregation": {"kind": "set_aggregation", "method": "mean"},
        **(answers or {}),
    }
    for key in ORDER:
        if key == stop_before:
            return
        if key in ("event", "task"):
            drive.artifact("target_info")  # the outcome's reading decides whether it is asked
        step = drive.reach(key, timeout=300)
        if step["status"] not in ("open", "waiting"):
            continue
        if key == "roles":
            proposed = {c["column"]: c["proposed"] for c in drive.artifact("roles")["columns"]}
            if confirm_roles:
                drive.decide_roles(proposed)
            else:
                drive.decide({"kind": "set_roles", "roles": proposed})
            continue
        drive.decide(base[key])


def _refused(drive, body: dict) -> dict:
    """The refusal ``body`` meets, once the Router asks it (a recorded unit recomputes the findings
    and the stages behind them; the question waits for them, ``not_yet``, as a client does)."""
    from turbotab.core.tests.acceptance.server_drive import _post_when_reached

    # Under inference the exposure and the adjustment set come first (WP17), answered from the
    # fixture's truth as the questions before any other are.
    r = _post_when_reached(drive.c, f"/api/projects/{drive.pid}/decisions", body,
                           unblock=lambda: drive.answer_wp17_before(body))
    assert r.status_code == 409, r.text
    return r.json()["error"]


def _one_reading_each(exits: list[dict]) -> list[dict]:
    """The exits' decisions, each confirming one reading of one column (never several)."""
    out = [x["decision"] for x in exits if x.get("decision")
           and x["decision"]["kind"] != "confirm_readings"]
    for decision in out:
        if decision["kind"] == "confirm_reading":
            assert set(decision) == {"kind", "reading", "column", "value"}, decision
    # BLUEPRINT §14.2: a block confirmation lists exactly the readings asked, each once.
    for x in exits:
        block = x.get("decision") or {}
        if block.get("kind") == "confirm_readings":
            listed = [(i["reading"], i["column"]) for i in block["items"]]
            assert len(listed) == len(set(listed)), listed
            assert set(listed) <= {(o["reading"], o["column"]) for o in out
                                   if o["kind"] == "confirm_reading"}, listed
    return out


# ═════════════════════════════════════════════════════════════════════════════
# A · The gate's seven failures, replayed
# ═════════════════════════════════════════════════════════════════════════════


def test_g1_a_repeating_id_code_clusters_no_interval_unasked_through_the_server(tmp_path):
    """Gate failure 1 (logs_s2a.out): in a stratified RCT, one row per participant (grain stated
    from ``participant_id``), randomization ``stratum_id`` (16 strata) was "identifier (high):
    Names each unit" and the intervals were "cluster-robust (CR2) by `stratum_id`, G = 16", unasked.

    Expected: ``stratum_id`` is proposed below high with its attention marker; the fit asks before
    it reads the role (the models answer is refused with one confirmation per column); once the user
    answers (a stratum is a covariate in a stratified trial), no interval clusters by it.
    Reference (pandas): 16 strata over 480 rows; ``participant_id`` unique."""
    from turbotab.core.tests.acceptance.server_drive import local_server, open_project

    from turbotab.core.tests.truths import Truth

    frame = _strata()
    assert frame["stratum_id"].nunique() == 16 and frame["participant_id"].is_unique
    path = _write(frame, tmp_path, "strata.csv")
    # The trial's truth (BLUEPRINT §14.3): the strata are codes, age and baseline SBP amounts.
    # WP17, the generator: the arm is randomized; the stratum's shift and the baseline SBP move
    # SBP at six months, age moves nothing.
    truth = Truth({"code_or_count:stratum_id": "code", "code_or_count:age": "amount",
                   "code_or_count:sbp_baseline": "amount", "exposure:sbp_6m": "arm",
                   "role:arm": "exposure",
                   "adjust:stratum_id": "no,yes,no", "adjust:sbp_baseline": "no,yes,no",
                   "adjust:age": "no,no,no"}, fixture="s2_strata")
    with local_server(tmp_path / "home") as client:
        drive = open_project(client, path, truth)
        _drive(drive, lens=["clinical"], target="sbp_6m", stop_before="models",
               grain={"kind": "set_grain", "grain": "one_row_per_unit",
                      "id_column": "participant_id"}, confirm_roles=False)
        roles = drive.artifact("roles")
        proposal = next(p for p in roles["columns"] if p["column"] == "stratum_id")
        assert (proposal["proposed"], proposal["confidence"]) == ("identifier", "medium"), proposal
        assert proposal["attention"] and "stratum_id" in roles["needs_confirmation"]
        assert "is asked" in proposal["reason"]
        error = _refused(drive, {"kind": "select_models", "models": ["linear"]})
        assert error["code"] == "reading_unsettled"
        asked = _one_reading_each(error["exits"])
        assert {"kind": "confirm_reading", "reading": "role", "column": "stratum_id",
                "value": "identifier"} in asked
        # The user's answer: the strata are adjusted for, not clustered by.
        truth["role:stratum_id"] = "covariate"
        mine = {p["column"]: p["proposed"] for p in roles["columns"]}
        mine["stratum_id"] = "covariate"
        drive.decide_roles(mine)
        drive.decide({"kind": "select_models", "models": ["linear"]})
        fit = drive.artifact("fit", timeout=600)
        info = fit["models"][0].get("inference") or {}
        assert "95%" in str(info.get("caption")), info.get("caption")  # intervals reported
        assert info.get("grouped_by") is None, info.get("grouped_by")
        clustered = [s for s in _strings(fit) if "cluster" in s.lower() and "stratum_id" in s]
        assert not clustered, clustered


def test_g1_repeating_codes_named_like_ids_are_asked_not_clustered_by(tmp_path):
    """Gate failure 1 at the stage level (logs_b8.out, logs_b1.out): ``interviewer_id``,
    ``rand_block_id``, ``EA_ID`` and ``recruiter_id`` were "identifier (high)", and with
    ``participant_id`` named the intervals clustered by ``recruiter_id`` (14 clusters); HCHS/SOL's
    ``PSU_ID`` likewise. Expected: none is high; after a bulk confirm the intervals ask (one
    confirmation per reading: the role, or whether its rows belong together), never cluster
    unasked; the user's "yes" clusters by it (pandas: 14 recruiters) and "no" by nothing."""
    from turbotab.core.models.inference import cluster_columns, resolve_clusters

    frame = _id_traps()
    codes = ("interviewer_id", "rand_block_id", "EA_ID", "recruiter_id")
    counts = {c: int(frame[c].nunique()) for c in codes}
    assert all(10 < k < len(frame) for k in counts.values()), counts  # repeating, > FEW_UNITS
    path = _write(frame, tmp_path, "ids.csv")
    art = _roles(path, lens=["clinical"], target="sbp")
    by = {p["column"]: p for p in art["columns"]}
    for c in codes:
        assert by[c]["confidence"] != "high" and by[c]["attention"], by[c]
    rows = frame.set_index(pd.Index(np.arange(len(frame)), name="row_id"))
    # BLUEPRINT §14.3: the grain answer always wins. Naming ``participant_id`` as the unit, it
    # settles that rows are independent units, and the repeating codes cluster nothing unasked
    # (said in the note).
    named = GrainSpec(grain="one_row_per_unit", id_column="participant_id")
    state = _bulk(art, lens=["clinical"], target="sbp", purpose="inference", grain=named)
    clusters = resolve_clusters(state, rows[cluster_columns(state, list(rows.columns))])
    assert not clusters.clustered and not clusters.refusal
    assert "The grain answer names `participant_id` as the unit" in (clusters.note or "")
    # With no column named by the grain answer, which rows belong together is asked.
    state = _bulk(art, lens=["clinical"], target="sbp", purpose="inference",
                  grain=GrainSpec(grain="one_row_per_unit"))
    clusters = resolve_clusters(state, rows[cluster_columns(state, list(rows.columns))])
    assert clusters.refusal and not clusters.clustered
    asked = _one_reading_each(list(clusters.exits))
    column = next(a["column"] for a in asked if a["kind"] == "confirm_reading")
    assert column in codes
    assert {"kind": "confirm_reading", "reading": "cluster", "column": column, "value": "yes"} in asked
    assert {"kind": "confirm_reading", "reading": "cluster", "column": column, "value": "no"} in asked
    # The user's own answers settle it, one reading each.
    yes = state.model_copy(update={"reading_confirmations": {
        **{f"cluster:{c}": "no" for c in codes}, "cluster:recruiter_id": "yes"}})
    clusters = resolve_clusters(yes, rows[cluster_columns(yes, list(rows.columns))])
    assert clusters.clustered and clusters.column == "recruiter_id"
    assert clusters.n_clusters == counts["recruiter_id"] == 14
    no = state.model_copy(update={"reading_confirmations": {f"cluster:{c}": "no" for c in codes}})
    clusters = resolve_clusters(no, rows[cluster_columns(no, list(rows.columns))])
    assert not clusters.clustered and not clusters.refusal
    # HCHS/SOL: the published PSU is no unit's identifier by its name and repeats.
    hchs = _hchs()
    assert 100 < hchs["PSU_ID"].nunique() < len(hchs)
    psu = next(p for p in _roles(_write(hchs, tmp_path, "hchs.csv"), lens=["clinical"],
                                 target="HBA1C")["columns"] if p["column"] == "PSU_ID")
    assert not (psu["proposed"] == "identifier" and psu["confidence"] == "high"), psu


def test_g2_a_day_count_the_parser_does_not_read_settles_no_screen_at_the_stage(tmp_path):
    """Gate failure 2 at the stage level (logs_b2.out): ``kcal_sum_d1_d2``, ``Energy (kcal) - 2
    recalls`` and the rest were read as one day's intake by the Atwater check (which says nothing
    about days). Expected: wherever such a column is read as total energy, its day count is not
    settled (the name numbers several days, or the median is no day's intake: NUTRITION_PRIOR),
    every screen on it is refused, and no coach line calls anyone an under- or over-reporter.
    Reference (pandas): each 2-day total's median is above the one-day prior's 2,600 kcal."""
    from turbotab.core.readings import day_count_reading
    from turbotab.core.stages.proposals import build_proposals

    for name, (days, frame) in _days_cases().items():
        assert float(frame[name].median()) > 2_600 or days == 1, (name, frame[name].median())
        assert not day_count_reading(name, frame[name]).settled, name
        path = _write(frame, tmp_path, "d.csv")
        t = _table(path)
        roles = {p["column"]: p["proposed"] for p in _roles(path, lens=["dietary"],
                                                            target="ldl")["columns"]}
        out = build_proposals(t.frame(), t.info["columns"], lens=["dietary"], target="ldl",
                              roles=roles)
        reading = out["energy_unit"]
        if reading is not None:
            assert reading["confirmed"] is False, (name, reading)
        assert all(e["refused"] for e in out["exclusions"]), (name, out["exclusions"])
        assert "reporting" not in str(out["coach"]), (name, out["coach"])


def test_g2_two_day_totals_the_atwater_check_passes_are_asked_through_the_server(tmp_path):
    """Gate failure 2 through the real server (logs_s1a.out): ``energy_kcal_day1_day2`` (2-day
    totals, median about 4,160) was {kcal, atwater, days 1, confirmed}; the finding said "135
    records report an implausible daily intake", the coach "likely over-reporting", and the
    500–5,000 screen and the Goldberg screen were offered unrefused (135 and 218 of 500 removed).

    Expected: the unit is kcal by the Atwater identity, the days are not settled; the finding asks
    for the days; every screen is refused; the screen answer is refused with exits that record one
    day or a 2-day total; recorded as a 2-day total, the screen reads 1,000–10,000 and removes
    exactly the rows pandas counts outside it."""
    from turbotab.core.tests.acceptance.server_drive import local_server, open_project
    from turbotab.core.tests.truths import Truth

    frame = _days_server()
    E = "energy_kcal_day1_day2"
    v = frame[E]
    assert 4_000 < float(v.median()) < 4_400
    two_day_out = int(((v < 1_000) | (v > 10_000)).sum())
    one_day_out = int(((v < 500) | (v > 5_000)).sum())
    assert one_day_out > two_day_out
    path = _write(frame, tmp_path, "days.csv")
    with local_server(tmp_path / "home") as client:
        drive = open_project(client, path, Truth(ADULTS_CAUSAL("ldl", "_day1_day2", "weight_kg",
                                                               "height_cm"), fixture="s1_days"))
        _drive(drive, lens=["dietary"], target="ldl", stop_before="exclusions",
               grain={"kind": "set_grain", "grain": "one_row_per_unit",
                      "id_column": "participant_id"})
        prop = drive.artifact("proposals")
        unit = prop["energy_unit"]
        assert (unit["unit"], unit["basis"], unit["confirmed"]) == ("kcal", "atwater", False), unit
        assert unit["days_unsettled"] and 2 in unit["days_candidates"]
        assert prop["exclusions"] and all(x["refused"] for x in prop["exclusions"])
        assert "reporting" not in str(prop["coach"])
        found = drive.artifact("findings")["findings"]
        titles = [f["title"] for f in found]
        assert any(t.startswith(f"The days in `{E}` are not settled") for t in titles), titles
        assert not any("implausible daily intake" in t for t in titles), titles
        screen = next(x for x in prop["exclusions"] if x["key"] == "sex_neutral_500_5000")
        error = _refused(drive, {"kind": "set_exclusions", "rules": [screen["rule"]]})
        assert error["code"] == "energy_unit_unconfirmed"
        exits = [x["decision"] for x in error["exits"]]
        assert {"kind": "set_column_unit", "column": E, "unit": "kcal", "days": 1} in exits
        assert {"kind": "set_column_unit", "column": E, "unit": "kcal", "days": 2} in exits
        drive.decide({"kind": "set_column_unit", "column": E, "unit": "kcal", "days": 2})
        prop = drive.artifact("proposals")
        screen = next(x for x in prop["exclusions"] if x["key"] == "sex_neutral_500_5000")
        assert screen["refused"] is None and screen["affected"] == two_day_out
        drive.reach("exclusions", timeout=300)  # the findings re-read the recorded unit first
        drive.decide({"kind": "set_exclusions", "rules": [screen["rule"]]})
        assert drive.artifact("cohort")["n_final"] == len(frame) - two_day_out


def test_g3_a_weight_in_pounds_never_sets_the_goldberg_screen_unasked(tmp_path):
    """Gate failure 3 (logs_b6b.out, logs_s5.out): ``weight`` in lb (median about 154) was read
    as kg by its 25–200 median band; the Goldberg screen was offered unrefused and accepted,
    excluding 136 of 500 rows (5 in kg). Expected: a weight's unit known only from its median is
    asked (kg and lb overlap there; NHANES names it: NHANES_BMX), and so is a bare ``age``'s; the
    screen is refused with one confirmation per reading; the user who records lb is told the
    screen reads kg; nothing is excluded. Reference (pandas): the weight's median is 154 (lb)."""
    from turbotab.core.tests.acceptance.server_drive import local_server, open_project

    assert NHANES_BMX["BMXWT"].endswith("(kg)")
    from turbotab.core.tests.truths import Truth

    frame = _women_lb()
    assert 140 < float(frame["weight"].median()) < 170  # pounds: a kg median here is implausible
    path = _write(frame, tmp_path, "women_lb.csv")
    # The cohort's truth: energy one day's kcal (BLUEPRINT §14.3: a day count is recorded).
    truth = Truth({"code_or_count:age": "amount",
                   **ADULTS_CAUSAL("ldl", "_g", "weight", "height")}, fixture="b6b_goldberg")
    with local_server(tmp_path / "home") as client:
        drive = open_project(client, path, truth)
        _drive(drive, lens=["dietary"], target="ldl", stop_before="exclusions",
               grain={"kind": "set_grain", "grain": "one_row_per_unit", "id_column": "pid"})
        drive.decide({"kind": "set_column_unit", "column": "energy_kcal", "unit": "kcal", "days": 1})
        drive.reach("exclusions", timeout=300)
        prop = drive.artifact("proposals")
        gold = next(x for x in prop["exclusions"] if x["key"] == "goldberg_schofield")
        assert gold["refused"] and "`weight`" in gold["refused"] and "recorded" in gold["refused"]
        error = _refused(drive, {"kind": "set_exclusions", "rules": [gold["rule"]]})
        assert error["code"] == "reading_unsettled"
        asked = _one_reading_each(error["exits"])
        assert {"kind": "confirm_reading", "reading": "unit", "column": "weight",
                "value": "kg"} in asked
        assert {"kind": "confirm_reading", "reading": "unit", "column": "age",
                "value": "years"} in asked
        # The user knows the weights are in pounds: the rule offered before read kg, so it is
        # refused, with the rule that reads lb (converted exactly) as an exit (BLUEPRINT §14.3).
        drive.decide({"kind": "confirm_reading", "reading": "unit", "column": "weight",
                      "value": "lb"})
        drive.decide({"kind": "confirm_reading", "reading": "unit", "column": "age",
                      "value": "years"})
        error = _refused(drive, {"kind": "set_exclusions", "rules": [gold["rule"]]})
        assert error["code"] == "reading_unsettled" and "`weight` is in lb" in error["message"]
        assert any((x["decision"] or {}).get("rules", [{}])[0].get("weight_unit") == "lb"
                   for x in error["exits"] if x.get("decision"))
        drive.decide({"kind": "set_exclusions", "rules": []})
        assert drive.artifact("cohort")["n_final"] == len(frame)


def test_g4_recalls_across_named_occasions_are_asked_never_stated(tmp_path):
    """Gate failure 4 at the stage level (logs_b4.out): recalls numbered 1–4 across occasions named
    ``assessment``, ``time``, ``month``, ``stage``, ``year`` or ``trimester`` were stated "repeats"
    (``repeat_kind`` skipped, the mean recommended), erasing a 22% fall in one arm. Expected, for
    every occasion name: the repeat-kind question is asked (its reading is the proposal), no
    consumer reads a repeat kind before the answer, and no combining is recommended before it.
    Reference (pandas): the intervention arm's mean energy falls by 15–30% at follow-up."""
    from turbotab.core.interview import route
    from turbotab.core.stages.working import effective_repeat_kind, structure_stage

    fresh = {s: {"status": "fresh"} for s in (
        "ingest", "oriented", "profile", "findings", "structure", "working", "target_info", "roles",
        "proposals", "cohort", "split", "shelf", "design", "fit", "substitution")}
    rng = np.random.default_rng(31004)
    for occ, values in OCCASIONS.items():
        frame = _trial(rng, occ, values)
        arm = frame[frame["arm"] == "intervention"].groupby(occ)["energy_kcal"].mean()
        fall = 1 - arm[values[1]] / arm[values[0]]
        assert 0.15 < fall < 0.30, (occ, fall)
        state = ProjectState(lens=["dietary"], target="energy_kcal", task="regression",
                             purpose="inference",
                             grain=GrainSpec(grain="repeated", id_column="participant_id"))
        structure = _table(_write(frame, tmp_path, "trial.csv")).run(structure_stage, state)
        assert effective_repeat_kind(state, structure) is None, occ
        assert structure["aggregation"] is None, occ
        steps = {s.key: s for s in route(state, fresh, {"structure": structure})}
        assert steps["repeat_kind"].status == "open", (occ, steps["repeat_kind"])
        assert steps["temporal"].status != "not_applicable" or "repeats" not in (
            steps["temporal"].reason or ""), occ


def test_g4_the_assessment_trial_through_the_server_asks_and_follows_the_answer(tmp_path):
    """Gate failure 4 through the real server (logs_s3.out, ``assessment``): ``repeat_kind`` was
    skipped, ``temporal`` "not applicable (repeats)", and the mean recommended. Expected: the
    question is asked; answered "time points", no default combining is recommended (the mean
    would erase the arm's change)."""
    from turbotab.core.tests.acceptance.server_drive import local_server, open_project

    frame = _trial(np.random.default_rng(31004), "assessment", ("baseline", "follow-up"),
                   server=True)
    path = _write(frame, tmp_path, "occ.csv")
    with local_server(tmp_path / "home") as client:
        drive = open_project(client, path)
        _drive(drive, lens=["dietary"], target="sbp", stop_before="repeat_kind",
               grain={"kind": "set_grain", "grain": "repeated", "id_column": "participant_id"})
        step = drive.reach("repeat_kind", timeout=300)
        assert step["status"] == "open", step
        assert drive.artifact("structure")["aggregation"] is None
        drive.decide({"kind": "set_repeat_kind", "repeat_kind": "time_points"})
        drive.decide({"kind": "set_unit", "unit": "unit"})
        menu = drive.artifact("structure")["aggregation"]
        assert menu["kind"] == "time_points" and menu["recommended"] != "mean", menu


def test_g5_a_bare_amount_is_no_outcome_unit_any_sentence_states(tmp_path):
    """Gate failure 5 at the stage level (logs_b9.out): ``bmi_kg`` was stated "in kg" (BMI is
    kg/m²: NHANES_BMX), ``ldl_mg``/``glucose_mg`` "in mg", ``hb_g`` "in g", with no proposal shown.
    Expected: no unit is stated for them (``unit`` and ``unit_source`` null, the record's sentence
    names none); the pack's units are offered; a whole unit the quantity takes is still stated
    (controls: ``weight_kg`` in kg, ``sbp_mmhg`` in mmHg)."""
    from turbotab.core import voice
    from turbotab.core.stages.target import target_info_stage

    assert NHANES_BMX["BMXBMI"].endswith("(kg/m**2)")
    candidates = {"bmi_kg": {"kg/m²"}, "ldl_mg": {"mg/dL", "mmol/L"},
                  "glucose_mg": {"mg/dL", "mmol/L"}, "hb_g": {"g/dL", "g/L"}}
    for name, frame in _unit_tables().items():
        t = _table(_write(frame, tmp_path, f"{name}.csv"))
        out = t.run(target_info_stage, ProjectState(lens=["clinical"], target=name,
                                                    purpose="inference"))
        assert (out["unit"], out["unit_source"]) == (None, None), (name, out["unit"])
        assert set(out["unit_candidates"]) == candidates[name], (name, out["unit_candidates"])
        said = voice.sentence_for(d.SetTarget(column=name), ProjectState(),
                                  {"frame": frame, "outcome_unit": out["unit"]})
        assert ", in " not in said, said
    rng = np.random.default_rng(1)
    # BLUEPRINT §14.3 (names never count as corroboration): a whole unit the name spells out is
    # the best guess a question leads with, stated only once recorded.
    for name, unit in (("weight_kg", "kg"), ("sbp_mmhg", "mmHg")):
        frame = pd.DataFrame({"pid": np.arange(50), name: rng.normal(80, 10, 50).round(1)})
        out = _table(_write(frame, tmp_path, f"{name}.csv")).run(
            target_info_stage, ProjectState(lens=["clinical"], target=name))
        assert (out["unit"], out["unit_source"]) == (None, None), (name, out["unit"])
        assert out["proposed_unit"] == unit, (name, out["proposed_unit"])


def test_g5_ldl_mg_through_the_server_states_no_unit_until_recorded(tmp_path):
    """Gate failure 5 through the real server (logs_s6.out): ``ldl_mg``'s record read "chosen as
    the outcome, in mg" and the fit's labels carried "mg". Expected: no unit in the record or the
    fit until the user records one; recorded (mg/dL), the record states it."""
    from turbotab.core.tests.acceptance.server_drive import local_server, open_project

    from turbotab.core.tests.truths import Truth

    frame = _ldl_mg()
    path = _write(frame, tmp_path, "ldl.csv")
    with local_server(tmp_path / "home") as client:
        # WP17, the generator: age and statin use move LDL, drawn apart; sex moves nothing.
        drive = open_project(client, path, Truth({
            "code_or_count:age": "amount", "exposure:ldl_mg": "statin",
            "adjust:age": "no,yes,no", "adjust:sex": "no,no,no"}, fixture="s6"))
        _drive(drive, lens=["clinical"], target="ldl_mg",
               grain={"kind": "set_grain", "grain": "one_row_per_unit", "id_column": "pid"})
        info = drive.artifact("target_info")
        assert info["unit"] is None and set(info["unit_candidates"]) == {"mg/dL", "mmol/L"}
        record = next(r for r in drive.view()["decisions"]
                      if r["decision"]["kind"] == "set_target")
        assert record["sentence"].startswith("`ldl_mg` was chosen as the outcome")
        assert ", in mg" not in record["sentence"]
        fit = drive.artifact("fit", timeout=600)
        assert not [s for s in _strings(fit) if " mg" in s and "mg/" not in s]
        drive.decide({"kind": "set_outcome_unit", "column": "ldl_mg", "unit": "mg/dL"})
        assert drive.artifact("target_info")["unit"] == "mg/dL"


def test_g6_whole_number_counts_are_combined_as_the_user_says_through_the_server(tmp_path):
    """Gate failure 6 through the real server (logs_s7.out): ``coffee_cups`` and
    ``eating_occasions`` (per-recall counts) were combined by their mode under "mean", changing 112
    and 130 of 150 people's values, and the sentence said only "any codes took their most frequent
    value". Expected: the combining answer is refused with one pair of exits per column (a count,
    or codes), never one for both; answered "counts", each person's value is the pandas mean of
    their recalls for all 150; the sentence names no column as a code."""
    from turbotab.core.graph import artifact_dir
    from turbotab.core.tests.acceptance.server_drive import local_server, open_project

    from turbotab.core.tests.truths import Truth, asked as readings

    frame = _recalls_with_counts()
    truth = frame.groupby("participant_id")[["coffee_cups", "eating_occasions"]].mean()
    path = _write(frame, tmp_path, "recalls.csv")
    with local_server(tmp_path / "home") as client:
        drive = open_project(client, path, Truth({"code_or_count:energy_kcal": "amount"},
                                                 fixture="s7_combine"))
        _drive(drive, lens=["dietary"], target="hba1c", stop_before="aggregation",
               grain={"kind": "set_grain", "grain": "repeated", "id_column": "participant_id"},
               answers={"repeat_kind": {"kind": "set_repeat_kind", "repeat_kind": "repeats"}})
        drive.reach("aggregation", timeout=300)
        error = _refused(drive, {"kind": "set_aggregation", "method": "mean"})
        assert error["code"] == "reading_unsettled"
        asked = _one_reading_each(error["exits"])
        # BLUEPRINT §14.3: every whole-valued column that changes within a person is asked (the
        # whole kcal too), each with its pair of exits.
        assert sorted((a["column"], a["value"]) for a in asked) == [
            ("coffee_cups", "amount"), ("coffee_cups", "code"),
            ("eating_occasions", "amount"), ("eating_occasions", "code"),
            ("energy_kcal", "amount"), ("energy_kcal", "code")]
        assert sorted(c for _, c in readings(error["exits"])) == [
            "coffee_cups", "eating_occasions", "energy_kcal"]
        for column in ("coffee_cups", "eating_occasions", "energy_kcal"):
            drive.decide({"kind": "confirm_reading", "reading": "code_or_count", "column": column,
                          "value": "amount"})
        drive.reach("aggregation", timeout=300)  # the table re-reads the answers first
        drive.decide({"kind": "set_aggregation", "method": "mean"})
        record = drive.view()["decisions"][-1]
        assert "codes" not in record["sentence"], record["sentence"]
        drive.artifact("working")
        key = drive.view()["stages"]["working"]["key"]
        cache = drive.c.app.state.service.workspace.cache_dir(drive.pid)
        tables = sorted(Path(artifact_dir(cache, "working", key)).rglob("*.parquet"))
        table = pd.read_parquet(next(p for p in tables if "map" not in p.name))
        got = table.set_index("participant_id")[["coffee_cups", "eating_occasions"]]
        joined = got.join(truth, rsuffix="_mean")
        assert len(joined) == 150
        for column in ("coffee_cups", "eating_occasions"):
            assert np.allclose(joined[column], joined[f"{column}_mean"]), column


def test_g7_a_design_the_user_set_by_role_is_asked_and_offered(tmp_path):
    """Gate failure 7 (checked on the state): NDNS RP (``wti_Y911``, ``astrata1``, ``area``) and
    MEPS (``PERWT19F``, ``VARSTR``, ``VARPSU``) set to ``design`` by the user still gave "No column
    reads as a survey weight …", so the survey question was never asked and the estimate was
    unweighted. Expected: the user's settled design roles make the question apply under inference;
    the options weight by the column the values place as a weight, each placement of strata and
    PSU named (MEPS_DESIGN, NDNS_STRATA, NDNS_PSU); the option naming the documented design is
    accepted, and the fit builds a population design from it. A design role that rode along
    unconfirmed is no design here: the fit asks for it first."""
    from turbotab.core.interview import route
    from turbotab.core.methods.survey import for_fit, proposal
    from turbotab.core.survey import not_applicable_reason, reading_of

    assert "VARSTR and VARPSU" in MEPS_DESIGN and "PSU" in MEPS_VARIABLES["VARPSU"]
    fresh = {s: {"status": "fresh"} for s in ("ingest", "oriented", "profile", "findings",
                                               "structure", "working", "target_info", "roles",
                                               "proposals", "cohort", "split")}
    cases = {
        "ndns": (_ndns(), "Glucose", ["wti_Y911", "wtn_Y911", "wtb_Y911", "astrata1", "area"],
                 ("wti_Y911", "astrata1", "area")),
        "meps": (_meps(), "BMINDX53", ["PERWT19F", "VARSTR", "VARPSU"],
                 ("PERWT19F", "VARSTR", "VARPSU")),
    }
    for name, (frame, target, design, (weight, strata, psu)) in cases.items():
        path = _write(frame, tmp_path, f"{name}.csv")
        t = _table(path)
        art = _roles(path, lens=["survey"], target=target)
        # The user's own roles: the design columns set to design (an answer, not a ride-along).
        roles = {p["column"]: p["proposed"] for p in art["columns"]}
        roles.update({c: "design" for c in design})
        state = ProjectState(lens=["survey"], target=target, purpose="inference", roles=roles,
                             grain=GrainSpec(grain="one_row_per_unit"))
        assert set(reading_of(state).unplaced) | set(reading_of(state).columns()) >= set(design)
        assert not_applicable_reason(state) is None, name
        steps = {s.key: s for s in route(state, fresh, {})}
        assert steps["survey"].status in ("open", "waiting"), (name, steps["survey"])
        with t.store() as store:
            offer = proposal(state, store)
        options = [o["decision"] for o in offer["options"]]
        documented = {"kind": "set_survey", "estimand": "population", "weight": weight,
                      "strata": strata, "psu": psu, "cycle": None, "four_year_weight": None,
                      "acknowledged": False}
        assert documented in options, (name, options)
        assert options[-1] == {"kind": "set_survey", "estimand": "sample"}
        ctx = {"state": state, "columns": list(frame.columns), "target": target,
               "column_info": {c["name"]: c for c in t.info["columns"]}}
        d.validate(documented, ctx)
        answered = state.model_copy(update={"survey": d.SurveySpec(
            **{k: v for k, v in documented.items() if k != "kind"})})
        with t.store() as store:
            fitted = for_fit(answered, store)
        assert fitted.answer == "population" and fitted.design is not None and not fitted.refusal
        # Bulk-confirmed design roles are unsettled: not read as a design, and the fit asks first.
        waiting = state.model_copy(update={"roles_unconfirmed": list(design)})
        assert not set(reading_of(waiting).unplaced) & set(design)
        with pytest.raises(d.Refusal) as refused:
            d.validate({"kind": "select_models", "models": ["linear"]}, {**ctx, "state": waiting})
        assert refused.value.code == "reading_unsettled"
        assert {x["decision"]["column"] for x in refused.value.exits if x["decision"]
                and "column" in x["decision"]} >= set(design)


# ═════════════════════════════════════════════════════════════════════════════
# B · The ledger's mechanics
# ═════════════════════════════════════════════════════════════════════════════


def test_b1_a_reading_is_settled_only_by_its_values_at_high_or_by_the_user():
    """BLUEPRINT_14_1, as a truth table."""
    from turbotab.core.readings import Reading, settled

    def r(confidence: str, corroborated: bool, state: str = "proposed") -> Reading:
        return Reading(("x",), "unit", "kg", confidence, "", corroborated, state)

    assert settled(r("high", True)) is True
    assert settled(r("high", False)) is False  # a name read high is no corroboration
    assert settled(r("medium", True)) is False
    assert settled(r("low", False, "confirmed")) is True
    assert settled(None) is False


def test_b2_one_confirmation_decision_one_reading_per_record_and_old_records_stay_valid():
    """``confirm_reading`` names one column, one kind and one value; a role's is kept where
    ``confirm_role`` kept it (so the old records fold as before), the table-shaping kinds apart
    from the rest; a revert restores the reading's earlier state."""
    from datetime import datetime, timezone

    from turbotab.core.readings import (
        code_or_count_reading, confirmation, role_reading, time_column_reading,
    )

    fields = set(d.ConfirmReading.model_fields)
    assert fields == {"kind", "reading", "column", "value"}
    roles = {"weight": "covariate", "smoker": "covariate", "visit_date": "time"}

    def rec(seq: int, decision: dict) -> d.DecisionRecord:
        return d.DecisionRecord(id=f"r{seq}", seq=seq, at=datetime.now(timezone.utc),
                                decision=d.parse_decision(decision))

    log = [rec(1, {"kind": "set_roles", "roles": roles,
                   "unconfirmed": ["weight", "smoker", "visit_date"]}),
           rec(2, {"kind": "confirm_role", "column": "weight", "role": "covariate"}),
           rec(3, {"kind": "confirm_reading", "reading": "role", "column": "smoker",
                   "value": "covariate"}),
           rec(4, {"kind": "confirm_reading", "reading": "code_or_count", "column": "smoker",
                   "value": "code"}),
           rec(5, {"kind": "confirm_reading", "reading": "time_column", "column": "visit_date",
                   "value": "orders"}),
           rec(6, {"kind": "confirm_reading", "reading": "unit", "column": "weight",
                   "value": "kg"})]
    state = d.fold(log)
    assert state.role_confirmations == {"weight": "covariate", "smoker": "covariate"}
    assert state.shape_confirmations == {"code_or_count:smoker": "code",
                                         "time_column:visit_date": "orders"}
    # A unit is kept with the recorded units, where ``set_column_unit`` keeps them (one store,
    # BLUEPRINT §14.3 amendment after the fifth gate), its days unrecorded.
    assert state.column_units == {"weight": d.ColumnUnitSpec(unit="kg", days=None)}
    assert role_reading(state, "weight").settled and role_reading(state, "smoker").settled
    # Naming ``visit_date`` as the column that orders a unit's records is the user's own answer
    # that it is the time (BLUEPRINT §14.3: the time role is settled by the user, never by values).
    assert role_reading(state, "visit_date").settled
    assert confirmation(state, "unit", "weight") == "kg"
    assert code_or_count_reading(state, "smoker", dtype="integer", n_unique=3).settled
    assert time_column_reading(state, None).column == "visit_date"
    undone = d.fold([*log, rec(7, {"kind": "revert", "decision_id": "r3"})])
    assert not role_reading(undone, "smoker").settled and role_reading(undone, "weight").settled
    for bad in ({"kind": "confirm_reading", "reading": "unit", "column": "weight", "value": "stone"},
                {"kind": "confirm_reading", "reading": "code_or_count", "column": "smoker",
                 "value": "maybe"},
                {"kind": "confirm_reading", "reading": "role", "column": "y", "value": "exposure"},
                {"kind": "confirm_reading", "reading": "day_count", "column": "weight",
                 "value": "0"}):
        with pytest.raises(d.Refusal):
            d.validate(bad, {"columns": ["weight", "smoker", "visit_date", "y"], "target": "y"})


def test_b3_a_bulk_roles_answer_is_read_against_the_proposals_a_client_was_shown():
    """The census: "a set_roles recorded with no fresh roles artifact settles everything". While
    the roles stage recomputes, the newest proposals it computed (stale) are what a client was
    shown, and the bulk answer is read against them; when none was ever computed, nothing could
    ride along, and the roles are the user's own."""
    proposals = {"columns": [{"column": "a", "proposed": "covariate", "confidence": "low"},
                             {"column": "b", "proposed": "exposure", "confidence": "high"}]}
    answer = {"kind": "set_roles", "roles": {"a": "covariate", "b": "exposure"}}
    stale = {"columns": ["a", "b", "y"], "target": "y", "artifact": lambda stage: None,
             "shown": lambda stage: proposals if stage == "roles" else None}
    assert d.validate(answer, stale).unconfirmed == ["a"]
    fresh = {**stale, "artifact": lambda stage: proposals if stage == "roles" else None,
             "shown": None}
    assert d.validate(answer, fresh).unconfirmed == ["a"]
    never = {**stale, "shown": lambda stage: None}
    assert d.validate(answer, never).unconfirmed == []


def test_b4_the_fit_asks_for_each_whole_number_predictor_it_would_read_as_an_amount():
    """The census (``pipeline.is_categorical``): every integer column entered the fit as a slope
    unless declared categorical, and no question asked. Expected: a predictor with 3–10 whole-
    number values is asked (a code, or an amount: one exit pair per column), a declared or
    confirmed one is not, two values and many values are settled by the values."""
    roles = {"education": "covariate", "smoker": "exposure", "age": "covariate", "male": "covariate"}
    info = {"education": {"dtype": "integer", "n_unique": 5}, "smoker": {"dtype": "integer", "n_unique": 3},
            "age": {"dtype": "integer", "n_unique": 52},
            "male": {"dtype": "integer", "n_unique": 2, "whole": True, "zero_one": True}}
    state = ProjectState(target="y", roles=roles)
    ctx = {"state": state, "columns": [*roles, "y"], "target": "y", "column_info": info}
    with pytest.raises(d.Refusal) as refused:
        d.validate({"kind": "select_models", "models": ["linear"]}, ctx)
    asked = _one_reading_each(refused.value.exits)
    # BLUEPRINT §14.3 (the consumer sets the scope): every whole-valued predictor is asked, any
    # count of values (age's 52 too); exactly 0/1 is one indicator either way.
    assert sorted((a["column"], a["value"]) for a in asked) == [
        ("age", "amount"), ("age", "code"), ("education", "amount"), ("education", "code"),
        ("smoker", "amount"), ("smoker", "code")]
    said = state.model_copy(update={"categorical": ["education"],
                                    "shape_confirmations": {"code_or_count:smoker": "amount",
                                                            "code_or_count:age": "amount"}})
    d.validate({"kind": "select_models", "models": ["linear"]}, {**ctx, "state": said})


def test_b5_a_confirmation_a_recorded_answer_reads_cannot_be_undone_silently():
    """BLUEPRINT §13's "invalidates", for readings: undoing the weight's unit while the recorded
    Goldberg screen reads it, or the order a unit's records were combined by, is refused until that
    answer changes; once it changes, the undo is accepted."""
    from datetime import datetime, timezone

    def rec(seq: int, decision: dict) -> d.DecisionRecord:
        return d.DecisionRecord(id=f"r{seq}", seq=seq, at=datetime.now(timezone.utc),
                                decision=d.parse_decision(decision))

    gold = {"kind": "goldberg", "column": "energy_kcal", "days": 1, "sex": "sex", "female": ["F"],
            "male": ["M"], "age": "age", "weight": "weight", "equation": "schofield", "pal": 1.55,
            "reason": "implausible energy reports"}
    log = [rec(1, {"kind": "set_roles", "roles": {"energy_kcal": "energy", "age": "covariate",
                                                  "weight": "covariate", "sex": "covariate"}}),
           rec(2, {"kind": "confirm_reading", "reading": "unit", "column": "weight", "value": "kg"}),
           rec(3, {"kind": "confirm_reading", "reading": "unit", "column": "age", "value": "years"}),
           rec(4, {"kind": "set_exclusions", "rules": [gold]})]
    undo = {"kind": "revert", "decision_id": "r2"}
    ctx = {"state": d.fold(log), "records": lambda: log}
    with pytest.raises(d.Refusal) as refused:
        d.validate(undo, ctx)
    assert refused.value.code == "role_unconfirmed" and "`weight`" in refused.value.message
    log.append(rec(5, {"kind": "set_exclusions", "rules": []}))
    d.validate(undo, {"state": d.fold(log), "records": lambda: log})
    # The order a unit's records were combined in.
    timed = [rec(1, {"kind": "set_grain", "grain": "repeated", "id_column": "pid"}),
             rec(2, {"kind": "set_repeat_kind", "repeat_kind": "time_points"}),
             rec(3, {"kind": "set_unit", "unit": "unit"}),
             rec(4, {"kind": "confirm_reading", "reading": "time_column", "column": "visit_date",
                     "value": "orders"}),
             rec(5, {"kind": "set_aggregation", "method": "last", "outcome": "last"})]
    with pytest.raises(d.Refusal):
        d.validate({"kind": "revert", "decision_id": "r4"},
                   {"state": d.fold(timed), "records": lambda: timed})


# ═════════════════════════════════════════════════════════════════════════════
# C · The structural check: the census against the ledger
# ═════════════════════════════════════════════════════════════════════════════

# Census entries whose consumers act only on the user's own recorded answer, are settled by their
# own values, or are this package's stated deviations, each with its reason (deviations in the
# package's report): every other entry the census found unsettled ("no" or "partly") must have a
# number-changing consumer that reads through the ledger.
ACCEPTED = {
    "missing_not_asked": "the leave-out answer names every column it drops (set_missing "
                         "drop_columns), the user's own record",
    "below_detection": "the columns are named by the user's answer; the detection limit read as "
                       "the smallest detected value is a deviation (not yet asked)",
    "missing_level": "two values give one line either way; the user chose blanks as a level",
    "ingest_values": "deviation: NULL_TOKENS blanks 'NA'/'NULL' at ingest, below the ledger "
                     "(the data layer)",
    "free_text": "a many-level category is excluded high by its distinct counts (values)",
    "plausibility": "values change only through the user's apply_repair with its parameters "
                    "shown; the assumed 1 = male coding in its counts is a deviation",
    "survey_cycle": "the option names the cycle column it pools by and the user records it",
}


def _census() -> list[dict[str, Any]]:
    return json.loads(CENSUS.read_text("utf-8"))["census"]


class _Calls:
    """The call graph of turbotab/core and turbotab/server, read with ``ast`` (never imported):
    which functions call into ``turbotab.core.readings``, and what each calls."""

    LEDGER = "turbotab.core.readings"

    def __init__(self) -> None:
        self.defs: dict[tuple[str, str], ast.AST] = {}
        self.imports: dict[str, dict[str, tuple[str, str | None]]] = {}
        for base in ("turbotab/core", "turbotab/server"):
            for path in sorted((ROOT / base).rglob("*.py")):
                if "/tests/" in str(path):
                    continue
                module = ".".join(path.relative_to(ROOT).with_suffix("").parts)
                tree = ast.parse(path.read_text("utf-8"))
                self.imports[module] = self._imports(tree.body, module)
                for node in tree.body:
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        self.defs[(module, node.name)] = node
                    elif isinstance(node, ast.ClassDef):
                        for item in node.body:
                            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                self.defs[(module, f"{node.name}.{item.name}")] = item

    @staticmethod
    def _imports(body: Any, module: str) -> dict[str, tuple[str, str | None]]:
        """name -> (module, attribute or None for a module alias), from the import statements."""
        out: dict[str, tuple[str, str | None]] = {}
        for node in body if isinstance(body, list) else ast.walk(body):
            if isinstance(node, ast.ImportFrom) and node.module:
                for alias in node.names:
                    target = f"{node.module}.{alias.name}"
                    out[alias.asname or alias.name] = ((target, None) if target == _Calls.LEDGER
                                                       else (node.module, alias.name))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    out[alias.asname or alias.name.split(".")[0]] = (alias.name, None)
        return out

    def _bindings(self, module: str, fn: ast.AST) -> dict[str, tuple[str, str | None]]:
        local = self._imports(fn, module)
        return {**self.imports.get(module, {}), **local}

    def _targets(self, module: str, key: str) -> tuple[bool, list[tuple[str, str]]]:
        """(calls the ledger itself, the functions it calls that this graph holds)."""
        fn = self.defs[(module, key)]
        names = self._bindings(module, fn)
        ledger = False
        out: list[tuple[str, str]] = []
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            if isinstance(f, ast.Name):
                bound = names.get(f.id)
                if bound and bound[0] == self.LEDGER and bound[1] is not None:
                    ledger = True
                elif bound and (bound[0], bound[1] or "") in self.defs:
                    out.append((bound[0], bound[1] or ""))
                elif (module, f.id) in self.defs:
                    out.append((module, f.id))
            elif isinstance(f, ast.Attribute):
                owner = f.value.id if isinstance(f.value, ast.Name) else None
                bound = names.get(owner) if owner else None
                if bound and bound[1] is None and bound[0] == self.LEDGER:
                    ledger = True
                elif bound and bound[1] is None and (bound[0], f.attr) in self.defs:
                    out.append((bound[0], f.attr))
                else:  # a method: one of this module's classes'
                    out += [k for k in self.defs if k[0] == module and k[1].endswith(f".{f.attr}")]
        return ledger, out

    def reads_ledger(self, module: str, key: str, depth: int = 3) -> bool:
        """It calls into the ledger itself, or through what it calls (up to ``depth`` hops)."""
        seen: set[tuple[str, str]] = set()
        frontier = [(module, key)]
        for _ in range(depth + 1):
            nxt = []
            for node in frontier:
                if node in seen or node not in self.defs:
                    continue
                seen.add(node)
                ledger, targets = self._targets(*node)
                if ledger:
                    return True
                nxt += targets
            frontier = nxt
        return False

    def reaches(self, source: tuple[str, str], target: tuple[str, str], depth: int = 4) -> bool:
        seen: set[tuple[str, str]] = set()
        frontier = [source]
        for _ in range(depth + 1):
            nxt = []
            for node in frontier:
                if node == target:
                    return True
                if node in seen or node not in self.defs:
                    continue
                seen.add(node)
                nxt += self._targets(*node)[1]
            frontier = nxt
        return False


def _where(consumer: Any) -> tuple[str, str]:
    module, key = consumer.where.split(":")
    return module, key


def test_c1_every_reader_in_the_census_is_covered_by_the_ledgers_registry():
    from turbotab.core.readings import CONSUMERS

    census = _census()
    assert len(census) == 51 and len({e["id"] for e in census}) == 51
    covered = {i for c in CONSUMERS for i in c.census}
    ids = {e["id"] for e in census}
    assert ids - covered == set(), ids - covered
    assert covered - ids == set(), covered - ids  # the registry names no reader the census lacks


def test_c2_every_unsettled_reader_has_a_consumer_that_reads_the_ledger_or_a_stated_reason():
    from turbotab.core.readings import ASK, CONSUMERS, NO_UNIT, SETTLED_ONLY

    leashed = (ASK, SETTLED_ONLY, NO_UNIT)
    missing = []
    for entry in _census():
        verdict = entry["settled_today"].lower()
        if not (verdict.startswith("no") or verdict.startswith("partly")):
            continue
        ours = [c for c in CONSUMERS if entry["id"] in c.census]
        if entry["id"] in ACCEPTED:
            continue
        if not any(c.changes and c.path in leashed for c in ours):
            missing.append(entry["id"])
    assert missing == [], missing
    # Only census entries the census itself found unsettled may be accepted, each with a reason.
    unsettled = {e["id"] for e in _census()
                 if not e["settled_today"].lower().startswith("yes")}
    assert set(ACCEPTED) <= unsettled and all(ACCEPTED.values())


def test_c3_every_number_changing_consumer_reads_through_the_ledger():
    """The AST call graph (``_Calls``): each registered consumer exists; each that can change a
    number calls into ``turbotab.core.readings`` itself or through what it calls, or names the
    function (``via``) that settles its readings and reaches it."""
    from turbotab.core.readings import CONSUMERS

    calls = _Calls()
    absent = [c.where for c in CONSUMERS if _where(c) not in calls.defs]
    assert absent == [], absent
    unleashed = []
    for c in CONSUMERS:
        if not c.changes:
            continue
        here = _where(c)
        if calls.reads_ledger(*here):
            continue
        if c.via is not None:
            via = tuple(c.via.split(":"))
            assert via in calls.defs, c.via
            if calls.reads_ledger(*via) and calls.reaches(via, here):
                continue
        if c.path.startswith(("acts only on the user", "settled by its own values")):
            continue  # its readings are the user's own answer, or its own values'
        unleashed.append(c.where)
    assert unleashed == [], unleashed


def test_c3_the_call_graph_check_can_fail():
    """Positive controls of the checker: a function that reads roles as given and calls nothing
    in the ledger is caught; one that does is passed; a ``via`` that does not reach its consumer
    does not vouch for it."""
    calls = _Calls()
    assert not calls.reads_ledger("turbotab.core.models.pipeline", "predictors_from_roles")
    assert not calls.reads_ledger("turbotab.core.stages.rows", "predictors")
    assert calls.reads_ledger("turbotab.core.models.pipeline", "model_predictors")
    assert calls.reads_ledger("turbotab.core.leash", "unsettled") is False  # the shim imports only
    assert not calls.reaches(("turbotab.core.units", "outcome_unit"),
                             ("turbotab.core.models.pipeline", "design_spec"))
