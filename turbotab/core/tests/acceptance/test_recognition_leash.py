"""Recognition's leash (BLUEPRINT §14, 2026-10-03): a recognizer may be wrong; a wrong recognition may
never silently change a number.

Three rounds of intelligence fixes each closed the names they were shown, and each next verifier's
fresh real-world names opened new failures. The standard is structural, so these tests are too:

1. **"High" is earned by values, never by a name alone, codebook names included.** A nutrient's
   values are plausible as an intake and rise with total energy at r ≥ 0.3 (an effect size, not
   p < α), or add up with the other macronutrients to total energy (the Atwater identity); an
   identifier has at least 3 distinct values and a unit structure; a flag is binary and tied to a
   base column's missingness; a time varies within units. A name-only reading is medium at most.
2. **Number-changing defaults read settled roles only.** A proposal below high carries an attention
   marker and the roles payload lists it; a bulk ``set_roles`` records it unconfirmed (the server
   fills ``unconfirmed``), and only its own ``confirm_role`` settles it. The energy card's nutrients
   and energy column, the intake screens, the grouping identifier, the survey weight and the
   "self-reported intake" line read settled roles; a decision naming an unsettled one is refused
   with one exit per column, never one for all.
3. **Ambiguity that touches a number is asked**: a day count in a total-energy name; a subsample
   weight's zeros; a multi-cycle weight's name; a repeat kind from a date the same on every row of a
   unit, or recalls numbered across occasions the study names.

Section A replays every open item and unasked assumption of the intelligence gate's second report
(``prior_gate_intelligence_2.json``), draw for draw from the verifier's generators (round 2, seeds
9101–9921). Section B checks the mechanics through the real server. Section C is the corpus: real
naming conventions (Hologic DXA, InBody, ASA24, CDISC ADaM, UK Biobank, NHANES multi-cycle,
wearables, EHR flags) with each column's documented meaning, asserting that no name-only reading is
high, that every high reading is right, and that no number-changing default comes from an
unconfirmed medium or low proposal; it reports the recognizers' accuracy.

Expected values never come from the code under test: NumPy/pandas computations on the fixtures, the
FAO Atwater factors, and quoted primary sources (fetched 2026-10-03).
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from turbotab.core import decisions as d
from turbotab.core.decisions import GrainSpec, ProjectState
from turbotab.core.tests.stage_harness import Ingested

ATWATER = {"protein": 4.0, "carbohydrate": 4.0, "fat": 9.0, "alcohol": 7.0}  # FAO general factors
KCAL_PER_KJ = 4.184

# ── primary sources, quoted ──────────────────────────────────────────────────

BLUEPRINT_14 = "A recognizer may be wrong. A wrong recognition may never silently change a number."
# NHANES Tutorials, Weighting Module (wwwn.cdc.gov/nchs/nhanes/tutorials/weighting.aspx):
TUTORIAL_MEC6YR = "if sddsrvyr in (2,3,4) then MEC6YR = 1/3 * WTMEC2YR;"
TUTORIAL_LCD = ("use 'the least common denominator' where the variable that was collected on the "
                "smallest number of respondents is the 'least common denominator.'")
# NHANES 2017–2018 GLU_J codebook (wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/GLU_J.htm),
# WTSAF2YR: SAS label, and its value table's zero row (Code or Value | Description | Count).
GLU_J_WTSAF2YR = "Fasting Subsample 2 Year MEC Weight"
GLU_J_ZERO_ROW = ("0", "No Lab Result or Not Fasting for 8 to <24 hours", 325)
# CLSA Data Support Document, Whole Body DEXA Reanalysis (Baseline), v3.0 2020Aug27: Hologic Apex
# variable names, "WBTOT_MASS = WBTOT_FAT + WBTOT_LEAN" and "SYM_WBC_WBTOT_PFAT 100 x
# SYM_WBC_WBTOT_FAT …" (whole-body percent fat).
CLSA_DXA = "WBTOT_MASS = WBTOT_FAT + WBTOT_LEAN"
# InBody, The Professional's Guide to the InBody Result Sheet: the body is "broken into even smaller
# pieces: Total Body Water, Protein, Minerals, and Body Fat Mass", and "By adding Total Body Water,
# Protein, and Minerals, you get Fat Free Mass (FFM)" (``Protein`` is body protein, in kg).
INBODY = "Total Body Water, Protein, Minerals, and Body Fat Mass"
# IDATA ASA24 Totals data dictionary (NCI CDAS, dictionary_idata_asa24_tns): "tns_kcal_asa24 Energy
# (kcal)", "tns_prot_asa24 Protein (g)", "tns_alc_asa24 Alcohol (g)".
ASA24_TNS = {"tns_kcal_asa24": "Energy (kcal)", "tns_prot_asa24": "Protein (g)",
             "tns_alc_asa24": "Alcohol (g)"}
# UK Biobank showcase: field 100002 "Energy", units "Continuous, KJ"; field 30120 "Lymphocyte
# count", units "10^9 cells/Litre".
UKB = {100002: ("Energy", "KJ"), 30120: ("Lymphocyte count", "10^9 cells/Litre")}
# Fitabase Fitbit Data Dictionary, Daily Activity: "Calories … Total estimated energy expenditure
# (in kilocalories)."
FITABASE_CALORIES = "Total estimated energy expenditure (in kilocalories)."


def _write(frame: pd.DataFrame, folder: Path, name: str) -> Path:
    path = Path(folder) / name
    frame.to_csv(path, index=False)
    return path


def _table(path: Path) -> Ingested:
    return Ingested(path, Path(tempfile.mkdtemp()))


def _roles_artifact(path: Path, *, lens, target=None, purpose="inference") -> dict:
    from turbotab.core.stages.rows import roles_stage

    return _table(path).run(roles_stage, ProjectState(lens=lens, target=target, purpose=purpose))


def _roles(path: Path, **kw) -> dict:
    return {p["column"]: p for p in _roles_artifact(path, **kw)["columns"]}


def _bulk_state(artifact: dict, *, lens, target, purpose="inference", grain=None) -> ProjectState:
    """The state a bulk "confirm the roles of N columns" leaves, computed here: every proposal
    recorded as proposed, the ones below high recorded unconfirmed (BLUEPRINT §14 rule 2)."""
    roles = {p["column"]: p["proposed"] for p in artifact["columns"]}
    waiting = [p["column"] for p in artifact["columns"] if p["confidence"] != "high"]
    return ProjectState(lens=lens, target=target, purpose=purpose, roles=roles,
                        roles_unconfirmed=waiting, grain=grain)


def _proposals_stage(path: Path, state: ProjectState, artifact: dict) -> dict:
    """The proposals stage as the engine runs it, on the bulk-confirmed state."""
    from turbotab.core.stages.proposals import proposals_stage

    return _table(path).run(proposals_stage, state, inputs={"roles": artifact})


def _high(artifact: dict) -> set[str]:
    return {p["column"] for p in artifact["columns"] if p["confidence"] == "high"}


def _r(x, y) -> float:
    both = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"),
                         "y": pd.to_numeric(y, errors="coerce")}).dropna()
    return float(np.corrcoef(both["x"], both["y"])[0, 1])


# ═════════════════════════════════════════════════════════════════════════════
# Fixtures: the verifier's round-2 generators, draw for draw
# ═════════════════════════════════════════════════════════════════════════════


def _adults(rng: np.random.Generator, n: int) -> dict:
    """round2/probes/fx_make.py ``adults``: energy from fat-free mass × PAL, reported with error;
    macronutrient shares drawn per person."""
    sex = rng.choice(["F", "M"], n)
    male = sex == "M"
    age = rng.integers(25, 70, n)
    height = np.where(male, rng.normal(176, 7, n), rng.normal(163, 6.5, n))
    pbf = np.where(male, rng.normal(24, 6, n), rng.normal(34, 6, n)).clip(8, 55)
    weight = np.where(male, rng.normal(84, 14, n), rng.normal(70, 14, n)).clip(45, 160)
    ffm = weight * (1 - pbf / 100)
    eer = (500 + 22 * ffm) * rng.uniform(1.4, 1.9, n)
    kcal = eer * np.exp(rng.normal(-0.1, 0.25, n))
    p_share = rng.normal(0.16, 0.03, n).clip(0.08, 0.3)
    f_share = rng.normal(0.34, 0.06, n).clip(0.15, 0.55)
    c_share = (1 - p_share - f_share - 0.03).clip(0.2, 0.7)
    return dict(sex=sex, age=age, height=height.round(1), weight=weight.round(1), pbf=pbf, ffm=ffm,
                kcal=kcal, protein_g=kcal * p_share / 4, fat_g=kcal * f_share / 9,
                carb_g=kcal * c_share / 4, f_share=f_share)


def _inbody() -> dict[str, pd.DataFrame]:
    """fx_make A (``default_rng(9101)``, n = 400): InBody 770 result names beside a food record,
    with an EHR CBC (``ALC``: absolute lymphocyte count, 10^9/L); and A2: the InBody columns first,
    ``PBF`` exported as ``Fat%``."""
    rng = np.random.default_rng(9101)
    a = _adults(rng, 400)
    n = 400
    inbody_protein = a["ffm"] * 0.73 * 0.265
    A = pd.DataFrame({
        "participant_id": np.arange(1, n + 1), "sex": a["sex"], "age": a["age"],
        "Weight": a["weight"], "Protein": inbody_protein.round(1),
        "Minerals": (a["ffm"] * 0.068).round(2),
        "Body Fat Mass": (a["weight"] * a["pbf"] / 100).round(1), "PBF": a["pbf"].round(1),
        "energy_kcal": a["kcal"].round(0), "protein_g": a["protein_g"].round(1),
        "carbohydrate_g": a["carb_g"].round(1), "fat_g": a["fat_g"].round(1),
        "ALC": np.exp(rng.normal(np.log(1.9), 0.35, n)).round(2),
        "ANC": np.exp(rng.normal(np.log(3.8), 0.4, n)).round(2),
        "hba1c": rng.normal(5.6, 0.5, n).round(1)})
    A2 = A[["participant_id", "Protein", "PBF", "sex", "age", "Weight", "energy_kcal", "protein_g",
            "carbohydrate_g", "fat_g", "ALC", "hba1c"]].rename(columns={"PBF": "Fat%"})
    return {"A": A, "A2": A2}


def _children_hologic() -> pd.DataFrame:
    """fx_make B (``default_rng(9102)``, n = 450): children 6–11 with an FFQ and Hologic Apex
    regional fat in grams and whole-body percent fat (the CLSA DSD's names)."""
    rng = np.random.default_rng(9102)
    n = 450
    age = rng.uniform(6, 11.9, n)
    sex = rng.choice(["F", "M"], n)
    weight = (20 + 3.3 * (age - 6)) * np.exp(rng.normal(0, 0.18, n))
    pbf = (np.where(sex == "F", 26, 22) + 18 * (np.log(weight) - np.log(20 + 3.3 * (age - 6)))
           + rng.normal(0, 4, n)).clip(10, 50)
    fat_mass_g = weight * pbf / 100 * 1000
    kcal = (900 + 45 * weight) * np.exp(rng.normal(0, 0.22, n))
    return pd.DataFrame({
        "child_id": np.arange(1, n + 1), "age_years": age.round(1), "sex": sex,
        "weight_kg": weight.round(1), "energy_kcal": kcal.round(0),
        "protein_g": (kcal * rng.normal(0.15, 0.02, n) / 4).round(1),
        "carbohydrate_g": (kcal * rng.normal(0.52, 0.05, n) / 4).round(1),
        "fat_g": (kcal * rng.normal(0.32, 0.04, n) / 9).round(1),
        "LARM_FAT": (fat_mass_g * 0.062 * np.exp(rng.normal(0, 0.1, n))).round(0),
        "RARM_FAT": (fat_mass_g * 0.060 * np.exp(rng.normal(0, 0.1, n))).round(0),
        "HEAD_FAT": (fat_mass_g * 0.09 * np.exp(rng.normal(0, 0.08, n))).round(0),
        "TRUNK_FAT": (fat_mass_g * 0.38 * np.exp(rng.normal(0, 0.1, n))).round(0),
        "WBTOT_FAT": fat_mass_g.round(0), "WBTOT_PFAT": pbf.round(1)})


def _liking() -> pd.DataFrame:
    """fx_make D (``default_rng(9104)``, n = 500): hedonic liking (0–100) and craving (1–5) scores
    built to rise with fat and carbohydrate intake, beside a diet record."""
    rng = np.random.default_rng(9104)
    a = _adults(rng, 500)
    n = 500

    def z(x):
        return (x - x.mean()) / x.std()

    return pd.DataFrame({
        "id": np.arange(1, n + 1), "sex": a["sex"], "age": a["age"],
        "energy_kcal": a["kcal"].round(0), "protein_g": a["protein_g"].round(1),
        "carbohydrate_g": a["carb_g"].round(1), "fat_g": a["fat_g"].round(1),
        "fat_liking": (55 + 8 * z(a["fat_g"]) + rng.normal(0, 14, n)).clip(0, 100).round(0),
        "carb_craving": (2.4 + 0.25 * z(a["carb_g"]) + rng.normal(0, 0.6, n)).clip(1, 5).round(2),
        "bmi": (a["weight"] / (a["height"] / 100) ** 2).round(1)})


def _case_control() -> pd.DataFrame:
    """round2/probes/ids.py (``default_rng(9921)``, n = 400): identifier words holding categories."""
    rng = np.random.default_rng(9921)
    n = 400
    return pd.DataFrame({"study_no": np.arange(1, n + 1), "patient": rng.choice([0, 1], n),
                         "respondent": rng.choice(["Self", "Proxy"], n, p=[0.85, 0.15]),
                         "participant": rng.choice(["yes", "no"], n),
                         "person": rng.choice(["mother", "father", "grandparent"], n),
                         "age": rng.integers(30, 80, n), "fiber_g": rng.normal(20, 6, n).round(1),
                         "crp": np.exp(rng.normal(0.5, 0.8, n)).round(2)})


def _adam_flags() -> pd.DataFrame:
    """round2/probes/roles2.py (``default_rng(9201)``, n = 360): EHR flags, imputed copies, CDISC
    ADaM arms, a crossover period and a cycle day, one row per subject."""
    rng = np.random.default_rng(9201)
    n = 360
    sbp = rng.normal(132, 16, n).round(0)
    sbp_obs = sbp.copy()
    sbp_obs[rng.random(n) < 0.15] = np.nan
    sbp_imp = np.where(np.isnan(sbp_obs), np.nanmean(sbp_obs) + rng.normal(0, 5, n), sbp_obs).round(1)
    return pd.DataFrame({
        "USUBJID": [f"STUDY01-{i:04d}" for i in range(n)], "SITEID": rng.integers(101, 109, n),
        "TRT01P": rng.choice(["Placebo", "Low fat diet"], n), "TRT01PN": rng.choice([1, 2], n),
        "ARMCD": rng.choice(["PBO", "LFD"], n),
        "diabetes_flag": rng.choice([0, 1], n, p=[0.8, 0.2]),
        "statin_flag": rng.choice([0, 1], n, p=[0.7, 0.3]),
        "current_smoker_flag": rng.choice([0, 1], n, p=[0.85, 0.15]),
        "sbp": sbp_obs, "sbp_imp": sbp_imp,
        "hba1c_imputed": rng.normal(5.8, 0.6, n).round(1),
        "imp_ldl": rng.normal(3.1, 0.8, n).round(2),
        "period": rng.choice([1, 2], n), "sequence": rng.choice(["AB", "BA"], n),
        "years_since_menopause": rng.integers(0, 25, n), "cycle_day": rng.integers(1, 29, n),
        "AVISITN": rng.choice([0, 4, 8, 12], n), "CHG": rng.normal(-3, 8, n).round(1),
        "ldl_wk12": rng.normal(3.0, 0.8, n).round(2)})


def _day_totals() -> dict[str, pd.DataFrame]:
    """round2/probes/kj_days.py (``default_rng(9301)``, n = 500): totals over the days the names
    say (``kcal_2d``, ``energy_kcal_7d``, ``kcal_4day_total``, ``kcal_sum_3d``), each value a day's
    intake times the days."""
    rng = np.random.default_rng(9301)
    n = 500
    sex = rng.choice(["Female", "Male"], n)
    day = np.where(sex == "Male", rng.normal(2350, 520, n), rng.normal(1850, 430, n)).clip(700, 4800)
    rng.integers(1_000_000, 6_000_000, n), rng.integers(40, 70, n)  # the UKB table's draws
    rng.normal(2800, 700, n), rng.normal(3.5, 0.8, n)
    out = {}
    for name, days in (("kcal_2d", 2), ("energy_kcal_7d", 7), ("kcal_4day_total", 4),
                       ("kcal_sum_3d", 3)):
        out[name] = pd.DataFrame({
            "id": np.arange(n), "sex": sex, "age": rng.integers(40, 70, n),
            name: (day * days * np.exp(rng.normal(0, 0.05, n))).round(0),
            "sodium_mg": rng.normal(2800, 700, n).round(0), "ldl": rng.normal(3.5, 0.8, n).round(2)})
    return out


def _nhanes_weights() -> dict[str, pd.DataFrame]:
    """round2/probes/weights.py (``default_rng(9401)``, n = 900): the tutorial's own pooled names
    (``MEC6YR``, ``SAF6YR``), the ``WT``-prefixed pooled names, and lowercase 2-year names, each
    with a fasting weight that is zero off the fasting subsample (as GLU_J's is) and ``LBXGLU``."""
    rng = np.random.default_rng(9401)
    n = 900
    fasting = rng.random(n) < 0.45
    mec = rng.uniform(5000, 90000, n)
    base = pd.DataFrame({
        "SEQN": np.arange(83732, 83732 + n), "SDDSRVYR": rng.choice([8, 9, 10], n),
        "SDMVPSU": rng.choice([1, 2], n), "SDMVSTRA": rng.integers(119, 148, n),
        "RIAGENDR": rng.choice([1, 2], n), "RIDAGEYR": rng.integers(20, 80, n),
        "DR1TKCAL": rng.normal(2100, 600, n).clip(600).round(0),
        "DR1TSFAT": rng.normal(26, 9, n).clip(3).round(1)})
    glucose = np.where(fasting, rng.normal(102, 18, n).round(0), np.nan)
    variants = {
        "W1": dict(MEC6YR=(mec / 3).round(1), SAF6YR=np.where(fasting, mec * 2.1 / 3, 0).round(1)),
        "W2": dict(WTMEC6YR=(mec / 3).round(1),
                   WTSAF6YR=np.where(fasting, mec * 2.1 / 3, 0).round(1),
                   WTDR6YR=(mec * 1.1 / 3).round(1)),
        "W3": dict(wtmec2yr=mec.round(1), wtsaf2yr=np.where(fasting, mec * 2.1, 0).round(1),
                   wtdrd1=(mec * 1.1).round(1)),
    }
    out = {}
    for label, extra in variants.items():
        df = base.copy()
        for k, v in extra.items():
            df[k] = v
        df["LBXGLU"] = glucose
        if label == "W3":
            df.columns = [c.lower() for c in df.columns]
        out[label] = df
    return out


def _visits_dob_first() -> pd.DataFrame:
    """round2/probes/repeats_dob.py (``default_rng(9701)``): 150 people × 4 visits 91 days apart,
    dated by ``visit_date``; ``dob`` and ``randomization_date`` the same on every row of a person."""
    rng = np.random.default_rng(9701)
    rows = []
    for pid in range(150):
        dob = pd.Timestamp("1950-01-01") + pd.Timedelta(days=int(rng.integers(0, 15000)))
        rand = pd.Timestamp("2019-01-01") + pd.Timedelta(days=int(rng.integers(0, 365)))
        base_ldl = rng.normal(3.3, 0.7)
        for v in range(4):
            rows.append({"pid": f"P{pid:03d}", "dob": dob.date().isoformat(),
                         "randomization_date": rand.date().isoformat(),
                         "visit_date": (rand + pd.Timedelta(days=91 * v + int(rng.integers(-7, 8)))
                                        ).date().isoformat(),
                         "visit": v + 1, "ldl": round(base_ldl + rng.normal(0, 0.3), 2),
                         "fiber_g": round(rng.normal(22, 6), 1)})
    return pd.DataFrame(rows)


def _two_waves() -> pd.DataFrame:
    """round2/probes/repeats_wave.py (``default_rng(9702)``): two recalls at each of two waves of a
    two-arm trial, numbered ``recall_no`` 1–4, a ``wave`` column naming the occasion, no dates; the
    low-energy arm cuts energy by 400 kcal at month 6."""
    rng = np.random.default_rng(9702)
    rows = []
    for u in range(200):
        arm = ["control", "low_energy"][u % 2]
        usual = rng.normal(2100, 400)
        for r in range(4):
            wave = "baseline" if r < 2 else "month6"
            eff = -400 if (arm == "low_energy" and wave == "month6") else 0
            k = usual + eff + rng.normal(0, 450)
            rows.append({"participant_id": u, "arm": arm, "wave": wave, "recall_no": r + 1,
                         "energy_kcal": round(k), "fat_g": round(k * 0.35 / 9, 1)})
    return pd.DataFrame(rows)


def _ffq_grams_and_peak_areas() -> dict[str, pd.DataFrame]:
    """round2/probes/hints.py (``default_rng(9801)``): an FFQ in whole grams per food (130 items,
    intakes from single grams to hundreds), and a Metabolon-style table of integer peak areas."""
    rng = np.random.default_rng(9801)
    n = 300
    foods = ["tea", "coffee", "water", "whole_milk", "semi_skimmed_milk", "white_bread",
             "brown_bread", "porridge", "breakfast_cereal", "rice", "pasta", "boiled_potatoes",
             "chips", "beef", "pork", "lamb", "chicken", "bacon", "ham", "sausages", "white_fish",
             "oily_fish", "eggs", "cheese", "yogurt", "butter", "margarine", "olive_oil", "apples",
             "bananas", "oranges", "grapes", "strawberries", "carrots", "broccoli", "peas",
             "green_beans", "cabbage", "onions", "garlic", "tomatoes", "lettuce", "cucumber",
             "peppers", "mushrooms", "baked_beans", "lentils", "nuts", "crisps", "chocolate",
             "biscuits", "cake", "ice_cream", "sugar_added", "jam", "honey", "fruit_juice", "cola",
             "diet_cola", "beer", "wine", "spirits", "soup", "pizza", "pies", "salt_added", "pepper",
             "herbs", "ketchup", "mayonnaise"]
    foods = foods + [f"item_{i:03d}" for i in range(130 - len(foods))]
    scale = np.exp(rng.uniform(np.log(0.5), np.log(900), len(foods)))
    scale[:3] = [550, 300, 800]
    # Columns are drawn in the probe's order and assembled once (no fragmented inserts).
    F = {"idno": np.arange(n), "age": rng.integers(40, 75, n), "sex": rng.choice([1, 2], n)}
    for f, s in zip(foods, scale):
        eaters = rng.random(n) < 0.8
        F[f"{f}_g"] = np.where(eaters, np.round(s * np.exp(rng.normal(0, 0.7, n))), 0).astype(int)
    F["energy_kcal"] = rng.normal(2000, 450, n).round(0)
    names = [f"compound_{i}" for i in range(400)]
    names[:10] = ["glucose", "lactate", "alanine", "glycine", "citrate", "urate", "creatinine",
                  "cholesterol", "palmitate", "carnitine"]
    M = {"PARENT_SAMPLE_NAME": [f"SAMP{i:04d}" for i in range(120)],
         "GROUP": rng.choice(["Control", "Case"], 120)}
    for c in names:
        M[c] = np.round(np.exp(rng.normal(rng.uniform(np.log(1e4), np.log(1e9)), 0.6, 120))
                        ).astype(np.int64)
    return {"ffq_grams": pd.DataFrame(F), "peak_areas": pd.DataFrame(M)}


# ═════════════════════════════════════════════════════════════════════════════
# A · The gate's open items and unasked assumptions, replayed
# ═════════════════════════════════════════════════════════════════════════════


def test_a1_names_the_values_do_not_corroborate_are_never_high_and_never_prefill_the_energy_card(
        tmp_path):
    """Gate WP13-1 / unasked assumption 1: under the dietary lens ``ALC`` (a lymphocyte count, its
    INFOODS whole-name match skipping the value checks), a BIA ``Fat%``, InBody body ``Protein``,
    children's Hologic ``LARM_FAT``/``RARM_FAT``, ``WBTOT_PFAT`` read as PUFA, ``fat_liking`` and
    ``carb_craving`` were each "exposure (high): A nutrient that carries energy" and entered the
    energy card's nutrients. Expected, from NumPy on the fixtures: ALC's r with energy is under 0.3;
    ``Fat%`` is a percentage; InBody Protein's Atwater reconstruction strays from energy further
    than protein_g's; the arm fats carry more energy than the child's day on most rows;
    ``WBTOT_PFAT``'s r is under 0.3; the liking and craving names carry words no intake's name does.
    None is high, each needs its own confirmation, and none is a default nutrient, whether the
    roles are proposed or bulk-confirmed."""
    from turbotab.core.recognizers import CORROBORATE_R, read_nutrient, tokens

    assert CORROBORATE_R == 0.3 and BLUEPRINT_14.startswith("A recognizer may be wrong")
    ib = _inbody()
    A, A2, B, D = ib["A"], ib["A2"], _children_hologic(), _liking()
    # Independent evidence, computed here.
    assert abs(_r(A["ALC"], A["energy_kcal"])) < 0.3
    assert "pct" in tokens("Fat%")
    dev = {}
    for protein in ("Protein", "protein_g"):
        recon = 4 * A[protein] + 4 * A["carbohydrate_g"] + 9 * A["fat_g"]
        dev[protein] = float(np.median(np.abs(np.log(A["energy_kcal"] / recon))))
    assert dev["Protein"] > dev["protein_g"] + 0.02, dev
    for arm in ("LARM_FAT", "RARM_FAT"):
        assert float((B[arm] * ATWATER["fat"] > B["energy_kcal"]).mean()) > 0.5
    assert _r(B["WBTOT_PFAT"], B["energy_kcal"]) < 0.3
    assert set(read_nutrient("fat_liking").foreign) == {"liking"}
    assert set(read_nutrient("carb_craving").foreign) == {"craving"}
    # InBody's body protein with no recall protein beside it, in kg and in lb (US exports): under 5%
    # of the day's energy as protein, the lowest acceptable range of any age group (IOM: 5–20% at
    # 1–3 years), so it is no protein intake whatever its r with energy.
    for unit, factor in (("kg", 1.0), ("lb", 2.20462)):
        alone = A.drop(columns=["protein_g"]).assign(Protein=(A["Protein"] * factor).round(1))
        share = float(alone["Protein"].median()) * ATWATER["protein"] / float(
            alone["energy_kcal"].median())
        assert share < 0.05 and _r(alone["Protein"], alone["energy_kcal"]) >= 0.3, (unit, share)
        path = _write(alone, tmp_path, f"inbody_alone_{unit}.csv")
        p = _roles(path, lens=["dietary"], target="hba1c")["Protein"]
        assert (p["proposed"], p["confidence"]) == ("covariate", "low"), (unit, p)
        assert "under 5% of a day's energy" in p["reason"], p
    cases = [("A", A, "hba1c", ["ALC", "Protein"]), ("A2", A2, "hba1c", ["ALC", "Protein", "Fat%"]),
             ("B", B, "WBTOT_PFAT", ["LARM_FAT", "RARM_FAT"]),
             ("B_fat", B.drop(columns=["WBTOT_FAT"]), "age_years",
              ["LARM_FAT", "RARM_FAT", "WBTOT_PFAT"]),
             ("D", D, "bmi", ["fat_liking", "carb_craving"])]
    for label, frame, target, wrong in cases:
        path = _write(frame, tmp_path, f"{label}.csv")
        artifact = _roles_artifact(path, lens=["dietary"], target=target)
        roles = {p["column"]: p for p in artifact["columns"]}
        for c in wrong:
            p = roles[c]
            assert p["confidence"] != "high", (label, c, p)
            assert p["attention"] is True and c in artifact["needs_confirmation"], (label, c)
        # The real intakes are high, by their values.
        for c in ("protein_g", "carbohydrate_g", "fat_g"):
            assert roles[c]["confidence"] == "high", (label, c, roles[c])
        # Proposed (nothing recorded yet) and after a bulk confirm, through the proposals stage.
        for state in (ProjectState(lens=["dietary"], target=target, purpose="inference"),
                      _bulk_state(artifact, lens=["dietary"], target=target)):
            nutrients = _proposals_stage(path, state, artifact)["energy"]["nutrients"]
            assert nutrients == ["protein_g", "carbohydrate_g", "fat_g"], (label, nutrients)


def test_a2_identifier_flag_and_time_words_need_their_values(tmp_path):
    """Gate WP13-2 / unasked assumptions 2–3: a case–control ``patient`` 0/1, ``participant``
    yes/no, ``respondent`` Self/Proxy and ``person`` mother/father/grandparent were "identifier
    (high)" and left the model; ``sbp_imp`` (continuous imputed SBP) was "flag (high)" and EHR
    ``diabetes_flag``/``statin_flag`` "flag (medium)"; a crossover ``period`` and a ``cycle_day`` on
    one row per subject were "time". Expected, from pandas' counts: two values, or a few words,
    are a category's labels, never an identifier; continuous values are never a flag, and a yes/no
    with no column to mark is a characteristic; on one row per unit nothing varies within units, so
    no time-named column is time. Each stays a predictor, below high."""
    from turbotab.core.stages.rows import PREDICTOR_ROLES

    cc = _case_control()
    assert {c: cc[c].nunique() for c in ("patient", "respondent", "participant", "person")} == {
        "patient": 2, "respondent": 2, "participant": 2, "person": 3}
    assert cc["study_no"].nunique() == len(cc)
    for lens in (["clinical"], ["dietary"]):
        roles = _roles(_write(cc, tmp_path, "cc.csv"), lens=lens, target="crp")
        for c in ("patient", "respondent", "participant", "person"):
            assert roles[c]["proposed"] in PREDICTOR_ROLES and roles[c]["confidence"] == "low", (
                lens, roles[c])
            assert "a category's labels, not units" in roles[c]["reason"], roles[c]
        assert (roles["study_no"]["proposed"], roles["study_no"]["confidence"]) == ("identifier", "high")
    adam = _adam_flags()
    assert adam["sbp_imp"].nunique() > 2 and adam["USUBJID"].is_unique
    for lens in (["clinical"], ["dietary"], None):
        roles = _roles(_write(adam, tmp_path, "adam.csv"), lens=lens, target="ldl_wk12")
        for c in ("sbp_imp", "hba1c_imputed", "imp_ldl", "diabetes_flag", "statin_flag",
                  "current_smoker_flag", "period", "cycle_day"):
            assert roles[c]["proposed"] in PREDICTOR_ROLES, (lens, c, roles[c])
            assert roles[c]["confidence"] != "high", (lens, c, roles[c])
        for c in ("sbp_imp", "hba1c_imputed", "imp_ldl"):
            assert f"`{adam[c].nunique():,}` different values" in roles[c]["reason"], roles[c]
    # A true flag: binary, and exactly where its base is blank (pandas).
    marked = adam.drop(columns=["sbp_imp"]).assign(sbp_flag=adam["sbp"].isna().astype(int))
    roles = _roles(_write(marked, tmp_path, "marked.csv"), lens=["clinical"], target="ldl_wk12")
    assert (roles["sbp_flag"]["proposed"], roles["sbp_flag"]["confidence"],
            roles["sbp_flag"]["linked_to"]) == ("flag", "high", "sbp")
    # The same flag shuffled is no longer tied to the blanks: below high.
    shuffled = marked.assign(sbp_flag=np.random.default_rng(1).permutation(marked["sbp_flag"]))
    roles = _roles(_write(shuffled, tmp_path, "shuffled.csv"), lens=["clinical"], target="ldl_wk12")
    assert roles["sbp_flag"]["confidence"] != "high", roles["sbp_flag"]


def test_a2_the_case_control_inference_never_clusters_by_a_category(tmp_path):
    """Gate WP13-2, through the real server as the verifier drove it: roles confirmed exactly as
    proposed, the grain answered one row per unit by ``study_no``, inference with the linear model.
    The fit printed "No intervals: `participant` has 2 units … No interval or p-value is reported".
    Expected: the table reports intervals, clustering by no category (NumPy: ``study_no`` is unique,
    so no unit repeats)."""
    from turbotab.core.tests.acceptance.server_drive import local_server, open_project

    frame = _case_control()
    path = _write(frame, tmp_path, "cc.csv")
    with local_server(tmp_path / "home") as client:
        drive = open_project(client, path)
        _drive_to_fit(drive, lens=["clinical"], target="crp",
                      grain={"kind": "set_grain", "grain": "one_row_per_unit",
                             "id_column": "study_no"})
        fit = drive.artifact("fit")
        model = fit["models"][0]
        info = model.get("inference") or {}
        caption = str(info.get("caption") or "")
        assert "No intervals" not in caption, caption
        assert "`participant`" not in caption and "`patient`" not in caption, caption
        assert "95%" in caption, caption


def test_a3_a_day_count_in_a_total_energy_name_is_asked(tmp_path):
    """Gate WP13-5 / unasked assumption 6: ``kcal_2d`` (a 2-day total, median 4,021) was settled by
    name as one day's intake; the 500–5,000 screen was offered unrefused (111 of 500 rows), the coach
    said "likely over-reporting" and the finding "111 records report an implausible daily intake".
    Expected: each day count (``2d``, ``7d``, ``4day``, ``3d``) is read as a question, every screen
    is refused until the days are recorded, the finding counts under both readings (NumPy's counts),
    the coach calls no one an over-reporter, and the refusal's exits record the total or the mean."""
    from turbotab.core.recognizers import day_count
    from turbotab.core.stages.findings import findings_stage
    from turbotab.core.stages.proposals import build_proposals

    totals = _day_totals()
    for name, days in (("kcal_2d", 2), ("energy_kcal_7d", 7), ("kcal_4day_total", 4),
                       ("kcal_sum_3d", 3)):
        assert day_count(name) == days
        frame = totals[name]
        v = frame[name]
        one_day = int(((v < 500) | (v > 5000)).sum())
        spanned = int(((v < 500 * days) | (v > 5000 * days)).sum())
        path = _write(frame, tmp_path, f"{name}.csv")
        t = _table(path)
        roles = {c: p["proposed"] for c, p in _roles(path, lens=["dietary"], target="ldl").items()}
        assert roles[name] == "energy", (name, roles)
        out = build_proposals(t.frame(), t.info["columns"], lens=["dietary"], target="ldl",
                              roles=roles)
        reading = out["energy_unit"]
        assert (reading["confirmed"], reading["days_in_name"]) == (False, days), reading
        assert out["exclusions"] and all(e["refused"] for e in out["exclusions"]), name
        assert "over-reporting" not in str(out["coach"]) and "under-reporting" not in str(out["coach"])
        found = t.run(findings_stage, ProjectState(lens=["dietary"], target="ldl"))["findings"]
        f = next(x for x in found if x["id"] == "pack::dietary::implausible_intake"
                 or x["title"].startswith("The days in"))
        assert f["title"].rstrip(".") == f"The days in `{name}` are not settled", f["title"]
        assert f"{spanned:,} of {len(frame):,} rows" in f["detail"], f["detail"]
        assert f"read as one day's intake, {one_day:,} fall outside" in f["detail"], f["detail"]
        assert "implausible daily intake" not in f["title"]
        # The screen's refusal, as the server's validator gives it, offers both readings.
        state = ProjectState(lens=["dietary"], target="ldl", roles=roles)
        ctx = {"state": state, "columns": list(frame.columns), "target": "ldl",
               "store": lambda t=t: t.store()}
        rule = {"column": name, "low": 500, "high": 5000, "reason": "implausible intakes"}
        with pytest.raises(d.Refusal) as refused:
            d.validate({"kind": "set_exclusions", "rules": [rule]}, ctx)
        assert refused.value.code == "energy_unit_unconfirmed"
        exits = [x["decision"] for x in refused.value.exits]
        assert {"kind": "set_column_unit", "column": name, "unit": "kcal", "days": days} in exits
        assert {"kind": "set_column_unit", "column": name, "unit": "kcal", "days": 1} in exits


def test_a4_subsample_weights_with_zeros_and_pooled_names_follow_the_least_common_denominator(
        tmp_path):
    """Gate WP13-6: the CDC GLU_J WTSAF2YR has 325 zeros (source check below), and the real fasting
    weight was proposed "covariate (low)" for them; the tutorial's own pooled ``MEC6YR``/``SAF6YR``
    were covariates and the survey question offered only "these participants"; with ``WTMEC6YR``,
    ``WTSAF6YR`` and ``WTDR6YR`` beside ``LBXGLU`` the finding said the fasting weight was missing
    ("needs `WTSAF2YR`") and ``WTDR6YR`` was offered first. Expected: zeros are no bar to a weight;
    each pooled name is read beside the design; the least common denominator names the fasting
    weight (``LBXGLU`` recorded only where it is positive, by pandas) and ranks fasting, then
    examination, then dietary."""
    from turbotab.core.recognizers import least_common_denominator, rank_weights
    from turbotab.core.survey import offered

    assert GLU_J_WTSAF2YR == "Fasting Subsample 2 Year MEC Weight" and GLU_J_ZERO_ROW[2] == 325
    assert TUTORIAL_MEC6YR.endswith("MEC6YR = 1/3 * WTMEC2YR;") and "least common" in TUTORIAL_LCD
    expected = {"W1": ("SAF6YR", ["SAF6YR", "MEC6YR"]),
                "W2": ("WTSAF6YR", ["WTSAF6YR", "WTMEC6YR", "WTDR6YR"]),
                "W3": ("wtsaf2yr", ["wtsaf2yr", "wtmec2yr", "wtdrd1"])}
    for label, frame in _nhanes_weights().items():
        fasting_weight, order = expected[label]
        glucose = "LBXGLU" if label != "W3" else "lbxglu"
        inside = frame[fasting_weight] > 0
        assert (frame[fasting_weight] == 0).sum() > 0 and not (frame[glucose].notna() & ~inside).any()
        path = _write(frame, tmp_path, f"{label}.csv")
        artifact = _roles_artifact(path, lens=["dietary"], target=glucose)
        roles = {p["column"]: p for p in artifact["columns"]}
        for w in order:
            assert roles[w]["proposed"] == "design", (label, w, roles[w])
        lcd = least_common_denominator(list(frame.columns), frame)
        assert (lcd["use"], lcd["missing"]) == (fasting_weight, None), lcd
        assert rank_weights(order[::-1], lcd) == order
        # The survey question, read on the bulk-confirmed roles: fasting first, dietary last.
        state = _bulk_state(artifact, lens=["dietary"], target=glucose)
        options = [o["decision"].get("weight") for o in offered(state, None, frame)
                   if o["decision"].get("weight")]
        assert options == order, (label, options)
        # The finding no longer asks for a weight the table has.
        from turbotab.core.stages.findings import findings_stage

        found = _table(path).run(findings_stage, ProjectState(lens=["dietary"], target=glucose))
        for f in found["findings"]:
            if f["id"] == "pack::dietary::survey_weights":
                assert "is not in this table" not in f["detail"], f["detail"]
                assert f"`{fasting_weight}`" in f["title"], f["title"]


def test_a5_a_date_the_same_on_every_row_of_a_unit_is_never_repeats_evidence(tmp_path):
    """Gate WP14-6 / unasked assumption 4: the legacy spacing reader kept the first date column on a
    tie, so a visit table whose first date is ``dob`` or ``randomization_date`` was stated "repeats"
    ("every one of a unit's records carries the same date in `dob`") and its mean recommended; and
    recalls numbered 1–4 across a ``wave`` naming baseline and month 6 were stated repeats, a mean
    that erases the low-energy arm's fall (NumPy: 2,091 → 1,631 kcal). Expected: pandas finds
    ``dob`` and ``randomization_date`` constant within every person and ``visit_date`` varying, so
    the spacing is read from ``visit_date``; the two-wave recalls are asked, naming ``wave``."""
    from turbotab.core.detectors import repeats as reading
    from turbotab.core.stages.working import structure_stage

    visits = _visits_dob_first()
    per = visits.groupby("pid").nunique()
    assert (per["dob"] == 1).all() and (per["randomization_date"] == 1).all()
    assert (per["visit_date"] == 4).all()
    for lens in (["clinical"], ["dietary"]):
        r = reading.read(visits, "pid", lens)
        assert (r.get("spacing") or {}).get("column") == "visit_date", r
        assert "carries the same date" not in r["sentence"], r["sentence"]
    path = _write(visits[["pid", "randomization_date", "visit_date", "ldl", "fiber_g"]], tmp_path,
                  "rand.csv")
    out = _table(path).run(structure_stage, ProjectState(
        lens=["clinical"], grain=GrainSpec(grain="repeated", id_column="pid")))
    assert (out["repeats"]["spacing"] or {}).get("column") == "visit_date", out["repeats"]
    waves = _two_waves()
    means = waves.groupby(["arm", "wave"])["energy_kcal"].mean().round()
    assert means[("low_energy", "baseline")] - means[("low_energy", "month6")] > 300
    assert (waves.groupby("participant_id")["wave"].nunique() == 2).all()
    r = reading.read(waves, "participant_id", ["dietary"])
    assert (r["reading"], r["stated"]) == (None, False), r["sentence"]
    assert "`wave` names each record's occasion" in r["sentence"], r["sentence"]


def test_a6_lens_hints_need_positive_evidence(tmp_path):
    """Gate WP14-7: an FFQ in whole grams per food (spread 1.4 decades) was hinted genomics and drew
    the critical lens contradiction under its own dietary lens; Metabolon-style integer peak areas
    (2.3 decades) were hinted genomics. Expected: every FFQ block column is named in grams, and no
    peak area is 10 or under (NumPy), so neither is a count matrix; no genomics hint, and no
    contradiction under the dietary lens."""
    from turbotab.core.detectors import lenses

    fx = _ffq_grams_and_peak_areas()
    ffq, peaks = fx["ffq_grams"], fx["peak_areas"]
    grams = [c for c in ffq.columns if c.endswith("_g")]
    assert len(grams) == 130
    cells = peaks.drop(columns=["PARENT_SAMPLE_NAME", "GROUP"]).to_numpy(float)
    assert float((cells <= 10).mean()) == 0.0
    for frame in (ffq, peaks):
        assert "genomics" not in [h["lens"] for h in lenses.hints(frame)]
    assert lenses.contradiction_finding(ffq, ["dietary"]) is None


def test_a7_an_outcome_read_as_a_nutrient_by_name_alone_keeps_the_dispute(tmp_path):
    """Gate WP15-3 / unasked assumption 5: Hologic ``WBTOT_PFAT``/``WBTOT_FAT``/``SUBTOT_FAT``, BIA
    ``Fat%`` and ``fat_kg`` outcomes were read as nutrients, so ``outcome_relation`` said "other",
    no dispute was stated and the energy finding stayed CONVENTION. Expected: a nutrient reading of
    the outcome counts only where its values corroborate it (none of these does: NumPy r with energy
    under 0.3 for the random draws of the verifier's probe); the dispute is stated as a condition;
    and on the children's table with only the diet record and the outcome, the energy card carries
    the DISPUTED line."""
    from turbotab.core.methods.dietary_caveats import outcome_relation

    rng = np.random.default_rng(9901)
    n = 300
    base = pd.DataFrame({"energy_kcal": rng.normal(2000, 400, n), "protein_g": rng.normal(80, 20, n),
                         "fat_g": rng.normal(75, 20, n), "carbohydrate_g": rng.normal(250, 60, n)})
    for name in ("WBTOT_PFAT", "WBTOT_FAT", "Fat%", "fat_kg", "SUBTOT_FAT"):
        df = base.copy()
        df[name] = rng.normal(30, 6, n)
        assert abs(_r(df[name], df["energy_kcal"])) < 0.3
        relation = outcome_relation(name, df, energy="energy_kcal")
        assert relation["basis"] != "other", (name, relation)
    B = _children_hologic()
    for target in ("WBTOT_PFAT", "WBTOT_FAT"):
        keep = ["child_id", "age_years", "sex", "energy_kcal", "protein_g", "carbohydrate_g",
                "fat_g", target]
        path = _write(B[keep], tmp_path, f"B2_{target}.csv")
        artifact = _roles_artifact(path, lens=["dietary"], target=target)
        state = _bulk_state(artifact, lens=["dietary"], target=target)
        energy = _proposals_stage(path, state, artifact)["energy"]
        dispute = energy["outcome_dispute"]
        assert dispute is not None and dispute["evidence"]["status"] == "DISPUTED", (target, energy)


# ═════════════════════════════════════════════════════════════════════════════
# B · The leash's mechanics, through the real server
# ═════════════════════════════════════════════════════════════════════════════

ORDER = ["lens", "orientation", "target", "event", "task", "purpose", "grain", "repeat_kind",
         "unit", "aggregation", "temporal", "roles", "survey", "exclusions", "missing", "split",
         "energy_adjustment", "models"]


def _drive_to_fit(drive, *, lens, target, grain, energy=None) -> None:
    """Answer the Router in order as the verifier did, the roles confirmed exactly as proposed
    (the roles card's one-click "Confirm the roles of N columns")."""
    answers = {
        "lens": {"kind": "set_lens", "lenses": lens},
        "target": {"kind": "set_target", "column": target},
        "task": {"kind": "set_task", "column": target, "task": "regression"},
        "purpose": {"kind": "set_purpose", "purpose": "inference"},
        "grain": grain, "temporal": {"kind": "set_temporal", "temporal": False},
        "survey": {"kind": "set_survey", "estimand": "sample"},
        "exclusions": {"kind": "set_exclusions", "rules": []},
        "missing": {"kind": "set_missing", "strategy": "complete_case"},
        "split": {"kind": "set_split", "holdout": 0.2, "seed": 0, "folds": 5},
        "models": {"kind": "select_models", "models": ["linear"]},
    }
    for key in ORDER:
        step = drive.reach(key, timeout=240)
        if step["status"] not in ("open", "waiting"):
            continue
        if key == "roles":
            roles = drive.artifact("roles")
            body = {"kind": "set_roles", "roles": {c["column"]: c["proposed"]
                                                   for c in roles["columns"]}}
        elif key == "energy_adjustment":
            body = energy(drive) if energy else {"kind": "set_energy_adjustment", "method": "none"}
        else:
            body = answers[key]
        drive.decide(body)


def test_b1_a_bulk_confirm_records_what_rode_along_and_only_a_confirmation_settles_it(tmp_path):
    """BLUEPRINT §14 rule 2, through the real server on the gate's InBody table (A2). The verifier
    confirmed the roles exactly as proposed and sent the energy decision as ChoiceQuestions.tsx
    builds it; ``Protein``, ``Fat%`` and ``ALC`` were adjusted as nutrients "in place of other
    energy", and the measurement-error line called ``Protein`` and ``Fat%`` "self-reported
    intakes". Expected:

    * the roles payload marks every proposal below high (``attention``) and lists them;
    * the recorded ``set_roles`` carries, as the server read them, exactly those proposals recorded
      as proposed, whatever the client sent;
    * the energy card's nutrients are the three gram totals; an energy decision naming ``ALC`` is
      refused (409) with one exit per unsettled column (a ``confirm_role`` each) and none that
      confirms them together;
    * ``confirm_role`` for ``ALC`` settles it, and the same decision is then recorded; undoing the
      confirmation is refused while the adjustment reads ``ALC``, and once the adjustment changes
      the undo unsettles it again;
    * the fit's measurement-error line names no unsettled column as a self-reported intake."""
    from turbotab.core.tests.acceptance.server_drive import local_server, open_project

    frame = _inbody()["A2"]
    path = _write(frame, tmp_path, "A2.csv")

    def as_the_card_sends(drive) -> dict:
        e = drive.artifact("proposals")["energy"]
        assert e["nutrients"] == ["protein_g", "carbohydrate_g", "fat_g"], e["nutrients"]
        assert "ALC" in e["unconfirmed"], e["unconfirmed"]
        left = {x["column"]: x["reason"] for x in e["not_adjusted"]}
        assert left["ALC"] == "proposed below high confidence; confirm its role to adjust it"
        return {"kind": "set_energy_adjustment", "method": "standard",
                "energy_column": e["energy_column"], "nutrients": e["nutrients"],
                "log_transform": False, "strata": None}

    with local_server(tmp_path / "home") as client:
        drive = open_project(client, path)
        _drive_to_fit(drive, lens=["dietary"], target="hba1c",
                      grain={"kind": "set_grain", "grain": "one_row_per_unit",
                             "id_column": "participant_id"},
                      energy=as_the_card_sends)
        roles = drive.artifact("roles")
        below = [p["column"] for p in roles["columns"] if p["confidence"] != "high"]
        assert all(p["attention"] == (p["confidence"] != "high") for p in roles["columns"])
        assert roles["needs_confirmation"] == below
        record = next(r for r in drive.view()["decisions"] if r["decision"]["kind"] == "set_roles")
        assert record["decision"]["unconfirmed"] == below
        assert {"Protein", "ALC"} <= set(below)
        # A client that claims nothing rode along is overruled by the server.
        claimed = {"kind": "set_roles", "unconfirmed": [],
                   "roles": {c["column"]: c["proposed"] for c in roles["columns"]}}
        drive.decide(claimed)
        assert drive.view()["decisions"][-1]["decision"]["unconfirmed"] == below
        e = drive.artifact("proposals")["energy"]
        body = {"kind": "set_energy_adjustment", "method": "standard",
                "energy_column": e["energy_column"],
                "nutrients": [*e["nutrients"], "ALC"], "log_transform": False, "strata": None}
        r = drive.post(body)
        assert r.status_code == 409, r.text
        error = r.json()["error"]
        assert error["code"] == "role_unconfirmed"
        confirms = [x["decision"] for x in error["exits"]
                    if (x["decision"] or {}).get("kind") == "confirm_role"]
        assert confirms == [{"kind": "confirm_role", "column": "ALC", "role": "exposure"}]
        drive.decide({"kind": "confirm_role", "column": "ALC", "role": "exposure"})
        confirmation = drive.view()["decisions"][-1]
        drive.decide(body)
        assert drive.view()["state"]["energy_adjustment"]["nutrients"][-1] == "ALC"
        # Undoing the confirmation while the energy adjustment reads ALC would leave a default on a
        # role nobody confirmed: refused until the adjustment changes (never kept silently).
        undo = {"kind": "revert", "decision_id": confirmation["id"]}
        r = drive.post(undo)
        assert r.status_code == 409 and r.json()["error"]["code"] == "role_unconfirmed", r.text
        assert "energy adjustment" in r.json()["error"]["message"]
        drive.decide({"kind": "set_energy_adjustment", "method": "standard",
                      "energy_column": e["energy_column"], "nutrients": e["nutrients"],
                      "log_transform": False, "strata": None})
        drive.decide(undo)
        assert drive.view()["state"]["role_confirmations"] in (None, {})
        r = drive.post(body)
        assert r.status_code == 409 and r.json()["error"]["code"] == "role_unconfirmed"
        # The fit: its intake line names settled columns only.
        fit = drive.artifact("fit")
        text = str(fit)
        assert "self-reported" in text
        for column in ("Protein", "Fat%", "ALC"):
            assert f"`{column}`" not in _intake_line(fit), (column, _intake_line(fit))


def _intake_line(fit: dict) -> str:
    """The fit's measurement-error limitation, wherever the artifact carries it."""
    found = []

    def walk(x):
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, str) and "self-reported" in x:
            found.append(x)

    walk(fit)
    return " ".join(found)


def test_b2_number_changing_decisions_on_unsettled_roles_are_refused_one_column_at_a_time():
    """BLUEPRINT §14 rule 2 at the validators, on a state a bulk confirm left: an exclusion on an
    unsettled energy column, a survey weight that rode along, and an energy adjustment naming two
    unsettled nutrients are refused with one ``confirm_role`` exit per column; the same decisions on
    settled roles pass these checks; a confirmation of the outcome, or of a column the table lacks,
    is refused."""
    roles = {"id": "identifier", "energy": "energy", "protein_g": "exposure", "fat_g": "exposure",
             "carb_g": "exposure", "ALC": "exposure", "Protein": "exposure", "wt": "design",
             "strata": "design", "psu": "design", "age": "covariate"}
    columns = [*roles, "y"]
    state = ProjectState(lens=["dietary"], target="y", purpose="inference", roles=roles,
                         roles_unconfirmed=["energy", "ALC", "Protein", "wt"])
    ctx = {"state": state, "columns": columns, "target": "y"}
    cases = [
        ({"kind": "set_exclusions", "rules": [{"column": "energy", "low": 500, "high": 5000,
                                               "reason": "implausible intakes"}]}, ["energy"]),
        ({"kind": "set_survey", "estimand": "population", "weight": "wt", "strata": "strata",
          "psu": "psu"}, ["wt"]),
    ]
    for decision, waiting in cases:
        with pytest.raises(d.Refusal) as refused:
            d.validate(decision, ctx)
        assert refused.value.code == "role_unconfirmed", decision
        confirms = [x["decision"]["column"] for x in refused.value.exits
                    if (x["decision"] or {}).get("kind") == "confirm_role"]
        assert confirms == waiting
    settled = state.model_copy(update={"role_confirmations": {"energy": "energy", "ALC": "exposure",
                                                              "Protein": "exposure",
                                                              "wt": "design"}})
    from turbotab.core.leash import unsettled

    assert unsettled(state) == ["energy", "ALC", "Protein", "wt"]
    assert unsettled(settled) == []
    # A confirmation for another role than the recorded one does not settle it.
    other = state.model_copy(update={"role_confirmations": {"ALC": "covariate"}})
    assert "ALC" in unsettled(other)
    # A role the user changed from the proposal is the user's own answer.
    from turbotab.core.leash import rode_along

    proposals = [{"column": "ALC", "proposed": "exposure", "confidence": "medium"},
                 {"column": "Protein", "proposed": "exposure", "confidence": "medium"},
                 {"column": "fat_g", "proposed": "exposure", "confidence": "high"}]
    assert rode_along({"ALC": "exposure", "Protein": "covariate", "fat_g": "exposure"},
                      proposals) == ["ALC"]
    for bad in ({"kind": "confirm_role", "column": "y", "role": "exposure"},
                {"kind": "confirm_role", "column": "nope", "role": "exposure"}):
        with pytest.raises(d.Refusal):
            d.validate(bad, ctx)


def test_b3_the_grouping_identifier_is_read_from_settled_roles_only(tmp_path):
    """BLUEPRINT §14 rule 2 for the clustering the intervals and the seal use: an identifier that
    rode along unconfirmed and repeats is asked (the intervals are refused with a ``confirm_role``
    exit; the seal draws by row and says why), never silently clustered by or ignored; confirmed,
    it clusters (pandas: 40 units of 5 rows)."""
    from turbotab.core.models.inference import cluster_columns, resolve_clusters
    from turbotab.core.seal import decide_basis

    rng = np.random.default_rng(3)
    frame = pd.DataFrame({"subject": np.repeat(np.arange(40), 5), "x": rng.normal(size=200)},
                         index=pd.Index(np.arange(200), name="row_id"))
    assert frame["subject"].nunique() == 40 and frame["subject"].value_counts().max() == 5
    waiting = ProjectState(roles={"subject": "identifier", "x": "exposure"},
                           roles_unconfirmed=["subject"],
                           grain=GrainSpec(grain="one_row_per_unit"))
    clusters = resolve_clusters(waiting, frame[cluster_columns(waiting, list(frame.columns))])
    assert clusters.refusal and not clusters.clustered
    assert {"kind": "confirm_role", "column": "subject", "role": "identifier"} in [
        x["decision"] for x in clusters.exits]
    settled = waiting.model_copy(update={"role_confirmations": {"subject": "identifier"}})
    clusters = resolve_clusters(settled, frame[cluster_columns(settled, list(frame.columns))])
    assert clusters.clustered and clusters.column == "subject" and clusters.n_clusters == 40
    repeated = waiting.model_copy(update={"grain": GrainSpec(grain="repeated")})
    basis, column = decide_basis(repeated, frame, [], unconfirmed=["subject"])
    assert column is None and basis.state == "abandoned" and "awaits its own confirmation" in basis.sentence


# ═════════════════════════════════════════════════════════════════════════════
# C · The corpus: real naming conventions, each column's documented meaning
# ═════════════════════════════════════════════════════════════════════════════
#
# Each fixture is generated from a stated process under the names a real export uses, and every
# column carries its meaning ("truth"): intake (a nutrient eaten), energy (total energy intake),
# unit (a unit's identifier), flag (marks another column's blanks), time (orders a unit's rows),
# design (a survey weight, stratum or PSU), or other (body composition, a lab, a device, a label, a
# characteristic). The recognizers' readings are scored against it.

INTAKE, ENERGY, UNIT, FLAG, TIME, DESIGN, OTHER = ("intake", "energy", "unit", "flag", "time",
                                                    "design", "other")


def _diet(rng: np.random.Generator, n: int, *, alcohol: bool = False) -> dict:
    """A diet record whose energy is what its macronutrients carry (4P + 4C + 9F + 7A), with
    shares drawn per person, so nutrients rise with energy as real intakes do."""
    kcal = np.exp(rng.normal(np.log(2100), 0.28, n))
    p = rng.normal(0.16, 0.03, n).clip(0.08, 0.3)
    f = rng.normal(0.34, 0.06, n).clip(0.15, 0.55)
    a = np.where(rng.random(n) < 0.6, rng.gamma(2, 0.02, n), 0.0) if alcohol else np.zeros(n)
    c = (1 - p - f - a).clip(0.15, 0.75)
    return {"kcal": kcal, "protein": kcal * p / 4, "fat": kcal * f / 9, "carb": kcal * c / 4,
            "alcohol": kcal * a / 7}


def _corpus() -> dict[str, tuple[pd.DataFrame, list, str, dict, str | None]]:
    """``{name: (frame, lens, target, truth, unit column)}``."""
    out = {}
    rng = np.random.default_rng(14_001)

    # Hologic Apex whole-body DXA (the CLSA DSD's names) beside a diet record, adults.
    n = 400
    diet = _diet(rng, n)
    fat_mass = rng.normal(25000, 7000, n).clip(8000)
    lean = rng.normal(48000, 9000, n).clip(28000)
    dxa = pd.DataFrame({
        "participant_id": np.arange(1, n + 1), "age": rng.integers(30, 75, n),
        "sex": rng.choice(["F", "M"], n), "energy_kcal": diet["kcal"].round(),
        "protein_g": diet["protein"].round(1), "carbohydrate_g": diet["carb"].round(1),
        "fat_g": diet["fat"].round(1),
        "WBTOT_FAT": fat_mass.round(), "WBTOT_LEAN": lean.round(),
        "WBTOT_MASS": (fat_mass + lean).round(), "WBTOT_PFAT": (100 * fat_mass / (fat_mass + lean)).round(1),
        "TRUNK_FAT": (fat_mass * 0.48).round(), "LARM_FAT": (fat_mass * 0.055).round(),
        "RARM_FAT": (fat_mass * 0.056).round(), "LLEG_FAT": (fat_mass * 0.17).round(),
        "RLEG_FAT": (fat_mass * 0.17).round(), "HEAD_FAT": (fat_mass * 0.05).round(),
        "SUBTOT_FAT": (fat_mass * 0.95).round(), "ANDROID_FAT": (fat_mass * 0.08).round(),
        "GYNOID_FAT": (fat_mass * 0.16).round(), "hba1c": rng.normal(5.6, 0.5, n).round(1)})
    truth = {c: OTHER for c in dxa.columns}
    truth.update(participant_id=UNIT, energy_kcal=ENERGY, protein_g=INTAKE, carbohydrate_g=INTAKE,
                 fat_g=INTAKE)
    out["hologic_dxa"] = (dxa, ["dietary"], "hba1c", truth, None)
    B = _children_hologic()
    truth = {c: OTHER for c in B.columns}
    truth.update(child_id=UNIT, energy_kcal=ENERGY, protein_g=INTAKE, carbohydrate_g=INTAKE,
                 fat_g=INTAKE)
    out["hologic_dxa_children"] = (B, ["dietary"], "WBTOT_PFAT", truth, None)

    # InBody 770 result names beside a 24-hour recall, with a CBC.
    for label, frame in _inbody().items():
        truth = {c: OTHER for c in frame.columns}
        truth.update(participant_id=UNIT, energy_kcal=ENERGY, protein_g=INTAKE,
                     carbohydrate_g=INTAKE, fat_g=INTAKE)
        out[f"inbody_{label}"] = (frame, ["dietary"], "hba1c", truth, None)

    # ASA24 Totals (two recalls per user, a week apart) under its own names and the IDATA ones.
    users, per = 160, 2
    diet = _diet(rng, users * per, alcohol=True)
    start = pd.Timestamp("2022-03-01")
    when = [start + pd.Timedelta(days=int(u % 90) + 7 * r) for u in range(users) for r in range(per)]
    asa = pd.DataFrame({
        "UserName": [f"u{u:03d}" for u in range(users) for _ in range(per)],
        "RecallNo": [r + 1 for _ in range(users) for r in range(per)],
        "IntakeStartDateTime": [w.strftime("%m/%d/%Y 00:00") for w in when],
        "KCAL": (diet["kcal"]).round(1), "PROT": diet["protein"].round(1),
        "TFAT": diet["fat"].round(1), "CARB": diet["carb"].round(1),
        "ALC": diet["alcohol"].round(1), "SFAT": (diet["fat"] * 0.33).round(1),
        "MFAT": (diet["fat"] * 0.37).round(1), "PFAT": (diet["fat"] * 0.22).round(1),
        "FIBE": (diet["kcal"] / 1000 * rng.normal(9, 2, users * per).clip(2)).round(1),
        "SODI": (diet["kcal"] * rng.normal(1.6, 0.3, users * per)).round(),
        "ldl": rng.normal(3.2, 0.7, users * per).round(2)})
    truth = {c: INTAKE for c in asa.columns}
    truth.update(UserName=UNIT, RecallNo=TIME, IntakeStartDateTime=TIME, KCAL=ENERGY, ldl=OTHER)
    out["asa24_totals"] = (asa, ["dietary"], "ldl", truth, "UserName")
    idata = asa.rename(columns={"KCAL": "tns_kcal_asa24", "PROT": "tns_prot_asa24",
                                "TFAT": "tns_tfat_asa24", "CARB": "tns_carb_asa24",
                                "ALC": "tns_alc_asa24", "SFAT": "tns_sfat_asa24"})
    truth = {c: INTAKE for c in idata.columns}
    truth.update(UserName=UNIT, RecallNo=TIME, IntakeStartDateTime=TIME, tns_kcal_asa24=ENERGY,
                 ldl=OTHER)
    out["asa24_idata"] = (idata, ["dietary"], "ldl", truth, "UserName")

    # CDISC ADaM BDS (ADLB-like): subjects × visits, the analysis value the outcome.
    subjects, visits = 120, 4
    sid = np.repeat(np.arange(subjects), visits)
    vis = np.tile([0, 4, 8, 12], subjects)
    base = np.repeat(rng.normal(3.4, 0.7, subjects), visits)
    aval = base + np.where(vis > 0, -0.05 * vis, 0) + rng.normal(0, 0.25, subjects * visits)
    arm = np.repeat(rng.choice(["Placebo", "Low fat diet"], subjects), visits)
    adt = [pd.Timestamp("2021-01-04") + pd.Timedelta(days=int(s % 60) + 7 * int(v))
           for s, v in zip(sid, vis)]
    adam = pd.DataFrame({
        "USUBJID": [f"STUDY01-{s:04d}" for s in sid], "SUBJID": sid + 1001,
        "SITEID": np.repeat(rng.integers(101, 109, subjects), visits), "TRT01P": arm,
        "TRT01PN": np.where(arm == "Placebo", 1, 2), "ARMCD": np.where(arm == "Placebo", "PBO", "LFD"),
        "AGE": np.repeat(rng.integers(25, 70, subjects), visits),
        "SEX": np.repeat(rng.choice(["F", "M"], subjects), visits),
        "ITTFL": np.repeat(rng.choice(["Y", "N"], subjects, p=[0.9, 0.1]), visits),
        "AVISITN": vis, "AVISIT": np.where(vis == 0, "Baseline", [f"Week {v}" for v in vis]),
        "ADT": [x.date().isoformat() for x in adt], "ADY": [int(v) * 7 + 1 for v in vis],
        "BASE": base.round(2), "CHG": (aval - base).round(2),
        "ABLFL": np.where(vis == 0, "Y", "N"), "AVAL": aval.round(2)})
    truth = {c: OTHER for c in adam.columns}
    truth.update(USUBJID=UNIT, SUBJID=UNIT, AVISITN=TIME, AVISIT=TIME, ADT=TIME, ADY=TIME)
    out["cdisc_adam"] = (adam, ["clinical"], "AVAL", truth, "USUBJID")

    # UK Biobank: RAP field names, and the showcase's titles (Energy in KJ; a lymphocyte count).
    n = 500
    diet = _diet(rng, n, alcohol=True)
    ukb = pd.DataFrame({
        "eid": rng.choice(np.arange(1_000_000, 6_000_000), n, replace=False),
        "p31": rng.choice([0, 1], n), "p21003_i0": rng.integers(40, 70, n),
        "p21001_i0": rng.normal(27, 4.5, n).round(2),
        "p100002_i0": (diet["kcal"] * KCAL_PER_KJ).round(), "p100003_i0": diet["protein"].round(1),
        "p100004_i0": diet["fat"].round(1), "p100005_i0": diet["carb"].round(1),
        "p100022_i0": diet["alcohol"].round(1),
        "p30120_i0": np.exp(rng.normal(np.log(1.9), 0.35, n)).round(2),
        "ldl": rng.normal(3.5, 0.8, n).round(2)})
    truth = {c: OTHER for c in ukb.columns}
    truth.update(eid=UNIT, p100002_i0=ENERGY, p100003_i0=INTAKE, p100004_i0=INTAKE,
                 p100005_i0=INTAKE, p100022_i0=INTAKE)
    out["ukbiobank_fields"] = (ukb, ["dietary"], "ldl", truth, None)
    titled = ukb.rename(columns={"p31": "Sex", "p21003_i0": "Age at recruitment",
                                 "p21001_i0": "Body mass index (BMI)", "p100002_i0": "Energy",
                                 "p100003_i0": "Protein", "p100004_i0": "Fat",
                                 "p100005_i0": "Carbohydrate", "p100022_i0": "Alcohol",
                                 "p30120_i0": "Lymphocyte count"})
    truth = {c: OTHER for c in titled.columns}
    truth.update(eid=UNIT, Energy=ENERGY, Protein=INTAKE, Fat=INTAKE, Carbohydrate=INTAKE,
                 Alcohol=INTAKE)
    out["ukbiobank_titles"] = (titled, ["dietary"], "ldl", truth, None)

    # NHANES pooled cycles: the tutorial's names, a fasting weight with zeros, the dietary file.
    for label, frame in _nhanes_weights().items():
        lower = label == "W3"
        frame = frame.copy()
        diet = _diet(rng, len(frame), alcohol=True)
        names = {"DR1TPROT": diet["protein"], "DR1TCARB": diet["carb"], "DR1TTFAT": diet["fat"],
                 "DR1TALCO": diet["alcohol"]}
        kcal = "dr1tkcal" if lower else "DR1TKCAL"
        frame[kcal] = diet["kcal"].round()
        for c, v in names.items():
            frame[c.lower() if lower else c] = v.round(1)
        frame["dr1tsfat" if lower else "DR1TSFAT"] = (diet["fat"] * 0.34).round(1)
        truth = {c: OTHER for c in frame.columns}
        for c in frame.columns:
            u = c.upper()
            if u == "SEQN":
                truth[c] = UNIT
            elif u in ("SDMVSTRA", "SDMVPSU", "SDDSRVYR", "MEC6YR", "SAF6YR", "WTMEC6YR",
                       "WTSAF6YR", "WTDR6YR", "WTMEC2YR", "WTSAF2YR", "WTDRD1"):
                truth[c] = DESIGN
            elif u == "DR1TKCAL":
                truth[c] = ENERGY
            elif u.startswith("DR1T"):
                truth[c] = INTAKE
        out[f"nhanes_{label}"] = (frame, ["dietary"], "lbxglu" if lower else "LBXGLU", truth, None)

    # Wearables beside a diet diary: Fitbit daily activity (per day per Id), Oura and Garmin summaries.
    ids, days = 60, 7
    diet = _diet(rng, ids * days)
    steps = rng.lognormal(8.9, 0.45, ids * days).round()
    tee = (1600 + 0.045 * steps + rng.normal(0, 160, ids * days)).round()
    fitbit = pd.DataFrame({
        "Id": np.repeat(1503960366 + np.arange(ids) * 7919, days),
        "ActivityDate": [(pd.Timestamp("2016-04-12") + pd.Timedelta(days=d)).date().isoformat()
                         for _ in range(ids) for d in range(days)],
        "TotalSteps": steps.astype(int), "TotalDistance": (steps * 0.00076).round(2),
        "VeryActiveMinutes": rng.poisson(21, ids * days), "Calories": tee.astype(int),
        "energy_kcal": diet["kcal"].round(), "protein_g": diet["protein"].round(1),
        "carbohydrate_g": diet["carb"].round(1), "fat_g": diet["fat"].round(1),
        "glucose": rng.normal(98, 12, ids * days).round()})
    truth = {c: OTHER for c in fitbit.columns}
    truth.update(Id=UNIT, ActivityDate=TIME, energy_kcal=ENERGY, protein_g=INTAKE,
                 carbohydrate_g=INTAKE, fat_g=INTAKE)
    out["fitbit_daily"] = (fitbit, ["dietary"], "glucose", truth, "Id")
    n = 300
    tee = rng.normal(2400, 350, n).round()
    oura = pd.DataFrame({"record_id": np.arange(1, n + 1), "sex": rng.choice(["F", "M"], n),
                         "age": rng.integers(25, 70, n), "total_calories": tee,
                         "active_calories": (tee * rng.uniform(0.15, 0.35, n)).round(),
                         "score": rng.integers(50, 100, n),
                         "fruit_servings": rng.poisson(2.5, n), "sbp": rng.normal(124, 15, n).round()})
    truth = {c: OTHER for c in oura.columns}
    truth.update(record_id=UNIT)
    out["oura_no_diet"] = (oura, ["dietary"], "sbp", truth, None)
    garmin = oura.rename(columns={"total_calories": "totalKilocalories",
                                  "active_calories": "activeKilocalories"})
    out["garmin_no_diet"] = (garmin, ["dietary"], "sbp",
                             {**{c: OTHER for c in garmin.columns}, "record_id": UNIT}, None)

    # An EHR extract: encounters per patient, a birth date, indicator and imputation columns, a CBC.
    patients, visits = 150, 3
    pid = np.repeat(np.arange(patients), visits)
    sbp = rng.normal(132, 16, patients * visits).round()
    sbp[rng.random(patients * visits) < 0.12] = np.nan
    ehr = pd.DataFrame({
        "patient_id": [f"MRN{p:05d}" for p in pid],
        "encounter_date": [(pd.Timestamp("2020-01-01") + pd.Timedelta(days=int(p % 30) + 120 * v)
                            ).date().isoformat() for p, v in zip(pid, np.tile(range(visits), patients))],
        "dob": np.repeat([(pd.Timestamp("1950-01-01") + pd.Timedelta(days=int(x))).date().isoformat()
                          for x in rng.integers(0, 15000, patients)], visits),
        "diabetes_flag": np.repeat(rng.choice([0, 1], patients, p=[0.8, 0.2]), visits),
        "statin_flag": np.repeat(rng.choice([0, 1], patients, p=[0.7, 0.3]), visits),
        "sbp": sbp, "sbp_flag": np.isnan(sbp).astype(int),
        "sbp_imp": np.where(np.isnan(sbp), np.nanmean(sbp), sbp).round(1),
        "ALC": np.exp(rng.normal(np.log(1.9), 0.35, patients * visits)).round(2),
        "ANC": np.exp(rng.normal(np.log(3.8), 0.4, patients * visits)).round(2),
        "WBC": np.exp(rng.normal(np.log(6.5), 0.3, patients * visits)).round(1),
        "hba1c": rng.normal(6.2, 0.9, patients * visits).round(1)})
    truth = {c: OTHER for c in ehr.columns}
    truth.update(patient_id=UNIT, encounter_date=TIME, sbp_flag=FLAG)
    out["ehr_encounters"] = (ehr, ["dietary", "clinical"], "hba1c", truth, "patient_id")

    cc = _case_control()
    truth = {c: OTHER for c in cc.columns}
    truth.update(study_no=UNIT, fiber_g=INTAKE)
    out["case_control"] = (cc, ["clinical"], "crp", truth, None)
    return out


_CLAIM = {"identifier": UNIT, "energy": ENERGY, "flag": FLAG, "time": TIME, "design": DESIGN}


def _claim(p: dict) -> str | None:
    """What a proposal says the column is, in the corpus's terms."""
    if p["proposed"] in _CLAIM:
        return _CLAIM[p["proposed"]]
    if p["proposed"] == "exposure" and p["reason"].startswith(("A nutrient", "Named as a nutrient")):
        return INTAKE
    return None


def _verify_high(name: str, p: dict, frame: pd.DataFrame, truth: dict, unit: str | None) -> str:
    """Why the values corroborate a high reading, computed here from §14's criteria; raises when
    they do not (a name-only reading called high)."""
    claim = _claim(p)
    x = frame[name]
    energy_col = next((c for c, t in truth.items() if t == ENERGY), None)
    macros = {c: t for c, t in truth.items() if t == INTAKE}
    if claim == UNIT:
        k, rows = x.nunique(), x.notna().sum()
        assert k >= 3 and (k == rows or k > 10), (name, k, rows)
        return "unit structure"
    if claim == ENERGY:
        from turbotab.core.recognizers import read_nutrient

        parts = []
        for c in macros:
            reading = read_nutrient(c)
            if reading is not None and reading.part is None and reading.macro in ATWATER:
                parts.append(frame[c] * ATWATER[reading.macro])
        assert parts, name
        assert _r(x, sum(parts)) >= 0.7, name
        return "Atwater"
    if claim == INTAKE:
        v = pd.to_numeric(x, errors="coerce").dropna()
        assert v.nunique() > 2 and v.min() >= 0, name
        r = _r(x, frame[energy_col]) if energy_col else float("nan")
        if math.isfinite(r) and r >= 0.3:
            return "r"
        from turbotab.core.recognizers import read_nutrient

        totals = {}
        for c in macros:
            reading = read_nutrient(c)
            if reading is not None and reading.part is None and reading.macro in ATWATER:
                totals.setdefault(reading.macro, c)
        recon = sum(frame[c] * ATWATER[m] for m, c in totals.items())
        ratio = float((frame[energy_col] / recon).median())
        assert 0.9 <= ratio <= 1.1 or 0.9 <= ratio / KCAL_PER_KJ <= 1.1, (name, ratio)
        share = float((frame[name] * ATWATER[read_nutrient(name).macro] / recon).median())
        assert share >= 0.05, (name, share)
        return "identity"
    if claim == FLAG:
        base = p["linked_to"]
        assert x.nunique() == 2 and base in frame.columns, name
        blank = frame[base].isna()
        assert any(((x == lvl) == blank).mean() >= 0.99 for lvl in x.dropna().unique()), name
        return "tied to blanks"
    if claim == TIME:
        assert unit is not None, name
        varying = frame.groupby(unit)[name].nunique()
        assert float((varying > 1).mean()) >= 0.5, name
        return "varies within units"
    if claim == DESIGN:
        v = pd.to_numeric(x, errors="coerce").dropna()
        assert v.min() >= 0, name
        return "design values"
    if p["proposed"] == "covariate":
        low = str(name).lower()
        if "sex" in low or "gender" in low:
            assert 2 <= x.nunique() <= 3, name
        elif "age" in low.split("_") or low == "age":
            assert pd.to_numeric(x).median() <= 120, name
        return "characteristic"
    if p["proposed"] == "excluded":
        assert x.nunique() <= 1 or not pd.api.types.is_numeric_dtype(x), name
        return "constant or text"
    return "other"


def test_c_the_corpus_no_name_only_reading_is_high_and_no_default_rides_along(tmp_path):
    """The corpus (Hologic DXA, InBody, ASA24, CDISC ADaM, UK Biobank, NHANES multi-cycle, wearables,
    EHR flags, a case–control table): for every fixture,

    * every high proposal's corroboration is recomputed here from §14's criteria (pandas/NumPy) and
      agrees with the column's documented meaning: no name-only reading is high, and high is right;
    * every misreading (a proposal whose claim is not the column's meaning) is below high, marked for
      attention and listed for its own confirmation;
    * after a bulk confirm, no number-changing default comes from a proposal below high: the energy
      card's nutrients and energy column, an unrefused intake screen, the first survey weight
      offered, the clustering the intervals and the seal read;
    * the name-only twins (each recognized column's values shuffled, or made constant within units)
      lose their high readings.

    The recognizers' accuracy on the corpus is reported (``-s`` prints it)."""
    from turbotab.core import leash
    from turbotab.core.models.inference import cluster_columns, resolve_clusters

    corpus = _corpus()
    totals = {"columns": 0, "meaningful": 0, "recognized": 0, "high": 0, "high_right": 0,
              "misread": 0, "misread_high": 0}
    for label, (frame, lens, target, truth, unit) in corpus.items():
        path = _write(frame, tmp_path, f"{label}.csv")
        artifact = _roles_artifact(path, lens=lens, target=target)
        props = {p["column"]: p for p in artifact["columns"]}
        assert artifact["needs_confirmation"] == [p["column"] for p in artifact["columns"]
                                                  if p["confidence"] != "high"], label
        for name, p in props.items():
            assert p["attention"] == (p["confidence"] != "high"), (label, p)
            claim = _claim(p)
            meaning = truth[name]
            totals["columns"] += 1
            totals["meaningful"] += meaning != OTHER
            if claim is not None and claim == meaning:
                totals["recognized"] += 1
            if claim is not None and claim != meaning:
                totals["misread"] += 1
                totals["misread_high"] += p["confidence"] == "high"
                assert p["confidence"] != "high" and p["attention"], (label, name, meaning, p)
            if p["confidence"] == "high":
                totals["high"] += 1
                _verify_high(name, p, frame, truth, unit)
                assert claim is None or claim == meaning, (label, name, meaning, p)
                totals["high_right"] += 1

        # A bulk confirm: what the number-changing defaults read.
        high = _high(artifact)
        state = _bulk_state(artifact, lens=lens, target=target)
        out = _proposals_stage(path, state, artifact)
        energy = out.get("energy") or {}
        for c in energy.get("nutrients") or []:
            assert c in high, (label, "nutrient", c)
        e = energy.get("energy_column")
        if e is not None and e not in high:
            assert all(not v["ok"] for v in energy["applicability"].values()), (label, e)
            assert all(x["refused"] for x in out["exclusions"]), (label, e)
        for x in out["exclusions"]:
            if not x["refused"]:
                assert x["rule"]["column"] in high, (label, x)
        survey = out.get("survey") or {}
        for option in survey.get("options") or []:
            w = option["decision"].get("weight")
            if w and w not in high:
                assert option["needs_confirmation"], (label, option)
        present = [c for c in cluster_columns(state, list(frame.columns))]
        if present:
            indexed = frame.set_index(pd.Index(np.arange(len(frame)), name="row_id"))
            clusters = resolve_clusters(state, indexed[present])
            if clusters.clustered:
                assert clusters.column in high, (label, clusters.column)
        assert set(leash.settled_columns(state)) == high | {c for c in state.roles
                                                            if c not in state.roles_unconfirmed}

        # Name-only twins: the recognized columns' values no longer corroborate their names.
        twin = frame.copy()
        shuffle = np.random.default_rng(7)
        relational = [c for c, p in props.items() if p["confidence"] == "high"
                      and _claim(p) in (INTAKE, ENERGY, FLAG)]
        for c in relational:
            twin[c] = shuffle.permutation(twin[c].to_numpy())
        timed = [c for c, p in props.items() if p["confidence"] == "high" and _claim(p) == TIME]
        if unit is not None:
            for c in timed:
                twin[c] = twin.groupby(unit)[c].transform("first")
        units = [c for c, p in props.items() if p["confidence"] == "high" and _claim(p) == UNIT]
        for c in units:
            twin[c] = np.where(np.arange(len(twin)) % 2, "a", "b") if c != unit else twin[c]
        if relational or timed or units:
            twin_props = _roles(_write(twin, tmp_path, f"{label}_twin.csv"), lens=lens,
                                target=target)
            for c in [*relational, *timed, *[u for u in units if u != unit]]:
                tp = twin_props[c]
                assert not (tp["confidence"] == "high" and _claim(tp) == _claim(props[c])), (
                    label, c, tp)

    precision = totals["high_right"] / max(totals["high"], 1)
    print(f"\nRecognition corpus: {len(corpus)} tables, {totals['columns']} columns, "
          f"{totals['meaningful']} with a role to recognize; {totals['recognized']} read as what "
          f"they are (recall {totals['recognized'] / max(totals['meaningful'], 1):.0%}); "
          f"{totals['high']} high, precision {precision:.0%}; {totals['misread']} misread, "
          f"{totals['misread_high']} of them high.")
    assert totals["misread_high"] == 0 and precision == 1.0
