"""The intelligence gate's repair: the leash first, then the seven items it left open (2026-10-03).

The independent verifier of the intelligence layer (WP13–WP15) closed nothing on these items:
WP13-1 (nutrients and total energy), WP13-2 (arms, treatments, acquisition words), WP13-5 (kJ
outside the adult band), WP13-6 (the fasting weight), WP14-4 (children and ages), WP14-6 (repeats
versus time points) and WP14-7 (lens hints), and WP15-3 (an energy-related outcome). Its finding
behind all seven: an uncertain recognition fed a number-changing default, or a SETTLED claim, without
the user ever confirming it, and each earlier repair added the names it was shown.

**The leash first** (BLUEPRINT §11.3, North star 5): every reading carries what corroborated it.

* A role proposal is ``high`` only where the values agree with the name; a name-only reading is
  ``medium`` at most and its reason says so (section 0).
* A unit nothing settles (the energy's by its median or by nothing; an age's by a bare ``age``) is
  a proposal: no screen, band or count is applied in it, and ``set_column_unit`` records it.
* A repeat kind is stated only on unambiguous evidence; spacing alone asks.
* A claim that depends on an unplaced outcome is stated as a condition the user answers.

**Then corroboration, not names**: total energy is read against the energy its macronutrients
carry; a unit is read as a whole expression (g/kg is not grams); a subsample's analytes are read
by where they are recorded; an identifier must name units; a growth index or an outcome that
tracks a body-size column is energy-related; a count matrix needs genes' spread of abundance.

Expected values never come from the code under test: NumPy/SciPy/pandas computations on the
fixtures, the FAO Atwater factors (4/4/9/7 kcal/g), the CDC growth-chart LMS tables read and
transformed here, and quoted primary sources (fetched 2026-10-03). Fixtures replay the verifier's
probes draw for draw (``/private/tmp/turbotab-fix/gate-intelligence/probes``: wp13_energy_devices,
wp13_roles, wp13_kj_toddlers, wp13_weights, wp14_plaus_months, wp14_repeats, wp14_hints).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from turbotab.core import decisions as d
from turbotab.core.decisions import ProjectState
from turbotab.core.tests.stage_harness import SAMPLES, Ingested

HERE = Path(__file__).resolve().parent / "wp14_data"
KCAL_PER_KJ = 4.184  # FAO: 1 kcal = 4.184 kJ
ATWATER = {"protein": 4.0, "carbohydrate": 4.0, "fat": 9.0, "alcohol": 7.0}  # FAO general factors

# ── primary sources, quoted ──────────────────────────────────────────────────

# Fitabase Fitbit Data Dictionary (fitabase.com/media/2126/fitabase-fitbit-data-dictionary-as-of-
# 05162025.pdf), Daily Activity: "Calories integer Total estimated energy expenditure (in
# kilocalories)."
FITABASE_CALORIES = "Total estimated energy expenditure (in kilocalories)."
# NHANES 2017–2018 TRIGLY_J codebook (wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/
# TRIGLY_J.htm): "WTSAF2YR - Fasting Subsample 2 Year MEC Weight", "LBDLDNSI - LDL-Cholesterol, NIH
# equation 2 (mmol/L)", "LBDLDMSI - LDL-Cholesterol, Martin-Hopkins (mmol/L)", and "Specific sample
# weights for this subsample are included in this data file and should be used when analyzing
# these data."
TRIGLY_J = {"WTSAF2YR": "Fasting Subsample 2 Year MEC Weight",
            "LBDLDNSI": "LDL-Cholesterol, NIH equation 2 (mmol/L)",
            "LBDLDMSI": "LDL-Cholesterol, Martin-Hopkins (mmol/L)"}
# WHO anthro R package (cran.r-project.org/web/packages/anthro/anthro.pdf), anthro_zscores' value:
# "zlen Length/Height-for-age z-score", "zwei Weight-for-age z-score", "zwfl Weight-for-length/
# height z-score", "zbmi BMI-for-age z-score", "zac Arm circumference-for-age z-score", "zts Triceps
# skinfold-for-age z-score", "zss Subscapular skinfold-for-age z-score"; "fwei 1, if zwei < -6 or
# zwei > 5" (a flag, not the index).
WHO_ANTHRO = {"zlen": "Length/Height-for-age z-score", "zwei": "Weight-for-age z-score",
              "zwfl": "Weight-for-length/height z-score", "zbmi": "BMI-for-age z-score",
              "zac": "Arm circumference-for-age z-score", "zts": "Triceps skinfold-for-age z-score",
              "zss": "Subscapular skinfold-for-age z-score"}
# Framingham Heart Study Coding Manual, Food Frequency Questionnaire (dbGaP phd001373): "nutrient
# fields starting with NUT_", "NUT_CALOR DERIVED FIELD: CALORIES, (kcal)", "NUT_PROT … PROTEIN,
# (gm)", "NUT_CARBO … CARBOHYDRATES, (gm)", "NUT_ALCO … ALCOHOL, (gm)", "NUT_SATFAT … SATURATED
# FAT, (gm)"; the validity marker reads "CALORIES (NUT_CALOR) BETWEEN 600 – 4199" for men.
# Byrne et al. 2019, "Sources and Determinants of Discretionary Food Intake in a Cohort of
# Australian Children Aged 12–14 Months" (PMC6981432), Table 2: "Energy (kJ) all foods 4040
# (±954.7 SD)".
SMILE_KJ = (4040.0, 954.7)


def _write(frame: pd.DataFrame, folder: Path, name: str) -> Path:
    path = Path(folder) / name
    frame.to_csv(path, index=False)
    return path


def _table(path: Path) -> Ingested:
    return Ingested(path, Path(tempfile.mkdtemp()))


def _roles(path: Path, *, lens, target=None, purpose=None) -> dict:
    from turbotab.core.stages.rows import roles_stage

    out = _table(path).run(roles_stage, ProjectState(lens=lens, target=target, purpose=purpose))
    return {"columns": {p["column"]: p for p in out["columns"]}, "repeats": out["repeats"],
            "categorical": out["categorical"]}


def _proposals(path: Path, *, lens, target=None, roles=None, units=None, purpose=None,
               settled=None) -> dict:
    """The proposals artifact as the stage builds it, over the roles the roles stage proposes.
    ``settled``: the columns whose roles were confirmed (BLUEPRINT §14); None reads the values only."""
    from turbotab.core.stages.proposals import build_proposals

    t = _table(path)
    if roles is None:
        roles = {c: p["proposed"] for c, p in _roles(path, lens=lens, target=target,
                                                       purpose=purpose)["columns"].items()}
    return build_proposals(t.frame(), t.info["columns"], lens=lens, target=target, roles=roles,
                           purpose=purpose, units=units, settled=settled)


def _findings(path: Path, *, lens, target=None, units=None) -> list[dict]:
    from turbotab.core.stages.findings import findings_stage

    return _table(path).run(findings_stage, ProjectState(lens=lens, target=target,
                                                         column_units=units))["findings"]


def _reconstructed(frame: pd.DataFrame, macros: dict[str, str]) -> pd.Series:
    """FAO: E = 4P + 4C + 9F (+ 7A), on the columns this test names."""
    return sum(frame[c] * ATWATER[m] for m, c in macros.items())


# ═════════════════════════════════════════════════════════════════════════════
# Fixtures: the verifier's probes, draw for draw
# ═════════════════════════════════════════════════════════════════════════════


def _devices() -> dict[str, pd.DataFrame]:
    """probes/wp13_energy_devices.py (``default_rng(4041)``, n = 320): a Fitbit daily summary and an
    ActiLife summary, each merged with a 24-hour recall whose energy is 4P + 4C + 9F."""
    rng = np.random.default_rng(4041)
    n = 320
    sex = rng.choice(["F", "M"], n)
    ei = np.clip(rng.normal(np.where(sex == "M", 2400, 1850), 450), 700, 4800).round()
    prot = (ei * rng.uniform(0.13, 0.19, n) / 4).round(1)
    fat = (ei * rng.uniform(0.28, 0.40, n) / 9).round(1)
    carb = ((ei - 4 * prot - 9 * fat) / 4).round(1)
    steps = rng.lognormal(8.9, 0.45, n).round()
    tee = (np.where(sex == "M", 1750, 1400) + 0.045 * steps + rng.normal(0, 160, n)).round()
    weight = (70 + 0.004 * (ei - 2000) + rng.normal(0, 12, n)).round(1)
    fitbit = pd.DataFrame({
        "Id": 1503960366 + np.arange(n) * 7919, "TotalSteps": steps.astype(int),
        "TotalDistance": (steps * 0.00076).round(2), "VeryActiveMinutes": rng.poisson(21, n),
        "SedentaryMinutes": rng.normal(990, 120, n).round().astype(int), "Calories": tee.astype(int),
        "sex": sex, "energy_kcal": ei, "protein_g": prot, "carbohydrate_g": carb, "fat_g": fat,
        "sodium_mg": rng.normal(3300, 900, n).round(), "weight_kg": weight})
    kcals = np.clip(rng.gamma(6, 110, n), 120, None).round(1)
    acti = pd.DataFrame({
        "subject_id": [f"S{i:04d}" for i in range(n)], "Axis1": rng.lognormal(12.6, 0.4, n).round(),
        "Steps": steps.astype(int), "MVPA_min": rng.gamma(2.5, 12, n).round(1), "Kcals": kcals,
        "sex": sex, "Energy (kcal)": ei, "Protein (g)": prot, "Carbohydrate (g)": carb,
        "Total Fat (g)": fat,
        "hba1c": (5.4 + 0.0002 * (ei - 2000) + rng.normal(0, 0.45, n)).round(1)})
    return {"fitbit": fitbit, "actilife": acti}


def _toddlers_and_two_day() -> tuple[pd.DataFrame, pd.DataFrame]:
    """probes/wp13_kj_toddlers.py (``default_rng(1214)``, n = 500): toddlers' energy in kJ at the
    SMILE cohort's Table 2 distribution, in a column named only ``energy``, with sodium beside it;
    and adults' energy summed over two recalls, named ``total_energy``."""
    rng = np.random.default_rng(1214)
    n = 500
    kj = np.clip(rng.normal(*SMILE_KJ, n), 900, None).round()
    tod = pd.DataFrame({"child_id": np.arange(n) + 1, "sex": rng.choice(["boy", "girl"], n),
                        "energy": kj, "sodium_mg": rng.normal(1100, 300, n).round(),
                        "weight_kg": rng.normal(10.2, 1.1, n).round(2)})
    two = np.clip(rng.normal(2050, 480, n), 600, None) + np.clip(rng.normal(2050, 480, n), 600, None)
    adults = pd.DataFrame({"participant_id": np.arange(n) + 1, "sex": rng.choice(["F", "M"], n),
                           "total_energy": two.round(), "sodium_mg": rng.normal(3300, 900, n).round(),
                           "sbp": rng.normal(126, 15, n).round()})
    return tod, adults


def _three_arms_and_sheets() -> dict[str, pd.DataFrame]:
    """probes/wp13_roles.py (``default_rng(303)``): a three-arm feeding trial keyed ``trt_id``
    (and a copy keyed ``tx_id``), a metabolomics sample sheet with ``sample_wt`` (tissue mass), and
    a plate-size feeding study whose ``plate`` (small/large) is the intervention."""
    rng = np.random.default_rng(303)
    n = 312
    arm = np.repeat([1, 2, 3], n // 3)
    rng.shuffle(arm)
    a = pd.DataFrame({"USUBJID": [f"DASH-{i:04d}" for i in range(n)],
                      "SITEID": rng.choice([101, 102, 103, 104], n), "trt_id": arm,
                      "age": rng.integers(22, 76, n), "sex": rng.choice(["F", "M"], n),
                      "sbp_baseline": rng.normal(134, 10, n).round(),
                      "sbp_change": (np.select([arm == 1, arm == 2, arm == 3], [-1.0, -5.5, -8.9])
                                     + rng.normal(0, 8, n)).round(1)})
    m = 90
    b = pd.DataFrame({"Sample Name": [f"S{i:03d}" for i in range(m)],
                      "Group": rng.choice(["case", "control"], m),
                      "Batch": np.repeat([1, 2, 3], m // 3), "Injection Order": rng.permutation(m) + 1,
                      "sample_wt": rng.normal(48, 6, m).round(1)})
    for j in range(40):
        b[f"M{100 + 7 * j}T{60 + 3 * j}"] = rng.lognormal(10, 1, m).round(1)
    b["age"] = rng.integers(30, 70, m)
    c = pd.DataFrame({"participant_id": np.arange(200) + 1, "plate": rng.choice(["small", "large"], 200),
                      "age": rng.integers(18, 60, 200), "bmi": rng.normal(25, 4, 200).round(1)})
    c["kcal_eaten"] = (650 + np.where(c["plate"] == "large", 90, 0) + rng.normal(0, 150, 200)).round()
    return {"trt": a, "tx": a.rename(columns={"trt_id": "tx_id"}), "sheet": b, "plate": c}


def _nhanes_renamed() -> tuple[pd.DataFrame, pd.DataFrame]:
    """probes/wp13_weights.py (``default_rng(2018)``, n = 600): a 2017–2018 merge whose fasting
    glucose was renamed ``fasting_glucose``; and the gate's LBDLDNSI variant (a TRIGLY_J LDL kept
    by its codebook name), the same table with LDL in place of glucose."""
    rng = np.random.default_rng(2018)
    n = 600
    fast = rng.random(n) < 0.45
    df = pd.DataFrame({
        "SEQN": 93703 + np.arange(n), "RIAGENDR": rng.choice([1, 2], n),
        "RIDAGEYR": rng.integers(20, 80, n), "SDMVSTRA": rng.integers(134, 149, n),
        "SDMVPSU": rng.choice([1, 2], n), "WTMEC2YR": rng.lognormal(10.3, 0.6, n).round(1),
        "WTDRD1": rng.lognormal(10.3, 0.7, n).round(1),
        "WTSAF2YR": np.where(fast, rng.lognormal(11.0, 0.7, n), 0).round(1),
        "DR1TKCAL": rng.normal(2050, 600, n).round(), "DR1TSFAT": rng.normal(27, 9, n).round(2),
        "DR1TCARB": rng.normal(240, 70, n).round(1)})
    df["fasting_glucose"] = np.where(fast, rng.normal(105, 22, n).round(), np.nan)
    ldl = df.drop(columns=["fasting_glucose"]).assign(
        LBDLDNSI=np.where(fast, rng.normal(2.9, 0.8, n).round(2), np.nan),
        LBDLDMSI=np.where(fast, rng.normal(2.9, 0.8, n).round(2), np.nan))
    return df, ldl


def _under_five() -> pd.DataFrame:
    """probes/wp14_plaus_months.py (``default_rng(659)``, n = 600): an under-five anthropometry
    survey whose ``age`` holds months (6–59), weights and heights near the WHO medians."""
    rng = np.random.default_rng(659)
    n = 600
    age_mo = rng.integers(6, 60, n)
    sex = rng.choice([1, 2], n)
    w = 7.9 + (18.0 - 7.9) * (age_mo - 6) / 53 + rng.normal(0, 1.2, n)
    h = 67 + (109 - 67) * (age_mo - 6) / 53 + rng.normal(0, 3.5, n)
    return pd.DataFrame({"child_id": np.arange(n) + 1, "sex": sex, "age": age_mo,
                         "weight": w.round(1), "height": h.round(1),
                         "hb": rng.normal(11.2, 1.1, n).round(1)})


def _repeats_fixtures() -> dict[str, pd.DataFrame]:
    """probes/wp14_repeats.py (``default_rng(120)``): an OGTT drawn at 0–120 minutes on one
    morning, by date and minute (``meal``) and by per-draw timestamp (``meal_ts``); and four 24-hour
    recalls a season apart (``seasons``, the SEASONS design)."""
    rng = np.random.default_rng(120)
    mins = [0, 15, 30, 45, 60, 90, 120]
    shape = np.array([0, 2.4, 3.6, 3.2, 2.4, 1.0, 0.2])
    rows, rows2 = [], []
    for pid in range(1, 41):
        day = pd.Timestamp("2025-02-03") + pd.Timedelta(days=int(rng.integers(0, 90)))
        base = rng.normal(5.1, 0.4)
        amp = rng.lognormal(0, 0.25)
        age = int(rng.integers(25, 65))
        for m, s in zip(mins, shape):
            g = round(base + amp * s + rng.normal(0, 0.25), 2)
            ins = round(max(2, 8 + 60 * amp * s / 3.6 + rng.normal(0, 6)), 1)
            rows.append({"participant_id": pid, "visit_date": day.date().isoformat(), "time_min": m,
                         "age": age, "glucose": g, "insulin": ins})
            rows2.append({"participant_id": pid,
                          "sample_time": (day + pd.Timedelta(hours=8, minutes=m)).isoformat(sep=" "),
                          "age": age, "glucose": g, "insulin": ins})
    rows3 = []
    for pid in range(1, 151):
        start_day = pd.Timestamp("2024-01-08") + pd.Timedelta(days=int(rng.integers(0, 40)))
        usual = rng.normal(2100, 400)
        for q, season in enumerate(["winter", "spring", "summer", "autumn"]):
            day = start_day + pd.Timedelta(days=91 * q + int(rng.integers(-10, 11)))
            e = max(600, usual + rng.normal(0, 550))
            rows3.append({"participant_id": pid, "interview_date": day.date().isoformat(),
                          "season": season, "energy_kcal": round(e),
                          "fat_g": round(e * rng.uniform(0.28, 0.4) / 9, 1),
                          "protein_g": round(e * rng.uniform(0.13, 0.19) / 4, 1),
                          "ldl": round(rng.normal(3.2, 0.8), 2)})
    return {"meal": pd.DataFrame(rows), "meal_ts": pd.DataFrame(rows2),
            "seasons": pd.DataFrame(rows3)}


def _wide_tables() -> dict[str, pd.DataFrame]:
    """probes/wp14_hints.py (``default_rng(77)``, n = 400): a raw 152-item FFQ on a 9-category
    frequency code, a web recall's portions (206 foods), 120 food groups in whole grams, and
    minute-level accelerometer counts (1,440 columns)."""
    rng = np.random.default_rng(77)
    n = 400
    base = pd.DataFrame({"participant_id": np.arange(1, n + 1), "age": rng.integers(30, 75, n),
                         "sex": rng.choice([1, 2], n), "energy_kcal": rng.normal(2000, 450, n).round()})
    p9 = np.array([0.30, 0.15, 0.15, 0.12, 0.10, 0.08, 0.05, 0.03, 0.02])
    ffq = pd.DataFrame({f"FFQ{j:03d}": rng.choice(np.arange(1, 10), n, p=rng.permutation(p9))
                        for j in range(1, 153)})
    foods = [f"food_{j:03d}_portions" for j in range(1, 207)]
    webq = pd.DataFrame({c: rng.choice([0, 1, 2, 3, 4, 5], n, p=[.78, .12, .05, .03, .01, .01])
                         for c in foods})
    groups = pd.DataFrame({f"fg{j:03d}_g": np.where(rng.random(n) < 0.55, 0,
                                                    rng.lognormal(4, 0.8, n)).round().astype(int)
                           for j in range(1, 121)})
    acc = pd.DataFrame({f"min_{j:04d}": np.where(rng.random(n) < 0.35, 0,
                                                 rng.lognormal(5.5, 1.3, n)).round().astype(int)
                        for j in range(1, 1441)})
    return {"ffq_codes": pd.concat([base, ffq], axis=1), "webq": pd.concat([base, webq], axis=1),
            "food_groups_g": pd.concat([base, groups], axis=1),
            "accelerometer": pd.concat([base.drop(columns="energy_kcal"), acc], axis=1)}


# ═════════════════════════════════════════════════════════════════════════════
# 0 · The leash: "high" only where the values agree with the name
# ═════════════════════════════════════════════════════════════════════════════


def test_0_a_role_is_proposed_high_only_where_the_values_corroborate_the_name(tmp_path):
    """Verifier, unasked assumptions: "Role proposals rated 'high' where only the name, or a `g`
    word, corroborates, with a generic reason that hides the doubt." The roles question marks every
    proposal below high (RolesAsk.tsx: ``info.confidence !== "high"``), so a high name-only reading
    was a silent default. Expected, across the gate's fixtures: an energy or energy-bearing nutrient
    proposal is high exactly when an independent check passes (NumPy r of total energy with
    4P + 4C + 9F at least 0.7; SciPy's one-sided Pearson test of a nutrient against the chosen
    energy at 1%); a name-only reading is medium and its reason says what was, or was not,
    checked."""
    from scipy import stats

    dev = _devices()
    rng = np.random.default_rng(8)
    n = 300
    no_macros = pd.DataFrame({"energy_kcal": rng.normal(2100, 450, n).round(),
                              "sodium_mg": rng.normal(3200, 800, n).round(),
                              "glucose": rng.normal(100, 12, n).round()})
    no_energy = pd.DataFrame({"protein_g": rng.normal(80, 20, n).round(1),
                              "fat_g": rng.normal(75, 20, n).round(1),
                              "glucose": rng.normal(100, 12, n).round()})
    cases = {
        # name: (frame, target, the energy column and macros this test names)
        "fitbit": (dev["fitbit"], "weight_kg",
                   ("energy_kcal", {"protein": "protein_g", "carbohydrate": "carbohydrate_g",
                                    "fat": "fat_g"})),
        "actilife": (dev["actilife"], "hba1c",
                     ("Energy (kcal)", {"protein": "Protein (g)", "carbohydrate": "Carbohydrate (g)",
                                        "fat": "Total Fat (g)"})),
        "recalls": (pd.read_csv(SAMPLES / "dietary_recalls.csv"), "hba1c",
                    ("energy_kcal", {"protein": "protein_g", "carbohydrate": "carbohydrate_g",
                                     "fat": "fat_g"})),
        "no_macros": (no_macros, "glucose", ("energy_kcal", {})),
        "no_energy": (no_energy, "glucose", (None, {})),
    }
    for label, (frame, target, (energy, macros)) in cases.items():
        roles = _roles(_write(frame, tmp_path, f"{label}.csv"), lens=["dietary"],
                       target=target)["columns"]
        if energy is not None:
            p = roles[energy]
            assert p["proposed"] == "energy", (label, p)
            if macros:
                r = float(np.corrcoef(frame[energy], _reconstructed(frame, macros))[0, 1])
                assert r >= 0.7, (label, r)
                assert p["confidence"] == "high" and f"r = {r:.2f}" in p["reason"], (label, p)
            else:
                assert p["confidence"] == "medium", (label, p)
                assert "no macronutrients here to check it against" in p["reason"], p
        for name in macros.values():
            p = roles[name]
            test = stats.pearsonr(frame[name], frame[energy], alternative="greater")
            assert test.pvalue < 0.01
            assert (p["proposed"], p["confidence"]) == ("exposure", "high"), (label, p)
            assert f"r = {test.statistic:.2f}" in p["reason"], (label, p)
        for c, p in roles.items():
            # A dietary reading the name alone makes says so, and is never high.
            if p["reason"].startswith(("Named as", "Also named", "Named like total energy",
                                       "Named like an acquisition column",
                                       "Named like an identifier, but")):
                assert p["confidence"] != "high", (label, c, p)
    no_energy_roles = _roles(_write(no_energy, tmp_path, "ne.csv"), lens=["dietary"],
                             target="glucose")["columns"]
    for name in ("protein_g", "fat_g"):
        p = no_energy_roles[name]
        assert (p["proposed"], p["confidence"]) == ("exposure", "medium"), p
        assert p["reason"] == "Named as a nutrient that carries energy; only its name and unit say so."


# ═════════════════════════════════════════════════════════════════════════════
# WP13-1 · Nutrients and total energy: values must agree with the name
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("device, energy, intake, macros, target", [
    ("fitbit", "Calories", "energy_kcal", ("protein_g", "carbohydrate_g", "fat_g"), "weight_kg"),
    ("actilife", "Kcals", "Energy (kcal)", ("Protein (g)", "Carbohydrate (g)", "Total Fat (g)"),
     "hba1c"),
])
def test_wp13_1a_a_devices_energy_expenditure_is_not_total_energy_intake(tmp_path, device, energy,
                                                                         intake, macros, target):
    """Verifier (a): Fitbit's ``Calories`` and ActiLife's ``Kcals`` were proposed "energy (high):
    Total energy intake", the real intake became "covariate (low)", the implausible-intake finding
    counted "84 of 320 rows report `Kcals` below 500", and nutrients were adjusted against energy
    spent. Source check, the Fitabase Fitbit Data Dictionary, Daily Activity: "Calories integer
    Total estimated energy expenditure (in kilocalories)."

    Expected: the intake column is total energy (its NumPy r with 4P + 4C + 9F is near 1); the
    device column (r under 0.3, or below the intake's) is not, and its reason says why; the
    proposals, the energy finding and the implausible-intake finding are about the intake column;
    with the roles confirmed, the model reads only the intake column as total energy."""
    from turbotab.core.methods.energy import total_energy_columns

    assert FITABASE_CALORIES.startswith("Total estimated energy expenditure")
    frame = _devices()[device]
    recon = _reconstructed(frame, dict(zip(("protein", "carbohydrate", "fat"), macros)))
    r_device = float(np.corrcoef(frame[energy], recon)[0, 1])
    r_intake = float(np.corrcoef(frame[intake], recon)[0, 1])
    assert r_intake > 0.99 and r_device < r_intake - 0.5, (r_device, r_intake)
    path = _write(frame, tmp_path, f"{device}.csv")
    roles = _roles(path, lens=["dietary"], target=target, purpose="inference")["columns"]
    assert roles[intake]["proposed"] == "energy" and roles[intake]["confidence"] == "high"
    assert roles[energy]["proposed"] == "covariate" and roles[energy]["confidence"] == "low"
    assert roles[energy]["reason"].startswith(("Named like total energy", "Also named as energy"))
    assert f"{r_device:.2f}" in roles[energy]["reason"], roles[energy]

    out = _proposals(path, lens=["dietary"], target=target, purpose="inference")
    assert out["energy"]["energy_column"] == intake
    assert out["energy"]["nutrients"] == list(macros)
    for screen in out["exclusions"]:
        assert screen["rule"]["column"] == intake
    coach = (out["coach"].get("exclusions") or {}).get("text", "")
    assert f"`{energy}`" not in coach
    found = _findings(path, lens=["dietary"], target=target)
    for f in found:
        if f["id"].startswith(("pack::dietary::implausible_intake",
                               "pack::dietary::energy_adjustment")):
            assert energy not in f["affected_columns"] and f"`{energy}`" not in f["summary"], f
            assert intake in f["affected_columns"], f
    implausible = [f for f in found if f["id"].startswith("pack::dietary::implausible_intake")]
    outside = int(((frame[intake] < 500) | (frame[intake] > 5000)).sum())
    assert (len(implausible) == 1) == (outside > 0)
    confirmed = {c: p["proposed"] for c, p in roles.items()}
    assert total_energy_columns([intake, energy, *macros], confirmed) == [intake]


def test_wp13_1b_a_unit_is_read_as_a_whole_expression(tmp_path):
    """Verifier (b): ``protein_g_kg`` (g/kg body weight), ``fat_g_1000kcal``, ``fibre_g_MJ`` and
    ``alcohol_g_week`` each passed as "its name and its unit agree, and its values fit a day's
    intake" and were proposed "exposure (high): A nutrient that carries energy"; the energy card
    offered density on a value already per energy. Expected: each unit is read whole (per body
    weight, per energy, per week), none carries a day's energy, none is an energy-adjustment target,
    and the reason says it is no day's amount. And the lower bound: a bare ``protein`` holding
    g/kg values carries under 1% of a day's energy (FAO 4 kcal/g × its NumPy median, against the
    1% floor read leniently in kJ), so its values withdraw the name."""
    from turbotab.core.methods.energy import energy_factor, unit_of
    from turbotab.core.recognizers import intake_check
    from turbotab.core.stages.proposals import energy_bearing

    expected = {"protein_g_kg": "per_body", "fat_g_1000kcal": "density", "fibre_g_MJ": "density",
                "alcohol_g_week": "per_period", "fat_g_per_1000_kcal": "density",
                "protein_g_per_kg_bw": "per_body", "glucose_mg_dl": "concentration",
                "protein_g": "grams", "protein_g_day": "grams", "Protein (g)": "grams",
                "energy_kcal_day": "kcal", "sodium_mg": "milligrams"}
    for name, unit in expected.items():
        assert unit_of(name) == unit, (name, unit_of(name))
    for name in ("protein_g_kg", "fat_g_1000kcal", "fibre_g_MJ", "alcohol_g_week"):
        assert energy_factor(name).factor is None and not energy_bearing(name), name

    rng = np.random.default_rng(56)
    n = 400
    ei = rng.normal(2100, 400, n).round()
    frame = pd.DataFrame({"participant_id": np.arange(n) + 1, "age": rng.integers(65, 90, n),
                          "energy_kcal": ei, "protein_g_kg": (rng.normal(1.1, 0.25, n)).round(2),
                          "fat_g_1000kcal": rng.normal(37, 6, n).round(1),
                          "fibre_g_MJ": rng.normal(3.0, 0.6, n).round(2),
                          "alcohol_g_week": rng.gamma(1.5, 40, n).round(),
                          "carb_g": (ei * 0.5 / 4).round(1), "grip_kg": rng.normal(30, 7, n).round(1)})
    path = _write(frame, tmp_path, "per_kg.csv")
    roles = _roles(path, lens=["dietary"], target="grip_kg")["columns"]
    for name, words in (("protein_g_kg", "per kg of body weight"),
                        ("fat_g_1000kcal", "per unit of energy"), ("fibre_g_MJ", "per unit of energy"),
                        ("alcohol_g_week", "per week, month or year")):
        p = roles[name]
        assert p["proposed"] == "exposure" and p["confidence"] == "medium", p
        assert p["reason"] == f"A nutrient intake {words}: an exposure, not a day's amount.", p
    out = _proposals(path, lens=["dietary"], target="grip_kg")
    assert out["energy"]["nutrients"] == ["carb_g"]
    left = {e["column"]: e["reason"] for e in out["energy"]["not_adjusted"]}
    assert left["fat_g_1000kcal"] == left["fibre_g_MJ"] == "already a share of energy"

    # The floor: protein named bare, holding g/kg values.
    g_kg = frame["protein_g_kg"]
    assert float(g_kg.median()) * ATWATER["protein"] * KCAL_PER_KJ / float(ei.mean()) < 0.01
    check = intake_check("protein", g_kg, energy=frame["energy_kcal"])
    # Recognition's leash (BLUEPRINT §14): protein's floor is the lowest acceptable range of any age
    # group (IOM: 5–20% for children 1–3), 5%.
    assert check is not None and not check.corroborated and "under 5%" in check.why
    real = (ei * 0.15 / 4).round(1)
    assert intake_check("protein", real, energy=frame["energy_kcal"]).corroborated


def test_wp13_1c_the_framingham_ffq_codebook_is_read(tmp_path):
    """Verifier (c): the Framingham FFQ file's ``NUT_CALOR``, ``NUT_PROT``, ``NUT_CARBO``,
    ``NUT_ALCO`` and ``NUT_SATFAT`` read as no nutrient at all ("nut" is a food word). Source check,
    the FHS coding manual (dbGaP phd001373): "nutrient fields starting with NUT_", "NUT_CALOR
    DERIVED FIELD: CALORIES, (kcal)", "NUT_PROT … PROTEIN, (gm)". Expected: the codebook is read
    with its units; on an FFQ table built as the file is (energy = 4P + 4C + 7A + 9 × total fat
    in parts), total energy and the macronutrients are proposed with high confidence, no nutrient
    is a covariate, and a nut in grams is still a food."""
    from turbotab.core.methods.energy import energy_factor, unit_of
    from turbotab.core.recognizers import read_nutrient, reads_as_total_energy

    assert reads_as_total_energy("NUT_CALOR") and unit_of("NUT_CALOR") == "kcal"
    for name, (macro, part) in {"NUT_PROT": ("protein", None), "NUT_CARBO": ("carbohydrate", None),
                                "NUT_ALCO": ("alcohol", None), "NUT_SATFAT": ("fat", "sfa"),
                                "NUT_MONFAT": ("fat", "mufa"), "NUT_POLY": ("fat", "pufa"),
                                "NUT_DTFIB": ("fiber", None)}.items():
        reading = read_nutrient(name)
        assert (reading.macro, reading.part, reading.source) == (macro, part, "fhs"), name
        assert unit_of(name) == "grams"
    assert energy_factor("NUT_PROT").factor == ATWATER["protein"]
    assert read_nutrient("nut_g") is None  # nuts in grams: a food
    rng = np.random.default_rng(88)
    n = 300
    P = rng.normal(80, 20, n).clip(20); C = rng.normal(240, 60, n).clip(50)
    A = rng.gamma(1, 6, n); F = rng.normal(70, 18, n).clip(15)
    frame = pd.DataFrame({"SHAREID": np.arange(n) + 1, "NUT_CALOR": (4 * P + 4 * C + 7 * A + 9 * F).round(),
                          "NUT_PROT": P.round(1), "NUT_CARBO": C.round(1), "NUT_ALCO": A.round(1),
                          "NUT_SATFAT": (F * 0.35).round(1), "NUT_MONFAT": (F * 0.38).round(1),
                          "NUT_POLY": (F * 0.2).round(1), "NUT_SODIUM": rng.normal(2800, 700, n).round(),
                          "ldl": rng.normal(130, 30, n).round()})
    roles = _roles(_write(frame, tmp_path, "fhs.csv"), lens=["dietary"], target="ldl")["columns"]
    assert (roles["NUT_CALOR"]["proposed"], roles["NUT_CALOR"]["confidence"]) == ("energy", "high")
    # Recognition's leash (BLUEPRINT §14, 2026-10-03): a codebook name is read, never trusted; "high"
    # is earned by the values. Independently (NumPy): protein and carbohydrate carry well over 5% of
    # the energy they add up to; the fat parts rise with total energy at r >= 0.3; alcohol (a gamma
    # draw, under 5% of energy) and sodium (drawn apart from energy) do neither, and stand as their
    # codebook's reading, medium, with the reason saying so.
    energy = frame["NUT_CALOR"]
    expected = {}
    for name in ("NUT_PROT", "NUT_CARBO", "NUT_ALCO", "NUT_SATFAT", "NUT_MONFAT", "NUT_POLY",
                 "NUT_SODIUM"):
        r = float(np.corrcoef(frame[name], energy)[0, 1])
        macro = {"NUT_PROT": "protein", "NUT_CARBO": "carbohydrate", "NUT_ALCO": "alcohol"}.get(name)
        share = float(np.median(frame[name] * ATWATER[macro] / energy)) if macro else 0.0
        expected[name] = "high" if (r >= 0.3 or share >= 0.05) else "medium"
    assert expected == {"NUT_PROT": "high", "NUT_CARBO": "high", "NUT_ALCO": "medium",
                        "NUT_SATFAT": "high", "NUT_MONFAT": "high", "NUT_POLY": "high",
                        "NUT_SODIUM": "medium"}, expected
    for name, confidence in expected.items():
        assert roles[name]["proposed"] == "exposure", (name, roles[name])
        assert roles[name]["confidence"] == confidence, (name, roles[name])
        if confidence == "medium":
            assert roles[name]["reason"].startswith("Named as"), roles[name]


# ═════════════════════════════════════════════════════════════════════════════
# WP13-2 · Trial arms, treatments and acquisition words stay what they are
# ═════════════════════════════════════════════════════════════════════════════


def test_wp13_2_an_identifier_names_units_and_acquisition_and_weights_need_their_context(tmp_path):
    """Verifier: in a three-arm trial ``trt_id`` and ``tx_id`` (1/2/3) were "identifier (high):
    Names each unit; `3` units, up to `104` rows each", and the roles' ``repeats`` named the arm as
    the repeating unit; ``sample_wt`` (tissue mass on a metabolomics sheet) was "design: A sampling
    weight"; ``plate`` (small/large, the intervention of a plate-size study) was excluded under
    prediction as an acquisition column. Expected, from pandas' counts: an ``_id`` holding three
    values on 312 rows is a group code kept in the model (and offered as categorical), never the
    unit; a sampling weight read from an ambiguous name needs the table to name its survey design;
    an acquisition word needs an assay (an assay lens, or another acquisition column)."""
    fx = _three_arms_and_sheets()
    for key, name, lens in (("trt", "trt_id", ["clinical"]), ("tx", "tx_id", ["dietary"])):
        frame = fx[key]
        assert frame[name].nunique() == 3 and frame[name].value_counts().max() == 104
        for purpose in ("inference", "prediction"):
            out = _roles(_write(frame, tmp_path, f"{key}.csv"), lens=lens, target="sbp_change",
                         purpose=purpose)
            p = out["columns"][name]
            assert p["proposed"] in ("exposure", "covariate") and p["confidence"] == "low", p
            assert p["reason"] == ("Named like an identifier, but `3` values on `312` rows: a "
                                   "group code, not a unit.")
            assert out["repeats"] is None
            assert name in [c["column"] for c in out["categorical"]]
            assert out["columns"]["USUBJID"]["proposed"] == "identifier"

    sheet = _write(fx["sheet"], tmp_path, "sheet.csv")
    for purpose in ("inference", "prediction"):
        p = _roles(sheet, lens=["metabolomics"], target="age", purpose=purpose)["columns"]["sample_wt"]
        assert p["proposed"] != "design" and "sampling weight" not in p["reason"], p
    surveyed = fx["sheet"].assign(strata=np.repeat([1, 2, 3], 30), psu=np.tile([1, 2], 45))
    p = _roles(_write(surveyed, tmp_path, "surveyed.csv"), lens=["metabolomics"],
               target="age")["columns"]["sample_wt"]
    assert p["proposed"] == "design", p  # beside strata and PSUs, it is the survey's weight

    plate = _write(fx["plate"], tmp_path, "plate.csv")
    assert set(fx["plate"]["plate"]) == {"small", "large"}
    for purpose in ("inference", "prediction"):
        p = _roles(plate, lens=["dietary"], target="kcal_eaten", purpose=purpose)["columns"]["plate"]
        assert p["proposed"] == "covariate" and p["kind"] is None, (purpose, p)
        assert p["reason"].startswith("Named like an acquisition column")
    # Under an assay lens the same word is acquisition, and its role follows the purpose.
    p = _roles(plate, lens=["metabolomics"], target="kcal_eaten",
               purpose="prediction")["columns"]["plate"]
    assert (p["kind"], p["proposed"]) == ("acquisition", "excluded")


# ═════════════════════════════════════════════════════════════════════════════
# WP13-5 · An energy unit nothing settles is recorded before anything is screened in it
# ═════════════════════════════════════════════════════════════════════════════


def test_wp13_5_toddlers_kj_and_two_day_totals_wait_for_their_unit(tmp_path):
    """Verifier: toddlers' energy at the SMILE cohort's distribution (Table 2: "Energy (kJ) all
    foods 4040 (±954.7 SD)") in a column named ``energy`` was read as kcal, assumed; the finding
    said "76 records report an implausible daily intake", the coach "likely over-reporting", and the
    500–5,000 kcal screen was offered unrefused; two recalls summed were the same. Expected: the
    unit is a proposal (``confirmed`` False); every screen shows its count and is refused until the
    unit is recorded; the finding asks for the unit with both readings counted (NumPy); no line
    says anyone misreports. Recorded as kJ, the toddlers' screens count in kJ (NumPy: 500–5,000
    kcal is 2,092–20,920 kJ); recorded as a two-day total in kcal, the screens read twice a day's
    bounds."""
    tod, adults = _toddlers_and_two_day()
    for frame, col, target in ((tod, "energy", "weight_kg"), (adults, "total_energy", "sbp")):
        path = _write(frame, tmp_path, f"{col}.csv")
        # Recognition's leash (BLUEPRINT §14): with no macronutrients beside it, the column is total
        # energy by its name alone, so its screens first wait for its role to be confirmed on its
        # own; confirmed, they wait for its unit.
        out = _proposals(path, lens=["dietary"], target=target)
        for screen in out["exclusions"]:
            assert screen["refused"].startswith(f"Refused until `{col}` is confirmed"), screen
        confirmed = set(frame.columns)
        out = _proposals(path, lens=["dietary"], target=target, settled=confirmed)
        assert (out["energy_unit"]["basis"], out["energy_unit"]["confirmed"]) == ("assumed", False)
        for screen in out["exclusions"]:
            assert screen["refused"].startswith(f"Refused until `{col}`'s unit is recorded"), screen
        coach = out["coach"]["exclusions"]["text"]
        assert "reporting" not in coach and coach.endswith("record its unit."), coach
        f = next(x for x in _findings(path, lens=["dietary"], target=target)
                 if x["id"] == "pack::dietary::implausible_intake")
        as_kcal = int(((frame[col] < 500) | (frame[col] > 5000)).sum())
        as_kj = int(((frame[col] < 500 * KCAL_PER_KJ) | (frame[col] > 5000 * KCAL_PER_KJ)).sum())
        assert f["title"] == f"The unit of `{col}` is not settled."
        assert f"Read in kcal, {as_kcal:,} of {len(frame):,} rows" in f["detail"]
        assert f"read in kJ, {as_kj:,} fall outside" in f["detail"]
        assert "records report" not in f["title"] and "misreport" in f["detail"]

    # Recorded: the toddlers in kJ.
    path = _write(tod, tmp_path, "tod.csv")
    units = {"energy": d.ColumnUnitSpec(unit="kj")}
    out = _proposals(path, lens=["dietary"], target="weight_kg", units=units,
                     settled=set(tod.columns))
    screen = next(e for e in out["exclusions"] if e["key"] == "sex_neutral_500_5000")
    kj = tod["energy"]
    assert screen["refused"] is None
    assert screen["affected"] == int(((kj < 500 * KCAL_PER_KJ) | (kj > 5000 * KCAL_PER_KJ)).sum())
    assert (screen["rule"]["low"], screen["rule"]["high"]) == (round(500 * KCAL_PER_KJ, 1),
                                                               round(5000 * KCAL_PER_KJ, 1))
    # Recorded: the adults' two-day total in kcal, screened at twice a day's bounds.
    path = _write(adults, tmp_path, "adults.csv")
    units = {"total_energy": d.ColumnUnitSpec(unit="kcal", days=2)}
    out = _proposals(path, lens=["dietary"], target="sbp", units=units,
                     settled=set(adults.columns))
    screen = next(e for e in out["exclusions"] if e["key"] == "sex_neutral_500_5000")
    two = adults["total_energy"]
    assert (screen["rule"]["low"], screen["rule"]["high"]) == (1000.0, 10000.0)
    assert screen["affected"] == int(((two < 1000) | (two > 10000)).sum()) and screen["refused"] is None
    f = next(x for x in _findings(path, lens=["dietary"], target="sbp", units=units)
             if x["id"] == "pack::dietary::implausible_intake")
    assert "over 2 days" in f["detail"]


def test_wp13_5_through_the_api_the_screen_is_refused_until_the_unit_is_recorded(tmp_path):
    """The same loop through the real server, as the verifier drove it: ``set_exclusions`` with the
    500–5,000 kcal rule on the toddlers' ``energy`` is refused (409) with exits that record the
    unit; recording kJ (``set_column_unit``) is accepted and states its sentence; the proposals then
    count in kJ; the rule read in kJ is accepted and the cohort drops exactly NumPy's count."""
    from turbotab.core.tests.acceptance.server_drive import local_server, open_project

    tod, _ = _toddlers_and_two_day()
    path = _write(tod, tmp_path, "toddlers.csv")
    with local_server(tmp_path / "home") as client:
        drive = open_project(client, path)
        drive.decide({"kind": "set_lens", "lenses": ["dietary"]})
        drive.reach("target")
        drive.decide({"kind": "set_target", "column": "weight_kg"})
        drive.answer("task", {"kind": "set_task", "column": "weight_kg", "task": "regression"})
        drive.reach("purpose")
        drive.decide({"kind": "set_purpose", "purpose": "inference"})
        drive.answer("grain", {"kind": "set_grain", "grain": "one_row_per_unit",
                               "id_column": "child_id"})
        drive.reach("roles")
        proposed = {p["column"]: p["proposed"] for p in drive.artifact("roles")["columns"]}
        assert proposed["energy"] == "energy"
        drive.decide({"kind": "set_roles", "roles": proposed})
        drive.reach("exclusions")
        rule = {"column": "energy", "low": 500, "high": 5000, "reason": "implausible intakes"}
        # Recognition's leash (BLUEPRINT §14): ``energy`` is total energy by its name alone (no
        # macronutrients here to check it against), so the bulk confirm recorded it unconfirmed;
        # the screen waits for its own confirmation first, then for its unit.
        r = drive.post({"kind": "set_exclusions", "rules": [rule]})
        assert r.status_code == 409, r.text
        error = r.json()["error"]
        assert error["code"] == "role_unconfirmed"
        # The readings ledger (BLUEPRINT §14.1): one confirmation decision for every reading;
        # ``confirm_role`` records stay valid, and this one is recorded as the ledger's.
        assert {"kind": "confirm_reading", "reading": "role", "column": "energy",
                "value": "energy"} in [x["decision"] for x in error["exits"]]
        drive.decide({"kind": "confirm_role", "column": "energy", "role": "energy"})
        r = drive.post({"kind": "set_exclusions", "rules": [rule]})
        assert r.status_code == 409, r.text
        error = r.json()["error"]
        assert error["code"] == "energy_unit_unconfirmed"
        exits = [x["decision"] for x in error["exits"]]
        assert {"kind": "set_column_unit", "column": "energy", "unit": "kj", "days": 1} in exits
        drive.decide({"kind": "set_column_unit", "column": "energy", "unit": "kj"})
        record = drive.view()["decisions"][-1]
        assert record["sentence"].startswith("`energy` was recorded as a day's energy in kJ")
        proposals = drive.artifact("proposals")
        assert proposals["energy_unit"]["basis"] == "decision"
        kj_rule = next(e["rule"] for e in proposals["exclusions"]
                       if e["key"] == "sex_neutral_500_5000")
        drive.decide({"kind": "set_exclusions", "rules": [kj_rule]})
        cohort = drive.artifact("cohort")
        kj = tod["energy"]
        expected = int(((kj < 500 * KCAL_PER_KJ) | (kj > 5000 * KCAL_PER_KJ)).sum())
        assert cohort["n_final"] == len(tod) - expected


# ═════════════════════════════════════════════════════════════════════════════
# WP13-6 · The fasting weight, named by where its analytes are recorded
# ═════════════════════════════════════════════════════════════════════════════


def test_wp13_6_a_fasting_analyte_is_read_by_where_it_is_recorded(tmp_path):
    """Verifier: FASTING_ANALYTES was a fixed name list. With LBDLDNSI (TRIGLY_J, "LDL-Cholesterol,
    NIH equation 2 (mmol/L)") or a fasting glucose renamed on import, the finding said "Use the
    dietary weights, not the examination weight", SETTLED, and never named WTSAF2YR ("Fasting
    Subsample 2 Year MEC Weight"). Source check, TRIGLY_J: "Specific sample weights for this
    subsample are included in this data file and should be used when analyzing these data."
    Expected: the analyte is recorded only where WTSAF2YR is positive (pandas), so the rule names
    WTSAF2YR, and the survey question offers it first; a WTSAF2YR beside no fasting analyte at all
    is named as a question, never SETTLED."""
    from turbotab.core.survey import offered

    assert TRIGLY_J["WTSAF2YR"] == "Fasting Subsample 2 Year MEC Weight"
    renamed, ldl = _nhanes_renamed()
    for frame, analytes in ((renamed, ["fasting_glucose"]), (ldl, ["LBDLDNSI", "LBDLDMSI"])):
        inside = frame["WTSAF2YR"] > 0
        for a in analytes:
            assert not (frame[a].notna() & ~inside).any() and frame[a].isna().mean() > 0.05
        path = _write(frame, tmp_path, f"{analytes[0]}.csv")
        f = next(x for x in _findings(path, lens=["dietary"], target=analytes[0])
                 if x["id"] == "pack::dietary::survey_weights")
        assert f["title"] == "Use the fasting subsample weight, `WTSAF2YR`."
        assert f"`{analytes[0]}`" in f["detail"] and "morning fasting subsample" in f["detail"]
        assert f["evidence"]["status"] == "CONVENTION"
        roles = {"SEQN": "identifier", "DR1TKCAL": "energy", "DR1TSFAT": "exposure",
                 "DR1TCARB": "exposure", "WTDRD1": "design", "WTMEC2YR": "design",
                 "WTSAF2YR": "design", "SDMVSTRA": "design", "SDMVPSU": "design"}
        state = ProjectState(lens=["dietary"], target=analytes[0], roles=roles, purpose="inference")
        assert offered(state, None, frame)[0]["decision"]["weight"] == "WTSAF2YR"

    # WTSAF2YR beside no fasting analyte: the dietary weight, with the fasting weight named as the
    # user's question, and no SETTLED badge.
    plain = renamed.drop(columns=["fasting_glucose"]).assign(
        BMXBMI=np.random.default_rng(5).normal(28, 5, len(renamed)).round(1))
    f = next(x for x in _findings(_write(plain, tmp_path, "plain.csv"), lens=["dietary"],
                                  target="BMXBMI")
             if x["id"] == "pack::dietary::survey_weights")
    assert "`WTSAF2YR` weights the morning fasting subsample" in f["detail"]
    assert f["evidence"]["status"] != "SETTLED"
    assert "WTSAF2YR" in f["summary"]


# ═════════════════════════════════════════════════════════════════════════════
# WP14-4 · An age's unit the body sizes contradict is asked, not judged in
# ═════════════════════════════════════════════════════════════════════════════


def _cdc_modified_z(values: np.ndarray, months: np.ndarray, sex: np.ndarray) -> np.ndarray:
    """CDC's modified z-score, written here from the CDC's description ("computed by extrapolating
    one-half of the distance between 0 and +2 (or between 0 and -2) z-scores") over the CDC's
    weight-for-age LMS table (wtage.csv, as published), linear in age between the table's rows."""
    lms = pd.read_csv(Path(__file__).resolve().parents[2] / "detectors" / "data" / "wtage.csv")
    lms = lms[pd.to_numeric(lms["Sex"], errors="coerce").isin([1, 2])].astype(float)
    out = np.full(len(values), np.nan)
    for code in (1.0, 2.0):
        part = lms[lms["Sex"] == code].sort_values("Agemos")
        at = (sex == code) & (months >= part["Agemos"].min()) & (months <= part["Agemos"].max())
        L = np.interp(months[at], part["Agemos"], part["L"])
        M = np.interp(months[at], part["Agemos"], part["M"])
        S = np.interp(months[at], part["Agemos"], part["S"])
        sd2_up = M * (1 + 2 * L * S) ** (1 / L) - M
        sd2_down = M - M * (1 - 2 * L * S) ** (1 / L)
        x = values[at]
        out[at] = np.where(x >= M, (x - M) / (sd2_up / 2), (x - M) / (sd2_down / 2))
    return out


def test_wp14_4_an_age_in_months_read_as_years_is_asked_not_judged(tmp_path):
    """Verifier: an under-five survey's ``age`` holding months (6–59) was read as years; 444
    toddlers became "adult values outside 47.2–157.4 kg … unusual but real, and they must be kept",
    150 of 156 "children aged 2 to 19" biologically implausible, under a SETTLED badge routed to
    exclusions; ``age_m`` and ``age_mths`` were unread. Expected: read as years, the weights flag
    most rows (pandas: every row read as an adult is under the adult 1st percentile), which no
    reference does to the people it describes, so the finding asks the unit of ``age``, judges no
    one by age, is not SETTLED and offers no range; recorded as months (``set_column_unit``), the
    children aged 2 and over are judged by the CDC charts and the count of implausible weights
    equals the modified z computed here (below −5 or above 8)."""
    from turbotab.core.detectors import plausibility

    frame = _under_five()
    path = _write(frame, tmp_path, "under5.csv")
    p01 = plausibility.reference()["variables"]["weight"]["improbable"]["p01"]
    as_adults = frame[frame["age"] >= 20]
    assert (as_adults["weight"] < float(p01)).mean() > 0.5
    found = [f for f in _findings(path, lens=["clinical"], target="hb")
             if f["id"].startswith("pack::clinical::impossible_vs_extreme")]
    assert len(found) == 1
    f = found[0]
    assert f["title"] == "The unit of `age` is in question."
    assert f["evidence"]["status"] != "SETTLED"
    assert "unusual but real" not in f["detail"] and "biologically implausible" not in f["detail"]
    assert f["routes_to"] is None and not f.get("repairs")

    units = {"age": d.ColumnUnitSpec(unit="months")}
    found = [f for f in _findings(path, lens=["clinical"], target="hb", units=units)
             if f["id"].startswith("pack::clinical::impossible_vs_extreme")]
    z = _cdc_modified_z(frame["weight"].to_numpy(float), frame["age"].to_numpy(float),
                        frame["sex"].to_numpy(float))
    flagged = int(((z < -5) | (z > 8)).sum())
    assert flagged == 0  # weights drawn near the WHO medians
    reading = plausibility.read(frame, units)
    weight = next(e for e in reading["columns"] if e["column"] == "weight")
    assert weight["children"]["n_flagged"] == flagged
    assert weight["children"]["n_read"] == int((frame["age"] >= 24).sum())
    assert weight["n_outside_central_98"] == 0 and not found  # nothing to report
    for name in ("age_m", "age_mths", "AgeMonths", "child_age_months", "agemos"):
        assert plausibility.age_reading(frame.rename(columns={"age": name}))["unit"] == "months"
    # An adult table's bare ``age`` is still read in years, without a question.
    adults = pd.read_csv(SAMPLES / "clinical_longitudinal.csv")
    assert plausibility.read(adults)["age"]["basis"] == "assumed"
    assert not plausibility.read(adults)["age"].get("in_question")


# ═════════════════════════════════════════════════════════════════════════════
# WP14-6 · Repeats versus time points: spacing alone asks
# ═════════════════════════════════════════════════════════════════════════════


def test_wp14_6_spacing_alone_never_states_repeats_or_time_points():
    """Verifier: (a) an OGTT drawn 0–120 minutes on one morning was stated "repeats" ("every one of
    a unit's records carries the same date"), its mean recommended; per-draw timestamps 15 minutes
    apart read the same, the spacing being in days; (b) four 24-hour recalls a season apart were
    stated "time points: averaging them destroys the signal". Expected: each is asked, the sentence
    naming what orders the day (``time_min``; clock times 15 minutes apart, from pandas) or the
    intake reported on each record; a schedule the study names as visits (the shipped clinic
    visits, about 90 days apart) is still stated, and same-date duplicates with nothing ordering
    the day still read as repeats."""
    from turbotab.core.detectors import repeats as reading
    from turbotab.core.interview import _repeat_kind_gate
    from turbotab.core.decisions import GrainSpec

    fx = _repeats_fixtures()
    gaps = pd.to_datetime(fx["meal_ts"]["sample_time"]).groupby(fx["meal_ts"]["participant_id"]).diff()
    assert float(gaps.dropna().dt.total_seconds().median()) == 15 * 60
    for key, lens, says in (("meal", ["clinical"], "`time_min` orders them within the day"),
                            ("meal_ts", ["clinical"], "clock times 15 minutes apart"),
                            ("meal", ["dietary"], "`time_min` orders them within the day"),
                            ("seasons", ["dietary"], "`energy_kcal` is an intake reported")):
        r = reading.read(fx[key], "participant_id", lens)
        assert r["stated"] is False and r["reading"] is None, (key, r["sentence"])
        assert says in r["sentence"], (key, r["sentence"])
        state = ProjectState(grain=GrainSpec(grain="repeated", id_column="participant_id"))
        assert _repeat_kind_gate(state, {"repeats": r, "units": {"column": "participant_id"}}) is None
    visits = pd.read_csv(SAMPLES / "clinical_longitudinal.csv")
    r = reading.read(visits, "subject_id", ["clinical"])
    assert (r["reading"], r["stated"]) == ("time_points", True) and "`visit`" in r["sentence"]
    # Recognition's leash (BLUEPRINT §14 rule 1, 2026-10-03): a date the same on every row of a unit
    # is never repeats evidence (it reads exactly as a birth or randomization date does), so
    # same-date records with nothing ordering the day are asked, the sentence naming the date.
    duplicates = fx["meal"].drop(columns=["time_min"])
    r = reading.read(duplicates, "participant_id", ["clinical"])
    assert (r["reading"], r["stated"]) == (None, False), r["sentence"]
    assert "the same on every row of a unit" in r["sentence"]


# ═════════════════════════════════════════════════════════════════════════════
# WP14-7 · Whole numbers hint genomics only when they spread like genes
# ═════════════════════════════════════════════════════════════════════════════


def _iqr_log_means(frame: pd.DataFrame) -> float:
    """The interquartile range of log10(1 + mean) over the non-negative whole-number columns."""
    cols = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])
            and (frame[c] >= 0).all() and np.all(np.mod(frame[c].to_numpy(float), 1) == 0)]
    q1, q3 = np.percentile(np.log10(1 + frame[cols].mean().to_numpy(float)), [25, 75])
    return float(q3 - q1)


def test_wp14_7_a_wide_block_of_whole_numbers_is_no_count_matrix_unless_it_spreads_like_genes(
        tmp_path):
    """Verifier: any table with 100 or more non-negative integer columns hinted genomics ("what a
    count matrix looks like"): a raw 152-item FFQ on a 9-category code, web-recall portions, food
    groups in whole grams and minute accelerometer counts; each then drew a critical "The lens you
    chose and the table disagree" under its own lens, the FFQ even under the survey lens.
    Expected, from NumPy's interquartile range of the columns' log mean counts: the non-assay
    tables spread under a tenth of a decade and get no genomics hint and no contradiction; the
    public count matrices GSE60450 and GSE147507 spread over two decades and are still hinted."""
    from turbotab.core.detectors import lenses

    for name, frame in _wide_tables().items():
        assert _iqr_log_means(frame) < 0.1, name
        assert "genomics" not in [h["lens"] for h in lenses.hints(frame)], name
        for lens in (["dietary"], ["survey"], ["clinical"]):
            assert lenses.contradiction_finding(frame, lens) is None, (name, lens)
    path = _write(_wide_tables()["ffq_codes"], tmp_path, "ffq.csv")
    assert "voice::lens_contradiction" not in {f["id"] for f in _findings(path, lens=["survey"])}
    raw = pd.read_csv(HERE / "GSE60450_Lactation-GenewiseCounts_5000.tsv.gz", sep="\t")
    gse60450 = raw.drop(columns=["EntrezGeneID", "Length"]).T.reset_index(drop=True)
    gse60450.columns = [f"g{e}" for e in raw["EntrezGeneID"]]
    raw = pd.read_csv(HERE / "GSE147507_RawReadCounts_Human_5000.tsv.gz", sep="\t", index_col=0)
    gse147507 = raw.T.reset_index(drop=True)
    gse147507.columns = [f"g_{g}" for g in raw.index]
    for frame in (gse60450, gse147507):
        assert _iqr_log_means(frame) > 2.0
        assert "genomics" in [h["lens"] for h in lenses.hints(frame)]


# ═════════════════════════════════════════════════════════════════════════════
# WP15-3 · An energy-related outcome pushes the dispute; an unplaced one is asked
# ═════════════════════════════════════════════════════════════════════════════


def _child_diet(outcome: str, values: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    n = len(values)
    P = rng.normal(40, 10, n).clip(10); C = rng.normal(150, 35, n).clip(40)
    F = rng.normal(45, 12, n).clip(10)
    return pd.DataFrame({"child_id": np.arange(n) + 1, "sex": rng.choice([1, 2], n),
                         "agemos": rng.integers(6, 60, n), "energy_kcal": (4 * P + 4 * C + 9 * F).round(),
                         "protein_g": P.round(1), "fat_g": F.round(1), "carb_g": C.round(1),
                         outcome: values})


@pytest.mark.parametrize("name, kind", [
    ("zwei", "body weight"), ("zwfl", "body weight"), ("zbmi", "BMI"), ("zlen", "child growth"),
    ("zts", "adiposity"), ("WHZ", "body weight"), ("waz", "body weight"), ("baz", "BMI"),
    ("wfh_z", "body weight"), ("FMI", "adiposity"), ("WHtR", "waist size"), ("VAT", "adiposity"),
    ("BW", "body weight"), ("bw_g", "body weight"), ("BMXHIP", "hip size"),
    ("DXDTOPF", "adiposity"), ("WC", "waist size"), ("incident_dm", "diabetes"),
    ("DM", "diabetes"), ("diab_status", "diabetes"),
])
def test_wp15_3_growth_indices_and_standard_abbreviations_push_the_dispute(tmp_path, name, kind):
    """Verifier: zwei, zwfl, WHZ, FMI and WHtR gave ``outcome_dispute`` False and a CONVENTION badge;
    waz/baz, VAT, BW/bw_g, BMXHIP, DXDTOPF, WC, incident_dm/DM and diab_status were missed. Source
    check, the WHO anthro package: "zwei Weight-for-age z-score", "zwfl Weight-for-length/height
    z-score", "zbmi BMI-for-age z-score", "zlen Length/Height-for-age z-score". Expected: each reads
    as its kind; on a children's dietary table the energy card carries the DISPUTED note and the
    energy finding the DISPUTED badge."""
    from turbotab.core.methods.dietary_caveats import energy_related

    if name in WHO_ANTHRO:
        assert WHO_ANTHRO[name].endswith("z-score")
    assert energy_related(name) == kind
    rng = np.random.default_rng(12)
    frame = _child_diet(name, rng.normal(0, 1, 300).round(2), rng)
    path = _write(frame, tmp_path, f"{name}.csv")
    card = _proposals(path, lens=["dietary"], target=name)["energy"]
    assert card["outcome_dispute"]["kind"] == kind and card["outcome_dispute"]["basis"] == "name"
    assert card["outcome_dispute"]["evidence"]["status"] == "DISPUTED"
    energy = next(f for f in _findings(path, lens=["dietary"], target=name)
                  if f["id"] == "pack::dietary::energy_adjustment")
    assert energy["evidence"]["status"] == "DISPUTED"
    assert f"`{name}` reads as {kind}" in energy["why_it_matters"]


def test_wp15_3_an_outcome_is_placed_by_its_values_or_its_dispute_is_asked(tmp_path):
    """The leash for outcomes no name list can hold. Expected: an outcome named ``index_a`` that
    tracks the table's ``weight_kg`` at |r| ≥ 0.7 (NumPy) reads as body weight by its values; one
    nothing places (``cognition``) carries the dispute as a condition the user answers, DISPUTED,
    never dropped; one the names place elsewhere (an LDL change, NHANES's LBDLDNSI, glucose,
    HbA1c, systolic pressure) keeps the CONVENTION badge and no note; WHO anthro's flag ``fwei`` is
    no index."""
    from turbotab.core.methods.dietary_caveats import energy_related, outcome_relation

    assert energy_related("fwei") is None and energy_related("zwei_flag") is None
    rng = np.random.default_rng(21)
    n = 300
    weight = rng.normal(14, 3, n)
    index_a = (weight - 14) / 3 + rng.normal(0, 0.3, n)
    r = float(np.corrcoef(weight, index_a)[0, 1])
    assert r >= 0.7
    frame = _child_diet("index_a", index_a.round(3), rng).assign(weight_kg=weight.round(2))
    relation = outcome_relation("index_a", frame)
    assert (relation["kind"], relation["basis"], relation["via"]) == ("body weight", "values",
                                                                      "weight_kg")
    path = _write(frame, tmp_path, "index_a.csv")
    card = _proposals(path, lens=["dietary"], target="index_a")["energy"]
    assert card["outcome_dispute"]["basis"] == "values"
    assert card["notes"] == [f"`index_a` tracks `weight_kg` (r {r:.2f}), so it reads as body weight: "
                             f"adjusting for energy is disputed."]

    frame = _child_diet("cognition", rng.normal(100, 15, n).round(), rng)
    path = _write(frame, tmp_path, "cognition.csv")
    card = _proposals(path, lens=["dietary"], target="cognition")["energy"]
    assert card["outcome_dispute"]["basis"] == "unconfirmed" and card["outcome_dispute"]["kind"] is None
    assert card["notes"] == ["If `cognition` measures body size, adiposity or diabetes, energy may "
                             "be on its causal path, and adjusting for it is disputed."]
    energy = next(f for f in _findings(path, lens=["dietary"], target="cognition")
                  if f["id"] == "pack::dietary::energy_adjustment")
    assert energy["evidence"]["status"] == "DISPUTED"
    assert "Nothing here says whether `cognition` is energy-related" in energy["why_it_matters"]

    for name in ("LDLChange", "LBDLDNSI", "glucose", "hba1c", "sbp"):
        assert outcome_relation(name)["basis"] == "other", name
        frame = _child_diet(name, rng.normal(0, 1, n).round(2), rng)
        path = _write(frame, tmp_path, f"{name}.csv")
        card = _proposals(path, lens=["dietary"], target=name)["energy"]
        assert card["outcome_dispute"] is None and card["notes"] == [], name
        energy = next(f for f in _findings(path, lens=["dietary"], target=name)
                      if f["id"] == "pack::dietary::energy_adjustment")
        assert energy["evidence"]["status"] == "CONVENTION", name


# ═════════════════════════════════════════════════════════════════════════════
# The gate's other unasked assumptions
# ═════════════════════════════════════════════════════════════════════════════


def test_an_outcome_unit_is_stated_only_from_its_whole_suffix():
    """Verifier: ``units.from_name`` took the last word, so ``protein_g_kg`` (g/kg/day) stated "kg"
    and ``wbc_k_ul`` (10³/µL) "U/L" in estimand sentences. Expected: a suffix after another unit
    or a count's multiplier is a denominator, and nothing is stated; whole suffixes still are."""
    from turbotab.core.units import from_name, outcome_unit

    for name in ("protein_g_kg", "wbc_k_ul", "kcal_per_kg", "rbc_m_ul"):
        assert from_name(name) is None and outcome_unit(name) == (None, None), name
    for name, unit in (("glucose_mg_dl", "mg/dL"), ("ldl_mmol_l", "mmol/L"), ("weight_kg", "kg"),
                       ("serum_k_mmol_l", "mmol/L"), ("alt_ul", "U/L"), ("choline_mg_day", "mg/day")):
        assert from_name(name) == unit, name


def test_a_half_read_survey_design_asks_the_survey_question(tmp_path):
    """Verifier: with BRFSS-style columns, ``_PSU`` was design while ``_LLCPWT`` (the final weight)
    was a covariate, and the survey question was "not_applicable: No column reads as a survey
    weight", so the unweighted estimand was applied without asking; and ``survey.py`` kept a weight
    reader of its own. Expected: beside the PSU, ``_LLCPWT`` (every value positive) reads as the
    survey's weight in the roles and in the survey question, which is asked and offers it; one
    weight reader serves both, so a laboratory's ``sample_wt`` is no weight without a design."""
    from turbotab.core.survey import not_applicable_reason, offered, read_design

    rng = np.random.default_rng(2023)
    n = 400
    frame = pd.DataFrame({"SEQNO": 2023000001 + np.arange(n), "_STSTR": rng.integers(11011, 11099, n),
                          "_PSU": 2023000000 + rng.integers(1, 300, n),
                          "_LLCPWT": rng.lognormal(5.5, 0.9, n).round(2), "_AGEG5YR": rng.integers(1, 14, n),
                          "SEXVAR": rng.choice([1, 2], n), "FRUTDA2_": rng.gamma(2, 60, n).round(),
                          "_BMI5": rng.normal(2800, 550, n).round()})
    assert (frame["_LLCPWT"] > 0).all()
    roles = _roles(_write(frame, tmp_path, "brfss.csv"), lens=[], target="_BMI5",
                   purpose="inference")["columns"]
    assert roles["_PSU"]["proposed"] == "design" and roles["_LLCPWT"]["proposed"] == "design"
    state = ProjectState(target="_BMI5", purpose="inference",
                         roles={c: p["proposed"] for c, p in roles.items()})
    assert read_design(list(state.roles)).weights == ["_LLCPWT"]
    assert not_applicable_reason(state) is None
    assert [o["decision"].get("weight") for o in offered(state)] == ["_LLCPWT", None]
    assert read_design(["sample_wt", "Batch", "age"]).weights == []
    assert read_design(["sample_wt", "strata", "psu"]).weights == ["sample_wt"]
    # The design is no licence to read every "…wt" as a weight: NHANES's body weight (BMX_J
    # "BMXWT - Weight (kg)") beside SDMVSTRA and SDMVPSU is the body's.
    nhanes = ["SEQN", "SDMVSTRA", "SDMVPSU", "WTMEC2YR", "BMXWT", "BMXBMI", "LBXGLU"]
    assert read_design(nhanes).weights == ["WTMEC2YR"]


def test_an_energy_total_named_per_week_waits_for_its_days():
    """A total named per week (``kcal_week``) is no day's intake: its unit is a proposal until the
    days it totals are recorded, and the refusal's exits include a week's total (seven days)."""
    rng = np.random.default_rng(7)
    week = pd.DataFrame({"kcal_week": (rng.normal(2100, 400, 300) * 7).round(),
                         "glucose": rng.normal(100, 12, 300).round()})

    class _Store:
        columns = list(week.columns)
        n_rows = len(week)

        def materialize(self, cols, rows=None):
            return week[list(cols)]

    from turbotab.core.stages.proposals import energy_unit_reading

    assert energy_unit_reading(week, "kcal_week")["confirmed"] is False
    rule = d.ExclusionRule(column="kcal_week", low=500, high=5000, reason="implausible intakes")
    state = ProjectState(lens=["dietary"], target="glucose", roles={"kcal_week": "energy"})
    with pytest.raises(d.Refusal) as refused:
        d.validate(d.SetExclusions(rules=[rule]), {"state": state, "store": lambda: _Store()})
    assert refused.value.code == "energy_unit_unconfirmed"
    assert {"kind": "set_column_unit", "column": "kcal_week", "unit": "kcal", "days": 7} in [
        x["decision"] for x in refused.value.exits]
