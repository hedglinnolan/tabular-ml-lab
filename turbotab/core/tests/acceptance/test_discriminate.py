"""Corroboration must discriminate (BLUEPRINT §14.3, 2026-10-03): package DISCRIMINATE.

The fourth gate (``/private/tmp/turbotab-fix/ledger_run.json``, field ``gate``) found that the
readings ledger held, but six readers settled themselves on evidence that fits the alternatives as
well (repeating values read as a subject's identifier, a value that varies within units read as
time, "integer, at most 10 levels" read as the only codes worth asking about, a label count read as
free text, ``uL`` read as U/L, ``sex`` coded 1/2 read as the CDC coding), and one reader held a
confirmation the fit never read. The rule is now explicit:

* a kind of reading is settled by its values only through a test that rejects every plausible
  alternative it declares (``readings.KIND_RULES``), and each alternative has a fixture here
  showing the test does not settle it (section C); a kind without such a test is the user's;
* the consumer sets the question's scope (the fit asks about every whole-valued predictor, any type,
  any count of values; clustering reads the grain answer or the user; a sentence states a unit only
  once recorded);
* every confirmation is honored: for each consumer and each kind it reads, confirming each
  alternative produces that alternative's behavior (section D, over ``readings.CONSUMERS``), and
  every test driver answers from its fixture's declared truth (``truths.Truth``), never a constant;
* the ask stays light: one block confirmation settles exactly the readings it lists (section E).

Section A replays the gate's six open items and section B its nine unasked assumptions, draw for
draw from the gate's generators (``gate-ledger/p1``–``p12``, the seeds named at each), through the
real server where the gate drove it. Each server replay first runs the opening sequence with nothing
confirmed (the roles recorded in bulk as proposed) and checks that the fit asks or refuses and
computes no number (section F's criterion), then answers from the fixture's truth.

Expected values never come from the code under test: NumPy least squares, pandas, the published
EFSA Schofield tables and Black 2000's cut-offs, and quoted primary sources (fetched 2026-10-03).
"""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from turbotab.core import decisions as d
from turbotab.core import readings as R
from turbotab.core.decisions import ColumnUnitSpec, GrainSpec, ProjectState
from turbotab.core.tests.acceptance.server_drive import Truth, local_server, open_project
from turbotab.core.tests.stage_harness import Ingested
from turbotab.core.tests.truths import ASKING, asked

ROOT = Path(__file__).resolve().parents[4]

# ── primary sources, quoted (fetched 2026-10-03) ─────────────────────────────

BLUEPRINT_14_3 = ("A kind is settled by its values only when it declares its plausible alternatives "
                  "and a value test that rejects each of them.")
BLUEPRINT_NAMES = "Names never count as corroboration."
BLUEPRINT_DRIVERS = "Test drivers answer from the fixture's truth, never with a constant."
BLUEPRINT_BLOCK = ("A block confirmation settles exactly the readings it lists, each with the value "
                   "it shows.")
# MEPS HC-216 documentation (meps.ahrq.gov/data_stats/download_data/pufs/h216/h216doc.shtml):
MEPS_PID = ("A three-digit person number (PID) uniquely identifies each person within the DU. The "
            "variable DUPERSID is the combination of the variables DUID and PID.")
# NHANES 2017–2018 codebooks (wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/):
NHANES_CBC_J = {"LBXWBCSI": "White blood cell count (1000 cells/uL)",
                "LBXPLTSI": "Platelet count (1000 cells/uL)"}
NHANES_BIOPRO_J = {"LBXSATSI": "Alanine Aminotransferase (ALT) (U/L)"}
NHANES_DR1IFF_J = {"DR1_030Z": "Name of eating occasion",
                   "codes": {1: "Breakfast", 2: "Lunch", 3: "Dinner", 19: "Bebida", 91: "Other",
                             99: "Don't know"}}
# UK Biobank data coding 1001 (biobank.ndph.ox.ac.uk/showcase/coding.cgi?id=1001):
UKB_1001 = ("This is a hierarchical tree-structured dictionary which uses integers to represent "
            "categories or special values.")
# BRFSS 2022 codebook (cdc.gov/brfss/annual_data/2022/zip/codebook22_llcp-v2-508.zip):
BRFSS_STATE = ("Label: State FIPS Code", "SAS Variable Name: _STATE", "1 Alabama", "2 Alaska",
               "4 Arizona")
# pandas user guide, "Working with missing data" (pandas.pydata.org/docs/user_guide/
# missing_data.html): integers become floats when a value is missing.
PANDAS_NA = "the original data type will be coerced to np.float64 or object"
# The international yard and pound agreement (1959): 1 lb = 0.45359237 kg exactly.
LB_TO_KG = 0.45359237


def test_the_quoted_rule_is_the_blueprints_own():
    blueprint = " ".join((ROOT / "docs/turbotab-next/BLUEPRINT.md").read_text("utf-8").split())
    for sentence in (BLUEPRINT_14_3, BLUEPRINT_NAMES, BLUEPRINT_DRIVERS, BLUEPRINT_BLOCK):
        assert sentence in blueprint, sentence


# ═════════════════════════════════════════════════════════════════════════════
# The gate's generators (gate-ledger/p*.py), draw for draw
# ═════════════════════════════════════════════════════════════════════════════


def meps() -> pd.DataFrame:
    """p1_ids.py ``meps`` (rng 41001): 1,400 dwelling units; PID numbers people within each, from
    101, with 2xx/5xx/6xx for persons who joined later (MEPS documentation); DUPERSID = DUID ‖ PID.
    Households share BMI."""
    rng = np.random.default_rng(41001)
    rows = []
    for du in range(1400):
        duid = 2290000 + du * 7
        size = int(rng.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                              p=np.array([.25, .3, .15, .12, .07, .04, .02, .015, .01, .007, .005,
                                          .004, .002, .001]) / 0.994))
        pids = [101 + i for i in range(size)]
        for i in range(size):
            if rng.random() < 0.06:
                pids[i] = int(rng.choice([201, 202, 203, 501, 502, 601]))
        pids = list(dict.fromkeys(pids))
        for pid in pids:
            rows.append(dict(DUID=duid, PID=pid, DUPERSID=f"{duid}{pid}"))
    f = pd.DataFrame(rows)
    n = len(f)
    f["AGE19X"] = rng.integers(18, 85, n)
    f["fruit_servings"] = rng.gamma(2.0, 1.0, n).round(1)
    hh = pd.factorize(f["DUID"])[0]
    u = rng.normal(0, 2.0, hh.max() + 1)[hh]
    f["bmi"] = (27 + 0.04 * (f["AGE19X"] - 50) - 0.3 * f["fruit_servings"] + u
                + rng.normal(0, 4, n)).round(1)
    return f


def roster() -> pd.DataFrame:
    """p5_roster.py ``table`` (rng 45005): 900 households, a person line number 1..k in each."""
    rng = np.random.default_rng(45005)
    rows = []
    for h in range(900):
        k = int(min(16, 1 + rng.poisson(3.2)))
        if rng.random() < 0.02:
            k = int(rng.integers(11, 19))
        for p in range(1, k + 1):
            rows.append(dict(hhid=f"H{h:05d}", person_no=p))
    f = pd.DataFrame(rows)
    n = len(f)
    f["age_years"] = rng.integers(18, 80, n)
    f["sugary_drinks_wk"] = rng.poisson(3, n)
    hh = pd.factorize(f["hhid"])[0]
    u = rng.normal(0, 1.5, hh.max() + 1)[hh]
    f["waist_cm"] = (88 + 0.15 * (f.age_years - 45) + 0.6 * f.sugary_drinks_wk + u
                     + rng.normal(0, 8, n)).round(1)
    return f


UKB_ETH = [1, 2, 3, 4, 5, 6, 1001, 1002, 1003, 2001, 2002, 2003, 2004, 3001, 3002, 3003, 3004,
           4001, 4002, 4003, -1, -3]
FIPS = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
        28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51,
        53, 54, 55, 56]


def coded() -> pd.DataFrame:
    """p2_codes.py ``table`` (rng 42002): UK Biobank's ethnic background (coding 1001, 22 codes),
    education 1–5 written 1.0–5.0 because 5% are blank, BRFSS ``_STATE`` (51 FIPS codes)."""
    rng = np.random.default_rng(42002)
    n = 1500
    f = pd.DataFrame({"eid": np.arange(1000001, 1000001 + n)})
    p = np.r_[np.full(6, 0.01), 0.80, 0.02, 0.03, np.full(4, 0.007), np.full(4, 0.008),
              np.full(3, 0.01), 0.004, 0.004]
    f["ethnic_background"] = rng.choice(UKB_ETH, n, p=p / p.sum())
    edu = rng.integers(1, 6, n).astype(float)
    edu[rng.random(n) < 0.05] = np.nan
    f["education"] = edu
    f["_STATE"] = rng.choice(FIPS, n)
    f["age"] = rng.integers(40, 70, n)
    f["fruit_veg_servings"] = rng.gamma(3, 1.2, n).round(1)
    eth_eff = {1001: 0.0, 4001: 2.0, 4002: 2.5, 3001: 1.0}
    f["sbp"] = (125 + 0.5 * (f.age - 55) - 1.0 * f.fruit_veg_servings
                + f.ethnic_background.map(eth_eff).fillna(0.5) * 3 + rng.normal(0, 12, n)).round(0)
    return f


def smoking() -> pd.DataFrame:
    """p3_code_confirm.py ``table`` (rng 43003): smoking 1 never, 2 former, 3 current."""
    rng = np.random.default_rng(43003)
    n = 900
    f = pd.DataFrame({"participant_id": [f"P{i:04d}" for i in range(n)]})
    f["smoking"] = rng.choice([1, 2, 3], n, p=[0.5, 0.3, 0.2])
    f["age"] = rng.integers(30, 70, n)
    f["fiber_g"] = rng.gamma(4, 5, n).round(1)
    eff = np.select([f.smoking == 1, f.smoking == 2, f.smoking == 3], [0.0, 6.0, 1.0])
    f["crp_mg_l"] = (2 + eff + 0.02 * f.age - 0.03 * f.fiber_g + rng.normal(0, 1.5, n)).round(2)
    return f


def lab_tables() -> dict[str, pd.DataFrame]:
    """p4_units.py ``table`` (one rng, 44004, drawn for ``WBC (x10^3/uL)`` then ``ALT (IU)``)."""
    rng = np.random.default_rng(44004)
    out = {}
    for outcome in ("WBC (x10^3/uL)", "ALT (IU)"):
        n = 600
        f = pd.DataFrame({"participant_id": [f"S{i:04d}" for i in range(n)]})
        f["age"] = rng.integers(25, 75, n)
        f["fiber_g"] = rng.gamma(4, 5, n).round(1)
        f["smoker"] = rng.integers(0, 2, n)
        f[outcome] = (6.5 + 0.8 * f.smoker - 0.02 * f.fiber_g + rng.normal(0, 1.4, n)).round(1)
        out[outcome] = f
    return out


DR1_OCC = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 91]


def dr1iff() -> pd.DataFrame:
    """p7_occasion_codes.py ``table`` (rng 47007): NHANES individual-foods rows per SEQN, with the
    eating occasion's code ``DR1_030Z``."""
    rng = np.random.default_rng(47007)
    rows = []
    for i in range(300):
        seqn = 93703 + i
        k = int(rng.integers(8, 22))
        base = rng.normal(0, 1)
        for line in range(1, k + 1):
            rows.append({"SEQN": seqn, "DR1ILINE": line, "DR1_030Z": int(rng.choice(DR1_OCC)),
                         "DR1IKCAL": round(float(rng.gamma(2, 80)), 0),
                         "DR1ISUGR": round(float(rng.gamma(1.5, 6)), 1),
                         "LBXGH": round(5.5 + 0.3 * base, 1)})
    return pd.DataFrame(rows)


def sleep_diary() -> pd.DataFrame:
    """p9_time_words.py (rng 49009): a sleep diary's ``hours`` and an activity log's ``days``."""
    rng = np.random.default_rng(49009)
    rows = []
    for i in range(120):
        for night in range(1, 8):
            rows.append({"participant_id": f"P{i:03d}", "night": night,
                         "hours": round(float(rng.normal(7, 1.1)), 2),
                         "days": int(rng.integers(0, 8)),
                         "caffeine_mg": round(float(rng.gamma(2, 80)), 0),
                         "next_day_kcal": round(float(rng.normal(2100, 400)), 0)})
    return pd.DataFrame(rows)


def hours_table() -> pd.DataFrame:
    """p10_hours_fit.py ``table`` (rng 41010): next-day energy falls 60 kcal per hour slept."""
    rng = np.random.default_rng(41010)
    rows = []
    for i in range(150):
        u = rng.normal(0, 150)
        for night in range(1, 8):
            h = float(rng.normal(7, 1.1))
            rows.append({"participant_id": f"P{i:03d}", "night": night, "hours": round(h, 2),
                         "caffeine_mg": round(float(rng.gamma(2, 80)), 0),
                         "next_day_kcal": round(2100 + u - 60 * (h - 7) + float(rng.normal(0, 250)),
                                                0)})
    return pd.DataFrame(rows)


def pounds() -> tuple[pd.DataFrame, np.ndarray]:
    """p11_goldberg_lb.py ``table`` (rng 41111): a US cohort's ``weight`` in pounds (the kg drawn,
    times 2.20462, rounded); returns the frame and the weights in kg as drawn."""
    rng = np.random.default_rng(41111)
    n = 500
    sex = rng.choice(["F", "M"], n)
    male = sex == "M"
    age = rng.integers(25, 70, n)
    height = np.where(male, rng.normal(176, 7, n), rng.normal(163, 6.5, n)).round(1)
    wkg = np.where(male, rng.normal(84, 14, n), rng.normal(70, 14, n)).clip(45, 160)
    ffm = wkg * 0.7
    kcal = (500 + 22 * ffm) * rng.uniform(1.4, 1.9, n) * np.exp(rng.normal(-0.1, 0.25, n))
    p = rng.normal(0.16, 0.03, n).clip(0.08, 0.3)
    fa = rng.normal(0.34, 0.06, n).clip(0.15, 0.55)
    c = 1 - p - fa
    frame = pd.DataFrame({"participant_id": [f"U{i:04d}" for i in range(n)], "sex": sex,
                          "age_years": age, "height_cm": height, "weight": (wkg * 2.20462).round(1),
                          "energy_kcal": kcal.round(0), "protein_g": (kcal * p / 4).round(1),
                          "fat_g": (kcal * fa / 9).round(1), "carbohydrate_g": (kcal * c / 4).round(1),
                          "sbp": (120 + 0.3 * (age - 45) + rng.normal(0, 12, n)).round(0)})
    return frame, wkg


def countries() -> pd.DataFrame:
    """p12_many_levels.py (rng 41212): country of birth, 70 labels on 800 rows, half one label."""
    rng = np.random.default_rng(41212)
    n = 800
    names = [f"Country {i:02d}" for i in range(70)]
    p = np.r_[0.5, np.full(69, 0.5 / 69)]
    return pd.DataFrame({"participant_id": [f"P{i:04d}" for i in range(n)],
                         "cob": rng.choice(names, n, p=p),
                         "age_years": rng.integers(20, 70, n),
                         "sodium_mg": rng.normal(3400, 900, n).round(0),
                         "sbp": rng.normal(125, 15, n).round(0)})


# ═════════════════════════════════════════════════════════════════════════════
# Driving the real server
# ═════════════════════════════════════════════════════════════════════════════

ORDER = ["lens", "orientation", "target", "event", "task", "purpose", "grain", "repeat_kind",
         "unit", "aggregation", "temporal", "roles", "survey", "exclusions", "missing", "split",
         "energy_adjustment", "models"]


def answers(target: str, lens: list[str], **over: Any) -> dict[str, dict[str, Any]]:
    out = {"lens": {"kind": "set_lens", "lenses": lens},
           "target": {"kind": "set_target", "column": target},
           "task": {"kind": "set_task", "column": target, "task": "regression"},
           "purpose": {"kind": "set_purpose", "purpose": "inference"},
           "grain": {"kind": "set_grain", "grain": "one_row_per_unit"},
           "temporal": {"kind": "set_temporal", "temporal": False},
           "survey": {"kind": "set_survey", "estimand": "sample"},
           "exclusions": {"kind": "set_exclusions", "rules": []},
           "missing": {"kind": "set_missing", "strategy": "complete_case"},
           "split": {"kind": "set_split", "holdout": 0.2, "seed": 0, "folds": 5},
           "energy_adjustment": {"kind": "set_energy_adjustment", "method": "none"},
           "models": {"kind": "select_models", "models": ["linear"]},
           "unit": {"kind": "set_unit", "unit": "unit"},
           "aggregation": {"kind": "set_aggregation", "method": "mean"}}
    out.update(over)
    return out


def drive_unsettled(drive: Any, plan: dict[str, dict[str, Any]], *, roles: dict[str, str],
                    stop_before: str = "models") -> list[dict[str, Any]]:
    """The opening sequence in the Router's order with nothing confirmed: the roles recorded in
    bulk as proposed (with the author's ``roles`` where they differ: what a bulk answer carries
    below high rides along unconfirmed), every other answer as given. A refusal that only asks for
    readings is answered from the fixture's truth (the Drive's), except at ``stop_before``. Returns
    the roles stage's proposals."""
    proposals: list[dict[str, Any]] = []
    for key in ORDER:
        if key == stop_before:
            return proposals
        step = drive.reach(key, timeout=300)
        if step["status"] not in ("open", "waiting"):
            continue
        if key == "roles":
            proposals = drive.artifact("roles")["columns"]
            given = {c["column"]: c["proposed"] for c in proposals}
            given.update(roles)
            for column, role in given.items():
                drive.truth.setdefault(f"role:{column}", role)
            r = drive.post({"kind": "set_roles", "roles": given})
            assert r.status_code == 200, r.text[:600]
            continue
        drive.decide(plan[key])
    return proposals


def fit_refused_without_a_number(drive: Any, body: dict[str, Any]) -> dict[str, Any]:
    """Section F's criterion: with readings unsettled, the fit asks or refuses and computes no
    number. Returns the refusal."""
    r = drive.post(body)
    assert r.status_code == 409, r.text[:600]
    error = r.json()["error"]
    assert error["code"] in ASKING, error
    stage = drive.view()["stages"]["fit"]
    assert stage["status"] != "fresh", stage
    got = drive.c.get(f"/api/projects/{drive.pid}/stages/fit").json()
    assert not got.get("artifact"), "a fit computed a number on unsettled readings"
    return error


def coefficients(fit: dict[str, Any]) -> dict[str, dict[str, Any]]:
    model = next(m for m in fit["models"] if m.get("coefficients"))
    return {c["feature"]: c for c in model["coefficients"]}


def strings(x: Any) -> list[str]:
    if isinstance(x, dict):
        return [s for v in x.values() for s in strings(v)]
    if isinstance(x, list):
        return [s for v in x for s in strings(v)]
    return [x] if isinstance(x, str) else []


def write(frame: pd.DataFrame, folder: Path, name: str) -> Path:
    path = folder / name
    frame.to_csv(path, index=False)
    return path


def roles_of(frame: pd.DataFrame, *, lens: list[str], target: str, grain: GrainSpec | None = None,
             folder: Path) -> dict[str, dict[str, Any]]:
    from turbotab.core.stages.rows import roles_stage

    path = write(frame, folder, "t.csv")
    state = ProjectState(lens=lens, target=target, purpose="inference",
                         grain=grain or GrainSpec(grain="one_row_per_unit"))
    art = Ingested(path, Path(tempfile.mkdtemp(dir=folder))).run(roles_stage, state)
    data = getattr(art, "data", art)
    return {c["column"]: c for c in data["columns"]}


def ols(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(X, y, rcond=None)[0]


# ═════════════════════════════════════════════════════════════════════════════
# A · the gate's six open items
# ═════════════════════════════════════════════════════════════════════════════


def test_a1_a_household_line_number_never_clusters_the_intervals_through_the_server(tmp_path):
    """Gate item 1 (MEPS ``PID``). Source: MEPS HC-216 (quoted): "A three-digit person number (PID)
    uniquely identifies each person within the DU" — so PID repeats across dwelling units, as a
    subject's identifier in a long table would. Before: ``PID`` was "identifier (high)" and the
    intervals were CR2 by ``PID``, G = 19, over the grain answer naming ``DUPERSID``.

    Expected: the reading is asked (below high, attention), and the fit computes nothing until it is
    answered; with the fixture's truth (PID a line number, left out; DUID the household, a cluster
    of units) the intervals are CR2 by ``DUID`` with G = pandas' count of dwelling units (1,400);
    and with ``PID`` confirmed as an identifier against the grain answer naming ``DUPERSID``, the
    grain answer wins: nothing clusters by ``PID``."""
    frame = meps()
    assert frame["PID"].nunique() == 19 and frame["DUID"].nunique() == 1400
    path = write(frame, tmp_path, "meps.csv")
    plan = answers("bmi", ["clinical"], grain={"kind": "set_grain", "grain": "one_row_per_unit",
                                                "id_column": "DUPERSID"})
    truth = Truth({"code_or_count:AGE19X": "amount"}, fixture="MEPS (p1_ids.py)")
    with local_server(tmp_path / "home") as client:
        drive = open_project(client, path, truth)
        proposals = drive_unsettled(drive, plan, roles={"PID": "excluded", "DUID": "cluster",
                                                        "DUPERSID": "identifier"})
        pid = next(p for p in proposals if p["column"] == "PID")
        assert pid["confidence"] != "high" and pid["attention"], pid
        error = fit_refused_without_a_number(drive, plan["models"])
        # The author's own roles for PID and DUID are their answers; what rode along is asked.
        listed = asked(error["exits"])
        assert ("code_or_count", "AGE19X") in listed and ("role", "fruit_servings") in listed
        assert ("role", "PID") not in listed and ("role", "DUID") not in listed
        drive.decide(plan["models"])
        fit = drive.artifact("fit")
        said = " ".join(strings(fit))
        assert f"cluster-robust (CR2) by `DUID`, G = {frame['DUID'].nunique():,} clusters" in said
        assert "by `PID`" not in said

        wrong = open_project(client, write(frame, tmp_path, "meps_pid.csv"),
                             Truth({"code_or_count:AGE19X": "amount"}, fixture="MEPS, PID kept"))
        drive_unsettled(wrong, plan, roles={"PID": "identifier", "DUID": "excluded",
                                            "DUPERSID": "identifier"})
        wrong.decide(plan["models"])
        said = " ".join(strings(wrong.artifact("fit")))
        assert "by `PID`" not in said and "G = 19" not in said
        assert "The grain answer names `DUPERSID` as the unit" in said


def test_a1_a_rosters_line_number_never_wins_over_the_household_the_user_confirmed(tmp_path):
    """Gate item 1 (a household roster's ``person_no``, 1..k in each household). Before:
    ``person_no`` was "identifier (high)" and the intervals clustered by it (G = 18) over ``hhid``,
    which the user had confirmed as a cluster. Expected: ``person_no`` is asked; with the truth
    (``person_no`` a line number, ``hhid`` the household) the intervals are CR2 by ``hhid`` with
    G = pandas' 900 households; and with ``person_no`` confirmed as an identifier beside the
    confirmed household, the household (a group of units the user named) still wins."""
    frame = roster()
    assert frame["person_no"].nunique() == 18
    plan = answers("waist_cm", ["survey"])
    truth = {"code_or_count:age_years": "amount", "code_or_count:sugary_drinks_wk": "amount"}
    with local_server(tmp_path / "home") as client:
        drive = open_project(client, write(frame, tmp_path, "roster.csv"),
                             Truth(truth, fixture="roster (p5_roster.py)"))
        proposals = drive_unsettled(drive, plan, roles={"person_no": "excluded", "hhid": "cluster"})
        person = next(p for p in proposals if p["column"] == "person_no")
        assert person["confidence"] != "high" and person["attention"], person
        fit_refused_without_a_number(drive, plan["models"])
        drive.decide(plan["models"])
        said = " ".join(strings(drive.artifact("fit")))
        assert f"by `hhid`, G = {frame['hhid'].nunique():,} clusters" in said

        both = open_project(client, write(frame, tmp_path, "roster2.csv"),
                            Truth(truth, fixture="roster, person_no kept"))
        drive_unsettled(both, plan, roles={"person_no": "identifier", "hhid": "cluster"})
        both.decide(plan["models"])
        said = " ".join(strings(both.artifact("fit")))
        assert f"by `hhid`, G = {frame['hhid'].nunique():,} clusters" in said
        assert "by `person_no`" not in said


LAB_HEADERS = {"WBC (x10^3/uL)": "1000 cells/uL", "WBC (/uL)": "cells/uL",
               "Platelets (/uL)": "1000 cells/uL", "ALT (IU)": "U/L", "ALT_IU": "U/L"}


def test_a2_a_lab_headers_unit_is_quoted_never_parsed_until_recorded(tmp_path):
    """Gate item 2. Sources (quoted): NHANES CBC_J "LBXWBCSI: White blood cell count (1000
    cells/uL)", BIOPRO_J "LBXSATSI: Alanine Aminotransferase (ALT) (U/L)". Before: ``WBC
    (x10^3/uL)`` was stated "in U/L" and ``ALT (IU)`` "in IU" in the record. Expected (stage level,
    every header): the target stage states no unit and the record's sentence quotes the header
    with none; once ``set_outcome_unit`` records the codebook's unit, it is stated as recorded."""
    from turbotab.core import voice
    from turbotab.core.stages.target import target_info_stage

    assert NHANES_CBC_J["LBXWBCSI"].endswith("(1000 cells/uL)")
    assert NHANES_BIOPRO_J["LBXSATSI"].endswith("(U/L)")
    rng = np.random.default_rng(0)
    for i, (header, unit) in enumerate(LAB_HEADERS.items()):
        one = pd.DataFrame({header: rng.normal(6.5, 1.4, 300).round(1)})
        path = write(one, tmp_path, f"lab{i}.csv")
        state = ProjectState(target=header, task="regression")
        info = Ingested(path, Path(tempfile.mkdtemp(dir=tmp_path))).run(target_info_stage, state)
        assert (info["unit"], info["unit_source"]) == (None, None), (header, info["unit"])
        assert info["proposed_unit"] != "U/L" or header in ("ALT (IU)", "ALT_IU"), header
        sentence = voice.sentence_for(d.SetTarget(column=header), None, {"frame": one})
        assert sentence == f"`{header}` was chosen as the outcome.", sentence
        recorded = state.model_copy(update={"outcome_unit": unit})
        info = Ingested(path, Path(tempfile.mkdtemp(dir=tmp_path))).run(target_info_stage, recorded)
        assert (info["unit"], info["unit_source"]) == (unit, "decision")
        assert voice.sentence_for(d.SetTarget(column=header), recorded, {"frame": one}) == \
            f"`{header}` was chosen as the outcome, in {unit}."


def test_a2_the_gates_two_headers_through_the_server_state_no_parsed_unit(tmp_path):
    """Gate item 2 through the server, as the gate drove it (p4_units.py, rng 44004): the record's
    sentence and every string of the fit carry neither "U/L" nor " IU" for the two headers."""
    tables = lab_tables()
    with local_server(tmp_path / "home") as client:
        for i, (header, frame) in enumerate(tables.items()):
            drive = open_project(client, write(frame, tmp_path, f"units{i}.csv"),
                                 Truth({"code_or_count:age": "amount"}, fixture=header))
            plan = answers(header, ["clinical"])
            drive_unsettled(drive, plan, roles={"participant_id": "identifier"})
            drive.decide(plan["models"])
            info = drive.artifact("target_info")
            assert info["unit"] is None and info["unit_source"] is None, info["unit"]
            said = [x["sentence"] for x in drive.view()["decisions"]
                    if x["decision"]["kind"] == "set_target"]
            assert said == [f"`{header}` was chosen as the outcome; it is measured on all `600` "
                            f"rows."], said
            text = " ".join(strings(drive.artifact("fit")))
            assert "U/L" not in text and " IU" not in text.replace(f"`{header}`", "")


def test_a3_the_users_codes_reach_the_fit_as_one_indicator_per_level(tmp_path):
    """Gate item 3 (smoking 1/2/3, p3_code_confirm.py rng 43003). Before: the user picked the fit's
    own exit "holds codes for categories (one indicator per level)" and the fit still entered one
    slope, 1.074. Expected: the fit asks (smoking guessed codes, age an amount), computes nothing
    until answered, and then follows the answer: codes give the NumPy indicator fit (5.761 and
    0.812), an amount gives the NumPy single slope (1.074)."""
    frame = smoking()
    y = frame["crp_mg_l"].to_numpy(float)
    one = np.ones(len(frame))
    indicators = ols(y, np.column_stack([one, frame.smoking == 2, frame.smoking == 3, frame.age,
                                         frame.fiber_g]).astype(float))
    slope = ols(y, np.column_stack([one, frame.smoking, frame.age, frame.fiber_g]).astype(float))
    assert round(indicators[1], 3) == 5.761 and round(indicators[2], 3) == 0.812
    assert round(slope[1], 3) == 1.074
    plan = answers("crp_mg_l", ["clinical"])
    with local_server(tmp_path / "home") as client:
        for truth, want in ((("code"), {"smoking_2": indicators[1], "smoking_3": indicators[2]}),
                            (("amount"), {"smoking": slope[1]})):
            drive = open_project(client, write(frame, tmp_path, f"smoking_{truth}.csv"), Truth(
                {"code_or_count:smoking": truth, "code_or_count:age": "amount"},
                fixture=f"p3, smoking as {truth}"))
            drive_unsettled(drive, plan, roles={"participant_id": "identifier"})
            error = fit_refused_without_a_number(drive, plan["models"])
            block = error["exits"][0]["decision"]
            assert block["kind"] == "confirm_readings"
            shown = {(i["column"], i["value"]) for i in block["items"]}
            assert {("smoking", "code"), ("age", "amount")} <= shown, shown
            drive.decide(plan["models"])
            coef = coefficients(drive.artifact("fit"))
            assert {k for k in coef if k.startswith("smoking")} == set(want)
            for name, value in want.items():
                assert coef[name]["estimate"] == pytest.approx(value, abs=1e-6), name


def test_a4_codes_after_a_blank_and_beyond_ten_levels_are_asked_and_fit_as_indicators(tmp_path):
    """Gate item 4, the fit (p2_codes.py, rng 42002). Sources (quoted): UK Biobank coding 1001 is
    "a hierarchical tree-structured dictionary which uses integers to represent categories or
    special values"; BRFSS 2022 ``_STATE`` "State FIPS Code" (1 Alabama, 2 Alaska, 4 Arizona …);
    pandas: with a value missing "the original data type will be coerced to np.float64". Before:
    each entered as one slope, unasked. Expected: the fit asks about all four whole-valued
    predictors (any type, any count of values) and computes nothing until answered; with the truth
    (three code lists, age an amount) each code list enters as one indicator per level beyond the
    first, and the education indicators equal a NumPy least-squares fit with every indicator."""
    frame = coded()
    assert frame["education"].dtype == np.float64 and frame["_STATE"].nunique() == 51
    assert frame["ethnic_background"].nunique() == 22
    plan = answers("sbp", ["clinical"], grain={"kind": "set_grain", "grain": "one_row_per_unit",
                                               "id_column": "eid"})
    truth = Truth({"code_or_count:ethnic_background": "code", "code_or_count:education": "code",
                   "code_or_count:_STATE": "code", "code_or_count:age": "amount"}, fixture="p2")
    with local_server(tmp_path / "home") as client:
        drive = open_project(client, write(frame, tmp_path, "codes.csv"), truth)
        drive_unsettled(drive, plan, roles={"eid": "identifier"})
        error = fit_refused_without_a_number(drive, plan["models"])
        assert {c for k, c in asked(error["exits"]) if k == "code_or_count"} == \
            {"ethnic_background", "education", "_STATE", "age"}
        drive.decide(plan["models"])
        coef = coefficients(drive.artifact("fit"))
    rows = frame.dropna(subset=["education"])
    for column in ("ethnic_background", "education", "_STATE"):
        made = [k for k in coef if k.startswith(f"{column}_")]
        assert len(made) == rows[column].nunique() - 1, (column, len(made))
        assert column not in coef  # never one slope
    dummies = [pd.get_dummies(rows[c].astype(float), prefix=c, prefix_sep="_", drop_first=True,
                              dtype=float)
               for c in ("ethnic_background", "education", "_STATE")]
    X = pd.concat([pd.Series(1.0, index=rows.index, name="(intercept)"), rows[["age"]].astype(float),
                   rows[["fruit_veg_servings"]], *dummies], axis=1)
    b = pd.Series(ols(rows["sbp"].to_numpy(float), X.to_numpy(float)), index=X.columns)
    for level in (2.0, 3.0, 4.0, 5.0):
        assert coef[f"education_{level}"]["estimate"] == pytest.approx(b[f"education_{level}"],
                                                                        abs=1e-6)


def test_a4_eating_occasion_codes_are_combined_by_their_mode_not_averaged(tmp_path):
    """Gate item 4, combining (p7_occasion_codes.py, rng 47007). Source (quoted): NHANES DR1IFF_J
    ``DR1_030Z`` "Name of eating occasion", codes 1 Breakfast … 19 Bebida, 91 Other, 99 Don't know.
    Before: combined per SEQN by its mean (22.11, 12.95), unasked. Expected: the combining answer
    asks about every whole-valued column that changes within a person (``DR1_030Z`` among them),
    then follows the truth: each person's ``DR1_030Z`` is their most frequent code (pandas, ties to
    the earliest record), never a mean."""
    assert NHANES_DR1IFF_J["codes"][91] == "Other"
    frame = dr1iff()
    means = frame.groupby("SEQN")["DR1_030Z"].mean()
    assert round(means.iloc[0], 2) == 22.11 and round(means.iloc[1], 2) == 12.95
    plan = answers("LBXGH", ["dietary"],
                   grain={"kind": "set_grain", "grain": "repeated", "id_column": "SEQN"},
                   repeat_kind={"kind": "set_repeat_kind", "repeat_kind": "repeats"})
    truth = Truth({"code_or_count:DR1_030Z": "code", "code_or_count:DR1IKCAL": "amount",
                   "code_or_count:DR1ILINE": "code"}, fixture="p7 (DR1IFF)")
    with local_server(tmp_path / "home") as client:
        drive = open_project(client, write(frame, tmp_path, "dr1iff.csv"), truth)
        drive_unsettled(drive, plan, roles={}, stop_before="aggregation")
        drive.reach("aggregation", timeout=300)
        r = drive.post(plan["aggregation"])
        assert r.status_code == 409 and r.json()["error"]["code"] == "reading_unsettled", r.text
        assert ("code_or_count", "DR1_030Z") in asked(r.json()["error"]["exits"])
        drive.decide(plan["aggregation"])
        receipt = drive.artifact("working")["aggregation"]
        window = client.get(f"/api/projects/{drive.pid}/table",
                            params={"limit": 1000, "columns": "SEQN,DR1_030Z"}).json()
    entry = next(c for c in receipt["columns"] if c["column"] == "DR1_030Z")
    assert (entry["kind"], entry["rule"]) == ("code", "mode"), entry
    got = pd.DataFrame(window["rows"], columns=window["columns"]).set_index("SEQN")["DR1_030Z"]
    # Against pandas: each person's most frequent code, ties to the earliest record.
    want = frame.groupby("SEQN", sort=False)["DR1_030Z"].agg(
        lambda s: next(v for v in s if v in set(s.value_counts()[lambda c: c == c.max()].index)))
    assert len(got) == frame["SEQN"].nunique() == 300
    assert got.loc[want.index].astype(int).tolist() == want.astype(int).tolist()
    assert not np.isclose(got.loc[means.index].to_numpy(float), means.to_numpy(float)).all()


def test_a5_a_bare_unit_word_is_no_time_and_stays_in_the_model(tmp_path):
    """Gate item 5 (a sleep diary's ``hours``, an activity log's ``days``). Before: each was
    "time (high)" because it changed within units, and ``hours`` left every fit (NumPy slope
    −52 kcal per hour). Expected (stage, p9 rng 49009): neither is proposed as time at high, and
    both carry attention; (server, p10 rng 41010): the fit asks and computes nothing until the
    roles are answered; with the truth (``hours`` a measurement, ``night`` the time column named by
    the repeats answer) ``hours`` is in the model at NumPy's slope."""
    found = roles_of(sleep_diary(), lens=["clinical"], target="next_day_kcal",
                     grain=GrainSpec(grain="repeated", id_column="participant_id"),
                     folder=tmp_path)
    for column in ("hours", "days"):
        assert not (found[column]["proposed"] == "time" and found[column]["confidence"] == "high")
        assert found[column]["attention"], found[column]
    frame = hours_table()
    b = ols(frame["next_day_kcal"].to_numpy(float),
            np.column_stack([np.ones(len(frame)), frame.hours, frame.caffeine_mg]).astype(float))
    plan = answers("next_day_kcal", ["clinical"],
                   grain={"kind": "set_grain", "grain": "repeated", "id_column": "participant_id"},
                   repeat_kind={"kind": "set_repeat_kind", "repeat_kind": "time_points",
                                "time_column": "night"},
                   unit={"kind": "set_unit", "unit": "row"})
    truth = Truth({"code_or_count:caffeine_mg": "amount"}, fixture="p10")
    with local_server(tmp_path / "home") as client:
        drive = open_project(client, write(frame, tmp_path, "hours.csv"), truth)
        drive_unsettled(drive, plan, roles={"hours": "exposure", "night": "time"})
        fit_refused_without_a_number(drive, plan["models"])
        drive.decide(plan["models"])
        coef = coefficients(drive.artifact("fit"))
    assert set(coef) == {"(intercept)", "hours", "caffeine_mg"}
    assert coef["hours"]["estimate"] == pytest.approx(b[1], abs=1e-6)
    assert round(b[1], 1) == -52.3


def test_a6_a_category_with_many_labels_is_asked_never_dropped_as_free_text(tmp_path):
    """Gate item 6 (country of birth, 70 labels on 800 rows, half one label; p12 rng 41212).
    Before: "excluded (high): free text", leaving the predictors unasked. Expected: the proposal
    is below high with attention and leads with the values' best guess (a category to keep); each
    answer is honored: confirmed as left out it is in no fit, confirmed as a covariate it enters
    as one indicator per label beyond the first."""
    from turbotab.core.models.pipeline import design_spec, model_predictors

    frame = countries()
    assert frame["cob"].nunique() == 70 and frame["cob"].value_counts().iloc[0] / len(frame) > 0.45
    found = roles_of(frame, lens=["clinical"], target="sbp", folder=tmp_path)["cob"]
    assert found["confidence"] != "high" and found["attention"]
    assert found["proposed"] != "excluded", found
    base = {"participant_id": "identifier", "age_years": "covariate", "sodium_mg": "exposure"}
    for role, inside in (("excluded", False), ("covariate", True)):
        state = ProjectState(target="sbp", roles={**base, "cob": role}, roles_unconfirmed=["cob"],
                             role_confirmations={"cob": role})
        assert ("cob" in model_predictors(state)) is inside
        if inside:
            spec = design_spec(state, frame.drop(columns=["sbp"]), model_predictors(state))
            assert "cob" in spec.categorical


# ═════════════════════════════════════════════════════════════════════════════
# B · the gate's nine unasked assumptions
# ═════════════════════════════════════════════════════════════════════════════


def test_b1_the_fit_and_the_imputation_model_read_codes_wherever_the_answer_is_kept():
    """Assumption 1: ``design_spec`` and the multiple-imputation frame read ``state.categorical``
    and the dtype only, so a ``confirm_reading`` code answer (kept in ``shape_confirmations``) was
    never read. Expected: a code answer kept anywhere (its own confirmation, the block
    confirmation, ``set_categorical``) makes the column categorical in the spec and in the
    chained equations' kinds; an amount answer kept the same ways makes it numeric."""
    from turbotab.core.methods.missing import imputation_frame
    from turbotab.core.models.pipeline import design_spec

    rng = np.random.default_rng(3)
    X = pd.DataFrame({"smoking": rng.choice([1.0, 2.0, 3.0], 200), "age": rng.normal(50, 9, 200)})
    X.loc[:9, "smoking"] = np.nan
    roles = {"smoking": "covariate", "age": "covariate"}
    for kept, want in ((dict(shape_confirmations={"code_or_count:smoking": "code"}), "categorical"),
                       (dict(categorical=["smoking"]), "categorical"),
                       (dict(shape_confirmations={"code_or_count:smoking": "amount"}), "numeric")):
        state = ProjectState(target="y", roles=roles, missing="multiple_imputation", **kept)
        spec = design_spec(state, X, ["smoking", "age"])
        assert ("smoking" in spec.categorical) is (want == "categorical"), kept
        _, _, kinds, _, _ = imputation_frame(spec, X, pd.Series(rng.normal(size=200)), "regression")
        assert kinds["smoking"] == want, (kept, kinds)
    folded = d.fold([d.DecisionRecord(id="a", seq=1, at="2026-10-03T00:00:00Z", decision=d.ConfirmReadings(
        items=[d.ReadingItem(reading="code_or_count", column="smoking", value="code")]))])
    assert "smoking" in design_spec(folded.model_copy(update={"roles": roles, "target": "y"}), X,
                                    ["smoking", "age"]).categorical


def test_b2_whole_numbers_settle_nothing_whatever_their_type_or_count():
    """Assumption 2: codes written as floats after a blank and code lists beyond ten values were
    no question. Expected: the fit's scope is every whole-valued numeric predictor with two or more
    values (exactly 0/1 excepted: one indicator either way), any dtype, any count."""
    facts = {"education": {"whole": True, "zero_one": False, "n_values": 5, "min": 1.0, "max": 5.0},
             "_STATE": {"whole": True, "zero_one": False, "n_values": 51, "min": 1.0, "max": 56.0},
             "eth": {"whole": True, "zero_one": False, "n_values": 22, "min": -3.0, "max": 4003.0},
             "smoker": {"whole": True, "zero_one": True, "n_values": 2, "min": 0.0, "max": 1.0},
             "bmi": {"whole": False, "zero_one": False, "n_values": 300, "min": 17.2, "max": 41.0}}
    state = ProjectState(target="y")
    asked_ = {r.column: r.value for r in R.unsettled_codes(state, list(facts), facts)}
    assert set(asked_) == {"education", "_STATE", "eth"}
    assert asked_["eth"] == "code" and asked_["education"] == "code"  # the guesses lead
    combine = {r.column for r in R.unsettled_codes(state, list(facts), facts, scope="combine")}
    assert combine == {"education", "_STATE", "eth", "smoker"}


def test_b3_combining_asks_for_every_whole_valued_column_that_changes_within_units(tmp_path):
    """Assumption 3: ``column_kinds`` combined whole numbers with more than 10 values by their mean
    unasked. Expected: the structure reading lists every whole-valued column that changes within a
    unit (DR1_030Z's 19 codes, DR1IKCAL's whole kcal, the line number), never a fractional one."""
    from turbotab.core.stages.working import code_or_count_facts

    frame = dr1iff()
    facts = code_or_count_facts(frame, "SEQN", exclude=["LBXGH"])
    assert set(facts) == {"DR1ILINE", "DR1_030Z", "DR1IKCAL"}
    assert facts["DR1_030Z"]["n_values"] == frame["DR1_030Z"].nunique() == 19


def test_b4_a_repeating_identifier_is_never_settled_by_its_values():
    """Assumption 4: a subject-named identifier with more than 10 repeating values was "identifier
    (high)". Expected: :func:`readings.names_rows` settles only one value per row that no
    measurement would take; any repeat (a line number, a long table's subject) leaves it to the
    user, and the cluster reading has no value test at all."""
    assert not R.names_rows(meps()["PID"]).settles
    assert not R.names_rows(roster()["person_no"]).settles
    assert R.KIND_RULES["cluster"].test is None


def test_b5_varying_within_units_settles_no_time_role():
    """Assumption 5: a bare unit word that varies within units was "time (high)"."""
    frame = sleep_diary()
    for column in ("hours", "days", "night"):
        assert not R.dates_order_rows(frame[column], frame["participant_id"]).settles


def test_b6_no_header_settles_an_outcome_unit():
    """Assumption 6: a ``ul`` suffix read as U/L and a bare ``iu`` as IU, both settled."""
    for header in LAB_HEADERS:
        r = R.outcome_unit_reading(header)
        assert r is None or not r.settled, header
        assert R.stated_outcome_unit(header) == (None, None)
    assert R.stated_outcome_unit("ALT (IU)", "U/L") == ("U/L", "decision")


def test_b7_a_label_count_settles_no_free_text():
    """Assumption 7: "excluded (high): free text" by a count of labels."""
    assert R.KIND_RULES["role:free_text"].test is None


def _teens_with_numeric_sex() -> pd.DataFrame:
    """Teenagers 14–19 with heights from their own sex's CDC median, plus 20 tall girls (195 cm:
    implausible on the girls' chart only) and 30 tall boys (199 cm: implausible on the girls'
    chart only); sex coded 1/2 as NHANES codes it (1 male, 2 female)."""
    from turbotab.core.detectors import plausibility

    lms = plausibility._lms("height")
    rng = np.random.default_rng(8)
    n = 300
    sex = rng.choice([1.0, 2.0], n)
    months = rng.uniform(170, 235, n).round(0)
    median = np.array([np.interp(m, lms[lms["Sex"] == s]["Agemos"], lms[lms["Sex"] == s]["M"])
                       for m, s in zip(months, sex)])
    height = median * rng.normal(1.0, 0.02, n)
    girls, boys = np.flatnonzero(sex == 2.0)[:20], np.flatnonzero(sex == 1.0)[:30]
    height[girls], height[boys] = 195.0, 199.0
    return pd.DataFrame({"sex": sex, "age": months, "height": height.round(1)})


def _cdc_flags(frame: pd.DataFrame, female: float | None) -> int:
    """Independent of the detector: the CDC's modified z-score for height-for-age (z = (X − M) /
    ((X₊₂ − M) / 2) above the median, X₊₂ = M(1 + 2LS)^(1/L), and the mirror below) from the CDC's
    published LMS table (statage.csv), flagged outside the CDC's BIV cut-offs for height (below −5
    or above 4). ``female``: the code read as female (the CDC table's 2); None: no row's sex is
    known, so a value is flagged only where both sexes' charts flag it."""
    from turbotab.core.detectors import plausibility

    lms = plausibility._lms("height")
    lo, hi = plausibility.BIV_CUTOFFS["height"]

    def z(row: Any, cdc_sex: float) -> float:
        part = lms[lms["Sex"] == cdc_sex].sort_values("Agemos")
        L, M, S = (np.interp(row.age, part["Agemos"], part[k]) for k in ("L", "M", "S"))
        plus2 = M * (1 + 2 * L * S) ** (1 / L)
        minus2 = M * (1 - 2 * L * S) ** (1 / L)
        return (row.height - M) / ((plus2 - M) / 2) if row.height >= M else \
            (row.height - M) / ((M - minus2) / 2)

    def biv(value: float) -> bool:
        return value < lo or value > hi

    count = 0
    for row in frame.itertuples():
        if female is None:
            count += bool(biv(z(row, 1.0)) and biv(z(row, 2.0)))
        else:
            count += bool(biv(z(row, 2.0 if row.sex == female else 1.0)))
    return count


def test_b8_a_numeric_sex_coding_is_read_only_as_confirmed():
    """Assumption 8: any numeric ``sex`` with values {1, 2} was read as the CDC's 1 = male,
    2 = female, unasked, behind the children's BIV flags. Expected: unconfirmed, no row's sex is
    known (a value is flagged only where both charts flag it); confirmed either way, each row is
    judged on its own sex's chart; each count equals an independent recomputation, and the three
    differ (the 20 tall girls and 30 tall boys)."""
    from turbotab.core.detectors import plausibility

    frame = _teens_with_numeric_sex()
    units = {"age": ColumnUnitSpec(unit="months")}

    def flagged(codings: Any) -> int:
        out = plausibility.read(frame, units, codings)
        entry = next(e for e in out["columns"] if e["variable"] == "height")
        return int((entry.get("children") or {}).get("n_flagged") or 0)

    unknown = flagged(None)
    cdc = flagged({"sex": "female=2,male=1"})
    reversed_ = flagged({"sex": "female=1,male=2"})
    assert unknown == _cdc_flags(frame, None)
    assert cdc == _cdc_flags(frame, 2.0)
    assert reversed_ == _cdc_flags(frame, 1.0)
    assert unknown < min(cdc, reversed_) and cdc != reversed_
    assert not R.labels_spell_sex(frame["sex"]).settles


def test_b9_a_weight_recorded_in_pounds_is_converted_exactly_in_the_goldberg_offer(tmp_path):
    """Assumption 9 (p11_goldberg_lb.py, rng 41111): after the user recorded ``weight`` in lb, the
    Goldberg offer showed "affected 159", computed reading lb as kg. Expected: the rule records the
    unit and the screen converts 1 lb = 0.45359237 kg exactly, so the offer's count is the screen's
    own: EFSA's printed Schofield equations (weight in kg, height in m) and Black 2000's cut-offs at
    PAL 1.55 over one day, applied to the converted weights, within the rows whose ratio sits
    within 1.5% of a cut-off (where EFSA's rounded kcal equations and the MJ ones may differ)."""
    from turbotab.core.tests.acceptance.test_wp12c_goldberg import efsa_bmr_kcal

    frame, _ = pounds()
    kg = frame["weight"].to_numpy(float) * LB_TO_KG
    sex = np.where(frame["sex"] == "F", "female", "male")
    bmr = efsa_bmr_kcal(sex, frame["age_years"].to_numpy(float), kg,
                        frame["height_cm"].to_numpy(float))
    s = math.sqrt(23.0 ** 2 / 1 + 8.5 ** 2 + 15.0 ** 2)  # Black 2000: CV_wEI/√d, CV_wB, CV_tP
    lower, upper = 1.55 * math.exp(-2 * s / 100), 1.55 * math.exp(2 * s / 100)
    ratio = frame["energy_kcal"].to_numpy(float) / bmr
    reference = int(((ratio < lower) | (ratio > upper)).sum())
    near = int(((np.abs(ratio / lower - 1) < 0.015) | (np.abs(ratio / upper - 1) < 0.015)).sum())
    as_kg = efsa_bmr_kcal(sex, frame["age_years"].to_numpy(float), frame["weight"].to_numpy(float),
                          frame["height_cm"].to_numpy(float))
    wrong = int((((frame["energy_kcal"] / as_kg) < lower) | ((frame["energy_kcal"] / as_kg) > upper)).sum())
    assert abs(wrong - 159) <= 3  # the gate's wrong count, reproduced

    plan = answers("sbp", ["dietary"])
    truth = Truth({"unit:weight": "lb", "unit:age_years": "years", "unit:energy_kcal": "kcal",
                   "day_count:energy_kcal": "1", "code_or_count:age_years": "amount"},
                  fixture="p11")
    with local_server(tmp_path / "home") as client:
        drive = open_project(client, write(frame, tmp_path, "pounds.csv"), truth)
        drive_unsettled(drive, plan, roles={}, stop_before="exclusions")
        r = drive.post({"kind": "confirm_reading", "reading": "unit", "column": "weight",
                        "value": "lb"})
        assert r.status_code == 200, r.text
        drive.reach("exclusions", timeout=300)
        offer = next(e for e in drive.artifact("proposals")["exclusions"]
                     if e["key"] == "goldberg_schofield")
        assert offer["rule"]["weight_unit"] == "lb" and "converted from lb" in offer["label"]
        assert abs(offer["affected"] - reference) <= near, (offer["affected"], reference, near)
        assert abs(offer["affected"] - wrong) > 50
        drive.decide({"kind": "set_exclusions", "rules": [offer["rule"]]})
        rules = drive.view()["state"]["exclusions"]
        assert rules[0]["weight_unit"] == "lb"


# ═════════════════════════════════════════════════════════════════════════════
# C · the kind registry: each value test rejects every alternative it declares
# ═════════════════════════════════════════════════════════════════════════════


def _energy_frame(unit: str, grams: bool = True) -> pd.DataFrame:
    """A day's macronutrients in grams (mg when ``grams`` is False) and total energy in ``unit``."""
    rng = np.random.default_rng(13)
    n = 300
    P, C, F = rng.normal(80, 20, n).clip(20), rng.normal(250, 60, n).clip(50), rng.normal(75, 20, n).clip(15)
    kcal = 4 * P + 4 * C + 9 * F
    scale = 1.0 if grams else 1000.0
    return pd.DataFrame({"energy": (kcal * (4.184 if unit == "kj" else 1.0)).round(0),
                         "protein_g": (P * scale).round(1), "carbohydrate_g": (C * scale).round(1),
                         "fat_g": (F * scale).round(1)})


def _requirement_frame() -> pd.DataFrame:
    """Macronutrients eaten beside an estimated energy requirement (Mifflin's resting rate × 1.6
    from body size), which tracks what people eat only loosely."""
    rng = np.random.default_rng(15)
    frame = _energy_frame("kcal")
    weight = 55 + (frame["energy"] - frame["energy"].mean()) / 120 + rng.normal(0, 12, len(frame))
    frame["energy"] = (1.6 * (10 * weight + 6.25 * 170 - 5 * 45 + 5)).round(0)
    return frame


# kind -> {alternative: (the fixture's verdict, the value the alternative must not be settled as;
# None: it must not be settled at all)}. One fixture per declared alternative (BLUEPRINT §14.3).
ALTERNATIVE_FIXTURES: dict[str, dict[str, tuple[Any, Any]]] = {
    "role:identifier": {
        "a code that repeats: a household's line number, a stratum, an interviewer":
            (lambda: R.names_rows(roster()["person_no"]), "identifier"),
        "a measurement whose values happen to all differ":
            (lambda: R.names_rows(pd.Series(np.random.default_rng(1).choice(
                np.arange(2000, 20000), 50, replace=False))), "identifier"),
    },
    "role:time": {
        "a measurement in a time unit that changes within units (hours slept, days active)":
            (lambda: R.dates_order_rows(sleep_diary()["hours"], sleep_diary()["participant_id"]),
             "time"),
        "a crossover's treatment, which changes within units":
            (lambda: R.dates_order_rows(pd.Series(np.tile([0, 1, 1, 0], 50)),
                                        pd.Series(np.repeat(np.arange(100), 2))), "time"),
        "a date constant within units (a birth or randomization date)":
            (lambda: R.dates_order_rows(pd.Series(pd.to_datetime(np.repeat(
                pd.date_range("1960-01-01", periods=50, freq="7D"), 4))),
                pd.Series(np.repeat(np.arange(50), 4))), "time"),
    },
    "role:flag": {
        "a yes/no characteristic":
            (lambda: R.flag_marks_blanks(pd.Series(np.tile([0, 1], 100)), pd.Series(
                np.where(np.arange(200) % 7 == 0, np.nan, 1.0))), "flag"),
        "an imputed copy of a measurement":
            (lambda: R.flag_marks_blanks(pd.Series(np.random.default_rng(2).normal(120, 15, 200)),
                                         pd.Series(np.where(np.arange(200) % 7 == 0, np.nan, 1.0))),
             "flag"),
    },
    "role:exposure": {
        "a lab count named like a nutrient (ALC: lymphocytes)":
            (lambda: R.intake_rises_with_energy("ALC", pd.Series(
                np.random.default_rng(3).normal(2.1, 0.6, 300)),
                pd.Series(np.random.default_rng(4).normal(2100, 400, 300))), "exposure"),
        "a body measure named like a nutrient (BIA Fat%)":
            (lambda: R.intake_rises_with_energy("fat_g", pd.Series(
                np.random.default_rng(5).normal(25000, 4000, 300)),
                pd.Series(np.random.default_rng(6).normal(2100, 400, 300))), "exposure"),
        "a yes/no named like a nutrient":
            (lambda: R.intake_rises_with_energy("fat_g", pd.Series(np.tile([0, 1], 150)), pd.Series(
                np.random.default_rng(7).normal(2100, 400, 300))), "exposure"),
    },
    "role:energy": {
        "a device's energy expenditure (Fitbit Calories)":
            (lambda: R.energy_follows_macronutrients(
                _energy_frame("kcal").assign(energy=np.random.default_rng(14).normal(2400, 300, 300)),
                "energy"), "energy"),
        "an energy requirement computed from body size":
            (lambda: R.energy_follows_macronutrients(_requirement_frame(), "energy"), "energy"),
    },
    "role:covariate": {
        "a column left out of the model (an identifier, a constant)":
            (lambda: R.characteristic_predictor("sex", pd.Series(np.arange(300))), "covariate"),
        "the time axis of repeated rows (an age at each visit)":
            (lambda: R.characteristic_predictor(
                "age", pd.Series(np.tile([3.0, 4.0, 5.0, 6.0], 60) + np.repeat(np.arange(60) % 5, 4)),
                pd.Series(np.repeat(np.arange(60), 4))), "covariate"),
    },
    "role:excluded": {
        "a predictor": (lambda: R.constant_values(pd.Series(
            np.random.default_rng(8).normal(size=50))), "excluded"),
    },
    "code_or_count": {
        # codes 1–5 written 1.0–5.0 after a blank: never settled as amounts
        "codes for categories (one indicator per level; a unit's rows take the most frequent)":
            (lambda: R.amounts_by_values(coded()["education"]), "amount"),
        # 51 FIPS codes, whole: never settled as codes either
        "amounts (one slope; a unit's rows averaged)":
            (lambda: R.amounts_by_values(pd.Series(FIPS * 3)), "code"),
    },
    "unit:height": {
        "inches": (lambda: R.height_in_band(pd.Series(np.random.default_rng(9).normal(66, 3, 200))),
                   None),
        "metres, against centimetres":
            (lambda: R.height_in_band(pd.Series(np.random.default_rng(9).normal(1.68, 0.09, 200))),
             "cm"),
    },
    "unit:energy": {
        "kJ, against kcal": (lambda: R.atwater_unit(_energy_frame("kj"), "energy"), "kcal"),
        "macronutrients in another unit than grams":
            (lambda: R.atwater_unit(_energy_frame("kcal", grams=False), "energy"), None),
    },
    "sex_coding": {
        "1 male, 2 female (NHANES, the CDC growth charts)":
            (lambda: R.labels_spell_sex(pd.Series([1, 2] * 50)), None),
        "1 female, 2 male": (lambda: R.labels_spell_sex(pd.Series([2, 1] * 50)), None),
        "0/1 either way": (lambda: R.labels_spell_sex(pd.Series([0, 1] * 50)), None),
    },
}
# The reading itself, settled by its own test (the test can settle; it is no constant False).
POSITIVE_FIXTURES: dict[str, tuple[Any, Any]] = {
    "role:identifier": (lambda: R.names_rows(pd.Series(np.arange(1, 501))), "identifier"),
    "role:time": (lambda: R.dates_order_rows(
        pd.Series(pd.to_datetime("2024-01-01") + pd.to_timedelta(np.tile([0, 30, 60, 90], 50), "D")
                  + pd.to_timedelta(np.repeat(np.arange(50), 4), "D")),
        pd.Series(np.repeat(np.arange(50), 4))), "time"),
    "role:flag": (lambda: R.flag_marks_blanks(
        pd.Series((np.arange(200) % 7 == 0).astype(int)),
        pd.Series(np.where(np.arange(200) % 7 == 0, np.nan, 1.0))), "flag"),
    "role:excluded": (lambda: R.constant_values(pd.Series([3.0] * 40)), "excluded"),
    "role:energy": (lambda: R.energy_follows_macronutrients(_energy_frame("kcal"), "energy"),
                    "energy"),
    "role:covariate": (lambda: R.characteristic_predictor(
        "age", pd.Series(np.repeat(np.arange(30, 90), 2).astype(float)),
        pd.Series(np.repeat(np.arange(60), 2))), "covariate"),
    "code_or_count": (lambda: R.amounts_by_values(
        pd.Series(np.random.default_rng(10).normal(25, 4, 100))), "amount"),
    "unit:height": (lambda: R.height_in_band(pd.Series(
        np.random.default_rng(11).normal(168, 9, 200))), "cm"),
    "unit:energy": (lambda: R.atwater_unit(_energy_frame("kcal"), "energy"), "kcal"),
    "sex_coding": (lambda: R.labels_spell_sex(pd.Series(["F", "M"] * 50)), "female=F"),
}


def test_c1_every_value_settled_kind_declares_its_alternatives_and_a_fixture_for_each():
    """BLUEPRINT §14.3: "Each alternative has a fixture showing that the test does not settle it."
    Every kind with a value test rejects each alternative it declares, and has one fixture per
    alternative here; a kind without one is settled by the user."""
    for kind, rule in R.KIND_RULES.items():
        assert rule.alternatives, kind
        if not rule.value_settleable:
            assert rule.settled_by and not rule.rejects, kind
            continue
        rejected = {a for a, _ in rule.rejects}
        assert rejected == set(rule.alternatives), (kind, set(rule.alternatives) ^ rejected)
        assert set(ALTERNATIVE_FIXTURES.get(kind, {})) == rejected, kind
        assert kind in POSITIVE_FIXTURES or kind == "role:exposure", kind


@pytest.mark.parametrize("kind,alternative", [(k, a) for k, alts in ALTERNATIVE_FIXTURES.items()
                                              for a in alts])
def test_c2_no_value_test_settles_a_reading_its_alternative_would_also_pass(kind, alternative):
    make, forbidden = ALTERNATIVE_FIXTURES[kind][alternative]
    verdict = make()
    assert isinstance(verdict, R.Verdict)
    assert not (verdict.settles and (forbidden is None or verdict.value == forbidden)), \
        (kind, alternative, verdict)


@pytest.mark.parametrize("kind", list(POSITIVE_FIXTURES))
def test_c3_each_value_test_can_settle_its_own_reading(kind):
    make, value = POSITIVE_FIXTURES[kind]
    verdict = make()
    assert verdict.settles and str(verdict.value).startswith(value), (kind, verdict)


def test_c4_the_gates_six_fixtures_each_go_unsettled():
    """The fixtures the gate named: a household line number (1..k repeating across households);
    a measurement varying within units; FIPS 1–56 and float-written 1.0–5.0 codes; a 70-label
    category; x10^3/uL against U/L; sex 1/2 coded either way."""
    assert not R.names_rows(roster()["person_no"]).settles
    diary = sleep_diary()
    assert not R.dates_order_rows(diary["hours"], diary["participant_id"]).settles
    assert not R.amounts_by_values(pd.Series(FIPS)).settles
    assert not R.amounts_by_values(coded()["education"]).settles
    assert R.KIND_RULES["role:free_text"].test is None
    assert R.outcome_unit_reading("WBC (x10^3/uL)") is None or \
        not R.outcome_unit_reading("WBC (x10^3/uL)").settled
    assert not R.labels_spell_sex(pd.Series([1, 2] * 10)).settles
    assert not R.labels_spell_sex(pd.Series([2, 1] * 10)).settles


# ═════════════════════════════════════════════════════════════════════════════
# D · every confirmation is honored: the property over the CONSUMERS registry
# ═════════════════════════════════════════════════════════════════════════════


def _confirmed(**slots: Any) -> ProjectState:
    return ProjectState(target="y", **slots)


def _probe_design_spec(value: str) -> bool:
    from turbotab.core.models.pipeline import design_spec

    X = pd.DataFrame({"smoking": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]})
    spec = design_spec(_confirmed(roles={"smoking": "covariate"},
                                  shape_confirmations={"code_or_count:smoking": value}),
                       X, ["smoking"])
    return "smoking" in spec.categorical


def _probe_imputation(value: str) -> bool:
    from turbotab.core.methods.missing import imputation_frame
    from turbotab.core.models.pipeline import design_spec

    X = pd.DataFrame({"smoking": [1.0, 2.0, 3.0, np.nan, 2.0, 3.0, 1.0, 2.0]})
    state = _confirmed(roles={"smoking": "covariate"}, missing="multiple_imputation",
                       shape_confirmations={"code_or_count:smoking": value})
    spec = design_spec(state, X, ["smoking"])
    return imputation_frame(spec, X, pd.Series(np.arange(8.0)), "regression")[2]["smoking"] == "categorical"


def _probe_models_validator(value: str) -> bool:
    state = _confirmed(roles={"smoking": "covariate"},
                       shape_confirmations=({"code_or_count:smoking": value} if value else {}))
    ctx = {"state": state, "column_info": {"smoking": {"dtype": "integer", "n_unique": 3}},
           "columns": ["smoking", "y"], "target": "y", "task": "regression"}
    try:
        d.validate({"kind": "select_models", "models": ["linear"]}, ctx)
    except d.Refusal as refused:
        return refused.code == "reading_unsettled"
    return False


def _probe_model_predictors(value: str) -> bool:
    from turbotab.core.models.pipeline import model_predictors

    state = _confirmed(roles={"hours": value, "age": "covariate"}, roles_unconfirmed=["hours"],
                       role_confirmations={"hours": value})
    return "hours" in model_predictors(state)


def _clusters_frame() -> pd.DataFrame:
    """Households of 3 (40 of them) inside 12 sites of 10 rows: two groupings, each repeating."""
    return pd.DataFrame({"hhid": np.repeat(np.arange(40), 3),
                         "person_no": np.repeat(np.arange(12), 10)}, index=np.arange(120))


def _probe_resolve_clusters(value: str) -> Any:
    from turbotab.core.models.inference import resolve_clusters

    state = _confirmed(grain=GrainSpec(grain="one_row_per_unit"),
                       roles={"hhid": "identifier", "person_no": "identifier"},
                       reading_confirmations={f"cluster:{c}": ("yes" if c == value else "no")
                                              for c in ("hhid", "person_no")})
    found = resolve_clusters(state, _clusters_frame())
    return found.column, found.n_clusters


def _probe_split_inputs(value: str) -> Any:
    from turbotab.core.stages.rows import split_inputs

    class Store:
        def materialize(self, columns: Any, ids: Any) -> pd.DataFrame:
            return _clusters_frame().loc[ids, list(columns)]

    state = _confirmed(grain=GrainSpec(grain="one_row_per_unit"),
                       roles={"hhid": "identifier", "person_no": "identifier"},
                       reading_confirmations={f"cluster:{c}": ("yes" if c == value else "no")
                                              for c in ("hhid", "person_no")})
    return split_inputs(state, np.arange(120), Store(), "regression")["grouped_by"]


def _probe_seal_inputs(value: str) -> Any:
    from turbotab.core.seal import seal_inputs

    class Store:
        columns = ["hhid", "person_no"]

        def materialize(self, columns: Any, ids: Any) -> pd.DataFrame:
            return _clusters_frame().loc[ids, list(columns)]

    state = _confirmed(grain=GrainSpec(grain="repeated"),
                       roles={"hhid": "identifier", "person_no": "identifier"},
                       reading_confirmations={f"cluster:{c}": ("yes" if c == value else "no")
                                              for c in ("hhid", "person_no")})
    return seal_inputs(state, np.arange(120), Store(), "regression", holdout=0.2, seed=0).grouped_by


def _probe_outcome_unit(value: str) -> Any:
    from turbotab.core.units import outcome_unit

    return outcome_unit("WBC (x10^3/uL)", recorded=value or None)


def _probe_voice_unit(value: str) -> str:
    from turbotab.core import voice

    state = ProjectState(target="ALT (IU)", outcome_unit=value or None)
    return voice.sentence_for(d.SetTarget(column="ALT (IU)"), state, {})


def _probe_coach_suffix(value: str) -> str:
    from turbotab.core.coach import _unit_suffix

    state = ProjectState(column_units={"energy": ColumnUnitSpec(unit=value)} if value else None)
    return _unit_suffix("energy", state)


def _probe_sex_codes(value: str) -> Any:
    from turbotab.core.detectors.plausibility import sex_codes

    coded_ = sex_codes(pd.DataFrame({"sex": [1, 2, 2]}), {"sex": value} if value else None)
    return None if coded_ is None else tuple(coded_.tolist())


def _probe_sex_column(value: str) -> Any:
    from turbotab.core.stages.proposals import sex_column

    frame = pd.DataFrame({"sex": [1, 2, 1, 2]})
    state = ProjectState(sex_codings={"sex": value} if value else None)
    return sex_column({"sex": {"dtype": "integer"}}, frame, {}, state=state)


def _probe_goldberg(value: str) -> Any:
    from turbotab.core.stages.proposals import goldberg_proposal

    frame, _ = pounds()
    frame = frame.rename(columns={"age_years": "age", "height_cm": "height"})
    state = ProjectState(target="sbp", reading_confirmations={"unit:weight": value})
    info = {c: {"dtype": "numeric"} for c in frame.columns}
    offer = goldberg_proposal(frame, info, energy="energy_kcal", unit="kcal", sex="sex",
                              sex_levels={"F": "female", "M": "male"}, roles={}, target="sbp",
                              base=pd.Series(True, index=frame.index), state=state)
    return offer["rule"]["weight_unit"], offer["affected"]


def _probe_energy_days(value: str) -> Any:
    from turbotab.core.stages.proposals import energy_unit_reading

    frame = pd.DataFrame({"energy_kcal": np.random.default_rng(12).normal(2100, 400, 200)})
    recorded = ColumnUnitSpec(unit="kcal", days=int(value)) if value else None
    reading = energy_unit_reading(frame, "energy_kcal", recorded)
    return reading["days"], reading["confirmed"]


def _probe_time_column(value: str) -> Any:
    state = _confirmed(shape_confirmations={"time_column:visit": value} if value else None)
    r = R.time_column_reading(state, {"repeats": {"replicate_index": "visit"}})
    return r.settled


# (consumer, kind) -> (probe, {alternative or None for unanswered: the behavior it must produce})
PROBES: dict[tuple[str, str], tuple[Any, dict[Any, Any]]] = {
    ("turbotab.core.models.pipeline:design_spec", "code_or_count"):
        (_probe_design_spec, {"code": True, "amount": False}),
    ("turbotab.core.methods.missing:imputation_frame", "code_or_count"):
        (_probe_imputation, {"code": True, "amount": False}),
    ("turbotab.core.decisions:_models_read_settled_readings", "code_or_count"):
        (_probe_models_validator, {"": True, "code": False, "amount": False}),
    ("turbotab.core.models.pipeline:model_predictors", "role"):
        (_probe_model_predictors, {"time": False, "exposure": True, "excluded": False,
                                   "covariate": True}),
    ("turbotab.core.models.inference:resolve_clusters", "cluster"):
        (_probe_resolve_clusters, {"hhid": ("hhid", 40), "person_no": ("person_no", 12),
                                   "neither": (None, 0)}),
    ("turbotab.core.seal:seal_inputs", "cluster"):
        (_probe_seal_inputs, {"hhid": "hhid", "person_no": "person_no", "neither": None}),
    ("turbotab.core.stages.rows:split_inputs", "cluster"):
        (_probe_split_inputs, {"hhid": "hhid", "person_no": "person_no", "neither": None}),
    ("turbotab.core.units:outcome_unit", "outcome_unit"):
        (_probe_outcome_unit, {"": (None, None), "1000 cells/uL": ("1000 cells/uL", "decision")}),
    ("turbotab.core.voice:_outcome_unit", "outcome_unit"):
        (_probe_voice_unit, {"": "`ALT (IU)` was chosen as the outcome.",
                             "U/L": "`ALT (IU)` was chosen as the outcome, in U/L."}),
    ("turbotab.core.coach:_unit_suffix", "unit:energy"):
        (_probe_coach_suffix, {"": "", "kcal": " kcal", "kj": " kJ"}),
    ("turbotab.core.detectors.plausibility:sex_codes", "sex_coding"):
        (_probe_sex_codes, {"": None, "female=2,male=1": (1.0, 2.0, 2.0),
                            "female=1,male=2": (2.0, 1.0, 1.0)}),
    ("turbotab.core.stages.proposals:sex_column", "sex_coding"):
        (_probe_sex_column, {"": (None, {}),
                             "female=2,male=1": ("sex", {"2": "female", "1": "male"}),
                             "female=1,male=2": ("sex", {"1": "female", "2": "male"})}),
    ("turbotab.core.stages.proposals:energy_unit_reading", "day_count"):
        (_probe_energy_days, {"": (1, False), "1": (1, True), "2": (2, True)}),
    ("turbotab.core.stages.working:time_column", "time_column"):
        (_probe_time_column, {"": False, "orders": True}),
}


def test_d1_every_number_changing_consumer_of_a_registry_kind_has_a_probe():
    """The property ranges over :data:`readings.CONSUMERS`: each consumer that can change a number
    names the registry kinds it reads, and each (consumer, kind) has a probe below that confirms
    every alternative."""
    declared = {(c.where, k) for c in R.CONSUMERS if c.changes for k in c.kinds}
    assert declared, "no consumer declares the kinds it reads"
    assert declared <= set(PROBES) | GOLDBERG_PROBED, declared - set(PROBES) - GOLDBERG_PROBED
    for _, kind in declared:
        assert kind in R.KINDS or kind.startswith("role") or kind in R.KIND_RULES, kind


GOLDBERG_PROBED = {("turbotab.core.stages.proposals:goldberg_proposal", "unit:weight")}


@pytest.mark.parametrize("key", list(PROBES), ids=lambda k: f"{k[0].split(':')[1]}-{k[1]}")
def test_d2_confirming_each_alternative_produces_its_behavior(key):
    probe, behaviors = PROBES[key]
    seen = {}
    for value, want in behaviors.items():
        got = probe(value)
        assert got == want, (key, value, got, want)
        seen[value] = got
    assert len({json.dumps(v, default=str, sort_keys=True) for v in seen.values()}) > 1, key


def test_d3_the_goldberg_offer_honors_the_recorded_weight_unit():
    """(stages.proposals:goldberg_proposal, unit:weight): kg and lb give different rules and counts, each
    the screen's own (the lb count reads the weights converted exactly)."""
    kg_unit, kg_count = _probe_goldberg("kg")
    lb_unit, lb_count = _probe_goldberg("lb")
    assert (kg_unit, lb_unit) == ("kg", "lb") and kg_count != lb_count


# ═════════════════════════════════════════════════════════════════════════════
# E · the ask stays light
# ═════════════════════════════════════════════════════════════════════════════


def _record(seq: int, decision: Any) -> d.DecisionRecord:
    return d.DecisionRecord(id=f"r{seq}", seq=seq, at="2026-10-03T00:00:00Z", decision=decision)


def test_e1_a_block_confirmation_settles_exactly_the_readings_it_lists():
    """BLUEPRINT §14.2 (quoted above): each listed reading lands where its own confirmation would,
    with the value shown; nothing else in the state changes; one revert undoes the block."""
    roles = d.SetRoles(roles={"a": "exposure", "b": "covariate", "c": "covariate"},
                       unconfirmed=["a", "b", "c"])
    block = d.ConfirmReadings(items=[
        d.ReadingItem(reading="role", column="a", value="exposure"),
        d.ReadingItem(reading="code_or_count", column="b", value="code"),
        d.ReadingItem(reading="sex_coding", column="sex", value="female=2,male=1"),
        d.ReadingItem(reading="unit", column="weight", value="lb")])
    before = d.fold([_record(1, roles)])
    after = d.fold([_record(1, roles), _record(2, block)])
    assert after.role_confirmations == {"a": "exposure"}
    assert after.shape_confirmations == {"code_or_count:b": "code"}
    assert after.sex_codings == {"sex": "female=2,male=1"}
    assert after.reading_confirmations == {"unit:weight": "lb"}
    changed = {k for k in ProjectState.model_fields if getattr(before, k) != getattr(after, k)}
    assert changed == {"role_confirmations", "shape_confirmations", "sex_codings",
                       "reading_confirmations"}
    assert R.unsettled(after) == ["b", "c"]  # b's role and c were never listed
    undone = d.fold([_record(1, roles), _record(2, block), _record(3, d.Revert(decision_id="r2"))])
    assert undone == before
    with pytest.raises(Exception):
        d.ConfirmReadings(items=[d.ReadingItem(reading="role", column="a", value="exposure"),
                                 d.ReadingItem(reading="role", column="a", value="covariate")])


def test_e2_the_ask_lists_each_reading_once_by_consequence_with_families_grouped():
    """The unsettled readings a consumer needs are listed once, ordered by consequence (which rows
    belong together, then roles, then codes by how many indicators they would make), a homogeneous
    family (one kind, one guess, one name pattern) as one line; the block exit lists each reading
    with its best guess."""
    genes = [R.Reading((f"ENSG{i:011d}",), "role", "exposure") for i in range(40)]
    codes = [R.Reading(("smoking",), "code_or_count", "code", evidence="`3` whole-number values"),
             R.Reading(("_STATE",), "code_or_count", "amount", evidence="`51` whole-number values")]
    cluster = R.Reading(("hhid",), "cluster", "yes")
    listed = R.ordered([*codes, *genes, cluster, codes[0]])
    assert [r.kind for r in listed][:2] == ["cluster", "role"] and len(listed) == 43
    assert [r.column for r in listed if r.kind == "code_or_count"] == ["_STATE", "smoking"]
    groups = R.families(listed)
    assert len(groups) == 4 and len(groups[1]) == 40
    text = R.ask_text(listed)
    assert text.count("40 columns like `ENSG") == 1 and text.count("smoking") == 1
    block = R.ask_exits(listed)[0]["decision"]
    assert block["kind"] == "confirm_readings" and len(block["items"]) == 43


NHANES_ROLES_TRUTH = {"SEQN": "identifier", "cycle_begin_year": "covariate", "gender": "covariate",
                      "kcal": "energy"}


@pytest.mark.parametrize("purpose", ["inference", "prediction"])
def test_e3_the_nhanes_reference_journey_asks_few_questions(purpose, tmp_path):
    """BLUEPRINT §14.3: "The gate also reports how many questions a reference journey asks." The
    NHANES export through the server, answered from its declared truth (``truths.FIXTURE_TRUTHS``),
    to the fit: the interview's questions plus the asks (refusals that only ask for readings), each
    ask one card. Recorded in ``nhanes_questions_<purpose>.json`` beside the run for the report."""
    from turbotab.core.tests.stage_harness import NHANES
    from turbotab.core.tests.truths import fixture_truth

    if not NHANES.is_file():
        pytest.skip("the NHANES export is not on this machine")
    plan = answers("glucose", ["dietary"], purpose={"kind": "set_purpose", "purpose": purpose})
    truth = fixture_truth(NHANES.name)
    counted = {"questions": 0, "asks": 0, "readings_asked": 0}
    with local_server(tmp_path / "home") as client:
        drive = open_project(client, NHANES, truth)
        url = f"/api/projects/{drive.pid}/decisions"
        for key in ORDER:
            step = drive.reach(key, timeout=600)
            if step["status"] not in ("open", "waiting"):
                continue
            counted["questions"] += 1
            body = plan.get(key)
            if key == "roles":
                proposals = drive.artifact("roles")["columns"]
                body = {"kind": "set_roles", "roles": {c["column"]: c["proposed"] for c in proposals}}
                for column, role in body["roles"].items():
                    truth.setdefault(f"role:{column}", role)
            from turbotab.core.tests.acceptance.server_drive import _post_when_reached

            r = _post_when_reached(client, url, body)
            while r.status_code == 409 and r.json()["error"]["code"] in ASKING:
                counted["asks"] += 1
                listed = asked(r.json()["error"]["exits"])
                counted["readings_asked"] += len(listed) or 1
                counted.setdefault("listed", []).extend(f"{k}:{c}" for k, c in listed)
                from turbotab.core.tests.truths import answers as truth_answers

                for decision in truth_answers(r.json()["error"], truth):
                    assert client.post(url, json=decision).status_code == 200
                r = _post_when_reached(client, url, body)
            assert r.status_code == 200, (key, r.text[:600])
        drive.artifact("fit", timeout=900)
    (tmp_path / f"nhanes_questions_{purpose}.json").write_text(json.dumps(counted))
    print(f"NHANES {purpose}: {counted}")
    assert counted["asks"] <= 3, counted
