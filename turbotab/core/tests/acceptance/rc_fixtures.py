"""Fixtures for MS5 · regression calibration (``test_ms5_regression_calibration.py``).

Each generator draws people with a known usual intake, their recall days with classical day-to-day
error, and an outcome on the usual intake, so every truth the tests check against is the
generator's own: the coefficients, the error covariance and the design.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

RSCRIPT = shutil.which("Rscript")
needs_r = pytest.mark.skipif(RSCRIPT is None, reason="R (Rscript) is not installed")

# The usual intakes' kcal from each source (all-components), their mean and the outcome's
# coefficient per 100 kcal of each: carbohydrate is the reference the contrasts are read against.
SOURCES = ("protein", "fat", "carbohydrate", "alcohol")
ATWATER = {"protein": 4.0, "fat": 9.0, "carbohydrate": 4.0, "alcohol": 7.0}
MEAN_KCAL = {"protein": 330.0, "fat": 700.0, "carbohydrate": 900.0, "alcohol": 90.0, "other": 80.0}
BETA_PER_100 = {"protein": -1.0, "fat": 2.0, "carbohydrate": 0.5, "alcohol": 3.0, "other": 0.0}
# Day-to-day standard deviation of each source's kcal: about the between-person spread, a
# within/between variance ratio near 1.3 (24-hour recalls run from about 1 to 4).
DAY_SD = {"protein": 90.0, "fat": 200.0, "carbohydrate": 240.0, "alcohol": 50.0, "other": 25.0}
USUAL_SD = {"protein": 80.0, "fat": 200.0, "carbohydrate": 240.0, "alcohol": 60.0, "other": 25.0}


def run_r(script: str, workdir: Path) -> dict[str, Any]:
    """Run ``script`` with Rscript in ``workdir``; the script writes ``out.json``."""
    path = workdir / "reference.R"
    path.write_text(script)
    done = subprocess.run([RSCRIPT, str(path)], cwd=workdir, capture_output=True, text=True,
                          timeout=600)
    assert done.returncode == 0, done.stderr[-3000:]
    return json.loads((workdir / "out.json").read_text())


def replicate_people(rng: np.random.Generator, n: int, k: Any, *, beta: float = 0.5,
                     error_sd: float = 2.8, binary: bool = False):
    """One intake: true X given age and sex (residual var 4), recalls W = X + U (sd ``error_sd``),
    the outcome on X. Returns (age, sex, W days, person, y, x)."""
    age = rng.normal(50, 10, n)
    sex = rng.integers(0, 2, n).astype(float)
    x = 10 + 0.05 * age + 1.0 * sex + rng.normal(0, 2, n)
    k = np.broadcast_to(np.asarray(k), (n,))
    person = np.repeat(np.arange(n), k)
    w = x[person] + rng.normal(0, error_sd, person.size)
    if binary:
        eta = -1 + beta * (x - 11) + 0.02 * (age - 50) - 0.3 * sex
        y = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(float)
    else:
        y = 2 + beta * x + 0.03 * age - 0.4 * sex + rng.normal(0, 2, n)
    return age, sex, w, person, y, x


def usual_kcal(rng: np.random.Generator, n: int, age: np.ndarray, female: np.ndarray) -> dict:
    """Each person's usual kcal from every source, correlated through total intake (a person who
    eats more eats more of everything) and shifted by age and sex."""
    scale = np.exp(rng.normal(0, 0.18, n) - 0.12 * female - 0.003 * (age - 50))
    out = {}
    for s in (*SOURCES, "other"):
        own = rng.normal(0, USUAL_SD[s] * 0.6, n)
        out[s] = np.maximum(MEAN_KCAL[s] * scale + own, 0.15 * MEAN_KCAL[s])
    return out


def recall_days(rng: np.random.Generator, usual: dict, k: np.ndarray) -> tuple[np.ndarray, dict]:
    """Each person's ``k`` recall days: each source's kcal is the usual plus classical error, the
    sources' errors correlated (a big day is big in everything), floored at a small positive
    amount so grams stay positive."""
    n = len(next(iter(usual.values())))
    person = np.repeat(np.arange(n), k)
    common = rng.normal(0, 1, person.size)
    days = {}
    for s in (*SOURCES, "other"):
        error = DAY_SD[s] * (0.5 * common + np.sqrt(1 - 0.25) * rng.normal(0, 1, person.size))
        days[s] = np.maximum(usual[s][person] + error, 0.05 * MEAN_KCAL[s])
    return person, days


def chain2_table(seed: int = 2, strata: int = 8, psus: int = 2, per_psu: int = 22, k: int = 2,
                 missing_share: float = 0.15, household: bool = False
                 ) -> tuple[pd.DataFrame, dict[str, Any]]:
    """MODELING_SEQUENCE §6 chain 2: repeated 24-hour recalls (``k`` per person, long format) in a
    stratified two-stage design (``strata`` × ``psus`` PSUs, ``per_psu`` people each, weights by
    PSU), every energy source and total energy recorded on each day, BMI blank for a share of
    people (more often among the older: missing at random given age), systolic blood pressure on
    the usual intakes. Returns (table, truth)."""
    rng = np.random.default_rng(seed)
    n = strata * psus * per_psu
    stratum = np.repeat(np.arange(1, strata + 1), psus * per_psu)
    psu = np.tile(np.repeat(np.arange(1, psus + 1), per_psu), strata)
    weight = np.round(np.exp(rng.normal(9.5, 0.4, strata * psus)), 1)[(stratum - 1) * psus + psu - 1]
    age = np.round(rng.uniform(20, 80, n), 1)
    female = rng.integers(0, 2, n)
    usual = usual_kcal(rng, n, age, female)
    bmi = np.round(22 + 0.06 * (age - 50) + 1.5 * (1 - female) + 0.002 * (usual["fat"] - 700)
                   + rng.normal(0, 3.0, n), 1)
    sbp = (110 + sum(BETA_PER_100[s] * usual[s] / 100 for s in usual) + 0.35 * age
           - 3 * female + 0.6 * bmi + rng.normal(0, 6, n))
    blank = rng.random(n) < missing_share * (0.4 + 1.2 * (age - 20) / 60)
    kk = np.full(n, k)
    person, days = recall_days(rng, usual, kk)
    rows = pd.DataFrame({
        "seqn": 10_000 + person, "day": np.concatenate([np.arange(1, j + 1) for j in kk]),
        "sdmvstra": stratum[person], "sdmvpsu": psu[person], "wtdr2d": weight[person],
        "age": age[person], "sex": np.where(female[person] == 1, "F", "M"),
        "bmi": np.where(blank[person], np.nan, bmi[person]),
        **{f"{s}_g": np.round(days[s] / ATWATER[s], 2) for s in SOURCES},
        "sbp": np.round(sbp[person], 1)})
    rows["energy_kcal"] = np.round(
        sum(ATWATER[s] * rows[f"{s}_g"] for s in SOURCES) + days["other"], 1)
    if household:  # two people to a household, the same PSU
        rows["household"] = 50_000 + person // 2
    truth = {"beta_per_kcal": {s: BETA_PER_100[s] / 100 for s in BETA_PER_100}, "n": n,
             "usual": usual, "blank": blank, "weight": weight, "stratum": stratum, "psu": psu}
    return rows, truth


def protein_recalls(seed: int = 5, n: int = 600, k: int = 2) -> pd.DataFrame:
    """``k`` 24-hour recalls of protein and energy per person (long format): each day's energy and
    protein vary around the person's usual intake; LDL depends on usual protein density, age and
    sex."""
    rng = np.random.default_rng(seed)
    age = np.round(rng.uniform(25, 75, n), 1)
    sex = rng.choice(["F", "M"], n)
    energy = rng.normal(2100, 330, n) + 250 * (sex == "M")
    protein = 0.035 * energy + 7 * rng.standard_normal(n) + 0.05 * (age - 50)
    ldl = 100 + 600 * protein / energy + 0.2 * age + 4 * (sex == "M") + rng.normal(0, 6, n)
    rows = []
    for i in range(n):
        for d in range(k):
            e = energy[i] + rng.normal(0, 450)
            rows.append({"participant_id": f"P{i:04d}", "day": d + 1, "age": age[i], "sex": sex[i],
                         "energy_kcal": round(e, 1),
                         "protein_g": round(protein[i] * e / energy[i] + rng.normal(0, 11), 2),
                         "ldl": round(ldl[i], 1)})
    return pd.DataFrame(rows)


__all__ = ["ATWATER", "BETA_PER_100", "RSCRIPT", "SOURCES", "chain2_table", "needs_r",
           "protein_recalls", "recall_days", "replicate_people", "run_r", "usual_kcal"]
