"""Fixtures and independent references for the MS8 scales acceptance tests.

Every reference here is independent of ``turbotab.core.methods.scales``: pandas arithmetic for the
keyed items and the score, NumPy by hand for the conditional regression calibration, statsmodels
for the outcome models, and R (``psych``, ``mecor``) run as a subprocess on CSVs the test writes.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pytest

RSCRIPT = shutil.which("Rscript")
needs_r = pytest.mark.skipif(RSCRIPT is None, reason="R (Rscript) is not installed")

PSS_ITEMS = [f"pss_{j}" for j in range(1, 11)]
PSS_REVERSE = ["pss_4", "pss_5", "pss_7", "pss_8"]  # the PSS-10's positively worded items
DQ_ITEMS = [f"dq_{j}" for j in range(1, 7)]
DQ_RETEST = [f"{c}_t2" for c in DQ_ITEMS]


def run_r(script: str, workdir: Path) -> dict[str, Any]:
    """Run ``script`` with Rscript in ``workdir``; the script writes ``out.json``."""
    path = workdir / "reference.R"
    path.write_text(script)
    done = subprocess.run([RSCRIPT, str(path)], cwd=workdir, capture_output=True, text=True,
                          timeout=600)
    assert done.returncode == 0, done.stderr[-3000:]
    return json.loads((workdir / "out.json").read_text())


def keyed(frame: pd.DataFrame, items: Sequence[str], reverse: Sequence[str], low: int,
          high: int) -> pd.DataFrame:
    """The items as the instrument's key reads them, by pandas: reversed ones are low + high − x."""
    out = frame[list(items)].astype(float).copy()
    for c in reverse:
        out[c] = low + high - out[c]
    return out


def likert(rng: np.random.Generator, latent: np.ndarray, loadings: Sequence[float], low: int,
           high: int, spread: float = 1.1) -> np.ndarray:
    """Answers on ``low``–``high`` from a common latent variable: λ·latent + unique part, rounded and
    clipped (a congeneric reflective scale)."""
    n = len(latent)
    mid = (low + high) / 2
    cols = []
    for lam in loadings:
        z = lam * latent + np.sqrt(1 - lam ** 2) * rng.standard_normal(n)
        cols.append(np.clip(np.round(mid + spread * z), low, high))
    return np.column_stack(cols)


def survey_scale_table(seed: int = 4, n: int = 1500) -> pd.DataFrame:
    """Chain 4's table (MODELING_SEQUENCE §6): a reflective survey scale and a formative diet score
    with a repeat administration, an ordered outcome, and two covariates.

    * ``pss_1`` … ``pss_10`` (0–4): a perceived-stress scale, one factor; ``pss_4``, ``pss_5``,
      ``pss_7`` and ``pss_8`` are recorded positively worded (4 − the answer), as the PSS-10's are.
    * ``dq_1`` … ``dq_6`` (0–10): a diet-quality index's components, each its usual intake plus
      day-to-day error; a repeat administration ``dq_1_t2`` … ``dq_6_t2`` for 45% of participants.
    * ``wellbeing`` (1–5): the cumulative logit of 0.6·stress + 0.08·(diet quality − 30) + age and sex.
    """
    rng = np.random.default_rng(seed)
    age = np.round(rng.uniform(25, 75, n), 1)
    male = rng.random(n) < 0.5
    stress = rng.standard_normal(n) + 0.01 * (age - 50) - 0.25 * male
    pss = likert(rng, stress, [0.75, 0.7, 0.65, 0.6, 0.7, 0.55, 0.65, 0.6, 0.7, 0.75], 0, 4)
    frame = pd.DataFrame({"participant_id": [f"P{i:05d}" for i in range(n)], "age": age,
                          "sex": np.where(male, "M", "F")})
    for j, c in enumerate(PSS_ITEMS):
        frame[c] = (4 - pss[:, j]) if c in PSS_REVERSE else pss[:, j]
        frame[c] = frame[c].astype(int)
    care = rng.standard_normal(n)  # a weak common cause: components of an index barely cohere
    usual = np.column_stack([np.clip(5 + 1.6 * rng.standard_normal(n) + 0.4 * care, 0, 10)
                             for _ in DQ_ITEMS])
    first = np.clip(np.round(usual + 1.4 * rng.standard_normal(usual.shape)), 0, 10)
    second = np.clip(np.round(usual + 1.4 * rng.standard_normal(usual.shape)), 0, 10)
    repeat = rng.random(n) < 0.45
    for j, c in enumerate(DQ_ITEMS):
        frame[c] = first[:, j].astype(int)
    for j, c in enumerate(DQ_RETEST):
        frame[c] = np.where(repeat, second[:, j], np.nan)
    eta = 0.6 * stress + 0.08 * (usual.sum(axis=1) - 30) + 0.01 * (age - 50) + 0.3 * male
    latent = eta + rng.logistic(size=n)
    cuts = np.quantile(latent, [0.15, 0.4, 0.7, 0.9])
    frame["wellbeing"] = 1 + np.searchsorted(cuts, latent)
    return frame


def linear_scale_table(seed: int = 11, n: int = 1200, missing: float = 0.0) -> pd.DataFrame:
    """A reflective 8-item scale (1–5, items 3 and 6 recorded reversed), two covariates it
    correlates with, and a continuous outcome on the latent trait. ``missing``: the share of each
    item's answers blanked, more often for older participants (missing at random given age)."""
    rng = np.random.default_rng(seed)
    age = np.round(rng.uniform(20, 80, n), 1)
    bmi = np.round(rng.normal(27, 4, n) + 0.03 * (age - 50), 1)
    trait = rng.standard_normal(n) + 0.02 * (age - 50) + 0.05 * (bmi - 27)
    items = likert(rng, trait, [0.8, 0.7, 0.75, 0.6, 0.65, 0.7, 0.55, 0.75], 1, 5, spread=1.0)
    frame = pd.DataFrame({"pid": np.arange(n), "age": age, "bmi": bmi})
    names = [f"sat_{j}" for j in range(1, 9)]
    for j, c in enumerate(names):
        frame[c] = (6 - items[:, j]) if c in ("sat_3", "sat_6") else items[:, j]
    frame["sbp"] = np.round(120 + 4.0 * trait + 0.3 * (age - 50) + 0.4 * (bmi - 27)
                            + rng.normal(0, 8, n), 1)
    # A calibration substudy: a reference measure of the trait (in the score's units) for 35% of
    # the participants, blank for the rest.
    substudy = rng.random(n) < 0.35
    frame["sat_ref"] = np.where(substudy, np.round(24 + 4.6 * trait + rng.normal(0, 0.8, n), 2),
                                np.nan)
    if missing:
        p = missing * (0.5 + (age - 20) / 60)
        for c in names:
            frame[c] = frame[c].astype(float).where(rng.random(n) >= p)
    return frame


SAT = [f"sat_{j}" for j in range(1, 9)]
SAT_REVERSE = ["sat_3", "sat_6"]


def household_scale_table(seed: int = 21, households: int = 80, size: int = 10) -> pd.DataFrame:
    """The 8-item reflective scale of :func:`linear_scale_table` answered by ``size`` people in each
    of ``households`` households (``hh``), whose members share part of the trait and part of the
    outcome's error: rows within a household are not independent, so an interval that treats them
    as independent is too narrow."""
    rng = np.random.default_rng(seed)
    n = households * size
    hh = np.repeat(np.arange(households), size)
    age = np.round(rng.uniform(20, 80, n), 1)
    bmi = np.round(rng.normal(27, 4, n) + 0.03 * (age - 50), 1)
    trait = (0.8 * rng.standard_normal(households)[hh] + 0.6 * rng.standard_normal(n)
             + 0.02 * (age - 50))
    trait = (trait - trait.mean()) / trait.std()
    items = likert(rng, trait, [0.8, 0.7, 0.75, 0.6, 0.65, 0.7, 0.55, 0.75], 1, 5, spread=1.0)
    frame = pd.DataFrame({"pid": np.arange(n), "hh": [f"H{h:03d}" for h in hh], "age": age,
                          "bmi": bmi})
    for j, c in enumerate(SAT):
        frame[c] = (6 - items[:, j]) if c in SAT_REVERSE else items[:, j]
    frame["sbp"] = np.round(120 + 4.0 * trait + 0.3 * (age - 50) + 0.4 * (bmi - 27)
                            + rng.normal(0, 6, households)[hh] + rng.normal(0, 6, n), 1)
    return frame


HEI = [f"hei_{j}" for j in range(1, 14)]
HEI_RETEST = [f"{c}_t2" for c in HEI]
# The HEI-2015's thirteen components: six scored 0–5 and seven 0–10 (Reedy et al. 2018, Table 1).
HEI_MAX = [5, 5, 5, 5, 5, 5, 10, 10, 10, 10, 10, 10, 10]


def hei_table(seed: int = 8, n: int = 1000, repeat: float = 0.4) -> pd.DataFrame:
    """A diet-quality index as the HEI-2015 scores it: thirteen continuous component scores, each
    prorated on 0–5 or 0–10 (decimals, as the NCI's SAS macros write them), from a day's intake
    that measures the usual one with day-to-day error, and a repeat recall's components for
    ``repeat`` of the participants; a continuous outcome rising with usual diet quality."""
    rng = np.random.default_rng(seed)
    age = np.round(rng.uniform(20, 80, n), 1)
    quality = rng.standard_normal(n) + 0.01 * (age - 50)
    usual = np.column_stack([np.clip(0.55 * m + 0.18 * m * (0.6 * quality
                                                              + 0.8 * rng.standard_normal(n)),
                                     0, m) for m in HEI_MAX])
    day = np.column_stack([np.clip(usual[:, j] + 0.2 * m * rng.standard_normal(n), 0, m)
                           for j, m in enumerate(HEI_MAX)])
    again = np.column_stack([np.clip(usual[:, j] + 0.2 * m * rng.standard_normal(n), 0, m)
                             for j, m in enumerate(HEI_MAX)])
    frame = pd.DataFrame({"pid": np.arange(n), "age": age})
    for j, c in enumerate(HEI):
        frame[c] = np.round(day[:, j], 2)
    has = rng.random(n) < repeat
    for j, c in enumerate(HEI_RETEST):
        frame[c] = np.where(has, np.round(again[:, j], 2), np.nan)
    frame["sbp"] = np.round(140 - 0.25 * usual.sum(axis=1) + 0.3 * (age - 50)
                            + rng.normal(0, 6, n), 1)
    return frame


def calibrate_by_hand(W: np.ndarray, Z: np.ndarray, error_variance: float) -> tuple[np.ndarray, float]:
    """E[X | W, Z] under classical error of known variance, by the covariance matrix of (W, Z)
    (n − 1): λ = (s²_{W|Z} − σ²_U) / s²_{W|Z}, the Schur complement s²_{W|Z} = S_WW − S_WZ S_ZZ⁻¹ S_ZW."""
    S = np.cov(np.column_stack([W, Z]), rowvar=False)
    s2 = S[0, 0] - S[0, 1:] @ np.linalg.solve(S[1:, 1:], S[1:, 0])
    lam = (s2 - error_variance) / s2
    beta = np.linalg.solve(S[1:, 1:], S[1:, 0])
    fitted = W.mean() + (Z - Z.mean(axis=0)) @ beta
    return fitted + lam * (W - fitted), float(lam)


def omega_script(csv: str, nfactors: int, extra: str = "") -> str:
    """R: psych::omega on the keyed items in ``csv`` (flip = FALSE), the score's own ω from its
    Schmid–Leiman solution and the items' standard deviations, and psych::alpha's raw α."""
    return f"""
suppressMessages({{library(psych); library(jsonlite)}})
x <- read.csv("{csv}")
o <- suppressWarnings(suppressMessages(omega(x, nfactors = {nfactors}, plot = FALSE, flip = FALSE)))
sl <- o$schmid$sl; s <- apply(x, 2, sd); C <- cov(x)
a <- suppressWarnings(alpha(x, check.keys = FALSE, warnings = FALSE))
out <- list(omega_tot = o$omega.tot, omega_h = o$omega_h, alpha = a$total$raw_alpha,
            score_omega_tot = 1 - sum(s^2 * sl[, "u2"]) / sum(C),
            score_omega_h = sum(s * sl[, 1])^2 / sum(C), g = as.numeric(sl[, 1]))
{extra}
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE), "out.json")
"""
