"""R as the independent reference for the MS4 survey acceptance tests (MODELING_SEQUENCE §5 MS4).

R's ``survey`` package (Lumley; 4.5 here) is the reference implementation of design-based
estimation that the field cites: ``svyglm``, ``svycoxph``, ``svyolr``, ``svycontrast`` and
``svyrecvar``. Each test writes its table to a CSV, runs an R script on it in a subprocess, and
reads R's numbers back as JSON. R is never imported or called by the app; a machine without
``Rscript`` skips these tests (:data:`needs_r`).

The fixtures are NHANES-shaped (:func:`nhanes_mortality`): strata of two or three masked variance
units, informative exam weights (oversampled groups carry small weights), and follow-up as the
NCHS public-use linked mortality files give it (``permth_exm``, whole months from the exam to death
or to the end of follow-up, so many deaths tie; ``mortstat`` 1 for a death). Every simulation is
seeded and its truth is the generator's.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pytest

RSCRIPT = shutil.which("Rscript")
needs_r = pytest.mark.skipif(RSCRIPT is None, reason="R (Rscript) is not installed")

R_HEADER = """
suppressMessages({library(survey); library(jsonlite)})
options(survey.lonely.psu = "adjust")
out <- function(x) cat(toJSON(x, digits = NA, auto_unbox = TRUE))
"""


def run_r(script: str, frames: Mapping[str, pd.DataFrame], folder: Path) -> Any:
    """Write each frame to ``<name>.csv`` in ``folder``, run ``script`` there (after
    :data:`R_HEADER`), and return the JSON it prints last."""
    folder.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_csv(folder / f"{name}.csv", index=False)
    path = folder / "reference.R"
    path.write_text(R_HEADER + script)
    done = subprocess.run([RSCRIPT, "--vanilla", str(path)], cwd=folder, capture_output=True,
                          text=True, timeout=900)
    if done.returncode:
        raise RuntimeError(f"R failed:\n{done.stderr[-4000:]}")
    return json.loads(done.stdout.strip().splitlines()[-1])


def nhanes_mortality(seed: int = 2026, *, strata: int = 15, n_per_psu: int = 55,
                     lonely: int = 0, three_psu: int = 1) -> pd.DataFrame:
    """An NHANES-shaped table with linked-mortality follow-up.

    ``strata`` strata of two PSUs each (``three_psu`` of them with three), the last ``lonely`` of
    them with one; ``n_per_psu`` people per PSU, each with a PSU effect. Half the people belong to
    an oversampled group (exam weight about a fifth of the others'). Fiber (g/day) runs lower in
    the oversampled group; the true log hazard ratio per 10 g of fiber is −0.15 in the rest and
    −0.05 in it, so the population and the sample answer differently. Death times are exponential
    in months, follow-up ends 60–200 months after the exam, and both are rounded up to whole months,
    so deaths tie as they do in the public-use files. ``eligible`` is 0 for the under-20s (no
    mortality follow-up in the files: a domain, not a deletion). An ordinal self-rated health
    (1 = excellent … 4 = poor) and a weekly-servings count are drawn too.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for h in range(1, strata + 1):
        n_psu = 1 if h > strata - lonely else (3 if h <= three_psu else 2)
        for j in range(1, n_psu + 1):
            effect = rng.normal(0, 0.25)
            for _ in range(n_per_psu):
                over = rng.random() < 0.5
                age = float(rng.uniform(12, 80))
                female = int(rng.random() < 0.52)
                fiber = float(rng.gamma(4.0, (3.6 if over else 4.6)))
                kcal = float(rng.normal(2050 if not female else 1750, 420))
                weight = float(rng.uniform(2500, 9000) if over else rng.uniform(15000, 52000))
                slope = -0.005 if over else -0.015
                log_hazard = (-6.2 + 0.075 * (age - 50) + 0.25 * (1 - female)
                              + slope * (fiber - 18) + 0.3 * over + effect)
                death = rng.exponential(1.0 / np.exp(log_hazard))
                end = float(rng.uniform(60, 200))
                months = int(np.ceil(min(death, end)))
                health_latent = (0.03 * (age - 50) - 0.04 * (fiber - 18) + 0.5 * over + effect
                                 + rng.logistic())
                health = int(np.digitize(health_latent, [-1.2, 0.4, 1.8])) + 1
                rows.append({
                    "SDMVSTRA": 100 + h, "SDMVPSU": j, "WTMEC2YR": round(weight, 2),
                    "RIDAGEYR": round(age, 1), "female": female, "fiber": round(fiber, 2),
                    "kcal": round(kcal, 1), "over": int(over),
                    "permth_exm": months, "mortstat": int(death <= end),
                    "entry_age": round(age, 3), "exit_age": round(age + months / 12.0, 3),
                    "health": health, "eligible": int(age >= 20),
                })
    frame = pd.DataFrame(rows)
    frame.insert(0, "SEQN", np.arange(1, len(frame) + 1))
    return frame


def nhanes_diet(seed: int = 77, *, strata: int = 15, n_per_psu: int = 60, lonely: int = 0
                ) -> pd.DataFrame:
    """An NHANES-shaped dietary table: one day's protein, fat and carbohydrate (g), total energy
    (kcal: their Atwater energy plus the rest, alcohol and fiber), a dietary day-one weight
    ``WTDRD1``, and two outcomes. The oversampled group (about a fifth of the others' weight) eats
    more carbohydrate and responds to fat more strongly, so the population's substitution differs
    from the sample's. ``crp`` is continuous (mg/L); ``high_crp`` is ``crp`` above 3 mg/L, the
    AHA/CDC high-risk cut point. ``eligible`` is 0 for the under-20s (a domain)."""
    rng = np.random.default_rng(seed)
    rows = []
    for h in range(1, strata + 1):
        n_psu = 1 if h > strata - lonely else 2
        for j in range(1, n_psu + 1):
            effect = rng.normal(0, 0.3)
            for _ in range(n_per_psu):
                over = rng.random() < 0.5
                age = float(rng.uniform(12, 80))
                female = int(rng.random() < 0.52)
                size = rng.normal(0, 1)
                kcal_base = (2050 - 300 * female) * np.exp(0.18 * size)
                share_p = float(np.clip(rng.normal(0.16, 0.03), 0.06, 0.35))
                share_f = float(np.clip(rng.normal(0.34 - 0.04 * over, 0.05), 0.12, 0.55))
                share_o = float(np.clip(rng.normal(0.05, 0.02), 0.0, 0.15))
                share_c = 1.0 - share_p - share_f - share_o
                protein, fat, carb = (kcal_base * share_p / 4, kcal_base * share_f / 9,
                                      kcal_base * share_c / 4)
                kcal = 4 * protein + 9 * fat + 4 * carb + kcal_base * share_o
                fat_effect = 0.006 if over else 0.002  # mg/L per g of fat, carbohydrate held
                crp = (1.2 + fat_effect * (fat - 75) + 0.002 * (carb - 250) + 0.0004 * (kcal - 2000)
                       + 0.025 * (age - 45) + 0.5 * female + 0.4 * over + effect
                       + rng.normal(0, 1.0))
                rows.append({
                    "SDMVSTRA": 200 + h, "SDMVPSU": j,
                    "WTDRD1": round(float(rng.uniform(2500, 9000) if over
                                          else rng.uniform(15000, 52000)), 2),
                    "age": round(age, 1), "female": female, "protein_g": round(protein, 2),
                    "fat_g": round(fat, 2), "carb_g": round(carb, 2), "energy_kcal": round(kcal, 1),
                    "crp": round(crp, 3), "high_crp": int(crp > 3.0), "over": int(over),
                    "eligible": int(age >= 20),
                })
    frame = pd.DataFrame(rows)
    frame.insert(0, "SEQN", np.arange(1, len(frame) + 1))
    return frame


__all__ = ["R_HEADER", "RSCRIPT", "needs_r", "nhanes_diet", "nhanes_mortality", "run_r"]
