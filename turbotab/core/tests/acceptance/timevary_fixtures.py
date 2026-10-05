"""Fixtures and R references for the time-varying exposure acceptance tests (``test_timevary.py``).

**R is the independent reference** for the weights (``ipw::ipwtm``), the marginal structural
model's estimate and robust variance (``geepack::geeglm``, independence working correlation), the
parametric g-formula (``gfoRmula::gformula``) and the E-values (``EValue``). Each test writes its
table to a CSV, runs an R script on it in a subprocess, and reads R's numbers back as JSON. R is never
imported or called by the app; a machine without ``Rscript`` skips those tests (:data:`needs_r`).
The R fits are converged past their defaults (``glm.control(epsilon = 1e-14)``,
``geese.control(epsilon = 1e-14)``), so agreement measures the methods, not the stopping rules.

**Simulated cohorts with a known truth** are the reference where R is not:

* :func:`feedback_cohort`: a DASH-style diet followed at each visit, systolic blood pressure that
  the diet lowers and that prompts the diet, and an unmeasured vascular risk behind both the pressure
  and the cardiovascular event. The diet has **no effect** on the event, so the marginal structural
  model's coefficient is 0 by construction. Standard regression is biased whichever way it treats the
  pressure: adjusted for, it opens diet → pressure ← risk → event; left out, pressure → diet and
  pressure ← risk → event confound.
* :func:`gformula_cohort`: a binary time-varying confounder (hypertension) with no unmeasured cause,
  so the g-formula's lag-1 models are the true ones. Its risks under "always" and "never" are
  computed exactly by enumerating the confounder's two states at each visit (:func:`exact_risks`),
  with no Monte Carlo.

Every simulation is seeded, and its truth is the generator's.
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
from scipy.special import expit

RSCRIPT = shutil.which("Rscript")
needs_r = pytest.mark.skipif(RSCRIPT is None, reason="R (Rscript) is not installed")

R_HEADER = """
suppressMessages({library(jsonlite)})
out <- function(x) cat(toJSON(x, digits = NA, auto_unbox = TRUE))
ctl <- glm.control(epsilon = 1e-14, maxit = 100)
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


def r_dataset(name: str, package: str, folder: Path) -> pd.DataFrame:
    """A dataset shipped with an R package (``haartdat`` from ``ipw``, ``basicdata_nocomp`` from
    ``gfoRmula``), as R holds it."""
    script = (f"suppressMessages(library({package})); d <- as.data.frame({name}); "
              f"write.csv(d, '{name}.csv', row.names = FALSE); out(list(n = nrow(d)))")
    run_r(script, {}, folder)
    return pd.read_csv(folder / f"{name}.csv")


# ── a cohort with treatment–confounder feedback and a null effect ────────────


def feedback_cohort(n: int = 3000, visits: int = 6, seed: int = 2026,
                    censoring: bool = True) -> pd.DataFrame:
    """One row per person per visit until the event or loss to follow-up (see the module text).

    ``sbp_t = 128 + 8 U − 6 dash_{t−1} + N(0, 4²)``;
    ``logit P(dash_t) = −1 + 0.08 (sbp_t − 128) + 1.2 dash_{t−1} + 0.3 female``;
    ``logit P(cvd_t) = −4.2 + 0.9 U + 0.02 (age − 55)``: no term in the diet;
    ``logit P(lost_t) = −3.2 + 0.05 (sbp_t − 128) − 0.4 dash_t`` after a visit without the event
    (``lost`` is 1 on the last visit seen).
    """
    rng = np.random.default_rng(seed)
    female = rng.binomial(1, 0.5, n)
    age = rng.uniform(40, 70, n).round(1)
    U = rng.normal(0, 1, n)
    rows = []
    prev = np.zeros(n)
    alive = np.ones(n, dtype=bool)
    for t in range(visits):
        sbp = 128 + 8 * U - 6 * prev + rng.normal(0, 4, n)
        dash = rng.binomial(1, expit(-1 + 0.08 * (sbp - 128) + 1.2 * prev + 0.3 * female))
        cvd = rng.binomial(1, expit(-4.2 + 0.9 * U + 0.02 * (age - 55)))
        lost = (rng.binomial(1, expit(-3.2 + 0.05 * (sbp - 128) - 0.4 * dash)) * (cvd == 0)
                if censoring else np.zeros(n, dtype=int))
        for i in np.flatnonzero(alive):
            rows.append({"pid": f"P{i:05d}", "visit": t + 1, "dash": int(dash[i]),
                         "sbp": round(float(sbp[i]), 1), "female": int(female[i]),
                         "age": float(age[i]), "lost": int(lost[i]), "cvd": int(cvd[i])})
        alive &= (lost == 0) & (cvd == 0)
        prev = dash.astype(float)
    frame = pd.DataFrame(rows)
    if not censoring:
        frame = frame.drop(columns=["lost"])
    return frame


# ── a cohort whose g-formula models are the true ones ────────────────────────

# logit P(htn_t = 1) = a0 + a_v female + a_l htn_{t−1} + a_a diet_{t−1} + a_t t
HTN = {"a0": -1.2, "female": -0.4, "lag_htn": 2.2, "lag_diet": -1.0, "t": 0.05}
HTN0 = {"a0": -0.8, "female": -0.4}  # at the first visit (no history)
# logit P(diet_t = 1) = b0 + b_v female + b_l htn_t + b_a diet_{t−1}
DIET = {"b0": -1.0, "female": 0.3, "htn": 1.1, "lag_diet": 1.5}
# logit P(event_t = 1) = c0 + c_v female + c_l htn_t + c_a diet_t + c_t t
EVENT = {"c0": -3.0, "female": -0.3, "htn": 0.9, "diet": -0.5, "t": 0.08}


def gformula_cohort(n: int = 4000, visits: int = 5, seed: int = 7) -> pd.DataFrame:
    """One row per person per visit until the event: ``htn`` a binary time-varying confounder the
    diet lowers and that prompts the diet; the diet lowers the event's hazard directly and through
    ``htn``. Every model is logistic in the previous visit's values (:data:`HTN`, :data:`DIET`,
    :data:`EVENT`)."""
    rng = np.random.default_rng(seed)
    female = rng.binomial(1, 0.5, n)
    rows = []
    alive = np.ones(n, dtype=bool)
    htn = np.zeros(n)
    diet = np.zeros(n)
    for t in range(visits):
        if t == 0:
            p = expit(HTN0["a0"] + HTN0["female"] * female)
        else:
            p = expit(HTN["a0"] + HTN["female"] * female + HTN["lag_htn"] * htn
                      + HTN["lag_diet"] * diet + HTN["t"] * t)
        htn = rng.binomial(1, p).astype(float)
        diet = rng.binomial(1, expit(DIET["b0"] + DIET["female"] * female + DIET["htn"] * htn
                                     + DIET["lag_diet"] * diet)).astype(float)
        event = rng.binomial(1, expit(EVENT["c0"] + EVENT["female"] * female + EVENT["htn"] * htn
                                      + EVENT["diet"] * diet + EVENT["t"] * t))
        for i in np.flatnonzero(alive):
            rows.append({"pid": i + 1, "visit": t, "diet": int(diet[i]), "htn": int(htn[i]),
                         "female": int(female[i]), "event": int(event[i])})
        alive &= event == 0
    return pd.DataFrame(rows)


def exact_risks(visits: int, value: int, female_share: float = 0.5) -> float:
    """The risk of the event by the last visit had everyone's diet been ``value`` at every visit,
    by enumerating ``htn``'s two states: ``g_t(l)`` is the probability of being event-free before
    visit t with ``htn_t = l``; the survivors of visit t carry into visit t + 1."""
    total = 0.0
    for female, share in ((1, female_share), (0, 1 - female_share)):
        p1 = expit(HTN0["a0"] + HTN0["female"] * female)
        g = np.array([1 - p1, p1])
        for t in range(visits):
            h = np.array([expit(EVENT["c0"] + EVENT["female"] * female + EVENT["htn"] * l
                                + EVENT["diet"] * value + EVENT["t"] * t) for l in (0, 1)])
            survive = g * (1 - h)
            if t == visits - 1:
                total += share * (1 - survive.sum())
                break
            nxt = np.zeros(2)
            for l in (0, 1):
                q = expit(HTN["a0"] + HTN["female"] * female + HTN["lag_htn"] * l
                          + HTN["lag_diet"] * value + HTN["t"] * (t + 1))
                nxt[1] += survive[l] * q
                nxt[0] += survive[l] * (1 - q)
            g = nxt
    return float(total)


def repeated_measure_cohort(n: int = 800, visits: int = 4, seed: int = 11) -> pd.DataFrame:
    """A repeated continuous outcome (fasting glucose at each visit) with a binary exposure that
    switches, a time-varying confounder (waist) and two baseline covariates."""
    rng = np.random.default_rng(seed)
    female = rng.binomial(1, 0.5, n)
    age = rng.uniform(30, 70, n).round(1)
    rows = []
    prev = np.zeros(n)
    for t in range(visits):
        waist = 92 + 6 * rng.normal(0, 1, n) - 2 * prev - 5 * female
        walk = rng.binomial(1, expit(-0.5 + 0.06 * (waist - 90) + 1.0 * prev))
        glucose = 95 + 0.4 * (waist - 90) - 2.0 * walk + 0.1 * (age - 50) + rng.normal(0, 6, n)
        for i in range(n):
            rows.append({"pid": i + 1, "wave": t, "walk": int(walk[i]),
                         "waist": round(float(waist[i]), 2), "female": int(female[i]),
                         "age": float(age[i]), "glucose": round(float(glucose[i]), 2)})
        prev = walk.astype(float)
    return pd.DataFrame(rows)


__all__ = ["DIET", "EVENT", "HTN", "HTN0", "RSCRIPT", "exact_risks", "feedback_cohort",
           "gformula_cohort", "needs_r", "r_dataset", "repeated_measure_cohort", "run_r"]
