"""E3 (SWARM_ENGINE_PLAN) under a survey design: the ``mattered`` stage's specification curve over
the surveyed population, every specification design-based and each exclusion rule a domain of the
whole design, held to R's ``survey`` package (``svyglm`` on ``subset`` of one ``svydesign`` over
every row, t on its ``degf``), as ``methods/spec_curve.py``'s own design test is.

The fixture is NHANES-shaped: twelve strata of two PSUs (one of three), a PSU effect, an
oversampled group with about a fifth of the others' weights and a weaker fiber slope, so the
population's estimate is not the sample's. Blanks in ``activity`` and ``bmi`` and the two exclusion
rules of ``turbotab/core/tests/test_mattered.py`` make each specification's domain its own. The
generator's structure is the author's; every expected number is R's.
"""
from __future__ import annotations

import subprocess

import numpy as np
import pandas as pd
import pytest

from turbotab.core import decisions as d
from turbotab.core.tests.acceptance import estimand_fixtures as ef
from turbotab.core.tests.acceptance.r_reference import RSCRIPT, needs_r, run_r
from turbotab.core.tests.test_mattered import ADJUSTED, RULES, Run, departures

ROLES = {"pid": "identifier", "strata": "design", "psu": "design", "wt": "design",
         **{c: "covariate" for c in (*ADJUSTED, "bmi")}, "fiber": "exposure"}
SHAPES = {**ef.AMOUNTS, "code_or_count:strata": "code", "code_or_count:psu": "code"}


def _r_has(package: str) -> bool:
    if RSCRIPT is None:
        return False
    done = subprocess.run([RSCRIPT, "-e", f'cat(requireNamespace("{package}", quietly = TRUE))'],
                          capture_output=True, text=True, timeout=120)
    return done.stdout.strip().endswith("TRUE")


def surveyed(seed: int = 5, strata: int = 12, per_psu: int = 34) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for h in range(1, strata + 1):
        for j in range(1, (3 if h == 1 else 2) + 1):
            effect = rng.normal(0, 3)
            for _ in range(per_psu):
                over = bool(rng.random() < 0.4)
                age = float(rng.normal(50, 12))
                sex = str(rng.choice(["female", "male"]))
                smoking = float(rng.random() < 0.3)
                activity = float(rng.integers(0, 8))
                fiber = float(20 - 3 * smoking + 0.05 * (age - 50) + (sex == "male")
                              - 2 * over + rng.normal(0, 4))
                bmi = float(27 - 0.1 * (fiber - 20) + rng.normal(0, 3))
                slope = -0.2 if over else -0.5
                glucose = float(100 + slope * (fiber - 20) + 6 * smoking + 0.3 * (age - 50)
                                + 0.5 * (bmi - 27) - 1.2 * activity + 2 * (sex == "male")
                                + effect + rng.normal(0, 8))
                wt = float(rng.uniform(800, 3000) if over else rng.uniform(5000, 20000))
                rows.append({"pid": len(rows), "strata": h, "psu": j, "wt": wt, "age": age,
                             "sex": sex, "smoking": smoking, "activity": activity, "bmi": bmi,
                             "fiber": fiber, "glucose": glucose})
    frame = pd.DataFrame(rows)
    frame.loc[rng.choice(len(frame), 25, replace=False), "activity"] = np.nan
    frame.loc[rng.choice(len(frame), 30, replace=False), "bmi"] = np.nan
    return frame


def _r_specs() -> str:
    lines = []
    for key, (rules, covariates) in departures().items():
        r = ", ".join(str(i) for i in rules)
        f = ", ".join(f'"{c}"' for c in covariates)
        lines.append(f'  "{key}" = list(r = c({r}), f = c({f}))')
    return "specs <- list(\n" + ",\n".join(lines) + ")\n"


SVY = """
suppressPackageStartupMessages(library(survey))
options(survey.lonely.psu = "adjust")
df <- read.csv(d_csv)
des <- svydesign(ids = ~psu, strata = ~strata, weights = ~wt, nest = TRUE, data = df)
keep <- function(s) {
  k <- rep(TRUE, nrow(df))
  if (0 %in% s$r) k <- k & df$age >= 30 & df$age <= 75
  if (1 %in% s$r) k <- k & (is.na(df$bmi) | df$bmi <= 33)
  k & complete.cases(df[, c("glucose", "fiber", s$f)])
}
res <- lapply(specs, function(s) {
  k <- keep(s)
  sub <- subset(des, k)
  fit <- svyglm(as.formula(paste("glucose ~", paste(c("fiber", s$f), collapse = " + "))),
                design = sub)
  t <- qt(0.975, degf(sub))
  b <- coef(fit)[["fiber"]]
  se <- SE(fit)[["fiber"]]
  c(b, b - t * se, b + t * se, sum(k))
})
out(res)
"""


@needs_r
def test_under_the_surveyed_population_each_specification_is_svyglm_on_its_domain(tmp_path):
    if not _r_has("survey"):
        pytest.skip("R's survey is not installed (set R_LIBS to a library that has it)")
    st = ef.state(target="glucose", task="regression", measure="mean_difference", roles=ROLES,
                  amounts=SHAPES, plan_locked=True,
                  exclusions=[d.ExclusionRule(**r) for r in RULES],
                  survey=d.SurveySpec(estimand="population", weight="wt", strata="strata",
                                      psu="psu"))
    run = Run(surveyed(), tmp_path / "stages", st)
    art = run.mattered()
    assert art["refused"] is None, art["refused"]
    curve = art["curve"]
    assert curve["weighted"] is True and curve["clustered"] is False
    ref = run_r(_r_specs() + SVY, {"d": run.table}, tmp_path / "r")
    got = {s["key"]: s for s in curve["specs"]}
    assert set(got) == set(ref) == set(departures())
    for key, (est, low, high, n) in ref.items():
        s = got[key]
        assert s["covariance"] == "design", key
        assert s["estimate"] == pytest.approx(est, rel=1e-9), key
        assert s["low"] == pytest.approx(low, rel=1e-8) and s["high"] == pytest.approx(high,
                                                                                        rel=1e-8)
        assert s["n"] == int(n), key
    # The primary is the reported estimate, the fit's own design-based row, which is R's too.
    primary = next(s for s in curve["specs"] if s["primary"])
    reported = run.reported()
    assert art["reported"]["covariance"] == "design"
    assert primary["estimate"] == pytest.approx(reported["estimate"], rel=1e-12)
    assert reported["estimate"] == pytest.approx(ref[primary["key"]][0], rel=1e-9)
    assert "design-based" in art["methods"] and "domain of the full design" in art["methods"]


def test_the_population_s_curve_is_not_the_participants(tmp_path):
    """The same table answered as these participants: no design, every specification model-based
    (HC3), and its primary another number than the population's (the oversampled group's weaker
    slope counts for less in the population). The answer is read both ways."""
    common = dict(target="glucose", task="regression", measure="mean_difference", roles=ROLES,
                  amounts=SHAPES, plan_locked=True,
                  exclusions=[d.ExclusionRule(**r) for r in RULES])
    population = Run(surveyed(), tmp_path / "population", ef.state(
        **common, survey=d.SurveySpec(estimand="population", weight="wt", strata="strata",
                                      psu="psu"))).mattered()
    sample = Run(surveyed(), tmp_path / "sample",
                 ef.state(**common, survey=d.SurveySpec(estimand="sample"))).mattered()
    assert sample["refused"] is None and sample["curve"]["weighted"] is False
    assert {s["covariance"] for s in sample["curve"]["specs"]} == {"HC3"}
    p = next(s for s in population["curve"]["specs"] if s["primary"])
    q = next(s for s in sample["curve"]["specs"] if s["primary"])
    assert p["covariance"] == "design" and q["n"] == p["n"]
    assert p["estimate"] != pytest.approx(q["estimate"], rel=1e-3)
