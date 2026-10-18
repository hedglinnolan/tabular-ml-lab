"""E3 (SWARM_ENGINE_PLAN) where rows repeat within a unit: the ``mattered`` stage reads which rows
belong together on each specification's own rows, as the fit reads the rows of the plan that
specification is (``inference.resolve_clusters``). A specification whose rows repeat by `pid` is
CR2 by `pid`; one whose rows do not repeat is HC3, whatever the primary's rows do. Held to R:
``clubSandwich::coef_test`` (CR2 with Satterthwaite degrees of freedom) by `pid`, and
``sandwich::vcovHC`` (HC3) on t(n − p), each on the rows its definition gives.

Two constructions on ``test_mattered``'s cohort, its grain one row per unit by `pid`:

* **revisits outside the analysis:** 200 second visits of persons already in it, aged 80 to 85 so
  the age rule excludes them. The primary's rows do not repeat; with the age rule dropped they do.
* **revisits inside it:** 200 second visits within the age range with `bmi` not recorded. The
  primary's rows repeat; with `bmi` added (the covariate the answers leave out) the revisits leave
  and the rows stop repeating.

Each construction's departure is also fitted as a plan of its own through the fit stage, which
clusters it the same way. The generator is the author's; every expected number is R's.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from turbotab.core import decisions as d
from turbotab.core import mattered as M
from turbotab.core.tests.acceptance import estimand_fixtures as ef
from turbotab.core.tests.acceptance.r_reference import needs_r, run_r
from turbotab.core.tests.acceptance.test_mattered_survey import _r_has
from turbotab.core.tests.test_mattered import ORDER, RULES, Run, cohort, departures, state


def revisits(*, inside: bool, seed: int = 4) -> pd.DataFrame:
    frame = cohort()
    rng = np.random.default_rng(seed)
    pick = rng.choice(frame.index[frame["age"].between(30, 75)], 200, replace=False)
    again = frame.loc[pick].copy()
    if inside:
        again["bmi"] = np.nan
    else:
        again["age"] = 80.0 + rng.uniform(0, 5, len(again))
    again["glucose"] = frame.loc[pick, "glucose"].to_numpy() + rng.normal(0, 1, len(again))
    return pd.concat([frame, again], ignore_index=True)


def key(changed: str, pick: str) -> str:
    return "|".join(f"{c}={pick if c == changed else M.RECORDED}" for c in ORDER)


def _r_specs() -> str:
    lines = []
    for k, (rules, covariates) in departures().items():
        r = ", ".join(str(i) for i in rules)
        f = ", ".join(f'"{c}"' for c in covariates)
        lines.append(f'  "{k}" = list(r = c({r}), f = c({f}))')
    return "specs <- list(\n" + ",\n".join(lines) + ")\n"


SANDWICHES = """
suppressPackageStartupMessages({library(clubSandwich); library(sandwich)})
df <- read.csv(d_csv)
df$sex <- factor(df$sex)
keep <- function(s) {
  k <- rep(TRUE, nrow(df))
  if (0 %in% s$r) k <- k & df$age >= 30 & df$age <= 75
  if (1 %in% s$r) k <- k & (is.na(df$bmi) | df$bmi <= 33)
  k & complete.cases(df[, c("glucose", "fiber", s$f)])
}
res <- lapply(specs, function(s) {
  k <- keep(s)
  sub <- df[k, ]
  fit <- lm(as.formula(paste("glucose ~", paste(c("fiber", s$f), collapse = " + "))), data = sub)
  b <- coef(fit)[["fiber"]]
  ct <- coef_test(fit, vcov = "CR2", cluster = sub$pid, test = "Satterthwaite")
  i <- which(ct$Coef == "fiber")
  tc <- qt(0.975, ct$df_Satt[i])
  hc <- sqrt(vcovHC(fit, type = "HC3")["fiber", "fiber"])
  th <- qt(0.975, df.residual(fit))
  list(cr2 = c(b, b - tc * ct$SE[i], b + tc * ct$SE[i]), hc3 = c(b, b - th * hc, b + th * hc),
       n = sum(k), repeats = sum(duplicated(sub$pid)))
})
out(res)
"""


def _reference(run: Run, tmp_path) -> dict:
    if not _r_has("clubSandwich") or not _r_has("sandwich"):
        pytest.skip("R's clubSandwich or sandwich is not installed")
    return run_r(_r_specs() + SANDWICHES, {"d": run.table}, tmp_path / "r")


def _held_to_r(curve: dict, ref: dict) -> None:
    """Each specification is CR2 by `pid` where its own rows repeat, else HC3, as R gives it."""
    got = {s["key"]: s for s in curve["specs"]}
    assert set(got) == set(ref) == set(departures()) and curve["not_estimated"] == []
    for k, r in ref.items():
        covariance = "CR2" if r["repeats"] > 0 else "HC3"
        est, low, high = r["cr2"] if covariance == "CR2" else r["hc3"]
        s = got[k]
        assert s["covariance"] == covariance, k
        assert s["estimate"] == pytest.approx(est, rel=1e-9), k
        assert s["low"] == pytest.approx(low, rel=1e-8), k
        assert s["high"] == pytest.approx(high, rel=1e-8), k
        assert s["n"] == int(r["n"]), k
        if covariance == "CR2":  # the two intervals differ, so the test tells them apart
            assert abs(r["cr2"][1] - r["hc3"][1]) > 1e-3, k
    assert curve["clustered"] is any(r["repeats"] > 0 for r in ref.values())


def _linear(run: Run) -> dict:
    model = next(m for m in run.fit.data["models"] if m["family"] == "linear")
    return {"inference": model["inference"],
            "row": next(r for r in model["coefficients"] if r["feature"] == "fiber")}


@needs_r
def test_a_departure_whose_own_rows_repeat_is_cr2_by_the_unit_the_rest_hc3(tmp_path):
    run = Run(revisits(inside=False), tmp_path / "curve", state())
    art = run.mattered()
    assert art["refused"] is None, art["refused"]
    ref = _reference(run, tmp_path)
    dropped = key("exclusion:0", "dropped")
    assert {k for k, r in ref.items() if r["repeats"] > 0} == {dropped}  # the construction
    _held_to_r(art["curve"], ref)
    assert art["reported"]["covariance"] == "HC3"  # the primary's rows do not repeat
    assert "cluster-robust (CR2) intervals by that unit" in art["methods"]
    # The plan with the age rule dropped, through the fit stage: CR2 by `pid`, R's interval.
    plan = Run(revisits(inside=False), tmp_path / "plan",
               state(exclusions=[d.ExclusionRule(**RULES[1])]))
    fitted = _linear(plan)
    assert fitted["inference"]["covariance"] == "CR2"
    assert fitted["row"]["ci_low"] == pytest.approx(ref[dropped]["cr2"][1], rel=1e-8)
    assert fitted["row"]["ci_high"] == pytest.approx(ref[dropped]["cr2"][2], rel=1e-8)


@needs_r
def test_a_clustered_primary_s_departure_whose_rows_stop_repeating_is_hc3(tmp_path):
    run = Run(revisits(inside=True), tmp_path / "curve", state())
    art = run.mattered()
    assert art["refused"] is None, art["refused"]
    ref = _reference(run, tmp_path)
    added = key("left_out:bmi", "added")
    assert {k for k, r in ref.items() if r["repeats"] == 0} == {added}  # the construction
    _held_to_r(art["curve"], ref)
    primary = next(s for s in art["curve"]["specs"] if s["primary"])
    assert art["reported"]["covariance"] == "CR2"
    assert primary["estimate"] == pytest.approx(run.reported()["estimate"], rel=1e-12)
    # The plan adjusting for `bmi`, through the fit stage: its rows do not repeat, so HC3.
    plan = Run(revisits(inside=True), tmp_path / "plan",
               state(answers={**ef.ANSWERS, "bmi": ef.CONFOUNDER}))
    fitted = _linear(plan)
    assert fitted["inference"]["covariance"] == "HC3"
    assert fitted["row"]["ci_low"] == pytest.approx(ref[added]["hc3"][1], rel=1e-8)
    assert fitted["row"]["ci_high"] == pytest.approx(ref[added]["hc3"][2], rel=1e-8)
