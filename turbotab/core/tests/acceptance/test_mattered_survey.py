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
from turbotab.core.tests.test_mattered import ADJUSTED, RULES, Run, departures, rows_of

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


SVY_UNITS = """
suppressPackageStartupMessages(library(survey))
options(survey.lonely.psu = "adjust")
df <- read.csv(d_csv)
by_row <- svydesign(ids = ~1, strata = ~strata, weights = ~wt, data = df)
by_pid <- svydesign(ids = ~pid, strata = ~strata, weights = ~wt, nest = TRUE, data = df)
keep <- function(s) {
  k <- rep(TRUE, nrow(df))
  if (0 %in% s$r) k <- k & df$age >= 30 & df$age <= 75
  if (1 %in% s$r) k <- k & (is.na(df$bmi) | df$bmi <= 33)
  k & complete.cases(df[, c("glucose", "fiber", s$f)])
}
fit_in <- function(des, s, k) {
  sub <- subset(des, k)
  fit <- svyglm(as.formula(paste("glucose ~", paste(c("fiber", s$f), collapse = " + "))),
                design = sub)
  t <- qt(0.975, degf(sub))
  b <- coef(fit)[["fiber"]]
  se <- SE(fit)[["fiber"]]
  c(b, b - t * se, b + t * se)
}
res <- lapply(specs, function(s) {
  k <- keep(s)
  list(by_row = fit_in(by_row, s, k), by_pid = fit_in(by_pid, s, k), n = sum(k),
       repeats = sum(duplicated(df$pid[k])))
})
out(res)
"""


@needs_r
def test_without_a_psu_a_departure_whose_rows_repeat_takes_the_unit_as_its_sampling_unit(
        tmp_path):
    """With no PSU column the design takes the unit the rows repeat in as its sampling unit
    (``methods.survey.for_fit``), the unit read on each specification's own rows as the fit reads
    its plan's. 150 second visits of persons in the table, aged 80 to 85 so the age rule excludes
    them: the primary's rows do not repeat, so each row is its own sampling unit (``ids = ~1``);
    with the age rule dropped they repeat, so each `pid` is one (``ids = ~pid``). Each
    specification is svyglm on its domain of that design over every row; the two designs give the
    departure different intervals, so the test tells them apart."""
    if not _r_has("survey"):
        pytest.skip("R's survey is not installed (set R_LIBS to a library that has it)")
    frame = surveyed().drop(columns=["psu"])
    rng = np.random.default_rng(8)
    pick = rng.choice(frame.index[frame["age"].between(30, 75)], 150, replace=False)
    again = frame.loc[pick].copy()
    again["age"] = 80.0 + rng.uniform(0, 5, len(again))
    again["glucose"] = frame.loc[pick, "glucose"].to_numpy() + rng.normal(0, 1, len(again))
    frame = pd.concat([frame, again], ignore_index=True)
    roles = {c: r for c, r in ROLES.items() if c != "psu"}
    st = ef.state(target="glucose", task="regression", measure="mean_difference", roles=roles,
                  amounts={k: v for k, v in SHAPES.items() if k != "code_or_count:psu"},
                  plan_locked=True, exclusions=[d.ExclusionRule(**r) for r in RULES],
                  survey=d.SurveySpec(estimand="population", weight="wt", strata="strata"))
    run = Run(frame, tmp_path / "stages", st)
    art = run.mattered()
    assert art["refused"] is None, art["refused"]
    ref = run_r(_r_specs() + SVY_UNITS, {"d": run.table}, tmp_path / "r")
    dropped = "|".join(f"{c}={'dropped' if c == 'exclusion:0' else 'recorded'}"
                       for c in [*(f"exclusion:{i}" for i in range(2)),
                                 *(f"adjusted:{c}" for c in ADJUSTED), "left_out:bmi"])
    assert {k for k, r in ref.items() if r["repeats"] > 0} == {dropped}  # the construction
    assert abs(ref[dropped]["by_pid"][1] - ref[dropped]["by_row"][1]) > 1e-3
    got = {s["key"]: s for s in art["curve"]["specs"]}
    assert set(got) == set(ref) and art["curve"]["not_estimated"] == []
    for k, r in ref.items():
        est, low, high = r["by_pid"] if r["repeats"] > 0 else r["by_row"]
        s = got[k]
        assert s["covariance"] == "design", k
        assert s["estimate"] == pytest.approx(est, rel=1e-9), k
        assert s["low"] == pytest.approx(low, rel=1e-8), k
        assert s["high"] == pytest.approx(high, rel=1e-8), k
        assert s["n"] == int(r["n"]), k


def population(**slots) -> d.ProjectState:
    """The surveyed population over a design with a PSU column, the two exclusion rules recorded."""
    base = dict(target="glucose", task="regression", measure="mean_difference", roles=ROLES,
                amounts=SHAPES, plan_locked=True, exclusions=[d.ExclusionRule(**r) for r in RULES],
                survey=d.SurveySpec(estimand="population", weight="wt", strata="strata",
                                    psu="psu"))
    return ef.state(**{**base, **slots})


def revisited(*, own_psu: bool) -> pd.DataFrame:
    """``surveyed()`` with the test above's 150 second visits of persons aged 30 to 75, aged 80 to
    85 so the age rule excludes them, each in its person's stratum: in its person's own PSU, or in
    another PSU of that stratum (the next one; stratum 1 has three)."""
    frame = surveyed()
    rng = np.random.default_rng(8)
    pick = rng.choice(frame.index[frame["age"].between(30, 75)], 150, replace=False)
    again = frame.loc[pick].copy()
    again["age"] = 80.0 + rng.uniform(0, 5, len(again))
    again["glucose"] = frame.loc[pick, "glucose"].to_numpy() + rng.normal(0, 1, len(again))
    if not own_psu:
        psus = frame.groupby("strata")["psu"].max()
        again["psu"] = [p % psus[h] + 1 for h, p in zip(again["strata"], again["psu"])]
    return pd.concat([frame, again], ignore_index=True)


# The departure with the age rule dropped: the one specification whose rows hold the second visits.
AGE_DROPPED = next(k for k, (rules, _covariates) in departures().items() if rules == (1,))


def assert_svyglm(got: dict, ref: dict) -> None:
    """Each specification in ``ref`` is R's design-based fit there (``SVY``)."""
    for key, (est, low, high, n) in ref.items():
        s = got[key]
        assert s["covariance"] == "design", key
        assert s["estimate"] == pytest.approx(est, rel=1e-9), key
        assert s["low"] == pytest.approx(low, rel=1e-8), key
        assert s["high"] == pytest.approx(high, rel=1e-8), key
        assert s["n"] == int(n), key


@needs_r
def test_a_departure_whose_persons_cross_psus_is_not_estimated_with_the_designs_reason(tmp_path):
    """A specification whose survey design refuses is not estimated, with the design's reason
    (``mattered.curve_for``'s ``fitter_for``): here, where it should. Each second visit sits in
    another PSU of its person's stratum, so with the age rule dropped the rows repeat by `pid` and
    no PSU holds a person's rows together (``methods.survey.for_fit``, over every row: 150 persons,
    by construction). That departure is listed with that reason, which is the fit stage's own
    refusal of the plan the departure is (the age rule dropped). Every other specification's rows
    do not repeat, and each is R's svyglm (``ids = ~psu``) on its domain."""
    from pathlib import Path

    from turbotab.core.datastore import DataStore
    from turbotab.core.methods.survey import for_fit

    if not _r_has("survey"):
        pytest.skip("R's survey is not installed (set R_LIBS to a library that has it)")
    frame = revisited(own_psu=False)
    run = Run(frame, tmp_path / "stages", population())
    art = run.mattered()
    assert art["refused"] is None, art["refused"]
    with DataStore(Path(run.paths["data"]), 1 << 30) as store:
        why = for_fit(run.state, store, "pid").refusal
    assert why is not None and why.startswith("150 `pid` units have rows in more than one PSU")
    [listed] = art["curve"]["not_estimated"]
    assert listed["key"] == AGE_DROPPED and listed["primary"] is False
    assert listed["reason"] == "With this choice changed, the fit would report no interval: " + why
    plan = Run(frame, tmp_path / "plan", population(exclusions=[d.ExclusionRule(**RULES[1])]))
    linear = next(m for m in plan.fit.data["models"] if m["family"] == "linear")
    assert linear["inference"]["refused"] == why
    ref = run_r(_r_specs() + SVY, {"d": run.table}, tmp_path / "r")
    got = {s["key"]: s for s in art["curve"]["specs"]}
    assert set(got) == set(ref) - {AGE_DROPPED}
    assert_svyglm(got, {k: v for k, v in ref.items() if k != AGE_DROPPED})


@needs_r
def test_a_departure_whose_persons_keep_their_psu_is_svyglm_on_its_domain(tmp_path):
    """The same guard silent where it should be: each second visit in its person's own PSU, so the
    PSU holds a person's rows together and the departure with the age rule dropped, whose rows
    repeat by `pid`, is estimated as every specification is, R's svyglm (``ids = ~psu``) on its
    domain of the one design over every row."""
    if not _r_has("survey"):
        pytest.skip("R's survey is not installed (set R_LIBS to a library that has it)")
    run = Run(revisited(own_psu=True), tmp_path / "stages", population())
    art = run.mattered()
    assert art["refused"] is None, art["refused"]
    assert art["curve"]["not_estimated"] == []
    keep = rows_of(run.table, rules=(1,), columns=["glucose", "fiber", *ADJUSTED])
    assert run.table.loc[keep, "pid"].duplicated().sum() > 0  # the construction: its rows repeat
    ref = run_r(_r_specs() + SVY, {"d": run.table}, tmp_path / "r")
    got = {s["key"]: s for s in art["curve"]["specs"]}
    assert set(got) == set(ref) == set(departures())
    assert_svyglm(got, ref)
