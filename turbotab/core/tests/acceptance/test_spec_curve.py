"""T3: the specification curve as an engine method core (`turbotab.core.methods.spec_curve`).

Every specification's number is held to an independent reference: R's ``lm`` with sandwich's HC3
covariance on t(n − p); ``glm`` with Wald intervals; clubSandwich's CR2 with Satterthwaite degrees
of freedom; survey's ``svyglm`` on a subset design (the exclusion a domain). The enumeration, the
sampling rule beyond the cap and the summary counts are checked by hand. A package R does not find
skips its test (set ``R_LIBS`` to a library that holds it).
"""
from __future__ import annotations

import itertools
import math
import subprocess

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from turbotab.core.methods import spec_curve as S
from turbotab.core.tests.acceptance.r_reference import RSCRIPT, needs_r, run_r


def _r_has(package: str) -> bool:
    if RSCRIPT is None:
        return False
    done = subprocess.run([RSCRIPT, "-e", f'cat(requireNamespace("{package}", quietly = TRUE))'],
                          capture_output=True, text=True, timeout=120)
    return done.stdout.strip().endswith("TRUE")


def _skip_without(*packages: str) -> None:
    for package in packages:
        if not _r_has(package):
            pytest.skip(f"R's {package} is not installed (set R_LIBS to a library that has it)")


def _frame(seed: int = 20261009, n: int = 240) -> pd.DataFrame:
    """A numeric outcome, a yes/no one, an exposure, two numeric covariates, a three-level
    category, a person id (two to three rows each), a few missing covariate values, and a design."""
    rng = np.random.default_rng(seed)
    person = np.repeat(np.arange(1, 91), 3)[:n]
    effect = rng.normal(0, 0.6, 91)[person]
    x = rng.normal(5, 2, n)
    a = rng.normal(0, 1, n) + 0.3 * x
    b = rng.normal(0, 1, n)
    g = rng.choice(["north", "south", "west"], n)
    y = 1.0 + 0.4 * x + 0.5 * a - 0.3 * b + (g == "west") * 0.8 + effect + rng.normal(0, 1.5, n)
    e = (rng.random(n) < 1 / (1 + np.exp(-(-2 + 0.35 * x + 0.3 * a)))).astype(int)
    b[[3, 40, 77]] = np.nan
    stratum = (np.arange(n) % 6) + 1
    psu = ((np.arange(n) // 6) % 3) + 1
    w = rng.uniform(1, 6, n) * np.where(g == "west", 0.4, 1.0)
    return pd.DataFrame({"y": y, "e": e, "x": x, "x_alt": x + rng.normal(0, 0.1, n), "a": a,
                         "b": b, "g": g, "person": person, "stratum": stratum, "psu": psu, "w": w})


def _choices(frame: pd.DataFrame) -> list[S.Choice]:
    return [
        S.Choice("adjustment", "Adjusted for",
                 (S.Alternative("a", "a"), S.Alternative("ab", "a and b", add=("b",)),
                  S.Alternative("ag", "a and region", add=("g",)),
                  S.Alternative("none", "nothing", drop=("a",))), "a", decision="set_roles"),
        S.Choice("rows", "Rows analyzed",
                 (S.Alternative("all", "every row"),
                  S.Alternative("trim", "x within 1 to 9", keep=frame["x"].between(1, 9))), "all",
                 decision="set_exclusions"),
    ]


SPECS_R = """
specs <- list(
  "adjustment=a|rows=all" = list(f = c("a"), keep = rep(TRUE, nrow(df))),
  "adjustment=a|rows=trim" = list(f = c("a"), keep = df$x >= 1 & df$x <= 9),
  "adjustment=ab|rows=all" = list(f = c("a", "b"), keep = rep(TRUE, nrow(df))),
  "adjustment=ab|rows=trim" = list(f = c("a", "b"), keep = df$x >= 1 & df$x <= 9),
  "adjustment=ag|rows=all" = list(f = c("a", "g"), keep = rep(TRUE, nrow(df))),
  "adjustment=ag|rows=trim" = list(f = c("a", "g"), keep = df$x >= 1 & df$x <= 9),
  "adjustment=none|rows=all" = list(f = character(0), keep = rep(TRUE, nrow(df))),
  "adjustment=none|rows=trim" = list(f = character(0), keep = df$x >= 1 & df$x <= 9))
form <- function(out, s) as.formula(paste(out, "~", paste(c("x", s$f), collapse = " + ")))
rows <- function(s) { k <- s$keep & complete.cases(df[, c("y", "x", s$f), drop = FALSE]); k }
"""


def _by_key(curve: S.SpecCurve) -> dict[str, S.Spec]:
    return {s.key: s for s in curve.specs}


# ── the enumeration and its rule beyond the cap ──────────────────────────────


def test_every_combination_is_enumerated_with_the_primary_first():
    frame = _frame()
    picks, total, rule = S.enumerate_specs(_choices(frame))
    assert total == 8 and len(picks) == 8 and rule is None
    assert picks[0] == {"adjustment": "a", "rows": "all"}
    expected = {f"adjustment={a}|rows={r}" for a, r in itertools.product(
        ["a", "ab", "ag", "none"], ["all", "trim"])}
    assert {S._key(p, ["adjustment", "rows"]) for p in picks} == expected


def test_one_at_a_time_is_the_primary_and_each_single_departure():
    picks, total, rule = S.enumerate_specs(_choices(_frame()), crossing="one_at_a_time")
    assert total == 8 and rule is None
    assert picks == [{"adjustment": "a", "rows": "all"}, {"adjustment": "ab", "rows": "all"},
                     {"adjustment": "ag", "rows": "all"}, {"adjustment": "none", "rows": "all"},
                     {"adjustment": "a", "rows": "trim"}]


def _wide(n_choices: int = 6, n_options: int = 4) -> list[S.Choice]:
    return [S.Choice(f"c{i}", f"Choice {i}",
                     tuple(S.Alternative(f"o{j}", f"option {j}") for j in range(n_options)), "o0")
            for i in range(n_choices)]


def test_beyond_the_cap_the_rule_keeps_the_primary_every_departure_and_a_seeded_draw():
    choices = _wide()  # 4^6 = 4,096 combinations
    picks, total, rule = S.enumerate_specs(choices, cap=100, seed=7)
    order = [c.key for c in choices]
    keys = [S._key(p, order) for p in picks]
    assert total == 4096 and len(picks) == 100 and len(set(keys)) == 100
    assert picks[0] == {c.key: "o0" for c in choices}
    departures = [{**picks[0], c.key: f"o{j}"} for c in choices for j in range(1, 4)]
    assert picks[1:1 + 18] == departures  # 6 choices × 3 alternatives, in order
    assert "4,096 combinations exceed the 100 fitted" in rule and "81 combinations drawn" in rule
    again, _, _ = S.enumerate_specs(choices, cap=100, seed=7)
    other, _, _ = S.enumerate_specs(choices, cap=100, seed=8)
    assert again == picks and other[:19] == picks[:19] and other != picks
    # The draw is uniform over the combinations: each option's share of the drawn rest is about a
    # quarter (4,096 draws over many seeds, by hand: binomial sd √(n·¼·¾)).
    counts = np.zeros(4)
    for seed in range(40):
        drawn, _, _ = S.enumerate_specs(choices, cap=100, seed=seed)
        for p in drawn[19:]:
            counts[int(p["c0"][1])] += 1
    share = counts / counts.sum()
    assert np.all(np.abs(share - 0.25) < 4 * math.sqrt(0.25 * 0.75 / counts.sum()))


def test_departures_alone_beyond_the_cap_are_refused_with_an_exit():
    with pytest.raises(S.SpecCurveRefused, match="exceed the 10 fitted at most") as caught:
        S.enumerate_specs(_wide(), cap=10)
    assert caught.value.exits[0]["label"] == "Declare fewer alternatives"


# ── the lock and the goal ────────────────────────────────────────────────────


def test_before_the_lock_nothing_is_fitted_and_the_view_is_sealed():
    calls = []

    def fit(picks):
        calls.append(picks)
        raise AssertionError("fitted before the lock")

    curve = S.specification_curve(_choices(_frame()), fit, lock=None,
                                  estimate_label="Difference in y per unit of x")
    assert calls == [] and curve.specs == [] and curve.sealed == S.SEALED
    assert curve.exits[0]["stage"] == "fit"
    view = curve.to_view()
    assert view["specs"] == [] and view["sealed"] == S.SEALED and view["level"] is None
    assert [c["key"] for c in view["choices"]] == ["adjustment", "rows"]
    assert set(view) == {"estimateLabel", "choices", "specs", "zero", "level", "sealed"}


@pytest.mark.parametrize("goal", ["predict", "describe"])
def test_the_curve_is_offered_under_estimate_only(goal):
    with pytest.raises(S.SpecCurveRefused) as caught:
        S.specification_curve(_choices(_frame()), lambda p: None, lock="lock-1",
                              estimate_label="z", goal=goal)
    assert caught.value.exits
    assert S.eligible("estimate").offered


def test_a_choice_without_an_alternative_is_refused():
    lone = [S.Choice("adjustment", "Adjusted for", (S.Alternative("a", "a"),), "a")]
    with pytest.raises(S.SpecCurveRefused, match="no declared alternative"):
        S.specification_curve(lone, lambda p: None, lock="L", estimate_label="z")
    with pytest.raises(S.SpecCurveRefused, match="nothing to vary"):
        S.specification_curve([], lambda p: None, lock="L", estimate_label="z")


# ── each specification against R ─────────────────────────────────────────────


@needs_r
def test_least_squares_specifications_match_lm_with_hc3(tmp_path):
    _skip_without("sandwich")
    frame = _frame()
    choices = _choices(frame)
    fit = S.linear_fitter(frame, outcome="y", exposure="x", covariates=["a"], choices=choices,
                          task="regression")
    curve = S.specification_curve(choices, fit, lock="lock-1",
                                  estimate_label="Difference in y per unit of x",
                                  scale_key="difference:x")
    ref = run_r("""
library(sandwich)
df <- read.csv(d_csv)
""" + SPECS_R + """
res <- lapply(specs, function(s) {
  d <- df[rows(s), ]
  f <- lm(form("y", s), data = d)
  se <- sqrt(diag(vcovHC(f, type = "HC3")))["x"]
  t <- qt(0.975, df.residual(f))
  unname(c(coef(f)["x"], coef(f)["x"] - t * se, coef(f)["x"] + t * se, nrow(d)))
})
out(res)
""", {"d": frame}, tmp_path)
    got = _by_key(curve)
    assert set(got) == set(ref) and curve.plan_lock == "lock-1"
    for key, (est, lo, hi, n) in ref.items():
        s = got[key]
        assert s.estimate == pytest.approx(est, rel=1e-9)
        assert s.low == pytest.approx(lo, rel=1e-9) and s.high == pytest.approx(hi, rel=1e-9)
        assert s.n == n and s.covariance == "HC3"
    assert got["adjustment=a|rows=all"].primary
    assert sum(s.primary for s in curve.specs) == 1
    view = curve.to_view()
    assert len(view["specs"]) == 8 and view["level"] == 0.95 and view["sealed"] is None
    assert all(s["scaleKey"] == "difference:x" for s in view["specs"])
    assert set(view["specs"][0]) == {"key", "estimate", "low", "high", "n", "primary", "picks",
                                     "scaleKey"}


@needs_r
def test_logistic_specifications_match_glm_wald_intervals(tmp_path):
    frame = _frame()
    choices = _choices(frame)
    fit = S.linear_fitter(frame, outcome="e", exposure="x", covariates=["a"], choices=choices,
                          task="binary", classes=[0, 1])
    curve = S.specification_curve(choices, fit, lock="lock-1",
                                  estimate_label="Difference in log-odds of e per unit of x")
    ref = run_r("""
df <- read.csv(d_csv)
""" + SPECS_R.replace('c("y", "x", s$f)', 'c("e", "x", s$f)') + """
res <- lapply(specs, function(s) {
  d <- df[rows(s), ]
  f <- glm(form("e", s), data = d, family = binomial)
  ci <- confint.default(f)["x", ]
  unname(c(coef(f)["x"], ci, nrow(d)))
})
out(res)
""", {"d": frame}, tmp_path)
    got = _by_key(curve)
    for key, (est, lo, hi, n) in ref.items():
        s = got[key]
        assert s.estimate == pytest.approx(est, rel=1e-6)
        assert s.low == pytest.approx(lo, rel=1e-6) and s.high == pytest.approx(hi, rel=1e-6)
        assert s.n == n


@needs_r
def test_repeated_rows_give_cr2_intervals_as_clubsandwich_computes_them(tmp_path):
    _skip_without("clubSandwich")
    frame = _frame()
    choices = _choices(frame)
    fit = S.linear_fitter(frame, outcome="y", exposure="x", covariates=["a"], choices=choices,
                          task="regression", units=frame["person"], unit_name="person")
    curve = S.specification_curve(choices, fit, lock="lock-1", estimate_label="z", clustered=True)
    ref = run_r("""
suppressPackageStartupMessages(library(clubSandwich))
df <- read.csv(d_csv)
""" + SPECS_R + """
res <- lapply(specs, function(s) {
  d <- df[rows(s), ]
  f <- lm(form("y", s), data = d)
  ct <- coef_test(f, vcov = "CR2", cluster = d$person, test = "Satterthwaite")
  i <- which(ct$Coef == "x")
  unname(c(ct$beta[i], ct$SE[i], ct$df_Satt[i]))
})
out(res)
""", {"d": frame}, tmp_path)
    got = _by_key(curve)
    for key, (est, se, dof) in ref.items():
        s = got[key]
        t = stats.t.ppf(0.975, dof)
        assert s.covariance == "CR2"
        assert s.estimate == pytest.approx(est, rel=1e-9)
        assert s.se == pytest.approx(se, rel=1e-7) and s.df == pytest.approx(dof, rel=1e-6)
        assert s.low == pytest.approx(est - t * se, rel=1e-6)


@needs_r
def test_under_a_design_every_specification_is_svyglm_and_an_exclusion_is_a_domain(tmp_path):
    _skip_without("survey")
    from turbotab.core.models.survey import build_design

    frame = _frame()
    design = build_design(frame, frame["w"].to_numpy(), weight_column="w",
                          strata_column="stratum", psu_column="psu")
    choices = _choices(frame)
    fit = S.linear_fitter(frame, outcome="y", exposure="x", covariates=["a"], choices=choices,
                          task="regression", design=design)
    curve = S.specification_curve(choices, fit, lock="lock-1", estimate_label="z", weighted=True)
    ref = run_r("""
suppressPackageStartupMessages(library(survey))
options(survey.lonely.psu = "adjust")
df <- read.csv(d_csv)
des <- svydesign(ids = ~psu, strata = ~stratum, weights = ~w, nest = TRUE, data = df)
""" + SPECS_R + """
res <- lapply(specs, function(s) {
  sub <- subset(des, rows(s))
  f <- svyglm(form("y", s), sub)
  t <- qt(0.975, degf(sub))
  unname(c(coef(f)["x"], coef(f)["x"] - t * SE(f)["x"], coef(f)["x"] + t * SE(f)["x"],
           sum(rows(s))))
})
out(res)
""", {"d": frame}, tmp_path)
    got = _by_key(curve)
    for key, (est, lo, hi, n) in ref.items():
        s = got[key]
        assert s.covariance == "design"
        assert s.estimate == pytest.approx(est, rel=1e-9)
        assert s.low == pytest.approx(lo, rel=1e-8) and s.high == pytest.approx(hi, rel=1e-8)
        assert s.n == n
    assert "design-based" in S.spec_curve_sentence(curve)


# ── the summary, the scale, a specification with no estimate ─────────────────


def test_the_summary_counts_are_the_hand_counts():
    table = {"p": (0.5, 0.1, 0.9), "q": (-0.2, -0.5, -0.1), "r": (0.1, -0.1, 0.3), "s": (0.9, 0.7, 1.1)}
    choices = [S.Choice("c", "C", tuple(S.Alternative(k, k) for k in table), "r")]

    def fit(picks):
        est, lo, hi = table[picks["c"]]
        return S.SpecFit(est, lo, hi, 10)

    curve = S.specification_curve(choices, fit, lock="L", estimate_label="z")
    s = curve.summary
    assert (s.k, s.median, s.lowest, s.highest) == (4, 0.3, -0.2, 0.9)
    assert (s.above, s.below, s.spanning, s.primary_rank) == (2, 1, 1, 2)
    assert "median 0.3" in curve.says and "the primary, 0.1, is the reported estimate" in curve.says


def test_a_specification_with_no_estimate_is_listed_and_the_primary_must_have_one():
    choices = [S.Choice("c", "C", (S.Alternative("p", "p"), S.Alternative("q", "q")), "p")]
    curve = S.specification_curve(
        choices, lambda picks: S.SpecFit(1.0, 0.5, 1.5, 9) if picks["c"] == "p" else
        S.SpecFit(math.nan, None, None, 3, refused="only 3 rows"), lock="L", estimate_label="z")
    assert [s.key for s in curve.specs] == ["c=p"]
    assert curve.not_estimated[0].reason == "only 3 rows"
    with pytest.raises(S.SpecCurveRefused, match="primary specification has no estimate"):
        S.specification_curve(choices, lambda picks: S.SpecFit(math.nan, None, None, 3,
                                                               refused="none"),
                              lock="L", estimate_label="z")


def test_an_exposure_on_another_scale_and_a_category_exposure_are_refused():
    frame = _frame()
    other = [S.Choice("exposure", "Exposure measured as",
                      (S.Alternative("x", "x"), S.Alternative("alt", "x per 10", exposure="x_alt",
                                                              scale="per 10 units")), "x")]
    with pytest.raises(S.SpecCurveRefused, match="another scale") as caught:
        S.linear_fitter(frame, outcome="y", exposure="x", covariates=["a"], choices=other,
                        task="regression", exposure_scale="per unit")
    assert caught.value.exits
    same = [S.Choice("exposure", "Exposure measured as",
                     (S.Alternative("x", "x"), S.Alternative("alt", "x, second reading",
                                                             exposure="x_alt", scale="per unit")),
                     "x")]
    fit = S.linear_fitter(frame, outcome="y", exposure="x", covariates=["a"], choices=same,
                          task="regression", exposure_scale="per unit")
    curve = S.specification_curve(same, fit, lock="L", estimate_label="z")
    assert len(curve.specs) == 2
    fit = S.linear_fitter(frame, outcome="y", exposure="g", covariates=["a"], choices=_choices(frame),
                          task="regression")
    with pytest.raises(S.SpecCurveRefused, match="is a category"):
        fit({"adjustment": "a", "rows": "all"})


def test_the_contract_is_registered_with_its_sentence_and_relations():
    from turbotab.core.contracts import contract

    c = contract("specification_curve")
    assert c.slot == "evaluation" and c.scope == "model"
    assert [o["key"] for o in c.options_for("inference")] == ["all", "one_at_a_time"]
    assert {o["rung"] for o in c.options_for("prediction")} == {"not_offered"}
    for rid in ("after_the_lock", "primary_reported", "sampling_rule", "design_based_specs",
                "descriptive_only", "predict_refused", "mixed_scales"):
        assert c.relation(rid).enforced_by.startswith("turbotab.core.methods.spec_curve:")
    assert "Simonsohn, Simmons & Nelson 2020" in S.spec_curve_sentence()
    assert "No joint test" in S.spec_curve_sentence()


# ── what the probes found: scales, the lock, a dropped exposure, the domain's n ─


@pytest.mark.parametrize("alt_scale,primary_scale", [(None, ""), ("per year", ""), (None, "per SD")])
def test_an_exposure_swap_with_an_undeclared_scale_is_refused(alt_scale, primary_scale):
    frame = _frame()
    swap = [S.Choice("exposure", "Exposure",
                     (S.Alternative("x", "x"), S.Alternative("age", "a instead", exposure="a",
                                                             scale=alt_scale)), "x")]
    with pytest.raises(S.SpecCurveRefused, match="not declared") as caught:
        S.linear_fitter(frame, outcome="y", exposure="x", covariates=["b"], choices=swap,
                        task="regression", exposure_scale=primary_scale)
    assert caught.value.exits


@pytest.mark.parametrize("lock", ["", "   ", False, 0, object(), True])
def test_a_placeholder_lock_keeps_the_curve_sealed(lock):
    called = []
    curve = S.specification_curve(_choices(_frame()), lambda p: called.append(p), lock=lock,
                                  estimate_label="z")
    assert curve.sealed == S.SEALED and not called and curve.to_view()["specs"] == []


def test_a_lock_plan_record_opens_the_curve_and_another_record_does_not():
    from types import SimpleNamespace as NS

    choices = [S.Choice("c", "C", (S.Alternative("p", "p"), S.Alternative("q", "q")), "p")]
    fit = lambda picks: S.SpecFit(1.0, 0.5, 1.5, 9)  # noqa: E731
    lock = NS(id="rec-7", decision=NS(kind="lock_plan"))
    opened = S.specification_curve(choices, fit, lock=lock, estimate_label="z")
    assert opened.sealed is None and opened.plan_lock == "rec-7" and len(opened.specs) == 2
    other = NS(id="rec-8", decision=NS(kind="set_roles"))
    assert S.specification_curve(choices, fit, lock=other, estimate_label="z").sealed == S.SEALED


@pytest.mark.parametrize("drop", [("x",), ("x_alt",)])
def test_an_option_that_drops_the_exposure_is_refused(drop):
    frame = _frame()
    exposure = "x" if drop == ("x",) else "x_alt"
    choices = [S.Choice("adjustment", "Adjusted for",
                        (S.Alternative("base", "a"),
                         S.Alternative("dropz", "without the exposure", drop=drop,
                                       exposure=None if exposure == "x" else "x_alt",
                                       scale=None if exposure == "x" else "per unit")), "base")]
    with pytest.raises(S.SpecCurveRefused, match="drops") as caught:
        S.linear_fitter(frame, outcome="y", exposure="x", covariates=["a"], choices=choices,
                        task="regression", exposure_scale="per unit")
    assert caught.value.exits


def test_under_a_design_a_specification_s_n_is_its_domain():
    from turbotab.core.models.survey import build_design

    frame = _frame()
    w = frame["w"].to_numpy().copy()
    w[:30] = 0.0
    w[30:35] = np.nan
    design = build_design(frame, w, weight_column="w", strata_column="stratum", psu_column="psu")
    choices = _choices(frame)
    fit = S.linear_fitter(frame, outcome="y", exposure="x", covariates=["a"], choices=choices,
                          task="regression", design=design)
    curve = S.specification_curve(choices, fit, lock="lock-1", estimate_label="z", weighted=True)
    weighted = np.isfinite(w) & (w > 0)
    for s in curve.specs:
        cols = {"a": ["a"], "ab": ["a", "b"], "ag": ["a", "g"], "none": []}[s.picks["adjustment"]]
        rows = frame[["y", "x", *cols]].notna().all(axis=1).to_numpy() & weighted
        if s.picks["rows"] == "trim":
            rows &= frame["x"].between(1, 9).to_numpy()
        assert s.n == int(rows.sum())


def test_a_primary_with_no_estimate_is_refused_with_exits():
    choices = [S.Choice("c", "C", (S.Alternative("p", "p"), S.Alternative("q", "q")), "p")]
    with pytest.raises(S.SpecCurveRefused) as caught:
        S.specification_curve(choices, lambda picks: S.SpecFit(math.nan, None, None, 3,
                                                               refused="none"),
                              lock="L", estimate_label="z")
    assert caught.value.exits
