"""Wave 1b's seams: what holds only once MI (MS1–MS3) and VALID (MS6) run on wave 1a's engine.

Each package's own acceptance file tests its methods against its references. This file tests the
places where they meet wave 1a, each a rule one package wrote that the other's method must obey:

* **One registry** (BLUEPRINT §13). MI, VALID and wave 1a's SURVEY each wrote a contract shape of
  their own; every method now enters through ``turbotab.core.contracts`` with one vocabulary for
  slots, scopes and rungs, one run order that every *precedes* relation agrees with, and a data
  scope the lockbox test (constitution §06) observes by perturbation. MODELING_SEQUENCE §1.1's
  inference order holds in it: the copies are drawn first, then each copy's scale scores, then
  each copy's design-based fit.
* **Two times on the follow-up's scale** (the routing gate and MS6). The routing gate's repair
  (landed beside wave 1b) made ``horizon`` the end of follow-up: events after it are censored
  there, so the outcome changes. MS6 had named the time predicted risks are scored and calibrated
  at ``horizon`` too, which changes no row. They are two answers: MS6's is
  ``prediction_horizon``, and it lies after the landmark and before the end of follow-up, where
  Graf's censoring weights are still positive.
* **The population answer binds the pooled curve** (MS2–MS4). Under the surveyed population every
  display is design-based or blocked and recorded (MODELING_SEQUENCE §4), and under multiple
  imputation every display is pooled (§2) with ν_com the design df (§1.1). The substitution curve
  meets both: each copy's curve is the surveyed population's, its variance the design's, pooled by
  Rubin's rules on the design's degrees of freedom; the row bootstrap, which ignores the strata and
  PSUs, is not run; a family with no design-based estimator stays blocked with the sample-only exit.

References: R ``survey`` (``svyglm``, ``svycontrast``, ``svyrecvar``) for each copy's curve and its
variance, and ``mice::pool.scalar`` for Rubin's rules with the design's complete-data df, on CSVs
of the app's own imputed copies (the copies themselves are MI's acceptance file's to check against
R ``smcfcs``). The rows on support at each k are the app's definition
(``methods.substitution.Shift``), handed to R as columns, as MS4's acceptance file does.
"""
from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from turbotab.core import contracts as C
from turbotab.core.methods.missing import MI_CONTRACTS
from turbotab.core.models.survey import SURVEY_CONTRACTS
from turbotab.core.models.validation import VALIDATION_CONTRACTS
from turbotab.core.tests.acceptance.survey_r import needs_r, nhanes_diet, run_r
from turbotab.core.tests.acceptance.test_ms4_survey_across_families import (
    DIET_ROLES, assert_design_based_or_blocked, survey_stages)

WAVE1B = (*MI_CONTRACTS, *VALIDATION_CONTRACTS, *SURVEY_CONTRACTS)
DECLARING = ("turbotab.core.methods.missing", "turbotab.core.models.validation",
             "turbotab.core.models.survey")


# ── one registry ─────────────────────────────────────────────────────────────


def test_every_wave1b_method_enters_through_the_one_registry():
    """BLUEPRINT §13 in one registry with one vocabulary: slot, data scope, needs, routing (its
    question; each option labeled customary and sound for each purpose, with a rung for each),
    storyboard, sentence and relations. No package keeps a contract shape of its own; a conflict is
    refused or blocked and recorded, with its ways forward; every relation names code that exists;
    and the run order agrees with every *precedes* relation and with MODELING_SEQUENCE §1.1 under
    inference: the imputed copies first, each copy's scale scored from its imputed items, then each
    copy's design-based fit, then the evaluation."""
    registry = C.contracts()
    assert set(WAVE1B) <= set(registry)
    assert set(DECLARING) <= set(C.DECLARING_MODULES)
    for name in DECLARING:
        module = importlib.import_module(name)
        for shape in ("MethodContract", "Relation", "register_contract", "CONTRACTS"):
            assert not hasattr(module, shape), (name, shape)
    for key in WAVE1B:
        c = registry[key]
        assert c.slot in C.SLOTS and c.scope in C.SCOPES, key
        assert c.needs and c.question and c.storyboard and c.options and c.relations, key
        assert c.sentence and c.sources, key
        for o in c.options:
            assert o.label and o.customary, (key, o.key)
            for purpose in C.PURPOSES:
                assert o.sound[purpose] and o.rung[purpose] in C.RUNGS, (key, o.key, purpose)
                assert o.order[purpose] >= 0
        for r in c.relations:
            if r.kind == "conflicts":
                assert r.rung in ("refused", "block_and_record") and r.exits, (key, r.target)
            if r.enforced_by:
                module, name = r.enforced_by.split(":")
                assert callable(getattr(importlib.import_module(module), name)), r.enforced_by
        if c.slot in C.BEFORE_THE_SEAL:
            assert c.scope in C.PRE_SEAL_SCOPES, key
    order = C.run_order(list(registry))  # raises on a precedes relation the order breaks
    assert order.index("multiple_imputation_compatible") < order.index("scales")
    assert order.index("scales") < order.index("survey_linear") < order.index("survey_substitution")
    assert order.index("survey_linear") < order.index("proper_primary") < order.index("bbc_cv")
    fired = {f.relation.name for f in C.fired({"multiple_imputation_compatible": "compatible",
                                               "scales": None}, "inference")}
    assert "mi.items_before_the_score" in fired
    assert not C.fired({"multiple_imputation_compatible": "compatible", "scales": None},
                       "prediction")


def test_multiple_imputation_declares_the_scope_the_perturbation_test_observes():
    """Lockbox constitution §06's test, run on MI's own code: shuffling the outcome moves a blank
    row's imputed value, so compatible multiple imputation learns from the outcome (``model``, as
    declared: the imputation model holds the outcome, BLUEPRINT §12 ruling 4); the data's own
    copies are passed through as they came, so a row's values depend on no other row and on no
    outcome (``row_local``, as declared)."""
    from turbotab.core.decisions import MissingSpec, ProjectState
    from turbotab.core.methods.missing import impute_for_inference, supplied_copies
    from turbotab.core.models.pipeline import design_spec

    rng = np.random.default_rng(5)
    n = 150
    x2 = rng.normal(size=n)
    x1 = 0.6 * x2 + rng.normal(size=n)
    y = 1.0 + x1 + 0.5 * x2 + rng.normal(size=n)
    frame = pd.DataFrame({"x1": np.where(rng.random(n) < 0.15, np.nan, x1), "x2": x2})
    st = ProjectState(purpose="inference", target="y",
                      roles={"x1": "covariate", "x2": "covariate"},
                      missing=MissingSpec(strategy="multiple_imputation"))
    spec = design_spec(st, frame, ["x1", "x2"])

    def drawn(f: pd.DataFrame, reference: pd.Series, outcome: np.ndarray) -> pd.DataFrame:
        return impute_for_inference(spec, f[spec.inputs], outcome, "regression", seed=0).frames[0]

    blank = int(np.flatnonzero(frame["x1"].isna().to_numpy())[0])
    none = np.zeros(n, dtype=bool)
    assert C.observed_scope(drawn, frame, none, y, blank, columns=["x1", "x2"]) == "model"
    assert C.contract("multiple_imputation_compatible").scope == "model"

    copies = pd.DataFrame({"SEQN": np.repeat(np.arange(30), 5), "_MULT_": np.tile(np.arange(1, 6), 30),
                           "x": rng.normal(size=150)})
    outcome = rng.normal(size=150)

    def passed(f: pd.DataFrame, reference: pd.Series, out: np.ndarray) -> pd.DataFrame:
        return pd.concat(supplied_copies(f, "_MULT_", "SEQN", ["x"], out).frames).loc[f.index]

    assert C.observed_scope(passed, copies, np.zeros(150, dtype=bool), outcome, 7,
                            columns=["x"]) == "row_local"
    assert C.contract("imputed_copies_pooled").scope == "row_local"


# ── the population answer binds the pooled curve ─────────────────────────────


COLUMNS = ["SDMVSTRA", "SDMVPSU", "WTDRD1", "protein_g", "fat_g", "carb_g", "energy_kcal", "age",
           "female", "eligible"]

POOLED_LOGIT_R = """
suppressMessages(library(mice))
ks <- scan("ks.txt", quiet = TRUE)
M <- as.integer(scan("m.txt", quiet = TRUE))
K <- length(ks)
Q <- matrix(NA_real_, M, K); U <- matrix(NA_real_, M, K); dg <- integer(0)
for (j in seq_len(M)) {
  f <- read.csv(sprintf("f%d.csv", j))
  d <- svydesign(ids = ~SDMVPSU, strata = ~SDMVSTRA, weights = ~WTDRD1, nest = TRUE, data = f)
  s <- subset(d, eligible == 1)
  g <- svyglm(high_crp ~ protein_g + fat_g + carb_g + energy_kcal + age + female, design = s,
              family = quasibinomial(), influence = TRUE,
              control = glm.control(epsilon = 1e-14, maxit = 100))
  inf <- attr(g, "influence")
  rows <- s$variables
  w <- weights(s)
  for (i in seq_len(K)) {
    m <- rows[[paste0("m", i)]]
    if (ks[i] == 0 || sum(m) == 0) next
    moved <- rows
    moved$fat_g <- rows[[paste0("fat", i)]]
    moved$carb_g <- rows[[paste0("carb", i)]]
    eta0 <- predict(g, newdata = rows, type = "link")
    eta1 <- predict(g, newdata = moved, type = "link")
    X0 <- model.matrix(delete.response(terms(g)), rows)
    X1 <- model.matrix(delete.response(terms(g)), moved)
    d0 <- g$family$mu.eta(eta0); d1 <- g$family$mu.eta(eta1)
    delta <- g$family$linkinv(eta1) - g$family$linkinv(eta0)
    wm <- w * m
    theta <- sum(wm * delta) / sum(wm)
    grad <- colSums(wm * (d1 * X1 - d0 * X0)) / sum(wm)
    z <- wm * (delta - theta) / sum(wm) + inf %*% grad
    Q[j, i] <- theta
    U[j, i] <- svyrecvar(z, s$cluster, s$strata, s$fpc)[1, 1]
  }
  dg <- c(dg, degf(s))
}
dfcom <- min(dg)
est <- rep(-1, K); half <- rep(-1, K); df <- rep(-1, K)
for (i in seq_len(K)) {
  if (any(is.na(Q[, i]))) next
  p <- pool.scalar(Q[, i], U[, i], n = dfcom + 1, k = 1)
  est[i] <- p$qbar; df[i] <- p$df; half[i] <- qt(0.975, p$df) * sqrt(p$t)
}
out(list(est = est, half = half, df = df, degf = dfcom))
"""

POOLED_CONTRAST_R = """
suppressMessages(library(mice))
M <- as.integer(scan("m.txt", quiet = TRUE))
Q <- c(); U <- c(); dg <- integer(0)
for (j in seq_len(M)) {
  f <- read.csv(sprintf("f%d.csv", j))
  d <- svydesign(ids = ~SDMVPSU, strata = ~SDMVSTRA, weights = ~WTDRD1, nest = TRUE, data = f)
  s <- subset(d, eligible == 1)
  g <- svyglm(crp ~ protein_g + fat_g + carb_g + energy_kcal + age + female, design = s)
  v <- svycontrast(g, c(fat_g = -1 / 9, carb_g = 1 / 4))
  Q <- c(Q, unname(coef(v))); U <- c(U, unname(SE(v))^2); dg <- c(dg, degf(s))
}
p <- pool.scalar(Q, U, n = min(dg) + 1, k = 1)
out(list(est = p$qbar, half = qt(0.975, p$df) * sqrt(p$t), df = p$df, degf = min(dg)))
"""


@pytest.fixture(scope="module")
def holed():
    """MS4's NHANES-shaped dietary table with 8% of the adults' protein blank (the under-20s, outside
    the analysis but inside the design, stay complete): energy is recorded on every row, so MI
    imputes protein within the energy identity and leaves the recorded total as it is."""
    diet = nhanes_diet()
    adults = np.flatnonzero(diet["eligible"].to_numpy() == 1)
    holes = np.random.default_rng(3).choice(adults, size=int(0.08 * len(adults)), replace=False)
    frame = diet.copy()
    frame.loc[holes, "protein_g"] = np.nan
    return {"diet": diet, "frame": frame, "adults": adults, "holes": holes}


def _copies_for_r(holed: dict, run: dict, outcome: str, ks: np.ndarray | None,
                  live: list[int]) -> dict[str, pd.DataFrame]:
    """Each imputed copy as R reads it: every row of the table (the design), the adults' values the
    copy's own, and at each live k the rows on support and their moved fat and carbohydrate."""
    from turbotab.core.methods.substitution import Shift

    copies = run["fit"].objects["imputations"]["frames"]
    out = {}
    for j, copy in enumerate(copies, start=1):
        f = holed["diet"][[*COLUMNS, outcome]].copy()
        f.loc[copy.index, ["protein_g", "energy_kcal"]] = copy[["protein_g", "energy_kcal"]]
        assert not f.isna().any().any()
        if ks is not None:
            X = copy[["protein_g", "fat_g", "carb_g", "energy_kcal", "age", "female"]]
            shift = Shift(X, donor="fat_g", recipient="carb_g",
                          kcal_per_unit={"fat_g": 9.0, "carb_g": 4.0}, total="energy_kcal")
            assert shift.valid(X).all()
            for i, k in enumerate(ks, start=1):
                f[f"m{i}"], f[f"fat{i}"], f[f"carb{i}"] = 0, f["fat_g"], f["carb_g"]
                if i - 1 not in live:
                    continue
                moved, amount, composition = shift.checks(X, float(k))
                f.loc[X.index[amount & composition], f"m{i}"] = 1
                f.loc[X.index, f"fat{i}"] = moved["fat_g"].to_numpy()
                f.loc[X.index, f"carb{i}"] = moved["carb_g"].to_numpy()
        out[f"f{j}"] = f
    return out


@needs_r
def test_under_the_population_answer_each_copys_curve_is_the_populations_pooled_on_the_design_df(
        holed, tmp_path):
    """A yes/no outcome (CRP above 3 mg/L) under the surveyed population with multiple imputation,
    fat moved to carbohydrate: the curve is nonlinear in the coefficients, so each copy's curve is
    drawn and pooled at each k (``pooled = "per_k"``). Each copy's curve is the population's
    average change in the survey-weighted fit's predicted probability, its variance the design's
    linearization (Graubard & Korn 1999); Rubin's rules pool them with the design's df as the
    complete-data df (Barnard & Rubin 1999). The 40 bootstrap refits asked for are not run. The GEE
    family has no design-based estimator, so its curve is blocked with the sample-only exit, as its
    table is.

    Reference, for each of the app's copies: R ``svyglm`` (quasibinomial, ``influence = TRUE``),
    R's ``predict`` and ``mu.eta`` for the change and its gradient, ``svyrecvar`` for its variance;
    then ``mice::pool.scalar(n = d + 1, k = 1)`` over the copies at each k. Bounds: the pooled curve
    1e-6, the interval's half-width 1e-4 and its degrees of freedom 1e-6, relative."""
    from turbotab.core.decisions import MissingSpec

    frame = holed["frame"].drop(columns=["eligible", "over", "crp"])
    run = survey_stages(frame, tmp_path / "stages", target="high_crp", task="binary",
                        roles=DIET_ROLES, models=["linear", "gee"], analyzed=holed["adults"],
                        event="1", missing=MissingSpec(strategy="multiple_imputation"),
                        substitution={"donor": "fat_g", "recipient": "carb_g",
                                      "step_kcal": 100.0, "n_boot": 40})
    fit, sub = run["fit"].data, run["substitution"]
    assert assert_design_based_or_blocked(fit, sub) == {"linear": "design", "gee": "blocked"}
    curve = next(c for c in sub["models"] if c["family"] == "linear")
    assert curve["pooled"] == "per_k" and curve["refused"] is None
    m = len(run["fit"].objects["imputations"]["frames"])
    assert m == fit["models"][0]["inference"]["missing"]["m"] == 20  # 8% incomplete: the floor
    ks = np.asarray(sub["ks"], dtype=float)
    live = [i for i, v in enumerate(curve["delta"]) if v is not None and ks[i] > 0]
    assert len(live) >= 3
    folder = tmp_path / "r"
    folder.mkdir()
    np.savetxt(folder / "ks.txt", ks)
    (folder / "m.txt").write_text(f"{m}\n")
    r = run_r(POOLED_LOGIT_R, _copies_for_r(holed, run, "high_crp", ks, live), folder)
    for i in live:
        assert curve["delta"][i] == pytest.approx(r["est"][i], rel=1e-6)
        assert curve["ci_high"][i] - curve["delta"][i] == pytest.approx(r["half"][i], rel=1e-4)
        assert curve["delta"][i] - curve["ci_low"][i] == pytest.approx(r["half"][i], rel=1e-4)
        assert curve["df"][i] == pytest.approx(r["df"][i], rel=1e-6)
        assert curve["df"][i] <= r["degf"]
    band = sub["band"]
    assert band["method"] == "design" and band["n_boot"] == 0 and band["df"] == r["degf"]
    adults = holed["diet"].iloc[holed["adults"]]
    psus = adults.groupby(["SDMVSTRA", "SDMVPSU"]).ngroups
    strata = adults["SDMVSTRA"].nunique()
    assert psus - strata == r["degf"]
    caption = (f"Shaded bands: 95% intervals pooled over {m} imputations by Rubin's rules, each "
               f"copy's variance by Taylor linearization over the survey design (weights `WTDRD1`; "
               f"{psus} PSUs in {strata} strata hold the {len(adults):,} analysis rows), on "
               f"Barnard–Rubin degrees of freedom with the design's {r['degf']} as the "
               f"complete-data degrees of freedom; each curve the population mean of the change in "
               f"the survey-weighted fit's prediction (Graubard & Korn 1999).")
    assert band["caption"] == caption
    assert (f"The curve is pooled over the {m} imputations: each copy's survey-weighted curve over "
            f"the surveyed population, pooled at each k by Rubin's rules with each copy's "
            f"Taylor-linearized variance as its within-copy variance and the design's degrees of "
            f"freedom as the complete-data df.") in sub["note"]
    assert "The 40 bootstrap refits asked for are not drawn" in sub["note"]
    gee = next(c for c in sub["models"] if c["family"] == "gee")
    assert "has no design-based estimator" in gee["refused"]


@needs_r
def test_under_the_population_answer_the_linear_contrast_pools_the_copies_design_tables(
        holed, tmp_path):
    """A continuous outcome under the surveyed population with multiple imputation: with every term
    linear, moving k kcal from fat to carbohydrate changes every row's prediction by the same
    contrast of the coefficients, so the pooled curve is that contrast of the pooled design-based
    coefficients, k (−β_fat/9 + β_carb/4), with Rubin's total variance and Barnard–Rubin degrees
    of freedom on the design's complete-data df (``pooled = "contrast"``).

    Reference, for each of the app's copies: R ``svyglm`` and ``svycontrast`` of
    ``(−1/9)·β_fat + (1/4)·β_carb`` per kcal; then ``mice::pool.scalar(n = d + 1, k = 1)``. Bounds:
    the curve 1e-6, the half-width 1e-4 and the df 1e-6, relative, at every k."""
    from turbotab.core.decisions import MissingSpec

    frame = holed["frame"].drop(columns=["eligible", "over", "high_crp"])
    run = survey_stages(frame, tmp_path / "stages", target="crp", task="regression",
                        roles=DIET_ROLES, models=["linear"], analyzed=holed["adults"],
                        missing=MissingSpec(strategy="multiple_imputation"),
                        substitution={"donor": "fat_g", "recipient": "carb_g",
                                      "step_kcal": 100.0})
    sub = run["substitution"]
    assert assert_design_based_or_blocked(run["fit"].data, sub) == {"linear": "design"}
    (curve,) = sub["models"]
    assert curve["pooled"] == "contrast"
    m = len(run["fit"].objects["imputations"]["frames"])
    folder = tmp_path / "r"
    folder.mkdir()
    (folder / "m.txt").write_text(f"{m}\n")
    r = run_r(POOLED_CONTRAST_R, _copies_for_r(holed, run, "crp", None, []), folder)
    ks = np.asarray(sub["ks"], dtype=float)
    live = [i for i, v in enumerate(curve["delta"]) if v is not None and ks[i] > 0]
    assert len(live) >= 3
    for i in live:
        k = ks[i]
        assert curve["delta"][i] == pytest.approx(r["est"] * k, rel=1e-6)
        assert curve["ci_high"][i] - curve["delta"][i] == pytest.approx(r["half"] * k, rel=1e-4)
        assert curve["df"][i] == pytest.approx(r["df"], rel=1e-6)
    assert sub["band"]["method"] == "design" and sub["band"]["df"] == r["degf"]
    # The pooled df sits below the design's: Barnard–Rubin never exceeds the complete-data df.
    assert 0 < r["df"] <= r["degf"]


# ── two times on the follow-up's scale ───────────────────────────────────────


def test_follow_up_ends_at_its_horizon_and_risks_are_judged_at_the_prediction_horizon(tmp_path):
    """One follow-up answer with both times: follow-up ends at 3 (the routing gate: later events
    censored at 3) and predicted risks are judged at 2 (MS6). The fit scores and calibrates at 2
    and says so; with no prediction horizon declared it reads the median follow-up time of the
    outcome as censored at 3, computed here by NumPy from the raw columns. A prediction horizon at
    or past the end of follow-up is refused when the answer is made: at 3 every row still at risk
    is censored, so the censoring survival Graf's weights divide by is 0 (Kaplan–Meier of the
    censoring by NumPy). The record's sentence says both, verbatim. The gate's own exits, offered
    for a rule on the follow-up time, keep a declared prediction horizon that still fits and drop
    one that no longer does."""
    from turbotab.core import decisions as d
    from turbotab.core import voice
    from turbotab.core.decisions import FollowUpSpec, SplitSpec
    from turbotab.core.tests.acceptance.test_ms6_prediction_validation import _stages, _state

    rng = np.random.default_rng(41)
    n = 240
    X = rng.normal(size=(n, 2))
    t_event = rng.exponential(np.exp(-(X @ np.array([0.7, -0.4]))) * 3.0)
    t_cens = rng.uniform(1.0, 6.0, n)
    time = np.minimum(t_event, t_cens)
    event = (t_event <= t_cens).astype(int)
    frame = pd.DataFrame({"x1": X[:, 0], "x2": X[:, 1], "pid": np.arange(n), "time": time,
                          "dead": event})
    split = SplitSpec(holdout=0.0, seed=4, folds=5)

    def fitted(follow_up: FollowUpSpec, folder: str):
        st = _state(frame, task="time_to_event", target="dead", models=["cox"], split=split,
                    event="1", follow_up=follow_up)
        return _stages(frame, st, tmp_path / folder)[-1].data

    both = fitted(FollowUpSpec(time_column="time", horizon=3.0, prediction_horizon=2.0), "both")
    assert both["horizon"] == 2.0 and both["primary_metric"] == "brier_t"
    assert both["horizon_note"] == ("Scored and calibrated by `time` = `2`, the declared "
                                    "prediction horizon.")
    assert both["models"][0]["calibration_horizon"]["horizon"] == 2.0
    ended = fitted(FollowUpSpec(time_column="time", horizon=3.0), "ended")
    median = float(np.median(np.minimum(time, 3.0)))
    assert ended["horizon"] == pytest.approx(median, rel=1e-12)
    assert ended["horizon_note"].endswith("as no prediction horizon was declared with the "
                                          "follow-up.")

    # Why the prediction horizon must come before the end of follow-up: the censoring's
    # Kaplan–Meier survival at 3, once follow-up ends there, is 0.
    t3, e3 = np.minimum(time, 3.0), np.where(time > 3.0, 0, event)
    survival = 1.0
    for t in np.unique(t3[e3 == 0]):
        survival *= 1 - np.sum((t3 == t) & (e3 == 0)) / np.sum(t3 >= t)
    assert survival == 0.0
    with pytest.raises(ValueError, match="before follow-up ends"):
        d.SetFollowUp(column="dead", time_column="time", horizon=3.0, prediction_horizon=3.0)
    with pytest.raises(ValueError, match="after the landmark"):
        d.SetFollowUp(column="dead", time_column="time", landmark=1.0, prediction_horizon=0.5)

    st = _state(frame, task="time_to_event", target="dead", models=["cox"], split=split,
                event="1", follow_up=FollowUpSpec(time_column="time", horizon=3.0,
                                                  prediction_horizon=2.0))
    said = voice.sentence_for(d.SetFollowUp(column="dead", time_column="time", horizon=3.0,
                                            prediction_horizon=2.0), st, {})
    assert said == ("`dead` was analyzed as a time to event, each row followed until `time`, at the "
                    "event or when follow-up ended without it; follow-up ended at `3`: events "
                    "after it were censored there; predicted risks were scored and calibrated by "
                    "`time` = `2`, the declared prediction horizon.")

    from turbotab.core.decisions import ExclusionRule, _follow_up_rule_refusal

    def ended_by(high: float) -> tuple[float, float | None]:
        rule = ExclusionRule(column="time", high=high, reason="followed up to then")
        refusal = _follow_up_rule_refusal(st, [rule], d.SetExclusions(rules=[]))
        answer = next(e["decision"] for e in refusal.exits
                      if str(e["label"]).startswith("End follow-up"))
        return answer["horizon"], answer["prediction_horizon"]

    assert ended_by(2.5) == (2.5, 2.0)
    assert ended_by(1.5) == (1.5, None)
