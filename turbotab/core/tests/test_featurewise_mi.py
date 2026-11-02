"""Feature-wise regression pooled over multiple imputations drawn one study factor at a time (the
plasma scope's item 1; the orchestrator's methods ruling of 2026-11-01, findings M1 and M2).

Every plasma analyst chose the censoring-aware multiple imputation the engine offered for values
below detection, and the feature-wise family refused it at the first Fit, after the lock; its way
forward, one censoring-aware fill per feature, is intercept-only and ignores the outcome, so under
the limma design (the feature is the response) it shrank a true difference of 0.5 to 0.357 at 60%
censored, coverage 0.865 (the scope's ``fill_sim.py``). Now each study factor's blanks are drawn m
times from its own censored-normal model given the outcome and the covariates, each copy is tested
by the family's own tests, each study factor pooled by Rubin's rules with Barnard and Rubin's df,
then Benjamini–Hochberg on the pooled p-values; a plan the table cannot pool is refused before the
lock, and under inference the single censoring-aware fill leaves the feature-wise exits.

The truths, each independent of the engine's output:

* (1) the pooling: ``mice::pool`` (3.19.0) on ``lm`` fits to the engine's own copies (``dfcom`` from
  ``df.residual``), then ``p.adjust(p, "BH")``: each study factor's estimate, standard error, df,
  p and q at a relative 1e-8;
* (2) the draws: ``survival::survreg``'s left-censored Gaussian fit of each study factor on the
  event and the covariate; at m = 200 the pooled estimate lies within a few Monte Carlo errors of it;
* (3) a simulation by construction (the difference known, 30% and 60% censored): coverage and the
  null rejection rate near 0.95 and 0.05 within Monte Carlo error, the false-discovery proportion
  at or below q (``test_featurewise_mi_simulation.py``, marked slow);
* (4) a study factor with no blank: its pooled row is its complete-data test (B = 0, so T = Ū and
  ν = ν_com, a derivation), statsmodels' least squares;
* (5) the budget's words interpolated from the counts, refused beyond it and silent within it.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

import turbotab.core.models  # noqa: F401 - registers the families
from turbotab.core.methods.imputation import Censoring
from turbotab.core.tests.acceptance.r_reference import needs_r, run_r


def censored_panel(seed: int, n: int = 100, n_event: int = 58,
                   shares: tuple[float, ...] = (0.2, 0.3, 0.4, 0.5, 0.6, 0.35),
                   deltas: tuple[float, ...] = (0.5, 0.0, 0.8, 0.3, 0.5, 0.0),
                   complete: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """``n`` rows, ``n_event`` of them events, two covariates; each study factor ``f{j}`` is
    10 + ``deltas[j]`` × event + 0.3 z0 − 0.2 z1 + N(0, 1), blank below its ``shares[j]`` quantile
    (left-censored at that limit), and ``complete`` more with no blank (``g{j}``). The matrix as the
    pooled table reads it: the study factors, then the adjustment columns."""
    rng = np.random.default_rng(seed)
    event = np.r_[np.ones(n_event), np.zeros(n - n_event)]
    z = rng.standard_normal((n, 2))
    cols: dict[str, np.ndarray] = {}
    for j, (share, delta) in enumerate(zip(shares, deltas)):
        x = 10 + delta * event + 0.3 * z[:, 0] - 0.2 * z[:, 1] + rng.standard_normal(n)
        limit = np.quantile(x, share)
        cols[f"f{j}"] = np.where(x < limit, np.nan, x)
    for j in range(complete):
        cols[f"g{j}"] = 10 + 0.4 * event + 0.3 * z[:, 0] + rng.standard_normal(n)
    frame = pd.DataFrame(cols)
    frame["z0"], frame["z1"] = z[:, 0], z[:, 1]
    return frame, event


def limits_of(frame: pd.DataFrame, features: list[str]) -> dict[str, Censoring]:
    """Each censored study factor's limit: its smallest detected value, on its own scale (the
    pooled table's scale after the design's log)."""
    return {f: Censoring(limit=float(np.nanmin(frame[f].to_numpy(dtype=float))), log=False)
            for f in features if frame[f].isna().any()}


MICE_POOL = """
suppressPackageStartupMessages({library(mice)})
d <- read.csv(design_csv)
cp <- read.csv(copies_csv)
m <- max(cp$copy)
res <- list()
for (f in unique(cp$feature)) {
  fits <- lapply(seq_len(m), function(k) {
    x <- cp$value[cp$feature == f & cp$copy == k]
    lm(x ~ disease + z0 + z1, data = cbind(d, x = x))
  })
  s <- summary(pool(as.mira(fits)))
  r <- s[s$term == "disease", ]
  res[[f]] <- list(estimate = r$estimate, se = r$std.error, df = r$df, p = r$p.value,
                   dfcom = fits[[1]]$df.residual)
}
p <- sapply(res, function(o) o$p)
q <- p.adjust(p, "BH")
for (f in names(res)) res[[f]]$q <- unname(q[f])
out(list(version = as.character(packageVersion("mice")), rows = res))
"""


@needs_r
def test_1_each_pooled_row_is_mice_pool_on_the_engines_own_copies(tmp_path):
    """Six censored study factors (20–60% below their limits), two covariates: the engine's copies
    handed to R, each copy's ``lm(x ~ disease + z0 + z1)`` pooled by ``mice::pool`` (Rubin's rules,
    Barnard–Rubin df with ``dfcom`` = ``df.residual`` = n − 4 = ν_com), then ``p.adjust(p, "BH")``:
    every study factor's estimate, standard error, df, p and q agree at a relative 1e-8."""
    from turbotab.core.methods import featurewise_mi as fm

    frame, event = censored_panel(seed=11)
    features = [c for c in frame.columns if c.startswith("f")]
    table, copies = fm.pooled_featurewise_table(frame, event, "binary", features,
                                                censored=limits_of(frame, features), seed=3)
    assert copies is not None and copies.m == 60  # m_rule: the most incomplete model, 60 of 100
    long = pd.DataFrame([{"feature": f, "copy": k + 1, "row": i + 1, "value": v}
                         for f in features for k in range(copies.m)
                         for i, v in enumerate(copies.values[f][:, k])])
    design = pd.DataFrame({"disease": event, "z0": frame["z0"], "z1": frame["z1"]})
    ref = run_r(MICE_POOL, {"design": design, "copies": long}, tmp_path)
    assert ref["version"] == "3.19.0"
    rows = {r["feature"]: r for r in table.rows}
    for f in features:
        want = ref["rows"][f]
        assert want["dfcom"] == len(frame) - 4
        got = rows[f]
        for key in ("estimate", "se", "df", "p", "q"):
            assert got[key] == pytest.approx(want[key], rel=1e-8), (f, key)


def test_4_a_study_factor_with_no_blank_is_its_complete_data_test():
    """`g0` and `g1` have no blank: no copy of them is drawn, and Rubin's rules with B = 0 give
    T = Ū and ν = ν_com, so each pooled row is the complete-data test, statsmodels' least squares
    of the factor on the event and the covariates (estimate, standard error, p, df = n − 4)."""
    from turbotab.core.methods import featurewise_mi as fm

    frame, event = censored_panel(seed=5, shares=(0.4,), deltas=(0.5,), complete=2)
    features = ["f0", "g0", "g1"]
    table, copies = fm.pooled_featurewise_table(frame, event, "binary", features,
                                                censored=limits_of(frame, features), seed=1)
    assert sorted(copies.values) == ["f0"]
    rows = {r["feature"]: r for r in table.rows}
    for f in ("g0", "g1"):
        X = sm.add_constant(pd.DataFrame({"disease": event, "z0": frame["z0"], "z1": frame["z1"]}))
        fit = sm.OLS(frame[f].to_numpy(float), X).fit()
        assert rows[f]["estimate"] == pytest.approx(float(fit.params["disease"]), rel=1e-10)
        assert rows[f]["se"] == pytest.approx(float(fit.bse["disease"]), rel=1e-10)
        assert rows[f]["p"] == pytest.approx(float(fit.pvalues["disease"]), rel=1e-8)
        assert rows[f]["df"] == len(frame) - 4


SURVREG = """
suppressPackageStartupMessages(library(survival))
d <- read.csv(panel_csv)
out_rows <- list()
for (f in c("f0", "f1")) {
  x <- d[[f]]
  cens <- is.na(x)
  limit <- min(x, na.rm = TRUE)
  t <- ifelse(cens, limit, x)
  fit <- survreg(Surv(t, !cens, type = "left") ~ disease + z0, data = d, dist = "gaussian")
  out_rows[[f]] <- unname(coef(fit)["disease"])
}
out(out_rows)
"""


@needs_r
def test_2_at_m_200_the_pooled_estimate_sits_on_survregs_censored_fit(tmp_path):
    """Two study factors, 40% and 60% below their limits, on 300 rows with one covariate: the
    maximum-likelihood difference of a left-censored Gaussian (``survival::survreg``), which the
    copies' model is, against the pooled estimate at m = 200, whose Monte Carlo error is √(B/m)
    (its ``mc_se``). Proper imputation's pooled estimate converges to that fit as m grows, up to
    an O(1/n) difference, so it lies within three Monte Carlo errors of it. (Over six seeds of
    this panel the gap was 0.7 to 1.9 Monte Carlo errors, always below survreg's, about 0.05 of a
    standard error: that O(1/n) difference.)"""
    from turbotab.core.methods import featurewise_mi as fm

    frame, event = censored_panel(seed=29, n=300, n_event=170, shares=(0.4, 0.6),
                                  deltas=(0.5, 0.3))
    frame = frame.drop(columns=["z1"])
    features = ["f0", "f1"]
    table, copies = fm.pooled_featurewise_table(frame, event, "binary", features,
                                                censored=limits_of(frame, features), m=200,
                                                seed=8)
    assert copies.m == 200
    ref = run_r(SURVREG, {"panel": frame.assign(disease=event)}, tmp_path)
    for row in table.rows:
        gap = abs(row["estimate"] - ref[row["feature"]])
        assert gap <= 3 * row["mc_se"], (row["feature"], gap, row["mc_se"])


def test_5_beyond_the_budget_the_copies_are_refused_in_words_from_the_counts(monkeypatch):
    """The budget is the draws, (study factors drawn) × m × sweeps, against MAX_FEATURE_COPIES:
    six study factors drawn, m = 60, one sweep each (the covariates are complete and the rows
    independent), so 360 draws; refused at a budget of 359 in words built from those counts, and
    drawn at 360."""
    from turbotab.core.methods import featurewise_mi as fm

    frame, event = censored_panel(seed=11)
    features = [c for c in frame.columns if c.startswith("f")]
    monkeypatch.setattr(fm, "MAX_FEATURE_COPIES", 359)
    with pytest.raises(fm.Unpooled) as refused:
        fm.pooled_featurewise_table(frame, event, "binary", features,
                                    censored=limits_of(frame, features), seed=3)
    assert refused.value.reason == (
        "Feature-wise regression is pooled over multiple imputations drawn one study factor at a "
        "time: here 6 study factors × 60 imputations make 360 draws, more than the 359 it makes.")
    monkeypatch.setattr(fm, "MAX_FEATURE_COPIES", 360)
    table, copies = fm.pooled_featurewise_table(frame, event, "binary", features,
                                                censored=limits_of(frame, features), seed=3)
    assert copies.m == 60 and all(r["p"] is not None for r in table.rows)


MICE_CHAINED = """
suppressPackageStartupMessages({library(mice)})
d <- read.csv(panel_csv)
imp <- mice(d[, c("x", "z0", "disease")], method = c("norm", "norm", ""), m = 200, maxit = 10,
            seed = 7, printFlag = FALSE)
p <- pool(with(imp, lm(x ~ disease + z0)))
r <- p$pooled[p$pooled$term == "disease", ]
out(list(estimate = r$estimate, b = r$b, t = r$t, m = p$m))
"""


@needs_r
def test_8_where_a_covariate_is_blank_the_chained_copies_center_on_mices(tmp_path):
    """A study factor missing at random on 30% of 300 rows and its covariate on 25%: its copies
    are drawn by chained equations on {factor, outcome, covariate} (``smcfcs.impute``'s fcs mode,
    ten sweeps), the covariate's draws seeing the factor its test treats as the response. mice
    (3.19.0) with Bayesian linear-regression draws (``norm``) for both and the same predictors
    samples the same model, so at m = 200 each the two pooled differences lie within three of
    their combined Monte Carlo errors (√(B/m) each), and the totals of variance within 10%."""
    from turbotab.core.methods import featurewise_mi as fm

    rng = np.random.default_rng(41)
    n = 300
    event = (rng.random(n) < 0.55).astype(float)
    z = rng.standard_normal(n)
    x = 10 + 0.5 * event + 0.6 * z + rng.standard_normal(n)
    frame = pd.DataFrame({"x": np.where(rng.random(n) < 0.3, np.nan, x),
                          "z0": np.where(rng.random(n) < 0.25, np.nan, z)})
    table, copies = fm.pooled_featurewise_table(frame, event, "binary", ["x"], censored={},
                                                m=200, seed=5)
    assert copies.sweeps == 10 and copies.adjust  # chained: the covariate drawn with it
    (row,) = table.rows
    ref = run_r(MICE_CHAINED, {"panel": frame.assign(disease=event)}, tmp_path)
    assert ref["m"] == 200
    error = math.sqrt(row["mc_se"] ** 2 + ref["b"] / ref["m"])
    assert abs(row["estimate"] - ref["estimate"]) <= 3 * error
    assert row["se"] ** 2 == pytest.approx(ref["t"], rel=0.10)


# ── (6) the check before the lock, both ways (fix (b)) ───────────────────────


class _Store:
    """The table a decision context reads (the server's ``DataStore`` shape)."""

    def __init__(self, frame: pd.DataFrame):
        self.frame = frame
        self.columns = list(frame.columns)

    def materialize(self, columns: list[str], rows=None) -> pd.DataFrame:
        return self.frame[list(columns)]


MI = {"strategy": "multiple_imputation", "below_detection": "censoring_aware",
      "censored_columns": [f"f{j}" for j in range(6)]}
ONCE = {"strategy": "impute", "below_detection": "censoring_aware",
        "censored_columns": [f"f{j}" for j in range(6)], "acknowledged": True}


def plan_state(models: list[str], missing: dict, purpose: str = "inference"):
    """The six censored study factors and two covariates of :func:`censored_panel`, a two-level
    outcome, under ``purpose``, with ``models`` chosen and ``missing`` answered."""
    from turbotab.core.decisions import SplitSpec
    from turbotab.core.tests import modeling_fixtures as mf

    roles = {**{f"f{j}": "exposure" for j in range(6)}, "z0": "covariate", "z1": "covariate"}
    return mf.state(lens=["clinical"], target="disease", task="binary", event="1",
                    purpose=purpose, roles=roles, role_confirmations=dict(roles), models=models,
                    missing=missing, split=SplitSpec(holdout=0.0, seed=0, folds=5),
                    shape_confirmations={})


def plan_ctx(state) -> dict:
    frame, event = censored_panel(seed=11)
    frame = frame.assign(disease=event)
    return {"state": state, "columns": list(frame.columns), "store": lambda: _Store(frame),
            "analyzed": lambda: np.arange(len(frame))}


BUDGET_WORDS = ("Feature-wise regression is pooled over multiple imputations drawn one study "
                "factor at a time: here 6 study factors × 60 imputations make 360 draws, more than "
                "the 359 it makes.")


@pytest.mark.parametrize("order", ["models_after_missing", "missing_after_models", "lock"])
def test_6_a_plan_beyond_the_budget_is_refused_before_the_lock_in_either_order(monkeypatch,
                                                                               order):
    """The models answer after the missing-values answer, the missing-values answer after the
    models answer, and the Fit press's ``lock_plan``: each refuses a feature-wise plan whose draws,
    counted on the plan's rows (6 censored study factors, m = 60 by the rule over the most
    incomplete one's 60 of 100 rows), exceed the budget, in the fit's words, with complete cases
    (counted here: the rows that hold all six) and another family as ways forward, none filling
    once; at the budget each passes."""
    from turbotab.core import decisions as d
    from turbotab.core.methods import featurewise_mi as fm

    decision = {"models_after_missing": d.SelectModels(models=["featurewise"]),
                "missing_after_models": d.SetMissing(**MI),
                "lock": d.LockPlan()}[order]
    state = plan_state(["featurewise"], MI)
    monkeypatch.setattr(fm, "MAX_FEATURE_COPIES", 359)
    with pytest.raises(d.Refusal) as refused:
        d.validate(decision, plan_ctx(state))
    assert refused.value.code == fm.UNPOOLED
    assert refused.value.message == BUDGET_WORDS
    frame, _ = censored_panel(seed=11)
    whole = int(frame[[f"f{j}" for j in range(6)]].notna().all(axis=1).sum())
    assert [e["label"] for e in refused.value.exits] == [
        f"Complete cases ({whole} of the 100 rows)", "Choose another model family"]
    monkeypatch.setattr(fm, "MAX_FEATURE_COPIES", 360)
    d.validate(decision, plan_ctx(state))  # silent within the budget


def test_6_beside_another_family_the_refusal_offers_it_without_feature_wise_regression(
        monkeypatch):
    """Chosen beside the linear model, the refusal's last way forward fits the linear model
    alone."""
    from turbotab.core import decisions as d
    from turbotab.core.methods import featurewise_mi as fm

    monkeypatch.setattr(fm, "MAX_FEATURE_COPIES", 359)
    state = plan_state(["linear"], MI)
    with pytest.raises(d.Refusal) as refused:
        d.validate(d.SelectModels(models=["linear", "featurewise"]), plan_ctx(state))
    assert refused.value.code == fm.UNPOOLED
    assert refused.value.exits[-1] == {
        "label": "Fit the other family chosen, without feature-wise regression",
        "decision": d.SelectModels(models=["linear"]).model_dump(mode="json")}


def _marginal_ctx(measure: str, blanks: int = 40, missing: Any = "complete_case") -> dict:
    """``estimand_fixtures.cohort`` (300 rows), `activity` blank on its first ``blanks`` rows, the
    yes/no outcome `dm` with the linear family (logistic for it) and the estimand ``measure``."""
    from turbotab.core.tests.acceptance import estimand_fixtures as ef

    frame = ef.cohort(300, seed=5)
    frame.loc[frame.index[:blanks], "activity"] = np.nan
    st = ef.state(target="dm", task="binary", measure=measure, event="yes", models=["linear"],
                  missing=missing)
    return {"state": st, "columns": list(frame.columns), "store": lambda: _Store(frame),
            "analyzed": lambda: np.arange(len(frame))}


@pytest.mark.parametrize("decision", ["set_missing", "lock"])
def test_6_the_marginal_risks_under_multiple_imputation_are_refused_before_the_lock(decision):
    """The marginal risk difference, which the effects stage standardizes from the logistic model
    and refuses under multiple imputation (its words, ``effects.MARGINAL_UNPOOLED``), is refused
    where the plan would draw copies (`activity` is blank on 40 of the 300 rows): at the
    missing-values answer and at the lock, the conditional odds ratio its first way forward."""
    from turbotab.core import decisions as d
    from turbotab.core.stages.effects import MARGINAL_UNPOOLED

    ctx = _marginal_ctx("risk_difference", missing=(
        {"strategy": "multiple_imputation"} if decision == "lock" else "complete_case"))
    with pytest.raises(d.Refusal) as refused:
        d.validate(d.SetMissing(strategy="multiple_imputation") if decision == "set_missing"
                   else d.LockPlan(), ctx)
    assert refused.value.code == "marginal_risks_unpooled"
    assert refused.value.message == MARGINAL_UNPOOLED
    first = refused.value.exits[0]
    assert first["label"] == "Report the conditional odds ratio"
    assert (first["decision"]["kind"], first["decision"]["measure"]) == ("set_estimand",
                                                                        "odds_ratio")


def test_6_the_conditional_odds_ratio_or_a_plan_with_no_blank_passes():
    """Silent the other way: the odds ratio under multiple imputation, and the risk difference
    where no predictor is blank (no copy is drawn)."""
    from turbotab.core import decisions as d

    d.validate(d.SetMissing(strategy="multiple_imputation"), _marginal_ctx("odds_ratio"))
    d.validate(d.SetMissing(strategy="multiple_imputation"),
               _marginal_ctx("risk_difference", blanks=0))


# ── (7) the single censoring-aware fill leaves the feature-wise exits under inference (M2) ──

SINGLE_FILL = (
    "One censoring-aware fill gives every value below detection in a feature the same number, from "
    "that feature's own censored-normal fit alone, without the outcome or the covariates its test "
    "reads. Under a two-level outcome feature-wise regression tests each feature as the response, "
    "so such a fill biases its difference toward zero, and under either outcome one value per "
    "blank makes its intervals too narrow. Under inference its tests are pooled over "
    "censoring-aware multiple imputations instead: each feature's values below detection are drawn "
    "from its own model given the outcome and the covariates.")


@pytest.mark.parametrize("order", ["missing_after_models", "models_after_missing", "lock"])
def test_7_under_inference_one_censoring_aware_fill_is_refused_for_the_pooled_route(order):
    """Under inference the feature-wise table takes no single censoring-aware fill, recorded or
    not, at either answer or the lock: refused in words that say what the fill does, the pooled
    route (multiple imputation, censoring-aware, the answer's columns kept) its first way forward
    and complete cases its second; no way forward fills once."""
    from turbotab.core import decisions as d
    from turbotab.core.methods import featurewise_mi as fm

    decision = {"missing_after_models": d.SetMissing(**ONCE),
                "models_after_missing": d.SelectModels(models=["featurewise"]),
                "lock": d.LockPlan()}[order]
    with pytest.raises(d.Refusal) as refused:
        d.validate(decision, plan_ctx(plan_state(["featurewise"], ONCE)))
    assert refused.value.code == fm.SINGLE_FILL
    assert refused.value.message == SINGLE_FILL
    pooled, cases = [e["decision"] for e in refused.value.exits]
    assert (pooled["strategy"], pooled["below_detection"], pooled["acknowledged"]) == (
        "multiple_imputation", "censoring_aware", False)
    assert pooled["censored_columns"] == ONCE["censored_columns"]
    assert cases["strategy"] == "complete_case"


def test_7_the_single_fill_stays_where_the_feature_wise_table_is_not_chosen_or_under_prediction():
    """Silent the other way: the linear model under inference keeps its recorded single fill (its
    own block-and-record), and under prediction the in-fold censoring-aware fill is the answer the
    menu offers and takes."""
    from turbotab.core import decisions as d
    from turbotab.core.methods.omics import detection_limit_options

    d.validate(d.SetMissing(**ONCE), plan_ctx(plan_state(["linear"], ONCE)))
    d.validate(d.SetMissing(**{**ONCE, "acknowledged": False}),
               plan_ctx(plan_state(["linear"], ONCE, purpose="prediction")))
    aware = next(o for o in detection_limit_options("prediction", 0.4)
                 if o["key"] == "censoring_aware")
    assert aware["sound"] == ("Sound: the expected value below the limit under a censored-normal "
                              "fit, in each training fold without the outcome.")


def test_7_the_feature_wise_exits_offer_no_single_censoring_aware_fill():
    """The feature-wise table's ways forward where its copies are not pooled for it
    (``omics.featurewise_missing_exits``): complete cases where rows enough hold every value, and
    no censoring-aware fill where values below detection are named (on the base it was offered);
    with no complete case left and no value below detection, one fill recorded as a limitation, as
    before."""
    from turbotab.core.methods.omics import featurewise_missing_exits

    frame, _ = censored_panel(seed=11)
    state = plan_state(["featurewise"], MI)
    assert [e["way"] for e in featurewise_missing_exits(state, frame)] == ["complete_case"]
    few = frame.copy()
    few.loc[few.notna().all(axis=1), "f0"] = np.nan  # no row holds every value
    assert [e["way"] for e in featurewise_missing_exits(state, few)] == []
    plain = plan_state(["featurewise"], {"strategy": "multiple_imputation"})
    assert [e["way"] for e in featurewise_missing_exits(plain, few)] == ["single_fill"]


# ── (9) what the copies do not draw, refused both ways ───────────────────────


def _cohort_fit(tmp_path, blank: dict[str, int]) -> tuple[Any, Any]:
    """``estimand_fixtures.cohort(300)`` with each column of ``blank`` blank on its first rows,
    feature-wise regression of `glucose` on `fiber` (its covariates `age`, `sex`, `smoking`,
    `activity`) under multiple imputation: the fit's feature-wise entry, and the state."""
    from turbotab.core.stages.modeling import design_stage, fit_stage
    from turbotab.core.tests import modeling_fixtures as mf
    from turbotab.core.tests.acceptance import estimand_fixtures as ef

    frame = ef.cohort(300, seed=5)
    for column, rows in blank.items():
        frame.loc[frame.index[:rows], column] = np.nan
    st = ef.state(target="glucose", task="regression", measure="mean_difference",
                  models=["featurewise"], missing={"strategy": "multiple_imputation", "m": 20})
    paths = mf.ingest_frame(frame, tmp_path)
    split = mf.split_bundle(np.arange(len(frame)), holdout=0.0, seed=st.split.seed)
    info = mf.target_info(st.task, st.target)
    design = design_stage(mf.context(st, {"split": split, "target_info": info}, paths))
    fit = fit_stage(mf.context(st, {"design": design, "split": split, "target_info": info},
                               paths))
    return next(m for m in fit.data["models"] if m["family"] == "featurewise"), st


CATEGORY_WORDS = ("Feature-wise regression is pooled over multiple imputations one study factor at "
                  "a time, drawing numbers; `sex` is a category among the covariates with blanks, "
                  "which these copies do not draw.")


def test_9_a_category_among_the_covariates_with_blanks_is_refused_at_the_fit_and_the_plan(
        tmp_path):
    """`sex` (text, a covariate) blank on 30 of the 300 rows: the copies draw numbers, so the fit
    refuses the table in words that name it, complete cases its first way forward, and the check
    before the lock refuses the models answer in the same words (counted on the plan's rows). With
    `activity`, a number, blank instead, both are silent: the table is pooled."""
    from turbotab.core import decisions as d
    from turbotab.core.methods import featurewise_mi as fm
    from turbotab.core.tests.acceptance import estimand_fixtures as ef

    entry, st = _cohort_fit(tmp_path / "sex", {"sex": 30})
    assert entry["inference"]["refused"] == CATEGORY_WORDS and not entry["coefficients"]
    assert entry["inference"]["exits"][0]["decision"]["strategy"] == "complete_case"
    frame = ef.cohort(300, seed=5)
    frame.loc[frame.index[:30], "sex"] = np.nan
    ctx = {"state": st, "columns": list(frame.columns), "store": lambda: _Store(frame),
           "analyzed": lambda: np.arange(len(frame))}
    with pytest.raises(d.Refusal) as refused:
        d.validate(d.SelectModels(models=["featurewise"]), ctx)
    assert (refused.value.code, refused.value.message) == (fm.UNPOOLED, CATEGORY_WORDS)
    entry, _ = _cohort_fit(tmp_path / "activity", {"activity": 30})
    assert entry["inference"]["refused"] is None and entry["coefficients"]
    frame = ef.cohort(300, seed=5)
    frame.loc[frame.index[:30], "activity"] = np.nan
    d.validate(d.SelectModels(models=["featurewise"]),
               {**ctx, "store": lambda: _Store(frame)})


def test_9_a_study_factor_with_one_observed_value_is_refused_with_leaving_it_out():
    """A study factor observed on one row has nothing to model its blanks from (the engine's
    chained equations refuse fewer than two observed values): the copies are refused naming it.
    With two observed values it is drawn."""
    from turbotab.core.methods import featurewise_mi as fm
    from turbotab.core.methods.imputation import ImputationRefused

    frame, event = censored_panel(seed=3, shares=(0.4,), deltas=(0.5,))
    frame["f1"] = np.nan
    frame.loc[0, "f1"] = 9.0
    with pytest.raises(ImputationRefused) as refused:
        fm.pooled_featurewise_table(frame, event, "binary", ["f0", "f1"], censored={}, m=20)
    assert str(refused.value) == "`f1` has fewer than two observed values, so it cannot be imputed."
    assert refused.value.columns == ["f1"]
    frame.loc[1, "f1"] = 11.0
    table, copies = fm.pooled_featurewise_table(frame, event, "binary", ["f0", "f1"],
                                                censored={}, m=20)
    assert sorted(copies.values) == ["f0", "f1"]
