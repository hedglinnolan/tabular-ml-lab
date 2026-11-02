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
    assert refused.value.reason == _budget_words(6, 60, 359)
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
                "the 359 one table may draw, a limit that bounds how long its fit takes.")


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


# ── the repair round (the verifier's FWMI-V1 … V8) ───────────────────────────


def _spy_kinds(monkeypatch) -> list:
    """Each call's (column, kind) pairs as the engine's sampler is handed them (``smcfcs.impute``'s
    ``variables``): what each blank is drawn by, read off the sampler's own input."""
    from turbotab.core.methods import smcfcs

    seen: list = []
    real = smcfcs.impute

    def spy(data, variables, **kw):
        seen.append(sorted({(v.name, v.kind) for v in variables}))
        return real(data, variables, **kw)

    monkeypatch.setattr(smcfcs, "impute", spy)
    return seen


def _two_valued_panel(seed: int = 4, n: int = 200) -> tuple[pd.DataFrame, np.ndarray]:
    """``n`` rows: `f0`, a study factor below its 30% quantile blank (censored at it); `z0` a
    number and `b` a 0/1 covariate, blank on its first 40 rows."""
    rng = np.random.default_rng(seed)
    event = (rng.random(n) < 0.5).astype(float)
    z = rng.standard_normal(n)
    b = (rng.random(n) < 0.4).astype(float)
    x = 10 + 0.5 * event + 0.3 * z + 0.4 * b + rng.standard_normal(n)
    frame = pd.DataFrame({"f0": np.where(x < np.quantile(x, 0.3), np.nan, x), "z0": z, "b": b})
    frame.loc[frame.index[:40], "b"] = np.nan
    return frame, event


def _lead(frame: pd.DataFrame, how: str) -> str:
    """The per-feature sentence up to its pooling, built by hand from the case: the rows with a
    value drawn in `f0`'s model (its own blanks or a covariate's) as a percentage of the rows, and
    m by the rule over them (at least 20, at least that percentage)."""
    pct = round(100.0 * int(frame.isna().any(axis=1).sum()) / len(frame), 1)
    m, floor = max(20, math.ceil(pct)), 20
    return (f"Missing values were multiply imputed one study factor at a time, each from its own "
            f"model given the outcome and the covariates, the terms of its test ({how}); m = {m} "
            f"imputations, at least {floor} and at least the {pct:g}% of rows with an imputed "
            f"value in the most incomplete study factor's model (White, Royston & Wood 2011); the "
            f"estimates were pooled by Rubin's rules")


def test_10_a_two_valued_covariates_blanks_are_said_drawn_by_logistic_regression(monkeypatch):
    """FWMI-V1: `b` (0/1) is blank on 40 rows beside the censored `f0`. The sampler is handed `b`
    as two-valued (a logistic draw) and `f0` as censored, and the methods sentence says exactly
    those two models, no normal linear one (on the base it said "from a normal linear model" alone
    of the chained copies, and "any other blank" where none was). With `z0` blank too, a number,
    the sentence adds it, from a normal linear model; the models' parameter draws are said
    approximate for the censored-normal and logistic models, exact for the normal linear one
    (mice's ``norm``). The table's caption names `f0`'s own model alone."""
    from turbotab.core.methods import featurewise_mi as fm

    seen = _spy_kinds(monkeypatch)
    frame, event = _two_valued_panel()
    table, copies = fm.pooled_featurewise_table(frame, event, "binary", ["f0"],
                                                censored=limits_of(frame, ["f0"]), seed=2)
    assert seen[-1] == [("b", "binary"), ("f0", "censored")]
    sentence = table.info["missing"]["sentence"]
    assert sentence.startswith(_lead(frame, (
        "values below a detection limit from a censored-normal (Tobit) model and a two-valued "
        "column's blanks from a logistic regression, each model's parameters drawn from their "
        "approximate posterior"))), sentence
    assert "normal linear" not in sentence
    assert table.info["caption"].startswith(
        f"Multiple imputation, m = {copies.m}, one study factor at a time: each study factor's "
        f"blanks were drawn from its own model given the outcome and the covariates (below a "
        f"detection limit from a censored-normal (Tobit) model), the covariates' blanks drawn with "
        f"it by chained equations;")
    frame.loc[frame.index[50:70], "z0"] = np.nan
    table, _ = fm.pooled_featurewise_table(frame, event, "binary", ["f0"],
                                           censored=limits_of(frame, ["f0"]), seed=2)
    assert seen[-1] == [("b", "binary"), ("f0", "censored"), ("z0", "numeric")]
    assert table.info["missing"]["sentence"].startswith(_lead(frame, (
        "values below a detection limit from a censored-normal (Tobit) model, a two-valued "
        "column's blanks from a logistic regression and any other blank from a normal linear "
        "model, each model's parameters drawn from their posterior, approximately for the "
        "censored-normal and logistic models")))


def test_10_a_two_valued_study_factor_is_said_drawn_by_logistic_regression(monkeypatch):
    """FWMI-V1 for the study factors' own blanks: `s` (0/1) blank on 30 rows is drawn as
    two-valued, so the caption and the sentence say a logistic regression for it, and a number
    drawn alone (`g`) a normal linear model; where only numbers are drawn the sentence says no
    censored-normal or logistic model and no "other" blank."""
    from turbotab.core.methods import featurewise_mi as fm

    seen = _spy_kinds(monkeypatch)
    rng = np.random.default_rng(9)
    n = 150
    event = (rng.random(n) < 0.5).astype(float)
    s = (rng.random(n) < 0.3 + 0.3 * event).astype(float)
    g = 5 + 0.5 * event + rng.standard_normal(n)
    frame = pd.DataFrame({"s": s, "g": g})
    frame.loc[frame.index[:30], "s"] = np.nan
    frame.loc[frame.index[100:120], "g"] = np.nan
    table, _ = fm.pooled_featurewise_table(frame, event, "binary", ["s", "g"], censored={}, m=20)
    assert sorted(map(tuple, (pair for call in seen for pair in call))) == [
        ("g", "numeric"), ("s", "binary")]
    assert ("(a two-valued study factor's from a logistic regression, otherwise from a normal "
            "linear model)") in table.info["caption"]
    pct = round(100.0 * 30 / n, 1)
    assert table.info["missing"]["sentence"].startswith(
        f"Missing values were multiply imputed one study factor at a time, each from its own model "
        f"given the outcome, the terms of its test (a two-valued column's blanks from a logistic "
        f"regression and any other blank from a normal linear model, each model's parameters "
        f"drawn from their posterior, approximately for the logistic model); m = 20 imputations, "
        f"at least 20 and at least the {pct:g}% of rows")
    table, _ = fm.pooled_featurewise_table(frame[["g"]], event, "binary", ["g"], censored={}, m=20)
    sentence = table.info["missing"]["sentence"]
    assert ("the terms of its test (each blank from a normal linear model, its parameters drawn "
            "from their posterior); m = 20") in sentence
    assert "Tobit" not in sentence and "logistic" not in sentence and "other" not in sentence


def test_10_the_sentences_floor_of_m_is_the_answers_where_it_is_the_larger():
    """FWMI-V7: the sentence's "at least" is the floor the engine's rule used, the answer's m
    where it is larger than the rule's 20 (BLUEPRINT §12 ruling 4): asked 30 on a panel whose most
    incomplete model has 20% of its rows drawn, m = 30 and the sentence says at least 30; asked
    nothing, at least 20."""
    from turbotab.core.methods import featurewise_mi as fm

    frame, event = censored_panel(seed=5, shares=(0.2,), deltas=(0.5,))
    for asked, m in ((30, 30), (20, 20)):
        table, copies = fm.pooled_featurewise_table(frame, event, "binary", ["f0"],
                                                    censored=limits_of(frame, ["f0"]),
                                                    asked=asked, seed=1)
        assert copies.m == m
        assert f"m = {m} imputations, at least {asked} and at least the 20% of rows" in (
            table.info["missing"]["sentence"])


def _budget_words(factors: int, m: int, cap: int, sweeps: int = 1) -> str:
    """The budget's refusal, built from the case's counts: the study factors drawn × m (× each
    chain's sweeps) against the cap."""
    each = f" × {sweeps} sweeps of chained equations" if sweeps > 1 else ""
    noun = "study factor" if factors == 1 else "study factors"
    return (f"Feature-wise regression is pooled over multiple imputations drawn one study factor "
            f"at a time: here {factors} {noun} × {m} imputations{each} make "
            f"{factors * m * sweeps:,} draws, more than the {cap:,} one table may draw, a limit "
            f"that bounds how long its fit takes.")


def _fit_entry(tmp_path, frame: pd.DataFrame, st) -> dict:
    """The fit's feature-wise entry for ``frame`` under ``st``, every row analyzed."""
    from turbotab.core.stages.modeling import design_stage, fit_stage
    from turbotab.core.tests import modeling_fixtures as mf

    paths = mf.ingest_frame(frame, tmp_path)
    split = mf.split_bundle(np.arange(len(frame)), holdout=0.0, seed=st.split.seed)
    info = mf.target_info(st.task, st.target)
    design = design_stage(mf.context(st, {"split": split, "target_info": info}, paths))
    fit = fit_stage(mf.context(st, {"design": design, "split": split, "target_info": info},
                               paths))
    return next(m for m in fit.data["models"] if m["family"] == "featurewise")


def _decide(frame: pd.DataFrame, st) -> dict:
    return {"state": st, "columns": list(frame.columns), "store": lambda: _Store(frame),
            "analyzed": lambda: np.arange(len(frame))}


MI20 = {"strategy": "multiple_imputation", "m": 20}


def _sites(clustered: bool) -> tuple[pd.DataFrame, Any]:
    """``estimand_fixtures.cohort(300)``, `fiber` (the study factor) blank on 90 rows; with
    ``clustered`` its rows grouped three to a `site` by the cluster answer, else no site."""
    from turbotab.core import decisions as d
    from turbotab.core.tests.acceptance import estimand_fixtures as ef

    frame = ef.cohort(300, seed=5)
    frame.loc[frame.index[:90], "fiber"] = np.nan
    if not clustered:
        return frame, ef.state(target="glucose", task="regression", measure="mean_difference",
                               models=["featurewise"], missing=MI20)
    frame["site"] = np.repeat(np.arange(100), 3)
    return frame, ef.state(target="glucose", task="regression", measure="mean_difference",
                           roles={**ef.ROLES, "site": "cluster"}, models=["featurewise"],
                           missing=MI20,
                           clusters=d.ClusterSpec(column="site", adjust="cluster_only"))


def test_11_rows_the_cluster_answer_groups_are_counted_as_the_fit_draws_them(tmp_path,
                                                                             monkeypatch):
    """FWMI-V2(a): `fiber` blank on 90 of 300 rows, so m = 30 (the rule over 30% of the rows).
    Grouped three to a `site` by the cluster answer, the fit clusters its intervals by `site` and
    draws `fiber`'s copies by chained equations, ten sweeps (the engine's count): 300 draws. At a
    cap of 100 the fit refuses, and so now does each check before the lock, in the same words (on
    the base they passed: they read the grain answer alone, one row per unit here). With no site,
    one fit of `fiber`'s own model makes its 30 draws: the checks are silent and the fit pools."""
    from turbotab.core import decisions as d
    from turbotab.core.methods import featurewise_mi as fm

    monkeypatch.setattr(fm, "MAX_FEATURE_COPIES", 100)
    frame, st = _sites(clustered=True)
    words = _budget_words(1, 30, 100, sweeps=10)
    for decision in (d.SelectModels(models=["featurewise"]), d.SetMissing(**MI20), d.LockPlan()):
        with pytest.raises(d.Refusal) as refused:
            d.validate(decision, _decide(frame, st))
        assert (refused.value.code, refused.value.message) == (fm.UNPOOLED, words)
    entry = _fit_entry(tmp_path / "site", frame, st)
    assert entry["inference"]["refused"] == words
    frame, st = _sites(clustered=False)
    for decision in (d.SelectModels(models=["featurewise"]), d.SetMissing(**MI20), d.LockPlan()):
        d.validate(decision, _decide(frame, st))
    entry = _fit_entry(tmp_path / "none", frame, st)
    assert entry["inference"]["refused"] is None and entry["coefficients"]


def test_11_rows_the_grain_says_repeat_but_that_do_not_are_counted_as_independent(tmp_path,
                                                                                monkeypatch):
    """FWMI-V2(a), the other way: the grain answer says rows repeat by `pid`, but no `pid` repeats
    among the analyzed rows (as after each unit's rows are combined into one), so the fit's rows do
    not cluster and `fiber`'s 30 copies come from one fit of its model. At a cap of 100 the checks
    before the lock now pass, as the fit pools (on the base they counted ten sweeps from the grain
    answer, 300 draws, and refused)."""
    from turbotab.core import decisions as d
    from turbotab.core.methods import featurewise_mi as fm

    monkeypatch.setattr(fm, "MAX_FEATURE_COPIES", 100)
    frame, st = _sites(clustered=False)
    st = st.model_copy(update={"grain": d.GrainSpec(grain="repeated", id_column="pid")})
    for decision in (d.SelectModels(models=["featurewise"]), d.SetMissing(**MI20), d.LockPlan()):
        d.validate(decision, _decide(frame, st))
    entry = _fit_entry(tmp_path, frame, st)
    assert entry["inference"]["refused"] is None and entry["coefficients"]


def test_11_a_number_declared_as_codes_among_the_covariates_with_blanks_is_refused_before_the_lock(
        tmp_path):
    """FWMI-V2(b): `activity` (whole numbers) declared as codes is a category to the design
    (one-hot encoded), so with blanks on 30 rows the fit refuses the table in the category's words;
    each check before the lock now refuses it in the same words (on the base they read text alone
    as a category, and passed). Declared as amounts it is a number the copies draw: every check is
    silent."""
    from turbotab.core import decisions as d
    from turbotab.core.methods import featurewise_mi as fm
    from turbotab.core.tests.acceptance import estimand_fixtures as ef

    frame = ef.cohort(300, seed=5)
    frame.loc[frame.index[:30], "activity"] = np.nan
    words = CATEGORY_WORDS.replace("`sex`", "`activity`")
    for reading, refuses in (("code", True), ("amount", False)):
        st = ef.state(target="glucose", task="regression", measure="mean_difference",
                      models=["featurewise"], missing=MI20,
                      amounts={"code_or_count:smoking": "amount",
                               "code_or_count:activity": reading})
        for decision in (d.SelectModels(models=["featurewise"]), d.SetMissing(**MI20),
                         d.LockPlan()):
            if refuses:
                with pytest.raises(d.Refusal) as refused:
                    d.validate(decision, _decide(frame, st))
                assert (refused.value.code, refused.value.message) == (fm.UNPOOLED, words)
            else:
                d.validate(decision, _decide(frame, st))
        if refuses:
            entry = _fit_entry(tmp_path / reading, frame, st)
            assert entry["inference"]["refused"] == words


COMBAT_WORDS = ("Feature-wise regression is pooled over multiple imputations one study factor at a "
                "time, and ComBat adjusts every feature together, so no study factor's copies can "
                "be drawn and adjusted on their own.")


def test_12_combat_over_columns_with_blanks_is_refused_and_silent_without_blanks_or_combat():
    """FWMI-V3(1): reference ComBat adjusts the study factors together, so a blank in any of them
    is refused before any copy is drawn, complete cases (the rows that hold every value, counted
    here) and another family its ways forward. With no blank the pipeline is fitted, nothing
    drawn; without ComBat the same blanks are the ones drawn."""
    from sklearn.pipeline import Pipeline

    from turbotab.core.methods import featurewise_mi as fm
    from turbotab.core.methods.batch import ReferenceComBat
    from turbotab.core.models.featurewise import FeatureWiseTests

    frame, event = censored_panel(seed=11)
    features = [f"f{j}" for j in range(6)]
    frame["batch"] = np.where(np.arange(len(frame)) % 2 == 0, "B1", "B2")

    def template(combat: bool) -> Pipeline:
        model = ("model", FeatureWiseTests(task="binary", features=features))
        return Pipeline([("batch", ReferenceComBat(features, "batch", drop=True)), model]
                        if combat else [model])

    with pytest.raises(fm.Unpooled) as refused:
        fm.prepare(template(True), frame, event)
    whole = int(frame.notna().all(axis=1).sum())
    assert refused.value.reason == COMBAT_WORDS
    assert [e["label"] for e in refused.value.exits] == [
        f"Complete cases ({whole} of the 100 rows)", "Choose another model family"]
    rng = np.random.default_rng(0)
    full = frame.copy()
    for f in features:
        full[f] = full[f].fillna(pd.Series(rng.normal(9, 0.5, len(full)), index=full.index))
    prepared = fm.prepare(template(True), full, event)
    assert prepared.drawn == []
    prepared = fm.prepare(template(False), frame.drop(columns=["batch"]), event)
    assert prepared.drawn == features


def test_12_a_spline_of_a_covariate_with_blanks_is_refused_and_silent_where_another_is_blank(
        tmp_path):
    """FWMI-V3(2): a declared 4-knot spline of `age` (a covariate) forms its two nonlinear columns
    from `age` (`age'` and `age''`, Harrell's restricted cubic spline: knots − 2 of them), so with
    `age` blank on 30 rows those columns hold blanks no copy draws: the fit refuses the table
    naming them. With `bmi` blank instead the spline's columns are whole: the table is pooled."""
    from turbotab.core import decisions as d
    from turbotab.core.tests.acceptance import estimand_fixtures as ef

    st = ef.state(target="glucose", task="regression", measure="mean_difference",
                  models=["featurewise"], missing=MI20,
                  exposure_forms={"age": d.ExposureFormSpec(form="spline", knots=4)})
    formed = ["age" + "'" * j for j in range(1, 4 - 1)]
    for blank, refuses in (("age", True), ("bmi", False)):
        frame = ef.cohort(300, seed=5)
        frame.loc[frame.index[:30], blank] = np.nan
        entry = _fit_entry(tmp_path / blank, frame, st)
        if refuses:
            assert entry["inference"]["refused"] == (
                f"Feature-wise regression is pooled over multiple imputations one study factor at "
                f"a time; `{formed[0]}` and `{formed[1]}` hold blanks after a step that forms them "
                f"from other columns, which these copies do not draw.")
            assert entry["inference"]["exits"][-1]["label"] == "Choose another model family"
        else:
            assert entry["inference"]["refused"] is None and entry["coefficients"]


def test_12_a_column_confirmed_as_one_value_per_unit_is_refused_on_clustered_rows(tmp_path):
    """FWMI-V3(3): rows grouped three to a `site`, `activity` blank on 30 rows. Confirmed as one
    value per `site`, its blanks would be drawn once per unit, which these copies (row by row) do
    not do: the fit refuses the table in words naming the column and the unit. Confirmed as
    changing between a unit's rows it is drawn row by row and the table is pooled."""
    from turbotab.core import decisions as d
    from turbotab.core.tests.acceptance import estimand_fixtures as ef

    frame = ef.cohort(300, seed=5)
    frame["site"] = np.repeat(np.arange(100), 3)
    frame.loc[frame.index[:30], "activity"] = np.nan
    for value in ("yes", "no"):
        st = ef.state(target="glucose", task="regression", measure="mean_difference",
                      roles={**ef.ROLES, "site": "cluster"}, models=["featurewise"],
                      missing=MI20, clusters=d.ClusterSpec(column="site", adjust="cluster_only"),
                      reading_confirmations={"time_invariant:activity": value})
        entry = _fit_entry(tmp_path / value, frame, st)
        if value == "yes":
            assert entry["inference"]["refused"] == (
                "Feature-wise regression is pooled over multiple imputations drawn one study "
                "factor at a time, row by row; `activity` is confirmed as one value per `site`, "
                "which these copies do not draw once per unit.")
            assert entry["inference"]["exits"][-1]["label"] == "Choose another model family"
        else:
            assert entry["inference"]["refused"] is None and entry["coefficients"]


def test_12_a_censored_normal_model_with_no_finite_fit_is_refused_and_silent_where_it_has_one():
    """FWMI-V3(4): the censored-normal (Tobit) likelihood has a finite maximum only where the
    detected rows' design has full column rank and does not fit the detected values exactly
    (Olsen 1978: the log-likelihood is concave in (β/σ, 1/σ), and bounded above exactly then).
    Each case is refused before any draw, in words from its counts: every detected value in rows
    of one outcome (on the base, for these values, the copies were drawn and the pooled difference
    was about 10⁵, a number no fit certifies; for others the solver's own refusal fired, so which
    happened rested on the solver), and every detected value equal (on the base a LinAlgError
    escaped). With one detected value among the other outcome's rows the model is identified, and
    drawn."""
    from turbotab.core.methods import featurewise_mi as fm
    from turbotab.core.methods.imputation import ImputationRefused

    rng = np.random.default_rng(0)
    n = 100
    event = np.r_[np.ones(58), np.zeros(42)]
    z = rng.standard_normal(n)
    constant = np.where(rng.random(n) < 0.4, np.nan, 7.0)
    separated = np.where(event == 1, 10 + rng.standard_normal(n), np.nan)

    def draw(x: np.ndarray):
        frame = pd.DataFrame({"f0": x, "z0": z})
        return fm.pooled_featurewise_table(frame, event, "binary", ["f0"],
                                           censored=limits_of(frame, ["f0"]), m=20, seed=1)

    with pytest.raises(ImputationRefused) as refused:
        draw(separated)
    assert str(refused.value) == (
        "`f0`'s censored-normal model has no finite fit on these rows: all 58 of its detected "
        "values are in rows with the same outcome, so how far below the limit the other rows lie "
        "cannot be estimated, and its values below detection cannot be drawn.")
    assert refused.value.columns == ["f0"]
    with pytest.raises(ImputationRefused) as refused:
        draw(constant)
    detected = int(np.isfinite(constant).sum())
    assert str(refused.value) == (
        f"`f0`'s censored-normal model has no finite fit on these rows: its {detected} detected "
        f"values are all equal, so their spread cannot be estimated, and its values below "
        f"detection cannot be drawn.")
    once = separated.copy()
    once[70] = 10.5  # one non-event detected
    table, copies = draw(once)
    (row,) = table.rows
    assert copies.m >= 20 and math.isfinite(row["estimate"]) and row["se"] > 0


def test_12_the_fit_refuses_the_table_beyond_the_budget_and_pools_it_at_the_budget(tmp_path,
                                                                                  monkeypatch):
    """FWMI-V3(5), through the fit: `fiber` blank on 90 of 300 rows, m = 30, one fit of its model,
    30 draws. At a cap of 29 the fit refuses the table in the budget's words, with complete cases
    (the 210 rows that hold `fiber`) and another family as ways forward; at 30 it pools it."""
    from turbotab.core.methods import featurewise_mi as fm

    frame, st = _sites(clustered=False)
    monkeypatch.setattr(fm, "MAX_FEATURE_COPIES", 29)
    entry = _fit_entry(tmp_path / "over", frame, st)
    assert entry["inference"]["refused"] == _budget_words(1, 30, 29)
    assert [e["label"] for e in entry["inference"]["exits"]] == [
        f"Complete cases ({300 - 90} of the 300 rows)", "Choose another model family"]
    assert not entry["coefficients"]
    monkeypatch.setattr(fm, "MAX_FEATURE_COPIES", 30)
    entry = _fit_entry(tmp_path / "at", frame, st)
    assert entry["inference"]["refused"] is None and entry["coefficients"]


def test_13_a_feature_wise_refusal_always_carries_a_way_forward():
    """FWMI-V6: where values below detection are named (no single fill is offered: the pooled
    route replaces it) and no row holds every value (complete cases are not offered), the
    feature-wise table's refusal offers another family and says so; where complete cases keep
    rows enough, they are its way forward, as before."""
    from turbotab.core.methods.omics import unpooled_refusal

    frame, _ = censored_panel(seed=11)
    state = plan_state(["featurewise"], MI)
    few = frame.copy()
    few.loc[few.notna().all(axis=1), "f0"] = np.nan
    reason, exits = unpooled_refusal("Feature-wise regression", state, few)
    assert exits == [{"label": "Choose another model family", "decision": None}]
    assert reason == (
        "Feature-wise regression is not pooled over multiple imputations here. Complete cases are "
        "not offered: 0 of the 100 rows hold every value, fewer than the 6 the design needs. "
        "Choose another model family.")
    reason, exits = unpooled_refusal("Feature-wise regression", state, frame)
    whole = int(frame.notna().all(axis=1).sum())
    assert [e["label"] for e in exits] == [f"Complete cases ({whole} of the 100 rows)"]
    assert reason.endswith(" Choose complete cases.")


def test_13_the_marginal_risks_refusal_offers_complete_cases_only_where_the_record_takes_them(
        tmp_path):
    """FWMI-V8: the marginal risks' refusal under multiple imputation offers complete cases only
    where taking them is accepted (``row_floor.accepted_exit``, as every set_missing refusal's
    exits are), counted on the ingested table the record counts on, no missing-values answer
    recorded yet: with `activity` blank on 40 of 300 rows they keep 260 rows and are offered;
    blank on 296, complete cases would keep 4, fewer than a fit needs (the row floor refuses them
    when taken), and are not offered."""
    from turbotab.core import decisions as d
    from turbotab.core.datastore import DataStore
    from turbotab.core.tests import modeling_fixtures as mf
    from turbotab.core.tests.acceptance import estimand_fixtures as ef

    for blanks, offered in ((40, True), (296, False)):
        frame = ef.cohort(300, seed=5)
        frame.loc[frame.index[:blanks], "activity"] = np.nan
        paths = mf.ingest_frame(frame, tmp_path / str(blanks))
        st = ef.state(target="dm", task="binary", measure="risk_difference", event="yes",
                      models=["linear"], missing=None)
        with DataStore(paths["data"], 2 << 30) as store:
            ctx = {"state": st, "columns": list(frame.columns), "store": lambda: store,
                   "analyzed": lambda: np.arange(len(frame))}
            with pytest.raises(d.Refusal) as refused:
                d.validate(d.SetMissing(strategy="multiple_imputation"), ctx)
            assert refused.value.code == "marginal_risks_unpooled"
            labels = [e["label"] for e in refused.value.exits]
            assert labels == (["Report the conditional odds ratio",
                               "Complete cases, with their assumption stated"] if offered
                              else ["Report the conditional odds ratio"])
            if not offered:
                with pytest.raises(d.Refusal):  # taken, the record refuses it
                    d.validate(d.SetMissing(strategy="complete_case"), ctx)


def _card(purpose: str, covariate: bool, family: str = "featurewise") -> dict[str, str]:
    """The design card's lines for five intensities, two blank below detection (censoring-aware)
    between PQN and the log, under multiple imputation; with ``covariate``, `age` adjusts."""
    from turbotab.core.models import get_family
    from turbotab.core.models.pipeline import DesignSpec, describe_steps

    feats = [f"mz_{j}" for j in range(5)]
    predictors = [*feats, *(["age"] if covariate else [])]
    roles = {**{c: "exposure" for c in feats}, **({"age": "covariate"} if covariate else {})}
    spec = DesignSpec(predictors=predictors, inputs=predictors, categorical=[],
                      numeric=predictors, energy=None, impute=True, roles=roles,
                      normalization={"method": "pqn_log2", "kind": "intensities",
                                     "columns": feats},
                      censored={"method": "censoring_aware", "columns": feats[:2]},
                      missing={"strategy": "multiple_imputation",
                               "below_detection": "censoring_aware"},
                      lenses=["metabolomics"], purpose=purpose)
    return {s["key"]: s["detail"] for s in describe_steps(spec, get_family(family), "binary",
                                                         purpose)}


def test_14_the_design_card_says_values_below_detection_are_drawn_where_the_tests_draw_them():
    """FWMI-V4: under inference with multiple imputation the feature-wise tests draw each value
    below detection within each imputation (``featurewise_mi.blank_pipeline`` keeps them blank
    through the detect step and the log), so the card's detect and log lines say so, given the
    outcome and, only where the design has one, the covariates; the impute line too. On the base
    they described the single in-fold fill. Under prediction, and for a family that does not draw
    its own copies (the linear model's in-fold fill serves its scores), the lines are as before."""
    drawn = ("Blanks in mz_0, mz_1 are values below detection: none is filled here; each is "
             "drawn below the column's smallest detected value within each multiple imputation, "
             "from a censored-normal (Tobit) model given the outcome{}.")
    logged = ("Takes log2 of 5 assay columns; a value below detection (a blank, or zero or less) "
              "stays blank through the log and is drawn below the limit on the log scale within "
              "each multiple imputation, given the outcome{}.")
    for covariate, given in ((True, " and the covariates"), (False, "")):
        card = _card("inference", covariate)
        assert card["detect"] == drawn.format(given)
        assert card["log"] == logged.format(given)
        assert card["impute"].endswith(
            f" each from its own model given the outcome{given}, more when more rows are "
            f"incomplete.")
    filled = ("Blanks in mz_0, mz_1 are values below detection: each becomes its expected value "
              "below the smallest detected one under a censored-normal fit, learned within each "
              "training fold without the outcome.")
    before = ("Takes log2 of 5 assay columns, after values below detection are filled, so no zero "
              "or blank reaches the log.")
    for card in (_card("prediction", True), _card("inference", True, family="linear")):
        assert (card["detect"], card["log"]) == (filled, before)
