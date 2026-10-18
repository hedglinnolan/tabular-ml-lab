"""The models area's claims (``turbotab/core/claims/models.py``; R0_PLAN.md §R0b): each rule's
known-truth tests both ways, and the split of the two rules once named ``SMALL_N``.

* **The small-sample standard** (claims backtest A-001; RECIPES §4.6, ruled 2026-10-11): below an
  effective size of 300 a searched family keeps one candidate, boosted trees and XGBoost their
  small-sample standard (written out here from the ruling); from 300 the search draws 8 Sobol
  candidates (§4.2) after the libraries' defaults (scikit-learn's own ``get_params``; XGBoost's
  documented defaults, written out). Under Estimate the design stage's plans are the same,
  through its own entry point with the purpose set (the plan reads none), and so are they for
  several classes or ordered ones under either purpose (it reads no task). For every task the
  standard acts on, under each purpose, a sentence renders, and exactly one tuning claim's regime
  holds.
* **The shelf's rows** (RECIPES §2.6, and draft 2's shelf for boosted trees): fair below 500
  rows, boosted trees poor below 200, on the boundary both ways (the rows the shelf ranks for,
  training rows under Predict and every analyzed row under Estimate, as its stage passes them).
* **The baseline verdict's floor** (A-029): the true risks of known logistic models, scored here
  by hand, so no model can beat them: a rare outcome's gain of 6% to 12% of the baseline's loss is
  better, a real gain of 0.3% of it is too little to matter; on a score better when higher (R²,
  and Harrell's C for a time to event with delayed entry), fold scores written out.
* **Rows × SMD** (A-124): floored at "could bias" whatever its cases, never capped, and the floor
  is its convention's own; cases and values built here. Under Predict the same, on the score,
  read on what the model adjusts for, which leaves out a column with the exposure role.
* **The collinear line** (C-170): a Confirm under Predict, the noticing's tier under Estimate, on
  states built here.

Then the split itself: each constant a claim names is bound, in its module's source, to
``claims.value`` of that claim; the four family sentences state the size from which their plan
searches, asked of the plan; and every tuning plan's seeds and candidate counts are the ones the
base (turbotab-next at 3655d6ad) computed, so the split moved no plan. No test here names a claim
by its id or reads a claim's value, so none can take its expected value from the claim it tests
(meta-test 2): the source check reads only the claims' ids and parameter names, and the
rendering check only whether a sentence renders, each through the area module's ``CLAIMS``.
"""
from __future__ import annotations

import ast
import importlib
import inspect

import numpy as np
import pytest

import turbotab.core.models  # noqa: F401 - registers the families
from turbotab.core import decisions as d
from turbotab.core import materiality as M
from turbotab.core import quest
from turbotab.core.claims import models as area
from turbotab.core.models.base import Situation, assessment, families, get_family
from turbotab.core.models.baseline import versus_baseline
from turbotab.core.models.tuning import fit_seed, make_plan, tuning_for

LOSS = {"regression": "mse", "binary": "log_loss", "multiclass": "log_loss", "ordinal": "rps"}
UNIT = {"regression": "units", "binary": "events", "multiclass": "rarest_class",
        "ordinal": "rarest_class"}


def plan(key: str, task: str, n_plan: int, *, plan_rows: int | None = None, split_seed: int = 7):
    return make_plan(get_family(key), task=task, loss=LOSS[task], n_plan=n_plan,
                     plan_rows=plan_rows if plan_rows is not None else 3 * n_plan,
                     unit=UNIT[task], split_seed=split_seed)


# ── the small-sample standard (A-001) ────────────────────────────────────────────────────────

# RECIPES §4.6 (ruled 2026-10-11), as each family declares its values
BOOSTED_SMALL = {"learning_rate": 0.02, "max_leaf_nodes": 8, "min_samples_leaf": "default",
                 "l2_regularization": 1.0, "max_features": 1.0, "max_iter": 100}
XGBOOST_SMALL = {"learning_rate": 0.02, "max_depth": 3, "min_child_weight": 20.0,
                 "subsample": 1.0, "colsample_bytree": 1.0, "reg_lambda": 1.0,
                 "n_estimators": 100}
# XGBoost's documented defaults (RECIPES §4.1): eta 0.3, max_depth 6, min_child_weight 1 (the
# family's unscaled "default"), subsample and colsample_bytree 1, lambda 1, 100 rounds
XGBOOST_DEFAULTS = {"learning_rate": 0.3, "max_depth": 6, "min_child_weight": "default",
                    "subsample": 1.0, "colsample_bytree": 1.0, "reg_lambda": 1.0,
                    "n_estimators": 100}


@pytest.mark.parametrize("task", ["regression", "binary"])
def test_below_an_effective_size_of_300_a_searched_family_is_fit_once_at_its_standard(task):
    """At an effective size of 299 every searched family keeps one candidate: boosted trees and
    XGBoost their small-sample standard, never stopping early even above 10,000 of the plan's
    rows, and the forest its usual standard, the one it holds from 300 (it keeps its standard)."""
    boosted = plan("boosted_trees", task, 299, plan_rows=20_000)
    xgboost = plan("xgboost", task, 299, plan_rows=20_000)
    forest = plan("random_forest", task, 299)
    for p in (boosted, xgboost, forest):
        assert len(p.candidates) == 1 and p.candidates[0].standard, p.family
    assert dict(boosted.candidates[0].values) == BOOSTED_SMALL
    assert dict(xgboost.candidates[0].values) == XGBOOST_SMALL
    assert not boosted.standard_stops and not boosted.early_stopping
    assert not xgboost.standard_stops and not xgboost.early_stopping
    assert forest.candidates[0].values == plan("random_forest", task, 300).candidates[0].values


@pytest.mark.parametrize("task", ["regression", "binary"])
def test_from_an_effective_size_of_300_the_search_runs_with_the_library_defaults_first(task):
    """At 300 the plan draws 8 Sobol candidates after the standard one (§4.2), and the standard is
    the library's defaults, not the small-sample standard: scikit-learn's own parameters for
    boosted trees, XGBoost's documented ones."""
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    library = (HistGradientBoostingRegressor() if task == "regression"
               else HistGradientBoostingClassifier()).get_params()
    boosted, xgboost, forest = (plan(k, task, 300) for k in ("boosted_trees", "xgboost",
                                                              "random_forest"))
    for p in (boosted, xgboost, forest):
        assert len(p.candidates) == 9, p.family
        assert p.candidates[0].standard and not any(c.standard for c in p.candidates[1:])
    standard = dict(boosted.candidates[0].values)
    for name in ("learning_rate", "max_leaf_nodes", "l2_regularization", "max_features",
                 "max_iter"):
        assert standard[name] == library[name], name
    assert standard["min_samples_leaf"] == "default"  # scikit-learn's own 20, resolved per fit
    assert dict(xgboost.candidates[0].values) == XGBOOST_DEFAULTS


def design_plans(task: str, size: int, purpose: str) -> dict:
    """Boosted trees' and XGBoost's plans as the design stage makes them
    (``stages.modeling.tuning_plans``) under ``purpose``, every row a training row in five folds:
    ``size`` rows of a number, ``size`` events among 1,000 rows of a yes/no outcome, or, for
    several classes or ordered ones, ``size`` rows of the rarest of three classes beside 500 of
    each other, so one outer training fold holds ⌊size·4/5⌋ units, events or units of the rarest
    class (by hand: 374 → 299, 375 → 300)."""
    from types import SimpleNamespace

    import pandas as pd

    from turbotab.core.graph import Bundle
    from turbotab.core.stages.modeling import tuning_plans

    rng = np.random.default_rng(3)
    if task == "regression":
        y = rng.normal(size=size)
    elif task == "binary":
        y = rng.permutation(np.r_[np.ones(size, dtype=int), np.zeros(1000 - size, dtype=int)])
    else:
        y = rng.permutation(np.r_[np.zeros(size, dtype=int), np.ones(500, dtype=int),
                                  np.full(500, 2, dtype=int)])
    ids = np.arange(len(y)) + 1
    table = pd.DataFrame({"y": y}, index=pd.Index(ids, name="row_id"))

    class Store:
        columns = list(table.columns)

        def materialize(self, columns, row_ids):
            return table.loc[np.asarray(row_ids), list(columns)]

    assignment = pd.DataFrame({"row_id": ids, "partition": "train", "fold": np.arange(len(y)) % 5})
    split = Bundle(data={"folds": 5, "seed": 9, "grouped_by": None},
                   frames={"assignment": assignment})
    ctx = SimpleNamespace(state=SimpleNamespace(target="y", split=None, survey=None,
                                                purpose=purpose), inputs={"split": split})
    return tuning_plans(ctx, task, [get_family("boosted_trees"), get_family("xgboost")],
                        store=Store(), train_ids=ids)


@pytest.mark.parametrize("task", ["regression", "binary"])
def test_under_estimate_below_300_the_plan_is_one_fit_at_the_small_sample_standard(task):
    """Under Estimate, one outer training fold of 299 units or events: boosted trees and XGBoost
    each keep one candidate, their small-sample standard (the ruling's values, written out), and
    the plan is the one Predict makes on the same rows."""
    plans = design_plans(task, 374, "inference")
    assert set(plans) == {"boosted_trees", "xgboost"}
    for key, small in (("boosted_trees", BOOSTED_SMALL), ("xgboost", XGBOOST_SMALL)):
        p = plans[key]
        assert p.n_plan == 299 and len(p.candidates) == 1 and p.candidates[0].standard, key
        assert dict(p.candidates[0].values) == small, key
        assert not p.standard_stops and not p.early_stopping, key
    predict = design_plans(task, 374, "prediction")
    assert {k: p.to_dict() for k, p in plans.items()} == {k: p.to_dict()
                                                          for k, p in predict.items()}


@pytest.mark.parametrize("task", ["regression", "binary"])
def test_under_estimate_from_300_the_plan_searches_after_the_library_defaults(task):
    """Under Estimate at 300: 8 Sobol candidates after the standard one, which is the library's
    defaults (scikit-learn's own parameters; XGBoost's documented ones), not the small-sample
    standard."""
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    library = (HistGradientBoostingRegressor() if task == "regression"
               else HistGradientBoostingClassifier()).get_params()
    plans = design_plans(task, 375, "inference")
    for key in ("boosted_trees", "xgboost"):
        p = plans[key]
        assert p.n_plan == 300 and len(p.candidates) == 9, key
        assert p.candidates[0].standard and not any(c.standard for c in p.candidates[1:]), key
    standard = dict(plans["boosted_trees"].candidates[0].values)
    for name in ("learning_rate", "max_leaf_nodes", "l2_regularization", "max_features",
                 "max_iter"):
        assert standard[name] == library[name], name
    assert dict(plans["xgboost"].candidates[0].values) == XGBOOST_DEFAULTS


CLASSES = [(task, purpose) for task in ("multiclass", "ordinal")
           for purpose in ("prediction", "inference")]


@pytest.mark.parametrize("task, purpose", CLASSES)
def test_for_a_class_or_ordered_outcome_below_300_the_plan_is_one_fit_at_the_standard(task,
                                                                                       purpose):
    """Three classes, or three ordered ones, the rarest 299 units in one outer training fold,
    under either purpose: boosted trees and XGBoost each keep one candidate, their small-sample
    standard (the ruling's values, written out), never stopping early, and the plan is the one the
    other purpose makes on the same rows."""
    plans = design_plans(task, 374, purpose)
    assert set(plans) == {"boosted_trees", "xgboost"}
    for key, small in (("boosted_trees", BOOSTED_SMALL), ("xgboost", XGBOOST_SMALL)):
        p = plans[key]
        assert p.n_plan == 299 and len(p.candidates) == 1 and p.candidates[0].standard, key
        assert dict(p.candidates[0].values) == small, key
        assert not p.standard_stops and not p.early_stopping, key
    other = design_plans(task, 374, "inference" if purpose == "prediction" else "prediction")
    assert {k: p.to_dict() for k, p in plans.items()} == {k: p.to_dict()
                                                          for k, p in other.items()}


@pytest.mark.parametrize("task, purpose", CLASSES)
def test_for_a_class_or_ordered_outcome_from_300_the_plan_searches_after_the_library_defaults(
        task, purpose):
    """The rarest class at 300 units in one outer training fold: 8 Sobol candidates after the
    standard one, which is the library's defaults (scikit-learn's own classifier, which boosted
    trees fit for several classes and ordered ones alike; XGBoost's documented defaults), not the
    small-sample standard."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    library = HistGradientBoostingClassifier().get_params()
    plans = design_plans(task, 375, purpose)
    for key in ("boosted_trees", "xgboost"):
        p = plans[key]
        assert p.n_plan == 300 and len(p.candidates) == 9, key
        assert p.candidates[0].standard and not any(c.standard for c in p.candidates[1:]), key
    standard = dict(plans["boosted_trees"].candidates[0].values)
    for name in ("learning_rate", "max_leaf_nodes", "l2_regularization", "max_features",
                 "max_iter"):
        assert standard[name] == library[name], name
    assert dict(plans["xgboost"].candidates[0].values) == XGBOOST_DEFAULTS


def test_a_sentence_renders_for_every_task_and_purpose_the_standard_acts_on():
    """Every task whose plan at an effective size of 299 differs from its plan at 300, in its
    settings or in how many candidates it holds, the tasks the rule below the threshold acts on
    (asked of the plan), is regression, yes/no, several classes and ordered ones, for boosted
    trees and XGBoost (a small-sample standard, and no search) and for the random forest (no
    search) alike. For each, under each purpose, with the family's own tuning declaration, the
    standard's sentence renders, falling back where its regime fails (lenient), and exactly one
    of the area's tuning claims holds there (strict), so the rule never acts where no claim
    covers it (CLAIMS_REGISTRY §2.2). The ids are read off the area's claims, never typed
    here."""
    from turbotab.core import claims

    tuning = [c for c in area.CLAIMS if c.decision == "set_tuning"]
    # the standard's own claim: the one no other tuning claim falls back to
    (first,) = [c for c in tuning if not any(t.fallback == c.id for t in tuning)]
    acted = {}
    for key in ("boosted_trees", "xgboost", "random_forest"):
        family = get_family(key)
        acted[key] = [t for t in family.tasks if tuning_for(family, t) is not None
                      and (plan(key, t, 299).candidates[0].values
                           != plan(key, t, 300).candidates[0].values
                           or len(plan(key, t, 299).candidates)
                           != len(plan(key, t, 300).candidates))]
    assert acted == {k: ["regression", "binary", "multiclass", "ordinal"] for k in acted}
    for key, tasks in acted.items():
        for task in tasks:
            decl = tuning_for(get_family(key), task)
            for purpose in ("prediction", "inference"):
                ctx = {"task": task, "purpose": purpose, "kind": decl.kind,
                       "small_standard_source": decl.small_standard_source}
                assert "effective size of" in claims.card(first.id, ctx, strict=False)
                assert "effective size of" in claims.methods(first.id, ctx, strict=False)
                held = []
                for c in tuning:
                    try:
                        claims.card(c.id, ctx, strict=True)
                    except claims.OutsideRegime:
                        continue
                    held.append(c.id)
                assert len(held) == 1, (key, task, purpose, held)


# ── the shelf's rows ─────────────────────────────────────────────────────────────────────────


def judged(key: str, rows: int):
    a = assessment(get_family(key), Situation(task="regression", purpose="prediction",
                                             n_rows=rows, n_features=4))
    return a.score, a.fit, a.concerns


def test_below_500_rows_the_shelf_rates_boosted_trees_fair_and_below_200_poor():
    """499 rows: fair at 1.0, with the small-table concern; 199: poor at 0.5. XGBoost reads boosted
    trees' rating less 1.0, at least 0.5, with the same concerns (RECIPES §2.6)."""
    score, fit, concerns = judged("boosted_trees", 499)
    assert (score, fit) == (1.0, "fair")
    assert concerns == ("499 rows is small for boosted trees: they overfit easily and their "
                        "cross-validated score is noisy.",)
    score, fit, concerns = judged("boosted_trees", 199)
    assert (score, fit) == (0.5, "poor") and "199 rows is small for boosted trees" in concerns[0]
    assert judged("xgboost", 499) == (0.5, "fair", judged("boosted_trees", 499)[2])


def test_from_500_rows_boosted_trees_carry_no_small_table_concern():
    """500 rows: good at 2.0 and no concern; 200: fair at 1.0, not poor; 2,000: 3.0."""
    assert judged("boosted_trees", 500) == (2.0, "good", ())
    assert judged("boosted_trees", 200)[:2] == (1.0, "fair")
    assert judged("boosted_trees", 2000) == (3.0, "good", ())
    assert judged("xgboost", 500) == (1.0, "good", ())


def test_below_500_rows_the_shelf_rates_a_random_forest_fair():
    """RECIPES §2.6: "fair" at 1.0 below 500, with the small-table concern."""
    assert judged("random_forest", 499) == (1.0, "fair", (
        "499 rows is small for a random forest: its cross-validated score is noisy.",))


def test_from_500_rows_a_random_forest_carries_no_small_table_concern():
    """RECIPES §2.6: 1.5 from 500 training rows, 2.0 from 2,000, good and with no concern (the
    rows the shelf ranks for: training rows under Predict, every analyzed row under Estimate)."""
    assert judged("random_forest", 500) == (1.5, "good", ())
    assert judged("random_forest", 2000) == (2.0, "good", ())


# ── the baseline verdict's floor (A-029) ─────────────────────────────────────────────────────


def _true_risks(metric: str, a: float, b: float, n: int, seed: int):
    """Five folds' scores, worked here, of the true risks of a known logistic model (logit
    a + b·x, x ~ N(0, 1)) and of the training folds' event share: the log loss, or the Brier score
    of having had the event by the horizon, which with no censoring before it is the Brier score
    at the horizon. No model predicts better than the true risks."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    p = 1.0 / (1.0 + np.exp(-(a + b * x)))
    y = (rng.random(n) < p).astype(float)
    fold = rng.permutation(n) % 5
    model, base = [], []
    for k in range(5):
        test, prior = fold == k, float(y[fold != k].mean())
        for out, q in ((model, p[test]), (base, np.full(int(test.sum()), prior))):
            if metric == "log_loss":
                out.append(float(-np.mean(y[test] * np.log(q) + (1 - y[test]) * np.log(1 - q))))
            else:
                out.append(float(np.mean((y[test] - q) ** 2)))
    return model, base


def test_a_real_gain_under_a_hundredth_of_the_baseline_loss_is_too_little_to_matter():
    """A weak predictor at 30% prevalence over 400,000 rows: its true risks gain 0.2% to 0.5% of
    the class prior's log loss (a McFadden R² of about 0.003, worked here), the interval above
    zero. The gain is real and too small to matter: no better."""
    model, base = _true_risks("log_loss", -0.85, 0.14, 400_000, 4)
    share = (float(np.mean(base)) - float(np.mean(model))) / float(np.mean(base))
    assert 0.002 <= share <= 0.005
    versus = versus_baseline("log_loss", model, base, test_share=0.25)
    assert versus.ci_low > 0 and versus.verdict == "no_better", versus.tolerance_basis


# A score better when higher, and its no-predictor baseline: R² 0, and Harrell's C 0.5 (a constant
# risk ties every pair), the score a time to event with delayed entry is read on.
HIGHER = [("r2", 0.0), ("c_index", 0.5)]


@pytest.mark.parametrize("metric, base", HIGHER)
def test_a_gain_under_a_hundredth_on_a_score_better_when_higher_is_too_little_to_matter(metric,
                                                                                       base):
    """Folds 0.004 to 0.006 above the baseline: a gain of 0.0048, its interval above zero, is no
    better."""
    gains = [0.004, 0.006, 0.005, 0.005, 0.004]
    versus = versus_baseline(metric, [base + g for g in gains], [base] * 5)
    assert versus.gain == pytest.approx(0.0048) and versus.ci_low > 0
    assert versus.verdict == "no_better"


@pytest.mark.parametrize("metric, a, b, n, seed", [
    ("log_loss", -5.3, 1.1, 60_000, 0),  # 0.9% have the event; AUC 0.79
    ("log_loss", -5.0, 1.0, 60_000, 1),  # 1.0%; AUC 0.74
    ("brier_t", -2.6, 0.9, 20_000, 2),  # 9% by the horizon; AUC 0.73
    ("brier_t", -3.4, 1.15, 20_000, 3),  # 6% by the horizon; AUC 0.78
])
def test_a_rare_outcomes_true_risks_beat_the_baseline_by_enough_to_matter(metric, a, b, n, seed):
    """The true risks of a rare outcome gain 6% to 12% of the baseline's own loss (a McFadden R²
    or a scaled Brier score of 0.06 to 0.12, worked here), though under a hundredth on the loss's
    own scale, with an interval above zero: better."""
    model, base = _true_risks(metric, a, b, n, seed)
    gain = float(np.mean(base) - np.mean(model))
    assert 0.06 <= gain / float(np.mean(base)) <= 0.12 and gain < 0.01
    versus = versus_baseline(metric, model, base, test_share=0.25)
    assert versus.ci_low > 0 and versus.verdict == "better", versus.tolerance_basis


@pytest.mark.parametrize("metric, base", HIGHER)
def test_a_gain_of_three_hundredths_on_a_score_better_when_higher_is_better(metric, base):
    gains = [0.03, 0.032, 0.029, 0.031, 0.03]
    versus = versus_baseline(metric, [base + g for g in gains], [base] * 5)
    assert versus.ci_low > 0 and versus.verdict == "better"


# ── rows × SMD (A-124) ───────────────────────────────────────────────────────────────────────


def case(m_pre: float, m_post: float, band_post: int, thread: str = "t") -> M.Case:
    return M.Case(journey="j", pipeline="p", thread=thread, alternative="a", instrument="rows_smd",
                  m_pre=m_pre, instrument_post="sensitivity", m_post=m_post, band_post=band_post)


# Six cases, each put below noise and realized below noise: as reassuring as cases can be.
QUIET = [case(0.001 * (i + 1), 0.01, 0, thread=str(i)) for i in range(6)]


def screen(value: float) -> M.Noticing:
    return M.Noticing(thread="diet-implausible-reporters", family="S5", stage="cohort",
                      subject=["kcal"], summary="s", decides_by="meaning",
                      alternative="keeping every row",
                      predicted=M.movement("rows_smd", value, "The energy screen removes rows."))


def test_rows_times_smd_is_never_calibrated_and_never_below_could_bias():
    """On the committed conventions, six cases that all held below noise do not calibrate rows ×
    SMD, and a screen whose leavers do not differ at all (0) could bias the estimate: it is never
    "doesn't change your numbers here"."""
    out = M.calibrate(QUIET, M.calibration())["instruments"]["rows_smd"]
    assert out["cases"] == 6 and out["confusion"]["matrix"][0][0] == 6
    assert not out["calibrated"] and out["uncalibrated"]
    assert M.band_of("rows_smd", 0.0) == 1
    disposition, reason, _ = M.recommend(screen(0.0))
    assert disposition == "could_bias" and "below noise" not in reason


def test_rows_times_smd_past_its_band_two_convention_is_act_on_it():
    """The floor raises, never caps: 0.25, past the band-2 starting convention of 0.2 (SURFACING
    POLICY §2.4), is act on it, and 0.1 between them could bias."""
    assert M.band_of("rows_smd", 0.25) == 2 and M.recommend(screen(0.25))[0] == "act_on_it"
    assert M.band_of("rows_smd", 0.1) == 1


def unfloored() -> dict:
    """The committed conventions with rows × SMD's floor taken off."""
    start = M.calibration()
    rows_smd = start["instruments"]["rows_smd"]
    return {**start, "instruments": {**start["instruments"], "rows_smd": {
        **rows_smd, "convention": {k: v for k, v in rows_smd["convention"].items()
                                   if k != "floored"}}}}


def test_a_proxy_whose_convention_is_not_floored_may_say_below_noise(monkeypatch):
    """The floor is the convention's own: with it taken off, the same six cases calibrate the
    proxy, and on that calibration a screen below its band-1 cut-off is band 0, "doesn't change
    your numbers here"."""
    out = M.calibrate(QUIET, unfloored())
    assert out["instruments"]["rows_smd"]["calibrated"]
    monkeypatch.setattr(M, "calibration", lambda: out)
    assert M.band_of("rows_smd", 0.0) == 0 and M.recommend(screen(0.0))[0] == "no_change"


def test_under_predict_rows_times_smd_is_never_below_could_bias_the_score():
    """Under Predict the same floor: a screen whose leavers do not differ at all (0) could bias
    the score, never "doesn't change your numbers here"."""
    disposition, reason, _ = M.recommend(screen(0.0), purpose="prediction")
    assert disposition == "could_bias" and "could bias the score" in reason
    assert "below noise" not in reason


def test_under_predict_the_floor_never_caps_and_is_its_conventions_own(monkeypatch):
    """Under Predict the floor raises, never caps (0.25 is act on it), and it is the convention's
    own: with it taken off, the six quiet cases calibrate the proxy and a screen at 0 is
    "doesn't change your numbers here", below noise in the score."""
    assert M.recommend(screen(0.25), purpose="prediction")[0] == "act_on_it"
    out = M.calibrate(QUIET, unfloored())
    assert out["instruments"]["rows_smd"]["calibrated"]
    monkeypatch.setattr(M, "calibration", lambda: out)
    disposition, reason, _ = M.recommend(screen(0.0), purpose="prediction")
    assert disposition == "no_change" and "below noise in the score" in reason


def test_under_predict_the_screen_reads_what_the_model_adjusts_for_less_the_exposure():
    """What the Predict fallback's sentences say the product is read on. A hundred of 1,000 rows
    leave; every column but sugar takes the values 0 to 9 equally often among leavers and stayers,
    so their SMD is 0, and the leavers' sugar is 10 higher. With sugar given the exposure role the
    screen's largest SMD on what the model adjusts for is 0; with sugar a covariate it is sugar's,
    the pooled-SD SMD worked here."""
    import pandas as pd

    n, gone = 1000, 100
    cycle = np.tile(np.arange(10.0), n // 10)
    leaving = np.arange(n) < gone
    sugar = cycle + 10.0 * leaving
    frame = pd.DataFrame({"SEQN": np.arange(n), "sugar": sugar, "kcal": cycle,
                          "protein": cycle, "glucose": cycle})
    rule = d.ExclusionRule(column="kcal", low=500, high=5000, reason="implausible intake")
    kept = np.flatnonzero(~leaving)
    a, b = sugar[leaving], sugar[~leaving]
    by_hand = gone / n * abs(a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    for role, want in (("exposure", 0.0), ("covariate", by_hand)):
        s = d.ProjectState(lens=["dietary"], target="glucose", purpose="prediction",
                           roles={"SEQN": "identifier", "sugar": role, "kcal": "energy",
                                  "protein": "covariate"}, exclusions=[rule])
        noticed = M.reporters_noticing(s, frame, kept, np.arange(n))
        assert noticed.predicted.value == pytest.approx(want, rel=1e-9, abs=1e-12), role
        assert ("`sugar`" in noticed.predicted.words) is (role == "covariate"), role


# ── the collinear line (C-170) ───────────────────────────────────────────────────────────────

ADJUST = ["kcal", "protein", "carb", "fat", "alcohol", "age"]


def state(**update) -> d.ProjectState:
    roles = {c: "covariate" for c in ADJUST}
    roles.update(sugar="exposure", kcal="energy")
    base = dict(lens=["dietary"], target="glucose", purpose="inference", roles=roles,
                estimand=d.EstimandSpec(exposure="sugar", measure="mean_difference"))
    return d.ProjectState(**{**base, **update})


def recommended(s: d.ProjectState) -> list[str]:
    from turbotab.core.stages.explore import _collinear_lever

    return [o.key for o in _collinear_lever(s).options if o.rung == "recommended"]


def test_under_predict_the_collinear_line_is_a_confirm_whatever_the_pair_holds():
    """Under Predict the line is a Confirm with no near dependency noticed, whether the pair holds
    what you study or two adjustment terms, and keeping both is the recommended answer."""
    s = state(purpose="prediction")
    for pairs in ([("sugar", "carb")], [("fat", "protein")], []):
        assert M.collinear_label(s, None, pairs) == quest.CONFIRM, pairs
    assert recommended(s) == ["keep_both"]


def test_the_collinear_methods_sentence_says_where_keeping_both_takes_back_the_filter():
    """The verifier's V9 on claims backtest C-170, the sentence against the code: under Predict,
    with the line's own elastic-net filter recorded, keeping both takes it back on a table with
    fewer candidate predictors than rows and leaves it on a wide one, where the width's lever
    offers the same filter (asked of the line itself); the collinear claim's methods sentence says
    both."""
    from turbotab.core import claims
    from turbotab.core.stages.explore import _collinear_lever

    now = state(purpose="prediction", selection=d.SelectionSpec(method="elastic_net"))
    after = {}
    for wide in (False, True):
        [keep] = [o for o in _collinear_lever(now, wide=wide).options if o.key == "keep_both"]
        after[wide] = (now if keep.decision is None
                       else d.fold_onto(now, d.parse_decision(keep.decision))).selection
    assert after == {False: d.SelectionSpec(method="none"),
                     True: d.SelectionSpec(method="elastic_net")}, after
    said = claims.methods(area.COLLINEAR_UNDER_PREDICTION.id, {"purpose": "prediction"})
    assert ("taking back the line's own elastic-net filter where it was recorded on a table with "
            "fewer candidate predictors than rows") in said, said
    assert "on a wide table" in said and "so it stays" in said, said


def test_under_estimate_a_pair_holding_what_you_study_is_decided_at_the_adjustment_question():
    """Under Estimate the Confirm does not leak: a pair that holds what you study is a Decide, a
    pair of adjustment terms alone For the record, and the recommended answer is the adjustment
    set's, not keeping both."""
    s = state()
    assert M.collinear_label(s, None, [("sugar", "carb")]) == quest.DECIDE
    assert M.collinear_label(s, None, [("fat", "protein")]) == quest.RECORD
    assert recommended(s) == ["adjustment"]


# ── the split: each constant reads its claim, and nothing moved ──────────────────────────────


def _assignment(module: str, name: str) -> ast.expr | None:
    """The expression ``module``'s source binds ``name`` to at top level, the last binding."""
    tree = ast.parse(inspect.getsource(importlib.import_module(module)))
    bound = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name
                                                for t in node.targets):
            bound = node.value
    return bound


def test_each_constant_takes_its_value_from_its_claim():
    """Every constant a claim of this area names is bound, in its module's source, to
    ``claims.value(<that claim's id>, <its parameter>)``: no number is typed there. The two 500s
    are read from two claims, each in its own module, and neither is the tuning's threshold."""
    read: dict[str, str] = {}
    for c in area.CLAIMS:
        for key, p in c.params.items():
            if p.constant is None:
                continue
            module, _, name = p.constant.partition(":")
            bound = _assignment(module, name)
            assert isinstance(bound, ast.Call), p.constant
            assert ast.unparse(bound.func) == "claims.value", p.constant
            assert [a.value for a in bound.args] == [c.id, key], p.constant
            read[module] = c.id
    models = "turbotab.core.models"
    assert set(read) == {f"{models}.{m}" for m in ("tuning", "boosted_trees", "forest",
                                                   "baseline")}
    assert len({read[f"{models}.{m}"] for m in ("tuning", "boosted_trees", "forest")}) == 3


def _sentence_strings(module) -> list[str]:
    """The string literals of ``module`` outside its docstrings: what it can say to a user."""
    tree = ast.parse(inspect.getsource(module))
    docstrings = {id(n.body[0].value) for n in ast.walk(tree)
                  if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef))
                  and n.body and isinstance(n.body[0], ast.Expr)
                  and isinstance(n.body[0].value, ast.Constant)}
    return [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)
            and isinstance(n.value, str) and id(n) not in docstrings]


def _search_threshold(key: str) -> int:
    """The smallest effective size at which the family's plan draws a search, asked of the plan."""
    return next(n for n in range(2, 5_000) if len(plan(key, "regression", n).candidates) > 1)


def test_the_family_sentences_state_the_search_threshold_their_plan_uses():
    """Boosted trees' and XGBoost's tuning reasons and descriptions say "from an effective size of
    N" with N the size from which their plan searches, and no literal in either module retypes
    it: the number is filled from the constant that reads its claim."""
    from turbotab.core.models import boosted_trees, xgboost_family

    for module, key in ((boosted_trees, "boosted_trees"), (xgboost_family, "xgboost")):
        n, family = _search_threshold(key), get_family(key)
        assert f"from an effective size of {n:,}, " in family.tuning.reason, family.tuning.reason
        for task in ("regression", "binary"):
            assert f"From an effective size of {n:,} (units" in family.describe(task, None)[1]
        assert not [s for s in _sentence_strings(module) if f"size of {n}" in s], module.__name__


# Computed on the base (turbotab-next at 3655d6ad), before the split, by make_plan at each size
# and split seed, with the method contracts loaded (the omics chain registers a family of its own,
# ``screened_elastic_net``): each family's plan seed at split seeds 0 and 11, the fit seed at each,
# and the candidates a plan holds at an effective size of 299, 300 and 1,000 for each task it
# tunes. A family that declares tuning later adds its row.
SEEDS = {"boosted_trees": (711232876, 2477195315), "elastic_net": (3043560225, 2366252822),
         "huber": (3508388161, 4191025961), "random_forest": (3101457763, 765780772),
         "ridge": (4060255269, 245685185), "screened_elastic_net": (1108140009, 4198822290),
         "xgboost": (2596488209, 999713740)}
FIT_SEEDS = (603630213, 2780830988)
COUNTS = {"boosted_trees": (1, 9, 17), "huber": (1, 1, 1), "random_forest": (1, 9, 9),
          "ridge": (50, 50, 50), "xgboost": (1, 9, 17)}
PATHS = {"elastic_net": {"regression": (600, 600, 600), "binary": (24, 24, 24),
                         "multiclass": (24, 24, 24), "ordinal": (24, 24, 24)},
         "screened_elastic_net": {"regression": (600, 600, 600), "binary": (24, 24, 24)}}


def test_every_tuning_plans_seeds_and_candidates_are_the_bases():
    from turbotab.core.contracts import contracts

    contracts()
    tuned = [(f, task) for f in families() for task in f.tasks if tuning_for(f, task) is not None]
    assert {f.key for f, _ in tuned} == set(SEEDS)
    for family, task in tuned:
        counts = []
        for n in (299, 300, 1000):
            for split, seed, fit in zip((0, 11), SEEDS[family.key], FIT_SEEDS):
                p = plan(family.key, task, n, split_seed=split)
                assert (p.seed, fit_seed(p)) == (seed, fit), (family.key, task, n, split)
            counts.append(len(p.candidates))
        want = PATHS[family.key][task] if family.key in PATHS else COUNTS[family.key]
        assert tuple(counts) == want, (family.key, task)


# ── the regime predicates, both ways ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("predicate, holds, fails", [
    (area.under_prediction, {"purpose": "prediction"}, {"purpose": "inference"}),
    (area.under_inference, {"purpose": "inference"}, {"purpose": "prediction"}),
    (area.standard_from_the_study,
     {"small_standard_source": "TurboTab's small-sample standard, chosen by simulation "
                               "(notes/SMALL_SAMPLE_STANDARD.md)"},
     {"small_standard_source": "scikit-learn's defaults"}),
    (area.searched_without_a_small_sample_standard,
     {"kind": "search", "small_standard_source": ""},
     {"kind": "search", "small_standard_source": "TurboTab's small-sample standard, chosen by "
                                                 "simulation (notes/SMALL_SAMPLE_STANDARD.md)"}),
    (area.searched_without_a_small_sample_standard,
     {"kind": "search", "small_standard_source": "scikit-learn's defaults"},
     {"kind": "none", "small_standard_source": ""}),
    (area.yes_no_or_numeric, {"task": "binary"}, {"task": "multiclass"}),
    (area.classes_or_ordered, {"task": "multiclass"}, {"task": "binary"}),
    (area.classes_or_ordered, {"task": "ordinal"}, {"task": "regression"}),
    (area.under_either_purpose, {"purpose": "inference"}, {"purpose": "describe"}),
    (area.under_either_purpose, {"purpose": "prediction"}, {"purpose": None}),
    (area.rated_before_any_fit, {"n_rows": 499, "n_features": 4},
     {"n_rows": 499.5, "n_features": 4}),
    (area.verdict_metric, {"metric": "brier_t"}, {"metric": "auc"}),
    (area.verdict_metric, {"metric": "c_index"}, {"metric": "accuracy"}),
    (area.rows_times_smd, {"instrument": "rows_smd"}, {"instrument": "attenuation"}),
    # R1-F's first batch (A-106, C-110)
    (area.yes_no_left_uncorrected, {"task": "binary", "imbalance": "none"},
     {"task": "binary", "imbalance": "weights"}),
    (area.yes_no_left_uncorrected, {"task": "binary"}, {"task": "regression"}),
    (area.drawn_by_row, {"state": "abandoned"}, {"state": "grouped"}),
    (area.drawn_by_row, {"state": "undetermined"}, {"state": "one_row_per_unit"}),
])
def test_each_regime_predicate_holds_where_its_words_say_and_nowhere_else(predicate, holds, fails):
    assert predicate(holds) is True and predicate(fails) is False and predicate({}) is False


def test_a_predicate_reads_an_object_as_it_reads_a_mapping():
    """A project's state, the shelf's situation and a family's tuning declaration, as the engine
    holds them: the small-sample study's standard is boosted trees' and XGBoost's declaration,
    not the forest's, which keeps its own."""
    assert area.under_prediction(state(purpose="prediction"))
    assert not area.under_prediction(state())
    assert area.rated_before_any_fit(Situation(task="binary", purpose="prediction", n_rows=499,
                                               n_features=4))
    studied = {k: area.standard_from_the_study(tuning_for(get_family(k), "binary"))
               for k in ("boosted_trees", "xgboost", "random_forest")}
    assert studied == {"boosted_trees": True, "xgboost": True, "random_forest": False}
    unstudied = {k: area.searched_without_a_small_sample_standard(tuning_for(get_family(k),
                                                                             "binary"))
                 for k in ("boosted_trees", "xgboost", "random_forest")}
    assert unstudied == {"boosted_trees": False, "xgboost": False, "random_forest": True}


def test_every_area_claim_names_its_predicates_from_this_area_module():
    for c in area.CLAIMS:
        assert c.regime.requires, c.regime.words
        for target in c.regime.requires:
            module, _, name = target.partition(":")
            assert module == area.__name__ and callable(getattr(area, name)), target
