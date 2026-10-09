"""Guarantee 1 (audit §3.1, recommendation 4): held-out rows pass through the steps fitted on the
training rows, and nothing else.

Classic's old failure was a separate path for held-out rows: they never passed through the training
transforms. In v2 one fitted pipeline scores them (``stages/modeling.py``: ``final = fit(...)`` on
the training rows, then ``score``/``predict`` on the held-out rows with the same object). These tests
pin it for every registered family, on every task it fits, under every recipe option the design
can build today:

* the held-out predictions (``metrics.predict``, what the fit stage scores and calibrates) equal
  the fitted steps applied by hand to the held-out rows, one ``transform`` at a time, then the model;
* those fitted steps are the training rows' own: each step and the model refit by hand, in order, on
  the training rows alone, predict the same held-out values;
* through the fit stage itself, the sealed held-out scores are those of the hand-applied predictions.

A new family is in the grid by registering; a new design option fails
:func:`test_every_design_option_is_in_the_grid` until it has a row here.
"""
from __future__ import annotations

import dataclasses
from typing import Callable, get_args

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from turbotab.core.contracts import contracts
from turbotab.core.models import families
from turbotab.core.models.inner_cv import fit_pipeline
from turbotab.core.models.metrics import predict
from turbotab.core.models.pipeline import DesignSpec, build_pipeline
from turbotab.core.tests import modeling_fixtures as mf

contracts()  # every family registered, the screened elastic net with the omics chain included

TOL = 1e-10
N_TRAIN, N_HOLD = 160, 60
NUTRIENTS = ["protein", "carb", "fat_total"]
FACTORS = {"protein": 4.0, "carb": 4.0, "fat_total": 9.0}
FEATURES = ["f1", "f2", "f3", "f4", "f5"]
COUNTS = ["g1", "g2", "g3", "g4"]
ITEMS = ["q1", "q2", "q3"]


def _table(seed: int = 3) -> pd.DataFrame:
    """A seeded table with every kind of column a recipe option reads: nutrients and total energy,
    blanks in numbers, codes and a two-valued number, a mass at zero, intensities with values below
    detection, counts, a batch column, scale items, and outcomes for every task."""
    rng = np.random.default_rng(seed)
    n = N_TRAIN + N_HOLD
    f = pd.DataFrame(index=pd.Index(np.arange(n) + 1000, name="row_id"))
    f["age"] = rng.uniform(20, 80, n).round()
    f["sex"] = rng.choice(["F", "M"], n)
    f["protein"] = rng.gamma(8, 10, n)
    f["carb"] = rng.gamma(10, 25, n)
    f["fat_total"] = rng.gamma(8, 9, n)
    f["kcal"] = 4 * f["protein"] + 4 * f["carb"] + 9 * f["fat_total"] + rng.normal(0, 80, n)
    f["bmi"] = np.where(rng.random(n) < 0.15, np.nan, rng.normal(27, 4, n))
    f["fiber"] = np.where(rng.random(n) < 0.15, np.nan, 0.01 * f["kcal"] + rng.normal(0, 3, n))
    f["smoker"] = np.where(rng.random(n) < 0.1, np.nan, rng.integers(0, 2, n).astype(float))
    f["region"] = np.where(rng.random(n) < 0.15, None, rng.choice(["north", "south", "east"], n))
    f["alcohol"] = np.where(rng.random(n) < 0.4, 0.0, rng.gamma(2, 6, n))
    base = rng.lognormal(3, 0.5, (n, len(FEATURES))) * rng.lognormal(0, 0.2, (n, 1))
    feats = pd.DataFrame(base, columns=FEATURES, index=f.index)
    for c in FEATURES[:2]:  # below detection: the lowest values unread
        feats.loc[feats[c] < feats[c].quantile(0.12), c] = np.nan
    f[FEATURES] = feats
    f[COUNTS] = rng.poisson(rng.lognormal(5, 0.4, (n, 1)) * [1.0, 2.0, 0.5, 1.5]).astype(float)
    f["plate"] = rng.choice(["p1", "p2", "p3"], n, p=[0.5, 0.3, 0.2])
    f[ITEMS] = rng.integers(1, 6, (n, len(ITEMS))).astype(float)
    signal = (0.04 * f["age"] + 0.02 * f["protein"] - 0.004 * f["carb"]
              + 0.3 * np.log(f["f3"]) + rng.normal(0, 1, n))
    f["y_regression"] = signal
    f["y_binary"] = (signal > signal.median()).astype(int)
    f["y_multiclass"] = pd.qcut(signal, 3, labels=["a", "b", "c"]).astype(str)
    f["y_ordinal"] = pd.qcut(signal, 3, labels=[0, 1, 2]).astype(int)
    f["time"] = rng.exponential(np.exp(-0.5 * (signal - signal.mean()))) + 0.05
    f["event"] = (rng.random(n) < 0.7).astype(int)
    f["person"] = np.arange(n) // 2  # two rows per person: the repeated-measures families' unit
    return f


def _outcome(frame: pd.DataFrame, task: str) -> np.ndarray:
    if task == "time_to_event":
        from turbotab.core.models.survival import survival_outcome

        return survival_outcome(frame["event"].to_numpy(), frame["time"].to_numpy())
    return frame[f"y_{task}"].to_numpy()


# ── the recipe options the design builds today (DesignSpec), one row each ─────

CORE = ["age", "sex", "protein", "carb", "fat_total", "kcal"]
ROLES = {"age": "covariate", "sex": "covariate", "protein": "exposure", "carb": "exposure",
         "fat_total": "exposure", "kcal": "energy", "bmi": "covariate", "fiber": "exposure",
         "smoker": "covariate", "region": "covariate", "alcohol": "exposure",
         **{c: "exposure" for c in FEATURES + COUNTS}, "plate": "covariate", "score": "covariate"}
CATEGORICAL = {"sex", "region", "plate"}


def _spec(predictors: list[str], *, inputs: list[str] | None = None, **fields) -> DesignSpec:
    inputs = list(inputs or predictors)
    return DesignSpec(predictors=list(predictors), inputs=inputs,
                      categorical=[c for c in inputs if c in CATEGORICAL],
                      numeric=[c for c in inputs if c not in CATEGORICAL],
                      energy=fields.pop("energy", None), impute=fields.pop("impute", False),
                      roles={c: ROLES[c] for c in predictors if c in ROLES}, **fields)


def _energy(method: str, **kw) -> dict:
    return {"method": method, "energy_column": "kcal", "nutrients": list(NUTRIENTS),
            "log_transform": False, "strata": None, **kw}


def _factors() -> dict:
    return {c: {"factor": v, "why": "Atwater", "unit": "g"} for c, v in FACTORS.items()}


def _forms(form: dict) -> dict:
    return {"protein": dict(form)}


def _omics(columns: list[str], **fields) -> DesignSpec:
    return _spec(["age", *columns], **fields)


def _intensities(method: str) -> dict:
    return {"method": method, "kind": "intensities", "columns": list(FEATURES)}


OPTIONS: dict[str, Callable[[], DesignSpec]] = {
    "as_given": lambda: _spec(CORE),
    "fill_median": lambda: _spec([*CORE, "bmi"], impute=True),
    "fill_with_indicators": lambda: _spec([*CORE, "bmi"], impute=True, indicators=True),
    "fill_two_valued": lambda: _spec([*CORE, "bmi", "smoker"], impute=True, two_valued=["smoker"]),
    "fill_codes": lambda: _spec([*CORE, "region"], impute=True),
    "fill_energy_aware": lambda: _spec([*CORE, "fiber"], impute=True, indicators=True,
                                       energy_fill={"energy": "kcal", "nutrients": ["fiber"]}),
    "blanks_as_a_level": lambda: _spec([*CORE, "region", "bmi"], impute=True, levels=["region"]),
    **{f"energy_{m}": (lambda m=m: _spec(CORE, energy=_energy(m)))
       for m in ("none", "standard", "residual", "residual_energy_dropped", "density_multivariate",
                 "density")},
    **{f"energy_{m}": (lambda m=m: _spec(CORE, energy=_energy(m), energy_factors=_factors()))
       for m in ("partition", "all_components")},
    "energy_residual_logged_by_stratum": lambda: _spec(
        CORE, energy=_energy("residual", log_transform=True, strata="sex")),
    "form_spline": lambda: _spec(CORE, exposure_forms=_forms({"form": "spline", "knots": 4})),
    "form_quintiles": lambda: _spec(CORE, exposure_forms=_forms({"form": "quintiles"})),
    "form_categories": lambda: _spec(CORE, exposure_forms=_forms({"form": "categories",
                                                                  "cuts": [60.0, 100.0]})),
    "form_optimal": lambda: _spec(CORE, exposure_forms=_forms({"form": "optimal"})),
    "form_zero_spline": lambda: _spec([*CORE, "alcohol"], exposure_forms={
        "alcohol": {"form": "zero_spline", "knots": 3}}),
    "form_after_energy": lambda: _spec(CORE, energy=_energy("residual"),
                                       exposure_forms=_forms({"form": "spline", "knots": 3})),
    **{f"normalize_{m}": (lambda m=m: _omics(FEATURES[2:], normalization={
        **_intensities(m), "columns": FEATURES[2:]}))
       for m in ("pqn_log2", "qc_pqn_log2", "log2", "declared_normalized")},
    **{f"normalize_{m}": (lambda m=m: _omics(COUNTS, normalization={
        "method": m, "kind": "counts", "columns": list(COUNTS)}))
       for m in ("log_cpm_tmm", "log_cpm")},
    **{f"detect_{m}": (lambda m=m: _omics(FEATURES, censored={"method": m,
                                                               "columns": FEATURES[:2]}))
       for m in ("half_minimum", "censoring_aware", "qrilc")},
    **{f"detect_{m}_between_pqn_and_log": (lambda m=m: _omics(
        FEATURES, normalization=_intensities("pqn_log2"),
        censored={"method": m, "columns": FEATURES[:2]}))
       for m in ("half_minimum", "censoring_aware", "qrilc")},
    "batch_reference_combat": lambda: _omics(FEATURES[2:], inputs=["age", *FEATURES[2:], "plate"],
                                             batch={"column": "plate", "method": "reference_combat",
                                                    "columns": FEATURES[2:], "drop": True}),
    "d_ratio_filter": lambda: _omics(FEATURES, impute=True, d_ratio={
        "columns": list(FEATURES), "qc_sd": {c: 0.3 for c in FEATURES}, "threshold": 0.5}),
    "scale_score": lambda: _spec([*CORE, *ITEMS], scales=[{
        "name": "score", "items": list(ITEMS), "reverse": [ITEMS[0]], "low": 1, "high": 5,
        "scoring": "sum", "role": "covariate"}]),
    **{f"lever_forms_{m}": (lambda m=m: _spec(CORE, levers={"forms": m, "variance_filter": "none",
                                                             "keep": None, "imbalance": "none"}))
       for m in ("rule", "inner_cv")},
    "lever_filter_near_zero": lambda: _spec(CORE, levers={
        "forms": "none", "variance_filter": "near_zero", "keep": None, "imbalance": "none"}),
    "lever_filter_top": lambda: _spec(CORE, levers={
        "forms": "none", "variance_filter": "top", "keep": 4, "imbalance": "none"}),
    **{f"lever_imbalance_{m}": (lambda m=m: _spec(CORE, levers={
        "forms": "none", "variance_filter": "none", "keep": None, "imbalance": m}))
       for m in ("weights", "undersample", "oversample")},
    **{f"select_{m}": (lambda m=m: _spec(CORE, selection={
        "method": m, "where": "in_fold", "keep": 3, "threshold": None, "q": 2,
        "pre_selected": None, "sensitivity": False}))
       for m in ("elastic_net", "stability", "screening", "stepwise", "univariable", "vip")},
}

# The DesignSpec fields an option row sets, and those that name the table rather than a step.
COVERED = {"impute", "indicators", "two_valued", "energy_fill", "levels", "energy", "energy_factors",
           "exposure_forms", "normalization", "censored", "batch", "d_ratio", "scales", "levers",
           "selection", "categorical"}
NOT_A_STEP = {"predictors", "inputs", "numeric", "roles", "lenses",
              # the missing-values answer as recorded: its single fill is ``impute`` (above); its
              # multiple imputation pools completed copies of the table, each fit by this pipeline
              "missing"}


def _applies(option: str, task: str) -> bool:
    """An option the design builds for this task (the decision validators refuse the rest)."""
    if option.startswith("lever_imbalance"):
        return task == "binary"  # an imbalance correction is for a yes/no outcome
    if option.startswith("select_"):
        from turbotab.core.models.variable_selection import SUPPORTED_TASKS

        return task in SUPPORTED_TASKS
    if option.startswith("lever_forms") or option == "form_optimal":
        return task in ("regression", "binary")  # forms chosen on a number or a yes/no outcome
    return True


# Every option on each family's first task; on its other tasks, the options whose steps or model
# read the outcome (so a task changes what they fit), with the plain design and the fill.
BY_OUTCOME = ("as_given", "fill_with_indicators", "form_optimal", "lever_", "select_")


def _cases(predicting: bool) -> list:
    return [pytest.param(f.key, task, option, id=f"{f.key}-{task}-{option}")
            for f in families() if bool(f.predicts) == predicting
            for i, task in enumerate(f.tasks) for option in OPTIONS
            if _applies(option, task) and (i == 0 or option.startswith(BY_OUTCOME))]


CASES = _cases(True)


def _units(frame: pd.DataFrame) -> pd.Series:
    return frame["person"].astype(str)


def _fitted(family_key: str, task: str, spec: DesignSpec, frame: pd.DataFrame):
    from turbotab.core.models import get_family
    from turbotab.core.stages.modeling import with_units

    train, hold = frame.iloc[:N_TRAIN], frame.iloc[N_TRAIN:]
    X_train, X_hold = train[spec.inputs], hold[spec.inputs]
    y_train = _outcome(train, task)
    pipe = build_pipeline(spec, get_family(family_key), task, "prediction", N_TRAIN,
                          len(spec.predictors))
    with_units(pipe, _units(frame))
    groups = train["person"].to_numpy()
    return fit_pipeline(pipe, X_train, y_train, groups=groups), X_train, y_train, X_hold


def _by_hand(task: str, model, Xt) -> np.ndarray:
    """ŷ, a time to event's risk score, or the class probabilities: the model's own call."""
    if task in ("regression", "time_to_event"):
        return np.asarray(model.predict(Xt), dtype=float)
    return np.asarray(model.predict_proba(Xt), dtype=float)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return _table()


@pytest.mark.parametrize("family_key, task, option", CASES)
def test_held_out_predictions_are_the_training_fitted_steps_applied_by_hand(frame, family_key,
                                                                            task, option):
    spec = OPTIONS[option]()
    final, X_train, y_train, X_hold = _fitted(family_key, task, spec, frame)
    served = predict(task, final, X_hold)
    if task == "time_to_event":
        served = served[:, 0]  # the risk score; the risk by a horizon is a function of it
    # 1. the fitted steps, one transform at a time, then the model
    Xt = X_hold
    for _, step in final.steps[:-1]:
        Xt = step.transform(Xt)
    np.testing.assert_allclose(served, _by_hand(task, final.steps[-1][1], Xt), rtol=0, atol=TOL,
                               err_msg="held-out rows took a path other than the fitted steps")
    # 2. those steps are the training rows' own: each refit by hand on the training rows alone
    Xf, Xh = X_train, X_hold
    for _, step in final.steps[:-1]:
        mine = clone(step)
        Xf = mine.fit(Xf, y_train).transform(Xf)
        Xh = mine.transform(Xh)
    model = clone(final.steps[-1][1]).fit(Xf, y_train)
    np.testing.assert_allclose(served, _by_hand(task, model, Xh), rtol=0, atol=1e-8,
                               err_msg="the fitted steps are not the training rows' own")


@pytest.mark.parametrize("family_key, task", sorted({(c.values[0], c.values[1])
                                                     for c in _cases(False)}))
def test_a_family_that_only_tests_predicts_no_held_out_row(frame, family_key, task):
    """A family that only tests (WP11) has no held-out score: its fitted model has no prediction to
    give, so no path for held-out rows exists to pin."""
    final, *_ = _fitted(family_key, task, OPTIONS["as_given"](), frame)
    assert not hasattr(final, "predict") and not hasattr(final, "predict_proba")


def test_every_design_option_is_in_the_grid():
    """A field added to DesignSpec is either a recipe option with a row in OPTIONS (so the guarantee
    covers it) or named here as describing the table, not a step. Every registered family is in the
    grid with every task it fits."""
    fields = {f.name for f in dataclasses.fields(DesignSpec)}
    assert fields == COVERED | NOT_A_STEP, (
        f"unclassified design fields: {sorted(fields - COVERED - NOT_A_STEP)}; give each an "
        f"OPTIONS row (or name it in NOT_A_STEP with why)")
    set_by_rows = set()
    for make in OPTIONS.values():
        spec = make()
        empty = _spec(["age"])
        set_by_rows |= {f for f in COVERED if getattr(spec, f) != getattr(empty, f)}
    assert set_by_rows == COVERED, f"no row sets {sorted(COVERED - set_by_rows)}"
    from turbotab.core.decisions import (EnergyMethod, ImbalanceCorrection, LeverForms,
                                         SelectionMethod, VarianceFilter)

    names = set(OPTIONS)
    assert {f"energy_{m}" for m in get_args(EnergyMethod)} <= names
    assert {f"lever_forms_{m}" for m in get_args(LeverForms) if m != "none"} <= names
    assert {f"select_{m}" for m in get_args(SelectionMethod) if m != "none"} <= names
    assert {f"lever_filter_{m}" for m in get_args(VarianceFilter) if m != "none"} <= names
    assert {f"lever_imbalance_{m}" for m in get_args(ImbalanceCorrection) if m != "none"} <= names
    from turbotab.core.decisions import BelowDetection

    # "as_missing" adds no step: those values are blanks, filled as the missing-values answer says
    assert {f"detect_{m}" for m in get_args(BelowDetection) if m != "as_missing"} <= names
    from turbotab.core.methods.exposure_form import FORMS
    from turbotab.core.methods.omics import COUNT_METHODS, INTENSITY_METHODS

    assert {f"form_{m}" for m in FORMS if m != "linear"} <= names
    assert {f"normalize_{m}" for m in (*COUNT_METHODS, *INTENSITY_METHODS)} <= names
    covered = {(key, task) for key, task, _ in (c.values for c in [*CASES, *_cases(False)])}
    assert covered == {(f.key, t) for f in families() for t in f.tasks}
    first = {(key, option) for key, task, option in (c.values for c in CASES)}
    assert first >= {(f.key, o) for f in families() if f.predicts for o in OPTIONS
                     if _applies(o, f.tasks[0])}


# ── through the fit stage: the sealed held-out scores ─────────────────────────


def _stage_case(tmp_path, task: str, **slots):
    from turbotab.core.stages.modeling import design_stage, fit_stage
    from turbotab.core.tests.test_modeling import raw_inputs, train_arrays

    frame = mf.nhanes_like(300, seed=7, missing=True)
    frame["glucose_high"] = np.where(frame["glucose"] > frame["glucose"].median(), "high", "normal")
    paths = mf.ingest_frame(frame, tmp_path)
    frame = frame[frame["glucose"].notna()]
    split = mf.split_bundle(frame.index.to_numpy(), seed=7)
    target = "glucose" if task == "regression" else "glucose_high"
    st = mf.state(target=target, **slots)
    ti = mf.target_info(task, target)
    design = design_stage(mf.context(st, {"split": split, "target_info": ti}, paths))
    fit = fit_stage(mf.context(st, {"design": design, "split": split, "target_info": ti}, paths))
    train_ids, _, hold_ids = train_arrays(split)
    columns = design.objects["spec"]["inputs"]

    def read(ids):
        """The inputs as the parquet holds them, blanks as NaN; a yes/no column with blanks as
        1.0, 0.0 and NaN."""
        X = raw_inputs(paths, columns, ids)
        for c in X.columns[X.dtypes == object]:
            present = X[c].dropna()
            if present.map(lambda v: isinstance(v, (bool, np.bool_))).all():
                X[c] = np.where(X[c].isna(), np.nan, X[c].eq(True).astype(float))
            else:
                X[c] = X[c].where(X[c].notna(), np.nan)
        return X

    return (fit, read(train_ids), frame.loc[train_ids, target].to_numpy(), read(hold_ids),
            frame.loc[hold_ids, target].to_numpy())


@pytest.mark.parametrize("task", ["regression", "binary"])
def test_the_fit_stage_scores_held_out_rows_through_the_training_fitted_steps(tmp_path, task):
    """The fit stage's sealed held-out scores, for every family it fits, are the scores of the
    training-fitted steps applied by hand to the held-out rows, then the model; and those steps,
    refit by hand on the training rows alone, give the same predictions. The design fills blanks
    with indicators, adjusts energy, and (yes/no) corrects the imbalance with splines by the rule,
    or (a number) screens predictors on the outcome. Reference: scikit-learn's metrics."""
    from sklearn.metrics import (brier_score_loss, log_loss, mean_absolute_error, roc_auc_score,
                                 root_mean_squared_error)

    from turbotab.core import decisions as d
    from turbotab.core.tests.test_modeling import opened

    explore = ({"levers": d.LeverSpec(forms="rule", imbalance="weights")} if task == "binary"
               else {"selection": d.SelectionSpec(method="screening", keep=4)})
    fit, X_train, y_train, X_hold, y_hold = _stage_case(
        tmp_path, task, missing=d.MissingSpec(strategy="impute", indicators=True),
        energy_adjustment=mf.energy("residual"), **explore)
    models = opened(fit)["models"]
    assert {m["family"] for m in models} == {"linear", "elastic_net", "boosted_trees"}
    for m in models:
        final = fit.objects["fitted"][m["family"]]
        steps = [name for name, _ in final.steps[:-1]]
        assert {"impute", "energy", "lever_forms" if task == "binary" else "select"} <= set(steps)
        Xt = X_hold
        for _, step in final.steps[:-1]:
            Xt = step.transform(Xt)
        model = final.steps[-1][1]
        got = m["holdout"]
        pred = _by_hand(task, model, Xt)
        if task == "regression":
            assert got["rmse"] == pytest.approx(root_mean_squared_error(y_hold, pred), abs=1e-12)
            assert got["mae"] == pytest.approx(mean_absolute_error(y_hold, pred), abs=1e-12)
        else:
            assert list(model.classes_) == ["high", "normal"]
            positive = y_hold == "normal"
            assert got["auc"] == pytest.approx(roc_auc_score(positive, pred[:, 1]), abs=1e-12)
            assert got["brier"] == pytest.approx(brier_score_loss(positive, pred[:, 1]), abs=1e-12)
            assert got["log_loss"] == pytest.approx(log_loss(y_hold, pred), abs=1e-12)
        Xf, Xh = X_train, X_hold
        for _, step in final.steps[:-1]:
            mine = clone(step)
            Xf = mine.fit(Xf, y_train).transform(Xf)
            Xh = mine.transform(Xh)
        again = _by_hand(task, clone(model).fit(Xf, y_train), Xh)
        np.testing.assert_allclose(again, pred, rtol=0, atol=1e-8, err_msg=m["family"])
