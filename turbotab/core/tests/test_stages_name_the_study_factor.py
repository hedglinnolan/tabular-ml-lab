"""Every stage that fits a coefficient table tells it the study factor's model-matrix columns
(``effects.study_columns``), so a dependent design serves the coefficients it identifies and
refuses by row those in a dependency, and the whole table is refused only where a study factor's
column is in one (the orchestrator's ruling (1) on the swarm's tail; the verifier's T1V-1).

On the base every stage named nothing studied, so ``inference.by_identified`` refused every
dependent design whole: a covariate beside its own rescaling (`age2` = 3 × `age` + 1, which puts
the intercept in the dependency too) withheld `fiber`'s estimate at the fit, at each model of the
effects stage, at the curve of what mattered and at every stage that refits the primary.

Truth from construction: `fiber`'s estimate and standard error are statsmodels' (OLS with HC3,
Logit) on the design with `age2` left out, which spans the same columns, so a coefficient in no
dependency is the same whichever aliased column is left out. The words of a refusal are written
from the construction. The guard's other side: with `fiber`'s own double among the covariates the
fit's table is refused whole, the effects stage reports no adjusted model, and the curve of what
mattered is refused, each in the dependency's words, as on the base.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from turbotab.core import decisions as d
from turbotab.core.tests import modeling_fixtures as mf
from turbotab.core.tests.acceptance import estimand_fixtures as ef

N = 400
MODEL_2 = ["age", "sex", "smoking", "activity"]
COVARIATE = ("On these 400 rows the model matrix is singular: `age` and `age2` are fixed "
             "combinations of the other columns and the intercept, so their coefficients are not "
             "identified, nor is the intercept. No estimate, interval or p-value is reported for "
             "`age`, `age2` and the intercept; the other coefficients, which these rows identify, "
             "are estimated with `age2` left out. To report every coefficient, leave one of the "
             "dependent columns out.")
STUDIED = ("On these 400 rows the model matrix is singular: `fiber` and `fiber2` are fixed "
           "combinations of the other columns and the intercept, so their coefficients are not "
           "identified. No estimate, interval or p-value is reported; leave one of the dependent "
           "columns out.")
LEAVE_ONE_OUT = "Leave one of the dependent columns out of the model"


def _frame(copy: str) -> pd.DataFrame:
    """``ef.cohort(400)`` with `age2` = 3 × `age` + 1 (a covariate in a dependency with `age` and
    the intercept) or `fiber2` = 2 × `fiber` (the study factor in one)."""
    frame = ef.cohort(n=N, seed=7)
    if copy == "age2":
        frame["age2"] = 3 * frame["age"] + 1
    else:
        frame["fiber2"] = 2 * frame["fiber"]
    return frame


def _state(copy: str, task: str = "regression", **slots: Any) -> d.ProjectState:
    roles = {**ef.ROLES, copy: "covariate"}
    answers = {**ef.ANSWERS, copy: ef.CONFOUNDER}
    if task == "binary":
        return ef.state(target="dm", task="binary", event="yes", measure="odds_ratio",
                        roles=roles, answers=answers, **slots)
    return ef.state(target="glucose", task="regression", measure="mean_difference", roles=roles,
                    answers=answers, **slots)


def _truth(frame: pd.DataFrame, task: str, columns: list[str]) -> tuple[float, float]:
    """`fiber`'s (estimate, standard error) from statsmodels on `fiber` and ``columns``: OLS with
    HC3 for `glucose`, Logit (its inverse information) for `dm` = "yes"."""
    X = ef.design_matrix(frame, ["fiber", *columns])
    if task == "regression":
        fit = sm.OLS(frame["glucose"].to_numpy(float), X).fit(cov_type="HC3")
    else:
        fit = sm.Logit((frame["dm"] == "yes").to_numpy(float), X).fit(disp=0)
    return float(fit.params["fiber"]), float(fit.bse["fiber"])


def _rank(frame: pd.DataFrame, columns: list[str]) -> tuple[int, int]:
    """(numpy's rank, the column count) of the least-squares design on ``columns``."""
    X = ef.design_matrix(frame, columns).to_numpy(float)
    return int(np.linalg.matrix_rank(X)), X.shape[1]


@pytest.mark.parametrize("task", ["regression", "binary"])
def test_a_covariate_in_a_dependency_is_refused_by_row_and_the_study_factor_served(tmp_path,
                                                                                    task):
    """`age2` beside `age`: the fit's table serves `fiber` at statsmodels' estimate and standard
    error with `age2` left out, and refuses `age`, `age2` and the intercept by row, in words that
    name them; the effects stage's Model 2 and Model 3 report `fiber` at statsmodels' too, and its
    methods text says each reports it. On the base the fit's table was refused whole and the
    methods text said Model 2 reported none."""
    frame = _frame("age2")
    assert _rank(frame, ["fiber", *MODEL_2, "age2"]) == (6, 7)  # the premise: one dependency
    out = ef.run(frame, tmp_path, _state("age2", task))
    [model] = out["fit_raw"]["models"]
    assert model["inference"]["refused"] is None
    rows = {r["feature"]: r for r in model["coefficients"]}
    for name in ("(intercept)", "age", "age2"):
        assert rows[name]["estimate"] is None and rows[name]["ci_low"] is None, name
    estimate, se = _truth(frame, task, MODEL_2)
    assert rows["fiber"]["estimate"] == pytest.approx(estimate, rel=1e-8)
    assert rows["fiber"]["se"] == pytest.approx(se, rel=1e-6)
    assert rows["fiber"]["ci_low"] is not None
    assert COVARIATE in model["concerns"]
    sequence = ef.sequence(out["effects"])
    for key, adjust in (("model_2", MODEL_2), ("model_3", [*MODEL_2, "bmi"])):
        assert sequence[key]["inference"]["refused"] is None, key
        (row,) = sequence[key]["effects"]
        assert row["estimate"] == pytest.approx(_truth(frame, task, adjust)[0], rel=1e-8), key
    methods = out["effects"]["methods"]
    assert methods.startswith(
        "The estimate of `fiber` is reported across a declared sequence of models fit on all 400 "
        "analyzed rows: unadjusted; Model 2, the primary, adjusted for `age`, `sex`, `smoking`, "
        "`activity` and `age2`; Model 3, further adjusted for `bmi`, a possible mediator, so not "
        "a total effect.")
    assert "reports none" not in methods and "singular" not in methods


def test_the_study_factor_in_a_dependency_refuses_the_whole_table_as_before(tmp_path):
    """The guard's other side: `fiber2` = 2 × `fiber` among the covariates, so the study factor is
    in the dependency. The fit's table is refused whole, with no coefficient, in the dependency's
    words and its one way forward; the effects stage reports `fiber` from the unadjusted model
    alone and says Model 2 and Model 3 report none in those words (as on the base)."""
    frame = _frame("fiber2")
    out = ef.run(frame, tmp_path, _state("fiber2"))
    [model] = out["fit_raw"]["models"]
    assert model["coefficients"] == []
    assert model["inference"]["refused"] == STUDIED
    assert [e["label"] for e in model["inference"]["exits"]] == [LEAVE_ONE_OUT]
    sequence = ef.sequence(out["effects"])
    assert sequence["model_2"]["effects"] is None and sequence["model_3"]["effects"] is None
    assert out["effects"]["methods"] == (
        f"The estimate of `fiber` is reported from the unadjusted model alone, fit on all 400 "
        f"analyzed rows. Model 2, the primary, adjusted for `age`, `sex`, `smoking`, `activity` "
        f"and `fiber2`, reports none: {STUDIED} Model 3, further adjusted for `bmi`, reports "
        f"none: {STUDIED}")


@pytest.mark.parametrize("form", ["quintiles", "spline"])
def test_the_exposures_form_tests_beside_a_covariates_dependency_are_those_without_it(tmp_path,
                                                                                      form):
    """`fiber` in quintiles (the global test and the trend test, a refit with the quintiles
    scored by their medians) or a spline of 4 knots (the test of association, of nonlinearity,
    and the companion quintiles and their trend, refits), beside `age2`: each test the fit's
    table carries is the one it carries with `age2` left out, which spans the same columns, its
    statistic, p-value and estimate to 1e-9. On the base the table was refused whole and carried
    none; told nothing studied, each refit would be refused whole and its test not computed."""
    spec = d.ExposureFormSpec(form=form, **({"knots": 4} if form == "spline" else {}))
    tests = {}
    for copy in ("age2", None):
        frame = _frame("age2").drop(columns=["bmi"])
        roles = {k: v for k, v in {**ef.ROLES, "age2": "covariate"}.items() if k != "bmi"}
        answers = {k: v for k, v in {**ef.ANSWERS, "age2": ef.CONFOUNDER}.items() if k != "bmi"}
        if copy is None:
            frame = frame.drop(columns=["age2"])
            roles.pop("age2"), answers.pop("age2")
        st = ef.state(target="glucose", task="regression", measure="mean_difference", roles=roles,
                      answers=answers, exposure_forms={"fiber": spec})
        out = ef.run(frame, tmp_path / str(copy), st)
        [model] = out["fit_raw"]["models"]
        assert model["inference"]["refused"] is None
        tests[copy] = {t["test"]: t for t in model["exposure_tests"]}
    expected = ({"global", "trend"} if form == "quintiles" else
                {"overall", "nonlinear", "companion_q2", "companion_q3", "companion_q4",
                 "companion_q5", "companion_trend"})
    assert set(tests["age2"]) == set(tests[None]) == expected
    for name, test in tests["age2"].items():
        for key in ("statistic", "p", "estimate"):
            if tests[None][name].get(key) is not None:
                assert test[key] == pytest.approx(tests[None][name][key], rel=1e-9), (name, key)


def test_a_confounder_in_a_declared_form_inside_a_dependency_refuses_the_table_whole(tmp_path):
    """The guard's firing side for a declared form's terms (``effects.study_columns``): `age` in a
    spline of 4 knots beside `age2`, whose straight-line term `age2` copies. The fit's form tests
    read `age`'s terms from the table, so the table is refused whole in the dependency's words
    (as on the base) rather than served with `age`'s straight-line term empty, where its test of
    association would be missing beside a concern that the table gives no Wald covariance."""
    frame = _frame("age2").drop(columns=["bmi"])
    roles = {k: v for k, v in {**ef.ROLES, "age2": "covariate"}.items() if k != "bmi"}
    answers = {k: v for k, v in {**ef.ANSWERS, "age2": ef.CONFOUNDER}.items() if k != "bmi"}
    st = ef.state(target="glucose", task="regression", measure="mean_difference", roles=roles,
                  answers=answers,
                  exposure_forms={"age": d.ExposureFormSpec(form="spline", knots=4)})
    [model] = ef.run(frame, tmp_path, st)["fit_raw"]["models"]
    assert model["coefficients"] == [] and model["exposure_tests"] == []
    assert model["inference"]["refused"] == (
        "On these 400 rows the model matrix is singular: `age` and `age2` are fixed combinations "
        "of the other columns and the intercept, so their coefficients are not identified. No "
        "estimate, interval or p-value is reported; leave one of the dependent columns out.")
    assert not any("Wald covariance" in c for c in model["concerns"])


def _mattered(frame: pd.DataFrame, st: d.ProjectState, tmp_path: Path) -> dict[str, Any]:
    """The design, fit and what-mattered stages over these rows, as a worker runs them."""
    from turbotab.core.mattered import mattered_stage
    from turbotab.core.stages.modeling import design_stage, fit_stage

    paths = mf.ingest_frame(frame, tmp_path)
    split = mf.split_bundle(np.arange(len(frame)), holdout=0.0, seed=st.split.seed)
    inputs = {"split": split, "target_info": mf.target_info(st.task, st.target)}
    inputs["design"] = design_stage(mf.context(st, inputs, paths))
    inputs["fit"] = fit_stage(mf.context(st, inputs, paths))
    inputs["ingest"] = {}
    found = mattered_stage(mf.context(st, inputs, paths))
    return getattr(found, "data", found)


@pytest.mark.parametrize("copy", ["age2", "fiber2"])
def test_the_curve_of_what_mattered_follows_the_fits_table(tmp_path, copy):
    """What mattered follows the fit's row of `fiber` (``mattered.reported_of``). Beside `age2` it
    follows statsmodels' estimate with `age2` left out, where on the base it was refused in the
    dependency's words; beside `fiber2` it is refused in them, with the way forward that leaves one
    out, as on the base. `bmi` is left out (unknown timing, it would only add a Model 3)."""
    frame = _frame(copy).drop(columns=["bmi"])
    roles = {k: v for k, v in {**ef.ROLES, copy: "covariate"}.items() if k != "bmi"}
    answers = {k: v for k, v in {**ef.ANSWERS, copy: ef.CONFOUNDER}.items() if k != "bmi"}
    st = ef.state(target="glucose", task="regression", measure="mean_difference", roles=roles,
                  answers=answers, plan_locked=True)
    found = _mattered(frame, st, tmp_path)
    if copy == "fiber2":
        assert found["curve"] is None
        assert found["refused"]["message"] == (
            f"The linear model reports no estimate here, so the curve has no primary to follow: "
            f"{STUDIED}")
        assert [e["label"] for e in found["refused"]["exits"]] == [LEAVE_ONE_OUT]
        return
    assert found["refused"] is None
    estimate, se = _truth(frame, "regression", MODEL_2)
    assert found["reported"]["estimate"] == pytest.approx(estimate, rel=1e-8)
    assert found["reported"]["se"] == pytest.approx(se, rel=1e-6)


# ── the stages that refit the primary ────────────────────────────────────────


def _inputs(frame: pd.DataFrame, st: d.ProjectState, tmp_path: Path) -> tuple[dict, dict]:
    from turbotab.core.stages.modeling import design_stage

    paths = mf.ingest_frame(frame, tmp_path)
    split = mf.split_bundle(np.arange(len(frame)), holdout=0.0, seed=st.split.seed)
    inputs = {"split": split, "target_info": mf.target_info(st.task, st.target)}
    inputs["design"] = design_stage(mf.context(st, inputs, paths))
    return inputs, paths


def test_the_secondary_models_serve_the_study_factor_beside_a_covariates_dependency(tmp_path):
    """The secondary stage's primary and its model further adjusted for `bmi`, each beside `age2`:
    `fiber` at statsmodels' estimate with `age2` left out, the dependency's columns refused by row
    in its words, and the methods text says `fiber` is reported from each. On the base both were
    refused whole."""
    from turbotab.core.stages.secondary import secondary_stage

    frame = _frame("age2")
    st = _state("age2")
    inputs, paths = _inputs(frame, st, tmp_path)
    data = secondary_stage(mf.context(st, inputs, paths)).data
    [linear] = data["families"]
    fits = {f["label"]: f for f in linear["fits"]}
    for label, adjust in (("Primary", MODEL_2), ("Further adjusted for `bmi`", [*MODEL_2, "bmi"])):
        fit = fits[label]
        assert fit["inference"]["refused"] is None, label
        (row,) = fit["coefficients"]
        assert row["feature"] == "fiber"
        assert row["estimate"] == pytest.approx(_truth(frame, "regression", adjust)[0],
                                                rel=1e-8), label
        assert COVARIATE in fit["concerns"], label
    assert data["methods"].endswith("; the estimate of `fiber` is reported from each.")


def test_a_sensitivity_analysis_serves_the_study_factor_beside_a_covariates_dependency(tmp_path):
    """The sensitivity stage's primary and an analysis on the first 300 rows (a rule on a `visit`
    column no model reads), each beside `age2`: `fiber` at statsmodels' estimate on those rows with
    `age2` left out. On the base both were refused whole."""
    from turbotab.core.datastore import DataStore
    from turbotab.core.stages.sensitivity import sensitivity_stage

    frame = _frame("age2").drop(columns=["bmi"]).assign(visit=np.arange(float(N)))
    rule = d.ExclusionRule(column="visit", low=0, high=299, reason="the first 300")
    roles = {k: v for k, v in {**ef.ROLES, "age2": "covariate"}.items() if k != "bmi"}
    answers = {k: v for k, v in {**ef.ANSWERS, "age2": ef.CONFOUNDER}.items() if k != "bmi"}
    st = ef.state(target="glucose", task="regression", measure="mean_difference", roles=roles,
                  answers=answers, sensitivity=[d.SensitivityAnalysis(label="first", rules=[rule])])
    inputs, paths = _inputs(frame, st, tmp_path)
    with DataStore(paths["data"], 1 << 30) as store:
        inputs["ingest"] = store.info().to_dict()
    [linear] = sensitivity_stage(mf.context(st, inputs, paths)).data["families"]
    fits = {f["label"]: f for f in linear["fits"]}
    for label, rows in (("Primary", frame), ("first", frame.iloc[:300])):
        fit = fits[label]
        assert fit["inference"]["refused"] is None, label
        fiber = next(r for r in fit["coefficients"] if r["feature"] == "fiber")
        assert fiber["estimate"] == pytest.approx(_truth(rows, "regression", MODEL_2)[0],
                                                  rel=1e-8), label


def _calibrated(tmp_path: Path, copy: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Regression calibration of chain 2's recalls (``rc_fixtures.chain2_table``: 160 people, two
    recalls each, every energy source an exposure, the all-components model), with `age2` =
    3 × `age` + 1 among the covariates when ``copy``, through the graph as the server runs it:
    (the calibration's rows, the fit's coefficient rows by name)."""
    from turbotab.core.tests.acceptance.rc_fixtures import chain2_table
    from turbotab.core.tests.acceptance.test_ms5_regression_calibration import (CHAIN_ROLES,
                                                                                chain_state,
                                                                                declared)
    from turbotab.core.tests.graph_runner import GraphRun

    frame, _ = chain2_table(seed=4, per_psu=10, missing_share=0.0)
    roles = dict(CHAIN_ROLES)
    if copy:
        frame["age2"] = 3 * frame["age"] + 1
        roles["age2"] = "covariate"
    frame.to_csv(tmp_path / "chain.csv", index=False)
    run = GraphRun(tmp_path / "chain.csv", tmp_path / "project")
    try:
        state = chain_state(roles=roles, split={"holdout": 0.0, "seed": 7, "folds": 5})
        state = declared(state, frame, run.run(state, upto=["findings"]), n_boot=50)
        out = run.public(run.run(state, upto=["calibration", "fit"]))
    finally:
        run.close()
    assert out["calibration"]["applies"]
    [model] = out["fit"]["models"]
    return (out["calibration"]["exposures"],
            {r["feature"]: r for r in model["coefficients"] or []})


def test_the_calibrations_uncorrected_estimate_is_served_beside_a_covariates_dependency(
        tmp_path):
    """Regression calibration's uncorrected estimate beside `age2`: each calibrated intake's is the
    one the stage gives with `age2` left out, which spans the same columns, where on the base it
    was none (its table refused whole). The calibrated estimate beside it is that run's too (it
    was on the base: the correction's own least squares read the same span). The fit's table
    beside it serves each energy source's coefficient and its average relative effect (a refit on
    the moved energy columns, ``methods.energy.relative_effect_rows``) as that run does, where on
    the base it was refused whole."""
    (tmp_path / "with").mkdir()
    (tmp_path / "without").mkdir()
    found, fit = _calibrated(tmp_path / "with", True)
    truth, fit_truth = _calibrated(tmp_path / "without", False)
    found = {e["feature"]: e for e in found}
    truth = {e["feature"]: e for e in truth}
    assert sorted(found) == sorted(truth) and len(found) == 4
    for feature, e in found.items():
        assert e["naive"] is not None and e["naive_ci_low"] is not None, feature
        for key in ("naive", "naive_ci_low", "naive_ci_high", "p", "estimate"):
            assert e[key] == pytest.approx(truth[feature][key], rel=1e-9), (feature, key)
    relative = [f for f in fit_truth if f.endswith("_relative")]
    assert len(relative) == 4  # the premise: each source's average relative effect
    for name in [*relative, *found]:
        for key in ("estimate", "se", "ci_low", "ci_high"):
            assert fit[name][key] == pytest.approx(fit_truth[name][key], rel=1e-9), (name, key)
    assert fit["age"]["estimate"] is None and fit["age2"]["estimate"] is None


SCALE_OUTCOMES = {"regression": ("sbp", {"models": ["linear"]}),
                  "binary": ("high", {"models": ["linear"], "event": "yes"}),
                  "ordinal": ("grade", {"models": ["proportional_odds"],
                                        "outcome_order": ["low", "mid", "high"]})}


def _scale(frame: pd.DataFrame, tmp_path: Path, covariates: list[str],
           task: str = "regression") -> dict[str, Any]:
    """The scales stage on ``frame``: its 8 items scored as `sat_score` (internal consistency,
    regression calibration) beside ``covariates``, the outcome `sbp`, `high` (`sbp` above its
    median) or `grade` (its thirds)."""
    from turbotab.core.stages.scales import scales_stage
    from turbotab.core.tests.acceptance.scales_fixtures import SAT, SAT_REVERSE

    target, extra = SCALE_OUTCOMES[task]
    roles = {**{c: "covariate" for c in covariates}, **{c: "covariate" for c in SAT}}
    st = mf.state(purpose="inference", target=target, task=task, roles=roles,
                  role_confirmations=dict(roles), lens=["clinical"],
                  split=d.SplitSpec(holdout=0.0, seed=7, folds=5), shape_confirmations={},
                  scales=[d.ScaleSpec(name="sat_score", items=SAT, reverse=SAT_REVERSE, low=1,
                                      high=5, kind="reflective",
                                      correction="regression_calibration", n_boot=50)], **extra)
    inputs, paths = _inputs(frame[["pid", *covariates, *SAT, target]], st, tmp_path)
    [scale] = scales_stage(mf.context(st, inputs, paths)).data["scales"]
    return scale


@pytest.mark.parametrize("task", list(SCALE_OUTCOMES))
def test_a_scale_is_corrected_beside_a_covariates_dependency_as_without_the_aliased_column(
        tmp_path, task):
    """`age2` = 3 × `age` + 1 beside `age` and `bmi`: the score's uncorrected coefficient is
    statsmodels' (OLS, Logit, OrderedModel) on the score (the keyed items' sum, by pandas), `age`
    and `bmi`, and its correction is the one the stage makes with `age2` left out, which spans the
    same columns (the calibration's least squares on them, and the outcome model's maximum, are
    the same fit). On the base the score was not corrected, the dependency's words its reason."""
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    from turbotab.core.tests.acceptance.scales_fixtures import (SAT, SAT_REVERSE, keyed,
                                                                linear_scale_table)

    frame = linear_scale_table(n=300).drop(columns=["sat_ref"])
    frame["age2"] = 3 * frame["age"] + 1
    frame["high"] = np.where(frame["sbp"] > frame["sbp"].median(), "yes", "no")
    frame["grade"] = pd.cut(frame["sbp"], frame["sbp"].quantile([0, 1 / 3, 2 / 3, 1]),
                            labels=["low", "mid", "high"], include_lowest=True).astype(str)
    scale = _scale(frame, tmp_path / "with", ["age", "bmi", "age2"], task)
    assert scale["not_corrected"] is None and scale["correction"] is not None
    score = keyed(frame, SAT, SAT_REVERSE, 1, 5).sum(axis=1).rename("sat_score")
    X = pd.concat([score, frame[["age", "bmi"]]], axis=1)
    if task == "regression":
        naive = sm.OLS(frame["sbp"].to_numpy(dtype=float), sm.add_constant(X)).fit().params
    elif task == "binary":
        naive = sm.Logit((frame["high"] == "yes").to_numpy(float), sm.add_constant(X)).fit(
            disp=0).params
    else:
        codes = frame["grade"].map({"low": 0, "mid": 1, "high": 2}).to_numpy()
        naive = OrderedModel(codes, X, distr="logit").fit(method="newton", disp=0,
                                                          maxiter=200).params
    assert scale["correction"]["naive"] == pytest.approx(float(naive["sat_score"]), rel=1e-6)
    without = _scale(frame, tmp_path / "without", ["age", "bmi"], task)["correction"]
    for key in ("naive", "estimate", "se", "ci_low", "ci_high", "attenuation"):
        assert scale["correction"][key] == pytest.approx(without[key], rel=1e-9), key
