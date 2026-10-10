"""F15-rest (RECIPES_AND_TUNING §1 F15, §4.3; WAVE_C6A_PLAN §7 rulings 9 and 13): the split's
seed, the rows' survey design and the stage's cancel reach every refit a stage makes after the fit,
and a spec a stage builds afresh keeps the design stage's tuning plans.

Every expected value comes from outside the code under test:

* **the seed** is the one the test writes into the split answer (7, or the estimand fixture's 3);
* **the design** of each refit's rows is read from the table the test wrote, by row id: its ``wt``,
  ``stratum`` and ``psu`` columns (one cycle, so the analysis weight is the weight as given,
  NHANES Analytic Guidelines §3.1.3);
* **the plans** are the design stage's own record (``design.objects["spec"]["plans"]``), compared
  with what each later stage hands ``build_pipeline``;
* **the cancel**: a pressed Cancel raised inside the tuned family's refit itself (``jobs.Cancelled``
  out of ``fit_pipeline``), not at the stage's next progress report;
* **completeness**: every ``fit_pipeline`` call outside the fit stage's own module, read from the
  source, names a seed and a design (never the literal None) and sits in a cancel scope, and every
  call of a refit helper (``fit_with``, ``fit_on_rows``, ``class_family_entries``,
  ``correct_together``) hands it the rows' design and the stage's cancel;
* **the moved numbers**: under the population answer a tuned family's refit draws its inner splits
  by whole PSU even at split seed 0, read against the table's own stratum and PSU columns.

Fixtures are small (at most 400 rows).
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from turbotab.core import decisions as d
from turbotab.core.jobs import Cancelled
from turbotab.core.models import inner_cv, tuning
from turbotab.core.tests import modeling_fixtures as mf

CORE = Path(__file__).resolve().parents[1]
# The fit stage's module (its own closures are F15's first half), the function's definition and
# the engine's own nested fits.
EXEMPT = {CORE / "stages" / "modeling.py", CORE / "models" / "inner_cv.py",
          CORE / "models" / "tuning.py"}


def _name(func: ast.AST) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _in_cancel_scope(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    while node in parents:
        node = parents[node]
        if isinstance(node, ast.With) and any(
                isinstance(i.context_expr, ast.Call) and _name(i.context_expr.func) == "cancel_scope"
                for i in node.items):
            return True
    return False


def _keyword(call: ast.Call, name: str) -> ast.AST | None:
    return next((k.value for k in call.keywords if k.arg == name), None)


def _given(call: ast.Call, name: str) -> bool:
    """Whether ``call`` passes ``name=`` as something other than the literal None."""
    value = _keyword(call, name)
    return value is not None and not (isinstance(value, ast.Constant) and value.value is None)


# The refit helpers a stage hands its rows' design and its cancel to, with what each must be given
# (the helpers whose ``seed`` defaults to the recorded split's need no ``seed=``).
HELPERS = {"fit_with": ("seed", "designs", "cancelled"),
           "fit_on_rows": ("designs", "cancelled"),
           "class_family_entries": ("designs", "cancelled"),
           "correct_together": ("seed", "designs", "cancelled")}
# F15's last two call sites, in files the integrator owns this phase (stages/modeling.py and
# stages/secondary.py): each is removed here when its call passes them.
PENDING: set[str] = set()


def _sources() -> list[tuple[Path, ast.AST]]:
    return [(path, ast.parse(path.read_text("utf-8"))) for path in sorted(CORE.rglob("*.py"))
            if "tests" not in path.relative_to(CORE).parts]


def test_every_refit_outside_the_fit_stage_names_the_seed_and_design_inside_a_cancel_scope() -> None:
    """Read from the source: each ``inner_cv.fit_pipeline`` call outside ``stages/modeling.py``
    passes ``seed=`` (never the literal 0 it defaults to, nor None) and ``design=`` (never the
    literal None), inside ``with cancel_scope(...)``; each call to ``stages.modeling.fit_with``
    (which opens the scope and reads the rows' design itself) passes ``seed=``, ``designs=`` and
    ``cancelled=``, none of them the literal None."""
    problems: list[str] = []
    for path, tree in _sources():
        if path in EXEMPT:
            continue
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _name(node.func)
            where = f"{path.relative_to(CORE)}:{node.lineno}"
            if name == "fit_pipeline":
                seed = _keyword(node, "seed")
                if not _given(node, "seed") or (isinstance(seed, ast.Constant) and seed.value == 0):
                    problems.append(f"{where}: no split seed")
                if not _given(node, "design"):
                    problems.append(f"{where}: no design")
                if not _in_cancel_scope(node, parents):
                    problems.append(f"{where}: outside a cancel scope")
            elif name == "fit_with":
                for key in HELPERS["fit_with"]:
                    if not _given(node, key):
                        problems.append(f"{where}: no {key}")
    assert problems == []


def test_every_call_of_a_refit_helper_hands_it_the_design_and_the_cancel() -> None:
    """Read from the source, every module included: each call of a stage's refit helper
    (``fit_on_rows``, ``class_family_entries``, ``correct_together``, ``fit_with``) passes the
    rows' design and the stage's cancel (and ``seed=`` where the helper has no split to read it
    from), none of them the literal None. ``PENDING`` is empty since the integration of C6a phase
    3 gave the substitution stage's class refits and the secondary stage's refits theirs."""
    problems: set[str] = set()
    for path, tree in _sources():
        for node in ast.walk(tree):
            name = _name(node.func) if isinstance(node, ast.Call) else None
            if name in HELPERS:
                for key in HELPERS[name]:
                    if not _given(node, key):
                        problems.add(f"{path.relative_to(CORE)}: {name}: no {key}")
    assert problems == PENDING


# ── the threading, observed: sensitivity and explain under the surveyed population ──────────


def _design_table(seed: int = 0, strata: int = 8, psus: int = 2, per: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for h in range(strata):
        for j in range(psus):
            for _ in range(per):
                x1, x2, x3 = rng.normal(size=3)
                rows.append({"x1": x1, "x2": x2, "x3": x3, "stratum": 100 + h, "psu": j + 1,
                             "wt": rng.uniform(500, 5000),
                             "y": 1.0 + 0.8 * x1 - 0.5 * x2 + 0.2 * x3 + rng.normal()})
    frame = pd.DataFrame(rows)
    frame.insert(0, "pid", np.arange(len(frame)))
    return frame


class _Spy:
    """Every ``inner_cv.fit_pipeline`` call: the stage running, the seed, the design, whether a
    cancel scope was open, and the rows' ids."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.stage = "?"
        self.real = inner_cv.fit_pipeline

    def __call__(self, pipeline: Any, X: Any, y: Any, **kw: Any) -> Any:
        self.calls.append({"stage": self.stage, "seed": kw.get("seed", 0),
                           "design": kw.get("design"), "scoped": bool(tuning._cancel_checks()),
                           "tuned": type(pipeline).__name__ == "TunedPipeline",
                           "plan": getattr(pipeline, "search", None),
                           "ids": np.asarray(getattr(X, "index", np.arange(len(X))))})
        return self.real(pipeline, X, y, **kw)


@pytest.fixture(scope="module")
def population_run(tmp_path_factory):
    from turbotab.core.tests.graph_runner import GraphRun

    frame = _design_table()
    folder = tmp_path_factory.mktemp("f15")
    frame.to_csv(folder / "t.csv", index=False)
    run = GraphRun(folder / "t.csv", folder / "p")
    roles = {"pid": "identifier", "x1": "exposure", "x2": "covariate", "x3": "covariate",
             "stratum": "design", "psu": "design", "wt": "design"}
    state = d.ProjectState(
        lens=["survey"], target="y", task="regression", purpose="prediction", roles=roles,
        missing="complete_case", grain=d.GrainSpec(grain="one_row_per_unit", id_column="pid"),
        split=d.SplitSpec(holdout=0.2, seed=7, folds=5), models=["linear", "random_forest"],
        survey=d.SurveySpec(estimand="population", weight="wt", strata="stratum", psu="psu"),
        sensitivity=[d.SensitivityAnalysis(label="x2 above -1", rules=[
            d.ExclusionRule(column="x2", low=-1, high=5, reason="a narrower range")])],
        explain=d.ExplainSpec(reseeds=2))
    spy = _Spy()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(inner_cv, "fit_pipeline", spy)
        out = run.run(state, upto=["sensitivity", "explain"],
                      before=lambda stage: setattr(spy, "stage", stage))
    yield frame, out, spy.calls
    run.close()


@pytest.mark.parametrize("stage", ["sensitivity", "explain"])
def test_each_refit_gets_the_splits_seed_and_its_rows_design_inside_a_cancel_scope(
        population_run, stage) -> None:
    """Under the surveyed population each refit the sensitivity analyses and the explanations'
    stability make, the random forest's search among them, is handed seed 7 (the split's), and
    the weight, stratum and PSU of exactly the rows it is fit on (read here from the table by row
    id), while a cancel scope is open."""
    frame, out, calls = population_run
    assert stage in out
    mine = [c for c in calls if c["stage"] == stage]
    assert mine, f"{stage} refit nothing"
    if stage == "sensitivity":
        assert any(c["tuned"] for c in mine)  # the forest searched there
    for c in mine:
        assert c["seed"] == 7
        assert c["scoped"]
        design = c["design"]
        assert design is not None
        rows = frame.loc[c["ids"]]
        np.testing.assert_array_equal(design.weights, rows["wt"].to_numpy(dtype=float))
        np.testing.assert_array_equal(design.strata.astype(np.int64), rows["stratum"].to_numpy())
        np.testing.assert_array_equal(design.psu.astype(np.int64), rows["psu"].to_numpy())


# ── fresh specs keep the plans: effects and the interaction ──────────────────────────────────


@pytest.fixture(scope="module")
def inference_run(tmp_path_factory):
    from turbotab.core.methods import interaction as ix
    from turbotab.core.models import pipeline as P
    from turbotab.core.stages.effects import effects_stage
    from turbotab.core.stages.modeling import design_stage
    from turbotab.core.tests.acceptance import estimand_fixtures as est_f

    frame = est_f.cohort(n=400, seed=17)
    st = est_f.state(target="dm", task="binary", event="yes", measure="odds_ratio",
                     models=["linear", "random_forest"],
                     modifications={"sex": d.ModificationSpec(kind="effect_modification",
                                                              exposure="fiber")})
    paths = mf.ingest_frame(frame, tmp_path_factory.mktemp("f15_inference"))
    split = mf.split_bundle(np.arange(len(frame)), holdout=0.0, seed=st.split.seed)
    info = mf.target_info(st.task, st.target)
    design = design_stage(mf.context(st, {"split": split, "target_info": info}, paths))
    inputs = {"design": design, "split": split, "target_info": info}
    spy = _Spy()
    built: list[tuple[str, str, Any]] = []
    real_build = P.build_pipeline

    def build(spec: Any, family: Any, *args: Any, **kw: Any) -> Any:
        built.append((spy.stage, family.key, spec))
        return real_build(spec, family, *args, **kw)

    out = {}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(inner_cv, "fit_pipeline", spy)
        patch.setattr(P, "build_pipeline", build)
        spy.stage = "effects"
        out["effects"] = effects_stage(mf.context(st, inputs, paths)).data
        spy.stage = "modification"
        out["modification"] = ix.modification_stage(mf.context(st, inputs, paths)).data
    return st, design, out, built, spy.calls


@pytest.mark.parametrize("stage", ["effects", "modification"])
def test_a_spec_built_afresh_keeps_the_design_stages_plans(inference_run, stage) -> None:
    """Effects' Model 3 and the interaction's model each build a spec of their own columns; it
    holds the plans the design stage made, so the random forest (a tuned family) builds from it as
    the searched pipeline that plan describes, where it raised ``MissingPlan`` before. The forest
    itself never reaches either stage (it has no coefficient table), and no registered tuned
    family does today (ridge, Huber and XGBoost declare no product terms or no inference), so
    this checks the spec; the probe family below goes through the interaction end to end."""
    from turbotab.core.models import get_family
    from turbotab.core.models.pipeline import build_pipeline
    from turbotab.core.models.tuning import TunedPipeline

    st, design, out, built, _ = inference_run
    plans = design.objects["spec"]["plans"]
    assert "random_forest" in plans  # the design stage planned the forest's search
    specs = [spec for where, _, spec in built if where == stage]
    assert specs, f"{stage} built no pipeline of its own"
    forest = get_family("random_forest")
    for spec in specs:
        assert spec.plans == plans
        tuned = build_pipeline(spec, forest, "binary", "inference", 400, len(spec.inputs) + 1)
        assert isinstance(tuned, TunedPipeline)
        assert tuned.search.to_dict() == plans["random_forest"]


@pytest.mark.parametrize("stage", ["effects", "modification"])
def test_effects_and_the_interaction_refit_at_the_splits_seed_inside_a_cancel_scope(
        inference_run, stage) -> None:
    """Every refit effects and the interaction make gets the split's seed (the fixture's 3) and a
    cancel scope; no survey is answered here, so the design is None."""
    st, _, out, _, calls = inference_run
    mine = [c for c in calls if c["stage"] == stage]
    assert mine
    assert {c["seed"] for c in mine} == {st.split.seed} == {3}
    assert all(c["scoped"] and c["design"] is None for c in mine)


def _assert_rows_design(design: Any, rows: pd.DataFrame) -> None:
    """``design`` (a ``tuning.FitDesign``) is the weight, stratum and PSU of ``rows``, read from
    the table the test wrote."""
    assert design is not None
    np.testing.assert_array_equal(design.weights, rows["wt"].to_numpy(dtype=float))
    np.testing.assert_array_equal(design.strata.astype(np.int64), rows["stratum"].to_numpy())
    np.testing.assert_array_equal(design.psu.astype(np.int64), rows["psu"].to_numpy())


def _with_design_columns(frame: pd.DataFrame, seed: int = 5) -> pd.DataFrame:
    """``frame`` with 8 strata of 2 PSUs (rows dealt in turn) and a weight per row."""
    rng = np.random.default_rng(seed)
    out = frame.copy()
    out["stratum"] = 100 + np.arange(len(out)) % 8
    out["psu"] = 1 + (np.arange(len(out)) // 8) % 2
    out["wt"] = np.round(rng.uniform(500, 5000, len(out)), 1)
    return out


SURVEYED = {"stratum": "design", "psu": "design", "wt": "design"}


# ── effects and the interaction under the surveyed population ────────────────────────────────


@pytest.fixture(scope="module")
def population_inference_run(tmp_path_factory):
    from turbotab.core.methods import interaction as ix
    from turbotab.core.stages.effects import effects_stage
    from turbotab.core.stages.modeling import design_stage
    from turbotab.core.tests.acceptance import estimand_fixtures as est_f

    frame = _with_design_columns(est_f.cohort(n=320, seed=17))
    st = est_f.state(target="dm", task="binary", event="yes", measure="odds_ratio",
                     roles={**est_f.ROLES, **SURVEYED}, models=["linear"],
                     lens=["clinical", "survey"],
                     survey=d.SurveySpec(estimand="population", weight="wt", strata="stratum",
                                         psu="psu"),
                     modifications={"sex": d.ModificationSpec(kind="effect_modification",
                                                              exposure="fiber")})
    paths = mf.ingest_frame(frame, tmp_path_factory.mktemp("f15_population_inference"))
    split = mf.split_bundle(np.arange(len(frame)), holdout=0.0, seed=st.split.seed)
    info = mf.target_info(st.task, st.target)
    design = design_stage(mf.context(st, {"split": split, "target_info": info}, paths))
    inputs = {"design": design, "split": split, "target_info": info}
    spy = _Spy()
    out = {}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(inner_cv, "fit_pipeline", spy)
        spy.stage = "effects"
        out["effects"] = effects_stage(mf.context(st, inputs, paths)).data
        spy.stage = "modification"
        out["modification"] = ix.modification_stage(mf.context(st, inputs, paths)).data
    return frame, out, spy.calls


@pytest.mark.parametrize("stage", ["effects", "modification"])
def test_effects_and_the_interaction_hand_each_refit_its_rows_design(population_inference_run,
                                                                     stage) -> None:
    """Under the surveyed population every refit effects and the interaction make is handed the
    weight, stratum and PSU of exactly the rows it is fit on (read here from the table by row id),
    with the split's seed (the fixture's 3), while a cancel scope is open."""
    frame, out, calls = population_inference_run
    assert out[stage]
    mine = [c for c in calls if c["stage"] == stage]
    assert mine
    for c in mine:
        assert c["seed"] == 3 and c["scoped"]
        _assert_rows_design(c["design"], frame.loc[c["ids"]])


# ── a tuned family with a coefficient table through the interaction ──────────────────────────


def _probe_family() -> Any:
    """A tuned family that tests product terms: the linear family under another key, searching
    its solver's convergence tolerance (two candidates; the fitted model barely moves). Only a
    test registers it."""
    from turbotab.core.models.linear import Linear
    from turbotab.core.models.tuning import Dimension, TuningDecl

    class ProbeTuned(Linear):
        key = "probe_tuned_linear"
        label = "Probe tuned linear"
        tuning = TuningDecl(
            "search", dimensions=(Dimension("tol", "how closely the fit converges", "tolerance",
                                            1e-10, 1e-8, "log", source="a probe"),),
            standard={"tol": 1e-8}, space_version="probe/1", reason="A probe.")

    return ProbeTuned()


def test_a_tuned_family_with_product_terms_is_refit_by_the_interaction_as_its_plan_says(
        tmp_path) -> None:
    """A tuned family that reports coefficients and tests product terms (a probe: no registered
    family does both yet) goes through effects and the interaction end to end: the interaction's
    spec keeps the design stage's plan, so its refits are the searched pipeline that plan
    describes (``MissingPlan`` stopped it before any refit on the base), each at the split's seed
    (the fixture's 3) inside a cancel scope, and effects refits its primary model the same way."""
    from turbotab.core.methods import interaction as ix
    from turbotab.core.models import base as B
    from turbotab.core.stages.effects import effects_stage
    from turbotab.core.stages.modeling import design_stage
    from turbotab.core.tests.acceptance import estimand_fixtures as est_f

    probe = _probe_family()
    spy = _Spy()
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(B._REGISTRY, probe.key, probe)
        frame = est_f.cohort(n=320, seed=17)
        st = est_f.state(target="dm", task="binary", event="yes", measure="odds_ratio",
                         models=["linear", probe.key],
                         modifications={"sex": d.ModificationSpec(kind="effect_modification",
                                                                  exposure="fiber")})
        paths = mf.ingest_frame(frame, tmp_path)
        split = mf.split_bundle(np.arange(len(frame)), holdout=0.0, seed=st.split.seed)
        info = mf.target_info(st.task, st.target)
        design = design_stage(mf.context(st, {"split": split, "target_info": info}, paths))
        plan = tuning.TuningPlan.from_dict(design.objects["spec"]["plans"][probe.key])
        inputs = {"design": design, "split": split, "target_info": info}
        patch.setattr(inner_cv, "fit_pipeline", spy)
        spy.stage = "effects"
        effects_stage(mf.context(st, inputs, paths))
        spy.stage = "modification"
        ix.modification_stage(mf.context(st, inputs, paths))
    for stage in ("effects", "modification"):
        searched = [c for c in spy.calls if c["stage"] == stage and c["tuned"]]
        assert searched, f"{stage} never refit the tuned family"
        for c in searched:
            assert c["plan"].to_dict() == plan.to_dict()
            assert c["seed"] == 3 and c["scoped"]


# ── scales, calibration and class substitution, observed ─────────────────────────────────────


def _designs_from(frame: pd.DataFrame) -> Any:
    """A stand-in for ``stages.modeling.fit_designs``: each row's weight, stratum and PSU read
    from the table the test wrote (row id = position), for the rows asked."""

    def fit_designs(state: Any, store: Any, row_ids: Any) -> pd.DataFrame:
        rows = frame.iloc[np.asarray(row_ids, dtype=int)]
        return pd.DataFrame({"weight": rows["wt"].to_numpy(dtype=float),
                             "stratum": rows["stratum"].to_numpy(dtype=object),
                             "psu": rows["psu"].to_numpy(dtype=object)},
                            index=pd.Index(np.asarray(row_ids)))

    return fit_designs


def test_the_scales_correction_refits_each_copy_with_its_rows_design(tmp_path) -> None:
    """The scales stage's regression calibration refits the outcome model on each copy at the
    split's seed (7), inside a cancel scope, handed the rows' design. Under the population answer
    the correction is refused before any refit today (no design-based estimator), so the design
    the stage reads (``fit_designs``) is supplied here from the table's own columns under the
    sample answer, to watch it reach each refit by row id."""
    from turbotab.core.stages import modeling as M
    from turbotab.core.stages.modeling import design_stage
    from turbotab.core.stages.scales import scales_stage
    from turbotab.core.tests.acceptance.scales_fixtures import linear_scale_table

    items = [f"sat_{j}" for j in range(1, 9)]
    frame = _with_design_columns(linear_scale_table(n=300))
    roles = {"age": "covariate", "bmi": "covariate", **{c: "covariate" for c in items}}
    st = mf.state(purpose="inference", target="sbp", task="regression", models=["linear"],
                  roles=roles, role_confirmations=dict(roles), lens=["clinical"],
                  split=d.SplitSpec(holdout=0.0, seed=7, folds=5), shape_confirmations={},
                  scales=[d.ScaleSpec(name="sat_score", items=items, reverse=["sat_3", "sat_6"],
                                      low=1, high=5, kind="reflective",
                                      correction="regression_calibration", n_boot=50)])
    paths = mf.ingest_frame(frame, tmp_path)
    split = mf.split_bundle(np.arange(len(frame)), holdout=0.0, seed=7)
    ti = mf.target_info("regression", "sbp")
    design = design_stage(mf.context(st, {"split": split, "target_info": ti}, paths))
    spy = _Spy()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(M, "fit_designs", _designs_from(frame))
        patch.setattr(inner_cv, "fit_pipeline", spy)
        out = scales_stage(mf.context(st, {"design": design, "split": split, "target_info": ti},
                                      paths)).data
    assert out["scales"][0]["correction"] is not None
    assert spy.calls
    for c in spy.calls:
        assert c["seed"] == 7 and c["scoped"]
        _assert_rows_design(c["design"], frame.iloc[c["ids"]])


def test_regression_calibration_refits_under_the_population_with_each_persons_design(
        tmp_path) -> None:
    """Under the surveyed population every refit regression calibration makes (the first on every
    person, each bootstrap replicate's) is handed the split's seed (7), inside a cancel scope, and
    a design whose every row is a person of the table: its stratum and PSU a pair the table holds
    and its weight that PSU's (the fixture weights by PSU). The refit on every person carries each
    PSU as many times as the table has people in it."""
    from turbotab.core.tests.acceptance.rc_fixtures import chain2_table
    from turbotab.core.tests.acceptance.test_ms5_regression_calibration import (POPULATION,
                                                                                chain_state,
                                                                                declared)
    from turbotab.core.tests.graph_runner import GraphRun

    frame, _ = chain2_table(seed=4, per_psu=10, missing_share=0.0)
    frame.to_csv(tmp_path / "chain.csv", index=False)
    people = frame.groupby("seqn").first()
    weight_of = people.groupby(["sdmvstra", "sdmvpsu"])["wtdr2d"].agg(["first", "nunique"])
    assert (weight_of["nunique"] == 1).all()
    run = GraphRun(tmp_path / "chain.csv", tmp_path / "project")
    spy = _Spy()
    try:
        state = chain_state(survey=POPULATION, split={"holdout": 0.0, "seed": 7, "folds": 5})
        state = declared(state, frame, run.run(state, upto=["findings"]), n_boot=50)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(inner_cv, "fit_pipeline", spy)
            out = run.public(run.run(state, upto=["calibration"],
                                     before=lambda stage: setattr(spy, "stage", stage)))
    finally:
        run.close()
    assert out["calibration"]["applies"]
    mine = [c for c in spy.calls if c["stage"] == "calibration"]
    assert len(mine) > 1
    for c in mine:
        assert c["seed"] == 7 and c["scoped"]
        pairs = list(zip(c["design"].strata.astype(np.int64), c["design"].psu.astype(np.int64)))
        expected = weight_of["first"].reindex(pd.MultiIndex.from_tuples(pairs)).to_numpy()
        np.testing.assert_array_equal(c["design"].weights, expected)
    first = mine[0]["design"]
    counted = pd.Series(list(zip(first.strata.astype(np.int64), first.psu.astype(np.int64)))
                        ).value_counts().sort_index()
    by_table = people.groupby(["sdmvstra", "sdmvpsu"]).size()
    assert counted.to_dict() == by_table.to_dict()


def test_class_substitution_refits_at_the_splits_seed_with_the_design_it_is_handed(
        tmp_path) -> None:
    """Class substitution's refit on every analyzed row, called as the substitution stage calls it
    (the split's seed, its designs and its cancel), is fit at the recorded split's seed (7, where
    it was 0 on the base: the curves of a family with random draws move with it), inside a cancel
    scope, with the design it is handed (``designs=``, here the table's own columns by row id)."""
    from turbotab.core.decisions import EnergyAdjustment, SubstitutionSpec
    from turbotab.core.stages import class_substitution as CS
    from turbotab.core.stages.modeling import design_stage, fit_stage, substitution_stage
    from turbotab.core.tests.acceptance.test_multisub_class_curves import (DONOR, RECIPIENT, ROLES,
                                                                           SOURCES, STEP,
                                                                           diet_classes)

    frame = _with_design_columns(diet_classes(seed=13, n=300))
    exposures = [c for c, r in ROLES.items() if r == "exposure"]
    paths = mf.ingest_frame(frame, tmp_path)
    st = mf.state(roles=ROLES, target="y", task="multiclass", models=["linear"],
                  purpose="inference",
                  energy_adjustment=EnergyAdjustment(method="all_components",
                                                     energy_column="kcal", nutrients=exposures),
                  substitution=SubstitutionSpec(donor=DONOR, recipient=RECIPIENT, step_kcal=STEP,
                                                n_boot=0),
                  split=d.SplitSpec(holdout=0.0, seed=7, folds=5), column_units=mf.grams(*SOURCES))
    split = mf.split_bundle(np.arange(len(frame)), holdout=0.0, seed=7)
    ti = mf.target_info("multiclass", "y")
    design = design_stage(mf.context(st, {"split": split, "target_info": ti}, paths))
    fit = fit_stage(mf.context(st, {"design": design, "split": split, "target_info": ti}, paths))
    designs = _designs_from(frame)(st, None, np.arange(len(frame)))
    real = CS.class_family_entries

    def handed(*args: Any, **kw: Any) -> Any:
        # the substitution stage's call hands the split's seed, its own designs (None here: no
        # population answer) and its cancel; the table's design columns are handed in their place
        assert kw["seed"] == 7 and kw["designs"] is None and callable(kw["cancelled"])
        return real(*args, **{**kw, "designs": designs})

    spy = _Spy()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(CS, "class_family_entries", handed)
        patch.setattr(inner_cv, "fit_pipeline", spy)
        sub = substitution_stage(mf.context(st, {"design": design, "fit": fit}, paths))
    assert len(sub["models"]) == 3
    assert spy.calls
    for c in spy.calls:
        assert c["seed"] == 7 and c["scoped"]
        _assert_rows_design(c["design"], frame.iloc[c["ids"]])


# ── the moved numbers: a tuned family's sensitivity refit at split seed 0 ───────────────────


def test_at_split_seed_0_a_tuned_refit_under_the_population_draws_whole_psus(tmp_path) -> None:
    """Why a tuned family's sensitivity numbers move even where the split's seed is 0: under the
    population answer its refit is handed the rows' design, so its search draws each inner split
    by whole PSU (RECIPES §4.3) where the base drew them by row and a PSU's rows sat on both sides.
    Ridge, the 'x2 above -1' analysis: each inner split's training and validation rows share no
    stratum-and-PSU pair, read from the table's own columns."""
    from turbotab.core.tests.graph_runner import GraphRun

    frame = _design_table()
    frame.to_csv(tmp_path / "t.csv", index=False)
    psu = pd.Series([f"{s}|{p}" for s, p in zip(frame["stratum"], frame["psu"])],
                    index=frame["pid"])
    roles = {"pid": "identifier", "x1": "exposure", "x2": "covariate", "x3": "covariate",
             "stratum": "design", "psu": "design", "wt": "design"}
    state = d.ProjectState(
        lens=["survey"], target="y", task="regression", purpose="prediction", roles=roles,
        missing="complete_case", grain=d.GrainSpec(grain="one_row_per_unit", id_column="pid"),
        split=d.SplitSpec(holdout=0.2, seed=0, folds=5), models=["ridge"],
        survey=d.SurveySpec(estimand="population", weight="wt", strata="stratum", psu="psu"),
        sensitivity=[d.SensitivityAnalysis(label="x2 above -1", rules=[
            d.ExclusionRule(column="x2", low=-1, high=5, reason="a narrower range")])])
    now = ["?"]
    drawn: list[Any] = []

    def seen(found: Any) -> None:
        if now[0] == "sensitivity" and found.kind == "inner":
            drawn.append(found)

    run = GraphRun(tmp_path / "t.csv", tmp_path / "p")
    try:
        with tuning.observing(seen):
            out = run.run(state, upto=["sensitivity"], before=lambda stage: now.__setitem__(0, stage))
    finally:
        run.close()
    assert "sensitivity" in out and drawn
    for found in drawn:
        assert not set(psu.loc[found.train]) & set(psu.loc[found.validation])


# ── cancel reaches a sensitivity refit ───────────────────────────────────────────────────────


def test_a_pressed_cancel_stops_a_sensitivity_refit_inside_the_forests_search(tmp_path) -> None:
    """Cancel is pressed as the sensitivity analysis starts refitting the random forest: the
    forest's search raises ``Cancelled`` from inside that refit (the engine checks before each
    candidate fit and the refit, RECIPES §4.3), rather than running to the end and the stage
    noticing at its next progress report."""
    from turbotab.core.datastore import DataStore
    from turbotab.core.stages.modeling import design_stage
    from turbotab.core.stages.sensitivity import sensitivity_stage

    frame = mf.nhanes_like(200, seed=3)
    paths = mf.ingest_frame(frame, tmp_path)
    split = mf.split_bundle(np.arange(200), seed=5)
    state = mf.state(models=["linear", "random_forest"], split=d.SplitSpec(holdout=0.2, seed=5),
                     sensitivity=[d.SensitivityAnalysis(label="under 70", rules=[
                         d.ExclusionRule(column="age", low=0, high=69, reason="under 70")])])
    inputs = {"split": split, "target_info": mf.target_info("regression")}
    design = design_stage(mf.context(state, inputs, paths))
    with DataStore(paths["data"], 1 << 30) as store:
        ingest = store.info().to_dict()
    pressed = [False]
    raised: list[str] = []
    real = inner_cv.fit_pipeline

    def spy(pipeline: Any, X: Any, y: Any, **kw: Any) -> Any:
        if type(pipeline).__name__ == "TunedPipeline":
            pressed[0] = True
        try:
            return real(pipeline, X, y, **kw)
        except Cancelled:
            raised.append(type(pipeline).__name__)
            raise

    ctx = mf.context(state, {**inputs, "design": design, "ingest": ingest}, paths,
                     cancelled=lambda: pressed[0])
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(inner_cv, "fit_pipeline", spy)
        with pytest.raises(Cancelled):
            sensitivity_stage(ctx)
    assert pressed[0]
    assert raised == ["TunedPipeline"]
