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
  source, names a seed and a design and sits in a cancel scope.

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


def test_every_refit_outside_the_fit_stage_names_the_seed_and_design_inside_a_cancel_scope() -> None:
    """Read from the source: each ``inner_cv.fit_pipeline`` call outside ``stages/modeling.py``
    passes ``seed=`` (never the literal 0 it defaults to) and ``design=``, inside
    ``with cancel_scope(...)``; each call to ``stages.modeling.fit_with`` (which opens the scope
    and reads the rows' design itself) passes ``seed=``, ``designs=`` and ``cancelled=``."""
    problems: list[str] = []
    for path in sorted(CORE.rglob("*.py")):
        if "tests" in path.relative_to(CORE).parts or path in EXEMPT:
            continue
        tree = ast.parse(path.read_text("utf-8"))
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _name(node.func)
            where = f"{path.relative_to(CORE)}:{node.lineno}"
            if name == "fit_pipeline":
                seed = _keyword(node, "seed")
                if seed is None or (isinstance(seed, ast.Constant) and seed.value == 0):
                    problems.append(f"{where}: no split seed")
                if _keyword(node, "design") is None:
                    problems.append(f"{where}: no design")
                if not _in_cancel_scope(node, parents):
                    problems.append(f"{where}: outside a cancel scope")
            elif name == "fit_with":
                for key in ("seed", "designs", "cancelled"):
                    if _keyword(node, key) is None:
                        problems.append(f"{where}: no {key}")
    assert problems == []


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
    the searched pipeline that plan describes, where it raised ``MissingPlan`` before."""
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
