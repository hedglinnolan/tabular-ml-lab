"""The previews count and name every column the model sees on a plan wider than the lineage's
display fold (PP2 W1: glucose's plans have about 49 inputs).

The premise checked: ``lineage.trace`` folds a plan of more than ``COLLAPSE_AT`` raw columns for
display, keeping up to ``SHOWN`` of the columns a step changed and folding the rest into count
nodes, which name no column. The energy preview counted the model's columns as the drawing's named
matrix nodes ("The model sees 19 columns" of this file's wide table's 58, and `kcal`, which stays,
said to arrive), and ``plan_previews.lineage_change``, which tells what arrives and leaves for the form,
categorical and fixed-effects previews, compared two drawings' named nodes: a column folded in one
and drawn in the other read as arriving or leaving, and a folded change was not told at all (the
categorical preview said "`site` becomes 0 indicators"). They now read every column by name (the
energy preview's lineage unfolded, the plan previews' designs' matrices), and the drawing stays
folded. A plan of ``COLLAPSE_AT`` raw columns, which the lineage never folds, reads as before.

The same compare of two drawings told the fixed-effects preview its indicators (folded, none
arrived, and the view was dropped), the scales preview its items leaving (the view was dropped) and
the batch preview the columns it emphasizes (reference ComBat keeps every feature's name, so none
arrives, but features folded in the plan's drawing and drawn in the answer's were emphasized as
arriving); the selection and levers previews compared two lineages of
Explore's steps, and told part of what arrives and leaves ("`gender_male`, `t00_b` and 21 more
arrive" of 55, "The model sees the same columns" where nine leave). They read every column by
name too: the plan previews' designs, and Explore's steps' fitted matrices.

Expected values come from construction (the table's predictors, the residual method's nutrients,
the four levels of `site`, a quintile form's five groups, the nine of `clinic`, the three columns
that carry the outcome and so are what a screen of three terms keeps, the five knots Harrell's rule
gives these rows, the batch table's features, each keeping its name through ComBat), and from
scikit-learn's ``OneHotEncoder`` (the indicators of the text columns,
and so the width of the matrix).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import OneHotEncoder

from turbotab.core.consequences import PreviewContext, after_state, plan
from turbotab.core.datastore import DataStore
from turbotab.core.decisions import (LeverSpec, ScaleSpec, SelectionSpec, SelectModels,
                                     SetBatch, SetCategorical, SetClusters, SetEnergyAdjustment,
                                     SetExposureForm, SetLevers, SetScales, SetSelection)
from turbotab.core.models.lineage import COLLAPSE_AT, SHOWN
from turbotab.core.tests import modeling_fixtures as mf

N = 600
# Three-level text predictors, each a change the drawing keeps in view (its two indicators), one
# more of them than the fold keeps: a change made after them is folded on a wide plan.
TEXT = [f"t{j:02d}" for j in range(SHOWN + 1)]
NUTRIENTS = ["protein", "carb", "fat_total"]
PREDICTOR_ROLES = ("covariate", "exposure", "energy")
NHANES_INPUTS = [c for c, r in mf.NHANES_ROLES.items() if r in PREDICTOR_ROLES]
WIDE = COLLAPSE_AT + 5
NARROW = COLLAPSE_AT
# The first numeric `z..` columns, each the outcome plus noise of sd 0.01 (a correlation with it
# above 0.999; no other predictor's passes 0.5): the three terms a screen of three keeps.
KEPT = ["z00", "z01", "z02"]
# `clinic`, a grouping that is no predictor: nine named groups, the rows dealt to them in turn.
CLINICS = [f"c{j}" for j in range(9)]
# Harrell's rule (``methods.levers.SPLINE_RULE``): 5 knots above an effective size of 100, as
# these training rows are, so a restricted cubic spline adds 5 - 2 columns beside the linear one.
KNOTS = 5


def _table(inputs: int) -> tuple[pd.DataFrame, dict[str, str]]:
    """:func:`modeling_fixtures.nhanes_like`'s columns, the :data:`TEXT` columns, numeric `z..`
    columns (the first of them :data:`KEPT`) and a whole-numbered `site` of four levels, last, so
    that the plan has ``inputs`` predictors; `clinic` (:data:`CLINICS`) and `high`, glucose above
    its median, with no role; and the roles."""
    rng = np.random.default_rng(5)
    frame = mf.nhanes_like(N, seed=21)
    for c in TEXT:
        frame[c] = rng.choice(["a", "b", "c"], N)
    numeric = [f"z{j:02d}" for j in range(inputs - len(NHANES_INPUTS) - len(TEXT) - 1)]
    for c in numeric:
        frame[c] = rng.normal(size=N)
    frame["site"] = rng.choice([1, 2, 3, 4], N)
    carried = np.random.default_rng(6)
    for c in KEPT[:len(numeric)]:
        frame[c] = frame["glucose"] + carried.normal(0, 0.01, N)
    frame["clinic"] = [CLINICS[i % len(CLINICS)] for i in range(N)]
    frame["high"] = (frame["glucose"] > frame["glucose"].median()).astype(int)
    roles = {**mf.NHANES_ROLES, **{c: "covariate" for c in [*TEXT, *numeric, "site"]}}
    assert sum(r in PREDICTOR_ROLES for r in roles.values()) == inputs
    return frame, roles


def _ingested(inputs: int, frame: pd.DataFrame, roles: dict[str, str], folder: Path):
    """``(inputs, frame, roles, store, training row ids)``: the table ingested and split."""
    paths = mf.ingest_frame(frame, folder)
    store = DataStore(Path(paths["data"]), 1 << 30)
    a = mf.split_bundle(np.arange(N), seed=21).frames["assignment"]
    return inputs, frame, roles, store, a.loc[a["partition"] == "train", "row_id"].to_numpy()


@pytest.fixture(scope="module", params=[WIDE, NARROW], ids=["folded", "never-folded"])
def project(request, tmp_path_factory):
    found = _ingested(request.param, *_table(request.param), tmp_path_factory.mktemp("wide"))
    yield found
    found[3].close()


@pytest.fixture(scope="module", params=[WIDE, NARROW - 1], ids=["folded", "never-folded"])
def grouped(request, tmp_path_factory):
    """:func:`project`'s table for an answer that adds `clinic` to the plan's inputs: the narrow
    one has a predictor fewer, so that the plan the answer leaves is never folded either."""
    found = _ingested(request.param, *_table(request.param), tmp_path_factory.mktemp("grouped"))
    yield found
    found[3].close()


def _context(project, *, site_settled: bool = True, state=None, **slots) -> PreviewContext:
    """`site` read as an amount, or (``site_settled`` False) its whole numbers not yet read: codes
    or amounts, which a categorical answer settles; ``slots`` recorded on the state, or ``state``
    itself."""
    _, _, roles, store, train = project
    confirmed = {**mf.NHANES_TRUTH, **({"code_or_count:site": "amount"} if site_settled else {}),
                 **slots.pop("shape_confirmations", {})}
    st = state if state is not None else mf.state(energy_adjustment=None, roles=roles,
                                                  shape_confirmations=confirmed, **slots)
    return PreviewContext(project_id="t", state=st, datastore=store, artifact=lambda name: None,
                          training_row_ids=train, cohort_row_ids=None, sample_size=1000)


def _recorded(project, decision, **slots) -> PreviewContext:
    """The context once ``decision`` is recorded: its preview has nothing left to change."""
    return _context(project, state=after_state(decision, _context(project, **slots)))


def _width(project) -> int:
    """The model matrix's width by construction: one column per predictor but each text column,
    which scikit-learn's encoder makes into one indicator per level after the first (`gender`
    and the :data:`TEXT` columns); `site` read as an amount."""
    inputs, frame, roles, _, _ = project
    text = ["gender", *TEXT]
    encoded = OneHotEncoder(drop="first").fit(frame[text]).get_feature_names_out()
    return inputs - len(text) + len(encoded)


def _the_lineage_view(result, title: str):
    [view] = [v for v in result.views if v.kind == "lineage" and v.title == title]
    return view


def _lineage_titles(result) -> list[str]:
    return [v.title for v in result.views if v.kind == "lineage"]


def test_the_tables_plan_has_the_inputs_it_is_built_with(project):
    """The premise of every test here: the design's inputs are the table's predictors, so the wide
    plan is one the lineage folds for display and the narrow one is not."""
    from turbotab.core.plan_previews import fit_design

    inputs, _, _, _, train = project
    ctx = _context(project)
    design = fit_design(ctx, ctx.state, ctx.sample_row_ids(train))
    assert len(design.spec.inputs) == inputs
    assert design.lineage.collapsed is (inputs > COLLAPSE_AT)
    assert len(design.columns) == _width(project)


def test_the_energy_preview_counts_and_names_every_column_the_model_sees(project):
    """Residual adjustment of three nutrients, none on record: each nutrient leaves and its
    adjusted column arrives, energy stays, and the model sees the matrix's whole width. On the base
    the wide plan said "`t07_b`, `t07_c` and 6 more leave; `kcal`, `protein_adj` and 2 more
    arrive. The model sees 19 columns.": the drawings' named nodes, among them columns that stay,
    drawn before and folded after, and `kcal`, folded before and drawn after."""
    inputs = project[0]
    ctx = _context(project)
    result = plan(SetEnergyAdjustment(method="residual", energy_column="kcal", nutrients=NUTRIENTS),
                  ctx, basis="test")
    view = _the_lineage_view(result, "Columns the model will see")
    assert view.caption == (f"`protein`, `carb` and 1 more leave; `protein_adj`, `carb_adj` and 1 "
                            f"more arrive. The model sees {_width(project)} columns.")
    # the drawing keeps the display fold where the plan is wide
    assert view.after.collapsed is (inputs > COLLAPSE_AT)
    assert view.before.collapsed is (inputs > COLLAPSE_AT)
    assert sum(n.count for n in view.after.nodes if n.lane == "matrix") == _width(project)


def test_the_energy_preview_draws_the_lineage_trace_draws_and_reads_the_whole_matrix(project):
    """The energy preview's drawing is ``lineage.trace``'s own, folded where it folds, and the
    columns it counts are the fitted matrix's, every one (scikit-learn's ``transform``); on the
    base the preview's lineage was the drawing alone."""
    from turbotab.core.decisions import EnergyAdjustment
    from turbotab.core.models.lineage import missing_counts, trace
    from turbotab.core.models.pipeline import (design_spec, input_columns, model_predictors,
                                               shared_steps, transformer)
    from turbotab.core.models.previews import _lineage, _read

    ctx = _context(project)
    adjustment = EnergyAdjustment(method="residual", energy_column="kcal", nutrients=NUTRIENTS)
    predictors = model_predictors(ctx.state)
    frame = _read(ctx, input_columns(predictors, adjustment))
    drawn, columns = _lineage(ctx.state, frame, predictors, adjustment)
    spec = design_spec(ctx.state, frame, predictors, energy=adjustment)
    fitted = transformer(shared_steps(spec)).fit(frame[spec.inputs])
    assert drawn == trace(fitted.steps, spec.inputs, spec.roles, missing_counts(frame[spec.inputs]))
    assert drawn.collapsed is (project[0] > COLLAPSE_AT)
    assert columns == [str(c) for c in fitted.transform(frame[spec.inputs]).columns]
    assert len(columns) == _width(project)


def test_the_energy_preview_of_the_standard_model_keeps_the_whole_width(project):
    """The standard model, none on record, changes no column: each nutrient as recorded, energy
    beside them. On the base the wide plan said "The model matrix keeps the same 23 columns.": the
    drawing's named nodes."""
    ctx = _context(project)
    result = plan(SetEnergyAdjustment(method="standard", energy_column="kcal", nutrients=NUTRIENTS),
                  ctx, basis="test")
    view = _the_lineage_view(result, "Columns the model will see")
    assert view.caption == f"The model matrix keeps the same {_width(project)} columns."


def test_the_energy_preview_of_no_adjustment_counts_energy_leaving(project):
    """"None", none on record: energy leaves the model and every nutrient enters as recorded, so
    the model sees one column fewer than the matrix's whole width. On the base the wide plan said
    "`t10_b` and `t10_c` leave. The model sees 21 columns.": two columns that stay, drawn before
    and folded after, and not `kcal`, folded before."""
    ctx = _context(project)
    result = plan(SetEnergyAdjustment(method="none"), ctx, basis="test")
    view = _the_lineage_view(result, "Columns the model will see")
    assert view.caption == f"`kcal` leaves. The model sees {_width(project) - 1} columns."


def test_the_models_preview_counts_the_shared_matrix_whole(project):
    """A pin of the reader that was right on the base: the models preview sums the folded count
    nodes, so its width is the matrix's whole width on either plan."""
    from turbotab.core.models import get_family

    result = plan(SelectModels(models=["linear"]), _context(project), basis="test")
    [view] = result.views
    assert view.caption.startswith(f"{get_family('linear').label}: {_width(project)} columns, ")


def test_the_categorical_preview_counts_every_indicator_of_a_coded_column(project):
    """`site`'s four codes become three indicators, one per level after the first, though the
    drawing folds them behind the :data:`TEXT` columns' changes on the wide plan (on the base,
    "`site` becomes 0 indicators")."""
    _, frame, _, _, train = project
    ctx = _context(project, site_settled=False)
    levels = frame.loc[ctx.sample_row_ids(train), "site"].nunique()
    assert levels == 4
    result = plan(SetCategorical(columns=["site"]), ctx, basis="test")
    [view] = result.views
    assert view.caption == f"`site` becomes {levels - 1} indicators, one per level after the first."


def test_the_form_preview_names_what_leaves_and_how_many_arrive(project):
    """Quintiles of `bmi`: `bmi` leaves and its five groups' four indicators, one per group after
    the first, arrive; nothing else moves. On the base the wide plan said "`t10_b` and `t10_c`
    leave; `bmi_Q2`, `bmi_Q3` and 2 more arrive.": two columns that stay, drawn before and folded
    after, and not `bmi`, folded before."""
    result = plan(SetExposureForm(column="bmi", form="quintiles"), _context(project), basis="test")
    view = _the_lineage_view(result, "Columns the model will see")
    assert view.caption.startswith("`bmi` leaves; `bmi_")
    assert view.caption.endswith(f" and {4 - 2} more arrive.")
    assert view.after.collapsed is (project[0] > COLLAPSE_AT)


def test_a_change_between_two_designs_names_every_column_that_arrives_and_leaves(project):
    """``lineage_change`` reads two designs' matrices, every column by name (on the base it read
    lineages alone, and a design was no argument): the quintile form's `bmi` leaves and its four
    indicators arrive. Two lineages drawn unfolded read as their designs do."""
    from turbotab.core.plan_previews import fit_design, lineage_change

    _, _, _, _, train = project
    ctx = _context(project)
    decision = SetExposureForm(column="bmi", form="quintiles")
    ids = ctx.sample_row_ids(train)
    now, then = fit_design(ctx, ctx.state, ids), fit_design(ctx, after_state(decision, ctx), ids)
    arrive, leave = lineage_change(now, then)
    assert leave == ["bmi"]
    assert len(arrive) == 4 and all(c.startswith("bmi_") for c in arrive)
    assert lineage_change(None, then) == (then.columns, [])
    if not then.lineage.collapsed:
        assert lineage_change(now.lineage, then.lineage) == (arrive, leave)


def test_three_columns_carry_the_outcome_and_none_other_comes_near(project):
    """The premise of the selection's and levers' tests, read with numpy over the table: each
    :data:`KEPT` column correlates with glucose above 0.999 and with `high`, glucose above its
    median, above 0.75; no other column the model sees (each text column's indicators included)
    passes 0.5 with either, so a screen of three terms keeps :data:`KEPT` under either outcome."""
    _, frame, roles, _, _ = project
    text = ["gender", *TEXT]
    numbers = [c for c, r in roles.items() if r in PREDICTOR_ROLES and c not in text]
    encoded = OneHotEncoder(drop="first", sparse_output=False).fit(frame[text])
    seen = pd.concat([frame[numbers].astype(float),
                      pd.DataFrame(encoded.transform(frame[text]),
                                   columns=encoded.get_feature_names_out(), index=frame.index)],
                     axis=1)
    assert len(seen.columns) == _width(project)
    for outcome, floor in (("glucose", 0.999), ("high", 0.75)):
        r = seen.apply(lambda col: abs(np.corrcoef(col, frame[outcome].astype(float))[0, 1]))
        assert (r[KEPT] > floor).all()
        assert r.drop(KEPT).max() < 0.5


def test_a_screen_of_three_terms_keeps_the_columns_that_carry_the_outcome(project):
    """The screen keeps :data:`KEPT` of every column the model sees: a pin of a reader that was
    right on the base (the selection's terms are its fitted matrix's columns, never a drawing's)."""
    ctx = _context(project)
    result = plan(SetSelection(method="screening", keep=3, pre_selected="no"), ctx, basis="test")
    assert ctx.read["selection"]["kept"] == KEPT
    [view] = result.views
    assert view.kind == "lineage"
    assert f"it keeps `{len(KEPT)}` of `{_width(project)}` columns;" in view.caption


def test_no_selection_after_a_screen_brings_back_every_column_it_left_out(project):
    """A screen of three terms recorded, and the answer "none": the columns the screen left out,
    every one but :data:`KEPT`, arrive, and none leaves. On the base the wide plan said "`gender_male`,
    `t00_b` and 21 more arrive." (and, with other kept columns, "`age` and 2 more leave"): the
    drawings' named nodes, the kept columns drawn after the screen and folded without it."""
    ctx = _context(project, selection=SelectionSpec(method="screening", keep=3, pre_selected="no"))
    result = plan(SetSelection(method="none", pre_selected="no"), ctx, basis="test")
    view = _the_lineage_view(result, "No selection: every term enters")
    arrive = _width(project) - len(KEPT)
    assert view.caption.endswith(f" and {arrive - 2} more arrive.")
    assert "leave" not in view.caption
    assert not any(f"`{c}`" in view.caption for c in KEPT)


def test_no_form_rule_after_one_takes_each_kept_columns_bends_away(project):
    """A yes/no outcome (`high`), Harrell's rule recorded with a screen of three terms, and the
    answer no form rule with class weights: the screen keeps :data:`KEPT` with or without the rule,
    so what leaves is each kept column's restricted cubic spline columns beside its linear one,
    ``KNOTS - 2`` each, and nothing arrives. On the base the wide plan said "The model sees the same
    columns.": the spline columns were folded in the drawing before."""
    binary = {"target": "high", "task": "binary"}
    # The premise: with the rule, the screen keeps each KEPT column's spline, KNOTS - 1 columns.
    ruled = _context(project, levers=LeverSpec(forms="rule"), **binary)
    plan(SetSelection(method="screening", keep=3, pre_selected="no"), ruled, basis="test")
    kept = ruled.read["selection"]["kept"]
    assert {c.rstrip("'") for c in kept} == set(KEPT)
    assert len(kept) == len(KEPT) * (KNOTS - 1)
    ctx = _context(project, levers=LeverSpec(forms="rule"),
                   selection=SelectionSpec(method="screening", keep=3, pre_selected="no"), **binary)
    result = plan(SetLevers(forms="none", imbalance="weights"), ctx, basis="test")
    view = _the_lineage_view(result, "Explore's levers, in each training fold")
    assert view.caption == f"`{KEPT[0]}'` and {len(KEPT) * (KNOTS - 2) - 1} more leave."


def test_the_fixed_effects_preview_counts_every_indicator_of_the_grouping(grouped):
    """Under inference, `clinic`'s nine groups as fixed effects enter as eight indicators, one per
    group after the first, though the drawing folds them behind the :data:`TEXT` columns' changes
    on the wide plan (on the base the wide plan dropped the view: no indicator was drawn)."""
    inputs, frame, _, _, train = grouped
    ctx = _context(grouped, purpose="inference")
    groups = frame.loc[ctx.sample_row_ids(train), "clinic"].nunique()
    assert groups == len(CLINICS)
    result = plan(SetClusters(column="clinic", adjust="fixed_effects"), ctx, basis="test")
    view = _the_lineage_view(result, "`clinic` as fixed effects")
    assert view.caption == (f"`clinic` enters as `{groups - 1}` indicators, one per group after "
                            f"the first.")
    # `clinic` is one input more: the drawings keep the display fold where the plans are wide
    assert view.before.collapsed is (inputs > COLLAPSE_AT)
    assert view.after.collapsed is (inputs + 1 > COLLAPSE_AT)


def test_the_fixed_effects_preview_draws_no_indicator_once_the_grouping_is_recorded(grouped):
    """The guard's other side: the same grouping recorded, nothing arrives, and no view says
    `clinic` enters."""
    decision = SetClusters(column="clinic", adjust="fixed_effects")
    result = plan(decision, _recorded(grouped, decision, purpose="inference"), basis="test")
    assert "`clinic` as fixed effects" not in _lineage_titles(result)


ITEMS = [f"item{j}" for j in range(4)]


@pytest.fixture(scope="module", params=[WIDE, NARROW], ids=["folded", "never-folded"])
def scaled(request, tmp_path_factory):
    """:func:`_table` with four answers on a 1–5 scale among its predictors (``inputs`` in all)."""
    frame, roles = _table(request.param - len(ITEMS))
    rng = np.random.default_rng(4)
    for c in ITEMS:
        frame[c] = rng.integers(1, 6, N).astype(float)
        roles[c] = "covariate"
    found = _ingested(request.param, frame, roles, tmp_path_factory.mktemp("scaled"))
    yield found
    found[3].close()


SCALE = SetScales(scales=[ScaleSpec(name="score", items=ITEMS, low=1, high=5, kind="reflective")])
ITEMS_READ = {f"code_or_count:{c}": "amount" for c in ITEMS}


def test_the_scales_preview_draws_the_items_leaving_on_a_wide_plan(scaled):
    """The four items leave the model and the score enters as one column, though the wide plan's
    drawing folds the items (on the base the wide plan dropped the view: no column was drawn
    arriving or leaving)."""
    from turbotab.core.plan_previews import fit_design

    ctx = _context(scaled, shape_confirmations=ITEMS_READ)
    assert len(fit_design(ctx, ctx.state, ctx.sample_row_ids(scaled[4])).spec.inputs) == scaled[0]
    result = plan(SCALE, ctx, basis="test")
    view = _the_lineage_view(result, "Items into one score")
    assert view.caption == (f"`{len(ITEMS)}` items leave the model; `score` enters as one "
                            f"column.")
    assert view.before.collapsed is (scaled[0] > COLLAPSE_AT)


def test_the_scales_preview_draws_no_items_leaving_once_the_scale_is_recorded(scaled):
    """The guard's other side: the scale recorded, no column arrives or leaves, and no view says
    the items leave."""
    result = plan(SCALE, _recorded(scaled, SCALE, shape_confirmations=ITEMS_READ), basis="test")
    assert "Items into one score" not in _lineage_titles(result)


# The omics table of ``preview_fixtures.batch_table``: features `m..` measured in three batches.
# Reference ComBat reads `batch` as one more input of the plan and leaves the model none: the
# answer's drawing has a raw column more than the plan's, so the one a feature fewer than the fold
# is the plan neither drawing folds, and at the fold only the answer's drawing folds.
BATCHED = {"folded": WIDE, "the answer's drawing folded": NARROW, "never-folded": NARROW - 1}
COMBAT = SetBatch(column="batch", method="reference_combat")


@pytest.fixture(scope="module", params=list(BATCHED.values()), ids=list(BATCHED))
def batched(request, tmp_path_factory):
    """``(features, project)``: the batch table's plan under prediction, every `m..` feature an
    exposure, `batch` and the sample id no predictor (``preview_fixtures.batch_project``'s plan,
    ``features`` wide)."""
    from turbotab.core import decisions as d
    from turbotab.core.decisions import ProjectState
    from turbotab.core.tests.acceptance import preview_fixtures as F
    from turbotab.core.tests.acceptance.preview_harness import Project

    frame = F.batch_table(p=request.param)
    roles = {"sample_id": "identifier", "batch": "excluded",
             **{c: "exposure" for c in frame.columns if c.startswith("m")}}
    state = ProjectState(
        lens=["metabolomics"], target="y", task="regression", purpose="prediction",
        roles=roles, role_confirmations=dict(roles),
        grain=d.GrainSpec(grain="one_row_per_unit", id_column="sample_id"),
        exclusions=[], missing=d.MissingSpec(strategy="complete_case"),
        split=d.SplitSpec(holdout=0.2, seed=0, folds=5), models=["linear"])
    project = Project(frame, Path(tmp_path_factory.mktemp("batched")), state,
                      upto=["design", "cohort", "split", "findings"])
    yield request.param, project
    project.close()


def test_the_batch_preview_emphasizes_no_feature_as_arriving_on_a_wide_plan(batched):
    """Reference ComBat under prediction: every feature keeps its name and `batch` is no column of
    the model, so nothing arrives and the view emphasizes `batch` alone, wide plan or narrow. On
    the base the wide plan emphasized `batch`, `m00`, `m01` and nine more: features folded in the
    plan's drawing and drawn in the answer's, which the compare of two drawings read as arriving."""
    features, project = batched
    result, _ = project.preview(COMBAT)
    view = _the_lineage_view(result, "Where `batch` enters")
    assert view.caption == (f"`{features}` features batch-adjusted inside each training fold, "
                            f"without the outcome.")
    assert view.emphasis == ["batch"]
    # the premise: the drawings fold as the table is built to (`batch` one raw column more after)
    assert view.before.collapsed is (features > COLLAPSE_AT)
    assert view.after.collapsed is (features + 1 > COLLAPSE_AT)
