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

Expected values come from construction (the table's predictors, the residual method's nutrients,
the four levels of `site`, a quintile form's five groups) and from scikit-learn's
``OneHotEncoder`` (the indicators of the text columns, and so the width of the matrix).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import OneHotEncoder

from turbotab.core.consequences import PreviewContext, after_state, plan
from turbotab.core.datastore import DataStore
from turbotab.core.decisions import (SelectModels, SetCategorical, SetEnergyAdjustment,
                                     SetExposureForm)
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


def _table(inputs: int) -> tuple[pd.DataFrame, dict[str, str]]:
    """:func:`modeling_fixtures.nhanes_like`'s columns, the :data:`TEXT` columns, numeric `z..`
    columns and a whole-numbered `site` of four levels, last, so that the plan has ``inputs``
    predictors; and the roles."""
    rng = np.random.default_rng(5)
    frame = mf.nhanes_like(N, seed=21)
    for c in TEXT:
        frame[c] = rng.choice(["a", "b", "c"], N)
    numeric = [f"z{j:02d}" for j in range(inputs - len(NHANES_INPUTS) - len(TEXT) - 1)]
    for c in numeric:
        frame[c] = rng.normal(size=N)
    frame["site"] = rng.choice([1, 2, 3, 4], N)
    roles = {**mf.NHANES_ROLES, **{c: "covariate" for c in [*TEXT, *numeric, "site"]}}
    assert sum(r in PREDICTOR_ROLES for r in roles.values()) == inputs
    return frame, roles


@pytest.fixture(scope="module", params=[WIDE, NARROW], ids=["folded", "never-folded"])
def project(request, tmp_path_factory):
    frame, roles = _table(request.param)
    paths = mf.ingest_frame(frame, tmp_path_factory.mktemp("wide"))
    store = DataStore(Path(paths["data"]), 1 << 30)
    split = mf.split_bundle(np.arange(N), seed=21)
    a = split.frames["assignment"]
    train = a.loc[a["partition"] == "train", "row_id"].to_numpy()
    yield request.param, frame, roles, store, train
    store.close()


def _context(project, *, site_settled: bool = True) -> PreviewContext:
    """`site` read as an amount, or (``site_settled`` False) its whole numbers not yet read: codes
    or amounts, which a categorical answer settles."""
    _, _, roles, store, train = project
    confirmed = {**mf.NHANES_TRUTH, **({"code_or_count:site": "amount"} if site_settled else {})}
    st = mf.state(energy_adjustment=None, roles=roles, shape_confirmations=confirmed)
    return PreviewContext(project_id="t", state=st, datastore=store, artifact=lambda name: None,
                          training_row_ids=train, cohort_row_ids=None, sample_size=1000)


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
