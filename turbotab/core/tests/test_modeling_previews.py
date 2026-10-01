"""Tier B: the modeling agent's consequence previews (M1_CONTRACT §4) — shapes, budgets, rows."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from turbotab.core.consequences import (
    CAPTION_WORDS,
    FRAME_WORDS,
    MAX_VIEWS,
    TITLE_WORDS,
    PreviewContext,
    PreviewResult,
    plan,
    words,
)
from turbotab.core.datastore import DataStore
from turbotab.core.decisions import SelectModels, SetEnergyAdjustment
from turbotab.core.methods.energy import EnergyAdjuster
from turbotab.core.models.previews import fit_words, lineage_caption
from turbotab.core.tests import modeling_fixtures as mf

# Macronutrient totals in grams; a fat subtype would make the partition refuse (its unit is unconfirmed).
NUTRIENTS = ["protein", "carb", "fat_total"]


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    frame = mf.nhanes_like(1500, seed=21)
    paths = mf.ingest_frame(frame, tmp_path_factory.mktemp("previews"))
    store = DataStore(Path(paths["data"]), 1 << 30)
    split = mf.split_bundle(np.arange(len(frame)), seed=21)
    a = split.frames["assignment"]
    train = a.loc[a["partition"] == "train", "row_id"].to_numpy()
    yield frame, store, train
    store.close()


def context(store, st, train) -> PreviewContext:
    return PreviewContext(project_id="t", state=st, datastore=store, artifact=lambda name: None,
                          training_row_ids=train, cohort_row_ids=None, sample_size=1000)


def energy_decision(method: str, **kw) -> SetEnergyAdjustment:
    if method == "none":
        return SetEnergyAdjustment(method="none")
    return SetEnergyAdjustment(method=method, energy_column="kcal", nutrients=NUTRIENTS, **kw)


def check_budgets(result: PreviewResult) -> None:
    PreviewResult.model_validate(result.model_dump())
    assert 1 <= len(result.views) <= MAX_VIEWS
    for view in result.views:
        assert words(view.title) <= TITLE_WORDS, view.title
        assert words(view.caption) <= CAPTION_WORDS, view.caption
        assert view.caption.endswith((".", "…")), view.caption


@pytest.mark.parametrize("method", ["none", "standard", "residual", "density_multivariate", "density",
                                    "partition"])
def test_every_energy_option_previews_within_the_budgets(project, method):
    _, store, train = project
    st = mf.state(energy_adjustment=None)
    result = plan(energy_decision(method), context(store, st, train), basis="test")
    check_budgets(result)
    kinds = [v.kind for v in result.views]
    assert kinds[0] == "relationship" and kinds[1] == "lineage"
    changes_values = method in ("residual", "density", "density_multivariate", "partition")
    assert kinds == (["relationship", "lineage", "distribution"] if changes_values
                     else ["relationship", "lineage"])
    rel = result.views[0]
    assert rel.x_label == "kcal" and len(rel.points_before) <= 800
    assert len(rel.points_before) == len(rel.points_after) > 0


def test_the_residual_preview_shows_the_most_energy_correlated_nutrient_losing_its_correlation(project):
    frame, store, train = project
    st = mf.state(energy_adjustment=None)
    ctx = context(store, st, train)
    result = plan(energy_decision("residual"), ctx, basis="test")
    rel, lineage, dist = result.views
    sample = frame.loc[ctx.sample_row_ids(train)]
    strongest = max(NUTRIENTS, key=lambda n: abs(np.corrcoef(sample[n], sample["kcal"])[0, 1]))
    assert rel.y_label_before == strongest and rel.y_label_after == f"{strongest}_adj"
    assert rel.r_before == pytest.approx(np.corrcoef(sample[strongest], sample["kcal"])[0, 1], abs=1e-12)
    assert abs(rel.r_after) < 1e-8  # zero by construction on the rows it was fit on
    assert f"`{strongest}` correlates" in rel.caption and "0.00" in rel.caption
    # The model loses kcal and gains the adjusted nutrients.
    assert lineage.caption.startswith("`kcal`, `protein` and 2 more leave; `protein_adj`")
    assert {f"{n}_adj" for n in NUTRIENTS} <= set(lineage.emphasis)
    assert dist.column == strongest and sum(dist.before.counts) == len(sample)


def test_the_preview_is_fit_on_training_rows_only(project, monkeypatch):
    _, store, train = project
    seen: list[set] = []
    original = EnergyAdjuster.fit

    def spy(self, X, y=None):
        seen.append(set(X.index.tolist()))
        return original(self, X, y)

    monkeypatch.setattr(EnergyAdjuster, "fit", spy)
    plan(energy_decision("residual", strata="gender"), context(store, mf.state(), train), basis="test")
    plan(SelectModels(models=["elastic_net"]), context(store, mf.state(energy_adjustment=mf.energy()), train),
         basis="test")
    assert seen
    allowed = set(train.tolist())
    assert all(rows <= allowed for rows in seen)


def test_before_the_split_there_is_nothing_to_fit_on(project):
    _, store, _ = project
    result = plan(energy_decision("residual"), context(store, mf.state(), None), basis="test")
    assert result.views == [] and result.note


def test_a_method_that_cannot_run_says_why_on_the_users_rows(project, tmp_path):
    frame = mf.nhanes_like(300, seed=22)
    frame.loc[:4, "kcal"] = 0.0
    paths = mf.ingest_frame(frame, tmp_path)
    with DataStore(Path(paths["data"]), 1 << 30) as store:
        result = plan(energy_decision("density"), context(store, mf.state(), np.arange(len(frame))),
                      basis="test")
    check_budgets(result)
    (rel,) = result.views
    assert rel.points_after == [] and rel.r_after is None
    assert "kcal is zero or negative in 5" in rel.caption


def test_each_family_previews_its_own_model_matrix_new_ones_first(project):
    _, store, train = project
    st = mf.state(energy_adjustment=mf.energy("residual"), models=["linear"])
    result = plan(SelectModels(models=["linear", "boosted_trees", "elastic_net"]),
                  context(store, st, train), basis="test")
    check_budgets(result)
    assert [v.kind for v in result.views] == ["lineage"] * 3
    trees, net, linear = result.views
    assert trees.title == "Model matrix for boosted trees"
    assert net.title == "Model matrix for elastic net"
    assert "unscaled" in trees.caption and "unscaled" in linear.caption
    assert "standardized on training rows" in net.caption
    assert trees.after == trees.before  # trees skip scaling: the matrix is the shared one
    ops = {link.operation for link in net.after.links if link.target.startswith("mx:")}
    assert "scaled" in ops and "one-hot, scaled" in ops


def test_caption_helpers_stay_within_their_budgets():
    long = "one two three four five six seven eight nine ten, eleven twelve thirteen fourteen"
    assert fit_words(long, 10) == "one two three four five six seven eight nine ten."
    assert words(fit_words(" ".join(["w"] * 40), 20)) <= 20
    caption = lineage_caption(["a", "kcal", "b"], ["a", "b", "x_adj", "y_adj", "z_adj"])
    assert caption == "`kcal` leaves; `x_adj`, `y_adj` and 1 more arrive. The model sees 5 columns."
    assert lineage_caption(["a"], ["a"]) == "The model matrix keeps the same 1 column."


def test_a_family_registered_later_previews_and_traces_itself_with_no_new_code(project, monkeypatch):
    """The scaling strategy: a new family declares its steps; lineage, steps and preview follow."""
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import SplineTransformer

    from turbotab.core.models import lineage as lineage_module
    from turbotab.core.models import previews
    from turbotab.core.models.base import Assessment, FamilyBase, register_family, unregister_family
    from turbotab.core.models.pipeline import DesignSpec, describe_steps

    class SplineLinear(FamilyBase):
        key = "spline_linear"
        label = "Spline model"
        tasks = ("regression",)
        inductive_bias = "Smooth curves for age; straight lines for everything else."
        strengths = ("Bends where the data bend.",)
        cautions = ("Knots are a choice.",)

        def preprocess(self, spec: DesignSpec):
            basis = SplineTransformer(n_knots=4, degree=3)
            return [("spline", ColumnTransformer([("spline", basis, ["age"])], remainder="passthrough",
                                                 verbose_feature_names_out=False))]

        def describe_step(self, name):
            return ("Spline basis", "age becomes six smooth basis columns.") if name == "spline" else None

        def build(self, task, purpose, n_rows, n_features):
            return LinearRegression()

        def describe(self, task, purpose):
            return "Least squares on a spline basis", "Fits a smooth curve for age."

        def assess(self, situation):
            return Assessment(1.0, "good")

    _, store, train = project
    monkeypatch.setitem(lineage_module.OPERATIONS, "SplineTransformer", "spline basis")
    monkeypatch.setitem(previews.STEP_PHRASES, "spline", "age as a spline basis")
    register_family(SplineLinear())
    try:
        st = mf.state(energy_adjustment=mf.energy("residual"), models=["linear"])
        result = plan(SelectModels(models=["spline_linear"]), context(store, st, train), basis="test")
        check_budgets(result)
        (view,) = result.views
        assert "age as a spline basis" in view.caption
        into = {(link.source, link.operation) for link in view.after.links if link.target == "mx:age_sp_0"}
        assert into == {("adj:age", "spline basis")}
        assert not any(n.column == "age" for n in view.after.nodes if n.lane == "matrix")
        spec = DesignSpec(predictors=["age", "kcal"], inputs=["age", "kcal"], categorical=[],
                          numeric=["age", "kcal"], energy=None, impute=False)
        steps = describe_steps(spec, SplineLinear(), "regression", "prediction")
        assert [s["label"] for s in steps] == ["Spline basis", "Least squares on a spline basis"]
    finally:
        unregister_family("spline_linear")


# ── storyboards (M1_CONTRACT §12.1) ──────────────────────────────────────────


def check_story(result: PreviewResult) -> None:
    for view in result.views:
        for frame in view.story:
            assert 1 <= words(frame.label) <= FRAME_WORDS, frame.label


def test_the_residual_storyboard_fits_the_line_then_keeps_the_residuals_centered_at_zero(project):
    frame, store, train = project
    ctx = context(store, mf.state(energy_adjustment=None), train)
    result = plan(energy_decision("residual"), ctx, basis="test")
    check_story(result)
    rel, lineage, dist = result.views
    fit, resid = rel.story
    assert fit.points == rel.points_before and fit.r == pytest.approx(rel.r_before, abs=1e-12)
    n = rel.y_label_before
    sample = frame.loc[ctx.sample_row_ids(train)]
    slope, intercept = np.polyfit(sample["kcal"], sample[n], 1)
    assert fit.fit_line.slope == pytest.approx(slope, rel=1e-9)
    assert fit.fit_line.intercept == pytest.approx(intercept, rel=1e-9)
    residuals = (sample[n] - (intercept + slope * sample["kcal"])).to_numpy()
    assert abs(residuals.mean()) < 1e-9 and abs(resid.r) < 1e-8  # centered; energy explains none of it
    ys = np.array([y for _, y in resid.points])
    after = np.array([y for _, y in rel.points_after])
    # The after is the residual plus the one average added back: the same points, shifted.
    np.testing.assert_allclose(after - ys, np.full(len(ys), (after - ys)[0]), atol=1e-9)
    assert resid.y_label == f"{n} residual" and resid.fit_line.slope == 0.0
    assert [f.label for f in dist.story] == [fit.label, resid.label]
    assert sum(dist.story[1].hist.counts) == len(sample)
    assert lineage.story == []


def test_the_partition_storyboard_splits_energy_into_its_parts_on_the_lineage(project):
    _, store, train = project
    result = plan(energy_decision("partition"), context(store, mf.state(energy_adjustment=None), train),
                  basis="test")
    check_story(result)
    rel, lineage, dist = result.views
    assert rel.story == [] and dist.story == []
    first, second = lineage.story
    adjusted = [{n.column for n in f.lineage.nodes if n.lane == "adjusted"} for f in (first, second)]
    assert {"kcal_from_protein", "kcal_from_carb", "kcal"} <= adjusted[0]
    assert "kcal_from_other" not in adjusted[0] and {"kcal_from_other", "kcal"} <= adjusted[1]
    after = {n.column for n in lineage.after.nodes if n.lane == "adjusted"}
    assert "kcal" not in after and "kcal_from_other" in after  # the after lets energy go
    ids = {n.id for f in (first, second) for n in f.lineage.nodes}
    assert all(link.source in ids and link.target in ids for f in (first, second) for link in f.lineage.links)


@pytest.mark.parametrize("method", ["none", "standard", "density", "density_multivariate"])
def test_methods_without_intermediate_steps_have_no_storyboard(project, method):
    _, store, train = project
    result = plan(energy_decision(method), context(store, mf.state(energy_adjustment=None), train),
                  basis="test")
    assert all(view.story == [] for view in result.views)
