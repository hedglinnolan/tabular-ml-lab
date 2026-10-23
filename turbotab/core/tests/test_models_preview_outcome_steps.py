"""The Models preview of a family whose own step reads the outcome.

Package 3's crash sweep found it (2026-10-23): the preview of the default plan's answer
``select_models [elastic_net, screened_elastic_net]`` answered 500. The preview reads no outcome,
and it left out only the selection step; the screened elastic net's own step, ``screen``
(``omics.UnivariateScreen``), refuses to be fitted without the outcome ("Screening reads the outcome
of the rows it is fit on."). The preview now leaves out every step that can be fitted only with the
outcome, shows the matrix of the steps that read none, and names the step it left out by the label
the design's step list gives it (``pipeline.describe_steps``).

Each claim from construction: the screened elastic net is the elastic net with the screen added
before the scaling (its pipeline's own steps, read here), so the matrix the preview shows for it,
without the screen, is the elastic net's, column for column; the elastic net, which has no such
step, previews as it does alone. A family's own step that fails with an outcome as well as without
one is not taken for one that reads it: its error is not hidden.

Explore's forms (the verifier's P3R1-1, below): the forms by inner cross-validation read the outcome
they are given, so they are left out and named, and the matrix is that of no forms; the forms by
the stated rule are left out and named only where Harrell's effective size is the rarer class's
count (a yes/no outcome, counted by pandas), and under a numeric outcome show each continuous column
as Harrell's rule splines it above 100 rows, as before; the variance filter reads no outcome and
previews as before. Three steps left out are named by their count, within the word budget.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin

from turbotab.core import decisions as d
from turbotab.core.tests.acceptance import test_previews_leash as L
from turbotab.core.tests.acceptance.preview_harness import Project

BOTH = ["elastic_net", "screened_elastic_net"]
READS = "each training fold's outcome"


def by_title(served: dict) -> dict[str, dict]:
    return {v["title"]: v for v in served["views"]}


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    """The walk's table under Predict (glucose on the diet, age, BMI and sex; carb, protein and
    fat_total the study factors the screen correlates with the outcome), split, with no
    selection."""
    state = L.walk_state(purpose="prediction", selection=d.SelectionSpec(method="none"))
    p = Project(L.walk_table(False), Path(tmp_path_factory.mktemp("screen")), state,
                upto=["design", "cohort", "split"])
    yield p, state
    p.close()


@pytest.fixture(scope="module")
def previews(project):
    p, state = project
    return {"both": L.served(p, d.SelectModels(models=BOTH), state),
            "alone": L.served(p, d.SelectModels(models=["elastic_net"]), state)}


def test_the_screened_family_is_the_elastic_net_with_the_screen_before_its_scaling(project):
    """The construction the rest reads: the two families' pipelines, as the design builds them."""
    from turbotab.core.models import get_family
    from turbotab.core.models.pipeline import design_spec, family_steps, model_predictors
    from turbotab.core.models.previews import _read

    p, state = project
    predictors = model_predictors(state)
    frame = _read(p.context(state=state), list(p.store.columns))
    spec = design_spec(state, frame, predictors)
    plain = [n for n, _ in family_steps(spec, get_family("elastic_net"))]
    screened = [n for n, _ in family_steps(spec, get_family("screened_elastic_net"))]
    assert "screen" in screened and "screen" not in plain
    assert [n for n in screened if n != "screen"] == plain
    assert screened.index("screen") == screened.index("scale") - 1


def test_the_preview_of_both_families_answers_and_shows_the_steps_that_read_no_outcome(previews):
    both = previews["both"]
    assert "refused" not in both and len(both["views"]) == 2, both
    views = by_title(both)
    plain, screened = views["Model matrix for elastic net"], views[
        "Model matrix for screened elastic net"]
    # Without the screen, the screened family's steps are the elastic net's: the same matrix.
    assert screened["after"] == plain["after"]
    assert screened["before"] == plain["before"]


def test_the_caption_names_the_screen_by_the_design_step_lists_label(previews):
    from turbotab.core.models import get_family

    label, _ = get_family("screened_elastic_net").describe_step("screen")
    caption = by_title(previews["both"])["Model matrix for screened elastic net"]["caption"]
    assert f"without “{label}”, which reads {READS}" in caption, caption
    assert caption.startswith("Screened elastic net: ")


def test_with_a_selection_too_the_caption_names_both_within_its_word_budget(project):
    """The selection and the screen both left out: each named in the one clause, the selection as
    the selection (by its method's name alone, the elastic net would read as the family left
    out of its own caption), the budget's first cut (at a semicolon, a colon or a comma) never
    reached."""
    from turbotab.core.consequences import CAPTION_WORDS
    from turbotab.core.models import get_family
    from turbotab.core.models.variable_selection import NAMES

    p, _ = project
    state = L.walk_state(purpose="prediction", selection=d.SelectionSpec(method="elastic_net"))
    served = L.served(p, d.SelectModels(models=["screened_elastic_net"]), state)
    assert "refused" not in served, served
    (view,) = served["views"]
    label, _ = get_family("screened_elastic_net").describe_step("screen")
    clause = f"without the selection and “{label}”, which read {READS}"
    assert clause in view["caption"] and len(view["caption"].split()) <= CAPTION_WORDS, view
    assert NAMES["elastic_net"] not in view["caption"], view


def test_a_family_with_no_such_step_previews_as_it_does_alone(previews):
    plain = by_title(previews["both"])["Model matrix for elastic net"]
    (alone,) = previews["alone"]["views"]
    assert plain == alone
    assert READS not in plain["caption"]


class Broken(TransformerMixin, BaseEstimator):
    """A family's own step that fails on these rows whether or not it is given an outcome."""

    def fit(self, X, y=None):
        raise ValueError("This step fails on these rows, outcome or not.")

    def transform(self, X):  # pragma: no cover - never fitted
        return X

    def get_feature_names_out(self, input_features=None):  # pragma: no cover - never fitted
        return input_features


def test_a_step_that_fails_with_an_outcome_too_is_not_taken_for_one_that_reads_it(project):
    from turbotab.core.models import base, get_family
    from turbotab.core.models.elastic_net import ElasticNet

    class BrokenNet(ElasticNet):
        key = "broken_net"
        label = "Broken net"

        def preprocess(self, spec):
            return [("broken", Broken())]

    p, state = project
    base.register_family(BrokenNet())
    try:
        assert get_family("broken_net").key == "broken_net"
        pctx = p.context(state=state)
        with pytest.raises(ValueError, match="outcome or not"):
            p.preview(d.SelectModels(models=["broken_net"]), ctx=pctx)
    finally:
        base.unregister_family("broken_net")


# ── Explore's forms: steps that read the outcome they are given, and none that refuse without it ──
#
# The verifier's P3R1-1 (2026-10-23): the preview left out the selection and the steps that refuse
# to be fitted without an outcome, and fitted every other step without one. Explore's forms by
# inner cross-validation (``levers.InnerCVForms``) read the outcome they are given and, given none,
# keep every input linear, so the preview showed a matrix the forms step chose nothing for and
# called it "lever forms". The forms by the stated rule (``levers.RuleSplines``) read the outcome
# only for its class counts: under a yes/no outcome Harrell's effective size is the rarer class's
# count, which sets the number of knots, so without the outcome the preview took the rows' count
# and showed more spline columns than the fit builds. Each such step is now left out and named.


def lever_state(state: d.ProjectState, forms: str = "none", variance_filter: str = "none"):
    return state.model_copy(update={"levers": d.LeverSpec(forms=forms,
                                                          variance_filter=variance_filter)})


def matrix(view: dict) -> list[str]:
    return [n["column"] for n in view["after"]["nodes"] if n["lane"] == "matrix"]


def one_view(served: dict) -> dict:
    assert "refused" not in served, served
    (view,) = served["views"]
    return view


def step_label(project: Project, state: d.ProjectState, family_key: str, step: str) -> str:
    """The label the design's step list gives ``step`` (``pipeline.describe_steps``)."""
    from turbotab.core.models import get_family
    from turbotab.core.models.pipeline import design_spec, describe_steps, model_predictors
    from turbotab.core.models.previews import _read

    predictors = model_predictors(state)
    frame = _read(project.context(state=state), list(project.store.columns))
    spec = design_spec(state, frame, predictors)
    listed = {s["key"]: s["label"] for s in describe_steps(spec, get_family(family_key), state.task,
                                                            state.purpose)}
    return listed[step]


def training_frame(project: Project, table: pd.DataFrame) -> pd.DataFrame:
    """``table``, as written, at the project's training rows (by ``SEQN``)."""
    a = project.artifacts["split"].frames["assignment"]
    ids = np.sort(a.loc[a["partition"] == "train", "row_id"].to_numpy(dtype="int64"))
    seqn = project.store.materialize(["SEQN"], ids)["SEQN"].to_numpy()
    return table.set_index("SEQN").loc[seqn]


@pytest.fixture(scope="module")
def levers(project):
    p, state = project
    cases = {"none": ("none", "none"), "inner_cv": ("inner_cv", "none"), "rule": ("rule", "none"),
             "filter": ("none", "near_zero")}
    return {key: one_view(L.served(p, d.SelectModels(models=["elastic_net"]),
                                   lever_state(state, *answer)))
            for key, answer in cases.items()}


def test_forms_by_inner_cross_validation_read_the_outcome_they_are_given():
    """The construction that makes them an outcome-reading step: y bends in a and not in b, and the
    inner comparison splines a only when it is given y; given none, it keeps every input linear."""
    from turbotab.core.methods.levers import InnerCVForms

    rng = np.random.default_rng(7)
    X = pd.DataFrame({"a": rng.uniform(-2, 2, 400), "b": rng.normal(size=400)})
    y = X["a"].to_numpy() ** 2 + rng.normal(scale=0.3, size=400)
    given = InnerCVForms(["a", "b"], "regression").fit(X, y)
    assert set(given.knots_) == {"a"}
    blind = InnerCVForms(["a", "b"], "regression").fit(X)
    assert blind.knots_ == {} and blind.note_


def test_the_preview_under_inner_cv_forms_leaves_the_forms_step_out_and_names_it(project, levers):
    p, state = project
    view = levers["inner_cv"]
    # Without the forms step, the elastic net's steps are those of no forms: the same matrix.
    assert view["after"] == levers["none"]["after"]
    label = step_label(p, lever_state(state, "inner_cv"), "elastic_net", "lever_forms")
    assert f"without “{label}”, which reads {READS}" in view["caption"], view["caption"]
    assert "lever forms" not in view["caption"]


def test_no_forms_preview_as_no_levers_do(previews, levers):
    (alone,) = previews["alone"]["views"]
    assert levers["none"] == alone


def test_forms_by_the_rule_under_a_numeric_outcome_preview_as_before(project, levers):
    """Under a numeric outcome the rule's effective size is the rows' count, read without the
    outcome: Harrell's rule gives 5 knots above 100 rows, so each continuous column of the matrix
    becomes itself and 3 nonlinear terms; the two coded levels of gender stay as they are."""
    p, _ = project
    assert len(training_frame(p, L.walk_table(False))) > 100
    view = levers["rule"]
    expected = []
    for column in matrix(levers["none"]):
        expected += [column] if column.startswith("gender_") else [column + "'" * j
                                                                   for j in range(4)]
    assert matrix(view) == expected
    assert READS not in view["caption"] and "without" not in view["caption"], view["caption"]


def test_the_variance_filter_previews_as_before(project, levers):
    """No predictor is near-zero variance by caret's rule (the most common value over the second
    above 95/5 with at most 10% distinct values, or one value), pandas' counts on the training
    rows: the filter keeps every column, and reads no outcome."""
    p, _ = project
    frame = training_frame(p, L.walk_table(False))
    for column in ["gender", "age", "bmi", "kcal", "carb", "protein", "fat_total"]:
        counts = frame[column].value_counts()
        ratio = counts.iloc[0] / counts.iloc[1]
        assert len(counts) > 1 and not (ratio > 95 / 5 and len(counts) / len(frame) <= 0.10), column
    view = levers["filter"]
    assert matrix(view) == matrix(levers["none"])
    assert all("kept by the in-fold variance filter" in link["operation"]
               for link in view["after"]["links"] if link["target"].startswith("mx:"))
    assert READS not in view["caption"] and "without" not in view["caption"], view["caption"]


def test_three_steps_left_out_are_counted_within_the_word_budget(project):
    """The selection, the forms by inner cross-validation and the screen: three steps that read
    the outcome, named by their count in one clause the budget keeps whole."""
    from turbotab.core.consequences import CAPTION_WORDS

    p, _ = project
    state = lever_state(L.walk_state(purpose="prediction",
                                     selection=d.SelectionSpec(method="elastic_net")), "inner_cv")
    view = one_view(L.served(p, d.SelectModels(models=["screened_elastic_net"]), state))
    assert f"without the 3 steps that read {READS}" in view["caption"], view["caption"]
    assert len(view["caption"].split()) <= CAPTION_WORDS, view["caption"]


RARE = 0.9  # the yes/no outcome: glucose above its 90th percentile, so about 60 rows of 600


@pytest.fixture(scope="module")
def binary(tmp_path_factory):
    """The walk's table under Predict with a yes/no outcome whose rarer class has between 30 and
    100 training rows, while the rows number more than 100."""
    table = L.walk_table(False)
    cut = table["glucose"].quantile(RARE)
    table["high"] = np.where(table["glucose"].isna(), None,
                             np.where(table["glucose"] > cut, "yes", "no"))
    roles = {**L.ROLES, "high": "excluded"}
    state = L.walk_state(purpose="prediction", target="high", task="binary", event="yes",
                         roles=roles, role_confirmations=dict(roles),
                         selection=d.SelectionSpec(method="none"))
    p = Project(table, Path(tmp_path_factory.mktemp("rule_binary")), state,
                upto=["design", "cohort", "split"])
    yield p, state, table
    p.close()


def test_forms_by_the_rule_under_a_yes_no_outcome_are_left_out_and_named(binary):
    """Harrell's effective size under a yes/no outcome is the rarer class's count (pandas, on the
    training rows): between 30 and 100, so the rule's fit takes 4 knots, while the rows, which a
    fit without the outcome counts, number more than 100, for 5. The matrix the preview could show
    without the outcome is not the fit's, so the step is left out and named."""
    from turbotab.core.methods.levers import RuleSplines

    p, state, table = binary
    frame = training_frame(p, table)
    recorded = frame["high"].dropna()
    rarer = int(recorded.value_counts().min())
    assert 30 <= rarer <= 100 < len(recorded)
    X = frame.loc[recorded.index, ["age"]].astype(float)
    assert RuleSplines(["age"], "binary").fit(X, recorded.to_numpy()).k_ == 4
    assert RuleSplines(["age"], "binary").fit(X).k_ == 5

    none = one_view(L.served(p, d.SelectModels(models=["elastic_net"]), lever_state(state)))
    view = one_view(L.served(p, d.SelectModels(models=["elastic_net"]),
                             lever_state(state, "rule")))
    assert view["after"] == none["after"]
    label = step_label(p, lever_state(state, "rule"), "elastic_net", "lever_forms")
    assert f"without “{label}”, which reads {READS}" in view["caption"], view["caption"]
