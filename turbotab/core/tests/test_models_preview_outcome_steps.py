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
"""
from __future__ import annotations

from pathlib import Path

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
