"""A form previewed for a column the model does not hold (D1 of the 2026-10 sweep: on Sullivan 2021,
adjustment answers that left `RIDAGEYR` out of the model, then the plan's spline for it previewed
again, raised KeyError in ``plan_previews``, which the server returned as a 500).

A form shapes only a column the model holds (``exposure_form.design_forms``: the present
predictors), so a preview of one the answers leave out draws nothing and says so in its note, a
typed answer like ``set_categorical``'s for a column outside the model; the one-tap answer draws a
column the model holds when it names one. The truth is the construction's: ESTIMAND's cohort, whose
declared answers make `hscrp` a consequence of fiber (after the exposure, so the total effect's
primary model leaves it out), `bmi` of unknown timing (left out of the primary, beside it in the
further-adjusted model) and `age` a confounder the model holds; `age`'s knots are Harrell's
percentiles of its analyzed rows (NumPy), as ``test_previews_wave2b`` holds them.
"""
from __future__ import annotations

import numpy as np
import pytest

from turbotab.core import decisions as d
from turbotab.core.consequences import PreviewResult
from turbotab.core.tests.acceptance import preview_fixtures as F
from turbotab.core.tests.acceptance.test_previews_kinds import Planned

# Harrell's default knot percentiles for four knots (RMS 2nd ed. §2.4.6).
HARRELL_4 = [0.05, 0.35, 0.65, 0.95]
SPLINE = d.ExposureFormSpec(form="spline", knots=4)


@pytest.fixture(scope="module")
def est(tmp_path_factory):
    planned = Planned(F.estimand_project(tmp_path_factory.mktemp("form_preview_left_out")))
    yield planned
    planned.project.close()


def typed(result) -> PreviewResult:
    """The preview as the route serves it: it validates as the response model."""
    return PreviewResult.model_validate(result.model_dump(mode="json"))


def analyzed_age(planned: Planned) -> np.ndarray:
    a = planned.project.artifacts["split"].frames["assignment"]
    ids = np.sort(a["row_id"].to_numpy(dtype="int64"))
    return planned.project.store.materialize(["age"], ids)["age"].to_numpy(dtype=float)


def test_the_construction_answers_hscrp_and_bmi_out_of_the_primary_model(est):
    from turbotab.core.models.pipeline import model_predictors

    assert F.ESTIMAND_ANSWERS["hscrp"]["after_exposure"] == "yes"
    assert F.ESTIMAND_ANSWERS["bmi"]["after_exposure"] == "unknown"
    held = model_predictors(est.project.state)
    assert "hscrp" not in held and "bmi" not in held and "age" in held


def test_a_form_for_a_column_the_answers_leave_out_is_said_not_drawn(est):
    one, _ = est.preview("forms:hscrp", d.SetForms(forms={"hscrp": SPLINE}))
    single, _ = est.preview("form:hscrp", d.SetExposureForm(column="hscrp", form="spline", knots=4))
    for result in (one, single):
        assert typed(result).views == [] and result.caution is None
        assert result.note == ("The adjustment answers leave `hscrp` out of the primary model, so "
                               "its form changes nothing that model sees.")
    both, _ = est.preview("forms:hscrp,bmi", d.SetForms(forms={"hscrp": SPLINE, "bmi": SPLINE}))
    assert typed(both).views == []
    assert both.note == ("The adjustment answers leave `hscrp` and `bmi` out of the primary model, "
                         "so their forms change nothing that model sees.")


def test_a_column_left_out_by_another_answer_is_not_among_the_predictors(est):
    """The missing-values answer leaves `activity` out (``decisions.left_out``), not the adjustment
    answers, so the note names no adjustment answer."""
    state = est.project.state.model_copy(update={"missing": d.MissingSpec(
        strategy="complete_case", drop_columns=["activity"])})
    result, _ = est.preview("forms:activity dropped", d.SetForms(forms={"activity": SPLINE}),
                            state=state)
    assert typed(result).views == []
    assert result.note == ("`activity` is not among the model's predictors, so its form changes "
                           "nothing the model sees.")


def test_the_one_tap_answer_draws_the_column_the_model_holds(est):
    """Named after a column the answers leave out, `age` is drawn as its own answer draws it."""
    mixed, mixed_ctx = est.preview("forms:hscrp,age",
                                   d.SetForms(forms={"hscrp": SPLINE, "age": SPLINE}))
    alone, alone_ctx = est.preview("forms:age", d.SetForms(forms={"age": SPLINE}))
    assert mixed_ctx.read["form"]["column"] == "age" and mixed.note is None
    assert [v.model_dump() for v in typed(mixed).views] == [v.model_dump() for v in alone.views]
    by_hand = np.quantile(analyzed_age(est), HARRELL_4, method="linear")
    assert mixed_ctx.read["form"]["knots"] == pytest.approx(list(by_hand), rel=1e-12)


def test_the_guard_is_silent_for_a_column_the_model_holds(est):
    """`age` alone previews as before: the spline's terms, its knots at Harrell's percentiles of the
    analyzed rows, the columns the model will see; no note."""
    result, ctx = est.preview("forms:age", d.SetForms(forms={"age": SPLINE}))
    assert [v.kind for v in typed(result).views] == ["relationship", "distribution", "lineage"]
    assert result.note is None and ctx.read["form"]["column"] == "age"
    assert result.views[0].title == "How `age` enters the model"
    by_hand = np.quantile(analyzed_age(est), HARRELL_4, method="linear")
    assert ctx.read["form"]["knots"] == pytest.approx(list(by_hand), rel=1e-12)
