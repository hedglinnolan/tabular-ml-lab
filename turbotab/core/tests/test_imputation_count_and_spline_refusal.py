"""Two fixes from the slow-fit audit before the freeze.

* **The design step's number of imputations** (``pipeline.imputation_count``, ``impute_detail``):
  the impute step's line states how many imputations the coefficient table draws, White, Royston
  & Wood's (2011) rule over the design's rows ("the number of imputations should be at least equal
  to the percentage of incomplete cases", as White, Pandis & Pham put it, and at least 20 here),
  not the answer's m.
* **A spline on too few values** (``exposure_form.rcs_knots``): ``Hmisc::rcspline.eval`` stops
  below six values; the engine's stop is a fit-time refusal with its code and ways forward, so the
  record classes it as refused, not a crash (``jobs.failure_of``), with the words a person reads
  unchanged. Model 3 under complete cases, fit on the rows its added column is recorded on, holds
  none when that column is never recorded.

Expected values come from the construction: the count of incomplete rows each table is drawn
with, and Hmisc's six values.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import turbotab.core.models  # noqa: F401 - registers the families
from turbotab.core import decisions as d
from turbotab.core.decisions import MissingSpec, ProjectState
from turbotab.core.graph import StageRefusal
from turbotab.core.jobs import failure_of
from turbotab.core.methods import exposure_form as ef_methods
from turbotab.core.models import get_family
from turbotab.core.models.pipeline import describe_steps, design_spec

LINEAR = get_family("linear")


# ── N1: the impute step says the number of imputations the table draws ──────


def _impute_line(n_incomplete: int, m: int = 20, n: int = 100) -> tuple[str, object, pd.DataFrame]:
    """The impute step's line for ``n`` rows of which the first ``n_incomplete`` have `z` blank,
    under multiple imputation with the answer's ``m``; the spec and the rows."""
    rng = np.random.default_rng(2)
    frame = pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n)})
    frame.loc[frame.index[:n_incomplete], "z"] = np.nan
    st = ProjectState(target="y", task="regression", purpose="inference",
                      roles={"x": "exposure", "z": "covariate"},
                      missing=MissingSpec(strategy="multiple_imputation", m=m))
    spec = design_spec(st, frame, ["x", "z"])
    [step] = [s for s in describe_steps(spec, LINEAR, "regression", "inference")
              if s["key"] == "impute"]
    return step["detail"], spec, frame


def test_the_impute_step_states_the_rules_number_where_more_rows_are_incomplete_than_m():
    """38 of the 100 rows have a blank: the rule's m is 38, the answer's 20; the line says 38,
    the number the table's imputation draws."""
    line, spec, frame = _impute_line(38)
    assert line.endswith(" It serves the cross-validated scores; the coefficients are pooled over "
                         "38 multiple imputations with the outcome.")
    # The table's own imputation plan draws as many.
    from turbotab.core.methods.missing import imputation_plan

    y = np.random.default_rng(3).normal(size=len(frame))
    assert imputation_plan(spec, frame, y, "regression").m == 38


def test_the_impute_step_states_the_answers_m_where_it_is_the_larger():
    """10 of 100 rows incomplete: the rule asks for 10, so the answer's 20 is drawn, and a larger
    answer (50) is drawn as asked."""
    line, _, _ = _impute_line(10)
    assert "pooled over 20 multiple imputations with the outcome." in line
    line, _, _ = _impute_line(38, m=50)
    assert "pooled over 50 multiple imputations with the outcome." in line


def test_the_impute_step_says_nothing_is_imputed_where_no_value_is_missing():
    line, _, _ = _impute_line(0)
    assert line.endswith(" It serves the cross-validated scores; no predictor value is missing "
                         "among the analyzed rows, so the coefficients are not imputed.")
    assert "multiple imputations" not in line


# ── N2: a spline on too few values is a typed refusal ────────────────────────


def test_a_spline_on_five_values_is_a_typed_refusal_with_its_ways_forward():
    """Below Hmisc's six values the knots cannot be placed: a fit-time refusal (still a
    ``ValueError`` for every caller that handles one), coded, with its ways forward, and shown as
    the same ``ValueError`` words."""
    with pytest.raises(StageRefusal) as refused:
        ef_methods.rcs_knots(np.arange(5.0), 4)
    exc = refused.value
    assert isinstance(exc, ValueError)
    assert exc.code == "spline_too_few_values"
    assert [e["label"] for e in exc.exits] == [
        "Enter the column as a straight line, which places no knots",
        "Keep more rows: under inference, multiple imputation keeps the rows complete cases drop",
        "Leave the column out"]
    shown, refusal = failure_of(exc)
    assert shown == ("ValueError: A spline needs at least 6 recorded values to place its knots; "
                     "there are 5.")
    assert refusal is not None and refusal["code"] == "spline_too_few_values"
    assert len(refusal["exits"]) == 3


def test_a_spline_on_six_values_places_its_knots():
    """Six values are Hmisc's fewest: the interior four are the knots, and nothing refuses."""
    knots, _ = ef_methods.rcs_knots(np.arange(6.0), 4)
    assert list(knots) == [1.0, 2.0, 3.0, 4.0]


def _model_three_run(tmp_path: Path, recorded: int) -> dict:
    from turbotab.core.tests.acceptance import estimand_fixtures as ef

    frame = ef.cohort(300, seed=5)
    frame.loc[frame.index[recorded:], "bmi"] = np.nan
    st = ef.state(target="glucose", task="regression", measure="mean_difference",
                  exposure_forms={"fiber": d.ExposureFormSpec(form="spline", knots=4)})
    return ef.run(frame, tmp_path, st, fit=False)


def test_model_three_with_no_row_recording_its_added_column_is_refused_not_a_crash(tmp_path):
    """Under complete cases Model 3 is fit on the rows that record `bmi`; with none, its spline of
    `fiber` has no value to place knots on, and the effects stage refuses in words, typed, where it
    raised a bare ``ValueError`` (the slow-fit audit's N2)."""
    with pytest.raises(StageRefusal) as refused:
        _model_three_run(tmp_path, recorded=0)
    shown, refusal = failure_of(refused.value)
    assert shown == ("ValueError: A spline needs at least 6 recorded values to place its knots; "
                     "there are 0.")
    assert refusal is not None and refusal["code"] == "spline_too_few_values"
    assert refusal["exits"]


def test_model_three_on_the_rows_recording_its_added_column_is_fit(tmp_path):
    """With `bmi` recorded on 60 rows, Model 3 is fit on those 60 and nothing refuses."""
    from turbotab.core.tests.acceptance import estimand_fixtures as ef

    run = _model_three_run(tmp_path, recorded=60)
    three = ef.sequence(run["effects"])["model_3"]
    assert three["n_rows"] == 60 and three["effects"]
