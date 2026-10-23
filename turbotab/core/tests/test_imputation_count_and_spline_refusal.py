"""Two fixes from the slow-fit audit before the freeze.

* **The design step's number of imputations** (``pipeline.imputation_count``, ``impute_detail``):
  the impute step's line states how many imputations the coefficient table draws, White, Royston
  & Wood's (2011) rule over the design's rows ("the number of imputations should be at least equal
  to the percentage of incomplete cases", as White, Pandis & Pham put it, and at least 20 here),
  not the answer's m.
* **A spline on too few values** (``exposure_form.rcs_knots``): ``Hmisc::rcspline.eval`` stops
  below six values; the engine's stop is a fit-time refusal with its code and ways forward, so the
  record classes it as refused, not a crash (``jobs.failure_of``), with the words a person reads
  unchanged; its last way forward, more rows that record the column, holds wherever it fires, the
  imputation's knots too. Model 3 under any answer but multiple imputation (complete cases, a
  single fill), fit on the rows its added column is recorded on, holds none when that column is
  never recorded.

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


def more_rows(named: str = "the column") -> str:
    return f"Keep more rows that record {named}: the knots are placed on its recorded values"


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
        "Enter the column as a straight line, which places no knots", more_rows()]
    shown, refusal = failure_of(exc)
    assert shown == ("ValueError: A spline needs at least 6 recorded values to place its knots; "
                     "there are 5.")
    assert refusal is not None and refusal["code"] == "spline_too_few_values"
    assert len(refusal["exits"]) == 2


def test_a_spline_of_a_named_column_says_its_name_and_offers_no_straight_line_on_no_row():
    """Where the column is known (the form step's fit) the refusal names it and its straight
    line; with no value at all a straight line has no row to be fit on either, so it is not
    offered."""
    with pytest.raises(StageRefusal) as refused:
        ef_methods.rcs_knots(np.arange(5.0), 4, column="fiber")
    assert str(refused.value) == ("A spline of `fiber` needs at least 6 recorded values to place "
                                  "its knots; there are 5.")
    assert [e["label"] for e in refused.value.exits] == [
        "Enter `fiber` as a straight line, which places no knots", more_rows("`fiber`")]
    with pytest.raises(StageRefusal) as refused:
        ef_methods.rcs_knots(np.array([np.nan, np.nan]), 4, column="fiber")
    assert str(refused.value).endswith("there are 0.")
    assert [e["label"] for e in refused.value.exits] == [more_rows("`fiber`")]


def test_a_spline_on_six_values_places_its_knots():
    """Six values are Hmisc's fewest: the interior four are the knots, and nothing refuses."""
    knots, _ = ef_methods.rcs_knots(np.arange(6.0), 4)
    assert list(knots) == [1.0, 2.0, 3.0, 4.0]


def _model_three_run(tmp_path: Path, recorded: int, *, form: str = "spline",
                     waist: int | None = None, missing: object = None) -> dict:
    """``ef.cohort(300)`` with `bmi` (unknown timing: Model 3's added column) recorded on its first
    ``recorded`` rows, `fiber` entered as ``form``, under complete cases (or the ``missing``
    answer given); with ``waist``, a second added column recorded on the last ``waist`` rows."""
    from turbotab.core.tests.acceptance import estimand_fixtures as ef

    frame = ef.cohort(300, seed=5)
    frame.loc[frame.index[recorded:], "bmi"] = np.nan
    roles, answers = dict(ef.ROLES), dict(ef.ANSWERS)
    if waist is not None:
        frame["waist"] = 90 + 0.5 * (frame["fiber"] - 20)
        frame.loc[frame.index[:len(frame) - waist], "waist"] = np.nan
        roles["waist"], answers["waist"] = "covariate", ef.UNKNOWN_TIMING
    forms = {"fiber": d.ExposureFormSpec(form="spline", knots=4)} if form == "spline" else None
    st = ef.state(target="glucose", task="regression", measure="mean_difference", roles=roles,
                  answers=answers, **({"exposure_forms": forms} if forms else {}),
                  **({"missing": missing} if missing is not None else {}))
    return ef.run(frame, tmp_path, st, fit=False)


@pytest.mark.parametrize("form", ["spline", "linear"])
def test_model_three_with_no_row_recording_its_added_column_is_refused_not_a_crash(tmp_path, form):
    """Under complete cases Model 3 is fit on the rows that record `bmi`; with none it has no row
    to be fit on, whatever form `fiber` takes, and the effects stage refuses before any fit, in
    words, typed, where it raised a bare ``ValueError`` (the slow-fit audit's N2: from the spline's
    knots, and, with `fiber` a straight line, from the fit on no row). The way forward leaves
    `bmi` out; multiple imputation is not offered, as it has no recorded `bmi` to start from."""
    with pytest.raises(StageRefusal) as refused:
        _model_three_run(tmp_path, recorded=0, form=form)
    shown, refusal = failure_of(refused.value)
    assert shown == ("ValueError: Model 3 adds `bmi` and is fit on the rows that record it: none "
                     "of the 300 analyzed rows does, so Model 3 cannot be fit.")
    assert refusal is not None and refusal["code"] == "model_3_no_complete_rows"
    assert [e["label"] for e in refusal["exits"]] == ["Leave `bmi` out of Model 3"]


def test_model_three_whose_added_columns_no_row_records_together_offers_multiple_imputation(
        tmp_path):
    """`bmi` recorded on the first 150 rows, `waist` on the last 150: each has values to impute
    from, no row records both, so Model 3's complete cases hold no row; the ways forward leave
    them out or impute them."""
    from turbotab.core.decisions import parse_decision

    with pytest.raises(StageRefusal) as refused:
        _model_three_run(tmp_path, recorded=150, waist=150)
    assert str(refused.value) == ("Model 3 adds `bmi` and `waist` and is fit on the rows that "
                                  "record them all: none of the 300 analyzed rows does, so Model 3 "
                                  "cannot be fit.")
    leave, impute = refused.value.exits
    assert leave == {"label": "Leave `bmi` and `waist` out of Model 3", "decision": None}
    assert impute["label"] == ("Multiple imputation (m at least 20, more when more rows are "
                               "incomplete)")
    assert parse_decision(impute["decision"]).strategy == "multiple_imputation"


def test_model_three_on_five_rows_names_the_spline_column_it_cannot_place_knots_for(tmp_path):
    """With `bmi` on five rows Model 3 has rows, but its spline of `fiber` has five values: the
    spline's refusal names `fiber` and its straight line."""
    with pytest.raises(StageRefusal) as refused:
        _model_three_run(tmp_path, recorded=5)
    assert refused.value.code == "spline_too_few_values"
    assert str(refused.value) == ("A spline of `fiber` needs at least 6 recorded values to place "
                                  "its knots; there are 5.")
    assert [e["label"] for e in refused.value.exits] == [
        "Enter `fiber` as a straight line, which places no knots", more_rows("`fiber`")]


def test_model_three_under_a_single_fill_with_no_row_recording_its_added_column_names_no_answer(
        tmp_path):
    """A single fill kept as a recorded limitation under inference fits Model 3, as complete cases
    do, on the rows that record `bmi`, so with none it is refused in the same words, which name
    no missing-values answer (they said "under complete cases", which misnamed this one)."""
    with pytest.raises(StageRefusal) as refused:
        _model_three_run(tmp_path, recorded=0,
                         missing={"strategy": "impute", "acknowledged": True})
    assert refused.value.code == "model_3_no_complete_rows"
    assert str(refused.value) == ("Model 3 adds `bmi` and is fit on the rows that record it: none "
                                  "of the 300 analyzed rows does, so Model 3 cannot be fit.")
    assert "complete cases" not in str(refused.value)
    assert [e["label"] for e in refused.value.exits] == ["Leave `bmi` out of Model 3"]


def test_a_spline_too_few_values_under_multiple_imputation_offers_no_multiple_imputation(
        tmp_path):
    """Under multiple imputation the knots are placed once on `fiber`'s recorded values, here
    five: the refusal's last way forward asks for more rows that record it, true there too, and
    no longer offers multiple imputation, the answer in use."""
    from turbotab.core.tests.acceptance import estimand_fixtures as ef

    frame = ef.cohort(300, seed=5)
    frame.loc[frame.index[5:], "fiber"] = np.nan
    st = ef.state(target="glucose", task="regression", measure="mean_difference",
                  missing={"strategy": "multiple_imputation", "m": 20},
                  exposure_forms={"fiber": d.ExposureFormSpec(form="spline", knots=4)},
                  reading_confirmations={"code_or_count:fiber": "amount",
                                         "code_or_count:age": "amount"})
    with pytest.raises(StageRefusal) as refused:
        ef.run(frame, tmp_path, st, fit=True)
    assert refused.value.code == "spline_too_few_values"
    assert str(refused.value).endswith("needs at least 6 recorded values to place its knots; "
                                       "there are 5.")
    labels = [e["label"] for e in refused.value.exits]
    assert labels[-1].startswith("Keep more rows that record ")
    assert labels[-1].endswith(": the knots are placed on its recorded values")
    assert not [label for label in labels if "multiple imputation" in label]


def test_model_three_on_the_rows_recording_its_added_column_is_fit(tmp_path):
    """With `bmi` recorded on 60 rows, Model 3 is fit on those 60 and nothing refuses."""
    from turbotab.core.tests.acceptance import estimand_fixtures as ef

    run = _model_three_run(tmp_path, recorded=60)
    three = ef.sequence(run["effects"])["model_3"]
    assert three["n_rows"] == 60 and three["effects"]


# ── the impute step where the count is the fit's to make ─────────────────────


def _clustered_line(levels: str | None) -> tuple[str, object, pd.DataFrame, object]:
    """100 rows, two to each of 50 units (`pid`), `z` confirmed as one value per unit and blank on
    both rows of the first 25 units (50 rows) and on the second row of the next 20 (20 rows): 70
    rows incomplete, 50 once each unit's recorded value is carried to its blank row."""
    from turbotab.core.models.inference import Clusters

    n = 100
    rng = np.random.default_rng(4)
    pid = np.repeat(np.arange(50), 2)
    z = np.repeat(rng.normal(size=50), 2)
    blank = (pid < 25) | ((pid >= 25) & (pid < 45) & (np.arange(n) % 2 == 1))
    frame = pd.DataFrame({"pid": pid, "x": rng.normal(size=n), "z": np.where(blank, np.nan, z)})
    missing = MissingSpec(strategy="multiple_imputation", m=20,
                          **({"imputation_levels": levels} if levels else {}))
    st = ProjectState(target="y", task="regression", purpose="inference",
                      roles={"pid": "identifier", "x": "exposure", "z": "covariate"},
                      missing=missing, reading_confirmations={"time_invariant:z": "yes"})
    spec = design_spec(st, frame[["x", "z"]], ["x", "z"])
    [step] = [s for s in describe_steps(spec, LINEAR, "regression", "inference")
              if s["key"] == "impute"]
    clusters = Clusters(column="pid", codes=pid.astype(np.int64), n_clusters=50)
    return step["detail"], spec, frame, clusters


def test_the_impute_step_states_a_floor_where_a_unit_carries_its_value_to_blank_rows():
    """Under clustered imputation the confirmed column is carried within each unit before the
    count is made, which the design does not do: the line states the floor, 20, and the table's
    own imputation draws at least that many (here 50, the rows still blank after the carry)."""
    from turbotab.core.methods.missing import imputation_plan

    line, spec, frame, clusters = _clustered_line(None)
    assert spec.imputations is None
    assert line.endswith(" It serves the cross-validated scores; the coefficients are pooled over "
                         "at least 20 multiple imputations with the outcome, more when more rows "
                         "are incomplete.")
    y = np.random.default_rng(3).normal(size=len(frame))
    plan = imputation_plan(spec, frame[["x", "z"]], y, "regression", clusters=clusters,
                           time_invariant=["z"])
    assert plan.m == 50 >= 20


def test_the_impute_step_states_the_count_where_the_answer_imputes_at_one_level():
    """Imputing at one level carries nothing: the 70 incomplete rows set the count, which the line
    states and the table's own imputation draws."""
    from turbotab.core.methods.missing import imputation_plan

    line, spec, frame, clusters = _clustered_line("single_level")
    assert spec.imputations == 70
    assert line.endswith(" It serves the cross-validated scores; the coefficients are pooled over "
                         "70 multiple imputations with the outcome.")
    y = np.random.default_rng(3).normal(size=len(frame))
    plan = imputation_plan(spec, frame[["x", "z"]], y, "regression", clusters=clusters,
                           time_invariant=["z"])
    assert plan.m == 70
