"""The NHANES glucose journey's refusals and defaults (SWARM_PLAN §10 package 6a; R1_PLAN.md R1-B),
each held to what the code does, both ways, from known truth.

* **Whole numbers wait for their reading.** A predictor of five whole numbers is asked whether they
  are codes or amounts before the fit, since the two readings fit different models to the same rows
  (numpy's least squares, here); a predictor of two values is not asked, since its code's one
  indicator and its slope fit the same values; once read as an amount it is not asked again.
* **A blank yes/no answer reads as not asked** on at least half the rows with the outcome
  measured, and not on one row fewer; the share is of the rows with the outcome measured, not of
  every row, both ways; a measurement's blanks never do, however many.
* **Under Predict a fill learned in each fold ranks above complete cases**: scikit-learn's least
  squares refuses a new row with a blank, and the same model behind a fill learned on the training
  rows scores it. Under Estimate multiple imputation leads.
* **Under Predict keeping every row ranks first** among the intake screens; under Estimate a
  screen leads and every row is reported beside it.
* **An intake screen waits for the energy column's days**: while they are unsettled it is refused
  in the claim's words; once recorded it is offered.
* **A nearly collinear pair** is named at |r| just above nine tenths, of either sign, and not just
  under it (numpy's own correlation of the built pair).

Every expectation comes from construction or an independent library. This file names no claim id
and none of the constants the claims set (meta-test 2).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

from turbotab.core import custom_sound, decisions, readings
from turbotab.core.stages import explore, proposals
from turbotab.core.tests.test_stage_registry import record

# ── whole numbers wait for their reading ─────────────────────────────────────


def _state(roles: dict[str, str], *extra: dict) -> decisions.ProjectState:
    return decisions.fold([
        record(1, {"kind": "set_lens", "lenses": ["clinical"]}),
        record(2, {"kind": "set_target", "column": "y"}),
        record(3, {"kind": "set_purpose", "purpose": "prediction"}),
        record(4, {"kind": "set_roles", "roles": roles}),
        *(record(5 + i, d) for i, d in enumerate(extra))])


def _fitted(design: np.ndarray, y: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return design @ beta


def _indicators(x: np.ndarray) -> np.ndarray:
    levels = np.unique(x)
    return np.column_stack([np.ones(len(x)), *[(x == v).astype(float) for v in levels[1:]]])


def test_a_whole_number_predictor_of_five_values_waits_for_its_reading_since_the_two_codings_fit_apart():
    """Five whole numbers read as codes (four indicators beside an intercept) and as an amount (one
    slope) fit different values to the same rows, so the values cannot settle the reading, and the
    fit refuses until it is answered."""
    grade = np.tile(np.arange(1, 6), 8).astype(float)
    y = np.array([0.0, 3.0, 1.0, 4.0, 2.0])[grade.astype(int) - 1] + 0.01 * np.arange(len(grade))
    as_codes = _fitted(_indicators(grade), y)
    as_amount = _fitted(np.column_stack([np.ones(len(grade)), grade]), y)
    assert np.max(np.abs(as_codes - as_amount)) > 0.5  # the reading changes the model

    state = _state({"grade": "covariate"})
    with pytest.raises(readings.Unsettled) as asked:
        readings.predictors_or_ask(state, {"grade": {"dtype": "integer", "n_unique": 5}})
    assert [(r.kind, r.column) for r in asked.value.readings] == [("code_or_count", "grade")]
    message = str(asked.value)
    assert message.startswith("`grade` holds whole numbers, which may be codes for categories "
                              "(one indicator per level) or amounts (one slope); the fit waits "
                              "for the answer for each."), message
    assert {e["decision"]["value"] for e in asked.value.exits if e.get("decision")} >= {
        "code", "amount"}


def test_a_two_valued_predictor_is_not_asked_since_either_coding_fits_alike():
    """Two values: the code's one indicator is the amount shifted and scaled, so beside an
    intercept both fit the same values, and the fit does not ask."""
    flag = np.tile([3.0, 7.0], 20)
    y = np.sin(np.arange(len(flag))) + (flag == 7.0)
    as_codes = _fitted(_indicators(flag), y)
    as_amount = _fitted(np.column_stack([np.ones(len(flag)), flag]), y)
    assert np.max(np.abs(as_codes - as_amount)) < 1e-9

    state = _state({"flag": "covariate"})
    assert readings.predictors_or_ask(state, {"flag": {"dtype": "integer", "n_unique": 2}}) == [
        "flag"]


def test_a_whole_number_predictor_read_as_an_amount_is_not_asked_again():
    state = _state({"grade": "covariate"}, {"kind": "confirm_reading", "reading": "code_or_count",
                                            "column": "grade", "value": "amount"})
    assert readings.predictors_or_ask(state, {"grade": {"dtype": "integer", "n_unique": 5}}) == [
        "grade"]


# ── a blank yes/no answer reads as not asked ─────────────────────────────────


def _blank_reading(name: str, values: list, dtype: str, outcome: list | None = None) -> dict:
    """``name``'s reading beside an outcome ``y``, measured on every row unless ``outcome`` (one
    value per row, None where it was not measured) says otherwise."""
    y = np.arange(len(values), dtype=float) if outcome is None else outcome
    frame = pd.DataFrame({"y": pd.array(y, dtype="Float64").astype(float), name: values})
    s = frame[name]
    columns = [{"name": "y", "dtype": "float", "n_unique": int(frame["y"].nunique()),
                "n_missing": int(frame["y"].isna().sum())},
               {"name": name, "dtype": dtype, "n_unique": int(s.nunique()),
                "n_missing": int(s.isna().sum())}]
    reading = proposals.missing_reading(frame, columns, {name: "covariate"}, "y")
    return next(e for e in reading["columns"] if e["column"] == name)


def test_a_yes_no_answer_blank_on_half_the_rows_reads_as_not_asked():
    """Five of ten rows blank is half of them."""
    entry = _blank_reading("answered", [True, False, True, False, True] + [None] * 5, "boolean")
    assert entry["n_missing"] == 5 and entry["share"] == 0.5
    assert entry["likely_not_asked"] is True
    assert entry["reason"] == "A yes/no answer blank on `50%` of rows: blank usually means not asked."


def test_a_yes_no_answer_blank_on_one_row_fewer_than_half_does_not():
    entry = _blank_reading("answered", [True, False, True, False, True, True] + [None] * 4,
                           "boolean")
    assert entry["n_missing"] == 4
    assert entry["likely_not_asked"] is False
    assert "not asked" not in entry["reason"]


# The outcome measured on six of ten rows: the share is counted among those six.
MEASURED_ON_SIX = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, None, None, None, None]


def test_a_yes_no_answer_blank_on_half_the_rows_with_the_outcome_measured_reads_as_not_asked():
    """Blank on three of the six rows with the outcome measured, which is half of them, and on
    none of the four without it: three of ten rows in all, under half, yet read as not asked,
    since the share is of the rows with the outcome measured."""
    values = [True, None, False, None, True, None, True, False, True, False]
    entry = _blank_reading("answered", values, "boolean", outcome=MEASURED_ON_SIX)
    assert sum(v is None for v in values) / len(values) == 0.3  # of every row: under half
    assert entry["n_missing"] == 3 and entry["share"] == 0.5  # of the six measured: half
    assert entry["likely_not_asked"] is True


def test_a_yes_no_answer_blank_on_half_of_every_row_but_few_with_the_outcome_measured_does_not():
    """Blank on one of the six rows with the outcome measured and on all four without it: half of
    every row, but a sixth of the rows the share counts, so not read as not asked."""
    values = [True, None, False, True, False, True, None, None, None, None]
    entry = _blank_reading("answered", values, "boolean", outcome=MEASURED_ON_SIX)
    assert sum(v is None for v in values) / len(values) == 0.5  # of every row: half
    assert entry["n_missing"] == 1 and entry["share"] == round(1 / 6, 4)
    assert entry["likely_not_asked"] is False
    assert "not asked" not in entry["reason"]


def test_a_measurement_blank_on_most_rows_is_never_read_as_not_asked():
    """A waist measured on three of ten rows: blank on 70%, and nothing says a question was not
    asked."""
    entry = _blank_reading("waist", [81.5, 97.25, 88.0] + [None] * 7, "float")
    assert entry["share"] == 0.7
    assert entry["likely_not_asked"] is False
    assert entry["reason"] == "Blank on `70%` of rows; nothing says the blanks mean not asked."


# ── under Predict a fill learned in each fold ranks above complete cases ──────

FOLD_FILLS = ("impute", "missing_category", "indicators")


def test_under_prediction_a_fill_learned_in_each_fold_ranks_above_complete_cases_and_scores_a_new_row_with_a_blank():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 2))
    y = X @ np.array([1.0, -2.0]) + rng.normal(scale=0.1, size=60)
    X[rng.choice(60, 12, replace=False), 1] = np.nan
    complete = ~np.isnan(X).any(axis=1)
    new_row = np.array([[0.5, np.nan]])
    with pytest.raises(ValueError):  # a complete-case model has no rule for a blank
        LinearRegression().fit(X[complete], y[complete]).predict(new_row)
    filled = make_pipeline(SimpleImputer(strategy="median"), LinearRegression()).fit(X, y)
    assert np.isfinite(filled.predict(new_row)).all()

    q = custom_sound.missing("prediction")
    keys = [o.key for o in q.options]
    assert keys[0] in FOLD_FILLS and keys.index("complete_case") > 0
    assert keys[-1] == "multiple_imputation"  # it reads the outcome, which a new row lacks
    assert q.tension == ("Complete cases are the field's habit, but a new row with a blank cannot "
                         "be scored; a fill learned in each training fold without the outcome can "
                         "(Sisk et al. 2023, Stat Methods Med Res 32:1461).")


def test_under_estimation_multiple_imputation_leads_and_the_tension_names_its_own_reason():
    q = custom_sound.missing("inference")
    assert q.options[0].key == "multiple_imputation"
    assert q.tension is not None and "Sterne" in q.tension
    assert "cannot be scored" not in q.tension


# ── under Predict keeping every row ranks first ──────────────────────────────

SCREENS = ["goldberg_schofield", "willett_2013_by_sex", "nhs_hpfs_by_sex", "sex_neutral_500_3500"]


def test_under_prediction_keeping_every_row_ranks_first_among_the_screens():
    q = custom_sound.exclusions("prediction", SCREENS)
    assert [o.key for o in q.options][0] == "keep_every_row"
    assert {o.key for o in q.options} == {"keep_every_row", *SCREENS}
    assert q.tension == ("Screens are the field's habit; under prediction the people they remove "
                         "will still be scored, so keeping every row comes first.")


def test_under_estimation_a_screen_ranks_first_and_every_row_is_reported_beside_it():
    q = custom_sound.exclusions("inference", SCREENS)
    keys = [o.key for o in q.options]
    assert keys[0] in SCREENS and keys[-1] == "keep_every_row"
    assert q.tension is not None and "reported beside" in q.tension
    assert "comes first" not in q.tension


# ── an intake screen waits for the energy column's days ──────────────────────


def test_a_screen_waits_while_the_energy_columns_days_are_unsettled():
    """A two-day total of 6,000 kcal is 3,000 a day: outside 500 to 5,000 read as one day's
    intake, inside them read as two days', so the screen waits for the days."""
    one_day, two_days = 6000.0, 6000.0 / 2
    assert not 500 <= one_day <= 5000 and 500 <= two_days <= 5000
    reading = {"unit": "kcal", "basis": "decision", "days": 1, "confirmed": False,
               "unit_settled": True, "days_unsettled": True,
               "days_reading": {"evidence": "its median, 1,950, is an adult's day's intake"}}
    assert proposals.unit_refusal("kcal", reading) == (
        "Refused until `kcal`'s days are recorded: its median, 1,950, is an adult's day's intake, "
        "so whether each value is one day's intake or a total over several is not settled, and the "
        "screen's bounds are read in a day's.")


def test_a_screen_is_offered_once_the_days_are_recorded():
    frame = pd.DataFrame({"kcal": [1950.0, 2400.0, 1200.0, 3100.0]})
    for days in (1, 2):
        reading = proposals.energy_unit_reading(frame, "kcal", {"unit": "kcal", "days": days})
        assert reading["confirmed"] is True
        assert proposals.unit_refusal("kcal", reading) is None


# ── a nearly collinear pair ──────────────────────────────────────────────────


def _pair(r: float, n: int = 200, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Two columns whose sample correlation is ``r``: x and z centered, orthogonal and of unit
    norm, and y = r·x + √(1 − r²)·z."""
    rng = np.random.default_rng(seed)
    x, z = rng.normal(size=(2, n))
    x -= x.mean()
    x /= np.linalg.norm(x)
    z -= z.mean()
    z -= (z @ x) * x
    z /= np.linalg.norm(z)
    return x, r * x + np.sqrt(1 - r * r) * z


def test_a_pair_correlated_just_above_nine_tenths_either_way_is_named():
    for r in (0.9005, -0.9005):
        x, y = _pair(r)
        measured = float(np.corrcoef(x, y)[0, 1])
        assert abs(measured) >= 0.9 and measured == pytest.approx(r, abs=1e-12)
        pairs = explore._collinear(pd.DataFrame({"a": x, "b": y}), ["a", "b"])
        assert [(a, b) for a, b, _ in pairs] == [("a", "b")]
        assert pairs[0][2] == pytest.approx(measured, abs=1e-12)


def test_a_pair_correlated_just_under_nine_tenths_is_not():
    x, y = _pair(0.8995)
    assert abs(float(np.corrcoef(x, y)[0, 1])) < 0.9
    assert explore._collinear(pd.DataFrame({"a": x, "b": y}), ["a", "b"]) == []
