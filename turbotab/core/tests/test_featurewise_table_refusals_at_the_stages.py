"""Feature-wise regression's refused tables as the fit and effects stages serve them (the
orchestrator's rulings (a), (b) and (e) on the swarm's tail: T2-C-V1, T2-C-open-0, T2-C-open-3).

* (e) A refused table has no estimate, so its caption is ``No tests:`` and the reason alone: the
  fit stage added "Estimated from all 400 analyzed rows" beside "No tests" (the formed branch
  recorded no rows, so the stage told it them). The table records the rows it was handed, 400
  by construction. An ordinary table beside it keeps its line.
* (a) The effects stage's feature-wise tables (the unadjusted model, Model 2, Model 3) refuse a
  study factor in a declared spline in the fit's words: they tested each column of the spline
  (`fiber`, `fiber'`, `fiber''`) as a study factor of its own, three numbers no one asked for. A
  straight-line study factor's tables keep their numbers: the unadjusted one is statsmodels'
  least squares of the outcome on `fiber` alone, with its classical standard error.
* (b) A study factor the model matrix holds only as indicators (a category) refuses the
  feature-wise table, not the fit: the linear model chosen beside it is fit, its coefficients
  statsmodels' least squares on the design written out here, and feature-wise regression's
  tables at the fit and the effects stage are refused in the words the fit refused in.

The cohort is ``estimand_fixtures.cohort`` (400 rows); the run is design, fit and effects on
every row, as a worker runs them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

import turbotab.core.models  # noqa: F401 - registers the families
from turbotab.core import decisions as d
from turbotab.core.tests.acceptance import estimand_fixtures as ef

N = 400
LEAD = ("Feature-wise regression tests each study factor as one straight-line column, one at a "
        "time, adjusted for the rest")
FORMED = (f"{LEAD}; `fiber` has that role, and its declared form, a restricted cubic spline, "
          f"enters it as its spline terms, not one straight-line column.")
NO_FACTOR = ("No study factor to test: feature-wise regression tests the columns with the role of "
             "what you study, one at a time, adjusted for the rest; `grp` has that role, but the "
             "model matrix holds no column of that name.")
ANOTHER = {"label": "Choose another model family", "decision": None}


def _model(out: dict, family: str) -> dict:
    return next(m for m in out["fit_raw"]["models"] if m["family"] == family)


def _sequence(out: dict, family: str) -> dict:
    index = next(i for i, f in enumerate(out["effects"]["families"]) if f["family"] == family)
    return ef.sequence(out["effects"], index)


def _effects_refused(out: dict, words: str) -> None:
    """Feature-wise regression's table refused in every model of the effects stage: no number,
    the words its reason and its whole caption."""
    sequence = _sequence(out, "featurewise")
    assert set(sequence) >= {"crude", "model_2"}
    for key, model in sequence.items():
        assert not model.get("effects"), key
        assert model["inference"]["refused"] == words, key
        assert model["inference"]["caption"] == f"No tests: {words}", key


def _fit_refused(out: dict, words: str) -> None:
    """... and at the fit, which records the rows it was handed."""
    fit = _model(out, "featurewise")
    assert fit["coefficients"] in (None, [])
    assert fit["inference"]["refused"] == words
    assert fit["inference"]["caption"] == f"No tests: {words}"
    assert fit["inference"]["n_rows"] == N


@pytest.fixture(scope="module")
def spline(tmp_path_factory):
    st = ef.state(target="glucose", task="regression", measure="mean_difference",
                  models=["linear", "featurewise"],
                  exposure_forms={"fiber": d.ExposureFormSpec(form="spline", knots=4)})
    return ef.run(ef.cohort(N, seed=7), tmp_path_factory.mktemp("spline"), st)


def test_a_refused_feature_wise_tables_caption_states_no_estimate(spline):
    _fit_refused(spline, FORMED)
    # silent: the linear table beside it was estimated, and says from which rows
    linear = _model(spline, "linear")
    assert linear["coefficients"]
    assert linear["inference"]["caption"].endswith(f" Estimated from all {N} analyzed rows.")


def test_the_effects_stages_feature_wise_tables_refuse_a_study_factor_in_a_spline(spline):
    _effects_refused(spline, FORMED)
    assert _sequence(spline, "linear")["model_2"]["effects"]  # the family beside it is fit


def test_a_straight_line_study_factors_effects_tables_keep_their_numbers(tmp_path):
    frame = ef.cohort(N, seed=7)
    st = ef.state(target="glucose", task="regression", measure="mean_difference",
                  models=["featurewise"])
    out = ef.run(frame, tmp_path, st)
    (row,) = _sequence(out, "featurewise")["crude"]["effects"]
    fit = sm.OLS(frame["glucose"].to_numpy(float),
                 sm.add_constant(frame[["fiber"]].to_numpy(float))).fit()
    assert row["feature"] == "fiber"
    assert row["estimate"] == pytest.approx(float(fit.params[1]), rel=1e-9)
    assert row["se"] == pytest.approx(float(fit.bse[1]), rel=1e-9)
    assert _model(out, "featurewise")["inference"]["caption"].endswith(
        f" Estimated from all {N} analyzed rows.")


def test_a_category_as_the_study_factor_refuses_the_feature_wise_table_not_the_fit(tmp_path):
    frame = ef.cohort(N, seed=7)
    rng = np.random.default_rng(11)
    frame["grp"] = rng.choice(["a", "b", "c"], N)
    frame["glucose"] = frame["glucose"] + np.where(frame["grp"] == "c", 3.0, 0.0)
    roles = {"pid": "identifier", "age": "covariate", "sex": "covariate", "smoking": "covariate",
             "activity": "covariate", "bmi": "covariate", "grp": "exposure"}
    st = ef.state(target="glucose", task="regression", measure="mean_difference", exposure="grp",
                  roles=roles, models=["linear", "featurewise"])
    out = ef.run(frame, tmp_path, st)
    _fit_refused(out, NO_FACTOR)
    _effects_refused(out, NO_FACTOR)
    assert _model(out, "featurewise")["inference"]["exits"] == [ANOTHER]
    # The linear model chosen beside it is fit: least squares on the design written out (the
    # first level of each category its reference, as the one-hot step drops it).
    design = pd.DataFrame({"(intercept)": 1.0, "sex_male": (frame["sex"] == "male").astype(float),
                           "age": frame["age"], "smoking": frame["smoking"],
                           "activity": frame["activity"],
                           "grp_b": (frame["grp"] == "b").astype(float),
                           "grp_c": (frame["grp"] == "c").astype(float)})
    expected = sm.OLS(frame["glucose"].to_numpy(float), design).fit().params
    got = {c["feature"]: c["estimate"] for c in _model(out, "linear")["coefficients"]}
    assert set(got) == set(design.columns)
    for column in design.columns:
        assert got[column] == pytest.approx(float(expected[column]), rel=1e-8), column
