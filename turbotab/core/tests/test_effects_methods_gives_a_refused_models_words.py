"""Where Model 2, the primary, reports the study factor and a later model of the sequence is fit
and its table refused, the effects methods text gives that model's refusal in its own words, as
it does where Model 2 itself reports none (``effects._without_the_primary``); "could not be fit
and is not reported" stays for a model nothing was fit for (the verifier's T5bV-3).

Before this repair, with `bmi` = 2 × `fiber` + 3, Model 3 (further adjusted for `bmi`) was refused
for the dependency, and the methods text said only that it "could not be fit and is not reported".

Truth from construction: numpy finds the design of `fiber`, Model 2's covariates and `bmi`
singular and that of `fiber` and Model 2's covariates not; Model 2's estimate of `fiber` is
statsmodels' OLS; the dependency's words are written from the construction (`fiber` and `bmi` in
it). The silent side: an ordinary cohort, every model fit, reads as it did, byte for byte, up to
its sensitivity analyses; and a model held by its own missing-values question still "could not be
fit" (``acceptance/test_wave1b_repairs_integration.py``, run by name).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from turbotab.core.tests.acceptance import estimand_fixtures as ef

N = 400
MODEL_2 = ["age", "sex", "smoking", "activity"]
DEPENDENCY = ("On these 400 rows the model matrix is singular: `fiber` and `bmi` are fixed "
              "combinations of the other columns and the intercept, so their coefficients are not "
              "identified. No estimate, interval or p-value is reported; leave one of the dependent "
              "columns out.")
SEQUENCE = ("The estimate of `fiber` is reported across a declared sequence of models fit on all "
            "400 analyzed rows: unadjusted; Model 2, the primary, adjusted for `age`, `sex`, "
            "`smoking` and `activity`")
APART = (" Only the exposure's estimates are shown as effects; every other coefficient is listed "
         "apart as an adjustment term, not an effect estimate (Westreich & Greenland 2013, Am J "
         "Epidemiol 177:292).")


def _design(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    X = pd.DataFrame({"const": np.ones(len(frame))}, index=frame.index)
    for c in columns:
        X[c] = (frame[c] == "male").astype(float) if c == "sex" else frame[c].astype(float)
    return X.to_numpy(dtype=float)


def _state():
    return ef.state(target="glucose", task="regression", measure="mean_difference")


def test_a_later_models_refusal_is_said_in_its_words_where_model_2_reports(tmp_path):
    """`bmi` = 2 × `fiber` + 3: Model 2 reports `fiber` (statsmodels, to 1e-9), Model 3 is refused
    for the dependency of `fiber` and `bmi`, and the methods text gives that refusal after Model
    3's name, where it said Model 3 "could not be fit"."""
    frame = ef.cohort(n=N, seed=7)
    frame["bmi"] = 2 * frame["fiber"] + 3
    three, two = _design(frame, ["fiber", *MODEL_2, "bmi"]), _design(frame, ["fiber", *MODEL_2])
    assert np.linalg.matrix_rank(three) == three.shape[1] - 1  # the premise
    assert np.linalg.matrix_rank(two) == two.shape[1]
    out = ef.run(frame, tmp_path, _state())
    seq = ef.sequence(out["effects"])
    [row] = [r for r in seq["model_2"]["effects"] if r["feature"] == "fiber"]
    truth = sm.OLS(frame["glucose"].to_numpy(float), two).fit()
    assert row["estimate"] == pytest.approx(float(truth.params[1]), rel=1e-9)
    assert seq["model_3"]["effects"] is None
    assert seq["model_3"]["inference"]["refused"] == DEPENDENCY
    methods = out["effects"]["methods"]
    assert methods.startswith(f"{SEQUENCE}. Model 3, further adjusted for `bmi`, reports none: "
                              f"{DEPENDENCY}{APART}")
    assert "could not be fit" not in methods


def test_an_ordinary_cohorts_methods_text_reads_as_before(tmp_path):
    """Silent: the cohort as drawn, every model fit, reads as it did, byte for byte, up to its
    sensitivity analyses (the sequence, Model 3's words, the display rule)."""
    out = ef.run(ef.cohort(n=N, seed=7), tmp_path, _state())
    assert out["effects"]["methods"].startswith(
        f"{SEQUENCE}; Model 3, further adjusted for `bmi`, a possible mediator, so not a total "
        f"effect.{APART}")
