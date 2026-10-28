"""A specification-curve departure whose design is dependent is estimated where what is studied
is identified, at least squares on the columns that identify it, and is not estimated, in the
dependency's words, where what is studied is in the dependency; the curve never raises.

On the base the departure raised an untyped ``StopIteration`` (``linear_fitter`` looked for `x`'s
row in a table refused with no rows), which neither the curve nor the mattered stage catches. The
sweep's own option ``exclusions=levels@built:RIAGENDR`` is such a departure: keeping one sex
leaves the coded sex constant, a dependency with the intercept.

Truth from construction and statsmodels: `x`'s coefficient on the men alone is statsmodels' OLS of
y on (1, x, age) on those rows, the constant `RIAGENDR` left out, and with `age_months` = 12·age
added it is OLS on (1, x, age); `x_twice` = 2·x puts `x` itself in a dependency.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from turbotab.core.methods import spec_curve as S

N = 300


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"y": rng.normal(size=N), "x": rng.normal(size=N),
                          "age": rng.normal(50, 10, size=N), "RIAGENDR": rng.choice([1, 2], size=N)})
    return frame.assign(age_months=12 * frame["age"], x_twice=2 * frame["x"])


def _curve(choice: S.Choice) -> S.SpecCurve:
    fit = S.linear_fitter(_frame(), outcome="y", exposure="x", covariates=["age", "RIAGENDR"],
                          choices=[choice], task="regression")
    return S.specification_curve([choice], fit, lock="L1", estimate_label="x")


def test_a_departure_that_leaves_a_covariate_constant_keeps_x():
    """Men only: `RIAGENDR` is constant, so it and the intercept are a dependency, `x` is not, and
    the departure's estimate is statsmodels' OLS on (1, x, age) on the men."""
    frame = _frame()
    men = (frame["RIAGENDR"] == 1).to_numpy()
    choice = S.Choice("excl", "exclusion", (S.Alternative("all", "everyone"),
                                            S.Alternative("men", "men only", keep=men)), "all")
    curve = _curve(choice)
    assert curve.not_estimated == []
    by_key = {s.key: s for s in curve.specs}
    rows = frame[men]
    ols = sm.OLS(rows["y"], sm.add_constant(rows[["x", "age"]])).fit()
    assert by_key["excl=men"].estimate == pytest.approx(float(ols.params["x"]), rel=1e-9)
    assert by_key["excl=men"].low is not None
    full = sm.OLS(frame["y"], sm.add_constant(frame[["x", "age", "RIAGENDR"]])).fit()
    assert by_key["excl=all"].estimate == pytest.approx(float(full.params["x"]), rel=1e-9)


def test_a_departure_that_adds_a_copy_of_a_covariate_keeps_x():
    """Age in months beside age in years: a dependency `x` is in none of, so the departure's
    estimate is OLS without the copy."""
    frame = _frame()
    choice = S.Choice("adj", "adjustment", (S.Alternative("base", "base"),
                                            S.Alternative("months", "add age in months",
                                                          add=("age_months",))), "base")
    curve = _curve(choice)
    assert curve.not_estimated == []
    ols = sm.OLS(frame["y"], sm.add_constant(frame[["x", "age", "RIAGENDR"]])).fit()
    estimate = {s.key: s.estimate for s in curve.specs}["adj=months"]
    assert estimate == pytest.approx(float(ols.params["x"]), rel=1e-9)


def test_a_departure_that_puts_x_in_a_dependency_is_not_estimated_in_its_words():
    """A copy of `x` added: `x` is in the dependency, so that specification is not estimated, its
    reason the dependency's words naming `x` and `x_twice`; the primary is estimated."""
    choice = S.Choice("adj", "adjustment", (S.Alternative("base", "base"),
                                            S.Alternative("copy", "add twice x",
                                                          add=("x_twice",))), "base")
    curve = _curve(choice)
    assert [s.key for s in curve.specs] == ["adj=base"]
    (missing,) = curve.not_estimated
    assert missing.key == "adj=copy"
    assert missing.reason == (
        f"On these {N} rows the model matrix is singular: `x` and `x_twice` are fixed "
        f"combinations of the other columns and the intercept, so their coefficients are not "
        f"identified. No estimate, interval or p-value is reported; leave one of the dependent "
        f"columns out.")
