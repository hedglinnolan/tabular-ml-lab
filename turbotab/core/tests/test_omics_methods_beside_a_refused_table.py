"""The methods paragraph beside a feature-wise table that reports no test (the verifier's T2V-2;
the orchestrator's ruling on the swarm's tail, package T3).

The fit writes each model's methods paragraph (``methods.omics.fit_methods``), and feature-wise
regression's ends with its tests: "Each of the 12 features was tested in its own linear model
with the covariates; Benjamini–Hochberg q-values across the 12 tests ...". Beside a refused table
it said so of 0 features and 0 tests, a test and an adjustment that never ran. Now, where the
table is refused (its record's ``refused``) or no row carries a p-value, the paragraph states no
test, no count of tests and no adjustment: it says the table was refused, or, where the table
still gives its estimates (no intervals), that its tests and intervals were, in the words of the
refusal the table holds.

Constructions, each on ``estimand_fixtures.cohort`` (400 rows), design and fit on every row:

* (1) a category, `grp` (3 levels), as the study factor: the model matrix holds it as indicators
  only, so the table is refused, the fit is not (the linear model beside it is fit);
* (2) the study factor `fiber` in a declared spline: refused as a formed study factor;
* (3) the rows in 4 `site` units, fewer than the cluster floor: the table gives each estimate
  (statsmodels' least squares on the design written out) and no interval or p-value.

An ordinary feature-wise fit under the metabolomics and genomics lenses keeps its sentence byte for
byte, its p-value statsmodels' and its q-value Benjamini–Hochberg's by statsmodels'
``multipletests``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

import turbotab.core.models  # noqa: F401 - registers the families
from turbotab.core import decisions as d
from turbotab.core.methods import omics
from turbotab.core.tests.acceptance import estimand_fixtures as ef

N = 400
NO_FACTOR = ("No study factor to test: feature-wise regression tests the columns with the role of "
             "what you study, one at a time, adjusted for the rest; `grp` has that role, but the "
             "model matrix holds no column of that name.")
FORMED = ("Feature-wise regression tests each study factor as one straight-line column, one at a "
          "time, adjusted for the rest; `fiber` has that role, and its declared form, a restricted "
          "cubic spline, enters it as its spline terms, not one straight-line column.")
# The claims a paragraph makes only of tests that ran.
CLAIMS = ("Each of the", "was tested", "tests (", "q-values", "Benjamini–Hochberg",
          "unadjusted p-values", "multiplicity control")


def _featurewise(out: dict) -> dict:
    return next(m for m in out["fit_raw"]["models"] if m["family"] == "featurewise")


def _claims_no_test(said: str) -> None:
    for claim in CLAIMS:
        assert claim not in said, (claim, said)


def _adjusted_design(frame: pd.DataFrame, *columns: str) -> pd.DataFrame:
    """The covariates the adjustment answers keep (age, sex, smoking, activity; bmi's timing is
    unknown, so it is left out), with the columns named, as a least-squares design."""
    return ef.design_matrix(frame, ["age", "sex", "smoking", "activity", *columns])


@pytest.mark.parametrize("lens", ["metabolomics", "genomics", "clinical"])
def test_1_a_category_as_the_study_factor_writes_the_refusal_not_a_test(tmp_path, lens):
    frame = ef.cohort(N, seed=7)
    frame["grp"] = np.random.default_rng(11).choice(["a", "b", "c"], N)
    roles = {**{k: v for k, v in ef.ROLES.items() if k != "fiber"}, "grp": "exposure"}
    st = ef.state(target="glucose", task="regression", measure="mean_difference", exposure="grp",
                  roles=roles, models=["linear", "featurewise"], lens=[lens])
    out = ef.run(frame, tmp_path, st, fit=True)
    model = _featurewise(out)
    assert model["inference"]["refused"] == NO_FACTOR
    assert not model["coefficients"]
    said = model["methods"]
    _claims_no_test(said)
    assert said == f"The feature-wise table was refused. {NO_FACTOR}"


def test_2_a_study_factor_in_a_spline_writes_the_refusal_not_a_test(tmp_path):
    st = ef.state(target="glucose", task="regression", measure="mean_difference",
                  models=["linear", "featurewise"], lens=["metabolomics"],
                  exposure_forms={"fiber": d.ExposureFormSpec(form="spline", knots=4)})
    out = ef.run(ef.cohort(N, seed=7), tmp_path, st)
    model = _featurewise(out)
    assert model["inference"]["refused"] == FORMED
    said = model["methods"]
    _claims_no_test(said)
    assert said == f"The feature-wise table was refused. {FORMED}"


def test_3_below_the_cluster_floor_the_paragraph_gives_the_estimates_and_no_test(tmp_path):
    """The table gives the estimate, statsmodels' least squares of glucose on fiber and the
    adjusted covariates, and no p-value: its tests and intervals were refused, in its own words."""
    from turbotab.core.models.inference import min_clusters

    frame = ef.cohort(N, seed=7)
    frame["site"] = frame["pid"] % 4
    st = ef.state(target="glucose", task="regression", measure="mean_difference",
                  roles={**ef.ROLES, "site": "cluster"}, models=["featurewise"],
                  lens=["metabolomics"], reading_confirmations={"cluster:site": "yes"})
    out = ef.run(frame, tmp_path, st)
    model = _featurewise(out)
    reason = model["inference"]["refused"]
    assert 4 < min_clusters() and reason.startswith(
        f"`site` has 4 units, fewer than the {min_clusters()} TurboTab requires")
    (row,) = model["coefficients"]
    expected = sm.OLS(frame["glucose"].to_numpy(float), _adjusted_design(frame, "fiber")).fit()
    assert row["feature"] == "fiber" and row["p"] is None and row["q"] is None
    assert row["estimate"] == pytest.approx(float(expected.params["fiber"]), rel=1e-9)
    said = model["methods"]
    _claims_no_test(said)
    assert said == (f"The feature-wise table gives estimates alone: its tests and intervals were "
                    f"refused. {reason}")


@pytest.mark.parametrize("lens", ["metabolomics", "genomics"])
def test_an_ordinary_feature_wise_fit_keeps_its_sentence(tmp_path, lens):
    """Silent: one study factor tested, its p-value statsmodels' and its q-value Benjamini–
    Hochberg's, the sentence as it was."""
    frame = ef.cohort(N, seed=7)
    st = ef.state(target="glucose", task="regression", measure="mean_difference",
                  models=["featurewise"], lens=[lens])
    out = ef.run(frame, tmp_path, st)
    model = _featurewise(out)
    assert model["inference"]["refused"] is None
    expected = sm.OLS(frame["glucose"].to_numpy(float), _adjusted_design(frame, "fiber")).fit()
    (row,) = model["coefficients"]
    p = float(expected.pvalues["fiber"])
    assert row["p"] == pytest.approx(p, rel=1e-6)
    found = int(np.sum(multipletests([p], method="fdr_bh")[1] < 0.05))
    assert model["methods"] == (
        f"Each of the 1 features was tested in its own linear model with the covariates; "
        f"Benjamini–Hochberg q-values across the 1 tests ({found} below 0.05), every feature "
        f"shown.")


def test_a_table_with_no_residual_degrees_of_freedom_is_said_to_be_refused():
    """Three rows and two covariates beside the intercept: n − rank[1, Z] − 1 = 3 − 3 − 1 < 1, so
    the table holds no estimate and no p-value, and its sentence is its refusal."""
    from turbotab.core.models.featurewise import featurewise_table

    rng = np.random.default_rng(5)
    matrix = pd.DataFrame({"x": rng.normal(size=3), "a": rng.normal(size=3),
                           "b": rng.normal(size=3)})
    table = featurewise_table(matrix, rng.normal(size=3), "regression", ["x"])
    reason = table.info["refused"]
    assert reason.startswith("3 rows leave no residual degrees of freedom")
    assert [(r["estimate"], r["p"]) for r in table.rows] == [(None, None)]
    said = omics.featurewise_details({"rows": table.rows, "refused": reason}, "bh")
    _claims_no_test(said)
    assert said == f"The feature-wise table was refused. {reason}"


@pytest.mark.parametrize("method", ["bh", "stated_count", "none"])
def test_rows_with_no_p_value_and_no_refusal_claim_no_test(method):
    """A table no one refused whose one row has no p-value (a study factor with nothing left once
    the covariates are projected out): no test, count or adjustment is claimed, under each
    multiplicity method; with a p-value the method's sentence is written (silent)."""
    untested = {"rows": [{"feature": "x", "estimate": None, "p": None, "q": None}]}
    said = omics.featurewise_details(untested, method)
    _claims_no_test(said)
    assert said == "The feature-wise table holds no p-value, so no feature-wise test is reported."
    tested = {"rows": [{"feature": "x", "estimate": 1.0, "p": 0.01, "q": 0.01}]}
    assert omics.featurewise_details(tested, method).startswith(
        "Each of the 1 features was tested in its own linear model with the covariates; ")
