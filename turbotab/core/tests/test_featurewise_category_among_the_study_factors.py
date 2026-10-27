"""A category among feature-wise regression's study factors (the verifier's T2V-3 on the swarm's
tail).

Feature-wise regression tests each study factor that the model matrix holds as one column of its
name, adjusted for every other column but the other study factors tested (``split_columns``). A
category enters the matrix as indicators of its levels, not one column, so it is not tested, and
its indicators are among the adjustment columns of each test. The fit stage's table has always
done so, saying nothing of it; the effects stage tested each indicator as a study factor of its
own, so its Model 2 row for `fiber` (adjusted for `grp`'s indicators one at a time, beside the
others tested) was not the fit's. Now both stages test the fit's study factors, and say which one
is not tested and where its indicators went.

The construction: ``estimand_fixtures.cohort`` (400 rows) with a category `grp` (levels a, b, c;
the one-hot step drops the first level, so `grp_b` and `grp_c` enter) shifting `glucose`. The
truth is statsmodels' least squares on the design written out here, with its classical standard
error (the family's, ``classical_tests``):

* the exposure family {`fiber`, `grp`}: the fit's row for `fiber` and the effects stage's Model 2
  row are each that of glucose ~ fiber + the Model 2 covariates + `grp`'s indicators, the
  unadjusted one that of glucose ~ fiber + `grp`'s indicators, Model 3's beside `bmi`; no row is
  `grp`'s or an indicator's, and each caption names `grp` and its 2 indicators;
* silent where the matrix holds no indicator of it: `fiber` the one study factor of the estimand,
  `grp` another column with that role, the unadjusted model holds `fiber` alone, so its caption
  says nothing of `grp`, and Model 2's does;
* silent for a family of numeric study factors: each is tested without the other, its caption
  pinned whole, its false-discovery count statsmodels' Benjamini–Hochberg.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

import turbotab.core.models  # noqa: F401 - registers the families
from turbotab.core.tests.acceptance import estimand_fixtures as ef

N = 400
ROLES = {"pid": "identifier", "age": "covariate", "sex": "covariate", "smoking": "covariate",
         "activity": "covariate", "bmi": "covariate", "fiber": "exposure"}
MODEL_2 = ["sex_male", "age", "smoking", "activity"]  # the confounders and the precision column
MODEL_3 = [*MODEL_2, "bmi"]  # beside the column of unknown timing
GRP = ["grp_b", "grp_c"]
NOT_TESTED = ("`grp` also has the role of what you study, but enters the model matrix as "
              "indicators of its levels, not one column, so it is not tested: its 2 indicators "
              "are among the adjustment columns.")


def _frame(**extra) -> pd.DataFrame:
    frame = ef.cohort(N, seed=7)
    rng = np.random.default_rng(11)
    frame["grp"] = rng.choice(["a", "b", "c"], N)
    frame["glucose"] = frame["glucose"] + np.where(frame["grp"] == "c", 3.0, 0.0)
    for column, values in extra.items():
        frame[column] = values
    return frame


def _design(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    X = pd.DataFrame({"(intercept)": np.ones(len(frame))}, index=frame.index)
    for c in columns:
        if c == "sex_male":
            X[c] = (frame["sex"] == "male").astype(float)
        elif c in GRP:
            X[c] = (frame["grp"] == c[-1]).astype(float)
        else:
            X[c] = frame[c].astype(float)
    return X


def _ols(frame: pd.DataFrame, tested: str, adjust: list[str]) -> tuple[float, float, float]:
    """(estimate, classical standard error, p) of ``tested`` in glucose ~ tested + adjust."""
    fit = sm.OLS(frame["glucose"].to_numpy(float), _design(frame, [tested, *adjust])).fit()
    return float(fit.params[tested]), float(fit.bse[tested]), float(fit.pvalues[tested])


def _fit_model(out: dict) -> dict:
    return next(m for m in out["fit_raw"]["models"] if m["family"] == "featurewise")


def _sequence(out: dict) -> dict:
    index = next(i for i, f in enumerate(out["effects"]["families"])
                 if f["family"] == "featurewise")
    return ef.sequence(out["effects"], index)


def _only(rows: list[dict], key: str = "feature") -> dict:
    (row,) = rows
    assert row[key] == "fiber"
    return row


def _matches(row: dict, truth: tuple[float, float, float]) -> None:
    assert row["estimate"] == pytest.approx(truth[0], rel=1e-9)
    assert row["se"] == pytest.approx(truth[1], rel=1e-9)


@pytest.fixture(scope="module")
def family(tmp_path_factory):
    frame = _frame()
    st = ef.state(target="glucose", task="regression", measure="mean_difference",
                  roles={**ROLES, "grp": "exposure"}, models=["linear", "featurewise"],
                  family=True, multiplicity="fdr_bh")
    return frame, ef.run(frame, tmp_path_factory.mktemp("family"), st)


def test_a_category_in_the_family_is_adjusted_for_not_tested_at_the_fit_and_the_effects_stage(
        family):
    frame, out = family
    fit = _only(_fit_model(out)["coefficients"])
    _matches(fit, _ols(frame, "fiber", [*MODEL_2, *GRP]))
    sequence = _sequence(out)
    model_2 = _only(sequence["model_2"]["effects"])
    assert (model_2["estimate"], model_2["se"]) == (fit["estimate"], fit["se"])
    _matches(_only(sequence["crude"]["effects"]), _ols(frame, "fiber", GRP))
    _matches(_only(sequence["model_3"]["effects"]), _ols(frame, "fiber", [*MODEL_3, *GRP]))
    caption = _fit_model(out)["inference"]["caption"]
    assert caption.startswith("Each of 1 study factors tested on its own with 6 adjustment columns;")
    assert NOT_TESTED in caption
    for key, n_adjust in (("crude", 2), ("model_2", 6), ("model_3", 7)):
        said = sequence[key]["inference"]["caption"]
        assert said.startswith(f"Each of 1 study factors tested on its own with {n_adjust} "
                               f"adjustment columns;"), key
        assert NOT_TESTED in said, key


def test_a_category_beside_the_estimands_one_study_factor_is_named_where_the_model_holds_it(
        tmp_path):
    frame = _frame()
    st = ef.state(target="glucose", task="regression", measure="mean_difference",
                  roles={**ROLES, "grp": "exposure"}, models=["linear", "featurewise"])
    out = ef.run(frame, tmp_path, st)
    assert NOT_TESTED in _fit_model(out)["inference"]["caption"]
    sequence = _sequence(out)
    crude = sequence["crude"]
    _matches(_only(crude["effects"]), _ols(frame, "fiber", []))
    assert crude["inference"]["caption"].startswith(
        "Each of 1 study factors tested on its own with no adjustment columns;")
    assert "`grp`" not in crude["inference"]["caption"]  # silent: the model holds no indicator of it
    _matches(_only(sequence["model_2"]["effects"]), _ols(frame, "fiber", [*MODEL_2, *GRP]))
    assert NOT_TESTED in sequence["model_2"]["inference"]["caption"]


def test_a_family_of_numeric_study_factors_tests_each_without_the_other(tmp_path):
    rng = np.random.default_rng(5)
    frame = _frame(sugar=rng.normal(50, 10, N))
    frame["glucose"] = frame["glucose"] + 0.2 * frame["sugar"]
    st = ef.state(target="glucose", task="regression", measure="mean_difference",
                  roles={**ROLES, "sugar": "exposure"}, models=["linear", "featurewise"],
                  family=True, multiplicity="fdr_bh")
    out = ef.run(frame, tmp_path, st)
    model = _fit_model(out)
    rows = {r["feature"]: r for r in model["coefficients"]}
    assert set(rows) == {"fiber", "sugar"}
    truth = {x: _ols(frame, x, MODEL_2) for x in rows}
    for x, row in rows.items():
        _matches(row, truth[x])
    found = int(multipletests([truth["fiber"][2], truth["sugar"][2]], method="fdr_bh")[1]
                .__lt__(0.05).sum())
    df = N - (1 + len(MODEL_2)) - 1
    assert model["inference"]["caption"] == (
        f"Each of 2 study factors tested on its own with {len(MODEL_2)} adjustment columns; the "
        f"estimate is the change in the outcome per unit of each study factor, adjusted for them. "
        f"95% intervals from each test's own residual variance (classical least squares, exact "
        f"under the null for normal errors), on t({df}). Benjamini–Hochberg q < 0.05: {found} of "
        f"2. Estimated from all {N} analyzed rows.")
    model_2 = {r["feature"]: r for r in _sequence(out)["model_2"]["effects"]}
    for x in rows:
        assert (model_2[x]["estimate"], model_2[x]["se"]) == (rows[x]["estimate"], rows[x]["se"])


def test_the_table_counts_each_categorys_indicators_and_no_longer_predictors_columns():
    """On a matrix written out by hand: `g` (3 levels) and `h` (2) are categories with the role,
    `g_size` a covariate whose name begins with `g`'s, so its column is its own, not `g`'s
    (``estimand.primary_features``). The one test of `x` is least squares on every other column."""
    from turbotab.core.models import featurewise as fw

    rng = np.random.default_rng(3)
    n = 120
    g = rng.choice(["a", "b", "c"], n)
    h = rng.choice(["n", "y"], n)
    matrix = pd.DataFrame({"x": rng.normal(size=n), "g_size": rng.normal(size=n),
                           "g_b": (g == "b").astype(float), "g_c": (g == "c").astype(float),
                           "h_y": (h == "y").astype(float)})
    y = 1.0 + 0.5 * matrix["x"] + matrix["g_c"] + rng.normal(size=n)
    columns = list(matrix.columns)
    entered = fw.indicator_columns(columns, ["x"], ["g", "h"], ["x", "g_size", "g", "h"])
    assert entered == {"g": ["g_b", "g_c"], "h": ["h_y"]}
    table = fw.featurewise_table(matrix, y, "regression", ["x", "g", "h"], entered=entered)
    (row,) = table.rows
    fit = sm.OLS(y.to_numpy(), sm.add_constant(matrix)).fit()
    assert row["estimate"] == pytest.approx(float(fit.params["x"]), rel=1e-9)
    assert row["se"] == pytest.approx(float(fit.bse["x"]), rel=1e-9)
    assert table.info["caption"].endswith(
        "`g` and `h` also have the role of what you study, but each enters the model matrix as "
        "indicators of its levels, not one column, so neither is tested: their indicators (2 of "
        "`g` and 1 of `h`) are among the adjustment columns.")
    alone = fw.featurewise_table(matrix, y, "regression", ["x", "h"], entered={"h": ["h_y"]})
    assert alone.info["caption"].endswith(
        "`h` also has the role of what you study, but enters the model matrix as indicators of its "
        "levels, not one column, so it is not tested: its indicator is among the adjustment "
        "columns.")
    plain = fw.featurewise_table(matrix, y, "regression", ["x"])
    assert plain.info["caption"].endswith("Benjamini–Hochberg q < 0.05: 1 of 1.")  # silent
