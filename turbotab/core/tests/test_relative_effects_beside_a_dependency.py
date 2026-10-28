"""Under the all-components model, each nutrient's average relative effect beside a dependency of
the model matrix's columns: served where the dependency holds no energy column, and refused in
words the table carries where it holds one (the verifier's T5bV-1).

The relative effect of nutrient j is θ_j = β_j − Σ_k w_k β_k over every other energy column k
(``methods.energy.relative_effect_rows``), so it combines the coefficient of every energy column.
With `fat2` = 2 × `fat_g` among the covariates, `kcal_from_fat_g` (9 × `fat_g`) and `fat2` are
in a dependency: their coefficients are refused by row and the others served, and every relative
effect, which needs `kcal_from_fat_g`'s coefficient, has none. Before this repair the fit and the
effects stage dropped all four relative effects and no sentence said so; on the base the whole
table was refused.

Truth from construction: chain 2's recalls (``rc_fixtures.chain2_table``), each person's mean kcal
from each source by pandas (Atwater's factors, the rest as energy less the four:
``test_ms5_regression_calibration.kcal_days``), and statsmodels' OLS with HC3 on the moved energy
columns (x_k + w_k x_j, w_k each other source's share of the remaining energy) for θ_j and its
standard error, per gram (× the source's kcal per gram). The guard's silent side: `age2` =
3 × `age` + 1, a dependency of no energy column, serves each relative effect as statsmodels gives
it with `age2` left out, which spans the same columns.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from turbotab.core.tests.acceptance import estimand_fixtures as ef

NUTRIENTS = ["protein_g", "fat_g", "carbohydrate_g", "alcohol_g"]
REFUSED = ("No average relative effect is reported for `protein_g`, `fat_g`, `carbohydrate_g` "
           "and `alcohol_g`: each combines the coefficients of every energy column, and "
           "`kcal_from_fat_g` is in the dependency, so its coefficient is not identified.")
FAT_DEPENDENCY = ("On these 160 rows the model matrix is singular: `kcal_from_fat_g` and `fat2` "
                  "are fixed combinations of the other columns and the intercept, so their "
                  "coefficients are not identified.")


def _frame(copy: str | None) -> pd.DataFrame:
    from turbotab.core.tests.acceptance.rc_fixtures import chain2_table

    frame, _ = chain2_table(seed=4, per_psu=10, missing_share=0.0)
    if copy == "fat2":
        frame["fat2"] = 2 * frame["fat_g"]
    elif copy == "age2":
        frame["age2"] = 3 * frame["age"] + 1
    return frame


def _run(tmp_path: Path, copy: str | None) -> dict[str, Any]:
    """The fit and effects stages on chain 2 (160 people, two recalls each, `protein_g` the
    exposure, adjusted for `age`, `sex` and `bmi`, every energy source in the all-components
    model), ``copy`` among the covariates."""
    from turbotab.core.tests.acceptance.test_ms5_regression_calibration import (CHAIN_ROLES,
                                                                                chain_state)
    from turbotab.core.tests.graph_runner import GraphRun

    frame = _frame(copy)
    roles = dict(CHAIN_ROLES)
    adjusted = ["age", "sex", "bmi", *([copy] if copy else [])]
    if copy:
        roles[copy] = "covariate"
    frame.to_csv(tmp_path / "chain.csv", index=False)
    run = GraphRun(tmp_path / "chain.csv", tmp_path / "project")
    try:
        state = chain_state(roles=roles, split={"holdout": 0.0, "seed": 7, "folds": 5},
                            estimand={"exposure": "protein_g", "measure": "mean_difference"},
                            adjustment={c: {"exposure": "protein_g", **ef.CONFOUNDER}
                                        for c in adjusted})
        out = run.public(run.run(state, upto=["effects", "fit"]))
    finally:
        run.close()
    [model] = out["fit"]["models"]
    [two] = [s for s in out["effects"]["families"][0]["sequence"] if s["key"] == "model_2"]
    return {"frame": frame, "fit": model, "model_2": two}


def _people(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Each person's design by pandas: `sex` = M, `age`, `bmi` and the mean kcal from each source
    and the rest; and `sbp`."""
    from turbotab.core.tests.acceptance.test_ms5_regression_calibration import kcal_days

    D, names, people = kcal_days(frame)
    X = pd.DataFrame(D.mean(axis=1), columns=names, index=people.index)
    X.insert(0, "bmi", people["bmi"].to_numpy(dtype=float))
    X.insert(0, "age", people["age"].to_numpy(dtype=float))
    X.insert(0, "sex_M", (people["sex"] == "M").to_numpy(dtype=float))
    return X, people["sbp"].astype(float)


def _relative_truth(frame: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Each nutrient's average relative effect per gram, (estimate, HC3 standard error), from
    statsmodels on the moved energy columns."""
    from turbotab.core.tests.acceptance.rc_fixtures import ATWATER

    X, y = _people(frame)
    kcal = [c for c in X.columns if c.startswith("kcal_from_")]
    out = {}
    for n in NUTRIENTS:
        j = f"kcal_from_{n}"
        others = [k for k in kcal if k != j]
        means = X[others].mean()
        moved = X.copy()
        for k in others:
            moved[k] = X[k] + means[k] / means.sum() * X[j]
        fit = sm.OLS(y.to_numpy(), sm.add_constant(moved)).fit(cov_type="HC3")
        per = ATWATER[n.removesuffix("_g")]
        out[n] = (float(fit.params[j]) * per, float(fit.bse[j]) * per)
    return out


def test_beside_a_copy_of_an_energy_source_the_relative_effects_are_refused_in_words(tmp_path):
    """`fat2` = 2 × `fat_g`: the fit's table serves `kcal_from_protein_g` and refuses
    `kcal_from_fat_g` and `fat2` by row, and it reports no average relative effect, saying why in
    a concern that names `kcal_from_fat_g`; so does the effects stage's Model 2, which shows the
    exposure's own coefficient alone."""
    frame = _frame("fat2")
    X, _ = _people(frame)
    X["fat2"] = frame.groupby("seqn", sort=True)["fat2"].mean().to_numpy()
    values = sm.add_constant(X).to_numpy(dtype=float)
    assert np.linalg.matrix_rank(values) == values.shape[1] - 1  # the premise: one dependency
    run = _run(tmp_path, "fat2")
    fit, two = run["fit"], run["model_2"]
    assert fit["inference"]["refused"] is None
    rows = {r["feature"]: r for r in fit["coefficients"]}
    assert rows["kcal_from_fat_g"]["estimate"] is None and rows["fat2"]["estimate"] is None
    assert rows["kcal_from_protein_g"]["estimate"] is not None
    assert not [f for f in rows if f.endswith("_relative")]
    assert any(c.startswith(FAT_DEPENDENCY) for c in fit["concerns"])
    assert REFUSED in fit["concerns"]
    assert [e["feature"] for e in two["effects"]] == ["kcal_from_protein_g"]
    assert REFUSED in two["concerns"]


def test_beside_a_covariates_own_rescaling_the_relative_effects_are_served_as_statsmodels(
        tmp_path):
    """`age2` = 3 × `age` + 1, a dependency of no energy column: each relative effect at the fit,
    and `protein_g`'s at the effects stage's Model 2, is statsmodels' with `age2` left out (to
    1e-9), and no concern says one is not reported."""
    truth = _relative_truth(_frame(None))
    run = _run(tmp_path, "age2")
    fit, two = run["fit"], run["model_2"]
    rows = {r["feature"]: r for r in fit["coefficients"]}
    assert rows["age"]["estimate"] is None and rows["age2"]["estimate"] is None
    for n in NUTRIENTS:
        row = rows[f"{n}_relative"]
        assert (row["estimate"], row["se"]) == pytest.approx(truth[n], rel=1e-9), n
    [served] = [e for e in two["effects"] if e["feature"] == "protein_g_relative"]
    assert (served["estimate"], served["se"]) == pytest.approx(truth["protein_g"], rel=1e-9)
    said = [*fit["concerns"], *two["concerns"]]
    assert not [c for c in said if "relative effect" in c]


def test_the_refusals_words_name_the_energy_columns_in_the_dependency_or_quote_the_refit():
    """The words, from construction: the energy columns the table refuses by row, one or several;
    where it refuses none, the refit's own refusal; none where no refit was refused."""
    from turbotab.core.stages.modeling import relative_refusal

    rows = [{"feature": "kcal_from_fat_g", "estimate": None},
            {"feature": "kcal_from_protein_g", "estimate": 0.1}, {"feature": "fat2", "estimate": None}]
    refused = {n: "On these rows the model matrix is singular." for n in NUTRIENTS}
    assert relative_refusal(NUTRIENTS, refused, rows) == REFUSED
    two = [*rows, {"feature": "kcal_from_alcohol_g", "estimate": None}]
    assert relative_refusal(["protein_g", "alcohol_g"], refused, two) == (
        "No average relative effect is reported for `protein_g` and `alcohol_g`: each combines "
        "the coefficients of every energy column, and `kcal_from_fat_g` and `kcal_from_alcohol_g` "
        "are in the dependency, so their coefficients are not identified.")
    served = [{"feature": "kcal_from_fat_g", "estimate": 0.2}]
    assert relative_refusal(["fat_g"], refused, served) == (
        "No average relative effect is reported for `fat_g`: the refit it is read from was "
        "refused. On these rows the model matrix is singular.")
    assert relative_refusal(NUTRIENTS, {}, rows) is None
