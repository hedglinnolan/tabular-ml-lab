"""Tier A: energy adjustment (NUTRITION_PACK §04), checked against known answers."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score, cross_validate
from sklearn.pipeline import Pipeline

from turbotab.core.methods.energy import (
    METHODS,
    EnergyAdjuster,
    EnergyAdjustmentNotApplicable,
    applicable_methods,
    describe_method,
)

NUTRIENTS = ["protein_g", "fat_g", "carbohydrate_g"]


def diet(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Simulated intakes in which every macronutrient rises with total energy."""
    rng = np.random.default_rng(seed)
    energy = rng.normal(2100.0, 450.0, n).clip(900.0, 4200.0)
    protein = (15.0 + 0.030 * energy + rng.normal(0, 12, n)).clip(20.0)
    fat = (10.0 + 0.032 * energy + rng.normal(0, 14, n)).clip(15.0)
    carbohydrate = (20.0 + 0.110 * energy + rng.normal(0, 30, n)).clip(60.0)
    sodium = (800.0 + 1.1 * energy + rng.normal(0, 400, n)).clip(300.0)
    sex = rng.choice(["F", "M"], n)
    age = rng.integers(20, 80, n)
    y = 0.02 * protein - 0.015 * fat + 0.004 * energy + 0.01 * age + rng.normal(0, 1.0, n)
    return pd.DataFrame({
        "age": age, "sex": sex, "energy_kcal": energy, "protein_g": protein, "fat_g": fat,
        "carbohydrate_g": carbohydrate, "sodium_mg": sodium, "y": y,
    })


def ols(columns: list, y: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(y)), *columns])
    return np.linalg.lstsq(design, y, rcond=None)[0]


def manual_residual(n_fit, e_fit, n_new, e_new):
    """N_adj = residual of N ~ E on the fitting rows + N_hat(mean E), written out longhand."""
    slope = np.cov(n_fit, e_fit, ddof=1)[0, 1] / np.var(e_fit, ddof=1)
    intercept = n_fit.mean() - slope * e_fit.mean()
    residual = n_new - (intercept + slope * e_new)
    return residual + (intercept + slope * e_fit.mean())


# ── residual method ──────────────────────────────────────────────────────────

def test_residual_is_uncorrelated_with_energy_on_the_fitting_rows():
    df = diet()
    out = EnergyAdjuster("residual", "energy_kcal", NUTRIENTS).fit_transform(df)
    for n in NUTRIENTS:
        assert abs(np.corrcoef(df[n], df["energy_kcal"])[0, 1]) > 0.3  # it was correlated
        assert abs(np.corrcoef(out[f"{n}_adj"], df["energy_kcal"])[0, 1]) < 1e-8
        # The added constant is N_hat at mean E, so the adjusted mean equals the raw mean.
        assert out[f"{n}_adj"].mean() == pytest.approx(df[n].mean(), rel=1e-12)


def test_residual_transform_uses_the_fitting_rows_coefficients():
    df = diet(600, seed=1)
    a, b = df.iloc[:350], df.iloc[350:]
    step = EnergyAdjuster("residual", "energy_kcal", NUTRIENTS).fit(a)
    out_b = step.transform(b)
    for n in NUTRIENTS:
        expected = manual_residual(a[n].to_numpy(), a["energy_kcal"].to_numpy(),
                                   b[n].to_numpy(), b["energy_kcal"].to_numpy())
        np.testing.assert_allclose(out_b[f"{n}_adj"].to_numpy(), expected, rtol=0, atol=1e-10)
        # B's own coefficients give a different answer, so the check above can fail.
        own = manual_residual(b[n].to_numpy(), b["energy_kcal"].to_numpy(),
                              b[n].to_numpy(), b["energy_kcal"].to_numpy())
        assert np.max(np.abs(own - expected)) > 1e-3
    assert out_b.index.equals(b.index)


def test_residual_log_variant_matches_a_manual_log_log_fit():
    df = diet(600, seed=2)
    a, b = df.iloc[:400], df.iloc[400:]
    step = EnergyAdjuster("residual", "energy_kcal", NUTRIENTS, log_transform=True).fit(a)
    fitted_a, out_b = step.transform(a), step.transform(b)
    for n in NUTRIENTS:
        x, y = np.log(a["energy_kcal"].to_numpy()), np.log(a[n].to_numpy())
        slope = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)
        expected = np.exp(np.log(b[n].to_numpy()) - slope * (np.log(b["energy_kcal"].to_numpy()) - x.mean()))
        np.testing.assert_allclose(out_b[f"{n}_adj"].to_numpy(), expected, rtol=1e-12)
        assert abs(np.corrcoef(np.log(fitted_a[f"{n}_adj"]), x)[0, 1]) < 1e-8
        params = next(e["params"] for e in step.lineage() if e["output"] == f"{n}_adj")
        assert params["reference_energy"] == pytest.approx(np.exp(x.mean()))
        assert params["reference_kind"] == "geometric mean"


def test_residual_log_refuses_zeros_rather_than_padding_them():
    df = diet()
    df.loc[df.index[:5], "fat_g"] = 0.0
    with pytest.raises(ValueError, match=r"fat_g is zero or negative in 5"):
        EnergyAdjuster("residual", "energy_kcal", ["fat_g"], log_transform=True).fit(df)


def test_residual_and_standard_models_give_the_same_nutrient_coefficient():
    """Y ~ N_adj + E reparametrizes Y ~ N + E, so the residual method with energy kept gives the
    standard coefficient; without energy and without covariates, Frisch-Waugh-Lovell gives it too."""
    df = diet(800, seed=3)
    y = df["y"].to_numpy()
    out = EnergyAdjuster("residual", "energy_kcal", ["protein_g"]).fit_transform(df)
    standard = ols([df["protein_g"], df["energy_kcal"]], y)[1]
    kept = ols([out["protein_g_adj"], out["energy_kcal"]], y)[1]
    assert kept == pytest.approx(standard, rel=1e-8, abs=1e-12)
    dropped = EnergyAdjuster("residual_energy_dropped", "energy_kcal", ["protein_g"]).fit_transform(df)
    assert "energy_kcal" not in dropped.columns
    assert ols([dropped["protein_g_adj"]], y)[1] == pytest.approx(standard, rel=1e-8, abs=1e-12)
    # The same through a Pipeline, where the step feeds an sklearn model.
    pipe = Pipeline([("energy", EnergyAdjuster("residual", "energy_kcal", ["protein_g"])),
                     ("model", LinearRegression())]).set_output(transform="pandas")
    pipe.fit(df[["energy_kcal", "protein_g"]], y)
    assert list(pipe[:-1].get_feature_names_out()) == ["energy_kcal", "protein_g_adj"]
    assert pipe[-1].coef_[1] == pytest.approx(standard, rel=1e-8)


def test_residual_missing_values_stay_missing_and_are_left_out_of_the_fit():
    df = diet(300, seed=4)
    df.loc[df.index[:10], "protein_g"] = np.nan
    step = EnergyAdjuster("residual", "energy_kcal", ["protein_g"]).fit(df)
    out = step.transform(df)
    assert out["protein_g_adj"].iloc[:10].isna().all()
    complete = df.iloc[10:]
    expected = manual_residual(complete["protein_g"].to_numpy(), complete["energy_kcal"].to_numpy(),
                               complete["protein_g"].to_numpy(), complete["energy_kcal"].to_numpy())
    np.testing.assert_allclose(out["protein_g_adj"].iloc[10:].to_numpy(), expected, atol=1e-10)
    assert step.params_["protein_g"]["n_fit"] == 290


# ── density, standard, none ─────────────────────────────────────────────────

def test_density_is_n_over_e_and_energy_leaves_the_model():
    df = diet()
    out = EnergyAdjuster("density", "energy_kcal", NUTRIENTS).fit_transform(df)
    for n in NUTRIENTS:
        np.testing.assert_array_equal(out[f"{n}_per_energy_kcal"], df[n] / df["energy_kcal"])
    assert "energy_kcal" not in out.columns
    for col in ["age", "sex", "sodium_mg", "y"]:
        pd.testing.assert_series_equal(out[col], df[col])


def test_multivariate_density_keeps_energy_as_its_own_term():
    df = diet()
    step = EnergyAdjuster("density_multivariate", "energy_kcal", ["fat_g"])
    out = step.fit_transform(df)
    np.testing.assert_array_equal(out["fat_g_per_energy_kcal"], df["fat_g"] / df["energy_kcal"])
    pd.testing.assert_series_equal(out["energy_kcal"], df["energy_kcal"])
    assert list(out.columns) == ["age", "sex", "energy_kcal", "protein_g", "fat_g_per_energy_kcal",
                                 "carbohydrate_g", "sodium_mg", "y"]


def test_density_refuses_zero_energy():
    df = diet()
    df.loc[df.index[3], "energy_kcal"] = 0.0
    with pytest.raises(ValueError, match="N/E is undefined"):
        EnergyAdjuster("density", "energy_kcal", ["fat_g"]).fit(df)


def test_standard_leaves_every_column_as_it_was():
    df = diet()
    out = EnergyAdjuster("standard", "energy_kcal", NUTRIENTS).fit_transform(df)
    pd.testing.assert_frame_equal(out, df)


def test_none_takes_total_energy_out_and_leaves_every_other_column_as_it_was():
    """No energy adjustment means total energy is not in the model (audit ME-02)."""
    df = diet()
    step = EnergyAdjuster("none", "energy_kcal", NUTRIENTS)
    out = step.fit_transform(df)
    pd.testing.assert_frame_equal(out, df.drop(columns="energy_kcal"))
    assert step.dropped_columns_ == ["energy_kcal"]
    # Further energy-role columns leave with it.
    both = df.assign(energy_kj=df["energy_kcal"] * 4.184)
    out = EnergyAdjuster("none", "energy_kcal", [], leave_out=["energy_kj"]).fit_transform(both)
    pd.testing.assert_frame_equal(out, df.drop(columns="energy_kcal"))


# ── partition ────────────────────────────────────────────────────────────────

def test_partition_converts_grams_with_atwater_factors_and_keeps_the_total():
    df = diet()
    step = EnergyAdjuster("partition", "energy_kcal", ["fat_g", "carbohydrate_g"])
    out = step.fit_transform(df)
    np.testing.assert_allclose(out["kcal_from_fat_g"], 9.0 * df["fat_g"], rtol=0, atol=0)
    np.testing.assert_allclose(out["kcal_from_carbohydrate_g"], 4.0 * df["carbohydrate_g"], rtol=0, atol=0)
    np.testing.assert_allclose(out["kcal_from_other"],
                               df["energy_kcal"] - 9.0 * df["fat_g"] - 4.0 * df["carbohydrate_g"],
                               rtol=0, atol=1e-9)
    np.testing.assert_allclose(out["kcal_from_fat_g"] + out["kcal_from_carbohydrate_g"]
                               + out["kcal_from_other"], df["energy_kcal"], rtol=1e-12)
    assert "energy_kcal" not in out.columns and "fat_g" not in out.columns


def test_partition_takes_explicit_factors_by_column():
    df = diet().rename(columns={"fat_g": "sfa_g"})
    step = EnergyAdjuster("partition", "energy_kcal", ["sfa_g"], atwater={"sfa_g": 9.0}).fit(df)
    np.testing.assert_allclose(step.transform(df)["kcal_from_sfa_g"], 9.0 * df["sfa_g"])


def test_a_nutrient_without_energy_makes_partition_inapplicable_and_fit_refuses():
    df = diet()
    verdicts = applicable_methods(list(df.columns), "energy_kcal", ["fat_g", "sodium_mg"])
    assert verdicts["partition"]["ok"] is False
    assert "sodium_mg carries no energy" in verdicts["partition"]["reason"]
    assert all(verdicts[m]["ok"] for m in ["none", "standard", "residual", "density", "density_multivariate"])
    with pytest.raises(EnergyAdjustmentNotApplicable, match="sodium_mg carries no energy"):
        EnergyAdjuster("partition", "energy_kcal", ["fat_g", "sodium_mg"]).fit(df)


def test_unmarked_units_are_confirmed_by_the_atwater_reconstruction():
    """NHANES names declare no unit; partition runs only once the arithmetic says grams."""
    df = diet(300, seed=5)
    alcohol = np.random.default_rng(5).gamma(0.5, 8.0, len(df))
    energy = 4 * df["protein_g"] + 4 * df["carbohydrate_g"] + 9 * df["fat_g"] + 7 * alcohol
    nhanes = pd.DataFrame({"DR1TKCAL": energy * 1.02, "DR1TPROT": df["protein_g"],
                           "DR1TCARB": df["carbohydrate_g"], "DR1TTFAT": df["fat_g"],
                           "DR1TALCO": alcohol})
    verdict = applicable_methods(list(nhanes.columns), "DR1TKCAL", ["DR1TTFAT"])["partition"]
    assert verdict["ok"] and "Atwater reconstruction" in verdict["reason"]
    step = EnergyAdjuster("partition", "DR1TKCAL", ["DR1TTFAT"]).fit(nhanes)
    assert step.atwater_check_["verdict"] == "pass"
    np.testing.assert_allclose(step.transform(nhanes)["kcal_from_DR1TTFAT"], 9.0 * nhanes["DR1TTFAT"])

    in_kj = nhanes.assign(DR1TKCAL=nhanes["DR1TKCAL"] * 4.184)
    with pytest.raises(EnergyAdjustmentNotApplicable, match="kJ"):
        EnergyAdjuster("partition", "DR1TKCAL", ["DR1TTFAT"]).fit(in_kj)
    # The unit-free methods do not care what unit energy is in.
    EnergyAdjuster("residual", "DR1TKCAL", ["DR1TTFAT"]).fit(in_kj)
    # Without protein the reconstruction cannot vouch for anything, so an unmarked
    # column is refused and a declared one is not.
    no_protein = nhanes.drop(columns="DR1TPROT")
    with pytest.raises(EnergyAdjustmentNotApplicable, match="protein, carbohydrate and fat"):
        EnergyAdjuster("partition", "DR1TKCAL", ["DR1TTFAT"]).fit(no_protein)
    declared = no_protein.rename(columns={"DR1TTFAT": "fat_g"})
    EnergyAdjuster("partition", "DR1TKCAL", ["fat_g"]).fit(declared)


# ── applicability ────────────────────────────────────────────────────────────

def test_nothing_but_none_works_without_an_energy_column():
    for energy in (None, "energy_kcal"):
        verdicts = applicable_methods(["protein_g", "fat_g"], energy, ["protein_g"])
        assert verdicts["none"]["ok"]
        for method in METHODS[1:]:
            assert verdicts[method]["ok"] is False
            assert "energy" in verdicts[method]["reason"]
    with pytest.raises(EnergyAdjustmentNotApplicable, match="energy_kcal is not in this table"):
        EnergyAdjuster("residual", "energy_kcal", ["protein_g"]).fit(diet().drop(columns="energy_kcal"))


def test_an_energy_share_is_not_adjusted_for_energy_again():
    verdicts = applicable_methods(["energy_kcal", "protein_pct_kcal"], "energy_kcal", ["protein_pct_kcal"])
    for method in ["residual", "density", "density_multivariate", "partition"]:
        assert verdicts[method]["ok"] is False
    assert "twice" in verdicts["residual"]["reason"]
    assert verdicts["standard"]["ok"]


def test_log_transform_belongs_to_the_residual_method():
    with pytest.raises(ValueError, match="residual methods only"):
        EnergyAdjuster("density", "energy_kcal", ["fat_g"], log_transform=True).fit(diet())


# ── leakage ──────────────────────────────────────────────────────────────────

SEEN: list = []


class RecordingAdjuster(EnergyAdjuster):
    def fit(self, X, y=None):
        SEEN.append(np.asarray(X.index))
        return super().fit(X, y)


def test_inside_cross_validation_fit_sees_only_training_fold_rows():
    df = diet(500, seed=6)
    X, y = df.drop(columns=["y", "sex"]), df["y"]
    folds = KFold(5, shuffle=True, random_state=0)
    pipe = Pipeline([("energy", RecordingAdjuster("residual", "energy_kcal", NUTRIENTS)),
                     ("model", LinearRegression())])
    SEEN.clear()
    cross_val_score(pipe, X, y, cv=folds)
    expected = [np.asarray(X.index[train]) for train, _ in folds.split(X)]
    assert len(SEEN) == len(expected)
    for seen, train in zip(SEEN, expected):
        np.testing.assert_array_equal(np.sort(seen), np.sort(train))

    # And the coefficients each fold carries are the training rows' own.
    result = cross_validate(pipe, X, y, cv=folds, return_estimator=True, return_indices=True)
    for fitted, train in zip(result["estimator"], result["indices"]["train"]):
        rows = X.iloc[train]
        for n in NUTRIENTS:
            slope = np.cov(rows[n], rows["energy_kcal"], ddof=1)[0, 1] / np.var(rows["energy_kcal"], ddof=1)
            assert fitted["energy"].params_[n]["slope"] == pytest.approx(slope, rel=1e-10)
            assert fitted["energy"].params_[n]["center"] == pytest.approx(rows["energy_kcal"].mean(), rel=1e-12)
            assert fitted["energy"].params_[n]["n_fit"] == len(train)


# ── what the UI reads ────────────────────────────────────────────────────────

@pytest.mark.parametrize("method", METHODS)
def test_every_method_names_its_outputs_lineage_and_estimand(method):
    df = diet()
    # The all-components model needs every main energy source as its own term.
    nutrients = NUTRIENTS if method == "all_components" else ["fat_g", "carbohydrate_g"]
    step = EnergyAdjuster(method, "energy_kcal", nutrients).set_output(transform="pandas")
    out = step.fit_transform(df)
    assert list(out.columns) == list(step.get_feature_names_out())
    lineage = step.lineage()
    assert [entry["output"] for entry in lineage] == list(out.columns)
    for entry in lineage:
        assert set(entry) >= {"output", "inputs", "operation", "formula"}
        assert all(col in df.columns for col in entry["inputs"])
        assert entry["output"] in entry["formula"] or entry["inputs"][0] in entry["formula"]
    json.dumps(lineage, allow_nan=False)  # the Columns panel receives it as it is
    sentence = step.estimand()
    assert sentence.endswith(".") and sentence == describe_method(method)["estimand"]
    assert clone(step).get_params() == step.get_params()


def test_estimands_name_the_pack_s_distinctions():
    kinds = {m: describe_method(m)["kind"] for m in METHODS}
    # Tomova et al. 2022: the multivariable density model's coefficient is "an obscure quantity".
    assert kinds == {"none": "absolute", "standard": "substitution", "residual": "substitution",
                     "residual_energy_dropped": "substitution only without energy-correlated "
                                                "covariates",
                     "density_multivariate": "obscure", "density": "obscure", "partition": "addition",
                     "all_components": "addition and relative effect"}
    assert "substitution" in EnergyAdjuster("standard").estimand()
    assert "addition" in EnergyAdjuster("partition").estimand()
    assert "obscure" in EnergyAdjuster("density").estimand()
    assert "average of all other energy sources" in EnergyAdjuster("residual").estimand()


def test_residual_lineage_reports_the_fitted_numbers():
    df = diet()
    step = EnergyAdjuster("residual", "energy_kcal", ["protein_g"]).fit(df)
    entry = next(e for e in step.lineage() if e["output"] == "protein_g_adj")
    assert entry["inputs"] == ["protein_g", "energy_kcal"]
    params = entry["params"]
    assert params["reference_energy"] == pytest.approx(df["energy_kcal"].mean())
    assert params["constant_added"] == pytest.approx(df["protein_g"].mean())
    assert abs(params["r_after"]) < 1e-8 and 0 < params["r2"] < 1
    assert params["r2"] == pytest.approx(params["r_before"] ** 2)
    assert step.dropped_columns_ == []  # total energy stays in the outcome model (ruling 1)
    dropped = EnergyAdjuster("residual_energy_dropped", "energy_kcal", ["protein_g"]).fit(df)
    assert dropped.dropped_columns_ == ["energy_kcal"]


def test_transform_refuses_a_frame_with_different_columns():
    df = diet()
    step = EnergyAdjuster("residual", "energy_kcal", ["fat_g"]).fit(df)
    with pytest.raises(ValueError, match="missing age"):
        step.transform(df.drop(columns="age"))
