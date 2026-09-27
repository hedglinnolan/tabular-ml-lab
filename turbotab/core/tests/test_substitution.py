"""Tier A: the substitution curve (PRODUCT_VISION §06c), checked against known answers."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from turbotab.core.methods.energy import EnergyAdjuster
from turbotab.core.methods.substitution import substitution_curve

KCAL = {"fat_kcal": 1.0, "carb_kcal": 1.0}
GRAMS = {"fat_g": 9.0, "carbohydrate_g": 4.0, "protein_g": 4.0}


def composition(n: int = 500, seed: int = 0) -> pd.DataFrame:
    """Macronutrient energy in kcal, with total energy the sum plus other sources."""
    rng = np.random.default_rng(seed)
    fat = rng.gamma(9.0, 70.0, n)
    carb = rng.gamma(12.0, 80.0, n)
    protein = rng.gamma(10.0, 35.0, n)
    other = rng.gamma(2.0, 40.0, n)
    energy = fat + carb + protein + other
    age = rng.integers(20, 80, n).astype(float)
    y = 1.0 + 0.004 * fat - 0.002 * carb + 0.001 * protein + 0.02 * age + rng.normal(0, 0.5, n)
    return pd.DataFrame({"fat_kcal": fat, "carb_kcal": carb, "protein_kcal": protein,
                         "energy_kcal": energy, "age": age, "y": y})


def in_grams(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"fat_g": df["fat_kcal"] / 9.0, "carbohydrate_g": df["carb_kcal"] / 4.0,
                         "protein_g": df["protein_kcal"] / 4.0, "energy_kcal": df["energy_kcal"],
                         "age": df["age"]})


def manual_mask(donor: np.ndarray, recipient: np.ndarray, k_donor: float, k_recipient: float):
    shifted_d, shifted_r = donor - k_donor, recipient + k_recipient
    return ((shifted_d >= donor.min()) & (shifted_d <= donor.max()) & (shifted_r >= recipient.min())
            & (shifted_r <= recipient.max()) & (shifted_d >= 0) & (shifted_r >= 0))


KS = [0, 25, 50, 100, 150, 200, 300, 400, 600, 800]


# ── known answers ────────────────────────────────────────────────────────────

def test_a_linear_model_in_kcal_moves_by_exactly_k_times_the_coefficient_gap():
    df = composition()
    X = df.drop(columns="y")
    model = LinearRegression().fit(X, df["y"])
    b = dict(zip(X.columns, model.coef_))
    curve = substitution_curve(model.predict, X, donor="fat_kcal", recipient="carb_kcal",
                               kcal_per_unit=KCAL, ks=KS)
    assert curve["delta"][0] == 0.0
    for k, delta in zip(curve["ks"], curve["delta"]):
        if delta is not None:
            assert delta == pytest.approx(k * (b["carb_kcal"] - b["fat_kcal"]), rel=1e-9, abs=1e-12)
    assert any(d is not None for d in curve["delta"][1:])


def test_a_linear_model_in_grams_converts_through_kcal_per_unit():
    df = composition(seed=1)
    X = in_grams(df)
    model = LinearRegression().fit(X, df["y"])
    b = dict(zip(X.columns, model.coef_))
    curve = substitution_curve(model.predict, X, donor="fat_g", recipient="carbohydrate_g",
                               kcal_per_unit=GRAMS, ks=KS)
    for k, delta in zip(curve["ks"], curve["delta"]):
        if delta is not None:
            assert delta == pytest.approx(k * (b["carbohydrate_g"] / 4.0 - b["fat_g"] / 9.0), rel=1e-9)
    assert curve["units_moved"] == {"fat_g": -1 / 9.0, "carbohydrate_g": 1 / 4.0}


def test_support_shrinks_with_k_and_the_curve_stops_below_min_support():
    df = composition(seed=2)
    X = df.drop(columns="y")
    model = LinearRegression().fit(X, df["y"])
    curve = substitution_curve(model.predict, X, donor="fat_kcal", recipient="carb_kcal",
                               kcal_per_unit=KCAL, ks=KS, min_support=0.5)
    fractions = curve["on_support_fraction"]
    assert fractions[0] == 1.0
    assert all(a >= b for a, b in zip(fractions, fractions[1:]))
    fat, carb = X["fat_kcal"].to_numpy(), X["carb_kcal"].to_numpy()
    expected = [manual_mask(fat, carb, k, k).mean() for k in curve["ks"]]
    np.testing.assert_allclose(fractions, expected, rtol=0, atol=0)
    first_below = next(k for k, f in zip(curve["ks"], expected) if f < 0.5)
    assert curve["stopped_at"] == first_below
    for k, delta in zip(curve["ks"], curve["delta"]):
        assert (delta is None) == (k >= first_below)
    assert f"stopped at k = {first_below:g}" in curve["note"]


def test_rows_that_would_go_negative_are_masked_even_inside_the_observed_range():
    # Row 0 carries an impossible -5 g, which widens the observed minimum; row 1 holds
    # 1 g of fat, so moving 18 kcal (2 g) would take it to -1 g: inside [min, max], negative.
    X = pd.DataFrame({"fat_g": [-5.0, 1.0, 40.0, 60.0, 80.0, 90.0],
                      "carbohydrate_g": [200.0, 210.0, 220.0, 230.0, 240.0, 300.0]})
    seen = []

    def predict(frame: pd.DataFrame) -> np.ndarray:
        seen.append(frame.copy())
        return frame["fat_g"].to_numpy() * 0.0 + frame["carbohydrate_g"].to_numpy() * 0.01

    curve = substitution_curve(predict, X, donor="fat_g", recipient="carbohydrate_g",
                               kcal_per_unit=GRAMS, ks=[0, 18], min_support=0.0)
    shifted_at_18 = seen[-1]
    assert 1 not in shifted_at_18.index and 0 not in shifted_at_18.index
    assert (shifted_at_18["fat_g"] >= 0).all()
    assert 0 not in seen[0].index  # a row negative before any shift is never averaged
    # At k = 18 the recipient gains 4.5 g: row 5 (300 g) leaves the observed max.
    assert curve["n_on_support"] == [5, 3]
    assert curve["on_support_fraction"] == [5 / 6, 3 / 6]
    assert curve["delta"][1] == pytest.approx(4.5 * 0.01)


def test_rows_with_missing_donor_or_recipient_are_off_support_at_every_k():
    df = composition(seed=3)
    X = df.drop(columns="y")
    model = LinearRegression().fit(X, df["y"])
    X.loc[X.index[:50], "fat_kcal"] = np.nan
    curve = substitution_curve(lambda f: model.predict(f.fillna(0.0)), X, donor="fat_kcal",
                               recipient="carb_kcal", kcal_per_unit=KCAL, ks=[0, 50])
    assert curve["on_support_fraction"][0] == pytest.approx(450 / 500)


def test_a_tree_gives_a_piecewise_constant_curve_with_a_stated_k():
    df = composition(seed=4)
    X = df.drop(columns="y")
    tree = DecisionTreeRegressor(max_depth=1, random_state=0).fit(X[["fat_kcal"]], df["y"])
    threshold = tree.tree_.threshold[0]
    left, right = tree.tree_.value[1][0][0], tree.tree_.value[2][0][0]

    def predict(frame):
        return tree.predict(frame[["fat_kcal"]])

    ks = np.arange(0, 301, 5)
    curve = substitution_curve(predict, X, donor="fat_kcal", recipient="carb_kcal",
                               kcal_per_unit=KCAL, ks=ks, min_support=0.0)
    fat, carb = X["fat_kcal"].to_numpy(), X["carb_kcal"].to_numpy()
    before = np.where(fat <= threshold, left, right)
    for k, delta, count in zip(curve["ks"], curve["delta"], curve["n_on_support"]):
        mask = manual_mask(fat, carb, k, k)
        after = np.where(fat - k <= threshold, left, right)
        assert count == mask.sum()
        assert delta == pytest.approx(np.mean((after - before)[mask]), rel=0, abs=1e-12)
        # Piecewise constant: the average moves only in whole steps of (left - right),
        # one per row that crosses the split.
        crossings = delta * count / (left - right)
        assert crossings == pytest.approx(round(crossings), abs=1e-9)
    assert len(set(curve["delta"])) < len(ks)  # it has flat stretches

    assert curve["label_k"] == 100.0
    delta_100 = curve["delta"][curve["ks"].index(100.0)]
    assert curve["effect_label"].endswith("per 100 kcal at k = 100")
    assert curve["effect_label"].startswith(("+", "-", "0"))
    assert float(curve["effect_label"].split(" ")[0]) == pytest.approx(delta_100, rel=5e-3)


def test_the_label_scales_to_100_kcal_and_names_the_k_it_was_measured_at():
    df = composition(seed=5)
    X = df.drop(columns="y")
    model = LinearRegression().fit(X, df["y"])
    gap = model.coef_[list(X.columns).index("carb_kcal")] - model.coef_[list(X.columns).index("fat_kcal")]
    curve = substitution_curve(model.predict, X, donor="fat_kcal", recipient="carb_kcal",
                               kcal_per_unit=KCAL, ks=[0, 50, 150], label_k=50)
    assert curve["effect_label"].endswith("per 100 kcal at k = 50")
    assert float(curve["effect_label"].split(" ")[0]) == pytest.approx(100 * gap, rel=5e-3)
    default = substitution_curve(model.predict, X, donor="fat_kcal", recipient="carb_kcal",
                                 kcal_per_unit=KCAL, ks=[0, 50, 150])
    assert default["label_k"] == 50.0  # nearest to 100, ties to the smaller k


# ── the band ─────────────────────────────────────────────────────────────────

def test_the_bootstrap_band_collapses_for_a_linear_model_and_is_reproducible():
    df = composition(seed=6)
    X = df.drop(columns="y")
    model = LinearRegression().fit(X, df["y"])
    kwargs = dict(donor="fat_kcal", recipient="carb_kcal", kcal_per_unit=KCAL, ks=KS, n_boot=100)
    curve = substitution_curve(model.predict, X, **kwargs)
    for delta, low, high in zip(curve["delta"], curve["ci_low"], curve["ci_high"]):
        if delta is None:
            assert low is None and high is None
        else:  # every row changes by the same k * gap, so resampling rows changes nothing
            assert low == pytest.approx(delta, abs=1e-9) and high == pytest.approx(delta, abs=1e-9)
    assert substitution_curve(model.predict, X, **kwargs)["ci_low"] == curve["ci_low"]
    assert "fitted model held fixed" in curve["note"]


def test_the_bootstrap_band_brackets_a_tree_curve():
    df = composition(seed=7)
    X = df.drop(columns="y")
    tree = DecisionTreeRegressor(max_depth=3, random_state=0).fit(X, df["y"])
    curve = substitution_curve(tree.predict, X, donor="fat_kcal", recipient="carb_kcal",
                               kcal_per_unit=KCAL, ks=KS, n_boot=200, random_state=3)
    widths = []
    for delta, low, high in zip(curve["delta"], curve["ci_low"], curve["ci_high"]):
        if delta is not None:
            assert low <= delta + 1e-12 and delta <= high + 1e-12
            widths.append(high - low)
    assert max(widths) > 0
    assert curve["ci_low"] is not None and substitution_curve(
        tree.predict, X, donor="fat_kcal", recipient="carb_kcal", kcal_per_unit=KCAL,
        ks=KS)["ci_low"] is None


# ── through the energy-adjustment pipeline ──────────────────────────────────

def test_through_a_residual_pipeline_the_curve_is_the_adjusted_coefficient_gap():
    df = composition(seed=8)
    X = in_grams(df)
    nutrients = ["protein_g", "fat_g", "carbohydrate_g"]
    pipe = Pipeline([("energy", EnergyAdjuster("residual", "energy_kcal", nutrients)),
                     ("model", LinearRegression())]).fit(X, df["y"])
    c = dict(zip(pipe[:-1].get_feature_names_out(), pipe[-1].coef_))
    curve = substitution_curve(pipe.predict, X, donor="fat_g", recipient="carbohydrate_g",
                               kcal_per_unit=GRAMS, ks=KS)
    # Energy is unchanged, so each adjusted nutrient moves exactly as its raw value does.
    for k, delta in zip(curve["ks"], curve["delta"]):
        if delta is not None:
            assert delta == pytest.approx(k * (c["carbohydrate_g_adj"] / 4 - c["fat_g_adj"] / 9), rel=1e-9)


def test_through_a_partition_pipeline_the_curve_is_the_kcal_coefficient_gap():
    df = composition(seed=9)
    X = in_grams(df)
    pipe = Pipeline([("energy", EnergyAdjuster("partition", "energy_kcal", ["fat_g", "carbohydrate_g"])),
                     ("model", LinearRegression())]).fit(X, df["y"])
    c = dict(zip(pipe[:-1].get_feature_names_out(), pipe[-1].coef_))
    curve = substitution_curve(pipe.predict, X, donor="fat_g", recipient="carbohydrate_g",
                               kcal_per_unit=GRAMS, ks=KS)
    for k, delta in zip(curve["ks"], curve["delta"]):
        if delta is not None:
            assert delta == pytest.approx(k * (c["kcal_from_carbohydrate_g"] - c["kcal_from_fat_g"]), rel=1e-9)


# ── what it says ─────────────────────────────────────────────────────────────

def test_the_note_says_how_the_total_was_treated_and_where_the_floor_comes_from():
    df = composition(seed=10)
    X = df.drop(columns="y")
    model = LinearRegression().fit(X, df["y"])
    common = dict(donor="fat_kcal", recipient="carb_kcal", kcal_per_unit=KCAL, ks=[0, 100])
    variable = substitution_curve(model.predict, X, **common)
    fixed = substitution_curve(model.predict, X, total_kind="fixed", **common)
    assert variable["total_kind"] == "variable" and "held fixed by assumption" in variable["note"]
    assert fixed["total_kind"] == "fixed" and "held fixed by assumption" not in fixed["note"]
    assert "fixed budget" in fixed["note"]
    for curve in (variable, fixed):
        assert "practitioner convention, not a sourced threshold" in curve["note"]
        assert set(curve) >= {"donor", "recipient", "ks", "delta", "on_support_fraction", "stopped_at",
                              "ci_low", "ci_high", "total_kind", "effect_label", "note"}
        json.dumps(curve, allow_nan=False)  # the server sends it as it is


@pytest.mark.parametrize("kwargs, message", [
    (dict(ks=[-100, 0]), "swap donor and recipient"),
    (dict(kcal_per_unit={"fat_kcal": 1.0}), "no factor for carb_kcal"),
    (dict(recipient="fat_kcal"), "must be different"),
    (dict(total_kind="elastic"), "total_kind"),
    (dict(min_support=1.5), "min_support"),
])
def test_it_refuses_what_it_cannot_compute(kwargs, message):
    X = composition(n=50).drop(columns="y")
    args = dict(donor="fat_kcal", recipient="carb_kcal", kcal_per_unit=KCAL, ks=[0, 100])
    args.update(kwargs)
    with pytest.raises(ValueError, match=message):
        substitution_curve(lambda f: np.zeros(len(f)), X, **args)
