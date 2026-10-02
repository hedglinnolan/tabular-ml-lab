"""Tier A: the substitution curve (PRODUCT_VISION §06c), its nested parts and its refit band,
checked against known answers."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from turbotab.core.methods.energy import EnergyAdjuster
from turbotab.core.methods.substitution import Shift, refit_band, substitution_curve

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


# ── nested parts (M1_CONTRACT §12.5) ─────────────────────────────────────────

FAT = {"fat_total": 9.0, "fat_sat": 9.0, "fat_mon": 9.0, "fat_poly": 9.0, "carb": 4.0}
PARTS = ["fat_sat", "fat_mon", "fat_poly"]


def fats(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Total fat in grams with its three parts and an unlisted rest, and carbohydrate."""
    rng = np.random.default_rng(seed)
    total = rng.gamma(8.0, 9.0, n)
    shares = rng.dirichlet([3.0, 3.5, 2.0, 1.0], n)  # sat, mon, poly, the rest
    frame = pd.DataFrame({"fat_total": total, "carb": rng.gamma(10.0, 25.0, n),
                          "age": rng.integers(20, 80, n).astype(float)})
    for j, part in enumerate(PARTS):
        frame[part] = total * shares[:, j]
    return frame


def test_the_parts_of_a_total_are_found_from_their_names_and_their_values():
    from turbotab.core.methods.nesting import nested_components

    X = fats()
    assert nested_components(X) == {p: "fat_total" for p in PARTS}
    broken = X.assign(fat_sat=X["fat_total"] * 1.5)  # a "part" larger than its total is not one
    assert "fat_sat" not in nested_components(broken)


@pytest.mark.parametrize("k", [0.0, 45.0, 180.0])
def test_moving_a_total_moves_its_parts_in_proportion_and_they_still_sum_to_it(k):
    X = fats(seed=1)
    nested = {p: "fat_total" for p in PARTS}
    rest = X["fat_total"] - X[PARTS].sum(axis=1)
    for donor, recipient, sign in (("fat_total", "carb", -1.0), ("carb", "fat_total", 1.0)):
        shift = Shift(X, donor=donor, recipient=recipient, kcal_per_unit=FAT, nested=nested)
        new = shift.values(X, k)
        total = X["fat_total"].to_numpy() + sign * k / 9.0
        np.testing.assert_allclose(new["fat_total"], total, rtol=0, atol=1e-12)
        ratio = total / X["fat_total"].to_numpy()
        for part in PARTS:  # each part keeps its share of the total, row by row
            np.testing.assert_allclose(new[part] / new["fat_total"], X[part] / X["fat_total"], rtol=1e-12)
            np.testing.assert_allclose(new[part], X[part].to_numpy() * ratio, rtol=1e-12)
        # The parts and the unlisted rest, scaled alike, still add up to the total on every row.
        np.testing.assert_allclose(sum(new[p] for p in PARTS) + rest.to_numpy() * ratio,
                                   new["fat_total"], rtol=0, atol=1e-9)
        assert set(shift.carried) == set(PARTS)


def test_moving_a_part_carries_its_total_by_the_same_amount_and_leaves_the_other_parts():
    X = fats(seed=2)
    nested = {p: "fat_total" for p in PARTS}
    shift = Shift(X, donor="fat_sat", recipient="carb", kcal_per_unit=FAT, nested=nested)
    new = shift.values(X, 90.0)
    np.testing.assert_allclose(new["fat_sat"], X["fat_sat"] - 10.0)
    np.testing.assert_allclose(new["fat_total"], X["fat_total"] - 10.0)
    assert set(new) == {"fat_sat", "carb", "fat_total"}  # fat_mon and fat_poly hold
    assert shift.note() == ("On every row, fat_sat is part of fat_total, so fat_total moved with it "
                            "by the same amount while its other parts held.")


def test_a_total_and_its_own_part_are_never_a_substitution():
    X = fats()
    with pytest.raises(ValueError, match="fat_sat is part of fat_total"):
        Shift(X, donor="fat_total", recipient="fat_sat", kcal_per_unit=FAT,
              nested={p: "fat_total" for p in PARTS})


def test_through_a_linear_model_the_curve_is_the_coefficients_times_every_column_that_moved():
    X = fats(seed=3)
    rng = np.random.default_rng(3)
    y = 1 + 0.02 * X["fat_sat"] - 0.01 * X["fat_poly"] + 0.004 * X["carb"] + rng.normal(0, 0.3, len(X))
    model = LinearRegression().fit(X, y)
    b = dict(zip(X.columns, model.coef_))
    nested = {p: "fat_total" for p in PARTS}
    ks = [0.0, 30.0, 90.0]
    curve = substitution_curve(model.predict, X, donor="fat_total", recipient="carb",
                               kcal_per_unit=FAT, ks=ks, nested=nested)
    lo = {c: X[c].min() for c in X.columns}
    hi = {c: X[c].max() for c in X.columns}
    for k, delta in zip(curve["ks"], curve["delta"]):
        d_total = -k / 9.0
        ratio = (X["fat_total"] + d_total) / X["fat_total"]
        moved = {"fat_total": X["fat_total"] + d_total, "carb": X["carb"] + k / 4.0,
                 **{p: X[p] * ratio for p in PARTS}}
        on = np.ones(len(X), dtype=bool)
        for c, v in moved.items():
            on &= (v >= lo[c]).to_numpy() & (v <= hi[c]).to_numpy() & (v >= 0).to_numpy()
        change = sum(b[c] * (moved[c] - X[c]) for c in moved)
        assert delta == pytest.approx(float(change[on].mean()), abs=1e-12)
    assert "fat_sat, fat_mon and fat_poly are parts of fat_total and moved with it" in curve["note"]


# ── the band: refits, never one fitted model held fixed (M1_CONTRACT §12.7) ──

def _refit(X: pd.DataFrame, y: np.ndarray):
    return LinearRegression().fit(X, y).predict


def test_the_refit_band_has_width_for_a_linear_model_and_holds_the_curve():
    df = composition(seed=6)
    X, y = df.drop(columns="y"), df["y"].to_numpy()
    model = LinearRegression().fit(X, y)
    curve = substitution_curve(model.predict, X, donor="fat_kcal", recipient="carb_kcal",
                               kcal_per_unit=KCAL, ks=KS)
    shift = Shift(X, donor="fat_kcal", recipient="carb_kcal", kcal_per_unit=KCAL)
    band = refit_band(_refit, X, y, shift=shift, ks=KS, live=curve["live"], n_boot=60,
                      center=curve["delta"])
    for k, delta, low, high in zip(curve["ks"], curve["delta"], band["ci_low"], band["ci_high"]):
        if delta is None:
            assert low is None and high is None
        elif k == 0:
            assert low == pytest.approx(0.0, abs=1e-12) and high == pytest.approx(0.0, abs=1e-12)
        else:  # refitting moves the coefficient gap, so the band opens with k
            assert low < delta < high
    again = refit_band(_refit, X, y, shift=shift, ks=KS, live=curve["live"], n_boot=60,
                       center=curve["delta"])
    assert again["ci_low"] == band["ci_low"] and band["n_ok"] == 60 and band["failed"] == 0


def test_the_refit_band_is_the_percentile_or_normal_interval_of_the_refits_coefficient_gaps():
    df = composition(n=300, seed=11)
    X, y = df.drop(columns="y"), df["y"].to_numpy()
    shift = Shift(X, donor="fat_kcal", recipient="carb_kcal", kcal_per_unit=KCAL)
    ks = [0.0, 10.0]  # small enough that every row stays on support
    rng = np.random.default_rng(5)
    gaps = []
    for _ in range(40):  # the same resamples, drawn by hand
        idx = rng.integers(0, len(X), size=len(X))
        coef = dict(zip(X.columns, LinearRegression().fit(X.iloc[idx], y[idx]).coef_))
        gaps.append(10.0 * (coef["carb_kcal"] - coef["fat_kcal"]))
    band = refit_band(_refit, X, y, shift=shift, ks=ks, live=[True, True], n_boot=40, random_state=5,
                      interval="percentile")
    assert band["interval"] == "percentile" and band["scale"] == 1.0
    assert band["ci_low"][1] == pytest.approx(np.percentile(gaps, 2.5), rel=1e-9)
    assert band["ci_high"][1] == pytest.approx(np.percentile(gaps, 97.5), rel=1e-9)
    # 40 refits are too few for percentile endpoints, so "auto" gives the normal interval.
    auto = refit_band(_refit, X, y, shift=shift, ks=ks, live=[True, True], n_boot=40, random_state=5)
    half = 1.959963984540054 * np.std(gaps, ddof=1)
    assert auto["interval"] == "normal"
    assert auto["ci_low"][1] == pytest.approx(np.mean(gaps) - half, rel=1e-9)
    assert auto["ci_high"][1] == pytest.approx(np.mean(gaps) + half, rel=1e-9)


def test_the_refit_band_stops_when_its_progress_says_stop():
    df = composition(seed=12)
    X, y = df.drop(columns="y"), df["y"].to_numpy()
    shift = Shift(X, donor="fat_kcal", recipient="carb_kcal", kcal_per_unit=KCAL)
    fits = []

    def fit(Xb, yb):
        fits.append(len(Xb))
        return _refit(Xb, yb)

    class Stop(Exception):
        pass

    def progress(done, total):
        if done == 3:
            raise Stop()

    with pytest.raises(Stop):
        refit_band(fit, X, y, shift=shift, ks=[0, 100], live=[True, True], n_boot=50,
                   progress=progress)
    assert len(fits) == 3


def test_a_grouped_band_resamples_whole_units():
    df = composition(n=120, seed=13)
    X, y = df.drop(columns="y"), df["y"].to_numpy()
    groups = np.repeat(np.arange(40), 3)
    shift = Shift(X, donor="fat_kcal", recipient="carb_kcal", kcal_per_unit=KCAL)
    seen = []

    def fit(Xb, yb):
        seen.append(Xb.index.to_numpy())
        return _refit(Xb, yb)

    refit_band(fit, X, y, shift=shift, ks=[0, 50], live=[True, True], n_boot=5, groups=groups)
    for idx in seen:  # every unit drawn comes with all three of its rows
        counts = pd.Series(groups[idx]).value_counts()
        assert (counts % 3 == 0).all()


def test_a_row_with_no_unit_recorded_is_a_unit_of_its_own_in_the_band():
    df = composition(n=60, seed=14)
    X, y = df.drop(columns="y"), df["y"].to_numpy()
    groups = np.repeat(np.arange(20), 3).astype(object)
    groups[-6:] = None  # six rows whose identifier is missing
    shift = Shift(X, donor="fat_kcal", recipient="carb_kcal", kcal_per_unit=KCAL)
    drawn = []

    def fit(Xb, yb):
        drawn.append(Xb.index.to_numpy())
        return _refit(Xb, yb)

    band = refit_band(fit, X, y, shift=shift, ks=[0, 50], live=[True, True], n_boot=40, groups=groups)
    assert band["n_units"] == 18 + 6  # 18 whole units, and each blank row alone
    rows = np.concatenate(drawn)
    assert set(range(54, 60)) <= set(rows.tolist())  # the blank rows are resampled, one at a time
    for idx in drawn:  # a recorded unit's three rows still come together (rows 3u, 3u+1, 3u+2)
        counts = pd.Series(idx[idx < 54] // 3).value_counts()
        assert (counts % 3 == 0).all()
        assert len(idx) == 3 * int((idx < 54).sum() // 3) + int((idx >= 54).sum())


def test_a_band_from_fewer_units_than_there_are_is_rescaled_to_all_of_them():
    df = composition(n=400, seed=15)
    X, y = df.drop(columns="y"), df["y"].to_numpy()
    model = LinearRegression().fit(X, y)
    curve = substitution_curve(model.predict, X, donor="fat_kcal", recipient="carb_kcal",
                               kcal_per_unit=KCAL, ks=[0, 50])
    shift = Shift(X, donor="fat_kcal", recipient="carb_kcal", kcal_per_unit=KCAL)
    groups = np.repeat(np.arange(100), 4)
    sizes = []

    def fit(Xb, yb):
        sizes.append(len(Xb))
        return _refit(Xb, yb)

    band = refit_band(fit, X, y, shift=shift, ks=[0, 50], live=[True, True], n_boot=10,
                      groups=groups, max_rows=100, center=curve["delta"])
    # 100 rows of 400 is a quarter: 25 of the 100 units, and the spread times sqrt(25 / 100)
    assert band["n_units"] == 100 and band["resample_size"] == 25 and band["scale"] == 0.5
    assert set(sizes) == {100}
    full = refit_band(_refit, X, y, shift=shift, ks=[0, 50], live=[True, True], n_boot=10,
                      groups=groups, max_rows=400)
    assert full["resample_size"] == 100 and full["scale"] == 1.0
    with pytest.raises(ValueError, match="needs the curve's own deltas"):
        refit_band(_refit, X, y, shift=shift, ks=[0, 50], live=[True, True], n_boot=10, max_rows=100)
    with pytest.raises(ValueError, match="not both"):
        refit_band(_refit, X, y, shift=shift, ks=[0, 50], live=[True, True], n_boot=10,
                   max_rows=100, resample_size=50, center=curve["delta"])


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
                              "live", "total_kind", "effect_label", "note"}
        assert "ci_low" not in curve  # no band through one fitted model
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
