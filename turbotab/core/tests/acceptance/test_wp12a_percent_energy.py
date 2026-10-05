"""WP12a · Substitution in percent of energy (AUDIT_REPORT §5 WP12; minors B24 and D19).

B24: "Exposures in % of energy cannot be substituted ('5% of energy from X replaced by Y')." D19:
"%E substitution and compositional models are not offered." §5 names the package's deliverable,
"%E substitution (B24/D19)", without a numbered test; the tests below hold it to the field's own
model and to statsmodels.

**Source check.** Hu et al. 1997, *N Engl J Med* 337:1491 (the abstract, read from PubMed 9366580
on 2026-10-02): "Mutivariate analyses included age, smoking status, total energy intake, dietary
cholesterol intake, percentages of energy obtained from protein and specific types of fat, and
other risk factors" and "Each increase of 5 percent of energy intake from saturated fat, as compared
with equivalent energy intake from carbohydrates, was associated with a 17 percent increase in the
risk of coronary disease". That is the leave-one-out model in percent of energy: every source but
carbohydrate, plus total energy, so each source's coefficient is the effect of taking its energy in
place of carbohydrate's. NUTRITION_PACK §05 names the figure: "Rows of the form '5% of energy from X
replaced by Y'".

**References**, none from the engine (``turbotab/core/methods/percent_energy.py`` and the
substitution stage):

* statsmodels OLS on the share-of-energy columns: the change is ``5 (β_Y − β_X)``, and statsmodels'
  leave-one-out fit (Hu's model) gives the same number as ``−5 γ_X``;
* through gram amounts, ``0.05 · mean(E) · (β_Y/4 − β_X/9)`` over the rows on support, the support
  counted by hand with numpy;
* the band's width against statsmodels' t interval for the same contrast.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats

from turbotab.core import decisions as d
from turbotab.core import voice
from turbotab.core.decisions import Refusal, SubstitutionSpec
from turbotab.core.methods.percent_energy import PercentEnergyShift, is_percent_of_energy
from turbotab.core.models.artifacts import SubstitutionArtifact
from turbotab.core.stages.modeling import design_stage, fit_stage, substitution_stage
from turbotab.core.tests import modeling_fixtures as mf

REPO = Path(__file__).resolve().parents[4]
SHARES = ["fat_pct_kcal", "carb_pct_kcal", "protein_pct_kcal"]


def _shares(n: int, seed: int) -> pd.DataFrame:
    """Fat, carbohydrate, protein and alcohol in percent of energy, summing to 100 on every row."""
    rng = np.random.default_rng(seed)
    energy = rng.normal(2100, 450, n).clip(900)
    fat = rng.normal(34, 6, n).clip(12, 55)
    protein = rng.normal(16, 3, n).clip(7, 30)
    alcohol = rng.gamma(1.2, 2.5, n).clip(0, 18)
    carb = 100 - fat - protein - alcohol
    age = rng.uniform(20, 80, n)
    y = (50 + 0.2 * age + 0.10 * fat - 0.05 * carb + 0.02 * protein + 0.004 * energy
         + rng.normal(0, 3, n))
    return pd.DataFrame({"age": age, "fat_pct_kcal": fat, "carb_pct_kcal": carb,
                         "protein_pct_kcal": protein, "alcohol_pct_kcal": alcohol,
                         "energy_kcal": energy, "y": y})


def _grams(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    energy = rng.normal(2100, 450, n).clip(900)
    fat = energy * 0.34 / 9 * rng.lognormal(0, 0.2, n)
    carb = energy * 0.48 / 4 * rng.lognormal(0, 0.15, n)
    protein = energy * 0.16 / 4 * rng.lognormal(0, 0.15, n)
    y = 5 + 0.03 * fat + 0.004 * carb + 0.001 * energy + rng.normal(0, 3, n)
    return pd.DataFrame({"fat_g": fat, "carb_g": carb, "protein_g": protein,
                         "energy_kcal": energy, "y": y})


def _stage(frame: pd.DataFrame, folder: Path, roles: dict, donor: str, recipient: str, *,
           n_boot: int = 0, holdout: float = 0.0, models: tuple[str, ...] = ("linear",)) -> dict:
    paths = mf.ingest_frame(frame, folder)
    st = mf.state(roles=roles, target="y", models=list(models), purpose="inference",
                  substitution=SubstitutionSpec(donor=donor, recipient=recipient,
                                                scale="percent_energy", step_percent=5.0,
                                                n_boot=n_boot),
                  column_units=mf.grams(*[c for c in frame.columns if c.endswith("_g")]))
    d.validate(d.SetSubstitution(donor=donor, recipient=recipient, scale="percent_energy",
                                 step_percent=5.0), {"state": st})
    split = mf.split_bundle(np.arange(len(frame)), holdout=holdout, seed=2)
    ti = mf.target_info("regression", "y")
    design = design_stage(mf.context(st, {"split": split, "target_info": ti}, paths))
    fit = fit_stage(mf.context(st, {"design": design, "split": split, "target_info": ti}, paths))
    out = substitution_stage(mf.context(st, {"design": design, "fit": fit}, paths))
    SubstitutionArtifact.model_validate(out)
    out["_fit"] = fit.data
    out["_training"] = split.frames["assignment"].query("partition == 'train'")["row_id"].to_numpy()
    return out


SHARE_ROLES = {"age": "covariate", **{c: "exposure" for c in SHARES}, "energy_kcal": "energy"}


def test_b24_five_percent_of_energy_from_fat_to_carbohydrate_equals_hus_leave_one_out_model(tmp_path):
    frame = _shares(2_000, seed=1)
    out = _stage(frame, tmp_path, SHARE_ROLES, "fat_pct_kcal", "carb_pct_kcal")
    model = out["models"][0]
    assert out["scale"] == "percent_energy" and out["ks"][:3] == [0.0, 5.0, 10.0]
    at5 = model["delta"][out["ks"].index(5.0)]

    columns = ["age", *SHARES, "energy_kcal"]  # alcohol is the source left out
    full = sm.OLS(frame["y"], sm.add_constant(frame[columns])).fit()
    expected = 5 * (full.params["carb_pct_kcal"] - full.params["fat_pct_kcal"])
    assert at5 == pytest.approx(expected, rel=1e-8)
    # Hu et al.'s model: every source but carbohydrate, plus total energy.
    loo = sm.OLS(frame["y"], sm.add_constant(
        frame[["age", "fat_pct_kcal", "protein_pct_kcal", "alcohol_pct_kcal", "energy_kcal"]])).fit()
    assert at5 == pytest.approx(-5 * loo.params["fat_pct_kcal"], rel=1e-8)
    # Linear in k, and labeled per 5% of energy.
    assert model["delta"][out["ks"].index(10.0)] == pytest.approx(2 * expected, rel=1e-8)
    assert model["effect_label"].endswith("per 5% of energy at k = 5")
    # Support at k = 5, counted by hand: each shifted share inside its observed range.
    fat, carb = frame["fat_pct_kcal"], frame["carb_pct_kcal"]
    on = ((fat - 5).between(fat.min(), fat.max()) & (fat - 5 >= 0)
          & (carb + 5).between(carb.min(), carb.max()))
    assert model["on_support_fraction"][1] == pytest.approx(on.mean(), abs=1e-12)
    assert "k percent of each row's own total energy" in out["estimand"]


def test_b24_through_gram_amounts_each_row_moves_five_percent_of_its_own_energy(tmp_path):
    frame = _grams(2_000, seed=2)
    roles = {"fat_g": "exposure", "carb_g": "exposure", "protein_g": "exposure",
             "energy_kcal": "energy"}
    out = _stage(frame, tmp_path, roles, "fat_g", "carb_g")
    model = out["models"][0]
    at5 = model["delta"][out["ks"].index(5.0)]
    fit = sm.OLS(frame["y"], sm.add_constant(frame[["fat_g", "carb_g", "protein_g", "energy_kcal"]])).fit()
    energy = frame["energy_kcal"].to_numpy()
    fat, carb = frame["fat_g"].to_numpy(), frame["carb_g"].to_numpy()
    moved = 0.05 * energy
    new_fat, new_carb = fat - moved / 9, carb + moved / 4
    amount = ((new_fat >= fat.min()) & (new_fat <= fat.max()) & (new_fat >= 0)
              & (new_carb >= carb.min()) & (new_carb <= carb.max()))
    fat_share, carb_share = 9 * fat / energy, 4 * carb / energy
    share = ((9 * new_fat / energy >= fat_share.min()) & (9 * new_fat / energy <= fat_share.max())
             & (4 * new_carb / energy >= carb_share.min()) & (4 * new_carb / energy <= carb_share.max()))
    on = amount & share
    assert model["on_support_fraction"][1] == pytest.approx(on.mean(), abs=1e-12)
    expected = 0.05 * energy[on].mean() * (fit.params["carb_g"] / 4 - fit.params["fat_g"] / 9)
    assert at5 == pytest.approx(expected, rel=1e-8)


def test_b24_under_inference_with_a_holdout_the_curve_reads_every_analyzed_row(tmp_path):
    """Repair round (verifier; BLUEPRINT §12 ruling 3): under inference the coefficient table is
    estimated from every analyzed row, and the substitution curve read the training rows only, so
    the two disagreed (the verifier's −0.311 against −0.271 per 5% of energy).

    With a 20% holdout drawn, both paths must equal statsmodels OLS on all rows, not on the
    training rows: the share path ``5 (β_carb − β_fat)``, and the gram path
    ``0.05 · mean(E) · (β_carb/4 − β_fat/9)`` over the rows on support, counted by hand."""
    frame = _shares(1_500, seed=97)
    out = _stage(frame, tmp_path / "shares", SHARE_ROLES, "fat_pct_kcal", "carb_pct_kcal",
                 holdout=0.2)
    at5 = out["models"][0]["delta"][out["ks"].index(5.0)]
    columns = ["age", *SHARES, "energy_kcal"]
    every = sm.OLS(frame["y"], sm.add_constant(frame[columns])).fit()
    train = frame.loc[out["_training"]]
    trained = sm.OLS(train["y"], sm.add_constant(train[columns])).fit()
    expected = 5 * (every.params["carb_pct_kcal"] - every.params["fat_pct_kcal"])
    assert len(train) == 1_200 and out["_fit"]["models"][0]["coefficients_n"] == 1_500
    assert at5 == pytest.approx(expected, rel=1e-8)
    assert abs(at5 - 5 * (trained.params["carb_pct_kcal"] - trained.params["fat_pct_kcal"])) > 1e-4
    assert out["basis"] == "Averaged over 1,500 analyzed rows."
    rows = {r["feature"]: r for r in out["_fit"]["models"][0]["coefficients"]}
    table = 5 * (rows["carb_pct_kcal"]["estimate"] - rows["fat_pct_kcal"]["estimate"])
    assert at5 == pytest.approx(table, rel=1e-8)  # the curve agrees with the coefficient table

    frame = _grams(1_500, seed=98)
    roles = {"fat_g": "exposure", "carb_g": "exposure", "protein_g": "exposure",
             "energy_kcal": "energy"}
    out = _stage(frame, tmp_path / "grams", roles, "fat_g", "carb_g", holdout=0.2)
    model = out["models"][0]
    fit = sm.OLS(frame["y"], sm.add_constant(frame[["fat_g", "carb_g", "protein_g", "energy_kcal"]])).fit()
    energy = frame["energy_kcal"].to_numpy()
    fat, carb = frame["fat_g"].to_numpy(), frame["carb_g"].to_numpy()
    new_fat, new_carb = fat - 0.05 * energy / 9, carb + 0.05 * energy / 4
    amount = ((new_fat >= fat.min()) & (new_fat <= fat.max()) & (new_fat >= 0)
              & (new_carb >= carb.min()) & (new_carb <= carb.max()))
    fat_share, carb_share = 9 * fat / energy, 4 * carb / energy
    share = ((9 * new_fat / energy >= fat_share.min()) & (9 * new_fat / energy <= fat_share.max())
             & (4 * new_carb / energy >= carb_share.min()) & (4 * new_carb / energy <= carb_share.max()))
    on = amount & share
    assert model["on_support_fraction"][1] == pytest.approx(on.mean(), abs=1e-12)
    expected = 0.05 * energy[on].mean() * (fit.params["carb_g"] / 4 - fit.params["fat_g"] / 9)
    assert model["delta"][out["ks"].index(5.0)] == pytest.approx(expected, rel=1e-8)


def test_b24_a_family_without_a_table_is_refit_on_every_analyzed_row_for_its_curve(tmp_path):
    """Under inference with a holdout, boosted trees (no coefficient table, so no every-row refit
    in the fit stage) are refit on every analyzed row when a curve is asked. Reference:
    scikit-learn's ``HistGradientBoostingRegressor(random_state=0)`` fit here on all 1,500 rows
    (the family's model; nothing in its pipeline transforms these columns), and its mean change in
    prediction over the rows on support when 5% of each row's energy moves from fat to
    carbohydrate, the support counted by hand."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    frame = _grams(1_500, seed=99)
    roles = {"fat_g": "exposure", "carb_g": "exposure", "protein_g": "exposure",
             "energy_kcal": "energy"}
    out = _stage(frame, tmp_path, roles, "fat_g", "carb_g", holdout=0.2,
                 models=("linear", "boosted_trees"))
    trees = next(m for m in out["models"] if m["family"] == "boosted_trees")
    columns = ["fat_g", "carb_g", "protein_g", "energy_kcal"]
    model = HistGradientBoostingRegressor(random_state=0).fit(frame[columns], frame["y"])
    energy = frame["energy_kcal"].to_numpy()
    fat, carb = frame["fat_g"].to_numpy(), frame["carb_g"].to_numpy()
    new_fat, new_carb = fat - 0.05 * energy / 9, carb + 0.05 * energy / 4
    amount = ((new_fat >= fat.min()) & (new_fat <= fat.max()) & (new_fat >= 0)
              & (new_carb >= carb.min()) & (new_carb <= carb.max()))
    fat_share, carb_share = 9 * fat / energy, 4 * carb / energy
    share = ((9 * new_fat / energy >= fat_share.min()) & (9 * new_fat / energy <= fat_share.max())
             & (4 * new_carb / energy >= carb_share.min()) & (4 * new_carb / energy <= carb_share.max()))
    on = amount & share
    moved = frame[columns].copy()
    moved["fat_g"], moved["carb_g"] = new_fat, new_carb
    expected = float(np.mean(model.predict(moved[on]) - model.predict(frame.loc[on, columns])))
    assert trees["delta"][out["ks"].index(5.0)] == pytest.approx(expected, rel=1e-6, abs=1e-9)
    assert out["basis"] == "Averaged over 1,500 analyzed rows."


def test_b24_shares_of_energy_are_energy_sources_to_the_router_the_validator_and_the_label(tmp_path):
    """Repair round (verifier): on a table of percent-of-energy columns (fat, carbohydrate and
    protein, alcohol left out) the substitution question was not applicable ("No exposure carries
    energy"), and posting it under inference was refused as "the model holds no energy source",
    with only a false attestation as the exit.

    References, none from the engine: the omitted share counted with numpy from the fixture's own
    columns (1 − Σ shares / 100, here alcohol's share, since the four sum to 100), against
    ``MAX_OMITTED_SHARE``; Hu et al.'s leave-one-out reading (the module docstring's source check)
    for the label."""
    from turbotab.core.datastore import DataStore
    from turbotab.core.interview import _substitution_applicability
    from turbotab.core.methods.energy import MAX_OMITTED_SHARE, omitted_energy
    from turbotab.core.stages.rows import energy_bearing

    frame = _shares(1_500, seed=97)
    roles = {**SHARE_ROLES, "alcohol_pct_kcal": "excluded"}
    st = mf.state(roles=roles, target="y", models=["linear"], purpose="inference")
    assert _substitution_applicability(st, energy_bearing) is None  # the question is asked

    alcohol = float(frame["alcohol_pct_kcal"].mean()) / 100
    assert alcohol < MAX_OMITTED_SHARE
    reading = omitted_energy(frame, "energy_kcal", SHARES)
    assert reading["columns"] == SHARES
    assert reading["sources"] == ["fat", "carbohydrate", "protein"]  # in the columns' order
    assert reading["omitted"] == ["alcohol", "other"]
    assert reading["mean_share"] == pytest.approx(alcohol, rel=1e-9)

    paths = mf.ingest_frame(frame, tmp_path)
    store = DataStore(Path(paths["data"]), 1 << 30)
    try:
        ctx = {"state": st, "store": lambda: store, "sealed": lambda: []}
        swap = {"kind": "set_substitution", "donor": "fat_pct_kcal", "recipient": "carb_pct_kcal",
                "scale": "percent_energy", "step_percent": 5.0}
        d.validate(swap, ctx)  # alcohol and other: under the stated share, accepted
        two = st.model_copy(update={"roles": {**roles, "protein_pct_kcal": "excluded"}})
        left = float((frame["protein_pct_kcal"] + frame["alcohol_pct_kcal"]).mean()) / 100
        with pytest.raises(Refusal) as refused:
            d.validate(swap, {**ctx, "state": two})
        said = str(refused.value)
        assert refused.value.code == "omitted_energy_sources"
        assert "the model holds `fat_pct_kcal` and `carb_pct_kcal`" in said
        assert "no energy source" not in said and f"{left:.0%}" in said
        labels = [e["label"] for e in refused.value.exits]
        # The four shares sum to 100%, so the exits add the missing ones but one, the reference
        # (the methods gate, item C; test_gate_methods_repair.py holds it to Hu's model).
        assert labels[:2] == [
            "Add `alcohol_pct_kcal` to the model, with `protein_pct_kcal` left out as the reference",
            "Add `protein_pct_kcal` to the model, with `alcohol_pct_kcal` left out as the reference"]
    finally:
        store.close()

    split = mf.split_bundle(np.arange(len(frame)), holdout=0.0)
    ti = mf.target_info("regression", "y")
    design = design_stage(mf.context(st, {"split": split, "target_info": ti}, paths))
    assert {"donor": "fat_pct_kcal", "recipient": "carb_pct_kcal"} in design.data["substitution_pairs"]
    assert not any("energy_kcal" in p.values() for p in design.data["substitution_pairs"])
    assert design.data["terms"]["fat_pct_kcal"] == ("one point of energy from fat in place of "
                                                     "alcohol and other energy, total energy fixed")
    assert "leave-one-out model" in design.data["estimand"]
    # Every share fixed, more total energy scales every source alike: not "energy from alcohol".
    assert design.data["terms"]["energy_kcal"] == "more total energy with each nutrient's share of it fixed"
    assert "in place of the energy sources not in the model: alcohol, other" in design.data["estimand"]


def test_b24_the_band_has_the_width_of_statsmodels_interval_for_the_contrast(tmp_path):
    """200 refits: a bootstrap SE carries about 5% Monte Carlo error, so ±15% is 3 such errors."""
    frame = _shares(2_000, seed=3)
    out = _stage(frame, tmp_path, SHARE_ROLES, "fat_pct_kcal", "carb_pct_kcal", n_boot=200)
    model = out["models"][0]
    i = out["ks"].index(5.0)
    half = (model["ci_high"][i] - model["ci_low"][i]) / 2
    columns = ["age", *SHARES, "energy_kcal"]
    full = sm.OLS(frame["y"], sm.add_constant(frame[columns])).fit()
    c = np.zeros(len(columns) + 1)
    c[1 + columns.index("carb_pct_kcal")], c[1 + columns.index("fat_pct_kcal")] = 5.0, -5.0
    se = float(np.sqrt(c @ full.cov_params().to_numpy() @ c))
    analytic = stats.t.ppf(0.975, full.df_resid) * se
    assert half == pytest.approx(analytic, rel=0.15)


def test_b24_the_swap_is_isocaloric_on_every_row():
    frame = _grams(500, seed=4)
    shift = PercentEnergyShift(frame, donor="fat_g", recipient="carb_g",
                               kcal_per_unit={"fat_g": 9.0, "carb_g": 4.0}, total="energy_kcal")
    new = shift.values(frame, 5.0)
    assert set(new) == {"fat_g", "carb_g"}  # total energy is not touched
    kcal_out = 9 * (frame["fat_g"].to_numpy() - new["fat_g"])
    kcal_in = 4 * (new["carb_g"] - frame["carb_g"].to_numpy())
    np.testing.assert_allclose(kcal_out, 0.05 * frame["energy_kcal"].to_numpy(), rtol=1e-12)
    np.testing.assert_allclose(kcal_in, kcal_out, rtol=1e-12)


def test_b24_shares_are_never_moved_in_kcal_and_fractions_are_refused():
    st = mf.state(roles=SHARE_ROLES, target="y")
    assert all(is_percent_of_energy(c) for c in SHARES)
    with pytest.raises(Refusal) as refused:
        d.validate(d.SetSubstitution(donor="fat_pct_kcal", recipient="carb_pct_kcal", step_kcal=100),
                   {"state": st})
    assert refused.value.code == "percent_of_energy"
    assert refused.value.exits[0]["decision"]["scale"] == "percent_energy"
    no_energy = mf.state(roles={"fat_g": "exposure", "carb_g": "exposure"}, target="y")
    with pytest.raises(Refusal) as refused:
        d.validate(d.SetSubstitution(donor="fat_g", recipient="carb_g", scale="percent_energy"),
                   {"state": no_energy})
    assert refused.value.code == "no_total_energy"
    fractions = _shares(100, seed=5)
    fractions[SHARES] = fractions[SHARES] / 100
    with pytest.raises(ValueError, match="fractions"):
        PercentEnergyShift(fractions, donor="fat_pct_kcal", recipient="carb_pct_kcal",
                           kcal_per_unit={}, percent=["fat_pct_kcal", "carb_pct_kcal"])


def test_b24_the_methods_sentence_states_the_share_and_the_pack_names_the_figure():
    text = voice.sentence_for(d.SetSubstitution(donor="fat_pct_kcal", recipient="carb_pct_kcal",
                                                scale="percent_energy", step_percent=5),
                              mf.state(roles=SHARE_ROLES, target="y"), None)
    assert "steps of `5`% of each participant's own total energy" in text
    pack = (REPO / "docs" / "turbotab" / "research" / "NUTRITION_PACK.md").read_text()
    assert '"5% of energy from X replaced by Y"' in " ".join(pack.split())
