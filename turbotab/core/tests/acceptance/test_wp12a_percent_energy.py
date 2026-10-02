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
           n_boot: int = 0) -> dict:
    paths = mf.ingest_frame(frame, folder)
    st = mf.state(roles=roles, target="y", models=["linear"], purpose="inference",
                  substitution=SubstitutionSpec(donor=donor, recipient=recipient,
                                                scale="percent_energy", step_percent=5.0,
                                                n_boot=n_boot))
    d.validate(d.SetSubstitution(donor=donor, recipient=recipient, scale="percent_energy",
                                 step_percent=5.0), {"state": st})
    split = mf.split_bundle(np.arange(len(frame)), holdout=0.0)
    ti = mf.target_info("regression", "y")
    design = design_stage(mf.context(st, {"split": split, "target_info": ti}, paths))
    fit = fit_stage(mf.context(st, {"design": design, "split": split, "target_info": ti}, paths))
    out = substitution_stage(mf.context(st, {"design": design, "fit": fit}, paths))
    SubstitutionArtifact.model_validate(out)
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
