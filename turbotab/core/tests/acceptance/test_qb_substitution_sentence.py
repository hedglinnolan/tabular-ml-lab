"""Q-b (WAVE_C6A_PLAN §3, §5 (b)): the substitution a nutrient's coefficient estimates is worded
from the adjustment set, one way, on the caption, the methods sentence, the preview, the card and
Table 2.

With total carbohydrate and energy in the model, sugar's coefficient is 1 g more sugar in place
of other carbohydrate, total carbohydrate and energy fixed. Without carbohydrate it is a swap for
the energy sources not in the model, named (ME-04). Each wording is checked against a statsmodels
identity on the reparameterized model, never against the app's own output:

* with carbohydrate held, ``other = carb − sugar`` gives β′_sugar − β′_other = β_sugar;
* without it, ``rest = kcal − 4·sugar − 4·protein − 9·fat`` (the energy from the sources not in the
  model) gives β″_sugar − 4·β″_rest = β_sugar: 1 g more sugar and 4 kcal less of the rest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from turbotab.core import estimand as est
from turbotab.core import voice
from turbotab.core.decisions import EnergyAdjustment, EstimandSpec, ProjectState, SetEstimand
from turbotab.core.plan_previews import estimand_line
from turbotab.core.tests.acceptance.test_wp6_energy_estimands import _stages

WITH_CARB = "1 g more sugar in place of other carbohydrate (total carbohydrate and energy fixed)"
# Written by hand: protein, carbohydrate (sugar is carbohydrate) and fat are in the model, so
# alcohol is the main source left out, with the carbohydrate that is not sugar and any other energy.
WITHOUT_CARB = ("1 g more sugar in place of other carbohydrate, alcohol and other energy "
                "(total energy fixed)")


def _diet(n: int = 400, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    kcal = rng.normal(2100, 400, n)
    carb = 0.12 * kcal + rng.normal(0, 25, n)
    sugar = carb * rng.uniform(0.15, 0.45, n)  # a part never above its total
    protein = 0.04 * kcal + rng.normal(0, 8, n)
    fat = 0.035 * kcal + rng.normal(0, 7, n)
    age = rng.normal(50, 12, n)
    y = 0.02 * sugar - 0.004 * (carb - sugar) + 0.01 * protein + 0.001 * kcal + 0.05 * age \
        + rng.normal(0, 1, n)
    return pd.DataFrame({"sugar_g": sugar, "carb_g": carb, "protein_g": protein, "fat_g": fat,
                         "kcal": kcal, "age": age, "y": y})


def _ols(frame: pd.DataFrame, columns: list[str]):
    return sm.OLS(frame["y"].to_numpy(float), sm.add_constant(frame[columns].astype(float))).fit()


def _state(roles: dict[str, str], nutrients: list[str]) -> ProjectState:
    return ProjectState(
        lens=["dietary"], target="y", task="regression", purpose="inference", roles=roles,
        estimand=EstimandSpec(exposure="sugar_g", contrast="substitution", measure="mean_difference"),
        energy_adjustment=EnergyAdjustment(method="standard", energy_column="kcal", nutrients=nutrients))


def _surfaces(st: ProjectState) -> dict[str, str]:
    """The four sentences that say what the estimate is: the caption, the methods sentence, the
    preview's line and the card's consequence for the substitution."""
    decision = SetEstimand(exposure="sugar_g", contrast="substitution", measure="mean_difference")
    card = est.estimand_card(st, "regression")
    [sub] = [c for c in card["contrasts"] if c["contrast"] == "substitution"]
    return {"caption": est.caption(st, "regression"),
            "voice": voice.sentence_for(decision, st),
            "preview": estimand_line(st),
            "card": sub["consequence"]}


def test_with_carbohydrate_and_energy_held_sugar_is_a_swap_for_other_carbohydrate(tmp_path):
    frame = _diet()
    original = _ols(frame, ["sugar_g", "carb_g", "protein_g", "fat_g", "kcal", "age"])
    swapped = frame.assign(other_g=frame["carb_g"] - frame["sugar_g"])
    reparam = _ols(swapped, ["sugar_g", "other_g", "protein_g", "fat_g", "kcal", "age"])
    swap = reparam.params["sugar_g"] - reparam.params["other_g"]
    # the reference: the identity holds whatever the app says
    assert swap == pytest.approx(original.params["sugar_g"], abs=1e-10)

    roles = {"sugar_g": "exposure", "carb_g": "covariate", "protein_g": "covariate",
             "fat_g": "covariate", "kcal": "energy", "age": "covariate"}
    nutrients = ["sugar_g", "carb_g", "protein_g", "fat_g"]
    run = _stages(frame, tmp_path / "with", roles, EnergyAdjustment(
        method="standard", energy_column="kcal", nutrients=nutrients))
    row = run["rows"]["sugar_g"]
    assert row["estimate"] == pytest.approx(swap, rel=1e-8)
    # Table 2 says the same swap (the amount is the per-unit row's own)
    assert row["meaning"] == WITH_CARB.removeprefix("1 g more ")

    said = _surfaces(_state(roles, nutrients))
    for where, text in said.items():
        assert WITH_CARB in text, (where, text)
        assert "other energy sources at fixed total energy" not in text, (where, text)
        assert "other calories" not in text, (where, text)
    assert said["card"] == WITH_CARB + "."


def test_without_carbohydrate_the_swap_names_the_sources_left_out(tmp_path):
    frame = _diet(seed=11)
    original = _ols(frame, ["sugar_g", "protein_g", "fat_g", "kcal", "age"])
    rest = frame["kcal"] - 4 * frame["sugar_g"] - 4 * frame["protein_g"] - 9 * frame["fat_g"]
    reparam = _ols(frame.assign(rest=rest), ["sugar_g", "protein_g", "fat_g", "rest", "age"])
    swap = reparam.params["sugar_g"] - 4 * reparam.params["rest"]
    assert swap == pytest.approx(original.params["sugar_g"], abs=1e-10)

    roles = {"sugar_g": "exposure", "protein_g": "covariate", "fat_g": "covariate",
             "kcal": "energy", "age": "covariate"}
    nutrients = ["sugar_g", "protein_g", "fat_g"]
    run = _stages(frame, tmp_path / "without", roles, EnergyAdjustment(
        method="standard", energy_column="kcal", nutrients=nutrients))
    row = run["rows"]["sugar_g"]
    assert row["estimate"] == pytest.approx(swap, rel=1e-8)
    assert row["meaning"] == WITHOUT_CARB.removeprefix("1 g more ")

    said = _surfaces(_state(roles, nutrients))
    for where, text in said.items():
        # The preview's line keeps the measure within its caption words; the "1 g" gives way
        # there, as on the card (the swap itself is whole).
        assert (WITHOUT_CARB.removeprefix("1 g ") if where == "preview" else WITHOUT_CARB) in text, \
            (where, text)
        assert "other carbohydrate (total carbohydrate" not in text, (where, text)


def test_the_preview_line_keeps_the_swap_whole_within_its_caption_words():
    from turbotab.core.consequences import CAPTION_WORDS, words

    roles = {"sugar_g": "exposure", "carb_g": "covariate", "protein_g": "covariate",
             "fat_g": "covariate", "kcal": "energy", "age": "covariate"}
    line = estimand_line(_state(roles, ["sugar_g", "carb_g", "protein_g", "fat_g"]))
    assert words(line) <= CAPTION_WORDS and WITH_CARB in line, line
