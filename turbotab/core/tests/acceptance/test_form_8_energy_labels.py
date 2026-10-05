"""FORM · 8 · what a form does to the energy estimands' labels (MODELING_SEQUENCE §2, corrected;
§0 ruling 11).

* A spline or categories on an energy residual is the nutrient's curve on the energy-adjusted scale
  at mean energy, not the substitution curve; its label says so and spline(N) + E is offered as the
  route to the substitution curve (the server-path assertion is in
  ``test_form_1_order_and_invalidation``; the label's words are the review's).
* A log or a spline on an energy component: the substitution curve still moves k kcal, but each
  point is an average over the rows of each person's effect at that k (k-specific), never a
  coefficient difference (Draft 1's "ratio rather than difference" was wrong).
* A request to reallocate shares (compositional, isometric log-ratio) is refused with that reason
  (v2.x), the kcal substitution and the share-of-energy swap offered instead.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from turbotab.core import decisions as d
from turbotab.core.decisions import EnergyAdjustment, SubstitutionSpec
from turbotab.core.methods import exposure_form as ef
from turbotab.core.tests import modeling_fixtures as mf

ROLES = {"fat_g": "exposure", "carb_g": "exposure", "protein_g": "exposure",
         "energy_kcal": "energy"}
LOG_LABEL = ("With a log of `carb_g` and `fat_g`, moving k kcal no longer has one effect per kcal: "
             "each person's effect depends on k and on their own intake, so each point of the "
             "curve is an average of those effects over the rows the curve reads, at that k "
             "(k-specific), not a coefficient difference; an all-components contrast is not "
             "defined on logged components")
SPLINE_LABEL = ("With a nonlinear form of `fat_g`, moving k kcal no longer has one effect per "
                "kcal: each person's effect depends on k and on their own intake, so each point "
                "of the curve is an average of those effects over the rows the curve reads, at "
                "that k (k-specific), not a coefficient difference")


def _grams(n: int = 500, seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    energy = rng.normal(2100, 450, n).clip(900)
    fat = energy * 0.34 / 9 * rng.lognormal(0, 0.2, n)
    carb = energy * 0.48 / 4 * rng.lognormal(0, 0.15, n)
    protein = energy * 0.16 / 4 * rng.lognormal(0, 0.15, n)
    y = 5 + 0.03 * fat + 0.004 * carb + 0.001 * energy + rng.normal(0, 3, n)
    return pd.DataFrame({"fat_g": fat, "carb_g": carb, "protein_g": protein,
                         "energy_kcal": energy, "y": y})


def _curve(folder: Path, **slots) -> dict:
    from turbotab.core.stages.modeling import design_stage, fit_stage, substitution_stage

    frame = _grams()
    paths = mf.ingest_frame(frame, folder)
    st = mf.state(roles=dict(ROLES), role_confirmations=dict(ROLES), target="y",
                  models=["linear"], purpose="prediction",
                  substitution=SubstitutionSpec(donor="carb_g", recipient="fat_g", step_kcal=100),
                  column_units=mf.grams("fat_g", "carb_g", "protein_g"), **slots)
    split = mf.split_bundle(np.arange(len(frame)), holdout=0.0, seed=2)
    ti = mf.target_info("regression", "y")
    design = design_stage(mf.context(st, {"split": split, "target_info": ti}, paths))
    fit = fit_stage(mf.context(st, {"design": design, "split": split, "target_info": ti}, paths))
    return substitution_stage(mf.context(st, {"design": design, "fit": fit}, paths))


def test_8_a_log_on_an_energy_component_labels_the_curve_a_k_specific_average(tmp_path):
    adj = EnergyAdjustment(method="residual", energy_column="energy_kcal",
                           nutrients=["fat_g", "carb_g", "protein_g"], log_transform=True)
    sub = _curve(tmp_path, energy_adjustment=adj)
    assert sub["curve_label"] == LOG_LABEL


def test_8_a_spline_on_an_energy_component_labels_the_curve_a_k_specific_average(tmp_path):
    adj = EnergyAdjustment(method="standard", energy_column="energy_kcal",
                           nutrients=["fat_g", "carb_g", "protein_g"])
    sub = _curve(tmp_path, energy_adjustment=adj,
                 exposure_forms={"fat_g": d.ExposureFormSpec(form="spline", knots=4)})
    assert sub["curve_label"] == SPLINE_LABEL
    # a model linear in both moved components keeps its coefficient-difference reading
    linear = mf.state(energy_adjustment=adj)
    assert ef.substitution_curve_label(linear, "carb_g", "fat_g") is None


def test_8_a_residual_curves_label_and_route_are_the_reviews_words():
    adj = EnergyAdjustment(method="residual", energy_column="energy_kcal",
                           nutrients=["fat_g", "carb_g"])
    st = mf.state(roles=dict(ROLES), purpose="inference", energy_adjustment=adj,
                  exposure_forms={"fat_g": d.ExposureFormSpec(form="categories",
                                                              cuts=[60.0, 80.0])})
    assert ef.residual_form(st, "fat_g")  # categories too
    assert ef.residual_label("fat_g", "energy_kcal") == (
        "the curve of `fat_g` on the energy-adjusted scale at mean energy (its residual on "
        "`energy_kcal`), not the substitution curve at fixed total energy: the residual model "
        "equals the standard model only for a straight line; a spline of `fat_g` with "
        "`energy_kcal` beside it (the standard model) is the route to the substitution curve")
    assert not ef.residual_form(mf.state(roles=dict(ROLES), energy_adjustment=adj), "fat_g")


def test_8_a_share_reallocation_is_refused_with_the_ilr_reason():
    st = mf.state(roles=dict(ROLES), purpose="inference")
    with pytest.raises(d.Refusal) as refused:
        d.validate({"kind": "set_substitution", "donor": "carb_g", "recipient": "fat_g",
                    "scale": "share_reallocation"}, {"state": st})
    assert refused.value.code == "share_reallocation_v2x"
    assert refused.value.message == (
        "Reallocating shares of a composition (an isometric log-ratio model of the diet, Leite "
        "2016) is not in this version: it goes to v2.x (MODELING_SEQUENCE §0 ruling 11). The kcal "
        "substitution moves energy from one source to another at fixed total energy; the "
        "share-of-energy swap moves a percentage of each person's energy.")
    exits = refused.value.exits
    assert [e["label"] for e in exits] == ["The kcal substitution", "The share-of-energy swap"]
    assert exits[0]["decision"]["scale"] == "kcal"
    assert exits[1]["decision"]["scale"] == "percent_energy"
