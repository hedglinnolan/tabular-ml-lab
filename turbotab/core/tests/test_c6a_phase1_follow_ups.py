"""P1-FU · phase 1's cross-file follow-ups (WAVE_C6A_PLAN §3, the integrator's remaining issues on
feat/wave-c6a1-int).

1. The quest log's collinear line takes its label from the K5 noticing
   (``materiality.collinear_label``): Decide only when what you study is inside the dependency,
   For the record at band 0. The references are Belsley's proportions computed here by SVD.
4. Under the density and the energy-dropped residual forms, the methods line, the caption, the
   preview and Table 2 state the same measure: the unit each was written by hand from the
   energy model's specification (METHOD_TABLE: ``Y ~ (N/E) + C``; ``N_adj = residual +
   N_hat(mean E)``).
5. When the names suggest sugar is part of carbohydrate but the values reject it (sugar above
   carbohydrate on many rows), every surface says the swap Table 2 says: sugar in place of alcohol
   and other energy, total energy fixed, shown by a statsmodels identity on the reparameterized
   model.
7. ``card_label`` is documented among the family's members in MODEL_FAMILY_CONTRACT.md.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from turbotab.core import estimand as est
from turbotab.core import materiality as M
from turbotab.core import quest, voice
from turbotab.core.decisions import (CovariateAnswers, DecisionLog, EnergyAdjustment, EstimandSpec,
                                     ProjectState, SetAdjustment, SetEnergyAdjustment, SetEstimand,
                                     SetLens, SetPurpose, SetRoles, SetTarget, SetTask)
from turbotab.core.plan_previews import estimand_line
from turbotab.core.provenance import methods_text
from turbotab.core.tests.acceptance.test_qb_substitution_sentence import _diet
from turbotab.core.tests.acceptance.test_wp6_energy_estimands import _stages
from turbotab.core.tests.test_collinear_adjustment import ADJUST, hand_proportions, state, table

REPO = Path(__file__).resolve().parents[3]


# ── 1 · the collinear line's tier ────────────────────────────────────────────


def _sugar_share(f: pd.DataFrame, columns: list[str]) -> float:
    """Sugar's Belsley proportion summed over the near dependencies (η ≥ 30, two or more columns
    with π ≥ 0.5), by hand."""
    names, pi, eta = hand_proportions(f, ["sugar", *columns])
    near = [k for k in range(len(eta)) if eta[k] >= 30
            and sum(pi[j, k] >= 0.5 for j in range(1, len(names))) >= 2]
    assert near, "the drawn table holds a near dependency"
    return float(sum(pi[names.index("sugar"), k] for k in near))


def test_the_collinear_line_is_for_the_record_when_what_you_study_is_outside_it():
    f = table()
    assert _sugar_share(f, ADJUST) < 0.5  # outside the dependency: band 0, For the record
    s = state(ADJUST)
    place = quest.explore_place("collinear", s, noticed=M.collinear_noticing(s, f))
    assert (place.stage, place.item, place.label) == (
        "models", "noticing:explore::collinear", "For the record")


def test_the_collinear_line_is_decide_when_what_you_study_is_inside_it():
    f = table(sugar_inside=True)
    columns = [*ADJUST, "starch"]
    assert _sugar_share(f, columns) >= 0.5  # inside: which terms are held changes the meaning
    s = state(columns)
    assert quest.explore_place("collinear", s,
                               noticed=M.collinear_noticing(s, f)).label == "Decide"


def test_the_pair_scan_alone_and_the_other_findings_keep_their_places():
    s = state(ADJUST)
    # no near dependency measured: a pair of adjustment terms changes nothing here
    assert quest.explore_place("collinear", s, pairs=[("fat", "protein")]).label == "For the record"
    assert quest.explore_place("collinear", s, pairs=[("sugar", "carb")]).label == "Decide"
    # under Predict the pair scan's lever (a penalty in each fold) is a decision
    assert quest.explore_place("collinear", state(ADJUST, purpose="prediction"),
                               pairs=[("fat", "protein")]).label == "Decide"
    # every other finding sits where the crosswalk places it, as before
    assert quest.explore_place("wide", s) == quest.EXPLORE_FINDINGS["wide"]
    assert quest.explore_place("low_variance", s).label == "Confirm"


# ── 4 · one measure on every surface ─────────────────────────────────────────

ROLES = {"sugar_g": "exposure", "carb_g": "covariate", "protein_g": "covariate",
         "fat_g": "covariate", "kcal": "energy", "age": "covariate"}
NUTRIENTS = ["sugar_g", "carb_g", "protein_g", "fat_g"]
CONFOUNDER = CovariateAnswers(causes_exposure="yes", causes_outcome="yes", after_exposure="no")
# Written by hand from each form's specification: the density enters N/E, so the coefficient is
# per unit of sugar per unit of energy; the residual is per unit of sugar's residual on energy,
# re-centered at mean energy.
DENSITY = "per unit of `sugar_g` per unit of `kcal` (a density)"
RESIDUAL = "per unit of `sugar_g`'s energy-adjusted residual on `kcal`, at mean energy"
TABLE_2 = {"density": "sugar per unit of kcal (a density); total energy not in the model (obscure)",
           "residual_energy_dropped": ("sugar's energy-adjusted residual on kcal, at mean energy; "
                                       "total energy not in the outcome model")}
# Sugar beside its total (the values keep it inside carbohydrate): the same measure, then the swap.
TABLE_2_NESTED = {
    "density": "sugar_g per unit of kcal (a density), in place of the rest of carb_g (carb_g fixed)",
    "residual_energy_dropped": ("sugar_g's energy-adjusted residual on kcal, at mean energy, in "
                                "place of the rest of carb_g (carb_g fixed)")}


def _methods(tmp_path: Path, method: str) -> tuple[str, ProjectState]:
    """The methods line of the estimand, answered in the Router's order: the estimand first, then
    the adjustment set and the energy method."""
    log = DecisionLog(tmp_path / "decisions.jsonl")
    for d in (SetLens(lenses=["dietary"]), SetTarget(column="y"),
              SetTask(column="y", task="regression"), SetPurpose(purpose="inference"),
              SetRoles(roles=ROLES),
              SetEstimand(exposure="sugar_g", contrast="substitution", measure="mean_difference"),
              SetAdjustment(exposure="sugar_g",
                            answers={c: CONFOUNDER for c in ("carb_g", "protein_g", "fat_g", "age")}),
              SetEnergyAdjustment(method=method, energy_column="kcal", nutrients=NUTRIENTS)):
        log.append(d, sentence=lambda d, before: voice.sentence_for(d, before, None))
    [line] = [m.sentence for m in methods_text(log.records()).lines
              if "estimates the total effect" in m.sentence]
    return line, log.state()


# The preview names the measured quantity as what the effect is of, and keeps the substitution
# (verifier, P1-FU: the line with the measure and the contrast is too long for the caption's 20
# words, and the substitution must not be what gives way).
PREVIEW = {
    "density": ("Total effect of `sugar_g` per unit of `kcal` (a density) in place of other "
                "calories on `y`: mean difference."),
    "residual_energy_dropped": ("Total effect of `sugar_g`'s energy-adjusted residual on `kcal`, "
                                "at mean energy, in place of other calories on `y`: mean "
                                "difference.")}


@pytest.mark.parametrize("method,unit", [("density", DENSITY),
                                         ("residual_energy_dropped", RESIDUAL)])
def test_the_methods_line_the_caption_and_the_preview_state_one_measure(tmp_path, method, unit):
    line, st = _methods(tmp_path, method)
    assert unit in est.caption(st, "regression")
    assert unit in line, line
    assert "per unit of `sugar_g`." not in line and "per unit of `sugar_g`," not in line, line
    preview = estimand_line(st)
    assert preview == PREVIEW[method]
    assert len(preview.split()) <= 20
    assert unit.removeprefix("per unit of ") in preview


@pytest.mark.parametrize("method", ["standard", "residual", "none"])
def test_the_other_energy_answers_keep_the_methods_line_they_had(tmp_path, method):
    """Only the density and the energy-dropped residual change the methods line's unit (P1-FU's
    remit): under the other answers it still reads per unit of the column, as it did."""
    line, _st = _methods(tmp_path, method)
    assert line.endswith("as a difference in the mean outcome per unit of `sugar_g`."), line


@pytest.mark.parametrize("method", ["density", "residual_energy_dropped"])
def test_table_2_states_that_measure(tmp_path, method):
    frame = _diet(seed=5)
    alone = {c: r for c, r in ROLES.items() if c != "carb_g"}
    run = _stages(frame.drop(columns=["carb_g"]), tmp_path / method, alone, EnergyAdjustment(
        method=method, energy_column="kcal", nutrients=["sugar_g", "protein_g", "fat_g"]))
    [row] = [r for f, r in run["rows"].items() if f.startswith("sugar_g")]
    assert row["meaning"] == TABLE_2[method]
    run = _stages(frame, tmp_path / f"{method}_nested", ROLES, EnergyAdjustment(
        method=method, energy_column="kcal", nutrients=NUTRIENTS))
    [row] = [r for f, r in run["rows"].items() if f.startswith("sugar_g")]
    assert row["meaning"] == TABLE_2_NESTED[method]


# ── 5 · the values reject the nesting the names suggest ──────────────────────

# Written by hand: protein, carbohydrate and fat are in the model, sugar a source of its own beside
# them (the values say it is no part of carbohydrate), so 1 g more sugar at fixed total energy
# displaces the energy from no named source: alcohol and any other.
REJECTED = "1 g more sugar in place of alcohol and other energy (total energy fixed)"
NAMES_GUESS = "1 g more sugar in place of other carbohydrate (total carbohydrate and energy fixed)"


def _rejected(n: int = 400, seed: int = 3) -> pd.DataFrame:
    """Sugar drawn apart from carbohydrate, above it on many rows (recorded on another basis than
    the names say)."""
    rng = np.random.default_rng(seed)
    kcal = rng.normal(2100, 400, n)
    carb = 0.12 * kcal + rng.normal(0, 25, n)
    sugar = rng.gamma(4, 60, n)
    protein = 0.04 * kcal + rng.normal(0, 8, n)
    fat = 0.035 * kcal + rng.normal(0, 7, n)
    age = rng.normal(50, 12, n)
    y = 0.02 * sugar + 0.01 * protein + 0.001 * kcal + 0.05 * age + rng.normal(0, 1, n)
    return pd.DataFrame({"sugar_g": sugar, "carb_g": carb, "protein_g": protein, "fat_g": fat,
                         "kcal": kcal, "age": age, "y": y})


def _state() -> ProjectState:
    return ProjectState(
        lens=["dietary"], target="y", task="regression", purpose="inference", roles=ROLES,
        estimand=EstimandSpec(exposure="sugar_g", contrast="substitution",
                              measure="mean_difference"),
        energy_adjustment=EnergyAdjustment(method="standard", energy_column="kcal",
                                           nutrients=NUTRIENTS))


def test_when_the_values_reject_the_nesting_every_surface_says_table_2s_swap(tmp_path):
    frame = _rejected()
    # the values reject it: sugar is above carbohydrate on far more than 1% of rows
    assert float((frame["sugar_g"] > frame["carb_g"]).mean()) > 0.2
    # the reference: rest = kcal − 4·sugar − 4·carb − 4·protein − 9·fat is the energy from no named
    # source; sugar's coefficient is 1 g more sugar and 4 kcal less of that rest
    X = ["sugar_g", "carb_g", "protein_g", "fat_g", "kcal", "age"]
    original = sm.OLS(frame["y"], sm.add_constant(frame[X])).fit()
    rest = (frame["kcal"] - 4 * frame["sugar_g"] - 4 * frame["carb_g"] - 4 * frame["protein_g"]
            - 9 * frame["fat_g"])
    reparam = sm.OLS(frame["y"], sm.add_constant(frame.assign(rest=rest)[
        ["sugar_g", "carb_g", "protein_g", "fat_g", "rest", "age"]])).fit()
    swap = reparam.params["sugar_g"] - 4 * reparam.params["rest"]
    assert swap == pytest.approx(original.params["sugar_g"], abs=1e-10)

    run = _stages(frame, tmp_path / "rejected", ROLES, EnergyAdjustment(
        method="standard", energy_column="kcal", nutrients=NUTRIENTS))
    row = run["rows"]["sugar_g"]
    assert row["estimate"] == pytest.approx(swap, rel=1e-8)
    assert row["meaning"] == REJECTED.removeprefix("1 g more ")

    st = _state()
    nested = est.values_nesting(st, frame=frame)
    assert nested == {}
    decision = SetEstimand(exposure="sugar_g", contrast="substitution", measure="mean_difference")
    [sub] = [c for c in est.estimand_card(st, "regression", nested=nested)["contrasts"]
             if c["contrast"] == "substitution"]
    said = {"caption": est.caption(st, "regression", nested=nested),
            "voice": voice.sentence_for(decision, st, {"frame": frame}),
            "preview": estimand_line(st, nested=nested),
            "card": sub["consequence"]}
    for where, text in said.items():
        assert REJECTED in text, (where, text)
        assert "other carbohydrate" not in text, (where, text)
    assert said["card"] == REJECTED + "."
    # The served fit's caption reads the values it is given, as the server passes them.
    served = est.annotate_fit({"task": "regression", "models": []}, st, nested=nested)
    assert REJECTED in served["estimand"]["caption"]


def test_where_the_values_agree_or_are_not_read_the_names_guess_stands():
    st = _state()
    frame = _diet(seed=5)  # a part never above its total
    nested = est.values_nesting(st, frame=frame)
    assert nested == {"sugar_g": "carb_g"}
    assert NAMES_GUESS in est.caption(st, "regression", nested=nested)
    assert NAMES_GUESS in voice.sentence_for(
        SetEstimand(exposure="sugar_g", contrast="substitution", measure="mean_difference"), st,
        {"frame": frame})
    # no table at hand: the names alone speak
    assert est.values_nesting(st) is None
    assert NAMES_GUESS in est.caption(st, "regression")


# ── 7 · card_label is documented ─────────────────────────────────────────────


def test_the_contract_lists_card_label_beside_label():
    text = (REPO / "docs" / "turbotab-next" / "MODEL_FAMILY_CONTRACT.md").read_text("utf-8")
    start = text.index("### How it is declared")
    block = text[start:text.index("### C1 · Identity")]
    [line] = [l for l in block.splitlines() if l.strip().startswith("card_label:")]
    assert "card" in line.split("#", 1)[1] and "methods register" in line, line
