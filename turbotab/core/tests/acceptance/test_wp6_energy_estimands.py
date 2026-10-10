"""WP6 · Energy-model estimands: the label equals the model fitted (docs/turbotab-next/audit/
AUDIT_REPORT.md §5).

Closes ME-02 ("no energy adjustment" fitted the standard model and labeled it absolute intake),
ME-03 (the residual method dropped total energy but claimed the standard model's estimand), ME-04
(with several energy-bearing nutrients each coefficient is a swap for the sources left out, labeled
a swap for the average of all others), ME-05 (substitution curves never checked that every energy
source is in the model), ME-14 (the all-components model was not offered and its relative effect
never computed) and ME-15 (a total beside its own parts), with the minors B19 (fiber beside
carbohydrate by difference) and D14/G17 (the log residual's estimand and methods sentence). One
test, or a few, per acceptance test in §5, numbered as there.

BLUEPRINT §12, rulings 1 and 2, bind the defaults: the residual method keeps total energy in the
outcome model and the energy-dropped form is offered under its own estimand; under inference the
all-components model ranks first for substitution questions, and under prediction the app says the
choice matters little.

Every reference is computed by a path independent of the code under test: statsmodels' OLS (or
numpy's least squares) on model matrices built by hand from the fixture's own columns, linear
contrasts of those coefficients, HC3 written out from its definition (``references.py``), the
data-generating coefficients (the truths), and primary sources quoted in the docstrings. Each
measured number comes through the app's own path: ingest, the design and fit stages (and the
substitution stage, the decision validators, the preview card and the methods sentence).

Fixtures are the audit's (``repro.tar.gz``): ``D-skeptic/s1_none.py`` (test 1), ``B/exp1_residual.py``
and ``D/exp2_residual_vs_standard.py`` (tests 2 and 3), ``B/exp14_loo.py`` (test 4),
``D/exp3_substitution_composite.py`` (test 5), ``D/exp1_energy_models.py`` (test 6) and the NHANES
export with ``D-skeptic/s7_nest.py`` (test 7), each regenerated here from its seed.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats

from turbotab.core import voice
from turbotab.core.consequences import PreviewContext, plan
from turbotab.core.datastore import DataStore
from turbotab.core.decisions import (
    EnergyAdjustment,
    ProjectState,
    Refusal,
    SetEnergyAdjustment,
    SetRoles,
    SplitSpec,
    SubstitutionSpec,
    parse_decision,
    validate,
)
from turbotab.core.methods.energy import (
    LOG_ESTIMAND,
    MAX_OMITTED_SHARE,
    METHOD_TABLE,
    TENSION,
    applicable_methods,
    describe_model,
    rank_methods,
)
from turbotab.core.models.pipeline import design_spec, model_predictors, shared_steps, transformer
from turbotab.core.stages.modeling import design_stage, fit_stage, substitution_stage
from turbotab.core.stages.proposals import build_proposals
from turbotab.core.teaching import entry
from turbotab.core.tests import modeling_fixtures as mf
from turbotab.core.tests.acceptance import references as ref


def _nhanes_export() -> Path | None:
    """The real NHANES export: the tracked fixture, decompressed (``stage_harness.NHANES``)."""
    from turbotab.core.tests.stage_harness import NHANES

    return NHANES if NHANES.is_file() else None


NHANES = _nhanes_export()


# ── the app's own path ───────────────────────────────────────────────────────


def _stages(frame: pd.DataFrame, folder: Path, roles: dict[str, str], adjustment: Any, *,
            purpose: str = "inference", target: str = "y",
            substitution: SubstitutionSpec | None = None) -> dict[str, Any]:
    """Ingest ``frame`` and run the design and fit stages (and the substitution stage when asked),
    with the linear family, every row a training row. Returns the artifacts and what made them."""
    paths = mf.ingest_frame(frame, folder)
    st = mf.state(roles=roles, target=target, task="regression", models=["linear"],
                  energy_adjustment=adjustment, purpose=purpose, substitution=substitution,
                  split=SplitSpec(holdout=0.0, seed=0, folds=5))
    split = mf.split_bundle(np.arange(len(frame)), holdout=0.0)
    ti = mf.target_info("regression", target)
    design = design_stage(mf.context(st, {"split": split, "target_info": ti}, paths))
    fit = fit_stage(mf.context(st, {"design": design, "split": split, "target_info": ti}, paths))
    out = {"design": design.data, "fit": fit.data, "paths": paths, "state": st,
           "rows": {r["feature"]: r for r in fit.data["models"][0]["coefficients"]}}
    if substitution is not None:
        out["substitution"] = substitution_stage(
            mf.context(st, {"design": design, "fit": fit}, paths))
    return out


def _matrix_columns(design: dict[str, Any]) -> set[str]:
    return {n["column"] for n in design["lineage"]["nodes"] if n["lane"] == "matrix"}


def _ols(y: np.ndarray, exog: pd.DataFrame) -> Any:
    """statsmodels OLS with an intercept: the reference fit."""
    return sm.OLS(np.asarray(y, dtype=float), sm.add_constant(exog.astype(float))).fit()


def _shown(value: float) -> str:
    """A coefficient as the card and the design warning print it: three significant digits."""
    return f"{value:+.3g}".replace("-", "−")


# ── 1 · "None" ───────────────────────────────────────────────────────────────


def _none_fixture() -> pd.DataFrame:
    """D-skeptic/s1_none.py: n = 5,000; protein ~16% of energy; y = 0.03·protein + 0.002·E +
    0.1·age + noise, so energy confounds protein."""
    rng = np.random.default_rng(3)
    n = 5000
    E = rng.normal(2100, 450, n)
    age = rng.normal(50, 12, n)
    protein = (0.16 + rng.normal(0, .03, n)) * E / 4
    y = 0.03 * protein + 0.002 * E + 0.1 * age + rng.normal(0, 5, n)
    return pd.DataFrame({"protein_g": protein, "energy_kcal": E, "age": age, "y": y})


NONE_ROLES = {"protein_g": "exposure", "energy_kcal": "energy", "age": "covariate"}


def test_1_none_fits_the_truly_unadjusted_model(tmp_path):
    """§5 WP6.1: "The model matrix excludes the energy-role column; on the D fixture the protein
    coefficient is 0.0561 (the truly unadjusted model), not 0.0299."

    Reference: statsmodels OLS of y on protein_g and age (``Y ~ N + C``), 0.0561, and on protein_g,
    energy_kcal and age (the standard model), 0.0299, both on the fixture's own columns. Measured:
    the inference coefficient table the fit stage serves under "No energy adjustment", recorded as
    the interface records it (``method="none"``, no energy column named: the energy role alone says
    which column is total energy). The design's estimand, its energy form and the methods sentence
    say what was fitted.
    """
    frame = _none_fixture()
    unadjusted = _ols(frame["y"], frame[["protein_g", "age"]]).params["protein_g"]
    standard = _ols(frame["y"], frame[["protein_g", "energy_kcal", "age"]]).params["protein_g"]
    assert unadjusted == pytest.approx(0.0561, abs=5e-5)  # the audit's numbers, reproduced
    assert standard == pytest.approx(0.0299, abs=5e-5)

    adjustment = EnergyAdjustment(method="none")
    out = _stages(frame, tmp_path, NONE_ROLES, adjustment)
    assert "energy_kcal" not in _matrix_columns(out["design"])
    assert "energy_kcal" not in out["rows"]
    assert out["rows"]["protein_g"]["estimate"] == pytest.approx(unadjusted, rel=1e-9)
    assert abs(out["rows"]["protein_g"]["estimate"] - standard) > 0.02

    assert out["design"]["energy_form"] == "none"
    assert "total energy is not in the model" in out["design"]["estimand"]
    assert out["rows"]["protein_g"]["meaning"] == ("absolute intake of protein; total energy not in "
                                                   "the model")
    said = voice.sentence_for(SetEnergyAdjustment(method="none"), out["state"])
    assert "`energy_kcal` was left out of the models" in said


@pytest.mark.parametrize("role", ["covariate", "exposure"])
def test_1c_total_energy_kept_by_another_role_reads_as_the_standard_model(tmp_path, role):
    """Repair round (verifier, ME-02 residual path): with `energy_kcal`'s role changed from the
    proposed "energy" to "covariate", the energy question does not apply, the model is the
    standard one (protein 0.0299), and the app labeled it "Not energy-adjusted … absolute intake".
    ME-02's recommendation: "build the estimand sentence from the fitted matrix, not from the
    method's name".

    Reference: statsmodels OLS on the fixture's own columns, with and without `energy_kcal` (the
    roles say which columns the model holds). Measured: the design and fit stages with no energy
    answer recorded, as the interview leaves it. The coefficient is the standard model's, and the
    estimand, the row meaning and a design warning say so; leaving the column out ("excluded")
    gives the unadjusted 0.0561 with its absolute-intake label."""
    frame = _none_fixture()
    standard = _ols(frame["y"], frame[["protein_g", "energy_kcal", "age"]]).params["protein_g"]
    unadjusted = _ols(frame["y"], frame[["protein_g", "age"]]).params["protein_g"]

    kept = _stages(frame, tmp_path / "kept", {**NONE_ROLES, "energy_kcal": role}, None)
    assert "energy_kcal" in _matrix_columns(kept["design"])
    assert kept["rows"]["protein_g"]["estimate"] == pytest.approx(standard, rel=1e-9)
    assert kept["design"]["energy_form"] == "standard"
    assert "absolute intake" not in kept["design"]["estimand"]
    assert kept["design"]["estimand"] == METHOD_TABLE["standard"]["estimand"]
    assert kept["rows"]["protein_g"]["meaning"] == (
        "protein in place of the average of all other energy sources (total energy fixed)")
    assert any(w.startswith("energy_kcal reads as total energy and is in the model as")
               and "(the standard model), not an absolute intake" in w
               for w in kept["design"]["warnings"])

    left = _stages(frame, tmp_path / "left", {**NONE_ROLES, "energy_kcal": "excluded"}, None)
    assert "energy_kcal" not in _matrix_columns(left["design"])
    assert left["rows"]["protein_g"]["estimate"] == pytest.approx(unadjusted, rel=1e-9)
    assert left["design"]["energy_form"] == "none"
    assert left["rows"]["protein_g"]["meaning"] == ("absolute intake of protein; total energy not in "
                                                    "the model")


def test_1d_an_expenditure_or_a_macronutrient_is_not_read_as_total_energy():
    """The recognizer behind 1c reads total energy intake only: an energy expenditure, a basal
    rate, a macronutrient's own kcal or share, and a partition's term are not total energy."""
    from turbotab.core.methods.energy import reads_as_total_energy

    for name in ("energy_kcal", "kcal", "DR1TKCAL", "total_energy", "energy_kj", "calories"):
        assert reads_as_total_energy(name), name
    for name in ("energy_expenditure_kcal", "tee_kcal", "bmr_kcal", "protein_kcal", "fat_pct_kcal",
                 "kcal_from_fat", "energy_requirement", "kcal_goal", "age"):
        assert not reads_as_total_energy(name), name


def _matrix(frame: pd.DataFrame, roles: dict[str, str], adjustment: Any, **slots: Any) -> pd.DataFrame:
    """The model matrix the models see: the state's predictors through the shared steps."""
    st = ProjectState(target="y", roles=roles, energy_adjustment=adjustment, **slots)
    predictors = model_predictors(st)
    spec = design_spec(st, frame, predictors)
    return transformer(shared_steps(spec)).fit_transform(frame[spec.inputs])


def _mixed(seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 400
    E = rng.normal(2000, 400, n)
    frame = pd.DataFrame({
        "protein_g": 0.04 * E + rng.normal(0, 10, n), "fat_g": 0.035 * E + rng.normal(0, 8, n),
        "energy_kcal": E, "energy_kj": E * 4.184, "age": rng.uniform(20, 80, n),
        "sex": rng.choice(["F", "M"], n), "sodium_mg": rng.normal(3000, 600, n),
        "y": rng.normal(0, 1, n)})
    frame.loc[rng.random(n) < 0.05, "fat_g"] = np.nan
    return frame


CASES = {
    "the energy column named": (
        {"protein_g": "exposure", "energy_kcal": "energy", "age": "covariate"},
        EnergyAdjustment(method="none", energy_column="energy_kcal"), {}),
    "the energy role alone": (
        {"protein_g": "exposure", "energy_kcal": "energy", "age": "covariate"},
        EnergyAdjustment(method="none"), {}),
    "two energy columns": (
        {"protein_g": "exposure", "energy_kcal": "energy", "energy_kj": "energy"},
        EnergyAdjustment(method="none"), {}),
    "imputed, one-hot and a nutrient without energy": (
        {"protein_g": "exposure", "fat_g": "exposure", "sodium_mg": "exposure",
         "energy_kcal": "energy", "sex": "covariate"},
        EnergyAdjustment(method="none"), {"missing": "impute"}),
}


@pytest.mark.parametrize("case", list(CASES))
def test_1b_the_none_and_standard_matrices_differ_whenever_an_energy_column_exists(case):
    """§5 WP6.1: "a test asserts the "none" and "standard" matrices differ whenever an energy
    column exists."

    For each way a table can carry total energy (the column named on the answer, the energy role
    alone, two energy-role columns, and beside imputation, one-hot encoding and a nutrient that
    carries no energy), the matrix built under "none" holds no energy-role column, the standard
    matrix holds every one, and the rest is the same. The estimand read off each matrix
    (``describe_model``) names the model it is: "none" and "standard". Reference: the roles
    themselves (which columns are total energy), not the code under test.
    """
    roles, none, slots = CASES[case]
    frame = _mixed()
    energy = {c for c, r in roles.items() if r == "energy"}
    nutrients = [c for c, r in roles.items() if r == "exposure" and c != "sodium_mg"]
    standard = EnergyAdjustment(method="standard", energy_column="energy_kcal", nutrients=nutrients)
    m_none = _matrix(frame, roles, none, **slots)
    m_standard = _matrix(frame, roles, standard, **slots)
    assert energy and not energy & set(m_none.columns)
    assert energy <= set(m_standard.columns)
    assert set(m_standard.columns) - set(m_none.columns) == energy
    assert list(m_none.columns) != list(m_standard.columns)
    predictors = [c for c, r in roles.items() if r in ("exposure", "covariate", "energy")]
    assert describe_model(none, predictors, roles, list(m_none.columns)).form == "none"
    assert describe_model(standard, predictors, roles, list(m_standard.columns)).form == "standard"


# ── 2 · Residual ─────────────────────────────────────────────────────────────


def _covariates_fixture() -> pd.DataFrame:
    """B/exp1_residual.py: n = 4,000; sex, age and physical activity all move total energy, and
    the outcome depends on fat, energy and all three."""
    rng = np.random.default_rng(1)
    n = 4000
    male = rng.integers(0, 2, n)
    age = rng.normal(50, 12, n)
    pa = rng.normal(0, 1, n)
    logE = np.log(1700) + 0.25 * male + 0.10 * pa - 0.004 * (age - 50) + rng.normal(0, 0.22, n)
    E = np.exp(logE)
    fat_share = np.clip(rng.normal(0.34, 0.06, n), 0.1, 0.6)
    fat_g = fat_share * E / 9
    y = 0.02 * fat_g + 0.004 * E + 1.5 * male + 0.8 * pa + 0.03 * age + rng.normal(0, 2, n)
    return pd.DataFrame({"fat_g": fat_g, "energy_kcal": E, "male": male.astype(float), "age": age,
                         "pa": pa, "y": y})


COVARIATE_ROLES = {"fat_g": "exposure", "energy_kcal": "energy", "male": "covariate",
                   "age": "covariate", "pa": "covariate"}


def test_2_the_residual_with_energy_reproduces_the_standard_coefficient(tmp_path):
    """§5 WP6.2: "Residual-plus-energy reproduces the standard coefficient to 10⁻¹⁰ with sex, age
    and activity covariates." Source check, McCullough & Byrd 2023 (*AJE* 192(11):1801-1805,
    read at https://academic.oup.com/aje/article/192/11/1801/6568373): "A variation on the simple
    nutrient residual model proposed by Willett and Stampfer includes the nutrient residual plus a
    term for total energy intake."

    Ruling 1 (BLUEPRINT §12): this is the residual method's default form. Reference: statsmodels
    OLS of y on fat_g, energy_kcal, male, age and pa (the standard model). Measured: fat_g_adj in
    the inference table the fit stage serves for ``method="residual"``, whose matrix keeps
    energy_kcal. Its interval is the standard model's too (the same column space).
    """
    frame = _covariates_fixture()
    reference = _ols(frame["y"], frame[["fat_g", "energy_kcal", "male", "age", "pa"]])
    adjustment = EnergyAdjustment(method="residual", energy_column="energy_kcal", nutrients=["fat_g"])
    out = _stages(frame, tmp_path, COVARIATE_ROLES, adjustment)
    assert "energy_kcal" in _matrix_columns(out["design"])
    row = out["rows"]["fat_g_adj"]
    assert abs(row["estimate"] - reference.params["fat_g"]) < 1e-10
    assert out["design"]["energy_form"] == "residual"
    assert "Y ~ N_adj + E + C" in METHOD_TABLE["residual"]["specification"]
    assert "McCullough & Byrd 2023" in out["design"]["estimand"]
    said = voice.sentence_for(SetEnergyAdjustment(**adjustment.model_dump()), out["state"])
    assert "with total energy kept in the outcome model" in said

    # The same model through the standard method, to the same tolerance, and its interval.
    standard = _stages(frame, tmp_path / "standard", COVARIATE_ROLES, EnergyAdjustment(
        method="standard", energy_column="energy_kcal", nutrients=["fat_g"]))
    assert abs(standard["rows"]["fat_g"]["estimate"] - row["estimate"]) < 1e-10
    assert row["ci_low"] == pytest.approx(standard["rows"]["fat_g"]["ci_low"], rel=1e-8)
    assert row["ci_high"] == pytest.approx(standard["rows"]["fat_g"]["ci_high"], rel=1e-8)


def _sign_flip_fixture() -> pd.DataFrame:
    """D/exp2_residual_vs_standard.py: n = 50,000; men eat more and a higher protein share, and
    total energy affects the outcome."""
    rng = np.random.default_rng(7)
    n = 50000
    male = rng.integers(0, 2, n)
    E = 1800 + 500 * male + rng.normal(0, 400, n)
    prot_share = 0.15 + 0.03 * male + rng.normal(0, 0.03, n)
    protein_g = prot_share * E / 4
    y = 0.02 * protein_g + 0.004 * E + 3.0 * male + rng.normal(0, 5, n)
    return pd.DataFrame({"protein_g": protein_g, "kcal": E, "male": male.astype(float), "y": y})


SIGN_ROLES = {"protein_g": "exposure", "kcal": "energy", "male": "covariate"}


def _gap_reference(frame: pd.DataFrame) -> tuple[float, float]:
    """(energy dropped, standard): numpy's polyfit residual on the rows given, then statsmodels."""
    slope, _ = np.polyfit(frame["kcal"], frame["protein_g"], 1)
    adjusted = frame["protein_g"] - slope * (frame["kcal"] - frame["kcal"].mean())
    dropped = _ols(frame["y"], pd.DataFrame({"a": adjusted, "male": frame["male"]})).params["a"]
    standard = _ols(frame["y"], frame[["protein_g", "kcal", "male"]]).params["protein_g"]
    return float(dropped), float(standard)


def test_2b_the_energy_dropped_form_says_so_and_the_card_shows_the_gap(tmp_path):
    """§5 WP6.2: "If the energy-dropped form is kept, its label and methods sentence say energy
    left the outcome model, and on the sign-flip fixture (standard +0.0199, energy-dropped
    −0.0067) the card shows the gap."

    Reference: numpy's polyfit for the residual and statsmodels OLS for both models, on every row
    (the design's training rows: +0.01987 and −0.00665, as the audit's script prints them) and on
    the rows the preview card
    reads (its sample of training rows). Measured: the option's label, the design's estimand, the
    fitted row's meaning, the methods sentence, the design warning and the preview card's caption.

    Both numbers are the outcome model's estimates, so under inference they wait for the plan's
    lock (the previews leash, calm/FOUNDATION §5 rule 6): the design's warning says what the form
    does without them (the design cannot see the lock) and gives them under prediction, and the
    card gives them once the plan is locked.
    """
    frame = _sign_flip_fixture()
    dropped, standard = _gap_reference(frame)
    # The audit script prints b_N = 0.01987 and -0.00665; §5 quotes them as +0.0199 and −0.0067.
    assert standard == pytest.approx(0.01987, abs=5e-6) and dropped == pytest.approx(-0.00665, abs=5e-6)

    adjustment = EnergyAdjustment(method="residual_energy_dropped", energy_column="kcal",
                                  nutrients=["protein_g"])
    out = _stages(frame, tmp_path, SIGN_ROLES, adjustment)
    assert "kcal" not in _matrix_columns(out["design"])
    assert out["rows"]["protein_g_adj"]["estimate"] == pytest.approx(dropped, rel=1e-8)
    assert METHOD_TABLE["residual_energy_dropped"]["label"].endswith("total energy left out")
    assert out["design"]["estimand"].startswith("Total energy is not in the outcome model")
    assert "total energy not in the outcome model" in out["rows"]["protein_g_adj"]["meaning"]
    said = voice.sentence_for(SetEnergyAdjustment(**adjustment.model_dump()), out["state"])
    assert "total energy left out of the outcome model" in said and "`kcal` then left" in said

    unseen = next(w for w in out["design"]["warnings"]
                  if w.startswith("kcal left the outcome model"))
    assert unseen == ("kcal left the outcome model: each nutrient's coefficient is the standard "
                      "model's (with kcal kept) only when no covariate correlates with kcal.")
    predicted = _stages(frame, tmp_path / "prediction", SIGN_ROLES, adjustment, purpose="prediction")
    gap = next(w for w in predicted["design"]["warnings"]
               if w.startswith("kcal left the outcome model"))
    assert f"is {_shown(dropped)}, against {_shown(standard)} with kcal kept" in gap

    # The card, on the preview's own sample of the training rows: before the lock without the
    # coefficients, once it is locked with them.
    store = DataStore(Path(out["paths"]["data"]), 1 << 30)
    try:
        state = out["state"].model_copy(update={"energy_adjustment": None})
        captions = []
        for locked in (None, True):
            ctx = PreviewContext(project_id="t",
                                 state=state.model_copy(update={"plan_locked": locked}),
                                 datastore=store, artifact=lambda name: None,
                                 training_row_ids=np.arange(len(frame)), cohort_row_ids=None)
            result = plan(SetEnergyAdjustment(**adjustment.model_dump()), ctx, basis="test")
            captions.append(result.views[0].caption)
        rows = frame.loc[ctx.sample_row_ids(ctx.training_row_ids)]
        card_dropped, card_standard = _gap_reference(rows)
        assert card_dropped < 0 < card_standard  # the sign flip, on the card's own rows
        assert "coefficient" not in captions[0] and "`kcal` leaves the outcome model" in captions[0]
        assert (f"coefficient {_shown(card_dropped)}, against {_shown(card_standard)} with `kcal` "
                f"kept") in captions[1], captions[1]
    finally:
        store.close()


# ── 3 · Log residual ─────────────────────────────────────────────────────────


def test_3_the_log_residual_carries_its_own_estimand(tmp_path):
    """§5 WP6.3: "It carries its own estimand, distinct from the linear residual's." (Closes the
    estimand half of B6 and the methods-sentence misstatement of G17.)

    The log variant rescales each row by its own energy, N × (E/G)^(−b), so it is a different
    model, not a reparametrization. References, on the B/exp1 fixture: numpy's polyfit of log N on
    log E (the elasticity b the estimand states); statsmodels OLS of y on that rescaled nutrient
    built by hand, energy and the covariates (the coefficient the app must report), which differs
    from the standard model's by more than 3% (statsmodels); and with energy dropped, numpy's
    correlation of the rescaled nutrient with the density N/E above 0.999 (the audit's 0.9999),
    which is why that form's estimand reads like the density model's.
    """
    frame = _covariates_fixture()
    logN, logE = np.log(frame["fat_g"]), np.log(frame["energy_kcal"])
    b, _ = np.polyfit(logE, logN, 1)
    rescaled = np.exp(logN - b * (logE - logE.mean()))
    covariates = frame[["male", "age", "pa"]]
    kept_ref = _ols(frame["y"], pd.concat([rescaled.rename("r"), frame["energy_kcal"], covariates],
                                          axis=1)).params["r"]
    standard = _ols(frame["y"], frame[["fat_g", "energy_kcal", "male", "age", "pa"]]).params["fat_g"]
    assert abs(kept_ref / standard - 1) > 0.03
    assert np.corrcoef(rescaled, frame["fat_g"] / frame["energy_kcal"])[0, 1] > 0.999

    estimands = {}
    for method in ("residual", "residual_energy_dropped"):
        linear = _stages(frame, tmp_path / f"{method}-linear", COVARIATE_ROLES, EnergyAdjustment(
            method=method, energy_column="energy_kcal", nutrients=["fat_g"]))
        logged = EnergyAdjustment(method=method, energy_column="energy_kcal", nutrients=["fat_g"],
                                  log_transform=True)
        out = _stages(frame, tmp_path / f"{method}-log", COVARIATE_ROLES, logged)
        text = out["design"]["estimand"]
        assert text.startswith(LOG_ESTIMAND[method])
        assert text != linear["design"]["estimand"]
        assert re.search(r"b = ([0-9.]+) for fat_g", text)
        assert float(re.search(r"b = ([0-9.]+) for fat_g", text).group(1)) == pytest.approx(b, abs=5e-5)
        said = voice.sentence_for(SetEnergyAdjustment(**logged.model_dump()), out["state"])
        assert "geometric-mean energy" in said and "not the standard model's" in said
        assert "replaced by the residual plus the nutrient's mean" not in said  # G17
        estimands[method] = text
        if method == "residual":
            assert out["rows"]["fat_g_adj"]["estimate"] == pytest.approx(kept_ref, rel=1e-8)
    assert "density" in estimands["residual_energy_dropped"]
    assert estimands["residual"] != estimands["residual_energy_dropped"]


# ── 4 · Omitted sources ──────────────────────────────────────────────────────


def _omitted_fixture() -> tuple[pd.DataFrame, dict[str, float]]:
    """B/exp14_loo.py: n = 20,000; energy shares of fat, carbohydrate, protein and alcohol drawn
    from a Dirichlet, so they make up total energy exactly; per-kcal effects fat 0.004,
    carbohydrate 0.001, protein 0.006, alcohol 0."""
    rng = np.random.default_rng(4)
    n = 20000
    E = np.exp(rng.normal(np.log(2000), 0.3, n))
    sh = rng.dirichlet([34, 48, 16, 2], n)
    kf, kc, kp, ka = (sh * E[:, None]).T
    eff = {"fat": 0.004, "carb": 0.001, "protein": 0.006, "alcohol": 0.0}
    y = eff["fat"] * kf + eff["carb"] * kc + eff["protein"] * kp + eff["alcohol"] * ka + rng.normal(0, 1, n)
    frame = pd.DataFrame({"fat_kcal": kf, "carb_kcal": kc, "protein_kcal": kp, "energy_kcal": E, "y": y})
    w = np.array([kc.mean(), kp.mean(), ka.mean()])
    w = w / w.sum()
    truths = {"fat_vs_alcohol": eff["fat"] - eff["alcohol"],
              "fat_vs_average_of_others": eff["fat"] - float(w @ [eff["carb"], eff["protein"], eff["alcohol"]])}
    return frame, truths


OMITTED_ROLES = {"fat_kcal": "exposure", "carb_kcal": "exposure", "protein_kcal": "exposure",
                 "energy_kcal": "energy"}


@pytest.mark.parametrize("method", ["standard", "residual"])
def test_4_with_three_sources_the_estimand_names_the_ones_left_out(tmp_path, method):
    """§5 WP6.4: "With fat, carbohydrate and protein plus energy, the estimand reads "in place of
    the energy sources not in the model: alcohol, other" and the coefficient 0.0040 is described as
    fat versus alcohol." NUTRITION_PACK §05a: "Each coefficient is the effect of substituting that
    component for the omitted one. Name the omitted component in your results."

    Reference: statsmodels OLS of y on the three sources and total energy (the coefficient), and
    the data-generating effects: fat versus alcohol 0.0040, fat versus the energy-weighted average
    of the others 0.0018, the old label's quantity. The coefficient's standard error is about
    0.00024, so 0.0040 is within one and 0.0018 about nine away.
    """
    frame, truths = _omitted_fixture()
    reference = _ols(frame["y"], frame[["fat_kcal", "carb_kcal", "protein_kcal", "energy_kcal"]])
    assert truths["fat_vs_alcohol"] == pytest.approx(0.0040)
    assert truths["fat_vs_average_of_others"] == pytest.approx(0.0018, abs=1e-4)
    se = reference.bse["fat_kcal"]
    assert abs(reference.params["fat_kcal"] - truths["fat_vs_alcohol"]) < 2 * se
    assert abs(reference.params["fat_kcal"] - truths["fat_vs_average_of_others"]) > 6 * se

    out = _stages(frame, tmp_path, OMITTED_ROLES, EnergyAdjustment(
        method=method, energy_column="energy_kcal", nutrients=["fat_kcal", "carb_kcal", "protein_kcal"]))
    estimand = out["design"]["estimand"]
    assert "in place of the energy sources not in the model: alcohol, other" in estimand
    assert "in place of the average of all other energy sources" not in estimand
    fat = out["rows"]["fat_kcal" if method == "standard" else "fat_kcal_adj"]
    assert fat["estimate"] == pytest.approx(reference.params["fat_kcal"], rel=1e-8)
    assert fat["meaning"] == "fat in place of alcohol and other energy (total energy fixed)"


def test_4b_fiber_beside_carbohydrate_by_difference_is_warned(tmp_path):
    """§5 WP6.4: "Fiber beside carbohydrate-by-difference triggers a warning." (B19.)

    Carbohydrate by difference (USDA's, and NHANES's total) already holds fiber, so fiber's
    energy counts twice in any kcal accounting. Reference: the design warnings of two otherwise
    identical tables, one with a total carbohydrate column beside fiber and one with sugars
    (which are not carbohydrate by difference) beside fiber.
    """
    rng = np.random.default_rng(19)
    n = 300
    E = rng.normal(2000, 400, n)
    base = pd.DataFrame({"fiber_g": rng.normal(18, 5, n), "carb_g": 0.12 * E + rng.normal(0, 20, n),
                         "fat_g": 0.035 * E + rng.normal(0, 8, n), "energy_kcal": E,
                         "y": rng.normal(0, 1, n)})
    roles = {"fiber_g": "exposure", "carb_g": "exposure", "fat_g": "exposure", "energy_kcal": "energy"}
    adjustment = EnergyAdjustment(method="standard", energy_column="energy_kcal",
                                  nutrients=["fiber_g", "carb_g", "fat_g"])
    warned = _stages(base, tmp_path / "carb", roles, adjustment, purpose="prediction")
    assert any(w.startswith("fiber_g sits beside carb_g") and "fiber's energy counts twice" in w
               for w in warned["design"]["warnings"]), warned["design"]["warnings"]

    sugars = base.rename(columns={"carb_g": "sugar_g"})
    roles2 = {("sugar_g" if c == "carb_g" else c): r for c, r in roles.items()}
    adjustment2 = adjustment.model_copy(update={"nutrients": ["fiber_g", "sugar_g", "fat_g"]})
    quiet = _stages(sugars, tmp_path / "sugar", roles2, adjustment2, purpose="prediction")
    assert not any("counts twice" in w for w in quiet["design"]["warnings"])


# ── 5 · Substitution completeness ────────────────────────────────────────────


def _composite_fixture() -> pd.DataFrame:
    """D/exp3_substitution_composite.py: n = 20,000; per-kcal effects protein 0.010, carbohydrate
    0.004, fat 0, alcohol 0.020; a shared lifestyle factor moves alcohol and fat with protein.
    The direct 100 kcal carbohydrate → protein effect is 100 × (0.010 − 0.004) = 0.600."""
    rng = np.random.default_rng(3)
    n = 20000
    z = rng.normal(0, 1, (n, 4))
    L = rng.normal(0, 1, n)
    kp = 330 + 60 * z[:, 0] + 40 * L
    kc = 1000 + 200 * z[:, 1] - 80 * L
    kf = 750 + 150 * z[:, 2] + 70 * L
    ka = np.clip(80 + 60 * z[:, 3] + 50 * L, 0, None)
    kcal = kp + kc + kf + ka
    y = 0.010 * kp + 0.004 * kc + 0.0 * kf + 0.020 * ka + rng.normal(0, 3, n)
    return pd.DataFrame({"protein_g": kp / 4, "carb_g": kc / 4, "fat_g": kf / 9, "alcohol_g": ka / 7,
                         "kcal": kcal, "y": y})


SWAP = SubstitutionSpec(donor="carb_g", recipient="protein_g", step_kcal=100.0)
TRUE_SWAP = 100 * (0.010 - 0.004)


def _contrast(frame: pd.DataFrame, columns: list[str]) -> tuple[float, float]:
    """(estimate, SE) of 100·(β_protein/4 − β_carb/4) from statsmodels OLS on ``columns``."""
    fit = _ols(frame["y"], frame[columns])
    c = np.zeros(len(columns) + 1)
    c[1 + columns.index("protein_g")] = 100 / 4
    c[1 + columns.index("carb_g")] = -100 / 4
    return float(c @ fit.params.to_numpy()), float(np.sqrt(c @ fit.cov_params().to_numpy() @ c))


def test_5_a_two_nutrient_swap_lists_the_omitted_sources_and_inference_blocks_it(tmp_path):
    """§5 WP6.5: "A two-nutrient model lists the omitted sources with a concern (inference: block
    and record above a stated omitted share)". Source: Tomova, Gilthorpe & Tennant 2022
    (PMC9630885): "Wherever ≥2 components are involved in the substitution, there is scope for
    composite variable bias unless the individual effects are estimated and combined using an
    all-components approach."

    The stated share is ``MAX_OMITTED_SHARE`` (5% of total energy on average, the Atwater factors'
    own error). Reference: the omitted share counted with numpy from the fixture's kcal columns,
    (fat + alcohol kcal) / total, 38% on average. Measured: the substitution stage's
    ``omitted_energy`` and note; the ``set_substitution`` validator under inference (refused,
    with an exit that adds the missing sources and one that records the attestation), under
    prediction (accepted), and with the attestation (accepted).
    """
    frame = _composite_fixture()
    share = float(((frame["fat_g"] * 9 + frame["alcohol_g"] * 7) / frame["kcal"]).mean())
    assert share > MAX_OMITTED_SHARE
    roles = {"protein_g": "exposure", "carb_g": "exposure", "kcal": "energy"}
    adjustment = EnergyAdjustment(method="standard", energy_column="kcal", nutrients=["protein_g", "carb_g"])
    out = _stages(frame, tmp_path, roles, adjustment, substitution=SWAP)
    omitted = out["substitution"]["omitted_energy"]
    assert omitted["omitted"] == ["fat", "alcohol", "other"]
    assert omitted["sources"] == ["protein", "carbohydrate"]
    assert omitted["mean_share"] == pytest.approx(share, rel=1e-9)
    assert "fat, alcohol and other energy make up the rest of total energy" in out["substitution"]["note"]

    store = DataStore(Path(out["paths"]["data"]), 1 << 30)
    try:
        ctx = {"state": out["state"], "store": lambda: store, "sealed": lambda: []}
        decision = {"kind": "set_substitution", "donor": "carb_g", "recipient": "protein_g"}
        with pytest.raises(Refusal) as refused:
            validate(decision, ctx)
        assert refused.value.code == "omitted_energy_sources"
        assert "fat, alcohol, other energy" in str(refused.value) and f"{share:.0%}" in str(refused.value)
        exits = {e["label"]: e["decision"] for e in refused.value.exits}
        # The readings ledger (BLUEPRINT §14.1): each source the names read is added on its own,
        # one reading per answer, never several in one decision.
        for column in ("fat_g", "alcohol_g"):
            added = parse_decision(exits[f"Add `{column}` to the model as a study factor"])
            assert isinstance(added, SetRoles)
            assert added.roles[column] == "exposure"
            assert {c: r for c, r in added.roles.items() if c != column} == roles
        kept = parse_decision(exits["Keep this swap; the curve carries their confounding"])
        assert kept.acknowledged is True
        validate(kept, ctx)  # the recorded attestation is accepted
        said = voice.sentence_for(kept, out["state"])
        assert "kept although energy sources are missing from the model" in said
        prediction = {**ctx, "state": out["state"].model_copy(update={"purpose": "prediction"})}
        validate(decision, prediction)  # a model contrast: stated, not blocked
        # With every source in the model, the swap is accepted under inference.
        complete = out["state"].model_copy(update={"roles": {
            **roles, "fat_g": "exposure", "alcohol_g": "exposure"}})
        validate(decision, {**ctx, "state": complete})
    finally:
        store.close()


@pytest.mark.parametrize("method", ["standard", "residual"])
def test_5b_adding_the_remaining_sources_moves_the_swap_to_the_truth(tmp_path, method):
    """§5 WP6.5: "adding the remaining sources moves carbohydrate → protein from +0.940 to about
    +0.58 (truth 0.600)."

    Reference: the 100 kcal contrast 100·(β_protein/4 − β_carb/4) from statsmodels OLS on the
    fixture's columns, with protein and carbohydrate only (+0.940, SE 0.033) and with all four
    sources (+0.578, SE 0.032); the truth is the data-generating 0.600. Measured: the substitution
    stage's curve at k = 100 (a linear model, so the curve is the contrast exactly). Bounds: the
    complete model within 2.5 standard errors of the truth, the two-nutrient one more than 8 away.
    """
    frame = _composite_fixture()
    curves = {}
    for exposures in (["protein_g", "carb_g"], ["protein_g", "carb_g", "fat_g", "alcohol_g"]):
        roles = {e: "exposure" for e in exposures}
        roles["kcal"] = "energy"
        adjustment = EnergyAdjustment(method=method, energy_column="kcal", nutrients=exposures)
        out = _stages(frame, tmp_path / str(len(exposures)), roles, adjustment, substitution=SWAP)
        sub = out["substitution"]
        estimate, se = _contrast(frame, [*exposures, "kcal"])
        curve = sub["models"][0]["delta"][sub["ks"].index(100.0)]
        assert curve == pytest.approx(estimate, rel=1e-6)
        curves[len(exposures)] = (curve, se)
    two, four = curves[2], curves[4]
    assert two[0] == pytest.approx(0.940, abs=0.005)
    assert four[0] == pytest.approx(0.578, abs=0.005)
    assert abs(four[0] - TRUE_SWAP) < 2.5 * four[1]
    assert abs(two[0] - TRUE_SWAP) > 8 * two[1]


# ── 6 · All components ───────────────────────────────────────────────────────


def _all_components_fixture() -> tuple[pd.DataFrame, float]:
    """D/exp1_energy_models.py: n = 20,000; per-kcal effects protein 0.010, carbohydrate 0.004,
    fat 0; older people eat less; total energy is the three sources exactly. Returns the frame and
    the true average relative effect per g protein, 4 × (0.010 − (w_c·0.004 + w_f·0)) with w the
    sources' mean shares of the remaining energy: 0.0305."""
    rng = np.random.default_rng(1)
    n = 20000
    age = rng.normal(50, 15, n)
    size = rng.normal(0, 1, n) - 0.03 * (age - 50)
    E_base = 2100 + 450 * size
    share_p = np.clip(rng.normal(0.16, 0.03, n), 0.05, 0.4)
    share_f = np.clip(rng.normal(0.34, 0.05, n), 0.1, 0.6)
    share_c = 1 - share_p - share_f
    kp, kf, kc = E_base * share_p, E_base * share_f, E_base * share_c
    y = 0.010 * kp + 0.004 * kc + 0.0 * kf + 0.2 * (age - 50) + rng.normal(0, 5, n)
    frame = pd.DataFrame({"protein_g": kp / 4, "carb_g": kc / 4, "fat_g": kf / 9,
                          "kcal": kp + kf + kc, "age": age, "y": y})
    w_c = kc.mean() / (kc.mean() + kf.mean())
    return frame, 4 * (0.010 - (w_c * 0.004 + (1 - w_c) * 0.0))


ALL_ROLES = {"protein_g": "exposure", "carb_g": "exposure", "fat_g": "exposure", "kcal": "energy",
             "age": "covariate"}
ALL = EnergyAdjustment(method="all_components", energy_column="kcal",
                       nutrients=["protein_g", "carb_g", "fat_g"])


def test_6_the_all_components_model_recovers_the_average_relative_effect(tmp_path):
    """§5 WP6.6: "The named option recovers an average relative effect of 0.0307 (truth 0.0305)
    with an interval". Source check, Tomova, Arnold, Gilthorpe & Tennant 2022 (*AJCN*
    115(1):189-198, full text read at Europe PMC, PMC8755101): "Accurate estimates of both the
    total and average relative causal effects may instead be derived by simultaneously adjusting
    for all dietary components, an approach we term the "all-components model."" and "This is
    achieved by subtracting a weighted average of the estimated effects of all other individual
    component sources of energy from the total causal effect of the exposure (…, where [w] is the
    proportion of the remaining energy intake contributed by each component)."

    Reference: statsmodels OLS of y on the three sources in kcal and age (the all-components
    model); θ = 4·(β_p − w_c·β_c − w_f·β_f) with w the sources' mean shares of the remaining
    energy, and its interval from HC3 written out from the definition (``references.py``; the app
    reports HC3 for independent rows under inference since WP2) on t(n − p). Measured: the
    ``protein_g_relative`` row of the fit stage's inference table.
    """
    frame, truth = _all_components_fixture()
    assert truth == pytest.approx(0.0305, abs=5e-5)
    columns = ["kp", "kc", "kf", "age"]
    X = pd.DataFrame({"kp": frame["protein_g"] * 4, "kc": frame["carb_g"] * 4,
                      "kf": frame["fat_g"] * 9, "age": frame["age"]})
    fit = _ols(frame["y"], X)
    w_c = X["kc"].mean() / (X["kc"].mean() + X["kf"].mean())
    c = np.zeros(len(columns) + 1)
    c[1:4] = [4.0, -4.0 * w_c, -4.0 * (1 - w_c)]
    theta = float(c @ fit.params.to_numpy())
    exog = sm.add_constant(X).to_numpy(dtype=float)
    V = ref.hc3_by_definition(exog, np.asarray(fit.resid))
    se = float(np.sqrt(c @ V @ c))
    q = stats.t.ppf(0.975, len(X) - exog.shape[1])
    assert theta == pytest.approx(0.0307, abs=5e-5)  # the audit's number, reproduced

    out = _stages(frame, tmp_path, ALL_ROLES, ALL)
    assert {"kcal_from_protein_g", "kcal_from_carb_g", "kcal_from_fat_g"} <= _matrix_columns(out["design"])
    row = out["rows"]["protein_g_relative"]
    assert row["estimate"] == pytest.approx(theta, rel=1e-8)
    assert row["ci_low"] == pytest.approx(theta - q * se, rel=1e-6)
    assert row["ci_high"] == pytest.approx(theta + q * se, rel=1e-6)
    assert row["ci_low"] < truth < row["ci_high"]
    assert "average relative effect" in row["meaning"] and "carb_g 59%, fat_g 41%" in row["meaning"]
    total = out["rows"]["kcal_from_protein_g"]
    assert total["estimate"] == pytest.approx(fit.params["kp"], rel=1e-8)
    assert "(total effect)" in total["meaning"]
    assert "all-components model" in out["design"]["estimand"]
    said = voice.sentence_for(SetEnergyAdjustment(**ALL.model_dump()), out["state"])
    assert "all-components model (Tomova et al. 2022)" in said

    # Under prediction the same average relative effect, as a point estimate.
    predicted = _stages(frame, tmp_path / "prediction", ALL_ROLES, ALL, purpose="prediction")
    point = predicted["rows"]["protein_g_relative"]
    assert point["estimate"] == pytest.approx(theta, rel=1e-6) and point["ci_low"] is None


def test_6b_under_inference_all_components_ranks_first_with_its_cost_in_one_line():
    """§5 WP6.6: "under inference it ranks first with the dispute and precision cost in one line."
    BLUEPRINT §12 ruling 2: "all-components ranks first for substitution questions under
    inference, and under prediction the app says the choice matters little." Source check,
    Tomova et al. 2022: "this strategy does introduce a trade-off between minimizing bias (by
    including the largest number of components at the finest level of detail) and maximizing
    precision (by having to estimate many parameters, i.e., 1 for each additional dietary
    component)"; the dispute is Willett, Stampfer & Tobias 2022 (*AJCN* 116(2):608-609, "Re:
    Adjustment for energy intake in nutritional research", per Europe PMC).

    Measured: the ranking the proposals stage serves for the energy question on the fixture's
    table, under each purpose, and the order of soundness itself. Reference: the rulings.
    """
    frame, _ = _all_components_fixture()
    columns = [{"name": c, "dtype": "numeric", "n_unique": int(frame[c].nunique()),
                "n_missing": 0} for c in frame.columns]
    served = {}
    for purpose in ("inference", "prediction"):
        reading = build_proposals(frame, columns, lens=["dietary"], target="y", roles=ALL_ROLES,
                                  purpose=purpose)["energy"]
        assert reading["applicability"]["all_components"]["ok"], reading["applicability"]
        served[purpose] = reading["ranking"]
    inference = served["inference"]
    assert inference["purpose"] == "inference" and inference["order"][0] == "all_components"
    line = inference["line"]
    assert line == TENSION["inference"] and "\n" not in line and line.count(".") == 1
    assert "disputed" in line and "precision" in line and "Tomova 2022" in line

    prediction = served["prediction"]
    assert "matters little" in prediction["line"]
    keeps_energy = {"standard", "residual", "density_multivariate", "partition", "all_components"}
    first = prediction["order"][:len(keeps_energy)]
    assert set(first) == keeps_energy  # the energy-dropping models rank after every one that keeps it
    assert prediction["order"].index("residual_energy_dropped") > prediction["order"].index("residual")

    # An inapplicable method never leads: without fat the all-components model cannot run. The
    # line then says so instead of claiming it ranks first (repair round: the line said "ranks
    # first" while the order listed it last).
    applicability = applicable_methods(list(frame.columns), "kcal", ["protein_g", "carb_g"])
    ranked = rank_methods("inference", applicability)
    assert ranked["first"] == "standard" and ranked["order"][-1] == "all_components"
    assert ranked["line"].startswith("All components would rank first")
    assert "cannot run on these columns" in ranked["line"] and "standard" in ranked["line"]
    assert "\n" not in ranked["line"] and ranked["line"].count(".") == 1
    # The option is taught under its own name, with its dispute badged.
    taught = entry("energy_adjustment")
    assert "all_components" in {o.value for o in taught.options}
    drawer = next(s for s in taught.drawer.sections if s.heading == "All components")
    assert drawer.evidence.status == "DISPUTED"


def test_6c_a_refused_partition_says_why_in_a_whole_sentence():
    """Repair round (verifier minor 4): on a table whose energy is in kcal on some rows and kJ on
    others, the partition and all-components options read "There is no single factor to apply, so
    this is not a conversion the app can offer; the rows have." — the reason cut inside a clause.

    Reference: the fixture's own arithmetic, declared over reconstructed energy (4·protein +
    4·carbohydrate + 9·fat) per row, which sits near 1 on half the rows and near 4.184 on the
    other half, so no single factor converts it. Measured: the energy card's reasons, each a whole
    sentence within the card's 20-word budget, and the ranking line, which names the method that
    leads instead of claiming all components does."""
    from turbotab.core.teaching import COMPOSED_BUDGETS
    from turbotab.core.voice import words

    rng = np.random.default_rng(1)
    n = 400
    protein, carb, fat = rng.normal(80, 15, n), rng.normal(250, 40, n), rng.normal(70, 15, n)
    kcal = 4 * protein + 4 * carb + 9 * fat + rng.normal(0, 30, n)
    kcal[: n // 2] *= 4.184  # half the rows in kilojoules
    frame = pd.DataFrame({"protein_g": protein, "carb_g": carb, "fat_g": fat, "energy_kcal": kcal,
                          "y": rng.normal(size=n)})
    ratio = kcal / (4 * protein + 4 * carb + 9 * fat)
    assert np.median(ratio[: n // 2]) == pytest.approx(4.184, rel=0.02)
    assert np.median(ratio[n // 2:]) == pytest.approx(1.0, rel=0.02)
    columns = [{"name": c, "dtype": "numeric", "n_unique": n, "n_missing": 0} for c in frame]
    roles = {"protein_g": "exposure", "carb_g": "exposure", "fat_g": "exposure",
             "energy_kcal": "energy"}
    # The roles are the author's, each confirmed on its own (BLUEPRINT §14, recognition's leash):
    # half the rows in kJ keep total energy from following its macronutrients by the values alone.
    reading = build_proposals(frame, columns, lens=["dietary"], target="y", roles=roles,
                              purpose="inference", settled=set(roles))["energy"]
    for method in ("partition", "all_components"):
        verdict = reading["applicability"][method]
        assert not verdict["ok"]
        reason = verdict["reason"]
        assert "no single factor" in reason and reason.endswith("separate the rows by source first.")
        assert words(reason) <= COMPOSED_BUDGETS["option_reason"]
        assert not reason.endswith("the rows have.")
    ranking = reading["ranking"]
    assert ranking["order"][0] == "standard" and ranking["order"][-2:] == ["all_components", "partition"]
    assert "cannot run on these columns" in ranking["line"]


# ── 7 · Nested totals ────────────────────────────────────────────────────────

NESTED_NUTRIENTS = ["protein", "sugar", "carb", "fat_total", "fat_sat", "fat_mon", "fat_poly"]


def _nested_check(frame: pd.DataFrame, folder: Path) -> dict[str, float]:
    """Fit the standard model with every part and with totals only, through the app, and check each
    total's row against statsmodels; returns the fat_total coefficients."""
    out: dict[str, float] = {}
    for label, exposures in (("parts", NESTED_NUTRIENTS), ("totals", ["protein", "carb", "fat_total"])):
        roles = {e: "exposure" for e in exposures}
        roles.update({"kcal": "energy", "age": "covariate"})
        run = _stages(frame, folder / label, roles, EnergyAdjustment(
            method="standard", energy_column="kcal", nutrients=exposures), target="glucose")
        reference = _ols(frame["glucose"], frame[["age", "kcal", *exposures]])
        fat = run["rows"]["fat_total"]
        assert fat["estimate"] == pytest.approx(reference.params["fat_total"], rel=1e-8)
        if label == "parts":
            # Q-b: the parts named as nouns, as every surface names them (not as columns)
            assert fat["meaning"].startswith("remaining fat (holding saturated fat, monounsaturated "
                                             "fat and polyunsaturated fat fixed)")
            assert run["rows"]["carb"]["meaning"].startswith("remaining carbohydrate (holding sugar fixed)")
            assert ("fat_total sits beside its own parts fat_sat, fat_mon and fat_poly"
                    in run["design"]["estimand"])
            assert any("fat_total's coefficient is the remainder" in w for w in run["design"]["warnings"])
        else:
            assert fat["meaning"] == "fat in place of alcohol and other energy (total energy fixed)"
            assert "remaining" not in run["design"]["estimand"]
        out[label] = float(fat["estimate"])
    return out


def test_7_a_total_beside_its_parts_reads_as_the_remainder(tmp_path):
    """§5 WP6.7: "With SFA, MUFA and PUFA beside total fat, the total's row reads "remaining fat
    (holding SFA, MUFA, PUFA fixed)"" — on an NHANES-shaped table that runs everywhere (the real
    export's columns, simulated; ``modeling_fixtures.nhanes_like``).

    Reference: statsmodels OLS of glucose on age, kcal and the nutrients, with the parts and with
    totals only. Measured: the fit stage's coefficient rows, their meanings, the design's
    estimand and its warning.
    """
    frame = mf.nhanes_like(3000, seed=31)[["glucose", "age", "kcal", *NESTED_NUTRIENTS]]
    _nested_check(frame, tmp_path)


@pytest.mark.skipif(NHANES is None, reason="the NHANES export is not on this machine")
def test_7b_the_nhanes_reference_with_and_without_the_parts(tmp_path):
    """§5 WP6.7, on the real export: "(NHANES reference: 0.3127 with parts against 0.0734 without)."

    The audit's rows (``D-skeptic/s7_nest.py``): kcal within 500–5,000 and every model column
    present. Reference: statsmodels OLS on those rows, 0.3127 and 0.0734. Skipped where the
    export is absent, as every test of the real export is (``stage_harness.NHANES``);
    ``test_7`` holds the labels everywhere.
    """
    d = pd.read_csv(NHANES)
    d = d[(d.kcal >= 500) & (d.kcal <= 5000)].dropna(subset=["glucose", "age", "kcal", *NESTED_NUTRIENTS])
    frame = d[["glucose", "age", "kcal", *NESTED_NUTRIENTS]].reset_index(drop=True)
    got = _nested_check(frame, tmp_path)
    assert got["parts"] == pytest.approx(0.3127, abs=5e-5)
    assert got["totals"] == pytest.approx(0.0734, abs=5e-5)
