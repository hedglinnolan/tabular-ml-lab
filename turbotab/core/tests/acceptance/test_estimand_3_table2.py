"""ESTIMAND · 3 · the Table 2 display (MODELING_SEQUENCE §1 row 11, inference: "The declared object
is a crude model, a declared adjustment sequence (Model 1: age, sex and energy; Model 2: plus
confounders; optional Model 3: plus possible mediators, labeled) and the primary model, shown for
the exposure only (the Table 2 fallacy: adjustment terms are moved to an appendix titled
'adjustment terms, not effect estimates')").

Sources (read 2026-10-05): Westreich D, Greenland S. The Table 2 fallacy. *Am J Epidemiol*
2013;177:292–298: "Presentation of exposure and confounder effect estimates from a single model may
lead to several interpretative difficulties, inviting confusion of direct-effect estimates with
total-effect estimates for covariates in the model. These effect estimates may also be confounded
even though the effect estimate for the main exposure is not confounded." STROBE cohort checklist
item 16(a): "Give unadjusted estimates and, if applicable, confounder-adjusted estimates and their
precision (eg, 95% confidence interval). Make clear which confounders were adjusted for and why
they were included."

References: statsmodels' least squares with HC3 standard errors on t(n − p) (the linear family's
interval, ``models/inference.py``) on designs written out here, model by model; under multiple
imputation the same fit in each completed copy, pooled by Rubin's rules with Barnard–Rubin degrees
of freedom written out here.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats

from turbotab.core.models.effects import APPENDIX_TITLE, split_rows
from turbotab.core.tests.acceptance import estimand_fixtures as ef

MODELS = {"crude": [], "model_1": ["age", "sex"], "model_2": ["age", "sex", "smoking", "activity"],
          "model_3": ["age", "sex", "smoking", "activity", "bmi"]}


def _ols(frame: pd.DataFrame, columns: list[str], outcome: str = "glucose") -> sm.regression.linear_model.RegressionResultsWrapper:
    X = ef.design_matrix(frame, ["fiber", *columns])
    return sm.OLS(frame[outcome].to_numpy(dtype=float), X).fit(cov_type="HC3", use_t=True)


@pytest.fixture(scope="module")
def linear(tmp_path_factory):
    frame = ef.cohort(700, seed=5)
    st = ef.state(target="glucose", task="regression", measure="mean_difference",
                  model_1=["age", "sex"])
    return frame, ef.run(frame, tmp_path_factory.mktemp("table2"), st)


def test_3_each_declared_model_shows_the_exposure_only_and_agrees_with_statsmodels(linear):
    frame, run = linear
    seq = ef.sequence(run["effects"])
    assert list(seq) == ["crude", "model_1", "model_2", "model_3"]
    assert [seq[k]["label"] for k in seq] == ["Unadjusted", "Model 1", "Model 2 (primary)", "Model 3"]
    for key, columns in MODELS.items():
        model = seq[key]
        assert [r["feature"] for r in model["effects"]] == ["fiber"], key
        reference = _ols(frame, columns)
        row = model["effects"][0]
        assert row["estimate"] == pytest.approx(reference.params["fiber"], rel=1e-8), key
        assert row["se"] == pytest.approx(reference.bse["fiber"], rel=1e-8), key
        low, high = reference.conf_int().loc["fiber"]
        assert (row["ci_low"], row["ci_high"]) == pytest.approx((low, high), rel=1e-8), key
        assert set(model["adjusted_for"]) == set(columns), key
        assert model["n_rows"] == len(frame)
    assert seq["model_3"]["note"] == ("further adjusted for `bmi`, a possible mediator: not a total "
                                      "effect.")
    assert seq["crude"]["note"].startswith("STROBE 16a's unadjusted estimate")
    # Model 2 is the primary: the fit stage's own table for the exposure
    primary = next(r for r in run["fit_raw"]["models"][0]["coefficients"] if r["feature"] == "fiber")
    assert seq["model_2"]["effects"][0]["estimate"] == pytest.approx(primary["estimate"], rel=1e-12)


def test_3_every_other_row_is_an_adjustment_term_in_the_appendix(linear):
    frame, run = linear
    family = run["effects"]["families"][0]
    assert run["effects"]["appendix_title"] == APPENDIX_TITLE == "adjustment terms, not effect estimates"
    appendix = {a["key"]: a for a in family["appendix"]}
    reference = _ols(frame, MODELS["model_2"])
    terms = {t["feature"]: t for t in appendix["model_2"]["terms"]}
    assert set(terms) == {"(intercept)", "age", "sex_male", "smoking", "activity"}
    assert terms["age"]["estimate"] == pytest.approx(reference.params["age"], rel=1e-8)
    assert terms["age"]["why"].startswith("an adjustment term")
    assert terms["(intercept)"]["why"] == "the model's baseline, not an effect"
    assert {t["feature"] for t in appendix["crude"]["terms"]} == {"(intercept)"}
    # the served fit: the exposure's rows only, the rest under the appendix's title
    served = run["fit"]
    model = served["models"][0]
    assert [r["feature"] for r in model["coefficients"]] == ["fiber"]
    assert {r["feature"] for r in model["adjustment_terms"]} == {
        "(intercept)", "age", "sex_male", "smoking", "activity"}
    assert all(r["why"] for r in model["adjustment_terms"])
    assert served["estimand"]["appendix"] == APPENDIX_TITLE
    assert served["estimand"]["features"] == ["fiber"]


def test_3_the_unadjusted_estimate_is_always_shown(tmp_path):
    """With no Model 1 declared and no possible mediator beside the primary, the sequence is the
    unadjusted model and the primary; never the primary alone (STROBE 16a)."""
    frame = ef.cohort(400, seed=9)
    answers = {c: a for c, a in ef.ANSWERS.items() if c != "bmi"}
    roles = {c: r for c, r in ef.ROLES.items() if c != "bmi"}
    st = ef.state(target="glucose", task="regression", measure="mean_difference", roles=roles,
                  answers=answers)
    run = ef.run(frame, tmp_path, st, fit=False)
    seq = ef.sequence(run["effects"])
    assert list(seq) == ["crude", "model_2"]
    crude = _ols(frame, [])
    assert seq["crude"]["effects"][0]["estimate"] == pytest.approx(crude.params["fiber"], rel=1e-8)
    # Model 1 is offered for declaration with the pack's guess, never read from a name silently
    card = run["effects"]["model_1"]
    assert card["declared"] is None and card["guess"] == ["age", "sex"]
    assert card["decision"] == {"kind": "set_model_sequence", "exposure": "fiber",
                                "model_1": ["age", "sex"]}


def test_3_a_modifier_main_effect_is_never_an_effect_estimate():
    """``split_rows`` is the one rule both displays apply: a declared modifier's main effect goes to
    the appendix with its reason, as does every row that is not the exposure's."""
    rows = [{"feature": "(intercept)", "estimate": 1.0}, {"feature": "fiber", "estimate": -0.4},
            {"feature": "sex_male", "estimate": 2.0}, {"feature": "age", "estimate": 0.3}]
    shown, appendix = split_rows(rows, ["fiber"], modifiers=["sex"])
    assert [r["feature"] for r in shown] == ["fiber"]
    why = {r["feature"]: r["why"] for r in appendix}
    assert why["sex_male"] == ("a modifier's main effect: the exposure's effect is read within its "
                               "levels")
    assert why["age"].startswith("an adjustment term")


def _all_components(n: int = 1500, seed: int = 11) -> pd.DataFrame:
    """Total energy is exactly the Atwater sum of protein, carbohydrate and fat, so each source's
    kcal per gram is settled by the values (``readings.factor_in_grams``)."""
    rng = np.random.default_rng(seed)
    age = rng.normal(50, 15, n)
    size = rng.normal(0, 1, n) - 0.03 * (age - 50)
    energy = 2100 + 450 * size
    share_p = np.clip(rng.normal(0.16, 0.03, n), 0.05, 0.4)
    share_f = np.clip(rng.normal(0.34, 0.05, n), 0.1, 0.6)
    kp, kf = energy * share_p, energy * share_f
    kc = energy - kp - kf
    y = 0.010 * kp + 0.004 * kc + 0.2 * (age - 50) + rng.normal(0, 5, n)
    return pd.DataFrame({"pid": np.arange(n), "protein_g": kp / 4, "carb_g": kc / 4,
                         "fat_g": kf / 9, "kcal": kp + kf + kc, "age": age, "y": y})


def test_3_under_the_all_components_model_each_model_carries_the_substitution(tmp_path):
    """BLUEPRINT §12 ruling 2 ranks the all-components model first for a substitution under
    inference; its substitution is the average relative effect (Tomova et al. 2022), a weighted
    contrast of the sources. Every model of the sequence that holds the other sources carries it
    beside the exposure's own coefficient (an addition); the unadjusted model holds no other source,
    so it has none. Reference: statsmodels' least squares on the kcal columns written out, the
    contrast ``4·(β_p − w_c β_c − w_f β_f)`` with ``w`` the other sources' mean shares, and its HC3
    variance ``c'Vc``."""
    from turbotab.core import decisions as d

    frame = _all_components()
    roles = {"pid": "identifier", "protein_g": "exposure", "carb_g": "covariate",
             "fat_g": "covariate", "kcal": "energy", "age": "covariate"}
    answers = {"age": ef.CONFOUNDER,
               "carb_g": {"causes_exposure": "unknown", "causes_outcome": "unknown",
                          "after_exposure": "no"},
               "fat_g": {"causes_exposure": "unknown", "causes_outcome": "unknown",
                         "after_exposure": "no"}}
    st = ef.state(target="y", task="regression", exposure="protein_g", measure="mean_difference",
                  roles=roles, answers=answers, model_1=["kcal"], lens=["dietary"],
                  energy_adjustment=d.EnergyAdjustment(method="all_components", energy_column="kcal",
                                                       nutrients=["protein_g", "carb_g", "fat_g"]))
    st = st.model_copy(update={"estimand": st.estimand.model_copy(update={"contrast": "substitution"})})
    run = ef.run(frame, tmp_path, st, fit=False)
    seq = ef.sequence(run["effects"])
    assert [r["feature"] for r in seq["crude"]["effects"]] == ["kcal_from_protein_g"]
    kp, kc, kf = frame["protein_g"] * 4, frame["carb_g"] * 4, frame["fat_g"] * 9
    for key, extra in (("model_1", []), ("model_2", ["age"])):
        X = pd.DataFrame({"const": 1.0, "kp": kp, "kc": kc, "kf": kf,
                          **{c: frame[c] for c in extra}})
        fit = sm.OLS(frame["y"].to_numpy(float), X).fit(cov_type="HC3")
        w_c = kc.mean() / (kc.mean() + kf.mean())
        c = np.zeros(X.shape[1])
        c[1], c[2], c[3] = 4.0, -4.0 * w_c, -4.0 * (1 - w_c)
        theta = float(c @ fit.params.to_numpy())
        se = float(np.sqrt(c @ fit.cov_params().to_numpy() @ c))
        rows = {r["feature"]: r for r in seq[key]["effects"]}
        assert set(rows) == {"kcal_from_protein_g", "protein_g_relative"}, key
        assert rows["protein_g_relative"]["estimate"] == pytest.approx(theta, rel=1e-8), key
        assert rows["protein_g_relative"]["se"] == pytest.approx(se, rel=1e-8), key
        assert rows["protein_g_relative"]["meaning"].startswith("protein_g in place of the other "
                                                                "energy sources")
        assert rows["kcal_from_protein_g"]["estimate"] == pytest.approx(fit.params["kp"], rel=1e-8)
    [sens_add, sens_sub] = run["effects"]["families"][0]["sensitivity"]
    assert sens_add["feature"] == "kcal_from_protein_g" and sens_add["methods"][0] == "robustness_value"
    assert sens_sub["feature"] == "protein_g_relative" and sens_sub["methods"] == ["e_value"]
    assert sens_sub["not_computed"].startswith("No robustness value: the average relative effect")


def _rubin(estimates: list[float], ses: list[float], df_com: float) -> tuple[float, float, float, float]:
    """Rubin's rules with Barnard & Rubin's (1999) degrees of freedom, written out."""
    m = len(estimates)
    q = float(np.mean(estimates))
    w = float(np.mean(np.square(ses)))
    b = float(np.var(estimates, ddof=1))
    t = w + (1 + 1 / m) * b
    lam = (1 + 1 / m) * b / t
    nu_m = (m - 1) / lam ** 2
    nu_obs = (df_com + 1) / (df_com + 3) * df_com * (1 - lam)
    nu = 1 / (1 / nu_m + 1 / nu_obs)
    half = float(stats.t.ppf(0.975, nu)) * math.sqrt(t)
    return q, math.sqrt(t), q - half, q + half


def test_3_under_multiple_imputation_each_model_is_pooled_by_rubins_rules(tmp_path):
    """Each model of the sequence is fit in every completed copy and pooled by Rubin's rules. The
    copies are the stage's own (the imputation is not what is tested here); the fits and the
    pooling are statsmodels and the rules written out."""
    from turbotab.core.datastore import DataStore
    from turbotab.core.methods.missing import impute_for_inference
    from turbotab.core.models.pipeline import DesignSpec, modeling_frame
    from turbotab.core.stages.modeling import read_assignment
    from turbotab.core.tests import modeling_fixtures as mf

    frame = ef.cohort(500, seed=12)
    rng = np.random.default_rng(4)
    frame.loc[rng.random(len(frame)) < 0.12, "age"] = np.nan
    frame.loc[rng.random(len(frame)) < 0.08, "fiber"] = np.nan
    roles = {c: r for c, r in ef.ROLES.items() if c != "bmi"}
    answers = {c: a for c, a in ef.ANSWERS.items() if c != "bmi"}
    st = ef.state(target="glucose", task="regression", measure="mean_difference", roles=roles,
                  answers=answers, model_1=["age", "sex"],
                  missing={"strategy": "multiple_imputation", "m": 20})
    run = ef.run(frame, tmp_path / "mi", st, fit=False)
    seq = ef.sequence(run["effects"])
    spec = DesignSpec.from_dict(run["design"].objects["spec"])
    split = mf.split_bundle(np.arange(len(frame)), holdout=0.0, seed=st.split.seed)
    with DataStore(tmp_path / "mi" / "data" / "raw.parquet", 1 << 30) as store:
        rows = modeling_frame(store, [*spec.inputs, "glucose"],
                              read_assignment(split).index.to_numpy(), outcome="glucose")
    y = rows["glucose"].to_numpy(dtype=float)
    copies = impute_for_inference(spec, rows[list(spec.inputs)], y, "regression",
                                  seed=st.split.seed)
    assert copies.m == 20
    for key, columns in (("crude", []), ("model_1", ["age", "sex"]),
                         ("model_2", ["age", "sex", "smoking", "activity"])):
        fits = []
        for X_k in copies.frames:
            completed = X_k.assign(glucose=y)
            fits.append(_ols(completed, columns))
        df_com = float(fits[0].df_resid)
        q, se, low, high = _rubin([f.params["fiber"] for f in fits], [f.bse["fiber"] for f in fits],
                                  df_com)
        row = seq[key]["effects"][0]
        assert row["estimate"] == pytest.approx(q, rel=1e-8), key
        assert row["se"] == pytest.approx(se, rel=1e-8), key
        assert (row["ci_low"], row["ci_high"]) == pytest.approx((low, high), rel=1e-6), key
        assert seq[key]["inference"]["caption"].startswith("Multiple imputation, m = 20")
