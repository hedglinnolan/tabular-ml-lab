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
from turbotab.core.tests.acceptance.r_reference import needs_r, run_r

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


# ── one primary, every family, every display (REPAIR-ESTIMAND) ───────────────


def test_3_model_2_is_the_fits_primary_and_model_3_alone_takes_fewer_rows(tmp_path):
    """BLUEPRINT §12 ruling 3: inference estimates from every analyzed row. Under complete cases,
    with `bmi` (Model 3's possible mediator) missing on some rows, the unadjusted model, Model 1 and
    Model 2 are fit on every analyzed row, so Model 2 is the fit's own primary estimate; Model 3
    alone is fit on the rows with `bmi` recorded, and the primary's adjustment is refit on those
    same rows beside it, so the two differ by `bmi` alone. Reference: statsmodels' least squares
    with HC3 standard errors on each design written out, on each model's own rows."""
    frame = ef.cohort(650, seed=24, prevalence=0.3)
    gaps = np.random.default_rng(1).random(len(frame)) < 0.07
    frame.loc[gaps, "bmi"] = np.nan
    st = ef.state(target="glucose", task="regression", measure="mean_difference",
                  model_1=["age", "sex"])
    run = ef.run(frame, tmp_path / "linear", st)
    seq = ef.sequence(run["effects"])
    kept = frame.loc[~gaps]
    for key in ("crude", "model_1", "model_2"):
        reference = _ols(frame, MODELS[key])
        row = seq[key]["effects"][0]
        assert seq[key]["n_rows"] == len(frame), key
        assert row["estimate"] == pytest.approx(reference.params["fiber"], rel=1e-8), key
        assert row["se"] == pytest.approx(reference.bse["fiber"], rel=1e-8), key
    primary = next(r for r in run["fit_raw"]["models"][0]["coefficients"] if r["feature"] == "fiber")
    assert seq["model_2"]["effects"][0]["estimate"] == pytest.approx(primary["estimate"], rel=1e-12)
    assert seq["model_2"]["effects"][0]["se"] == pytest.approx(primary["se"], rel=1e-12)
    three = seq["model_3"]
    assert three["n_rows"] == len(kept) < len(frame)
    reference = _ols(kept, MODELS["model_3"])
    assert three["effects"][0]["estimate"] == pytest.approx(reference.params["fiber"], rel=1e-8)
    assert three["effects"][0]["se"] == pytest.approx(reference.bse["fiber"], rel=1e-8)
    [same_rows] = three["comparison"]
    reference = _ols(kept, MODELS["model_2"])
    assert same_rows["feature"] == "fiber"
    assert same_rows["estimate"] == pytest.approx(reference.params["fiber"], rel=1e-8)
    assert same_rows["se"] == pytest.approx(reference.bse["fiber"], rel=1e-8)
    n, m = len(frame), len(kept)
    assert three["note"] == (f"further adjusted for `bmi`, a possible mediator: not a total effect. "
                             f"Fit on the {m:,} of the {n:,} analyzed rows with `bmi` recorded; the "
                             f"primary's adjustment refit on those rows is shown beside it.")
    assert three["comparison_label"] == (f"The primary's adjustment refit on Model 3's {m:,} rows, "
                                         f"so Model 3 differs from it by `bmi` alone.")
    assert run["effects"]["rows"] == f"all {n:,} analyzed rows"
    assert run["effects"]["methods"].startswith(
        f"The estimate of `fiber` is reported across a declared sequence of models fit on all "
        f"{n:,} analyzed rows: unadjusted; Model 1, adjusted for `age` and `sex`; Model 2, the "
        f"primary, adjusted for `age`, `sex`, `smoking` and `activity`; Model 3, further adjusted "
        f"for `bmi`, a possible mediator, so not a total effect, on the {m:,} rows with `bmi` "
        f"recorded, beside the primary's adjustment refit on those rows. ")
    # The marginal estimand is the analyzed rows', all of them: with an intercept the logistic
    # model's maximum likelihood reproduces the event's share, so the standardized risk at the
    # observed exposure is the share over every analyzed row, not over Model 3's.
    binary = ef.state(target="dm", task="binary", event="yes", measure="risk_difference")
    found = ef.run(frame, tmp_path / "binary", binary, fit=False)
    [c] = found["effects"]["families"][0]["marginal"]["contrasts"]
    everyone, fewer = (frame["dm"] == "yes").mean(), (kept["dm"] == "yes").mean()
    assert c["risk_low"] == pytest.approx(everyone, rel=1e-10) and everyone != fewer


def test_3_under_multiple_imputation_model_2_is_pooled_over_the_fits_own_copies(tmp_path):
    """With a possible mediator beside the primary, the crude model, Model 1 and Model 2 are pooled
    over the completed copies the fit itself analyzes (imputed from the primary's columns and the
    outcome, the same seed), so Model 2 is the fit's primary estimate; Model 3 is pooled over
    copies imputed with `bmi` too. Reference: statsmodels in each of the primary's copies
    (``impute_for_inference`` on the primary design, as the fit calls it), pooled by Rubin's rules
    written out (:func:`_rubin`)."""
    from turbotab.core.datastore import DataStore
    from turbotab.core.methods.missing import impute_for_inference
    from turbotab.core.models.pipeline import DesignSpec, modeling_frame
    from turbotab.core.stages.modeling import read_assignment
    from turbotab.core.tests import modeling_fixtures as mf

    frame = ef.cohort(400, seed=31)
    rng = np.random.default_rng(8)
    frame.loc[rng.random(len(frame)) < 0.12, "age"] = np.nan
    frame.loc[rng.random(len(frame)) < 0.10, "bmi"] = np.nan
    st = ef.state(target="glucose", task="regression", measure="mean_difference",
                  missing={"strategy": "multiple_imputation", "m": 20})
    run = ef.run(frame, tmp_path / "mi", st)
    seq = ef.sequence(run["effects"])
    assert list(seq) == ["crude", "model_2", "model_3"]
    spec = DesignSpec.from_dict(run["design"].objects["spec"])
    assert "bmi" not in spec.inputs
    split = mf.split_bundle(np.arange(len(frame)), holdout=0.0, seed=st.split.seed)
    with DataStore(tmp_path / "mi" / "data" / "raw.parquet", 1 << 30) as store:
        rows = modeling_frame(store, [*spec.inputs, "glucose"],
                              read_assignment(split).index.to_numpy(), outcome="glucose")
    y = rows["glucose"].to_numpy(dtype=float)
    copies = impute_for_inference(spec, rows[list(spec.inputs)], y, "regression",
                                  seed=st.split.seed)
    fits = [_ols(X_k.assign(glucose=y), MODELS["model_2"]) for X_k in copies.frames]
    q, se, _, _ = _rubin([f.params["fiber"] for f in fits], [f.bse["fiber"] for f in fits],
                         float(fits[0].df_resid))
    row = seq["model_2"]["effects"][0]
    assert row["estimate"] == pytest.approx(q, rel=1e-8)
    assert row["se"] == pytest.approx(se, rel=1e-8)
    primary = next(r for r in run["fit_raw"]["models"][0]["coefficients"] if r["feature"] == "fiber")
    assert row["estimate"] == pytest.approx(primary["estimate"], rel=1e-12)
    assert seq["model_3"]["inference"]["caption"].startswith("Multiple imputation, m = 20")
    assert seq["model_3"]["n_rows"] == len(frame) and seq["model_3"]["comparison"] is None


def _repeated(seed: int = 10) -> pd.DataFrame:
    """Two rows per person: the second a noisy repeat of the first's fiber and glucose, and a yes/no
    outcome that agrees with the first's three times in four."""
    one = ef.cohort(200, seed=seed)
    rng = np.random.default_rng(seed + 1)
    flip = rng.random(200) < 0.25
    again = np.where(flip, np.where(one["dm"] == "yes", "no", "yes"), one["dm"])
    return pd.concat([one, one.assign(glucose=one["glucose"] + rng.normal(0, 5, 200),
                                      fiber=one["fiber"] + rng.normal(0, 2, 200), dm=again)],
                     ignore_index=True)


LMER_R = """
suppressPackageStartupMessages(library(lme4))
d <- read.csv(rows_csv); d$sex <- factor(d$sex, levels = c("female", "male"))
fs <- list(crude = glucose ~ fiber, model_1 = glucose ~ fiber + age + sex,
           model_2 = glucose ~ fiber + age + sex + smoking + activity)
ctl <- lmerControl(optimizer = "bobyqa", optCtrl = list(rhobeg = 0.01, rhoend = 1e-12, maxfun = 1e5))
res <- lapply(fs, function(f) {
  m <- lmer(update(f, . ~ . + (1 | pid)), data = d, REML = TRUE, control = ctl)
  list(estimate = unname(fixef(m)["fiber"]), se = unname(sqrt(vcov(m)["fiber", "fiber"])))
})
out(res)
"""


@needs_r
def test_3_the_unadjusted_estimate_is_shown_for_every_family(tmp_path):
    """STROBE 16a, and the contract's ``crude-always`` relation ("always"): the families that model
    the unit (the random-intercept mixed model, GEE) show the unadjusted model and Model 1 as
    every other family does, each refit on the primary's model matrix restricted to its columns,
    and the methods sentence names exactly the models shown. References: R ``lme4::lmer`` (REML;
    to 1e-5, the optimizer's tolerance, measured 5e-7) for the mixed model, and statsmodels'
    ``GEE`` (exchangeable), the GEE family's own documented estimator, on each design written out,
    to 1e-8."""
    import statsmodels.api as sm

    from turbotab.core import decisions as d

    frame = _repeated()
    st = ef.state(target="glucose", task="regression", measure="mean_difference",
                  models=["mixed", "gee"], model_1=["age", "sex"],
                  grain=d.GrainSpec(grain="repeated", id_column="pid"))
    run = ef.run(frame, tmp_path / "stage", st)
    r = run_r(LMER_R, {"rows": frame}, tmp_path / "r")
    fit = {m["family"]: m for m in run["fit_raw"]["models"]}
    for i, name in enumerate(("mixed", "gee")):
        seq = ef.sequence(run["effects"], i)
        assert run["effects"]["families"][i]["family"] == name
        assert list(seq) == ["crude", "model_1", "model_2", "model_3"], name
        for key in ("crude", "model_1", "model_2"):
            row = seq[key]["effects"][0]
            assert row["feature"] == "fiber" and seq[key]["n_rows"] == len(frame)
            if name == "mixed":
                assert row["estimate"] == pytest.approx(r[key]["estimate"], rel=1e-5), key
                assert row["se"] == pytest.approx(r[key]["se"], rel=1e-5), key
            else:
                X = ef.design_matrix(frame, ["fiber", *MODELS[key]])
                ref = sm.GEE(frame["glucose"].to_numpy(float), X, groups=frame["pid"],
                             cov_struct=sm.cov_struct.Exchangeable()).fit(maxiter=200, ctol=1e-10)
                assert row["estimate"] == pytest.approx(ref.params["fiber"], rel=1e-8), key
        primary = next(c for c in fit[name]["coefficients"] if c["feature"] == "fiber")
        assert seq["model_2"]["effects"][0]["estimate"] == pytest.approx(primary["estimate"],
                                                                         rel=1e-12)
    assert run["effects"]["methods"].startswith(
        f"The estimate of `fiber` is reported across a declared sequence of models fit on all "
        f"{len(frame):,} analyzed rows: unadjusted; Model 1, adjusted for `age` and `sex`; Model 2, "
        f"the primary, adjusted for `age`, `sex`, `smoking` and `activity`; Model 3, ")


def _surveyed(n: int = 900, seed: int = 11) -> pd.DataFrame:
    """A cohort drawn with unequal weights that track the exposure, ten strata of three PSUs, a
    time to event and an ordered outcome."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.binomial(1, 0.4, n).astype(float)
    stratum = rng.integers(1, 11, n)
    psu = stratum * 10 + rng.integers(1, 4, n)
    w = np.exp(0.7 * x1 + rng.normal(0, 0.3, n)) * 1000
    hazard = np.exp(0.5 * x1 + 0.3 * x2 + 0.4 * x1 ** 2)
    T = rng.exponential(1 / hazard)
    C = rng.exponential(1.5, n)
    latent = 0.6 * x1 + 0.3 * x2 + 0.3 * x1 ** 2 + rng.logistic(size=n)
    health = np.where(latent < -0.5, "poor", np.where(latent < 1.0, "fair", "good"))
    return pd.DataFrame({"pid": np.arange(n), "x1": x1, "x2": x2, "time": np.minimum(T, C),
                         "event": (T <= C).astype(int), "health": health, "w": w,
                         "stratum": stratum, "psu": psu})


SURVEY_R = """
suppressPackageStartupMessages({library(survey); library(EValue)})
d <- read.csv(rows_csv)
d$health <- factor(d$health, levels = c("poor", "fair", "good"), ordered = TRUE)
s <- svydesign(ids = ~psu, strata = ~stratum, weights = ~w, nest = TRUE, data = d)
ctl <- coxph.control(eps = 1e-14, iter.max = 200, toler.chol = 1e-15)
c0 <- svycoxph(Surv(time, event) ~ x1, design = s, control = ctl)
c2 <- svycoxph(Surv(time, event) ~ x1 + x2, design = s, control = ctl)
o0 <- svyolr(health ~ x1, design = s, control = list(reltol = 1e-15, maxit = 10000, ndeps = rep(1e-6, 3)))
o2 <- svyolr(health ~ x1 + x2, design = s, control = list(reltol = 1e-15, maxit = 10000, ndeps = rep(1e-6, 4)))
share <- unname(coef(svymean(~event, s)))
out(list(cox = list(crude = unname(coef(c0)["x1"]), crude_se = unname(sqrt(vcov(c0)["x1", "x1"])),
                    model_2 = unname(coef(c2)["x1"]), model_2_se = unname(sqrt(vcov(c2)["x1", "x1"]))),
         olr = list(crude = unname(coef(o0)["x1"]), crude_se = unname(sqrt(vcov(o0)["x1", "x1"])),
                    model_2 = unname(coef(o2)["x1"]), model_2_se = unname(sqrt(vcov(o2)["x1", "x1"]))),
         share = share, unweighted = mean(d$event)))
"""

EVALUE_HR_R = """
library(EValue)
v <- read.csv(values_csv)
e <- suppressMessages(evalues.HR(v$hr, v$lo, v$hi, rare = as.logical(v$rare)))
out(list(point = e[2, 1], limit = if (is.na(e[2, 2])) e[2, 3] else e[2, 2]))
"""


@needs_r
def test_3_under_the_surveyed_population_every_model_is_design_based(tmp_path):
    """MODELING_SEQUENCE §0 ruling 6: the population estimand "binds every family and every
    display". Under the surveyed-population answer the Cox model's and the proportional-odds
    model's declared models are all design-based (Binder's pseudo-likelihood; the weighted
    cumulative logit), so Model 2 is the fit's design-based primary, never an unweighted refit;
    the E-value is the design-based hazard ratio's, read as rare or common by the population's
    event share; a family with no design-based estimator (feature-wise regression) is blocked and
    recorded with the fit's exits. References: R ``survey::svycoxph`` (coefficients 1e-6, standard
    errors 1e-4, as MS4 holds them), ``svyolr`` (1e-5, 1e-4), ``svymean`` for the share, and
    ``EValue::evalues.HR`` on the stage's own interval (1e-8)."""
    from turbotab.core import decisions as d

    frame = _surveyed()
    roles = {"pid": "identifier", "x1": "exposure", "x2": "covariate", "time": "time",
             "w": "design", "stratum": "design", "psu": "design"}
    amounts = {"code_or_count:x2": "amount", "code_or_count:stratum": "code",
               "code_or_count:psu": "code"}
    survey = d.SurveySpec(estimand="population", weight="w", strata="stratum", psu="psu")
    cox = ef.state(target="event", task="time_to_event", exposure="x1", measure="hazard_ratio",
                   roles=roles, answers={"x2": ef.CONFOUNDER}, event="1", models=["cox"],
                   amounts=amounts, follow_up=d.FollowUpSpec(time_column="time"), survey=survey)
    run = ef.run(frame.drop(columns=["health"]), tmp_path / "cox", cox)
    r = run_r(SURVEY_R, {"rows": frame}, tmp_path / "r")
    seq = ef.sequence(run["effects"])
    for key in ("crude", "model_2"):
        row = seq[key]["effects"][0]
        assert seq[key]["inference"]["covariance"] == "design", key
        assert row["estimate"] == pytest.approx(r["cox"][key], rel=1e-6), key
        assert row["se"] == pytest.approx(r["cox"][f"{key}_se"], rel=1e-4), key
    primary = run["fit_raw"]["models"][0]["coefficients"][0]
    shown = seq["model_2"]["effects"][0]
    assert (shown["estimate"], shown["se"]) == pytest.approx((primary["estimate"], primary["se"]),
                                                             rel=1e-12)
    [sens] = run["effects"]["families"][0]["sensitivity"]
    rare = r["share"] < 0.15  # the population's event share decides it (``svymean``)
    assert sens["e_value"]["measure"] == "HR" and sens["e_value"]["rare"] is rare
    values =pd.DataFrame([{"hr": shown["ratio"], "lo": shown["ratio_low"], "hi": shown["ratio_high"],
                            "rare": str(rare).upper()}])
    e = run_r(EVALUE_HR_R, {"values": values}, tmp_path / "e")
    assert sens["e_value"]["point"] == pytest.approx(e["point"], rel=1e-8)
    assert sens["e_value"]["limit"] == pytest.approx(e["limit"], rel=1e-8)
    assert ("Each model is design-based: weighted by `w`, with Taylor-linearized intervals over "
            "the survey's strata and PSUs, so the estimates describe the surveyed population."
            in run["effects"]["methods"])

    ordinal = ef.state(target="health", task="ordinal", exposure="x1",
                       measure="cumulative_odds_ratio", roles={k: v for k, v in roles.items()
                                                               if k != "time"},
                       answers={"x2": ef.CONFOUNDER}, models=["proportional_odds"],
                       amounts=amounts, outcome_order=["poor", "fair", "good"], survey=survey)
    found = ef.run(frame.drop(columns=["time", "event"]), tmp_path / "olr", ordinal, fit=False)
    seq = ef.sequence(found["effects"])
    for key in ("crude", "model_2"):
        row = seq[key]["effects"][0]
        assert seq[key]["inference"]["covariance"] == "design", key
        assert row["estimate"] == pytest.approx(r["olr"][key], rel=1e-5), key
        assert row["se"] == pytest.approx(r["olr"][f"{key}_se"], rel=1e-4), key

    linear = ef.state(target="x2", task="regression", exposure="x1", measure="mean_difference",
                      roles={"pid": "identifier", "x1": "exposure", "w": "design",
                             "stratum": "design", "psu": "design"},
                      answers={}, models=["featurewise"], amounts=amounts, survey=survey)
    blocked = ef.run(frame[["pid", "x1", "x2", "w", "stratum", "psu"]], tmp_path / "fw", linear,
                     fit=False)
    [only] = blocked["effects"]["families"][0]["sequence"]
    assert only["effects"] is None and "has no design-based estimator" in only["inference"]["refused"]
    exits = [e["decision"] for e in only["inference"]["exits"]]
    assert exits == [{"kind": "select_models", "models": ["linear"]},
                     {"kind": "set_survey", "estimand": "sample"}]
    assert blocked["effects"]["methods"].startswith("No estimate of `x1` is reported: Feature-wise "
                                                    "regression has no design-based estimator")
