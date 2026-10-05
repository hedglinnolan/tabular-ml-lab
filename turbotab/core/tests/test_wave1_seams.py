"""The seams between the wave-1 methods packages (WP6, WP8, WP9, WP10, WP11, WP12a–c), as the
integration merge found them.

Each package was finished and tested on its own branch from the same base. Where one package's
code reads what another changed, the merged behavior is held here: every test names the two
packages it joins and the reference it is checked against, computed by a path independent of the
code under test (statsmodels on hand-built matrices, or the definition written out).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats

from turbotab.core.decisions import EnergyAdjustment, SplitSpec
from turbotab.core.stages.modeling import design_stage, fit_stage
from turbotab.core.tests import modeling_fixtures as mf
from turbotab.core.tests.acceptance import references as ref



# These fixtures' truth (BLUEPRINT §14.3: whole numbers settle nothing by their values): `age` is
# drawn in whole years, an amount.
AGE_IS_AN_AMOUNT = {"code_or_count:age": "amount"}

def _stages(frame: pd.DataFrame, folder: Path, roles: dict[str, str], adjustment: Any, *,
            task: str = "regression", target: str = "y", holdout: float = 0.2,
            purpose: str = "inference", seed: int = 3) -> tuple[dict[str, Any], np.ndarray]:
    """Ingest, design and fit with the linear family and a holdout. Returns the fit artifact's
    first model and the training mask over the frame's rows."""
    paths = mf.ingest_frame(frame, folder)
    st = mf.state(roles=roles, target=target, task=task, models=["linear"],
                  energy_adjustment=adjustment, purpose=purpose,
                  split=SplitSpec(holdout=holdout, seed=seed, folds=5))
    split = mf.split_bundle(np.arange(len(frame)), holdout=holdout, seed=seed)
    ti = mf.target_info(task, target)
    design = design_stage(mf.context(st, {"split": split, "target_info": ti}, paths))
    fit = fit_stage(mf.context(st, {"design": design, "split": split, "target_info": ti}, paths))
    train = split.frames["assignment"]["partition"].to_numpy() == "train"
    return fit.data["models"][0], train


# ── WP6 × WP8: the all-components relative effect comes from the table's own rows ────────────


def _all_components(n: int, seed: int) -> pd.DataFrame:
    """WP6's all-components fixture (audit D/exp1_energy_models.py), at a smaller n."""
    rng = np.random.default_rng(seed)
    age = rng.normal(50, 15, n)
    size = rng.normal(0, 1, n) - 0.03 * (age - 50)
    energy = 2100 + 450 * size
    share_p = np.clip(rng.normal(0.16, 0.03, n), 0.05, 0.4)
    share_f = np.clip(rng.normal(0.34, 0.05, n), 0.1, 0.6)
    kp, kf = energy * share_p, energy * share_f
    kc = energy - kp - kf
    y = 0.010 * kp + 0.004 * kc + 0.2 * (age - 50) + rng.normal(0, 5, n)
    return pd.DataFrame({"protein_g": kp / 4, "carb_g": kc / 4, "fat_g": kf / 9,
                         "kcal": kp + kf + kc, "age": age, "y": y})


ROLES = {"protein_g": "exposure", "carb_g": "exposure", "fat_g": "exposure", "kcal": "energy",
         "age": "covariate"}
ALL = EnergyAdjustment(method="all_components", energy_column="kcal",
                       nutrients=["protein_g", "carb_g", "fat_g"])


def test_the_relative_effect_is_estimated_from_every_analyzed_row_under_inference(tmp_path):
    """WP8 estimates the inference table from every analyzed row (BLUEPRINT §12 ruling 3); WP6's
    average relative effect (Tomova et al. 2022) is a contrast of that table, so it must come from
    the same rows. Before the merge fix it was computed from the training fit with the all-rows
    clusters, which disagree in length.

    Reference: statsmodels OLS of y on the three sources in kcal and age over all rows,
    θ = 4·(β_p − w_c·β_c − w_f·β_f) with w the sources' mean shares of the remaining energy over
    all rows, and its HC3 interval written out from the definition on t(n − p).
    """
    frame = _all_components(1500, seed=11)
    model, train = _stages(frame, tmp_path, ROLES, ALL)
    assert model["coefficients_n"] == len(frame) and (~train).sum() > 0

    X = pd.DataFrame({"kp": frame["protein_g"] * 4, "kc": frame["carb_g"] * 4,
                      "kf": frame["fat_g"] * 9, "age": frame["age"]})
    fit = sm.OLS(frame["y"].to_numpy(), sm.add_constant(X)).fit()
    w_c = X["kc"].mean() / (X["kc"].mean() + X["kf"].mean())
    c = np.array([0.0, 4.0, -4.0 * w_c, -4.0 * (1 - w_c), 0.0])
    theta = float(c @ fit.params.to_numpy())
    exog = sm.add_constant(X).to_numpy(dtype=float)
    se = float(np.sqrt(c @ ref.hc3_by_definition(exog, np.asarray(fit.resid)) @ c))
    q = stats.t.ppf(0.975, len(X) - exog.shape[1])

    rows = {r["feature"]: r for r in model["coefficients"]}
    row = rows["protein_g_relative"]
    assert row["estimate"] == pytest.approx(theta, rel=1e-8)
    assert row["ci_low"] == pytest.approx(theta - q * se, rel=1e-6)
    assert row["ci_high"] == pytest.approx(theta + q * se, rel=1e-6)
    assert rows["kcal_from_protein_g"]["estimate"] == pytest.approx(fit.params["kp"], rel=1e-8)


def test_a_binary_relative_effect_is_on_the_tables_odds_ratio_scale(tmp_path):
    """WP8 puts a binary table on the odds-ratio scale (exp of the estimate and of the interval's
    ends, ME-07); WP6's relative-effect rows are part of that table, so they carry the same ratios.
    Reference: exp of the row's own per-gram estimate and interval ends."""
    frame = _all_components(1500, seed=12)
    frame["y"] = np.where(frame["y"] > frame["y"].median(), "high", "low")
    model, _ = _stages(frame, tmp_path, ROLES, ALL, task="binary")
    assert model["inference"]["scale"] == "odds_ratio"
    row = {r["feature"]: r for r in model["coefficients"]}["protein_g_relative"]
    assert row["ratio"] == pytest.approx(np.exp(row["estimate"]), rel=1e-12)
    assert row["ratio_low"] == pytest.approx(np.exp(row["ci_low"]), rel=1e-12)
    assert row["ratio_high"] == pytest.approx(np.exp(row["ci_high"]), rel=1e-12)


# ── WP8 × WP9: choosing among families under repeated k-fold ─────────────────────────────────


def test_selection_optimism_under_repeated_kfold_bootstraps_one_repeats_predictions(tmp_path):
    """WP8's bootstrap bias-corrected CV (Tsamardinos et al. 2018) resamples Π, each training row's
    out-of-fold prediction, once per row. WP9's repeated k-fold scores every row once per repeat,
    so the fit stage keeps the first repeat's predictions for Π (the repeat WP9's out-of-fold
    calibration reads too) and passes the other repeats' fits through.

    Reference: Π rebuilt here by refitting each family's pipeline on the split's first fold column
    alone (scikit-learn pipelines through ``fit_pipeline``), then the same resampling; the fit
    stage's selection must equal it. Π built from the last repeat instead gives another estimate,
    so the check discriminates.
    """
    import warnings

    from sklearn.base import clone

    from turbotab.core.decisions import GrainSpec, ProjectState
    from turbotab.core.models.inner_cv import fit_pipeline
    from turbotab.core.models.metrics import LABELS, cross_validate, fold_pairs
    from turbotab.core.models.pipeline import modeling_frame
    from turbotab.core.models.selection import OutOfFold, selection_optimism
    from turbotab.core.stages.modeling import coded_outcome, read_assignment
    from turbotab.core.stages.rows import cohort_stage, split_stage
    from turbotab.core.stages.target import target_info_stage
    from turbotab.core.tests.stage_harness import Ingested

    rng = np.random.default_rng(5)
    n = 300
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=[f"x{i}" for i in range(4)])
    risk = 1 / (1 + np.exp(-(X.to_numpy() @ np.array([0.7, -0.5, 0.2, 0.0]))))
    frame = X.assign(id=np.arange(n), event=np.where(rng.random(n) < risk, "yes", "no"))
    source = tmp_path / "t.csv"
    frame.to_csv(source, index=False)
    table = Ingested(source, tmp_path / "i")
    families = ["linear", "elastic_net"]
    st = ProjectState(lens=["clinical"], target="event", task="binary", event="yes",
                      purpose="prediction", roles={"id": "identifier", **{c: "covariate" for c in X}},
                      missing="complete_case",
                      grain=GrainSpec(grain="one_row_per_unit", id_column="id"), models=families,
                      split=SplitSpec(holdout=0.0, seed=4, folds=5, validation="repeated_kfold",
                                      repeats=3))
    info = table.run(target_info_stage, st)
    cohort = table.run(cohort_stage, st, {"target_info": info})
    split = table.run(split_stage, st, {"cohort": cohort, "target_info": info})
    design = table.run(design_stage, st, {"split": split, "target_info": info})
    fit = table.run(fit_stage, st, {"design": design, "split": split, "target_info": info})
    assert fit.data["repeats"] == 3
    selection = fit.data["selection"]
    assert selection is not None and selection["families"] == families

    # The training rows as the fit stage reads them (from the ingested store, in the split's order:
    # elastic net's inner folds are keyed by the rows' stored contents).
    assignment = read_assignment(split)
    with table.store() as store:
        # the outcome as stored (``outcome=``): the CSV's yes/no reads as True/False, which the
        # fit stage codes 1/0, and the outcome is part of each row's inner-fold key
        stored = modeling_frame(store, [*X.columns, "event"], assignment.index.to_numpy(),
                                outcome="event")
    Xs = stored[list(X.columns)]
    ys = np.asarray(coded_outcome("binary", stored["event"].to_numpy(), "yes"))
    assert set(np.unique(ys)) == {0, 1}
    pipelines = design.objects["pipelines"]

    def rebuilt(column: str) -> dict[str, Any]:
        pairs = fold_pairs(assignment[column].to_numpy().astype(int))
        oof = OutOfFold("binary", Xs, ys, pairs)
        results = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for key in families:
                wrapped = oof.wrap(key, lambda m, Xf, yf, r: fit_pipeline(m, Xf, yf))
                results[key] = cross_validate("binary", lambda _k=key: clone(pipelines[_k]), Xs, ys,
                                              pairs, fit=wrapped)
        # The best family and its CV score are the fit's own: MS6 chooses on the strictly proper
        # primary (log loss), its estimate the mean over every repeat of the comparison substrate,
        # whose first repeats are the split's own.
        cv = {m["family"]: m["compared_on"]["estimate"] for m in fit.data["models"]}
        results = {k: _Fixed(cv[k]) for k in results}
        return selection_optimism("binary", "log_loss", results, oof,
                                  {m["family"]: m["label"] for m in fit.data["models"]},
                                  LABELS["log_loss"], extras=["auc"])

    first = rebuilt("fold")
    assert selection["best"] == first["best"]
    assert selection["corrected"] == pytest.approx(first["corrected"], abs=1e-9)
    assert selection["optimism"] == pytest.approx(first["optimism"], abs=1e-9)
    assert selection["wins"] == first["wins"]
    assert selection["extras"]["auc"]["corrected"] == pytest.approx(
        first["extras"]["auc"]["corrected"], abs=1e-9)
    last = rebuilt("fold_r2")
    assert abs(last["corrected"] - selection["corrected"]) > 1e-6


class _Fixed:
    """A cross-validation result whose summary is one fixed estimate (the fit's own CV AUC)."""

    def __init__(self, estimate: float):
        self.estimate = estimate

    def summary(self, task: str) -> dict[str, Any]:
        return {"auc": {"estimate": self.estimate}, "log_loss": {"estimate": self.estimate}}


# ── WP10 × WP6 × WP8: a surveyed population ──────────────────────────────────────────────────


def _surveyed(n: int, seed: int) -> pd.DataFrame:
    """The all-components fixture drawn as a two-PSU-per-stratum survey with informative weights."""
    rng = np.random.default_rng(seed)
    frame = _all_components(n, seed)
    frame["SDMVSTRA"] = rng.integers(1, 13, n)
    frame["SDMVPSU"] = rng.integers(1, 3, n)
    frame["WTMEC2YR"] = np.round(np.where(frame["age"] > 50, 4000.0, 12000.0)
                                 * rng.uniform(0.7, 1.3, n), 1)
    return frame


def test_the_relative_effect_under_a_surveyed_population_is_design_based(tmp_path):
    """WP10 makes the linear family's table design-based under the "surveyed population" answer;
    WP8 estimates it from every analyzed row, held-out ones included; WP6's average relative effect
    is a contrast of that table. All three hold at once: the contrast is the survey-weighted one,
    its interval Taylor-linearized over the design on the design's degrees of freedom.

    Reference: survey-weighted least squares and its linearized variance written out from the
    definition over every row (``survey_references.wls_by_definition``, R survey's svyrecvar
    semantics); θ = c'β with c = 4·(e_p − w_c e_c − w_f e_f), w the sources' unweighted mean shares
    of the remaining energy (as Tomova et al. define them, over the rows the model was fit on);
    SE √(c'Vc); df = PSUs − strata (``design_df_by_definition``).
    """
    from turbotab.core.decisions import SurveySpec
    from turbotab.core.tests.acceptance import survey_references as sref

    frame = _surveyed(1500, seed=21)
    roles = {**ROLES, "WTMEC2YR": "design", "SDMVSTRA": "design", "SDMVPSU": "design"}
    paths = mf.ingest_frame(frame, tmp_path)
    survey = SurveySpec(estimand="population", weight="WTMEC2YR", strata="SDMVSTRA", psu="SDMVPSU")
    st = mf.state(roles=roles, target="y", task="regression", models=["linear"],
                  energy_adjustment=ALL, purpose="inference", survey=survey,
                  split=SplitSpec(holdout=0.2, seed=3, folds=5))
    split = mf.split_bundle(np.arange(len(frame)), holdout=0.2, seed=3)
    ti = mf.target_info("regression", "y")
    design = design_stage(mf.context(st, {"split": split, "target_info": ti}, paths))
    fit = fit_stage(mf.context(st, {"design": design, "split": split, "target_info": ti}, paths))
    model = fit.data["models"][0]
    assert model["inference"]["covariance"] == "design"
    assert model["inference"]["survey"]["n_domain"] == len(frame) == model["coefficients_n"]

    X = np.column_stack([np.ones(len(frame)), frame["protein_g"] * 4, frame["carb_g"] * 4,
                         frame["fat_g"] * 9, frame["age"]])
    beta, V = sref.wls_by_definition(X, frame["y"].to_numpy(float), frame["WTMEC2YR"].to_numpy(float),
                                     frame["SDMVSTRA"], frame["SDMVPSU"], np.ones(len(frame), bool))
    kc, kf = X[:, 2].mean(), X[:, 3].mean()
    w_c = kc / (kc + kf)
    c = np.array([0.0, 4.0, -4.0 * w_c, -4.0 * (1 - w_c), 0.0])
    theta, se = float(c @ beta), float(np.sqrt(c @ V @ c))
    df = sref.design_df_by_definition(frame["SDMVSTRA"], frame["SDMVPSU"], [True] * len(frame))
    q = stats.t.ppf(0.975, df)

    rows = {r["feature"]: r for r in model["coefficients"]}
    assert rows["kcal_from_protein_g"]["estimate"] == pytest.approx(beta[1], rel=1e-9)
    row = rows["protein_g_relative"]
    assert row["estimate"] == pytest.approx(theta, rel=1e-9)
    assert row["se"] == pytest.approx(se, rel=1e-9)
    assert row["df"] == df
    assert row["ci_low"] == pytest.approx(theta - q * se, rel=1e-8)
    assert row["ci_high"] == pytest.approx(theta + q * se, rel=1e-8)


def test_a_binary_design_based_table_is_on_the_odds_ratio_scale_and_names_its_domain(tmp_path):
    """WP8 puts every binary inference table on the odds-ratio scale with the event named; WP10's
    design-based table is one, so it carries exp(β) and the scale, and says how many rows it was
    estimated from when rows with no positive weight fall outside the domain.

    Reference: samplics ``SurveyGLM(LOGISTIC)`` on the domain rows (the rows outside it carry a
    zero weight, R survey's ``subset()`` semantics), exponentiated.
    """
    import warnings

    from turbotab.core.models import get_family
    from turbotab.core.models.inference import INDEPENDENT, Outcome
    from turbotab.core.models.survey import build_design

    frame = _surveyed(1200, seed=22)
    frame.index = pd.Index(np.arange(len(frame)), name="row_id")
    frame.loc[frame.index[:40], "WTMEC2YR"] = 0.0  # forty rows represent no one
    event = (frame["y"] > frame["y"].median()).astype(int).to_numpy()
    X = frame[["protein_g", "age"]]
    design = build_design(frame, frame["WTMEC2YR"].to_numpy(float), weight_column="WTMEC2YR",
                          strata_column="SDMVSTRA", psu_column="SDMVPSU")
    table = get_family("linear").inference_matrix(
        X, event, task="binary", classes=[0, 1], clusters=INDEPENDENT,
        outcome=Outcome(name="y_high", labels={1: "high", 0: "low"}), rows="all", survey=design)
    assert table.info["covariance"] == "design" and table.info["scale"] == "odds_ratio"
    assert table.info["event"] == "high" and table.info["reference"] == "low"
    assert table.info["n_rows"] == len(frame) - 40
    assert f"Estimated from the {len(frame) - 40:,} of the {len(frame):,} analyzed rows" in \
        table.info["caption"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from samplics.regression import SurveyGLM
        from samplics.utils.types import ModelType

        glm = SurveyGLM(model=ModelType.LOGISTIC)
        glm.estimate(y=event.astype(float), x=X.to_numpy(float),
                     samp_weight=frame["WTMEC2YR"].to_numpy(float),
                     stratum=frame["SDMVSTRA"].to_numpy(), psu=frame["SDMVPSU"].to_numpy(),
                     add_intercept=True)
    beta = np.asarray(glm.beta["point_est"])
    rows = {r["feature"]: r for r in table.rows}
    assert rows["(intercept)"]["ratio"] is None
    for j, name in enumerate(["protein_g", "age"], start=1):
        assert rows[name]["estimate"] == pytest.approx(beta[j], rel=1e-7)
        assert rows[name]["ratio"] == pytest.approx(np.exp(beta[j]), rel=1e-7)
        assert rows[name]["ratio_low"] == pytest.approx(np.exp(rows[name]["ci_low"]), rel=1e-12)


# ── WP11 × WP8: the feature-wise tests under inference ───────────────────────────────────────


def test_feature_wise_tests_use_every_analyzed_row_and_its_clusters(tmp_path):
    """WP11's feature-wise family has its own entry in the fit (it makes no predictions); WP8
    estimates every inference table from every analyzed row and clusters by the unit that repeats
    among them. Joined, the feature-wise table must take those rows and those clusters: before the
    merge fix it took the training rows beside the all-row clusters and could not be computed.

    Reference: statsmodels OLS of the outcome on one exposure and the covariate over every
    analyzed row (the held-out people included), and CR2 with Bell–McCaffrey degrees of freedom
    written out from the definition (``references.cr2_by_definition``) by person.
    """
    frame = mf.nhanes_like(240, seed=31, repeats=2)
    paths = mf.ingest_frame(frame, tmp_path)
    roles = {"SEQN": "identifier", "age": "covariate", "protein": "exposure", "carb": "exposure"}
    st = mf.state(roles=roles, target="glucose", task="regression", models=["featurewise"],
                  energy_adjustment=None, purpose="inference")
    split = mf.split_bundle(np.arange(len(frame)), holdout=0.25, seed=5,
                            groups=frame["SEQN"].to_numpy(), grouped_by="SEQN")
    ti = mf.target_info("regression", "glucose")
    design = design_stage(mf.context(st, {"split": split, "target_info": ti}, paths))
    fit = fit_stage(mf.context(st, {"design": design, "split": split, "target_info": ti}, paths))
    model = fit.data["models"][0]
    assert not any("could not be computed" in c for c in model["concerns"]), model["concerns"]
    assert model["coefficients_n"] == len(frame) and fit.data["n_holdout"] > 0
    assert model["inference"]["covariance"] == "CR2" and model["inference"]["rows"] == "all"

    X = sm.add_constant(frame[["protein", "age"]].astype(float))
    ols = sm.OLS(frame["glucose"].to_numpy(float), X).fit()
    codes = pd.factorize(frame["SEQN"])[0]
    V, df = ref.cr2_by_definition(X.to_numpy(), np.asarray(ols.resid), codes)
    row = {r["feature"]: r for r in model["coefficients"]}["protein"]
    assert row["estimate"] == pytest.approx(ols.params["protein"], rel=1e-8)
    assert row["se"] == pytest.approx(float(np.sqrt(V[1, 1])), rel=1e-6)
    assert row["df"] == pytest.approx(df[1], rel=1e-6)


# ── WP12a × WP8 × WP9: an ordered outcome ────────────────────────────────────────────────────


def _ordered(n: int, seed: int) -> pd.DataFrame:
    """Three predictors and a five-level outcome cut from a logistic latent variable."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({"age": rng.normal(50, 10, n), "fiber_g": rng.gamma(3, 5, n),
                          "male": rng.integers(0, 2, n).astype(float)})
    latent = (0.03 * frame["age"] - 0.05 * frame["fiber_g"] + 0.4 * frame["male"]
              + rng.logistic(size=n))
    levels = np.array(["none", "mild", "moderate", "marked", "severe"])
    frame["grade"] = levels[np.digitize(latent, (0.5, 1.5, 2.5, 3.5))]
    return frame


ORDER = ["none", "mild", "moderate", "marked", "severe"]
ORDINAL_ROLES = {"age": "covariate", "fiber_g": "exposure", "male": "covariate"}


def test_an_ordinal_table_is_on_the_cumulative_odds_ratio_scale_from_every_analyzed_row(tmp_path):
    """WP12a's proportional-odds family under WP8's inference contract: the table is estimated from
    every analyzed row (a holdout drawn), and declares its scale, each coefficient carrying its
    cumulative odds ratio, the cut-points none.

    Reference: statsmodels ``OrderedModel`` (logit) on every analyzed row, the held-out ones
    included; the ratios are exp of the row's own estimate and interval ends.
    """
    import warnings

    from statsmodels.miscmodels.ordinal_model import OrderedModel

    frame = _ordered(700, seed=41)
    paths = mf.ingest_frame(frame, tmp_path)
    st = mf.state(roles=ORDINAL_ROLES, target="grade", task="ordinal", purpose="inference",
                  models=["proportional_odds"], outcome_order=ORDER, energy_adjustment=None,
                  split=SplitSpec(holdout=0.2, seed=2, folds=5))
    split = mf.split_bundle(np.arange(len(frame)), holdout=0.2, seed=2)
    ti = mf.target_info("ordinal", "grade")
    design = design_stage(mf.context(st, {"split": split, "target_info": ti}, paths))
    fit = fit_stage(mf.context(st, {"design": design, "split": split, "target_info": ti}, paths))
    model = fit.data["models"][0]
    assert fit.data["n_holdout"] > 0 and model["coefficients_n"] == len(frame)
    info = model["inference"]
    assert info["scale"] == "odds_ratio" and info["axis"] == "log" and info["rows"] == "all"
    assert "Cumulative odds ratio of a higher level of `grade`" in info["effect"]

    codes = frame["grade"].map({lv: k for k, lv in enumerate(ORDER)}).to_numpy()
    X = frame[["age", "fiber_g", "male"]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reference = OrderedModel(codes, X, distr="logit").fit(method="bfgs", maxiter=20_000,
                                                              gtol=1e-10, disp=False)
    rows = {r["feature"]: r for r in model["coefficients"]}
    for name in ("age", "fiber_g", "male"):
        assert rows[name]["estimate"] == pytest.approx(reference.params[name], abs=1e-6)
        assert rows[name]["ratio"] == pytest.approx(np.exp(rows[name]["estimate"]), rel=1e-12)
        assert rows[name]["ratio_high"] == pytest.approx(np.exp(rows[name]["ci_high"]), rel=1e-12)
    cuts = [r for f, r in rows.items() if f.startswith("(cut-point")]
    assert len(cuts) == 4 and all(r["ratio"] is None for r in cuts)


def test_choosing_among_families_on_an_ordinal_outcome_bootstraps_its_concordance(tmp_path):
    """WP8's selection optimism with WP12a's ordered outcome and WP9's repeated k-fold: the
    families are chosen on the ranked probability score (MS6: the strictly proper primary; C,
    WP12a's earlier primary, is the customary headline, corrected for the same choice), and the
    bootstrap scores the pooled out-of-fold predictions. Before the merge fix the pooled score had
    no ordinal branch, so fitting two families on an ordered outcome raised.

    Reference: the pooled score the bootstrap uses, checked against lifelines'
    ``concordance_index`` (an independent implementation of Harrell's C) on the predicted mean
    level of every row; and the selection's own consistency (best family = highest CV C, the
    optimism its CV C less the corrected estimate, every resample won by one family).
    """
    import warnings

    from lifelines.utils import concordance_index

    from turbotab.core.models.selection import pooled_score

    frame = _ordered(500, seed=42)
    paths = mf.ingest_frame(frame, tmp_path)
    st = mf.state(roles=ORDINAL_ROLES, target="grade", task="ordinal", purpose="prediction",
                  models=["proportional_odds", "linear"], outcome_order=ORDER,
                  energy_adjustment=None,
                  split=SplitSpec(holdout=0.0, seed=2, folds=5, validation="repeated_kfold",
                                  repeats=2))
    split = mf.split_bundle(np.arange(len(frame)), holdout=0.0, seed=2)
    split.frames["assignment"]["fold_r1"] = np.random.default_rng(9).permutation(
        split.frames["assignment"]["fold"].to_numpy())
    split.data["validation"] = "repeated_kfold"
    ti = mf.target_info("ordinal", "grade")
    design = design_stage(mf.context(st, {"split": split, "target_info": ti}, paths))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = fit_stage(mf.context(st, {"design": design, "split": split, "target_info": ti}, paths))
    assert fit.data["primary_metric"] == "rps" and fit.data["repeats"] == 2
    assert fit.data["headline_metric"] == "c_index"
    selection = fit.data["selection"]
    assert selection is not None and selection["metric"] == "rps"
    assert selection["extras"]["c_index"]["corrected"] is not None
    cv = {m["family"]: m["compared_on"]["estimate"] for m in fit.data["models"]}
    assert selection["best"] == min(cv, key=cv.get) and selection["cv"] == min(cv.values())
    assert selection["optimism"] == pytest.approx(selection["corrected"] - selection["cv"], abs=1e-12)
    assert sum(selection["wins"].values()) == selection["replicates"]

    rng = np.random.default_rng(3)
    y = rng.integers(0, 5, 400)
    proba = rng.dirichlet(np.ones(5), 400) * 0.5 + np.eye(5)[y] * 0.5
    expected = proba @ np.arange(5)
    assert pooled_score("ordinal", "c_index", y, proba, np.full(400, np.nan), list(range(5))) == \
        pytest.approx(concordance_index(y, expected), abs=1e-12)


# ── WP12a × WP6 × WP8 × WP10: a formed exposure ──────────────────────────────────────────────


def _protein_frame(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    kcal = rng.normal(2100, 450, n)
    protein = (0.16 + rng.normal(0, 0.03, n)) * kcal / 4
    age = rng.uniform(20, 80, n)
    y = 5 + 0.03 * age + 0.002 * kcal + 1.5 * np.log(protein) + rng.normal(0, 1.0, n)
    return pd.DataFrame({"protein_g": protein, "kcal": kcal, "age": age, "y": y})


PROTEIN_ROLES = {"protein_g": "exposure", "kcal": "energy", "age": "covariate"}
STANDARD = EnergyAdjustment(method="standard", energy_column="kcal", nutrients=["protein_g"])


def _quintile_fit(frame: pd.DataFrame, folder: Path, **slots: Any) -> dict[str, Any]:
    from turbotab.core.decisions import ExposureFormSpec

    paths = mf.ingest_frame(frame, folder)
    roles = slots.pop("roles", PROTEIN_ROLES)
    st = mf.state(roles=roles, target="y", task="regression", models=["linear"],
                  energy_adjustment=STANDARD, purpose="inference",
                  exposure_forms={"protein_g": ExposureFormSpec(form="quintiles")},
                  split=SplitSpec(holdout=0.2, seed=6, folds=5), **slots)
    split = mf.split_bundle(np.arange(len(frame)), holdout=0.2, seed=6)
    ti = mf.target_info("regression", "y")
    design = design_stage(mf.context(st, {"split": split, "target_info": ti}, paths))
    fit = fit_stage(mf.context(st, {"design": design, "split": split, "target_info": ti}, paths))
    return {"design": design.data, "model": fit.data["models"][0], "n_holdout": fit.data["n_holdout"]}


def _quintile_score(x: pd.Series) -> pd.Series:
    group = pd.qcut(x, 5, labels=False)
    return group.map(x.groupby(group).median()).astype(float)


def test_a_quintile_exposure_keeps_its_energy_meaning_and_its_trend_uses_every_row(tmp_path):
    """WP6 reads each coefficient's meaning off the model matrix; WP12a turns an exposure into
    quintile indicators, so the column the meaning was read for is gone from the matrix (before
    the merge fix the design stage failed on it). Each indicator now carries the energy model's
    meaning, quintile by quintile. WP8 estimates the table, and so the trend test WP12a refits,
    from every analyzed row.

    Reference: statsmodels OLS with HC3 on every analyzed row (the held-out ones included) of y on
    total energy, age and protein scored by its quintile's median, the quintiles cut by
    ``pandas.qcut`` on those rows.
    """
    frame = _protein_frame(1000, seed=51)
    out = _quintile_fit(frame, tmp_path)
    model = out["model"]
    assert out["n_holdout"] > 0 and model["coefficients_n"] == len(frame)
    rows = {r["feature"]: r for r in model["coefficients"]}
    for g in range(2, 6):
        meaning = rows[f"protein_g_Q{g}"]["meaning"]
        assert meaning.endswith(f": quintile {g} against the lowest") and "total energy fixed" in meaning

    X = pd.DataFrame({"kcal": frame["kcal"], "age": frame["age"],
                      "score": _quintile_score(frame["protein_g"])})
    reference = sm.OLS(frame["y"].to_numpy(), sm.add_constant(X)).fit(cov_type="HC3", use_t=True)
    (trend,) = [t for t in model["exposure_tests"] if t["test"] == "trend"]
    assert trend["estimate"] == pytest.approx(float(reference.params["score"]), rel=1e-8)
    assert trend["p"] == pytest.approx(float(reference.pvalues["score"]), rel=1e-6, abs=1e-300)


def test_under_a_surveyed_population_the_trend_is_design_based(tmp_path):
    """WP10 makes the table design-based under the population answer; WP12a's quintile trend is a
    refit of that table, so it is design-based too, on the design's degrees of freedom.

    Reference: survey-weighted least squares with its linearized variance written out from the
    definition (``survey_references.wls_by_definition``) over every row, on t(PSUs − strata).
    """
    from turbotab.core.decisions import SurveySpec
    from turbotab.core.tests.acceptance import survey_references as sref

    frame = _protein_frame(900, seed=52)
    rng = np.random.default_rng(53)
    frame["SDMVSTRA"] = rng.integers(1, 11, len(frame))
    frame["SDMVPSU"] = rng.integers(1, 3, len(frame))
    frame["WTMEC2YR"] = np.round(rng.uniform(2000, 15000, len(frame)), 1)
    roles = {**PROTEIN_ROLES, "SDMVSTRA": "design", "SDMVPSU": "design", "WTMEC2YR": "design"}
    survey = SurveySpec(estimand="population", weight="WTMEC2YR", strata="SDMVSTRA", psu="SDMVPSU")
    model = _quintile_fit(frame, tmp_path, roles=roles, survey=survey)["model"]
    assert model["inference"]["covariance"] == "design"

    X = np.column_stack([np.ones(len(frame)), frame["kcal"], frame["age"],
                         _quintile_score(frame["protein_g"])])
    beta, V = sref.wls_by_definition(X, frame["y"].to_numpy(float), frame["WTMEC2YR"].to_numpy(float),
                                     frame["SDMVSTRA"], frame["SDMVPSU"], np.ones(len(frame), bool))
    df = sref.design_df_by_definition(frame["SDMVSTRA"], frame["SDMVPSU"], [True] * len(frame))
    (trend,) = [t for t in model["exposure_tests"] if t["test"] == "trend"]
    assert trend["estimate"] == pytest.approx(beta[3], rel=1e-8)
    se = float(np.sqrt(V[3, 3]))
    assert trend["p"] == pytest.approx(2 * stats.t.sf(abs(beta[3] / se), df), rel=1e-6)
    assert trend["df_den"] == df


# ── WP12b × WP8 × WP9: Cox, the mixed model ──────────────────────────────────────────────────


def _wp12b_stages(frame: pd.DataFrame, st: Any, predictors: list[str], task: str,
                  folder: Path) -> tuple[Any, Any, Any]:
    """The real split, design and fit stages over these rows (as WP12b's acceptance tests run them)."""
    from turbotab.core.stages.rows import split_stage

    paths = mf.ingest_frame(frame, folder)
    ids = np.arange(len(frame))
    ti = mf.target_info(task, st.target)
    split = split_stage(mf.context(st, {"cohort": mf.cohort_bundle(ids, predictors),
                                        "target_info": ti}, paths))
    design = design_stage(mf.context(st, {"split": split, "target_info": ti}, paths))
    fit = fit_stage(mf.context(st, {"design": design, "split": split, "target_info": ti}, paths))
    return split, design, fit


def test_a_cox_table_with_a_holdout_is_estimated_from_every_analyzed_row(tmp_path):
    """WP12b's Cox model under WP8's inference contract, with a holdout drawn: the table is
    estimated from every analyzed row, and WP9's sealed held-out detail is written without an
    interval it cannot compute for a C-index (and without calibration).

    Reference: lifelines ``CoxPHFitter`` (Efron ties) on every analyzed row, held-out ones
    included, estimates and standard errors to 10⁻⁶ relative.
    """
    from turbotab.core.decisions import FollowUpSpec, GrainSpec, ProjectState
    from turbotab.core.seal import SEALED_DETAIL, details_by_family
    from turbotab.core.tests.acceptance.test_wp12b_cox_mixed_gee import (
        _lifelines,
        _staggered_entry_cohort,
    )

    frame = _staggered_entry_cohort().iloc[:1200].reset_index(drop=True)
    st = ProjectState(lens=["clinical"], target="cvd_event", task="time_to_event",
                      purpose="inference", missing="complete_case",
                      roles={"participant_id": "identifier", "fiber_g": "exposure",
                             "age": "covariate", "followup_years": "time"},
                      split=SplitSpec(holdout=0.2, seed=0, folds=5), models=["cox"],
                      grain=GrainSpec(grain="one_row_per_unit"),
                      follow_up=FollowUpSpec(time_column="followup_years"))
    _, _, fit = _wp12b_stages(frame, st, ["fiber_g", "age"], "time_to_event", tmp_path)
    assert fit.data["n_holdout"] > 0
    model = fit.data["models"][0]
    assert model["coefficients_n"] == len(frame) and model["inference"]["rows"] == "all"
    reference = _lifelines(frame, "followup_years", "cvd_event", ["fiber_g", "age"])
    rows = {r["feature"]: r for r in model["coefficients"]}
    for name in ("fiber_g", "age"):
        assert rows[name]["estimate"] == pytest.approx(reference.params_[name], rel=1e-6)
        assert rows[name]["se"] == pytest.approx(reference.standard_errors_[name], rel=1e-6)
    detail = details_by_family(fit.frames[SEALED_DETAIL])["cox"]
    # MS6: the held-out rows carry the Brier score at the horizon with its interval, and its
    # calibration by the horizon; the binary-and-numeric calibration does not apply.
    assert set(detail["intervals"]) == {"brier_t"} and detail["calibration"] is None
    assert detail["calibration_horizon"] is not None or detail["calibration_note"]


def test_a_mixed_model_table_with_a_holdout_uses_every_analyzed_row_and_its_units(tmp_path):
    """WP12b's random-intercept model is told each row's unit; WP8 refits the table on every
    analyzed row. Joined, the table's fit is told the units of every analyzed row (before the merge
    fix the units were indexed by the training rows alone).

    Reference: statsmodels ``MixedLM`` by REML on every analyzed row, the fixed effects to 10⁻⁴
    relative (statsmodels' numerical optimum)."""
    import warnings

    from turbotab.core.decisions import GrainSpec, ProjectState

    rng = np.random.default_rng(61)
    units, per = 30, 8
    pid = np.repeat(np.arange(units) + 5001, per)
    sodium = np.repeat(rng.normal(0, 1, units), per) + rng.normal(0, 0.5, units * per)
    age = np.repeat(rng.integers(30, 70, units), per).astype(float)
    sbp = (110 + 2.0 * sodium + 0.2 * age + np.repeat(rng.normal(0, 4, units), per)
           + rng.normal(0, 2, units * per))
    frame = pd.DataFrame({"participant_id": pid, "sodium": sodium, "age": age, "sbp": sbp})
    st = ProjectState(lens=["clinical"], target="sbp", task="regression", purpose="inference",
                      roles={"participant_id": "identifier", "sodium": "exposure", "age": "covariate"},
                      missing="complete_case", split=SplitSpec(holdout=0.2, seed=1, folds=5),
                      models=["mixed"], grain=GrainSpec(grain="repeated", id_column="participant_id"),
                      shape_confirmations=AGE_IS_AN_AMOUNT)
    _, _, fit = _wp12b_stages(frame, st, ["sodium", "age"], "regression", tmp_path)
    assert fit.data["n_holdout"] > 0
    model = fit.data["models"][0]
    assert not any("could not be computed" in c for c in model["concerns"]), model["concerns"]
    assert model["coefficients_n"] == len(frame)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reference = sm.MixedLM(frame["sbp"].to_numpy(float),
                               sm.add_constant(frame[["sodium", "age"]].astype(float)),
                               groups=frame["participant_id"]).fit(reml=True)
    rows = {r["feature"]: r for r in model["coefficients"]}
    for name in ("sodium", "age"):
        assert rows[name]["estimate"] == pytest.approx(reference.fe_params[name], rel=1e-4)


def test_the_bootstrap_optimism_of_a_mixed_model_refits_a_mixed_model_on_each_resample(tmp_path):
    """WP9's bootstrap optimism correction refits the pipeline on each resample of whole units;
    WP12b's mixed model needs each row's unit, and the k-th copy of a unit is a unit of its own.
    Before the merge fix the resample fits were told no units and silently fit least squares.

    Reference: the resamples replayed as WP9's ``optimism_bootstrap`` documents them
    (``numpy.random.default_rng(seed).integers(0, G, G)`` over the units in order of first
    appearance), each fit by statsmodels ``MixedLM`` (REML, every copy of a unit its own group);
    the optimism of R² is the mean over resamples of R² on the resample less R² on the training
    rows, each against the resample's mean."""
    import warnings

    from turbotab.core.decisions import GrainSpec, ProjectState
    from turbotab.core.stages.modeling import read_assignment

    rng = np.random.default_rng(62)
    units, per = 12, 20  # enough units for the split to group by them (the seal's floor is 8)
    pid = np.repeat(np.arange(units) + 7001, per)
    sodium = np.repeat(rng.normal(0, 1, units), per) + rng.normal(0, 0.4, units * per)
    age = np.repeat(rng.integers(30, 70, units), per).astype(float)
    sbp = (110 + 2.0 * sodium + 0.1 * age + np.repeat(rng.normal(0, 3, units), per)
           + rng.normal(0, 2, units * per))
    frame = pd.DataFrame({"participant_id": pid, "sodium_mg": sodium * 500 + 3000, "age": age,
                          "sbp": sbp})
    n_boot = 20
    st = ProjectState(lens=["clinical"], target="sbp", task="regression", purpose="prediction",
                      roles={"participant_id": "identifier", "sodium_mg": "exposure",
                             "age": "covariate"}, missing="complete_case",
                      split=SplitSpec(holdout=0.0, seed=3, folds=5, validation="bootstrap",
                                      n_boot=n_boot),
                      models=["mixed"], grain=GrainSpec(grain="repeated", id_column="participant_id"),
                      shape_confirmations=AGE_IS_AN_AMOUNT)
    split, _, fit = _wp12b_stages(frame, st, ["sodium_mg", "age"], "regression", tmp_path)
    optimism = fit.data["models"][0]["optimism"]
    assert optimism is not None and optimism["refused"] is None
    assert optimism["n_ok"] == n_boot and optimism["resampled"] == "participant_id units"

    rows = read_assignment(split).index.to_numpy()
    X = sm.add_constant(frame.loc[rows, ["sodium_mg", "age"]].astype(float)).to_numpy()
    y = frame.loc[rows, "sbp"].to_numpy(float)
    labels = frame.loc[rows, "participant_id"].astype(str).to_numpy()
    codes = pd.factorize(labels)[0]
    rows_of = [np.flatnonzero(codes == u) for u in range(codes.max() + 1)]
    draws = np.random.default_rng(3)
    gaps = []
    for _ in range(n_boot):
        draw = draws.integers(0, len(rows_of), len(rows_of))
        take = np.concatenate([rows_of[u] for u in draw])
        copies = np.concatenate([np.full(len(rows_of[u]), k) for k, u in enumerate(draw)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            beta = sm.MixedLM(y[take], X[take], groups=copies).fit(reml=True).fe_params
        ref = float(y[take].mean())

        def r2(yy: np.ndarray, pred: np.ndarray) -> float:
            return 1 - float(((yy - pred) ** 2).sum()) / float(((yy - ref) ** 2).sum())

        gaps.append(r2(y[take], X[take] @ beta) - r2(y, X @ beta))
    assert optimism["estimates"]["r2"]["optimism"] == pytest.approx(float(np.mean(gaps)), abs=2e-4)


# ── WP12c × WP10 × WP8: sensitivity analyses of a surveyed population ────────────────────────


def test_sensitivity_analyses_of_a_surveyed_population_are_design_based_domains(tmp_path_factory):
    """WP12c refits the primary's model on each sensitivity analysis's rows; under WP10's
    "surveyed population" answer the primary is design-based over every analyzed row (WP8). Joined,
    each analysis is a domain of the one design: estimated with the weights on its own rows, its
    variance over every stratum and PSU of the design (NHANES Analytic Guidelines 2011–2016
    §3.2.3), and the primary's estimate is the fit's. Before the merge fix the sensitivity analyses
    were unweighted HC3 fits beside a design-based primary.

    Reference: survey-weighted least squares with R survey's linearized variance written out from
    the definition (``survey_references.wls_by_definition``) with the rows outside the analysis
    weighted zero, on t(PSUs − strata holding analysis rows) (``design_df_by_definition``).
    """
    from turbotab.core.tests.acceptance import survey_references as sref
    from turbotab.core.tests.acceptance import test_wp10_survey_design as w10
    from turbotab.server.tests.conftest import make_client, wait_for

    folder = tmp_path_factory.mktemp("seam_survey_sensitivity")
    path = w10.informative_tables(folder)["F5"]
    frame = pd.read_csv(path)
    rule = {"kind": "range", "column": "DR1TKCAL", "low": 1500, "high": 2700,
            "reason": "a narrower energy range"}
    with make_client(folder / "home", "local", 2, "http://127.0.0.1") as client:
        pid = w10.open_project(client, path, "LBXCRP", "inference", w10.DIET_ROLES)
        w10.accepted(client, pid, w10.survey_options(client, pid)[0]["decision"])
        w10.accepted(client, pid, {"kind": "set_exclusions", "rules": []})
        w10.accepted(client, pid, {"kind": "set_missing", "strategy": "complete_case"})
        w10.accepted(client, pid, {"kind": "set_split", "holdout": 0.2, "seed": 0, "folds": 5})
        w10.accepted(client, pid, {"kind": "select_models", "models": ["linear"]})
        w10.accepted(client, pid, {"kind": "set_sensitivity",
                                   "analyses": [{"label": "1,500–2,700 kcal", "rules": [rule]}]})
        wait_for(client, pid, {"fit": "fresh", "sensitivity": "fresh"}, timeout=240)
        sensitivity = client.get(f"/api/projects/{pid}/stages/sensitivity").json()["artifact"]
        primary_fit = w10.linear_model(client, pid)

    fits = sensitivity["families"][0]["fits"]
    assert [f["label"] for f in fits] == ["Primary", "1,500–2,700 kcal"]
    Xc = np.column_stack([np.ones(len(frame)), frame[["DR1TKCAL", "DR1TFIBE"]].to_numpy(float)])
    y, w = frame["LBXCRP"].to_numpy(float), frame["WTDRD1"].to_numpy(float)
    every = np.ones(len(frame), bool)
    narrower = ((frame["DR1TKCAL"] >= 1500) & (frame["DR1TKCAL"] <= 2700)).to_numpy()
    for fit_, domain in zip(fits, (every, narrower)):
        assert fit_["n_rows"] == int(domain.sum()) and fit_["inference"]["covariance"] == "design"
        beta, V = sref.wls_by_definition(Xc, y, w, frame["SDMVSTRA"], frame["SDMVPSU"], domain)
        df = sref.design_df_by_definition(frame["SDMVSTRA"], frame["SDMVPSU"], domain)
        got = next(r for r in fit_["coefficients"] if r["feature"] == "DR1TFIBE")
        assert got["estimate"] == pytest.approx(beta[2], rel=1e-9)
        assert got["se"] == pytest.approx(float(np.sqrt(V[2, 2])), rel=1e-9)
        assert got["df"] == df
    primary = next(r for r in fits[0]["coefficients"] if r["feature"] == "DR1TFIBE")
    assert primary["estimate"] == pytest.approx(w10.row(primary_fit, "DR1TFIBE")["estimate"], rel=1e-12)
