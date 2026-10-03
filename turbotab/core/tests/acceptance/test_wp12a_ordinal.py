"""WP12a · The proportional-odds family (AUDIT_REPORT §5 WP12, test 2).

Closes ME-19 ("No ordinal outcome model") and supplies the ordinal family RO-10 points to. §5 WP12
acceptance test 2: "Proportional-odds model matches statsmodels ``OrderedModel`` or R
``MASS::polr``." Both are matched here, on paths that share nothing with the engine
(``turbotab/core/models/ordinal.py``):

* **R ``MASS::polr``, published output.** The ``polr`` help page's example,
  ``polr(Sat ~ Infl + Type + Cont, weights = Freq, data = housing)``, prints (MASS reference
  manual, as rendered at haoen-cui.github.io/SOA-Exam-PA-R-Package-Documentation/MASS/reference/
  polr.html, read 2026-10-02):

      Coefficients:
         InflMedium      InflHigh TypeApartment    TypeAtrium   TypeTerrace      ContHigh
          0.5663937     1.2888191    -0.5723501    -0.3661866    -1.0910149     0.3602841
      Intercepts:
       Low|Medium Medium|High
       -0.4961353   0.6907083
      Residual Deviance: 3479.149

  and ``summary(house.plr, digits = 3)`` gives the standard errors 0.1047, 0.1272, 0.1192, 0.1552,
  0.1515, 0.0955 and 0.125, 0.125. The ``housing`` table (Madsen 1976; 72 cells, 1,681 people) is
  written out below from MASS's data, as Rdatasets serves it (``csv/MASS/housing.csv``).
* **statsmodels ``OrderedModel``** (``distr="logit"``), fit by its own optimizer: coefficients,
  cut-points (statsmodels stores the first cut-point and log increments; it converts them with
  ``transform_threshold_params``), standard errors (the cut-points' by the delta method from
  statsmodels' covariance) and, with ``cov_type="cluster"``, the cluster-robust standard errors.
* **The Brant check** has no Python reference; it is held to its size and its power by simulation,
  and its per-cut-point slopes are statsmodels ``Logit`` fits of 1{Y > j}.
* **Scores.** Harrell's C against lifelines' ``concordance_index``; the ranked probability score
  against its definition (Epstein 1969) summed level by level.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from lifelines.utils import concordance_index
from statsmodels.miscmodels.ordinal_model import OrderedModel

from turbotab.core.models import get_family, info, rank
from turbotab.core.models.artifacts import FitArtifact
from turbotab.core.models.base import ORDER_BLIND, Situation
from turbotab.core.models.inference import INDEPENDENT
from turbotab.core.models.linear import as_clusters
from turbotab.core.models.metrics import concordance, ordinal_scores
from turbotab.core.models.ordinal import (
    ProportionalOddsRegression,
    brant_test,
    ordinal_outcome,
    ordinal_table,
)
from turbotab.core.stages.modeling import design_stage, fit_stage
from turbotab.core.tests import modeling_fixtures as mf

# ── MASS::housing and polr's published fit ───────────────────────────────────

# Freq for Sat (Low, Medium, High) within Infl (Low, Medium, High) within Type (Tower, Apartment,
# Atrium, Terrace) within Cont (Low, High): MASS's row order.
HOUSING_FREQ = [
    21, 21, 28, 34, 22, 36, 10, 11, 36, 61, 23, 17, 43, 35, 40, 26, 18, 54,
    13, 9, 10, 8, 8, 12, 6, 7, 9, 18, 6, 7, 15, 13, 13, 7, 5, 11,
    14, 19, 37, 17, 23, 40, 3, 5, 23, 78, 46, 43, 48, 45, 86, 15, 25, 62,
    20, 23, 20, 10, 22, 24, 7, 10, 21, 57, 23, 13, 31, 21, 13, 5, 6, 13,
]
POLR_COEF = {"InflMedium": 0.5663937, "InflHigh": 1.2888191, "TypeApartment": -0.5723501,
             "TypeAtrium": -0.3661866, "TypeTerrace": -1.0910149, "ContHigh": 0.3602841}
POLR_CUTS = [-0.4961353, 0.6907083]
POLR_DEVIANCE = 3479.149
POLR_SE = {"InflMedium": 0.1047, "InflHigh": 0.1272, "TypeApartment": 0.1192,
           "TypeAtrium": 0.1552, "TypeTerrace": 0.1515, "ContHigh": 0.0955}
POLR_CUT_SE = [0.125, 0.125]
LEVELS = {"Sat": ["Low", "Medium", "High"], "Infl": ["Low", "Medium", "High"],
          "Type": ["Tower", "Apartment", "Atrium", "Terrace"], "Cont": ["Low", "High"]}


def _housing() -> pd.DataFrame:
    cells = [{"Sat": s, "Infl": i, "Type": t, "Cont": c}
             for c in LEVELS["Cont"] for t in LEVELS["Type"] for i in LEVELS["Infl"]
             for s in LEVELS["Sat"]]
    frame = pd.DataFrame(cells)
    frame["Freq"] = HOUSING_FREQ
    return frame


def _r_coded(frame: pd.DataFrame) -> pd.DataFrame:
    """R's treatment contrasts: the first level of each factor is the reference."""
    return pd.DataFrame({name: (frame[factor] == level).astype(float)
                         for name, factor, level in [
                             ("InflMedium", "Infl", "Medium"), ("InflHigh", "Infl", "High"),
                             ("TypeApartment", "Type", "Apartment"), ("TypeAtrium", "Type", "Atrium"),
                             ("TypeTerrace", "Type", "Terrace"), ("ContHigh", "Cont", "High")]})


def _people(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame.index.repeat(frame["Freq"]), ["Sat", "Infl", "Type", "Cont"]].reset_index(drop=True)


def test_2_housing_counts_are_madsens_1681_people():
    frame = _housing()
    assert len(frame) == 72 and frame["Freq"].sum() == 1681


@pytest.mark.parametrize("weighted", [True, False], ids=["frequency-weights", "one-row-each"])
def test_2_reproduces_mass_polr_published_housing_fit(weighted):
    frame = _housing()
    data = frame if weighted else _people(frame)
    X = _r_coded(data)
    y, _ = ordinal_outcome(data["Sat"], LEVELS["Sat"])
    model = ProportionalOddsRegression().fit(X, y, sample_weight=data["Freq"] if weighted else None)
    assert model.converged_
    for j, name in enumerate(X.columns):
        assert model.coef_[j] == pytest.approx(POLR_COEF[name], abs=1e-6), name
    np.testing.assert_allclose(model.thresholds_, POLR_CUTS, atol=1e-6)
    assert -2 * model.loglik_ == pytest.approx(POLR_DEVIANCE, abs=1e-3)
    se = np.sqrt(np.diag(np.linalg.inv(model.information_)))
    # Printed to 3 significant digits: half a unit in the last place, plus polr's numerical Hessian.
    for j, name in enumerate(X.columns):
        assert se[j] == pytest.approx(POLR_SE[name], abs=5e-5 + 1e-5), name
    np.testing.assert_allclose(se[len(X.columns):], POLR_CUT_SE, atol=5e-4 + 1e-5)


def test_2_the_fit_stage_reproduces_polr_on_the_housing_table(tmp_path):
    """Through the app: text levels in a declared order, categorical predictors one-hot coded by
    the pipeline, under inference. Whatever reference levels the pipeline picks, each column is a
    contrast of polr's coefficients, with polr's estimate (and standard error, where the contrast
    is one of polr's own coefficients)."""
    people = _people(_housing())
    paths = mf.ingest_frame(people, tmp_path)
    st = mf.state(roles={"Infl": "covariate", "Type": "covariate", "Cont": "covariate"},
                  target="Sat", task="ordinal", models=["proportional_odds"], purpose="inference",
                  outcome_order=LEVELS["Sat"])
    split = mf.split_bundle(np.arange(len(people)), holdout=0.0)
    ti = mf.target_info("ordinal", "Sat")
    design = design_stage(mf.context(st, {"split": split, "target_info": ti}, paths))
    fit = fit_stage(mf.context(st, {"design": design, "split": split, "target_info": ti}, paths))
    artifact = FitArtifact.model_validate(fit.data)
    assert artifact.levels == LEVELS["Sat"] and artifact.primary_metric == "c_index"
    rows = {r.feature: r for r in artifact.models[0].coefficients}
    polr = {**{f"{f}{lv}": POLR_COEF.get(f"{f}{lv}", 0.0) for f in ("Infl", "Type", "Cont")
               for lv in LEVELS[f]}}
    se_of = {**{f"{f}{lv}": POLR_SE.get(f"{f}{lv}") for f in ("Infl", "Type", "Cont")
                for lv in LEVELS[f]}}
    checked = 0
    for factor in ("Infl", "Type", "Cont"):
        present = [lv for lv in LEVELS[factor] if f"{factor}_{lv}" in rows]
        (reference,) = [lv for lv in LEVELS[factor] if lv not in present]
        for level in present:
            row = rows[f"{factor}_{level}"]
            expected = polr[f"{factor}{level}"] - polr[f"{factor}{reference}"]
            assert row.estimate == pytest.approx(expected, abs=2e-6), (factor, level)
            single = [se_of[f"{factor}{v}"] for v in (level, reference) if se_of[f"{factor}{v}"]]
            if len(single) == 1:  # the contrast is one of polr's coefficients: same SE
                assert (row.ci_high - row.ci_low) / (2 * 1.959964) == pytest.approx(single[0], abs=6e-5)
            checked += 1
    assert checked == 6
    cuts = [r for name, r in rows.items() if name.startswith("(cut-point")]
    assert [c.feature for c in cuts] == ["(cut-point Low | Medium)", "(cut-point Medium | High)"]
    # The fitted model's probabilities in each of the 24 cells equal polr's.
    pipeline = fit.objects["fitted"]["proportional_odds"]
    cells = _housing().drop_duplicates(["Infl", "Type", "Cont"])[["Infl", "Type", "Cont"]]
    ours = pipeline.predict_proba(cells.reset_index(drop=True))
    eta = _r_coded(cells).to_numpy() @ np.array(list(POLR_COEF.values()))
    cumulative = 1 / (1 + np.exp(-(np.array(POLR_CUTS)[None, :] - eta[:, None])))
    theirs = np.diff(np.column_stack([np.zeros(len(eta)), cumulative, np.ones(len(eta))]), axis=1)
    np.testing.assert_allclose(ours, theirs, atol=2e-6)


def test_2_cut_points_keep_their_level_names_when_inference_draws_a_holdout(tmp_path):
    """Repair round (verifier, WP12a × WP8): under inference with a 20% holdout drawn, the table is
    refit on every analyzed row, and that refit named its cut-points by the internal codes, so on a
    1–5 Likert outcome "(cut-point 1 | 2)" carried the 2|3 threshold.

    Reference: statsmodels ``OrderedModel`` (logit) on all 800 rows, its thresholds converted by
    ``transform_threshold_params``. Measured: the fit stage's inference table, with a holdout of
    160 rows. Each "(cut-point j | j+1)" row must equal statsmodels' threshold between the levels
    ``j`` and ``j + 1``, and the coefficients come from every row (n = 800)."""
    rng = np.random.default_rng(55)
    n = 800
    frame = pd.DataFrame({"age": rng.normal(50, 10, n), "fiber_g": rng.gamma(3, 5, n)})
    latent = 0.03 * frame["age"] - 0.05 * frame["fiber_g"] + rng.logistic(size=n)
    frame["likert"] = np.digitize(latent, [0.0, 1.0, 2.0, 3.0]) + 1  # levels 1 to 5
    reference = _ordered_model(frame[["age", "fiber_g"]], frame["likert"].to_numpy() - 1)
    params = reference.params.to_numpy()
    thresholds = reference.model.transform_threshold_params(params)[1:-1]

    paths = mf.ingest_frame(frame, tmp_path)
    st = mf.state(roles={"age": "covariate", "fiber_g": "exposure"}, target="likert",
                  task="ordinal", models=["proportional_odds"], purpose="inference", lens=["survey"])
    split = mf.split_bundle(np.arange(n), holdout=0.2, seed=1)
    assert split.data["n_holdout"] == 160
    ti = mf.target_info("ordinal", "likert")
    design = design_stage(mf.context(st, {"split": split, "target_info": ti}, paths))
    fit = fit_stage(mf.context(st, {"design": design, "split": split, "target_info": ti}, paths))
    model = fit.data["models"][0]
    assert model["coefficients_n"] == n and model["inference"]["rows"] == "all"
    rows = {r["feature"]: r for r in model["coefficients"]}
    names = [f"(cut-point {j} | {j + 1})" for j in range(1, 5)]
    assert [r for r in rows if r.startswith("(cut-point")] == names
    for name, expected in zip(names, thresholds):
        assert rows[name]["estimate"] == pytest.approx(expected, abs=1e-5), name
    assert rows["fiber_g"]["estimate"] == pytest.approx(params[1], abs=1e-5)


def test_2_text_levels_are_never_ordered_by_their_labels():
    with pytest.raises(ValueError, match="must be declared"):
        ordinal_outcome(pd.Series(["mild", "severe", "none", "moderate"]))
    codes, names = ordinal_outcome(pd.Series(["mild", "severe", "none", "moderate"]),
                                   ["none", "mild", "moderate", "severe"])
    assert codes.tolist() == [1, 3, 0, 2] and names == ["none", "mild", "moderate", "severe"]
    codes, names = ordinal_outcome(pd.Series([3, 1, 5, 2]))  # numbers order themselves
    assert codes.tolist() == [2, 0, 3, 1] and names == ["1", "2", "3", "5"]


# ── statsmodels OrderedModel ─────────────────────────────────────────────────


def _latent(n: int, seed: int, cuts=(0.5, 1.5, 2.5, 3.5)) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"age": rng.normal(50, 10, n), "fiber_g": rng.gamma(3, 5, n),
                      "male": rng.integers(0, 2, n).astype(float)})
    latent = 0.03 * X["age"] - 0.05 * X["fiber_g"] + 0.4 * X["male"] + rng.logistic(size=n)
    return X, np.digitize(latent, cuts)


def _ordered_model(X: pd.DataFrame, y: np.ndarray, **fit: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return OrderedModel(y, X, distr="logit").fit(method="bfgs", maxiter=20_000, gtol=1e-10,
                                                      disp=False, **fit)


@pytest.mark.parametrize("seed, cuts", [(0, (0.5, 1.5, 2.5, 3.5)), (1, (1.0, 2.0)),
                                        (2, (-0.5, 0.0, 0.5, 1.0, 2.0, 3.0))])
def test_2_matches_statsmodels_ordered_model(seed, cuts):
    X, y = _latent(600, seed, cuts)
    reference = _ordered_model(X, y)
    p = X.shape[1]
    params = reference.params.to_numpy()
    cuts_ref = reference.model.transform_threshold_params(params)[1:-1]
    # Cut-point covariance by the delta method: θ_1 = a, θ_j = a + Σ_{i<j} exp(c_i).
    q = len(cuts_ref)
    J = np.zeros((q, q))
    J[:, 0] = 1.0
    for j in range(1, q):
        J[j, 1:j + 1] = np.exp(params[p + 1:p + j + 1])
    cov = reference.cov_params().to_numpy()
    cut_se = np.sqrt(np.diag(J @ cov[p:, p:] @ J.T))

    table = ordinal_table(X, y, None, INDEPENDENT)
    est = np.array([r["estimate"] for r in table.rows])
    se = np.array([r["se"] for r in table.rows])
    np.testing.assert_allclose(est[:p], params[:p], atol=1e-6)
    np.testing.assert_allclose(est[p:], cuts_ref, atol=1e-6)
    np.testing.assert_allclose(se[:p], reference.bse.to_numpy()[:p], rtol=1e-4)
    np.testing.assert_allclose(se[p:], cut_se, rtol=1e-4)
    model = ProportionalOddsRegression().fit(X, y)
    assert model.loglik_ == pytest.approx(reference.llf, abs=1e-6)
    np.testing.assert_allclose(model.predict_proba(X), reference.predict(), atol=1e-6)


def test_2_cluster_robust_intervals_match_statsmodels_sandwich_on_t_g_minus_1():
    """statsmodels' cluster sandwich carries (G/(G − 1))·((n − 1)/(n − k)); Stata's for a
    maximum-likelihood model, which the table uses, carries G/(G − 1) only, so statsmodels'
    standard errors are divided by √((n − 1)/(n − k)) before comparing."""
    X, y = _latent(600, 3)
    groups = np.repeat(np.arange(150), 4)
    reference = _ordered_model(X, y, cov_type="cluster", cov_kwds={"groups": groups})
    n, k = len(y), reference.params.size
    expected = reference.bse.to_numpy()[:3] / np.sqrt((n - 1) / (n - k))
    table = ordinal_table(X, y, None, as_clusters(groups))
    assert table.info["covariance"] == "CR1"
    rows = table.rows[:3]
    np.testing.assert_allclose([r["se"] for r in rows], expected, rtol=1e-4)
    from scipy import stats

    for r in rows:
        assert r["df"] == 149
        half = stats.t.ppf(0.975, 149) * r["se"]
        assert r["ci_high"] - r["estimate"] == pytest.approx(half, rel=1e-9)


# ── the proportional-odds check (Brant) ──────────────────────────────────────


def _non_proportional(n: int, rng: np.random.Generator, spread: float) -> tuple[np.ndarray, np.ndarray]:
    """Three cut-points; x1's log odds ratio is 0.5 − spread, 0.5, 0.5 + spread across them."""
    X = np.column_stack([rng.normal(size=n), rng.normal(size=n)])
    cuts = np.array([-1.0, 0.0, 1.0])
    slopes = np.array([0.5 - spread, 0.5, 0.5 + spread])
    above = 1 / (1 + np.exp(-(X[:, [0]] * slopes[None, :] + 0.3 * X[:, [1]] - cuts[None, :])))
    u = rng.random(n)[:, None]
    # P(Y > j) must fall with j; the spreads used keep it so over the bulk of x.
    above = np.minimum.accumulate(above, axis=1)
    return X, (u < above).sum(axis=1)


def test_2_brant_slopes_are_statsmodels_logits_and_the_test_holds_size_and_power():
    rng = np.random.default_rng(11)
    X, y = _non_proportional(800, rng, 0.0)
    result = brant_test(X, y, ["x1", "x2"])
    for j, slopes in enumerate(result["slopes"]):
        logit = sm.Logit((y > j).astype(float), sm.add_constant(X)).fit(disp=False, method="newton",
                                                                         tol=1e-12)
        np.testing.assert_allclose(slopes, logit.params[1:], atol=1e-6)
    # Size: proportional odds true, 300 datasets of 500; nominal 0.05, MC SE 0.013.
    size = np.mean([brant_test(*_non_proportional(500, np.random.default_rng(100 + r), 0.0),
                               ["x1", "x2"])["p"] < 0.05 for r in range(300)])
    # Power: x1's odds ratio runs 0.5 − 0.6 to 0.5 + 0.6 on the log scale across cut-points.
    power = np.mean([brant_test(*_non_proportional(500, np.random.default_rng(900 + r), 0.6),
                                ["x1", "x2"])["p"] < 0.05 for r in range(100)])
    assert size <= 0.09, size
    assert power >= 0.80, power


# ── the shelf: registered, ordered by soundness, order-blind families say so ─


def test_2_the_family_is_registered_with_its_purpose_task_bias_and_cautions():
    family = info(get_family("proportional_odds"))
    assert family.tasks == ["ordinal"]
    assert 0 < len(family.inductive_bias.split()) <= 20
    assert any("proportional odds" in c.lower() for c in family.cautions)
    assert "Brant" in get_family("proportional_odds").describe("ordinal", "inference")[1]


@pytest.mark.parametrize("purpose", ["inference", "prediction"])
def test_2_on_an_ordinal_task_the_proportional_odds_family_ranks_first(purpose):
    situation = Situation(task="ordinal", purpose=purpose, n_rows=1_681, n_features=6,
                          n_classes=3, class_counts=(567, 446, 668))
    ranked = rank(situation)
    keys = [f.key for f, _ in ranked]
    assert keys[0] == "proportional_odds"
    assert set(keys) == {"proportional_odds", "linear", "elastic_net", "boosted_trees"}  # never shortened
    for family, assessment in ranked[1:]:
        assert assessment.concerns[0] == ORDER_BLIND
        assert assessment.score < ranked[0][1].score


# ── scores ───────────────────────────────────────────────────────────────────


def test_2_harrells_c_matches_lifelines_and_the_rps_its_definition():
    X, y = _latent(400, 5)
    model = ProportionalOddsRegression().fit(X, y)
    proba = model.predict_proba(X)
    expected_level = proba @ np.arange(proba.shape[1])
    scores = ordinal_scores(y, proba, list(model.classes_))
    assert scores["c_index"] == pytest.approx(
        concordance_index(y, expected_level, np.ones(len(y))), abs=1e-12)
    # Ties in the score count one half, in both.
    rounded = np.round(expected_level, 1)
    assert concordance(y, rounded) == pytest.approx(concordance_index(y, rounded, np.ones(len(y))),
                                                    abs=1e-12)
    K = proba.shape[1]
    rps = np.mean([sum((proba[i, :j + 1].sum() - float(y[i] <= j)) ** 2 for j in range(K - 1)) / (K - 1)
                   for i in range(len(y))])
    assert scores["rps"] == pytest.approx(rps, rel=1e-12)
