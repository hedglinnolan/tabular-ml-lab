"""WP8 · Inference reporting: the right scale, every row, one declared model (AUDIT_REPORT.md §5).

Closes ME-07, ME-12 and ME-13, and the minors A21/E12. The four acceptance tests, as written there:

1. **Odds ratios.** "A binary inference table shows exp(β) on a log axis with the event and reference
   named (fiber: OR 1.070 per g); multinomial rows show relative-risk ratios against a named
   reference."
2. **All rows.** "Under inference, the coefficient table is fit on every analyzed row with n stated;
   on the NHANES export sugar is −0.0576 (−0.1096, −0.0057) whatever the seed."
3. **Final model.** "Opening the seal requires a declared final family (chosen on CV); its holdout is
   the reported result and the others are marked secondary; the selection optimism is stated or
   estimated (null reference: +0.036 AUC)."
4. **Sample-size guidance.** "Under prediction the shelf uses Riley-type criteria; for OLS under
   inference the warning threshold is about 2 subjects per variable (Austin & Steyerberg 2015)."

Every measured value runs through the code the app runs (the design and fit stages, the shelf, the
server's decisions route); every reference comes from somewhere else: a maximum-likelihood fit
written out here and solved by scipy's general optimizer, least squares by numpy with HC3 written
from its definition (``references.py``), statsmodels for the audit's own quoted numbers, Riley et
al.'s published worked examples and their own Python port of ``pmsampsize`` (Whittle & Ensor, a
test-only reference), and analytic truths of a null simulation. Simulations are seeded, so each run
gives the same numbers; the Monte Carlo error of each is stated beside its bound.
"""
from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pmsampsize.pmsampsize import pmsampsize  # test-only reference (requirements-dev.txt)
from scipy import stats

from turbotab.core.decisions import GrainSpec, ProjectState, SplitSpec
from turbotab.core.models import get_family
from turbotab.core.models.base import Situation
from turbotab.core.stages.modeling import design_stage, fit_stage, shelf_stage
from turbotab.core.stages.rows import cohort_stage, split_stage
from turbotab.core.stages.target import target_info_stage
from turbotab.core.tests import modeling_fixtures as mf
from turbotab.core.tests.acceptance.references import hc3_by_definition
from turbotab.core.tests.stage_harness import NHANES, Ingested
from turbotab.core.tests.acceptance.server_drive import press_fit, served

Z = stats.norm.ppf(0.975)


# ── references ───────────────────────────────────────────────────────────────


def multinomial_mle(X: np.ndarray, codes: np.ndarray, K: int) -> tuple[np.ndarray, np.ndarray]:
    """Maximum likelihood for a multinomial logit with class 0 as the reference (K = 2 is the
    logistic model), and its Wald standard errors.

    The log-likelihood ``Σ_i [Σ_k y_ik η_ik − log(1 + Σ_k exp η_ik)]``, its gradient and its Hessian
    (the observed information, block (a, b) = ``Xᵀ diag(p_a (1[a = b] − p_b)) X``) are written out
    here and solved by scipy's general trust-region optimizer, not by statsmodels' Newton iterations
    that the app runs. The columns after the first (the constant) are centered and scaled for the
    optimizer, ``X_s = X T``, and mapped back, ``β = T β_s`` and ``V = T V_s Tᵀ``. Returns the
    (K − 1) × P coefficients, row k for class k + 1 against class 0, and their standard errors from
    the inverse information.
    """
    from scipy.optimize import minimize, root

    n, P = X.shape
    T = np.eye(P)
    for j in range(1, P):
        m, s = X[:, j].mean(), X[:, j].std()
        T[j, j], T[0, j] = 1 / s, -m / s
    X_raw, X = X, X @ T
    Y = np.eye(K)[codes][:, 1:]

    def probs(theta: np.ndarray) -> np.ndarray:
        eta = np.column_stack([np.zeros(n), X @ theta.reshape(K - 1, P).T])
        eta -= eta.max(axis=1, keepdims=True)
        e = np.exp(eta)
        return (e / e.sum(axis=1, keepdims=True))[:, 1:]

    def nll(theta: np.ndarray) -> float:
        eta = X @ theta.reshape(K - 1, P).T
        return float(-(np.sum(Y * eta) - np.logaddexp.reduce(
            np.column_stack([np.zeros(n), eta]), axis=1).sum()))

    def grad(theta: np.ndarray) -> np.ndarray:
        return -((Y - probs(theta)).T @ X).ravel()

    def hess(theta: np.ndarray) -> np.ndarray:
        p = probs(theta)
        H = np.zeros(((K - 1) * P, (K - 1) * P))
        for a in range(K - 1):
            for b in range(K - 1):
                w = p[:, a] * (float(a == b) - p[:, b])
                H[a * P:(a + 1) * P, b * P:(b + 1) * P] = (X * w[:, None]).T @ X
        return H

    start = minimize(nll, np.zeros((K - 1) * P), jac=grad, hess=hess, method="trust-exact",
                     options={"gtol": 1e-9}).x
    # The log-likelihood is flat to rounding near its maximum, so the last digits come from solving
    # the score equations themselves (scipy's hybrid root finder, the Hessian as their Jacobian).
    fit = root(grad, start, jac=hess, method="hybr", options={"xtol": 1e-14})
    assert np.abs(grad(fit.x)).max() < 1e-9, fit.message
    V_s = np.linalg.inv(hess(fit.x))
    block = np.kron(np.eye(K - 1), T)  # the same map for every class's coefficients
    beta = block @ fit.x
    se = np.sqrt(np.diag(block @ V_s @ block.T))
    assert np.allclose(X_raw @ beta.reshape(K - 1, P).T, X @ fit.x.reshape(K - 1, P).T)
    return beta.reshape(K - 1, P), se.reshape(K - 1, P)


def by_feature(rows: list[dict]) -> dict[str, dict]:
    return {r["feature"]: r for r in rows}


def fit_linear(frame: pd.DataFrame, paths: dict, *, target: str, task: str, roles: dict,
               purpose: str = "inference", event: str | None = None, seed: int = 0,
               holdout: float = 0.2, models: list[str] | None = None):
    """The design and fit stages, as the worker runs them, on a split drawn over every row."""
    split = mf.split_bundle(np.arange(len(frame)), seed=seed, holdout=holdout)
    st = mf.state(roles=roles, target=target, task=task, purpose=purpose, event=event,
                  models=models or ["linear"], energy_adjustment=None)
    ti = mf.target_info(task, target)
    design = design_stage(mf.context(st, {"split": split, "target_info": ti}, paths))
    return split, fit_stage(mf.context(st, {"design": design, "split": split, "target_info": ti},
                                       paths))


# ── 1 · odds ratios and relative-risk ratios ─────────────────────────────────


def fiber_table() -> pd.DataFrame:
    """The audit's fixture for ME-07 (``repro.tar.gz``: ``E-skeptic/e4_logit.py``), draw for draw."""
    rng = np.random.default_rng(5)
    n = 1500
    fiber = rng.normal(20, 6, n)
    age = rng.normal(50, 10, n)
    p = 1 / (1 + np.exp(-(-1 - 0.08 * (fiber - 20) + 0.04 * (age - 50))))
    dm = np.where(rng.random(n) < p, "diabetic", "not_diabetic")
    return pd.DataFrame({"fiber_g": fiber, "age": age, "dm": dm})


def glycemia_table() -> pd.DataFrame:
    """Three outcome levels; fiber lowers the odds of each raised level against normoglycemia."""
    rng = np.random.default_rng(8)
    n = 2000
    fiber = rng.normal(20, 6, n)
    age = rng.normal(50, 10, n)
    eta = np.column_stack([np.zeros(n),
                           -0.5 - 0.05 * (fiber - 20) + 0.03 * (age - 50),
                           -1.5 - 0.10 * (fiber - 20) + 0.06 * (age - 50)])
    p = np.exp(eta) / np.exp(eta).sum(axis=1, keepdims=True)
    u = rng.random(n)[:, None]
    level = (u > p.cumsum(axis=1)).sum(axis=1)
    names = np.array(["normoglycemia", "prediabetes", "diabetes"])
    return pd.DataFrame({"fiber_g": fiber, "age": age, "glycemia": names[level]})


@pytest.fixture(scope="module")
def fiber(tmp_path_factory):
    frame = fiber_table()
    return frame, mf.ingest_frame(frame, tmp_path_factory.mktemp("wp8_fiber"))


@pytest.fixture(scope="module")
def glycemia(tmp_path_factory):
    frame = glycemia_table()
    return frame, mf.ingest_frame(frame, tmp_path_factory.mktemp("wp8_glycemia"))


ROLES = {"fiber_g": "exposure", "age": "covariate"}


@pytest.mark.parametrize("event", ["diabetic", None])
def test_1a_a_binary_table_reports_odds_ratios_for_the_named_event_on_a_log_axis(fiber, event):
    """Fiber against diabetes, the audit's fixture, under inference.

    Reference: logistic maximum likelihood written out and solved by scipy (:func:`multinomial_mle`,
    K = 2) on every analyzed row, the event coded 1. Each row's ``ratio`` is exp(β) and its interval
    exp of the Wald interval's ends (to 10⁻⁶); the table says it is an odds ratio, to be drawn on a
    log axis, and names the event and the reference level, in a sentence that names the outcome.

    The audit's number: "Fiber 0.0674 log-odds is an odds ratio of 1.070 per gram" (ME-07), the odds
    of ``not_diabetic``, the level that sorts last, which the models took as the event when none
    was named. With no event named the table now says so (event ``not_diabetic``, reference
    ``diabetic``) and shows 1.070; with ``diabetic`` named it shows the same association the other
    way round, 1/1.070 = 0.935 per gram.
    """
    frame, paths = fiber
    _, fit = fit_linear(frame, paths, target="dm", task="binary", roles=ROLES, event=event)
    model = fit.data["models"][0]
    info = model["inference"]
    named = event or "not_diabetic"
    other = "not_diabetic" if named == "diabetic" else "diabetic"
    assert info["scale"] == "odds_ratio" and info["axis"] == "log"
    assert info["event"] == named and info["reference"] == other
    assert info["effect"] == (f"Odds ratio of `dm` being `{named}` rather than `{other}`, per unit "
                              f"of each input, holding the others.")
    assert info["caption"].startswith(f"Estimates are log-odds of `{named}` against `{other}`; "
                                      f"each odds ratio is exp(estimate).")

    X = np.column_stack([np.ones(len(frame)), frame[["fiber_g", "age"]].to_numpy()])
    B, SE = multinomial_mle(X, (frame["dm"] == named).to_numpy().astype(int), 2)
    got = by_feature(model["coefficients"])
    for j, name in ((1, "fiber_g"), (2, "age")):
        b, se = B[0, j], SE[0, j]
        assert got[name]["estimate"] == pytest.approx(b, rel=1e-6)  # the log-odds, kept
        assert got[name]["ratio"] == pytest.approx(math.exp(b), rel=1e-6)
        assert got[name]["ratio_low"] == pytest.approx(math.exp(b - Z * se), rel=1e-6)
        assert got[name]["ratio_high"] == pytest.approx(math.exp(b + Z * se), rel=1e-6)
    intercept = got["(intercept)"]
    assert intercept["ratio"] is None and intercept["ratio_low"] is None  # a baseline, not a ratio
    fiber_or = got["fiber_g"]["ratio"]
    if event is None:
        assert round(fiber_or, 3) == 1.070  # the audit's figure
    else:
        assert round(fiber_or, 3) == 0.935 and fiber_or == pytest.approx(1 / 1.0697513, rel=1e-6)


def test_1b_multinomial_rows_are_relative_risk_ratios_against_a_named_reference(glycemia):
    """Three glycemia levels. Reference: multinomial maximum likelihood written out and solved by
    scipy (:func:`multinomial_mle`, K = 3), the first level as the models code it (``diabetes``,
    the first in sorted order) as the reference. Each non-intercept row's ``ratio`` is exp(β_k), the
    relative-risk ratio of its level against ``diabetes``, with exp of the Wald interval's ends (to
    10⁻⁶); the table names the reference and the scale, drawn on a log axis."""
    frame, paths = glycemia
    _, fit = fit_linear(frame, paths, target="glycemia", task="multiclass", roles=ROLES)
    model = fit.data["models"][0]
    info = model["inference"]
    classes = sorted(frame["glycemia"].unique())
    assert classes[0] == "diabetes"
    assert info["scale"] == "relative_risk_ratio" and info["axis"] == "log"
    assert info["reference"] == "diabetes" and info["event"] is None
    assert info["effect"] == ("Relative-risk ratio of each level of `glycemia` against `diabetes`, "
                              "per unit of each input, holding the others.")
    X = np.column_stack([np.ones(len(frame)), frame[["fiber_g", "age"]].to_numpy()])
    codes = pd.Categorical(frame["glycemia"], categories=classes).codes
    B, SE = multinomial_mle(X, np.asarray(codes), 3)
    got = by_feature(model["coefficients"])
    for k, level in enumerate(classes[1:]):
        for j, name in ((1, "fiber_g"), (2, "age")):
            row = got[f"{name} [{level}]"]
            b, se = B[k, j], SE[k, j]
            assert row["ratio"] == pytest.approx(math.exp(b), rel=1e-6)
            assert row["ratio_low"] == pytest.approx(math.exp(b - Z * se), rel=1e-6)
            assert row["ratio_high"] == pytest.approx(math.exp(b + Z * se), rel=1e-6)
        assert got[f"(intercept) [{level}]"]["ratio"] is None


def test_1c_a_continuous_outcome_stays_a_difference_in_means_on_a_linear_axis(tmp_path):
    """The other side of the same rule: a least-squares table is a difference in the mean outcome,
    no ratio, drawn on a linear axis, in words that name the outcome."""
    frame = mf.nhanes_like(300, seed=6)
    paths = mf.ingest_frame(frame, tmp_path)
    _, fit = fit_linear(frame, paths, target="glucose", task="regression",
                        roles={"age": "covariate", "sugar": "exposure"})
    info = fit.data["models"][0]["inference"]
    assert info["scale"] == "difference" and info["axis"] == "linear"
    assert info["effect"] == "Difference in mean `glucose` per unit of each input, holding the others."
    assert all(r["ratio"] is None for r in fit.data["models"][0]["coefficients"])


# ── 2 · every analyzed row ───────────────────────────────────────────────────


def _ols_hc3(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Least squares by numpy and HC3 from its definition (``references.hc3_by_definition``)."""
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    V = hc3_by_definition(X, y - X @ beta)
    return beta, np.sqrt(np.diag(V)), X.shape[0] - X.shape[1]


SIM_ROLES = {"age": "covariate", "gender": "covariate", "bmi": "covariate", "sugar": "exposure",
             "carb": "exposure", "fat_total": "exposure"}


@pytest.fixture(scope="module")
def simulated(tmp_path_factory):
    frame = mf.nhanes_like(600, seed=4)
    return frame, mf.ingest_frame(frame, tmp_path_factory.mktemp("wp8_rows"))


def test_2a_under_inference_the_table_uses_every_analyzed_row_whatever_the_seed(simulated):
    """On an NHANES-shaped table (600 rows, 20% held out), three seals drawn with three seeds.

    Reference: numpy least squares with HC3 written from its definition, on all 600 rows. Under
    inference every seed's table equals it (estimates and interval ends to 10⁻⁸), the number of rows
    is stated (``coefficients_n``, ``inference.n_rows``, and in the caption), and a concern says the
    held-out rows were included and why. Under prediction the coefficients stay the training rows'
    (480), as "sealed once, opened once" requires (BLUEPRINT §12 ruling 3): the same numpy fit on
    those rows reproduces them.
    """
    frame, paths = simulated
    columns = ["age", "gender_male", "bmi", "sugar", "carb", "fat_total"]
    design = frame.assign(gender_male=(frame["gender"] == "male").astype(float))
    X_all = np.column_stack([np.ones(len(frame)), design[columns].to_numpy(float)])
    beta, se, dof = _ols_hc3(X_all, frame["glucose"].to_numpy(float))
    q = stats.t.ppf(0.975, dof)
    tables = []
    for seed in (0, 1, 2):
        split, fit = fit_linear(frame, paths, target="glucose", task="regression", roles=SIM_ROLES,
                                seed=seed)
        model = fit.data["models"][0]
        assert fit.data["n_holdout"] == 120 and fit.data["n_train"] == 480
        assert model["coefficients_n"] == model["inference"]["n_rows"] == 600
        assert model["inference"]["rows"] == "all"
        assert model["inference"]["caption"].endswith("Estimated from all 600 analyzed rows.")
        assert any(c.startswith("Under inference the coefficients are estimated from all 600 "
                                "analyzed rows, the 120 held-out ones included") for c in model["concerns"])
        got = by_feature(model["coefficients"])
        for j, name in enumerate(columns, start=1):
            assert got[name]["estimate"] == pytest.approx(beta[j], rel=1e-8)
            assert got[name]["ci_low"] == pytest.approx(beta[j] - q * se[j], rel=1e-8)
            assert got[name]["ci_high"] == pytest.approx(beta[j] + q * se[j], rel=1e-8)
        tables.append([got[c]["estimate"] for c in columns])
    np.testing.assert_allclose(tables[1:], [tables[0], tables[0]], rtol=1e-12, atol=0)

    # Prediction keeps its lockbox: the coefficients are the training rows'.
    split, fit = fit_linear(frame, paths, target="glucose", task="regression", roles=SIM_ROLES,
                            purpose="prediction", seed=0)
    model = fit.data["models"][0]
    a = split.frames["assignment"]
    train = a.loc[a["partition"] == "train", "row_id"].to_numpy()
    assert model["coefficients_n"] == 480 and model["inference"] is None
    b_train = np.linalg.lstsq(X_all[train], frame["glucose"].to_numpy(float)[train], rcond=None)[0]
    got = by_feature(model["coefficients"])
    for j, name in enumerate(columns, start=1):
        assert got[name]["estimate"] == pytest.approx(b_train[j], rel=1e-8)
    assert abs(got["sugar"]["estimate"] - beta[columns.index("sugar") + 1]) > 1e-4  # it discriminates


NHANES_PREDICTORS = ["gender", "age", "bp_sys", "bp_di", "weight", "height", "bmi", "waist", "kcal",
                     "protein", "sugar", "carb", "fat_total", "fat_sat", "fat_mon", "fat_poly", "hdl",
                     "triglycerides", "meds_hbp", "meds_chol"]
needs_nhanes = pytest.mark.skipif(not NHANES.is_file(),
                                  reason="the NHANES export (_tt_tmp_nhanes.csv) is not on this machine")


@needs_nhanes
def test_2b_on_the_nhanes_export_sugar_is_the_all_rows_estimate_whatever_the_seed(tmp_path):
    """The audit's run (``repro.tar.gz``: ``E-skeptic/nhanes_offline.py``): glucose on the twenty
    predictors the app proposed, complete cases (2,996 rows), purpose = inference, drawn through the
    real target, cohort, split, design and fit stages with three seeds.

    Reference: the audit's own fit, re-run here with statsmodels on the CSV read by pandas: all 2,996
    rows give sugar −0.0576, model-based interval (−0.1096, −0.0057). Every seed's table gives that
    estimate (to 10⁻⁸ against numpy least squares), states n = 2,996, and is the same table. The
    interval is HC3 (WP2 made heteroskedasticity-robust intervals the default for independent rows
    under inference, AUDIT_REPORT MA-07), so it is checked against HC3 written from its definition:
    (−0.1151, −0.0002), not against the model-based one the acceptance text quotes; the audit's
    model-based interval is reproduced by statsmodels below, so the fixture is the audit's.
    Today's training-rows table moved with the seed (twenty seals, −0.093 to −0.034 here, "p < 0.05"
    in 9 of 20).
    """
    import statsmodels.api as sm

    raw = pd.read_csv(NHANES)
    cc = raw.dropna(subset=NHANES_PREDICTORS + ["glucose"]).copy()
    assert len(cc) == 2996
    cc["gender_male"] = (cc["gender"] == "male").astype(float)
    for c in ("meds_hbp", "meds_chol"):
        cc[c] = cc[c].map(lambda v: 1.0 if str(v).upper() == "TRUE" else 0.0)
    columns = ["gender_male"] + [p for p in NHANES_PREDICTORS if p != "gender"]
    X = np.column_stack([np.ones(len(cc)), cc[columns].to_numpy(float)])
    y = cc["glucose"].to_numpy(float)
    audit = sm.OLS(y, X).fit()
    j = columns.index("sugar") + 1
    assert round(audit.params[j], 4) == -0.0576
    assert np.round(audit.conf_int()[j], 4).tolist() == [-0.1096, -0.0057]
    beta, se, dof = _ols_hc3(X, y)
    q = stats.t.ppf(0.975, dof)

    table = Ingested(NHANES, tmp_path)
    seen = []
    for seed in (0, 1, 2):
        st = ProjectState(lens=["dietary"], target="glucose", task="regression", purpose="inference",
                          grain=GrainSpec(grain="one_row_per_unit", id_column="SEQN"),
                          roles={"SEQN": "identifier", **{c: "covariate" for c in NHANES_PREDICTORS}},
                          missing="complete_case", exclusions=[],
                          split=SplitSpec(holdout=0.2, seed=seed, folds=5), models=["linear"],
                          shape_confirmations=dict(mf.NHANES_TRUTH))  # the export's truth
        info = table.run(target_info_stage, st)
        cohort = table.run(cohort_stage, st, {"target_info": info})
        split = table.run(split_stage, st, {"cohort": cohort, "target_info": info})
        design = table.run(design_stage, st, {"split": split, "target_info": info})
        fit = table.run(fit_stage, st, {"design": design, "split": split, "target_info": info})
        model = fit.data["models"][0]
        assert fit.data["n_holdout"] > 500  # a holdout was drawn, and the table still used it
        assert model["coefficients_n"] == model["inference"]["n_rows"] == 2996
        assert "Estimated from all 2,996 analyzed rows." in model["inference"]["caption"]
        sugar = by_feature(model["coefficients"])["sugar"]
        assert sugar["estimate"] == pytest.approx(beta[j], rel=1e-8)
        assert round(sugar["estimate"], 4) == -0.0576
        assert sugar["ci_low"] == pytest.approx(beta[j] - q * se[j], rel=1e-8)
        assert sugar["ci_high"] == pytest.approx(beta[j] + q * se[j], rel=1e-8)
        seen.append((sugar["estimate"], sugar["ci_low"], sugar["ci_high"]))
    assert seen[0] == seen[1] == seen[2]  # whatever the seed


# ── 3 · one declared model ───────────────────────────────────────────────────

FAMILIES = ["linear", "elastic_net", "boosted_trees"]


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from turbotab.server.tests.conftest import make_client

    with make_client(tmp_path_factory.mktemp("wp8_home"), "local", 2, "http://127.0.0.1") as c:
        yield c


def _post(client, pid: str, decision: dict):
    return client.post(f"/api/projects/{pid}/decisions", json=decision)


def _fitted_project(client, models: list[str]) -> str:
    """dietary_recalls.csv (the seal tests' table), every question before the models given its
    usual answer (prediction, a 20% holdout), these families fitted."""
    from turbotab.server.tests.conftest import open_by_path, prepare, wait_for

    pid = open_by_path(client)
    wait_for(client, pid, {"ingest": "fresh", "profile": "fresh"}, timeout=120)
    for decision in ({"kind": "set_lens", "lenses": ["dietary"]},
                     {"kind": "set_target", "column": "hba1c"}):
        assert _post(client, pid, decision).status_code == 200
    models_decision = {"kind": "select_models", "models": models}
    prepare(client, pid, models_decision)
    from turbotab.server.tests.conftest import answer_settled

    # The readings the fit asks about, from dietary_recalls.csv's truth (BLUEPRINT §14.3).
    assert answer_settled(client, pid, None, models_decision).status_code == 200
    # MS6: under prediction every family is also fit on the comparisons' 10 × 5 folds.
    wait_for(client, pid, {"fit": "fresh"}, timeout=900)
    prepare(client, pid, {"kind": "open_seal"})  # the questions after the models, as usual
    wait_for(client, pid, {"fit": "fresh"}, timeout=900)
    return pid


def _fit(client, pid: str) -> dict:
    return served(client, pid, "fit")


def test_3a_opening_the_seal_needs_a_declared_final_family_whose_holdout_is_the_result(client):
    """Three families fitted, a 20% holdout, through the real server.

    * An opening that names no family is refused (409 ``final_model_needed``), with one exit per
      fitted family, the best cross-validated score first; naming a family that was not fitted is
      refused too. Nothing held out is revealed by either refusal.
    * Declared on cross-validation, the family's held-out score is the result (``final_model``,
      ``role: final``) and the other two are ``secondary``, said in one sentence; the record's
      sentence names the declared family.
    * With one family fitted, the opening declares it: the recorded decision names it.

    Reference: the CV scores in the fit artifact (the order of the exits), and the held-out scores
    the fit stage wrote to disk (``seal.read_sealed_scores``), which the opened fit shows unchanged.
    Varma & Simon 2006 (*BMC Bioinformatics* 7:91, abstract): "Using CV to compute an error estimate
    for a classifier that has itself been tuned using CV gives a significantly biased estimate of
    the true error"; the same holds for the best of several held-out scores picked after seeing
    them (audit A7: +0.023 AUC on 150-row holdouts).
    """
    from turbotab.core import seal
    from turbotab.server.tests.test_seal_api import assert_nothing_held_out_in

    service = client.app.state.service
    pid = _fitted_project(client, FAMILIES)
    fit = _fit(client, pid)
    assert fit["holdout_sealed"] is True and fit["n_holdout"] > 0
    assert fit["final_model"] is None and all(m["role"] is None for m in fit["models"])
    metric = fit["primary_metric"]
    # MS6: the families are compared, and the best named, on the comparison substrate (every
    # repeat of the repeated k-fold), on a strictly proper score (the MSE: lower is better).
    cv = {m["family"]: m["compared_on"]["estimate"] for m in fit["models"]}
    lower = metric in ("mse", "log_loss", "rps", "brier_t")

    undeclared = _post(client, pid, {"kind": "open_seal"})
    assert undeclared.status_code == 409, undeclared.text
    error = undeclared.json()["error"]
    assert error["code"] == "final_model_needed"
    assert [e["decision"]["family"] for e in error["exits"]] == sorted(cv, key=cv.get,
                                                                         reverse=not lower)
    # The cost of choosing on cross-validation is stated with the choice (test 3b checks its size).
    selection = fit["selection"]
    assert selection["best"] == error["exits"][0]["decision"]["family"]
    if selection["optimism"] > 0:
        assert "Choosing on cross-validation flatters the best family's CV" in error["message"]
    stray = _post(client, pid, {"kind": "open_seal", "family": "random_forest"})
    assert stray.status_code == 409 and stray.json()["error"]["code"] == "not_a_fitted_family"
    key = service.engine.status(pid)["fit"].key
    sealed = seal.read_sealed_scores(service.workspace.cache_dir(pid), key)
    constant = {m["family"] for m in fit["models"]  # an elastic net that kept no predictor
                if m["cv"]["r2"]["folds"] and all(v == 0.0 for v in m["cv"]["r2"]["folds"])}
    assert_nothing_held_out_in([undeclared.json(), stray.json()], sealed, constant)

    final = error["exits"][0]["decision"]["family"]
    opened = _post(client, pid, {"kind": "open_seal", "family": final})
    assert opened.status_code == 200, opened.text
    record = opened.json()["decisions"][-1]
    assert (record["decision"]["kind"], record["decision"]["family"]) == ("open_seal", final)
    assert record["decision"]["scores"] == sealed  # WP16 (RO-05): the opening keeps its scores
    assert "declared the final model on cross-validation beforehand" in record["sentence"]
    fit = _fit(client, pid)
    assert fit["final_model"] == final
    assert {m["family"]: m["role"] for m in fit["models"]} == {
        f: "final" if f == final else "secondary" for f in FAMILIES}
    assert {m["family"]: m["holdout"] for m in fit["models"]} == sealed
    label = next(m["label"] for m in fit["models"] if m["family"] == final)
    assert fit["final_note"] == (
        f"{label} was declared the final model on cross-validation before the held-out rows were "
        f"opened, so its held-out {fit['metric_labels'][metric]} is the reported result; the other 2 "
        f"families' held-out scores are secondary.")

    single = _fitted_project(client, ["linear"])
    before = _post(client, single, {"kind": "open_seal"})  # the held-out rows open after Fit
    assert before.status_code == 409 and before.json()["error"]["code"] == "fit_not_yet"
    assert press_fit(client, single)
    assert _post(client, single, {"kind": "open_seal"}).status_code == 200
    record = client.get(f"/api/projects/{single}").json()["decisions"][-1]
    assert (record["decision"]["kind"], record["decision"]["family"]) == ("open_seal", "linear")
    assert _fit(client, single)["final_model"] == "linear"


NULL_REPLICATES = 100


def test_3b_the_optimism_of_choosing_the_best_family_is_estimated_on_null_data():
    """The audit's null scenario (``repro.tar.gz``: ``A/r5_select.py``): 150 rows, ten noise
    predictors, a binary outcome independent of them, the three families cross-validated on five
    stratified folds; the family with the best CV AUC is the one a user would pick.

    Reference: analytic. Every model's AUC on new rows is exactly 0.5 when the outcome is
    independent of the predictors, so the optimism of the pick is its CV AUC less 0.5: the audit
    measured +0.036 (0.536 against 0.50, 60 datasets); here, over 100 datasets, the mean is +0.039
    (Monte Carlo SE 0.005), and the test asserts it lies in [0.025, 0.047], the audit's figure
    within two standard errors, so that the scenario is the audit's. The fit stage's estimate
    (:func:`selection_optimism`, bootstrap bias-corrected cross-validation over the out-of-fold
    predictions it kept) must remove it: the mean corrected AUC is within 0.01 of 0.5 (measured
    0.503, Monte Carlo SE 0.005), so the mean estimated optimism (+0.036) is within 0.01 of the
    true one. A correction of zero would leave the corrected AUC near 0.539; Tibshirani &
    Tibshirani's fold-wise correction, tried first, recovered about two thirds of the optimism on
    this scenario (+0.023 of +0.035 over 100 other datasets), which the bound also rejects.

    The pooled AUC the bootstrap scores with is checked against scikit-learn's ``roc_auc_score`` on
    every dataset's full out-of-fold predictions.

    Source check — Tsamardinos, Greasidou & Borboudakis, *Mach Learn* 2018;107:1895, abstract:
    "the cross-validated performance of the best configuration is optimistically biased. We present
    an efficient bootstrap method that corrects for the bias, called Bootstrap Bias Corrected CV
    (BBC-CV). BBC-CV's main idea is to bootstrap the whole process of selecting the best-performing
    configuration on the out-of-sample predictions of each configuration, without additional
    training of models."
    """
    from sklearn.base import clone
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    from turbotab.core.models.inner_cv import fit_pipeline
    from turbotab.core.models.metrics import cross_validate, fold_pairs
    from turbotab.core.models.pipeline import DesignSpec, build_pipeline
    from turbotab.core.models.selection import OutOfFold, selection_optimism

    n, p = 150, 10
    cols = [f"x{i}" for i in range(p)]
    from turbotab.core.models.pipeline import with_plans

    spec = DesignSpec(predictors=cols, inputs=cols, categorical=[], numeric=cols, energy=None,
                      impute=False)
    # Boosted trees are tuned (RT-5a), so they are built through the plan the design makes. With
    # about 60 events in an outer training fold (n_plan below 300) it holds the standard settings
    # alone: scikit-learn's defaults, one fit per fold, whatever each replicate's outcome.
    spec = with_plans(spec, [get_family(k) for k in FAMILIES], "binary", np.arange(n) % 2)
    trees = spec.plans["boosted_trees"]
    assert trees["n_plan"] < 300 and len(trees["candidates"]) == 1
    pipes = {k: build_pipeline(spec, get_family(k), "binary", "prediction", n, p) for k in FAMILIES}
    rng = np.random.default_rng(0)
    best_cv, corrected, estimated = [], [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for r in range(NULL_REPLICATES):
            X = pd.DataFrame(rng.normal(size=(n, p)), columns=cols)
            y = (rng.random(n) < 0.5).astype(int)
            folds = np.zeros(n, dtype=int)
            for f, (_, test) in enumerate(StratifiedKFold(5, shuffle=True, random_state=r).split(X, y)):
                folds[test] = f
            pairs = fold_pairs(folds)
            oof = OutOfFold("binary", X, y, pairs)
            results = {}
            for key in FAMILIES:
                fit = oof.wrap(key, lambda m, Xf, yf, rows: fit_pipeline(m, Xf, yf))
                results[key] = cross_validate("binary", lambda _k=key: clone(pipes[_k]), X, y, pairs,
                                              fit=fit)
                assert oof.score("auc", key, np.arange(n)) == pytest.approx(
                    roc_auc_score(y, oof.predictions[key][:, 1]), abs=1e-12)
            chosen = selection_optimism("binary", "auc", results, oof)
            cv = {k: results[k].summary("binary")["auc"]["estimate"] for k in FAMILIES}
            assert chosen["best"] == max(cv, key=cv.get) and chosen["cv"] == max(cv.values())
            best_cv.append(chosen["cv"])
            corrected.append(chosen["corrected"])
            estimated.append(chosen["optimism"])
    true_optimism = float(np.mean(best_cv)) - 0.5
    assert 0.025 <= true_optimism <= 0.047, true_optimism
    assert abs(float(np.mean(corrected)) - 0.5) <= 0.01, np.mean(corrected)
    assert abs(float(np.mean(estimated)) - true_optimism) <= 0.01, (np.mean(estimated), true_optimism)


def test_3c_the_fit_states_the_optimism_only_when_there_was_a_choice(simulated):
    """The fit stage's ``selection``: with three families it names the CV-best and states its
    optimism in a sentence; with one there was no choice and it is absent."""
    frame, paths = simulated
    _, fit = fit_linear(frame, paths, target="glucose", task="regression", roles=SIM_ROLES,
                        purpose="prediction", models=FAMILIES)
    selection = fit.data["selection"]
    # MS6: chosen on the strictly proper primary (the MSE, better when lower) over the
    # comparison substrate; R² is corrected for the same choice beside it.
    cv = {m["family"]: m["compared_on"]["estimate"] for m in fit.data["models"]}
    assert selection["best"] == min(cv, key=cv.get) and selection["cv"] == min(cv.values())
    assert selection["metric"] == "mse" and selection["replicates"] >= 500
    assert selection["corrected_low"] <= selection["corrected"] <= selection["corrected_high"]
    assert selection["optimism"] == pytest.approx(selection["corrected"] - selection["cv"], abs=1e-12)
    assert sum(selection["wins"].values()) == selection["replicates"]
    assert selection["extras"]["r2"]["corrected"] is not None
    assert selection["text"].startswith("Choosing the best of 3 families by cross-validated MSE")
    _, single = fit_linear(frame, paths, target="glucose", task="regression", roles=SIM_ROLES,
                           purpose="prediction")
    assert single.data["selection"] is None


# ── 4 · sample-size guidance ─────────────────────────────────────────────────


def test_4a_rileys_minimum_matches_the_published_examples_and_pmsampsize():
    """The criteria behind the shelf's sample-size sentence under prediction.

    References, in order of independence:

    * Riley et al., *BMJ* 2020;368:m441, worked examples (read from the published article):
      example 1, "For an outcome proportion of 0.05, the max(R2cs) value is 0.33 … the anticipated
      R2cs value is 0.15×0.33=0.05 … pmsampsize, type(b) rsquared(0.05) parameters(30)
      prevalence(0.05) This indicates that at least 5249 women are required … This is driven by
      criterion B3"; "with 20 rather than 30 candidate predictors, the required sample size to
      meet all four criteria is at least 3500"; example 3, "pmsampsize, type(c) rsquared(0.9)
      parameters(20) intercept(26.7) sd (8.7) This returns that at least 254 participants are
      required", driven by the residual standard deviation (234 + 20; the box calls it C2).
    * ``pmsampsize`` 0.1.0, the Python port by Whittle & Ensor of Ensor's R package, over a grid of
      parameters, outcome proportions and R² (binary) and of parameters and R² (continuous, with an
      intercept far from zero so that its criterion does not bind; the port's search for that
      criterion multiplies by the lower t quantile inside its loop and stops at the first step).
    * The intercept criterion C1 where it binds, from its definition (Riley et al. 2020, box 1,
      supplementary S1): the smallest n whose 95% interval for the mean, t(n − p − 1) ×
      √(σ²(1 − R²)/n), is within 10% of the mean, found here by a vectorized scan with scipy.
    """
    from turbotab.core.models.sample_size import binary_minimum, continuous_minimum, max_r2_cs

    assert round(max_r2_cs(0.05), 2) == 0.33
    example1 = binary_minimum(30, 0.05, 0.05)
    assert example1.n == 5249 and example1.binding.key == "B3"
    assert binary_minimum(20, 0.05, 0.05).n == 3500
    example3 = continuous_minimum(20, 0.9, mean=26.7, sd=8.7)
    assert example3.n == 254 and {c.key: c.n for c in example3.criteria}["C2"] == 254

    def port(**kw):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return pmsampsize(noprint=True, **kw)["sample_size"]

    for p in (1, 2, 5, 10, 24, 40, 80):
        for phi in (0.02, 0.1, 0.174, 0.3, 0.5):
            for share in (0.05, 0.15, 0.4):
                r2 = share * max_r2_cs(phi)
                assert binary_minimum(p, phi, r2).n == port(type="b", csrsquared=r2, parameters=p,
                                                            prevalence=phi), (p, phi, share)
        for r2 in (0.05, 0.15, 0.3, 0.6, 0.9):
            assert continuous_minimum(p, r2, mean=100.0, sd=1.0).n == port(
                type="c", rsquared=r2, parameters=p, intercept=100.0, sd=1.0), (p, r2)

    # C1 where it binds: a mean of 1 with an SD of 5.
    p, r2, mean, sd = 10, 0.15, 1.0, 5.0
    n = np.arange(p + 3, 20_000)
    half = stats.t.isf(0.025, n - p - 1) * np.sqrt(sd * sd * (1 - r2) / n)
    c1 = int(n[np.flatnonzero((mean + half) / mean <= 1.1)[0]])
    got = continuous_minimum(p, r2, mean=mean, sd=sd)
    assert {c.key: c.n for c in got.criteria}["C1"] == c1 == got.n
    assert c1 > 10 * 234  # it binds, far above the others


def _assess(task: str, purpose: str, n: int, p: int, **kw):
    situation = Situation(task=task, purpose=purpose, n_rows=n, n_features=p, n_parameters=p, **kw)
    return get_family("linear").assess(situation)


def test_4b_the_linear_family_is_judged_by_riley_under_prediction_and_by_spv_or_epv_under_inference():
    """The shelf's sentences, on situations where the old rule (10 rows, or 10 events, per
    predictor, for every purpose) and the purpose-scoped ones disagree.

    * Prediction, binary, 10 predictors, 400 rows with 200 events (EPV 20, which the old rule
      passed): Riley et al.'s minimum, from ``pmsampsize`` with R²_CS at 15% of its maximum, is 749,
      so the shelf names Riley et al. and 749. Van Smeden et al. 2019 (*Stat Methods Med Res*
      28:2455, abstract): "EPV does not have a strong relation with metrics of predictive
      performance, and is not an appropriate criterion for (binary) prediction model development
      studies."
    * Inference, least squares, 10 predictors: at 25 rows (2.5 per variable, which the old rule
      called "unstable") nothing is said; at 15 rows (1.5 per variable) the concern names Austin &
      Steyerberg and the threshold of 2. Source check — Austin & Steyerberg, *J Clin Epidemiol*
      2015;68:627–636, abstract: "A minimum of approximately two SPV tended to result in estimation
      of regression coefficients with relative bias of less than 10%. Furthermore, with this
      minimum number of SPV, the standard errors of the regression coefficients were accurately
      estimated and estimated confidence intervals had approximately the advertised coverage
      rates." Conclusion: "Linear regression models require only two SPV for adequate estimation of
      regression coefficients, standard errors, and confidence intervals."
    * Inference, logistic: ten events per parameter remains the rule for coefficients (Peduzzi et
      al. 1996), which the audit judged closer to justified there (§3, the EPV row).
    """
    from turbotab.core.models.sample_size import max_r2_cs

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        riley = pmsampsize(type="b", csrsquared=0.15 * max_r2_cs(0.5), parameters=10, prevalence=0.5,
                           noprint=True)["sample_size"]
    assert riley == 749
    said = _assess("binary", "prediction", 400, 10, n_events=200)
    assert said.fit == "fair"
    assert any("Riley et al. 2020 ask for at least 749" in c for c in said.concerns), said.concerns
    assert not any("rule of thumb" in c or "Peduzzi" in c for c in said.concerns)
    assert _assess("binary", "prediction", 800, 10, n_events=400).concerns == ()

    assert _assess("regression", "inference", 25, 10).concerns == ()
    short = _assess("regression", "inference", 15, 10)
    assert short.fit == "fair"
    assert any("Austin & Steyerberg 2015" in c and "about 2 per parameter" in c
               for c in short.concerns), short.concerns

    assert _assess("binary", "inference", 400, 10, n_events=100).concerns == ()
    few = _assess("binary", "inference", 400, 10, n_events=60)
    assert any("Peduzzi" in c for c in few.concerns), few.concerns


def test_4c_the_shelf_counts_a_categorys_levels_as_parameters(tmp_path):
    """The shelf stage hands the family k − 1 parameters for a category of k levels, as the one-hot
    step encodes it and as Riley et al. count "candidate predictor parameters". On a 300-row table
    with a 12-level category and two numbers (13 parameters, not 3), under prediction: the shelf's
    sentence quotes Riley's minimum for 13 parameters, computed by ``pmsampsize``."""
    from turbotab.core.models.sample_size import R2_SHARE

    rng = np.random.default_rng(12)
    n = 300
    frame = pd.DataFrame({"site": rng.choice([f"s{i:02d}" for i in range(12)], n),
                          "age": rng.normal(50, 10, n), "sugar": rng.gamma(4, 20, n)})
    frame["glucose"] = 90 + 0.2 * frame["age"] + rng.normal(0, 10, n)
    paths = mf.ingest_frame(frame, tmp_path)
    roles = {"site": "covariate", "age": "covariate", "sugar": "exposure"}
    st = mf.state(roles=roles, target="glucose", task="regression", purpose="prediction",
                  energy_adjustment=None)
    cohort = mf.cohort_bundle(np.arange(n), ["site", "age", "sugar"])
    shelf = shelf_stage(mf.context(st, {"cohort": cohort,
                                        "target_info": mf.target_info("regression")}, paths))
    linear = next(f for f in shelf["families"] if f["key"] == "linear")
    y = frame["glucose"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        riley = pmsampsize(type="c", rsquared=R2_SHARE, parameters=13, intercept=float(y.mean()),
                           sd=float(y.std(ddof=1)), noprint=True)["sample_size"]
    assert any(f"300 rows for 13 predictor parameters: Riley et al. 2020 ask for at least {riley:,}"
               in c for c in linear["concerns"]), linear["concerns"]
