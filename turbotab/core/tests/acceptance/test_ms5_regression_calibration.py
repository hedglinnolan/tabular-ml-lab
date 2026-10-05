"""MS5 · Regression calibration (docs/turbotab-next/MODELING_SEQUENCE.md §0 ruling 7, §1.1, §2
"Regression calibration", §5 MS5, §6 chain 2).

The package's six acceptance items, in its order:

1. Univariate calibration from replicate 24-hour recalls, with every outcome-model covariate in
   the calibration model and never the outcome: the calibrated coefficient agrees with a NumPy
   closed form (Rosner, Willett & Spiegelman 1989) and with R ``mecor`` to 1e-6.
2. Multivariate calibration (Rosner, Spiegelman & Willett 1990) of several error-prone intakes
   (every energy source of the all-components model): the calibrated coefficient vector and its
   delta-method covariance agree with the NumPy closed form to 1e-8; this removes the §2 conflict
   between calibration and the all-components model.
3. For residual or density energy models, energy is adjusted per recall day first, then calibrated
   (the order asserted).
4. Inside multiple imputation, calibration runs within each completed copy; the interval comes from
   a whole-chain bootstrap (PSUs within strata under a survey, clusters under repeated units)
   combined with the imputations by a stated rule (Schomaker & Heumann 2018, "Boot MI"); the 95%
   interval's simulated coverage is at least 93% over at least 200 datasets; model-based and
   Rubin-only intervals are refused.
5. Calibration is a declared secondary analysis beside the uncorrected estimate, whose test of no
   association stays the uncorrected model's, labeled "corrects only within-person random error,
   assuming recalls are unbiased for usual intake (customary; sound under that assumption)"; a
   change to the adjustment set invalidates it.
6. MODELING_SEQUENCE §6 chain 2 end to end, with its methods sentence.

**Sources** (quoted in ``turbotab/core/methods/calibration.py`` and the review record
``docs/turbotab-next/audit/modeling-sequence-review.json``): Rosner, Willett & Spiegelman 1989;
Rosner, Spiegelman & Willett 1990; Carroll, Ruppert, Stefanski & Crainiceanu 2006 §4.4 (the
replication-data estimator); Freedman et al. 2011; Boe et al. 2023 (every outcome-model confounder
in the calibration equation); Keogh, Shaw & Gustafson 2020 §6.1.2 (model standard errors too small);
Schomaker & Heumann 2018 (Boot MI, read 2026-10-05 from arXiv:1602.07933, the paper's §3 and §7);
Rao & Wu 1988 (the rescaling bootstrap with n_h − 1 PSUs).

**References, each independent of the code under test:** R ``mecor`` (``MeasErrorRandom``, run as
a subprocess on a CSV written here; the test that needs R skips without it); NumPy by hand from the
CSV's recall days (the pooled within-person covariance, the Schur complement of the (n − 1)
covariance of the mean recalls and the covariates, least squares, and β = (Γᵀ)⁻¹β̃); the delta
method's Jacobian by complex-step differentiation (exact to machine precision, a different route
from the engine's analytic one); statsmodels for the uncorrected tables; and simulations with a
known truth (usual intakes and coefficients drawn by the generator).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from turbotab.core.decisions import ProjectState, Refusal, SetMeasurementError, validate
from turbotab.core.methods import calibration as RC
from turbotab.core.tests.acceptance.rc_fixtures import (ATWATER, SOURCES, chain2_table, needs_r,
                                                        protein_recalls, replicate_people, run_r)
from turbotab.core.tests.graph_runner import GraphRun

LABEL = ("corrects only within-person random error, assuming recalls are unbiased for usual intake "
         "(customary; sound under that assumption)")
CHAIN_2_REVIEWERS = (
    "Usual intakes of all energy sources were calibrated jointly with every outcome-model covariate "
    "in the calibration model; the substitution is the difference of calibrated coefficients in the "
    "all-components model; CIs by a bootstrap resampling PSUs within strata that repeats "
    "calibration, imputation and the outcome model.")


# ── the closed form, by hand ─────────────────────────────────────────────────


def by_hand(days: np.ndarray, Z: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> dict:
    """Rosner et al.'s closed form from the recall days (``days``: people × k × p, every person on
    k days), the covariates ``Z`` and the outcome ``y``, (weighted when ``w`` is given):

    * Σ_uu = Σ_i w̃_i Σ_j (W_ij − W̄_i)(W_ij − W̄_i)ᵀ / Σ_i w̃_i (k − 1), w̃ the weights scaled to sum
      to n (all ones without weights);
    * S = the (n − 1) (weighted) covariance matrix of (W̄, Z), and its Schur complement S_W·Z;
    * Γᵀ = I − S_W·Z⁻¹ Σ_uu / k;
    * β̃ = the (weighted) least-squares coefficients of y on (1, W̄, Z), β = (Γᵀ)⁻¹ β̃."""
    n, k, p = days.shape
    wt = np.ones(n) if w is None else np.asarray(w, dtype=float) * n / float(np.sum(w))
    Wb = days.mean(axis=1)
    dev = days - Wb[:, None, :]
    Su = np.einsum("i,ikp,ikq->pq", wt, dev, dev) / (wt.sum() * (k - 1))
    V = np.column_stack([Wb, Z])
    centre = (wt[:, None] * V).sum(axis=0) / wt.sum()
    S = ((V - centre) * wt[:, None]).T @ (V - centre) / (n - 1)
    S_wz = S[:p, :p] - S[:p, p:] @ np.linalg.solve(S[p:, p:], S[p:, :p])
    A = np.eye(p) - np.linalg.solve(S_wz, Su) / k
    D = np.column_stack([np.ones(n), Wb, Z])
    root = np.sqrt(wt)
    coef = np.linalg.lstsq(D * root[:, None], y * root, rcond=None)[0]
    naive = coef[1:p + 1]
    return {"naive": naive, "calibrated": np.linalg.solve(A, naive), "Su": Su, "S_wz": S_wz,
            "coef": coef, "D": D}


def rel(a: Any, b: Any) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))


# ── (1) univariate: Rosner 1989's closed form and mecor ──────────────────────


@needs_r
@pytest.mark.parametrize("k", [2, 3])
def test_1_univariate_calibration_is_rosners_closed_form_and_mecors(k, tmp_path):
    """One error-prone intake, k recalls each, three covariates in the outcome model (age, sex and
    a third that predicts the intake). The engine's calibrated coefficient (``methods.calibration.
    correct``) is (a) NumPy's closed form β = β̃/λ by hand from the CSV's recall days (λ = 1 − (σ²_u
    /k)/S_W·Z) and (b) R mecor's ``MeasErrorRandom(substitute = W̄, variance = σ²_u/k)`` with every
    covariate in the formula, σ²_u computed in R from the recalls by the one-way ANOVA, both to
    1e-6 relative (measured about 1e-15); the uncorrected coefficient is mecor's uncorrected fit."""
    rng = np.random.default_rng(40 + k)
    n = 900
    age, sex, w, person, y, x = replicate_people(rng, n, k)
    third = rng.normal(0, 1, n) + 0.3 * (x - x.mean())
    rec = RC.Recalls.of(w, person, n)
    X = np.column_stack([rec.means()[:, 0], age, sex, third])
    got = RC.correct(X, [0], rec, y)
    days = w.reshape(n, k, 1)
    hand = by_hand(days, np.column_stack([age, sex, third]), y)
    frame = pd.DataFrame({"y": y, "age": age, "sex": sex, "third": third,
                          **{f"w{j + 1}": days[:, j, 0] for j in range(k)}})
    frame.to_csv(tmp_path / "recalls.csv", index=False)
    ws = " + ".join(f"dd$w{j + 1}" for j in range(k))
    devs = " + ".join(f"(dd$w{j + 1} - dd$wbar)^2" for j in range(k))
    ref = run_r(f"""
suppressMessages({{library(mecor); library(jsonlite)}})
dd <- read.csv("recalls.csv")
dd$wbar <- ({ws}) / {k}
s2u <- sum({devs}) / (nrow(dd) * ({k} - 1))
m <- mecor(y ~ MeasErrorRandom(substitute = wbar, variance = s2u / {k}) + age + sex + third,
           data = dd, method = "standard")
writeLines(toJSON(list(corrected = unname(m$corfit$coef["cor_wbar"]),
                       naive = unname(m$uncorfit$coef["wbar"]), s2u = s2u),
                  digits = NA, auto_unbox = TRUE), "out.json")
""", tmp_path)
    print(f"\n[k={k}] engine {got.estimate[0]:.10f} | by hand {hand['calibrated'][0]:.10f} | "
          f"mecor {ref['corrected']:.10f} | naive {got.naive[0]:.6f}")
    assert got.estimate[0] == pytest.approx(hand["calibrated"][0], rel=1e-6)
    assert got.estimate[0] == pytest.approx(ref["corrected"], rel=1e-6)
    assert got.naive[0] == pytest.approx(ref["naive"], rel=1e-9)
    assert got.calibration.sigma2_u == pytest.approx(ref["s2u"], rel=1e-9)
    # Rosner's division: the calibrated coefficient is the uncorrected one over λ.
    lam = got.calibration.slope(k)[0, 0]
    assert got.estimate[0] == pytest.approx(got.naive[0] / lam, rel=1e-12)
    assert 0.3 < lam < 0.9 and got.naive[0] < got.estimate[0]


def test_1_every_outcome_model_covariate_is_in_the_calibration_and_never_the_outcome():
    """Boe et al. 2023: "the calibration equation should include all confounders included in the
    outcome model". The engine's calibration covariates are every other column of the outcome
    model's matrix; with one left out the answer is NumPy's closed form *without* it, which is not
    the engine's (so the full set is what the engine used). The outcome is never in it: shuffling
    the outcome leaves every calibrated value unchanged to the last bit, and lockbox §06's scope
    test (``contracts.observed_scope``) observes "training rows" (other people's recalls move it,
    the outcome does not), the scope the contract declares."""
    from turbotab.core.contracts import contract, observed_scope

    rng = np.random.default_rng(3)
    n, k = 800, 2
    age, sex, w, person, y, x = replicate_people(rng, n, k)
    third = rng.normal(0, 1, n) + 0.5 * (x - x.mean())
    rec = RC.Recalls.of(w, person, n)
    X = np.column_stack([age, rec.means()[:, 0], sex, third])
    got = RC.correct(X, [1], rec, y)
    assert got.columns == [1] and got.covariates == [0, 2, 3]
    days = w.reshape(n, k, 1)
    full = by_hand(days, np.column_stack([age, sex, third]), y)
    short = by_hand(days, np.column_stack([age, sex]), y)
    assert got.estimate[1] == pytest.approx(full["calibrated"][0], rel=1e-9)
    assert abs(got.calibration.slope(k)[0, 0] - (1 - full["Su"][0, 0] / k / full["S_wz"][0, 0])) \
        < 1e-12
    # The omitted-covariate calibration is a different (wrong) λ: the covariate matters here.
    lam_short = 1 - short["Su"][0, 0] / k / short["S_wz"][0, 0]
    assert abs(got.calibration.slope(k)[0, 0] - lam_short) > 0.01
    shuffled = rng.permutation(y)
    again = RC.correct(X, [1], rec, shuffled)
    assert np.array_equal(again.calibration.calibrated, got.calibration.calibrated)

    def fit_transform(frame: pd.DataFrame, reference: pd.Series, yy: np.ndarray) -> np.ndarray:
        r = RC.Recalls.of(frame[["d1", "d2"]].to_numpy().ravel(), np.repeat(np.arange(len(frame)), 2),
                          len(frame))
        Z = frame[["age", "sex"]].to_numpy()
        return RC.calibrate(r, Z).calibrated[:, 0]

    frame = pd.DataFrame({"d1": days[:200, 0, 0], "d2": days[:200, 1, 0], "age": age[:200],
                          "sex": sex[:200]})
    scope = observed_scope(fit_transform, frame, np.zeros(200, dtype=bool), y[:200], row=7,
                           columns=["d1", "d2"])
    assert scope == "training_fold" == contract("regression_calibration").scope


# ── (2) multivariate: Rosner 1990's closed form and its delta-method covariance ───────────────


def _complex_step_covariance(b: np.ndarray, Vb: np.ndarray, Su: np.ndarray, S: np.ndarray, k: int,
                             df_u: float, df_s: float) -> tuple[np.ndarray, np.ndarray]:
    """β = (I − S⁻¹ Σ_uu/k)⁻¹ b and its delta-method covariance, the Jacobian by complex-step
    differentiation (f(θ + ih e)/h's imaginary part, h = 1e-30: no subtraction, so exact to machine
    precision), with Cov(b) = Vb and the Wishart covariances of Σ_uu (df_u) and S (df_s), the three
    independent."""
    p = len(b)

    def f(bb, uu, ss):
        return np.linalg.solve(np.eye(p) - np.linalg.solve(ss, uu) / k, bb)

    pairs = [(a, c) for a in range(p) for c in range(a, p)]
    h = 1e-30

    def column(which: int, a: int, c: int) -> np.ndarray:
        args = [b.astype(complex), Su.astype(complex), S.astype(complex)]
        if which == 0:
            args[0][a] += 1j * h
        else:
            E = np.zeros((p, p), dtype=complex)
            E[a, c] += 1j * h
            if a != c:
                E[c, a] += 1j * h
            args[which] = args[which] + E
        return f(*args).imag / h

    Jb = np.column_stack([column(0, a, 0) for a in range(p)])
    Ju = np.column_stack([column(1, a, c) for a, c in pairs])
    Js = np.column_stack([column(2, a, c) for a, c in pairs])

    def wishart(M: np.ndarray, df: float) -> np.ndarray:
        return np.array([[(M[a, e] * M[c, g] + M[a, g] * M[c, e]) / df for e, g in pairs]
                         for a, c in pairs])

    V = Jb @ Vb @ Jb.T + Ju @ wishart(Su, df_u) @ Ju.T + Js @ wishart(S, df_s) @ Js.T
    return f(b, Su, S), V


def _five_sources(rng: np.random.Generator, n: int, k: int):
    """Five energy sources' usual kcal (correlated through total intake), k recall days each with
    correlated classical errors, two covariates, an outcome on the usual intakes."""
    from turbotab.core.tests.acceptance.rc_fixtures import recall_days, usual_kcal

    age = rng.uniform(20, 80, n)
    female = rng.integers(0, 2, n).astype(float)
    usual = usual_kcal(rng, n, age, female)
    person, days = recall_days(rng, usual, np.full(n, k))
    names = [*SOURCES, "other"]
    beta = np.array([-0.01, 0.02, 0.005, 0.03, 0.0])
    U = np.column_stack([usual[s] for s in names])
    y = 110 + U @ beta + 0.35 * age - 3 * female + rng.normal(0, 6, n)
    D = np.stack([np.column_stack([days[s] for s in names])[person == i] for i in range(n)])
    return age, female, D, person, y, beta


def test_2_multivariate_calibration_and_its_delta_covariance_are_the_numpy_closed_form():
    """Every energy source of the all-components model (protein, fat, carbohydrate, alcohol and the
    rest, in kcal) calibrated jointly, two recalls each, with age and sex in the model. The engine's
    calibrated coefficient vector (``correct`` with five error-prone columns) is NumPy's closed form
    β = (Γᵀ)⁻¹β̃ by hand to 1e-8, and its delta-method covariance (``delta_covariance``, analytic) is
    the complex-step covariance by hand (the outcome model's σ²(DᵀD)⁻¹ by statsmodels-free NumPy,
    Wishart covariances of Σ_uu on Σ(k − 1) and of S_W·Z on n − 3 degrees of freedom) to 1e-8. Both
    measured about 1e-14."""
    rng = np.random.default_rng(12)
    n, k = 1200, 2
    age, female, D, person, y, beta = _five_sources(rng, n, k)
    rec = RC.Recalls.of(D.reshape(n * k, 5), np.repeat(np.arange(n), k), n)
    X = np.column_stack([rec.means(), age, female])
    J = [0, 1, 2, 3, 4]
    got = RC.correct(X, J, rec, y)
    hand = by_hand(D, np.column_stack([age, female]), y)
    assert rel(got.estimate[J], hand["calibrated"]) < 1e-8
    assert rel(got.naive[J], hand["naive"]) < 1e-10
    Dm = hand["D"]
    resid = y - Dm @ hand["coef"]
    Vb = (resid @ resid / (n - Dm.shape[1]) * np.linalg.inv(Dm.T @ Dm))[1:6, 1:6]
    beta_cs, V_cs = _complex_step_covariance(hand["naive"], Vb, hand["Su"], hand["S_wz"], k,
                                             df_u=n * (k - 1), df_s=n - 3)
    V = RC.delta_covariance(got, RC.model_covariance(RC.ols_fit, got.X, y)[np.ix_(J, J)])
    assert rel(beta_cs, hand["calibrated"]) < 1e-12
    assert rel(V, V_cs) < 1e-8
    print(f"\ncalibrated {np.round(got.estimate[J], 4)} (truth {beta}) | delta SE "
          f"{np.round(np.sqrt(np.diag(V)), 4)}")
    # Several error-prone intakes at once: the slope of E[X | W̄, Z] is a matrix, not a scalar.
    gamma = got.calibration.slope(k)
    assert np.max(np.abs(gamma - np.diag(np.diag(gamma)))) > 0.01


def test_2_by_simulation_joint_calibration_recovers_the_truth_where_one_at_a_time_does_not():
    """Two error-prone intakes whose usual intakes (r = 0.6) and day-to-day errors (r = 0.5) are
    correlated, true coefficients (0.5, −0.3). Over 200 datasets of 800 people with two recalls:
    the joint calibration averages within 0.03 of each (observed 0.499 and −0.299; Monte Carlo
    errors about 0.004), each intake calibrated on its own while the other stays uncorrected (the
    univariate method applied one nutrient at a time, which the contract refuses) is off by more
    than 0.1 (observed 0.657 and −0.371: the first inflated by 31%), and the delta-method standard
    error averages within 15% of the estimates' spread (observed within 4%)."""
    rng = np.random.default_rng(2026)
    truth = np.array([0.5, -0.3])
    cov_x = np.array([[4.0, 2.4], [2.4, 4.0]])
    cov_u = np.array([[5.0, 2.5], [2.5, 5.0]])
    joint, alone, ses = [], [], []
    for _ in range(200):
        n, k = 800, 2
        age = rng.normal(50, 10, n)
        x = rng.multivariate_normal([10, 8], cov_x, n) + 0.05 * (age - 50)[:, None]
        person = np.repeat(np.arange(n), k)
        W = x[person] + rng.multivariate_normal([0, 0], cov_u, n * k)
        y = 1 + x @ truth + 0.03 * age + rng.normal(0, 1.5, n)
        rec = RC.Recalls.of(W, person, n)
        X = np.column_stack([rec.means(), age])
        both = RC.correct(X, [0, 1], rec, y)
        joint.append(both.estimate[:2])
        V = RC.delta_covariance(both, RC.model_covariance(RC.ols_fit, both.X, y)[:2, :2])
        ses.append(np.sqrt(np.diag(V)))
        one = [RC.correct(X, [j], RC.Recalls.of(W[:, j], person, n), y).estimate[j] for j in (0, 1)]
        alone.append(one)
    joint, alone, ses = np.array(joint), np.array(alone), np.array(ses)
    print(f"\njoint {joint.mean(0).round(3)} | one at a time {alone.mean(0).round(3)} | delta SE "
          f"{ses.mean(0).round(4)} vs spread {joint.std(0, ddof=1).round(4)}")
    assert np.all(np.abs(joint.mean(0) - truth) < 0.03)
    assert np.max(np.abs(alone.mean(0) - truth)) > 0.1
    assert np.all(np.abs(ses.mean(0) / joint.std(0, ddof=1) - 1) < 0.15)


# ── through the stage graph ──────────────────────────────────────────────────

NUTRIENTS = [f"{s}_g" for s in SOURCES]
CHAIN_ROLES = {"seqn": "identifier", "day": "excluded", "sdmvstra": "design", "sdmvpsu": "design",
               "wtdr2d": "design", "age": "covariate", "sex": "covariate", "bmi": "covariate",
               "energy_kcal": "energy", **{c: "exposure" for c in NUTRIENTS}}
POPULATION = {"estimand": "population", "weight": "wtdr2d", "strata": "sdmvstra", "psu": "sdmvpsu"}


def chain_state(**slots: Any) -> ProjectState:
    base: dict[str, Any] = {
        "lens": ["dietary"], "target": "sbp", "task": "regression", "purpose": "inference",
        "grain": {"grain": "repeated", "id_column": "seqn"}, "unit": "unit",
        "repeat_kind": {"repeat_kind": "repeats"}, "aggregation": {"method": "mean"},
        "temporal": {"temporal": False}, "roles": dict(CHAIN_ROLES), "exclusions": [],
        "missing": {"strategy": "complete_case"}, "split": {"holdout": 0.0, "seed": 0, "folds": 5},
        "energy_adjustment": {"method": "all_components", "energy_column": "energy_kcal",
                              "nutrients": NUTRIENTS},
        "column_units": {c: {"unit": "g", "days": None} for c in NUTRIENTS},
        "survey": {"estimand": "sample"},
        "substitution": {"donor": "fat_g", "recipient": "carbohydrate_g", "step_kcal": 100.0},
        "models": ["linear"],
        "shape_confirmations": {"code_or_count:sdmvstra": "code", "code_or_count:sdmvpsu": "code",
                                "code_or_count:day": "code"}}
    base.update(slots)
    return ProjectState.model_validate(base)


def ctx_of(state: ProjectState, frame: pd.DataFrame, out: dict) -> dict:
    """The validators' context, as the server builds it."""
    info = {c["name"]: c for c in out["ingest"]["columns"]}
    return {"state": state, "columns": list(frame.columns), "column_info": info,
            "artifact": lambda name: getattr(out.get(name), "data", out.get(name))}


def declared(state: ProjectState, frame: pd.DataFrame, out: dict, **answer: Any) -> ProjectState:
    """``state`` with the calibration answer recorded through the validators and completion."""
    decision = validate(SetMeasurementError(**{"method": "regression_calibration", **answer}),
                        ctx_of(state, frame, out))
    return ProjectState.model_validate({**state.model_dump(),
                                        "measurement_error": decision.model_dump(exclude={"kind"})})


def kcal_days(frame: pd.DataFrame) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """Each person's recall days of every energy source's kcal, by pandas from the CSV (Atwater's
    factors, the rest as total energy less the four), people in ``seqn`` order: (people × days ×
    5, the matrix's names, the person table)."""
    f = frame.sort_values(["seqn", "day"])
    kcal = {f"kcal_from_{c}": ATWATER[c.removesuffix("_g")] * f[c] for c in NUTRIENTS}
    kcal["kcal_from_other"] = f["energy_kcal"] - sum(kcal.values())
    names = ["kcal_from_other", *[f"kcal_from_{c}" for c in NUTRIENTS]]
    k = int(f.groupby("seqn").size().iloc[0])
    D = np.stack([np.asarray(kcal[c], dtype=float).reshape(-1, k) for c in names], axis=2)
    people = f.groupby("seqn", sort=True).first()
    return D, names, people


@pytest.fixture(scope="module")
def complete(tmp_path_factory) -> dict:
    """Chain 2's table with no blank, the sample answer, the all-components model: the graph run
    once with the calibration declared through the validators."""
    folder = tmp_path_factory.mktemp("ms5_complete")
    frame, truth = chain2_table(seed=4, per_psu=20, missing_share=0.0)
    path = folder / "chain.csv"
    frame.to_csv(path, index=False)
    run = GraphRun(path, folder / "project")
    state = chain_state()
    out = run.run(state, upto=["findings"])
    state = declared(state, frame, out, n_boot=60)
    out = run.run(state, upto=["calibration", "fit"])
    yield {"frame": frame, "truth": truth, "state": state, "out": run.public(out), "run": run}
    run.close()


def test_2_the_all_components_model_is_calibrated_jointly_as_numpy_does_it(complete):
    """MODELING_SEQUENCE §0 ruling 7: "Without it, RC conflicts with the all-components model that
    §12 ranks first". Through the stage graph, the all-components model's five kcal terms (each
    source and the rest) are calibrated jointly with sex, age and BMI in the calibration equation;
    each calibrated coefficient, the uncorrected one, Σ_uu and the attenuation matrix are NumPy's
    from the CSV's recall days to 1e-8 (Atwater's factors from the generator, the rest as energy
    less the four); the uncorrected coefficients are the fit stage's; the delta-method check is
    defined (two recalls each, unweighted, independent people). The §2 conflict is gone from the
    one registry: the contract enables the all-components model, and the survey contract's
    calibration relation is design-based, not a block."""
    from turbotab.core.contracts import contract

    cal, fit = complete["out"]["calibration"], complete["out"]["fit"]
    assert cal["applies"] and cal["calibration"] == "multivariate"
    D, names, people = kcal_days(complete["frame"])
    assert cal["calibrated"] == names
    assert cal["covariates"] == ["sex_M", "age", "bmi"]
    Z = np.column_stack([(people["sex"] == "M").astype(float), people["age"], people["bmi"]])
    hand = by_hand(D, Z, people["sbp"].to_numpy(dtype=float))
    rows = {e["feature"]: e for e in cal["exposures"]}
    assert set(rows) == set(names[1:])  # the four sources are the exposures; the rest is calibrated
    table = {c["feature"]: c for c in fit["models"][0]["coefficients"]}
    for j, name in enumerate(names):
        if name not in rows:
            continue
        e = rows[name]
        assert e["estimate"] == pytest.approx(hand["calibrated"][j], rel=1e-8)
        assert e["naive"] == pytest.approx(hand["naive"][j], rel=1e-8)
        assert e["naive"] == pytest.approx(table[name]["estimate"], rel=1e-9)
        assert e["p"] == pytest.approx(table[name]["p"], rel=1e-9)
        assert e["within_variance"] == pytest.approx(hand["Su"][j, j], rel=1e-8)
        assert e["delta_se"] is not None and e["delta_se"] > 0
        assert e["ci_low"] < e["estimate"] < e["ci_high"]
    assert rel(cal["within_covariance"], hand["Su"]) < 1e-8
    gamma = np.eye(5) - np.linalg.solve(hand["S_wz"], hand["Su"]) / 2
    assert rel(np.asarray(cal["attenuation"]).T, gamma) < 1e-8
    assert cal["resampling"] == "persons" and not cal["weighted"] and cal["n_boot_ok"] == 60
    rc = contract("regression_calibration")
    assert rc.relation("all_components").kind == "enables"
    assert not [r for r in rc.relations if r.kind == "conflicts" and "all-components" in r.target]
    survey = contract("survey_population").relation("regression_calibration")
    assert survey.kind == "implies" and survey.rung is None


# ── (3) energy adjusted per recall day first ─────────────────────────────────

PROTEIN_ROLES = {"participant_id": "identifier", "day": "excluded", "age": "covariate",
                 "sex": "covariate", "energy_kcal": "energy", "protein_g": "exposure"}


def protein_state(method: str, **slots: Any) -> ProjectState:
    base: dict[str, Any] = {
        "lens": ["dietary"], "target": "ldl", "task": "regression", "purpose": "inference",
        "grain": {"grain": "repeated", "id_column": "participant_id"}, "unit": "unit",
        "repeat_kind": {"repeat_kind": "repeats"}, "aggregation": {"method": "mean"},
        "temporal": {"temporal": False}, "roles": dict(PROTEIN_ROLES), "exclusions": [],
        "missing": {"strategy": "complete_case"}, "split": {"holdout": 0.0, "seed": 0, "folds": 5},
        "energy_adjustment": {"method": method, "energy_column": "energy_kcal",
                              "nutrients": ["protein_g"]},
        "models": ["linear"], "shape_confirmations": {"code_or_count:day": "code"}}
    base.update(slots)
    return ProjectState.model_validate(base)


@pytest.fixture(scope="module")
def protein(tmp_path_factory) -> dict:
    folder = tmp_path_factory.mktemp("ms5_protein")
    frame = protein_recalls(seed=8, n=700)
    path = folder / "recalls.csv"
    frame.to_csv(path, index=False)
    run = GraphRun(path, folder / "project")
    found = {}
    for method in ("residual_energy_dropped", "density", "residual"):
        state = protein_state(method)
        out = run.run(state, upto=["findings"])
        state = declared(state, frame, out, n_boot=50)
        found[method] = run.public(run.run(state, upto=["calibration", "fit"]))
    yield {"frame": frame, "runs": found}
    run.close()


def test_3_energy_is_adjusted_on_each_recall_day_and_then_calibrated(protein):
    """MODELING_SEQUENCE §2: "for residual or density models, energy is adjusted per recall day
    first, then calibrated". On two recalls of protein and energy per person, by pandas from the
    CSV:

    * the residual method (energy out of the model): the slope b of the person means N̄ on Ē (what
      the energy step fits on the analysis rows), each day adjusted with it, N_d − b(E_d − mean Ē);
      the engine's day-to-day variance is that of the adjusted days to 1e-9, and its calibrated
      coefficient is the closed form on them to 1e-8;
    * the density: each day's own ratio N_d/E_d, its day-to-day variance to 1e-9, the closed form
      on the people's N̄/Ē (the value the model saw) to 1e-8;
    * the other order (calibrate the raw protein, then adjust) is not what ran: its day-to-day
      variance differs from the adjusted one's by more than half;
    * with energy kept in the residual model, the adjusted protein and energy are calibrated
      jointly (both from the recalls).

    The artifact records the run order: the energy model refit, then each recall day through it,
    then the calibration, then the outcome model."""
    f = protein["frame"].sort_values(["participant_id", "day"])
    n = f["participant_id"].nunique()
    N = f["protein_g"].to_numpy(dtype=float).reshape(n, 2)
    E = f["energy_kcal"].to_numpy(dtype=float).reshape(n, 2)
    people = f.groupby("participant_id", sort=True).first()
    Z = np.column_stack([(people["sex"] == "M").astype(float), people["age"]])
    y = people["ldl"].to_numpy(dtype=float)
    Nb, Eb = N.mean(1), E.mean(1)
    b = np.polyfit(Eb, Nb, 1)[0]
    adjusted = N - b * (E - Eb.mean())
    residual = protein["runs"]["residual_energy_dropped"]["calibration"]
    (row,) = residual["exposures"]
    hand = by_hand(adjusted[:, :, None], Z, y)
    assert row["feature"] == "protein_g_adj" and residual["calibration"] == "univariate"
    assert row["within_variance"] == pytest.approx(hand["Su"][0, 0], rel=1e-9)
    assert row["estimate"] == pytest.approx(hand["calibrated"][0], rel=1e-8)
    raw = by_hand(N[:, :, None], Z, y)["Su"][0, 0]
    assert abs(raw / row["within_variance"] - 1) > 0.5
    assert residual["order"] == ["the energy model refit, then applied to each recall day",
                                 "regression calibration", "the outcome model",
                                 "the whole-chain bootstrap"]
    assert residual["methods"] == (
        "The usual intake of `protein_g_adj` was calibrated with every outcome-model covariate in "
        "the calibration model; CIs by a bootstrap resampling participants that repeats calibration "
        "and the outcome model. Calibration (Rosner, Willett & Spiegelman 1989, Stat Med 8:1051; "
        "Carroll, Ruppert, Stefanski & Crainiceanu 2006, Measurement Error in Nonlinear Models, "
        "§4.4) used the within-person variance of the recalls of the 700 participants with two or "
        "more (2 recalls each). Energy was adjusted on each recall day before calibration. The "
        "interval is the percentile interval of 50 bootstrap resamples. It is a declared secondary "
        f"analysis beside the uncorrected estimate, whose test of no association is the primary's; "
        f"it {LABEL}.")

    density = protein["runs"]["density"]["calibration"]
    (drow,) = density["exposures"]
    ratio = N / E
    within = float(((ratio - ratio.mean(1, keepdims=True)) ** 2).sum() / n)
    # the people's value is the ratio of their means; their days add only their spread
    days = (ratio - ratio.mean(1, keepdims=True) + (Nb / Eb)[:, None])[:, :, None]
    dhand = by_hand(days, Z, y)
    assert drow["feature"] == "protein_g_per_energy_kcal"
    assert drow["within_variance"] == pytest.approx(within, rel=1e-9)
    assert drow["estimate"] == pytest.approx(dhand["calibrated"][0], rel=1e-8)

    kept = protein["runs"]["residual"]["calibration"]
    assert kept["calibration"] == "multivariate"
    assert kept["calibrated"] == ["energy_kcal", "protein_g_adj"]  # the matrix's order
    both = by_hand(np.stack([E, adjusted], axis=2), Z, y)
    assert kept["exposures"][0]["feature"] == "protein_g_adj"
    assert kept["exposures"][0]["estimate"] == pytest.approx(both["calibrated"][1], rel=1e-8)
    assert rel(kept["within_covariance"], both["Su"]) < 1e-9


# ── (4) inside multiple imputation, the whole-chain bootstrap ─────────────────


def test_4_the_bootstrap_draws_whole_psus_within_strata_as_rao_and_wu_do():
    """Rao & Wu's (1988) rescaling bootstrap with n_h − 1 PSUs (R survey's ``subbootstrap``):
    every replicate draws n_h − 1 of a stratum's n_h PSUs with replacement, each drawn person
    whole and weighted n_h/(n_h − 1); a stratum with a single PSU is kept as it is. Over 4,000
    replicates the bootstrap variance of a weighted total is the linearization variance by hand,
    Σ_h n_h/(n_h − 1) Σ_j (t_hj − t̄_h)² (exact in expectation for a linear statistic), to 6% (the
    Monte Carlo error is about 2%). Clusters: every replicate holds whole clusters."""
    rng = np.random.default_rng(9)
    strata = np.repeat(np.arange(6), 40)
    # strata 0–2: two PSUs of 20; strata 3–4: three PSUs (12, 14, 14); stratum 5: one PSU (lonely)
    sizes = [20, 20] * 3 + [12, 14, 14] * 2 + [40]
    psu = np.repeat(np.arange(len(sizes)), sizes)
    w = rng.uniform(1, 3, 240)
    v = rng.normal(5, 2, 240) + psu * 0.1
    totals = pd.Series(w * v).groupby(psu).sum()
    of = pd.Series(strata).groupby(psu).first()
    lin = 0.0
    for h, t in totals.groupby(of):
        if len(t) > 1:
            lin += len(t) / (len(t) - 1) * float(((t - t.mean()) ** 2).sum())
    estimates = []
    for _ in range(4000):
        d = RC.psu_draw(strata, psu, rng)
        estimates.append(float(np.sum(w[d.rows] * d.factor * v[d.rows])))
        assert set(np.unique(strata[d.rows])) == set(range(6))
    assert np.var(estimates, ddof=1) == pytest.approx(lin, rel=0.06)
    d = RC.psu_draw(strata, psu, rng)
    for h, (draws, factor) in enumerate([(1, 2.0)] * 3 + [(2, 1.5)] * 2 + [(1, 1.0)]):
        on = strata[d.rows] == h
        assert len(np.unique(d.unit[on])) == draws  # n_h − 1 draws (one for a lonely stratum)
        assert np.all(d.factor[on] == factor)
        for u in np.unique(d.unit[on]):  # each drawn PSU whole
            members = d.rows[d.unit == u]
            assert len(members) == int(np.sum(psu == psu[members[0]]))
    codes = rng.integers(0, 30, 200)
    c = RC.cluster_draw(codes, rng)
    for u in np.unique(c.unit):
        members = c.rows[c.unit == u]
        assert set(codes[members]) == {codes[members[0]]}
        assert len(members) == int(np.sum(codes == codes[members[0]]))


def test_4_under_repeated_units_the_whole_chain_resamples_whole_clusters(tmp_path):
    """People two to a household (the grouping the user confirmed as a cluster), the sample
    answer: the whole-chain bootstrap resamples whole households (ruling 7: "resampling PSUs within
    strata, or clusters"), the uncorrected test beside the calibrated estimate is the fit stage's
    cluster-robust one, the calibrated estimate is still the closed form by hand (clustering moves
    the interval, not the point) to 1e-8, and the delta-method check, which knows nothing of
    clusters, is not shown."""
    frame, _ = chain2_table(seed=6, per_psu=20, missing_share=0.0, household=True)
    path = tmp_path / "households.csv"
    frame.to_csv(path, index=False)
    run = GraphRun(path, tmp_path / "project")
    try:
        state = chain_state(roles={**CHAIN_ROLES, "household": "cluster"},
                            reading_confirmations={"cluster:household": "yes"})
        out = run.run(state, upto=["findings"])
        state = declared(state, frame, out, n_boot=50)
        out = run.public(run.run(state, upto=["calibration", "fit"]))
    finally:
        run.close()
    cal, fit = out["calibration"], out["fit"]
    assert cal["applies"] and cal["resampling"] == "clusters"
    assert "CIs by a bootstrap resampling whole clusters that repeats calibration and the outcome " \
           "model." in cal["methods"]
    # A resample the calibration cannot carry (here one, whose true-intake covariance estimate is
    # not positive definite) is left out and said, never silently: at least 90% of them must stand
    # for an interval to be shown.
    assert cal["n_boot_ok"] == 49 and cal["concerns"][0] == (
        "49 of 50 bootstrap resamples could be calibrated (1 had no positive true-intake "
        "covariance); the interval rests on those.")
    assert fit["models"][0]["inference"]["grouped_by"] == "household"
    D, names, people = kcal_days(frame)
    Z = np.column_stack([(people["sex"] == "M").astype(float), people["age"], people["bmi"]])
    hand = by_hand(D, Z, people["sbp"].to_numpy(dtype=float))
    table = {c["feature"]: c for c in fit["models"][0]["coefficients"]}
    for e in cal["exposures"]:
        assert e["estimate"] == pytest.approx(hand["calibrated"][names.index(e["feature"])], rel=1e-8)
        primary = table[e["feature"]]
        assert (e["naive_ci_low"], e["naive_ci_high"], e["p"]) == pytest.approx(
            (primary["ci_low"], primary["ci_high"], primary["p"]), rel=1e-9)
        assert e["delta_se"] is None


def _boot_mi_dataset(rng: np.random.Generator, n: int = 300):
    """True intake X (slope 0.5 on the outcome) given age and a covariate z; one recall each, a
    second for a quarter of the people (a replicate subsample, as many cohorts have); z blank for
    about 30%, more often at high outcomes and ages (missing at random given them)."""
    age = rng.normal(50, 10, n)
    z = rng.normal(0, 1, n)
    x = 10 + 0.05 * age + 0.8 * z + rng.normal(0, 2, n)
    k = np.where(rng.random(n) < 0.25, 2, 1)
    person = np.repeat(np.arange(n), k)
    w = x[person] + rng.normal(0, 2.0, person.size)
    y = 2 + 0.5 * x + 0.03 * age + 0.6 * z + rng.normal(0, 2, n)
    p = 1 / (1 + np.exp(-(-1.2 + 0.25 * (y - y.mean()) + 0.03 * (age - 50))))
    return age, np.where(rng.random(n) < p, np.nan, z), w, person, y


def test_4_by_simulation_the_whole_chain_interval_covers_and_a_rubin_only_one_does_not():
    """Known truth: slope 0.5 per unit of true intake. Over 200 datasets of 300 people (two recalls
    for a quarter of them, a covariate missing at random for about 30%), each analysis is the one
    the stage runs (the methods' own pieces): m = 10 copies by chained equations with the outcome,
    the mean recall and age in the imputation model (one incomplete variable, so one sweep is the
    exact conditional draw); the calibration inside each copy, the estimate their mean; the 95%
    interval Boot MI's (Schomaker & Heumann's eq. 3.5): 100 bootstrap resamples of people with
    their recalls, each imputed ``BOOT_COPIES`` = 5 times and calibrated in each copy, the
    percentile interval of their means. It covers 0.5 in at least 93% of the datasets (observed
    0.945; the binomial Monte Carlo error at 0.95 is 0.015). Rubin's rules over each copy's
    model-based variance, the interval the leash refuses, cover only about 0.86 here: they ignore
    the uncertainty of the attenuation itself (Keogh et al. 2020 §6.1.2). About 2.5 minutes."""
    from turbotab.core.methods.imputation import chained_equations, pool_scalar

    kinds = {"age": "numeric", "z": "numeric", "wbar": "numeric", "y": "numeric"}

    def calibrated(copies: list[pd.DataFrame], rec: RC.Recalls, y: np.ndarray,
                   with_variance: bool = False) -> tuple[list[float], list[float]]:
        est, var = [], []
        for c in copies:
            X = np.column_stack([rec.means()[:, 0], c["age"], c["z"]])
            got = RC.correct(X, [0], rec, y)
            est.append(float(got.estimate[0]))
            if with_variance:
                Xc = np.column_stack([got.calibration.calibrated[:, 0], X[:, 1:]])
                var.append(float(RC.model_covariance(RC.ols_fit, Xc, y)[0, 0]))
        return est, var

    covered = rubin = 0
    estimates = []
    reps = 200
    for r in range(reps):
        rng = np.random.default_rng(1000 + r)
        age, z, w, person, y = _boot_mi_dataset(rng)
        n = len(y)
        rec = RC.Recalls.of(w, person, n)
        data = pd.DataFrame({"age": age, "z": z, "wbar": rec.means()[:, 0], "y": y})
        copies = chained_equations(data, impute=["z"], m=10, iterations=1, seed=r,
                                   kinds=kinds).frames
        est, var = calibrated(copies, rec, y, with_variance=True)
        estimates.append(np.mean(est))
        pooled = pool_scalar(est, var)
        rubin += pooled.ci_low <= 0.5 <= pooled.ci_high

        def replicate(draw: RC.Draw, b: int) -> np.ndarray:
            idx = draw.rows
            again = chained_equations(data.iloc[idx].reset_index(drop=True), impute=["z"],
                                      m=RC.BOOT_COPIES, iterations=1, seed=7919 * r + b,
                                      kinds=kinds).frames
            return np.array([np.mean(calibrated(again, rec.take(idx), y[idx])[0])])

        boot = RC.whole_chain(replicate, RC.Resampling("persons", n), 100, seed=r)
        low, high = boot.interval()
        covered += low[0] <= 0.5 <= high[0]
    print(f"\nmean {np.mean(estimates):.3f} | Boot MI coverage {covered / reps:.3f} | Rubin-only "
          f"{rubin / reps:.3f}")
    assert abs(np.mean(estimates) - 0.5) < 0.05
    assert covered / reps >= 0.93
    assert rubin / reps < 0.90


def test_4_model_based_and_rubin_only_intervals_are_refused_with_exits(complete):
    """Keogh, Shaw & Gustafson 2020 §6.1.2: "one cannot use the usual model standard errors from the
    outcome regression model when performing RC, as these will be too small". Asking for a
    model-based or a Rubin-only interval for a calibrated coefficient is refused with the reason
    and two exits, each a decision that the validators accept: the whole-chain bootstrap, or no
    correction."""
    frame, state = complete["frame"], complete["state"]
    out = {"ingest": complete["run"].info}
    for kind in ("model_based", "rubin_only"):
        with pytest.raises(Refusal) as refused:
            validate(SetMeasurementError(method="regression_calibration", interval=kind),
                     ctx_of(state, frame, out))
        assert refused.value.code == "calibrated_interval"
        assert refused.value.message == RC.INTERVAL_REFUSALS[kind]
        labels = [e["label"] for e in refused.value.exits]
        assert labels == ["Take the interval from a bootstrap over the whole chain",
                          "Record no calibration"]
        for e in refused.value.exits:
            validate(e["decision"], ctx_of(state, frame, out))
    assert "too small" in RC.INTERVAL_REFUSALS["model_based"]
    assert "Boot MI" in RC.INTERVAL_REFUSALS["rubin_only"]


# ── (4)–(6): chain 2 end to end ──────────────────────────────────────────────


@pytest.fixture(scope="module")
def chain2(tmp_path_factory) -> dict:
    """MODELING_SEQUENCE §6 chain 2: repeated 24-hour recalls, multivariate calibration of every
    energy source, the all-components substitution, multiple imputation of BMI, and the surveyed
    population (a bootstrap by PSU within strata that repeats calibration, imputation and the
    outcome model). The answers are recorded through the validators; the graph runs once with a spy
    on the calibration (``methods.calibration.correct``) that keeps what each call was given, then
    again after the adjustment set changes, and again with the calibration declared anew."""
    folder = tmp_path_factory.mktemp("ms5_chain2")
    frame, truth = chain2_table(seed=2, per_psu=20)
    path = folder / "chain.csv"
    frame.to_csv(path, index=False)
    run = GraphRun(path, folder / "project")
    state = chain_state(missing={"strategy": "multiple_imputation", "m": 20}, survey=POPULATION)
    out = run.run(state, upto=["findings"])
    state = declared(state, frame, out, n_boot=50)
    calls: list[dict] = []
    real = RC.correct

    def spy(X, columns, rec, y, *, fit=RC.ols_fit, weights=None):
        calls.append({"X": np.array(X, copy=True), "columns": list(columns), "y": np.array(y),
                      "weights": None if weights is None else np.array(weights)})
        return real(X, columns, rec, y, fit=fit, weights=weights)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(RC, "correct", spy)
        out = run.public(run.run(state, upto=["calibration", "fit"]))
    # The adjustment set changes after the calibration was declared: BMI leaves the model.
    moved = ProjectState.model_validate({**state.model_dump(),
                                         "roles": {**CHAIN_ROLES, "bmi": "excluded"}})
    stale = run.public(run.run(moved, upto=["calibration"]))
    again = declared(moved, frame, {"ingest": run.info}, n_boot=50)
    redone = run.public(run.run(again, upto=["calibration"]))
    yield {"frame": frame, "truth": truth, "state": state, "out": out, "calls": calls,
           "stale": stale, "moved": moved, "again": again, "redone": redone, "run": run}
    run.close()


def test_4_inside_multiple_imputation_each_copy_is_calibrated_with_its_imputed_covariates(chain2):
    """MODELING_SEQUENCE §1.1: for each imputed copy, (1) the imputation, (2) the derived terms,
    (3) the calibration, (4) the fit. The first m calls the stage makes are the point estimate,
    one per copy (m = 20, the rule's minimum here): each copy holds BMI imputed (no blank; the
    blanks' values differ across copies) and BMI is a covariate of that copy's calibration; the
    uncorrected coefficients averaged over the copies are the fit stage's pooled ones (so these are
    the copies the coefficient table pools); and each calibrated coefficient is the mean over the
    copies of NumPy's weighted closed form by hand on that copy (the recall days, sex and age from
    the CSV, BMI from the copy, the survey weights from the CSV), to 1e-8. The weights the engine
    used are the CSV's, up to the one scale factor every weighted fit is invariant to."""
    cal, fit = chain2["out"]["calibration"], chain2["out"]["fit"]
    m = cal["imputations"]
    assert m == 20 and cal["boot_copies"] == RC.BOOT_COPIES == 5
    first = chain2["calls"][:m]
    D, names, people = kcal_days(chain2["frame"])
    blank = people["bmi"].isna().to_numpy()
    assert blank.mean() > 0.05
    columns = cal["covariates"] + cal["calibrated"]
    matrix_names = [c["feature"] for c in fit["models"][0]["coefficients"]
                    if c["feature"] != "(intercept)" and not c["feature"].endswith("_relative")]
    bmi_at = matrix_names.index("bmi")
    bmis = np.stack([c["X"][:, bmi_at] for c in first])
    assert np.isfinite(bmis).all()
    assert np.allclose(bmis[:, ~blank], people["bmi"].to_numpy()[~blank])
    assert np.all(bmis[:, blank].std(axis=0) > 0)  # each blank drawn anew in each copy
    assert "bmi" in cal["covariates"] and set(columns) == set(matrix_names)
    weights = people["wtdr2d"].to_numpy(dtype=float)
    ratio = first[0]["weights"] / weights  # the design's weights, rescaled as the domain holds them
    assert np.allclose(ratio, ratio[0], rtol=1e-12)
    y = people["sbp"].to_numpy(dtype=float)
    sex = (people["sex"] == "M").astype(float).to_numpy()
    by_copy = [by_hand(D, np.column_stack([sex, people["age"], bmi]), y, weights) for bmi in bmis]
    calibrated = np.mean([h["calibrated"] for h in by_copy], axis=0)
    naive = np.mean([h["naive"] for h in by_copy], axis=0)
    table = {c["feature"]: c for c in fit["models"][0]["coefficients"]}
    for e in cal["exposures"]:
        j = names.index(e["feature"])
        assert e["estimate"] == pytest.approx(calibrated[j], rel=1e-8)
        assert e["naive"] == pytest.approx(naive[j], rel=1e-8)
        assert e["naive"] == pytest.approx(table[e["feature"]]["estimate"], rel=1e-9)
        assert e["p"] == pytest.approx(table[e["feature"]]["p"], rel=1e-9)
        assert e["delta_se"] is None  # no closed form over imputations, weights and PSUs
    # The bootstrap repeats the imputation: each of its replicates calibrates BOOT_COPIES copies.
    assert len(chain2["calls"]) == m + cal["n_boot_ok"] * RC.BOOT_COPIES


def test_5_the_calibration_sits_beside_the_uncorrected_estimate_whose_test_is_the_primarys(chain2):
    """Ruling 7 and Freedman et al. 2011 ("reporting estimates adjusted for measurement error along
    with the usual relative risk estimates"; the usual test "remains theoretically valid"): the
    artifact is a declared secondary analysis (``role``), labeled with what it corrects and what
    it assumes, word for word; each row carries the uncorrected estimate with the primary's own
    interval and test (the fit stage's pooled, design-based ones) beside the calibrated estimate
    and its bootstrap interval, and the calibrated estimate carries no test of its own. Declared
    before the estimates: the answer is a slot of the analysis-plan lock, and the artifact shows
    estimates (so serving it locks the plan)."""
    from turbotab.core.plan_lock import plan_slots, shows_estimates

    cal, fit = chain2["out"]["calibration"], chain2["out"]["fit"]
    assert cal["role"] == "secondary" and cal["label"] == LABEL
    assert cal["test"].startswith("The test of no association is the uncorrected model's")
    table = {c["feature"]: c for c in fit["models"][0]["coefficients"]}
    for e in cal["exposures"]:
        primary = table[e["feature"]]
        assert (e["naive_ci_low"], e["naive_ci_high"]) == pytest.approx(
            (primary["ci_low"], primary["ci_high"]), rel=1e-9)
        assert e["ci_low"] < e["estimate"] < e["ci_high"]
        assert set(e) >= {"p", "naive", "estimate"} and "estimate_p" not in e
    assert LABEL in cal["methods"]
    assert "measurement_error" in plan_slots() and shows_estimates("calibration", cal)


def test_5_a_change_to_the_adjustment_set_invalidates_the_calibration(chain2):
    """MODELING_SEQUENCE §2: "a change to the adjustment set invalidates it". The answer records the
    set it was declared under (the completion fills it from the state, never from the client:
    every settled predictor but the exposures); BMI then leaves the model, and the stage re-asks
    instead of calibrating under a set it was not declared for, naming both sets, with exits that
    are decisions: declare it again (which runs, its equation now without BMI) or record none."""
    state = chain2["state"]
    assert state.measurement_error.adjustment == ["age", "bmi", "energy_kcal", "sex"]
    stale = chain2["stale"]["calibration"]
    assert stale["applies"] is False
    assert stale["reason"] == (
        "Regression calibration was declared when the model adjusted for `age`, `bmi`, "
        "`energy_kcal` and `sex`; it now adjusts for `age`, `energy_kcal` and `sex`. Its "
        "calibration equation must hold every covariate of the outcome model, so the declaration "
        "is re-asked, not kept (MODELING_SEQUENCE §2).")
    assert [e["label"] for e in stale["exits"]] == [
        "Declare the calibration again under the current adjustment set", "Record no calibration"]
    assert stale["exits"][0]["decision"] == {"kind": "set_measurement_error",
                                             "method": "regression_calibration", "exposures": [],
                                             "n_boot": 50}
    assert chain2["again"].measurement_error.adjustment == ["age", "energy_kcal", "sex"]
    redone = chain2["redone"]["calibration"]
    assert redone["applies"] and redone["covariates"] == ["sex_M", "age"]


def test_6_chain_2_runs_end_to_end_with_its_relations_and_methods_sentence(chain2):
    """MODELING_SEQUENCE §6 chain 2, its order, its relations and its methods sentence:

    * the run order (§1.1): the contracts' (multiple imputation, then the calibration, in-fold
      steps; the survey's model after them) and the artifact's record of the steps as run;
    * every §2 relation of the calibration fires in this chain (``contracts.fired``), and each
      shows: every outcome-model covariate in the calibration (the matrix's other columns), all
      five sources calibrated jointly (multivariate), the all-components model calibrated, inside
      each copy (Boot MI's copies), PSUs resampled within strata with the fits weighted, the
      secondary beside the uncorrected estimate; the adjustment-set and interval relations are
      the two tests above;
    * the substitution is the difference of calibrated coefficients: 100 × (β_carbohydrate −
      β_fat) exactly, its interval the bootstrap's;
    * the methods text, word for word: its first sentence the reviewers'
      (``audit/modeling-sequence-review.json``, chain 2), the rest naming the sources, the recall
      days, the weights and Rao–Wu, Boot MI with its counts, and the label; and the record's
      sentence for the answer under the surveyed population."""
    from turbotab.core import voice
    from turbotab.core.contracts import contract, fired, run_order

    cal = chain2["out"]["calibration"]
    assert run_order({"multiple_imputation_compatible": "compatible",
                      "regression_calibration": "multivariate",
                      "survey_population": "population"}) == [
        "multiple_imputation_compatible", "regression_calibration", "survey_population"]
    assert cal["order"] == [
        "multiple imputation compatible with the analysis model (m = 20)",
        "the energy model refit on each copy, then applied to each recall day",
        "regression calibration in each copy", "the outcome model in each copy",
        "the whole-chain bootstrap (Boot MI)"]
    rc = contract("regression_calibration")
    consequences = [r.target for r in rc.relations] + [
        "regression calibration by PSU within strata"]
    firing = fired({"multiple_imputation_compatible": "compatible",
                    "regression_calibration": "multivariate", "survey_population": "population"},
                   "inference", consequences)
    ours = {f.relation.name for f in firing if f.source == "regression_calibration"}
    assert ours == {r.name for r in rc.relations}
    assert any(f.source == "survey_population" and f.relation.name == "regression_calibration"
               for f in firing)
    D, names, _ = kcal_days(chain2["frame"])
    assert cal["calibration"] == "multivariate" and cal["calibrated"] == names
    assert cal["covariates"] == ["sex_M", "age", "bmi"]
    assert (cal["resampling"], cal["weighted"]) == ("psu_within_strata", True)
    assert cal["n_boot"] == cal["n_boot_ok"] == 50
    rows = {e["feature"]: e for e in cal["exposures"]}
    (swap,) = cal["contrasts"]
    assert (swap["donor"], swap["recipient"], swap["step_kcal"]) == ("fat_g", "carbohydrate_g", 100)
    assert swap["estimate"] == pytest.approx(
        100 * (rows["kcal_from_carbohydrate_g"]["estimate"] - rows["kcal_from_fat_g"]["estimate"]),
        rel=1e-12)
    assert swap["naive"] == pytest.approx(
        100 * (rows["kcal_from_carbohydrate_g"]["naive"] - rows["kcal_from_fat_g"]["naive"]),
        rel=1e-12)
    assert swap["ci_low"] < swap["estimate"] < swap["ci_high"]
    assert cal["methods"].startswith(CHAIN_2_REVIEWERS)
    assert cal["methods"] == (
        f"{CHAIN_2_REVIEWERS} Calibration (Rosner, Spiegelman & Willett 1990, Am J Epidemiol "
        "132:734; Carroll, Ruppert, Stefanski & Crainiceanu 2006, Measurement Error in Nonlinear "
        "Models, §4.4) used the within-person covariance of the recalls of the 320 participants "
        "with two or more (2 recalls each). The calibration and the outcome model were "
        "survey-weighted, PSUs resampled by Rao and Wu's bootstrap (Rao & Wu 1988, J Am Stat Assoc "
        "83:231). Each of 50 bootstrap resamples was imputed 5 times and the interval is the "
        "percentile interval of their means (Boot MI, Schomaker & Heumann 2018, Stat Med 37:2252); "
        "the estimate is the mean over the 20 imputed copies. It is a declared secondary analysis "
        "beside the uncorrected estimate, whose test of no association is the primary's; it "
        f"{LABEL}.")
    decision = SetMeasurementError(**chain2["state"].measurement_error.model_dump())
    assert voice.sentence_for(decision, chain2["state"]) == (
        "Regression calibration of every intake the recalls measure from the repeated recalls was "
        "declared as a secondary analysis beside the uncorrected estimate, with intervals from `50` "
        "bootstrap resamples of the whole chain, resampling PSUs within strata with the fits "
        "survey-weighted.")


def test_6_the_contract_is_in_the_one_registry_with_live_code_behind_each_relation():
    """BLUEPRINT §13: the method enters through the one registry (``turbotab.core.contracts``) with
    slot, scope, needs, routing (each option labeled customary and sound for both purposes, with a
    rung), storyboard, sentence and relations; every relation names code that exists; the one-at-
    a-time calibration is refused under inference, with its reason, and everything is refused
    under prediction."""
    import importlib

    from turbotab.core.contracts import contract, contracts

    assert "regression_calibration" in contracts()
    c = contract("regression_calibration")
    assert (c.slot, c.scope, c.package, c.decision, c.stage) == (
        "in_fold", "training_fold", "RC", "set_measurement_error", "calibration")
    for r in c.relations:
        module, name = r.enforced_by.split(":")
        assert hasattr(importlib.import_module(module), name), r.enforced_by
    options = {o["key"]: o for o in c.options_for("inference")}
    assert options["one_at_a_time"]["rung"] == "refused"
    assert "either way" in options["one_at_a_time"]["sound"]
    assert options["multivariate"]["rung"] == "recommended"
    assert all(o["rung"] == "refused" for o in c.options_for("prediction"))
    module, name = str(c.sentence).split(":")
    assert hasattr(importlib.import_module(module), name)
