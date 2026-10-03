"""WP3 · Energy-transform algebra (docs/turbotab-next/audit/AUDIT_REPORT.md §5).

Closes MA-02 (the within-sex residual manufactures an exposure effect) and B20 (a stratum got its
own residual regression from as few as 3 rows). One test per acceptance test in §5, numbered as
there. Every reference is computed by a path independent of the code under test: numpy's
``polyfit``, scipy's ``pearsonr``, statsmodels' OLS, or the analytic null.

The null fixture is the audit skeptic's (``repro.tar.gz``: ``B-skeptic/s1_strata.py``): n = 3,000;
energy and fat share by sex; fat has no effect at fixed energy and energy none at all; the outcome
depends on sex and age.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats

from turbotab.core.decisions import EnergyAdjustment, ProjectState, SplitSpec
from turbotab.core.methods.energy import MIN_LEVEL_ROWS
from turbotab.core.models import get_family
from turbotab.core.models.linear import model_matrix
from turbotab.core.models.pipeline import build_pipeline, design_spec, input_columns, model_predictors
from turbotab.core.models.steps import StratifiedEnergyAdjuster, energy_step
from turbotab.core.stages.modeling import design_stage
from turbotab.core.tests import modeling_fixtures as mf

REPO = Path(__file__).resolve().parents[4]
PACK = REPO / "docs" / "turbotab" / "research" / "NUTRITION_PACK.md"

ROLES = {"fat_g": "exposure", "energy_kcal": "energy", "age": "covariate", "sex": "excluded"}
STRATIFIED = EnergyAdjustment(method="residual", energy_column="energy_kcal", nutrients=["fat_g"],
                              strata="sex")


def _state(**slots) -> ProjectState:
    base = dict(lens=["dietary"], target="y", task="regression", purpose="inference",
                roles=dict(ROLES), missing="complete_case",
                split=SplitSpec(holdout=0.0, seed=0, folds=5), models=["linear"],
                energy_adjustment=STRATIFIED)
    base.update(slots)
    return ProjectState(**base)


def _null_fixture(seed: int, n: int = 3_000) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """The skeptic's s1_strata fixture: fat (at fixed energy) and energy have no effect."""
    rng = np.random.default_rng(seed)
    male = rng.integers(0, 2, n)
    energy = rng.normal(1800 + 450 * male, 350, n).clip(600)
    fat_g = rng.normal(0.33, 0.05, n).clip(0.1, 0.6) * energy / 9
    age = rng.uniform(20, 80, n)
    y = 5.0 * male + 0.05 * age + rng.normal(0, 3, n)
    frame = pd.DataFrame({"fat_g": fat_g, "energy_kcal": energy, "age": age,
                          "sex": np.where(male == 1, "M", "F")})
    return frame, y, male


def _ols_p(y: np.ndarray, exog: pd.DataFrame, name: str) -> tuple[float, float]:
    """(estimate, p) for one column of a statsmodels OLS with an intercept."""
    fit = sm.OLS(y, sm.add_constant(exog)).fit()
    return float(fit.params[name]), float(fit.pvalues[name])


def _hc3_p(y: np.ndarray, exog: pd.DataFrame, name: str) -> tuple[float, float]:
    """(estimate, p) for one column of OLS with an intercept, p from HC3 standard errors on
    t(n − p), written out from the definition (MacKinnon & White 1985) in numpy and scipy, not
    statsmodels: the app's inference engine (WP2) is the one that calls statsmodels' HC3.

    V = (X'X)⁻¹ X' diag(e_i² / (1 − h_ii)²) X (X'X)⁻¹, h_ii the leverages."""
    X = np.column_stack([np.ones(len(y)), exog.to_numpy(dtype=float)])
    j = 1 + list(exog.columns).index(name)
    bread = np.linalg.inv(X.T @ X)
    beta = bread @ X.T @ y
    e = y - X @ beta
    h = np.einsum("ij,jk,ik->i", X, bread, X)
    meat = (X * (e / (1.0 - h))[:, None] ** 2).T @ X
    se = float(np.sqrt((bread @ meat @ bread)[j, j]))
    dof = len(y) - X.shape[1]
    return float(beta[j]), float(2 * stats.t.sf(abs(beta[j]) / se, dof))


def _design(frame: pd.DataFrame, tmp_path: Path, **slots):
    """The real design stage on ``frame`` with every row a fitting row (no holdout)."""
    frame = frame.assign(y=np.arange(len(frame), dtype=float))
    paths = mf.ingest_frame(frame, tmp_path)
    split = mf.split_bundle(np.arange(len(frame)), holdout=0.0)
    ctx = mf.context(_state(**slots), {"split": split, "target_info": mf.target_info("regression", "y")},
                     paths)
    return design_stage(ctx)


def _formula(design, column: str) -> str:
    node = next(n for n in design.data["lineage"]["nodes"] if n["id"] == f"adj:{column}")
    return node["formula"]


# ── 1 · the null fixture ─────────────────────────────────────────────────────

REPLICATES = 1_000  # the spec's 200 are the first 200 of these
SPEC_REPLICATES = 200


def test_1_the_within_sex_residual_finds_no_effect_on_the_null_fixture():
    """§5 WP3.1: "On the null fixture (n = 3,000; fat has no effect; outcome depends on sex), the
    within-sex residual with sex not among the predictors gives p > 0.05 in ≥ 95% of 200
    replicates and pooled |r(fat_adj, sex)| < 0.05. Reference today: p ≈ 10⁻¹⁰⁶ to 10⁻²⁰⁹,
    r = 0.56–0.58."

    Measured through the app's own path: the state's roles give the predictors (sex has a
    non-predictor role, so it is a strata input only), ``design_spec`` and ``build_pipeline``
    build the linear pipeline, and the inference coefficient table (``Linear.coefficients``: OLS
    with HC3 standard errors on t(n − p) since WP2) gives fat_g_adj's p. r is numpy's, on the
    model matrix the fit saw.

    Reference: the analytic null. A valid test rejects at most 5% of the time; here the omitted
    sex effect sits in the residual variance, so the test is conservative and the expected share
    of p > 0.05 is about 0.99. 1,000 replicates are run; the bound is asserted on the spec's first
    200 and on all 1,000 (Monte Carlo SE of the share ≈ 0.004 at 0.985). Unbiasedness: the mean
    estimate lies within 3 Monte Carlo SEs of the true 0.

    Positive control: on replicate 0, the old algebra (each sex's own constant, computed here with
    numpy) through the same OLS gives p < 10⁻⁵⁰ and |r(fat_adj, sex)| > 0.5, so the fixture and
    the bounds discriminate.
    """
    state = _state()
    family = get_family("linear")
    predictors = model_predictors(state)
    assert "sex" not in predictors
    inputs = input_columns(predictors, STRATIFIED)
    assert "sex" in inputs

    ps, estimates, rs = [], [], []
    for seed in range(REPLICATES):
        frame, y, male = _null_fixture(seed)
        X = frame[inputs]
        spec = design_spec(state, X, predictors)
        fitted = build_pipeline(spec, family, "regression", "inference", len(X), len(predictors)).fit(X, y)
        matrix = model_matrix(fitted, X)
        assert "sex" not in matrix.columns
        rows = family.coefficients(fitted, X, y, task="regression", purpose="inference")
        row = next(r for r in rows if r["feature"] == "fat_g_adj")
        ps.append(row["p"])
        estimates.append(row["estimate"])
        rs.append(np.corrcoef(matrix["fat_g_adj"].to_numpy(), male)[0, 1])

        if seed == 0:
            # The app's fat_g_adj is each sex's residual plus one constant, the cohort mean fat_g.
            fat, energy = frame["fat_g"].to_numpy(), frame["energy_kcal"].to_numpy()
            ours, old = np.empty_like(fat), np.empty_like(fat)
            for level in (0, 1):
                m = male == level
                slope, intercept = np.polyfit(energy[m], fat[m], 1)
                residual = fat[m] - (intercept + slope * energy[m])
                ours[m] = residual + fat.mean()
                old[m] = residual + fat[m].mean()  # each sex's own constant: the defect
            np.testing.assert_allclose(matrix["fat_g_adj"].to_numpy(), ours, rtol=0, atol=1e-9)
            _, p_old = _ols_p(y, pd.DataFrame({"fat_old": old, "age": frame["age"]}), "fat_old")
            assert p_old < 1e-50
            assert abs(np.corrcoef(old, male)[0, 1]) > 0.5

    ps, estimates, rs = np.asarray(ps), np.asarray(estimates), np.abs(np.asarray(rs))
    assert np.mean(ps[:SPEC_REPLICATES] > 0.05) >= 0.95, np.mean(ps[:SPEC_REPLICATES] > 0.05)
    assert np.mean(ps > 0.05) >= 0.95, np.mean(ps > 0.05)
    assert rs[:SPEC_REPLICATES].max() < 0.05 and rs.max() < 0.05, rs.max()
    mc_se = estimates.std(ddof=1) / np.sqrt(len(estimates))
    assert abs(estimates.mean()) < 3 * mc_se, (estimates.mean(), mc_se)


@pytest.mark.parametrize("method", ["residual_energy_dropped", "residual"])
def test_1_the_null_holds_with_sex_in_the_model_too(method):
    """The fix must not depend on the strata column being out of the model: with sex a
    predictor (a covariate), the same fixture gives the same fat_g_adj coefficient as the
    reference OLS ``y ~ fat_within_sex_residual + age + sex`` from statsmodels, and it is null.

    Its p is the reported one: independent rows under inference get HC3 standard errors on
    t(n − p) since WP2 (MA-01), so the reference p is HC3's, from the definition (``_hc3_p``). The
    classical p is null too.

    Both forms of the residual method (WP6, BLUEPRINT §12 ruling 1): this test was written when the
    residual step dropped total energy, and that form (``residual_energy_dropped``) is held to the
    original reference unchanged. The default form keeps total energy in the outcome model, so its
    reference adds the energy term: ``y ~ residual + energy + age + sex``."""
    frame, y, male = _null_fixture(11)
    adjustment = STRATIFIED.model_copy(update={"method": method})
    state = _state(roles={**ROLES, "sex": "covariate"}, energy_adjustment=adjustment)
    family = get_family("linear")
    predictors = model_predictors(state)
    assert "sex" in predictors
    X = frame[input_columns(predictors, adjustment)]
    spec = design_spec(state, X, predictors)
    fitted = build_pipeline(spec, family, "regression", "inference", len(X), len(predictors)).fit(X, y)
    rows = family.coefficients(fitted, X, y, task="regression", purpose="inference")
    ours = next(r for r in rows if r["feature"] == "fat_g_adj")
    assert ("energy_kcal" in model_matrix(fitted, X).columns) == (method == "residual")

    fat, energy = frame["fat_g"].to_numpy(), frame["energy_kcal"].to_numpy()
    residual = np.empty_like(fat)
    for level in (0, 1):
        m = male == level
        slope, intercept = np.polyfit(energy[m], fat[m], 1)
        residual[m] = fat[m] - (intercept + slope * energy[m])
    exog = pd.DataFrame({"r": residual, "age": frame["age"], "male": male})
    if method == "residual":
        exog["energy"] = energy
    reference, p_classical = _ols_p(y, exog, "r")
    hc3_estimate, p = _hc3_p(y, exog, "r")
    assert hc3_estimate == pytest.approx(reference, rel=1e-9, abs=1e-12)
    assert ours["estimate"] == pytest.approx(reference, rel=1e-9, abs=1e-12)
    assert ours["p"] == pytest.approx(p, rel=1e-6)
    assert ours["p"] > 0.05 and p_classical > 0.05


# ── 2 · per-level and pooled checks, reported in the lineage ─────────────────


def _three_levels(seed: int = 1) -> pd.DataFrame:
    """F and M of 500 rows, X of 12 (on the pooled slope, with a within-level slope of its own
    that differs, so its r is not zero), and two rows whose sex is blank."""
    rng = np.random.default_rng(seed)
    sex = np.array(["F"] * 500 + ["M"] * 500 + ["X"] * 12, dtype=object)
    energy = rng.normal(1800, 350, sex.size) + 450 * (sex == "M")
    fat = 0.33 * energy / 9 + rng.normal(0, 8, sex.size) + 5 * (sex == "X")
    frame = pd.DataFrame({"fat_g": fat, "energy_kcal": energy, "age": rng.uniform(20, 80, sex.size),
                          "sex": sex})
    frame.loc[[3, 7], "sex"] = np.nan
    return frame


def _independent_checks(adj: np.ndarray, energy: np.ndarray, level: pd.Series) -> dict:
    """r(N_adj, E) pooled and per level (scipy), r with each level's indicator (scipy), and the
    correlation ratio with the strata column as sqrt(R²) of ``adj ~ C(level)`` (statsmodels)."""
    labels = level.fillna("(blank)").astype(str).to_numpy()
    ok = np.isfinite(adj) & np.isfinite(energy)
    adj, energy, labels = adj[ok], energy[ok], labels[ok]
    dummies = pd.get_dummies(labels, drop_first=True, dtype=float)
    eta = float(np.sqrt(max(sm.OLS(adj, sm.add_constant(dummies)).fit().rsquared, 0.0)))
    per = {}
    for name in sorted(set(labels)):
        inside = labels == name
        within = (stats.pearsonr(adj[inside], energy[inside]).statistic
                  if inside.sum() >= 3 else None)
        per[name] = {"rows": int(inside.sum()), "r_adj_energy": within,
                     "r_adj_indicator": stats.pearsonr(adj, inside.astype(float)).statistic}
    return {"pooled": {"rows": int(ok.sum()), "r_adj_energy": stats.pearsonr(adj, energy).statistic,
                       "r_adj_strata": eta}, "levels": per}


@pytest.mark.parametrize("log", [False, True], ids=["linear", "log"])
def test_2_per_level_and_pooled_correlations_are_reported_in_the_lineage(log, tmp_path):
    """§5 WP3.2: "Per-level and pooled r(N_adj, E) and r(N_adj, strata) are both reported in the
    lineage."

    Per level, r(N_adj, E) is within the level, and r(N_adj, strata) is r with the level's
    indicator over every row (within one level the strata column is constant and has no r);
    pooled, r(N_adj, E) is over every fitting row and r(N_adj, strata) is the correlation ratio
    η. Each reported number (``params["check"]``) must equal scipy's or statsmodels' on the
    adjusted values as the model sees them. The fixture makes several of them nonzero (level X
    on the pooled slope; the log variant back-transformed), so a reporter that printed zeros
    would fail. The formula the design stage puts on the lineage node carries the pooled and
    per-level values.
    """
    frame = _three_levels()
    adjustment = STRATIFIED.model_copy(update={"log_transform": log})
    predictors = ["fat_g", "energy_kcal", "age"]
    inputs = input_columns(predictors, adjustment)
    step = energy_step(adjustment, predictors).fit(frame[inputs])
    out = step.transform(frame[inputs])
    entry = next(e for e in step.lineage() if e["output"] == "fat_g_adj")
    reported = entry["params"]["check"]
    expected = _independent_checks(out["fat_g_adj"].to_numpy(), frame["energy_kcal"].to_numpy(),
                                   frame["sex"])

    assert reported["pooled"]["rows"] == expected["pooled"]["rows"] == len(frame)
    for key in ("r_adj_energy", "r_adj_strata"):
        assert reported["pooled"][key] == pytest.approx(expected["pooled"][key], abs=1e-9)
    assert set(reported["levels"]) == set(expected["levels"]) == {"F", "M", "X", "(blank)"}
    for level, mine in reported["levels"].items():
        theirs = expected["levels"][level]
        assert mine["rows"] == theirs["rows"]
        if theirs["r_adj_energy"] is None:
            assert mine["r_adj_energy"] is None  # 2 rows: undefined, and said so
        else:
            assert mine["r_adj_energy"] == pytest.approx(theirs["r_adj_energy"], abs=1e-9)
        assert mine["r_adj_indicator"] == pytest.approx(theirs["r_adj_indicator"], abs=1e-9)
    # Not a reporter of zeros: these are nonzero on this fixture.
    assert abs(reported["levels"]["X"]["r_adj_energy"]) > 0.1
    if log:
        assert abs(reported["pooled"]["r_adj_energy"]) > 0.01
        assert reported["pooled"]["r_adj_strata"] > 0.005

    def r3(v: float) -> str:
        return "0.000" if abs(v) < 0.0005 else f"{v:.3f}".replace("-", "−")

    design = _design(frame, tmp_path, energy_adjustment=adjustment)
    formula = _formula(design, "fat_g_adj")
    pooled = reported["pooled"]
    assert (f"r(fat_g_adj, energy_kcal) is {r3(pooled['r_adj_energy'])} pooled, "
            f"{r3(reported['levels']['F']['r_adj_energy'])} within F, "
            f"{r3(reported['levels']['M']['r_adj_energy'])} within M, "
            f"{r3(reported['levels']['X']['r_adj_energy'])} within X, undefined within (blank)") in formula
    assert (f"r(fat_g_adj, sex) is {r3(pooled['r_adj_strata'])} pooled (correlation ratio)") in formula


# ── 3 · the pack agrees with itself, and the code with the pack ──────────────


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace(">", " "))


def test_3_the_pack_adds_the_predicted_intake_at_the_cohort_mean_energy():
    """§5 WP3.3: "NUTRITION_PACK line 480 is corrected to agree with line 446 ('at the cohort
    mean energy')."

    Source check, NUTRITION_PACK.md §04. The coaching line (446) reads: "Default recommendation:
    the Willett residual method, computed within the final analytic sample and within sex, on
    log-transformed nutrient and energy, with the predicted nutrient at the cohort mean energy
    added back." The methods-sentence generator (line 480) read "adding the predicted intake at
    the sex-specific mean energy (men: 2,180 kcal/d; women: 1,720 kcal/d)"; it now reads
    "adding the predicted intake at the cohort mean energy (1,950 kcal/d) to the residuals."

    And the code adds what the pack says: the stratified step's constant equals the predicted
    nutrient at the mean energy of all fitting rows, from a statsmodels OLS of N on E (of
    log N on log E for the log variant), the same for every level.
    """
    pack = PACK.read_text(encoding="utf-8")
    section = pack[pack.index("## 04"):pack.index("## 05")]
    flat = _normalized(section)
    assert "with the predicted nutrient at the cohort mean energy added back" in flat
    generator = re.search(r"\*\*Methods sentence generator:\*\* \*\"(.+?)\"\*", flat)
    assert generator is not None
    sentence = generator.group(1)
    assert "within sex" in sentence
    assert "adding the predicted intake at the cohort mean energy" in sentence
    assert "sex-specific" not in sentence

    frame, _, _ = _null_fixture(5)
    fat, energy = frame["fat_g"].to_numpy(), frame["energy_kcal"].to_numpy()
    for log in (False, True):
        step = StratifiedEnergyAdjuster("energy_kcal", ["fat_g"], strata="sex",
                                        log_transform=log).fit(frame)
        n_, e_ = (np.log(fat), np.log(energy)) if log else (fat, energy)
        ols = sm.OLS(n_, sm.add_constant(e_)).fit()
        at_mean = float(ols.predict(np.array([[1.0, e_.mean()]]))[0])
        params = next(e for e in step.lineage() if e["output"] == "fat_g_adj")["params"]
        assert params["constant"] == pytest.approx(at_mean, rel=1e-12)
        # One constant for every level: each level's adjusted mean (on the regression's scale)
        # is that constant, so no level differs from another by construction.
        adjusted = step.transform(frame)["fat_g_adj"].to_numpy()
        scaled = np.log(adjusted) if log else adjusted
        for level in ("F", "M"):
            assert scaled[frame["sex"].to_numpy() == level].mean() == pytest.approx(at_mean, rel=1e-12)


# ── 4 · a small stratum falls back to the pooled slope, and says so ──────────


def _small_levels(seed: int = 4) -> pd.DataFrame:
    """F and M of 400 rows; X of 29 rows (one under the floor) and Y of 30 (at it)."""
    rng = np.random.default_rng(seed)
    sex = np.array(["F"] * 400 + ["M"] * 400 + ["X"] * 29 + ["Y"] * 30, dtype=object)
    energy = rng.normal(1800, 350, sex.size) + 450 * (sex == "M")
    slope = np.where(sex == "X", 0.02, np.where(sex == "Y", 0.05, 0.037))
    fat = slope * energy + rng.normal(0, 8, sex.size) + 6 * (sex == "Y")
    return pd.DataFrame({"fat_g": fat, "energy_kcal": energy, "age": rng.uniform(20, 80, sex.size),
                         "sex": sex})


def test_4_a_stratum_under_the_floor_uses_the_pooled_slope_and_the_lineage_says_so(tmp_path):
    """§5 WP3.4: "A stratum with fewer than a stated minimum of rows (proposed 30) falls back to
    the pooled slope, and the lineage says so." (B20: a stratum got its own regression from as
    few as 3 rows.)

    The floor is 30 fitting rows. Level X (29 rows) uses the pooled slope, from numpy's
    ``polyfit`` over every row, centered on X's own means; level Y (30 rows) gets its own slope,
    from ``polyfit`` on Y. A level never seen in fit uses the pooled regression. The design
    stage's lineage node and its warnings both name X and the floor.
    """
    assert MIN_LEVEL_ROWS == 30
    frame = _small_levels()
    fat, energy, sex = frame["fat_g"].to_numpy(), frame["energy_kcal"].to_numpy(), frame["sex"].to_numpy()
    step = StratifiedEnergyAdjuster("energy_kcal", ["fat_g"], strata="sex", drop_strata=True).fit(frame)
    adjusted = step.transform(frame)["fat_g_adj"].to_numpy()
    pooled_slope, pooled_intercept = np.polyfit(energy, fat, 1)
    cohort = fat.mean()

    x = sex == "X"
    expected_x = fat[x] - pooled_slope * (energy[x] - energy[x].mean()) - fat[x].mean() + cohort
    np.testing.assert_allclose(adjusted[x], expected_x, rtol=0, atol=1e-9)
    for level in ("F", "M", "Y"):
        m = sex == level
        own_slope, _ = np.polyfit(energy[m], fat[m], 1)
        expected = fat[m] - own_slope * (energy[m] - energy[m].mean()) - fat[m].mean() + cohort
        np.testing.assert_allclose(adjusted[m], expected, rtol=0, atol=1e-9)

    unseen = frame.iloc[:3].assign(sex="Z")
    np.testing.assert_allclose(step.transform(unseen)["fat_g_adj"].to_numpy(),
                               unseen["fat_g"] - pooled_slope * (unseen["energy_kcal"] - energy.mean()),
                               rtol=0, atol=1e-9)

    params = next(e for e in step.lineage() if e["output"] == "fat_g_adj")["params"]
    assert params["min_level_rows"] == 30
    assert params["pooled_slope_levels"] == ["X"]
    assert params["levels"]["X"]["slope_from"] == "pooled" and params["levels"]["X"]["n_fit"] == 29
    assert params["levels"]["Y"]["slope_from"] == "level" and params["levels"]["Y"]["n_fit"] == 30

    design = _design(frame, tmp_path)
    formula = _formula(design, "fat_g_adj")
    b = np.format_float_positional(pooled_slope, precision=4, unique=False, fractional=False, trim="-")
    assert f"X the pooled slope b = {b} on its own means (29 rows; fewer than 30 fitting rows)" in formula
    assert "Y b = " in formula and "(30 rows)" in formula
    # Under inference the design's rows are every analyzed row (BLUEPRINT §12 ruling 3; the
    # methods gate), and the note names them so; with no holdout they are the same 29.
    assert any(w.startswith("Level X of sex has 29 analyzed rows, fewer than the 30") and "pooled slope" in w
               for w in design.data["warnings"]), design.data["warnings"]
    assert not any("Level Y" in w for w in design.data["warnings"])
