"""The swarm's graded fixtures (SWARM_PLAN §5 items 4 to 6, §7 "Decision quality, anchored to known
truth"): three generators, each a function of a seed, and the truths each plants, computed from the
generator itself and never from a draw a run is graded on.

Every generator takes ``seed`` and returns one table. ``numpy.random.SeedSequence(seed).spawn(k)``
gives independent streams for the parts of a draw that must not depend on each other (the
outcome's noise is never the columns' stream), so a fresh draw is the same generator at another
seed (``core/export/replay.py``'s fresh-draw flag). A generator's parameters (coefficients, the
true feature set, each feature's batch offsets) are fixed: only the rows are drawn again.

* :func:`readmission` (the hardened Predict fixture, item 4): ``clinical_readmission_predict``'s
  columns and logistic model (``make_fixtures.clinical_risk``), plus a U-shaped sodium term and an
  age-by-albumin interaction, which additive straight lines miss. :func:`readmission_regrets`
  computes, from the generator, the Brier regret of the population-best model of three classes
  against the true risk: additive and linear, additive with splines, and splines with the
  interaction, so a path's choices change its regret. Which class a path's plan falls in, and so
  r0 (SWARM_PLAN §7), is not settled here.
* :func:`blood_pressure` (the Estimate fixture, item 5): a non-null effect of a daily walking habit
  on systolic blood pressure (a difference in means, the measure the engine reports for a numeric
  outcome), a confounder (chronic pain), a collider (a health-concern score caused by both), and
  the confounder missing more often where blood pressure is high (missing at random given the
  outcome), so complete cases are biased and imputation is not. :func:`population_effects` derives
  the population coefficient on the exposure under the oracle plan and each planted wrong path in
  closed form (module section below), so b0 (half the smallest planted bias) is computed from the
  generator.
* :func:`metabolite_panel` (the measurement fixture, item 6): a known set of features associated
  with case status with known log2 fold changes, under detection-limit censoring, run-order drift,
  batch offsets that go up and down in run order, sample dilution, and pooled QC injections, laid
  out as the repository's untargeted metabolomics table is. Its graded decisions
  (:func:`graded_measurement_decisions`) are the engine's own menus, read from the engine.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ── shared ───────────────────────────────────────────────────────────────────


def streams(seed: int, k: int) -> list[np.random.Generator]:
    """``k`` independent random streams of ``seed``."""
    return [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(k)]


def expit(z: Any) -> Any:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


# ── the hardened Predict fixture ─────────────────────────────────────────────

READMISSION_ROWS = 2000
READMISSION_SEED = 20261022  # the committed table's seed
READMISSION_FILE = "readmissions.csv"
READMISSION_OUTCOME = "readmit_30d"
# The outcome is 1 with probability expit(intercept + Σ coef × (f(column) − center) + the U-shape +
# the interaction), every term read from the table's own (rounded) columns. The linear part is
# ``make_fixtures.clinical_risk``'s, term for term.
READMISSION_MODEL: dict[str, Any] = {
    "link": "logit", "intercept": -1.35,
    "terms": [
        {"column": "age", "coef": 0.030, "center": 68, "transform": "identity"},
        {"column": "charlson_index", "coef": 0.26, "center": 2.2, "transform": "identity"},
        {"column": "prior_admissions_12mo", "coef": 0.50, "center": 0.9, "transform": "identity"},
        {"column": "albumin_g_dl", "coef": -0.95, "center": 3.6, "transform": "identity"},
        {"column": "creatinine_mg_dl", "coef": 0.45, "center": 1.05, "transform": "log_ratio"},
        {"column": "hemoglobin_g_dl", "coef": -0.16, "center": 11.9, "transform": "identity"},
        {"column": "length_of_stay_days", "coef": 0.075, "center": 4.6, "transform": "identity"},
    ],
    # coef × (((x − center) / scale)² − 1): risk rises on both sides of a normal sodium, so a
    # straight line through it is nearly flat.
    "u_shape": {"column": "sodium_mmol_l", "coef": 0.5, "center": 138, "scale": 4.0},
    # coef × ((a − ca) / sa) × ((b − cb) / sb): low albumin weighs more in older patients.
    "interaction": {"columns": ["age", "albumin_g_dl"], "coef": -0.4, "centers": [68, 3.6],
                    "scales": [13, 0.45]},
    "no_signal": ["encounter_id", "sex"],
}


def readmission_columns(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """The cohort's columns, drawn as ``make_fixtures.clinical_risk`` draws them."""
    age = np.clip(rng.normal(68, 13, n), 22, 97).round(0)
    sex = rng.choice(["F", "M"], n)
    charlson = rng.poisson(2.2, n).clip(0, 12)
    prior = rng.poisson(0.9, n).clip(0, 8)
    albumin = np.clip(rng.normal(3.6, 0.45, n), 1.6, 5.2).round(1)
    creatinine = np.clip(rng.lognormal(np.log(1.05), 0.42, n), 0.3, 9.0).round(2)
    hemoglobin = np.clip(rng.normal(11.9, 1.7, n), 6.0, 18.0).round(1)
    sodium = np.clip(rng.normal(138, 4.0, n), 118, 155).round(0)
    los = np.clip(rng.gamma(2.2, 2.1, n), 1, 40).round(0)
    return pd.DataFrame({
        "encounter_id": [f"ENC{i:05d}" for i in range(n)],
        "age": age.astype(int), "sex": sex, "charlson_index": charlson.astype(int),
        "prior_admissions_12mo": prior.astype(int), "albumin_g_dl": albumin,
        "creatinine_mg_dl": creatinine, "hemoglobin_g_dl": hemoglobin,
        "sodium_mmol_l": sodium.astype(int), "length_of_stay_days": los.astype(int),
    })


def readmission_log_odds(frame: pd.DataFrame, model: dict[str, Any] = READMISSION_MODEL,
                         *, linear_only: bool = False) -> np.ndarray:
    """The true log-odds of each row of ``frame``, from its own columns (``linear_only``: the
    additive straight-line part alone)."""
    z = np.full(len(frame), float(model["intercept"]))
    for t in model["terms"]:
        x = frame[t["column"]].to_numpy(dtype=float)
        z = z + t["coef"] * (np.log(x / t["center"]) if t["transform"] == "log_ratio"
                             else x - t["center"])
    if linear_only:
        return z
    u = model["u_shape"]
    s = (frame[u["column"]].to_numpy(dtype=float) - u["center"]) / u["scale"]
    z = z + u["coef"] * (s ** 2 - 1.0)
    i = model["interaction"]
    a, b = (frame[c].to_numpy(dtype=float) for c in i["columns"])
    z = z + i["coef"] * ((a - i["centers"][0]) / i["scales"][0]) * ((b - i["centers"][1])
                                                                    / i["scales"][1])
    return z


def readmission_risk(frame: pd.DataFrame) -> np.ndarray:
    """The true risk of each row: the oracle's prediction."""
    return expit(readmission_log_odds(frame))


def readmission(seed: int = READMISSION_SEED, n: int = READMISSION_ROWS) -> pd.DataFrame:
    """The hardened Predict fixture's table at ``seed``: the columns from the first stream, the
    outcome a Bernoulli draw of the true risk from the second."""
    columns, outcome = streams(seed, 2)
    frame = readmission_columns(columns, n)
    frame[READMISSION_OUTCOME] = (outcome.random(n) < readmission_risk(frame)).astype(int)
    return frame


SPLINE_COLUMNS = ["age", "albumin_g_dl", "creatinine_mg_dl", "hemoglobin_g_dl", "sodium_mmol_l",
                  "length_of_stay_days"]
REGRET_ROWS = (300_000, 200_000)  # the fit's rows and the rows its regret is read on
REGRET_SEEDS = (1, 2)


def _design(frame: pd.DataFrame, kind: str) -> Any:
    """The model class's columns: ``linear`` every predictor as it is (sex as 0/1), ``spline`` the
    continuous ones as cubic B-splines (five knots), ``interaction`` those and age × albumin."""
    from sklearn.preprocessing import SplineTransformer

    base = frame.drop(columns=["encounter_id", READMISSION_OUTCOME], errors="ignore").copy()
    base["sex"] = (base["sex"] == "M").astype(float)
    if kind == "linear":
        return base.to_numpy(dtype=float), None
    spline = SplineTransformer(n_knots=5, degree=3).fit(base[SPLINE_COLUMNS])
    return base, spline


def _class_fit(frame: pd.DataFrame, y: np.ndarray, kind: str) -> Any:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, spline = _design(frame, kind)
    if spline is None:
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1e6, max_iter=5000))
        return model.fit(X, y), None
    M = _expanded(X, spline, kind)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1e6, max_iter=5000))
    return model.fit(M, y), spline


def _expanded(X: pd.DataFrame, spline: Any, kind: str) -> np.ndarray:
    rest = X.drop(columns=SPLINE_COLUMNS).to_numpy(dtype=float)
    parts = [spline.transform(X[SPLINE_COLUMNS]), rest]
    if kind == "interaction":
        parts.append((X["age"] * X["albumin_g_dl"]).to_numpy(dtype=float)[:, None])
    return np.column_stack(parts)


def readmission_regrets(rows: tuple[int, int] = REGRET_ROWS,
                        seeds: tuple[int, int] = REGRET_SEEDS) -> dict[str, float]:
    """The Brier regret against the true risk, E[(p̂ − p)²], of the population-best model of each
    class (an unpenalized logistic fit on ``rows[0]`` rows of the generator at ``seeds[0]``, read
    on ``rows[1]`` rows at ``seeds[1]``): ``linear`` (additive straight lines), ``spline``
    (additive splines) and ``interaction`` (splines and age × albumin). The
    oracle's regret is 0. E[(p̂ − y)²] − E[(p − y)²] = E[(p̂ − p)²] for a binary y with risk p, so
    this is the Brier score's excess over the oracle's."""
    fit_frame, read_frame = readmission(seeds[0], rows[0]), readmission(seeds[1], rows[1])
    y = fit_frame[READMISSION_OUTCOME].to_numpy()
    p = readmission_risk(read_frame)
    out: dict[str, float] = {}
    for kind in ("linear", "spline", "interaction"):
        model, spline = _class_fit(fit_frame, y, kind)
        X, _ = _design(read_frame, kind)
        M = X if spline is None else _expanded(X, spline, kind)
        out[kind] = float(np.mean((model.predict_proba(M)[:, 1] - p) ** 2))
    return out


# ── the Estimate fixture ─────────────────────────────────────────────────────

ESTIMATE_ROWS = 1200
ESTIMATE_SEED = 20261023  # the committed table's seed
ESTIMATE_FILE = "walking_bp.csv"
# sbp = intercept + effect × walks + confounder effect × pain + age slope × (age − center) + ε;
# walks ~ Bernoulli(expit(a0 + a1 × pain)); the collider = c0 + c_A × walks + c_Y × (sbp − c) + η;
# pain is recorded when U < r0 + r1 × (sbp − c) / s, U standard normal (a probit in the outcome).
ESTIMATE_MODEL: dict[str, Any] = {
    "exposure": "walks_daily", "outcome": "sbp_mmhg",
    "intercept": 118.0, "effect": -5.0, "sd": 12.0,
    "age": {"column": "age", "low": 40, "high": 79, "center": 60, "slope": 0.5},
    "confounder": {"column": "chronic_pain", "prevalence": 0.35, "on_outcome": 8.0,
                   "exposure_log_odds": [0.2, -1.4]},
    "collider": {"column": "health_concern_score", "intercept": 50.0, "on_exposure": 6.0,
                 "on_outcome": 0.6, "center": 118.0, "sd": 8.0},
    "unrelated": {"column": "height_cm", "mean": 170.0, "sd": 9.0},
    "missing": {"column": "chronic_pain", "link": "probit", "intercept": 0.6, "slope": -1.0,
                "center": 118.0, "scale": 12.0},
}
# The plans graded against the truth (SWARM_PLAN §7): each its model's columns and its rows. The
# oracle adjusts for the confounder and age, leaves out the collider and height, and fills the
# confounder by multiple imputation, whose population limit is the full data's fit (the imputation
# model is the right one, :func:`population_effects`).
ESTIMATE_PLANS: dict[str, dict[str, Any]] = {
    "oracle": {"columns": ["walks_daily", "chronic_pain", "age"], "rows": "all"},
    "collider_kept": {"columns": ["walks_daily", "chronic_pain", "age", "health_concern_score"],
                      "rows": "all"},
    "confounder_dropped": {"columns": ["walks_daily", "age"], "rows": "all"},
    "complete_cases": {"columns": ["walks_daily", "chronic_pain", "age"], "rows": "complete"},
}
WRONG_PATHS = ("collider_kept", "confounder_dropped", "complete_cases")


def blood_pressure(seed: int = ESTIMATE_SEED, n: int = ESTIMATE_ROWS) -> pd.DataFrame:
    """The Estimate fixture's table at ``seed``. Five streams: the people (age, pain, height), the
    exposure, the outcome's noise, the collider's noise, and the blanks."""
    m = ESTIMATE_MODEL
    people, exposure, noise, concern, blanks = streams(seed, 5)
    a = m["age"]
    age = people.integers(a["low"], a["high"] + 1, n)
    c = m["confounder"]
    pain = (people.random(n) < c["prevalence"]).astype(int)
    height = np.round(people.normal(m["unrelated"]["mean"], m["unrelated"]["sd"], n), 1)
    a0, a1 = c["exposure_log_odds"]
    walks = (exposure.random(n) < expit(a0 + a1 * pain)).astype(int)
    sbp = np.round(m["intercept"] + m["effect"] * walks + c["on_outcome"] * pain
                   + a["slope"] * (age - a["center"]) + noise.normal(0.0, m["sd"], n), 1)
    k = m["collider"]
    score = np.round(k["intercept"] + k["on_exposure"] * walks + k["on_outcome"]
                     * (sbp - k["center"]) + concern.normal(0.0, k["sd"], n), 1)
    r = m["missing"]
    recorded = blanks.standard_normal(n) < r["intercept"] + r["slope"] * (sbp - r["center"]) \
        / r["scale"]
    frame = pd.DataFrame({
        "person_id": [f"P{i + 1:05d}" for i in range(n)], "age": age.astype(int),
        "chronic_pain": pd.array(np.where(recorded, pain, 0), dtype="Int64"),
        "walks_daily": walks, "height_cm": height, "health_concern_score": score,
        "sbp_mmhg": sbp,
    })
    frame.loc[~recorded, "chronic_pain"] = pd.NA
    return frame


def _cells() -> list[tuple[float, int, int, int]]:
    """Every (probability, pain, walks, age) cell of the people and the exposure, exactly."""
    m = ESTIMATE_MODEL
    a, c = m["age"], m["confounder"]
    ages = range(a["low"], a["high"] + 1)
    a0, a1 = c["exposure_log_odds"]
    out = []
    for pain in (0, 1):
        p_pain = c["prevalence"] if pain else 1 - c["prevalence"]
        p_walk = float(expit(a0 + a1 * pain))
        for walks in (0, 1):
            p = p_pain * (p_walk if walks else 1 - p_walk) / len(ages)
            out.extend((p, pain, walks, age) for age in ages)
    return out


def _phi(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _Phi(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2))


def population_effects() -> dict[str, float]:
    """The population least-squares coefficient on the exposure under each plan of
    :data:`ESTIMATE_PLANS`, in closed form.

    Given a cell (pain, walks, age), the outcome is normal with mean m and variance σ², and the
    collider is c0 + c_A·walks + c_Y·(sbp − c) + η, so every first and second moment the normal
    equations E[ZZ'] β = E[Z·sbp] need is exact, and the cells are a finite sum. Complete cases
    weight each cell's moments by the chance the confounder is recorded: R = 1{V > 0}, V = r0 +
    r1 (sbp − c)/s − U is normal with mean μ_V and standard deviation s_V, and for a W jointly
    normal with V, E[W 1{V>0}] = E[W] Φ(μ_V/s_V) + Cov(W, V)/s_V φ(μ_V/s_V), and E[W² 1{V>0}] =
    (E[W]² + Var W) Φ + 2 E[W] Cov(W,V)/s_V φ − Cov(W,V)² (μ_V/s_V) φ / s_V². The table's sbp is
    rounded to 0.1 mmHg and its blanks drawn from the rounded value; the derivation reads it
    unrounded (the tests check it against a simulation of the generator itself).

    The oracle's multiple imputation has the full data's fit as its limit because its imputation
    model is the right one: with the outcome normal given the cell and both pain and the exposure
    yes/no, the log-odds of pain given the exposure, age and sbp is exactly linear in them (the
    exposure enters through log p(walks | pain), linear in a yes/no; sbp through −(sbp − m)²/2σ²,
    whose pain terms are linear in sbp), and the collider depends on pain only through them."""
    m = ESTIMATE_MODEL
    a, c, k, r = m["age"], m["confounder"], m["collider"], m["missing"]
    sd = m["sd"]
    out: dict[str, float] = {}
    for name, plan in ESTIMATE_PLANS.items():
        cols = plan["columns"]
        dim = len(cols) + 1
        M, v = np.zeros((dim, dim)), np.zeros(dim)
        for p, pain, walks, age in _cells():
            mean = (m["intercept"] + m["effect"] * walks + c["on_outcome"] * pain
                    + a["slope"] * (age - a["center"]))
            fixed = {"walks_daily": walks, "chronic_pain": pain, "age": age}
            if plan["rows"] == "complete":
                mu_v = r["intercept"] + r["slope"] * (mean - r["center"]) / r["scale"]
                s_v = math.sqrt(1.0 + (r["slope"] * sd / r["scale"]) ** 2)
                cov = r["slope"] * sd ** 2 / r["scale"]
                t = mu_v / s_v
                w = _Phi(t)
                ey = mean * w + cov / s_v * _phi(t)
                z = np.array([1.0, *(float(fixed[col]) for col in cols)])
                M += p * w * np.outer(z, z)
                v += p * ey * z
                continue
            # Every row: the collider is the one random column a design may hold.
            mc = k["intercept"] + k["on_exposure"] * walks + k["on_outcome"] * (mean - k["center"])
            var_c = (k["on_outcome"] * sd) ** 2 + k["sd"] ** 2
            cov_cy = k["on_outcome"] * sd ** 2
            means = np.array([1.0, *(mc if col == k["column"] else float(fixed[col])
                                     for col in cols)])
            second = np.outer(means, means)
            if k["column"] in cols:
                j = cols.index(k["column"]) + 1
                second[j, j] += var_c
            ezy = means * mean
            if k["column"] in cols:
                ezy[cols.index(k["column"]) + 1] += cov_cy
            M += p * second
            v += p * ezy
        beta = np.linalg.solve(M, v)
        out[name] = float(beta[cols.index(m["exposure"]) + 1])
    return out


def planted_biases() -> dict[str, Any]:
    """Each plan's population bias on the exposure's effect, and b0 (SWARM_PLAN §7): half the
    smallest planted bias in size, computed from the generator (:func:`population_effects`)."""
    effects = population_effects()
    truth = float(ESTIMATE_MODEL["effect"])
    bias = {name: effects[name] - truth for name in effects}
    smallest = min(abs(bias[p]) for p in WRONG_PATHS)
    return {"effect": truth, "coefficients": effects, "bias": bias,
            "smallest_planted_bias": smallest, "b0": smallest / 2.0}


TUNING_DRAWS, TUNING_IMPUTATIONS = 100, 20
TUNING_FIRST_SEED = 1  # the tuning's draws are seeds 1 to 100, none the committed table's
B0_SE_RATIO = 4.0  # SWARM_PLAN §5 item 5: b0 at least four standard errors of the oracle's mean bias


def _ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Least squares with an intercept: the coefficients, their classical variances and the
    residual degrees of freedom (statsmodels' OLS)."""
    import statsmodels.api as sm

    fit = sm.OLS(y, sm.add_constant(X, has_constant="add")).fit()
    return np.asarray(fit.params), np.asarray(fit.bse) ** 2, int(fit.df_resid)


def _imputed_estimates(frame: pd.DataFrame, analysis: list[str], m: int,
                       rng: np.random.Generator) -> tuple[float, float, float]:
    """Multiple imputation of the confounder, independent of the engine: m proper draws from a
    logistic model of it given every other analysis column and the outcome (statsmodels' Logit;
    its coefficients drawn from N(β̂, its covariance) before each fill), each completed table fit
    by least squares, pooled by Rubin's rules with Barnard and Rubin's degrees of freedom. Returns
    the pooled coefficient on the exposure, its total variance and its degrees of freedom."""
    import statsmodels.api as sm

    pain, outcome, exposure = "chronic_pain", ESTIMATE_MODEL["outcome"], ESTIMATE_MODEL["exposure"]
    others = [c for c in analysis if c != pain]
    seen = frame[pain].notna().to_numpy()
    Xi = sm.add_constant(frame[others + [outcome]].to_numpy(dtype=float), has_constant="add")
    logit = sm.Logit(frame.loc[seen, pain].to_numpy(dtype=float), Xi[seen]).fit(disp=0)
    mean, cov = np.asarray(logit.params), np.asarray(logit.cov_params())
    j = analysis.index(exposure) + 1
    estimates, variances, df = [], [], 0
    for _ in range(m):
        beta = rng.multivariate_normal(mean, cov, method="cholesky")
        filled = frame[pain].to_numpy(dtype=float, na_value=np.nan).copy()
        blank = ~seen
        filled[blank] = (rng.random(int(blank.sum())) < expit(Xi[blank] @ beta)).astype(float)
        X = frame[analysis].to_numpy(dtype=float, na_value=np.nan).copy()
        X[:, analysis.index(pain)] = filled
        b, v, df = _ols(frame[outcome].to_numpy(dtype=float), X)
        estimates.append(b[j])
        variances.append(v[j])
    q, u, b_var = float(np.mean(estimates)), float(np.mean(variances)), float(np.var(estimates,
                                                                                    ddof=1))
    t = u + (1 + 1 / m) * b_var
    lam = (1 + 1 / m) * b_var / t
    old = (m - 1) / lam ** 2 if lam > 0 else math.inf
    obs = (df + 1) / (df + 3) * df * (1 - lam)
    return q, t, (old * obs / (old + obs)) if math.isfinite(old) else obs


def b0_tuning(draws: int = TUNING_DRAWS, m: int = TUNING_IMPUTATIONS, rows: int = ESTIMATE_ROWS,
              first_seed: int = TUNING_FIRST_SEED) -> dict[str, Any]:
    """The tuning run SWARM_PLAN §5 item 5 asks for, with an analysis independent of the engine:
    each plan of :data:`ESTIMATE_PLANS` on ``draws`` fresh draws of :func:`blood_pressure` (seeds
    ``first_seed`` up), the confounder filled by :func:`_imputed_estimates` where the plan's
    model holds it on every row, by complete cases where the plan says so, least squares
    otherwise. Per plan: the mean estimate, its mean bias against the planted effect, the standard
    deviation over the draws and the mean bias's standard error (that over √draws); the oracle's
    95% intervals' coverage; and whether b0 (:func:`planted_biases`) is at least
    :data:`B0_SE_RATIO` times the oracle's standard error, the oracle's mean bias within b0, and
    every wrong path's beyond it."""
    from scipy import stats

    planted = planted_biases()
    truth, b0 = planted["effect"], planted["b0"]
    outcome, exposure = ESTIMATE_MODEL["outcome"], ESTIMATE_MODEL["exposure"]
    found: dict[str, list[float]] = {name: [] for name in ESTIMATE_PLANS}
    covered = 0
    for i in range(draws):
        seed = first_seed + i
        frame = blood_pressure(seed, rows)
        rng = np.random.default_rng([seed, 1])
        for name, plan in ESTIMATE_PLANS.items():
            cols = plan["columns"]
            if "chronic_pain" in cols and plan["rows"] == "all":
                q, t, df = _imputed_estimates(frame, cols, m, rng)
                if name == "oracle":
                    half = stats.t.ppf(0.975, df) * math.sqrt(t)
                    covered += int(q - half <= truth <= q + half)
            else:
                rows_ = frame.dropna(subset=cols) if plan["rows"] == "complete" else frame
                b, _, _ = _ols(rows_[outcome].to_numpy(dtype=float),
                               rows_[cols].to_numpy(dtype=float, na_value=np.nan))
                q = float(b[cols.index(exposure) + 1])
            found[name].append(q)
    plans = {}
    for name, values in found.items():
        arr = np.asarray(values)
        sd = float(arr.std(ddof=1))
        plans[name] = {"mean": float(arr.mean()), "mean_bias": float(arr.mean() - truth), "sd": sd,
                       "se_of_mean_bias": sd / math.sqrt(draws)}
    se = plans["oracle"]["se_of_mean_bias"]
    return {"draws": draws, "imputations": m, "rows": rows,
            "seeds": [first_seed, first_seed + draws - 1], "b0": b0, "plans": plans,
            "oracle_coverage": covered / draws, "b0_over_oracle_se": b0 / se,
            "holds": bool(b0 >= B0_SE_RATIO * se),
            "oracle_within_b0": abs(plans["oracle"]["mean_bias"]) < b0,
            "wrong_paths_beyond_b0": {p: abs(plans[p]["mean_bias"]) > b0 for p in WRONG_PATHS}}


# ── the measurement fixture ──────────────────────────────────────────────────

PANEL_SEED = 20261024  # the committed table's seed
PANEL_DESIGN_SEED = 2026102401  # the panel's fixed parameters: never the rows'
PANEL_FILE = "plasma_panel.csv"
PANEL_OUTCOME = "disease"
PANEL: dict[str, Any] = {
    "batches": 4, "injections_per_batch": 32,
    # A pooled QC opens and closes each batch and comes every fifth injection between: seven per
    # batch, at most five injections apart (qc_drift's MIN_QC and MAX_GAP hold).
    "qc_positions": [0, 5, 10, 15, 20, 25, 31],
    "features": 150, "associated": 15,
    "log2_mean": [8.0, 18.0], "log2_sd": [0.4, 0.9], "log2_effect": [0.6, 1.2],
    # Dilution: each participant's sample is diluted by 2^d, d ~ N(case_shift × case, sd): the
    # cases' samples are more dilute, which moves every feature alike.
    "dilution": {"sd": 0.3, "case_shift": -0.35},
    # Drift: within each batch feature j's signal moves by slope_j × (place − mid) / (places − 1)
    # on the log2 scale, slope_j in ±drift, and starts again in the next batch (the instrument's
    # response wanders over a batch and is reset between batches): a trend in injection order.
    "drift": 1.5,
    # Batch offsets: feature j's offset in batch b is pattern[b] × size_j plus a little noise, size_j
    # of either sign, so the offsets go up and down along injection order (no trend in injection
    # order removes them) and move features in both directions (no sample's dilution absorbs them).
    "batch_pattern": [0.4, -0.4, 0.4, -0.4], "batch_size": [0.3, 1.0], "batch_noise": 0.05,
    "technical_sd": 0.12,
    "detection_limit_log2": 8.5,  # an injection's value below 2^8.5 is not detected: a blank
    # Case status: log-odds intercept + batch_shift in batches 2 and 4 + order_slope × the
    # injection's standardized place in its batch, so the cases sit where the offsets and the drift
    # are, as they do when samples are not randomized to injection order.
    "case_log_odds": {"intercept": -0.3, "batch_shift": 1.5, "order_slope": 1.2},
}


def panel_design(seed: int = PANEL_DESIGN_SEED) -> dict[str, Any]:
    """The panel's fixed parameters, from their own seed: each feature's log2 mean and biological
    SD, the associated set and its log2 fold changes (alternating in sign), each feature's drift
    slope and batch offsets."""
    p = PANEL
    rng = np.random.default_rng(seed)
    f = p["features"]
    names = [f"mz_{j + 1:04d}" for j in range(f)]
    mean = rng.uniform(*p["log2_mean"], f)
    sd = rng.uniform(*p["log2_sd"], f)
    chosen = np.sort(rng.choice(f, p["associated"], replace=False))
    effect = np.zeros(f)
    sizes = rng.uniform(*p["log2_effect"], p["associated"])
    effect[chosen] = sizes * np.where(np.arange(p["associated"]) % 2 == 0, 1.0, -1.0)
    slope = rng.uniform(-p["drift"], p["drift"], f)
    size = rng.uniform(*p["batch_size"], f) * rng.choice([-1.0, 1.0], f)
    offsets = (np.outer(size, p["batch_pattern"])
               + rng.normal(0.0, p["batch_noise"], (f, p["batches"])))
    return {"names": names, "log2_mean": mean, "log2_sd": sd, "log2_effect": effect,
            "associated": [names[j] for j in chosen], "drift_slope": slope,
            "batch_offset": offsets}


def panel_layout() -> pd.DataFrame:
    """Every injection: its run order (1 up), batch (1 up), place in the batch, and whether it is
    a pooled QC."""
    p = PANEL
    rows = []
    for b in range(p["batches"]):
        for i in range(p["injections_per_batch"]):
            rows.append({"run_order": b * p["injections_per_batch"] + i + 1, "batch": b + 1,
                         "place": i, "qc": i in p["qc_positions"]})
    return pd.DataFrame(rows)


def panel_latent(seed: int = PANEL_SEED) -> dict[str, Any]:
    """A draw's latent parts: the layout, each participant's case status and covariates, the true
    log2 concentrations (before dilution, drift, batch and the instrument), the log2 dilution and
    the measured log2 values before the detection limit. Five streams: case status, biology,
    dilution, the instrument's noise, the covariates."""
    p, d = PANEL, panel_design()
    lay = panel_layout()
    cases, biology, dilute, instrument, people = streams(seed, 5)
    n = len(lay)
    qc = lay["qc"].to_numpy()
    batch = lay["batch"].to_numpy()
    per = p["injections_per_batch"]
    place = (lay["place"].to_numpy() - (per - 1) / 2) / np.std(np.arange(per))
    cl = p["case_log_odds"]
    logit = cl["intercept"] + cl["batch_shift"] * np.isin(batch, (2, 4)) + cl["order_slope"] * place
    case = np.where(qc, 0, (cases.random(n) < expit(logit)).astype(int))
    f = len(d["names"])
    noise = biology.normal(0.0, 1.0, (n, f)) * d["log2_sd"]
    true = d["log2_mean"] + np.outer(case, d["log2_effect"]) + noise
    # A pooled QC is one pooled sample: no biology of its own, half the cases' fold change.
    true[qc] = d["log2_mean"] + 0.5 * d["log2_effect"]
    dil = np.where(qc, 0.0, dilute.normal(0.0, p["dilution"]["sd"], n)
                   + p["dilution"]["case_shift"] * case)
    trend = np.outer((lay["place"].to_numpy(dtype=float) - (per - 1) / 2) / (per - 1),
                     d["drift_slope"])
    offset = d["batch_offset"][:, batch - 1].T
    measured = (true + dil[:, None] + trend + offset
                + instrument.normal(0.0, p["technical_sd"], (n, f)))
    return {"layout": lay, "case": case, "true_log2": true, "dilution_log2": dil,
            "drift_log2": trend, "batch_log2": offset, "measured_log2": measured,
            "people": people}


def metabolite_panel(seed: int = PANEL_SEED) -> pd.DataFrame:
    """The measurement fixture's table at ``seed``: one row per injection (participants and pooled
    QCs) in injection order, the participants' age, sex and BMI (unrelated to every feature and to
    case status), each feature's intensity (2 to the measured log2 value, two decimals; blank below
    the detection limit), and the outcome, blank on a QC."""
    p, d = PANEL, panel_design()
    lat = panel_latent(seed)
    lay, people = lat["layout"], lat["people"]
    n = len(lay)
    qc = lay["qc"].to_numpy()
    values = np.round(np.power(2.0, lat["measured_log2"]), 2)
    values[lat["measured_log2"] < p["detection_limit_log2"]] = np.nan
    age = people.integers(28, 78, n).astype(float)
    sex = people.choice(["F", "M"], n)
    bmi = np.round(people.normal(27.5, 4.6, n), 1)
    numbers = iter(range(1, n + 1))
    qcs = iter(range(1, n + 1))
    frame = {
        "sample_id": [f"QC{next(qcs):02d}" if q else f"S{next(numbers):03d}" for q in qc],
        "sample_type": np.where(qc, "pooled_qc", "participant"),
        "run_order": lay["run_order"].to_numpy(),
        "batch": [f"B{b}" for b in lay["batch"]],
        "age": np.where(qc, np.nan, age), "sex": np.where(qc, "", sex),
        "bmi": np.where(qc, np.nan, bmi),
    }
    for j, name in enumerate(d["names"]):
        frame[name] = values[:, j]
    out = pd.DataFrame(frame)
    out["age"] = out["age"].astype("Int64")
    out[PANEL_OUTCOME] = pd.array(np.where(qc, 0, lat["case"]), dtype="Int64")
    out.loc[qc, PANEL_OUTCOME] = pd.NA
    return out


def panel_truth() -> dict[str, Any]:
    """The measurement fixture's planted truth, by construction (:func:`panel_design`): the
    associated features and each feature's log2 fold change between cases and controls in the
    true concentration (0 for every other feature)."""
    d = panel_design()
    return {"associated": list(d["associated"]),
            "log2_fold_change": {n: float(e) for n, e in zip(d["names"], d["log2_effect"])
                                 if e != 0.0},
            "features": len(d["names"]), "detection_limit_log2": PANEL["detection_limit_log2"]}


def graded_measurement_decisions() -> list[dict[str, Any]]:
    """The measurement decisions the fixture grades: exactly the engine's own menus, read from
    the engine (SWARM_PLAN §5 item 6 and §10's watch note on package 2). The normalization and
    the log scale are one answer, the omics-scale finding's repair; batch handling is the batch
    answer and the pooled-QC finding's drift-and-batch correction (QC-RLSC); detection-limit
    handling is the missing-values answer's fill for values below the limit."""
    from typing import get_args

    from turbotab.core import decisions
    from turbotab.core.methods import omics, qc_drift

    return [
        {"decision": "normalization and the log scale",
         "records": "apply_repair on the omics_scale finding",
         "options": list(omics.INTENSITY_METHODS)},
        {"decision": "batch handling", "records": "set_batch",
         "options": list(get_args(decisions.BatchMethod))},
        {"decision": "drift and batch correction from the pooled QCs",
         "records": f"apply_repair on {qc_drift.FINDING}",
         "options": list(qc_drift.OPTIONS)},
        {"decision": "detection-limit handling", "records": "set_missing below_detection",
         "options": list(get_args(decisions.BelowDetection))},
    ]


# Each committed table: its generator, the seed it is drawn at and its file under FIXTURES. A run's
# table is this file (the intake copies it; its manifest names the generator and the seed), and a
# fresh draw is the same generator at another seed (``core/export/replay.py``).
FIXTURES = Path(__file__).resolve().parent / "fixtures"
TABLES: dict[str, dict[str, Any]] = {
    READMISSION_FILE: {"function": "readmission", "seed": READMISSION_SEED,
                       "rows": READMISSION_ROWS},
    ESTIMATE_FILE: {"function": "blood_pressure", "seed": ESTIMATE_SEED, "rows": ESTIMATE_ROWS},
    PANEL_FILE: {"function": "metabolite_panel", "seed": PANEL_SEED,
                 "rows": PANEL["batches"] * PANEL["injections_per_batch"]},
}


def table_bytes(name: str, seed: int | None = None) -> bytes:
    """The table ``name`` as its generator writes it at ``seed`` (default: the committed one's):
    CSV, no index, as the intake and a fresh draw write a frame."""
    entry = TABLES[name]
    frame = globals()[entry["function"]](entry["seed"] if seed is None else seed)
    return frame.to_csv(index=False).encode("utf-8")


def main(argv: list[str] | None = None) -> int:
    """``python -m research.swarm.generators [--check]``: write each committed table under
    ``research/swarm/fixtures/`` from its generator and seed, or (``--check``) fail where a file is
    not its generator's bytes."""
    import argparse

    parser = argparse.ArgumentParser(description="Write the swarm's generated fixture tables.")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    bad = []
    for name in TABLES:
        path, data = FIXTURES / name, table_bytes(name)
        if args.check:
            if not path.is_file() or path.read_bytes() != data:
                bad.append(name)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        print(f"wrote {path} ({len(data):,} bytes)")
    for name in bad:
        print(f"{name}: not its generator's bytes")
    return 1 if bad else 0


__all__ = ["ESTIMATE_MODEL", "ESTIMATE_PLANS", "FIXTURES", "PANEL", "READMISSION_MODEL", "TABLES",
           "b0_tuning", "blood_pressure", "graded_measurement_decisions", "main", "metabolite_panel",
           "panel_design", "panel_latent", "panel_truth", "planted_biases", "population_effects",
           "readmission", "readmission_log_odds", "readmission_regrets", "readmission_risk",
           "table_bytes"]


if __name__ == "__main__":
    import sys

    sys.exit(main())
