"""MS1–MS3 · Multiple imputation compatible with the analysis model, the MI frame, and pooling
everything (docs/turbotab-next/MODELING_SEQUENCE.md §0 rulings 5 and 12, §1.1, §2, §4, §5; the
review record docs/turbotab-next/audit/modeling-sequence-review.json). WP18's open deviation (the
NHANES DXA imputed-outcome copies) closes here too.

The package's acceptance items, in its order (each a test below, named by its number):

1. SMC-FCS for a substantive model with a spline or a product term, for linear, logistic and Cox
   outcomes: pooled estimates agree with R ``smcfcs::smcfcs`` on the same data (m = 50) within 2
   Monte Carlo SE; in Bartlett et al. 2015's quadratic design the quadratic coefficient is recovered
   (mean ≈ 1.0, 95% CI covers 1.0 in ≥ 90% of 200 reps) where passive imputation gives ≈ 0.70.
2. Logged quantities are imputed on the log scale: no completed copy holds a non-positive value of a
   logged column; a NaN made inside a copy never reaches the median imputer.
3. The energy identity: the sources and "other" are imputed and E derived as their sum; other is
   never negative in any copy.
4. Knots are placed once on observed values and fixed across copies (identical basis in every copy).
5. Under a survey design the imputation model holds strata, PSU and the weight, and Rubin's rules use
   ν_com = the design df: pooled coefficient and interval agree with R (survey + mitools::MIcombine
   with df.complete) to 1e-6 / 1e-4.
6. Clustered MI (ruling 12): time-invariant variables imputed once per unit; time-varying ones with
   the unit means of the others; single-level MI on clustered rows is block-and-record.
7. m = max(20, % of rows with any imputed value), and the Monte Carlo error in the record.
8. D1 for multi-df tests (spline overall and nonlinearity, interaction, global quintile) agrees with
   R ``mitml`` (D1) to 1e-6.
9. Every estimate under inference with MI is pooled: substitution curves (linear all-components: the
   exact contrast of the pooled coefficients with the pooled covariance, against NumPy; otherwise
   per-copy curves pooled at each k), margins and form tests; the single-fill curve is gone.
10. Imputed-outcome copies (NHANES DXA): each copy carries its own outcome and Rubin's rules pool
    them (against R mitools), replacing WP18's block.
11. Leash: passive MI with a declared nonlinear term is block-and-record under inference, and the
    methods sentence says the nonlinearity tests would be biased toward the null.

Then the §13 contracts and the §2 relations in a chain test.

**How two items are read.** (1) "Within 2 Monte Carlo SE" is applied to the p pooled coefficients
together: their difference lies in the 2-SE Monte Carlo region (χ²_p at 95.45%), and no coefficient
is beyond 3 SE alone; each coefficient on its own at 2 SE would fail an exact implementation by
chance alone about one time in five per model (:func:`agree_within_2_mcse`). The engine declares no
product terms yet (effect modification is M3.5's), so the product term's SMC-FCS (1) and the
interaction's D1 (8) are checked on the sampler and the pooling directly; splines go through the
engine. (5) ``mitools::MIcombine`` computes ν_obs with Ū/(Ū + B) where Barnard & Rubin (1999) have
1 − λ = Ū/T; the engine follows Barnard & Rubin, as ``mice`` does, so its coefficient and total
variance are checked against MIcombine and its df and interval against ``mice::pool.scalar``, with
MIcombine's own interval bounded within 1% of a standard error.

**Independent references.** R 4.6.1 (``Rscript``, called on CSV files the tests write; never by the
app): ``smcfcs`` 2.0.2, ``mitml`` 0.4.5, ``mitools`` 2.7, ``mice`` 3.19.0, ``survey`` 4.5, ``Hmisc``
via ``rms`` 8.1.1, ``survival``. Each R test is skipped where ``Rscript`` is not installed. Where R
has no counterpart, the reference is NumPy arithmetic written out here, or a simulation's known
truth (Bartlett et al. 2015's design: X ~ N(2, 1), Y = 4 − 4X + X² + ε with R² = 0.5, so σ²ε = 2,
30% of X missing completely at random, n = 1000, 10 imputations of 10 iterations; their Table 2,
normal X, MCAR: linear passive 0.696 (SD 0.041, coverage 0.0%), SMC-FCS 0.998 (0.038, 93.9%)).

**Source checks** (the quotes the rulings rest on are in the review record):

* Bartlett JW, Seaman SR, White IR, Carpenter JR. *Stat Methods Med Res* 2015;24:462 (PMC4513015),
  §7.1: "X had mean 2 and variance 1"; "β₀ = 4, β₁ = −4, β₂ = 1"; "The variance σε2 was chosen
  such that the coefficient of determination R2 was equal to 0.5"; "P(R=1|X,Y)=0.7"; "We used 10
  iterations per imputation in SMC-FCS".
* White IR, Royston P, Wood AM. *Stat Med* 2011;30:377, as White, Pandis & Pham put it: "the number
  of imputations should be at least equal to the percentage of incomplete cases … More important is
  to use one's statistical software to report the Monte Carlo errors".
* Reiter JP, Raghunathan TE, Kinney SK. *Survey Methodology* 2006;32:143: "the safest course of
  action is to include design variables in the specification of imputation models".
* Lüdtke O, Robitzsch A, Grund S. *Psychol Methods* 2017;22:141: "the imputation model used to
  generate the imputed values must be at least as general as the analysis model".
* CDC, NHANES 1999–2006 DXA "Multiple Imputation Details": "The preferred statistical approach is
  to analyze EACH OF THE FIVE datasets separately … and then combining the estimates and standard
  errors using the combining rules".
* Schomaker M, Heumann C. *Stat Med* 2018;37:2252: bootstrap inference with multiple imputation.

Monte Carlo error is stated beside each simulated bound.
"""
from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import turbotab.core.models  # noqa: F401 - registers the families
from turbotab.core.decisions import (EnergyAdjustment, ExposureFormSpec, MissingSpec,
                                     ProjectState, Refusal, SetMissing, SplitSpec,
                                     SubstitutionSpec, validate)
from turbotab.core.methods.imputation import pool_scalar, pooled_wald
from turbotab.core.methods.missing import impute_for_inference
from turbotab.core.methods.smcfcs import Outcome as SMOutcome
from turbotab.core.methods.smcfcs import Substantive, Variable, impute
from turbotab.core.models import get_family
from turbotab.core.models.inference import INDEPENDENT, Outcome
from turbotab.core.models.inner_cv import fit_pipeline
from turbotab.core.models.pipeline import build_pipeline, design_spec
from turbotab.core.models.survival import survival_outcome
from turbotab.core.stages.modeling import pooled_table

RSCRIPT = shutil.which("Rscript")
needs_r = pytest.mark.skipif(RSCRIPT is None, reason="R (Rscript) is not installed: the R reference "
                                                     "is skipped")
LINEAR = get_family("linear")
COX = get_family("cox")


def run_r(folder: Path, script: str, **frames: pd.DataFrame) -> Path:
    """Write ``frames`` as CSV into ``folder`` and run ``script`` there with Rscript (the user's
    library, so ``--vanilla``'s ``--no-environ`` is not used); returns ``folder``."""
    folder.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_csv(folder / f"{name}.csv", index=False, na_rep="NA")
    (folder / "reference.R").write_text(script)
    done = subprocess.run([RSCRIPT, "--no-save", "--no-restore", "reference.R"], cwd=folder,
                          capture_output=True, text=True, timeout=900)
    assert done.returncode == 0, done.stderr[-4000:]
    return folder


def rcs_r(x: str, knots: Any) -> list[str]:
    """Harrell's restricted cubic spline terms (norm = 2) of ``x`` at ``knots`` as R expressions,
    written out from Harrell 2015 §2.4.5 (each smcfcs passive variable is one of them)."""
    t = [float(v) for v in knots]
    kd = (t[-1] - t[0]) ** (2 / 3)
    out = []
    for j in range(len(t) - 2):
        out.append(f"pmax(({x} - {t[j]!r})/{kd!r}, 0)^3 + (({t[-2]!r} - {t[j]!r})*pmax(({x} - "
                   f"{t[-1]!r})/{kd!r}, 0)^3 - ({t[-1]!r} - {t[j]!r})*pmax(({x} - {t[-2]!r})/"
                   f"{kd!r}, 0)^3)/({t[-1]!r} - {t[-2]!r})")
    return out


SMCFCS_SCRIPT = """
suppressMessages({{library(smcfcs); library(survival)}})
dat <- read.csv("data.csv")
set.seed({seed})
method <- c({method})
invisible(capture.output(imps <- smcfcs(dat, smtype = "{smtype}", smformula = "{formula}",
                                        method = method, m = {m}, numit = 10)))
est <- c()
for (i in 1:{m}) {{
  d <- imps$impDatasets[[i]]
  fit <- {analysis}
  est <- rbind(est, coef(fit))
}}
write.csv(est, "estimates.csv", row.names = FALSE)
"""


def smcfcs_reference(folder: Path, data: pd.DataFrame, method: list[str], smtype: str,
                     formula: str, analysis: str, seed: int, m: int = 50) -> np.ndarray:
    """R smcfcs's m imputations of ``data``, each analyzed by ``analysis``: the m × p estimates."""
    script = SMCFCS_SCRIPT.format(seed=seed, method=", ".join(f'"{x}"' for x in method),
                                  smtype=smtype, formula=formula, m=m, analysis=analysis)
    run_r(folder, script, data=data)
    return pd.read_csv(folder / "estimates.csv").to_numpy(dtype=float)


TWO_SE = 2 * stats.norm.cdf(2) - 1  # the share of a normal within 2 SE: 0.9545


def agree_within_2_mcse(ours: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    """The pooled estimates of two sets of m independent imputations (rows: imputations; columns:
    coefficients) against each other, on the Monte Carlo error of their difference: its covariance
    is B₁/m₁ + B₂/m₂ (each B the between-imputation covariance of the per-copy estimates).

    ``z``: each coefficient's |difference| over its Monte Carlo SE. ``joint``: the squared
    Mahalanobis length of the difference vector, which lies within the 2-SE region of p coefficients
    when it is at most ``bound``, χ²_p at 95.45% (2 SE's share): "within 2 Monte Carlo SE" for the p
    estimates together. Each coefficient on its own at 2 SE would, by chance alone, fail an exact
    implementation with probability 1 − 0.9545^p (about 21% for p = 5)."""
    d = ours.mean(axis=0) - reference.mean(axis=0)
    S = (np.atleast_2d(np.cov(ours, rowvar=False)) / len(ours)
         + np.atleast_2d(np.cov(reference, rowvar=False)) / len(reference))
    z = np.abs(d) / np.sqrt(np.diag(S))
    joint = float(d @ np.linalg.solve(S, d))
    return {"ours": ours.mean(axis=0), "reference": reference.mean(axis=0), "z": z, "joint": joint,
            "bound": float(stats.chi2.ppf(TWO_SE, len(d)))}


def assert_agree(found: dict[str, Any]) -> None:
    """Within 2 Monte Carlo SE jointly, and no coefficient beyond 3 on its own."""
    assert found["joint"] <= found["bound"], found
    assert float(np.max(found["z"])) <= 3.0, found


# ── 1 · SMC-FCS against R smcfcs, and Bartlett et al.'s quadratic design ─────


def spline_table(seed: int = 11, n: int = 600) -> pd.DataFrame:
    """y = 1 + sin(1.5x) + 0.6x + 0.5z + ε; x missing at random given z (about 30%)."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    x = 0.5 * z + rng.normal(size=n)
    y = 1 + np.sin(1.5 * x) + 0.6 * x + 0.5 * z + rng.normal(0, 0.7, n)
    x = np.where(rng.random(n) < 1 / (1 + np.exp(-(-1 + z))), np.nan, x)
    return pd.DataFrame({"y": y, "x": x, "z": z})


def engine_copies(frame: pd.DataFrame, task: str, y: Any, forms: dict[str, Any], m: int, seed: int,
                  family: Any = LINEAR) -> tuple[Any, Any, list[Any]]:
    """The engine's imputations of ``frame``'s x and z under inference with ``forms``, each copy
    fit by ``family`` as the fit stage fits it: (imputations, spec, per-copy inference tables)."""
    roles = {"x": "exposure", "z": "covariate"}
    st = ProjectState(target="y", task=task, purpose="inference", roles=roles,
                      missing=MissingSpec(strategy="multiple_imputation", m=m),
                      exposure_forms={c: ExposureFormSpec(**f) for c, f in forms.items()})
    X = frame[["x", "z"]]
    spec = design_spec(st, X, list(roles))
    imputations = impute_for_inference(spec, X, y, task, seed=seed)
    pipe = build_pipeline(spec, family, task, "inference", len(X), 2)
    collected: dict[str, Any] = {}
    pooled_table(family, pipe, imputations, y, task=task, clusters=INDEPENDENT,
                 outcome=Outcome(name="y"), survey=None, design=None, spec=spec, rows="all",
                 fit=lambda model, X_k: fit_pipeline(model, X_k, y), collect=collected)
    return imputations, spec, collected["tables"]


def estimates(tables: list[Any], names: list[str]) -> np.ndarray:
    return np.asarray([[next(r["estimate"] for r in t.rows if r["feature"] == n) for n in names]
                       for t in tables], dtype=float)


@needs_r
def test_1_smcfcs_linear_spline_agrees_with_r_smcfcs(tmp_path):
    """Linear outcome, a 4-knot restricted cubic spline of x and a covariate z: the engine's own
    SMC-FCS (``impute_for_inference``, the analysis pipeline as its substantive model) against R
    ``smcfcs(smtype = "lm")`` with the spline's terms as passive variables at the same knots, both
    m = 50, each copy analyzed by OLS (``lm``); the pooled coefficients within 2 Monte Carlo SE
    (:func:`agree_within_2_mcse`)."""
    frame = spline_table()
    y = frame["y"].to_numpy()
    imputations, spec, tables = engine_copies(frame, "regression", y,
                                              {"x": {"form": "spline", "knots": 4}}, 50, seed=1)
    assert imputations.method == "smcfcs" and imputations.m == 50
    knots = imputations.plan["forms"]["x"]["knots"]
    ours = estimates(tables, ["(intercept)", "x", "x'", "x''", "z"])
    passive = rcs_r("x", knots)
    data = frame.assign(s1=np.nan, s2=np.nan)
    reference = smcfcs_reference(tmp_path, data, ["", "norm", "", *passive], "lm",
                                 "y ~ x + s1 + s2 + z", "lm(y ~ x + s1 + s2 + z, data = d)", seed=1)
    assert_agree(agree_within_2_mcse(ours, reference))


@needs_r
def test_1_smcfcs_cox_spline_agrees_with_r_smcfcs(tmp_path):
    """Cox outcome (event indicator and follow-up), a 3-knot spline of x and z: the engine's SMC-FCS
    with the Cox family's pipeline against R ``smcfcs(smtype = "coxph")``, m = 50 each, every copy
    fit by Cox's partial likelihood (Efron ties; ``coxph``); the pooled log hazard ratios within 2
    Monte Carlo SE."""
    rng = np.random.default_rng(13)
    n = 800
    z = rng.normal(size=n)
    x = 0.5 * z + rng.normal(size=n)
    hazard = 0.1 * np.exp(0.5 * x + 0.4 * np.maximum(x, 0) ** 2 + 0.3 * z)
    event_time, censor = rng.exponential(1 / hazard), rng.uniform(0, 15, n)
    t, d = np.minimum(event_time, censor), (event_time <= censor).astype(float)
    x = np.where(rng.random(n) < 1 / (1 + np.exp(-(-1 + 0.7 * z))), np.nan, x)
    frame = pd.DataFrame({"t": t, "d": d, "x": x, "z": z})
    y = survival_outcome(d, t)
    imputations, _, tables = engine_copies(frame, "time_to_event", y,
                                           {"x": {"form": "spline", "knots": 3}}, 50, seed=3,
                                           family=COX)
    assert imputations.method == "smcfcs" and imputations.plan["substantive"] == "cox"
    ours = estimates(tables, ["x", "x'", "z"])
    knots = imputations.plan["forms"]["x"]["knots"]
    reference = smcfcs_reference(tmp_path, frame.assign(s1=np.nan), ["", "", "norm", "",
                                                                      *rcs_r("x", knots)],
                                 "coxph", "Surv(t, d) ~ x + s1 + z",
                                 "coxph(Surv(t, d) ~ x + s1 + z, data = d)", seed=3)
    assert_agree(agree_within_2_mcse(ours, reference))


@needs_r
def test_1_smcfcs_logistic_product_term_agrees_with_r_smcfcs(tmp_path):
    """Logistic outcome with a product term x·z (z binary, complete): SMC-FCS (``methods.smcfcs``,
    the substantive design [x, z, x·z]) against R ``smcfcs(smtype = "logistic")`` with the product
    as a passive variable, m = 50 each, every copy fit by logistic maximum likelihood (statsmodels'
    Logit; R's ``glm``); the pooled coefficients within 2 Monte Carlo SE. The engine declares no
    product terms yet (effect modification is M3.5's), so the sampler is checked directly."""
    import statsmodels.api as sm

    rng = np.random.default_rng(12)
    n = 1000
    z = (rng.random(n) < 0.4).astype(float)
    x = 0.4 * z + rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(-0.5 + 0.7 * x + 0.5 * z - 0.6 * x * z)))).astype(float)
    x = np.where(rng.random(n) < 1 / (1 + np.exp(-(-1.2 + 0.8 * z))), np.nan, x)

    def design(f: pd.DataFrame) -> np.ndarray:
        xv, zv = f["x"].to_numpy(dtype=float), f["z"].to_numpy(dtype=float)
        return np.column_stack([xv, zv, xv * zv])

    imputations = impute(pd.DataFrame({"x": x, "z": z}), [Variable("x", "numeric")], mode="smcfcs",
                         substantive=Substantive(SMOutcome("logistic", y=y), design), m=50, seed=2)
    ours = np.asarray([sm.Logit(y, sm.add_constant(design(f), has_constant="add")).fit(disp=0).params
                       for f in imputations.frames])
    reference = smcfcs_reference(tmp_path, pd.DataFrame({"y": y, "x": x, "z": z, "xz": np.nan}),
                                 ["", "norm", "", "x*z"], "logistic", "y ~ x + z + xz",
                                 "glm(y ~ x + z + xz, family = binomial, data = d)", seed=2)
    assert_agree(agree_within_2_mcse(ours, reference))


def bartlett_replicate(seed: int, mode: str) -> tuple[float, bool]:
    """One replicate of Bartlett et al. 2015's §7.1 design (normal X, MCAR): the pooled quadratic
    coefficient of OLS y ~ x + x² over 10 imputations of 10 iterations, and whether its 95% interval
    (Rubin's rules, Barnard–Rubin df with ν_com = n − 3) covers the true 1."""
    rng = np.random.default_rng(seed)
    n = 1000
    x = rng.normal(2, 1, n)
    y = 4 - 4 * x + x ** 2 + rng.normal(0, math.sqrt(2.0), n)  # Var((X − 2)²) = 2, so R² = 0.5
    observed = rng.random(n) < 0.7
    data = pd.DataFrame({"x": np.where(observed, x, np.nan)})
    if mode == "smcfcs":
        sub = Substantive(SMOutcome("linear", y=y), lambda f: np.column_stack(
            [f["x"].to_numpy(), f["x"].to_numpy() ** 2]))
        imps = impute(data, [Variable("x", "numeric")], mode="smcfcs", substantive=sub, m=10,
                      iterations=10, seed=seed)
    else:  # linear passive: x from a model linear in y, x² derived in each copy (the old engine)
        imps = impute(data.assign(__outcome=y), [Variable("x", "numeric")], mode="fcs", m=10,
                      iterations=10, seed=seed)
    q, u = [], []
    for f in imps.frames:
        X = np.column_stack([np.ones(n), f["x"], f["x"] ** 2])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        q.append(beta[2])
        u.append(float(resid @ resid / (n - 3) * np.linalg.inv(X.T @ X)[2, 2]))
    pooled = pool_scalar(q, u, df_com=n - 3)
    return pooled.estimate, bool(pooled.ci_low <= 1.0 <= pooled.ci_high)


def test_1_bartletts_quadratic_design_recovers_the_quadratic_where_passive_imputation_does_not():
    """Bartlett et al. 2015 §7.1, normal X, MCAR, 200 replicates: SMC-FCS recovers β₂ = 1 (mean
    within 3 Monte Carlo SE of 1.0, where their SD 0.038 gives an MCSE of 0.0027 over 200; interval
    coverage at least 90%, their 93.9%), while passive imputation from a model linear in y gives
    about 0.70 (their 0.696, SD 0.041) and never covers."""
    smc = [bartlett_replicate(seed, "smcfcs") for seed in range(200)]
    passive = [bartlett_replicate(seed, "passive") for seed in range(200)]
    est = np.array([e for e, _ in smc])
    assert abs(est.mean() - 1.0) <= 3 * est.std(ddof=1) / math.sqrt(len(est))
    assert np.mean([c for _, c in smc]) >= 0.90
    old = np.array([e for e, _ in passive])
    assert abs(old.mean() - 0.696) <= 3 * old.std(ddof=1) / math.sqrt(len(old)) + 0.01
    assert np.mean([c for _, c in passive]) <= 0.05


# ── 2–4 · the log scale, the energy identity, fixed knots ────────────────────

DIET_ROLES = {"fiber_g": "exposure", "protein_g": "exposure", "fat_g": "exposure",
              "carb_g": "exposure", "kcal": "energy", "age": "covariate"}
DIET_FACTORS = {"protein_g": 4.0, "fat_g": 9.0, "carb_g": 4.0}  # as the ledger settles grams


def diet_frame(seed: int = 21, n: int = 600) -> pd.DataFrame:
    """Total energy is the sources' Atwater kcal plus 'other' (always positive); fiber rises with
    energy; y bends with fiber. Blanks: fiber (about 25%, more at older ages), protein (15%), total
    energy (15%) and age (15%), each at random given what is recorded."""
    rng = np.random.default_rng(seed)
    age = rng.normal(50, 10, n)
    size = np.exp(rng.normal(np.log(2100), 0.25, n))
    kp = size * np.clip(rng.normal(0.16, 0.03, n), 0.06, 0.35)
    kf = size * np.clip(rng.normal(0.34, 0.05, n), 0.15, 0.55)
    kc = size * np.clip(rng.normal(0.42, 0.05, n), 0.2, 0.65)
    other = size * rng.uniform(0.02, 0.12, n)
    fiber = np.exp(rng.normal(np.log(18), 0.35, n)) * (size / 2100) ** 0.6
    y = (120 + 8 * np.sin(2.5 * (np.log(fiber) - np.log(18))) - 0.3 * (fiber - 18)
         + 0.3 * (age - 50) + rng.normal(0, 5, n))
    frame = pd.DataFrame({"fiber_g": fiber, "protein_g": kp / 4, "fat_g": kf / 9, "carb_g": kc / 4,
                          "kcal": kp + kf + kc + other, "age": age, "y": y})
    frame.loc[rng.random(n) < 1 / (1 + np.exp(-(-1.3 + 0.06 * (age - 50)))), "fiber_g"] = np.nan
    for column, share in (("protein_g", 0.15), ("kcal", 0.15), ("age", 0.15)):
        frame.loc[rng.random(n) < share, column] = np.nan
    return frame


def diet_state(**extra: Any) -> ProjectState:
    base = dict(target="y", task="regression", purpose="inference", roles=dict(DIET_ROLES),
                missing=MissingSpec(strategy="multiple_imputation"),
                energy_adjustment=EnergyAdjustment(method="residual", energy_column="kcal",
                                                   nutrients=["fiber_g", "protein_g", "fat_g",
                                                              "carb_g"], log_transform=True),
                exposure_forms={"fiber_g": ExposureFormSpec(form="spline", knots=4)})
    base.update(extra)
    return ProjectState(**base)


@pytest.fixture(scope="module")
def diet_run():
    """The engine's imputations of :func:`diet_frame` under the log residual and a fiber spline,
    and each copy fit by the linear family as the fit stage fits it (``pooled_table``), with each
    copy's impute step watched: every frame it was handed is kept."""
    from turbotab.core.methods import missing as mm

    frame = diet_frame()
    y = frame["y"].to_numpy()
    X = frame[list(DIET_ROLES)]
    spec = design_spec(diet_state(), X, list(DIET_ROLES))
    imputations = impute_for_inference(spec, X, y, "regression", seed=4, factors=DIET_FACTORS)
    seen: list[pd.DataFrame] = []
    original = mm.CopyGuard._check

    def watch(self: Any, frame_: pd.DataFrame) -> None:
        seen.append(frame_.copy())
        original(self, frame_)

    mm.CopyGuard._check = watch
    try:
        collected: dict[str, Any] = {}
        pipe = build_pipeline(spec, LINEAR, "regression", "inference", len(X), len(DIET_ROLES))
        table, rows, tests, _ = pooled_table(
            LINEAR, pipe, imputations, y, task="regression", clusters=INDEPENDENT,
            outcome=Outcome(name="y"), survey=None, design=None, spec=spec, rows="all",
            fit=lambda model, X_k: fit_pipeline(model, X_k, y), collect=collected)
    finally:
        mm.CopyGuard._check = original
    return {"frame": frame, "X": X, "y": y, "spec": spec, "imputations": imputations,
            "table": table, "rows": rows, "tests": tests, "fits": collected["fits"],
            "tables": collected["tables"], "seen": seen}


LOGGED = ["fiber_g", "protein_g", "fat_g", "carb_g", "kcal"]


def test_2_logged_quantities_are_imputed_on_the_log_scale(diet_run):
    """The log residual logs fiber, the sources and total energy: the imputation is SMC-FCS (the
    analysis model is nonlinear in them), the record lists each incomplete logged column drawn on
    the log scale (fiber and protein; total energy is derived from its parts), every recorded value
    is kept, and no completed copy holds a non-positive value of any logged column (pandas' minimum
    over every copy and row)."""
    imputations = diet_run["imputations"]
    assert imputations.method == "smcfcs"
    assert set(imputations.plan["logged"]) == {"fiber_g", "protein_g"}
    frame = diet_run["frame"]
    recorded = frame[LOGGED].notna().to_numpy()
    for copy in imputations.frames:
        assert copy[LOGGED].notna().all().all()
        assert float(copy[LOGGED].min().min()) > 0
        assert np.array_equal(copy[LOGGED].to_numpy()[recorded], frame[LOGGED].to_numpy()[recorded])


def test_2_a_blank_made_inside_a_copy_never_reaches_the_median_imputer(diet_run):
    """Every copy's pipeline holds no median imputer: its impute step is the guard, and the guard was
    handed no blank in any copy (the frames it saw, counted with pandas). A blank that does reach it
    (a zero a quotient normalization's log turned missing) is refused and routed to the
    detection-limit question, never filled."""
    from sklearn.impute import SimpleImputer

    from turbotab.core.methods.imputation import ImputationRefused
    from turbotab.core.methods.missing import CopyGuard

    for fitted in diet_run["fits"]:
        steps = dict(fitted.steps)
        assert isinstance(steps["impute"], CopyGuard)
        assert not any(isinstance(s, SimpleImputer) for s in steps.values())
    assert len(diet_run["seen"]) >= diet_run["imputations"].m
    assert all(int(f.isna().sum().sum()) == 0 for f in diet_run["seen"])
    guard = CopyGuard().fit(pd.DataFrame({"mz_1": [1.0, 2.0]}))
    with pytest.raises(ImputationRefused, match="detection-limit question"):
        guard.transform(pd.DataFrame({"mz_1": [1.0, np.nan]}))


def test_2_a_recorded_zero_in_a_logged_column_is_refused_not_filled():
    """Property over random tables: whenever a column the analysis logs holds a recorded zero, the
    imputation refuses (a zero a log would turn into a blank goes to the detection-limit question);
    whenever it holds none, every copy of every logged column is positive."""
    from turbotab.core.methods.imputation import ImputationRefused

    for seed in range(8):
        frame = diet_frame(seed=100 + seed, n=300)
        rng = np.random.default_rng(seed)
        zero = bool(seed % 2)
        if zero:
            rows = np.flatnonzero(frame["protein_g"].notna().to_numpy())
            frame.loc[frame.index[rng.choice(rows, 2, replace=False)], "protein_g"] = 0.0
        X = frame[list(DIET_ROLES)]
        st = diet_state(missing=MissingSpec(strategy="multiple_imputation", m=20),
                        exposure_forms={})
        spec = design_spec(st, X, list(DIET_ROLES))
        if zero:
            with pytest.raises(ImputationRefused, match="detection-limit question"):
                impute_for_inference(spec, X, frame["y"].to_numpy(), "regression", seed=seed,
                                     factors=DIET_FACTORS)
            continue
        imputations = impute_for_inference(spec, X, frame["y"].to_numpy(), "regression",
                                           seed=seed, factors=DIET_FACTORS)
        assert all(float(c[LOGGED].min().min()) > 0 for c in imputations.frames)


def test_3_the_energy_sources_and_other_are_imputed_and_energy_is_their_sum(diet_run):
    """In every copy, by NumPy arithmetic on the copy's own columns: total energy is the recorded
    value wherever it was recorded and 4·protein + 9·fat + 4·carbohydrate + other wherever it was
    blank, and other = E − Σ is never negative on any row."""
    frame, imputations = diet_run["frame"], diet_run["imputations"]
    assert imputations.plan["identity"]["energy"] == "kcal"
    assert imputations.plan["identity"]["sources"] == ["protein_g", "fat_g", "carb_g"]
    blank_e = frame["kcal"].isna().to_numpy()
    recorded_other = (frame["kcal"] - sum(f * frame[s] for s, f in DIET_FACTORS.items())).dropna()
    assert float(recorded_other.min()) > 0
    for copy in imputations.frames:
        sources = sum(f * copy[s].to_numpy() for s, f in DIET_FACTORS.items())
        other = copy["kcal"].to_numpy() - sources
        assert float(other.min()) >= 0.0
        assert np.array_equal(copy["kcal"].to_numpy()[~blank_e], frame["kcal"].to_numpy()[~blank_e])
    assert imputations.imputed["kcal"] == int(blank_e.sum())
    record = diet_run["table"].info["missing"]
    assert record["identity_energy"] == "kcal"
    assert record["identity_sources"] == ["protein_g", "fat_g", "carb_g"]


@needs_r
def test_4_knots_are_placed_once_on_observed_values_and_held_in_every_copy(diet_run, tmp_path):
    """The knots of the energy-adjusted fiber are Hmisc's (``rcspline.eval(knots.only = TRUE)``, 4
    knots) on its observed values, the log residual computed in R on the rows recording fiber and
    energy (``lm(log fiber ~ log kcal)``); every copy's fitted form step holds exactly those knots,
    so every copy's spline basis is the same function."""
    frame = diet_run["frame"]
    run_r(tmp_path, """
suppressMessages(library(Hmisc))
d <- read.csv("frame.csv")
ok <- !is.na(d$fiber_g) & !is.na(d$kcal)
lf <- log(d$fiber_g[ok]); le <- log(d$kcal[ok])
b <- coef(lm(lf ~ le))[2]
adj <- exp(lf - b * (le - mean(le)))
write.csv(data.frame(knot = rcspline.eval(adj, nk = 4, knots.only = TRUE)), "knots.csv",
          row.names = FALSE)
""", frame=frame)
    reference = pd.read_csv(tmp_path / "knots.csv")["knot"].to_numpy()
    fixed = diet_run["imputations"].plan["forms"]["fiber_g_adj"]["knots"]
    assert np.allclose(fixed, reference, rtol=1e-9, atol=1e-12)
    for fitted in diet_run["fits"]:
        assert np.array_equal(fitted.named_steps["form"].knots_["fiber_g_adj"], np.asarray(fixed))
    assert diet_run["table"].info["missing"]["knots"] == {"fiber_g_adj": list(fixed)}


# ── 5 · the survey design in the imputation model; ν_com = the design df ─────


def survey_frame(seed: int = 31, n: int = 800) -> pd.DataFrame:
    """Eight strata of two PSUs; weights differ by stratum; y and z carry stratum and PSU effects;
    x blank at random given z and the stratum (about 30%)."""
    rng = np.random.default_rng(seed)
    stratum = rng.integers(0, 8, n)
    psu = rng.integers(1, 3, n)
    w = (50 + 40 * stratum) * rng.uniform(0.8, 1.25, n)
    cell = rng.normal(0, 0.5, (8, 3))[stratum, psu]
    z = 0.3 * stratum + rng.normal(size=n)
    x = 0.5 * z + rng.normal(size=n)
    y = 1 + 0.8 * x + 0.4 * z + 0.2 * stratum + cell + rng.normal(0, 1, n)
    x = np.where(rng.random(n) < 1 / (1 + np.exp(-(-1.2 + 0.5 * z - 0.1 * stratum))), np.nan, x)
    return pd.DataFrame({"y": y, "x": x, "z": z, "stratum": stratum, "psu": psu, "w": w})


@pytest.fixture(scope="module")
def survey_run():
    from turbotab.core.models.survey import build_design

    frame = survey_frame()
    design = build_design(frame, frame["w"].to_numpy(), weight_column="w", strata_column="stratum",
                          psu_column="psu")
    roles = {"x": "exposure", "z": "covariate"}
    st = ProjectState(target="y", task="regression", purpose="inference", roles=roles,
                      missing=MissingSpec(strategy="multiple_imputation"))
    X = frame[["x", "z"]]
    y = frame["y"].to_numpy()
    spec = design_spec(st, X, list(roles))
    imputations = impute_for_inference(spec, X, y, "regression", seed=5, survey=design)
    pipe = build_pipeline(spec, LINEAR, "regression", "inference", len(X), 2)
    table, rows, _, _ = pooled_table(LINEAR, pipe, imputations, y, task="regression",
                                     clusters=INDEPENDENT, outcome=Outcome(name="y"), survey=design,
                                     design=None, spec=spec, rows="all",
                                     fit=lambda model, X_k: fit_pipeline(model, X_k, y))
    return {"frame": frame, "design": design, "imputations": imputations, "table": table,
            "rows": {r["feature"]: r for r in rows}}


def test_5_the_imputation_model_holds_the_strata_psu_and_weight(survey_run):
    """The imputation model's predictors include the design's strata, its PSU within stratum and
    the weight (the record names their columns; the imputation data hold one column for each), and
    the record's complete-data df is the design's, PSUs minus strata (16 − 8 = 8, counted with
    pandas)."""
    frame, imputations = survey_run["frame"], survey_run["imputations"]
    design = imputations.plan["design"]
    assert design["columns"] == ["__stratum", "__psu", "__weight"]
    assert (design["strata"], design["psu"], design["weight"]) == ("stratum", "psu", "w")
    record = survey_run["table"].info["missing"]
    assert (record["design_strata"], record["design_psu"], record["design_weight"]) == \
        ("stratum", "psu", "w")
    assert {"stratum", "psu", "w"} <= set(record["variables"])
    psus = frame.groupby(["stratum", "psu"]).ngroups
    assert record["df_com"] == psus - frame["stratum"].nunique() == 8
    said = record["sentence"].replace("`", "")
    assert "the survey strata (stratum), PSU (psu) and weight (w) were in the imputation model" in said
    assert "on the design's 8 degrees of freedom" in said


@needs_r
def test_5_pooled_coefficients_and_intervals_agree_with_r_survey_and_mitools(survey_run, tmp_path):
    """Each completed copy analyzed by R ``survey::svyglm`` (Taylor linearization, strata and PSUs
    nested, the weight), the copies combined by ``mitools::MIcombine(df.complete = degf(design))``:
    the pooled coefficient and Rubin's total variance agree with the engine's to 1e-6. The interval:
    the engine follows Barnard & Rubin (1999), ν_obs = (ν_com + 1)/(ν_com + 3) ν_com (1 − λ) with
    λ = (1 + 1/m)B/T, as ``mice::pool.scalar`` (``n = ν_com + 1, k = 1``) computes it; it agrees with
    that to 1e-4. MIcombine's own ν_obs uses Ū/(Ū + B) for 1 − λ (``mitools:::MIcombine.default``),
    so its interval differs slightly; that difference is bounded here at 1% of the standard error."""
    frame, imputations = survey_run["frame"], survey_run["imputations"]
    copies = pd.concat([f[["x", "z"]].assign(y=frame["y"], stratum=frame["stratum"], psu=frame["psu"],
                                             w=frame["w"], copy=k + 1)
                        for k, f in enumerate(imputations.frames)], ignore_index=True)
    run_r(tmp_path, """
suppressMessages({library(survey); library(mitools); library(mice)})
d <- read.csv("copies.csv")
coefs <- list(); vars <- list()
for (k in sort(unique(d$copy))) {
  des <- svydesign(ids = ~psu, strata = ~stratum, weights = ~w, nest = TRUE, data = d[d$copy == k, ])
  f <- svyglm(y ~ x + z, design = des)
  coefs[[k]] <- coef(f); vars[[k]] <- vcov(f)
}
dfc <- degf(des)
res <- MIcombine(coefs, vars, df.complete = dfc)
# MIcombine's own interval, as summary.MIresult prints it (confint() would fall to qnorm)
ci <- cbind(coef(res) - qt(0.975, res$df) * sqrt(diag(vcov(res))),
            coef(res) + qt(0.975, res$df) * sqrt(diag(vcov(res))))
Q <- do.call(rbind, coefs); U <- t(sapply(vars, diag))
br <- t(sapply(seq_len(ncol(Q)), function(j) {
  p <- pool.scalar(Q[, j], U[, j], n = dfc + 1, k = 1)
  c(p$qbar, p$t, p$df, p$qbar - qt(0.975, p$df) * sqrt(p$t), p$qbar + qt(0.975, p$df) * sqrt(p$t))
}))
write.csv(data.frame(term = names(coef(res)), est = coef(res), var = diag(vcov(res)),
                     mi_low = ci[, 1], mi_high = ci[, 2], mi_df = res$df, br_est = br[, 1], br_t = br[, 2],
                     br_df = br[, 3], br_low = br[, 4], br_high = br[, 5], dfc = dfc),
          "pooled.csv", row.names = FALSE)
""", copies=copies)
    ref = pd.read_csv(tmp_path / "pooled.csv").set_index("term")
    assert int(ref["dfc"].iloc[0]) == 8
    rows = survey_run["rows"]
    for ours, theirs in (("(intercept)", "(Intercept)"), ("x", "x"), ("z", "z")):
        row, r = rows[ours], ref.loc[theirs]
        assert row["estimate"] == pytest.approx(r["est"], abs=1e-6)
        assert row["se"] ** 2 == pytest.approx(r["var"], rel=1e-6)
        assert row["df"] == pytest.approx(r["br_df"], rel=1e-6)
        assert row["ci_low"] == pytest.approx(r["br_low"], abs=1e-4)
        assert row["ci_high"] == pytest.approx(r["br_high"], abs=1e-4)
        assert abs(row["ci_low"] - r["mi_low"]) <= 0.01 * row["se"]
        assert abs(row["ci_high"] - r["mi_high"]) <= 0.01 * row["se"]


# ── 6 · clustered rows (ruling 12) ───────────────────────────────────────────


def visits_frame(seed: int = 41, units: int = 150, visits: int = 4) -> pd.DataFrame:
    """Repeated visits: income and sex are the person's (constant within ``id``); x varies by visit
    around the person's own level (an intraclass correlation of about 0.6); y depends on x, income
    and a person effect. Blanks: income on whole persons (15%) and on single visits (10%), sex on
    whole persons (10%), x on visits (25%), at random given what is recorded."""
    rng = np.random.default_rng(seed)
    pid = np.repeat(np.arange(units), visits)
    income = np.repeat(rng.normal(50, 12, units), visits)
    sex = np.repeat(rng.integers(0, 2, units).astype(float), visits)
    level = np.repeat(rng.normal(0, 1.2, units), visits)
    x = 0.03 * (income - 50) + level + rng.normal(0, 1, len(pid))
    y = 2 + 0.7 * x + 0.04 * income + 0.5 * sex + np.repeat(rng.normal(0, 1, units), visits) \
        + rng.normal(0, 1, len(pid))
    frame = pd.DataFrame({"id": pid, "x": x, "income": income, "sex": sex, "y": y})
    gone = np.repeat(rng.random(units) < 0.15, visits) | (rng.random(len(pid)) < 0.10)
    frame.loc[gone, "income"] = np.nan
    frame.loc[np.repeat(rng.random(units) < 0.10, visits), "sex"] = np.nan
    frame.loc[rng.random(len(pid)) < 0.25, "x"] = np.nan
    return frame


def icc(values: np.ndarray, groups: np.ndarray) -> float:
    """The one-way ANOVA estimator of the intraclass correlation, balanced groups of size k:
    (MSB − MSW) / (MSB + (k − 1) MSW)."""
    frame = pd.DataFrame({"v": values, "g": groups})
    k = frame.groupby("g").size().iloc[0]
    means = frame.groupby("g")["v"].transform("mean")
    g = frame["g"].nunique()
    msw = float(((frame["v"] - means) ** 2).sum() / (len(frame) - g))
    msb = float(k * ((frame.groupby("g")["v"].mean() - frame["v"].mean()) ** 2).sum() / (g - 1))
    return (msb - msw) / (msb + (k - 1) * msw)


def clustered_imputations(frame: pd.DataFrame, levels: str, seed: int = 6) -> Any:
    from turbotab.core.models.inference import Clusters

    roles = {"x": "exposure", "income": "covariate", "sex": "covariate"}
    st = ProjectState(target="y", task="regression", purpose="inference", roles=roles,
                      missing=MissingSpec(strategy="multiple_imputation", m=20,
                                          imputation_levels=levels))
    X = frame[list(roles)]
    spec = design_spec(st, X, list(roles))
    codes = pd.factorize(frame["id"])[0]
    clusters = Clusters(column="id", codes=codes, n_clusters=int(codes.max()) + 1)
    return impute_for_inference(spec, X, frame["y"].to_numpy(), "regression", seed=seed,
                                clusters=clusters if levels == "clustered" else None), spec


def test_6_time_invariant_variables_are_imputed_once_per_unit():
    """With rows clustered by ``id``: income and sex, constant within every person wherever recorded
    (pandas: one value per person), are imputed once per person, so in every copy each person has
    one income and one sex, the recorded one where any visit records it; x, which varies by visit,
    is imputed row by row with the person means of the others. The record and its sentence say so."""
    frame = visits_frame()
    imputations, spec = clustered_imputations(frame, "clustered")
    assert imputations.plan["unit"] == "id"
    assert sorted(imputations.plan["unit_level"]) == ["income", "sex"]
    recorded = frame.groupby("id")["income"].first()
    for copy in imputations.frames:
        per = copy.assign(id=frame["id"]).groupby("id")
        assert int(per["income"].nunique().max()) == 1 and int(per["sex"].nunique().max()) == 1
        assert np.allclose(per["income"].first()[recorded.notna()], recorded.dropna())
        assert set(np.unique(copy["sex"])) <= {0.0, 1.0}
    from turbotab.core.methods.missing import multiple_imputation_info

    record = multiple_imputation_info(imputations, spec, len(frame))
    assert ("; with rows clustered by `id`, `income` and `sex` were imputed once per `id` and each "
            "row-level variable with the `id` means of the others and its own mean over the `id`'s "
            "other rows") in record["sentence"]


def test_6_clustered_imputation_keeps_the_intraclass_correlation_single_level_erases():
    """Simulation truth: the intraclass correlation of x before its blanks (0.558). Imputed with
    the person means of the other variables and its own mean over the person's other visits, the
    completed copies keep it (within 0.03; ICC by the one-way ANOVA estimator, averaged over the
    copies); imputed one row at a time it falls (to about 0.42 here; Lüdtke, Robitzsch & Grund 2017:
    "substantial negative bias in estimates of intraclass correlations")."""
    full = visits_frame()
    complete = visits_frame_complete()  # the same draws, before any blank
    truth = icc(complete["x"].to_numpy(), complete["id"].to_numpy())
    clustered, _ = clustered_imputations(full, "clustered")
    single, _ = clustered_imputations(full, "single_level")
    got_c = np.mean([icc(c["x"].to_numpy(), full["id"].to_numpy()) for c in clustered.frames])
    got_s = np.mean([icc(c["x"].to_numpy(), full["id"].to_numpy()) for c in single.frames])
    assert abs(got_c - truth) < 0.03
    assert truth - got_s > 0.1


def visits_frame_complete(seed: int = 41, units: int = 150, visits: int = 4) -> pd.DataFrame:
    """:func:`visits_frame` before its blanks (the same draws)."""
    rng = np.random.default_rng(seed)
    pid = np.repeat(np.arange(units), visits)
    income = np.repeat(rng.normal(50, 12, units), visits)
    rng.integers(0, 2, units)
    level = np.repeat(rng.normal(0, 1.2, units), visits)
    x = 0.03 * (income - 50) + level + rng.normal(0, 1, len(pid))
    return pd.DataFrame({"id": pid, "x": x})


def test_6_single_level_imputation_on_clustered_rows_is_blocked_and_recorded():
    """Under inference, single-level MI on rows a unit repeats is refused at the answer (its exits:
    the clustered imputation first, complete cases, the recorded attestation) and held at the table;
    the attestation keeps it, and its sentence carries the limitation."""
    from turbotab.core.decisions import GrainSpec
    from turbotab.core.methods.missing import SINGLE_LEVEL_CAUTION, missing_block
    from turbotab.core.voice import sentence_for

    st = ProjectState(target="y", purpose="inference",
                      roles={"x": "exposure", "income": "covariate"},
                      grain=GrainSpec(grain="repeated", id_column="id"), unit="row")
    answer = SetMissing(strategy="multiple_imputation", imputation_levels="single_level")
    with pytest.raises(Refusal) as refused:
        validate(answer, {"state": st})
    assert refused.value.code == "single_level_imputation_on_clustered_rows"
    exits = [e["decision"] for e in refused.value.exits]
    assert exits[0]["imputation_levels"] == "clustered"
    assert exits[1]["strategy"] == "complete_case" and exits[2]["acknowledged"] is True
    validate(exits[2], {"state": st})
    said = sentence_for(exits[2], st)
    assert SINGLE_LEVEL_CAUTION in said
    held = missing_block(answer.model_dump(), "inference", clustered=True)
    assert held is not None and SINGLE_LEVEL_CAUTION in held[0]
    assert [e["decision"]["imputation_levels"] for e in held[1][:1]] == ["clustered"]
    assert missing_block({**answer.model_dump(), "acknowledged": True}, "inference",
                         clustered=True) is None


# ── 7 · m by the rule; the Monte Carlo error in the record ───────────────────


def test_7_m_is_at_least_20_and_at_least_the_percentage_of_incomplete_rows(diet_run):
    """m = max(20, ⌈the percentage of rows with any imputed value⌉), counted with pandas over the
    imputed columns; an answer asking for more is kept; the record states the percentage."""
    frame = diet_run["frame"]
    pct = 100 * frame[list(DIET_ROLES)].isna().any(axis=1).mean()
    imputations = diet_run["imputations"]
    assert imputations.m == max(20, math.ceil(pct)) > 20
    record = diet_run["table"].info["missing"]
    assert record["m"] == imputations.m and record["percent_incomplete"] == round(pct, 1)
    X = frame[list(DIET_ROLES)]
    spec = design_spec(diet_state(missing=MissingSpec(strategy="multiple_imputation", m=80)), X,
                       list(DIET_ROLES))
    from turbotab.core.methods.missing import imputation_plan

    assert imputation_plan(spec, X, frame["y"].to_numpy(), "regression",
                           factors=DIET_FACTORS).m == 80


def test_7_the_monte_carlo_error_is_reported_in_the_record(diet_run):
    """Each pooled row carries its Monte Carlo error √(B/m) (White, Royston & Wood 2011, §7), and the
    record the largest as a share of its standard error, both checked against NumPy over the copies'
    own estimates; the methods sentence states it."""
    tables, rows = diet_run["tables"], {r["feature"]: r for r in diet_run["rows"]}
    m = len(tables)
    worst, feature = 0.0, None
    for name, row in rows.items():
        per_copy = [next(r["estimate"] for r in t.rows if r["feature"] == name) for t in tables]
        mc = float(np.std(per_copy, ddof=1) / math.sqrt(m))
        assert row["mc_se"] == pytest.approx(mc, rel=1e-9)
        if not name.startswith("(") and mc / row["se"] > worst:
            worst, feature = mc / row["se"], name
    record = diet_run["table"].info["missing"]
    assert record["mc_max_ratio"] == pytest.approx(worst, rel=1e-9)
    assert record["mc_feature"] == feature
    assert (f"the largest Monte Carlo error, for `{feature}`, was {100 * worst:.1f}% of its "
            f"standard error") in record["sentence"]


# ── 8 · D1 for multi-df tests, against R mitml ───────────────────────────────

HC3_R = """
hc3 <- function(fit) {
  X <- model.matrix(fit); e <- resid(fit); h <- hatvalues(fit)
  B <- solve(crossprod(X))
  B %*% crossprod(X * (e / (1 - h))) %*% B
}
"""


def form_run(form: dict[str, Any], seed: int = 8) -> dict[str, Any]:
    """:func:`spline_table`'s x in ``form``, the engine's imputations (m by the rule) and its pooled
    form tests, with the copies."""
    frame = spline_table()
    y = frame["y"].to_numpy()
    roles = {"x": "exposure", "z": "covariate"}
    st = ProjectState(target="y", task="regression", purpose="inference", roles=roles,
                      missing=MissingSpec(strategy="multiple_imputation"),
                      exposure_forms={"x": ExposureFormSpec(**form)})
    X = frame[["x", "z"]]
    spec = design_spec(st, X, list(roles))
    imputations = impute_for_inference(spec, X, y, "regression", seed=seed)
    pipe = build_pipeline(spec, LINEAR, "regression", "inference", len(X), 2)
    table, rows, tests, _ = pooled_table(LINEAR, pipe, imputations, y, task="regression",
                                         clusters=INDEPENDENT, outcome=Outcome(name="y"),
                                         survey=None, design=None, spec=spec, rows="all",
                                         fit=lambda model, X_k: fit_pipeline(model, X_k, y))
    copies = pd.concat([f[["x", "z"]].assign(y=y, copy=k + 1)
                        for k, f in enumerate(imputations.frames)], ignore_index=True)
    return {"imputations": imputations, "tests": {t["test"]: t for t in tests}, "copies": copies,
            "table": table}


def mitml_d1(folder: Path, copies: pd.DataFrame, terms: str, model: str,
             constraints: dict[str, list[str]], df_com: str = "NULL") -> dict[str, dict[str, float]]:
    """R: each copy fit by ``lm(model)`` on its own columns (``terms`` builds them), its HC3
    covariance by hand, then ``mitml::testConstraints(method = "D1")`` for each named set (a
    constraint is an expression tested against zero, mitml's form)."""
    sets = ",\n  ".join(f'{name} = c({", ".join(repr(c) for c in cs)})'
                        for name, cs in constraints.items())
    run_r(folder, f"""
suppressMessages({{library(mitml); library(Hmisc)}})
{HC3_R}
d <- read.csv("copies.csv")
Q <- NULL; U <- NULL
for (k in sort(unique(d$copy))) {{
  dk <- d[d$copy == k, ]
  {terms}
  fit <- lm({model}, data = dk)
  Q <- cbind(Q, coef(fit)); U <- if (is.null(U)) array(hc3(fit), c(dim(hc3(fit)), 1)) else
    array(c(U, hc3(fit)), c(dim(hc3(fit)), dim(U)[3] + 1))
}}
rownames(Q) <- gsub("[()]", "", names(coef(fit)))
dimnames(U) <- list(rownames(Q), rownames(Q), NULL)
sets <- list(
  {sets})
out <- do.call(rbind, lapply(names(sets), function(s) {{
  t <- testConstraints(qhat = Q, uhat = U, constraints = sets[[s]], method = "D1",
                       df.com = {df_com})$test
  data.frame(set = s, F = t[1, "F.value"], df1 = t[1, "df1"], df2 = t[1, "df2"], p = t[1, "P(>F)"])
}}))
write.csv(out, "d1.csv", row.names = FALSE)
""", copies=copies)
    out = pd.read_csv(folder / "d1.csv").set_index("set")
    return {s: out.loc[s].to_dict() for s in out.index}


def assert_d1(ours: dict[str, Any], theirs: dict[str, float]) -> None:
    assert ours["distribution"] == "F" and ours["df_num"] == int(theirs["df1"])
    assert ours["statistic"] == pytest.approx(theirs["F"], rel=1e-6)
    assert ours["df_den"] == pytest.approx(theirs["df2"], rel=1e-6)
    assert ours["p"] == pytest.approx(theirs["p"], rel=1e-6, abs=1e-12)


@needs_r
def test_8_spline_overall_and_nonlinearity_tests_agree_with_mitml_d1(tmp_path):
    """The engine's pooled tests of a 4-knot spline (all terms; the nonlinear terms) against R:
    every completed copy refit by ``lm`` on its own basis (``Hmisc::rcspline.eval`` at the knots the
    engine fixed, ``inclx = TRUE``) with its HC3 covariance written out, pooled by
    ``mitml::testConstraints(method = "D1")``: F, its denominator df and p to 1e-6."""
    run = form_run({"form": "spline", "knots": 4})
    knots = ", ".join(repr(float(k)) for k in run["imputations"].plan["forms"]["x"]["knots"])
    found = mitml_d1(tmp_path, run["copies"],
                     f"b <- rcspline.eval(dk$x, knots = c({knots}), inclx = TRUE); "
                     f"dk$s0 <- b[, 1]; dk$s1 <- b[, 2]; dk$s2 <- b[, 3]",
                     "y ~ s0 + s1 + s2 + z",
                     {"overall": ["s0", "s1", "s2"], "nonlinear": ["s1", "s2"]})
    assert_d1(run["tests"]["overall"], found["overall"])
    assert_d1(run["tests"]["nonlinear"], found["nonlinear"])


@needs_r
def test_8_the_global_quintile_test_agrees_with_mitml_d1(tmp_path):
    """Quintiles of x at the cut points the engine fixed on the observed values: the pooled global
    test (all four indicators) against R's copies refit with ``cut`` at the same points, HC3 by
    hand, ``testConstraints(method = "D1")``."""
    run = form_run({"form": "quintiles"})
    cuts = ", ".join(repr(float(c)) for c in run["imputations"].plan["forms"]["x"]["cuts"])
    found = mitml_d1(tmp_path, run["copies"],
                     f"g <- cut(dk$x, c(-Inf, {cuts}, Inf), right = TRUE); "
                     f"for (j in 2:5) dk[[paste0('q', j)]] <- as.numeric(g == levels(g)[j])",
                     "y ~ q2 + q3 + q4 + q5 + z",
                     {"global": ["q2", "q3", "q4", "q5"]})
    assert_d1(run["tests"]["global"], found["global"])


@needs_r
def test_8_an_interaction_test_agrees_with_mitml_d1(tmp_path):
    """A product term's multi-df test (x·z and z·w together) pooled by ``pooled_wald`` over copies
    fit here by NumPy OLS with the HC3 covariance written out, against R refitting the same copies
    and pooling with ``testConstraints(method = "D1")``. (Declared interactions reach the engine with
    M3.5's effect modification; the pooling is checked on its own.)"""
    rng = np.random.default_rng(81)
    n = 500
    z, w = rng.normal(size=n), (rng.random(n) < 0.5).astype(float)
    x = 0.4 * z + rng.normal(size=n)
    y = 1 + 0.5 * x + 0.3 * z + 0.2 * w + 0.4 * x * z - 0.3 * z * w + rng.normal(size=n)
    x = np.where(rng.random(n) < 0.3, np.nan, x)

    def design(f: pd.DataFrame) -> np.ndarray:
        xv, zv, wv = (f[c].to_numpy(dtype=float) for c in ("x", "z", "w"))
        return np.column_stack([xv, zv, wv, xv * zv, zv * wv])

    imputations = impute(pd.DataFrame({"x": x, "z": z, "w": w}), [Variable("x", "numeric")],
                         mode="smcfcs", substantive=Substantive(SMOutcome("linear", y=y), design),
                         m=30, seed=8)
    Q, U = [], []
    for f in imputations.frames:
        X = np.column_stack([np.ones(n), design(f)])
        inv = np.linalg.inv(X.T @ X)
        beta = inv @ X.T @ y
        e = y - X @ beta
        h = np.einsum("ij,jk,ik->i", X, inv, X)
        cov = inv @ (X * (e / (1 - h))[:, None]).T @ (X * (e / (1 - h))[:, None]) @ inv
        Q.append(beta[4:6])
        U.append(cov[4:6, 4:6])
    ours = pooled_wald(np.asarray(Q), np.asarray(U))
    copies = pd.concat([f.assign(y=y, copy=k + 1) for k, f in enumerate(imputations.frames)],
                       ignore_index=True)
    found = mitml_d1(tmp_path, copies, "dk$xz <- dk$x * dk$z; dk$zw <- dk$z * dk$w",
                     "y ~ x + z + w + xz + zw", {"interaction": ["xz", "zw"]})
    assert_d1({**ours, "distribution": "F"}, found["interaction"])


@needs_r
def test_8_d1_on_the_design_df_takes_reiters_denominator_df_as_mitml_does(tmp_path):
    """Under a survey design a spline's pooled tests take D1's denominator df from the design's
    ν_com by Reiter (2007), as ``mitml`` does with ``df.com``. The engine (each copy design-based,
    Taylor-linearized) against R: every copy refit by ``survey::svyglm`` on the basis
    ``Hmisc::rcspline.eval`` makes at the engine's fixed knots, its ``vcov``, pooled by
    ``testConstraints(method = "D1", df.com = degf(design))``: F, df and p to 1e-6."""
    from turbotab.core.models.survey import build_design

    frame = survey_frame(seed=32)
    design = build_design(frame, frame["w"].to_numpy(), weight_column="w", strata_column="stratum",
                          psu_column="psu")
    roles = {"x": "exposure", "z": "covariate"}
    st = ProjectState(target="y", task="regression", purpose="inference", roles=roles,
                      missing=MissingSpec(strategy="multiple_imputation"),
                      exposure_forms={"x": ExposureFormSpec(form="spline", knots=3)})
    X, y = frame[["x", "z"]], frame["y"].to_numpy()
    spec = design_spec(st, X, list(roles))
    imputations = impute_for_inference(spec, X, y, "regression", seed=9, survey=design)
    pipe = build_pipeline(spec, LINEAR, "regression", "inference", len(X), 2)
    table, _, tests, _ = pooled_table(LINEAR, pipe, imputations, y, task="regression",
                                      clusters=INDEPENDENT, outcome=Outcome(name="y"),
                                      survey=design, design=None, spec=spec, rows="all",
                                      fit=lambda model, X_k: fit_pipeline(model, X_k, y))
    tests = {t["test"]: t for t in tests}
    assert table.info["missing"]["df_com"] == 8 and "Reiter" in tests["overall"]["caption"]
    knots = ", ".join(repr(float(k)) for k in imputations.plan["forms"]["x"]["knots"])
    copies = pd.concat([f[["x", "z"]].assign(y=y, stratum=frame["stratum"], psu=frame["psu"],
                                             w=frame["w"], copy=k + 1)
                        for k, f in enumerate(imputations.frames)], ignore_index=True)
    run_r(tmp_path, f"""
suppressMessages({{library(survey); library(mitml); library(Hmisc)}})
d <- read.csv("copies.csv")
Q <- NULL; U <- list()
for (k in sort(unique(d$copy))) {{
  dk <- d[d$copy == k, ]
  b <- rcspline.eval(dk$x, knots = c({knots}), inclx = TRUE)
  dk$s0 <- b[, 1]; dk$s1 <- b[, 2]
  des <- svydesign(ids = ~psu, strata = ~stratum, weights = ~w, nest = TRUE, data = dk)
  fit <- svyglm(y ~ s0 + s1 + z, design = des)
  Q <- cbind(Q, coef(fit)); U[[k]] <- vcov(fit)
}}
rownames(Q) <- gsub("[()]", "", rownames(Q))
U <- array(unlist(U), c(nrow(Q), nrow(Q), ncol(Q)), dimnames = list(rownames(Q), rownames(Q), NULL))
out <- do.call(rbind, lapply(list(overall = c("s0", "s1"), nonlinear = c("s1")), function(cs) {{
  t <- testConstraints(qhat = Q, uhat = U, constraints = cs, method = "D1", df.com = degf(des))$test
  data.frame(F = t[1, "F.value"], df1 = t[1, "df1"], df2 = t[1, "df2"], p = t[1, "P(>F)"])
}}))
out$set <- c("overall", "nonlinear")
write.csv(out, "d1.csv", row.names = FALSE)
""", copies=copies)
    theirs = pd.read_csv(tmp_path / "d1.csv").set_index("set")
    assert_d1(tests["overall"], theirs.loc["overall"].to_dict())
    nonlinear = tests["nonlinear"]  # one term: still D1 (k = 1), F on Reiter's df
    assert_d1(nonlinear, theirs.loc["nonlinear"].to_dict())


# ── 9 · every estimate under MI is pooled: the substitution curve ────────────

SOURCES = {"protein_g": 4.0, "carb_g": 4.0, "fat_g": 9.0}


def energy_frame(seed: int, n: int, binary: bool = False) -> pd.DataFrame:
    """Three sources and other energy (kcal); y depends on kcal from each and on age; blanks in
    protein (15%), total energy (10%) and age (20%)."""
    rng = np.random.default_rng(seed)
    age = rng.normal(50, 12, n)
    size = 2100 + 450 * rng.normal(size=n) - 9 * (age - 50)
    kp = size * np.clip(rng.normal(0.16, 0.03, n), 0.05, 0.4)
    kf = size * np.clip(rng.normal(0.34, 0.05, n), 0.1, 0.6)
    kc = size * np.clip(rng.normal(0.42, 0.05, n), 0.1, 0.7)
    other = size * rng.uniform(0.02, 0.1, n)
    eta = 0.010 * kp + 0.004 * kc + 0.002 * other + 0.2 * (age - 50)
    if binary:
        y = (rng.random(n) < 1 / (1 + np.exp(-(eta - eta.mean()) / 4))).astype(int)
    else:
        y = eta + rng.normal(0, 5, n)
    frame = pd.DataFrame({"protein_g": kp / 4, "carb_g": kc / 4, "fat_g": kf / 9,
                          "kcal": kp + kf + kc + other, "age": age, "y": y})
    for column, share in (("protein_g", 0.15), ("kcal", 0.10), ("age", 0.20)):
        frame.loc[rng.random(n) < share, column] = np.nan
    return frame


def substitution_run(folder: Path, frame: pd.DataFrame, task: str, method: str,
                     n_boot: int = 0) -> dict[str, Any]:
    """The design, fit and substitution stages under inference with multiple imputation (100 kcal
    from carbohydrate to protein), as the engine runs them."""
    from turbotab.core.stages.modeling import design_stage, fit_stage, substitution_stage
    from turbotab.core.tests import modeling_fixtures as mf

    roles = {"protein_g": "exposure", "carb_g": "exposure", "fat_g": "exposure", "kcal": "energy",
             "age": "covariate"}
    paths = mf.ingest_frame(frame, folder)
    st = mf.state(roles=roles, target="y", task=task, models=["linear"], purpose="inference",
                  energy_adjustment=EnergyAdjustment(method=method, energy_column="kcal",
                                                     nutrients=list(SOURCES)),
                  missing=MissingSpec(strategy="multiple_imputation"),
                  substitution=SubstitutionSpec(donor="carb_g", recipient="protein_g",
                                                step_kcal=100, n_boot=n_boot),
                  split=SplitSpec(holdout=0.0, seed=0, folds=5),
                  column_units=mf.grams(*SOURCES))
    split = mf.split_bundle(np.arange(len(frame)), holdout=0.0)
    ti = mf.target_info(task, "y")
    design = design_stage(mf.context(st, {"split": split, "target_info": ti}, paths))
    fit = fit_stage(mf.context(st, {"design": design, "split": split, "target_info": ti}, paths))
    sub = substitution_stage(mf.context(st, {"design": design, "fit": fit}, paths))
    return {"fit": fit, "sub": sub, "copies": fit.objects["imputations"]["frames"]}


@pytest.fixture(scope="module")
def contrast_run(tmp_path_factory):
    frame = energy_frame(seed=91, n=1000)
    return {"frame": frame, **substitution_run(tmp_path_factory.mktemp("contrast"), frame,
                                               "regression", "all_components")}


def test_9_a_linear_all_components_curve_is_the_exact_contrast_of_the_pooled_coefficients(contrast_run):
    """Linear all-components model: moving k kcal from carbohydrate to protein changes the model
    matrix by +k on kcal from protein and −k on kcal from carbohydrate on every row, so the pooled
    curve is k cᵀQ̄ with Rubin's variance k² cᵀTc. Reference: every completed copy refit with NumPy
    (OLS on [1, 4·protein, 4·carbohydrate, 9·fat, kcal − Σ, age], its HC3 covariance written out),
    the contrast pooled by Rubin's rules with Barnard–Rubin df (ν_com = n − 6), and the same numbers
    from the pooled coefficient vector and its pooled covariance T = Ū + (1 + 1/m)B; the curve, its
    interval and its df agree to 1e-8."""
    frame, sub = contrast_run["frame"], contrast_run["sub"]
    y = frame["y"].to_numpy()
    n = len(y)
    c = np.array([0, 1, -1, 0, 0, 0], dtype=float)  # per kcal moved: protein +1, carbohydrate −1
    betas, covs = [], []
    for copy in contrast_run["copies"]:
        kcal = {s: f * copy[s].to_numpy() for s, f in SOURCES.items()}
        X = np.column_stack([np.ones(n), kcal["protein_g"], kcal["carb_g"], kcal["fat_g"],
                             copy["kcal"].to_numpy() - sum(kcal.values()), copy["age"].to_numpy()])
        inv = np.linalg.inv(X.T @ X)
        beta = inv @ X.T @ y
        e = y - X @ beta
        h = np.einsum("ij,jk,ik->i", X, inv, X)
        meat = (X * (e / (1 - h))[:, None]).T @ (X * (e / (1 - h))[:, None])
        betas.append(beta)
        covs.append(inv @ meat @ inv)
    betas, covs = np.asarray(betas), np.asarray(covs)
    m = len(betas)
    q = betas @ c
    u = np.einsum("i,kij,j->k", c, covs, c)
    qbar, ubar, b = q.mean(), u.mean(), q.var(ddof=1)
    t = ubar + (1 + 1 / m) * b
    T_full = covs.mean(axis=0) + (1 + 1 / m) * np.cov(betas, rowvar=False)
    assert c @ betas.mean(axis=0) == pytest.approx(qbar, rel=1e-12)
    assert c @ T_full @ c == pytest.approx(t, rel=1e-10)
    lam = (1 + 1 / m) * b / t
    nu_com = n - 6
    nu_old = (m - 1) / lam ** 2
    nu_obs = (nu_com + 1) / (nu_com + 3) * nu_com * (1 - lam)
    nu = nu_old * nu_obs / (nu_old + nu_obs)
    model = sub["models"][0]
    assert model["pooled"] == "contrast"
    live = [i for i, d in enumerate(model["delta"]) if d is not None and sub["ks"][i] > 0]
    assert live
    for i in live:
        k = sub["ks"][i]
        half = stats.t.ppf(0.975, nu) * math.sqrt(t) * k
        assert model["delta"][i] == pytest.approx(qbar * k, rel=1e-8)
        assert model["ci_low"][i] == pytest.approx(qbar * k - half, rel=1e-8)
        assert model["ci_high"][i] == pytest.approx(qbar * k + half, rel=1e-8)
        assert model["df"][i] == pytest.approx(nu, rel=1e-8)


def test_9_the_single_fill_curve_is_gone(contrast_run):
    """Under multiple imputation the substitution note says how the curve was pooled over the copies
    and never that it follows one fill."""
    note = contrast_run["sub"]["note"]
    m = len(contrast_run["copies"])
    assert (f"The curve is pooled over the {m} imputations: for the linear all-components model the "
            f"curve is the exact contrast of the pooled coefficients, with Rubin's total variance and "
            f"Barnard–Rubin degrees of freedom.") in note
    assert "not pooled over the multiple imputations" not in note
    assert "filled once" not in note


def numpy_curve(copy: pd.DataFrame, beta: np.ndarray, ks: list[float]) -> tuple[list, list]:
    """One copy's curve written out with NumPy: each k's mean change in the predicted probability
    over the rows on support (the moved intakes and their shares of total energy within the
    copy's observed ranges, the intakes not negative), stopping at the first k with under half the
    rows; and the fixed population's curve (the rows on support at every k reached)."""
    p, c, e = (copy[s].to_numpy() for s in ("protein_g", "carb_g", "kcal"))
    X0 = np.column_stack([np.ones(len(copy)), copy[["protein_g", "carb_g", "fat_g", "kcal",
                                                     "age"]].to_numpy()])
    base = 1 / (1 + np.exp(-(X0 @ beta)))
    ranges = {"p": (p.min(), p.max()), "c": (c.min(), c.max())}
    shares = {"p": ((4 * p / e).min(), (4 * p / e).max()), "c": ((4 * c / e).min(), (4 * c / e).max())}
    masks, diffs, stop = [], [], None
    for k in ks:
        new_p, new_c = p + k / 4, c - k / 4
        ok = ((new_p >= ranges["p"][0]) & (new_p <= ranges["p"][1]) & (new_p >= 0)
              & (new_c >= ranges["c"][0]) & (new_c <= ranges["c"][1]) & (new_c >= 0)
              & (4 * new_p / e >= shares["p"][0]) & (4 * new_p / e <= shares["p"][1])
              & (4 * new_c / e >= shares["c"][0]) & (4 * new_c / e <= shares["c"][1]))
        if stop is None and ok.mean() < 0.5:
            stop = k
        X1 = X0.copy()
        X1[:, 1], X1[:, 2] = new_p, new_c
        masks.append(ok)
        diffs.append(1 / (1 + np.exp(-(X1 @ beta))) - base)
    live = [stop is None or k < stop for k in ks]
    fixed = np.all([mk for mk, lv in zip(masks, live) if lv], axis=0)
    curve = [float(d[mk].mean()) if lv else None for d, mk, lv in zip(diffs, masks, live)]
    fixed_curve = [float(d[fixed].mean()) if lv else None for d, lv in zip(diffs, live)]
    return curve, fixed_curve


def test_9_a_nonlinear_curve_is_each_copys_curve_pooled_at_each_k(tmp_path):
    """A logistic model (the standard energy model): the curve is not linear in the coefficients, so
    each copy's curve is computed on that copy's own rows and fit, and pooled at each k. Reference:
    every copy refit by statsmodels' unpenalized logistic MLE, its curve and its fixed-population
    curve written out with NumPy (:func:`numpy_curve`), averaged over the copies; agreement to 1e-5
    (scikit-learn's Newton–Cholesky stops at its default gradient tolerance). With a band, each k's
    interval is Rubin's over each copy's bootstrap variance (the refits split over the copies), on
    Barnard–Rubin df, and contains the pooled curve."""
    import statsmodels.api as sm

    frame = energy_frame(seed=92, n=700, binary=True)
    run = substitution_run(tmp_path, frame, "binary", "standard", n_boot=40)
    sub = run["sub"]
    model = sub["models"][0]
    assert model["pooled"] == "per_k"
    y = frame["y"].to_numpy()
    curves, fixed = [], []
    for copy in run["copies"]:
        X = sm.add_constant(copy[["protein_g", "carb_g", "fat_g", "kcal", "age"]].to_numpy())
        beta = sm.Logit(y, X).fit(disp=0, method="newton", tol=1e-12, maxiter=200).params
        curve, fixed_curve = numpy_curve(copy, beta, sub["ks"])
        curves.append(curve)
        fixed.append(fixed_curve)
    for i, k in enumerate(sub["ks"]):
        if model["delta"][i] is None or k == 0:
            continue
        assert model["delta"][i] == pytest.approx(np.mean([c_[i] for c_ in curves]), rel=1e-5)
        assert model["fixed_delta"][i] == pytest.approx(np.mean([f[i] for f in fixed]), rel=1e-5)
        assert model["df"][i] is not None and model["df"][i] > 0
        assert model["ci_low"][i] < model["delta"][i] < model["ci_high"][i]
    assert sub["band"]["interval"] == "normal" and "Schomaker & Heumann 2018" in sub["band"]["caption"]


# ── 10 · the data's own imputed copies (NHANES DXA) ──────────────────────────


def dxa_frame(seed: int = 1999, people: int = 240) -> pd.DataFrame:
    """NHANES DXA's shape (as WP18's fixture): each SEQN five times, ``_MULT_`` 1–5; a third of the
    people have an imputed percent fat that differs between their copies; age and fiber are the
    same in all five."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(people):
        age = round(float(rng.normal(45, 12)), 1)
        fiber = round(float(rng.gamma(4, 4)), 1)
        fat = 30 + 0.1 * (age - 45) - 0.2 * (fiber - 16) + float(rng.normal(0, 4))
        imputed = rng.random() < 1 / 3
        for k in range(1, 6):
            value = fat + (float(rng.normal(0, 2)) if imputed else 0.0)
            rows.append({"SEQN": 30000 + i, "_MULT_": k, "RIDAGEYR": age, "fiber_g": fiber,
                         "DXDTOPF": round(value, 1)})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def dxa_run(tmp_path_factory):
    """The design and fit stages under inference, the copies kept as records (``set_unit`` row)."""
    from turbotab.core.decisions import GrainSpec, RepeatSpec
    from turbotab.core.stages.modeling import design_stage, fit_stage
    from turbotab.core.tests import modeling_fixtures as mf

    frame = dxa_frame()
    paths = mf.ingest_frame(frame, tmp_path_factory.mktemp("dxa"))
    st = mf.state(roles={"RIDAGEYR": "covariate", "fiber_g": "exposure", "SEQN": "identifier"},
                  target="DXDTOPF", task="regression", models=["linear"], purpose="inference",
                  lens=["clinical"], grain=GrainSpec(grain="repeated", id_column="SEQN"),
                  repeat_kind=RepeatSpec(repeat_kind="imputed_copies", implicate_column="_MULT_"),
                  unit="row", split=SplitSpec(holdout=0.0, seed=0, folds=5),
                  shape_confirmations={"code_or_count:_MULT_": "code"})
    split = mf.split_bundle(np.arange(len(frame)), holdout=0.0, groups=frame["SEQN"].to_numpy(),
                            grouped_by="SEQN")
    ti = mf.target_info("regression", "DXDTOPF")
    design = design_stage(mf.context(st, {"split": split, "target_info": ti}, paths))
    fit = fit_stage(mf.context(st, {"design": design, "split": split, "target_info": ti}, paths))
    return {"frame": frame, "fit": fit, "model": fit.data["models"][0]}


def test_10_each_imputed_copy_is_analyzed_with_its_own_outcome_and_pooled(dxa_run):
    """The table is pooled over the data's five copies, each analyzed with its own outcome vector
    (pandas: the outcome differs between copies for the imputed third); the record names the copies
    and its sentence says so verbatim."""
    frame, model = dxa_run["frame"], dxa_run["model"]
    assert (frame.groupby("SEQN")["DXDTOPF"].nunique() > 1).sum() > 0
    record = model["inference"]["missing"]
    assert record["model"] == "supplied" and record["m"] == 5
    assert record["copies"] == ["1", "2", "3", "4", "5"] and record["implicate"] == "_MULT_"
    outcomes = dxa_run["fit"].objects["imputations"]["outcomes"]
    assert len(outcomes) == 5 and not all(np.array_equal(outcomes[0], o) for o in outcomes[1:])
    feature = record["mc_feature"]
    assert record["sentence"] == (
        f"Each of the data's 5 imputed copies (numbered 1 to 5 by `_MULT_`) was analyzed as its own "
        f"completed dataset with its own outcome, and the estimates were pooled by Rubin's rules "
        f"(NCHS's combining rules); the largest Monte Carlo error, for `{feature}`, was "
        f"{100 * record['mc_max_ratio']:.1f}% of its standard error.")
    assert "imputed copies" in model["inference"]["caption"]


@needs_r
def test_10_the_pooled_table_agrees_with_r_mitools(dxa_run, tmp_path):
    """Each copy (``_MULT_`` = k) analyzed in R by ``lm`` with its HC3 covariance written out, the
    five combined by ``mitools::MIcombine(df.complete = n − 3)``: the pooled coefficients and Rubin's
    total variances agree with the engine's to 1e-6; the Barnard–Rubin df and interval
    (``mice::pool.scalar``) to 1e-4."""
    frame, model = dxa_run["frame"], dxa_run["model"]
    run_r(tmp_path, HC3_R + """
suppressMessages({library(mitools); library(mice)})
d <- read.csv("frame.csv")
coefs <- list(); vars <- list()
for (k in 1:5) {
  fit <- lm(DXDTOPF ~ RIDAGEYR + fiber_g, data = d[d$copy == k, ])
  coefs[[k]] <- coef(fit); vars[[k]] <- hc3(fit)
}
dfc <- df.residual(fit)
res <- MIcombine(coefs, vars, df.complete = dfc)
Q <- do.call(rbind, coefs); U <- t(sapply(vars, diag))
br <- t(sapply(seq_len(ncol(Q)), function(j) {
  p <- pool.scalar(Q[, j], U[, j], n = dfc + 1, k = 1)
  c(p$df, p$qbar - qt(0.975, p$df) * sqrt(p$t), p$qbar + qt(0.975, p$df) * sqrt(p$t))
}))
write.csv(data.frame(term = names(coef(res)), est = coef(res), var = diag(vcov(res)),
                     df = br[, 1], low = br[, 2], high = br[, 3]), "pooled.csv", row.names = FALSE)
""", frame=frame.drop(columns=["_MULT_"]).assign(copy=frame["_MULT_"]))
    ref = pd.read_csv(tmp_path / "pooled.csv").set_index("term")
    rows = {r["feature"]: r for r in model["coefficients"]}
    for ours, theirs in (("(intercept)", "(Intercept)"), ("RIDAGEYR", "RIDAGEYR"),
                         ("fiber_g", "fiber_g")):
        row, r = rows[ours], ref.loc[theirs]
        assert row["estimate"] == pytest.approx(r["est"], abs=1e-6)
        assert row["se"] ** 2 == pytest.approx(r["var"], rel=1e-6)
        assert row["df"] == pytest.approx(r["df"], rel=1e-4)
        assert row["ci_low"] == pytest.approx(r["low"], abs=1e-4)
        assert row["ci_high"] == pytest.approx(r["high"], abs=1e-4)


# ── 11 · the leash: passive MI with a declared nonlinear term ────────────────


def staged_fit(folder: Path, frame: pd.DataFrame, missing: MissingSpec,
               forms: dict[str, Any] | None = None, task: str = "regression") -> dict[str, Any]:
    """The design and fit stages under inference on :func:`spline_table`-shaped data (x, z, y)."""
    from turbotab.core.stages.modeling import design_stage, fit_stage
    from turbotab.core.tests import modeling_fixtures as mf

    paths = mf.ingest_frame(frame, folder)
    st = mf.state(roles={"x": "exposure", "z": "covariate"}, target="y", task=task,
                  models=["linear"], purpose="inference", missing=missing,
                  exposure_forms={c: ExposureFormSpec(**f) for c, f in (forms or {}).items()},
                  split=SplitSpec(holdout=0.0, seed=0, folds=5))
    split = mf.split_bundle(np.arange(len(frame)), holdout=0.0)
    ti = mf.target_info(task, "y")
    design = design_stage(mf.context(st, {"split": split, "target_info": ti}, paths))
    fit = fit_stage(mf.context(st, {"design": design, "split": split, "target_info": ti}, paths))
    return {"state": st, "design": design, "fit": fit, "model": fit.data["models"][0]}


SPLINE = {"x": {"form": "spline", "knots": 4}}


def test_11_passive_imputation_with_a_spline_is_refused_at_the_answer():
    """Under inference with a spline declared, the passive answer is refused with its exits: the
    compatible imputation first, complete cases, then the recorded attestation; the attestation is
    accepted and its sentence carries the caution."""
    from turbotab.core.methods.missing import PASSIVE_CAUTION
    from turbotab.core.voice import sentence_for

    st = ProjectState(target="y", purpose="inference", roles={"x": "exposure", "z": "covariate"},
                      exposure_forms={"x": ExposureFormSpec(form="spline", knots=4)})
    answer = SetMissing(strategy="multiple_imputation", imputation_model="passive")
    with pytest.raises(Refusal) as refused:
        validate(answer, {"state": st})
    assert refused.value.code == "passive_imputation_with_nonlinear_terms"
    assert "a restricted cubic spline of `x`" in str(refused.value)
    exits = [e["decision"] for e in refused.value.exits]
    assert exits[0]["imputation_model"] == "compatible"
    assert exits[1]["strategy"] == "complete_case" and exits[2]["acknowledged"] is True
    validate(exits[2], {"state": st})
    assert PASSIVE_CAUTION in sentence_for(exits[2], st)
    assert "the nonlinearity tests would be biased toward the null" in PASSIVE_CAUTION
    validate(answer, {"state": st.model_copy(update={"exposure_forms": None})})  # no term: fine


def test_11_the_card_offers_every_imputation_model_with_its_labels():
    """§11.3: the menu is never shortened and no option is offered silently. The missing-values card
    offers multiple imputation's sub-answers under inference, soundest first, each labeled customary
    and sound separately (north star 5) with its rung: the compatible imputation recommended, the
    passive and the single-level ones blocked and recorded."""
    from turbotab.core.methods.missing import imputation_model_options

    offered = imputation_model_options("inference")
    assert [o["key"] for o in offered] == ["compatible", "passive", "single_level"]
    assert [o["rung"] for o in offered] == ["recommended", "block_and_record", "block_and_record"]
    passive = offered[1]
    assert passive["customary"].startswith("Customary") and "biased toward the null" in passive["sound"]
    assert passive["decision"] == {"strategy": "multiple_imputation", "imputation_model": "passive"}
    assert all(o["rung"] == "refused" for o in imputation_model_options("prediction"))


def test_11_passive_imputation_with_a_spline_is_blocked_at_the_table_and_kept_recorded(tmp_path):
    """Answered before the spline was declared, the passive answer holds the inference table (no
    coefficient; the reason and its exits). With the attestation the table is pooled over passive
    copies, and the record's methods sentence, verbatim, says the nonlinearity tests would be biased
    toward the null."""
    frame = spline_table()
    held = staged_fit(tmp_path / "held", frame,
                      MissingSpec(strategy="multiple_imputation", imputation_model="passive"), SPLINE)
    info = held["model"]["inference"]
    assert held["model"]["coefficients"] == []
    assert info["refused"].startswith("Under inference passive multiple imputation with a "
                                      "restricted cubic spline of `x` is blocked until it is recorded")
    assert [e["decision"].get("imputation_model") for e in info["exits"]][:1] == ["compatible"]
    kept = staged_fit(tmp_path / "kept", frame,
                      MissingSpec(strategy="multiple_imputation", imputation_model="passive",
                                  acknowledged=True), SPLINE)
    record = kept["model"]["inference"]["missing"]
    assert record["model"] == "chained_equations" and record["compatible"] is False
    knots = ", ".join(f"{k:.4g}" for k in record["knots"]["x"])
    assert record["sentence"] == (
        f"Missing values were multiply imputed by chained equations with the outcome in the "
        f"imputation model, with a restricted cubic spline of `x` derived in each copy (passive "
        f"imputation), kept as a recorded limitation: passive imputation draws each value from a "
        f"model linear in it and only then derives the declared nonlinear terms, so the curvature "
        f"and the nonlinearity tests would be biased toward the null (Bartlett et al. 2015); m = "
        f"{record['m']} imputations, at least 20 and at least the {record['percent_incomplete']:g}% "
        f"of rows with an imputed value (White, Royston & Wood 2011); the knots of `x` ({knots}) "
        f"were placed once on its observed values and held in every copy; the estimates were "
        f"pooled by Rubin's rules, multi-parameter tests by D1 (Li, Raghunathan & Rubin 1991); the "
        f"largest Monte Carlo error, for `{record['mc_feature']}`, was "
        f"{100 * record['mc_max_ratio']:.1f}% of its standard error.")


def test_11_an_outcome_with_no_compatible_imputation_holds_a_nonlinear_term_too():
    """An ordinal or multinomial outcome has no SMC-FCS here, so its copies with a declared spline
    would be passive: held the same way, the reason saying why."""
    from turbotab.core.methods.missing import PASSIVE_CAUTION, missing_block

    held = missing_block({"strategy": "multiple_imputation"}, "inference",
                         nonlinear=["a restricted cubic spline of `x`"], passive=True)
    assert held is not None and PASSIVE_CAUTION in held[0]
    assert "No imputation compatible with this outcome's model is built here" in held[0]
    assert [e["label"] for e in held[1]] == ["Complete cases, with their assumption stated",
                                            "Keep it, recorded as a limitation"]


# ── §13 contracts and the §2 relations: the chain test ───────────────────────

CHAIN_ROLES = {"SEQN": "identifier", "SDMVSTRA": "design", "SDMVPSU": "design",
               "WTMEC2YR": "design", "age": "covariate", "fiber_g": "exposure",
               "protein_g": "exposure", "fat_g": "exposure", "carb_g": "exposure", "kcal": "energy"}


def chain_frame(seed: int = 57, n: int = 900) -> pd.DataFrame:
    """NHANES-shaped: eight strata of two PSUs and their weights; :func:`diet_frame`'s diet and
    outcome, with a stratum effect; its blanks."""
    rng = np.random.default_rng(seed)
    diet = diet_frame(seed=seed, n=n)
    stratum = rng.integers(1, 9, n)
    diet["y"] = diet["y"] + 0.5 * stratum
    return diet.assign(SEQN=np.arange(n) + 90000, SDMVSTRA=stratum,
                       SDMVPSU=rng.integers(1, 3, n),
                       WTMEC2YR=(8000 + 3000 * stratum) * rng.uniform(0.7, 1.4, n))


def chain_stages(folder: Path, frame: pd.DataFrame, forms: dict[str, Any]) -> dict[str, Any]:
    from turbotab.core.decisions import SurveySpec
    from turbotab.core.stages.modeling import design_stage, fit_stage, substitution_stage
    from turbotab.core.tests import modeling_fixtures as mf

    paths = mf.ingest_frame(frame, folder)
    st = mf.state(roles=dict(CHAIN_ROLES), target="y", task="regression", models=["linear"],
                  purpose="inference",
                  energy_adjustment=EnergyAdjustment(method="residual", energy_column="kcal",
                                                     nutrients=["fiber_g", "protein_g", "fat_g",
                                                                "carb_g"], log_transform=True),
                  exposure_forms={c: ExposureFormSpec(**f) for c, f in forms.items()},
                  missing=MissingSpec(strategy="multiple_imputation"),
                  survey=SurveySpec(estimand="population", weight="WTMEC2YR", strata="SDMVSTRA",
                                    psu="SDMVPSU"),
                  substitution=SubstitutionSpec(donor="carb_g", recipient="protein_g",
                                                step_kcal=100),
                  split=SplitSpec(holdout=0.0, seed=0, folds=5),
                  column_units=mf.grams("protein_g", "fat_g", "carb_g"))
    split = mf.split_bundle(np.arange(len(frame)), holdout=0.0)
    ti = mf.target_info("regression", "y")
    design = design_stage(mf.context(st, {"split": split, "target_info": ti}, paths))
    fit = fit_stage(mf.context(st, {"design": design, "split": split, "target_info": ti}, paths))
    sub = substitution_stage(mf.context(st, {"design": design, "fit": fit}, paths))
    model = fit.data["models"][0]
    return {"frame": frame, "fit": fit, "model": model, "sub": sub,
            "record": model["inference"]["missing"], "imputed": fit.objects["imputations"]}


@pytest.fixture(scope="module")
def chain_run(tmp_path_factory):
    frame = chain_frame()
    folder = tmp_path_factory.mktemp("chain")
    return {"spline": chain_stages(folder / "spline", frame, {"fiber_g": {"form": "spline",
                                                                           "knots": 4}}),
            "linear": chain_stages(folder / "linear", frame, {})}


def test_chain_contracts_declare_every_part_of_section_13():
    """Every MI method's contract, in the one registry (``turbotab.core.contracts``), names its slot,
    data scope, needs, routing (question; its option labeled customary and sound for each purpose,
    with the leash's rung), storyboard, the function that writes its sentence, and its relations;
    each conflict names its ways forward and each relation the code that makes it fire."""
    import importlib

    from turbotab.core import contracts as C
    from turbotab.core.methods.missing import MI_CONTRACTS

    assert set(MI_CONTRACTS) == {"multiple_imputation_compatible", "multiple_imputation_passive",
                                 "multiple_imputation_single_level", "imputed_copies_pooled"}
    for key in MI_CONTRACTS:
        contract = C.contract(key)
        assert contract.slot in C.SLOTS and contract.needs and contract.storyboard
        assert contract.relations and contract.question
        assert contract.scope == ("row_local" if key == "imputed_copies_pooled" else "model")
        for option in contract.options:
            assert option.customary and set(option.sound) == set(C.PURPOSES)
            assert all(option.sound.values()) and set(option.rung) == set(C.PURPOSES)
            assert all(r in C.RUNGS for r in option.rung.values())
        module, function = contract.sentence.split(":")
        assert callable(getattr(importlib.import_module(module), function))
        for relation in contract.relations:
            assert relation.kind in ("implies", "enables", "disables", "invalidates", "conflicts",
                                     "precedes")
            assert bool(relation.exits) == (relation.kind == "conflicts")
            module, function = relation.enforced_by.split(":")
            assert callable(getattr(importlib.import_module(module), function))


def test_chain_the_inference_run_follows_the_execution_order(chain_run):
    """MODELING_SEQUENCE §1.1, inference, for each copy: (1) imputation compatible with the analysis
    model (SMC-FCS; logs on the log scale; sources and the rest imputed, energy derived; the design
    in the model); (2) derived terms computed in the copy (the energy residual's line re-estimated:
    it differs between copies); (4) the spline basis on the fixed knots (identical in every copy);
    (5) the fit; then pooling (Rubin's rules on the design's df, D1 for the spline's tests, the
    curve pooled per k)."""
    run = chain_run["spline"]
    record, imputed = run["record"], run["imputed"]
    assert record["model"] == "smcfcs"
    fits = imputed["fits"]["linear"]
    slopes = {round(f.named_steps["energy"].params_["fiber_g"]["slope"], 12) for f in fits}
    assert len(slopes) > 1
    knots = [tuple(f.named_steps["form"].knots_["fiber_g_adj"]) for f in fits]
    assert len(set(knots)) == 1
    survey = run["model"]["inference"]["survey"]
    assert record["df_com"] == survey["df"] == 8
    tests = {t["test"]: t for t in run["model"]["exposure_tests"]}
    assert "D1" in tests["overall"]["caption"] and "Reiter" in tests["overall"]["caption"]
    assert run["sub"]["models"][0]["pooled"] == "per_k"


def test_chain_every_relation_the_contracts_declare_fires(chain_run, dxa_run):
    """Each relation in the MI contracts, checked on the run it governs; none is listed without a
    check here."""
    from turbotab.core.contracts import contract
    from turbotab.core.decisions import GrainSpec, RepeatSpec, SetUnit
    from turbotab.core.methods.missing import MI_CONTRACTS
    from turbotab.core.stages import build_graph

    run = chain_run["spline"]
    record, imputed, frame = run["record"], run["imputed"], run["frame"]

    def pools_every_estimate() -> None:
        assert "Rubin's rules" in run["model"]["inference"]["caption"]
        assert all(t["caption"].startswith("Pooled over") for t in run["model"]["exposure_tests"])
        assert run["sub"]["models"][0]["pooled"] == "per_k"
        assert f"The curve is pooled over the {record['m']} imputations" in run["sub"]["note"]

    def nonlinear_term_implies_compatible() -> None:
        assert record["model"] == "smcfcs" and record["substantive"] == "linear"
        assert record["terms"] == [
            "a restricted cubic spline of `fiber_g`",
            "the log residual of `fiber_g`, `protein_g`, `fat_g` and `carb_g` on `kcal`"]

    def logs_imply_log_scale() -> None:
        assert set(record["logged"]) == {"fiber_g", "protein_g"}
        for copy in imputed["frames"]:
            assert float(copy[LOGGED].min().min()) > 0

    def energy_identity() -> None:
        assert record["identity_energy"] == "kcal"
        assert record["identity_sources"] == ["protein_g", "fat_g", "carb_g"]
        for copy in imputed["frames"]:
            parts = sum(f * copy[s].to_numpy() for s, f in DIET_FACTORS.items())
            assert float((copy["kcal"].to_numpy() - parts).min()) >= 0

    def knots_fixed() -> None:
        fixed = record["knots"]["fiber_g_adj"]
        for fitted in imputed["fits"]["linear"]:
            assert np.array_equal(fitted.named_steps["form"].knots_["fiber_g_adj"], fixed)

    def form_change_invalidates_imputations() -> None:
        graph = build_graph()  # the stage graph reruns the fit when the design's reads change
        assert {"exposure_forms", "energy_adjustment", "missing"} <= set(graph["design"].reads)
        assert "design" in graph["fit"].deps
        redrawn = chain_run["linear"]["record"]
        assert redrawn["knots"] == {} and "a restricted cubic spline of `fiber_g`" not in redrawn["terms"]
        assert redrawn["model"] == "smcfcs"  # the log residual still needs it

    def survey_implies_design_variables_and_df() -> None:
        assert (record["design_strata"], record["design_psu"], record["design_weight"]) == \
            ("SDMVSTRA", "SDMVPSU", "WTMEC2YR")
        assert record["df_com"] == run["model"]["inference"]["survey"]["df"]
        rows = [r for r in run["model"]["coefficients"] if r.get("df") is not None]
        assert rows and all(r["df"] <= record["df_com"] + 1e-9 for r in rows)

    def clusters_imply_clustered_imputation() -> None:
        imputations, _ = clustered_imputations(visits_frame(), "clustered")
        assert imputations.plan["unit"] == "id" and imputations.plan["unit_level"]

    def items_before_the_score() -> None:
        from turbotab.core.decisions import ScaleSpec
        from turbotab.core.models.pipeline import shared_steps
        from turbotab.core.tests.acceptance.scales_fixtures import linear_scale_table

        items = [f"sat_{j}" for j in range(1, 9)]
        scale = linear_scale_table(n=200, missing=0.15)
        st = ProjectState(purpose="inference", target="sbp",
                          roles={"age": "covariate", "bmi": "covariate",
                                 **{c: "covariate" for c in items}},
                          missing=MissingSpec(strategy="multiple_imputation"),
                          scales=[ScaleSpec(name="sat_score", items=items,
                                            reverse=["sat_3", "sat_6"], low=1, high=5,
                                            kind="reflective")])
        X = scale[["age", "bmi", *items]]
        spec = design_spec(st, X, ["age", "bmi", *items])
        steps = [name for name, _ in shared_steps(spec)]
        assert steps.index("impute") < steps.index("score")
        copies = impute_for_inference(spec, X[spec.inputs], scale["sbp"].to_numpy(dtype=float),
                                      "regression", seed=0)
        assert set(items) <= set(copies.variables) and "sat_score" not in copies.variables

    def passive_conflicts_with_nonlinear_term() -> None:
        st = ProjectState(target="y", purpose="inference", roles={"x": "exposure"},
                          exposure_forms={"x": ExposureFormSpec(form="spline", knots=4)})
        with pytest.raises(Refusal) as refused:
            validate(SetMissing(strategy="multiple_imputation", imputation_model="passive"),
                     {"state": st})
        assert refused.value.exits[0]["label"].startswith("Multiple imputation compatible")
        declared = contract("multiple_imputation_passive").relation(
            "mi.passive_conflicts_with_nonlinear_term")
        assert [e["label"] for e in refused.value.exits] == list(declared.exits)

    def single_level_conflicts_with_clusters() -> None:
        st = ProjectState(target="y", purpose="inference", roles={"x": "exposure"},
                          grain=GrainSpec(grain="repeated", id_column="id"))
        with pytest.raises(Refusal) as refused:
            validate(SetMissing(strategy="multiple_imputation", imputation_levels="single_level"),
                     {"state": st})
        assert refused.value.exits[0]["label"] == "Clustered multiple imputation (time-invariant " \
                                                  "values once per unit)"
        declared = contract("multiple_imputation_single_level").relation(
            "mi.single_level_conflicts_with_clusters")
        assert [e["label"] for e in refused.value.exits] == list(declared.exits)

    def copies_imply_rubins_rules() -> None:
        assert dxa_run["model"]["inference"]["missing"]["model"] == "supplied"

    def copies_conflict_with_combining() -> None:
        st = ProjectState(target="y", purpose="inference",
                          grain=GrainSpec(grain="repeated", id_column="SEQN"),
                          repeat_kind=RepeatSpec(repeat_kind="imputed_copies",
                                                 implicate_column="_MULT_"))
        with pytest.raises(Refusal) as refused:
            validate(SetUnit(unit="unit"), {"state": st})
        assert refused.value.code == "imputed_copies_combined"
        assert refused.value.exits[0]["label"] == "Keep each copy as a record, pooled by Rubin's rules"
        declared = contract("imputed_copies_pooled").relation("copies.conflict_with_combining")
        assert [e["label"] for e in refused.value.exits] == list(declared.exits)

    checks = {
        "mi.pools_every_estimate": pools_every_estimate,
        "mi.nonlinear_term_implies_compatible": nonlinear_term_implies_compatible,
        "mi.logs_imply_log_scale": logs_imply_log_scale,
        "mi.energy_identity": energy_identity,
        "mi.knots_fixed": knots_fixed,
        "mi.form_change_invalidates_imputations": form_change_invalidates_imputations,
        "mi.survey_implies_design_variables_and_df": survey_implies_design_variables_and_df,
        "mi.clusters_imply_clustered_imputation": clusters_imply_clustered_imputation,
        "mi.items_before_the_score": items_before_the_score,
        "mi.passive_conflicts_with_nonlinear_term": passive_conflicts_with_nonlinear_term,
        "mi.single_level_conflicts_with_clusters": single_level_conflicts_with_clusters,
        "copies.imply_rubins_rules": copies_imply_rubins_rules,
        "copies.conflict_with_combining": copies_conflict_with_combining,
    }
    declared = {r.id for key in MI_CONTRACTS for r in contract(key).relations}
    assert declared == set(checks)
    for relation_id in sorted(declared):
        checks[relation_id]()


def test_chain_the_methods_sentence_is_written_as_the_record_says(chain_run):
    """The chain's methods sentence, verbatim (its numbers from the record it states)."""
    record = chain_run["spline"]["record"]
    knots = ", ".join(f"{k:.4g}" for k in record["knots"]["fiber_g_adj"])
    assert record["sentence"] == (
        f"Missing values were multiply imputed by substantive-model-compatible fully conditional "
        f"specification (SMC-FCS; Bartlett et al. 2015), compatible with the analysis model, a "
        f"linear model with a restricted cubic spline of `fiber_g` and the log residual of "
        f"`fiber_g`, `protein_g`, `fat_g` and `carb_g` on `kcal`; m = {record['m']} imputations, at "
        f"least 20 and at least the {record['percent_incomplete']:g}% of rows with an imputed value "
        f"(White, Royston & Wood 2011); `fiber_g` and `protein_g` were imputed on the log scale, "
        f"as the analysis takes their log; the energy sources (`protein_g`, `fat_g` and "
        f"`carb_g`) and the rest of energy were imputed and total energy (`kcal`) derived as their "
        f"sum, so the rest is never negative; the knots of `fiber_g_adj` ({knots}) were placed once "
        f"on its observed values and held in every copy; the survey strata (`SDMVSTRA`), PSU "
        f"(`SDMVPSU`) and weight (`WTMEC2YR`) were in the imputation model; the estimates were "
        f"pooled by Rubin's rules, multi-parameter tests by D1 (Li, Raghunathan & Rubin 1991), on "
        f"the design's 8 degrees of freedom; the largest Monte Carlo error, for "
        f"`{record['mc_feature']}`, was {100 * record['mc_max_ratio']:.1f}% of its standard error.")


def test_2_a_declared_code_column_enters_the_imputation_model_as_a_category():
    """A complete number column the user declared codes (RIDRETH3's 1, 2, 3, 4, 6, 7: six groups,
    not one line; BLUEPRINT §14.3) is a category in every covariate model of the imputation, as
    the analysis model holds it, whatever its values look like."""
    rng = np.random.default_rng(71)
    n = 300
    race = rng.choice([1.0, 2.0, 3.0, 4.0, 6.0, 7.0], n)
    x = rng.normal(size=n) + 0.3 * (race == 6.0)
    y = 1 + 0.5 * x + 0.4 * (race == 3.0) + rng.normal(size=n)
    frame = pd.DataFrame({"x": np.where(rng.random(n) < 0.3, np.nan, x), "race": race})
    roles = {"x": "exposure", "race": "covariate"}
    st = ProjectState(target="y", task="regression", purpose="inference", roles=roles,
                      missing=MissingSpec(strategy="multiple_imputation"),
                      shape_confirmations={"code_or_count:race": "code"})
    spec = design_spec(st, frame, list(roles))
    assert "race" in spec.categorical
    imputations = impute_for_inference(spec, frame, y, "regression", seed=7)
    assert imputations.kinds["race"] == "categorical"


@needs_r
def test_8_a_cox_splines_tests_are_pooled_by_d1_as_mitml_pools_them(tmp_path):
    """A Cox model's 3-knot spline: the engine's pooled overall and nonlinearity tests (each copy's
    partial-likelihood information as its covariance) against R refitting every copy with
    ``survival::coxph`` (Efron ties) on the basis ``Hmisc::rcspline.eval`` makes at the fixed knots,
    pooled by ``mitml::testConstraints(method = "D1")``: F, df and p to 1e-6."""
    rng = np.random.default_rng(83)
    n = 700
    z = rng.normal(size=n)
    x = 0.5 * z + rng.normal(size=n)
    hazard = 0.1 * np.exp(0.5 * x + 0.4 * np.maximum(x, 0) ** 2 + 0.3 * z)
    event_time, censor = rng.exponential(1 / hazard), rng.uniform(0, 15, n)
    t, d = np.minimum(event_time, censor), (event_time <= censor).astype(float)
    x = np.where(rng.random(n) < 1 / (1 + np.exp(-(-1 + 0.7 * z))), np.nan, x)
    frame = pd.DataFrame({"x": x, "z": z})
    y = survival_outcome(d, t)
    roles = {"x": "exposure", "z": "covariate"}
    st = ProjectState(target="y", task="time_to_event", purpose="inference", roles=roles,
                      missing=MissingSpec(strategy="multiple_imputation"),
                      exposure_forms={"x": ExposureFormSpec(form="spline", knots=3)})
    spec = design_spec(st, frame, list(roles))
    imputations = impute_for_inference(spec, frame, y, "time_to_event", seed=11)
    pipe = build_pipeline(spec, COX, "time_to_event", "inference", n, 2)
    _, _, tests, _ = pooled_table(COX, pipe, imputations, y, task="time_to_event",
                                  clusters=INDEPENDENT, outcome=Outcome(name="y"), survey=None,
                                  design=None, spec=spec, rows="all",
                                  fit=lambda model, X_k: fit_pipeline(model, X_k, y))
    tests = {t_["test"]: t_ for t_ in tests}
    knots = ", ".join(repr(float(k)) for k in imputations.plan["forms"]["x"]["knots"])
    copies = pd.concat([f[["x", "z"]].assign(t=t, d=d, copy=k + 1)
                        for k, f in enumerate(imputations.frames)], ignore_index=True)
    run_r(tmp_path, f"""
suppressMessages({{library(survival); library(mitml); library(Hmisc)}})
d <- read.csv("copies.csv")
Q <- NULL; U <- list()
for (k in sort(unique(d$copy))) {{
  dk <- d[d$copy == k, ]
  b <- rcspline.eval(dk$x, knots = c({knots}), inclx = TRUE)
  dk$s0 <- b[, 1]; dk$s1 <- b[, 2]
  fit <- coxph(Surv(t, d) ~ s0 + s1 + z, data = dk, ties = "efron")
  Q <- cbind(Q, coef(fit)); U[[k]] <- vcov(fit)
}}
U <- array(unlist(U), c(nrow(Q), nrow(Q), ncol(Q)), dimnames = list(rownames(Q), rownames(Q), NULL))
out <- do.call(rbind, lapply(list(overall = c("s0", "s1"), nonlinear = c("s1")), function(cs) {{
  t <- testConstraints(qhat = Q, uhat = U, constraints = cs, method = "D1")$test
  data.frame(F = t[1, "F.value"], df1 = t[1, "df1"], df2 = t[1, "df2"], p = t[1, "P(>F)"])
}}))
out$set <- c("overall", "nonlinear")
write.csv(out, "d1.csv", row.names = FALSE)
""", copies=copies)
    theirs = pd.read_csv(tmp_path / "d1.csv").set_index("set")
    assert_d1(tests["overall"], theirs.loc["overall"].to_dict())
    assert_d1(tests["nonlinear"], theirs.loc["nonlinear"].to_dict())


def test_2_a_recorded_zero_in_a_logged_column_is_refused_before_any_imputation(tmp_path):
    """Through the stages: a recorded zero in a column the log residual logs is refused by the
    energy model at the design stage, before any imputation could fill it (its message names the
    column and the two ways forward); the imputation refuses it as well when reached directly
    (:func:`test_2_a_recorded_zero_in_a_logged_column_is_refused_not_filled`)."""
    from turbotab.core.stages.modeling import design_stage
    from turbotab.core.tests import modeling_fixtures as mf

    frame = diet_frame(seed=131, n=300)
    rows = np.flatnonzero(frame["protein_g"].notna().to_numpy())
    frame.loc[frame.index[rows[:2]], "protein_g"] = 0.0
    paths = mf.ingest_frame(frame, tmp_path)
    st = mf.state(roles=dict(DIET_ROLES), target="y", task="regression", models=["linear"],
                  purpose="inference",
                  energy_adjustment=EnergyAdjustment(method="residual", energy_column="kcal",
                                                     nutrients=["fiber_g", "protein_g"],
                                                     log_transform=True),
                  missing=MissingSpec(strategy="multiple_imputation"),
                  split=SplitSpec(holdout=0.0, seed=0, folds=5))
    split = mf.split_bundle(np.arange(len(frame)), holdout=0.0)
    ti = mf.target_info("regression", "y")
    with pytest.raises(ValueError, match="protein_g is zero or negative in 2 of the fitting rows"):
        design_stage(mf.context(st, {"split": split, "target_info": ti}, paths))
