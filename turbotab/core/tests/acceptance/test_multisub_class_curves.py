"""MULTISUB · Multiclass substitution curves, one per class (V2 definition of done §2, "Dietary,
extended"; MODELING_SEQUENCE §2 and §4; the package's acceptance items, in its order).

1. For a multiclass outcome (multinomial logistic), one substitution curve per class (the change in
   the predicted probability as k kcal move from the donor to the recipient), with refit bands by
   bootstrap: the point curves agree with NumPy multinomial refits to 1e-8.
2. The curves sum to zero across classes at every k (asserted).
3. Under multiple imputation the per-class curves are pooled per k; under a population estimand
   they use the weights and the design-based variance (wave 1's APIs).

   The verifier's repair (MODELING_SEQUENCE §2, MS3: "A single-fill curve under inference is a
   defect"): where the missing-values answer blocks the coefficient table (passive imputation with
   a declared spline; imputation that cannot run), every curve is blocked and recorded with the
   table's own refusal and exits, for one curve per class and for one curve alike; a family with no
   table of its own is refit on each imputed copy and pooled; with no copies drawn at all, the
   curves are blocked with a way to draw them; and a design with no degrees of freedom left draws
   no curve over the surveyed population (the table refused, and both with the sample-only exit).
4. The label states the estimand (probability scale, isocaloric, at the stated population), and the
   energy-model rules of §2 hold (omitted sources block and record under inference).

Then the §13 contract and its chain test: every relation the contract declares fires.

**References, independent of the app's code.**

* **NumPy**: a multinomial logit fit by Newton–Raphson written here (:func:`numpy_multinomial`, on
  standardized columns, which leaves every probability unchanged), the all-components model
  matrix written out by hand (kcal from protein, carbohydrate and fat, the rest of energy, age), the
  curve's support written out by hand (:func:`numpy_support`: each moved intake within its observed
  range and not negative, its share of total energy within its observed range, the curve stopping
  at the first k with under half the rows), and Rubin's rules with Barnard & Rubin's degrees of
  freedom written out. The band's bootstrap is reproduced on the same resamples (the app's
  seeded draws), each refit by the NumPy fit.
* **Simulation and counting** for the repair: which rows are blank and which (stratum, PSU) pairs
  hold the analysis rows are counted from the table with pandas, so the design's degrees of freedom
  (PSUs minus strata) and the columns to impute are known without the app; each copy's curve for a
  family with no table is averaged over the support written out by hand, through that family
  refit on the copy (the fitted model is the family's; the curve, the support and the pooling are
  the reference's).
* **R 4.6** in a subprocess (``survey_r.run_r``; skipped without ``Rscript``): ``VGAM::vglm`` with
  the survey weights for the weighted multinomial fit, R's ``predict`` for each class's probability,
  each row's influence from VGAM's fit written in R, and ``survey::svyrecvar`` for the design's
  variance of the linearized predictive margin (Graubard & Korn 1999); and, as a variance estimator
  that shares no formula with linearization, the JKn jackknife over the same PSUs with VGAM refit on
  every replicate's weights.

**Sources.** Long & Freese 2014, *Regression Models for Categorical Dependent Variables Using
Stata*, 3rd ed., ch. 8 (average discrete changes of a multinomial model, one per outcome, summing to
zero); Graubard & Korn 1999, *Biometrics* 55:652 (predictive margins with survey data); Rubin 1987
and Barnard & Rubin 1999, *Biometrika* 86:948 (pooling); Schomaker & Heumann 2018, *Stat Med*
37:2252 (bootstrap with multiple imputation); Tomova, Gilthorpe & Tennant 2022 (composite variable
bias when energy sources are left out of a substitution model).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.special import logsumexp

import turbotab.core.models  # noqa: F401 - registers the families
from turbotab.core import voice
from turbotab.core.decisions import (EnergyAdjustment, ExposureFormSpec, MissingSpec, Refusal,
                                     SetSubstitution, SplitSpec, SubstitutionSpec, SurveySpec,
                                     parse_decision, validate)
from turbotab.core.models.artifacts import SubstitutionArtifact
from turbotab.core.stages.modeling import design_stage, fit_stage, substitution_stage
from turbotab.core.tests import modeling_fixtures as mf
from turbotab.core.tests.acceptance.survey_r import needs_r, nhanes_diet, run_r

SOURCES = {"protein_g": 4.0, "carb_g": 4.0, "fat_g": 9.0}
CLASSES = ["high", "low", "mid"]  # as the models hold them: sorted
ROLES = {"protein_g": "exposure", "carb_g": "exposure", "fat_g": "exposure", "kcal": "energy",
         "age": "covariate"}
STEP = 50.0
DONOR, RECIPIENT = "carb_g", "protein_g"


# ── the fixture: a three-class outcome of energy from each source ────────────


def diet_classes(seed: int = 7, n: int = 1200, blanks: bool = False) -> pd.DataFrame:
    """Protein, carbohydrate and fat (g), the rest of energy, total energy (kcal) and age; a
    three-class outcome drawn from a multinomial logit on kcal from each source and age (``mid``
    the reference): more protein raises ``high`` and lowers ``low``, more carbohydrate raises
    ``low``. With ``blanks``, protein (8%), total energy (5%) and age (5%) are blank."""
    rng = np.random.default_rng(seed)
    age = rng.normal(50, 12, n)
    size = 2100 * np.exp(0.2 * rng.normal(size=n)) - 6 * (age - 50)
    sp = np.clip(rng.normal(0.16, 0.04, n), 0.06, 0.35)
    sf = np.clip(rng.normal(0.34, 0.07, n), 0.12, 0.55)
    so = rng.uniform(0.02, 0.08, n)
    sc = 1.0 - sp - sf - so
    kp, kf, kc, ko = size * sp, size * sf, size * sc, size * so
    eta_low = 0.2 - 0.004 * (kp - 336) + 0.0015 * (kc - 1000) + 0.01 * (age - 50)
    eta_high = -0.4 + 0.004 * (kp - 336) + 0.002 * (kf - 714) - 0.015 * (age - 50)
    full = np.column_stack([np.zeros(n), eta_low, eta_high])
    p = np.exp(full - logsumexp(full, axis=1, keepdims=True))
    u = rng.random(n)
    y = np.where(u < p[:, 0], "mid", np.where(u < p[:, 0] + p[:, 1], "low", "high"))
    frame = pd.DataFrame({"protein_g": kp / 4, "carb_g": kc / 4, "fat_g": kf / 9,
                          "kcal": kp + kf + kc + ko, "age": age, "y": y})
    if blanks:
        for column, share in (("protein_g", 0.08), ("kcal", 0.05), ("age", 0.05)):
            frame.loc[rng.random(n) < share, column] = np.nan
    return frame


def run(folder: Path, frame: pd.DataFrame, *, purpose: str = "inference", n_boot: int = 0,
        models: tuple[str, ...] = ("linear",), holdout: float = 0.0, missing: Any = None,
        roles: dict[str, str] | None = None, acknowledged: bool = False,
        method: str = "all_components", forms: dict[str, Any] | None = None,
        task: str = "multiclass") -> dict[str, Any]:
    """The design, fit and substitution stages (``STEP`` kcal from carbohydrate to protein; the
    energy model ``method`` of the exposures and total energy, all-components by default; the
    declared ``forms``; a ``task`` other than multiclass for the one-curve path), as the engine
    runs them."""
    roles = roles or ROLES
    exposures = [c for c, r in roles.items() if r == "exposure"]
    paths = mf.ingest_frame(frame, folder)
    slots: dict[str, Any] = {}
    if missing is not None:
        slots["missing"] = missing
    if forms:
        slots["exposure_forms"] = forms
    st = mf.state(roles=roles, target="y", task=task, models=list(models), purpose=purpose,
                  energy_adjustment=EnergyAdjustment(method=method, energy_column="kcal",
                                                     nutrients=exposures),
                  substitution=SubstitutionSpec(donor=DONOR, recipient=RECIPIENT, step_kcal=STEP,
                                                n_boot=n_boot, acknowledged=acknowledged),
                  split=SplitSpec(holdout=holdout, seed=0, folds=5),
                  column_units=mf.grams(*SOURCES), **slots)
    split = mf.split_bundle(np.arange(len(frame)), holdout=holdout)
    ti = mf.target_info(task, "y")
    design = design_stage(mf.context(st, {"split": split, "target_info": ti}, paths))
    fit = fit_stage(mf.context(st, {"design": design, "split": split, "target_info": ti}, paths))
    sub = substitution_stage(mf.context(st, {"design": design, "fit": fit}, paths))
    SubstitutionArtifact.model_validate(sub)
    return {"state": st, "design": design, "fit": fit, "sub": sub, "split": split, "paths": paths}


# ── the NumPy reference ──────────────────────────────────────────────────────


def numpy_multinomial(Z: np.ndarray, y: Any, weights: np.ndarray | None = None,
                      classes: list[str] | None = None) -> Any:
    """The multinomial logit's maximum-likelihood fit by Newton–Raphson, against the first class of
    ``CLASSES``, on ``Z``'s columns standardized (which changes no probability); step-halving on
    the log-likelihood, until the largest step is below 1e-13 of the coefficients. Returns the
    probability function (rows of Z → n × K, in the order of ``classes``, default ``CLASSES``)."""
    classes = CLASSES if classes is None else classes
    Z = np.asarray(Z, dtype=float)
    n = len(Z)
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    mu, sd = Z.mean(axis=0), Z.std(axis=0)
    X = np.column_stack([np.ones(n), (Z - mu) / sd])
    P = X.shape[1]
    K = len(classes)
    codes = np.array([classes.index(v) for v in np.asarray(y)])
    Y = np.zeros((n, K))
    Y[np.arange(n), codes] = 1.0

    def eta(b: np.ndarray, M: np.ndarray) -> np.ndarray:
        return np.column_stack([np.zeros(len(M)), M @ b])

    def loglik(b: np.ndarray) -> float:
        e = eta(b, X)
        return float(np.sum(w * (np.sum(Y * e, axis=1) - logsumexp(e, axis=1))))

    beta = np.zeros((P, K - 1))
    for _ in range(200):
        e = eta(beta, X)
        prob = np.exp(e - logsumexp(e, axis=1, keepdims=True))
        gradient = np.concatenate([X.T @ (w * (Y[:, a] - prob[:, a])) for a in range(1, K)])
        hessian = np.zeros(((K - 1) * P, (K - 1) * P))
        for a in range(1, K):
            for b in range(1, K):
                cell = w * prob[:, a] * ((a == b) - prob[:, b])
                hessian[(a - 1) * P:a * P, (b - 1) * P:b * P] = X.T @ (X * cell[:, None])
        step = np.linalg.solve(hessian, gradient).reshape(K - 1, P).T
        base, t = loglik(beta), 1.0
        while loglik(beta + t * step) < base - 1e-12 * abs(base) and t > 1e-10:
            t /= 2.0
        beta = beta + t * step
        if np.max(np.abs(t * step)) < 1e-13 * (1.0 + np.max(np.abs(beta))):
            break

    def proba(Znew: np.ndarray) -> np.ndarray:
        Xn = np.column_stack([np.ones(len(Znew)), (np.asarray(Znew, dtype=float) - mu) / sd])
        e = eta(beta, Xn)
        return np.exp(e - logsumexp(e, axis=1, keepdims=True))

    return proba


def all_components(frame: pd.DataFrame) -> np.ndarray:
    """The all-components model's columns, written out: kcal from protein, carbohydrate and fat,
    the rest of total energy, and age."""
    kcal = {s: f * frame[s].to_numpy(dtype=float) for s, f in SOURCES.items()}
    rest = frame["kcal"].to_numpy(dtype=float) - sum(kcal.values())
    return np.column_stack([kcal["protein_g"], kcal["carb_g"], kcal["fat_g"], rest,
                            frame["age"].to_numpy(dtype=float)])


def moved(frame: pd.DataFrame, k: float) -> pd.DataFrame:
    """Every row with k kcal moved from carbohydrate to protein, total energy unchanged."""
    out = frame.copy()
    out[DONOR] = out[DONOR] - k / SOURCES[DONOR]
    out[RECIPIENT] = out[RECIPIENT] + k / SOURCES[RECIPIENT]
    return out


def numpy_support(frame: pd.DataFrame, ks: list[float]) -> tuple[list, list, np.ndarray]:
    """Each k's rows on support, written out: carbohydrate and protein after the move within their
    observed ranges and not negative, and each one's share of total energy within the range of that
    share observed; the curve stops at the first k with under half the rows. Returns (the masks,
    which ks are live, the fixed population: the rows on support at every live k)."""
    d, r, e = (frame[c].to_numpy(dtype=float) for c in (DONOR, RECIPIENT, "kcal"))
    fd, fr = SOURCES[DONOR], SOURCES[RECIPIENT]
    share = {"d": (fd * d / e), "r": (fr * r / e)}
    masks, stop = [], None
    for k in ks:
        nd, nr = d - k / fd, r + k / fr
        ok = ((nd >= d.min()) & (nd <= d.max()) & (nd >= 0) & (nr >= r.min()) & (nr <= r.max())
              & (nr >= 0) & (fd * nd / e >= share["d"].min()) & (fd * nd / e <= share["d"].max())
              & (fr * nr / e >= share["r"].min()) & (fr * nr / e <= share["r"].max()))
        if stop is None and ok.mean() < 0.5:
            stop = k
        masks.append(ok)
    live = [(stop is None or k < stop) and bool(mk.any()) for k, mk in zip(ks, masks)]
    fixed = np.all([mk for mk, lv in zip(masks, live) if lv], axis=0)
    return masks, live, fixed


def numpy_curves(proba: Any, frame: pd.DataFrame, ks: list[float], masks: list, live: list,
                 fixed: np.ndarray, weights: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Each class's curve and fixed-population curve (k × class; NaN where not live): the
    (weighted) mean over the rows on support of each row's change in that class's probability."""
    w = np.ones(len(frame)) if weights is None else weights
    base = proba(all_components(frame))
    curve = np.full((len(ks), len(CLASSES)), np.nan)
    fixed_curve = np.full((len(ks), len(CLASSES)), np.nan)
    for i, k in enumerate(ks):
        if not live[i]:
            continue
        diff = proba(all_components(moved(frame, k))) - base
        curve[i] = (w[masks[i]] @ diff[masks[i]]) / w[masks[i]].sum()
        fixed_curve[i] = (w[fixed] @ diff[fixed]) / w[fixed].sum()
    return curve, fixed_curve


def by_class(sub: dict, family: str = "linear") -> dict[str, dict]:
    return {m["level"]: m for m in sub["models"] if m["family"] == family and m.get("level")}


def bootstrap_se(frame: pd.DataFrame, y: np.ndarray, ks: list[float], masks: list, live: list,
                 fixed: np.ndarray, n_boot: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Each class's bootstrap standard error at each k (k × class), from NumPy refits on the
    resamples the app draws (``np.random.default_rng(seed).integers(0, n, n)`` per refit: the
    ordinary bootstrap of rows), each refit's curve averaged over the curve's rows as often as each
    was drawn; and the fixed population's."""
    n = len(frame)
    rng = np.random.default_rng(seed)
    Z = all_components(frame)
    shifted = {i: all_components(moved(frame, k)) for i, k in enumerate(ks) if live[i]}
    draws = np.full((n_boot, len(ks), len(CLASSES)), np.nan)
    fixed_draws = np.full_like(draws, np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        proba = numpy_multinomial(Z[idx], y[idx])
        base = proba(Z)
        count = np.bincount(idx, minlength=n).astype(float)
        for i, Z1 in shifted.items():
            diff = proba(Z1) - base
            on = masks[i]
            draws[b, i] = (count[on] @ diff[on]) / count[on].sum()
            fixed_draws[b, i] = (count[fixed] @ diff[fixed]) / count[fixed].sum()
    return draws.std(axis=0, ddof=1), fixed_draws.std(axis=0, ddof=1)


Z95 = stats.norm.ppf(0.975)


# ── 1 · one curve per class, with refit bands ────────────────────────────────


@pytest.fixture(scope="module")
def inference_run(tmp_path_factory):
    frame = diet_classes()
    return {"frame": frame, **run(tmp_path_factory.mktemp("inference"), frame, n_boot=200)}


def test_1_each_class_curve_agrees_with_numpy_multinomial_refits(inference_run):
    """Under inference the curves read every analyzed row and the linear family's multinomial logit
    fit on all of them. Reference: the NumPy multinomial fit (:func:`numpy_multinomial`) on the
    all-components matrix written out by hand, each class's mean change in its predicted
    probability over the support written out by hand. Bound: 1e-8, absolute, at every live k, for
    each class's curve and its fixed-population curve; the curves stop where the reference's do."""
    frame, sub = inference_run["frame"], inference_run["sub"]
    ks = sub["ks"]
    assert ks == [STEP * i for i in range(11)]
    proba = numpy_multinomial(all_components(frame), frame["y"].to_numpy())
    masks, live, fixed = numpy_support(frame, ks)
    curve, fixed_curve = numpy_curves(proba, frame, ks, masks, live, fixed)
    classes = by_class(sub)
    assert sorted(classes) == CLASSES and len(sub["models"]) == 3
    assert sum(live) >= 4
    for c, level in enumerate(CLASSES):
        entry = classes[level]
        assert entry["label"] == f"Linear model: {level}"
        for i, k in enumerate(ks):
            if not live[i]:
                assert entry["delta"][i] is None
                continue
            assert abs(entry["delta"][i] - curve[i, c]) <= 1e-8, (level, k)
            assert abs(entry["fixed_delta"][i] - fixed_curve[i, c]) <= 1e-8, (level, k)
    assert sub["support"]["fixed_rows"] == int(fixed.sum())
    assert classes["high"]["stopped_at"] == next(k for k, lv in zip(ks, live) if not lv)


def test_1_each_class_band_is_its_refits_spread_on_the_same_resamples(inference_run):
    """The band asked for (200 refits): each class's curve ± z standard errors of its refits (fewer
    than 1,000 refits: the normal interval). Reference: the same 200 resamples of rows refit by
    NumPy, each refit's class curves averaged over the analyzed rows as often as each was drawn;
    the standard deviation of the 200 curves at each k. Bound: 1e-8 on every band edge, for the
    curve and the fixed population."""
    frame, sub = inference_run["frame"], inference_run["sub"]
    ks = sub["ks"]
    y = frame["y"].to_numpy()
    masks, live, fixed = numpy_support(frame, ks)
    se, fixed_se = bootstrap_se(frame, y, ks, masks, live, fixed, 200, seed=0)
    band = sub["band"]
    assert band["n_boot"] == 200 and band["interval"] == "normal" and band["failed"] == 0
    assert band["n_units"] == band["resample_units"] == len(frame) and band["scale"] == 1.0
    for c, level in enumerate(CLASSES):
        entry = by_class(sub)[level]
        assert entry["band_ok"] == 200
        for i in range(len(ks)):
            if not live[i] or ks[i] == 0:
                continue
            for got, mid, spread in ((entry["ci_low"][i], entry["delta"][i], -se[i, c]),
                                     (entry["ci_high"][i], entry["delta"][i], se[i, c]),
                                     (entry["fixed_ci_low"][i], entry["fixed_delta"][i],
                                      -fixed_se[i, c]),
                                     (entry["fixed_ci_high"][i], entry["fixed_delta"][i],
                                      fixed_se[i, c])):
                assert abs(got - (mid + Z95 * spread)) <= 1e-8, (level, ks[i])
    assert "Refits that succeeded: Linear model 200 of 200." in band["caption"]


def test_1_under_prediction_the_curves_read_the_training_rows_and_fit(tmp_path):
    """Under prediction the curves read the training rows and the training fit (a holdout of 20%
    drawn): the linear family's agree with the NumPy fit on the training rows to 1e-8, and with no
    band asked for, one refit is timed so the offer of a band can say what it costs."""
    frame = diet_classes(seed=11, n=900)
    out = run(tmp_path, frame, purpose="prediction", holdout=0.2)
    sub = out["sub"]
    assignment = out["split"].frames["assignment"]
    train = assignment.loc[assignment["partition"] == "train", "row_id"].to_numpy()
    assert len(train) == 720
    rows = frame.iloc[train]
    proba = numpy_multinomial(all_components(rows), rows["y"].to_numpy())
    masks, live, fixed = numpy_support(rows, sub["ks"])
    curve, _ = numpy_curves(proba, rows, sub["ks"], masks, live, fixed)
    linear = by_class(sub, "linear")
    assert sum(live) >= 4
    for c, level in enumerate(CLASSES):
        for i in range(len(sub["ks"])):
            if live[i]:
                assert abs(linear[level]["delta"][i] - curve[i, c]) <= 1e-8
    assert sub["basis"] == f"Averaged over {len(train):,} training rows."
    assert sub["estimand"].endswith("over the 720 training rows, each counted once. The class "
                                    "curves sum to zero at every k, because each row's class "
                                    "probabilities sum to one.")
    assert sub["band"] is None and sub["band_estimate"]["n_boot"] == 200
    assert_sums_to_zero(linear)


def test_1_a_family_with_no_likelihood_draws_its_own_class_curves(tmp_path):
    """Gradient-boosted trees (no likelihood to refit by hand) draw their own curve per class.
    Under inference the fit stage leaves them fit on the training rows only (they have no
    coefficient table), so their curves read their refit on every analyzed row, as the one-curve
    stage does; each class's curve is drawn where the linear family's is, and the three sum to zero
    at every k."""
    frame = diet_classes(seed=13, n=600)
    out = run(tmp_path, frame, models=("linear", "boosted_trees"))
    sub = out["sub"]
    trees, linear = by_class(sub, "boosted_trees"), by_class(sub, "linear")
    assert sorted(trees) == CLASSES
    assert [m["label"] for m in sub["models"]] == [
        "Linear model: high", "Linear model: low", "Linear model: mid", "Boosted trees: high",
        "Boosted trees: low", "Boosted trees: mid"]
    for level in CLASSES:
        assert ([v is None for v in trees[level]["delta"]]
                == [v is None for v in linear[level]["delta"]])
    assert any(abs(v) > 1e-6 for v in trees["high"]["delta"][1:] if v is not None)
    assert_sums_to_zero(trees)


def test_1_a_resample_without_a_class_fails_its_refit_and_is_counted():
    """A bootstrap resample can miss a rare class. Its refit cannot say that class's probability,
    so the refit fails and is counted (never read with its columns shifted onto the wrong classes),
    and a band is drawn only when at least 90% of the refits succeed. Measured on a table whose
    ``high`` class has 4 rows of 150: a resample misses all four with probability
    (1 − 4/150)^150 ≈ 1.7%, so of 300 refits about 5 fail (the count is checked against the
    resamples themselves, drawn again here)."""
    from sklearn.linear_model import LogisticRegression

    from turbotab.core.methods.substitution import Shift, class_refit_band
    from turbotab.core.stages.class_substitution import class_predictor

    import warnings

    from sklearn.pipeline import Pipeline

    frame = diet_classes(seed=17, n=150)
    y = np.where(np.arange(150) < 4, "high", np.where(np.arange(150) % 2 == 0, "low", "mid"))
    X = frame[["protein_g", "carb_g"]]

    def fit(Xb: pd.DataFrame, yb: np.ndarray) -> Any:  # as the stage refits: warnings kept quiet
        model = LogisticRegression(C=np.inf, solver="newton-cholesky", max_iter=1000)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline = Pipeline([("model", model)]).fit(Xb, yb)
        return class_predictor(pipeline, Xb, yb, CLASSES)

    with pytest.raises(ValueError, match="does not hold every class"):
        fit(X.iloc[4:], y[4:])
    shift = Shift(X, donor=DONOR, recipient=RECIPIENT, kcal_per_unit=SOURCES)
    band = class_refit_band(fit, X, y, classes=CLASSES, shift=shift, ks=[0, 10], live=[True, True],
                            n_boot=300, random_state=4)
    rng = np.random.default_rng(4)
    missing = sum(not (rng.integers(0, 150, size=150) < 4).any() for _ in range(300))
    assert missing > 0 and band["failed"] == missing and band["n_ok"] == 300 - missing
    assert band["refused"] is None and band["classes"][0]["ci_low"][1] is not None


# ── 2 · the curves sum to zero at every k ────────────────────────────────────


def assert_sums_to_zero(classes: dict[str, dict]) -> None:
    """At every k, each class's curve (and fixed-population curve) drawn, and their sum zero to
    1e-12."""
    for key in ("delta", "fixed_delta"):
        curves = [c[key] for c in classes.values()]
        for i in range(len(curves[0])):
            values = [v[i] for v in curves]
            assert all(v is None for v in values) or all(v is not None for v in values), (key, i)
            if values[0] is not None:
                assert abs(math.fsum(values)) <= 1e-12, (key, i, math.fsum(values))


def test_2_the_class_curves_sum_to_zero_at_every_k(inference_run):
    """Each row's three probabilities sum to one before and after the move, so their changes, and
    every average of them over the same rows, sum to zero: probability moves between classes and
    none is made. Checked on the drawn curves (to 1e-12), and the check is the app's own: a
    predictor whose probabilities do not sum to one is refused before any curve is drawn."""
    from turbotab.core.methods.substitution import class_curves

    assert_sums_to_zero(by_class(inference_run["sub"]))
    frame = inference_run["frame"]
    X = frame[["protein_g", "carb_g", "fat_g", "kcal", "age"]]
    proba = numpy_multinomial(all_components(frame), frame["y"].to_numpy())

    def faithful(f: pd.DataFrame) -> np.ndarray:
        return proba(all_components(f))

    args = dict(classes=CLASSES, donor=DONOR, recipient=RECIPIENT,
                kcal_per_unit=SOURCES, ks=[0, 50, 100], total="kcal")
    drawn = class_curves(faithful, X, **args)
    assert drawn["class_sum"] <= 1e-12

    def leaky(f: pd.DataFrame) -> np.ndarray:  # probabilities that do not sum to one
        return proba(all_components(f)) * np.array([1.0, 1.0, 0.5])

    with pytest.raises(ArithmeticError, match="not to zero"):
        class_curves(leaky, X, **args)


# ── 3 · multiple imputation and the surveyed population ──────────────────────


@pytest.fixture(scope="module")
def imputed_run(tmp_path_factory):
    frame = diet_classes(seed=23, n=900, blanks=True)
    out = run(tmp_path_factory.mktemp("imputed"), frame, n_boot=40,
              missing=MissingSpec(strategy="multiple_imputation"))
    return {"frame": frame, **out, "copies": out["fit"].objects["imputations"]["frames"]}


def test_3_under_multiple_imputation_each_class_curve_is_pooled_per_k(imputed_run):
    """Each completed copy's class curves are drawn through that copy's own fit on that copy's
    rows, and pooled at each k. Reference: every copy refit by the NumPy multinomial fit, its
    curves on its own support, written out; the pooled curve the mean over the copies (bound 1e-8).
    The band: each copy's within-copy variance its bootstrap's (40 refits split over the copies:
    10 per copy, each copy's resamples drawn with the copy's number as the seed), reproduced by
    NumPy refits on the same resamples, then Rubin's rules, T = Ū + (1 + 1/m)B, on Barnard & Rubin's
    ν = (m − 1)/λ² (no complete-data df: large-sample rows); bound 1e-8 on the interval edges and
    1e-6, relative, on ν."""
    sub, copies, frame = imputed_run["sub"], imputed_run["copies"], imputed_run["frame"]
    y = frame["y"].to_numpy()
    ks = sub["ks"]
    m = len(copies)
    assert m == 20
    curves, fixed_curves, ses, lives = [], [], [], []
    for j, copy in enumerate(copies):
        assert not copy.isna().any().any()
        proba = numpy_multinomial(all_components(copy), y)
        masks, live, fixed = numpy_support(copy, ks)
        curve, fixed_curve = numpy_curves(proba, copy, ks, masks, live, fixed)
        curves.append(curve)
        fixed_curves.append(fixed_curve)
        lives.append(live)
    live = [all(lv[i] for lv in lives) for i in range(len(ks))]
    for j, copy in enumerate(copies):
        masks, _, _ = numpy_support(copy, ks)
        fixed = np.all([mk for mk, lv in zip(masks, live) if lv], axis=0)
        se, _ = bootstrap_se(copy, y, ks, masks, live, fixed, 10, seed=j)
        ses.append(se)
    Q = np.asarray(curves)  # m × k × class
    U = np.asarray(ses) ** 2
    classes = by_class(sub)
    for c, level in enumerate(CLASSES):
        entry = classes[level]
        assert entry["pooled"] == "per_k"
        for i, k in enumerate(ks):
            if not live[i]:
                assert entry["delta"][i] is None
                continue
            assert abs(entry["delta"][i] - Q[:, i, c].mean()) <= 1e-8, (level, k)
            assert abs(entry["fixed_delta"][i]
                       - np.asarray(fixed_curves)[:, i, c].mean()) <= 1e-8, (level, k)
            if k == 0:
                continue
            ubar, b = U[:, i, c].mean(), Q[:, i, c].var(ddof=1)
            t = ubar + (1 + 1 / m) * b
            lam = (1 + 1 / m) * b / t
            nu = (m - 1) / lam ** 2
            half = stats.t.ppf(0.975, nu) * math.sqrt(t)
            assert entry["df"][i] == pytest.approx(nu, rel=1e-6), (level, k)
            assert abs(entry["ci_low"][i] - (Q[:, i, c].mean() - half)) <= 1e-8, (level, k)
            assert abs(entry["ci_high"][i] - (Q[:, i, c].mean() + half)) <= 1e-8, (level, k)
    assert_sums_to_zero(classes)
    band = sub["band"]
    assert band["n_boot"] == 40 and band["failed"] == 0 and band["interval"] == "normal"
    assert all(e["band_ok"] == 10 * m for e in classes.values())
    assert (f"Each class's curve is pooled over the {m} imputations: each copy's class curves, "
            f"pooled at each k by Rubin's rules with each copy's bootstrap variance as its "
            f"within-copy variance (40 refits split over the copies; Schomaker & Heumann 2018).") \
        in sub["note"]
    assert "filled once" not in sub["note"]


DIET_ROLES = {"SEQN": "identifier", "protein_g": "exposure", "fat_g": "exposure",
              "carb_g": "exposure", "energy_kcal": "energy", "age": "covariate",
              "female": "covariate", "WTDRD1": "design", "SDMVSTRA": "design",
              "SDMVPSU": "design"}
POPULATION = {"estimand": "population", "weight": "WTDRD1", "strata": "SDMVSTRA",
              "psu": "SDMVPSU"}
CRP_CLASSES = ["average", "high", "low"]


def crp_table() -> pd.DataFrame:
    """MS4's NHANES-shaped dietary table (``survey_r.nhanes_diet``), its CRP in the AHA/CDC risk
    classes: low (below 1 mg/L), average (1 to 3) and high (above 3)."""
    f = nhanes_diet()
    f["crp_class"] = np.where(f["crp"] < 1, "low", np.where(f["crp"] <= 3, "average", "high"))
    return f


def survey_run(folder: Path, f: pd.DataFrame, *, models: list[str], n_boot: int = 0,
               survey: dict | None = POPULATION, missing: Any = None,
               target: str = "crp_class", task: str = "multiclass") -> dict[str, Any]:
    """The design, fit and substitution stages under inference on the survey table (every row in
    the design, the under-20s outside the analysis), ``STEP`` kcal from fat to carbohydrate; the
    CRP classes, or with ``target`` ``high_crp`` the yes/no outcome (the one-curve path)."""
    analyzed = np.flatnonzero(f["eligible"].to_numpy() == 1)
    other = "high_crp" if target == "crp_class" else "crp_class"
    frame = f.drop(columns=["eligible", "over", other, "crp"])
    paths = mf.ingest_frame(frame, folder)
    confirmations = {"code_or_count:female": "code", "code_or_count:SDMVSTRA": "code",
                     "code_or_count:SDMVPSU": "code"}
    slots: dict[str, Any] = {}
    if missing is not None:
        slots["missing"] = missing
    st = mf.state(roles=DIET_ROLES, target=target, task=task, models=models,
                  purpose="inference", split=SplitSpec(holdout=0.0, seed=0, folds=5),
                  survey=SurveySpec(**survey) if survey else None,
                  substitution=SubstitutionSpec(donor="fat_g", recipient="carb_g", step_kcal=STEP,
                                                n_boot=n_boot),
                  column_units=mf.grams("protein_g", "fat_g", "carb_g"),
                  shape_confirmations=confirmations, **slots)
    split = mf.split_bundle(analyzed, holdout=0.0)
    ti = mf.target_info(task, target)
    design = design_stage(mf.context(st, {"split": split, "target_info": ti}, paths))
    fit = fit_stage(mf.context(st, {"design": design, "split": split, "target_info": ti}, paths))
    sub = substitution_stage(mf.context(st, {"design": design, "fit": fit}, paths))
    SubstitutionArtifact.model_validate(sub)
    return {"state": st, "fit": fit, "sub": sub, "analyzed": analyzed}


@pytest.fixture(scope="module")
def crp() -> pd.DataFrame:
    return crp_table()


@pytest.fixture(scope="module")
def population_run(crp, tmp_path_factory):
    return survey_run(tmp_path_factory.mktemp("population"), crp, models=["linear"], n_boot=200)


SURVEY_R = """
suppressMessages(library(VGAM))
f <- read.csv("f.csv")
d <- svydesign(ids = ~SDMVPSU, strata = ~SDMVSTRA, weights = ~WTDRD1, nest = TRUE, data = f)
s <- subset(d, eligible == 1)
rows <- s$variables
w <- weights(s)
levels <- c("average", "high", "low")
rows$cls <- factor(rows$crp_class, levels = levels)
form <- ~ protein_g + fat_g + carb_g + energy_kcal + age + female
rows$pw <- w / mean(w)
fit_on <- function(pw) {  # a row of zero weight adds nothing to the likelihood: left out
  rows$pw <- pw
  suppressWarnings(vglm(cls ~ protein_g + fat_g + carb_g + energy_kcal + age + female,
                        multinomial(refLevel = 1), weights = pw, data = rows[pw > 0, ],
                        control = vglm.control(epsilon = 1e-15, maxit = 200)))
}
fit <- fit_on(w / mean(w))
ks <- scan("ks.txt", quiet = TRUE)
K <- 3; X0 <- model.matrix(form, rows); P <- ncol(X0)
P0 <- predict(fit, newdata = rows, type = "response")
Y <- sapply(levels, function(l) as.numeric(rows$cls == l))
S <- do.call(cbind, lapply(2:K, function(a) w * (Y[, a] - P0[, a]) * X0))
I <- matrix(0, (K - 1) * P, (K - 1) * P)
for (a in 2:K) for (b in 2:K) {
  cell <- w * P0[, a] * ((a == b) - P0[, b])
  I[((a - 2) * P + 1):((a - 1) * P), ((b - 2) * P + 1):((b - 1) * P)] <- t(X0) %*% (X0 * cell)
}
psi <- S %*% solve(I)
G <- function(X, Pm, c) do.call(cbind, lapply(2:K, function(a) Pm[, c] * ((c == a) - Pm[, a]) * X))
margins <- function(fitted, i) {
  movedrows <- rows
  movedrows$fat_g <- rows[[paste0("fat", i)]]
  movedrows$carb_g <- rows[[paste0("carb", i)]]
  list(P0 = predict(fitted, newdata = rows, type = "response"),
       P1 = predict(fitted, newdata = movedrows, type = "response"),
       X1 = model.matrix(form, movedrows))
}
est <- list(); se <- list()
for (i in seq_along(ks)) {
  m <- rows[[paste0("m", i)]]
  if (sum(m) == 0 || ks[i] == 0) next
  g <- margins(fit, i)
  wm <- w * m
  for (c in 1:K) {
    delta <- g$P1[, c] - g$P0[, c]
    theta <- sum(wm * delta) / sum(wm)
    grad <- colSums(wm * (G(g$X1, g$P1, c) - G(X0, g$P0, c))) / sum(wm)
    z <- wm * (delta - theta) / sum(wm) + psi %*% grad
    key <- paste0(levels[c], "_", i)
    est[[key]] <- theta
    se[[key]] <- sqrt(svyrecvar(z, s$cluster, s$strata, s$fpc)[1, 1])
  }
}
rd <- subset(as.svrepdesign(d, type = "JKn"), eligible == 1)
reps <- weights(rd, "analysis")
live <- which(sapply(seq_along(ks), function(i) sum(rows[[paste0("m", i)]]) > 0 && ks[i] > 0))
thetas <- array(NA, c(ncol(reps), length(ks), K))
for (r in seq_len(ncol(reps))) {
  pw <- reps[, r]
  fitted <- fit_on(pw / mean(pw))
  for (i in live) {
    m <- rows[[paste0("m", i)]]
    g <- margins(fitted, i)
    for (c in 1:K) thetas[r, i, c] <- sum(pw * m * (g$P1[, c] - g$P0[, c])) / sum(pw * m)
  }
}
jk <- list()
for (i in live) for (c in 1:K) {
  key <- paste0(levels[c], "_", i)
  v <- svrVar(thetas[, i, c], rd$scale, rd$rscales, mse = rd$mse, coef = est[[key]])
  jk[[key]] <- sqrt(as.numeric(v))
}
out(list(est = est, se = se, jk = jk, degf = degf(s)))
"""


@needs_r
def test_3_under_the_surveyed_population_each_class_curve_is_design_based(crp, population_run,
                                                                        tmp_path):
    """Moving k kcal from fat to carbohydrate on the AHA/CDC CRP classes, under the population
    answer, through the real design, fit and substitution stages: the linear family's multinomial
    logit is refit with the dietary weight ``WTDRD1`` (pseudo-maximum likelihood), each class's
    curve is the population's weighted mean of each participant's change in that class's
    probability, and its band is Taylor linearization over the design on t(d). The under-20s are
    outside the analysis and stay in the design. The 200 refits asked for are not drawn.

    Reference: R ``VGAM::vglm`` with the weights (the fit), R's ``predict`` (each class's
    probability before and after the move), each row's influence on the coefficients and the
    linearized margin written in R from VGAM's fit, and ``survey::svyrecvar`` (the design's
    variance of the total); the rows on support at each k and the moved intakes are the app's
    (``methods.substitution.Shift``), handed to R as columns. Bounds: each curve 1e-8 and its
    standard error 1e-6, relative (VGAM's iteratively reweighted least squares run to a deviance
    tolerance of 1e-15; measured: 1e-12). And the JKn jackknife over the same PSUs, refitting VGAM
    on every replicate's weights, a variance estimator that shares no formula with linearization:
    each standard error within 5% of the jackknife's (the two agree as the PSUs grow; here 30 PSUs
    in 15 strata, measured within 1.5%)."""
    from turbotab.core.methods.substitution import Shift

    f = crp
    sub = population_run["sub"]
    analyzed = population_run["analyzed"]
    ks = np.asarray(sub["ks"], dtype=float)
    X = f.iloc[analyzed][["protein_g", "fat_g", "carb_g", "energy_kcal", "age", "female"]]
    shift = Shift(X, donor="fat_g", recipient="carb_g", kcal_per_unit={"fat_g": 9.0, "carb_g": 4.0},
                  total="energy_kcal")
    assert shift.valid(X).all()
    classes = by_class(sub)
    assert sorted(classes) == CRP_CLASSES
    handed = f.copy()
    for i, k in enumerate(ks, start=1):
        moved_x, amount, composition = shift.checks(X, float(k))
        on = amount & composition & (classes["high"]["delta"][i - 1] is not None)
        handed[f"m{i}"] = 0
        handed[f"fat{i}"], handed[f"carb{i}"] = handed["fat_g"], handed["carb_g"]
        handed.loc[X.index[on], f"m{i}"] = 1
        handed.loc[X.index, f"fat{i}"] = moved_x["fat_g"].to_numpy()
        handed.loc[X.index, f"carb{i}"] = moved_x["carb_g"].to_numpy()
    (tmp_path / "r").mkdir()
    np.savetxt(tmp_path / "r" / "ks.txt", ks)
    r = run_r(SURVEY_R, {"f": handed}, tmp_path / "r")
    band = sub["band"]
    assert band["method"] == "design" and band["n_boot"] == 0 and band["df"] == r["degf"] == 15
    q = stats.t.ppf(0.975, band["df"])
    live = [i for i in range(len(ks)) if classes["high"]["delta"][i] is not None and ks[i] > 0]
    assert len(live) >= 4
    for level in CRP_CLASSES:
        entry = classes[level]
        for i in live:
            key = f"{level}_{i + 1}"
            assert entry["delta"][i] == pytest.approx(r["est"][key], rel=1e-8), key
            se = (entry["ci_high"][i] - entry["delta"][i]) / q
            assert se == pytest.approx(r["se"][key], rel=1e-6), key
            assert entry["ci_low"][i] == pytest.approx(entry["delta"][i] - q * se, rel=1e-9)
            assert abs(se / r["jk"][key] - 1) < 0.05, (key, se, r["jk"][key])
    assert_sums_to_zero(classes)
    assert sub["basis"] == (f"Averaged over the {len(analyzed):,} analyzed rows in the survey "
                            f"design, each counted as the people its weight `WTDRD1` stands for.")
    assert ("The 200 bootstrap refits asked for are not drawn: resampling rows ignores the strata "
            "and PSUs, so under the surveyed population the band is the design's.") in sub["note"]
    assert "Taylor linearization over the survey design" in band["caption"]


def test_3_multiple_imputation_under_the_surveyed_population_pools_design_based_curves(
        crp, tmp_path):
    """Both at once (MS2-MS4): each completed copy's class curves are the surveyed population's,
    each copy's variance its design-based linearization, pooled at each k by Rubin's rules with the
    design's degrees of freedom as the complete-data df, so no interval's ν exceeds the design's 15.
    The pooled curve is the mean of the copies' design-based curves (checked against each copy's
    own weighted NumPy fit, 1e-7)."""
    from turbotab.core.models.survey import domain_of

    f = crp.copy()
    rng = np.random.default_rng(5)
    f.loc[rng.random(len(f)) < 0.08, "protein_g"] = np.nan
    f.loc[rng.random(len(f)) < 0.05, "age"] = np.nan
    out = survey_run(tmp_path, f, models=["linear"],
                     missing=MissingSpec(strategy="multiple_imputation"))
    sub, fit = out["sub"], out["fit"]
    imputed = fit.objects["imputations"]
    copies = imputed["frames"]
    design = fit.objects["survey_design"]
    assert sub["band"]["method"] == "design"
    classes = by_class(sub)
    y = f["crp_class"].to_numpy()[out["analyzed"]]
    weight = f["WTDRD1"].to_numpy(dtype=float)[out["analyzed"]]
    means = []
    for copy in copies:
        domain = domain_of(copy.index, design)
        assert domain.keep.all()
        Z = np.column_stack([copy[c].to_numpy(dtype=float) for c in
                             ("protein_g", "fat_g", "carb_g", "energy_kcal", "age", "female")])
        proba = fit_weighted(Z, y, weight)
        means.append(weighted_curves(proba, copy, sub["ks"], weight, sub))
    for c, level in enumerate(CRP_CLASSES):
        entry = classes[level]
        assert entry["pooled"] == "per_k"
        for i, k in enumerate(sub["ks"]):
            if entry["delta"][i] is None or k == 0:
                continue
            assert abs(entry["delta"][i] - np.mean([mean[i, c] for mean in means])) <= 1e-7
            assert 0 < entry["df"][i] <= 15 + 1e-9
            assert entry["ci_low"][i] < entry["delta"][i] < entry["ci_high"][i]
    assert_sums_to_zero(classes)
    assert ("Each class's curve is pooled over the 20 imputations: each copy's survey-weighted "
            "class curves over the surveyed population, pooled at each k by Rubin's rules with "
            "each copy's Taylor-linearized variance as its within-copy variance and the design's "
            "degrees of freedom as the complete-data df.") in sub["note"]


def fit_weighted(Z: np.ndarray, y: np.ndarray, weight: np.ndarray) -> Any:
    """The NumPy multinomial fit with survey weights, on the CRP classes."""
    return numpy_multinomial(Z, y, weight / weight.mean(), classes=CRP_CLASSES)


def weighted_curves(proba: Any, copy: pd.DataFrame, ks: list[float], weight: np.ndarray,
                    sub: dict) -> np.ndarray:
    """A copy's weighted class curves (k × class) over the app's own support for that copy."""
    from turbotab.core.methods.substitution import Shift

    X = copy[["protein_g", "fat_g", "carb_g", "energy_kcal", "age", "female"]]
    shift = Shift(X, donor="fat_g", recipient="carb_g", kcal_per_unit={"fat_g": 9.0, "carb_g": 4.0},
                  total="energy_kcal")
    base = proba(X.to_numpy(dtype=float))
    out = np.full((len(ks), 3), np.nan)
    for i, k in enumerate(ks):
        moved_x, on = shift.apply(X, float(k))
        if not on.any():
            continue
        diff = proba(moved_x.to_numpy(dtype=float)) - base
        out[i] = (weight[on] @ diff[on]) / weight[on].sum()
    return out


# ── 3 · the repair: under inference no curve follows one fill ────────────────
#
# The verifier (MS3; the contract's own pooled_per_k: "a curve from one fill is never shown"): with
# the multiple-imputation answer blocked and recorded, the coefficient table was blocked with its
# exits while the class curves were still served from one outcome-free fill with a bootstrap band,
# under a note saying the table used multiple imputations it did not.

# The table's refusal, word for word (MODELING_SEQUENCE §4, "passive MI with a declared nonlinear
# term": block and record; a multiclass outcome has no compatible imputation built here).
PASSIVE_REFUSAL = (
    "Under inference passive multiple imputation with a restricted cubic spline of `fat_g` is "
    "blocked until it is recorded: passive imputation draws each value from a model linear in it "
    "and only then derives the declared nonlinear terms, so the curvature and the nonlinearity "
    "tests would be biased toward the null (Bartlett et al. 2015). No imputation compatible with "
    "this outcome's model is built here, so the copies would be passive.")
# What the substitution note says when the answer holds every curve, word for word.
HELD_NOTE = ("No curve is drawn while the missing-values answer is blocked: the curve reads the "
             "same rows as the coefficient table, and on them it would follow one fill of their "
             "blanks, which under inference is never shown (MODELING_SEQUENCE §2).")
NO_COPIES_NOTE = ("No curve is drawn: under multiple imputation every estimate shown under "
                  "inference is pooled over the imputed copies, and a curve on one fill of the "
                  "blanks would leave out the imputations' uncertainty (MODELING_SEQUENCE §2).")
COMPLETE_CASES = "Complete cases, with their assumption stated"
KEEP = "Keep it, recorded as a limitation"
SPLINE = {"fat_g": ExposureFormSpec(form="spline", knots=4)}


def one_thread() -> Any:
    """Boosted trees' OpenMP threads held to one while these runs fit it: on a machine other jobs
    share, two threads spin against each other (a 400-row fit measured at 12 s with two, 0.1 s
    with one). The numbers do not depend on it: the app's run and the reference refit both run
    under it."""
    from threadpoolctl import threadpool_limits

    return threadpool_limits(limits=1, user_api="openmp")


def blank_columns(frame: pd.DataFrame) -> list[str]:
    """The model inputs with a blank, counted from the table (in the roles' order)."""
    return [c for c in ROLES if frame[c].isna().any()]


def as_yes_no(frame: pd.DataFrame) -> pd.DataFrame:
    """The same table with a yes/no outcome (``high`` or not): the one-curve path."""
    return frame.assign(y=np.where(frame["y"] == "high", "yes", "no"))


def frame_curves(predict: Any, frame: pd.DataFrame, ks: list[float], masks: list, live: list,
                 fixed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """:func:`numpy_curves` through any model of the raw inputs (``predict``: a frame → its
    n × class probabilities)."""
    base = np.asarray(predict(frame), dtype=float)
    curve = np.full((len(ks), base.shape[1]), np.nan)
    fixed_curve = np.full_like(curve, np.nan)
    for i, k in enumerate(ks):
        if not live[i]:
            continue
        diff = np.asarray(predict(moved(frame, k)), dtype=float) - base
        curve[i] = diff[masks[i]].mean(axis=0)
        fixed_curve[i] = diff[fixed].mean(axis=0)
    return curve, fixed_curve


def assert_held(out: dict, refused: str, exits: list[dict], note: str, n_families: int) -> None:
    """Every family's curve is blocked and recorded with ``refused`` and ``exits`` (one entry per
    family, no class, no point, no band), and the note says so: ``note`` then the refusal."""
    sub = out["sub"]
    assert len(sub["models"]) == n_families
    for entry in sub["models"]:
        assert entry["level"] is None and entry["refused"] == refused, entry["label"]
        assert entry["exits"] == exits
        assert all(v is None for v in entry["delta"])
        assert all(v is None for v in entry["fixed_delta"])
        assert entry["ci_low"] is None and entry["fixed_ci_low"] is None
        assert entry["pooled"] is None and entry["effect_label"] is None
    assert sub["band"] is None and sub["band_estimate"] is None
    assert f"{note} {refused}" in sub["note"]
    for said in ("filled once", "not pooled over", "Each class's curve is pooled over",
                 "The curve is pooled over", "shaded bands", "bootstrap"):
        assert said not in sub["note"], said
    for e in exits:
        if e["decision"] is not None:
            parse_decision(e["decision"])  # each way forward is a decision the app takes


@pytest.fixture(scope="module")
def passive_blocked(tmp_path_factory):
    """Passive imputation with a spline on fat under the standard energy model (the verifier's
    case), a band asked for, two families chosen."""
    frame = diet_classes(seed=41, n=500, blanks=True)
    with one_thread():
        out = run(tmp_path_factory.mktemp("passive"), frame, method="standard", forms=SPLINE,
                  missing=MissingSpec(strategy="multiple_imputation"), n_boot=20,
                  models=("linear", "boosted_trees"))
    return {"frame": frame, **out}


def test_3_a_blocked_imputation_blocks_every_class_curve_with_the_tables_refusal_and_exits(
        passive_blocked):
    """MODELING_SEQUENCE §4, "passive MI with a declared nonlinear term": block and record under
    inference. The coefficient table is blocked with its exits (complete cases; keep it, recorded),
    and so is every family's set of class curves, with the same refusal and the same exits, word
    for word: no curve, no band from one outcome-free fill, no claim of multiple imputations."""
    fit = passive_blocked["fit"]
    table = next(m for m in fit.data["models"] if m["family"] == "linear")["inference"]
    assert table["refused"] == PASSIVE_REFUSAL
    exits = table["exits"]
    assert [e["label"] for e in exits] == [COMPLETE_CASES, KEEP]
    assert fit.objects["imputations"] is None  # no copy was drawn
    assert_held(passive_blocked, PASSIVE_REFUSAL, exits, HELD_NOTE, n_families=2)
    sub = passive_blocked["sub"]
    assert [m["label"] for m in sub["models"]] == ["Linear model", "Boosted trees"]
    assert sub["support"]["n_rows"] == 500


def test_3_a_the_recorded_exit_draws_class_curves_pooled_over_the_passive_copies(passive_blocked,
                                                                                 tmp_path):
    """The exit "Keep it, recorded as a limitation", taken: the passive copies are drawn and each
    class's curve is pooled over them at each k, through each copy's own fit. Reference: each
    copy's model matrix (the fit's spline basis on that copy, knots fixed across copies) refit by
    the NumPy multinomial fit, its curves averaged over the support written out by hand, the pooled
    curve their mean over the copies (bound 1e-8, as item 1's)."""
    from turbotab.core.models.linear import model_matrix

    exits = next(m for m in passive_blocked["fit"].data["models"]
                 if m["family"] == "linear")["inference"]["exits"]
    kept = next(e["decision"] for e in exits if e["label"] == KEEP)
    assert kept["acknowledged"] is True and kept["strategy"] == "multiple_imputation"
    frame = passive_blocked["frame"]
    out = run(tmp_path, frame, method="standard", forms=SPLINE,
              missing=MissingSpec(**{k: v for k, v in kept.items() if k != "kind"}))
    sub, imputed = out["sub"], out["fit"].objects["imputations"]
    classes = by_class(sub)
    assert sorted(classes) == CLASSES and all(e.get("refused") is None for e in sub["models"])
    copies, fits = imputed["frames"], imputed["fits"]["linear"]
    m = len(copies)
    assert m == 20
    y = frame["y"].to_numpy()
    ks = sub["ks"]
    curves, lives = [], []
    for copy, fitted in zip(copies, fits):
        columns = list(model_matrix(fitted, copy).columns)
        assert any("fat_g" in str(c) for c in columns) and len(columns) > 5  # the spline's basis
        proba = numpy_multinomial(model_matrix(fitted, copy).to_numpy(dtype=float), y)
        masks, live, fixed = numpy_support(copy, ks)
        curve, _ = frame_curves(lambda f, _f=fitted, _p=proba: _p(
            model_matrix(_f, f).to_numpy(dtype=float)), copy, ks, masks, live, fixed)
        curves.append(curve)
        lives.append(live)
    live = [all(lv[i] for lv in lives) for i in range(len(ks))]
    Q = np.asarray(curves)
    for c, level in enumerate(CLASSES):
        assert classes[level]["pooled"] == "per_k"
        for i in range(len(ks)):
            if live[i]:
                assert abs(classes[level]["delta"][i] - Q[:, i, c].mean()) <= 1e-8, (level, i)
            else:
                assert classes[level]["delta"][i] is None
    assert_sums_to_zero(classes)
    assert (f"Each class's curve is pooled over the {m} imputations: the mean at each k of each "
            f"copy's class curves, with no band until one is asked for.") in sub["note"]


@pytest.fixture(scope="module")
def cannot_run(tmp_path_factory):
    """Imputation that cannot run: `age` recorded on one row only (a covariate with nothing to
    impute from), with two families, as one curve per class and as one curve for a yes/no outcome;
    and the verifier's own case, a covariate (`bmi`) blank on every row, as one curve per class."""
    frame = diet_classes(seed=43, n=400, blanks=True)
    frame.loc[frame.index[1:], "age"] = np.nan
    blank = diet_classes(seed=43, n=400, blanks=True).assign(bmi=np.nan)
    folder = tmp_path_factory.mktemp("cannot")
    mi = MissingSpec(strategy="multiple_imputation")
    with one_thread():
        return {"frame": frame, "blank": blank,
                "classes": run(folder / "classes", frame, missing=mi, n_boot=20,
                               models=("linear", "boosted_trees")),
                "yes_no": run(folder / "yes_no", as_yes_no(frame), missing=mi, n_boot=20,
                              models=("linear", "boosted_trees"), task="binary"),
                "all_blank": run(folder / "all_blank", blank, missing=mi, n_boot=20,
                                 models=("linear", "boosted_trees"),
                                 roles={**ROLES, "bmi": "covariate"})}


def test_3_imputation_that_cannot_run_blocks_one_curve_per_class_and_one_curve_alike(cannot_run):
    """With `age` recorded once (counted here: one observed value, so no imputation model can be
    fit for it), multiple imputation cannot run; the table is blocked, and every family's curves
    take that refusal and its exits, word for word, for the multiclass outcome and for the yes/no
    outcome (the one-curve path the multiclass path inherits from). With `bmi` blank on every row
    (counted here) the same holds, under its own refusal. Complete cases were the exit; they would
    keep one row and no row, which the record refuses (``row_floor``), so they are not offered
    (the engine residue's verifier: every exit offered must be accepted). Leaving the column out,
    the copies drawn without it, is offered instead."""
    frame = cannot_run["frame"]
    assert int(frame["age"].notna().sum()) == 1
    assert int(cannot_run["blank"]["bmi"].notna().sum()) == 0
    once = ("Multiple imputation cannot run on these data: `age` has fewer than two observed "
            "values, so it cannot be imputed.")
    never = ("Multiple imputation cannot run on these data: `bmi` has no recorded value to start "
             "its imputations from.")
    for name, refused in (("classes", once), ("yes_no", once), ("all_blank", never)):
        out = cannot_run[name]
        table = next(m for m in out["fit"].data["models"] if m["family"] == "linear")["inference"]
        assert table["refused"] == refused, name
        exits = table["exits"]
        column = "bmi" if name == "all_blank" else "age"
        assert [e["label"] for e in exits] == [f"Leave `{column}` out"], name
        assert exits[0]["decision"]["strategy"] == "multiple_imputation"
        assert exits[0]["decision"]["drop_columns"] == [column]
        assert_held(out, refused, exits, HELD_NOTE, n_families=2)


@pytest.fixture(scope="module")
def no_copies(tmp_path_factory):
    """Boosted trees alone under multiple imputation: no chosen family has a coefficient table, so
    the fit draws no copies; as one curve per class and as one curve."""
    frame = diet_classes(seed=47, n=400, blanks=True)
    folder = tmp_path_factory.mktemp("no_copies")
    mi = MissingSpec(strategy="multiple_imputation")
    with one_thread():
        return {"frame": frame,
                "classes": run(folder / "classes", frame, missing=mi, models=("boosted_trees",)),
                "yes_no": run(folder / "yes_no", as_yes_no(frame), missing=mi,
                              models=("boosted_trees",), task="binary")}


def test_3_with_no_imputed_copies_drawn_no_curve_is_drawn_on_one_fill(no_copies):
    """Under inference with the multiple-imputation answer, a family with no coefficient table is
    fit on one fill of the blanks; with no copies drawn there is nothing to pool its curves over,
    so they are blocked and recorded, word for word, the blank columns named (counted from the
    table here). The exits: the linear model added (its table draws the copies; the next test takes
    it), and complete cases."""
    frame = no_copies["frame"]
    gaps = blank_columns(frame)
    assert gaps == ["protein_g", "kcal", "age"]
    refused = ("Under inference with multiple imputation each curve is pooled over the imputed "
               "copies, and none were drawn for this fit: no chosen family has a coefficient "
               "table, and the copies are drawn with the table. A curve on one fill of the blanks "
               "in `protein_g`, `kcal` and `age` would leave out the imputations' uncertainty.")
    for name in ("classes", "yes_no"):
        out = no_copies[name]
        assert out["fit"].objects["imputations"] is None
        exits = out["sub"]["models"][0]["exits"]
        assert exits[0] == {"label": "Add the linear model, whose coefficient table draws the "
                                     "imputed copies",
                            "decision": {"kind": "select_models",
                                         "models": ["boosted_trees", "linear"]}}
        assert exits[1]["label"] == COMPLETE_CASES
        assert exits[1]["decision"]["kind"] == "set_missing"
        assert exits[1]["decision"]["strategy"] == "complete_case"
        assert_held(out, refused, exits, NO_COPIES_NOTE, n_families=1)


@pytest.fixture(scope="module")
def refit_on_copies(tmp_path_factory):
    """The exit taken: the linear model added beside boosted trees, so the copies are drawn; as one
    curve per class and as one curve (the yes/no outcome)."""
    frame = diet_classes(seed=47, n=400, blanks=True)
    folder = tmp_path_factory.mktemp("refit")
    mi = MissingSpec(strategy="multiple_imputation")
    with one_thread():
        return {"frame": frame,
                "classes": run(folder / "classes", frame, missing=mi,
                               models=("boosted_trees", "linear")),
                "yes_no": run(folder / "yes_no", as_yes_no(frame), missing=mi,
                              models=("boosted_trees", "linear"), task="binary")}


def per_copy_trees(out: dict, y: np.ndarray, columns: list[int]) -> tuple:
    """Each copy's boosted-trees curves (the ``columns`` of its predicted probabilities), refit on
    the copy and averaged over the support written out by hand: (curves, fixed-population curves,
    the ks live on every copy), copies × k × column."""
    from sklearn.base import clone

    from turbotab.core.models.inner_cv import fit_pipeline

    imputed = out["fit"].objects["imputations"]
    assert "boosted_trees" not in (imputed["fits"] or {})  # the table's copies hold no fit of it
    template = out["design"].objects["pipelines"]["boosted_trees"]
    ks = out["sub"]["ks"]
    curves, fixed_curves, lives = [], [], []
    for copy in imputed["frames"]:
        with one_thread():
            model = fit_pipeline(clone(template), copy, y, groups=None)
            masks, live, fixed = numpy_support(copy, ks)
            curve, fixed_curve = frame_curves(lambda f, _m=model: _m.predict_proba(f)[:, columns],
                                              copy, ks, masks, live, fixed)
        curves.append(curve)
        fixed_curves.append(fixed_curve)
        lives.append(live)
    live = [all(lv[i] for lv in lives) for i in range(len(ks))]
    return np.asarray(curves), np.asarray(fixed_curves), live


def test_3_a_family_with_no_table_is_refit_on_each_copy_and_pooled_per_k(refit_on_copies):
    """Boosted trees has no coefficient table, so the table's copies hold no fit of it; its curves
    were drawn on one fill beside the pooled linear ones, under a note saying every curve was
    pooled. Now it is refit on each completed copy and pooled at each k, one curve per class and
    one curve alike. Reference: on each copy the family refit (the same estimator, same seed), each
    class's curve (or the yes/no outcome's) averaged over the support written out by hand, the
    pooled curve their mean over the copies (bound 1e-12); the note says the refit, word for
    word."""
    frame = refit_on_copies["frame"]
    out = refit_on_copies["classes"]
    sub, ks = out["sub"], out["sub"]["ks"]
    Q, F, live = per_copy_trees(out, frame["y"].to_numpy(), [0, 1, 2])
    assert sum(live) >= 4
    trees = by_class(sub, "boosted_trees")
    for c, level in enumerate(CLASSES):
        assert trees[level]["pooled"] == "per_k"
        for i in range(len(ks)):
            if not live[i]:
                assert trees[level]["delta"][i] is None
                continue
            assert abs(trees[level]["delta"][i] - Q[:, i, c].mean()) <= 1e-12, (level, i)
            assert abs(trees[level]["fixed_delta"][i] - F[:, i, c].mean()) <= 1e-12, (level, i)
    assert_sums_to_zero(trees)
    # The copies' curves differ (a real pooling, not one fill drawn twenty times).
    assert max(float(Q[:, i, 0].std()) for i in range(len(ks)) if live[i] and ks[i] > 0) > 1e-4
    assert all(e["pooled"] == "per_k" for e in by_class(sub, "linear").values())
    refit = ("Boosted trees has no coefficient table of its own, so it was refit on each completed "
             "copy.")
    assert refit in sub["note"]
    assert ("Each class's curve is pooled over the 20 imputations: the mean at each k of each "
            "copy's class curves, with no band until one is asked for.") in sub["note"]
    yes_no = refit_on_copies["yes_no"]["sub"]
    Q, F, live = per_copy_trees(refit_on_copies["yes_no"],
                                as_yes_no(frame)["y"].to_numpy(), [1])  # P(yes)
    (entry,) = [m for m in yes_no["models"] if m["family"] == "boosted_trees"]
    assert entry["pooled"] == "per_k" and entry.get("refused") is None
    for i in range(len(ks)):
        if live[i]:
            assert abs(entry["delta"][i] - Q[:, i, 0].mean()) <= 1e-12, i
            assert abs(entry["fixed_delta"][i] - F[:, i, 0].mean()) <= 1e-12, i
        else:
            assert entry["delta"][i] is None
    assert refit in yes_no["note"]
    assert ("The curve is pooled over the 20 imputations: the mean at each k of each copy's "
            "curve, with no band until one is asked for.") in yes_no["note"]


@pytest.fixture(scope="module")
def lonely(crp, tmp_path_factory):
    """MS4's dietary table with each stratum's two PSUs merged into one: the analysis rows lie in
    as many PSUs as strata. One curve per class, one curve (the yes/no outcome), and the exit."""
    f = crp.assign(SDMVPSU=1)
    folder = tmp_path_factory.mktemp("lonely")
    return {"f": f,
            "classes": survey_run(folder / "classes", f, models=["linear"], n_boot=20),
            "yes_no": survey_run(folder / "yes_no", f, models=["linear"], n_boot=20,
                                 target="high_crp", task="binary"),
            "sample": survey_run(folder / "sample", f, models=["linear"], n_boot=20,
                                 survey={"estimand": "sample"})}


def test_3_a_design_with_no_degrees_of_freedom_draws_no_curve_and_names_the_exit(lonely):
    """The verifier's minor finding (MS4's one-curve design path, inherited): with the analysis
    rows' PSUs no more than their strata (counted from the table: 15 PSUs in 15 strata, so 0 design
    degrees of freedom), the coefficient table is refused, and the class curves were still drawn as
    weighted points with no band, under a caption promising 95% intervals on t(0). Now no curve is
    drawn, one curve per class or one curve, each blocked and recorded with the sample-only
    attestation as its exit, which the table now names too; the exit, taken, draws the curves for
    these participants with their refit band."""
    from turbotab.core.models.survey import SAMPLE_EXIT

    f = lonely["f"]
    held = f.loc[f["eligible"] == 1, ["SDMVSTRA", "SDMVPSU"]].drop_duplicates()
    psu, strata = len(held), held["SDMVSTRA"].nunique()
    assert (psu, strata) == (15, 15)
    why = (f"The analysis rows lie in {psu} PSUs of {strata} strata: no design degrees of freedom "
           f"are left for an interval.")
    exit_ = {"label": SAMPLE_EXIT, "decision": {"kind": "set_survey", "estimand": "sample"}}
    for name, prefix in (("classes", "No design-based curves: "),
                         ("yes_no", "No design-based curve: ")):
        out = lonely[name]
        table = out["fit"].data["models"][0]["inference"]
        assert table["refused"] == why and table["exits"] == [exit_], name
        sub = out["sub"]
        (entry,) = sub["models"]
        assert entry["refused"] == prefix + why and entry["exits"] == [exit_], name
        assert all(v is None for v in entry["delta"]) and entry["ci_low"] is None
        assert sub["band"] is None
        assert f"Linear model: no curve. {prefix}{why}" in sub["note"]
        assert "t(0)" not in sub["note"] and "Taylor linearization" not in sub["note"]
    sample = lonely["sample"]["sub"]
    classes = by_class(sample)
    assert sorted(classes) == CRP_CLASSES and all(e.get("refused") is None
                                                  for e in sample["models"])
    assert sample["band"]["n_boot"] == 20 and sample["band"]["method"] == "bootstrap"
    assert_sums_to_zero(classes)


# ── 4 · the label, and the energy model's rules ──────────────────────────────


def test_4_the_label_states_the_probability_scale_the_isocaloric_move_and_the_population(
        inference_run, population_run):
    """The estimand, word for word, under the sample's own rows and under the surveyed population;
    each class's label is the change in its probability per 100 kcal at the stated k; and the
    methods sentence of the substitution answer, word for word, both ways."""
    sub = inference_run["sub"]
    assert sub["estimand"] == (
        "For each class of y (high, low and mid), the average change in its predicted probability, "
        "on the probability scale, when k kcal move from carb_g to protein_g at the same total "
        "energy (isocaloric), with every other input, total energy included, left as it was, over "
        "the 1,200 analyzed rows, each counted once (an average over these participants, not over "
        "a population they were sampled from). The class curves sum to zero at every k, because "
        "each row's class probabilities sum to one.")
    k100 = sub["ks"].index(100.0)
    for level, entry in by_class(sub).items():
        value = entry["delta"][k100]
        assert entry["effect_label"] == (
            f"{np.format_float_positional(value, precision=3, unique=False, fractional=False, trim='-', sign=True)}"
            f" in the probability of {level} per 100 kcal at k = 100")
    population = population_run["sub"]
    n = len(population_run["analyzed"])
    assert population["estimand"] == (
        f"For each class of crp_class (average, high and low), the average change in its "
        f"predicted probability, on the probability scale, when k kcal move from fat_g to carb_g "
        f"at the same total energy (isocaloric), with every other input, total energy included, "
        f"left as it was, over the surveyed population (the {n:,} analyzed rows in the survey "
        f"design, each counted as the people its weight `WTDRD1` stands for). The class curves "
        f"sum to zero at every k, because each row's class probabilities sum to one.")
    said = voice.sentence_for(SetSubstitution(donor=DONOR, recipient=RECIPIENT, step_kcal=STEP,
                                              n_boot=200), inference_run["state"])
    assert said == (
        "The substitution studied is `carb_g` replaced by `protein_g`, in steps of `50` kcal with "
        "`kcal` held fixed; the outcome `y` has unordered classes, so there is one curve per "
        "class, each the average change in that class's predicted probability, and the curves sum "
        "to zero at every k; each class's band comes from `200` refits of each model on bootstrap "
        "resamples of every analyzed row, or, above `10,000` analyzed rows, of about `10,000` of "
        "them each, drawn as whole units where a unit has several rows, the band's spread then "
        "rescaled by √(m/n) to the full sample (an m-out-of-n bootstrap).")
    said = voice.sentence_for(SetSubstitution(donor="fat_g", recipient="carb_g", step_kcal=STEP,
                                              n_boot=200), population_run["state"])
    assert said == (
        "The substitution studied is `fat_g` replaced by `carb_g`, in steps of `50` kcal at the "
        "same total energy; the outcome `crp_class` has unordered classes, so there is one curve "
        "per class, each the average change in that class's predicted probability, and the "
        "curves sum to zero at every k; over the surveyed population each class's curve is the "
        "weighted mean of each participant's change in that class's probability under the "
        "survey-weighted fit, and its band comes from Taylor linearization over the survey design "
        "(the `200` bootstrap refits asked for are not drawn: a row bootstrap ignores the strata "
        "and PSUs).")


TWO_SOURCES = {"protein_g": "exposure", "carb_g": "exposure", "kcal": "energy", "age": "covariate"}


@pytest.fixture(scope="module")
def omitted(tmp_path_factory):
    """Fat (about a third of energy) left out of the model: protein and carbohydrate only, in the
    standard energy model (the all-components model refuses to leave a source out)."""
    frame = diet_classes(seed=31, n=700)
    share = float((frame["kcal"] - 4 * frame["protein_g"] - 4 * frame["carb_g"]).div(
        frame["kcal"]).mean())
    folder = tmp_path_factory.mktemp("omitted")
    kept = run(folder / "kept", frame, roles=TWO_SOURCES, acknowledged=True, method="standard")
    told = run(folder / "told", frame, roles=TWO_SOURCES, purpose="prediction", holdout=0.2,
               method="standard")
    return {"frame": frame, "share": share, "kept": kept, "told": told}


def omitted_refusal(out: dict, decision: Any) -> Refusal | None:
    from turbotab.core.datastore import DataStore

    store = DataStore(Path(out["paths"]["data"]), 1 << 30)
    try:
        ctx = {"state": out["state"], "store": lambda: store, "sealed": lambda: []}
        try:
            validate(decision, ctx)
        except Refusal as refused:
            return refused
        return None
    finally:
        store.close()


def test_4_omitted_energy_sources_block_and_record_under_inference_and_are_stated_under_prediction(
        omitted):
    """MODELING_SEQUENCE §4, "omitted energy sources in substitution": under inference, block and
    record above the stated share (5% of total energy); under prediction, state the concern. With
    fat out of the model (its share of energy counted here with NumPy), the multiclass swap is
    refused under inference with its exits (add fat to the model; keep the swap with the recorded
    attestation; choose another), the attestation is accepted, its curves are drawn and its
    sentence records the attestation; under prediction the swap is accepted and its note names
    what the model leaves out."""
    from turbotab.core.methods.energy import MAX_OMITTED_SHARE

    share = omitted["share"]
    assert share > MAX_OMITTED_SHARE
    kept = omitted["kept"]
    asked = {"kind": "set_substitution", "donor": DONOR, "recipient": RECIPIENT,
             "step_kcal": STEP}
    inference_state = kept["state"].model_copy(update={"substitution": None})
    refused = omitted_refusal({**kept, "state": inference_state}, asked)
    assert refused is not None and refused.code == "omitted_energy_sources"
    assert f"{share:.0%}" in str(refused)
    labels = [e["label"] for e in refused.exits]
    assert labels == ["Add `fat_g` to the model as a study factor",
                      "Keep this swap; the curve carries their confounding", "Choose another swap"]
    attested = parse_decision(refused.exits[1]["decision"])
    assert attested.acknowledged is True
    assert omitted_refusal({**kept, "state": inference_state}, attested) is None
    sub = kept["sub"]
    assert sorted(by_class(sub)) == CLASSES
    assert_sums_to_zero(by_class(sub))
    stated = (f"The model holds protein_g and carb_g; fat, alcohol and other energy make up the "
              f"rest of total energy, {share:.0%} of it on average")
    assert stated in sub["note"]
    assert sub["omitted_energy"]["mean_share"] == pytest.approx(share, rel=1e-9)
    said = voice.sentence_for(attested, inference_state)
    assert said.endswith("; it was kept although energy sources are missing from the model, so "
                         "the curve carries the confounding of the sources total energy holds as "
                         "one composite.")
    told = omitted["told"]
    prediction_state = told["state"].model_copy(update={"substitution": None})
    assert omitted_refusal({**told, "state": prediction_state}, asked) is None
    assert "fat, alcohol and other energy make up the rest of total energy" in told["sub"]["note"]


# ── the contract (BLUEPRINT §13) and its chain test ──────────────────────────


def test_contract_declares_every_part_of_section_13():
    """The multiclass curves' contract, in the one registry: slot, data scope, needs, routing (its
    question; each option labeled customary and sound for both purposes, with its rung), the leash,
    storyboard, the sentence's writer, the methods clause, and its relations (each conflict blocked
    and recorded with its exits; each relation naming the code that makes it fire). The declared
    scope ("model") is the one lockbox §06's test observes on the NumPy reference: row i's change
    in each class's probability moves when the outcome is shuffled."""
    import importlib

    from turbotab.core import contracts as C
    from turbotab.core.methods.substitution import CLASS_CONTRACT

    registry = C.contracts()
    c = registry[CLASS_CONTRACT]
    assert c.package == "MULTISUB" and c.slot == "evaluation" and c.scope == "model"
    assert c.decision == "set_substitution" and c.stage == "substitution"
    assert c.needs and c.question and len(c.storyboard) == 6 and c.sources
    assert c.leash == {"inference": "recommended", "prediction": "recommended"}
    assert [o.key for o in c.options] == ["class_curves", "reference_ratios"]
    for o in c.options:
        assert o.label and o.customary
        for purpose in C.PURPOSES:
            assert o.sound[purpose] and o.rung[purpose] in C.RUNGS
    for r in c.relations:
        if r.kind == "conflicts":
            assert r.rung == "block_and_record" and r.exits, r.name
        module, name = r.enforced_by.split(":")
        assert callable(getattr(importlib.import_module(module), name)), r.enforced_by
    module, name = c.sentence.split(":")
    assert callable(getattr(importlib.import_module(module), name))
    assert "turbotab.core.methods.substitution" in C.DECLARING_MODULES
    import turbotab.core.methods.substitution as S

    for shape in ("MethodContract", "Relation", "register_contract", "CONTRACTS"):
        assert not hasattr(S, shape), shape
    C.run_order(list(registry))
    frame = diet_classes(seed=3, n=240)
    Z = frame[["protein_g", "carb_g", "fat_g", "kcal", "age"]]

    def changes(f: pd.DataFrame, reference: pd.Series, yy: np.ndarray) -> np.ndarray:
        proba = numpy_multinomial(all_components(f), yy)
        return proba(all_components(moved(f, 50.0))) - proba(all_components(f))

    observed = C.observed_scope(changes, Z, np.zeros(len(Z), dtype=bool), frame["y"].to_numpy(), 4,
                                columns=["protein_g", "carb_g", "fat_g", "kcal", "age"])
    assert observed == "model" == c.scope
    assert C.paragraph({CLASS_CONTRACT: "class_curves"},
                       {"donor": DONOR, "recipient": RECIPIENT, "step_kcal": STEP, "n_boot": 200},
                       "inference") == (
        "For the multiclass outcome, one substitution curve per class was drawn, the average "
        "change in that class's predicted probability as energy moved from `carb_g` to "
        "`protein_g` in steps of 50 kcal at the same total energy, the curves summing to zero at "
        "every k; each class's band came from 200 refits on bootstrap resamples.")
    assert C.paragraph({"survey_population": "population", CLASS_CONTRACT: "class_curves"},
                       {"weight": "WTDRD1", "strata": "SDMVSTRA", "psu": "SDMVPSU",
                        "estimand": "population", "donor": "fat_g", "recipient": "carb_g",
                        "step_kcal": STEP, "m": 20}, "inference") == (
        "The estimates describe the surveyed population: rows were weighted by `WTDRD1`, and "
        "standard errors were estimated by Taylor series linearization over `SDMVPSU` nested "
        "within `SDMVSTRA`, with t intervals on the PSUs minus the strata that hold the analysis "
        "rows; for the multiclass outcome, one substitution curve per class was drawn, the "
        "average change in that class's predicted probability as energy moved from `fat_g` to "
        "`carb_g` in steps of 50 kcal at the same total energy, the curves summing to zero at "
        "every k; each class's curve came from the survey-weighted multinomial fit, averaged with "
        "the weights, with its band by Taylor linearization over the survey design (Graubard & "
        "Korn 1999), pooled over 20 imputations by Rubin's rules at each k.")


@pytest.fixture(scope="module")
def blocked_run(crp, tmp_path_factory):
    """Under the surveyed population with a family that has no design-based estimator chosen too."""
    return survey_run(tmp_path_factory.mktemp("blocked"), crp, models=["linear", "boosted_trees"])


def test_chain_every_relation_the_contract_declares_fires(inference_run, imputed_run,
                                                         population_run, blocked_run, omitted,
                                                         passive_blocked, no_copies,
                                                         refit_on_copies, lonely):
    """The chain test (BLUEPRINT §13): each relation the contract declares, observed on the run it
    governs. A relation added to the contract without a check here fails."""
    from turbotab.core.contracts import contract, fired
    from turbotab.core.methods.substitution import CLASS_CONTRACT
    from turbotab.core.models.survey import SAMPLE_EXIT

    declared = contract(CLASS_CONTRACT)

    def curves_sum_to_zero() -> bool:
        for out in (inference_run, imputed_run, population_run, omitted["kept"], omitted["told"]):
            assert_sums_to_zero(by_class(out["sub"]))
        return True

    def refit_band() -> bool:
        band = inference_run["sub"]["band"]
        return (band["n_boot"] == 200 and band["method"] == "bootstrap"
                and all(e["band_ok"] == 200 for e in by_class(inference_run["sub"]).values()))

    def pooled_per_k() -> bool:
        entries = by_class(imputed_run["sub"]).values()
        trees = by_class(refit_on_copies["classes"]["sub"],
                         "boosted_trees").values()  # refit on each copy
        return (all(e["pooled"] == "per_k" and e["df"] is not None for e in entries)
                and all(e["pooled"] == "per_k" for e in trees)
                and "Each class's curve is pooled over the 20 imputations" in imputed_run["sub"]["note"])

    def blocked_with_the_table() -> bool:
        fit, sub = passive_blocked["fit"], passive_blocked["sub"]
        table = next(m for m in fit.data["models"] if m["family"] == "linear")["inference"]
        held = no_copies["classes"]["sub"]["models"]
        return (all(e["refused"] == table["refused"] and e["exits"] == table["exits"]
                    for e in sub["models"])
                and sub["note"].count(table["refused"]) == 1
                and all(e["exits"][0]["decision"] == {"kind": "select_models",
                                                      "models": ["boosted_trees", "linear"]}
                        for e in held))

    def no_design_df() -> bool:
        from turbotab.core.models.survey import SAMPLE_EXIT

        entries = lonely["classes"]["sub"]["models"]
        return (len(entries) == 1 and entries[0]["refused"].endswith(
                    "no design degrees of freedom are left for an interval.")
                and [e["label"] for e in entries[0]["exits"]] == [SAMPLE_EXIT]
                and lonely["classes"]["sub"]["band"] is None)

    def design_based() -> bool:
        sub = population_run["sub"]
        return (sub["band"]["method"] == "design" and sub["band"]["df"] == 15
                and all(e["ci_low"] is not None for e in by_class(sub).values())
                and "200 bootstrap refits asked for are not drawn" in sub["note"])

    def blocked_family() -> bool:
        sub = blocked_run["sub"]
        trees = [m for m in sub["models"] if m["family"] == "boosted_trees"]
        assert len(trees) == 1 and all(v is None for v in trees[0]["delta"])
        exits = trees[0]["exits"]
        return (trees[0]["refused"].startswith("Boosted trees has no design-based estimator, so "
                                               "its substitution curves would describe these "
                                               "participants")
                and exits[0] == {"label": "Use survey-weighted multinomial logistic regression",
                                 "decision": {"kind": "select_models", "models": ["linear"]}}
                and exits[1] == {"label": SAMPLE_EXIT,
                                 "decision": {"kind": "set_survey", "estimand": "sample"}}
                and sorted(by_class(sub)) == CRP_CLASSES
                and f"{trees[0]['label']}: no curve." in sub["note"])

    def omitted_sources() -> bool:
        kept = omitted["kept"]
        state = kept["state"].model_copy(update={"substitution": None})
        refused = omitted_refusal({**kept, "state": state},
                                  {"kind": "set_substitution", "donor": DONOR,
                                   "recipient": RECIPIENT, "step_kcal": STEP})
        relation = declared.relation("omitted_sources")
        labels = [e["label"] for e in refused.exits]
        return (refused.code == "omitted_energy_sources" and labels[1:] == list(relation.exits[1:])
                and labels[0].startswith("Add `fat_g`"))

    def omitted_stated() -> bool:
        return ("fat, alcohol and other energy make up the rest of total energy"
                in omitted["told"]["sub"]["note"])

    def estimand_label() -> bool:
        for out in (inference_run, population_run):
            text = out["sub"]["estimand"]
            assert "on the probability scale" in text and "(isocaloric)" in text
        return ("over the surveyed population" in population_run["sub"]["estimand"]
                and "analyzed rows, each counted once" in inference_run["sub"]["estimand"])

    checks = {"curves_sum_to_zero": curves_sum_to_zero, "refit_band": refit_band,
              "pooled_per_k": pooled_per_k, "blocked_with_the_table": blocked_with_the_table,
              "design_based": design_based, "blocked_family": blocked_family,
              "no_design_df": no_design_df, "omitted_sources": omitted_sources,
              "omitted_stated": omitted_stated, "estimand_label": estimand_label}
    assert set(checks) == {r.name for r in declared.relations}
    failed = [name for name, check in checks.items() if not check()]
    assert failed == [], failed
    under = {f.relation.name for f in fired({CLASS_CONTRACT: "class_curves",
                                             "multiple_imputation_compatible": "compatible",
                                             "survey_population": "population"}, "inference")}
    assert {"pooled_per_k", "design_based", "blocked_family", "no_design_df"} <= under
    held = {f.relation.name for f in fired({CLASS_CONTRACT: "class_curves"}, "inference",
                                           consequences=("imputation_blocked",))}
    assert "blocked_with_the_table" in held
    assert "blocked_with_the_table" not in {
        f.relation.name for f in fired({CLASS_CONTRACT: "class_curves"}, "prediction",
                                       consequences=("imputation_blocked",))}
