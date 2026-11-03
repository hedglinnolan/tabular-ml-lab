"""The censored-normal (Tobit) fit returns a certified maximum or refuses (the verifier's FWMI-R2-1
and FWMI-R2-2, the orchestrator's ruling of 2026-11-03).

``imputation.tobit_fit`` fed every censored draw (the feature-wise copies, the chained equations'
censored model) and the single censoring-aware fill. It ran BFGS from σ = 1e−6 where the detected
values were fitted exactly and never checked that it had converged: on the verifier's designs with
3 or 4 detected values of 100 and four terms it stopped short of the maximum (σ ≈ 2.6e−6 against
survreg's 0.385 on design 335, whose pooled row then read as a discovery) or raised OverflowError
(cases 17 and 232), which crashed the effects stage. Now it fits by Newton's method in Olsen's
concave parametrization and returns only a point it certifies: the score zero and the information
positive definite there; anything else, its arithmetic included, is one typed failure
(``imputation.CensoredFitRefused``), which every caller refuses in words.

The truths, each independent of the engine's output:

* ``survival::survreg``'s left-censored Gaussian fit (3.8) on the same design: coefficients, scale
  and standard errors, on every one of the verifier's designs whose model has a finite maximum
  (an aliased column NA in survreg, 0 with no variance here);
* where the model has no finite maximum (the verifier's other designs: detected only in rows of one
  outcome, or fitted exactly with every value below detection under the limit, each read by
  ``featurewise_mi.identification_refusal``'s linear programs and derivations; a 0/1 column whose
  ones or whose zeros are all below detection, raw or standardized, by a derivation checked with
  scipy's normal, FWMI-R3-1), the refusal, through the joint route's fit and declared models too;
* the censored normal's location–scale equivariance (a derivation): the fit of a z + b is the fit
  of z moved and scaled;
* an intercept and equal detected values have no finite maximum (σ → 0 raises the likelihood
  without bound: a derivation), and the single fill refuses them in words.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import turbotab.core.models  # noqa: F401 - registers the families
from turbotab.core.tests.acceptance.r_reference import needs_r, run_r

SURVREG_EACH = """
suppressPackageStartupMessages(library(survival))
d <- read.csv(designs_csv)
res <- list()
for (i in unique(d$design)) {
  s <- d[d$design == i, ]
  X <- as.matrix(s[, grep("^D", names(s))])
  f <- survreg(Surv(s$y, s$det == 1, type = "left") ~ X, dist = "gaussian",
               control = survreg.control(maxiter = 500))
  co <- unname(coef(f))
  v <- vcov(f)
  se <- rep(0, ncol(v))
  ok <- which(diag(v) > 0)
  se[ok] <- sqrt(diag(v)[ok])
  res[[as.character(i)]] <- list(coef = ifelse(is.na(co), 0, co), scale = f$scale,
                                 iter = f$iter[1], se = se)
}
out(res)
"""


def verifier_columns(seed: int = 21, count: int = 600, low: int = 2, high: int = 13):
    """The verifier's designs (FWMI-R2-1, its ``solver_wide.py``): 100 rows, the event (55%) and
    two covariates; x = 5 + 0.6 event + 0.4 z0 − 0.3 z1 + N(0, 1 or 0.2²), its ``low`` to
    ``high`` − 1 largest values detected, half of them positive (exp(x)) and fit on their log
    scale. Each: (index, the column with its values below detection blank, on its own scale, the
    event, the covariates, whether it is fit on its log scale)."""
    rng = np.random.default_rng(seed)
    for i in range(count):
        n = 100
        event = (rng.random(n) < 0.55).astype(float)
        Z = rng.standard_normal((n, 2))
        log_scale = bool(rng.random() < 0.5)
        x = 5 + 0.6 * event + Z @ np.array([0.4, -0.3]) + rng.standard_normal(n) * rng.choice(
            [1.0, 0.2])
        n_detected = int(rng.integers(low, high))
        detected = np.zeros(n, dtype=bool)
        detected[np.argsort(-x)[:n_detected]] = True
        if log_scale:
            x = np.exp(x)
        yield i, np.where(detected, x, np.nan), event, Z, log_scale


def verifier_designs(seed: int = 21, count: int = 600):
    """:func:`verifier_columns` as the censored-normal fit reads each: (index, the engine's design
    D, the column on its fitted scale with the rows below detection at the limit, which rows are
    below, the limit, the number detected, the identification check's refusal or None)."""
    from turbotab.core.methods import featurewise_mi as fm

    for i, x, event, Z, log_scale in verifier_columns(seed, count):
        below = ~np.isfinite(x)
        z = np.log(x) if log_scale else x
        limit = float(np.nanmin(z))
        reason = fm.identification_refusal("x", z, below, event, Z, limit=limit)
        yield (i, fm._design(event, Z), np.where(below, limit, z), below, limit,
               int((~below).sum()), reason)


def _frame(designs: list) -> pd.DataFrame:
    blocks = []
    for i, D, y, below, _limit, _nd, _reason in designs:
        block = pd.DataFrame({f"D{j}": D[:, j] for j in range(1, D.shape[1])})
        block["y"], block["det"], block["design"] = y, (~below).astype(int), i
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


@needs_r
def test_1_the_fit_is_survregs_where_the_model_has_a_maximum_and_refused_where_it_has_none(
        tmp_path):
    """FWMI-R2-1 both ways, on the verifier's 600 designs. Where the identification check passes
    (the model has one finite maximum), the engine's fit is survreg's on the same design: each
    coefficient, the scale and each standard error, the 3- and 4-detected designs the base fit
    short of the maximum (design 335: σ 2.7e−6 against 0.385) or overflowed on among them, and the
    certification silent on every design with more detected values than terms. Where it refuses
    for no finite maximum, the solver refuses too, never returning a point (the base returned one
    on every such design)."""
    from turbotab.core.methods.imputation import CensoredFitRefused, tobit_fit

    fitted, refused = [], 0
    for design in verifier_designs():
        _i, D, y, below, limit, _nd, reason = design
        if reason is None:
            fitted.append(design)
            continue
        assert "no finite fit" in reason  # never tied terms: every design has full rank
        with pytest.raises(CensoredFitRefused):
            tobit_fit(D, y, below, limit)
        refused += 1
    few = [d for d in fitted if d[5] <= 4]  # no more detected values than its four terms
    assert refused > 100 and len(few) > 30 and len(fitted) - len(few) > 200
    ref = run_r(SURVREG_EACH, {"designs": _frame(fitted)}, tmp_path)
    for i, D, y, below, limit, n_detected, _reason in fitted:
        r = ref[str(i)]
        assert r["iter"] < 500, i  # survreg converged
        theta, V = tobit_fit(D, y, below, limit)
        assert theta[:-1] == pytest.approx(r["coef"], rel=1e-5, abs=1e-6), (i, n_detected)
        assert math.exp(theta[-1]) == pytest.approx(r["scale"], rel=1e-5), (i, n_detected)
        assert np.sqrt(np.diag(V)) == pytest.approx(r["se"], rel=1e-4), (i, n_detected)


def overflow_cases() -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """The verifier's cases 17 and 232 (FWMI-R2-2, its ``solver_e2e.py``, seed 5): 100 rows, the
    event and two covariates, 4 values detected; each (x with its values below detection blank, the
    event, the covariates, a complete second study factor)."""
    rng = np.random.default_rng(5)
    out = {}
    for i in range(300):
        n = 100
        event = (rng.random(n) < 0.55).astype(float)
        Z = rng.standard_normal((n, 2))
        x = 5 + 0.6 * event + Z @ np.array([0.4, -0.3]) + rng.standard_normal(n) * rng.choice(
            [1.0, 0.2])
        n_detected = int(rng.integers(2, 9))
        detected = np.zeros(n, dtype=bool)
        detected[np.argsort(-x)[:n_detected]] = True
        if i in (17, 232):
            good = 5 + 0.6 * event + Z @ np.array([0.4, -0.3]) + np.random.default_rng(
                i).standard_normal(n)
            out[i] = (np.where(detected, x, np.nan), event, Z, good)
    return out


@needs_r
def test_2_cases_17_and_232_are_fit_to_survregs_maximum_where_the_solver_overflowed(tmp_path):
    """FWMI-R2-2's cases: survreg converges (6 and 7 iterations, scales 0.4476 and 0.4765), and the
    base solver raised OverflowError (``math.exp(tau)``). Now each is fit to survreg's maximum."""
    from turbotab.core.methods import featurewise_mi as fm
    from turbotab.core.methods.imputation import tobit_fit

    designs = []
    for i, (x, event, Z, _good) in overflow_cases().items():
        below = ~np.isfinite(x)
        limit = float(np.nanmin(x))
        designs.append((i, fm._design(event, Z), np.where(below, limit, x), below, limit, 4,
                        None))
    ref = run_r(SURVREG_EACH, {"designs": _frame(designs)}, tmp_path)
    for i, D, y, below, limit, _nd, _r in designs:
        theta, V = tobit_fit(D, y, below, limit)
        assert ref[str(i)]["iter"] < 500
        assert theta[:-1] == pytest.approx(ref[str(i)]["coef"], rel=1e-5, abs=1e-6), i
        assert math.exp(theta[-1]) == pytest.approx(ref[str(i)]["scale"], rel=1e-5), i
        assert np.sqrt(np.diag(V)) == pytest.approx(ref[str(i)]["se"], rel=1e-4), i


@needs_r
def test_3_a_column_the_earlier_ones_span_is_dropped_as_survreg_drops_it(tmp_path):
    """A clustered chain's design holds each complete column's unit mean beside it; where the
    column is one value per unit the two are one to rounding (the verifier's FWMI-R1-1). survreg
    reports the later column's coefficient NA and its variance 0, and fits the rest; so does the
    engine (0, a zero row of the inverse information), and its fit is the fit without the column.
    On the base the information had no inverse and the draw's Cholesky factor failed."""
    from turbotab.core.methods.imputation import _standardize, tobit_fit

    rng = np.random.default_rng(3)
    codes = np.repeat(np.arange(30), 4)
    event = np.repeat((rng.random(30) < 0.5).astype(float), 4)
    unit_mean = (np.bincount(codes, weights=event) / np.bincount(codes))[codes]
    z0 = rng.standard_normal(len(codes))
    x = 8 + 0.5 * event + 0.3 * z0 + np.repeat(rng.standard_normal(30) * 0.7, 4) + \
        rng.standard_normal(len(codes))
    limit = float(np.quantile(x, 0.3))
    below = x < limit
    D = np.column_stack([np.ones(len(codes)), _standardize(np.column_stack([event, z0])),
                         _standardize(unit_mean.reshape(-1, 1))])
    assert np.max(np.abs(D[:, 1] - D[:, 3])) < 1e-12  # one column twice, to rounding
    y = np.where(below, limit, x)
    ref = run_r(SURVREG_EACH, {"designs": _frame([(0, D, y, below, limit, 0, None)])}, tmp_path)
    theta, V = tobit_fit(D, y, below, limit)
    assert theta[:-1] == pytest.approx(ref["0"]["coef"], rel=1e-5, abs=1e-6)
    assert theta[3] == 0 and not V[3].any() and not V[:, 3].any()
    assert math.exp(theta[-1]) == pytest.approx(ref["0"]["scale"], rel=1e-5)
    assert np.sqrt(np.diag(V)) == pytest.approx(ref["0"]["se"], rel=1e-4)
    kept = [0, 1, 2]
    alone, V_alone = tobit_fit(D[:, kept], y, below, limit)
    np.testing.assert_allclose(theta[[0, 1, 2, 4]], alone, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(V[np.ix_([0, 1, 2, 4], [0, 1, 2, 4])], V_alone, rtol=1e-12,
                               atol=1e-15)


@pytest.mark.parametrize("scale,offset", [(1e-6, 0.0), (1e8, 0.0), (1e8, 1e11), (1e3, -5e5)])
def test_4_the_fit_moves_with_the_columns_units_and_offset(scale, offset):
    """The censored normal is location–scale equivariant (a derivation): z' = a z + b below
    a L + b is fit by β' = a β + b e₀ (the intercept) and σ' = a σ, the inverse information of
    (β', log σ') that of (β, log σ) with the β rows times a. So the fit of a column in other units,
    or offset by a constant, is the moved fit, to 1e−8. At a = 1e8, b = 1e11 the base solver's σ was
    0.7 off (it stopped where BFGS's tolerance read the gradient as small)."""
    from turbotab.core.methods.imputation import tobit_fit

    rng = np.random.default_rng(0)
    n = 200
    x1 = rng.standard_normal(n)
    z = 1 + 0.5 * x1 + rng.standard_normal(n)
    limit = float(np.quantile(z, 0.4))
    below = z < limit
    X = np.column_stack([np.ones(n), x1])
    theta, V = tobit_fit(X, np.where(below, limit, z), below, limit)
    moved, V_moved = tobit_fit(X, np.where(below, scale * limit + offset, scale * z + offset),
                               below, scale * limit + offset)
    assert moved[0] == pytest.approx(scale * theta[0] + offset, rel=1e-8)
    assert moved[1] == pytest.approx(scale * theta[1], rel=1e-8)
    assert moved[2] == pytest.approx(theta[2] + math.log(scale), rel=1e-8, abs=1e-8)
    units = np.r_[scale, scale, 1.0]
    np.testing.assert_allclose(V_moved, V * np.outer(units, units), rtol=1e-6)


def test_5_any_arithmetic_failure_in_the_solver_is_the_one_typed_failure(monkeypatch):
    """FWMI-R2-2: an OverflowError (the base's ``math.exp(tau)``), a FloatingPointError or a
    LinAlgError inside the solver comes out as ``CensoredFitRefused``, whose words say no maximum
    could be confirmed, never as the arithmetic's own message ("math range error"); and the same
    design fits where nothing fails."""
    from turbotab.core.methods import imputation as imp

    from turbotab.core.methods.featurewise_mi import _design

    x, event, Z, _good = overflow_cases()[17]
    below = ~np.isfinite(x)
    limit = float(np.nanmin(x))
    D = _design(event, Z)
    theta, _ = imp.tobit_fit(D, np.where(below, limit, x), below, limit)
    assert np.all(np.isfinite(theta))
    for error in (OverflowError("math range error"), FloatingPointError("overflow"),
                  np.linalg.LinAlgError("Matrix is not positive definite")):
        def failing(*args, _error=error, **kwargs):
            raise _error

        monkeypatch.setattr(imp, "_olsen", failing)
        with pytest.raises(imp.CensoredFitRefused) as refused:
            imp.tobit_fit(D, np.where(below, limit, x), below, limit)
        assert str(refused.value) == imp.CENSORED_FIT_WORDS
        assert "math range" not in str(refused.value)
        monkeypatch.undo()


def test_6_the_single_fill_refuses_equal_detected_values_in_words_and_fills_where_they_differ():
    """The censoring-aware single fill (``missing.BelowDetectionFill``, intercept-only): five
    detected values all equal to 3 and twenty blanks have no finite maximum (σ → 0), and the base
    filled them with the limit itself (σ 2.7e−16). Now the step refuses in words that say why, with
    the other fill and leaving the column out as its ways forward (a ``graph.StageRefusal``). The
    same column with detected values that differ is filled, by E[X | X < L] at its fit (the guard
    silent), as the log-scale fill's test against survreg pins (``test_ms7_omics`` test 4e)."""
    from turbotab.core.graph import StageRefusal
    from turbotab.core.methods.missing import CENSORED_FILL_REFUSED, BelowDetectionFill

    frame = pd.DataFrame({"m": np.r_[np.full(5, 3.0), np.full(20, np.nan)]})
    with pytest.raises(StageRefusal) as refused:
        BelowDetectionFill(["m"], "censoring_aware").fit(frame)
    assert refused.value.code == CENSORED_FILL_REFUSED
    assert refused.value.message == (
        "`m`'s values below detection cannot be filled censoring-aware: its censored-normal model "
        "could not be fit on these rows (its 5 detected values are all equal, so their spread "
        "cannot be estimated).")
    assert [e["label"] for e in refused.value.exits] == [
        "In the missing-values answer, fill values below detection with half the smallest "
        "detected value", "In the missing-values answer, leave `m` out"]
    spread = pd.DataFrame({"m": np.r_[3.0, 3.5, 4.0, 4.5, 5.0, np.full(20, np.nan)]})
    fill = BelowDetectionFill(["m"], "censoring_aware").fit(spread)
    assert 0 < fill.fill_["m"] < 3.0


def test_7_the_log_scale_fill_refuses_a_failed_fit_in_words(monkeypatch):
    """The fill before a log (``omics.LogScaleCensoredFill``): where its fit is refused
    (``CensoredFitRefused``, here made to fail) the step refuses in words with the same ways
    forward, never the solver's error; where nothing fails it fills."""
    from turbotab.core.graph import StageRefusal
    from turbotab.core.methods import imputation as imp
    from turbotab.core.methods.omics import LogScaleCensoredFill

    rng = np.random.default_rng(8)
    x = np.exp(rng.normal(2.0, 0.8, 300))
    x[x < np.quantile(x, 0.3)] = np.nan
    frame = pd.DataFrame({"m": x})
    assert 0 < LogScaleCensoredFill(["m"]).fit(frame).fill_["m"] < np.nanmin(x)

    def failing(*args, **kwargs):
        raise OverflowError("math range error")

    monkeypatch.setattr(imp, "_olsen", failing)
    with pytest.raises(StageRefusal) as refused:
        LogScaleCensoredFill(["m"]).fit(frame)
    assert refused.value.message == (
        "`m`'s values below detection cannot be filled censoring-aware: its censored-normal model "
        "could not be fit on these rows (no maximum of its likelihood could be confirmed).")
    assert len(refused.value.exits) == 2


def test_8_the_chained_equations_refuse_a_censored_model_with_no_maximum_in_words():
    """The joint route (``missing.impute_for_inference``, chained equations with the outcome): a
    censored covariate whose 60 detected values are all equal has no finite maximum. The base
    drew from wherever its solver stopped, or raised; now the imputation is refused in words naming
    the column (an ``ImputationRefused``, which the stages turn into a refusal with its ways
    forward). With detected values that differ it draws m copies (the guard silent)."""
    from turbotab.core.decisions import MissingSpec, ProjectState
    from turbotab.core.methods.imputation import CensoredFitRefused, ImputationRefused
    from turbotab.core.methods.missing import impute_for_inference
    from turbotab.core.models.pipeline import design_spec

    rng = np.random.default_rng(4)
    n = 200
    z = rng.standard_normal(n)
    m = 10 + 0.5 * z + rng.standard_normal(n)
    y = 0.5 * m + 0.3 * z + rng.standard_normal(n)
    lod = float(np.quantile(m, 0.7))
    st = ProjectState(target="y", task="regression", purpose="inference",
                      roles={"m": "exposure", "z": "covariate"},
                      missing=MissingSpec(strategy="multiple_imputation",
                                          below_detection="censoring_aware",
                                          censored_columns=["m"]))
    for spread in (False, True):
        frame = pd.DataFrame({"m": np.where(m >= lod, m if spread else round(lod, 1), np.nan),
                              "z": z})
        spec = design_spec(st, frame, ["m", "z"])
        if spread:
            drawn = impute_for_inference(spec, frame, y, "regression", seed=1)
            assert drawn.m >= 20 and drawn.imputed["m"] == int(np.isnan(frame["m"]).sum())
            continue
        with pytest.raises(ImputationRefused) as refused:
            impute_for_inference(spec, frame, y, "regression", seed=1)
        assert isinstance(refused.value, CensoredFitRefused)
        assert str(refused.value) == (
            "`m`'s censored-normal model could not be fit on these rows: no maximum of its "
            "likelihood could be confirmed, so its values below detection cannot be drawn.")
        assert refused.value.columns == ["m"]


SURVREG_INDICATOR = """
suppressPackageStartupMessages(library(survival))
d <- read.csv(designs_csv)
res <- list()
for (i in unique(d$design)) {
  s <- d[d$design == i, ]
  X <- as.matrix(s[, grep("^D", names(s))])
  f <- survreg(Surv(s$y, s$det == 1, type = "left") ~ X, dist = "gaussian",
               control = survreg.control(maxiter = 500))
  v <- vcov(f)
  res[[i]] <- list(coef = unname(coef(f)), scale = f$scale, iter = f$iter[1],
                   se = sqrt(pmax(diag(v), 0)))
}
out(res)
"""


def indicator_design(group: str, coding: str, detected_one: bool):
    """200 rows, x = 5 + 0.5 z + 1.5 b + N(0, 1), b a 0/1 column with 12 ones, below detection
    under the 40th percentile, and every row of ``group`` (b = 1, "ones", or b = 0, "zeros") below
    detection, but for its largest value where ``detected_one``. b enters the design ``coding``
    "raw" (0/1, as a chain's category indicator enters) or "standardized" (as the feature-wise
    design enters a column). Returns (X, y with the rows below detection at the limit, which rows
    are below, the limit, the group's rows)."""
    rng = np.random.default_rng(12)
    n = 200
    b = np.zeros(n)
    b[rng.choice(n, size=12, replace=False)] = 1
    z = rng.standard_normal(n)
    x = 5 + 0.5 * z + 1.5 * b + rng.standard_normal(n)
    limit = float(np.quantile(x, 0.4))
    rows = b == (1 if group == "ones" else 0)
    below = (x < limit) | rows
    if detected_one:
        pick = np.flatnonzero(rows)[np.argmax(x[rows])]
        below[pick] = False
        x[pick] = max(x[pick], limit + 0.25)
    column = (b - b.mean()) / b.std() if coding == "standardized" else b
    X = np.column_stack([np.ones(n), z, column])
    return X, np.where(below, limit, x), below, limit, rows


@needs_r
def test_9_a_0_1_column_whose_ones_or_zeros_are_all_below_detection_is_refused(tmp_path):
    """FWMI-R3-1. Where every row of one level of a 0/1 column is below detection, the
    censored-normal likelihood has no finite maximum (a derivation): along the direction v with
    X v = −1 on those rows and 0 elsewhere, no detected row moves and each of those rows'
    Φ((L − xᵀβ)/σ) rises toward 1, so the likelihood rises toward a bound it never reaches
    (checked from survreg's own stopping point with scipy's normal, row by row). survreg stops on
    that ridge, with a standard error on b over 100 times the one it reports once a single row of
    the level is detected. The engine refuses each, raw or standardized, ones or zeros, never
    returning a point; the base returned one where the ones were all below detection in a raw 0/1
    column, as a chain's category indicator enters its design (b's coefficient −10.5, its
    standard error 1.1e9). With one row of the level detected the model has its maximum, and the
    engine's fit is survreg's: each coefficient, the scale and each standard error."""
    from scipy import stats

    from turbotab.core.methods.imputation import CensoredFitRefused, tobit_fit

    cases = {f"{g}-{c}-{'one' if one else 'none'}": indicator_design(g, c, one)
             for g in ("ones", "zeros") for c in ("raw", "standardized") for one in (False, True)}
    blocks = []
    for key, (X, y, below, _limit, _rows) in cases.items():
        block = pd.DataFrame({"D1": X[:, 1], "D2": X[:, 2], "y": y, "det": (~below).astype(int)})
        block["design"] = key
        blocks.append(block)
    ref = run_r(SURVREG_INDICATOR, {"designs": pd.concat(blocks, ignore_index=True)}, tmp_path)
    for key, (X, y, below, limit, rows) in cases.items():
        r = ref[key]
        assert r["iter"] < 500, key
        if key.endswith("-none"):
            assert rows.sum() and below[rows].all(), key  # the premise
            v = np.linalg.lstsq(X, -rows.astype(float), rcond=None)[0]
            np.testing.assert_allclose(X @ v, -rows.astype(float), atol=1e-12)
            # Along v from survreg's stopping point each moved row's log Φ rises (scipy's normal,
            # row by row: their sum's gain is below the log-likelihood's rounding there).
            a = (limit - X[rows] @ np.asarray(r["coef"])) / r["scale"]
            for s in (1.0, 2.0, 4.0):
                gain = stats.norm.logcdf(a + s / r["scale"]) - stats.norm.logcdf(a)
                assert np.all(gain > 0), (key, s)
            identified = ref[key.replace("-none", "-one")]
            assert r["se"][2] > 100 * identified["se"][2], key
            with pytest.raises(CensoredFitRefused):
                tobit_fit(X, y, below, limit)
            continue
        theta, V = tobit_fit(X, y, below, limit)
        assert theta[:-1] == pytest.approx(r["coef"], rel=1e-5, abs=1e-6), key
        assert math.exp(theta[-1]) == pytest.approx(r["scale"], rel=1e-5), key
        assert np.sqrt(np.diag(V)) == pytest.approx(r["se"], rel=1e-4), key


def site_panel(one_detected: bool) -> pd.DataFrame:
    """The verifier's ``joint_cat.py``: 300 rows, m = 10 + 0.5 z + N(0, 1), y = 0.5 m + 0.3 z +
    N(0, 1), `site` A, B or C (10%), m below detection under its 40th percentile and on every
    level-C row, but for level C's largest value where ``one_detected``."""
    rng = np.random.default_rng(4)
    n = 300
    site = rng.choice(["A", "B", "C"], size=n, p=[0.45, 0.45, 0.10])
    z = rng.standard_normal(n)
    m = 10 + 0.5 * z + rng.standard_normal(n)
    y = 0.5 * m + 0.3 * z + rng.standard_normal(n)
    lod = float(np.quantile(m, 0.4))
    detected = (m >= lod) & (site != "C")
    if one_detected:
        rows = np.flatnonzero(site == "C")
        pick = rows[np.argmax(m[rows])]
        m[pick], detected[pick] = max(m[pick], lod + 0.5), True
    return pd.DataFrame({"pid": np.arange(n), "m": np.where(detected, m, np.nan), "z": z,
                         "site": site, "y": y, "m_full": m})


def test_10_the_joint_route_refuses_a_category_level_wholly_below_detection_in_words(tmp_path):
    """FWMI-R3-1 through the joint route (chained equations with the outcome, the linear family,
    `m` the exposure, `z` and `site` confounders): every level-C row of `m` is below detection,
    and a chain enters a category as raw 0/1 indicators, so `m`'s censored-normal model has no
    finite maximum (test 9's derivation, on level C's indicator). The base drew level C's `m` at
    the limit or near −4e9 and served an unadjusted estimate of 0.349 (95% CI −0.64 to 1.34); the
    linear fit refused with a false reason, that `site_C` and `m` are fixed combinations of the
    other columns, which the draws created. Now the fit and the declared models refuse in words
    naming `m`, with a way forward, and report no estimate. With one level-C value detected the
    model has its maximum, and both pool `m` (the guard silent), Model 2's estimate within three of
    its standard errors of statsmodels' least squares on the values before censoring."""
    import statsmodels.formula.api as smf

    from turbotab.core.tests.acceptance import estimand_fixtures as ef

    words = ("Multiple imputation cannot run on these data: `m`'s censored-normal model could not "
             "be fit on these rows: no maximum of its likelihood could be confirmed, so its values "
             "below detection cannot be drawn.")
    roles = {"pid": "identifier", "m": "exposure", "z": "covariate", "site": "covariate"}
    mi = {"strategy": "multiple_imputation", "below_detection": "censoring_aware",
          "censored_columns": ["m"], "m": 20}
    st = ef.state(target="y", task="regression", exposure="m", measure="mean_difference",
                  roles=roles, answers={"z": ef.CONFOUNDER, "site": ef.CONFOUNDER},
                  models=["linear"], missing=mi, amounts={"code_or_count:m": "amount"})
    for one_detected in (False, True):
        panel = site_panel(one_detected)
        assert panel.loc[panel["site"] == "C", "m"].notna().sum() == int(one_detected)
        out = ef.run(panel.drop(columns="m_full"), tmp_path / str(one_detected), st)
        entry = next(e for e in out["fit"]["models"] if e["family"] == "linear")
        seq = ef.sequence(out["effects"])
        if not one_detected:
            assert entry["inference"]["refused"] == words and not entry["coefficients"]
            assert [e["label"] for e in entry["inference"]["exits"]] == [
                "Complete cases, with their assumption stated"]
            assert seq["model_2"]["inference"]["refused"] == words
            assert seq["model_2"]["inference"]["exits"] == entry["inference"]["exits"]
            assert not any(s.get("effects") for s in seq.values())
            assert out["effects"]["methods"].startswith(f"No estimate of `m` is reported: {words}")
            continue
        assert entry["inference"]["refused"] is None
        primary = seq["model_2"]
        assert primary["inference"]["refused"] is None
        assert primary["inference"]["caption"].startswith("Multiple imputation, m = ")
        (effect,) = primary["effects"]
        full = smf.ols("y ~ m_full + z + C(site)", data=panel).fit()
        assert abs(effect["estimate"] - full.params["m_full"]) < 3 * effect["se"]
