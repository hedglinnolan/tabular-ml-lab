"""Two of the chained equations' own draws, ``imputation._binary`` and ``imputation._norm_draw``
(INBOX: the PP2 scope's 11.1005 and INBOX:1006), each against truth from construction, numpy,
scipy or a derivation from R mice's code (mice 3.19.0's ``.norm.draw``, ``sym`` and
``remove.lindep``).

**A 0/1 column whose recorded values hold one level** is drawn as that level, as a category of one
level is (``_categorical``). Its logistic model has no finite maximum: on 190 recorded zeros the
base's fit gave its intercept −31.3 with a variance of 2.1e11, and the drawn intercept had either
sign, so all ten blanks were drawn as 1, a level never recorded, in 10 of 20 copies. mice does not
stabilize this: it leaves such a column unimputed by default (``remove.constant``), and made to
draw it (``remove.constant = FALSE``), its ``logreg`` drew all ten as 1 in 9 of 20 copies, since
its augmentation (White, Daniel & Royston 2010) adds rows only at the levels recorded. The truth is
construction: every recorded value is the one level, so every draw is. The model's random numbers
are still consumed, so every later draw in the copy is the base's.

**A number's draw** (mice's ``.norm.draw``: β* = β̂ + σ* R z with R Rᵀ = V, V = (XᵀX + ridge)⁺)
raised numpy's bare LinAlgError where V has no Cholesky factor in float64 (INBOX:1006: `x`
recorded on 48 of 70 rows beside a 30-level category, 49 numbers and the outcome's term). A
predictor exactly zero on the recorded rows (the indicator of a level none of them hold) makes V
zero along it, and V's other directions are large (more columns than rows), so its rounding is
negative beyond the 10⁻¹² the factor adds. Where the factor fails, R is now V's root by its
eigendecomposition, V symmetrized first as mice's ``sym`` does, and the draw takes the same z.
The tests recover R Rᵀ from the draws and their normal deviates and check it against scipy's
``pinvh`` of mice's ridged cross-product (``estimice``: ``xtx + diag(ridge * diag(xtx))``, ridge
10⁻⁵), whose pseudo-inverse gives the zero column's coefficient no variance, as ``remove.lindep``
drops a predictor whose variance on the recorded rows is at most 10⁻⁴. Where the factor exists,
every draw is the base's, bit for bit. A cross-product numpy cannot factor is refused in plain
words, naming no library, and as a LinAlgError too, so the feature-wise copies still refuse their
study factor in the words that name it and offer to leave it out (the verifier's PP2D-V1); the
chained equations refuse it in words naming the column whose model it is, which the fit's table
offers to leave out where it is a covariate (PP2D-V2-2).
"""
from __future__ import annotations

import math
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.linalg import pinvh

from turbotab.core.methods.imputation import (ImputationRefused, _binary, _expit, _logistic,
                                              _norm_draw)
from turbotab.core.methods.smcfcs import Variable, impute
from turbotab.core.tests.test_imputation_width import _one_level

MICE_RIDGE = 1e-5  # mice's estimice(ridge = 1e-05): the penalty is ridge * diag(xtx)


# ── a 0/1 column recorded at one level ───────────────────────────────────────


def _recorded_at(level: float, rows: int = 200, blanks: int = 10, seed: int = 0
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The scope's case: a 0/1 column recorded at ``level`` on 190 of 200 rows and blank on its
    first 10, its model an intercept and one covariate."""
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(rows), rng.normal(size=rows)])
    miss = np.arange(rows) < blanks
    y = np.where(miss, np.nan, level)
    return X, y, miss


@pytest.mark.parametrize("level", [0.0, 1.0], ids=["zeros", "ones"])
def test_a_zero_one_column_recorded_at_one_level_draws_every_blank_as_that_level(level):
    """190 recorded zeros (or ones) and 10 blanks, seeds 0 to 19: the base drew all ten blanks as
    the other level in half the copies (zeros: 10 of 20). Every recorded value is ``level``, so
    every draw is (construction)."""
    X, y, miss = _recorded_at(level)
    drawn = [_binary(X, y, miss, np.random.default_rng(seed)).tolist() for seed in range(20)]
    assert drawn == [[level] * 10] * 20


def test_the_one_level_draw_consumes_the_logistic_model_s_random_numbers():
    """The base's draw takes one standard normal per coefficient (β*), then one uniform per blank
    (its Bernoulli draws); the one-level draw takes the same, so the generator is left where the
    base left it and every later draw in the copy is the base's."""
    X, y, miss = _recorded_at(0.0)
    rng = np.random.default_rng(7)
    _binary(X, y, miss, rng)
    reference = np.random.default_rng(7)
    reference.standard_normal(X.shape[1])
    reference.random(int(miss.sum()))
    assert rng.bit_generator.state == reference.bit_generator.state


def test_a_zero_one_column_of_two_recorded_levels_is_drawn_by_its_logistic_model_as_before():
    """The guard stays silent where the recorded values hold both levels: the draw is the base's
    logistic draw, bit for bit (β* = β̂ + L z, L the Cholesky factor of the inverse information
    with 10⁻¹² on its diagonal, then a uniform below each blank's probability)."""
    X, y, miss = _recorded_at(0.0)
    y[miss.size - 3:] = 1.0  # three ones among the recorded values
    for seed in range(5):
        beta, V = _logistic(X[~miss], y[~miss])
        rng = np.random.default_rng(seed)
        star = beta + np.linalg.cholesky(V + np.eye(len(V)) * 1e-12) @ rng.standard_normal(2)
        base = rng.random(10) < _expit(X[miss] @ star)
        assert np.array_equal(_binary(X, y, miss, np.random.default_rng(seed)),
                              base.astype(float))


def test_the_chain_draws_a_column_of_thirty_recorded_ones_as_one_in_every_copy():
    """The INBOX's second case: `b` recorded as 1 on 30 of 40 rows beside two complete numbers;
    over 20 copies (seed 1) the base drew all ten blanks as 0 in 11. Every recorded value is 1, so
    every copy's blanks are 1."""
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"b": np.r_[np.ones(30), np.full(10, np.nan)],
                          "z1": rng.normal(size=40), "z2": rng.normal(size=40)})
    drawn = impute(frame, [Variable("b", "binary")], m=20, seed=1)
    assert drawn.imputed == {"b": 10}
    assert all(f["b"].iloc[30:].tolist() == [1.0] * 10 for f in drawn.frames)


def test_a_wide_model_s_zero_one_column_recorded_at_one_level_is_drawn_as_that_level():
    """`b` recorded as 1 on 100 of 120 rows beside 80 numbers and the outcome's term (82 columns,
    ``test_imputation_width``'s fixture, not held to the rows rule): every blank of every copy is
    1."""
    frame, variable = _one_level(text=False, second=False)
    drawn = impute(frame, [variable], m=5, iterations=2)
    assert all(f["b"].iloc[100:].tolist() == [1.0] * 20 for f in drawn.frames)


# ── a number's draw where V has no Cholesky factor ───────────────────────────


def _inbox_1006(seed: int, rows: int = 70, recorded: int = 48) -> pd.DataFrame:
    """INBOX:1006's model: `x` recorded on its first 48 of 70 rows beside 49 complete numbers, a
    complete category of 30 levels and the outcome's term, 80 columns with the category as its 29
    indicators."""
    rng = np.random.default_rng(seed)
    g = np.array([f"L{k:02d}" for k in range(30)] * 3)[:rows]
    rng.shuffle(g)
    frame = pd.DataFrame({"x": rng.normal(size=rows),
                          **{f"c{j:02d}": rng.normal(size=rows) for j in range(1, 50)},
                          "g": g, "__outcome": rng.normal(size=rows)})
    frame.loc[recorded:, "x"] = np.nan
    return frame


@pytest.mark.parametrize("seed", [1, 5])
def test_inbox_1006_s_model_is_drawn_in_every_copy(seed):
    """On these seeds the base raised numpy's LinAlgError ("Matrix is not positive definite") from
    `x`'s draw. Each copy now holds a finite value in every blank of `x`, its recorded values
    untouched (whichever factor the platform's float64 gives, the claim is the same: drawn)."""
    frame = _inbox_1006(seed)
    drawn = impute(frame, [Variable("x", "numeric")], m=2, iterations=2, seed=seed)
    assert drawn.imputed == {"x": 22}
    for copy in drawn.frames:
        assert np.isfinite(copy["x"].to_numpy(dtype=float)).all()
        assert copy["x"].iloc[:48].equals(frame["x"].iloc[:48])


def _wide_design(seed: int = 1, rows: int = 70, recorded: int = 48
                 ) -> tuple[np.ndarray, np.ndarray]:
    """A number's design as the chain builds it for INBOX:1006's `x`, on its 48 recorded rows: an
    intercept, the 29 raw 0/1 indicators of a 30-level category (on seed 1, three levels none of
    the recorded rows hold, so three zero columns) and 50 standardized numbers."""
    rng = np.random.default_rng(seed)
    g = np.array(list(range(30)) * 3)[:rows]
    rng.shuffle(g)
    numbers = rng.normal(size=(rows, 50))
    numbers = (numbers - numbers.mean(axis=0)) / numbers.std(axis=0)
    X = np.column_stack([np.ones(rows), *[(g == k).astype(float) for k in range(1, 30)],
                         numbers])[:recorded]
    return X, rng.normal(size=recorded)


def _full_rank_design(rows: int = 200) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(3)
    X = np.column_stack([np.ones(rows), rng.normal(size=(rows, 6))])
    return X, X @ rng.normal(size=7) + rng.normal(size=rows)


def _deviates(seed: int, df: int, p: int) -> np.ndarray:
    """The z a draw on ``seed`` takes: after σ*'s χ²(df) (``.norm.draw``'s order), p standard
    normals."""
    rng = np.random.default_rng(seed)
    rng.chisquare(df)
    return rng.standard_normal(p)


def _no_cholesky(*_: object, **__: object) -> np.ndarray:
    raise np.linalg.LinAlgError("Matrix is not positive definite")


@pytest.mark.parametrize("design, factor", [
    ("wide", "as numpy gives it"), ("wide", "no Cholesky factor"),
    ("full rank", "no Cholesky factor")])
def test_the_draw_s_covariance_is_mice_s_ridged_pseudo_inverse(design, factor, monkeypatch):
    """β* − β̂ = σ* R z for the z each draw takes, so p draws on seeds 0 to p − 1 give R = D Z⁻¹
    (D the p draws' (β* − β̂)/σ*, Z their z) and R Rᵀ, the covariance the draws are taken from.
    It is scipy's ``pinvh`` of mice's ridged cross-product XᵀX + 10⁻⁵ diag(XᵀX), to 10⁻⁸ of its
    largest entry, on the wide design (where the base's Cholesky factor failed on this machine)
    and wherever the factor is refused (forced here, as a platform's float64 may refuse it), and
    each zero column's coefficient is drawn with no variance and centered at 0, to 10⁻⁸ of the
    largest coefficient (the same float64 bound as the covariance; a 10⁻¹⁰ bound tipped on Linux).
    The base raised LinAlgError on each."""
    X, y = _wide_design() if design == "wide" else _full_rank_design()
    if factor == "no Cholesky factor":
        monkeypatch.setattr(np.linalg, "cholesky", _no_cholesky)
    n, p = X.shape
    df = max(n - p, 1)
    D, Z = np.empty((p, p)), np.empty((p, p))
    for seed in range(p):
        beta, star, sigma = _norm_draw(X, y, np.random.default_rng(seed))
        D[:, seed], Z[:, seed] = (star - beta) / sigma, _deviates(seed, df, p)
    R = D @ np.linalg.inv(Z)
    XtX = X.T @ X
    V = pinvh(XtX + np.diag(MICE_RIDGE * np.diag(XtX)))
    assert np.abs(R @ R.T - V).max() <= 1e-8 * np.abs(V).max()
    zero = np.flatnonzero(~X.any(axis=0))
    assert len(zero) == (3 if design == "wide" else 0)
    assert np.abs(beta[zero]).max(initial=0.0) <= 1e-8 * np.abs(beta).max()


def test_where_the_cholesky_factor_exists_the_draw_is_the_base_s_bit_for_bit():
    """The fallback stays silent where V has a Cholesky factor: β̂, β* and σ* are the base's
    (``pinv`` of XᵀX with 10⁻⁵ of its diagonal, floored at 10⁻¹², added; σ*² = RSS/χ²(df); β* =
    β̂ + σ* L z, L the Cholesky factor of V + 10⁻¹² I), bit for bit."""
    X, y = _full_rank_design()
    XtX = X.T @ X
    V = np.linalg.pinv(XtX + np.diag(1e-5 * np.maximum(np.diag(XtX), 1e-12)))
    beta = V @ (X.T @ y)
    for seed in range(5):
        rng = np.random.default_rng(seed)
        sigma = math.sqrt(float((y - X @ beta) @ (y - X @ beta)) / rng.chisquare(len(y) - 7))
        star = beta + sigma * (np.linalg.cholesky(V + np.eye(7) * 1e-12) @ rng.standard_normal(7))
        got = _norm_draw(X, y, np.random.default_rng(seed))
        assert np.array_equal(got[0], beta) and np.array_equal(got[1], star) and got[2] == sigma


@pytest.mark.parametrize("fails", ["pinv", "eigh"])
def test_a_cross_product_numpy_cannot_factor_is_refused_in_words(fails, monkeypatch):
    """Where numpy's SVD (``pinv``) or, with no Cholesky factor, its eigendecomposition does not
    converge (LAPACK allows either), the draw is refused in plain words, never numpy's bare
    LinAlgError: an ImputationRefused, whose words name no library and carry none of LAPACK's text
    (that is kept, unshown, as ``why``), and a LinAlgError too, so a caller that refuses a chained
    model's linear-algebra failure in its own words still does (the next test). The tests above,
    where each converges, are drawn."""
    message = "SVD did not converge" if fails == "pinv" else "Eigenvalues did not converge"

    def failing(*_: object, **__: object) -> None:
        raise np.linalg.LinAlgError(message)

    monkeypatch.setattr(np.linalg, fails, failing)
    if fails == "eigh":
        monkeypatch.setattr(np.linalg, "cholesky", _no_cholesky)
    X, y = _full_rank_design()
    with pytest.raises(ImputationRefused) as refused:
        _norm_draw(X, y, np.random.default_rng(0))
    assert isinstance(refused.value, np.linalg.LinAlgError)
    assert str(refused.value) == ("The model that imputes a number's blanks could not be computed "
                                  "on these rows.")
    assert refused.value.why == f"LinAlgError: {message}"


def test_the_calibration_bootstrap_counts_a_replicate_whose_imputation_is_refused_as_not_imputed(
        monkeypatch):
    """PP2D-V2-1: the calibration's bootstrap tallies each replicate it leaves out under its
    reason, and the researcher reads that reason's words (``stages.calibration.FAILURE_WORDS``).
    A replicate whose number's draw cannot be computed (``imputation.NormDrawRefused``) or whose
    censored-normal fit certifies no maximum (``imputation.CensoredFitRefused``) could not be
    imputed, and is counted so, never under its class's name; on 652cb0bc they were tallied as
    `NormDrawRefused` and `CensoredFitRefused`. The tally the other way: numpy's bare LinAlgError
    (a singular fit) and a calibration refused keep their own words."""
    from turbotab.core.methods.calibration import CalibrationRefused, whole_chain
    from turbotab.core.methods.imputation import CensoredFitRefused
    from turbotab.core.stages.calibration import FAILURE_WORDS

    X, y = _full_rank_design()

    def estimate(_: object, b: int) -> np.ndarray:
        if b in (1, 2):
            _norm_draw(X, y, np.random.default_rng(b))  # its pseudo-inverse fails (below)
        if b == 3:
            raise CensoredFitRefused(why="the solver's own account")
        if b == 4:
            raise np.linalg.LinAlgError("Singular matrix")
        if b == 5:
            raise CalibrationRefused("no positive true-intake covariance")
        return np.array([1.0])

    class Rows:
        def draw(self, rng: np.random.Generator) -> None:
            return None

    _pinv_fails_in_the_draw(monkeypatch)
    boot = whole_chain(estimate, Rows(), 8, 0)
    assert boot.failures == {"ImputationRefused": 3, "LinAlgError": 1, "CalibrationRefused": 1}
    assert {FAILURE_WORDS.get(k, k) for k in boot.failures} == {
        "could not be imputed", "had a singular fit", "had no positive true-intake covariance"}


def _pinv_fails_in_the_draw(monkeypatch: pytest.MonkeyPatch) -> None:
    """numpy's ``pinv`` made to fail as LAPACK's SVD may (``models/ridge.py`` records gesdd not
    converging on Accelerate), only where ``_norm_draw`` calls it."""
    real = np.linalg.pinv

    def pinv(*args: object, **kwargs: object) -> np.ndarray:
        if sys._getframe(1).f_code.co_name == "_norm_draw":
            raise np.linalg.LinAlgError("SVD did not converge")
        return real(*args, **kwargs)

    monkeypatch.setattr(np.linalg, "pinv", pinv)


def _factor_beside_a_blank_covariate(rows: int = 120, seed: int = 1
                                     ) -> tuple[pd.DataFrame, np.ndarray]:
    """A study factor `f1` = 1 + 0.5 `a1` + 0.5 × event + N(0, 1), blank on a fifth of its rows,
    beside an adjustment column `a1` blank on a fifth of its own, so the factor's copies are drawn
    by the chained equations (``featurewise_mi.draw_copies``), `f1`'s by a number's model."""
    rng = np.random.default_rng(seed)
    a1 = rng.normal(size=rows)
    event = (rng.random(rows) < 0.5).astype(float)
    f1 = 1 + 0.5 * a1 + 0.5 * event + rng.normal(size=rows)
    f1[rng.random(rows) < 0.2] = np.nan
    a1[rng.random(rows) < 0.2] = np.nan
    return pd.DataFrame({"f1": f1, "a1": a1}), event


def test_a_study_factor_whose_number_s_model_cannot_be_computed_is_refused_naming_it(monkeypatch):
    """PP2D-V1: where `f1`'s model cannot be computed in its chain, the feature-wise copies refuse
    `f1` in the words that name it, and offer to leave it out (``columns``), as the base did for
    numpy's LinAlgError; on fe4ff420 (round 1's head) the refusal escaped that handler, its
    words numpy's and LAPACK's, naming no column. Where the model is computed, `f1` is drawn:
    every blank of every copy finite, its recorded values untouched."""
    from turbotab.core.methods.featurewise_mi import draw_copies

    frame, event = _factor_beside_a_blank_covariate()
    blank = frame["f1"].isna().to_numpy()
    copies = draw_copies(frame, event, ["f1"], ["a1"], {}, 5, seed=1)
    assert copies.imputed == {"f1": int(blank.sum())} and copies.values["f1"].shape == (120, 5)
    assert np.isfinite(copies.values["f1"]).all()
    assert (copies.values["f1"][~blank] == frame["f1"].to_numpy()[~blank, None]).all()
    _pinv_fails_in_the_draw(monkeypatch)
    with pytest.raises(ImputationRefused) as refused:
        draw_copies(frame, event, ["f1"], ["a1"], {}, 5, seed=1)
    assert str(refused.value) == ("The chained equations that draw `f1`'s copies could not fit "
                                  "their models on these rows, so its copies cannot be drawn.")
    assert refused.value.columns == ["f1"]


def _number_beside_others(path: str, rows: int = 120, seed: int = 1
                          ) -> tuple[pd.DataFrame, list[Variable], dict[str, object]]:
    """A number `f1` = 1 + 0.5 `a1` + N(0, 1), blank on a fifth of its rows, beside a complete
    number `a1`, drawn by each of the chained equations' three number's models, so that in each
    `f1`'s is the only number's model the chain fits: ``fcs``, beside `b1`, 0/1 and blank on a
    fifth of its own (a logistic model); ``smcfcs``, its covariate model under a linear
    substantive model of an outcome `y` on `f1` and `a1`; ``units``, constant within units of two
    rows and imputed once per unit from a model of the units."""
    rng = np.random.default_rng(seed)
    a1 = rng.normal(size=rows)
    if path == "units":
        a1 = np.repeat(a1[: rows // 2], 2)
    f1 = 1 + 0.5 * a1 + rng.normal(size=rows)
    blank = rng.random(rows) < 0.2
    if path == "units":
        f1, blank = np.repeat(f1[: rows // 2], 2), np.repeat(blank[: rows // 2], 2)
    frame = pd.DataFrame({"f1": np.where(blank, np.nan, f1), "a1": a1})
    if path == "fcs":
        b1 = (rng.random(rows) < 0.4).astype(float)
        frame["b1"] = np.where(rng.random(rows) < 0.2, np.nan, b1)
        return frame, [Variable("f1", "numeric"), Variable("b1", "binary")], {}
    if path == "smcfcs":
        from turbotab.core.methods.smcfcs import Outcome, Substantive

        y = f1 + 0.5 * a1 + rng.normal(size=rows)
        design = lambda f: f[["f1", "a1"]].to_numpy(dtype=float)  # noqa: E731
        return frame, [Variable("f1", "numeric")], {
            "mode": "smcfcs", "substantive": Substantive(Outcome("linear", y=y), design)}
    return frame, [Variable("f1", "numeric", unit_level=True)], {
        "units": np.repeat(np.arange(rows // 2), 2)}


@pytest.mark.parametrize("path", ["fcs", "smcfcs", "units"])
def test_the_imputation_of_the_whole_model_refuses_the_number_whose_model_it_cannot_compute(
        path, monkeypatch):
    """The copies the fit's coefficient table, calibration and scales draw
    (``missing.impute_for_inference``, ``smcfcs.impute``) refuse a number's model numpy cannot
    compute in words naming the column, wherever the chain draws it (each of its three number's
    models), so the researcher reads which model failed and an exit may offer to leave it out
    (``columns``, the verifier's PP2D-V2-2; on 652cb0bc the words named none). Still an
    ImputationRefused and a LinAlgError, as the draw's own refusal is. The truth is construction:
    `f1`'s is the only number's model the chain fits. Where the model is computed, every blank is
    drawn."""
    frame, variables, how = _number_beside_others(path)
    drawn = impute(frame, variables, m=2, seed=1, iterations=2, **how)
    names = [v.name for v in variables]
    assert all(np.isfinite(f[names].to_numpy()).all() for f in drawn.frames)
    _pinv_fails_in_the_draw(monkeypatch)
    with pytest.raises(ImputationRefused) as refused:
        impute(frame, variables, m=2, seed=1, iterations=2, **how)
    assert str(refused.value) == ("The model that imputes `f1`'s blanks could not be computed on "
                                  "these rows.")
    assert refused.value.columns == ["f1"]
    assert isinstance(refused.value, np.linalg.LinAlgError)


def _fit_of_a_blank(folder: object, blank: str) -> dict[str, object]:
    """The linear family's coefficient table, through the stage graph, on 200 rows of `y` = `x` +
    0.5 `a1` + N(0, 1) under multiple imputation, `x` the exposure and `a1` a covariate, with
    ``blank`` blank on a fifth of its rows."""
    from turbotab.core.decisions import MissingSpec, ProjectState
    from turbotab.core.tests.test_imputation_width import FIT, _fit_table

    rng = np.random.default_rng(4)
    x, a1 = rng.normal(size=200), rng.normal(size=200)
    table = pd.DataFrame({"id": np.arange(200), "y": x + 0.5 * a1 + rng.normal(size=200),
                          "x": x, "a1": a1})
    table.loc[rng.random(200) < 0.2, blank] = np.nan
    path = folder / "blank.csv"
    table.to_csv(path, index=False)
    state = ProjectState(target="y", task="regression", purpose="inference",
                         grain={"grain": "one_row_per_unit"},
                         roles={"id": "identifier", "x": "exposure", "a1": "covariate"},
                         exclusions=[], missing=MissingSpec(strategy="multiple_imputation"), **FIT)
    return _fit_table(path, folder / "project", state)


@pytest.mark.parametrize("blank, leave", [("a1", ["Leave `a1` out"]), ("x", [])],
                         ids=["covariate", "exposure"])
def test_the_fit_names_the_number_whose_model_it_cannot_compute_and_offers_to_leave_a_covariate_out(
        tmp_path, monkeypatch, blank, leave):
    """PP2D-V2-2 through the stage graph: where the model that imputes a covariate's blanks cannot
    be computed, the fit's table is refused in the words naming it, and offers complete cases and
    leaving it out; where it is the exposure's, the words name it and offer only complete cases,
    as the table never offers to leave out what the analysis studies (``modeling._leavable``)."""
    from turbotab.core.methods.smcfcs import REFUSED_LEAD

    _pinv_fails_in_the_draw(monkeypatch)
    said = _fit_of_a_blank(tmp_path, blank)
    assert said["caption"] == (f"No coefficients: {REFUSED_LEAD}The model that imputes `{blank}`'s "
                               f"blanks could not be computed on these rows.")
    assert [e["label"] for e in said["exits"]] == [
        "Complete cases, with their assumption stated", *leave]
