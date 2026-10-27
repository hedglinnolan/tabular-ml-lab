"""SMC-FCS's draw of the substantive model on an aliased or ill-conditioned design (the Sullivan
2021 crash sweep, B1).

The sweep met ``LinAlgError: Matrix is not positive definite`` in the linear outcome draw
(``smcfcs._linear_draw``) twice. Under the "these participants" survey answer the model further
adjusted for `race5` beside `RIDRETH3` held both (`race5` a coarser recode of `RIDRETH3`'s levels,
so an exact linear combination of the intercept and `RIDRETH3`'s indicators); under an exclusion
that left one level of `RIAGENDR`, the column was constant on every analyzed row (aliased with the
intercept). Both designs were finite, so no NaN or inf reached the draw: (XᵀX)⁺ of the aliased
design was indefinite to rounding (smallest eigenvalue −1e−7, asymmetric to 3e−5) and the Cholesky
factor of σ*²(XᵀX)⁺ failed. R's ``lm`` leaves such a column out (its coefficient NA; ``qr`` at tol
1e−7, the rule ``models.effects.column_basis`` writes out), and so do ``glm`` and ``coxph``.

**Truth from construction and derivation.** A column built as an exact linear combination of the
intercept and the columns before it adds nothing to the design's column space, so the model with
it is the model without it: the same linear predictor for every row, the candidates' rows
included (each candidate changes only the imputed variable, and the aliased column here is built
from complete columns), so the same likelihood f(y | x, z) and the same posterior draw of it. The
draw on the aliased design must therefore be the draw on the design without the column, with that
coefficient zero: bit for bit, the same random numbers drawn in the same order. A column whose
part outside the others' span is 1e−5 of its length (by construction, above R's 1e−7) is not
aliased and is kept.

The aliased constructions put an energy on a scale of millions (kcal × 1,000) beside a code's
indicators; there the linear draw raised on most seeds at the base commit (numpy's LAPACK on
macOS). Whether float64 tips one seed into the failure differs by platform, so the tests pin the
claim (the draw is the draw without the column), not that a seed raised. The guard's silent side
is drawn at the scale the sweep met (kcal), where the near-aliased design's draw ran at the base
commit too.

**The linear draw on a design no column of which is aliased.** With `race5` left out, the sweep's
further-adjusted design was full rank by R's rule, and its draw still raised: (XᵀX)⁺ at a condition
number of 1.6e13 (kcal beside indicators, and the near-collinear spline terms of a skewed minutes
column) was indefinite by more than the factor's jitter, 1e−12 σ*² on the diagonal. That jitter
was absolute, so where the factor did exist it inflated the drawn variance of a coefficient on a
large scale. The linear draw is now made from the QR decomposition of the design with each column
scaled to unit length, no cross-product matrix inverted, and mapped back: the same normal, and the
same factor of it, in exact arithmetic. Truth by derivation: the posterior of a
coefficient whose column is multiplied by c is the posterior of the original divided by c, so the
draw does not depend on the unit (exactly, with the scaled design the same matrix whatever the
unit, up to rounding); and the draws' variance is E[σ*²](XᵀX)⁻¹, with E[σ*²] = σ̂² df/(df − 2) for
σ*² = σ̂² df/χ²(df), computed here with NumPy's least squares and inverse.
"""
from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

from turbotab.core.methods import smcfcs
from turbotab.core.methods.smcfcs import Outcome, Substantive, Variable, _draw_outcome, impute

RSCRIPT = shutil.which("Rscript")
needs_r = pytest.mark.skipif(RSCRIPT is None, reason="R (Rscript) is not installed: the R reference "
                                                     "is skipped")
LEVELS = 6  # the code's levels; its indicators are the design's first five columns
RECODE = np.array([1.0, 1.0, 2.0, 3.0, 4.0, 5.0])  # a coarser recode of the code's six levels


def _columns(seed: int, n: int, alias: str, unit: float = 1000.0
             ) -> tuple[np.ndarray, int, np.random.Generator]:
    """The design without an intercept: a six-level code's five indicators, an energy (kcal times
    ``unit``), a number, then the aliased column (``alias``): ``"constant"`` 2.0 on every row (one
    level an exclusion left), ``"recode"`` the code's coarser recode, ``"combination"`` 3·number −
    energy/10⁶ + 1, or ``"near"`` the recode plus a part outside the others' span of 1e−5 of its
    length (not aliased). Returns the design, the aliased column's position and the generator."""
    rng = np.random.default_rng(seed)
    code = rng.integers(0, LEVELS, n)
    indicators = np.column_stack([(code == k).astype(float) for k in range(1, LEVELS)])
    energy = rng.normal(2000.0, 900.0, n) * unit
    number = rng.normal(size=n)
    base = np.column_stack([indicators, energy, number])
    if alias == "constant":
        extra = np.full(n, 2.0)
    elif alias == "recode":
        extra = RECODE[code]
    elif alias == "combination":
        extra = 3.0 * number - energy / 1e6 + 1.0
    else:  # "near": the recode, moved off the span by 1e-5 of its length
        extra = RECODE[code]
        span = np.column_stack([np.ones(n), base])
        noise = rng.normal(size=n)
        noise -= span @ np.linalg.lstsq(span, noise, rcond=None)[0]
        extra = extra + noise * (1e-5 * np.linalg.norm(extra) / np.linalg.norm(noise))
    return np.column_stack([base, extra]), base.shape[1], rng


def _outcome(kind: str, M: np.ndarray, rng: np.random.Generator) -> Outcome:
    """An outcome of ``kind`` that depends on the number and the code's indicators."""
    n = len(M)
    eta = 0.6 * M[:, 6] + M[:, :5] @ np.array([0.2, -0.1, 0.3, 0.1, -0.2])
    if kind == "linear":
        return Outcome("linear", y=90.0 + 15.0 * eta + rng.normal(0.0, 15.0, n))
    if kind == "logistic":
        return Outcome("logistic", y=(rng.random(n) < 1.0 / (1.0 + np.exp(-eta))).astype(float))
    time = rng.exponential(1.0 / (0.1 * np.exp(eta)))
    censor = rng.uniform(0.0, 20.0, n)
    return Outcome("cox", time=np.minimum(time, censor), event=(time <= censor).astype(float),
                   entry=np.zeros(n))


@pytest.mark.parametrize("alias", ["constant", "recode", "combination"])
def test_the_aliased_column_is_the_one_r_lm_leaves_out(alias):
    """Exactly the constructed column is left out, from the intercept-first design; every column
    but it is kept, the intercept first."""
    M, at, _ = _columns(0, 400, alias)
    X = np.column_stack([np.ones(len(M)), M])
    assert smcfcs.identified_columns(X) == [j for j in range(X.shape[1]) if j != at + 1]


def test_a_column_off_the_span_by_more_than_r_s_tolerance_is_kept():
    """The recode moved off the others' span by 1e−5 of its length is not aliased (R's tol is
    1e−7): every column is kept, so the draw is made on the whole design."""
    M, _, _ = _columns(0, 400, "near")
    X = np.column_stack([np.ones(len(M)), M])
    assert smcfcs.identified_columns(X) == list(range(X.shape[1]))


@needs_r
@pytest.mark.parametrize("alias", ["constant", "recode", "combination", "near"])
def test_the_columns_kept_are_those_r_lm_gives_a_coefficient(tmp_path, alias):
    """Against R's ``lm``: the columns whose coefficient is not NA are the columns kept."""
    M, _, rng = _columns(0, 400, alias)
    names = [f"c{j}" for j in range(M.shape[1])]
    frame = pd.DataFrame(M, columns=names).assign(y=_outcome("linear", M, rng).y)
    frame.to_csv(tmp_path / "data.csv", index=False)
    (tmp_path / "reference.R").write_text(
        'd <- read.csv("data.csv")\n'
        f'fit <- lm(y ~ {" + ".join(names)}, data = d)\n'
        'writeLines(as.character(which(!is.na(coef(fit))) - 1), "kept.txt")\n')
    done = subprocess.run([RSCRIPT, "--no-save", "--no-restore", "reference.R"], cwd=tmp_path,
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    kept = [int(v) for v in (tmp_path / "kept.txt").read_text().split()]
    assert smcfcs.identified_columns(np.column_stack([np.ones(len(M)), M])) == kept


@pytest.mark.parametrize("kind", ["linear", "logistic", "cox"])
@pytest.mark.parametrize("alias", ["constant", "recode", "combination"])
@pytest.mark.parametrize("seed", range(4))
def test_the_draw_on_an_aliased_design_is_the_draw_without_the_column(kind, alias, seed):
    """The outcome model's draw on the design with the aliased column equals, bit for bit, the
    draw on the design without it from the same generator state, with the aliased column's
    coefficient zero: the linear model's σ*² and the Cox model's baseline hazard at each row too.
    At the base commit the linear draw raised (for most of these seeds), the logistic draw put a
    coefficient on the aliased column (its ridge-penalized information's flat direction), and the
    Cox draw overflowed its baseline hazard on some."""
    M, at, rng = _columns(seed, 300, alias)
    out = _outcome(kind, M, rng)
    with_it = _draw_outcome(M, out, np.random.default_rng(100 + seed))
    without = _draw_outcome(np.delete(M, at, axis=1), out, np.random.default_rng(100 + seed))
    position = at if kind == "cox" else at + 1  # the linear and logistic draws lead with β₀
    assert with_it.beta[position] == 0.0
    np.testing.assert_array_equal(np.delete(with_it.beta, position), without.beta)
    if kind == "linear":
        assert with_it.sigma2 == without.sigma2
    if kind == "cox":
        np.testing.assert_array_equal(with_it.H, without.H)


@pytest.mark.parametrize("kind", ["linear", "logistic", "cox"])
def test_a_design_with_no_aliased_column_is_drawn_whole(kind):
    """The guard is silent where no column is aliased: with the recode moved off the span by 1e−5
    of its length, every coefficient is drawn (none is zero), from the whole design."""
    M, _, rng = _columns(1, 300, "near", unit=1.0)
    fit = _draw_outcome(M, _outcome(kind, M, rng), np.random.default_rng(7))
    assert len(fit.beta) == M.shape[1] + (0 if kind == "cox" else 1)
    assert np.all(fit.beta != 0.0)


def test_smcfcs_imputes_with_an_aliased_column_as_it_does_without_it():
    """The sampler end to end, as the sweep's secondary model reached it: a linear outcome whose
    analysis model is quadratic in x (so SMC-FCS, Bartlett et al. 2015's design) beside a code's
    indicators, an energy and the code's recode; x blank on 30% of rows. The copies equal those
    drawn with the recode left out of the substantive model, value for value. At the base commit
    the chain raised ``LinAlgError`` in the outcome draw."""
    rng = np.random.default_rng(3)
    n = 300
    code = rng.integers(0, LEVELS, n)
    energy = rng.normal(2000.0, 900.0, n) * 1000.0
    x = rng.normal(2.0, 1.0, n)
    y = 4.0 - 4.0 * x + x ** 2 + 0.3 * (code == 2) + rng.normal(0.0, np.sqrt(2.0), n)
    data = pd.DataFrame({"x": np.where(rng.random(n) < 0.7, x, np.nan), "code": code,
                         "energy": energy})

    def design(with_recode: bool):
        def matrix(f: pd.DataFrame) -> np.ndarray:
            xv = f["x"].to_numpy(dtype=float)
            c = f["code"].to_numpy(dtype=float).astype(int)
            parts = [xv, xv ** 2, *[(c == k).astype(float) for k in range(1, LEVELS)],
                     f["energy"].to_numpy(dtype=float)]
            return np.column_stack(parts + ([RECODE[c]] if with_recode else []))
        return matrix

    copies = {}
    for with_recode in (True, False):
        sub = Substantive(Outcome("linear", y=y), design(with_recode))
        copies[with_recode] = impute(data, [Variable("x", "numeric")], mode="smcfcs",
                                     substantive=sub, m=3, iterations=4, seed=5,
                                     kinds={"code": "numeric"})
    for a, b in zip(copies[True].frames, copies[False].frames):
        np.testing.assert_array_equal(a["x"].to_numpy(), b["x"].to_numpy())
    assert np.isfinite(copies[True].frames[0]["x"].to_numpy()).all()


@pytest.mark.parametrize("alias", ["near", None])
def test_the_linear_draw_does_not_depend_on_a_column_s_unit(alias):
    """The energy in kcal and in thousandths of a kcal (× 1,000): the draw on the second is the
    draw on the first with the energy's coefficient divided by 1,000, every other coefficient and
    σ*² the same (to rounding). With the near-aliased recode beside it the design is the one whose
    draw raised at the base commit; without it, the absolute jitter made the small unit's draw
    differ."""
    M, at, rng = _columns(2, 300, alias or "constant", unit=1.0)
    if alias is None:
        M = np.delete(M, at, axis=1)
    y = _outcome("linear", M, rng).y
    small = M.copy()
    small[:, 5] *= 1000.0  # the energy column
    kcal = _draw_outcome(M, Outcome("linear", y=y), np.random.default_rng(11))
    milli = _draw_outcome(small, Outcome("linear", y=y), np.random.default_rng(11))
    expected = kcal.beta.copy()
    expected[6] /= 1000.0  # the energy's coefficient, behind the intercept
    np.testing.assert_allclose(milli.beta, expected, rtol=1e-6)
    assert milli.sigma2 == pytest.approx(kcal.sigma2, rel=1e-9)


def test_a_large_unit_s_drawn_variance_is_its_posterior_variance():
    """Income in dollars beside a number, 10,000 rows: over 4,000 draws the variance of each drawn
    coefficient is E[σ*²](XᵀX)⁻¹ within 10% (the sample variance of 4,000 draws has a relative
    standard error of √(2/4000) = 2.2%). At the base commit the income coefficient's drawn variance
    was 5 times its posterior variance (the absolute jitter beside a (XᵀX)⁻¹ entry of 2.5e−13)."""
    rng = np.random.default_rng(0)
    n = 10_000
    x = rng.normal(size=n)
    income = rng.normal(50_000.0, 20_000.0, n)
    y = x + 1e-5 * income + rng.normal(size=n)
    M = np.column_stack([x, income])
    X = np.column_stack([np.ones(n), M])
    beta, rss, *_ = np.linalg.lstsq(X, y, rcond=None)
    df = n - X.shape[1]
    expected = float(rss[0]) / df * df / (df - 2) * np.diag(np.linalg.inv(X.T @ X))
    draws = np.asarray([_draw_outcome(M, Outcome("linear", y=y),
                                      np.random.default_rng(10_000 + k)).beta
                        for k in range(4000)])
    ratio = draws.var(axis=0) / expected
    assert np.all(np.abs(ratio - 1.0) < 0.10), ratio
    # the draws' mean is the least-squares fit, within 4 Monte Carlo SE of each coefficient
    assert np.all(np.abs(draws.mean(axis=0) - beta) < 4 * np.sqrt(expected / 4000))


def test_every_stage_that_draws_smcfcs_copies_computes_again():
    """The stages whose artifacts pool SMC-FCS copies, each of which draws its own (the fit's
    coefficient table, the sensitivity analyses, the calibration's uncorrected estimate, the
    secondary model, the scales' uncorrected coefficient, the effects' declared models), are each
    one version past the base commit's, so none serves an artifact cached from copies drawn the old
    way."""
    from turbotab.core.stages import build_graph

    versions = {s.name: s.version for s in build_graph().stages()}
    stages = ("fit", "sensitivity", "calibration", "secondary", "scales", "effects")
    assert tuple(versions[s] for s in stages) == (33, 17, 15, 8, 6, 8)
