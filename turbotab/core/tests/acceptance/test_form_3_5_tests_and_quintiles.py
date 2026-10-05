"""FORM · 3 and 5 · the spline's test of association is its overall test, never a linear refit, and
the quintiles are produced beside it with the p for linear trend labeled customary
(MODELING_SEQUENCE §1 row 5, §0 ruling 2).

The independent references are written out here with NumPy: least squares with its HC3 covariance
(``form_fixtures.ols_hc3``), Harrell's restricted cubic spline basis from his formula (RMS eq. 2.25
over (t_k − t₁)²), R's type-7 quantiles for the quintile boundaries, each quintile's median, and the
Wald F on the HC3 covariance. The cohort is ESTIMAND's (``estimand_fixtures.cohort``): glucose
falls with fiber along a straight line, so the truth has no curvature to find.

3. A spline declared on an exposure whose relation is a straight line: its nonlinearity test is not
   significant, and the fit still carries every spline term; the reported test of association is
   the overall Wald test (computed here by hand), and no 1-df test of a refitted line appears.
5. Beside the declared exposure's spline, the quintile table: each quintile against the lowest, the
   boundaries and the reference stated (STROBE 16b), and the "p for linear trend (customary)" from
   the category medians, each agreeing with a NumPy refit to 1e-8.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from turbotab.core import decisions as d
from turbotab.core.methods import exposure_form as ef
from turbotab.core.tests.acceptance import estimand_fixtures as est_f
from turbotab.core.tests.acceptance import form_fixtures as ff

ADJUSTED = ["age", "sex", "smoking", "activity"]  # the primary set the answers derive (bmi: secondary)


def _rcs_by_hand(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Harrell 2015, eq. 2.25 (Devlin & Weeks), divided by (t_k − t₁)² (rcspline.eval's norm 2)."""
    k = len(t)
    p3 = lambda u: np.where(u > 0, u, 0.0) ** 3  # noqa: E731
    cols = []
    for j in range(k - 2):
        cols.append((p3(x - t[j]) - p3(x - t[k - 2]) * (t[k - 1] - t[j]) / (t[k - 1] - t[k - 2])
                     + p3(x - t[k - 1]) * (t[k - 2] - t[j]) / (t[k - 1] - t[k - 2]))
                    / (t[k - 1] - t[0]) ** 2)
    return np.column_stack(cols)


@pytest.fixture(scope="module")
def spline_run(tmp_path_factory) -> dict[str, Any]:
    frame = est_f.cohort(n=800, seed=7)
    st = est_f.state(target="glucose", task="regression", measure="mean_difference",
                     exposure_forms={"fiber": d.ExposureFormSpec(form="spline", knots=4)})
    run = est_f.run(frame, tmp_path_factory.mktemp("spline"), st)
    return {"frame": frame, "run": run, "model": run["fit_raw"]["models"][0]}


def _design(frame: pd.DataFrame, spline: np.ndarray | None = None,
            extra: dict[str, np.ndarray] | None = None) -> tuple[np.ndarray, list[str]]:
    X = est_f.design_matrix(frame, ADJUSTED)
    if spline is not None:
        X["fiber"] = frame["fiber"].to_numpy(float)
        for j in range(spline.shape[1]):
            X[f"fiber{chr(39) * (j + 1)}"] = spline[:, j]
    for name, values in (extra or {}).items():
        X[name] = values
    return X.to_numpy(float), list(X.columns)


# ── 3 · the overall test is the test of association; no silent linear refit ──


def test_3_a_non_significant_nonlinearity_test_keeps_every_spline_term(spline_run):
    model = spline_run["model"]
    tests = [t for t in model["exposure_tests"] if t["column"] == "fiber"]
    by = {t["test"]: t for t in tests}
    assert by["nonlinear"]["p"] > 0.05  # the truth is a straight line
    features = {r["feature"] for r in model["coefficients"]}
    assert {"fiber", "fiber'", "fiber''"} <= features  # the curve is kept, not refit as a line
    # The overall test is the reported test of association, written out here: F on the three
    # spline terms' HC3 covariance, on n − p df.
    frame = spline_run["frame"]
    knots = np.asarray(by["overall"]["knots"])
    assert np.allclose(knots, ff.harrell_knots(frame["fiber"], 4), rtol=0, atol=1e-10)
    X, names = _design(frame, _rcs_by_hand(frame["fiber"].to_numpy(float), knots))
    beta, cov = ff.ols_hc3(X, frame["glucose"].to_numpy(float))
    idx = [names.index(n) for n in ("fiber", "fiber'", "fiber''")]
    F, p = ff.wald_f(beta, cov, idx, len(frame) - X.shape[1])
    assert by["overall"]["statistic"] == pytest.approx(F, rel=1e-8)
    assert by["overall"]["p"] == pytest.approx(p, rel=1e-8)
    F_nl, p_nl = ff.wald_f(beta, cov, idx[1:], len(frame) - X.shape[1])
    assert by["nonlinear"]["statistic"] == pytest.approx(F_nl, rel=1e-8)
    assert by["overall"]["caption"].startswith("The test of association: Wald test that all 3 "
                                               "terms of `fiber` are zero (HC3 covariance)")
    assert by["nonlinear"]["caption"].endswith(
        "A non-significant result does not refit a straight line: dropping the curve after its "
        "test inflates the test of association's type I error (Grambsch & O'Brien 1991, Stat Med "
        "10:697).")
    # No test of a refitted line: the spline's two tests and the quintiles beside it, nothing else.
    assert {t["test"] for t in tests} == {"overall", "nonlinear", "companion_q2", "companion_q3",
                                          "companion_q4", "companion_q5", "companion_trend"}


def test_3_the_contract_disables_a_linear_refit_and_names_the_code_that_enforces_it():
    from turbotab.core.contracts import contract

    relation = contract("functional_form").relation("no-silent-linear-refit")
    assert relation.kind == "disables" and relation.purposes == ("inference",)
    assert relation.enforced_by == "turbotab.core.methods.exposure_form:exposure_tests"


# ── 5 · quintiles beside the spline, the trend labeled customary ─────────────


def test_5_quintiles_are_produced_beside_the_spline_with_boundaries_and_reference(spline_run):
    frame = spline_run["frame"]
    x = frame["fiber"].to_numpy(float)
    cuts = np.quantile(x, [0.2, 0.4, 0.6, 0.8], method="linear")  # R type 7
    tests = {t["test"]: t for t in spline_run["model"]["exposure_tests"] if t["column"] == "fiber"}
    for g in range(2, 6):
        t = tests[f"companion_q{g}"]
        assert np.allclose(t["boundaries"], cuts, rtol=0, atol=1e-12)
        assert t["reference"] == f"quintile 1, `fiber` ≤ {cuts[0]:.4g}"
        assert t["form"] == "quintiles"
    words = (f"quintile 1: `fiber` ≤ {cuts[0]:.4g} (the reference); quintile 2: {cuts[0]:.4g} < "
             f"`fiber` ≤ {cuts[1]:.4g}; quintile 3: {cuts[1]:.4g} < `fiber` ≤ {cuts[2]:.4g}; "
             f"quintile 4: {cuts[2]:.4g} < `fiber` ≤ {cuts[3]:.4g}; quintile 5: `fiber` > "
             f"{cuts[3]:.4g}")
    assert tests["companion_q2"]["caption"].endswith(f"; {words}.")
    # Each quintile against the lowest: least squares with the four indicators in place of the
    # spline, HC3, written out.
    group = np.searchsorted(cuts, x, side="left")
    indicators = {f"fiber_Q{g}": (group == g - 1).astype(float) for g in range(2, 6)}
    X, names = _design(frame, extra=indicators)
    beta, cov = ff.ols_hc3(X, frame["glucose"].to_numpy(float))
    for g in range(2, 6):
        j = names.index(f"fiber_Q{g}")
        t = tests[f"companion_q{g}"]
        assert t["estimate"] == pytest.approx(beta[j], rel=1e-8)
        assert (t["estimate"] / t["statistic"]) == pytest.approx(np.sqrt(cov[j, j]), rel=1e-8)


def test_5_the_p_for_linear_trend_is_labeled_customary_and_agrees_with_a_numpy_refit(spline_run):
    frame = spline_run["frame"]
    x = frame["fiber"].to_numpy(float)
    cuts = np.quantile(x, [0.2, 0.4, 0.6, 0.8], method="linear")
    group = np.searchsorted(cuts, x, side="left")
    medians = np.array([np.median(x[group == g]) for g in range(5)])
    score = medians[group]
    X, names = _design(frame, extra={"score": score})
    beta, cov = ff.ols_hc3(X, frame["glucose"].to_numpy(float))
    j = names.index("score")
    from scipy import stats

    se = np.sqrt(cov[j, j])
    p = 2 * stats.t.sf(abs(beta[j] / se), len(frame) - X.shape[1])
    trend = next(t for t in spline_run["model"]["exposure_tests"]
                 if t["column"] == "fiber" and t["test"] == "companion_trend")
    assert trend["label"] == "p for linear trend (customary)" == ef.TREND_LABEL
    assert np.allclose(trend["medians"], medians, rtol=0, atol=1e-12)
    assert trend["estimate"] == pytest.approx(beta[j], rel=1e-8)
    assert trend["p"] == pytest.approx(p, rel=1e-8)
    assert trend["caption"].startswith("Beside the spline, the p for linear trend (customary): ")
    assert trend["caption"].endswith("The spline's overall test is the test of association; this "
                                     "is a test of a linear trend, not of a dose–response.")


def test_5_quintiles_as_the_form_label_their_trend_customary_too(tmp_path):
    frame = est_f.cohort(n=600, seed=9)
    st = est_f.state(target="glucose", task="regression", measure="mean_difference",
                     exposure_forms={"fiber": d.ExposureFormSpec(form="quintiles")})
    model = est_f.run(frame, tmp_path, st)["fit_raw"]["models"][0]
    trend = next(t for t in model["exposure_tests"] if t["test"] == "trend")
    assert trend["label"] == "p for linear trend (customary)"
    assert trend["caption"].startswith("The p for linear trend (customary) across quintiles of "
                                       "`fiber`")
    assert trend["reference"].startswith("quintile 1, `fiber` ≤ ")
    # under inference quintiles rank lower; the spline leads; the companion is produced by default
    options = {o["value"]: o["rung"] for o in ef.options("inference")}
    assert options["spline"] == "recommended" and options["quintiles"] == "rank_lower"
