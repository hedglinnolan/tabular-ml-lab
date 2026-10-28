"""A column far from its own origin (an SD of one about a mean of 1,000 or 1,000,000) beside `x1`:
the Cox, survey Cox, mixed and GEE tables are served at the numbers of the same column at its
origin, and a pair of columns that nearly copy each other is still refused as nearly singular,
naming both (T1V2-1, 2026-10-27).

On the base those tables read their near-singular concern (``linear.collinearity_concern``) on the
columns as they are, beside an intercept column, so a shifted column was nearly a copy of the
intercept (a scaled condition number of 1,973 at a mean of 1,000) and each table was refused in
words that called one column "close to a fixed combination of each other", though a slope does not
move with its column's origin and each fit is right there: the Cox fit centers its columns, REML
factors the design by QR, and the GEE is now fit on centered columns (``inference.by_identified``).
The concern now reads the columns centered at their means, the intercept or the baseline hazard
taking their origin, as each table is fit.

Truth from construction: shifting a column by ``M`` leaves every slope's row unchanged, every
field, and moves the intercept by ``−b·M``; the unshifted tables are anchored at statsmodels'
PHReg (Efron ties) and MixedLM (REML). The fires side: ``x2 = x1 + 10⁻³·d`` has a centered,
scaled condition number near 2,000, past the concern's 1,000.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from turbotab.core.models.inference import INDEPENDENT, Clusters
from turbotab.core.models.linear import NEAR_SINGULAR, collinearity_concern
from turbotab.core.models.repeated import gee_table, mixed_table
from turbotab.core.models.survey import SurveyDesign
from turbotab.core.models.survival import cox_table, survey_cox_table, survival_outcome

N, G = 400, 80
SHIFTS = (1e3, 1e6)


def _data():
    rng = np.random.default_rng(42)
    codes = np.repeat(np.arange(G), N // G).astype(np.int64)
    x1, year, d = rng.normal(size=(3, N))
    eta = 0.4 * x1 + 0.2 * year
    yr = 1 + eta + rng.normal(size=N) + rng.normal(size=G)[codes]
    yb = (rng.random(N) < 1 / (1 + np.exp(-eta))).astype(int)
    ys = survival_outcome(rng.random(N) < 0.7, rng.exponential(np.exp(-eta)))
    strata = np.repeat(np.arange(10), N // 10)
    design = SurveyDesign(row_ids=np.arange(N), weight=rng.uniform(1, 3, N),
                          stratum=strata.astype(np.int64),
                          psu=(strata * 4 + np.arange(N) % 4).astype(np.int64),
                          weight_column="w", strata_column="s", psu_column="p")
    return x1, year, d, codes, yr, yb, ys, design


def _tables():
    _, _, _, codes, yr, yb, ys, design = _data()
    cl = Clusters(column="pid", codes=codes, n_clusters=G, n_missing=0)
    S = {"study": ["x1"]}
    return {
        "Cox": lambda m: cox_table(m, ys, INDEPENDENT, **S),
        "Cox (clustered)": lambda m: cox_table(m, ys, cl, **S),
        "survey Cox": lambda m: survey_cox_table(m, ys, design, **S),
        "mixed model": lambda m: mixed_table(m, yr, cl, **S),
        "GEE": lambda m: gee_table(m, yr, cl, "regression", **S),
        "GEE (logit)": lambda m: gee_table(m, yb, cl, "binary", [0, 1], **S),
    }


def _frame(**columns) -> pd.DataFrame:
    return pd.DataFrame(columns, index=pd.RangeIndex(N))


def _same(a, b, rel: float) -> bool:
    if a is None or b is None:
        return a is b
    return a == pytest.approx(b, rel=rel, abs=1e-12)


@pytest.mark.parametrize("shift", SHIFTS)
@pytest.mark.parametrize("family", list(_tables()))
def test_a_column_far_from_its_origin_is_served_at_the_tables_numbers_at_its_origin(family, shift):
    """Every slope's row is the unshifted table's, every field (to 10⁻⁷), and the intercept's
    estimate is moved by `year`'s coefficient times the shift. On the base each table was refused
    as nearly singular."""
    x1, year, *_ = _data()
    build = _tables()[family]
    table = build(_frame(x1=x1, year=year + shift))
    reference = build(_frame(x1=x1, year=year))
    assert reference.info["refused"] is None, reference.info["refused"]
    assert table.info["refused"] is None, table.info["refused"]
    ref = {r["feature"]: r for r in reference.rows}
    assert [r["feature"] for r in table.rows] == list(ref)
    for row in table.rows:
        other = ref[row["feature"]]
        if row["feature"] == "(intercept)":
            moved = other["estimate"] - ref["year"]["estimate"] * shift
            assert row["estimate"] == pytest.approx(moved, rel=1e-7, abs=1e-6)
            continue
        for key in ("estimate", "se", "ci_low", "ci_high", "df", "p"):
            assert _same(row[key], other[key], 1e-7), (row["feature"], key)


def test_the_unshifted_cox_and_mixed_tables_are_statsmodels():
    """The anchor: the Cox table is PHReg's (Efron ties) and the mixed table's estimates REML's."""
    x1, year, _, codes, yr, _, ys, _ = _data()
    frame = _frame(x1=x1, year=year)
    cox = cox_table(frame, ys, INDEPENDENT, study=["x1"])
    phreg = sm.PHReg(ys["time"], frame.to_numpy(), status=ys["event"].astype(int),
                     ties="efron").fit()
    np.testing.assert_allclose([r["estimate"] for r in cox.rows], phreg.params, rtol=1e-7)
    np.testing.assert_allclose([r["se"] for r in cox.rows], phreg.bse, rtol=1e-6)
    cl = Clusters(column="pid", codes=codes, n_clusters=G, n_missing=0)
    mixed = mixed_table(frame, yr, cl, study=["x1"])
    reml = sm.MixedLM(yr, sm.add_constant(frame.to_numpy()), groups=codes).fit(reml=True)
    np.testing.assert_allclose([r["estimate"] for r in mixed.rows], reml.fe_params, rtol=1e-5)


@pytest.mark.parametrize("family", list(_tables()))
def test_a_pair_that_nearly_copies_itself_is_still_refused_naming_both(family):
    """The fires side: `x1` beside `x2 = x1 + 10⁻³·d` (and `year` beside them) is refused as nearly
    singular, in words that name `x1` and `x2` and not `year`, with no interval."""
    x1, year, d, *_ = _data()
    table = _tables()[family](_frame(x1=x1, x2=x1 + 1e-3 * d, year=year + 1e3))
    reason = table.info["refused"]
    assert reason is not None and reason.startswith("The model matrix is nearly singular"), reason
    assert "x1" in reason and "x2" in reason and "year" not in reason
    assert all(r.get("ci_low") is None and r.get("p") is None for r in table.rows)


def test_the_concern_reads_no_origin_and_names_a_near_copy_in_its_own_condition_number():
    """``collinearity_concern`` is silent for a column far from its origin, whose condition number
    beside the intercept as it stands is past the concern's mark (the base's reading), and fires
    for a near copy, naming both, at the condition number of the centered columns each scaled to
    unit length, written out here."""
    x1, year, d, *_ = _data()
    shifted = _frame(x1=x1, year=year + 1e3)
    X = np.column_stack([np.ones(N), shifted.to_numpy()])
    s = np.linalg.svd(X / np.linalg.norm(X, axis=0), compute_uv=False)
    assert s[0] / s[-1] > NEAR_SINGULAR
    assert collinearity_concern(shifted) is None
    pair = _frame(x1=x1, x2=x1 + 1e-3 * d)
    Z = pair.to_numpy() - pair.to_numpy().mean(axis=0)
    s = np.linalg.svd(Z / np.linalg.norm(Z, axis=0), compute_uv=False)
    assert collinearity_concern(pair) == (
        f"The model matrix is nearly singular (scaled condition number {s[0] / s[-1]:,.0f}): x1 "
        f"and x2 are close to a fixed combination of each other, so their separate coefficients "
        f"are not well determined; leave one of them out.")
