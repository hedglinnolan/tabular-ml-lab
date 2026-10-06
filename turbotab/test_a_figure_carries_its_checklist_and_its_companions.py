"""`GUIDED-051` — the figure spec, and the two figures that had to survive it.

`DOMAIN_SCIENCE.md` §02. The app has seven geometries and nineteen EDA actions
and none of them knows what field it is looking at. The research says that is
the wrong axis:

> Every pack specified its signature figures as a **checklist**, and the
> checklist items are overwhelmingly about **annotation rather than geometry**.

So the figure layer is a caption-and-annotation engine wrapped around a plotting
library, not the other way round — and the spec has five fields, of which
`companions` has no analogue in the app today and is the load-bearing one.

## Two figures, deliberately

Calibration and PCA scores were chosen to be maximally different: confirmatory
against exploratory, needs-a-fitted-model against needs-a-numeric-block,
do-not-truncate against aspect-proportional-to-variance, has-a-companion against
makes-no-claim, not-promotable against promotable. A third is not built until
the spec has survived both.

## What the seams turned out to be

**The checklist found a real gap on its first run.** `annotation_box` requires
the calibration intercept and slope, and `ml/calibration.py` computed neither —
it had the hierarchy's first and third rungs and not the second, which the
clinical pack calls mandatory and which is the single most useful number on the
figure. The item failed against a real render, which is exactly what a checklist
scored against a render is for.

**A checklist item can be about what is NOT done.** *"Do not truncate the axis"*
is scored by comparing the drawn range against the observed one, so the payload
has to carry both. An earlier draft carried only what it drew, and the item was
unscoreable — `GUIDED-045`'s axis one layer into the figure layer.
"""
# The tests here that drove the retired legacy app (turbotab/api.py, the page, the
# figure and manuscript modules, the gates) were removed with it, BLUEPRINT §9.1; git keeps them.
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbotab import cohort_findings as CF                            # noqa: E402


def _calibrated(n=600, extreme=1.0, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    truth = 1.0 / (1.0 + np.exp(-x))
    y = rng.binomial(1, truth)
    p = 1.0 / (1.0 + np.exp(-extreme * x))
    return y, p


def _assay(seed=1, n=60, p=25):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.lognormal(0, 1, (n, p)),
                      columns=[f"mz_{i:03d}" for i in range(p)])
    df["group"] = ["case"] * (n // 2) + ["control"] * (n - n // 2)
    return df, [i % 10 == 0 for i in range(n)]


# ── the calibration plot ────────────────────────────────────────────────────


def test_the_weak_calibration_numbers_are_the_engines_and_are_right():
    """The gap the checklist found, closed in the engine rather than here.

    Perfect calibration is intercept 0 and slope 1; predictions twice too
    extreme give a slope near 0.5, which is the reading that makes the number
    worth printing.
    """
    from ml.calibration import c_statistic, weak_calibration

    y, p = _calibrated(n=4000)
    intercept, slope = weak_calibration(y, p)
    assert abs(intercept) < 0.15 and abs(slope - 1.0) < 0.15

    _, too_extreme = _calibrated(n=4000, extreme=2.0)
    _, slope2 = weak_calibration(y, too_extreme)
    assert 0.35 < slope2 < 0.65, slope2

    assert 0.5 < c_statistic(y, p) < 1.0


def test_an_undefined_fit_reports_nothing_rather_than_perfection():
    """`(0.0, 1.0)` are the values of PERFECT calibration. Returning them for
    'could not compute' would be the app reporting an ideal result where it has
    none — the governing rule's own failure, in two floats."""
    from ml.calibration import c_statistic, weak_calibration

    assert weak_calibration(np.ones(20), np.full(20, 0.5)) == (None, None)
    assert weak_calibration(np.r_[np.ones(10), np.zeros(10)],
                            np.full(20, 0.3)) == (None, None)
    assert c_statistic(np.ones(10), np.linspace(0, 1, 10)) is None


# ── Part D · the finding with no column ─────────────────────────────────────

def test_a_finding_about_the_cohort_states_its_scope_rather_than_nothing():
    """An empty chip row inside a full card frame reads as a card that failed
    to load. An absence is read as a missing name, never as 'there is no
    name'."""
    finding = {"id": "profile_sample_size_0", "severity": "warning",
               "title": "Small sample", "fix_kind": "none", "params": {}}
    shape = CF.render_shape(finding)
    assert shape["scope"] == CF.COHORT
    assert shape["has_chips"] is False
    assert shape["subject_line"], "the card would render an empty subject"
    assert "study as a whole" in shape["subject_line"]


def test_the_scope_is_derived_and_not_a_field_a_producer_must_remember():
    """Two of the producers are frozen modules that know nothing about scopes,
    and a finding whose scope was never set would default to `columns` and
    render the empty chip row this exists to prevent."""
    assert CF.scope_of({"affected_columns": ["age"]}) == CF.COLUMNS
    assert CF.scope_of({"params": {"columns": ["age"]}}) == CF.COLUMNS
    assert CF.scope_of({"params": {"rows": [1, 2, 3]}}) == CF.ROWS
    assert CF.scope_of({"params": {}}) == CF.COHORT
    # An explicit scope wins, so a producer that DOES know can say so.
    assert CF.scope_of({"scope": CF.COHORT, "affected_columns": ["age"]}) == CF.COHORT


def test_a_repair_whose_columns_are_gone_is_withdrawn_rather_than_offered():
    """`ml/import_doctor.py:954` filters a finding's columns against the frame,
    and an empty intersection drops nothing and reports having dropped nothing —
    a repair that silently succeeds at doing nothing.

    That module is frozen, so the Guided door does the reading BEFORE offering
    the repair, and the refusal says which columns are gone rather than
    rendering a button that will no-op.
    """
    finding = {"id": "drop_columns", "fix_kind": "drop_columns",
               "affected_columns": ["ghost_a", "ghost_b"], "params": {}}
    refusal = CF.check_subject_survives(finding, ["age", "sex"])
    # On the distinctive CLAIM, not on a phrase that happens to be nearby —
    # the first draft asserted "not in the table any more" and the sentence
    # reads "none of those columns are in the table any more".
    assert refusal
    assert "withdrawn rather than offered" in refusal
    assert "change nothing and report success" in refusal
    assert "`ghost_a`" in refusal, "the refusal does not name what is gone"
    # A finding that still has a subject is offered normally.
    assert CF.check_subject_survives(finding, ["ghost_a"]) is None
    # And a cohort finding has no columns to lose.
    assert CF.check_subject_survives({"params": {}}, []) is None
