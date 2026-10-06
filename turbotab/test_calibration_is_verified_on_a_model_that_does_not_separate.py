"""`GUIDED-135` — the fixture rule at the opposite polarity.

`GUIDED-097` was written from *do not verify against the fixture that works*: a
0/1 target where `float()` succeeded, hiding a string-outcome defect for two
loops. **This is the mirror — the fixture that degenerately fails.**

`leaky_sepsis.csv` is the fixture behind every calibration claim this repository
makes *from a real project*, and its held-out C-statistic is **1.000**: complete
separation on 24 rows with 16 events. So `weak_calibration` returns
`(None, None)`, the annotation box renders *not estimable* for the intercept and
the slope with the reason attached, and the `annotation_box` checklist item
**fails**. Every one of those behaviors is correct, and `GUIDED-129` was closed
`NOT-A-DEFECT` on exactly that reading.

The consequence is what was wrong. The flagship clinical figure had been
asserted for six loops **only in the state where two of its seven required
numbers cannot exist**, so no table anybody could upload had ever been observed
producing a passing calibration figure.

## What was actually true, stated precisely

The checklist *has* passed — in
`test_a_figure_carries_its_checklist_and_its_companions.py::test_the_calibration_checklist_passes_on_a_real_render`,
against `_calibrated()`, which is **two synthetic numpy arrays**. What had never
happened is the checklist passing on a payload that came out of an
`AnalysisProject`: a file, a target, a seal, a fitted model and the held-out
rows. `figure_bundle` is the path a user reaches, and on that path the item had
only ever been observed red.

That distinction is the finding. A synthetic array pair proves the arithmetic;
it cannot prove that any table a researcher could bring produces the figure.

## The pair, and why it is a pair

Both fixtures are clinical, both are binary classification, both reach the
calibration figure through the same journey. They differ in **one** property:

| | `leaky_sepsis.csv` | `clinical_risk.csv` |
|---|---|---|
| held-out rows / events | 24 / 16 | 120 / 31 |
| C-statistic | 1.000 | 0.719 |
| intercept / slope | not estimable | +0.034 / 0.795 |
| `annotation_box` | **fails**, correctly | **passes** |

`leaky_sepsis` keeps its job. The not-estimable branch is a real path — a very
good model on a small sample is exactly what produces it — and a fixture that
holds it is worth having. What it cannot do is show the figure passing, and
until `clinical_risk.csv` nothing else could either.
"""
# The tests here that drove the retired legacy app (turbotab/api.py, the page, the
# figure and manuscript modules, the gates) were removed with it, BLUEPRINT §9.1; git keeps them.
from __future__ import annotations

import pandas as pd


#: `GUIDED-097`, applied. Two clinical fixtures of the same target shape and
#: opposite CALIBRATION shape — which is the axis this figure lives on, and the
#: one the fixture rule had never been applied to.
#:
#: `separates` is the expectation, asserted rather than discovered: a fixture
#: that quietly stopped separating would silently turn this pair back into one
#: fixture, and the assertion is what would notice.
CALIBRATION_FIXTURES = {
    "separating — complete separation, two numbers undefined": {
        "file": "leaky_sepsis.csv", "target": "sepsis", "separates": True,
    },
    "ordinary — a model that does not separate": {
        "file": "clinical_risk.csv", "target": "readmit_30d", "separates": False,
    },
}

#: NOT COVERED, said out loud. A sweep that reports only what it covered has not
#: reported its coverage.
#:
#: A STRING-LABELED CLINICAL OUTCOME. `clinic_visits.csv` has one and is not a
#: prediction fixture — it carries no model-worthy predictors — so the
#: calibration pair is two numeric 0/1 targets. `predictions_for` binarizes
#: against `positive_label` and `GUIDED-093`'s own test covers that path; what
#: is uncovered is the *combination* of a string label with a non-separating
#: fit.
#:
#: MULTICLASS. `multiclass_stage.csv` exists and the clinical figures decline
#: it by design (`test_the_clinical_figures_decline_a_three_class_target`), so
#: there is no multiclass calibration branch to verify. Nothing here changes
#: that; `GUIDED-132` is the open row.
#:
#: A COHORT LARGE ENOUGH FOR THE CURVE'S OWN CONFIDENCE BAND. Both fixtures are
#: small enough to load instantly, which is the property a drive needs. Neither
#: exercises a 10-bin flexible curve at n in the thousands.
#:
#: CENSORED / SURVIVAL. `GUIDED-118`; the refusal stands.
SHAPES_NOT_COVERED = [
    "a string-labeled clinical outcome fitted to a NON-separating model — "
    "the two halves are covered separately and never together",
    "multiclass — the clinical figures decline a three-class target by design "
    "(GUIDED-132 is the open row about the shelf, not about this figure)",
    "n in the thousands — both fixtures are sized to load instantly",
    "time-to-event (GUIDED-118, refusal stands)",
]

#: The seven the box is required to carry. Named here rather than re-derived in
#: each test, because the count is the claim: five would pass on
#: `leaky_sepsis.csv` too.
SEVEN_NUMBERS = ("calibration_intercept", "calibration_slope", "c_statistic",
                 "e_avg", "e_max", "n", "events")


# ═══════════ THE FIXTURE IS WHAT IT SAYS IT IS ═══════════

def test_the_new_fixture_carries_no_leakage_and_says_so():
    """`clinical_risk.csv`'s companion `.md` claims no column is measured after
    the outcome. A claim in a markdown file that nothing checks is a claim that
    decays — `README.md`'s own opening paragraph is this project's worked
    example of that.

    Checked as a **correlation ceiling** rather than by naming columns, because
    the property is *no proxy for the outcome*, and a name list would pass a
    fixture that grew one.
    """
    df = pd.read_csv("turbotab/sample_data/clinical_risk.csv")
    numeric = df.select_dtypes("number").drop(columns=["readmit_30d"])
    worst = numeric.corrwith(df["readmit_30d"]).abs().max()
    assert worst < 0.5, (
        f"a predictor correlates {worst:.4f} with the outcome; leakage is "
        f"`leaky_sepsis.csv`'s job and a second fixture carrying it would make "
        f"both files about two things")

    sepsis = pd.read_csv("turbotab/sample_data/leaky_sepsis.csv")
    leak = (sepsis.select_dtypes("number").drop(columns=["sepsis"])
            .corrwith(sepsis["sepsis"]).abs().max())
    assert leak > 0.99, (
        "leaky_sepsis.csv no longer leaks, which is the one thing it is for")


def test_the_fixture_generator_reproduces_the_committed_file():
    """A fixture whose generator has drifted from its output is a fixture
    nobody can adjust — the reason `make_fixtures.py` is committed at all."""
    import sys

    sys.path.insert(0, "turbotab/sample_data")
    import make_fixtures                                       # noqa: E402

    built = make_fixtures.clinical_risk()
    on_disk = pd.read_csv("turbotab/sample_data/clinical_risk.csv")
    assert list(built.columns) == list(on_disk.columns)
    assert len(built) == len(on_disk)
    pd.testing.assert_frame_equal(
        built.reset_index(drop=True), on_disk.reset_index(drop=True),
        check_dtype=False)
