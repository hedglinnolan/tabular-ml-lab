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
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from turbotab import eventfixture
from turbotab import figure_bundle as FB
from turbotab import figure_specs as FS
from turbotab import figures
from turbotab import training as T
from turbotab.project import AnalysisProject

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


def _fitted(spec) -> AnalysisProject:
    """A sealed, fitted project — the path a user reaches, not an array pair.

    The whole point of this file is that the payload comes out of a *project*.
    Calling `calibration_payload` on two numpy vectors proves the arithmetic and
    proves nothing about whether any table produces it.
    """
    df = pd.read_csv(f"turbotab/sample_data/{spec['file']}")
    df = df[df[spec["target"]].notna()].copy()
    p = AnalysisProject.from_dataframe(df, spec["file"])
    p.target, p.task_type = spec["target"], "classification"
    p.set_grain("not_sure")
    p.set_eligibility("everyone")
    rng = np.random.default_rng(42)
    idx = list(p.df.index)
    rng.shuffle(idx)
    labels = idx[:int(round(len(idx) * 0.25))]
    p.seal_lockbox(labels, fraction=len(labels) / len(p.df))
    # `DRIVE-041`. The fit refuses while nobody has said which level is the
    # event, and this fixture never said. Recorded through `engine.record_fix`,
    # the same function the page's answer travels, rather than written onto the
    # project — see `turbotab/eventfixture.py`.
    eventfixture.choose_event(p, required=True)
    p.training_run = T.train(p, ["logreg"])
    return p


def _calibration_row(project):
    bundle = FB.render(project)
    row = next((r for r in bundle["admitted"] + bundle["held"]
                if r["id"] == "calibration"), None)
    if row is None:
        why = {r["id"]: r["why"] for r in bundle["unavailable"]}
        pytest.fail("the calibration figure was not drawn at all: "
                    f"{why.get('calibration', 'not offered')}")
    return row


# ═══════════ THE PAIR IS A PAIR ═══════════

@pytest.mark.parametrize("shape", sorted(CALIBRATION_FIXTURES))
def test_the_calibration_figure_is_drawn_from_a_real_project(shape):
    """Both fixtures reach the figure. The difference is in the numbers, and
    a difference in reachability would make the comparison meaningless."""
    row = _calibration_row(_fitted(CALIBRATION_FIXTURES[shape]))
    assert row["payload"]["scored_on"] == "held-out rows only"
    assert row["caption"] and row["annotations"]


@pytest.mark.parametrize("shape", sorted(CALIBRATION_FIXTURES))
def test_the_fixture_separates_exactly_when_it_is_declared_to(shape):
    """**The assertion that keeps this a pair.**

    If `clinical_risk.csv` were regenerated into a separating model, or
    `leaky_sepsis.csv` into a mixed one, every test below would still pass while
    the coverage silently collapsed back to one fixture. `separates` is a
    declared expectation and this is what checks it.
    """
    spec = CALIBRATION_FIXTURES[shape]
    payload = _calibration_row(_fitted(spec))["payload"]
    undefined = payload["calibration_intercept"] is None
    assert undefined is spec["separates"], (
        f"{spec['file']} was declared separates={spec['separates']} and the "
        f"weak-calibration fit is "
        f"{'undefined' if undefined else 'defined'}: "
        f"C={payload['c_statistic']}, intercept={payload['calibration_intercept']}")


# ═══════════ THE BRANCH THAT HAD NEVER BEEN OBSERVED ═══════════

def test_a_model_that_does_not_separate_produces_all_seven_numbers():
    """`GUIDED-135`'s whole point, and it is a *project*, not two arrays."""
    payload = _calibration_row(
        _fitted(CALIBRATION_FIXTURES[
            "ordinary — a model that does not separate"]))["payload"]
    missing = [k for k in SEVEN_NUMBERS if payload.get(k) is None]
    assert not missing, (
        f"the non-separating fixture is missing {missing}, which is the state "
        f"this file exists to make impossible")
    # The slope below 1 is the reading that makes the number worth printing —
    # predictions too extreme, which is what a small sample does to a model.
    assert 0.0 < payload["calibration_slope"] < 1.0, payload["calibration_slope"]
    assert 0.5 < payload["c_statistic"] < 1.0, (
        "a C-statistic at 1.0 is the separating case and at 0.5 is a coin "
        "flip; neither can demonstrate this figure")


def test_the_annotation_box_checklist_item_passes_for_the_first_time():
    """Six loops of this item being red on every project anybody could build.

    Scored against the **rendered bundle row**, not against a payload composed
    here, because `figures.bundle` is what a consumer reads.
    """
    row = _calibration_row(
        _fitted(CALIBRATION_FIXTURES[
            "ordinary — a model that does not separate"]))
    failed = [item["id"] for item in row["checklist"] if not item["passed"]]
    assert not failed, (
        f"the calibration figure still fails {failed} on a fixture built to "
        f"make it pass")
    assert len(row["checklist"]) == 5
    assert "annotation_box" in {i["id"] for i in row["checklist"]}


def test_no_annotation_renders_as_a_blank_on_the_passing_fixture():
    row = _calibration_row(
        _fitted(CALIBRATION_FIXTURES[
            "ordinary — a model that does not separate"]))
    for annotation in row["annotations"]:
        assert annotation["value"] not in ("", None), annotation
        assert annotation["value"] != "not estimable", (
            f"{annotation['label']} is not estimable on the fixture chosen "
            f"because everything is estimable on it")


# ═══════════ AND THE BRANCH THAT WAS ALL THERE WAS ═══════════

def test_the_separating_fixture_still_refuses_the_two_numbers_it_cannot_have():
    """**`leaky_sepsis.csv` keeps its job.** This is not a regression to route
    around — it is the honest branch, and losing it would cost more than the
    passing one gained.

    `weak_calibration` returns `(None, None)` under separation rather than
    `(0.0, 1.0)`, which are the values of *perfect* calibration. Returning those
    from ignorance is the trap the project has named nine times.
    """
    row = _calibration_row(
        _fitted(CALIBRATION_FIXTURES[
            "separating — complete separation, two numbers undefined"]))
    payload = row["payload"]
    assert payload["calibration_intercept"] is None
    assert payload["calibration_slope"] is None
    assert payload["c_statistic"] == pytest.approx(1.0), (
        "this fixture is here because it separates; it no longer does")
    # The five that ARE defined are still there — the figure degrades, it does
    # not collapse.
    for key in ("c_statistic", "e_avg", "e_max", "n", "events"):
        assert payload.get(key) is not None, key


def test_the_separating_fixture_states_the_absence_and_fails_the_item():
    """Two different jobs, and the docstring on `annotation_box` says so:
    *failing the checklist and rendering honestly are different jobs.*

    The box renders the ABSENCE with its reason, and the checklist item still
    goes red — because a figure without those numbers is not publication-grade.
    A rendering that passed the item by writing something in the cell would be
    the app asserting a number it does not have.
    """
    row = _calibration_row(
        _fitted(CALIBRATION_FIXTURES[
            "separating — complete separation, two numbers undefined"]))
    failed = {item["id"] for item in row["checklist"] if not item["passed"]}
    assert failed == {"annotation_box"}, (
        f"expected exactly annotation_box to fail under separation, got "
        f"{failed}")

    # Keyed on `key`, not on `label`. `annotation_rows` lets a figure's own
    # `annotation_box` win, and this figure's box carries SHORTER labels than
    # its spec declares — `Calibration intercept` against the spec's
    # `Calibration intercept (95% CI)`. That gap is `GUIDED-137`, filed rather
    # than fixed here; keying on the label would make this test depend on which
    # of the two strings won.
    stated = {a["key"]: a for a in row["annotations"]
              if a["value"] == "not estimable"}
    assert {"calibration_intercept", "calibration_slope"} <= set(stated), (
        f"the undefined numbers are not stated as undefined: "
        f"{[(a['key'], a['value']) for a in row['annotations']]}")
    for annotation in stated.values():
        assert annotation["why"], (
            "`not estimable` with no reason reads as a rendering fault, which "
            "is the thing the absence-rendering exists to prevent")


def test_the_caption_says_not_estimable_rather_than_printing_nothing():
    """The caption is prose a reviewer reads, and it is composed from the same
    payload. A blank there would be worse than in the box."""
    separating = _calibration_row(
        _fitted(CALIBRATION_FIXTURES[
            "separating — complete separation, two numbers undefined"]))
    assert "not estimable" in separating["caption"]

    ordinary = _calibration_row(
        _fitted(CALIBRATION_FIXTURES[
            "ordinary — a model that does not separate"]))
    assert "not estimable" not in ordinary["caption"]
    assert "slope" in ordinary["caption"]


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
