"""`MISC-028` and `MISC-031` — one check, two branches, opposite failures.

*Split counts reconcile to analysis population* sums `train_n`, `val_n` and
`test_n` and compares the result to `analysis_total`. Until `L65` the Guided
producer made **both sides itself**, in two different and equally useless ways:

- **the run branch** defined `analysis_total = train_n + test_n` with `val_n`
  pinned to the literal `0`, so the check added up the terms of its own
  comparand. Driven over 406 `(n_train, n_test)` pairs including `(0, 0)`,
  `(1, 0)` and `(999999, 1)`: zero violations, and the only `val_n` ever
  observed was `0`. **It could not FAIL.**
- **the lockbox branch** wrote only `analysis_total` and `test_n`, so
  `int(population.get(key) or 0)` made the split sum `test_n` alone. Driven
  over 4,000 randomized bundles the verdict set was `{'FAIL'}` and nothing
  else. **It could not PASS** — an unfitted project was shown a validation
  failure that describes no manuscript defect and that no edit its author could
  make would ever clear.

Same check, opposite failure, two branches, which is why a repair aimed at one
of them is not a repair.

## And the lockbox branch meant a different population under the same key

`analysis_total` was `lockbox["n_total"]` — `len(df)`, the whole uploaded table
— while the run branch's was the rows with an outcome. So on
`metabolomics_untargeted.csv`, where 8 of 80 rows have no `responder`, the
abstract read *"A dataset of **80** observations was analyzed, of which **16**
were held out for evaluation"* when 72 rows have an outcome and 12 of the
held-out ones do. Two wrong numbers in one sentence, in the artifact that leaves
the building. Both are asserted below.

## The comparand needed no plumbing

`lockbox["resolution"]` is written at the seal by `turbotab/resolution.py` from
three separate reductions over the frame, and its `n` is exactly
`len(project.outcome_rows)` — which two docstrings already assert is what
`analysis_total` means (`project.py::outcome_rows`, and this package's Table 1
builder) while nothing checked it. `analysis_total` comes from the seal now and
the split comes from the run, so the check spans two derivations by different
code at different moments, and a post-seal row drop separates them.

## Where there is only one derivation, it says so

On a project with no run nothing has partitioned anything except the seal, so
both sides come from `resolution` and the sum restates the total. That is a fact
about the app rather than a failure of the repair, and it is declared through
`MISC-029`'s mechanism rather than hidden — which is the **seam this part found
in that one**: `MISC-029` declares where an *input* is absent, and this needed
the same treatment where the *comparand* is. One criterion, two faces.

## `GUIDED-097` — the fixture rule, and trap #3

Both target shapes, and both are made to have outcome-blank rows, because a
fixture where `outcome_rows == df` supplies what production cannot and every
assertion here would pass over one population wearing two names.
`metabolomics_untargeted.csv` has 8 real blanks; `survey_instrument.csv`'s `age`
is complete, so blanks are injected and the separation is **asserted in the
fixture** rather than assumed.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.manuscript_validator import validate_manuscript_bundle
from turbotab import engine
from turbotab import eventfixture
from turbotab import manuscript as MS
from turbotab import training as T
from turbotab.project import AnalysisProject

FIXTURES = Path(__file__).resolve().parent / "sample_data"

CHECK = "Split counts reconcile to analysis population"

#: `(fixture, target, task, model, outcomes to blank)`. The blank count is 0
#: where the file already has them and the fixture asserts the separation
#: either way.
TARGET_SHAPES = {
    "binary_classification": ("metabolomics_untargeted.csv", "responder",
                              "classification", "logreg", 0),
    "continuous_regression": ("survey_instrument.csv", "age", "regression",
                              "ridge", 12),
}

#: NOT COVERED, said out loud.
SHAPES_NOT_COVERED = [
    "multiclass classification — nothing here drives a third level, and the "
    "reconciliation does not read the target's levels, so this is a gap in "
    "coverage rather than a known difference",
    "survival / time-to-event — no task type exists in this app at all, so "
    "there is no split for a survival study to reconcile",
    "a project whose seal could not describe itself "
    "(`lockbox['resolution_unavailable']`). `_counts` drops the population "
    "block entirely there rather than falling back to `len(df)`, and no "
    "fixture here reaches that branch because `resolution.statement` does not "
    "raise on these files",
    "the CLASSIC producer end to end. `pages/10_Report_Export.py` is a "
    "Streamlit page and cannot be driven from pytest; its context is asserted "
    "structurally in "
    "`test_the_classic_producer_declares_its_own_tautology` instead",
]


def _project(shape, *, fit=True, drop_after_seal=0):
    fixture, target, task, model, blanks = TARGET_SHAPES[shape]
    frame = pd.read_csv(FIXTURES / fixture)
    if blanks:
        frame = frame.copy()
        frame.loc[frame.sample(blanks, random_state=11).index, target] = np.nan
    if drop_after_seal:
        # Rows that carry an outcome at the seal and are removed afterwards, in
        # the shape `drop_empty_rows` removes: every column blank but the
        # outcome. `project.check_repair_allowed` permits `drop_empty_rows` and
        # `drop_rows` AFTER the barrier — they only remove rows and the
        # survivors keep their labels — so this is a state a user reaches.
        extra = frame.sample(drop_after_seal, random_state=3).copy()
        for column in frame.columns:
            if column != target:
                extra[column] = np.nan
        frame = pd.concat([frame, extra], ignore_index=True)

    project = AnalysisProject.from_dataframe(frame, fixture)
    project.target, project.task_type = target, task
    project.set_grain("not_sure")
    project.set_eligibility("everyone")
    rng = np.random.default_rng(42)
    idx = list(project.df.index)
    rng.shuffle(idx)
    labels = idx[:int(round(len(idx) * 0.20))]
    project.seal_lockbox(labels, fraction=len(labels) / len(project.df))
    eventfixture.choose_event(project, required=(task == "classification"))

    # TRAP #3, ASSERTED RATHER THAN AVOIDED. A shape where every row has an
    # outcome makes `outcome_rows` and the whole table one population, and
    # every claim below would hold over a fixture production cannot produce.
    assert len(project.outcome_rows) < len(project.df), (
        f"{shape} has no outcome-blank rows, so this fixture cannot tell the "
        f"analysis cohort from the uploaded table")

    if drop_after_seal:
        blank_rows = project.df.drop(columns=[target]).isna().all(axis=1)
        assert int(blank_rows.sum()) >= drop_after_seal, blank_rows.sum()
        project.df = project.df.loc[~blank_rows]

    run = T.train(project, [model]).to_dict() if fit else None
    return project, run


def _row(out):
    return next(r for r in out["rows"] if r["Check"] == CHECK)


# ═══════════ 1 · THE POPULATION MEANS ONE THING ON BOTH BRANCHES ═══════════

@pytest.mark.parametrize("shape", sorted(TARGET_SHAPES))
@pytest.mark.parametrize("fit", [True, False], ids=["run", "no_run"])
def test_the_analysis_total_is_the_cohort_the_study_describes(shape, fit):
    """The two docstrings that asserted this are made true, on both branches.

    `project.py::outcome_rows` says it is *"the one Table 1 and the
    manuscript's `analysis_total` mean"*, and the Table 1 builder says the same.
    Neither was checked, and on the lockbox branch neither was true.
    """
    project, run = _project(shape, fit=fit)
    counts = MS.structure(project.to_dict(),
                          run=run)["context"]["population_counts"]

    assert counts["analysis_total"] == len(project.outcome_rows), counts
    assert counts["analysis_total"] == project.lockbox["resolution"]["n"]
    assert counts["analysis_total"] < len(project.df), (
        "the analysis total is the whole uploaded table, which is the "
        "population the lockbox branch used to report")
    # All four keys on both branches. Two of them were missing on the lockbox
    # branch, and a missing key sums as zero.
    for key in ("train_n", "val_n", "test_n"):
        assert key in counts, (key, counts)
    assert counts["train_n"] + counts["val_n"] + counts["test_n"] == \
        counts["analysis_total"], counts


def test_the_abstract_states_the_population_it_analyzed():
    """`MISC-028`'s user-visible half, in the artifact that leaves the building.

    Held to the classification shape because its blanks are real rather than
    injected: this is the sentence a person reads, and it should be checked
    against a file as it ships.
    """
    project, _ = _project("binary_classification", fit=False)
    doc = MS.structure(project.to_dict(), run=None)
    report = MS.to_markdown(doc)["report"]

    total = len(project.outcome_rows)
    held = project.lockbox["resolution"]["n_test"]
    assert total < len(project.df) and held < project.lockbox["n_test"], (
        "this fixture cannot distinguish the two populations")
    assert (f"A dataset of {total:,} observations was analyzed, of which "
            f"{held:,} were held out") in report, report
    assert f"dataset of {len(project.df):,} observations" not in report, (
        f"the abstract still states the uploaded row count as the analysis "
        f"population:\n{report}")


# ═══════════ 2 · THE CHECK CAN NOW BE WRONG, AND CAN NOW BE RIGHT ══════════

@pytest.mark.parametrize("shape", sorted(TARGET_SHAPES))
def test_the_reconciliation_fails_when_the_fit_and_the_seal_disagree(shape):
    """**The falsifiability proof, through a path a user reaches.**

    `check_repair_allowed` permits `drop_empty_rows` and `drop_rows` after the
    barrier. Rows removed then are gone from the fit and still counted in the
    seal's frozen statement, so the manuscript would claim a cohort larger than
    the one the model saw — which is exactly the inconsistency this check is
    named for and could not previously report.
    """
    project, run = _project(shape, fit=True, drop_after_seal=40)
    out = MS.validate(project.to_dict(), run=run)
    row = _row(out)

    assert row["scored"] is True, row
    assert row["Status"] == "FAIL", row["Detail"]
    assert str(project.lockbox["resolution"]["n"]) in row["Detail"], row
    assert str(run["n_train"] + run["n_test"]) in row["Detail"], row
    assert "two derivations" in row["Detail"], row["Detail"]


@pytest.mark.parametrize("shape", sorted(TARGET_SHAPES))
def test_the_reconciliation_passes_when_they_agree(shape):
    """The control beside it. A check that only ever fails is the other half
    of the defect."""
    project, run = _project(shape, fit=True)
    row = _row(MS.validate(project.to_dict(), run=run))
    assert row["scored"] is True and row["Status"] == "PASS", row


@pytest.mark.parametrize("shape", sorted(TARGET_SHAPES))
def test_the_lockbox_branch_is_no_longer_pinned_at_failure(shape):
    """`MISC-031`, and it is measured the way it was found.

    Not one project — 400 randomized bundles against a fixed unfitted context,
    because the row's own evidence is a verdict SET of `{'FAIL'}` over 4,000 of
    them. A single PASS would not distinguish a repair from a lucky draw.
    """
    project, _ = _project(shape, fit=False)
    doc = MS.structure(project.to_dict(), run=None)
    context = doc["context"]
    rendered = MS.to_markdown(doc)
    rng = random.Random(31)
    corpus = ["", rendered["methods"], rendered["report"],
              "## Abstract (Draft)\nA dataset of 1 observations.\n",
              "### Predictor Variables\nWe retained 4 predictors for final "
              "modeling.\n", "[PLACEHOLDER] **bold**"]

    verdicts = set()
    for _ in range(400):
        def blob():
            return "\n".join(rng.sample(corpus, rng.randint(0, len(corpus))))
        report = validate_manuscript_bundle(
            context, blob(), blob(), blob(), TARGET_SHAPES[shape][2])
        verdicts.add(next(c.status for c in report.checks
                          if c.name == CHECK))
    assert verdicts == {"PASS"}, (
        f"the lockbox branch reports {verdicts} for {CHECK!r}. `FAIL` there is "
        f"MISC-031 — a defect that describes the producer rather than the "
        f"draft, and that no author can clear.")


@pytest.mark.parametrize("shape", sorted(TARGET_SHAPES))
def test_it_declares_itself_where_only_one_derivation_exists(shape):
    """The seam this part found in `MISC-029`'s mechanism.

    On an unfitted project both sides come from the seal, so the sum restates
    the total. That is not repairable — nothing else in the project has counted
    those rows — so it is declared, by the same criterion the other three use:
    there is no second thing to compare against.
    """
    project, _ = _project(shape, fit=False)
    out = MS.validate(project.to_dict(), run=None)
    row = _row(out)

    assert row["scored"] is False, row
    assert row["Status"] == "PASS", row
    assert "sealed_cohort" in row["declared_because"], row["declared_because"]
    assert "restates the total" in row["declared_because"]
    counts = out["checklist_counts"]
    assert CHECK in {d["Check"] for d in counts["declared"]}, counts
    assert counts["n_declared_that_failed"] == 0, counts

    # And it is SCORED with a run, on the same project shape — the declaration
    # is a property of the context rather than of the check.
    fitted, run = _project(shape, fit=True)
    assert _row(MS.validate(fitted.to_dict(), run=run))["scored"] is True


def test_an_unsealed_project_is_not_told_its_split_does_not_reconcile():
    """**`MISC-031`'s class one state earlier, found by driving the page.**

    The row names the lockbox branch. A project that has not been SEALED
    reaches neither branch of `_counts`, so `population_counts` is `{}` and the
    check compared `None` against `0` — `FAIL`, permanently, on a project whose
    author has simply not reached the seal yet. Driven through the API rather
    than reasoned about, because the state is what a person sees after
    answering the lens and the target and nothing else.
    """
    from fastapi.testclient import TestClient

    from turbotab import api

    client = TestClient(api.app)
    with (FIXTURES / "clinical_risk.csv").open("rb") as handle:
        pid = client.post("/project", files={
            "file": ("clinical_risk.csv", handle, "text/csv")}).json()["id"]
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_lens", "payload": {"lens": ["clinical"]}})
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_target",
                      "payload": {"column": "readmit_30d"}})
    body = client.get(f"/project/{pid}/manuscript").json()

    assert body["document"]["context"]["population_counts"] == {}, (
        "this project now has population counts, so it is not the unsealed "
        "state this test is about")
    row = _row(body)
    assert row["Status"] == "PASS", row["Detail"]
    assert row["scored"] is False, row
    assert "has not been sealed" in row["declared_because"], row
    assert body["checklist_counts"]["n_declared_that_failed"] == 0


# ═══════════ 3 · THE THIRD PRODUCER, WHICH IS ON THE OTHER DOOR ═══════════

def test_the_classic_producer_declares_its_own_tautology():
    """**One consumer, three producers, and the third is one door over.**

    `pages/10_Report_Export.py` sets `analysis_total` to the literal sum of the
    three terms this check adds up, so it is an identity there and always was.
    There is no second count of that cohort anywhere in a Classic session to
    reconcile against, and inventing one would be a number the validator then
    confirms against the arithmetic it came from — so it is annotated rather
    than repaired, and the check declares itself instead of reporting scrutiny
    it did not apply.
    """
    page = (Path(__file__).resolve().parents[1] / "pages"
            / "10_Report_Export.py").read_text(encoding="utf-8")
    assert "'analysis_total': train_n + val_n + test_n," in page, (
        "the Classic producer no longer derives the total from the split, so "
        "the annotation below may be describing something that stopped being "
        "true")
    assert "'analysis_total_source': 'split'," in page
    assert "'split_source': 'split'," in page

    classic_like = {
        "population_counts": {"upload_total": 500, "analysis_total": 100,
                              "train_n": 60, "val_n": 20, "test_n": 20,
                              "analysis_total_source": "split",
                              "split_source": "split"},
        "feature_counts": {"original": 12, "selected": 8},
        "feature_names_for_manuscript": ["age"],
        "manuscript_primary_model": "rf", "best_metric_name": "auc",
        "included_models": ["rf"],
    }
    report = validate_manuscript_bundle(classic_like, "", "", "",
                                        "classification")
    declared = [c.name for c in report.declared_checks]
    assert declared == [CHECK], declared
    # Only the reconciliation. Every other Classic check stays SCORED, because
    # `_build_manuscript_context` writes the keys whose absence declares them
    # on the Guided path — the empty bundle above makes several of them FAIL,
    # which is the point: they are live enough to fail.
    assert next(c for c in report.checks if c.name == CHECK).status == "PASS"
    assert len(report.scored_checks) == len(report.checks) - 1 == 12


def test_a_producer_that_names_no_source_is_scored_rather_than_excused():
    """The default direction, chosen deliberately.

    A context with neither key is treated as two derivations, so an unknown
    producer gets its check SCORED. The other direction would let any caller
    silence a check by omitting a field, which is a gate switched off rather
    than satisfied.
    """
    unknown = {"population_counts": {"analysis_total": 100, "train_n": 60,
                                     "val_n": 20, "test_n": 20}}
    report = validate_manuscript_bundle(unknown, "", "", "", "classification")
    row = next(c for c in report.checks if c.name == CHECK)
    assert row.scored is True and row.declared_because == ""
    assert row.status == "PASS"

    wrong = {"population_counts": {"analysis_total": 999, "train_n": 60,
                                   "val_n": 20, "test_n": 20}}
    row = next(c for c in validate_manuscript_bundle(
        wrong, "", "", "", "classification").checks if c.name == CHECK)
    assert row.status == "FAIL" and row.scored is True


def test_the_shapes_not_covered_are_named():
    """A fixture list that does not say what it omits reads as coverage."""
    assert len(SHAPES_NOT_COVERED) >= 4
    assert all(len(s) > 60 for s in SHAPES_NOT_COVERED)


def test_engine_still_permits_the_repair_this_file_drives():
    """The premise `drop_after_seal` rests on, asserted rather than assumed.

    If `drop_empty_rows` ever became pre-barrier-only, the divergence above
    would stop being a state a user can reach and the falsifiability proof
    would be a fixture supplying what production cannot.
    """
    from turbotab.project import PRE_BARRIER_ONLY_FIXES

    assert "drop_empty_rows" not in PRE_BARRIER_ONLY_FIXES
    assert "drop_rows" not in PRE_BARRIER_ONLY_FIXES
    assert "promote_header" in PRE_BARRIER_ONLY_FIXES, (
        "the barrier no longer refuses anything, so this set is not the rule "
        "it is being read as")
    assert hasattr(engine, "record_fix")
