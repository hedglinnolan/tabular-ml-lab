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
# The tests here that drove the retired legacy app (turbotab/api.py, the page, the
# figure and manuscript modules, the gates) were removed with it, BLUEPRINT §9.1; git keeps them.
from __future__ import annotations

from pathlib import Path


from ml.manuscript_validator import validate_manuscript_bundle
from turbotab import engine

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


def _row(out):
    return next(r for r in out["rows"] if r["Check"] == CHECK)


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
