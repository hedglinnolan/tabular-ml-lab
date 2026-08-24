"""`AUDIT-019` — the seal's parameter count is over the frame the models get.

The seal's methods sentence stated a candidate-parameter count taken over the
whole table: **344 on `survey_instrument.csv` where the models are handed 45**,
and 299 of the 344 were `respondent_id`, a column
`training.feature_frame` structurally refuses to encode. `GUIDED-108` added
identifier exclusion to the frame the model is fed and nothing brought it to
the count, so the number describing the fit described the spreadsheet instead.

It matters where it lands rather than only where it is computed: this sentence
is what `PRODUCT_VISION.md` §04 puts in the manuscript's methods section, and
`§A5.4`'s whole point is that the candidate-parameter count is the input to
Riley's minimum sample size. A count inflated sevenfold by a column nobody
fits is a number about a fit nobody performed.

**Driven, not described.** Every assertion here runs a real project through
`AnalysisProject.seal_lockbox` and reads `project.lockbox`, and the manuscript
assertion folds the real `draft.draft` over the real transcript. Nothing here
inspects source text.

`GUIDED-097` — THE FIXTURE RULE. Three target shapes, and the shapes not
covered are named below.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from turbotab import draft as draft_mod
from turbotab import identifiers as ID
from turbotab import resolution as R
from turbotab import training as T
from turbotab.project import AnalysisProject

#: `GUIDED-097`. Three target shapes, three real fixtures, each carrying a real
#: per-row identifier — and the identifier's parameter cost is stated here so a
#: fixture that stopped carrying one would fail loudly rather than make the
#: absence assertions trivially true.
TARGET_SHAPES = {
    "binary classification": ("leaky_sepsis.csv", "sepsis", "classification",
                              "admission_id", 159),
    "continuous regression": ("survey_instrument.csv", "age", "regression",
                              "respondent_id", 299),
    "multiclass classification": ("multiclass_stage.csv", "disease_stage",
                                  "classification", "record_id", 239),
}

#: NOT COVERED, said out loud rather than left to be discovered.
#:
#: GROUPED GRAIN — every project here answers `not_sure`, so `group_col` is
#: `None` and the count drops two sets rather than three. The group column was
#: already dropped before this row and is not what `AUDIT-019` is about, but
#: the interaction (a group column that is ALSO unique per row) is undriven.
#:
#: A COHORT FILTER ACTIVE AT THE SEAL — `resolution.statement` counts levels
#: over `project.df` and `training.feature_frame` reads `project.working_table`,
#: which a cohort filter narrows. The two frames have the same COLUMNS, so the
#: exclusion set is the same; a categorical whose rarest level sits only in the
#: filtered-out rows could still make the two totals differ by one. Undriven,
#: and named because it is the one way the equality below could be false.
#:
#: A STRING OUTCOME — `GUIDED-097`'s own defect shape. The exclusion does not
#: read the target at all, so the behavior is expected to hold; it is not
#: driven here.
SHAPES_NOT_COVERED = [
    "grouped grain — every project here answers 'not_sure', so the "
    "group-column interaction with a unique-per-row column is undriven",
    "a cohort filter active at the seal — the record counts levels over "
    "project.df and the fed frame is project.working_table",
    "a string outcome — the exclusion is target-blind, but undriven",
]


def _sealed(name, target, task, fraction=0.25):
    """A real project, sealed the way the Guided door seals one."""
    df = pd.read_csv(f"turbotab/sample_data/{name}")
    df = df[df[target].notna()].copy()
    p = AnalysisProject.from_dataframe(df, name)
    p.target, p.task_type = target, task
    p.set_grain("not_sure")
    p.set_eligibility("everyone")
    rng = np.random.default_rng(42)
    idx = list(p.df.index)
    rng.shuffle(idx)
    labels = idx[:int(round(len(idx) * fraction))]
    p.seal_lockbox(labels, fraction=len(labels) / len(p.df))
    return p


def _parameters_of(frame: pd.DataFrame) -> int:
    """The parameter count of a frame that is already all-features.

    A sentinel target name that is not a column, so `candidate_parameters`
    drops nothing and counts every column it is given.
    """
    assert "__not_a_column__" not in frame.columns
    return R.candidate_parameters(frame, "__not_a_column__")["total"]


# ═══════════ THE LOAD-BEARING EQUALITY ═══════════

@pytest.mark.parametrize("shape", sorted(TARGET_SHAPES))
def test_the_recorded_count_equals_the_parameters_of_the_frame_the_models_get(shape):
    """`AUDIT-019`'s own failing assertion, as the regression test.

    The seal records a number and `training.feature_frame` decides what a
    model is handed. Before this fix they disagreed by the whole cost of the
    identifier columns; the count that is right is the one the frame supports,
    because `§A5.4`'s quantity is *the parameters the model may spend*.
    """
    name, target, task, column, cost = TARGET_SHAPES[shape]
    p = _sealed(name, target, task)

    recorded = p.lockbox["resolution"]["parameters"]["total"]
    fed = _parameters_of(T.feature_frame(p))
    assert recorded == fed, (
        f"{shape}: the seal recorded {recorded:,} candidate predictor "
        f"parameters and the models are handed a frame worth {fed:,}. "
        f"The recorded number is what the methods section reports and what "
        f"a sample-size calculation would be sized against.")


@pytest.mark.parametrize("shape", sorted(TARGET_SHAPES))
def test_the_identifier_is_in_the_table_and_out_of_the_recorded_count(shape):
    """THE POSITIVE CONTROL, and it is why the equality above means anything.

    An absence assertion over a shrinking tree gets easier to satisfy as the
    tree empties (`GUIDED-045`). So this checks the identifier is really in the
    fixture, really costs what the row said it costs, and really is what the
    two counts differ by — before asserting it is gone from the record.
    """
    name, target, task, column, cost = TARGET_SHAPES[shape]
    p = _sealed(name, target, task)

    assert column in p.df.columns, "the fixture no longer carries the identifier"
    assert column in ID.excluded(p), (
        f"{column} is not being excluded, so this test is asserting nothing")

    over_the_whole_table = R.candidate_parameters(p.df, target)
    per_column = {d["column"]: d["parameters"]
                  for d in over_the_whole_table["largest"]}
    assert per_column.get(column) == cost, (
        f"{column} costs {per_column.get(column)} parameters, not {cost}; "
        f"the fixture changed and the numbers in this file are stale")

    recorded = p.lockbox["resolution"]["parameters"]
    assert over_the_whole_table["total"] - recorded["total"] == cost, (
        f"{shape}: excluding {column} should move the count by exactly its "
        f"own {cost:,} parameters")


# ═══════════ THE SHELF IS NEVER SHORTENED ═══════════

@pytest.mark.parametrize("shape", sorted(TARGET_SHAPES))
def test_what_left_the_count_is_named_and_its_cost_is_stated(shape):
    """`PRODUCT_VISION.md`, *the shelf is never shortened*.

    A smaller number with no account of what left it is a number a reader
    cannot reconcile against their own column list — which is the same defect
    one direction over. The record names the columns and states what they
    would have added.
    """
    name, target, task, column, cost = TARGET_SHAPES[shape]
    p = _sealed(name, target, task)

    params = p.lockbox["resolution"]["parameters"]
    assert column in params["excluded_columns"], (
        f"{shape}: {column} left the count without being named")
    assert params["excluded_parameters"] >= cost, (
        f"{shape}: the record says {params['excluded_parameters']:,} "
        f"parameters were set aside; {column} alone is {cost:,}")

    sentence = p.lockbox["resolution"]["sentence"]
    assert column in sentence, (
        f"{shape}: the methods sentence does not say which column is not in "
        f"its count")
    assert f"{cost:,}" in sentence, (
        f"{shape}: the methods sentence does not say what the excluded "
        f"column would have added")


def test_a_project_with_no_identifier_gets_no_exclusion_clause():
    """The other branch, and it is not decoration.

    A sentence that named an exclusion on every study would make the clause
    wallpaper and the number unreadable. With nothing set aside the record
    carries an empty list and the sentence says nothing about exclusions.
    """
    df = pd.DataFrame({
        "y": [0, 1] * 30,
        # NOT unique per row — a column of 60 different ages would itself be
        # an identifier by the arithmetic rule, which is the trap this fixture
        # would fall into and did.
        "age": [40 + (i % 12) for i in range(60)],
        "site": ["a", "b", "c"] * 20,
    })
    p = AnalysisProject.from_dataframe(df, "no_identifier.csv")
    p.target, p.task_type = "y", "classification"
    p.set_grain("not_sure")
    p.set_eligibility("everyone")
    p.seal_lockbox(list(p.df.index)[:15], fraction=0.25)

    params = p.lockbox["resolution"]["parameters"]
    assert params["excluded_columns"] == []
    assert params["excluded_parameters"] == 0
    assert params["total"] == 3, "age is 1 and a 3-level site is 2"
    assert "excluding" not in p.lockbox["resolution"]["sentence"]


# ═══════════ IT REACHES THE MANUSCRIPT ═══════════

@pytest.mark.parametrize("shape", sorted(TARGET_SHAPES))
def test_the_methods_section_carries_the_corrected_count(shape):
    """Where the number does its damage.

    `PRODUCT_VISION.md` §04 puts this sentence in the methods section and
    `draft.py` folds it there from the seal decision's own payload. A count
    corrected on the lockbox and stale in the draft would be the fix landing
    everywhere except where a reviewer reads it.
    """
    name, target, task, column, cost = TARGET_SHAPES[shape]
    p = _sealed(name, target, task)

    recorded = p.lockbox["resolution"]["parameters"]["total"]
    whole_table = R.candidate_parameters(p.df, target)["total"]
    assert whole_table != recorded, "this fixture cannot tell the two apart"

    # `seal_lockbox` folds into the `target` section — "Outcome and analysis
    # population" — which is where `_KIND_SECTION` puts it.
    sections = {s["key"]: s for s in draft_mod.draft(p.to_dict())["sections"]}
    methods = " ".join(str(s["text"]) for s in sections["target"]["sentences"])

    assert f"{recorded:,} candidate predictor parameters" in methods, (
        f"{shape}: the methods section does not report the {recorded:,} "
        f"parameters the models are handed")
    assert f"{whole_table:,} candidate predictor parameters" not in methods, (
        f"{shape}: the methods section still reports the whole table's "
        f"{whole_table:,} parameters")


def test_the_survey_fixture_reports_the_number_the_row_names():
    """`AUDIT-019` names two numbers — 344 and 45 — and this is the one
    assertion that pins them, so a future refactor cannot satisfy the equality
    above by moving both sides together."""
    p = _sealed("survey_instrument.csv", "sought_support", "classification")

    assert R.candidate_parameters(p.df, "sought_support")["total"] == 344
    assert p.lockbox["resolution"]["parameters"]["total"] == 45
    assert p.lockbox["resolution"]["parameters"]["excluded_parameters"] == 299
