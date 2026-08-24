"""`GUIDED-158` · the binary card titles each column by what it actually holds.

## The measurement, on the product owner's own file

Nine hits on one card, the first card he met on his NHANES export, and **eight
of the nine titles were false about the data**:

| columns | what they are | the title they got |
|---|---|---|
| `imputed_bmi`, `imputed_bp_di`, `imputed_bp_sys`, `imputed_height`, `imputed_waist`, `imputed_weight` | dtype **`bool`** | *"… is a binary variable written as text"* |
| `meds_chol`, `meds_hbp` | dtype `object` holding Python `True`/`False` | *"… is a binary variable written as text"* |
| `GENDER` | dtype `object` holding `{female, male}` | *"… is a binary variable written as text"* |

Nothing downstream was wrong. The repair is correct for all nine and the frame
it produces is what he wanted, so the cost is **trust**: eight assertions he
could disprove in one glance at his own CSV, on the first card, from a tool
whose whole apparatus of badges and sources is spent buying belief about the
things he cannot check.

## The reproduction, and why a fixture had to be added

**Every one of the sixteen shipped CSVs holds only the one shape the sentence
was true of.** Swept: eleven of them raise a `read_as_binary` finding, fifteen
findings between them, and all fifteen are two-level strings. No shipped
fixture had a `bool` column or an `object` column of Python booleans — which is
exactly why the defect survived, and it is the gap `GUIDED-097`'s own ledger
note already named as *cheap to add and not in `sample_data`*.

So `turbotab/sample_data/binary_shapes.csv` was added: 24 rows carrying all
three shapes at once, in the same proportions and under the same column names
as the file he drove. Read by pandas it gives `imputed_bmi` and
`imputed_waist` as dtype `bool`, `meds_chol` and `meds_hbp` as `object` holding
Python booleans — object because their blanks forbid a bool column — and
`gender` as two strings.

## What changed

`ml/binary_text.value_shape` reads the shape off the series and
`ml/binary_text.written_as` supplies the sentence for it. The repair is
untouched: same `fix_kind`, same mapping, same 0/1 frame. **The claim changed
and the behavior did not**, which is the whole of this row.

## Fixture shapes — `GUIDED-097`

`WRITTEN_SHAPES` runs the load-bearing claim against all four writings the
composer distinguishes. `SHAPES_NOT_COVERED` names the rest.
"""
from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml import binary_text as B                    # noqa: E402
from turbotab import api                           # noqa: E402

DATA = Path(__file__).resolve().parent / "sample_data"
FIXTURE = "binary_shapes.csv"

#: The four writings, and the column of `binary_shapes.csv` that carries each.
#: `mixed` has no column because a CSV cannot produce one — see below.
WRITTEN_SHAPES = {
    B.BOOL_DTYPE: "imputed_bmi",
    B.OBJECT_BOOLS: "meds_chol",
    B.OBJECT_TEXT: "gender",
}

SHAPES_NOT_COVERED = {
    B.OBJECT_NUMBERS: (
        "A column of 1 and 0 read from a CSV comes back as `int64`, which "
        "`_is_texty` excludes, so no uploaded file can reach this branch. It "
        "is driven as a constructed frame below and NOT through the API."),
    B.MIXED_WRITING: (
        "Same reason and one step further: a column holding `True` and the "
        "string `'false'` cannot survive a CSV round trip as two types. "
        "Driven as a constructed frame only."),
    "nullable_boolean": (
        "pandas' `boolean` extension dtype answers `is_bool_dtype` and would "
        "land in `bool_dtype`, which is the true answer for it. Nothing in "
        "this app constructs one, so it is asserted at unit level and not "
        "driven."),
}

#: The sentence that used to sit on every hit.
OLD = "is a binary variable written as text"


@pytest.fixture(scope="module")
def client():
    return TestClient(api.app)


def _frame():
    return pd.read_csv(DATA / FIXTURE)


# ── 0 · the fixture is the reproduction, not a convenient shape ──────────────

def test_the_fixture_really_carries_all_three_shapes():
    """A fixture that manufactured the shape the assertion needs would be trap
    #3. This reads what pandas actually produced from the committed CSV."""
    df = _frame()
    assert B.value_shape(df["imputed_bmi"]) == B.BOOL_DTYPE
    assert df["imputed_bmi"].dtype == bool                            # control
    assert B.value_shape(df["meds_chol"]) == B.OBJECT_BOOLS
    assert df["meds_chol"].dtype == object
    assert all(isinstance(v, (bool, np.bool_))
               for v in df["meds_chol"].dropna())
    assert int(df["meds_chol"].isna().sum()) > 0, (
        "the object-bool column has no blanks, which is the only reason "
        "pandas leaves such a column untyped")
    assert B.value_shape(df["gender"]) == B.OBJECT_TEXT
    assert set(df["gender"].dropna()) == {"male", "female"}


# ── 1 · the finding, on every shape ──────────────────────────────────────────

@pytest.mark.parametrize("shape", sorted(WRITTEN_SHAPES), ids=sorted(WRITTEN_SHAPES))
def test_a_column_is_not_called_text_unless_it_holds_text(shape):
    """**The finding.** Only the one shape the sentence describes may carry it."""
    df = _frame()
    column = WRITTEN_SHAPES[shape]
    finding = B.binary_text_finding(column, df[column])
    assert finding is not None, (column, "no finding at all")         # control
    assert finding.fix_kind == "read_as_binary"

    if shape == B.OBJECT_TEXT:
        assert OLD in finding.title, finding.title
    else:
        assert OLD not in finding.title, (
            f"'{column}' is {shape} and the card calls it text: "
            f"{finding.title!r}")
        # The detail may DENY that it is text — `meds_chol`'s does — but it
        # may not assert it. Checked against the sentence the composer says it
        # supplies for this shape, so the two cannot drift.
        assert finding.detail.endswith(
            B.written_as(column, shape, int(df[column].isna().sum()))["detail"]
        ), finding.detail
        assert "not a number stored as text" not in finding.detail

    # The shape rides in the payload beside the sentence, so anything reading
    # the record rather than the prose gets the same answer.
    assert finding.params["written_as"] == shape


def test_the_four_writings_get_four_different_sentences():
    """A correction that gave every shape the same softer sentence would pass
    the assertion above and say nothing. Each writing gets its own."""
    titles = {shape: B.written_as("col", shape)["title"]
              for shape in (B.BOOL_DTYPE, B.OBJECT_BOOLS, B.OBJECT_TEXT,
                            B.OBJECT_NUMBERS, B.MIXED_WRITING)}
    assert len(set(titles.values())) == 5, titles
    assert OLD in titles[B.OBJECT_TEXT]
    assert "already" in titles[B.BOOL_DTYPE], titles[B.BOOL_DTYPE]
    assert "true/false" in titles[B.OBJECT_BOOLS], titles[B.OBJECT_BOOLS]
    # THE MIXED CASE SAYS LESS RATHER THAN SAYING SOMETHING WRONG. It is the
    # same instinct as returning no number: where the writing is not one
    # writing, the sentence claims only what is certain.
    assert "more than one way" in titles[B.MIXED_WRITING]


def test_the_shapes_a_csv_cannot_carry_are_still_read_correctly():
    """`OBJECT_NUMBERS` and `MIXED_WRITING` are unreachable through an upload,
    so they are constructed here rather than left unasserted."""
    numbers = pd.Series([1, 0, 1, 0, 1, 1, 0], dtype=object)
    assert B.value_shape(numbers) == B.OBJECT_NUMBERS
    f = B.binary_text_finding("consent", numbers)
    assert f is not None
    assert OLD not in f.title, f.title
    assert "two numbers" in f.title

    mixed = pd.Series([True, "false", True, "false", True, False], dtype=object)
    assert B.value_shape(mixed) == B.MIXED_WRITING
    g = B.binary_text_finding("flag", mixed)
    assert g is not None
    assert OLD not in g.title, g.title

    # And the nullable boolean extension dtype, which is a bool column.
    nullable = pd.Series([True, False, None, True, False, True],
                         dtype="boolean")
    assert B.value_shape(nullable) == B.BOOL_DTYPE


# ── 2 · the claim changed and the repair did not ─────────────────────────────

@pytest.mark.parametrize("shape", sorted(WRITTEN_SHAPES), ids=sorted(WRITTEN_SHAPES))
def test_the_repair_is_unchanged_for_every_shape(shape):
    """`GUIDED-158`'s own words: *the repair is the same for all three and the
    CLAIM is not*. If the repair had moved, this row would be a behavior change
    wearing a copy fix.
    """
    df = _frame()
    column = WRITTEN_SHAPES[shape]
    finding = B.binary_text_finding(column, df[column])
    out = B.apply_read_as_binary(df, finding)
    frame = out[0] if isinstance(out, tuple) else out
    values = set(frame[column].dropna().unique().tolist())
    assert values <= {0, 1}, (column, values)
    assert len(values) == 2, (column, values)
    assert int(frame[column].isna().sum()) == int(df[column].isna().sum()), (
        "the repair changed how many blanks the column has")


# ── 3 · the card the user actually opens ─────────────────────────────────────

def test_the_repair_group_card_titles_every_member_by_what_it_holds(client):
    """The surface the finding was raised against: one card, five members, each
    with its own title. Driven through the real API rather than composed here.
    """
    with open(DATA / FIXTURE, "rb") as fh:
        body = client.post("/project", files={
            "file": (FIXTURE, fh, "text/csv")}).json()
    pid = body["id"]
    hits = [f for f in body["findings"] if f["fix_kind"] == "read_as_binary"]
    assert len(hits) == 5, [f["title"] for f in hits]                 # control

    df = _frame()
    text_columns = {c for c in df.columns
                    if B.value_shape(df[c]) == B.OBJECT_TEXT}
    called_text = {f["affected_columns"][0] for f in hits if OLD in f["title"]}
    assert called_text == (text_columns & {f["affected_columns"][0]
                                           for f in hits}), (
        f"the card calls these columns text: {sorted(called_text)}; only "
        f"{sorted(text_columns)} hold text")
    assert len(called_text) == 1, sorted(called_text)
    assert len(hits) - len(called_text) == 4, (
        "the reproduction is meant to carry four hits the old sentence was "
        "wrong about")

    group = client.get(f"/project/{pid}/repair_group/read_as_binary").json()
    member_titles = {m["title"] for m in group["members"]}
    assert member_titles == {f["title"] for f in hits}, (
        "the group card and the finding list give different titles for the "
        "same columns")
    wrong = [t for t in member_titles
             if OLD in t and not any(c in t for c in text_columns)]
    assert not wrong, wrong


# ── 4 · the sweep, over every shipped fixture ────────────────────────────────

def test_no_shipped_fixture_produces_a_title_its_column_contradicts():
    """`AGENT_ONBOARD.md` §08's fifth check, applied to the whole sample set.

    Sixteen CSVs, every `read_as_binary` finding in all of them, each title
    checked against its own column's writing. It also records the count that
    made this row invisible for as long as it was: before `binary_shapes.csv`,
    every one of these was `object_text`.
    """
    seen: dict = {}
    checked = 0
    for path in sorted(glob.glob(str(DATA / "*.csv"))):
        df = pd.read_csv(path)
        for finding in B.detect_binary_text(df):
            column = finding.affected_columns[0]
            shape = B.value_shape(df[column])
            checked += 1
            seen.setdefault(shape, []).append(f"{os.path.basename(path)}:{column}")
            expected = B.written_as(column, shape,
                                    int(df[column].isna().sum()))["title"]
            assert finding.title == expected, (
                f"{os.path.basename(path)}:{column} is {shape} and is titled "
                f"{finding.title!r}")
    assert checked >= 15, checked                                     # control
    assert B.OBJECT_TEXT in seen and B.BOOL_DTYPE in seen and B.OBJECT_BOOLS in seen, (
        f"the sample set no longer carries all three writings: "
        f"{ {k: len(v) for k, v in seen.items()} }")
    # The number behind the docstring's claim, re-derived rather than quoted.
    assert len(seen[B.OBJECT_TEXT]) >= 15, (
        f"{len(seen[B.OBJECT_TEXT])} text hits across the sample set")
