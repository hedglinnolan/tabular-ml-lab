"""L51-A2 — `GUIDED-217`. The one input the research says must be refused.

`GENOMICS_PACK.md`'s scope paragraph is one sentence and it is quantified:

> **Single-cell is out of scope and must be detected and refused** — zero
> fraction >80–90% with >1,000 columns and median non-zero count ≤3.

L50 built the nine-signature reader §02 specifies and did not build this, and
**the omission was worse than neutral**. A single-cell count matrix is integral,
non-negative and reaches past ten thousand per cell, so it satisfied §02 row 1
exactly: the app classified it **raw counts at high confidence** and its
capability matrix then reported the negative-binomial route as *enabled*. The
loop that built the reader made the app confident about the one matrix the pack
had already declared it may not speak about.

## Where the refusal goes, and why the position is the fix

**In front of the cascade, not after it.** A refusal reached only once a
classification has been made is a classification the app has already made — and
`_classify` already establishes the pattern by putting §02's negatives rule at
the outermost branch rather than applying it as a filter afterwards. The scope
paragraph is not a caveat on the reading; it is a statement about which
matrices this reader may speak about at all.

## The one place a range became a number, and which end was taken

The research says *"zero fraction >80–90%"*. The app takes **80**, the lower
end, and the direction is the safe one: refusing at 80 refuses slightly more
than the research demands, and the two errors are not symmetric. A false
refusal costs a user being told to bring bulk data. A false acceptance costs a
count model fitted on a matrix whose variance structure does not support it,
running silently, with the app's confidence attached.

## What is NOT covered

- **A single-cell matrix that is already normalized** — CPM-per-cell, or
  log1p'd. Those are not integral and would not have classified as raw counts
  in the first place, so the defect this closes did not reach them; the refusal
  does not either, and that gap is stated rather than papered over.
- **The 1,000-column threshold against a wide bulk matrix.** The three criteria
  are conjunctive, and no shipped bulk fixture is sparse enough to test the
  interaction — asserted on constructed frames below instead.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

DATA = Path(__file__).resolve().parent / "sample_data"
FIXTURE = "genomics_single_cell.csv"


def _frame(name: str = FIXTURE) -> pd.DataFrame:
    return pd.read_csv(DATA / name)


def test_the_fixture_meets_all_three_of_the_scope_paragraphs_criteria():
    """The positive control, before anything is asserted about the refusal.

    A refusal fixture that does not actually meet the criteria would make every
    assertion below vacuous — and it would be trap #3, the fixture
    manufacturing the thing whose absence is the defect, with the sign flipped.
    """
    from turbotab import packs

    frame = _frame()
    reading = packs.read_matrix(frame)
    assert reading is not None, "the fixture is not wide enough to be read"
    assert reading["pct_zeros"] > 0.80, reading["pct_zeros"]
    assert reading["n_columns"] > 1000, reading["n_columns"]
    assert reading["median_nonzero"] <= 3.0, reading["median_nonzero"]
    assert reading["all_integral"], (
        "the fixture is not integral, so it would never have classified as raw "
        "counts and this test would be closing a defect that could not happen")


def test_it_is_refused_and_not_classified():
    """`GUIDED-217` itself, driven on the shipped fixture."""
    from turbotab import packs

    card = packs.data_type_card(_frame())
    assert card is not None, "no card at all — the matrix is wide enough"
    assert card["read"] is False, (
        f"the app classified a single-cell matrix as "
        f"{card.get('classification')!r}. The research says this input must be "
        f"refused, and a classification is the opposite of a refusal")
    assert card.get("out_of_scope") == packs.SINGLE_CELL, (
        f"the matrix was not read, and the app does not say WHY — "
        f"{card.get('reason', '')[:120]!r}. *Matched none of the nine* and "
        f"*out of scope* are different answers and only one of them is true")
    assert "single-cell" in card["reason"], card["reason"]
    assert card["criteria"] == pytest.approx(
        {k: packs.read_matrix(_frame())[k]
         for k in ("pct_zeros", "n_columns", "median_nonzero")}), (
        "the refusal does not carry the three numbers it rests on, so a reader "
        "cannot check the app's arithmetic against their own table")


def test_no_capability_row_is_offered_on_a_refused_matrix():
    """The consequence, which is the half that would have hurt someone.

    The classification is not the damage — the capability matrix is. `GUIDED-217`
    records that the count-model route came back ENABLED.
    """
    from turbotab import packs

    card = packs.data_type_card(_frame())
    assert "capabilities" not in card or not card["capabilities"], (
        f"a refused matrix still carries a capability matrix: "
        f"{str(card.get('capabilities'))[:200]}. That is the sentence that "
        f"opened a negative-binomial route on out-of-scope data")
    assert not card["evidence"]["rows"], (
        "the refusal carries signature evidence, which is the reading it "
        "declined to make")


def test_the_bulk_fixtures_are_not_refused():
    """The negative control, and it is the one that matters.

    A refusal that fires on ordinary bulk data has moved the defect rather than
    fixed it — every one of §02's eight shipped signatures must still classify.
    """
    from turbotab import packs

    bulk = ["genomics_expression.csv", "genomics_estimated_counts.csv",
            "genomics_cpm.csv", "genomics_tmm_cpm.csv", "genomics_fpkm.csv",
            "genomics_vst.csv", "genomics_microarray.csv"]
    for name in bulk:
        card = packs.data_type_card(_frame(name))
        assert card is not None, name
        assert card.get("out_of_scope") is None, (
            f"{name} is bulk data and the single-cell refusal fired on it")
        assert card["read"] is True, (
            f"{name} stopped classifying, so the refusal cost a reading it "
            f"should not have")


@pytest.mark.parametrize("zeros,columns,median_nz,refused", [
    (0.90, 1200, 1.0, True),
    (0.79, 1200, 1.0, False),   # sparse, and not sparse enough
    (0.90, 900, 1.0, False),    # sparse and narrow — a bulk matrix can be
    (0.90, 1200, 4.0, False),   # sparse and wide with real depth
], ids=["all three", "zeros below", "too narrow", "depth too high"])
def test_the_three_criteria_are_conjunctive(zeros, columns, median_nz, refused):
    """Each one alone describes ordinary bulk data, and the pack names all three.

    Asserted against a constructed reading rather than a frame, because the
    claim is about the predicate: building four frames with exactly these
    statistics would be the fixture deciding the answer.
    """
    from turbotab import packs

    reading = {"pct_zeros": zeros, "n_columns": columns,
               "median_nonzero": median_nz}
    assert packs.reads_as_single_cell(reading) is refused


def test_the_threshold_took_the_safe_end_of_the_research_range():
    """The range is 80–90% and the app takes 80. Pinned with the reason.

    A later loop raising it to 90 would be relaxing a refusal, and this is where
    that argument has to be had rather than in a diff.
    """
    from turbotab import packs

    assert packs._SINGLE_CELL_ZERO_SHARE == 0.80, (
        "the zero-fraction threshold moved. The research states a RANGE, "
        "80–90%, and the app takes the lower end because a false refusal costs "
        "a user being told to bring bulk data while a false acceptance costs a "
        "count model fitted where its variance assumption does not hold")
    assert packs._SINGLE_CELL_MIN_COLUMNS == 1000
    assert packs._SINGLE_CELL_MEDIAN_NONZERO == 3.0
