"""The seal states its own basis — three states, never two.

Lockbox constitution §03. `group_col: None` used to carry two different claims:
*this study has one row per person*, and *we could not tell*. A consumer reading
the record rather than the chip could not separate them, which is what
`IMPORT-020` exploited — failure to detect was indistinguishable from success,
behind a clean lock icon.

Leaking and saying so is the governing rule's **refuse** branch; that is why
`IMPORT-021` closed. Leaking behind a lock icon is its **assert something
false** branch. So `undetermined` is first-class: persisted, asserted here, and
never rendered as a clean lock. It is an advisory with exploratory labeling and
not a hard block — a user who genuinely does not know their own data's shape
should get honest numbers, not a locked door.

Findings: `IMPORT-020` (the leak this makes visible), `IMPORT-021` (the half
left open — the record, as opposed to the chip).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import streamlit as st

from utils.test_lockbox import (BASIS_DETECTED, BASIS_INHERITED,
                                BASIS_USER_STATED, SEAL_ABANDONED,
                                SEAL_CROSS_SECTIONAL, SEAL_GROUPED,
                                SEAL_UNDETERMINED, _MAX_ROWS_PER_GROUP,
                                detect_repeated_subjects, ensure_lockbox)


@pytest.fixture(autouse=True)
def clean():
    st.session_state.clear()
    yield
    st.session_state.clear()


def cohort(ids, key="SEQN", seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({key: list(ids),
                         "age": rng.integers(20, 80, len(list(ids))),
                         "y": rng.integers(0, 2, len(list(ids)))})


def seal(df, key="SEQN"):
    st.session_state["raw_data"] = df
    lb = ensure_lockbox(df, "y", "classification")
    sealed = set(lb["labels"])
    held = set(df.loc[df.index.isin(sealed), key])
    trained = set(df.loc[~df.index.isin(sealed), key])
    return lb, len(held & trained)


# ── the four bases, each recorded ────────────────────────────────────────

def test_a_grouped_seal_says_it_is_grouped():
    lb, leak = seal(cohort(np.repeat(range(60), 3)))
    assert lb["seal_basis"] == SEAL_GROUPED
    assert lb["group_col"] == "SEQN"
    assert leak == 0


def test_an_abandoned_seal_says_so_and_is_not_cross_sectional():
    """`IMPORT-021`: it leaks, and closing required only that it say so.

    The chip already disclosed. What was missing was the *record*: a consumer
    reading the dict saw `group_col: None` and could not tell this from a study
    with one row per person.
    """
    lb, leak = seal(cohort(np.repeat(range(7), 5)))
    assert lb["seal_basis"] == SEAL_ABANDONED
    assert lb["seal_basis"] != SEAL_CROSS_SECTIONAL
    assert leak > 0, "the fixture no longer exercises the leak"
    assert st.session_state.get("_lockbox_grouping_abandoned")


def test_an_undetermined_seal_is_never_recorded_as_cross_sectional():
    """The state `IMPORT-020` exploited, now nameable.

    Ten subjects after a many-to-many merge: 60 rows per value, above the
    identifier band. The column is not usable for grouping and the repetition
    is unmistakable, so the honest answer is "could not tell" — not "no
    repetition".
    """
    lb, leak = seal(cohort(np.repeat(range(10), 60)))
    assert lb["seal_basis"] == SEAL_UNDETERMINED
    assert lb["seal_basis"] != SEAL_CROSS_SECTIONAL
    assert leak > 0, "the fixture no longer exercises the leak"
    assert lb["undetermined_because"], "the record does not say what was unclear"
    assert lb["undetermined_because"][0]["rows_per"] > _MAX_ROWS_PER_GROUP


def test_a_genuinely_cross_sectional_seal_says_cross_sectional():
    """Silence must not become the answer to everything."""
    lb, leak = seal(cohort(range(200)))
    assert lb["seal_basis"] == SEAL_CROSS_SECTIONAL
    assert lb["undetermined_because"] is None
    assert leak == 0


def test_the_four_bases_are_distinct():
    assert len({SEAL_GROUPED, SEAL_ABANDONED, SEAL_UNDETERMINED,
                SEAL_CROSS_SECTIONAL}) == 4


# ── the record carries HOW we know, not just what we concluded ───────────

def test_every_seal_records_its_source():
    """Constitution §02 lands later without a schema change.

    `basis_source` is written now — everything is `detected` today — so
    `user_stated` and `inherited_from_assembly` arrive without migrating a
    persisted, round-tripped artifact.
    """
    for ids in (range(200), np.repeat(range(60), 3), np.repeat(range(7), 5),
                np.repeat(range(10), 60)):
        st.session_state.clear()
        lb, _ = seal(cohort(ids))
        assert lb["basis_source"] == BASIS_DETECTED
    assert len({BASIS_DETECTED, BASIS_USER_STATED, BASIS_INHERITED}) == 3


def test_the_basis_survives_a_round_trip_through_the_session_archive():
    """The record is persisted and reloaded; the basis must come back."""
    import json

    lb, _ = seal(cohort(np.repeat(range(10), 60)))
    restored = json.loads(json.dumps(
        {k: v for k, v in lb.items() if k != "labels"}, default=str))
    assert restored["seal_basis"] == SEAL_UNDETERMINED
    assert restored["basis_source"] == BASIS_DETECTED


# ── the lower bound was wrong on its own terms ───────────────────────────

def test_partial_follow_up_is_detected():
    """`IMPORT-020` gate 1. `rows_per < 1.5` rejected 1.30.

    Only some subjects have a second visit — the commonest longitudinal shape
    there is — and any `k < n` means repetition exists by definition. Removing
    the bound does not close the hole (constitution §02 is explicit that
    nothing about ratios can); it is removed because it was wrong, and a
    heuristic demoted to *suggestion and contradiction detector* is worse at
    both jobs with a wrong bound.
    """
    df = cohort(list(range(100)) + list(range(30)))
    assert detect_repeated_subjects(df) == ("SEQN", 100, 130)
    lb, leak = seal(df)
    assert lb["seal_basis"] == SEAL_GROUPED
    assert leak == 0, "partial follow-up still puts subjects on both sides"


def test_a_many_to_many_product_is_undetermined_rather_than_silent():
    """`IMPORT-020` gate 1b. The upper bound has a real purpose and keeps it.

    A category repeats hundreds of times, so the bound stays — but a merge
    product repeats *harder* than an identifier, not less, and rejecting it
    silently is what produced a clean lock over a 10-of-10 leak.
    """
    df = cohort(np.repeat(range(10), 60))
    assert detect_repeated_subjects(df) is None
    assert st.session_state.get("_lockbox_repetition_unclear"), (
        "the upper-bound rejection left no trace")
    lb, leak = seal(df)
    assert lb["seal_basis"] == SEAL_UNDETERMINED
    assert leak == 10


# ── what is still false, pinned so it cannot be forgotten ────────────────

def test_KNOWN_GAP_an_unrecognized_id_name_is_still_recorded_as_cross_sectional():
    """KNOWN GAP, asserted so it stays visible: `IMPORT-022`.

    The `KNOWN_GAP_` prefix is load-bearing and belongs in the NAME, not only
    here. CI prints the name and nothing else, so a green line reading
    `test_an_unrecognized_id_name_is_still_recorded_as_cross_sectional PASSED`
    is indistinguishable from a regression guard — it scans as the suite
    endorsing the behavior it is in fact recording as broken. Same silence this
    project keeps finding, one layer up. See `FEATURE_PARITY.md`, "name every
    test after the defect it guards".

    `SUBJ` repeats 60 values × 3 rows — a textbook repeated-measures shape —
    and the name-token gate rejects it before the repetition is ever measured,
    so the seal records `cross_sectional` and renders a clean lock over a
    23-of-60 leak.

    Constitution §02 says name lists cannot close this and must not be tuned as
    though they could: the fix is to *ask*. This test asserts the current, wrong
    behavior deliberately — it will fail when the grain question ships, which is
    the point. Change it then; do not change it before.
    """
    df = cohort(np.repeat(range(60), 3), key="SUBJ")
    assert detect_repeated_subjects(df) is None
    lb, leak = seal(df, key="SUBJ")
    assert leak > 0, "the fixture no longer exercises the leak"
    assert lb["seal_basis"] == SEAL_CROSS_SECTIONAL, (
        "the name gate now routes to a truthful basis — good; update "
        "IMPORT-022 and this test together")
