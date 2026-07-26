"""The lockbox must group by the person, not by whatever the person sits in.

detect_repeated_subjects once ranked by MOST distinct values, so a near-
continuous lab value outranked the real subject_id. The fix flipped it to
FEWEST distinct — coarsest wins — on the argument that "grouping by a unit that
contains the subject can only keep more of a person on one side". The code
never checked containment, and `_SUBJECT_ID_TOKENS` matches the bare token
`id`, so site_id, plate_id, batch_id, clinic_id and run_id all qualified as
subject IDs and every one of them is coarser than the subject_id beside it.

Two ways that goes wrong, both silent, both driven by the audit:
  - crossed: samples randomized across assay plates, so grouping by plate puts
    the same person on both sides — the exact leak the detector exists for.
  - too few levels: 4 clinics is under the 8-group minimum, so grouping was
    abandoned altogether and a row-wise split ran with no warning at all.

Ranking is now by what the column IS, and containment is actually tested.
"""
import numpy as np
import pandas as pd
import pytest
import streamlit as st

from utils.test_lockbox import (
    detect_repeated_subjects, rank_grouping_candidates, ensure_lockbox,
    _id_kind, _nests_within,
)


@pytest.fixture(autouse=True)
def clean():
    st.session_state.clear(); yield; st.session_state.clear()


def crossed_plates(n_sub=40, visits=3, plates=12):
    """Participants x visits, samples randomized across assay plates."""
    rng = np.random.default_rng(7)
    n = n_sub * visits
    return pd.DataFrame({
        "subject_id": np.repeat(np.arange(n_sub), visits),
        "plate_id": rng.integers(0, plates, n),      # crossed, not nested
        "crp": rng.normal(3, 1, n),
        "y": rng.integers(0, 2, n),
    })


def nested_clinics(n_sub=25, visits=4, clinics=4):
    """Participants recruited at a handful of clinics; each person at one."""
    rng = np.random.default_rng(9)
    return pd.DataFrame({
        "subject_id": np.repeat(np.arange(n_sub), visits),
        "clinic_id": np.repeat(rng.integers(0, clinics, n_sub), visits),
        "crp": rng.normal(3, 1, n_sub * visits),
        "y": rng.integers(0, 2, n_sub * visits),
    })


def test_a_crossed_plate_never_outranks_the_participant():
    df = crossed_plates()
    assert detect_repeated_subjects(df)[0] == "subject_id"
    assert not _nests_within(df, "subject_id", "plate_id"), "fixture is not crossed"


def test_a_clinic_that_contains_people_still_does_not_outrank_them():
    """Even genuine nesting must not silently turn a person-split into a site-split."""
    df = nested_clinics()
    assert _nests_within(df, "subject_id", "clinic_id"), "fixture is not nested"
    assert detect_repeated_subjects(df)[0] == "subject_id"


@pytest.mark.parametrize("name,kind", [
    ("subject_id", "subject"), ("SubjectID", "subject"), ("participant_id", "subject"),
    ("seqn", "subject"), ("patient_id", "subject"),
    ("site_id", "cluster"), ("plate_id", "cluster"), ("batch_id", "cluster"),
    ("clinic_id", "cluster"), ("run_id", "cluster"), ("cohort_id", "cluster"),
    ("visit_id", None), ("replicate_id", None),
    ("uric_acid", None), ("RIDAGEYR", None),
])
def test_the_name_says_what_kind_of_thing_it_identifies(name, kind):
    assert _id_kind(name) == kind


def test_the_split_is_grouped_by_the_participant_not_the_plate():
    df = crossed_plates()
    st.session_state["raw_data"] = df
    lb = ensure_lockbox(df, "y", "classification")
    assert lb["group_col"] == "subject_id"
    assert lb["group_kind"] == "subject"
    held = set(df.loc[lb["labels"], "subject_id"])
    trained = set(df.loc[[i for i in df.index if i not in set(lb["labels"])], "subject_id"])
    assert not (held & trained), "the same person is on both sides of the lockbox"


def test_too_few_groups_to_split_by_is_said_out_loud():
    """4 clinics, no subject column: grouping is impossible and must not be silent."""
    df = nested_clinics(n_sub=25, visits=4, clinics=4).drop(columns=["subject_id"])
    st.session_state["raw_data"] = df
    lb = ensure_lockbox(df, "y", "classification")
    assert lb["group_col"] is None, "4 clinics cannot support a grouped split"
    note = st.session_state.get("_lockbox_grouping_abandoned")
    assert note, "the fallback to a row-wise split was silent"
    assert note["column"] == "clinic_id" and note["n_groups"] == 4


def test_a_cluster_grouping_is_labeled_as_what_it_is():
    """No subject column, 10 sites: grouping is legitimate, 'subjects' is not."""
    rng = np.random.default_rng(5)
    df = pd.DataFrame({
        "site_id": np.repeat(np.arange(10), 18),
        "crp": rng.normal(3, 1, 180),
        "y": rng.integers(0, 2, 180),
    })
    st.session_state["raw_data"] = df
    lb = ensure_lockbox(df, "y", "classification")
    assert lb["group_col"] == "site_id" and lb["group_kind"] == "cluster"
    assert "subject" not in lb["group_noun"], (
        f"{lb['n_test_groups']} sites reported as {lb['group_noun']}")


def test_a_per_sample_barcode_does_not_outrank_the_participant():
    """The failure the coarsest-wins rule was reaching for, still fixed."""
    rng = np.random.default_rng(13)
    n_sub, reps = 30, 4
    df = pd.DataFrame({
        "subject_id": np.repeat(np.arange(n_sub), reps),
        "aliquot_id": np.arange(n_sub * reps) // 2,   # finer than a person
        "y": rng.integers(0, 2, n_sub * reps),
    })
    assert detect_repeated_subjects(df)[0] == "subject_id"


def test_the_finest_person_column_climbs_to_the_one_that_contains_it():
    """subject_visit_id nests inside subject_id: group by the person."""
    rng = np.random.default_rng(17)
    n_sub, visits = 30, 3
    df = pd.DataFrame({
        "subject_id": np.repeat(np.arange(n_sub), visits * 2),
        "subject_visit_id": np.repeat(np.arange(n_sub * visits), 2),
        "y": rng.integers(0, 2, n_sub * visits * 2),
    })
    ranked = rank_grouping_candidates(df)
    assert ranked[0]["column"] == "subject_id", [c["column"] for c in ranked]
