"""CONTRACT-001 / MINE-014 / STATE-014 / STATE-025: the held-out rows are the
rows the split named, or the page refuses.

Train & Compare used to resolve split membership to POSITIONS in the frame
`get_data()` returned at split time (`df.index.get_indexer(...)`) and store them
as train/val/test_indices. Explainability then fetched `get_data()` FRESH and
did `df_raw.iloc[test_indices]` for the SHAP/permutation matrix, for `y`, and
for the subgroup strata. Any row-set change in between — page 05's Build
Pipelines writing or popping `filtered_data`, feature engineering re-applied, a
row-dropping repair — made those positions name different people. Where the new
frame was shorter the IndexError was swallowed by a bare `except Exception:
pass`; where it was the same length or longer there was no error at all, and the
subgroup path had no try/except: one person's prediction was tabulated under
another person's stratum, bootstrapped 200 times, and drawn as a forest plot.

The fix is one convention: labels are the identity, `resolve_split_rows` is the
only way back to the rows, and it refuses when any recorded row is missing from
the active frame.

STATE-037 / CONTRACT-013 are the other half — `get_data()` used to let
`df_engineered` SHADOW `filtered_data`, so a plausibility filter applied after
feature engineering (the documented page order) removed no rows from anything
downstream while page 05 said it had.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.splits import SplitIdentityError, SplitSpec, make_split, resolve_split_rows

REPO = Path(__file__).resolve().parent.parent


def _study(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "age": rng.normal(50, 12, n),
            "bmi": rng.normal(27, 5, n),
            "sex": rng.choice(["male", "female"], n),
            "glucose": rng.normal(100, 15, n),
        },
        index=pd.RangeIndex(n),
    )


def _split_of(df: pd.DataFrame):
    return make_split(df, ["age", "bmi"], "glucose", "regression",
                      SplitSpec(random_state=42))


# ── the silent-misalignment route ────────────────────────────────────────

def test_contract_001_a_longer_frame_returns_the_same_people_not_the_same_slots():
    """Page 05 pops `filtered_data`; the frame grows; positions shift silently.

    This is the route the verifier established: Build Pipelines with
    plausibility mode off pops `filtered_data`, so the frame `get_data()`
    serves goes back to its full length. Every held-out row is still there —
    only its POSITION moved — so `.iloc` raised nothing and explained whoever
    now sat at those offsets.
    """
    study = _study()
    # The frame at split time: page 05 had filtered some rows out.
    trained_on = study.drop(index=study.index[10:30])
    split = _split_of(trained_on)

    positions = trained_on.index.get_indexer(split.test_labels).tolist()

    # The frame at explain time: the filter was popped, the rows came back.
    now = study

    # The old convention, verbatim, so the fixture is known to bite.
    by_position = list(now.iloc[positions].index)
    assert by_position != list(split.test_labels), (
        "fixture no longer shifts the positions, so it proves nothing")

    # The new one. Same people, in the order the split recorded them.
    by_label = resolve_split_rows(now, split.test_labels, part="test")
    assert list(by_label.index) == list(split.test_labels)
    pd.testing.assert_frame_equal(by_label, study.loc[list(split.test_labels)])


def test_contract_001_missing_held_out_rows_are_refused_not_answered():
    """Rows dropped after the split: there is no answer, so there is a refusal."""
    study = _study()
    split = _split_of(study)

    # Page 05 in filter mode, exploratory (no lockbox to restore): some of the
    # held-out rows are gone, and the frame is still long enough to index.
    dropped = list(split.test_labels[:5])
    now = study.drop(index=dropped)

    positions = study.index.get_indexer(split.test_labels).tolist()
    silently_wrong = list(now.iloc[positions].index)
    assert silently_wrong != list(split.test_labels), (
        "fixture no longer misaligns, so it proves nothing")

    with pytest.raises(SplitIdentityError) as exc:
        resolve_split_rows(now, split.test_labels, part="test")
    message = str(exc.value)
    assert "5 of" in message and "test row" in message
    # Actionable: it says what changed and what to do about it.
    assert "Prepare Splits" in message
    assert str(dropped[0]) in message


def test_state_014_rows_come_back_in_the_order_the_split_recorded():
    """`y_test` and the strata must line up person for person."""
    study = _study()
    split = _split_of(study)

    # A frame whose row ORDER changed but whose rows did not.
    shuffled = study.sample(frac=1.0, random_state=3)

    rows = resolve_split_rows(shuffled, split.test_labels, part="test")
    assert list(rows.index) == list(split.test_labels)
    # The pairing the subgroup table depends on.
    np.testing.assert_allclose(rows["glucose"].values, split.y_test)


def test_state_025_duplicate_labels_are_refused():
    """A duplicated label names two rows, so `.loc` would return more than the
    split had and misalign every vector stored beside it."""
    study = _study()
    split = _split_of(study)
    doubled = pd.concat([study, study.loc[[split.test_labels[0]]]])

    with pytest.raises(SplitIdentityError, match="duplicate row labels"):
        resolve_split_rows(doubled, split.test_labels, part="test")


def test_state_014_an_unrecorded_split_is_refused_rather_than_guessed():
    with pytest.raises(SplitIdentityError, match="No test row labels"):
        resolve_split_rows(_study(), None, part="test")
    with pytest.raises(SplitIdentityError, match="no active dataset"):
        resolve_split_rows(None, [0, 1, 2], part="test")


# ── the accessor is what the pages call ──────────────────────────────────

@pytest.fixture
def session():
    import streamlit as st
    st.session_state.clear()
    yield st.session_state
    st.session_state.clear()


def test_contract_001_get_split_rows_refuses_when_the_frame_changed(session):
    """End to end through session state: train, change the rows, explain."""
    from utils.session_state import get_split_rows

    study = _study()
    session["raw_data"] = study
    split = _split_of(study)
    # What page 06 stores.
    session["test_row_labels"] = list(split.test_labels)

    # Nothing changed yet: the rows resolve.
    assert list(get_split_rows("test").index) == list(split.test_labels)

    # Page 05 writes a filtered frame that drops some held-out rows.
    session["filtered_data"] = study.drop(index=list(split.test_labels[:3]))
    with pytest.raises(SplitIdentityError):
        get_split_rows("test")


def test_mine_014_the_positional_split_keys_are_gone(session):
    """Page 06 must not write positions, and page 07 must not read them.

    The keys are the trap: while they exist, any future page can index a fresh
    frame with them and be wrong without an error.
    """
    page06 = (REPO / "pages" / "06_Train_and_Compare.py").read_text()
    for part in ("train", "val", "test"):
        assert not re.search(rf"session_state\.{part}_indices\s*=", page06), (
            f"page 06 still writes {part}_indices as positions")
        assert not re.search(rf"session_state\[[\"']{part}_indices[\"']\]\s*=", page06)

    for part in ("train", "val", "test"):
        assert f"{part}_row_labels" in page06, (
            f"page 06 no longer records the {part} row labels")

    page07 = (REPO / "pages" / "07_Explainability.py").read_text()
    assert "test_indices" not in page07, (
        "page 07 still reads the positional test set")
    assert ".iloc[test_indices]" not in page07
    # Both consumers — the SHAP/permutation matrix and the subgroup strata —
    # go through the one accessor that refuses.
    assert page07.count("_held_out_rows(") >= 3


# ── STATE-037 / CONTRACT-013 · the row filter is a mask, not a rival frame ──

def test_state_037_the_plausibility_filter_survives_feature_engineering(session):
    """FE first, filter second — the documented order — must still filter.

    `df_engineered` used to win outright, so the rows page 05 said it had
    removed were in every count and in the training data.
    """
    from utils.session_state import get_data

    study = _study()
    session["raw_data"] = study
    # Feature engineering ran first and added a column.
    engineered = study.assign(bmi_sq=study["bmi"] ** 2)
    session["df_engineered"] = engineered
    # Then page 05's plausibility filter removed rows.
    implausible = list(study.index[:8])
    session["filtered_data"] = study.drop(index=implausible)

    active = get_data()
    assert not set(implausible) & set(active.index), (
        "the excluded rows are still in the frame every page trains on")
    assert len(active) == len(study) - len(implausible)
    # And the engineered columns are still there — a mask, not a fallback.
    assert "bmi_sq" in active.columns


def test_contract_013_a_filter_written_after_engineering_is_not_a_no_op(session):
    """Order-independence: filter first then engineer gives the same rows."""
    from utils.session_state import get_data

    study = _study()
    session["raw_data"] = study
    kept = study.drop(index=list(study.index[:8]))
    session["filtered_data"] = kept
    session["df_engineered"] = kept.assign(bmi_sq=kept["bmi"] ** 2)

    active = get_data()
    assert list(active.index) == list(kept.index)
    assert "bmi_sq" in active.columns


def test_state_037_a_foreign_filtered_frame_does_not_empty_the_dataset(session):
    """A `filtered_data` from another vintage shares no labels; masking with it
    would leave nothing. Only set_data and a new cohort can produce that, and
    both pop it — but an empty dataset must never be the answer."""
    from utils.session_state import get_data

    study = _study()
    session["raw_data"] = study
    session["df_engineered"] = study.assign(bmi_sq=study["bmi"] ** 2)
    session["filtered_data"] = study.set_index(study.index + 10_000)

    active = get_data()
    assert len(active) == len(study)
