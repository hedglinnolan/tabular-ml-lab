"""The extraction, verified: `ml.splits.make_split` reproduces the page.

Move-then-verify. `tests/integration/test_characterization_split.py` pins what
`pages/06_Train_and_Compare.py` does today by driving the real page. This file
runs the *extracted* function over the same inputs and asserts it produces the
same partition — not a similar one, the same rows on the same side.

If a test here fails, the extraction changed behavior and is wrong. The fix is
the extracted code, never this file and never the characterization file.

Also covers the two identity-barrier tests `T0-ID-001` calls for:

- no post-barrier operation changes a surviving row's index;
- a split's labels still name the rows they were chosen from.
"""
import numpy as np
import pandas as pd
import pytest

from tests.integration.conftest import (
    build_classification_dataframe, build_test_dataframe, inject_data_state)
from tests.integration.test_characterization_split import (
    _inject_pipeline, _partition, _prepare, _route_grouped, _route_time, _seal_lockbox)

from ml.splits import Split, SplitError, SplitSpec, choose_strategy, make_split

pytestmark = pytest.mark.timeout(300)


def _page_partition(df, target_col="glucose", task_type="regression",
                    route=None, lockbox=None):
    """Run the real page and return its partition."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("pages/06_Train_and_Compare.py")
    inject_data_state(at, df, target_col=target_col, task_type=task_type)
    _inject_pipeline(at, df, target_col=target_col)
    if lockbox is not None:
        _seal_lockbox(at, df, lockbox)
    if route:
        route(at)
    return _partition(_prepare(at))


def _feature_cols(df, target_col):
    """The features the page selects: everything but the target."""
    return [c for c in df.columns if c != target_col]


def _same_partition(page, split, label="partition"):
    """The page stores POSITIONS into the split frame; so does `Split`."""
    assert sorted(page["train_indices"]) == sorted(split.train_positions), \
        f"{label}: train rows differ"
    assert sorted(page["val_indices"]) == sorted(split.val_positions), \
        f"{label}: validation rows differ"
    assert sorted(page["test_indices"]) == sorted(split.test_positions), \
        f"{label}: test rows differ"
    assert page["cv_strategy"] == split.cv_strategy, f"{label}: cv scheme differs"


# ── branch 4 · plain random ──────────────────────────────────────────────

def test_random_split_matches_the_page():
    df = build_test_dataframe(n=200)
    page = _page_partition(df)
    split = make_split(df, _feature_cols(df, "glucose"), "glucose", "regression",
                       SplitSpec(random_state=42))
    _same_partition(page, split, "random")


def test_stratified_split_matches_the_page():
    df = build_classification_dataframe(n=240)
    target = "outcome" if "outcome" in df.columns else df.columns[-1]
    page = _page_partition(df, target_col=target, task_type="classification")
    split = make_split(df, _feature_cols(df, target), target, "classification",
                       SplitSpec(random_state=42, stratify=True))
    _same_partition(page, split, "stratified")


# ── branch 3 · lockbox ───────────────────────────────────────────────────

def test_lockbox_split_matches_the_page():
    df = build_test_dataframe(n=200)
    rng = np.random.RandomState(42)
    labels = sorted(rng.choice(df.index.values, size=30, replace=False).tolist())

    page = _page_partition(df, lockbox=labels)
    split = make_split(df, _feature_cols(df, "glucose"), "glucose", "regression",
                       SplitSpec(random_state=42), lockbox_labels=labels)

    _same_partition(page, split, "lockbox")
    assert split.lockbox_applied is True
    # And the sealed labels really are the test set, by label not position.
    assert set(split.test_labels) == set(labels)


# ── branch 1 · grouped ───────────────────────────────────────────────────

def test_grouped_split_matches_the_page():
    base = build_test_dataframe(n=120)
    df = pd.concat([base, base, base], ignore_index=True)
    df["subject_id"] = np.tile(np.arange(len(base)), 3)

    page = _page_partition(df, route=lambda at: _route_grouped(at, "subject_id"))
    split = make_split(
        df, _feature_cols(df, "glucose"), "glucose", "regression",
        SplitSpec(random_state=42, use_group_split=True, entity_id_col="subject_id"))

    _same_partition(page, split, "grouped")
    assert split.cv_strategy == "group"
    assert split.cv_groups_train is not None
    assert len(split.cv_groups_train) == len(split.X_train)
    # The promise the branch exists for.
    subj = df["subject_id"]
    assert not (set(subj.loc[split.train_labels]) & set(subj.loc[split.test_labels]))


# ── branch 2 · chronological ─────────────────────────────────────────────

def test_chronological_split_matches_the_page():
    df = build_test_dataframe(n=200)
    df["visit_date"] = pd.date_range("2020-01-01", periods=len(df), freq="D")
    df = df.sample(frac=1.0, random_state=7).reset_index(drop=True)

    page = _page_partition(df, route=lambda at: _route_time(at, "visit_date"))
    split = make_split(
        df, _feature_cols(df, "glucose"), "glucose", "regression",
        SplitSpec(random_state=42, use_time_split=True, datetime_col="visit_date"))

    _same_partition(page, split, "chronological")
    assert split.cv_strategy == "time"
    dates = df["visit_date"]
    assert dates.loc[split.train_labels].max() <= dates.loc[split.val_labels].min()
    assert dates.loc[split.val_labels].max() <= dates.loc[split.test_labels].min()


def test_stored_indices_match_the_page_when_rows_are_dropped():
    """The case where positions and labels genuinely diverge.

    `pages/06` stores `original_indices[idx]` — positions into the **source**
    frame. `Split.*_positions` are offsets into the **split** frame, the rows
    left after a missing target is dropped. With nothing dropped the two
    coincide, which is why the other equivalence tests pass on positions alone;
    with rows dropped they do not, and the translation has to go through labels.

    This pins that translation, so rewiring the page cannot get it subtly wrong
    for exactly the datasets that have missing values.
    """
    df = build_test_dataframe(n=200)
    df.loc[df.index[:25], "glucose"] = np.nan

    page = _page_partition(df)
    split = make_split(df, _feature_cols(df, "glucose"), "glucose", "regression",
                       SplitSpec(random_state=42))

    # Positions in the split frame are NOT what the page stores here.
    assert sorted(page["train_indices"]) != sorted(split.train_positions), (
        "the fixture no longer drops rows, so this test proves nothing")

    # Going through the labels reproduces the page exactly.
    for page_key, labels in (("train_indices", split.train_labels),
                             ("val_indices", split.val_labels),
                             ("test_indices", split.test_labels)):
        expected = sorted(df.index.get_indexer(labels).tolist())
        assert sorted(page[page_key]) == expected, f"{page_key} did not translate"


# ── the priority order ───────────────────────────────────────────────────

def test_branch_priority_is_group_then_time_then_lockbox():
    df = build_test_dataframe(n=90)
    df["subject_id"] = np.arange(len(df))
    df["visit_date"] = pd.date_range("2020-01-01", periods=len(df), freq="D")
    labels = list(df.index[:15])

    both = SplitSpec(use_group_split=True, entity_id_col="subject_id",
                     use_time_split=True, datetime_col="visit_date")
    assert choose_strategy(df, both, labels) == "grouped"

    time_only = SplitSpec(use_time_split=True, datetime_col="visit_date")
    assert choose_strategy(df, time_only, labels) == "chronological"

    assert choose_strategy(df, SplitSpec(), labels) == "lockbox"
    assert choose_strategy(df, SplitSpec(), None) == "random"
    assert choose_strategy(df, SplitSpec(stratify=True), None) == "stratified"


# ── purity ───────────────────────────────────────────────────────────────

def test_make_split_does_not_mutate_its_input():
    df = build_test_dataframe(n=120)
    before = df.copy(deep=True)
    make_split(df, _feature_cols(df, "glucose"), "glucose", "regression")
    pd.testing.assert_frame_equal(df, before)


def test_make_split_is_reproducible():
    df = build_test_dataframe(n=200)
    a = make_split(df, _feature_cols(df, "glucose"), "glucose", "regression",
                   SplitSpec(random_state=7))
    b = make_split(df, _feature_cols(df, "glucose"), "glucose", "regression",
                   SplitSpec(random_state=7))
    assert a.train_labels == b.train_labels and a.test_labels == b.test_labels


def test_make_split_imports_without_streamlit():
    """`ml/splits.py` is engine code now; Streamlit must not be reachable from it."""
    import ml.splits
    import sys
    assert "streamlit" not in getattr(ml.splits, "__dict__", {})
    src = open(ml.splits.__file__, encoding="utf-8").read()
    assert "streamlit" not in src, "the extracted module still mentions Streamlit"


# ── the identity barrier · T0-ID-001 ─────────────────────────────────────

def test_no_post_barrier_operation_changes_a_surviving_rows_index():
    """First identity-barrier test.

    Splitting happens after rows acquire identities. Whatever it drops, the rows
    that survive keep the labels they arrived with — so a label chosen for the
    test set still names the same row afterwards.
    """
    df = build_test_dataframe(n=200)
    # A frame that has already been filtered once: labels are not 0..n-1, which
    # is precisely when positions and labels stop agreeing.
    df = df.iloc[::2].copy()
    df.loc[df.index[:10], "glucose"] = np.nan

    split = make_split(df, _feature_cols(df, "glucose"), "glucose", "regression")

    every = list(split.train_labels) + list(split.val_labels) + list(split.test_labels)
    assert set(every) <= set(df.index), "the split invented labels"
    # Every label still names the row it was chosen from — checked by value.
    for lbl in split.test_labels[:20]:
        row_in_split = split.X_test.iloc[split.test_labels.index(lbl)]
        for col in split.feature_names:
            a, b = row_in_split[col], df.at[lbl, col]
            assert (pd.isna(a) and pd.isna(b)) or a == b, (
                f"label {lbl} names a different row after the split")
    split.assert_identity_preserved(df)


def test_a_renumbered_frame_is_caught_rather_than_silently_wrong():
    """Second identity-barrier test.

    If a pre-barrier repair runs *after* the split — the thing the barrier
    forbids — the stored labels no longer name the rows they were chosen from.
    `reset_index(drop=True)` is exactly what four of the nine repair kinds do.
    Nothing can recover the mapping, so it must be detected, not absorbed.
    """
    df = build_test_dataframe(n=200)
    df.index = range(500, 500 + len(df))
    split = make_split(df, _feature_cols(df, "glucose"), "glucose", "regression")
    split.assert_identity_preserved(df)          # holds against the real frame

    renumbered = df.reset_index(drop=True)       # the forbidden post-barrier repair
    with pytest.raises(SplitError, match="renumbered"):
        split.assert_identity_preserved(renumbered)


def test_labels_and_positions_disagree_when_the_frame_was_filtered():
    """Why labels are the identity and positions are a convenience.

    On a frame whose index is not a clean `RangeIndex`, a stored position and a
    stored label point at different rows. `pages/07` reads positions today; this
    records the size of that gap so it is a known cost, not a surprise.
    """
    df = build_test_dataframe(n=100)
    df.index = range(1000, 1100)
    split = make_split(df, _feature_cols(df, "glucose"), "glucose", "regression")

    assert split.test_positions != split.test_labels
    assert all(l >= 1000 for l in split.test_labels)
    assert all(p < 100 for p in split.test_positions)


# ── refusals ─────────────────────────────────────────────────────────────

def test_a_single_class_target_is_refused():
    df = build_test_dataframe(n=60)
    df["const"] = "only"
    with pytest.raises(SplitError, match="Single-class"):
        make_split(df, ["glucose"], "const", "classification")


def test_fractions_that_do_not_sum_are_refused():
    df = build_test_dataframe(n=60)
    with pytest.raises(SplitError, match="sum to 1.0"):
        make_split(df, _feature_cols(df, "glucose"), "glucose", "regression",
                   SplitSpec(train_size=0.9, val_size=0.3, test_size=0.3))


def test_a_missing_feature_is_named():
    df = build_test_dataframe(n=60)
    with pytest.raises(SplitError, match="not in the data"):
        make_split(df, ["glucose_typo"], "glucose", "regression")
