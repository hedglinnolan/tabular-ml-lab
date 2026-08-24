"""CONTRACT-005 / MINE-002: a save/restore must not renumber the sealed rows.

The lockbox stores index LABELS and tests membership by label
(utils/test_lockbox.py), and the repo says so twice: session_manager's lockbox
block ("must survive a save/restore VERBATIM") and ml/pipeline.py's
apply_plausibility_filter ("renumbering the survivors 0..n-1 would leave both of
those sets pointing at whoever now happens to sit at those positions").

But `_df_to_parquet_bytes` wrote `to_parquet(index=False)`, so every restored
frame came back with a fresh 0..n-1 RangeIndex while the labels stayed where
they were. Nothing raised: `_lockbox_signature` hashes with `index=False`, so
the signature was IDENTICAL after the renumbering and `ensure_lockbox` returned
the stale seal. The measured result on a 200-row frame with a gappy index (what
page 01's "Drop duplicate rows" leaves behind): 12 of 30 sealed rows still
sealed, 5 labels naming rows that no longer exist, and the chip still reporting
n=30.

Two things are asserted here:
  1. round trip — a gappy index survives, so lockbox membership is preserved
     row-for-row and train_row_mask partitions the same people.
  2. old archives — a file saved before the index was persisted cannot be told
     apart from a good one after the fact, so its lockbox is REFUSED with a
     warning rather than applied to renumbered rows.
"""
from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from tests.test_session_manager import fake_session  # noqa: F401
from utils import session_manager


def _gappy_study(n: int = 200) -> pd.DataFrame:
    """A frame with holes in its index, as `d.drop_duplicates()` leaves it.

    pages/01_Upload_and_Audit.py offers "Drop duplicate rows" as a one-click
    suggested action; utils/session_state.py resets only NON-unique indexes, so
    the gappy one reaches raw_data intact.
    """
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=n),
                       "b": rng.integers(0, 5, n),
                       "y": rng.integers(0, 2, n)})
    # duplicates in the MIDDLE, so the surviving labels have holes rather than
    # simply stopping short
    df = pd.concat([df.iloc[:60], df.iloc[:30], df.iloc[60:]],
                   ignore_index=True).drop_duplicates()
    assert df.index.max() > len(df) - 1, "fixture must leave a gappy index"
    return df


def _seal(state, df: pd.DataFrame) -> list:
    """Seal a holdout the way the app does, but without importing the lockbox's
    own streamlit host — this module's fake state is session_manager's."""
    rng = np.random.default_rng(7)
    labels = sorted(int(x) for x in rng.choice(df.index, size=30, replace=False))
    state["raw_data"] = df
    state["test_lockbox"] = {
        "labels": labels, "fraction": 0.15, "seed": 42,
        "n_total": len(df), "n_test": len(labels),
        "signature": "sig", "stratified": False,
    }
    return labels


def test_sealed_rows_survive_the_round_trip_with_a_gappy_index(fake_session):
    """The sealed rows after a restore must be the SAME PEOPLE, not the same
    numbers. This fails on `to_parquet(index=False)`: the labels come back
    intact and name different rows."""
    df = _gappy_study()
    labels = _seal(fake_session, df)
    before = df.loc[labels].reset_index(drop=True)

    archive, _ = session_manager._collect_session_data()
    fake_session.clear()
    _, _, warnings = session_manager._restore_session_data(archive)

    assert warnings == [], f"a clean round trip must not warn: {warnings}"
    back = fake_session["raw_data"]
    assert list(back.index) == list(df.index), "the index did not survive"

    restored_labels = fake_session["test_lockbox"]["labels"]
    assert restored_labels == labels, "labels must survive VERBATIM"
    missing = [lbl for lbl in restored_labels if lbl not in back.index]
    assert not missing, f"{len(missing)} sealed labels name rows that are gone"

    after = back.loc[restored_labels].reset_index(drop=True)
    pd.testing.assert_frame_equal(before, after)

    # and the partition every page reads is the same partition. train_row_mask
    # reads the real host, so hand the restored seal to it there.
    import streamlit as st
    from utils.test_lockbox import train_row_mask
    st.session_state.clear()
    try:
        st.session_state["test_lockbox"] = fake_session["test_lockbox"]
        mask = train_row_mask(back.index)
        assert int((~mask).sum()) == len(labels), (
            "the sealed count changed under the restore")
        assert sorted(int(x) for x in back.index[~mask]) == labels
    finally:
        st.session_state.clear()


def test_a_cohort_run_keeps_its_own_rows(fake_session):
    """Same mechanism, same file: a run holds index labels too, so a renumbered
    restore reports one group's name over another group's rows."""
    df = _gappy_study()
    fake_session["raw_data"] = df
    group = sorted(int(x) for x in df.index[df["b"] < 2])
    fake_session["cohort_run"] = {
        "column": "b", "value": 0, "label": "low b", "labels": group,
        "n_rows": len(group), "n_total": len(df), "position": 1, "of": 2,
        "order": ["low b", "high b"], "target_col": "y", "dropped_features": [],
    }

    archive, _ = session_manager._collect_session_data()
    fake_session.clear()
    _, _, warnings = session_manager._restore_session_data(archive)

    assert warnings == []
    back = fake_session["raw_data"]
    restored = fake_session["cohort_run"]["labels"]
    assert restored == group
    assert (back.loc[restored, "b"] < 2).all(), "the run named other people"


def _legacy_archive(df: pd.DataFrame, labels: list) -> bytes:
    """An archive exactly as the OLD code wrote it: no index in the parquet and
    no `index_preserved` flag in the manifest."""
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    members = {
        "manifest.json": json.dumps({
            "schema_version": session_manager.SAVE_SCHEMA_VERSION,
            "saved_at": "2026-06-01T00:00:00", "workflow_step": "05_Preprocess",
            "saved_keys": ["raw_data", "test_lockbox"], "skipped_keys": [],
            "members": [],
        }).encode(),
        "config.json": b"{}",
        "data/raw_data.parquet": buf.getvalue(),
        "lockbox.json": json.dumps({
            "labels": labels, "fraction": 0.15, "seed": 42,
            "n_total": len(df), "n_test": len(labels), "signature": "sig",
            "stratified": False,
        }).encode(),
    }
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return out.getvalue()


def test_an_archive_without_a_saved_index_refuses_its_lockbox(fake_session):
    """The old format cannot be rescued: nothing in it says whether the frame's
    index was contiguous at save time, and the labels fall inside the renumbered
    range either way. So the seal is refused OUT LOUD and re-drawn from the
    saved seed and fraction — never applied to rows it may not name."""
    df = _gappy_study()
    labels = sorted(int(x) for x in df.index[:30])

    restored, _, warnings = session_manager._restore_session_data(
        _legacy_archive(df, labels))

    assert "test_lockbox" not in fake_session, (
        "a seal was applied to rows the archive cannot vouch for")
    assert any("Sealed test set could not be restored" in w for w in warnings), (
        f"the refusal must be visible to the user: {warnings}")
    assert any("renumbered" in w for w in warnings), (
        "the warning must say WHY, so the user knows to re-draw")
    # the rest of the session still loads — refusing the seal is not refusing
    # the file
    assert restored >= 1
    assert fake_session["raw_data"] is not None


def test_a_lockbox_naming_missing_rows_is_refused(fake_session):
    """Even a current-format archive is checked: if the labels do not exist in
    the restored frame, the seal is wrong and gets refused rather than sealing
    whatever is at those positions."""
    df = _gappy_study()
    _seal(fake_session, df)
    fake_session["test_lockbox"]["labels"] = [10 ** 6, 10 ** 6 + 1]

    archive, _ = session_manager._collect_session_data()
    fake_session.clear()
    _, _, warnings = session_manager._restore_session_data(archive)

    assert "test_lockbox" not in fake_session
    assert any("absent from the restored data" in w for w in warnings), warnings


def test_the_manifest_records_that_the_index_travelled(fake_session):
    """The flag is what lets a restore tell an old file from a new one; a save
    that stopped writing it would silently reinstate the refusal path."""
    fake_session["raw_data"] = _gappy_study()
    _, manifest = session_manager._collect_session_data()
    assert manifest[session_manager._INDEX_PRESERVED_FLAG] is True


# ── the datetime variant: three functions that had to be made to agree ────
#
# `_check_row_labels` used `lbl in frame.index`, which COERCES — the string
# '2020-01-01 00:00:00' compares equal into a DatetimeIndex — so a stringified
# datetime label passed the restore check. `train_row_mask` then tested the same
# label in a plain Python set, where it does not match a Timestamp, and 10
# sealed rows became 0 with no warning while the chip went on reporting 10.

def _dated_study(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({"a": rng.normal(size=n),
                         "b": rng.integers(0, 5, n),
                         "y": rng.integers(0, 2, n)},
                        index=pd.date_range("2020-01-01", periods=n))


def _sealed_rows_seen_by_the_mask(lockbox, index) -> int:
    """What `train_row_mask` — the function every page reads — actually seals."""
    import streamlit as st
    from utils.test_lockbox import train_row_mask
    st.session_state.clear()
    try:
        st.session_state["test_lockbox"] = dict(lockbox)
        return int((~train_row_mask(index)).sum())
    finally:
        st.session_state.clear()


class TestContract005DatetimeLabelsRoundTripOrAreRefused:
    """No silent path: either the labels come back as Timestamps, or it warns."""

    def _seal(self, state, df, k=10):
        labels = list(df.index[:k])
        state["raw_data"] = df
        state["test_lockbox"] = {
            "labels": labels, "fraction": 0.2, "seed": 42,
            "n_total": len(df), "n_test": len(labels),
            "signature": "sig", "stratified": False,
        }
        return labels

    def test_a_datetime_seal_still_seals_the_same_rows_after_a_restore(
            self, fake_session):
        df = _dated_study()
        labels = self._seal(fake_session, df)
        assert _sealed_rows_seen_by_the_mask(
            fake_session["test_lockbox"], df.index) == len(labels)

        archive, _ = session_manager._collect_session_data()
        fake_session.clear()
        _, _, warnings = session_manager._restore_session_data(archive)

        assert warnings == [], f"a clean round trip must not warn: {warnings}"
        back = fake_session["raw_data"]
        restored = fake_session["test_lockbox"]["labels"]
        assert all(isinstance(x, pd.Timestamp) for x in restored), (
            "labels came back as text, which no consumer can match")
        assert restored == labels
        assert _sealed_rows_seen_by_the_mask(
            fake_session["test_lockbox"], back.index) == len(labels), (
            "the seal passed the restore check and then sealed nobody")

    def test_the_saved_record_carries_the_label_type(self, fake_session):
        """Reconstruction needs the dtype; without it the restore is guessing."""
        df = _dated_study()
        self._seal(fake_session, df)
        archive, _ = session_manager._collect_session_data()
        with zipfile.ZipFile(io.BytesIO(archive)) as zf:
            saved = json.loads(zf.read("lockbox.json").decode())
        assert "datetime64" in str(saved.get("labels_dtype"))
        assert isinstance(saved["labels"][0], str), (
            "the labels themselves still travel as JSON scalars")

    def test_a_label_that_cannot_be_rebuilt_is_refused_out_loud(
            self, fake_session):
        """The other honest ending. A label the index does not hold must fail
        the check rather than coerce its way past it."""
        df = _dated_study()
        self._seal(fake_session, df)
        archive, _ = session_manager._collect_session_data()
        # Corrupt one label into a date the study does not contain.
        buf = io.BytesIO()
        with zipfile.ZipFile(io.BytesIO(archive)) as src, \
                zipfile.ZipFile(buf, "w") as out:
            for item in src.infolist():
                data = src.read(item.filename)
                if item.filename == "lockbox.json":
                    payload = json.loads(data.decode())
                    payload["labels"][0] = "1999-12-31 00:00:00"
                    data = json.dumps(payload).encode()
                out.writestr(item, data)
        fake_session.clear()
        _, _, warnings = session_manager._restore_session_data(buf.getvalue())
        assert "test_lockbox" not in fake_session
        assert any("absent from the restored data" in w for w in warnings), warnings

    def test_a_cohort_run_over_dates_survives_the_same_way(self, fake_session):
        df = _dated_study()
        fake_session["raw_data"] = df
        group = list(df.index[df["b"] < 2])
        fake_session["cohort_run"] = {
            "column": "b", "value": 0, "label": "low b", "labels": group,
            "n_rows": len(group), "n_total": len(df), "position": 1, "of": 2,
            "order": ["low b", "high b"], "target_col": "y",
            "dropped_features": [],
        }
        archive, _ = session_manager._collect_session_data()
        fake_session.clear()
        _, _, warnings = session_manager._restore_session_data(archive)
        assert warnings == []
        restored = fake_session["cohort_run"]["labels"]
        assert restored == group
        assert (fake_session["raw_data"].loc[restored, "b"] < 2).all()

    def test_an_integer_index_is_unaffected(self, fake_session):
        """The fix must not change the case that already worked."""
        df = _gappy_study()
        labels = _seal(fake_session, df)
        archive, _ = session_manager._collect_session_data()
        fake_session.clear()
        _, _, warnings = session_manager._restore_session_data(archive)
        assert warnings == []
        assert fake_session["test_lockbox"]["labels"] == labels


class TestSweep008TheArchiveCarriesTheWholeSeal:
    """Enumerating the saved fields dropped every field added afterwards."""

    def _sealed_session(self, state):
        rng = np.random.default_rng(4)
        n = 180
        df = pd.DataFrame({"SUBJ": np.repeat(range(60), 3),
                           "age": rng.integers(20, 80, n),
                           "y": rng.integers(0, 2, n)})
        state["raw_data"] = df
        import streamlit as st
        from utils.test_lockbox import ensure_lockbox, record_lockbox_open
        st.session_state.clear()
        try:
            lb = ensure_lockbox(df, "y", "classification")
            record_lockbox_open("Train & Compare")
            record_lockbox_open("Train & Compare")
            record_lockbox_open("Sensitivity Analysis (seed sweep)")
            lb = dict(st.session_state["test_lockbox"])
        finally:
            st.session_state.clear()
        state["test_lockbox"] = lb
        return df, lb

    def test_every_field_the_seal_carries_survives_the_round_trip(
            self, fake_session):
        df, before = self._sealed_session(fake_session)
        archive, _ = session_manager._collect_session_data()
        fake_session.clear()
        _, _, warnings = session_manager._restore_session_data(archive)
        assert warnings == [], warnings
        after = fake_session["test_lockbox"]
        missing = sorted(set(before) - set(after))
        assert not missing, f"the archive dropped {missing} from the seal"

    def test_the_open_count_and_its_timestamps_come_back(self, fake_session):
        df, before = self._sealed_session(fake_session)
        assert before["opened_count"] == 3
        archive, _ = session_manager._collect_session_data()
        fake_session.clear()
        session_manager._restore_session_data(archive)
        after = fake_session["test_lockbox"]
        assert after["opened_count"] == 3, (
            "a restored session forgot three openings and the Methods sentence "
            "went back to 'accessed only for the final evaluation'")
        assert len(after["opened_at"]) == 3

    def test_a_restored_undetermined_seal_renders_its_warning_again(
            self, fake_session, monkeypatch):
        df, before = self._sealed_session(fake_session)
        assert before["seal_basis"] == "undetermined", "fixture must be undetermined"
        archive, _ = session_manager._collect_session_data()
        fake_session.clear()
        session_manager._restore_session_data(archive)
        after = fake_session["test_lockbox"]
        assert after["seal_basis"] == "undetermined"
        assert after["undetermined_because"], "the record lost WHY"

        import streamlit as st
        from utils.session_state import DataConfig
        from utils.test_lockbox import render_lockbox_status
        warnings_seen: list = []
        st.session_state.clear()
        try:
            st.session_state["test_lockbox"] = after
            st.session_state["raw_data"] = df
            st.session_state["data_config"] = DataConfig(
                target_col="y", feature_cols=["age"], task_type="classification")
            monkeypatch.setattr(st, "warning",
                                lambda msg, **kw: warnings_seen.append(str(msg)))
            monkeypatch.setattr(st, "caption", lambda msg, **kw: None)
            render_lockbox_status()
        finally:
            st.session_state.clear()
        assert any("Could not tell whether one person" in w for w in warnings_seen), (
            f"a restored undetermined seal rendered a clean lock: {warnings_seen}")
