"""The seal's subject detection, its status chip, and its open count.

One test per finding in the lockbox-subject cluster of the paper-risk sprint.
Each fails if its fix is reverted:

- `IMPORT-020` / `IMPORT-022` — measured repetition was DISCARDED when the
  column's name was outside `_SUBJECT_ID_TOKENS`, and the seal then recorded
  `cross_sectional`: a positive claim that the study has one row per person,
  asserted over repetition the ranking loop had already counted.
- `IMPORT-257` — nothing could DECLARE the subject column, so the
  declared-entity paths in the seal and in Train & Compare were dead code.
- `IMPORT-207` — a JSON payload could install a non-unique index; the seal
  addresses rows by label, so a 15% chip sat over a 42% holdout.
- `MINE-005` — `train_row_mask` returns all-True with no lockbox while page 04
  asserted that held-out rows were excluded.
- `SWEEP-008` — "opened once at Train & Compare" was printed beside two
  re-runnable train buttons, with nothing counting the openings.
- `IMPORT-209` — one blank ID in a JSON file floated the column and collapsed
  every ID above 2**53 at LOAD time.
- `_lockbox_signature` — content-only hashing let a renumbered frame reuse a
  stale seal whose labels named different rows.
"""
from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd
import pytest
import streamlit as st

from utils.session_state import CohortStructureDetection, DataConfig
from utils.test_lockbox import (SEAL_CROSS_SECTIONAL, SEAL_GROUPED,
                                SEAL_UNDETERMINED, BASIS_DETECTED,
                                BASIS_USER_STATED, _lockbox_signature,
                                ensure_lockbox, get_lockbox,
                                lockbox_absence_reason, lockbox_open_count,
                                quarantine_is_active, rank_grouping_candidates,
                                record_lockbox_open, render_lockbox_status,
                                train_row_mask)

PAGE_01 = "pages/01_Upload_and_Audit.py"
PAGE_04 = "pages/04_Feature_Selection.py"
PAGE_06 = "pages/06_Train_and_Compare.py"


@pytest.fixture(autouse=True)
def clean():
    st.session_state.clear()
    yield
    st.session_state.clear()


def cohort(ids, key="SUBJ", seed=0):
    """A repeated-measures frame keyed by `key`."""
    ids = list(ids)
    rng = np.random.default_rng(seed)
    return pd.DataFrame({key: ids,
                         "age": rng.integers(20, 80, len(ids)),
                         "y": rng.integers(0, 2, len(ids))})


def leak(df, lb, key):
    """How many values of `key` sit on BOTH sides of the seal."""
    sealed = set(lb["labels"])
    held = set(df.loc[df.index.isin(sealed), key])
    trained = set(df.loc[~df.index.isin(sealed), key])
    return len(held & trained)


def source(path):
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


class _Captured:
    """Collects what the chip renders, in bare mode."""

    def __init__(self, monkeypatch):
        self.warnings = []
        self.captions = []
        self.infos = []
        monkeypatch.setattr(st, "warning",
                            lambda msg, **kw: self.warnings.append(str(msg)))
        monkeypatch.setattr(st, "caption",
                            lambda msg, **kw: self.captions.append(str(msg)))
        monkeypatch.setattr(st, "info",
                            lambda msg, **kw: self.infos.append(str(msg)))

    @property
    def text(self):
        return " ".join(self.warnings + self.captions + self.infos)


# ── IMPORT-020 ───────────────────────────────────────────────────────────

class TestImport020MeasuredRepetitionIsNeverDiscarded:
    def test_an_unrecognized_name_reaches_unclear_instead_of_being_dropped(self):
        """The repetition is counted, so it must reach the record.

        `rank_grouping_candidates` measured k and n and then `continue`d on the
        name alone. The count is the evidence; throwing it away is what let the
        seal claim `cross_sectional`.
        """
        df = cohort(np.repeat(range(60), 3), key="ptno")
        assert rank_grouping_candidates(df) == [], (
            "shape alone must not promote a column to the group column — "
            "which column identifies a person is the user's to state")
        unclear = st.session_state.get("_lockbox_repetition_unclear")
        assert unclear, "the measured repetition was discarded again"
        assert unclear[0]["column"] == "ptno"
        assert unclear[0]["reason"] == "unrecognized_name"
        assert unclear[0]["n_groups"] == 60 and unclear[0]["n_rows"] == 180

    def test_the_seal_never_asserts_cross_sectional_over_repetition(self):
        df = cohort(np.repeat(range(60), 3), key="ptno")
        lb = ensure_lockbox(df, "y", "classification")
        assert lb["seal_basis"] == SEAL_UNDETERMINED
        assert lb["seal_basis"] != SEAL_CROSS_SECTIONAL
        assert lb["undetermined_because"][0]["column"] == "ptno"

    def test_a_genuinely_cross_sectional_study_is_still_a_clean_lock(self):
        """The disclosure must cost something to be worth anything.

        A detector that calls every dataset `undetermined` is a detector nobody
        reads. One row per person, ordinary covariates: no repetition, no
        warning.
        """
        rng = np.random.default_rng(3)
        n = 400
        df = pd.DataFrame({"age": rng.integers(20, 80, n),
                           "sex": rng.integers(0, 2, n),
                           "bmi": rng.normal(27, 4, n),
                           "y": rng.integers(0, 2, n)})
        lb = ensure_lockbox(df, "y", "classification")
        assert lb["seal_basis"] == SEAL_CROSS_SECTIONAL
        assert st.session_state.get("_lockbox_repetition_unclear") is None


# ── IMPORT-022 ───────────────────────────────────────────────────────────

class TestImport022AnUnrecognizedIdIsNotSealedAsCrossSectional:
    def test_subj_is_not_recorded_as_a_cross_sectional_study(self):
        df = cohort(np.repeat(range(60), 3), key="SUBJ")
        lb = ensure_lockbox(df, "y", "classification")
        assert leak(df, lb, "SUBJ") > 0, "the fixture no longer leaks"
        assert lb["seal_basis"] == SEAL_UNDETERMINED, (
            "a leak may be disclosed; it may not be sealed as a study with one "
            "row per person")

    def test_the_chip_names_the_column_and_the_control(self, monkeypatch):
        df = cohort(np.repeat(range(60), 3), key="SUBJ")
        st.session_state["data_config"] = DataConfig(target_col="y",
                                                     feature_cols=["age"],
                                                     task_type="classification")
        ensure_lockbox(df, "y", "classification")
        cap = _Captured(monkeypatch)
        render_lockbox_status()
        assert any("`SUBJ`" in w for w in cap.warnings), cap.warnings
        assert "Upload & Audit" in cap.text

    def test_declaring_it_seals_by_subject_and_leaks_nobody(self):
        df = cohort(np.repeat(range(60), 3), key="SUBJ")
        st.session_state["cohort_structure_detection"] = CohortStructureDetection(
            entity_id_override_enabled=True, entity_id_override_value="SUBJ")
        lb = ensure_lockbox(df, "y", "classification")
        assert lb["seal_basis"] == SEAL_GROUPED
        assert lb["group_col"] == "SUBJ"
        assert leak(df, lb, "SUBJ") == 0


# ── IMPORT-257 ───────────────────────────────────────────────────────────

class TestImport257TheSubjectColumnCanBeDeclared:
    def test_the_page_renders_a_control_that_writes_the_declaration(self):
        """The dead code was a missing widget, so the page is the evidence."""
        text = source(PAGE_01)
        assert "subject_id_declaration" in text
        assert "entity_id_override_enabled" in text
        assert "entity_id_override_value" in text

    def test_a_declaration_beats_the_name_heuristic(self):
        """`site` is a cluster the heuristic would rank below `SEQN`."""
        ids = np.repeat(range(40), 3)
        df = cohort(ids, key="SEQN")
        df["site"] = np.repeat(range(10), 12)
        assert rank_grouping_candidates(df)[0]["column"] == "SEQN"
        st.session_state["cohort_structure_detection"] = CohortStructureDetection(
            entity_id_override_enabled=True, entity_id_override_value="site")
        lb = ensure_lockbox(df, "y", "classification")
        assert lb["group_col"] == "site"
        assert lb["basis_source"] == BASIS_USER_STATED

    def test_declaring_one_row_per_participant_stops_the_heuristic(self):
        df = cohort(np.repeat(range(60), 3), key="SEQN")
        st.session_state["cohort_structure_detection"] = CohortStructureDetection(
            entity_id_override_enabled=True, entity_id_override_value=None)
        lb = ensure_lockbox(df, "y", "classification")
        assert lb["group_col"] is None
        assert lb["seal_basis"] == SEAL_CROSS_SECTIONAL
        assert lb["basis_source"] == BASIS_USER_STATED, (
            "an answer and a guess must not be recorded as the same thing")

    def test_an_undeclared_seal_still_says_it_was_detected(self):
        df = cohort(np.repeat(range(60), 3), key="SEQN")
        lb = ensure_lockbox(df, "y", "classification")
        assert lb["basis_source"] == BASIS_DETECTED

    def test_the_declaration_reaches_the_train_val_split(self):
        """Train & Compare reads `entity_id_final` — the same record."""
        cohort_rec = CohortStructureDetection(
            entity_id_override_enabled=True, entity_id_override_value="SUBJ")
        assert cohort_rec.entity_id_final == "SUBJ"
        assert "entity_id_final" in source(PAGE_06)


# ── IMPORT-207 ───────────────────────────────────────────────────────────

class TestImport207ANonUniqueIndexIsNeverInstalled:
    def test_the_loader_keeps_duplicate_labels_as_a_column(self):
        from data_processor import load_json
        frame = pd.DataFrame(
            {"v": range(120), "y": np.random.default_rng(1).normal(size=120)},
            index=np.repeat([f"S{i:03d}" for i in range(40)], 3))
        loaded = load_json(io.BytesIO(frame.to_json(orient="split").encode()))
        assert loaded.index.is_unique, (
            "a repeated label cannot name one row, and every downstream "
            "membership test is by label")
        assert len(loaded) == 120
        assert "index" in loaded.columns
        assert loaded["index"].tolist()[:3] == ["S000", "S000", "S000"]

    def test_a_unique_index_is_still_installed(self):
        from data_processor import load_json
        frame = pd.DataFrame({"v": [1, 2, 3]}, index=["a", "b", "c"])
        loaded = load_json(io.BytesIO(frame.to_json(orient="split").encode()))
        assert loaded.index.tolist() == ["a", "b", "c"]
        assert "index" not in loaded.columns

    def test_the_seal_refuses_duplicate_row_labels_and_says_why(self):
        rng = np.random.default_rng(2)
        df = pd.DataFrame({"v": range(120), "y": rng.integers(0, 2, 120)},
                          index=np.repeat([f"S{i:03d}" for i in range(40)], 3))
        assert ensure_lockbox(df, "y", "classification") is None
        why = lockbox_absence_reason()
        assert why["reason"] == "duplicate_row_labels"
        assert why["n_duplicated"] == 40

    def test_the_refusal_is_rendered_not_silent(self, monkeypatch):
        rng = np.random.default_rng(2)
        df = pd.DataFrame({"v": range(120), "y": rng.integers(0, 2, 120)},
                          index=np.repeat([f"S{i:03d}" for i in range(40)], 3))
        st.session_state["data_config"] = DataConfig(target_col="y",
                                                     feature_cols=["v"],
                                                     task_type="classification")
        ensure_lockbox(df, "y", "classification")
        cap = _Captured(monkeypatch)
        render_lockbox_status()
        assert cap.warnings, "absence rendered nothing at all"
        assert "repeated row label" in cap.warnings[0]


# ── MINE-005 ─────────────────────────────────────────────────────────────

class TestMine005AbsenceIsARenderedState:
    def test_no_lockbox_means_the_quarantine_is_not_active(self):
        idx = pd.RangeIndex(50)
        assert get_lockbox() is None
        assert bool(train_row_mask(idx).all()), "the mask excludes nothing"
        assert quarantine_is_active() is False, (
            "the mask excluded nothing, so no caption may claim an exclusion")

    def test_the_chip_renders_the_absence_once_a_target_exists(self, monkeypatch):
        st.session_state["data_config"] = DataConfig(target_col="y",
                                                     feature_cols=["age"],
                                                     task_type="classification")
        cap = _Captured(monkeypatch)
        render_lockbox_status()
        assert cap.warnings, "a missing lockbox rendered nothing"
        assert "No held-out test set is in force" in cap.warnings[0]

    def test_nothing_is_rendered_before_a_target_is_chosen(self, monkeypatch):
        cap = _Captured(monkeypatch)
        render_lockbox_status()
        assert not cap.warnings and not cap.captions

    def test_page_04_does_not_claim_exclusion_on_exploratory_mode_alone(self):
        text = source(PAGE_04)
        assert "quarantine_is_active()" in text
        claim = "held-out test rows are excluded to prevent selection leakage"
        assert claim in text
        before = text.split(claim)[0]
        guard = before.rfind("if ")
        assert "quarantine_is_active" in before[guard:], (
            "the exclusion claim is guarded by something other than the mask "
            "that would have to have excluded anything")


# ── SWEEP-008 ────────────────────────────────────────────────────────────

class TestSweep008OpeningTheSealedSetIsCounted:
    def _sealed(self):
        rng = np.random.default_rng(5)
        df = pd.DataFrame({"age": rng.integers(20, 80, 200),
                           "y": rng.integers(0, 2, 200)})
        st.session_state["data_config"] = DataConfig(target_col="y",
                                                     feature_cols=["age"],
                                                     task_type="classification")
        return df, ensure_lockbox(df, "y", "classification")

    def test_a_fresh_seal_has_not_been_opened(self):
        _, lb = self._sealed()
        assert lb["opened_count"] == 0
        assert lockbox_open_count() == 0

    def test_each_run_is_counted_with_a_timestamp(self):
        self._sealed()
        record_lockbox_open("Train & Compare")
        record_lockbox_open("Train & Compare")
        assert lockbox_open_count() == 2
        assert len(get_lockbox()["opened_at"]) == 2

    def test_the_chip_stops_saying_opened_once_after_the_second_open(
            self, monkeypatch):
        self._sealed()
        record_lockbox_open("t")
        cap = _Captured(monkeypatch)
        render_lockbox_status()
        assert "opened once" in cap.text
        record_lockbox_open("t")
        cap2 = _Captured(monkeypatch)
        render_lockbox_status()
        assert "opened once" not in cap2.text
        assert "opened 2 times" in cap2.text
        assert any("2 times" in w for w in cap2.warnings), (
            "the count belongs in a warning, not only in a caption")

    def test_train_and_compare_records_the_open_where_metrics_are_computed(self):
        text = source(PAGE_06)
        assert "record_lockbox_open" in text
        for promise in ("opens the held-out test set exactly once",
                        "Test is scored once"):
            assert promise not in text, (
                f"the page may not promise a count nothing enforces: {promise}")

    def test_the_methods_sentence_drops_the_claim_when_the_count_denies_it(self):
        from ml.narrative_engine import NarrativeEngine
        from utils.insight_ledger import Insight, InsightLedger
        from utils.workflow_provenance import WorkflowProvenance

        prov = WorkflowProvenance()
        prov.record_upload(target_col="y", task_type="classification",
                           feature_cols=["age"], n_samples=200)
        ledger = InsightLedger()
        ledger.upsert(Insight(
            id="upload_test_lockbox", source_page="01_Upload_and_Audit",
            category="study_design", severity="info",
            finding="A 15% test set was held out at upload.",
            implication="x", resolved=True,
            resolution_details={"action_type": "test_lockbox",
                                "params": {"fraction": 0.15, "seed": 42,
                                           "n_test": 30}},
        ))
        self._sealed()

        record_lockbox_open("t")
        once = NarrativeEngine(prov, ledger).generate()
        assert "accessed only for the final evaluation" in once.study_design

        record_lockbox_open("t")
        twice = NarrativeEngine(prov, ledger).generate()
        assert "accessed only for the final evaluation" not in twice.study_design
        assert "accessed 2 times" in twice.study_design
        assert "optimistically biased" in twice.study_design


# ── IMPORT-209 ───────────────────────────────────────────────────────────

class TestImport209LargeIdsSurviveABlankCell:
    def test_a_single_null_no_longer_collapses_ids_above_2_53(self):
        from data_processor import load_json
        payload = [{"SEQN": 9007199254740993, "glucose": 95},
                   {"SEQN": 9007199254740992, "glucose": 102},
                   {"SEQN": None, "glucose": 110}]
        df = load_json(io.BytesIO(json.dumps(payload).encode()))
        assert df["SEQN"].dropna().nunique() == 2, (
            "two participants became one before any join could see it")
        assert str(df["SEQN"].dtype) == "Int64"
        assert int(df["SEQN"].iloc[0]) == 9007199254740993

    def test_the_split_orient_carries_the_same_repair(self):
        from data_processor import load_json
        payload = {"columns": ["SEQN", "g"],
                   "data": [[9007199254740993, 1], [9007199254740992, 2],
                            [None, 3]],
                   "index": [0, 1, 2]}
        df = load_json(io.BytesIO(json.dumps(payload).encode()))
        assert df["SEQN"].dropna().nunique() == 2

    def test_ordinary_columns_are_left_alone(self):
        from data_processor import load_json
        payload = [{"a": 1, "b": "x"}, {"a": None, "b": "y"}]
        df = load_json(io.BytesIO(json.dumps(payload).encode()))
        assert str(df["a"].dtype) == "float64", (
            "the repair must fire on the precision hazard, not on every null")


# ── the signature ────────────────────────────────────────────────────────

class TestLockboxSignatureCoversRowIdentity:
    """Identity here is the row LABELS, because the seal stores labels.

    A renumbered frame is a different set of labels and must force a redraw. A
    re-upload of the same file rebuilds the same labels and must NOT, or every
    reload would silently re-partition the study.
    """

    def _df(self, seed=0):
        rng = np.random.default_rng(seed)
        return pd.DataFrame({"age": rng.integers(20, 80, 60),
                             "y": rng.integers(0, 2, 60)})

    def test_a_renumbered_frame_has_a_different_signature(self):
        df = self._df()
        renumbered = df.copy()
        renumbered.index = pd.RangeIndex(1000, 1060)
        assert (_lockbox_signature(df, "y", "classification", 0.15, 42)
                != _lockbox_signature(renumbered, "y", "classification", 0.15, 42))

    def test_an_identical_re_upload_still_matches(self):
        df = self._df()
        again = df.copy()
        assert (_lockbox_signature(df, "y", "classification", 0.15, 42)
                == _lockbox_signature(again, "y", "classification", 0.15, 42))

    def test_a_renumbered_frame_does_not_reuse_the_stale_seal(self):
        df = self._df()
        first = ensure_lockbox(df, "y", "classification")
        renumbered = df.copy()
        renumbered.index = pd.RangeIndex(1000, 1060)
        second = ensure_lockbox(renumbered, "y", "classification")
        assert second["signature"] != first["signature"]
        assert set(second["labels"]) <= set(renumbered.index), (
            "the stale seal's labels named rows this frame does not have")
