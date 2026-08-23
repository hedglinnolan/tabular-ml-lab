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
                                _roster_shape, declaration_contradiction,
                                ensure_lockbox, get_lockbox,
                                lockbox_absence_reason, lockbox_open_count,
                                quarantine_is_active, rank_grouping_candidates,
                                record_lockbox_open, render_lockbox_status,
                                train_row_mask)

PAGE_01 = "pages/01_Upload_and_Audit.py"
PAGE_04 = "pages/04_Feature_Selection.py"
PAGE_06 = "pages/06_Train_and_Compare.py"
PAGE_08 = "pages/08_Sensitivity_Analysis.py"


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
    """Two ways the shape test was blind, and the damping that has to hold.

    The name-blind detector is what makes the token list survivable, so a gap
    IN the detector reinstates the whole finding. Both gaps below were measured
    at HEAD after `IMPORT-022` was first closed.
    """

    def test_a_float_id_column_is_measured_not_skipped(self):
        """ONE blank cell in a CSV floats the column; a dtype gate then skips it.

        This is not a hypothetical dtype: `read_csv` produces it from a single
        empty ID cell, and the repo's own `_tt_tmp_nhanes.csv` loads `SEQN` as
        float64 for exactly that reason.
        """
        df = cohort(np.repeat(np.arange(60, dtype=float), 3), key="SUBJ")
        assert str(df["SUBJ"].dtype) == "float64", "the fixture stopped floating"
        lb = ensure_lockbox(df, "y", "classification")
        assert leak(df, lb, "SUBJ") > 0, "the fixture no longer leaks"
        assert lb["seal_basis"] == SEAL_UNDETERMINED, (
            "a float dtype is not evidence that a column is a measurement — "
            "the VALUES are, and these are whole numbers")

    def test_a_blank_cell_in_a_real_csv_reaches_the_same_verdict(self):
        """End to end through pandas, not through a hand-built frame."""
        csv = "SUBJ,age,y\n" + "".join(
            f"{'' if i == 0 else i // 3},{40 + i % 30},{i % 2}\n" for i in range(180))
        df = pd.read_csv(io.StringIO(csv))
        lb = ensure_lockbox(df, "y", "classification")
        assert lb["seal_basis"] == SEAL_UNDETERMINED
        named = [c["column"] for c in lb["undetermined_because"]]
        assert named[0] == "SUBJ", (
            f"the warning must point at the roster, not at a balanced "
            f"covariate: {named}")

    def test_irregular_follow_up_is_a_roster_too(self):
        """Between one and five visits per subject: regular share about 0.26.

        Requiring most subjects to carry the SAME number of rows rejected the
        commonest longitudinal shape there is, and sealed `cross_sectional`
        over 28 subjects sitting on both sides.
        """
        rng = np.random.default_rng(7)
        counts = rng.integers(1, 6, 80)
        ids = np.repeat(np.arange(80), counts)
        df = cohort(ids, key="ptno")
        shape = _roster_shape(df["ptno"], 80, len(df))
        assert shape and shape["regular_share"] < 0.5, (
            "the fixture is no longer irregular")
        lb = ensure_lockbox(df, "y", "classification")
        assert leak(df, lb, "ptno") > 0, "the fixture no longer leaks"
        assert lb["seal_basis"] == SEAL_UNDETERMINED

    def test_two_rows_per_subject_is_a_roster(self):
        df = cohort(np.repeat(range(100), 2), key="ptno")
        lb = ensure_lockbox(df, "y", "classification")
        assert lb["seal_basis"] == SEAL_UNDETERMINED

    # ── and the damping, which is what makes the disclosure worth reading ──

    @pytest.mark.parametrize("name,values", [
        # 10 exam cycles across 400 rows: perfectly regular, and not a roster.
        ("exam_year", np.repeat(np.arange(2010, 2020), 40)),
        # a dose battery tiled down the file
        ("dose_mg", np.tile([0, 5, 10, 20, 40, 80, 120, 160], 50)),
        # a questionnaire item battery, same tiling
        ("q_item", np.tile(np.arange(1, 21), 20)),
    ])
    def test_a_balanced_covariate_does_not_fire_undetermined(self, name, values):
        """What separates these from a roster is CARDINALITY relative to rows.

        Each of them is more regular than any real follow-up schedule, so
        regularity alone called them rosters. A participant list grows with the
        people; `exam_year` has ten levels whether the study has 400 rows or
        40,000.
        """
        rng = np.random.default_rng(1)
        n = len(values)
        df = pd.DataFrame({name: values,
                           "age": rng.integers(20, 80, n),
                           "y": rng.integers(0, 2, n)})
        lb = ensure_lockbox(df, "y", "classification")
        assert lb["seal_basis"] == SEAL_CROSS_SECTIONAL, (
            f"`{name}` was read as a list of people")
        assert st.session_state.get("_lockbox_repetition_unclear") is None

    def test_a_whole_battery_of_covariates_together_is_still_a_clean_lock(self):
        rng = np.random.default_rng(0)
        n = 400
        df = pd.DataFrame({"exam_year": np.repeat(np.arange(2010, 2020), 40),
                           "dose_mg": np.tile([0, 5, 10, 20, 40, 80, 120, 160], 50),
                           "q_item": np.tile(np.arange(1, 21), 20),
                           "age": rng.integers(20, 80, n),
                           "sbp": rng.integers(90, 180, n),
                           "y": rng.integers(0, 2, n)})
        lb = ensure_lockbox(df, "y", "classification")
        assert lb["seal_basis"] == SEAL_CROSS_SECTIONAL
        assert lb["undetermined_because"] is None

    def test_a_fractional_measurement_is_never_a_roster(self):
        """Cast and check, not dtype-gate — but a real measurement still fails."""
        rng = np.random.default_rng(2)
        n = 200
        df = pd.DataFrame({"bmi": np.round(rng.normal(27, 4, n), 1),
                           "y": rng.integers(0, 2, n)})
        assert _roster_shape(df["bmi"], int(df["bmi"].nunique()), n) is None

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
        """Over a frame with NO repetition, the answer stands and is recorded."""
        df = cohort(range(200), key="SEQN")
        st.session_state["cohort_structure_detection"] = CohortStructureDetection(
            entity_id_override_enabled=True, entity_id_override_value=None)
        lb = ensure_lockbox(df, "y", "classification")
        assert lb["group_col"] is None
        assert lb["seal_basis"] == SEAL_CROSS_SECTIONAL
        assert lb["basis_source"] == BASIS_USER_STATED, (
            "an answer and a guess must not be recorded as the same thing")
        assert lb["contradiction"] is None

    # ── the declaration is not above the measurement ──────────────────────

    def test_declaring_one_row_per_participant_over_repetition_contradicts(self):
        """A stated answer beats a name list. It does not beat a count.

        `SEQN` holds 60 values across 180 rows — measured, not guessed — and the
        declaration used to win outright: the evidence was popped from session
        state unread and the seal recorded a clean `cross_sectional` with 24
        subjects on both sides of it and no warning anywhere.
        """
        df = cohort(np.repeat(range(60), 3), key="SEQN")
        st.session_state["cohort_structure_detection"] = CohortStructureDetection(
            entity_id_override_enabled=True, entity_id_override_value=None)
        lb = ensure_lockbox(df, "y", "classification")
        assert leak(df, lb, "SEQN") > 0, "the fixture no longer leaks"
        assert lb["seal_basis"] != SEAL_CROSS_SECTIONAL, (
            "a declaration the data contradicts may not produce a clean seal")
        assert lb["seal_basis"] == SEAL_UNDETERMINED

    def test_the_record_carries_both_the_declaration_and_the_measurement(self):
        df = cohort(np.repeat(range(60), 3), key="SEQN")
        st.session_state["cohort_structure_detection"] = CohortStructureDetection(
            entity_id_override_enabled=True, entity_id_override_value=None)
        lb = ensure_lockbox(df, "y", "classification")
        contra = lb["contradiction"]
        assert contra, "the seal did not record the disagreement"
        assert contra["kind"] == "stated_unique_but_data_repeats"
        assert lb["basis_source"] == BASIS_USER_STATED, (
            "the declaration is still what was said — it is the CONCLUSION "
            "that has to change")
        evidence = contra["evidence"][0]
        assert evidence["column"] == "SEQN"
        assert evidence["n_groups"] == 60 and evidence["n_rows"] == 180

    def test_the_contradiction_is_rendered_loudly(self, monkeypatch):
        df = cohort(np.repeat(range(60), 3), key="SEQN")
        st.session_state["data_config"] = DataConfig(
            target_col="y", feature_cols=["age"], task_type="classification")
        st.session_state["cohort_structure_detection"] = CohortStructureDetection(
            entity_id_override_enabled=True, entity_id_override_value=None)
        ensure_lockbox(df, "y", "classification")
        cap = _Captured(monkeypatch)
        render_lockbox_status()
        assert any("disagree" in w for w in cap.warnings), cap.warnings
        assert any("`SEQN`" in w for w in cap.warnings)

    def test_a_cluster_column_does_not_contradict_the_declaration(self):
        """A site repeating says nothing about whether a PERSON repeats.

        People are nested inside recruitment sites by design, so contradicting
        the user with a site would be escalating on the size of a consequence
        rather than on evidence of an error.
        """
        rng = np.random.default_rng(0)
        n = 200
        df = pd.DataFrame({"site_id": np.repeat(range(10), 20),
                           "age": rng.integers(20, 80, n),
                           "y": rng.integers(0, 2, n)})
        assert declaration_contradiction(df, None) is None

    def test_declaring_a_column_that_is_unique_per_row_contradicts_too(self):
        """The mirror case: grouping by it holds out one row per group."""
        df = cohort(range(200), key="row_key")
        contra = declaration_contradiction(df, "row_key")
        assert contra and contra["kind"] == "stated_repeats_but_column_is_unique"

    # ── a declared column that vanished ───────────────────────────────────

    def test_a_vanished_declared_column_is_not_a_cross_sectional_claim(self):
        """`user_stated cross_sectional` is a positive claim nobody made.

        The user named a subject column. Feature engineering dropped it. The
        seal then recorded that the study has one row per person, sourced to
        the user — over a frame the user had said was grouped.
        """
        df = cohort(np.repeat(range(60), 3), key="SUBJ")
        st.session_state["cohort_structure_detection"] = CohortStructureDetection(
            entity_id_override_enabled=True, entity_id_override_value="SUBJ")
        first = ensure_lockbox(df, "y", "classification")
        assert first["seal_basis"] == SEAL_GROUPED

        gone = df.drop(columns=["SUBJ"])
        lb = ensure_lockbox(gone, "y", "classification")
        assert not (lb["seal_basis"] == SEAL_CROSS_SECTIONAL
                    and lb["basis_source"] == BASIS_USER_STATED), (
            "a vanished column was turned into a statement about the study")
        assert lb["seal_basis"] == SEAL_UNDETERMINED
        assert lb["basis_source"] == BASIS_DETECTED

    def test_the_vanished_column_is_named_on_the_seal_record(self):
        """Persisted, so a restored session renders the same disclosure."""
        df = cohort(np.repeat(range(60), 3), key="SUBJ")
        st.session_state["cohort_structure_detection"] = CohortStructureDetection(
            entity_id_override_enabled=True, entity_id_override_value="SUBJ")
        ensure_lockbox(df, "y", "classification")
        lb = ensure_lockbox(df.drop(columns=["SUBJ"]), "y", "classification")
        assert lb["declared_column_missing"] == "SUBJ"
        assert any(c.get("reason") == "declared_column_missing"
                   for c in lb["undetermined_because"])

    def test_the_vanished_column_renders_from_the_record_alone(self, monkeypatch):
        """Session state may be gone; the record is what a restore has."""
        st.session_state["data_config"] = DataConfig(
            target_col="y", feature_cols=["age"], task_type="classification")
        st.session_state["test_lockbox"] = {
            "labels": [1, 2, 3], "fraction": 0.15, "n_test": 3, "seed": 42,
            "signature": "s", "seal_basis": SEAL_UNDETERMINED,
            "basis_source": BASIS_DETECTED, "declared_column_missing": "SUBJ",
            "undetermined_because": [{"column": "SUBJ",
                                      "reason": "declared_column_missing"}],
        }
        cap = _Captured(monkeypatch)
        render_lockbox_status()
        assert any("`SUBJ`" in w and "not in the data any more" in w
                   for w in cap.warnings), cap.warnings

    def test_answering_after_the_seal_was_drawn_updates_the_record(self):
        """The declaration is not in the signature, and must still be recorded.

        Putting it there would redraw the holdout for a change that moves no
        rows, invalidating every downstream result. So the still-valid seal has
        its BASIS refreshed instead — otherwise a user who answered the grain
        question after upload kept a seal describing itself as `detected
        cross_sectional` over a contradiction measured since.
        """
        df = cohort(np.repeat(range(60), 3), key="SEQN")
        first = ensure_lockbox(df, "y", "classification")
        assert first["seal_basis"] == SEAL_GROUPED

        # Now answer "each row is a different participant" over the same frame.
        st.session_state["cohort_structure_detection"] = CohortStructureDetection(
            entity_id_override_enabled=True, entity_id_override_value=None)
        second = ensure_lockbox(df, "y", "classification")
        assert second["basis_source"] == BASIS_USER_STATED
        assert second["contradiction"], "the answer arrived and nothing recorded it"

    def test_withdrawing_a_contradicted_answer_clears_the_record(self):
        """It must narrow AND widen: a stale `undetermined` is a claim too."""
        rng = np.random.default_rng(3)
        n = 400
        df = pd.DataFrame({"age": rng.integers(20, 80, n),
                           "sex": rng.integers(0, 2, n),
                           "y": rng.integers(0, 2, n)})
        st.session_state["cohort_structure_detection"] = CohortStructureDetection(
            entity_id_override_enabled=True, entity_id_override_value="missing_col")
        first = ensure_lockbox(df, "y", "classification")
        assert first["seal_basis"] == SEAL_UNDETERMINED

        st.session_state["cohort_structure_detection"] = CohortStructureDetection(
            entity_id_override_enabled=False, entity_id_override_value=None)
        second = ensure_lockbox(df, "y", "classification")
        assert second["seal_basis"] == SEAL_CROSS_SECTIONAL
        assert second["declared_column_missing"] is None
        assert second["undetermined_because"] is None

    def test_the_page_keeps_a_vanished_declaration_instead_of_reverting(self):
        """pages/01 reset the widget AND the answer; only the widget may reset."""
        text = source(PAGE_01)
        assert "_subject_declaration_vanished" in text
        assert "_clear_vanished_declaration" in text

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

    # ── the in-session counting semantics, previously only string-grepped ──

    def _run_page_06_scoring_block(self, n_models: int, raise_at: int = -1) -> int:
        """The shape of page 06's scoring loop: one open per RUN, at the end.

        Page 06 evaluates every trained model in one run and records ONE open,
        because the sealed rows were consulted once. Re-running the button is a
        second consultation and counts again. A run that raises before any
        held-out metric exists consulted nothing and must not count.
        """
        for i in range(n_models):
            if i == raise_at:
                raise RuntimeError("fit failed before scoring")
        record_lockbox_open("Train & Compare")
        return lockbox_open_count()

    def test_a_multi_model_run_counts_once_not_once_per_model(self):
        self._sealed()
        assert self._run_page_06_scoring_block(n_models=4) == 1, (
            "four models scored against ONE consultation of the sealed rows")

    def test_retraining_increments_the_count(self):
        self._sealed()
        self._run_page_06_scoring_block(n_models=4)
        self._run_page_06_scoring_block(n_models=4)
        assert lockbox_open_count() == 2
        assert len(get_lockbox()["opened_at"]) == 2

    def test_a_failure_before_scoring_does_not_count(self):
        self._sealed()
        with pytest.raises(RuntimeError):
            self._run_page_06_scoring_block(n_models=4, raise_at=2)
        assert lockbox_open_count() == 0, (
            "nothing was scored, so nothing was opened")

    def test_the_count_is_recorded_after_the_metrics_exist_not_before(self):
        """Ordering is the whole reason the failure case is honest.

        `record_lockbox_open` must sit AFTER the held-out evaluation in page 06,
        not before it — a call at the top of the handler counts a run that may
        never produce a number.
        """
        text = source(PAGE_06)
        call = text.index("record_lockbox_open(")
        block_start = text.rindex("if st.button", 0, call)
        assert "test" in text[block_start:call].lower()
        # And ONE per run across every model in it: page 06 loops the trained
        # models inside a single handler, and a call without this guard would
        # count four consultations where the sealed rows were read once.
        assert "_opened_this_run" in text[block_start:call], (
            "the open is counted per model rather than per run")
        assert text.count("record_lockbox_open(") == 1, (
            "a second call site on this page would double-count one run")

    def test_each_open_records_where_it_happened(self):
        """The chip may not assert Train & Compare over an open somewhere else."""
        self._sealed()
        record_lockbox_open("Sensitivity Analysis (seed sweep)")
        entry = get_lockbox()["opened_at"][0]
        assert "Sensitivity Analysis" in entry

    def test_the_chip_names_the_page_that_opened_it(self, monkeypatch):
        self._sealed()
        record_lockbox_open("Sensitivity Analysis (seed sweep)")
        cap = _Captured(monkeypatch)
        render_lockbox_status()
        assert "Sensitivity Analysis" in cap.text
        assert "opened once, at Train & Compare" not in cap.text, (
            "the chip named a page the seal was never opened on")

    # ── pages/08 pools the sealed rows back in ────────────────────────────

    def test_the_seed_sweep_counts_the_rows_it_pools_back_in(self):
        """Page 08 re-splits the SEALED rows and retrains over them.

        Disclosure alone left the count at 1 while the models had been fit on
        the held-out people, so the Methods sentence went on saying the set was
        accessed only for the final evaluation.
        """
        text = source(PAGE_08)
        assert "record_lockbox_open" in text, (
            "the seed sweep trains on the sealed rows without counting it")
        pool = text.index("Pool the stored splits back together")
        call = text.index("_record_open(", pool)
        assert call - pool < 4000, (
            "the count must sit with the pooling it describes")

    def test_the_seed_sweep_also_says_so_on_the_page(self):
        text = source(PAGE_08)
        assert "pooled the **sealed test rows** back in" in text

    def test_the_seed_summary_header_reports_achieved_seeds(self):
        """`len(seed_list)` is what was ASKED for.

        The methodology log already carried the achieved count (`MINE-030`);
        the header above the table it describes still printed the request.
        """
        text = source(PAGE_08)
        header = text.index("Across-seed ")
        window = text[header - 900:header + 300]
        assert "_seeds_phrase" in window
        assert "{len(seed_list)} seeds, fresh split each" not in window

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

    def test_a_permuted_frame_is_the_same_identity(self):
        """The docstring promised order-insensitivity; the hash did not deliver.

        `hash_pandas_object` hashes each label with its POSITION by default, so
        a sorted frame — by date, by group, by anything — hashed differently
        from the frame the seal was drawn on and silently redrew the split over
        the same people, resetting every downstream result.
        """
        df = self._df()
        permuted = df.sample(frac=1, random_state=1)
        assert (_lockbox_signature(df, "y", "classification", 0.15, 42)
                == _lockbox_signature(permuted, "y", "classification", 0.15, 42))

    def test_a_permuted_frame_does_not_redraw_the_seal(self):
        df = self._df()
        first = ensure_lockbox(df, "y", "classification")
        again = ensure_lockbox(df.sample(frac=1, random_state=1),
                               "y", "classification")
        assert again["signature"] == first["signature"]
        assert set(again["labels"]) == set(first["labels"])
        assert not st.session_state.get("_lockbox_redrawn"), (
            "a re-ordering of the same rows re-partitioned the study")

    def test_relabeled_content_is_a_different_identity(self):
        """The other half of the contract: content is hashed WITH its label.

        Hashing content alone made re-labeling invisible, so a filtered frame
        renumbered back to 0..n-1 matched the original signature and the stale
        seal's labels named different people.
        """
        df = self._df()
        filtered = df.iloc[10:]
        assert (_lockbox_signature(filtered, "y", "classification", 0.15, 42)
                != _lockbox_signature(filtered.reset_index(drop=True),
                                      "y", "classification", 0.15, 42))
        # The strict case: the SAME label multiset over the SAME rows, paired
        # differently. Label identity cannot see this and a content-only hash
        # cannot either, so only hashing the pair catches it — and the seal
        # names labels, so a re-pairing means the sealed labels now hold other
        # people's data.
        reassigned = df.sample(frac=1, random_state=1).set_index(df.index)
        assert list(reassigned.index) == list(df.index)
        assert (_lockbox_signature(df, "y", "classification", 0.15, 42)
                != _lockbox_signature(reassigned, "y", "classification", 0.15, 42))

    def test_an_index_identity_is_order_insensitive_on_its_own(self):
        from utils.test_lockbox import _index_identity
        df = self._df()
        assert (_index_identity(df)
                == _index_identity(df.sample(frac=1, random_state=3)))
        renumbered = df.copy()
        renumbered.index = pd.RangeIndex(1000, 1060)
        assert _index_identity(df) != _index_identity(renumbered)

    def test_a_renumbered_frame_does_not_reuse_the_stale_seal(self):
        df = self._df()
        first = ensure_lockbox(df, "y", "classification")
        renumbered = df.copy()
        renumbered.index = pd.RangeIndex(1000, 1060)
        second = ensure_lockbox(renumbered, "y", "classification")
        assert second["signature"] != first["signature"]
        assert set(second["labels"]) <= set(renumbered.index), (
            "the stale seal's labels named rows this frame does not have")


class TestARefusedFrameDoesNotWearAnotherFramesSeal:
    """IMPORT-207 adjacent, found by the adversarial pass: `_cannot_seal`
    returned whatever seal was stored, so a frame refused for duplicate
    labels rendered the PREVIOUS frame's chip — its n_test over rows it
    never sealed. A refusal over a changed frame now retires the stale
    seal (with the redraw's downstream invalidation) and records why.
    """

    def _sealed_frame(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"y": rng.normal(size=60), "x": range(60)})
        lb = ensure_lockbox(df, "y", "regression")
        assert lb is not None
        return df, lb

    def test_a_duplicate_label_frame_retires_the_previous_seal(self):
        st.session_state.clear()
        self._sealed_frame()
        dup = pd.DataFrame(
            {"y": np.random.default_rng(1).normal(size=60), "x": range(60)},
            index=[i // 2 for i in range(60)])
        out = ensure_lockbox(dup, "y", "regression")
        assert out is None
        record = st.session_state.get("_lockbox_not_sealed", {})
        assert record.get("reason") == "duplicate_row_labels"
        assert record.get("previous_seal_retired") is True
        assert st.session_state.get("test_lockbox") is None

    def test_a_transient_no_frame_call_keeps_the_seal(self):
        st.session_state.clear()
        _, lb = self._sealed_frame()
        assert ensure_lockbox(None, "y", "regression") == lb
        assert st.session_state.get("test_lockbox") == lb

    def test_a_refusal_over_the_same_frame_keeps_the_seal(self):
        # Too-few-rows refusal on the SAME sealed frame (labels all present,
        # no duplicates) is not a frame change; the standing seal stays.
        st.session_state.clear()
        df, lb = self._sealed_frame()
        sparse = df.copy()
        sparse.loc[sparse.index[5:], "y"] = np.nan
        out = ensure_lockbox(sparse, "y", "regression")
        assert out == lb
