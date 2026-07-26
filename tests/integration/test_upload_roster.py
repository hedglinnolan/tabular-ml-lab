"""Step 1 as a place to assemble a study, not just drop one file.

Someone bringing demographics + labs + diet asks the same question after every
upload: "what do I have so far, and is it the right file?" These tests cover
the roster that answers it — visible without a click, renameable, removable —
and the state invalidation that keeps a stale working table from outliving the
files it was built from.
"""
from __future__ import annotations

import os
import sys

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from utils.session_projects import SessionProjectManager  # noqa: E402

PAGE = os.path.join(PROJECT_ROOT, "pages", "01_Upload_and_Audit.py")


def _frames():
    return {
        "demographics": pd.DataFrame({"SEQN": range(1000, 1200),
                                      "age": range(200)}),
        "labs": pd.DataFrame({"SEQN": range(1000, 1200),
                              "glucose": [90.0 + i * 0.1 for i in range(200)]}),
        "diet": pd.DataFrame({"SEQN": range(1000, 1200),
                              "kcal": range(1500, 1700)}),
    }


def _app(frames) -> AppTest:
    at = AppTest.from_file(PAGE, default_timeout=180)
    datasets, registry = {}, {}
    for i, (name, df) in enumerate(frames.items(), start=1):
        datasets[i] = {
            "id": i, "project_id": 1, "name": name, "filename": f"{name}.csv",
            "file_type": "csv", "shape_rows": df.shape[0], "shape_cols": df.shape[1],
            "columns": list(df.columns), "column_types": None,
            "upload_timestamp": f"2026-01-0{i}T00:00:00+00:00", "is_transposed": False,
        }
        registry[i] = df
    at.session_state["sp_projects"] = {1: {
        "id": 1, "name": "t", "description": "", "active": True,
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
        "datasets": datasets, "merge_configs": {},
    }}
    at.session_state["sp_counter_project"] = 1
    at.session_state["sp_counter_dataset"] = len(datasets)
    at.session_state["datasets_registry"] = registry
    return at


def _text(at) -> str:
    return " ".join([m.value for m in at.markdown] + [c.value for c in at.caption]
                    + [i.value for i in at.info] + [w.value for w in at.warning]
                    + [e.value for e in at.error])


# ── the roster ───────────────────────────────────────────────────────────

def test_every_uploaded_file_is_listed_without_a_click():
    at = _app(_frames()).run()
    assert not at.exception
    text = _text(at)
    for name in ("demographics", "labs", "diet"):
        assert name in text, f"{name} should be visible in the roster"


def test_roster_states_shape_of_each_file():
    at = _app(_frames()).run()
    text = _text(at)
    assert "200 rows" in text
    assert "In this project (3)" in text


def test_step_one_copy_invites_multiple_files():
    at = _app(_frames()).run()
    text = _text(at).lower()
    # The old copy told a multi-file researcher they were off the happy path.
    assert "most analyses should begin with a single dataset" not in text


def test_each_file_has_a_remove_control():
    at = _app(_frames()).run()
    assert len([b for b in at.button if b.label == "Remove"]) == 3


def test_removing_a_file_drops_it_from_the_project():
    at = _app(_frames()).run()
    [b for b in at.button if b.label == "Remove"][0].click().run()
    assert not at.exception
    remaining = at.session_state["sp_projects"][1]["datasets"]
    assert len(remaining) == 2


def test_removing_a_file_invalidates_the_working_table():
    at = _app(_frames())
    at.run()
    at.session_state["working_table"] = pd.DataFrame({"stale": [1, 2, 3]})
    at.session_state["_combine_signature"] = "demographics|diet|labs|3"
    [b for b in at.button if b.label == "Remove"][0].click().run()
    # A table combined from three files must not survive the loss of one.
    still_there = ("working_table" in at.session_state
                   and at.session_state["working_table"] is not None
                   and "stale" in at.session_state["working_table"].columns)
    assert not still_there


def test_rename_field_is_prefilled_with_current_name():
    at = _app(_frames()).run()
    values = [ti.value for ti in at.text_input]
    for name in ("demographics", "labs", "diet"):
        assert name in values


# ── rename, at the manager level ─────────────────────────────────────────

class TestRenameDataset:
    def _pm(self):
        import streamlit as st
        st.session_state.clear()
        pm = SessionProjectManager()
        pid = pm.create_project("p")
        d1 = pm.add_dataset(pid, "raw_export_v3", "a.csv", "csv", 10, 2, ["a", "b"])
        d2 = pm.add_dataset(pid, "labs", "b.csv", "csv", 10, 2, ["a", "c"])
        return pm, pid, d1, d2

    def test_rename_succeeds(self):
        pm, pid, d1, _ = self._pm()
        assert pm.rename_dataset(d1, "demographics") is True
        assert pm.get_dataset(d1)["name"] == "demographics"

    def test_rename_to_taken_name_is_refused(self):
        pm, pid, d1, _ = self._pm()
        assert pm.rename_dataset(d1, "labs") is False
        assert pm.get_dataset(d1)["name"] == "raw_export_v3"

    def test_rename_to_its_own_name_is_allowed(self):
        pm, pid, d1, _ = self._pm()
        assert pm.rename_dataset(d1, "raw_export_v3") is True

    def test_blank_rename_is_refused(self):
        pm, pid, d1, _ = self._pm()
        assert pm.rename_dataset(d1, "   ") is False
        assert pm.get_dataset(d1)["name"] == "raw_export_v3"

    def test_rename_trims_whitespace(self):
        pm, pid, d1, _ = self._pm()
        pm.rename_dataset(d1, "  demographics  ")
        assert pm.get_dataset(d1)["name"] == "demographics"

    def test_rename_unknown_dataset_returns_false(self):
        pm, _, _, _ = self._pm()
        assert pm.rename_dataset(9999, "whatever") is False
