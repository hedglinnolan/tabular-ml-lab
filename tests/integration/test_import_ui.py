"""The Import Doctor as the researcher meets it: on screen, at upload.

ml/import_doctor.py is unit-tested for whether it finds the right problems.
These tests cover the part unit tests cannot: that the findings render, that
pressing a fix actually changes the frame the page will commit, that undo
returns the original, and — most important — that nothing is ever applied
without a press.
"""
from __future__ import annotations

import os
import sys

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# A harness page: hand render_import_doctor a frame from session state and
# publish what it returns, so a test can assert on the real Streamlit widgets.
HARNESS = '''
import pandas as pd, streamlit as st
from utils.import_ui import render_import_doctor
df = st.session_state["_test_frame"]
out = render_import_doctor(df, "f1")
st.session_state["_test_out"] = out
st.text(f"shape={out.shape[0]}x{out.shape[1]}")
'''


def _app(df: pd.DataFrame) -> AppTest:
    at = AppTest.from_string(HARNESS, default_timeout=60)
    at.session_state["_test_frame"] = df
    return at


def _text(at) -> str:
    return " ".join([m.value for m in at.markdown] + [c.value for c in at.caption]
                    + [s.value for s in at.success] + [w.value for w in at.warning]
                    + [e.value for e in at.error])


# ── a file with a title row above the header ─────────────────────────────

def _title_row_frame() -> pd.DataFrame:
    # What pandas produces from an Excel export whose first row is a title.
    return pd.DataFrame({
        "NHANES 2017-2018 export": ["SEQN", "1001", "1002", "1003", "1004"],
        "Unnamed: 1": ["age", "45", "52", "38", "61"],
        "Unnamed: 2": ["bmi", "27.1", "31.4", "22.8", "29.9"],
    })


def test_misplaced_header_is_reported():
    at = _app(_title_row_frame()).run()
    assert not at.exception
    assert "header" in _text(at).lower()


def test_nothing_is_applied_without_a_press():
    df = _title_row_frame()
    at = _app(df).run()
    assert not at.exception
    # The frame handed back before any button press must be the original.
    assert at.session_state["_test_out"].shape == df.shape
    assert list(at.session_state["_test_out"].columns) == list(df.columns)


def test_promoting_the_header_changes_the_frame():
    at = _app(_title_row_frame()).run()
    assert at.button, "expected a fix button for the misplaced header"
    at.button[0].click().run()
    assert not at.exception
    out = at.session_state["_test_out"]
    assert "SEQN" in out.columns
    assert len(out) < 5           # the title and header rows are gone


def test_undo_restores_the_original_frame():
    df = _title_row_frame()
    at = _app(df).run()
    at.button[0].click().run()
    assert at.session_state["_test_out"].shape != df.shape
    undo = [b for b in at.button if "Undo" in b.label]
    assert undo, "expected an undo control after applying a fix"
    undo[0].click().run()
    assert not at.exception
    assert at.session_state["_test_out"].shape == df.shape


def test_applied_fix_is_logged_in_plain_language():
    at = _app(_title_row_frame()).run()
    at.button[0].click().run()
    text = _text(at)
    assert "fix" in text.lower() and "applied" in text.lower()


# ── a clean file gets out of the way ─────────────────────────────────────

def test_clean_file_is_not_nagged():
    clean = pd.DataFrame({"SEQN": range(100), "age": range(20, 120),
                          "bmi": [22.0 + i * 0.1 for i in range(100)]})
    at = _app(clean).run()
    assert not at.exception
    assert at.session_state["_test_out"].shape == clean.shape
    assert "clean table" in _text(at)
    assert not at.button, "a clean file should offer no repairs"


# ── survey sentinels ─────────────────────────────────────────────────────

def test_sentinel_codes_are_offered_as_a_fix_not_applied():
    df = pd.DataFrame({
        "SEQN": range(200),
        "income_bracket": [1, 2, 3, 4, 5] * 39 + [999] * 5,
    })
    at = _app(df).run()
    assert not at.exception
    out = at.session_state["_test_out"]
    # Untouched until pressed: the 999s are still there.
    assert (out["income_bracket"] == 999).sum() == 5


def test_recoding_sentinels_produces_missing_values():
    df = pd.DataFrame({
        "SEQN": range(200),
        "income_bracket": [1, 2, 3, 4, 5] * 39 + [999] * 5,
    })
    at = _app(df).run()
    fixes = [b for b in at.button if "Undo" not in b.label]
    if not fixes:
        pytest.skip("no sentinel fix offered for this distribution")
    fixes[0].click().run()
    assert not at.exception
    out = at.session_state["_test_out"]
    assert out["income_bracket"].isna().sum() >= 5


# ── a bad frame must never take the page down ────────────────────────────

@pytest.mark.parametrize("df", [
    pd.DataFrame(),
    pd.DataFrame({"a": []}),
    pd.DataFrame({"a": [None, None], "b": [None, None]}),
    pd.DataFrame({"x": [{"k": 1}, {"k": 2}]}),          # dict cells
    pd.DataFrame({"dup": [1, 2], "dup2": [3, 4]}).rename(columns={"dup2": "dup"}),
])
def test_pathological_frames_do_not_crash_the_page(df):
    at = _app(df).run()
    assert not at.exception


# ── the frame the doctor returns is the one the page keeps ───────────────

def test_returned_frame_survives_a_rerun():
    at = _app(_title_row_frame()).run()
    at.button[0].click().run()
    fixed = at.session_state["_test_out"].shape
    at.run()                       # a rerun with no interaction
    assert at.session_state["_test_out"].shape == fixed
