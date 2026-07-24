"""Import Doctor: structural diagnosis of messy research files.

The contract these pin is not "parse anything correctly" — that is impossible.
It is: detect visibly, propose reversibly, never mutate silently, and stay
quiet on clean data.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from ml.import_doctor import (  # noqa: E402
    ShapeFinding, apply_fix, diagnose, summarize,
)


def _ids(findings):
    return {f.id for f in findings}


def _first(findings, prefix):
    for f in findings:
        if f.id == prefix or f.id.startswith(prefix):
            return f
    raise AssertionError(f"no finding starting with {prefix!r}; got {sorted(_ids(findings))}")


# ── the cardinal rule: silence on clean data ─────────────────────────────

def test_clean_data_produces_no_findings():
    df = pd.DataFrame({
        "age": [40, 55, 61, 47],
        "bmi": [22.1, 28.4, 31.0, 24.5],
        "sex": ["M", "F", "M", "F"],
    })
    assert diagnose(df) == []
    assert "No structural problems" in summarize([])


def test_diagnose_never_mutates_input():
    df = pd.DataFrame({"a": [1, 2, 999, 4, 5, 6, 7, 8, 9, 10, 11],
                       "b": ["x ", "X", "y", "y", "y", "y", "y", "y", "y", "y", "y"]})
    before = df.copy(deep=True)
    diagnose(df)
    pd.testing.assert_frame_equal(df, before)


def test_empty_frame_is_safe():
    assert diagnose(pd.DataFrame()) == []
    assert diagnose(None) == []


# ── header stuck below a title block ─────────────────────────────────────

def _titled_export():
    return pd.DataFrame({
        "Unnamed: 0": ["NHANES Export", None, "subject_id", "1", "2", "3"],
        "Unnamed: 1": [None, None, "age", "40", "55", "61"],
        "Unnamed: 2": [None, None, "glucose", "95", "102", "110"],
    })


def test_header_in_later_row_detected():
    f = _first(diagnose(_titled_export()), "header_in_later_row")
    assert f.severity == "critical" and f.confidence == "high"
    assert f.params["row"] == 2  # zero-based -> "row 3" in the message


def test_header_finding_suppresses_downstream_noise():
    """A misplaced header makes every dtype wrong; reporting twenty derived
    symptoms would bury the one fix that matters."""
    findings = diagnose(_titled_export())
    assert len(findings) == 1 and findings[0].id == "header_in_later_row"


def test_promote_header_fix():
    df = _titled_export()
    fixed, desc = apply_fix(df, _first(diagnose(df), "header_in_later_row"))
    assert list(fixed.columns) == ["subject_id", "age", "glucose"]
    assert len(fixed) == 3
    assert "row 3" in desc
    assert len(df) == 6  # original untouched


# ── numeric sentinels ────────────────────────────────────────────────────

def test_sentinel_detected_when_far_outside_distribution():
    df = pd.DataFrame({"age": [40, 55, 61, 999, 47, 52, 33, 999, 61, 44, 39]})
    f = _first(diagnose(df), "sentinel_missing__age")
    assert f.severity == "critical"
    assert 999 in f.params["values"]


def test_plausible_value_is_not_flagged_as_sentinel():
    """999 is a plausible triglyceride value — flagging it would be worse than
    missing it, so the check requires the value to sit far outside the spread."""
    rng = np.random.RandomState(0)
    df = pd.DataFrame({"triglycerides": np.append(rng.randint(700, 1100, 40), [999, 999])})
    assert not any(f.id.startswith("sentinel_missing") for f in diagnose(df))


def test_recode_missing_fix_numeric():
    df = pd.DataFrame({"age": [40, 55, 61, 999, 47, 52, 33, 999, 61, 44, 39]})
    fixed, desc = apply_fix(df, _first(diagnose(df), "sentinel_missing__age"))
    assert int(fixed["age"].isna().sum()) == 2
    assert "999" in desc
    assert int(df["age"].isna().sum()) == 0  # original untouched


# ── text missing tokens ──────────────────────────────────────────────────

def test_text_missing_tokens_detected_and_recoded():
    df = pd.DataFrame({"income": ["45000", "N/A", "52000", "n/a", "48000", "51000"]})
    f = _first(diagnose(df), "text_missing__income")
    fixed, _ = apply_fix(df, f)
    assert int(fixed["income"].isna().sum()) == 2


# ── numeric stored as text ───────────────────────────────────────────────

@pytest.mark.parametrize("values", [
    ["45,000", "52,300", "61,000", "48,000", "55,500", "39,900"],       # thousands
    ["180 mg/dL", "195 mg/dL", "210 mg/dL", "175 mg/dL", "188 mg/dL"],  # units
    ["12%", "15%", "9%", "22%", "18%", "31%"],                          # percent
])
def test_numeric_stored_as_text_detected(values):
    df = pd.DataFrame({"v": values})
    f = _first(diagnose(df), "numeric_as_text__v")
    fixed, _ = apply_fix(df, f)
    assert pd.api.types.is_numeric_dtype(fixed["v"])
    assert fixed["v"].notna().all()


def test_genuine_text_column_is_not_coerced():
    df = pd.DataFrame({"site": ["Boston", "Denver", "Austin", "Boston", "Denver", "Austin"]})
    assert not any(f.id.startswith("numeric_as_text") for f in diagnose(df))


def test_below_detection_limit_is_disclosed():
    """'<0.01' becomes 0.01 — a real semantic loss, so the fix must say that
    comparison signs were removed rather than convert silently."""
    df = pd.DataFrame({"chol": ["<0.01", "180", "195", "210", "175", "188"]})
    f = _first(diagnose(df), "numeric_as_text__chol")
    _, desc = apply_fix(df, f)
    assert "comparison signs" in desc


# ── categorical variants ─────────────────────────────────────────────────

def test_case_and_whitespace_variants_merged():
    df = pd.DataFrame({"sex": ["Male", "female", "Female ", "male", "MALE", "Female"]})
    f = _first(diagnose(df), "category_variants__sex")
    fixed, _ = apply_fix(df, f)
    assert set(fixed["sex"].unique()) == {"Male", "Female"}


# ── duplicates, empties, footers, constants ──────────────────────────────

def test_duplicate_columns_detected_and_renamed():
    df = pd.DataFrame(np.arange(12).reshape(4, 3), columns=["a", "b", "a"])
    fixed, _ = apply_fix(df, _first(diagnose(df), "duplicate_columns"))
    assert len(set(fixed.columns)) == 3


def test_empty_column_and_row_detected():
    df = pd.DataFrame({"a": [1, 2, None], "notes": [None, None, None]})
    ids = _ids(diagnose(df))
    assert "empty_columns" in ids


def test_drop_empty_rows_fix():
    df = pd.DataFrame({"a": [1, None, 3], "b": [4, None, 6]})
    fixed, desc = apply_fix(df, _first(diagnose(df), "empty_rows"))
    assert len(fixed) == 2 and "1" in desc


def test_footer_row_detected():
    df = pd.DataFrame({
        "id": list(range(1, 9)) + ["Total"],
        "v": [1, 2, 3, 4, 5, 6, 7, 8, None],
    })
    f = _first(diagnose(df), "footer_rows")
    fixed, _ = apply_fix(df, f)
    assert "Total" not in fixed["id"].astype(str).tolist()


def test_constant_column_is_low_confidence():
    """A constant may be a meaningful study-level label, so it is never
    pre-selected for removal."""
    df = pd.DataFrame({"site": ["A"] * 6, "v": [1, 2, 3, 4, 5, 6]})
    f = _first(diagnose(df), "constant_columns")
    assert f.severity == "info" and f.confidence == "low"
    assert f.auto_suggestable is False


# ── repeated measures ────────────────────────────────────────────────────

def test_wide_repeated_measures_detected_and_melted():
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "bp_1": [120, 118, 130],
        "bp_2": [122, 117, 133],
        "bp_3": [119, 116, 131],
    })
    f = _first(diagnose(df), "wide_repeated_measures")
    assert f.confidence == "low"          # reshaping is a research decision
    fixed, desc = apply_fix(df, f)
    assert len(fixed) == 9                # 3 subjects x 3 timepoints
    assert {"measurement", "value"} <= set(fixed.columns)


# ── contracts that keep the UI honest ────────────────────────────────────

def test_only_high_confidence_findings_are_auto_suggestable():
    df = pd.DataFrame({
        "age": [40, 55, 61, 999, 47, 52, 33, 999, 61, 44, 39],
        "site": ["A"] * 11,
    })
    for f in diagnose(df):
        if f.auto_suggestable:
            assert f.confidence == "high"


def test_every_finding_is_actionable_and_explained():
    df = pd.DataFrame({
        "age": [40, 55, 61, 999, 47, 52, 33, 999, 61, 44, 39],
        "sex": ["Male", "male ", "F", "F", "F", "F", "F", "F", "F", "F", "F"],
        "inc": ["45,000", "52,300", "61,000", "48,000", "55,500",
                "39,900", "72,000", "50,000", "47,250", "60,100", "41,000"],
    })
    findings = diagnose(df)
    assert findings
    for f in findings:
        assert f.title and f.detail and f.why_it_matters and f.fix_label
        assert f.severity in {"critical", "warning", "info"}
        assert f.confidence in {"high", "medium", "low"}
        # Every proposed fix must actually run and return a new frame.
        out, desc = apply_fix(df, f)
        assert isinstance(out, pd.DataFrame) and desc


def test_unknown_fix_kind_raises():
    bad = ShapeFinding(id="x", severity="info", title="t", detail="d",
                       why_it_matters="w", fix_label="l", fix_kind="nope")
    with pytest.raises(ValueError):
        apply_fix(pd.DataFrame({"a": [1]}), bad)


def test_summarize_counts_by_severity():
    df = pd.DataFrame({
        "age": [40, 55, 61, 999, 47, 52, 33, 999, 61, 44, 39],
        "site": ["A"] * 11,
    })
    text = summarize(diagnose(df))
    assert "Found" in text


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
