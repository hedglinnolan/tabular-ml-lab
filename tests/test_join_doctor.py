"""Join Doctor: the five ways merging silently ruins a researcher's data.

Each of these was reproduced against the app before the module existed. The
tests are written as the user's experience, not as API coverage: what did the
researcher see, and would they have noticed the damage?
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from ml.join_doctor import (  # noqa: E402
    diagnose_join, execute_join, find_key_candidates, normalize_key,
    plain_summary, repair_keys, suggest_best,
)

RNG = np.random.RandomState(0)
N = 200
SEQN = np.arange(83732, 83732 + N)          # NHANES-style identifiers


def demographics(n=N):
    return pd.DataFrame({"SEQN": SEQN[:n], "age": RNG.randint(18, 80, n)})


def labs(n=N, key="SEQN"):
    return pd.DataFrame({key: SEQN[:n], "glucose": RNG.normal(100, 25, n).round(1)})


# ── FAILURE 1: same key name, different dtype ────────────────────────────

def test_dtype_mismatch_is_blocking_and_explained_in_english():
    """Previously surfaced as: 'You are trying to merge on str and int64
    columns … you should use pd.concat' — meaningless to a researcher."""
    left = pd.DataFrame({"SEQN": ["001", "002", "003"], "age": [40, 55, 61]})
    right = pd.DataFrame({"SEQN": [1, 2, 3], "glucose": [95, 102, 110]})

    d = diagnose_join(left, right, "SEQN", "SEQN")
    assert d.dtype_mismatch and d.blocking and not d.can_proceed
    msg = " ".join(d.blocking).lower()
    assert "text" in msg and "numbers" in msg
    assert "concat" not in msg and "int64" not in msg
    assert "3" in " ".join(d.blocking)     # says how many would match after fixing


def test_dtype_mismatch_is_repairable_and_then_joins():
    left = pd.DataFrame({"SEQN": ["001", "002", "003"], "age": [40, 55, 61]})
    right = pd.DataFrame({"SEQN": [1, 2, 3], "glucose": [95, 102, 110]})
    merged, desc = execute_join(left, right, "SEQN", "SEQN", "inner")
    assert len(merged) == 3
    assert "age" in merged.columns and "glucose" in merged.columns
    assert desc


def test_zero_padded_and_float_ids_normalize_together():
    s = pd.Series(["001", "2", "003.0"])
    assert normalize_key(s).tolist() == ["1", "2", "3"]
    assert normalize_key(pd.Series([1.0, 2.0])).tolist() == ["1", "2"]


# ── FAILURE 2: keys named differently ────────────────────────────────────

def test_differently_named_keys_are_found_by_their_values():
    """Name-only matching reported 'no matching columns' — a dead end."""
    c = suggest_best(demographics(), labs(key="patient_id"))
    assert c is not None
    assert {c.left_col, c.right_col} == {"SEQN", "patient_id"}
    assert c.confidence in {"high", "medium"}


def test_case_differing_column_names_are_found():
    c = suggest_best(demographics(), labs(key="seqn"))
    assert c is not None and c.right_col == "seqn"


# ── FAILURE 3: duplicate keys multiply the cohort ────────────────────────

def test_repeated_measures_fan_out_is_warned():
    """3 subjects x visits -> 6 rows. Not larger than either input, so a naive
    row-count check misses it; every later 'n =' would be wrong."""
    subj = pd.DataFrame({"id": [1, 2, 3], "age": [40, 55, 61]})
    visits = pd.DataFrame({"id": [1, 1, 1, 2, 2, 3],
                           "visit": [1, 2, 3, 1, 2, 1],
                           "bp": [120, 122, 119, 118, 117, 130]})
    d = diagnose_join(subj, visits, "id", "id")
    assert d.row_multiplication
    assert d.predicted_rows == 6
    text = " ".join(d.warnings).lower()
    assert "repeated" in text or "several rows per id" in text
    assert "sample size" in text


def test_many_to_many_is_called_out_as_probably_wrong():
    a = pd.DataFrame({"id": [1, 1, 2, 2], "x": [1, 2, 3, 4]})
    b = pd.DataFrame({"id": [1, 1, 2, 2], "y": [5, 6, 7, 8]})
    d = diagnose_join(a, b, "id", "id")
    assert d.row_multiplication and d.predicted_rows == 8
    assert "mistake" in " ".join(d.warnings).lower()


def test_one_to_one_join_raises_no_fanout_warning():
    d = diagnose_join(demographics(), labs(), "SEQN", "SEQN")
    assert not d.row_multiplication
    assert d.predicted_rows == N


# ── FAILURE 4: silent cohort loss ────────────────────────────────────────

def test_inner_join_dropping_cohort_is_warned_with_numbers():
    left = pd.DataFrame({"subject_id": range(1, 101), "age": range(100)})
    right = pd.DataFrame({"subject_id": range(50, 151), "glucose": range(101)})
    d = diagnose_join(left, right, "subject_id", "subject_id", "inner")
    text = " ".join(d.warnings)
    assert "49" in text and "dropped" in text
    assert "left join" in text.lower()      # tells them how to keep the rows


def test_left_join_keeps_everything_and_predicts_rows():
    left = pd.DataFrame({"subject_id": range(1, 101), "age": range(100)})
    right = pd.DataFrame({"subject_id": range(50, 151), "glucose": range(101)})
    d = diagnose_join(left, right, "subject_id", "subject_id", "left")
    assert d.predicted_rows == 100


def test_predicted_row_count_matches_reality():
    """The number shown before the merge must equal what the merge produces."""
    left = pd.DataFrame({"id": range(1, 51), "a": range(50)})
    right = pd.DataFrame({"id": range(25, 76), "b": range(51)})
    for how in ("inner", "left", "right", "outer"):
        d = diagnose_join(left, right, "id", "id", how)
        merged, _ = execute_join(left, right, "id", "id", how)
        assert d.predicted_rows == len(merged), f"{how}: said {d.predicted_rows}, got {len(merged)}"


# ── FAILURE 5: whitespace / capitalisation in key VALUES ─────────────────

def test_whitespace_and_case_variants_still_match():
    left = pd.DataFrame({"id": [f"A{i:02d}" for i in range(1, 21)], "age": range(20)})
    right = pd.DataFrame({"id": [f"a{i:02d} " for i in range(1, 21)], "glucose": range(20)})
    d = diagnose_join(left, right, "id", "id")
    assert d.needs_normalization and d.matched_keys == 20
    merged, _ = execute_join(left, right, "id", "id")
    assert len(merged) == 20


# ── not misleading people ────────────────────────────────────────────────

def test_unrelated_files_get_no_suggestion():
    """Confidently proposing 'age ↔ gdp' is far worse than saying nothing."""
    a = pd.DataFrame({"SEQN": SEQN, "age": RNG.randint(18, 80, N)})
    b = pd.DataFrame({"country": RNG.choice(["US", "UK", "FR"], N),
                      "gdp": RNG.normal(5e4, 1e4, N)})
    assert suggest_best(a, b) is None


def test_row_counters_are_not_mistaken_for_keys():
    """Any two files carrying a 0..N counter overlap 100% by coincidence."""
    a = pd.DataFrame({"row": range(50), "age": RNG.randint(18, 80, 50)})
    b = pd.DataFrame({"row": range(50), "gdp": RNG.normal(5e4, 1e4, 50)})
    cands = find_key_candidates(a, b)
    counter = [c for c in cands if c.left_col == "row"]
    if counter:
        # It may be offered (the names do match), but never as a safe default
        # when the values are a bare counter.
        assert counter[0].index_like


def test_low_cardinality_column_is_never_a_key():
    """'sex' overlaps perfectly between any two cohorts but identifies nobody."""
    a = pd.DataFrame({"sex": ["M", "F"] * 50, "age": RNG.randint(18, 80, 100)})
    b = pd.DataFrame({"sex": ["M", "F"] * 50, "glucose": RNG.normal(100, 20, 100)})
    assert not [c for c in find_key_candidates(a, b) if c.left_col == "sex"]


# ── plumbing that keeps the merge honest ─────────────────────────────────

def test_colliding_columns_are_preserved_with_suffixes():
    a = pd.DataFrame({"id": [1, 2, 3], "bmi": [22.0, 28.0, 31.0]})
    b = pd.DataFrame({"id": [1, 2, 3], "bmi": [22.5, 28.5, 31.5]})
    d = diagnose_join(a, b, "id", "id")
    assert "bmi" in d.column_collisions
    merged, _ = execute_join(a, b, "id", "id", left_name="demo", right_name="labs")
    assert len(merged.columns) == 3          # id + both bmi columns, nothing lost
    assert any("bmi" in str(c) for c in merged.columns)


def test_no_overlap_at_all_is_blocking():
    a = pd.DataFrame({"id": [1, 2, 3], "x": [1, 2, 3]})
    b = pd.DataFrame({"id": [97, 98, 99], "y": [1, 2, 3]})
    d = diagnose_join(a, b, "id", "id")
    assert d.blocking and not d.can_proceed


def test_diagnosis_never_mutates_inputs():
    a, b = demographics(), labs()
    a0, b0 = a.copy(deep=True), b.copy(deep=True)
    diagnose_join(a, b, "SEQN", "SEQN")
    repair_keys(a, b, "SEQN", "SEQN")
    execute_join(a, b, "SEQN", "SEQN")
    pd.testing.assert_frame_equal(a, a0)
    pd.testing.assert_frame_equal(b, b0)


def test_plain_summary_is_written_for_a_human():
    d = diagnose_join(demographics(), labs(), "SEQN", "SEQN")
    s = plain_summary(d)
    assert "rows" in s and str(N) in s.replace(",", "")
    for jargon in ("dtype", "int64", "NaN", "pd.", "DataFrame"):
        assert jargon not in s


def test_execute_join_description_is_methods_ready():
    _, desc = execute_join(demographics(), labs(), "SEQN", "SEQN",
                           left_name="demographics", right_name="labs")
    assert "demographics" in desc and "labs" in desc and "inner join" in desc


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
