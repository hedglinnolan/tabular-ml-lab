"""Stacking files: the operation researchers ask for most and the app lacked.

Combining NHANES cycles, study sites, or registry years is STACKING (same
measurements, different people), not joining. The previous UI offered only
joins, so "combine my cycles" had no answer at all.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.combine import (  # noqa: E402
    SOURCE_COLUMN, execute_stack, plan_stack, relationship_hint, reserved_columns,
)

RNG = np.random.RandomState(0)

@pytest.fixture(autouse=True)
def _deterministic_test_data():
    """Reseed the shared RNG before every test in this module.

    Without this, every test draws from one advancing stream, so the data a
    test sees depends on how many tests ran before it — and the suite is green
    only for the collection order it happens to run in. Two tests in this
    branch genuinely failed once the order was shuffled.
    """
    RNG.seed(0)



def cycle(start, n, extra=None):
    df = pd.DataFrame({
        "SEQN": range(start, start + n),
        "age": RNG.randint(18, 80, n),
        "glucose": RNG.normal(100, 20, n).round(1),
    })
    if extra:
        df[extra] = RNG.normal(10, 3, n).round(2)
    return df


def test_stacking_two_cycles_predicts_and_produces_the_same_rows():
    frames = {"1999-2000": cycle(1, 100), "2001-2002": cycle(101, 130)}
    plan = plan_stack(frames)
    assert plan.can_proceed and plan.total_rows == 230
    out, desc = execute_stack(frames)
    assert len(out) == plan.total_rows
    assert "230" in desc


def test_source_column_records_provenance():
    frames = {"1999-2000": cycle(1, 10), "2001-2002": cycle(11, 10)}
    out, _ = execute_stack(frames)
    assert set(out[SOURCE_COLUMN].unique()) == {"1999-2000", "2001-2002"}
    assert SOURCE_COLUMN in reserved_columns()


def test_summary_column_count_matches_the_result():
    """The number promised before stacking must match what is produced —
    including the source column."""
    frames = {"a": cycle(1, 10), "b": cycle(11, 10)}
    plan = plan_stack(frames)
    out, _ = execute_stack(frames)
    promised = len(plan.all_columns) + 1
    assert promised == out.shape[1]
    assert str(promised) in plan.summary()


def test_columns_missing_from_one_file_are_reported():
    frames = {"1999-2000": cycle(1, 50), "2001-2002": cycle(51, 50, extra="insulin")}
    plan = plan_stack(frames)
    assert "insulin" in plan.partial_columns
    assert plan.partial_columns["insulin"] == ["1999-2000"]
    assert plan.notes or plan.warnings


def test_type_conflict_across_files_is_warned():
    """A column that is numbers in one cycle and text in another becomes text
    overall — unusable, and silent without this warning."""
    a = cycle(1, 50)
    b = cycle(51, 50)
    b["glucose"] = b["glucose"].astype(str)
    plan = plan_stack({"a": a, "b": b})
    assert "glucose" in plan.type_conflicts
    assert any("different kinds of value" in w for w in plan.warnings)


def test_files_with_nothing_in_common_are_blocked_with_a_pointer():
    plan = plan_stack({"a": pd.DataFrame({"x": [1, 2]}), "b": pd.DataFrame({"y": [3, 4]})})
    assert not plan.can_proceed
    assert "shared ID" in " ".join(plan.blocking)      # points at linking instead


def test_single_file_cannot_be_stacked():
    assert not plan_stack({"a": cycle(1, 10)}).can_proceed
    with pytest.raises(ValueError):
        execute_stack({"a": cycle(1, 10)})


@pytest.mark.parametrize("frames,expected", [
    ({"c1": pd.DataFrame({"SEQN": range(20), "age": range(20), "glucose": range(20)}),
      "c2": pd.DataFrame({"SEQN": range(20, 40), "age": range(20), "glucose": range(20)})}, "stack"),
    ({"demo": pd.DataFrame({"SEQN": range(20), "age": range(20)}),
      "labs": pd.DataFrame({"SEQN": range(20), "glucose": range(20)})}, "link"),
])
def test_relationship_hint(frames, expected):
    assert relationship_hint(frames) == expected


def test_stacking_never_mutates_inputs():
    a, b = cycle(1, 10), cycle(11, 10)
    a0, b0 = a.copy(deep=True), b.copy(deep=True)
    execute_stack({"a": a, "b": b})
    pd.testing.assert_frame_equal(a, a0)
    pd.testing.assert_frame_equal(b, b0)


def test_stacked_frame_is_hashable_for_invalidation():
    """set_data() fingerprints with hash_pandas_object; a stacked frame with
    mixed object columns must not break that gate."""
    a, b = cycle(1, 10), cycle(11, 10)
    b["glucose"] = b["glucose"].astype(str)
    out, _ = execute_stack({"a": a, "b": b})
    pd.util.hash_pandas_object(out, index=False)


def test_a_users_own_source_file_column_is_not_overwritten():
    """`__source_file` is a legal column name, and a table exported from this
    app carries one. Re-stacking that export must not rewrite its provenance
    labels, and the promised column count must match what is delivered."""
    a = pd.DataFrame({"SEQN": [1, 2], SOURCE_COLUMN: ["orig_1999", "orig_1999"],
                      "age": [40, 55]})
    b = pd.DataFrame({"SEQN": [3, 4], SOURCE_COLUMN: ["orig_2001", "orig_2001"],
                      "age": [61, 33]})
    plan = plan_stack({"exportA": a, "exportB": b})
    out, _ = execute_stack({"exportA": a, "exportB": b})

    kept = [c for c in out.columns if c != SOURCE_COLUMN and SOURCE_COLUMN in str(c)]
    assert kept, (
        f"the user's own '{SOURCE_COLUMN}' column is gone: {list(out.columns)}")
    assert sorted(out[kept[0]].dropna().unique()) == ["orig_1999", "orig_2001"], (
        "the user's provenance labels were rewritten")
    assert out[SOURCE_COLUMN].tolist() == ["exportA", "exportA", "exportB", "exportB"]
    assert f"{out.shape[1]} column" in plan.summary() or \
        f"and {out.shape[1]} columns" in plan.summary(), (
        f"summary promises a different width than delivered "
        f"({out.shape[1]} cols): {plan.summary()}")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
