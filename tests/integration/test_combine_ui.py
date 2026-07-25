"""Step 2 driven end to end: several clean files -> one working table.

The user story: "Can I come to this app with multiple datasets that are
independently clean, but which I haven't taken the time to combine yet?"

These drive the real page with AppTest, so they catch what unit tests on the
pure engines cannot: whether the screen actually renders, whether the button
produces the table it promised, and whether the promise and the result agree.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

PAGE = os.path.join(PROJECT_ROOT, "pages", "01_Upload_and_Audit.py")
RNG = np.random.RandomState(0)


def _scenario(kind):
    if kind == "link":
        return {"demographics": pd.DataFrame({"SEQN": range(1000, 1200),
                                              "age": RNG.randint(18, 80, 200)}),
                "labs": pd.DataFrame({"patient_id": range(1050, 1250),
                                      "glucose": RNG.normal(100, 20, 200)})}
    if kind == "dtype":
        return {"demographics": pd.DataFrame({"SEQN": [f"{i:04d}" for i in range(1000, 1200)],
                                              "age": RNG.randint(18, 80, 200)}),
                "labs": pd.DataFrame({"SEQN": list(range(1000, 1200)),
                                      "glucose": RNG.normal(100, 20, 200)})}
    return {"1999-2000": pd.DataFrame({"SEQN": range(1, 101), "age": RNG.randint(18, 80, 100),
                                       "glucose": RNG.normal(100, 20, 100)}),
            "2001-2002": pd.DataFrame({"SEQN": range(101, 231), "age": RNG.randint(18, 80, 130),
                                       "glucose": RNG.normal(102, 21, 130)})}


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


@pytest.mark.parametrize("kind", ["link", "stack", "dtype"])
def test_combine_step_renders_without_error(kind):
    at = _app(_scenario(kind))
    at.run()
    assert not at.exception, at.exception
    assert "Step 2: Combine your files" in [h.value for h in at.header]


@pytest.mark.parametrize("kind", ["link", "stack", "dtype"])
def test_the_question_is_asked_in_plain_language(kind):
    """No SQL vocabulary: the user is asked how their FILES relate, not which
    join type they want."""
    at = _app(_scenario(kind))
    at.run()
    relation = [r for r in at.radio if any("people" in o for o in r.options)]
    assert relation, "the relationship question is missing"
    opts = " ".join(relation[0].options).lower()
    assert "same measurements on different people" in opts
    assert "different measurements on the same people" in opts
    for jargon in ("inner join", "outer join", "foreign key", "cardinality"):
        assert jargon not in _text(at).lower()


@pytest.mark.parametrize("kind,expected_rows", [("link", 150), ("stack", 230), ("dtype", 200)])
def test_combining_delivers_exactly_what_it_promised(kind, expected_rows):
    """The row count shown above the button must equal the table produced."""
    at = _app(_scenario(kind))
    at.run()
    button = [b for b in at.button if b.label == "Combine files"]
    assert button, "no Combine button rendered"
    button[0].click().run()
    assert not at.exception, at.exception
    assert "working_table" in at.session_state
    assert len(at.session_state["working_table"]) == expected_rows


def test_mismatched_id_types_are_explained_not_crashed():
    """'001' in one file and 1 in the other used to surface pandas' 'you should
    use pd.concat'; it must now read as English and still combine."""
    at = _app(_scenario("dtype"))
    at.run()
    blob = _text(at).lower()
    assert "stored as text" in blob and "numbers" in blob
    assert "concat" not in blob and "int64" not in blob


def test_partial_overlap_warns_before_dropping_anyone():
    at = _app(_scenario("link"))
    at.run()
    assert "no match" in _text(at).lower()


def test_stacked_table_records_which_file_each_row_came_from():
    from utils.combine import SOURCE_COLUMN
    at = _app(_scenario("stack"))
    at.run()
    [b for b in at.button if b.label == "Combine files"][0].click().run()
    assert SOURCE_COLUMN in at.session_state["working_table"].columns


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
