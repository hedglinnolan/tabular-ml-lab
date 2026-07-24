"""JSON / JSON Lines ingestion.

Researchers export JSON from REDCap, APIs, and notebooks in several shapes.
These pin the shapes we accept, the flattening of nested fields, and — just as
important — that non-tabular JSON is rejected with a message a non-programmer
can act on rather than a raw pandas traceback.
"""
from __future__ import annotations

import io
import os
import sys

import pandas as pd
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from data_processor import detect_file_type, load_json, load_tabular_data  # noqa: E402


def _b(text: str) -> io.BytesIO:
    return io.BytesIO(text.encode("utf-8"))


# ── extension detection ──────────────────────────────────────────────────

@pytest.mark.parametrize("filename,expected", [
    ("d.json", "json"),
    ("d.JSON", "json"),
    ("d.jsonl", "jsonl"),
    ("d.ndjson", "jsonl"),
    ("d.csv", "csv"),
    ("d.parquet", "parquet"),
    ("d.tsv", "tsv"),
])
def test_detect_file_type(filename, expected):
    assert detect_file_type(filename) == expected


# ── shapes that must load ────────────────────────────────────────────────

def test_records_array():
    df = load_json(_b('[{"age":40,"bmi":22.1},{"age":55,"bmi":28.4}]'))
    assert df.shape == (2, 2)
    assert list(df.columns) == ["age", "bmi"]
    assert df["age"].tolist() == [40, 55]


@pytest.mark.parametrize("wrapper", ["data", "records", "results", "rows", "items"])
def test_wrapped_payload(wrapper):
    """API exports wrap the rows in a key alongside metadata."""
    df = load_json(_b('{"%s":[{"a":1},{"a":2}],"meta":{"version":1}}' % wrapper))
    assert df.shape == (2, 1)
    assert df["a"].tolist() == [1, 2]


def test_pandas_orient_split():
    df = load_json(_b('{"columns":["a","b"],"data":[[1,2],[3,4]],"index":[0,1]}'))
    assert list(df.columns) == ["a", "b"]
    assert df.shape == (2, 2)


def test_pandas_orient_columns():
    df = load_json(_b('{"a":{"0":1,"1":2},"b":{"0":3,"1":4}}'))
    assert df.shape == (2, 2)
    assert set(df.columns) == {"a", "b"}


def test_pandas_orient_table():
    df = load_json(_b('{"schema":{"fields":[]},"data":[{"a":1},{"a":2}]}'))
    assert df.shape == (2, 1)


def test_nested_objects_are_flattened():
    df = load_json(_b('[{"id":1,"vitals":{"bp":120,"hr":70}}]'))
    assert "vitals.bp" in df.columns and "vitals.hr" in df.columns
    assert df["vitals.bp"].iloc[0] == 120


def test_single_record_becomes_one_row():
    df = load_json(_b('{"age":40,"bmi":22.1}'))
    assert df.shape == (1, 2)


def test_array_of_arrays_and_scalars():
    assert load_json(_b("[[1,2],[3,4]]")).shape == (2, 2)
    assert load_json(_b("[1,2,3]")).shape == (3, 1)


def test_json_lines_explicit():
    df = load_json(_b('{"a":1}\n{"a":2}\n{"a":3}'), lines=True)
    assert df.shape == (3, 1)


def test_ndjson_mislabeled_as_json_still_loads():
    """A .json file containing NDJSON is a very common export; the loader
    falls back rather than blocking the user on a naming convention."""
    df = load_json(_b('{"a":1}\n{"a":2}'))
    assert df.shape == (2, 1)


def test_records_key_disambiguates():
    df = load_json(_b('{"x":[{"a":1}],"y":[{"b":2}]}'), records_key="x")
    assert list(df.columns) == ["a"]


def test_ragged_records_produce_union_of_columns():
    df = load_json(_b('[{"a":1},{"a":2,"b":3}]'))
    assert set(df.columns) == {"a", "b"}
    assert pd.isna(df["b"].iloc[0])


def test_utf8_bom_is_tolerated():
    raw = '﻿[{"a":1}]'.encode("utf-8")
    assert load_json(io.BytesIO(raw)).shape == (1, 1)


# ── shapes that must fail, with an actionable message ────────────────────

@pytest.mark.parametrize("payload,needle", [
    ("   ", "empty"),
    ("[]", "empty list"),
    ("42", "single value"),
    ('{"a": 1,,}', "not valid JSON"),
    ('{"x":[{"a":1}],"y":[{"b":2}]}', "several possible row sets"),
])
def test_bad_json_raises_readable_error(payload, needle):
    with pytest.raises(ValueError) as exc:
        load_json(_b(payload))
    msg = str(exc.value)
    assert needle in msg, f"unhelpful message: {msg}"
    # No raw pandas/json internals leaking to a non-programmer.
    assert "Traceback" not in msg


def test_ambiguous_error_names_the_candidate_keys():
    with pytest.raises(ValueError) as exc:
        load_json(_b('{"x":[{"a":1}],"y":[{"b":2}]}'))
    assert "x" in str(exc.value) and "y" in str(exc.value)


# ── end-to-end through the shared loader ─────────────────────────────────

@pytest.mark.parametrize("filename,payload,shape", [
    ("study.json", '[{"a":1},{"a":2}]', (2, 1)),
    ("study.jsonl", '{"a":1}\n{"a":2}', (2, 1)),
    ("study.ndjson", '{"a":1}\n{"a":2}\n{"a":3}', (3, 1)),
])
def test_load_tabular_data_routes_json(filename, payload, shape):
    df = load_tabular_data(_b(payload), filename=filename)
    assert df.shape == shape


def test_load_tabular_data_transpose_still_applies():
    df = load_tabular_data(_b('[{"a":1,"b":2}]'), filename="x.json", transpose=True)
    assert df.shape == (2, 1)



# ── defects found by adversarial stress-testing ──────────────────────────

def test_nested_arrays_do_not_poison_the_frame():
    """A list in a cell makes nunique/duplicated/hash_pandas_object raise
    'unhashable type: list'. The app fingerprints uploads with
    hash_pandas_object to decide when to invalidate downstream results, so one
    list cell silently disabled that gate and let stale models outlive a data
    change. Cells are rendered as JSON text instead."""
    df = load_json(_b('[{"id":1,"visits":[1,2,3]},{"id":2,"visits":[4,5]}]'))
    assert isinstance(df["visits"].iloc[0], str)
    df.nunique()
    df.duplicated()
    pd.util.hash_pandas_object(df, index=False)


def test_deeply_nested_objects_are_also_hashable():
    df = load_json(_b('[{"id":1,"a":{"b":{"c":{"d":1}}}}]'))
    pd.util.hash_pandas_object(df, index=False)


def test_geojson_is_rejected_not_flattened():
    """GeoJSON flattened into geometry fragments looks like a real table and
    a user would try to model it."""
    payload = ('{"type":"FeatureCollection","features":[{"type":"Feature",'
               '"geometry":{"type":"Point","coordinates":[1,2]},"properties":{"a":1}}]}')
    with pytest.raises(ValueError, match="GeoJSON"):
        load_json(_b(payload))


def test_jsonl_containing_one_array_line_is_read_as_records():
    """A .jsonl holding an ordinary JSON array on one line previously became a
    1-row frame whose single cell was a list of dicts."""
    df = load_json(_b('[{"a":1},{"a":2}]'), lines=True)
    assert df.shape == (2, 1)
    assert df["a"].tolist() == [1, 2]


def test_utf16_json_loads():
    """PowerShell's ConvertTo-Json | Out-File writes UTF-16 by default."""
    assert load_json(io.BytesIO('[{"a":1},{"a":2}]'.encode("utf-16"))).shape == (2, 1)

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
