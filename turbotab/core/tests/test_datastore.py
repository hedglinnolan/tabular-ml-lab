"""The data layer: ingest fidelity and row identity are Tier A (BLUEPRINT §8).

Every sample CSV must arrive with pandas' row count, pandas' column names and
its values in file order; ``__row_id`` must be 0..n-1 in file order, including
on a file DuckDB reads in parallel. The rest is known-answer checks on the
queries the UI and the modeling bridge make.
"""
from __future__ import annotations

import csv
import datetime as dt
import decimal
import json
import math
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from turbotab.core import datastore as ds_mod
from turbotab.core.datastore import (ROW_ID, DataStore, MemoryBudgetExceeded, UnknownColumn,
                                     ingest, json_safe, pandas_style_names)

REPO_ROOT = Path(__file__).resolve().parents[3]
SAMPLE_CSVS = sorted((REPO_ROOT / "turbotab" / "sample_data").glob("*.csv"))
BUDGET = 4 * 1024**3

INFO_KEYS = {"n_rows", "n_cols", "columns", "source_bytes", "parquet_bytes", "ingest_seconds",
             "fingerprint", "warnings"}
COLUMN_KEYS = {"name", "dtype", "physical_type", "n_missing", "n_unique", "sample"}
SUMMARY_KEYS = {"name", "dtype", "n", "n_missing", "n_unique", "mean", "std", "min", "q25",
                "median", "q75", "max", "top"}
DTYPES = {"numeric", "integer", "boolean", "categorical", "datetime", "text"}


def _ingest(tmp_path: Path, source: Path, budget: int = BUDGET):
    dest = tmp_path / "project" / "data" / "raw.parquet"
    info = ingest(source, dest)
    return info, DataStore(dest, budget), dest


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _row_ids(parquet: Path) -> np.ndarray:
    return pq.read_table(parquet, columns=[ROW_ID]).column(ROW_ID).to_numpy()


def _is_plain_number(series: pd.Series) -> bool:
    return (pd.api.types.is_numeric_dtype(series)
            and not pd.api.types.is_bool_dtype(series))


def _close(a, b) -> bool:
    if a is None or b is None or (isinstance(b, float) and math.isnan(b)):
        return (a is None) and (b is None or (isinstance(b, float) and math.isnan(b)))
    return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-9)


# ── ingest fidelity, every fixture ───────────────────────────────────────────

@pytest.mark.parametrize("source", SAMPLE_CSVS, ids=[p.name for p in SAMPLE_CSVS])
def test_every_sample_csv_arrives_as_pandas_reads_it(tmp_path, source):
    info, store, dest = _ingest(tmp_path, source)
    expected = pd.read_csv(source)

    assert info.n_rows == len(expected)
    assert [c.name for c in info.columns] == list(expected.columns)
    assert info.n_cols == expected.shape[1]
    assert store.columns == list(expected.columns) and ROW_ID not in store.columns

    # Row identity: dense and in file order.
    assert np.array_equal(_row_ids(dest), np.arange(len(expected)))

    # Values in file order, column by column, wherever both readers agree on
    # the kind of column (pandas and TurboTab differ on some NA tokens).
    frame = store.materialize()
    assert list(frame.index) == list(range(len(expected)))
    compared = 0
    for col in expected.columns:
        ours, theirs = frame[col], expected[col]
        if _is_plain_number(ours) and _is_plain_number(theirs):
            assert np.allclose(ours.to_numpy(float), theirs.to_numpy(float),
                               equal_nan=True), col
            compared += 1
        elif ours.dtype == object and theirs.dtype == object:
            both = ours.notna() & theirs.notna()
            assert (ours[both].astype(str) == theirs[both].astype(str)).all(), col
            compared += 1
    assert compared >= min(3, expected.shape[1])

    # Every header TurboTab had to rename is named in the warnings.
    with open(source, newline="", encoding="utf-8") as fh:
        raw = next(csv.reader(fh))
    for cell, name in zip(raw, info.columns):
        if cell != name.name:
            assert any(f'"{name.name}"' in w for w in info.warnings), name.name

    # Summaries agree with pandas on every numeric column.
    summaries = {s["name"]: s for s in store.summaries()}
    for col in expected.columns:
        s = summaries[col]
        assert set(s) == SUMMARY_KEYS and s["dtype"] in DTYPES
        assert s["n"] + s["n_missing"] == info.n_rows
        if s["dtype"] in ("numeric", "integer") and _is_plain_number(expected[col]):
            x = expected[col].astype(float)
            assert s["n"] == x.notna().sum()
            q = x.quantile([0.25, 0.5, 0.75])
            for key, want in (("mean", x.mean()), ("std", x.std()), ("min", x.min()),
                              ("max", x.max()), ("q25", q[0.25]), ("median", q[0.5]),
                              ("q75", q[0.75])):
                assert _close(s[key], want), (col, key, s[key], want)

    json.dumps(info.to_dict(), allow_nan=False)
    json.dumps(store.summaries(), allow_nan=False)


def test_row_ids_follow_file_order_on_a_file_read_in_parallel(tmp_path):
    """~29 MB is several of DuckDB's 8 MB parallel CSV chunks and >1 row group."""
    n = 1_300_000
    rng = np.random.default_rng(3)
    values = rng.normal(size=n)
    tags = np.array(["alpha", "beta", "gamma"])[rng.integers(0, 3, n)]
    path = tmp_path / "long.csv"
    with open(path, "w") as fh:
        fh.write("line,value,tag\n")
        fh.write("".join(f"{i},{v:.6f},{t}\n" for i, v, t in zip(range(n), values, tags)))
    assert path.stat().st_size > 3 * 8 * 1024**2

    info, store, dest = _ingest(tmp_path, path)
    assert info.n_rows == n
    table = pq.read_table(dest, columns=[ROW_ID, "line"])
    ids, lines = table.column(ROW_ID).to_numpy(), table.column("line").to_numpy()
    assert np.array_equal(ids, np.arange(n))
    assert np.array_equal(lines, ids)

    # A window across a row-group boundary is still exact and ordered.
    first_group = pq.ParquetFile(dest).metadata.row_group(0).num_rows
    assert pq.ParquetFile(dest).metadata.num_row_groups > 1
    window = store.window(first_group - 3, 6, columns=["line"])
    assert window["rows"] == [[first_group - 3 + k] for k in range(6)]

    # Materialize keeps the requested order and the ids as the index.
    picked = store.materialize(["line"], row_ids=[5, 1_000_000, 3])
    assert list(picked.index) == [5, 1_000_000, 3]
    assert list(picked["line"]) == [5, 1_000_000, 3]


# ── identifiers, missing values, windows ─────────────────────────────────────

def test_leading_zero_identifiers_stay_text(tmp_path):
    src = _write(tmp_path, "ids.csv", "id,score\n007,1\n010,2\n3,3\n")
    info, store, _ = _ingest(tmp_path, src)
    col = {c.name: c for c in info.columns}
    assert col["id"].physical_type == "VARCHAR" and col["id"].dtype == "categorical"
    assert col["score"].dtype == "integer"
    assert store.window(0, 3, columns=["id"])["rows"] == [["007"], ["010"], ["3"]]
    assert list(store.materialize(["id"])["id"]) == ["007", "010", "3"]


def test_a_leading_zero_past_the_sniffers_sample_still_stays_text(tmp_path):
    n, late = 30_000, 25_000
    lines = ["id,count"] + [f"{'007' if i == late else i + 1},{i}" for i in range(n)]
    src = _write(tmp_path, "late.csv", "\n".join(lines) + "\n")
    info, store, _ = _ingest(tmp_path, src)
    col = {c.name: c for c in info.columns}
    assert info.n_rows == n
    assert col["id"].physical_type == "VARCHAR"
    assert col["count"].physical_type == "BIGINT"
    assert any('"id"' in w and "007" in w for w in info.warnings)
    assert store.window(late, 1, columns=["id"])["rows"] == [["007"]]
    assert store.window(late + 1, 1, columns=["id"])["rows"] == [[str(late + 2)]]


def test_a_window_is_the_exact_slice_and_missing_is_none(tmp_path):
    src = _write(tmp_path, "w.csv", "a,b,c\n1,x,1.5\n2,,NA\n3,z,NaN\n4,w,2.5\n5,v,\n")
    info, store, _ = _ingest(tmp_path, src)
    assert {c.name: c.n_missing for c in info.columns} == {"a": 0, "b": 1, "c": 3}

    w = store.window(1, 3)
    assert w == {"columns": ["a", "b", "c"], "rows": [[2, None, None], [3, "z", None],
                                                     [4, "w", 2.5]],
                 "total_rows": 5, "offset": 1}
    json.dumps(w, allow_nan=False)
    assert store.window(4, 10)["rows"] == [[5, "v", None]]
    assert store.window(10, 5)["rows"] == []
    assert store.window(0, 0)["rows"] == []
    assert store.window(0, 2, columns=["c", "a"]) == {
        "columns": ["c", "a"], "rows": [[1.5, 1], [None, 2]], "total_rows": 5, "offset": 0}
    assert store.window(0, 2, columns=[])["rows"] == [[], []]
    with pytest.raises(UnknownColumn):
        store.window(0, 2, columns=["nope"])
    with pytest.raises(UnknownColumn):
        store.window(0, 2, columns=[ROW_ID])
    with pytest.raises(ValueError):
        store.window(-1, 2)


def test_json_safe_makes_every_value_a_browser_can_take():
    assert json_safe(float("nan")) is None and json_safe(np.float64("inf")) is None
    assert json_safe(np.int64(3)) == 3 and type(json_safe(np.int64(3))) is int
    assert json_safe(np.bool_(True)) is True
    assert json_safe(decimal.Decimal("1.5")) == 1.5
    assert json_safe(dt.date(2024, 1, 2)) == "2024-01-02"
    assert json_safe(dt.datetime(2024, 1, 2, 3, 4, 5)) == "2024-01-02T03:04:05"
    assert json_safe(pd.Timestamp("2024-01-02 03:04:05")) == "2024-01-02T03:04:05"
    assert json_safe(pd.NaT) is None and json_safe(pd.NA) is None
    assert json_safe([np.float32(0.5), {"k": np.nan}]) == [0.5, {"k": None}]


def test_header_names_are_kept_exactly_and_renamed_as_pandas_does(tmp_path):
    src = _write(tmp_path, "h.csv", '" a ","b""q","é,x",dup,dup,,last\n1,2,3,4,5,6,7\n')
    info, store, _ = _ingest(tmp_path, src)
    names = [c.name for c in info.columns]
    assert names == list(pd.read_csv(src).columns)
    assert names == [" a ", 'b"q', "é,x", "dup", "dup.1", "Unnamed: 5", "last"]
    assert any('"dup.1"' in w for w in info.warnings)
    assert any('"Unnamed: 5"' in w for w in info.warnings)
    assert store.window(0, 1, columns=["é,x", 'b"q'])["rows"] == [[3, 2]]
    assert pandas_style_names(["x", "x", "x.1", None])[0] == ["x", "x.1", "x.1.1", "Unnamed: 3"]


def test_type_detection_retries_on_every_row_before_giving_up(tmp_path):
    n = 25_000
    lines = ["n,m"] + [f"{i},{i}" for i in range(n - 1)] + [f"abc,{n}"]
    src = _write(tmp_path, "late_text.csv", "\n".join(lines) + "\n")
    info, store, _ = _ingest(tmp_path, src)
    col = {c.name: c for c in info.columns}
    assert info.n_rows == n
    assert col["n"].physical_type == "VARCHAR" and col["m"].physical_type == "BIGINT"
    assert any("detected again from every row" in w for w in info.warnings)
    assert store.window(n - 1, 1)["rows"] == [["abc", n]]


def test_when_detection_keeps_failing_every_column_is_text_and_it_says_so(
        tmp_path, monkeypatch):
    real = ds_mod._sniff_csv

    def wrong_types(*args, **kwargs):
        dialect = real(*args, **kwargs)
        dialect.types[0] = "BIGINT"   # "name" holds words; any integer read fails
        return dialect

    monkeypatch.setattr(ds_mod, "_sniff_csv", wrong_types)
    src = _write(tmp_path, "names.csv", "name,age\nalice,30\nbob,41\n")
    info, store, _ = _ingest(tmp_path, src)
    assert info.n_rows == 2
    assert [c.physical_type for c in info.columns] == ["VARCHAR", "VARCHAR"]
    assert any("every column was read as text" in w for w in info.warnings)
    assert store.window(0, 2)["rows"] == [["alice", "30"], ["bob", "41"]]


def test_short_rows_are_padded_as_pandas_pads_them(tmp_path):
    src = _write(tmp_path, "ragged.csv", "a,b,c\n1,2,3\n4,5\n6,7,8\n")
    info, store, _ = _ingest(tmp_path, src)
    assert info.n_rows == len(pd.read_csv(src)) == 3
    assert store.window(1, 1)["rows"] == [[4, 5, None]]
    assert any("fewer fields" in w for w in info.warnings)


def test_a_latin1_file_is_read_and_flagged(tmp_path):
    src = tmp_path / "latin.csv"
    src.write_bytes("name,val\nJos\xe9,1\nZo\xeb,2\n".encode("latin-1"))
    info, store, _ = _ingest(tmp_path, src)
    assert store.window(0, 2)["rows"] == [["José", 1], ["Zoë", 2]]
    assert any("Latin-1" in w for w in info.warnings)


# ── other sources ────────────────────────────────────────────────────────────

def test_parquet_source_gets_row_ids_and_nan_becomes_missing(tmp_path):
    src = tmp_path / "in.parquet"
    pq.write_table(pa.table({
        "x": pa.array(np.array([1.0, np.nan, 3.0]), from_pandas=False),
        "k": pa.array([1, 2, 3], pa.int64()),
        "s": pa.array(["a", None, "c"]),
        "tags": pa.array([[1, 2], [], None], pa.list_(pa.int64())),
        ROW_ID: pa.array([9, 8, 7], pa.int64()),
    }), src)
    assert pq.read_table(src).column("x").null_count == 0   # a real NaN went in
    info, store, dest = _ingest(tmp_path, src)
    col = {c.name: c for c in info.columns}
    assert info.n_rows == 3 and np.array_equal(_row_ids(dest), np.arange(3))
    assert col["x"].n_missing == 1 and col["x"].dtype == "numeric"
    assert col["tags"].physical_type == "VARCHAR"
    assert f"{ROW_ID} (source)" in col
    assert any("already has a column" in w for w in info.warnings)
    assert store.window(0, 3, columns=["x", "s"])["rows"] == [[1.0, "a"], [None, None],
                                                              [3.0, "c"]]
    assert {s["name"]: s for s in store.summaries(["x"])}["x"]["mean"] == 2.0


def test_excel_source_keeps_text_ids_and_reads_the_first_sheet(tmp_path):
    pytest.importorskip("openpyxl")
    src = tmp_path / "book.xlsx"
    frame = pd.DataFrame({
        "id": ["007", "010", "3"],
        "x": [1.5, np.nan, 3.0],
        "when": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
        "flag": [True, False, True],
        "mixed": [1, "a", 2.5],
    })
    with pd.ExcelWriter(src) as writer:
        frame.to_excel(writer, sheet_name="data", index=False)
        frame.head(1).to_excel(writer, sheet_name="notes", index=False)
    info, store, _ = _ingest(tmp_path, src)
    col = {c.name: c for c in info.columns}
    assert info.n_rows == 3 and list(col) == ["id", "x", "when", "flag", "mixed"]
    assert col["id"].physical_type == "VARCHAR"
    assert col["x"].n_missing == 1 and col["when"].dtype == "datetime"
    assert col["flag"].dtype == "boolean"
    assert store.window(0, 3, columns=["id", "mixed"])["rows"] == [
        ["007", "1"], ["010", "a"], ["3", "2.5"]]
    assert any("2 sheets" in w for w in info.warnings)
    assert any('"mixed"' in w for w in info.warnings)


def test_unsupported_and_empty_files_are_refused_by_name(tmp_path):
    with pytest.raises(ValueError, match="CSV.*Parquet.*Excel"):
        ingest(_write(tmp_path, "data.json", "{}"), tmp_path / "out.parquet")
    with pytest.raises(ValueError, match="empty"):
        ingest(_write(tmp_path, "empty.csv", ""), tmp_path / "out.parquet")
    with pytest.raises(FileNotFoundError):
        ingest(tmp_path / "missing.csv", tmp_path / "out.parquet")


# ── metadata, summaries, histograms ──────────────────────────────────────────

def test_info_matches_the_contract_and_survives_a_lost_sidecar(tmp_path):
    src = SAMPLE_CSVS[0]
    info, store, dest = _ingest(tmp_path, src)
    d = info.to_dict()
    assert set(d) == INFO_KEYS and all(set(c) == COLUMN_KEYS for c in d["columns"])
    assert all(c["dtype"] in DTYPES for c in d["columns"])
    assert len(d["fingerprint"]) == 32 and d["source_bytes"] == src.stat().st_size
    assert d["parquet_bytes"] == dest.stat().st_size
    assert store.info().to_dict() == d

    dest.with_name("raw.info.json").unlink()
    again = DataStore(dest, BUDGET).info()
    assert (again.n_rows, again.n_cols) == (info.n_rows, info.n_cols)
    assert [c.to_dict() for c in again.columns] == d["columns"]
    assert again.warnings and "recomputed" in again.warnings[0]


def test_summaries_are_known_answers_and_cached(tmp_path):
    src = _write(tmp_path, "s.csv", "x,g,b,t\n1,a,true,2024-01-01\n2,a,false,2024-01-02\n"
                                    "3,b,true,\n4,a,,2024-01-02\n,c,true,2024-01-03\n")
    info, store, dest = _ingest(tmp_path, src)
    s = {row["name"]: row for row in store.summaries()}
    assert all(set(row) == SUMMARY_KEYS for row in s.values())
    x = s["x"]
    assert (x["n"], x["n_missing"], x["n_unique"]) == (4, 1, 4)
    assert x["mean"] == 2.5 and _close(x["std"], np.std([1, 2, 3, 4], ddof=1))
    assert (x["min"], x["q25"], x["median"], x["q75"], x["max"]) == (1, 1.75, 2.5, 3.25, 4)
    assert x["top"] is None
    assert s["g"]["top"] == [{"value": "a", "count": 3}, {"value": "b", "count": 1},
                             {"value": "c", "count": 1}]
    assert s["b"]["dtype"] == "boolean" and s["b"]["mean"] == 0.75
    assert s["b"]["top"] == [{"value": True, "count": 3}, {"value": False, "count": 1}]
    assert s["t"]["dtype"] == "datetime" and s["t"]["top"][0] == {"value": "2024-01-02",
                                                                  "count": 2}
    assert dest.with_name("raw.summaries.json").exists()
    assert DataStore(dest, BUDGET).summaries(["g", "x"]) == [s["g"], s["x"]]


def test_histograms_count_every_finite_value_once(tmp_path):
    rows = [f"{i},{i % 5 + 1},7,{'a' if i % 2 else 'b'}" for i in range(10)]
    src = _write(tmp_path, "hist.csv", "x,k,c,g\n" + "\n".join(rows) + "\n,,,a\n,,,a\n")
    info, store, _ = _ingest(tmp_path, src)

    h = store.histogram("x", bins=5)
    assert set(h) == {"column", "edges", "counts", "n_missing"}
    assert h["edges"] == pytest.approx([0, 1.8, 3.6, 5.4, 7.2, 9])
    assert h["counts"] == [2, 2, 2, 2, 2] and h["n_missing"] == 2
    assert sum(h["counts"]) + h["n_missing"] == info.n_rows

    k = store.histogram("k", bins=30)       # one bin per integer, centered
    assert k["edges"] == [0.5, 1.5, 2.5, 3.5, 4.5, 5.5] and k["counts"] == [2] * 5
    c = store.histogram("c")
    assert c["edges"] == [6.5, 7.5] and c["counts"] == [10]
    with pytest.raises(ValueError, match="categorical"):
        store.histogram("g")
    with pytest.raises(ValueError):
        store.histogram("x", bins=0)
    with pytest.raises(UnknownColumn):
        store.histogram("nope")

    # And on a real fixture: the invariant for every numeric column.
    info, store, _ = _ingest(tmp_path / "labs", REPO_ROOT / "turbotab" / "sample_data" /
                             "clinical_labs.csv")
    for col in info.columns:
        if col.dtype in ("numeric", "integer"):
            h = store.histogram(col.name)
            assert sum(h["counts"]) == info.n_rows - col.n_missing == info.n_rows - h["n_missing"]
            assert len(h["edges"]) == len(h["counts"]) + 1


# ── materialize under a budget ───────────────────────────────────────────────

def test_materialize_refuses_before_reading_when_over_budget(tmp_path):
    src = _write(tmp_path, "m.csv", "id,x,g\n007,1.5,a\n010,,b\n3,3.5,a\n")
    info, _, dest = _ingest(tmp_path, src)

    tiny = DataStore(dest, memory_budget_bytes=100)

    def no_reading():
        raise AssertionError("materialize read data before checking the budget")

    tiny._cursor = no_reading  # type: ignore[method-assign]
    with pytest.raises(MemoryBudgetExceeded) as caught:
        tiny.materialize()
    exc = caught.value
    assert exc.budget_bytes == 100 and exc.estimate_bytes > 100
    assert "GB" in str(exc) and "run TurboTab on a server with more memory" in str(exc)

    store = DataStore(dest, BUDGET)
    assert store.estimate_bytes(["x"], n_rows=1) < store.estimate_bytes(["x"]) < \
        store.estimate_bytes()
    frame = store.materialize()
    assert frame.index.name == "row_id" and list(frame.index) == [0, 1, 2]
    assert list(frame.columns) == ["id", "x", "g"]
    assert list(frame["id"]) == ["007", "010", "3"]
    assert frame["x"].dtype == np.float64 and math.isnan(frame["x"].iloc[1])
    picked = store.materialize(["x"], row_ids=np.array([2, 0]))
    assert list(picked.index) == [2, 0] and list(picked["x"]) == [3.5, 1.5]
    with pytest.raises(ValueError, match="outside"):
        store.materialize(row_ids=[0, 3])


def test_sample_is_deterministic(tmp_path):
    src = _write(tmp_path, "d.csv", "v\n" + "\n".join(str(i) for i in range(100)) + "\n")
    _, store, _ = _ingest(tmp_path, src)
    a, b = store.sample(10, seed=1), store.sample(10, seed=1)
    assert a.equals(b) and len(a) == 10 and a.index.is_unique
    assert list(a["v"]) == list(a.index)            # the rows are the rows they claim
    assert not store.sample(10, seed=2).equals(a)
    assert len(store.sample(1_000)) == 100


def test_the_store_answers_from_many_threads_at_once(tmp_path):
    _, store, _ = _ingest(tmp_path, REPO_ROOT / "turbotab" / "sample_data" /
                          "clinical_labs.csv")
    numeric = [c.name for c in store.info().columns if c.dtype == "numeric"]
    expected_window = store.window(10, 20)
    expected_summary = store.summaries(numeric[:3])

    def work(i: int):
        kind = i % 4
        if kind == 0:
            return store.window(10, 20) == expected_window
        if kind == 1:
            return store.summaries(numeric[:3]) == expected_summary
        if kind == 2:
            return sum(store.histogram(numeric[0])["counts"]) > 0
        return len(store.materialize(numeric[:2], row_ids=[1, 2, 3])) == 3

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert all(pool.map(work, range(64)))


def test_progress_rises_to_one_and_a_raising_callback_cancels_cleanly(tmp_path):
    src = REPO_ROOT / "turbotab" / "sample_data" / "clinical_labs.csv"
    seen: list[float] = []
    ingest(src, tmp_path / "ok" / "raw.parquet", progress=lambda f, m: seen.append(f))
    assert seen[-1] == 1.0 and seen == sorted(seen)

    class Cancelled(Exception):
        pass

    def cancel(fraction: float, message: str) -> None:
        if fraction >= 0.1:
            raise Cancelled()

    dest = tmp_path / "cancelled" / "raw.parquet"
    with pytest.raises(Cancelled):
        ingest(src, dest, progress=cancel)
    assert not dest.exists()
    assert [p.name for p in dest.parent.iterdir()] == []


# ── scale ────────────────────────────────────────────────────────────────────

def test_a_wide_table_ingests_and_summarizes(tmp_path):
    """200 x 20,000 floats. Measured ~15 s here (M2 owns the wide benchmark)."""
    n_rows, n_cols = 200, 20_000
    data = np.random.default_rng(7).normal(size=(n_rows, n_cols))
    path = tmp_path / "wide.csv"
    np.savetxt(path, data, delimiter=",", fmt="%.6g", comments="",
               header=",".join(f"f{j}" for j in range(n_cols)))

    t0 = time.perf_counter()
    info, store, _ = _ingest(tmp_path, path)
    t1 = time.perf_counter()
    summaries = store.summaries()
    t2 = time.perf_counter()
    window = store.window(0, 50)
    t3 = time.perf_counter()
    print(f"\nMEASURE wide 200x20000 csv={path.stat().st_size / 1e6:.1f}MB "
          f"ingest={t1 - t0:.1f}s summaries={t2 - t1:.1f}s window50={t3 - t2:.2f}s")

    assert (info.n_rows, info.n_cols) == (n_rows, n_cols)
    assert len(summaries) == n_cols and len(window["rows"]) == 50
    assert len(window["rows"][0]) == n_cols
    probe = summaries[12_345]
    assert probe["name"] == "f12345"
    assert math.isclose(probe["mean"], float(data[:, 12_345].mean()), abs_tol=1e-4)
    assert t2 - t0 < 60


def _nhanes_path() -> Path | None:
    candidates = [os.environ.get("TURBOTAB_NHANES_CSV"), REPO_ROOT / "_tt_tmp_nhanes.csv"]
    try:  # a linked worktree: the export lives in the main checkout
        out = subprocess.run(["git", "-C", str(REPO_ROOT), "worktree", "list", "--porcelain"],
                             capture_output=True, text=True, timeout=10).stdout
        main = out.splitlines()[0].split(" ", 1)[1] if out else None
        if main:
            candidates.append(Path(main) / "_tt_tmp_nhanes.csv")
    except (OSError, subprocess.SubprocessError, IndexError):
        pass
    for candidate in candidates:
        if candidate and Path(candidate).is_file() and os.access(candidate, os.R_OK):
            return Path(candidate)
    return None


def test_the_real_nhanes_export_ingests(tmp_path):
    source = _nhanes_path()
    if source is None:
        pytest.skip("the NHANES export (_tt_tmp_nhanes.csv) is not on this machine")
    t0 = time.perf_counter()
    info, store, _ = _ingest(tmp_path, source)
    print(f"\nMEASURE nhanes ingest={time.perf_counter() - t0:.2f}s "
          f"({info.n_rows}x{info.n_cols}, {source.stat().st_size / 1e6:.1f}MB)")
    assert info.n_rows == 21_849 and info.n_cols == 29
    assert info.n_rows == len(pd.read_csv(source))
    assert len(store.summaries()) == 29
