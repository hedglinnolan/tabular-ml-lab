"""Ingest once, columnar; answer UI questions with queries (BLUEPRINT §2).

``ingest`` turns a CSV/TSV/TXT, Parquet or Excel file into one Parquet file
with an added ``__row_id`` BIGINT column, ``0..n-1`` in file order — the stable
row identity everything downstream keys on — plus a small JSON sidecar
(``raw.info.json``) holding the DatasetInfo.

``DataStore`` answers row windows, column summaries and histograms as DuckDB
queries over that Parquet file, and materializes pandas frames for modeling
only under a memory budget. A wide table (more columns than one DuckDB batch,
M2_CONTRACT §5) is summarized and counted through Arrow instead, and frames are
materialized through pyarrow: DuckDB pays per column per statement, seconds at
20,000 columns. The two paths give the same answers (test_wide_data). ``__row_id`` never appears in a column list, a
window or a summary; it comes back only as the index of a materialized frame.
"""
from __future__ import annotations

import datetime as _dt
import decimal
import hashlib
import json
import math
import os
import shutil
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import duckdb
import numpy as np
import pandas as pd

ROW_ID = "__row_id"
INDEX_NAME = "row_id"
SIDECAR_VERSION = 1

CSV_SUFFIXES = (".csv", ".tsv", ".txt")
COMPRESSED_SUFFIXES = (".gz", ".zst")
PARQUET_SUFFIXES = (".parquet", ".pq")
EXCEL_SUFFIXES = (".xlsx", ".xls")
SUPPORTED_TYPES = "CSV, TSV or TXT (optionally .gz), Parquet, or Excel (.xlsx, .xls)"

# Read as missing in delimited text AND in Excel cells (one list, so a CSV and an xlsx copy of one
# table agree on what is missing: audit MA-17): pandas' default NA tokens, less "None" (a real
# answer on a questionnaire) and the rare "-1.#IND"-style spellings.
NULL_TOKENS = ("", "NA", "N/A", "n/a", "NaN", "nan", "NULL", "null", "#N/A", "<NA>")

# Beyond 2**53 a double cannot hold every integer: 9007199254740993 is stored as ...992.
EXACT_INT = 2 ** 53

SNIFF_SAMPLE_ROWS = 20_480
BATCH_COLUMNS = 200            # aggregate queries touch at most this many columns
SAMPLE_VALUES = 5              # ColumnInfo.sample length
SAMPLE_HEAD_ROWS = 50          # ColumnInfo.sample is drawn from the first rows
TOP_K = 5
EXACT_MAX_ROWS = 2_000_000     # above this, n_unique and quartiles are approximate
EXACT_TOP_MAX_UNIQUE = 1_000   # above this, approx_top_k picks the top values
                               # (their counts are still exact)
MAX_WINDOW_ROWS = 10_000
MAX_HISTOGRAM_BINS = 1_000
PEAK_FACTOR = 2                # materializing holds the Arrow buffers and the frame at once
PY_STR_OVERHEAD = 49           # bytes of a CPython str header

# Wide tables (M2_CONTRACT §5). DuckDB pays per column per statement — binding, planning and
# one aggregate state per column — so 20,000 columns cost seconds a query however few rows
# there are. Past one batch of columns, per-column statistics and materializing go through
# Arrow instead, a chunk of whole columns at a time; DuckDB keeps the tall tables it is fast on.
ARROW_MAX_ROWS = 100_000       # above this, per-column statistics stay in DuckDB
ARROW_CHUNK_CELLS = 10_000_000 # cells (rows × columns) the Arrow paths hold at once
ROW_GROUP_CELLS = 10_000_000   # a wide table's Parquet row groups hold about this many cells
DUCKDB_ROW_GROUP = 122_880     # DuckDB's default row group, in rows
DUCKDB_VECTOR = 2_048          # DuckDB writes row groups in whole vectors of rows
DUCKDB_MAX_LINE = 2_097_152    # DuckDB's default max_line_size, in bytes
FIELD_BYTES = 64               # room per field when a line must hold very many fields

_INT_TYPES = {"TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT", "UTINYINT",
              "USMALLINT", "UINTEGER", "UBIGINT", "UHUGEINT", "INT1", "INT2", "INT4",
              "INT8", "INT128"}
_FLOAT_TYPES = {"FLOAT", "DOUBLE", "REAL", "FLOAT4", "FLOAT8"}

ProgressFn = Callable[[float, str], None]


# ── contract dataclasses ─────────────────────────────────────────────────────

@dataclass
class ColumnInfo:
    name: str
    dtype: str           # numeric | integer | boolean | categorical | datetime | text
    physical_type: str   # the DuckDB type in the Parquet file, e.g. "BIGINT"
    n_missing: int
    n_unique: int
    sample: list[Any] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "dtype": self.dtype, "physical_type": self.physical_type,
                "n_missing": int(self.n_missing), "n_unique": int(self.n_unique),
                "sample": [json_safe(v) for v in self.sample]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ColumnInfo":
        return cls(name=d["name"], dtype=d["dtype"], physical_type=d["physical_type"],
                   n_missing=int(d["n_missing"]), n_unique=int(d["n_unique"]),
                   sample=list(d.get("sample") or []))


@dataclass
class DatasetInfo:
    n_rows: int
    n_cols: int
    columns: list[ColumnInfo]
    source_bytes: int
    parquet_bytes: int
    ingest_seconds: float
    fingerprint: str
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"n_rows": int(self.n_rows), "n_cols": int(self.n_cols),
                "columns": [c.to_dict() for c in self.columns],
                "source_bytes": int(self.source_bytes),
                "parquet_bytes": int(self.parquet_bytes),
                "ingest_seconds": float(self.ingest_seconds),
                "fingerprint": self.fingerprint, "warnings": list(self.warnings)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DatasetInfo":
        return cls(n_rows=int(d["n_rows"]), n_cols=int(d["n_cols"]),
                   columns=[ColumnInfo.from_dict(c) for c in d["columns"]],
                   source_bytes=int(d["source_bytes"]), parquet_bytes=int(d["parquet_bytes"]),
                   ingest_seconds=float(d["ingest_seconds"]), fingerprint=str(d["fingerprint"]),
                   warnings=list(d.get("warnings") or []))


def _gb(n: int) -> str:
    value = n / 1e9
    return f"{value:.1f}" if value >= 1 else f"{value:.2f}"


class MemoryBudgetExceeded(Exception):
    """Materializing would need more memory than TurboTab may use on this machine."""

    def __init__(self, estimate_bytes: int, budget_bytes: int):
        super().__init__(estimate_bytes, budget_bytes)
        self.estimate_bytes = int(estimate_bytes)
        self.budget_bytes = int(budget_bytes)

    def __str__(self) -> str:
        return (f"this dataset needs about {_gb(self.estimate_bytes)} GB but TurboTab's "
                f"budget on this machine is {_gb(self.budget_bytes)} GB — run TurboTab on "
                "a server with more memory")


class UnknownColumn(KeyError, ValueError):
    """A requested column is not in the dataset (``__row_id`` never is)."""

    def __init__(self, column: str):
        super().__init__(column)
        self.column = column

    def __str__(self) -> str:
        return f"no column named {self.column!r} in this dataset"


# ── small helpers ────────────────────────────────────────────────────────────

def _lit(value: str | os.PathLike[str]) -> str:
    """A SQL string literal."""
    return "'" + str(value).replace("'", "''") + "'"


def _ident(name: str) -> str:
    """A quoted SQL identifier; any column name survives it."""
    return '"' + name.replace('"', '""') + '"'


def _base_type(physical: str) -> str:
    return physical.upper().split("(")[0].strip()


def _is_int(physical: str) -> bool:
    return _base_type(physical) in _INT_TYPES


def _is_float(physical: str) -> bool:
    return _base_type(physical) in _FLOAT_TYPES


def _is_decimal(physical: str) -> bool:
    return _base_type(physical) in ("DECIMAL", "NUMERIC")


def _is_temporal(physical: str) -> bool:
    base = _base_type(physical)
    return base.startswith("DATE") or base.startswith("TIMESTAMP") or base.startswith("TIME")


def _is_nested(physical: str) -> bool:
    upper = physical.upper()
    return (upper.endswith("]") or upper.startswith(("STRUCT", "MAP", "UNION", "LIST"))
            or _base_type(physical) in ("BLOB", "BIT", "INTERVAL", "VARIANT"))


def logical_dtype(physical: str, n_unique: int, n_rows: int) -> str:
    """The contract's logical dtype for a DuckDB physical type."""
    if _base_type(physical) == "BOOLEAN":
        return "boolean"
    if _is_int(physical):
        return "integer"
    if _is_float(physical) or _is_decimal(physical):
        return "numeric"
    if _is_temporal(physical):
        return "datetime"
    return "categorical" if n_unique <= max(50, 0.05 * n_rows) else "text"


def json_safe(value: Any) -> Any:
    """A value the JSON encoder (and a browser) can take without surprise."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        # A browser reads every JSON number as a double, so an integer past 2**53 would arrive
        # changed (an identifier off by one); it travels as its exact digits instead.
        return int(value) if abs(int(value)) <= EXACT_INT else str(int(value))
    if isinstance(value, (float, np.floating)):
        f = float(value)
        return f if math.isfinite(f) else None
    if isinstance(value, str):
        return value
    if isinstance(value, decimal.Decimal):
        f = float(value)
        return f if math.isfinite(f) else None
    if value is pd.NaT or value is pd.NA:
        return None
    if isinstance(value, (_dt.datetime, _dt.date, _dt.time)):
        return value.isoformat()
    if isinstance(value, np.datetime64):
        return None if np.isnat(value) else pd.Timestamp(value).isoformat()
    if isinstance(value, (_dt.timedelta, np.timedelta64)):
        return str(pd.Timedelta(value))
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).hex()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    return str(value)


def _stat(value: Any) -> Any:
    """A summary statistic for JSON: a number always (json_safe sends an integer past 2**53 as
    its exact digits, which suits an identifier's value, not a column's minimum)."""
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)) \
            and abs(int(value)) > EXACT_INT:
        return float(value)
    return json_safe(value)


def _chunks(items: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _connect(temp_dir: Path | None = None, *,
             metadata_cache: bool = False) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    # File order is row identity: __row_id is numbered by an order-preserving
    # scan, and test_datastore proves it on a file DuckDB reads in parallel.
    con.execute("SET preserve_insertion_order = true")
    if metadata_cache:  # only for a file that no longer changes (the store's)
        con.execute("SET parquet_metadata_cache = true")
    if temp_dir is not None:
        con.execute(f"SET temp_directory = {_lit(temp_dir)}")
    return con


def fingerprint_file(path: Path, stop: threading.Event | None = None) -> str:
    """blake2b of the file's bytes (hex, 32 characters), streamed."""
    digest = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            if stop is not None and stop.is_set():
                return ""
            digest.update(chunk)
    return digest.hexdigest()


def pandas_style_names(raw: Sequence[Any]) -> tuple[list[str], list[str]]:
    """Header cells → unique column names, the way ``pandas.read_csv`` makes them.

    A blank header becomes ``"Unnamed: <i>"`` and a repeat of ``"x"`` becomes
    ``"x.1"``, ``"x.2"``, … — the renaming signature the legacy domain packs
    already recognize. Returns the names and one warning per rename.
    """
    names: list[str] = []
    notes: list[str] = []
    for i, cell in enumerate(raw):
        if cell is None or (isinstance(cell, float) and math.isnan(cell)):
            text = ""
        else:
            text = str(cell)
        if text == "":
            names.append(f"Unnamed: {i}")
            notes.append(f"column {i + 1} has no name in the header; it is called "
                         f"\"Unnamed: {i}\"")
        else:
            names.append(text)
    counts: dict[str, int] = {}
    for i, col in enumerate(names):
        original = col
        cur = counts.get(col, 0)
        while cur > 0:
            counts[col] = cur + 1
            col = f"{col}.{cur}"
            cur = counts.get(col, 0)
        if col != original:
            notes.append(f"column {i + 1} repeats the header \"{original}\"; it is called "
                         f"\"{col}\"")
        names[i] = col
        counts[col] = cur + 1
    return names, notes


# ── progress plumbing ────────────────────────────────────────────────────────

class _Progress:
    """Monotone progress; the caller's callback may raise to cancel the ingest."""

    def __init__(self, fn: ProgressFn | None):
        self.fn = fn
        self.last = 0.0
        self._lock = threading.Lock()

    def __call__(self, fraction: float, message: str) -> None:
        if self.fn is None:
            return
        with self._lock:  # held across the call, so two threads cannot reorder it
            fraction = min(1.0, max(self.last, float(fraction)))
            self.last = fraction
            self.fn(fraction, message)


def _execute(con: duckdb.DuckDBPyConnection, sql: str, report: _Progress,
             lo: float, hi: float, message: str) -> None:
    """Run one long statement, mapping DuckDB's own progress onto [lo, hi].

    The poller thread calls ``report``; if the callback raises (a cancelled
    job), the statement is interrupted and that exception is re-raised here.
    """
    if report.fn is None:
        con.execute(sql)
        return
    report(lo, message)
    con.execute("SET enable_progress_bar = true")
    con.execute("SET enable_progress_bar_print = false")
    done = threading.Event()
    failure: list[BaseException] = []

    def poll() -> None:
        while not done.wait(0.2):
            try:
                pct = con.query_progress()
                if pct is not None and pct >= 0:
                    report(lo + (hi - lo) * min(pct, 100.0) / 100.0, message)
            except BaseException as exc:  # noqa: BLE001 — carried to the caller
                failure.append(exc)
                try:
                    con.interrupt()
                except Exception:
                    pass
                return

    poller = threading.Thread(target=poll, name="turbotab-ingest-progress", daemon=True)
    poller.start()
    try:
        con.execute(sql)
    except duckdb.InterruptException:
        if failure:
            raise failure[0] from None
        raise
    finally:
        done.set()
        poller.join()
    if failure:
        raise failure[0]
    report(hi, message)


# ── ingest ───────────────────────────────────────────────────────────────────

def _source_kind(source: Path) -> str:
    name = source.name.lower()
    for comp in COMPRESSED_SUFFIXES:
        if name.endswith(comp):
            name = name[: -len(comp)]
            break
    else:
        comp = ""
    if name.endswith(CSV_SUFFIXES):
        return "csv"
    if not comp and name.endswith(PARQUET_SUFFIXES):
        return "parquet"
    if not comp and name.endswith(EXCEL_SUFFIXES):
        return "excel"
    raise ValueError(f"TurboTab cannot read {source.name!r}; it reads {SUPPORTED_TYPES}")


def _select_expr(name: str, physical: str, out: str | None = None,
                 decimal_comma: bool = False) -> str:
    """Select one source column for the Parquet file (nested values become text).

    Plain column references on purpose: DuckDB's planner cost per CASE
    expression grows with the statement's width (15 s for 20,000 columns), so
    NaN → NULL is a second, targeted pass (``_null_the_nans``) that only runs
    for the float columns that actually hold a NaN. ``decimal_comma``: the
    text column writes its decimals with a comma (``2000,5``; :func:`_spellings`
    found every value reads so), and is read as the numbers it holds.
    """
    expr = _ident(name)
    if decimal_comma:
        expr = decimal_comma_sql(expr)
    elif _is_nested(physical):
        expr = f"CAST({expr} AS VARCHAR)"
    return f"{expr} AS {_ident(out if out is not None else name)}"


# A number written with a decimal comma: "2000,5", "-0,25", "12", or with dots between thousands,
# "1.234,5". And a number a comma could split into thousands: "1,234", "12,045,300".
DECIMAL_COMMA = r"[+-]?(\d+(,\d+)?|\d{1,3}(\.\d{3})+(,\d+)?)"
THOUSANDS_COMMA = r"[+-]?[1-9]\d{0,2}(,\d{3})+"


def decimal_comma_sql(x: str) -> str:
    """SQL reading decimal-comma text as a DOUBLE: thousands dots out, the comma becomes the point."""
    return f"TRY_CAST(replace(replace(trim(CAST({x} AS VARCHAR)), '.', ''), ',', '.') AS DOUBLE)"


def row_group_rows(n_columns: int, n_rows: int | None = None) -> int | None:
    """Rows per Parquet row group for a table this wide; None keeps DuckDB's default.

    DuckDB buffers a whole row group before writing it, so 122,880 rows × 20,000 columns
    would hold ~20 GB; and a row window or a sample reads whole row groups. A wide table's
    groups hold about ROW_GROUP_CELLS cells instead, in whole DuckDB vectors (2,048 rows).
    A table of a few groups' rows (``n_rows``, known or estimated) keeps the default: DuckDB
    would otherwise flush each reader thread's rows as a group of their own, and a 500-row
    table would get four groups — four times the metadata of 20,000 column chunks.
    """
    if n_columns * DUCKDB_ROW_GROUP <= ROW_GROUP_CELLS:
        return None
    rows = max(DUCKDB_VECTOR, ROW_GROUP_CELLS // max(1, n_columns) // DUCKDB_VECTOR * DUCKDB_VECTOR)
    if n_rows is not None and n_rows <= 2 * rows:
        return None
    return rows


def _parquet_options(n_columns: int, n_rows: int | None = None) -> str:
    rows = row_group_rows(n_columns, n_rows)
    group = f", ROW_GROUP_SIZE {rows}" if rows is not None else ""
    return f"(FORMAT parquet, COMPRESSION zstd{group})"


def _copy_sql(select_list: Sequence[str], source_sql: str, dest: Path,
              n_rows: int | None = None) -> str:
    cols = ", ".join([*select_list, f"CAST(row_number() OVER () - 1 AS BIGINT) AS {ROW_ID}"])
    return (f"COPY (SELECT {cols} FROM {source_sql}) TO {_lit(dest)} "
            f"{_parquet_options(len(select_list) + 1, n_rows)}")


def estimated_csv_rows(source: Path) -> int | None:
    """Data rows in a delimited file, guessed from its size and its header line's length."""
    if source.name.lower().endswith(COMPRESSED_SUFFIXES):
        return None
    try:
        with open(source, "rb") as fh:
            header = len(fh.readline(MAX_HEADER_BYTES))
        return max(0, source.stat().st_size // max(1, header) - 1)
    except OSError:
        return None


class _EncodingError(Exception):
    pass


@dataclass
class _Dialect:
    delim: str
    quote: str
    escape: str
    comment: str
    skip: int
    types: list[str]
    dateformat: str | None
    timestampformat: str | None
    max_line: int | None = None  # bytes; None keeps DuckDB's default


WIDE_HEADER_FIELDS = 1_000
MAX_HEADER_BYTES = 1 << 30


def _wide_delimiter(source: Path) -> str | None:
    """The delimiter of a very wide file, read off its first line; else None.

    DuckDB's sniffer tries every quote/escape pairing, and at 20,000 columns
    that costs ~4 s against ~0.4 s once the dialect is pinned. Only a header
    with at least WIDE_HEADER_FIELDS separators is trusted to name it.
    """
    if source.name.lower().endswith(COMPRESSED_SUFFIXES):
        return None
    with open(source, "rb") as fh:
        head = fh.read(1 << 20)
    first = head.split(b"\n", 1)[0]
    counts = {d: first.count(d.encode()) for d in (",", "\t", ";", "|")}
    delim, n = max(counts.items(), key=lambda kv: kv[1])
    return delim if n >= WIDE_HEADER_FIELDS else None


def max_line_size(source: Path) -> int | None:
    """A ``max_line_size`` for a file whose lines DuckDB's default (2 MB) cannot hold; else None.

    Read off the header line: twice its length, and at least FIELD_BYTES per field, so data
    rows longer than the header (numbers written to many digits) still fit. Past ~32,000
    fields the default is too small; at 150,000 the header alone is over 2 MB.
    """
    name = source.name.lower()
    try:
        if name.endswith(".gz"):
            import gzip
            fh: Any = gzip.open(source, "rb")
        elif name.endswith(COMPRESSED_SUFFIXES):
            return None
        else:
            fh = open(source, "rb")
        with fh:
            line = fh.readline(MAX_HEADER_BYTES)
    except (OSError, EOFError):
        return None
    fields = 1 + max(line.count(d) for d in (b",", b"\t", b";", b"|"))
    need = max(2 * len(line), fields * FIELD_BYTES)
    return need if need > DUCKDB_MAX_LINE else None


def _line_options(max_line: int | None) -> str:
    """read_csv options for very long lines (the buffer must hold at least one line)."""
    if max_line is None:
        return ""
    return f", max_line_size = {int(max_line)}, buffer_size = {2 * int(max_line)}"


def _sniff_csv(con: duckdb.DuckDBPyConnection, source: Path, *, encoding: str,
               sample_size: int, delim: str | None = None,
               null_padding: bool = False, max_line: int | None = None) -> _Dialect:
    """DuckDB's sniffer, with the header forced on (as every CSV reader defaults)."""
    nulls = "[" + ", ".join(_lit(t) for t in NULL_TOKENS) + "]"
    pinned = (f", delim = {_lit(delim)}, quote = '\"', escape = '\"'"
              if delim is not None else "")
    if null_padding:
        pinned += ", null_padding = true"
    pinned += _line_options(max_line)
    row = con.execute(
        f"SELECT Delimiter, Quote, Escape, Comment, SkipRows, Columns, DateFormat, "
        f"TimestampFormat FROM sniff_csv({_lit(source)}, header = true, "
        f"sample_size = {int(sample_size)}, nullstr = {nulls}, "
        f"encoding = {_lit(encoding)}{pinned})"
    ).fetchone()
    delim, quote, escape, comment, skip, columns, dfmt, tfmt = row

    def given(v: Any) -> str:
        return "" if v in (None, "(empty)") else str(v)

    quote = given(quote) or '"'
    escape = given(escape) or quote
    return _Dialect(delim=given(delim) or ",", quote=quote, escape=escape,
                    comment=given(comment), skip=int(skip or 0),
                    types=[str(c["type"]) for c in columns],
                    dateformat=given(dfmt) or None, timestampformat=given(tfmt) or None,
                    max_line=max_line)


def _csv_sql(source: Path, d: _Dialect, *, encoding: str, header: bool,
             columns: dict[str, str] | None = None, all_varchar: bool = False,
             null_padding: bool = False) -> str:
    opts = ["auto_detect = false", f"header = {'true' if header else 'false'}",
            f"delim = {_lit(d.delim)}", f"quote = {_lit(d.quote)}",
            f"escape = {_lit(d.escape)}", f"skip = {d.skip}",
            f"encoding = {_lit(encoding)}",
            "nullstr = [" + ", ".join(_lit(t) for t in NULL_TOKENS) + "]"]
    if d.comment:
        opts.append(f"comment = {_lit(d.comment)}")
    if null_padding:
        opts.append("null_padding = true")
    if d.max_line is not None:
        opts.append(_line_options(d.max_line).lstrip(", "))
    if columns is not None:
        spec = ", ".join(f"{_lit(n)}: {_lit('VARCHAR' if all_varchar else t)}"
                         for n, t in columns.items())
        opts.append("columns = {" + spec + "}")
        if not all_varchar:
            if d.dateformat:
                opts.append(f"dateformat = {_lit(d.dateformat)}")
            if d.timestampformat:
                opts.append(f"timestampformat = {_lit(d.timestampformat)}")
    elif all_varchar:
        opts.append("all_varchar = true")
    return f"read_csv({_lit(source)}, {', '.join(opts)})"


def _raw_header(con: duckdb.DuckDBPyConnection, source: Path, d: _Dialect,
                encoding: str, n_expected: int) -> list[Any]:
    """The header line's cells exactly as written (blank cells come back None)."""
    placeholders = {f"c{i}": "VARCHAR" for i in range(n_expected)}
    sql = _csv_sql(source, d, encoding=encoding, header=False, columns=placeholders,
                   all_varchar=True, null_padding=True)
    row = con.execute(f"SELECT * FROM {sql} LIMIT 1").fetchone()
    return list(row) if row is not None else [None] * n_expected


def _is_encoding_error(exc: BaseException) -> bool:
    return "invalid unicode" in str(exc).lower()


_SHORT_ROWS_NOTE = ("some rows have fewer fields than the header; the missing fields were "
                    "read as missing values")


def _first_line_splits(source: Path) -> bool:
    """Whether the first line holds a common delimiter at all."""
    if source.name.lower().endswith(COMPRESSED_SUFFIXES):
        return False
    with open(source, "rb") as fh:
        first = fh.read(1 << 16).split(b"\n", 1)[0]
    return any(d in first for d in (b",", b"\t", b";", b"|"))


def _ingest_csv(con: duckdb.DuckDBPyConnection, source: Path, tmp: Path,
                warnings: list[str], report: _Progress) -> None:
    for encoding in ("utf-8", "latin-1"):
        try:
            _ingest_csv_as(con, source, tmp, warnings, report, encoding)
            return
        except _EncodingError:
            if encoding == "latin-1":
                raise ValueError(f"could not decode {source.name!r} as UTF-8 or Latin-1")
            warnings.append("the file is not valid UTF-8; it was read as Latin-1 "
                            "(ISO-8859-1), so check that accented characters look right")


def _ingest_csv_as(con: duckdb.DuckDBPyConnection, source: Path, tmp: Path,
                   warnings: list[str], report: _Progress, encoding: str) -> None:
    null_padding = False
    header_noted = False
    names: list[str] | None = None
    last_error: BaseException | None = None
    dialect: _Dialect | None = None
    attempts = (("sniff", SNIFF_SAMPLE_ROWS), ("sniff", -1), ("text", 0))
    wide_delim = _wide_delimiter(source)
    max_line = max_line_size(source)
    step = 0
    while step < len(attempts):
        mode, sample_size = attempts[step]
        try:
            if mode == "sniff":
                report(0.02, "Detecting the file's layout and column types")
                dialect = _sniff_csv(con, source, encoding=encoding, sample_size=sample_size,
                                     delim=wide_delim if step == 0 else None,
                                     null_padding=null_padding, max_line=max_line)
                if len(dialect.types) == 1 and not null_padding and _first_line_splits(source):
                    # A short row makes every real delimiter look inconsistent,
                    # and the sniffer then settles on "one column" without a
                    # word. Try again allowing short rows.
                    padded = _sniff_csv(con, source, encoding=encoding,
                                        sample_size=sample_size, null_padding=True,
                                        max_line=max_line)
                    if len(padded.types) > 1:
                        dialect, null_padding = padded, True
                        warnings.append(_SHORT_ROWS_NOTE)
                types = list(dialect.types)
            elif dialect is None:
                raise ValueError(f"could not detect the layout of {source.name!r}: {last_error}")
            else:
                types = ["VARCHAR"] * len(dialect.types)
            if names is None or len(names) != len(types):
                raw = _raw_header(con, source, dialect, encoding, len(types))
                names, rename_notes = pandas_style_names(raw)
                if not header_noted:
                    warnings.extend(rename_notes)
                    header_noted = True
            # What the sniffer's types cannot say, read from every row: dates that read two
            # ways stay text, and decimal commas are read as the numbers they write.
            text_notes, commas = (_spellings(con, source, dialect, encoding, names, types,
                                             null_padding, report)
                                  if mode == "sniff" else ([], set()))

            def copy(types: list[str]) -> None:
                columns = dict(zip(names, types))
                src = _csv_sql(source, dialect, encoding=encoding, header=True,
                               columns=columns, null_padding=null_padding)
                select = [_select_expr(n, t, decimal_comma=n in commas) for n, t in columns.items()]
                _execute(con, _copy_sql(select, src, tmp, estimated_csv_rows(source)),
                         report, 0.1, 0.7,
                         "Writing the columnar copy")

            copy(types)
            # DuckDB's sniffer already refuses to read "007" as an integer, but
            # only within the rows it sampled. Past the sample, check every row.
            n_rows = _count_rows(con, tmp)
            if sample_size != -1 and n_rows > SNIFF_SAMPLE_ROWS // 2:
                zeros = _protect_leading_zeros(con, source, dialect, encoding, names,
                                               types, null_padding, report)
                if zeros:
                    text_notes += zeros
                    copy(types)
            # An integer past 2**53 read as a double is silently another integer (C14).
            big = _protect_big_integers(con, tmp, source, dialect, encoding, names, types,
                                        commas, null_padding)
            if big:
                text_notes += big
                copy(types)
        except duckdb.InterruptException:
            raise
        except duckdb.Error as exc:
            if _is_encoding_error(exc) and encoding != "latin-1":
                raise _EncodingError(str(exc)) from exc
            last_error = exc
            if "expected number of columns" in str(exc).lower() and not null_padding:
                null_padding = True
                warnings.append(_SHORT_ROWS_NOTE)
                step = 0
                continue
            step += 1
            continue
        warnings.extend(text_notes)
        if step == 1:
            warnings.append(f"column types guessed from the first {SNIFF_SAMPLE_ROWS:,} rows "
                            "did not fit the rest of the file; they were detected again "
                            "from every row")
        elif step == 2:
            first_line = str(last_error).strip().splitlines()[0] if last_error else ""
            warnings.append("column types could not be detected (" + first_line +
                            "); every column was read as text")
        return
    raise ValueError(f"could not read {source.name!r} as delimited text: {last_error}")


def _protect_leading_zeros(con: duckdb.DuckDBPyConnection, source: Path, d: _Dialect,
                           encoding: str, names: list[str], types: list[str],
                           null_padding: bool, report: _Progress) -> list[str]:
    """Integer-typed columns whose raw text has leading zeros ("007") stay text.

    Checked on every row, not the sniffer's sample: DuckDB would otherwise read
    a late "007" as 7 without a word. Rewrites ``types`` in place.
    """
    candidates = [i for i, t in enumerate(types) if _is_int(t)]
    if not candidates:
        return []
    src = _csv_sql(source, d, encoding=encoding, header=True,
                   columns=dict(zip(names, types)), all_varchar=True,
                   null_padding=null_padding)
    notes: list[str] = []
    n_batches = math.ceil(len(candidates) / BATCH_COLUMNS)
    for batch_no, batch in enumerate(_chunks(candidates, BATCH_COLUMNS)):
        # A value that parsed as an integer is digits with optional padding and
        # sign, so "starts with 0 and is longer than one digit" is exactly "has
        # a leading zero". A negative ("-007") is left numeric, as DuckDB's own
        # sniffer leaves it. (FILTER + regexp was quadratic in the column count.)
        aggs = []
        for i in batch:
            c = _ident(names[i])
            bare = f"ltrim({c}, ' +')"
            aggs.append(f"max(CASE WHEN starts_with({bare}, '0') AND length(rtrim({bare})) > 1 "
                        f"THEN {c} END)")
        report(0.7 + 0.1 * batch_no / n_batches,
               "Checking identifier columns for leading zeros")
        row = con.execute(f"SELECT {', '.join(aggs)} FROM {src}").fetchone()
        for i, example in zip(batch, row):
            if example is not None:
                types[i] = "VARCHAR"
                notes.append(f"column \"{names[i]}\" has values with leading zeros "
                             f"(e.g. \"{example.strip()}\"); it is kept as text so "
                             "identifiers are not changed")
    return notes


def _spellings(con: duckdb.DuckDBPyConnection, source: Path, d: _Dialect, encoding: str,
               names: list[str], types: list[str], null_padding: bool,
               report: _Progress) -> tuple[list[str], set[str]]:
    """Readings the sniffer's types cannot make, checked on every row (audit MA-05, MA-16).

    * A DATE or TIMESTAMP column read with a numeric day-month format (``%d/%m/%Y``) whose every
      value the other order reads too (no day above 12) is kept as text: the sniffer picks one
      order without a word, and reading US visit dates day-first moved them all into January.
      ``types`` is rewritten in place; the date-reading repair asks which order is meant.
    * A text column whose every value is a number written with a decimal comma (``2000,5``), at
      least one of them unmistakably so (``22,1`` or ``0,25``, never only ``1,234``, which a
      comma could also split into thousands), is read as the numbers it holds: returned.
    """
    from turbotab.core import dates

    checks: list[tuple[int, str, str | None]] = []  # (column index, kind, the other date order)
    for i, t in enumerate(types):
        base = _base_type(t)
        fmt = d.dateformat if base == "DATE" else d.timestampformat if base.startswith("TIMESTAMP") else None
        other = dates.swapped(fmt)
        if other is not None:
            checks.append((i, "date", other))
        elif base == "VARCHAR":
            checks.append((i, "comma", None))
    if not checks:
        return [], set()
    src = _csv_sql(source, d, encoding=encoding, header=True, columns=dict(zip(names, types)),
                   all_varchar=True, null_padding=null_padding)
    notes: list[str] = []
    commas: set[str] = set()
    for batch in _chunks(checks, BATCH_COLUMNS):
        report(0.06, "Checking how dates and decimals are written")
        aggs: list[str] = []
        for i, kind, other in batch:
            c = f"trim({_ident(names[i])})"
            first = f"first({c}) FILTER (WHERE {c} IS NOT NULL)"
            if kind == "date":
                aggs += [f"count({c})", f"count(try_strptime({c}, {_lit(str(other))}))", first]
            else:
                comma = f"regexp_full_match({c}, '{DECIMAL_COMMA}')"
                plain = f"(contains({c}, ',') AND NOT regexp_full_match({c}, '{THOUSANDS_COMMA}'))"
                aggs += [f"count({c})", f"count_if({comma})", f"count_if({comma} AND contains({c}, ','))",
                         f"count_if({comma} AND {plain})",
                         f"first({c}) FILTER (WHERE {comma} AND {plain})"]
        row = con.execute(f"SELECT {', '.join(aggs)} FROM {src}").fetchone()
        k = 0
        for i, kind, other in batch:
            name = names[i]
            if kind == "date":
                n, both, example = int(row[k] or 0), int(row[k + 1] or 0), row[k + 2]
                k += 3
                if n and both == n:
                    types[i] = "VARCHAR"
                    notes.append(f"column \"{name}\" holds dates such as \"{example}\" that read the "
                                 f"same month-first and day-first (no day is above 12), so it is "
                                 f"kept as text until you say which")
            else:
                n, fits, with_comma, plain, example = row[k:k + 5]
                k += 5
                if n and int(fits or 0) == int(n) and int(with_comma or 0) and int(plain or 0):
                    commas.add(name)
                    notes.append(f"column \"{name}\" writes decimals with a comma (e.g. "
                                 f"\"{example}\"); it was read as the numbers it holds")
    return notes, commas


def _protect_big_integers(con: duckdb.DuckDBPyConnection, parquet: Path, source: Path,
                          d: _Dialect, encoding: str, names: list[str], types: list[str],
                          commas: set[str], null_padding: bool) -> list[str]:
    """Integer-valued columns read as doubles past 2**53 stay text (audit C14).

    A double holds every integer only up to 2**53, so ``9007199254740993`` written in a file comes
    back ``...992``: an identifier silently becomes another. The Parquet footer's statistics name
    the double columns that reach that far; their raw text decides (every value an integer
    literal). Rewrites ``types`` in place.
    """
    import pyarrow.parquet as pq

    meta = pq.read_metadata(parquet)
    index = {meta.schema.column(j).name: j for j in range(meta.num_columns)}
    reach: dict[int, float] = {}
    for i, t in enumerate(types):
        j = index.get(names[i])
        if not _is_float(t) or names[i] in commas or j is None:
            continue
        for g in range(meta.num_row_groups):
            stats = meta.row_group(g).column(j).statistics
            if stats is not None and stats.has_min_max:
                lo, hi = float(stats.min), float(stats.max)
                if math.isfinite(lo) and math.isfinite(hi):
                    reach[i] = max(reach.get(i, 0.0), abs(lo), abs(hi))
    wide = [i for i, v in reach.items() if v >= EXACT_INT]
    if not wide:
        return []
    src = _csv_sql(source, d, encoding=encoding, header=True, columns=dict(zip(names, types)),
                   all_varchar=True, null_padding=null_padding)
    aggs = []
    for i in wide:
        c = f"trim({_ident(names[i])})"
        aggs += [f"count({c})", f"count_if(regexp_full_match({c}, '[+-]?\\d+'))",
                 f"max({c}) FILTER (WHERE length({c}) > 15)"]
    row = con.execute(f"SELECT {', '.join(aggs)} FROM {src}").fetchone()
    notes = []
    for k, i in enumerate(wide):
        n, whole, example = row[3 * k: 3 * k + 3]
        if n and int(whole or 0) == int(n):
            types[i] = "VARCHAR"
            notes.append(f"column \"{names[i]}\" holds integers too long to store exactly as "
                         f"numbers (e.g. \"{example}\"); it is kept as text so they are not changed")
    return notes


def _ingest_parquet(con: duckdb.DuckDBPyConnection, source: Path, tmp: Path,
                    warnings: list[str], report: _Progress) -> None:
    src = f"read_parquet({_lit(source)})"
    described = con.execute(f"DESCRIBE SELECT * FROM {src}").fetchall()
    select = []
    for name, physical, *_ in described:
        name = str(name)
        out = name
        if name == ROW_ID:
            out = f"{ROW_ID} (source)"
            warnings.append(f"the file already has a column named \"{ROW_ID}\"; it is "
                            f"kept as \"{out}\"")
        if _is_nested(str(physical)):
            warnings.append(f"column \"{name}\" holds {physical} values; they are kept "
                            "as text")
        select.append(_select_expr(name, str(physical), out))
    n_rows = int(con.execute(f"SELECT count(*) FROM {src}").fetchone()[0])  # from the footer
    _execute(con, _copy_sql(select, src, tmp, n_rows), report, 0.05, 0.8,
             "Writing the columnar copy")


def _ingest_excel(con: duckdb.DuckDBPyConnection, source: Path, tmp: Path,
                  warnings: list[str], report: _Progress) -> None:
    report(0.05, "Reading the workbook")
    try:
        book = pd.ExcelFile(source)
    except ImportError as exc:
        raise ValueError(f"reading {source.suffix} files needs a package that is not "
                         f"installed ({exc}); save the sheet as .xlsx or CSV") from None
    with book:
        sheets = book.sheet_names
        if not sheets:
            raise ValueError(f"{source.name!r} has no sheets")
        if len(sheets) > 1:
            warnings.append(f"the workbook has {len(sheets)} sheets; only the first, "
                            f"\"{sheets[0]}\", was read")
        # dtype=object: pandas would otherwise run text cells through number
        # inference and turn an identifier "007" into 7. Cells keep the type
        # the workbook gave them and each column is typed below. The missing
        # tokens are the CSV reader's own (NULL_TOKENS), not pandas' defaults,
        # which also blank "None", "-nan" and "1.#IND": an xlsx and a CSV copy
        # of one table must agree on what is missing (audit MA-17).
        tokens = dict(keep_default_na=False, na_values=list(NULL_TOKENS))
        header = book.parse(sheets[0], header=None, nrows=1, dtype=object, **tokens)
        frame = book.parse(sheets[0], header=0, dtype=object, **tokens)
    raw = list(header.iloc[0]) if len(header) else []
    raw = [c if not isinstance(c, float) or not c.is_integer() else int(c) for c in raw]
    names, notes = pandas_style_names(raw)
    warnings.extend(notes)
    if len(names) != frame.shape[1]:
        names = [str(c) for c in frame.columns]
    frame.columns = names
    numbers = (int, float, np.integer, np.floating)
    for col in names:
        s = frame[col]
        present = s.notna()
        kinds = {type(v) for v in s[present]}
        if not kinds:
            frame[col] = pd.Series([None] * len(s), index=s.index, dtype="string")
        elif kinds <= {str}:
            frame[col] = s.where(present, None)
        elif kinds <= {bool, np.bool_}:
            frame[col] = s.astype("boolean")
        elif all(issubclass(k, numbers) and not issubclass(k, (bool, np.bool_))
                 for k in kinds):
            frame[col] = pd.to_numeric(s)
        elif all(issubclass(k, (_dt.datetime, _dt.date, np.datetime64)) for k in kinds):
            frame[col] = pd.to_datetime(s)
        else:
            frame[col] = s.map(lambda v: None if v is None or (isinstance(v, float) and
                                                                math.isnan(v)) else str(v))
            if not kinds <= {str, _dt.time}:
                warnings.append(f"column \"{col}\" mixes text and other cell types; it is "
                                "kept as text")
    report(0.3, "Writing the columnar copy")
    con.register("__turbotab_excel", frame)
    try:
        described = con.execute("DESCRIBE SELECT * FROM __turbotab_excel").fetchall()
        select = [_select_expr(str(n), str(t)) for n, t, *_ in described]
        _execute(con, _copy_sql(select, "__turbotab_excel", tmp, len(frame)), report, 0.3, 0.8,
                 "Writing the columnar copy")
    finally:
        con.unregister("__turbotab_excel")


def _schema(con: duckdb.DuckDBPyConnection, rel: str) -> list[tuple[str, str]]:
    rows = con.execute(f"DESCRIBE SELECT * FROM {rel}").fetchall()
    return [(str(r[0]), str(r[1])) for r in rows if r[0] != ROW_ID]


def _parquet_rel(path: Path) -> str:
    return f"read_parquet({_lit(path)})"


def _count_rows(con: duckdb.DuckDBPyConnection, parquet: Path) -> int:
    return int(con.execute(f"SELECT count(*) FROM {_parquet_rel(parquet)}").fetchone()[0])


def _head_values(parquet: Path, n: int) -> dict[str, list[Any]]:
    """The first ``n`` rows of every column, in file order, as Python values."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(parquet)
    try:
        batch = next(pf.iter_batches(batch_size=n))
    except StopIteration:
        return {}
    finally:
        pf.close()
    return {name: batch.column(i).to_pylist() for i, name in enumerate(batch.schema.names)}


_ARROW_TYPES = ((_INT_TYPES - {"HUGEINT", "UHUGEINT", "INT128"}) | _FLOAT_TYPES
                | {"BOOLEAN", "VARCHAR", "DATE", "TIMESTAMP", "TIME"})


def arrow_safe(physical: str) -> bool:
    """Whether Arrow reads this column of our Parquet files exactly as DuckDB does.

    A zoned timestamp is not (DuckDB answers in the session's zone, Arrow in UTC), nor is a
    decimal (Arrow's cast to double is off in the last digit: 37198.520000000004). Both go
    through DuckDB; neither comes out of a CSV.
    """
    return _base_type(physical) in _ARROW_TYPES


def arrow_columnwise(n_columns: int, n_rows: int) -> bool:
    """Whether per-column statistics over the whole table go through Arrow (wide, short)."""
    return n_columns > BATCH_COLUMNS and n_rows <= ARROW_MAX_ROWS


def _arrow_column_chunks(parquet: Path, names: Sequence[str], n_rows: int,
                         metadata: Any = None) -> Iterator[Any]:
    """Arrow tables holding whole columns of ``parquet``, about ARROW_CHUNK_CELLS cells each."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(parquet, metadata=metadata)
    try:
        schema = pf.schema_arrow
        per = max(1, ARROW_CHUNK_CELLS // max(1, n_rows))
        for batch in _chunks(list(names), per):
            # By index, not name: pyarrow reads a dotted name as a nested path.
            indices = [schema.get_field_index(n) for n in batch]
            yield pf.reader.read_all(column_indices=indices, use_threads=True)
    finally:
        pf.close()


# (non-missing, distinct, mean byte length of a text column or None, holds a NaN)
_Counts = tuple[int, int, "float | None", bool]


def _column_counts_duckdb(con: duckdb.DuckDBPyConnection, rel: str,
                          schema: Sequence[tuple[str, str]], exact: bool,
                          report: _Progress | None, lo: float, hi: float) -> dict[str, _Counts]:
    out: dict[str, _Counts] = {}
    batches = list(_chunks(schema, BATCH_COLUMNS))
    for b, batch in enumerate(batches):
        if report is not None:
            report(lo + (hi - lo) * b / max(1, len(batches)), "Counting values per column")
        aggs: list[str] = []
        for name, physical in batch:
            c = _ident(name)
            aggs.append(f"count({c})")
            aggs.append(f"count(DISTINCT {c})" if exact else f"approx_count_distinct({c})")
            if _base_type(physical) == "VARCHAR":
                aggs.append(f"avg(strlen({c}))")
            if _is_float(physical):
                aggs.append(f"count_if(isnan({c}))")
        row = con.execute(f"SELECT {', '.join(aggs)} FROM {rel}").fetchone()
        k = 0
        for name, physical in batch:
            n_present, n_unique = int(row[k] or 0), int(row[k + 1] or 0)
            k += 2
            length = None
            if _base_type(physical) == "VARCHAR":
                length = float(row[k] or 0.0)
                k += 1
            has_nan = False
            if _is_float(physical):
                has_nan = bool(int(row[k] or 0))
                k += 1
            out[name] = (n_present, n_unique, length, has_nan)
    return out


def _column_counts_arrow(parquet: Path, schema: Sequence[tuple[str, str]], n_rows: int,
                         report: _Progress | None, lo: float, hi: float) -> dict[str, _Counts]:
    """The DuckDB counts, from Arrow: distinct counts are exact, NaN is one value and
    ``-0.0`` is ``0.0`` (as DuckDB's ``count(DISTINCT)`` has them), lengths are in bytes."""
    import pyarrow.compute as pc

    out: dict[str, _Counts] = {}
    physical = dict(schema)
    n_chunks = max(1, math.ceil(len(schema) / max(1, ARROW_CHUNK_CELLS // max(1, n_rows))))
    chunks = _arrow_column_chunks(parquet, [n for n, _ in schema], n_rows)
    for b, table in enumerate(chunks):
        if report is not None:
            report(lo + (hi - lo) * b / n_chunks, "Counting values per column")
        for name, col in zip(table.column_names, table.columns):
            phys = physical[name]
            n_present = len(col) - col.null_count
            values, has_nan = col, False
            if _is_float(phys) and n_present:
                has_nan = bool(pc.any(pc.is_nan(col)).as_py())
                values = pc.add(col, 0.0)  # -0.0 + 0.0 is 0.0
            n_unique = int(pc.count_distinct(values, mode="only_valid").as_py()) if n_present else 0
            length = None
            if _base_type(phys) == "VARCHAR":
                length = float(pc.mean(pc.binary_length(col)).as_py() or 0.0) if n_present else 0.0
            out[name] = (n_present, n_unique, length, has_nan)
    return out


def _column_stats(con: duckdb.DuckDBPyConnection, parquet: Path, n_rows: int,
                  report: _Progress | None = None, lo: float = 0.0, hi: float = 1.0,
                  arrow: bool | None = None) -> tuple[list[ColumnInfo], dict[str, Any]]:
    """ColumnInfo for every column of an ingested Parquet file.

    ``extras`` carries what the contract does not: mean string length (for
    memory estimates) and the float columns that still hold NaN. A wide, short
    table is counted through Arrow (``arrow=None`` decides by shape; the tests
    force either path), any other through DuckDB, with the same answers.
    """
    rel = _parquet_rel(parquet)
    schema = _schema(con, rel)
    exact = n_rows <= EXACT_MAX_ROWS
    if arrow is None:
        arrow = arrow_columnwise(len(schema), n_rows)
    counts: dict[str, _Counts] = {}
    if arrow:
        counts = _column_counts_arrow(parquet, [(n, t) for n, t in schema if arrow_safe(t)],
                                      n_rows, report, lo, hi)
    rest = [(n, t) for n, t in schema if n not in counts]
    if rest:
        counts.update(_column_counts_duckdb(con, rel, rest, exact, report, lo, hi))
    infos: list[ColumnInfo] = []
    avg_len: dict[str, float] = {}
    nan_columns: list[str] = []
    heads = _head_values(parquet, SAMPLE_HEAD_ROWS)
    for name, physical in schema:
        n_present, n_unique, length, has_nan = counts[name]
        if length is not None:
            avg_len[name] = length
        if has_nan:
            nan_columns.append(name)
        n_unique = min(n_unique, n_present)
        sample: list[Any] = []
        seen: set[Any] = set()
        for v in heads.get(name, []):
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            key = json.dumps(json_safe(v), sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            sample.append(json_safe(v))
            if len(sample) >= SAMPLE_VALUES:
                break
        infos.append(ColumnInfo(name=name, dtype=logical_dtype(physical, n_unique, n_rows),
                                physical_type=physical, n_missing=n_rows - n_present,
                                n_unique=n_unique, sample=sample))
    return infos, {"avg_len": avg_len, "exact": exact, "nan_columns": nan_columns}


def _null_the_nans(con: duckdb.DuckDBPyConnection, parquet: Path, nan_columns: list[str],
                   token: str) -> None:
    """Rewrite ``parquet`` with NaN → NULL in the named float columns.

    Missing is one thing downstream: a float NaN in a data file means "no
    value" in practically every research export, and NULL is what DuckDB,
    pandas (NaN again on the way out) and the summaries all count as missing.
    """
    targets = set(nan_columns)
    select = []
    for name, _physical in _schema(con, _parquet_rel(parquet)):
        c = _ident(name)
        select.append(f"CASE WHEN isnan({c}) THEN NULL ELSE {c} END AS {c}"
                      if name in targets else c)
    select.append(ROW_ID)
    out = parquet.with_name(f"{parquet.name}.{token}.nan")
    con.execute(f"COPY (SELECT {', '.join(select)} FROM {_parquet_rel(parquet)} "
                f"ORDER BY {ROW_ID}) TO {_lit(out)} "
                f"{_parquet_options(len(select), _count_rows(con, parquet))}")
    os.replace(out, parquet)


def _sidecar_path(parquet: Path) -> Path:
    return parquet.with_name(parquet.stem + ".info.json")


def _summaries_path(parquet: Path) -> Path:
    return parquet.with_name(parquet.stem + ".summaries.json")


def _write_json_atomic(path: Path, payload: Any) -> None:
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, allow_nan=False)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def ingest(source: Path, dest_parquet: Path, *,
           progress: ProgressFn | None = None) -> DatasetInfo:
    """Write ``dest_parquet`` (+ its ``.info.json`` sidecar) from ``source``.

    ``progress(fraction, message)`` is called with a non-decreasing fraction,
    possibly from a helper thread; if it raises, the ingest stops, nothing is
    left at ``dest_parquet``, and that exception propagates.
    """
    started = time.perf_counter()
    source = Path(source)
    dest = Path(dest_parquet)
    if not source.is_file():
        raise FileNotFoundError(f"no file at {source}")
    kind = _source_kind(source)
    if source.stat().st_size == 0:
        raise ValueError(f"{source.name!r} is empty")
    dest.parent.mkdir(parents=True, exist_ok=True)
    report = _Progress(progress)
    report(0.0, "Starting")

    stop = threading.Event()
    hashed: dict[str, Any] = {}

    def hash_source() -> None:
        try:
            hashed["value"] = fingerprint_file(source, stop)
        except BaseException as exc:  # noqa: BLE001 — carried to the caller
            hashed["error"] = exc

    hasher = threading.Thread(target=hash_source, name="turbotab-fingerprint", daemon=True)
    hasher.start()

    token = uuid.uuid4().hex[:8]
    tmp = dest.with_name(f".{dest.name}.{token}.tmp")
    temp_dir = dest.parent / f".duckdb-{token}"
    con = _connect(temp_dir)
    warnings: list[str] = []
    try:
        if kind == "csv":
            _ingest_csv(con, source, tmp, warnings, report)
        elif kind == "parquet":
            _ingest_parquet(con, source, tmp, warnings, report)
        else:
            _ingest_excel(con, source, tmp, warnings, report)
        n_rows = _count_rows(con, tmp)
        columns, extras = _column_stats(con, tmp, n_rows, report, 0.8, 0.95)
        if extras["nan_columns"]:
            report(0.95, "Marking NaN values as missing")
            _null_the_nans(con, tmp, extras["nan_columns"], token)
            columns, extras = _column_stats(con, tmp, n_rows)
        if not extras["exact"]:
            warnings.append(f"the table has more than {EXACT_MAX_ROWS:,} rows, so distinct "
                            "counts and quartiles are approximate")
        hasher.join()
        if "error" in hashed:
            raise hashed["error"]
        for stale in (_sidecar_path(dest), _summaries_path(dest)):
            try:
                stale.unlink()  # what described a previous ingest is void
            except FileNotFoundError:
                pass
        os.replace(tmp, dest)
    except BaseException:
        stop.set()
        raise
    finally:
        con.close()
        hasher.join()
        for leftover in (tmp, tmp.with_name(f"{tmp.name}.{token}.nan")):
            try:
                leftover.unlink()
            except FileNotFoundError:
                pass
        shutil.rmtree(temp_dir, ignore_errors=True)

    info = DatasetInfo(n_rows=n_rows, n_cols=len(columns), columns=columns,
                       source_bytes=source.stat().st_size, parquet_bytes=dest.stat().st_size,
                       ingest_seconds=round(time.perf_counter() - started, 3),
                       fingerprint=hashed["value"], warnings=warnings)
    _write_json_atomic(_sidecar_path(dest), {"version": SIDECAR_VERSION,
                                             "info": info.to_dict(), "extras": extras})
    report(1.0, "Ready")
    return info


def _increasing(values: np.ndarray) -> bool:
    return values.size < 2 or bool(np.all(values[1:] > values[:-1]))


def _positions(found: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Where each of ``ids`` sits in ``found`` (row ids as read); every id must be there."""
    if found.size and found[-1] - found[0] == found.size - 1 and _increasing(found):
        positions = ids - found[0]           # a dense run of row ids: direct
        ok = (positions >= 0) & (positions < found.size)
    elif not found.size:
        positions, ok = np.zeros_like(ids), np.zeros(ids.shape, dtype=bool)
    else:
        order = None if _increasing(found) else np.argsort(found, kind="stable")
        at = np.minimum(np.searchsorted(found, ids, sorter=order), found.size - 1)
        positions = at if order is None else order[at]
        ok = found[positions] == ids
    if not bool(np.all(ok)):
        raise ValueError(f"{int((~ok).sum())} row id(s) are not in this table")
    return positions.astype(np.int64, copy=False)


def _number_summary(values: np.ndarray, *, integer: bool) -> dict[str, Any]:
    """mean, std, min, quartiles, max of the finite present values, as DuckDB's aggregates give
    them, and how many values are infinite (``n_infinite``).

    An infinity (a ratio over a zero) is a value, not a blank, so it is counted rather than
    averaged: one ``inf`` once made the whole profile fail (audit MA-18). ``stddev_samp`` of one
    value is missing; any non-finite answer is missing (json_safe). Quartiles interpolate in
    double precision and come back in the column's own precision (``quantile_cont`` of a FLOAT is
    a FLOAT).
    """
    n_infinite = 0
    if not integer:
        n_infinite = int(np.isinf(values).sum())
        values = values[np.isfinite(values)]
    n = int(values.size)
    if n == 0:
        return {"mean": None, "std": None, "min": None, "max": None, "q25": None, "median": None,
                "q75": None, "n_infinite": n_infinite}
    with np.errstate(all="ignore"):
        x = values.astype(np.float64, copy=False)
        if integer:
            low, high = int(values.min()), int(values.max())
            if max(abs(low), abs(high)) * n < 2 ** 62:
                total: Any = int(values.sum(dtype=np.int64))
            else:
                total = sum(int(v) for v in values.tolist())
            mean: Any = total / n                    # exact sum, correctly rounded division
            lo_hi: tuple[Any, Any] = (low, high)
        else:
            mean = float(np.mean(x))
            lo_hi = (float(np.min(x)), float(np.max(x)))
        std = float(np.std(x, ddof=1)) if n > 1 else None
        quartiles = np.quantile(x, [0.25, 0.5, 0.75])
        if values.dtype == np.float32:
            quartiles = quartiles.astype(np.float32)
        q25, median, q75 = (float(q) for q in quartiles)
    return {"mean": json_safe(mean), "std": json_safe(std), "min": _stat(lo_hi[0]),
            "max": _stat(lo_hi[1]), "q25": json_safe(q25), "median": json_safe(median),
            "q75": json_safe(q75), "n_infinite": n_infinite}


# ── the store ────────────────────────────────────────────────────────────────

class DataStore:
    """Queries over one ingested Parquet file. Safe to share between threads."""

    def __init__(self, parquet: Path, memory_budget_bytes: int):
        self.parquet = Path(parquet)
        self.memory_budget_bytes = int(memory_budget_bytes)
        if not self.parquet.is_file():
            raise FileNotFoundError(f"no ingested table at {self.parquet}")
        # Queries name the file directly rather than a view: binding a view
        # re-expands every column, which costs ~1 s a query at 20,000 columns.
        self._rel = _parquet_rel(self.parquet)
        self._lock = threading.Lock()
        self._con: duckdb.DuckDBPyConnection | None = None
        self._info: DatasetInfo | None = None
        self._extras: dict[str, Any] = {}
        self._summaries: dict[str, dict[str, Any]] | None = None
        self._row_group_bounds: tuple[Any, list[tuple[int, int]]] | None = None
        self._cmap: dict[str, ColumnInfo] | None = None
        self._names: list[str] | None = None

    # ── connections ───────────────────────────────────────────────────────────
    @contextmanager
    def _cursor(self) -> Iterator[duckdb.DuckDBPyConnection]:
        """A cursor of one shared in-memory DuckDB; each call gets its own."""
        with self._lock:
            if self._con is None:
                self._con = _connect(self.parquet.parent / ".duckdb-tmp", metadata_cache=True)
            cur = self._con.cursor()
        try:
            yield cur
        finally:
            cur.close()

    def close(self) -> None:
        with self._lock:
            if self._con is not None:
                self._con.close()
                self._con = None

    def __enter__(self) -> "DataStore":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ── metadata ──────────────────────────────────────────────────────────────
    def info(self) -> DatasetInfo:
        if self._info is not None:
            return self._info
        info = extras = None
        try:
            with open(_sidecar_path(self.parquet), encoding="utf-8") as fh:
                side = json.load(fh)
            if side.get("version") == SIDECAR_VERSION:
                info = DatasetInfo.from_dict(side["info"])
                extras = side.get("extras") or {}
                if info.parquet_bytes != self.parquet.stat().st_size:
                    info = None  # the Parquet file changed under its sidecar
        except (OSError, ValueError, KeyError, TypeError):
            info = None
        if info is None:
            with self._cursor() as cur:
                n_rows = _count_rows(cur, self.parquet)
                columns, extras = _column_stats(cur, self.parquet, n_rows)
            size = self.parquet.stat().st_size
            info = DatasetInfo(n_rows=n_rows, n_cols=len(columns), columns=columns,
                               source_bytes=size, parquet_bytes=size, ingest_seconds=0.0,
                               fingerprint=fingerprint_file(self.parquet),
                               warnings=["no ingest record was found beside the Parquet "
                                         "file; its details were recomputed from the file"])
        self._cmap = {c.name: c for c in info.columns}  # 20,000 columns: build it once
        self._names = [c.name for c in info.columns]
        self._info, self._extras = info, extras or {}
        return info

    @property
    def columns(self) -> list[str]:
        self.info()
        assert self._names is not None
        return list(self._names)

    @property
    def n_rows(self) -> int:
        return self.info().n_rows

    def _column_map(self) -> dict[str, ColumnInfo]:
        self.info()
        assert self._cmap is not None
        return self._cmap

    def _resolve(self, columns: Sequence[str] | str | None) -> list[str]:
        known = self._column_map()
        if columns is None:
            return list(known)
        if isinstance(columns, str):
            columns = [columns]
        out: list[str] = []
        seen: set[str] = set()  # not `in out`: quadratic, half a second at 20,000 columns
        for col in columns:
            if col not in known:
                raise UnknownColumn(str(col))
            if col not in seen:
                seen.add(col)
                out.append(col)
        return out

    # ── UI reads ──────────────────────────────────────────────────────────────
    def window(self, offset: int, limit: int,
               columns: Sequence[str] | None = None) -> dict[str, Any]:
        """TableWindow: rows [offset, offset + limit) in file order, JSON-safe."""
        offset, limit = int(offset), int(limit)
        if offset < 0 or limit < 0:
            raise ValueError("offset and limit must not be negative")
        limit = min(limit, MAX_WINDOW_ROWS)
        cols = self._resolve(columns)
        lo, hi = offset, min(offset + limit, self.n_rows)
        rows: list[list[Any]] = []
        if hi > lo:
            # Read straight from the row groups that hold [lo, hi): with dense
            # ids that is exact, and pyarrow projects 20,000 columns in ~0.2 s
            # where a DuckDB scan of the same takes ~7 s.
            import pyarrow as pa
            import pyarrow.compute as pc
            import pyarrow.parquet as pq
            metadata, bounds = self._row_groups()
            wanted = [i for i, (first, last) in enumerate(bounds) if last >= lo and first < hi]
            pf = pq.ParquetFile(self.parquet, metadata=metadata)
            try:
                # By index, not name: pyarrow reads a dotted name as a nested
                # path, so "x" would also pull in a column called "x.1".
                schema = pf.schema_arrow
                indices = [schema.get_field_index(c) for c in (ROW_ID, *cols)]
                parts = [pf.reader.read_row_group(i, column_indices=indices)
                         .select([ROW_ID, *cols]) for i in wanted]
            finally:
                pf.close()
            table = pa.concat_tables(parts)
            ids = table.column(ROW_ID)
            table = table.filter(pc.and_(pc.greater_equal(ids, lo), pc.less(ids, hi)))
            table = table.sort_by(ROW_ID)
            values = [table.column(c).to_pylist() for c in cols]
            rows = ([[json_safe(v) for v in r] for r in zip(*values)] if cols
                    else [[] for _ in range(table.num_rows)])
        return {"columns": cols, "rows": rows, "total_rows": self.n_rows, "offset": offset}

    def _row_groups(self) -> tuple[Any, list[tuple[int, int]]]:
        """Parquet metadata and each row group's (first, last) ``__row_id``."""
        with self._lock:
            if self._row_group_bounds is None:
                import pyarrow.parquet as pq
                metadata = pq.read_metadata(self.parquet)
                col = metadata.schema.names.index(ROW_ID)
                bounds: list[tuple[int, int]] = []
                start = 0
                for i in range(metadata.num_row_groups):
                    group = metadata.row_group(i)
                    stats = group.column(col).statistics
                    if stats is not None and stats.has_min_max:
                        bounds.append((int(stats.min), int(stats.max)))
                    else:  # ids are written in file order, so the running count
                        bounds.append((start, start + group.num_rows - 1))
                    start += group.num_rows
                self._row_group_bounds = (metadata, bounds)
            return self._row_group_bounds

    def summaries(self, columns: Sequence[str] | None = None) -> list[dict[str, Any]]:
        """ColumnSummary per column (cached beside the Parquet file)."""
        cols = self._resolve(columns)
        info = self.info()
        with self._lock:
            if self._summaries is None:
                self._summaries = self._load_summaries(info.fingerprint)
            cached = dict(self._summaries)
        todo = [c for c in cols if c not in cached]
        if todo:
            fresh = self._compute_summaries(todo)
            with self._lock:
                assert self._summaries is not None
                self._summaries.update(fresh)
                cached.update(fresh)
                snapshot = dict(self._summaries)
            try:
                _write_json_atomic(_summaries_path(self.parquet),
                                   {"version": SIDECAR_VERSION,
                                    "fingerprint": info.fingerprint, "columns": snapshot})
            except OSError:
                pass  # a read-only workspace still answers; it just recomputes
        return [cached[c] for c in cols]

    def _load_summaries(self, fingerprint: str) -> dict[str, dict[str, Any]]:
        try:
            with open(_summaries_path(self.parquet), encoding="utf-8") as fh:
                data = json.load(fh)
            if (data.get("version") == SIDECAR_VERSION
                    and data.get("fingerprint") == fingerprint):
                return dict(data["columns"])
        except (OSError, ValueError, KeyError, TypeError):
            pass
        return {}

    def _compute_summaries(self, cols: list[str],
                           arrow: bool | None = None) -> dict[str, dict[str, Any]]:
        """Summaries of ``cols``: through Arrow for many columns of a short table, DuckDB
        otherwise (``arrow`` forces either path, for the tests that hold them equal)."""
        info = self.info()
        cmap = self._column_map()
        if arrow is None:
            arrow = arrow_columnwise(len(cols), info.n_rows)
        out: dict[str, dict[str, Any]] = {}
        if arrow:
            out = self._summaries_arrow([c for c in cols if arrow_safe(cmap[c].physical_type)])
        rest = [c for c in cols if c not in out]
        if rest:
            out.update(self._summaries_duckdb(rest))
        return out

    def _summary_shell(self, name: str, n: int) -> dict[str, Any]:
        ci = self._column_map()[name]
        return {"name": name, "dtype": ci.dtype, "n": int(n), "n_missing": self.n_rows - int(n),
                "n_unique": ci.n_unique, "mean": None, "std": None, "min": None, "q25": None,
                "median": None, "q75": None, "max": None, "top": None, "n_infinite": 0}

    def _summaries_arrow(self, cols: list[str]) -> dict[str, dict[str, Any]]:
        """The DuckDB summaries, computed from Arrow columns with numpy.

        Quartiles are linear interpolation between order statistics (DuckDB's
        ``quantile_cont``); integer means divide the exact sum; ``top`` ranks the
        exact value counts by count, then by the value's text — for a column with
        more than EXACT_TOP_MAX_UNIQUE values too, where DuckDB finds the candidates
        approximately (``approx_top_k``) and then counts them exactly.
        """
        import pyarrow.compute as pc

        cmap = self._column_map()
        n_rows = self.n_rows
        metadata, _ = self._row_groups()
        out: dict[str, dict[str, Any]] = {}
        for table in _arrow_column_chunks(self.parquet, cols, n_rows, metadata):
            for name, col in zip(table.column_names, table.columns):
                ci = cmap[name]
                n = len(col) - col.null_count
                summary = self._summary_shell(name, n)
                out[name] = summary
                if ci.dtype in ("numeric", "integer"):
                    if n == 0:
                        continue
                    values = pc.drop_null(col).to_numpy()
                    summary.update(_number_summary(values, integer=_is_int(ci.physical_type)))
                elif ci.dtype == "boolean":
                    n_true = int(pc.sum(col).as_py() or 0) if n else 0
                    summary["mean"] = n_true / n if n else None
                    top = [{"value": True, "count": n_true}, {"value": False, "count": n - n_true}]
                    summary["top"] = sorted([t for t in top if t["count"] > 0],
                                            key=lambda t: -t["count"])
                else:
                    counts = pc.value_counts(pc.drop_null(col).combine_chunks()) if n else None
                    pairs = (list(zip(counts.field("values").to_pylist(),
                                      counts.field("counts").to_pylist())) if n else [])
                    ranked = sorted(pairs, key=lambda kv: (-kv[1], str(kv[0])))
                    summary["top"] = [{"value": json_safe(v), "count": int(c)}
                                      for v, c in ranked[:TOP_K]]
        return out

    def _summaries_duckdb(self, cols: list[str]) -> dict[str, dict[str, Any]]:
        info = self.info()
        cmap = self._column_map()
        exact = info.n_rows <= EXACT_MAX_ROWS
        out: dict[str, dict[str, Any]] = {}
        for batch in _chunks(cols, BATCH_COLUMNS):
            aggs: list[str] = []
            plan: list[tuple[str, str, int]] = []   # (column, kind, n aggregates)
            for name in batch:
                ci = cmap[name]
                c = _ident(name)
                if ci.dtype in ("numeric", "integer"):
                    x = f"CAST({c} AS DOUBLE)" if (_is_decimal(ci.physical_type) or
                                                   "HUGEINT" in ci.physical_type) else c
                    # Over the finite values only, with the infinities counted (MA-18):
                    # stddev_samp over an inf raises "out of range" and lost the whole batch.
                    fin = f" FILTER (WHERE isfinite({x}))" if _is_float(ci.physical_type) else ""
                    q = (f"quantile_cont({x}, [0.25, 0.5, 0.75]){fin}" if exact
                         else f"approx_quantile({x}, [0.25, 0.5, 0.75]){fin}")
                    aggs += [f"count({x})", f"avg({x}){fin}", f"stddev_samp({x}){fin}",
                             f"min({x}){fin}", f"max({x}){fin}", q,
                             f"count_if(isinf({x}))" if fin else "0"]
                    plan.append((name, "number", 7))
                elif ci.dtype == "boolean":
                    aggs += [f"count({c})", f"avg(CAST({c} AS INTEGER))",
                             f"count_if({c})", f"count_if(NOT {c})"]
                    plan.append((name, "boolean", 4))
                elif ci.n_unique <= EXACT_TOP_MAX_UNIQUE:
                    aggs += [f"count({c})", f"histogram({c})"]
                    plan.append((name, "hist", 2))
                else:
                    aggs += [f"count({c})", f"approx_top_k({c}, {TOP_K})"]
                    plan.append((name, "topk", 2))
            with self._cursor() as cur:
                row = cur.execute(f"SELECT {', '.join(aggs)} FROM {self._rel}").fetchone()
                k = 0
                pending_topk: list[tuple[str, list[Any]]] = []
                for name, kind, width in plan:
                    vals = row[k:k + width]
                    k += width
                    ci = cmap[name]
                    summary = {"name": name, "dtype": ci.dtype, "n": int(vals[0] or 0),
                               "n_missing": info.n_rows - int(vals[0] or 0),
                               "n_unique": ci.n_unique, "mean": None, "std": None,
                               "min": None, "q25": None, "median": None, "q75": None,
                               "max": None, "top": None, "n_infinite": 0}
                    if kind == "number":
                        quart = vals[5] or [None, None, None]
                        summary.update(mean=json_safe(vals[1]), std=json_safe(vals[2]),
                                       min=_stat(vals[3]), max=_stat(vals[4]),
                                       q25=json_safe(quart[0]), median=json_safe(quart[1]),
                                       q75=json_safe(quart[2]), n_infinite=int(vals[6] or 0))
                    elif kind == "boolean":
                        summary["mean"] = json_safe(vals[1])
                        top = [{"value": True, "count": int(vals[2] or 0)},
                               {"value": False, "count": int(vals[3] or 0)}]
                        summary["top"] = sorted([t for t in top if t["count"] > 0],
                                                key=lambda t: -t["count"])
                    elif kind == "hist":
                        counts = vals[1] or {}
                        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))
                        summary["top"] = [{"value": json_safe(v), "count": int(n)}
                                          for v, n in ranked[:TOP_K]]
                    else:
                        pending_topk.append((name, list(vals[1] or [])))
                    out[name] = summary
                if pending_topk:
                    count_aggs: list[str] = []
                    params: list[Any] = []
                    for name, values in pending_topk:
                        for v in values:
                            count_aggs.append(f"count_if({_ident(name)} = ?)")
                            params.append(v)
                    counts_row = (cur.execute(f"SELECT {', '.join(count_aggs)} FROM {self._rel}",
                                              params).fetchone() if count_aggs else ())
                    j = 0
                    for name, values in pending_topk:
                        pairs = []
                        for v in values:
                            pairs.append({"value": json_safe(v), "count": int(counts_row[j])})
                            j += 1
                        pairs.sort(key=lambda t: (-t["count"], str(t["value"])))
                        out[name]["top"] = pairs
        return out

    def histogram(self, column: str, bins: int = 30) -> dict[str, Any]:
        """Histogram over the finite values of a numeric/integer column.

        ``n_missing`` counts the values not drawn: missing, NaN or infinite, so
        ``sum(counts) + n_missing == n_rows``. Bins are aligned to the column's resolution (audit
        MI-01): values recorded to a step (whole years, a score, HbA1c to 0.1) get bins a whole
        number of steps wide with edges half a step off the grid, so no bin is a sawtooth artifact;
        the rule and the bin assignment are ``turbotab.core.detectors.bins``, which the preview
        histograms (``consequences._histogram_pair``) use too. ``bins`` is then the most bins
        drawn; values with no common step get exactly ``bins`` equal bins.
        """
        from turbotab.core.detectors import bins as binning

        (col,) = self._resolve([column])
        ci = self._column_map()[col]
        if ci.dtype not in ("numeric", "integer"):
            raise ValueError(f"column {col!r} is {ci.dtype}; a histogram needs a numeric "
                             "or integer column")
        bins = int(bins)
        if not 1 <= bins <= MAX_HISTOGRAM_BINS:
            raise ValueError(f"bins must be between 1 and {MAX_HISTOGRAM_BINS}")
        x = f"CAST({_ident(col)} AS DOUBLE)"
        finite = f"{x} IS NOT NULL AND isfinite({x})"
        with self._cursor() as cur:
            n_ok, lo, hi = cur.execute(
                f"SELECT count(*) FILTER (WHERE {finite}), min({x}) FILTER (WHERE {finite}), "
                f"max({x}) FILTER (WHERE {finite}) FROM {self._rel}").fetchone()
            n_ok = int(n_ok or 0)
            n_rows = self.n_rows
            if n_ok == 0:
                return {"column": col, "edges": [], "counts": [], "n_missing": n_rows}
            lo, hi = float(lo), float(hi)
            grid = cur.execute(
                f"SELECT {binning.resolution_sql('v')} FROM "
                f"(SELECT DISTINCT {x} AS v FROM {self._rel} WHERE {finite})").fetchone()
            step = next((s for s, ok in zip(binning.STEPS, grid) if ok), None)
            start, width, nb, edges = binning.layout(lo, hi, step, bins)
            rows = cur.execute(
                f"SELECT least(greatest(CAST(floor(({x} - ?) / ?) AS BIGINT), 0), ?) AS b, "
                f"count(*) FROM {self._rel} WHERE {finite} GROUP BY b",
                [start, width, nb - 1]).fetchall()
        counts = [0] * nb
        for b, n in rows:
            counts[int(b)] += int(n)
        return {"column": col, "edges": edges, "counts": counts, "n_missing": n_rows - n_ok}

    # ── modeling reads ────────────────────────────────────────────────────────
    def estimate_bytes(self, columns: Sequence[str] | None = None,
                       n_rows: int | None = None) -> int:
        """Peak memory to materialize these columns for ``n_rows`` rows (default: all)."""
        info = self.info()
        cmap = self._column_map()
        cols = self._resolve(columns)
        rows = info.n_rows if n_rows is None else max(0, int(n_rows))
        avg_len = self._extras.get("avg_len", {})
        frame = 8 * rows  # the row_id index
        for name in cols:
            ci = cmap[name]
            if ci.dtype in ("numeric", "integer", "datetime"):
                frame += 8 * rows
            elif ci.dtype == "boolean":
                frame += (1 if ci.n_missing == 0 else 8) * rows
            else:  # Python str objects; pyarrow shares repeated strings
                length = float(avg_len.get(name, 16.0))
                frame += 8 * rows + min(ci.n_unique, rows) * (PY_STR_OVERHEAD + length)
        return int(frame * PEAK_FACTOR)

    def materialize(self, columns: Sequence[str] | None = None,
                    row_ids: Sequence[int] | np.ndarray | None = None) -> pd.DataFrame:
        """A pandas frame indexed by ``row_id``, in ``row_ids`` order (default: file order).

        Raises MemoryBudgetExceeded before reading anything when the estimate
        exceeds the budget. Read through pyarrow (only the row groups that hold
        the rows asked for), or through DuckDB for a column type Arrow does not
        read as DuckDB does; the two give the same frame (test_wide_data).
        """
        cols = self._resolve(columns)
        n_total = self.n_rows
        ids: np.ndarray | None = None
        if row_ids is not None:
            ids = np.asarray(row_ids)
            if ids.size and not np.issubdtype(ids.dtype, np.integer):
                raise ValueError("row_ids must be integers")
            ids = ids.astype(np.int64, copy=False).ravel()
            if ids.size and (ids.min() < 0 or ids.max() >= n_total):
                bad = int(((ids < 0) | (ids >= n_total)).sum())
                raise ValueError(f"{bad} row id(s) are outside 0..{n_total - 1}")
        n = n_total if ids is None else int(ids.size)
        estimate = self.estimate_bytes(cols, n)
        if estimate > self.memory_budget_bytes:
            raise MemoryBudgetExceeded(estimate, self.memory_budget_bytes)
        cmap = self._column_map()
        if all(arrow_safe(cmap[c].physical_type) for c in cols):
            return self._materialize_arrow(cols, ids)
        return self._materialize_duckdb(cols, ids)

    def _materialize_duckdb(self, cols: list[str], ids: np.ndarray | None) -> pd.DataFrame:
        """A DuckDB projection: exact for every type, but ~0.5 ms a column per query."""
        cmap = self._column_map()

        def expr(name: str, q: str = "") -> str:
            phys = cmap[name].physical_type
            c = q + _ident(name)
            if _is_decimal(phys) or "HUGEINT" in phys.upper():
                return f"CAST({c} AS DOUBLE) AS {_ident(name)}"
            return f"{c} AS {_ident(name)}"

        with self._cursor() as cur:
            if ids is None:
                select = ", ".join([ROW_ID, *(expr(c) for c in cols)])
                table = cur.execute(f"SELECT {select} FROM {self._rel} ORDER BY {ROW_ID}"
                                    ).to_arrow_table()
            else:
                import pyarrow as pa
                cur.register("__turbotab_ids", pa.table({
                    "ord": np.arange(ids.size, dtype=np.int64), "id": ids}))
                try:
                    select = ", ".join([f"d.{ROW_ID}", *(expr(c, "d.") for c in cols)])
                    table = cur.execute(
                        f"SELECT {select} FROM __turbotab_ids i JOIN {self._rel} d "
                        f"ON d.{ROW_ID} = i.id ORDER BY i.ord").to_arrow_table()
                finally:
                    cur.unregister("__turbotab_ids")
        frame = table.to_pandas(date_as_object=False)
        del table
        frame = frame.set_index(ROW_ID)
        frame.index.name = INDEX_NAME
        return frame

    def _materialize_arrow(self, cols: list[str], ids: np.ndarray | None) -> pd.DataFrame:
        """pyarrow reads of whole row groups, keeping only the rows asked for.

        Row groups that hold none of ``ids`` are never read; the ones that do are read a
        few at a time (ARROW_CHUNK_CELLS), so a sample of a tall table does not hold the
        table.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        metadata, bounds = self._row_groups()
        wanted = None if ids is None else (ids if _increasing(ids) else np.unique(ids))
        if wanted is None:
            groups = list(range(len(bounds)))
        else:
            groups = [g for g, (first, last) in enumerate(bounds)
                      if np.searchsorted(wanted, first) < np.searchsorted(wanted, last, "right")]
        width = len(cols) + 1
        pf = pq.ParquetFile(self.parquet, metadata=metadata)
        parts: list[Any] = []
        try:
            schema = pf.schema_arrow
            # By index, not name: pyarrow reads a dotted name as a nested path.
            indices = [schema.get_field_index(c) for c in (ROW_ID, *cols)]
            batch: list[int] = []
            cells = 0

            def flush() -> None:
                part = pf.reader.read_row_groups(batch, column_indices=indices, use_threads=True)
                part = part.select([ROW_ID, *cols])
                if wanted is not None:  # keep only the rows asked for, in row-id order
                    found = part.column(ROW_ID).to_numpy()
                    if found.size and _increasing(found):
                        inside = wanted[np.searchsorted(wanted, found[0]):
                                        np.searchsorted(wanted, found[-1], side="right")]
                        if found[-1] - found[0] == found.size - 1:  # a dense run: direct
                            at = inside - found[0]
                        else:
                            at = np.searchsorted(found, inside)
                            at = at[found[np.minimum(at, found.size - 1)] == inside]
                        if at.size != found.size:
                            part = part.take(pa.array(at))
                    else:
                        part = part.filter(pa.array(np.isin(found, wanted)))
                parts.append(part)

            for g in groups:
                rows = metadata.row_group(g).num_rows
                if batch and wanted is not None and (cells + rows * width) > ARROW_CHUNK_CELLS:
                    flush()
                    batch, cells = [], 0
                batch.append(g)
                cells += rows * width
            if batch:
                flush()
        finally:
            pf.close()
        if parts:
            table = pa.concat_tables(parts) if len(parts) > 1 else parts[0]
        else:
            fields = [schema.field(i) for i in indices]
            table = pa.schema(fields).empty_table()
        found = table.column(ROW_ID).to_numpy()
        if ids is None:
            if found.size > 1 and not bool(np.all(found[1:] > found[:-1])):
                order = np.argsort(found, kind="stable")
                table, found = table.take(pa.array(order)), found[order]
            index = found
        else:
            if not (found.size == ids.size and np.array_equal(found, ids)):
                table = table.take(pa.array(_positions(found, ids)))
            index = ids
        table = table.select(cols)
        frame = table.to_pandas(date_as_object=False)
        del table
        frame.index = pd.Index(np.asarray(index, dtype=np.int64), name=INDEX_NAME)
        return frame

    def sample(self, n: int, columns: Sequence[str] | None = None,
               seed: int = 0) -> pd.DataFrame:
        """``n`` rows drawn without replacement, the same rows for the same seed."""
        total = self.n_rows
        k = min(max(0, int(n)), total)
        ids = np.sort(np.random.default_rng(seed).choice(total, size=k, replace=False))
        return self.materialize(columns, ids)
