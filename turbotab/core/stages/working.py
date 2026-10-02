"""The table the analysis reads (M2_CONTRACT §2): ``oriented``, ``structure`` and ``working``.

Structural answers change what the table *is*, so every stage after them reads a working table
rather than the raw file::

    ingest ─▶ oriented ─▶ findings ─────────┐
                 │  ├────▶ structure ──────┤
                 │  └────▶ profile (lens hints, asked before the working table can exist)
                 └─────────────────────────┴─▶ working ─▶ target_info · roles · proposals · cohort · …

* **oriented** (reads ``orientation``): the raw table, or its transpose when the user said each
  row is a feature. The transpose runs through DuckDB and pyarrow a block of samples at a time,
  so the whole table is never in memory at once, and the new rows are named by the header (the
  sample identifiers) in a ``sample_id`` column. It also carries the shape reading question 1.5
  fires on (:func:`orientation_reading`, the statistic of ``turbotab/orientation.py``) and whether
  the table can be turned at all (:func:`turn_check`). Answering nothing, or "each row is a
  sample", references the raw file instead of copying it.
* **structure** (reads ``grain``, ``target``, ``lens``, ``repeat_kind``): what the opening
  sequence asks from, read off the oriented table. The grain suggestion and the evidence its
  contradiction check uses (``turbotab/grain.py``), the repeats-or-time-points reading for the
  named unit (``turbotab/repeats.py``), whether the outcome varies within a unit, and the
  domain-shaped aggregation menu. Suggestions only: nothing here is an answer.
* **working** (reads ``findings``, ``target``, ``grain``, ``unit``, ``aggregation``,
  ``repeat_kind``): row-local repairs (SQL column expressions from ``turbotab/core/repairs.py``,
  when that module exists), then one row per unit by DuckDB ``GROUP BY`` when the user combined
  rows. It writes ``table.parquet`` (``__row_id`` dense over its rows) and, when rows were
  combined, ``row_map.parquet``: working ``row_id`` → oriented ``source_row_id``, one line per
  source row. That map is how the participant flow and provenance stay true. When nothing
  structural is recorded it is a pass-through that references the oriented file.

Every stage after these opens its table through :func:`table_path` (``stages.data.open_store``
calls it): the working table when the stage depends on ``working``, the oriented table when it
depends on ``oriented``, else the raw file.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from turbotab.core.graph import Bundle, StageContext

TABLE = "table.parquet"
SIDECAR = "table.info.json"
ROW_MAP = "row_map.parquet"
SAMPLE_COLUMN = "sample_id"
ROW_ID = "__row_id"

NUMERIC = ("numeric", "integer")
ASSAY_LENSES = ("metabolomics", "genomics")
TRANSPOSE_CELLS = 5_000_000      # cells per transposed block: a block of samples at a time
STRUCTURE_CELLS = 20_000_000     # the structure reading materializes every column below this
STRUCTURE_MAX_COLUMNS = 2_000    # …and at most this many non-float columns above it
READING_UNITS = 5_000            # units the repeats-or-time-points reading walks, at most
SUGGESTED = 5

# Internal names in the aggregation query; a data column cannot be called this by accident.
_T, _UNIT, _K, _M = "__tt_order", "__tt_unit", "__tt_k", "__tt_m"

log = logging.getLogger(__name__)


class StructureError(ValueError):
    """A structural answer the table cannot honor as recorded; the message says what to change."""


# ── where a stage's table is ──────────────────────────────────────────────────


def _bundle_table(artifact: Any) -> Path | None:
    files = getattr(artifact, "files", None) or {}
    path = files.get(TABLE)
    return Path(path) if path is not None else None


def table_path(ctx: StageContext) -> Path:
    """The table this stage reads: working, else oriented, else the raw file (by its deps)."""
    for stage in ("working", "oriented"):
        if stage in ctx.inputs:
            path = _bundle_table(ctx.inputs[stage])
            if path is None:
                raise RuntimeError(f"the {stage} artifact has no {TABLE}")
            return path
    return Path(ctx.paths["data"])


def table_info(ctx: StageContext) -> dict[str, Any]:
    """DatasetInfo (as a dict) of the table this stage reads; ``ingest``'s for the raw file."""
    for stage in ("working", "oriented"):
        if stage in ctx.inputs:
            return dict(getattr(ctx.inputs[stage], "data", ctx.inputs[stage]))
    return dict(ctx.inputs["ingest"])


def working_paths(ctx_or_bundle: Any) -> dict[str, Path | None]:
    """``{"table": …, "row_map": … or None}`` for a working artifact (or a context holding one)."""
    bundle = ctx_or_bundle.inputs["working"] if isinstance(ctx_or_bundle, StageContext) else ctx_or_bundle
    files = getattr(bundle, "files", None) or {}
    row_map = files.get(ROW_MAP)
    return {"table": _bundle_table(bundle), "row_map": Path(row_map) if row_map is not None else None}


def working_store(ctx: StageContext) -> Any:
    """A DataStore over the working table, under the memory budget the server passed down."""
    from turbotab.core.stages.data import open_store

    if "working" not in ctx.inputs:
        raise KeyError("this stage does not depend on 'working'")
    return open_store(ctx)


def row_map(bundle: Any) -> Any:
    """Working ``row_id`` → oriented ``source_row_id`` (a DataFrame, one line per source row).

    The identity when no rows were combined.
    """
    import numpy as np
    import pandas as pd

    path = working_paths(bundle)["row_map"]
    if path is not None:
        return pd.read_parquet(path)
    n = int(bundle.data["n_rows"])
    ids = np.arange(n, dtype=np.int64)
    return pd.DataFrame({"row_id": ids, "source_row_id": ids})


# ── small helpers ─────────────────────────────────────────────────────────────


def _ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _lit(value: Any) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _scratch(ctx: StageContext, stage: str, name: str) -> Path:
    """A path beside the raw table (the project's own disk), moved into the cache on write."""
    folder = Path(ctx.paths["data"]).parent
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f".tt-{stage}-{uuid.uuid4().hex[:12]}-{name}"


def _reference(target: Path, ctx: StageContext, stage: str) -> Path:
    """A link to ``target`` (never a copy): how a pass-through names the table it passes."""
    target = Path(target).resolve()
    link = _scratch(ctx, stage, TABLE)
    try:
        os.symlink(target, link)
    except OSError:  # no symlinks here (e.g. Windows without the privilege): a hard link
        os.link(target, link)
    return link


def _connect(ctx: StageContext, stage: str) -> tuple[Any, Path]:
    from turbotab.core.datastore import _connect as connect

    temp = _scratch(ctx, stage, "duckdb")
    con = connect(temp)
    budget = ctx.settings.get("memory_budget_bytes")
    if budget:
        con.execute(f"SET memory_limit = '{max(256, int(budget) // (1 << 20))}MB'")
    return con, temp


def _execute(con: Any, ctx: StageContext, sql: str, lo: float, hi: float, message: str) -> None:
    """One long statement, cancellable through the stage's progress callback."""
    from turbotab.core.datastore import _execute as execute
    from turbotab.core.datastore import _Progress

    execute(con, sql, _Progress(ctx.progress), lo, hi, message)


_DUCK_TYPES = {"double": "DOUBLE", "float": "FLOAT", "int64": "BIGINT", "int32": "INTEGER",
               "int16": "SMALLINT", "int8": "TINYINT", "uint64": "UBIGINT", "uint32": "UINTEGER",
               "uint16": "USMALLINT", "uint8": "UTINYINT", "bool": "BOOLEAN", "string": "VARCHAR",
               "large_string": "VARCHAR", "string_view": "VARCHAR", "date32[day]": "DATE",
               "null": "VARCHAR"}


def _duck_type(arrow_type: Any) -> str:
    """The DuckDB name of an Arrow type, as ``DataStore`` reads physical types."""
    import pyarrow as pa

    if pa.types.is_timestamp(arrow_type):
        return "TIMESTAMP WITH TIME ZONE" if arrow_type.tz else "TIMESTAMP"
    if pa.types.is_decimal(arrow_type):
        return f"DECIMAL({arrow_type.precision},{arrow_type.scale})"
    return _DUCK_TYPES.get(str(arrow_type), str(arrow_type).upper())


def _describe(parquet: Path, sidecar: Path, started: float,
              warnings: Sequence[str] = ()) -> dict[str, Any]:
    """DatasetInfo for a table this module wrote, plus the sidecar the DataStore reads it from.

    Counted with pyarrow a block of columns at a time (exact distinct counts at any length): at
    20,000 columns that is about a second, where a DuckDB pass over the same file takes ten.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    from turbotab.core.datastore import (
        SAMPLE_HEAD_ROWS,
        SAMPLE_VALUES,
        SIDECAR_VERSION,
        ColumnInfo,
        DatasetInfo,
        fingerprint_file,
        json_safe,
        logical_dtype,
    )

    pf = pq.ParquetFile(parquet)
    try:
        n_rows = int(pf.metadata.num_rows)
        wanted = [i for i, name in enumerate(pf.schema_arrow.names) if name != ROW_ID]
        columns: list[Any] = []
        avg_len: dict[str, float] = {}
        nan_columns: list[str] = []
        for start in range(0, len(wanted), 200):
            block = pf.reader.read_all(column_indices=wanted[start:start + 200], use_threads=True)
            for name, col in zip(block.column_names, block.columns):
                physical = _duck_type(col.type)
                n_missing = int(col.null_count)
                present = n_rows > n_missing
                n_unique = int(pc.count_distinct(col, mode="only_valid").as_py()) if present else 0
                if physical == "VARCHAR" and present:
                    avg_len[name] = float(pc.mean(pc.utf8_length(col)).as_py() or 0.0)
                if pa.types.is_floating(col.type) and present and pc.any(pc.is_nan(col)).as_py():
                    nan_columns.append(name)
                sample: list[Any] = []
                for value in col.slice(0, SAMPLE_HEAD_ROWS).to_pylist():
                    value = json_safe(value)
                    if value is None or value in sample:
                        continue
                    sample.append(value)
                    if len(sample) >= SAMPLE_VALUES:
                        break
                columns.append(ColumnInfo(name=name, dtype=logical_dtype(physical, n_unique, n_rows),
                                          physical_type=physical, n_missing=n_missing,
                                          n_unique=n_unique, sample=sample))
    finally:
        pf.close()
    extras = {"avg_len": avg_len, "exact": True, "nan_columns": nan_columns}
    size = parquet.stat().st_size
    info = DatasetInfo(n_rows=n_rows, n_cols=len(columns), columns=columns, source_bytes=size,
                       parquet_bytes=size, ingest_seconds=round(time.perf_counter() - started, 3),
                       fingerprint=fingerprint_file(parquet), warnings=list(warnings))
    sidecar.write_text(json.dumps({"version": SIDECAR_VERSION, "info": info.to_dict(),
                                   "extras": extras}, ensure_ascii=False, allow_nan=False), "utf-8")
    return info.to_dict()


def _dataset_fields(info: Mapping[str, Any]) -> dict[str, Any]:
    keys = ("n_rows", "n_cols", "columns", "source_bytes", "parquet_bytes", "ingest_seconds",
            "fingerprint", "warnings")
    return {k: info[k] for k in keys if k in info}


def _cleanup(*paths: Path) -> None:
    for path in paths:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path, ignore_errors=True)
        else:
            try:
                path.unlink()
            except FileNotFoundError:
                pass


# ── question 1.5: which way round (oriented) ──────────────────────────────────


def _blank_reading(n_rows: int, n_numeric: int, sentence: str) -> dict[str, Any]:
    return {"reading": "undetermined", "ratio": None, "s_rows": None, "s_cols": None,
            "n_rows": int(n_rows), "n_numeric": int(n_numeric), "sentence": sentence,
            "confidence": "low"}


def orientation_reading(parquet: Path, info: Mapping[str, Any]) -> dict[str, Any]:
    """``turbotab.orientation.read``'s statistic, streamed over the Parquet file.

    The spread of row means over the spread of column means, both as sd(log10 |mean|), over the
    numeric columns more than half filled in. Row groups are read one at a time, so a table of any
    length is read without holding it.
    """
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    from turbotab import orientation as o

    n = int(info["n_rows"])
    cols = [str(c["name"]) for c in info["columns"]
            if c["dtype"] in NUMERIC and n and (n - int(c["n_missing"])) / n > 0.5]
    if len(cols) < o.MIN_NUMERIC_COLUMNS or n < o.MIN_ROWS:
        return _blank_reading(n, 0, "There is not enough of a numeric block to say which way "
                                    "round this table is.")
    pf = pq.ParquetFile(parquet)
    try:
        schema = pf.schema_arrow
        indices = [schema.get_field_index(c) for c in cols]
        row_means: list[Any] = []
        sums = np.zeros(len(cols))
        counts = np.zeros(len(cols))
        for group in range(pf.metadata.num_row_groups):
            table = pf.reader.read_row_group(group, column_indices=indices)
            block = np.column_stack([
                pc.cast(table.column(i), pa.float64()).to_numpy(zero_copy_only=False)
                for i in range(table.num_columns)])
            present = ~np.isnan(block)
            with np.errstate(invalid="ignore", divide="ignore"):
                row_means.append(np.where(present.any(axis=1),
                                          np.nansum(block, axis=1) / present.sum(axis=1), np.nan))
            sums += np.nansum(block, axis=0)
            counts += present.sum(axis=0)
    finally:
        pf.close()
    with np.errstate(invalid="ignore", divide="ignore"):
        col_means = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    s_rows = o._spread(pd.Series(np.concatenate(row_means) if row_means else []))
    s_cols = o._spread(pd.Series(col_means))
    if s_rows is None or s_cols is None or s_cols <= 0:
        return _blank_reading(n, len(cols), "The numeric block is not spread enough on either axis "
                                            "to say which way round this table is.")
    ratio = s_rows / s_cols
    reading = o.UNDETERMINED
    if ratio >= o.FEATURE_MAJOR_RATIO and s_rows >= o.MIN_ROW_SPREAD:
        reading = o.FEATURE_MAJOR
    elif ratio <= o.SAMPLE_MAJOR_RATIO:
        reading = o.SAMPLE_MAJOR
    rows, numeric = n, len(cols)
    if reading == o.FEATURE_MAJOR:
        sentence = (f"Across {rows:,} rows and {numeric:,} numeric columns, the rows differ from "
                    f"each other by orders of magnitude and the columns barely differ at all. In "
                    f"an assay table that is what features in rows looks like: different analytes "
                    f"have very different abundances, and samples of the same kind do not.")
    elif reading == o.SAMPLE_MAJOR:
        sentence = (f"Across {rows:,} rows and {numeric:,} numeric columns, the columns differ "
                    f"from each other by orders of magnitude and the rows barely differ. That is "
                    f"what one row per sample looks like.")
    else:
        sentence = (f"The rows and the columns of the {rows:,} × {numeric:,} numeric block vary "
                    f"by similar amounts, which does not say which way round this table is.")
    return {"reading": reading, "ratio": round(float(ratio), 3), "s_rows": round(float(s_rows), 3),
            "s_cols": round(float(s_cols), 3), "n_rows": rows, "n_numeric": numeric,
            "sentence": sentence,
            # never "high": a reading that could auto-advance would transpose on the app's say-so
            "confidence": "medium" if reading != o.UNDETERMINED else "low"}


def label_column(info: Mapping[str, Any]) -> str | None:
    """The column naming the features in a feature-major table (``orientation.label_column``).

    The first column that is not numbers, is filled on every row, and is near-unique (≥ 90%):
    near-unique rather than unique, so a duplicated feature name is recognized — and refused —
    instead of silently discarding the names.
    """
    n = int(info["n_rows"])
    for c in info["columns"]:
        if c["dtype"] in (*NUMERIC, "boolean") or not n:
            continue
        if int(c["n_missing"]) == 0 and int(c["n_unique"]) >= 0.9 * n:
            return str(c["name"])
    return None


def turn_check(con: Any, parquet: Path, info: Mapping[str, Any]) -> dict[str, Any]:
    """Whether the table can be turned around, and why not (OPENING_SEQUENCE §03, 1.5).

    Two refusals, each because a transpose would quietly corrupt the table: two rows with one name
    become two columns with one name (also when the names differ only in case, which DuckDB reads
    as one), and a row named ``sample_id`` collides with the column the samples are named in.
    """
    label = label_column(info)
    names = [str(c["name"]) for c in info["columns"] if str(c["name"]) != label]
    out: dict[str, Any] = {"label_column": label, "n_features": int(info["n_rows"]),
                           "n_samples": len(names), "refusal": None, "code": None}
    if label is None:
        return out
    rel = f"read_parquet({_lit(parquet)})"
    v = f"CAST({_ident(label)} AS VARCHAR)"
    dupe = con.execute(f"SELECT {v} AS v FROM {rel} GROUP BY v HAVING count(*) > 1 "
                       f"ORDER BY v LIMIT 1").fetchone()
    if dupe is not None:
        out["code"] = "duplicate_features"
        out["refusal"] = (f"Two rows are both named `{dupe[0]}`. Turned around, they would be two "
                          f"columns with one name, and every reading after would silently use one "
                          f"of them. Give the rows distinct names first.")
        return out
    case = con.execute(f"SELECT min({v}), max({v}) FROM {rel} GROUP BY lower({v}) "
                       f"HAVING count(DISTINCT {v}) > 1 ORDER BY 1 LIMIT 1").fetchone()
    if case is not None:
        out["code"] = "duplicate_features"
        out["refusal"] = (f"Rows `{case[0]}` and `{case[1]}` differ only in case. Turned around, "
                          f"they would be two columns the query engine reads as one. Give the rows "
                          f"distinct names first.")
        return out
    taken = con.execute(f"SELECT {v} FROM {rel} WHERE lower({v}) IN "
                        f"({_lit(SAMPLE_COLUMN)}, {_lit(ROW_ID)}) LIMIT 1").fetchone()
    if taken is not None:
        out["code"] = "sample_id_taken"
        out["refusal"] = (f"One of the rows is named `{taken[0]}`, the name the turned-around "
                          f"table needs for its sample identifiers. Rename that row first.")
    return out


def _double(name: str, physical: str) -> str:
    from turbotab.core.datastore import _is_decimal, _is_float, _is_int

    q = _ident(name)
    if _is_int(physical) or _is_float(physical) or _is_decimal(physical):
        return f"CAST({q} AS DOUBLE)"
    return f"TRY_CAST(CAST({q} AS VARCHAR) AS DOUBLE)"


def transpose(ctx: StageContext, con: Any, raw: Path, info: Mapping[str, Any],
              check: Mapping[str, Any]) -> tuple[Path, Path, dict[str, Any]]:
    """Write the turned-around table; returns (table, sidecar, DatasetInfo).

    One output row per input column (bar the label column), named in ``sample_id`` by its header;
    one output column per input row, named by its label (``row_<i>`` without one). A feature whose
    every value is a number becomes a DOUBLE column; any other stays text, so a value that will not
    read as a number is kept rather than blanked. A block of samples is transposed at a time.
    """
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    from turbotab.core.datastore import _is_decimal, _is_float, _is_int

    started = time.perf_counter()
    rel = f"read_parquet({_lit(raw)})"
    label = check["label_column"]
    physical = {str(c["name"]): str(c["physical_type"]) for c in info["columns"]}
    samples = [str(c["name"]) for c in info["columns"] if str(c["name"]) != label]
    n_features = int(info["n_rows"])
    if label is not None:
        names = [str(r[0]) for r in con.execute(
            f"SELECT CAST({_ident(label)} AS VARCHAR) FROM {rel} ORDER BY {ROW_ID}").fetchall()]
    else:
        names = [f"row_{i}" for i in range(n_features)]
    if SAMPLE_COLUMN in names or ROW_ID in names:
        raise StructureError(check.get("refusal") or "A row is named like the sample identifiers.")

    ctx.progress(0.15, "Reading which features are numbers")
    numeric = np.ones(n_features, dtype=bool)
    text_samples = [c for c in samples if not (_is_int(physical[c]) or _is_float(physical[c])
                                              or _is_decimal(physical[c]))]
    for start in range(0, len(text_samples), 200):
        batch = text_samples[start:start + 200]
        test = " AND ".join(f"({_ident(c)} IS NULL OR {_double(c, physical[c])} IS NOT NULL)"
                            for c in batch)
        flags = con.execute(f"SELECT {test} AS ok FROM {rel} ORDER BY {ROW_ID}").fetchnumpy()["ok"]
        numeric &= np.asarray(flags, dtype=bool)
    any_text = not bool(numeric.all())

    fields = [pa.field(SAMPLE_COLUMN, pa.string())]
    fields += [pa.field(name, pa.float64() if numeric[i] else pa.string())
               for i, name in enumerate(names)]
    fields.append(pa.field(ROW_ID, pa.int64()))
    schema = pa.schema(fields)

    table_path = _scratch(ctx, "oriented", TABLE)
    sidecar = _scratch(ctx, "oriented", SIDECAR)
    block = max(1, min(len(samples) or 1, TRANSPOSE_CELLS // max(1, n_features)))
    writer = pq.ParquetWriter(table_path, schema, compression="zstd")
    try:
        for start in range(0, len(samples), block):
            ctx.progress(0.2 + 0.6 * start / max(1, len(samples)), "Turning the table around")
            chunk = samples[start:start + block]
            select = [f"{_double(c, physical[c])} AS v{i}" for i, c in enumerate(chunk)]
            if any_text:
                select += [f"CAST({_ident(c)} AS VARCHAR) AS s{i}" for i, c in enumerate(chunk)]
            got = con.execute(f"SELECT {', '.join(select)} FROM {rel} ORDER BY {ROW_ID}").to_arrow_table()
            values = np.column_stack([got.column(f"v{i}").to_numpy(zero_copy_only=False)
                                      for i in range(len(chunk))]).astype(np.float64)
            texts = (np.column_stack([np.asarray(got.column(f"s{i}").to_pylist(), dtype=object)
                                      for i in range(len(chunk))]) if any_text else None)
            arrays = [pa.array(chunk, pa.string())]
            for f in range(n_features):
                if numeric[f]:
                    row = values[f]
                    arrays.append(pa.array(row, pa.float64(), mask=np.isnan(row)))
                else:
                    arrays.append(pa.array(list(texts[f]), pa.string()))  # type: ignore[index]
            arrays.append(pa.array(np.arange(start, start + len(chunk), dtype=np.int64)))
            writer.write_table(pa.Table.from_arrays(arrays, schema=schema), row_group_size=len(chunk))
        writer.close()
        ctx.progress(0.85, "Describing the turned-around table")
        described = _describe(table_path, sidecar, started)
    except BaseException:  # cancelled or failed: leave nothing beside the raw table
        writer.close()
        _cleanup(table_path, sidecar)
        raise
    return table_path, sidecar, described


def oriented_stage(ctx: StageContext) -> Bundle:
    """The raw table, or its transpose once the user says each row is a feature."""
    raw = Path(ctx.paths["data"])
    info = dict(ctx.inputs["ingest"])
    ctx.progress(0.02, "Reading which way round the table is")
    reading = orientation_reading(raw, info)
    con, temp = _connect(ctx, "oriented")
    try:
        check = turn_check(con, raw, info)
        if ctx.state.orientation != "feature_major":
            data = {**_dataset_fields(info), "transposed": False, "reading": reading, "turn": check}
            return Bundle(data=data, files={TABLE: _reference(raw, ctx, "oriented")})
        if check["refusal"]:
            raise StructureError(check["refusal"])
        table, sidecar, described = transpose(ctx, con, raw, info, check)
    finally:
        con.close()
        _cleanup(temp)
    data = {**described, "transposed": True, "reading": reading,
            "turn": {**check, "sample_column": SAMPLE_COLUMN}}
    return Bundle(data=data, files={TABLE: table, SIDECAR: sidecar})


# ── what the sequence asks from (structure) ───────────────────────────────────


def effective_repeat_kind(state: Any, structure: Mapping[str, Any] | None) -> str | None:
    """The answered repeat kind, else the stated reading (a skip the user has not reopened)."""
    spec = getattr(state, "repeat_kind", None)
    if spec is not None:
        return str(spec.repeat_kind)
    reading = (structure or {}).get("repeats") or {}
    return str(reading["reading"]) if reading.get("stated") and reading.get("reading") else None


def time_column(state: Any, structure: Mapping[str, Any] | None) -> str | None:
    """The column that orders a unit's rows: as answered, else the one the reading spaced them by."""
    for spec in (getattr(state, "repeat_kind", None), getattr(state, "temporal", None)):
        column = getattr(spec, "time_column", None) if spec is not None else None
        if column:
            return str(column)
    reading = (structure or {}).get("repeats") or {}
    spacing = reading.get("spacing") or {}
    return spacing.get("column") or reading.get("replicate_index")


def _quiet_streamlit() -> None:
    """``utils.test_lockbox`` reaches for a Streamlit session; outside one it only warns."""
    for name in list(logging.root.manager.loggerDict):
        if name.startswith("streamlit"):
            logging.getLogger(name).setLevel(logging.ERROR)


_MENU_KEY = {"change_from_baseline": "change"}


def _without_assay_block(frame: Any, state: Any, keep: set[str]) -> Any:
    """``frame`` without an assay's count block when an assay lens is on (the lens is field
    knowledge, OPENING_SEQUENCE §01): 495 gene counts taking a few small values each repeat
    "like a roster" by shape alone, and are measurements, not people. Columns the name heuristic
    offers as identifiers stay, as do the named unit and the outcome."""
    if not any(lens in ASSAY_LENSES for lens in (getattr(state, "lens", None) or [])):
        return frame
    from turbotab import packs

    block = packs.count_matrix(frame)
    if not block:
        return frame
    try:
        from utils.test_lockbox import rank_grouping_candidates

        named = {str(c.get("column")) for c in rank_grouping_candidates(frame)}
    except Exception:  # noqa: BLE001 - no name reading: every count column is a measurement
        named = set()
    drop = [c for c in block["columns"] if c not in named and c not in keep]
    return frame.drop(columns=drop) if drop else frame


def structure_stage(ctx: StageContext) -> dict[str, Any]:
    """The grain suggestion, the repeats reading, and the outcome's behavior within a unit."""
    from turbotab import grain as grain_mod
    from turbotab import repeats
    from turbotab.core.stages.data import open_store

    info = table_info(ctx)
    columns = [str(c["name"]) for c in info["columns"]]
    dtypes = {str(c["name"]): str(c["dtype"]) for c in info["columns"]}
    state = ctx.state
    target = state.target if state.target in dtypes else None
    spec = state.grain
    unit = spec.id_column if (spec is not None and spec.grain == "repeated"
                              and spec.id_column in dtypes) else None
    n_rows = int(info["n_rows"])
    if n_rows * max(1, len(columns)) <= STRUCTURE_CELLS:
        wanted = columns
    else:  # a float column is a measurement: the rosters, dates and indices are the rest
        wanted = [c for c in columns if dtypes[c] != "numeric"][:STRUCTURE_MAX_COLUMNS]
        wanted = list(dict.fromkeys([*wanted, *(c for c in (unit, target) if c)]))
    ctx.progress(0.1, "Reading the identifiers, dates and indices")
    with open_store(ctx) as store:
        frame = store.materialize(wanted).reset_index(drop=True)

    ctx.progress(0.4, "Looking for columns that repeat like a roster")
    _quiet_streamlit()
    roster = _without_assay_block(frame, state, keep={c for c in (unit, target) if c})
    suggestion = grain_mod.suggestion(roster)
    _quiet_streamlit()
    evidence = [{k: e[k] for k in ("column", "n_distinct", "n_rows", "rows_per", "modal_rows_per",
                                   "regular_share")} for e in suggestion.get("evidence") or []]
    # What answering "one row per unit" would contradict (grain.contradiction, name-blind): the
    # outcome is left out, since a repeating outcome is not a roster.
    found = grain_mod.contradiction(roster.drop(columns=[target]) if target else roster,
                                    grain_mod.ONE_ROW_PER_PERSON)
    contradiction = ({"columns": list(found["columns"]), "message": str(found["message"])}
                     if found else None)
    _quiet_streamlit()
    out: dict[str, Any] = {
        "grain": {"suggested": [c for c in suggestion.get("columns") or [] if c != target][:SUGGESTED],
                  "evidence": evidence[:SUGGESTED], "if_one_row": contradiction},
        "units": None, "repeats": None, "outcome": None, "aggregation": None,
        "time_columns": [], "time_column": None,
    }
    if unit is None:
        return out

    ctx.progress(0.6, "Reading what varies between a unit's rows")
    ids = frame[unit]
    sizes = ids.dropna().value_counts()
    out["units"] = {"column": unit, "n_units": int(len(sizes) + ids.isna().sum()),
                    "max_rows_per_unit": int(sizes.max()) if len(sizes) else 1,
                    "min_rows_per_unit": int(sizes.min()) if len(sizes) else 1,
                    "n_missing": int(ids.isna().sum())}
    # The reading walks each unit in Python, so a long table is read on a fixed sample of units:
    # it states a likely answer the user can reopen, never a fact the analysis counts on.
    known = sizes.index.to_numpy()
    sampled = frame
    if len(known) > READING_UNITS:
        import numpy as np

        keep = np.random.default_rng(0).choice(known, size=READING_UNITS, replace=False)
        sampled = frame[ids.isin(keep)]
    reading = repeats.read(sampled, unit)
    out["repeats"] = {k: reading.get(k) for k in ("reading", "stated", "confidence", "evidence",
                                                   "sentence", "spacing", "replicate_index")}
    out["repeats"]["n_units_read"] = int(min(len(known), READING_UNITS))
    out["time_columns"] = [c for c in repeats._date_columns(sampled) if c != unit]
    if reading.get("replicate_index") and reading["replicate_index"] not in out["time_columns"]:
        out["time_columns"].append(reading["replicate_index"])
    if target is not None:
        varying = frame.groupby(ids, dropna=True)[target].nunique(dropna=True)
        n_varying = int((varying > 1).sum())
        out["outcome"] = {"column": target, "varies": n_varying > 0, "n_units_varying": n_varying,
                          "numeric": dtypes.get(target) in NUMERIC}
    kind = effective_repeat_kind(state, out)
    if kind is not None:
        menu = repeats.menu(kind, list(state.lens or []))
        out["aggregation"] = {
            "kind": kind,
            "recommended": _MENU_KEY.get(menu["recommended"], menu["recommended"]),
            "reason": menu.get("reason"), "marker": menu.get("marker"),
            "from_pack": menu.get("from_pack"),
            "options": [_MENU_KEY.get(o["key"], o["key"]) for o in menu["options"]],
        }
    out["time_column"] = time_column(state, out)
    return out


# ── the working table ─────────────────────────────────────────────────────────


def repair_expressions(findings: Any, dispositions: Mapping[str, Any] | None) -> dict[str, str]:
    """``{column: SQL expression}`` for the row-local repairs recorded so far.

    ``turbotab/core/repairs.py`` (the repairs agent's) owns the expressions; until it exists, or
    while nothing is applied, there are none. Each expression is evaluated in a ``SELECT`` over the
    oriented table, so it names columns as double-quoted identifiers.
    """
    applied = {k: v for k, v in (dispositions or {}).items()
               if getattr(v, "action", None) == "applied"}
    if not applied:
        return {}
    try:
        from turbotab.core import repairs
    except ImportError:
        return {}
    fn = getattr(repairs, "column_expressions", None)
    if fn is None:
        return {}
    return {str(k): str(v) for k, v in (fn(findings, dispositions or {}) or {}).items()}


def aggregation_plan(state: Any, columns: set[str], structure: Mapping[str, Any] | None
                     ) -> dict[str, Any] | None:
    spec, agg = state.grain, state.aggregation
    if spec is None or spec.grain != "repeated" or state.unit != "unit" or agg is None:
        return None
    if not spec.id_column:
        raise StructureError("Rows can only be combined per unit once the column naming the unit "
                             "is recorded with the grain answer.")
    if spec.id_column not in columns:
        raise StructureError(f"There is no column named `{spec.id_column}` to combine rows by.")
    order = time_column(state, structure)
    index = ((structure or {}).get("repeats") or {}).get("replicate_index")
    # The columns that say WHEN a record was taken are taken from a record, never averaged or
    # differenced: a mean recall number of 1.5 describes nothing.
    ordering = [c for c in dict.fromkeys([order, index]) if c and c in columns]
    return {"id_column": spec.id_column, "method": agg.method, "outcome_rule": agg.outcome,
            "time_column": order if order in columns else None, "ordering": ordering}


def _order_expr(column: str | None, physical: str) -> str:
    from turbotab.core.datastore import _is_decimal, _is_float, _is_int, _is_temporal

    if column is None:
        return "NULL"
    q = _ident(column)
    if _is_int(physical) or _is_float(physical) or _is_decimal(physical) or _is_temporal(physical):
        return q
    return f"TRY_CAST({q} AS TIMESTAMP)"


def rank_units(con: Any, src_sql: str, plan: Mapping[str, Any],
               physical: Mapping[str, str]) -> None:
    """``ranked``: the source rows, each with its unit, its place in the unit's order, the unit's size.

    A unit is the rows sharing the id (a row with no id is a unit of its own), named by its first
    row; the order is the time column (missing last), then file order.
    """
    key, order = plan["id_column"], plan["time_column"]
    q_key = _ident(key)
    part = f"PARTITION BY {q_key}, CASE WHEN {q_key} IS NULL THEN {ROW_ID} END"
    con.execute(
        f"CREATE OR REPLACE TEMP TABLE ranked AS SELECT *, "
        f"min({ROW_ID}) OVER ({part}) AS {_UNIT}, "
        f"row_number() OVER ({part} ORDER BY {_T} ASC NULLS LAST, {ROW_ID}) AS {_K}, "
        f"count(*) OVER ({part}) AS {_M} "
        f"FROM (SELECT *, {_order_expr(order, physical.get(order or '', ''))} AS {_T} FROM {src_sql})")


def combine_sql(info: Mapping[str, Any], plan: Mapping[str, Any], target: str | None,
                varies: bool) -> tuple[str, list[str], list[str]]:
    """The ``SELECT`` over ``ranked`` that makes one row per unit, and the numeric and other columns.

    Numeric columns follow the method (mean, first record, last record, last minus first); the
    columns that say when a record was taken, and every non-numeric one, come from the first record
    (the last for "last"); the outcome is its one value per unit, or, when it varies, the rule.
    """
    from turbotab.core.datastore import _is_float, _is_int

    physical = {str(c["name"]): str(c["physical_type"]) for c in info["columns"]}
    dtypes = {str(c["name"]): str(c["dtype"]) for c in info["columns"]}
    key, method, rule = plan["id_column"], plan["method"], plan["outcome_rule"]

    def first(c: str) -> str:
        return f"first({_ident(c)}) FILTER (WHERE {_K} = 1)"

    def last(c: str) -> str:
        return f"first({_ident(c)}) FILTER (WHERE {_K} = {_M})"

    def change(c: str) -> str:
        """Last minus first; signed integers and floats keep their type, anything else is DOUBLE."""
        a, b = last(c), first(c)
        kind = physical[c].upper()
        native = _is_float(kind) or (_is_int(kind) and not kind.startswith("U") and "HUGE" not in kind)
        if not native:
            a, b = f"CAST({a} AS DOUBLE)", f"CAST({b} AS DOUBLE)"
        return f"CASE WHEN max({_M}) > 1 THEN {a} - {b} END"

    combined: list[str] = []
    numeric_cols: list[str] = []
    other_cols: list[str] = []
    for c in (str(c["name"]) for c in info["columns"]):
        q = _ident(c)
        if c == key:
            expr = f"first({q})"
        elif c == target:
            if not varies:
                expr = f"any_value({q})"
            elif rule == "mean":
                expr = f"avg({q})"
            else:
                pick = "arg_min" if rule == "first" else "arg_max"
                expr = f"{pick}({q}, {_K}) FILTER (WHERE {q} IS NOT NULL)"
        elif c in plan["ordering"] or dtypes[c] not in NUMERIC:
            expr = last(c) if method == "last" else first(c)
            other_cols.append(c)
        else:
            numeric_cols.append(c)
            expr = {"mean": f"avg({q})", "first": first(c), "last": last(c),
                    "change": change(c)}[method]
        combined.append(f"{expr} AS {q}")
    combined.append(f"CAST(row_number() OVER (ORDER BY {_UNIT}) - 1 AS BIGINT) AS {ROW_ID}")
    sql = f"SELECT {', '.join(combined)} FROM ranked GROUP BY {_UNIT} ORDER BY {_UNIT}"
    return sql, numeric_cols, other_cols


def outcome_rule_problem(target: str | None, key: str, varies: bool, rule: str | None,
                         dtypes: Mapping[str, str]) -> str | None:
    """Why the outcome cannot be combined as recorded, or None."""
    if target is None or not varies:
        return None
    if rule is None:
        return (f"`{target}` changes within a `{key}`, so combining the rows needs to know which "
                f"value to keep. Answer the combining question again and choose the outcome.")
    if rule == "mean" and dtypes.get(target) not in NUMERIC:
        return (f"`{target}` is not a number, so its mean cannot be the outcome; keep the first or "
                f"the last value instead.")
    return None


def _aggregate(ctx: StageContext, con: Any, src_sql: str, info: Mapping[str, Any],
               plan: dict[str, Any], table_out: Path, map_out: Path) -> dict[str, Any]:
    """One row per unit (DuckDB GROUP BY) and the row map; returns what the receipt states."""
    physical = {str(c["name"]): str(c["physical_type"]) for c in info["columns"]}
    dtypes = {str(c["name"]): str(c["dtype"]) for c in info["columns"]}
    key, method, order = plan["id_column"], plan["method"], plan["time_column"]
    rank_units(con, src_sql, plan, physical)

    target = ctx.state.target if ctx.state.target in physical else None
    varies = False
    if target is not None:
        varies = bool(con.execute(
            f"SELECT count(*) FROM (SELECT count(DISTINCT {_ident(target)}) AS d FROM ranked "
            f"GROUP BY {_UNIT}) WHERE d > 1").fetchone()[0])
    rule = plan["outcome_rule"]
    problem = outcome_rule_problem(target, key, varies, rule, dtypes)
    if problem:
        raise StructureError(problem)

    select, numeric_cols, other_cols = combine_sql(info, plan, target, varies)
    _execute(con, ctx, f"COPY ({select}) TO {_lit(table_out)} (FORMAT parquet, COMPRESSION zstd)",
             0.35, 0.7, "Combining each unit's rows")
    _execute(con, ctx,
             f"COPY (SELECT CAST(dense_rank() OVER (ORDER BY {_UNIT}) - 1 AS BIGINT) AS row_id, "
             f"{ROW_ID} AS source_row_id FROM ranked ORDER BY row_id, source_row_id) "
             f"TO {_lit(map_out)} (FORMAT parquet, COMPRESSION zstd)",
             0.7, 0.8, "Writing which rows became which")

    varying: list[str] = []
    if method in ("mean", "change"):
        mixed = [c for c in other_cols if c not in plan["ordering"]]
        for start in range(0, len(mixed), 200):
            batch = mixed[start:start + 200]
            counts = ", ".join(f"count(DISTINCT {_ident(c)}) AS n{i}" for i, c in enumerate(batch))
            maxima = ", ".join(f"max(n{i})" for i in range(len(batch)))
            row = con.execute(f"SELECT {maxima} FROM (SELECT {counts} FROM ranked "
                              f"GROUP BY {_UNIT})").fetchone()
            varying += [c for c, m in zip(batch, row) if int(m or 0) > 1]
    n_units, single = con.execute(
        f"SELECT count(*), count(*) FILTER (WHERE m = 1) FROM "
        f"(SELECT max({_M}) AS m FROM ranked GROUP BY {_UNIT})").fetchone()
    con.execute("DROP TABLE ranked")
    return {
        "id_column": key, "method": method,
        "outcome": ({"column": target, "varies": varies,
                     "rule": (rule if varies else "constant")} if target else None),
        "time_column": order, "ordered_by": order or "file order",
        "n_source_rows": int(info["n_rows"]), "n_units": int(n_units),
        "single_record_units": int(single), "varying": varying,
        "combined_numeric": len(numeric_cols),
    }


def source_sql(source: Path, names: Sequence[str], repairs: Mapping[str, str]) -> str:
    """The oriented table with the row-local repairs applied, as a subquery."""
    select = ", ".join([*(f"{repairs[c]} AS {_ident(c)}" if c in repairs else _ident(c)
                          for c in names), ROW_ID])
    return f"(SELECT {select} FROM read_parquet({_lit(source)}))"


def working_stage(ctx: StageContext) -> Bundle:
    """Row-local repairs, then one row per unit when the user combined rows; else a pass-through."""
    oriented = ctx.inputs["oriented"]
    source = _bundle_table(oriented)
    if source is None:
        raise RuntimeError(f"the oriented artifact has no {TABLE}")
    info = dict(oriented.data)
    names = [str(c["name"]) for c in info["columns"]]
    transposed = bool(info.get("transposed"))
    findings = ctx.inputs.get("findings")
    structure = ctx.inputs.get("structure")
    structure = getattr(structure, "data", structure)

    ctx.progress(0.02, "Reading the recorded repairs and the grain")
    repairs = repair_expressions(findings, ctx.state.findings)
    unknown = [c for c in repairs if c not in names]
    if unknown:
        raise StructureError(f"A recorded repair names `{unknown[0]}`, which is not a column.")
    plan = aggregation_plan(ctx.state, set(names), structure)
    base = {"transposed": transposed, "n_source_rows": int(info["n_rows"]),
            "repairs": [{"column": c, "expression": e} for c, e in repairs.items()]}
    if not repairs and plan is None:
        data = {**_dataset_fields(info), **base, "pass_through": True, "aggregation": None,
                "row_map": "identity"}
        return Bundle(data=data, files={TABLE: _reference(source, ctx, "working")})

    started = time.perf_counter()
    src_sql = source_sql(source, names, repairs)
    table_out = _scratch(ctx, "working", TABLE)
    sidecar = _scratch(ctx, "working", SIDECAR)
    map_out = _scratch(ctx, "working", ROW_MAP)
    con, temp = _connect(ctx, "working")
    files: dict[str, Path] = {TABLE: table_out, SIDECAR: sidecar}
    try:
        aggregation = None
        if plan is None:
            _execute(con, ctx, f"COPY (SELECT * FROM {src_sql} ORDER BY {ROW_ID}) TO "
                               f"{_lit(table_out)} (FORMAT parquet, COMPRESSION zstd)",
                     0.1, 0.8, "Applying the repairs")
        else:
            aggregation = _aggregate(ctx, con, src_sql, info, plan, table_out, map_out)
            files[ROW_MAP] = map_out
        ctx.progress(0.85, "Describing the working table")
        described = _describe(table_out, sidecar, started)
    except BaseException:
        _cleanup(table_out, sidecar, map_out)
        raise
    finally:
        con.close()
        _cleanup(temp)
    data = {**described, **base, "pass_through": False, "aggregation": aggregation,
            "row_map": ROW_MAP if aggregation is not None else "identity"}
    return Bundle(data=data, files=files)


__all__ = [
    "ROW_MAP", "SAMPLE_COLUMN", "TABLE", "StructureError", "aggregation_plan", "combine_sql",
    "effective_repeat_kind", "label_column", "orientation_reading", "oriented_stage",
    "outcome_rule_problem", "rank_units", "repair_expressions", "row_map", "source_sql",
    "structure_stage", "table_info", "table_path", "time_column", "transpose", "turn_check",
    "working_paths", "working_stage", "working_store",
]
