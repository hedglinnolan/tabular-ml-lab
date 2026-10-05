"""The table the analysis reads (M2_CONTRACT §2): ``oriented``, ``structure`` and ``working``.

Structural answers change what the table *is*, so every stage after them reads a working table
rather than the raw file::

    ingest ─▶ oriented ─▶ findings ─────────┐
                 │  ├────▶ structure ──────┤
                 │  └────▶ profile (lens hints, asked before the working table can exist)
                 └─────────────────────────┴─▶ working ─▶ target_info · roles · proposals · cohort · …

* **oriented** (reads ``orientation``, ``feature_table``): the raw table, or its transpose when
  the user said each row is a feature. Before turning, :func:`turn_plan` partitions the columns
  into the label, the feature annotations (m/z, retention time, IDs: kept in ``features.parquet``,
  never turned into samples) and the samples (audit MA-04). The transpose runs through DuckDB and
  pyarrow a block of samples at a time, so the whole table is never in memory at once, and the new
  rows are named by the header (the sample identifiers) in a ``sample_id`` column. It also carries
  the shape reading question 1.5 fires on (:func:`orientation_reading`, the statistic of
  ``turbotab/orientation.py``) and whether the table can be turned at all (:func:`turn_check`).
  Answering nothing, or "each row is a sample", references the raw file instead of copying it.
* **structure** (reads ``grain``, ``target``, ``lens``, ``repeat_kind``, ``findings``): what the
  opening sequence asks from, read off the oriented table. The grain suggestion and the evidence
  its contradiction check uses (``turbotab/grain.py``), the repeats-or-time-points reading for the
  named unit (``turbotab/repeats.py``), whether the outcome varies within a unit, the
  domain-shaped aggregation menu, and whether the time column can order a unit's records
  (:func:`time_order`). Text dates are read by ``turbotab.core.dates`` (the date-reading repair, in
  ``findings``, settles month-first or day-first); a column that reads both ways is left out of
  the reading until then. Suggestions only: nothing here is an answer.
* **working** (reads ``findings``, ``target``, ``grain``, ``unit``, ``aggregation``,
  ``repeat_kind``): row-local repairs (SQL column expressions from ``turbotab/core/repairs.py``),
  then one row per unit by DuckDB ``GROUP BY`` when the user combined rows, each column by the
  rule that fits it (:func:`column_rule`; audit MA-14), in the time column's real order (audit
  MA-03). It writes ``table.parquet`` (``__row_id`` dense over its rows) and, when rows were
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
import re
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
    # WP14 (audit IN-11): the names are read first (turbotab.core.detectors.orientation.cue), and
    # columns that describe the features (m/z, RT, MZmine's "row …") stay out of the shape's block.
    from turbotab.core.detectors import orientation as cues

    cols = [c for c in cols if not is_feature_annotation(c)] or cols
    pf = pq.ParquetFile(parquet)
    try:
        schema = pf.schema_arrow
        indices = [schema.get_field_index(c) for c in cols]
        row_means: list[Any] = []
        sums = np.zeros(len(cols))
        counts = np.zeros(len(cols))
        lo, hi = np.inf, -np.inf
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
            if present.any():
                lo, hi = min(lo, float(np.nanmin(block))), max(hi, float(np.nanmax(block)))
    finally:
        pf.close()
    with np.errstate(invalid="ignore", divide="ignore"):
        col_means = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    every_row = np.concatenate(row_means) if row_means else np.array([])
    read = cues.shape(every_row, col_means, lo, hi, n, len(cols))
    if read is not None:
        return {**read, "n_rows": n, "n_numeric": len(cols), "basis": "shape",
                # never "high": a reading that could auto-advance would transpose on the app's say-so
                "confidence": "medium" if read["reading"] != o.UNDETERMINED else "low"}
    s_rows = o._spread(pd.Series(every_row))
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


def named_reading(parquet: Path, info: Mapping[str, Any], shape: dict[str, Any]) -> dict[str, Any]:
    """The orientation reading with the names read first (METABOLOMICS_PACK §01's cascade: header
    tokens, then feature- and sample-name grammar, then shape); ``shape`` when the names say
    nothing. Never high confidence (audit IN-11)."""
    import pyarrow.parquet as pq

    from turbotab.core.detectors import orientation as cues

    columns = [str(c["name"]) for c in info["columns"] if str(c["name"]) != ROW_ID]
    numeric = [str(c["name"]) for c in info["columns"] if c["dtype"] in NUMERIC
               and str(c["name"]) != ROW_ID]
    label = label_column(info)
    values: list[str] = []
    if label is not None:
        try:
            values = [str(v) for v in pq.read_table(parquet, columns=[label]).column(0)
                      .slice(0, 2000).to_pylist() if v is not None]
        except Exception:  # noqa: BLE001 - no label values: the column names still speak
            values = []
    found = cues.cue(columns, numeric, label, values)
    if found is None:
        return {**shape, "basis": shape.get("basis", "shape")}
    return {**shape, "reading": found["reading"], "sentence": found["sentence"],
            "basis": "names", "cue": found["cue"], "shape_reading": shape.get("reading"),
            "confidence": "medium"}


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


# ── what a features-in-rows table's columns are (audit MA-04) ─────────────────
#
# Turned round, every column but the label becomes a sample, so a column that describes the
# features (m/z, retention time, an HMDB or Entrez ID) became a "participant" whose values were
# m/z readings, and a text annotation made every feature text. Before turning, the columns are
# partitioned into the label, the feature annotations (kept beside the features, never turned)
# and the samples; the stage, the turn check and the orientation preview read this one partition.

def _name_key(name: str) -> str:
    """``"row m/z"`` → ``"row m z"``; ``"RT (min)"`` → ``"rt min"``: lowercase words, nothing else."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(name).lower()).split())


# Whole column names (as :func:`_name_key` writes them) that describe a feature, not a sample.
FEATURE_ANNOTATIONS = frozenset({
    "mz", "m z", "mass", "exact mass", "monoisotopic mass", "neutral mass", "mw",
    "molecular weight", "ppm", "delta ppm", "charge", "rt", "rt min", "rt s", "retention time",
    "retention time min", "ret time", "ri", "retention index", "ccs", "id", "feature id",
    "feature", "compound id", "metabolite id", "entrez", "entrez id", "entrezid",
    "entrez gene id", "gene id", "geneid", "probe id", "probeset id", "chr", "chromosome",
    "start", "end", "stop", "strand", "length", "gene length", "position", "pos", "cas",
    "pubchem", "pubchem cid", "cid", "kegg", "kegg id", "hmdb", "hmdb id", "chebi", "inchikey",
    "smiles", "formula", "adduct", "isotope", "index", "msi level", "msi", "score",
})
# Words that name a column of feature identifiers (a numeric label: Entrez IDs, row IDs).
FEATURE_ID_WORDS = frozenset({"id", "ids", "entrez", "entrezid", "geneid", "gene", "probe",
                              "probeset", "feature", "compound", "metabolite", "row", "index"})
COHERENCE_FEATURES = 2_000   # features the sample-coherence guard reads, at most
COHERENCE_CELLS = 4_000_000  # …and cells (features × samples)
COHERENT_BLOCK = 0.6         # median rank correlation of a sample with the block's median
NOT_A_SAMPLE = 0.2           # a column below this, in a coherent block, is not a sample


def is_feature_annotation(name: str) -> bool:
    """A numeric column whose name says it describes the features: m/z, RT, an ID, a position.

    The whole name is read, never a word inside it, so a sample called ``gene_KO_1`` stays a
    sample. MZmine writes every feature column ``row …`` (``row m/z``, ``row retention time``,
    ``row ID``); a sample column never starts so.
    """
    key = _name_key(name)
    return key in FEATURE_ANNOTATIONS or key.startswith("row ")


def _feature_id_name(name: str) -> bool:
    return bool(set(_name_key(name).split()) & FEATURE_ID_WORDS)


def _number_shares(con: Any, rel: str, columns: Sequence[str]) -> dict[str, float]:
    """The share of each text column's values that read as numbers. At least half: a sample with
    a few text cells ("<LOD", "n.d."); less: an annotation (an HMDB ID, a formula)."""
    out: dict[str, float] = {}
    for start in range(0, len(columns), 200):
        batch = list(columns[start:start + 200])
        aggs = []
        for c in batch:
            x = _ident(c)
            aggs += [f"count({x})", f"count(TRY_CAST(trim(CAST({x} AS VARCHAR)) AS DOUBLE))"]
        row = con.execute(f"SELECT {', '.join(aggs)} FROM {rel}").fetchone()
        for i, c in enumerate(batch):
            n, numbers = int(row[2 * i] or 0), int(row[2 * i + 1] or 0)
            out[c] = numbers / n if n else 0.0
    return out


def _whole_numbers(con: Any, rel: str, column: str, physical: str) -> bool:
    """Every value is a whole number (an integer column, or doubles with nothing after the point)."""
    from turbotab.core.datastore import _is_float, _is_int

    if _is_int(physical):
        return True
    if not _is_float(physical):
        return False
    x = _ident(column)
    whole = con.execute(f"SELECT bool_and({x} = floor({x})) FROM {rel} WHERE {x} IS NOT NULL").fetchone()[0]
    return bool(whole)


def _incoherent(con: Any, rel: str, samples: Sequence[str], n_features: int) -> list[tuple[str, float, float]]:
    """Columns that do not move with the rest of the measurement block: ``(column, its rank
    correlation with the block's median, the block's median correlation)``.

    In an assay table a feature abundant in one sample is abundant in every sample, so the samples'
    ranks across features agree (Spearman ρ near 1); an m/z or retention-time column does not.
    Read on the first COHERENCE_FEATURES features. Empty when the block itself is not coherent.
    """
    import numpy as np
    import pandas as pd

    if len(samples) < 3 or n_features < 10:
        return []
    # A cell budget, so 20,000 samples read ~200 features (ρ's noise is then about ±0.07).
    rows = max(50, min(COHERENCE_FEATURES, COHERENCE_CELLS // len(samples)))
    block = np.empty((min(rows, n_features), 0))
    for start in range(0, len(samples), 200):
        cols = ", ".join(f"TRY_CAST(CAST({_ident(c)} AS VARCHAR) AS DOUBLE) AS v{i}"
                         for i, c in enumerate(samples[start:start + 200]))
        part = con.execute(f"SELECT {cols} FROM {rel} ORDER BY {ROW_ID} LIMIT {rows}"
                           ).df().to_numpy(dtype=float)
        block = np.hstack([block, part])
    import warnings

    ranks = pd.DataFrame(block).rank(axis=0).to_numpy()
    rho = []
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)  # a constant or empty column: no ρ
        median = np.nanmedian(ranks, axis=1)
        for j in range(ranks.shape[1]):
            # With few samples a column's own ranks pull the median toward it: compare it with
            # the median of the others.
            ref = (np.nanmedian(np.delete(ranks, j, axis=1), axis=1) if ranks.shape[1] <= 50
                   else median)
            ok = ~np.isnan(ranks[:, j]) & ~np.isnan(ref)
            r = float(np.corrcoef(ranks[ok, j], ref[ok])[0, 1]) if ok.sum() >= 10 else np.nan
            rho.append(r)
    rho_arr = np.asarray(rho)
    typical = float(np.nanmedian(rho_arr)) if np.isfinite(rho_arr).any() else float("nan")
    if not typical >= COHERENT_BLOCK:
        return []
    return [(samples[j], float(r), typical) for j, r in enumerate(rho_arr)
            if np.isfinite(r) and r < NOT_A_SAMPLE]


def turn_plan(con: Any, parquet: Path, info: Mapping[str, Any],
              declared: Any = None) -> dict[str, Any]:
    """The partition a turn follows: the label, the feature annotations, the samples; or a refusal.

    * **label**: the first text column that is filled and near-unique (:func:`label_column`); else,
      the first column when it holds a different whole number on every row and its name says it
      identifies features (``entrez_id``, ``row ID``): numeric IDs stay feature names, never a
      sample whose values are gene IDs.
    * **annotations**: text columns that do not read as numbers (an HMDB ID, a formula), and
      numeric columns whose whole name says they describe the features (:func:`is_feature_annotation`).
    * **samples**: the rest, the measurement block.

    ``declared`` (the ``feature_table`` answer, a FeatureTableSpec) replaces the reading: its label
    and its annotation list are the user's, and the guards below do not second-guess them.
    Refusals (each with ``code``): duplicate or case-colliding feature names, a feature named like
    the sample column, a first column of unique whole numbers whose name does not say what it is,
    and a column that does not move with the measurement block (it reads like an annotation).
    """
    rel = f"read_parquet({_lit(parquet)})"
    columns = [str(c["name"]) for c in info["columns"]]
    physical = {str(c["name"]): str(c["physical_type"]) for c in info["columns"]}
    dtypes = {str(c["name"]): str(c["dtype"]) for c in info["columns"]}
    n = int(info["n_rows"])
    out: dict[str, Any] = {"label_column": None, "label_kind": None, "annotations": [],
                           "samples": [], "n_features": n, "n_samples": 0, "refusal": None,
                           "code": None, "exits": [], "declared": declared is not None}

    def refuse(code: str, message: str, exits: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        out.update(code=code, refusal=message, exits=exits or [])
        return out

    if declared is not None:  # the user's reading: no inference, no second-guessing
        label = getattr(declared, "label", None)
        annotations = [c for c in getattr(declared, "annotations", None) or [] if c in physical]
        if label is not None and label not in physical:
            label = None
    else:
        label, annotations = label_column(info), []
    if declared is None and label is None and columns and n:
        first = columns[0]
        c_info = next(c for c in info["columns"] if str(c["name"]) == first)
        unique = int(c_info["n_missing"]) == 0 and int(c_info["n_unique"]) == n
        if unique and dtypes[first] in NUMERIC and _whole_numbers(con, rel, first, physical[first]):
            if _feature_id_name(first):
                label = first
            else:
                return refuse(
                    "label_unclear",
                    f"The first column, `{first}`, holds a different whole number on every row, so "
                    f"it reads like the features' identifiers (Entrez IDs, row numbers), not a "
                    f"sample; turned round, its values would become a sample's measurements. Say "
                    f"what it is first.",
                    exits=[{"label": f"`{first}` names the features",
                            "decision": {"kind": "set_feature_table", "label": first,
                                         "annotations": []}},
                           {"label": f"`{first}` is a sample",
                            "decision": {"kind": "set_feature_table", "label": None,
                                         "annotations": []}}])
    label_kind = None if label is None else ("number" if dtypes.get(label) in NUMERIC else "text")
    rest = [c for c in columns if c != label]
    if declared is None:
        shares = _number_shares(con, rel, [c for c in rest if dtypes[c] not in (*NUMERIC, "boolean")])
        for c in rest:
            if dtypes[c] in NUMERIC:
                if is_feature_annotation(c):
                    annotations.append(c)
            elif dtypes[c] == "boolean" or shares.get(c, 0.0) < 0.5:
                annotations.append(c)
    samples = [c for c in rest if c not in annotations]
    out.update(label_column=label, label_kind=label_kind, annotations=annotations, samples=samples,
               n_samples=len(samples))
    if label is not None:
        problem = _label_problem(con, rel, label, label_kind)
        if problem is not None:
            return refuse(*problem)
    if declared is None:
        odd = _incoherent(con, rel, samples, n)
        if odd:
            names = [c for c, _, _ in odd]
            col, r, typical = odd[0]
            more = f" (and {len(odd) - 1} more)" if len(odd) > 1 else ""
            return refuse(
                "not_a_sample",
                f"`{col}`{more} does not move with the other samples: its ranks across the features "
                f"agree with theirs at ρ = `{r:.2f}`, where a typical sample agrees at `{typical:.2f}`. "
                f"It reads like a description of the features (m/z, retention time, an ID), and "
                f"turned round it would become a sample.",
                exits=[{"label": "Keep it beside the features",
                        "decision": {"kind": "set_feature_table", "label": label,
                                     "annotations": [*annotations, *names]}},
                       {"label": "It is a sample",
                        "decision": {"kind": "set_feature_table", "label": label,
                                     "annotations": list(annotations)}}])
    return out


def _label_sql(label: str, kind: str | None) -> str:
    """The features' names as text: a whole-number label is written without a decimal point."""
    x = _ident(label)
    if kind == "number":
        return f"CAST(CAST({x} AS HUGEINT) AS VARCHAR)"
    return f"CAST({x} AS VARCHAR)"


def _label_problem(con: Any, rel: str, label: str, kind: str | None) -> tuple[str, str] | None:
    """Why the label's values cannot name columns, or None (``(code, message)``)."""
    v = _label_sql(label, kind)
    blank = con.execute(f"SELECT count(*) FROM {rel} WHERE {_ident(label)} IS NULL").fetchone()[0]
    if int(blank or 0):
        return ("unnamed_features", f"`{int(blank):,}` rows have no name in `{label}`; turned round, "
                                    f"they would be columns with no name. Name every row first.")
    dupe = con.execute(f"SELECT {v} AS v FROM {rel} GROUP BY v HAVING count(*) > 1 "
                       f"ORDER BY v LIMIT 1").fetchone()
    if dupe is not None:
        return ("duplicate_features",
                f"Two rows are both named `{dupe[0]}`. Turned around, they would be two columns with "
                f"one name, and every reading after would silently use one of them. Give the rows "
                f"distinct names first.")
    case = con.execute(f"SELECT min({v}), max({v}) FROM {rel} GROUP BY lower({v}) "
                       f"HAVING count(DISTINCT {v}) > 1 ORDER BY 1 LIMIT 1").fetchone()
    if case is not None:
        return ("duplicate_features",
                f"Rows `{case[0]}` and `{case[1]}` differ only in case. Turned around, they would be "
                f"two columns the query engine reads as one. Give the rows distinct names first.")
    taken = con.execute(f"SELECT {v} FROM {rel} WHERE lower({v}) IN "
                        f"({_lit(SAMPLE_COLUMN)}, {_lit(ROW_ID)}) LIMIT 1").fetchone()
    if taken is not None:
        return ("sample_id_taken",
                f"One of the rows is named `{taken[0]}`, the name the turned-around table needs for "
                f"its sample identifiers. Rename that row first.")
    return None


def turn_check(con: Any, parquet: Path, info: Mapping[str, Any], declared: Any = None) -> dict[str, Any]:
    """Whether the table can be turned around, and why not (OPENING_SEQUENCE §03, 1.5): the
    :func:`turn_plan` without its sample list (which may run to tens of thousands of names)."""
    plan = turn_plan(con, parquet, info, declared)
    return {k: v for k, v in plan.items() if k != "samples"}


def _double(name: str, physical: str) -> str:
    from turbotab.core.datastore import _is_decimal, _is_float, _is_int

    q = _ident(name)
    if _is_int(physical) or _is_float(physical) or _is_decimal(physical):
        return f"CAST({q} AS DOUBLE)"
    return f"TRY_CAST(CAST({q} AS VARCHAR) AS DOUBLE)"


FEATURES = "features.parquet"   # a turned table's features: name and annotations, one row each


def feature_names(con: Any, rel: str, plan: Mapping[str, Any], limit: int | None = None) -> list[str]:
    """The turned table's column names, one per feature row in file order (``row_<i>`` unnamed)."""
    label = plan["label_column"]
    cap = f" LIMIT {int(limit)}" if limit is not None else ""
    if label is None:
        n = int(plan["n_features"]) if limit is None else min(int(limit), int(plan["n_features"]))
        return [f"row_{i}" for i in range(n)]
    return [str(r[0]) for r in con.execute(
        f"SELECT {_label_sql(label, plan['label_kind'])} FROM {rel} ORDER BY {ROW_ID}{cap}").fetchall()]


def numeric_features(con: Any, rel: str, plan: Mapping[str, Any], physical: Mapping[str, str],
                     limit: int | None = None) -> Any:
    """Per feature row: is every sample's value a number? (Then the feature is a DOUBLE column.)"""
    import numpy as np

    from turbotab.core.datastore import _is_decimal, _is_float, _is_int

    n = int(plan["n_features"]) if limit is None else min(int(limit), int(plan["n_features"]))
    numeric = np.ones(n, dtype=bool)
    cap = f" LIMIT {int(limit)}" if limit is not None else ""
    text = [c for c in plan["samples"]
            if not (_is_int(physical[c]) or _is_float(physical[c]) or _is_decimal(physical[c]))]
    for start in range(0, len(text), 200):
        batch = text[start:start + 200]
        test = " AND ".join(f"({_ident(c)} IS NULL OR {_double(c, physical[c])} IS NOT NULL)"
                            for c in batch)
        flags = con.execute(f"SELECT {test} AS ok FROM {rel} ORDER BY {ROW_ID}{cap}").fetchnumpy()["ok"]
        numeric &= np.asarray(flags, dtype=bool)
    return numeric


def turn_block(con: Any, rel: str, chunk: Sequence[str], physical: Mapping[str, str],
               numeric: Any, limit: int | None = None) -> list[Any]:
    """One Arrow array per feature row, each holding the ``chunk`` samples' values in that row.

    The stage writes every block of samples with this, and the orientation preview draws its
    corner with it (``limit`` feature rows), so the preview shows what the stage writes.
    """
    import numpy as np
    import pyarrow as pa

    any_text = not bool(np.asarray(numeric).all())
    cap = f" LIMIT {int(limit)}" if limit is not None else ""
    select = [f"{_double(c, physical[c])} AS v{i}" for i, c in enumerate(chunk)]
    if any_text:
        select += [f"CAST({_ident(c)} AS VARCHAR) AS s{i}" for i, c in enumerate(chunk)]
    got = con.execute(f"SELECT {', '.join(select)} FROM {rel} ORDER BY {ROW_ID}{cap}").to_arrow_table()
    values = np.column_stack([got.column(f"v{i}").to_numpy(zero_copy_only=False)
                              for i in range(len(chunk))]).astype(np.float64)
    texts = (np.column_stack([np.asarray(got.column(f"s{i}").to_pylist(), dtype=object)
                              for i in range(len(chunk))]) if any_text else None)
    arrays = []
    for f in range(len(numeric)):
        if numeric[f]:
            row = values[f]
            arrays.append(pa.array(row, pa.float64(), mask=np.isnan(row)))
        else:
            arrays.append(pa.array(list(texts[f]), pa.string()))  # type: ignore[index]
    return arrays


def turned_corner(con: Any, parquet: Path, info: Mapping[str, Any], plan: Mapping[str, Any],
                  n_samples: int, n_features: int) -> Any:
    """The turned table's top-left corner (``sample_id`` and the first features) as a DataFrame,
    computed by the stage's own functions: the orientation preview draws this."""
    import pandas as pd

    rel = f"read_parquet({_lit(parquet)})"
    physical = {str(c["name"]): str(c["physical_type"]) for c in info["columns"]}
    names = feature_names(con, rel, plan, n_features)
    numeric = numeric_features(con, rel, plan, physical, n_features)
    chunk = list(plan["samples"][:n_samples])
    if not chunk:
        return pd.DataFrame(columns=[SAMPLE_COLUMN, *names])
    arrays = turn_block(con, rel, chunk, physical, numeric, n_features)
    data = {SAMPLE_COLUMN: chunk}
    for name, array in zip(names, arrays):
        data[name] = array.to_pylist()
    return pd.DataFrame(data)


def transpose(ctx: StageContext, con: Any, raw: Path, info: Mapping[str, Any],
              plan: Mapping[str, Any]) -> tuple[Path, Path, Path, dict[str, Any]]:
    """Write the turned-around table; returns (table, sidecar, features, DatasetInfo).

    One output row per sample column of the :func:`turn_plan`, named in ``sample_id`` by its
    header; one output column per input row, named by its label (``row_<i>`` without one). The
    label and the feature annotations are never turned: they are written to ``features.parquet``,
    one row per feature in file order. A feature whose every sample value is a number becomes a
    DOUBLE column; any other stays text, so a value that will not read as a number is kept rather
    than blanked. A block of samples is transposed at a time.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    started = time.perf_counter()
    rel = f"read_parquet({_lit(raw)})"
    physical = {str(c["name"]): str(c["physical_type"]) for c in info["columns"]}
    samples = list(plan["samples"])
    n_features = int(info["n_rows"])
    names = feature_names(con, rel, plan)
    if SAMPLE_COLUMN in names or ROW_ID in names:
        raise StructureError(plan.get("refusal") or "A row is named like the sample identifiers.")

    ctx.progress(0.15, "Reading which features are numbers")
    numeric = numeric_features(con, rel, plan, physical)
    fields = [pa.field(SAMPLE_COLUMN, pa.string())]
    fields += [pa.field(name, pa.float64() if numeric[i] else pa.string())
               for i, name in enumerate(names)]
    fields.append(pa.field(ROW_ID, pa.int64()))
    schema = pa.schema(fields)

    table_path = _scratch(ctx, "oriented", TABLE)
    sidecar = _scratch(ctx, "oriented", SIDECAR)
    features_path = _scratch(ctx, "oriented", FEATURES)
    block = max(1, min(len(samples) or 1, TRANSPOSE_CELLS // max(1, n_features)))
    writer = pq.ParquetWriter(table_path, schema, compression="zstd")
    try:
        for start in range(0, len(samples), block):
            ctx.progress(0.2 + 0.6 * start / max(1, len(samples)), "Turning the table around")
            chunk = samples[start:start + block]
            arrays = [pa.array(chunk, pa.string()), *turn_block(con, rel, chunk, physical, numeric),
                      pa.array(range(start, start + len(chunk)), pa.int64())]
            writer.write_table(pa.Table.from_arrays(arrays, schema=schema), row_group_size=len(chunk))
        writer.close()
        label = plan["label_column"]
        name_sql = (_label_sql(label, plan["label_kind"]) if label is not None
                    else f"'row_' || CAST({ROW_ID} AS VARCHAR)")
        kept = "".join(f", {_ident(c)}" for c in plan["annotations"])
        con.execute(f"COPY (SELECT {name_sql} AS feature{kept} FROM {rel} ORDER BY {ROW_ID}) "
                    f"TO {_lit(features_path)} (FORMAT parquet, COMPRESSION zstd)")
        ctx.progress(0.85, "Describing the turned-around table")
        described = _describe(table_path, sidecar, started)
    except BaseException:  # cancelled or failed: leave nothing beside the raw table
        writer.close()
        _cleanup(table_path, sidecar, features_path)
        raise
    return table_path, sidecar, features_path, described


def oriented_stage(ctx: StageContext) -> Bundle:
    """The raw table, or its transpose once the user says each row is a feature."""
    raw = Path(ctx.paths["data"])
    info = dict(ctx.inputs["ingest"])
    ctx.progress(0.02, "Reading which way round the table is")
    reading = named_reading(raw, info, orientation_reading(raw, info))
    con, temp = _connect(ctx, "oriented")
    declared = getattr(ctx.state, "feature_table", None)
    try:
        plan = turn_plan(con, raw, info, declared)
        check = {k: v for k, v in plan.items() if k != "samples"}
        if ctx.state.orientation != "feature_major":
            data = {**_dataset_fields(info), "transposed": False, "reading": reading, "turn": check}
            return Bundle(data=data, files={TABLE: _reference(raw, ctx, "oriented")})
        if plan["refusal"]:
            raise StructureError(plan["refusal"])
        table, sidecar, features, described = transpose(ctx, con, raw, info, plan)
    finally:
        con.close()
        _cleanup(temp)
    data = {**described, "transposed": True, "reading": reading,
            "turn": {**check, "sample_column": SAMPLE_COLUMN}}
    return Bundle(data=data, files={TABLE: table, SIDECAR: sidecar, FEATURES: features})




# ── what the sequence asks from (structure) ───────────────────────────────────


def effective_repeat_kind(state: Any, structure: Mapping[str, Any] | None) -> str | None:
    """The repeat kind a consumer may read: the answered one, or a reading the ledger holds
    settled. The structure stage's stated reading is medium at most (spacing, an index, a recall
    or occasion word), so it is the question's proposal, never its answer (BLUEPRINT §14.1; the
    gate: recalls numbered across an ``assessment``'s occasions were stated repeats and the mean
    erased an arm's change)."""
    from turbotab.core.readings import repeat_kind_reading

    found = repeat_kind_reading(state, structure)
    return str(found.value) if found is not None and found.settled else None


def stated_grain(structure: Mapping[str, Any] | None) -> Any:
    """The grain the structure stage states rather than asks (a ``GrainSpec``), or None.

    Stated only when a recognized person identifier is unique on every row and nothing else
    repeats like a roster (M2_CONTRACT §10): then each person is one row, and the column is named.
    """
    from turbotab.core.decisions import GrainSpec

    stated = ((structure or {}).get("grain") or {}).get("stated") or {}
    column = stated.get("column")
    return GrainSpec(grain="one_row_per_unit", id_column=str(column)) if column else None


def effective_grain(state: Any, structure: Mapping[str, Any] | None) -> Any:
    """The answered grain, else the stated one (a skip the user has not reopened), else None."""
    answered = getattr(state, "grain", None)
    return answered if answered is not None else stated_grain(structure)


def with_effective_grain(state: Any, structure: Mapping[str, Any] | None) -> tuple[Any, bool]:
    """``state`` with the stated grain in its slot when none is answered, and whether it was."""
    if getattr(state, "grain", None) is not None:
        return state, False
    stated = stated_grain(structure)
    if stated is None:
        return state, False
    return state.model_copy(update={"grain": stated}), True


def person_identifiers(columns: Sequence[str], target: str | None) -> list[str]:
    """Columns whose names say they identify a person (``SEQN``, ``participant_id``, ``eid``,
    ``patid``, ``USUBJID``; never a sample, visit, record or site column): the one recognizer every
    stage shares (:func:`turbotab.core.recognizers.names_a_person`; audit IN-06)."""
    from turbotab.core.recognizers import names_a_person

    return [c for c in columns if c not in (target, ROW_ID) and names_a_person(c)]


def unit_suggestions(columns: Sequence[str], dtypes: Mapping[str, str], frame: Any,
                     target: str | None) -> list[str]:
    """The grain question's candidate unit columns, never a measurement (audit IN-06: the shape
    reading offered ``length_of_stay_days``, ``age`` and ``sodium_mmol_l``): a column the
    recognizer reads as a person's identifier first, then other identifiers, then shape-only
    candidates whose names and values do not read as measurements."""
    from turbotab.core.recognizers import id_kind, reads_as_measurement

    def measured(c: str) -> bool:
        values = frame[c] if c in frame.columns else None
        return reads_as_measurement(c, dtype=dtypes.get(c), values=values) is not None

    kept = [c for c in columns if c != target and c != ROW_ID and not measured(c)]
    rank = {"subject": 0, "record": 1, "cluster": 2}
    return sorted(kept, key=lambda c: (rank.get(str(id_kind(c)), 3), list(columns).index(c)))


def _stated_grain_reading(frame: Any, candidates: Sequence[str], n_rows: int,
                          contradiction: Any) -> dict[str, Any] | None:
    """``{column, n_rows, sentence}`` when a recognized person identifier is unique on every row.

    Checked on every row (no blank, no value twice), and only when no column repeats like a
    roster: grain stays a question whenever there is no identifier or something repeats.
    """
    if contradiction or n_rows < 2:
        return None
    from turbotab.core.voice import stated_grain_reason

    for column in candidates:
        if column not in frame.columns:
            continue
        values = frame[column]
        if len(values) != n_rows or values.isna().any() or int(values.nunique()) != n_rows:
            continue
        return {"column": column, "n_rows": n_rows, "sentence": stated_grain_reason(column)}
    return None


def time_column(state: Any, structure: Mapping[str, Any] | None) -> str | None:
    """The column that orders a unit's rows, settled: named with the repeat kind or the temporal
    answer, or the reading's column confirmed on its own (``confirm_reading``). The repeats
    reading's spacing column or index is only proposed (:func:`proposed_time_column`): first, last
    and change take no value by a column nobody named (BLUEPRINT §14.1)."""
    from turbotab.core.readings import time_column_reading

    found = time_column_reading(state, structure)
    return found.column if found is not None and found.settled else None


def proposed_time_column(state: Any, structure: Mapping[str, Any] | None) -> str | None:
    """The column the repeats reading would order a unit's rows by, settled or not: what the
    questions offer to confirm."""
    from turbotab.core.readings import time_column_reading

    found = time_column_reading(state, structure)
    return found.column if found is not None else None


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
    # The outcome as the table spells it: a log-scale outcome (WP18) is derived in the working
    # table, so its rows are read through the column it is the log of.
    from turbotab.core.structural import outcome_source

    source = outcome_source(state)
    target = source if source in dtypes else None
    spec = state.grain
    unit = spec.id_column if (spec is not None and spec.grain == "repeated"
                              and spec.id_column in dtypes) else None
    n_rows = int(info["n_rows"])
    people = person_identifiers(columns, target)  # what may state the grain (M2_CONTRACT §10)
    if n_rows * max(1, len(columns)) <= STRUCTURE_CELLS:
        wanted = columns
    else:  # a float column is a measurement: the rosters, dates and indices are the rest
        wanted = [c for c in columns if dtypes[c] != "numeric"][:STRUCTURE_MAX_COLUMNS]
        wanted = list(dict.fromkeys([*wanted, *(c for c in (unit, target) if c), *people]))
    ctx.progress(0.1, "Reading the identifiers, dates and indices")
    with open_store(ctx) as store:
        frame = store.materialize(wanted).reset_index(drop=True)
    # Dates written as text are read by the formats every value fits (turbotab.core.dates), never
    # by pandas' lenient guess; a column that reads both month-first and day-first is left out of
    # the date reading until the date-reading repair says which (audit MA-05).
    declared = declared_date_formats(state)
    as_written = frame
    frame, unread = read_date_columns(frame, declared)

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
    suggested = unit_suggestions([str(c) for c in suggestion.get("columns") or []], dtypes,
                                 frame, target)
    out: dict[str, Any] = {
        "grain": {"suggested": suggested[:SUGGESTED],
                  "evidence": [e for e in evidence if e["column"] in suggested][:SUGGESTED],
                  "if_one_row": contradiction,
                  "stated": _stated_grain_reading(frame, people, n_rows, contradiction)},
        "units": None, "repeats": None, "outcome": None, "aggregation": None,
        "time_columns": [], "time_column": None, "unread_dates": unread, "time_order": None,
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
    sampled = sampled.drop(columns=[c for c in unread if c != unit])
    # WP14 (audit IN-12): stated only when unambiguous; time-point and recall evidence are read.
    from turbotab.core.detectors import repeats as repeat_reading

    reading = repeat_reading.read(sampled, unit, list(state.lens or []))
    out["repeats"] = {k: reading.get(k) for k in ("reading", "stated", "confidence", "evidence",
                                                   "sentence", "spacing", "replicate_index",
                                                   "implicate_column")}
    out["repeats"]["n_units_read"] = int(min(len(known), READING_UNITS))
    out["time_columns"] = [c for c in repeats._date_columns(sampled) if c != unit]
    if reading.get("replicate_index") and reading["replicate_index"] not in out["time_columns"]:
        out["time_columns"].append(reading["replicate_index"])
    if target is not None:
        varying = frame.groupby(ids, dropna=True)[target].nunique(dropna=True)
        n_varying = int((varying > 1).sum())
        out["outcome"] = {"column": state.target, "varies": n_varying > 0,
                          "n_units_varying": n_varying, "numeric": dtypes.get(target) in NUMERIC}
    kind = effective_repeat_kind(state, out)
    if kind == "imputed_copies":
        # Audit I18 (WP18): copies of one imputed record have no order and no recommended summary;
        # their mean, or each copy as a row, is what the menu holds, with its concern.
        from turbotab.core.structural import COPIES_CONCERN

        out["aggregation"] = {"kind": kind, "recommended": None, "reason": COPIES_CONCERN[:1].upper()
                              + COPIES_CONCERN[1:] + ".", "marker": "offered",
                              "from_pack": None, "options": ["mean"]}
    elif kind is not None:
        menu = repeats.menu(kind, list(state.lens or []))
        out["aggregation"] = {
            "kind": kind,
            "recommended": _MENU_KEY.get(menu["recommended"], menu["recommended"]),
            "reason": menu.get("reason"), "marker": menu.get("marker"),
            "from_pack": menu.get("from_pack"),
            "options": [_MENU_KEY.get(o["key"], o["key"]) for o in menu["options"]],
        }
    out["time_column"] = time_column(state, out)
    out["proposed_time_column"] = proposed_time_column(state, out)
    # Whole-number columns that change within units and may be codes or counts (BLUEPRINT §14.1):
    # combining them takes the user's answer for each, never a guess (the gate: per-recall
    # ``coffee_cups`` and ``eating_occasions`` combined by their mode under "mean").
    # (An imputed copy's number, WP18's I18, numbers the copies; it is no value to combine.)
    asked = code_or_count_facts(
        frame, unit, exclude=[c for c in (target, out["time_column"], out["proposed_time_column"],
                                          reading.get("implicate_column"), *out["time_columns"])
                              if c])
    out["code_or_count"] = list(asked)
    out["code_or_count_facts"] = asked
    if out["time_column"] in as_written.columns:
        # Whether that column can put a unit's records in order: what combining by first, last
        # or change needs, and what the aggregation answer is refused on (sequence.py).
        out["time_order"] = frame_time_order(as_written, out["time_column"],
                                             declared_levels(state, out["time_column"]),
                                             declared.get(out["time_column"]))
    return out


# ── putting a unit's records in order (audit MA-03) ───────────────────────────
#
# Combining by first, last or change orders each unit's records by the time column. A text column
# that is not a date (visit labels: "baseline", "month_6") or a date the cast could not read
# ("Mar 3, 2021", "03/14/2021") once cast to NULL, so the order silently fell back to file order
# while the receipt said "ordered by visit_date"; a change from baseline came out wrong-signed.
# Now a column orders by its numbers, by its dates (every format turbotab.core.dates reads), or by
# a declared order of its levels; records it cannot place are undated and counted; and a column
# that places fewer than half of them is refused, with the exits to declare the order.

ORDERABLE = 0.5      # share of a time column's values it must place to order the records
MAX_LEVELS = 50      # text levels listed for declaring an order


def _levels_key(level: str) -> tuple[Any, ...]:
    """A natural order for visit labels: baseline first, then by the time they name
    ("week 4" before "month 3" before "month 12"; "V2" before "V10"), then alphabetically."""
    words = [w for w in re.split(r"[^a-z0-9.]+", str(level).lower()) if w]
    start = {"baseline", "bl", "screening", "screen", "enrollment", "enrolment", "pre", "pretest",
             "randomization", "randomisation", "entry"}
    if start & set(words):
        return (0, 0.0, str(level))
    scale = {"day": 1, "days": 1, "d": 1, "week": 7, "weeks": 7, "wk": 7, "w": 7, "month": 30.44,
             "months": 30.44, "mo": 30.44, "m": 30.44, "year": 365.25, "years": 365.25, "yr": 365.25,
             "y": 365.25}
    parts = re.findall(r"([a-z]*)\s*([0-9]+(?:\.[0-9]+)?)", " ".join(words))
    for unit_word, number in parts:
        return (1, float(number) * scale.get(unit_word, 1.0), str(level))
    return (2, 0.0, str(level))


def natural_order(levels: Sequence[str]) -> list[str]:
    return sorted((str(v) for v in levels), key=_levels_key)


def time_order(con: Any, rel: str, column: str, physical: str, levels: Sequence[str] | None = None,
               declared: str | None = None) -> dict[str, Any]:
    """How ``column`` of relation ``rel`` puts records in order, read over every value.

    ``kind``: ``numbers`` or ``dates`` (a number, date or timestamp column), ``dates`` (text every
    format of :mod:`turbotab.core.dates` reads, or the declared one), ``levels`` (text, by the
    declared order), ``ambiguous`` (dates that read both month-first and day-first), ``mixed``
    (both orders in one column) or ``none`` (text that is not dates). ``placed`` counts the values
    it orders; the rest are undated. ``expr`` is the SQL ordering expression (not for the
    artifact). ``levels`` lists the text levels in the order first seen; ``proposed`` a natural order.
    """
    from turbotab.core import dates
    from turbotab.core.datastore import _is_decimal, _is_float, _is_int, _is_temporal

    x = _ident(column)
    n = int(con.execute(f"SELECT count({x}) FROM {rel}").fetchone()[0] or 0)
    out: dict[str, Any] = {"column": column, "kind": "none", "n": n, "placed": 0, "orderable": False,
                           "levels": [], "proposed": [], "examples": [], "expr": "NULL"}
    if _is_int(physical) or _is_float(physical) or _is_decimal(physical) or _is_temporal(physical):
        out.update(kind="dates" if _is_temporal(physical) else "numbers", placed=n, expr=x)
    else:
        text = f"CAST({x} AS VARCHAR)"
        if levels:
            listed = ", ".join(_lit(v) for v in levels)
            placed = con.execute(f"SELECT count_if({text} IN ({listed})) FROM {rel}").fetchone()[0]
            cases = " ".join(f"WHEN {_lit(v)} THEN {i}" for i, v in enumerate(levels))
            out.update(kind="levels", placed=int(placed or 0), expr=f"CASE {text} {cases} END")
        else:
            reading = dates.read_text_dates(con, rel, column, declared)
            out.update(kind=reading.kind, examples=reading.examples)
            if reading.kind == "dates":
                out.update(placed=reading.parsed, expr=dates.parse_sql(x, reading.formats))
        if out["kind"] in ("none", "levels"):
            seen = con.execute(
                f"SELECT v FROM (SELECT {text} AS v, row_number() OVER () AS r FROM {rel}) "
                f"WHERE v IS NOT NULL GROUP BY v ORDER BY min(r) LIMIT {MAX_LEVELS + 1}").fetchall()
            out["levels"] = [str(r[0]) for r in seen[:MAX_LEVELS]]
            out["proposed"] = natural_order(out["levels"]) if len(seen) <= MAX_LEVELS else []
    out["orderable"] = (out["kind"] in ("numbers", "dates", "levels") and n > 0
                        and out["placed"] >= ORDERABLE * n)
    return out


def frame_time_order(frame: Any, column: str, levels: Sequence[str] | None = None,
                     declared: str | None = None) -> dict[str, Any]:
    """:func:`time_order` over one column of a pandas frame (the structure stage's), for the artifact."""
    import duckdb
    import pandas as pd
    import pyarrow as pa

    s = frame[column]
    if pd.api.types.is_bool_dtype(s):
        physical = "VARCHAR"
    elif pd.api.types.is_numeric_dtype(s):
        physical = "DOUBLE"
    elif pd.api.types.is_datetime64_any_dtype(s):
        physical = "TIMESTAMP"
    else:
        physical = "VARCHAR"
    if physical == "VARCHAR":
        values = pa.array([None if pd.isna(v) else str(v) for v in s.tolist()], pa.string())
    else:
        values = pa.array(s, from_pandas=True)  # NaN and NaT are missing, not values
    con = duckdb.connect()
    try:
        con.register("__order", pa.table({column: values}))
        found = time_order(con, "__order", column, physical, levels, declared)
    finally:
        con.close()
    found.pop("expr", None)
    return found


def declared_levels(state: Any, column: str | None) -> list[str] | None:
    """The declared order of a text time column's levels (the repeats answer's ``levels``)."""
    spec = getattr(state, "repeat_kind", None)
    levels = getattr(spec, "levels", None) if spec is not None else None
    named = getattr(spec, "time_column", None) if spec is not None else None
    if not levels or (named and column and named != column):
        return None
    return [str(v) for v in levels]


def declared_date_formats(state: Any) -> dict[str, str]:
    """Column -> the strptime format the date-reading repair chose (month or day first)."""
    try:
        from turbotab.core.repairs import date_formats
    except ImportError:  # pragma: no cover - the repairs module is part of the engine
        return {}
    return date_formats(state)


_DATEISH = re.compile(r"\d{1,4}[-/.]\d{1,2}|(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)",
                      re.IGNORECASE)


def read_date_columns(frame: Any, declared: Mapping[str, str]) -> tuple[Any, list[str]]:
    """``frame`` with its text date columns read as dates, and the columns that read two ways.

    A text column is tried when half of its first values look like a date (digits with separators,
    or a month's name); it becomes datetime64 when the formats :mod:`turbotab.core.dates` reads
    (and the declared one) read at least 90% of it. Ambiguous and mixed columns stay text and are
    returned, so no reader downstream guesses their order.
    """
    import pandas as pd

    from turbotab.core import dates

    out = frame
    unread: list[str] = []
    for c in frame.columns:
        s = frame[c]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s) \
                or pd.api.types.is_bool_dtype(s):
            continue
        head = s.dropna().astype(str).head(20)
        if head.empty or head.map(lambda v: bool(_DATEISH.search(v))).mean() < 0.5:
            continue
        reading = dates.read_series(s, declared.get(str(c)), column=str(c))
        if reading.kind == "dates" and reading.share >= dates.READS_AS_DATES:
            if out is frame:
                out = frame.copy()
            out[c] = dates.to_timestamps(s, reading.formats)
        elif reading.kind in ("ambiguous", "mixed"):
            unread.append(str(c))
    return out, unread


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


# ── combining a unit's records (audit MA-03, MA-14; B21, B22) ─────────────────
#
# One rule for every numeric column once differenced a sex code into 0 for everyone and averaged
# smoking codes 1/2/3 into 1.67, and the receipt listed nothing. Each column now gets the rule
# that fits what it is:
#
#   kind \ method   mean       first    last     change
#   constant        its value  ·        ·        ·         (never varies within a unit)
#   amount          mean       first    last     last − first
#   code            mode       first    last     first (the baseline value)
#   category        mode       first    last     first
#
# Whether a whole-valued number is a code or an amount is the user's answer (BLUEPRINT §14.3:
# whole numbers fit both, whatever their count or type; fractional values are amounts by their
# values); a 0/1's mean is a share of the records. "first" and "last" are the value AT the first and last dated
# record, for predictors and the outcome alike: never a value from another visit standing in for a
# missing one. Undated records are left out of first, last and change, and counted. The aggregation
# answer's ``columns`` may set any column's rule; the receipt lists every column that varied within
# a unit, with the rule it got.

CODE_LEVELS = 10
COMBINE_RULES = ("mean", "first", "last", "change", "mode")
_RULE_WORDS = {"mean": "mean", "first": "first record", "last": "last record",
               "change": "last minus first", "mode": "most frequent", "constant": "its one value"}


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
    overrides = dict(getattr(agg, "columns", None) or {})
    unknown = [c for c in overrides if c not in columns]
    if unknown:
        raise StructureError(f"The combining answer sets a rule for `{unknown[0]}`, which is not a "
                             f"column.")
    from turbotab.core.readings import confirmation

    if order is None and needs_order(agg.method, agg.outcome, overrides):
        proposed = proposed_time_column(state, structure)
        if proposed is not None:
            # BLUEPRINT §14.1: first, last and change take values by an order nobody confirmed
            # only by asking (the aggregation answer's refusal carries the exits).
            raise StructureError(
                f"Combining by {agg.method if agg.method != 'mean' else agg.outcome} takes each "
                f"unit's values from a particular record, and the repeats reading would order the "
                f"records by `{proposed}`, which nobody has confirmed. Confirm `{proposed}` as the "
                f"column that orders them, or name another, before combining.")
    codes = {c for c in columns if confirmation(state, "code_or_count", c) == "code"}
    amounts = {c for c in columns if confirmation(state, "code_or_count", c) == "amount"}
    return {"id_column": spec.id_column, "method": agg.method, "outcome_rule": agg.outcome,
            # The whole-valued columns that change within units (the structure stage's reading of
            # every row): each is combined only as the user said, codes or amounts.
            "asked": [c for c in ((structure or {}).get("code_or_count") or []) if c in columns],
            "read_asked": "code_or_count" in (structure or {}),
            "time_column": order if order in columns else None,
            "levels": declared_levels(state, order), "overrides": overrides,
            "codes": sorted(codes), "amounts": sorted(amounts),
            # The column the repeats reading would order by, unconfirmed: an index of the records,
            # never a code or a count to combine (it keeps its first record's value).
            "index_column": (proposed_time_column(state, structure)
                             if order is None and proposed_time_column(state, structure) in columns
                             else None)}


def needs_order(method: str, outcome_rule: str | None, overrides: Mapping[str, str] | None = None) -> bool:
    """Whether combining takes values from a particular record (so the records need an order)."""
    taken = {method, outcome_rule or "", *(overrides or {}).values()}
    return bool(taken & {"first", "last", "change"})


def order_refusal(order: Mapping[str, Any], method: str) -> str:
    """Why the time column cannot order the records, and what to do: the stage's and the
    aggregation answer's refusal (sequence.py), worded once."""
    col, kind = f"`{order['column']}`", order["kind"]
    if kind == "ambiguous":
        ex = (order.get("examples") or [{}])[0]
        said = (f" (`{ex['text']}` is {ex['month_first']} month-first and {ex['day_first']} "
                f"day-first)") if ex else ""
        return (f"{col} holds dates that read two ways{said}, so it cannot order each unit's "
                f"records until you say whether the month or the day comes first.")
    if kind == "mixed":
        return (f"{col} writes some dates month-first and others day-first, so it cannot order "
                f"each unit's records. Choose another column, or combine by the mean.")
    if kind in ("none", "levels") and order.get("levels"):
        shown = ", ".join(f"`{v}`" for v in order["levels"][:4])
        more = " …" if len(order["levels"]) > 4 else ""
        placed = (f"; the declared order places `{order['placed']:,}` of its `{order['n']:,}` "
                  f"values") if kind == "levels" else ""
        return (f"{col} holds text labels ({shown}{more}) that do not say their order{placed}. "
                f"Declare the order of the levels, or choose another column, before combining "
                f"by {method}.")
    return (f"{col} places only `{order['placed']:,}` of its `{order['n']:,}` values in time, so "
            f"it cannot order each unit's records. Choose another column, or combine by the mean.")


def rank_units(con: Any, src_sql: str, plan: Mapping[str, Any], order_expr: str) -> None:
    """``ranked``: the source rows, each with its unit, its place in the unit's order, and the
    unit's count of dated records.

    A unit is the rows sharing the id (a row with no id is a unit of its own), named by its first
    row. ``_K`` orders a unit's dated records by the time column, then file order, and its undated
    records after them; ``_M`` counts the dated ones, so the first dated record is ``_K = 1`` and
    the last is ``_K = _M``. With no time column every record is dated, in file order.
    """
    key = plan["id_column"]
    q_key = _ident(key)
    part = f"PARTITION BY {q_key}, CASE WHEN {q_key} IS NULL THEN {ROW_ID} END"
    con.execute(
        f"CREATE OR REPLACE TEMP TABLE ranked AS SELECT *, "
        f"min({ROW_ID}) OVER ({part}) AS {_UNIT}, "
        f"row_number() OVER ({part} ORDER BY {_T} ASC NULLS LAST, {ROW_ID}) AS {_K}, "
        f"count({_T}) OVER ({part}) AS {_M} "
        f"FROM (SELECT *, {order_expr} AS {_T} FROM {src_sql})")


def column_kinds(con: Any, columns: Sequence[str], physical: Mapping[str, str]) -> dict[str, dict[str, Any]]:
    """Per column of ``ranked``: did it vary within any unit, and is it an amount, a code, a
    category or constant (see the table above)."""
    from turbotab.core.datastore import _is_decimal, _is_float, _is_int

    out: dict[str, dict[str, Any]] = {}
    for start in range(0, len(columns), 100):
        batch = list(columns[start:start + 100])
        within = ", ".join(f"count(DISTINCT {_ident(c)}) AS d{i}" for i, c in enumerate(batch))
        varied = con.execute(f"SELECT {', '.join(f'max(d{i})' for i in range(len(batch)))} FROM "
                             f"(SELECT {within} FROM ranked GROUP BY {_UNIT})").fetchone()
        aggs = []
        for c in batch:
            x, phys = _ident(c), physical[c]
            numeric = _is_int(phys) or _is_float(phys) or _is_decimal(phys)
            if numeric:
                whole = "true" if _is_int(phys) else (
                    f"coalesce(bool_and({x} = floor({x})) FILTER (WHERE isfinite(CAST({x} AS DOUBLE))), true)")
                aggs += [f"count(DISTINCT {x})", whole,
                         f"coalesce(bool_and(CAST({x} AS DOUBLE) IN (0, 1)), true)", f"count({x})"]
            else:
                aggs += ["0", "false", "false", "0"]
        row = con.execute(f"SELECT {', '.join(aggs)} FROM ranked").fetchone()
        for i, c in enumerate(batch):
            phys = physical[c]
            numeric = _is_int(phys) or _is_float(phys) or _is_decimal(phys)
            n_values, whole, indicator, n_present = row[4 * i: 4 * i + 4]
            n_values = int(n_values or 0)
            # BLUEPRINT §14.3: whole numbers are codes or amounts whatever their count (FIPS states
            # hold 51 codes; NHANES ``DR1_030Z`` 21 eating occasions), so a whole-valued number that
            # changes within units is "whole": combined only as the user says (``prepare_combine``).
            # Fractional values are amounts by their values.
            if not int(varied[i] or 0) > 1:
                kind = "constant"
            elif not numeric:
                kind = "category"
            elif whole:
                kind = "whole"
            else:
                kind = "amount"
            out[c] = {"varied": int(varied[i] or 0) > 1, "kind": kind, "numeric": numeric,
                      "whole": bool(numeric and whole), "zero_one": bool(numeric and indicator),
                      "n_values": n_values}
    return out


def code_or_count_facts(frame: Any, unit: str, exclude: Sequence[str] = ()) -> dict[str, dict[str, Any]]:
    """The columns combining a unit's rows must ask about (BLUEPRINT §14.3, the consumer sets the
    scope), over ``frame`` with pandas: every whole-valued number that changes within some unit,
    whatever its count of values or its type (codes written 1.0–5.0 after a blank are whole; 0/1 is
    too, a share under the mean and the majority under the mode), with what the values say
    (``whole``, ``zero_one``, ``n_values``, ``min``, ``max``) for the question's best guess. Values
    with decimals are asked too unless they settle "amount" by the one value test
    (``readings.amounts_by_values``: ICD-9-CM 307.1 and 250.02 are codes written with a decimal
    point, and averaging them makes no diagnosis). Numbers written as text are asked too (the
    sixth gate: a SAS ``.`` makes a BMI text, whose most frequent value is no mean): text the value
    test does not settle as labels."""
    import numpy as np
    import pandas as pd

    from turbotab.core.readings import by_values

    if unit not in frame.columns:
        return {}
    out: dict[str, dict[str, Any]] = {}
    skip = {unit, ROW_ID, *exclude}
    for c in frame.columns:
        if c in skip:
            continue
        x = frame[c]
        if pd.api.types.is_bool_dtype(x):
            continue
        if not pd.api.types.is_numeric_dtype(x):
            head = x.dropna().astype(str).head(50).str.strip().str.lstrip("<>").str.replace(",", ".")
            if head.empty or pd.to_numeric(head, errors="coerce").notna().mean() < 0.5:
                continue
            verdict = by_values("code_or_count", x)
            found = verdict.detail if isinstance(verdict.detail, dict) else {}
            if verdict.settles or not found.get("text_numbers"):
                continue
            varied = frame.groupby(unit, dropna=True)[c].nunique(dropna=True)
            if len(varied) and int(varied.max()) > 1:
                out[str(c)] = {"text_numbers": True, "whole": bool(found.get("whole")),
                               "n_values": int(found.get("distinct") or 0),
                               "distinct": int(found.get("distinct") or 0),
                               "min": found.get("min"), "max": found.get("max"),
                               "below": dict(found.get("below") or {}),
                               "evidence": verdict.evidence}
            continue
        present = x.dropna()
        if present.empty:
            continue
        values = present.to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if not len(values):
            continue
        distinct = np.unique(values)
        if len(distinct) < 2:
            continue
        whole = bool(np.all(values == np.floor(values)))
        facts: dict[str, Any] = {"whole": whole, "zero_one": bool(set(distinct) <= {0.0, 1.0}),
                                 "n_values": int(len(distinct)), "min": float(distinct.min()),
                                 "max": float(distinct.max())}
        if not whole:
            verdict = by_values("code_or_count", pd.Series(values))
            if verdict.settles:
                continue
            facts.update(amount_by_values=False, evidence=verdict.evidence)
        varied = frame.groupby(unit, dropna=True)[c].nunique(dropna=True)
        if len(varied) and int(varied.max()) > 1:
            out[str(c)] = facts
    return out


def column_rule(kind: str, method: str) -> str:
    if kind == "constant":
        return "constant"
    if kind == "amount":
        return method
    return {"mean": "mode", "first": "first", "last": "last", "change": "first"}[method]


def _rule_sql(c: str, rule: str, physical: str) -> str:
    from turbotab.core.datastore import _is_float, _is_int

    q = _ident(c)
    first = f"first({q}) FILTER (WHERE {_K} = 1 AND {_M} > 0)"
    last = f"first({q}) FILTER (WHERE {_K} = {_M})"
    if rule == "constant":
        return f"any_value({q})"
    if rule == "mean":
        return f"avg({q})"
    if rule == "first":
        return first
    if rule == "last":
        return last
    if rule == "mode":
        return f"mode({q} ORDER BY {_K})"  # ties: the earliest record
    # change: last minus first; signed integers and floats keep their type, anything else is DOUBLE
    kind = physical.upper()
    native = _is_float(kind) or (_is_int(kind) and not kind.startswith("U") and "HUGE" not in kind)
    a, b = (last, first) if native else (f"CAST({last} AS DOUBLE)", f"CAST({first} AS DOUBLE)")
    return f"CASE WHEN max({_M}) > 1 THEN {a} - {b} END"


def combine_sql(names: Sequence[str], physical: Mapping[str, str], plan: Mapping[str, Any],
                target: str | None, varies: bool, rules: Mapping[str, str]) -> str:
    """The ``SELECT`` over ``ranked`` that makes one row per unit, column by column.

    ``rules``: column -> its rule (:func:`column_rule`, or the answer's override). The unit's id is
    its one value; the time column is taken from the record the method keeps (the last for
    "last", else the first); the outcome is its one value, or, when it varies, the outcome rule.
    """
    key, method, rule = plan["id_column"], plan["method"], plan["outcome_rule"]
    order = plan["time_column"]
    combined: list[str] = []
    for c in names:
        q = _ident(c)
        if c == key:
            expr = f"any_value({q})"
        elif c == target:
            expr = f"any_value({q})" if not varies else _rule_sql(c, str(rule), physical[c])
        elif c == order:
            expr = _rule_sql(c, "last" if method == "last" else "first", physical[c])
        else:
            expr = _rule_sql(c, rules[c], physical[c])
        combined.append(f"{expr} AS {q}")
    combined.append(f"CAST(row_number() OVER (ORDER BY {_UNIT}) - 1 AS BIGINT) AS {ROW_ID}")
    return f"SELECT {', '.join(combined)} FROM ranked GROUP BY {_UNIT} ORDER BY {_UNIT}"


def outcome_rule_problem(target: str | None, key: str, varies: bool, rule: str | None,
                         dtypes: Mapping[str, str], n_levels: int | None = None) -> str | None:
    """Why the outcome cannot be combined as recorded, or None."""
    if target is None or not varies:
        return None
    if rule is None:
        return (f"`{target}` changes within a `{key}`, so combining the rows needs to know which "
                f"value to keep. Answer the combining question again and choose the outcome.")
    if rule == "mean" and dtypes.get(target) not in NUMERIC:
        return (f"`{target}` is not a number, so its mean cannot be the outcome; keep the first or "
                f"the last value instead.")
    if rule == "mean" and n_levels is not None and n_levels <= 2:
        return (f"`{target}` takes two values, so its mean (a share of records) is not one of its "
                f"levels; keep the first or the last value instead.")
    return None


def prepare_combine(con: Any, src_sql: str, plan: Mapping[str, Any], target: str | None) -> dict[str, Any]:
    """Everything combining decides before it writes, read over the whole source: the types as
    the repairs left them, the time column's order, ``ranked`` (:func:`rank_units`), whether the
    outcome varies, and each column's kind and rule. The working stage and its preview both start
    here, so the preview's rules are the table's. Raises :class:`StructureError` with what to do.
    """
    from turbotab.core.datastore import _is_decimal, _is_float, _is_int

    # The types as the repairs left them (a text column read as numbers is a number now).
    physical = {str(r[0]): str(r[1]) for r in con.execute(f"DESCRIBE SELECT * FROM {src_sql}").fetchall()
                if str(r[0]) != ROW_ID}
    names = list(physical)
    numeric = {c: _is_int(t) or _is_float(t) or _is_decimal(t) for c, t in physical.items()}
    dtypes = {c: ("numeric" if numeric[c] else "other") for c in names}
    key, method, order = plan["id_column"], plan["method"], plan["time_column"]
    target = target if target in physical else None
    rule, overrides = plan["outcome_rule"], dict(plan.get("overrides") or {})

    reading = None
    order_expr = f"CAST({ROW_ID} AS BIGINT)"  # no time column: file order, every record dated
    if order is not None:
        reading = time_order(con, src_sql, order, physical[order], plan.get("levels"))
        if reading["orderable"]:
            order_expr = reading["expr"]
        elif needs_order(method, rule, overrides):
            raise StructureError(order_refusal(reading, method))
    rank_units(con, src_sql, plan, order_expr)

    varies, n_levels = False, None
    if target is not None:
        varies = bool(con.execute(
            f"SELECT count(*) FROM (SELECT count(DISTINCT {_ident(target)}) AS d FROM ranked "
            f"GROUP BY {_UNIT}) WHERE d > 1").fetchone()[0])
        n_levels = int(con.execute(f"SELECT count(DISTINCT {_ident(target)}) FROM ranked").fetchone()[0])
    problem = outcome_rule_problem(target, key, varies, rule, dtypes, n_levels)
    if problem:
        raise StructureError(problem)

    index = plan.get("index_column")
    others = [c for c in names if c not in (key, target, order)]
    kinds = column_kinds(con, others, physical)
    codes, amounts = set(plan.get("codes") or ()), set(plan.get("amounts") or ())
    asked = set(plan.get("asked") or ())
    waiting: list[str] = []
    rules: dict[str, str] = {c: "first" for c in others if c == index and c not in overrides}
    for c in others:
        if c in rules:
            continue
        chosen = overrides.get(c)
        if chosen is not None:
            if chosen not in COMBINE_RULES:
                raise StructureError(f"`{chosen}` is not a way to combine `{c}`; use one of "
                                     f"{', '.join(COMBINE_RULES)}.")
            if chosen in ("mean", "change") and not numeric[c]:
                raise StructureError(f"`{c}` is not a number, so its {chosen} is not one of its "
                                     f"values; keep the first, the last or the most frequent.")
            rules[c] = chosen
        elif kinds[c]["kind"] == "category" and c in amounts:
            # Text the user said holds amounts that the working table could not read as numbers
            # yet: its values below a detection limit (or its commas) wait for their answer.
            raise StructureError(
                f"`{c}` holds amounts, as you said, but some of its values are not numbers yet "
                f"(values below a detection limit, or commas that read two ways): answer how they "
                f"read (the findings' repair for `{c}`) before combining.")
        elif kinds[c]["kind"] == "category" and c in asked and c not in codes:
            waiting.append(c)
        elif kinds[c]["kind"] in ("constant", "category"):
            rules[c] = column_rule(kinds[c]["kind"], method)
        elif c in codes:
            rules[c] = column_rule("code", method)
            kinds[c] = {**kinds[c], "kind": "code"}
        elif c in amounts:
            rules[c] = column_rule("amount", method)
            kinds[c] = {**kinds[c], "kind": "amount"}
        elif kinds[c]["kind"] == "whole" and (c in asked or not plan.get("read_asked")):
            waiting.append(c)
        elif kinds[c]["kind"] == "whole":
            # Whole numbers the structure reading did not ask about (a time column or the records'
            # index, which leave the model by their role): the value at the record each rule takes.
            rules[c] = column_rule("code", method)
        else:
            rules[c] = column_rule("amount", method)
    if waiting:
        # BLUEPRINT §14.3: whole numbers fit a count (cups of coffee per recall) as well as a code
        # (smoking 1/2/3; NHANES ``DR1_030Z``'s 21 eating occasions; FIPS states), whatever their
        # count; combining them by a guess averaged eating occasions into 22.11 in the gate.
        # Asked, each with its best guess.
        from turbotab.core.readings import listing

        one = len(waiting) == 1
        raise StructureError(
            f"{listing(waiting)} {'holds' if one else 'hold'} whole numbers that change within "
            f"units, which may be codes for categories (combined by the most frequent value) or "
            f"counts (combined by the {method}). Say which for {'it' if one else 'each'} before "
            f"combining.")
    for c in others:
        if kinds[c]["kind"] == "whole":  # the receipt's words: how it was combined
            kinds[c] = {**kinds[c], "kind": "code" if rules.get(c) in ("mode", "first") else "amount"}
    return {"physical": physical, "names": names, "numeric": numeric, "target": target,
            "reading": reading, "ordered": reading is not None and bool(reading["orderable"]),
            "order_expr": order_expr, "varies": varies, "others": others, "kinds": kinds,
            "rules": rules, "overrides": overrides}


def _aggregate(ctx: StageContext, con: Any, src_sql: str, info: Mapping[str, Any],
               plan: dict[str, Any], table_out: Path, map_out: Path) -> dict[str, Any]:
    """One row per unit (DuckDB GROUP BY) and the row map; returns what the receipt states."""
    prep = prepare_combine(con, src_sql, plan, ctx.state.target)
    names, physical, numeric = prep["names"], prep["physical"], prep["numeric"]
    target, varies, others = prep["target"], prep["varies"], prep["others"]
    kinds, rules, overrides = prep["kinds"], prep["rules"], prep["overrides"]
    reading, ordered = prep["reading"], prep["ordered"]
    key, method, order, rule = plan["id_column"], plan["method"], plan["time_column"], plan["outcome_rule"]

    select = combine_sql(names, physical, plan, target, varies, rules)
    _execute(con, ctx, f"COPY ({select}) TO {_lit(table_out)} (FORMAT parquet, COMPRESSION zstd)",
             0.35, 0.7, "Combining each unit's rows")
    _execute(con, ctx,
             f"COPY (SELECT CAST(dense_rank() OVER (ORDER BY {_UNIT}) - 1 AS BIGINT) AS row_id, "
             f"{ROW_ID} AS source_row_id FROM ranked ORDER BY row_id, source_row_id) "
             f"TO {_lit(map_out)} (FORMAT parquet, COMPRESSION zstd)",
             0.7, 0.8, "Writing which rows became which")

    n_units, single, no_dated, undated = con.execute(
        f"SELECT count(*), count(*) FILTER (WHERE m = 1), count(*) FILTER (WHERE m = 0), "
        f"sum(n - m) FROM (SELECT max({_M}) AS m, count(*) AS n FROM ranked GROUP BY {_UNIT})"
    ).fetchone()
    outcome = None
    if target is not None:
        missing = con.execute(f"SELECT count(*) FILTER (WHERE {_ident(target)} IS NULL) FROM "
                              f"read_parquet({_lit(table_out)})").fetchone()[0]
        outcome = {"column": target, "varies": varies, "rule": (rule if varies else "constant"),
                   "n_missing": int(missing or 0)}
    con.execute("DROP TABLE ranked")
    listed = [{"column": c, "kind": kinds[c]["kind"], "rule": rules[c], "varied": kinds[c]["varied"],
               "chosen": c in overrides}
              for c in others if kinds[c]["varied"] or c in overrides]
    order_kind = ("file order" if not ordered else
                  "declared levels" if reading["kind"] == "levels" else reading["kind"])
    return {
        "id_column": key, "method": method, "outcome": outcome,
        "time_column": order, "ordered_by": order if ordered else "file order", "order": order_kind,
        "undated_records": int(undated or 0) if ordered else 0,
        "units_without_dated_records": int(no_dated or 0) if ordered else 0,
        "n_source_rows": int(info["n_rows"]), "n_units": int(n_units),
        "single_record_units": int(single), "varying": [c for c in others if kinds[c]["varied"]
                                                        and not numeric[c]],
        "combined_numeric": sum(1 for c in others if numeric[c]),
        "constant_columns": sum(1 for c in others if kinds[c]["kind"] == "constant"),
        "columns": listed,
    }


def text_amounts(state: Any, source: Path, names: Sequence[str],
                 repairs: Mapping[str, str]) -> dict[str, str]:
    """``{column: SQL}`` reading as numbers each text column the user said holds amounts
    (BLUEPRINT §14.3, every confirmation is honored; the sixth gate: a SAS-exported BMI confirmed
    "amount" still entered the fit as 175 indicators). For every consumer, on the working table:
    missing marks (``.``, ``NA``, blanks) and other text become blank; a value below a detection
    limit (``<0.20``) takes the detection-limit answer (the below-detection repair, which reads the
    column itself and stands here), so a column holding one waits for that answer, as does one whose
    commas read two ways. A column a repair already rewrites is the repair's."""
    import duckdb
    import numpy as np
    import pandas as pd

    from turbotab.core.readings import by_values, confirmation, detection_limit
    from turbotab.core.repairs import number_sql

    wanted = [c for c in names if c not in repairs
              and confirmation(state, "code_or_count", c) == "amount"]
    if not wanted:
        return {}
    rel = f"read_parquet({_lit(str(source))})"
    out: dict[str, str] = {}
    con = duckdb.connect()
    try:
        types = {str(r[0]): str(r[1]).upper() for r in con.execute(
            f"DESCRIBE SELECT {', '.join(_ident(c) for c in wanted)} FROM {rel}").fetchall()}
        for c in wanted:
            if not types.get(c, "").startswith("VARCHAR"):
                continue
            q = _ident(c)
            rows = con.execute(f"SELECT {q} AS v, count(*) FROM {rel} WHERE {q} IS NOT NULL "
                               f"GROUP BY v ORDER BY v").fetchall()
            verdict = by_values("code_or_count", pd.Series([r[0] for r in rows], dtype=object),
                                counts=np.asarray([r[1] for r in rows], dtype=float))
            found = verdict.detail if isinstance(verdict.detail, dict) else {}
            if not found.get("text_numbers") or found.get("ambiguous_comma"):
                continue
            factor = detection_limit(state, c)
            if found.get("below") and factor is None:
                continue  # waits for the detection-limit answer (the fit asks for it)
            out[c] = number_sql(q, str(found.get("decimal") or "."), bool(found.get("thousands")),
                                factor if found.get("below") else None)
    finally:
        con.close()
    return out


def source_sql(source: Path, names: Sequence[str], repairs: Mapping[str, str],
               where: str | None = None, derived: Mapping[str, str] | None = None) -> str:
    """The oriented table with the row-local repairs applied, as a subquery. ``where`` keeps only
    the rows it holds for (reference rows leave, audit RO-13); ``derived`` adds columns computed
    row by row (``{name: SQL}``: the log-scale outcome, audit RO-10)."""
    select = ", ".join([*(f"{repairs[c]} AS {_ident(c)}" if c in repairs else _ident(c)
                          for c in names),
                        *(f"{sql} AS {_ident(name)}" for name, sql in (derived or {}).items()),
                        ROW_ID])
    clause = f" WHERE {where}" if where else ""
    return f"(SELECT {select} FROM read_parquet({_lit(source)}){clause})"


def row_local_additions(state: Any, findings: Any, names: Sequence[str],
                        repairs: Mapping[str, str]) -> tuple[str | None, dict[str, str],
                                                             list[dict[str, Any]]]:
    """What the working table takes from WP18's row-local answers: the condition that keeps every
    row but the reference rows a recorded repair excludes, the derived log-scale outcome (each
    value's natural log, blank at or below 0), and the reference rules themselves."""
    from turbotab.core.reference_rows import reference_filter, reference_rules
    from turbotab.core.structural import derived_columns

    rules = [r for r in reference_rules(getattr(state, "findings", None), findings)
             if r["column"] in names]
    derived: dict[str, str] = {}
    for name, column in derived_columns(state).items():
        if column in names and name not in names:
            x = f"TRY_CAST({repairs.get(column, _ident(column))} AS DOUBLE)"
            derived[name] = f"CASE WHEN {x} > 0 THEN ln({x}) END"
    return reference_filter(rules), derived, rules


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
    repairs.update(text_amounts(ctx.state, source, names, repairs))
    unknown = [c for c in repairs if c not in names]
    if unknown:
        raise StructureError(f"A recorded repair names `{unknown[0]}`, which is not a column.")
    from turbotab.core.structural import derived_columns

    where, derived, rules = row_local_additions(ctx.state, findings, names, repairs)
    plan = aggregation_plan(ctx.state, set(names) | set(derived), structure)
    base = {"transposed": transposed, "n_source_rows": int(info["n_rows"]),
            "repairs": [{"column": c, "expression": e} for c, e in repairs.items()],
            # WP18: the reference rows that left (RO-13) and the columns derived row by row (RO-10)
            "reference_rows": [],
            "derived": [{"column": name, "source": src, "expression": "ln"}
                        for name, src in derived_columns(ctx.state).items() if name in derived]}
    if rules:
        from turbotab.core.reference_rows import reference_filter

        con, temp = _connect(ctx, "working")
        try:
            for rule in rules:
                gone = con.execute(f"SELECT count(*) FROM read_parquet({_lit(source)}) WHERE NOT "
                                   f"({reference_filter([rule])})").fetchone()[0]
                base["reference_rows"].append({**rule, "n": int(gone or 0)})
        finally:
            con.close()
            _cleanup(temp)
    if not repairs and plan is None and not where and not derived:
        data = {**_dataset_fields(info), **base, "pass_through": True, "aggregation": None,
                "row_map": "identity"}
        return Bundle(data=data, files={TABLE: _reference(source, ctx, "working")})

    started = time.perf_counter()
    src_sql = source_sql(source, names, repairs, where=where, derived=derived)
    table_out = _scratch(ctx, "working", TABLE)
    sidecar = _scratch(ctx, "working", SIDECAR)
    map_out = _scratch(ctx, "working", ROW_MAP)
    con, temp = _connect(ctx, "working")
    files: dict[str, Path] = {TABLE: table_out, SIDECAR: sidecar}
    try:
        aggregation = None
        if plan is None and where:
            # Reference rows left (WP18, RO-13): the rows that stay are numbered 0…n − 1 again, as
            # every reader of the working table expects, and the row map says which source row
            # each one is.
            renumbered = f"CAST(row_number() OVER (ORDER BY {ROW_ID}) - 1 AS BIGINT)"
            _execute(con, ctx, f"COPY (SELECT * REPLACE ({renumbered} AS {ROW_ID}) FROM "
                               f"{src_sql} ORDER BY {ROW_ID}) TO {_lit(table_out)} "
                               f"(FORMAT parquet, COMPRESSION zstd)",
                     0.1, 0.7, "Applying the repairs and leaving out the reference rows")
            _execute(con, ctx, f"COPY (SELECT {renumbered} AS row_id, {ROW_ID} AS source_row_id "
                               f"FROM {src_sql} ORDER BY row_id) TO {_lit(map_out)} "
                               f"(FORMAT parquet, COMPRESSION zstd)",
                     0.7, 0.8, "Writing which rows stayed")
            files[ROW_MAP] = map_out
        elif plan is None:
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
            "row_map": ROW_MAP if ROW_MAP in files else "identity"}
    return Bundle(data=data, files=files)


__all__ = [
    "CODE_LEVELS", "COMBINE_RULES", "FEATURES", "ROW_MAP", "SAMPLE_COLUMN", "TABLE",
    "StructureError", "aggregation_plan", "column_kinds", "column_rule", "combine_sql",
    "declared_date_formats", "declared_levels", "effective_repeat_kind", "feature_names",
    "frame_time_order", "is_feature_annotation", "label_column", "natural_order", "needs_order",
    "numeric_features", "order_refusal", "orientation_reading", "oriented_stage",
    "outcome_rule_problem", "prepare_combine", "rank_units", "read_date_columns",
    "repair_expressions", "row_map", "source_sql", "structure_stage", "table_info", "table_path",
    "time_column", "time_order", "transpose", "turn_block", "turn_check", "turn_plan",
    "turned_corner", "working_paths", "working_stage", "working_store",
]
