"""The model matrix as the export records it and a replay compares it (V2 definition of done §3.6:
"Replaying the record reproduces the model matrix and the estimates").

The design stage keeps the matrix its shared steps made (impute → energy model → forms → levels →
one-hot; ``models/pipeline.shared_steps``), indexed by row id, as a file of its cache artifact
(:data:`FILE`, written by :func:`write`), so no stage downstream reads it. The lineage figure ends
in it. The bundle never carries its rows, which are the participants' data: it carries two hashes
of it (:func:`record_file`).

* **The parquet hash** (:func:`canonical_parquet`): SHA-256 of the matrix written as Parquet with
  every writer option fixed (a ``row_id`` column, then the matrix's columns in their order; numbers
  as IEEE doubles; no pandas metadata; no statistics; zstd at level 3). The same matrix written by
  the same pyarrow is the same bytes, so a replay on the same software reproduces it byte for byte.
* **The content hash** (:func:`content_sha256`): SHA-256 over the values themselves (the column
  names, the row ids as little-endian int64, each column as little-endian doubles with every NaN
  written as one bit pattern). It does not depend on how Parquet is encoded, so a replay under
  another pyarrow can still say whether the matrix is the same.
"""
from __future__ import annotations

import hashlib
import io
import json
from typing import Any

import numpy as np
import pandas as pd

ROW_ID = "row_id"
FILE = "model_matrix.parquet"  # the design artifact's file
COMPRESSION = "zstd"
COMPRESSION_LEVEL = 3
NAN = np.frombuffer(np.float64(np.nan).tobytes(), dtype="<f8")[0]  # the one NaN bit pattern


def cacheable(matrix: pd.DataFrame) -> pd.DataFrame:
    """The matrix as the design stage's cache keeps it: string column names (unique), numbers as
    they are, booleans as 0/1, anything else as text; the index named ``row_id``. Never raises on
    an odd column: the design stage must not fail for the export's sake."""
    index = pd.Index(np.asarray(matrix.index), name=ROW_ID)
    names = [str(c) for c in matrix.columns]
    plain = all(pd.api.types.is_numeric_dtype(t) and not pd.api.types.is_bool_dtype(t)
                for t in matrix.dtypes)
    if plain and len(set(names)) == len(names):  # the usual case, 20,000 columns or 20
        out = matrix.copy(deep=False)
        out.columns = names
        out.index = index
        return out
    data: dict[str, Any] = {}
    seen: dict[str, int] = {}
    for i, text in enumerate(names):
        if text in seen:  # never expected of a fitted pipeline; kept apart rather than lost
            seen[text] += 1
            text = f"{text}#{seen[text]}"
        else:
            seen[text] = 1
        values = matrix.iloc[:, i]
        if pd.api.types.is_bool_dtype(values):
            values = values.astype(float)
        elif not pd.api.types.is_numeric_dtype(values):
            values = values.astype(object).where(values.notna(), None).map(
                lambda v: None if v is None else str(v))
        data[text] = values.to_numpy()
    return pd.DataFrame(data, index=index)


def _columns(frame: pd.DataFrame) -> list[tuple[str, np.ndarray | list[str | None]]]:
    cols: list[tuple[str, Any]] = []
    for name in frame.columns:
        values = frame[name]
        if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values):
            cols.append((str(name), values.to_numpy(dtype="float64")))
        elif pd.api.types.is_bool_dtype(values):
            cols.append((str(name), values.to_numpy(dtype="float64")))
        else:
            cols.append((str(name), [None if v is None or (isinstance(v, float) and np.isnan(v))
                                     else str(v) for v in values.tolist()]))
    return cols


def row_ids(frame: pd.DataFrame) -> np.ndarray:
    return np.asarray(frame.index, dtype="int64")


def canonical_parquet(frame: pd.DataFrame) -> bytes:
    """The matrix as Parquet bytes with every writer option fixed (module docstring)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    arrays = [pa.array(row_ids(frame), type=pa.int64())]
    names = [ROW_ID]
    for name, values in _columns(frame):
        names.append(name)
        if isinstance(values, np.ndarray):
            arrays.append(pa.array(values, type=pa.float64()))
        else:
            arrays.append(pa.array(values, type=pa.string()))
    table = pa.Table.from_arrays(arrays, names=names)
    sink = io.BytesIO()
    pq.write_table(table, sink, compression=COMPRESSION, compression_level=COMPRESSION_LEVEL,
                   use_dictionary=False, write_statistics=False, store_schema=False,
                   version="2.6", data_page_version="1.0")
    return sink.getvalue()


def content_sha256(frame: pd.DataFrame) -> str:
    """SHA-256 over the matrix's values, independent of any file format (module docstring)."""
    digest = hashlib.sha256()
    columns = _columns(frame)
    header = {"format": "turbotab-matrix/1", "n_rows": int(len(frame)),
              "columns": [[name, "float64" if isinstance(v, np.ndarray) else "string"]
                          for name, v in columns]}
    digest.update(json.dumps(header, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False).encode("utf-8"))
    digest.update(row_ids(frame).astype("<i8").tobytes())
    for _, values in columns:
        if isinstance(values, np.ndarray):
            v = values.astype("<f8", copy=True)
            v[np.isnan(v)] = NAN
            digest.update(v.tobytes())
        else:
            for item in values:
                digest.update(b"\x01" if item is None else b"\x00" + item.encode("utf-8") + b"\x00")
    return digest.hexdigest()


def write(matrix: pd.DataFrame, path: Any) -> None:
    """The matrix's canonical Parquet bytes (:func:`canonical_parquet`), at ``path``."""
    from pathlib import Path

    Path(path).write_bytes(canonical_parquet(cacheable(matrix)))


def read(raw: bytes) -> pd.DataFrame:
    """The matrix a canonical Parquet file holds, indexed by row id."""
    import pyarrow.parquet as pq

    frame = pq.read_table(io.BytesIO(raw)).to_pandas()
    return frame.set_index(ROW_ID)


def record_file(raw: bytes) -> dict[str, Any]:
    """:func:`record` of the matrix a canonical Parquet file holds; the parquet hash is the
    file's own bytes'."""
    out = record(read(raw))
    if hashlib.sha256(raw).hexdigest() != out["parquet_sha256"]:
        # Written by another pyarrow: its bytes are what the analysis left; say so by hashing them.
        out["parquet_sha256"] = hashlib.sha256(raw).hexdigest()
    return out


def record(frame: pd.DataFrame) -> dict[str, Any]:
    """What the provenance record keeps of the matrix: its shape, its columns and both hashes."""
    import pyarrow as pa

    raw = canonical_parquet(frame)
    return {
        "n_rows": int(len(frame)),
        "n_cols": int(frame.shape[1]),
        "columns": [str(c) for c in frame.columns],
        "parquet_sha256": hashlib.sha256(raw).hexdigest(),
        "content_sha256": content_sha256(frame),
        "parquet": {"writer": f"pyarrow {pa.__version__}", "compression": COMPRESSION,
                    "compression_level": COMPRESSION_LEVEL, "row_id": ROW_ID},
    }


__all__ = ["FILE", "ROW_ID", "cacheable", "canonical_parquet", "content_sha256", "read", "record",
           "record_file", "row_ids", "write"]
