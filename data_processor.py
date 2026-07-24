"""
Data processing utilities for the interactive predictor.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Any, List, Tuple, Optional, Union
import io
import json


def detect_file_type(filename: str) -> str:
    """Detect file type from filename extension."""
    filename_lower = filename.lower()
    if filename_lower.endswith('.csv'):
        return 'csv'
    elif filename_lower.endswith(('.xlsx', '.xls')):
        return 'excel'
    elif filename_lower.endswith('.parquet'):
        return 'parquet'
    elif filename_lower.endswith(('.jsonl', '.ndjson')):
        return 'jsonl'
    elif filename_lower.endswith('.json'):
        return 'json'
    elif filename_lower.endswith(('.tsv', '.txt')):
        return 'tsv'
    else:
        # Default to CSV for unknown types
        return 'csv'


def transpose_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Transpose a DataFrame (rows ↔ columns)."""
    return df.T


def load_csv(file: Union[str, io.BytesIO], encoding: Optional[str] = None) -> pd.DataFrame:
    """Load CSV file. Tries utf-8 first, then latin-1 on failure."""
    encodings = [encoding] if encoding else ['utf-8', 'latin-1', 'cp1252']
    last_err = None
    for enc in encodings:
        try:
            if isinstance(file, str):
                df = pd.read_csv(file, encoding=enc)
            else:
                file.seek(0)
                df = pd.read_csv(file, encoding=enc)
            return df
        except Exception as e:
            last_err = e
    raise ValueError(f"Error loading CSV: {str(last_err)}")


def load_excel(file: Union[str, io.BytesIO], sheet_name: Optional[Union[str, int]] = 0) -> pd.DataFrame:
    """Load Excel file."""
    try:
        if isinstance(file, str):
            df = pd.read_excel(file, sheet_name=sheet_name)
        else:
            df = pd.read_excel(file, sheet_name=sheet_name)
        return df
    except Exception as e:
        raise ValueError(f"Error loading Excel: {str(e)}")


def load_parquet(file: Union[str, io.BytesIO]) -> pd.DataFrame:
    """Load Parquet file."""
    try:
        if isinstance(file, str):
            df = pd.read_parquet(file)
        else:
            df = pd.read_parquet(file)
        return df
    except Exception as e:
        raise ValueError(f"Error loading Parquet: {str(e)}")


def load_tsv(file: Union[str, io.BytesIO], encoding: Optional[str] = None) -> pd.DataFrame:
    """Load TSV (tab-separated) file. Tries utf-8 first, then latin-1 on failure."""
    encodings = [encoding] if encoding else ['utf-8', 'latin-1', 'cp1252']
    last_err = None
    for enc in encodings:
        try:
            if isinstance(file, str):
                df = pd.read_csv(file, sep='\t', encoding=enc)
            else:
                file.seek(0)
                df = pd.read_csv(file, sep='\t', encoding=enc)
            return df
        except Exception as e:
            last_err = e
    raise ValueError(f"Error loading TSV: {str(last_err)}")


# Keys commonly used to wrap a records array in API-style payloads
# ({"data": [...]}, {"results": [...]}). Ordered by how conventional they are.
_JSON_WRAPPER_KEYS = ("data", "records", "results", "rows", "items", "entries", "values")

# Nested objects are flattened this many levels by default ({"a": {"b": 1}} ->
# column "a.b"). Deeper nesting usually means the file is not really tabular.
_JSON_MAX_LEVEL = 2


def _strip_bom(text: str) -> str:
    """Drop a leading UTF-8 BOM.

    Windows and Excel exports routinely include one; decoding as plain utf-8
    leaves it as a stray \\ufeff character that makes json.loads fail on the
    very first byte with an unhelpful error.
    """
    return text.lstrip('﻿') if text else text


def _json_read_text(file: Union[str, io.BytesIO], encoding: Optional[str] = None) -> str:
    """Read a JSON/JSONL source to text, tolerating common encodings."""
    if isinstance(file, str):
        encodings = [encoding] if encoding else ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
        last_err = None
        for enc in encodings:
            try:
                with open(file, 'r', encoding=enc) as fh:
                    return _strip_bom(fh.read())
            except Exception as e:
                last_err = e
        raise ValueError(f"Error reading JSON file: {last_err}")

    file.seek(0)
    raw = file.read()
    if isinstance(raw, str):
        return _strip_bom(raw)
    encodings = [encoding] if encoding else ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
    last_err = None
    for enc in encodings:
        try:
            return _strip_bom(raw.decode(enc))
        except Exception as e:
            last_err = e
    raise ValueError(f"Error decoding JSON file: {last_err}")


def _json_lines_to_records(text: str) -> List[Any]:
    """Parse JSON Lines / NDJSON text into a list of objects."""
    records = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip().rstrip(',')
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Line {lineno} is not valid JSON ({e.msg}). JSON Lines files "
                f"need exactly one complete JSON object per line."
            )
    if not records:
        raise ValueError("The file contains no JSON records.")
    return records


def _json_obj_to_frame(obj: Any, records_key: Optional[str] = None,
                       max_level: int = _JSON_MAX_LEVEL) -> pd.DataFrame:
    """Convert a parsed JSON object into a DataFrame, or explain why it can't.

    Handles the shapes researchers actually export: an array of records, a
    wrapped payload ({"data": [...]}), pandas' own orients (split/columns/
    index/table), and nested objects (flattened via json_normalize).
    """
    # Caller explicitly told us where the rows live.
    if records_key is not None:
        if not isinstance(obj, dict) or records_key not in obj:
            raise ValueError(f"Key '{records_key}' was not found at the top level of this JSON.")
        return _json_obj_to_frame(obj[records_key], None, max_level)

    # --- array at the top level -------------------------------------------
    if isinstance(obj, list):
        if not obj:
            raise ValueError("This JSON contains an empty list — there are no rows to load.")
        if all(isinstance(x, dict) for x in obj):
            return pd.json_normalize(obj, max_level=max_level)
        if all(isinstance(x, (list, tuple)) for x in obj):
            return pd.DataFrame(obj)
        if all(not isinstance(x, (dict, list, tuple)) for x in obj):
            return pd.DataFrame({"value": obj})
        # Mixed content — normalize what we can rather than fail outright.
        return pd.json_normalize(
            [x if isinstance(x, dict) else {"value": x} for x in obj], max_level=max_level
        )

    # --- object at the top level ------------------------------------------
    if isinstance(obj, dict):
        keys = set(obj.keys())

        # pandas orient='split': {"columns": [...], "data": [[...]], "index": [...]}
        if {"columns", "data"} <= keys and isinstance(obj.get("data"), list):
            try:
                df = pd.DataFrame(obj["data"], columns=obj["columns"])
                if isinstance(obj.get("index"), list) and len(obj["index"]) == len(df):
                    df.index = obj["index"]
                return df
            except Exception:
                pass  # fall through to the generic handling below

        # pandas orient='table': {"schema": {...}, "data": [...]}
        if {"schema", "data"} <= keys and isinstance(obj.get("data"), list):
            return pd.json_normalize(obj["data"], max_level=max_level)

        # API-style wrapper: {"data": [...]}, {"results": [...]}, ...
        for wrapper in _JSON_WRAPPER_KEYS:
            value = obj.get(wrapper)
            if isinstance(value, list) and value:
                return _json_obj_to_frame(value, None, max_level)

        # Exactly one key holds a list of records — unambiguous enough to use.
        list_keys = [k for k, v in obj.items() if isinstance(v, list) and v]
        if len(list_keys) == 1:
            return _json_obj_to_frame(obj[list_keys[0]], None, max_level)

        # pandas orient='columns'/'index': {"col": {"0": v, "1": v}, ...}
        if obj and all(isinstance(v, dict) for v in obj.values()):
            return pd.DataFrame(obj)

        # A single flat record -> a one-row table.
        if obj and all(not isinstance(v, (dict, list)) for v in obj.values()):
            return pd.DataFrame([obj])

        if len(list_keys) > 1:
            raise ValueError(
                "This JSON has several possible row sets "
                f"({', '.join(sorted(list_keys)[:6])}). Pick which key holds "
                "your rows, or export just that part of the file."
            )
        raise ValueError(
            "This JSON does not look like a table. Top-level keys: "
            f"{', '.join(sorted(map(str, list(keys)))[:6]) or '(none)'}. "
            "A tabular JSON is usually a list of records, e.g. "
            '[{"age": 40, "bmi": 22.1}, ...].'
        )

    raise ValueError(
        "This JSON is a single value, not a table. A tabular JSON is usually a "
        'list of records, e.g. [{"age": 40, "bmi": 22.1}, ...].'
    )


def load_json(file: Union[str, io.BytesIO], lines: bool = False,
              records_key: Optional[str] = None,
              encoding: Optional[str] = None) -> pd.DataFrame:
    """Load JSON or JSON Lines data as a DataFrame.

    Args:
        file: path or file-like object
        lines: force JSON Lines (NDJSON) parsing — one object per line
        records_key: top-level key holding the rows, when auto-detection is
            ambiguous (e.g. "data")
        encoding: override text encoding
    """
    text = _json_read_text(file, encoding=encoding)
    if not text.strip():
        raise ValueError("The JSON file is empty.")

    if lines:
        return _json_obj_to_frame(_json_lines_to_records(text), records_key)

    try:
        obj = json.loads(text)
    except json.JSONDecodeError as first_err:
        # A .json file that is really NDJSON is a very common export; try it
        # before giving up so the user isn't blocked on a naming convention.
        try:
            return _json_obj_to_frame(_json_lines_to_records(text), records_key)
        except Exception:
            raise ValueError(
                f"This file is not valid JSON ({first_err.msg} at line "
                f"{first_err.lineno}, column {first_err.colno})."
            )

    df = _json_obj_to_frame(obj, records_key)
    if df.shape[1] == 0:
        raise ValueError("This JSON produced a table with no columns.")
    return df


def load_tabular_data(
    file: Union[str, io.BytesIO],
    filename: Optional[str] = None,
    transpose: bool = False,
    excel_sheet: Optional[Union[str, int]] = 0
) -> pd.DataFrame:
    """
    Load tabular data from various formats (CSV, Excel, Parquet, TSV).
    
    Args:
        file: File path or file-like object
        filename: Original filename (used to detect file type if file is BytesIO)
        transpose: Whether to transpose the data after loading
        excel_sheet: Sheet name or index for Excel files (default: first sheet)
    
    Returns:
        Loaded DataFrame, optionally transposed
    """
    # Detect file type
    if filename:
        file_type = detect_file_type(filename)
    elif isinstance(file, str):
        file_type = detect_file_type(file)
    else:
        # Default to CSV if we can't determine
        file_type = 'csv'
    
    # Load based on file type
    if file_type == 'csv':
        df = load_csv(file)
    elif file_type == 'excel':
        df = load_excel(file, sheet_name=excel_sheet)
    elif file_type == 'parquet':
        df = load_parquet(file)
    elif file_type == 'tsv':
        df = load_tsv(file)
    elif file_type == 'json':
        df = load_json(file)
    elif file_type == 'jsonl':
        df = load_json(file, lines=True)
    else:
        # Fallback to CSV
        df = load_csv(file)
    
    # Transpose if requested
    if transpose:
        df = transpose_dataframe(df)
    
    return df


def load_and_preview_csv(file_path: str, n_rows: int = 5) -> pd.DataFrame:
    """Load CSV and return preview. (Legacy function for backward compatibility)"""
    return load_csv(file_path)


def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    """Get list of numeric column names (at least one non-null).

    select_dtypes(np.number) already guarantees numeric dtype, so no
    per-column value revalidation is needed — a float64/int64 column cannot
    hold strings. The vectorized notna scan keeps this O(cells in C), which
    matters on wide data where a Python per-column loop costs seconds.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return []
    has_data = df[numeric_cols].notna().any()
    return [col for col in numeric_cols if has_data[col]]


def get_categorical_columns(df: pd.DataFrame) -> List[str]:
    """Get list of categorical column names (object, category, bool; at least one non-null)."""
    cand = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    return [col for col in cand if df[col].notna().sum() > 0]


def get_selectable_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Return (numeric_cols, categorical_cols) for target/feature selection.
    Use numeric + categorical for the full selectable pool.
    """
    numeric = get_numeric_columns(df)
    categorical = get_categorical_columns(df)
    return numeric, categorical


def prepare_data(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42
) -> Tuple:
    """
    Prepare data for training.
    
    Returns:
        (X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names)
    """
    # Check columns exist
    missing = set([target_col] + feature_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    # Extract features and target
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Handle missing values - target: drop rows (categorical) or median (numeric)
    target_is_categorical = y.dtype in ['object', 'category', 'bool'] or (
        hasattr(y.dtype, 'kind') and y.dtype.kind in ('O', 'b')
    )
    if target_is_categorical:
        mask = y.notna()
        X = X[mask].reset_index(drop=True)
        y = y[mask].reset_index(drop=True)
    else:
        y = y.fillna(y.median())
    
    X = X.fillna(X.median())
    
    # Convert to numeric, coercing errors to NaN
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    if not target_is_categorical:
        y = pd.to_numeric(y, errors='coerce')
    
    # Drop rows with NaN in target (numeric only; categorical already dropped)
    if not target_is_categorical:
        mask = y.notna()
        X = X[mask].reset_index(drop=True)
        y = y[mask].reset_index(drop=True)
    
    # Drop rows with all NaN features
    mask = X.notna().any(axis=1)
    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)
    
    # Fill remaining NaN with median
    X = X.fillna(X.median())
    
    # Split data
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=(test_size + val_size), random_state=seed
    )
    
    rel_val = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=(1.0 - rel_val), random_state=seed
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    return (
        X_train_scaled, X_val_scaled, X_test_scaled,
        y_train.values, y_val.values, y_test.values,
        scaler, feature_cols
    )


def validate_data_selection(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    task_type: Optional[str] = None,
) -> Tuple[bool, str]:
    """Validate that data selection is valid. Target/features must be in selectable pool (numeric + categorical)."""
    if not target_col:
        return False, "Please select a target column"

    if not feature_cols:
        return False, "Please select at least one feature column"

    if target_col in feature_cols:
        return False, "Target column cannot be in feature columns"

    if target_col not in df.columns:
        return False, f"Target column '{target_col}' not found in data"

    missing = set(feature_cols) - set(df.columns)
    if missing:
        return False, f"Feature columns not found: {missing}"

    numeric_cols, categorical_cols = get_selectable_columns(df)
    selectable = set(numeric_cols) | set(categorical_cols)

    if target_col not in selectable:
        return False, f"Target column '{target_col}' must be numeric or categorical (selectable)"

    invalid_features = set(feature_cols) - selectable
    if invalid_features:
        return False, f"Feature columns must be numeric or categorical: {invalid_features}"

    target_is_categorical = target_col in categorical_cols
    if target_is_categorical and task_type == "regression":
        return False, "Categorical target is only supported for classification; use a numeric target for regression."

    return True, "OK"
