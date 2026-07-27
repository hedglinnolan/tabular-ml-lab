"""The evidence a finding card puts on the table.

The drive's complaint, in one sentence: the physiologic card says "8 flags" and
offers "Verify units and data entry", and never shows the eight. The engine
holds the specific entries — which column, which row, which value, which bound
it violated — and the card printed a count and a piece of advice.

So this module is the other half of every claim: the flagged rows with their
values, or the plot the claim is about. It computes; it renders nothing. "The
engine refuses to guess" is a legitimate answer and stays available. "The engine
refuses to show" is not.

Three kinds of evidence:

  * **entries** — the specific cells behind a claim, addressed by row label,
    with the value and the bound it broke.
  * **histogram** — the shape behind a shape claim. A skew number with no
    distribution asks the reader to take shape on faith.
  * **correlation matrix** — pairwise structure, offered only where the feature
    count makes a matrix legible.

Findings: GUIDED-003, GUIDED-004, GUIDED-005.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ml.clinical_units import infer_unit
from ml.physiology_reference import (
    get_impossibility_band,
    get_reference_interval,
    load_reference_bundle,
    match_variable_key,
)

# Above this, a per-feature histogram pager and a correlation matrix stop being
# exploration and become a wall of plots. The drive asked for these only on
# small feature spaces, and the number is the design ruling, not a guess.
MAX_FEATURES_FOR_GALLERY = 50

# How many flagged entries travel with a card. The count is always exact; the
# list is capped, and says so, because a card is not a data export.
MAX_ENTRIES = 12

# Above this share, "these entries are impossible" is the less likely reading.
# A quarter of a column outside a hard physiologic floor is what a unit mismatch
# looks like, or a column whose name merely resembles a reference variable —
# `hba1c_proxy` matching `hba1c` is enough. Proposing per-entry deletion there
# would delete real data over a naming coincidence, and §09's budget rule says a
# false positive that cannot be cleanly resolved costs more than a missed
# advisory. So the block is still reported, with the reading corrected.
WHOLE_COLUMN_SUSPECT_SHARE = 0.25


def _label(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        f = float(value)
        return None if math.isnan(f) else f
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value if isinstance(value, (str, int, float, bool)) or value is None else str(value)


def _finite(value: Any) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(f) or math.isinf(f)) else f


# ─────────────────────────────────────────────────────────────────────────────
# Plausibility, in two tiers
# ─────────────────────────────────────────────────────────────────────────────

def plausibility_report(df: pd.DataFrame,
                        columns: Optional[Sequence[str]] = None,
                        reference: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Per-column impossible and improbable entries, with the rows named.

    `impossible` entries carry a repair proposal (set to missing); `improbable`
    ones stay advisory. The split is the whole point: a diastolic pressure of
    ~0 is an entry error, and calling it an outlier invites the user to model
    around it.
    """
    ref = (reference or load_reference_bundle())["nhanes"] if reference is None \
        else reference
    cols = list(columns) if columns is not None else [
        c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    impossible: List[Dict[str, Any]] = []
    improbable: List[Dict[str, Any]] = []

    for col in cols:
        if col not in df.columns:
            continue
        series = df[col]
        if isinstance(series, pd.DataFrame) or not pd.api.types.is_numeric_dtype(series):
            continue
        var_key = match_variable_key(str(col), ref)
        if not var_key:
            continue
        present = series.dropna()
        if present.empty:
            continue

        unit_info = infer_unit(str(col), present) or {}
        factor = unit_info.get("conversion_factor")
        if not factor:
            continue
        converted = present * factor

        band = get_impossibility_band(ref, var_key)
        if band is not None:
            floor, ceiling, unit = band
            hit = converted[(converted < floor) | (converted > ceiling)]
            if len(hit):
                impossible.append(_entry_block(
                    col, var_key, present, converted, hit, floor, ceiling, unit,
                    tier="impossible"))

        interval = get_reference_interval(ref, var_key)
        if interval is not None:
            low, high, unit = interval
            outside = converted[(converted < low) | (converted > high)]
            if band is not None:
                floor, ceiling, _ = band
                outside = outside[(outside >= floor) & (outside <= ceiling)]
            if len(outside):
                improbable.append(_entry_block(
                    col, var_key, present, converted, outside, low, high, unit,
                    tier="improbable"))

    # The count that earns a repair proposal excludes the whole-column suspects,
    # so a naming coincidence cannot inflate the number the interface reports as
    # impossible entries.
    repairable = [b for b in impossible if not b["whole_column_suspect"]]
    return {
        "reference_version": ref.get("version"),
        "impossible": impossible,
        "improbable": improbable,
        "n_impossible": sum(b["n_flagged"] for b in repairable),
        "n_improbable": sum(b["n_flagged"] for b in improbable),
        "n_suspect_columns": sum(1 for b in impossible if b["whole_column_suspect"]),
    }


def _entry_block(column, var_key, present, converted, hit,
                 low, high, unit, tier) -> Dict[str, Any]:
    entries = []
    for label in list(hit.index)[:MAX_ENTRIES]:
        value = _finite(present.loc[label])
        shown = _finite(converted.loc[label])
        if shown is None:
            continue
        entries.append({
            "row": _label(label),
            "value": value,
            "in_reference_unit": shown,
            "side": "below" if shown < low else "above",
            "bound": low if shown < low else high,
        })
    share = float(len(hit) / len(present)) if len(present) else 0.0
    suspect = bool(tier == "impossible" and share > WHOLE_COLUMN_SUSPECT_SHARE)
    return {
        "column": str(column),
        "variable": var_key,
        "tier": tier,
        "unit": unit,
        "low": low,
        "high": high,
        "n_flagged": int(len(hit)),
        "n_present": int(len(present)),
        "share": share,
        "entries": entries,
        "truncated": bool(len(hit) > len(entries)),
        # When most of a column is outside a hard bound, the entries are not the
        # problem — the reading of the column is. Carried as a flag rather than
        # by dropping the block, because staying silent about the values would
        # be the other way of asserting something false.
        "whole_column_suspect": suspect,
        "suspect_reason": (
            f"{share:.0%} of `{column}` falls outside the impossibility band for "
            f"`{var_key}`. That is what a unit mismatch looks like, or a column "
            f"whose name merely resembles a reference variable — not a column of "
            f"entry errors. No per-entry repair is proposed here; check the unit "
            f"and whether `{column}` is really `{var_key}`."
        ) if suspect else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Shape
# ─────────────────────────────────────────────────────────────────────────────

def histogram(series: pd.Series, bins: int = 24) -> Optional[Dict[str, Any]]:
    """Counts and edges for one numeric column. None when there is no shape."""
    if series is None or isinstance(series, pd.DataFrame):
        return None
    values = pd.to_numeric(series, errors="coerce").dropna()
    values = values[np.isfinite(values)]
    if len(values) < 2 or float(values.min()) == float(values.max()):
        return None
    counts, edges = np.histogram(values.to_numpy(), bins=min(bins, max(4, len(values) // 2)))
    skew = _finite(values.skew())
    return {
        "counts": [int(c) for c in counts],
        "edges": [float(e) for e in edges],
        "n": int(len(values)),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": _finite(values.mean()),
        "median": _finite(values.median()),
        "skew": skew,
    }


def histogram_gallery(df: pd.DataFrame,
                      columns: Optional[Sequence[str]] = None,
                      page: int = 0,
                      per_page: int = 6) -> Dict[str, Any]:
    """One page of per-feature histograms, for small feature spaces only.

    Above `MAX_FEATURES_FOR_GALLERY` this returns `available: False` with the
    reason, rather than paging through four hundred plots — and the caller
    renders the refusal instead of a control that would produce one.
    """
    numeric = [c for c in (columns if columns is not None else df.columns)
               if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric) > MAX_FEATURES_FOR_GALLERY:
        return {
            "available": False,
            "n_features": len(numeric),
            "limit": MAX_FEATURES_FOR_GALLERY,
            "reason": (f"{len(numeric)} numeric features. A per-feature gallery is "
                       f"offered up to {MAX_FEATURES_FOR_GALLERY}; beyond that it is "
                       "a wall of plots rather than a way to see anything."),
            "plots": [],
        }
    page = max(0, int(page))
    per_page = max(1, int(per_page))
    window = numeric[page * per_page:(page + 1) * per_page]
    plots = []
    for col in window:
        h = histogram(df[col])
        if h is not None:
            plots.append({"column": str(col), **h})
    return {
        "available": True,
        "n_features": len(numeric),
        "page": page,
        "per_page": per_page,
        "n_pages": max(1, (len(numeric) + per_page - 1) // per_page),
        "plots": plots,
    }


def correlation_matrix(df: pd.DataFrame,
                       columns: Optional[Sequence[str]] = None,
                       method: str = "pearson") -> Dict[str, Any]:
    """Pairwise correlations, for small feature spaces only.

    Same gate as the histogram pager, for the same reason: a 400×400 heatmap
    asserts that it can be read.
    """
    numeric = [c for c in (columns if columns is not None else df.columns)
               if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric) > MAX_FEATURES_FOR_GALLERY:
        return {
            "available": False,
            "n_features": len(numeric),
            "limit": MAX_FEATURES_FOR_GALLERY,
            "reason": (f"{len(numeric)} numeric features. A correlation matrix is "
                       f"offered up to {MAX_FEATURES_FOR_GALLERY}; a grid that large "
                       "cannot be read, and a picture that cannot be read is a claim "
                       "that it can."),
            "columns": [], "matrix": [],
        }
    if len(numeric) < 2:
        return {
            "available": False,
            "n_features": len(numeric),
            "limit": MAX_FEATURES_FOR_GALLERY,
            "reason": "A correlation matrix needs at least two numeric features.",
            "columns": [], "matrix": [],
        }
    corr = df[numeric].corr(method=method)
    matrix = [[_finite(corr.iat[i, j]) for j in range(len(numeric))]
              for i in range(len(numeric))]
    pairs = []
    for i in range(len(numeric)):
        for j in range(i + 1, len(numeric)):
            r = matrix[i][j]
            if r is not None and abs(r) >= 0.8:
                pairs.append({"a": str(numeric[i]), "b": str(numeric[j]), "r": r})
    pairs.sort(key=lambda p: -abs(p["r"]))
    return {
        "available": True,
        "method": method,
        "n_features": len(numeric),
        "columns": [str(c) for c in numeric],
        "matrix": matrix,
        "strong_pairs": pairs[:10],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Entries behind a named claim
# ─────────────────────────────────────────────────────────────────────────────

def flagged_entries(df: pd.DataFrame, column: str,
                    mask: pd.Series, limit: int = MAX_ENTRIES) -> Dict[str, Any]:
    """Rows where `mask` is true, with the value that made it true."""
    if column not in df.columns:
        return {"column": str(column), "n_flagged": 0, "entries": [], "truncated": False}
    hit = df.index[mask.fillna(False).to_numpy()]
    entries = [{"row": _label(lbl), "value": _label(df.at[lbl, column])}
               for lbl in list(hit)[:limit]]
    return {
        "column": str(column),
        "n_flagged": int(len(hit)),
        "entries": entries,
        "truncated": bool(len(hit) > len(entries)),
    }
