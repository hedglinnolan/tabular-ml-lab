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
import re
from dataclasses import dataclass, field
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

# How many ROW LABELS travel beside them (`DRIVE-007`). A card is not a data
# export and the row list is not one either — it is the thing you paste into a
# filter in the file you are about to fix, which is where the driver's first
# instinct went and where the app gave them nothing.
#
# Three orders of magnitude above `MAX_ENTRIES` because the two answer different
# questions: twelve entries are what a human reads, and the row list is what a
# spreadsheet consumes. The cap exists so the payload has a bound and is
# DECLARED alongside the list, because "every affected row" is a claim.
MAX_ROW_LIST = 20000

# ─────────────────────────────────────────────────────────────────────────────
# When to doubt the reading instead of the data
#
# The rule this implements, which is general and not about plausibility:
# **escalate on evidence that the interpretation is wrong, not on the magnitude
# of the consequence.** "Seventy-five rows would be deleted, that is a lot" is
# not a reason to hesitate; "seventy-five rows out of eighty are outside a hard
# physiologic floor, which is a property of the column and not of its entries"
# is. The first is squeamishness about the size of the harm and would make the
# tool hesitate exactly where it is most useful. The second is evidence.
#
# Three kinds of evidence, each one nameable in the verdict it produces:
#
#   * **derived name** — the column carries a modifier beside the reference key
#     (`hba1c_proxy`, `bp_sys_delta`, `weight_change`). `match_variable_key`
#     matches by substring, so a derived column inherits its parent's reference
#     intervals wholesale. Evidence the column is not the variable (T0-BUILD-003).
#   * **scale shift** — after unit conversion, the column still sits off the
#     reference by a factor the variable's own unit table knows about, or by a
#     power of ten. Evidence the unit inference picked wrong.
#   * **coherence** — the violations are one-sided and numerous. Entry errors
#     scatter; a systematic misreading does not.
#
# Only when none of these hold are the entries themselves the likely fault, and
# only then is a per-entry repair proposed.
# ─────────────────────────────────────────────────────────────────────────────

# Share above which the violation is a property of the column rather than of its
# entries, even with no other evidence about which misreading it is.
WHOLE_COLUMN_SUSPECT_SHARE = 0.25

# Share above which one-sidedness stops being coincidence.
COHERENCE_MIN_SHARE = 0.10
ONE_SIDED_SHARE = 0.95

# Share of the column a candidate unit must land inside the reference interval
# before it counts as having rescued it.
RESCUE_MIN_SHARE = 0.80

# Name segments that mark a column as derived from a variable rather than being
# it. Deliberately a closed list: guessing that an unfamiliar suffix means
# "derived" would suppress real findings on columns that are simply named oddly.
DERIVED_MARKERS = frozenset({
    "proxy", "flag", "delta", "change", "diff", "pct", "percent", "percentile",
    "score", "index", "ratio", "rate", "bin", "bins", "quartile", "quintile",
    "tertile", "decile", "quantile", "group", "grp", "cat", "category", "class",
    "band", "level", "z", "zscore", "log", "sqrt", "norm", "scaled", "std",
    "imputed", "pred", "predicted", "residual", "resid", "rank", "lag", "lead",
    "baseline", "followup", "prev", "next", "avg", "mean", "median", "max",
    "min", "sum", "count", "n", "missing", "isna", "any", "ever",
})

READING_ENTRIES = "entries"
READING_UNITS = "units"
READING_IDENTITY = "identity"
READING_UNCLEAR = "unclear"


def name_segments(column: str) -> List[str]:
    """A column name split into its words, lowercased."""
    return [s for s in re.split(r"[^0-9a-z]+", str(column).lower()) if s]


def derived_from(column: str, var_key: str) -> Optional[str]:
    """The modifier marking `column` as derived from `var_key`, or None.

    `match_variable_key` matches by substring, so `hba1c_proxy` inherits HbA1c's
    reference intervals and its impossibility band. This is the cheap, closed
    check for "the name says this is not that variable".
    """
    key_parts = name_segments(var_key)
    segments = name_segments(column)
    if not key_parts or not segments:
        return None
    # The key must actually be present as whole segments; otherwise this is a
    # different question (a raw substring match) that the caller already made.
    extras = [s for s in segments if s not in key_parts]
    for segment in extras:
        if segment in DERIVED_MARKERS:
            return segment
    return None


def known_scale_factors(var_key: str) -> List[float]:
    """Conversion factors the variable's own unit table knows about.

    A residual ratio matching one of these is evidence that unit inference
    picked the wrong hypothesis — not that the values are impossible.
    """
    factors = {10.0, 100.0, 1000.0, 0.1, 0.01, 0.001}
    try:
        from ml.clinical_units import CLINICAL_VARIABLES
        for unit_name, factor, _range in CLINICAL_VARIABLES.get(var_key, {}).get(
                "hypotheses", []):
            f = _finite(factor)
            if f and f > 0 and abs(f - 1.0) > 1e-9:
                factors.add(f)
                factors.add(1.0 / f)
    except Exception:
        pass
    return sorted(factors)


@dataclass
class InterpretationVerdict:
    """Whether to doubt the reading of a column instead of its entries.

    `reading` is one of `entries` / `units` / `identity` / `unclear`, and only
    `entries` licenses a per-entry repair proposal. `evidence` lists the signals
    that fired, by name, so the verdict is arguable rather than a threshold
    nobody can see.
    """
    reading: str
    evidence: List[str] = field(default_factory=list)
    statement: Optional[str] = None
    factor: Optional[float] = None

    @property
    def suspect(self) -> bool:
        """True when the reading is doubted and no repair may be proposed."""
        return self.reading != READING_ENTRIES


def rescues_the_column(values, reference, factors) -> Optional[float]:
    """The factor that would bring this column into the reference interval.

    The direct question, rather than a ratio of summary statistics: *does
    multiplying by a unit the variable actually has put the values where the
    reference says they belong?* If one does, the unit inference picked wrong
    and the values were never out of range at all.

    Returns the factor, or None when no known unit rescues the column.
    """
    if reference is None:
        return None
    ref_low, ref_high = float(reference[0]), float(reference[1])
    inside_now = float(((values >= ref_low) & (values <= ref_high)).mean())
    if inside_now >= RESCUE_MIN_SHARE:
        return None                      # already where it belongs
    for factor in factors:
        scaled = values * factor
        inside = float(((scaled >= ref_low) & (scaled <= ref_high)).mean())
        if inside >= RESCUE_MIN_SHARE:
            return factor
    return None


def interpretation_verdict(column: str, var_key: str, values, low: float,
                           high: float, flagged,
                           reference: Optional[Sequence[float]] = None) -> InterpretationVerdict:
    """The named predicate: is the reading wrong, or are the entries?

    `values` and `flagged` are the column and its violations, both already
    expressed in the reference unit. `low`/`high` are the bounds they broke;
    `reference` is the variable's reference interval, which is where the values
    would sit if the unit were right.
    """
    n_present = int(len(values))
    n_flagged = int(len(flagged))
    if not n_present or not n_flagged:
        return InterpretationVerdict(READING_ENTRIES)

    share = n_flagged / n_present
    evidence: List[str] = []

    marker = derived_from(column, var_key)
    if marker:
        evidence.append(f"derived-name:{marker}")

    below = int((flagged < low).sum())
    above = int((flagged > high).sum())
    one_sided = max(below, above) >= ONE_SIDED_SHARE * n_flagged
    if one_sided and share >= COHERENCE_MIN_SHARE:
        evidence.append("one-sided:" + ("below" if below >= above else "above"))

    factor = rescues_the_column(values, reference, known_scale_factors(var_key))
    if factor is not None:
        evidence.append(f"rescued-by:{factor:g}x")

    if share > WHOLE_COLUMN_SUSPECT_SHARE:
        evidence.append(f"share:{share:.0%}")

    # The ruling, in order of how directly each signal speaks to the reading.
    # A factor that puts the whole column inside its reference interval is the
    # strongest evidence available: it does not merely suggest the reading is
    # wrong, it names the right one.
    if factor is not None:
        reading = READING_UNITS
    elif marker:
        reading = READING_IDENTITY
    elif share > WHOLE_COLUMN_SUSPECT_SHARE:
        reading = READING_UNCLEAR
    else:
        return InterpretationVerdict(READING_ENTRIES)

    return InterpretationVerdict(
        reading=reading, evidence=evidence, factor=factor,
        statement=_reading_statement(column, var_key, reading, share, factor,
                                     marker, low, high))


def _reading_statement(column, var_key, reading, share, factor, marker,
                       low, high) -> str:
    """What the card says instead of flagging entries.

    The correction is about the column, in the user's terms. It never names a
    row, because no row is the problem.
    """
    if reading == READING_UNITS:
        return (f"`{column}` is probably not in the units we assumed. Multiplying "
                f"it by {factor:g} — a unit `{var_key}` is recorded in — puts the "
                f"column inside its reference range, so these values are not out "
                f"of range; they are in a different unit. Nothing here is "
                f"proposed as an error. Fix the unit and the check runs again.")
    if reading == READING_IDENTITY:
        return (f"`{column}` is probably not `{var_key}`. The reference range "
                f"({low:g}–{high:g}) was matched to this column by name, and the "
                f"name says it is a *{marker}* of `{var_key}` rather than the "
                f"measurement itself. The values below are shown as recorded; "
                f"nothing here is proposed as an error.")
    return (f"`{column}` is probably not in the units we assumed, or is not "
            f"`{var_key}` at all. {share:.0%} of it falls outside the range no "
            f"living person can be outside, which is a property of the column "
            f"rather than of its entries. Check the unit and the variable before "
            f"treating any of these as errors.")


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

        interval = get_reference_interval(ref, var_key)
        band = get_impossibility_band(ref, var_key)
        if band is not None:
            floor, ceiling, unit = band
            hit = converted[(converted < floor) | (converted > ceiling)]
            if len(hit):
                impossible.append(_entry_block(
                    col, var_key, present, converted, hit, floor, ceiling, unit,
                    tier="impossible",
                    # The interval, not the band, is where a correctly-united
                    # column belongs — so it is what the rescue check aims at.
                    reference=interval[:2] if interval else None))

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
                 low, high, unit, tier,
                 reference: Optional[Sequence[float]] = None) -> Dict[str, Any]:
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
    # The reading is doubted only on the tier that would propose a repair. An
    # advisory tier makes no claim strong enough to need it.
    verdict = (interpretation_verdict(column, var_key, converted, low, high, hit,
                                      reference=reference)
               if tier == "impossible"
               else InterpretationVerdict(READING_ENTRIES))
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
        # EVERY affected row label, not only the handful the card shows
        # (`DRIVE-007`). The card displays 12 of 125 because 125 rows is not a
        # table anybody reads — but the driver's first instinct on seeing an
        # impossible value is to go back to the CSV and fix it at source, and
        # the app made that hard by being the only thing that knew which rows.
        #
        # Capped, and the cap is DECLARED rather than silent, because *"every
        # affected row"* is a claim: a list that quietly stopped at 20,000 would
        # be the card asserting a completeness it does not have. In practice the
        # cap is unreachable on the impossible tier — a column with tens of
        # thousands of impossible entries is a mis-united column, and
        # `whole_column_suspect` replaces the entry list with the reading
        # correction before this is rendered at all.
        "all_rows": [_label(label) for label in list(hit.index)[:MAX_ROW_LIST]],
        "all_rows_truncated": bool(len(hit) > MAX_ROW_LIST),
        # The verdict travels with the block rather than the block being
        # dropped: staying silent about the values would be the other way of
        # asserting something false.
        "reading": verdict.reading,
        "reading_evidence": list(verdict.evidence),
        "reading_statement": verdict.statement,
        "scale_factor": verdict.factor,
        # Kept as the name the frontend and the earlier tests already use.
        "whole_column_suspect": verdict.suspect,
        "suspect_reason": verdict.statement,
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
