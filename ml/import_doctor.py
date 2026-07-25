"""Import Doctor — diagnose the structural problems in real research files.

Researchers do not export clean rectangles. They export Excel sheets with a
title row above the header, SPSS/NHANES files where 999 means "refused",
columns typed as text because one cell says "<0.01", repeated-measures
columns named bp_1/bp_2/bp_3, and footers that say "Total".

The app cannot deterministically PARSE that variety, and pretending otherwise
is how tools silently corrupt someone's analysis. So this module offers a
different contract, the same one the modeling coach already honours:

    never silently guess — diagnose visibly, propose reversibly, record it.

Every check returns a ShapeFinding describing what was seen (with numbers),
why it matters, and a fix the user can apply with one click and undo. Nothing
here mutates a DataFrame on its own, and nothing is auto-applied: `diagnose`
is pure, and `apply_fix` returns a NEW frame plus a plain-language description
suitable for the methods section.

Confidence is explicit. A finding marked 'low' is a question, not an
instruction, and the UI must never pre-select it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Values that mean "missing" in survey/clinical exports but arrive as numbers.
# Only treated as sentinels when they also sit far outside the real
# distribution — 999 is a plausible triglyceride value but not a plausible age.
NUMERIC_SENTINELS: Tuple[float, ...] = (
    -9999.0, -999.0, -99.0, -88.0, -77.0, -9.0, -8.0, -7.0, -1.0,
    7.0, 8.0, 9.0, 66.0, 77.0, 88.0, 99.0,
    666.0, 777.0, 888.0, 999.0,
    6666.0, 7777.0, 8888.0, 9999.0, 99999.0,
)

# Single-digit codes (7=refused, 8=not asked, 9=don't know) are only credible
# in a CODED variable — a short integer scale. In a continuous measurement a 9
# is just a 9, and flagging it would be worse than missing it.
_CODED_MAX_DISTINCT = 15

# Text that unambiguously means "no value was recorded".
TEXT_MISSING_TOKENS: Tuple[str, ...] = (
    "", ".", "..", "-", "--", "?", "na", "n/a", "n.a.", "null",
    "#n/a", "nan", "not available",
)

# Text that USUALLY means missing but can be a legitimate answer: "none" is a
# real response to "which medications?", "unknown" is a real category in a
# registry. Recoding these to missing destroys data, so they are reported at
# low confidence and never pre-selected.
AMBIGUOUS_MISSING_TOKENS: Tuple[str, ...] = (
    "none", "unknown", "missing", "not applicable", "n/app", "refused",
    "not stated", "no answer",
)

# Characters stripped when testing whether a text column is really numeric.
_NUMERIC_NOISE = re.compile(r"[,\s%$£€]|^[<>≤≥~≈]+")
_KNOWN_UNITS = (
    "mg/dl", "mmol/l", "mg/l", "g/dl", "g/l", "ug/dl", "µg/dl", "ng/ml", "pg/ml",
    "mmhg", "kg", "g", "lb", "lbs", "cm", "m", "mm", "in", "kcal", "cal", "kj",
    "years", "yrs", "yr", "months", "mo", "days", "iu/l", "u/l", "%", "bpm",
    "kg/m2", "kg/m^2", "ml", "l", "sec", "s", "min", "hr", "hours",
)
_TRAILING_UNIT = re.compile(
    r"\s*(?:" + "|".join(re.escape(u) for u in sorted(_KNOWN_UNITS, key=len, reverse=True)) + r")\s*$",
    re.IGNORECASE,
)
# Repeated-measures suffixes: bp_1, bp.2, bpV3, bp_visit4, bp_t5
_REPEAT_SUFFIX = re.compile(r"^(?P<stem>.+?)[._\- ]?(?:v|visit|t|time|wave|round)?(?P<n>\d{1,2})$",
                            re.IGNORECASE)


@dataclass
class ShapeFinding:
    """One structural problem, with a proposed reversible fix."""
    id: str
    severity: str                    # 'critical' | 'warning' | 'info'
    title: str
    detail: str                      # what we saw, with real numbers
    why_it_matters: str
    fix_label: str                   # button text, imperative
    fix_kind: str                    # dispatch key for apply_fix
    confidence: str = "medium"       # 'high' | 'medium' | 'low'
    params: Dict[str, Any] = field(default_factory=dict)
    affected_columns: List[str] = field(default_factory=list)

    @property
    def auto_suggestable(self) -> bool:
        """Only high-confidence fixes may be pre-selected in the UI."""
        return self.confidence == "high"


# ── helpers ──────────────────────────────────────────────────────────────

def _is_unnamed(col: Any) -> bool:
    s = str(col)
    return s.startswith("Unnamed:") or s.strip() == "" or s.lower() == "nan"


_DECIMAL_COMMA = re.compile(r"^[+-]?\d{1,3}(?:\.\d{3})*,\d+$|^[+-]?\d+,\d{1,2}$")


def _units_present(s: pd.Series) -> set:
    """Distinct recognised unit suffixes appearing in a text column."""
    found = set()
    for v in s.dropna().astype(str).unique()[:500]:
        m = _TRAILING_UNIT.search(v.strip())
        if m:
            found.add(m.group(0).strip().lower())
    return found


def _looks_decimal_comma(s: pd.Series) -> bool:
    """European decimal notation: 1,5 means one-and-a-half, not fifteen."""
    vals = s.dropna().astype(str).str.strip()
    if vals.empty:
        return False
    hits = vals.map(lambda v: bool(_DECIMAL_COMMA.match(v)))
    return bool(hits.mean() >= 0.6)


def _clean_numeric_text(s: pd.Series) -> pd.Series:
    """Best-effort numeric coercion for text that is 'almost' numeric."""
    t = s.astype(str).str.strip()
    if _looks_decimal_comma(s):
        # 1.234,5 -> 1234.5 ; 1,5 -> 1.5. Stripping the comma would multiply
        # every value by ten or more.
        t = t.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    t = t.str.replace(_TRAILING_UNIT, "", regex=True)
    t = t.str.replace(_NUMERIC_NOISE, "", regex=True)
    t = t.str.replace(r"^\((.*)\)$", r"-\1", regex=True)   # (1.2) -> -1.2
    return pd.to_numeric(t, errors="coerce")


# ── individual checks ────────────────────────────────────────────────────

def check_unnamed_columns(df: pd.DataFrame) -> List[ShapeFinding]:
    unnamed = [c for c in df.columns if _is_unnamed(c)]
    if not unnamed:
        return []
    frac = len(unnamed) / max(1, df.shape[1])
    # A majority of unnamed columns almost always means the real header is
    # further down the sheet; that is a different (and better) fix.
    if frac >= 0.5:
        return []
    empty = [c for c in unnamed if df[c].notna().sum() == 0]
    return [ShapeFinding(
        id="unnamed_columns",
        severity="warning" if empty else "info",
        title=f"{len(unnamed)} column(s) have no name",
        detail=(f"Columns {', '.join(map(str, unnamed[:5]))}"
                f"{'…' if len(unnamed) > 5 else ''} are unnamed"
                + (f"; {len(empty)} of them are completely empty." if empty else ".")),
        why_it_matters=("Unnamed columns are usually spreadsheet artifacts — "
                        "spacer columns or notes — and they clutter every later step."),
        fix_label=f"Drop {len(empty) or len(unnamed)} unnamed column(s)",
        fix_kind="drop_columns",
        confidence="high" if empty else "low",
        params={"columns": empty or unnamed},
        affected_columns=list(map(str, empty or unnamed)),
    )]


def check_header_in_later_row(df: pd.DataFrame, max_scan: int = 10) -> List[ShapeFinding]:
    """Detect a title/metadata block sitting above the real header row."""
    if df.empty or df.shape[1] < 2:
        return []
    unnamed_frac = sum(_is_unnamed(c) for c in df.columns) / df.shape[1]
    if unnamed_frac < 0.5:
        return []

    scan = df.head(min(max_scan, len(df)))
    best_row, best_score = None, 0.0
    for i in range(len(scan)):
        row = scan.iloc[i]
        non_null = row.notna().sum()
        if non_null < df.shape[1] * 0.7:
            continue
        vals = [str(v).strip() for v in row.tolist() if pd.notna(v)]
        if not vals:
            continue
        uniq = len(set(vals)) / len(vals)
        stringy = sum(not _looks_numeric(v) for v in vals) / len(vals)
        score = uniq * stringy * (non_null / df.shape[1])
        if score > best_score:
            best_row, best_score = i, score

    if best_row is None or best_score < 0.5:
        return []
    preview = [str(v) for v in df.iloc[best_row].tolist()[:4]]
    return [ShapeFinding(
        id="header_in_later_row",
        severity="critical",
        title=f"The real column names look like they are in row {best_row + 1}",
        detail=(f"{unnamed_frac:.0%} of columns are unnamed, and row {best_row + 1} "
                f"reads like a header: {', '.join(preview)}…"),
        why_it_matters=("The file probably has a title or notes above the table. "
                        "Until the header is fixed, every column is misnamed and the "
                        "rows above it are fake data."),
        fix_label=f"Use row {best_row + 1} as the column names",
        fix_kind="promote_header",
        confidence="high",
        params={"row": int(best_row)},
    )]


def _looks_numeric(v: str) -> bool:
    try:
        float(str(v).replace(",", ""))
        return True
    except (TypeError, ValueError):
        return False


def check_duplicate_columns(df: pd.DataFrame) -> List[ShapeFinding]:
    dupes = pd.Index(df.columns)[pd.Index(df.columns).duplicated()].unique().tolist()
    if not dupes:
        return []
    return [ShapeFinding(
        id="duplicate_columns",
        severity="critical",
        title=f"{len(dupes)} column name(s) appear more than once",
        detail=f"Repeated names: {', '.join(map(str, dupes[:5]))}{'…' if len(dupes) > 5 else ''}.",
        why_it_matters=("Selecting a duplicated name returns several columns at once, "
                        "which breaks preprocessing and modelling in confusing ways."),
        fix_label="Make duplicate names unique",
        fix_kind="dedupe_columns",
        confidence="high",
        params={},
        affected_columns=list(map(str, dupes)),
    )]


def check_empty_rows_and_columns(df: pd.DataFrame) -> List[ShapeFinding]:
    out: List[ShapeFinding] = []
    empty_cols = [c for c in df.columns if df[c].notna().sum() == 0 and not _is_unnamed(c)]
    if empty_cols:
        out.append(ShapeFinding(
            id="empty_columns",
            severity="warning",
            title=f"{len(empty_cols)} column(s) are completely empty",
            detail=f"{', '.join(map(str, empty_cols[:5]))}{'…' if len(empty_cols) > 5 else ''} contain no values.",
            why_it_matters="Empty columns carry no information and clutter every later step.",
            fix_label=f"Drop {len(empty_cols)} empty column(s)",
            fix_kind="drop_columns",
            confidence="high",
            params={"columns": empty_cols},
            affected_columns=list(map(str, empty_cols)),
        ))
    n_empty_rows = int(df.isna().all(axis=1).sum())
    if n_empty_rows:
        out.append(ShapeFinding(
            id="empty_rows",
            severity="warning",
            title=f"{n_empty_rows} completely empty row(s)",
            detail=f"{n_empty_rows} of {len(df):,} rows contain no values at all.",
            why_it_matters="Blank rows inflate your sample size and distort missingness summaries.",
            fix_label=f"Drop {n_empty_rows} empty row(s)",
            fix_kind="drop_empty_rows",
            confidence="high",
            params={},
        ))
    return out


def check_footer_rows(df: pd.DataFrame, max_scan: int = 5) -> List[ShapeFinding]:
    """Trailing 'Total'/'Notes' rows that spreadsheets append below the data."""
    if len(df) < 5:
        return []
    tail = df.tail(min(max_scan, len(df)))
    suspects: List[int] = []
    for pos in range(len(tail)):
        row = tail.iloc[pos]
        filled = row.notna().sum()
        if filled == 0:
            continue
        text = " ".join(str(v).lower() for v in row.tolist() if pd.notna(v))
        looks_summary = any(w in text for w in ("total", "sum", "mean", "average", "note", "source:"))
        if looks_summary and filled <= max(1, df.shape[1] // 2):
            suspects.append(len(df) - len(tail) + pos)
    if not suspects:
        return []
    return [ShapeFinding(
        id="footer_rows",
        severity="warning",
        title=f"{len(suspects)} row(s) at the bottom look like a summary, not data",
        detail=(f"Row(s) {', '.join(str(i + 1) for i in suspects)} are mostly blank and "
                f"contain words like 'total' or 'note'."),
        why_it_matters=("A totals row treated as a participant silently corrupts every "
                        "mean, model, and figure."),
        fix_label=f"Drop {len(suspects)} footer row(s)",
        fix_kind="drop_rows",
        confidence="medium",
        params={"positions": suspects},
    )]


def check_numeric_sentinels(df: pd.DataFrame, min_count: int = 2) -> List[ShapeFinding]:
    """Codes like 999 / -9 / 7 that mean 'missing' but arrive as real numbers."""
    out: List[ShapeFinding] = []
    for col in df.select_dtypes(include=[np.number]).columns:
        col_s = df[col]
        if isinstance(col_s, pd.DataFrame):
            continue
        s = col_s.dropna()
        if len(s) < 10:
            continue

        # Single-digit codes are only credible in a CODED variable (a short
        # integer scale like 1=yes 2=no 7=refused). In a continuous measurement
        # a 9 is just a 9.
        try:
            is_integral = bool(np.all(np.equal(np.mod(s.to_numpy(dtype=float), 1), 0)))
        except Exception:
            is_integral = False
        coded = is_integral and s.nunique() <= _CODED_MAX_DISTINCT

        present = [v for v in NUMERIC_SENTINELS
                   if int((s == v).sum()) >= min_count
                   and (abs(v) >= 10 or coded)]
        if not present:
            continue

        # Every candidate must be excluded before judging the real spread, or
        # two sentinels mask each other (7 looks in-range while 9 is present).
        rest = s[~s.isin(present)]
        if rest.empty:
            continue
        lo, hi = float(rest.min()), float(rest.max())
        spread = max(abs(hi - lo), 1e-9)

        hits = {v: int((s == v).sum()) for v in present
                if v > hi + 0.5 * spread or v < lo - 0.5 * spread}
        if not hits:
            continue

        def _fmt(v: float) -> str:
            return str(int(v)) if float(v).is_integer() else str(v)

        vals = ", ".join(f"{_fmt(v)} ({n}x)" for v, n in sorted(hits.items()))
        out.append(ShapeFinding(
            id=f"sentinel_missing__{col}",
            severity="critical",
            title=f"'{col}' may use numeric codes for missing values",
            detail=(f"Found {vals} \u2014 far outside the rest of the column "
                    f"({lo:.4g} to {hi:.4g})."),
            why_it_matters=("Survey and clinical exports code missing answers as 999, "
                            "-9, or 7/8/9 in coded questions. Left as numbers they are "
                            "averaged into your results and quietly wreck every model."),
            fix_label=f"Treat {', '.join(_fmt(v) for v in sorted(hits))} as missing in '{col}'",
            fix_kind="recode_missing",
            confidence="medium",
            params={"column": col, "values": sorted(hits.keys())},
            affected_columns=[str(col)],
        ))
    return out


def check_text_missing_tokens(df: pd.DataFrame) -> List[ShapeFinding]:
    out: List[ShapeFinding] = []
    for col in df.select_dtypes(include=["object", "string"]).columns:
        s = df[col]
        if isinstance(s, pd.DataFrame):
            continue
        s = s.dropna().astype(str).str.strip().str.lower()
        if s.empty:
            continue
        clear = sorted({v for v in s.unique() if v in TEXT_MISSING_TOKENS})
        maybe = sorted({v for v in s.unique() if v in AMBIGUOUS_MISSING_TOKENS})

        if clear:
            n = int(s.isin(clear).sum())
            out.append(ShapeFinding(
                id=f"text_missing__{col}",
                severity="warning",
                title=f"'{col}' spells missing values as text",
                detail=f"{n:,} cell(s) contain {', '.join(repr(v) for v in clear[:5])}.",
                why_it_matters=("These read as ordinary categories, so the column looks "
                                "complete when it is not — and missingness never gets handled."),
                fix_label=f"Treat those as missing in '{col}'",
                fix_kind="recode_missing",
                confidence="high",
                params={"column": col, "values": clear, "text": True},
                affected_columns=[str(col)],
            ))
        if maybe:
            n = int(s.isin(maybe).sum())
            out.append(ShapeFinding(
                id=f"text_missing_ambiguous__{col}",
                severity="info",
                title=f"'{col}' may use words that mean either 'missing' or a real answer",
                detail=f"{n:,} cell(s) contain {', '.join(repr(v) for v in maybe[:5])}.",
                why_it_matters=(
                    "Only you know which it is: 'none' is a genuine answer to "
                    "'which medications?' but means missing in a lab column. Recoding "
                    "the wrong way silently deletes real data, so nothing is changed "
                    "unless you say so."
                ),
                fix_label=f"Treat those as missing in '{col}'",
                fix_kind="recode_missing",
                confidence="low",
                params={"column": col, "values": maybe, "text": True},
                affected_columns=[str(col)],
            ))
    return out


def check_numeric_stored_as_text(df: pd.DataFrame, min_parse: float = 0.8) -> List[ShapeFinding]:
    """Columns typed as text only because of commas, units, or '<0.01'."""
    out: List[ShapeFinding] = []
    for col in df.select_dtypes(include=["object", "string"]).columns:
        s = df[col].dropna()
        if len(s) < 5:
            continue
        raw_numeric = pd.to_numeric(s, errors="coerce").notna().mean()
        if raw_numeric >= 0.99:
            continue  # pandas would already have typed it numeric
        units = _units_present(s)
        if len(units) > 1:
            # mg/dL and mmol/L on one scale is not a formatting problem, it is
            # two different measurements. Coercing merges them into nonsense.
            out.append(ShapeFinding(
                id=f"mixed_units__{col}",
                severity="critical",
                title=f"'{col}' mixes different units",
                detail=f"Found {', '.join(sorted(units))} in the same column.",
                why_it_matters=("Converting these to plain numbers would put values "
                                "measured on different scales side by side, which no "
                                "model or statistic can interpret. They must be "
                                "converted to one unit first — a decision only you "
                                "can make."),
                fix_label="Cannot fix automatically — convert to one unit first",
                fix_kind="none",
                confidence="low",
                params={"column": col, "units": sorted(units)},
                affected_columns=[str(col)],
            ))
            continue

        parsed = _clean_numeric_text(s)
        rate = float(parsed.notna().mean())
        if rate < min_parse:
            continue
        n_blanked = int(parsed.isna().sum())
        offenders = s[parsed.isna()].astype(str).unique()[:3].tolist()
        examples = s[parsed.notna()].astype(str).unique()[:3].tolist()
        out.append(ShapeFinding(
            id=f"numeric_as_text__{col}",
            severity="warning",
            title=f"'{col}' looks numeric but is stored as text",
            detail=(f"{rate:.0%} of values parse as numbers after removing units, "
                    f"commas and comparison signs (e.g. {', '.join(map(repr, examples))})."
                    + (f" Non-numeric leftovers: {', '.join(map(repr, offenders))}." if len(offenders) else "")),
            why_it_matters=("As text this column cannot be modelled, correlated, or "
                            "plotted — it will silently sit out of your analysis."),
            fix_label=(f"Convert '{col}' to numbers"
                       + (f" (blanks {n_blanked} value(s) that cannot be read)" if n_blanked else "")),
            fix_kind="coerce_numeric",
            # Any conversion that discards values can destroy data, so it is
            # never pre-selected — the user must look at what would be lost.
            confidence="high" if n_blanked == 0 else "low",
            params={"column": col},
            affected_columns=[str(col)],
        ))
    return out


def check_categorical_variants(df: pd.DataFrame, max_levels: int = 50) -> List[ShapeFinding]:
    """'Male', 'male ', 'MALE' are one category typed three ways."""
    out: List[ShapeFinding] = []
    for col in df.select_dtypes(include=["object", "string"]).columns:
        s = df[col].dropna().astype(str)
        if s.empty or s.nunique() > max_levels:
            continue
        groups: Dict[str, set] = {}
        for v in s.unique():
            groups.setdefault(v.strip().lower(), set()).add(v)
        collisions = {k: v for k, v in groups.items() if len(v) > 1}
        if not collisions:
            continue
        sample = "; ".join(
            " / ".join(map(repr, sorted(v)[:3])) for v in list(collisions.values())[:3]
        )
        out.append(ShapeFinding(
            id=f"category_variants__{col}",
            severity="warning",
            title=f"'{col}' has categories that differ only by spacing or case",
            detail=f"{len(collisions)} group(s) collide, e.g. {sample}.",
            why_it_matters=("They are counted as separate groups, which splits your "
                            "sample and inflates the number of categories a model must learn."),
            fix_label=f"Merge those variants in '{col}'",
            fix_kind="normalize_categories",
            confidence="high",
            params={"column": col},
            affected_columns=[str(col)],
        ))
    return out


def check_constant_columns(df: pd.DataFrame) -> List[ShapeFinding]:
    const = [c for c in df.columns
             if df[c].notna().sum() > 0 and df[c].nunique(dropna=True) == 1]
    if not const:
        return []
    return [ShapeFinding(
        id="constant_columns",
        severity="info",
        title=f"{len(const)} column(s) have the same value in every row",
        detail=f"{', '.join(map(str, const[:5]))}{'…' if len(const) > 5 else ''}.",
        why_it_matters=("A constant carries no information for any model, though it may "
                        "still be worth keeping as a study-level label."),
        fix_label=f"Drop {len(const)} constant column(s)",
        fix_kind="drop_columns",
        confidence="low",
        params={"columns": const},
        affected_columns=list(map(str, const)),
    )]


def check_wide_repeated_measures(df: pd.DataFrame, min_members: int = 3) -> List[ShapeFinding]:
    """bp_1, bp_2, bp_3 … — one subject per row, many timepoints per subject."""
    stems: Dict[str, List[str]] = {}
    for col in df.columns:
        m = _REPEAT_SUFFIX.match(str(col))
        if m:
            stems.setdefault(m.group("stem"), []).append(str(col))
    families = {k: sorted(v) for k, v in stems.items() if len(v) >= min_members}
    if not families:
        return []
    shown = "; ".join(f"{k} ({len(v)}: {', '.join(v[:3])}…)" for k, v in list(families.items())[:3])
    return [ShapeFinding(
        id="wide_repeated_measures",
        severity="info",
        title=f"{len(families)} group(s) of columns look like repeated measures",
        detail=f"{shown}.",
        why_it_matters=("Wide repeated measures are fine for prediction, but if each row "
                        "is a subject measured several times you may want one row per "
                        "measurement instead — and clustered rows need a group-aware split."),
        fix_label="Reshape to one row per measurement (long format)",
        fix_kind="melt_repeated",
        confidence="low",
        params={"families": families},
        affected_columns=[c for v in families.values() for c in v],
    )]


ALL_CHECKS = (
    check_header_in_later_row,
    check_duplicate_columns,
    check_unnamed_columns,
    check_empty_rows_and_columns,
    check_footer_rows,
    check_numeric_sentinels,
    check_text_missing_tokens,
    check_numeric_stored_as_text,
    check_categorical_variants,
    check_constant_columns,
    check_wide_repeated_measures,
)

_SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}


def diagnose(df: pd.DataFrame) -> List[ShapeFinding]:
    """Run every structural check. Pure: never mutates `df`.

    A structural problem can mask others (a header stuck in row 3 makes every
    dtype wrong), so when the header itself is suspect we report only that —
    reporting twenty derived symptoms would bury the one fix that matters.
    """
    if df is None or df.empty:
        return []
    header = check_header_in_later_row(df)
    if header:
        return header

    findings: List[ShapeFinding] = []
    for check in ALL_CHECKS:
        if check is check_header_in_later_row:
            continue
        try:
            findings.extend(check(df))
        except Exception:
            # A malformed frame must never crash the upload page; a check that
            # cannot run simply reports nothing.
            continue
    findings.sort(key=lambda f: (_SEVERITY_ORDER.get(f.severity, 3), f.id))
    return findings


# ── fixes ────────────────────────────────────────────────────────────────

def apply_fix(df: pd.DataFrame, finding: ShapeFinding) -> Tuple[pd.DataFrame, str]:
    """Apply one finding's fix. Returns (new_frame, plain-language description).

    Never mutates the input. The description is written for a methods section.
    """
    kind = finding.fix_kind
    p = finding.params or {}
    out = df

    if kind == "promote_header":
        row = int(p["row"])
        new_cols = [str(v).strip() if pd.notna(v) else f"col_{i}"
                    for i, v in enumerate(df.iloc[row].tolist())]
        from utils.column_utils import make_unique_columns
        out = df.iloc[row + 1:].reset_index(drop=True)
        out.columns = make_unique_columns(new_cols)
        desc = f"Promoted row {row + 1} to column headers and dropped the {row + 1} row(s) above it."

    elif kind == "drop_columns":
        cols = [c for c in p.get("columns", []) if c in df.columns]
        out = df.drop(columns=cols)
        desc = f"Dropped {len(cols)} column(s): {', '.join(map(str, cols[:8]))}."

    elif kind == "dedupe_columns":
        from utils.column_utils import make_unique_columns
        out = df.copy()
        out.columns = make_unique_columns(list(df.columns))
        desc = "Renamed duplicate column names so each is unique."

    elif kind == "drop_empty_rows":
        before = len(df)
        out = df.dropna(how="all").reset_index(drop=True)
        desc = f"Dropped {before - len(out)} completely empty row(s)."

    elif kind == "drop_rows":
        positions = [i for i in p.get("positions", []) if 0 <= i < len(df)]
        keep = [i for i in range(len(df)) if i not in set(positions)]
        out = df.iloc[keep].reset_index(drop=True)
        desc = f"Dropped {len(positions)} non-data row(s) from the bottom of the file."

    elif kind == "recode_missing":
        col = p["column"]
        values = p.get("values", [])
        out = df.copy()
        if p.get("text"):
            norm = out[col].astype(str).str.strip().str.lower()
            out.loc[norm.isin(values), col] = np.nan
        else:
            out.loc[out[col].isin(values), col] = np.nan
        shown = ", ".join(str(int(v)) if isinstance(v, float) and float(v).is_integer() else str(v)
                          for v in values)
        desc = f"Recoded {shown} as missing in '{col}'."

    elif kind == "coerce_numeric":
        col = p["column"]
        out = df.copy()
        before = out[col].notna().sum()
        out[col] = _clean_numeric_text(out[col])
        lost = int(before - out[col].notna().sum())
        desc = (f"Converted '{col}' to numeric (removing units, separators and "
                f"comparison signs)"
                + (f"; {lost} value(s) could not be read and are now blank." if lost else "."))

    elif kind == "normalize_categories":
        col = p["column"]
        out = df.copy()
        mask = out[col].notna()
        out.loc[mask, col] = out.loc[mask, col].astype(str).str.strip()
        # Map every casing variant onto the most frequent spelling.
        counts = out.loc[mask, col].value_counts()
        canonical: Dict[str, str] = {}
        for value in counts.index:
            canonical.setdefault(str(value).lower(), str(value))
        out.loc[mask, col] = out.loc[mask, col].map(lambda v: canonical.get(str(v).lower(), v))
        desc = f"Merged spacing/capitalisation variants in '{col}'."

    elif kind == "melt_repeated":
        families = p.get("families", {})
        value_cols = [c for v in families.values() for c in v if c in df.columns]
        id_cols = [c for c in df.columns if c not in value_cols]
        var_name, val_name = "measurement", "value"
        while var_name in id_cols:
            var_name += "_"
        while val_name in id_cols or val_name == var_name:
            val_name += "_"
        out = df.melt(id_vars=id_cols, value_vars=value_cols,
                      var_name=var_name, value_name=val_name)
        desc = (f"Reshaped {len(value_cols)} repeated-measure column(s) into long format "
                f"(one row per measurement).")

    elif kind == "none":
        return df, "No automatic change is possible here; this needs a human decision."

    else:
        raise ValueError(f"Unknown fix kind: {kind!r}")

    return out, desc


def summarize(findings: List[ShapeFinding]) -> str:
    """One-line headline for the upload page."""
    if not findings:
        return "No structural problems detected."
    crit = sum(1 for f in findings if f.severity == "critical")
    warn = sum(1 for f in findings if f.severity == "warning")
    info = sum(1 for f in findings if f.severity == "info")
    bits = []
    if crit:
        bits.append(f"{crit} needing attention")
    if warn:
        bits.append(f"{warn} worth checking")
    if info:
        bits.append(f"{info} note{'s' if info != 1 else ''}")
    return "Found " + ", ".join(bits) + "."
