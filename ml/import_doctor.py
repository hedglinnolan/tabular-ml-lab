"""Import Doctor — diagnose the structural problems in real research files.

Researchers do not export clean rectangles. They export Excel sheets with a
title row above the header, SPSS/NHANES files where 999 means "refused",
columns typed as text because one cell says "<0.01", repeated-measures
columns named bp_1/bp_2/bp_3, and footers that say "Total".

The app cannot deterministically PARSE that variety, and pretending otherwise
is how tools silently corrupt someone's analysis. So this module offers a
different contract, the same one the modeling coach already honors:

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

def has_duplicate_labels(df: pd.DataFrame) -> bool:
    """True when two columns share a name.

    Real exports do this constantly — two 'bp' columns, three blank headers —
    and it breaks label indexing in a way that is easy to miss: df['bp']
    returns a two-column DataFrame, so `df['bp'].nunique() == 1` is a Series
    comparison and `if` on it raises "truth value of a Series is ambiguous".
    """
    try:
        return bool(pd.Index(df.columns).duplicated().any())
    except Exception:
        return False


def _each_column(df: pd.DataFrame):
    """Yield (label, Series) for every column, duplicates included.

    Iterates by POSITION, so a repeated label yields each of its columns
    separately as a real Series instead of collapsing them into a DataFrame.
    """
    for i in range(df.shape[1]):
        try:
            yield df.columns[i], df.iloc[:, i]
        except Exception:
            continue


def _each_column_matching(df: pd.DataFrame, selector) -> Any:
    """_each_column filtered by a predicate on the Series (dtype tests, etc.)."""
    for label, s in _each_column(df):
        try:
            if selector(s):
                yield label, s
        except Exception:
            continue


def _is_text(s: pd.Series) -> bool:
    return s.dtype == object or isinstance(s.dtype, pd.StringDtype) or \
        pd.api.types.is_string_dtype(s)


def _is_unnamed(col: Any) -> bool:
    s = str(col)
    return s.startswith("Unnamed:") or s.strip() == "" or s.lower() == "nan"


# European decimal notation. Two unambiguous shapes only:
#   1.234,5   dot-thousands AND a decimal comma — the dot group is REQUIRED (+),
#             because with * this branch also matched a plain "45,000" and
#             claimed every US thousands-separated number as a European decimal,
#             turning $45,000 into 45.0.
#   22,5      one or two digits after the comma, which no thousands separator
#             ever produces.
_DECIMAL_COMMA = re.compile(r"^[+-]?\d{1,3}(?:\.\d{3})+,\d+$|^[+-]?\d+,\d{1,2}$")

# Exactly three digits after a single comma. "5,555" is five-point-five-five-five
# in Bonn and five thousand five hundred fifty-five in Boston, and nothing in the
# value settles it. The module's contract forbids guessing silently, so this
# shape is read the common way (thousands) but reported as an assumption.
_AMBIGUOUS_COMMA = re.compile(r"^[+-]?\d{1,3},\d{3}$")


def _units_present(s: pd.Series) -> set:
    """Distinct recognized unit suffixes appearing in a text column."""
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


def comma_reading_is_ambiguous(s: pd.Series) -> bool:
    """True when the commas in this column could be either decimal or thousands.

    Only fires for the genuinely undecidable shape — a single comma followed by
    exactly three digits — and only when the column has no unambiguous values to
    settle it. A column holding both "1.234,5" and "5,555" is European
    throughout; one holding "22,5" is too.
    """
    vals = s.dropna().astype(str).str.strip()
    vals = vals[vals != ""]
    if vals.empty:
        return False
    if bool(vals.map(lambda v: bool(_DECIMAL_COMMA.match(v))).any()):
        return False              # something in the column settles it
    return bool(vals.map(lambda v: bool(_AMBIGUOUS_COMMA.match(v))).mean() >= 0.6)


_LEADING_ZERO = re.compile(r"^[+-]?0\d")
_PLAIN_INT = re.compile(r"^[+-]?\d+$")


def numeric_conversion_would_lose(s: pd.Series) -> Optional[str]:
    """Why turning this text column into numbers would destroy meaning, or None.

    Two cases, both everywhere in research exports and both invisible until a
    merge quietly fails:

    - Leading zeros are part of the value. Subject '007' is not subject 7, and
      a zip code '02139' is not 2139.
    - Integers beyond 2**53 cannot be held exactly in float64, so a long
      barcode or an NHS/MRN number silently changes digits.
    """
    vals = s.dropna().astype(str).str.strip()
    vals = vals[vals != ""]
    if vals.empty:
        return None
    if bool(vals.map(lambda v: bool(_LEADING_ZERO.match(v))).any()):
        return "leading zeros that are part of the identifier"
    ints = vals[vals.map(lambda v: bool(_PLAIN_INT.match(v)))]
    if not ints.empty:
        try:
            if bool(ints.map(lambda v: abs(int(v)) > 2 ** 53).any()):
                return "more digits than a number can store exactly"
        except Exception:
            return None
    return None


def reinfer_types(df: pd.DataFrame) -> pd.DataFrame:
    """Give a frame the dtypes a correct read would have produced.

    Promoting a header builds the frame out of rows pandas had already read as
    text, so every column arrives as object even when it holds nothing but
    numbers. Left alone, the file passes every check while every number in it
    sits out of the analysis — `age.mean()` raises and `age` gets treated as a
    category with sixty levels.

    Columns that would lose information by converting are left as text.
    """
    out = df.copy()
    for i in range(out.shape[1]):
        s = out.iloc[:, i]
        if not _is_text(s):
            continue
        nonnull = s.dropna()
        if nonnull.empty:
            continue
        parsed = pd.to_numeric(nonnull.astype(str).str.strip(), errors="coerce")
        if parsed.notna().mean() < 1.0:
            continue                      # not purely numeric; leave it alone
        if numeric_conversion_would_lose(s):
            continue                      # an identifier, not a measurement
        out.isetitem(i, pd.to_numeric(s.astype(str).str.strip(), errors="coerce"))
    return out


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
    empty = [c for c, s in _each_column(df) if _is_unnamed(c) and s.notna().sum() == 0]
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
                        "which breaks preprocessing and modeling in confusing ways."),
        fix_label="Make duplicate names unique",
        fix_kind="dedupe_columns",
        confidence="high",
        params={},
        affected_columns=list(map(str, dupes)),
    )]


def check_empty_rows_and_columns(df: pd.DataFrame) -> List[ShapeFinding]:
    out: List[ShapeFinding] = []
    empty_cols = [c for c, s in _each_column(df)
                  if s.notna().sum() == 0 and not _is_unnamed(c)]
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
    for col, col_s in _each_column_matching(df, pd.api.types.is_numeric_dtype):
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

        # Single-digit codes are only credible in a coded variable — EXCEPT
        # negative ones. -9 and -8 are standard survey missing codes (NHANES
        # uses them), and a negative value in a column of positive measurements
        # is never a real observation. Requiring abs(v) >= 10 meant a column of
        # 0-100 scores containing -9 sailed through and the -9s were averaged
        # into every result. Safe to relax now that a candidate must also lie
        # beyond every real value to be reported.
        present = [v for v in NUMERIC_SENTINELS
                   if int((s == v).sum()) >= min_count
                   and (abs(v) >= 10 or coded or v < 0)]
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

        # On a DENSE integer scale, a code is only a code if the data stops
        # before it. Excluding 7/8/9 as candidates before measuring the spread
        # manufactures a gap: in a genuine 1-9 Likert scale the 7s and 8s are
        # real answers, removing them leaves 1-6, and the 9 then looks like it
        # sits far outside a distribution it is actually part of.
        #
        # So for a dense scale the RAW values are split into runs of
        # consecutive integers. One unbroken run is a scale and nothing in it
        # is a code; where the data breaks, whichever run holds the most
        # OBSERVATIONS is the real scale and the rest are codes. That reads
        # 1-5 plus 7,8,9 correctly without mistaking 1-9 for it.
        #
        # Sparse data (lab values, ages) never forms runs, so it keeps the
        # distance test above, which is the right question there.
        observed = np.sort(s.unique().astype(float))
        span = float(observed[-1] - observed[0]) + 1.0
        dense = is_integral and len(observed) / max(span, 1.0) >= 0.5
        if dense:
            counts = s.value_counts()
            runs: List[List[float]] = [[float(observed[0])]]
            for v in observed[1:]:
                if float(v) - runs[-1][-1] <= 1.0:
                    runs[-1].append(float(v))
                else:
                    runs.append([float(v)])
            if len(runs) == 1:
                hits = {}
            else:
                main = set(max(runs, key=lambda r: sum(int(counts.get(v, 0)) for v in r)))
                # Judge every candidate by run membership rather than by the
                # distance test, which under-reports the near edge of a code
                # block: with a 1-5 scale plus 7,8,9, the 7 sits just inside the
                # distance threshold and would be left behind while 8 and 9 are
                # recoded — splitting one block of codes down the middle.
                hits = {v: int((s == v).sum()) for v in present if v not in main}
        # Whatever the tests above concluded, a missing-value code sits BEYOND
        # the observations, never among them. If real values exist on both
        # sides of a candidate, it is an observation — 77 in an age column that
        # also holds 78 and 80 is a 77-year-old, and 88 in a systolic column
        # reaching 174 is a blood pressure.
        #
        # Without this gate the run test misfired on ordinary clinical columns:
        # integer ages and blood pressures have natural gaps at n=300-500, the
        # run splits, the smaller run is called "codes", and a CRITICAL finding
        # appeared on a clean file offering a one-click button that recodes real
        # measurements to missing. It fired on 13 of 40 clean NHANES-shaped
        # files. The detail sentence it printed — "far outside the rest of the
        # column (18 to 80)" about the value 77 — was contradicted by the same
        # page's own numeric summary.
        real = s[~s.isin(list(hits))]
        if real.empty:
            continue
        real_lo, real_hi = float(real.min()), float(real.max())
        hits = {v: n for v, n in hits.items() if v > real_hi or v < real_lo}
        if not hits:
            continue
        # Describe the range the codes are actually outside of.
        lo, hi = real_lo, real_hi

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
    for col, s in _each_column_matching(df, _is_text):
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
    for col, s in _each_column_matching(df, _is_text):
        s = s.dropna()
        if len(s) < 5:
            continue
        raw_numeric = pd.to_numeric(s, errors="coerce").notna().mean()
        if raw_numeric >= 0.99:
            # Pure numeric text. A fresh read_csv WOULD have typed this numeric,
            # which is why this used to be skipped — but a frame built by
            # promoting a header, or loaded from JSON, or read from an Excel
            # sheet whose cells are text-formatted, arrives as object. Skipping
            # it there hands back a clean bill of health on a file where every
            # number sits out of the analysis.
            lossy = numeric_conversion_would_lose(s)
            if lossy:
                continue                  # an identifier: leaving it as text is correct
            examples = s.astype(str).unique()[:3].tolist()
            # The gate is >= 0.99, not == 1.0, so "every value parses" is false
            # for any rate in [0.99, 1.0) — and this branch never counted the
            # casualties, so a fix that blanks real values was labeled "high"
            # (the tier the UI pre-selects) with no count in the button. The
            # SAME situation at a 90% parse rate was correctly reported as low
            # confidence with the count shown: the safety logic was inverted
            # exactly where the data looks cleanest.
            n_blanked = int((~pd.to_numeric(s, errors="coerce").notna()).sum())
            if n_blanked:
                unreadable = (s[pd.to_numeric(s, errors="coerce").isna()]
                              .astype(str).unique()[:3].tolist())
                detail = (f"Almost every value is a plain number (e.g. "
                          f"{', '.join(map(repr, examples))}), but {n_blanked:,} "
                          f"cannot be read as one (e.g. "
                          f"{', '.join(map(repr, unreadable))}).")
                fix_label = (f"Convert '{col}' to numbers "
                             f"(blanks {n_blanked:,} value(s) that cannot be read)")
                confidence = "low"
            else:
                detail = (f"Every value is a plain number (e.g. "
                          f"{', '.join(map(repr, examples))}) but the column is typed "
                          f"as text.")
                fix_label = f"Convert '{col}' to numbers"
                confidence = "high"       # nothing is lost: every value parses
            out.append(ShapeFinding(
                id=f"numeric_as_text__{col}",
                severity="warning",
                title=f"'{col}' holds numbers but is stored as text",
                detail=detail,
                why_it_matters=("As text this column cannot be modeled, correlated "
                                "or plotted — it will silently sit out of your "
                                "analysis while appearing perfectly fine."),
                fix_label=fix_label,
                fix_kind="coerce_numeric",
                confidence=confidence,
                params={"column": col},
                affected_columns=[str(col)],
            ))
            continue
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

        # "5,555" is 5.555 in Bonn and 5555 in Boston. Reading it the common way
        # is fine; reading it the common way SILENTLY is not, because the wrong
        # guess is a 1000x rescale that nothing downstream will look odd about.
        ambiguous = comma_reading_is_ambiguous(s)
        assumption = ""
        if ambiguous:
            sample = next((v for v in examples if _AMBIGUOUS_COMMA.match(str(v))),
                          examples[0] if examples else "5,555")
            assumption = (f" The commas here could mark thousands or decimals — "
                          f"'{sample}' is read as "
                          f"{_clean_numeric_text(pd.Series([sample])).iloc[0]:,.0f} "
                          f"(thousands separator). If these are European decimals, "
                          f"convert them in your source file instead.")
        out.append(ShapeFinding(
            id=f"numeric_as_text__{col}",
            severity="warning",
            title=f"'{col}' looks numeric but is stored as text",
            detail=(f"{rate:.0%} of values parse as numbers after removing units, "
                    f"commas and comparison signs (e.g. {', '.join(map(repr, examples))})."
                    + (f" Non-numeric leftovers: {', '.join(map(repr, offenders))}." if len(offenders) else "")
                    + assumption),
            why_it_matters=("As text this column cannot be modeled, correlated, or "
                            "plotted — it will silently sit out of your analysis."),
            fix_label=(f"Convert '{col}' to numbers"
                       + (f" (blanks {n_blanked} value(s) that cannot be read)" if n_blanked else "")),
            fix_kind="coerce_numeric",
            # Any conversion that discards values can destroy data, so it is
            # never pre-selected — the user must look at what would be lost. An
            # undecidable comma is the same kind of risk: getting it wrong
            # rescales the column by 1000 and nothing downstream looks wrong.
            confidence="low" if (n_blanked or ambiguous) else "high",
            params={"column": col},
            affected_columns=[str(col)],
        ))
    return out


def check_categorical_variants(df: pd.DataFrame, max_levels: int = 50) -> List[ShapeFinding]:
    """'Male', 'male ', 'MALE' are one category typed three ways."""
    out: List[ShapeFinding] = []
    for col, s in _each_column_matching(df, _is_text):
        s = s.dropna().astype(str)
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
    const = [c for c, s in _each_column(df)
             if s.notna().sum() > 0 and s.nunique(dropna=True) == 1]
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

    # Duplicate labels are the same kind of masking problem as a misplaced
    # header. Every per-column fix names its target by label — "recode 999 in
    # bp" — and with two columns called 'bp' that instruction is ambiguous:
    # apply_fix would hit both, or neither. So the rename is reported on its
    # own, and the full per-column diagnosis runs once labels are unique.
    if has_duplicate_labels(df):
        out = list(check_duplicate_columns(df))
        for check in (check_empty_rows_and_columns, check_footer_rows):
            try:
                out.extend(check(df))          # frame-level: no label needed
            except Exception:
                continue
        out.sort(key=lambda f: (_SEVERITY_ORDER.get(f.severity, 3), f.id))
        return out

    findings: List[ShapeFinding] = []
    failed: List[str] = []
    for check in ALL_CHECKS:
        if check is check_header_in_later_row:
            continue
        try:
            findings.extend(check(df))
        except Exception:
            # A malformed frame must never crash the upload page. But silently
            # reporting nothing would show a clean bill of health on a file the
            # app could not actually inspect, so the gap is disclosed instead.
            failed.append(getattr(check, "__name__", "a check")
                          .replace("check_", "").replace("_", " "))
    if failed:
        findings.append(ShapeFinding(
            id="checks_failed",
            severity="warning",
            title=f"{len(failed)} structural check(s) could not run on this file",
            detail=f"Could not check: {', '.join(sorted(failed))}.",
            why_it_matters=("Something about this file's shape stopped part of the "
                            "review from running, so treat a clean result here as "
                            "incomplete rather than as a guarantee."),
            fix_label="",
            fix_kind="none",
            confidence="low",
        ))
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
        # Everything below the old header was read as text. Re-infer, so the
        # frame ends up as it would have been had the file been read correctly
        # in the first place — otherwise every numeric column stays a string.
        out = reinfer_types(out)
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
        desc = f"Merged spacing/capitalization variants in '{col}'."

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
