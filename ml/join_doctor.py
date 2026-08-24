"""Join Doctor — make merging two research files safe and legible.

Merging is where non-programmers lose their data quietly. The failures we
actually reproduced against this app:

1. Both files call the key SEQN, but one stores "001" and the other 1.
   pandas raises "You are trying to merge on str and int64 columns … you
   should use pd.concat", which is meaningless to a nutrition researcher.
2. The files call the key SEQN and seqn (or patient_id). Name-only matching
   finds nothing and the user is told "no matching columns" — a dead end,
   though the join is trivial.
3. One file has several rows per subject. The join silently multiplies the
   cohort and every later "n =" in the manuscript is wrong.
4. An inner join keeps half the cohort and nothing says so.
5. The IDs differ only by stray whitespace or capitalization.

None of that is discoverable by looking at column names, which is all the
previous helpers did. This module looks at the VALUES: it proposes keys by
measuring how well they actually overlap, explains what a join will do in
plain language BEFORE it runs, and offers reversible normalizations for the
three mechanical mismatches (type, whitespace, case).

Nothing here mutates the caller's frames, and nothing is applied
automatically — same contract as the Import Doctor.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Upper bound on DISTINCT key values canonicalized per column (guards pathological
# wide/long files without ever comparing two different random subsets).
_MAX_DISTINCT = 200_000
# A key must identify rows, not group them: at least this share of values must
# be distinct on one side, or "sex" would look like a perfect key.
_MIN_UNIQUENESS = 0.5
# Below this overlap a pairing is not worth proposing.
_MIN_COVERAGE = 0.05


# Text that means "no identifier". These must NEVER match each other: two
# subjects whose ID is "unknown" are not the same subject.
_KEY_MISSING_TOKENS = {"", "nan", "none", "null", "na", "n/a", "n.a.", ".", "-",
                       "--", "?", "missing", "unknown", "not available"}

# The subset of that list a real coding scheme also uses: 'NA' is a study centre
# in North America, '-' is a control specimen, '?' is a category. Prose blanks
# ('unknown', 'null', 'not available') are never a code and stay out of this
# set — admitting them would fuse every unknown subject into one person, which
# is the opposite failure and the worse one.
_CODED_MISSING_TOKENS = frozenset({"na", ".", "-", "--", "?"})

# float64 carries 53 bits of mantissa. Above this, consecutive whole numbers are
# no longer distinguishable and two IDs have already become one. Narrower floats
# stop counting far sooner — float32 at 2**24, which an ordinary MRN clears —
# so the limit is read from the value's own storage, not assumed.
_EXACT_INT_LIMIT = 2 ** 53

# Hidden column carrying the user's ORIGINAL key values through the merge,
# so matching can use the canonical form without the output inheriting it.
_ORIGINAL_KEY = "__original_key__"

_INT_RE = re.compile(r"^[+-]?\d+$")
_DECIMAL_RE = re.compile(r"^[+-]?\d+\.\d*$")
# A value written as a plain number, however it is stored.
_NUMERIC_TEXT_RE = re.compile(r"^[+-]?\d+(\.\d*)?$")


def _canon_scalar(v: Any, keep_tokens: frozenset = frozenset()) -> Optional[str]:
    """Canonical token for one key value, or None when it identifies nobody.

    Integers are canonicalized with exact arbitrary-precision arithmetic, NOT
    through float: passing IDs through float64 silently collides values above
    2^53 (9007199254740993 and ...992 become the same subject), which is a
    false merge — the worst outcome this module can produce.

    `keep_tokens` names spellings that LOOK like blanks but are this column's
    own codes; the decision belongs to the column, not to this scalar, so it is
    made once by _coded_key_tokens and handed down.
    """
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    if v is pd.NaT:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(v, bool):
        return "true" if v else "false"

    s = str(v).strip()
    low = s.lower()
    if low in _KEY_MISSING_TOKENS and low not in keep_tokens:
        return None

    # "007" and 7 are the same subject; 9007199254740993 keeps every digit.
    if _INT_RE.match(s):
        return str(int(s))
    # "3.0" is the integer 3; "3.50" and "3.5" are the same value.
    if _DECIMAL_RE.match(s):
        whole, _, frac = s.partition(".")
        frac = frac.rstrip("0")
        sign = "-" if whole.startswith("-") else ""
        digits = whole.lstrip("+-").lstrip("0") or "0"
        base = f"{sign}{int(digits)}"
        return base if not frac else f"{base}.{frac}"
    return s


def _value_shape(s: str) -> str:
    """Character-class signature of a value: 'A01' -> 'A99', 'NA' -> 'AA'."""
    return re.sub(r"[0-9]", "9", re.sub(r"[A-Za-z]", "A", str(s)))


def _key_vocabulary(s: pd.Series) -> Optional[pd.Series]:
    """This column's distinct key spellings, stripped — or None if it has none.

    Distinct values only: every question asked of them is about the column's
    vocabulary, and a full pass costs as much as the canonicalization these
    decisions are meant to precede.
    """
    try:
        if s is None or isinstance(s, pd.DataFrame):
            return None
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            return None
        raw = s.dropna().drop_duplicates().astype(str).str.strip()
        return None if raw.empty else raw
    except Exception:
        return None


def _written_as_numbers(raw: pd.Series) -> bool:
    """Are this column's real values all written as plain numbers?

    A column of subject numbers has no coding scheme for a blank-looking token
    to belong to, so 'NA' there is R's blank and must stay one however often it
    appears. This is the guard that keeps the evidence rules below from fusing
    every unreadable subject number into one participant.
    """
    others = [v for v in raw.unique() if v.lower() not in _KEY_MISSING_TOKENS]
    return bool(others) and all(_NUMERIC_TEXT_RE.match(v) for v in others)


def _coded_key_tokens(s: pd.Series) -> frozenset:
    """Blank-looking spellings this column is actually using as CODES.

    'NA' is R's blank in a column of subject numbers and a study centre in a
    column of centre codes. Deleting the second deletes a whole stratum, and
    the app then reports the row as having had no ID — a false reason for a
    value it refused to read. So ask the column: a token that matches the SHAPE
    of one of the column's other values belongs to their coding scheme; one
    that matches nothing is a blank. A numeric column has no coding scheme to
    belong to.

    Requiring HALF the column to share the token's shape was the wrong bar, and
    it failed exactly where real coding schemes live: 'NA' among EU, APAC and
    LATAM matches one code in three, and a two-centre study (NA, NB) was
    refused outright for having "nothing to be consistent with". One other
    value of the same shape is a coding scheme; the pair rule below carries the
    cases where even that is missing.
    """
    raw = _key_vocabulary(s)
    if raw is None:
        return frozenset()
    try:
        low = raw.str.lower()
        present = {t for t in _CODED_MISSING_TOKENS if bool((low == t).any())}
        if not present or _written_as_numbers(raw):
            return frozenset()
        shapes = [_value_shape(v) for v in raw.unique()
                  if v.lower() not in _KEY_MISSING_TOKENS]
        keep = {t for t in present
                if _value_shape(raw[low == t].iloc[0]) in shapes}
        return frozenset(keep)
    except Exception:
        return frozenset()


def _paired_key_tokens(left: pd.Series, right: pd.Series) -> frozenset:
    """Blank-looking spellings BOTH files use in the columns being joined.

    Evidence from the pair beats a fixed name list. A value that appears in
    both files' key columns is doing an identifier's job whatever it spells —
    two files that both carry centre 'NA' are describing the same centre, not
    independently failing to record one — and deleting it drops a whole stratum
    and then reports those rows as having had no ID at all.

    Prose blanks are still never admitted (they are not in the candidate set):
    fusing every subject whose ID reads 'unknown' into one participant is the
    opposite failure and the worse one. Neither is a column written as numbers.
    """
    lraw, rraw = _key_vocabulary(left), _key_vocabulary(right)
    if lraw is None or rraw is None:
        return frozenset()
    try:
        if _written_as_numbers(lraw) or _written_as_numbers(rraw):
            return frozenset()
        llow, rlow = set(lraw.str.lower()), set(rraw.str.lower())
        return frozenset(_CODED_MISSING_TOKENS & llow & rlow)
    except Exception:
        return frozenset()


def _fold_is_safe(tokens: pd.Series) -> bool:
    """True when lower-casing these canonical tokens merges no distinct IDs."""
    text = tokens.dropna().astype(str)
    return bool(len(text) == 0 or text.str.lower().nunique() == text.nunique())


@dataclass(frozen=True)
class KeyReading:
    """How BOTH sides of one join read their key column.

    Every decision in here was previously taken per column, by whichever call
    happened to be normalizing that side. One stray-case duplicate on one side
    then disabled folding for that side ALONE, so the two files compared in two
    different canonical spaces: rows vanish, and a surviving row is paired with
    the wrong partner — a wrong number with nothing on screen to catch it.
    Decide once for the pair, and hand the same decision to both sides.
    """
    fold_case: bool
    keep_tokens: frozenset = frozenset()


def key_reading(left: pd.Series, right: pd.Series) -> KeyReading:
    """The one canonical space both sides of a join must be read into."""
    keep = (_coded_key_tokens(left) | _coded_key_tokens(right)
            | _paired_key_tokens(left, right))
    fold = all(
        _fold_is_safe(s.dropna().drop_duplicates().map(lambda v: _canon_scalar(v, keep)))
        for s in (left, right))
    return KeyReading(fold_case=fold, keep_tokens=keep)


def _exact_int_limit(kind: Any) -> int:
    """Where this float storage stops telling consecutive whole numbers apart.

    2**(mantissa bits + 1): 2**53 for float64, 2**24 for float32, 2**11 for
    float16. Accepts a dtype, a pandas extension dtype or a scalar type.
    """
    try:
        return int(2 ** (np.finfo(getattr(kind, "numpy_dtype", kind)).nmant + 1))
    except Exception:
        return _EXACT_INT_LIMIT


def _float_value_limit(v: Any) -> Optional[int]:
    """The limit this ONE value has already breached, or None if it is safe."""
    if isinstance(v, bool) or not isinstance(v, (float, np.floating)):
        return None
    try:
        if not np.isfinite(v):
            return None
        limit = _exact_int_limit(type(v))
        return limit if abs(float(v)) >= limit else None
    except (TypeError, ValueError):
        return None


def numeric_key_precision_loss(s: pd.Series) -> Optional[Tuple[int, float, int]]:
    """(rows at risk, largest magnitude, the limit passed) for an untrustworthy
    float key, else None.

    _canon_scalar defends against float conversion THIS module performs; it
    cannot undo one already done. An ID column that arrived as a float — which
    happens as soon as one row's ID is blank in any loader — has already lost
    its low digits above its storage's exact-integer limit, and canonicalizing
    the collapsed digits presents two participants as one row identity. The
    digits are gone, so the only honest move is to refuse the column and say
    why, never to canonicalize a collapsed value as though it were exact.

    The question is about VALUES, not about the container. Gating on float64
    and assuming 2^53 let two live shapes through: a float32 key column, which
    a parquet upload preserves and which collapses at 2^24, and an object
    column holding python floats, which is what stacking a text ID onto a
    numeric one produces. Both were then narrated as ordinary fan-out.
    """
    try:
        if s is None or isinstance(s, pd.DataFrame):
            return None
        # >=, not >. At exactly the limit the spacing between representable
        # values is already 2, so a value sitting ON it may itself be a
        # collapsed limit+1 — which is the very pair the docstring names.
        if pd.api.types.is_float_dtype(s):
            limit = _exact_int_limit(s.dtype)
            v = pd.to_numeric(s, errors="coerce").dropna()
            v = v[np.abs(v) >= limit]
            if v.empty:
                return None
            return int(len(v)), float(np.abs(v).max()), limit
        if not pd.api.types.is_object_dtype(s):
            return None
        # An object column is mixed by definition: only the floats in it have
        # lost anything, and each carries the limit of its own storage. Asked
        # of the DISTINCT values first — the scan is per-value python, and a
        # column of a million IDs holds far fewer spellings than rows.
        vals = s.dropna()
        at_risk = [(v, lim) for v, lim in
                   ((v, _float_value_limit(v)) for v in pd.unique(vals)) if lim]
        if not at_risk:
            return None
        n = int(vals.isin([v for v, _ in at_risk]).sum())
        return (n, max(abs(float(v)) for v, _ in at_risk),
                min(lim for _, lim in at_risk))
    except Exception:
        return None


def _precision_refusal(col: Any, file_name: str, n_at_risk: int, biggest: float,
                       limit: int = _EXACT_INT_LIMIT) -> str:
    return (
        f"'{col}' in {file_name} is stored as a decimal number, and {n_at_risk:,} of its "
        f"ID(s) reach {limit:,} or beyond — the point where that storage can no longer "
        f"tell consecutive whole numbers apart (the largest is about {biggest:,.0f}). Those "
        f"IDs have already lost their last digits, so matching on them would merge "
        f"different people into one participant. Re-import this file with the ID column "
        f"read as text, then join again."
    )


def normalize_key(s: pd.Series, fold_case: Optional[bool] = None,
                  keep_tokens: Optional[frozenset] = None) -> pd.Series:
    """Canonical comparison form for a key column.

    Missing values become NaN and never match anything — two rows with no ID
    are not the same subject, and pandas' merge (unlike SQL) would otherwise
    pair every blank with every blank.

    Case folding is applied only when it does not itself merge distinct IDs;
    if a column genuinely contains both "abc" and "ABC" they are kept apart.
    Pass fold_case and keep_tokens explicitly — as key_reading computes them —
    to read two columns into ONE canonical space; deciding either per column
    puts the two sides of a join in spaces that do not compare.
    """
    if s is None:
        return pd.Series(dtype="object")
    keep = _coded_key_tokens(s) if keep_tokens is None else keep_tokens
    out = s.map(lambda v: _canon_scalar(v, keep))

    if fold_case is None:
        fold_case = _fold_is_safe(out)
    if fold_case:
        out = out.map(lambda v: v.lower() if isinstance(v, str) else v)
    return out.astype("object")


def _case_collision_example(s: pd.Series,
                            keep_tokens: frozenset = frozenset()) -> Optional[Tuple[str, str]]:
    """Two values in this column that differ only in case, if any exist."""
    try:
        tokens = s.map(lambda v: _canon_scalar(v, keep_tokens)).dropna().astype(str)
        seen: Dict[str, str] = {}
        for t in tokens.unique():
            low = t.lower()
            if low in seen and seen[low] != t:
                return seen[low], t
            seen.setdefault(low, t)
    except Exception:
        pass
    return None


def unreadable_key_spellings(s: pd.Series,
                             keep_tokens: frozenset = frozenset()) -> List[str]:
    """Non-blank values this reading discarded, spelled as the user wrote them.

    The disclosure that names them has to be TRUE: "this row had no ID" is a
    false account of a row whose ID the app refused to read.
    """
    try:
        raw = s.dropna().astype(str).str.strip()
        return sorted({v for v in raw.unique()
                       if v and _canon_scalar(v, keep_tokens) is None})
    except Exception:
        return []


def _name_similarity(a: str, b: str) -> float:
    ca, cb = re.sub(r"[^a-z0-9]", "", str(a).lower()), re.sub(r"[^a-z0-9]", "", str(b).lower())
    if not ca or not cb:
        return 0.0
    if ca == cb:
        return 1.0
    return SequenceMatcher(None, ca, cb).ratio()


def _base_is_numeric(s: pd.Series) -> bool:
    """Is the key numeric UNDERNEATH its container?

    pandas reports a Categorical of integers as non-numeric, which made a
    perfectly good category-vs-number join look like text-vs-numbers and get
    blocked with an explanation that was simply untrue.
    """
    try:
        if isinstance(s.dtype, pd.CategoricalDtype):
            return bool(pd.api.types.is_numeric_dtype(s.cat.categories))
    except Exception:
        pass
    return bool(pd.api.types.is_numeric_dtype(s))


# Generic counter names carry no identity: two unrelated exports both called
# "Unnamed: 0" overlap 100% by construction. A named identifier is different —
# a study numbering its participants 1..N, or this app's own execute_stack
# producing SEQN 1..200 from two stacked cycles, is a REAL key that merely
# happens to be contiguous.
# "id" itself belongs here. Almost every CSV export carries one, so two
# unrelated files both holding id 1..50 is a coincidence, not evidence — the
# exemption below needs a SPECIFIC name (SEQN, subject_id, USUBJID), not the
# bare word. This line is the difference between rescuing a real study's
# participant numbers and merging a survey file into a GDP file.
_ROW_COUNTER_NAMES = frozenset({
    "", "id", "ids", "key", "index", "idx", "row", "rows", "rowid", "row_id",
    "rownum", "row_number", "rownumber", "n", "no", "num", "number", "seq",
    "level_0",
})


def _is_generic_counter_name(col: Any) -> bool:
    name = str(col).strip().lower()
    return (name in _ROW_COUNTER_NAMES
            or name.startswith("unnamed")
            or name.replace(" ", "").replace("_", "") in _ROW_COUNTER_NAMES)


def _named_identifier_pair(left_col: Any, right_col: Any) -> bool:
    """Both sides carry the SAME specific identifier name."""
    from utils.combine import _looks_like_an_id_name
    if _is_generic_counter_name(left_col) or _is_generic_counter_name(right_col):
        return False
    if not (_looks_like_an_id_name(left_col) and _looks_like_an_id_name(right_col)):
        return False
    norm = lambda c: str(c).strip().lower().replace("_", "").replace(" ", "")
    return norm(left_col) == norm(right_col)


def _looks_like_row_index(raw: pd.Series) -> bool:
    """True for 0..N-1 / 1..N style counters.

    These are the great false friend of value-based key matching: any two
    files that happen to carry a row counter overlap 100% by coincidence, so
    'age' in one file will 'match' 'gdp' in an unrelated one. Such columns are
    only credible as keys when the column NAMES also agree.
    """
    if not pd.api.types.is_numeric_dtype(raw):
        return False
    v = pd.to_numeric(raw, errors="coerce").dropna()
    if len(v) < 5 or not np.all(np.equal(np.mod(v, 1), 0)):
        return False
    u = np.sort(v.unique())
    if len(u) < 0.9 * len(v):
        return False
    return bool(u[0] in (0, 1) and np.array_equal(u, np.arange(u[0], u[0] + len(u))))


@dataclass
class KeyCandidate:
    """One possible way to link two files, scored on real value overlap."""
    left_col: str
    right_col: str
    coverage_left: float       # share of left keys found in right
    coverage_right: float
    n_matched: int             # distinct key values present in both
    left_unique: int
    right_unique: int
    left_rows: int
    right_rows: int
    dtype_mismatch: bool = False
    needs_normalization: bool = False   # whitespace/case only
    left_has_duplicates: bool = False
    right_has_duplicates: bool = False
    name_similarity: float = 0.0
    index_like: bool = False    # EITHER side is a plain row counter
    # True when the key space had to be sampled to stay tractable. The counts
    # are then ESTIMATES, and the app must not assert an estimate as fact.
    sampled: bool = False

    @property
    def repeats_on_both_sides(self) -> bool:
        """True when the column has duplicate values in BOTH files.

        An identifier is allowed to repeat on one side — one subject with many
        visits is an ordinary 1:many link. But a column that repeats on BOTH
        sides cannot identify anybody: every left copy pairs with every right
        copy, which is a Cartesian product, not a link.

        'age' matching 'age' across two survey cycles is the classic trap. It
        presents as a 77%-coverage, identically-named key and is a measurement.
        """
        return self.left_has_duplicates and self.right_has_duplicates

    @property
    def distinctness(self) -> float:
        """Share of rows carrying a distinct value, on the better side.

        A real identifier is near 1.0 on at least ONE side. The other side is
        free to repeat — that is just a 1:many link, one subject with several
        visits. Taking the worse side here would reject every repeated-measures
        design, which is most of longitudinal nutrition research.
        """
        return max(
            self.left_unique / max(1, self.left_rows),
            self.right_unique / max(1, self.right_rows),
        )

    @property
    def score(self) -> float:
        """Rank by how well the key actually links the two files.

        Name agreement carries real weight: two files that genuinely describe
        the same subjects almost always name the identifier similarly, and
        without that signal a coincidental overlap of row counters outranks
        the true key.
        """
        overlap = (self.coverage_left + self.coverage_right) / 2
        uniq = max(
            self.left_unique / max(1, self.left_rows),
            self.right_unique / max(1, self.right_rows),
        )
        s = overlap * 0.45 + uniq * 0.10 + self.name_similarity * 0.45
        if self.index_like:
            # _looks_like_row_index's own docstring says such columns "are only
            # credible as keys when the column NAMES also agree" — that
            # exemption was documented and never implemented. Without it a
            # study numbering participants 1..N had its correct SEQN<->SEQN key
            # scored 0.15/"low", dropped from the dropdown by combine_ui, and a
            # measurement column pre-selected in its place.
            s *= 0.6 if _named_identifier_pair(self.left_col, self.right_col) else 0.15
        if self.repeats_on_both_sides:
            s *= 0.25   # a measurement that happens to overlap
        return s

    @property
    def confidence(self) -> str:
        """How safely a UI may present this — never auto-apply 'low'.

        'high' is the tier the UI is allowed to pre-select, so 'high' means the
        app is ASSERTING this is the right key. Two things must therefore never
        reach it, however good their overlap looks:

        - A plain row counter (0..N-1 / 1..N). Matching names do not rescue it:
          two unrelated exports both called 'Unnamed: 0' or 'row' overlap 100%
          by construction, and the app cannot tell that from two files that
          genuinely list the same people in the same order.
        - A column that repeats on both sides, which is a measurement.
        """
        if self.index_like:
            # Offered, never asserted: the app still cannot PROVE that two
            # contiguous runs describe the same people, so "medium" (visible,
            # user-confirmed) is the ceiling even when the names agree.
            return ("medium"
                    if _named_identifier_pair(self.left_col, self.right_col)
                    else "low")
        if self.repeats_on_both_sides:
            return "low"
        if self.name_similarity >= 0.85 and min(self.coverage_left, self.coverage_right) >= 0.5:
            tier = "high"
        elif max(self.coverage_left, self.coverage_right) >= 0.8:
            tier = "high" if self.name_similarity >= 0.6 else "medium"
        else:
            tier = "medium" if max(self.coverage_left, self.coverage_right) >= 0.5 else "low"
        if self.sampled and tier == "high":
            # An estimate must never reach the tier the UI pre-selects. This is
            # a CEILING and has to be applied last: checked first it became a
            # floor, promoting every junk pair on a file above _MAX_DISTINCT
            # distinct values straight from 'low' to 'medium', where combine_ui
            # offers it and pre-selects the top-scoring one.
            return "medium"
        return tier

    @property
    def is_clean(self) -> bool:
        return not (self.dtype_mismatch or self.needs_normalization)

    def headline(self, left_name: str = "the first file",
                 right_name: str = "the second file") -> str:
        """One sentence a non-programmer can act on."""
        if self.dtype_mismatch:
            return (f"'{self.left_col}' and '{self.right_col}' look like the same ID, but one "
                    f"file stores it as text and the other as numbers — so nothing matches "
                    f"until that is fixed ({self.n_matched:,} would match after fixing).")
        if self.needs_normalization:
            return (f"'{self.left_col}' and '{self.right_col}' match after ignoring "
                    f"capitalization and stray spaces ({self.n_matched:,} IDs).")
        if self.sampled:
            return (f"'{self.left_col}' and '{self.right_col}' share about "
                    f"{self.n_matched:,} IDs — an estimate, because these files "
                    f"hold more than {_MAX_DISTINCT:,} distinct IDs and a "
                    f"representative slice of them was compared "
                    f"({self.coverage_left:.0%} of {left_name}, "
                    f"{self.coverage_right:.0%} of {right_name}).")
        return (f"'{self.left_col}' and '{self.right_col}' share {self.n_matched:,} IDs "
                f"({self.coverage_left:.0%} of {left_name}, {self.coverage_right:.0%} of {right_name}).")


@dataclass
class JoinDiagnosis:
    """What a specific join will actually do, in plain language."""
    left_key: str
    right_key: str
    how: str
    predicted_rows: int
    left_rows: int
    right_rows: int
    matched_keys: int
    unmatched_left: int
    unmatched_right: int
    row_multiplication: bool = False
    dtype_mismatch: bool = False
    needs_normalization: bool = False
    column_collisions: List[str] = field(default_factory=list)
    blocking: List[str] = field(default_factory=list)   # must fix before joining
    warnings: List[str] = field(default_factory=list)   # should understand first
    notes: List[str] = field(default_factory=list)

    @property
    def can_proceed(self) -> bool:
        return not self.blocking


def _key_tokens(df: pd.DataFrame, col: str) -> Optional[pd.Series]:
    """Canonical tokens for a column's DISTINCT values, or None if it cannot
    be a key.

    Deliberately NOT a row sample. Sampling each file independently compares
    two different random subsets, so on files above the sample size the
    measured overlap collapses toward zero and the true key stops being
    proposed exactly when the data is large enough to matter. Instead we cheaply
    reject columns that cannot identify rows, then canonicalize only the
    distinct values of the survivors.
    """
    try:
        s = df[col]
        if isinstance(s, pd.DataFrame):      # duplicated column label
            return None
        n = len(s)
        if n == 0:
            return None
        try:
            n_unique = int(s.nunique(dropna=True))
        except TypeError:
            return None                       # unhashable cells
        if n_unique < 2:
            return None                       # constant: identifies nothing
        # NOTE: deliberately no per-column uniqueness floor here. A long-format
        # file repeats its subject ID on every visit — three visits per person
        # puts uniqueness at 0.33 — and rejecting that column outright made
        # every repeated-measures design unjoinable, which is most of
        # longitudinal nutrition research. The real rule ("at least ONE side
        # must identify subjects") needs both files, so it lives in
        # find_key_candidates, where the pair is known.
        uniques = s.dropna().drop_duplicates()
        if len(uniques) > _MAX_DISTINCT:
            # `.iloc[:_MAX_DISTINCT]` was a POSITIONAL head-truncation, applied
            # independently to each side — precisely what the docstring above
            # promises this function never does. Two files listing the same
            # subjects in different row orders had their overlap measured
            # between two disjoint slices, so coverage collapsed toward zero
            # and at ~2x the cap find_key_candidates returned nothing at all.
            #
            # Keep a VALUE-determined subset instead: a stable hash of the
            # value decides membership, so both sides retain the same region of
            # the key space and the measured overlap is an unbiased estimate of
            # the real one. Callers are told it is an estimate via
            # _tokens_were_sampled below.
            # Hash the CANONICAL token, not the raw value. pd.util.hash_array
            # buckets 1 (int64), 1.0 (float64) and '1' (object) differently, so
            # hashing raw values gave two files holding the identical ID set
            # disjoint regions of the key space whenever their dtypes differed
            # — the very "two different random subsets" failure this branch
            # exists to avoid, and cross-dtype matching is a supported case
            # here (dtype_mismatch has its own headline).
            tokens = normalize_key(uniques).dropna()
            keep = _keep_fraction(len(uniques))
            digest = pd.util.hash_array(tokens.to_numpy())
            return tokens[digest < _hash_ceiling(keep)]
        return normalize_key(uniques).dropna()
    except Exception:
        return None


def _keep_fraction(n_distinct: int) -> float:
    """Share of the key space a column of this size retains. 1.0 when unsampled."""
    return 1.0 if n_distinct <= _MAX_DISTINCT else _MAX_DISTINCT / float(n_distinct)


def _hash_ceiling(keep: float) -> np.uint64:
    return np.uint64(keep * float(np.iinfo(np.uint64).max))


def _full_distinct(df: pd.DataFrame, col: Any, fallback: int) -> int:
    """Distinct values in the WHOLE column, however few tokens were sampled."""
    try:
        s = df[col]
        if isinstance(s, pd.DataFrame):
            return fallback
        return int(s.nunique(dropna=True))
    except Exception:
        return fallback


def _tokens_were_sampled(df: pd.DataFrame, col: str) -> bool:
    """True when _key_tokens had to sample, so its counts are estimates."""
    try:
        s = df[col]
        if isinstance(s, pd.DataFrame):
            return False
        return int(s.nunique(dropna=True)) > _MAX_DISTINCT
    except Exception:
        return False


# Column names researchers give the thing that identifies a participant.
_ID_NAME_HINT = re.compile(
    r"(^|[^a-z])(id|seqn|subject|participant|patient|pid|mrn|record|sample|"
    r"specimen|barcode|accession|usubjid|respondent|person|case)([^a-z]|$)",
    re.IGNORECASE)


def _columns_worth_testing(df: pd.DataFrame, other: pd.DataFrame,
                           limit: int) -> List[Any]:
    """Which columns to test as keys, when there are more than we can afford.

    This used to be `list(df.columns)[:limit]`, which is fine for a clinical
    export and useless for -omics: a transcriptomics matrix is samples by
    twenty thousand genes, and if the sample ID sits after a block of
    annotation columns it was never inspected at all. The join then failed on
    a study whose key was sitting right there.

    Ordering by how likely a column is to BE an identifier costs nothing and
    fixes it. A column sharing a name with one in the other file comes first —
    that is the cheapest and strongest signal there is — then names that look
    like identifiers, then non-float columns, and only then position.
    """
    cols = list(df.columns)
    if len(cols) <= limit:
        return cols

    shared = {str(c) for c in other.columns}

    def rank(col: Any) -> tuple:
        name = str(col)
        in_both = name in shared
        id_like = bool(_ID_NAME_HINT.search(name))
        try:
            # A float column is almost never an identifier; measurements are
            # floats and IDs are text or whole numbers.
            floaty = pd.api.types.is_float_dtype(df[col])
        except Exception:
            floaty = False
        return (not in_both, not id_like, floaty)

    return sorted(cols, key=rank)[:limit]


def find_key_candidates(left: pd.DataFrame, right: pd.DataFrame,
                        max_columns: int = 60,
                        min_coverage: float = _MIN_COVERAGE) -> List[KeyCandidate]:
    """Propose ways to link two files, ranked by real value overlap.

    Column NAMES are only a tiebreak — a key named SEQN in one file and
    patient_id in the other is found because the values line up.
    """
    if left is None or right is None or left.empty or right.empty:
        return []

    lcols = _columns_worth_testing(left, right, max_columns)
    rcols = _columns_worth_testing(right, left, max_columns)

    # Precompute normalized value sets once per column.
    lnorm: Dict[str, pd.Series] = {}
    rnorm: Dict[str, pd.Series] = {}
    for c in lcols:
        t = _key_tokens(left, c)
        if t is not None and not t.empty:
            lnorm[c] = t
    for c in rcols:
        t = _key_tokens(right, c)
        if t is not None and not t.empty:
            rnorm[c] = t

    out: List[KeyCandidate] = []
    for lc, ls in lnorm.items():
        if ls.empty:
            continue
        lset = set(ls.unique())
        if not lset:
            continue
        for rc, rs in rnorm.items():
            if rs.empty:
                continue
            rset = set(rs.unique())
            if not rset:
                continue
            matched = lset & rset
            if not matched:
                continue
            sampled_pair = (_tokens_were_sampled(left, lc)
                            or _tokens_were_sampled(right, rc))
            # Both sides keep the tokens whose hash falls below their own
            # threshold, so a shared ID survives into the measured intersection
            # only if it clears the STRICTER of the two. Rescaling by the left
            # column's ratio alone understated the true key by the size ratio
            # of the two files — up to 20x — and coverage was never rescaled at
            # all, so the min_coverage gate below dropped the real key before it
            # could be scored whenever the files differed in size.
            _n_l = _full_distinct(left, lc, len(lset))
            _n_r = _full_distinct(right, rc, len(rset))
            _keep = min(_keep_fraction(_n_l), _keep_fraction(_n_r))
            n_matched_est = (len(matched) if not sampled_pair
                             else int(round(len(matched) / max(_keep, 1e-12))))
            cov_l = min(1.0, n_matched_est / max(_n_l, 1))
            cov_r = min(1.0, n_matched_est / max(_n_r, 1))
            if max(cov_l, cov_r) < min_coverage:
                continue
            lraw, rraw = left[lc], right[rc]
            if isinstance(lraw, pd.DataFrame) or isinstance(rraw, pd.DataFrame):
                continue                      # duplicated column label
            dtype_mismatch = (
                pd.api.types.is_numeric_dtype(lraw) != pd.api.types.is_numeric_dtype(rraw)
            )
            # Whitespace/case only matters when both sides are text.
            needs_norm = False
            if not dtype_mismatch and not pd.api.types.is_numeric_dtype(lraw):
                try:
                    raw_match = (set(lraw.dropna().astype(str).unique())
                                 & set(rraw.dropna().astype(str).unique()))
                    needs_norm = len(raw_match) < len(matched)
                except Exception:
                    needs_norm = False

            candidate = KeyCandidate(
                left_col=str(lc), right_col=str(rc),
                coverage_left=cov_l, coverage_right=cov_r,
                # Already scaled back to the real key space above. Reporting the
                # sampled count would tell a researcher their two 2,000-person
                # files share 521 IDs.
                n_matched=n_matched_est,
                # Uniqueness must come from the FULL column, not the token
                # set. Above _MAX_DISTINCT the tokens are a value-determined
                # sample, so len(lset)/len(left) reads as 0.25 on a perfectly
                # unique key and the "at least one side identifies subjects"
                # test rejected it — the true key stopped being proposed
                # exactly when the file was large enough to matter.
                left_unique=_n_l,
                right_unique=_n_r,
                left_rows=max(len(left), len(lset)), right_rows=max(len(right), len(rset)),
                dtype_mismatch=dtype_mismatch,
                needs_normalization=needs_norm,
                left_has_duplicates=bool(_n_l < len(left)),
                right_has_duplicates=bool(_n_r < len(right)),
                name_similarity=_name_similarity(lc, rc),
                # OR, not AND. A 1..N counter overlaps anything 1..N by
                # construction, so ONE counter is enough to make the overlap
                # meaningless. Under the conjunction only the honest
                # counter-to-counter pairing was penalized, and a measurement
                # paired against a counter ('age' <-> 'row') escaped both the
                # penalty and the confidence downgrade to outrank it at
                # medium — the penalty made the ranking worse than none.
                index_like=_looks_like_row_index(lraw) or _looks_like_row_index(rraw),
                sampled=sampled_pair,
            )
            # At least ONE side must actually identify subjects. This is what
            # separates a real key from a shared category: 'sex' matches 'sex'
            # perfectly in every pair of files and identifies nobody, while a
            # subject ID is near-unique on the side that has one row per person
            # even when the other side repeats it once per visit.
            if candidate.distinctness < _MIN_UNIQUENESS:
                continue
            out.append(candidate)

    out.sort(key=lambda c: c.score, reverse=True)
    return out


def resolve_column(df: pd.DataFrame, key: Any) -> Any:
    """Return the real column label that `key` names.

    find_key_candidates reports a column by its PRINTED form, because that is
    what a person reads in a dropdown. For a frame with tuple labels — a
    parquet file that round-tripped MultiIndex columns, or an Excel sheet read
    with a two-row header — the printed form is "('key', 'SEQN')" and looking
    it up raises KeyError even though the column is sitting right there.
    """
    try:
        if key in df.columns:
            return key
    except Exception:
        pass
    target = str(key)
    for c in df.columns:
        if str(c) == target:
            return c
    raise ValueError(
        f"The column '{key}' is not in this file. Its columns are: "
        f"{', '.join(map(str, list(df.columns)[:8]))}"
        + ("…" if df.shape[1] > 8 else ""))


def detect_nested_ids(left: pd.Series, right: pd.Series) -> Optional[str]:
    """One file's IDs are the START of the other's — say so, in plain language.

    Two shapes dominate the -omics data researchers bring:

      TCGA-02-0001              clinical, one row per patient
      TCGA-02-0001-01A-21R-...  the assay, one row per aliquot
      ENSG00000141510           a gene
      ENSG00000141510.16        the same gene, with an annotation version

    Neither pair shares a single exact value, so the honest verdict is "nothing
    to join on" — true, and completely unhelpful, because the relationship is
    obvious to a human. This does not perform the join; it names the
    relationship and says what to change, because deriving a key silently is
    exactly the guess this module refuses to make.

    Examples quote the values as the USER wrote them. Matching happens on the
    case-folded form, and echoing that back shows a TCGA researcher
    'tcga-02-0001' for an ID their file spells in capitals.
    """
    def _pairs(s: pd.Series):
        raw = s.dropna().astype(str)
        norm = normalize_key(s).dropna().astype(str)
        both = pd.DataFrame({"norm": norm, "raw": raw.reindex(norm.index)})
        both = both.drop_duplicates(subset="norm").head(2000)
        return dict(zip(both["norm"], both["raw"]))

    lmap, rmap = _pairs(left), _pairs(right)
    if len(lmap) < 5 or len(rmap) < 5:
        return None
    if set(lmap) & set(rmap):
        return None                      # they already match; nothing to explain

    if len(next(iter(lmap))) <= len(next(iter(rmap))):
        short, long_, short_side, long_side = lmap, rmap, "first", "second"
    else:
        short, long_, short_side, long_side = rmap, lmap, "second", "first"
    sset = set(short)

    # A version or sample suffix after the last separator.
    for sep in (".", "_", "-"):
        trimmed = {v.rsplit(sep, 1)[0]: v for v in long_ if sep in v}
        hit = set(trimmed) & sset
        if trimmed and len(hit) >= 0.8 * min(len(sset), len(trimmed)):
            example_key = next(iter(hit))
            return (f"The IDs in the {long_side} file carry an extra piece after "
                    f"the last '{sep}' — for example "
                    f"'{long_[trimmed[example_key]]}' against "
                    f"'{short[example_key]}'. They are the same identifier with a "
                    f"version or sample suffix on one side. Remove everything "
                    f"after the last '{sep}' in your source file and they match.")

    # A fixed-length prefix, as with a TCGA patient barcode.
    widths = {len(v) for v in list(sset)[:200]}
    if len(widths) == 1:
        width = widths.pop()
        if width >= 6:
            heads = {v[:width]: v for v in long_}
            hit = set(heads) & sset
            if len(hit) >= 0.8 * min(len(sset), len(heads)):
                example_key = next(iter(hit))
                return (f"Every ID in the {long_side} file begins with an ID from "
                        f"the {short_side} file — the first {width} characters "
                        f"match, for example '{long_[heads[example_key]]}' starts "
                        f"with '{short[example_key]}'. One file identifies people "
                        f"and the other identifies samples taken from them. Add a "
                        f"column holding just the first {width} characters and "
                        f"join on that.")
    return None


def diagnose_join(left: pd.DataFrame, right: pd.DataFrame,
                  left_key: str, right_key: str, how: str = "inner",
                  left_name: str = "the first file",
                  right_name: str = "the second file") -> JoinDiagnosis:
    """Explain what this join will do BEFORE it runs.

    Predicts the row count (including fan-out from duplicate keys), names
    every mechanical blocker, and says out loud how much of each cohort is
    about to be dropped.
    """
    left_key, right_key = resolve_column(left, left_key), resolve_column(right, right_key)
    ls, rs = left[left_key], right[right_key]
    if isinstance(ls, pd.DataFrame) or isinstance(rs, pd.DataFrame):
        raise ValueError(
            f"The column '{left_key}' appears more than once in one of these files. "
            f"Rename or remove the duplicate before joining."
        )
    # ONE canonical space for both sides — see KeyReading. Normalizing each side
    # in its own call is what silently mis-pairs rows.
    reading = key_reading(ls, rs)
    ln = normalize_key(ls, reading.fold_case, reading.keep_tokens)
    rn = normalize_key(rs, reading.fold_case, reading.keep_tokens)
    n_missing_left, n_missing_right = int(ln.isna().sum()), int(rn.isna().sum())
    lvalid, rvalid = ln.dropna(), rn.dropna()
    lset, rset = set(lvalid.unique()), set(rvalid.unique())
    matched = lset & rset

    # A type problem is one the user CANNOT SEE: the values print identically
    # and still refuse to match, as with "001" against 1. Judging it from the
    # pandas dtypes instead was wrong in both directions — a Categorical of
    # integers is not "numeric", so a working category-vs-number join was
    # blocked with the false claim that one file stores the key as text; and
    # when two columns share no values at all the type message fired ahead of
    # the real problem and told the user that fixing the types would "match 0
    # IDs", which is advice to do something pointless.
    # Compare the UNDERLYING type, not the container. A Categorical of integers
    # is not "numeric" by pandas' test, so a working category-vs-number join was
    # blocked with the false claim that one file stores the key as text.
    dtype_mismatch = bool(matched) and _base_is_numeric(ls) != _base_is_numeric(rs)
    raw_matched = len(set(ls.dropna().astype(str).unique())
                      & set(rs.dropna().astype(str).unique()))
    needs_norm = bool(matched) and raw_matched < len(matched) and not dtype_mismatch

    # Predict rows, accounting for fan-out when a key repeats.
    lcounts = lvalid.value_counts()
    rcounts = rvalid.value_counts()
    matched_rows = int(sum(lcounts.get(k, 0) * rcounts.get(k, 0) for k in matched))
    unmatched_left_rows = int(sum(lcounts.get(k, 0) for k in (lset - matched)))
    unmatched_right_rows = int(sum(rcounts.get(k, 0) for k in (rset - matched)))

    # Rows whose ID is blank match nobody, but a left/right/outer join still
    # keeps them on the side being preserved.
    if how == "inner":
        predicted = matched_rows
    elif how == "left":
        predicted = matched_rows + unmatched_left_rows + n_missing_left
    elif how == "right":
        predicted = matched_rows + unmatched_right_rows + n_missing_right
    else:  # outer
        predicted = (matched_rows + unmatched_left_rows + unmatched_right_rows
                     + n_missing_left + n_missing_right)

    # Columns the app itself adds while combining are its own bookkeeping;
    # warning the user about a clash in one of them reads as a problem with
    # their data and invites them to "decide" about a column they never made.
    collisions = [str(c) for c in (set(left.columns) & set(right.columns))
                  if str(c) not in {str(left_key), str(right_key)}
                  and not str(c).startswith("__source_file")]

    # Fan-out is about SUBJECTS being duplicated, not about total row counts:
    # 3 subjects joined to 6 visits yields 6 rows, which is not larger than
    # either input yet every subject now appears several times. Detect it from
    # repeated keys among the matched set.
    dup_left = any(lcounts.get(k, 0) > 1 for k in matched)
    dup_right = any(rcounts.get(k, 0) > 1 for k in matched)

    d = JoinDiagnosis(
        left_key=str(left_key), right_key=str(right_key), how=how,
        predicted_rows=predicted, left_rows=len(left), right_rows=len(right),
        matched_keys=len(matched),
        unmatched_left=len(lset - matched), unmatched_right=len(rset - matched),
        row_multiplication=bool(dup_left or dup_right),
        dtype_mismatch=dtype_mismatch, needs_normalization=needs_norm,
        column_collisions=sorted(collisions),
    )

    # --- blocking problems -------------------------------------------------
    # Checked first: a key whose digits have already collapsed makes every
    # count below an answer to the wrong question.
    for _s, _fname, _col in ((ls, left_name, left_key), (rs, right_name, right_key)):
        _loss = numeric_key_precision_loss(_s)
        if _loss:
            d.blocking.append(_precision_refusal(_col, _fname, *_loss))
    if d.blocking:
        # Return with the refusal and NOTHING else. Every count below was
        # measured on keys that have already collided, so the fan-out warning
        # would explain the collision as repeated visits — a confident and
        # false account of corrupted row identity.
        return d
    if dtype_mismatch:
        d.blocking.append(
            f"'{left_key}' is stored as {'numbers' if _base_is_numeric(ls) else 'text'} "
            f"in {left_name} but as {'numbers' if _base_is_numeric(rs) else 'text'} in "
            f"{right_name}. They look identical on screen but will not match. "
            f"Fixing this matches {len(matched):,} IDs."
        )
    elif not matched:
        # "Check you picked the right columns" is the wrong advice when the
        # columns ARE right and something invisible is stopping the match. Two
        # date columns that look identical on screen but carry different
        # timezones is the common one, and the generic message sends the user
        # hunting for a mistake they did not make.
        why = ""
        l_dt = pd.api.types.is_datetime64_any_dtype(ls)
        r_dt = pd.api.types.is_datetime64_any_dtype(rs)
        if l_dt and r_dt:
            l_tz = getattr(getattr(ls, "dt", None), "tz", None)
            r_tz = getattr(getattr(rs, "dt", None), "tz", None)
            if (l_tz is None) != (r_tz is None):
                with_tz = left_name if l_tz is not None else right_name
                without = right_name if l_tz is not None else left_name
                why = (f" These are both dates, but the ones in {with_tz} carry a "
                       f"timezone and the ones in {without} do not, so they never "
                       f"count as equal even where they show the same day. Export "
                       f"both without a timezone, or as plain YYYY-MM-DD text.")
            elif str(l_tz) != str(r_tz):
                why = (f" These are both dates, but recorded in different timezones "
                       f"({l_tz} and {r_tz}), so the same instant is written two ways.")
            else:
                why = (" These are both dates. If one file stores a time of day and "
                       "the other stores midnight, no two values will be equal — "
                       "compare on the date alone.")
        if not why:
            nested = detect_nested_ids(ls, rs)
            if nested:
                why = " " + nested
        d.blocking.append(
            f"None of the values in '{left_key}' appear in '{right_key}', so there is nothing "
            f"to join on.{why or ' Check you picked the right columns.'}"
        )

    # A join that multiplies into millions of rows will not finish on the
    # laptop this app runs on — it exhausts memory and the browser tab simply
    # stops responding, which reads as "the app is broken" rather than "that
    # join was wrong". Refuse it, and say what to do instead.
    _BLOWUP_ROWS = 5_000_000
    if predicted > _BLOWUP_ROWS and predicted > 20 * max(len(left), len(right), 1):
        d.blocking.append(
            f"This join would produce about {predicted:,} rows from {len(left):,} "
            f"and {len(right):,} — every matching row on one side is paired with "
            f"every matching row on the other. That is far more data than either "
            f"file contains and will not finish on a laptop. Usually it means "
            f"'{left_key}' identifies a group rather than a person: summarize one "
            f"file to one row per subject first, or pick a column that is unique "
            f"in at least one file."
        )

    # --- things to understand first ---------------------------------------
    # A preserving join keeps rows that matched nothing, and every column from
    # the other file arrives blank for them. Silence here is how a researcher
    # ends up modeling a variable that is 50% missing by construction and
    # blames the data.
    if how in ("left", "outer") and unmatched_left_rows:
        d.warnings.append(
            f"{unmatched_left_rows:,} row(s) of {left_name} "
            f"({unmatched_left_rows / max(len(left), 1):.0%}) have no match in "
            f"{right_name}. They are kept, but every column coming from "
            f"{right_name} will be blank for them."
        )
    if how in ("right", "outer") and unmatched_right_rows:
        d.warnings.append(
            f"{unmatched_right_rows:,} row(s) of {right_name} "
            f"({unmatched_right_rows / max(len(right), 1):.0%}) have no match in "
            f"{left_name}. They are kept, but every column coming from "
            f"{left_name} will be blank for them."
        )
    if needs_norm:
        d.warnings.append(
            f"Some IDs are written differently in the two files — capitalization, "
            f"stray spaces, or a trailing '.0'. Cleaning them up matches "
            f"{len(matched):,} IDs instead of fewer."
        )
    # Folding was refused for the PAIR, so say what that costs. Silence here is
    # how a file that spells one ID two ways quietly shrinks the other file's
    # cohort as well.
    if not reading.fold_case:
        folded = (set(normalize_key(ls, True, reading.keep_tokens).dropna().unique())
                  & set(normalize_key(rs, True, reading.keep_tokens).dropna().unique()))
        extra = len(folded) - len(matched)
        example = (_case_collision_example(ls, reading.keep_tokens)
                   or _case_collision_example(rs, reading.keep_tokens))
        d.warnings.append(
            f"Capitalization is being treated as meaningful in BOTH files, because one "
            f"of them holds two IDs that differ only in case"
            + (f" ({example[0]!r} and {example[1]!r})" if example else "")
            + ". "
            + (f"{extra:,} more ID(s) would match if capitalization were ignored. "
               if extra > 0 else "")
            + "If those are really one subject typed two ways, fix the spelling in your "
              "source file — otherwise every ID here is matched exactly as written."
        )
    if reading.keep_tokens:
        _kept = sorted({v for v in
                        set(ls.dropna().astype(str).str.strip().unique())
                        | set(rs.dropna().astype(str).str.strip().unique())
                        if v.lower() in reading.keep_tokens})
        # State the evidence that was actually used. "it matches the shape of
        # the other codes" is false of a token kept because both files use it,
        # and a false reason for a right decision still teaches the researcher
        # something untrue about their own data.
        _one = len(_kept) == 1
        _shared = (set(ls.dropna().astype(str).str.strip().str.lower())
                   & set(rs.dropna().astype(str).str.strip().str.lower()))
        _on_both = {v for v in _kept if v.lower() in _shared}
        if len(_on_both) == len(_kept):
            _why = f"both files use {'it' if _one else 'them'} in the ID column"
        elif not _on_both:
            _why = (f"{'it matches' if _one else 'they match'} the shape of the "
                    f"other codes in these columns")
        else:
            _why = (f"these columns are using {'it' if _one else 'them'} as one "
                    f"of their own codes")
        d.notes.append(
            f"{', '.join(repr(v) for v in _kept)} would normally be read as a blank, but "
            f"{_why}, so {'it is' if _one else 'they are'} "
            f"being matched as {'a real ID' if _one else 'real IDs'}."
        )
    if dup_left and dup_right:
        d.warnings.append(
            f"Both files have several rows per ID, so every combination is produced: "
            f"{len(matched):,} shared IDs become {matched_rows:,} rows. This is usually a "
            f"mistake — check whether one file should be summarized to one row per subject "
            f"first."
        )
    elif dup_left or dup_right:
        many = right_name if dup_right else left_name
        d.warnings.append(
            f"{many} has several rows per ID (e.g. repeated visits), so {len(matched):,} "
            f"subjects become {matched_rows:,} rows. That is correct for repeated measures, "
            f"but each subject now appears several times — your sample size is no longer the "
            f"number of subjects, and models will need a group-aware split."
        )
    if how == "inner" and unmatched_left_rows:
        pct = unmatched_left_rows / max(1, len(left))
        msg = (f"{unmatched_left_rows:,} row(s) of {left_name} ({pct:.0%}) have no match and will "
               f"be dropped.")
        (d.warnings if pct >= 0.10 else d.notes).append(
            msg + (" Use a left join to keep them." if pct >= 0.10 else "")
        )
    if how == "inner" and unmatched_right_rows:
        d.notes.append(f"{unmatched_right_rows:,} row(s) of {right_name} have no match and will be dropped.")
    if n_missing_left or n_missing_right:
        parts = []
        if n_missing_left:
            parts.append(f"{n_missing_left:,} in {left_name}")
        if n_missing_right:
            parts.append(f"{n_missing_right:,} in {right_name}")
        kept = how in ("left", "outer") and n_missing_left or how in ("right", "outer") and n_missing_right
        # Name the spellings and the reason IN THE FIRST SENTENCE. "no ID at
        # all" is a false account of a row whose ID the app read as a blank,
        # and correcting it further down does not undo it: the researcher acts
        # on the opening claim and goes looking for a gap in their file that is
        # not there. When a spelling was refused, the refusal IS the headline.
        unreadable = sorted(set(unreadable_key_spellings(ls, reading.keep_tokens))
                            | set(unreadable_key_spellings(rs, reading.keep_tokens)))
        if unreadable:
            shown = ", ".join(repr(v) for v in unreadable[:5])
            lead = (f"{' and '.join(parts)} row(s) have no ID this app can read: their ID "
                    f"is blank, or reads {shown}{'…' if len(unreadable) > 5 else ''} — "
                    f"{'a spelling' if len(unreadable) == 1 else 'spellings'} this app "
                    f"treats as a way of writing 'no value'. ")
        else:
            lead = f"{' and '.join(parts)} row(s) have no ID at all — the cell is blank. "
        d.warnings.append(
            lead
            + ("They are kept but will have no matching information attached."
               if kept else "They cannot be matched and will be dropped.")
        )
    if collisions:
        d.warnings.append(
            f"Both files have column(s) named {', '.join(collisions[:5])}"
            f"{'…' if len(collisions) > 5 else ''}. They will be kept side by side with "
            f"suffixes so nothing is overwritten."
        )
    return d


def plain_summary(d: JoinDiagnosis, left_name: str = "the first file",
                  right_name: str = "the second file") -> str:
    """The one line to show above the Merge button."""
    if d.blocking:
        return "This join will not work yet — see below."
    verb = {"inner": "keeping only IDs found in both files",
            "left": f"keeping every row of {left_name}",
            "right": f"keeping every row of {right_name}",
            "outer": "keeping every row of both files"}.get(d.how, "")
    return (f"Result: **{d.predicted_rows:,} rows** — matching on {d.matched_keys:,} shared IDs, "
            f"{verb}.")


def repair_keys(left: pd.DataFrame, right: pd.DataFrame,
                left_key: str, right_key: str) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """Make two key columns comparable. Returns NEW frames plus a description.

    Writes a canonical string form into both key columns so that "001", 1 and
    1.0 — or "A01 " and "a01" — refer to the same subject.
    """
    left_key, right_key = resolve_column(left, left_key), resolve_column(right, right_key)
    l2, r2 = left.copy(), right.copy()
    # The canonical space is a property of the PAIR, not of either column. Two
    # separate normalize_key calls put the two files in two different spaces the
    # moment one of them holds a case collision.
    reading = key_reading(l2[left_key], r2[right_key])
    l2[left_key] = normalize_key(l2[left_key], reading.fold_case, reading.keep_tokens)
    r2[right_key] = normalize_key(r2[right_key], reading.fold_case, reading.keep_tokens)
    l2.loc[l2[left_key] == "", left_key] = np.nan
    r2.loc[r2[right_key] == "", right_key] = np.nan
    return l2, r2, (
        f"Standardized the join keys ('{left_key}', '{right_key}') so that IDs written "
        f"differently (leading zeros, decimals, capitalization, stray spaces) refer to the "
        f"same subject."
    )


def execute_join(left: pd.DataFrame, right: pd.DataFrame,
                 left_key: str, right_key: str, how: str = "inner",
                 left_name: str = "left", right_name: str = "right",
                 repair: bool = True) -> Tuple[pd.DataFrame, str]:
    """Perform the join safely. Returns (frame, methods-ready description)."""
    # pandas will not merge frames whose column indexes have different depths,
    # and a two-level header reaches here from parquet. Flatten first — but
    # resolve the keys by POSITION across the rename, because flattening turns
    # ('key', 'SEQN') into 'key_SEQN' and the caller is holding the old name.
    if isinstance(left.columns, pd.MultiIndex) or isinstance(right.columns, pd.MultiIndex):
        from data_processor import flatten_columns
        li = list(left.columns).index(resolve_column(left, left_key))
        ri = list(right.columns).index(resolve_column(right, right_key))
        left, right = flatten_columns(left), flatten_columns(right)
        left_key, right_key = left.columns[li], right.columns[ri]
    left_key = resolve_column(left, left_key)
    right_key = resolve_column(right, right_key)
    # Refuse rather than merge on digits that are already gone. diagnose_join
    # blocks this in the UI; the guard is repeated here because execute_join is
    # reachable without it, and a false merge is not recoverable downstream.
    for _s, _fname, _col in ((left[left_key], left_name, left_key),
                             (right[right_key], right_name, right_key)):
        _loss = numeric_key_precision_loss(_s)
        if _loss:
            raise ValueError(_precision_refusal(_col, _fname, *_loss))
    steps: List[str] = []
    l, r = left, right
    if repair:
        l, r, desc = repair_keys(left, right, left_key, right_key)
        steps.append(desc)
        # repair_keys writes the CANONICAL form into the key column so the two
        # sides compare equal. That form is text with leading zeros stripped,
        # so a numeric SEQN came back as "1","10","2" and sorted lexically ever
        # after, and subject "001" came back as "1" — the app silently
        # retyping, and in the second case CORRUPTING, the researcher's
        # identifier. Match on the canonical form; hand back what they gave us.
        l, r = l.copy(), r.copy()
        # Stash as plain objects. A Categorical key kept its dtype here, and
        # the restore step below (`merged[lo].where(notna, merged[ro])`) then
        # raised on a right/outer join, because merged[ro] holds right-only
        # values that are not in the left column's category list. Parquet
        # round-trips produce Categorical keys routinely, and the raw pandas
        # error is exactly the failure this module exists to eliminate.
        l[_ORIGINAL_KEY] = np.asarray(left[left_key].astype(object).values)
        r[_ORIGINAL_KEY] = np.asarray(right[right_key].astype(object).values)

    suffixes = (f"_{_slug(left_name)}", f"_{_slug(right_name)}")

    # pandas merges NaN to NaN, so a plain merge pairs every ID-less row on the
    # left with every ID-less row on the right — fabricating participants that
    # carry real measurements. Rows without an ID are therefore held out of the
    # merge entirely and re-attached afterwards on whichever side the join type
    # preserves (SQL's semantics, which is what a researcher expects).
    # Use the canonical notion of "missing" so an empty string or "unknown"
    # counts as no-ID even when the keys were not repaired — otherwise the
    # diagnosis and the merge disagree about which rows can match.
    _blank_reading = key_reading(l[left_key], r[right_key])
    lmask = normalize_key(l[left_key], _blank_reading.fold_case,
                          _blank_reading.keep_tokens).isna()
    rmask = normalize_key(r[right_key], _blank_reading.fold_case,
                          _blank_reading.keep_tokens).isna()
    l_blank, r_blank = l[lmask], r[rmask]
    merged = l[~lmask].merge(
        r[~rmask], left_on=left_key, right_on=right_key, how=how, suffixes=suffixes
    )

    extras: List[pd.DataFrame] = []
    # Subtracting BOTH key names is wrong when they differ. A right-hand column
    # merely NAMED like the left key (demographics keyed on SEQN, labs keyed on
    # pid but also carrying its own SEQN notes column) is a genuine collision:
    # pandas suffixes it, so `SEQN` stops existing in the result, the drop of
    # right_key removes the other one, and the restore step's
    # `left_key in merged.columns` is False — the key column the researcher
    # joined on is simply gone from their data.
    overlap = (set(l.columns) & set(r.columns)) - {left_key} - {right_key}
    if left_key != right_key and left_key in r.columns:
        overlap.add(left_key)
    if how in ("left", "outer") and len(l_blank):
        extras.append(l_blank.rename(columns={c: f"{c}{suffixes[0]}" for c in overlap}))
    if how in ("right", "outer") and len(r_blank):
        rb = r_blank.rename(columns={c: f"{c}{suffixes[1]}" for c in overlap})
        if left_key != right_key:
            rb = rb.rename(columns={right_key: left_key})
        extras.append(rb)
    if extras:
        merged = pd.concat([merged] + extras, ignore_index=True)

    if left_key != right_key and right_key in merged.columns:
        merged = merged.drop(columns=[right_key])
    # pandas suffixed the left key too when the right file carried that name;
    # put it back so the column the user joined on is present and unambiguous.
    if left_key not in merged.columns:
        suffixed = f"{left_key}{suffixes[0]}"
        if suffixed in merged.columns:
            merged = merged.rename(columns={suffixed: left_key})

    if repair:
        # Restore the originals. On an outer join a row may have come from
        # either side, so take whichever side actually supplied it.
        lo = f"{_ORIGINAL_KEY}{suffixes[0]}"
        ro = f"{_ORIGINAL_KEY}{suffixes[1]}"
        restored = None
        if lo in merged.columns and ro in merged.columns:
            restored = merged[lo].where(merged[lo].notna(), merged[ro])
        elif lo in merged.columns:
            restored = merged[lo]
        elif ro in merged.columns:
            restored = merged[ro]
        elif _ORIGINAL_KEY in merged.columns:
            restored = merged[_ORIGINAL_KEY]
        drop = [c for c in (lo, ro, _ORIGINAL_KEY) if c in merged.columns]
        if drop:
            merged = merged.drop(columns=drop)
        if restored is not None and left_key in merged.columns:
            merged[left_key] = restored.values
            if pd.api.types.is_numeric_dtype(left[left_key]):
                # Whole numbers before the merge stay whole numbers after it,
                # even where an outer join introduced blanks.
                try:
                    merged[left_key] = pd.to_numeric(merged[left_key], errors="coerce")
                except Exception:
                    pass

    steps.append(
        f"Merged {left_name} ({len(left):,} rows) with {right_name} ({len(right):,} rows) "
        f"on '{left_key}' using a {how} join, giving {len(merged):,} rows."
    )
    return merged, " ".join(steps)


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(name)).strip("_")[:20] or "x"


def suggest_best(left: pd.DataFrame, right: pd.DataFrame,
                 include_low: bool = False) -> Optional[KeyCandidate]:
    """The single key we would propose, or None if nothing plausible links them.

    Low-confidence candidates are withheld by default: telling a researcher
    "these files join on age ↔ gdp" is far worse than saying nothing and
    letting them pick the key themselves.
    """
    for c in find_key_candidates(left, right):
        if include_low or c.confidence != "low":
            return c
    return None
