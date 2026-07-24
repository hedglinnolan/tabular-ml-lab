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
5. The IDs differ only by stray whitespace or capitalisation.

None of that is discoverable by looking at column names, which is all the
previous helpers did. This module looks at the VALUES: it proposes keys by
measuring how well they actually overlap, explains what a join will do in
plain language BEFORE it runs, and offers reversible normalisations for the
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

# Rows sampled when scoring candidate keys on wide/long files.
_SAMPLE_ROWS = 5000
# A key must identify rows, not group them: at least this share of values must
# be distinct on one side, or "sex" would look like a perfect key.
_MIN_UNIQUENESS = 0.5
# Below this overlap a pairing is not worth proposing.
_MIN_COVERAGE = 0.05


def normalize_key(s: pd.Series) -> pd.Series:
    """Canonical comparison form for a key column.

    Numeric-looking values are compared as numbers so that "001", 1 and 1.0
    are the same subject; everything else is trimmed and lower-cased. This is
    used for DIAGNOSIS and for the optional repair — never applied silently.
    """
    if s is None:
        return pd.Series(dtype="object")
    if pd.api.types.is_numeric_dtype(s):
        out = pd.to_numeric(s, errors="coerce")
        # 1 and 1.0 must land on the same token.
        return out.map(lambda v: "" if pd.isna(v) else
                       (str(int(v)) if float(v).is_integer() else str(float(v))))
    t = s.astype(str).str.strip()
    as_num = pd.to_numeric(t, errors="coerce")
    numeric_share = float(as_num.notna().mean()) if len(t) else 0.0
    if numeric_share >= 0.95:
        # Zero-padded IDs from CSV ("001") vs integers from Excel (1).
        return as_num.map(lambda v: "" if pd.isna(v) else
                          (str(int(v)) if float(v).is_integer() else str(float(v))))
    return t.str.lower()


def _name_similarity(a: str, b: str) -> float:
    ca, cb = re.sub(r"[^a-z0-9]", "", str(a).lower()), re.sub(r"[^a-z0-9]", "", str(b).lower())
    if not ca or not cb:
        return 0.0
    if ca == cb:
        return 1.0
    return SequenceMatcher(None, ca, cb).ratio()


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
    index_like: bool = False    # both sides are plain row counters

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
        if self.index_like and self.name_similarity < 0.85:
            s *= 0.15   # a coincidental counter overlap, not a real key
        return s

    @property
    def confidence(self) -> str:
        """How safely a UI may present this — never auto-apply 'low'."""
        if self.index_like and self.name_similarity < 0.85:
            return "low"
        if self.name_similarity >= 0.85 and min(self.coverage_left, self.coverage_right) >= 0.5:
            return "high"
        if max(self.coverage_left, self.coverage_right) >= 0.8 and not self.index_like:
            return "high" if self.name_similarity >= 0.6 else "medium"
        return "medium" if max(self.coverage_left, self.coverage_right) >= 0.5 else "low"

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
                    f"capitalisation and stray spaces ({self.n_matched:,} IDs).")
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


def _prep(df: pd.DataFrame, col: str, sample: bool = True) -> pd.Series:
    s = df[col]
    if sample and len(s) > _SAMPLE_ROWS:
        s = s.sample(_SAMPLE_ROWS, random_state=42)
    return s


def find_key_candidates(left: pd.DataFrame, right: pd.DataFrame,
                        max_columns: int = 60,
                        min_coverage: float = _MIN_COVERAGE) -> List[KeyCandidate]:
    """Propose ways to link two files, ranked by real value overlap.

    Column NAMES are only a tiebreak — a key named SEQN in one file and
    patient_id in the other is found because the values line up.
    """
    if left is None or right is None or left.empty or right.empty:
        return []

    lcols = list(left.columns)[:max_columns]
    rcols = list(right.columns)[:max_columns]

    # Precompute normalized value sets once per column.
    lnorm: Dict[str, pd.Series] = {}
    rnorm: Dict[str, pd.Series] = {}
    for c in lcols:
        try:
            lnorm[c] = normalize_key(_prep(left, c)).replace("", np.nan).dropna()
        except Exception:
            continue
    for c in rcols:
        try:
            rnorm[c] = normalize_key(_prep(right, c)).replace("", np.nan).dropna()
        except Exception:
            continue

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
            cov_l = len(matched) / len(lset)
            cov_r = len(matched) / len(rset)
            if max(cov_l, cov_r) < min_coverage:
                continue
            # A key must identify rows rather than group them.
            uniq = max(len(lset) / max(1, len(ls)), len(rset) / max(1, len(rs)))
            if uniq < _MIN_UNIQUENESS:
                continue

            lraw, rraw = left[lc], right[rc]
            dtype_mismatch = (
                pd.api.types.is_numeric_dtype(lraw) != pd.api.types.is_numeric_dtype(rraw)
            )
            # Whitespace/case only matters when both sides are text.
            needs_norm = False
            if not dtype_mismatch and not pd.api.types.is_numeric_dtype(lraw):
                raw_match = set(lraw.astype(str).dropna().unique()) & set(rraw.astype(str).dropna().unique())
                needs_norm = len(raw_match) < len(matched)

            out.append(KeyCandidate(
                left_col=str(lc), right_col=str(rc),
                coverage_left=cov_l, coverage_right=cov_r,
                n_matched=len(matched),
                left_unique=len(lset), right_unique=len(rset),
                left_rows=len(ls), right_rows=len(rs),
                dtype_mismatch=dtype_mismatch,
                needs_normalization=needs_norm,
                left_has_duplicates=bool(ls.duplicated().any()),
                right_has_duplicates=bool(rs.duplicated().any()),
                name_similarity=_name_similarity(lc, rc),
                index_like=_looks_like_row_index(lraw) and _looks_like_row_index(rraw),
            ))

    out.sort(key=lambda c: c.score, reverse=True)
    return out


def diagnose_join(left: pd.DataFrame, right: pd.DataFrame,
                  left_key: str, right_key: str, how: str = "inner",
                  left_name: str = "the first file",
                  right_name: str = "the second file") -> JoinDiagnosis:
    """Explain what this join will do BEFORE it runs.

    Predicts the row count (including fan-out from duplicate keys), names
    every mechanical blocker, and says out loud how much of each cohort is
    about to be dropped.
    """
    ls, rs = left[left_key], right[right_key]
    ln, rn = normalize_key(ls), normalize_key(rs)
    lvalid, rvalid = ln[ln != ""], rn[rn != ""]
    lset, rset = set(lvalid.unique()), set(rvalid.unique())
    matched = lset & rset

    dtype_mismatch = (
        pd.api.types.is_numeric_dtype(ls) != pd.api.types.is_numeric_dtype(rs)
    )
    needs_norm = False
    if not dtype_mismatch and not pd.api.types.is_numeric_dtype(ls):
        raw_match = set(ls.astype(str).dropna().unique()) & set(rs.astype(str).dropna().unique())
        needs_norm = len(raw_match) < len(matched)

    # Predict rows, accounting for fan-out when a key repeats.
    lcounts = lvalid.value_counts()
    rcounts = rvalid.value_counts()
    matched_rows = int(sum(lcounts.get(k, 0) * rcounts.get(k, 0) for k in matched))
    unmatched_left_rows = int(sum(lcounts.get(k, 0) for k in (lset - matched)))
    unmatched_right_rows = int(sum(rcounts.get(k, 0) for k in (rset - matched)))

    if how == "inner":
        predicted = matched_rows
    elif how == "left":
        predicted = matched_rows + unmatched_left_rows
    elif how == "right":
        predicted = matched_rows + unmatched_right_rows
    else:  # outer
        predicted = matched_rows + unmatched_left_rows + unmatched_right_rows

    collisions = [str(c) for c in (set(left.columns) & set(right.columns))
                  if str(c) not in {str(left_key), str(right_key)}]

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
    if dtype_mismatch:
        d.blocking.append(
            f"'{left_key}' is stored as {'numbers' if pd.api.types.is_numeric_dtype(ls) else 'text'} "
            f"in {left_name} but as {'numbers' if pd.api.types.is_numeric_dtype(rs) else 'text'} in "
            f"{right_name}. They look identical on screen but will not match. "
            f"Fixing this matches {len(matched):,} IDs."
        )
    elif not matched:
        d.blocking.append(
            f"None of the values in '{left_key}' appear in '{right_key}', so there is nothing "
            f"to join on. Check you picked the right columns."
        )

    # --- things to understand first ---------------------------------------
    if needs_norm:
        d.warnings.append(
            f"Some IDs differ only by capitalisation or stray spaces. Cleaning them up "
            f"matches {len(matched):,} IDs instead of fewer."
        )
    if dup_left and dup_right:
        d.warnings.append(
            f"Both files have several rows per ID, so every combination is produced: "
            f"{len(matched):,} shared IDs become {matched_rows:,} rows. This is usually a "
            f"mistake — check whether one file should be summarised to one row per subject "
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
    l2, r2 = left.copy(), right.copy()
    l2[left_key] = normalize_key(l2[left_key])
    r2[right_key] = normalize_key(r2[right_key])
    l2.loc[l2[left_key] == "", left_key] = np.nan
    r2.loc[r2[right_key] == "", right_key] = np.nan
    return l2, r2, (
        f"Standardised the join keys ('{left_key}', '{right_key}') so that IDs written "
        f"differently (leading zeros, decimals, capitalisation, stray spaces) refer to the "
        f"same subject."
    )


def execute_join(left: pd.DataFrame, right: pd.DataFrame,
                 left_key: str, right_key: str, how: str = "inner",
                 left_name: str = "left", right_name: str = "right",
                 repair: bool = True) -> Tuple[pd.DataFrame, str]:
    """Perform the join safely. Returns (frame, methods-ready description)."""
    steps: List[str] = []
    l, r = left, right
    if repair:
        l, r, desc = repair_keys(left, right, left_key, right_key)
        steps.append(desc)

    suffixes = (f"_{_slug(left_name)}", f"_{_slug(right_name)}")
    merged = l.merge(r, left_on=left_key, right_on=right_key, how=how, suffixes=suffixes)
    if left_key != right_key and right_key in merged.columns:
        merged = merged.drop(columns=[right_key])

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
