"""Before and after: exactly what combining does to a researcher's data.

The engines can already predict a combine correctly. That is not the same as a
researcher UNDERSTANDING it. "This produces 450 rows" is mechanically true and
analytically useless — it does not say which people survived, where their
columns went, which cells are blank because of the merge rather than because
the measurement was missing, or what any of that does to their study.

So this module turns a planned combine into a ChangeMap: the same operation
described three ways a person can actually read.

    WHO      every row accounted for — matched, dropped, or kept-and-blanked
    WHERE    every column in the result, and which file it came from
    SO WHAT  what this does to the ANALYSIS, not to the table

That last one is the point. A researcher does not need to be told that an
inner join drops rows; they need to be told that their cohort is now a
selected subsample, and that if the selection relates to their outcome the
effect estimate is biased. The engine knows enough to say so, and used to say
nothing.

Everything here is pure and returns data. Rendering lives in combine_ui.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ml.join_doctor import normalize_key
from utils.combine import SOURCE_COLUMN

# Where a column in the result came from. Used for color and for the legend.
FROM_KEY = "key"          # the identifier the files were linked on
FROM_LEFT = "left"
FROM_RIGHT = "right"
FROM_ADDED = "added"      # bookkeeping the app itself introduced
FROM_SHARED = "shared"    # stacking: present in every file


@dataclass
class RowGroup:
    """One outcome that some of the rows meet."""
    label: str                 # "in both files", "only in demographics"
    n: int
    outcome: str               # 'kept' | 'blanked' | 'dropped'
    detail: str = ""

    @property
    def is_loss(self) -> bool:
        return self.outcome == "dropped"


@dataclass
class ColumnMove:
    """One column of the result, and where it came from."""
    name: str
    origin: str                # FROM_* constant
    source_file: str = ""
    renamed_from: str = ""     # set when a name collision forced a suffix

    @property
    def was_renamed(self) -> bool:
        return bool(self.renamed_from)


@dataclass
class ChangeMap:
    """A planned combine, described so a person can check it."""
    operation: str                       # 'link' | 'stack'
    before: List[Tuple[str, int, int]]   # (file, rows, cols)
    after_rows: int
    after_cols: int
    row_groups: List[RowGroup] = field(default_factory=list)
    columns: List[ColumnMove] = field(default_factory=list)
    consequences: List[str] = field(default_factory=list)
    key_label: str = ""
    # Enough to tell "this person was never measured" from "this person was not
    # in that file" — identical in a table, completely different in meaning.
    left_only_keys: set = field(default_factory=set)    # link: blank on the RIGHT
    right_only_keys: set = field(default_factory=set)   # link: blank on the LEFT
    missing_from: Dict[str, List[str]] = field(default_factory=dict)  # stack: col -> files

    @property
    def rows_dropped(self) -> int:
        return sum(g.n for g in self.row_groups if g.outcome == "dropped")

    @property
    def rows_blanked(self) -> int:
        return sum(g.n for g in self.row_groups if g.outcome == "blanked")

    @property
    def renamed_columns(self) -> List[ColumnMove]:
        return [c for c in self.columns if c.was_renamed]

    def headline(self) -> str:
        before = " + ".join(f"{n} ({r:,} rows)" for n, r, _ in self.before)
        return f"{before}  →  {self.after_rows:,} rows × {self.after_cols} columns"


# Imported, never re-implemented: a private copy drifted from the real one on
# non-ASCII and empty names, so a predicted column name could disagree with the
# delivered one for a file called "é$@".
from ml.join_doctor import _slug


# ── linking ──────────────────────────────────────────────────────────────

def describe_join(left: pd.DataFrame, right: pd.DataFrame,
                  left_key: Any, right_key: Any, how: str,
                  left_name: str, right_name: str) -> ChangeMap:
    """What linking these two files does, row by row and column by column."""
    ln = normalize_key(left[left_key])
    rn = normalize_key(right[right_key])
    lset = set(ln.dropna().unique())
    rset = set(rn.dropna().unique())
    matched = lset & rset

    lcounts = ln.dropna().value_counts()
    rcounts = rn.dropna().value_counts()
    matched_rows_left = int(sum(lcounts.get(k, 0) for k in matched))
    matched_rows = int(sum(lcounts.get(k, 0) * rcounts.get(k, 0) for k in matched))
    only_left = int(sum(lcounts.get(k, 0) for k in (lset - matched)))
    only_right = int(sum(rcounts.get(k, 0) for k in (rset - matched)))
    no_id_left = int(ln.isna().sum())
    no_id_right = int(rn.isna().sum())

    keeps_left = how in ("left", "outer")
    keeps_right = how in ("right", "outer")

    groups: List[RowGroup] = [RowGroup(
        label="in both files", n=matched_rows_left, outcome="kept",
        detail=(f"{len(matched):,} shared IDs"
                + (f", producing {matched_rows:,} rows after matching"
                   if matched_rows != matched_rows_left else "")),
    )]
    if only_left:
        groups.append(RowGroup(
            label=f"only in {left_name.strip()}", n=only_left,
            outcome="blanked" if keeps_left else "dropped",
            detail=(f"kept, with every {right_name} column blank"
                    if keeps_left else f"not found in {right_name}")))
    if only_right:
        groups.append(RowGroup(
            label=f"only in {right_name.strip()}", n=only_right,
            outcome="blanked" if keeps_right else "dropped",
            detail=(f"kept, with every {left_name} column blank"
                    if keeps_right else f"not found in {left_name}")))
    for n_missing, side, keeps, other in ((no_id_left, left_name, keeps_left, right_name),
                                          (no_id_right, right_name, keeps_right, left_name)):
        if n_missing:
            groups.append(RowGroup(
                label=f"no ID in {side.strip()}", n=n_missing,
                outcome="blanked" if keeps else "dropped",
                detail=("a blank ID matches nobody — these are never paired with "
                        f"{other}")))

    # Columns, in the order the merge produces them.
    overlap = {str(c) for c in left.columns} & {str(c) for c in right.columns}
    overlap -= {str(left_key), str(right_key)}
    lsuf, rsuf = f"_{_slug(left_name)}", f"_{_slug(right_name)}"
    cols: List[ColumnMove] = [ColumnMove(str(left_key), FROM_KEY, "both files")]
    for c in left.columns:
        if str(c) == str(left_key):
            continue
        renamed = f"{c}{lsuf}" if str(c) in overlap else str(c)
        cols.append(ColumnMove(renamed, FROM_LEFT, left_name,
                               str(c) if renamed != str(c) else ""))
    for c in right.columns:
        if str(c) == str(right_key):
            continue
        renamed = f"{c}{rsuf}" if str(c) in overlap else str(c)
        cols.append(ColumnMove(renamed, FROM_RIGHT, right_name,
                               str(c) if renamed != str(c) else ""))

    predicted = matched_rows
    if keeps_left:
        predicted += only_left + no_id_left
    if keeps_right:
        predicted += only_right + no_id_right

    cm = ChangeMap(
        operation="link", after_rows=predicted, after_cols=len(cols),
        before=[(left_name, len(left), left.shape[1]),
                (right_name, len(right), right.shape[1])],
        row_groups=groups, columns=cols, key_label=str(left_key),
        left_only_keys=set(lset - matched), right_only_keys=set(rset - matched),
    )
    cm.consequences = _join_consequences(cm, left, right, lcounts, rcounts,
                                         matched, left_name, right_name, how)
    return cm


def _join_consequences(cm: ChangeMap, left, right, lcounts, rcounts, matched,
                       left_name: str, right_name: str, how: str) -> List[str]:
    """What this join does to the STUDY. The part nobody was being told."""
    out: List[str] = []

    dup_left = any(lcounts.get(k, 0) > 1 for k in matched)
    dup_right = any(rcounts.get(k, 0) > 1 for k in matched)
    if dup_left and dup_right:
        out.append(
            "**Every combination is produced.** Both files have several rows per "
            "ID, so each row on one side pairs with each row on the other. This is "
            "almost always a mistake — the result is not a cohort, and no count in "
            "it means anything.")
    elif dup_left or dup_right:
        many = left_name if dup_left else right_name
        out.append(
            f"**Your n is no longer the number of people.** {many} has several rows "
            f"per ID, so each person appears several times in the result. Anything "
            f"averaged across rows now weights people by how many measurements they "
            f"have, and the held-out test set has to be split by person rather than "
            f"by row — the app does that for you once it sees the repeats.")

    dropped = cm.rows_dropped
    if dropped:
        total_in = sum(r for _, r, _ in cm.before)
        share = dropped / max(total_in, 1)
        out.append(
            f"**Your cohort becomes a subsample.** {dropped:,} row(s) "
            f"({share:.0%} of what you brought) are not carried into the result. "
            f"If being present in both files is related to what you are predicting "
            f"— sicker people getting more lab work, say — the remaining sample is "
            f"selected, and estimates from it are biased for the full cohort.")

    blanked = cm.rows_blanked
    if blanked:
        out.append(
            f"**Some columns will be missing by construction.** {blanked:,} row(s) "
            f"are kept without a match, so every column from the other file is "
            f"blank for them. That missingness is a fact about the merge, not about "
            f"the measurement, and imputing it would invent data.")

    # A clash in a column the APP added is the app's problem, not something to
    # hand back to the researcher as a decision.
    user_renamed = [c for c in cm.renamed_columns
                    if not str(c.renamed_from).startswith(SOURCE_COLUMN)]
    if user_renamed:
        names = ", ".join(f"`{c.renamed_from}` → `{c.name}`" for c in user_renamed[:4])
        out.append(
            f"**Both files had columns with the same name**, so they are kept side "
            f"by side rather than one overwriting the other: {names}. Check which "
            f"one you actually want before modeling.")
    return out


# ── stacking ─────────────────────────────────────────────────────────────

# NHANES-style survey weights: WTMEC2YR, WTINT2YR, WTDRD1, WTSAF2YR. Matched
# narrowly ON PURPOSE. A body-weight column (BMXWT, weight_kg, birth_weight)
# must never trigger this, and neither must 'sample_weight' — in -omics and
# food chemistry that is the physical mass of a specimen, and telling someone
# to divide it by the number of cycles would be nonsense.
_SURVEY_WEIGHT = __import__("re").compile(
    r"^(WT[A-Z]{2,6}\d?YR\d?|WTDRD\d|WTDR\d?D?\d?|"
    r"(survey|sampling|analysis|person)_?weight|pweight|perweight)$",
    __import__("re").IGNORECASE)


def survey_weight_columns(frames: Dict[str, pd.DataFrame]) -> List[str]:
    """Columns that look like complex-survey sampling weights."""
    seen: List[str] = []
    for f in frames.values():
        if f is None:
            continue
        for c in f.columns:
            name = str(c)
            if _SURVEY_WEIGHT.match(name) and name not in seen:
                seen.append(name)
    return seen


def describe_stack(frames: Dict[str, pd.DataFrame]) -> ChangeMap:
    """What stacking these files does, row by row and column by column."""
    names = [n for n, f in frames.items() if f is not None]
    col_sets = {n: [str(c) for c in frames[n].columns] for n in names}
    shared = set.intersection(*[set(v) for v in col_sets.values()]) if names else set()
    union: List[str] = []
    for n in names:
        for c in col_sets[n]:
            if c not in union:
                union.append(c)

    groups = [RowGroup(label=f"from {n}", n=len(frames[n]), outcome="kept",
                       detail=f"{len(col_sets[n])} columns")
              for n in names]

    cols: List[ColumnMove] = []
    missing_from: Dict[str, List[str]] = {}
    for c in union:
        present = [n for n in names if c in col_sets[n]]
        if len(present) == len(names):
            cols.append(ColumnMove(c, FROM_SHARED, "every file"))
        else:
            cols.append(ColumnMove(c, FROM_LEFT, ", ".join(present)))
            missing_from[c] = [n for n in names if c not in col_sets[n]]
    cols.append(ColumnMove(SOURCE_COLUMN, FROM_ADDED, "added by the app"))

    total = sum(len(frames[n]) for n in names)
    cm = ChangeMap(
        operation="stack", after_rows=total, after_cols=len(cols),
        before=[(n, len(frames[n]), frames[n].shape[1]) for n in names],
        row_groups=groups, columns=cols, missing_from=missing_from,
    )

    partial = [c for c in cols if c.origin == FROM_LEFT]
    if partial:
        worst = partial[0]
        missing_in = [n for n in names if worst.name not in col_sets[n]]
        cm.consequences.append(
            f"**{len(partial)} column(s) are not in every file** — for example "
            f"`{worst.name}`, which {', '.join(missing_in)} does not have. Those "
            f"cells are blank for the files that lack the column. The blank means "
            f"'not collected here', not 'measured and missing', and treating the "
            f"two the same will distort any missingness summary.")
    weights = survey_weight_columns(frames)
    if weights and len(names) > 1:
        cm.consequences.append(
            f"**These files carry survey weights ({', '.join(weights[:4])}), and "
            f"stacking them does not combine the weights.** A 2-year NHANES "
            f"weight represents the population for ITS cycle only; using it "
            f"across {len(names)} stacked cycles counts the US population "
            f"{len(names)} times over. CDC's rule is to divide each 2-year "
            f"weight by the number of cycles you combined — here {len(names)} — "
            f"before any weighted analysis. (1999-2000 is the exception: it uses "
            f"a different population base, so NCHS publishes a 4-year weight to "
            f"use instead of dividing.)")

    cm.consequences.append(
        f"**Stacking creates a batch variable.** Rows now come from {len(names)} "
        f"different files, and differences between those files — collection year, "
        f"site, instrument — can look like a real effect. `{SOURCE_COLUMN}` records "
        f"the origin so you can check for it; it is kept out of the feature pool so "
        f"no model can predict it by accident.")
    return cm


# ── which cells the preview should mark ──────────────────────────────────

def blank_cell_mask(result: pd.DataFrame, cm: ChangeMap) -> pd.DataFrame:
    """True where a cell is blank BECAUSE of the combine, not because the
    measurement was missing.

    The two are identical on screen and mean completely different things: one
    is "we never measured this person", the other is "this person was not in
    that file at all". The second is a fact about the merge, and imputing it
    would invent data.

    Flagging every NaN — which is what this used to do — does not distinguish
    them, so it was worse than useless: it lent the wrong reading a color.
    """
    mask = pd.DataFrame(False, index=result.index, columns=result.columns)
    by_origin = {c.name: c.origin for c in cm.columns}

    if cm.operation == "link":
        if not cm.key_label or cm.key_label not in result.columns:
            return mask
        keys = normalize_key(result[cm.key_label])
        unmatched_right = keys.isin(cm.left_only_keys)   # left-only row
        unmatched_left = keys.isin(cm.right_only_keys)   # right-only row
        for col in result.columns:
            origin = by_origin.get(str(col))
            if origin == FROM_RIGHT:
                mask[col] = unmatched_right & result[col].isna()
            elif origin == FROM_LEFT:
                mask[col] = unmatched_left & result[col].isna()
        return mask

    if SOURCE_COLUMN not in result.columns:
        return mask
    src = result[SOURCE_COLUMN]
    for col, absent_in in cm.missing_from.items():
        if col in result.columns:
            mask[col] = src.isin(absent_in) & result[col].isna()
    return mask
