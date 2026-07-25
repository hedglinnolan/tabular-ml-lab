"""Combining several files into one analysis table.

Researchers arrive with files that relate in exactly two ways, and the app
previously supported only one of them:

  STACK  — the same measurements on different people. NHANES cycles
           1999-2000, 2001-2002, …; sites in a multi-center study; years of a
           registry. The columns line up; the rows add up.
  LINK   — different measurements on the same people. Demographics + diet +
           labs, all keyed by SEQN. The rows line up; the columns add up.

Stacking was not implemented at all, which is why "combine my NHANES cycles"
had no answer. This module provides both as pure functions so they can be
tested without a browser, and so the page can stay thin.

Same contract as the Import and Join Doctors: diagnose before acting, predict
what will happen, never guess silently, and return a plain-language
description fit for a methods section.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Column added to record which file each row came from. Reserved: the feature
# pool must exclude it or a model will happily "predict" the source file.
SOURCE_COLUMN = "__source_file"


@dataclass
class StackPlan:
    """What stacking these frames will produce, before it happens."""
    frames: List[str]
    shared_columns: List[str]
    total_rows: int
    all_columns: List[str]
    partial_columns: Dict[str, List[str]] = field(default_factory=dict)
    type_conflicts: Dict[str, List[str]] = field(default_factory=dict)
    blocking: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def can_proceed(self) -> bool:
        return not self.blocking

    def summary(self) -> str:
        if self.blocking:
            return "These files cannot be stacked yet — see below."
        return (f"Result: **{self.total_rows:,} rows** and "
                f"{len(self.all_columns) + 1} columns, stacking {len(self.frames)} files "
                f"that share {len(self.shared_columns)} column(s) — plus one column "
                f"recording which file each row came from.")


def _dtype_family(s: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        return "number"
    if pd.api.types.is_bool_dtype(s):
        return "true/false"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "date"
    return "text"


def plan_stack(frames: Dict[str, pd.DataFrame]) -> StackPlan:
    """Diagnose stacking WITHOUT doing it.

    Reports the columns every file has, the ones only some files have (which
    become blank for the others), and any column whose type disagrees between
    files — the quiet killer, because a column that is numbers in one cycle and
    text in another silently becomes unusable text overall.
    """
    names = [n for n, f in frames.items() if f is not None]
    plan = StackPlan(frames=names, shared_columns=[], total_rows=0, all_columns=[])
    if len(names) < 2:
        plan.blocking.append("Stacking needs at least two files.")
        return plan

    col_sets = {n: list(frames[n].columns) for n in names}
    shared = set(col_sets[names[0]])
    union: List[str] = []
    for n in names:
        shared &= set(col_sets[n])
        for c in col_sets[n]:
            if c not in union:
                union.append(c)

    plan.shared_columns = [c for c in union if c in shared]
    plan.all_columns = union
    plan.total_rows = int(sum(len(frames[n]) for n in names))

    if not shared:
        plan.blocking.append(
            "These files have no column names in common, so stacking them would "
            "produce one block of data per file with nothing lining up. Check "
            "whether you meant to link them by a shared ID instead."
        )
        return plan

    # Columns present in some files but not others.
    for c in union:
        missing_in = [n for n in names if c not in col_sets[n]]
        if missing_in:
            plan.partial_columns[str(c)] = missing_in

    # Type disagreements across files for a shared column.
    for c in plan.shared_columns:
        fams = {}
        for n in names:
            fams.setdefault(_dtype_family(frames[n][c]), []).append(n)
        if len(fams) > 1:
            plan.type_conflicts[str(c)] = sorted(fams.keys())

    overlap_frac = len(shared) / max(1, len(union))
    if overlap_frac < 0.5:
        plan.warnings.append(
            f"Only {len(shared)} of {len(union)} columns appear in every file "
            f"({overlap_frac:.0%}). The rest will be blank for the files that lack them."
        )
    elif plan.partial_columns:
        plan.notes.append(
            f"{len(plan.partial_columns)} column(s) are missing from at least one file "
            f"and will be blank for those rows."
        )
    if plan.type_conflicts:
        detail = "; ".join(f"'{c}' is " + " in some files and ".join(v)
                           for c, v in list(plan.type_conflicts.items())[:3])
        plan.warnings.append(
            f"{len(plan.type_conflicts)} column(s) hold different kinds of value in "
            f"different files ({detail}). After stacking they become text, which no "
            f"model can use until it is cleaned up."
        )
    return plan


def execute_stack(frames: Dict[str, pd.DataFrame],
                  add_source_column: bool = True) -> Tuple[pd.DataFrame, str]:
    """Stack frames end to end. Returns (frame, methods-ready description)."""
    names = [n for n, f in frames.items() if f is not None]
    if len(names) < 2:
        raise ValueError("Stacking needs at least two files.")

    parts = []
    for n in names:
        part = frames[n].copy()
        if add_source_column:
            part[SOURCE_COLUMN] = n
        parts.append(part)

    out = pd.concat(parts, ignore_index=True, sort=False)
    per_file = ", ".join(f"{n} ({len(frames[n]):,})" for n in names)
    desc = (f"Stacked {len(names)} files end to end — {per_file} — giving "
            f"{len(out):,} rows and {out.shape[1]} columns."
            + (f" The column '{SOURCE_COLUMN}' records which file each row came from."
               if add_source_column else ""))
    return out, desc


def relationship_hint(frames: Dict[str, pd.DataFrame]) -> str:
    """Guess whether these files should be stacked or linked, for the UI.

    Returns 'stack', 'link', or 'unclear'. A hint only — the user chooses.
    """
    names = [n for n, f in frames.items() if f is not None]
    if len(names) < 2:
        return "unclear"
    col_sets = [set(frames[n].columns) for n in names]
    shared = set.intersection(*col_sets)
    union = set.union(*col_sets)
    if not union:
        return "unclear"
    overlap = len(shared) / len(union)
    if overlap >= 0.8:
        return "stack"          # nearly identical schemas -> same measurements
    if overlap <= 0.4:
        return "link"           # mostly different columns -> different measurements
    return "unclear"


def reserved_columns() -> List[str]:
    """Columns the modeling feature pool must never offer as predictors."""
    return [SOURCE_COLUMN]


def is_reserved_column(name: Any) -> bool:
    """True for any bookkeeping column added while combining files.

    Matches on PREFIX, not equality. Stacking two groups and then linking them
    produces '__source_file_demo' and '__source_file_labs' — the join suffixes
    the collision — and an exact-match check would let both through into the
    feature pool. A model that can see which file a row came from will predict
    the batch, which is leakage wearing a lab coat.
    """
    return str(name).startswith(SOURCE_COLUMN)


# ── files that need BOTH operations ──────────────────────────────────────
#
# The two-way question (stack or link?) has no right answer for the shape most
# survey research actually arrives in:
#
#     demo_2017  demo_2019  labs_2017  labs_2019
#
# That is two cycles of two domains, and it needs stacking WITHIN a domain and
# linking ACROSS domains. Asked to pick one operation for all four files, the
# researcher gets 400 rows that are half empty, or a join proposed on `age`.
# Neither is their table, and both look plausible enough to keep working with.
#
# So the files are grouped first: files holding the same measurements form a
# group, groups are stacked internally, and the stacked results are linked.

_TOKEN_SPLIT = __import__("re").compile(r"[^A-Za-z0-9]+")


@dataclass
class FileGroup:
    """Files holding the same measurements on different people."""
    label: str
    members: List[str]

    @property
    def is_stack(self) -> bool:
        return len(self.members) > 1


@dataclass
class CombinationPlan:
    """How a whole set of files becomes one table."""
    groups: List[FileGroup]
    shape: str                       # single|stack|link|stack_then_link
    notes: List[str] = field(default_factory=list)

    @property
    def needs_stacking(self) -> bool:
        return any(g.is_stack for g in self.groups)

    @property
    def needs_linking(self) -> bool:
        return len(self.groups) > 1

    def describe(self) -> str:
        """The plan in the researcher's language, before anything runs."""
        if self.shape == "single":
            return "There is only one file, so it becomes your table directly."
        if self.shape == "stack":
            return (f"These {len(self.groups[0].members)} files hold the same "
                    f"measurements on different people, so their rows are added "
                    f"together into one table.")
        if self.shape == "link":
            return (f"These {len(self.groups)} files hold different measurements "
                    f"on the same people, so they are linked side by side by a "
                    f"shared ID.")
        stacked = [g for g in self.groups if g.is_stack]
        return (
            f"These files are {len(self.groups)} kinds of measurement collected "
            f"across several files each. First the files of each kind are stacked "
            f"("
            + "; ".join(f"{' + '.join(g.members)} → {g.label}" for g in stacked)
            + f"), then the {len(self.groups)} results are linked by a shared ID."
        )


def _column_overlap(a: pd.DataFrame, b: pd.DataFrame) -> float:
    """Jaccard overlap of two frames' column names."""
    sa, sb = set(map(str, a.columns)), set(map(str, b.columns))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# Names researchers actually give the column that identifies a participant.
_ID_NAME_TOKENS = ("id", "seqn", "subject", "participant", "patient", "pid",
                   "mrn", "record", "respondent", "person", "case", "study_id",
                   "usubjid", "sid", "uid")


def _looks_like_an_id_name(col: Any) -> bool:
    """Does this column's NAME claim to identify a person?"""
    import re
    name = re.sub(r"[^a-z0-9]+", "", str(col).lower())
    return any(tok.replace("_", "") in name for tok in _ID_NAME_TOKENS)


def _same_people(a: pd.DataFrame, b: pd.DataFrame) -> Optional[bool]:
    """Do these two files describe the same people? None if it cannot be told.

    Column names are a weak signal for this and were the only one being used.
    Two survey cycles that both gained a column sit at 0.67 overlap, under the
    0.8 threshold, and were classed as "different measurements on the same
    people" — so the app proposed linking two cycles that share no participants
    at all.

    The strong signal is the ID VALUES. demo_2017 and demo_2019 both have a
    SEQN column and not one participant in common: different people, stack
    them. demographics and labs have the same SEQN values: same people, link
    them. That distinction is what the researcher is actually being asked, and
    it is right there in the data.
    """
    from ml.join_doctor import normalize_key

    shared = [c for c in a.columns if str(c) in {str(x) for x in b.columns}]
    # When two shared columns disagree — a disjoint SEQN and a coincidentally
    # identical age — believe the one that is named like an identifier. Without
    # this, any unique overlapping column outvotes the real ID and two cycles
    # are declared to be about the same participants.
    id_named = [c for c in shared if _looks_like_an_id_name(c)]
    if id_named:
        shared = id_named
    best: Optional[float] = None
    for col in shared:
        try:
            av = set(normalize_key(a[col]).dropna().unique())
            bv = set(normalize_key(b[col]).dropna().unique())
        except Exception:
            continue
        if len(av) < 5 or len(bv) < 5:
            continue
        # Only a near-unique column can answer "same people". The bar is high
        # deliberately: at anything looser, `age` qualifies as an identifier in
        # a small cohort, its values overlap between any two files, and every
        # pair of cycles is declared to be about the same participants.
        if len(av) < 0.95 * len(a) or len(bv) < 0.95 * len(b):
            continue
        overlap = len(av & bv) / max(1, min(len(av), len(bv)))
        best = overlap if best is None else max(best, overlap)
    if best is None:
        return None
    return best >= 0.5


def _group_label(members: List[str]) -> str:
    """A name for a stack group, taken from what its members have in common.

    demo_2017 + demo_2019 -> 'demo'. Falls back to listing them.
    """
    if len(members) == 1:
        return members[0]
    token_lists = [[t for t in _TOKEN_SPLIT.split(m) if t] for m in members]
    common: List[str] = []
    for parts in zip(*token_lists):
        if len(set(parts)) == 1:
            common.append(parts[0])
        else:
            break
    if common:
        return "_".join(common)
    return " + ".join(members[:2]) + ("…" if len(members) > 2 else "")


def group_files(frames: Dict[str, pd.DataFrame],
                threshold: float = 0.8) -> List[List[str]]:
    """Cluster files whose schemas match closely enough to be stacked.

    Union-find over pairwise column overlap, so a chain of near-identical
    cycles ends up in one group even when the first and last differ slightly.
    """
    names = [n for n, f in frames.items() if f is not None]
    parent = {n: n for n in names}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(names):
        for b in names[i + 1:]:
            cols = _column_overlap(frames[a], frames[b])
            same = _same_people(frames[a], frames[b])
            # Different people who were measured the same way belong together.
            # Schema drift between cycles is normal — one year adds a question —
            # so a shared identifier with no values in common outweighs a column
            # list that does not quite line up.
            if same is False and cols >= 0.4:
                stack_together = True
            elif same is True:
                stack_together = False          # same people: these LINK
            else:
                stack_together = cols >= threshold
            if stack_together:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb

    clusters: Dict[str, List[str]] = {}
    for n in names:
        clusters.setdefault(find(n), []).append(n)
    order = {n: i for i, n in enumerate(names)}
    out = [sorted(v, key=lambda m: order[m]) for v in clusters.values()]
    return sorted(out, key=lambda g: order[g[0]])


def plan_combination(frames: Dict[str, pd.DataFrame],
                     threshold: float = 0.8) -> CombinationPlan:
    """Propose how this whole set of files becomes one table.

    A hint, not a decision — the UI shows it and lets the user move files
    between groups before anything runs.
    """
    live = {n: f for n, f in frames.items() if f is not None}
    if len(live) < 2:
        only = list(live)
        return CombinationPlan(
            groups=[FileGroup(label=only[0], members=only)] if only else [],
            shape="single")

    clusters = group_files(live, threshold=threshold)
    groups = [FileGroup(label=_group_label(c), members=c) for c in clusters]

    if len(groups) == 1:
        shape = "stack"
    elif not any(g.is_stack for g in groups):
        shape = "link"
    else:
        shape = "stack_then_link"

    notes: List[str] = []
    if shape == "stack_then_link":
        notes.append(
            "Grouped by how much the files' columns overlap. If a file is in the "
            "wrong group, move it — nothing runs until you confirm.")
    return CombinationPlan(groups=groups, shape=shape, notes=notes)
