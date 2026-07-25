"""Combining several files into one analysis table.

Researchers arrive with files that relate in exactly two ways, and the app
previously supported only one of them:

  STACK  — the same measurements on different people. NHANES cycles
           1999-2000, 2001-2002, …; sites in a multi-centre study; years of a
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
    """Columns the modelling feature pool must never offer as predictors."""
    return [SOURCE_COLUMN]
