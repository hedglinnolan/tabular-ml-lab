"""A finding whose subject is the cohort rather than a column. `GUIDED-053`.

The product owner has ruled that the finding component must stretch to carry an
observation about the study rather than about a column, **because there is no
other way to present it**. This loop is scoped to the shape: what such a finding
renders as, and what stops it looking like a broken card with a greyed-out
button.

## We have solved half of this once already

`test_a_finding_with_no_repair_still_offers_something.py` is the precedent. A
finding with **no action** used to end in a Close button and three paragraphs —
`explainOnly`, a terminal branch at the one moment the user needed to act. The
fix was not to invent an action but to notice that *something* is always
offerable: an option with a preview, or an **earmark** with a destination —
*"this is yours to do"* — and the prose fallback underneath for a phrase the
table does not know.

So the no-action half has an answer, and this file does not re-solve it. What is
new is **no subject**.

## Why an empty column list is not just an empty column list

Every finding card in the app answers *what is this about?* with a row of mono
column chips. A cohort finding has none, and the failure mode is specific and
worth naming: the chips row renders empty, the card keeps its full frame, and
the result reads as **a card that failed to load** — the exact impression
`GUIDED-006` is about, where an interface asserts a capability it does not have.
An empty space where a name belongs is read as a missing name, never as *there
is no name*.

The rule this file adds is therefore the recorded-absence rule (§09) applied to
a finding's subject:

> **A finding states its subject. Where the subject is not a column, it says
> what it is instead — never nothing.**

Three scopes, and the third is the one being added:

| scope | subject | rendered as |
|---|---|---|
| `columns` | named columns | mono chips, as today |
| `rows` | named rows | a count and a copyable list (`DRIVE-007`) |
| `cohort` | the study itself | a stated scope line, no chips |

## What this loop deliberately does not build

The resolution statement. `PRODUCT_VISION.md` marks it specified-unbuilt and it
stays that way — a cohort finding with no content to carry would be a frame
waiting for a subject, which is a different empty card. What is built is the
shape, so that when the statement arrives it has somewhere to land.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

COLUMNS = "columns"
ROWS = "rows"
COHORT = "cohort"
SCOPES = (COLUMNS, ROWS, COHORT)

# What the card says where the column chips would be. Not a label for an empty
# container — a statement, because the whole defect is that emptiness reads as
# failure.
SCOPE_LINE: Dict[str, str] = {
    COHORT: "About this study as a whole, not about any one column",
    ROWS: "About specific rows rather than a column",
}


class ScopeError(Exception):
    """A finding whose subject the app cannot state."""


def scope_of(finding: Dict[str, Any]) -> str:
    """What this finding is about, derived rather than stored.

    Derived, because the alternative is a field that has to be set correctly on
    every producer — including two frozen modules — and a finding whose scope
    was never set would default to `columns` and render the empty chip row this
    exists to prevent. A rule read off the finding cannot be forgotten by a
    producer that does not know about it.
    """
    params = finding.get("params") or {}
    if finding.get("scope") in SCOPES:
        return str(finding["scope"])
    if finding.get("affected_columns") or params.get("columns"):
        return COLUMNS
    if params.get("rows") or params.get("row_labels"):
        return ROWS
    return COHORT


def subject_line(finding: Dict[str, Any]) -> str:
    """The sentence that replaces the chips when there are no chips.

    Always non-empty. A card that says nothing about its subject is the broken
    card this module exists to prevent, so the fallback is a statement rather
    than a blank.
    """
    scope = scope_of(finding)
    if scope == COLUMNS:
        columns = [str(c) for c in (finding.get("affected_columns")
                                    or (finding.get("params") or {}).get("columns")
                                    or [])]
        return ", ".join(columns)
    if scope == ROWS:
        rows = ((finding.get("params") or {}).get("rows")
                or (finding.get("params") or {}).get("row_labels") or [])
        return f"{len(rows):,} row(s)"
    return SCOPE_LINE[COHORT]


def render_shape(finding: Dict[str, Any]) -> Dict[str, Any]:
    """Everything the card needs to avoid looking broken, in one object.

    `has_chips` is what the page branches on. It is computed here rather than in
    the page for the reason every other server-side classification in this
    project is: a page that decided for itself which findings have subjects
    would hold a second copy of the rule.
    """
    scope = scope_of(finding)
    return {
        "scope": scope,
        "subject_line": subject_line(finding),
        "has_chips": scope == COLUMNS,
        # THE OTHER HALF, and it is the precedent's rather than new: a finding
        # with no repair still offers something. Carried here so the page has
        # one place to ask both questions.
        "has_repair": str(finding.get("fix_kind") or "none") not in ("none", ""),
    }


def check_subject_survives(finding: Dict[str, Any],
                           present: Sequence[str]) -> Optional[str]:
    """Whether a column-scoped finding still names a column that exists.

    `ml/import_doctor.py`'s apply path filters a finding's columns against the
    frame — `[c for c in p.get("columns", []) if c in df.columns]` — and when
    the intersection is empty it drops nothing and reports having dropped
    nothing, which is a repair that silently succeeds at doing nothing.

    That module is frozen, so this is the reading the Guided door does BEFORE
    offering the repair. Returns the refusal sentence, or `None` when the
    finding still has a subject.
    """
    if scope_of(finding) != COLUMNS:
        return None
    named = [str(c) for c in (finding.get("affected_columns")
                              or (finding.get("params") or {}).get("columns")
                              or [])]
    live = [c for c in named if c in set(present)]
    if live:
        return None
    return (f"This finding is about {', '.join(f'`{c}`' for c in named[:4])}, "
            f"and {'that column is' if len(named) == 1 else 'none of those columns are'} "
            f"in the table any more — an earlier repair renamed or removed "
            f"{'it' if len(named) == 1 else 'them'}. Applying it now would "
            f"change nothing and report success, so it is withdrawn rather "
            f"than offered.")
