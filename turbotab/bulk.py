"""turbotab.bulk — one decision over a set defined by a rule, not N decisions.

`GUIDED-029`, and it is filed `critical` for a reason the L20 matrix made
unarguable: `metabolomics_untargeted.csv` has 308 columns with blanks, so the
interview asked **313 questions before any lens** — roughly ten times the ~32
this project calls Classic's indictment. The metabolomics pack rescued it to 6.
A user with the same table who answered *"something else, or not sure"* still
got 313, and so does any wide table in a field no pack covers.

**The lens was masking an unscalable interview rather than accelerating a
scalable one.** That distinction is the whole finding: a benefit measured
against a broken baseline is a number flattering itself.

## The remedy, which was already specified

From the p ≫ n work: **operations apply to sets defined by a rule, and the user
edits the rule rather than the members.** The interview says *"294 numeric
columns have blanks. Answer once for all of them, or review them individually."*

Four properties, each of which the design would be wrong without:

* **Grouped by dtype**, because clause §07 already routes by dtype and the two
  branches take different strategies. One blanket answer across numeric and
  categorical would be a bulk affordance that had to be wrong for one of them.
* **The group is what REMAINS after pack priors settle their columns.** A bulk
  affordance offered over one leftover column is worse than asking, so a group
  of one is asked individually and no rule is invented for it.
* **One decision covering N columns, not N decisions.** The record carries the
  rule and its members, and the methods sentence reads *"missing values in 294
  numeric columns were imputed with the training-fold median"* — which is also
  the sentence a reader wants.
* **Bulk plus evidence-driven exceptions.** A single answer across 294 columns
  is not always true. Where the evidence disagrees with the bulk answer the
  columns are surfaced — and by the same escalation rule as everywhere else:
  *evidence that a reading is wrong, never the size of the consequence.*

## Why the exceptions are one question and not N

Because otherwise this module reintroduces the defect it exists to remove. If
500 columns disagree, 500 questions is the same unbounded interview arriving
through the back door. The exceptions are a **second group with its own rule**,
named and countable, and answering it is one more decision.

## The scaling claim, and how it is checked

The number of pushed questions is **O(1) in the column count**, not O(p).
`test_the_interview_is_the_same_size_at_twelve_columns_and_twelve_thousand`
asserts that at both ends, because a bulk affordance tested only on the wide
case can hide a threshold that makes the narrow case worse.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Below this a group is asked column by column. A "rule" over one column is not
# a rule, it is a column with extra words in front of it — and the brief for
# this module says so: a bulk affordance offered over one leftover column is
# worse than asking.
MIN_GROUP = 2

# The strength at which missingness is taken to carry signal about the outcome.
# A difference in outcome rate (or in outcome mean, scaled) between the rows
# where a column is blank and the rows where it is not.
EXCEPTION_EFFECT = 0.20


@dataclass(frozen=True)
class Group:
    """A set of columns defined by a rule, and the rule in words.

    `members` is the answer to *which columns*, and it is recorded — but the
    user edits `rule`, never `members`. That distinction is what makes this
    scale: a rule is one sentence at any p, and a member list is not.
    """
    question: str                       # `missingness`
    branch: str                         # `numeric` | `categorical`
    members: Tuple[str, ...]
    settled: Tuple[str, ...] = ()       # columns a pack prior already settled
    excepted: Tuple[str, ...] = ()      # columns the user pulled out of the rule

    @property
    def key(self) -> str:
        return f"{self.question}_bulk::{self.branch}"

    @property
    def n(self) -> int:
        return len(self.members)

    @property
    def is_bulk(self) -> bool:
        return self.n >= MIN_GROUP

    @property
    def rule(self) -> str:
        """The rule, as the user reads and edits it."""
        noun = "numeric" if self.branch == "numeric" else "categorical"
        base = f"every {noun} column with blanks"
        clauses = []
        if self.settled:
            clauses.append(f"{len(self.settled):,} already settled by the lens")
        if self.excepted:
            clauses.append(f"{len(self.excepted):,} you pulled out")
        return base + (f", excluding {' and '.join(clauses)}" if clauses else "")

    def title(self) -> str:
        noun = "numeric" if self.branch == "numeric" else "categorical"
        return (f"{self.n:,} {noun} columns have blanks. Could a blank in them "
                f"mean something?")

    def to_dict(self) -> Dict[str, Any]:
        return {"question": self.question, "branch": self.branch,
                "key": self.key, "n": self.n, "rule": self.rule,
                "title": self.title(),
                "members": list(self.members), "settled": list(self.settled),
                "excepted": list(self.excepted), "is_bulk": self.is_bulk}


def group_columns(rows: Sequence[Dict[str, Any]], question: str = "missingness",
                  settled: Optional[Dict[str, Sequence[str]]] = None,
                  excepted: Sequence[str] = ()) -> List[Group]:
    """The groups this step asks about, one per dtype branch.

    `rows` is `missingness.survey()`'s output — the columns with blanks and the
    branch each is on. `settled` maps a branch to the columns a pack prior has
    already answered for; those leave the group, because a bulk question over
    columns that are not being asked about would state a count the user cannot
    reconcile with what they are being shown.
    """
    settled = {k: set(v) for k, v in (settled or {}).items()}
    out_of = set(excepted)
    groups: List[Group] = []
    for branch in ("numeric", "categorical"):
        gone = settled.get(branch, set())
        members = tuple(sorted(
            r["column"] for r in rows
            if r["branch"] == branch and r["column"] not in gone
            and r["column"] not in out_of and not r.get("answered")))
        if not members:
            continue
        groups.append(Group(question=question, branch=branch, members=members,
                            settled=tuple(sorted(gone)),
                            excepted=tuple(sorted(out_of & {
                                r["column"] for r in rows
                                if r["branch"] == branch}))))
    return groups


def settled_groups(rows: Sequence[Dict[str, Any]],
                   priors_by_column: Dict[str, List[Dict[str, Any]]],
                   question: str = "missingness") -> List[Dict[str, Any]]:
    """The columns a pack prior settled, grouped so the SKIP scales too.

    **A rendered skip is still a rendered thing.** Wiring the priors layer at
    L20 turned 306 questions into 306 skips, which is a real improvement in what
    is being asked and no improvement at all in what is being drawn —
    `DESIGN_LANGUAGE.md` §09 wants skips to group *"so their density reads as
    machine work at a glance"*, and 306 of them is not a glance.

    So the same rule applies to the skip: one stated fact over a set, named and
    countable, rather than one per column. Grouped by branch and by the prior
    that settled them, because two packs settling different columns for
    different reasons are two facts and collapsing them would state neither.
    """
    buckets: Dict[Tuple[str, str, str], List[str]] = {}
    reasons: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for row in rows:
        priors = list(priors_by_column.get(row["column"]) or [])
        if len(priors) != 1 or priors[0].get("marker") != "derived":
            continue
        prior = priors[0]
        key = (row["branch"], prior["pack"], str(prior.get("mechanism") or ""))
        buckets.setdefault(key, []).append(row["column"])
        reasons[key] = prior
    out = []
    for (branch, pack, mechanism), columns in sorted(buckets.items()):
        if len(columns) < MIN_GROUP:
            continue
        prior = reasons[(branch, pack, mechanism)]
        out.append({
            "key": f"{question}_settled::{branch}::{pack}",
            "branch": branch, "pack": pack, "label": prior["label"],
            "columns": sorted(columns), "n": len(columns),
            "mechanism": mechanism, "reason": prior["reason"],
            # The badge travels with the group. A prior that settles 306
            # columns is the single loudest claim the pack makes on this table,
            # so it is the last one that should reach the user unbadged.
            "evidence_status": prior.get("evidence_status"),
            "source": prior.get("source"),
            "title": (f"{len(columns):,} {branch} columns were settled by the "
                      f"{prior['label'].lower()} lens"),
        })
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Evidence-driven exceptions
# ─────────────────────────────────────────────────────────────────────────────

def _standard_error(n_blank: int, n_present: int) -> float:
    """Roughly how large a difference this comparison produces by chance.

    A blank-versus-present comparison on 8 rows and 64 is a noisy comparison,
    and a fixed effect threshold treats it exactly like one on 4,000 and 4,000.
    """
    return float(np.sqrt(1.0 / max(n_blank, 1) + 1.0 / max(n_present, 1))) * 0.5


def _association_with_outcome(df: pd.DataFrame, column: str,
                              target: str) -> Optional[float]:
    """How differently the outcome behaves where this column is blank.

    A rate difference for a binary outcome, a standardized mean difference for a
    continuous one. `None` when there is nothing to compare — fewer than a
    handful of rows on either side, or an outcome with no variation.

    Deliberately one number and a plain one. A p-value across 294 columns is a
    multiple-testing problem the app would then have to explain; an effect size
    is a thing the user can look at beside the column.
    """
    if target not in df.columns or column not in df.columns:
        return None
    blank = df[column].isna()
    n_blank, n_present = int(blank.sum()), int((~blank).sum())
    if n_blank < 5 or n_present < 5:
        return None
    y = df[target]
    if y.isna().all():
        return None
    if pd.api.types.is_numeric_dtype(y) and y.nunique(dropna=True) > 2:
        a, b = y[blank].dropna(), y[~blank].dropna()
        if len(a) < 5 or len(b) < 5:
            return None
        pooled = float(y.std())
        if not pooled or not np.isfinite(pooled):
            return None
        return float(abs(a.mean() - b.mean()) / pooled)
    # Binary or categorical: the share of the most common level, either side.
    try:
        top = y.dropna().mode().iloc[0]
    except (IndexError, KeyError):
        return None
    a = y[blank].dropna()
    b = y[~blank].dropna()
    if len(a) < 5 or len(b) < 5:
        return None
    return float(abs((a == top).mean() - (b == top).mean()))


def exceptions(df: pd.DataFrame, group: Group, mechanism: str,
               target: Optional[str],
               threshold: float = EXCEPTION_EFFECT) -> Dict[str, Any]:
    """Columns in this group where the evidence disagrees with the bulk answer.

    **The direction that matters is `not_informative`.** A user who says a blank
    means nothing, over a column where the outcome behaves differently wherever
    that column is blank, has said something the data contradicts — and the
    consequence is a column that carried signal getting a median written over
    it by a well-meaning default.

    The other direction is deliberately NOT reported. *"You said this is
    informative and we see no association"* is an absence of evidence, and
    escalating on one would be the app arguing with a user about a claim it
    cannot check — the escalation rule is *evidence that a reading is wrong*,
    and no association is not that.

    Returns the columns, their effect sizes, and one sentence. A group again,
    because 500 exceptions asked one at a time is the unbounded interview
    arriving through the back door.
    """
    if mechanism != "not_informative" or not target or target not in df.columns:
        return {"columns": [], "evidence": {}, "sentence": ""}

    # THE FLOOR RISES WITH THE NOISE, and this is not a refinement — the first
    # version flagged 31 of 306 columns on `metabolomics_untargeted.csv`, which
    # is almost exactly the false-positive rate for a rate difference on 72
    # participants at a fixed 0.20 threshold. An exceptions question listing 31
    # columns none of which is real teaches the user to dismiss the next one,
    # which is the blocker-budget argument arriving at a different card.
    #
    # Two corrections, both crude on purpose. A comparison on 8 blanks and 64
    # present is noisier than one on 4,000 and 4,000, so the threshold scales
    # with the standard error; and testing 306 columns finds a 1-in-20 event
    # fifteen times, so it scales with the count as well. Neither is a p-value:
    # a multiple-testing correction the app would then have to explain is worse
    # than an effect size the user can look at beside the column.
    multiplicity = float(np.sqrt(np.log(max(len(group.members), 2))))
    hits: List[Tuple[str, float]] = []
    for column in group.members:
        effect = _association_with_outcome(df, column, target)
        if effect is None:
            continue
        blank = df[column].isna()
        floor = max(threshold,
                    2.0 * multiplicity * _standard_error(int(blank.sum()),
                                                         int((~blank).sum())))
        if effect >= floor:
            hits.append((column, round(effect, 3)))
    hits.sort(key=lambda p: -p[1])
    if not hits:
        return {"columns": [], "evidence": {}, "sentence": ""}

    worst = hits[0]
    return {
        "columns": [c for c, _ in hits],
        "evidence": dict(hits),
        "threshold": threshold,
        "sentence": (
            f"{len(hits):,} of these {group.n:,} columns look like exceptions. "
            f"You said a blank means nothing, and in `{worst[0]}` the outcome "
            f"behaves differently wherever it is blank — a difference of "
            f"{worst[1]:.0%} against the rest. That is not proof the blank "
            f"means something, and it is a disagreement worth looking at "
            f"before a median is written over it."),
    }


# ─────────────────────────────────────────────────────────────────────────────
# The record
# ─────────────────────────────────────────────────────────────────────────────

def receipt(group: Group, mechanism: str, strategy_key: str,
            label: str, defers: bool) -> str:
    """The methods sentence for one decision over N columns.

    Reads as prose about the study rather than about the software, and it is
    the sentence a reader wants: *"missing values in 294 numeric columns were
    imputed with the training-fold median."* N sentences saying the same thing
    about one column each is not a methods section, it is a log.
    """
    noun = "numeric" if group.branch == "numeric" else "categorical"
    where = " within each training fold" if defers else ""
    how = label.lower().replace("fill with ", "").replace("fill by ", "")
    if strategy_key == "leave":
        return (f"Missing values in {group.n:,} {noun} column(s) are left as "
                f"they are; no imputation is applied and none is scheduled.")
    if strategy_key == "explicit_category":
        return (f"Missing values in {group.n:,} {noun} column(s) are kept as an "
                f"explicit `Missing` category rather than filled.")
    if strategy_key == "indicator":
        return (f"A was-it-missing indicator is added for {group.n:,} {noun} "
                f"column(s); the underlying values are left blank.")
    return (f"Missing values in {group.n:,} {noun} column(s) will be filled "
            f"using {how}{where}.")
