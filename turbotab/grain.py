"""turbotab.grain — the question the seal cannot be drawn without.

Constitution §02 (`ROADMAP.md`) and `ASSEMBLY_SPEC.md` §05:

    "Can one person appear in more than one row?"

Asked **once**, pre-seal, and recorded once. Both consumers read the one
recorded answer: a project arriving through multi-file assembly has already
answered it and the seal inherits it; a single-file project is asked directly.

`IMPORT-020` and `IMPORT-022` exist because the app *guessed* at this instead of
asking, and a failed guess rendered as a clean lock over a real leak. So the
heuristics are demoted here from source of truth to two lesser roles:

* a **suggestion**, offered under "yes, people repeat" — never the answer;
* a **contradiction detector**, when the stated answer disagrees with the shape
  of the data.

**The contradiction detector is deliberately name-blind, and that is the whole
point of this module.** `rank_grouping_candidates` gates on a name-token list,
and `IMPORT-022` is precisely that gate failing: a column called ``SUBJ``
holding 60 values across 180 rows is rejected before its repetition is ever
measured, because ``subj`` is not among ``_SUBJECT_ID_TOKENS``. Building the
contradiction detector on the same heuristic would inherit the same blind spot
and the interruption would not fire on the one dataset it was written for. So
`repetition_evidence` below looks at **shape only** — how often values repeat,
and how regularly — and never at what a column is called.

Two name lists have already failed on ``SUBJ`` in this codebase, for different
purposes: `_SUBJECT_ID_TOKENS` here and `_ID_NAME_HINT` in `ml/join_doctor.py`
(`IMPORT-219`). Constitution §02 says in as many words that name lists cannot
close this and must not be tuned as though they could. A third list would be the
same mistake spelled differently.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from utils.test_lockbox import (
    BASIS_INHERITED, BASIS_USER_STATED,
    SEAL_CROSS_SECTIONAL, SEAL_GROUPED, SEAL_UNDETERMINED,
    _MAX_ROWS_PER_GROUP, _MIN_GROUPS_FOR_GROUPED_LOCKBOX,
    rank_grouping_candidates,
)

# The three answers. `NOT_SURE` is first-class, not a refusal to answer: a
# researcher who genuinely does not know their data's shape gets honest numbers
# with exploratory labeling, never a locked door (constitution §03, and the
# asymmetry that settled it — `IMPORT-021` leaks too and closes anyway, because
# it says so).
ONE_ROW_PER_PERSON = "one_row_per_person"
PEOPLE_REPEAT = "people_repeat"
NOT_SURE = "not_sure"
ANSWERS = (ONE_ROW_PER_PERSON, PEOPLE_REPEAT, NOT_SURE)

# A column is identifier-SHAPED when its values repeat a handful of times each.
# Below the floor it is a category ('sex' repeats hundreds of times); above the
# ceiling it is a many-to-many merge product, which repeats harder rather than
# less. Both bounds are the lockbox's own, imported rather than re-chosen, so
# the interruption and the seal cannot disagree about what "repeats" means.
_MIN_REPEATS = 2
_MAX_REPEATS = _MAX_ROWS_PER_GROUP
# Fewer distinct values than this and a repeating column is a stratum, not a
# roster. Same constant the grouped split needs to be meaningful at all.
_MIN_DISTINCT = _MIN_GROUPS_FOR_GROUPED_LOCKBOX


def _column_repetition(s: pd.Series, n_rows: int) -> Optional[Dict[str, Any]]:
    """Shape facts about one column, or None when it cannot be a roster.

    Name-blind by construction: this function is never told what the column is
    called, so it cannot be fooled by a spelling no list anticipated.
    """
    try:
        n_distinct = int(s.nunique(dropna=True))
    except TypeError:
        return None                                  # unhashable cells
    if n_distinct < _MIN_DISTINCT or n_distinct >= n_rows:
        return None                                  # a stratum, or already unique
    non_null = int(s.notna().sum())
    if not non_null:
        return None
    rows_per = non_null / n_distinct
    if rows_per < _MIN_REPEATS or rows_per > _MAX_REPEATS:
        return None
    try:
        sizes = s.dropna().value_counts()
    except Exception:
        return None
    # Regularity separates a roster from a coincidence. Three visits per person
    # is a study design; a free-text field that happens to average 2.4 uses is
    # not. Measured as the share of values whose count equals the modal count.
    modal = int(sizes.mode().iloc[0]) if len(sizes) else 0
    regular = float((sizes == modal).mean()) if len(sizes) else 0.0
    return {
        "column": str(s.name),
        "n_distinct": n_distinct,
        "n_rows": non_null,
        "rows_per": round(rows_per, 2),
        "modal_rows_per": modal,
        "regular_share": round(regular, 3),
    }


def repetition_evidence(df: pd.DataFrame, max_columns: int = 200
                        ) -> List[Dict[str, Any]]:
    """Every column whose SHAPE says values repeat like a roster of people.

    Ranked by regularity then by how many rows each value carries. This is the
    evidence the contradiction detector escalates on, and it is the reason the
    detector survives an identifier the name lists do not recognize.
    """
    if df is None or df.empty:
        return []
    n_rows = len(df)
    out: List[Dict[str, Any]] = []
    for col in list(df.columns)[:max_columns]:
        s = df[col]
        if isinstance(s, pd.DataFrame):              # duplicated column label
            continue
        # A float column is a measurement, not a roster. Integers are allowed:
        # SEQN is an integer and so is every study ID that was ever exported
        # from SAS.
        if pd.api.types.is_float_dtype(s):
            continue
        if pd.api.types.is_bool_dtype(s):
            continue
        ev = _column_repetition(s, n_rows)
        if ev:
            out.append(ev)
    # Many distinct values each repeating a FEW times is a roster; few values
    # each repeating many times is a stratum. So rank by distinct count, not by
    # repeat count -- sorting the other way puts `age` (30 values x 6 rows)
    # ahead of `SUBJ` (60 values x 3 rows) on a 60-subject longitudinal file,
    # which is the wrong column to offer first.
    out.sort(key=lambda e: (-e["regular_share"], -e["n_distinct"], e["column"]))
    return out


def suggestion(df: pd.DataFrame) -> Dict[str, Any]:
    """What to offer under "yes, people repeat" — a suggestion, never an answer.

    The name heuristic ranks first because when it fires it is usually right and
    its ordering is good (`rank_grouping_candidates` prefers the person over a
    finer barcode and over a coarser site). Shape-only candidates it missed are
    appended, which is where an unrecognized spelling like ``SUBJ`` comes back.
    """
    if df is None or df.empty:
        return {"columns": [], "evidence": []}
    try:
        ranked = rank_grouping_candidates(df)
    except Exception:
        ranked = []
    named = [str(c.get("column")) for c in ranked if c.get("column") is not None]
    evidence = repetition_evidence(df)
    shaped = [e["column"] for e in evidence if e["column"] not in named]
    return {
        # The order the picker offers. Named candidates first, then columns the
        # name list did not recognize but whose shape says otherwise.
        "columns": named + shaped,
        "from_name_heuristic": named,
        "from_shape_only": shaped,
        "evidence": evidence,
    }


# ── the two terminal exits ───────────────────────────────────────────────────
# `DESIGN_LANGUAGE.md` §09: a CONSEQUENCE **resolves or is attested, never a
# dead end.** The attestation exit already existed mechanically — `set_grain`
# takes `acknowledged_contradiction` and the API reads it — but it carried NO
# LABEL, so a frontend author reading the 409 would have known only to offer
# "change your answer". An exit nobody can find is not an exit, and a user
# whose data is genuinely unusual would have been stuck at a question they had
# answered correctly.
#
# Both are carried on the refusal itself so the interface cannot render one
# without the other.
_RESOLVE = {
    "id": "revise",
    "kind": "resolve",
    "label": "Change my answer",
    "detail": "Go back to the question and answer it differently.",
}


def _attest(what: str) -> Dict[str, str]:
    return {
        "id": "attest",
        "kind": "attest",
        "label": "My answer is right — the data really is like this",
        "detail": what,
    }


_EXITS_STATED_UNIQUE = [
    _RESOLVE,
    _attest("Continue with one row per person. The repetition is recorded as "
            "a noted disagreement, and it travels into the methods section as "
            "a stated limitation rather than disappearing."),
]

_EXITS_STATED_REPEATS = [
    _RESOLVE,
    _attest("Continue with this column as the identifier. The disagreement is "
            "recorded and carried into the methods section as a stated "
            "limitation."),
]


def contradiction(df: pd.DataFrame, answer: str,
                  group_col: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Evidence that the stated answer and the data disagree, or None.

    **Escalate on evidence of error, never on the magnitude of a consequence**
    (`ASSEMBLY_SPEC.md` §04, and the same rule that governs join drops). This
    returns something when one of the two — the user or the file — is *wrong*,
    and it does not care how large the resulting leak would be. A 3-row leak and
    a 300-row leak earn the same interruption, because the reason to interrupt
    is the contradiction, not its size.
    """
    if df is None or df.empty:
        return None

    if answer == ONE_ROW_PER_PERSON:
        evidence = [e for e in repetition_evidence(df) if e["regular_share"] >= 0.5]
        if not evidence:
            return None
        top = evidence[0]
        return {
            "kind": "stated_unique_but_data_repeats",
            "columns": [e["column"] for e in evidence[:3]],
            "evidence": evidence[:3],
            # Leads with the OBSERVATION, not with the user's claim. The first
            # draft opened "You said one row per person, but…", which attributes
            # a position to the user and then contradicts it — a courtroom
            # cadence for a message whose whole point is that either reading
            # could be the wrong one. Same facts, same refusal to assign blame.
            "message": (
                f"`{top['column']}` has {top['n_distinct']:,} distinct values "
                f"across {top['n_rows']:,} rows, about {top['rows_per']:g} each. "
                f"That is the shape of repeated measures, and you answered one "
                f"row per person. One of those two readings is wrong, and which "
                f"one changes how the held-out rows are chosen."),
            "exits": _EXITS_STATED_UNIQUE,
        }

    if answer == PEOPLE_REPEAT and group_col:
        if group_col not in df.columns:
            return {
                "kind": "named_column_absent",
                "columns": [group_col],
                "evidence": [],
                "message": (f"`{group_col}` is not a column in this table, so the "
                            f"held-out rows cannot be grouped by it."),
                # Only one exit here, and that is correct rather than a dead
                # end: a column that does not exist cannot be attested to. The
                # single exit is still a resolution, so §09 holds.
                "exits": [_RESOLVE],
            }
        s = df[group_col]
        if isinstance(s, pd.DataFrame):
            return None
        n_distinct = int(s.nunique(dropna=True))
        non_null = int(s.notna().sum())
        if n_distinct >= non_null and non_null:
            return {
                "kind": "stated_repeats_but_column_is_unique",
                "columns": [group_col],
                "evidence": [{"column": group_col, "n_distinct": n_distinct,
                              "n_rows": non_null, "rows_per": 1.0}],
                "message": (
                    f"You said people repeat, but '{group_col}' has a different "
                    f"value on every one of its {non_null:,} rows. Grouping by it "
                    f"would hold out one row per group, which is the row-level "
                    f"split you were trying to avoid."),
                "exits": _EXITS_STATED_REPEATS,
            }
    return None


# ── what the user reads after answering, and after the seal ──────────────────
# These did not exist. `seal_basis` returned `undetermined` and `draw_holdout`
# set `exploratory: True`, and NOTHING said anything — so a user who chose
# "I'm not sure" got a seal that rendered, as far as the interface was
# concerned, exactly like a confident one. That is constitution §03's own
# failure ("never rendered as a clean lock") reproduced in the door built to
# honor it.
#
# Written as prose the user reads, not as a flag a renderer might act on,
# because a flag is a thing somebody has to remember to check and a sentence is
# not.

_ANSWERED: Dict[str, str] = {
    ONE_ROW_PER_PERSON:
        "Recorded: one row per person. The held-out rows will be drawn at "
        "random, which is the right choice when every row is a different "
        "participant.",
    PEOPLE_REPEAT:
        "Recorded: people repeat, identified by `{group_col}`. Whole people "
        "will be held out rather than individual rows, so nobody appears on "
        "both sides of the split.",
    NOT_SURE:
        "Recorded: unknown. That is a legitimate answer and the analysis "
        "continues — but because the shape is unknown, the held-out rows are "
        "drawn by row, and if your rows do repeat people the same person will "
        "sit on both sides. Held-out performance would then read better than "
        "the model is. Your numbers will be labeled exploratory until this is "
        "settled, and you can settle it at any point before training.",
}

_SEALED: Dict[str, str] = {
    SEAL_CROSS_SECTIONAL:
        "{n_test:,} rows ({fraction:.0%}) are held out and will not be looked "
        "at again until the models are scored.",
    SEAL_GROUPED:
        "{n_test:,} rows ({fraction:.0%}) from {n_test_groups:,} "
        "{group_noun} are held out, chosen by {group_one} rather than by row, "
        "so no {group_one} appears in both halves.",
    SEAL_UNDETERMINED:
        "{n_test:,} rows ({fraction:.0%}) are held out, drawn BY ROW because "
        "the data's shape is unknown. This is not a verified clean split: if "
        "rows repeat people, the same person is on both sides and held-out "
        "performance will read better than the model is. Treat these numbers "
        "as exploratory, and answer the grain question when you can.",
    "repetition_found_grouping_abandoned":
        "{n_test:,} rows ({fraction:.0%}) are held out, drawn BY ROW. Rows do "
        "repeat per {group_one}, but there are too few {group_noun} to hold "
        "any out whole — so the same {group_one} can appear on both sides and "
        "held-out performance will read better than the model is. Treat these "
        "numbers as exploratory.",
}


def answer_disclosure(answer: str, group_col: Optional[str] = None) -> str:
    """What the user reads immediately after answering the grain question."""
    return _ANSWERED.get(answer, "").format(group_col=group_col or "")


def seal_disclosure(lockbox: Dict[str, Any]) -> str:
    """What the user reads once the seal is drawn.

    Keyed on the RECORDED BASIS rather than on a flag, so the three states
    constitution §03 insists on stay three different sentences. An undetermined
    seal and a verified cross-sectional one must never render alike; here they
    cannot, because they are not the same string.
    """
    basis = lockbox.get("seal_basis") or SEAL_UNDETERMINED
    noun = lockbox.get("group_noun") or "subjects"
    one = noun[:-1] if noun.endswith("s") else noun
    try:
        return _SEALED.get(basis, _SEALED[SEAL_UNDETERMINED]).format(
            n_test=lockbox.get("n_test", 0),
            fraction=float(lockbox.get("fraction", 0.0)),
            n_test_groups=lockbox.get("n_test_groups") or 0,
            group_noun=noun, group_one=one)
    except (KeyError, ValueError):            # pragma: no cover - formatting guard
        return _SEALED[SEAL_UNDETERMINED].format(
            n_test=lockbox.get("n_test", 0),
            fraction=float(lockbox.get("fraction", 0.0)))


def is_exploratory_basis(basis: Optional[str]) -> bool:
    """Which bases carry exploratory labeling. Two of the four, not one."""
    return basis in (SEAL_UNDETERMINED, "repetition_found_grouping_abandoned")


def seal_basis(answer: str, group_col: Optional[str] = None,
               n_groups: Optional[int] = None) -> str:
    """Which of the four bases a stated answer produces.

    `undetermined` is what "I'm not sure" means, and it is first-class: never
    `group_col: None`, which a consumer cannot tell from a verified
    cross-sectional seal (constitution §03).
    """
    if answer == NOT_SURE:
        return SEAL_UNDETERMINED
    if answer == ONE_ROW_PER_PERSON:
        return SEAL_CROSS_SECTIONAL
    if answer == PEOPLE_REPEAT and group_col:
        if n_groups is not None and n_groups < _MIN_GROUPS_FOR_GROUPED_LOCKBOX:
            # Repetition was stated AND believed; there are simply too few
            # people to hold any out by person. That is the third basis, and it
            # is not the same claim as "no repetition".
            return "repetition_found_grouping_abandoned"
        return SEAL_GROUPED
    # "people repeat" with no column named yet: the shape is known and the
    # grouping is not, which is undetermined rather than cross-sectional.
    return SEAL_UNDETERMINED


def basis_source(inherited: bool = False) -> str:
    """`user_stated`, or `inherited_from_assembly` once assembly ships.

    Both constants already exist in `utils/test_lockbox.py`, reserved by the
    L11 clause-03 work precisely so this lands without a schema change to a
    persisted, round-tripped artifact. `inherited_from_assembly` has no producer
    yet and is reachable rather than dead: assembly is behind an unmet freeze
    gate, and when it ships it sets this rather than adding a field.
    """
    return BASIS_INHERITED if inherited else BASIS_USER_STATED
