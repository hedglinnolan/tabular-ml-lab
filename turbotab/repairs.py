"""turbotab.repairs — one preview, a selectable set, one apply, one decision.

`DRIVE-002`. Nine NHANES features are binary written as text. The engine found
all nine and the driver had to open, read, preview and apply **nine times** for
one idea. The product owner's remedy is the specification:

> Show what it means for one, then let the user select which features to run it
> on, then apply to the selected set.

## This is `bulk.py`'s rule-scope, pointed at repairs

`turbotab/bulk.py` already established the principle for *questions*:
**operations apply to sets defined by a rule, and the user edits the rule rather
than the members.** It was built for the missingness question, where 308 columns
with blanks produced 308 questions. A repair is the same shape one object over —
nine columns, one idea, nine cycles — and it wants the same treatment rather than
a second mechanism beside it.

Two things are deliberately NOT inherited, because a repair is not a question.

**The set is editable member by member here, and that is not a contradiction of
the rule-scope principle.** A missingness rule ranges over hundreds of columns
where nobody can hold the membership in their head, so editing members is the
thing that does not scale. A repair group is small by construction — it is the
columns one detector claimed — and *"`sex` is binary text but I want it left
alone"* is a real and common judgment. So the members are checkboxes, and the
rule is still what the record carries.

**Every member is previewed, not just the representative.** The card SHOWS one
worked example because nine before/after tables are unreadable, but the apply is
per-column and each column's own diff is what runs. A preview of one column
presented as a preview of nine would be the blind consent the preview exists to
end.

## Why grouping is by `fix_kind` and not by title

Titles carry the column name — *"'sex' is a binary variable written as text"* —
so grouping on them groups nothing. `fix_kind` is the engine's own word for
*which repair this is*, it is what `apply_fix` dispatches on, and two findings
with one `fix_kind` are two instances of one operation by construction rather
than by resemblance.

## What stays out of a group

Anything the engine refuses to repair (`fix_kind == "none"`), which is already
`router._is_repairable`'s line; and `set_positive_class`, which needs an answer
the finding cannot supply and where each column's answer is a different research
question. A bulk affordance over those would be a single control standing in for
several unrelated decisions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Below this a repair is offered on its own card. One member is not a set, and a
# bulk affordance over a single column is a column with extra words in front of
# it — `bulk.MIN_GROUP`'s argument, and the same number for the same reason.
MIN_GROUP = 2

# Repairs that never group, with the reason each is excluded rather than a bare
# list. A name here is a claim that bulk would be WRONG, not merely awkward.
NEVER_GROUPED: Dict[str, str] = {
    "set_positive_class": (
        "which level is the event is the research question, and it is a "
        "different question for every column. One control standing in for "
        "several unrelated decisions is not a bulk affordance, it is a "
        "guess with a checkbox."),
    "melt_repeated": (
        "reshaping rebuilds what a row is, so two of them are not two "
        "instances of one operation — the second runs against a table the "
        "first replaced."),
    "promote_header": (
        "a table has one header row. Two findings proposing one are a "
        "disagreement to resolve, not a set to apply."),
    "drop_columns": (
        "the engine already reports these as one finding over N columns, so "
        "grouping them again would nest a set inside a set."),
}


# The OPERATION's name, for a card that is about N columns. The engine's own
# `fix_label` is an imperative for one button on one column, so it cannot title
# a group; these are the same operations said once. An unmapped kind falls
# through to the kind itself — a readable fallback, never an invented sentence.
KIND_LABELS: Dict[str, str] = {
    "read_as_binary": "read as binary",
    "coerce_numeric": "read as numbers",
    "recode_missing": "recode the missing-value codes as missing",
    "normalize_categories": "normalize the category spellings",
    "strip_whitespace": "strip surrounding whitespace",
    "parse_dates": "parse as dates",
}


@dataclass(frozen=True)
class RepairGroup:
    """N findings that are the same repair, and the words for saying so."""
    fix_kind: str
    findings: Tuple[Dict[str, Any], ...]

    @property
    def key(self) -> str:
        return f"repair_bulk::{self.fix_kind}"

    @property
    def n(self) -> int:
        return len(self.findings)

    @property
    def columns(self) -> List[str]:
        out: List[str] = []
        for f in self.findings:
            for c in (f.get("affected_columns") or []):
                if c not in out:
                    out.append(str(c))
        return out

    @property
    def label(self) -> str:
        """What the OPERATION is called, not what one column's button says.

        The engine's `fix_label` is per-column by design — it is an imperative
        for one button: *"Read 'batch' as binary (B2 = 1, B1 = 0)"*. Quoting it
        as the group's name would title a card about nine columns with the
        details of one, which is a sentence that is wrong about eight of them.

        So the group is named by its kind and each member keeps its own label in
        the member list. `KIND_LABELS` is a translation, not a second opinion:
        it renames nothing the engine decided and adds nothing the engine did
        not find, and an unmapped kind falls through to the kind itself rather
        than to a guess.
        """
        return KIND_LABELS.get(self.fix_kind, self.fix_kind.replace("_", " "))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "fix_kind": self.fix_kind,
            "label": self.label,
            "n": self.n,
            "columns": self.columns,
            "members": [
                {"id": f["id"], "title": f.get("title", ""),
                 "columns": [str(c) for c in (f.get("affected_columns") or [])],
                 "severity": f.get("severity"),
                 "confidence": f.get("confidence")}
                for f in self.findings],
            "severity": self.findings[0].get("severity"),
            "confidence": self.findings[0].get("confidence"),
        }


def group(findings: Sequence[Dict[str, Any]]) -> List[RepairGroup]:
    """Repairable findings that are the same operation, in the order ranked.

    Order is the caller's — the Router hands these in severity order — because a
    group's position in the interview is its most severe member's position, and
    re-sorting here would put a group of nine `info` repairs above one
    `critical`.
    """
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    order: List[str] = []
    for f in findings:
        kind = str(f.get("fix_kind") or "none")
        if kind in ("none", "") or kind in NEVER_GROUPED:
            continue
        if kind not in buckets:
            buckets[kind] = []
            order.append(kind)
        buckets[kind].append(dict(f))
    return [RepairGroup(fix_kind=k, findings=tuple(buckets[k]))
            for k in order if len(buckets[k]) >= MIN_GROUP]


def grouped_ids(findings: Sequence[Dict[str, Any]]) -> set:
    """Every finding id that a group has taken over.

    The Router uses this to withhold the individual questions. Derived from the
    same call that builds the groups, so a finding cannot be both in a group and
    asked on its own — which would be `GUIDED-040`'s duplicate-card defect
    reappearing one object over.
    """
    out = set()
    for g in group(findings):
        for f in g.findings:
            out.add(f["id"])
    return out


def encoding_phrase(encoding: Dict[str, Any]) -> str:
    """`` `Male` = 1, `Female` = 0 `` — the mapping, in the file's own spelling.

    `GUIDED-157`. Every spelling that maps to a side is named, not just the
    first: a column holding `Male` and `male` has two of them, and *"`Male` = 1"*
    over a column where half the positive rows say `male` is a sentence that is
    right about half the data.

    Backticked, and that is required rather than tidy. `devchecks.numbers_in`
    strips backticked spans before counting, so an unbackticked level carrying a
    digit — `T1`, `0`, `B2` — reads as an unsupported number and trips
    `a_decision_sentence_carries_a_number_its_payload_does_not`. The `1` and the
    `0` themselves are supported by the `mapping` the payload carries beside
    this.
    """
    pos = ", ".join(f"`{v}`" for v in (encoding.get("positive_values")
                                       or [encoding.get("positive")]))
    neg = ", ".join(f"`{v}`" for v in (encoding.get("negative_values")
                                       or [encoding.get("negative")]))
    return f"{pos} = 1, {neg} = 0"


def _named(columns: Sequence[str],
           encodings: Optional[Dict[str, Dict[str, Any]]] = None) -> Tuple[str, str]:
    """The column list for a sentence, and the separator it needs.

    A repair with no encoding to state reads exactly as it always did —
    ``` `a`, `b`, `c` ``` — because `read_as_binary` is the only kind that has a
    per-column mapping and giving every other kind a semicolon list would be
    punctuation invented for a distinction that does not exist there.
    """
    enc = encodings or {}
    if not any(str(c) in enc for c in columns):
        return ", ".join(f"`{c}`" for c in columns), ", "
    parts = []
    for c in columns:
        e = enc.get(str(c))
        parts.append(f"`{c}`: {encoding_phrase(e)}" if e else f"`{c}`")
    return "; ".join(parts), "; "


def sentence(label: str, columns: Sequence[str],
             declined: Sequence[str] = (),
             encodings: Optional[Dict[str, Dict[str, Any]]] = None) -> str:
    """The one sentence the record keeps for a bulk repair.

    It names the count, the operation and the columns — the columns, because a
    reader of the methods section has to be able to check it, and *"nine columns
    were converted"* without saying which is a claim nobody can verify.

    **And for a repair that rewrites values, it names what they became**
    (`GUIDED-157`). `read_as_binary` turns `gender ∈ {female, male}` into 0/1,
    and a record naming the column without naming the direction leaves *"is the
    coefficient on `male` or on `female`"* unanswerable — a reported number made
    uninterpretable by a record that is complete about columns and silent about
    values. `encodings` maps column name → `engine.fix_encoding`'s block; a kind
    that has no mapping passes none and its sentence is unchanged.

    **And it names the ones that were left alone.** A user who applied seven of
    nine made a decision about the other two, and the decision was *not this*.
    §09's recorded-absence rule: without a record, a column deliberately left as
    recorded is indistinguishable from a column nobody reached.

    That naming is load-bearing beyond tidiness. The interview marks the whole
    offered set answered, declined members included — so the sentence is the
    only place saying which way each went, and a count without names would leave
    the record unable to justify what the interview then does.
    """
    n = len(columns)
    if not n:
        # The whole group declined. It gets its own sentence rather than
        # "0 features were …", because a reader has to be able to tell a
        # considered refusal from an empty run.
        left = ", ".join(f"`{c}`" for c in declined) or "none"
        return (f"{len(declined)} feature"
                f"{'' if len(declined) == 1 else 's'} ({left}) could have been "
                f"{label} and were deliberately left as recorded.")
    named, _sep = _named(columns, encodings)
    head = (f"{n} feature{'' if n == 1 else 's'} ({named}) were "
            f"{label}." if n != 1 else
            f"1 feature ({named}) was {label}.")
    if declined:
        left = ", ".join(f"`{c}`" for c in declined)
        head += (f" {len(declined)} other"
                 f"{'' if len(declined) == 1 else 's'} in the same group "
                 f"({left}) {'was' if len(declined) == 1 else 'were'} "
                 f"deliberately left as recorded.")
    return head
