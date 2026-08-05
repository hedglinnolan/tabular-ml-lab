"""turbotab.packs — field awareness without a second app.

`DOMAIN_PACKS.md`, made executable. The lens question, the five packs, and the
one architectural claim that keeps breadth tractable:

> **The unit of domain knowledge is a finding.** Adding a domain means adding
> detectors and reference data. It never means adding interface.

So nothing here invents a card type. A pack emits findings in the engine's own
shape, with `fix_kind="none"` where it is reporting rather than proposing — and
`ml.router._is_repairable` already treats that as a report and not a fork, which
is why **guard #2 is structural rather than aspirational**: a pack that reports
cannot add a question, whatever it reports.

## The three guards

1. **A pack may not add interview components.** It supplies findings and
   defaults. The one exception is deliberate and narrow: `reverse_coding` is a
   real question, because reverse-coding needs a codebook the app does not have.
   It is gated on its own detector, so it exists only where it applies.
2. **A pack must not fire on non-matching data.** Every detector below reads
   SHAPE, never a label — *"the label sets priors; the data resolves them into
   findings"* (§06). `turbotab/test_a_pack_does_not_fire_on_the_wrong_data.py`
   runs every pack against every fixture and asserts the question count is
   unchanged everywhere it does not belong.
3. **Every default states its reason and is overturnable.** The confidence
   marker governs the treatment, and it is carried on the finding rather than
   implied: `derived` is pre-selected with its reason shown, `convention` is
   pre-selected and stated AS convention, `offered` is never defaulted at all.

A fourth, on voice: **conventions are stated as conventions.** *"The field
convention here is Pareto scaling"* is honest; *"you should use Pareto scaling"*
is not, because the app never speaks in the user's name — and a pack is the
easiest place in the product to break that rule.

## Why the lens is asked and not inferred

The same architecture as the grain question, for the same reason: the user knows
and the engine can only guess. Detection runs as a **suggestion** and as a
**contradiction detector**, never as the answer. A pack that fires on the wrong
data asserts something false *authoritatively*, which is harder for a user to
catch than an ordinary bug.

And "Something else, or not sure" is first-class. The app is fully functional
with no lens; a pack is an accelerator. Any design in which an unlisted field
degrades the experience has built a tool for five disciplines rather than a tool
that is unusually good at five.
"""
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ── the answers ──────────────────────────────────────────────────────────────

METABOLOMICS = "metabolomics"
GENOMICS = "genomics"
DIETARY = "dietary"
CLINICAL = "clinical"
SURVEY = "survey"
OTHER = "other"

LENS_KEYS: Tuple[str, ...] = (METABOLOMICS, GENOMICS, DIETARY, CLINICAL,
                              SURVEY, OTHER)

LENS_LABELS: Dict[str, str] = {
    METABOLOMICS: "Metabolomics or proteomics",
    GENOMICS: "Genomics or transcriptomics",
    DIETARY: "Dietary intake",
    CLINICAL: "Clinical measurements and labs",
    SURVEY: "Survey or questionnaire instruments",
    OTHER: "Something else, or not sure",
}

# WHY AN EMPTY LENS IS REFUSED, stated once and read twice: `normalize` raises
# it, and the Router serves it as the submit control's `min_reason` so the
# interface can say why the button is not offered. `GUIDED-038` is what happens
# when the interface composes its own version — the two drift, and the one the
# user reads is the one nothing tests.
LENS_EMPTY_REFUSAL = (
    "The lens question needs an answer, and 'Something else, or not sure' is "
    "one — the app is fully functional without a lens. An empty selection "
    "would be indistinguishable from never having asked.")

LENS_TITLE = "What kind of measurements are in this table?"
LENS_WHY = ("Pick all that apply. This changes what we look for and what we "
            "suggest — it never limits what you can do.")
LENS_CONSUMER = (
    "The structural diagnosis reads it first, because the diagnosis is itself "
    "field-sensitive: 400 columns across 80 rows reads as malformed to a "
    "general-purpose import doctor and is the expected shape for an assay "
    "panel. After that it sets priors on missingness, on model ranking, and on "
    "which figure answers a question. It never removes an option, and every "
    "default it raises states its reason and can be overturned.")


class PackError(Exception):
    """A lens answer the app cannot honestly record."""


def normalize(keys: Sequence[str]) -> List[str]:
    """The recorded answer, validated and ordered. `other` is a real answer.

    An empty selection is refused rather than silently read as `other`: the
    difference between *"the user said none of these apply"* and *"the question
    was never answered"* is the recorded-absence rule, and a default that
    swallows the first into the second is exactly what that rule forbids.
    """
    chosen = [str(k) for k in keys or []]
    unknown = [k for k in chosen if k not in LENS_KEYS]
    if unknown:
        raise PackError(
            f"{unknown[0]!r} is not one of {list(LENS_KEYS)}.")
    if not chosen:
        raise PackError(LENS_EMPTY_REFUSAL)
    seen: List[str] = []
    for k in LENS_KEYS:                       # a stable order, not click order
        if k in chosen and k not in seen:
            seen.append(k)
    if OTHER in seen and len(seen) > 1:
        # "Not sure" beside four confident answers is not an answer, it is two.
        raise PackError(
            "'Something else, or not sure' says the listed kinds do not "
            "describe this table. Selecting it beside one that does is two "
            "different answers, and the record could not say which.")
    return seen


def methods_sentence(keys: Sequence[str]) -> str:
    """What the manuscript carries. A lens the reader cannot see is unchecked.

    §01: *the answer is a recorded decision, not hidden state.* Every domain
    default downstream is licensed by this sentence, so the sentence has to say
    what was claimed AND what it was allowed to do.
    """
    chosen = list(keys)
    if chosen == [OTHER]:
        return ("The measurements in this dataset were not described as "
                "belonging to any of the offered domains, so no domain-specific "
                "defaults were applied and every preprocessing decision was "
                "made from the data alone.")
    names = [LENS_LABELS[k].lower() for k in chosen]
    joined = names[0] if len(names) == 1 else (
        ", ".join(names[:-1]) + " and " + names[-1])
    return (f"The measurements were described as {joined}. Domain conventions "
            f"for {'these fields' if len(names) > 1 else 'this field'} informed "
            f"the defaults offered below; each is stated with its reasoning and "
            f"was open to being overridden.")


# ─────────────────────────────────────────────────────────────────────────────
# Shape readings — name-blind, and the reason they are
# ─────────────────────────────────────────────────────────────────────────────

def _numeric(df: pd.DataFrame) -> List[str]:
    return [str(c) for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c])
            and not pd.api.types.is_bool_dtype(df[c])]


def _is_assay_wide(df: pd.DataFrame, minimum: int = 30) -> bool:
    """Many measurement columns. The precondition for every assay reading.

    Deliberately a floor on the COUNT rather than on p/n. A 500-row study with
    400 features is still an assay panel, and a 5-row file with 6 columns is not
    one however bad its p/n looks.
    """
    return len(_numeric(df)) >= minimum


def _in_count_block(finding: Dict[str, Any], df: pd.DataFrame) -> bool:
    """Is this finding's column part of the count matrix?"""
    block = count_matrix(df)
    if block is None:
        return False
    return str((finding.get("params") or {}).get("column")) in set(block["columns"])


def clinical_reference_columns(df: pd.DataFrame) -> List[str]:
    """Columns the engine's own reference matcher recognizes, that have blanks.

    The clinical pack's `missingness_direction` prior is scoped to these
    (`GUIDED-027`). On an NHANES-shaped table that is the labs and not the
    questionnaire items beside them, which is the whole point of the finding.

    Goes through `match_variable_key`, which is exact against the key or a
    declared alias — never a substring. Borrowing the vetted matcher is the
    opposite of adding a fifth name list.
    """
    try:
        from ml.physiology_reference import load_reference_bundle, match_variable_key
        reference = load_reference_bundle()["nhanes"]
    except Exception:                                      # pragma: no cover
        return []
    out = []
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if match_variable_key(str(c), reference) and bool(df[c].isna().any()):
            out.append(str(c))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# The evidence badge — the epistemic status of every claim a pack makes
#
# All four research threads arrived at this independently, and the clinical one
# states why: *"that single design decision is what would make TurboTab
# trustworthy to a methodologist, because it makes the tool's epistemic position
# legible rather than uniformly confident."*
#
# It is NOT a card type, so guard #1 holds. It is a token, rendered beside an
# advisory that already exists (`DESIGN_LANGUAGE.md` §11).
#
# **It subsumes and sharpens `derived` / `convention` / `offered`.** Those three
# describe THE APP'S confidence. These three describe THE FIELD'S — and the
# second is the honest one, because it is the one a reviewer can check. The
# mapping is not one-to-one, and the place it is not is the whole point:
# `offered` splits, because some offered items are genuinely disputed and some
# are merely expensive, and rendering those two the same way was the app being
# uniformly confident in a different direction.
SETTLED = "SETTLED"
CONVENTION_STATUS = "CONVENTION"
DISPUTED = "DISPUTED"
EVIDENCE_STATUSES = (SETTLED, CONVENTION_STATUS, DISPUTED)

# What each status PERMITS. Read by `recipes.register_default` and by the
# interface, so the obligation is enforced rather than documented.
MAY_PRESELECT = frozenset({SETTLED, CONVENTION_STATUS})


class EvidenceError(Exception):
    """A claim the app cannot honestly badge."""


@dataclass(frozen=True)
class Evidence:
    """Where a pack's claim comes from, and how firmly the field holds it.

    `status` is the field's position, not the app's confidence:

    * **SETTLED** — methodological consensus; a tool asserting the opposite
      would be wrong. May be a pre-selected default with its reason shown.
    * **CONVENTION** — no strong evidence base, but field expectation.
      May be pre-selected, and **must be stated as convention, never as fact**.
    * **DISPUTED** — live disagreement among competent methodologists.
      **Never defaulted silently.** Both positions stated, and a sensitivity
      analysis offered.

    `source` names a research file and a section in it, and
    `docs/turbotab/tools/evidence.py check` resolves both. The honest limit is
    exactly `ledger.py check`'s: **it verifies that a source is named and
    resolvable, not that the claim is faithful to it.** A citation that resolves
    to the wrong section is a defect this cannot see, and saying so is the
    difference between a gate and a reassurance.

    `both_sides` is required on DISPUTED and forbidden elsewhere. A disputed
    claim with one position stated is the app picking a side while wearing a
    badge that says it has not.
    """
    status: str
    source: str
    both_sides: Optional[str] = None

    def __post_init__(self) -> None:
        if self.status not in EVIDENCE_STATUSES:
            raise EvidenceError(
                f"{self.status!r} is not one of {list(EVIDENCE_STATUSES)}. The "
                f"badge is what makes the app's epistemic position legible, so "
                f"a claim it cannot describe is one the app may not make.")
        if not re.match(r"^research/[A-Z_]+\.md#.+", self.source or ""):
            raise EvidenceError(
                f"{self.source!r} is not a resolvable source. The form is "
                f"`research/FILE.md#Section heading`, and it is checked — a "
                f"citation nobody can follow is a citation nobody can check.")
        if self.status == DISPUTED and not (self.both_sides or "").strip():
            raise EvidenceError(
                f"{self.source}: a DISPUTED claim must state both positions. "
                f"One side stated under a DISPUTED badge is the app picking a "
                f"side while wearing a badge that says it has not.")
        if self.status != DISPUTED and self.both_sides:
            raise EvidenceError(
                f"{self.source}: `both_sides` belongs to DISPUTED only. On a "
                f"SETTLED or CONVENTION claim it invents a controversy.")

    @property
    def may_preselect(self) -> bool:
        return self.status in MAY_PRESELECT

    def to_dict(self) -> Dict[str, Any]:
        return {"evidence_status": self.status, "source": self.source,
                "both_sides": self.both_sides,
                "may_preselect": self.may_preselect}


# Which evidence status each marker may rest on. The two axes are genuinely
# different questions — `marker` is **what the app does**, `status` is **where
# the field stands** — so this is a compatibility table and not a translation.
#
# `offered` admits ALL THREE, and the first version of this forbade
# `SETTLED + offered` and was wrong. `DOMAIN_SCIENCE.md` §01.2 names the class
# it would have outlawed: *there is a class of thing the app must detect and
# must not act on.* Pooled QC rows are not participants — that is settled, not a
# convention — and the app still only OFFERS the exclusion, because acting on a
# high-confidence detection whose consequences are irreversible if wrong is the
# thing every pack's `hard_stops` list forbids. Settled science and a withheld
# hand are compatible, and the combination is one of the most important in the
# product.
#
# What is NOT compatible is the other direction, and it is enforced separately:
# a DISPUTED claim may never pre-select.
MARKER_STATUS: Dict[str, Tuple[str, ...]] = {
    "derived": (SETTLED,),
    "convention": (CONVENTION_STATUS,),
    "offered": (SETTLED, CONVENTION_STATUS, DISPUTED),
}


# Weakest first. The badge's obligations are ordered — DISPUTED may never be
# defaulted, CONVENTION may be defaulted stated AS convention, SETTLED may be
# defaulted outright — so *weakest* is the one a consumer must gate on.
_STATUS_RANK = {DISPUTED: 0, CONVENTION_STATUS: 1, SETTLED: 2}


@dataclass(frozen=True)
class Claim:
    """One sentence inside a statement, with the status the field holds IT at.

    **`GUIDED-064`, and it took four instances.** A finding carries one badge
    and can say two things the field holds differently. `counts_p_over_n`
    asserts that at p ≫ n an unregularized fit is degenerate — SETTLED — and in
    the same paragraph that CPM, TPM and VST are not interchangeable and no
    default is asserted, which is DISPUTED. The volcano's q-on-the-y-axis rule
    is SETTLED and the |log2FC| cut beside it is *[CONVENTION — arbitrary,
    justify biologically]*. The diverging bar is *[CONVENTION, near-universal]*
    and its neutral-midpoint treatment is *[DISPUTED]*.

    Nothing false reached a reader, because the finer status was in the prose.
    **The defect is that the badge a MACHINE reads was systematically coarser
    than the sentence a HUMAN reads** — which inverts the badge's whole purpose,
    since the badge exists to be the checkable form of the epistemic position.

    `statement` is the clause this status is about, in the same words the detail
    uses. `key` is what a consumer addresses it by. The claim set is additive:
    the headline `evidence` still says where the statement as a whole stands,
    and `claims` says where each part of it does.
    """
    key: str
    statement: str
    evidence: Evidence

    def __post_init__(self) -> None:
        if not isinstance(self.evidence, Evidence):
            raise EvidenceError(
                f"claim {self.key!r}: evidence must be an `Evidence`. A claim "
                f"badged with a dict bypasses the form check, and nothing then "
                f"resolves its source.")
        if len(self.statement) <= 20:
            raise EvidenceError(
                f"claim {self.key!r}: a claim states what it is a claim ABOUT. "
                f"A key with a status beside it is a badge on nothing.")

    def to_dict(self) -> Dict[str, Any]:
        return {"key": self.key, "statement": self.statement,
                **self.evidence.to_dict()}


@dataclass(frozen=True)
class Hedge(Claim):
    """A place a pack states a position rather than acting on one.

    `METABOLOMICS_PACK.md` §11, *"where confident automation would embarrass
    us"* — thirteen ranked items that are the pack's credibility rather than its
    confidence. **They are badges and refusals, not detectors**: nothing here
    reads the table, and every one of them is true of the sub-domain before any
    data arrives.

    **A `Hedge` IS a `Claim`**, by subclassing rather than by resemblance, and
    that is the load-bearing part. `evidence.py check` walks module-level tuples
    of `Claim` and resolves every source in them; a parallel structure beside
    `Claim` would have been outside the gate on the day it was written, which is
    `GUIDED-025`'s shape — two extension mechanisms that do not meet.

    Three fields `Claim` does not have, each because an obligation needs it:

    * **`sensitivity`** is `DOMAIN_SCIENCE.md` §01's third clause. A DISPUTED
      item is *"never defaulted silently. Both sides stated. **A sensitivity
      analysis offered.**"* `Evidence.both_sides` carries the second clause and
      nothing carried the third, so a DISPUTED badge could be well-formed while
      the app offered the user no way to find out whether the dispute mattered
      for their study. **Required on DISPUTED.**
    * **`stated_default`** is §11 item 1's *"assert a default with a stated
      rationale, never a rule."* It is a recommendation and NOT a
      pre-selection — `may_preselect` is still computed from the badge, so a
      DISPUTED hedge with a stated default recommends in prose and pre-selects
      nothing. Those are two different acts and only the second is forbidden.
    * **`what_the_app_does`** is required on all of them, because a position
      with no consequence is a paragraph. It is the sentence a reader checks the
      behavior against.

    **The status is per item and comes from the file, not from a policy.**
    Seven of the thirteen are DISPUTED, two are CONVENTION and four are SETTLED,
    and badging all thirteen DISPUTED would be the second, uncalibrated layer of
    caution `AGENT_ONBOARD.md` §00 names as a defect: it makes *"OPLS-DA is a
    rotation and does not reduce overfitting"* — which §08 marks settled among
    chemometricians — read exactly like *"nobody agrees on the QC RSD
    threshold"*, which is the failure the badge exists to prevent.
    """
    rank: int = 0
    what_the_app_does: str = ""
    sensitivity: Optional[str] = None
    stated_default: Optional[str] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.rank < 1:
            raise EvidenceError(
                f"hedge {self.key!r}: §11 is a RANKED list and the rank is part "
                f"of the content — item 1 is where the field's disagreement is "
                f"most likely to reach a reader. A hedge with no rank cannot be "
                f"served in the order the research put them in.")
        if len(self.what_the_app_does) <= 20:
            raise EvidenceError(
                f"hedge {self.key!r}: a position with no consequence is a "
                f"paragraph. State what the app does about it, which is the "
                f"sentence a reader can check the behavior against.")
        if self.evidence.status == DISPUTED and not (self.sensitivity or "").strip():
            raise EvidenceError(
                f"hedge {self.key!r}: DISPUTED is *never defaulted silently, "
                f"both sides stated, and a sensitivity analysis offered* "
                f"(`DOMAIN_SCIENCE.md` §01). `both_sides` carries the second "
                f"clause; this carries the third, and a DISPUTED item with no "
                f"way to find out whether the dispute matters for THIS study "
                f"leaves the user exactly where they started.")

    def to_dict(self) -> Dict[str, Any]:
        return {**super().to_dict(), "rank": self.rank,
                "what_the_app_does": self.what_the_app_does,
                "sensitivity": self.sensitivity,
                "stated_default": self.stated_default}


def _badge_payload(evidence: Evidence,
                   claims: Sequence["Claim"] = ()) -> Dict[str, Any]:
    """The badge a consumer reads, at the granularity the sentence has.

    **`may_preselect` is computed over the whole claim set**, and that is the
    part that makes this more than a display change. The headline evidence of
    `counts_p_over_n` is SETTLED, so `may_preselect` read True while one of the
    two things the finding says is DISPUTED — and *DISPUTED is never defaulted
    silently* is the badge's own obligation. A machine acting on the headline
    alone would have pre-selected across a disagreement it could not see.

    `weakest_status` is the one field a client needs to gate on, so acting
    correctly does not require walking the claims.
    """
    payload = dict(evidence.to_dict())
    if not claims:
        return payload
    statuses = [evidence.status] + [c.evidence.status for c in claims]
    payload["claims"] = [c.to_dict() for c in claims]
    # A SUBSET TEST rather than `all(s in MAY_PRESELECT for s in statuses)`.
    # Same answer, and the generator form is the one the name-registry guard
    # flags — *is any entry of this collection in that one* is one character
    # from a substring scan over names, and the guard cannot tell them apart.
    # `<=` on two sets can only mean exact membership.
    payload["may_preselect"] = set(statuses) <= set(MAY_PRESELECT)
    # `min` over an explicit rank rather than a first-match over an ordered
    # tuple. The ordered-tuple form reads as *"is any of these in that
    # collection"*, which `test_every_substring_match_against_a_name_is_declared`
    # flags on sight — and it was right to: the shape is one character away
    # from a substring scan over names, which is the hazard that guard exists
    # for. Writing the ordering down is also better than encoding it in the
    # order of a tuple somebody could reorder.
    payload["weakest_status"] = min(statuses, key=lambda s: _STATUS_RANK[s])
    return payload


def _check_badge(what: str, evidence: Optional[Evidence],
                 marker: Optional[str] = None) -> Evidence:
    """The badge obligation, in one place, for everything a pack says.

    `Prior.__post_init__` wrote this first and owned it alone, which is exactly
    how `GUIDED-059` happened: priors were guarded and findings and refusals —
    the two things a user actually reads — were not. One function, three
    callers, and the rendering obligation asserted before the compatibility
    table for the reason `Prior` records: written the other way round the
    specific message about pre-selection is unreachable, because `derived` and
    `convention` both fail the table first.
    """
    if evidence is None:
        raise PackError(
            f"{what}: a pack claim states where the field stands and where "
            f"that was read. A claim with no evidence badge asserts the app's "
            f"confidence as if it were the field's, which is the state "
            f"`DOMAIN_SCIENCE.md` §01.1 exists to end.")
    if not isinstance(evidence, Evidence):
        raise PackError(
            f"{what}: evidence must be an `Evidence`, not "
            f"{type(evidence).__name__}. A badge assembled as a dict bypasses "
            f"the form check, and nothing then resolves its source.")
    if marker is None:
        return evidence
    if marker not in MARKER_STATUS:
        raise PackError(
            f"{what}: marker must be derived, convention or offered. The "
            f"marker governs the treatment, so a claim without one cannot be "
            f"rendered honestly.")
    if evidence.status == DISPUTED and marker != "offered":
        raise PackError(
            f"{what}: DISPUTED is never defaulted silently, and marker "
            f"{marker!r} pre-selects. Both positions are stated and the user "
            f"chooses.")
    if evidence.status not in MARKER_STATUS[marker]:
        raise PackError(
            f"{what}: marker {marker!r} and evidence {evidence.status!r} "
            f"disagree. A `derived` claim is the engine being certain and can "
            f"only rest on SETTLED science; `offered` is the one that splits, "
            f"into {list(MARKER_STATUS['offered'])}.")
    return evidence


class PackRefusal(Exception):
    """A statement a pack declines to make, with its badge and its offer.

    **A refusal is the sharpest claim a pack makes** — *"nobody can compute
    this, not the app and not you with a spreadsheet"* — and until `GUIDED-059`
    it was the only kind that went out unbadged. The badge is required in the
    constructor rather than checked by a gate, for the reason the pre-commit
    hook exists: a rule a tired agent can skip is not a rule.

    `offer` is what the app CAN draw instead, because a refusal that offers
    nothing is indistinguishable from a missing feature and the user still has
    a real question.
    """

    def __init__(self, message: str, *, evidence: Optional[Evidence] = None,
                 offer: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.evidence = _check_badge(f"refusal {message[:40]!r}", evidence)
        self.offer = dict(offer or {})

    def to_dict(self) -> Dict[str, Any]:
        """The refusal as a payload — the reason, the offer, and the badge.

        One serializer, so an interface cannot surface the refusal without the
        badge that says where the field stands on it. `DRIVE-001`'s class is a
        status computed on the server and dropped at the boundary.
        """
        return {"refused": True, "reason": str(self), "offer": dict(self.offer),
                **self.evidence.to_dict()}


def _finding(fid: str, severity: str, title: str, detail: str,
             why: str, *, confidence: str, pack: str, marker: str,
             evidence: Optional[Evidence] = None,
             claims: Sequence["Claim"] = (),
             columns: Sequence[str] = (), params: Optional[Dict] = None,
             fix_label: str = "", fix_kind: str = "none") -> Dict[str, Any]:
    """One pack finding, in the engine's own shape.

    `fix_kind="none"` by default, and that default is load-bearing:
    `router._is_repairable` reads it as the engine refusing to guess — a report,
    not a fork — so a reporting pack cannot add a question. Guard #2 is a
    property of the data model rather than of anybody's restraint.

    `marker` is the confidence marker from `DOMAIN_PACKS.md` §07 — `derived`,
    `convention` or `offered` — and it is carried rather than implied because it
    governs the treatment. A `convention` rendered as a `derived` fact is the
    app speaking in the user's name.

    `evidence` is where the FIELD stands, and it is required. It is keyword-only
    with a `None` default for one reason and it is not convenience: a positional
    parameter would be silently satisfiable by argument order, and a default of
    `None` that raises gives the caller the sentence instead of a `TypeError`
    that says nothing about badges.

    **The badge is NESTED rather than spread into the finding**, and the
    collision is the reason. A finding already has a `source` — `structure`,
    `profile`, `pack` — naming the LAYER that produced it, and `ml.router`
    routes on it. `Evidence.to_dict()` also emits `source`, meaning the research
    citation. Two different questions sharing one key, so spreading would have
    quietly repurposed a field the router reads. `f["evidence"]["source"]` is
    the citation; `f["source"]` is still the layer.
    """
    _check_badge(fid, evidence, marker)
    return {
        "id": fid, "severity": severity, "title": title, "detail": detail,
        "why_it_matters": why, "fix_label": fix_label, "fix_kind": fix_kind,
        "confidence": confidence, "params": dict(params or {}),
        "affected_columns": [str(c) for c in columns],
        "source": "pack", "pack": pack, "marker": marker,
        # `_badge_payload` rather than `to_dict` — where the statement makes
        # more than one claim, the badge carries each of them and recomputes
        # `may_preselect` over the set (`GUIDED-064`).
        "evidence": _badge_payload(evidence, claims),
    }


# ── metabolomics ─────────────────────────────────────────────────────────────

# WHERE THE FIELD STANDS ON EACH DETECTOR'S CLAIM, beside the detector rather
# than in a table at the bottom of the file. A citation a reader has to go
# looking for is a citation nobody checks — which is most of what `GUIDED-059`
# turned out to be.
LEFT_CENSORED_EVIDENCE = Evidence(
    status=SETTLED,
    source="research/METABOLOMICS_PACK.md#03 · Missing data")

RUN_ORDER_EVIDENCE = Evidence(
    status=SETTLED,
    source="research/METABOLOMICS_PACK.md#05 · Batch correction and drift")

POOLED_QC_EVIDENCE = Evidence(
    status=SETTLED,
    source=("research/METABOLOMICS_PACK.md#Sample-role detection — the thing a "
            "generic tool cannot do"))


def _left_censored(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Missingness ordered by abundance is left censoring, not randomness.

    The reading is a rank correlation between a feature's missing rate and its
    mean abundance, and it is the whole finding: **the detection is derived**,
    because a detection limit is one instrument threshold and which features
    fall below it is decided by where they sit relative to it.

    Only the METHOD is a choice, and half-minimum wins on explainability over
    QRILC — *"values below the detection limit were imputed as half the minimum
    observed"* is a sentence a reader can evaluate.
    """
    cols = _numeric(df)
    if len(cols) < 30:
        return None
    rate = df[cols].isna().mean()
    with_blanks = [c for c in cols if rate[c] > 0]
    if len(with_blanks) < 5:
        return None
    abundance = df[cols].mean(numeric_only=True)
    usable = [c for c in cols if pd.notna(abundance[c]) and abundance[c] > 0]
    if len(usable) < 30:
        return None
    rho = pd.Series(rate[usable]).corr(
        pd.Series(np.log(abundance[usable])), method="spearman")
    if pd.isna(rho) or rho > -0.5:
        return None
    worst = rate[with_blanks].sort_values(ascending=False)
    return _finding(
        "pack::metabolomics::left_censored", "warning",
        "Your missing values cluster in the lowest-abundance features",
        (f"Across {len(usable):,} features, a feature's missing rate tracks its "
         f"abundance rank at a rank correlation of {rho:.2f}. "
         f"{len(with_blanks):,} features have blanks; the highest rate is "
         f"{worst.iloc[0]:.0%}, on one of the least abundant."),
        ("In metabolomics that usually means below the detection limit — "
         "left-censored rather than missing at random — and filling with a "
         "median would place non-detections in the middle of the distribution. "
         "Half the minimum observed is the convention, and it is the one a "
         "reader can check."),
        confidence="high", pack=METABOLOMICS, marker="derived",
        evidence=LEFT_CENSORED_EVIDENCE,
        columns=list(worst.index[:8]),
        params={"rho": round(float(rho), 3), "n_features": len(usable),
                "n_with_blanks": len(with_blanks),
                # THE FULL LIST, not the eight the card shows. The prior this
                # finding justifies is scoped to these columns (`GUIDED-027`),
                # and a prior scoped to a display truncation would be wrong
                # about 296 of them.
                "columns": list(worst.index),
                "suggested_method": "half_minimum"})


def _acquisition_order(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """A run-order column, and intensity that tracks it.

    Name-blind: a run-order column is an integer column that is a PERMUTATION of
    the row positions. That reading costs nothing on ordinary data — a study ID
    is not a permutation of 1..n unless it happens to be, and a permutation that
    nothing correlates with is not reported.

    Detection is derived; correction is `offered` and never automatic, because
    it alters every value in the table.
    """
    cols = _numeric(df)
    if len(cols) < 30:
        return None
    n = len(df)
    # THE PERMUTATION READING MOVED TO `_permutation_column` and is called here
    # rather than repeated. `_no_run_order` asserts *"there is no run order in
    # this file"* and needs the same answer this one gets; two copies of the
    # reading is the arrangement in which the app says that sentence on a table
    # where this detector has just named a run-order column.
    order_col = _permutation_column(df)
    if order_col is None:
        return None

    others = [c for c in cols if c != order_col]
    order = df[order_col].to_numpy(dtype=float)
    tracked = []
    for c in others:
        s = df[c]
        filled = s.fillna(s.median())
        if filled.nunique() < 3:
            continue
        with np.errstate(all="ignore"):
            r = np.corrcoef(order, np.log1p(np.clip(filled.to_numpy(dtype=float),
                                                    0, None)))[0, 1]
        if not np.isnan(r) and abs(r) > 0.3:
            tracked.append(c)
    share = len(tracked) / max(len(others), 1)
    if share < 0.15:
        return None
    return _finding(
        "pack::metabolomics::run_order", "warning",
        f"There is a run-order column, and intensity tracks it",
        (f"`{order_col}` runs 1 to {n:,} with every position used exactly once. "
         f"{len(tracked):,} of {len(others):,} features ({share:.0%}) correlate "
         f"with it above 0.3 in absolute value."),
        ("Instrument drift is often the largest single variance component in a "
         "metabolomics run, larger than the biology. Correction is not applied "
         "here: it alters every value in the table, so it is a decision rather "
         "than a default."),
        confidence="high", pack=METABOLOMICS, marker="offered",
        evidence=RUN_ORDER_EVIDENCE,
        columns=[order_col] + tracked[:6],
        params={"run_order_column": order_col, "n_tracking": len(tracked),
                "share_tracking": round(share, 3)})


def _pooled_qc(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Rows that are one sample injected repeatedly, not participants.

    **This is the class of error only the lens can see**, and the cheapest
    demonstration that the opening question earns its place: pooled QC rows look
    exactly like participants, must never enter a model, and are needed for
    quality assessment. A generic tool models them silently.

    Name-blind again, and the evidence is variance: a minority level of some
    categorical column whose rows are markedly *less* variable across the
    feature block than the majority's. One sample injected eight times has
    technical variation and no biological variation, and that shows.
    """
    cols = _numeric(df)
    if len(cols) < 30:
        return None
    n = len(df)
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            continue
        counts = s.value_counts(dropna=True)
        if len(counts) != 2:
            continue
        minority, majority = counts.index[-1], counts.index[0]
        n_minor = int(counts.iloc[-1])
        if n_minor < 3 or n_minor > 0.3 * n:
            continue
        block = df.loc[s == minority, cols]
        rest = df.loc[s == majority, cols]
        if len(rest) < 5:
            continue
        with np.errstate(all="ignore"):
            rsd_minor = float((block.std() / block.mean().abs()).median())
            rsd_rest = float((rest.std() / rest.mean().abs()).median())
        if not np.isfinite(rsd_minor) or not np.isfinite(rsd_rest) or rsd_rest <= 0:
            continue
        if rsd_minor > 0.6 * rsd_rest:
            continue
        return _finding(
            "pack::metabolomics::pooled_qc", "critical",
            f"{n_minor:,} rows look like pooled quality-control injections",
            (f"The {n_minor:,} rows where `{c}` is {minority!r} vary far less "
             f"across the {len(cols):,} features than the {int(counts.iloc[0]):,} "
             f"rows where it is {majority!r} — a median relative standard "
             f"deviation of {rsd_minor:.0%} against {rsd_rest:.0%}. That is one "
             f"sample injected repeatedly, not {n_minor:,} different people."),
            ("They are not participants. Modeling them is an error with no "
             "legitimate reading — they would contribute rows the model can fit "
             "perfectly and a held-out set could contain them. They stay in the "
             "table for quality assessment and out of the modeling rows."),
            confidence="high", pack=METABOLOMICS, marker="derived",
            evidence=POOLED_QC_EVIDENCE,
            columns=[str(c)],
            params={"column": str(c), "qc_value": str(minority),
                    "n_qc": n_minor, "rsd_qc": round(rsd_minor, 3),
                    "rsd_participants": round(rsd_rest, 3)})
    return None


# ── metabolomics · §11, where confident automation would embarrass us ────────
#
# Thirteen ranked positions, in the research's own order. **Badges and refusals,
# not detectors** — nothing below reads a table, because none of it is about one
# table. It is where the sub-domain does not agree with itself, and stating that
# is what makes the pack credible rather than confident.
#
# THE STATUS IS PER ITEM AND IS READ OUT OF THE FILE. Seven DISPUTED, two
# CONVENTION, four SETTLED. Sections §02, §04, §06, §07 and §08 badge these
# points themselves, and where the file says [SETTLED among chemometricians] the
# hedge says SETTLED — a uniform DISPUTED over all thirteen would make a settled
# technical fact read like a live disagreement, which is `GUIDED-170` pointing
# the other way and is the exact failure the badge exists to prevent.

_M11 = "research/METABOLOMICS_PACK.md#11 · Where confident automation would embarrass us"
_M02 = "research/METABOLOMICS_PACK.md#02 · Quality filtering"
_M03 = "research/METABOLOMICS_PACK.md#03 · Missing data"
_M04 = "research/METABOLOMICS_PACK.md#04 · Normalization, transformation, scaling"
_M05 = "research/METABOLOMICS_PACK.md#05 · Batch correction and drift"
_M06 = "research/METABOLOMICS_PACK.md#06 · EDA and presentation"
_M07 = "research/METABOLOMICS_PACK.md#07 · Is untargeted metabolomics compositional?"
_M08 = "research/METABOLOMICS_PACK.md#08 · Feature selection and modeling"


class SoftwareDefaultRefusal(PackRefusal):
    """A number that belongs to a version of somebody else's software.

    **§11 item 12 is a hard stop**: *any claim about a specific software default
    — MetaboAnalyst's IQR filter, `pmp`'s blank fold change, structToolbox's
    D-ratio — changes between versions, and the right move is to read it from
    the user's installed version rather than to hard-code it.* This app has read
    no version, so it does not carry the numbers and says so.

    A subclass rather than a bare `PackRefusal` for the reason
    `PrevalenceRefusal` is one: the gate that walks refusal call sites finds it
    by subclassing, so a refusal added later cannot go out unbadged because
    nobody remembered to extend a list.
    """


#: The three §11 item 12 names, and nothing else. A fourth would be this module
#: inventing a refusal the research did not ask for, which is the same defect as
#: inventing a threshold.
SOFTWARE_DEFAULTS_REFUSED = (
    ("metaboanalyst_iqr_filter",
     "the near-constant IQR filter, scaled to feature count"),
    ("pmp_blank_fold_change", "the blank fold change shipped by `pmp`"),
    ("structtoolbox_dratio", "the D-ratio acceptance criterion in structToolbox"),
)

SOFTWARE_DEFAULT_EVIDENCE = Evidence(status=SETTLED, source=_M11)


def software_default(key: str) -> Dict[str, Any]:
    """Always refuses, and the refusal is the feature.

    There is no branch that returns a number. A function that could return one
    would be a place a later loop puts a constant in, and the whole content of
    item 12 is that this app has no standing to state these values.
    """
    label = dict(SOFTWARE_DEFAULTS_REFUSED).get(key)
    if label is None:
        raise PackError(
            f"{key!r} is not one of the three software defaults §11 item 12 "
            f"names: {[k for k, _ in SOFTWARE_DEFAULTS_REFUSED]}.")
    raise SoftwareDefaultRefusal(
        f"This app does not carry {label}. It is a fact about a version of "
        f"somebody else's software, it changes between versions, and this app "
        f"has read no version — so quoting a number here would be asserting "
        f"something it has not checked.",
        evidence=SOFTWARE_DEFAULT_EVIDENCE,
        offer={
            "label": "Read it from the version you ran",
            "note": ("Take the value from the release you actually used and "
                     "state it in your methods with the software name and "
                     "version. The research asks for exactly that, and it is "
                     "also what the filtering chain has to report anyway: every "
                     "threshold, in order, with the features remaining after "
                     "each step."),
        })


METABOLOMICS_HEDGES: Tuple[Hedge, ...] = (
    Hedge(
        key="qc_rsd_threshold", rank=1,
        statement=("There is no consensus QC RSD threshold, and the scoping "
                   "review of pooled-QC practice says so outright."),
        evidence=Evidence(
            status=DISPUTED, source=_M02,
            both_sides=(
                "30% is the most commonly published untargeted cutoff and is "
                "the QComics acceptance criterion; 20% is stricter and is "
                "standard in targeted assays; 25% is a third value in "
                "circulation for LC-MS. Against all three, the scoping review "
                "of pooled-QC practice states plainly that there is no widely "
                "accepted metric for delineating acceptable from unacceptable "
                "data quality.")),
        stated_default=(
            "30%, because it is the most commonly published untargeted cutoff "
            "and is the QComics acceptance criterion. That is a default with a "
            "stated rationale and it is not a rule — pick one, state it in your "
            "methods, and the app will show what each one costs in features."),
        sensitivity=(
            "Read the QC RSD cumulative distribution with lines at 20, 25 and "
            "30 rather than accepting one of them unseen. That figure shows "
            "exactly what a threshold costs, and overlaying it before and after "
            "drift correction is the most persuasive QC figure in the field."),
        what_the_app_does=(
            "States all three values with the rationale for the one it "
            "suggests, and never filters on any of them without being told to. "
            "Which value was used goes in the filtering waterfall.")),
    Hedge(
        key="blank_ratio_fold_change", rank=2,
        statement=("The blank-ratio fold change is genuinely unsettled, and "
                   "the pack asserts none of the values in use."),
        evidence=Evidence(
            status=DISPUTED, source=_M02,
            both_sides=(
                "3x, 5x, 10x and 20x are all in use as the ratio of the median "
                "in biological samples to the median in blanks, and the "
                "research calls this genuinely no consensus. What is not in "
                "dispute is the blank itself: it has to be a process blank — "
                "water through the full extraction with the same labware — "
                "because a solvent-only injection cannot detect plasticizer "
                "contamination.")),
        sensitivity=(
            "Run the blank filter at the low and the high end of the range and "
            "report how many features each removes, as a row of the filtering "
            "waterfall rather than as a single number nobody can check."),
        what_the_app_does=(
            "Names the four values in circulation and asserts none of them. "
            "The number `pmp` ships is refused rather than quoted — see item "
            "12, which is a hard stop rather than a preference.")),
    Hedge(
        key="imputation_method", rank=3,
        statement=("The metabolomics and the proteomics imputation benchmarks "
                   "disagree with each other, and the disagreement is the "
                   "finding rather than a gap to be closed."),
        evidence=Evidence(
            status=DISPUTED, source=_M03,
            both_sides=(
                "The metabolomics benchmark makes QRILC the best performer "
                "under MNAR — it draws from a truncated distribution estimated "
                "by quantile regression, with much smaller error than random "
                "forest, SVD and kNN — and random forest the best under "
                "MCAR/MAR. A major proteomics benchmark reports random forest "
                "consistently robust across all MNAR situations and most "
                "suitable for label-free work when the mechanism is unknown. "
                "Both are cited, both are presented, and the pack does not "
                "pretend the contradiction is resolved.")),
        stated_default=(
            "Diagnose before choosing. If the missingness-versus-intensity plot "
            "shows censoring, QRILC or GSimp; if missingness is flat with "
            "respect to intensity, random forest or kNN. Half-minimum stays "
            "available as the match-what-everyone-else-published option with "
            "its caveat attached, and a feature whose missingness differs by "
            "group is reported as presence/absence rather than imputed."),
        sensitivity=(
            "Always run the primary analysis under two imputation schemes and "
            "report whether conclusions change. This sensitivity analysis is "
            "the single highest-value thing a tool can add here — cheap, almost "
            "never done, and it directly answers the reviewer's objection."),
        what_the_app_does=(
            "Carries both benchmarks and suppresses neither, and runs the fork "
            "for real: `turbotab/sensitivity.py` re-fits every model on the "
            "same training rows and scores it on the same held-out rows with "
            "the missing-value handling varied, and reports whether the "
            "substantive conclusion changed.")),
    Hedge(
        key="pareto_vs_autoscaling", rank=4,
        statement=("Pareto scaling is the metabolomics cultural default and is "
                   "stated here as a convention rather than as a fact: it "
                   "reduces masking by abundant metabolites, it is sensitive to "
                   "large fold changes, and van den Berg's own analysis "
                   "preferred autoscaling and range scaling."),
        evidence=Evidence(status=CONVENTION_STATUS, source=_M04),
        stated_default=(
            "Pareto, as the field convention, with autoscaling pushed beside it "
            "as the alternative rather than buried in a variant list."),
        sensitivity=(
            "Side-by-side PCA under two or three scaling choices, which makes "
            "the arbitrariness visible and honest."),
        what_the_app_does=(
            "Registers Pareto as the pack's scaling default with that reason "
            "attached and pushes autoscaling against it, and never presents "
            "Pareto as the correct choice. The near-universal published "
            "combination — sum or PQN, then log, then Pareto — is a convention "
            "and not settled, and a tool that presented it as correct would be "
            "confidently wrong. Scaling is also for the multivariate path only: "
            "fold changes and box plots are computed from the normalized but "
            "unscaled copy, because a fold change in z-units is meaningless.")),
    Hedge(
        key="compositionality", rank=5,
        statement=("Whether untargeted metabolomics is compositional is "
                   "genuinely disputed, and confident wrongness in either "
                   "direction would embarrass the tool."),
        evidence=Evidence(
            status=DISPUTED, source=_M07,
            both_sides=(
                "For: the detector and the ion source have finite capacity, ion "
                "suppression means one compound's abundance genuinely affects "
                "another's measured signal, and any normalization that divides "
                "by a total — TIC, sum, MSTUS, mol% — imposes closure and makes "
                "the data compositional by construction. Against: untargeted "
                "work observes a small, biased, technology-dependent subset "
                "while compositional theory concerns a closed whole; features "
                "sit on incommensurable scales, so the ratio of one feature's "
                "intensity to another's is not a ratio of amounts; and zeros "
                "and left-censoring are pervasive while CLR cannot tolerate "
                "them. The empirical evidence cuts both ways and neither half "
                "is suppressed here: a 2025 result reports CLR-transformed data "
                "explaining less variance in the first two components and "
                "failing to resolve sample clustering at the same resolution, "
                "and multiomic time-series work reports CLR revealing novel "
                "relationships and stronger associations.")),
        stated_default=(
            "Do not CLR-transform by default. Use PQN or median-fold plus log, "
            "warn about closure whenever a sum-based normalizer is chosen, and "
            "phrase results from such an analysis in relative language."),
        sensitivity=(
            "Offer CLR or ILR as an explicit sensitivity analysis with a "
            "documented zero-replacement strategy — multiplicative or "
            "Bayesian-multiplicative, not half-minimum, because that choice "
            "propagates into the geometric mean of every sample."),
        what_the_app_does=(
            "Neither position is suppressed, and the closure warning escalates "
            "rather than being uniform: it goes high-priority when the "
            "normalizer is sum, TIC or mol%, when a few features carry a large "
            "fraction of total signal, or when a treatment plausibly causes a "
            "large global shift — and in that last case the app says plainly "
            "that group differences in every other feature may be an artifact "
            "of the normalizer.")),
    Hedge(
        key="batch_correction", rank=6,
        statement=("ComBat is both the standard between-batch method and "
                   "demonstrably capable of manufacturing FDR-corrected false "
                   "positives on pure noise, so it is never presented as "
                   "safe."),
        evidence=Evidence(
            status=DISPUTED, source=_M05,
            both_sides=(
                "For correcting the data: ComBat is the most cited "
                "between-batch method and it works when batches are balanced "
                "with respect to the outcome. Against: a 2020 simulation "
                "applied ComBat to randomly generated data with no true signal "
                "and produced alarming numbers of FDR- and "
                "Bonferroni-corrected false positives, in balanced designs as "
                "well as unbalanced ones, and a 2016 reanalysis found ComBat "
                "inflating a result from eleven genes to over a thousand under "
                "an unbalanced design. Including batch as a covariate or a "
                "random effect in the model instead is statistically the "
                "cleanest, because it propagates uncertainty rather than "
                "pretending corrected values are observed — settled in "
                "biostatistics and contrarian in metabolomics practice, where "
                "correct-then-test dominates.")),
        stated_default=(
            "Model the batch rather than correct the data, on the univariate "
            "path. That is a recommendation with a reason, not a rule, and "
            "correcting the data stays available."),
        sensitivity=(
            "Report QC RSD and D-ratio both before and after correction, and "
            "check an INDEPENDENT quantity — technical-replicate correlation, "
            "or a known positive control. QC RSD improving after a QC-fitted "
            "correction is circular: the correction was fitted to minimize "
            "exactly that number, so the improvement is not evidence the "
            "analysis is sound."),
        what_the_app_does=(
            "Exposes the fork between correcting the data and modeling the "
            "batch rather than hiding it, and refuses a confounded design "
            "outright: if a group is wholly or nearly contained in one batch "
            "then batch and biology are the same variable, no method can "
            "separate them, and anything corrected there would be guessing.")),
    Hedge(
        key="oplsda_is_a_rotation", rank=7,
        statement=("OPLS-DA rotates the PLS solution so between-class "
                   "variation concentrates in one predictive component. It "
                   "improves interpretability; it does not improve predictive "
                   "performance and it does not reduce overfitting, because "
                   "the predictive subspace is the same."),
        evidence=Evidence(status=SETTLED, source=_M08),
        what_the_app_does=(
            "Will not imply that OPLS-DA fixes overfitting anywhere, and lists "
            "believing that it does in the anti-pattern registry. The research "
            "marks this settled among chemometricians and widely misunderstood "
            "by practitioners, and calling OPLS-DA overfitting-resistant is a "
            "technical error a chemometrician would catch instantly — which is "
            "why this item is SETTLED and not DISPUTED. The overfitting problem "
            "is real and belongs to the whole family: when features are at "
            "least twice samples, PLS-DA readily separates randomly assigned "
            "labels, and what answers that is permutation testing with "
            "selection redone inside every permutation.")),
    Hedge(
        key="q2_threshold", rank=8,
        statement=("Q-squared above 0.5 is a rule of thumb rather than a test, "
                   "and nothing in this app gates on it."),
        evidence=Evidence(
            status=DISPUTED, source=_M08,
            both_sides=(
                "In its favor, it is the field's habitual pass mark and is "
                "embedded in SIMCA-era practice. Against it, Triba et al. "
                "showed that in metabolomics the K-fold cross-validation "
                "parameters depend strongly on which individuals land in which "
                "validation subset, and that a simple permutation of dataset "
                "rows can flip the conclusion about model significance; "
                "Szymanska et al. found perfect classification or a Q-squared "
                "of 0.99 attainable purely by chance through a lucky split, and "
                "reported the number of misclassifications and AUROC to be more "
                "efficient and reliable diagnostic statistics than Q-squared.")),
        sensitivity=(
            "Report the nested-CV performance distribution — the spread of "
            "outer-fold values, not a single number — beside the permutation "
            "test, rather than one Q-squared against one line."),
        what_the_app_does=(
            "Carries no pass/fail gate on Q-squared anywhere, and no PASS/FAIL "
            "stamp on any threshold: using Q-squared as a gate is in the "
            "anti-pattern registry, and a permutation test with at least a "
            "thousand permutations is what the multivariate path reports "
            "instead.")),
    Hedge(
        key="sample_size", rank=9,
        statement=("No valid generic power calculation exists for untargeted "
                   "work, so any specific claim is framed as detectable-effect-"
                   "size guidance with its assumptions shown."),
        evidence=Evidence(
            status=DISPUTED, source=_M08,
            both_sides=(
                "Conventional power calculation requires an effect size that "
                "hypothesis-free untargeted work does not have. The practical "
                "guidance that circulates is real and is labeled as guidance: "
                "controlled interventions with large effects have identified "
                "biomarkers with four to twenty subjects per arm, human "
                "observational cohorts need substantially more, anything under "
                "about twenty per group is hypothesis-generating only, and "
                "anything claimed as a biomarker needs an independent "
                "validation cohort. Against reading any of it as a rule: a "
                "post-hoc power figure is statistically meaningless, which is "
                "why the answer is a detectable-effect-size curve instead.")),
        sensitivity=(
            "Given n, alpha after FDR correction, and the observed per-feature "
            "CV, report what fold change was detectable at 80% power — and put "
            "it in the limitations rather than leaving a reviewer to compute "
            "it."),
        what_the_app_does=(
            "Offers no power calculation and no post-hoc power number. A claim "
            "about what this study could see is framed as a detectable effect "
            "size with its assumptions printed beside it.")),
    Hedge(
        key="eighty_percent_rule", rank=10,
        statement=("The 80% rule has a plain form and a modified form and which "
                   "one is in use changes which features survive: the plain "
                   "rule keeps a feature detected in at least 80% of samples, "
                   "the modified rule keeps it if it is detected in at least "
                   "80% of the samples of at least one class."),
        evidence=Evidence(status=CONVENTION_STATUS, source=_M02),
        stated_default=(
            "The modified form, and the pack says why rather than asserting it: "
            "the plain rule silently deletes metabolites present in cases and "
            "absent in controls, which is precisely the kind of finding the "
            "study is looking for."),
        sensitivity=(
            "Report the feature count under both forms in the filtering "
            "waterfall, so the cost of the choice is visible instead of "
            "inferred."),
        what_the_app_does=(
            "States which form was applied rather than leaving it implicit, and "
            "applies every filter using only QCs, blanks and overall "
            "missingness — never the group labels, because filtering features "
            "by a group difference before testing them is circular and inflates "
            "false positives.")),
    Hedge(
        key="hotelling_versus_group_ellipses", rank=11,
        statement=("Hotelling's T-squared ellipse and group-wise 95% confidence "
                   "ellipses are different objects, and mixing them up in a "
                   "rendered figure would be a visible, elementary error."),
        evidence=Evidence(status=SETTLED, source=_M06),
        what_the_app_does=(
            "Draws them differently and labels them explicitly. The T-squared "
            "ellipse is a single ellipse over all samples defining the "
            "multivariate 95% region — an outlier boundary — and is rendered as "
            "a single dashed grey outline. Group-wise confidence ellipses "
            "describe where each group's mean and spread lie and are rendered "
            "filled and group-colored. Papers routinely mislabel one as the "
            "other, and a T-squared ellipse labeled as a group confidence "
            "ellipse is in the anti-pattern registry.")),
    Hedge(
        key="software_defaults", rank=12,
        statement=("A specific software default is a fact about a version, and "
                   "this app has read no version — so it does not carry those "
                   "numbers and refuses to supply them."),
        evidence=Evidence(status=SETTLED, source=_M11),
        what_the_app_does=(
            "Refuses all three the research names rather than quoting them: the "
            "near-constant IQR filter scaled to feature count, the blank fold "
            "change shipped by `pmp`, and the D-ratio acceptance criterion in "
            "structToolbox. They change between versions, the right source is "
            "the release you actually ran, and this app cannot read it. The "
            "refusals are served beside this item with what to do instead — a "
            "refusal that offers nothing is indistinguishable from a missing "
            "feature.")),
    Hedge(
        key="microbiome_analogy", rank=13,
        statement=("Compositional data analysis is settled for microbiome "
                   "relative-abundance data, and that consensus is not imported "
                   "into metabolomics by analogy."),
        evidence=Evidence(status=SETTLED, source=_M07),
        what_the_app_does=(
            "States the specific ways the analogy fails rather than gesturing "
            "at it. Untargeted metabolomics observes a small, biased, "
            "technology-dependent subset rather than a closed whole. Features "
            "sit on incommensurable scales — ionization efficiency varies by "
            "orders of magnitude — so a ratio of two feature intensities is not "
            "a ratio of amounts, and CLR's geometric mean is taken across "
            "exactly those incommensurable quantities. Zeros and left-censoring "
            "are pervasive while CLR cannot tolerate them, so every CLR "
            "analysis needs a zero replacement whose choice then contaminates "
            "all features through the geometric mean. And with absolute "
            "quantification the data are genuinely absolute.")),
)
# ── metabolomics · redundancy · METABOLOMICS_PACK.md §01 ────────────────────

REDUNDANCY_EVIDENCE = Evidence(
    status=SETTLED,
    source=("research/METABOLOMICS_PACK.md#Redundancy detection — a real "
            "differentiator"))

REDUNDANCY_CLAIMS = (
    Claim("not_independent",
          "Untargeted features are not independent: one compound produces "
          "adducts, isotopologues, dimers and in-source fragments, so a "
          "feature count is not a compound count.",
          REDUNDANCY_EVIDENCE),
    # THE THRESHOLDS ARE A CONVENTION AND THE CLAIM SAYS SO. §01 states
    # `r > 0.9` and `±0.05–0.1 min` without a citation behind either, which is
    # what a field convention looks like when it is written down. Badging the
    # arithmetic SETTLED alongside the biology would be the coarse-badge defect
    # `GUIDED-064` was filed for, in the direction that flatters the app.
    Claim("cut_points",
          "The r > 0.9 correlation cut and the ±0.05–0.1 min retention-time "
          "window are field conventions rather than derived quantities, and a "
          "different cut gives a different effective count.",
          Evidence(
              status=CONVENTION_STATUS,
              source=("research/METABOLOMICS_PACK.md#Redundancy detection — a "
                      "real differentiator"))),
)

#: §01's own number. Signed rather than absolute: two ions of one molecule rise
#: and fall together, and a strongly ANTI-correlated pair is evidence of two
#: compounds rather than one.
_REDUNDANCY_R = 0.9

#: How many samples two features must both be observed in before their
#: correlation is used. A correlation over eleven overlapping rows crosses 0.9
#: easily and means nothing, and the features with the fewest observations are
#: exactly the faint ionization products this reading is about.
_REDUNDANCY_MIN_OVERLAP = 20

#: How much of the block must collapse before this is worth saying. **The app's
#: number, not the research's** — §01 gives a method and a consequence and no
#: reporting threshold. The consequence is a claim in a manuscript's
#: data-description sentence, and a claim off by two features in four hundred is
#: not wrong in the way §01 means.
_REDUNDANCY_MIN_COLLAPSE = 0.05


def _correlation_clusters(matrix: np.ndarray,
                          threshold: float) -> List[List[int]]:
    """Connected components of the >threshold correlation graph.

    **Single linkage, and the chain is why the largest cluster is reported.**
    Two features joined through a third are one cluster here even where they do
    not correlate with each other, which is right for an adduct series — the
    parent ion links the sodium and potassium adducts that are each faint — and
    is also how a spuriously-linked pair drags two compounds together. The size
    of the largest cluster is what makes that visible to a reader instead of
    disappearing into a total.
    """
    n = matrix.shape[0]
    adjacency = matrix > threshold
    seen = np.zeros(n, dtype=bool)
    clusters: List[List[int]] = []
    for start in range(n):
        if seen[start]:
            continue
        seen[start] = True
        stack, member = [start], [start]
        while stack:
            node = stack.pop()
            for neighbor in np.nonzero(adjacency[node])[0]:
                if not seen[neighbor]:
                    seen[neighbor] = True
                    stack.append(int(neighbor))
                    member.append(int(neighbor))
        clusters.append(sorted(member))
    return clusters


def _redundancy(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """How many compounds the feature count is really counting.

    §01 asks for clustering on near-identical retention time **and** high
    inter-feature correlation. **Only the second half is built, and the first
    half is not built rather than approximated.** Neither retention time nor m/z
    is recoverable from any table this repository can read — `mz_0001` is an
    ordinal index and there is no RT column anywhere — so the co-elution test has
    no input. `test_the_redundancy_estimate_reaches_a_person` carries the RT half
    as a failing test naming what the data would have to supply, which is
    `GUIDED-119`'s form and is the honest alternative to a fabricated column.

    **The two corrections point opposite ways, and both are reported.**
    Requiring correlation alone merges *more* than requiring correlation and
    co-elution would, so supplying a retention time could only split these
    groups and never merge them — that correction raises the effective count.
    Pushing the other way: a column observed in fewer than
    `_REDUNDANCY_MIN_OVERLAP` samples has no computable correlation with
    anything and is counted here as independent, and measuring it could only
    lower the count. **So this is not a one-sided bound and the finding does not
    claim to be one.** The first draft of it did — *"a lower bound"* — which was
    true of the retention-time half alone and false of the reading as a whole,
    and it is exactly the kind of tidy sentence `DOMAIN_SCIENCE.md` §01 is about.
    Both directions are named, with the count of columns in the second.

    Reported, never applied. Collapsing a feature block changes what is
    analyzed, so it is `offered`.
    """
    cols = _numeric(df)
    # The pack's own assay precondition rather than a second one. Every other
    # metabolomics detector gates on `_is_assay_wide`, and a redundancy reading
    # with its own wider floor would be silent on a table the three beside it
    # spoke about, for a reason nothing records.
    if not _is_assay_wide(df):
        return None
    frame = df[cols]
    corr = frame.corr(method="pearson",
                      min_periods=_REDUNDANCY_MIN_OVERLAP).to_numpy(dtype=float)
    np.fill_diagonal(corr, 0.0)
    corr = np.nan_to_num(corr, nan=0.0)

    clusters = _correlation_clusters(corr, _REDUNDANCY_R)
    effective = len(clusters)
    collapsed = len(cols) - effective
    if collapsed < max(3, round(_REDUNDANCY_MIN_COLLAPSE * len(cols))):
        return None

    observed = frame.notna().sum()
    unmeasurable = int((observed < _REDUNDANCY_MIN_OVERLAP).sum())
    multi = sorted((c for c in clusters if len(c) > 1), key=len, reverse=True)
    largest = len(multi[0])
    factor = len(cols) / max(effective, 1)
    # A NAMED EXAMPLE, so the number is checkable against the table rather than
    # only reportable. The largest cluster, because it is also the one a chain
    # would show up in.
    example = [cols[i] for i in multi[0]]
    return _finding(
        "pack::metabolomics::redundancy", "warning",
        (f"{len(cols):,} numeric columns, but about {effective:,} "
         f"independent quantities"),
        (f"{len(multi):,} groups of columns move together at a Pearson "
         f"correlation above {_REDUNDANCY_R} — {collapsed:,} of the "
         f"{len(cols):,} numeric columns sit inside a group with at least one "
         f"other, and the largest group holds {largest:,}. The biggest is "
         f"`{'`, `'.join(example[:4])}`"
         + ("" if largest <= 4 else f" and {largest - 4:,} more") + ". "
         f"A compound count read off this table's width — {len(cols):,} — "
         f"overstates it by about {factor:.1f}×.\n\n"
         f"**Clustered on correlation only, and the two things that leaves out "
         f"pull opposite ways.** The research clusters on near-identical "
         f"retention time *and* correlation, and this table carries no "
         f"retention time; supplying one could only split these groups and "
         f"never merge them, which would raise the {effective:,}. Against that, "
         + (f"{unmeasurable:,} column"
            f"{'s are' if unmeasurable != 1 else ' is'} observed in fewer than "
            f"{_REDUNDANCY_MIN_OVERLAP} samples, so no correlation could be "
            f"computed for "
            f"{'them' if unmeasurable != 1 else 'it'} and "
            f"{'they are' if unmeasurable != 1 else 'it is'} counted here as "
            f"independent — measuring "
            f"{'them' if unmeasurable != 1 else 'it'} could only lower it."
            if unmeasurable else
            "every column here is observed often enough for its correlations "
            "to be computed, so nothing is counted as independent merely for "
            "want of overlap.")),
        ("Untargeted features are not independent. One compound leaves the "
         "source as several ions — `[M+H]+`, `[M+Na]+`, `[M+K]+`, `[M+NH4]+` — "
         "and shows up again as its carbon-13 isotopologue, its dimer and its "
         "in-source fragments, all rising and falling together because they are "
         "the same molecules. Two numbers depend on which count you use: the "
         "one in the manuscript's data-description sentence, and the "
         "denominator of the multiple-testing correction. Nothing is collapsed "
         "here — merging features changes what is analyzed, and which member of "
         "a group represents the compound is a decision about the chemistry."),
        confidence="high", pack=METABOLOMICS, marker="offered",
        evidence=REDUNDANCY_EVIDENCE, claims=REDUNDANCY_CLAIMS,
        columns=example[:8],
        params={
            "n_columns": len(cols),
            "effective_features": effective,
            # NOT "lower bound". Two corrections and they point opposite ways,
            # so a single direction word here would be the machine-readable form
            # asserting more than the sentence beside it — `GUIDED-064`'s class
            # in the direction that flatters the app.
            "co_elution_would": "split groups, never merge them",
            "n_columns_below_min_overlap": unmeasurable,
            "n_collapsed": collapsed,
            "n_groups": len(multi),
            "largest_group": largest,
            "overstatement_factor": round(factor, 2),
            "r_threshold": _REDUNDANCY_R,
            "min_overlapping_samples": _REDUNDANCY_MIN_OVERLAP,
            "clustered_on": ["inter-feature correlation"],
            "not_clustered_on": ["retention time"],
            "retention_time_column": None,
            "largest_group_columns": example,
            "largest_group_columns_shown": min(4, largest),
            "largest_group_columns_total": largest,
            "groups": [[cols[i] for i in c] for c in multi],
            "groups_shown": len(multi),
            "groups_total": len(multi),
        },
        fix_label="", fix_kind="none")
# ── metabolomics · §01, the three families a generic tool cannot do ──────────
#
# `METABOLOMICS_PACK.md` §01 specifies three diagnostic families, and this
# repository shipped detectors for parts of two of them. What follows is the
# rest, in the pack's own order: **sample-role detection**, **run order, batch
# and design**, and **value-state diagnostics**.
#
# THE REGEX LIBRARY IS TRANSCRIBED, NOT INVENTED. Every pattern below appears
# verbatim in §01. That matters more here than anywhere else in this file: a
# role library assembled from recollection would look exactly like this one and
# would be a set of claims with no record behind it, which is the failure the
# badge exists to prevent. Where the implementation makes a decision the pack
# does not make — a priority order, a boundary rule, an operationalization of
# "≈ 0" — it is marked INVENTED and says so, so a reader can tell the two
# apart. `evidence.py` cannot make that distinction; a comment can.

ROLES_EVIDENCE = Evidence(
    status=CONVENTION_STATUS,
    source=("research/METABOLOMICS_PACK.md#Sample-role detection — the thing a "
            "generic tool cannot do"))

DESIGN_EVIDENCE = Evidence(
    status=SETTLED,
    source="research/METABOLOMICS_PACK.md#Run order, batch, and design")

VALUE_STATE_EVIDENCE = Evidence(
    status=SETTLED,
    source="research/METABOLOMICS_PACK.md#Value-state diagnostics")

VALUE_STATE_CONVENTION = Evidence(
    status=CONVENTION_STATUS,
    source="research/METABOLOMICS_PACK.md#Value-state diagnostics")


# ─────────────────────────────────────────────────────────────────────────────
# D1 · Sample roles
# ─────────────────────────────────────────────────────────────────────────────

POOLED_QC = "pooled_qc"
DILUTION_QC = "dilution_qc"
BLANK = "blank"
SYSTEM_SUITABILITY = "system_suitability"
CALIBRANT = "calibrant"
PROTEOMICS_REFERENCE = "proteomics_reference"

ROLE_LABELS: Dict[str, str] = {
    POOLED_QC: "pooled QC",
    DILUTION_QC: "dilution QC",
    BLANK: "blank",
    SYSTEM_SUITABILITY: "system suitability",
    CALIBRANT: "calibrant or standard",
    PROTEOMICS_REFERENCE: "proteomics reference",
}

#: The six families, **in match priority order**, with §01's patterns verbatim.
#:
#: THE ORDER IS INVENTED — the pack gives a list, not a precedence — and it is
#: ordered most-specific-first because the families overlap in exactly two
#: places and both would otherwise resolve wrongly. `QC_2x` is a dilution QC and
#: matches pooled QC's `^QC`; `QC_HeLa` is a proteomics reference and matches it
#: too. Putting the two specific families ahead of the general one is the only
#: ordering under which every example the pack itself names lands where the pack
#: puts it, which is the nearest thing to a derivation available.
#:
#: `pool` IS DROPPED FROM THE PROTEOMICS FAMILY, and it is the one place this
#: departs from the transcription. §01 lists `pool` under both pooled QC and
#: proteomics, so as written the two families are not disjoint and every pooled
#: QC in a metabolomics run would be reported as a proteomics reference channel.
#: Pooled QC owns it — that is where the pack's coaching sentence lives — and
#: the proteomics family keeps the four tokens that are unambiguously its own.
ROLE_PATTERNS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    (DILUTION_QC, (r"dQC", r"DIL", r"QC[-_ ]?(1|2|4|8|16)x", r"RQC")),
    (PROTEOMICS_REFERENCE, (r"HeLa", r"QC_HeLa", r"iRT", r"bridge",
                            r"reference channel")),
    (BLANK, (r"blank", r"BLK", r"^B\d+", r"solvent",
             r"extraction[-_ ]?blank", r"process[-_ ]?blank", r"water")),
    (SYSTEM_SUITABILITY, (r"SST", r"sys[-_ ]?suit", r"cond", r"equil", r"wash")),
    (CALIBRANT, (r"CAL\d", r"STD", r"standard", r"IS", r"ISTD", r"NIST",
                 r"SRM1950", r"LTR")),
    (POOLED_QC, (r"^QC", r"_QC", r"PQC", r"pool", r"pooled", r"QCP", r"SQC",
                 r"QC[-_ ]?\d+")),
)

#: §01: *"applied to sample names **and** to any metadata column named `type`,
#: `sample_type`, `role`, `class`, `group`."* The restriction is not a
#: convenience — it is what stops the library reading `batch` values `B1`/`B2`
#: as `^B\d+`, i.e. as extraction blanks. Scanned columns are these plus the
#: sample-name column, and nothing else.
ROLE_COLUMN_NAMES = frozenset({"type", "sample_type", "role", "class", "group"})

#: Names a sample identifier is written under. Used to PREFER a column, never to
#: require one: a table whose id column is called `X.1` still gets read, by
#: shape, below.
SAMPLE_NAME_COLUMNS = frozenset({
    "sample", "sample_id", "sample_name", "name", "id", "injection",
    "injection_id", "file", "file_name", "raw_file", "run", "label"})

_BARE_WORD = re.compile(r"[A-Za-z0-9]+")


def _norm_name(column: Any) -> str:
    """A column name reduced to lowercase words joined by single underscores."""
    return re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_")


def _compile_role(pattern: str) -> Any:
    """One §01 pattern, with the boundary rule the pack leaves implicit.

    INVENTED, and load-bearing enough to be worth the paragraph. Six of the
    pack's patterns are anchored or structural — `^QC`, `_QC`, `^B\\d+`,
    `CAL\\d`, `QC[-_ ]?\\d+`, `sys[-_ ]?suit` — and are used exactly as written.
    The rest are bare words, and applied as unanchored substrings they are
    catastrophic: `IS` matches `HISTIDINE`, `cond` matches `condition`, `STD`
    matches `STDEV`, `water` matches `wastewater_1`. A pack that reports a
    histidine standard because a sample is named after an amino acid is the
    governing rule broken in the one place the app has promised it will not be.

    So a pattern that is **entirely alphanumeric** is given boundaries and every
    other pattern is used verbatim. The classification is derived from the
    pattern rather than hand-listed, so a pattern added later is bounded without
    anybody remembering to bound it.

    **The boundary blocks LETTERS and admits DIGITS**, which is the second
    version of this rule. Blocking both was tried first and it silenced the
    commonest way these names are actually written: `BLK03`, `STD3`, `SST1`,
    `LTR04` are all a token with an index glued to it, and a digit abutting a
    token is an index rather than a different word. Every hazard the rule exists
    for is a LETTER neighbour — `HISTIDINE` around `IS`, `condition` around
    `cond`, `STDEV` around `STD`, `wastewater` around `water` — so blocking
    letters keeps all of them out and lets the real names in.
    """
    if _BARE_WORD.fullmatch(pattern):
        return re.compile(r"(?<![A-Za-z])" + pattern + r"(?![A-Za-z])",
                          re.IGNORECASE)
    return re.compile(pattern, re.IGNORECASE)


_ROLE_RE: Dict[Tuple[str, str], Any] = {
    (family, pattern): _compile_role(pattern)
    for family, patterns in ROLE_PATTERNS for pattern in patterns}


def sample_name_column(df: pd.DataFrame) -> Optional[str]:
    """The column holding sample names, by name first and by shape second.

    Shape rather than uniqueness, because a duplicate sample id is one of the
    things §01 asks to be detected: requiring uniqueness here would make the
    duplicate-id detector unable to see the very table it is for.
    """
    if df is None or df.empty:
        return None
    text = [c for c in df.columns
            if not pd.api.types.is_numeric_dtype(df[c])
            and not pd.api.types.is_datetime64_any_dtype(df[c])]
    for column in text:
        if _norm_name(column) in SAMPLE_NAME_COLUMNS:
            return str(column)
    n = len(df)
    for column in text:
        s = df[column]
        try:
            distinct = int(s.nunique(dropna=True))
        except TypeError:                                  # unhashable cells
            continue
        if distinct >= 0.9 * n and s.astype(str).str.len().mean() <= 40:
            return str(column)
    return None


def sample_roles(df: pd.DataFrame) -> Dict[str, Any]:
    """§01's role library, applied. **The census, including the empty families.**

    Returns every family with its count, present or absent, because the absent
    ones are the half a reviewer reads: *"I couldn't find any pooled QC
    samples"* is only sayable by something that looked and can say what it
    looked at.

    One row gets at most one role — the first family in `ROLE_PATTERNS` order
    that matches — and each family records the column and the pattern that made
    the call, so the reading is inspectable rather than asserted.
    """
    empty = {
        "name_column": None, "role_columns": [], "scanned_columns": [],
        "families": {f: {"n": 0, "rows": [], "matches": []}
                     for f, _ in ROLE_PATTERNS},
        "present": [], "absent": [f for f, _ in ROLE_PATTERNS],
        "n_rows": 0, "n_biological": 0,
    }
    if df is None or df.empty:
        return empty

    name_column = sample_name_column(df)
    role_columns = [str(c) for c in df.columns
                    if _norm_name(c) in ROLE_COLUMN_NAMES
                    or _norm_name(c).replace("_", "") in
                    {n.replace("_", "") for n in ROLE_COLUMN_NAMES}]
    scanned = ([name_column] if name_column else []) + \
              [c for c in role_columns if c != name_column]
    if not scanned:
        out = dict(empty)
        out["n_rows"] = len(df)
        out["n_biological"] = len(df)
        return out

    values = {c: df[c].astype(str).where(df[c].notna(), "").tolist()
              for c in scanned}
    families: Dict[str, Dict[str, Any]] = {
        f: {"n": 0, "rows": [], "matches": []} for f, _ in ROLE_PATTERNS}

    for position in range(len(df)):
        for family, patterns in ROLE_PATTERNS:
            # EVERY column that matched, not the first. The row gets one role,
            # but §01's point is that the evidence lives in the sample name AND
            # in the run-type column, and a reading corroborated by both is
            # stronger than either — so recording only the first would throw
            # away the thing the two-place rule exists to produce.
            hits = []
            for column in scanned:
                text = values[column][position]
                if not text:
                    continue
                for pattern in patterns:
                    if _ROLE_RE[(family, pattern)].search(text):
                        hits.append((column, pattern, text))
                        break
            if hits:
                slot = families[family]
                slot["n"] += 1
                slot["rows"].append(position)
                known = [(m["column"], m["pattern"]) for m in slot["matches"]]
                for column, pattern, text in hits:
                    if (column, pattern) not in known:
                        slot["matches"].append(
                            {"column": column, "pattern": pattern,
                             "example": text})
                        known.append((column, pattern))
                break

    present = [f for f, _ in ROLE_PATTERNS if families[f]["n"]]
    absent = [f for f, _ in ROLE_PATTERNS if not families[f]["n"]]
    non_biological = sum(families[f]["n"] for f in present)
    return {
        "name_column": name_column,
        "role_columns": role_columns,
        "scanned_columns": scanned,
        "families": families,
        "present": present,
        "absent": absent,
        "n_rows": len(df),
        "n_biological": len(df) - non_biological,
    }


#: The list bound `GUIDED-209` requires. Row and column lists in a `params`
#: payload are cut here and every cut states `..._shown` beside `..._total`, so
#: a consumer can tell a short list from a truncated one.
_LIST_BOUND = 200


def _bounded(key: str, items: Sequence[Any],
             bound: int = _LIST_BOUND) -> Dict[str, Any]:
    """A list in a payload, with its bound stated beside it (`GUIDED-209`)."""
    items = list(items)
    return {key: items[:bound], f"{key}_shown": min(len(items), bound),
            f"{key}_total": len(items), f"{key}_bound": bound}


def _sentence(parts: Sequence[str]) -> str:
    """Clauses joined into one sentence, **without `str.capitalize`.**

    `capitalize()` upper-cases the first character and LOWER-CASES every other
    one, so a clause naming sample `S040` shipped it as `s040`. The finding was
    true about the count and false about the identifier, in a payload a user
    would search their run list with. Trap #7's shape at the smallest possible
    scale, and worth a named helper so it cannot come back.
    """
    text = "; ".join(p for p in parts if p)
    return (text[:1].upper() + text[1:] + ".") if text else ""


def _plural(n: int, one: str, many: str) -> str:
    """`n` with the right noun. A count that reads `1 samples` is sloppy in a
    surface whose whole argument is that it says precise things."""
    return f"{n:,} {one if n == 1 else many}"


def _roles_sentence(census: Dict[str, Any]) -> str:
    parts = []
    for family in census["present"]:
        slot = census["families"][family]
        match = slot["matches"][0]
        parts.append(
            f"{slot['n']:,} {ROLE_LABELS[family]} "
            f"(`{match['column']}` matches `{match['pattern']}`, "
            f"e.g. {match['example']!r})")
    return "; ".join(parts)


def _sample_roles_finding(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Rows that are instrument work rather than biology, named by family.

    **The class of error only the lens can see**, one level wider than
    `_pooled_qc`: that detector reads variance and can therefore see exactly one
    family, the one with enough replicate injections for a variance to mean
    anything. Blanks, calibrants and system-suitability injections come in twos
    and threes and are invisible to it, and they are the ones that most damage a
    model, because a solvent blank is a row of near-zero intensities that any
    fit will happily separate on.

    The two readings are independent and are meant to agree. Where the variance
    detector also fired, `params["corroborated_by"]` says so — two instruments
    agreeing is evidence, and a disagreement between them is worth seeing.
    """
    if not _is_assay_wide(df):
        return None
    census = sample_roles(df)
    if not census["present"]:
        return None
    n_non_biological = census["n_rows"] - census["n_biological"]
    columns = sorted({m["column"] for f in census["present"]
                      for m in census["families"][f]["matches"]})
    params: Dict[str, Any] = {
        "families": {f: census["families"][f]["n"] for f in census["present"]},
        "absent_families": census["absent"],
        "scanned_columns": census["scanned_columns"],
        "name_column": census["name_column"],
        "n_biological": census["n_biological"],
        "n_non_biological": n_non_biological,
        "matches": [m for f in census["present"]
                    for m in census["families"][f]["matches"]],
    }
    for family in census["present"]:
        params.update(_bounded(f"rows_{family}",
                               census["families"][family]["rows"]))
    if POOLED_QC in census["present"] and _pooled_qc(df) is not None:
        params["corroborated_by"] = "pack::metabolomics::pooled_qc"
    return _finding(
        "pack::metabolomics::sample_roles", "warning",
        f"{n_non_biological:,} of {census['n_rows']:,} rows are instrument "
        f"runs, not biological samples",
        (f"Reading the sample names and the run-type columns "
         f"({', '.join('`' + c + '`' for c in census['scanned_columns'])}) "
         f"against the field's naming conventions: {_roles_sentence(census)}. "
         f"That leaves {census['n_biological']:,} biological samples."),
        ("Quality-control, blank, standard and system-suitability injections "
         "are the instrument being checked, not people being measured. They "
         "belong in the quality assessment and out of the modeling rows, and "
         "the reading here is the naming convention rather than the values — "
         "so it is worth confirming against your run list."),
        confidence="high", pack=METABOLOMICS, marker="convention",
        evidence=ROLES_EVIDENCE,
        columns=columns,
        params=params)


#: §01's coaching sentence for the absent case, **quoted rather than
#: paraphrased**. It is the most valuable thing in this family: a reviewer reads
#: it as the tool refusing to compute three specific things and saying which,
#: and the last clause is the part that has a deadline attached to it.
NO_POOLED_QC_COACHING = (
    "I couldn't find any pooled QC samples. Pooled QCs — an aliquot of every "
    "sample, mixed, injected every 5–10 samples — are the field's standard "
    "evidence that your run was stable, and reviewers increasingly expect them "
    "(Broadhurst et al. 2018, Metabolomics 14:72; mQACC 2022). Without them I "
    "can't compute QC-RSD, D-ratio, or drift correction. If QCs were run but "
    "aren't in this file, add them now; they cannot be reconstructed later.")

#: The three things the absence makes impossible, named separately from the
#: sentence because a machine-readable payload that drops half of what the prose
#: said is trap #7 and this is exactly its shape.
WITHOUT_POOLED_QC = ("QC-RSD", "D-ratio", "drift correction")


def _no_pooled_qc(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """The refusal, with its reason — and it is the half a reviewer notices.

    `DOMAIN_SCIENCE.md` §03b: an app that says nothing about an absent control
    is indistinguishable from one that never looked. This looked, at named
    columns, with a named pattern library, and reports that it found nothing —
    which is a claim, and carries a badge like every other claim here.

    `offered` rather than `derived`: the ABSENCE is derived, and what follows
    from it is a choice the user makes about their own run. There is nothing to
    pre-select and no action to take inside the app.
    """
    if not _is_assay_wide(df):
        return None
    census = sample_roles(df)
    if POOLED_QC in census["present"]:
        return None
    # A table with nothing to read is not a table with no QCs. Silence over a
    # false assertion — `AGENT_ONBOARD.md` trap #9, in words rather than in a
    # return value.
    if not census["scanned_columns"]:
        return None
    return _finding(
        "pack::metabolomics::no_pooled_qc", "warning",
        "No pooled QC samples in this file",
        (NO_POOLED_QC_COACHING + " I looked at "
         + ", ".join("`" + c + "`" for c in census["scanned_columns"])
         + f" across all {census['n_rows']:,} rows, against the field's naming "
           f"conventions for pooled QCs."),
        ("Without pooled QCs there is no measurement of technical variation, "
         "so QC-RSD filtering, the D-ratio and drift correction cannot be "
         "computed here — not by this app and not by any other tool from this "
         "file. That is a limit on what the analysis can claim, and it belongs "
         "in the methods section rather than in a footnote."),
        confidence="high", pack=METABOLOMICS, marker="offered",
        evidence=POOLED_QC_EVIDENCE,
        columns=census["scanned_columns"],
        params={"scanned_columns": census["scanned_columns"],
                "roles_found": {f: census["families"][f]["n"]
                                for f in census["present"]},
                "cannot_compute": list(WITHOUT_POOLED_QC),
                "reconstructable": False,
                "n_rows": census["n_rows"]})

# ─────────────────────────────────────────────────────────────────────────────
# D2 · Run order, batch and design
# ─────────────────────────────────────────────────────────────────────────────

#: §01's list, verbatim, grouped by what each column IS. The pack gives one flat
#: list — `injection.order`, `inj_order`, `run_order`, `sequence`, `acq_order`,
#: `AcquisitionDateTime`, `batch`, `plate`, `well`, `position`, `plex`,
#: `TMT.channel`, `polarity` — and the grouping is this file's, because the
#: consequences differ: a missing run order disables half the diagnostics and a
#: missing `plate` disables none.
DESIGN_COLUMNS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("run_order", ("injection.order", "inj_order", "run_order", "sequence",
                   "acq_order")),
    ("timestamp", ("AcquisitionDateTime",)),
    ("batch", ("batch",)),
    ("plate", ("plate",)),
    ("well", ("well",)),
    ("position", ("position",)),
    ("plex", ("plex",)),
    ("tmt_channel", ("TMT.channel",)),
    ("polarity", ("polarity",)),
)

#: §01, second paragraph: *"Also detect: group/class column; **subject ID**, to
#: catch repeated measures (routinely missed); timepoint; known confounders
#: (age, sex, BMI, fasting status, medication, site, storage time, freeze-thaw
#: count)."* The names are this file's spellings of those concepts; the concepts
#: are the pack's.
STUDY_COLUMNS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("group", ("group", "class", "condition", "treatment", "arm", "phenotype")),
    ("subject", ("subject", "subject_id", "patient_id", "participant_id",
                 "individual", "donor", "person_id")),
    ("timepoint", ("timepoint", "time_point", "visit", "week", "day", "month")),
    ("confounder", ("age", "sex", "gender", "bmi", "fasting", "fasting_status",
                    "medication", "site", "storage_time", "freeze_thaw",
                    "freeze_thaw_count")),
)


def _name_matches(column: Any, token: str) -> bool:
    """Normalized equality, or the token as a leading or trailing word.

    Word-bounded rather than substring, for `_compile_role`'s reason one layer
    up: `age` as a substring matches `image_id` and `percentage`, and a design
    reader that calls an image id a confounder has asserted something false
    about the study.
    """
    name, token = _norm_name(column), _norm_name(token)
    return (name == token
            or name.startswith(token + "_")
            or name.endswith("_" + token)
            or ("_" + token + "_") in ("_" + name + "_"))


def design_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Every acquisition and design column §01 names, by role. Never empty-keyed.

    A column is claimed by the FIRST role that matches, so `batch` cannot also
    be a confounder and the counts below add up.
    """
    found: Dict[str, List[str]] = {role: [] for role, _ in
                                   DESIGN_COLUMNS + STUDY_COLUMNS}
    if df is None or df.empty:
        return found
    taken: set = set()
    for role, tokens in DESIGN_COLUMNS + STUDY_COLUMNS:
        for column in df.columns:
            if str(column) in taken:
                continue
            if any(_name_matches(column, token) for token in tokens):
                found[role].append(str(column))
                taken.add(str(column))
    return found


def _permutation_column(df: pd.DataFrame) -> Optional[str]:
    """A numeric column that is a PERMUTATION of the row positions.

    Factored out of `_acquisition_order` rather than reimplemented, and that is
    the point of it existing. `_no_run_order` asserts *"there is no run order in
    this file"*, and two independent readings of "is there a run order" is the
    arrangement in which the app says that sentence on a table where the other
    detector has just reported a run-order column. One reader, one answer.
    """
    n = len(df)
    for column in _numeric(df):
        s = df[column].dropna()
        if len(s) != n:
            continue
        try:
            values = np.sort(s.to_numpy())
        except (TypeError, ValueError):
            continue
        if not np.all(np.equal(np.mod(values, 1), 0)):
            continue
        if np.array_equal(values, np.arange(1, n + 1)) or \
           np.array_equal(values, np.arange(0, n)):
            return str(column)
    return None


#: A timestamp column has to parse as one for essentially every row before it is
#: called a timestamp. INVENTED — §01 says *"if an acquisition timestamp exists,
#: derive run order from it"* and does not say how sure to be. Set high because
#: the consequence is an ORDER, and an order derived from a column that is dates
#: for 70% of rows is an order that is wrong for the other 30% silently.
_TIMESTAMP_PARSE_SHARE = 0.95


def acquisition_timestamp(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """A column that is an acquisition time, and the run order it implies.

    Named first — `AcquisitionDateTime` and its spellings — then confirmed by
    parsing, because a column called `run_date` holding a study visit date is
    not an acquisition time and the name alone cannot tell them apart.
    """
    if df is None or df.empty:
        return None
    candidates = [str(c) for c in df.columns
                  if any(_name_matches(c, t) for t in
                         ("AcquisitionDateTime", "acquisition_datetime",
                          "acquisition_time", "acq_time", "acq_datetime",
                          "injection_time", "run_time", "datetime",
                          "timestamp"))]
    for column in candidates:
        s = df[column]
        if pd.api.types.is_numeric_dtype(s):
            continue
        try:
            parsed = pd.to_datetime(s, errors="coerce", format="mixed")
        except (TypeError, ValueError):                    # pragma: no cover
            continue
        share = float(parsed.notna().mean())
        if share < _TIMESTAMP_PARSE_SHARE:
            continue
        order = parsed.rank(method="first").astype("Int64")
        return {"column": column, "parsed_share": round(share, 3),
                "first": str(parsed.min()), "last": str(parsed.max()),
                "order": order}
    return None


def _acquisition_design(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """What this run's acquisition and design columns are, and what they permit.

    An inventory rather than a complaint, at `info`. §01's Presentation section
    asks for a data-inventory table whose first job is to become *"the seed of
    the manuscript's data-description sentence"*, and a sentence nobody can
    write is one nobody wrote down the inputs for.

    **Derives run order from a timestamp where one exists**, which is §01's one
    imperative in this family, and says that it derived it — a run order the app
    computed and presented as if it had been given is the record asserting
    something false about its own provenance.
    """
    if not _is_assay_wide(df):
        return None
    found = design_columns(df)
    stamp = acquisition_timestamp(df)
    permutation = _permutation_column(df)

    named_order = found["run_order"]
    structure = [role for role, _ in DESIGN_COLUMNS if found[role]]
    if stamp and "timestamp" not in structure:
        structure.append("timestamp")
    design_present = [role for role, _ in STUDY_COLUMNS
                      if role != "confounder" and found[role]]
    if not structure and not design_present:
        return None

    order_column, order_source = None, None
    if named_order:
        order_column, order_source = named_order[0], "named"
    elif stamp:
        order_column, order_source = stamp["column"], "derived_from_timestamp"
    elif permutation:
        order_column, order_source = permutation, "inferred_from_shape"

    said = []
    if order_column and order_source == "derived_from_timestamp":
        said.append(
            f"no run-order column, but `{order_column}` parses as an "
            f"acquisition time for {stamp['parsed_share']:.0%} of rows "
            f"({stamp['first']} to {stamp['last']}), so run order is derived "
            f"from it")
    elif order_column and order_source == "named":
        said.append(f"run order in `{order_column}`")
    elif order_column:
        said.append(f"no run-order column, but `{order_column}` runs 1 to "
                    f"{len(df):,} with every position used once")
    for role, _ in DESIGN_COLUMNS:
        if role in ("run_order", "timestamp") or not found[role]:
            continue
        said.append(f"{role.replace('_', ' ')} in "
                    + ", ".join("`" + c + "`" for c in found[role]))
    for role in design_present:
        said.append(f"{role} in " + ", ".join("`" + c + "`" for c in found[role]))
    if found["confounder"]:
        said.append("known confounders "
                    + ", ".join("`" + c + "`" for c in found["confounder"]))

    columns = [c for role, _ in DESIGN_COLUMNS + STUDY_COLUMNS
               for c in found[role]]
    if stamp and stamp["column"] not in columns:
        columns.append(stamp["column"])
    params: Dict[str, Any] = {
        "run_order_column": order_column,
        "run_order_source": order_source,
        "by_role": {role: found[role] for role in found if found[role]},
        "timestamp": {k: v for k, v in (stamp or {}).items() if k != "order"}
                     or None,
    }
    params.update(_bounded("columns", columns))
    return _finding(
        "pack::metabolomics::acquisition_design", "info",
        f"The acquisition and design columns in this run: "
        f"{len(columns):,} of them",
        "This run records " + "; ".join(said) + ".",
        ("Run order, batch and plate are what drift correction, batch "
         "correction and the run-order PCA overlay are computed against, and "
         "the group-by-batch crosstab is what shows a confound before it "
         "becomes a result. Nothing here is changed — this is the inventory the "
         "data-description sentence is written from."),
        confidence="high", pack=METABOLOMICS, marker="derived",
        evidence=DESIGN_EVIDENCE,
        columns=columns[:8],
        params=params)


#: §01: *"half the downstream diagnostics (drift, QC-RLSC, run-order PCA
#: overlay) become impossible."* Named separately from the sentence for trap
#: #7's reason — the structured payload is what everything downstream reads.
WITHOUT_RUN_ORDER = ("drift", "QC-RLSC", "run-order PCA overlay")


def _no_run_order(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """No run order, no timestamp, nothing shaped like either. **Said loudly.**

    §01 uses the words *"say so loudly"*, which is a specification and is met by
    saying what stops being possible rather than by an exclamation mark. Three
    named diagnostics, in the payload as well as in the prose.
    """
    if not _is_assay_wide(df):
        return None
    found = design_columns(df)
    if found["run_order"] or acquisition_timestamp(df) or _permutation_column(df):
        return None
    return _finding(
        "pack::metabolomics::no_run_order", "warning",
        "There is no run order in this file, and half the quality diagnostics "
        "need one",
        (f"No column here is named as an injection or acquisition order, none "
         f"parses as an acquisition timestamp, and none runs 1 to "
         f"{len(df):,} with every position used exactly once. I looked at all "
         f"{len(df.columns):,} columns."),
        ("Instrument drift is often the largest single variance component in a "
         "run, and every way of seeing it is ordered by injection: drift "
         "diagnostics, QC-RLSC correction and the run-order overlay on the PCA "
         "scores plot all become impossible without one. The order is in the "
         "instrument's sequence file — if you still have it, adding the column "
         "now recovers all three."),
        confidence="high", pack=METABOLOMICS, marker="derived",
        evidence=DESIGN_EVIDENCE,
        columns=[],
        params={"cannot_compute": list(WITHOUT_RUN_ORDER),
                "n_columns_examined": int(len(df.columns)),
                "n_rows": int(len(df)),
                "recoverable_from": "the instrument sequence file"})


def _repeated_subjects(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Subject IDs that repeat — §01's *"routinely missed"* — routed, not re-asked.

    **This is the grain question arriving through a second door**, and the whole
    design decision is what happens next. The app already asks it: question 3,
    `set_grain`, in the lockbox, with a contradiction detector and two terminal
    exits. Asking it again here because a lens happens to be metabolomics would
    be the app interrogating a user about something the record already holds,
    which this project has paid for before.

    So the reading is `grain.suggestion` — **the lockbox's own ranking, not a
    second one** — and the finding carries the pack's coaching plus a pointer to
    where the answer lives. It cannot become a question: `fix_kind="none"` makes
    that structural rather than a matter of restraint.

    The column named in `params` resolves in `grain.suggestion(df)["columns"]`
    by construction, because that is where it came from. `AGENT_ONBOARD.md` trap
    #3 is a fixture handing a collaborator an id that production cannot produce;
    taking the id FROM the collaborator is the arrangement in which that cannot
    happen.
    """
    if not _is_assay_wide(df):
        return None
    from turbotab import grain as _grain

    suggestion = _grain.suggestion(df)
    if not suggestion["columns"]:
        return None
    evidence = {e["column"]: e for e in suggestion["evidence"]}
    # THE NAME HAS TO CORROBORATE THE SHAPE, and this clause is a repair rather
    # than a precaution. The lockbox OFFERS candidates and asserts nothing about
    # them; this sentence says *"subject IDs repeat"*, which is a claim that the
    # column IS a subject id. Driven without the clause it made that claim about
    # `genomics_expression.csv` — *"60 samples from 28 subjects"* — on the
    # strength of a column whose shape merely repeats, and about
    # `survey_instrument.csv` in the same breath. §01 names the thing to detect
    # as **subject ID**; a roster-shaped column with no name to match is the
    # case the grain question exists to ask about, and asking it is what the app
    # already does.
    named = set(design_columns(df)["subject"])
    column = next((c for c in suggestion["columns"]
                   if c in evidence and c in named), None)
    if column is None:
        return None
    reading = evidence[column]
    n_subjects, n_samples = reading["n_distinct"], reading["n_rows"]
    if n_samples <= n_subjects:                            # pragma: no cover
        return None
    inflation = n_samples / n_subjects
    return _finding(
        "pack::metabolomics::repeated_subjects", "warning",
        f"Subject IDs repeat — {n_samples:,} samples from {n_subjects:,} "
        f"subjects",
        (f"`{column}` holds {n_subjects:,} distinct values across "
         f"{n_samples:,} rows, {reading['modal_rows_per']:,} rows per value for "
         f"{reading['regular_share']:.0%} of them. Treating these as "
         f"{n_samples:,} independent observations would inflate your apparent "
         f"sample size by about {inflation:.1f}× and inflate significance with "
         f"it."),
        ("This is the question the app has already asked — whether one person "
         "can appear in more than one row — and the answer you gave there is "
         "the one that governs. It decides how the held-out set is drawn, "
         "because a subject appearing in both halves makes the held-out "
         "estimate optimistic. Nothing is re-asked here; this is the same "
         "question seen from the assay side, and the reason it matters more "
         "in an assay is that the technical replicate and the repeat visit "
         "look identical in the table."),
        confidence="high", pack=METABOLOMICS, marker="offered",
        evidence=DESIGN_EVIDENCE,
        columns=[column],
        params={"routes_to": "set_grain",
                "grain_answer": _grain.PEOPLE_REPEAT,
                "group_column": column,
                "n_subjects": int(n_subjects),
                "n_samples": int(n_samples),
                "apparent_inflation": round(float(inflation), 2),
                "candidate_columns": list(suggestion["columns"]),
                "asks_nothing": True})

# ─────────────────────────────────────────────────────────────────────────────
# D3 · Value states
# ─────────────────────────────────────────────────────────────────────────────

def metabolite_columns(df: pd.DataFrame,
                       keep_degenerate: bool = False) -> List[str]:
    """The intensity block — the numeric columns that are measurements.

    Everything below reads *"the values"*, and which values is not obvious in a
    table that carries `run_order`, `age` and a binary outcome beside four
    hundred features. Getting it wrong is not cosmetic: a zero census over a
    block that includes a 0/1 outcome reports zeros in a table that has none,
    and a dynamic range computed over `run_order` reports 80 where the answer is
    89,000.

    Two exclusions, both derived from what the column IS rather than from a
    name list of its own: a column `design_columns` already claimed, and a
    column with at most two distinct values, which is an indicator and not an
    abundance.

    **`keep_degenerate` exists because the second exclusion ate the defect.** An
    all-zero feature and a constant feature both have exactly ONE distinct
    value, so the rule written to drop a 0/1 outcome column also dropped every
    column `_empty_blocks` is for — and it dropped them before the detector ran,
    so the detector reported *"1 empty sample"* on a table with eight all-zero
    features and said nothing false while missing most of the finding. With the
    flag set the floor becomes "exactly two distinct values", which still drops
    an indicator and keeps the degenerate columns that are the subject.

    This is `AGENT_ONBOARD.md` §07's third variant seen from the production
    side: the assertion was right and the input to it was wrong, which is why
    reading the detector never finds it and driving a fixture does.
    """
    if df is None or df.empty:
        return []
    claimed = {c for cols in design_columns(df).values() for c in cols}
    out = []
    for column in _numeric(df):
        if column in claimed:
            continue
        try:
            distinct = int(df[column].nunique(dropna=True))
        except TypeError:                                  # pragma: no cover
            continue
        if distinct == 2 or (distinct <= 2 and not keep_degenerate):
            continue
        out.append(column)
    return out


def _block(df: pd.DataFrame, minimum: int = 30) -> Optional[Tuple[List[str], np.ndarray]]:
    """The intensity block as a float array, or None if there is not one."""
    columns = metabolite_columns(df)
    if len(columns) < minimum:
        return None
    with np.errstate(all="ignore"):
        values = df[columns].to_numpy(dtype=float, na_value=np.nan)
    return columns, values


#: §01 names four vendors and what each writes into a cell it could not
#: quantify. Carried as data rather than as a sentence so the payload says
#: everything the prose does — trap #7 is the machine-readable form being the
#: lossier of the two, and a vendor list flattened into one string is exactly
#: that.
ZERO_CONVENTIONS: Tuple[Tuple[str, str], ...] = (
    ("XCMS", "fillPeaks writes a small number, not a zero"),
    ("MZmine", "writes 0"),
    ("MaxQuant", "writes 0, meaning not quantified"),
    ("Progenesis", "writes 0"),
)

#: INVENTED. An export that writes zeros for non-detections writes a lot of
#: them; one zero in a table of 2,700 cells is a value, not a convention. Driven
#: over the fixtures, `wide_assay.csv` produced *"1 zeros across 1 features"* —
#: true, ungrammatical, and an interruption about nothing. The floor is a share
#: rather than a count so it does not scale wrongly with the size of the panel.
_SYSTEMATIC_ZERO_SHARE = 0.01


def _zeros_or_missing(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Zeros in the intensity block, and the question the app will not answer.

    §01 is unambiguous and it is the reason this is a report rather than a
    default: *"The pack must ask: do zeros here mean 'not detected' or 'true
    zero'? Defaulting wrong corrupts every downstream step."*

    So this states the count, states that four widely-used tools disagree about
    what they mean, and states that nothing here has been assumed. It does not
    add a question — guard #1 forbids a pack inventing a card type and
    `fix_kind="none"` makes that structural — and the honest form of a question
    a pack may not ask is a report that says the app has not decided.
    """
    read = _block(df)
    if read is None:
        return None
    columns, values = read
    zeros = np.equal(values, 0.0) & np.isfinite(values)
    n_zeros = int(zeros.sum())
    finite = int(np.isfinite(values).sum())
    n_blank_all = int(np.isnan(values).sum())
    if n_zeros / max(finite + n_blank_all, 1) < _SYSTEMATIC_ZERO_SHARE:
        return None
    per_feature = zeros.sum(axis=0)
    with_zeros = [c for c, k in zip(columns, per_feature) if k]
    n_blank = int(np.isnan(values).sum())
    both = n_blank > 0
    params: Dict[str, Any] = {
        "n_zeros": n_zeros,
        "n_cells": finite + n_blank,
        "share_zero": round(n_zeros / max(finite + n_blank, 1), 5),
        "n_features_with_zeros": len(with_zeros),
        "n_features": len(columns),
        "n_blank_cells": n_blank,
        "blanks_and_zeros_coexist": both,
        "vendor_conventions": [{"tool": t, "writes": w}
                               for t, w in ZERO_CONVENTIONS],
        "not_defaulted": True,
        "decides": ("whether these cells are non-detections or measured zeros, "
                    "which changes filtering, imputation and every ratio "
                    "computed after them"),
    }
    params.update(_bounded("features_with_zeros", with_zeros))
    coexist = (
        f"There are also {n_blank:,} genuinely blank cells, so this table uses "
        f"both — which usually means the zeros and the blanks came from "
        f"different steps and mean different things."
        if both else
        "There are no blank cells at all, which is itself informative: a table "
        "with zeros and no blanks has usually had its non-detections written "
        "as zeros by the export.")
    return _finding(
        "pack::metabolomics::zeros_or_missing", "warning",
        f"{n_zeros:,} zeros across {len(with_zeros):,} features — and nothing "
        f"has assumed what they mean",
        (f"{n_zeros:,} of {finite + n_blank:,} cells in the "
         f"{len(columns):,}-feature block are exactly zero. " + coexist + " "
         "The tools disagree: "
         + "; ".join(f"{tool} {writes}" for tool, writes in ZERO_CONVENTIONS)
         + "."),
        ("A zero that means \"below the detection limit\" and a zero that means "
         "\"measured as none\" want opposite treatments — the first is imputed "
         "from the detection limit, the second is a real value — and defaulting "
         "wrong corrupts every step after it. Nothing here has defaulted: the "
         "app needs to be told which export wrote this file."),
        confidence="high", pack=METABOLOMICS, marker="offered",
        evidence=VALUE_STATE_EVIDENCE,
        columns=with_zeros[:8],
        params=params)


#: §01: *"a max below ~40 with a positive min and low dynamic range"*. The `~`
#: is the pack's; the constant is this file's reading of it.
_TRANSFORMED_MAX = 40.0

#: §01: *"raw untargeted intensities span 10^2–10^9. A ratio below 10^2 means
#: something has already been done to the data."*
_RAW_DYNAMIC_RANGE = 100.0

#: INVENTED, and it is the repair for a false statement this detector made
#: before it existed. §01's range readings — *"a max below ~40"*, *"a ratio
#: below 10²"* — are about ABUNDANCES. Driven across every fixture under a
#: stated metabolomics lens, they fired on `survey_instrument.csv`: a block of
#: 41 Likert items scored 1–5 has a maximum of 5 and a range of 5×, and the app
#: told a survey researcher their responses had already been log-transformed.
#: A block that is almost entirely whole numbers is a coded instrument or a
#: count matrix, and neither is an abundance, so neither reading applies to it.
_INTEGRAL_SHARE_MAX = 0.9

#: §01: *"or column means ≈ 0"*. INVENTED — the pack does not operationalize
#: "≈ 0", and a centered column is one whose mean is small RELATIVE TO ITS OWN
#: SPREAD, because an absolute threshold would call a column of small
#: concentrations centered. Named here rather than buried in an expression so a
#: reader can disagree with it.
_CENTERED_MEAN_RATIO = 0.1
#: And the share of features that must be centered before the table is. Also
#: INVENTED. Half, because scaling is applied to a whole block or to none of it.
_CENTERED_SHARE = 0.5


def transformation_signals(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """§01's three readings of *"this has already been transformed"*, measured.

    Returns the measurements whether or not any of them fired, and separately
    from the finding that reports them, because the readings are wanted by
    things that are not that finding: the sibling fixtures are checked against
    them, and a later step deciding whether a log transform is legal wants the
    measurement rather than the card.
    """
    read = _block(df)
    if read is None:
        return None
    columns, values = read
    finite = values[np.isfinite(values)]
    if not finite.size:
        return None
    positive = finite[finite > 0]
    ratio = (float(positive.max() / positive.min())
             if positive.size and positive.min() > 0 else None)
    with np.errstate(all="ignore"):
        means = np.nanmean(values, axis=0)
        sds = np.nanstd(values, axis=0)
        usable = np.isfinite(means) & np.isfinite(sds) & (sds > 0)
        centered = (np.abs(means[usable]) < _CENTERED_MEAN_RATIO * sds[usable])
    centered_share = float(centered.mean()) if centered.size else 0.0

    integral_share = float(np.mean(np.equal(np.mod(finite, 1), 0)))
    abundances = integral_share < _INTEGRAL_SHARE_MAX

    n_negative = int((finite < 0).sum())
    signals = []
    if n_negative:
        signals.append("negative_values")
    compressed = (abundances and ratio is not None
                  and ratio < _RAW_DYNAMIC_RANGE and finite.min() > 0)
    if compressed and finite.max() < _TRANSFORMED_MAX:
        signals.append("compressed_max")
    # THE RANGE IS A SIGNAL OF THIS FINDING RATHER THAN A FINDING OF ITS OWN,
    # and the merge is a closer reading of §01 than the split was. The pack's
    # two bullets are *"a max below ~40 with a positive min and low dynamic
    # range"* and *"a ratio below 10² means something has already been done to
    # the data"* — the second is the first's threshold, stated once and reusable
    # on its own. Built as a separate detector it was a capability with **no
    # fixture in this repository able to fire it**, because the shape it
    # uniquely covered (a compressed range with a maximum above 40) is a
    # targeted panel in concentration units and nothing here is one. Merged, the
    # weaker claim is still made — it composes a different sentence below when
    # it is the only signal — and it is reachable.
    elif compressed:
        signals.append("compressed_range")
    if centered_share >= _CENTERED_SHARE:
        signals.append("centered_columns")
    return {
        "columns": columns, "signals": signals,
        "n_negative": n_negative,
        "min": float(finite.min()), "max": float(finite.max()),
        "dynamic_range": ratio,
        "integral_share": round(integral_share, 3),
        "reads_as_abundances": abundances,
        "centered_share": round(centered_share, 3),
        "n_features": len(columns),
    }


def _already_transformed(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Values that have already had something done to them. **Warned hard.**

    §01: *"Warn hard; a second log transform is a silent catastrophe."* Silent
    is the operative word — a log of a log produces numbers, and every plot
    downstream renders. `critical`, because the consequence is a whole analysis
    that looks finished and is wrong.

    **The marker moves with the evidence and the id does not.** A negative value
    in an abundance table is derived: an ion count below zero is not a
    measurement, so nothing else explains it. A compressed maximum is a reading
    with an innocent alternative — a targeted panel reported in µM sits in the
    same range — so where that is the only signal the app offers the reading
    instead of asserting it. `atwater_finding`'s rule holds: the verdict is a
    parameter and the id is one thing, because `LooksFor` binds to an id.
    """
    read = transformation_signals(df)
    if read is None or not read["signals"]:
        return None
    derived = "negative_values" in read["signals"]
    said = []
    if derived:
        said.append(f"{read['n_negative']:,} values are negative (the smallest "
                    f"is {read['min']:,.3g})")
    if "compressed_max" in read["signals"]:
        said.append(f"the largest value in the whole block is "
                    f"{read['max']:,.3g}, with a positive minimum of "
                    f"{read['min']:,.3g} and a dynamic range of only "
                    f"{read['dynamic_range']:,.1f}×")
    elif "compressed_range" in read["signals"]:
        said.append(f"the values span only {read['dynamic_range']:,.1f}× "
                    f"({read['min']:,.4g} to {read['max']:,.4g}) where a raw "
                    f"run spans 10² to 10⁹, so something has been done to them "
                    f"— though none of the specific signatures of a log "
                    f"transform is present, so I can't say what")
    if "centered_columns" in read["signals"]:
        said.append(f"{read['centered_share']:.0%} of features have a mean "
                    f"within {_CENTERED_MEAN_RATIO:.0%} of their own standard "
                    f"deviation of zero, which is what centering leaves behind")
    params = {"signals": read["signals"], "n_negative": read["n_negative"],
              "min": read["min"], "max": read["max"],
              "dynamic_range": read["dynamic_range"],
              "centered_share": read["centered_share"],
              "n_features": read["n_features"],
              "raw_range_floor": _RAW_DYNAMIC_RANGE,
              "transformed_max_ceiling": _TRANSFORMED_MAX}
    return _finding(
        "pack::metabolomics::already_transformed", "critical",
        "These values look like they have already been transformed",
        ("Raw untargeted intensities span roughly 10² to 10⁹ and are strictly "
         "positive. Here, " + "; and ".join(said) + "."),
        ("A second log transform on already-logged data is a silent "
         "catastrophe: it produces numbers, every plot still renders, and the "
         "fold changes underneath are wrong by an amount nothing in the output "
         "reveals. Negative values also break the log outright. Before "
         "anything is transformed here, the app needs to know what was already "
         "done to this table and by which tool."),
        confidence="high", pack=METABOLOMICS,
        marker="derived" if derived else "offered",
        evidence=VALUE_STATE_EVIDENCE,
        columns=[],
        params=params)


#: pandas renames a repeated column label to `name.1`, `name.2`, … on read. A
#: duplicate feature id therefore never ARRIVES as a duplicate: it arrives
#: renamed, silently, and every downstream count of "how many features" is one
#: too many. Both forms are read here, because the first is what a constructed
#: frame carries and the second is what a real CSV carries, and a detector that
#: saw only the first would be untestable against any file on disk.
_MANGLED = re.compile(r"^(?P<stem>.+)\.(?P<n>[1-9]\d*)$")


def duplicate_ids(df: pd.DataFrame) -> Dict[str, Any]:
    """Repeated feature labels and repeated sample labels, and how they arrived."""
    labels = [str(c) for c in df.columns]
    seen: Dict[str, int] = {}
    for label in labels:
        seen[label] = seen.get(label, 0) + 1
    literal = sorted(l for l, k in seen.items() if k > 1)
    stems = set(labels)
    mangled = sorted(
        label for label in labels
        if (m := _MANGLED.match(label)) and m.group("stem") in stems)
    name_column = sample_name_column(df)
    rows: List[str] = []
    if name_column is not None:
        counts = df[name_column].value_counts(dropna=True)
        rows = sorted(str(v) for v, k in counts.items() if k > 1)
    return {"literal_features": literal, "mangled_features": mangled,
            "sample_column": name_column, "duplicate_samples": rows}


def _duplicate_ids(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """§01's *"duplicate feature/sample IDs"*, in both of the shapes they arrive.

    A duplicate id is not a nuisance here. A repeated feature is counted twice
    in every "we measured N metabolites" sentence and twice in every multiple-
    testing correction, and a repeated sample id is two injections the join
    downstream will fan out or silently pick one of.
    """
    if not _is_assay_wide(df):
        return None
    read = duplicate_ids(df)
    if not (read["literal_features"] or read["mangled_features"]
            or read["duplicate_samples"]):
        return None
    said = []
    if read["literal_features"]:
        said.append(_plural(len(read["literal_features"]),
                            "feature label appears", "feature labels appear")
                    + " more than once")
    if read["mangled_features"]:
        said.append(
            _plural(len(read["mangled_features"]),
                    "column arrived", "columns arrived")
            + " renamed with a numeric suffix — `"
            + "`, `".join(read["mangled_features"][:4])
            + "` — which is what the reader does when a file repeats a column "
              "name, so the file itself carries duplicates")
    if read["duplicate_samples"]:
        said.append(
            _plural(len(read["duplicate_samples"]),
                    "sample id appears", "sample ids appear")
            + f" in `{read['sample_column']}` on more than one row — "
            + ", ".join(repr(s) for s in read["duplicate_samples"][:4]))
    params: Dict[str, Any] = {"sample_column": read["sample_column"]}
    params.update(_bounded("literal_features", read["literal_features"]))
    params.update(_bounded("mangled_features", read["mangled_features"]))
    params.update(_bounded("duplicate_samples", read["duplicate_samples"]))
    columns = (read["literal_features"] + read["mangled_features"]
               + ([read["sample_column"]] if read["duplicate_samples"]
                  and read["sample_column"] else []))
    return _finding(
        "pack::metabolomics::duplicate_ids", "warning",
        "Some features or samples are in this table more than once",
        _sentence(said),
        ("A feature counted twice is counted twice in the number of metabolites "
         "you report and twice in every multiple-testing correction, which "
         "makes the correction slightly wrong in the anticonservative "
         "direction. A sample id on two rows is two injections of one sample, "
         "and whichever way a later join resolves it — fanning out, or keeping "
         "one arbitrarily — it does so without saying which."),
        confidence="high", pack=METABOLOMICS, marker="derived",
        evidence=VALUE_STATE_EVIDENCE,
        columns=columns[:8],
        params=params)


def _empty_blocks(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """All-zero features, constant features, all-zero samples. §01's fourth bullet.

    One finding for the three because they are one condition — a row or column
    carrying no information — and three cards saying *"this contributes
    nothing"* about the same table is three-quarters noise. Each is counted
    separately in the payload, because they are repaired differently: a constant
    feature is dropped, an empty sample is a failed injection somebody wants to
    know about.
    """
    # `keep_degenerate=True`, and it is the whole detector. The default block
    # drops any column with one distinct value, which is every column this
    # finding is about.
    columns = metabolite_columns(df, keep_degenerate=True)
    if len(columns) < 30:
        return None
    with np.errstate(all="ignore"):
        values = df[columns].to_numpy(dtype=float, na_value=np.nan)
        finite = np.isfinite(values)
        nonzero = finite & (values != 0.0)

        feature_has_value = nonzero.any(axis=0)
        empty_features = [c for c, ok in zip(columns, feature_has_value)
                          if not ok]
        constant = []
        for index, column in enumerate(columns):
            if not feature_has_value[index]:
                continue
            observed = values[:, index][finite[:, index]]
            if observed.size and float(np.nanmax(observed)) == float(
                    np.nanmin(observed)):
                constant.append(column)

        # A SAMPLE IS EMPTY WHEN NOTHING INFORMATIVE WAS MEASURED IN IT, and
        # the degenerate columns are excluded from that reading rather than
        # counted toward it. The case is concrete: a gap filler writes the same
        # constant into every row of a feature it could not detect, including
        # the row of an injection that failed outright. Counting that constant
        # as "this sample has a value" reports the failed injection as fine,
        # which is the app asserting something false about the one row a person
        # most needs to see. A constant column carries no information about any
        # sample by construction, so it cannot be evidence that one is present.
        informative = np.array(
            [c not in set(empty_features) | set(constant) for c in columns])
        sample_has_value = (nonzero[:, informative].any(axis=1)
                            if informative.any() else nonzero.any(axis=1))
    empty_samples = [int(i) for i, ok in enumerate(sample_has_value) if not ok]
    if not (empty_features or constant or empty_samples):
        return None

    name_column = sample_name_column(df)
    empty_sample_names = ([str(df[name_column].iloc[i]) for i in empty_samples]
                          if name_column else [])
    said = []
    if empty_features:
        said.append(f"{len(empty_features):,} of {len(columns):,} features are "
                    f"zero or blank in every sample")
    if constant:
        said.append(f"{len(constant):,} more hold one value in every sample "
                    f"they were observed in")
    if empty_samples:
        said.append(_plural(len(empty_samples),
                            "sample is", "samples are")
                    + " zero or blank across every feature that carries "
                      "information"
                    + (" — " + ", ".join(repr(n) for n in empty_sample_names[:4])
                       if empty_sample_names else ""))
    params: Dict[str, Any] = {
        "n_features": len(columns), "n_samples": int(len(df)),
        "sample_column": name_column}
    params.update(_bounded("empty_features", empty_features))
    params.update(_bounded("constant_features", constant))
    params.update(_bounded("empty_sample_rows", empty_samples))
    params.update(_bounded("empty_sample_names", empty_sample_names))
    return _finding(
        "pack::metabolomics::empty_blocks", "warning",
        "Some rows and columns here carry no information at all",
        _sentence(said),
        ("A feature that is zero everywhere and a feature that holds one value "
         "everywhere cannot separate anything, and both survive every filter "
         "that is written in terms of variance ratios because their variance is "
         "zero rather than small. A sample that is empty across every feature "
         "is a failed injection or a merge that matched nothing, and it is "
         "worth knowing which before it becomes a row in a model."),
        confidence="high", pack=METABOLOMICS, marker="derived",
        evidence=VALUE_STATE_EVIDENCE,
        columns=(empty_features + constant)[:8],
        params=params)


#: Polarity markers as a merged export writes them: appended to the feature
#: name. `mz_0001_pos`, `784.5876@8.21_neg`, `FT0012_positive`.
_POLARITY_TOKENS = {
    "positive": frozenset({"pos", "positive", "esipos", "pmode", "p"}),
    "negative": frozenset({"neg", "negative", "esineg", "nmode", "n"}),
}

#: **The marker has to be the LAST token of the feature name**, and the
#: restriction is a repair rather than a precaution. Read as "appears anywhere",
#: `negative_control_probe` is a negative-mode feature and `position_marker` is
#: a positive-mode one — a control probe on an array would put both in one
#: table and produce a confident, false report that the run carries two
#: polarities. A merged export appends the mode; a name that merely contains the
#: word is describing something else. The cost is that a PREFIX convention
#: (`pos_mz_0001`) is not read, and that is named in the test file's
#: `SHAPES_NOT_COVERED` rather than guessed at.
def _polarity_of(name: str) -> Optional[str]:
    # THE READER'S OWN RENAMING IS STRIPPED FIRST, and the two detectors
    # disagreeing is what surfaced it. `mz_0011_pos.1` is a positive-mode
    # column that `read_csv` renamed, and its last token is `1` — so the
    # polarity census reported 196 positive features on a table where
    # `duplicate_ids` had just reported 202 columns, six of them renamed. Two
    # findings about one file, disagreeing about what its columns are.
    mangled = _MANGLED.match(str(name))
    stem = mangled.group("stem") if mangled else str(name)
    tokens = [t for t in re.split(r"[^A-Za-z0-9]+", stem) if t]
    if not tokens:
        return None
    last = tokens[-1].lower()
    for mode, markers in _POLARITY_TOKENS.items():
        if last in markers:
            return mode
    return None


def _polarity_in_value(value: str) -> Optional[str]:
    """A polarity column's VALUE, where any token may carry the mode.

    Different from the name rule above and deliberately so: `ESI+ positive` and
    `neg` are both whole answers to *"which polarity"*, whereas a feature name
    is a compound in which the mode is the suffix. One rule over both would
    have to be the loose one, and the loose one is what reads a control probe
    as an acquisition.
    """
    for token in re.split(r"[^A-Za-z0-9]+", str(value)):
        for mode, markers in _POLARITY_TOKENS.items():
            if token.lower() in markers:
                return mode
    return None


def ion_modes(df: pd.DataFrame) -> Dict[str, Any]:
    """Which polarities this table carries, from feature names and from a column.

    Two independent readings because a merged export writes the mode into the
    feature name and an unmerged one writes it into a `polarity` column, and a
    detector that read only one of them would be silent on half the real files.
    """
    # `keep_degenerate=True`: a polarity census counts feature COLUMNS, and a
    # feature that came out all-zero is still a feature that was in the merged
    # list. Counting the informative ones would report 188 negative-mode
    # features where the file carries 196.
    columns = metabolite_columns(df, keep_degenerate=True)
    by_name: Dict[str, List[str]] = {"positive": [], "negative": []}
    for column in columns:
        mode = _polarity_of(column)
        if mode:
            by_name[mode].append(column)
    from_column: Dict[str, int] = {}
    polarity_columns = design_columns(df)["polarity"]
    for column in polarity_columns:
        for value, count in df[column].value_counts(dropna=True).items():
            mode = _polarity_in_value(value)
            if mode:
                from_column[mode] = from_column.get(mode, 0) + int(count)
    return {"by_name": by_name, "polarity_columns": polarity_columns,
            "from_column": from_column,
            "modes": sorted({m for m in ("positive", "negative")
                             if by_name[m] or from_column.get(m)})}


def _ion_modes(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Both polarities in one table. §01's last value-state bullet.

    `CONVENTION`, and it is the badge doing its job. That the two modes are
    normalized separately and then merged is what the field does; it is not a
    result anybody has established, and the merge strategy genuinely changes
    the answer. Stating it as convention is the difference between telling a
    user what is normally done and telling them what is true.
    """
    if not _is_assay_wide(df):
        return None
    read = ion_modes(df)
    if len(read["modes"]) < 2:
        return None
    counts = {m: len(read["by_name"][m]) for m in read["modes"]}
    where = ("the feature names" if all(counts.values())
             else f"`{read['polarity_columns'][0]}`")
    params: Dict[str, Any] = {"modes": read["modes"],
                              "n_features_by_mode": counts,
                              "polarity_columns": read["polarity_columns"],
                              "rows_by_mode": read["from_column"]}
    for mode in read["modes"]:
        params.update(_bounded(f"features_{mode}", read["by_name"][mode]))
    return _finding(
        "pack::metabolomics::ion_modes", "info",
        "Both ion modes are in this one table",
        (f"Positive and negative mode features both appear, read from {where}: "
         + ", ".join(f"{counts[m]:,} {m}" for m in read["modes"])
         + ". They are two separate acquisitions of the same samples."),
        ("The field convention is to normalize each polarity separately and "
         "merge afterwards, because the two acquisitions have different "
         "response ranges and different drift. Normalizing across the merged "
         "table treats them as one measurement scale. The merge strategy "
         "changes the result, so it is worth stating in the methods section "
         "either way — this is a convention rather than a settled finding."),
        confidence="high", pack=METABOLOMICS, marker="offered",
        evidence=VALUE_STATE_CONVENTION,
        columns=[],
        params=params)


# ── dietary ──────────────────────────────────────────────────────────────────

COMPOSITIONAL_EVIDENCE = Evidence(
    status=SETTLED,
    source=("research/NUTRITION_PACK.md#05 · Compositional structure and "
            "substitution modeling"))

# CONVENTION, not SETTLED, and the research is the reason rather than caution.
# §02 lists the fixed-kcal screens as competing conventions — Willett's
# 500/3,500 for women and 800/4,200 for men, sex-neutral 500–5,000, sex-neutral
# 500–3,500 — and says outright that *"the conventions genuinely differ across
# literatures"*. A SETTLED badge over one of four circulating bands would be the
# app asserting a consensus the field does not have.
IMPLAUSIBLE_INTAKE_EVIDENCE = Evidence(
    status=CONVENTION_STATUS,
    source="research/NUTRITION_PACK.md#02 · Implausible intake exclusions")

ENERGY_ADJUSTMENT_EVIDENCE = Evidence(
    status=SETTLED,
    source=("research/NUTRITION_PACK.md#04 · Energy adjustment — the "
            "methodological signature"))

ENERGY_ADJUSTMENT_CLAIMS = (
    Claim("adjustment_is_needed",
          "Every nutrient association is confounded by total intake, so an "
          "energy adjustment is needed. That is not in dispute.",
          ENERGY_ADJUSTMENT_EVIDENCE),
    Claim("which_model",
          "The residual method is the default form and nutrient density is "
          "offered beside it; which one to use is a convention rather than a "
          "fact.",
          Evidence(
              status=CONVENTION_STATUS,
              source=("research/NUTRITION_PACK.md#04 · Energy adjustment — the "
                      "methodological signature"))),
)


def _compositional(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Columns that sum to a constant. Parts of a whole.

    Correlation between parts of a whole is negatively biased by construction,
    so this **gates the collinearity figure** rather than adding a step — a
    correlation matrix over compositional parts is not a figure with a caveat,
    it is a figure that cannot be read.

    Bounded on purpose: the subset search runs only when there are twelve or
    fewer plausible parts. A compositional test over 400 assay features is a
    different and harder problem, and a detector that pretended to do it would
    be guessing at exactly the scale where guessing is least visible.
    """
    cols = _numeric(df)
    for total in (100.0, 1.0):
        candidates = [c for c in cols
                      if float(df[c].min(skipna=True)) >= -1e-9
                      and float(df[c].max(skipna=True)) <= total * 1.02
                      and float(df[c].mean(skipna=True)) > total * 0.005]
        if not 3 <= len(candidates) <= 12:
            continue
        for size in range(3, min(len(candidates), 6) + 1):
            for subset in itertools.combinations(candidates, size):
                sums = df[list(subset)].sum(axis=1, skipna=False)
                close = (sums - total).abs() <= total * 0.005
                if float(close.mean()) >= 0.95:
                    return _finding(
                        "pack::dietary::compositional", "warning",
                        f"{len(subset)} columns sum to {total:g} on every row",
                        ("`" + "`, `".join(subset) + "` add to "
                         f"{total:g} for {float(close.mean()):.0%} of rows."),
                        ("These are compositional — parts of a whole — and "
                         "ordinary correlation between them is not "
                         "interpretable: raising one necessarily lowers "
                         "another, so the negative correlation is arithmetic "
                         "rather than dietary. The collinearity figure is drawn "
                         "on log-ratios rather than on the parts, and the parts "
                         "are not offered as independent predictors."),
                        confidence="high", pack=DIETARY, marker="derived",
                        evidence=COMPOSITIONAL_EVIDENCE,
                        columns=list(subset),
                        params={"columns": list(subset), "total": total,
                                "share_closing": round(float(close.mean()), 3),
                                "gates": "collinearity_figure"})
    return None


def _reference_column(df: pd.DataFrame, variable: str) -> Optional[str]:
    """A column the engine's own reference matcher recognizes as `variable`.

    Goes through `physiology_reference.match_variable_key`, which is exact
    against the key or one of its aliases after case and separators are
    stripped — never by substring, which is what let `hba1c_proxy` inherit
    HbA1c's bounds. Borrowing the vetted matcher is the opposite of adding a
    third name list.
    """
    try:
        from ml.physiology_reference import load_reference_bundle, match_variable_key
        reference = load_reference_bundle()["nhanes"]
    except Exception:                                      # pragma: no cover
        return None
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if match_variable_key(str(c), reference) == variable:
            return str(c)
    return None


# Adult daily intake outside which a 24-hour recall is treated as a reporting
# error rather than a diet. Stated as a convention with its numbers visible,
# because it is one — and it is OFFERED rather than applied, because it changes
# N and an exclusion that changes N is an eligibility criterion the user states.
_PLAUSIBLE_KCAL = (500.0, 5000.0)


def _implausible_intake(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    low, high = _PLAUSIBLE_KCAL
    col = _reference_column(df, "kcal")
    if col is None:
        return None
    s = pd.to_numeric(df[col], errors="coerce")
    flagged = s[(s < low) | (s > high)]
    if flagged.empty:
        return None
    return _finding(
        "pack::dietary::implausible_intake", "info",
        f"{len(flagged):,} records report an implausible daily intake",
        (f"`{col}` is below {low:g} on {int((s < low).sum()):,} record(s) and "
         f"above {high:g} on {int((s > high).sum()):,}. Observed range "
         f"{float(s.min()):,.0f} to {float(s.max()):,.0f}."),
        ("These are possible days and poor estimates: a recall of 300 kcal is "
         "under-reporting rather than starvation. Excluding them is an "
         "eligibility criterion, which changes N and is reported in participant "
         "flow — so it is offered here and never applied. Nothing is filtered "
         "unless you say so."),
        confidence="medium", pack=DIETARY, marker="offered",
        evidence=IMPLAUSIBLE_INTAKE_EVIDENCE,
        columns=[col],
        params={"column": col, "minimum": low, "maximum": high,
                "n_flagged": int(len(flagged)),
                "offers": "eligibility_criterion"})


def _energy_adjustment(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    col = _reference_column(df, "kcal")
    if col is None:
        return None
    others = [c for c in _numeric(df) if c != col]
    if len(others) < 3:
        return None
    return _finding(
        "pack::dietary::energy_adjustment", "warning",
        "Nutrient associations need energy adjustment",
        (f"`{col}` is total energy, and {len(others):,} other numeric columns "
         f"are candidate nutrients."),
        ("Every nutrient association is confounded by total intake — people who "
         "eat more of everything eat more of anything. That the adjustment is "
         "needed is not in dispute. The residual method is the default form "
         "because it decorrelates the nutrient from energy explicitly, which "
         "makes the resulting coefficient interpretable; nutrient density is "
         "offered beside it, and that choice is a convention rather than a "
         "fact."),
        confidence="high", pack=DIETARY, marker="derived",
        evidence=ENERGY_ADJUSTMENT_EVIDENCE, claims=ENERGY_ADJUSTMENT_CLAIMS,
        columns=[col],
        params={"energy_column": col, "columns": [col],
                "default_form": "residual",
                "alternative": "nutrient_density"})


# ── dietary · the nutrition module's detectors, reaching the live path ───────
#
# `GUIDED-058`. `packs.findings(df, lens)` is wired — `engine.py:710` and
# `project.py:815` both call it — and `turbotab/nutrition.py` was imported by
# its own tests and by nothing else. Four detectors and a refusal, unreachable.
#
# **The import is deferred, and it has to be.** `nutrition.py` imports
# `_finding`, `Evidence` and `PackRefusal` from here, so naming it at module
# scope would be a cycle. Same shape and same reason as `Pack.recipes`, which
# is a callable for the neighboring reason: importing this module must not have
# side effects on what it imports.
#
# Thin wrappers rather than a lazy registry, because a wrapper is inspectable —
# `test_a_pack_names_what_it_will_look_for` runs `PACKS[key].detectors` against
# the pack's fixture and needs a callable, not a promise of one.


def _nutrition_atwater(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    from turbotab import nutrition
    return nutrition.atwater_finding(df)


def _nutrition_survey_weights(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    from turbotab import nutrition
    return nutrition.survey_weights_finding(df)


def _nutrition_partial_design(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    from turbotab import nutrition
    return nutrition.partial_design_finding(df)


def _nutrition_lonely_psu(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    from turbotab import nutrition
    return nutrition.lonely_psu_finding(df)


# ── clinical ─────────────────────────────────────────────────────────────────
#
# Same shape and the same reason as the four above: `turbotab/clinical.py`
# imports `_finding`, `Evidence` and `Claim` from here, so naming it at module
# scope would be a cycle, and a thin wrapper is inspectable where a lazy
# registry is not.
#
# **The pack held zero detectors until L41**, with a comment arguing the
# thinness was the point because physiologic bounds live in the core. That was
# true of §A1.2 and never true of §A1.3.


def _clinical_censored(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    from turbotab import clinical
    return clinical.censored_values_finding(df)


def _clinical_text_numeric(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    from turbotab import clinical
    return clinical.text_numeric_finding(df)


def _clinical_mixed_result(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    from turbotab import clinical
    return clinical.mixed_result_finding(df)


def _clinical_mixed_units(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    from turbotab import clinical
    return clinical.mixed_units_finding(df)


def _clinical_default_mass(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    from turbotab import clinical
    return clinical.default_value_mass_finding(df)


def _clinical_temporal(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    from turbotab import clinical
    return clinical.temporal_implausibility_finding(df)


def _clinical_number_format(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    from turbotab import clinical
    return clinical.number_format_finding(df)


def _clinical_impossible_vs_extreme(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    from turbotab import clinical
    return clinical.impossible_vs_extreme_finding(df)


# ── survey ───────────────────────────────────────────────────────────────────

# THE DETECTION, not the modeling treatment. §B4 — *"Ordinal vs interval — the
# long-running dispute"* — is DISPUTED and is carried by the `ordinal_encoding`
# prior, which is where a user meets that choice. What this finding asserts is
# narrower and is settled: a block of columns sharing one declared response
# scale is an instrument, and the order comes from the instrument.
ORDINAL_DECLARED_EVIDENCE = Evidence(
    status=SETTLED,
    source="research/CLINICAL_SURVEY_PACK.md#B1.1 Detecting Likert blocks")

# The response sets an instrument declares. A block of columns sharing exactly
# one of these is a scale; anything else is a set of numbers that happen to be
# small.
_LIKERT_SETS = ({1, 2, 3, 4, 5}, {1, 2, 3, 4}, {1, 2, 3, 4, 5, 6, 7},
                {0, 1, 2, 3}, {0, 1, 2, 3, 4})


# A response category holding more than this share of answers is a count
# distribution, not a response distribution. Measured, not guessed: the survey
# fixture's busiest category holds 29% and a low-expression gene's zero bin
# holds 78%.
_MAX_MODAL_SHARE = 0.60
# Required of most of the block, not all of it. A real instrument has floor
# effects — a screening item four in five people answer "never" is normal — and
# a rule that excluded those would reject the instruments it exists to find.
_MIN_SHARE_BALANCED = 0.7

# ── §B1.1's sentinel codes, and the extension they forced (`L41-C1`) ─────────
#
# **THE BLOCK DETECTOR USED TO MISS EXACTLY THE BLOCKS THE HIGHEST-YIELD CHECK
# IS ABOUT.** `values <= scale` is an exact-containment test, so a 1–5 item
# carrying a single `9` for *refused* failed it, the column dropped out of its
# own block, and a block where enough items carried one was not found at all.
# The check §B1.1 calls *"the highest-yield check in this pack"* was
# structurally unreachable from the detector that finds the blocks.
#
# So containment is now *scale plus a bounded tail of values that break the
# run*, and which values those were travels on the block. This is an extension
# of the one block detector rather than a second one beside it: `LOOP.md`'s
# `theory_anchors`/`theory_demos` lesson is that two registries describing one
# thing drift, and two block detectors would be that with the drift able to
# change what an instrument IS.
#
# §B1.1's own list. Corroborating evidence, never the rule — **the rule is
# that the value breaks the observed contiguous run**, because a codebook may
# use anything and the run is a property of the data.
KNOWN_SENTINELS = (7, 8, 9, 77, 88, 99, 98, 999, 9999, -1, -8, -9, -99)

# How much of a column may be out-of-run before it stops being an item with
# sentinels and becomes a different variable. Not the research's — §B1.1 states
# no share — so it is this module's own, chosen because a *don't know* rate
# above a quarter is a question about the question rather than a coding
# artifact.
_MAX_SENTINEL_SHARE = 0.25


def _breaks_the_run(value: int, support: set) -> bool:
    """Whether `value` sits outside the contiguous run the support forms.

    §B1.1's rule, and it is deliberately arithmetic rather than a lookup: a
    codebook may use any value, and `KNOWN_SENTINELS` is corroboration. A `6` in
    a 1–5 block breaks the run and is flagged even though no list names it; a
    `9` in a 0–9 block does not break it and is not, which is the *some
    legitimate scales do run 0–9* case the hard stop turns on.
    """
    return value < min(support) or value > max(support)


def likert_block(df: pd.DataFrame, minimum: int = 8) -> Optional[Dict[str, Any]]:
    """The largest set of columns sharing one declared response scale.

    Shared exactly, not approximately. Two columns on 1–5 and one on 1–7 are two
    instruments or one instrument and a stray, and averaging across them is the
    error the detector exists to avoid proposing.

    **Except for values that break the run**, which are candidate sentinel codes
    rather than responses and are carried on the block as `sentinels` instead of
    disqualifying the column. See `KNOWN_SENTINELS` above for why that had to
    change.

    **The support is the union across the block, never per item** — §B1.1's own
    instruction, and the reason is that a rarely-endorsed extreme category may
    be absent from a single item. Read per item, a 1–5 instrument where nobody
    picked 5 on `q14` has a 1–4 item in it, and a `5` elsewhere would then look
    like the sentinel.

    **And the block must look like RESPONSES, not like small counts.** This is
    the discriminator, and guard #2 found the need for it: 30 low-expression
    genes in `genomics_expression.csv` all take values in {0, 1, 2, 3}, share
    that scale exactly, and would otherwise have been read as a 30-item
    instrument — the survey pack firing on a count matrix, which is precisely
    the authoritative false assertion `DOMAIN_PACKS.md` §05 says would embarrass
    us.

    The evidence that separates them is the shape of the distribution, and it is
    stark: an instrument's categories are all used and roughly comparable — the
    survey fixture's busiest category holds 29% — while a low-expression gene
    decays, with 78% of samples at zero and one or two at the top. So a block
    needs every category of its scale used, and no category dominating, in most
    of its columns.
    """
    by_scale: Dict[Tuple[int, ...], List[str]] = {}
    outside: Dict[str, set] = {}
    for c in _numeric(df):
        s = df[c].dropna()
        if s.empty:
            continue
        try:
            values = set(int(v) for v in s.unique() if float(v).is_integer())
        except (TypeError, ValueError, OverflowError):
            continue
        if len(values) != s.nunique():
            continue                                    # non-integral values
        for scale in _LIKERT_SETS:
            inside = values & scale
            broken = {v for v in values - scale if _breaks_the_run(v, scale)}
            # Everything outside the scale must break the RUN. A `6` in a 1–5
            # block does not — it is inside 1..7 — and a column carrying one is
            # a different variable rather than an item with a sentinel, so it
            # drops out as it always did.
            if (values - scale) - broken:
                continue
            if not inside or len(inside) < len(scale) - 1:
                continue
            if broken:
                share = float(s.isin(list(broken)).mean())
                if share > _MAX_SENTINEL_SHARE:
                    continue
            by_scale.setdefault(tuple(sorted(scale)), []).append(str(c))
            outside[str(c)] = broken
            break
    if not by_scale:
        return None
    scale, columns = max(by_scale.items(), key=lambda kv: len(kv[1]))
    if len(columns) < minimum:
        return None

    balanced = 0
    for c in columns:
        s = df[c].dropna()
        # THE SENTINELS COME OUT BEFORE THE SHAPE IS JUDGED. A column that is
        # 20% `9` and otherwise flat would fail the modal-share test on the
        # sentinel rather than on its responses, which would reject the item
        # this whole extension exists to keep.
        s = s[~s.isin(list(outside.get(c) or ()))]
        if s.empty:
            continue
        used = set(int(v) for v in s.unique())
        if used != set(scale):
            continue                                    # a category never used
        if float(s.value_counts(normalize=True).max()) <= _MAX_MODAL_SHARE:
            balanced += 1
    if balanced / len(columns) < _MIN_SHARE_BALANCED:
        return None

    # THE SUPPORT, FROM THE UNION ACROSS THE BLOCK. §B1.1's instruction, and it
    # is computed here rather than taken from `scale` so that an instrument
    # where a category is unused anywhere is reported as what it is: the
    # declared scale is what the block matched, `observed_support` is what
    # anybody actually picked, and the two disagreeing is a floor or ceiling
    # effect rather than a detection failure.
    support: set = set()
    for c in columns:
        s = df[c].dropna()
        s = s[~s.isin(list(outside.get(c) or ()))]
        support |= {int(v) for v in s.unique()}
    sentinels = {c: sorted(v) for c, v in outside.items() if c in columns and v}
    return {"scale": list(scale), "columns": columns,
            "observed_support": sorted(support),
            "sentinels": sentinels}


def _ordinal_declared(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    block = likert_block(df)
    if block is None:
        return None
    scale, columns = block["scale"], block["columns"]
    return _finding(
        "pack::survey::ordinal_declared", "info",
        f"{len(columns):,} columns share one {len(scale)}-point response scale",
        (f"Every value in them is one of {scale}. The block runs "
         f"`{columns[0]}` … `{columns[-1]}`."),
        ("The order comes from the instrument, not from the data — which makes "
         "the encoding row-local: the number for a row depends on that row's "
         "own answer and on nothing else, so it is applied now rather than "
         "fitted inside the training folds. An encoding derived from the "
         "observed frequencies would be a different object, would have to be "
         "deferred, and would silently change meaning between cohorts."),
        confidence="high", pack=SURVEY, marker="derived",
        evidence=ORDINAL_DECLARED_EVIDENCE,
        columns=columns[:10],
        params={"scale": scale, "columns": columns, "encoding": "declared"})


def _survey_sentinel_codes(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """§B1.1's highest-yield check, deferred-imported for the usual cycle.

    `turbotab/survey.py` imports `likert_block`, `KNOWN_SENTINELS` and
    `_finding` from here, so naming it at module scope would be a cycle — the
    same shape as the nutrition and clinical wrappers above.
    """
    from turbotab import survey
    return survey.sentinel_codes_finding(df)


# ── genomics ─────────────────────────────────────────────────────────────────

# ONE BADGE OVER A FINDING THAT MAKES TWO CLAIMS, and the join is worth naming.
# The model-ranking half is SETTLED and is what this cites. The normalization
# half is the pack DECLINING to assert a default, and the field's disagreement
# there is DISPUTED and is carried by the `normalization` prior at
# `GENOMICS_PACK.md#04`. A finding carries one badge, so the two statuses cannot
# both travel on it — filed as `GUIDED-064` rather than resolved by inventing a
# per-sentence badge on one example.
COUNTS_EVIDENCE = Evidence(
    status=SETTLED,
    source="research/GENOMICS_PACK.md#08 · Modeling at p >> n")

# `GUIDED-064` RESOLVED HERE, on the instance that filed it. The finding says
# two things and the field holds them differently, so the badge says two things
# too — and `may_preselect` is recomputed over both, which is what stops a
# machine pre-selecting across a disagreement it could not see.
COUNTS_CLAIMS = (
    Claim("model_ranking",
          "At p much greater than n an unregularized fit is degenerate, so "
          "regularized models rank first and distance-based ones last.",
          COUNTS_EVIDENCE),
    Claim("normalization",
          "CPM, TPM and VST answer different questions and are not "
          "interchangeable, so no normalization default is asserted.",
          Evidence(
              status=DISPUTED,
              source=("research/GENOMICS_PACK.md#04 · Normalization — no "
                      "default asserted"),
              both_sides=(
                  "CPM, TPM and VST are not interchangeable and the choice "
                  "depends on the assay and the question. The research asserts "
                  "no default and neither does this pack; the disagreement is "
                  "the finding, and declining is recorded rather than absent."
              ))),
)


def count_matrix(df: pd.DataFrame, minimum: int = 100) -> Optional[Dict[str, Any]]:
    """A block of non-negative integer columns wide enough to be an assay.

    Integrality is the whole reading. Counts and concentrations are different
    objects and the difference decides whether a log transform is derived — it
    is, for concentrations, because they combine multiplicatively — or merely
    one option among several, which is what it is for counts.
    """
    cols = []
    for c in _numeric(df):
        s = df[c].dropna()
        if s.empty or float(s.min()) < 0:
            continue
        try:
            if not np.all(np.equal(np.mod(s.to_numpy(dtype=float), 1), 0)):
                continue
        except (TypeError, ValueError):
            continue
        cols.append(str(c))
    if len(cols) < minimum:
        return None
    return {"columns": cols, "p_over_n": len(cols) / max(len(df), 1)}


def _counts_at_p_over_n(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    block = count_matrix(df)
    if block is None or block["p_over_n"] < 2.0:
        return None
    cols = block["columns"]
    depth = df[cols].sum(axis=1)
    spread = float(depth.max() / max(depth.min(), 1))
    return _finding(
        "pack::genomics::counts_p_over_n", "warning",
        f"{len(cols):,} count columns against {len(df):,} samples",
        (f"Every value in them is a non-negative integer, p/n is "
         f"{block['p_over_n']:.1f}, and total depth per sample varies "
         f"{spread:.1f}-fold."),
        ("Two consequences, and only one of them is ours to settle. Model "
         "ranking is: at this p/n an unregularized fit is not merely optimistic "
         "but degenerate, so regularized models rank first and distance-based "
         "ones last — ordered, never filtered. Normalization is NOT: CPM, TPM "
         "and VST answer different questions and are not interchangeable, and "
         "the right one depends on the assay and on what you are asking. "
         "**No normalization default is asserted here**, and that is the "
         "considered position rather than an omission."),
        confidence="high", pack=GENOMICS, marker="derived",
        evidence=COUNTS_EVIDENCE, claims=COUNTS_CLAIMS,
        columns=cols[:8],
        params={"n_features": len(cols), "p_over_n": round(block["p_over_n"], 2),
                "depth_spread": round(spread, 2),
                "normalization_default": None,
                "model_prior": "regularized_first"})


# ── genomics · what your numbers are (§02) ──────────────────────────────────
#
# §02 is titled *"the highest-leverage diagnostic in the pack"* and says why:
# **it determines what is legal downstream, and getting it wrong is the
# commonest real failure.** Nine signatures, one hard rule, and five branches of
# coaching that each close or open a route.
#
# ## THE ORIENTATION THAT DECIDES EVERY STATISTIC
#
# §01 is explicit that the field convention is **genes in rows, samples in
# columns** — the transpose of what the rest of this app assumes. This app's
# tables are **samples in rows** (`turbotab/orientation.py` is the question that
# establishes it), so §02's *"per column"* is **per sample** here and a row sum
# is a **library size**. Get that backwards and every statistic below is
# transposed: the CPM test would ask whether every gene sums to 1e6, which is
# true of nothing.
#
# ## WHICH COLUMNS ARE THE MATRIX
#
# An expression export arrives with covariates beside it — `age` sits in every
# genomics fixture in this tree — and a covariate inside the block moves exactly
# one family of statistics: the **extremes**. A row sum barely notices one column
# in 496 and a global maximum is decided by it, which is the difference between
# reading `genomics_vst.csv` as topping out at 14.2 (VST) and at 79 (nothing in
# the table). So `expression_block` drops a column that is BOTH all-integer AND
# wholly outside the range the continuous majority occupies — measured, this
# drops `age` from the VST and microarray matrices and **nothing at all** from
# the other six, because on those `age` sits inside the range and changes no
# reading. On an all-integer matrix there is no continuous majority to be outside
# of, nothing is dropped, and that is correct: raw counts are integers too.
#
# It is a shape rule and not a name rule (guard #2), and the columns it dropped
# are named on the card so a reader can disagree with it.

DATA_TYPE_EVIDENCE = Evidence(
    # CONVENTION rather than SETTLED, and the distinction is the badge doing its
    # job. §02's table is a **reading convention** — the bands, the "≫1e4", the
    # "roughly but not exactly equal" — and the thresholds under it are this
    # module's, measured on the fixtures. What IS settled is the hard rule and
    # each branch of the coaching, and those travel as their own claims below.
    status=CONVENTION_STATUS,
    source=("research/GENOMICS_PACK.md#02 · Data-type detection — the "
            "highest-leverage diagnostic in the pack"))

# THE ONE LINE §02 STATES AS A RULE. Quoted rather than paraphrased: *"any
# negative value rules out raw counts, CPM, TPM and FPKM."* It is asserted as a
# rule and not as a heuristic, which means it runs FIRST and no later branch may
# reach past it.
NEGATIVES_RULE_EVIDENCE = Evidence(
    status=SETTLED,
    source=("research/GENOMICS_PACK.md#02 · Data-type detection — the "
            "highest-leverage diagnostic in the pack"))

# §00: *"Formally undecidable from the matrix alone. Both rescale each sample to
# exactly 1e6; TPM divides by effective length BEFORE the rescale, which erases
# the trace. No column-sum, distributional or skew test separates them."*
UNDECIDABLE_EVIDENCE = Evidence(
    status=SETTLED,
    source="research/GENOMICS_PACK.md#00 · The non-defaultable set")

NORMALIZATION_EVIDENCE = Evidence(
    status=SETTLED,
    source="research/GENOMICS_PACK.md#04 · Normalization — no default asserted")

FIGURE_INPUT_EVIDENCE = Evidence(
    status=SETTLED,
    source="research/GENOMICS_PACK.md#07 · EDA and presentation — the priority")

ID_VOCABULARY_EVIDENCE = Evidence(
    status=CONVENTION_STATUS,
    source="research/GENOMICS_PACK.md#01 · Import and structure")

# ── the nine signatures §02's table names ────────────────────────────────────

RAW_COUNTS = "raw_counts"
ESTIMATED_COUNTS = "estimated_counts"
CPM_OR_TPM = "cpm_or_tpm"
TMM_SCALED_CPM = "tmm_scaled_cpm"
FPKM = "fpkm"
VST = "vst"
RLOG = "rlog"
MICROARRAY = "microarray"
LOG_RATIO = "log_ratio"

#: In §02's own order, so a reader can put the table beside this list.
SIGNATURES: Tuple[str, ...] = (RAW_COUNTS, ESTIMATED_COUNTS, CPM_OR_TPM,
                               TMM_SCALED_CPM, FPKM, VST, RLOG, MICROARRAY,
                               LOG_RATIO)

SIGNATURE_NAMES: Dict[str, str] = {
    RAW_COUNTS: "raw counts",
    ESTIMATED_COUNTS: "estimated counts",
    CPM_OR_TPM: "CPM or TPM",
    TMM_SCALED_CPM: "TMM- or median-of-ratios-scaled CPM",
    FPKM: "FPKM/RPKM",
    VST: "variance-stabilized values (VST)",
    RLOG: "regularized-log values (rlog)",
    MICROARRAY: "microarray log2 intensity (RMA)",
    LOG_RATIO: "already log-ratio, z-scored or batch-corrected",
}

# ── the thresholds, each measured on the shipped fixtures ───────────────────
#
# Every one of these is written down with what it was measured against, for the
# reason `orientation.py` records: a threshold chosen against a constructed
# signal is a threshold nobody has seen fail.

#: The library-size total CPM and TPM rescale to. §02 row 3.
_LIBRARY_TOTAL = 1e6

#: §02 row 3's *"±1e-3"*, read as RELATIVE. `genomics_cpm.csv`'s widest row-sum
#: deviation is 79 in 1e6 — 7.9e-5, which is the `age` column and nothing else —
#: so the fixture clears this by more than an order of magnitude.
_LIBRARY_EXACT = 1e-3

#: §02 row 4's *"roughly but not exactly equal near 1e6"*. Measured:
#: `genomics_tmm_cpm.csv`'s loudest row is 1.140 of the total and its quietest
#: 0.854, so the widest TMM deviation is 0.140; `genomics_fpkm.csv`'s QUIETEST
#: deviation is 0.339 (its row sums run 0.355–0.661 of 1e6). A quarter sits
#: 1.8× above the loudest thing that must pass and 1.4× below the quietest thing
#: that must not.
_LIBRARY_NEAR = 0.25

#: §02 rows 1 and 2, *"max ≫1e4"*. `genomics_expression.csv` maxes at 26,471.
_COUNT_SCALE = 1e4

#: §02 rows 6 and 7, *"max ~15–25"*. `genomics_vst.csv` maxes at 14.2 and
#: `genomics_microarray.csv` at 15.8, so the ceiling is above both by design —
#: it is the top of the band §02 names, not a fitted bound.
_STABILIZED_CEILING = 25.0

#: §02 row 6's *"repeated floor"*, as a share of cells sitting on the minimum.
#: Measured: `genomics_vst.csv` puts 15.3% of its cells on 2.0 (every zero
#: count), `genomics_microarray.csv` 0.017% (five cells that hit the clip). Two
#: percent is 7.7× below the first and 118× above the second — and the two ARE
#: the pair this separates, because both are continuous, both top out near 15,
#: and both have a minimum of exactly 2.0.
_FLOOR_SHARE = 0.02

#: §02 row 7, rlog's *"small negatives permitted"*, read literally: how far
#: below zero the floor may sit as a fraction of the ceiling. rlog is a
#: log-expression scale whose zeros are shrunk to a small negative floor, so its
#: negatives are a rounding error against its range; row 9's matrix is centred
#: on zero, so its negatives are half of it.
#:
#: **NO SHIPPED FIXTURE IS RLOG AND THIS SAYS SO.** The separator is checked
#: against a constructed frame in
#: `test_the_genomics_data_type_card_reaches_a_person.py` and against
#: `wide_assay.csv` on the other side, where the ratio is 0.98 — two orders of
#: magnitude clear of this — so the value is a statement of the shape rather
#: than a boundary fitted between two examples.
_RLOG_NEGATIVE_SHARE = 0.1

#: Row 9's *"symmetric around 0"*, as the distance the median may sit from it.
_CENTRED_MEDIAN = 0.5

#: The two library-size spreads the estimated-counts/FPKM split is measured
#: against, named rather than folded into one midpoint so a reader can see how
#: far apart they actually are. §02 separates those two rows by max and skew and
#: those overlap; this is what does not. Measured over the block this module
#: reads, on `genomics_estimated_counts.csv` and `genomics_fpkm.csv`.
_ESTIMATED_COUNTS_CV = 0.271
_FPKM_CV = 0.114

#: How much of the block must carry a probe-style ID before §02 row 8's second
#: half is satisfied. `genomics_microarray.csv` renames every column, so this is
#: a floor and not a tuned value.
_PROBE_SHARE = 0.5

#: The ID vocabularies §01 lists, in its own order. Read for ONE purpose here —
#: telling an array probe from a gene — and deliberately not used to guess
#: orientation, which is `orientation.py`'s question and is asked, never
#: inferred.
_ID_PATTERNS: Tuple[Tuple[str, str], ...] = (
    ("ensembl_gene", r"^ENS[A-Z]{0,4}G\d{11}(\.\d+)?$"),
    ("ensembl_transcript", r"^ENS[A-Z]{0,4}T\d{11}(\.\d+)?$"),
    ("refseq", r"^[NX][MR]_\d+"),
    ("affymetrix", r"^\d+(_[a-z]+)?_at$"),
    ("illumina", r"^ILMN_\d+$"),
    ("agilent", r"^A_\d+_P\d+$"),
)

#: Which of those are array PROBES rather than gene or transcript identifiers.
#: §02 row 8's *"probe-style IDs"* is the second half of the microarray
#: signature, and an Ensembl id would satisfy the first half of it wrongly.
_PROBE_VOCABULARIES = frozenset({"affymetrix", "illumina", "agilent"})


def id_vocabulary(columns: Sequence[str]) -> Dict[str, Any]:
    """Which identifier vocabulary these column labels are drawn from.

    §01's detection cascade, step 1, used for the one question §02 asks of it.
    Returns the share of labels matching each vocabulary rather than a verdict,
    because a mixed vocabulary is itself a §01 finding and a function that
    returned one name could not say so.
    """
    labels = [str(c) for c in columns]
    hits: Dict[str, int] = {}
    for name, pattern in _ID_PATTERNS:
        n = sum(1 for label in labels if re.match(pattern, label))
        if n:
            hits[name] = n
    total = max(len(labels), 1)
    probe = sum(n for name, n in hits.items() if name in _PROBE_VOCABULARIES)
    return {"matched": {k: round(v / total, 4) for k, v in hits.items()},
            "probe_share": round(probe / total, 4),
            "n_labels": len(labels)}


def _integral_column(series: pd.Series) -> bool:
    values = series.dropna().to_numpy(dtype=float, na_value=np.nan)
    if values.size == 0:
        return False
    try:
        return bool(np.all(np.equal(np.mod(values, 1), 0)))
    except (TypeError, ValueError):                            # pragma: no cover
        return False


def expression_block(df: pd.DataFrame,
                     minimum: int = 30) -> Optional[Dict[str, Any]]:
    """The columns that are the expression matrix, and the ones that are not.

    See the section comment above for why this exists and what it may drop. Two
    properties worth stating rather than leaving to be discovered:

    * **It drops nothing from an all-integer table.** Raw counts are integers,
      so there is no continuous majority for a covariate to sit outside of, and
      the honest answer is to keep everything and say so.
    * **It never drops more than it can name.** `excluded` is the whole list,
      uncut (`GUIDED-209`), because the card shows it and a reader who thinks
      the app dropped a gene needs to be able to check.
    """
    cols = _numeric(df)
    if len(cols) < minimum or df.empty:
        return None
    integral = [c for c in cols if _integral_column(df[c])]
    continuous = [c for c in cols if c not in integral]
    excluded: List[str] = []
    if continuous:
        rest = df[continuous].to_numpy(dtype=float)
        lo, hi = float(np.nanmin(rest)), float(np.nanmax(rest))
        for c in integral:
            values = df[c].dropna().to_numpy(dtype=float)
            if values.size and (float(values.min()) > hi
                                or float(values.max()) < lo):
                excluded.append(str(c))
    kept = [str(c) for c in cols if str(c) not in set(excluded)]
    if len(kept) < minimum:
        return None
    return {"columns": kept, "excluded": excluded,
            "n_columns": len(kept), "n_excluded": len(excluded),
            "n_samples": int(len(df))}


def read_matrix(df: pd.DataFrame,
                minimum: int = 30) -> Optional[Dict[str, Any]]:
    """§02's statistics, on this app's orientation.

    *"Per column: sum, min, max, median, % zeros, % integer. Per matrix: global
    max, negatives present."* — read **per sample**, because this app's rows are
    samples and §02's are genes. A row sum here is a library size.

    ## THE % INTEGER TRAP, DECIDED RATHER THAN LEFT AMBIGUOUS

    Computed over ALL cells, *% integer* reads about 15% on the CPM, FPKM and
    VST matrices — **because a zero is an integer**, and those matrices are 15%
    zeros. That number describes the zero fraction and nothing else. So two are
    reported and they answer different questions:

    * `pct_integer` — over the **non-zero** cells, which is what *"are these
      integers"* means once the zeros are set aside, and what the card shows;
    * `all_integral` — whether **every** cell is a whole number, which is what
      §02 row 1 actually requires and what the classifier gates on.

    `pct_integer_all` is carried too, and only so the card can show why the
    naive reading is misleading.
    """
    block = expression_block(df, minimum=minimum)
    if block is None:
        return None
    cols = block["columns"]
    values = df[cols].to_numpy(dtype=float)
    if values.size == 0:                                       # pragma: no cover
        return None
    finite = np.isfinite(values)
    if not finite.any():                                       # pragma: no cover
        return None
    per_sample_sum = np.nansum(values, axis=1)
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    nonzero = values[finite & (values != 0)]
    whole = np.equal(np.mod(values[finite], 1), 0)
    floor_cells = int(np.sum(values[finite] == lo))
    with np.errstate(all="ignore"):
        deviation = np.abs(per_sample_sum - _LIBRARY_TOTAL) / _LIBRARY_TOTAL
    return {
        **block,
        # PER SAMPLE — §02's "per column", transposed to this app's rows.
        "library_size": {
            "min": float(per_sample_sum.min()),
            "max": float(per_sample_sum.max()),
            "median": float(np.median(per_sample_sum)),
            "cv": float(per_sample_sum.std(ddof=1) / per_sample_sum.mean())
            if len(per_sample_sum) > 1 and per_sample_sum.mean() else 0.0,
        },
        "sample_median": {"min": float(np.nanmedian(values, axis=1).min()),
                          "max": float(np.nanmedian(values, axis=1).max())},
        # PER MATRIX.
        "min": lo,
        "max": hi,
        "median": float(np.nanmedian(values)),
        "negatives": bool(lo < 0),
        "pct_zeros": float(np.mean(values[finite] == 0)),
        "pct_integer": float(np.mean(np.equal(np.mod(nonzero, 1), 0)))
        if nonzero.size else 0.0,
        "pct_integer_all": float(np.mean(whole)),
        "all_integral": bool(np.all(whole)),
        "floor_share": float(floor_cells / max(int(finite.sum()), 1)),
        "library_deviation": {"min": float(deviation.min()),
                              "max": float(deviation.max())},
        "ids": id_vocabulary(cols),
    }


def _classify(m: Dict[str, Any]) -> Dict[str, Any]:
    """§02's table, in §02's order, with the hard rule first.

    Returns the signature keys the reading supports — **more than one where the
    matrix genuinely cannot separate them**, which is two of the nine rows and
    is the whole reason this returns a list. Never an empty list and never a
    guess: an unreadable matrix comes back as `keys: []` with `read: False` one
    level up, because *"the app may be silent"* is an available answer and
    *"probably CPM"* is not.
    """
    keys: List[str] = []
    # ── THE HARD RULE, FIRST AND UNCONDITIONALLY (SETTLED) ──────────────────
    # *"Any negative value rules out raw counts, CPM, TPM and FPKM."* Written as
    # the outermost branch rather than as a filter applied afterwards, so no
    # later test can reach past it — a `sums to 1e6` test on a mean-centred
    # matrix would otherwise be free to fire.
    if m["negatives"]:
        if (m["max"] <= _STABILIZED_CEILING
                and m["floor_share"] >= _FLOOR_SHARE
                and m["max"] > 0
                and abs(m["min"]) <= _RLOG_NEGATIVE_SHARE * m["max"]):
            keys = [RLOG]
        elif abs(m["median"]) <= _CENTRED_MEDIAN:
            keys = [LOG_RATIO]
        return {"keys": keys, "ruled_out": [RAW_COUNTS, CPM_OR_TPM, FPKM]}

    within = m["library_deviation"]["max"]
    stabilized = (m["max"] <= _STABILIZED_CEILING and m["pct_zeros"] == 0.0)
    if m["all_integral"] and m["max"] > _COUNT_SCALE:
        keys = [RAW_COUNTS]
    elif within <= _LIBRARY_EXACT:
        # §00: formally undecidable. TWO NAMES, and the app does not choose.
        keys = [CPM_OR_TPM]
    elif within <= _LIBRARY_NEAR:
        keys = [TMM_SCALED_CPM]
    elif stabilized and m["floor_share"] >= _FLOOR_SHARE:
        keys = [VST]
    elif stabilized and m["ids"]["probe_share"] >= _PROBE_SHARE:
        keys = [MICROARRAY]
    elif (not m["all_integral"] and m["max"] > _COUNT_SCALE
          and m["pct_zeros"] > 0):
        # WHY THE ZERO CLAUSE, WHICH §02 DOES NOT ASK FOR. Without it this
        # branch fires on `metabolomics_untargeted.csv` under a genomics lens
        # and calls 395 LC-MS peak areas *"estimated counts or FPKM"*. Both of
        # those rows are derived FROM counts — a zero count is a zero TPM and a
        # zero FPKM — and §02 row 1 puts 20–60% zeros on the matrix they come
        # from, so a transcript quantification with no zero anywhere in it is
        # not one. Measured: all five count-family fixtures here are 15.3%
        # zeros and the metabolomics panel is 0.0%.
        #
        # The weakest possible form — ANY zero, not a band — because the band
        # would be a threshold this pack invented, and the direction to be wrong
        # in is silence on a real matrix rather than a confident sentence about
        # somebody else's assay.
        #
        # §02 SEPARATES THESE TWO ROWS BY MAX AND SKEW, AND THEY OVERLAP.
        # Measured on the two fixtures that carry them, the separator that does
        # work is the library-size spread: `genomics_estimated_counts.csv` keeps
        # the raw matrix's coefficient of variation because nothing normalized
        # it, and `genomics_fpkm.csv` sits at less than half of that because
        # FPKM divided the library size out. §02 says **ask** for this row, so
        # both names come back, ORDERED by that spread, and the card asks.
        estimated_first = m["library_size"]["cv"] >= (
            _ESTIMATED_COUNTS_CV + _FPKM_CV) / 2
        keys = ([ESTIMATED_COUNTS, FPKM] if estimated_first
                else [FPKM, ESTIMATED_COUNTS])
    return {"keys": keys, "ruled_out": []}


# ── the branched coaching (§02), quoted where §02 quotes ────────────────────

COACHING_EVIDENCE = Evidence(
    status=SETTLED,
    source=("research/GENOMICS_PACK.md#02 · Data-type detection — the "
            "highest-leverage diagnostic in the pack"))

# The normalization CHOICE, as against the fact that one is needed. Carried at
# DISPUTED with both positions, and it is the same disagreement the pack's
# `normalization` prior already declines to resolve — one text, so the card and
# the prior cannot drift.
NORMALIZATION_CHOICE_EVIDENCE = Evidence(
    status=DISPUTED,
    source="research/GENOMICS_PACK.md#04 · Normalization — no default asserted",
    both_sides=(
        "CPM, TPM and VST are not interchangeable and the choice depends on "
        "the assay and the question. The research asserts no default and "
        "neither does this pack; the disagreement is the finding, and "
        "declining is recorded rather than absent."))

COACHING: Dict[str, str] = {
    RAW_COUNTS: (
        "Raw counts are the one input that lets a count model estimate "
        "measurement precision. DESeq2: \"only the count values allow "
        "assessing the measurement precision correctly\", and it \"internally "
        "corrects for library size, so transformed or normalized values should "
        "not be used as input\". Do not pre-normalize these."),
    ESTIMATED_COUNTS: (
        "Estimated counts come out of salmon, kallisto or RSEM, and the "
        "fractions are real: a read that maps to several transcripts is split "
        "between them. They are not raw counts and they are not normalized "
        "values, and those two have opposite downstream treatment — which is "
        "why the research says to ask rather than to infer."),
    CPM_OR_TPM: (
        "These are already per-sample-normalized, which closes off the "
        "negative-binomial route because count-level variance has been "
        "destroyed. Either recover the raw counts, which is strongly "
        "preferred, or use a limma-style Gaussian workflow on log2(x + "
        "offset). Feeding these to a count model runs silently and its "
        "p-values are wrong."),
    TMM_SCALED_CPM: (
        "The scaling here corrects library composition and not only depth, "
        "which is what CPM, TPM and FPKM do not do: if a few genes take most "
        "of the reads in one sample, every other gene's value in that sample "
        "is deflated. In the comparison the research cites, TMM and "
        "median-of-ratios controlled the false-positive rate where total-count "
        "and RPKM normalization did not. They are still normalized values, so "
        "the count-model route is closed."),
    FPKM: (
        "FPKM and RPKM are not comparable across samples even in principle. "
        "Wagner, Kim and Lynch (Theory Biosci 131:281, 2012) showed RPKM/FPKM "
        "violates the invariance a relative-molar-concentration measure has to "
        "satisfy, and Dillies and colleagues (Brief Bioinform 14:671, 2013) "
        "found total-count and RPKM normalization did not control the "
        "false-positive rate where TMM and median-of-ratios did. They are also "
        "already per-sample-normalized, so the count-model route is closed."),
    VST: (
        "Variance-stabilized values are for visualization, clustering and PCA. "
        "They are never the input to a differential-expression test — a test "
        "on them is typically anticonservative, which means it reports more "
        "than it has found."),
    RLOG: (
        "Regularized-log values are for visualization, clustering and PCA. "
        "They are never the input to a differential-expression test — a test "
        "on them is typically anticonservative, which means it reports more "
        "than it has found."),
    MICROARRAY: (
        "This is a hybridization intensity rather than a count, and the whole "
        "count toolchain does not apply to it — no library size, no dispersion "
        "estimate, no negative binomial. limma is the tool built for this "
        "assay."),
    LOG_RATIO: (
        "There are negative values in this matrix, which rules out raw counts, "
        "CPM, TPM and FPKM — that is a rule rather than a reading. What is "
        "left is a matrix that has already been put on a relative scale: a "
        "log-ratio against a reference, a z-score within each gene, or a "
        "batch-corrected residual. Which of those it is decides what a "
        "comparison across samples means, and the numbers do not say."),
}

#: How firmly the reading holds, per signature, and why — composed here rather
#: than on the page, because a confidence a reader cannot interrogate is a
#: number with no argument behind it.
CONFIDENCE: Dict[str, Tuple[str, str]] = {
    RAW_COUNTS: ("high", "Every value is a whole number and the matrix reaches "
                         "well past ten thousand. Nothing else in the table "
                         "reads that way."),
    ESTIMATED_COUNTS: ("medium", "Two signatures fit these numbers and the "
                                 "research says to ask rather than to choose "
                                 "between them."),
    CPM_OR_TPM: ("high", "Every sample sums to one million, which is a "
                         "construction rather than a coincidence. Which of the "
                         "two it is cannot be recovered from the matrix at "
                         "all."),
    TMM_SCALED_CPM: ("high", "Every sample sums to near one million and none "
                             "of them sums to exactly one million, which is "
                             "what a composition-aware rescaling leaves "
                             "behind."),
    FPKM: ("medium", "Two signatures fit these numbers and the research says "
                     "to ask rather than to choose between them."),
    VST: ("high", "Continuous values with a ceiling in the teens and a floor "
                  "that a large share of the matrix sits exactly on — the "
                  "floor is every zero count, mapped to one value."),
    RLOG: ("medium", "The shape fits, and no shipped example of this "
                     "transform exists to check the reading against — so it "
                     "is offered as the likeliest rather than asserted."),
    MICROARRAY: ("high", "Continuous values inside the intensity band, no "
                         "zeros anywhere, and the column labels are array "
                         "probe identifiers rather than gene identifiers."),
    LOG_RATIO: ("high", "There are negative values, which is a rule and not a "
                        "reading, and the matrix is centred on zero."),
}

# ── the capability matrix — §02's *"single most valuable artifact"* ──────────
#
# *"A capability matrix showing which downstream steps are now enabled,
# disabled, or require input."*
#
# ## `GUIDED-207` IS THE DESIGN CONSTRAINT AND IT IS WORTH SAYING WHY
#
# The natural shape for this is `{"count_model": "disabled", "because":
# "normalized"}` — a KEY standing for a sentence somebody else has to write.
# That is `GUIDED-207` exactly: a field that NAMES what an interface must
# construct rather than DESCRIBING it, filed as trap #1 at field granularity.
# The interface would then hold a copy of the rule, the two would drift, and the
# one the user reads is the one nothing tests.
#
# **So the server composes the sentence and the page renders it.** Every row
# carries `because` in full, its own badge, and `state_label` — so even the
# three-way state does not have to be translated by a reader. `state` is left in
# the payload for styling and for the record, and nothing needs it to build the
# row.
#
# ## AND WHAT THESE ROWS ARE ABOUT
#
# TurboTab fits no count model and runs no differential-expression test. These
# rows are about **the data** and hold wherever it is taken next; `scope_note`
# says so on the card, because a matrix of capabilities beside an app that has
# none of them would read as a list of buttons.

ENABLED = "enabled"
DISABLED = "disabled"
REQUIRES_INPUT = "requires_input"

STATE_LABELS: Dict[str, str] = {
    ENABLED: "Open to you",
    DISABLED: "Ruled out",
    REQUIRES_INPUT: "Needs an answer first",
}

CAPABILITY_LABELS: Dict[str, str] = {
    "count_model": "A count model of differential expression (DESeq2, edgeR)",
    "gaussian_workflow": "A Gaussian workflow on log values (limma)",
    "per_sample_normalization": "Scaling each sample to a common library size",
    "log_transform": "A log transform before modeling",
    "pca_and_clustering":
        "PCA, clustering and the sample-to-sample correlation heatmap",
    "cross_sample_comparison": "Comparing one gene's values across samples",
}

#: The order the rows are served in, hardest-first: what the classification
#: CLOSES comes before what it leaves open.
CAPABILITIES: Tuple[str, ...] = (
    "count_model", "gaussian_workflow", "per_sample_normalization",
    "log_transform", "pca_and_clustering", "cross_sample_comparison")

_SCOPE_NOTE = (
    "These rows are about your data, not about this app: TurboTab fits no "
    "count model and runs no differential-expression test. They say what this "
    "matrix will and will not support, wherever you take it next.")


def _count_family(estimated: bool) -> Dict[str, Tuple[str, str, Evidence]]:
    """Raw and estimated counts. One function because they differ in one row."""
    return {
        "count_model": (
            (REQUIRES_INPUT,
             "These are estimated counts, and whether a count model may take "
             "them depends on the tool that produced them and on how they were "
             "imported — rounding them and whether an offset for effective "
             "length is carried across are both decisions, and the matrix "
             "records neither. Estimated counts and normalized values are both "
             "non-integer and their downstream treatment is opposite, which is "
             "why this is asked.",
             COACHING_EVIDENCE)
            if estimated else
            (ENABLED,
             "This is the one input that lets a count model estimate "
             "measurement precision — DESeq2: \"only the count values allow "
             "assessing the measurement precision correctly\". It corrects for "
             "library size internally, so do not hand it normalized values.",
             COACHING_EVIDENCE)),
        "gaussian_workflow": (
            ENABLED,
            "Counts can enter a Gaussian workflow through a mean-variance "
            "weighting, which is the standard route where a count model is not "
            "wanted. It is a second route rather than a better one.",
            COACHING_EVIDENCE),
        "per_sample_normalization": (
            REQUIRES_INPUT,
            "Total depth varies across the samples in this table, so a "
            "correction is needed — but no default is asserted for which. TMM "
            "and median-of-ratios correct library composition as well as depth "
            "and are near-interchangeable; CPM, TPM and FPKM correct depth "
            "only and are not substitutes for them. Which is right depends on "
            "the assay and on what you are asking.",
            NORMALIZATION_CHOICE_EVIDENCE),
        "log_transform": (
            REQUIRES_INPUT,
            "A log here is one option among several rather than something the "
            "app can derive. Concentrations are log-normal by construction, "
            "which is why a log is automatic elsewhere in this app; counts are "
            "not concentrations and that argument does not transfer.",
            NORMALIZATION_EVIDENCE),
        "pca_and_clustering": (
            REQUIRES_INPUT,
            "A sample PCA is computed on variance-stabilized values and never "
            "on raw counts — on counts the first component becomes a "
            "library-size artifact. Which stabilizing transform to use is the "
            "open question here, not whether to use one.",
            FIGURE_INPUT_EVIDENCE),
        "cross_sample_comparison": (
            REQUIRES_INPUT,
            "Not until the library sizes are corrected. Total depth varies "
            "across these samples, so an uncorrected comparison reads depth as "
            "expression.",
            NORMALIZATION_EVIDENCE),
    }


def _depth_normalized(cross: Tuple[str, str, Evidence]
                      ) -> Dict[str, Tuple[str, str, Evidence]]:
    """CPM/TPM, TMM-scaled CPM and FPKM. They differ only on the last row, and
    that row is where the research is sharpest."""
    return {
        "count_model": (
            DISABLED,
            "These values are already per-sample-normalized, which destroys "
            "count-level variance and closes off the negative-binomial route. "
            "Feeding them to a count model runs silently and its p-values are "
            "wrong — nothing errors, and the numbers that come out look "
            "ordinary.",
            COACHING_EVIDENCE),
        "gaussian_workflow": (
            ENABLED,
            "This is the route to use when the raw counts cannot be recovered: "
            "a limma-style Gaussian workflow on log2 of the value plus an "
            "offset. Recovering the raw counts is still strongly preferred.",
            COACHING_EVIDENCE),
        "per_sample_normalization": (
            DISABLED,
            "It has already been done — that is what these values are. "
            "Normalizing a second time leaves numbers nobody can interpret.",
            NORMALIZATION_EVIDENCE),
        "log_transform": (
            REQUIRES_INPUT,
            "log2 of the value plus an offset is the usual next step, and the "
            "offset is a real choice rather than a formality: it decides what "
            "happens to the zeros, and this matrix has them.",
            NORMALIZATION_EVIDENCE),
        "pca_and_clustering": (
            ENABLED,
            "These support a sample PCA once they are on a log scale. The "
            "thing to watch for is a first component that tracks library size "
            "or the zero fraction rather than the condition — that is a "
            "normalization problem, not biology.",
            FIGURE_INPUT_EVIDENCE),
        "cross_sample_comparison": cross,
    }


def _stabilized(name: str) -> Dict[str, Tuple[str, str, Evidence]]:
    return {
        "count_model": (
            DISABLED,
            f"{name} values are for visualization, clustering and PCA, and are "
            f"never the input to a differential-expression test. A test on "
            f"them is typically anticonservative — it reports more than it has "
            f"found.",
            COACHING_EVIDENCE),
        "gaussian_workflow": (
            DISABLED,
            "The same rule covers this one: a test on stabilized values is a "
            "test on the output of a transform built for looking at data, and "
            "it is typically anticonservative. Go back to the counts.",
            COACHING_EVIDENCE),
        "per_sample_normalization": (
            DISABLED,
            "Size-factor normalization is inside the transform that produced "
            "these values. Scaling them again would be normalizing twice.",
            NORMALIZATION_EVIDENCE),
        "log_transform": (
            DISABLED,
            "These are already on a log scale. Taking a log of them would be a "
            "log of a log, and the result is not interpretable in any unit.",
            NORMALIZATION_EVIDENCE),
        "pca_and_clustering": (
            ENABLED,
            "This is exactly what these values are for. It is the single most "
            "expected figure in the field and this is its intended input.",
            FIGURE_INPUT_EVIDENCE),
        "cross_sample_comparison": (
            ENABLED,
            "The transform is fitted across the samples together, so the "
            "values are on one comparable scale.",
            NORMALIZATION_EVIDENCE),
    }


CAPABILITY_MATRIX: Dict[str, Dict[str, Tuple[str, str, Evidence]]] = {
    RAW_COUNTS: _count_family(estimated=False),
    ESTIMATED_COUNTS: _count_family(estimated=True),
    CPM_OR_TPM: _depth_normalized((
        REQUIRES_INPUT,
        "CPM and TPM behave differently here and the matrix cannot say which "
        "this is. TPM divides by effective length before rescaling, which "
        "makes it the wrong basis for a cross-sample differential-expression "
        "comparison; CPM does not have that problem. Both rescale each sample "
        "to exactly one million, so the trace that would tell them apart is "
        "gone — your pipeline knows, and the numbers do not.",
        UNDECIDABLE_EVIDENCE)),
    TMM_SCALED_CPM: _depth_normalized((
        ENABLED,
        "This is what the scaling was for. Correcting library composition as "
        "well as depth is what makes a gene comparable between samples, and in "
        "the comparison the research cites this family controlled the "
        "false-positive rate where total-count and RPKM normalization did not.",
        NORMALIZATION_EVIDENCE)),
    FPKM: _depth_normalized((
        DISABLED,
        "FPKM and RPKM are not comparable across samples even in principle. "
        "Wagner, Kim and Lynch showed RPKM/FPKM violates the invariance a "
        "relative-molar-concentration measure has to satisfy, and Dillies and "
        "colleagues found total-count and RPKM normalization did not control "
        "the false-positive rate where TMM and median-of-ratios did. This is "
        "not a caution about precision; the quantity is not the same quantity "
        "in two samples.",
        COACHING_EVIDENCE)),
    VST: _stabilized("Variance-stabilized"),
    RLOG: _stabilized("Regularized-log"),
    MICROARRAY: {
        "count_model": (
            DISABLED,
            "This is a hybridization intensity rather than a count. The whole "
            "count toolchain does not apply — there is no library size, no "
            "dispersion to estimate and nothing for a negative binomial to "
            "model.",
            COACHING_EVIDENCE),
        "gaussian_workflow": (
            ENABLED,
            "limma was built for this assay and these values are its intended "
            "input: a linear model per probe with variance shrunk across "
            "probes, which is what makes a small number of arrays estimable at "
            "all.",
            COACHING_EVIDENCE),
        "per_sample_normalization": (
            DISABLED,
            "RMA already normalizes across arrays. Scaling again would be "
            "normalizing twice.",
            NORMALIZATION_EVIDENCE),
        "log_transform": (
            DISABLED,
            "RMA output is already log2. Taking a log of it would be a log of "
            "a log.",
            NORMALIZATION_EVIDENCE),
        "pca_and_clustering": (
            ENABLED,
            "Array intensities are a standard input to a sample PCA and to a "
            "sample-to-sample correlation heatmap.",
            FIGURE_INPUT_EVIDENCE),
        "cross_sample_comparison": (
            ENABLED,
            "RMA normalizes across the arrays together, so intensities are "
            "comparable between samples.",
            NORMALIZATION_EVIDENCE),
    },
    LOG_RATIO: {
        "count_model": (
            DISABLED,
            "There are negative values in this matrix, and any negative value "
            "rules out raw counts, CPM, TPM and FPKM. There is no count scale "
            "left here for a count model to work on.",
            NEGATIVES_RULE_EVIDENCE),
        "gaussian_workflow": (
            ENABLED,
            "A centred, roughly symmetric matrix is a Gaussian workflow's "
            "natural input. What it cannot tell you is what the values are "
            "centred against.",
            COACHING_EVIDENCE),
        "per_sample_normalization": (
            DISABLED,
            "These are ratios or standard scores rather than depths. There is "
            "no library size left in them to correct.",
            NORMALIZATION_EVIDENCE),
        "log_transform": (
            DISABLED,
            "Some of these values are negative, so a log of them is undefined "
            "— and they are already on a ratio or standard-score scale.",
            NORMALIZATION_EVIDENCE),
        "pca_and_clustering": (
            ENABLED,
            "A centred matrix is a direct PCA input, and at this width the "
            "sample-to-sample heatmap is the figure that replaces a "
            "feature-by-feature correlation matrix.",
            FIGURE_INPUT_EVIDENCE),
        "cross_sample_comparison": (
            REQUIRES_INPUT,
            "These values are already relative to something and the matrix "
            "does not say what. A log-ratio against a reference, a z-score "
            "within each gene and a batch-corrected residual mean three "
            "different things when compared across samples.",
            NORMALIZATION_EVIDENCE),
    },
}


def capability_rows(key: str) -> List[Dict[str, Any]]:
    """The capability matrix for one signature, as rows an interface renders.

    Never cut, and it says how many there are (`GUIDED-209`) — the count is
    added by the caller that serves it, beside a `showing` that equals it.
    """
    table = CAPABILITY_MATRIX[key]
    out = []
    for capability in CAPABILITIES:
        state, because, evidence = table[capability]
        out.append({"key": capability,
                    "label": CAPABILITY_LABELS[capability],
                    "state": state,
                    "state_label": STATE_LABELS[state],
                    "because": because,
                    **evidence.to_dict()})
    return out


def _reading(value: float) -> str:
    """A measured value as a reader would write it.

    `{:,.4g}` turns 26,471 into `2.647e+04` — correct, and not the sentence a
    person reads a count matrix's ceiling in. So: thousands separators above
    one, significant figures below it.
    """
    if value == 0:
        return "0"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 1:
        return f"{value:,.2f}".rstrip("0").rstrip(".")
    return f"{value:.3g}"


def _evidence_rows(m: Dict[str, Any], keys: Sequence[str]) -> List[Dict[str, Any]]:
    """The readings that drove the classification, in the app's own words.

    §02 asks the card to show *"the evidence that drove it"*. Each row is a
    measured quantity, its value already formatted, and the sentence saying what
    that value means — composed here so no interface has to know, for instance,
    that a floor share is about zero counts.

    **The library-size rows are omitted where the matrix has negatives**, and
    that is the recorded-absence rule rather than an oversight: a row sum over a
    matrix centred on zero is a number with no meaning, and `wide_assay.csv`
    reads a coefficient of variation of 900% for exactly that reason. Silence
    beats a true number that answers nothing.
    """
    rows: List[Dict[str, Any]] = []
    rows.append({
        "key": "shape",
        "label": "Shape",
        "value": f"{m['n_columns']:,} columns across {m['n_samples']:,} samples",
        "statement": (
            "Rows are samples and columns are features here, which is the "
            "transpose of the genomics convention — so every reading below is "
            "per sample where the field would say per gene, and a row total is "
            "a library size.")})
    if m["negatives"]:
        rows.append({
            "key": "negatives",
            "label": "Negative values",
            "value": f"lowest is {m['min']:,.2f}",
            "statement": (
                "Any negative value rules out raw counts, CPM, TPM and FPKM. "
                "That is asserted as a rule and not as a reading, so nothing "
                "below can reach past it.")})
    else:
        rows.append({
            "key": "negatives",
            "label": "Negative values",
            "value": "none",
            "statement": (
                "Nothing is below zero, so the rule that rules out raw counts, "
                "CPM, TPM and FPKM does not apply here.")})
        library = m["library_size"]
        fold = library["max"] / library["min"] if library["min"] else float("inf")
        rows.append({
            "key": "library_size",
            "label": "Library size per sample",
            "value": (f"{library['min']:,.0f} to {library['max']:,.0f} "
                      f"({fold:.1f}-fold, CV {library['cv']:.1%})"),
            "statement": (
                "The row total. Whether it is constant, near-constant or "
                "varying is what separates raw counts from CPM, from a "
                "composition-aware rescaling and from FPKM."
                if fold < float("inf") else
                "The row total, which one sample reads as zero.")})
    rows.append({
        "key": "integrality",
        "label": "Whole numbers",
        "value": ("every value" if m["all_integral"]
                  else f"{m['pct_integer']:.1%} of the non-zero values"),
        "statement": (
            "Read over the NON-ZERO values, deliberately. Over every cell it "
            f"would read {m['pct_integer_all']:.0%}, because a zero is a whole "
            f"number and {m['pct_zeros']:.0%} of this matrix is zero — a "
            "figure that describes the zeros and not the values.")})
    rows.append({
        "key": "zeros",
        "label": "Zeros",
        "value": f"{m['pct_zeros']:.1%} of cells",
        "statement": (
            "A count matrix is 20 to 60 per cent zeros, and everything derived "
            "from one inherits them — a zero count is a zero TPM.")})
    rows.append({
        "key": "range",
        "label": "Range",
        "value": f"{_reading(m['min'])} to {_reading(m['max'])}",
        "statement": (
            "The ceiling is most of the reading: past ten thousand is a count "
            "scale, and a ceiling in the teens is a value that has already "
            "been put on a log scale.")})
    # THE FLOOR ROW IS ABOUT A *NON-ZERO* FLOOR, and the guard is not cosmetic.
    # On raw counts 15% of the cells sit on zero, so an ungated row would have
    # said *"15.3% of cells sit on 0"* under the heading `Repeated floor` — true,
    # already said one row up as the zero fraction, and read as though the count
    # matrix carried a stabilizing floor it does not have.
    if m["floor_share"] >= _FLOOR_SHARE and m["min"] != 0:
        rows.append({
            "key": "floor",
            "label": "Repeated floor",
            "value": (f"{m['floor_share']:.1%} of cells sit on "
                      f"{_reading(m['min'])}"),
            "statement": (
                "A large share of the matrix on one exact value at the bottom "
                "is what a variance-stabilizing transform does to every zero "
                "count: they all land on the same number.")})
    if m["ids"]["probe_share"] > 0:
        rows.append({
            "key": "identifiers",
            "label": "Column identifiers",
            "value": f"{m['ids']['probe_share']:.0%} array probe IDs",
            "statement": (
                "Array probe identifiers rather than gene or transcript "
                "identifiers. That is half of the microarray signature and the "
                "numbers are the other half.")})
    if m["n_excluded"]:
        rows.append({
            "key": "excluded",
            "label": "Read as covariates, not features",
            "value": ", ".join(m["excluded"]),
            "statement": (
                "Whole-number columns lying entirely outside the range the "
                "rest of the matrix occupies. They are left out of the "
                "readings above, because one such column decides a maximum "
                "even though it barely moves a row total — and if one of them "
                "is a gene, the reading above is wrong and this is where you "
                "would see it.")})
    return rows


def data_type_card(df: pd.DataFrame,
                   minimum: int = 30) -> Optional[Dict[str, Any]]:
    """§02's *"what your numbers are"* card — the pack's most valuable artifact.

    *"Classification, confidence, the evidence that drove it, and a capability
    matrix showing which downstream steps are now enabled, disabled, or require
    input."*

    `None` where the table is not wide enough to be an expression matrix at all.
    A card that comes back with `read: False` is the other case and it is a real
    answer: the matrix was read and matched no signature, which §02 leaves open
    and which the app says out loud rather than rounding to the nearest row.
    """
    m = read_matrix(df, minimum=minimum)
    if m is None:
        return None
    verdict = _classify(m)
    keys = verdict["keys"]
    lead = keys[0] if keys else None
    if lead is None:
        return {
            "read": False,
            "reason": (
                "These columns do not match any of the nine shapes the "
                "research describes for an expression matrix. The app is not "
                "going to name the nearest one: the classification decides "
                "what is legal downstream, and a guess here is worse than no "
                "answer."),
            "block": {k: m[k] for k in ("n_columns", "n_samples", "excluded",
                                        "n_excluded")},
            "evidence": {"rows": _evidence_rows(m, keys),
                         "n": len(_evidence_rows(m, keys)),
                         "showing": len(_evidence_rows(m, keys))},
            **DATA_TYPE_EVIDENCE.to_dict(),
        }
    names = [SIGNATURE_NAMES[k] for k in keys]
    # `str.capitalize()` LOWERCASES THE TAIL, so it turns "TMM- or
    # median-of-ratios-scaled CPM" into "Tmm- … cpm" and "FPKM/RPKM" into
    # "Fpkm/rpkm". Every name here is an acronym or contains one.
    def _lead_capital(text: str) -> str:
        return text[:1].upper() + text[1:]
    confidence, confidence_because = CONFIDENCE[lead]
    undecidable = lead == CPM_OR_TPM
    ask = len(keys) > 1 or undecidable
    if undecidable:
        question = (
            "Which did your pipeline produce, CPM or TPM? Both rescale every "
            "sample to exactly one million and TPM divides by effective length "
            "before that rescale, so the trace is gone from the matrix — no "
            "test on these numbers separates them. You know; they do not.")
        label = "CPM or TPM, and the matrix cannot say which"
    elif ask:
        question = (
            f"Which tool produced this matrix? {_lead_capital(names[0])} and "
            f"{names[1]} are both non-integer, both non-negative and both have "
            f"varying totals per sample, and the research separates them by "
            f"maximum and skew, which overlap. What separates them here is the "
            f"spread of the library sizes, and that is evidence rather than a "
            f"determination.")
        label = f"{_lead_capital(names[0])} or {names[1]}"
    else:
        question = None
        label = _lead_capital(names[0])
    rows = _evidence_rows(m, keys)
    capabilities = capability_rows(lead)
    return {
        "read": True,
        "classification": {
            "keys": list(keys),
            "names": names,
            "label": label,
            "confidence": confidence,
            "confidence_because": confidence_because,
            "requires_input": bool(ask),
            "question": question,
            "coaching": COACHING[lead],
            "ruled_out": [SIGNATURE_NAMES[k] for k in verdict["ruled_out"]],
        },
        # BOTH LISTS UNCUT AND BOTH SAYING SO (`GUIDED-209`). `showing` equals
        # `n` because nothing is dropped, and it is served rather than implied
        # so that a later loop cutting one has to change a number a reader can
        # see.
        "evidence": {"rows": rows, "n": len(rows), "showing": len(rows)},
        "capabilities": {"rows": capabilities, "n": len(capabilities),
                         "showing": len(capabilities)},
        "scope_note": _SCOPE_NOTE,
        "block": {k: m[k] for k in ("n_columns", "n_samples", "excluded",
                                    "n_excluded")},
        "measured": {k: m[k] for k in (
            "min", "max", "median", "negatives", "pct_zeros", "pct_integer",
            "pct_integer_all", "all_integral", "floor_share", "library_size")},
        **DATA_TYPE_EVIDENCE.to_dict(),
    }


def _genomics_data_type(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """§02, as a finding — so the reading reaches the Explore stack too.

    The CARD is served on its own route and carries the capability matrix. This
    is the same reading in the engine's own shape, which is what puts it in
    front of a person who never opens the card: the classification, what drove
    it, and the branch of coaching it selects.
    """
    card = data_type_card(df)
    if card is None or not card.get("read"):
        return None
    reading = card["classification"]
    m = card["measured"]
    lead = reading["keys"][0]
    closed = [row["label"] for row in card["capabilities"]["rows"]
              if row["state"] == DISABLED]
    detail = (
        f"{card['block']['n_columns']:,} columns across "
        f"{card['block']['n_samples']:,} samples. "
        f"{reading['confidence_because']}")
    # A ROW TOTAL OVER A MATRIX CENTRED ON ZERO IS A NUMBER WITH NO MEANING —
    # `wide_assay.csv` reads a coefficient of variation of 900% for exactly that
    # reason. Trap 9: return nothing rather than a value, and say why in a
    # sentence rather than by leaving a key out, because an absent key and a
    # question nobody asked look the same.
    spread = None if m["negatives"] else round(m["library_size"]["cv"], 4)
    spread_note = (
        "Not reported: this matrix has negative values, so a row total is not "
        "a library size and its spread would be a number about nothing."
        if m["negatives"] else
        "How much the row totals vary across samples. It is what separates a "
        "matrix nothing has normalized from one that has been rescaled.")
    return _finding(
        "pack::genomics::data_type", "warning" if closed else "info",
        f"What your numbers are: {reading['label']}",
        detail,
        (reading["coaching"] + (" " + reading["question"]
                                if reading["question"] else "")),
        confidence=reading["confidence"], pack=GENOMICS, marker="convention",
        evidence=DATA_TYPE_EVIDENCE,
        claims=(
            Claim("classification",
                  f"These values read as {reading['label'].lower()}, and what "
                  f"that closes off downstream follows from it rather than "
                  f"from anything the app prefers.",
                  COACHING_EVIDENCE),
            Claim("hard_rule",
                  "Any negative value rules out raw counts, CPM, TPM and "
                  "FPKM. That is a rule and it runs before every reading.",
                  NEGATIVES_RULE_EVIDENCE),
        ),
        columns=list(card["block"]["excluded"]),
        params={"signatures": list(reading["keys"]),
                "requires_input": reading["requires_input"],
                "lead": lead,
                "n_features": card["block"]["n_columns"],
                "library_size_cv": spread,
                "library_size_cv_note": spread_note,
                "pct_zeros": round(m["pct_zeros"], 4),
                "pct_integer_nonzero": round(m["pct_integer"], 4),
                "matrix_max": round(m["max"], 4),
                "negatives": m["negatives"],
                # THE CLOSED ROWS AS SENTENCES, not as keys. A list of
                # capability names here would be `GUIDED-207` in the record
                # layer: nothing could render it and nothing could check it.
                "closed": closed})
# ── genomics · gene identifiers · GENOMICS_PACK.md §01, "Gene IDs" ───────────
#
# §01 asks for four readings off one classification — *"classify vocabulary;
# report version suffixes present …, duplicate IDs after symbol mapping
# (many-to-one), mixed vocabularies, and Excel corruption"* — so the
# classification is one function and the four findings read it. Four detectors
# rather than one finding with four paragraphs, because their severities are not
# the same thing: three of them describe identifiers that need reconciling before
# a join, and the fourth describes data that has already been destroyed.

GENE_ID_EVIDENCE = Evidence(
    status=SETTLED,
    source="research/GENOMICS_PACK.md#Gene IDs")

#: Identifier grammars, ordered so the specific ones are tried before the
#: general ones — `1000000_at` is an Affymetrix probe set and not the Entrez gene
#: 1000000, and the only thing that says so is that `_at` was tested first.
#:
#: MATCHED AS WHOLE STRINGS, never as substrings. That is the rule
#: `tests/test_a_name_registry_matches_exactly_or_says_nothing.py` exists for,
#: and it bites here: `MARCH1` contains `MAR`, `ENSG00000141510` contains
#: `ENSG0000014151`, and every symbol contains a shorter symbol.
_ID_GRAMMARS: Tuple[Tuple[str, "re.Pattern"], ...] = (
    ("Ensembl accessions", re.compile(r"^ENS(?:[A-Z]{3,4})?[EGTP]\d{11}(?:\.\d+)?$")),
    ("RefSeq accessions", re.compile(r"^[NX][MRP]_\d{5,9}(?:\.\d+)?$")),
    ("Affymetrix probe sets", re.compile(r"^\d+_(?:[a-z]{1,2}_)?at$")),
    ("Illumina probes", re.compile(r"^ILMN_\d{6,9}$")),
    ("Entrez gene IDs", re.compile(r"^\d{1,9}$")),
    ("HGNC symbols", re.compile(r"^[A-Z][A-Z0-9]{1,14}(?:-[A-Z0-9]{1,6})?$")),
)

#: The two vocabularies whose identifiers legitimately carry a `.N` version.
#: A `.1` on anything else is far more likely to be pandas de-duplicating a
#: repeated column name than an annotation release, and reading it as a version
#: would report a defect the file does not have.
_VERSIONED_VOCABULARIES = ("Ensembl accessions", "RefSeq accessions")

_VERSION_SUFFIX = re.compile(r"\.(\d+)$")

#: What Excel makes of a gene symbol. Both directions, because the rendering is
#: locale-dependent and `Mar-1` and `1-Mar` are the same accident.
_EXCEL_DATE = re.compile(
    r"^(?:\d{1,2}[-/](?:jan|feb|mar|apr|may|jun|jul|aug|sept?|oct|nov|dec)"
    r"|(?:jan|feb|mar|apr|may|jun|jul|aug|sept?|oct|nov|dec)[-/]\d{1,2})"
    r"(?:[-/]\d{2,4})?$", re.I)

#: Excel serial numbers for 1990-01-01 and 2035-12-31, from the `1899-12-30`
#: epoch. An autoconverted symbol lands in *the year the file was opened*, and
#: outside this band the reading is not worth making: there were no supplementary
#: Excel gene lists before 1990, and after 2035 is the future.
_EXCEL_SERIAL_WINDOW = (32874, 49674)

#: How much of the identifier set must be HGNC symbols before a bare five-digit
#: integer is read as a corrupted date rather than as an Entrez gene ID. **This
#: number is the app's and not the research's** — §01 names the corruption and
#: names no rule for telling `44621`-the-serial from `44621`-the-Entrez-ID,
#: because nothing in the string distinguishes them. What distinguishes them is
#: the company they keep, and this is the threshold on "company".
_SYMBOL_DOMINANCE = 0.5

#: Two floors before any of this is said at all. A table with eleven recognized
#: identifiers among four hundred numeric columns is not a gene matrix and the
#: readings below would be about noise.
_MIN_RECOGNIZED = 20
_MIN_RECOGNIZED_SHARE = 0.4


@dataclass(frozen=True)
class GeneIdReading:
    """One classification of an identifier set, read by four detectors.

    `vocabularies` maps a grammar name to the identifiers matching it, with the
    Excel-corrupted ones already removed — a corrupted symbol is not a member of
    the symbol vocabulary, it is the wreck of one.
    """
    columns: Tuple[str, ...]
    vocabularies: Dict[str, Tuple[str, ...]]
    unclassified: Tuple[str, ...]
    versioned: Tuple[str, ...]
    duplicate_bases: Dict[str, Tuple[str, ...]]
    excel_dates: Tuple[str, ...]
    excel_serials: Tuple[str, ...]

    @property
    def recognized(self) -> int:
        return (sum(len(v) for v in self.vocabularies.values())
                + len(self.excel_dates) + len(self.excel_serials))

    @property
    def corrupted(self) -> Tuple[str, ...]:
        return tuple(self.excel_dates) + tuple(self.excel_serials)


def _base_identifier(name: str) -> str:
    """The identifier with its version stripped, for collision counting."""
    return _VERSION_SUFFIX.sub("", name)


def read_gene_ids(df: pd.DataFrame) -> Optional[GeneIdReading]:
    """Classify the identifier set, or decline where there is not one.

    **The identifiers are the COLUMN NAMES here**, and that is orientation
    rather than an implementation detail. §01 is written for the field
    convention, genes in rows; this app's tables are samples in rows, so the
    gene identifiers arrive as a header. A file with an identifier *column* is
    the orientation problem §01 handles before this diagnostic runs, and
    `test_a_transposed_assay_table_is_turned_around_before_diagnosis` is where
    that lives.

    Returns `None` — never a reading with empty fields — where the numeric block
    is too narrow to be an assay or where too little of it belongs to any gene-ID
    grammar. `nhanes_dietary.csv`'s `DR1TKCAL` and `WTDRD1` are shaped exactly
    like HGNC symbols and there are ten of them, which is why the floor is a
    count *and* a share.
    """
    cols = _numeric(df)
    if not _is_assay_wide(df):
        return None

    buckets: Dict[str, List[str]] = {}
    unclassified: List[str] = []
    dates: List[str] = []
    for name in cols:
        if _EXCEL_DATE.match(name):
            dates.append(name)
            continue
        for vocabulary, grammar in _ID_GRAMMARS:
            if grammar.match(name):
                buckets.setdefault(vocabulary, []).append(name)
                break
        else:
            unclassified.append(name)

    # THE SERIAL AMBIGUITY, resolved by the company the integer keeps and by
    # nothing else. A five-digit integer is both an Excel serial and a perfectly
    # ordinary Entrez gene ID. Two conditions, and both are readings of the rest
    # of the set: no integer identifier falls OUTSIDE the serial window (one that
    # does proves the table uses Entrez), and the set is mostly HGNC symbols
    # (which is what Excel destroys). Where either fails, the integers stay
    # Entrez and this finding is not made — `DOMAIN_SCIENCE.md` §01.2's litmus,
    # applied to the detection rather than to the action.
    integers = buckets.get("Entrez gene IDs", [])
    low, high = _EXCEL_SERIAL_WINDOW
    in_window = [n for n in integers if low <= int(n) <= high]
    outside = [n for n in integers if n not in set(in_window)]
    symbols = buckets.get("HGNC symbols", [])
    non_integer = sum(len(v) for k, v in buckets.items() if k != "Entrez gene IDs")
    share = len(symbols) / max(non_integer + len(outside), 1)
    serials: List[str] = []
    if in_window and not outside and share >= _SYMBOL_DOMINANCE:
        serials = in_window
        buckets.pop("Entrez gene IDs", None)

    versioned = tuple(
        n for vocabulary in _VERSIONED_VOCABULARIES
        for n in buckets.get(vocabulary, [])
        if _VERSION_SUFFIX.search(n))

    collisions: Dict[str, Tuple[str, ...]] = {}
    for vocabulary, members in buckets.items():
        seen: Dict[str, List[str]] = {}
        for name in members:
            seen.setdefault(_base_identifier(name), []).append(name)
        for base, names in seen.items():
            if len(names) > 1:
                collisions[base] = tuple(names)

    reading = GeneIdReading(
        columns=tuple(cols),
        vocabularies={k: tuple(v) for k, v in sorted(buckets.items())},
        unclassified=tuple(unclassified), versioned=versioned,
        duplicate_bases=collisions, excel_dates=tuple(dates),
        excel_serials=tuple(serials))
    if reading.recognized < _MIN_RECOGNIZED:
        return None
    if reading.recognized / max(len(cols), 1) < _MIN_RECOGNIZED_SHARE:
        return None
    return reading


def _gene_id_excel_corruption(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Gene symbols Excel turned into dates. **The hard stop.**

    `DOMAIN_SCIENCE.md` §01.2's shape exactly: high-confidence detection,
    irreversible-if-wrong action, and no signal in the data that resolves the
    ambiguity. `1-Mar` could have been `MARCH1`; it could also have been a date
    somebody meant. `MARCH1` and `MARC1` are both real genes and only one of them
    converts. And a serial carries no month name at all.

    So the finding proposes nothing and pre-selects nothing —
    `fix_kind="none"`, which `router._is_repairable` reads as a report rather
    than a fork, so this cannot become a question the interview asks.
    """
    reading = read_gene_ids(df)
    if reading is None or not reading.corrupted:
        return None
    damaged = reading.corrupted
    return _finding(
        "pack::genomics::gene_id_excel_corruption", "critical",
        (f"{len(damaged):,} gene identifiers have been converted to dates"
         + (" and serial numbers" if reading.excel_serials else "")),
        (f"`{'`, `'.join(damaged[:6])}`"
         + ("" if len(damaged) <= 6 else
            f" and {len(damaged) - 6:,} more, {len(damaged):,} of "
            f"{len(reading.columns):,} numeric columns in all")
         + f". {len(reading.excel_dates):,} are date strings and "
           f"{len(reading.excel_serials):,} are bare serial numbers, which is "
           f"the same accident after a numeric cell format. Ziemann, Eren & "
           f"El-Osta (*Genome Biology* 17:177, 2016) found gene-name conversion "
           f"errors in ~20% of papers carrying supplementary Excel gene lists, "
           f"and Abeysooriya et al. (*PLoS Comput Biol* 2021) found the rate had "
           f"risen. HGNC renamed `SEPT*` to `SEPTIN*` and `MARCH*` to `MARCHF*` "
           f"partly because of it. **TurboTab has not repaired any of these and "
           f"will not.**"),
        ("The original symbols are gone, and which ones they were is not "
         "recoverable from this file. `1-Mar` is what Excel makes of `MARCH1`, "
         "and it is also what Excel makes of a date somebody typed — nothing "
         "here separates those. A serial is worse: it carries no month name at "
         "all. A repair that guessed would put a named gene into a results "
         "table on the strength of a guess, which is the one failure a gene "
         "list cannot survive. Re-export from the source with the identifier "
         "column formatted as text, or import it as text."),
        confidence="high", pack=GENOMICS, marker="offered",
        evidence=GENE_ID_EVIDENCE,
        columns=list(damaged),
        params={
            "columns": list(damaged),
            "columns_shown": min(6, len(damaged)),
            "columns_total": len(damaged),
            "n_date_strings": len(reading.excel_dates),
            "n_serial_numbers": len(reading.excel_serials),
            "date_strings": list(reading.excel_dates),
            "serial_numbers": list(reading.excel_serials),
            "serial_window": list(_EXCEL_SERIAL_WINDOW),
            # NAMED IN THE PAYLOAD, not only in the prose. `GUIDED-064`'s class:
            # the machine-readable form must not be lossier than the sentence,
            # and *never auto-repair* is the whole content of this finding.
            "hard_stop": "never_auto_repair_gene_symbols",
            "hard_stop_because": (
                "The research is explicit — never auto-repair, report and "
                "stop. Nothing in the file says which symbol a date or a "
                "serial used to be, so a repair would be a guess wearing the "
                "app's authority."),
        },
        fix_label="", fix_kind="none")


def _gene_id_versions(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Versioned accessions, which join to unversioned annotation as nothing.

    The failure §01 names is that the join **fails silently** — the rows do not
    error, they disappear — so the number that matters is how many identifiers
    would be dropped, and it is in the sentence.
    """
    reading = read_gene_ids(df)
    if reading is None or not reading.versioned:
        return None
    versioned = reading.versioned
    families = sorted(
        v for v in _VERSIONED_VOCABULARIES if reading.vocabularies.get(v))
    return _finding(
        "pack::genomics::gene_id_versions", "warning",
        f"{len(versioned):,} identifiers carry a version suffix",
        (f"`{'`, `'.join(versioned[:4])}`"
         + ("" if len(versioned) <= 4 else
            f" and {len(versioned) - 4:,} more")
         + f" — {len(versioned):,} of {len(reading.columns):,} numeric columns, "
           f"all of them {' or '.join(families)}. The suffix is the "
           f"annotation release the identifier was written against."),
        ("Joined against an unversioned annotation table, every one of these "
         "matches nothing — and the join does not error, it drops the row. A "
         "differential-expression result computed after that silent loss is "
         "computed on the genes that happened to survive, and the gene count in "
         "the methods section is the count before the loss. Stripping the "
         "suffix is usually right and it is not always right: two versions of "
         "one accession collapse onto each other, which is the reading beside "
         "this one."),
        confidence="high", pack=GENOMICS, marker="offered",
        evidence=GENE_ID_EVIDENCE,
        columns=list(versioned[:8]),
        params={"columns": list(versioned),
                "columns_shown": min(4, len(versioned)),
                "columns_total": len(versioned),
                "n_versioned": len(versioned),
                "n_identifiers": len(reading.columns),
                "vocabularies": families},
        fix_label="", fix_kind="none")


def _gene_id_duplicates(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """One gene, two columns — the many-to-one §01 warns about.

    **The honest scope, stated in the finding rather than only here.** §01 says
    *"duplicate IDs after symbol mapping"*, and this app has no symbol-mapping
    table: it cannot see two Ensembl accessions that map to one HGNC symbol.
    What it can see is a collision among the identifiers **as written**, which is
    what merging two annotation releases produces and is the commonest way the
    many-to-one arrives. The rest of the class is out of reach and the finding
    says which half it is reporting.
    """
    reading = read_gene_ids(df)
    if reading is None or not reading.duplicate_bases:
        return None
    bases = sorted(reading.duplicate_bases)
    doubled = sum(len(reading.duplicate_bases[b]) for b in bases)
    example = reading.duplicate_bases[bases[0]]
    return _finding(
        "pack::genomics::gene_id_duplicates", "warning",
        f"{len(bases):,} accessions appear more than once",
        (f"`{bases[0]}` is here as `{'` and `'.join(example[:2])}`"
         + ("" if len(bases) == 1 else
            f", and {len(bases) - 1:,} other accession"
            f"{'s' if len(bases) > 2 else ''} repeat the same way")
         + f" — {doubled:,} columns standing for {len(bases):,} genes. That is "
           f"what merging two annotation releases produces."),
        ("The same gene counted twice is counted twice by everything "
         "downstream: it is two rows in a multiple-testing correction, two "
         "features a regularized model can split its coefficient across, and "
         "two entries in a ranked list a reader will read as two findings. "
         "Which copy to keep depends on which annotation release the study "
         "means, and that is not in this file. This is the half of the "
         "many-to-one the app can see — identifiers that collide **as "
         "written**. Two distinct accessions that map to one gene symbol need "
         "the mapping table, which the app does not have."),
        confidence="high", pack=GENOMICS, marker="offered",
        evidence=GENE_ID_EVIDENCE,
        columns=[c for b in bases[:3] for c in reading.duplicate_bases[b]],
        params={"bases": bases,
                "bases_shown": min(1, len(bases)),
                "bases_total": len(bases),
                "n_duplicate_bases": len(bases),
                "n_duplicate_columns": doubled,
                "duplicates": {b: list(reading.duplicate_bases[b])
                               for b in bases},
                "covers": "identifiers that collide as written",
                "does_not_cover": "many-to-one that appears only after mapping "
                                  "to gene symbols, which needs a mapping table "
                                  "the app does not have"},
        fix_label="", fix_kind="none")


def _gene_id_mixed_vocabulary(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Two identifier grammars in one feature block.

    The floor is a count and a share together, because one stray `TP53` among
    four hundred Ensembl accessions is a typo and not a second vocabulary.
    """
    reading = read_gene_ids(df)
    if reading is None:
        return None
    total = sum(len(v) for v in reading.vocabularies.values())
    present = {name: members for name, members in reading.vocabularies.items()
               if len(members) >= 5 and len(members) / max(total, 1) >= 0.05}
    if len(present) < 2:
        return None
    ordered = sorted(present, key=lambda k: -len(present[k]))
    parts = ", ".join(f"{len(present[k]):,} {k}" for k in ordered)
    return _finding(
        "pack::genomics::gene_id_mixed_vocabulary", "warning",
        f"The feature names use {len(present)} different identifier vocabularies",
        (f"{parts} — {total:,} classified identifiers across "
         f"{len(reading.columns):,} numeric columns"
         + ("" if not reading.unclassified else
            f", with {len(reading.unclassified):,} belonging to no grammar the "
            f"app knows")
         + "."),
        ("Each vocabulary joins to a different annotation table, so there is no "
         "single join that annotates all of these features — and a merge that "
         "silently keeps only the vocabulary it recognized would drop the rest "
         "without erroring. Reconciling them means knowing which release "
         "produced each block, which is a fact about how the file was "
         "assembled rather than a fact in the file. The app reports the mix and "
         "maps nothing."),
        confidence="high", pack=GENOMICS, marker="offered",
        evidence=GENE_ID_EVIDENCE,
        columns=[present[k][0] for k in ordered],
        params={"vocabularies": {k: len(present[k]) for k in ordered},
                "examples": {k: list(present[k][:3]) for k in ordered},
                "examples_shown_per_vocabulary": 3,
                "n_classified": total,
                "n_unclassified": len(reading.unclassified),
                "n_columns": len(reading.columns)},
        fix_label="", fix_kind="none")


# ─────────────────────────────────────────────────────────────────────────────
# Reframing — a pack changes the ANSWER, not the question
# ─────────────────────────────────────────────────────────────────────────────

_ASSAY_PACKS = (METABOLOMICS, GENOMICS)


def reframe(findings: List[Dict[str, Any]], lens: Sequence[str],
            df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Annotate engine findings the lens reads differently. Never deletes one.

    **An extension point rather than a list of finding IDs** (`GUIDED-028`'s
    sibling). The first version covered exactly two hardcoded ids, so a pack
    could not contribute a reframing at all — and `IMPORT-267`, a `critical`
    asserting that a column of education levels mixes measurement units, was
    exactly the false alarm a lens should kill and could not. Every reframing
    now comes from a `Reframing` declared on a pack, and adding one is adding a
    declaration.

    **Annotation, not suppression**, and the distinction is the guard. A pack
    that DELETED `wide_repeated_measures` would also delete it on
    `clinic_visits.csv`, where `bp_1`/`bp_2`/`bp_3` is exactly what the finding
    is for and the reading is correct. What changes here is the ANSWER —
    severity drops, the offer is withdrawn, and the reason is carried on the
    finding so the record can say which lens said so.

    Returns a new list; the input findings are copied rather than mutated,
    because two callers reading one finding must not see each other's edits.
    """
    chosen = normalize_quiet(lens)
    if not chosen or chosen == [OTHER]:
        return list(findings)

    out: List[Dict[str, Any]] = []
    for raw in findings:
        f = dict(raw)
        applied: List[str] = []
        for key in chosen:
            for rule in PACKS[key].reframings:
                try:
                    if not rule.matches(f, df):
                        continue
                except Exception:
                    # A reframing that cannot read this finding declines it.
                    # Losing one reading must not lose the whole diagnosis, and
                    # it must not be quiet about having failed either.
                    from turbotab import devchecks
                    devchecks.swallowed(
                        f"packs.{key}::reframing", _last_exception(),
                        f"a reframing of {f.get('id')!r} was skipped, so the "
                        f"engine's original reading stands unannotated")
                    continue
                if key not in applied:
                    applied.append(key)
                f["severity"] = rule.severity
                f["fix_kind"] = "none"
                f["fix_label"] = ""
                f["reframe_note"] = rule.note(f, df)
                if rule.title is not None:
                    f["title"] = rule.title(f, df)
        if applied:
            f["reframed_by"] = applied
        out.append(f)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# The packs
# ─────────────────────────────────────────────────────────────────────────────

DATASET = "dataset"
COLUMNS = "columns"


@dataclass(frozen=True)
class Prior:
    """One thing a pack believes about a question that already exists.

    A prior is not a finding and never becomes a question. It changes what the
    existing question DEFAULTS to, and states why.

    **Scope is the field `GUIDED-027` was filed about.** `missingness_direction`
    is not a fact about a table — it is a fact about each column. NHANES-shaped
    data holds dietary columns, lab columns and questionnaire columns side by
    side, and a dataset-level *"below the detection limit"* prior is wrong for
    most of them. So a prior says which it is, and a `columns`-scoped prior
    names its `detector`: the columns it applies to are **the ones that
    detector identified**, not every column in the table.

    `model_ranking` is genuinely a fact about the dataset — p ≫ n is a property
    of the shape — and stays `DATASET`.
    """
    question: str
    marker: str                     # derived | convention | offered
    reason: str
    # WHERE THE FIELD STANDS, and where that was read (`GUIDED-047`). `marker`
    # is the app's confidence and `evidence` is the field's; a prior with the
    # first and not the second is the app being uniformly confident, which is
    # the state all four research threads independently asked to end.
    evidence: Optional["Evidence"] = None
    scope: str = DATASET
    # For a `COLUMNS`-scoped prior: the detector whose `params["columns"]` names
    # the columns this prior applies to. A columns-scoped prior with no detector
    # would apply to everything, which is the defect being repaired.
    detector: Optional[str] = None
    values: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.marker not in ("derived", "convention", "offered"):
            raise PackError(
                f"{self.question}: marker must be derived, convention or "
                f"offered. The marker governs the treatment, so a prior "
                f"without one cannot be rendered honestly.")
        if self.scope == COLUMNS and not self.detector:
            raise PackError(
                f"{self.question}: a column-scoped prior must name the "
                f"detector whose columns it applies to. Without one it applies "
                f"to every column, which is `GUIDED-027` restated.")
        # THE BADGE OBLIGATION IS SHARED WITH FINDINGS AND REFUSALS, and that is
        # `GUIDED-059`'s repair rather than a tidy-up. This check lived here and
        # only here, so priors were guarded and the two things a user actually
        # reads were not. `_check_badge` carries the rendering-obligation-first
        # ordering this method discovered: written the other way round, the
        # specific message about pre-selection is unreachable, because `derived`
        # and `convention` both fail the compatibility table first.
        _check_badge(self.question, self.evidence, self.marker)
        if len(self.reason) <= 40:
            raise PackError(
                f"{self.question}: a prior states its reason. That reason is "
                f"what the user reads beside the default, so a prior without "
                f"one raises confidence without earning it.")


@dataclass(frozen=True)
class Reframing:
    """A finding this pack reads differently, and the sentence it reads it with.

    **The extension point `GUIDED-028`'s sibling `reframe()` did not have.** The
    first version covered two hardcoded finding IDs, so a pack could not
    contribute a reframing at all — and `IMPORT-267`, a `critical` asserting
    that a column of education levels mixes measurement units, is exactly the
    false alarm a lens should kill and could not.

    `matches` receives the finding and the frame and returns True when this
    pack reads it differently. `annotate` never deletes: deleting
    `wide_repeated_measures` would also delete it on `clinic_visits.csv`, where
    `bp_1`/`bp_2`/`bp_3` is precisely what the finding is for.
    """
    matches: Callable[[Dict[str, Any], pd.DataFrame], bool]
    note: Callable[[Dict[str, Any], pd.DataFrame], str]
    title: Optional[Callable[[Dict[str, Any], pd.DataFrame], str]] = None
    severity: str = "info"




@dataclass(frozen=True)
class LooksFor:
    """One thing this pack will look for, nameable before the user answers.

    **`GUIDED-039`.** All six lens options carried the same hover string —
    `effectOf("set_lens", null)` — so hovering *Metabolomics or proteomics* said
    exactly what hovering *Dietary intake* said, and the one moment where the
    app could explain why the answer matters explained nothing. Picking a lens
    is a bet on what the app will then notice, and the app knows what it will
    notice.

    Two rules, both load-bearing.

    **`phrase` is a noun phrase, never a claim.** *"Your missing values cluster
    in the lowest-abundance features"* is a finding — it asserts something about
    a table nobody has looked at through this lens yet. On a hover, before the
    question is answered, that would be the governing rule's own violation in
    the smallest possible place. So the hover names what will be *looked for*
    and the finding remains the only thing that says what was *found*.

    **`source` binds it to the detector.** It is the finding id the detector
    emits, or `prior::<question>` where a pack sets a prior without a detector,
    or `question::<key>` for the one question a pack is allowed to add. That is
    what makes this a registry rather than a second description sitting beside
    the first: `test_a_pack_names_what_it_will_look_for` asserts the two sets
    match in both directions, which is exactly the key-match test
    `FEATURE_PARITY.md` says the `theory_anchors`/`theory_demos` pair is missing
    and is the most fragile thing in the app for want of.
    """
    source: str
    phrase: str


@dataclass(frozen=True)
class Pack:
    key: str
    label: str
    # What this pack will look for, in its own words, before anything is found.
    looks_for: Tuple[LooksFor, ...] = ()
    detectors: Tuple[Callable[[pd.DataFrame], Optional[Dict[str, Any]]], ...] = ()
    # Priors the pack sets on questions that already exist.
    priors: Tuple[Prior, ...] = ()
    # Findings this pack reads differently. Declared, not hardcoded in
    # `reframe()`.
    reframings: Tuple[Reframing, ...] = ()
    # WHERE THIS PACK DECLINES TO BE CONFIDENT. `METABOLOMICS_PACK.md` §11's
    # shape, and it is a field on `Pack` rather than a metabolomics-only global
    # because every research file has the same section under a different name —
    # `GENOMICS_PACK.md` and `CLINICAL_SURVEY_PACK.md` both carry one, and a
    # structure that fit exactly one pack would be the parallel dict
    # `GUIDED-025` is about. Empty is the honest state for a pack whose §11 has
    # not been built; it is not a claim that the pack has nothing to hedge.
    hedges: Tuple["Hedge", ...] = ()
    # THE RECIPE TABLE IS CANONICAL for anything that resolves to a variant of
    # a preprocessing operation (`GUIDED-025`). A pack's variant preferences
    # live here as `recipes.Operation` / `recipes.Default` and are registered at
    # pack-load time — never as a parallel dict that nothing resolves. The
    # callable is deferred so importing this module does not mutate the recipe
    # table as a side effect.
    recipes: Optional[Callable[[], None]] = None


# ─────────────────────────────────────────────────────────────────────────────
# The recipe table is where a pack's VARIANT preferences live
#
# `GUIDED-025`: `recipes.py` exposed `register_operation` and `register_default`
# with specificity resolution, origin tracking and a refusal for silent
# shadowing, all proven by the fake-pack test — and no real pack called either,
# while the packs declared their variant preferences in a parallel dict nothing
# resolved. Two extension mechanisms that did not meet.
#
# **The recipe table is canonical.** A prior that names an operation and a
# variant belongs there, because that is the structure that resolves it, records
# where it came from, and refuses to shadow core silently. `priors()` reads both
# homes so a consumer still has one entry point, and each fact has exactly one.
#
# Registration is deferred until a pack is LOADED — importing this module must
# not mutate the recipe table, or every test that touches recipes inherits five
# packs it never asked for.
# ─────────────────────────────────────────────────────────────────────────────

_LOADED: set = set()


def _metabolomics_recipes() -> None:
    from turbotab import recipes as _rec

    # Pareto scaling is not a variant core knows, so the operation is SHADOWED
    # — deliberately, with the flag `register_operation` demands. Everything
    # except the variant list and the pushed pair is copied from core rather
    # than reworded: a shadow that quietly rewrites the litmus answer would be
    # a pack changing a classification nobody can see.
    core = _rec.operation("scale")
    _rec.register_operation(
        _rec.Operation(
            key="scale", label=core.label,
            variants=tuple(core.variants) + ("pareto",),
            determinacy=core.determinacy, scope=core.scope,
            because=core.because, applies_to=core.applies_to,
            origin=f"{METABOLOMICS}_pack",
            pushed_alternatives=tuple(core.pushed_alternatives)
            + (("pareto", "standard"),)),
        replace_existing=True)
    _rec.register_default(_rec.Default(
        operation="scale", variant="pareto",
        selector="caps:requires_scaled_numeric",
        origin=f"{METABOLOMICS}_pack",
        reason=("The field convention here is Pareto scaling. Auto-scaling "
                "gives every feature equal weight including noise-dominated "
                "low-abundance ones; dividing by the square root of the "
                "standard deviation retains some magnitude information. A "
                "defensible compromise, not a fact — auto-scaling is offered "
                "beside it.")))
    _rec.register_default(_rec.Default(
        operation="power", variant="log1p", selector="*",
        origin=f"{METABOLOMICS}_pack",
        reason=("Concentrations are bounded below by zero and combine "
                "multiplicatively, so the resulting distribution is log-normal "
                "by construction rather than by convention. This is the one "
                "transform here that is derived rather than chosen.")))


def _survey_recipes() -> None:
    from turbotab import recipes as _rec
    _rec.register_default(_rec.Default(
        operation="encode", variant="ordinal", selector="*",
        origin=f"{SURVEY}_pack",
        reason=("The order comes from the instrument rather than from the "
                "data, so an integer code preserves a real ordering instead of "
                "inventing one. One-hot would spend a column per response "
                "level and throw the ordering away.")))


# Genomics registers NOTHING, and that is the position rather than an omission.
# CPM, TPM and VST are not interchangeable; there is no `normalize` operation to
# hold a default and no default to put in one. The considered refusal is
# recorded as a prior with `variant: None`, because an absent key would be
# indistinguishable from a pack that never asked the question.


PACKS: Dict[str, Pack] = {
    METABOLOMICS: Pack(
        key=METABOLOMICS, label=LENS_LABELS[METABOLOMICS],
        hedges=METABOLOMICS_HEDGES,
        # ORDERED BY §01's OWN ORDER — roles, then run order and design, then
        # value states — because `findings()` preserves declaration order and
        # that order is what a reader meets on the page. The three that predate
        # L50 stay first: they are the ones the fixture's companion document
        # names, and reordering them would move four assertions in
        # `test_the_fixtures_are_what_their_companions_claim` for no reason.
        # `_redundancy` (L50-E) is first and the placement is an argument:
        # it is the only detector here that computes over the MATRIX rather
        # than reading names and dtypes, so it is the one most likely to have
        # bent the abstraction — `LOOP.md` §02's hardest-first, judged by that
        # rather than by effort. It also answers the question a reader asks
        # before any of the others: how many metabolites do I actually have.
        # Merged from two worktrees that could not see each other.
        detectors=(_redundancy,
                   _left_censored, _acquisition_order, _pooled_qc,
                   _sample_roles_finding, _no_pooled_qc,
                   _acquisition_design, _no_run_order, _repeated_subjects,
                   _zeros_or_missing, _already_transformed,
                   _duplicate_ids, _empty_blocks, _ion_modes),
        looks_for=(
            LooksFor("pack::metabolomics::left_censored",
                     "missing values clustering in the lowest-abundance "
                     "features, which is usually a detection limit rather "
                     "than randomness"),
            LooksFor("pack::metabolomics::run_order",
                     "a run-order column, and instrument drift that tracks it"),
            LooksFor("pack::metabolomics::pooled_qc",
                     "pooled quality-control rows, which are not participants "
                     "and must not be modeled"),
            LooksFor("pack::metabolomics::redundancy",
                     "groups of features that rise and fall together, which is "
                     "one compound wearing several feature names — and how many "
                     "independent quantities the panel really holds"),
            # NOUN PHRASES, never claims. `LooksFor` is read on a hover BEFORE
            # the lens question is answered, so "your blanks are non-detections"
            # there would be the governing rule broken in the smallest possible
            # place. Each of these names a thing to be looked FOR.
            LooksFor("pack::metabolomics::sample_roles",
                     "blanks, calibrants, dilution series and system-suitability "
                     "injections among the rows, by the field's naming "
                     "conventions"),
            LooksFor("pack::metabolomics::no_pooled_qc",
                     "and whether pooled QCs are absent, which is the one "
                     "omission that cannot be repaired after the run"),
            LooksFor("pack::metabolomics::acquisition_design",
                     "the acquisition and design columns — injection order, "
                     "batch, plate, well, plex, polarity — including a run "
                     "order derivable from an acquisition timestamp"),
            LooksFor("pack::metabolomics::no_run_order",
                     "and whether there is no run order at all, which makes "
                     "drift, QC-RLSC and the run-order overlay impossible"),
            LooksFor("pack::metabolomics::repeated_subjects",
                     "subject ids that repeat, which is the question about one "
                     "row per person seen from the assay side"),
            LooksFor("pack::metabolomics::zeros_or_missing",
                     "zeros in the intensity block, which four widely-used "
                     "exports disagree about the meaning of"),
            LooksFor("pack::metabolomics::already_transformed",
                     "values that have already been logged or scaled — "
                     "negatives, a compressed maximum, an intensity range too "
                     "narrow to be a raw run, or columns already centred"),
            LooksFor("pack::metabolomics::duplicate_ids",
                     "features or samples present more than once, including "
                     "the renaming a reader does to a repeated column"),
            LooksFor("pack::metabolomics::empty_blocks",
                     "features that are zero everywhere, features that hold "
                     "one value, and samples that are empty across the panel"),
            LooksFor("pack::metabolomics::ion_modes",
                     "positive and negative mode features sharing one table, "
                     "which are two acquisitions rather than one"),
        ),
        recipes=_metabolomics_recipes,
        reframings=(
            Reframing(
                matches=lambda f, df: (f.get("id") == "wide_repeated_measures"
                                       and _is_assay_wide(df)),
                title=lambda f, df: "The wide shape is expected here",
                note=lambda f, df: (
                    "These are different analytes, not one analyte measured "
                    "several times. An untargeted panel names its features by "
                    "mass and retention time, which reads as a numbered series "
                    "to a general-purpose importer. Reshaping to long format "
                    "would rebuild what a row is and is not what this table "
                    "needs.")),),
        priors=(
            # SCOPED TO THE COLUMNS ITS DETECTOR NAMED (`GUIDED-027`). The
            # left-censoring reading is about the low-abundance features, and
            # applying it to `age` because the table is metabolomic would be
            # the dataset-level error the finding was filed about.
            Prior(question="missingness_direction", marker="derived",
                  evidence=Evidence(
                      status=SETTLED,
                      source="research/METABOLOMICS_PACK.md#03 · Missing data",
                  ),
                  scope=COLUMNS, detector="pack::metabolomics::left_censored",
                  values={"mechanism": "below_detection_limit",
                          "strategy": "half_minimum"},
                  reason=("A blank in one of these is usually a non-detection "
                          "rather than a missing observation: they are the "
                          "lowest-abundance features and the missing rate "
                          "tracks abundance rank. Filling with a median would "
                          "place non-detections in the middle of the "
                          "distribution.")),
            # DATASET, not COLUMNS, and the distinction is not pedantry:
            # this prior is about ROWS. Forcing it into a column list it does
            # not have would be the scope error `GUIDED-027` names, committed
            # in the act of repairing it.
            #
            # `offered`, NOT `derived`, and the demotion is the honest half of
            # `GUIDED-030` (`GUIDED-033`). The CLAIM is derived and the finding
            # carries it at `critical`: pooled QC injections are not
            # participants and modeling them is an error with no legitimate
            # reading. The ACTION is not. Excluding them changes N, and clause
            # §04 is unambiguous — an exclusion that changes N is an eligibility
            # criterion the user states, never a silent filter — so a `derived`
            # marker here would license pre-selecting exactly the thing the
            # clause forbids pre-selecting.
            #
            # Claim and action are two different things and the marker governs
            # the second. Saying so is what stops a `derived` prior describing
            # behavior the app does not have, which is the governing rule broken
            # by the layer built to enforce it.
            Prior(question="qc_rows_excluded", marker="offered",
                  evidence=Evidence(
                      status=SETTLED,
                      source="research/METABOLOMICS_PACK.md#01 · Import and structure",
                  ),
                  scope=DATASET,
                  values={"exclude": True, "detector": "pack::metabolomics::pooled_qc",
                          "offers": "eligibility_criterion"},
                  reason=("Pooled quality-control injections are not "
                          "participants, and modeling them is an error with no "
                          "legitimate reading. Excluding them changes N, so it "
                          "is an eligibility criterion you state and it is "
                          "reported in participant flow — the app offers it and "
                          "never applies it. They stay in the table for quality "
                          "assessment.")),
        )),
    GENOMICS: Pack(
        key=GENOMICS, label=LENS_LABELS[GENOMICS],
        # HARDEST FIRST (`LOOP.md` §02), which here is the one that decides what
        # the other is allowed to say: §02 calls data-type detection *"the
        # highest-leverage diagnostic in the pack"* because it determines what
        # is legal downstream, and a p/n reading on a matrix nobody has
        # classified is a true sentence about the wrong object.
        # Merged from two worktrees, hardest-first, and the order is an
        # argument rather than a convenience. `_genomics_data_type` decides
        # what every other reading is ALLOWED to say — §02 is titled *"the
        # highest-leverage diagnostic in the pack"* because it determines what
        # is legal downstream — so a p/n reading on a matrix nobody has
        # classified is a true sentence about the wrong object. The corruption
        # reading is next: it is the one that had to resolve an ambiguity the
        # data does not resolve (a five-digit integer is both an Excel serial
        # and an Entrez gene ID) and the only one here that is a hard stop.
        detectors=(_genomics_data_type, _gene_id_excel_corruption,
                   _counts_at_p_over_n, _gene_id_versions,
                   _gene_id_duplicates, _gene_id_mixed_vocabulary),
        looks_for=(
            # A NOUN PHRASE, and the second person is what a guard here
            # forbids: *"what your numbers actually are"* was the first
            # draft and `test_the_hover_names_what_is_looked_for_and_never_
            # what_was_found` caught it. A hover is read BEFORE the question is
            # answered, so a sentence about this table would be asserting
            # something about every table. The CARD says *what your numbers
            # are*, because by then it has read them.
            LooksFor("pack::genomics::data_type",
                     "which of the nine shapes an expression matrix comes in "
                     "these values are — raw counts, estimated counts, CPM or "
                     "TPM, a composition-scaled CPM, FPKM, a "
                     "variance-stabilized matrix, an array intensity — and "
                     "which downstream steps each of those closes off"),
            LooksFor("pack::genomics::counts_p_over_n",
                     "count columns far outnumbering samples, which orders the "
                     "model shelf toward regularized fits"),
            LooksFor("pack::genomics::gene_id_excel_corruption",
                     "gene symbols Excel has turned into dates or serial "
                     "numbers, which are reported and never repaired because "
                     "nothing says which symbol a date used to be"),
            LooksFor("pack::genomics::gene_id_versions",
                     "version suffixes on accessions, which join to unversioned "
                     "annotation as nothing at all and drop genes without "
                     "erroring"),
            LooksFor("pack::genomics::gene_id_duplicates",
                     "accessions that appear more than once, which is one gene "
                     "counted twice by everything downstream"),
            LooksFor("pack::genomics::gene_id_mixed_vocabulary",
                     "more than one identifier vocabulary in the feature "
                     "names, which no single annotation join can cover"),
            # The considered refusal is a thing the pack will do, so it is named
            # here for the same reason it is a prior with `variant: None`: a
            # decline nobody can see is indistinguishable from never asking.
            LooksFor("prior::normalization",
                     "and it asserts no normalization default, because CPM, TPM "
                     "and VST are not interchangeable"),
        ),
        reframings=(
            Reframing(
                matches=lambda f, df: (
                    f.get("id") == "wide_repeated_measures"
                    and count_matrix(df) is not None),
                title=lambda f, df: "The wide shape is expected here",
                note=lambda f, df: (
                    "These are different genes, not one gene measured several "
                    "times. Reshaping to long format would rebuild what a row "
                    "is.")),
            Reframing(
                matches=lambda f, df: (
                    f.get("id", "").startswith("sentinel_missing__")
                    and _in_count_block(f, df)),
                title=lambda f, df: (
                    f"`{(f.get('params') or {}).get('column')}` holds low "
                    f"counts, not missing-value codes"),
                note=lambda f, df: (
                    "This is a count column, and a small integer in a "
                    "low-expression gene is a count rather than a missing-value "
                    "code. The detector reads an integral column with few "
                    "distinct values as a coded variable, which is the right "
                    "reading for a survey item and the wrong one for a "
                    "transcript count.")),),
        priors=(
            # Genuinely a fact about the DATASET: p ≫ n is a property of the
            # shape, not of any column.
            Prior(question="model_ranking", marker="derived", scope=DATASET,
                  evidence=Evidence(
                      status=SETTLED,
                      source="research/GENOMICS_PACK.md#08 · Modeling at p >> n",
                  ),
                  values={"prefer": "regularized",
                          "discourage": "distance_based"},
                  reason=("At p much greater than n an unregularized fit is "
                          "degenerate, and a distance metric over hundreds of "
                          "features is dominated by noise. The shelf is "
                          "ordered by this and never filtered by it.")),
            Prior(question="normalization", marker="offered", scope=DATASET,
                  evidence=Evidence(
                      status=DISPUTED,
                      source="research/GENOMICS_PACK.md#04 · Normalization — no default asserted",
                      both_sides=(
                          "CPM, TPM and VST are not interchangeable and the choice "
                          "depends on the assay and the question. The research asserts no "
                          "default and neither does this pack; the disagreement is the "
                          "finding, and declining is recorded rather than absent."
                      ),
                  ),
                  values={"variant": None},
                  reason=("CPM, TPM and VST are not interchangeable and the "
                          "choice depends on the assay and the question. No "
                          "default is asserted, which is a position rather "
                          "than an omission — this key exists so that "
                          "declining is recorded rather than absent.")),
        )),
    DIETARY: Pack(
        key=DIETARY, label=LENS_LABELS[DIETARY],
        detectors=(_compositional, _implausible_intake, _energy_adjustment,
                   _nutrition_atwater, _nutrition_survey_weights,
                   _nutrition_partial_design, _nutrition_lonely_psu),
        looks_for=(
            LooksFor("pack::dietary::compositional",
                     "columns that sum to a constant, whose correlations with "
                     "each other are biased by construction"),
            LooksFor("pack::dietary::implausible_intake",
                     "implausible daily intakes, offered as an exclusion and "
                     "never applied on their own"),
            LooksFor("pack::dietary::energy_adjustment",
                     "a total-energy column, without which every nutrient "
                     "association is confounded by total intake"),
            LooksFor("pack::dietary::atwater",
                     "declared energy that does not reconstruct from the "
                     "macronutrients, which is the only way to infer an energy "
                     "unit"),
            LooksFor("pack::dietary::survey_weights",
                     "the survey weights, and which of them a dietary analysis "
                     "takes"),
            LooksFor("pack::dietary::partial_design",
                     "a survey weight with no strata or PSU beside it, which "
                     "leaves the standard errors too narrow"),
            LooksFor("pack::dietary::lonely_psu",
                     "a stratum holding one primary sampling unit, which makes "
                     "its variance contribution undefined rather than small"),
        ),
        priors=(
            # THE ONE IMPLEMENTATION of the averaging rule (`GUIDED-026`).
            # `repeats.menu()` reads this reason rather than restating it, and
            # a test asserts the rendered sentence came from here.
            Prior(question="repeat_treatment", marker="derived", scope=DATASET,
                  evidence=Evidence(
                      status=SETTLED,
                      source="research/NUTRITION_PACK.md#03 · ★ Repeated recalls and measurement error",
                  ),
                  values={"treatment": "mean"},
                  reason=("A single 24-hour recall is a noisy estimate of "
                          "usual intake, and that noise attenuates "
                          "diet–outcome associations toward the null. Using "
                          "their mean rather than a single day reduces the "
                          "within-person measurement error.")),
            Prior(question="energy_adjustment", marker="convention",
                  evidence=Evidence(
                      status=CONVENTION_STATUS,
                      source="research/NUTRITION_PACK.md#04 · Energy adjustment — the methodological signature",
                  ),
                  scope=COLUMNS, detector="pack::dietary::energy_adjustment",
                  values={"variant": "residual",
                          "alternative": "nutrient_density"},
                  reason=("The residual method decorrelates the nutrient from "
                          "energy explicitly, which makes the resulting "
                          "coefficient interpretable. Nutrient density is "
                          "offered beside it, and the choice between them is a "
                          "convention rather than a fact.")),
            Prior(question="collinearity_figure", marker="derived",
                  evidence=Evidence(
                      status=SETTLED,
                      source="research/NUTRITION_PACK.md#05 · Compositional structure and substitution modeling",
                  ),
                  scope=COLUMNS, detector="pack::dietary::compositional",
                  values={"gate": "log_ratio"},
                  reason=("These columns are parts of a whole, so ordinary "
                          "correlation between them is negatively biased by "
                          "construction — raising one necessarily lowers "
                          "another. The correlation figure is drawn on "
                          "log-ratios rather than on the parts.")),
        )),
    CLINICAL: Pack(
        key=CLINICAL, label=LENS_LABELS[CLINICAL],
        # **THE THINNEST PACK BECAME THE WIDEST, AND THE OLD ARGUMENT WAS HALF
        # RIGHT.** This read `detectors=()` until L41, under a comment saying
        # the thinness was the point because physiologic bounds and unit
        # harmonization already live in the core. That is true of §A1.2 — the
        # impossibility bands are `ml/physiology_reference.py`'s and this pack
        # reads them rather than copying them — and it was never true of §A1.3,
        # which specifies censoring tokens, detection limits inferred from the
        # data, and result columns that carry a qualifier inside the value.
        # None of that existed anywhere.
        #
        # Ordered hardest-first, per `LOOP.md` §02, which is also the order they
        # were built in.
        detectors=(_clinical_censored, _clinical_text_numeric,
                   _clinical_mixed_result, _clinical_mixed_units,
                   _clinical_default_mass, _clinical_temporal,
                   _clinical_number_format, _clinical_impossible_vs_extreme),
        looks_for=(
            # NO ANGLE BRACKETS IN A HOVER. The obvious phrasing here was
            # *"results recorded as `<0.3`"*, which is the clearest sentence and
            # the wrong one: the page renders an option note into a `data-tip`
            # attribute, so `<` and `>` come back HTML-escaped and the string
            # the user reads stops being the string this registry holds. Caught
            # by `test_the_page_shows_the_note_on_the_option_it_belongs_to`,
            # which reads the note back off the render rather than off a grep —
            # which is exactly what that test is for.
            LooksFor("pack::clinical::censored_values",
                     "lab results recorded as below the detection limit or "
                     "above the upper limit of quantitation rather than as a "
                     "number, with the limit read back per analyte — and "
                     "`TNTC` and `QNS` separated out, because those are "
                     "measurement failures rather than censoring"),
            LooksFor("pack::clinical::text_numeric",
                     "columns that arrived as text and are more than four "
                     "fifths numbers, which is where a qualifier is hiding "
                     "inside the result"),
            LooksFor("pack::clinical::mixed_result_type",
                     "a result column holding both a measured value and a "
                     "verdict — a troponin with `0.04` in some rows and "
                     "`negative` in others"),
            LooksFor("pack::clinical::mixed_units",
                     "an analyte whose values fall into two populations a "
                     "known conversion factor apart, which is two sites "
                     "reporting different units into one field"),
            LooksFor("pack::clinical::default_value_mass",
                     "vitals piling up on 120/80 and 98.6 — value preference "
                     "and manual entry rather than measurement"),
            LooksFor("pack::clinical::temporal_implausibility",
                     "trajectories that are not believable even where every "
                     "value in them is: an adult gaining height between "
                     "visits, a weight moving a third in three weeks"),
            LooksFor("pack::clinical::number_format",
                     "numbers written so they do not parse as numbers — "
                     "thousands separators and decimal commas"),
            LooksFor("pack::clinical::impossible_vs_extreme",
                     "the difference between a physiologically impossible "
                     "value and an abnormal one, which no generic outlier rule "
                     "can tell apart"),
            # The prior. Sourced `prior::` rather than `pack::` because there
            # is no finding behind it, which the key-match test reads as the
            # honest case rather than as a missing row.
            LooksFor("prior::missingness_direction",
                     "recognized clinical measurements, where a blank often "
                     "means a test was not ordered rather than a value lost — "
                     "the opposite direction from an assay"),
        ),
        # NO REFRAMINGS, and the reason is worth recording. `IMPORT-267` — a
        # column of education levels asserted at `critical` to mix measurement
        # units — was the false alarm this pack was going to reframe. The freeze
        # is lifted for repair of dispositioned `IMPORT-*` rows, so it was fixed
        # at source instead: a unit now needs a measurement in front of it. A
        # reframing is the right tool for a reading that is correct in general
        # and wrong under a lens; this one was wrong everywhere, and reframing
        # it would have left every door but Guided asserting it.
        priors=(
            # The whole clinical pack, and its thinness is the point:
            # physiologic bounds and unit harmonization already exist in the
            # core. This adds ONE prior, and it points the OPPOSITE way from
            # the metabolomics one — which is why `priors()` returns a list.
            #
            # Column-scoped to the clinical variables the engine's own
            # reference matcher recognizes. On an NHANES-shaped table that is
            # the labs and not the questionnaire items beside them.
            Prior(question="missingness_direction", marker="offered",
                  evidence=Evidence(
                      status=CONVENTION_STATUS,
                      source="research/CLINICAL_SURVEY_PACK.md#A2 · ★ Missing data — where TurboTab differentiates itself",
                  ),
                  scope=COLUMNS, detector="pack::clinical::reference_columns",
                  values={"mechanism": "not_ordered"},
                  reason=("Missingness in a clinical measurement often means "
                          "the test was not ordered — a clinician saw no "
                          "reason to run it — which is informative about the "
                          "patient rather than about the measurement. That is "
                          "the opposite direction from an assay. The mechanism "
                          "question already asks; this supplies the prior, not "
                          "the answer.")),
        )),
    SURVEY: Pack(
        key=SURVEY, label=LENS_LABELS[SURVEY],
        detectors=(_ordinal_declared, _survey_sentinel_codes),
        looks_for=(
            LooksFor("pack::survey::ordinal_declared",
                     "a block of items sharing one response scale, whose order "
                     "comes from the instrument rather than from the data"),
            LooksFor("pack::survey::sentinel_codes",
                     "values that break that scale's run — a 9 in a 1 to 5 "
                     "item is a refusal rather than strong agreement, and the "
                     "app reports what treating it as a response would do to "
                     "the item's mean without recoding anything"),
            # The one question a pack is allowed to add (guard #1's deliberate
            # exception), so it is named where the user decides whether to
            # invite it.
            LooksFor("question::state_reverse_coding",
                     "and it asks which of those items are reverse-coded, "
                     "because that needs a codebook and can never be inferred "
                     "from the numbers"),
        ),
        recipes=_survey_recipes,
        reframings=(
            Reframing(
                matches=lambda f, df: (f.get("id") == "wide_repeated_measures"
                                       and likert_block(df) is not None),
                title=lambda f, df: "The wide shape is expected here",
                note=lambda f, df: (
                    "These are the items of one instrument, not one quantity "
                    "measured several times. Items are combined by scoring the "
                    "scale, which is a decision about the instrument, not by "
                    "reshaping the table.")),
            ),
        priors=(
            # NOT migrated to the recipe table, and the reason is worth stating
            # rather than leaving as an omission: the pack's claim is that the
            # ordering is DECLARED, which makes the encoding row-local — a
            # change to the operation's SCOPE, not to its variant. The recipe
            # table carries scope per operation, not per column, so expressing
            # it there would mean shadowing `encode` for the whole table
            # including its genuinely stateful uses. The VARIANT preference did
            # migrate; see `_survey_recipes`.
            Prior(question="ordinal_encoding", marker="derived", scope=COLUMNS,
                  evidence=Evidence(
                      status=SETTLED,
                      source="research/CLINICAL_SURVEY_PACK.md#B2 · Scale construction",
                  ),
                  detector="pack::survey::ordinal_declared",
                  values={"source": "instrument", "row_local": True},
                  reason=("The order comes from the instrument, which makes "
                          "the encoding row-local: the number for a row "
                          "depends on that row's own answer and on nothing "
                          "else. An encoding derived from the observed "
                          "frequencies would have to be fitted inside the "
                          "training folds.")),
            Prior(question="reverse_coding", marker="offered", scope=COLUMNS,
                  evidence=Evidence(
                      status=DISPUTED,
                      source="research/CLINICAL_SURVEY_PACK.md#B4 · ★ Ordinal vs interval — the long-running dispute",
                      both_sides=(
                          "A negative item-rest correlation has four incompatible causes "
                          "- needs reversing, already reversed, a method factor, or the "
                          "item does not belong - and no correlational signature "
                          "separates them. The pack asks; it never infers."
                      ),
                  ),
                  detector="pack::survey::ordinal_declared",
                  values={"variant": None},
                  reason=("Reverse-coding requires a codebook the app does not "
                          "have. Inferring it from item correlations would be "
                          "right whenever the instrument is unidimensional and "
                          "confidently wrong whenever two subscales measure "
                          "opposing constructs — and nothing in the numbers "
                          "separates those cases.")),
        )),
    OTHER: Pack(key=OTHER, label=LENS_LABELS[OTHER]),
}


def findings(df: pd.DataFrame, lens: Sequence[str]) -> List[Dict[str, Any]]:
    """What the selected packs see in this table. Empty is the common answer.

    Ordered by pack, then by the order the detectors are declared in, so the
    same table and the same lens always produce the same list — the Router's
    determinism requirement applies to anything feeding it.
    """
    out: List[Dict[str, Any]] = []
    if df is None or df.empty:
        return out
    for key in normalize_quiet(lens):
        pack = PACKS.get(key)
        if pack is None:
            continue
        for detector in pack.detectors:
            try:
                found = detector(df)
            except Exception:
                # A detector that cannot read this table reports nothing. It
                # must not take the interview down with it, and it must not be
                # silent about having failed either.
                from turbotab import devchecks
                devchecks.swallowed(
                    f"packs.{key}::{getattr(detector, '__name__', '?')}",
                    _last_exception(),
                    "this pack detector found nothing, and would have been "
                    "indistinguishable from one that legitimately found nothing")
                continue
            if found:
                out.append(found)
    return out


def _last_exception() -> BaseException:
    import sys
    return sys.exc_info()[1] or RuntimeError("unknown")


def normalize_quiet(keys: Optional[Sequence[str]]) -> List[str]:
    """`normalize`, for callers that already hold a recorded answer."""
    return [k for k in (keys or []) if k in LENS_KEYS]


def load(lens: Sequence[str]) -> List[str]:
    """Register the selected packs' recipe contributions. Idempotent.

    Called where recipes are resolved rather than at import, because importing
    this module must not mutate the recipe table — every test that touches
    recipes would inherit five packs it never asked for.
    """
    loaded = []
    for key in normalize_quiet(lens):
        pack = PACKS.get(key)
        if pack is None or pack.recipes is None or key in _LOADED:
            continue
        pack.recipes()
        _LOADED.add(key)
        loaded.append(key)
    return loaded


def unload_for_test() -> None:
    """Forget which packs were loaded. Pairs with `recipes.restore`."""
    _LOADED.clear()


def loaded_for_test() -> frozenset:
    """Which packs this process has registered into the recipe table.

    Exists so a test can put the bookkeeping back exactly as it found it.
    Restoring the TABLE without restoring this leaves the two disagreeing:
    `load` would re-register rows the restore already put back, or skip rows
    the restore removed. Both are silent.
    """
    return frozenset(_LOADED)


def restore_loaded_for_test(keys) -> None:
    """Put the load bookkeeping back. Pairs with `loaded_for_test`."""
    _LOADED.clear()
    _LOADED.update(keys)


def prior_columns(pack_key: str, detector: str,
                  df: pd.DataFrame) -> Optional[List[str]]:
    """The columns a column-scoped prior applies to, from its own detector.

    `None` — not `[]` — when the detector did not fire. The difference is the
    whole of `GUIDED-027`: an empty list would say *"this prior applies to no
    columns"*, and `None` says *"the evidence for this prior is absent"*, which
    is why the prior is withheld rather than rendered over nothing.
    """
    if detector == "pack::clinical::reference_columns":
        found = clinical_reference_columns(df)
        return found or None
    for f in findings(df, [pack_key]):
        if f["id"] == detector:
            columns = (f.get("params") or {}).get("columns")
            if columns:
                return [str(c) for c in columns]
            return [str(c) for c in f.get("affected_columns") or []] or None
    return None


def priors(lens: Sequence[str], name: str,
           df: Optional[pd.DataFrame] = None) -> List[Dict[str, Any]]:
    """Every selected pack's prior on one question, with the pack named.

    **A list, and it stays a list.** The lens is multi-select and two packs can
    hold opposite priors on the same question — metabolomics and clinical
    genuinely disagree about what a blank means, and on a table that is both
    they are both right about different columns. Resolving that silently would
    pick one field's reading of a dataset that is two. The consumer surfaces the
    disagreement; it never settles it.

    **Column-scoped priors are withheld when their detector did not fire**
    (`GUIDED-027`). `missingness_direction` is a fact about each column, not
    about the table, and a dataset-level *"below the detection limit"* on an
    NHANES-shaped file would be wrong for most of its columns. Passing `df`
    resolves the scope; omitting it returns the declarations unscoped, which is
    what a caller reading the catalogue rather than a dataset wants.
    """
    out: List[Dict[str, Any]] = []
    for key in normalize_quiet(lens):
        for prior in PACKS[key].priors:
            if prior.question != name:
                continue
            entry: Dict[str, Any] = {
                "pack": key, "label": LENS_LABELS[key],
                "question": prior.question, "marker": prior.marker,
                "reason": prior.reason, "scope": prior.scope,
                # THE BADGE TRAVELS WITH THE PRIOR. A status computed on the
                # server and dropped at the boundary would be `DRIVE-001`'s
                # class: built, correct, and unreachable by a reader — and the
                # whole argument for the badge is that it reaches a reader.
                **prior.evidence.to_dict(),
                **prior.values,
            }
            if prior.scope == COLUMNS:
                entry["detector"] = prior.detector
                if df is not None:
                    columns = prior_columns(key, prior.detector, df)
                    if columns is None:
                        # The evidence is absent, so the prior has nothing to
                        # be about. Withheld rather than rendered over every
                        # column, which is the defect being repaired.
                        continue
                    entry["columns"] = columns
            out.append(entry)
    return out


def prior_for_column(lens: Sequence[str], name: str, column: str,
                     df: pd.DataFrame) -> List[Dict[str, Any]]:
    """The priors on one question that apply to ONE column.

    Dataset-scoped priors apply to every column and are included. Column-scoped
    ones are included only where their detector named this column — which is
    what makes a mixed table get the dietary reading on its recall columns and
    the lab reading on its labs.
    """
    return [p for p in priors(lens, name, df)
            if p["scope"] == DATASET or column in (p.get("columns") or [])]


def recipe_origins(lens: Sequence[str]) -> List[Dict[str, Any]]:
    """Which recipe rows the selected packs contributed, from the table itself.

    Read back out of `recipes` rather than mirrored here, because the recipe
    table is canonical (`GUIDED-025`) and a second copy of what a pack
    registered is the drift this whole finding is about.
    """
    from turbotab import recipes as _rec
    load(lens)
    wanted = {f"{k}_pack" for k in normalize_quiet(lens)}
    out = []
    for d in _rec.defaults():
        if d.origin in wanted:
            out.append({"operation": d.operation, "variant": d.variant,
                        "selector": d.selector, "origin": d.origin,
                        "reason": d.reason})
    return out


#: The one place the hedge block's own prose lives, so the page composes none of
#: it. `COPY_DECK.md`'s rule: a sentence a user reads is a sentence some server
#: composed.
HEDGE_TITLE = "Where this app declines to be confident"
HEDGE_WHY = (
    "These are the places the field does not agree with itself, or agrees on "
    "something the tools around it get wrong. None of them is a reading of your "
    "table — they are true of this kind of measurement before any data arrives, "
    "and they are here so the app's confidence is legible rather than uniform.")


def hedges(lens: Sequence[str]) -> Optional[Dict[str, Any]]:
    """Every selected pack's §11 positions, ranked, with the bound stated.

    `None` — not an empty block — when no selected pack has any. *Nothing to
    say* and *a section that says nothing* are different sentences, and a pack
    whose §11 has not been built must produce the first.

    **`GUIDED-209`: the list states its bound and is not cut.** `n` is what
    exists, `showing` is what is in `items`, and they are equal because nothing
    here is truncated — a hedge register that dropped its tail would be the
    surface most likely to drop the awkward one. The two numbers are served
    rather than implied so a reader and a test can both check the equality
    instead of trusting it.

    The refusals ride here rather than on their own route because they are §11
    item 12's whole content: three numbers this app will not supply. A refusal
    computed and reachable only from its own test is `GUIDED-060` again.
    """
    chosen = normalize_quiet(lens)
    items: List[Hedge] = []
    for key in chosen:
        pack = PACKS.get(key)
        if pack is not None:
            items.extend(pack.hedges)
    if not items:
        return None
    items.sort(key=lambda h: (h.rank, h.key))

    refusals = []
    for key, _label in SOFTWARE_DEFAULTS_REFUSED:
        try:
            software_default(key)
        except SoftwareDefaultRefusal as refusal:
            # THE REFUSAL IS RAISED ON THE SERVED PATH rather than described
            # here. `software_default` has no branch that returns a number, so
            # this loop is the only way the payload can be built and there is
            # nowhere for a later loop to put a constant.
            refusals.append({"key": key, **refusal.to_dict()})
    return {
        "title": HEDGE_TITLE,
        "why": HEDGE_WHY,
        "source": _M11,
        "items": [h.to_dict() for h in items],
        "n": len(items),
        "showing": len(items),
        "complete": True,
        "n_refused": len(refusals),
        "refuses": refusals,
        # The distribution, so a reader can see at a glance that the badges are
        # not uniform. A block where every item said DISPUTED would be telling
        # them nothing, and this is the number that shows it does not.
        "by_status": {status: sum(1 for h in items
                                  if h.evidence.status == status)
                      for status in EVIDENCE_STATUSES},
    }


def _and_list(items: Sequence[str]) -> str:
    items = [i for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    # A phrase that already opens with "and" is a continuation the author wrote
    # deliberately (genomics's refusal, survey's added question), so it joins
    # with a semicolon and is not given a second conjunction.
    head, tail = items[:-1], items[-1]
    joined = "; ".join(head)
    return f"{joined}; {tail}" if tail.startswith("and ") else f"{joined}; and {tail}"


def option_note(key: str) -> str:
    """What picking this lens sets the app looking for — one sentence.

    `GUIDED-039`. Composed from the pack's own `looks_for` entries rather than
    written here, so the hover and the detectors cannot drift: a detector added
    without a phrase fails `test_a_pack_names_what_it_will_look_for`, and a
    phrase for a detector that does not exist fails the same test from the other
    side.

    Every phrase is a noun phrase. The sentence says what will be *looked for*,
    never what has been *found* — the second would be a claim about a table
    nobody has read under this lens yet, on a control the user has not pressed.
    """
    if key == OTHER:
        return ("Records that the listed kinds do not describe this table. "
                "Nothing extra is looked for and nothing is limited — the app "
                "is fully functional with no lens.")
    pack = PACKS.get(key)
    if pack is None or not pack.looks_for:                 # pragma: no cover
        raise PackError(
            f"{key!r} has nothing to say about what it will look for. A lens "
            f"option whose hover cannot be written is a bet the user is being "
            f"asked to make blind.")
    return ("Sets the app looking for " + _and_list([lf.phrase for lf in pack.looks_for])
            + ". It changes what is looked for and what is suggested; it never "
              "removes an option.")


def question(suggestion: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The lens question, as the Router and the page both read it."""
    return {
        "key": "state_lens",
        "clause": "lockbox-01",
        "title": LENS_TITLE,
        "why": LENS_WHY,
        "consumer": LENS_CONSUMER,
        "multi_select": True,
        # Mandatory, and the reason is the one `normalize` refuses an empty
        # selection with rather than a second sentence saying the same thing.
        "min_selections": 1,
        "min_reason": LENS_EMPTY_REFUSAL,
        "options": [{"key": k, "label": LENS_LABELS[k],
                     "note": option_note(k)} for k in LENS_KEYS],
        "suggestion": suggestion or {},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Detection — a suggestion and a contradiction detector, never the answer
# ─────────────────────────────────────────────────────────────────────────────

def suggest(df: pd.DataFrame) -> Dict[str, Any]:
    """What the shape of this table hints at, offered beside the options.

    Never pre-selected. §01: *the user's answer is the answer; detection never
    overrides it.* This exists so a user who does not know the vocabulary has
    somewhere to start, and so the contradiction detector below has a reading to
    disagree with.
    """
    hints: List[Dict[str, str]] = []
    if df is None or df.empty:
        return {"hints": hints}
    if count_matrix(df) is not None:
        hints.append({"lens": GENOMICS,
                      "because": "every one of these columns holds "
                                 "non-negative whole numbers, which is what a "
                                 "count matrix looks like"})
    if _is_assay_wide(df) and count_matrix(df) is None:
        hints.append({"lens": METABOLOMICS,
                      "because": f"there are {len(_numeric(df)):,} measurement "
                                 f"columns across {len(df):,} rows"})
    if likert_block(df) is not None:
        block = likert_block(df)
        hints.append({"lens": SURVEY,
                      "because": f"{len(block['columns']):,} columns share one "
                                 f"{len(block['scale'])}-point response scale"})
    if _reference_column(df, "kcal") is not None:
        hints.append({"lens": DIETARY,
                      "because": "there is a total-energy column"})
    # CLINICAL WAS MISSING, and the asymmetry was the same defect as the
    # contradiction detector's (`GUIDED-028`): four lenses could be hinted and
    # the fifth could not, so the shape a clinical table has was the one shape
    # the app never named. The evidence goes through the engine's own exact
    # reference matcher — the same one `clinical_reference_columns` uses — so
    # this is not a sixth name list.
    recognized = _clinical_columns(df)
    # TWO, deliberately low. A hint is a suggestion and never an answer — it
    # costs nothing to ignore and it is not allowed to change a default — so
    # the bar is where a reader would start guessing, not where a detector
    # would be confident. `clinic_visits.csv` has exactly two recognized
    # measurements among fourteen columns and is unmistakably a clinic export.
    if len(recognized) >= 2 and not _is_assay_wide(df):
        hints.append({"lens": CLINICAL,
                      "because": f"{len(recognized)} columns are recognized "
                                 f"clinical measurements — "
                                 + ", ".join(f"`{c}`" for c in recognized[:3])})
    return {"hints": hints}


def _clinical_columns(df: pd.DataFrame) -> List[str]:
    """Numeric columns the reference vocabulary recognizes, blanks or not.

    `clinical_reference_columns` is the missingness-scoped sibling and requires
    blanks, because a prior about what a blank means has nothing to say about a
    column with none. A hint is about the table's KIND and wants both.
    """
    try:
        from ml.physiology_reference import load_reference_bundle, match_variable_key
        reference = load_reference_bundle()["nhanes"]
    except Exception:                                      # pragma: no cover
        return []
    return [str(c) for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c])
            and match_variable_key(str(c), reference)]


# The two terminal exits a CONTRADICTION carries, so an interface cannot render
# the interruption without also rendering its way out (`DESIGN_LANGUAGE.md`
# §09). The same shape `grain.py` uses, and deliberately the same words.
# Below this there are not enough blank columns for a rank correlation to mean
# anything, and the honest answer is silence rather than a reading taken from
# three points. `GUIDED-032`'s whole complaint was a detector confident on thin
# evidence.
_MIN_BLANKS_TO_READ = 10


def _lens_revise():
    # `GUIDED-184`, same shape as `grain._RESOLVE`: a resolve exit that is not
    # a request, so it carried no `retry` and the page greyed it out.
    from turbotab import exits as _exits
    return _exits.revise(
        "Change my answer",
        "Go back to the question and describe the table differently.")


_LENS_RESOLVE = _lens_revise()


def _lens_attest(what: str) -> Dict[str, Any]:
    # Built through `exits.attest`, which carries the payload key and a
    # ready-to-merge retry — `GUIDED-072`: this exit rendered as a way through
    # and told a client nothing about how to take it.
    from turbotab import exits
    return exits.attest(
        "My answer is right — the data really is like this", what,
        exits.ACKNOWLEDGE_CONTRADICTION)


def contradiction(df: pd.DataFrame, lens: Sequence[str]) -> Optional[Dict[str, Any]]:
    """Evidence that the stated lens and the table disagree. **Both directions.**

    §01, and the same escalation rule as everywhere else: *escalate on evidence
    that a reading is wrong, never on the size of the consequence.*

    **The reverse direction was missing and its cost has changed** (`GUIDED-028`).
    Before the priors layer was wired a wrong lens was cosmetic. It is not any
    more: a stated metabolomics lens over a 40-column clinical table now settles
    every one of its blank columns as *below the detection limit* at `derived`
    confidence, which withholds the mechanism question for all of them and
    schedules half-minimum imputation on data that wants a different reading
    entirely. That is the direction where a wrong lens does real work.

    **This fires before any prior is granted**, which is enforced rather than
    sequenced: `project.set_lens` refuses an unacknowledged contradiction, so
    there is no state in which the lens is recorded, the skips are taken, and
    the disagreement is raised afterwards. A user watching three hundred
    questions vanish and then being told the lens looks wrong has already lost
    the thread.

    Advisory, not a refusal — both exits are terminal and one of them is *"my
    answer is right"*. The user may be right and the shape unusual.
    """
    chosen = normalize_quiet(lens)
    if not chosen or chosen == [OTHER] or df is None or df.empty:
        return None

    # ── SILENT WHERE THE TABLE MAY BE THE OTHER WAY ROUND ───────────────────
    #
    # `GUIDED-042`, found while building question 1.5 and worth stating in full
    # because it is this detector's own failure mode turned on itself.
    #
    # Every reading below is computed PER COLUMN. On a table exported
    # features-in-rows the columns are samples, so *"the missing rate does not
    # track abundance"* is measured across the wrong axis and says nothing about
    # the lens at all. Driven, a transposed copy of `metabolomics_untargeted.csv`
    # produced exactly that: a 409 asserting, authoritatively and in the app's
    # most interruptive voice, that the user's blanks *"do not look like
    # non-detections"* — when read the right way round they do.
    #
    # Two readings compete: *the lens is wrong* and *the table is turned around*.
    # The second explains the first, and question 1.5 is where it is settled. So
    # this one stays quiet and lets the sequence do its work. That is the
    # escalation rule applied honestly — escalate on evidence that a reading is
    # wrong, and here the evidence is against THIS detector's reading.
    #
    # It is not a hole: 1.5 fires on exactly this condition, so the user is not
    # left with silence. They are asked the question that can actually be
    # answered, and the contradiction check runs again on whatever frame they
    # confirm.
    from turbotab import orientation as _orient
    if _orient.read(df).get("reading") == _orient.FEATURE_MAJOR:
        return None

    numeric = _numeric(df)

    # ── direction 1: the shape is an assay and the answer is not ────────────
    if len(numeric) >= 100 and not any(k in chosen for k in _ASSAY_PACKS):
        stated = ", ".join(LENS_LABELS[k].lower() for k in chosen)
        suggests = GENOMICS if count_matrix(df) is not None else METABOLOMICS
        return {
            "kind": "stated_lens_but_shape_is_an_assay",
            "message": (
                f"This table has {len(numeric):,} numeric columns across "
                f"{len(df):,} rows, which is the shape of an assay panel, and "
                f"you described it as {stated}. One of those two readings is "
                f"probably wrong, and which one changes what is looked for."),
            "n_numeric": len(numeric), "n_rows": len(df),
            "suggests": [suggests],
            "exits": [_LENS_RESOLVE, _lens_attest(
                "Continue with the lens as stated. The disagreement is "
                "recorded and travels into the methods section as a stated "
                "limitation rather than disappearing.")],
        }

    # ── direction 2: the assay reading of the BLANKS fails its own test ─────
    #
    # `GUIDED-032`. This used to key on column count — *"10 numeric columns
    # across 600 rows, too few to be a panel"* — and that sentence is **false**:
    # targeted metabolomics and proteomics panels routinely measure ten to fifty
    # analytes. A falsehood inside the mechanism built to catch false readings.
    #
    # It was also guarding a cost that could not occur. The metabolomics
    # missingness prior is scoped to the columns `_left_censored` names
    # (`GUIDED-027`), so on a table with no left-censoring the prior is withheld
    # and a wrong assay lens grants ZERO skips. The old check asserted something
    # untrue in order to prevent something that already could not happen, which
    # is strictly worse than saying nothing.
    #
    # What replaces it is the lens's own prediction, measured. An assay lens
    # says a blank is usually a non-detection, which means missing rates should
    # track abundance. Where there are enough blanks to read and they do NOT,
    # the lens has predicted something and the data has disagreed — evidence a
    # reading is wrong, in the only form that earns an interruption.
    assay = [k for k in chosen if k in _ASSAY_PACKS]
    if assay:
        blanks = [c for c in numeric if bool(df[c].isna().any())]
        if len(blanks) >= _MIN_BLANKS_TO_READ:
            rate = df[numeric].isna().mean()
            abundance = df[numeric].mean(numeric_only=True)
            usable = [c for c in numeric
                      if pd.notna(abundance.get(c)) and abundance.get(c, 0) > 0]
            rho = None
            if len(usable) >= _MIN_BLANKS_TO_READ:
                rho = pd.Series(rate[usable]).corr(
                    pd.Series(np.log(abundance[usable])), method="spearman")
            if rho is not None and pd.notna(rho) and rho > -0.3:
                stated = ", ".join(LENS_LABELS[k].lower() for k in assay)
                return {
                    "kind": "stated_assay_lens_but_blanks_are_not_censored",
                    "message": (
                        f"You described this as {stated}, and its blanks do not "
                        f"look like non-detections: across {len(usable):,} "
                        f"measurement columns the missing rate does not track "
                        f"abundance (rank correlation {rho:+.2f}, where an "
                        f"assay would be strongly negative). An assay lens "
                        f"reads a blank as below the detection limit; here that "
                        f"reading has nothing to rest on."),
                    "n_numeric": len(numeric), "n_rows": len(df),
                    "rho": round(float(rho), 3),
                    "suggests": [CLINICAL] if clinical_reference_columns(df)
                                else [OTHER],
                    "exits": [_LENS_RESOLVE, _lens_attest(
                        "Continue with the assay lens. It will still recognize "
                        "the shape; what it will not do is read these blanks as "
                        "non-detections, because nothing here says they are.")],
                }

    # ── direction 3: genomics stated, and the values are not counts ─────────
    #
    # Narrower than direction 2 and separate from it, because the two are wrong
    # about different things. A wide table of concentrations described as
    # genomics is a panel — direction 2 stays quiet — and the p >> n prior it
    # sets is right while the count reading underneath it is not.
    # `count_matrix` returns None for TWO different reasons — the values are
    # not integers, or there are not enough columns to be a matrix — and only
    # the first justifies this sentence. Keying on the function's None was
    # `GUIDED-032`'s defect about to ship a second time in the same mechanism:
    # on a 40-item Likert instrument it produced *"its measurement columns are
    # not counts — `` hold fractional values"*, which is false twice over and
    # names nothing.
    #
    # So the claim is made only where the columns that refute it exist, and it
    # names them.
    #
    # ── AND IT WAS STILL WRONG, BECAUSE IT EQUATED GENOMICS WITH COUNTS ─────
    #
    # *"495 of its measurement columns are not counts … Counts and
    # concentrations are different objects"* — raised at 409, in the app's most
    # interruptive voice, against `genomics_cpm.csv`, which is a CPM matrix, and
    # against the TMM, FPKM, VST, microarray, estimated-count and log-ratio
    # siblings beside it. **Six of the nine shapes §02 describes for an
    # expression matrix are non-integer**, so a fractional value is not evidence
    # against the genomics lens; it is evidence about which of the nine this is.
    # The premise was false and it blocked the lens on seven of the eight
    # genomics fixtures in this tree — the contradiction detector asserting
    # something false about its own field, in the mechanism built to catch a
    # false reading.
    #
    # `count_matrix(df) is None` therefore is no longer sufficient. The claim
    # now needs the data-type reader to have found NO signature at all: not
    # counts, and not any of the other eight either. That is a table this lens
    # genuinely does not describe, and `metabolomics_untargeted.csv` — 395
    # continuous columns with no zero anywhere — is still exactly it.
    non_integral = [c for c in numeric if not _is_integral(df[c])]
    reading = data_type_card(df) if GENOMICS in chosen else None
    recognized = bool(reading and reading.get("read"))
    if (GENOMICS in chosen and _is_assay_wide(df) and count_matrix(df) is None
            and len(non_integral) >= _MIN_BLANKS_TO_READ
            and not recognized):
        return {
            "kind": "stated_genomics_but_values_are_not_counts",
            "message": (
                f"You described this as {LENS_LABELS[GENOMICS].lower()}, and "
                f"these values are not counts and are not any of the other "
                f"eight shapes an expression matrix comes in either — not CPM "
                f"or TPM, not a composition-scaled CPM, not FPKM, not a "
                f"variance-stabilized matrix, not an array intensity, not a "
                f"log-ratio. {len(non_integral):,} of its measurement columns "
                f"hold fractional values — `"
                + "`, `".join(non_integral[:5])
                + "` among them — and nothing in the totals, the zeros or the "
                  "range matches. A continuous panel with no zero anywhere in "
                  "it reads as concentrations."),
            "n_numeric": len(numeric), "n_rows": len(df),
            "suggests": [METABOLOMICS],
            "exits": [_LENS_RESOLVE, _lens_attest(
                "Continue with the genomics lens. The p-much-greater-than-n "
                "prior applies; the disagreement about what the values are is "
                "recorded, and the data-type card will say it read nothing "
                "rather than name a shape.")],
        }
    return None


def _is_integral(s: pd.Series) -> bool:
    values = s.dropna()
    if values.empty:
        return False
    try:
        return bool(np.all(np.equal(np.mod(values.to_numpy(dtype=float), 1), 0)))
    except (TypeError, ValueError):                        # pragma: no cover
        return False
