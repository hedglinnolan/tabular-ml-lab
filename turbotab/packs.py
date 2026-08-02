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
    order_col = None
    for c in cols:
        s = df[c].dropna()
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
            order_col = c
            break
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


def likert_block(df: pd.DataFrame, minimum: int = 8) -> Optional[Dict[str, Any]]:
    """The largest set of columns sharing one declared response scale.

    Shared exactly, not approximately. Two columns on 1–5 and one on 1–7 are two
    instruments or one instrument and a stray, and averaging across them is the
    error the detector exists to avoid proposing.

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
            if values and values <= scale and len(values) >= len(scale) - 1:
                by_scale.setdefault(tuple(sorted(scale)), []).append(str(c))
                break
    if not by_scale:
        return None
    scale, columns = max(by_scale.items(), key=lambda kv: len(kv[1]))
    if len(columns) < minimum:
        return None

    balanced = 0
    for c in columns:
        s = df[c].dropna()
        if s.empty:
            continue
        used = set(int(v) for v in s.unique())
        if used != set(scale):
            continue                                    # a category never used
        if float(s.value_counts(normalize=True).max()) <= _MAX_MODAL_SHARE:
            balanced += 1
    if balanced / len(columns) < _MIN_SHARE_BALANCED:
        return None
    return {"scale": list(scale), "columns": columns}


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
        detectors=(_left_censored, _acquisition_order, _pooled_qc),
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
        detectors=(_counts_at_p_over_n,),
        looks_for=(
            LooksFor("pack::genomics::counts_p_over_n",
                     "count columns far outnumbering samples, which orders the "
                     "model shelf toward regularized fits"),
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
        detectors=(_ordinal_declared,),
        looks_for=(
            LooksFor("pack::survey::ordinal_declared",
                     "a block of items sharing one response scale, whose order "
                     "comes from the instrument rather than from the data"),
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


_LENS_RESOLVE = {
    "id": "revise", "kind": "resolve", "label": "Change my answer",
    "detail": "Go back to the question and describe the table differently.",
}


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
    non_integral = [c for c in numeric if not _is_integral(df[c])]
    if (GENOMICS in chosen and _is_assay_wide(df) and count_matrix(df) is None
            and len(non_integral) >= _MIN_BLANKS_TO_READ):
        return {
            "kind": "stated_genomics_but_values_are_not_counts",
            "message": (
                f"You described this as {LENS_LABELS[GENOMICS].lower()}, and "
                f"{len(non_integral):,} of its measurement columns are not "
                f"counts — `"
                + "`, `".join(non_integral[:5])
                + "` hold fractional values. Counts and concentrations are "
                  "different objects, and the difference decides whether a log "
                  "transform is derived or merely one option among several."),
            "n_numeric": len(numeric), "n_rows": len(df),
            "suggests": [METABOLOMICS],
            "exits": [_LENS_RESOLVE, _lens_attest(
                "Continue with the genomics lens. The p-much-greater-than-n "
                "prior applies; the disagreement about counts is recorded.")],
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
