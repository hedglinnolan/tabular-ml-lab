"""turbotab.eligibility — clause §04's question, and what it may not show.

> *Is your study restricted to part of this data?*

**Two operations that look identical in a spreadsheet and are not.**

| | Eligibility criterion | Robustness trim |
|---|---|---|
| What it says | who the model is *for* | how the fit is *stabilized* |
| Applied to | the whole dataset, **pre-seal** | the training partition only, **post-seal** |
| Changes N | yes — reported in participant flow with its reason | no |
| Test set | obeys it | **never touched** |

TRIPOD+AI names continuous-variable restrictions ("e.g. age range") as an
eligibility item reported in participant flow. So an exclusion here is not a
filter: it is a **criterion**, it changes N, and the number it changes N *by*
belongs in the flow diagram beside the reason.

**The withheld distribution is the whole rigor point, and it is a refusal.**
The question is asked in *scientific* terms — does your research question
restrict the population? — with the target's distribution **not shown**. An
eligibility criterion comes from the research question, not from the histogram.
A user who needs to see the shape to decide where to cut is doing data-driven
cohort selection, which is its own publishable bias, and showing them the shape
is the app causing it.

What the app may show is bounded by which question the evidence answers:

    permitted   "is this data corrupted?"   observed min/max, impossible-value
                                            flags, how many rows are missing it
    withheld    "where should I cut?"       the distribution, quantiles,
                                            histograms, the outcome's spread,
                                            any per-value or per-bin counts

:func:`permitted_evidence` returns only the first kind, and
:func:`test_the_question_withholds_the_distribution_and_says_why` asserts the
second never appears. The refusal is stated to the user rather than performed
silently — `DESIGN_LANGUAGE.md` §10 layer 2, where a disclosure explains *why
this question* and *what the answer changes*, and this one has to explain a
**subtraction**, which is harder and more worth doing.

Row-local by construction: an eligibility criterion is a predicate over one
row's own cells, so clause §06 classifies it a structural repair and it executes
immediately. That is why it can be pre-seal at all.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

# The two answers. `EVERYONE` is a recorded answer, not an absence: "the study
# is about everyone here" and "nobody was asked" must never look the same, which
# is the same rule the selection step follows for "every column".
EVERYONE = "everyone"
RESTRICTED = "restricted"
ANSWERS = (EVERYONE, RESTRICTED)


class EligibilityRefusal(Exception):
    """The eligibility step was asked for something clause §04 forbids."""


# ── the question ─────────────────────────────────────────────────────────────

QUESTION = "Is your study restricted to part of this data?"

WHY = (
    "If your research question only applies to certain participants — an age "
    "range, a range of outcome values, one site — say so now. It becomes an "
    "exclusion criterion in your methods, and those rows are removed before "
    "any held-out set is drawn.")

OPTIONS = ("No, the study is about everyone here",
           "Yes → which column, and what range?")

#: The label at index `i` records the answer at index `i`. Stated here because
#: this module owns both halves — `ANSWERS` is what `set_eligibility` validates
#: against and `OPTIONS` is what a person reads — and a pairing composed
#: anywhere else is a second copy of it.
#:
#: `DRIVE-023`. The Router served this question with `options` and no
#: `option_values`, and `Question.to_dict` falls back to `option_values or
#: options`. That fallback is correct for a question whose labels *are* its
#: values and silently wrong here, so the wire said the value of the first
#: option was the sentence "No, the study is about everyone here" — which
#: `set_eligibility` refuses with a 400. It is `GUIDED-037` on a second
#: question, and it stayed invisible only because `state_eligibility` sat in the
#: page's `HANDLED_QUESTION_KEYS` claiming a card that was never built, so
#: nothing ever pressed it.
assert len(OPTIONS) == len(ANSWERS), "an option with no answer cannot be pressed"

# The consumer disclosure the Router's `audit()` demands of every pushed FACT.
CONSUMER = (
    "The exclusion runs before the lockbox draws anything, so the held-out set "
    "is drawn from the population you actually studied rather than from a "
    "wider one. The count it removes is recorded as a participant-flow line "
    "with its reason, which is what a CONSORT or TRIPOD flow diagram needs. "
    "Answering after the seal is drawn is not possible — it would mean the "
    "held-out rows were chosen from people the study is not about.")

# §04's rigor point, said out loud. The app is REFUSING to show something, and
# a refusal the user cannot see reads as an omission.
WITHHELD_DISCLOSURE = (
    "We are not showing you the outcome's distribution here. An exclusion that "
    "comes from your research question is reportable; one that comes from "
    "looking at the data is a different thing, and it belongs later.")

#: The recorded-answer sentence, as a template, because two things say it: this
#: module composes it for the record and `copydeck.py` prints it in the deck.
#:
#: `DRIVE-031`. The deck RESTATED it as a literal, so when the false clause was
#: removed here the deck kept asserting it and `copydeck check` stayed green —
#: the check probes that a fragment still exists in the source file, and the
#: fragment it probes is not this sentence. Two copies of one string is the
#: failure this module's own `option_values` comment is about, one field over.
EVERYONE_SENTENCE = ("No eligibility restriction: all {n} rows are in the study "
                     "population.")

# What the permitted evidence is FOR, so an interface renders it as the answer
# to the corruption question rather than as a cut-point aid.
EVIDENCE_CAPTION = (
    "Observed range and impossible values only — enough to tell you whether "
    "this column is corrupted, not enough to pick a cut-point from.")


def question() -> Dict[str, Any]:
    """The question as data, so the deck and the page read the same source."""
    return {"key": "state_eligibility", "title": QUESTION, "why": WHY,
            "consumer": CONSUMER, "options": list(OPTIONS),
            "option_values": list(ANSWERS),
            "withheld": WITHHELD_DISCLOSURE, "clause": "lockbox-04"}


# ── the evidence that is permitted, and nothing else ─────────────────────────

def permitted_evidence(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """Observed min/max and impossible-value flags. Never a distribution.

    Deliberately thin, and the thinness is the design. Every field here answers
    *"is this data corrupted?"*; nothing here helps answer *"where should I
    cut?"*. Adding a median would be one keystroke and would defeat the clause,
    which is why the test asserts on the KEYS rather than on the prose.
    """
    if column not in df.columns:
        raise EligibilityRefusal(f"No column named '{column}' in this table.")
    s = df[column]
    out: Dict[str, Any] = {
        "column": str(column),
        "n_rows": int(len(s)),
        "n_missing": int(s.isna().sum()),
        "caption": EVIDENCE_CAPTION,
    }
    if pd.api.types.is_numeric_dtype(s):
        clean = s.dropna()
        out["observed_min"] = None if clean.empty else float(clean.min())
        out["observed_max"] = None if clean.empty else float(clean.max())
        # An impossible value is a corruption signal, not a shape signal: a
        # negative age is wrong whatever the distribution looks like.
        out["n_negative"] = int((clean < 0).sum())
        out["n_sentinel"] = int(clean.isin([-999, -99, -9999, 999]).sum())
    else:
        out["kind"] = "not numeric"
        # The distinct VALUES, for a site or arm column, because "which sites
        # exist" is what the question asks the user to choose among. Counts per
        # value are withheld: those are a distribution.
        out["values"] = sorted(str(v) for v in s.dropna().unique()[:50])
    return out


# ── the criterion ────────────────────────────────────────────────────────────

def build_criterion(df: pd.DataFrame, column: str,
                    minimum: Optional[float] = None,
                    maximum: Optional[float] = None,
                    keep_values: Optional[Sequence[Any]] = None,
                    reason: str = "") -> Dict[str, Any]:
    """One exclusion criterion, with the participant-flow numbers it produces.

    Returns the criterion AND the labels that survive it. Nothing is applied
    here — `AnalysisProject.set_eligibility` does that, and only pre-seal.

    `reason` is required for a restriction and is not decoration: participant
    flow reports *how many* and *why*, and a criterion with no stated reason
    cannot become a methods sentence.
    """
    if column not in df.columns:
        raise EligibilityRefusal(f"No column named '{column}' in this table.")
    if minimum is None and maximum is None and not keep_values:
        raise EligibilityRefusal(
            "A restriction needs a range or a set of values to keep. Without "
            "one, the honest answer is that the study is about everyone here, "
            "which is its own recorded answer.")
    if not (reason or "").strip():
        raise EligibilityRefusal(
            "An exclusion criterion needs its reason. Participant flow reports "
            "how many rows were excluded AND why; a criterion with no reason "
            "cannot become a methods sentence, and one that cannot be written "
            "down should not be applied.")

    s = df[column]
    mask = pd.Series(True, index=df.index)
    parts: List[str] = []
    if keep_values:
        wanted = {str(v) for v in keep_values}
        mask &= s.astype(str).isin(wanted)
        parts.append(f"`{column}` in {sorted(wanted)}")
    if minimum is not None:
        mask &= s >= minimum
        parts.append(f"`{column}` ≥ {minimum:g}")
    if maximum is not None:
        mask &= s <= maximum
        parts.append(f"`{column}` ≤ {maximum:g}")

    # A missing value cannot be shown to meet the criterion, so it does not.
    # Reported separately rather than folded into the excluded count, because
    # "excluded for being outside the range" and "excluded for being unknown"
    # are different lines in a flow diagram.
    missing = s.isna()
    mask &= ~missing

    kept = [l for l in df.index[mask]]
    n_before, n_after = int(len(df)), int(len(kept))
    if n_after == 0:
        raise EligibilityRefusal(
            f"That criterion removes every row ({n_before} of {n_before}). "
            "Either the range is wrong or the column is not what it looks "
            "like — nothing downstream can run on an empty study.")

    return {
        "answer": RESTRICTED,
        "column": str(column),
        "minimum": None if minimum is None else float(minimum),
        "maximum": None if maximum is None else float(maximum),
        "keep_values": None if not keep_values else [str(v) for v in keep_values],
        "reason": reason.strip(),
        "criterion": " and ".join(parts),
        # Participant flow, which is the point of recording this at all.
        "n_before": n_before,
        "n_excluded": n_before - n_after,
        "n_excluded_missing": int((missing).sum()),
        "n_after": n_after,
        "labels": kept,
        "sentence": (
            f"{n_before - n_after} of {n_before} rows were excluded before the "
            f"held-out set was drawn: {' and '.join(parts)}. {reason.strip()}"),
    }


def everyone(df: pd.DataFrame) -> Dict[str, Any]:
    """The other answer, recorded rather than assumed.

    "The study is about everyone here" and "nobody was asked" must not look the
    same in the record — the second is a step that did not happen, and a reader
    of the methods section has to be able to tell.
    """
    return {
        "answer": EVERYONE,
        "column": None, "minimum": None, "maximum": None, "keep_values": None,
        "reason": "", "criterion": None,
        "n_before": int(len(df)), "n_excluded": 0, "n_excluded_missing": 0,
        "n_after": int(len(df)), "labels": list(df.index),
        # `DRIVE-031`. THE SECOND CLAUSE WAS NOT THIS MODULE'S TO MAKE, and it
        # was false. It read "and the held-out set is drawn from all of them",
        # and the held-out set is not: `engine.draw_holdout` opens with
        # `eligible = df.index[y.notna()]`, so every row with a missing outcome
        # is dropped before anything is drawn. On the tester's NHANES file that
        # is 15,552 of 21,849 — the receipt said 21,849, the seal drew 945, and
        # 945/21,849 is 4.3% while the seal called it 15%.
        #
        # The first clause is TRUE and stays: eligibility really did exclude
        # nobody, and `n_before == n_after == len(df)` records exactly that.
        # What it must not do is describe a draw it does not perform. The seal
        # names its own base now, which is where that claim belongs — the
        # module that drops the rows is the module that reports it.
        "sentence": EVERYONE_SENTENCE.format(n=len(df)),
    }


def disclosure(record: Optional[Dict[str, Any]]) -> str:
    """What the user reads once eligibility is settled.

    Keyed on the recorded answer, for the same reason the seal's disclosure is
    keyed on its basis: an unrestricted study and a restricted one must not
    render alike, because they are not the same claim about who the model is for.
    """
    if not record:
        return ""
    if record["answer"] == EVERYONE:
        return record["sentence"]
    return (record["sentence"] + " Those rows are gone before anything is held "
            "out, so the held-out set describes the population you studied "
            "rather than a wider one.")


# ─────────────────────────────────────────────────────────────────────────────
# THE THREE INSTINCTS AT THE IMPOSSIBILITY CARD — `GUIDED-166`
#
# The product owner, looking at four `sbp` entries outside 40–300 mmHg, named
# three things he might want. They are three different objects and the app
# offered exactly one:
#
#   set the entries to missing   constitution clause 06's row-local repair.
#                                BUILT — `project.set_impossible_missing`.
#   exclude the rows             clause 04's ELIGIBILITY CRITERION, which is
#                                this module. BUILT — `set_eligibility`,
#                                applied pre-seal, changing N, reported in
#                                participant flow. `ml/pipeline.py`'s
#                                `apply_plausibility_filter` is the same idea
#                                on the Streamlit door and is reached from
#                                `pages/05_Preprocess.py`. It was UNROUTED from
#                                this door, which is `MISC-014`'s distinction:
#                                unrouted is not absent.
#   mark the column corrupted    `GUIDED-096`'s split, and deliberately
#                                UNBUILT. It is named here rather than omitted,
#                                because a shelf that silently holds two of
#                                three tells the user the third is not a thing
#                                you may want.
#
# **Why the composer lives in this module.** The route that was missing is §04's,
# and §04's rules are what make the route hard: pre-seal only, grain answered
# first, N changes, a reason is required because participant flow reports one.
# A composer that lived beside the card would restate those four things, and a
# restated rule is a rule that drifts from the one `set_eligibility` enforces.
#
# **Nothing here applies anything.** Each route carries the decision that takes
# it, in the shape `POST /project/{id}/decision` accepts, so a client holding
# the payload can act — which is this project's standing test for whether a
# capability is reachable rather than merely present.

#: The one field the app must not fill in. `build_criterion` refuses a
#: criterion with no reason because participant flow reports WHY, and a reason
#: the app invented would be a methods sentence nobody wrote. The route says so
#: rather than shipping a payload that 400s with no explanation.
ELIGIBILITY_NEEDS_A_REASON = (
    "Why this restriction, in your words. Participant flow reports how many "
    "rows were excluded and why, so this becomes a sentence in your methods "
    "section — which is why the app will not write it for you.")


def routes_from_impossible(block: Dict[str, Any]) -> Dict[str, Any]:
    """What may be done about one impossible-tier block, all three of it.

    Returns the routes AND, for a column whose reading is itself in doubt, the
    reason there are none — because an empty list with no explanation asserts
    that nothing can be done, and what is true there is that the question is
    the column's reading rather than its outliers.
    """
    column = str(block.get("column"))
    low, high = block.get("low"), block.get("high")
    unit = block.get("unit") or ""
    n_flagged = int(block.get("n_flagged") or 0)

    if block.get("whole_column_suspect"):
        return {
            "routes": [],
            "withheld": (
                f"`{column}` reads as a whole-column unit or coding problem "
                f"rather than as entry errors, so none of the three routes "
                f"below applies: repairing individual values would delete real "
                f"data and leave the reading wrong. The column's reading is "
                f"the question. " + str(block.get("suspect_reason") or "")).strip(),
        }

    return {"withheld": None, "routes": [
        {
            "id": "set_to_missing",
            "label": "Set these entries to missing",
            "built": True,
            "clause": "06",
            "what_it_does": (
                f"{n_flagged} cell(s) become blank and every other value on "
                f"those rows is kept. Row-local, so it happens now — and the "
                f"blanks it makes are recorded as this app's, so the "
                f"missingness question later can tell them from blanks the "
                f"file arrived with."),
            "decision": {"kind": "set_impossible_missing", "subject": column,
                         "payload": {"column": column}},
            "typed": None,
        },
        {
            "id": "exclude_the_rows",
            "label": "Exclude those rows from the study",
            "built": True,
            "clause": "04",
            "what_it_does": (
                f"The whole row leaves the table — every other measurement on "
                f"it too — before anything is held out, and N changes. That "
                f"makes it an eligibility criterion rather than a repair: "
                f"TRIPOD+AI reports it in participant flow with its count and "
                f"its reason. Keeping `{column}` between {low} and {high} "
                f"{unit} is the criterion this would record."),
            "decision": {
                "kind": "set_eligibility", "subject": column,
                "payload": {"answer": RESTRICTED, "column": column,
                            "minimum": low, "maximum": high, "reason": ""}},
            # The route names its own preconditions rather than failing at
            # them. §01 fixes the order and `set_eligibility` enforces it; a
            # route that did not say so would read as available at a moment it
            # is not.
            "needs": [
                "the grain question answered — §01 puts it before eligibility",
                "the held-out set not yet sealed — §04 refuses an exclusion "
                "afterwards, because the held-out rows were drawn from a "
                "population that included the rows you are excluding",
            ],
            "typed": {"field": "reason", "prompt": ELIGIBILITY_NEEDS_A_REASON},
        },
        {
            "id": "mark_the_column_corrupted",
            "label": "Mark the whole column as not trustworthy",
            "built": False,
            "clause": None,
            "what_it_does": (
                "Withdraws `{column}` from the models and says in the methods "
                "section that it was withdrawn and why — rather than repairing "
                "entries in a column you do not believe."
            ).format(column=column),
            "not_built_reason": (
                "`GUIDED-096`'s split, and it is not built. It is named here "
                "rather than left off the shelf: the two routes above are what "
                "the app can do, and a list of two with no third would say "
                "that distrusting the column is not a thing you may want. "
                "Until it exists, the app can record that you do not trust "
                "this column and cannot act on it."),
            "decision": None,
            "typed": None,
        },
    ]}
