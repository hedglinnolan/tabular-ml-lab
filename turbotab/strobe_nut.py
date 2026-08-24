"""turbotab.strobe_nut — the six things a nutrition reviewer checks, in order.

`GUIDED-123`, and it is the product owner's question answered concretely rather
than as four disconnected sentences. `research/NUTRITION_PACK.md` §09:

> *"A nutrition reviewer reads your methods in a fixed order and checks six
> things: (1) what instrument, administered how many times, covering what
> period; (2) what food composition database and version; (3) what validation
> exists in a population like yours, with the actual correlation coefficients;
> (4) how you handled misreporting, with the equation and cut-off; (5) which
> energy adjustment model, named; (6) how within-person variation was handled.
> **Missing any one is the most common reason for a methods revise-and-resubmit
> in this field.**"*

**§09's fixed order is part of the specification, not presentation.** A
checklist a reviewer reads top to bottom is a different artifact from the same
items in a different order, so `ITEMS` is a tuple and the order is asserted.

## What this is and what it is not

It is **the checklist reaching the manuscript**. Each item says who can answer
it — §09's own table marks each `user`, `app`, or `app detects + user
confirms` — and what the app currently holds. Where the app owes an answer and
does not have one, the item says so in the words of the gap rather than
rendering blank, because a checklist with empty rows reads as a study that had
none.

It is **not** the checklist engine (`DOMAIN_SCIENCE` primitive 6,
`GUIDED-111`). That is one artifact over four checklist definitions with two
column types, and it is unbuilt. This is one checklist's content, in the shape
that engine will read.

## The four the app owes and does not have

§09 marks these `app`, and none of them is computed anywhere:

* **energy adjustment, with the model named** — residual, standard multivariate,
  nutrient density or partition, and which one changes the estimand;
* **misreporting handling** — definition, equation, cut-off, PAL, n excluded;
* **usual-intake handling** — number of recalls, method, covariates;
* **complex survey design** — weights, strata, PSU, subpopulation handling,
  variance method.

They are **stated as owed** rather than silently missing, and each carries what
building it needs. That is the honest form of *err toward more information*:
the gaps are information too.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

#: Who can answer an item, in §09's own vocabulary.
USER = "user"
APP = "app"
BOTH = "app detects + user confirms"

#: Where the checklist comes from. Read at build; the gate resolves it.
EVIDENCE = {
    "source": "research/NUTRITION_PACK.md#09 · Reporting standards",
    "evidence_status": "CONVENTION",
    "claim": ("STROBE-nut (Lachat et al., PLoS Medicine 2016) adds 24 "
              "nutrition-specific recommendations to STROBE, developed by 21 "
              "experts with a 3-round Delphi involving 53 external experts, "
              "and is in the author instructions of a growing set of "
              "journals."),
}

#: **THE SIX, IN THE ORDER A REVIEWER READS THEM.** The order is specification:
#: §09 says *a nutrition reviewer reads your methods in a fixed order*, and a
#: checklist that reordered them would be a different artifact.
#:
#: Each entry is `(key, question, who, what_the_app_holds, what_it_needs)`.
#: `what_the_app_holds` is `None` where the app owes an answer and has none —
#: which is four of the six, and saying so is the point.
ITEMS: Tuple[Dict[str, Any], ...] = (
    {
        "key": "instrument",
        "order": 1,
        "question": ("What dietary assessment instrument, administered how "
                     "many times, covering what time frame, in what mode?"),
        "who": USER,
        "app_holds": None,
        "needs": ("The instrument's name and version, the number of "
                  "administrations and the period each covers. The app cannot "
                  "read any of this from a table of numbers — a column of "
                  "energy intakes looks the same whether it came from one "
                  "24-hour recall or a validated 110-item FFQ, and the two "
                  "support completely different claims."),
    },
    {
        "key": "food_composition_database",
        "order": 2,
        "question": ("Which food composition database and version, and how "
                     "were non-matching foods handled?"),
        "who": USER,
        "app_holds": None,
        "needs": ("The database and its release year — USDA FNDDS 2019-2020 "
                  "and 2021-2023 give different answers for the same food. "
                  "Not inferable from intake values."),
    },
    {
        "key": "validation",
        "order": 3,
        "question": ("What validation exists in a population like yours, "
                     "when, against what reference method, with the actual "
                     "correlation coefficients?"),
        "who": USER,
        "app_holds": None,
        "needs": ("The reference method, the population, and the validity "
                  "coefficients AS NUMBERS. §09 is explicit that *'the FFQ "
                  "was previously validated' is not reporting* — published "
                  "validation correlations in the substitution-modeling "
                  "literature range from 0.12 to 0.77 with a median of 0.43, "
                  "and a reader cannot judge a study without knowing where in "
                  "that range its instrument falls."),
    },
    {
        "key": "misreporting",
        "order": 4,
        "question": ("How was misreporting handled — with the definition, the "
                     "equation, the cut-off, the PAL, and the n excluded?"),
        "who": APP,
        "app_holds": None,
        "needs": ("A misreporting rule: a predicted-energy-requirement "
                  "equation, a physical activity level, a cut-off on the "
                  "ratio, and the count excluded by it. §09 lists "
                  "*'Implausible intakes were excluded' with no rule, PAL or "
                  "n* as a STROBE-nut violation, and the app owes this one — "
                  "it has the intakes and the covariates and computes "
                  "nothing. `GUIDED-123`."),
    },
    {
        "key": "energy_adjustment",
        "order": 5,
        "question": "Which energy adjustment model, named?",
        "who": APP,
        "app_holds": None,
        "needs": ("The model by name — residual, standard multivariate, "
                  "nutrient density or partition — because which one is used "
                  "changes what the coefficient means. §09 lists *'we "
                  "adjusted for energy' without naming the model* as a "
                  "STROBE-nut violation. The app owes it and computes no "
                  "adjustment at all. `GUIDED-123`."),
    },
    {
        "key": "within_person_variation",
        "order": 6,
        "question": ("How was within-person variation handled — how many "
                     "recalls, by what method, with which covariates?"),
        "who": APP,
        "app_holds": None,
        "needs": ("A usual-intake model. The app already asks the grain "
                  "question and knows how many rows each person contributes, "
                  "which is the input; what it does not have is the "
                  "measurement-error model that turns repeated recalls into "
                  "a usual-intake distribution. Reporting a prevalence from "
                  "single-day intakes without one is the pack's flagship "
                  "error. `GUIDED-123`."),
    },
)

#: Items §09 assigns to the app that are NOT in the reviewer's six but are
#: still owed. Kept separate so the six stay six — the reviewer's order is the
#: specification and padding it would break the thing it is for.
ALSO_OWED: Tuple[Dict[str, Any], ...] = (
    {
        "key": "complex_survey_design",
        "question": ("Weights, strata, PSU, subpopulation handling and "
                     "variance method"),
        "who": APP,
        "app_holds": None,
        "needs": ("Survey design objects. Analyzing NHANES or any complex "
                  "survey without them gives standard errors that are wrong "
                  "by a large factor, and the app applies none. "
                  "`GUIDED-123`."),
    },
    {
        "key": "units_and_basis",
        "question": ("Units, and whether intakes are per day, per 1000 kcal, "
                     "or as a percentage of energy"),
        "who": APP,
        "app_holds": None,
        "needs": ("The basis a nutrient column is on. The app records a lens "
                  "and a target and never asks this, so the same column can "
                  "be read three ways — per day, per 1000 kcal, or as a "
                  "percentage of energy — and a coefficient means something "
                  "different under each. `GUIDED-123`."),
    },
    {
        "key": "nutrient_requirement_standards",
        "question": ("Which DRI edition and country, and the inadequacy "
                     "method"),
        "who": APP,
        "app_holds": None,
        "needs": ("A Dietary Reference Intake table. `GUIDED-067`, and the "
                  "reason `turbotab/figure_specs` carries two PENDING figures "
                  "rather than drawing them."),
    },
    {
        "key": "missing_data",
        "question": "Mechanism assumed, method, and n affected",
        "who": APP,
        # THE ONE THE APP ACTUALLY HAS. §07 records a mechanism per column,
        # the strategy, and the count — and `draft.py` already exports the
        # sentence. Listed so the checklist is not uniformly red, which would
        # make it as uninformative as one that was uniformly green.
        "app_holds": "missingness",
        # NOT "the app cannot do this" — it can, and has not been asked yet.
        # An item waiting on a STEP and an item waiting on a BUILD are
        # different gaps, and a checklist that rendered them alike would tell
        # a user to wait for us when they should be answering a question.
        "needs": ("Nothing to build — route the missing values at the "
                  "Preprocess step and this fills in. §07 records a mechanism "
                  "per column, which is exactly what §09 asks for."),
        "waiting_on": "a step, not a build",
    },
    {
        "key": "participant_flow",
        "question": "Participant flow with dietary exclusions",
        "who": APP,
        "app_holds": "eligibility",
        "needs": ("Nothing to build — answer the eligibility question and "
                  "this fills in with the recorded criterion and the count "
                  "it excluded."),
        "waiting_on": "a step, not a build",
    },
)


def checklist(project: Any) -> Dict[str, Any]:
    """The six, in order, with what this project can answer.

    Every item comes back — `unanswered` is not a filter, it is a count — and
    an item the app owes and cannot answer carries the words of the gap rather
    than an empty cell.
    """
    rows: List[Dict[str, Any]] = []
    for item in ITEMS:
        rows.append({**item, "answer": _answer(project, item),
                     "answered": _answer(project, item) is not None})
    also = [{**item, "answer": _answer(project, item),
             "answered": _answer(project, item) is not None}
            for item in ALSO_OWED]

    owed_by_app = [r for r in rows + also
                   if r["who"] in (APP, BOTH) and not r["answered"]]
    return {
        "items": rows,
        "also_owed": also,
        "n_items": len(rows),
        "n_answered": sum(1 for r in rows if r["answered"]),
        "n_owed_by_the_app": len(owed_by_app),
        "owed_by_the_app": [r["key"] for r in owed_by_app],
        "headline": (
            f"{len(owed_by_app)} of the checklist items this app is "
            f"responsible for are not computed. A nutrition reviewer reads "
            f"the six methods questions in a fixed order and missing any one "
            f"is the most common reason for a methods revise-and-resubmit in "
            f"this field."),
        **EVIDENCE,
    }


def _answer(project: Any, item: Dict[str, Any]) -> Optional[str]:
    """What this project can say, or `None`.

    Only two items have a source today and both are answers the user already
    gave — which is the finding rather than an oversight, and returning `None`
    for the rest is what makes the count mean something.
    """
    holds = item.get("app_holds")
    if holds == "missingness":
        declared = getattr(project, "missingness", None) or []
        if not declared:
            return None
        mechanisms = {str(d.get("mechanism")) for d in declared}
        return (f"{len(declared)} column(s) have a recorded handling; "
                f"mechanism(s) recorded: {', '.join(sorted(mechanisms))}.")
    if holds == "eligibility":
        record = getattr(project, "eligibility", None)
        if not record:
            return None
        return str(record.get("sentence") or "")
    return None


def methods_sentences_from(result: Dict[str, Any]) -> List[str]:
    """The sentences from an already-computed checklist.

    Split from `methods_sentences` so the manuscript renders the SAME result
    it serves rather than recomputing one — two computations of one checklist
    are two checklists that agree today.
    """
    from turbotab import draft as _draft

    out: List[str] = []
    for row in list(result.get("items") or []) + list(result.get("also_owed") or []):
        if row.get("answered"):
            out.append(str(row["answer"]))
        elif row.get("who") in (APP, BOTH):
            out.append(f"{_draft.AUTHOR_GAP} — {row['question']} "
                       f"The app does not compute this yet.")
        else:
            out.append(f"{_draft.AUTHOR_GAP} — {row['question']}")
    return out


def methods_sentences(project: Any) -> List[str]:
    """The checklist as manuscript prose, in the reviewer's order.

    Only the answered items become sentences — an unanswered one is a gap for
    the author to fill, and `draft.AUTHOR_GAP` is how this app has always said
    that rather than writing a plausible sentence in the user's name.
    """
    from turbotab import draft as _draft

    return methods_sentences_from(checklist(project))
