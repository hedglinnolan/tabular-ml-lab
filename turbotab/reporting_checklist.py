"""L52-C — the reporting checklist, and the half of it that depends on nothing.

Three of four research threads independently named the auto-populated reporting
checklist as the deliverable most likely to make researchers recommend this
tool. It has slipped twice, both times correctly, because it reads the
manuscript and the manuscript was changing.

**That argument only ever covered half of it.** `DOMAIN_SCIENCE.md` §01.7 says
the checklist is *"a checklist-shaped artifact with two column types: what the
app knows, and what it must ask."* **The second column depends on nothing.** So
the artifact and the asking column are built here, and L53 wires population into
a structure that already exists rather than inventing both at once.

## One checklist, and which one

`DOMAIN_SCIENCE.md` §05 item 6 is explicit that this seeds with one instrument
rather than four. **TRIPOD+AI**, because a clinical prediction model is what
this app mostly builds and TRIPOD+AI is the instrument a reviewer will actually
apply. STROBE-nut, COSMIN and mQACC come after it proves the artifact.

## The count, and the part of it this file does not have

`research/CLINICAL_SURVEY_PACK.md` §A6: TRIPOD+AI is a **27-item checklist**
(Collins et al., *BMJ* 2024;385:e078378), superseding TRIPOD 2015. **§A6
enumerates twelve of those twenty-seven** — the ones it says TurboTab can
substantially auto-populate — and those twelve are what `ITEMS` below contains.

**The other fifteen are not here, and they are not invented.** This project
takes domain science from `docs/turbotab/research/` by section and never from
recollection, and a checklist item written from memory of a BMJ paper is exactly
the sentence the governing rule forbids: an assertion the app cannot source.
`missing_items()` states the gap in the artifact rather than leaving the reader
to infer completeness from a table that looks complete. Filling it is pack work
— read the instrument, add the items to §A6 — not a loop that guesses.

## The four fill-states, and why not a boolean

`research/NUTRITION_PACK.md` §09's table is the two column types made concrete:
fourteen requirements each marked `user`, `app`, `app detects + user confirms`,
or `user + app template`. **A two-value column would lose the third state**,
which is a real one — *the app detects it and you confirm* is neither "we know
it" nor "you tell us", and a checklist that collapses it either claims a fact
the user never agreed to or asks for one the app already has.

## L53-B — the column fills itself, by QUOTING

The manuscript has stopped moving: `AUDIT-026` and `029`'s cross-validation
claims are `FIXED`, `AUDIT-016`'s caption is `FIXED`, and `AUDIT-022`/`023`
closed this loop. So `auto_filled` and `where_addressed` are populated — **from
the draft's own sentences, quoted verbatim.**

**A cell that paraphrased would be a second copy of the claim.** L36 settled
this for the methods section: it quotes the record's own string, and that is
only safe because L35-B made the string true of the fit. The same reasoning
applies one surface up — the checklist is read beside the manuscript, and two
wordings of one fact is how they drift apart. `fill()` therefore does exactly
one thing: it selects sentences the draft already composed.

**`where_addressed` names a section the draft actually contains**, checked
against the draft it was handed rather than a constant. A pointer to a section
the document does not have is `AUDIT-001`'s shape inside the artifact built to
catch it.

**An item the draft cannot fill keeps its sentence.** Auto-population never
overwrites a `not_filled_because` with a blank — and where the draft says WHY a
section is empty, that reason replaces the generic one, because the draft's is
specific to this study.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

#: `research/CLINICAL_SURVEY_PACK.md` §A6, verbatim.
INSTRUMENT = "TRIPOD+AI"
CITATION = "Collins et al., BMJ 2024;385:e078378"
SOURCE = "research/CLINICAL_SURVEY_PACK.md#A6"

#: §A6: *"a 27-item checklist superseding TRIPOD 2015"*. This is the
#: INSTRUMENT's count, not this file's — see `missing_items`.
INSTRUMENT_ITEM_COUNT = 27

# ── the fill-states, `research/NUTRITION_PACK.md` §09 ────────────────────────
USER = "user"
APP = "app"
APP_DETECTS_USER_CONFIRMS = "app detects + user confirms"
USER_PLUS_APP_TEMPLATE = "user + app template"
FILL_STATES = (USER, APP, APP_DETECTS_USER_CONFIRMS, USER_PLUS_APP_TEMPLATE)
FILL_STATE_SOURCE = "research/NUTRITION_PACK.md#09"

#: `GUIDED-179`, and the same string `turbotab/figures.py` uses. A quantity the
#: app does not have is named as absent rather than rendered as `None`.
NOT_YET_FILLED = "not filled by the app yet"

_WHY_NOT_FILLED = (
    "Auto-population is deliberately unbuilt in this loop: it reads the "
    "manuscript, and the manuscript is being corrected by the same loop. The "
    "column exists so that filling it is wiring rather than invention.")


@dataclass(frozen=True)
class ChecklistItem:
    """One row of the rendered table.

    `needs_from_you` is **always** populated — it is the column that depends on
    nothing, and an item that cannot say what it wants from the author is an
    item that has not been thought about.
    """

    key: str
    item: str
    fills: str
    needs_from_you: str
    #: The EXACT run of words in §A6 this item comes from, so *not invented* is
    #: a check rather than a claim. `test_every_item_traces_to_the_pack`
    #: resolves each one against the file; a reworded item whose trace no
    #: longer resolves fails there rather than shipping as domain science
    #: nobody sourced.
    traces_to: str = ""
    source: str = SOURCE
    auto_filled: Optional[str] = None


#: The twelve §A6 enumerates. Each `item` is that section's own wording; each
#: `needs_from_you` is what the author must supply even once L53 lands, which is
#: why it is written now and not later.
ITEMS: List[ChecklistItem] = [
    ChecklistItem(
        key="identification",
        item="Title and abstract identify the study as developing and/or "
             "validating a prediction model",
        fills=APP_DETECTS_USER_CONFIRMS,
        needs_from_you=(
            "Confirm whether this is development, validation, or both. The app "
            "can see that a model was fitted and whether a held-out set was "
            "scored; it cannot see whether you intend the paper to make a "
            "validation claim about a DIFFERENT population, which is the "
            "distinction the reader is being told about."),
        traces_to="title/abstract identification as a development and/or validation study",
    ),
    ChecklistItem(
        key="data_source",
        item="Source of data, setting, eligibility criteria, and participant "
             "flow",
        fills=APP_DETECTS_USER_CONFIRMS,
        needs_from_you=(
            "The setting and the eligibility criteria. The app can count rows "
            "in and rows out at every step it performed, and it cannot know "
            "what happened before the file reached it — who was approached, "
            "who declined, and what the inclusion rule was."),
        traces_to="source of data, setting, eligibility",
    ),
    ChecklistItem(
        key="outcome",
        item="Outcome definition, and whether outcome assessment was blinded "
             "to the predictors",
        fills=USER_PLUS_APP_TEMPLATE,
        needs_from_you=(
            "How the outcome was ascertained and whether the assessor could "
            "see the predictors. The app knows which COLUMN is the outcome; "
            "blinding is a fact about how the data was collected and is not "
            "recoverable from the table."),
        traces_to="outcome definition and blinding of outcome assessment to predictors",
    ),
    ChecklistItem(
        key="predictors",
        item="Predictor definitions and measurement, including their timing "
             "relative to the index",
        fills=APP_DETECTS_USER_CONFIRMS,
        needs_from_you=(
            "When each predictor was measured relative to the index. This is "
            "the one that sinks papers: a predictor recorded AFTER the moment "
            "the model is meant to predict from is leakage, and nothing in a "
            "flat table records when a column was measured."),
        traces_to="predictor definitions and measurement, including timing relative to the index",
    ),
    ChecklistItem(
        key="missing_data",
        item="Handling of missing data, with the mechanism discussed rather "
             "than the method merely named",
        fills=APP,
        needs_from_you=(
            "Whether the mechanism you told the app is the one you believe. "
            "The app records the method, the scope it was fitted over, and the "
            "mechanism you declared per column; the JUSTIFICATION for that "
            "mechanism is domain knowledge and §A6 asks for it explicitly."),
        traces_to="handling of missing data with the mechanism discussed, not just the method named",
    ),
    ChecklistItem(
        key="sample_size",
        item="Sample size justification",
        fills=APP,
        needs_from_you=(
            "Nothing, if the app's criterion is the one you want to report. It "
            "counts candidate predictor PARAMETERS rather than columns and "
            "names the criterion it used. If you sized the study in advance "
            "against a different target, report yours and say so."),
        traces_to="sample size justification",
    ),
    ChecklistItem(
        key="model_building",
        item="Model-building procedure including predictor selection, and all "
             "hyperparameters and the tuning procedure",
        fills=APP,
        needs_from_you=(
            "Nothing the app cannot supply, provided every choice went through "
            "it. A step taken outside the app — a column dropped by hand "
            "before upload, a threshold chosen by eye — is invisible here and "
            "is the usual reason this item is wrong."),
        traces_to="model-building procedure including selection, and",
    ),
    ChecklistItem(
        key="performance",
        item="Performance reported as discrimination AND calibration AND "
             "clinical utility",
        fills=APP,
        needs_from_you=(
            "The decision threshold, and the relative cost of a false positive "
            "against a false negative, if clinical utility is to mean "
            "anything. §A6 requires all three; discrimination alone is the "
            "most common shortfall and the app cannot choose your threshold."),
        traces_to="performance: discrimination AND calibration AND clinical utility",
    ),
    ChecklistItem(
        key="model_presentation",
        item="Model presentation sufficient for others to compute predictions "
             "— full coefficients and intercept, or the model object",
        fills=APP,
        needs_from_you=(
            "Where you will deposit the model object, if the model is not one "
            "whose coefficients fit in a table. A reader who cannot compute a "
            "prediction from your paper has not been given the model."),
        traces_to="model presentation sufficient for others to compute predictions",
    ),
    ChecklistItem(
        key="fairness",
        item="Fairness and subgroup evaluation",
        fills=USER_PLUS_APP_TEMPLATE,
        needs_from_you=(
            "Which subgroups matter for this population and this use. The app "
            "can compute performance within any grouping you name; it cannot "
            "decide which groupings a reader will hold you to, and choosing "
            "them from what is in the file is how the relevant one gets "
            "missed."),
        traces_to="fairness/subgroup evaluation",
    ),
    ChecklistItem(
        key="open_science",
        item="Open-science items: data availability, code availability, "
             "protocol, funding, conflicts of interest",
        fills=USER,
        needs_from_you=(
            "All five. None of them is a property of the data, and the app "
            "will not draft a funding statement or a conflicts declaration on "
            "your behalf."),
        traces_to="open-science items (data, code, protocol, funding, conflicts)",
    ),
    ChecklistItem(
        key="limitations",
        item="Limitations, intended use, intended users, and setting",
        fills=USER_PLUS_APP_TEMPLATE,
        needs_from_you=(
            "Who is meant to use this model, on whom, and to decide what. The "
            "app can list the limitations it MEASURED — sample size, "
            "missingness, subgroup coverage — and intended use is a claim "
            "about the world that only you can make."),
        traces_to="limitations and intended use, users and setting",
    ),
]

#: §A6, one sentence and not a second checklist. Quoted rather than paraphrased.
PROBAST_NOTE = (
    "PROBAST (Wolff et al., Ann Intern Med 2019) — 4 domains and 20 signaling "
    "questions. “If your paper is later included in a systematic review, "
    "this is the instrument that will be applied to it. Reading it before you "
    "write is cheap insurance.”")
PROBAST_SOURCE = SOURCE


def missing_items() -> Dict[str, Any]:
    """The gap between the instrument's 27 and this file's 12, stated.

    A checklist that renders twelve rows under the name of a twenty-seven-item
    instrument is claiming completeness it does not have — the same shape as a
    truncated list nobody records (`GUIDED-195`). The number is carried in the
    artifact so the reader sees it beside the table.
    """
    return {
        "instrument_items": INSTRUMENT_ITEM_COUNT,
        "enumerated_here": len(ITEMS),
        "not_yet_enumerated": INSTRUMENT_ITEM_COUNT - len(ITEMS),
        "why": (
            f"{SOURCE} enumerates {len(ITEMS)} of {INSTRUMENT}'s "
            f"{INSTRUMENT_ITEM_COUNT} items — the ones it says this app can "
            f"substantially auto-populate. The remaining "
            f"{INSTRUMENT_ITEM_COUNT - len(ITEMS)} are not reproduced here "
            f"because they are not in the pack, and writing them from "
            f"recollection of the source paper is the one thing this project "
            f"does not do with domain science."),
        "how_to_close": (
            f"Add the remaining items to {SOURCE} from the instrument itself, "
            f"then extend ITEMS. It is pack authoring, not a code change."),
    }


#: Which draft section answers which checklist item. The values are the titles
#: `turbotab/draft.py` composes; `fill()` resolves each against the draft it is
#: handed and refuses to point at one that is not there, so a renamed section
#: shows up as an unfilled item rather than as a dangling pointer.
SECTION_FOR = {
    "data_source": "Data preparation",
    "outcome": "Outcome and analysis population",
    "predictors": "Feature handling",
    "missing_data": "Missing data and preprocessing",
    "limitations": "Limitations",
}
# TWO ITEMS WERE MAPPED HERE IN A FIRST DRAFT AND BOTH WERE WRONG, in the way
# this whole project is about. A section is only this item's answer if its
# sentences ANSWER THE ITEM — being nearby is not enough.
#
#   `sample_size` -> "Data preparation" quoted *"clinical_labs.csv was loaded:
#   288 rows, 19 columns."* under an item that asks for a sample-size
#   JUSTIFICATION. A row count is not a justification, and putting it in that
#   cell would have the app assert that the study's size had been justified
#   when nothing had justified it. §A5.4 is [SETTLED] that the criteria-based
#   calculation is what is owed and that EPV>=10 is superseded (`AUDIT-021`).
#
#   `identification` -> the same section quoted *"modeled as a classification
#   problem"* under an item asking whether this is a DEVELOPMENT or a
#   VALIDATION study. That is the task type, which is a different fact.
#
# Both are in `_FILLED_BY_STEP` now, naming what would fill them. An unfilled
# cell with a reason costs a reader nothing; a filled cell with the wrong
# sentence costs them the thing they were checking.

#: Items §A6 lists that no draft section carries yet, each with the step that
#: would fill it. They are NOT in `SECTION_FOR` because inventing a pointer is
#: worse than an honest absence, and `fill()` gives each of these its own
#: reason rather than the generic one.
_FILLED_BY_STEP = {
    "sample_size": (
        "a sample-size check has run over the candidate predictor PARAMETERS. "
        "`ml/sample_size.py` holds that arithmetic and `turbotab/resolution.py` "
        "the criteria-based calculation; neither reaches the draft yet, and the "
        "row count in Data preparation is a count rather than a justification"),
    "identification": (
        "the app can tell a development study from a validation one. It can see "
        "that a model was fitted and whether a held-out set was scored; the "
        "draft records the TASK, which is a different fact"),
    "model_building": "the models are chosen and the split is drawn",
    "performance": "a run has scored the held-out rows",
    "model_presentation": "a model has been fitted",
    "fairness": "you name the subgroups this population is held to",
    "open_science": "you supply them; none is a property of the data",
}


def _sections(draft: Any) -> Dict[str, Dict[str, Any]]:
    return {str(s.get("title")): s
            for s in (draft or {}).get("sections", []) or []
            if s.get("title")}


def fill(item: ChecklistItem, draft: Any) -> Dict[str, Any]:
    """What the draft says about this item, QUOTED — never re-composed.

    Returns `where_addressed`, `auto_filled` and, when either is absent, the
    reason. The only string this function creates is the join between sentences
    the draft already wrote; every claim in the cell is the draft's.
    """
    sections = _sections(draft)
    title = SECTION_FOR.get(item.key)

    if title is None:
        step = _FILLED_BY_STEP.get(item.key)
        return {"where_addressed": None, "auto_filled": None,
                "not_filled_because": (
                    f"No section of the draft carries this yet. It is filled "
                    f"when {step}." if step else _WHY_NOT_FILLED)}

    if title not in sections:
        # `AUDIT-001`'s shape: a pointer at a section the document lacks. The
        # honest form is to say the pointer did not resolve, not to print it.
        return {"where_addressed": None, "auto_filled": None,
                "not_filled_because": (
                    f"This item is addressed in the draft's "
                    f"'{title}' section, and this draft has no such section "
                    f"({', '.join(sorted(sections)) or 'the draft is empty'}).")}

    section = sections[title]
    said = [str(x.get("text", "")).strip()
            for x in (section.get("sentences") or []) if x.get("text")]
    if not said:
        # The draft's OWN reason beats the generic one: it is about this study.
        return {"where_addressed": None, "auto_filled": None,
                "not_filled_because": (
                    str(section.get("waiting_for") or "").strip()
                    or f"The draft's '{title}' section is empty.")}

    return {"where_addressed": title,
            "auto_filled": " ".join(said),
            "not_filled_because": None}


def render(project: Any = None, draft: Any = None) -> Dict[str, Any]:
    """The artifact, in §A6's four columns, populated from the draft.

    §A6's presentation, verbatim: **item | where addressed | auto-filled text |
    ⚠ needs your input.**

    `draft` is `turbotab/draft.py`'s document. Absent, every cell renders as an
    honest absence — which is the L52 behavior, kept rather than replaced, so
    a caller that cannot produce a draft still gets the asking column.
    """
    rows = []
    n_filled = 0
    for item in ITEMS:
        got = fill(item, draft)
        if got["auto_filled"]:
            n_filled += 1
        rows.append({
            "key": item.key,
            "item": item.item,
            "where_addressed": got["where_addressed"],
            "where_addressed_text": got["where_addressed"] or NOT_YET_FILLED,
            "auto_filled": got["auto_filled"],
            "auto_filled_text": got["auto_filled"] or NOT_YET_FILLED,
            "not_filled_because": got["not_filled_because"],
            "needs_from_you": item.needs_from_you,
            "fills": item.fills,
            "source": item.source,
        })
    return {
        "instrument": INSTRUMENT,
        "citation": CITATION,
        "source": SOURCE,
        "columns": ["item", "where addressed", "auto-filled text",
                    "needs your input"],
        "fill_states": list(FILL_STATES),
        "fill_state_source": FILL_STATE_SOURCE,
        "rows": rows,
        "coverage": missing_items(),
        "probast": {"note": PROBAST_NOTE, "source": PROBAST_SOURCE},
        "n_auto_filled": n_filled,
        "auto_population_built": True,
    }
