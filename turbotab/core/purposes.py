"""Every element earns its place: the purpose registry, the Python half (BLUEPRINT §11.2).

Nolan, 2026-10-01: *"every part of what is presented to the user serves a legitimate pedagogical
purpose."* An element on screen answers a question the user has at that moment, and there are five
such questions. Every consequence view kind the server can send, and every kind of coach anchor,
declares here which of them it answers and what it tells the user. A kind with no entry fails
``tests/test_purposes.py``, so a new kind arrives with a purpose or not at all. The frontend's
``src/components/stage/purposes.ts`` mirrors this for what it renders (M2_CONTRACT §11).

The closed vocabulary of views (BLUEPRINT §11 rule 2) is why this scales to a myriad of decisions:
each kind's purpose is audited once and holds for every decision that uses it. The parts every
preview shares (its title, caption, basis line, note, caution, storyboard) are declared too, so the
pedagogy reviewer can read one table for the whole stage.
"""
from __future__ import annotations

from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field

# The five questions, in BLUEPRINT §11.2's order. The keys are the contract the frontend mirrors.
Question = Literal[
    "what_is_this_choice",
    "what_will_it_change",
    "why_does_it_matter",
    "is_my_data_okay",
    "what_did_i_decide",
]
QUESTIONS: dict[str, str] = {
    "what_is_this_choice": "What is this choice?",
    "what_will_it_change": "What will it change in my data or my model?",
    "why_does_it_matter": "Why does that matter for my result?",
    "is_my_data_okay": "Is my data okay?",
    "what_did_i_decide": "What did I decide, and can a reviewer reproduce it?",
}
SAYS_WORDS = 20


class Purpose(BaseModel):
    """What one kind of element is for: the questions it answers, and what it tells the user."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    answers: tuple[Question, ...] = Field(min_length=1)
    says: str  # ≤ SAYS_WORDS words: what the user learns from it, on their own data


def _p(*answers: Question, says: str) -> Purpose:
    return Purpose(answers=answers, says=says)


# Every ``kind`` in ``consequences.ConsequenceView``.
VIEW_KINDS: dict[str, Purpose] = {
    "row_flow": _p(
        "what_will_it_change", "what_did_i_decide",
        says="Which rows the choice keeps, combines, excludes or holds out, counted step by step."),
    "lineage": _p(
        "what_will_it_change", "what_did_i_decide",
        says="Which columns reach the model matrix, and what each became on the way."),
    "table_focus": _p(
        "what_will_it_change", "is_my_data_okay",
        says="The very cells the choice changes, on a few of the user's own rows."),
    "distribution": _p(
        "what_will_it_change", "is_my_data_okay",
        says="How one column's values spread before and after, with the choice's cut-offs marked."),
    "relationship": _p(
        "what_will_it_change", "why_does_it_matter",
        says="How two columns move together before and after: what the method takes out."),
}

# Every ``kind`` of ``consequences.CoachAnchor``: what a coach note pointing there is for.
COACH_ANCHORS: dict[str, Purpose] = {
    "column": _p(
        "is_my_data_okay", "why_does_it_matter",
        says="A fact about one column of the user's data that bears on this choice."),
    "range": _p(
        "is_my_data_okay", "why_does_it_matter",
        says="How many rows fall in a stretch of values the choice treats differently."),
    "points": _p(
        "is_my_data_okay",
        says="Which rows in the picture the fact is about."),
    "step": _p(
        "what_will_it_change", "why_does_it_matter",
        says="What one step of the row flow costs or keeps, and why it matters."),
}

# The parts every preview carries beside its views (``consequences.PreviewResult`` and ``_View``).
PREVIEW_PARTS: dict[str, Purpose] = {
    "title": _p("what_is_this_choice", says="Names what the picture shows."),
    "caption": _p("what_will_it_change",
                  says="The one fact the picture shows, in numbers from the user's table."),
    "story": _p("what_will_it_change",
                says="The method's own real steps between before and after, played by the flip."),
    "marks": _p("what_will_it_change", says="Where the choice cuts or flags values on the axis."),
    "basis": _p("what_did_i_decide",
                says="Which rows the preview read, and that held-out rows stayed sealed."),
    "note": _p("what_will_it_change", says="What the choice does that no picture can show."),
    "caution": _p("why_does_it_matter",
                  says="A concern about this choice on this data, with the control that resolves it."),
}


def view_kinds() -> set[str]:
    """Every view kind the consequence planner can send (from its discriminated union)."""
    from turbotab.core.consequences import ConsequenceView

    union = get_args(get_args(ConsequenceView)[0])
    return {str(model.model_fields["kind"].default) for model in union}


def anchor_kinds() -> set[str]:
    """Every kind of coach anchor (from ``CoachAnchor.kind``)."""
    from turbotab.core.consequences import CoachAnchor

    return set(get_args(CoachAnchor.model_fields["kind"].annotation))


def purpose_of(kind: str) -> Purpose:
    """The purpose of a view kind, else of a coach anchor kind; KeyError when it has none."""
    if kind in VIEW_KINDS:
        return VIEW_KINDS[kind]
    return COACH_ANCHORS[kind]


__all__ = ["COACH_ANCHORS", "PREVIEW_PARTS", "QUESTIONS", "Purpose", "Question", "SAYS_WORDS",
           "VIEW_KINDS", "anchor_kinds", "purpose_of", "view_kinds"]
