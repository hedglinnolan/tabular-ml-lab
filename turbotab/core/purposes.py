"""Every element earns its place: the purpose registry, the Python half (BLUEPRINT §11.2).

Nolan, 2026-10-01: *"every part of what is presented to the user serves a legitimate pedagogical
purpose."* An element on screen answers a question the user has at that moment, and there are five
such questions. Every consequence view kind the server can send, every kind of coach anchor, and
every part a preview carries beside its views declares here which of them it answers and what it
shows. A kind with no entry fails ``tests/test_purposes.py``, so a new kind arrives with a purpose
or not at all.

The frontend keeps the same registry for what it draws
(``turbotab/frontend/src/components/stage/purposes.ts``): the same question ids, and the same
entry, word for word, for every view kind and coach anchor kind. It adds the pictures it composes
from these views, the stage's other elements and the Record's components. The test checks that the
two halves agree.

The closed vocabulary of views (BLUEPRINT §11 rule 2) is why this scales to a myriad of decisions:
each kind's purpose is audited once and holds for every decision that uses it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

# The five questions, in BLUEPRINT §11.2's order. The ids are the contract purposes.ts mirrors.
QuestionId = Literal["what", "change", "matters", "data_ok", "provenance"]

QUESTIONS: dict[QuestionId, str] = {
    "what": "What is this choice?",
    "change": "What will it change in my data or my model?",
    "matters": "Why does that matter for my result?",
    "data_ok": "Is my data okay?",
    "provenance": "What did I decide, and can a reviewer reproduce it?",
}

# An entry's own sentence: what it shows, on the user's data (at most this many words).
ANSWER_WORDS = 16


@dataclass(frozen=True)
class Purpose:
    question: QuestionId
    answer: str


# Every ``ConsequenceView`` kind (turbotab/core/consequences.py).
VIEW_PURPOSES: dict[str, Purpose] = {
    "row_flow": Purpose("change", "which rows stay in the analysis, step by step, and how many each step removes"),
    "lineage": Purpose("change", "which columns enter the model, and what each one becomes on the way"),
    "table_focus": Purpose("change", "what the choice writes into real rows, cell by cell"),
    "distribution": Purpose("change", "how one column's values move, with the cut-offs marked on the axis"),
    "relationship": Purpose("matters", "how two columns move together before and after, which is what the model sees"),
}

# Every ``CoachAnchor`` kind: what a note pinned there points out.
COACH_ANCHOR_PURPOSES: dict[str, Purpose] = {
    "column": Purpose("data_ok", "a fact about one column of this table the picture cannot say alone"),
    "range": Purpose("data_ok", "how many rows sit in a stretch of the axis, and what that likely means"),
    "points": Purpose("matters", "which rows in the picture drive what the choice does"),
    "step": Purpose("matters", "what one step of the row flow costs, and whom it removes"),
}

# The parts every preview carries beside its views (``consequences.PreviewResult`` and a view's own
# title, caption, storyboard and marks), so the pedagogy reviewer reads one table for the stage.
PREVIEW_PARTS: dict[str, Purpose] = {
    "title": Purpose("what", "names what the picture shows"),
    "caption": Purpose("change", "the one fact the picture shows, in numbers from the user's table"),
    "story": Purpose("change", "the method's own real steps between before and after, played by the flip"),
    "marks": Purpose("change", "where the choice cuts or flags values on the axis"),
    "basis": Purpose("provenance", "which rows the preview read, and that held-out rows stayed sealed"),
    "note": Purpose("what", "what the choice does when there is no picture of it to draw"),
    "caution": Purpose("data_ok", "a concern with this choice, beside the control that resolves it"),
}


def view_kinds() -> set[str]:
    """Every view kind the consequence planner can send (from its discriminated union)."""
    from turbotab.core.consequences import ConsequenceView

    union = get_args(get_args(ConsequenceView)[0])
    return {kind for model in union for kind in get_args(model.model_fields["kind"].annotation)}


def anchor_kinds() -> set[str]:
    """Every kind of coach anchor (from ``CoachAnchor.kind``)."""
    from turbotab.core.consequences import CoachAnchor

    return set(get_args(CoachAnchor.model_fields["kind"].annotation))


def purpose_of(kind: str) -> Purpose | None:
    """The purpose of a view kind or a coach anchor kind, or None (the test's failure)."""
    return VIEW_PURPOSES.get(kind) or COACH_ANCHOR_PURPOSES.get(kind)


__all__ = ["ANSWER_WORDS", "COACH_ANCHOR_PURPOSES", "PREVIEW_PARTS", "Purpose", "QUESTIONS",
           "QuestionId", "VIEW_PURPOSES", "anchor_kinds", "purpose_of", "view_kinds"]
