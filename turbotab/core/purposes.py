"""The purpose registry (BLUEPRINT §11.2): every picture the stage can draw answers a question the
user has at that moment.

There are five such questions. Every consequence view kind and every coach anchor kind the server
can send declares which one it answers, and a test fails when a kind arrives without an entry ("new
kinds arrive with a purpose or not at all"). The frontend keeps the same registry for what it draws
(``turbotab/frontend/src/components/stage/purposes.ts``), including the pictures it composes from
these views and the Record's components.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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


# Every ``ConsequenceView`` kind (turbotab/core/consequences.py), closed vocabulary (BLUEPRINT §11
# rule 2): audited once, the purpose holds for every decision that uses the kind.
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
    "step": Purpose("matters", "what one step of the row flow costs, and who it removes"),
}


def purpose_of(kind: str) -> Purpose | None:
    """The purpose of a view kind or a coach anchor kind, or None (the test's failure)."""
    return VIEW_PURPOSES.get(kind) or COACH_ANCHOR_PURPOSES.get(kind)


__all__ = ["ANSWER_WORDS", "COACH_ANCHOR_PURPOSES", "Purpose", "QUESTIONS", "VIEW_PURPOSES",
           "purpose_of"]
