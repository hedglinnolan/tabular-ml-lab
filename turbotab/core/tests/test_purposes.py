"""The purpose registry gate (BLUEPRINT §11.2, M2_CONTRACT §11): a view kind or a coach anchor kind
with no declared purpose fails here, so a new kind arrives with a purpose or not at all. The kinds
are read from the planner's own unions, so a new kind or a removed one fails too."""
from __future__ import annotations

import re
from pathlib import Path
from typing import get_args

from turbotab.core import purposes

FRONTEND = Path(__file__).resolve().parents[2] / "frontend" / "src" / "components" / "stage" / "purposes.ts"


def test_every_consequence_view_kind_declares_its_purpose():
    kinds = purposes.view_kinds()
    assert kinds, "no view kinds found: the gate would pass vacuously"
    missing = sorted(kinds - set(purposes.VIEW_PURPOSES))
    assert not missing, f"view kinds with no purpose: {missing} (add them to turbotab/core/purposes.py)"
    extra = sorted(set(purposes.VIEW_PURPOSES) - kinds)
    assert not extra, f"purposes for view kinds the server never sends: {extra}"


def test_every_coach_anchor_kind_declares_its_purpose():
    kinds = purposes.anchor_kinds()
    assert kinds, "no anchor kinds found: the gate would pass vacuously"
    assert kinds == set(purposes.COACH_ANCHOR_PURPOSES)


def test_every_preview_part_beside_the_views_declares_its_purpose():
    from turbotab.core.consequences import DistributionView, PreviewResult, RowFlowView

    beside = set(PreviewResult.model_fields) - {"kind", "views"}  # basis, note, caution
    on_a_view = {"title", "caption", "story"} & set(RowFlowView.model_fields)
    marks = {"marks"} & set(DistributionView.model_fields)
    assert beside | on_a_view | marks <= set(purposes.PREVIEW_PARTS)


def test_each_purpose_names_one_of_the_five_questions_in_a_short_sentence():
    assert list(purposes.QUESTIONS) == list(get_args(purposes.QuestionId))
    entries = {**purposes.VIEW_PURPOSES, **purposes.COACH_ANCHOR_PURPOSES, **purposes.PREVIEW_PARTS}
    for kind, p in entries.items():
        assert p.question in purposes.QUESTIONS, kind
        assert 0 < len(p.answer.split()) <= purposes.ANSWER_WORDS, kind
    for kind in {**purposes.VIEW_PURPOSES, **purposes.COACH_ANCHOR_PURPOSES}:
        assert purposes.purpose_of(kind) is not None
    assert purposes.purpose_of("embedding") is None  # a later milestone's kind: none until declared


def test_the_frontend_registry_says_the_same_for_every_kind_the_server_sends():
    """purposes.ts mirrors this file: the same question ids, the same entry word for word."""
    text = FRONTEND.read_text(encoding="utf-8")
    ids = re.search(r"export type QuestionId = ([^;]+);", text)
    assert ids and re.findall(r'"([a-z_]+)"', ids.group(1)) == list(purposes.QUESTIONS)
    for kind, p in {**purposes.VIEW_PURPOSES, **purposes.COACH_ANCHOR_PURPOSES}.items():
        found = re.search(rf'\b{kind}: \{{\s*question: "([a-z_]+)",\s*answer:\s*"([^"]+)"', text)
        assert found, f"{kind} has no entry in purposes.ts"
        assert (found.group(1), found.group(2)) == (p.question, p.answer), kind
