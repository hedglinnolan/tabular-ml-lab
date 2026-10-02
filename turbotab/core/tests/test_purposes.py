"""The purpose registry gate (BLUEPRINT §11.2, M2_CONTRACT §11): a view kind or a coach anchor kind
with no declared purpose fails here, so a new kind arrives with a purpose or not at all."""
from __future__ import annotations

from typing import get_args

from turbotab.core import purposes


def test_every_consequence_view_kind_declares_its_purpose():
    kinds = purposes.view_kinds()
    assert kinds, "no view kinds found: the gate would pass vacuously"
    assert kinds - set(purposes.VIEW_KINDS) == set(), "view kinds with no purpose"
    assert set(purposes.VIEW_KINDS) - kinds == set(), "purposes for view kinds that no longer exist"


def test_every_coach_anchor_kind_declares_its_purpose():
    kinds = purposes.anchor_kinds()
    assert kinds, "no anchor kinds found: the gate would pass vacuously"
    assert kinds == set(purposes.COACH_ANCHORS)


def test_every_preview_part_beside_the_views_declares_its_purpose():
    from turbotab.core.consequences import DistributionView, PreviewResult, RowFlowView

    beside = set(PreviewResult.model_fields) - {"kind", "views"}  # basis, note, caution
    on_a_view = {"title", "caption", "story"} & set(RowFlowView.model_fields)
    marks = {"marks"} & set(DistributionView.model_fields)
    assert beside | on_a_view | marks <= set(purposes.PREVIEW_PARTS)


def test_each_purpose_answers_one_of_the_five_questions_in_a_short_sentence():
    assert list(purposes.QUESTIONS) == list(get_args(purposes.Question))
    for name, entry in {**purposes.VIEW_KINDS, **purposes.COACH_ANCHORS,
                        **purposes.PREVIEW_PARTS}.items():
        assert entry.answers and set(entry.answers) <= set(purposes.QUESTIONS), name
        assert len(entry.says.split()) <= purposes.SAYS_WORDS, (name, entry.says)
        assert entry.says.endswith("."), name


def test_a_kind_without_a_purpose_is_found():
    assert purposes.purpose_of("row_flow").answers
    try:
        purposes.purpose_of("embedding")  # a later milestone's kind: no purpose until declared
    except KeyError:
        pass
    else:  # pragma: no cover - the registry must not invent purposes
        raise AssertionError("purpose_of answered for a kind nobody declared")
