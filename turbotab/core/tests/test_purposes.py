"""The purpose registry (BLUEPRINT §11.2): a view or coach anchor kind with no declared purpose fails."""
from __future__ import annotations

import typing

from turbotab.core import consequences, purposes


def _literal_kinds(model: type) -> set[str]:
    field = model.model_fields["kind"]
    return set(typing.get_args(field.annotation))


def _view_kinds() -> set[str]:
    union = typing.get_args(consequences.ConsequenceView)[0]
    return set().union(*(_literal_kinds(m) for m in typing.get_args(union)))


def test_every_consequence_view_kind_declares_its_purpose():
    kinds = _view_kinds()
    assert kinds  # a vacuous sweep would pass anything
    missing = sorted(kinds - set(purposes.VIEW_PURPOSES))
    assert not missing, f"view kinds with no purpose: {missing} (add them to turbotab/core/purposes.py)"
    extra = sorted(set(purposes.VIEW_PURPOSES) - kinds)
    assert not extra, f"purposes for view kinds the server never sends: {extra}"


def test_every_coach_anchor_kind_declares_its_purpose():
    kinds = _literal_kinds(consequences.CoachAnchor)
    assert kinds == set(purposes.COACH_ANCHOR_PURPOSES)


def test_each_purpose_names_one_of_the_five_questions_in_a_short_sentence():
    assert len(purposes.QUESTIONS) == 5
    for kind, p in {**purposes.VIEW_PURPOSES, **purposes.COACH_ANCHOR_PURPOSES}.items():
        assert p.question in purposes.QUESTIONS, kind
        assert 0 < len(p.answer.split()) <= purposes.ANSWER_WORDS, kind
        assert purposes.purpose_of(kind) is p
