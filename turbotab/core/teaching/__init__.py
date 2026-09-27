"""Teaching: the words each interview question carries (M1_CONTRACT §5, BLUEPRINT §11).

Four opt-in layers, and nothing past the second is needed to answer:

0. ``question`` + ``one_liner`` — what is asked, and the one sentence needed to answer it;
1. each option's ``consequence`` — what choosing it does (the preview shows it on the user's data);
2. ``why`` — the in-place *why?*, and ``consumer``, who reads the answer;
3. the ``drawer`` — the pack's content, each section with its evidence badge and section reference.

``terms`` define themselves in one sentence (a dotted underline in the client). Word budgets are
enforced by ``turbotab/core/tests/test_word_budgets.py``; drawer bodies may be longer.

The content is in :mod:`turbotab.core.teaching.content`; every claim there cites
``docs/turbotab/research/*`` with the pack's status (SETTLED · CONVENTION · DISPUTED).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, ConfigDict

QuestionKey = Literal[
    "lens", "target", "task", "purpose", "roles", "exclusions", "missing", "split",
    "energy_adjustment", "models", "substitution",
]
QUESTION_KEYS: tuple[str, ...] = (
    "lens", "target", "task", "purpose", "roles", "exclusions", "missing", "split",
    "energy_adjustment", "models", "substitution",
)
EvidenceStatus = Literal["SETTLED", "CONVENTION", "DISPUTED"]

# Words, per field (M1_CONTRACT §5).
BUDGETS: dict[str, int] = {
    "title": 8,
    "question": 14,
    "one_liner": 22,
    "why": 60,
    "consumer": 16,
    "option_label": 4,
    "option_consequence": 16,
    "term_definition": 25,
}


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True,
                              json_schema_serialization_defaults_required=True)


class Evidence(_Model):
    """A pack's badge: where the field stands, and the section that says so."""

    status: EvidenceStatus
    source: str


class TeachingOption(_Model):
    value: str
    label: str
    consequence: str


class TeachingTerm(_Model):
    term: str
    definition: str


class DrawerSection(_Model):
    heading: str
    body: str
    evidence: Evidence | None = None


class Drawer(_Model):
    sections: list[DrawerSection]


class TeachingEntry(_Model):
    key: QuestionKey
    title: str
    question: str
    one_liner: str
    why: str
    consumer: str
    options: list[TeachingOption]
    terms: list[TeachingTerm]
    drawer: Drawer | None
    evidence: Evidence | None


@lru_cache(maxsize=1)
def _entries() -> tuple[TeachingEntry, ...]:
    from turbotab.core.teaching.content import ENTRIES

    built = tuple(TeachingEntry.model_validate(e) for e in ENTRIES)
    keys = [e.key for e in built]
    if tuple(keys) != QUESTION_KEYS:
        raise RuntimeError(f"teaching entries are {keys}, expected {list(QUESTION_KEYS)}")
    return built


def entries() -> list[TeachingEntry]:
    """One entry per question key, in asking order."""
    return list(_entries())


def entry(key: str) -> TeachingEntry:
    for e in _entries():
        if e.key == key:
            return e
    raise KeyError(key)


__all__ = [
    "BUDGETS", "Drawer", "DrawerSection", "Evidence", "QUESTION_KEYS", "QuestionKey",
    "TeachingEntry", "TeachingOption", "TeachingTerm", "entries", "entry",
]
