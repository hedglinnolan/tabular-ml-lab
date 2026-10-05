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

# The interview's questions, in asking order (M2_CONTRACT §1): what a finding may route to.
# Opening the seal is the Router's last step (M2_CONTRACT §12.1).
QuestionKey = Literal[
    "lens", "orientation", "target", "event", "task", "follow_up", "purpose", "grain",
    "repeat_kind", "unit", "aggregation", "temporal", "roles", "clusters", "survey", "exclusions",
    "missing", "split", "estimand", "adjustment", "energy_adjustment", "causal", "models",
    "substitution", "open_seal",
]
QUESTION_KEYS: tuple[str, ...] = (
    "lens", "orientation", "target", "event", "task", "follow_up", "purpose", "grain",
    "repeat_kind", "unit", "aggregation", "temporal", "roles", "clusters", "survey", "exclusions",
    "missing", "split", "estimand", "adjustment", "energy_adjustment", "causal", "models",
    "substitution", "open_seal",
)
# What is taught, in the sequence's order: every question, plus the one card that is not a
# question — the repairs offered before the outcome.
TeachingKey = Literal[
    "lens", "orientation", "repairs", "target", "event", "task", "follow_up", "purpose", "grain",
    "repeat_kind", "unit", "aggregation", "temporal", "roles", "clusters", "survey", "exclusions",
    "missing", "split", "estimand", "adjustment", "energy_adjustment", "causal", "models",
    "substitution", "open_seal",
]
TEACHING_KEYS: tuple[str, ...] = (
    "lens", "orientation", "repairs", "target", "event", "task", "follow_up", "purpose", "grain",
    "repeat_kind", "unit", "aggregation", "temporal", "roles", "clusters", "survey", "exclusions",
    "missing", "split", "estimand", "adjustment", "energy_adjustment", "causal", "models",
    "substitution", "open_seal",
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

# Words on text the app composes onto a card or the stage from data (M2_CONTRACT §6): the gate in
# ``tests/test_word_budgets.py`` holds every fixture under every lens to these.
COMPOSED_BUDGETS: dict[str, int] = {
    "card_line": 22,  # a data line on a decision card (the proposals' notes, joined as shown)
    "option_reason": 20,  # why an option stands as it does: a refusal's cause, a proposal's label
    "finding_summary": 20,
    "coach": 12,  # one coach note on a view, or the card's one coach line
    "preview_note": 30,  # what a preview says beneath its views
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
    key: TeachingKey
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
    if tuple(keys) != TEACHING_KEYS:
        raise RuntimeError(f"teaching entries are {keys}, expected {list(TEACHING_KEYS)}")
    return built


def entries() -> list[TeachingEntry]:
    """One entry per question key (and the repairs and open-the-seal cards), in asking order."""
    return list(_entries())


def entry(key: str) -> TeachingEntry:
    for e in _entries():
        if e.key == key:
            return e
    raise KeyError(key)


__all__ = [
    "BUDGETS", "COMPOSED_BUDGETS", "Drawer", "DrawerSection", "Evidence", "QUESTION_KEYS",
    "QuestionKey", "TEACHING_KEYS", "TeachingEntry", "TeachingKey", "TeachingOption",
    "TeachingTerm", "entries", "entry",
]
