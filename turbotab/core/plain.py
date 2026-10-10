"""Plain words on a card (RECIPES_AND_TUNING §6.0; V2_DEFINITION_OF_DONE: "Plain words on every
card").

The card speaks plain language and the technical name rides along as a quiet label. Three terms
of the causal vocabulary stay quiet: a card says what the term means, and the term itself appears
only as a label (a term's own name), never in a question, an option, a reason, a refusal or a
preview.

====================  ===========================================  ==================
quiet term            the card says                                a count noun
====================  ===========================================  ==================
exposure              "what you study"                             "study factor"
confounder            "what else could explain the link"           "something that could explain the link"
estimand              "the comparison you want"                    "comparison"
====================  ===========================================  ==================

Where a plain phrase would change the meaning, the term stays on the card and is defined where it
stands, in :data:`DEFINITION_WORDS` words or fewer: ``estimand (the quantity a study estimates)``.
A use with such a parenthetical beside it is a defined term, not a forbidden one.

The methods prose (the voice's sentences, the manuscript, the export and the readings of a
result) keeps the technical register, because a reader of the paper expects it, and a card and a
sentence that share a fact word it twice (``estimand.precision_note(methods=...)``,
``estimand.ACTION_LABELS`` beside ``ACTION_WORDS``). ``tests/test_plain_words.py`` holds the cards
to this list: it scans the card-facing strings and skips the quiet-label fields and the methods
register; ``tests/acceptance/test_plain_words_served.py`` walks the cards a real journey serves.
"""
from __future__ import annotations

import re

#: The terms a card does not use outside a quiet label.
QUIET_TERMS: tuple[str, ...] = ("exposure", "confounder", "estimand")

#: What a card says instead.
PLAIN_WORDS: dict[str, str] = {
    "exposure": "what you study",
    "confounder": "what else could explain the link",
    "estimand": "the comparison you want",
}

#: A term kept on a card for a meaning no plain phrase carries is defined where it stands, in no
#: more than this many words.
DEFINITION_WORDS = 12

FORBIDDEN = re.compile(r"\b(exposures?|confounders?|estimands?)\b", re.IGNORECASE)
_PLACEHOLDER = re.compile(r"\{[^{}]*\}")  # a format field is a machine word, not a card word
_DEFINED = re.compile(r"\b(?:exposures?|confounders?|estimands?)\s*\(([^()]*)\)", re.IGNORECASE)

#: Fields of a served payload that are quiet labels or identifiers, not card words: a term's own
#: name, where a claim comes from, the machine's keys.
QUIET_KEYS = frozenset({
    "term", "quiet", "quiet_label", "source", "sources", "key", "kind", "value", "values", "column",
    "columns", "role", "roles", "id", "decision",
})


def _undefined(match: re.Match[str]) -> str:
    """A quiet term with its definition beside it, in :data:`DEFINITION_WORDS` words or fewer, is
    defined in place and passes; any other use is left for the scan to find."""
    return "" if len(match.group(1).split()) <= DEFINITION_WORDS else match.group(0)


def forbidden_terms(text: str) -> list[str]:
    """The quiet terms a string uses, in order (case folded); empty for a plain card string."""
    text = _DEFINED.sub(_undefined, _PLACEHOLDER.sub("", text or ""))
    return [m.group(0).lower() for m in FORBIDDEN.finditer(text)]


def is_plain(text: str) -> bool:
    return not forbidden_terms(text)
