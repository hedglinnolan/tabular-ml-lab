"""Recognition's leash (BLUEPRINT §14): a recognizer may be wrong; a wrong recognition may never
silently change a number.

The rules live in the readings ledger (:mod:`turbotab.core.readings`, BLUEPRINT §14.1), which
holds every interpretation the engine makes about the data and the one rule for when a
number-changing consumer may read it. This module keeps the role-reading names the earlier code
and records used; each is the ledger's own.
"""
from __future__ import annotations

from turbotab.core.readings import (  # noqa: F401 - the ledger's role readings, under their old names
    ATTENTION,
    attention_columns,
    confirm_exits,
    is_settled,
    proposals_of,
    rode_along,
    role_words,
    settled_columns,
    unsettled,
    unsettled_message,
)

__all__ = [
    "ATTENTION", "attention_columns", "confirm_exits", "is_settled", "proposals_of", "rode_along",
    "role_words", "settled_columns", "unsettled", "unsettled_message",
]
