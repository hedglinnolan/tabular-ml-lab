"""The app's voice: the sentence each decision record carries (M1_CONTRACT.md §2).

A stub so the rows agent's wiring runs: the M1 "voice" agent owns this module and its real
``sentence_for`` replaces this one at the merge. Returning None keeps a record's ``sentence``
unset, which the Record treats as an M0-era record.
"""
from __future__ import annotations

from typing import Any


def sentence_for(decision: Any, state_before: Any, ctx: Any) -> str | None:
    """One publishable sentence for ``decision``; None until the voice agent's module lands."""
    return None
