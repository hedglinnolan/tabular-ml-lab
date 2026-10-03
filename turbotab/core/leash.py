"""Recognition's leash (BLUEPRINT §14): a recognizer may be wrong; a wrong recognition may never
silently change a number.

Three rules, applied structurally rather than name by name:

1. **"High" is earned by values** (:mod:`turbotab.core.recognizers`, the roles stage). A reading
   the name alone makes is ``medium`` at most and says so.
2. **Number-changing defaults read only settled roles.** A proposal below high carries an
   ``attention`` marker in the roles payload, and the payload lists them (``needs_confirmation``).
   A bulk ``set_roles`` that records such a proposal as proposed records it *unconfirmed* (the
   server fills ``SetRoles.unconfirmed`` from the roles it was shown); it is settled only by its
   own ``confirm_role`` decision, one column per record, or by the user giving it another role.
   The energy card's nutrient list, its energy column and screens, the grouping identifier the
   seal and the intervals read, the survey weight and the "self-reported intake" line read
   settled roles only; a decision that names an unsettled one is refused with the exits that
   confirm it, one at a time.
3. **Ambiguity that touches a number is asked** (the energy unit's day count, a unit known only
   from magnitude, a repeat kind read from a date that does not vary within units).

The gate's criterion follows: a misrecognition proposed at medium or low, its evidence visible and
confirmed individually, is an accepted limitation; one that changes a number unasked is a failure.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

ATTENTION = ("medium", "low")

_ROLE_WORDS = {
    "exposure": "an exposure", "covariate": "a covariate", "energy": "total energy intake",
    "identifier": "the unit's identifier", "cluster": "a cluster of units",
    "design": "part of the survey design", "time": "the time of each row",
    "flag": "a flag on another column", "excluded": "left out of the models",
}


def role_words(role: str | None) -> str:
    return _ROLE_WORDS.get(str(role), str(role))


def proposals_of(artifact: Any) -> list[dict[str, Any]]:
    """The roles stage's proposals from its artifact (a Bundle, a dict, or None)."""
    data = getattr(artifact, "data", artifact)
    if not isinstance(data, Mapping):
        return []
    return [dict(p) for p in data.get("columns") or [] if isinstance(p, Mapping)]


def attention_columns(proposals: Iterable[Mapping[str, Any]]) -> list[str]:
    """The proposals that need their own confirmation: every one below high."""
    return [str(p["column"]) for p in proposals if p.get("confidence") in ATTENTION]


def rode_along(roles: Mapping[str, str], proposals: Iterable[Mapping[str, Any]]) -> list[str]:
    """The attention proposals a ``set_roles`` records exactly as proposed: what a bulk confirm
    carried without the user's own look. A role the user changed is the user's answer."""
    out = []
    for p in proposals:
        column = str(p.get("column"))
        if p.get("confidence") in ATTENTION and column in roles \
                and roles[column] == p.get("proposed"):
            out.append(column)
    return out


def unsettled(state: Any, columns: Iterable[str] | None = None) -> list[str]:
    """The recorded roles a number-changing default may not read yet: carried by a bulk confirm
    below high, and not confirmed one by one since (``confirm_role`` for the same role)."""
    roles = getattr(state, "roles", None) or {}
    waiting = set(getattr(state, "roles_unconfirmed", None) or [])
    confirmed = getattr(state, "role_confirmations", None) or {}
    names = list(roles) if columns is None else [str(c) for c in columns]
    return [c for c in names if c in waiting and c in roles and confirmed.get(c) != roles[c]]


def is_settled(state: Any, column: str | None) -> bool:
    return column is None or not unsettled(state, [column])


def settled_columns(state: Any, artifact: Any = None) -> set[str] | None:
    """The columns whose role a number-changing default may read: the recorded roles less the
    unsettled ones; before any roles are recorded, the proposals the values made high. None when
    neither is known (every caller then reads the roles as given)."""
    roles = getattr(state, "roles", None) or {}
    if roles:
        waiting = set(unsettled(state))
        return {c for c in roles if c not in waiting}
    proposals = proposals_of(artifact)
    if not proposals:
        return None
    return {str(p["column"]) for p in proposals if p.get("confidence") == "high"}


def confirm_exits(state: Any, columns: Sequence[str]) -> list[dict[str, Any]]:
    """One exit per unsettled column: its own confirmation, never one for all of them. Each
    decision is its JSON form, so an exit travels in a refusal and in a stage's artifact alike."""
    from turbotab.core.decisions import ConfirmRole

    roles = getattr(state, "roles", None) or {}
    return [{"label": f"Confirm `{c}` as {role_words(roles.get(c))}",
             "decision": ConfirmRole(column=c, role=roles[c]).model_dump(mode="json")}
            for c in columns if c in roles]


def unsettled_message(columns: Sequence[str], what: str) -> str:
    listed = _listing(columns)
    one = len(columns) == 1
    return (f"{listed} {'was' if one else 'were'} proposed below high confidence and recorded "
            f"with the other roles, not confirmed on {'its' if one else 'their'} own, so "
            f"{'it' if one else 'they'} cannot set {what} yet. Confirm "
            f"{'it' if one else 'each'} after reading why it was proposed, or leave "
            f"{'it' if one else 'them'} out.")


def _listing(columns: Sequence[str]) -> str:
    quoted = [f"`{c}`" for c in columns]
    if len(quoted) == 1:
        return quoted[0]
    return f"{', '.join(quoted[:-1])} and {quoted[-1]}"


__all__ = [
    "ATTENTION", "attention_columns", "confirm_exits", "is_settled", "proposals_of", "rode_along",
    "role_words", "settled_columns", "unsettled", "unsettled_message",
]
