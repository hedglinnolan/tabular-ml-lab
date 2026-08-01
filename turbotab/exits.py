"""The way out of a refusal, in a form a client can act on.

`GUIDED-072`, found by the adjudicator hitting it. A 409 carries
`exits=[revise, attest]`, the attest exit is labeled *"My answer is right — the
data really is like this"*, and **nothing in the payload says what to send.**
They read the exit, sent what it described, and got a second 409 — because the
key is `acknowledge_contradiction` and the exit object never said so.

**And the keys are not even uniform.** The lens and grain contradictions read
`acknowledge_contradiction`; the missingness and purpose blockers read
`acknowledge_signal_loss`. Four exits, two keys, and `acknowledge_blocker`
beside them on a different endpoint. So a client receiving `kind="attest"`
cannot construct the retry from the record; it has to hold an out-of-band map
from *which endpoint refused* to *which key unlocks it*.

**That is the coupling `api.py`'s `_disclosures` argues against in its own
words** — the disclosure is served rather than composed in the page precisely
because an interface that writes its own disclosure drifts from what the record
says. An exit whose mechanism lives only in the interface is that drift with
the direction reversed.

And it made a stated invariant untrue in practice: `api.py` says a consequence
resolves or is attested, never a dead end, and the comment above the 409 says
the exits travel WITH the refusal so an interface cannot render the interruption
without also rendering its way out. Both held for a human reader and neither
held for a client.

## The shape of the fix

Every attest exit is built here and carries two fields beyond its prose:

* `payload_key` — the single key the decision handler reads.
* `retry` — a **ready-to-merge payload fragment**. A client merges
  `exit["retry"]["payload"]` into the request that was refused and posts it
  again, to the same endpoint. It needs to know nothing else.

The endpoint is deliberately NOT carried: the client already knows which
request it sent, and duplicating the route in four modules would be a second
copy of the router's own knowledge — the drift this exists to remove, one layer
along.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

# The keys the decision handlers actually read, named once. A key not in this
# table is a key nothing unlocks, and `attest` refuses to build an exit around
# one — which is the whole failure, made impossible rather than documented.
ACKNOWLEDGE_CONTRADICTION = "acknowledge_contradiction"
ACKNOWLEDGE_SIGNAL_LOSS = "acknowledge_signal_loss"
ACKNOWLEDGE_BLOCKER = "acknowledge_blocker"

PAYLOAD_KEYS = (ACKNOWLEDGE_CONTRADICTION, ACKNOWLEDGE_SIGNAL_LOSS,
                ACKNOWLEDGE_BLOCKER)

ATTEST = "attest"
RESOLVE = "resolve"


class ExitError(Exception):
    """An exit that would not open."""


def attest(label: str, detail: str, payload_key: str,
           typed: Optional[str] = None) -> Dict[str, Any]:
    """One attest exit, carrying the key and a payload a client can post.

    `payload_key` is validated against `PAYLOAD_KEYS` rather than accepted as
    written, because a typo here produces exactly the defect this repairs: an
    exit that renders perfectly, describes a real way through, and unlocks
    nothing. A wrong key is not a smaller version of a missing key.
    """
    if payload_key not in PAYLOAD_KEYS:
        raise ExitError(
            f"{payload_key!r} is not a key any decision handler reads "
            f"({list(PAYLOAD_KEYS)}). An exit built around one would render as "
            f"a way through and open nothing, which is `GUIDED-072` exactly.")
    out = {
        "id": ATTEST,
        "kind": ATTEST,
        "label": label,
        "detail": detail,
        # WHAT TO SEND, in the record rather than in the interface.
        "payload_key": payload_key,
        "retry": {"payload": {payload_key: True},
                  "how": (f"Send the same request again with "
                          f"`{payload_key}: true` added to its payload.")},
    }
    if typed:
        # THE SENTENCE A TYPED ACKNOWLEDGMENT REQUIRES, served rather than
        # composed. `web/index.html` wrote its own — *I am keeping X although
        # it may leak the outcome* — which is correct for the leakage blocker
        # and wrong for every other consequence, so a second one would have
        # asked the user to type a sentence about leakage (`GUIDED-076`).
        out["typed"] = typed
        out["retry"]["typed"] = typed
    return out


def is_actionable(exit_row: Dict[str, Any]) -> bool:
    """Whether a client holding only this object could act on it.

    The unifying test for `GUIDED-064` and `GUIDED-072` stated as a predicate:
    a `resolve` exit sends the user back to the question and needs nothing; an
    `attest` exit is a request the client must construct, so it has to carry
    the construction.
    """
    if exit_row.get("kind") != ATTEST:
        return True
    key = exit_row.get("payload_key")
    retry = (exit_row.get("retry") or {}).get("payload") or {}
    return bool(key) and retry.get(key) is True
