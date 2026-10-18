"""Decoy claims generated from real candidates (R1_PLAN.md R1-G, item 5).

A candidate is one claim on its way to a grade, as a JSON object: an ``id`` and a ``statement``,
the fields a referee reads beside it (:data:`MERITS_FIELDS`, the claims backtest's), and, where it
is known, its ``author``. Each generator takes one candidate and returns a decoy beside it, or None
where its kind does not apply:

* ``overreach``: the regime widened to :data:`WIDE`, and the statement led by saying so;
* ``constant``: a number the statement states, outside a citation and not a year, moved by a
  factor wherever the statement states it;
* ``direction``: a word turned to its opposite (:data:`OPPOSITES`), wherever it stands;
* ``mechanism``: the reason after "because" replaced by another candidate's;
* ``citation``: the basis replaced by the basis a candidate on another decision gives.

They follow the kinds of the claims backtest's hand-written decoys
(``docs/turbotab-next/notes/claims-backtest/scripts/decoys_ACD.py`` and ``decoys_B.py``), and its
README's call for overstatements. A decoy is false by construction only where its candidate is
true and the change matters, so a promoted decoy goes to adjudication: a moved number can land
inside a source's own range. The backtest's vacuous kind (true, but deciding nothing) needs a
writer and is not generated here.

A decoy keeps its candidate's id as its ``sibling`` and its candidate's ``author``, so the gate
checks its referees against the same author, and says in ``why_false`` what was changed.
"""
from __future__ import annotations

import random
import re
from typing import Any, Callable, Sequence

from turbotab.core.claims import citation_spans

Candidate = dict[str, Any]
# What a referee reads of a candidate (the backtest's build_batches.py), beside a neutral id.
MERITS_FIELDS: tuple[str, ...] = ("statement", "goal", "regime", "decision", "app_action",
                                  "claimed_basis")
WIDE = "any study design, any outcome and either purpose"
_LEADS = ("In any study design and for any outcome, ",
          "Under either purpose and whatever the outcome, ",
          "Whatever the design, the outcome or the purpose, ")
OPPOSITES: tuple[tuple[str, str], ...] = (
    ("above", "below"), ("more", "fewer"), ("larger", "smaller"), ("higher", "lower"),
    ("wider", "narrower"), ("wide", "narrow"), ("increases", "decreases"),
    ("raises", "lowers"), ("overstates", "understates"), ("at least", "at most"),
    ("always", "never"), ("before", "after"))
# A number standing alone, its thousands grouped by commas or not ("10,000"): not inside a word,
# an id ("A-001"), a section ("§3") or a decimal.
_NUMBER = re.compile(r"(?<![\w.,§-])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?!\w|[.,]\d)")
_YEAR = re.compile(r"(?:1[89]|20)\d\d")
_BECAUSE = re.compile(r"\s+because\s+", re.IGNORECASE)
_FACTORS = (0.2, 0.5, 2, 5)


def _decoy(c: Candidate, kind: str, why_false: str, **changed: str) -> Candidate:
    return {"sibling": c["id"], "type": kind,
            **{f: c.get(f, "") or "" for f in MERITS_FIELDS}, **changed,
            "author": c.get("author"), "why_false": why_false}


def _numbers(text: str) -> list[re.Match[str]]:
    """The numbers ``text`` states outside a citation, a year left out."""
    spans = citation_spans(text)
    return [m for m in _NUMBER.finditer(text)
            if not _YEAR.fullmatch(m.group()) and not any(a <= m.start() < b for a, b in spans)]


def _moved(token: str, rng: random.Random) -> str:
    """``token`` moved by one of :data:`_FACTORS`, written as it reads (a share stays a share, and
    grouped thousands stay grouped)."""
    value = float(token.replace(",", ""))
    if "." not in token:
        whole = sorted({int(value * f) for f in _FACTORS} - {int(value), 0}) or [int(value) + 1]
        return ("{:,}" if "," in token else "{}").format(rng.choice(whole))
    moved = [value * f for f in _FACTORS]
    if 0 < value < 1:
        moved = [v for v in moved if 0 < v < 1] or moved
    written = sorted({f"{v:.6g}" for v in moved} - {token, f"{value:.6g}"})
    return rng.choice(written)


def overreach(c: Candidate, rng: random.Random, pool: Sequence[Candidate]) -> Candidate | None:
    regime = (c.get("regime") or "").strip()
    if not regime or regime == WIDE:
        return None
    s = c["statement"]
    if s[:1].isupper() and not s[1:2].isupper():
        s = s[0].lower() + s[1:]
    return _decoy(c, "overreach", statement=rng.choice(_LEADS) + s, regime=WIDE,
                  why_false=f"its candidate's evidence covers {regime}; it says the same of "
                            f"{WIDE}")


def constant(c: Candidate, rng: random.Random, pool: Sequence[Candidate]) -> Candidate | None:
    found = _numbers(c["statement"])
    if not found:
        return None
    old = rng.choice(sorted({m.group() for m in found}))
    new = _moved(old, rng)
    s = c["statement"]
    for m in reversed([m for m in found if m.group() == old]):
        s = s[:m.start()] + new + s[m.end():]
    return _decoy(c, "constant", statement=s,
                  why_false=f"it states {new} where its candidate states {old}")


def _turn(text: str, a: str, b: str) -> str:
    def other(m: re.Match[str]) -> str:
        word = m.group(0)
        turned = b if word.casefold() == a else a
        return turned[0].upper() + turned[1:] if word[0].isupper() else turned
    return re.sub(rf"\b(?:{re.escape(a)}|{re.escape(b)})\b", other, text, flags=re.IGNORECASE)


def direction(c: Candidate, rng: random.Random, pool: Sequence[Candidate]) -> Candidate | None:
    s = c["statement"]
    present = [(a, b) for a, b in OPPOSITES
               if re.search(rf"\b(?:{re.escape(a)}|{re.escape(b)})\b", s, re.IGNORECASE)]
    if not present:
        return None
    a, b = rng.choice(present)
    return _decoy(c, "direction", statement=_turn(s, a, b),
                  why_false=f"it turns {a!r} and {b!r} each to the other")


def mechanism(c: Candidate, rng: random.Random, pool: Sequence[Candidate]) -> Candidate | None:
    m = _BECAUSE.search(c["statement"])
    if m is None:
        return None
    own = c["statement"][m.end():].strip()
    others = []
    for o in sorted(pool, key=lambda x: x["id"]):
        mo = _BECAUSE.search(o["statement"]) if o["id"] != c["id"] else None
        if mo is not None and o["statement"][mo.end():].strip() != own:
            others.append((o, o["statement"][mo.end():]))
    if not others:
        return None
    o, reason = rng.choice(others)
    return _decoy(c, "mechanism", statement=c["statement"][:m.end()] + reason,
                  why_false=f"the reason after 'because' is {o['id']}'s, not its own")


def citation(c: Candidate, rng: random.Random, pool: Sequence[Candidate]) -> Candidate | None:
    basis = (c.get("claimed_basis") or "").strip()
    others = [o for o in sorted(pool, key=lambda x: x["id"])
              if o["id"] != c["id"] and (o.get("claimed_basis") or "").strip()
              and o["claimed_basis"].strip() != basis and o.get("decision") != c.get("decision")]
    if not others:
        return None
    o = rng.choice(others)
    return _decoy(c, "citation", claimed_basis=o["claimed_basis"],
                  why_false=f"its basis is the one {o['id']} gives for "
                            f"{o.get('decision') or 'another decision'}, not its own")


Generator = Callable[[Candidate, random.Random, Sequence[Candidate]], "Candidate | None"]
GENERATORS: dict[str, Generator] = {"overreach": overreach, "constant": constant,
                                    "direction": direction, "mechanism": mechanism,
                                    "citation": citation}
KINDS: tuple[str, ...] = tuple(GENERATORS)


def generate(candidates: Sequence[Candidate], n: int, seed: int) -> list[Candidate]:
    """``n`` decoys from ``candidates``, keyed ``G001`` on: the kinds taken in turn (the next kind
    where one does not apply), each from the candidate used least so far, in an order the seed
    shuffles, and no two decoys alike. Fewer come back where no kind applies to any candidate."""
    rng = random.Random(seed)
    pool = sorted(candidates, key=lambda c: c["id"])
    order = pool[:]
    rng.shuffle(order)
    used: dict[str, int] = {}
    seen: set[tuple[str, str, str]] = set()
    out: list[Candidate] = []
    for i in range(n):
        made = None
        for step in range(len(KINDS)):
            kind = KINDS[(i + step) % len(KINDS)]
            for c in sorted(order, key=lambda c: used.get(c["id"], 0)):
                d = GENERATORS[kind](c, rng, pool)
                if d is not None and (d["statement"], d["regime"], d["claimed_basis"]) not in seen:
                    made = d
                    break
            if made is not None:
                break
        if made is None:
            break
        seen.add((made["statement"], made["regime"], made["claimed_basis"]))
        used[made["sibling"]] = used.get(made["sibling"], 0) + 1
        out.append({"key": f"G{len(out) + 1:03d}", **made})
    return out
