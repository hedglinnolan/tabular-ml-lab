"""The prior-driven share (SWARM_PLAN §9, the plumbing test's pre-stated outputs): how much of what
the theorist proposed from the corpus it also proposed from the shuffled copy, whose regime and
measurement markers are another dataset's (the purpose and its loss stay each record's own, as its
question does: ``shuffle.py``).

    python -m research.swarm.claim_cycle.overlap --real <proposals.jsonl> --shuffled <proposals.jsonl>

Each file's proposals are read through the validator (``proposal.read_proposals``); a file holding
one the validator refuses is refused, and no share is given. Two proposals **match** where they
name the same decision, the same set of live-marker names in their regions (thresholds and
comparisons ignored, the markers' order too) and the same direction. Proposals of one file that
match one another are one distinct proposal.

**The share** is the share of the real corpus's distinct proposals that match a proposal from the
shuffled copy. Its denominator is the real corpus's, so the planted effect, carried by the shuffled
copy alone, cannot dilute it. With no proposal from the real corpus there is no share, and it is
refused; with none from the copy, it is 0. The report gives both files' counts of proposals and of
distinct ones, the matched keys, the share and a sentence saying it, in numbers interpolated from
these counts.

**It is an upper bound on the prior-driven share (the share of those proposals matched because of
the theorist's prior), not that share itself, while the copy keeps the purpose axis.** The copy
keeps each record's own live markers of the purpose and its loss (``shuffle.restates``), as it
keeps its question and its decisions' materiality, so a proposal whose region names one of them
rests there on the same evidence in both corpora, and its match may come from the data, not the
prior; the share counts every match. So a low share bounds the
prior-driven share, and a high one does not say the proposals came from the prior (SWARM_PLAN §9
reads a high share so, of a copy whose markers are permuted across datasets). The sentence says
so, naming those markers from the vocabulary, wherever the shuffle keeps any.

**One file for both is refused.** ``--real`` and ``--shuffled`` name two cycles' proposals files:
two that are one file (in two spellings, or through a link) are refused, and no share is given.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from research.swarm import vocabulary as V
from research.swarm.claim_cycle import proposal as P

Key = tuple[str, frozenset[str], str]


class OverlapRefused(ValueError):
    """No share can be given, in words."""


def key(p: P.Proposal) -> Key:
    """What two proposals match on: the decision, the set of live-marker names, the direction."""
    return (p.decision, frozenset(b.marker for b in p.region), p.direction)


def _shown(k: Key) -> dict[str, Any]:
    return {"decision": k[0], "markers": sorted(k[1]), "direction": k[2]}


def _order(k: Key) -> tuple[str, list[str], str]:
    return (k[0], sorted(k[1]), k[2])


def kept_by_the_copy() -> list[str]:
    """The live markers the shuffled copy keeps each record's own (``shuffle.restates``: the
    purpose and its loss), in the vocabulary's order."""
    from research.swarm import shuffle as S

    return [n for n in V.NAMES if S.restates({"name": n})]


def bound(kept: Sequence[str]) -> str:
    """The sentence that says the share bounds the prior-driven share (the module docstring),
    naming the markers the copy keeps; none where it keeps none."""
    if not kept:
        return ""
    return (" This share counts every match, so it is an upper bound on the prior-driven share "
            "(the share of those proposals matched because of the theorist's prior), not that "
            "share itself: the copy keeps each record's own live markers of the purpose and its "
            "loss (" + ", ".join(f"`{n}`" for n in kept) + "), as it keeps its question and its "
            "decisions' materiality, so a proposal whose region names one of them rests there on "
            "the same evidence in both corpora, and its match may come from the data. A low share "
            "bounds the prior-driven share; a high one does not say the proposals came from the "
            "prior.")


def overlap(real: Iterable[P.Proposal], shuffled: Iterable[P.Proposal]) -> dict[str, Any]:
    """The share (the module docstring), with the counts it is read from."""
    real, shuffled = list(real), list(shuffled)
    mine, theirs = {key(p) for p in real}, {key(p) for p in shuffled}
    if not mine:
        raise OverlapRefused("The real corpus's cycle proposed nothing, so no share of its "
                             "proposals can be prior-driven; there is no share.")
    matched = sorted(mine & theirs, key=_order)
    share = len(matched) / len(mine)
    return {"real": {"proposals": len(real), "distinct": len(mine)},
            "shuffled": {"proposals": len(shuffled), "distinct": len(theirs)},
            "matched": [_shown(k) for k in matched], "share": share,
            "says": (f"{len(matched)} of the {len(mine)} distinct proposals from the real corpus "
                     f"(matched on the decision, the set of live-marker names and the direction) "
                     f"{'was' if len(matched) == 1 else 'were'} also proposed from the shuffled "
                     f"copy." + bound(kept_by_the_copy()))}


def one_file(real: str | Path, shuffled: str | Path) -> None:
    """Refused where ``real`` and ``shuffled`` are one file: the same path resolved, or, both
    existing, one file through a link (the module docstring)."""
    a, b = Path(real), Path(shuffled)
    same = a.resolve() == b.resolve()
    if not same and a.exists() and b.exists():
        same = os.path.samefile(a, b)
    if same:
        raise OverlapRefused(f"--real and --shuffled name one file ({real}); the share compares "
                             f"two cycles' proposals, each written to its own file, so there is "
                             f"no share.")


def main(argv: Sequence[str] | None = None, out: Callable[[str], Any] = print) -> int:
    parser = argparse.ArgumentParser(prog="python -m research.swarm.claim_cycle.overlap",
                                     description=__doc__.split("\n\n")[0])
    parser.add_argument("--real", required=True, help="the proposals from the corpus")
    parser.add_argument("--shuffled", required=True, help="the proposals from the shuffled copy")
    args = parser.parse_args(argv)
    try:
        one_file(args.real, args.shuffled)
        body = overlap(P.read_proposals(args.real), P.read_proposals(args.shuffled))
    except (P.ProposalRefused, OverlapRefused) as exc:
        out(json.dumps({"refused": str(exc)}, ensure_ascii=False))
        return 1
    out(json.dumps(body, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
