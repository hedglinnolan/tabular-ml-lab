"""The commands the theorist and the experimentalist run (SWARM_PLAN §9), each reading a record only
through the view (``view.py``) and never the sealed file:

    python -m research.swarm.claim_cycle.tools index <corpus>
    python -m research.swarm.claim_cycle.tools records <corpus> [--run ID ...] [--dataset LABEL]
                                                                [--fields F1,F2]
    python -m research.swarm.claim_cycle.tools proposals <file>
    python -m research.swarm.claim_cycle.tools propose --out <file> '<one proposal as JSON>'

* ``index``: one line per record: its run, its dataset's label, its purpose, its decisions and fits
  counted;
* ``records``: each record's view, or those of the runs ``--run`` names or of the dataset
  ``--dataset`` labels; ``--fields`` keeps the run's id and the fields named (``view.FIELDS``), and
  refuses, with why, a field the view withholds;
* ``proposals``: the proposals file's proposals, each through the validator;
* ``propose``: validates one proposal (``proposal.py``) and appends it to the proposals file, or
  says why it is refused; nothing is written then.

Each prints one JSON object: what it read, or ``refused`` with why (exit status 1). Which agent may
run which is the hook's (``hook.py``): the experimentalist proposes nothing, and neither agent names
a folder or a file but the cycle's pinned corpus folder and proposals file (``pin.py``). The
command line is ``pin.build_parser``'s, so the hook reads a command as the tools do.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Callable, Sequence

from research.swarm.claim_cycle import pin as N
from research.swarm.claim_cycle import proposal as P
from research.swarm.claim_cycle import view as W

COMMANDS = N.COMMANDS


def build_parser() -> argparse.ArgumentParser:
    """The tools' command line, the one the hook reads a command with (``pin.build_parser``)."""
    return N.build_parser()


def _run(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "index":
        return {"records": W.index(W.read(args.corpus))}
    if args.command == "records":
        fields = [f.strip() for f in args.fields.split(",") if f.strip()] if args.fields else None
        if fields is not None:
            W.check_fields(fields)
        vs = W.read(args.corpus)
        if args.run:
            vs = [v for v in vs if v["run"]["id"] in set(args.run)]
        if args.dataset:
            vs = [v for v in vs if v["dataset"]["id"] == args.dataset]
        return {"records": [W.project(v, fields) if fields is not None else v for v in vs]}
    if args.command == "proposals":
        return {"proposals": [p.model_dump(mode="json") for p in P.read_proposals(args.file)]}
    try:
        data = json.loads(args.proposal)
    except ValueError as exc:
        raise P.ProposalRefused(f"The proposal is not JSON ({exc.msg}, at character {exc.pos}); "
                                f"it is one JSON object, in single quotes.") from None
    return {"proposed": P.append(args.out, data).model_dump(mode="json")}


def main(argv: Sequence[str] | None = None, out: Callable[[str], Any] = print) -> int:
    args = build_parser().parse_args(argv)
    try:
        body = _run(args)
    except (W.ReadRefused, P.ProposalRefused) as exc:
        out(json.dumps({"refused": str(exc)}, ensure_ascii=False))
        return 1
    out(json.dumps(body, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
