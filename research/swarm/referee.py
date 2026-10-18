"""The referee audit's sampler and prompts (DRIVER_PLAN D5; SWARM_PLAN §3).

A referee, an Opus agent with Nolan spot-checking, audits a random sample of the corpus's attempts
and asks of each whether its reason was real and whether the analyst's reading matches the
numbers. This module draws the sample and writes each sample's prompt (``referee_prompt.md``
filled in); it never runs a referee. The referee runs belong to the pilot.

    python -m research.swarm.referee --seed 20261018 --n 20 --out <folder> [research/corpus]

**The sample.** The pool is every attempt (a fit record) that carries a reason, the analyst's
attempts, across the records given, ordered by run id and fit seq; ``n`` of them are drawn without
replacement by Python's ``random.Random(seed)``, so a seed draws the same sample from the same
corpus. ``index.json`` lists the seed and each sample.

**What a prompt shows.** The run's goal and outcome; the records credited with the fit's change
from the fit before (``Attempt.changed``), each with its option (a revert with the record it
withdrew), each record it replaced or restored, and its reason (the record's note), and each record
that left or returned to force credited to none; the fit's numbers as the engine served them
(each family's cross-validated summary of the primary metric under Predict, the exposure's rows
under Estimate); the engine's movement against the fit before, in its own words, or its why; and
the reasons the analyst recorded after the fit completed and before the next fit (its reading).
It never shows whether the dataset is a control.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from string import Template
from typing import Any, Iterable, Sequence

from research.swarm.record import Attempt, Record, validate

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "referee_prompt.md"
NONE = "(none)"


def pool(records: Iterable[Record]) -> list[tuple[str, int]]:
    """Every attempt that carries a reason, as ``(run id, fit seq)``, ordered."""
    return sorted((rec.run.id, a.seq) for rec in records for a in rec.attempts if a.reasons)


def sample(records: Iterable[Record], n: int, seed: int) -> list[tuple[str, int]]:
    """``n`` attempts drawn from :func:`pool` by ``random.Random(seed)`` (all of them when the
    pool holds fewer), in the order drawn."""
    candidates = pool(records)
    return random.Random(seed).sample(candidates, min(n, len(candidates)))


def _option(option: dict[str, Any]) -> str:
    body = {k: v for k, v in option.items() if k != "kind"}
    text = json.dumps(body, ensure_ascii=False, sort_keys=True)
    return text if len(text) <= 400 else text[:399] + "…"


def _changed(rec: Record, attempt: Attempt) -> str:
    by_seq = {d.seq: d for d in rec.decisions}
    by_id = {d.id: d for d in rec.decisions}

    def withdrawn_by(d: Any) -> Any:
        return by_id.get(str(d.option.get("decision_id"))) if d.kind == "revert" else None

    def named(d: Any) -> str:
        target = withdrawn_by(d)
        if target is not None:
            return (f"Record #{d.seq}, `revert` of record #{target.seq}, `{target.kind}` "
                    f"{_option(target.option)}")
        return f"Record #{d.seq}, `{d.kind}` {_option(d.option)}"

    def other(seq: int) -> str:
        d = by_seq[seq]
        return f"record #{d.seq}, `{d.kind}` {_option(d.option)}"

    lines = []
    for s in attempt.changed or []:
        d = by_seq[s]
        target = withdrawn_by(d)
        text = f"- {named(d)}"
        for w in attempt.withdrawn:
            if w.by == s and (target is None or w.seq != target.seq):
                verb = "withdrew" if d.kind == "revert" else "replaced"
                text += f"\n  It {verb} {other(w.seq)}."
        for w in attempt.reinstated:
            if w.by == s:
                text += f"\n  It restored {other(w.seq)}."
        lines.append(f"{text}\n  Reason: {d.reason or '(no reason recorded)'}")
    for shifts, what in ((attempt.withdrawn, "left force"), (attempt.reinstated,
                                                            "returned to force")):
        lines += [f"- {named(by_seq[w.seq])}, {what}, and no record of the log is credited with "
                  f"it." for w in shifts if w.by is None]
    return "\n".join(lines) or NONE


def _numbers(attempt: Attempt) -> str:
    lines = []
    metric = attempt.primary_metric
    for family, summary in attempt.scores.items():
        s = (summary or {}).get(metric) or {}
        if s:
            lines.append(f"- `{family}`: cross-validated {metric} {s.get('estimate')} "
                         f"(interval {s.get('ci_low')} to {s.get('ci_high')}, "
                         f"{len(s.get('folds') or [])} folds)")
    for family, rows in attempt.effects.items():
        for row in rows or []:
            lines.append(f"- `{family}`: `{row.get('feature')}` {row.get('estimate')} "
                         f"(interval {row.get('ci_low')} to {row.get('ci_high')})")
    if not lines and attempt.withheld:
        lines.append(f"- Withheld: {attempt.withheld}")
    return "\n".join(lines) or NONE


def _movement(attempt: Attempt) -> str:
    lines = [f"- `{m.get('family')}`: " + ((m.get("movement") or {}).get("words")
                                           or m.get("why") or NONE)
             for m in attempt.movement]
    if not lines and attempt.movement_why:
        lines.append(f"- {attempt.movement_why}")
    return "\n".join(lines) or NONE


def _reading(rec: Record, attempt: Attempt) -> str:
    after = [a.at for a in rec.attempts if a.seq > attempt.seq]
    end = min(after) if after else None
    lines = [f"- Record #{d.seq}, `{d.kind}`: {d.reason}" for d in rec.decisions
             if d.source == "analyst" and d.reason and d.at > attempt.at
             and (end is None or d.at <= end)]
    return "\n".join(lines) or NONE


def prompt(rec: Record, seq: int, template: str | None = None) -> str:
    """The referee's prompt for fit ``seq`` of ``rec``."""
    attempt = next(a for a in rec.attempts if a.seq == seq)
    text = template if template is not None else TEMPLATE.read_text("utf-8")
    return Template(text).substitute(
        run_id=rec.run.id, design=rec.run.design, purpose=rec.question.purpose,
        target=rec.question.target, attempt_seq=attempt.seq, n_attempts=len(rec.attempts),
        changed=_changed(rec, attempt), numbers=_numbers(attempt),
        movement=_movement(attempt), reading=_reading(rec, attempt))


def write(records: Sequence[Record], n: int, seed: int, out: Path) -> Path:
    """Draw the sample and write each prompt to ``out/<run id>__fit<seq>.md``, with
    ``out/index.json``. Returns the index's path."""
    by_run = {rec.run.id: rec for rec in records}
    out.mkdir(parents=True, exist_ok=True)
    drawn = []
    for run_id, seq in sample(records, n, seed):
        path = out / f"{run_id}__fit{seq}.md"
        path.write_text(prompt(by_run[run_id], seq), "utf-8")
        drawn.append({"run": run_id, "fit": seq, "prompt": path.name})
    index = out / "index.json"
    index.write_text(json.dumps({"seed": seed, "n": n, "pool": len(pool(records)),
                                 "samples": drawn}, indent=1) + "\n", "utf-8")
    return index


def main(argv: Sequence[str] | None = None) -> int:
    from research.swarm.assemble import CORPUS

    parser = argparse.ArgumentParser(prog="python -m research.swarm.referee",
                                     description=__doc__.split("\n\n")[0])
    parser.add_argument("corpus", type=Path, nargs="?", default=CORPUS)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    records = [validate(json.loads(p.read_text("utf-8")))
               for p in sorted(args.corpus.glob("*.json"))]
    print(write(records, args.n, args.seed, args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
