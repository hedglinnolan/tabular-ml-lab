"""The decoy queue: decoys generated from real candidates, planted beside them in a grading queue,
and the false-promotion rate of the two-referee gate over the referees' verdicts (R1_PLAN.md R1-G,
item 5; CLAIMS_REGISTRY.md §8).

    python -m research.registry.decoy_queue candidates --out CANDIDATES.jsonl
    python -m research.registry.decoy_queue build --candidates CANDIDATES.jsonl [...] --out DIR \\
        [--decoys N] [--seed S] [--batches B]
    python -m research.registry.decoy_queue score --key DIR/key.json \\
        --verdicts "NAME=GLOB" [--verdicts "NAME=GLOB" ...] [--report REPORT.json]

``candidates`` writes the claims registry's live claims as candidates, one JSON object a line, in
the fields a referee reads (``decoys.MERITS_FIELDS``) with their id, author and origin; any file of
that shape serves, a research swarm's candidates or the claims backtest's inventory among them.

``build`` generates decoys from the candidates (``decoys.generate``: a quarter as many as there are
candidates unless ``--decoys`` says), gives every item a neutral id in an order the seed shuffles,
and deals them into batches of about a dozen, no decoy in its sibling's batch. It writes
``DIR/batches/batch_NN.jsonl``, what a referee sees (the id and the merits fields alone), and
``DIR/key.json``, which item is real and which a decoy, with its kind, sibling, author and why it
is false, sealed by its SHA-256 in ``DIR/key.sha256``. Show a referee its batch, never the key.

``score`` reads the key and each referee's verdicts: ``NAME=GLOB`` names the referee whose
verdicts the files the glob matches hold, one JSON object a line with the queue ``id`` and a
``truth``; a line may give its own ``referee`` and the agent it is a ``fork_of``. A truth of
"true" upholds the item, and any other does not. An item is promoted when it passes the registry's
own two-referee gate (``turbotab.core.claims.referee_gate``): two distinct referees upheld it,
neither its author nor a fork, and none who counts did not. The false-promotion rate is the share
of the false decoys promoted (a vacuous decoy is true, so left out), with its 95% Wilson interval,
and the rule of three's bound where none was promoted; the report also gives each kind's rate,
each referee's own passes, the real candidates promoted and the items short of two referees who
count. A promoted decoy is listed to adjudicate: a generated decoy can be true by accident.

Generalized from the claims backtest's ``build_batches.py`` and ``score.py``
(``docs/turbotab-next/notes/claims-backtest/scripts/``), which read and wrote one session's
scratchpad by a path written into them; this tool reads and writes only the paths it is given. Its
decoys are generated here from the candidates given; it reads no other decoy set.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable, Sequence

from research.registry.decoys import MERITS_FIELDS, generate
from turbotab.core import claims
from turbotab.core.claims import Referee, fill, grader, referee_gate, same_person

SEED = 20261018
SHARE = 0.25  # decoys per real candidate, unless --decoys says
PER_BATCH = 12  # items a batch holds, about (the backtest's 147 in 12)
TRUE_KINDS = ("vacuous",)  # a decoy kind that is true, so no false promotion
Z95 = NormalDist().inv_cdf(0.975)


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                    encoding="utf-8")


# ── the candidates ───────────────────────────────────────────────────────────────────────────

def registry_candidates(reg: claims.Registry | None = None) -> list[dict[str, Any]]:
    """The registry's live claims as candidates: the methods sentence (the card where there is
    none), the regime's words, the decision, the verdict, and each part's grade and basis."""
    reg = claims.registry() if reg is None else reg
    out = []
    for c in reg:
        if c.retracted:
            continue
        basis = []
        for p in c.parts:
            b = p.basis
            said = [*b.cites, b.where, b.reason, b.theorem, b.derivation, *b.tests]
            basis.append(f"{p.name} ({p.grade}): " + "; ".join(x for x in said if x))
        out.append({"id": c.id, "statement": fill(c.methods or c.card, c.params),
                    "goal": " and ".join(c.goal) or "any", "regime": c.regime.words,
                    "decision": c.decision, "app_action": c.verdict,
                    "claimed_basis": " | ".join(basis), "author": c.author, "origin": c.origin})
    return out


# ── the queue ────────────────────────────────────────────────────────────────────────────────

def build(candidates: Sequence[dict[str, Any]], out: Path, decoys: int | None = None,
          seed: int = SEED, batches: int | None = None) -> dict[str, Any]:
    """Write the queue for ``candidates`` under ``out`` and return its key."""
    ids = [c["id"] for c in candidates]
    if len(set(ids)) != len(ids):
        raise ValueError("two candidates share an id")
    if any(not str(c.get("statement") or "").strip() for c in candidates):
        raise ValueError("every candidate states something")
    n = decoys if decoys is not None else max(1, round(SHARE * len(candidates)))
    made = generate(candidates, n, seed)
    items = [dict(orig=c["id"], kind="real", stratum=c.get("stratum", "real"), also=[],
                  sibling=None, author=c.get("author")) for c in candidates]
    items += [dict(orig=d["key"], kind="decoy", stratum="decoy", also=[], sibling=d["sibling"],
                   author=d.get("author")) for d in made]
    order = list(range(len(items)))
    random.Random(seed + 99).shuffle(order)
    for number, i in enumerate(order, 1):
        items[i]["rid"] = f"K-{number:03d}"
    nb = batches or max(1, math.ceil(len(items) / PER_BATCH))
    rng = random.Random(seed + 7)
    shuffled = items[:]
    rng.shuffle(shuffled)
    groups: list[list[dict[str, Any]]] = [[] for _ in range(nb)]
    where: dict[str, int] = {}
    for it in [x for x in shuffled if x["kind"] == "real"]:
        b = min(range(nb), key=lambda k: len(groups[k]))
        groups[b].append(it)
        where[it["orig"]] = b
    for it in [x for x in shuffled if x["kind"] == "decoy"]:
        free = [k for k in range(nb) if k != where.get(it["sibling"])] or list(range(nb))
        b = min(free, key=lambda k: (sum(x["kind"] == "decoy" for x in groups[k]),
                                     len(groups[k]), k))
        groups[b].append(it)
        where[it["orig"]] = b
    for g in groups:
        rng.shuffle(g)
    source = {**{c["id"]: c for c in candidates}, **{d["key"]: d for d in made}}
    for k, g in enumerate(groups, 1):
        write_jsonl(out / "batches" / f"batch_{k:02d}.jsonl",
                    ({"id": it["rid"], **{f: source[it["orig"]].get(f, "") or ""
                                          for f in MERITS_FIELDS}} for it in g))
    key = {"items": [{**{x: it[x] for x in ("rid", "orig", "kind", "stratum", "also", "sibling",
                                            "author")}, "batch": where[it["orig"]] + 1}
                     for it in items],
           "decoys": made, "seed": seed,
           "generated_by": "python -m research.registry.decoy_queue build"}
    out.mkdir(parents=True, exist_ok=True)
    text = json.dumps(key, ensure_ascii=False, indent=1) + "\n"
    (out / "key.json").write_text(text, encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    (out / "key.sha256").write_text(f"{digest}  key.json\n", encoding="utf-8")
    return key


# ── the score ────────────────────────────────────────────────────────────────────────────────

def read_verdicts(specs: Sequence[str]) -> dict[str, list[Referee]]:
    """Each queue id's verdicts, from ``NAME=GLOB`` specs (the module's docstring)."""
    by_id: dict[str, list[Referee]] = defaultdict(list)
    for spec in specs:
        name, sep, pattern = spec.partition("=")
        if not sep or not name.strip() or not pattern.strip():
            raise ValueError(f"a verdicts spec reads NAME=GLOB, not {spec!r}")
        paths = sorted(glob.glob(pattern))
        if not paths:
            raise ValueError(f"no verdicts file matches {pattern!r}")
        for path in paths:
            for r in read_jsonl(path):
                if "id" not in r or "truth" not in r:
                    continue
                by_id[r["id"]].append(Referee(
                    who=str(r.get("referee") or name).strip(),
                    verdict="upheld" if r["truth"] == "true" else "not_upheld",
                    fork_of=str(r.get("fork_of") or "")))
    return dict(by_id)


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """The Wilson score interval for ``k`` of ``n`` (score.py's, at the exact normal quantile)."""
    p, d = k / n, 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, center - half), min(1.0, center + half)


def _name(r: Referee) -> str:
    return grader(r.who) or r.who.strip()


def _counted(votes: Sequence[Referee], author: str | None) -> list[str]:
    """The distinct referees whose verdicts count: not the author, and no fork."""
    names: list[str] = []
    for r in votes:
        name = _name(r)
        if (author is not None and same_person(name, author)) or r.fork_of.strip():
            continue
        if not any(same_person(name, n) for n in names):
            names.append(name)
    return names


def score(key: dict[str, Any], verdicts: dict[str, list[Referee]]) -> dict[str, Any]:
    """The false-promotion report for the queue ``key`` describes."""
    kinds = {d["key"]: d.get("type", "") for d in key.get("decoys", [])}
    items = key["items"]

    def author(it: dict[str, Any]) -> str | None:
        return it.get("author") or None

    def promoted(it: dict[str, Any]) -> bool:
        return not referee_gate(verdicts.get(it["rid"], []), author(it))

    false = [it for it in items if it["kind"] == "decoy" and kinds[it["orig"]] not in TRUE_KINDS]
    true_decoys = [it for it in items if it["kind"] == "decoy" and kinds[it["orig"]] in TRUE_KINDS]
    real = [it for it in items if it["kind"] == "real"]
    passed = [it for it in false if promoted(it)]
    n, k = len(false), len(passed)
    by_type: dict[str, dict[str, int]] = {}
    for it in false:
        row = by_type.setdefault(kinds[it["orig"]], {"decoys": 0, "promoted": 0})
        row["decoys"] += 1
        row["promoted"] += it in passed
    single: dict[str, dict[str, Any]] = {}
    for it in false:
        for r in verdicts.get(it["rid"], []):
            row = single.setdefault(_name(r), {"upheld_false_decoys": [], "verdicts": Counter()})
            row["verdicts"][r.verdict] += 1
            if r.verdict == "upheld":
                row["upheld_false_decoys"].append([it["rid"], it["orig"], kinds[it["orig"]]])
    return {
        "queue": {"real": len(real), "decoys": len(false) + len(true_decoys),
                  "false_decoys": n, "vacuous_decoys": len(true_decoys)},
        "false_promotion": {
            "decoys": n, "promoted": k, "rate": k / n if n else None,
            "wilson_95": list(wilson(k, n)) if n else None,
            "rule_of_three_95": 3 / n if n and k == 0 else None,
            "promoted_ids": [[it["rid"], it["orig"], kinds[it["orig"]], it["sibling"]]
                             for it in passed]},
        "by_type": by_type,
        "single_referee": {name: {"upheld_false_decoys": row["upheld_false_decoys"],
                                  "verdicts": dict(row["verdicts"])}
                           for name, row in sorted(single.items())},
        "vacuous": {"decoys": len(true_decoys),
                    "promoted": sum(promoted(it) for it in true_decoys)},
        "real": {"candidates": len(real), "promoted": sum(promoted(it) for it in real)},
        "missing_verdicts": sorted(it["rid"] for it in items
                                   if len(_counted(verdicts.get(it["rid"], []), author(it))) < 2),
        "authors_recorded": all(author(it) for it in items),
    }


def summary(report: dict[str, Any]) -> list[str]:
    """The report in a few lines, the rate first."""
    q, fp = report["queue"], report["false_promotion"]
    lines = [f"queue: {q['real']} real candidates and {q['decoys']} decoys "
             f"({q['false_decoys']} false, {q['vacuous_decoys']} vacuous)"]
    if fp["decoys"]:
        lo, hi = fp["wilson_95"]
        line = (f"false-promotion rate: {fp['promoted']} of {fp['decoys']} false decoys promoted "
                f"({fp['rate']:.1%}); 95% Wilson interval {lo:.1%} to {hi:.1%}")
        if fp["rule_of_three_95"] is not None:
            line += f"; at most {fp['rule_of_three_95']:.1%} by the rule of three"
        lines.append(line)
    else:
        lines.append("false-promotion rate: the queue holds no false decoy")
    lines.append(f"real candidates promoted: {report['real']['promoted']} of "
                 f"{report['real']['candidates']}")
    if fp["promoted_ids"]:
        lines.append("promoted decoys to adjudicate (a generated decoy can be true): "
                     + ", ".join(f"{rid} ({kind}, from {sib})"
                                 for rid, _, kind, sib in fp["promoted_ids"]))
    missing = report["missing_verdicts"]
    if missing:
        lines.append(f"{len(missing)} item(s) short of two referees who count: "
                     + ", ".join(missing[:12]) + (", ..." if len(missing) > 12 else ""))
    if not report["authors_recorded"]:
        lines.append("the key records no author for some items, so their referees were not "
                     "checked against one")
    return lines


# ── the command line ─────────────────────────────────────────────────────────────────────────

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("candidates", help="write the registry's live claims as candidates")
    c.add_argument("--out", type=Path, required=True)
    b = sub.add_parser("build", help="plant generated decoys beside the candidates in a queue")
    b.add_argument("--candidates", type=Path, nargs="+", required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--decoys", type=int, default=None)
    b.add_argument("--seed", type=int, default=SEED)
    b.add_argument("--batches", type=int, default=None)
    s = sub.add_parser("score", help="the false-promotion rate over the referees' verdicts")
    s.add_argument("--key", type=Path, required=True)
    s.add_argument("--verdicts", action="append", required=True, metavar="NAME=GLOB")
    s.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.command == "candidates":
        rows = registry_candidates()
        write_jsonl(args.out, rows)
        print(f"wrote {len(rows)} candidates to {args.out}")
        return 0
    if args.command == "build":
        rows = [r for path in args.candidates for r in read_jsonl(path)]
        try:
            key = build(rows, args.out, decoys=args.decoys, seed=args.seed, batches=args.batches)
        except ValueError as e:
            print(e, file=sys.stderr)
            return 1
        batches = len({it["batch"] for it in key["items"]})
        kinds = Counter(d["type"] for d in key["decoys"])
        print(f"queued {len(rows)} real candidates and {len(key['decoys'])} decoys "
              f"({', '.join(f'{v} {t}' for t, v in sorted(kinds.items()))}) in {batches} "
              f"batches under {args.out / 'batches'}; the key is {args.out / 'key.json'}, "
              "which no referee sees")
        return 0
    try:
        verdicts = read_verdicts(args.verdicts)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1
    report = score(json.loads(args.key.read_text(encoding="utf-8")), verdicts)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n",
                               encoding="utf-8")
    print("\n".join(summary(report)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
