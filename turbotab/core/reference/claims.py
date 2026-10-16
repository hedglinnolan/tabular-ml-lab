"""The claims reference and the claims' history, generated from the claims registry
(CLAIMS_REGISTRY.md §5, §6).

    python -m turbotab.core.reference.claims          # writes the reference, appends the history
    python -m turbotab.core.reference.claims --check  # exits 1 when either is stale; writes nothing

writes ``docs/turbotab-next/reference/CLAIMS_REFERENCE.md``: every claim the registry
(``turbotab.core.claims``) holds, by status (built; ruled, not built; retracted), each with its
sentences in both registers, its parameters, its regime, its decision, its parts and their grades,
its tests and their truth, the code that enforces it, its version and its content hash.

``turbotab/core/claims/history.jsonl`` keeps every version of every claim the generator has
written, oldest first, one JSON object a line (``id``, ``version``, ``hash``, ``claim``). When a
claim's content hash changes, its old version stays where it is and the new one is appended after
it, so an old project's claims-in-force manifest resolves its hash to the words and grade it ran
under (§5). ``--check`` fails when the reference differs from the registry's, or when the history's
newest version of a claim is not the one in force.

The text is a pure function of the registry and the files its status is derived from: nothing
reads the clock, and claims are listed by id. ``turbotab/core/tests/test_claims_reference.py``
regenerates it and fails when the committed copy is stale.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from turbotab.core import claims as registry_module
from turbotab.core.claims import GRADES, Claim, Registry, content, content_hash, fill

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "docs" / "turbotab-next" / "reference" / "CLAIMS_REFERENCE.md"
HISTORY = REPO / "turbotab" / "core" / "claims" / "history.jsonl"
COMMAND = "python -m turbotab.core.reference.claims"
HASH_SHOWN = 12  # hex digits of a content hash the reference prints
SECTIONS = (
    ("built", "Built",
     "Claims whose enforcers resolve and whose tests exist (CLAIMS_REGISTRY.md §6)."),
    ("proposed", "Ruled, not built",
     "Claims whose code or tests do not exist yet. Each is a ruling, or a claim on its way in; "
     "none renders a sentence until it is built, and a doc that describes one as current is "
     "ahead of the code."),
    ("retracted", "Retracted",
     "Claims withdrawn, each with its date and reason. A retired id stays, and is never reused."),
)


def code(text: str) -> str:
    return f"`{text}`"


def _sentence(c: Claim, register: str) -> str:
    template = getattr(c, register)
    return fill(template, c.params) if template else "none"


def _part(p: Any, c: Claim) -> str:
    b = p.basis
    basis = [*(f"cites {code(k)}" for k in b.cites)]
    for label, value in (("where", b.where), ("reason", b.reason), ("theorem", b.theorem),
                         ("derivation", b.derivation)):
        if value:
            basis.append(f"{label}: {value}")
    basis += [f"test {code(t)}" for t in b.tests]
    basis += [f"held out on {h.fixture}: {h.rate:g} over {h.cases} cases" for h in b.heldout]
    grader = p.graded_by or c.graded_by or "not graded"
    return (f"  - {p.name}: *{p.grade}*, graded by {grader}"
            + (f" ({'; '.join(basis)})" if basis else ""))


def claim_section(c: Claim, status: str) -> list[str]:
    """One claim's entry."""
    lines = [f"### {code(c.id)}", "",
             f"*{c.kind}* · weakest part *{c.grade}* · {status} · version {c.version} · "
             f"hash {code(content_hash(c)[:HASH_SHOWN])}", "",
             f"- **Card:** {_sentence(c, 'card')}",
             f"- **Methods:** {_sentence(c, 'methods')}"]
    if c.params:
        lines.append("- **Parameters:** " + "; ".join(
            f"{code(k)} = {p.value} {p.unit}" + (f", read by {code(p.constant)}" if p.constant
                                                 else "")
            for k, p in sorted(c.params.items())))
    regime = c.regime.words or "not stated"
    if c.regime.requires:
        regime += "; predicates " + ", ".join(code(r) for r in c.regime.requires)
    if c.regime.disjoint_from:
        regime += "; disjoint from " + ", ".join(code(r) for r in c.regime.disjoint_from)
    lines.append(f"- **Regime:** {regime}")
    if c.decision or c.options or c.verdict or c.goal:
        what = [f"decision {code(c.decision)}" if c.decision else "",
                "options " + ", ".join(code(o) for o in c.options) if c.options else "",
                f"verdict *{c.verdict}*" if c.verdict else "",
                "goal " + ", ".join(c.goal) if c.goal else ""]
        lines.append("- **Decision:** " + "; ".join(w for w in what if w))
    lines.append("- **Parts:**")
    lines += [_part(p, c) for p in c.parts]
    if c.tests:
        lines.append("- **Tests:** " + "; ".join(
            f"{t.side} {code(t.test)} ({t.truth})" for t in c.tests))
    if c.enforced_by:
        lines.append("- **Enforced by:** " + ", ".join(code(e) for e in c.enforced_by))
    for label, ids in (("Supersedes", c.supersedes), ("Falls back to", (c.fallback,))):
        if any(ids):
            lines.append(f"- **{label}:** " + ", ".join(code(i) for i in ids if i))
    lines.append(f"- **Author:** {c.author}")
    if c.ruled:
        lines.append(f"- **Ruled:** {c.ruled}")
    if c.retracted:
        lines.append(f"- **Retracted:** {c.retracted}")
    lines.append("")
    return lines


def render(reg: Registry) -> str:
    """The claims reference, as Markdown."""
    status = {c.id: reg.status(c.id) for c in reg}
    counts = {s: sum(1 for v in status.values() if v == s) for s, _, _ in SECTIONS}
    lines = [
        "# TurboTab v2 claims reference", "",
        f"*Generated by `{COMMAND}` from the claims registry (`turbotab/core/claims/`, "
        "CLAIMS_REGISTRY.md). Do not edit it by hand: "
        "`turbotab/core/tests/test_claims_reference.py` regenerates it and fails when this copy "
        "is stale. Every version of every claim is kept in "
        "`turbotab/core/claims/history.jsonl`.*", "",
    ]
    if not len(reg):
        lines += ["No claim is registered yet.", ""]
    else:
        lines += [f"It holds {len(reg)} claims: {counts['built']} built, {counts['proposed']} "
                  f"ruled and not built, {counts['retracted']} retracted.", ""]
    lines += [
        "## How to read an entry", "",
        "- **Kind**: a *rule* (a guardrail, refusal, routing or default), a *parameter* (a number "
        "with a unit) or *teaching* (a statement of fact about a method).",
        "- **Grade**: each part of a sentence is graded on its own (" + ", ".join(
            f"*{g}*" for g in GRADES) + ", strongest first), and the sentence shows its weakest "
        "part's. Above judgment, someone other than the author grades.",
        "- **Status** is derived: *built* once every enforcer resolves and every test exists, "
        "*ruled, not built* until then; *retracted* is the only status anyone declares.",
        "- **Card** and **Methods**: the sentence in its two registers, filled from its "
        "parameters.",
        f"- **Hash**: the first {HASH_SHOWN} hex digits of the SHA-256 of the claim's content; a "
        "manifest that names it resolves through the history.", "",
    ]
    for key, title, about in SECTIONS:
        lines += [f"## {title}", "", about, ""]
        chosen = [c for c in reg if status[c.id] == key]
        if not chosen:
            lines += ["None.", ""]
        for c in chosen:
            lines += claim_section(c, title.lower())
    return "\n".join(lines)


def read_history(path: Path = HISTORY) -> list[dict[str, Any]]:
    """The history's versions, oldest first."""
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def pending(reg: Registry, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The versions in force that the history's newest entry for their claim is not."""
    newest = {h["id"]: h["hash"] for h in history}
    return [{"id": c.id, "version": c.version, "hash": content_hash(c), "claim": content(c)}
            for c in reg if newest.get(c.id) != content_hash(c)]


def line(record: dict[str, Any]) -> str:
    return json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"


def main(argv: Sequence[str] | None = None, registry: Registry | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="exit 1 when the reference or the history is stale; write nothing")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--history", type=Path, default=HISTORY)
    args = parser.parse_args(argv)
    reg = registry_module.registry() if registry is None else registry
    text = render(reg)
    todo = pending(reg, read_history(args.history))
    if args.check:
        current = args.out.read_text(encoding="utf-8") if args.out.is_file() else ""
        stale = []
        if current != text:
            stale.append(f"{args.out} is stale: run {COMMAND}")
        if todo:
            stale.append(f"{args.history} lacks the version in force of "
                         f"{', '.join(r['id'] for r in todo)}: run {COMMAND}")
        for message in stale:
            print(message, file=sys.stderr)
        if stale:
            return 1
        print(f"{args.out} and {args.history} are current")
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    args.history.parent.mkdir(parents=True, exist_ok=True)
    with args.history.open("a", encoding="utf-8") as f:
        f.writelines(line(r) for r in todo)
    print(f"wrote {args.out} ({len(text.encode('utf-8')):,} bytes); appended {len(todo)} "
          f"version(s) to {args.history}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
