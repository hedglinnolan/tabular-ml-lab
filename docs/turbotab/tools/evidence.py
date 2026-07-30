"""The evidence gate — every pack claim carries a badge and a resolvable source.

`DOMAIN_SCIENCE.md` §01.1. All four research threads independently asked for the
same primitive: **surface the epistemic status of every claim the app makes.**

This is the check that keeps it true. Three assertions, and one deliberate
non-assertion.

## What it checks

1. **Every pack prior carries `evidence`** — a status and a source. A prior with
   a `marker` and no badge states the app's confidence as if it were the
   field's, which is the state the research asked to end.
2. **Every source resolves** — the named file exists under
   `docs/turbotab/research/`, and the named section is a heading in it.
3. **No `[verify-at-build]` number ships as a hard-coded constant.** The
   research threads hit an egress proxy and marked the numbers they could not
   read from primary text. Shipping one as a literal is the single worst failure
   mode a pack has, and the packs say so themselves.

## What it deliberately does not check

**Whether the claim is faithful to the section it names.** A citation that
resolves to the wrong heading passes here, and that is the same honest limit
`ledger.py check` has: it enforces that a `FIXED` row *names* a test and cannot
tell whether the test is any good. Saying so is the difference between a gate
and a reassurance — and the reviewable form of the missing half is already
specified in `DOMAIN_PACKS.md` §06: *here are the default choices and the
methods sentence each produces — which would you object to?*

Usage — from the repository root:

    venv/bin/python docs/turbotab/tools/evidence.py check
"""
from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
RESEARCH = ROOT / "docs" / "turbotab" / "research"
PACKS = ROOT / "turbotab" / "packs.py"

# A heading in a research file, at any level.
_HEADING = re.compile(r"^#{1,6}\s+(.*?)\s*$", re.M)

# `[verify-at-build]` and `[verify-at-build: what]`. The second form names the
# numbers, and those are the ones that must not become literals.
_VERIFY = re.compile(r"\[verify-at-build:?\s*([^\]]*)\]")

# A bare number inside a verify-at-build note. `50%`, `0.8`, `40`.
_NUMBER = re.compile(r"\b(\d+(?:\.\d+)?)\s*%?")


def _sections(path: pathlib.Path) -> set:
    return {m.group(1).strip() for m in _HEADING.finditer(
        path.read_text(encoding="utf-8"))}


def _priors():
    sys.path.insert(0, str(ROOT))
    from turbotab import packs
    for key, pack in packs.PACKS.items():
        for prior in pack.priors:
            yield key, prior


def check() -> int:
    problems = []

    # 1 + 2 · every prior badged, every source resolvable.
    seen = 0
    for pack_key, prior in _priors():
        seen += 1
        evidence = prior.evidence
        if evidence is None:                               # pragma: no cover
            problems.append(f"{pack_key}/{prior.question}: no evidence badge")
            continue
        filename, _, section = evidence.source.partition("#")
        path = ROOT / "docs" / "turbotab" / filename
        if not path.exists():
            problems.append(
                f"{pack_key}/{prior.question}: {filename} does not exist")
            continue
        if section not in _sections(path):
            problems.append(
                f"{pack_key}/{prior.question}: {filename} has no section "
                f"{section!r}")
    if not seen:
        problems.append("no pack priors found at all; the walk is wrong")

    # 3 · no `[verify-at-build]` number as a literal in pack code.
    source = PACKS.read_text(encoding="utf-8")
    # Strings and comments are prose, not constants. Only code lines count —
    # otherwise quoting a number in a `reason` would fail the gate, and the
    # reasons are exactly where a number SHOULD be discussed.
    code = "\n".join(
        line.split("#", 1)[0] for line in source.split("\n")
        if not line.strip().startswith("#"))
    code = re.sub(r'"(?:[^"\\]|\\.)*"', '""', code)
    code = re.sub(r"'(?:[^'\\]|\\.)*'", "''", code)

    unverified = set()
    for path in sorted(RESEARCH.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for m in _VERIFY.finditer(text):
            for n in _NUMBER.findall(m.group(1)):
                unverified.add((path.name, n))
    for filename, number in sorted(unverified):
        # `\b` on both sides so `50` does not match inside `250`.
        if re.search(rf"(?<![\w.]){re.escape(number)}(?![\w.])", code):
            problems.append(
                f"{filename} marks {number} [verify-at-build] and it appears as "
                f"a literal in turbotab/packs.py. A number nobody has read from "
                f"primary text may not ship as a constant.")

    if problems:
        print("EVIDENCE GATE FAILED")
        for p in problems:
            print(f"  ✗ {p}")
        print("\n  Every pack claim carries `evidence=Evidence(status=…, "
              "source='research/FILE.md#Section')`.\n  The gate resolves the "
              "source; it does not check the claim is faithful to it.")
        return 1
    print(f"ok — {seen} pack priors badged, every source resolves, "
          f"{len(unverified)} [verify-at-build] number(s) held out of the code")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] != "check":
        raise SystemExit(f"unknown command {sys.argv[1]!r}; only `check`")
    raise SystemExit(check())
