"""Census the DOM writes in `turbotab/web/index.html`, and refuse to count prose.

Written for `L66` because three separate documents carried three different numbers for the same
quantity, and all three were produced by counting grep lines. The page's own block comments quote
the tokens being counted -- `:3461` writes the L47 figure inside backticks and `:3462` says
"zero `startViewTransition`" -- so a raw line count reports the file's commentary about itself as
if it were code.

`DRIVE-054` said 127 assignments, one reader said 127, another said 126, and the adjudicator's
first pass said `replaceChildren` had three call sites when it has one. This module exists so the
number has one derivation that anybody can re-run:

    python3 docs/turbotab/tools/count_dom_writes.py

Two independent methods are applied to the assignment count and the run fails if they disagree,
because a census with one method is a claim and a census with two that agree is a measurement.

The rule for exclusion is deliberately narrow: a line is prose only when the match sits inside a
backticked fragment. That is the form this file's comments actually use, and a broader rule (drop
every line inside a block comment) would need a JavaScript parser to be correct and would silently
change the number if a comment style changed.
"""

from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
PAGE = ROOT / "turbotab" / "web" / "index.html"

# The match sits inside a backticked fragment -- the page's comments quote these tokens.
IN_BACKTICKS = re.compile(r"`[^`]*innerHTML\s*=")
ASSIGN = re.compile(r"innerHTML\s*\+?=")
DOTTED = re.compile(r"\.innerHTML\s*\+?=")
# An assignment whose value begins on the NEXT line. A per-line matcher cannot see these.
LINE_BROKEN = re.compile(r"innerHTML\s*=\s*$")

NAMED = ("replaceChildren", "insertBefore", "startViewTransition", "appendChild", "removeChild")
SCROLL = ("scrollIntoView", "window.scrollTo", ".scrollTop", "scrollBy(")


def census(src: str) -> dict:
    lines = src.split("\n")
    raw = [(i + 1, ln) for i, ln in enumerate(lines) if ASSIGN.search(ln)]
    prose = [(n, ln) for n, ln in raw if IN_BACKTICKS.search(ln)]
    real = [(n, ln) for n, ln in raw if not IN_BACKTICKS.search(ln)]

    # Method two: count dot-prefixed occurrences across the whole source. A backticked
    # `innerHTML =` carries no leading dot, so this reaches the same set by a different route.
    dotted = DOTTED.findall(src)

    if len(real) != len(dotted):
        raise SystemExit(
            "the two methods disagree: %d line-based vs %d occurrence-based. "
            "Do not quote either number until this is resolved." % (len(real), len(dotted))
        )

    return {
        "raw": len(raw),
        "prose": prose,
        "real": len(real),
        "compound": len([d for d in dotted if "+" in d]),
        "line_broken": [n for n, ln in real if LINE_BROKEN.search(ln)],
    }


def call_sites(src: str, name: str) -> tuple[list, list]:
    """Split hits into code and comment by opening the line, not by counting it.

    A token inside backticks is the page describing itself. `startViewTransition` appears twice
    and both are comments asserting the app has none, which is exactly the trap that made an
    earlier count say two call sites where there are zero.
    """
    code, comment = [], []
    for i, ln in enumerate(src.split("\n")):
        if name not in ln:
            continue
        stripped = ln.strip()
        quoted = re.search(r"`[^`]*" + re.escape(name), ln)
        if quoted or stripped.startswith(("*", "/*", "//")):
            comment.append((i + 1, stripped))
        else:
            code.append((i + 1, stripped))
    return code, comment


def main() -> int:
    src = PAGE.read_text(encoding="utf-8")
    c = census(src)

    print("turbotab/web/index.html -- %d lines" % len(src.split("\n")))
    print()
    print("innerHTML assignments")
    print("  raw lines matching        %4d" % c["raw"])
    print("  of those, prose           %4d" % len(c["prose"]))
    for n, ln in c["prose"]:
        print("      :%d  %s" % (n, ln.strip()[:96]))
    print("  REAL ASSIGNMENTS          %4d   (two methods agree)" % c["real"])
    print("  compound (+=)             %4d" % c["compound"])
    print("  value begins next line    %4d   %s" % (len(c["line_broken"]), c["line_broken"]))
    print()
    for name in NAMED:
        code, comment = call_sites(src, name)
        print("%-20s call sites %3d   comments %3d" % (name, len(code), len(comment)))
        for n, ln in code[:6]:
            print("      :%d  %s" % (n, ln[:96]))
    print()
    for name in SCROLL:
        code, comment = call_sites(src, name)
        print("%-20s call sites %3d   comments %3d" % (name, len(code), len(comment)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
