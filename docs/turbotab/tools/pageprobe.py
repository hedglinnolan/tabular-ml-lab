"""The revert probe, pointed at the frontend. `GUIDED-045`.

## Why this exists

`test_answering_the_lens_changes_the_recorded_lens.py` states the class in its
own docstring and does not file it:

> Every other frontend assertion in this tree is a text search over
> `index.html`, and a text search cannot tell a page that READS a field from a
> page that merely names it.

That is general, it is correct, and twenty-six tests across six files assert
that way. The question a sweep has to answer for each of them is not *"is a text
search bad?"* — sometimes it is exactly right, because the claim really is about
the file — but **is this assertion's pass set broader than its claim?**

## The probe

Three mutations, each destroying a whole category of truth about the page:

* **`script`** — the controller replaced with an empty function body. Every
  behavioral claim is now false.
* **`style`** — every `<style>` block emptied. Every claim about a token, a
  rule, or a treatment is now false.
* **`all`** — style, body and script all emptied. **There is no page.**

`script` and `style` were the first two and they were not enough, which is
itself the finding. They leave the static markup alone, so a test reading the
body survives both for a perfectly good reason — and worse, an assertion of the
form *"this string does not appear"* gets **monotonically easier as the page
loses content**, so gutting the controller makes it pass harder. Three tests in
this tree were green against a page emptied to `<body></body>`, where of course
no placeholder survives, because nothing does.

So the verdict is `all`: **a test whose claim is about the page must go red when
there is no page.** It is the one criterion no absence assertion can satisfy by
accident.

This is deliberately cruder than a per-test hand-written mutation, and the
crudeness is the point. A hand-written mutation is chosen by somebody who
already believes they know what the test checks; these two are chosen by what
they destroy, so they cannot be tuned to a test's assumptions. A test that
survives the deletion of everything it is about has told you something no
carefully-aimed edit would have.

## The honest limit

Surviving both mutations is proof of a broad pass set. **Dying under one is not
proof of a tight one** — a test can go red for the right category and still
assert far less than it claims within it. That is what `H.run` is for, and the
sweep's verdict per test says which of the two it earned.

Usage — from the repository root:

    venv/bin/python docs/turbotab/tools/pageprobe.py            # sweep all
    venv/bin/python docs/turbotab/tools/pageprobe.py --file X   # one file
"""
from __future__ import annotations

import argparse
import ast
import os
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
PAGE = ROOT / "turbotab" / "web" / "index.html"

# The six files the class was named in, plus the one that named it.
FILES = [
    "test_a_finding_with_no_repair_still_offers_something",
    "test_a_hard_question_carries_its_teaching",
    "test_guided_drive",
    "test_question_grammar",
    "test_skeleton",
    "test_the_page_asks_what_the_router_serves",
]

# A test is IN THE CLASS when an assertion of its own reads the page. Merely
# living in a file that reads the page is not enough: two of the first sweep's
# five "survivors" — `test_every_suggested_action_the_engine_can_emit_is_
# classified` and `test_the_three_kinds_are_named_in_the_router_not_only_in_the_
# page` — assert only on engine objects and were counted because the module
# around them has a `PAGE` constant. Reporting them as broad would have been the
# probe committing the defect it measures.
READS_PAGE = re.compile(r"_page\(\)|\bPAGE\b|\bBODY\b|index\.html|read_text")


def _asserts_on_page(body: str) -> bool:
    lines = body.split("\n")
    reads = [i for i, l in enumerate(lines) if READS_PAGE.search(l)]
    if not reads:
        return False
    # The page has to reach an assertion — directly, or through a local the
    # test binds from it.
    names = set()
    for i in reads:
        m = re.match(r"\s*(\w+)\s*=", lines[i])
        if m:
            names.add(m.group(1))
    for line in lines:
        if not line.strip().startswith("assert"):
            continue
        if READS_PAGE.search(line) or any(re.search(rf"\b{n}\b", line) for n in names):
            return True
    # An assertion inside a loop over something derived from the page counts
    # too; a body that reads the page and asserts at all is the honest default,
    # so the narrow reading only excludes tests with no page-derived name in
    # any assertion AND no direct read in one.
    return False


def _mutate(text: str, kind: str) -> str:
    if kind == "script":
        head = text.index("<script>") + len("<script>")
        tail = text.rindex("</script>")
        # A parseable no-op, so `node --check` still passes and the probe
        # measures the ASSERTIONS rather than a syntax error.
        return text[:head] + "\n(function(){ 'use strict'; })();\n" + text[tail:]
    if kind == "style":
        # Every `<style>…</style>` emptied. `re.sub` rather than an index walk:
        # the first version advanced `at` to the CLOSING tag, so the next search
        # found the same block's `<style>` again and eventually ran off the end.
        return re.sub(r"<style>.*?</style>", "<style></style>", text, flags=re.S)
    if kind == "all":
        # THE DECISIVE ONE, and it exists because the first two were not.
        #
        # `script` and `style` leave the static markup alone, so a test reading
        # the body survives both for a perfectly good reason. Worse — and this
        # is the finding — an assertion of the form *"this string does not
        # appear"* gets MONOTONICALLY EASIER as the page loses content, so
        # deleting the controller makes it pass harder.
        #
        # A test whose claim is about the page must go red when there is no
        # page. That is the one criterion no absence assertion can satisfy by
        # accident, and it is what separates "asserts less than it appears to"
        # from "asserts something the page cannot affect".
        head = text[:text.index("<style>")]
        return head + "<style></style>\n<body></body>\n<script>\n(function(){ 'use strict'; })();\n</script>\n"
    raise SystemExit(f"unknown mutation {kind!r}")


def tests_in(name: str):
    path = ROOT / "turbotab" / f"{name}.py"
    src = path.read_text(encoding="utf-8")
    lines = src.split("\n")
    out = []
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test"):
            continue
        body = "\n".join(lines[node.lineno - 1:node.end_lineno])
        if _asserts_on_page(body):
            out.append(node.name)
    return out


def _run(target: str) -> bool:
    """True when the test passes."""
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    p = subprocess.run(
        [sys.executable, "-m", "pytest", target, "-q", "--no-header",
         "-p", "no:cacheprovider", "-p", "no:randomly"],
        cwd=ROOT, capture_output=True, text=True, env=env)
    return p.returncode == 0


def sweep(only: str | None = None):
    original = PAGE.read_text(encoding="utf-8")
    rows = []
    try:
        for kind in ("script", "style", "all"):
            PAGE.write_text(_mutate(original, kind), encoding="utf-8")
            for name in ([only] if only else FILES):
                for test in tests_in(name):
                    target = f"turbotab/{name}.py::{test}"
                    rows.append((kind, target, _run(target)))
    finally:
        PAGE.write_text(original, encoding="utf-8")

    verdicts = {}
    for kind, target, passed in rows:
        verdicts.setdefault(target, {})[kind] = passed

    broad, tight = [], []
    for target, r in sorted(verdicts.items()):
        # SURVIVING `all` is the verdict. A test whose claim is about the page
        # must go red when there is no page; one that does not is asserting
        # something the page's entire content cannot affect.
        if r.get("all"):
            broad.append(target)
        else:
            which = [k for k in ("script", "style", "all") if not r.get(k)]
            tight.append((target, "+".join(which)))

    print(f"{len(verdicts)} text-search frontend tests probed\n")
    print(f"  BOUNDED — red when the page loses what they are about: {len(tight)}")
    for t, which in tight:
        print(f"     red under {which:<18s} {t}")
    print(f"\n  BROADER THAN THE CLAIM — green against an EMPTY page: {len(broad)}")
    for t in broad:
        print(f"     {t}")
    return broad, tight


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--file")
    args = ap.parse_args()
    survivors, _ = sweep(args.file)
    raise SystemExit(1 if survivors else 0)
