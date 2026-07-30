"""`GUIDED-045` — the assertion whose pass set is broader than its claim.

`test_answering_the_lens_changes_the_recorded_lens.py` named the class in its
own docstring and did not file it:

> Every other frontend assertion in this tree is a text search over
> `index.html`, and a text search cannot tell a page that READS a field from a
> page that merely names it.

Twenty-three tests across six files assert that way. `docs/turbotab/tools/
pageprobe.py` swept them by mutation — gut the controller, gut the stylesheets,
gut the page entirely — and the interesting result was not the ones that use a
text search. It was **three tests that were green against a page emptied to
`<body></body>`**.

## The sharp form

All three assert *"this string does not appear"*. An absence assertion over a
file is **monotonically easier to satisfy as the file loses content**: delete
the thing being guarded and the guard passes harder. `test_no_internal_
placeholder_string_renders` passes perfectly on an empty page, because an empty
page contains no placeholders. `test_nothing_but_the_blocker_borrows_the_
blocker_treatment` passes on a stylesheet with no blocker treatment at all —
"nothing ELSE wears it" is trivially true when nothing wears it.

That is not the usual failure. The five axes `FEATURE_PARITY.md` already
catalogues — space, time, occasion, environment, and the call-site family — all
describe a guard that **does not run**, or runs against the wrong thing. This
one runs, runs against the right thing, and **asserts less than it appears to**.
Its pass set is a superset of its claim, and the gap is invisible because every
run is green and green is what correct looks like.

## The rule this file enforces

**An absence assertion needs a positive control.** Assert the thing you are
searching within is THERE, and the deletion that would otherwise satisfy you
fails instead.

Checked structurally and cheaply, so it runs on every suite run rather than
depending on somebody remembering the slow mutation sweep. `pageprobe.py`
remains the deeper instrument and its trigger is named in `GUIDED-045`: run it
in any loop that edits `web/index.html`.

## What this cannot do

It reads test source and classifies assertions by shape. A test asserting
`x not in page` where `page` is bound from something other than the file will
not be seen, and a positive control that is present but vacuous
(`assert page is not None`) will satisfy it. It is an existence check, like
`ledger.py check` and `test_every_clause_is_tracked` — precise about whether a
control is there, silent about whether it is any good.
"""
from __future__ import annotations

import ast
import os
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HERE = Path(__file__).resolve().parent

# The page-reading names a test binds the file to. Same vocabulary as
# `pageprobe.py`; a name here that is not there is the two lists drifting.
PAGE_NAMES = re.compile(r"_page\(\)|\bPAGE\b|\bBODY\b|index\.html|read_text")

# An assertion is an ABSENCE claim when it can only be made false by content
# being present — `not in`, `== []`, `not re.search`, `not <collection>`.
#
# `== 0` is DELIBERATELY NOT HERE. It matched `out.returncode == 0` in
# `test_the_pages_javascript_parses`, which is a success claim about a
# subprocess and not an absence claim about the page — the classifier
# committing, on its first run, a smaller version of the defect it measures.
ABSENCE = re.compile(r"assert\s+(not\s|.*\bnot in\b|.*==\s*(\[\]|None)\b)")

# A positive control is an assertion that fails on an empty page: a membership
# test, a length floor, or a non-empty collection.
CONTROL = re.compile(r"assert\s+(?!not\b).*(\bin\b|len\(|>\s*\d|\bis not None\b)")


def _page_tests(path: Path):
    src = path.read_text(encoding="utf-8")
    lines = src.split("\n")
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test"):
            continue
        body = "\n".join(lines[node.lineno - 1:node.end_lineno])
        asserts = [l.strip() for l in body.split("\n")
                   if l.strip().startswith("assert")]
        if not asserts or not PAGE_NAMES.search(body):
            continue
        yield node.name, asserts


FILES = sorted(p for p in HERE.glob("test_*.py"))


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_no_page_test_asserts_only_absences(path):
    """The rule, applied to every test in this tree that reads the page.

    A test whose assertions are ALL absence claims passes on an empty file, so
    it is guarding nothing at its strongest and everything at its weakest.
    """
    offenders = []
    for name, asserts in _page_tests(path):
        absences = [a for a in asserts if ABSENCE.match(a)]
        if not absences or len(absences) != len(asserts):
            continue
        if any(CONTROL.match(a) for a in asserts):
            continue
        offenders.append(name)
    assert not offenders, (
        f"{path.name}: these tests assert only that things are ABSENT from the "
        f"page, so they pass hardest on an empty file and guard nothing: "
        f"{offenders}. Assert the page is there first — see `GUIDED-045` and "
        f"`docs/turbotab/tools/pageprobe.py`.")


def test_the_three_repaired_tests_now_fail_on_an_empty_page():
    """The three the sweep found, named.

    A general rule with no named instances is a rule nobody can check they
    fixed. These are the three, and each is asserted to carry the control that
    was added — read out of the source, because the alternative is re-running a
    six-minute mutation sweep to learn one bit.
    """
    repaired = {
        "test_guided_drive.py": [
            ("test_no_internal_placeholder_string_renders", "len(BODY) > 20_000"),
            ("test_nothing_but_the_blocker_borrows_the_blocker_treatment",
             "assert worn"),
        ],
        "test_skeleton.py": [
            ("test_the_frontend_has_no_synthetic_constants_left",
             "len(body) > 20_000"),
        ],
    }
    for filename, cases in repaired.items():
        src = (HERE / filename).read_text(encoding="utf-8")
        for test_name, control in cases:
            start = src.index(f"def {test_name}(")
            body = src[start:src.index("\ndef ", start + 1)]
            assert control in body, (
                f"{filename}::{test_name} lost its positive control "
                f"({control!r}), so it is green on an empty page again")


def test_the_probe_and_this_check_share_one_vocabulary():
    """Principle-locality: two readers of "what counts as reading the page".

    A name in one and not the other means the fast check and the slow sweep
    disagree about which tests are in the class, and the disagreement would
    show up as a test nobody swept.
    """
    probe = (HERE.parent / "docs" / "turbotab" / "tools" / "pageprobe.py"
             ).read_text(encoding="utf-8")
    theirs = re.search(r"READS_PAGE = re\.compile\(r\"([^\"]+)\"\)", probe)
    assert theirs, "pageprobe.py no longer declares READS_PAGE"
    assert theirs.group(1) == PAGE_NAMES.pattern, (
        f"the probe reads the page by {theirs.group(1)!r} and this check by "
        f"{PAGE_NAMES.pattern!r}; the two disagree about which tests are in "
        f"the class")
