"""L48-A2 — the detector for trap #3b, in the shape a sweep can run.

`AGENT_ONBOARD.md` §07 names the trap and the standing answer has always been
*read the docstring against the body*, by hand, one test at a time. That answer
found `GUIDED-145` once and then found nothing for five loops, and in the loop
that was handed the trap **in writing** a new instance shipped:
`test_a_refusal_lands_beside_the_control_that_caused_it`'s docstring said *"the
press is a `data-feat-add`"* while its body dispatched `data-dismiss`. Every
assertion in it was true. That is the variant reading assertions never finds.

## What this measures, and the definition it publishes

> **A test whose docstring names a control that really exists in the page must
> mention that control somewhere in its body.**

Two halves, and both matter:

- **"really exists in the page"** is the positive control, and it is trap #3's
  rule applied to prose: a docstring token that resolves to nothing is a typo or
  a hypothetical, not a claim about a control. Without this filter the scan
  reported four hits, **all four of them false** — `test_no_participant_
  data_appears_in_a_serialized_project` reads as naming `data-appears-in-a-
  serialized-project` if you hyphenate the function name. A detector for false
  claims that makes them is worse than none.
- **"mention somewhere in the body"** is deliberately weak. It does not check
  that the press *happened*, only that the body is about the thing the docstring
  says it is about. Stronger versions need to know which dispatch reaches which
  handler, which is the thing under test. **Weak and exact beats strong and
  approximate** — this is a syntactic check that cannot be wrong about what it
  claims, and it caught two real instances on its first run.

## What it cannot see, said here

- **A docstring naming a control by prose rather than by attribute.** *"the
  dismiss button"* is invisible to this. Only `data-*` tokens are scanned.
- **A docstring whose consequence verb is not about a control at all** —
  *routes*, *fits*, *draws*. `GUIDED-145`, the original instance, would NOT be
  caught here: it named a strategy, not an attribute. This covers the
  page-control half of the trap and says so.
- **Anything outside `turbotab/`.** The inherited suite is not scanned.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent
PAGE = ROOT / "web" / "index.html"

_ATTR = re.compile(r"data-[a-z][a-z0-9-]*")

#: Docstrings that name a control they deliberately do not touch, each with the
#: reason. An entry here is a DECISION; a test silently absent is a hole — the
#: same distinction `GUIDED-180` draws one layer down for unlisted decision
#: kinds. Empty today, and that is the state the gate below wants.
NAMED_BUT_NOT_PRESSED: Dict[str, str] = {}


def _real_attributes() -> set:
    """Every `data-*` the page actually uses, as the stand-in registry.

    Trap #3: a test that hands a collaborator a name standing for a registered
    object has to show the name resolves. Here the collaborator is the reader.
    """
    page = PAGE.read_text(encoding="utf-8")
    return (set(re.findall(r"\[(data-[a-z0-9-]+)\]", page))
            | set(re.findall(r"(data-[a-z0-9-]+)=", page)))


def scan() -> List[Tuple[str, str, int, List[str], List[str]]]:
    """Every test whose docstring names a real control its body never mentions.

    Returns `(file, test, line, named-but-absent, mentioned)`.
    """
    real = _real_attributes()
    out = []
    for path in sorted(ROOT.glob("test_*.py")):
        source = path.read_text(encoding="utf-8")
        lines = source.splitlines()
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not node.name.startswith("test_"):
                continue
            doc = ast.get_docstring(node)
            if not doc:
                continue
            named = {a for a in _ATTR.findall(doc) if a in real}
            if not named:
                continue
            # The docstring is stripped by its own AST line range, not by
            # string replacement: `get_docstring` dedents, so a `.replace()`
            # against the raw source never matches an indented docstring and
            # the scan silently finds nothing. It did, on the first attempt —
            # a detector reporting zero because of its own bug reads exactly
            # like a codebase with no instances.
            doc_node = node.body[0]
            body = "\n".join(lines[doc_node.end_lineno:node.end_lineno])
            mentioned = set(_ATTR.findall(body))
            absent = sorted(named - mentioned)
            if absent:
                out.append((path.name, node.name, node.lineno, absent,
                            sorted(mentioned & real)))
    return out


def test_the_detector_finds_the_instance_it_was_built_from():
    """The positive control, and it is a synthetic one on purpose.

    A detector whose only evidence is *"it reports zero"* is `GUIDED-119`'s
    shape: green, and possibly measuring nothing. This hands it the exact text
    of the instance that motivated it and requires a report.
    """
    real = _real_attributes()
    assert "data-feat-add" in real and "data-dismiss" in real, (
        "the two attributes this control is written against are gone from the "
        "page, so the fixture no longer stands for anything real")

    source = (
        'def test_x():\n'
        '    """The press is a `data-feat-add`."""\n'
        '    dispatch({"data-dismiss": "f1"})\n')
    lines = source.splitlines()
    node = ast.parse(source).body[0]
    doc = ast.get_docstring(node)
    named = {a for a in _ATTR.findall(doc) if a in real}
    body = "\n".join(lines[node.body[0].end_lineno:node.end_lineno])
    assert named == {"data-feat-add"}
    assert named - set(_ATTR.findall(body)) == {"data-feat-add"}, (
        "the detector no longer reports a docstring naming a control the body "
        "does not mention, which is the only thing it exists to do")


def test_the_name_derived_filter_does_not_manufacture_hits():
    """The negative control, from the four false positives the first scan made.

    An underscore-to-hyphen reading of a test NAME produces plausible-looking
    attribute tokens out of ordinary English. The page-resolution filter is what
    kills them, so it is checked directly rather than assumed.
    """
    real = _real_attributes()
    for invented in ("data-appears-in-a-serialized-project",
                     "data-reports-no-fit-rather-than-a-divergent-iterate",
                     "data-the-scaling-choice-is-not-worth-asking"):
        assert invented not in real, (
            f"{invented} resolves in the page, so the filter that removed it "
            f"from the scan was removing a real control")


def test_no_test_names_a_control_its_body_never_touches(capsys):
    """The gate. Trap #3b, for the page-control half of the trap.

    Reports its own coverage first — a sweep that prints only its failures has
    not said how much it looked at (`LOOP.md` §10).
    """
    real = _real_attributes()
    files = sorted(ROOT.glob("test_*.py"))
    hits = scan()
    unexplained = [h for h in hits if h[1] not in NAMED_BUT_NOT_PRESSED]

    # THE POSITIVE CONTROL, and it is the difference between "nothing is wrong"
    # and "nothing was looked at" (`GUIDED-045`). Zero unexplained hits is the
    # same output on an emptied page, an emptied tree, or a broken parser — so
    # the scan has to show it had something to be wrong about first.
    assert len(real) > 40, (
        f"only {len(real)} `data-*` attributes resolve in the page, so almost "
        f"every docstring token would be filtered out as unreal and the scan "
        f"below would report zero for the wrong reason")
    assert len(files) > 50, f"only {len(files)} test files found in {ROOT}"
    naming = sum(1 for path in files
                 for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
                 if isinstance(node, ast.FunctionDef)
                 and node.name.startswith("test_")
                 and {a for a in _ATTR.findall(ast.get_docstring(node) or "")
                      if a in real})
    # FOUR, and the scope is genuinely thin — five docstrings across 98 files
    # name a page control at all. That is the honest size of this instrument and
    # it is printed below rather than left implicit: the trap is real and common
    # in prose, and only a small fraction of prose names controls by attribute.
    # The floor exists to catch the tree emptying, not to claim breadth.
    assert naming >= 4, (
        f"only {naming} test docstrings name a control that exists in the page, "
        f"so this gate has nothing in scope and passing means nothing")

    with capsys.disabled():
        print("\n  ── L48-A2 · does a test press what its docstring names ──")
        print("  DEFINITION: a docstring naming a `data-*` that RESOLVES IN THE")
        print("  PAGE must mention that attribute somewhere in the body.")
        print(f"  test files scanned                  {len(files)}")
        print(f"  page attributes resolvable          {len(real)}")
        print(f"  docstrings IN SCOPE (name a real one) {naming}")
        print(f"  docstrings naming an unpressed one  {len(hits)}")
        for f, name, line, absent, mentioned in hits:
            print(f"      {f}:{line} {name}")
            print(f"        names {absent}  mentions {mentioned or '(none)'}")
        print(f"  declared exceptions                 "
              f"{len(NAMED_BUT_NOT_PRESSED)}")
        print("  NOT covered: prose names ('the dismiss button'), consequence")
        print("  verbs with no attribute (GUIDED-145's own shape), and every")
        print("  test outside turbotab/.")

    assert not unexplained, (
        "these docstrings name a control the body never touches — trap #3b. "
        "Either mention the control or say what the body does:\n  "
        + "\n  ".join(f"{f}:{ln} {n} names {a}" for f, n, ln, a, _ in unexplained))
