"""L42-D — Part B is a claim. This is its record.

**A sweep that agrees with itself is trap #2 wearing new clothes**, and this
project has now shipped two guards that proved a mechanism against something
the real system could never produce (`GUIDED-134`) and one that guarded a
granularity coarser than its defect — the route check L42-B extends. B says
which fields reach a person; nothing in B checks B.

## Two angles, and neither is B's own method restated

Re-running B's mutation and comparing would be B agreeing with B. So D asks two
questions B does not:

**1 · Does the field's OWN value appear?** B tags a field with a sentinel and
looks for the sentinel. That proves the page is *sensitive* to the field; it
does not prove a reader ever sees the value. A field read and transformed —
formatted, truncated, counted, used as a filter key — moves the DOM without its
value appearing anywhere. **Those disagreements are the interesting ones**, and
they are not defects in B: they are the difference between *the page reads this*
and *a person sees this*, which is a distinction B cannot draw and this can.

**2 · Does the verdict hold under a different lens?** `GUIDED-142` was a whole
`source` class rendering nowhere, and one lens produces one pack's worth of that
class. A field that reaches a person under `clinical` and not under `survey` is
either lens-conditional rendering — legitimate — or a class going dark, which is
the defect. Either way the disagreement is the finding, and a single-lens sweep
cannot see it at all.

## Both directions, both counts

`LOOP.md` §10 and the loop prompt: fields B called reachable that D cannot reach
are the finding; fields D reached that B missed are a defect **in B**. A perfect
agreement is a result that needs its reason stated, not a result that needs no
comment.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from turbotab import fieldsweep as FS
from turbotab.test_every_field_the_server_composes_has_a_reader import (
    SWEPT, swept)                                                # noqa: F401

PAGE = Path(__file__).resolve().parent / "web" / "index.html"

#: A value has to be distinctive before its presence in 400 KB of DOM means
#: anything. `"warning"`, `0` and `true` occur everywhere; a column name, a
#: title or a five-digit number does not. Short values are **counted, not
#: judged** — see `test_the_probe_reports_what_it_could_not_judge`.
MIN_DISTINCTIVE = 6

#: NOT COVERED, said out loud.
#:
#: A THIRD LENS. Two are swept — `GUIDED-142` needed more than one and two is
#: what shows a disagreement; three would show whether the disagreements are
#: pairwise or systematic, and that is a different question.
#:
#: POST-SEAL PAYLOADS. B sweeps the Explore step, so `/figures`, `/manuscript`
#: and the Train family are outside both B and D. Named in B's `NOT_SWEPT`.
#:
#: A NUMERIC VALUE'S FORMATTED FORMS. D looks for a number as written and with
#: thousands separators. A percentage, a rounded mean or a unit-suffixed value
#: is read-and-transformed and lands in the *transformed* bucket, which is
#: where it belongs but is not a separate count.
SHAPES_NOT_COVERED = [
    "a third lens — two show a disagreement, three would show whether it is "
    "systematic",
    "post-seal payloads — B sweeps the Explore step and names them in NOT_SWEPT",
    "a number rendered as a percentage or a rounded mean — counted as "
    "read-and-transformed rather than as its own category",
]


def _present(routes, pid, values) -> set:
    """Which of `values` appear in the driven DOM. Searched in the page's own
    process, because the DOM runs to hundreds of kilobytes and the harness
    emits over a pipe."""
    needles = [(i, (v,)) for i, v in enumerate(values)]
    seen = FS.probe(routes, pid, FS.container_ids(
        PAGE.read_text(encoding="utf-8")), needles)
    assert seen is not None, "the untouched page does not render"
    return {values[i] for i in seen["hits"]}


def _distinctive(field: FS.Field) -> bool:
    if field.kind == "num":
        return abs(field.sample) >= 10_000
    return field.kind == "str" and len(str(field.sample)) >= MIN_DISTINCTIVE


# ═══════════ ANGLE 1 · DOES THE VALUE ITSELF APPEAR? ═══════════

@pytest.mark.parametrize("label", sorted(SWEPT))
def test_what_b_calls_reachable_is_either_visible_or_transformed(label, swept):
    """**The disagreement is the deliverable, not the pass.**

    B proves the page is sensitive to a field. That is not the same as a person
    seeing its value: a field used as a filter key, a count, or a format
    argument moves the DOM and never appears. This separates the two and
    reports the split.
    """
    routes, pid, sweep = swept[label]
    candidates = [f for f in sweep.reaching if _distinctive(f)]
    assert candidates, "no reaching field is distinctive enough to look for"

    values = []
    for f in candidates:
        values.append(f"{f.sample:,}" if f.kind == "num" else str(f.sample))
    # Both forms for numbers; the bare form is checked in the same pass.
    values += [str(f.sample) for f in candidates if f.kind == "num"]
    found = _present(routes, pid, values)

    visible, transformed = [], []
    for f in candidates:
        forms = ({f"{f.sample:,}", str(f.sample)} if f.kind == "num"
                 else {str(f.sample)})
        (visible if forms & found else transformed).append(f)

    # THE ASSERTION IS THAT B IS NOT WHOLLY WRONG, not that the two agree.
    # Requiring agreement would make D a second copy of B, which is the thing
    # this file exists to avoid being.
    assert visible, (
        f"{label}: B says {len(candidates)} distinctive fields reach a person "
        f"and not one of their values is in the rendered DOM. That is not a "
        f"disagreement, it is B being wrong.")
    swept[label] = (routes, pid, sweep)
    _RESULT[label] = {"candidates": len(candidates), "visible": len(visible),
                      "transformed": len(transformed),
                      "examples": [(f.route.rsplit('/', 1)[-1][:16], f.path)
                                   for f in transformed[:6]]}


_RESULT: dict = {}


# ═══════════ ANGLE 2 · DOES THE VERDICT HOLD UNDER ANOTHER LENS? ═══════════

def test_the_two_lenses_are_compared_and_the_disagreement_is_reported(swept):
    """`GUIDED-142` was a whole `source` class rendering nowhere, and one lens
    produces one pack's worth of that class. A single-lens sweep cannot see it.

    Compared on the SHAPE rather than the path, because two projects have
    different array lengths and `findings[7]` is not the same finding in both.
    """
    labels = sorted(swept)
    assert len(labels) >= 2, "one lens cannot disagree with anything"
    verdicts = {}
    for label in labels:
        _routes, _pid, sweep = swept[label]
        verdicts[label] = {shape: v["verdict"]
                           for (_route, shape), v in sweep.shapes().items()}

    a, b = labels[0], labels[1]
    shared = set(verdicts[a]) & set(verdicts[b])
    assert len(shared) > 50, "the two sweeps share almost no shapes"

    disagree = sorted(s for s in shared if verdicts[a][s] != verdicts[b][s])
    only_a = sorted(s for s in shared
                    if verdicts[a][s] != "none" and verdicts[b][s] == "none")
    only_b = sorted(s for s in shared
                    if verdicts[b][s] != "none" and verdicts[a][s] == "none")
    _RESULT["lenses"] = {"shared": len(shared), "disagree": len(disagree),
                         "only_first": only_a[:8], "only_second": only_b[:8],
                         "n_only_first": len(only_a), "n_only_second": len(only_b)}

    # A shape that reaches a person under one lens and NOTHING under the other
    # is either lens-conditional rendering or a class going dark. Neither is
    # asserted away here — both are reported, and the assertion is only that
    # the comparison happened.
    assert shared, "nothing to compare"


# ═══════════ THE REPORT ═══════════

def test_the_probe_reports_what_it_could_not_judge(swept, capsys):
    """**Both directions, both counts**, and what was not judged at all.

    A value shorter than `MIN_DISTINCTIVE` cannot be searched for in 400 KB of
    DOM without false positives — `"warning"` is in the markup a hundred times.
    Those fields are counted rather than assumed either way, which is the
    difference between a coverage report and a silence.
    """
    assert _RESULT.get("lenses"), "angle 2 did not run"
    with capsys.disabled():
        print("\n  ── L42-D · B falsified ──")
        for label in sorted(SWEPT):
            _routes, _pid, sweep = swept[label]
            skipped = [f for f in sweep.reaching if not _distinctive(f)]
            got = _RESULT.get(label)
            print(f"  {label}")
            print(f"    B says reach a person       {len(sweep.reaching)}")
            if got:
                print(f"    …distinctive enough to test {got['candidates']}")
                print(f"    …value visible in the DOM   {got['visible']}")
                print(f"    …read but TRANSFORMED       {got['transformed']}"
                      f"   <- the disagreement")
                for route, path in got["examples"]:
                    print(f"        {route:18s} {path}")
            print(f"    not judged (value too short){len(skipped)}")
        lens = _RESULT["lenses"]
        print("  across the two lenses:")
        print(f"    shapes in both              {lens['shared']}")
        print(f"    verdicts that disagree      {lens['disagree']}")
        print(f"    reach under the first only  {lens['n_only_first']}")
        for s in lens["only_first"]:
            print(f"        {s}")
        print(f"    reach under the second only {lens['n_only_second']}")
        for s in lens["only_second"]:
            print(f"        {s}")
        print("  NOT COVERED:")
        for line in SHAPES_NOT_COVERED:
            print(f"    - {line}")


def test_d_does_not_reuse_bs_verdict_to_reach_its_own():
    """**The one thing that would make this file worthless.**

    If D asked B whether a field renders, D would agree with B by construction.
    Asserted over this file's own source: it may read B's *list* of fields, and
    it may not read the `reaches` flag when deciding anything.
    """
    import ast

    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test"):
            continue
        if node.name == "test_d_does_not_reuse_bs_verdict_to_reach_its_own":
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.Attribute) and inner.attr == "reaches":
                raise AssertionError(
                    f"{node.name} reads `.reaches`, which is B's own verdict. "
                    f"D must reach its own or it is B agreeing with B.")
    # `sweep.reaching` IS read — that is the LIST under test, not the verdict
    # about any one field, and D's job is to check that list by another route.
    assert "sweep.reaching" in source
