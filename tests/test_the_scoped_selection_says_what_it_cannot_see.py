"""`L56-A1` — the test selector, and specifically the claims it must not make.

The product owner's constraint is that the two-hour sweep is unworkable in the
current loop. The answer is a scoped selection. **The danger is entirely in one
direction**: a selector that returns too few files converts *"I did not look
there"* into *"there was nothing there"*, which is the governing rule's
*assert something false* branch wearing a CLI. Returning too many is merely
slow.

So this file spends almost all of its assertions on the refusals, and only two
on the selection working.

## The measurement that shaped the tool, kept here so it is not re-litigated

`turbotab/api.py` is an aggregator: 104 modules import it, and it imports most
of the package. Full transitive reachability therefore answers *"could this
reach the change"* with **yes** for nearly everything — on this tree a one-line
change to `turbotab/models.py` selects **131 of 147** files and `models/glm.py`
selects **134**. That is true and nearly useless. The default is direct
importers plus named triggers; `--closure` is the over-approximation; **both
counts print either way**, because the gap is what tells a reader whether to
trust the scoping.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

TOOL = os.path.join(PROJECT_ROOT, "docs", "turbotab", "tools", "affected.py")


def _run(*args):
    out = subprocess.run([sys.executable, TOOL, *args],
                         cwd=PROJECT_ROOT, capture_output=True, text=True,
                         timeout=600)
    return out.returncode, out.stdout + out.stderr


def _json(*args):
    code, text = _run(*args, "--json")
    start = text.index("{")
    return code, json.loads(text[start:])


# ═══════════ 1 · the walk is measuring the right tree ═══════════════════════

def test_the_graph_walks_the_repository_and_not_a_virtualenv():
    """A positive control on the measurement itself.

    The first draft excluded `venv` and not `.venv`, and this tree contains a
    `turbotab/.venv/` — **3,272 site-packages files** walked in as production
    modules, which put `pytest` and `numpy` in the graph and made every count
    meaningless. A walk is a measurement, and this one was measuring the wrong
    tree while looking perfectly healthy.
    """
    from docs.turbotab.tools import affected as A          # noqa: F401
    graph, by_module = A.build_graph()
    tests = [p for p in by_module.values() if p.startswith("turbotab/test_")]

    assert 100 <= len(tests) <= 400, (
        f"the walk found {len(tests)} `turbotab/test_*.py` files. The suite is "
        f"about 150; a number in the thousands means an environment directory "
        f"is being walked as source.")
    strays = [p for p in by_module.values()
              if "site-packages" in p or "/." in p or p.startswith(".")]
    assert not strays, f"the graph contains non-source paths: {strays[:5]}"
    assert "turbotab.api" in graph, "the graph is missing the module every test drives"


# ═══════════ 2 · the refusals, which are the point ══════════════════════════

#: A module with **no direct importer among the test files** and a large
#: closure — the shape the refusal is about.
#:
#: **This was `models/glm.py` and it had to be repointed within the hour.**
#: `L56-C3` added `test_a_wrapped_estimator_forwards_its_coefficients.py`, which
#: imports `GLMWrapper` directly, so the wrapper stopped having the shape and
#: the positive control below went red — telling this file its subject had moved
#: rather than letting it pass while asserting nothing. That is the control
#: doing its job, and the reason it is a control and not a comment.
UNSCOPABLE = "ml/bootstrap.py"


def test_an_empty_direct_selection_refuses_instead_of_reporting_zero():
    """**The sharpest thing the tool does.**

    `ml/bootstrap.py` has zero direct importers among the 147 turbotab test
    files and 135 under closure — no test imports it, and every test that
    drives the API reaches it through the aggregator. Printing "0 selected"
    here would be the tool's own headline failure, so it escalates instead.
    """
    code, body = _json("--files", UNSCOPABLE)

    # POSITIVE CONTROL — the shape this test is about really is present. If
    # some future change gives the module a direct importer, this test is
    # asserting nothing and should be REPOINTED rather than deleted. It has
    # already fired once, for exactly that reason.
    assert body["counts"]["direct"] == 0 and body["counts"]["closure"] > 0, (
        f"`{UNSCOPABLE}` no longer has the zero-direct/non-zero-closure shape "
        f"this test is about: {body['counts']}. Repoint it at a module that "
        f"does — the tool's refusal is still the claim under test.")

    assert body["escalations"], (
        "the tool reported an empty selection for a change 134 test files can "
        "reach through the aggregator, and said nothing about it")
    assert code == 2, f"an escalation must not exit 0; got {code}"


def test_pytest_args_emits_nothing_when_it_could_not_scope():
    """The mode a caller pipes into `xargs pytest`.

    A caller that ignores the exit status must get an EMPTY list rather than a
    confident wrong one — running zero tests is visibly wrong, and running a
    plausible-looking subset that omits the affected ones is not.
    """
    code, text = _run("--files", UNSCOPABLE, "--pytest-args")
    assert code == 2, f"escalation must exit 2 in --pytest-args mode; got {code}"
    assert not text.strip(), (
        f"the tool printed a selection it had just refused to make: {text!r}")


def test_a_conftest_change_cannot_be_scoped_at_all():
    """Unbounded reach. `conftest.py` can rewrite collection for a whole tree."""
    code, body = _json("--files", "tests/conftest.py")
    assert body["escalations"], "a conftest change was scoped as if it were ordinary"
    assert code == 2


# ═══════════ 3 · the blind spots are printed whether or not they fire ══════

def test_the_blind_spots_are_printed_on_every_run_including_a_clean_one():
    """Not only when triggered.

    A tool that lists its limits only when it notices one has hit is a tool
    whose silence means nothing — which is the same defect as a matcher that
    fires on prose, one layer out.
    """
    _code, text = _run("--files", "turbotab/deck_faces.py")
    assert "CANNOT SEE" in text, (
        f"a clean, well-scoped run printed no blind-spot list:\n{text}")
    assert "reflection" in text, "the reflection blind spot is not named"
    # CASE-INSENSITIVE, because the tool shouts that line and the first draft
    # of this assertion did not. A matcher that fails on capitalization is the
    # same class as one that fires on prose: it is answering about the string
    # rather than about the claim.
    assert "scoped" in text.lower(), "the output never uses the word 'scoped'"
    assert "never reported as a full run" in text.lower(), (
        "the output does not say that a scoped run is not a full run, which is "
        f"the rule this tool exists to serve:\n{text}")

    _code, body = _json("--files", "turbotab/deck_faces.py")
    assert len(body["blind_spots"]) >= 4, (
        f"the machine-readable output carries {len(body['blind_spots'])} blind "
        f"spots; the docstring names more than that")


# ═══════════ 4 · the trigger for the edge an import walk cannot see ════════

def test_a_page_change_selects_the_harness_tests_it_is_invisible_to():
    """`turbotab/web/index.html` is JavaScript. No test imports it.

    Every interface claim runs through `pageharness.py`, which READS that file
    — so the page is reachable by 56 tests and visible to none of them as an
    import. This is the named trigger, and the positive control is that the
    import walk really does find nothing.
    """
    from docs.turbotab.tools import affected as A

    graph, by_module = A.build_graph()
    assert "turbotab/web/index.html" not in by_module.values(), (
        "the page is in the module graph, so this test's premise is wrong")

    _code, body = _json("--files", "turbotab/web/index.html")
    picked = {row["file"] for row in body["selected"]}
    assert len(picked) >= 20, (
        f"a page change selected {len(picked)} test files; the harness-driven "
        f"set is far larger than that, so the trigger did not fire")

    harness_tests = {p for m, p in by_module.items()
                     if p.startswith("turbotab/test_")
                     and ("turbotab.pageharness" in graph[m]
                          or "pageharness" in graph[m])}
    assert harness_tests <= picked, (
        f"{len(harness_tests - picked)} tests drive the page harness and were "
        f"not selected for a page change: {sorted(harness_tests - picked)[:4]}")
    for row in body["selected"]:
        assert row["reason"], f"{row['file']} was selected with no reason given"


def test_a_narrow_change_actually_scopes():
    """The other direction, so this file is not only about refusing.

    A selector that refuses everything is safe and worthless.
    """
    _code, body = _json("--files", "turbotab/deck_faces.py")
    picked = {row["file"] for row in body["selected"]}
    assert picked == {"turbotab/test_the_deck_keeps_its_cards.py"}, (
        f"expected the one file that imports `deck_faces`, got {sorted(picked)}")
    assert not body["escalations"]
    assert body["counts"]["closure"] < body["counts"]["total"] * 0.1, (
        f"a change to a leaf module reaches {body['counts']['closure']} of "
        f"{body['counts']['total']} files under closure, which would mean the "
        f"graph has no leaves and scoping is impossible in principle")
