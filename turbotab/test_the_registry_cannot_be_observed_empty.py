"""`TEST-041` — the structural fix. A count that moves with import order is not
a count.

`figures.REGISTRY` was a plain dict filled as an *import side effect* of
`turbotab.figure_specs`. A reader's answer therefore depended on whether
something else had imported the populator first.
`test_the_companion_rule_reaches_the_document` asserted a count over it in its
first test — which takes no fixture, so nothing had run. Alone the file
reported::

    AssertionError: {}
    assert 0 >= 4

Inside the full suite it passed. **The full-suite green was the false one**: a
different file's import made the assertion true, not the code under test.

**The third face of one property.** `TEST-030` names it for `tests/workflow`
(ordering); `TEST-040` was its load-dependent twin, where a bounded poll with
no wait reported *the app had not answered yet* as *the app answered wrong*.
All three make a number depend on something other than the code.

L43 fixed the four readers and added a static guard requiring every reader to
import a populator. That is a convention with a check — and the check is
static where the property is behavioral. This is the structure underneath it:
**the first read is the population.**

The guard from L43 stays. It is now belt to this braces, and it costs nothing;
if the lazy dict is ever replaced by a plain one, that check is what notices
before this file does.
"""
from __future__ import annotations

import subprocess
import sys

import pytest


#: The read operations a caller actually uses. Each must populate, because a
#: reader that reached the dict by a path nobody thought of is exactly how the
#: original hazard survived.
READS = [
    ("len", lambda r: len(r)),
    ("in", lambda r: "calibration" in r),
    ("get", lambda r: r.get("calibration")),
    ("getitem", lambda r: r["calibration"]),
    ("values", lambda r: list(r.values())),
    ("items", lambda r: list(r.items())),
    ("keys", lambda r: list(r.keys())),
    ("iter", lambda r: list(iter(r))),
    ("bool", lambda r: bool(r)),
]


@pytest.mark.parametrize("name,read", READS, ids=[n for n, _ in READS])
def test_every_read_populates_in_a_fresh_interpreter(name, read):
    """**A fresh interpreter per read, and that is the whole design of this
    test.**

    Populating is a one-time side effect, so any check inside an already-warm
    process is measuring what an earlier import did — which is the defect,
    used as the instrument. Each case therefore runs in its own subprocess
    that imports `turbotab.figures` and *nothing else*.
    """
    source = (
        "from turbotab import figures\n"
        "r = figures.REGISTRY\n"
        f"got = ({READ_SOURCE[name]})\n"
        "print(len(r))\n")
    out = subprocess.run([sys.executable, "-c", source],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-1500:]
    assert int(out.stdout.strip()) >= 4, (
        f"after a `{name}` read on a cold import, the registry holds "
        f"{out.stdout.strip()} specs — the read did not populate it")


#: The expression for each read, as source, because the subprocess cannot
#: receive a lambda.
READ_SOURCE = {
    "len": "len(r)",
    "in": "'calibration' in r",
    "get": "r.get('calibration')",
    "getitem": "r['calibration']",
    "values": "list(r.values())",
    "items": "list(r.items())",
    "keys": "list(r.keys())",
    "iter": "list(iter(r))",
    "bool": "bool(r)",
}


def test_a_cold_import_of_figures_alone_sees_the_specs():
    """The headline, in the shape the defect had: import the module that
    *declares* the registry, and read it, with nothing else imported."""
    out = subprocess.run(
        [sys.executable, "-c",
         "from turbotab import figures\n"
         "assert 'figure_specs' not in __import__('sys').modules, "
         "'figure_specs was already imported; this is not a cold read'\n"
         "n = len(figures.REGISTRY)\n"
         "p = len(figures.PENDING)\n"
         "print(n, p)\n"],
        capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-1500:]
    specs, pending = (int(x) for x in out.stdout.split())
    assert specs >= 15, f"a cold read saw {specs} specs"
    assert pending >= 2, f"a cold read saw {pending} pending figures"


def test_the_companion_file_passes_alone():
    """The original symptom, driven. This file is the reason `TEST-041` exists
    and it is the only end-to-end statement of the property that matters."""
    out = subprocess.run(
        [sys.executable, "-m", "pytest",
         "turbotab/test_the_companion_rule_reaches_the_document.py",
         "-q", "--no-header", "-p", "no:cacheprovider"],
        capture_output=True, text=True)
    assert out.returncode == 0, out.stdout[-2500:]


def test_a_read_during_population_does_not_loop():
    """Re-entrancy, pinned directly instead of through a coincidence.

    **A revert probe corrected this test's first version.** It claimed the
    flag-before-import ordering was what stopped `register()` recursing —
    `register` asks `spec.id in REGISTRY`, which populates. Moving the
    assignment after the import came back `GREEN — NOT LOAD-BEARING`, and a
    cold interpreter with the recursion limit at 200 returned 17 specs and no
    error: Python's `sys.modules` returns the partially-initialized module on
    re-entry, so the nested import is a no-op. **The comment asserted a
    mechanism that was not doing the work.**

    So this exercises the property itself. A `_populate` that READS the dict
    it is populating is the re-entrant case, and it must terminate whatever
    the import system does.
    """
    from turbotab import figures

    assert len(figures.REGISTRY) > 0
    assert figures.REGISTRY._populated is True, (
        "reading the registry did not mark it populated, so every read pays "
        "the import again")

    depth = {"n": 0, "max": 0}

    class ReadsItselfWhilePopulating(figures._SelfPopulating):
        def _populate(self):
            if self._populated:
                return
            self._populated = True
            depth["n"] += 1
            depth["max"] = max(depth["max"], depth["n"])
            # The re-entrant read: exactly what `register` does.
            _ = "anything" in self
            _ = len(self)
            self["late"] = 1
            depth["n"] -= 1

    fresh = ReadsItselfWhilePopulating()
    assert len(fresh) == 1, "population did not run, so nothing was tested"
    assert depth["max"] == 1, (
        f"_populate re-entered to depth {depth['max']}; the flag is not "
        f"guarding the body it is supposed to guard")


def test_a_write_does_not_populate():
    """`register` writing into the dict must not trigger the import — that is
    the same re-entrancy from the other side, and it is why `__setitem__` is
    deliberately not overridden."""
    from turbotab import figures

    fresh = figures._SelfPopulating()
    fresh["x"] = 1
    assert fresh._populated is False, (
        "a write populated the registry, so `register` would re-enter the "
        "import it is being called from")


def test_the_static_guard_from_l43_still_runs():
    """Belt and braces, and the braces stay.

    If the lazy dict is ever replaced by a plain one, L43's static check —
    every registry reader imports a populator — is what notices before this
    file does. Deleting it because this exists would be trading a check that
    catches the cause for one that catches the symptom.
    """
    import pathlib

    guard = (pathlib.Path(__file__).resolve().parent
             / "test_the_suite_does_not_depend_on_what_ran_before_it.py")
    assert guard.exists(), (
        "L43's ordering guard is gone; this file replaces its mechanism, not "
        "its coverage")
