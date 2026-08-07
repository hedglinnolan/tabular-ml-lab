"""A `FIXED` row's guard must RUN. A skipped test guards nothing.

The L17 sweep found `MODELS-001` closed against
`test_clone_and_refit_now_raises_instead_of_answering_from_the_old_model`, which
opens with `pytest.importorskip("torch")` — and torch is deliberately not
installed. It reports `SKIPPED`, which in `-q` output is one character and in a
summary line is a number nobody reads. The fix may be correct; nothing in this
environment demonstrates it.

That is a fourth angle on the family this project keeps finding:

    principle-locality      fails across SPACE       stated here, violated there
    expiring guarantees     fails across TIME        true then, false now
    an untriggered check    fails across OCCASION    correct always, run never
    THIS ONE                fails across ENVIRONMENT true somewhere, unverifiable here

`tools/ledger.py check` verifies a `FIXED` row *names* a test. This verifies the
named test **executes**. Between them the ledger's integrity claim — *a FIXED
row has a test that fails on revert* — has both of its halves checked, except
for load-bearingness, which only a revert probe can measure and which
`docs/turbotab/data/revert-probe-sweep-l17.md` samples.

**Deliberately not a rule against `importorskip`.** Skipping is right for a test
about an optional dependency. What is wrong is a `FIXED` row *resting* on one:
the row claims a defect is guarded, and here nobody can see the guard. So a row
in that position must either name a guard that runs, or say `PARTIAL` and why.
`SKIP_BACKED` is the exemption list, and every entry needs a reason.

## L52-A — the guard had outgrown its own budget (`TEST-061`)

It ran **1800.03s against its own `pytest.mark.timeout(1800)`** and died there,
reporting **not one offender** — so its red said nothing about the question it
exists to answer, while 347 closed rows rested on it. Its own docstring
predicted it: *"a twenty-minute check is a check somebody turns off."*

**What was expensive was the shape, not the work.** It ran every FILE any
`FIXED` row named — **2796 tests across 142 files** — and then asked which node
had skipped. It now runs **the named NODES**, which is 671 targets, because the
question was never about the files.

**Resolution is delegated to pytest rather than re-implemented.** The first
attempt matched `def test_x(` with a regex and built `path::test_x`, which is
wrong for every test inside a class — pytest wants `path::Class::test_x` — and
a regex approximating pytest's collection is a second implementation to keep in
sync, which is this project's most-repeated defect. `--collect-only` is asked
what exists and the ledger's names are matched against the answer.

**Two cheaper designs were measured and rejected, and the measurement is the
reason.** `--collect-only` and `--setup-plan` both finish in ~3s, and both
report **zero** skips for `tests/test_characterization_wrappers.py`, where a
real run reports **four**. An `importorskip` in a test BODY is invisible until
the body runs, and that is precisely the L17 case this file was written for. So
the nodes are executed; what was removed is executing everything around them.

## And a second check, because the first one could not see this

Four rows named a test function that **does not exist anywhere** — `STATE-110`,
`GUIDED-028`, `GUIDED-158`, `GUIDED-168`. The old guard could not notice: it
asked whether the named node SKIPPED, and a node that cannot be collected never
reports a skip, so those four passed silently. `ledger.py check` verifies a row
NAMES a test; nothing verified the name RESOLVED. That is now
`test_every_fixed_rows_named_test_exists`.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from typing import Dict, List, Set, Tuple

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

FINDINGS = os.path.join(PROJECT_ROOT, "docs", "turbotab", "data", "findings.json")

# Rows whose guard is allowed not to run here, each with the reason. Empty by
# design: the L17 sweep moved `MODELS-001` to PARTIAL rather than exempting it,
# because an exemption would have recorded the unverifiable claim as verified.
#
# An entry here is a promise that the guard runs SOMEWHERE and that somewhere is
# named. "It needs torch" is not a reason; "it needs torch, and CI installs
# torch, and the CI job is <name>" is.
SKIP_BACKED: Dict[str, str] = {}

#: Rows whose `test` field names something that is not a pytest target at all —
#: a document, a tool invocation, a data file. They are REPORTED rather than
#: silently dropped, because "no pytest target" is a fact about the row and a
#: check that quietly ignores a class of rows is the shrug this file is against.
_PATH = re.compile(r"([A-Za-z0-9_./-]+\.py)")
# A NAME, not the English word. The first draft was `(?:test|Test)[A-Za-z0-9_]*`
# and it matched "tests" inside `(seven tests)`, reporting forty-odd rows as
# naming a function that does not exist — the matcher-fires-on-prose failure,
# which this project keeps meeting one level down from wherever it is looking.
# A pytest function is `test_` with the underscore; a class is `Test` followed
# by a capital.
_FUNC = re.compile(r"\b(test_[A-Za-z0-9_]+|Test[A-Z][A-Za-z0-9_]*)")


def _rows() -> List[Dict]:
    data = json.load(open(FINDINGS, encoding="utf-8"))
    return data["findings"] if isinstance(data, dict) and "findings" in data else data


def _fixed_rows() -> List[Dict]:
    return [r for r in _rows() if r.get("status") == "FIXED" and r.get("test")]


def _collected() -> List[str]:
    """Every node id pytest can collect from the files `FIXED` rows name.

    ASKED, NOT RE-DERIVED. Class methods, parametrization ids and conftest
    collection rules are pytest's business, and a regex that reproduces them is
    a second implementation of pytest.
    """
    files = sorted({p for r in _fixed_rows() for p in _PATH.findall(r["test"])
                    if os.path.exists(os.path.join(PROJECT_ROOT, p))})
    if not files:
        return []
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "--no-header",
         "-p", "no:randomly", "--continue-on-collection-errors", *files],
        cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=600)
    return [line.strip() for line in out.stdout.splitlines()
            if "::" in line and line.strip().startswith(("tests/", "turbotab/", "utils/"))]


def _resolve() -> Tuple[Dict[str, List[str]], List[Tuple[str, List[str]]], List[str]]:
    """(row id -> its node ids, rows naming a missing function, rows with no .py)."""
    # EVERY SEGMENT, not just the leaf. A row may name the CLASS
    # (`TestCommaReading`) rather than the method, and an index keyed on leaves
    # only reports every such row as naming something that does not exist —
    # which is what the first draft did, for thirty rows.
    by_func: Dict[str, List[str]] = {}
    for nid in _collected():
        for segment in nid.split("::")[1:]:
            by_func.setdefault(segment.split("[")[0], []).append(nid)

    resolved: Dict[str, List[str]] = {}
    missing: List[Tuple[str, List[str]]] = []
    not_pytest: List[str] = []
    for r in _fixed_rows():
        paths = _PATH.findall(r["test"])
        if not paths:
            not_pytest.append(r["id"])
            continue
        stems = {p.split("/")[-1][:-3] for p in paths}
        funcs = [f for f in _FUNC.findall(r["test"]) if f not in stems]
        hits = sorted({n for f in funcs for n in by_func.get(f, [])
                       if n.split("::")[0] in paths})
        if hits:
            resolved[r["id"]] = hits
        elif funcs:
            missing.append((r["id"], sorted(set(funcs) - set(by_func))))
    return resolved, missing, not_pytest


def test_every_fixed_rows_named_test_exists():
    """`ledger.py check` verifies a row NAMES a test. This verifies it RESOLVES.

    A row naming a function that no longer exists has a guard nobody can run,
    and the skip check below cannot see it — an uncollectable node never
    reports a skip. Four rows were in this state when the check was written.
    """
    resolved, missing, _ = _resolve()
    assert resolved, "no FIXED row resolved to a node at all; the resolver is broken"
    offenders = [f"{rid} names {names}, which is in no collected test"
                 for rid, names in missing if names]
    assert not offenders, (
        "these FIXED rows name a test function that does not exist:\n  "
        + "\n  ".join(offenders) +
        "\n\nThe row's guard cannot be run by anyone reading the ledger. "
        "Rename the row's `test` to the guard that replaced it, or set the row "
        "PARTIAL — a name that resolves to nothing is not a named test.")


@pytest.mark.timeout(1800)
def test_every_fixed_rows_named_test_actually_runs():
    """The guard this file exists for.

    Runs the NODES that `FIXED` rows name — not the files that contain them —
    and fails on any row whose every named node skipped.
    """
    resolved, _missing, _not_pytest = _resolve()
    if not resolved:
        pytest.skip("no FIXED row names a node that exists")

    targets = sorted({n for ns in resolved.values() for n in ns})
    # PARALLEL, AND THIS IS THE FALLBACK THE ROW NAMED. Running the named NODES
    # instead of the files that contain them cut the work from 2796 tests to
    # 671 — a real 3x — and it was NOT ENOUGH: 671 nodes took 25:10, because the
    # guards a `FIXED` row names are themselves the slow tests, several of them
    # page drives that start a JS engine each. Raising the timeout was never on
    # the menu; it converts a check nobody can wait for into a check nobody
    # runs. So the runs go wide.
    #
    # `--dist load` and NOT `--dist loadfile`, WHICH WAS MEASURED. Keeping a
    # file on one worker sounds safer for module-scoped fixtures, and it puts
    # all four monsters in `turbotab/test_ask_me_anyway_reopens_the_question.py`
    # — 632s, 570s, 505s, 450s — on a single worker, which is 36 minutes of
    # serial work no other core can touch. Spreading individual tests instead
    # brings the same 880 targets in at 19:08.
    #
    # THE FLOOR IS THE LONGEST SINGLE TEST, 632s, and no amount of width goes
    # under it. If this guard ever breaches its budget again, that file is
    # where to look first — not here.
    #
    # SERIAL IF XDIST IS ABSENT, because a guard that cannot run where the
    # dependency is missing has just moved the failure rather than fixed it.
    try:
        import xdist                                        # noqa: F401
        wide = ["-n", "auto", "--dist", "load"]
    except ImportError:
        wide = []
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--no-header", "-rs",
         "--continue-on-collection-errors", "-p", "no:randomly",
         *wide, *targets],
        cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=1500)
    text = out.stdout + out.stderr

    # `-rs` reports `SKIPPED [1] path:line: reason`, which names the FILE and a
    # line rather than the node. Ran/skipped is therefore resolved per file and
    # then narrowed per node by `_node_skips`, which is one targeted run for
    # each candidate rather than for all of them.
    skipped_files = {line.split("] ", 1)[1].split(":")[0]
                     for line in text.splitlines()
                     if line.strip().startswith("SKIPPED [") and "] " in line}

    offenders = []
    for rid, nodes in resolved.items():
        if rid in SKIP_BACKED:
            continue
        candidates = [n for n in nodes if n.split("::")[0] in skipped_files]
        if not candidates or not all(_node_skips(n) for n in candidates):
            continue
        if len(candidates) < len(nodes):
            continue        # some parametrization ran; the guard is not dark
        offenders.append(f"{rid} is FIXED and its guard {nodes[0]} does not run here")

    assert not offenders, (
        "these FIXED rows rest on a test that SKIPS in this environment, so "
        "nothing demonstrates the fix:\n  " + "\n  ".join(offenders) +
        "\n\nEither name a guard that runs — the dangerous behavior is usually "
        "testable against a stub — or set the row PARTIAL saying why, which is "
        "what L17 did with MODELS-001. Adding it to SKIP_BACKED needs a reason "
        "that names where the guard DOES run.")


def _node_skips(node: str) -> bool:
    """Does this specific test node skip? One targeted run, cached per node."""
    if node in _NODE_CACHE:
        return _NODE_CACHE[node]
    out = subprocess.run(
        [sys.executable, "-m", "pytest", node, "-q", "--no-header", "-rs",
         "-p", "no:randomly"],
        cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=300)
    text = out.stdout + out.stderr
    _NODE_CACHE[node] = ("skipped" in text.lower()
                         and " passed" not in text
                         and " failed" not in text)
    return _NODE_CACHE[node]


_NODE_CACHE: Dict[str, bool] = {}


def test_the_rows_with_no_pytest_target_are_named_rather_than_ignored():
    """A row whose guard is a document or a tool is a real category.

    It is not an offender — `T0-DOC-001` names an `ARCHITECTURE.md` section and
    that is what its fix was. But a check that drops a class of rows without
    saying so reports cleaner coverage than it has, so the count is asserted to
    be small and the rows are listed when it is not.
    """
    _resolved, _missing, not_pytest = _resolve()
    assert len(not_pytest) <= 10, (
        f"{len(not_pytest)} FIXED rows name no `.py` file at all: {not_pytest}. "
        f"Each is outside this guard's reach, and a growing set of them is the "
        f"ledger routing around the check rather than a category of fix.")


def test_the_exemption_list_is_argued_rather_than_a_list():
    """`SKIP_BACKED` is where this check would be defeated, so it is checked too.

    An exemption says the guard runs somewhere. That somewhere has to be named:
    "it needs torch" is a restatement of the problem, not a reason.
    """
    thin = [k for k, why in SKIP_BACKED.items() if len(why) < 60]
    assert not thin, (
        f"{thin} claim an exemption without naming where the guard does run. "
        "An exemption is an argument, not a keyword.")
