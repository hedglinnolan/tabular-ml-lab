"""A `FIXED` row's guard must RUN. A skipped test guards nothing.

**~2,473 seconds. This file is the one every documented invocation excludes.**
The cheap half of the guard moved out at `L55-A1` and is
`tests/test_a_fixed_rows_named_test_resolves_in_five_seconds.py`; read that
file's docstring for why the split exists, because the reason is itself a
finding (`TEST-067`).

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

`tools/ledger.py check` verifies a `FIXED` row *names* a test. The cheap half
verifies the name RESOLVES. **This file verifies the named test EXECUTES.**
Between the three, the ledger's integrity claim — *a FIXED row has a test that
fails on revert* — has all of its cheap halves checked, except for
load-bearingness, which only a revert probe can measure and which
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
what exists and the ledger's names are matched against the answer. That
resolver now lives in `tests/fixed_row_guard.py` and is IMPORTED by both halves
rather than copied, because splitting a file is precisely when the second
implementation gets written.

**Two cheaper designs were measured and rejected, and the measurement is the
reason.** `--collect-only` and `--setup-plan` both finish in ~3s, and both
report **zero** skips for `tests/test_characterization_wrappers.py`, where a
real run reports **four**. An `importorskip` in a test BODY is invisible until
the body runs, and that is precisely the L17 case this file was written for. So
the nodes are executed; what was removed is executing everything around them.
"""
from __future__ import annotations

import os
import subprocess
import sys
from typing import Dict

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tests.fixed_row_guard import resolve                          # noqa: E402

# Rows whose guard is allowed not to run here, each with the reason. Empty by
# design: the L17 sweep moved `MODELS-001` to PARTIAL rather than exempting it,
# because an exemption would have recorded the unverifiable claim as verified.
#
# An entry here is a promise that the guard runs SOMEWHERE and that somewhere is
# named. "It needs torch" is not a reason; "it needs torch, and CI installs
# torch, and the CI job is <name>" is.
SKIP_BACKED: Dict[str, str] = {}


# L53-A2. THE CAP IS RAISED, AND WITH `AUDIT-040` QUOTED BESIDE IT, WHICH IS
# the difference between this and the act the row forbids. Raising a budget to
# make a red test green is relaxing a gate under pressure. What is happening
# here is that the budget was never achievable and now the reason is measured:
# `AUDIT-040` records that ONE FILE holds 2157s of this set in four tests —
# 632s, 570s, 505s, 450s — and parallelism cannot go below the longest single
# test, so 632s is a hard floor no width goes under.
#
# So the cap changes PURPOSE rather than value. It is not a budget any more; it
# is a HANG DETECTOR. The measured cost is 19:08 on the quiet machine §06
# already requires for a quotable suite, and 33:19 under a five-agent fan-out.
# 2700s catches a genuine hang and stops pretending to enforce a number the
# floor makes impossible.
#
# THE BUDGET CLAIM MOVES TO `AUDIT-040`, where it can actually be acted on:
# make the four monsters cheaper and this comes back under twenty minutes on
# its own. Raising this again without that row closing would be the thing the
# rule is against.
@pytest.mark.timeout(2700)
def test_every_fixed_rows_named_test_actually_runs():
    """The guard this file exists for.

    Runs the NODES that `FIXED` rows name — not the files that contain them —
    and fails on any row whose every named node skipped.
    """
    resolved, _missing, _not_pytest = resolve()
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
        cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=2400)
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


def test_the_exemption_list_is_argued_rather_than_a_list():
    """`SKIP_BACKED` is where this check would be defeated, so it is checked too.

    An exemption says the guard runs somewhere. That somewhere has to be named:
    "it needs torch" is a restatement of the problem, not a reason.

    Stays with the slow half because `SKIP_BACKED` does: the exemption list is
    an exemption from the SKIP check, and moving the argument away from the
    check it argues about is how a rule and its enforcement drift apart.
    """
    thin = [k for k, why in SKIP_BACKED.items() if len(why) < 60]
    assert not thin, (
        f"{thin} claim an exemption without naming where the guard does run. "
        "An exemption is an argument, not a keyword.")
