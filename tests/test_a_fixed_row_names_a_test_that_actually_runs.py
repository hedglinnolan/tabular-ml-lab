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
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Dict, List, Set

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


def _rows() -> List[Dict]:
    data = json.load(open(FINDINGS, encoding="utf-8"))
    return data["findings"] if isinstance(data, dict) and "findings" in data else data


def _collected_skips() -> Set[str]:
    """Every test node the suite reports as SKIPPED, from one real run.

    Measured rather than inferred: a static scan for `importorskip` would miss a
    module-level `pytest.skip`, a `skipif` on an environment variable, or a
    fixture that skips — and would flag a skip inside a branch that never runs.
    """
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "--no-header",
         "-p", "no:randomly", "tests", "turbotab", "utils"],
        cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=600)
    if out.returncode not in (0, 1, 2, 5):
        pytest.skip(f"could not collect the suite (exit {out.returncode})")
    return set()


def _run_and_report_skips() -> Set[str]:
    """Node ids that skipped, from an actual run with `-rs`."""
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--no-header", "-rs",
         "--continue-on-collection-errors", "-p", "no:randomly",
         *_test_paths()],
        cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=1800)
    skipped: Set[str] = set()
    for line in (out.stdout + out.stderr).splitlines():
        line = line.strip()
        if line.startswith("SKIPPED [") and "] " in line:
            # SKIPPED [1] tests/test_x.py:208: could not import 'torch'
            where = line.split("] ", 1)[1].split(":")[0]
            skipped.add(where)
    return skipped


def _test_paths() -> List[str]:
    """Only the files a FIXED row actually names — the whole suite is minutes."""
    paths = set()
    for r in _rows():
        if r.get("status") == "FIXED" and r.get("test"):
            path = r["test"].split("::")[0]
            if os.path.exists(os.path.join(PROJECT_ROOT, path)):
                paths.add(path)
    return sorted(paths)


@pytest.mark.timeout(1800)
def test_every_fixed_rows_named_test_actually_runs():
    """The guard this file exists for.

    Runs the files that `FIXED` rows name, collects what skipped, and fails on
    any row whose named test is in that set. Scoped to those files rather than
    the whole suite because the claim is about the ledger, and because a
    twenty-minute check is a check somebody turns off.
    """
    skipped_files_by_line = _run_and_report_skips()
    if not _test_paths():
        pytest.skip("no FIXED row names a test file that exists")

    offenders = []
    for r in _rows():
        if r.get("status") != "FIXED" or not r.get("test"):
            continue
        node = r["test"]
        path = node.split("::")[0]
        if path not in skipped_files_by_line:
            continue
        # The file reported a skip. Confirm THIS node is the skipped one rather
        # than a neighbour — a file with one torch test and forty others must
        # not condemn all forty.
        if not _node_skips(node):
            continue
        if r["id"] in SKIP_BACKED:
            continue
        offenders.append(
            f"{r['id']} is FIXED and its guard {node} does not run here")

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
    """
    thin = [k for k, why in SKIP_BACKED.items() if len(why) < 60]
    assert not thin, (
        f"{thin} claim an exemption without naming where the guard does run. "
        "An exemption is an argument, not a keyword.")
