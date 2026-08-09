"""The cheap half of the `FIXED`-row guard. **~5 seconds, so it runs every batch.**

The cost is in the filename on purpose, and so is the reason.

## Why this file exists — and the reason is the finding

`ledger.py check` verifies a `FIXED` row *names* a test.
:func:`test_every_fixed_rows_named_test_exists` verifies the name **resolves**
to something pytest can collect. That check costs one `--collect-only`
subprocess — about five seconds.

It used to live in `tests/test_a_fixed_row_names_a_test_that_actually_runs.py`,
beside `test_every_fixed_rows_named_test_actually_runs`, which costs **2,473
seconds**. Every documented invocation of the suite therefore passes
`--ignore=tests/test_a_fixed_row_names_a_test_that_actually_runs.py`, and
`--ignore` takes a **file**. So excluding the slow check silently excluded the
cheap one, and the check that validates the core claim of *every closed row in
the ledger* had not run since the loop that wrote it.

The first time it did run, it found a live violation: `TEST-063` was `FIXED`
naming `test_a_pack_may_replace_a_core_operation_only_by_saying_so`, which is in
no collected test anywhere in the tree. That row has been reopened.

**The class, stated so it is not rediscovered:** two checks in one file share one
exclusion, and `--ignore` cannot tell them apart. A cheap guard sharing a file
with an expensive one inherits the expensive one's exclusions and goes dark
without anything saying so — the suite stays green, the guard reports nothing,
and nothing distinguishes "ran and found nothing" from "never ran". `TEST-067`.

**What is NOT here.** `test_every_fixed_rows_named_test_actually_runs` — the
check that a named node *executes* rather than skipping — stays in the slow file
with its 2,700s hang detector, and stays excluded. That check is real and it is
run deliberately, not incidentally (`AGENT_ONBOARD.md` §03).

The resolver both halves read is `tests/fixed_row_guard.py`, which is a module
rather than a copy: the original file's own docstring records that a regex
approximating pytest's collection is a second implementation to keep in sync,
and splitting a file is exactly the moment that mistake gets made.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tests.fixed_row_guard import resolve                          # noqa: E402

#: The document that states how this project's suites are actually invoked.
#: Every count this project reports comes from the commands in §03 of this file,
#: because `make test` does not run (`TEST-038`).
ONBOARD = os.path.join(PROJECT_ROOT, "docs", "turbotab", "prompts",
                       "AGENT_ONBOARD.md")


def test_every_fixed_rows_named_test_exists():
    """`ledger.py check` verifies a row NAMES a test. This verifies it RESOLVES.

    A row naming a function that no longer exists has a guard nobody can run,
    and the skip check in the slow half cannot see it — an uncollectable node
    never reports a skip. Four rows were in this state when the check was
    written; a fifth (`TEST-063`) was found the first time the check actually
    ran, two loops later, for the reason in this file's docstring.
    """
    resolved, missing, _ = resolve()
    assert resolved, "no FIXED row resolved to a node at all; the resolver is broken"
    offenders = [f"{rid} names {names}, which is in no collected test"
                 for rid, names in missing if names]
    assert not offenders, (
        "these FIXED rows name a test function that does not exist:\n  "
        + "\n  ".join(offenders) +
        "\n\nThe row's guard cannot be run by anyone reading the ledger. "
        "Rename the row's `test` to the guard that replaced it, or set the row "
        "PARTIAL — a name that resolves to nothing is not a named test.")


def test_the_rows_with_no_pytest_target_are_named_rather_than_ignored():
    """A row whose guard is a document or a tool is a real category.

    It is not an offender — `T0-DOC-001` names an `ARCHITECTURE.md` section and
    that is what its fix was. But a check that drops a class of rows without
    saying so reports cleaner coverage than it has, so the count is asserted to
    be small and the rows are listed when it is not.

    Travels with the cheap half because it shares `resolve()` with the check
    above and costs nothing beyond it — and because it was excluded by the same
    `--ignore` for the same wrong reason.
    """
    _resolved, _missing, not_pytest = resolve()
    assert len(not_pytest) <= 10, (
        f"{len(not_pytest)} FIXED rows name no `.py` file at all: {not_pytest}. "
        f"Each is outside this guard's reach, and a growing set of them is the "
        f"ledger routing around the check rather than a category of fix.")


def test_the_documented_invocation_still_collects_this_check():
    """`TEST-067`. The split is only a fix while this file stays collected.

    The defect the split repaired was not that the check was wrong — it was that
    the check was **excluded**, by a `--ignore` aimed at a 2,473-second neighbour
    that `--ignore` could not tell apart from it. Moving the function to a new
    file fixes the instance and guards nothing: adding one more `--ignore` to the
    documented command, or moving the function back, restores the defect exactly
    and turns this guard dark again without any test going red.

    **So the subject of this test is the documented command itself.** The
    exclusion list is READ from `AGENT_ONBOARD.md` §03 rather than restated here,
    because a restated list is a second copy that drifts — and the drift would be
    silent in precisely the direction that hides a check.

    This is the one shape of guard that can fire for this class: collection is
    asked of pytest under the real flags, and the answer must contain this node.
    """
    text = open(ONBOARD, encoding="utf-8").read()
    ignores = sorted(set(re.findall(r"--ignore=(\S+)", text)))

    # POSITIVE CONTROL — the document was found, is the right document, and the
    # exclusions were parsed. Without this, a renamed file or a reworded section
    # makes the assertion below pass over an empty list, which is the
    # matcher-fires-on-nothing half of trap 5b.
    assert len(ignores) >= 2, (
        f"parsed {ignores} out of {ONBOARD}; §03's invocations name at least "
        f"two `--ignore` paths, so the parse is wrong or the document moved")
    assert "tests/test_a_fixed_row_names_a_test_that_actually_runs.py" in ignores, (
        f"the documented fast tier no longer excludes the 2,473-second half. "
        f"Parsed: {ignores}. If that check is now cheap enough to run every "
        f"batch this test should be deleted, not adjusted.")

    out = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "--no-header",
         "-p", "no:randomly", "tests/",
         *[f"--ignore={i}" for i in ignores]],
        cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=600)
    # THE LEAF SEGMENT, NOT A SUBSTRING, and the probe is why. A first draft
    # asked whether `"::test_every_fixed_rows_named_test_exists"` appeared in
    # the output, and a revert that renamed the function to
    # `test_every_fixed_rows_named_test_exists_RENAMED` came back GREEN — the
    # matcher had fired on the longer name that contains it. Trap 5b, inside the
    # guard written to close it. Node ids are structured, so the structure is
    # what gets compared.
    leaves = {line.strip().rsplit("::", 1)[-1].split("[")[0]
              for line in out.stdout.splitlines() if "::" in line}

    # POSITIVE CONTROL — collection produced something. An `--ignore` pointing
    # at a path that no longer exists makes pytest exit 4 with nothing
    # collected, and "the node is absent" would then be true for the wrong
    # reason.
    assert leaves, (
        f"the documented invocation collected no tests at all:\n"
        f"{(out.stdout + out.stderr)[-2000:]}")

    assert "test_every_fixed_rows_named_test_exists" in leaves, (
        "the check that every FIXED row's named test RESOLVES is not collected "
        "by the documented fast-tier invocation, so it does not run every "
        "batch — which is the exact state that let `TEST-063` sit FIXED naming "
        "a function that exists nowhere, through two green loops. "
        f"Exclusions read from AGENT_ONBOARD.md §03: {ignores}")
