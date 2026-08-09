"""`TEST-063`'s class, guarded where no single-node run can see it.

## The class

A test that mutates module-level state to prove a mutation is *permitted* leaves
the mutation behind, and the cost lands on whoever writes the next test that
reads that state — possibly loops later. `turbotab.recipes._OPERATIONS` and
`_DEFAULTS` are module-level, `packs.load` never unloads, and
`turbotab/test_the_recipe_table_is_a_table.py` registers a **fake** contributor
(`origin="fake_metabolomics_pack"`) in four places to prove the extension point
works.

## Why this file is a subprocess and not an assertion

The defect is **cross-test**: it is invisible to the test that causes it and
invisible to any run of that test alone. `revertprobe.py` invokes one node with
`-x`, so a probe of the causing test structurally cannot observe it. The only
instrument that can is *two nodes in one process, in order* — which is what this
does: it runs the whole mutating file and then a reader, in one interpreter, and
asks the reader what survived.

## What the reader asks, and why not "is the table equal to core"

Equality with core is the wrong question and would make this guard fire on
correct behavior. `GUIDED-099` ruled that the pack table is filtered at read
time rather than unloaded — *"a filter rather than an unload, deliberately: the
job queue runs two workers, and unloading would mean mutating the table around a
request"* — so in a full suite the **real** metabolomics pack legitimately
shadows `scale` with the core's four variants **plus** `pareto`, and stays. A
guard demanding core equality would report that as a leak.

So the reader asks the two questions that separate a legitimate pack from a
test's residue:

1. **No contributor whose origin names a test may survive.** Real packs register
   as `metabolomics_pack`; every fake in the tree registers as
   `fake_metabolomics_pack`. A `fake_` origin in the live table is a test's
   fixture that did not clean up, and there is no reading of it that is correct.
2. **Every core operation still offers at least the variants core offers.** A
   real shadow ADDS (`("standard","robust","minmax","none") + ("pareto",)`); the
   fake shadow REPLACES, dropping three of core's four. Core's variants are
   asked of `recipes` rather than copied into this file, because a hand-copied
   list is a second implementation that goes stale the day core grows a variant.

## What this is and is not load-bearing for — measured, not assumed

See the two probes recorded in the ledger note on `TEST-063`. The short form:
this guard is red when the `table` fixture's `finally: R.restore(state)` is
removed, and green when the inner `try/finally` inside
`test_a_pack_cannot_shadow_a_core_operation_silently` is removed — because the
fixture already restores what that inner block restores. The inner block is
defense in depth, not the load-bearing repair, and saying so is the point of
running the probe rather than reasoning about it.
"""
from __future__ import annotations

import os
import subprocess
import sys
from typing import Dict, List, Tuple

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbotab import recipes as R                                     # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Everything in the tree that registers a contributor into the process-global
#: table. Named rather than discovered: a grep for `register_operation` matches
#: the word inside a docstring, which is the matcher-fires-on-prose failure this
#: project keeps meeting one level down from wherever it is looking.
#:
#: THE SECOND ENTRY IS NODES AND NOT THE FILE, AND THE COST IS THE REASON.
#: `turbotab/test_a_pack_does_not_fire_on_the_wrong_data.py` runs in **226s**
#: whole and **0.86s** for the two nodes that touch the registry; the rest of
#: that file is pack-detector drives that never call `register_*`. Adding four
#: minutes to a suite `AUDIT-040` already records as over budget, to re-run
#: tests that cannot produce the residue, is a cap worth stating rather than
#: paying. The first entry is the whole file because it costs 0.83s and every
#: test in it takes the mutating `table` fixture.
MUTATING_TARGETS: Tuple[str, ...] = (
    "turbotab/test_the_recipe_table_is_a_table.py",
    "turbotab/test_a_pack_does_not_fire_on_the_wrong_data.py"
    "::test_a_pack_variant_preference_lives_in_the_recipe_table",
    "turbotab/test_a_pack_does_not_fire_on_the_wrong_data.py"
    "::test_loading_a_pack_twice_registers_it_once",
)

READER = (__file__.replace(ROOT + os.sep, "").replace(os.sep, "/")
          + "::test_no_test_fixtures_contributor_is_registered_here")


def _core_table() -> Dict[str, Tuple[str, ...]]:
    """Core's own operations, ASKED of `recipes` and put straight back.

    The live table is captured first and restored in a `finally`, so asking this
    question never becomes an instance of the defect the file is about.
    """
    saved = R.snapshot()
    try:
        R._install_core()
        return {op.key: tuple(op.variants) for op in R.operations()}
    finally:
        R.restore(saved)


def test_no_test_fixtures_contributor_is_registered_here():
    """The READER. Meaningful alone, and the observation point when run after.

    Alone this passes trivially — which is correct, and is why it is never run
    alone as evidence. Its job is to be the second node in
    :func:`test_the_registry_survives_the_files_that_rewrite_it`.
    """
    live_ops, live_defaults, _divergence = R.snapshot()

    leaked = sorted(
        {op.origin for op in live_ops.values()
         if "fake" in str(op.origin or "").lower()}
        | {d.origin for d in live_defaults
           if "fake" in str(d.origin or "").lower()})
    assert not leaked, (
        f"a test's fake contributor is still registered in this process: "
        f"{leaked}. `recipes._OPERATIONS` and `_DEFAULTS` are module-level, so "
        f"whichever test registered this left it for every test that runs "
        f"after it — and the cost lands on whoever next reads the table. "
        f"TEST-063.")

    core = _core_table()
    # POSITIVE CONTROL — core is not empty, so "every core operation is intact"
    # is a fact about the table rather than about a vacuous loop.
    assert core, "recipes._install_core() registered nothing; the reader has no subject"

    thinned: List[str] = []
    for key, variants in core.items():
        try:
            live = R.operation(key)
        except Exception:
            thinned.append(f"{key} is not registered at all")
            continue
        lost = sorted(set(variants) - set(live.variants))
        if lost:
            thinned.append(
                f"{key} lost {lost} (live: {tuple(live.variants)}, "
                f"core: {variants}, origin: {live.origin!r})")
    assert not thinned, (
        "a core operation offers FEWER variants than core does, so something "
        "replaced it rather than extending it:\n  " + "\n  ".join(thinned) +
        "\n\nA real pack shadows by copying core's variants and adding to "
        "them (`packs._metabolomics_recipes`). Dropping one is a test's fake "
        "pack that was not put back. TEST-063.")


@pytest.mark.timeout(300)
def test_the_registry_survives_the_files_that_rewrite_it():
    """The RUNNER: every mutating file, then the reader, in ONE interpreter.

    `-p no:randomly` because the order is the whole instrument — a reader that
    may run first is not a reader.
    """
    for target in MUTATING_TARGETS:
        rel = target.split("::")[0]
        assert os.path.exists(os.path.join(ROOT, rel)), (
            f"{rel} is named as a file that rewrites the recipe registry and "
            f"does not exist; this guard has lost its subject")

    out = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:randomly",
         *MUTATING_TARGETS, READER],
        cwd=ROOT, capture_output=True, text=True, timeout=280)
    text = out.stdout + out.stderr

    # A RENAMED NODE MUST NOT READ AS A PASS. pytest reports an unresolvable
    # node id as `ERROR: not found` and exits non-zero, so it is caught by the
    # returncode below — but it is named here so the message says which failure
    # this is rather than leaving a reader to work it out from a traceback.
    assert "not found:" not in text, (
        f"one of the named targets no longer resolves, so this guard ran "
        f"against fewer mutating tests than it claims:\n{text[-2000:]}")

    # POSITIVE CONTROL — the reader was actually collected and run. A node id
    # that stops resolving would otherwise turn this guard into a check that
    # the mutating files pass, which is a different and already-covered claim.
    assert " passed" in text and "no tests ran" not in text, (
        f"the ordered run collected nothing:\n{text[-3000:]}")

    assert out.returncode == 0, (
        "running the registry-mutating files and then a reader IN ONE PROCESS "
        "failed. If the failure is the reader, a test left a fake contributor "
        "in `recipes._OPERATIONS` / `_DEFAULTS` — which is invisible to that "
        f"test and to any run of it alone. TEST-063.\n\n{text[-4000:]}")
