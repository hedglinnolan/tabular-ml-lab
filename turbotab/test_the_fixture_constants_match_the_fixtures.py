"""L50 — `TEST-056`. The constants and the fixtures are measured from each other.

**A near-miss, and the reason it is worth a guard.**
`sample_data/make_genomics_siblings.py` draws from **one** seeded generator in
file order. The estimated-counts block was inserted between the CPM block and
the TMM block *after* the first measurement run, which shifted every subsequent
draw — so the library-size spread of `genomics_tmm_cpm.csv` and
`genomics_fpkm.csv` changed and the numbers handed to the L50-B subagent were
from before the insertion. It said 0.0635 and 0.102; the truth is 0.0524 and
0.1128.

**Nothing shipped wrong**, because the `.md` companions never quoted a
coefficient of variation and both subagents re-derived rather than trusting the
brief. That is the project's own rule working. But the near-miss names a real
property: **a seeded generator is reproducible only if nothing is inserted
upstream of the draw you care about**, and `packs.py` now holds two constants —
`_ESTIMATED_COUNTS_CV` and `_FPKM_CV` — whose values were measured off these
files. If the generator is edited again and the fixtures are regenerated, those
constants become claims about matrices that no longer exist, and the
classification that rests on them would drift silently.

So this asserts the two things that would go wrong, in the order they would:

1. **The generator is deterministic** — running it twice writes byte-identical
   files. Without this the rest is untestable.
2. **The constants still describe the fixtures they were measured from**, and
   the separator they form still separates.

It deliberately does **not** assert the constants' exact values. Pinning them
would make this a copy of `packs.py` rather than a check on it — the same
two-implementations failure the pack layer exists to avoid — and it would go red
on a legitimate re-measurement instead of on a drift.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
DATA = HERE / "sample_data"
GENERATOR = DATA / "make_genomics_siblings.py"

#: The metadata columns the generator carries through untouched.
META = ("sample_id", "batch", "sex", "age", "condition")

#: Every matrix the generator writes, and nothing else — derived from the
#: generator's own `_write` calls rather than listed, so a sibling added next
#: loop is covered without anyone remembering to add it here.
def _generated() -> list:
    source = GENERATOR.read_text(encoding="utf-8")
    import re
    return sorted(set(re.findall(r'_write\(out, "([^"]+)"', source)))


def _library_cv(name: str) -> float:
    frame = pd.read_csv(DATA / name)
    genes = [c for c in frame.columns
             if c not in META and frame[c].dtype.kind in "iuf"]
    sums = frame[genes].to_numpy(float).sum(axis=1)
    return float(sums.std(ddof=0) / sums.mean())


def test_the_generator_writes_every_sibling_it_claims_to():
    """The positive control, before anything is measured off them."""
    names = _generated()
    assert len(names) >= 5, f"only {len(names)} siblings are generated: {names}"
    for name in names:
        assert (DATA / name).exists(), f"{name} is generated and not on disk"
        assert (DATA / f"{name}.md").exists(), (
            f"{name} ships without the companion that says how it was derived, "
            f"which is the only thing separating a derived fixture from an "
            f"invented one")


def test_running_the_generator_twice_writes_the_same_bytes(tmp_path):
    """Determinism, which everything below rests on.

    A seeded generator that is not reproducible makes every constant measured
    from it a statement about one particular run.

    **RUN IN A SANDBOX, NOT IN `sample_data/`. `TEST-098`.** This used to spawn
    the generator in place, and the generator truncates and rewrites **twelve
    git-tracked files** — six matrices and six `.md` companions, 1,461,182
    bytes — over the 0.421 s it runs. Six concurrent readers doing exactly what
    the six cross-file readers of these fixtures do logged **56 corrupted
    observations in 261 reads**: 55 `EmptyDataError`, and one silent 59-row
    frame where 60 rows are committed. The silent short read is the one to
    fear — it does not raise, it computes statistics on a partial matrix. The
    tree never drifted, because the rewrite is byte-identical; only the disk
    moved, mid-run, which is why two full sweeps never saw it.

    The generator resolves its output directory from its own `__file__`, so a
    copy of it beside a copy of its one input writes into the copy's directory
    and touches nothing tracked. No change to the generator was needed.

    **And the assertion is STRICTLY STRONGER than the one it replaces.** The
    old form compared run N against run N+1, which a generator that is
    reproducibly *wrong* satisfies. This compares a fresh run against the
    **committed** bytes, so it also fails when the generator stops producing
    what ships.
    """
    names = _generated()
    sandbox = tmp_path / "sample_data"
    sandbox.mkdir()
    shutil.copy2(GENERATOR, sandbox / GENERATOR.name)
    shutil.copy2(DATA / "genomics_expression.csv",
                 sandbox / "genomics_expression.csv")

    out = subprocess.run([sys.executable, str(sandbox / GENERATOR.name)],
                         capture_output=True, text=True, cwd=str(tmp_path))
    assert out.returncode == 0, out.stderr[-1500:]

    # THE CONTROL, and it is what proves the redirect worked rather than
    # assuming it: if the generator had written to `sample_data/` after all,
    # nothing would be here and the absence would read as a pass.
    produced = sorted(p.name for p in sandbox.glob("*.csv")
                      if p.name != "genomics_expression.csv")
    assert produced == sorted(names), (
        f"the sandboxed generator wrote {produced}, not {sorted(names)}. If "
        f"this is empty it wrote into the real fixture directory instead, "
        f"which is the whole defect this sandbox exists to remove")

    committed = {n: hashlib.sha256((DATA / n).read_bytes()).hexdigest()
                 for n in names}
    fresh = {n: hashlib.sha256((sandbox / n).read_bytes()).hexdigest()
             for n in names}
    drifted = sorted(n for n in names if committed[n] != fresh[n])
    assert not drifted, (
        f"the generator no longer reproduces the committed {drifted}. The seed "
        f"is not the only thing deciding these files, so every number measured "
        f"off them — including `packs._ESTIMATED_COUNTS_CV` and `_FPKM_CV` — "
        f"is a claim about one run rather than about the fixture")

    # The companions travel with the matrices and are tracked too, so they are
    # part of what "the generator still produces what ships" means.
    companions = sorted(n for n in names
                        if (DATA / f"{n}.md").read_bytes()
                        != (sandbox / f"{n}.md").read_bytes())
    assert not companions, (
        f"the matrices reproduce and their companions do not: {companions}. "
        f"The `.md` beside each fixture is what separates a derived fixture "
        f"from an invented one")


def test_the_estimated_counts_and_fpkm_separator_is_measured():
    """`TEST-056`. The two constants still describe the files they came from.

    The classification of a non-integer, non-negative matrix with varying row
    sums turns on this one number, because `GENOMICS_PACK.md` §02 separates
    estimated counts from FPKM only by max and skew — which overlap. The
    measured separator is library-size spread, and if the fixtures move under
    the constants the split moves with them, silently.
    """
    from turbotab import packs

    est = _library_cv("genomics_estimated_counts.csv")
    fpkm = _library_cv("genomics_fpkm.csv")

    assert est > fpkm * 1.5, (
        f"the two fixtures no longer separate: estimated counts {est:.4f} "
        f"against FPKM {fpkm:.4f}. §02's own table cannot tell them apart, so "
        f"this spread is the whole of the distinction")

    declared_est = getattr(packs, "_ESTIMATED_COUNTS_CV", None)
    declared_fpkm = getattr(packs, "_FPKM_CV", None)
    # `AUDIT-039`, swept from `TEST-059`'s class. THIS WAS A CONDITIONAL SKIP
    # over the exact condition the test exists to detect: pytest counts a skip
    # as not-a-failure, so the regression would have silenced the guard
    # instead of turning it red.
    assert declared_est is not None and declared_fpkm is not None, (
        "`packs._ESTIMATED_COUNTS_CV` / `packs._FPKM_CV` are gone. The whole "
        "of this test is that the classifier's constants and the fixtures they "
        "were measured from agree; a classifier that no longer holds them by "
        "name has not made the check unnecessary, it has made it unanswerable")

    # A tolerance rather than equality, and the reason is in the docstring: an
    # honest re-measurement should not fail this, and a regenerated fixture
    # that moved should.
    assert abs(declared_est - est) < 0.02, (
        f"`packs._ESTIMATED_COUNTS_CV` is {declared_est} and the fixture it "
        f"was measured from now reads {est:.4f}. The generator was edited and "
        f"the constant was not re-derived")
    assert abs(declared_fpkm - fpkm) < 0.02, (
        f"`packs._FPKM_CV` is {declared_fpkm} and the fixture reads "
        f"{fpkm:.4f}")
    midpoint = (declared_est + declared_fpkm) / 2
    assert fpkm < midpoint < est, (
        f"the split at {midpoint:.4f} no longer sits between the two fixtures "
        f"({fpkm:.4f} and {est:.4f}), so one of them classifies as the other")


def test_the_companions_do_not_quote_a_number_that_can_drift():
    """Why nothing shipped wrong, asserted so it stays true.

    The `.md` files describe the derivation and state qualitative properties —
    *sums cluster near 1e6 and none equals it* — and quote no coefficient of
    variation. That is what kept them right when the generator moved under
    them, and it is a property worth keeping rather than a coincidence.
    """
    for name in _generated():
        prose = (DATA / f"{name}.md").read_text(encoding="utf-8")
        lowered = prose.lower()
        for phrase in ("sumcv", "coefficient of variation", "library-size cv"):
            assert phrase not in lowered, (
                f"{name}.md quotes {phrase!r}. A companion that states a "
                f"measured constant ages the moment the generator is edited — "
                f"which is `TEST-056` exactly, and the companions are the "
                f"reason it was a near-miss rather than a defect")
