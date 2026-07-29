"""The revert-probe harness, with reason checking.

The L16 rule extension: a probe must verify the REASON for failure, not merely
that it failed. So each revert declares `expect` — a fragment that must appear
in the failure output. A revert that turns the test red for an unrelated reason
(import error, fixture blowup, a different assertion) is reported as a MISS, not
as a pass, because that is the probe verifying nothing.

Usage — from the repository root:

    import sys; sys.path.insert(0, "docs/turbotab/tools")
    from revertprobe import run_probes
    run_probes("path/to/test.py::test_name", [
        (relative_path, old_text, new_text, expected_failure_fragment),
    ])

Three ways this reports, and they are deliberately distinct:

    RED, for '<reason>'       the revert broke the test, for the stated reason
    RED FOR THE WRONG REASON  the test broke, but not the way the probe claimed
    GREEN - NOT LOAD-BEARING  the revert changed nothing the test can see
    ANCHOR ERROR              `old` is not in the file exactly once

The last one matters as much as the others. An anchor spanning an implicit line
continuation matches nothing, so the revert is a no-op and the test stays green
- which reads as "not load-bearing" when the truth is "the probe was broken".

See FEATURE_PARITY.md, "a probe must verify the REASON for failure".
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]


def _run(test: str):
    # PYTHONDONTWRITEBYTECODE, and it is not hygiene.
    #
    # A probe writes a file, runs pytest, restores the file and runs pytest
    # again — three writes inside one second. CPython's source-mtime cache has
    # one-second granularity, so the restored run can import the REVERTED
    # bytecode and report `RESTORED: FAIL`. That reads as *"the fix does not
    # work"* when the fix is on disk and correct, which is the most misleading
    # thing this harness can say: it turns a verified repair into an apparent
    # regression at the exact moment somebody is deciding whether to close a
    # row.
    #
    # Observed at L22 on `GUIDED-033`. Writing no bytecode at all costs a
    # fraction of a second per run and removes the race.
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    p = subprocess.run(
        [sys.executable, "-m", "pytest", test, "-q", "--no-header", "-x",
         "-p", "no:cacheprovider"],
        cwd=ROOT, capture_output=True, text=True, env=env)
    return p.returncode, p.stdout + p.stderr


def run_probes(test: str, reverts) -> bool:
    rc, out = _run(test)
    print(f"UNREVERTED: {'PASS' if rc == 0 else 'FAIL'}")
    if rc != 0:
        print(out[-3000:])
        return False

    ok = True
    for rel, old, new, expect in reverts:
        f = ROOT / rel
        src = f.read_text()
        n = src.count(old)
        if n != 1:
            print(f"  ANCHOR ERROR {rel}: appears {n}x, not once")
            ok = False
            continue
        f.write_text(src.replace(old, new))
        try:
            rc, out = _run(test)
        finally:
            f.write_text(src)
        label = old.strip().splitlines()[0][:52]
        if rc == 0:
            print(f"  GREEN — NOT LOAD-BEARING  {rel}: {label!r}")
            ok = False
        elif expect not in out:
            print(f"  RED FOR THE WRONG REASON  {rel}: {label!r}")
            print(f"      expected {expect!r} in the failure; it was not there")
            tail = [l for l in out.splitlines() if l.startswith("E ")][:6]
            print("      " + "\n      ".join(tail))
            ok = False
        else:
            print(f"  RED, for {expect!r}   {rel}: {label!r}")

    rc, _ = _run(test)
    print(f"RESTORED: {'PASS' if rc == 0 else 'FAIL'}")
    return ok and rc == 0
