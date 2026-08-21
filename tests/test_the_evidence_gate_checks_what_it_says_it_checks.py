"""L64-D1. `TEST-107` — the evidence gate reported `ok` while checking half of
what it claimed.

Two `continue` pre-filters skipped any module whose text lacked `Claim(` or
`Evidence(`. Break both literals and the gate checks **32 of 67 claims and 0 of
51 module constants**, prints `ok`, and exits `0`. That is a false green in one
of the six pre-commit gates, and it is `AGENT_ONBOARD.md` §07 trap 5c's third
and worst shape: a silent pre-filter with no assertion behind it, where the
counts are printed and never checked.

## Why a FLOOR and not a ceiling

`TEST-107`'s `act` suggests bounding the number of modules skipped. That is the
wrong instrument. The quantity is 48 and 47 of 52, and it **grows every time an
unrelated module is added**, so the gate would go red for reasons that have
nothing to do with claims going unchecked. `repo_write_guard`'s ceiling bounds
destinations the instrument *knows* it cannot see — a quantity that only grows
when coverage genuinely shrinks. Different quantity, different instrument.

**And a zero-guard would not have caught this**: 32 of the 67 claims come from
the unfiltered figure-registry loop, so breaking the filter left 32, not 0. The
floors are per-walk for exactly that reason.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "docs", "turbotab", "tools"))

import evidence                                                # noqa: E402

#: Measured at `L64`. Floors, not equalities — a claim added next loop must not
#: turn this red, and a claim silently going unchecked must.
CLAIM_FLOOR = 30
CONSTANT_FLOOR = 45


def test_the_two_walks_are_callable_and_return_what_the_gate_counts():
    """`call_sites()` made this move first, and its docstring says why:
    *"A gate whose only interface is stdout is a gate a test can only check the
    description of."*
    """
    claims = evidence.module_claims()
    constants = evidence.module_constants()
    assert isinstance(claims, list) and isinstance(constants, list)
    for label, source in claims + constants:
        assert isinstance(label, str) and label
        assert isinstance(source, str) and source


def test_the_gate_checks_a_floor_of_claims_and_constants(capsys):
    """The assertion the printed counts could never be.

    These are the two walks the pre-filters gated. If either enumeration breaks
    the count collapses and this goes red — where before, the gate printed a
    smaller number in its success line and exited 0.
    """
    claims = evidence.module_claims()
    constants = evidence.module_constants()
    assert len(claims) >= CLAIM_FLOOR, (
        f"the module-constant claim walk found {len(claims)} claims against a "
        f"floor of {CLAIM_FLOOR}. The gate will still print `ok`; that is the "
        f"defect this floor exists to catch")
    assert len(constants) >= CONSTANT_FLOOR, (
        f"the module-constant Evidence walk found {len(constants)} against a "
        f"floor of {CONSTANT_FLOOR}")
    with capsys.disabled():
        print(f"\n  {len(claims)} module claims · {len(constants)} module "
              f"constants, both above their floors")


def test_the_floors_go_red_when_the_walk_is_pointed_at_nothing():
    """The positive control, driven rather than argued.

    An enumeration that returns nothing is exactly what a broken pre-filter
    produced, and it must be distinguishable from a clean corpus. Pointing the
    walk at an empty module list is that state, reachable without editing
    anything.
    """
    assert evidence.module_claims(modules=[]) == []
    assert evidence.module_constants(modules=[]) == []
    assert len(evidence.module_claims(modules=[])) < CLAIM_FLOOR
    assert len(evidence.module_constants(modules=[])) < CONSTANT_FLOOR


def test_the_pre_filters_are_gone():
    """The literals themselves, because their absence is the fix.

    Kept as a source assertion rather than a behavioral one because a
    reintroduced filter would be *invisible* behaviorally — the counts are
    identical on a healthy corpus, which is precisely how this survived.
    """
    source = open(os.path.join(ROOT, "docs", "turbotab", "tools",
                               "evidence.py"), encoding="utf-8").read()
    # CODE ONLY, THROUGH THE GATE'S OWN HELPER. The comment explaining why the
    # filters were removed necessarily QUOTES them, so a raw substring search
    # fires on the explanation — a matcher firing on prose, in the test for a
    # matcher firing on prose (trap 5b, and it caught me here). `_code_only`
    # tokenizes and blanks every string and comment, which is exactly the
    # distinction this needs and is already the gate's own instrument.
    code = evidence._code_only(source)
    assert "def module_claims" in code, "the helper blanked the code as well"
    for gone in ('if "Claim(" not in path.read_text',
                 'if "Evidence(" not in path.read_text'):
        assert gone not in code, (
            f"a text pre-filter is back: {gone!r}. Breaking its literal makes "
            f"the gate check less and still report ok")


def test_the_gate_still_passes_and_still_reports_its_counts():
    """End to end, because the refactor must not change the answer.

    Measured before and after: byte-identical counts, +0.65 s on a 52-module
    corpus where only 4 and 5 modules carried the literals.
    """
    out = subprocess.run(
        [sys.executable, os.path.join(ROOT, "docs", "turbotab", "tools",
                                      "evidence.py"), "check"],
        capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0, out.stdout[-2000:]
    assert "claims" in out.stdout and "module constants" in out.stdout
    claims = evidence.module_claims()
    # The printed total includes the figure-registry loop, which was never
    # filtered — so the printed number is strictly larger than this walk's, and
    # asserting equality here would be wrong.
    assert len(claims) < 67 or f"{len(claims)}" in out.stdout
