"""The suite must be green for reasons other than the order it happens to run in.

pytest-randomly is not installed here, so `-p no:randomly` was a no-op and the
suite had only ever run in one fixed order. Shuffling it exposed five test
modules sharing a module-level `RNG = np.random.RandomState(0)` that was never
reseeded: every test drew from one advancing stream, so a test's DATA depended
on how many tests ran before it. tests/test_stress_regressions.py alone has 103
tests drawing from a single stream. Two tests genuinely failed under other
orderings.

Each of those modules now reseeds before every test. This file guards the
property rather than the mechanism, so a new module that reintroduces a shared
stream is caught here.

To shuffle by hand:
    SHUFFLE_SEED=3 PYTHONPATH=tests python -m pytest tests/ \
        --ignore=tests/workflow -q -p shuffle_plugin

tests/workflow is excluded there on purpose: its test_step1..test_step9 suites
deliberately share state within a class, so shuffling breaks them by design.
"""
from __future__ import annotations

import ast
import glob
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _modules_with_a_shared_rng():
    out = []
    for path in glob.glob(os.path.join(ROOT, "tests", "**", "test_*.py"),
                          recursive=True):
        src = open(path, encoding="utf-8").read()
        tree = ast.parse(src)
        shares = any(
            isinstance(node, ast.Assign)
            and any(getattr(t, "id", "") == "RNG" for t in node.targets)
            for node in tree.body                      # module level only
        )
        if shares:
            out.append((os.path.relpath(path, ROOT), src))
    return out


def test_every_shared_rng_is_reseeded_per_test():
    offenders = [
        rel for rel, src in _modules_with_a_shared_rng()
        if "RNG.seed(" not in src
    ]
    assert not offenders, (
        "these modules share one advancing RNG across their tests, so each "
        "test's data depends on how many ran before it:\n  "
        + "\n  ".join(offenders)
        + "\n\nAdd an autouse fixture that calls RNG.seed(0)."
    )


def test_the_reseed_is_autouse_so_it_cannot_be_forgotten():
    missing = [
        rel for rel, src in _modules_with_a_shared_rng()
        if "RNG.seed(" in src and "autouse=True" not in src
    ]
    assert not missing, (
        f"reseeding exists but is not autouse in: {missing} — a new test that "
        f"forgets to request the fixture is order-dependent again"
    )
