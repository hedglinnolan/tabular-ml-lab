"""The legacy modules kept for Classic and turbotab/core keep their tests (BLUEPRINT §9.1).

When the legacy app was retired, §9.1 kept the legacy domain modules that Classic or
``turbotab/core`` imports, until they are absorbed into ``turbotab/core``. Their tests stay with
them. The first version of the retirement deleted every legacy test file that reached the retired
app anywhere, even when most of the file's tests never did. That left 12 of the 32 kept modules
with no test at all: ``clinical``, ``survey``, ``training``, ``repairs``, ``attention`` and seven
more. This checks that every kept module is imported by some test, either a legacy test beside it
or one of Classic's tests.
"""
from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
LEGACY = REPO / "turbotab"


def kept_modules() -> list[str]:
    # ``replay.py`` is v2's, and its tests are in turbotab/core/tests.
    return sorted(p.stem for p in LEGACY.glob("*.py")
                  if not p.stem.startswith("test_") and p.stem not in ("__init__", "replay"))


def imported(source: str) -> set[str]:
    """The ``turbotab.<module>`` names a file imports, at module level or inside a function."""
    found = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            if node.module == "turbotab":
                found |= {alias.name for alias in node.names}
            elif node.module.startswith("turbotab."):
                found.add(node.module.split(".")[1])
        elif isinstance(node, ast.Import):
            found |= {alias.name.split(".")[1] for alias in node.names
                      if alias.name.startswith("turbotab.")}
    return found


def modules_under_test() -> set[str]:
    tests = [*LEGACY.glob("test_*.py"), *(REPO / "tests").rglob("test_*.py")]
    return set().union(*(imported(t.read_text("utf-8")) for t in tests))


def test_every_kept_legacy_module_is_imported_by_a_test():
    untested = sorted(set(kept_modules()) - modules_under_test())
    assert not untested, (
        f"no test imports {untested}. A legacy module stays beside turbotab/core because Classic or "
        "core imports it (BLUEPRINT §9.1), and its tests stay with it. Restore them from git, without "
        "the tests that reached the retired app.")


def test_the_scan_reads_imports_and_finds_the_modules():
    """The positive control: an empty module list or a parser that reads nothing passes the check above."""
    assert imported("from turbotab import packs as P, grain\nimport turbotab.survey\n"
                    "def f():\n    from turbotab.clinical import x\n") == {"packs", "grain", "survey", "clinical"}
    assert imported("from turbotab.core import seal\n") == {"core"}
    modules = kept_modules()
    assert len(modules) >= 25, modules
    assert {"packs", "clinical", "survey", "grain", "training"} <= set(modules)
    assert {"packs", "clinical", "survey", "grain", "training"} <= modules_under_test()
