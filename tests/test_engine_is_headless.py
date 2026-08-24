"""L7's overall gate: every core module imports with Streamlit blocked.

`ARCHITECTURE.md` §01 counts the tainted modules by importing each one with
`streamlit` forced to fail. This is that census, as a test that runs in CI —
because a count in a document goes stale the first time someone adds an import,
and the whole point of the number is that it is going to zero.

Two things make it non-vacuous, both learned the hard way earlier in this work:

* a stub `streamlit` is put on the path first, so `streamlit` is genuinely
  importable and blocking it means something. Without that, a machine with no
  Streamlit installed passes while proving nothing;
* the blocker is asserted to actually bite before any core module is imported.
  It uses `find_spec`; the `find_module`/`load_module` protocol printed in older
  revisions of the doc was removed from the import system in Python 3.12 and
  blocks nothing on a current interpreter.

**Module-level imports only.** A lazy `import streamlit` inside a function is
not import-time taint — `ml.model_coach` was miscounted as tainted for exactly
that reason. It is still coupling, and it still has to go before the module can
run headless *at call time*, but it is a different problem with a different fix,
and conflating them is what produced the wrong census.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The engine. `ml/` and `models/` plus the two single-file modules the census
# counts as core. `utils/` is deliberately absent: most of it is the host.
CORE_PACKAGES = ("ml", "models")
CORE_SINGLE_MODULES = ("visualizations", "data_processor")

# Modules in `utils/` that the engine depends on and that must therefore also
# import clean. These are the record and state pieces L7 detaints.
CORE_UTILS = (
    "utils.insight_ledger",
    "utils.workflow_provenance",
    "utils.test_lockbox",
)


def _core_module_names():
    names = []
    for pkg in CORE_PACKAGES:
        base = os.path.join(ROOT, pkg)
        for fn in sorted(os.listdir(base)):
            if fn.endswith(".py") and not fn.startswith("_"):
                names.append(f"{pkg}.{fn[:-3]}")
    names.extend(CORE_SINGLE_MODULES)
    names.extend(CORE_UTILS)
    return names


CORE_MODULES = _core_module_names()


def _import_all_with_streamlit_blocked(tmp_path, modules):
    """Import each module in a fresh interpreter with `streamlit` blocked."""
    stub = tmp_path / "stub"
    stub.mkdir(exist_ok=True)
    (stub / "streamlit.py").write_text("MARKER = 'stub streamlit'\n", encoding="utf-8")

    script = textwrap.dedent(f"""
        import importlib, importlib.util, json, sys
        sys.path.insert(0, {str(stub)!r})
        sys.path.insert(0, {ROOT!r})

        assert importlib.util.find_spec("streamlit") is not None, "stub unreachable"

        class Blocker:
            def find_spec(self, name, path=None, target=None):
                if name == "streamlit" or name.startswith("streamlit."):
                    raise ImportError("BLOCKED: " + name)
                return None
        sys.meta_path.insert(0, Blocker())

        try:
            import streamlit
            raise SystemExit("blocker did not block")
        except ImportError as e:
            assert "BLOCKED" in str(e), e

        results = {{}}
        for name in {modules!r}:
            try:
                importlib.import_module(name)
                results[name] = None
            except Exception as exc:
                results[name] = f"{{type(exc).__name__}}: {{exc}}"
        print("@@" + json.dumps(results))
    """)
    proc = subprocess.run([sys.executable, "-c", script],
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, f"stdout={proc.stdout[-2000:]}\nstderr={proc.stderr[-2000:]}"
    line = [l for l in proc.stdout.splitlines() if l.startswith("@@")][-1]
    return json.loads(line[2:])


@pytest.fixture(scope="module")
def import_results(tmp_path_factory):
    return _import_all_with_streamlit_blocked(
        tmp_path_factory.mktemp("blocked"), CORE_MODULES)


def test_the_census_is_not_empty():
    """A gate over zero modules passes for the wrong reason."""
    assert len(CORE_MODULES) >= 40, (
        f"only {len(CORE_MODULES)} core modules found — the scan lost its tree")


def test_no_core_module_needs_streamlit_to_import(import_results):
    """The gate. Every core module imports with Streamlit blocked.

    A failure here names the module and the error, so the fix is obvious: either
    the import is dead and comes out, or the module is doing host work and needs
    splitting into logic and delivery.
    """
    blocked = {name: err for name, err in import_results.items()
               if err and "BLOCKED" in err}
    assert not blocked, (
        f"{len(blocked)} of {len(CORE_MODULES)} core modules cannot import "
        f"without Streamlit:\n" + "\n".join(f"  {k}: {v}" for k, v in sorted(blocked.items())))


def test_core_modules_import_for_any_reason(import_results):
    """Separate from the gate above: a module that fails to import for some
    *other* reason (a missing optional dependency, a syntax error) is also
    broken, but it is not a Streamlit-coupling problem and should not be
    reported as one."""
    other = {name: err for name, err in import_results.items()
             if err and "BLOCKED" not in err
             # Optional extras the repo documents as not installed by default.
             and not any(dep in err for dep in
                         ("shap", "umap", "giotto", "gtda", "decision_curve"))}
    assert not other, (
        "core modules failed to import for non-Streamlit reasons:\n"
        + "\n".join(f"  {k}: {v}" for k, v in sorted(other.items())))
