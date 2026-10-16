"""The claims registry stands alone (CLAIMS_REGISTRY.md §1.1; R0_PLAN.md R0a acceptance 6).

Importing ``turbotab.core.claims``, and loading its claims modules, loads no other
``turbotab.core`` module: a fresh interpreter is asked which modules it holds afterward. The claims
modules are found by a package scan, so no list of them exists: a public module in a scanned
package is imported, a private one is not.
"""
from __future__ import annotations

import json
import subprocess
import sys

from turbotab.core import claims
from turbotab.core.claims import Registry

REPO = claims.REPO


def _loaded_after(*statements: str) -> list[str]:
    probe = "\n".join(["import json, sys", *statements, 'print(json.dumps(sorted(m for m in '
                        'sys.modules if m.startswith("turbotab"))))'])
    out = subprocess.run([sys.executable, "-c", probe], cwd=REPO, capture_output=True, text=True,
                         timeout=120, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def _engine(modules: list[str]) -> list[str]:
    return [m for m in modules if m.startswith("turbotab.core.")
            and m != "turbotab.core.claims" and not m.startswith("turbotab.core.claims.")]


def test_importing_the_claims_package_loads_no_other_core_module():
    modules = _loaded_after("import turbotab.core.claims")
    assert "turbotab.core.claims" in modules
    assert _engine(modules) == []


def test_loading_the_default_registry_loads_no_other_core_module():
    modules = _loaded_after("import turbotab.core.claims as c", "c.registry()")
    assert _engine(modules) == []


def test_the_probe_sees_an_engine_import_when_there_is_one():
    """The probe itself is tested the other way: importing an engine module is seen."""
    modules = _loaded_after("import turbotab.core.claims", "import turbotab.core.contracts")
    assert _engine(modules) == ["turbotab.core.contracts"]


def test_the_scan_imports_every_public_module_and_no_private_one(tmp_path, monkeypatch):
    package = tmp_path / "planted_claims"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "trials.py").write_text("SCANNED = True\n", encoding="utf-8")
    (package / "tuning.py").write_text("SCANNED = True\n", encoding="utf-8")
    (package / "_private.py").write_text("raise RuntimeError('a private module was scanned')\n",
                                         encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        assert claims.scan("planted_claims") == ["planted_claims.trials", "planted_claims.tuning"]
        assert sys.modules["planted_claims.trials"].SCANNED
        assert "planted_claims._private" not in sys.modules
    finally:
        for name in [m for m in sys.modules if m.startswith("planted_claims")]:
            del sys.modules[name]


def test_the_default_registry_holds_no_claim_yet():
    """R0a ships the registry with no real claim in it; R0b registers the first ones."""
    assert isinstance(claims.registry(), Registry)
    assert len(claims.registry()) == 0
    assert claims.scan("turbotab.core.claims") == []
