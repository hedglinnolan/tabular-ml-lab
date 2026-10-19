"""The claims registry stands alone (CLAIMS_REGISTRY.md §1.1; R0_PLAN.md R0a acceptance 6).

Importing ``turbotab.core.claims``, and loading its claims modules, loads no other
``turbotab.core`` module: a fresh interpreter is asked which modules it holds afterward. The claims
modules are found by a package scan, so no list of them exists: a public module in a scanned
package is imported, a private one is not.

The research tools beside the app (``research/``, the decoy queue among them) may import the
registry, and no module under ``turbotab/`` imports them: each module's import statements are read
with ``ast``.

R0b's claims name their origin as their history shows (R1_PLAN.md R1-G): *audit* for the claims
that state one of the claims backtest's S1 findings (``notes/claims-backtest/FIXES.md``), as R0b
registered them, and *engine* for the splits of a shared constant and the fallbacks their regimes
needed, which R0b's builders wrote from the engine's own code. None is the swarm's.
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys

import pytest

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


R0B_IDS = {
    # inference (turbotab/core/claims/inference.py)
    "causal.numeric_exposure_r2", "designs.observational_default", "designs.sampled_by_outcome",
    "inference.few_clusters", "ordinal.cluster_robust_intervals", "results.tail_share",
    "survival.monotone_likelihood", "trials.few_clusters", "trials.few_clusters_extended",
    "trials.separation",
    # models (turbotab/core/claims/models.py)
    "baseline.min_gain", "boosted_trees.small_rows", "forest.small_rows",
    "materiality.collinear_under_prediction", "materiality.rows_smd_floor",
    "materiality.rows_smd_floor_prediction", "tuning.small_sample_standard",
    "tuning.small_sample_standard_classes", "tuning.small_sample_standard_estimate",
    "tuning.small_sample_no_search",
    # data (turbotab/core/claims/data.py)
    "data.clusters_adjust", "data.constant_column", "data.nhanes_weight", "data.omics_raw_scale",
    "data.tail_share", "data.ul_share", "data.ul_share_unread",
}


# R1-F's first batch (R1_PLAN.md R1-F): the claims backtest's S2 and S3 defects an analyst reads,
# each with the claims RF-1 registered for it, in the same three areas' modules.
BACKTEST_RF1 = {
    "A-106": {"modeling.no_resampling"},
    "A-117": {"missing.mi_refused_unless_inference"},
    "A-121": {"materiality.collinear_inside"},
    "C-110": {"seal.by_row_exploratory"},
    "C-144": {"adjustment.demographic_guess"},
    "C-241": {"explore.missing_by_group"},
    "D-001": {"energy.no_energy_no_swap"},
    "D-009": {"substitution.isocaloric_units"},
    "D-012": {"aggregation.replicate_mean"},
    "D-018": {"validation.resampling_first"},
    "D-021": {"validation.paired_interval"},
    "D-032": {"designs.adjustment_by_design"},
    "D-037": {"missing.ffq_blank_handling"},
    "D-044": {"repairs.tntc_censored", "repairs.tntc_beside_below_limit",
              "repairs.tntc_beside_unplaced"},
    "D-070": {"copies.pooled_by_rubin"},
}
RF1_IDS = set().union(*BACKTEST_RF1.values())


def test_the_default_registry_holds_r0bs_claims():
    """R0a shipped the registry with no real claim in it; R0b registers its three areas' claims,
    found by the package scan, and R1-F's first batch adds its own to the same modules, and no
    other."""
    assert isinstance(claims.registry(), Registry)
    assert claims.scan("turbotab.core.claims") == [
        "turbotab.core.claims.data", "turbotab.core.claims.inference",
        "turbotab.core.claims.models"]
    assert {c.id for c in claims.registry()} == R0B_IDS | RF1_IDS


# The S1 findings of the claims backtest (README.md's table, FIXES.md), each with the claims R0b
# registered for it.
BACKTEST_S1 = {
    "A-001": {"tuning.small_sample_standard"},
    "A-029": {"baseline.min_gain"},
    "A-102": {"ordinal.cluster_robust_intervals"},
    "A-124": {"materiality.rows_smd_floor"},
    "B-211": {"designs.observational_default", "designs.sampled_by_outcome"},
    "B-281": {"trials.separation"},
    "B-416": {"causal.numeric_exposure_r2"},
    "B-435": {"survival.monotone_likelihood"},
    "BX-040": {"data.ul_share"},
    "C-123": {"data.constant_column"},
    "C-143": {"data.nhanes_weight"},
    "C-170": {"materiality.collinear_under_prediction"},
    "C-193": {"data.omics_raw_scale"},
    "D-058": {"data.clusters_adjust"},
    "D-086": {"data.tail_share", "results.tail_share"},
}


def test_r0bs_claims_are_audit_where_they_state_a_backtest_finding_and_engine_otherwise():
    audit = set().union(*BACKTEST_S1.values())
    assert len(BACKTEST_S1) == 15 and len(audit) == 17
    assert len(BACKTEST_RF1) == 15 and len(RF1_IDS) == 17 and not RF1_IDS & R0B_IDS
    origin = {c.id: c.origin for c in claims.registry()}
    assert {i for i, o in origin.items() if o == "audit"} == audit | RF1_IDS
    assert {i for i, o in origin.items() if o == "engine"} == R0B_IDS - audit
    assert "swarm" not in origin.values()


def _imports_research(module: ast.Module) -> list[str]:
    """The ``research`` imports ``module`` makes, as written."""
    found = []
    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            found += [a.name for a in node.names if a.name.split(".")[0] == "research"]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module.split(".")[0] == "research":
                found.append(node.module)
    return found


def test_no_module_under_turbotab_imports_the_research_tools():
    found = {}
    for path in sorted((REPO / "turbotab").rglob("*.py")):
        if "node_modules" in path.parts:
            continue
        hits = _imports_research(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
        if hits:
            found[str(path.relative_to(REPO))] = hits
    assert found == {}


@pytest.mark.parametrize("source, seen", [
    ("import research\n", True),
    ("import research.registry.decoy_queue as queue\n", True),
    ("from research.registry import decoy_queue\n", True),
    ("from research import registry\n", True),
    ("import researcher\n", False),
    ("from turbotab.core import research_notes\n", False),
    ("from .research import notes\n", False),
    ("NAME = 'research'\n", False),
])
def test_the_scan_sees_a_research_import_and_nothing_else(source, seen):
    assert bool(_imports_research(ast.parse(source))) is seen
