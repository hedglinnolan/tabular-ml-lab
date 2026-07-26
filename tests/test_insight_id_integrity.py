"""CI scan: every referenced insight ID must be one the codebase produces.

The stale-identifier bug class has bitten twice: a resolution map referenced
'eda_skew_individual' (nothing produced it), and the sufficiency insights
checked enum values ('insufficient'/'borderline') that no longer existed.
Both failed silently — coaching/resolution logic that never fires.

This test extracts, via AST (no imports, no Streamlit runtime):
- PRODUCED ids: every `...Insight(id=...)` constructor call with a string
  or f-string id, across pages/, utils/, ml/.
- REFERENCED ids: string literals passed to ledger.resolve/acknowledge/
  get/remove calls (receiver name containing 'ledger'), plus the
  `_ACTION_TO_INSIGHT_MAP` exact/prefix entries.

It fails when a referenced id matches nothing the code can produce.
"""
import ast
import os

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Every directory that may produce or reference an insight id. `turbotab` is
# here because the migration moves code into it: a scanner that keeps looking
# only at the old directories keeps passing while the ids it was written to
# guard walk out from under it.
SCAN_DIRS = ["pages", "utils", "ml", "models", "turbotab"]

# Directories that must exist for this scan to mean anything. If one is renamed
# or emptied the scan silently shrinks, so absence is an error rather than a
# `continue`. This is the failure the whole file is about, applied to itself.
REQUIRED_SCAN_DIRS = ["pages", "utils", "ml"]

# Floors, not targets. They exist so that a scan which collapses to nothing —
# after a rename, a move, or an AST change that stops matching — fails loudly
# instead of reporting a clean bill of health over an empty set. Raise them when
# the real counts grow well past these; never lower them to make a run pass.
MIN_FILES_SCANNED = 40
MIN_PRODUCED_IDS = 30
MIN_REFERENCED_IDS = 5

LEDGER_METHODS = {"resolve", "acknowledge", "get", "remove"}

_SCANNED_FILES = []


def _iter_source_files():
    _SCANNED_FILES.clear()
    for d in SCAN_DIRS:
        base = os.path.join(PROJECT_ROOT, d)
        if not os.path.isdir(base):
            continue
        for root, _dirs, files in os.walk(base):
            # Virtualenvs and caches live under some of these directories and
            # are not this project's source.
            if any(part in root for part in (".venv", "venv", "__pycache__",
                                             "site-packages", "node_modules")):
                continue
            for fn in files:
                if fn.endswith(".py"):
                    path = os.path.join(root, fn)
                    _SCANNED_FILES.append(path)
                    yield path


def _receiver_mentions_ledger(node) -> bool:
    """True if the attribute call's receiver looks like a ledger object."""
    src = ast.dump(node)
    return "ledger" in src.lower()


def collect_ids():
    produced_exact, produced_prefixes = set(), set()
    referenced_exact, referenced_prefixes = {}, {}

    for path in _iter_source_files():
        rel = os.path.relpath(path, PROJECT_ROOT)
        # Explicit encoding: the default is cp1252 on Windows, and this tree
        # contains sources it cannot decode — without this the module raises
        # UnicodeDecodeError at *collection*, taking the whole tier-1 run with it.
        with open(path, encoding="utf-8") as f:
            try:
                tree = ast.parse(f.read())
            except SyntaxError:  # pragma: no cover
                pytest.fail(f"{rel} does not parse")

        for node in ast.walk(tree):
            # ── producers: SomethingInsight(id=...) ──
            if isinstance(node, ast.Call):
                func_name = ""
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr
                if func_name.endswith("Insight"):
                    for kw in node.keywords:
                        if kw.arg != "id":
                            continue
                        if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                            produced_exact.add(kw.value.value)
                        elif isinstance(kw.value, ast.JoinedStr):
                            parts = kw.value.values
                            if parts and isinstance(parts[0], ast.Constant):
                                produced_prefixes.add(str(parts[0].value))

                # ── consumers: <ledger>.resolve("id") etc. ──
                if (isinstance(node.func, ast.Attribute)
                        and node.func.attr in LEDGER_METHODS
                        and _receiver_mentions_ledger(node.func.value)
                        and node.args
                        and isinstance(node.args[0], ast.Constant)
                        and isinstance(node.args[0].value, str)):
                    referenced_exact.setdefault(node.args[0].value, set()).add(rel)

            # ── producers: coach-finding dict literals ──
            # ml/model_coach.py emits findings as {'id': ..., 'severity': ...,
            # 'finding': ...} dicts that the Train page ingests into the
            # ledger. Require the co-occurring keys so arbitrary dicts with an
            # 'id' key don't count as producers.
            if isinstance(node, ast.Dict):
                keys = {getattr(k, "value", None) for k in node.keys
                        if isinstance(k, ast.Constant)}
                if "id" in keys and len(keys & {"severity", "finding",
                                                "implication",
                                                "recommended_action"}) >= 2:
                    for k, v in zip(node.keys, node.values):
                        if (isinstance(k, ast.Constant) and k.value == "id"):
                            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                                produced_exact.add(v.value)
                            elif isinstance(v, ast.JoinedStr):
                                parts = v.values
                                if parts and isinstance(parts[0], ast.Constant):
                                    produced_prefixes.add(str(parts[0].value))

            # ── consumers: _ACTION_TO_INSIGHT_MAP entries ──
            if (isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "_ACTION_TO_INSIGHT_MAP"
                            for t in node.targets)
                    and isinstance(node.value, ast.Dict)):
                for v in node.value.values:
                    if not isinstance(v, ast.Dict):
                        continue
                    for k2, v2 in zip(v.keys, v.values):
                        key = getattr(k2, "value", None)
                        if key == "exact" and isinstance(v2, (ast.List, ast.Tuple)):
                            for elt in v2.elts:
                                if isinstance(elt, ast.Constant):
                                    referenced_exact.setdefault(elt.value, set()).add(rel)
                        elif key == "prefix" and isinstance(v2, ast.Constant):
                            referenced_prefixes.setdefault(v2.value, set()).add(rel)

    return produced_exact, produced_prefixes, referenced_exact, referenced_prefixes


PRODUCED_EXACT, PRODUCED_PREFIXES, REFERENCED_EXACT, REFERENCED_PREFIXES = collect_ids()


class TestInsightIdIntegrity:
    def test_the_scan_actually_scanned_something(self):
        """The non-vacuity floor.

        Every other assertion in this file is of the form "nothing referenced is
        missing", which an empty scan satisfies perfectly. After a rename-heavy
        migration that is exactly what happens: the ids move, the scan finds
        none of them, and a green test reports that no id is orphaned because no
        id was looked at.

        So the counts are asserted first, and the scanned directories are
        required to exist rather than skipped when absent.
        """
        for d in REQUIRED_SCAN_DIRS:
            assert os.path.isdir(os.path.join(PROJECT_ROOT, d)), (
                f"SCAN_DIRS names '{d}' but it does not exist. If it moved, update "
                "SCAN_DIRS — otherwise this file guards an empty set.")

        assert len(_SCANNED_FILES) >= MIN_FILES_SCANNED, (
            f"scanned only {len(_SCANNED_FILES)} files, expected at least "
            f"{MIN_FILES_SCANNED} — the scan has lost its source tree")
        assert len(PRODUCED_EXACT) >= MIN_PRODUCED_IDS, (
            f"found only {len(PRODUCED_EXACT)} produced insight ids, expected at least "
            f"{MIN_PRODUCED_IDS} — the producer pattern has stopped matching")
        assert len(REFERENCED_EXACT) >= MIN_REFERENCED_IDS, (
            f"found only {len(REFERENCED_EXACT)} referenced insight ids, expected at "
            f"least {MIN_REFERENCED_IDS} — the consumer pattern has stopped matching")

    def test_scan_finds_known_producers(self):
        """Guard against the scanner itself silently breaking."""
        for known in ("eda_sufficiency_insufficient", "eda_sufficiency_borderline",
                      "preprocess_high_cardinality", "eda_opportunity_clean_data"):
            assert known in PRODUCED_EXACT, f"scanner lost producer {known}"
        assert any(p.startswith("eda_leakage_") for p in PRODUCED_PREFIXES)

    def test_every_referenced_exact_id_is_produced(self):
        orphans = {
            rid: sorted(files) for rid, files in REFERENCED_EXACT.items()
            if rid not in PRODUCED_EXACT
            and not any(rid.startswith(p) for p in PRODUCED_PREFIXES)
        }
        assert not orphans, (
            "Referenced insight ids that nothing produces (stale-identifier "
            f"bug class): {orphans}"
        )

    def test_every_referenced_prefix_matches_a_producer(self):
        orphans = {
            pref: sorted(files) for pref, files in REFERENCED_PREFIXES.items()
            if not any(e.startswith(pref) for e in PRODUCED_EXACT)
            and not any(p.startswith(pref) or pref.startswith(p)
                        for p in PRODUCED_PREFIXES)
        }
        assert not orphans, (
            f"Referenced insight-id prefixes that match no producer: {orphans}"
        )
