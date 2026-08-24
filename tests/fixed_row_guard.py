"""The resolver both halves of the `FIXED`-row guard read. Not a test file.

`L55-A1` split `tests/test_a_fixed_row_names_a_test_that_actually_runs.py` in
two, because its two checks differ in cost by ~500× and every documented
invocation excludes the FILE for the sake of the slow one. This module exists so
the split does not become a second implementation of the resolver — which is the
defect the original file's own docstring rails against at two separate points
(*"a regex approximating pytest's collection is a second implementation to keep
in sync, which is this project's most-repeated defect"*).

**No `test_` prefix**, so pytest does not collect it. It is imported by both
halves as `tests.fixed_row_guard`.

Nothing here is new. The functions are the originals, moved with their comments
intact, with the leading underscores dropped because they are now a module's
public surface rather than a test file's private one. The only addition is
:data:`_COLLECT_CACHE` — see :func:`collected`.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

FINDINGS = os.path.join(PROJECT_ROOT, "docs", "turbotab", "data", "findings.json")

#: Rows whose `test` field names something that is not a pytest target at all —
#: a document, a tool invocation, a data file. They are REPORTED rather than
#: silently dropped, because "no pytest target" is a fact about the row and a
#: check that quietly ignores a class of rows is the shrug this guard is against.
_PATH = re.compile(r"([A-Za-z0-9_./-]+\.py)")
# A NAME, not the English word. The first draft was `(?:test|Test)[A-Za-z0-9_]*`
# and it matched "tests" inside `(seven tests)`, reporting forty-odd rows as
# naming a function that does not exist — the matcher-fires-on-prose failure,
# which this project keeps meeting one level down from wherever it is looking.
# A pytest function is `test_` with the underscore; a class is `Test` followed
# by a capital.
_FUNC = re.compile(r"\b(test_[A-Za-z0-9_]+|Test[A-Z][A-Za-z0-9_]*)")

# One `--collect-only` subprocess per session rather than one per caller. Three
# tests across the two halves call `resolve()`, and collection is the whole cost
# of the fast half — memoizing it is the difference between ~5s and ~15s in the
# tier that now runs every batch.
#
# SAFE BECAUSE THE INPUT CANNOT MOVE UNDER IT: the answer is a function of
# `findings.json` and of what pytest can collect, and no test in this repository
# writes either during a session. If one ever does, this cache is where the lie
# would live — so the guard is that a caller wanting a fresh answer clears it,
# and nothing in the tree does that silently.
_COLLECT_CACHE: Optional[List[str]] = None


def rows() -> List[Dict]:
    data = json.load(open(FINDINGS, encoding="utf-8"))
    return data["findings"] if isinstance(data, dict) and "findings" in data else data


def fixed_rows() -> List[Dict]:
    return [r for r in rows() if r.get("status") == "FIXED" and r.get("test")]


def collected() -> List[str]:
    """Every node id pytest can collect from the files `FIXED` rows name.

    ASKED, NOT RE-DERIVED. Class methods, parametrization ids and conftest
    collection rules are pytest's business, and a regex that reproduces them is
    a second implementation of pytest.
    """
    global _COLLECT_CACHE
    if _COLLECT_CACHE is not None:
        return _COLLECT_CACHE
    files = sorted({p for r in fixed_rows() for p in _PATH.findall(r["test"])
                    if os.path.exists(os.path.join(PROJECT_ROOT, p))})
    if not files:
        _COLLECT_CACHE = []
        return _COLLECT_CACHE
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "--no-header",
         "-p", "no:randomly", "--continue-on-collection-errors", *files],
        cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=600)
    _COLLECT_CACHE = [line.strip() for line in out.stdout.splitlines()
                      if "::" in line and line.strip().startswith(
                          ("tests/", "turbotab/", "utils/"))]
    return _COLLECT_CACHE


def _is_pytest_target(test: str, paths: List[str]) -> bool:
    """Is this `test` field naming something pytest can run at all?

    `_PATH` matches any `.py`, so a row whose guard is a **tool invocation**
    — `T0-TOOL-001` names `docs/turbotab/tools/ledger.py check`, `T0-PROC-001`
    names `tools/register.py check` — used to arrive as a row naming a test
    file. Before `L56-B1` they landed in the resolver's blind spot and were
    reported as nothing; naively fixed, they land in `missing` and read as
    **two closed rows resting on tests that do not exist**, which is a
    different false statement in the other direction.

    They are neither. `not_pytest`'s own docstring already names the category:
    *"a document, a tool invocation, a data file."* So the classification is by
    SHAPE, not by relaxing anything: a pytest target names a `test_*.py` file,
    or carries a `::` node id. A bare `.py` that is neither is a tool.
    """
    if "::" in test:
        return True
    return any(os.path.basename(p).startswith("test_") for p in paths)


def _named_functions(test: str, paths: List[str]) -> List[str]:
    """The function names a `test` field NAMES, read from the node ids.

    `TEST-071`, and it is the whole of `L56-B1`. This used to read the function
    out of the WHOLE string and then subtract the file stems:

        stems = {p.split("/")[-1][:-3] for p in paths}
        funcs = [f for f in _FUNC.findall(test) if f not in stems]

    The subtraction is there because `turbotab/test_foo.py` contains the token
    `test_foo`, which is a path fragment and not a named function. But **this
    project's dominant convention is `turbotab/test_foo.py::test_foo`** — the
    function name IS the file stem — so the filter discarded the real name too,
    `funcs` emptied, and `resolve`'s `if hits: … elif funcs: …` appended the row
    to **nothing at all.** Twenty-six of 362 `FIXED` rows landed there.

    **The consequence is why the row is `high`.** `missing` can only fill when
    `funcs` is non-empty, so for every row using the dominant convention, *a
    named test that does not exist produced exactly the same silence as one that
    does.* `TEST-063` was caught at L55 only because its function name happened
    to differ from its file name.

    **The fix is to stop subtracting and start splitting.** A node id is
    structured — `path::segment::segment` — so the names are read from the
    segments after the first `::` and the path is never scanned for them. A
    fragment with no `::` names a **file**, which is what those rows mean, and
    it is returned as such rather than discarded.
    """
    names: List[str] = []
    for fragment in re.split(r"[,\s]+AND[,\s]+|;", test):
        for part in fragment.split("::")[1:]:
            names.extend(_FUNC.findall(part))
    if names:
        return sorted(set(names))
    # No `::` anywhere: the row names a FILE. That is a real and common way to
    # write this field — `MISC-014` names a file and says "(21 tests, two
    # target shapes)" beside it — and it is not a row naming nothing.
    return []


def resolve() -> Tuple[Dict[str, List[str]], List[Tuple[str, List[str]]], List[str]]:
    """(row id -> its node ids, rows naming a missing function, rows with no .py).

    **The partition is asserted to sum**, and that assertion is the durable half
    of `TEST-071`. A three-way split that returns three lists and checks nothing
    is exhaustive is exactly how a silent fourth outcome survived the loop that
    wrote it and the adjudication that accepted it.
    """
    # EVERY SEGMENT, not just the leaf. A row may name the CLASS
    # (`TestCommaReading`) rather than the method, and an index keyed on leaves
    # only reports every such row as naming something that does not exist —
    # which is what the first draft did, for thirty rows.
    by_func: Dict[str, List[str]] = {}
    for nid in collected():
        for segment in nid.split("::")[1:]:
            by_func.setdefault(segment.split("[")[0], []).append(nid)

    by_file: Dict[str, List[str]] = {}
    for nid in collected():
        by_file.setdefault(nid.split("::")[0], []).append(nid)

    resolved: Dict[str, List[str]] = {}
    missing: List[Tuple[str, List[str]]] = []
    not_pytest: List[str] = []
    for r in fixed_rows():
        paths = _PATH.findall(r["test"])
        if not paths or not _is_pytest_target(r["test"], paths):
            not_pytest.append(r["id"])
            continue
        funcs = _named_functions(r["test"], paths)
        if funcs:
            hits = sorted({n for f in funcs for n in by_func.get(f, [])
                           if n.split("::")[0] in paths})
            if hits:
                resolved[r["id"]] = hits
            else:
                missing.append((r["id"], sorted(set(funcs) - set(by_func))))
            continue
        # A row naming a FILE and no function. It resolves to every node in
        # that file, and it is `missing` when the file collects nothing —
        # which is the same claim one level up and was previously unasked.
        hits = sorted({n for p in paths for n in by_file.get(p, [])})
        if hits:
            resolved[r["id"]] = hits
        else:
            missing.append((r["id"], sorted(paths)))
    return resolved, missing, not_pytest


def partition_is_exhaustive() -> Tuple[bool, Dict[str, int]]:
    """Does every `FIXED` row land in exactly one bucket?

    `TEST-071`'s durable half. The defect was not that a row resolved wrongly —
    it was that a row landed **nowhere**, and three lists that are never summed
    cannot report that.
    """
    resolved, missing, not_pytest = resolve()
    counts = {"resolved": len(resolved), "missing": len(missing),
              "not_pytest": len(not_pytest), "rows": len(fixed_rows())}
    total = counts["resolved"] + counts["missing"] + counts["not_pytest"]
    return total == counts["rows"], counts
