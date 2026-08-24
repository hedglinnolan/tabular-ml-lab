"""Records which third-party packages the pre-commit gates import DIRECTLY.

`TEST-110`. `.githooks/lib.sh`'s `gates_can_run` probes a hand-written list of
module names, and the list fell behind what the gates import — an interpreter
carrying the probed names and missing an unprobed one passed the probe and then
produced the exact `✗ … No module named …` the probe exists to prevent.

**A static walk cannot take this measurement, and that is why this file runs.**
`docs/turbotab/tools/evidence.py:302` reaches `turbotab.api` — and through it
`fastapi` — with `importlib.import_module(f"turbotab.{path.stem}")`, a path no
literal search can see. `AGENT_ONBOARD.md` §07 trap #5: a grep answers *does
this text appear*, and the question here is *does this run*.

## Why only first-party -> third-party edges are recorded

`sys.modules` after a gate holds forty-odd installed packages, nearly all of
them reached through another package. Requiring the probe to name `numpy`
because `pandas` imports it would make the list grow without making it stronger
— `numpy` cannot be absent while `pandas` is present. What the probe actually
needs is the set of packages **this repository's own code** imports directly,
because those are the ones nothing else guarantees.

So an edge is recorded when a file inside the checkout — and not inside a venv
that happens to live under it — executes an absolute import that resolves into
site-packages.

Placed in its own directory rather than in `tests/`, because `sitecustomize` is
imported by *every* interpreter whose `sys.path` contains it, and `conftest.py`
puts `tests/` on `sys.path`. A file that changes the behavior of every Python
process in the repository is not a file to leave lying in an imported package.

Driven only by `tests/test_the_pre_commit_hook_can_run_where_it_is_run.py`,
which sets both environment variables below.
"""
from __future__ import annotations

import atexit
import builtins
import json
import os
import sys
import sysconfig

_ROOT = os.environ.get("GATE_PROBE_ROOT", "")
_OUT = os.environ.get("GATE_PROBE_OUT", "")

if _ROOT and _OUT:
    _SITE = {p for p in (sysconfig.get_paths().get("purelib"),
                         sysconfig.get_paths().get("platlib")) if p}
    _real_import = builtins.__import__
    _edges: set = set()

    def _first_party(path: str) -> bool:
        """Inside the checkout, and not inside an environment under it."""
        if not path or not path.startswith(_ROOT + os.sep):
            return False
        parts = path[len(_ROOT) + 1:].split(os.sep)
        return not any(p in ("venv", ".venv", "site-packages") for p in parts)

    def _installed(top: str) -> bool:
        module = sys.modules.get(top)
        where = getattr(module, "__file__", None)
        if not where:
            # A namespace package has no `__file__`; its `__path__` still says
            # where it came from, and skipping it would silently under-report.
            paths = list(getattr(module, "__path__", []) or [])
            where = paths[0] if paths else ""
        if not where:
            return False
        return (any(where.startswith(s) for s in _SITE)
                or "site-packages" in where)

    def _record(name, globals=None, locals=None, fromlist=(), level=0):
        module = _real_import(name, globals, locals, fromlist, level)
        try:
            # A relative import is first-party by construction.
            if level == 0:
                caller = (globals or {}).get("__file__") or ""
                top = name.split(".")[0]
                if _first_party(caller) and _installed(top):
                    _edges.add((os.path.relpath(caller, _ROOT), top))
        except Exception:                                # pragma: no cover
            # A recorder that can break the thing it records is worse than no
            # recorder: it would make a gate fail for the probe's reason.
            pass
        return module

    builtins.__import__ = _record

    @atexit.register
    def _dump() -> None:
        with open(_OUT, "w", encoding="utf-8") as handle:
            json.dump(sorted(_edges), handle)
