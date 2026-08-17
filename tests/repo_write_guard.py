"""`TEST-098`'s class, resolved rather than tainted: *a test that writes a path
git tracks*.

**Why this is a resolver and not a grep.** The four writers this was built from
were found by a hand sweep of twenty write-shaped patterns over 370 files — 97
hits — and that sweep **missed the worst one entirely**, because
`test_the_fixture_constants_match_the_fixtures.py` does not write. It spawns a
subprocess that writes twelve tracked files. A pattern sweep returned **zero**
for the file holding the most damaging instance, which is `AGENT_ONBOARD.md`
§07 trap #5 in its most expensive form.

So this does two things a grep cannot:

1. **It resolves the destination to a concrete path** rather than deciding a
   name looks repo-ish. `shutil.copy2(GENERATOR, sandbox / name)` reads a
   tracked file and writes a temporary one; a taint-based check flags it and a
   resolver does not. Measured on the corpus at the time of writing: taint gave
   9 write hits of which **6 were false**, and every false one was either a
   copy *source* or `str.replace` on text read out of a tracked file.
2. **It follows a spawn into the script it spawns** and analyzes that script
   with the same resolver. That is the only way writer #1 is visible.

## What this cannot see, stated here rather than discovered later

Written down in the shape `turbotab/rankings.py:40-46` uses, because a sweep
that does not publish its blind spot reports coverage it does not have.

- **A destination composed at runtime.** `out / name` where `name` comes from a
  loop, an argument or a fixture resolves to nothing, and an unresolvable
  destination is counted in `unresolved` rather than passed silently. The count
  is asserted, so this hole cannot quietly grow.
- **A write through a helper the spawn chain does not reach.** Spawn following
  is one level by default: the spawned script's own writes and the module-level
  constants it composes them from. A spawned script that imports a second
  module and writes from there is invisible.
- **`os.walk` + `open` on a directory handed in as a string.** Resolution is
  syntactic; nothing here executes the test.
- **A C extension or a library that writes on its own.** `to_csv` is known
  because it is named; a writer this table does not list is not seen. The
  mitigation is the same one `rankings.py` names: the ordinary way a test
  writes a file in this codebase is one of the calls below.
"""
from __future__ import annotations

import ast
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: Path-shaped methods whose RECEIVER is the thing written.
#: `p.write_text(data)` — the argument is content, not a destination, and
#: reading it as one is where the taint prototype produced its false positives.
RECEIVER_WRITES = {
    "write_text", "write_bytes", "touch", "mkdir", "rmdir", "unlink",
    "chmod", "truncate", "symlink_to", "hardlink_to",
}
#: Methods whose FIRST POSITIONAL ARGUMENT is the destination, and whose
#: receiver is a frame, a figure or a writer rather than a path.
ARG0_WRITES = {
    "to_csv", "to_parquet", "to_json", "to_pickle", "to_feather", "to_excel",
    "to_hdf", "to_stata", "savefig",
}
#: `p.rename(q)` / `p.replace(q)` move the receiver AND land on the argument,
#: so both ends are destinations. Kept apart because `str.replace` shares the
#: name and is not a write at all — the resolver tells them apart by whether
#: the receiver resolves to a path.
BOTH_ENDS = {"rename", "replace"}

#: `module.function(...)` writers, with the index of the destination argument.
FUNC_WRITES: Dict[Tuple[str, str], Tuple[int, ...]] = {
    ("os", "remove"): (0,), ("os", "unlink"): (0,), ("os", "rmdir"): (0,),
    ("os", "removedirs"): (0,), ("os", "makedirs"): (0,),
    ("os", "mkdir"): (0,), ("os", "utime"): (0,), ("os", "chmod"): (0,),
    ("os", "truncate"): (0,),
    ("os", "rename"): (0, 1), ("os", "replace"): (0, 1),
    ("os", "symlink"): (1,), ("os", "link"): (1,),
    ("shutil", "copy"): (1,), ("shutil", "copy2"): (1,),
    ("shutil", "copyfile"): (1,), ("shutil", "copytree"): (1,),
    ("shutil", "move"): (0, 1), ("shutil", "rmtree"): (0,),
    ("shutil", "make_archive"): (0,),
    ("np", "save"): (0,), ("np", "savetxt"): (0,), ("np", "savez"): (0,),
    ("numpy", "save"): (0,), ("numpy", "savetxt"): (0,),
    ("plt", "savefig"): (0,),
}
SPAWNERS = {"run", "Popen", "call", "check_call", "check_output"}
#: Attributes that keep an expression a path.
PATHY_ATTRS = {"parent", "parents"}
#: Calls that keep an expression a path. `read_text` is deliberately absent —
#: propagating through it is what turned file CONTENTS into a destination.
PATHY_CALLS = {"resolve", "absolute", "expanduser", "joinpath",
               "with_name", "with_suffix"}
WRITE_MODES = set("wax+")


def _under(value) -> bool:
    """`value` is *somewhere under this directory*, leaf unknown."""
    return isinstance(value, tuple) and len(value) == 2 and value[0] == "under"


def _destination(value) -> Tuple[Optional[Path], bool]:
    """`(where it lands, whether the leaf is known)`."""
    if isinstance(value, Path):
        return value, True
    if _under(value):
        return value[1], False
    return None, False


class Site:
    """One write-shaped call, with where it lands."""

    def __init__(self, path: str, lineno: int, call: str, dest: Optional[Path],
                 exact: bool = True, via: str = "") -> None:
        self.path, self.lineno, self.call = path, lineno, call
        self.dest, self.exact, self.via = dest, exact, via

    @property
    def key(self) -> str:
        return f"{self.path}:{self.lineno}"

    def __repr__(self) -> str:
        if self.dest is None:
            where = "<unresolved>"
        else:
            try:
                shown = self.dest.relative_to(PROJECT_ROOT)
            except ValueError:
                shown = self.dest
            where = f"{shown}/*" if not self.exact else str(shown)
        return f"{self.key} {self.call} → {where}{self.via}"


def _eval(node, env: Dict[str, object], here: Path):
    """A path or a string fragment, syntactically, or `None`.

    Deliberately partial. Anything it cannot resolve returns `None` and is
    counted as unresolved rather than assumed harmless.
    """
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else None
    if isinstance(node, ast.Name):
        if node.id == "__file__":
            return str(here)
        return env.get(node.id)
    if isinstance(node, ast.JoinedStr):          # f"{name}.md" — unresolvable
        return None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left = _eval(node.left, env, here)
        right = _eval(node.right, env, here)
        if isinstance(left, Path) and isinstance(right, str):
            return left / right
        # THE HALF THAT CATCHES WRITER #1. `frame.to_csv(HERE / name)` inside
        # `make_genomics_siblings._write` has an unresolvable LEAF — `name` is
        # a parameter — and a resolver that gave up here would report the
        # single most damaging instance in the corpus as `<unresolved>`, which
        # is silence dressed as coverage. The DIRECTORY is what decides whether
        # a write lands in the checkout, and the directory is resolvable.
        base = left[1] if _under(left) else left
        if isinstance(base, Path):
            return ("under", base)
        return None
    if isinstance(node, ast.Attribute):
        base = _eval(node.value, env, here)
        if isinstance(base, Path) and node.attr == "parent":
            return base.parent
        if isinstance(base, Path) and node.attr == "parents":
            return ("parents", base)
        return None
    if isinstance(node, ast.Subscript):
        base = _eval(node.value, env, here)
        idx = node.slice
        if (isinstance(base, tuple) and base and base[0] == "parents"
                and isinstance(idx, ast.Constant)
                and isinstance(idx.value, int)):
            return base[1].parents[idx.value]
        return None
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Name) and func.id in ("Path", "str") and node.args:
            inner = _eval(node.args[0], env, here)
            if func.id == "Path":
                return Path(inner) if isinstance(inner, str) else inner
            return inner
        if isinstance(func, ast.Attribute):
            if func.attr in PATHY_CALLS:
                base = _eval(func.value, env, here)
                if func.attr == "joinpath" and isinstance(base, Path):
                    parts = [_eval(a, env, here) for a in node.args]
                    if all(isinstance(p, str) for p in parts):
                        return base.joinpath(*parts)
                    return None
                return base if isinstance(base, Path) else None
            # os.path.dirname(...) / os.path.abspath(...) / os.path.join(...)
            if func.attr in ("dirname", "abspath", "realpath") and node.args:
                inner = _eval(node.args[0], env, here)
                if isinstance(inner, str):
                    inner = Path(inner)
                if isinstance(inner, Path):
                    return inner.parent if func.attr == "dirname" else inner
                return None
            if func.attr == "join" and node.args:
                parts = [_eval(a, env, here) for a in node.args]
                if parts and isinstance(parts[0], (str, Path)) and \
                        all(isinstance(p, str) for p in parts[1:]):
                    return Path(parts[0]).joinpath(*parts[1:])
                return None
    return None


def _module_env(tree: ast.Module, here: Path) -> Dict[str, object]:
    env: Dict[str, object] = {}
    for _ in range(3):                       # forward references settle fast
        for stmt in tree.body:
            if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
                continue
            target = stmt.targets[0]
            if not isinstance(target, ast.Name):
                continue
            value = _eval(stmt.value, env, here)
            if isinstance(value, Path):
                env[target.id] = value
    return env


def _mode_of(node: ast.Call, positional: int) -> str:
    if len(node.args) > positional and isinstance(node.args[positional],
                                                  ast.Constant):
        return str(node.args[positional].value)
    for kw in node.keywords:
        if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
            return str(kw.value.value)
    return ""


def analyze(rel: str, follow: int = 1,
            source: Optional[str] = None) -> Tuple[List[Site], List[Site],
                                                   List[Site]]:
    """`(repo_writes, unresolved, spawns)` for one tracked file.

    `source` overrides what is on disk while keeping `rel` as the file's
    identity, so `__file__`-derived constants still resolve to where the file
    really lives. That is what lets the positive control plant a known writer
    at a real path instead of asserting against a synthetic tree.
    """
    path = PROJECT_ROOT / rel
    here = path.resolve()
    text = source if source is not None else path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=rel)
    module_env = _module_env(tree, here)

    repo: List[Site] = []
    unknown: List[Site] = []
    spawns: List[Site] = []

    def record(node, call: str, value, via: str = "") -> None:
        dest, exact = _destination(value)
        site = Site(rel, node.lineno, call, dest, exact, via)
        if site.dest is None:
            unknown.append(site)
        elif _inside_repo(site.dest):
            repo.append(site)

    def scan(body, env: Dict[str, object]) -> None:
        # One environment over the whole module rather than one per function.
        # Coarser than a real dataflow: a name bound to a repo path in one
        # function is still bound when a later function reuses the name. The
        # error direction is toward REPORTING a write rather than missing one,
        # which is the safe side for a guard, and it is written down here
        # rather than found later.
        for node in ast.walk(body):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Parameter defaults are bindings too — `def _check(tool=TOOL)`
                # is how three of these tests name the script they spawn.
                args = node.args
                slots = list(args.posonlyargs) + list(args.args)
                for arg, default in zip(slots[len(slots) - len(args.defaults):],
                                        args.defaults):
                    value = _eval(default, env, here)
                    if isinstance(value, Path):
                        env[arg.arg] = value
                for arg, default in zip(args.kwonlyargs, args.kw_defaults):
                    if default is None:
                        continue
                    value = _eval(default, env, here)
                    if isinstance(value, Path):
                        env[arg.arg] = value
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and \
                    isinstance(node.targets[0], ast.Name):
                value = _eval(node.value, env, here)
                if isinstance(value, Path):
                    env[node.targets[0].id] = value
            if not isinstance(node, ast.Call):
                continue
            func = node.func

            if isinstance(func, ast.Attribute):
                attr = func.attr
                if attr in RECEIVER_WRITES:
                    record(node, f".{attr}()", _eval(func.value, env, here))
                    continue
                if attr in BOTH_ENDS:
                    receiver = _eval(func.value, env, here)
                    # `str.replace` shares the name; only a resolved PATH
                    # receiver makes this a write at all.
                    if isinstance(receiver, Path):
                        record(node, f".{attr}()", receiver)
                        if node.args:
                            record(node, f".{attr}() dest",
                                   _eval(node.args[0], env, here))
                    continue
                if attr in ARG0_WRITES and node.args:
                    record(node, f".{attr}()", _eval(node.args[0], env, here))
                    continue
                if attr == "open":
                    if any(c in _mode_of(node, 0) for c in WRITE_MODES):
                        record(node, ".open(w)", _eval(func.value, env, here))
                    continue
                if isinstance(func.value, ast.Name):
                    slots = FUNC_WRITES.get((func.value.id, attr))
                    if slots:
                        for i in slots:
                            if len(node.args) > i:
                                record(node, f"{func.value.id}.{attr}()",
                                       _eval(node.args[i], env, here))
                        continue
                    if attr in SPAWNERS and func.value.id in ("subprocess",
                                                              "sp"):
                        _spawn(node, env, spawns, repo, unknown, rel, follow,
                               here)
                        continue

            if isinstance(func, ast.Name) and func.id == "open" and node.args:
                if any(c in _mode_of(node, 1) for c in WRITE_MODES):
                    record(node, "open(w)", _eval(node.args[0], env, here))

    scan(tree, dict(module_env))
    return repo, unknown, spawns


def _pathish(node) -> bool:
    """The expression is a COMPOSED path — `X / y`, or `str()`/`Path()` of one.

    Used only to decide whether a spawn whose target did not resolve deserves
    to be reported as unresolved. `sys.executable` and `"-m"` are not composed
    paths and must not turn every `python -m pytest` spawn into a blind spot;
    `str(sandbox / GENERATOR.name)` is one and must.
    """
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return True
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
            and node.func.id in ("str", "Path") and node.args:
        return _pathish(node.args[0])
    return False


def _spawn(node, env, spawns, repo, unknown, rel, follow, here) -> None:
    """A spawn is only interesting if what it spawns writes the repo.

    **The argv list and nothing else.** An earlier pass took any name anywhere
    in the call, and `str(sandbox / GENERATOR.name)` therefore resolved to
    `GENERATOR` — reporting a sandboxed spawn as a write to `sample_data/`,
    which is the exact inversion of the fix it was checking.
    """
    argv = node.args[0] if node.args else None
    if isinstance(argv, (ast.List, ast.Tuple)):
        elements = list(argv.elts)
    elif argv is not None:
        elements = [argv]
    else:
        elements = []
    resolved = [_eval(e, env, here) for e in elements]
    scripts = [t for t in resolved
               if isinstance(t, Path) and t.suffix == ".py" and t.exists()
               and _inside_repo(t)]
    if not scripts:
        if any(_pathish(e) and _eval(e, env, here) is None for e in elements):
            unknown.append(Site(rel, node.lineno,
                                "subprocess → <argv did not resolve>", None))
        return
    script = scripts[0]
    spawns.append(Site(rel, node.lineno, f"subprocess → {script.name}",
                       script))
    if follow <= 0:
        unknown.append(Site(rel, node.lineno,
                            f"subprocess → {script.name} (not followed)",
                            None))
        return
    inner_rel = str(script.relative_to(PROJECT_ROOT))
    inner_repo, inner_unknown, _ = analyze(inner_rel, follow - 1)
    for s in inner_repo:
        repo.append(Site(rel, node.lineno,
                         f"subprocess → {script.name} {s.call}", s.dest,
                         s.exact, via=f"  (via {s.key})"))
    for s in inner_unknown:
        unknown.append(Site(rel, node.lineno,
                            f"subprocess → {script.name} {s.call}", None,
                            via=f"  (via {s.key})"))


def _inside_repo(dest: Path) -> bool:
    """True where the destination lands in this checkout's working tree."""
    try:
        rel = dest.resolve().relative_to(PROJECT_ROOT)
    except (ValueError, OSError):
        return False
    top = rel.parts[0] if rel.parts else ""
    # `venv/` and the caches are in the tree and are not source; a test that
    # writes there is not this class.
    return top not in {"venv", ".venv", ".git", ".pytest_cache", "__pycache__",
                       ".cache", ".worktrees", ".claude"}


def tracked_test_files() -> List[str]:
    """Tracked test modules, from git rather than from a walk.

    `git ls-files` and not `rglob`, for `TEST-106`'s reason one file over: a
    walk reads nested checkouts and generated session material, and a guard
    about what git tracks should ask git what it tracks.
    """
    out = subprocess.run(
        ["git", "ls-files", "-z", "--", "turbotab", "tests"],
        cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
    names = []
    for line in out.stdout.split("\0"):
        if not line.endswith(".py"):
            continue
        base = os.path.basename(line)
        if base.startswith("test_") or base == "conftest.py":
            names.append(line)
    return sorted(names)


def sweep(follow: int = 1):
    """`(repo_writes, unresolved, spawns, n_files)` over the whole corpus."""
    repo: List[Site] = []
    unknown: List[Site] = []
    spawns: List[Site] = []
    files = tracked_test_files()
    for rel in files:
        r, u, s = analyze(rel, follow)
        repo.extend(r)
        unknown.extend(u)
        spawns.extend(s)
    return repo, unknown, spawns, len(files)
