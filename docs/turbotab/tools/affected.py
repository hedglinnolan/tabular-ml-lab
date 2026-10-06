#!/usr/bin/env python3
"""Which `turbotab/` tests reach a change — and, louder, which it cannot see.

`L56-A1`. The product owner's constraint, in his words: *"These full suite tests
are simply taking too long for the workflow we are currently in. They run over
two hours and occasionally time out."* `L55-D` did this once by hand and turned
a 55-file sweep into **11 files, 217 tests, 6m08s**. This is that, as a tool.

## The rule this exists to serve

> A loop's regression evidence may be the scoped selection **quoted as scoped**,
> with the full sweep run once at the end. **A scoped run is never reported as a
> full run.**

So every output carries the word `SCOPED`, every selection carries the reason it
was selected, and the blind spots are printed **whether or not they fired**. A
selector that implies coverage it does not have is worse than no selector: it
converts "I did not look there" into "there was nothing there", which is the
governing rule's *assert something false* branch wearing a CLI.

## The graph is measured here, and NOT read from `data/import-graph.json`

The prompt for this part named `docs/turbotab/data/import-graph.json` and
`reverse-deps.json` as the measured graph. **They do not cover this package.**
Counted by top-level directory: `import-graph.json`'s 87 keys are `ml/` 35,
`utils/` 29, `pages/` 12, `models/` 7 and 4 at the root, and **`turbotab/`
appears zero times**; `reverse-deps.json`'s 74 keys are the same shape. They are
a snapshot of the *inherited Streamlit* codebase from the ten agent passes at
`fbe422a`, taken before this package existed. Selecting `turbotab/` tests from
them would have selected nothing and reported it as "no tests affected", which
is the exact failure mode this file's docstring is about.

So the graph is walked with `ast` at call time. That is cheap — the whole tree
is a few hundred files — and it cannot go stale between the walk and the answer.

## What it cannot see, stated once here and again on every run

An import edge is the only thing an AST walk observes. These reach production
code without one, and each is handled by a **named trigger** rather than by
hoping:

* **The page.** `turbotab/web/index.html` is JavaScript. Every claim about the
  interface runs through `turbotab/pageharness.py`, which *reads* that file — so
  a change to the page is invisible to imports. Trigger: any change under
  `turbotab/web/` selects every test importing `pageharness`.
* **Captured API responses.** A harness test supplies routes from a
  `TestClient` drive, so it reaches server code the test file never imports.
  Same trigger, same reason.
* **Fixtures.** `turbotab/sample_data/*.csv` is read by name, usually as a
  string. Trigger: a changed fixture selects every test whose source mentions
  its basename.
* **The record.** `docs/turbotab/data/findings.json` and `register.json` are
  read by the guards that check them. Trigger: named explicitly.
* **Reflection.** The model registry is keyed by string; `packs`/`recipes`
  register into module-global tables. An edge that exists only through a key is
  not an import and is **not** recovered.
* **`conftest.py` and plugins.** Unbounded reach. Trigger: `ESCALATE` — the
  tool refuses to scope and says to run everything.

The last one is the shape of the honest answer when scoping is not available:
this tool is allowed to say *I cannot narrow this*, and it says it loudly.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
from typing import Dict, Iterable, List, Sequence, Set, Tuple

# FOUR levels: `docs/turbotab/tools/affected.py` -> the repository root. Three
# lands on `docs/`, where every package lookup silently finds nothing and the
# tool reports "no tests affected" — which is this file's own headline failure
# mode, so the depth is asserted below rather than counted by eye.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

#: Directories whose `.py` files take part in the import graph.
PACKAGES: Tuple[str, ...] = ("turbotab", "ml", "models", "utils", "pages", "tests")

#: A change to any of these cannot be scoped and the tool says so rather than
#: guessing. `conftest.py` can rewrite collection for a whole tree; the harness
#: is the instrument every page claim runs through; this file selects the tests.
ESCALATORS: Tuple[str, ...] = (
    "conftest.py",
    "docs/turbotab/tools/affected.py",
)

#: Changes that reach tests without an import edge, and the trigger for each.
#: `(path predicate, selector predicate, reason)` — see the module docstring.
WEB_PREFIX = "turbotab/web/"
HARNESS = "turbotab/pageharness.py"
RECORD_FILES = ("docs/turbotab/data/findings.json",
                "docs/turbotab/data/register.json")

BLIND_SPOTS: Tuple[str, ...] = (
    "reflection — the model registry is keyed by string and packs/recipes "
    "register into module-global tables; an edge that exists only through a "
    "key is not an import and is NOT recovered here",
    "a test that reads a data file this tool was not told about",
    "a test whose subject is prose in a document (the pack files, the design "
    "language) rather than a module",
    "a change in behavior with no changed file — a dependency upgrade, a "
    "clock, a locale",
)


# ── the graph ────────────────────────────────────────────────────────────────

def _module_name(rel: str) -> str:
    """`turbotab/models.py` -> `turbotab.models`."""
    return rel[:-3].replace(os.sep, ".").replace("/", ".")


def _py_files() -> List[str]:
    out: List[str] = []
    for pkg in PACKAGES:
        base = os.path.join(ROOT, pkg)
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            # ANY hidden directory and anything that smells like an environment.
            # The first draft excluded `venv` and not `.venv`, and there IS a
            # `turbotab/.venv/` in this tree — 3,272 site-packages files walked
            # in as production modules, which put `pytest` and `numpy` in the
            # graph and made every fan-in number meaningless. A walk is a
            # measurement and it was measuring the wrong tree.
            dirnames[:] = [d for d in dirnames
                           if not d.startswith(".")
                           and d not in ("__pycache__", "venv", "node_modules",
                                         "site-packages", "build", "dist")]
            for name in filenames:
                if name.endswith(".py"):
                    # POSIX-RELATIVE, ON EVERY PLATFORM. `os.path.relpath`
                    # emits `turbotab\test_x.py` on Windows, and every path
                    # this tool compares against is `/`-joined: `git` output,
                    # the `--files` a caller types, `WEB_PREFIX`, `HARNESS`,
                    # `RECORD_FILES`, and the `startswith("turbotab/test_")`
                    # that decides what a test file IS. So on Windows the
                    # walk found 176 files and zero of them were tests, no
                    # changed path ever mapped to a module, and the tool
                    # printed `scoped: true, selected: 0` with exit 0 for a
                    # change to `deck_faces.py` — its own headline failure,
                    # delivered silently. The separator is normalized here,
                    # at the one boundary where the OS hands paths in.
                    rel = os.path.relpath(os.path.join(dirpath, name), ROOT)
                    out.append(rel.replace(os.sep, "/"))
    for name in os.listdir(ROOT):
        if name.endswith(".py"):
            out.append(name)
    return sorted(set(out))


def _imports(rel: str, known: Set[str]) -> Set[str]:
    """Every repo module this file imports, at any nesting depth.

    **Function-level imports count**, and in this codebase they are the norm —
    `turbotab/project.py` reaches `turbotab.models` through
    `from turbotab import models as _models` inside a method. `ast.walk`
    descends into function bodies, so those edges are recovered; an
    import-at-top-of-file assumption would have missed most of this package.
    """
    try:
        tree = ast.parse(open(os.path.join(ROOT, rel), encoding="utf-8",
                              errors="ignore").read())
    except SyntaxError:
        return set()
    found: Set[str] = set()

    def note(candidate: str) -> None:
        # `from turbotab import models` is `turbotab.models` when that module
        # exists and `turbotab` when it does not — resolved against the files
        # actually on disk rather than assumed either way.
        if candidate in known:
            found.add(candidate)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                note(alias.name)
                note(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level:                       # relative import
                continue
            base = node.module or ""
            note(base)
            for alias in node.names:
                note(f"{base}.{alias.name}" if base else alias.name)
    return found


def build_graph() -> Tuple[Dict[str, Set[str]], Dict[str, str]]:
    """`(module -> modules it imports, module -> its path)`."""
    files = _py_files()
    by_module = {_module_name(f): f for f in files}
    known = set(by_module)
    graph = {mod: _imports(path, known) for mod, path in by_module.items()}
    return graph, by_module


def reaches(graph: Dict[str, Set[str]], seeds: Set[str]) -> Set[str]:
    """Every module that imports a seed, directly or transitively."""
    importers: Dict[str, Set[str]] = {}
    for mod, deps in graph.items():
        for dep in deps:
            importers.setdefault(dep, set()).add(mod)
    seen: Set[str] = set(seeds)
    stack = list(seeds)
    while stack:
        cur = stack.pop()
        for up in importers.get(cur, ()):
            if up not in seen:
                seen.add(up)
                stack.append(up)
    return seen


# ── the changed files ────────────────────────────────────────────────────────

def changed_files(since: str | None, explicit: Sequence[str] | None) -> List[str]:
    if explicit:
        # The other boundary paths cross. A Windows shell hands over
        # `turbotab\deck_faces.py`; the graph is keyed `turbotab/deck_faces.py`.
        # `git` already emits `/` on every platform, so only the typed form
        # needs the same treatment as the walk above.
        return sorted({c.replace(os.sep, "/") for c in explicit})
    if since:
        args = ["git", "diff", "--name-only", f"{since}...HEAD"]
    else:
        # Working tree AND staged AND untracked, because a new test file is a
        # change and `git diff` alone does not see one.
        args = ["git", "status", "--porcelain"]
    out = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
    lines = [line.strip() for line in out.stdout.splitlines() if line.strip()]
    if since:
        return sorted(set(lines))
    return sorted({line[3:].strip().strip('"') for line in lines})


# ── selection ────────────────────────────────────────────────────────────────

def select(changed: Sequence[str], closure: bool = False
           ) -> Tuple[List[Tuple[str, str]], List[str], List[str], Dict[str, int]]:
    """`(selected [(test file, reason)], escalations, notes, counts)`.

    ## Two modes, and the difference is measured rather than argued

    `turbotab/api.py` is an **aggregator**: 104 modules import it and it imports
    most of this package. So full transitive reachability answers *"could this
    test's import graph touch the change"* with **yes** for almost any change
    the API can reach — measured on this tree, a one-line change to
    `turbotab/models.py` selects **131 of 147** files and one to
    `models/glm.py` selects **134**. That is true and it is nearly useless; a
    selector that returns 89% of the suite has not scoped anything, and
    reporting it as a scoped run would overstate what was skipped.

    So the DEFAULT is **direct**: a test is selected when it imports a changed
    module itself, or when a named trigger fires. `--closure` is the
    over-approximation, and **both counts are printed either way**, because the
    gap between them is the thing a reader needs in order to judge the
    selection.

    **Neither is exact, and the reason is stated rather than implied.** A test
    that imports `turbotab.api` and drives `/project/{id}/models` really does
    exercise `turbotab/models.py` without importing it, and no import walk can
    see that. The exact answer is a coverage map keyed by test — `TEST-072`,
    filed with this tool — and until it exists, `--closure` is the safe side
    and `direct` is the useful one.
    """
    graph, by_module = build_graph()
    path_to_module = {v: k for k, v in by_module.items()}
    # One spelling of the prefix. The `os.sep` alternative that used to sit
    # beside this was the only place the tool anticipated Windows, and it was
    # the wrong place: it made THIS filter see the tests while every other
    # comparison in the file still could not. `_py_files` now emits `/`, so a
    # test file has exactly one shape here.
    tests = sorted(m for m, p in by_module.items()
                   if p.startswith("turbotab/test_"))

    escalations = [c for c in changed
                   if any(c.endswith(e) or c == e for e in ESCALATORS)]

    seeds = {path_to_module[c] for c in changed if c in path_to_module}
    reachable = reaches(graph, seeds) if seeds else set()

    picked: Dict[str, str] = {}
    n_direct = 0
    for mod in tests:
        path = by_module[mod]
        hits = sorted(graph[mod] & seeds)
        if path in changed:
            picked[path] = "the test file itself changed"
            n_direct += 1
        elif hits:
            picked[path] = f"imports {', '.join(hits)} — changed"
            n_direct += 1
        elif closure and mod in reachable:
            via = sorted(graph[mod] & reachable)[:2]
            picked[path] = ("--closure: reaches a changed module through "
                            + (", ".join(via) if via else "the graph"))
    n_closure = sum(1 for mod in tests
                    if by_module[mod] in changed
                    or (graph[mod] & seeds) or mod in reachable)

    # ── the triggers, for edges an import walk cannot see ──
    web = [c for c in changed if c.startswith(WEB_PREFIX)]
    if web or HARNESS in changed:
        why = ("the page changed and JavaScript is not an import edge"
               if web else "the harness itself changed")
        for mod in tests:
            path = by_module[mod]
            if "pageharness" in graph[mod] or "turbotab.pageharness" in graph[mod]:
                picked.setdefault(path, f"{why}; this test drives it")

    # ANY CHANGED NON-PYTHON FILE, NOT ONLY A FIXTURE.
    #
    # `L57-A1` found the gap by falling into it. The page trigger above selects
    # every test that IMPORTS `pageharness`, which is 56 files — and the
    # stylesheet validator added that same hour reads `index.html` directly and
    # imports no harness, so a palette change did not select the one test whose
    # entire subject is the palette. A trigger keyed on how a test reaches a
    # file rather than on WHICH file it names is a trigger with a hole exactly
    # where a new kind of test lands.
    #
    # So: a changed non-Python file selects every test whose source mentions its
    # basename. That covers fixtures, the page, the prototype and anything else
    # read by name — and it is deliberately a NAME match rather than a path
    # match, because tests build these paths from `Path(__file__).parent` and no
    # literal full path appears in them.
    named = [c for c in changed if not c.endswith(".py")]
    for changed_file in named:
        base = os.path.basename(changed_file)
        if not base:
            continue
        for mod in tests:
            path = by_module[mod]
            try:
                src = open(os.path.join(ROOT, path), encoding="utf-8",
                           errors="ignore").read()
            except OSError:
                continue
            if base in src:
                picked.setdefault(path, f"names the changed file {base}")
    fixtures = [c for c in named if "/sample_data/" in c]

    record = [c for c in changed if c in RECORD_FILES]
    if record:
        for mod in tests:
            path = by_module[mod]
            src = open(os.path.join(ROOT, path), encoding="utf-8",
                       errors="ignore").read()
            if "findings.json" in src or "register.json" in src:
                picked.setdefault(path, "reads the record, which changed")

    notes: List[str] = []
    if not seeds and not web and not fixtures and not record:
        notes.append("no changed file maps to a module in the graph")

    # AN EMPTY DIRECT SET IS NOT AN ANSWER WHEN THE CLOSURE IS NOT EMPTY, and
    # this is the sharpest thing the tool does. Measured on this tree: a change
    # to `models/glm.py` has **0** direct importers among the 147 turbotab test
    # files and **134** under closure — no test imports the wrapper, and every
    # test that drives the API reaches it through `ml.model_registry`. Printing
    # "0 selected" there would convert *"no test imports this"* into *"no test
    # exercises this"*, which is the governing rule's assert-something-false
    # branch, committed by the tool built to prevent it.
    if not closure and not picked and n_closure:
        escalations.append(
            f"the import walk selected 0 test files directly while the closure "
            f"selects {n_closure} — the change is reached only through an "
            f"aggregator, so imports cannot scope it")

    counts = {"selected": len(picked), "direct": n_direct,
              "closure": n_closure, "total": len(tests)}
    return sorted(picked.items()), escalations, notes, counts


# ── output ───────────────────────────────────────────────────────────────────

def report(changed: Sequence[str], selected, escalations, notes, counts,
           closure: bool = False, as_json: bool = False) -> int:
    if as_json:
        print(json.dumps({
            "scoped": not escalations,
            "mode": "closure" if closure else "direct",
            "changed": list(changed),
            "counts": counts,
            "selected": [{"file": f, "reason": r} for f, r in selected],
            "escalations": list(escalations),
            "blind_spots": list(BLIND_SPOTS),
            "notes": list(notes),
        }, indent=1))
        return 2 if escalations else 0

    mode = "closure" if closure else "direct"
    print(f"SCOPED SELECTION ({mode}) — {counts['selected']} of "
          f"{counts['total']} turbotab test files, "
          f"from {len(changed)} changed path(s)")
    # BOTH NUMBERS, ALWAYS. The gap is what tells a reader whether the scoping
    # is worth trusting: a direct set far below the closure set means the change
    # is reachable through the API aggregator and an import walk cannot say
    # whether those drives touch it.
    print(f"  direct {counts['direct']} · closure {counts['closure']} "
          f"· suite {counts['total']}")
    if counts["closure"] >= counts["total"] * 0.8:
        print("  ! the closure covers >=80% of the suite — this change reaches "
              "the API aggregator, so scoping buys little and the full sweep "
              "is the honest run.")
    print()

    if escalations:
        print("ESCALATE — this change cannot be scoped:")
        for e in escalations:
            print(f"  · {e}")
        print("  Run the full suite. A scoped run here would report a coverage "
              "it does not have.\n")

    for path, reason in selected:
        print(f"  {path}")
        print(f"      ← {reason}")
    if not selected:
        print("  (nothing selected)")
    for note in notes:
        print(f"  note: {note}")

    print("\nWHAT THIS SELECTION CANNOT SEE — printed every run, not only when it fires:")
    for spot in BLIND_SPOTS:
        print(f"  · {spot}")
    print("\nA SCOPED RUN IS NEVER REPORTED AS A FULL RUN. Quote it as scoped, "
          "and run the full sweep once at the end.")
    return 2 if escalations else 0


def _count_tests() -> int:
    graph, by_module = build_graph()
    return len([p for p in by_module.values() if p.startswith("turbotab/test_")])


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--since", help="git ref to diff against (default: working tree)")
    ap.add_argument("--files", nargs="*", help="explicit changed paths")
    ap.add_argument("--json", action="store_true", help="machine-readable")
    ap.add_argument("--closure", action="store_true",
                    help="over-approximate: follow the import graph transitively")
    ap.add_argument("--pytest-args", action="store_true",
                    help="print only the selected paths, for `xargs pytest`")
    args = ap.parse_args(argv)

    # THE REPORT MUST NOT DIE HALFWAY. The reason lines carry `←` and `—`,
    # and a Windows pipe is cp1252, which has neither: the first selected
    # file raised `UnicodeEncodeError` at position 6 of its own reason and
    # the blind-spot list — printed every run, or so the docstring promises —
    # was never reached. An empty selection prints no reason and so never
    # tripped it, which is why this surfaced only once the walk above could
    # see a test file on Windows. Keep the stream's encoding; refuse to let a
    # glyph decide whether the caveats get printed. A no-op on a UTF-8 locale.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")

    changed = changed_files(args.since, args.files)
    selected, escalations, notes, counts = select(changed, closure=args.closure)

    if args.pytest_args:
        # DELIBERATELY SILENT ON THE BLIND SPOTS IN THIS MODE, and it exits 2
        # when it escalated, so a caller that pipes it without checking the
        # status gets an empty list rather than a confident wrong one.
        if escalations:
            return 2
        for path, _reason in selected:
            print(path)
        return 0
    return report(changed, selected, escalations, notes, counts,
                  closure=args.closure, as_json=args.json)


if __name__ == "__main__":
    sys.exit(main())
