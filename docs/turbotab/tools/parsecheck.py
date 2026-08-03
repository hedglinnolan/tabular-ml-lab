"""The sixth gate: every tracked Python file parses. `TEST-042`.

**Why this exists, in one sentence the loop that caused it wrote:** *a check
whose input is a serialization of the thing it cares about will pass on a
broken input.*

L43-B committed an `IndentationError` in `ml/eda_actions.py` and every gate
stayed green. None of the five parses Python — the ledger and register gates
read JSON, the spelling gate reads generated markdown, the copy deck reads
strings, the evidence gate reads badges. And the loop's *own* new guard, which
sweeps shipped source for a contraindicated phrase, read the file with
`read_text` and matched regexes against it. A file that does not parse is
still text. The suite caught it three commits downstream, at collection, with
thirteen integration failures attached.

**What this is not.** It is not a substitute for running the tests, and nothing
about it should read as one. It answers exactly one question — *does this
compile* — and it answers it at the moment the mistake is made rather than
three commits later. A green parse gate says nothing about whether the code is
right.

**Why it is unconditional.** The hook's own comment already argues the case:
*conditional gates are how gates get skipped.* Measured on this tree, 363 files
in 0.44 s, which is a fifth of what the existing gates already cost.

**Why `ast.parse` and not `compile` or an import.** Importing runs module
bodies — `figure_specs` registers figures, `pages/*` reach for Streamlit — so
an import gate would be a test suite with worse error messages and a
dependency on the environment. `ast.parse` reads the file and answers the
syntax question and nothing else.
"""
from __future__ import annotations

import ast
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[3]


def tracked_python() -> list[pathlib.Path]:
    """Every `.py` file git tracks.

    **Tracked rather than globbed**, and the distinction is the whole reason
    this is not a `rglob`: a glob walks `venv/`, `node_modules/` and every
    stray scratch file, and would then need an exclusion list — which is a
    hand-maintained list of the kind this project keeps finding rotten. Git
    already knows what belongs to the repository.
    """
    out = subprocess.run(
        ["git", "ls-files", "-z", "*.py"],
        cwd=ROOT, capture_output=True, text=True, check=True)
    return [ROOT / name for name in out.stdout.split("\0") if name]


def _name(path: pathlib.Path) -> str:
    """Repo-relative where possible, absolute otherwise.

    `relative_to` RAISES on a path outside the root, and the first version of
    this called it unguarded — so pointing the checker at a file elsewhere on
    disk crashed with a `ValueError` about subpaths instead of reporting the
    syntax error it had just found. A reporting helper that can throw turns a
    found defect into a stack trace about the reporter.
    """
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def broken(paths) -> list[tuple[str, int, str]]:
    """`(name, line, message)` for each file that does not parse."""
    out = []
    for path in paths:
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:       # pragma: no cover
            out.append((_name(path), 0, f"unreadable: {exc}"))
            continue
        try:
            ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            out.append((_name(path), exc.lineno or 0,
                        exc.msg or "syntax error"))
    return out


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    verbose = "-v" in argv or "--verbose" in argv

    started = time.monotonic()
    paths = tracked_python()
    failures = broken(paths)
    elapsed = time.monotonic() - started

    if not paths:                                          # pragma: no cover
        print("no tracked Python files found — is this a git repository?")
        return 1

    if failures:
        print(f"{len(failures)} of {len(paths)} tracked Python files do not "
              f"parse:")
        for name, line, message in failures:
            print(f"  {name}:{line}: {message}")
        print()
        print("A file that does not compile cannot be imported, cannot be "
              "tested, and — the reason this gate exists — is still readable "
              "as TEXT by every check that greps source.")
        return 1

    if verbose:
        print(f"ok — {len(paths)} tracked Python files parse "
              f"({elapsed:.2f}s)")
    else:
        print(f"ok — {len(paths)} files parse")
    return 0


if __name__ == "__main__":                                 # pragma: no cover
    raise SystemExit(main())
