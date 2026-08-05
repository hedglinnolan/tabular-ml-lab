"""Create a subagent worktree, and refuse one whose base is wrong.

## Why this exists

L49 fanned three subagents out through the harness's built-in worktree
isolation. It branched all three from a commit **367 commits behind `TurboTab`
and 16 ahead of it, with no `turbotab/` directory on disk at all.** Two of the
three started working anyway. The third checked its base first, found every file
its brief named absent, and refused — correctly, because a diff computed there
would have presented 367 commits of `turbotab/` as **newly added files** and its
own base's 16 as **deletions**. Applied by an orchestrator that trusted it, that
is a destructive patch wearing a fix's clothes.

**Nothing said the base was wrong.** That is the whole defect: a wrong base is
silent until a patch lands, and the only thing that caught it was one agent's
judgment. Judgment is not a gate — this project has a standing ruling that an
instruction a tired agent can skip by punctuation is not one either. So the
check moves into the tool that hands out the worktree.

## What it asserts, and why exactly these two

* **`HEAD` descends from `TurboTab`.** A base that is merely STALE is
  recoverable and often fine; a base that has DIVERGED produces a patch that
  deletes work. `git merge-base --is-ancestor` is the one command that tells
  those apart, and the failure message prints the ahead/behind counts because
  they are what a reader needs to decide which case they are in.
* **`turbotab/` exists on disk.** Cheap, and it catches the case that actually
  happened — a tree that is a valid checkout of the wrong thing.

Deliberately not asserted: that the worktree is clean (a fresh one always is),
or that `venv/` is present (it never is, and the briefs say how to run Python).
A check that fires on a correct setup gets switched off within a day.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BRANCH = "TurboTab"
#: What a subagent's brief will name. A worktree without these is not this repo.
REQUIRED = ("turbotab", "docs/turbotab/tools/revertprobe.py")


def _git(*args: str, cwd: Path = ROOT) -> str:
    out = subprocess.run(["git", *args], cwd=cwd, capture_output=True,
                         text=True)
    if out.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed:\n{out.stderr.strip()}")
    return out.stdout.strip()


def check(path: Path) -> None:
    """Refuse a worktree whose base could produce a destructive diff.

    Raises `SystemExit` with the counts, because *how far wrong* decides
    whether the answer is `merge --ff-only` or `stop and re-dispatch`.
    """
    if not path.exists():
        raise SystemExit(f"no worktree at {path}")

    head = _git("rev-parse", "HEAD", cwd=path)
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", head, BRANCH],
        cwd=ROOT, capture_output=True, text=True)
    if ancestor.returncode != 0:
        counts = _git("rev-list", "--left-right", "--count",
                      f"{BRANCH}...{head}")
        behind, ahead = counts.split()
        raise SystemExit(
            f"REFUSED: {path}\n"
            f"  HEAD {head[:8]} does not descend from {BRANCH} — it is "
            f"{behind} behind and {ahead} ahead.\n"
            f"  A diff computed here would present {behind} commits of this "
            f"repository as newly added files and this base's {ahead} as "
            f"deletions. That is a destructive patch, not a fix.\n"
            f"  This is L49's failure exactly, and it was silent.")

    missing = [name for name in REQUIRED if not (path / name).exists()]
    if missing:
        raise SystemExit(
            f"REFUSED: {path}\n"
            f"  HEAD {head[:8]} descends from {BRANCH} and yet {missing} are "
            f"not on disk. The worktree is a valid checkout of something else.")

    print(f"ok — {path} at {head[:8]}, descends from {BRANCH}, "
          f"{len(REQUIRED)} required paths present")


def add(name: str, base: str = "HEAD") -> Path:
    """Create the worktree, then check it. Never returns an unchecked path."""
    path = ROOT / ".worktrees" / name
    if path.exists():
        raise SystemExit(f"{path} already exists — remove it first")
    _git("worktree", "add", "-q", "-B", f"wt-{name}", str(path), base)
    check(path)
    return path


def remove(name: str) -> None:
    path = ROOT / ".worktrees" / name
    subprocess.run(["git", "worktree", "remove", "--force", str(path)],
                   cwd=ROOT, capture_output=True, text=True)
    subprocess.run(["git", "branch", "-D", f"wt-{name}"], cwd=ROOT,
                   capture_output=True, text=True)
    print(f"removed {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add"); a.add_argument("name")
    a.add_argument("--base", default="HEAD")
    r = sub.add_parser("remove"); r.add_argument("name")
    c = sub.add_parser("check"); c.add_argument("path")
    args = parser.parse_args()
    if args.cmd == "add":
        print(add(args.name, args.base))
    elif args.cmd == "remove":
        remove(args.name)
    else:
        check(Path(args.path))


if __name__ == "__main__":
    main()
