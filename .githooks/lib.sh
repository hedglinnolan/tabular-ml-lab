#!/usr/bin/env bash
# Shared by pre-commit and pre-push: find the interpreter the gates run under.
#
# Both hooks called `python` directly. On a machine where the interpreter is
# `python3` and the project's dependencies live in `turbotab/.venv`, that is
# `command not found` — which the hooks' own `run()` reads as a RED GATE, so
# every commit is refused for a reason that has nothing to do with the code.
#
# That failure mode is worse than it sounds. A gate that cannot run looks
# exactly like a gate that failed, and the documented escape from a failing
# gate is `--no-verify`. So an unresolvable interpreter does not merely block
# work; it teaches the operator to bypass the gate, which is the one outcome
# LOOP.md's whole argument about `8127101` exists to prevent.
#
# Stated once here rather than twice in the hooks, per FEATURE_PARITY's
# principle-locality rule: a rule written in one place and applied in another
# is the same silence as a capability with no row.
#
# Order, most specific first:
#   $TURBOTAB_PYTHON   explicit override, for an operator who knows better
#   ./venv             the FULL environment (Makefile's `PYTHON`), where the
#                      whole suite runs
#   turbotab/.venv     the Guided door's minimal environment — pandas, numpy,
#                      FastAPI and nothing else. Enough for the four gates, and
#                      deliberately not enough for the suite: its EMPTINESS is
#                      a second, independent signal that the diagnose ->
#                      profile -> detect path needs no scikit-learn, and a
#                      signal that only exists while nothing is installed into
#                      it. Preferred second so the gates run under the fuller
#                      one when both exist.
#   python / python3   whatever the shell offers
set -uo pipefail

resolve_python() {
    if [ -n "${TURBOTAB_PYTHON:-}" ] && [ -x "${TURBOTAB_PYTHON}" ]; then
        printf '%s\n' "${TURBOTAB_PYTHON}"
        return 0
    fi
    local root main
    root=$(git rev-parse --show-toplevel 2>/dev/null) || root=.
    # THE MAIN WORKTREE TOO, AND THAT IS `TEST-108`. A linked worktree's
    # `--show-toplevel` is the WORKTREE's root, which has no `venv/` — every
    # agent worktree therefore fell through to bare `python3`, imported nothing,
    # and printed three ticks and three crosses. `git worktree list --porcelain`
    # names the main worktree first from every position and needs no flags;
    # `--git-common-dir` was the obvious probe and is wrong, because it is
    # ABSOLUTE inside a linked worktree and RELATIVE in a plain repo, so the
    # naive `dirname` of it yields `.` and works today only because both hooks
    # `cd` to the toplevel first.
    #
    # SEARCHED, NOT REPLACED: a worktree with its own venv still prefers it,
    # and $TURBOTAB_PYTHON still wins over both.
    main=$(git worktree list --porcelain 2>/dev/null | head -1 | cut -d' ' -f2-)
    local candidate_venv
    for candidate_venv in "${root}/venv" "${root}/turbotab/.venv" \
                          "${main}/venv" "${main}/turbotab/.venv"; do
        if [ -n "${candidate_venv}" ] && [ -x "${candidate_venv}/bin/python" ]; then
            printf '%s\n' "${candidate_venv}/bin/python"
            return 0
        fi
    done
    local candidate
    for candidate in python3 python; do
        if command -v "$candidate" >/dev/null 2>&1; then
            command -v "$candidate"
            return 0
        fi
    done
    return 1
}


# ── the third state: a gate that CANNOT RUN is not a gate that FAILED ────────
#
# `TEST-108`, and it is the half the row is really about. The fallback above
# returns a bare `python3` and exits 0, so the hooks' "no Python interpreter
# found" branch is UNREACHABLE — and the interpreter it hands back has none of
# the gates' dependencies. Every gate then dies on `ModuleNotFoundError`, the
# hooks print ✗ and `COMMIT REFUSED`, and the operator reads six red gates over
# code that is fine.
#
# That is worse than a blocked commit. This file's own header says why: the
# documented escape from a failing gate is `--no-verify`, so a gate that cannot
# run teaches the operator to bypass gates. A distinct state is the difference
# between "your code is wrong" and "this checkout cannot check your code".
gates_can_run() {
    "$1" - <<'PROBE' >/dev/null 2>&1
import importlib.util as u
import sys
missing = [m for m in ("pandas", "pytest") if u.find_spec(m) is None]
sys.exit(1 if missing else 0)
PROBE
}
