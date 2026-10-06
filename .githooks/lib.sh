#!/usr/bin/env bash
# Sourced by pre-commit: find the interpreter the gates run under, and say
# whether it can run them.
#
# An interpreter that cannot run a gate must not read as a gate that failed: the
# documented escape from a failing gate is `--no-verify`, so a gate that cannot
# run teaches the operator to bypass gates.
#
# Order, most specific first:
#   $TURBOTAB_PYTHON   explicit override, for an operator who knows better
#   ./venv             the full environment (the Makefile's `PYTHON`)
#   turbotab/.venv     a minimal environment some checkouts still carry
#   the same two in the MAIN worktree, because a linked worktree's toplevel has
#                      no venv of its own
#   python / python3   whatever the shell offers
set -uo pipefail

resolve_python() {
    if [ -n "${TURBOTAB_PYTHON:-}" ] && [ -x "${TURBOTAB_PYTHON}" ]; then
        printf '%s\n' "${TURBOTAB_PYTHON}"
        return 0
    fi
    local root main
    root=$(git rev-parse --show-toplevel 2>/dev/null) || root=.
    # `git worktree list --porcelain` names the main worktree first from every
    # position; `--git-common-dir` is absolute inside a linked worktree and
    # relative in a plain repo, so its `dirname` is not a safe probe.
    main=$(git worktree list --porcelain 2>/dev/null | head -1 | cut -d' ' -f2-)
    local candidate_venv candidate_python
    for candidate_venv in "${root}/venv" "${root}/turbotab/.venv" \
                          "${main}/venv" "${main}/turbotab/.venv"; do
        # Both layouts: `bin/python` on POSIX, `Scripts/python.exe` on Windows,
        # where git runs hooks under the sh it ships with.
        for candidate_python in "${candidate_venv}/bin/python" \
                                "${candidate_venv}/Scripts/python.exe"; do
            if [ -n "${candidate_venv}" ] && [ -x "${candidate_python}" ]; then
                printf '%s\n' "${candidate_python}"
                return 0
            fi
        done
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


# The third state: a gate that CANNOT RUN is not a gate that FAILED.
#
# The fallback above can hand back a bare `python3` that has none of the gates'
# dependencies. The probe names the third-party packages the two gates import:
#
#   python parses      —                (stdlib only)
#   American spelling  pytest, pandas   (pandas and numpy through tests/conftest.py;
#                                        numpy is a hard requirement of pandas)
#
# It reports WHICH names are missing, so the banner can name them.
GATES_MISSING=""
gates_can_run() {
    GATES_MISSING=$("$1" - <<'PROBE' 2>/dev/null
import importlib.util as u

missing = []
for name in ("pandas", "pytest"):
    try:
        present = u.find_spec(name) is not None
    except Exception:
        present = False          # a broken install is an absent one here
    if not present:
        missing.append(name)
print(" ".join(missing))
PROBE
    ) || GATES_MISSING="a working python"
    [ -z "${GATES_MISSING}" ]
}
