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
#   turbotab/.venv     this repo's own environment, where the gates' deps live
#   python / python3   whatever the shell offers
set -uo pipefail

resolve_python() {
    if [ -n "${TURBOTAB_PYTHON:-}" ] && [ -x "${TURBOTAB_PYTHON}" ]; then
        printf '%s\n' "${TURBOTAB_PYTHON}"
        return 0
    fi
    local root
    root=$(git rev-parse --show-toplevel 2>/dev/null) || root=.
    if [ -x "${root}/turbotab/.venv/bin/python" ]; then
        printf '%s\n' "${root}/turbotab/.venv/bin/python"
        return 0
    fi
    local candidate
    for candidate in python3 python; do
        if command -v "$candidate" >/dev/null 2>&1; then
            command -v "$candidate"
            return 0
        fi
    done
    return 1
}
