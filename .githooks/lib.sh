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
#                      FastAPI and nothing else. Deliberately not enough for
#                      the suite: its EMPTINESS is a second, independent signal
#                      that the diagnose -> profile -> detect path needs no
#                      scikit-learn, and a signal that only exists while
#                      nothing is installed into it. Preferred second so the
#                      gates run under the fuller one when both exist.
#
#                      THIS LINE SAID "Enough for the four gates" AND IT WAS
#                      FALSE IN BOTH HALVES. There are six, and this
#                      interpreter runs five: `evidence.py check` imports every
#                      `turbotab.*` module, one of which reaches `sklearn`,
#                      which is the package this environment exists to lack. So
#                      the emptiness that makes it a signal is the same
#                      emptiness that stops it gating — measured, not reasoned,
#                      and the reason `gates_can_run` below probes four names
#                      rather than two.
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
    local candidate_venv candidate_python
    for candidate_venv in "${root}/venv" "${root}/turbotab/.venv" \
                          "${main}/venv" "${main}/turbotab/.venv"; do
        # BOTH LAYOUTS. A virtualenv keeps its interpreter at `bin/python` on
        # POSIX and at `Scripts/python.exe` on Windows, where git runs these
        # hooks under the sh it ships with. With only the first spelling, a
        # Windows checkout with a fully provisioned `venv/` fell straight
        # through to whatever the shell offered — the Microsoft Store's
        # `python3` alias, which is a download prompt rather than an
        # interpreter — and the hook printed GATES CANNOT RUN over an
        # environment that could run every one of them.
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
#
# ── and the probe enumerated a SUBSET of what the gates import ───────────────
#
# `TEST-110`, found by driving `TEST-108`'s own fix. The list here was
# `("pandas", "pytest")` while the six gates import FOUR third-party packages
# directly from first-party code. An interpreter carrying the two probed names
# and missing either of the others passed this check and then produced the
# exact cross this state exists to prevent.
#
# IT WAS LIVE RATHER THAN HYPOTHETICAL, and in an environment listed twelve
# lines above as a gate interpreter: under `turbotab/.venv` the hook printed
# five ticks, `✗ evidence badges … No module named 'sklearn'` and `COMMIT
# REFUSED`. `resolve_python` selects that interpreter whenever `venv/` is
# absent, which is the state `LOOP.md` §05 already records the Makefile being
# found in.
#
# THE NAMES ARE MEASURED, NOT RECALLED. Each gate was run under an import
# recorder that logs only FIRST-PARTY -> THIRD-PARTY edges, so a package
# reached through another package is not counted twice:
#
#   python parses      —                       (stdlib only)
#   ledger schema      —                       (stdlib only)
#   register schema    —                       (stdlib only)
#   American spelling  pytest, pandas
#   copy deck          pandas
#   evidence badges    fastapi, sklearn, pandas, pydantic
#
# `numpy`, `pydantic` and `_pytest` are omitted because each is a hard
# requirement of a name already listed and cannot be absent while that one is
# present. `tests/test_the_pre_commit_hook_can_run_where_it_is_run.py` re-takes
# that measurement and fails when a gate grows a dependency this list does not
# carry — because naming today's four is how this recurs.
#
# It reports WHICH names are missing rather than a bare status, so the banner
# can name them instead of restating the two it used to assume.
GATES_MISSING=""
gates_can_run() {
    GATES_MISSING=$("$1" - <<'PROBE' 2>/dev/null
import importlib.util as u

missing = []
for name in ("pandas", "pytest", "fastapi", "sklearn"):
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
