"""`TEST-108` — the pre-commit hook, proved to run where an agent runs it.

## The debt this file pays

`L63` found the hook red in every agent worktree: `resolve_python` asked
`git rev-parse --show-toplevel`, which inside a linked worktree names the
*worktree* root, where no `venv/` exists — so the hook fell through to a bare
`python3`, imported nothing, and printed **three ticks and three crosses**
before `COMMIT REFUSED`. `L64-D2` repaired it. The disposition was then
downgraded to `PARTIAL`, because the row's named test —
`test_the_interpreter_the_documents_name_can_build_the_model_shelf` — has a
zero-line diff across that loop, never mentions `.githooks`, and **skips in a
linked worktree**, which is the exact environment the row is about. It cannot
go red for the fix in any state.

This is the assertion the row is owed, and it is driven rather than read.

## Three things that make the naive form of it wrong

**Do not assert on the return code.** The third state — `GATES CANNOT RUN` —
still exits 1. It changes the message, not the blocking, which is the whole
point of it. A test keyed on the exit status cannot tell the repaired hook from
the broken one.

**Do not assert six ticks.** The `python parses` gate is interpreter-*version*
dependent: under `/usr/bin/python3` (3.9.6) it reports a false failure on an
f-string this repository legitimately contains. A cross there is a real answer
about a real interpreter and is not this row's subject.

**Assert what the row actually claims: no cross attributable to a missing
module.** That is true of a healthy hook and true of the `GATES CANNOT RUN`
state — and it was false in the state that filed the row.

## Why the worktree is made by hand rather than by `worktree.py`

`docs/turbotab/tools/worktree.py add` is the project's tool and is the wrong
instrument here, for three measured reasons. It runs `git worktree add -B
wt-<name>`, which **force-writes a shared branch ref**, while
`tests/test_a_fixed_row_names_a_test_that_actually_runs.py` runs every `FIXED`
row's named node under `-n auto` — so two concurrent runs would collide on one
ref. It hard-fails when `.worktrees/<name>` survives a crashed run, wedging the
test until somebody cleans it by hand. And it plants a nested checkout inside
the repository.

`git worktree add --detach` into `tmp_path` has none of those properties: git
derives the worktree's registered name from the basename and **auto-suffixes a
collision** (driven: two adds of `hookwt` register `hookwt` and `hookwt1`), the
directory is outside the checkout, and a `--detach` add writes only
`.git/worktrees/<name>/` plus the new directory — verified with
`git status --porcelain` empty and `HEAD` unchanged across an add/remove cycle,
which is why `AGENT_ONBOARD.md` §06's tree-wide-git rule (`stash`, `checkout`,
`clean`, `reset`, `restore`) does not reach it.

**And one trap that is not the one you would guess.** `.worktrees/` is already
excluded by `tests/repo_write_guard.py` and `.gitignore`. The hazard is
`UNRESOLVED_CEILING = 32` in
`tests/test_no_test_writes_a_path_git_tracks.py`, which the live corpus sits at
**exactly** — zero headroom, so one new unresolvable write destination turns
that guard red. Hence the shape below: the worktree path is **bound to a name
first** and only `str(name)` reaches `argv`; composing `str(tmp_path / "wt")`
inline adds an unresolved destination, and so does any `.mkdir()` on a
`tmp_path`-derived name. `git worktree add` creates the directory itself, so no
`mkdir` is needed, and this file writes no file at all.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HOOK = ROOT / ".githooks" / "pre-commit"
LIB = ROOT / ".githooks" / "lib.sh"
PROBE_DIR = ROOT / "tests" / "gate_import_probe"

#: A gate's cross is only this row's subject when the reason is an interpreter
#: that could not import something. `command not found` is the same failure one
#: layer down — it is what `resolve_python` produced before `L61`.
_MISSING_MODULE = re.compile(
    r"ModuleNotFoundError|No module named|command not found")

_TICK = "  ✓ "
_CROSS = "  ✗ "


def _gate_labels() -> list:
    """The labels the hook prints, read out of the hook rather than retyped."""
    return re.findall(r'^run\s+"([^"]+)"', HOOK.read_text(encoding="utf-8"),
                      re.MULTILINE)


def _crosses(out: str) -> dict:
    """`{label: reported output}` for every gate the hook marked failed."""
    failures: dict = {}
    label = None
    for line in out.splitlines():
        if line.startswith(_CROSS):
            label = line[len(_CROSS):].strip()
            failures[label] = []
        elif line.startswith(_TICK):
            label = None
        elif label is not None:
            failures[label].append(line)
    return {k: "\n".join(v) for k, v in failures.items()}


def _run_the_hook(cwd: Path, env_extra: dict = None):
    env = dict(os.environ)
    env.pop("TURBOTAB_PYTHON", None)
    env.update(env_extra or {})
    return subprocess.run([str(HOOK)], cwd=str(cwd), env=env,
                          capture_output=True, text=True, timeout=600)


def _main_checkout() -> Path:
    """The main worktree, which is where `resolve_python` must reach."""
    out = subprocess.run(["git", "worktree", "list", "--porcelain"],
                         cwd=str(ROOT), capture_output=True, text=True,
                         check=True)
    return Path(out.stdout.splitlines()[0].split(" ", 1)[1])


# ── the assertion the row is owed ───────────────────────────────────────────

def test_the_hook_finds_an_interpreter_from_inside_a_linked_worktree(tmp_path,
                                                                     capsys):
    """**`TEST-108`, driven in the environment the row is about.**

    Every adjudication fan-out this project runs hands its agents worktrees, and
    six of fourteen tripped this. The claim is narrow on purpose: not that the
    hook passes, but that **no gate is reported failed for want of a module**.
    """
    assert HOOK.exists() and os.access(HOOK, os.X_OK), (
        f"{HOOK} is missing or not executable, so this file asserts nothing")

    worktree = tmp_path / "hookwt"
    added = subprocess.run(
        ["git", "worktree", "add", "--detach", str(worktree), "HEAD"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=300)
    assert added.returncode == 0, (
        f"could not create the worktree this test is about:\n{added.stderr}")
    try:
        done = _run_the_hook(worktree)
    finally:
        subprocess.run(["git", "worktree", "remove", "--force",
                        str(worktree)],
                       cwd=str(ROOT), capture_output=True, text=True,
                       timeout=300)

    out = done.stdout + done.stderr

    # THE POSITIVE CONTROL, BEFORE THE ABSENCE IS QUOTED. The assertion below
    # is that a set is empty, and a hook that died before printing anything
    # produces an empty set too (§07 trap 5c: a negative assertion over a
    # filtered population is not a check until something proves the population
    # is non-empty).
    labels = _gate_labels()
    assert len(labels) == 6, (
        f"parsed {labels} out of {HOOK.name}; the hook's own gate list is not "
        f"being read, so the reconciliation below means nothing")
    reached = [name for name in labels
               if f"{_TICK}{name}" in out or f"{_CROSS}{name}" in out]
    cannot_run = "GATES CANNOT RUN" in out
    assert reached or cannot_run, (
        f"the hook reported on none of {labels} and did not print the "
        f"cannot-run banner either. It produced:\n{out[:2000]}")

    blamed = {label: why for label, why in _crosses(out).items()
              if _MISSING_MODULE.search(why)}
    assert not blamed, (
        "inside a linked worktree the hook reported "
        + ", ".join(sorted(blamed))
        + " failed for want of a module. That is `TEST-108`: the interpreter "
          "was resolved from the WORKTREE root, which has no environment, so "
          "the gates died on imports and the operator is shown red gates over "
          "code that is fine — which teaches `--no-verify`, the one outcome "
          "the hook exists to prevent.\n\n"
        + "\n\n".join(f"{k}:\n{v}" for k, v in sorted(blamed.items())))

    with capsys.disabled():
        print(f"\n  worktree hook: exit {done.returncode} · "
              f"{len(reached)}/{len(labels)} gates reported · "
              f"{len(_crosses(out))} cross(es), "
              f"{len(blamed)} attributable to a missing module")


def test_the_resolver_reaches_the_main_checkouts_environment(tmp_path):
    """The repair itself, rather than only its absence of symptoms.

    The test above passes in the `GATES CANNOT RUN` state, because that state
    prints no crosses at all — correctly, since it is the honest answer on a
    machine with no provisioned interpreter anywhere. This one asserts the
    thing `L64-D2` actually built: from inside a linked worktree,
    `resolve_python` finds the **main** checkout's `venv`.

    The precondition is read off the disk and named, per §07 trap 3d — a test
    that declines to look when the subject is absent is green over its own
    defect. Here the absence is genuinely environmental: nothing is asserted
    about a clone where nobody has run `make venv`.
    """
    provisioned = _main_checkout() / "venv" / "bin" / "python"
    if not provisioned.exists():                           # pragma: no cover
        pytest.skip(f"{provisioned} does not exist, so there is no "
                    f"environment for the resolver to reach")

    worktree = tmp_path / "hookwt"
    added = subprocess.run(
        ["git", "worktree", "add", "--detach", str(worktree), "HEAD"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=300)
    assert added.returncode == 0, added.stderr
    try:
        done = _run_the_hook(worktree)
    finally:
        subprocess.run(["git", "worktree", "remove", "--force",
                        str(worktree)],
                       cwd=str(ROOT), capture_output=True, text=True,
                       timeout=300)

    out = done.stdout + done.stderr
    assert "GATES CANNOT RUN" not in out, (
        f"a provisioned interpreter exists at {provisioned} and the hook did "
        f"not find it from inside a linked worktree. `resolve_python` searches "
        f"the worktree root and then the MAIN worktree, named by `git worktree "
        f"list --porcelain | head -1`; that search is what this asserts.\n"
        f"{out[:1500]}")
    for label in _gate_labels():
        assert f"{_TICK}{label}" in out or f"{_CROSS}{label}" in out, (
            f"the hook never reached the `{label}` gate:\n{out[:1500]}")


def test_the_detector_fires_on_the_output_that_filed_the_row():
    """**The negative control, quoted from `TEST-108`'s own evidence field.**

    Every assertion above is an absence, and a matcher that recognizes no
    failure has silence that means nothing (§07 trap 5b). This is the hook's
    real pre-fix output, so the detector is checked against the state it exists
    to catch rather than against a state somebody invented.
    """
    historical = (
        "pre-commit gates:\n"
        "  ✓ python parses\n"
        "  ✓ ledger schema\n"
        "  ✓ register schema\n"
        "\n"
        "  ✗ American spelling\n"
        "      /usr/bin/python3: No module named pytest\n"
        "\n"
        "  ✗ copy deck\n"
        "      ModuleNotFoundError: No module named 'pandas'\n"
        "\n"
        "  ✗ evidence badges\n"
        "      ModuleNotFoundError: No module named 'pandas'\n"
        "\n"
        "COMMIT REFUSED — a gate is red.\n")
    crosses = _crosses(historical)
    assert set(crosses) == {"American spelling", "copy deck",
                            "evidence badges"}, crosses
    blamed = {k for k, v in crosses.items() if _MISSING_MODULE.search(v)}
    assert blamed == {"American spelling", "copy deck", "evidence badges"}, (
        f"the detector no longer recognizes the three crosses that filed "
        f"TEST-108, so its silence about today's hook means nothing: {blamed}")

    # And it must NOT fire on a gate that failed for a real reason, or every
    # red gate would read as an environment problem — a false green in the one
    # instrument that refuses commits.
    genuine = ("  ✗ ledger schema\n"
               "      FIXED row GUIDED-001 names no regression test\n")
    assert not {k for k, v in _crosses(genuine).items()
                if _MISSING_MODULE.search(v)}, (
        "the detector fires on a gate that failed for a real reason")


# ── `TEST-110` · the probe enumerated a subset of what the gates import ─────

#: Packages that cannot be absent while a probed package is present, each with
#: the reason. A dict rather than a set, for the reason `rankings.py` gives
#: about scopes: an exemption with no argument is a classification nobody can
#: revisit. The assertion below forces a NEW package into one list or the
#: other, which is the whole point — naming today's four is how this recurs.
GUARANTEED_BY = {
    "numpy": "pandas declares numpy as an install requirement",
    "pydantic": "fastapi declares pydantic as an install requirement",
    "_pytest": "the same distribution as pytest",
    "scipy": "scikit-learn declares scipy as an install requirement",
    "joblib": "scikit-learn declares joblib as an install requirement",
    "dateutil": "pandas declares python-dateutil as an install requirement",
    "pytz": "pandas declares pytz as an install requirement",
    "starlette": "fastapi declares starlette as an install requirement",
    # Not transitive — optional by construction. tests/conftest.py imports
    # streamlit behind try/except ImportError (for the AppTest.from_file
    # repo-root shim) and stays fully functional without it, so a
    # streamlit-less hook interpreter cannot produce the `✗ … No module
    # named …` state this probe exists to prevent: the import simply
    # doesn't happen there. The tracer cannot see the guard, so the
    # argument lives here instead.
    "streamlit": "guarded optional import in tests/conftest.py; absence is "
                 "handled, not fatal",
}


def _probed_names() -> list:
    """The probe's list, parsed out of `lib.sh` rather than restated here."""
    text = LIB.read_text(encoding="utf-8")
    match = re.search(r"for name in \(([^)]*)\)", text)
    assert match, "gates_can_run no longer has a parsable module list"
    return re.findall(r'"([^"]+)"', match.group(1))


def _measure_direct_imports(tmp_path) -> dict:
    """`{gate label: {top-level package}}`, measured by running each gate."""
    python = _main_checkout() / "venv" / "bin" / "python"
    commands = {
        "python parses": ["docs/turbotab/tools/parsecheck.py"],
        "ledger schema": ["docs/turbotab/tools/ledger.py", "check"],
        "register schema": ["docs/turbotab/tools/register.py", "check"],
        "American spelling": ["-m", "pytest",
                              "tests/test_american_spelling.py", "-q",
                              "--no-header"],
        "copy deck": ["docs/turbotab/tools/copydeck.py", "check"],
        "evidence badges": ["docs/turbotab/tools/evidence.py", "check"],
    }
    measured = {}
    for label, argv in commands.items():
        out = tmp_path / f"{label.replace(' ', '_')}.json"
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(PROBE_DIR)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH")
                                else []))
        env["GATE_PROBE_ROOT"] = str(ROOT)
        env["GATE_PROBE_OUT"] = str(out)
        subprocess.run([str(python)] + argv, cwd=str(ROOT), env=env,
                       capture_output=True, text=True, timeout=600)
        edges = json.loads(out.read_text(encoding="utf-8")) \
            if out.exists() else []
        measured[label] = {top for _caller, top in edges}
    return measured


def test_the_probe_covers_every_package_the_gates_import_directly(tmp_path,
                                                                  capsys):
    """**`TEST-110`, and it is measured rather than read.**

    A static walk cannot answer this: `evidence.py:302` reaches `turbotab.api`
    — and `fastapi` through it — with
    `importlib.import_module(f"turbotab.{path.stem}")`, which no literal search
    sees. So each gate is run under an import recorder that logs only
    first-party -> third-party edges.
    """
    provisioned = _main_checkout() / "venv" / "bin" / "python"
    if not provisioned.exists():                           # pragma: no cover
        pytest.skip(f"{provisioned} does not exist, so no gate can be run")

    measured = _measure_direct_imports(tmp_path)
    everything = set().union(*measured.values())

    # The recorder's own control. An empty measurement would report a probe
    # that covers everything, in the same words as a probe that does.
    assert "pandas" in everything, (
        f"the import recorder observed {sorted(everything)} across the six "
        f"gates and did not see pandas, which two of them import at module "
        f"level. The recorder is not running; its silence means nothing.")

    probed = _probed_names()
    uncovered = sorted(everything - set(probed) - set(GUARANTEED_BY))
    assert not uncovered, (
        f"the six gates import {uncovered} directly from first-party code and "
        f"`gates_can_run` does not probe for it. An interpreter carrying "
        f"{probed} and missing {uncovered} passes the probe and then produces "
        f"the exact `✗ … No module named …` the cannot-run state exists "
        f"to prevent. Add it to the probe in .githooks/lib.sh, or to "
        f"GUARANTEED_BY here with the requirement that makes it redundant.\n"
        f"  measured: "
        + "; ".join(f"{k} -> {sorted(v)}" for k, v in measured.items()))

    with capsys.disabled():
        print(f"\n  probe: {probed} · gates import {sorted(everything)} "
              f"directly · {len(GUARANTEED_BY)} declared transitive")


def test_the_coverage_check_fails_on_the_list_that_filed_the_row(tmp_path):
    """**The positive control, and it re-derives `TEST-110`'s own instance.**

    The assertion above is `not uncovered`, which is empty both when the probe
    is complete and when the measurement collapsed. This replays it against
    `("pandas", "pytest")` — the list `lib.sh` carried when the row was filed —
    and requires it to come back short.
    """
    provisioned = _main_checkout() / "venv" / "bin" / "python"
    if not provisioned.exists():                           # pragma: no cover
        pytest.skip(f"{provisioned} does not exist, so no gate can be run")

    everything = set().union(*_measure_direct_imports(tmp_path).values())
    missed = sorted(everything - {"pandas", "pytest"} - set(GUARANTEED_BY))
    assert missed, (
        "the historical two-name probe now covers everything the gates import, "
        "so the check above can no longer distinguish a complete probe from a "
        "broken measurement")
    assert "sklearn" in missed, (
        f"scikit-learn is what made TEST-110 live rather than hypothetical — "
        f"`turbotab/.venv`, named in lib.sh's own header as a gate "
        f"interpreter, carries pandas and pytest and not sklearn, and the hook "
        f"printed five ticks and `✗ evidence badges … No module named "
        f"'sklearn'` under it. The recorder no longer sees that edge: {missed}")
