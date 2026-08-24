"""`TEST-042` — the gate that would have caught L43-B's own mistake.

L43-B committed an `IndentationError` in `ml/eda_actions.py` and every one of
the five pre-commit gates stayed green. They read JSON, generated markdown,
user-facing strings and evidence badges. **None of them reads Python as
Python.** And the guard that loop had *just written* — a sweep of shipped
source for a contraindicated phrase — read the file with `read_text` and
matched regexes against it, so it passed over a file that does not compile.

The sentence that generalizes it, from the loop that paid for it:

> **A check whose input is a serialization of the thing it cares about will
> pass on a broken input.**

The suite caught it three commits downstream, at collection, with thirteen
integration failures attached. This closes the window between the mistake and
the signal. It is **not** a substitute for running the tests and nothing here
should read as one.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOL = ROOT / "docs" / "turbotab" / "tools" / "parsecheck.py"
HOOK = ROOT / ".githooks" / "pre-commit"

sys.path.insert(0, str(TOOL.parent))
import parsecheck  # noqa: E402


def test_the_tree_parses():
    """The gate's own claim about the current tree, so a break shows up here
    as well as in the hook — a developer who has not run
    `git config core.hooksPath .githooks` still gets the signal."""
    failures = parsecheck.broken(parsecheck.tracked_python())
    assert not failures, failures


def test_it_sweeps_the_files_it_claims_to():
    """The positive control. Everything else here is *nothing was broken*,
    which passes hardest on a sweep that read nothing.

    Anchored on files rather than a bare count so a `git ls-files` that
    silently returns the wrong set fails here.
    """
    paths = parsecheck.tracked_python()
    assert len(paths) > 300, (
        f"the sweep found {len(paths)} tracked Python files and there were "
        f"363 when this was written — `git ls-files` has stopped finding them")
    names = {p.relative_to(ROOT).as_posix() for p in paths}
    for expected in ("ml/eda_actions.py",          # the file that broke
                     "turbotab/api.py",
                     "docs/turbotab/tools/ledger.py",
                     "pages/06_Train_and_Compare.py"):
        assert expected in names, f"{expected} is not in the swept set"
    assert not any("venv" in p.parts for p in paths), (
        "the sweep is walking the virtualenv, which is why it reads git's "
        "index rather than globbing")


def test_it_refuses_a_deliberately_broken_file(tmp_path):
    """**The check the adjudicator said would be run.**

    A gate that cannot fail is not a gate — trap 2, applied to a hook. This
    reproduces the exact defect L43-B shipped: a statement dedented out of the
    `if` block above it.
    """
    broken_source = (
        "def coach(ratio):\n"
        "    warnings = []\n"
        "    if ratio < 0.5:\n"
        "        # a comment where the body should be\n"
        "    warnings.append('dedented out of the if')\n"
        "    return warnings\n")
    path = tmp_path / "eda_actions_like.py"
    path.write_text(broken_source, encoding="utf-8")

    failures = parsecheck.broken([path])
    assert len(failures) == 1, failures
    _name, line, message = failures[0]
    assert line, "the failure does not name a line"
    assert message, "the failure does not name a reason"
    assert "expected an indented block" in message, message


def test_a_file_that_merely_imports_nothing_real_still_passes():
    """The negative control, and it is what keeps the gate cheap.

    `ast.parse` must not care whether the imports resolve, whether Streamlit
    is installed, or whether the module would run. An import-based gate would
    execute `figure_specs`' registration side effects and every `pages/`
    module's Streamlit calls — a test suite with worse error messages and a
    dependency on the environment.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "imports_nothing.py"
        path.write_text(
            "import a_module_that_does_not_exist\n"
            "from another import missing_name\n"
            "x = missing_name(1)\n", encoding="utf-8")
        assert parsecheck.broken([path]) == [], (
            "the gate rejected a file that parses but does not import — it is "
            "running code, which is the suite's job")


def test_the_hook_runs_it_unconditionally():
    """The capability has to ship with its consumer — trap 1.

    A parse checker nothing calls is a module with a green test and no effect,
    which is this codebase's oldest habit.
    """
    hook = HOOK.read_text(encoding="utf-8")
    assert "parsecheck.py" in hook, (
        "the hook does not call the parse gate, so the gate does not run")
    line = next(ln for ln in hook.splitlines() if "parsecheck.py" in ln)
    assert line.strip().startswith("run "), (
        f"the parse gate is not invoked through `run`, so its failure would "
        f"not set FAILED and would not refuse the commit: {line.strip()!r}")
    assert " if " not in line and "[ " not in line, (
        f"the parse gate is conditional: {line.strip()!r}. The hook's own "
        f"comment says conditional gates are how gates get skipped.")


def test_the_hook_still_refuses_when_any_gate_is_red():
    """The wiring the gate depends on. `run` sets `FAILED`; `FAILED` exits 1.

    Pinned because commit `8127101` went out with a red gate for exactly this
    reason — the gates were chained with a newline instead of `&&`.
    """
    hook = HOOK.read_text(encoding="utf-8")
    assert "FAILED=1" in hook and 'if [ "$FAILED" -ne 0 ]; then' in hook
    assert "COMMIT REFUSED" in hook


def test_it_exits_nonzero_when_something_does_not_parse(tmp_path, monkeypatch):
    """End to end through `main`, because the hook reads the exit code and
    nothing else. A tool that prints a failure and exits 0 is silent where it
    matters."""
    bad = tmp_path / "bad.py"
    bad.write_text("def f(:\n", encoding="utf-8")
    monkeypatch.setattr(parsecheck, "tracked_python", lambda: [bad])
    assert parsecheck.main([]) == 1

    good = tmp_path / "good.py"
    good.write_text("def f():\n    return 1\n", encoding="utf-8")
    monkeypatch.setattr(parsecheck, "tracked_python", lambda: [good])
    assert parsecheck.main([]) == 0


@pytest.mark.parametrize("flag", ["-v", "--verbose"])
def test_the_tool_runs_from_the_command_line(flag):
    """Driven rather than asserted — trap 5. The hook calls a subprocess, so
    the subprocess is what has to work."""
    out = subprocess.run([sys.executable, str(TOOL), flag],
                         cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "files parse" in out.stdout, out.stdout
