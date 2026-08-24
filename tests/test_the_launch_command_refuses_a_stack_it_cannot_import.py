"""`DRIVE-035` / `MODELS-026` — the launcher refuses before it binds the port.

**Four human drives were lost to this and the cause was never in the code.**
`GET /models` answered twenty-one characters of *Internal Server Error* on every
file and every target, because `ml/model_registry.py` imports the whole
estimator stack at module scope and the serving interpreter had none of it.
Every test, every probe and every number in four loops ran under `venv/`; every
browser request was answered by `turbotab/.venv`, whose inventory is `fastapi`
and `pandas`. **A reproduction attempt in process asks a different interpreter
the same question.**

So the check has to run in the interpreter that will serve, and it has to run
**before the port is bound** — a server that starts happily and then 500s at
Train is the failure, not the symptom.

**These tests break the import rather than read the script.** Each one launches
`scripts/serve_turbotab.py` as a real subprocess with a shadowing module on
`PYTHONPATH` that raises `ModuleNotFoundError` exactly as an absent package
would, and asserts on the exit status, the message and the port. Reading the
source would tell you the branch exists; it would not tell you it fires.
"""
from __future__ import annotations

import pathlib
import socket
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "serve_turbotab.py"

#: The three the shelf cannot be built without, in the order
#: `ml/model_registry.py` imports them.
STACK = ("sklearn", "xgboost", "lightgbm")


def _shadow(tmp_path: pathlib.Path, module: str) -> dict:
    """An environment where `module` is genuinely ABSENT, not merely broken.

    **The first version of this helper was wrong and the tests caught it**,
    which is the reason it gets a paragraph. It dropped a `<module>.py` on
    `PYTHONPATH` that raised on import — and `find_spec` FOUND that file, so
    the cheap probe reported the package present, the launcher started, and the
    test timed out waiting on a server it had asked to refuse. The mistake is
    the same one the code under test is about: *found* and *importable* are
    different predicates.

    A `sitecustomize` blocker is the honest simulation. `site` imports it at
    interpreter startup, before anything else runs, and a `meta_path` finder
    that raises on the target makes BOTH `find_spec` and a real `import` fail
    exactly as an absent package does. `turbotab/headless_train.py` blocks
    Streamlit the same way for the same reason.
    """
    (tmp_path / "sitecustomize.py").write_text(
        "import sys\n"
        "\n"
        "\n"
        "class _Absent:\n"
        f"    TARGET = {module!r}\n"
        "\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == self.TARGET or name.startswith(self.TARGET + '.'):\n"
        "            raise ModuleNotFoundError(f\"No module named {name!r}\")\n"
        "        return None\n"
        "\n"
        "\n"
        "sys.meta_path.insert(0, _Absent())\n",
        encoding="utf-8")
    import os
    return dict(os.environ, PYTHONPATH=str(tmp_path))


def _run(env=None, *args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=str(ROOT), capture_output=True, text=True, timeout=180, env=env)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _is_listening(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


@pytest.mark.parametrize("module", STACK)
def test_it_refuses_and_names_the_package_it_cannot_import(module, tmp_path):
    """The refusal, driven once per member of the stack.

    Parametrized rather than run on `sklearn` alone because **which import
    fails first depends on the environment, and that is why two competent
    diagnoses disagreed**: run 4's sandbox had scikit-learn and not xgboost, so
    its traceback named xgboost at line 18; the host's serving interpreter had
    neither and failed at line 6 on sklearn, twelve lines earlier. Both were
    real. The general statement is that the module imports the whole stack
    eagerly and the first absence wins, and a refusal that only handled the
    first one would be the same mistake in the fix.
    """
    done = _run(_shadow(tmp_path, module), "--check-only")
    assert done.returncode == 2, (done.returncode, done.stdout[-800:])
    out = done.stdout
    assert "REFUSED" in out, out[-800:]
    assert module in out, (
        f"the refusal does not name {module}, so a person reading it cannot "
        f"act on it: {out[-800:]}")
    assert sys.executable in out, (
        "the refusal does not name the interpreter, which is the fact three "
        "drives could not establish")


def test_the_refusal_names_the_distribution_and_not_the_import_name(tmp_path):
    """`pip install sklearn` installs a stub whose whole purpose is to tell you
    that you wanted a different name. The fix line has to say
    `scikit-learn`."""
    done = _run(_shadow(tmp_path, "sklearn"), "--check-only")
    assert "scikit-learn" in done.stdout, done.stdout[-800:]


def test_it_does_not_bind_the_port_when_it_refuses(tmp_path):
    """**The load-bearing half.** Naming the package is worth nothing if the
    server starts anyway — that is precisely the state four drives were in.

    A free port is chosen, the launcher is run against it with a broken stack,
    and the port must be closed both while it runs and after it exits.
    """
    port = _free_port()
    assert not _is_listening(port), "the chosen port was already in use"
    done = _run(_shadow(tmp_path, "sklearn"), "--port", str(port))
    assert done.returncode == 2, (done.returncode, done.stdout[-500:])
    assert "The port was not bound" in done.stdout
    assert not _is_listening(port), (
        f"something is listening on {port} after the launcher refused — the "
        f"refusal printed and started the server anyway, which is the exact "
        f"state DRIVE-035 was about")


def test_it_starts_when_the_stack_is_there(tmp_path):
    """**The positive control, and it is not optional.** Every assertion above
    is about a refusal, and a launcher that refused unconditionally would pass
    all of them."""
    done = _run(None, "--check-only")
    assert done.returncode == 0, done.stdout[-800:]
    assert "REFUSED" not in done.stdout
    assert "model shelf  ready" in done.stdout, done.stdout[-800:]
    for module in STACK:
        assert module in done.stdout


def test_the_first_lines_answer_which_interpreter_and_which_build():
    """`TEST-087`'s question, answered by the terminal rather than by the app.

    Three drives carried a version question. The build stamp answered *which
    code*; nothing answered *which Python*, and that is what decided run 4.
    Asserted on the first four lines because a fact printed after a page of
    uvicorn logging is a fact nobody reads.
    """
    done = _run(None, "--check-only")
    lines = [line for line in done.stdout.splitlines() if line.strip()][:4]
    assert lines[0] == "TurboTab", lines
    assert sys.executable in lines[1], lines
    assert sys.prefix in lines[2], lines
    assert lines[3].startswith("  build"), lines


def test_the_absent_extras_are_reported_and_are_not_a_refusal(tmp_path):
    """`TEST-038`'s standard, held to. `torch` is deliberately not installed
    and `shap` shortens one explanation rather than the shelf, so neither may
    stop the server — but silence about them would make an install that cannot
    draw a SHAP plot indistinguishable from one that can."""
    done = _run(_shadow(tmp_path, "shap"), "--check-only")
    assert done.returncode == 0, done.stdout[-800:]
    assert "REFUSED" not in done.stdout
    assert "shap" in done.stdout and "the shelf is unaffected" in done.stdout
