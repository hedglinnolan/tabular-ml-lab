"""`TEST-090` — nothing checked that a documented command still works.

**The class, and it cost four human drives.** `turbotab/README.md`'s "Run it"
section was the only document in the repository that said how to start TurboTab.
It told you to build `turbotab/.venv` from `turbotab/requirements.txt` and run
uvicorn from that interpreter — an environment deliberately empty of
scikit-learn, with `tests/test_the_guided_door_installs_without_the_app.py`
keeping it that way. **So the documented launch was guaranteed to lose `GET
/models`, and a test held the guarantee in place.** The instruction was correct
when written, TurboTab having had no training step, and nothing announced the
expiry.

`L61` swept the instances — 17 files, 19 lines — and filed the class unbuilt
for want of room. This is the guard.

## Two halves, because the cheap one would not have caught it

**Structural.** A `make` target named in a document exists in the `Makefile`; a
script path named in a command exists on disk. Cheap, total, and it catches the
ordinary rot — a renamed target, a moved script.

**And it would have missed `TEST-090`'s own instance**, which is why the second
half is here: `turbotab/.venv/bin/python` *existed*. What was false was not the
path but the claim around it. So a document that says *this is how you start the
app* must name an interpreter that can actually build the model shelf, and that
is asked of the interpreter rather than of the sentence.

## What this does NOT do

It does not run arbitrary documented commands. A guard that executed every
fenced block in the repository would be a guard that installs packages, binds
ports and rewrites baselines, and the first time it did something irreversible
nobody would ever trust it again. It reads the commands and checks the claims
inside them that can be checked without side effects.
"""
from __future__ import annotations

import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: The documents a person or an agent is TOLD to follow. Not every markdown file
#: in the repository — a drive report quoting a command is a record of what
#: somebody ran, and holding a historical record to today's tree would make the
#: guard fire on the truth.
INSTRUCTIONAL = (
    "README.md",
    "turbotab/README.md",
    "docs/turbotab/README.md",
    "docs/turbotab/prompts/AGENT_ONBOARD.md",
    "docs/turbotab/LOOP.md",
)

#: Interpreters the repository builds, and what each is FOR. `turbotab/.venv` is
#: deliberately minimal and is not a mistake — it is the portability claim's
#: environment. What it may not be is the answer to *how do I run the app*.
FULL_ENVIRONMENT = "venv/bin/python"
GUIDED_DOOR_ENVIRONMENT = "turbotab/.venv"

#: A command that starts the server. Matched on the module path rather than on
#: the word "uvicorn", because `make turbotab` starts it too and names neither.
_STARTS_THE_APP = re.compile(r"uvicorn\s+turbotab\.api:app|serve_turbotab\.py"
                             r"|make\s+turbotab\b")
_FENCE = re.compile(r"```(?:bash|sh|shell|console|powershell)?\n(.*?)```",
                    re.DOTALL)
_MAKE_TARGET = re.compile(r"\bmake\s+([a-z][a-z0-9_-]*)")
_SCRIPT = re.compile(r"\b((?:scripts|docs/turbotab/tools)/[\w/]+\.py)\b")


def _documents():
    for name in INSTRUCTIONAL:
        path = ROOT / name
        if path.exists():
            yield name, path.read_text(encoding="utf-8")


def _commands():
    """Every line inside a fenced shell block, with the document it came from."""
    out = []
    for name, text in _documents():
        for block in _FENCE.findall(text):
            for line in block.splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    out.append((name, line))
    return out


# ── the sweep can see what it is looking for ────────────────────────────────

def test_the_documents_are_there_and_carry_commands(capsys):
    """**The positive control, first.** Every assertion below is about the
    absence of a broken command, and a parse that found nothing would report a
    clean repository it never read."""
    found = [name for name, _ in _documents()]
    assert len(found) >= 4, f"only {found} of {INSTRUCTIONAL} were found"
    commands = _commands()
    assert len(commands) >= 20, (
        f"parsed {len(commands)} commands out of {len(found)} documents; the "
        f"fence pattern is probably wrong")
    with capsys.disabled():
        print(f"\n  {len(commands)} commands in {len(found)} instructional "
              f"documents")


# ── half one · the claims that can be checked from the file ─────────────────

def test_every_documented_make_target_exists():
    """A renamed target leaves the sentence looking exactly as it did."""
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    declared = set(re.findall(r"^([a-z][a-z0-9_-]*):", makefile, re.MULTILINE))
    assert declared, "no targets parsed out of the Makefile"
    missing = sorted({
        f"{name}: make {target}"
        for name, line in _commands()
        for target in _MAKE_TARGET.findall(line)
        if target not in declared})
    assert not missing, (
        f"these documents name a make target that does not exist: {missing}. "
        f"Declared: {sorted(declared)}")


def test_every_documented_script_path_exists():
    """A moved script is the same failure one directory over."""
    missing = sorted({
        f"{name}: {script}"
        for name, line in _commands()
        for script in _SCRIPT.findall(line)
        if not (ROOT / script).exists()})
    assert not missing, (
        f"these documents name a script that is not on disk: {missing}")


# ── half two · the claim that cost four drives ──────────────────────────────

#: Ordered MOST SPECIFIC FIRST, and the negative control below is why.
#:
#: The first draft listed `venv/bin/python` first and asked `candidate in line`.
#: `"venv/bin/python" in "turbotab/.venv/bin/python"` is **True** — so the exact
#: sentence that cost four drives was classified as the FULL environment and the
#: rule would have passed over it. A matcher that fires on a substring, in the
#: guard written to catch a matcher problem. Caught by
#: `test_the_rule_fires_on_the_sentence_that_cost_four_drives`, which is what a
#: negative control is for.
_INTERPRETERS = ("turbotab/.venv/bin/python",
                 "turbotab/.venv/Scripts/python.exe",
                 "turbotab/.venv/Scripts/python",
                 "turbotab\\.venv\\Scripts\\python.exe",
                 FULL_ENVIRONMENT)


def _named_interpreter(line: str) -> str | None:
    for candidate in _INTERPRETERS:
        if candidate in line:
            return candidate
    return None


def _is_the_minimal_venv(named: str | None) -> bool:
    """Whether `named` is the Guided door's environment, in either spelling."""
    if not named:
        return False
    return named.replace("\\", "/").startswith(GUIDED_DOOR_ENVIRONMENT)


def test_no_document_tells_you_to_start_the_app_from_the_minimal_venv():
    """**`TEST-090`'s own instance, as a rule rather than as a sweep.**

    The structural half above would have passed on it: the interpreter existed.
    What was false was the claim wrapped around it.
    """
    offenders = sorted({
        f"{name}: {line}"
        for name, line in _commands()
        if _STARTS_THE_APP.search(line)
        and _is_the_minimal_venv(_named_interpreter(line))})
    assert not offenders, (
        f"these documents tell a reader to start the app from "
        f"{GUIDED_DOOR_ENVIRONMENT}, which "
        f"tests/test_the_guided_door_installs_without_the_app.py keeps empty "
        f"of scikit-learn: {offenders}. Every model-shelf request under that "
        f"interpreter fails, on every file and every target. Use `make "
        f"turbotab`.")


def test_the_interpreter_the_documents_name_can_build_the_model_shelf():
    """And the claim is asked of the INTERPRETER, not of the sentence.

    The rule above is a blocklist and would pass a document naming some third
    environment nobody has checked. This asks the one the documents do name
    whether it can import the stack — in a subprocess, because the question is
    about that interpreter and asking it here would answer about this one,
    which is the whole of what four drives got wrong.
    """
    python = ROOT / FULL_ENVIRONMENT
    if not python.exists():                                # pragma: no cover
        pytest.skip(f"{FULL_ENVIRONMENT} is not built on this machine")

    named = {line for _n, line in _commands() if _STARTS_THE_APP.search(line)}
    assert named, "no document says how to start the app at all, which is the "\
                  "state TEST-090's instance was filed from"

    done = subprocess.run(
        [str(python), "-c",
         "import sys; sys.path.insert(0, '.');"
         "from ml import engine_stack;"
         "print(','.join(engine_stack.missing()))"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-800:]
    missing = [m for m in done.stdout.strip().split(",") if m]
    assert not missing, (
        f"{FULL_ENVIRONMENT} — the interpreter every documented launch names — "
        f"cannot import {missing}, so the documented command starts a server "
        f"that answers every model request with an unhandled "
        f"ModuleNotFoundError. That is DRIVE-035 exactly.")


def test_the_minimal_venv_is_still_minimal_which_is_why_the_rule_exists():
    """The rule above is only meaningful while the premise holds.

    If somebody installs scikit-learn into `turbotab/.venv`, starting the app
    from it stops being a defect and this file is forbidding a true sentence —
    `AGENT_ONBOARD.md` trap #3c. `test_the_guided_door_installs_without_the_app`
    is the file that owns that premise; this asserts it is still there to own
    it, rather than re-measuring the venv and becoming a second opinion.
    """
    owner = ROOT / "tests" / "test_the_guided_door_installs_without_the_app.py"
    assert owner.exists(), (
        "the file that keeps turbotab/.venv minimal is gone, so the rule above "
        "forbids a command that may now be perfectly correct")
    text = owner.read_text(encoding="utf-8")
    assert "sklearn" in text and "FORBIDDEN" in text, (
        "that file no longer asserts scikit-learn stays out of the Guided "
        "door's environment")


def test_the_rule_fires_on_the_sentence_that_cost_four_drives():
    """**The negative control, and it is not optional.**

    Every assertion above is an absence. A matcher that fires on nothing has
    silence that means nothing (trap 5b), and this one is checked against the
    exact line `turbotab/README.md` carried until `L61` — quoted from the
    commit that removed it.
    """
    historical = ("turbotab/.venv/bin/python -m uvicorn turbotab.api:app "
                  "--port 8777")
    assert _STARTS_THE_APP.search(historical), (
        "the matcher no longer recognizes a command that starts the app, so "
        "its silence about today's documents means nothing")
    assert _is_the_minimal_venv(_named_interpreter(historical)), (
        "the interpreter check no longer recognizes the minimal venv in the "
        "line that cost four drives")

    # And the Windows spelling, because the README carried both and a rule that
    # saw one of them would have passed the file it was written for.
    windows = ("turbotab\\.venv\\Scripts\\python.exe -m uvicorn "
               "turbotab.api:app --port 8777")
    assert _STARTS_THE_APP.search(windows), windows


def test_make_turbotab_is_recognized_as_starting_the_app():
    """The replacement must be inside the rule too.

    `make turbotab` names neither uvicorn nor an interpreter, so a matcher
    keyed on either would have stopped covering the launch the moment `L61`
    fixed it — the guard going quiet at exactly the point it started
    mattering.
    """
    assert _STARTS_THE_APP.search("make turbotab")
    assert _STARTS_THE_APP.search("make turbotab PORT=8899")
    assert _named_interpreter("make turbotab") is None
