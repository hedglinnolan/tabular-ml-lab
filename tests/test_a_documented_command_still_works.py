"""`TEST-090` — nothing checked that a documented command still works.

**The class.** A document that tells a person how to start or test the app goes
stale silently: a `make` target is renamed, a script moves, and the sentence
around it reads exactly as it did. `L61` swept 17 files for the instances and
filed the class; this is the guard.

**What it checks.** A `make` target named in an instructional document exists
in the `Makefile`, and a script path named in a command exists on disk. Cheap,
total, and it catches the ordinary rot.

**What it no longer checks, and why.** The file used to have a second half:
the documents named an interpreter (`venv/bin/python`) for the legacy TurboTab
launcher (`scripts/serve_turbotab.py`, `make turbotab`), and that half asked the
interpreter whether it could build the model shelf. The launcher retired with
the legacy app (BLUEPRINT §9.1, 42c6d9f6), and that half retired with it on
2026-10-08, by Nolan's ruling on Classic's `tests/`. TurboTab v2's launcher
(`turbotab/deploy/launch.py`) builds its own environment from
`turbotab/server/requirements.txt`, and `.github/workflows/v2.yml` starts it on
macOS and Windows.

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

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: The documents a person or an agent is TOLD to follow. Not every markdown file
#: in the repository — a drive report quoting a command is a record of what
#: somebody ran, and holding a historical record to today's tree would make the
#: guard fire on the truth. The legacy app's instructions (its README, `LOOP.md`
#: and `prompts/AGENT_ONBOARD.md`) are history in `docs/turbotab/archive/` and
#: are not held to today's tree.
INSTRUCTIONAL = (
    "README.md",
    "QUICKSTART.md",
    "CONTRIBUTING.md",
    "turbotab/README.md",
    "docs/turbotab-next/DEPLOY.md",
)

#: A fenced block, paired fence to fence. Pairing by line start matters: a
#: pattern that let a closing fence open the next match would read the prose
#: between two blocks as commands.
_FENCE = re.compile(r"^```([\w-]*)[^\n]*\n(.*?)^```", re.DOTALL | re.MULTILINE)
#: The fences whose lines are commands. An unlabeled fence counts, as it always
#: has here; `nginx`, `python` and the like do not.
_SHELL = {"", "bash", "sh", "shell", "console", "powershell"}
_MAKE_TARGET = re.compile(r"\bmake\s+([a-z][a-z0-9_-]*)")
_SCRIPT = re.compile(r"\b((?:scripts|turbotab/deploy)/[\w/.-]+\.(?:py|sh|ps1))\b")


def _documents():
    for name in INSTRUCTIONAL:
        path = ROOT / name
        if path.exists():
            yield name, path.read_text(encoding="utf-8")


def _commands_in(text: str):
    """Every line inside a fenced shell block of `text`."""
    out = []
    for language, block in _FENCE.findall(text):
        if language.lower() not in _SHELL:
            continue
        for line in block.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                out.append(line)
    return out


def _commands():
    """Every command line, with the document it came from."""
    return [(name, line) for name, text in _documents()
            for line in _commands_in(text)]


def _declared_targets() -> set:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    return set(re.findall(r"^([a-z][a-z0-9_-]*):", makefile, re.MULTILINE))


# ── the sweep can see what it is looking for ────────────────────────────────

def test_the_documents_are_there_and_carry_commands(capsys):
    """**The positive control, first.** Every assertion below is about the
    absence of a broken command, and a parse that found nothing would report a
    clean repository it never read."""
    found = [name for name, _ in _documents()]
    assert found == list(INSTRUCTIONAL), (
        f"only {found} of {INSTRUCTIONAL} were found; a document named here "
        f"moved or was removed")
    commands = _commands()
    assert len(commands) >= 20, (
        f"parsed {len(commands)} commands out of {len(found)} documents; the "
        f"fence pattern is probably wrong")
    with capsys.disabled():
        print(f"\n  {len(commands)} commands in {len(found)} instructional "
              f"documents")


def test_the_fence_parser_reads_commands_and_not_the_prose_between_blocks():
    """The parser's own control. The first pattern accepted a closing fence as
    an opening one, so a block in a language it skips (`nginx`) let the prose
    after it be read as commands."""
    text = ("```nginx\nlocation / {}\n```\n\nThen make sure it runs.\n\n"
            "```bash\nmake serve\n```\n")
    assert _commands_in(text) == ["make serve"]


# ── the claims that can be checked from the file ────────────────────────────

def test_every_documented_make_target_exists():
    """A renamed target leaves the sentence looking exactly as it did."""
    declared = _declared_targets()
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


def test_the_checks_fire_on_the_commands_the_retirement_left_behind():
    """**The negative control, quoted from README.md before 2026-10-08.**

    Both checks above are absences, and a matcher that fires on nothing has
    silence that means nothing. These two lines are the instance this file last
    caught: README.md told readers to start the retired legacy app with them.
    """
    assert "turbotab" not in _declared_targets()
    assert _MAKE_TARGET.findall("make turbotab") == ["turbotab"], (
        "the make-target matcher no longer reads a target out of a command")
    scripts = _SCRIPT.findall(
        "venv/bin/python scripts/serve_turbotab.py --port 8777")
    assert scripts == ["scripts/serve_turbotab.py"], scripts
    assert not (ROOT / scripts[0]).exists()
    # And today's launcher is inside the rule, so the guard covers it.
    assert _SCRIPT.findall("bash turbotab/deploy/turbotab.sh") == [
        "turbotab/deploy/turbotab.sh"]
