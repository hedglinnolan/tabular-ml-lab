"""The analyst's PreToolUse hook (DRIVER_PLAN D2), tested both ways: for an ``analyst`` it denies
every tool but Bash and every Bash command that is not exactly the driver's invocation, and lets
the driver's own commands through; for every other caller it says nothing, whatever the call.

The forbidden forms are the brief's (no ``;``, ``&&``, ``|``, backticks, ``$( )``, redirections or
newlines, nothing but ``python -m research.swarm.driver ...`` from the repo root) and zsh's, the
shell an analyst's Bash runs on this machine (process substitution ``=(...)``, glob qualifiers
that run code, globs, ``~``, braces, history), written out here as a person would type them, never
read from the hook. Whether a form runs anything is not taken on trust: each allowed form is run
under ``zsh`` (and ``bash``) in a scratch folder whose ``scripts/py`` only writes down the words it
was given, which must be ``shlex``'s words of the command, with nothing else made; and the zsh
forms are shown to run their payload there when no hook stands in the way."""
from __future__ import annotations

import io
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from research.swarm import analyst_hook as hook

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "research" / "swarm" / "analyst_hook.py"
DRIVER = "scripts/py -m research.swarm.driver"


def call(command=None, *, tool="Bash", agent="analyst", cwd=REPO, tool_input=None):
    out = {"session_id": "s", "transcript_path": "/tmp/t.jsonl", "cwd": str(cwd) if cwd else None,
           "hook_event_name": "PreToolUse", "tool_name": tool,
           "tool_input": tool_input if tool_input is not None else {"command": command}}
    if agent is not None:
        out["agent_type"] = agent
    return out


def denied(answer) -> bool:
    return (answer is not None
            and answer["hookSpecificOutput"]["hookEventName"] == "PreToolUse"
            and answer["hookSpecificOutput"]["permissionDecision"] == "deny"
            and bool(answer["hookSpecificOutput"]["permissionDecisionReason"].strip()))


ALLOWED = [
    f"{DRIVER} ask --run r1",
    f'{DRIVER} answer --run r1 impute --reason "No values are missing, so a fill changes nothing."',
    f"{DRIVER} answer --run r1 linear ridge --reason 'compare plain and penalized'",
    f"{DRIVER} answer --run r1 proposed --set age=excluded --reason 'age is recorded after admission'",
    f"{DRIVER} preview --run r1 complete_case",
    f"{DRIVER} revisit --run r1 0d27d9a3 --reason 'the event level was read the wrong way'",
    f"{DRIVER} fit --run r1",
    f"{DRIVER} results --run r1 --wait 60",
    f"{DRIVER} seal --run r1 linear --reason 'the simpler model scores alike'",
    f"{DRIVER} export --run r1",
    "python -m research.swarm.driver ask --run r1",
    "python3 -m research.swarm.driver fit --run r1",
    "./scripts/py -m research.swarm.driver results --run r1",
    # quoted, a reason may say anything plain words say: an apostrophe, parentheses, a colon
    f"{DRIVER} answer --run r1 ridge --reason \"The model's CV AUC (0.7668) is ridge's: [about] "
    f"the same as plain logistic, so ~ nothing moved? #3 {{a,b}} * ^x =(y) stay words.\"",
    f"{DRIVER} answer --run r1 ridge --reason 'Quoted once: \"ridge\" (penalized) ~ 0.77 * 2? !'",
    f"{DRIVER} answer --run r1 custom --set column=age --set low=-5.5 --set high=95 --reason 'a range'",
    f"{DRIVER} answer --run r1 x-1a2b3c4d --reason \"the way forward's id\"",
    f"{DRIVER} answer --run r1 time:los fixed_effects:site --reason \"a: b, c+d @ 5%\"",
    f"{DRIVER} answer --run r1 proposed --set 'age (y)=excluded' --reason \"(y) is in the name\"",
    f"{DRIVER}\task\t--run\tr1",
    # package 1: a preview's reason, an answer's marker, a want and the account
    f"{DRIVER} preview --run r1 set_selection:elastic_net --reason 'does a penalty keep fewer?'",
    f"{DRIVER} answer --run r1 impute --reason 'a fill keeps the rows' --marker complete_case_share "
    f"--rejected complete_case",
    f"{DRIVER} want --run r1 'drift correction by run order' --kind option --reason 'the QC "
    f"samples drift' --marker run_order",
    f"{DRIVER} account --run r1 \"I kept the linear model: the penalties scored alike.\"",
]

FORBIDDEN = {
    "chained with ;": f"{DRIVER} ask --run r1; cat ~/.ssh/id_rsa",
    "chained with &&": f"{DRIVER} ask --run r1 && rm -rf research",
    "or-chained": f"{DRIVER} ask --run r1 || ls",
    "backgrounded": f"{DRIVER} fit --run r1 &",
    "piped": f"{DRIVER} ask --run r1 | head -5",
    "backticks": f"{DRIVER} answer --run r1 `cat notes/replication/bang2009.md` --reason x",
    "command substitution": f"{DRIVER} answer --run r1 $(ls) --reason x",
    "a variable": f"{DRIVER} answer --run r1 impute --reason $HOME",
    "redirected out": f"{DRIVER} export --run r1 > /tmp/bundle.json",
    "redirected in": f"{DRIVER} ask --run r1 < /etc/passwd",
    "a newline": f"{DRIVER} ask --run r1\ncat /etc/passwd",
    "a carriage return": f"{DRIVER} ask --run r1\rcat /etc/passwd",
    "another program": "cat turbotab/sample_data/clinical_risk.csv",
    "a listing": "ls research/runs",
    "a change of folder": "cd /tmp",
    "another module": "scripts/py -m turbotab.server --port 8787",
    "python code": "scripts/py -c 'import os; print(os.listdir())'",
    "a script": "scripts/py research/swarm/driver.py ask --run r1",
    "an environment prefix": f"TURBOTAB_SWARM_RUNS=/tmp {DRIVER} ask --run r1",
    "the orchestrator's start": f"{DRIVER} start --data /etc/passwd --budget-fits 9 --budget-seconds 9",
    "the orchestrator's stop": f"{DRIVER} stop --run r1",
    "no command": f"{DRIVER}",
    "a made-up command": f"{DRIVER} reseal --run r1",
    "an unclosed quote": f"{DRIVER} answer --run r1 impute --reason 'because",
    "empty": "   ",
    # zsh's own: each runs a program or expands to the file names here, outside quotes
    "zsh process substitution": f"{DRIVER} ask --run r1 =(curl -s https://example.org)",
    "zsh process substitution in a reason": f"{DRIVER} answer --run r1 a --reason ok =(cat ~/.ssh/id_rsa)",
    "a zsh glob qualifier that runs code": f"{DRIVER} preview --run r1 *(e:'touch pwned':)",
    "a zsh glob qualifier that calls a function": f"{DRIVER} ask --run r1 *(+rm)",
    "a bare glob": f"{DRIVER} ask --run *",
    "a recursive glob": f"{DRIVER} answer --run r1 **/* --reason x",
    "a one-character glob": f"{DRIVER} ask --run r?",
    "a bracket glob": f"{DRIVER} ask --run r[12]",
    "a tilde": f"{DRIVER} ask --run ~",
    "a tilde inside a word": f"{DRIVER} answer --run r1 impute --set x=~/notes --reason x",
    "braces": f"{DRIVER} ask --run {{r1,r2}}",
    "history": f"{DRIVER} ask --run r1 !!",
    "a comment": f"{DRIVER} ask --run r1 #x",
    "an extended-glob caret": f"{DRIVER} ask --run ^r1",
    "a subshell": f"{DRIVER} ask --run r1 (ls)",
    "an equals expansion": f"{DRIVER} ask --run =ls",
    "a doubled equals": f"{DRIVER} answer --run r1 a --set x==ls --reason x",
    "an equals after quotes": f"{DRIVER} answer --run r1 a \"\"=ls --reason x",
    "a backslash outside quotes": f"{DRIVER} ask --run r1\\ x",
    "a backslash in double quotes": f"{DRIVER} answer --run r1 a --reason \"a\\\"b\"",
    "history in double quotes": f"{DRIVER} answer --run r1 a --reason \"wow!!\"",
    "a vertical tab": f"{DRIVER} ask --run r1\x0bx",
    "a non-ASCII letter outside quotes": f"{DRIVER} ask --run ré",
}

# Run under zsh with no hook in the way, each of these runs its payload: the forms are real.
PAYLOADS = {
    "process substitution": "{driver} ask --run r1 =(touch {marker})",
    "a glob qualifier": "{driver} preview --run r1 *(e:'touch {marker}':)",
    "command substitution": "{driver} ask --run r1 $(touch {marker})",
    "backticks": "{driver} ask --run r1 `touch {marker}`",
    "chained": "{driver} ask --run r1; touch {marker}",
}


@pytest.mark.parametrize("command", ALLOWED)
def test_the_drivers_own_commands_pass_for_an_analyst(command):
    """Silent where it should be: each analyst command, in each interpreter form, from the root."""
    assert hook.decide(call(command)) is None


@pytest.mark.parametrize("name", sorted(FORBIDDEN))
def test_every_other_command_is_denied_to_an_analyst(name):
    """Fires where it should: each forbidden form of the brief, and every other program."""
    assert denied(hook.decide(call(FORBIDDEN[name]))), name


@pytest.mark.parametrize("cwd", [Path("/tmp"), REPO / "research", REPO.parent, None])
def test_the_driver_runs_only_from_the_repo_root(cwd):
    assert denied(hook.decide(call(f"{DRIVER} ask --run r1", cwd=cwd)))
    assert hook.decide(call(f"{DRIVER} ask --run r1", cwd=REPO)) is None


@pytest.mark.parametrize("tool", ["Edit", "Write", "NotebookEdit", "MultiEdit", "Read", "Glob",
                                  "Grep", "WebFetch", "WebSearch", "Agent", "Task",
                                  "mcp__claude_ai_Gmail__send_message",
                                  "mcp__claude-in-chrome__navigate"])
def test_every_tool_but_bash_is_denied_to_an_analyst(tool):
    answer = hook.decide(call(tool=tool, tool_input={"file_path": str(REPO / "x"), "url": "https://x"}))
    assert denied(answer)
    assert tool in answer["hookSpecificOutput"]["permissionDecisionReason"]


@pytest.mark.parametrize("tool", ["SubagentHandback", "StructuredOutput"])
def test_an_analyst_hands_its_report_back(tool):
    """The live check of 2026-10-22 found the harness's report tools denied, so no analyst's
    report reached the orchestrator: an analyst may call them, whatever their input."""
    assert hook.decide(call(tool=tool, tool_input={"report": "done", "outcome": "exported"})) is None
    assert hook.decide(call(tool=tool, tool_input={})) is None


@pytest.mark.parametrize("tool", ["SubagentHandbackX", "subagenthandback", "Structured_Output",
                                  "StructuredOutputs", "SubagentHandback ", ""])
def test_a_tool_named_near_the_report_tools_is_still_denied(tool):
    assert denied(hook.decide(call(tool=tool, tool_input={})))


@pytest.mark.parametrize("agent", [None, "builder", "verifier", "integrator", "auditor",
                                   "general-purpose", "analysts", "Analyst", "analyst-2"])
def test_it_says_nothing_for_any_other_caller(agent):
    """Silent for every other agent type and for the session itself (no ``agent_type``), on the
    very calls it denies an analyst."""
    for command in [*FORBIDDEN.values(), *ALLOWED]:
        assert hook.decide(call(command, agent=agent)) is None
    for tool in ("Edit", "Write", "WebFetch", "mcp__x__y"):
        assert hook.decide(call(tool=tool, agent=agent, tool_input={})) is None
    assert hook.decide(call(f"{DRIVER} ask --run r1", agent=agent, cwd=Path("/tmp"))) is None


def test_the_hook_answers_on_stdout_as_claude_code_reads_it():
    """main(): a denial is one JSON object on standard output with exit 0; a pass prints nothing;
    input that is not JSON is refused with exit 2, since the caller cannot be told."""
    out, err = io.StringIO(), io.StringIO()
    assert hook.main(io.StringIO(json.dumps(call("ls"))), out, err) == 0
    answer = json.loads(out.getvalue())
    assert answer == {"hookSpecificOutput": {
        "hookEventName": "PreToolUse", "permissionDecision": "deny",
        "permissionDecisionReason": answer["hookSpecificOutput"]["permissionDecisionReason"]}}
    out = io.StringIO()
    assert hook.main(io.StringIO(json.dumps(call(f"{DRIVER} ask --run r1"))), out, err) == 0
    assert out.getvalue() == ""
    out = io.StringIO()
    assert hook.main(io.StringIO(json.dumps(call("ls", agent="builder"))), out, err) == 0
    assert out.getvalue() == ""
    assert hook.main(io.StringIO("not json {"), io.StringIO(), err) == 2
    assert err.getvalue().strip()


def test_it_runs_as_a_script_on_the_standard_library_alone():
    """Wired as ``python3 <repo>/research/swarm/analyst_hook.py``: run with no site-packages (-S)
    and isolated from the environment (-I), it still denies and passes."""
    def run(payload):
        return subprocess.run([sys.executable, "-S", "-I", str(SCRIPT)], input=json.dumps(payload),
                              capture_output=True, text=True, timeout=30)

    r = run(call(f"{DRIVER} ask --run r1; ls"))
    assert r.returncode == 0, r.stderr
    assert denied(json.loads(r.stdout))
    r = run(call(f"{DRIVER} ask --run r1"))
    assert r.returncode == 0 and r.stdout == "", (r.stdout, r.stderr)


# ── what a shell makes of a command the hook lets through ────────────────────
#
# Claude Code runs a Bash command with the user's shell, through ``eval`` (its errors read
# ``(eval):1: ...``); on this machine that shell is zsh. Each shell here is run on no startup
# files, once more with the zsh options that widen what a word expands to.

SHELLS = {
    "zsh": ["zsh", "-f", "-c", 'eval "$1"', "zsh"],
    "zsh, extended": ["zsh", "-f", "-c", 'setopt extended_glob magic_equal_subst; eval "$1"', "zsh"],
    "bash": ["bash", "--norc", "--noprofile", "-c", 'eval "$1"', "bash"],
}
FAKE_PY = '#!/bin/sh\nprintf "%s\\0" "$@" > "$(dirname "$0")/../argv.bin"\n'


def scratch(tmp_path: Path) -> Path:
    """A folder whose ``scripts/py`` only writes down the words it was given, beside files a glob
    would match."""
    folder = tmp_path / "shell"
    (folder / "scripts").mkdir(parents=True)
    py = folder / "scripts" / "py"
    py.write_text(FAKE_PY)
    py.chmod(0o755)
    for name in ("r1", "r2", "notes"):
        (folder / name).write_text("x")
    return folder


def files(folder: Path) -> set[str]:
    return {str(p.relative_to(folder)) for p in folder.rglob("*")}


def in_shell(shell: str, command: str, folder: Path) -> list[str] | None:
    """The words ``scripts/py`` was given when ``shell`` ran ``command`` in ``folder``, or None
    when it was not run."""
    (folder / "argv.bin").unlink(missing_ok=True)
    subprocess.run([*SHELLS[shell], command], cwd=folder, capture_output=True, timeout=30,
                   env={"PATH": "/usr/bin:/bin", "HOME": str(folder)})
    try:
        return (folder / "argv.bin").read_bytes().decode().split("\0")[:-1]
    except OSError:
        return None


ON_DISK = [c for c in ALLOWED if c.split()[0] in ("scripts/py", "./scripts/py")]


@pytest.mark.parametrize("shell", [s for s in SHELLS if shutil.which(SHELLS[s][0])])
def test_a_command_the_hook_passes_runs_the_driver_alone_with_its_words_unchanged(tmp_path, shell):
    """Silent where it should be, and the silence is safe: each allowed command, run by the shell,
    hands ``scripts/py`` exactly the words ``shlex`` reads (nothing expanded, nothing run), and
    leaves no file behind but the one the fake program writes."""
    folder = scratch(tmp_path)
    before = files(folder)
    for command in ON_DISK:
        assert hook.decide(call(command)) is None, command
        assert in_shell(shell, command, folder) == shlex.split(command)[1:], command
        assert files(folder) - before == {"argv.bin"}, command


@pytest.mark.skipif(shutil.which("zsh") is None, reason="zsh is not installed here")
@pytest.mark.parametrize("name", sorted(PAYLOADS))
def test_each_zsh_payload_runs_without_the_hook_and_is_denied_with_it(tmp_path, name):
    """The forms are real: run by zsh with nothing in the way, each makes its marker file. And the
    hook denies each one to an analyst."""
    folder = scratch(tmp_path)
    marker = tmp_path / f"ran-{name.replace(' ', '-')}"
    command = PAYLOADS[name].format(driver=DRIVER, marker=marker)
    in_shell("zsh", command, folder)
    assert marker.exists(), command
    assert denied(hook.decide(call(command))), command


@pytest.mark.skipif(shutil.which("zsh") is None, reason="zsh is not installed here")
def test_through_the_hook_nothing_runs_but_the_driver(tmp_path):
    """The hook in the way, as Claude Code puts it: every command, allowed or not, is run by zsh
    only when the hook says nothing; afterwards no payload has run."""
    folder = scratch(tmp_path)
    markers = []
    for name, form in PAYLOADS.items():
        marker = tmp_path / f"through-{name.replace(' ', '-')}"
        markers.append(marker)
        for shell in ("zsh", "zsh, extended"):
            command = form.format(driver=DRIVER, marker=marker)
            if hook.decide(call(command)) is None:
                in_shell(shell, command, folder)
    for command in ON_DISK:
        if hook.decide(call(command)) is None:
            in_shell("zsh, extended", command, folder)
    assert [m.name for m in markers if m.exists()] == []


def test_its_command_list_is_the_drivers():
    from research.swarm import driver

    assert hook.ANALYST_COMMANDS == driver.ANALYST_COMMANDS
    assert set(driver.COMMANDS) - set(hook.ANALYST_COMMANDS) == {"start", "stop"}
