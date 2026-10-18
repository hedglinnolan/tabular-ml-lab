"""The analyst's PreToolUse hook (DRIVER_PLAN D2), tested both ways: for an ``analyst`` it denies
every tool but Bash and every Bash command that is not exactly the driver's invocation, and lets
the driver's own commands through; for every other caller it says nothing, whatever the call.

The forbidden forms are the brief's (no ``;``, ``&&``, ``|``, backticks, ``$( )``, redirections or
newlines, nothing but ``python -m research.swarm.driver ...`` from the repo root), written out here
as a person would type them, never read from the hook."""
from __future__ import annotations

import io
import json
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


def test_its_command_list_is_the_drivers():
    from research.swarm import driver

    assert hook.ANALYST_COMMANDS == driver.ANALYST_COMMANDS
    assert set(driver.COMMANDS) - set(hook.ANALYST_COMMANDS) == {"start", "stop"}
