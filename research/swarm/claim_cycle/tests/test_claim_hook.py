"""The claim cycle's PreToolUse hook (``claim_cycle/hook.py``): a theorist or an experimentalist
runs the claim cycle's tools and nothing else, so it reads records only through the view, and never
the sealed file by any other tool. Each rule both ways: what it lets through, and what it denies;
and it says nothing for any other agent."""
from __future__ import annotations

import io
import json

import pytest

from research.swarm.claim_cycle import hook as H
from research.swarm.claim_cycle import tools as T
from research.swarm.claim_cycle.tests.constructed import GOOD

REPO = H.REPO
TOOLS = "scripts/py -m research.swarm.claim_cycle.tools"
PROPOSE = f"{TOOLS} propose --out cycle/proposals.jsonl '{json.dumps(GOOD)}'"


def call(agent: str | None, tool: str = "Bash", command: str | None = None,
         cwd: str | None = str(REPO), **tool_input) -> dict:
    out = {"tool_name": tool, "tool_input": {**({"command": command} if command else {}),
                                             **tool_input}, "cwd": cwd}
    if agent is not None:
        out["agent_type"] = agent
    return out


def denied(answer: dict | None) -> str:
    assert answer is not None
    out = answer["hookSpecificOutput"]
    assert out["permissionDecision"] == "deny"
    return out["permissionDecisionReason"]


ALLOWED = {
    "theorist": [f"{TOOLS} index cycle/a", f"{TOOLS} records cycle/a --run run_01-ab",
                 f"{TOOLS} records cycle/a --fields regime.markers,decisions.kind",
                 f"{TOOLS} proposals cycle/proposals.jsonl", PROPOSE],
    "experimentalist": [f"{TOOLS} index cycle/a", f"{TOOLS} records cycle/a --dataset dataset_2",
                        f"{TOOLS} proposals cycle/proposals.jsonl"],
}


@pytest.mark.parametrize("agent, command", [(a, c) for a, cs in ALLOWED.items() for c in cs])
def test_the_tools_commands_of_its_role_pass(agent, command):
    assert H.decide(call(agent, command=command)) is None


@pytest.mark.parametrize("agent", list(ALLOWED))
@pytest.mark.parametrize("tool", ["Read", "Write", "Edit", "Glob", "Grep", "WebFetch", "Agent",
                                  "NotebookEdit", "mcp__claude_ai_Claude_Docs__read"])
def test_every_other_tool_is_denied(agent, tool):
    assert tool in denied(H.decide(call(agent, tool, file_path="/tmp/sealed/s.json")))


@pytest.mark.parametrize("agent", list(ALLOWED))
@pytest.mark.parametrize("tool", H.REPORT_TOOLS)
def test_its_report_tools_pass(agent, tool):
    assert H.decide(call(agent, tool)) is None


DENIED = {
    "reading a file": "cat /tmp/sealed/s.json",
    "the driver": "scripts/py -m research.swarm.driver ask --run run_01-ab",
    "another module": "scripts/py -m research.swarm.shuffle cycle/a --out x --sealed y",
    "python itself": "scripts/py -c 'print(1)'",
    "a chained command": f"{TOOLS} index cycle/a; cat /tmp/s.json",
    "a substitution": f"{TOOLS} records $(cat /tmp/path)",
    "a redirection": f"{TOOLS} index cycle/a > /tmp/out",
    "no command": TOOLS,
    "a command the tools lack": f"{TOOLS} sealed cycle/a",
}


@pytest.mark.parametrize("agent", list(ALLOWED))
@pytest.mark.parametrize("command", list(DENIED.values()), ids=list(DENIED))
def test_anything_but_its_tools_is_denied(agent, command):
    assert denied(H.decide(call(agent, command=command)))


def test_an_experimentalist_proposes_nothing():
    says = denied(H.decide(call("experimentalist", command=PROPOSE)))
    assert "propose" in says and "experimentalist" in says


def test_the_tools_run_from_the_repo_root_only():
    assert denied(H.decide(call("theorist", command=f"{TOOLS} index cycle/a", cwd="/tmp")))


@pytest.mark.parametrize("agent", [None, "analyst", "builder", "general-purpose"])
def test_it_says_nothing_for_any_other_caller(agent):
    assert H.decide(call(agent, "Read", file_path="/tmp/s.json")) is None
    assert H.decide(call(agent, command="cat /tmp/s.json")) is None


def test_the_hooks_commands_are_the_tools_commands():
    assert set().union(*H.COMMANDS.values()) == set(T.COMMANDS)
    assert set(H.COMMANDS["experimentalist"]) < set(H.COMMANDS["theorist"])


def test_input_that_is_not_a_hook_call_is_refused():
    err = io.StringIO()
    assert H.main(io.StringIO("not json"), io.StringIO(), err) == 2 and err.getvalue()
    out = io.StringIO()
    assert H.main(io.StringIO(json.dumps(call("theorist", "Read"))), out, io.StringIO()) == 0
    assert json.loads(out.getvalue())["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_it_runs_as_a_script_on_the_standard_library_alone():
    """Wired as ``python3 <repo>/research/swarm/claim_cycle/hook.py``: run with no site-packages
    (-S) and isolated from the environment (-I), so the repo is not on its path, it still loads the
    analyst hook's rules, denies and passes."""
    import subprocess
    import sys

    def run(payload):
        return subprocess.run([sys.executable, "-S", "-I", str(H.HERE / "hook.py")],
                              input=json.dumps(payload), capture_output=True, text=True,
                              timeout=30)

    r = run(call("theorist", "Read", file_path="/tmp/s.json"))
    assert r.returncode == 0, r.stderr
    assert "Read" in denied(json.loads(r.stdout))
    r = run(call("theorist", command=f"{TOOLS} index cycle/a; cat /tmp/s.json"))
    assert r.returncode == 0 and denied(json.loads(r.stdout))
    r = run(call("theorist", command=f"{TOOLS} index cycle/a"))
    assert r.returncode == 0 and r.stdout == "", (r.stdout, r.stderr)


def test_a_tool_input_that_is_not_an_object_is_denied_for_its_roles():
    answer = H.decide({"tool_name": "Bash", "tool_input": "cat x", "cwd": str(REPO),
                       "agent_type": "theorist"})
    assert denied(answer)
