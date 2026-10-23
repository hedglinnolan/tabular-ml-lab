"""The claim cycle's PreToolUse hook: a theorist or an experimentalist runs the claim cycle's tools
and nothing else, so it reads a record only through the view (``view.py``) and never the sealed
file by any other tool.

Claude Code runs this script before each tool call, with the call as JSON on standard input
(``tool_name``, ``tool_input``, ``cwd``, and ``agent_type`` when a subagent makes the call). For a
``theorist`` or an ``experimentalist`` it denies, with a plain reason the agent reads:

* every tool but Bash and the harness's report tools (``analyst_hook.REPORT_TOOLS``), by which it
  hands its report back;
* every Bash command that is not exactly the tools' invocation: ``python``, ``python3`` or
  ``scripts/py``, then ``-m research.swarm.claim_cycle.tools`` and one of its role's commands
  (:data:`COMMANDS`: the experimentalist proposes nothing), from the repo root, that no shell
  expands or runs anything in, under the analyst hook's rules (``analyst_hook.shell_refusal``:
  none of ``; & | $ < >``, a backtick or a line break anywhere, quoted or not; outside quotes only
  plain characters; so a proposal's JSON goes in single quotes).

For every other caller it says nothing: the call goes on to the permissions. An allowed command is
not approved here either. Input that does not read as a hook call is refused (exit 2).

Wired only in ``.claude/settings.local.json`` on the owner's machine, beside the analyst's hook,
never committed:

    {"hooks": {"PreToolUse": [{"matcher": "*", "hooks": [{"type": "command",
        "command": "python3 <repo>/research/swarm/claim_cycle/hook.py"}]}]}}

It uses the standard library and the analyst hook, which uses only the standard library, so any
Python 3 runs it.
"""
from __future__ import annotations

import importlib.util
import json
import shlex
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def _analyst_hook() -> Any:
    try:
        from research.swarm import analyst_hook
    except ImportError:  # run as a script, with the repo not on the path
        spec = importlib.util.spec_from_file_location("analyst_hook",
                                                      HERE.parent / "analyst_hook.py")
        analyst_hook = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(analyst_hook)
    return analyst_hook


A = _analyst_hook()
THEORIST = "theorist"
EXPERIMENTALIST = "experimentalist"
MODULE = "research.swarm.claim_cycle.tools"
# Each role's commands (``tools.COMMANDS``; a test holds the two lists equal).
COMMANDS = {THEORIST: ("index", "records", "proposals", "propose"),
            EXPERIMENTALIST: ("index", "records", "proposals")}
REPORT_TOOLS = A.REPORT_TOOLS
FORBIDDEN = A.FORBIDDEN
deny = A.deny


def form(role: str) -> str:
    return (f"scripts/py -m {MODULE} <command> ..., run from the repo root, where <command> is one "
            f"of {', '.join(COMMANDS[role])}")


def bash_refusal(command: Any, cwd: Any, role: str) -> str | None:
    """Why ``role``'s Bash ``command`` is not the tools' invocation, or None when it is."""
    if not isinstance(command, str) or not command.strip():
        return f"A {role} runs only the claim cycle's tools: {form(role)}."
    why = A.shell_refusal(command)
    if why:
        return why.replace(A.FORM, form(role))
    try:
        words = shlex.split(command)
    except ValueError:
        return f"The command does not read as one tools command (an unclosed quote?): {form(role)}."
    if len(words) < 4 or words[0] not in A.INTERPRETERS or words[1] != "-m" or words[2] != MODULE:
        return f"A {role} runs only the claim cycle's tools: {form(role)}."
    if words[3] not in COMMANDS[role]:
        return (f"{words[3]!r} is not a command a {role} runs; its commands are "
                f"{', '.join(COMMANDS[role])}.")
    try:
        here = Path(str(cwd)).resolve() if cwd else None
    except (OSError, RuntimeError):
        here = None
    if here != REPO:
        return f"The tools run from the repo root, {REPO}, not from {cwd or 'an unknown folder'}."
    return None


def decide(call: Any) -> dict[str, Any] | None:
    """The hook's answer to one tool call: a denial, or None to say nothing."""
    if not isinstance(call, dict) or call.get("agent_type") not in COMMANDS:
        return None
    role = call["agent_type"]
    tool = call.get("tool_name")
    if tool in REPORT_TOOLS:
        return None
    if tool != "Bash":
        return deny(f"A {role} works only through the claim cycle's tools, run with Bash; {tool} "
                    f"is not open to it.")
    given = call.get("tool_input")
    command = given.get("command") if isinstance(given, dict) else None
    why = bash_refusal(command, call.get("cwd"), role)
    return deny(why) if why else None


def main(stdin: Any = None, stdout: Any = None, stderr: Any = None) -> int:
    stdin, stdout, stderr = stdin or sys.stdin, stdout or sys.stdout, stderr or sys.stderr
    try:
        call = json.loads(stdin.read() or "")
    except ValueError:
        stderr.write("The claim cycle's hook could not read the tool call, so it is refused.\n")
        return 2
    answer = decide(call)
    if answer is not None:
        stdout.write(json.dumps(answer) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
